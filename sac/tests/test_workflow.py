"""Public CLI, export and independent-process inference regression coverage."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from env import V2GEnv
from env.recording import csv_columns
from sac.deliver import deliver, verify_independent_process
from sac.evaluate import evaluate_grid
from sac.networks import Actor, Encoder
from sac.policy import load_policy
from sac.protocol import configure, grid


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        configure()

    def checkpoint(self, path):
        statistics = {key: {'mean': [0.] * size, 'std': [1.] * size}
                      for key, size in [('timeseries', 3), ('global', 4), ('ev', 4)]}
        torch.save({'format_version': 1, 'encoder': Encoder().state_dict(),
                    'actor': Actor().state_dict(), 'normalizer': statistics}, path)

    def test_public_policy_in_independent_process_for_both_fleet_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'policy.pt'
            self.checkpoint(checkpoint)
            code = '''
import sys
import numpy as np
from env import V2GEnv
from sac.policy import load_policy
for n in (5, 10):
    policy = load_policy(sys.argv[1], n)
    for date in ('2026-10-01', '2026-10-03'):
        obs, _ = V2GEnv(n_slots=n).reset(seed=0, options={'date': date})
        action = policy.act(obs, deterministic=True)
        assert action.shape == (n+1,) and np.isfinite(action).all()
        assert (action[1:][obs['ev_present'] == 0] == 0).all()
'''
            subprocess.run([sys.executable, '-c', code, str(checkpoint)], check=True, capture_output=True)

    def test_delivery_uses_the_real_independent_process_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'policy.pt'
            self.checkpoint(checkpoint)
            result = verify_independent_process({5: checkpoint, 10: checkpoint})
            self.assertIn('N=5 and N=10', result)

    def test_resume_preserves_final_test_results_without_reselecting(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / 'frozen.pt'
            self.checkpoint(checkpoint)
            entry = {'path': str(checkpoint), 'source': str(folder / 'training' / 'best.pt'),
                     'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            (folder / 'final_test_manifest.json').write_text(
                json.dumps({'checkpoints': {'5': entry, '10': entry}}))
            fake_report = {'offline_opt': {'maximum_replay_error_sgd': 0.0}}
            fake_ext = {'scenarios': 980, 'audited_episodes': 4900, 'sac_departures': 4912,
                        'occupancy_histogram': [1] * 11, 'early_mixed_cases': 1, 'late_mixed_cases': 1,
                        'weekend_visitor_cases': 1, 'maximum_offline_replay_error_sgd': 0.0}
            with patch('sac.deliver.unit_checks'), \
                    patch('sac.deliver.validation_gate', side_effect=AssertionError('reselected')) as gate, \
                    patch('sac.deliver.evaluate_scenarios', return_value=fake_report), \
                    patch('sac.deliver.evaluate_extended', return_value=fake_ext), \
                    patch('sac.deliver.verify_exports', return_value={}), \
                    patch('sac.deliver.verify_independent_process', return_value='ok'), \
                    patch('sac.deliver.subprocess.run'):
                deliver('unused', 'unused', folder, resume=True)
                gate.assert_not_called()

    def test_standard_exports_reconcile_and_preserve_empty_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'policy.pt'
            self.checkpoint(checkpoint)
            for n in (5, 10):
                output = Path(temporary) / f'n{n}'
                summaries = evaluate_grid(load_policy(checkpoint, n), n,
                                          [('2026-10-01', 0), ('2026-10-03', 0)], output, attention=True)
                for path in output.glob('*.csv'):
                    with path.open(newline='') as handle:
                        reader = csv.DictReader(handle)
                        self.assertEqual(reader.fieldnames, csv_columns(n))
                        rows = list(reader)
                    summary = json.loads(path.with_suffix('.summary.json').read_text())
                    self.assertEqual(len(rows), 48)
                    self.assertAlmostEqual(sum(float(row['step_cost_sgd']) for row in rows),
                                           summary['variable_cost_sgd'], places=8)
                self.assertEqual(len(summaries), 2)
                self.assertTrue(all(record['success'] for summary in summaries for record in summary['departures']))

    def test_actual_training_cli_and_checkpoint(self):
        for config_path in ('sac/configs/default.json', 'sac/configs/ablation_no_features.json'):
            with tempfile.TemporaryDirectory() as temporary:
                config = json.loads(Path(config_path).read_text())
                config.update(steps=4, warmup_steps=1, batch_size=2, replay_capacity=10,
                              update_every=1, log_every=4, validation_every=4, normalization_seeds=[0])
                path = Path(temporary) / 'config.json'
                path.write_text(json.dumps(config))
                output = Path(temporary) / 'training'
                completed = subprocess.run([sys.executable, '-m', 'sac.train', '--config', str(path),
                                            '--n-slots', '5', '--output-dir', str(output)],
                                           capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertTrue((output / 'completed.json').exists())
                policy = load_policy(output / 'best.pt', 5)
                obs, _ = V2GEnv(n_slots=5).reset(seed=0, options={'date': '2026-10-01'})
                self.assertTrue(np.isfinite(policy.act(obs)).all())
                self.assertEqual(policy.metadata['normalization_split'], 'train')
                self.assertEqual(policy.metadata['training_seed_range'], [0, 99])

    def test_fixed_grid_separation(self):
        train, validation, test = grid('train', range(100)), grid('validation'), grid('test')
        self.assertEqual(len(train), 1400)
        self.assertEqual(len(validation), 35)
        self.assertEqual(len(test), 140)
        self.assertTrue({d for d, _ in train}.isdisjoint({d for d, _ in validation + test}))
        self.assertTrue({s for _, s in train}.isdisjoint({s for _, s in test}))


if __name__ == '__main__':
    unittest.main()
