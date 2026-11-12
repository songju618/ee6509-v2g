import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from sac.agent import SAC
from sac.deliver import deliver
from sac.metrics import price_regime_definition
from sac.normalization import Normalizer
from sac.policy import load_policy
from sac.protocol import configure
from sac.scenarios import extended_scenario
from sac.suite import evaluate_scenarios
from sac.train import training_observations


class SuiteProtocolTests(unittest.TestCase):
    def setUp(self):
        configure()

    def test_saved_trajectories_resume_without_new_policy_or_solver_runs(self):
        config = json.loads(Path('sac/configs/default.json').read_text())
        agent = SAC(config)
        normalizer = Normalizer.fit(training_observations(1, [0], True), version=2)
        checkpoint = {'format_version': 2, 'encoder': agent.encoder.state_dict(), 'actor': agent.actor.state_dict(),
                      'normalizer': normalizer.statistics, 'metadata': {'config': config}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            torch.save(checkpoint, folder / 'policy.pt')
            policy = load_policy(folder / 'policy.pt', 1)
            scenarios = [extended_scenario('2026-10-01', 0, n_slots=1).to_dict()]
            output = folder / 'evaluation'
            first = evaluate_scenarios(policy, scenarios, output, price_regime_definition())
            with patch('sac.suite.rollout', side_effect=AssertionError('trajectory repeated')), \
                    patch('sac.suite.solve_offline', side_effect=AssertionError('oracle repeated')):
                second = evaluate_scenarios(policy, scenarios, output, price_regime_definition())
            self.assertEqual(first, second)
            policy.checkpoint_sha256 = 'different policy'
            with self.assertRaises(FileExistsError):
                evaluate_scenarios(policy, scenarios, output, price_regime_definition())

    def test_persisted_accounting_error_is_detected(self):
        from env import V2GEnv
        from sac.audit import audit_saved_episode
        from sac.evaluate import export_trajectory, rollout
        with tempfile.TemporaryDirectory() as temporary:
            infos, summary, scenario, _ = rollout(V2GEnv(n_slots=1), '2026-10-01', 0, reference='immediate')
            export_trajectory(temporary, 'immediate', 1, infos, summary, scenario)
            path = next(Path(temporary).glob('*.csv'))
            audit_saved_episode(path, 1, 'immediate')
            with path.open(newline='') as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames
                rows = list(reader)
            rows[0]['step_cost_sgd'] = float(rows[0]['step_cost_sgd']) + .001
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(AssertionError):
                audit_saved_episode(path, 1, 'immediate')

    def test_failed_validation_cannot_freeze_or_reach_test_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch('sac.deliver.unit_checks'), \
                    patch('sac.deliver.validation_gate', side_effect=AssertionError('validation failed')), \
                    patch('sac.deliver.evaluate_scenarios') as evaluation, \
                    patch('sac.deliver.shutil.copy2') as freeze:
                with self.assertRaises(AssertionError):
                    deliver('unused_n5', 'unused_n10', output, 'unused_config')
                evaluation.assert_not_called()
                freeze.assert_not_called()
                self.assertFalse((output / 'final_test_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
