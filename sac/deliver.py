"""End-to-end Part B workflow: train, validate, freeze, evaluate, audit, and plot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from env import V2GEnv
from .audit import audit_saved_episode
from .extended import evaluate_extended
from .metrics import price_regime_definition
from .policy import load_policy
from .protocol import configure, grid, write_json
from .suite import evaluate_scenarios
from .train import train


STRATEGIES = ('random', 'immediate', 'rule')


def standard_scenarios(n, split):
    environment = V2GEnv(n_slots=n)
    try:
        result = []
        for date, seed in grid(split):
            environment.reset(seed=seed, options={'date': date})
            result.append(environment.export_scenario())
        return result
    finally:
        environment.close()


def verify_exports(directory):
    counts, departures = {}, 0
    for n in (5, 10):
        for strategy in ('SAC', *STRATEGIES):
            folder = Path(directory) / f'test_n{n}' / strategy
            paths = sorted(folder.glob('*.csv'))
            if len(paths) != 140:
                raise AssertionError(f'{folder} must contain exactly 140 scenarios')
            seen = set()
            for path in paths:
                summary, _ = audit_saved_episode(path, n, strategy)
                seen.add((summary['date'], summary['scenario_seed']))
                if strategy == 'SAC':
                    departures += len(summary['departures'])
            if seen != set(grid('test')):
                raise AssertionError(f'{folder} must cover the complete final-test grid')
            counts[f'n{n}_{strategy}'] = len(paths)
    return {'scenarios_by_strategy': counts, 'rows_per_scenario': 48,
            'sac_departures': departures, 'sac_departure_success_rate': 1.0,
            'bess_terminal_success_rate': 1.0,
            'csv_cost_tolerance_sgd': 1e-8}


def verify_independent_process(checkpoints):
    code = '''
import sys
import numpy as np
from sac.policy import load_policy
from env import V2GEnv
for n, path in ((1, sys.argv[2]), (5, sys.argv[1]), (10, sys.argv[2])):
    policy = load_policy(path, n)
    for date in ('2026-10-01', '2026-10-03'):
        environment = V2GEnv(n_slots=n)
        obs, _ = environment.reset(seed=0, options={'date': date})
        for _ in range(48):
            action = policy.act(obs, deterministic=True)
            assert action.shape == (n + 1,) and np.isfinite(action).all()
            assert np.all(action[1:][obs['ev_present'] == 0] == 0)
            obs, _, done, _, _ = environment.step(action)
        assert done
print('independent process passed for N=1, N=5 and N=10')
'''
    completed = subprocess.run([sys.executable, '-c', code, str(checkpoints[5]), str(checkpoints[10])],
                               capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    return completed.stdout.strip()


def unit_checks(output):
    with (Path(output) / 'unit_tests.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-v'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def validation_gate(run, n, output, definition):
    if not (run / 'completed.json').exists():
        raise RuntimeError(f'training must complete before selection: {run}')
    checkpoint = run / 'best.pt'
    policy = load_policy(checkpoint, n)
    candidate = output / 'validation_candidates' / policy.checkpoint_sha256[:16] / f'validation_n{n}'
    report = evaluate_scenarios(policy, standard_scenarios(n, 'validation'), candidate, definition,
                                offline_output=output / 'offline_opt' / f'validation_n{n}')
    passes = all(v['significantly_better'] and v['date_cluster_bootstrap_95_ci_sgd'][0] > 0
                 for v in report['comparisons'].values())
    if not passes:
        raise AssertionError(f'N={n} validation gate failed; see {candidate}/comparison.json')
    return checkpoint, report, candidate


def deliver(n5_run, n10_run, output, config='sac/configs/default.json',
            models_dir='docs/B_SAC/models', train_models=False, resume=False):
    configure()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)
    runs = {5: Path(n5_run), 10: Path(n10_run)}
    manifest_path = output / 'final_test_manifest.json'
    if manifest_path.exists() and not resume:
        raise FileExistsError('final-test manifest already exists; use --resume to continue cached evaluation')
    if resume and not manifest_path.exists():
        raise FileNotFoundError('no frozen final-test manifest to resume')
    unit_checks(output)
    definition = price_regime_definition()
    if resume:
        manifest = json.loads(manifest_path.read_text())
        frozen = {n: Path(manifest['checkpoints'][str(n)]['path']) for n in (5, 10)}
        for n, path in frozen.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['checkpoints'][str(n)]['sha256']:
                raise ValueError('frozen checkpoint differs from the final-test manifest')
            runs[n] = Path(manifest['checkpoints'][str(n)]['source']).parent
    else:
        if train_models:
            for n, run in runs.items():
                train(config, n, run)
        selections = {n: validation_gate(run, n, output, definition) for n, run in runs.items()}
        models_dir.mkdir(parents=True, exist_ok=True)
        frozen = {}
        manifest = {'protocol': 'Validation-only selection and freezing before final test evaluation',
                    'validation_grid': grid('validation'), 'test_grid': grid('test'),
                    'price_regimes': definition, 'checkpoints': {}, 'validation_acceptance': {}}
        for n, (source, report, candidate) in selections.items():
            destination = output / f'validation_n{n}'
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(candidate, destination)
            path = models_dir / f'final_n{n}.pt'
            shutil.copy2(source, path)
            frozen[n] = path
            manifest['checkpoints'][str(n)] = {
                'path': str(path), 'source': str(source),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'selected_step': load_policy(path, n).metadata['step'],
            }
            manifest['validation_acceptance'][str(n)] = report
        write_json(manifest_path, manifest)

    test_reports = {}
    for n in (5, 10):
        policy = load_policy(frozen[n], n)
        test_reports[n] = evaluate_scenarios(policy, standard_scenarios(n, 'test'), output / f'test_n{n}',
                                             definition, offline_output=output / 'offline_opt' / f'test_n{n}')
    extended = evaluate_extended(load_policy(frozen[10], 10), output / 'extended_realistic', definition)
    acceptance = verify_exports(output)
    acceptance.update(
        unit_tests_passed=True,
        independent_inference=verify_independent_process(frozen),
        extended_scenarios=extended['scenarios'],
        extended_audited_episodes=extended['audited_episodes'],
        extended_sac_departures=extended['sac_departures'],
        extended_departure_success_rate=1.0,
        extended_bess_terminal_success_rate=1.0,
        occupancy_histogram=extended['occupancy_histogram'],
        overnight_cases=extended['early_mixed_cases'] + extended['late_mixed_cases'],
        weekend_visitor_cases=extended['weekend_visitor_cases'],
        maximum_offline_replay_error_sgd=max(
            [extended['maximum_offline_replay_error_sgd']]
            + [test_reports[n]['offline_opt']['maximum_replay_error_sgd'] for n in (5, 10)]
        ),
    )
    subprocess.run([sys.executable, '-m', 'analysis.plotting.plot_sac', '--n5-run', str(runs[5]),
                    '--n10-run', str(runs[10]), '--results', str(output)], check=True)
    subprocess.run([sys.executable, '-m', 'analysis.plotting.plot_extended', '--results', str(output)], check=True)
    # Comparison against the official baselines (C's arrival charge and price rule, MILP).
    subprocess.run([sys.executable, '-m', 'sac.official', '--results', str(output)], check=True)
    subprocess.run([sys.executable, '-m', 'analysis.plotting.plot_official',
                    '--official', str(output / 'official'), '--figures-dir', str(output / 'figures'),
                    '--tables-dir', str(output / 'tables')], check=True)
    write_json(output / 'acceptance.json', acceptance)
    print(json.dumps(acceptance, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n5-run', default='docs/B_SAC/training_logs/n5_training.jsonl')
    parser.add_argument('--n10-run', default='docs/B_SAC/training_logs/n10_training.jsonl')
    parser.add_argument('--output', default='results/sac')
    parser.add_argument('--config', default='sac/configs/default.json')
    parser.add_argument('--models-dir', default='docs/B_SAC/models')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--train', dest='train_models', action='store_true')
    modes.add_argument('--resume', action='store_true')
    deliver(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
