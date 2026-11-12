"""Evaluate explicit scenario grids and export the environment's standard CSV."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from env import V2GEnv
from env.recording import csv_columns, occupancy_mask, write_csv
from .policy import load_policy
from .protocol import configure, grid, write_json
from .references import Reference


COST_KEYS = ('purchase_sgd', 'revenue_sgd', 'peak_increment_sgd', 'bess_wear_sgd', 'ev_wear_sgd')


def rollout(environment, date, seed, policy=None, reference='random', collect_attention=False, scenario=None):
    options = {'date': date} if scenario is None else {'scenario': scenario}
    obs, _ = environment.reset(seed=seed, options=options)
    scenario = environment.export_scenario()
    controller = Reference(reference, seed=6509 + seed)
    infos, attention_rows = [], []
    for _ in range(48):
        if policy is None:
            action = controller.act(obs, environment)
        else:
            action = policy.act(obs)
            if collect_attention:
                present = obs['ev_present'].astype(bool)
                weights = policy.attention(obs)
                received = weights[present].mean(axis=0) if present.any() else np.zeros(environment.n_slots)
                for slot in np.flatnonzero(present):
                    attention_rows.append({'t': len(infos), 'slot': int(slot),
                                           'remaining_steps': float(obs['ev'][slot, 2]),
                                           'received_attention': float(received[slot]),
                                           'present_count': int(present.sum())})
        obs, _, done, truncated, info = environment.step(action)
        infos.append(info)
        if done or truncated:
            break
    summary = summarize_infos(infos)
    validate_trajectory(infos, summary, scenario)
    return infos, summary, scenario, attention_rows


def summarize_infos(infos):
    summary = dict(infos[-1]['episode_summary'])
    for key in COST_KEYS:
        summary[key] = math.fsum(info[key] for info in infos)
    summary['bess_throughput_kwh'] = math.fsum(abs(info['p_bess_kw']) * .5 for info in infos)
    summary['ev_throughput_kwh'] = math.fsum(float(np.abs(info['p_ev_kw']).sum()) * .5 for info in infos)
    return summary


def validate_trajectory(infos, summary, scenario, horizon=48):
    if len(infos) != horizon or [info['t'] for info in infos] != list(range(horizon)):
        raise AssertionError(f'every scenario must contain {horizon} ordered steps')
    if not math.isclose(math.fsum(info['step_cost_sgd'] for info in infos),
                        summary['variable_cost_sgd'], abs_tol=1e-8, rel_tol=0):
        raise AssertionError('daily and step costs disagree')
    if not all(record['success'] for record in summary['departures']):
        raise AssertionError('departure constraint failed')
    if summary['bess_final_kwh'] < scenario['bess_initial_kwh'] - 1e-9:
        raise AssertionError('BESS terminal constraint failed')
    mask = occupancy_mask(scenario)
    for t, info in enumerate(infos):
        np.testing.assert_array_equal(mask[t], info['ev_present'].astype(bool))
        if np.any(info['p_ev_kw'][~mask[t]]) or np.any(info['e_ev_kwh'][~mask[t]]):
            raise AssertionError('empty-slot energy or power must be zero')


def export_trajectory(output_dir, strategy, n_slots, infos, summary, scenario, weights=None):
    stem = Path(output_dir) / f"{strategy}_{summary['date']}_seed{summary['scenario_seed']}_n{n_slots}"
    write_csv(stem.with_suffix('.csv'), infos, strategy=strategy)
    with stem.with_suffix('.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != csv_columns(n_slots) or len(list(reader)) != 48:
            raise AssertionError('exported CSV schema or row count mismatch')
    write_json(stem.with_suffix('.scenario.json'), scenario)
    if weights is not None:
        write_json(stem.with_suffix('.attention.json'), weights)
    write_json(stem.with_suffix('.summary.json'), summary)


def evaluate_grid(policy, n_slots, pairs, output_dir=None, strategy='SAC', attention=False):
    environment = V2GEnv(n_slots=n_slots)
    summaries = []
    try:
        for date, seed in pairs:
            infos, summary, scenario, weights = rollout(
                environment, date, seed, policy=policy, reference=strategy if policy is None else 'random',
                collect_attention=attention)
            summary.update(strategy=strategy, n_slots=n_slots)
            summaries.append(summary)
            if output_dir is not None:
                export_trajectory(output_dir, strategy, n_slots, infos, summary, scenario,
                                  weights if attention else None)
    finally:
        environment.close()
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n-slots', type=int, choices=range(1, 11), required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    parser.add_argument('--dates', nargs='+', help='Explicit ISO dates overriding split dates')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--output-dir', default='results/sac/evaluation')
    args = parser.parse_args()
    configure()
    pairs = grid(args.split, args.seeds)
    if args.dates:
        seeds = sorted({seed for _, seed in pairs})
        pairs = [(date, seed) for date in args.dates for seed in seeds]
    output = Path(args.output_dir)
    if (output / 'summaries.json').exists():
        raise FileExistsError(f'evaluation already exists at {output}')
    policy = load_policy(args.checkpoint, args.n_slots)
    summaries = evaluate_grid(policy, args.n_slots, pairs, output, attention=True)
    write_json(output / 'summaries.json', summaries)
    print(f'{len(summaries)} scenarios, mean variable cost '
          f'{np.mean([s["variable_cost_sgd"] for s in summaries]):.6f} SGD')


if __name__ == '__main__':
    main()
