"""SAC against the technical plan's official baselines, on the saved test and extended grids.

Official strategies (technical plan §6.1, implemented by C in baselines/ and milp/):
- arrival_charge: 无储能 / 到场就充, the cost upper bound;
- price_threshold: 规则策略 (低价充、高价放);
- milp: offline optimum on the whole day's true data, the cost lower bound. B's certified
  offline_opt solves the same problem and matches C's MILP to within 1e-4 S$ on every test
  case, so it stands in for C's MILP here and pulp is not needed.

C's two rule baselines run on exactly the scenarios SAC was evaluated on (read back from
the saved .scenario.json files) and are settled by A's env replay. Uncertainty is reported
two ways: seeds resampled within the fixed test days, and test days resampled. The 20 seeds
of one day share its prices and loads, so days are the independent unit.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from baselines.arrival_charge import arrival_charge_powers
from baselines.price_threshold import powers_from_threshold
from env import load_params
from env.replay import replay_power_schedule
from .comparison import compare, describe
from .evaluate import summarize_infos, validate_trajectory
from .protocol import write_json


OFFICIAL = ('arrival_charge', 'price_threshold')
STRATEGIES = ('arrival_charge', 'price_threshold', 'SAC', 'milp')
FAMILIES = ('ladder_0', 'ladder_2', 'ladder_4', 'ladder_6', 'ladder_8', 'ladder_10', 'mixed')


def _key(row):
    return row['date'], int(row['scenario_seed'])


def _read(path):
    return json.loads(Path(path).read_text())


def official_rows(suite_dir, sac_rows, params):
    """Run C's rule baselines on every scenario SAC saw in `suite_dir`."""
    rows = {strategy: [] for strategy in OFFICIAL}
    for sac in sac_rows:
        n = int(sac['n_slots'])
        stem = Path(suite_dir) / 'SAC' / f"SAC_{sac['date']}_seed{sac['scenario_seed']}_n{n}"
        scenario = _read(stem.with_suffix('.scenario.json'))
        for strategy in OFFICIAL:
            if strategy == 'arrival_charge':
                powers = arrival_charge_powers(scenario, params)
            else:
                powers = powers_from_threshold(scenario, params)[0]
            infos, _ = replay_power_schedule(scenario, np.asarray(powers, dtype=np.float64), tol=1e-6)
            summary = summarize_infos(infos)
            validate_trajectory(infos, summary, scenario)
            summary.update(strategy=strategy, n_slots=n, price_regime=sac['price_regime'])
            rows[strategy].append(summary)
    return rows


def _position(cost, milp, upper):
    """Where `cost` sits between the bounds: 0 % = MILP, 100 % = upper."""
    return float(100 * (cost - milp) / (upper - milp)) if upper - milp > 1e-9 else None


def _closed(cost, baseline, milp):
    """Share of the baseline-to-MILP gap that `cost` closes."""
    return float(100 * (baseline - cost) / (baseline - milp)) if baseline - milp > 1e-9 else None


def group_report(sac_rows, rows, offline_rows, building_rows):
    """Means, paired savings with both intervals, gap positions and per-case counts."""
    report = compare(sac_rows, [row for strategy in OFFICIAL for row in rows[strategy]])
    report['milp'] = describe(offline_rows)
    report['building_only'] = describe(building_rows)
    milp = report['milp']['variable_cost_sgd']
    upper = report['arrival_charge']['variable_cost_sgd']
    # 0 % = MILP lower bound, 100 % = arrival_charge upper bound.
    report['gap_position_pct'] = {s: _position(report[s]['variable_cost_sgd'], milp, upper)
                                  for s in STRATEGIES}
    sac_cost = report['SAC']['variable_cost_sgd']
    report['gap_closed_by_sac_pct'] = {s: _closed(sac_cost, report[s]['variable_cost_sgd'], milp)
                                       for s in OFFICIAL}
    sac = {_key(r): r['variable_cost_sgd'] for r in sac_rows}
    lower = {_key(r): r['variable_cost_sgd'] for r in offline_rows}
    report['per_case'] = {
        'cases': len(sac_rows),
        'sac_cheaper_cases': {s: int(sum(sac[_key(r)] < r['variable_cost_sgd'] - 1e-9 for r in rows[s]))
                              for s in OFFICIAL},
        'milp_not_above_sac_cases': int(sum(lower[k] <= sac[k] + 1e-6 for k in sac)),
    }
    report['days_won_by_sac'] = {
        s: int(sum(v > 0 for v in report['comparisons'][s]['saving_by_date_sgd'].values()))
        for s in OFFICIAL}
    report['days'] = len(report['comparisons'][OFFICIAL[0]]['saving_by_date_sgd'])
    groups = {}
    for group in ('flat_regular', 'high_volatility_spike'):
        pairs = {_key(r) for r in sac_rows if r.get('price_regime') == group}
        if not pairs:
            continue
        subset = lambda selected: [r for r in selected if _key(r) in pairs]
        entry = {s: describe(subset(rows[s])) for s in OFFICIAL}
        entry['SAC'] = describe(subset(sac_rows))
        entry['milp'] = describe(subset(offline_rows))
        entry['building_only'] = describe(subset(building_rows))
        entry['dates'] = sorted({d for d, _ in pairs})
        groups[group] = entry
    report['price_groups'] = groups
    return report


def suite_report(suite_dir, params):
    suite_dir = Path(suite_dir)
    sac_rows = _read(suite_dir / 'sac_summaries.json')
    rows = official_rows(suite_dir, sac_rows, params)
    offline_rows = _read(suite_dir / 'offline_summaries.json')
    building_rows = _read(suite_dir / 'building_summaries.json')
    return group_report(sac_rows, rows, offline_rows, building_rows), rows


def cross_check(rows, offline_rows, summary_csv):
    """Compare with C's own runner output (analysis.run_all) when it is available."""
    if not Path(summary_csv).exists():
        return {'status': 'not_run', 'summary_csv': str(summary_csv)}
    c_rows = defaultdict(dict)
    with Path(summary_csv).open(newline='') as handle:
        for row in csv.DictReader(handle):
            c_rows[(row['date'], int(row['seed']))][row['strategy']] = float(row['variable_cost_sgd'])
    worst = {}
    for strategy in OFFICIAL:
        worst[strategy] = max(abs(c_rows[_key(r)][strategy] - r['variable_cost_sgd']) for r in rows[strategy])
    worst['milp_vs_offline_opt'] = max(abs(c_rows[_key(r)]['milp'] - r['variable_cost_sgd'])
                                       for r in offline_rows)
    if max(worst[s] for s in OFFICIAL) > 1e-6 or worst['milp_vs_offline_opt'] > 1e-3:
        raise AssertionError(f'official baselines disagree with {summary_csv}: {worst}')
    return {'status': 'matched', 'summary_csv': str(summary_csv), 'max_abs_difference_sgd': worst}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/sac'))
    parser.add_argument('--c-results', type=Path, default=Path('results'),
                        help='directory holding C\'s summary_n{N}.csv, used for a cross-check if present')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    output = args.output if args.output is not None else args.results / 'official'
    params = load_params()
    for n in (5, 10):
        suite = args.results / f'test_n{n}'
        report, rows = suite_report(suite, params)
        report['cross_check_with_c'] = cross_check(
            rows, _read(suite / 'offline_summaries.json'), args.c_results / f'summary_n{n}.csv')
        write_json(output / f'test_n{n}.json', report)
        print(f"N={n}: SAC {report['SAC']['variable_cost_sgd']:.2f}, "
              + ', '.join(f"{s} {report[s]['variable_cost_sgd']:.2f}" for s in ('arrival_charge', 'price_threshold', 'milp'))
              + f"; C cross-check {report['cross_check_with_c']['status']}")
    extended = {}
    for family in FAMILIES:
        directory = args.results / 'extended_realistic' / family
        if directory.exists():
            extended[family], _ = suite_report(directory, params)
    if extended:
        write_json(output / 'extended.json', extended)
        print(f'extended families: {", ".join(extended)}')


if __name__ == '__main__':
    main()
