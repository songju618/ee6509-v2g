"""Common percentage denominators and training-fitted price regimes."""
from __future__ import annotations

import numpy as np

from env import load_day
from .comparison import compare, describe


def price_regime_definition():
    ranges = []
    for day in range(1, 15):
        price = load_day(f'2026-10-{day:02d}')['usep']
        ranges.append(float(np.ptp(price)))
    return {'fitted_split': 'train', 'statistic': 'daily_USEP_max_minus_min',
            'threshold_sgd_per_kwh': float(np.quantile(ranges, .75)), 'quantile': .75}


def regime(scenario, definition):
    span = float(np.ptp(np.asarray(scenario['truth'])[0]))
    return 'high_volatility_spike' if span >= definition['threshold_sgd_per_kwh'] else 'flat_regular'


def ratio(numerator, denominator):
    if abs(denominator) <= 1e-8:
        return {'percent': None, 'numerator_sgd': float(numerator), 'denominator_sgd': float(denominator),
                'status': 'zero_denominator'}
    return {'percent': float(100 * numerator / denominator), 'numerator_sgd': float(numerator),
            'denominator_sgd': float(denominator), 'status': 'defined'}


def percentage_metrics(cost, immediate, random, optimum, building):
    saving = immediate - cost
    return {
        'optimality_gap_closure_from_immediate': ratio(saving, immediate - optimum),
        'optimality_gap_closure_from_random': ratio(random - cost, random - optimum),
        'net_ev_storage_cost_reduction': ratio(saving, immediate - building),
        'relative_saving_from_immediate': ratio(saving, immediate),
    }


def enhanced_comparison(sac_rows, reference_rows, offline_rows, building_rows, definition):
    lower_by_case = {(r['date'], r['scenario_seed']): r['objective_lower_bound_sgd'] for r in offline_rows}
    for row in sac_rows + reference_rows:
        if row['variable_cost_sgd'] < lower_by_case[(row['date'], row['scenario_seed'])] - .01:
            raise AssertionError('a feasible strategy undercuts the certified offline lower bound')
    report = compare(sac_rows, reference_rows)
    report['offline_opt'] = describe(offline_rows)
    report['offline_opt']['objective_lower_bound_sgd'] = float(
        np.mean([r['objective_lower_bound_sgd'] for r in offline_rows]))
    report['offline_opt']['maximum_replay_error_sgd'] = max(
        r['objective_replay_error_sgd'] for r in offline_rows)
    report['building_only'] = describe(building_rows)
    optimum = report['offline_opt']['objective_lower_bound_sgd']
    immediate, random, building = [report[key]['variable_cost_sgd']
                                   for key in ('immediate', 'random', 'building_only')]
    report['percentage_by_strategy'] = {}
    for strategy in ('SAC', 'immediate', 'rule', 'random', 'offline_opt'):
        report['percentage_by_strategy'][strategy] = percentage_metrics(
            report[strategy]['variable_cost_sgd'], immediate, random, optimum, building)
    report['percentage_metrics'] = report['percentage_by_strategy']['SAC']
    report['percentage_aggregation'] = 'Ratio of paired-grid mean costs, equal weight per scenario'
    report['price_regime_definition'] = definition
    report['price_groups'] = {}
    for group in ('high_volatility_spike', 'flat_regular'):
        selected = [row for row in sac_rows if row.get('price_regime') == group]
        if not selected:
            continue
        pairs = {(r['date'], r['scenario_seed']) for r in selected}
        subset = lambda rows: [r for r in rows if (r['date'], r['scenario_seed']) in pairs]
        grouped = compare(selected, subset(reference_rows))
        grouped['offline_opt'] = describe(subset(offline_rows))
        grouped['building_only'] = describe(subset(building_rows))
        grouped['dates'] = sorted({r['date'] for r in selected})
        grouped['percentage_metrics'] = percentage_metrics(
            grouped['SAC']['variable_cost_sgd'], grouped['immediate']['variable_cost_sgd'],
            grouped['random']['variable_cost_sgd'],
            float(np.mean([r['objective_lower_bound_sgd'] for r in subset(offline_rows)])),
            grouped['building_only']['variable_cost_sgd'])
        report['price_groups'][group] = grouped
    return report
