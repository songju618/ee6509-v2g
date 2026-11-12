"""Paired fixed-date seed bootstrap plus date-cluster sensitivity intervals."""
from collections import defaultdict

import numpy as np


def describe(rows):
    keys = ('variable_cost_sgd', 'total_cost_sgd', 'purchase_sgd', 'revenue_sgd',
            'peak_increment_sgd', 'bess_wear_sgd', 'ev_wear_sgd', 'peak_grid_kw',
            'bess_throughput_kwh', 'ev_throughput_kwh')
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    departures = [record for row in rows for record in row['departures']]
    result.update(scenarios=len(rows), departures=len(departures),
                  departure_success_rate=(sum(d['success'] for d in departures) / len(departures)
                                          if departures else None))
    return result


def compare(sac_rows, reference_rows):
    lookup = {(row['date'], row['scenario_seed']): row for row in sac_rows}
    groups = defaultdict(list)
    for row in reference_rows:
        groups[row['strategy']].append(row)
    result = {'SAC': describe(sac_rows), 'comparisons': {}}
    for strategy, rows in groups.items():
        if {(row['date'], row['scenario_seed']) for row in rows} != set(lookup):
            raise ValueError('strategies must use the same explicit scenario grid')
        by_date = defaultdict(list)
        for row in rows:
            difference = row['variable_cost_sgd'] - lookup[(row['date'], row['scenario_seed'])]['variable_cost_sgd']
            by_date[row['date']].append(difference)
        differences = np.array([np.mean(by_date[date]) for date in sorted(by_date)])
        rng = np.random.default_rng(6509)
        resampled = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        cluster_lower, cluster_upper = np.quantile(resampled, [.025, .975])
        seed_rng = np.random.default_rng(6509)
        fixed_date_samples = np.zeros(10000)
        for date in sorted(by_date):
            values = np.asarray(by_date[date])
            indices = seed_rng.integers(len(values), size=(10000, len(values)))
            fixed_date_samples += values[indices].mean(axis=1) / len(by_date)
        lower, upper = np.quantile(fixed_date_samples, [.025, .975])
        baseline = describe(rows)
        improvement = float(differences.mean())
        result[strategy] = baseline
        result['comparisons'][strategy] = {
            'mean_saving_sgd': improvement,
            'saving_percent': 100 * improvement / baseline['variable_cost_sgd'],
            'fixed_date_seed_bootstrap_95_ci_sgd': [float(lower), float(upper)],
            'date_cluster_bootstrap_95_ci_sgd': [float(cluster_lower), float(cluster_upper)],
            'significantly_better': bool(lower > 0),
            'saving_by_date_sgd': {date: float(np.mean(by_date[date])) for date in sorted(by_date)},
        }
    return result
