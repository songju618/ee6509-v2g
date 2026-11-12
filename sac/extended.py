"""Held-out 10-slot vehicle-count ladder and day/night mixed parking experiments."""
from pathlib import Path
from datetime import date as Date

import numpy as np

from env.recording import occupancy_mask
from .protocol import grid, write_json
from .audit import audit_saved_episode
from .scenarios import LADDER_COUNTS, extended_scenario
from .suite import evaluate_scenarios


def evaluate_extended(policy, output, definition):
    output = Path(output)
    reports = {}
    histogram = np.zeros(11, dtype=int)
    early_cases = late_cases = weekend_visitor_cases = 0
    maximum_error = 0.
    audited_episodes = departures = 0
    for family, count in [('ladder', count) for count in LADDER_COUNTS] + [('mixed', None)]:
        name = f'ladder_{count}' if family == 'ladder' else 'mixed'
        scenarios = [extended_scenario(date,seed,family=family,count=count).to_dict()
                     for date,seed in grid('test')]
        counts = []
        for scenario in scenarios:
            mask = occupancy_mask(scenario)
            active = mask.sum(axis=1)
            histogram += np.bincount(active,minlength=11)
            counts.append(active.tolist())
            if family == 'mixed':
                early_cases += int(mask[:3].any())
                late_cases += int(mask[44:47].any())
                if Date.fromisoformat(scenario['date']).weekday() >= 5:
                    weekend_visitor_cases += int(mask.any())
        report = evaluate_scenarios(policy,scenarios,output/name,definition)
        report['actual_occupancy_mean_profile'] = np.mean(counts,axis=0).tolist()
        report['physical_slots'] = 10
        report['arrival_vehicle_count'] = count
        write_json(output/name/'comparison.json',report)
        scenarios_by_case = {}
        for strategy in ('SAC','immediate','rule','random','offline_opt'):
            paths = list((output/name/strategy).glob('*.csv'))
            if len(paths) != 140:
                raise AssertionError('each extended family and strategy must contain 140 trajectories')
            for path in paths:
                daily, exported = audit_saved_episode(path, 10, strategy)
                pair = (daily['date'], daily['scenario_seed'])
                if pair in scenarios_by_case and scenarios_by_case[pair] != exported:
                    raise AssertionError('extended strategies used different scenarios')
                scenarios_by_case[pair] = exported
                audited_episodes += 1
                if strategy == 'SAC':
                    departures += len(daily['departures'])
        reports[name] = report
        maximum_error = max(maximum_error,report['offline_opt']['maximum_replay_error_sgd'])
    if not all(histogram > 0) or min(early_cases,late_cases,weekend_visitor_cases) == 0:
        raise AssertionError('realistic suite must cover 0..10 vehicles, early/late and weekend parking')
    summary = {'families':reports,'scenarios':980,'strategies_per_scenario':5,
               'physical_slots':10,'audited_episodes':audited_episodes,'sac_departures':departures,
               'departure_success_rate':1.0,'bess_terminal_success_rate':1.0,'occupancy_histogram':histogram.tolist(),
               'early_mixed_cases':early_cases,'late_mixed_cases':late_cases,
               'weekend_visitor_cases':weekend_visitor_cases,'maximum_offline_replay_error_sgd':maximum_error}
    write_json(output/'summary.json',summary)
    return summary
