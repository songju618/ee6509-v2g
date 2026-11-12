"""Read-back audit shared by standard and extended trajectory exports."""
import csv
import json
import math
from pathlib import Path

from env.recording import csv_columns, occupancy_mask


def audit_saved_episode(path, n_slots, strategy):
    path = Path(path)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != csv_columns(n_slots):
            raise AssertionError(f'CSV columns mismatch at {path}')
        rows = list(reader)
    if len(rows) != 48 or [int(row['t']) for row in rows] != list(range(48)):
        raise AssertionError(f'CSV must contain 48 ordered steps at {path}')
    summary = json.loads(path.with_suffix('.summary.json').read_text())
    scenario = json.loads(path.with_suffix('.scenario.json').read_text())
    if (scenario['date'], scenario['seed']) != (summary['date'], summary['scenario_seed']):
        raise AssertionError(f'scenario and summary identifiers disagree at {path}')
    if not math.isclose(math.fsum(float(row['step_cost_sgd']) for row in rows),
                        summary['variable_cost_sgd'], rel_tol=0, abs_tol=1e-8):
        raise AssertionError(f'CSV settlement mismatch at {path}')
    mask = occupancy_mask(scenario)
    for t, row in enumerate(rows):
        if (row['strategy'] != strategy or row['date'] != summary['date']
                or int(row['seed']) != summary['scenario_seed']):
            raise AssertionError(f'CSV identifiers disagree at {path}')
        components = (float(row['purchase_sgd']) - float(row['revenue_sgd'])
                      + float(row['peak_increment_sgd']) + float(row['bess_wear_sgd'])
                      + float(row['ev_wear_sgd']))
        if not math.isclose(components, float(row['step_cost_sgd']), rel_tol=0, abs_tol=1e-8):
            raise AssertionError(f'CSV cost components disagree at {path}')
        for slot in range(n_slots):
            if not mask[t, slot] and (float(row[f'p_ev_{slot+1}_kw']) != 0
                                      or float(row[f'e_ev_{slot+1}_kwh']) != 0):
                raise AssertionError(f'empty slot has nonzero exported power or energy at {path}')
    if not all(record['success'] for record in summary['departures']):
        raise AssertionError(f'EV departure failed at {path}')
    if summary['bess_final_kwh'] < scenario['bess_initial_kwh'] - 1e-9:
        raise AssertionError(f'BESS terminal target failed at {path}')
    return summary, scenario
