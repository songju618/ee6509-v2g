"""Paired standard and realistic-suite evaluation using frozen policies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from env import V2GEnv
from .building import building_summary
from .evaluate import export_trajectory, rollout
from .metrics import enhanced_comparison, regime
from .offline import solve_offline
from .protocol import write_json


STRATEGIES = ('SAC', 'immediate', 'rule', 'random', 'offline_opt')


def evaluate_scenarios(policy, scenarios, output, definition, offline_output=None):
    output = Path(output)
    scenarios = list(scenarios)
    identity = {'policy_sha256': policy.checkpoint_sha256, 'price_regimes': definition,
                'scenario_digest': hashlib.sha256(
                    json.dumps(scenarios, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    identity_path = output / 'evaluation_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise FileExistsError(f'evaluation directory belongs to a different policy or scenario grid: {output}')
    write_json(identity_path, identity)
    offline_output = output / 'offline_opt' if offline_output is None else Path(offline_output)
    rows = {strategy: [] for strategy in STRATEGIES}
    building_rows = []
    environment = None
    try:
        for scenario in scenarios:
            n = len(scenario['vehicles'])
            if environment is None:
                environment = V2GEnv(n_slots=n)
            if environment.n_slots != n:
                raise ValueError('one paired suite must have a constant physical slot count')
            group = regime(scenario, definition)
            building = building_summary(scenario)
            building.update(price_regime=group, n_slots=n)
            building_rows.append(building)
            for strategy in STRATEGIES:
                directory = offline_output if strategy == 'offline_opt' else output / strategy
                stem = directory / f"{strategy}_{scenario['date']}_seed{scenario['seed']}_n{n}"
                summary_path = stem.with_suffix('.summary.json')
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                else:
                    if strategy == 'offline_opt':
                        infos, summary, exported, _ = solve_offline(scenario)
                        weights = None
                    else:
                        infos, summary, exported, weights = rollout(
                            environment, scenario['date'], scenario['seed'],
                            policy=policy if strategy == 'SAC' else None,
                            reference=strategy if strategy != 'SAC' else 'random',
                            collect_attention=strategy == 'SAC', scenario=scenario)
                    summary.update(strategy=strategy, n_slots=n, price_regime=group)
                    export_trajectory(directory, strategy, n, infos, summary, exported, weights)
                rows[strategy].append(summary)
        references = rows['immediate'] + rows['rule'] + rows['random']
        report = enhanced_comparison(rows['SAC'], references, rows['offline_opt'], building_rows, definition)
        write_json(output / 'sac_summaries.json', rows['SAC'])
        write_json(output / 'reference_summaries.json', references)
        write_json(output / 'offline_summaries.json', rows['offline_opt'])
        write_json(output / 'building_summaries.json', building_rows)
        write_json(output / 'comparison.json', report)
        return report
    finally:
        if environment is not None:
            environment.close()
