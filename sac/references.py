"""Local diagnostic comparators until C supplies the shared baseline modules.

All controllers use the same public environment, physical intervals and settlement.
Rule thresholds are fixed before validation, in SGD/kWh of USEP.
"""
import numpy as np


class Reference:
    def __init__(self, strategy, seed=6509):
        if strategy not in ('random', 'immediate', 'rule'):
            raise ValueError('unknown reference strategy')
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def act(self, obs, environment):
        if self.strategy == 'random':
            return self.rng.uniform(-1., 1., environment.n_slots + 1)
        bounds = environment.feasible_intervals()
        power = np.zeros(environment.n_slots + 1)
        # Immediate charging stops at the departure target, including cars arriving above it.
        gap = obs['ev'][:, 3].astype(np.float64)
        power[1:] = gap / (environment.ev.eta_ch * environment.dt)
        if self.strategy == 'rule':
            price = obs['timeseries'][0, 7]
            if price < .15:
                power[0] = environment.bess.charge_max_kw
            elif price > .25:
                power[0] = -environment.bess.discharge_max_kw
                # Leave enough energy to reach the target, with urgency from the projection.
                power[1:] = -environment.ev.discharge_max_kw
            else:
                remaining = obs['ev'][:, 2].clip(min=1)
                power[1:] /= remaining
        power = np.clip(power, bounds[:, 0], bounds[:, 1])
        span = bounds[:, 1] - bounds[:, 0]
        action = np.zeros_like(power)
        np.divide(2 * (power - bounds[:, 0]), span, out=action, where=span > 1e-12)
        action[span > 1e-12] -= 1
        return action
