"""Action-independent building bill, also used as a training control variate."""
import math

import numpy as np

from env import load_params
from env.physics import BatterySpec, settle_cost


class BuildingMeter:
    def __init__(self, params=None, initial_peak=0.):
        self.params = load_params() if params is None else params
        self.bess = BatterySpec.from_params(self.params['bess'])
        self.ev = BatterySpec.from_params(self.params['ev'])
        self.peak = initial_peak

    def step(self, net_load, usep):
        p = self.params
        info = settle_cost(load_kw=max(net_load, 0), pv_kw=max(-net_load, 0), usep=usep,
                           power=np.zeros(2), bess=self.bess, ev=self.ev, peak_before_kw=self.peak,
                           contract_capacity_kw=p['building']['contract_capacity_kw'],
                           ucc_daily=p['tariff']['ucc_month_sgd_per_kw'] / p['tariff']['month_to_day_divisor'],
                           adder=p['tariff']['buy_adder_sgd_per_kwh'], dt=.5)
        self.peak = info['peak_after_kw']
        return info


def building_summary(scenario, params=None):
    params = load_params() if params is None else params
    truth = np.asarray(scenario['truth'])
    meter = BuildingMeter(params, scenario['initial_peak_kw'])
    infos = [meter.step(load - pv, usep) for usep, load, pv in truth.T]
    keys = ('purchase_sgd', 'revenue_sgd', 'peak_increment_sgd', 'bess_wear_sgd', 'ev_wear_sgd')
    result = {key: math.fsum(info[key] for info in infos) for key in keys}
    variable = math.fsum(info['step_cost_sgd'] for info in infos)
    fixed = (params['pv']['daily_fixed_cost_sgd'] + params['tariff']['contract_month_sgd_per_kw']
             * params['building']['contract_capacity_kw'] / params['tariff']['month_to_day_divisor'])
    result.update(strategy='building_only', date=scenario['date'], scenario_seed=scenario['seed'],
                  variable_cost_sgd=variable, total_cost_sgd=variable + (fixed if truth.shape[1] == 48 else 0),
                  peak_grid_kw=meter.peak, departures=[], bess_throughput_kwh=0., ev_throughput_kwh=0.)
    return result
