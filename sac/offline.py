"""Full-truth LP/MILP reference with an original-model bound and physical replay."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeWarning, linprog, milp
from scipy.sparse import coo_matrix

from env import Scenario, load_params
from env.replay import replay_power_schedule
from .evaluate import summarize_infos, validate_trajectory


def _model(scenario, params, margin):
    horizon, devices = scenario.T, len(scenario.vehicles) + 1
    costs, lower, upper, integral = [], [], [], []

    def allocate(size, cost=0., lo=0., hi=np.inf, integer=0):
        start = len(costs)
        costs.extend(np.broadcast_to(cost, (size,)).tolist())
        lower.extend(np.broadcast_to(lo, (size,)).tolist())
        upper.extend(np.broadcast_to(hi, (size,)).tolist())
        integral.extend([integer] * size)
        return np.arange(start, start + size)

    cp = allocate(devices * horizon).reshape(devices, horizon)
    dp = allocate(devices * horizon).reshape(devices, horizon)
    buy = scenario.truth[0] + params['tariff']['buy_adder_sgd_per_kwh']
    imports = allocate(horizon, buy * .5)
    exports = allocate(horizon, -scenario.truth[0] * .5)
    contract = params['building']['contract_capacity_kw']
    ucc = params['tariff']['ucc_month_sgd_per_kw'] / params['tariff']['month_to_day_divisor']
    initial_excess = max(0., scenario.initial_peak_kw - contract)
    peak = int(allocate(1, ucc, lo=initial_excess)[0])
    equality, inequality, rhs_eq, rhs_ub = [], [], [], []
    binary = bool(np.any(scenario.truth[0] < 0) or min(params['bess']['wear_sgd_per_kwh'],
                                                                  params['ev']['wear_sgd_per_kwh']) <= 0)
    for device in range(devices):
        battery = params['bess'] if device == 0 else params['ev']
        vehicle = None if device == 0 else scenario.vehicles[device - 1]
        if device == 0:
            arrival, departure = 0, horizon
            initial = target = scenario.bess_initial_kwh
        elif vehicle is not None:
            arrival, departure = vehicle.arrival, vehicle.departure
            initial, target = vehicle.initial_kwh, vehicle.target_kwh
        else:
            arrival = departure = 0
            initial = target = 0.
        for t in range(horizon):
            active = arrival <= t < departure
            upper[int(cp[device, t])] = battery['charge_max_kw'] if active else 0.
            upper[int(dp[device, t])] = battery['discharge_max_kw'] if active else 0.
            costs[int(cp[device, t])] = battery['wear_sgd_per_kwh'] * .5
            costs[int(dp[device, t])] = battery['wear_sgd_per_kwh'] * .5
        if departure == arrival:
            continue
        minimum = battery['capacity_kwh'] * battery['soc_min']
        maximum = battery['capacity_kwh'] * battery['soc_max']
        safety = min(margin, max(0., maximum - target) / 4)
        energy = allocate(departure - arrival + 1, lo=minimum + safety, hi=maximum - safety)
        lower[int(energy[0])] = upper[int(energy[0])] = initial
        lower[int(energy[-1])] = max(minimum + safety, target + safety)
        for index, t in enumerate(range(arrival, departure)):
            equality.append({int(energy[index + 1]): 1., int(energy[index]): -1.,
                             int(cp[device, t]): -battery['eta_ch'] * .5,
                             int(dp[device, t]): .5 / battery['eta_dis']})
            rhs_eq.append(0.)
            if binary:
                sign = int(allocate(1, hi=1., integer=1)[0])
                inequality.append({int(cp[device, t]): 1., sign: -battery['charge_max_kw']})
                rhs_ub.append(0.)
                inequality.append({int(dp[device, t]): 1., sign: battery['discharge_max_kw']})
                rhs_ub.append(battery['discharge_max_kw'])
    for t in range(horizon):
        net = scenario.truth[1, t] - scenario.truth[2, t]
        balance = {int(imports[t]): 1., int(exports[t]): -1.}
        for device in range(devices):
            balance[int(cp[device, t])] = -1.
            balance[int(dp[device, t])] = 1.
        equality.append(balance)
        rhs_eq.append(net)
        inequality.append({int(imports[t]): 1., int(exports[t]): -1., peak: -1.})
        rhs_ub.append(contract)

    def sparse(rows):
        values, row_indices, columns = [], [], []
        for row, coefficients in enumerate(rows):
            for column, value in coefficients.items():
                row_indices.append(row)
                columns.append(column)
                values.append(value)
        return coo_matrix((values, (row_indices, columns)), shape=(len(rows), len(costs))).tocsr()

    return (np.array(costs), np.array(lower), np.array(upper), np.array(integral),
            sparse(equality), np.array(rhs_eq), sparse(inequality), np.array(rhs_ub),
            cp, dp, -ucc * initial_excess, binary)


def _solve(scenario, params, margin):
    c, lower, upper, integral, ae, be, au, bu, cp, dp, offset, binary = _model(scenario, params, margin)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=OptimizeWarning)
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=RuntimeWarning)
        if binary:
            result = milp(c, integrality=integral, bounds=Bounds(lower, upper),
                          constraints=[LinearConstraint(ae, be, be), LinearConstraint(au, -np.inf, bu)],
                          options={'mip_rel_gap': 1e-10, 'time_limit': 120., 'threads': 1})
        else:
            result = linprog(c, A_ub=au, b_ub=bu, A_eq=ae, b_eq=be,
                             bounds=list(zip(lower, upper)), method='highs',
                             options={'primal_feasibility_tolerance': 1e-9,
                                      'dual_feasibility_tolerance': 1e-9, 'threads': 1})
    if not result.success:
        raise RuntimeError(f'offline optimisation failed: {result.message}')
    power = (result.x[cp] - result.x[dp]).T
    simultaneous = float(np.minimum(result.x[cp], result.x[dp]).max())
    if simultaneous > 1e-6:
        raise AssertionError('offline optimum contains simultaneous device charge/discharge')
    bound = result.mip_dual_bound if binary else result.fun
    return power, float(result.fun + offset), float(bound + offset), 'MILP' if binary else 'LP'


def solve_offline(scenario, params=None):
    scenario = Scenario.from_dict(scenario) if isinstance(scenario, dict) else scenario
    params = load_params() if params is None else params
    _, _, lower_bound, method = _solve(scenario, params, 0.)
    powers, feasible_objective, _, _ = _solve(scenario, params, 1e-5)
    infos, _ = replay_power_schedule(scenario, powers, params=params)
    summary = summarize_infos(infos)
    validate_trajectory(infos, summary, scenario.to_dict(), horizon=scenario.T)
    error = abs(summary['variable_cost_sgd'] - lower_bound)
    if error >= .01 or abs(summary['variable_cost_sgd'] - feasible_objective) >= .01:
        raise AssertionError(f'offline replay/objective mismatch {error:.12g} SGD')
    summary.update(strategy='offline_opt', n_slots=len(scenario.vehicles),
                   objective_lower_bound_sgd=lower_bound, feasible_objective_sgd=feasible_objective,
                   objective_replay_error_sgd=error, solver_method=method,
                   numerical_energy_margin_kwh=1e-5)
    return infos, summary, scenario.to_dict(), powers
