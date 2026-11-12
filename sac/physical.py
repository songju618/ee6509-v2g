"""Observation-only feasible intervals and the zero-power anchored action map."""
from functools import lru_cache

import numpy as np

from env.data import load_params


@lru_cache(maxsize=1)
def physical_params():
    return load_params()


def observation_step(obs):
    sine, cosine = np.asarray(obs['global'], dtype=np.float64)[2:4]
    return int(np.rint(np.mod(np.arctan2(sine, cosine), 2 * np.pi) * 48 / (2 * np.pi))) % 48


def feasible_from_observation(obs, params=None):
    """Recover live full-day intervals, with float32-observation precision.

    Formal and extended scenarios retain the parameter-file BESS initial energy
    and EV departure target. Terminal observations are never passed to env.step.
    """
    params = physical_params() if params is None else params
    ev_state = np.asarray(obs['ev'], dtype=np.float64)
    n = len(ev_state)
    bess, ev = params['bess'], params['ev']
    soc = np.r_[float(obs['global'][0]), ev_state[:, 1]]
    capacity = np.r_[bess['capacity_kwh'], np.full(n, ev['capacity_kwh'])]
    eta_ch = np.r_[bess['eta_ch'], np.full(n, ev['eta_ch'])]
    eta_dis = np.r_[bess['eta_dis'], np.full(n, ev['eta_dis'])]
    charge = np.r_[bess['charge_max_kw'], np.full(n, ev['charge_max_kw'])]
    discharge = np.r_[bess['discharge_max_kw'], np.full(n, ev['discharge_max_kw'])]
    minimum = capacity * np.r_[bess['soc_min'], np.full(n, ev['soc_min'])]
    maximum = capacity * np.r_[bess['soc_max'], np.full(n, ev['soc_max'])]
    energy = np.clip(capacity * soc, minimum, maximum)
    target = capacity * np.r_[bess['initial_soc'], np.full(n, ev['departure_target_soc'])]
    remaining = np.r_[48 - observation_step(obs), np.maximum(ev_state[:, 2], 1)]
    lower_energy = np.maximum(minimum, target - eta_ch * charge * .5 * (remaining - 1))
    delta = lower_energy - energy
    lo = np.maximum(-discharge, np.where(delta >= 0, delta / (eta_ch * .5), eta_dis * delta / .5))
    hi = np.minimum(charge, (maximum - energy) / (eta_ch * .5))
    lo = np.minimum(lo, hi)
    intervals = np.column_stack((lo, hi))
    intervals[1:][np.asarray(obs['ev_present']) < .5] = 0
    return intervals


def anchored_power(internal_action, intervals):
    u = np.clip(np.asarray(internal_action, dtype=np.float64), -1, 1)
    bounds = np.asarray(intervals, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    zero = np.clip(np.zeros_like(lo), lo, hi)
    return zero + np.maximum(u, 0) * (hi - zero) + np.minimum(u, 0) * (zero - lo)


def environment_action(internal_action, obs, params=None):
    bounds = feasible_from_observation(obs, params)
    power = anchored_power(internal_action, bounds)
    span = bounds[:, 1] - bounds[:, 0]
    action = np.zeros_like(power)
    movable = span > 1e-12
    action[movable] = 2 * (power[movable] - bounds[movable, 0]) / span[movable] - 1
    action[1:][np.asarray(obs['ev_present']) < .5] = 0
    return np.clip(action, -1, 1).astype(np.float32)


def engineered_observation(obs, params=None):
    params = physical_params() if params is None else params
    bounds = feasible_from_observation(obs, params)
    bess, ev = params['bess'], params['ev']
    present = np.asarray(obs['ev_present'], dtype=np.float64)
    raw_ev = np.asarray(obs['ev'], dtype=np.float64)
    rated = np.r_[bess['charge_max_kw'], np.full(len(present), ev['charge_max_kw'])]
    time_remaining = 48 - observation_step(obs)
    bess_gap = max(0., (bess['initial_soc'] - float(obs['global'][0])) * bess['capacity_kwh'])
    urgency = np.r_[bess_gap / (bess['eta_ch'] * rated[0] * .5 * time_remaining),
                     raw_ev[:, 3] / (ev['eta_ch'] * ev['charge_max_kw'] * .5 * np.maximum(raw_ev[:, 2], 1))]
    device = np.column_stack((bounds / rated[:, None], urgency))
    ev_features = np.column_stack((raw_ev, device[1:])) * present[:, None]
    timeseries = np.asarray(obs['timeseries'], dtype=np.float64).copy()
    buy = timeseries[0] + params['tariff']['buy_adder_sgd_per_kwh']
    current, future = buy[7], buy[8:]
    price = [current - future.mean(), future.max() - current, current - future.min()]
    for battery in (bess, ev):
        efficiency = battery['eta_ch'] * battery['eta_dis']
        wear = battery['wear_sgd_per_kwh']
        price.extend([current - future.min() / efficiency - wear * (1 + 1 / efficiency),
                      future.max() * efficiency - current - wear * (1 + efficiency)])
    net_load = timeseries[1, 7] - timeseries[2, 7]
    headroom = max(params['building']['contract_capacity_kw'], float(obs['global'][1])) - net_load
    availability = [bounds[:, 1].sum() / 120., -bounds[:, 0].sum() / 120.]
    global_features = np.r_[obs['global'], device[0], price, headroom / 60., present.sum() / 10., availability]
    timeseries[0] = np.arcsinh(timeseries[0] / .2)
    return {'timeseries': timeseries, 'global': global_features, 'ev': ev_features,
            'ev_present': present}
