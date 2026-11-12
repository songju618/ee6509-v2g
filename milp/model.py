"""MILP model for the day-ahead reference, matching D's physics and A's settlement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from env import load_params


@dataclass
class MilpModel:
    problem: pulp.LpProblem
    p_bess: dict
    p_ev: dict
    p_grid_buy: dict
    p_grid_sell: dict
    z: dict  # kept for backwards compatibility; now a single 'peak' var
    peak: object = None


def build_model(scenario, params=None, verbose=False):
    params = load_params() if params is None else params
    truth = np.asarray(scenario["truth"], dtype=np.float64)
    T = truth.shape[1]
    N = len(scenario["vehicles"])
    dt = float(params["simulation"]["delta_t_h"])

    load = truth[1]
    pv = truth[2]
    usep = truth[0]
    buy_adder = float(params["tariff"]["buy_adder_sgd_per_kwh"])
    ucc_month = float(params["tariff"]["ucc_month_sgd_per_kw"])
    month_div = float(params["tariff"]["month_to_day_divisor"])
    ucc_daily = ucc_month / month_div
    contract = float(params["building"]["contract_capacity_kw"])

    bess = params["bess"]
    e_b_cap = float(bess["capacity_kwh"])
    e_b_min = e_b_cap * float(bess["soc_min"])
    e_b_max = e_b_cap * float(bess["soc_max"])
    e_b0 = e_b_cap * float(bess["initial_soc"])
    p_b_ch = float(bess["charge_max_kw"])
    p_b_dis = float(bess["discharge_max_kw"])
    eta_b_ch = float(bess["eta_ch"])
    eta_b_dis = float(bess["eta_dis"])
    c_bess = float(bess["wear_sgd_per_kwh"])

    ev = params["ev"]
    e_e_cap = float(ev["capacity_kwh"])
    e_e_min = e_e_cap * float(ev["soc_min"])
    e_e_max = e_e_cap * float(ev["soc_max"])
    p_e_ch = float(ev["charge_max_kw"])
    p_e_dis = float(ev["discharge_max_kw"])
    eta_e_ch = float(ev["eta_ch"])
    eta_e_dis = float(ev["eta_dis"])
    c_ev = float(ev["wear_sgd_per_kwh"])

    vehicles = scenario["vehicles"]
    present = np.zeros((T, N), dtype=bool)
    arrival = np.full(N, -1, dtype=int)
    departure = np.full(N, -1, dtype=int)
    initial_e = np.zeros(N, dtype=np.float64)
    target_e = np.zeros(N, dtype=np.float64)
    for i, v in enumerate(vehicles):
        if v is None:
            continue
        a, d = int(v["arrival"]), int(v["departure"])
        present[a:d, i] = True
        arrival[i] = a
        departure[i] = d
        initial_e[i] = float(v["initial_kwh"])
        target_e[i] = float(v["target_kwh"])

    prob = pulp.LpProblem("v2g_day_ahead", pulp.LpMinimize)
    idxT = range(T)
    idxN = range(N)

    # ---------------- BESS ----------------
    p_bess = {t: pulp.LpVariable(f"p_bess_{t}", lowBound=-p_b_dis, upBound=p_b_ch) for t in idxT}
    u_bess = {t: pulp.LpVariable(f"u_bess_{t}", cat=pulp.LpBinary) for t in idxT}
    p_b_pos = {t: pulp.LpVariable(f"p_b_pos_{t}", lowBound=0, upBound=p_b_ch) for t in idxT}
    p_b_neg = {t: pulp.LpVariable(f"p_b_neg_{t}", lowBound=0, upBound=p_b_dis) for t in idxT}

    e_b = {0: pulp.LpVariable("e_b_0", lowBound=e_b_min, upBound=e_b_max)}
    prob += e_b[0] == e_b0
    for t in range(1, T + 1):
        e_b[t] = pulp.LpVariable(f"e_b_{t}", lowBound=e_b_min, upBound=e_b_max)

    for t in idxT:
        prob += p_bess[t] == p_b_pos[t] - p_b_neg[t]
        prob += p_b_pos[t] <= p_b_ch * u_bess[t]
        prob += p_b_neg[t] <= p_b_dis * (1 - u_bess[t])
        prob += (e_b[t + 1] == e_b[t]
                 + eta_b_ch * p_b_pos[t] * dt
                 - p_b_neg[t] * (1.0 / eta_b_dis) * dt)

    prob += e_b[T] >= e_b0

    # ---------------- EV ----------------
    p_ev = {(t, i): pulp.LpVariable(f"p_ev_{t}_{i}", lowBound=-p_e_dis, upBound=p_e_ch)
            for t in idxT for i in idxN}
    u_ev = {(t, i): pulp.LpVariable(f"u_ev_{t}_{i}", cat=pulp.LpBinary)
            for t in idxT for i in idxN}
    p_e_pos = {(t, i): pulp.LpVariable(f"p_e_pos_{t}_{i}", lowBound=0, upBound=p_e_ch)
               for t in idxT for i in idxN}
    p_e_neg = {(t, i): pulp.LpVariable(f"p_e_neg_{t}_{i}", lowBound=0, upBound=p_e_dis)
               for t in idxT for i in idxN}

    e_ev = {}
    for i in idxN:
        if arrival[i] < 0:
            for t in range(T + 1):
                e_ev[(t, i)] = pulp.LpVariable(f"e_ev_{t}_{i}", lowBound=0, upBound=0)
                prob += e_ev[(t, i)] == 0
            continue
        a = arrival[i]
        d = departure[i]
        for t in range(T + 1):
            if t == a:
                e_ev[(t, i)] = pulp.LpVariable(
                    f"e_ev_{t}_{i}", lowBound=e_e_min, upBound=e_e_max)
                prob += e_ev[(t, i)] == initial_e[i]
            elif a < t <= d:
                e_ev[(t, i)] = pulp.LpVariable(
                    f"e_ev_{t}_{i}", lowBound=e_e_min, upBound=e_e_max)
            else:
                e_ev[(t, i)] = pulp.LpVariable(f"e_ev_{t}_{i}", lowBound=0, upBound=0)
                prob += e_ev[(t, i)] == 0

    for i in idxN:
        if arrival[i] < 0:
            for t in idxT:
                prob += p_ev[(t, i)] == 0
            continue
        a = arrival[i]
        d = departure[i]
        for t in idxT:
            prob += p_ev[(t, i)] == p_e_pos[(t, i)] - p_e_neg[(t, i)]
            prob += p_e_pos[(t, i)] <= p_e_ch * u_ev[(t, i)]
            prob += p_e_neg[(t, i)] <= p_e_dis * (1 - u_ev[(t, i)])
            if not present[t, i]:
                prob += p_ev[(t, i)] == 0
                continue
            prob += (e_ev[(t + 1, i)] == e_ev[(t, i)]
                     + eta_e_ch * p_e_pos[(t, i)] * dt
                     - p_e_neg[(t, i)] * (1.0 / eta_e_dis) * dt)
            k = d - t
            lower = max(e_e_min, target_e[i] - eta_e_ch * p_e_ch * max(k - 1, 0) * dt)
            prob += e_ev[(t + 1, i)] >= lower
            if t == d - 1:
                prob += e_ev[(t + 1, i)] >= target_e[i]

    # ---------------- Grid split and single peak variable ----------------
    p_grid = {t: pulp.LpVariable(f"p_grid_{t}", lowBound=-1e6, upBound=1e6) for t in idxT}
    u_grid = {t: pulp.LpVariable(f"u_grid_{t}", cat=pulp.LpBinary) for t in idxT}
    p_buy = {t: pulp.LpVariable(f"p_buy_{t}", lowBound=0) for t in idxT}
    p_sell = {t: pulp.LpVariable(f"p_sell_{t}", lowBound=0) for t in idxT}

    # Single peak variable: the max net import across the day, floored at contract.
    peak = pulp.LpVariable("peak", lowBound=contract)

    for t in idxT:
        prob += p_grid[t] == (load[t] - pv[t] + p_bess[t]
                              + pulp.lpSum(p_ev[(t, i)] for i in idxN))
        prob += p_buy[t] - p_sell[t] == p_grid[t]
        prob += p_buy[t] <= 1e6 * u_grid[t]
        prob += p_sell[t] <= 1e6 * (1 - u_grid[t])
        prob += peak >= p_grid[t]

    # ---------------- Objective ----------------
    # Note: capacity fee is ucc_daily * (peak - contract), charged once.
    objective = pulp.lpSum(
        (usep[t] + buy_adder) * p_buy[t] * dt
        - usep[t] * p_sell[t] * dt
        + c_bess * (p_b_pos[t] + p_b_neg[t]) * dt
        + pulp.lpSum(c_ev * (p_e_pos[(t, i)] + p_e_neg[(t, i)]) * dt for i in idxN)
        for t in idxT
    ) + ucc_daily * (peak - contract)
    prob += objective

    if verbose:
        print(f"MILP built: T={T}, N={N}, dt={dt}")

    return MilpModel(prob, p_bess, p_ev, p_buy, p_sell, {}, peak=peak)