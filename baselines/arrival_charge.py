"""Baseline: charge each present EV at max power until full; BESS idle.

This is the technical plan's 无储能 / 到场就充 baseline (§6.1), the cost upper bound.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.common import (load_scenario, present_mask, replay, summarize,
                              write_strategy_csv)
from env import V2GEnv, load_params


STRATEGY = "arrival_charge"


def arrival_charge_powers(scenario, params=None):
    """Charge every present EV at max, capped by the env's feasible upper bound.

    At a step where the EV's `hi` is < 6 kW (e.g. the battery is near e_max), the
    schedule charges at `hi`, not 6. Once the battery reaches e_max, `hi` is 0.
    """
    params = load_params() if params is None else params
    n_slots = len(scenario["vehicles"])
    T = len(scenario["truth"][0])
    p_ev_max = float(params["ev"]["charge_max_kw"])

    env = V2GEnv(n_slots=n_slots, params=params)
    try:
        env.reset(options={"scenario": scenario})
        powers = np.zeros((T, n_slots + 1), dtype=np.float64)
        for t in range(T):
            bounds = env.feasible_intervals()
            present = present_mask(scenario)[t]
            p = np.zeros(n_slots + 1, dtype=np.float64)
            for i in range(n_slots):
                if present[i]:
                    p[i + 1] = min(p_ev_max, bounds[i + 1, 1])
            powers[t] = p
            env.step_power(p)
        return powers
    finally:
        env.close()


def run(scenario, out_dir, params=None):
    params = load_params() if params is None else params
    powers = arrival_charge_powers(scenario, params)
    infos, summary = replay(scenario, powers, tol=1e-6)
    date = scenario["date"]
    seed = int(scenario["seed"])
    n_slots = len(scenario["vehicles"])
    csv_path = Path(out_dir) / f"{STRATEGY}_{date}_seed{seed}_n{n_slots}.csv"
    write_strategy_csv(csv_path, infos, STRATEGY)
    return summarize(date, seed, STRATEGY, summary, csv_path, n_slots=n_slots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help=".scenario.json path")
    parser.add_argument("--out", default="results/baselines")
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)
    row = run(scenario, args.out)
    print(f"wrote {row['csv_path']}")
    for key in ("variable_cost_sgd", "total_cost_sgd", "peak_grid_kw",
                "bess_final_kwh", "departure_success_rate"):
        print(f"{key:>22} = {row[key]}")


if __name__ == "__main__":
    main()