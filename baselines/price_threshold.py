"""Baseline: charge when USEP is low, discharge when USEP is high; BESS and EVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.common import (load_scenario, present_mask, replay, summarize,
                              train_usep, write_strategy_csv)
from env import V2GEnv, load_params


STRATEGY = "price_threshold"


def powers_from_threshold(scenario, params=None, low_q=0.30, high_q=0.70,
                          ev_charge_cap_kw=None):
    params = load_params() if params is None else params
    usep = np.asarray(scenario["truth"][0], dtype=np.float64)
    usep_train = train_usep(params)
    low = float(np.quantile(usep_train, low_q))
    high = float(np.quantile(usep_train, high_q))

    n_slots = len(scenario["vehicles"])
    T = len(scenario["truth"][0])
    mask = present_mask(scenario)
    ev_charge_max = float(params["ev"]["charge_max_kw"])
    ev_charge_cap = ev_charge_max if ev_charge_cap_kw is None else float(ev_charge_cap_kw)

    env = V2GEnv(n_slots=n_slots, params=params)
    try:
        env.reset(options={"scenario": scenario})
        powers = np.zeros((T, n_slots + 1), dtype=np.float64)
        for t in range(T):
            bounds = env.feasible_intervals()
            lo = bounds[:, 0].copy()
            hi = bounds[:, 1].copy()

            p = np.zeros(n_slots + 1, dtype=np.float64)
            if usep[t] < low:
                p[0] = hi[0]
                for i in range(n_slots):
                    if mask[t, i]:
                        p[i + 1] = min(hi[i + 1], ev_charge_cap)
            elif usep[t] > high:
                p[0] = lo[0]
                for i in range(n_slots):
                    if mask[t, i]:
                        p[i + 1] = lo[i + 1]

            # Defensive clip: absorbs ULP-level drift between read and check
            p = np.clip(p, lo, hi)
            powers[t] = p
            try:
                env.step_power(p, tol=1e-4)
            except ValueError as err:
                deviations = np.maximum(np.maximum(lo - p, p - hi), 0.0)
                worst = int(np.argmax(deviations))
                print(f"step_power rejected at t={t}: device={worst} "
                      f"p={p[worst]:+.9f} lo={lo[worst]:+.9f} hi={hi[worst]:+.9f} "
                      f"dev={deviations[worst]:+.3e}")
                raise
        return powers, low, high
    finally:
        env.close()


def run(scenario, out_dir, params=None, low_q=0.30, high_q=0.70, ev_charge_cap_kw=None):
    params = load_params() if params is None else params
    powers, low, high = powers_from_threshold(
        scenario, params, low_q=low_q, high_q=high_q, ev_charge_cap_kw=ev_charge_cap_kw)
    infos, summary = replay(scenario, powers, tol=1e-6)
    date = scenario["date"]
    seed = int(scenario["seed"])
    n_slots = len(scenario["vehicles"])
    csv_path = Path(out_dir) / f"{STRATEGY}_{date}_seed{seed}_n{n_slots}.csv"
    write_strategy_csv(csv_path, infos, STRATEGY)
    extra = {"low_q": low_q, "high_q": high_q, "usep_low": low, "usep_high": high,
             "ev_charge_cap_kw": ev_charge_cap_kw if ev_charge_cap_kw is not None else ""}
    return summarize(date, seed, STRATEGY, summary, csv_path, extra=extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", default="results/baselines")
    parser.add_argument("--low-q", type=float, default=0.30)
    parser.add_argument("--high-q", type=float, default=0.70)
    parser.add_argument("--ev-charge-cap-kw", type=float, default=None)
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)
    row = run(scenario, args.out, low_q=args.low_q, high_q=args.high_q,
              ev_charge_cap_kw=args.ev_charge_cap_kw)
    print(f"wrote {row['csv_path']}")
    for key in ("variable_cost_sgd", "total_cost_sgd", "peak_grid_kw",
                "bess_final_kwh", "departure_success_rate", "usep_low", "usep_high"):
        print(f"{key:>22} = {row[key]}")


if __name__ == "__main__":
    main()