"""Rule baseline (technical plan §6.1): fixed USEP thresholds, peak-aware, EVs by departure.

Decisions at step t use only what the env observes at t: the current USEP, load and PV,
the running grid peak, each EV's shortfall to its departure target, and the env's own
12-step USEP forecast.

1. Peak guard: voluntary charging may raise grid import only up to
   max(contract capacity, running peak), so the rule never adds an excess-capacity fee
   by choice.
2. EVs take the minimum the departure projection forces, then charge toward the
   departure target (not to full), earliest departure first, within the peak guard.
   EVs never discharge.
3. BESS charges when USEP is below the low threshold, within the peak guard. It charges
   above its initial energy only if the forecast shows USEP above the high threshold;
   otherwise it only refills what it has discharged. When USEP is above the high
   threshold it discharges at the maximum feasible rate. Otherwise it is idle.

Thresholds are quantiles of the training-split USEP only. Every action is clipped to
`env.feasible_intervals()`, so departure targets and the BESS end-of-day rule are
enforced by A's projection, not reimplemented here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.common import (load_scenario, replay, summarize, train_usep,
                              write_strategy_csv)
from env import V2GEnv, load_params


STRATEGY = "price_threshold"
DEFAULT_LOW_Q = 0.2
DEFAULT_HIGH_Q = 0.8


def thresholds(params, low_q=DEFAULT_LOW_Q, high_q=DEFAULT_HIGH_Q):
    """USEP thresholds (S$/kWh) from the training split only."""
    usep_train = train_usep(params)
    return float(np.quantile(usep_train, low_q)), float(np.quantile(usep_train, high_q))


def powers_from_threshold(scenario, params=None, low_q=DEFAULT_LOW_Q, high_q=DEFAULT_HIGH_Q,
                          ev_charge_cap_kw=None):
    params = load_params() if params is None else params
    low, high = thresholds(params, low_q, high_q)

    truth = np.asarray(scenario["truth"], dtype=np.float64)
    T = truth.shape[1]
    n_slots = len(scenario["vehicles"])
    departure = [v["departure"] if v is not None else T for v in scenario["vehicles"]]
    dt = float(params["simulation"]["delta_t_h"])
    contract_kw = float(params["building"]["contract_capacity_kw"])
    ev_eta_ch = float(params["ev"]["eta_ch"])
    ev_cap_kw = (float(params["ev"]["charge_max_kw"]) if ev_charge_cap_kw is None
                 else float(ev_charge_cap_kw))
    bess_capacity = float(params["bess"]["capacity_kwh"])
    bess_initial = bess_capacity * float(params["bess"]["initial_soc"])
    bess_eta_ch = float(params["bess"]["eta_ch"])

    env = V2GEnv(n_slots=n_slots, params=params)
    try:
        obs, _ = env.reset(options={"scenario": scenario})
        powers = np.zeros((T, n_slots + 1), dtype=np.float64)
        for t in range(T):
            bounds = env.feasible_intervals()
            lo = bounds[:, 0].copy()
            hi = bounds[:, 1].copy()
            usep = truth[0, t]
            net_load = truth[1, t] - truth[2, t]
            present = np.flatnonzero(np.asarray(obs["ev_present"], dtype=bool))

            p = np.zeros(n_slots + 1, dtype=np.float64)
            # EVs: the minimum the departure projection forces; never discharge.
            for i in present:
                p[i + 1] = max(lo[i + 1], 0.0)

            # Peak guard: charging by choice stays under the running peak (or contract).
            cap_kw = max(contract_kw, float(obs["global"][1]))
            headroom = cap_kw - (net_load + p[1:].sum())

            # EVs toward their departure target, earliest departure first.
            for i in sorted(present, key=lambda k: departure[k]):
                shortfall_kwh = float(obs["ev"][i, 3])
                want = min(hi[i + 1], ev_cap_kw, shortfall_kwh / (ev_eta_ch * dt))
                add = min(max(want - p[i + 1], 0.0), max(headroom, 0.0))
                p[i + 1] += add
                headroom -= add

            # BESS: valley charge (gated by the forecast above E0), peak discharge.
            if usep < low:
                room = hi[0]
                if not np.any(obs["timeseries"][0, 8:] > high):
                    energy = float(obs["global"][0]) * bess_capacity
                    room = min(room, max(0.0, (bess_initial - energy) / (bess_eta_ch * dt)))
                p[0] = min(room, max(headroom, 0.0))
            elif usep > high:
                p[0] = lo[0]

            # The projection may force BESS charging near the end of the day.
            p = np.clip(p, lo, hi)
            powers[t] = p
            try:
                obs, *_ = env.step_power(p, tol=1e-4)
            except ValueError:
                deviations = np.maximum(np.maximum(lo - p, p - hi), 0.0)
                worst = int(np.argmax(deviations))
                print(f"step_power rejected at t={t}: device={worst} "
                      f"p={p[worst]:+.9f} lo={lo[worst]:+.9f} hi={hi[worst]:+.9f} "
                      f"dev={deviations[worst]:+.3e}")
                raise
        return powers, low, high
    finally:
        env.close()


def run(scenario, out_dir, params=None, low_q=DEFAULT_LOW_Q, high_q=DEFAULT_HIGH_Q,
        ev_charge_cap_kw=None):
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
    return summarize(date, seed, STRATEGY, summary, csv_path, extra=extra, n_slots=n_slots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", default="results/baselines")
    parser.add_argument("--low-q", type=float, default=DEFAULT_LOW_Q)
    parser.add_argument("--high-q", type=float, default=DEFAULT_HIGH_Q)
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
