"""Run C's strategies over the test grid and check the README acceptance for C.

Strategies, one row each per (date, seed):
- arrival_charge: the technical plan's 无储能 / 到场就充 baseline, the cost upper bound.
- price_threshold: the plan's 规则策略 (低价充、高价放), see baselines/price_threshold.py.
- milp: offline optimum on the whole day's true data, the cost lower bound.
- late_charge: extra reference, not a plan baseline (formerly named no_storage). BESS
  idle; each EV gets only what its departure constraint forces, as late as possible.

Acceptance (root README, row C), checked on every run:
- every case: milp <= price_threshold <= arrival_charge (variable cost);
- every MILP case is Optimal and its replayed cost is within 0.01 S$ of its objective.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from baselines.arrival_charge import run as run_arrival_charge
from baselines.common import replay, summarize, test_grid, write_strategy_csv
from baselines.price_threshold import run as run_price_threshold
from env import V2GEnv, load_params
from milp.solve import run as run_milp


STRATEGIES = ("late_charge", "arrival_charge", "price_threshold", "milp")
ORDERING = ("milp", "price_threshold", "arrival_charge")  # cheapest first, on every case
COST_TOL_SGD = 1e-6
MILP_RESIDUAL_TOL_SGD = 0.01

SUMMARY_KEYS = [
    "strategy", "date", "seed", "n_slots",
    "variable_cost_sgd", "fixed_pv_sgd", "fixed_contract_sgd", "total_cost_sgd",
    "peak_grid_kw", "bess_final_kwh", "departure_success_rate",
    "csv_path",
    "status", "milp_objective", "replayed_variable_cost", "residual_sgd",
]


def run_late_charge(scenario, out_dir, params=None):
    """BESS idle; each EV power is zero clipped to the feasible interval, so an EV is
    charged only when the departure projection forces it, and only to its target."""
    params = load_params() if params is None else params
    T = len(scenario["truth"][0])
    N = len(scenario["vehicles"])

    env = V2GEnv(n_slots=N, params=params)
    try:
        env.reset(options={"scenario": scenario})
        powers = np.zeros((T, N + 1), dtype=np.float64)
        for t in range(T):
            bounds = env.feasible_intervals()
            p = np.zeros(N + 1, dtype=np.float64)
            p = np.clip(p, bounds[:, 0], bounds[:, 1])
            powers[t] = p
            env.step_power(p, tol=1e-4)
    finally:
        env.close()

    infos, summary = replay(scenario, powers, tol=1e-4)
    date = scenario["date"]
    seed = int(scenario["seed"])
    csv_path = Path(out_dir) / f"late_charge_{date}_seed{seed}_n{N}.csv"
    write_strategy_csv(csv_path, infos, "late_charge")
    return summarize(date, seed, "late_charge", summary, csv_path, n_slots=N)


def run_grid(n_slots=5, seeds=range(100, 120), out_dir="results",
             time_limit=300, gap_rel=1e-6):
    out_root = Path(out_dir)
    summary_path = out_root / f"summary_n{n_slots}.csv"
    diagnostics_path = out_root / f"diagnostics_n{n_slots}.csv"
    rows = []
    for date, seed, scenario in test_grid(n_slots=n_slots, seeds=seeds):
        print(f"[{date} seed={seed} N={n_slots}]", flush=True)
        rows.append(run_late_charge(scenario, out_root / "baselines"))
        rows.append(run_arrival_charge(scenario, out_root / "baselines"))
        rows.append(run_price_threshold(scenario, out_root / "baselines"))
        try:
            rows.append(run_milp(scenario, out_root / "milp",
                                 time_limit=time_limit, gap_rel=gap_rel))
        except Exception as error:
            print(f"  MILP failed on {date} seed={seed}: {type(error).__name__}: {error}",
                  flush=True)
            rows.append({
                "strategy": "milp", "date": date, "seed": seed, "n_slots": n_slots,
                "variable_cost_sgd": float("nan"), "fixed_pv_sgd": 0.0,
                "fixed_contract_sgd": 0.0, "total_cost_sgd": float("nan"),
                "peak_grid_kw": float("nan"), "bess_final_kwh": float("nan"),
                "departure_success_rate": None, "csv_path": "",
                "status": "error", "milp_objective": float("nan"),
                "replayed_variable_cost": float("nan"), "residual_sgd": float("nan"),
                "error": f"{type(error).__name__}: {error}",
            })
        # Flush a snapshot after each case so nothing is lost on a later crash
        _write_csv(summary_path, rows, SUMMARY_KEYS)
        _write_csv(diagnostics_path, rows,
                   ["strategy", "date", "seed", "n_slots", "low_q", "high_q", "usep_low",
                    "usep_high", "ev_charge_cap_kw", "error"], extrasaction="ignore")
    print(f"wrote {summary_path} ({len(rows)} rows)")
    return summary_path, rows


def _write_csv(path, rows, fieldnames, extrasaction="ignore"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        writer.writeheader()
        writer.writerows(rows)


def check_acceptance(rows):
    """Return a list of acceptance failures; an empty list means C's acceptance holds."""
    by_case = {}
    for row in rows:
        key = (row["date"], row["seed"], row["n_slots"])
        by_case.setdefault(key, {})[row["strategy"]] = row
    failures = []
    for key, case in sorted(by_case.items()):
        costs = {s: case[s]["variable_cost_sgd"] for s in ORDERING if s in case}
        if len(costs) < len(ORDERING) or any(math.isnan(c) for c in costs.values()):
            failures.append(f"{key}: missing or failed strategy, have {sorted(costs)}")
            continue
        for cheaper, dearer in zip(ORDERING, ORDERING[1:]):
            if costs[cheaper] > costs[dearer] + COST_TOL_SGD:
                failures.append(f"{key}: {cheaper}={costs[cheaper]:.4f} > "
                                f"{dearer}={costs[dearer]:.4f}")
        milp = case["milp"]
        if milp.get("status") != "Optimal":
            failures.append(f"{key}: MILP status {milp.get('status')}")
        residual = milp.get("residual_sgd")
        if residual is None or not abs(residual) <= MILP_RESIDUAL_TOL_SGD:
            failures.append(f"{key}: MILP replay residual {residual}")
        for strategy, row in case.items():
            rate = row.get("departure_success_rate")
            if rate is not None and rate < 1.0:
                failures.append(f"{key}: {strategy} departure success {rate}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-slots", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs=2, default=(100, 119))
    parser.add_argument("--out", default="results")
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--gap-rel", type=float, default=1e-6)
    args = parser.parse_args()
    seeds = range(args.seeds[0], args.seeds[1] + 1)
    path, rows = run_grid(n_slots=args.n_slots, seeds=seeds, out_dir=args.out,
                          time_limit=args.time_limit, gap_rel=args.gap_rel)
    n_cases = len({(row["date"], row["seed"]) for row in rows})
    failures = check_acceptance(rows)
    if failures:
        print(f"acceptance FAILED: {len(failures)} problem(s) over {n_cases} cases")
        for line in failures[:20]:
            print(f"  {line}")
        raise SystemExit(1)
    worst = max(abs(row["residual_sgd"]) for row in rows if row["strategy"] == "milp")
    print(f"acceptance OK over {n_cases} cases: milp <= price_threshold <= arrival_charge "
          f"on every case; every MILP Optimal, max |replay residual| = {worst:.2e} S$")


if __name__ == "__main__":
    main()
