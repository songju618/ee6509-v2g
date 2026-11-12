"""Run no_storage, arrival_charge, price_threshold, and MILP over the test grid."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from baselines.arrival_charge import run as run_arrival_charge
from baselines.common import replay, summarize, test_grid, write_strategy_csv
from baselines.price_threshold import run as run_price_threshold
from env import V2GEnv, load_params
from milp.solve import run as run_milp


STRATEGIES = ("no_storage", "arrival_charge", "price_threshold", "milp")

SUMMARY_KEYS = [
    "strategy", "date", "seed", "n_slots",
    "variable_cost_sgd", "fixed_pv_sgd", "fixed_contract_sgd", "total_cost_sgd",
    "peak_grid_kw", "bess_final_kwh", "departure_success_rate",
    "csv_path",
    "status", "milp_objective", "replayed_variable_cost", "residual_sgd",
]


def run_no_storage(scenario, out_dir, params=None):
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
    csv_path = Path(out_dir) / f"no_storage_{date}_seed{seed}_n{N}.csv"
    write_strategy_csv(csv_path, infos, "no_storage")
    return summarize(date, seed, "no_storage", summary, csv_path)


def run_grid(n_slots=5, seeds=range(100, 120), out_dir="results",
             time_limit=300, gap_rel=1e-6):
    out_root = Path(out_dir)
    rows = []
    for date, seed, scenario in test_grid(n_slots=n_slots, seeds=seeds):
        print(f"[{date} seed={seed} N={n_slots}]", flush=True)
        rows.append(run_no_storage(scenario, out_root / "baselines"))
        rows.append(run_arrival_charge(scenario, out_root / "baselines"))
        rows.append(run_price_threshold(scenario, out_root / "baselines",
                                        low_q=0.2, high_q=0.8, ev_charge_cap_kw=3.0))
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
        _write_csv(out_root / "summary.csv", rows, SUMMARY_KEYS)
        _write_csv(out_root / "diagnostics.csv", rows,
                   ["strategy", "date", "seed", "low_q", "high_q", "usep_low", "usep_high",
                    "ev_charge_cap_kw", "error"], extrasaction="ignore")
    print(f"wrote {out_root / 'summary.csv'} ({len(rows)} rows)")
    return out_root / "summary.csv", rows


def _write_csv(path, rows, fieldnames, extrasaction="ignore"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        writer.writeheader()
        writer.writerows(rows)


def check_ordering(rows):
    """Per (date, seed, n_slots): no_storage >= arrival_charge >= price_threshold >= milp (cost)."""
    by_case = {}
    for row in rows:
        if row["variable_cost_sgd"] != row["variable_cost_sgd"]:  # NaN check
            continue
        key = (row["date"], row["seed"], row["n_slots"])
        by_case.setdefault(key, {})[row["strategy"]] = row["total_cost_sgd"]
    violations = []
    for key, costs in by_case.items():
        for a, b in (("no_storage", "arrival_charge"),
                     ("arrival_charge", "price_threshold"),
                     ("price_threshold", "milp")):
            if a in costs and b in costs and costs[b] > costs[a] + 1e-6:
                violations.append((key, a, b, costs[a], costs[b]))
    return violations


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
    violations = check_ordering(rows)
    if violations:
        print(f"{len(violations)} ordering violations")
        for key, a, b, ca, cb in violations[:10]:
            print(f"  {key}: {a}={ca:.4f} -> {b}={cb:.4f}")
    else:
        print("ordering OK: no_storage >= arrival_charge >= price_threshold >= milp on all cases")


if __name__ == "__main__":
    main()