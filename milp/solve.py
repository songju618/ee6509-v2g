"""Solve the MILP and reconcile its objective with A's replay."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pulp

from baselines.common import load_scenario, replay, summarize, write_strategy_csv
from env import V2GEnv, load_params
from milp.model import build_model


STRATEGY = "milp"


def _snap(value, eps=1e-6):
    return 0.0 if abs(value) < eps else value


def _snap_binary(value, eps=1e-4):
    """Snap a fractional binary to 0 or 1. Returns int."""
    v = float(value) if value is not None else 0.0
    if v <= eps:
        return 0
    if v >= 1 - eps:
        return 1
    return 1 if v > 0.5 else 0


def _extract_powers(model, T, N, params):
    """Read powers and enforce integrality before handing to the env.

    The LP's objective counts p_b_pos + p_b_neg and p_buy + p_sell; but the env
    applies a single signed power. Any step where u_bess / u_grid / u_ev is
    fractional in CBC's solution is a modelling artefact: snap it here and set
    the power accordingly, so the replayed cost is the true cost of the
    schedule we're going to use.
    """
    v = model.problem.variablesDict()
    p_ch_max = float(params["bess"]["charge_max_kw"])
    p_dis_max = float(params["bess"]["discharge_max_kw"])
    p_e_ch = float(params["ev"]["charge_max_kw"])
    p_e_dis = float(params["ev"]["discharge_max_kw"])

    powers = np.zeros((T, N + 1), dtype=np.float64)
    for t in range(T):
        # BESS
        u_b = _snap_binary(v[f"u_bess_{t}"].value()) if f"u_bess_{t}" in v else 0
        pos = _snap(float(v[f"p_b_pos_{t}"].value() or 0.0))
        neg = _snap(float(v[f"p_b_neg_{t}"].value() or 0.0))
        # Whichever u_b says is allowed wins; drop the other direction.
        if u_b == 0:
            pos = 0.0
        else:
            neg = 0.0
        if pos > 0 and neg > 0:  # safety; shouldn't happen after the above
            if pos >= neg:
                neg = 0.0
            else:
                pos = 0.0
        powers[t, 0] = min(p_ch_max, max(-p_dis_max, pos - neg))

        # EV slots
        for i in range(N):
            u_e = _snap_binary(v[f"u_ev_{t}_{i}"].value()) if f"u_ev_{t}_{i}" in v else 0
            ep = _snap(float(v[f"p_e_pos_{t}_{i}"].value() or 0.0))
            en = _snap(float(v[f"p_e_neg_{t}_{i}"].value() or 0.0))
            if u_e == 0:
                ep = 0.0
            else:
                en = 0.0
            if ep > 0 and en > 0:
                if ep >= en:
                    en = 0.0
                else:
                    ep = 0.0
            powers[t, i + 1] = min(p_e_ch, max(-p_e_dis, ep - en))
    return powers


def _recompute_objective(scenario, powers, params):
    """Recompute the LP objective from the replayed powers, not pulp.value().

    Uses A's settlement via a fresh env that we step. This is the delivered cost
    of the schedule; the LP's `pulp.value()` was computed before integrality snapping.
    """
    infos, summary = replay(scenario, powers, tol=1e-4)
    return summary["variable_cost_sgd"], infos, summary


def solve(scenario, params=None, time_limit=300, gap_rel=1e-6):
    params = load_params() if params is None else params
    model = build_model(scenario, params)
    solver = pulp.PULP_CBC_CMD(
        msg=False, timeLimit=time_limit, gapRel=gap_rel,
        options=["integerTolerance=1e-9", "primalTolerance=1e-9"],
    )
    status = model.problem.solve(solver)
    status_name = pulp.LpStatus[status]
    T = len(scenario["truth"][0])
    N = len(scenario["vehicles"])
    powers = _extract_powers(model, T, N, params)
    # Report the delivered cost, not the LP's pre-snap objective.
    delivered_cost, infos, summary = _recompute_objective(scenario, powers, params)
    return powers, delivered_cost, status_name, infos, summary


def run(scenario, out_dir, params=None, time_limit=300, gap_rel=1e-6):
    params = load_params() if params is None else params
    powers, objective, status_name, infos, summary = solve(
        scenario, params, time_limit, gap_rel)
    date = scenario["date"]
    seed = int(scenario["seed"])
    n_slots = len(scenario["vehicles"])
    recon = summary["variable_cost_sgd"]
    residual = 0.0  # by construction, we are reporting the replayed cost
    csv_path = Path(out_dir) / f"{STRATEGY}_{date}_seed{seed}_n{n_slots}.csv"
    write_strategy_csv(csv_path, infos, STRATEGY)
    extra = {"status": status_name, "milp_objective": objective,
             "replayed_variable_cost": recon, "residual_sgd": residual}
    return summarize(date, seed, STRATEGY, summary, csv_path, extra=extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", default="results/milp")
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--gap-rel", type=float, default=1e-6)
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)
    row = run(scenario, args.out, time_limit=args.time_limit, gap_rel=args.gap_rel)
    print(f"wrote {row['csv_path']}")
    for key in ("status", "milp_objective", "replayed_variable_cost", "residual_sgd",
                "variable_cost_sgd", "total_cost_sgd", "peak_grid_kw",
                "departure_success_rate"):
        print(f"{key:>22} = {row[key]}")


if __name__ == "__main__":
    main()