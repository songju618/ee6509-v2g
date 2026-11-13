"""Aggregate run_all summaries (one file per N) into a per-strategy table and a bar chart."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ("arrival_charge", "price_threshold", "late_charge", "milp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", nargs="+", default=None,
                        help="summary CSVs from analysis.run_all (default: results/summary_n*.csv)")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    paths = args.summary or sorted(Path("results").glob("summary_n*.csv"))
    if not paths:
        raise SystemExit("no summary files found; run python -m analysis.run_all first")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    agg = (df.groupby(["n_slots", "strategy"])
             .agg(cases=("variable_cost_sgd", "size"),
                  mean_variable_cost=("variable_cost_sgd", "mean"),
                  mean_total_cost=("total_cost_sgd", "mean"),
                  mean_peak_kw=("peak_grid_kw", "mean"),
                  mean_departure_success=("departure_success_rate", "mean"),
                  max_abs_milp_residual=("residual_sgd", lambda s: s.abs().max()))
             .reset_index()
             .sort_values(["n_slots", "mean_total_cost"]))
    agg.to_csv(out / "strategy_summary.csv", index=False)
    print(agg.to_string(index=False))

    strategies = [s for s in ORDER if s in set(agg["strategy"])]
    strategies += sorted(set(agg["strategy"]) - set(strategies))
    sizes = sorted(agg["n_slots"].unique())
    width = 0.8 / len(sizes)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, n in enumerate(sizes):
        values = (agg[agg["n_slots"] == n].set_index("strategy")
                  .reindex(strategies)["mean_total_cost"])
        x = np.arange(len(strategies)) + (k - (len(sizes) - 1) / 2) * width
        ax.bar(x, values, width, label=f"N={n}")
    ax.set_xticks(np.arange(len(strategies)), strategies)
    ax.set_ylabel("mean total cost (SGD/day)")
    ax.set_title("Strategy mean daily cost over test grid")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out / "mean_total_cost.png", dpi=150)
    print(f"wrote {out / 'mean_total_cost.png'}")


if __name__ == "__main__":
    main()
