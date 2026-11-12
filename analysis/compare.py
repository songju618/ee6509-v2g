"""Aggregate results/summary.csv into per-strategy tables and plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    agg = (df.groupby("strategy")
             .agg(mean_variable_cost=("variable_cost_sgd", "mean"),
                  mean_total_cost=("total_cost_sgd", "mean"),
                  mean_peak_kw=("peak_grid_kw", "mean"),
                  mean_departure_success=("departure_success_rate", "mean"),
                  max_milp_residual=("residual_sgd", "max"))
             .sort_values("mean_total_cost"))
    agg.to_csv(out / "strategy_summary.csv")
    print(agg.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(agg.index, agg["mean_total_cost"], color="steelblue")
    ax.set_ylabel("mean total cost (SGD/day)")
    ax.set_title("Strategy mean daily cost over test grid")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "mean_total_cost.png", dpi=150)
    print(f"wrote {out / 'mean_total_cost.png'}")


if __name__ == "__main__":
    main()