"""Shared plumbing for C's baselines and MILP: scenario I/O, grid, CSV, summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from env import V2GEnv, load_params
from env.data import load_day, split_dates
from env.recording import csv_columns, info_to_row
from env.replay import replay_power_schedule


# ---------- scenario I/O ----------

def load_scenario(path):
    """Read a saved .scenario.json produced by env.export_scenario()."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_scenario(path, scenario):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenario, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


# ---------- test grid ----------

def test_grid(n_slots=5, seeds=range(100, 120), params=None):
    """Yield (date, seed, scenario_dict) for the 7 x 20 test grid."""
    params = load_params() if params is None else params
    dates = split_dates(params, "test")
    env = V2GEnv(n_slots=n_slots, params=params, split="test")
    try:
        for date in dates:
            for seed in seeds:
                env.reset(seed=seed, options={"date": date})
                yield date, seed, env.export_scenario()
    finally:
        env.close()


# ---------- presence mask from the scenario ----------

def present_mask(scenario):
    """(T, N) bool: slot i is occupied at step t when arrival <= t < departure."""
    T = len(scenario["truth"][0])
    vehicles = scenario["vehicles"]
    mask = np.zeros((T, len(vehicles)), dtype=bool)
    for i, v in enumerate(vehicles):
        if v is not None:
            mask[v["arrival"]:v["departure"], i] = True
    return mask


# ---------- training-split USEP quantiles ----------

def train_usep(params=None):
    params = load_params() if params is None else params
    values = []
    for date in split_dates(params, "train"):
        values.append(load_day(date)["usep"])
    return np.concatenate(values)


# ---------- replay + CSV ----------

def replay(scenario, powers, tol=1e-6):
    """Replay (T, N+1) kW schedule through A's env. Returns (infos, summary)."""
    powers = np.asarray(powers, dtype=np.float64)
    return replay_power_schedule(scenario, powers, tol=tol)


def write_strategy_csv(path, infos, strategy):
    """One trajectory -> one CSV, using A's canonical columns and row order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    infos = list(infos)
    if not infos:
        raise ValueError("cannot write an empty trajectory")
    first = infos[0]
    n_slots = len(first["p_ev_kw"])
    columns = csv_columns(n_slots)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(info_to_row(info, strategy=strategy) for info in infos)


def summarize(date, seed, strategy, summary, csv_path=None, extra=None):
    row = {
        "strategy": strategy,
        "date": date,
        "seed": seed,
        "n_slots": len(summary["departures"]) if summary["departures"] else None,
        "variable_cost_sgd": summary["variable_cost_sgd"],
        "fixed_pv_sgd": summary["fixed_pv_sgd"],
        "fixed_contract_sgd": summary["fixed_contract_sgd"],
        "total_cost_sgd": summary["total_cost_sgd"],
        "peak_grid_kw": summary["peak_grid_kw"],
        "bess_final_kwh": summary["bess_final_kwh"],
        "departure_success_rate": summary["departure_success_rate"],
        "csv_path": str(csv_path) if csv_path else "",
    }
    if extra:
        row.update(extra)
    return row