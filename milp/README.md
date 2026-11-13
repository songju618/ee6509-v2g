# MILP offline reference (C)

Solver: PuLP 2.9.0 + bundled CBC.

## Model

`milp/model.py` builds a mixed-integer linear program matching D §4.1/4.2:

- Rated power, SoC bounds, EV departure reachability
  `E_{t+1} >= target − eta_ch * P_ch,max * (k−1) * dt`, BESS end-of-day rule `E_T >= E_0`.
- Per-device binary mode variables (`u_bess`, `u_ev`) prevent simultaneous charge/discharge.
- Per-step grid buy/sell split with mutual exclusion (`u_grid`).
- **Single peak variable** `peak >= p_grid[t]` for all t, `peak >= contract`; objective term
  `ucc_daily * (peak − contract)` charged **once**, matching A's `settle_cost` incremental rule.

`milp/solve.py` extracts the signed power schedule from the solution (snapping `u_bess`,
`u_ev` binaries to 0/1), then replays the schedule through `env.V2GEnv.step_power(tol=1e-4)`.
Each summary row keeps three numbers: `milp_objective` (the solver's objective value),
`replayed_variable_cost` (A's settlement of the snapped schedule, also reported as
`variable_cost_sgd`), and `residual_sgd` = replayed − objective.

## Test grid

7 test days (2026-10-22 .. 2026-10-28) × 20 seeds (100–119), for N=5 and N=10.
Settings: `--time-limit 300 --gap-rel 1e-6`. One full grid takes about 10–15 s.

## Results (mean over 140 cases)

| N | strategy | mean variable cost (S$/day) | mean total cost (S$/day) | mean peak (kW) | departure success |
|---|---|---:|---:|---:|---:|
| 5 | **MILP (offline)** | **1172.54** | **1298.05** | **242.40** | 1.0 |
| 5 | price_threshold | 1234.57 | 1360.08 | 245.36 | 1.0 |
| 5 | late_charge | 1250.38 | 1375.89 | 245.89 | 1.0 |
| 5 | arrival_charge | 1260.09 | 1385.61 | 246.59 | 1.0 |
| 10 | **MILP (offline)** | **1169.33** | **1294.85** | **245.52** | 1.0 |
| 10 | price_threshold | 1246.29 | 1371.81 | 246.02 | 1.0 |
| 10 | late_charge | 1267.37 | 1392.89 | 251.72 | 1.0 |
| 10 | arrival_charge | 1295.28 | 1420.80 | 264.24 | 1.0 |

MILP saves 87.6 S$/day (N=5) and 126.0 S$/day (N=10) against `arrival_charge`, and
62.0 / 77.0 S$/day against `price_threshold`.

## Acceptance (root README, row C)

`analysis.run_all` checks it on every run. For N=5 and N=10, 140 cases each:

- **Ordering:** `milp <= price_threshold <= arrival_charge` on 140/140 cases.
- **Replay reconciliation:** every case is `Optimal`; the largest
  |replayed cost − MILP objective| is 1.5e-05 S$ (N=5) and 1.7e-05 S$ (N=10), against
  the 0.01 S$ limit.

## Notes

- `arrival_charge` is the technical plan's 无储能 upper bound.
- `late_charge` (formerly `no_storage`) is an extra reference, not a plan baseline. It buys
  only the energy the departure targets require, as late as possible, so it is cheaper than
  `arrival_charge`.
- How `price_threshold` decides is described in `baselines/README.md`.

## Artifacts

- `results/summary_n{N}.csv` — 560 rows per N, one per (strategy, date, seed).
- `results/diagnostics_n{N}.csv` — `price_threshold` threshold values and MILP errors.
- `results/baselines/*.csv`, `results/milp/*.csv` — per-case step CSVs (A's canonical schema).
- `results/figures/strategy_summary.csv`, `results/figures/mean_total_cost.png`.

## Reproducing

```powershell
python -m analysis.run_all --n-slots 5 --seeds 100 119 --time-limit 300 --gap-rel 1e-6
python -m analysis.run_all --n-slots 10 --seeds 100 119 --time-limit 300 --gap-rel 1e-6
python -m analysis.compare
```
