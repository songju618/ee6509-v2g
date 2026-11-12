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

`milp/solve.py` extracts the signed power schedule from the LP solution (snapping `u_bess`,
`u_ev` binaries to 0/1), then replays the schedule through `env.V2GEnv.step_power(tol=1e-4)`.
The delivered cost reported in `results/summary.csv` is the replayed cost, not
`pulp.value(model.objective)`; both are stored for cross-check.

## Test grid

7 test days (2026-10-22 .. 2026-10-28) × 20 seeds (100–119) × N=5.
Settings: `--time-limit 300 --gap-rel 1e-6`.

## Results (mean over 140 cases)

| strategy | mean variable cost (S$/day) | mean total cost (S$/day) | mean peak (kW) | departure success |
|---|---:|---:|---:|---:|
| **MILP (offline)** | **1172.54** | **1298.05** | **242.40** | 1.0 |
| no_storage | 1250.38 | 1375.89 | 245.89 | 1.0 |
| price_threshold | 1258.32 | 1383.84 | 270.30 | 1.0 |
| arrival_charge | 1260.09 | 1385.61 | 246.59 | 1.0 |

MILP saves **77.8 S$/day (5.7%)** vs arrival_charge, **85.8 S$/day** vs price_threshold,
**77.8 S$/day** vs no_storage. Mean peak reduces from ~246 kW (baselines) to 242.4 kW,
close to the 240 kW contract threshold.

## Replay reconciliation

Every MILP case was replayed through `env.V2GEnv.step_power(tol=1e-4)` and `step_cost_sgd`
summed across 48 steps.

- **140/140 cases**: `status = Optimal`, replay residual `<= 0.01 S$` (`max = 0`).
- **140/140 cases**: `MILP <= arrival_charge`, `MILP <= no_storage`, `MILP <= price_threshold`.

## Baseline caveats for the report

- **`no_storage`** charges EVs only the minimum required to meet departure targets, so it
  buys much less energy than `arrival_charge`. It is a *passive lower bound*, not a fair
  "worst case" and not a baseline SAC should beat on cost.
- **`price_threshold`** was tuned on 2026-10-01 only (`low_q=0.2, high_q=0.8, EV charge cap
  3 kW`). It beats `arrival_charge` on 80/140 cases and loses on 60 because training-split
  USEP thresholds don't always bracket test-day prices. The mean ranking holds; per-case it
  does not. Its mean peak (270.3 kW) is significantly higher than the other strategies
  because the rule discharges BESS at high USEP without regard to the current running peak.
- **`arrival_charge`** is the intended "charge everything on arrival" upper bound; it
  maintains the same mean peak as `no_storage` (~246 kW) but pays more for energy.

## Artifacts

- `results/summary.csv` — 560 rows, one per (strategy, date, seed, N=5).
- `results/diagnostics.csv` — `price_threshold` threshold values and MILP status.
- `results/baselines/*.csv`, `results/milp/*.csv` — per-case step CSVs (A's canonical schema).
- `results/figures/strategy_summary.csv`, `results/figures/mean_total_cost.png`.

## Reproducing

```powershell
python -m analysis.run_all --n-slots 5 --seeds 100 119 --time-limit 300 --gap-rel 1e-6
python -m analysis.compare