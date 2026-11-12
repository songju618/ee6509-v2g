# Baselines (C)

Two rule-based strategies, both written to run through A's `env.V2GEnv` so their costs
are settled by the same `settle_cost()` as the MILP and by SAC during training.

## Files

| File | Strategy | What it does |
|---|---|---|
| `arrival_charge.py` | `arrival_charge` | Charges each present EV at the maximum feasible rate (`min(6 kW, env.hi)`) for its whole stay. BESS idle. |
| `price_threshold.py` | `price_threshold` | Charges BESS and present EVs at `env.hi` when `USEP < usep_low`, discharges at `env.lo` when `USEP > usep_high`, zero otherwise. |
| `common.py` | — | Scenario loading, test-grid generator, presence mask, training-split USEP reader, replay + CSV writer. |

Both baselines build a `(T, N+1)` kW power schedule and hand it to
`env.replay.replay_power_schedule`, so power clipping, energy updates, departure
checks and cost accounting are exactly A's. No baseline reimplements physics.

## Tuning

`price_threshold` uses training-split USEP quantiles only
(`params["data"]["first_date"] .. train_last_date`). The grid run in
`analysis/run_all.py` uses `low_q=0.2, high_q=0.8, ev_charge_cap_kw=3.0`, tuned on
2026-10-01 only. See `milp/README.md` for the caveat that this does not beat
`arrival_charge` per-case on 60/140 test days.

## Usage

```powershell
python -m baselines.arrival_charge --scenario results/env_demo/random_2026-10-01_seed0_n5.scenario.json
python -m baselines.price_threshold --scenario results/env_demo/random_2026-10-01_seed0_n5.scenario.json --low-q 0.2 --high-q 0.8 --ev-charge-cap-kw 3