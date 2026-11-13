# Baselines (C)

Two rule-based strategies, both written to run through A's `env.V2GEnv` so their costs
are settled by the same `settle_cost()` as the MILP and by SAC during training.

## Files

| File | Strategy | What it does |
|---|---|---|
| `arrival_charge.py` | `arrival_charge` | Technical plan 无储能 / 到场就充: charges each present EV at the maximum feasible rate (`min(6 kW, env.hi)`) until full. BESS idle. |
| `price_threshold.py` | `price_threshold` | Technical plan 规则策略: fixed USEP thresholds, BESS 低价充、高价放, EVs charged by departure order, never raising the grid peak by choice. Details below. |
| `common.py` | — | Scenario loading, test-grid generator, presence mask, training-split USEP reader, replay + CSV writer. |

Both baselines build a `(T, N+1)` kW power schedule and hand it to
`env.replay.replay_power_schedule`, so power clipping, energy updates, departure
checks and cost accounting are exactly A's. No baseline reimplements physics.

## `price_threshold` rule

Decisions at step t use only what the env observes at t.

1. **Peak guard.** Charging by choice may raise grid import only up to
   `max(240 kW contract, running peak)`, so the rule never adds an excess-capacity fee
   (0.71 S$/kW/day) by itself.
2. **EVs.** Take the minimum the departure projection forces, then charge toward the
   65 % departure target (not to full), earliest departure first, within the peak guard.
   EVs never discharge.
3. **BESS.** Charge when USEP is below the low threshold, within the peak guard. Charge
   above the initial 60 kWh only if the env's 6-hour USEP forecast shows a price above the
   high threshold; otherwise only refill what was discharged. Discharge at the maximum
   feasible rate when USEP is above the high threshold. Idle otherwise.

Thresholds are the 20 % and 80 % quantiles of training-split USEP
(`params["data"]["first_date"] .. train_last_date`): 0.1893 and 0.4799 S$/kWh.
The rule was designed on the validation split (2026-10-15 .. 10-21, seeds 0–19), where it
is cheaper than `arrival_charge` on all 140 cases for N=5 and for N=10.

## Usage

```powershell
python -m baselines.arrival_charge --scenario results/env_demo/random_2026-10-01_seed0_n5.scenario.json
python -m baselines.price_threshold --scenario results/env_demo/random_2026-10-01_seed0_n5.scenario.json
```
