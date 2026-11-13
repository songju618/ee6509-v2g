# Analysis

Shared results directory for C (unified runner), B (plots), and G (analysis).
The three roles work from the same CSVs so the report is reproducible end to end.

| Owner | Responsibility | Entry point |
|---|---|---|
| **C** | Enumerate every strategy × test day × test seed, produce the per-case CSV schema fixed in the root README, aggregate means. | `run_all.py`, `compare.py` |
| **B** | Read C's CSVs and produce the technical-plan §6.4 figures: typical-day power/SoC curves, cost breakdown, attention weights. | `plot_*.py` (added by B) |
| **G** | Read the same CSVs and write the results/discussion section. | `analysis/notes/*.md` (added by G) |

## C — unified runner

### Files

| File | Purpose |
|---|---|
| `run_all.py` | Enumerates the test grid (7 days × 20 seeds, one N per run), runs `arrival_charge`, `price_threshold`, `milp` and the extra reference `late_charge` per case, writes `results/summary_n{N}.csv` and `results/diagnostics_n{N}.csv`, then checks C's acceptance and exits non-zero if it fails. |
| `compare.py` | Reads every `results/summary_n*.csv`, writes `results/figures/strategy_summary.csv` (mean per N and strategy) and `results/figures/mean_total_cost.png` (bar chart for G's report). |

### Strategies

| Name in the CSVs | Technical plan §6.1 | What it does |
|---|---|---|
| `arrival_charge` | 无储能 / 到场就充, cost upper bound | BESS idle; each EV charged at max power until full. |
| `price_threshold` | 规则策略（低价充、高价放） | Fixed USEP thresholds, peak-aware; see `baselines/README.md`. |
| `milp` | MILP 离线最优, cost lower bound | Solved on the whole day's true data, then replayed through A's env. |
| `late_charge` | not a plan baseline | BESS idle; each EV gets only what its departure constraint forces. Formerly `no_storage`. |

### Acceptance (root README, row C)

`run_all` checks it on every run and exits with code 1 if it fails:

- on every case, `milp <= price_threshold <= arrival_charge` (variable cost);
- every MILP case is `Optimal`, and its replayed cost is within 0.01 S$ of the MILP objective.

### Usage

```powershell
python -m analysis.run_all --n-slots 5 --seeds 100 119 --time-limit 300 --gap-rel 1e-6
python -m analysis.run_all --n-slots 10 --seeds 100 119 --time-limit 300 --gap-rel 1e-6
python -m analysis.compare
```
