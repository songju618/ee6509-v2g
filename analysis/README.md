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
| `run_all.py` | Enumerates the test grid (7 days × 20 seeds × N=5), runs `no_storage`, `arrival_charge`, `price_threshold`, `milp` per case, writes `results/summary.csv` and `results/diagnostics.csv`. |
| `compare.py` | Reads `results/summary.csv`, writes `results/figures/strategy_summary.csv` (mean per strategy) and `results/figures/mean_total_cost.png` (bar chart for G's report). |

### Usage

```powershell
python -m analysis.run_all --n-slots 5 --seeds 100 119 --time-limit 300 --gap-rel 1e-6
python -m analysis.compare