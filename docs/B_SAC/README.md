# Part B（SAC 调度与绘图）交付材料总索引

本目录集中汇总了 **B 部分（SAC 强化学习调度与可视化）** 的全部定稿交付物。组内成员 `git pull` 后直接按下方角色索引取用即可，无需翻阅训练源码目录。

所有策略对比都使用技术方案 §6.1 的正式基线（由 C 实现）：到场就充（无储能，成本上界）、低价充高价放（规则策略）、MILP 离线最优（成本下界）。

---

## 按分工角色快速取用

### 1. 给 C 同学（统一运行入口与基线联调）
* **定稿模型权重**：
  * `N = 5`：[models/final_n5.pt](models/final_n5.pt)（72k 步验证集最优）
  * `N = 10`：[models/final_n10.pt](models/final_n10.pt)（120k 步验证集最优）
* **调用说明与 5 行接入代码**：[C_integration_guide.md](C_integration_guide.md)
* **与 C 结果的核对**：`python -m sac.official` 在同一批场景上运行 C 的两个规则基线，并在 `results/summary_n{N}.csv` 存在时逐场景核对；当前到场就充与规则策略逐场景一致，MILP 与 B 的离线最优相差不超过 0.0001 SGD。

### 2. 给 F 同学（实验测试、灵敏度分析与合规检查）
* **验收与合规审计摘要**：[tables/acceptance_audit.json](tables/acceptance_audit.json)（标准测试集 1,500 次离场与扩展场景 4,912 次离场 **100% 达标**、BESS 日末约束 **100% 达标**、48 行 CSV 逐步成本与日汇总误差 `< 1e-8 SGD`、离线回放最大误差 `0.000123 SGD`）。
* **正式对比表（可直接用 Excel / Pandas 打开）**：
  * 标准测试集各策略均值、相对两个基线的节省与两种 95% 区间：[tables/official_test_summary.csv](tables/official_test_summary.csv)
  * 逐测试日的“基线减 SAC”差值：[tables/official_daily_differences.csv](tables/official_daily_differences.csv)
  * `0~10` 台车阶梯与日夜混合场景：[tables/official_extended_summary.csv](tables/official_extended_summary.csv)
* **训练与消融**：[tables/ablation_and_steps.csv](tables/ablation_and_steps.csv)
* **B 内部参照表（仅用于模型选择，不是正式基线）**：`test_summary_main.csv`、`validation_summary.csv`、`extended_0_to_10_summary.csv` 及各 `*_comparison.json` 中的 `immediate`（到场即充到 65% 目标）、`rule`（B 自写的简单阈值规则）、`random` 是训练阶段的内部对照，口径与正式基线不同。
* **如需重跑任意场景导出逐半小时 48 行标准 CSV**：直接执行 `sac.evaluate`（见 [C_integration_guide.md](C_integration_guide.md)，全测试集 140 个场景约 6 秒）。

### 3. 给 G 同学（结果分析与 5 页报告整合）
* **报告正文素材与分析初稿**：[B_SAC_report_draft.md](B_SAC_report_draft.md)（网络与训练设计、消融、标准测试与泛化结果、机理与局限、主图与图注）。
* **全套图表（均含 180 DPI `.png` 与矢量 `.pdf`）**：位于 [figures/](figures/) 目录：
  * `network_architecture_en.{png,pdf}`：SAC 网络架构图英文版（报告主图 1；中文原图为 `docs/network_architecture.png`）
  * `gap_position.{png,pdf}`：各策略在 MILP（0%）与到场就充（100%）之间的位置，及 SAC 逐场景更便宜的比例
  * `daily_differences.{png,pdf}`：各测试日“基线减 SAC”的成本差，含按天重抽的 95% 区间
  * `cost_components.{png,pdf}`：各策略日均可变成本及五项分项
  * `price_regime_cost_components.{png,pdf}`：平缓日与高波动日扣除纯楼宇账单后的增量分解
  * `extended_0_to_10_comparison.{png,pdf}`：`0~10` 台车与日夜混合场景下各策略成本及 SAC 填补的差距
  * `power_price_n{5,10}_2026-10-28.{png,pdf}` / `soc_n{5,10}_2026-10-28.{png,pdf}`：尖峰日（seed 100）的功率、电价与 SoC 轨迹
  * `power_price_n{5,10}_2026-10-22.{png,pdf}` / `soc_n{5,10}_2026-10-22.{png,pdf}`：平缓日（seed 100）的同类图，用于局限性讨论
  * `training_curves.{png,pdf}`：训练与固定验证网格收敛曲线（`N=5, 10`）
  * `attention_dwell_n5.{png,pdf}` / `attention_dwell_n10.{png,pdf}`：注意力权重与剩余停留步数（解读见报告第四节）
  * `extended_occupancy.{png,pdf}`：扩展场景的平均在场车数曲线

---

## 目录结构一览

```text
docs/B_SAC/
├── README.md                        # 本索引说明
├── B_SAC_report_draft.md            # 给 G/F 的报告章节文字素材、定量表格与图注说明
├── C_integration_guide.md           # 给 C/F 的模型加载接口与评估/重绘说明
├── models/                          # 定稿模型权重（含训练期冻结的标准化统计量）
│   ├── final_n5.pt
│   └── final_n10.pt
├── tables/                          # 正式对比表（official_*）、消融表与 B 内部参照表
├── training_logs/                   # N=5 与 N=10 完整训练过程记录（JSONL）
└── figures/                         # 全部结果图表的 PNG 与矢量 PDF 版本
```

## 重新生成

```bash
python -m analysis.run_all --n-slots 5 --seeds 100 119
python -m analysis.run_all --n-slots 10 --seeds 100 119
python -m sac.official
python -m analysis.plotting.plot_official --figures-dir docs/B_SAC/figures --tables-dir docs/B_SAC/tables
python -m analysis.plotting.plot_sac --figures-dir docs/B_SAC/figures
python -m analysis.plotting.plot_extended --figures-dir docs/B_SAC/figures
python -m analysis.plotting.plot_architecture --figures-dir docs/B_SAC/figures
```

前两行是 C 的统一入口（需要 `milp/requirements.txt` 里的 pulp），只用于逐场景核对，跳过也能生成全部图表。
