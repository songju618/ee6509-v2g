# Part B（SAC 调度与绘图）交付材料总索引

本目录集中汇总了 **B 部分（SAC 强化学习调度与可视化）** 的全部定稿交付物。组内成员 `git pull` 后直接按下方角色索引取用即可，无需翻阅训练源码目录。

---

## 按分工角色快速取用

### 1. 给 C 同学（统一运行入口与基线联调）
* **定稿模型权重**：
  * `N = 5`：[models/final_n5.pt](models/final_n5.pt)（72k 步验证集最优）
  * `N = 10`：[models/final_n10.pt](models/final_n10.pt)（120k 步验证集最优）
* **调用说明与 5 行接入代码**：[C_integration_guide.md](C_integration_guide.md)
* **核账参考数据**：[tables/test_summary_main.csv](tables/test_summary_main.csv)（含 B 侧自验算的 `immediate` 与 `offline_opt` 结果，可供你完成 `baselines/` 与 `milp/` 后交叉核对；正式报告中的 `rule` 阈值策略以你产出的统一 CSV 为准）。

### 2. 给 F 同学（实验测试、灵敏度分析与合规检查）
* **验收与合规审计摘要**：[tables/acceptance_audit.json](tables/acceptance_audit.json)（记录标准测试集 1,500 次离场与扩展场景 4,912 次离场 **100% 达标**、BESS 日末约束 **100% 达标**、48 行 CSV 逐步成本与日汇总误差 `< 1e-8 SGD`、离线回放最大误差 `0.000123 SGD`）。
* **汇总结果表（可直接用 Excel / Pandas 打开）**：
  * 标准测试集汇总（`N=5, 10`）：[tables/test_summary_main.csv](tables/test_summary_main.csv)
  * 验证集汇总（`N=5, 10`）：[tables/validation_summary.csv](tables/validation_summary.csv)
  * 训练步数与消融对比：[tables/ablation_and_steps.csv](tables/ablation_and_steps.csv)
  * `0~10` 台车阶梯与日夜混合灵敏度实验：[tables/extended_0_to_10_summary.csv](tables/extended_0_to_10_summary.csv)
* **如需重跑任意场景导出逐半小时 48 行标准 CSV**：直接执行 `sac.evaluate`（见 [C_integration_guide.md](C_integration_guide.md)，单次全测试集 140 个场景约 3 秒跑完；如需 B 本地已导出的全量原始 CSV 压缩包也可随时直接找我取）。

### 3. 给 G 同学（结果分析与 5 页报告整合）
* **报告正文素材与分析初稿（可直接摘录合入主报告）**：[B_SAC_report_draft.md](B_SAC_report_draft.md)
  * 已涵盖：网络与训练设计要点、超参数与消融结论、标准测试定量对比表（绝对成本分项 + 可优化空间捕获率百分比）、SAC 学到的物理策略与注意力可解释性、平缓日表现与单日边界局限说明，以及 5 页报告的 **4 张主图选图建议与图注**。
* **全套高清图表（均含 180 DPI `.png` 与矢量 `.pdf`）**：位于 [figures/](figures/) 目录：
  * `training_curves.{png,pdf}`：训练与固定验证网格收敛曲线（`N=5, 10`）
  * `power_price_n5.{png,pdf}` / `power_price_n10.{png,pdf}`：典型日（`2026-10-22, seed=100`）各设备功率与购电价叠加曲线
  * `soc_n5.{png,pdf}` / `soc_n10.{png,pdf}`：典型日 BESS 与在场 EV SoC 轨迹（已用 `occupancy_mask` 屏蔽空槽）
  * `attention_dwell_n5.{png,pdf}` / `attention_dwell_n10.{png,pdf}`：Self-Attention 注意力权重 vs 剩余驻留步数
  * `cost_components.{png,pdf}`：标准测试集各策略日均可变成本及五项分项柱状对比（含离线理论下界）
  * `percentage_metrics.{png,pdf}`：可优化空间捕获率、净 EV+储能增成本削减率与整单节省比例
  * `price_regime_cost_components.{png,pdf}`：常规平缓日 vs 高波动尖峰日扣除纯楼宇刚性基荷后的净增量分解
  * `extended_0_to_10_percentage.{png,pdf}` / `extended_occupancy.{png,pdf}`：`0~10` 台车动态规模与夜间留宿混合场景泛化表现

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
├── tables/                          # 标准测试、验证、消融与扩展实验的 CSV / JSON 汇总表
├── training_logs/                   # N=5 与 N=10 完整训练过程记录（JSONL）
└── figures/                         # 全部结果图表的 PNG 与矢量 PDF 版本
```
