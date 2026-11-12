# SAC 强化学习调度与可视化（Part B）

本目录实现楼宇 BESS + V2G 车队半小时调度的 Soft Actor-Critic（SAC）训练、评估、独立推理接口与结果可视化。全套交付物（定稿权重、汇总表、高清图表及报告文字说明）已集中整理于 [docs/B_SAC/](../docs/B_SAC/README.md)。

## 1. 环境安装与单测校验

在仓库根目录下使用 Python 3.10+（推荐 3.12）虚拟环境安装依赖并运行全部单元测试：

```bash
pip install -r env/requirements.txt -r sac/requirements.txt
python -m unittest discover -v
```

测试套件覆盖 `env` 与 `sac` 共 135 项测试（含网络形状、槽位置换等变、空槽零动作、全空车队前后向无 NaN、跨 `N=1..10` 参数共享、编码器梯度截断、零功率动作锚定、离线参考回放核账及独立进程推理）。

## 2. 外部推理接口（供 `analysis/` 统一入口调用）

```python
from sac.policy import load_policy

policy = load_policy("docs/B_SAC/models/final_n5.pt", n_slots=5)   # N=10 使用 final_n10.pt
action = policy.act(obs, deterministic=True)                       # shape=(n_slots+1,), float32 in [-1, 1]
```

* `obs` 直接传入 `V2GEnv.reset()` / `step()` 返回的原始观测字典；模型内部自带训练期冻结的特征变换与鲁棒标准化器。
* 返回的 `action` 直接传入 `env.step(action)`，空槽对应动作恒为 `0.0`。

## 3. 命令行复现命令

```bash
# 1) 在指定数据集划分上运行评估并导出标准 48 行 CSV
python -m sac.evaluate --checkpoint docs/B_SAC/models/final_n5.pt --n-slots 5 --split test --output-dir results/sac/eval_n5
python -m sac.evaluate --checkpoint docs/B_SAC/models/final_n10.pt --n-slots 10 --split test --output-dir results/sac/eval_n10

# 2) 从头训练（仅使用训练日 2026-10-01..14 × 种子 0..99，在固定验证网格选优保存 best.pt）
python -m sac.train --config sac/configs/default.json --n-slots 5 --output-dir sac/checkpoints/run_n5
python -m sac.train --config sac/configs/default.json --n-slots 10 --output-dir sac/checkpoints/run_n10

# 3) 由评估产物重新生成全部 PNG / PDF 图表
python -m analysis.plotting.plot_sac --results results/sac --figures-dir docs/B_SAC/figures
python -m analysis.plotting.plot_extended --results results/sac --figures-dir docs/B_SAC/figures
```

## 4. 网络结构与核心超参数（`sac/configs/default.json`）

* **架构**：1D-CNN 时序编码器（`3×20 → 64`）+ 4 头 EV Self-Attention 编码器（空槽掩码屏蔽 + `sqrt(n_present)` 尺度保持池化 + 共享槽位动作头）+ Twin Critic（编码器梯度在 Actor 处 `detach`）。
* **动作与奖励设计**：可行功率区间内的零功率待机锚定映射（`u=0 → 0 kW`）；训练期采用扣除纯楼宇外生步成本的方差缩减回报 `r_train = 0.05 × (r_env + C_building_step)`，`γ = 1.0`，验证与测试结算全程采用环境原始 `SGD` 口径。
* **关键超参数**：`steps = 120,000`（2,500 日），`batch_size = 128`，`replay_capacity = 120,000`，`update_every = 2`，`actor_lr = 1e-4`，`critic_lr = 3e-4`，`alpha_lr = 3e-4`，目标熵 `-1.5 × 有效动作维数`，验证选优步数为 `N=5 @ 72k`、`N=10 @ 120k`。

## 5. 核心验证与测试结果摘要

单位为日均可变成本（`SGD/天`），标准测试集为 `2026-10-22..28 × seed 100..119`（140 个场景均值，纯楼宇无车无储能基准为 `1233.94 SGD/天`）：

| 规模 | 验证集 `SAC` | 测试集 `offline_opt` | 测试集 `SAC` | 测试集 `immediate` | 测试集 `rule` | 测试集 `random` | 即充空间捕获率 | 离场达标率 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `N=5` | 1208.25 | 1172.54 | **1200.69** | 1243.08 | 1261.68 | 1325.24 | **60.1%** | **100%** |
| `N=10` | 1223.06 | 1169.33 | **1198.34** | 1256.19 | 1286.69 | 1349.45 | **66.6%** | **100%** |

详细分项成本、消融对比、`0~10` 台车动态规模与夜间停靠扩展实验及机理分析见 [docs/B_SAC/B_SAC_report_draft.md](../docs/B_SAC/B_SAC_report_draft.md)。
