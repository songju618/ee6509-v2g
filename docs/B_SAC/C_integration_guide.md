# 给 C / F 同学的模型接入与运行指南

本文档说明如何在统一运行入口（`analysis/`）或灵敏度测试中直接加载定稿 SAC 模型并导出标准 48 行 CSV。

---

## 1. 环境依赖

在仓库根目录下安装 `env` 与 `sac` 依赖（Python 3.10+，推荐 3.12）：

```bash
pip install -r env/requirements.txt -r sac/requirements.txt
```

---

## 2. 在统一运行入口中调用 `load_policy`（给 C 同学）

`sac.policy.load_policy(path, n_slots)` 在 CPU 上独立加载权重及训练期冻结的观测标准化参数，不依赖训练器状态，支持任意槽位数 `n_slots ∈ {1, ..., 10}`（含周末或空车队场景）。

```python
from env import V2GEnv
from env.recording import write_csv
from sac.policy import load_policy

# N=5 加载 final_n5.pt；N=10（或 1~10 动态车队测试）加载 final_n10.pt
policy = load_policy("docs/B_SAC/models/final_n5.pt", n_slots=5)

env = V2GEnv(n_slots=5)
obs, info = env.reset(seed=100, options={"date": "2026-10-22"})
infos = []
for _ in range(48):
    action = policy.act(obs, deterministic=True)  # shape=(n_slots+1,), float32, in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    infos.append(info)

write_csv("results/SAC_2026-10-22_seed100_n5.csv", infos, strategy="SAC")
```

### 接口说明与核对点
1. **输入观测 `obs`**：直接传入 `V2GEnv.reset()` / `step()` 返回的原始字典（`timeseries: (3, 20)`、`global: (4,)`、`ev: (N, 4)`、`ev_present: (N,)`），**切勿在外部重复做标准化**（`Policy` 内部已封装特征构造与训练集冻结的 `Normalizer`）。
2. **输出动作 `action`**：形状为 `(N + 1,)`、范围 `[-1, 1]` 的 `np.float32` 数组，顺序为 `[BESS, EV_1, ..., EV_N]`，可直接传给 `env.step(action)`（**使用 `env.step(action)`，不要传给 `env.step_power()`**）。空槽位对应的动作分量恒为 `0.0`。
3. **注意力权重提取（可选）**：如需读取当前步 EV Self-Attention 权重矩阵（形状 `(N, N)`），可调用 `weights = policy.attention(obs)`。

---

## 3. 命令行直接批量评估导出 CSV（给 F / C 同学）

如需直接通过命令行在指定数据集划分或自定义 `(日期, 种子)` 网格上批量导出标准 48 行 CSV、`.summary.json` 与 `.scenario.json`：

```bash
# 标准测试集（2026-10-22..28 × seed 100..119，共 140 个场景）
python -m sac.evaluate \
  --checkpoint docs/B_SAC/models/final_n5.pt \
  --n-slots 5 \
  --split test \
  --output-dir results/sac_eval_n5

python -m sac.evaluate \
  --checkpoint docs/B_SAC/models/final_n10.pt \
  --n-slots 10 \
  --split test \
  --output-dir results/sac_eval_n10
```

---

## 4. 对齐正式基线后重绘对比图

目前 `docs/B_SAC/figures/` 与 `tables/` 中的 `immediate`、`rule`、`offline_opt` 是由 `sac/references.py` 与 `sac/offline.py`（基于 SciPy HiGHS）在同一测试网格上跑出的内部核账对照：
* `immediate`（到场即按最大功率充至目标，BESS 不动）与 `offline_opt`（全天无噪真值 LP/MILP 理论下界，`env.step_power` 回放误差 `< 0.00013 SGD`）数学定义唯一，可与 C 同学的 `baselines/` 和 `milp/` 输出交叉核对（标准测试集日均可变成本：`N=5` 为 `1243.08` 与 `1172.54 SGD/天`；`N=10` 为 `1256.19` 与 `1169.33 SGD/天`）。
* 待 C 同学确定正式版 `rule`（低价充高价放）阈值并由统一入口产出全策略汇总后，如需替换柱状图中的 `rule` 数值，只需更新对应 `comparison.json` 后运行：

```bash
python -m analysis.plotting.plot_sac --results results/sac --figures-dir docs/B_SAC/figures
python -m analysis.plotting.plot_extended --results results/sac --figures-dir docs/B_SAC/figures
```
