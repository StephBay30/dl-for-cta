# dl-for-cta 项目复现记忆文档

这份文档是为了以后重新打开 `dl-for-cta` 时，快速想起这个项目到底在做什么、为什么这样设计、当前代码跑到了哪里，以及下一步应该怎么接着做。

## 一句话记住这个项目

本项目不是照搬论文的日频连续期货实验，而是把 “Slow Momentum with Fast Reversion” 的核心机制迁移到 A 股连续指数分钟级数据：先用论文版 GP-CPD 离线生成变点特征，再把这些状态特征输入 LSTM/DMN，让模型每分钟输出仓位，并用可学习调仓阈值控制换手。

## 本项目和论文的对应关系

| 论文原始设置 | 本项目迁移设置 |
|---|---|
| 50 个连续期货 | 4 个 A 股连续指数 |
| 日频 close 数据 | 分钟级 index bar |
| 每日输出仓位 | 每分钟输出仓位 |
| CPD LBW 为 `10/21/63/126/252` 日 | CPD window 为 `60/240/1200` 分钟，或通过 TOML 搜索 |
| CPD 输入日收益 | CPD 输入分钟收益 |
| LSTM 输入多周期收益、MACD、CPD | 本项目输入分钟特征、MACD、time-of-day、CPD |
| 论文主要看 raw signal 和交易成本附录 | 本项目 valid 直接用 after-cost Sharpe 选模 |
| 原论文没有 learned threshold | 本项目加入全局可学习 no-trade threshold 控制分钟级过度换手 |

最重要的迁移判断：指数是连续的，所以先只做指数；`IC/IF/IH/IM` 是合约目录，合约存在换月和不连续问题，暂时不纳入。

## 按论文步骤对照项目

以后想从论文一步步对照代码时，先看这张表。它不是文件清单，而是“论文机制在项目里落到哪里”的路线图。

| 论文步骤 | 项目对应 | 你应该看 |
|---|---|---|
| 价格变收益 | 分钟 close 变分钟收益 | `src/dl_for_cta/features/basic.py` |
| 取 rolling CPD window | `60/240/1200` 分钟窗口，或通过 TOML 搜索 | `configs/cpd_dmn_minute.toml` 的 `[cpd]` |
| Matérn 3/2 GP | 普通无变点 GP | `src/dl_for_cta/features/cpd_gp.py` |
| sigmoid changepoint GP | 有变点 GP | `src/dl_for_cta/features/cpd_gp.py` |
| `nu_t`, `gamma_t` | `cp_score_{window}`, `cp_loc_{window}` | `src/dl_for_cta/features/cpd_gp.py` |
| 多周期收益 + MACD + CPD 输入 LSTM | 分钟特征 + MACD + time-of-day + CPD | `src/dl_for_cta/features/basic.py`, `src/dl_for_cta/features/build_features.py` |
| LSTM 输出仓位 | `tanh` 输出 `[-1, 1]` 仓位 | `src/dl_for_cta/models/dmn_lstm.py` |
| Sharpe loss | negative Sharpe + turnover penalty | `src/dl_for_cta/models/losses.py` |
| validation 选模型 | valid after-cost Sharpe 选 epoch 和超参 | `src/dl_for_cta/experiments/run_train.py` |
| out-of-sample test | 只用 `best_model.pt` 跑 test | `src/dl_for_cta/experiments/run_backtest.py` |

## 数据

默认分钟数据路径：

```text
E:/quant/lyquant/short_arb_firm/data/min_bar
```

默认指数：

```text
000016.XSHG
000300.XSHG
000905.XSHG
000852.XSHG
```

已观察到的数据结构：

```text
data/min_bar/
  000016.XSHG/
    2015-01-05.parquet
    ...
  000300.XSHG/
  000905.XSHG/
  000852.XSHG/
  IC/
  IF/
  IH/
  IM/
```

单个 parquet 的字段：

```text
order_book_id
datetime
volume
open
total_turnover
high
low
close
```

每个正常交易日 240 根分钟 bar。

当前检查到的数据范围：

```text
000016.XSHG: 2015-01-05 到 2026-04-29
000300.XSHG: 2015-01-05 到 2026-04-29
000905.XSHG: 2015-01-05 到 2026-04-29
000852.XSHG: 2016-01-04 到 2026-04-29
```

## 工程结构

核心目录：

```text
configs/
  cpd_dmn_minute.toml

src/dl_for_cta/
  cli.py
  config/
  data/
  features/
  models/
  backtest/
  experiments/
  utils/

cache/
  features/
  cpd/

outputs/
  <experiment_name>/
```

注意：仓库名是 `dl-for-cta`，Python 包名是 `dl_for_cta`，这是因为 Python import 不能使用连字符。

## 运行方式

主配置：

```text
configs/cpd_dmn_minute.toml
```

常用命令：

```powershell
python -m dl_for_cta.cli inspect-data --config configs/cpd_dmn_minute.toml
python -m dl_for_cta.cli build-features --config configs/cpd_dmn_minute.toml
python -m dl_for_cta.cli build-cpd --config configs/cpd_dmn_minute.toml
python -m dl_for_cta.cli train --config configs/cpd_dmn_minute.toml
python -m dl_for_cta.cli backtest --config configs/cpd_dmn_minute.toml
```

完整 pipeline：

```powershell
python -m dl_for_cta.cli run-pipeline --config configs/cpd_dmn_minute.toml
```

实际研究时建议分阶段跑，因为 GP-CPD 很慢：

1. `inspect-data`
2. `build-features`
3. `build-cpd`
4. `train`
5. `backtest`

## 特征设计

基础特征在 `src/dl_for_cta/features/basic.py`：

- 多窗口分钟收益：`ret_1, ret_5, ...`
- momentum gap。
- rolling volatility。
- intraday range。
- short reversal。
- volume/turnover z-score。
- time-of-day sin/cos。
- MACD。

target：

```text
target_ret_1
target_ret_5
target_ret_30
```

训练时可以通过 TOML 的 `target_horizons` 和 `target_weights` 合成为 `target_ret`。

防 lookahead 原则：

- 所有 feature 只能用 `t` 及以前数据。
- forward target 只作为训练标签，不进入 feature columns。
- 回测中 `position_t` 只作用于 `t+1` 的收益。

## GP-CPD 实现

实现位置：

```text
src/dl_for_cta/features/cpd_gp.py
```

本项目没有使用轻量统计 proxy，而是实现论文机制：

```text
Matérn 3/2 GP
vs
sigmoid changepoint GP
```

每个 symbol、每个窗口、每个分钟点：

1. 取过去 `window` 分钟收益。
2. 窗口内标准化。
3. 优化普通 Matérn 3/2 GP 的 NLML。
4. 用普通 GP 参数初始化 changepoint GP。
5. 优化 sigmoid changepoint GP 的 NLML。
6. 输出：
   - `cp_score_{window}`
   - `cp_loc_{window}`

默认窗口：

```toml
[cpd]
windows = [60, 240, 1200]
```

CPD 会写入缓存：

```text
cache/cpd/
```

重要现实：论文版 GP-CPD 在分钟级数据上非常慢。已跑过的最小样本中，单指数 `000300.XSHG`、98 个交易日、窗口 60、约 23520 行，CPD 约耗时 24 分 39 秒。

## 模型和阈值

模型位置：

```text
src/dl_for_cta/models/dmn_lstm.py
src/dl_for_cta/models/threshold.py
```

LSTM/DMN：

- 输入：基础分钟特征 + CPD 特征。
- 输出：`raw_position_t in [-1, 1]`。
- 输出层用 `tanh` 限制仓位。

全局 learned threshold：

```text
如果 abs(raw_position_t - actual_position_{t-1}) <= theta:
    不调仓
否则:
    调仓到 raw_position_t
```

训练阶段使用 soft gate 近似，回测阶段使用 hard threshold。

这个 threshold 是项目相对论文的重要扩展，因为分钟级策略如果每分钟都调仓，交易成本会非常严重。

## Train / Valid / Test

训练入口：

```text
src/dl_for_cta/experiments/run_train.py
```

当前规则：

- train：`train_start <= datetime <= first_train_end`
- valid：`validation_start <= datetime <= validation_end`
- test：`datetime >= first_test_start`

训练流程：

1. 从 TOML 的 `[search]` 展开候选超参。
2. 每个候选单独训练。
3. 每个 epoch 后在 valid 区间跑 after-cost backtest。
4. 用 valid after-cost Sharpe 选该候选的 best epoch。
5. 所有候选里再选 valid after-cost Sharpe 最高者。
6. 保存为 `best_model.pt`。
7. test 只用 `best_model.pt` 跑一次。

输出：

```text
outputs/<experiment_name>/
  candidates/<candidate_id>/best_epoch.pt
  best_model.pt
  search_results.csv
  validation_metrics.csv
  test_metrics.csv
  test_positions.parquet
```

这个设计是为了避免把 test 当 validation 用。valid 用来选 epoch 和超参，test 只用于最终评估。

## 当前已经跑通过什么

已跑通过基础测试：

```powershell
python -m pytest -q -p no:cacheprovider
```

结果：

```text
9 passed
```

已跑通过一个最小真实样本：

```text
symbol: 000300.XSHG
train: 2015-01-05 到 2015-03-31
valid: 2015-04-01 到 2015-04-30
test: 2015-05-01 到 2015-05-29
CPD window: [60]
epochs: 3
```

输出目录：

```text
outputs/mini_3m_1m_1m
```

结果摘要：

```text
best epoch: 1
best valid after-cost Sharpe: -0.0521
test after-cost Sharpe: -5.3728
test annual return: -21.26%
test annual volatility: 3.96%
test max drawdown: -2.92%
test turnover: 0.00630
test trade count: 490
```

这个结果不是为了证明策略有效，只是证明工程链路已经闭合。

## GPU 当前状态

机器能看到：

```text
NVIDIA GeForce RTX 4060 Laptop GPU
```

但当前 Python 环境中的 PyTorch 是 CPU 版：

```text
torch 2.10.0+cpu
torch.cuda.is_available() = False
torch.version.cuda = None
```

所以目前训练日志会显示：

```text
[train] device=cpu
```

如果以后要用 GPU，需要安装 CUDA 版 PyTorch。

## 下一步应该做什么

优先级从高到低：

1. 加速 GP-CPD。
   - 现在最大瓶颈是 CPD。
   - 需要按 symbol/date/window 分片并行。
   - 需要更好的断点续算和进度记录。
2. 跑完整四指数小窗口实验。
   - 先不要直接上 `[60, 240, 1200]` 全量。
   - 可以先单窗口 `[60]`、短年份验证。
3. 补 baseline。
   - long-only。
   - minute momentum。
   - minute reversal。
   - MACD。
4. 实现真正 multi-window expanding evaluation。
   - 当前是单个 train/valid/test split。
   - 未来要按论文那样多段 expanding window。
5. 安装 CUDA 版 PyTorch。
   - 否则模型训练不会用 GPU。
6. 对 valid 指标做更稳健选择。
   - 当前用 after-cost Sharpe。
   - 后续可加 max drawdown、turnover 约束。

## 以后回忆时重点看这里

- 项目本质：论文机制的 A 股指数分钟级迁移。
- CPD 是模型前置特征，不是交易规则。
- CPD 必须输出 `cp_score` 和 `cp_loc`。
- 每分钟输出仓位，但超过 learned threshold 才调仓。
- valid 用 after-cost Sharpe 选模型，test 只跑一次。
- 当前最大瓶颈是 GP-CPD，不是 LSTM。
- 当前 PyTorch 是 CPU 版，没有用上 RTX 4060。
