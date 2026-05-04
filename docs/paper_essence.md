# Slow Momentum with Fast Reversion 论文记忆卡

这份文档是为了以后重新打开项目时，快速想起 Wood, Roberts, and Zohren (2022) 这篇论文到底在解决什么问题、核心方法是什么、公式在表达什么，以及它为什么适合迁移到分钟级指数择时。

## 一句话记住这篇论文

传统 CTA / time-series momentum 在趋势反转点附近反应慢，容易继续沿着旧趋势下注。论文的核心办法是：先用在线 Gaussian Process changepoint detection 识别市场是否进入非平稳/变点状态，再把变点强度和变点位置作为特征输入 Deep Momentum Network，让 LSTM 学会什么时候坚持慢动量、什么时候快速反转仓位。

## 背景：为什么慢动量会出问题

时间序列动量的基本信念是“强趋势会延续”。这在 CTA 里很重要，因为很多趋势跟踪策略本质上是在做：

```text
过去涨 -> 做多
过去跌 -> 做空
```

问题出在 momentum turning point。趋势刚反转时，长窗口动量信号还在看旧趋势，所以仓位会滞后。例如原来 12 个月上涨，最近 1 个月突然下跌，慢信号还可能让模型继续做多。这个阶段不是标准趋势延续，而更像 regime change 或局部 mean reversion。

论文把这种冲突理解成两类机制的平衡：

- slow momentum：用长窗口过滤噪声，抓持久趋势。
- fast reversion：在变点或局部失衡后快速翻仓，再在局部修正完成后切回。

所以论文不是简单说“短周期更好”，而是让模型根据 CPD 状态信息自己学习何时慢、何时快。

## 核心结构

整条 pipeline 可以记成：

```text
price -> returns -> GP-CPD -> cp_score/cp_loc
      -> normalized returns + MACD + CPD features
      -> LSTM/DMN
      -> position in (-1, 1)
      -> Sharpe loss
```

最重要的是：CPD 是模型前置特征，不是在 LSTM 内部临时判断。论文明确要求预计算每个 asset-time pair 的 changepoint severity 和 location，然后与多周期收益、MACD 一起输入 LSTM。

## GP-CPD 方法

### 1. 收益和窗口标准化

对资产 `i`，价格序列为 `p_t^{(i)}`。先计算简单收益：

```math
r_t^{(i)} = \frac{p_t^{(i)} - p_{t-1}^{(i)}}{p_{t-1}^{(i)}}
```

对每个时点 `T`，取过去 `l` 个观测构成 CPD lookback window。窗口内标准化：

```math
\hat r_t^{(i)} =
\frac{r_t^{(i)} - E_T[r_t^{(i)}]}
{\sqrt{Var_T[r_t^{(i)}]}}
```

这个步骤有两个作用：

- 去掉窗口内均值尺度。
- 让不同时间、不同资产的 CPD 输出更稳定。

### 2. 无变点模型：Matérn 3/2 GP

无变点假设认为整个窗口可以由一个平稳 GP 解释：

```math
\hat r_t = f(t) + \epsilon_t,\quad
f \sim GP(0, k_\xi),\quad
\epsilon_t \sim N(0, \sigma_n^2)
```

论文使用 Matérn 3/2 kernel：

```math
k(x,x') =
\sigma_h^2
\left(1 + \frac{\sqrt{3}|x-x'|}{\lambda}\right)
\exp\left(-\frac{\sqrt{3}|x-x'|}{\lambda}\right)
```

其中：

- `lambda`：时间尺度/相关长度。
- `sigma_h`：信号幅度。
- `sigma_n`：噪声。

把函数值积分掉后，窗口收益服从多元正态。优化目标是 negative log marginal likelihood：

```math
NLML_\xi =
\frac{1}{2}\hat r^T V^{-1}\hat r
+ \frac{1}{2}\log|V|
+ \frac{n}{2}\log(2\pi)
```

这里 `V = K + sigma_n^2 I`。

### 3. 有变点模型：sigmoid changepoint GP

硬切分的直觉是：变点 `c` 前后分别由两个不同 GP kernel 解释，并且前后区域基本不相关。

硬切分需要枚举很多 `c`，太贵。论文用 sigmoid gate 做 soft transition：

```math
\sigma(x) = \frac{1}{1 + e^{-s(x-c)}}
```

其中：

- `c`：变点位置。
- `s`：切换陡峭度。

changepoint kernel 可以理解为：

```math
k_{CP}(x,x') =
k_1(x,x') \cdot \sigma_{before}(x)\sigma_{before}(x')
+
k_2(x,x') \cdot \sigma_{after}(x)\sigma_{after}(x')
```

论文写法中使用的是两个 Matérn 3/2 kernel 通过 sigmoid 权重组合。这个结构允许：

- 变点前后有不同波动结构。
- 变点前后有不同相关长度。
- 变点可以是平滑过渡，而不是必须瞬间跳变。

这比“均值前后不同”的简单检测更强，因为它关注的是窗口内 covariance structure 是否改变。

### 4. `cp_score` 和 `cp_loc`

分别优化：

```text
NLML_M: 普通 Matérn GP 的负边际似然
NLML_CP: changepoint GP 的负边际似然
```

如果 changepoint GP 明显更好，那么：

```text
NLML_CP < NLML_M
```

说明窗口更像两个 regime。论文把改善幅度压缩成 `(0, 1)` 的 severity：

```math
\nu_t^{(i)}
=
1 - \frac{1}{1 + e^{-(NLML_{CP} - NLML_M)}}
```

等价地看，`NLML_M - NLML_CP` 越大，`cp_score` 越接近 1。

变点位置归一化为：

```math
\gamma_t^{(i)} =
\frac{c - (t-l)}{l}
```

所以：

- `cp_score` / `nu`：变点强度。
- `cp_loc` / `gamma`：变点在窗口内的位置，接近 1 表示更靠近当前时刻。

## DMN/LSTM 方法

Deep Momentum Network 的关键不是先预测收益再手工映射仓位，而是直接输出仓位：

```math
X_{T-\tau+1:T}^{(i)}
=
g(u_{T-\tau+1:T}^{(i)};\theta)
```

其中：

- `u` 是输入特征序列。
- `tau` 是 LSTM sequence length。
- `X_t` 是仓位，经过 `tanh` 限制在 `(-1, 1)`。

论文的输入包括：

- 多周期标准化收益。
- MACD indicators。
- CPD severity。
- CPD location。

在线预测时，虽然 LSTM 输出一段序列，但真正用于交易的是最后一个时点的仓位。

## 为什么直接优化 Sharpe

论文认为，仅仅预测方向不等价于赚钱，因为金融收益经常是少数大波动主导。DMN 直接优化风险调整收益：

```math
L_{Sharpe}(\theta)
=
-
\frac{\sqrt{252}E_\Omega[R_t^{(i)}]}
{\sqrt{Var_\Omega[R_t^{(i)}]}}
```

也就是最大化年化 Sharpe，训练目标取负号做 loss。

这里 `Omega` 是 asset-time pair 集合。这个设计让模型学习的不只是方向，而是仓位大小、风险和收益的权衡。

## 实验设计

论文原始实验：

- 数据：50 个流动性较好的连续期货。
- 资产类别：商品、权益指数、固定收益、外汇。
- 时间：1990-2020。
- 回测：expanding window。
- 模型：LSTM/DMN 与 LSTM+CPD 对比。
- CPD LBW：`10, 21, 63, 126, 252` 日，可固定，也可作为超参选择。
- 训练：Adam，最多 300 epoch，early stopping patience 25。
- LSTM sequence length：63。
- 超参：dropout、hidden size、batch size、learning rate、gradient norm、CPD LBW。

论文报告的核心结论：

- 加入 CPD 后，Sharpe 相比 LSTM baseline 明显提升。
- 在 2015-2020 这种传统动量表现变差、非平稳更强的阶段，CPD 帮助更大。
- CPD LBW 太短会噪声多，太长会只关注太久以前的大变点，论文发现中等窗口更有效。
- 交易成本会削弱 fast reversion 的收益；成本越高，模型越应该偏向更慢、更稳的变点窗口。

## 这篇论文真正有用的地方

对本项目最有价值的不是“日频期货配置”本身，而是这三个结构性想法：

1. 变点检测应该作为状态特征输入模型，而不是用硬规则直接交易。
2. CPD 输出不只是是否有变点，还要输出强度和位置。
3. LSTM/DMN 负责学习不同市场状态下的仓位映射。
4. 交易目标应该接近实际策略收益，而不是只做收益预测。
5. 在高频或分钟级场景，交易成本和换手控制必须进入训练/选模逻辑。

## 以后回忆时重点看这里

- 论文要解决的是趋势反转点附近的慢动量滞后。
- CPD 是前置 feature：`cp_score` 和 `cp_loc`。
- CPD 原版是 Matérn 3/2 GP vs sigmoid changepoint GP，不是简单统计 proxy。
- `cp_score` 来自有变点模型相对无变点模型的 NLML 改善。
- `cp_loc` 是优化出来的变点位置在窗口里的归一化坐标。
- LSTM/DMN 输入多周期收益、MACD、CPD 特征，输出仓位。
- loss 直接优化 negative Sharpe。
- 交易成本会让 fast reversion 失效，所以分钟级复现必须额外控制换手。
