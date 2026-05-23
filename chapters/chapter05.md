# 第5章 对数、指数与复利——金融世界的增长语言

> **本章核心问题**：为什么量化金融偏爱"对数收益率"？为什么一张股票价格图看起来像是指数函数在随机抖动？

---

## 5.1 动机：量化金融的增长语言

在量化金融中，我们每天都在和"增长"打交道：资产价格在增长（或缩水），账户净值在复利累积，策略收益在多期叠加。处理这些增长问题时，有两种数学工具始终站在舞台中央：

- **指数函数** $e^x$：描述连续增长的"自然语言"
- **对数函数** $\ln(x)$：将指数增长"拉平"成线性增长的"解码器"

如果你只用高中数学的视角看它们，$e$ 不过是一个约等于 $2.718$ 的无理数，$\ln$ 不过是"求以 $e$ 为底的对数"。但在量化金融中，它们承载着远比这更深刻的金融意义：

1. **连续复利**是衍生品定价（如 Black-Scholes 模型）的基石；
2. **对数收益率**是时间序列分析、风险建模和组合优化的标准语言；
3. **对数正态分布**假设是绝大多数金融随机过程的起点。

本章将回答一个关键问题：**为什么量化从业者几乎总是用对数收益率，而不是普通人更熟悉的"涨了百分之几"？**

---

## 5.2 数学原理

### 5.2.1 从复利到 $e$ 的自然涌现

想象你在银行存入 $1$ 元钱，年利率 $100\%$。

- **一年复利一次**：年末得到 $1 \times (1 + 1) = 2$ 元；
- **半年复利一次**：每半年利率 $50\%$，年末得到 $1 \times (1 + 0.5)^2 = 2.25$ 元；
- **每月复利一次**：每月利率 $1/12$，年末得到 $1 \times (1 + \frac{1}{12})^{12} \approx 2.613$ 元；
- **每天复利一次**：年末得到 $1 \times (1 + \frac{1}{365})^{365} \approx 2.714$ 元。

如果你疯狂到**每时每刻都在复利**（即复利次数 $n \to \infty$），年末你将得到：

$$
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e \approx 2.71828\cdots
$$

这就是 $e$ 的"金融定义"：**在单位时间内，以 $100\%$ 的利率连续复利，$1$ 元本金的增长极限**。

更一般地，本金 $P$，年利率 $r$，时间 $t$，连续复利的终值为：

$$
FV = P \cdot e^{rt}
$$

**几何直觉**：离散复利就像用越来越小的台阶去逼近一条光滑的指数曲线。当台阶无限细分，台阶的顶端就无缝贴合成了 $e^{rt}$ 的曲线。

---

### 5.2.2 离散复利与连续复利的统一

在实际金融产品中，我们遇到的多是**离散复利**（如每年、每季度付息一次）。但在量化建模中，**连续复利**数学上更优雅、更易处理（尤其是求导和积分）。

两者可以相互转换。设离散年复利率为 $r_d$，连续复利率为 $r_c$，要求两者等价：

$$
(1 + r_d)^t = e^{r_c t}
$$

取对数得：

$$
r_c = \ln(1 + r_d)
$$

反过来：

$$
r_d = e^{r_c} - 1
$$

**金融含义**：$\ln(1 + r)$ 就是把"离散利率"翻译成"连续利率"的汇率牌价。

---

### 5.2.3 对数收益率：为什么量化人更爱它？

假设某股票价格从 $P_{t-1}$ 涨到 $P_t$。

**简单收益率**（Simple Return）是我们日常生活中最常用的：

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1
$$

**对数收益率**（Log Return，又称连续复合收益率）则是：

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})
$$

为什么量化分析几乎清一色使用对数收益率？因为它有四个不可替代的优越性：

#### 优越性 1：时间可加性（Time Additivity）

这是最重要的一点。

如果你持有某股票两期，第一期涨 $10\%$，第二期涨 $20\%$。简单收益率的总收益不是 $10\% + 20\% = 30\%$，而是：

$$
(1 + 0.10)(1 + 0.20) - 1 = 32\%
$$

你需要做乘法，再做减法，才能算出总收益。

但对数收益率只需要**做加法**：

$$
r_{\text{total}} = \ln\left(\frac{P_2}{P_0}\right) = \ln\left(\frac{P_2}{P_1}\cdot\frac{P_1}{P_0}\right) = \ln\left(\frac{P_2}{P_1}\right) + \ln\left(\frac{P_1}{P_0}\right) = r_1 + r_2
$$

**多期对数收益率之和，严格等于总对数收益率。** 这使得在计算"年化收益"、"累计收益"时，数学处理极其简洁。

#### 优越性 2：对称性

假设价格先涨 $10\%$ 再跌 $10\%$。

- 简单收益率：$(1 + 0.10)(1 - 0.10) - 1 = -1\%$（你亏了！）
- 对数收益率：$\ln(1.10) + \ln(0.90) \approx 0.0953 - 0.1054 = -0.0101$（和为负，精确对称）

对数收益率天然满足"涨多少再跌多少，和为零"的直觉，而简单收益率在这方面是"有偏"的。

#### 优越性 3：近似百分比变化

当收益率较小时（比如日收益率通常在 $\pm 3\%$ 以内），有一个极其好用的近似：

$$
\ln(1 + x) \approx x \quad \text{当 } x \to 0 \text{ 时}
$$

这意味着**对数收益率在数值上近似等于简单收益率**。你可以把它理解为：对数收益率是"百分比变化的微积分版本"，在变化很小时两者几乎一样，但对数版本拥有更好的数学性质。

#### 优越性 4：正态分布的便利性

金融建模中有一个核心假设：**对数价格服从正态分布**，即价格服从**对数正态分布**（Log-normal Distribution）。

如果 $\ln(P_t) - \ln(P_{t-1}) \sim \mathcal{N}(\mu, \sigma^2)$，那么价格 $P_t$ 的分布就是良好定义的。这使得 Black-Scholes 期权定价、风险价值（VaR）计算、最大似然估计等一整套量化工具得以成立。

如果用简单收益率假设正态分布，会出现一个荒谬的结果：理论上价格可能跌到负数（因为正态分布取值范围是 $(-\infty, +\infty)$）。但对数正态分布的价格始终为正，这更符合现实。

---

## 5.3 核心公式速查

> 本节是前述各节公式的集中汇总，供复习和查阅使用。

| 公式 | 名称 | 符号说明 |
|------|------|----------|
| $FV = PV \cdot (1 + \frac{r}{n})^{nt}$ | 离散复利公式 | $PV$：现值； $r$ ：年利率； $n$ ：每年复利次数； $t$ ：年数 |
| $FV = PV \cdot e^{rt}$ | 连续复利公式 |  $e$ ：自然常数； $r_c = \ln(1+r_d)$ 为连续复利利率 |
| $r_c = \ln(1 + r_d)$ | 离散-连续利率转换 | $r_d$ : 离散利率;$r_c$ : 连续利率 |
| $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | 简单收益率 | $P_t$ : $t$ 时刻价格; $R_t$ : 简单收益率 |
| $r_t = \ln(\frac{P_t}{P_{t-1}}) = \ln(P_t) - \ln(P_{t-1})$ | 对数收益率 | $r_t$  : 对数收益率（连续复合收益率） |
| $r_{\text{total}} = \sum_{i=1}^{T} r_i = \ln(\frac{P_T}{P_0})$ | 对数收益率时间可加性 | $T$ 期对数收益率之和等于总对数收益率 |
| $\ln(1+x) \approx x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$ | 对数函数泰勒展开 | 当 $\|x\| < 1$ 时成立; $x \to 0$ 时 $\ln(1+x) \approx x$ |

---

## 5.4 Python 示例

### 示例 1：$e$ 的涌现——从离散复利到连续复利

```python
import numpy as np
import matplotlib.pyplot as plt

# 设置字体（文泉驿微米黑）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
r = 1.0      # 年利率 100%
t = 1.0      # 1 年
n_values = [1, 2, 4, 12, 52, 365, 1000, 10000]  # 复利次数

# 计算离散复利终值
discrete_fvs = [(1 + r/n)**(n*t) for n in n_values]
continuous_fv = np.exp(r * t)

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n_values, discrete_fvs, 'bo-', markersize=8, label='离散复利终值')
ax.axhline(y=continuous_fv, color='r', linestyle='--', label=f'连续复利极限 $e$ ≈ {continuous_fv:.5f}')

ax.set_xscale('log')
ax.set_xlabel('每年复利次数 $n$（对数刻度）', fontsize=12)
ax.set_ylabel('1 元本金终值', fontsize=12)
ax.set_title('从离散复利到连续复利：$e$ 的自然涌现', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 标注关键数值
for n, fv in zip(n_values[:4], discrete_fvs[:4]):
    ax.annotate(f'{fv:.3f}', xy=(n, fv), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print(f"连续复利极限 e^{r*t} = {continuous_fv:.10f}")
print(f"数学常数 e = {np.e:.10f}")
```

**运行结果解读**：随着复利次数 $n$ 增加，离散复利终值从 $2.0$（年复利）逐步逼近红线所示的 $e \approx 2.71828$。这就是"连续"的数学含义——无限细分后的极限。

---

### 示例 2：简单收益率 vs 对数收益率的计算与对比

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 模拟一个价格序列（带趋势和波动）
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=60, freq='B')  # 60 个交易日
returns = np.random.normal(loc=0.001, scale=0.02, size=60)  # 日简单收益率
prices = 100 * np.cumprod(1 + returns)  # 从 100 元开始累积

# 计算两种收益率
simple_returns = np.diff(prices) / prices[:-1]
log_returns = np.diff(np.log(prices))

# 创建 DataFrame 便于观察
df = pd.DataFrame({
    '价格': prices[1:],  # 从第2个价格开始，与收益率对齐
    '简单收益率': simple_returns,
    '对数收益率': log_returns,
    '差异': simple_returns - log_returns
}, index=dates[1:])

print("前 5 行数据对比：")
print(df.head())

# 可视化对比
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 价格序列
axes[0].plot(df.index, df['价格'], color='black', linewidth=1.5)
axes[0].set_title('模拟股票价格序列', fontsize=13)
axes[0].set_ylabel('价格（元）')
axes[0].grid(True, alpha=0.3)

# 收益率序列对比
axes[1].plot(df.index, df['简单收益率']*100, label='简单收益率', alpha=0.7, linewidth=1.2)
axes[1].plot(df.index, df['对数收益率']*100, label='对数收益率', alpha=0.7, linewidth=1.2)
axes[1].set_title('简单收益率 vs 对数收益率（%）', fontsize=13)
axes[1].set_ylabel('收益率（%）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 两者差异
axes[2].plot(df.index, df['差异']*10000, color='red', linewidth=1.2)  # 放大到基点
axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[2].set_title('简单收益率 - 对数收益率（单位：万分之一）', fontsize=13)
axes[2].set_ylabel('差异（bps）')
axes[2].set_xlabel('日期')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**关键观察**：在日度尺度上，简单收益率和对数收益率的差异极小（通常在万分之一级别），这就是为什么日常分析中可以近似互换。但在月度或年度尺度上，差异会显著放大。

---

### 示例 3：验证对数收益率的时间可加性

```python
import numpy as np

np.random.seed(2024)

# 模拟 20 期价格序列
initial_price = 100.0
periods = 20
simple_rets = np.random.normal(0.005, 0.03, periods)  # 每期简单收益率
prices = [initial_price]
for r in simple_rets:
    prices.append(prices[-1] * (1 + r))
prices = np.array(prices)

# 方法 1：逐期对数收益率求和
log_rets = np.diff(np.log(prices))
sum_of_log_rets = np.sum(log_rets)

# 方法 2：直接用首尾价格计算总对数收益率
total_log_ret = np.log(prices[-1] / prices[0])

# 方法 3：简单收益率"复合"计算（需要乘积）
total_simple_ret = np.prod(1 + simple_rets) - 1

print("=" * 50)
print("对数收益率时间可加性验证")
print("=" * 50)
print(f"初始价格: {prices[0]:.4f}")
print(f"最终价格: {prices[-1]:.4f}")
print(f"\n逐期对数收益率之和: {sum_of_log_rets:.8f}")
print(f"首尾价格总对数收益率: {total_log_ret:.8f}")
print(f"两者差异: {abs(sum_of_log_rets - total_log_ret):.2e}")
print(f"\n对比：简单收益率复合总收益: {total_simple_ret:.8f}")
print(f"简单收益率之和: {np.sum(simple_rets):.8f} （注意：这个不等于总收益！）")
print("=" * 50)

# 断言验证（数值精度范围内）
assert np.isclose(sum_of_log_rets, total_log_ret), "时间可加性验证失败！"
print("✓ 验证通过：对数收益率严格满足时间可加性")
```

**运行结果**：

```
==================================================
对数收益率时间可加性验证
==================================================
初始价格: 100.0000
最终价格: 120.2831

逐期对数收益率之和: 0.18467764
首尾价格总对数收益率: 0.18467764
两者差异: 2.78e-17

对比：简单收益率复合总收益: 0.20283063
简单收益率之和: 0.19635501 （注意：这个不等于总收益！）
==================================================
```

**核心结论**：逐期对数收益率相加（0.1847），精确等于用首尾价格直接算出的总对数收益率（0.1847），差异仅 $2.78 \times 10^{-17}$（机器精度）。而简单收益率的"和"（0.1964）不等于总收益（0.2028），必须做"乘积减一"的复合运算。

---

### 示例 4：验证 $\ln(1+x) \approx x$ 的近似精度

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 生成 -0.5 到 0.5 之间的收益率
x = np.linspace(-0.5, 0.5, 1000)
ln_1_plus_x = np.log(1 + x)
approx = x  # 一阶近似
second_order = x - x**2/2  # 二阶近似（泰勒展开）

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：函数对比
axes[0].plot(x, ln_1_plus_x, 'b-', linewidth=2, label=r'$\ln(1+x)$（精确值）')
axes[0].plot(x, approx, 'r--', linewidth=1.5, label=r'$x$（一阶近似）')
axes[0].plot(x, second_order, 'g-.', linewidth=1.5, label=r'$x - x^2/2$（二阶近似）')
axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[0].set_xlabel('x（收益率）', fontsize=12)
axes[0].set_ylabel('函数值', fontsize=12)
axes[0].set_title('对数收益率的泰勒近似', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-0.5, 0.5)

# 右图：近似误差
error_1st = np.abs(ln_1_plus_x - approx)
error_2nd = np.abs(ln_1_plus_x - second_order)

axes[1].semilogy(x, error_1st, 'r-', linewidth=1.5, label='一阶近似误差')
axes[1].semilogy(x, error_2nd, 'g-', linewidth=1.5, label='二阶近似误差')
axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
axes[1].set_xlabel('x（收益率）', fontsize=12)
axes[1].set_ylabel('绝对误差（对数刻度）', fontsize=12)
axes[1].set_title('近似误差分析', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-0.5, 0.5)

plt.tight_layout()
plt.show()

# 打印几个关键数值
test_values = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
print("\n近似精度检验表：")
print(f"{'简单收益率':>10} | {'对数收益率':>12} | {'一阶近似':>12} | {'误差':>10}")
print("-" * 60)
for v in test_values:
    log_ret = np.log(1 + v)
    approx_err = abs(log_ret - v)
    print(f"{v:10.2%} | {log_ret:12.6f} | {v:12.6f} | {approx_err:10.6f}")
```

**关键观察**：
- 当收益率在 $\pm 5\%$ 以内时，$\ln(1+x) \approx x$ 的误差约 $0.0012$（千分之一），几乎可以忽略；
- 当收益率达到 $\pm 20\%$ 时，误差扩大到约 $0.018$（1.8%），此时需要考虑二阶修正项 $-x^2/2$。

---

### 示例 5：从价格序列到对数收益率的完整工作流

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(2024)

# 模拟更真实的股价路径：几何布朗运动
# dP/P = mu*dt + sigma*dW
mu = 0.10      # 年化预期收益率 10%
sigma = 0.20   # 年化波动率 20%
T = 1.0        # 1 年
n_days = 252   # 252 个交易日
dt = T / n_days

# 生成价格路径
random_shocks = np.random.normal(0, 1, n_days)
log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
# 注意：这里用了 mu - 0.5*sigma^2，这是伊藤引理修正项，第24章会详细讲解

prices = 100 * np.exp(np.cumsum(log_returns))
prices = np.insert(prices, 0, 100.0)  # 第0天价格为100

dates = pd.date_range('2024-01-02', periods=n_days+1, freq='B')
df = pd.DataFrame({'收盘价': prices}, index=dates)

# 计算对数收益率
df['对数收益率'] = np.log(df['收盘价'] / df['收盘价'].shift(1))

# 计算累计对数收益率（从起点开始）
df['累计对数收益率'] = df['对数收益率'].cumsum()

# 验证：累计对数收益率应该等于 log(Price_t / Price_0)
df['验证_累计对数收益'] = np.log(df['收盘价'] / df['收盘价'].iloc[0])

print("数据前 5 行：")
print(df.head())

print(f"\n全年总对数收益率（求和法）: {df['对数收益率'].sum():.6f}")
print(f"全年总对数收益率（首尾法）: {np.log(df['收盘价'].iloc[-1] / df['收盘价'].iloc[0]):.6f}")
print(f"全年简单收益率: {(df['收盘价'].iloc[-1] / df['收盘价'].iloc[0]) - 1:.6f}")

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 价格走势
axes[0].plot(df.index, df['收盘价'], color='black', linewidth=1.2)
axes[0].set_title('模拟股票价格路径（几何布朗运动）', fontsize=13)
axes[0].set_ylabel('价格（元）')
axes[0].grid(True, alpha=0.3)

# 日对数收益率
axes[1].plot(df.index, df['对数收益率'] * 100, color='steelblue', linewidth=0.8)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_title('日对数收益率（%）', fontsize=13)
axes[1].set_ylabel('收益率（%）')
axes[1].grid(True, alpha=0.3)

# 累计对数收益率 vs 验证
axes[2].plot(df.index, df['累计对数收益率'] * 100, label='逐期累加', color='green', linewidth=1.5)
axes[2].plot(df.index, df['验证_累计对数收益'] * 100, label='首尾直接计算', color='red', 
             linestyle='--', linewidth=1.2, alpha=0.7)
axes[2].set_title('累计对数收益率验证（两条线应完全重合）', fontsize=13)
axes[2].set_ylabel('累计收益率（%）')
axes[2].set_xlabel('日期')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**工作流要点**：
1. 原始数据通常是**价格序列**；
2. 第一步转换为**对数收益率**：`np.log(P_t / P_{t-1})`；
3. 所有多期分析（年化、累计、波动率）都在**对数收益率层面**做加法或标准差运算；
4. 最后如果需要，再用指数映射回价格空间: $P_T = P_0 \cdot e^{\sum r_i}$ 

---

## 5.5 本章小结

| 要点 | 核心内容 |
|------|----------|
| **$e$ 的含义** | 连续复利的自然极限，是增长过程的"标准单位" |
| **连续复利优势** | 数学处理简洁（求导/积分方便），与衍生品定价天然兼容 |
| **对数收益率定义** | $r_t = \ln(P_t / P_{t-1})$ |
| **四大优越性** | 时间可加、对称性好、近似百分比变化、兼容正态假设 |
| **使用原则** | 日度分析中简单/对数收益率差异极小；所有多期复合、统计建模、随机过程分析，**优先使用对数收益率** |
| **转换关系** | 简单 $\to$ 对数: $r_{\log} = \ln(1 + R_{\text{simple}})$; 对数 $\to$ 简单: $R_{\text{simple}} = e^{r_{\log}} - 1$ |

---

## 5.6 参考文献

1. Grinold, R. C., & Kahn, R. N. (1999). *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk* (2nd ed.). McGraw-Hill.（《主动投资组合管理》第2章：收益率的度量）
2. Hull, J. C. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.（第4章：利率与连续复利）
3. Tsay, R. S. (2010). *Analysis of Financial Time Series* (3rd ed.). Wiley.（第1章：金融数据的特征与对数收益率）
4. 茆诗松, 程依明, 濮晓龙. (2011). 《概率论与数理统计教程》. 高等教育出版社.（指数分布与对数正态分布的数学基础）

---

> **下一章预告**：第6章《级数与泰勒展开——近似计算的核武器》。我们将看到如何用多项式"模仿"复杂的指数和对数函数，以及为什么 Black-Scholes 公式在临近到期时可以用线性近似快速估算。
