# 附录A: Python 数值计算生态指南

> 本附录汇总量化金融中最常用的 Python 库及其核心用法. 面向已有 Python 基础、需要快速上手数值计算的读者.

---

## A.1 NumPy: 一切数值计算的基石

### 向量化运算——不要写 for 循环

NumPy 的核心哲学是**向量化(Vectorization)**: 对整个数组的运算在 C 层面完成, 比 Python for 循环快 10-100 倍.

```python
import numpy as np

# 错误: Python for 循环
prices = [100, 101, 102.5, 103]
returns = []
for i in range(1, len(prices)):
    returns.append(prices[i] / prices[i-1] - 1)

# 正确: NumPy 向量化
prices = np.array([100, 101, 102.5, 103])
returns = np.diff(prices) / prices[:-1]
```

### Broadcasting 规则——不同形状数组的运算

Broadcasting 是 NumPy 最强大也最容易出错的功能. 规则: **从最后一维开始对齐, 维度为 1 的自动扩展到匹配**.

```python
# 每只股票的日收益率 (shape: 252, 3) 减去各自的均值 (shape: 3,)
returns = np.random.normal(0.001, 0.02, (252, 3))  # 3只股票, 252天
mean_rets = returns.mean(axis=0)                     # shape: (3,)
centered = returns - mean_rets                       # broadcasting: (252,3) - (3,)
```

### 常用函数速查

| 函数 | 用途 | 示例 |
|------|------|------|
| `np.dot(a, b)` 或 `a @ b` | 矩阵乘法 | `port_var = w @ cov @ w` |
| `np.diag(m)` | 提取对角线(方差) | `vols = np.sqrt(np.diag(cov))` |
| `np.linalg.inv(m)` | 矩阵求逆 | 最小方差组合权重 |
| `np.linalg.eigh(m)` | 特征分解(对称矩阵) | PCA 分析 |
| `np.cumsum(a)` | 累积和 | 价格 = 100 * exp(cumsum(对数收益)) |
| `np.cumprod(1+a)` | 累积乘积 | 价格 = 100 * cumprod(1+简单收益) |
| `np.percentile(a, q)` | 分位数 | VaR 计算 |
| `np.where(cond, x, y)` | 条件选择 | `signal = np.where(ret > 0, 1, -1)` |

---

## A.2 Pandas: 金融数据的标准容器

### 时间序列对齐——Pandas 的核心价值

金融数据最麻烦的是**对齐**: 不同资产的交易日历可能不同. Pandas 的 `DataFrame` 自动按索引对齐:

```python
import pandas as pd
import numpy as np

# 贵州茅台 (A股, 休市日与港股不同)
mt = pd.Series([0.01, -0.02, 0.015], index=pd.to_datetime(['2024-01-02','2024-01-03','2024-01-04']))
# 腾讯 (港股, 休市日不同)
tx = pd.Series([0.02, 0.01], index=pd.to_datetime(['2024-01-02','2024-01-04']))
# 放到一起自动对齐
df = pd.DataFrame({'茅台': mt, '腾讯': tx})
# NaN 出现在不对齐的日期
df = df.dropna()  # 只保留两个市场都交易的日期
```

### 常用操作

```python
# 收益率计算
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['simple_ret'] = df['close'].pct_change()

# 滚动统计 (需要 .rolling() 后接聚合函数)
df['vol_20d'] = df['log_ret'].rolling(20).std() * np.sqrt(252)

# 重采样 (日线 → 周线)
weekly = df['close'].resample('W').last()
weekly_ret = weekly.pct_change()

# groupby 分析
df['month'] = df.index.month
monthly_avg = df.groupby('month')['log_ret'].mean()
```

### 性能提示

- `df.iterrows()` 极慢, 不要用它遍历
- 优先用 `.rolling()`, `.resample()`, `.groupby()` 等内置方法
- 需要跨行计算时用 `.shift()`, `.diff()`, `.pct_change()`
- 数值密集型运算转 `df.values` 获得 NumPy 数组再用向量化

---

## A.3 Matplotlib: 金融可视化

### 常用图表模板

```python
import matplotlib.pyplot as plt

# 全局字体设置 (量化中文教程标配)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 价格走势图
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df['close'], lw=1.2, color='#2196F3')
ax.set_title('价格走势', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 收益率直方图
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(returns, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.set_title('收益率分布', fontsize=14, fontweight='bold')

# 子图网格
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].plot(...)  # 左上
axes[0,1].hist(...)  # 右上
axes[1,0].scatter(...)  # 左下
axes[1,1].bar(...)  # 右下
plt.tight_layout()  # 自动调整间距
```

### 常用配色

```python
COLORS = {
    'blue':   '#2196F3',  # 主数据系列
    'red':    '#E91E63',  # 对比系列 / 风险
    'green':  '#4CAF50',  # 正收益 / 多头
    'orange': '#FF9800',  # 关键标记点
    'gray':   '#999999',  # 参考线
}
```

---

## A.4 SciPy: 科学计算的瑞士军刀

### 统计模块 (`scipy.stats`)

```python
from scipy import stats

# 分布拟合与检验
nu, loc, scale = stats.t.fit(returns)           # t 分布拟合
jb_stat, jb_p = stats.jarque_bera(returns)       # 正态性检验
ks_stat, ks_p = stats.kstest(returns, 'norm')    # KS 检验

# 分位数 (VaR 计算)
var_95 = stats.norm.ppf(0.05, mu, sigma)         # 参数法 VaR
var_95_t = stats.t.ppf(0.05, nu, loc, scale)     # t 分布 VaR

# 概率密度
pdf_vals = stats.norm.pdf(x_grid, mu, sigma)
cdf_vals = stats.norm.cdf(x_grid, mu, sigma)

# 线性回归
beta, alpha, r_value, p_value, std_err = stats.linregress(x, y)
```

### 优化模块 (`scipy.optimize`)

```python
from scipy.optimize import minimize

# 最小方差组合 (无约束)
def portfolio_var(w, cov):
    return w @ cov @ w

result = minimize(portfolio_var, x0=np.ones(n)/n, args=(cov,),
                  constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
w_optimal = result.x
```

### 积分模块 (`scipy.integrate`)

```python
from scipy import integrate

# 数值积分 (期权定价)
price, error = integrate.quad(lambda x: payoff(x) * pdf(x), lower, upper)
```

---

## A.5 其他常用库速览

| 库 | 用途 | 本书出现章节 |
|----|------|------------|
| `statsmodels` | 回归分析 (OLS, ARIMA), 统计检验 | 第 12, 19-22 章 |
| `sklearn` | 机器学习, PCA, GMM, 收缩估计 | 第 27-29 章, 第 8-9 章 |
| `arch` | GARCH 波动率建模 | 第 21 章 |
| `cvxpy` | 凸优化 (带约束组合优化) | 第 18 章 |
| `jax` | 自动微分 | 第 3 章 |
| `sympy` | 符号数学 (公式推导验证) | 各章辅助 |
| `yfinance` | Yahoo Finance 数据下载 | 第 8 章(替代为本地 CSV) |
| `akshare` | A 股数据下载 | 第 8 章(替代为本地 CSV) |

---

## A.6 性能优化路线图

当你的回测或模拟变慢时:

1. **首先**: 确保用 NumPy 向量化, 消除所有 Python for 循环
2. **其次**: 用 Pandas 内置方法代替 apply/lambda
3. **然后**: 用 `numba` 的 `@jit` 装饰器加速热点函数
4. **最后**: 多进程并行 (`joblib.Parallel`) 处理相互独立的蒙特卡洛路径
