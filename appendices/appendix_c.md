# 附录C: 数据集获取与实验环境搭建

> 本附录提供量化金融教学与实践所需的数据源、环境配置及常用回测框架的速查指南.

---

## C.1 Conda 环境搭建

本书所有代码基于 `maths-in-quant` conda 环境. 完整依赖见项目根目录的 `environment.yaml`.

```bash
# 1. 创建环境 (约需 3-5 分钟)
conda env create -f environment.yaml

# 2. 激活环境
conda activate maths-in-quant

# 3. 注册 Jupyter 内核
python -m ipykernel install --user --name=maths-in-quant --display-name "Python (maths-in-quant)"

# 4. 验证安装
python -c "import numpy, pandas, scipy, matplotlib, statsmodels, cvxpy, sklearn, sympy, arch; print('All OK')"

# 5. 更新环境 (当 environment.yaml 有变更时)
conda env update -f environment.yaml --prune
```

**字体安装(Ubuntu)**:

```bash
sudo apt update
sudo apt install ttf-wqy-microhei
```

Notebook 内统一设置:

```python
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
```

---

## C.2 本书使用的数据集

### 内置数据集: `notebook/ifind_price_data.csv`

本书第 7 章起使用同花顺 iFind 导出的真实 A 股数据, 包含:

| 字段 | 说明 | 示例 |
|------|------|------|
| `open`, `high`, `low`, `close` | OHLC 价格 | 1701.00 |
| `volume` | 成交量(股) | 124150770 |
| `thscode` | 同花顺代码 | `600519.SH`(茅台), `000300.SH`(沪深300), `300750.SZ`(宁德时代) |
| `time` | 交易日期 | 2023-05-25 |
| `thsname_cn` | 中文名称 | 贵州茅台 |
| `currency` | 货币 | CNY |

**数据区间**: 2023-05-25 ~ 2026-05-22, 约 724 个交易日 / 标的.

**在代码中加载**:

```python
import pandas as pd
import os

csv_path = 'ifind_price_data.csv'
if not os.path.exists(csv_path):
    csv_path = 'notebook/ifind_price_data.csv'
df = pd.read_csv(csv_path, parse_dates=['time'])

# 筛选单只标的
df_mt = df[df['thscode'] == '600519.SH'].sort_values('time')
```

---

## C.3 外部数据源

当需要获取更多或更新的数据时, 以下数据源可供选择:

### Yahoo Finance (`yfinance`)

```bash
pip install yfinance
```

```python
import yfinance as yf

# 下载美股数据
aapl = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
# 下载 A 股 (需加后缀)
csi300 = yf.download('000300.SS', start='2020-01-01')  # 沪深300
```

**局限**: A 股数据覆盖不完整, 部分标的无数据. 本书因此使用本地 CSV 替代.

### AKShare (A 股专用, 免费)

```bash
pip install akshare
```

```python
import akshare as ak

# A 股个股日线
stock_df = ak.stock_zh_a_hist(symbol="600519", period="daily",
                               start_date="20230101", end_date="20241231")

# 指数日线
index_df = ak.stock_zh_index_daily(symbol="sh000300")
```

**优势**: 数据全, 免费, 无需 token. **注意**: API 可能随版本更新而变化.

### Tushare (A 股专用, 需注册 token)

```bash
pip install tushare
```

```python
import tushare as ts
ts.set_token('your_token_here')
pro = ts.pro_api()

df = pro.daily(ts_code='600519.SH', start_date='20230101', end_date='20241231')
```

**优势**: 数据质量高, 覆盖面广(含财务数据, 因子数据). **注意**: 部分接口需要积分.

---

## C.4 回测框架简介

### 为什么需要回测框架

手工写回测循环容易出错: 前视偏差(Look-ahead Bias), 生存偏差(Survivorship Bias), 分红/拆股处理, 滑点和手续费的模拟. 专业回测框架帮你处理这些细节.

### Backtrader (适合教学和小型策略)

```bash
pip install backtrader
```

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=20)

    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()
        elif self.data.close[0] < self.sma[0]:
            self.sell()

cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)
# ... 添加数据, 设置资金, 运行
cerebro.run()
cerebro.plot()
```

**优势**: 文档好, 社区活跃, 事件驱动架构直观. **不足**: 速度较慢(纯 Python), 不适合高频/大规模回测.

### Zipline-Reloaded (Quantopian 遗产)

```bash
pip install zipline-reloaded
```

Zipline 是 Quantopian 开源的回测引擎, 有完整的 Pipeline API 用于因子研究. `zipline-reloaded` 是社区维护的更新版本.

### 自研框架 (推荐进阶路线)

当你理解了回测的基本原理后, 自研框架可以给你最大的灵活性:

```python
# 最简回测循环 (教学用途, 生产环境需要更完善的错误处理)
def simple_backtest(prices, signal, initial_capital=1_000_000):
    """
    prices: DataFrame, 行=日期, 列=资产
    signal: DataFrame, 行=日期, 列=资产, 值∈[-1,1] (-1=满仓做空, 1=满仓做多)
    """
    returns = prices.pct_change().fillna(0)
    strategy_returns = (signal.shift(1) * returns).sum(axis=1)  # 等权组合
    equity_curve = (1 + strategy_returns).cumprod() * initial_capital
    return equity_curve, strategy_returns
```

---

## C.5 数据存储格式建议

| 场景 | 格式 | 原因 |
|------|------|------|
| 少量标的, 日线 | CSV | 人可读, Git 友好 |
| 大量标的, 分钟线 | Parquet | 压缩率高, 列式存储, 读取快 |
| 实时行情 | 数据库(SQLite/Postgres) | 增量写入, 查询灵活 |
| 中间计算结果 | HDF5 | 快速读写 NumPy 数组 |
| 缓存(因子值等) | Pickle / Joblib | Python 对象直接序列化 |

---

## C.6 常见环境问题排查

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'xxx'` | `conda activate maths-in-quant` 确认环境已激活 |
| 中文显示为方块 | 安装文泉驿字体: `sudo apt install ttf-wqy-microhei` |
| Jupyter kernel 未注册 | `python -m ipykernel install --user --name=maths-in-quant` |
| conda 下载慢 | 配置国内镜像: `conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/` |
| CUDA 相关警告(无 GPU 环境) | 忽略. JAX 等库会自动 fallback 到 CPU 模式 |
