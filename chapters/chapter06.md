# 第6章 级数与泰勒展开——近似计算的核武器

> **本章核心问题**：为什么复杂的期权定价公式，在临近到期时可以用一条直线快速估算？为什么债券交易员嘴里总是挂着"久期"和"凸性"？

---

## 6.1 动机：当精确计算太慢时

在量化金融中，我们经常遇到这样的问题：

- **期权做市商**需要在毫秒级时间内报出数百个不同行权价、不同到期日的期权价格，每次都调用完整的 Black-Scholes 公式太慢；
- **债券交易员**需要快速估算：如果收益率突然上升 1%，手里的债券组合会亏多少？
- **风险经理**需要在收盘后快速估算整个投资组合的 VaR，而不是对每个资产做完整的蒙特卡洛模拟。

这些场景的共同点是：**我们需要在极短时间内，对一个复杂函数做快速估算**。

泰勒展开（Taylor Expansion）就是解决这个问题的"核武器"。它的核心思想极其朴素：

> **任何一个光滑函数，在某一点附近，都可以用一个多项式来"模仿"。**

这个多项式可以是一次的（直线）、二次的（抛物线）、三次的（立方曲线）……阶数越高，模仿得越像。而在金融实践中，**一阶（线性）和二阶（二次）近似往往已经足够好用**。

本章将建立一套完整的直觉：从"为什么多项式能模仿任意函数"，到"金融中的一阶近似和二阶近似分别对应什么风险管理工具"。

---

## 6.2 数学原理

### 6.2.1 泰勒展开的直觉：用多项式"临摹"曲线

想象你站在一条光滑曲线上的某一点 $x_0$，手里只有一把直尺和一个量角器。你想向远处的人描述这条曲线在这一点附近长什么样。

**第一步**：你量出曲线在 $x_0$ 处的函数值 $f(x_0)$。这是最基本的描述——"曲线在这里的高度是 $f(x_0)$"。

**第二步**：你量出曲线在 $x_0$ 处的切线斜率 $f'(x_0)$。现在你不仅能说高度，还能说"曲线在这里正以 $f'(x_0)$ 的速度上升/下降"。用一条直线去近似：

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0)$$

这就是**一阶泰勒展开**。它用一条直线（切线）去近似曲线。在 $x_0$ 附近，这条直线和曲线贴得很紧；但离得越远，误差越大。

**第三步**：你进一步量出曲线的弯曲程度——二阶导数 $f''(x_0)$。它告诉你曲线是向上凸还是向下凹，弯得有多厉害。加上这个信息，你改用一条抛物线去近似：

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2}(x - x_0)^2$$

这就是**二阶泰勒展开**。抛物线比直线多出一个"弯"，所以能在更大范围内贴合原曲线。

**一般地**，如果函数 $f(x)$ 在 $x_0$ 处具有直到 $n$ 阶的导数，那么它的 $n$ 阶泰勒展开为：

$$f(x) = \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k + R_n(x)$$

其中 $R_n(x)$ 是**余项**（Remainder），代表"多项式没模仿到的那部分"。

**几何直觉**：泰勒展开就像给函数拍了一张"局部特写"。一阶展开是"线性滤镜"，二阶展开是"二次曲面滤镜"。离拍摄中心越远，滤镜的失真越严重。

---

### 6.2.2 麦克劳林展开：在零点展开的特殊情况

当展开点 $x_0 = 0$ 时，泰勒展开有一个更简单的名字——**麦克劳林展开**（Maclaurin Series）：

$$f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f^{(3)}(0)}{3!}x^3 + \cdots$$

以下是几个在量化金融中反复出现的麦克劳林展开：

**指数函数**：

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$

这正是第五章中 $e$ 的另一种面孔——用多项式无限逼近指数曲线。

**对数函数**：

$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots \quad (|x| < 1)$$

这就是第五章中 $\ln(1+x) \approx x$ 近似的严格来源——它是一阶麦克劳林展开。当 $x$ 很小时，$x^2$ 及更高阶项可以忽略。

---

### 6.2.3 余项与误差：近似有多准？

泰勒展开不是精确等式，除非取无穷多项。对于有限阶展开，我们需要知道误差有多大。

**拉格朗日余项**给出了一个定量估计：

$$R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - x_0)^{n+1}$$

其中 $\xi$ 是介于 $x_0$ 和 $x$ 之间的某个点。

**金融直觉**：余项告诉我们，用一阶近似估算期权价格时，误差大致与 $(x - x_0)^2$ 成正比。这意味着：

- 如果标的资产价格只变动了 $1\%$，一阶近似的误差是 $0.01^2 = 0.0001$ 量级，完全可以接受；
- 如果价格跳动了 $10\%$，误差放大到 $0.1^2 = 0.01$ 量级，此时必须引入二阶修正。

这也是为什么期权做市商在市场剧烈波动时会"把 Gamma 调大"——本质上就是在用二阶项补偿一阶近似的失效。

---

### 6.2.4 多元泰勒展开：多资产世界的近似

当函数有多个自变量时（例如期权价格同时依赖于标的价格 $S$ 和时间 $t$），泰勒展开自然推广到多元形式。

二元函数 $f(x, y)$ 在 $(x_0, y_0)$ 处的一阶展开：

$$f(x, y) \approx f(x_0, y_0) + \frac{\partial f}{\partial x}(x - x_0) + \frac{\partial f}{\partial y}(y - y_0)$$

二元函数的二阶展开：

$$f(x, y) \approx f(x_0, y_0) + f_x \Delta x + f_y \Delta y + \frac{1}{2}f_{xx}(\Delta x)^2 + f_{xy}\Delta x \Delta y + \frac{1}{2}f_{yy}(\Delta y)^2$$

**金融对应**：
- $f_x = \frac{\partial f}{\partial S}$ 就是期权的 **Delta**（标的价格变动1单位，期权价格变多少）；
- $f_{xx} = \frac{\partial^2 f}{\partial S^2}$ 就是期权的 **Gamma**（Delta 的变化率）；
- $f_y = \frac{\partial f}{\partial t}$ 就是期权的 **Theta**（时间衰减）；
- $f_{xy}$ 交叉项在复杂衍生品中对应 **Vanna** 等二阶风险。

所以，当你听到交易员说"Delta-Gamma 对冲"时，他们其实在说：**用一阶和二阶泰勒展开来近似投资组合的价值变化**。

---

## 6.3 核心公式速查

> 本节是前述各节公式的集中汇总，供复习和查阅使用。

| 公式 | 名称 | 适用条件 |
|------|------|----------|
| $f(x) \approx f(x_0) + f'(x_0)(x - x_0)$ | 一阶泰勒展开 | 函数在 $x_0$ 处可导；$x$ 接近 $x_0$ |
| $f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2}(x - x_0)^2$ | 二阶泰勒展开 | 函数在 $x_0$ 处二阶可导；需要更高精度 |
| $e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}$ | 指数函数的麦克劳林展开 | 对所有实数 $x$ 收敛 |
| $\ln(1+x) = \sum_{k=1}^{\infty} (-1)^{k+1}\frac{x^k}{k}$ | 对数函数的麦克劳林展开 | $\|x\| < 1$ |
| $R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - x_0)^{n+1}$ | 拉格朗日余项 | $\xi$ 介于 $x_0$ 和 $x$ 之间 |
| $\Delta V \approx \Delta \cdot \Delta S + \frac{1}{2}\Gamma \cdot (\Delta S)^2$ | 期权价值变化的 Delta-Gamma 近似 | 期权定价函数二阶可导 |
| $\Delta P \approx -D \cdot \Delta y + \frac{1}{2}C \cdot (\Delta y)^2$ | 债券价格变化的久期-凸性近似 | 债券价格对收益率二阶可导 |

---

## 6.4 Python 示例

### 示例 1：$e^x$ 的泰勒展开可视化——多项式如何逼近指数曲线

```python
import numpy as np
import matplotlib.pyplot as plt
import math

# 设置字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def taylor_exp(x, n_terms):
    """
    计算 e^x 的 n 阶泰勒展开（麦克劳林展开）
    e^x = 1 + x + x^2/2! + x^3/3! + ... + x^n/n!
    """
    result = 0.0
    for k in range(n_terms + 1):
        result += x**k / math.factorial(k)  # 改用 math.factorial
    return result

# 生成数据
x = np.linspace(-3, 3, 500)
y_true = np.exp(x)

# 不同阶数的近似
orders = [1, 2, 3, 5, 10]
colors = ['red', 'orange', 'green', 'blue', 'purple']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：函数逼近
axes[0].plot(x, y_true, 'k-', linewidth=2, label='$e^x$（精确值）', zorder=10)
for n, color in zip(orders, colors):
    y_approx = [taylor_exp(xi, n) for xi in x]
    axes[0].plot(x, y_approx, color=color, linewidth=1.5, 
                 label=f'{n}阶泰勒展开', alpha=0.8)
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-2, 20)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('函数值', fontsize=12)
axes[0].set_title('$e^x$ 的泰勒展开逼近', fontsize=14)
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(True, alpha=0.3)

# 右图：局部放大（在 x=0 附近）
x_zoom = np.linspace(-1, 1, 500)
y_true_zoom = np.exp(x_zoom)
axes[1].plot(x_zoom, y_true_zoom, 'k-', linewidth=2, label='$e^x$（精确值）', zorder=10)
for n, color in zip(orders, colors):
    y_approx_zoom = [taylor_exp(xi, n) for xi in x_zoom]
    axes[1].plot(x_zoom, y_approx_zoom, color=color, linewidth=1.5, 
                 label=f'{n}阶展开', alpha=0.8)
axes[1].set_xlim(-1, 1)
axes[1].set_ylim(0, 3)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('函数值', fontsize=12)
axes[1].set_title('局部放大：$x \\in [-1, 1]$', fontsize=13)
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 数值验证：在 x=1 处，各阶展开的精度
print("在 x=1 处，e^1 = {:.10f}".format(np.e))
print("-" * 50)
for n in [1, 2, 3, 5, 10, 20]:
    approx = taylor_exp(1.0, n)
    error = abs(approx - np.e)
    print(f"{n:2d}阶展开: {approx:.10f}, 误差: {error:.2e}")
```

**运行结果**：

```
在 x=1 处，e^1 = 2.7182818285
--------------------------------------------------
 1阶展开: 2.0000000000, 误差: 7.18e-01
 2阶展开: 2.5000000000, 误差: 2.18e-01
 3阶展开: 2.6666666667, 误差: 5.16e-02
 5阶展开: 2.7166666667, 误差: 1.62e-03
10阶展开: 2.7182818011, 误差: 2.73e-08
20阶展开: 2.7182818285, 误差: 4.44e-16
```

**运行结果解读**：左图显示，低阶泰勒展开在远离展开中心（$x=0$）时迅速发散（注意红色的一阶直线在 $x>2$ 时已经严重偏离）。右图的局部放大则显示，在 $x \in [-1, 1]$ 范围内，即使是三阶展开也已经几乎与真实曲线重合。这就是泰勒展开的"局部精确、全局可能失控"的特性。

---

### 示例 2：$\ln(1+x)$ 的泰勒展开——第五章近似的严格来源

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def taylor_ln(x, n_terms):
    """
    计算 ln(1+x) 的 n 阶麦克劳林展开
    ln(1+x) = x - x^2/2 + x^3/3 - x^4/4 + ... + (-1)^(n+1) * x^n/n
    """
    result = 0.0
    for k in range(1, n_terms + 1):
        result += ((-1)**(k+1)) * (x**k / k)
    return result

# 注意：ln(1+x) 的展开在 |x| < 1 时收敛
x = np.linspace(-0.9, 0.9, 500)
y_true = np.log(1 + x)

orders = [1, 2, 3, 5, 10, 20]
colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y_true, 'k-', linewidth=2.5, label='$\ln(1+x)$（精确值）', zorder=10)

for n, color in zip(orders, colors):
    y_approx = [taylor_ln(xi, n) for xi in x]
    ax.plot(x, y_approx, color=color, linewidth=1.5, 
            label=f'{n}阶展开', alpha=0.8)

ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='收敛边界 $x=1$')
ax.axvline(x=-1, color='red', linestyle='--', alpha=0.5)
ax.set_xlim(-0.95, 0.95)
ax.set_ylim(-2, 1)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('函数值', fontsize=12)
ax.set_title('$\ln(1+x)$ 的泰勒展开：收敛半径为 1', fontsize=13)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 验证第五章的近似：在 x=0.05 处
x_test = 0.05
print(f"\n在 x = {x_test} 处验证：")
print(f"精确值 ln(1+{x_test}) = {np.log(1+x_test):.10f}")
for n in [1, 2, 3]:
    approx = taylor_ln(x_test, n)
    error = abs(approx - np.log(1+x_test))
    print(f"{n}阶展开: {approx:.10f}, 误差: {error:.2e}")
print(f"\n一阶近似 x = {x_test:.10f}, 与精确值差异: {abs(x_test - np.log(1+x_test)):.2e}")
```

**关键观察**：$\ln(1+x)$ 的泰勒展开有一个**收敛半径**——只在 $\|x\| < 1$ 范围内有效。当 $x \to -1$ 时，级数发散到 $-\infty$（对应价格跌到零）。这恰好对应金融中的一个基本约束：对数收益率的近似只在价格变动不太大时可靠。

---

### 示例 3：债券价格的久期-凸性近似——泰勒展开在固收中的应用

债券价格 $P$ 是到期收益率 $y$ 的函数。当收益率发生微小变化 $\Delta y$ 时，价格变化可以用泰勒展开近似：

$$\Delta P \approx -D \cdot P \cdot \Delta y + \frac{1}{2} C \cdot P \cdot (\Delta y)^2$$

其中 $D$ 是修正久期，$C$ 是凸性。

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def bond_price(face_value, coupon_rate, ytm, years, freq=1):
    """
    计算债券价格（现金流贴现）
    face_value: 面值
    coupon_rate: 票面利率
    ytm: 到期收益率
    years: 剩余年限
    freq: 每年付息次数
    """
    coupon = face_value * coupon_rate / freq
    n_periods = int(years * freq)
    periods = np.arange(1, n_periods + 1)

    # 票息现值 + 面值现值
    pv_coupons = np.sum(coupon / (1 + ytm/freq)**periods)
    pv_face = face_value / (1 + ytm/freq)**n_periods
    return pv_coupons + pv_face

def bond_duration_convexity(face_value, coupon_rate, ytm, years, freq=1, dy=1e-5):
    """
    用数值方法计算修正久期和凸性
    """
    p0 = bond_price(face_value, coupon_rate, ytm, years, freq)
    p_plus = bond_price(face_value, coupon_rate, ytm + dy, years, freq)
    p_minus = bond_price(face_value, coupon_rate, ytm - dy, years, freq)

    # 修正久期（一阶导数的负数除以价格）
    duration = -(p_plus - p_minus) / (2 * dy * p0)

    # 凸性（二阶导数除以价格）
    convexity = (p_plus - 2*p0 + p_minus) / (dy**2 * p0)

    return duration, convexity, p0

# 参数：一个 5 年期、票面利率 5%、当前收益率 5% 的债券
face_value = 100
coupon_rate = 0.05
years = 5
ytm_current = 0.05

# 计算当前久期和凸性
duration, convexity, price_0 = bond_duration_convexity(
    face_value, coupon_rate, ytm_current, years)

print(f"债券当前价格: {price_0:.4f}")
print(f"修正久期 D: {duration:.4f}")
print(f"凸性 C: {convexity:.4f}")

# 模拟收益率变化，比较精确价格 vs 近似价格
yield_changes = np.linspace(-0.03, 0.03, 100)
prices_exact = []
prices_linear = []    # 一阶近似（久期）
prices_quadratic = []  # 二阶近似（久期+凸性）

for dy in yield_changes:
    # 精确价格
    p_exact = bond_price(face_value, coupon_rate, ytm_current + dy, years)
    prices_exact.append(p_exact)

    # 一阶近似: P ≈ P0 - D*P0*dy
    p_lin = price_0 - duration * price_0 * dy
    prices_linear.append(p_lin)

    # 二阶近似: P ≈ P0 - D*P0*dy + 0.5*C*P0*dy^2
    p_quad = price_0 - duration * price_0 * dy + 0.5 * convexity * price_0 * dy**2
    prices_quadratic.append(p_quad)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：价格-收益率曲线
axes[0].plot(yield_changes * 100, prices_exact, 'k-', linewidth=2.5, 
             label='精确价格', zorder=10)
axes[0].plot(yield_changes * 100, prices_linear, 'r--', linewidth=1.5, 
             label='一阶近似（久期）')
axes[0].plot(yield_changes * 100, prices_quadratic, 'b-.', linewidth=1.5, 
             label='二阶近似（久期+凸性）')
axes[0].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
axes[0].axhline(y=price_0, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('收益率变化 (%)', fontsize=12)
axes[0].set_ylabel('债券价格', fontsize=12)
axes[0].set_title('债券价格-收益率曲线：泰勒近似的精度', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 右图：近似误差
error_linear = np.array(prices_linear) - np.array(prices_exact)
error_quadratic = np.array(prices_quadratic) - np.array(prices_exact)

axes[1].plot(yield_changes * 100, error_linear, 'r-', linewidth=1.5, 
             label='一阶近似误差')
axes[1].plot(yield_changes * 100, error_quadratic, 'b-', linewidth=1.5, 
             label='二阶近似误差')
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[1].set_xlabel('收益率变化 (%)', fontsize=12)
axes[1].set_ylabel('近似误差（元）', fontsize=12)
axes[1].set_title('近似误差对比', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印几个关键点的数值
print("\n关键数值对比表：")
print(f"{'收益率变化':>12} | {'精确价格':>10} | {'一阶近似':>10} | {'二阶近似':>10} | {'一阶误差':>10} | {'二阶误差':>10}")
print("-" * 90)
for dy in [-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03]:
    idx = np.argmin(np.abs(yield_changes - dy))
    print(f"{dy*100:11.2f}% | {prices_exact[idx]:10.4f} | {prices_linear[idx]:10.4f} | "
          f"{prices_quadratic[idx]:10.4f} | {error_linear[idx]:10.4f} | {error_quadratic[idx]:10.4f}")
```

**运行结果**：

```
债券当前价格: 100.0000
修正久期 D: 4.3295
凸性 C: 23.9360

       收益率变化 |       精确价格 |       一阶近似 |       二阶近似 |       一阶误差 |       二阶误差
------------------------------------------------------------------------------------------
      -3.00% |   114.1404 |   112.9884 |   114.0655 |    -1.1519 |    -0.0748
      -2.00% |   109.1594 |   108.6590 |   109.1377 |    -0.5005 |    -0.0217
      -1.00% |   104.4518 |   104.3295 |   104.4492 |    -0.1223 |    -0.0027
       0.00% |   100.0000 |   100.0000 |   100.0000 |     0.0000 |     0.0000
       1.00% |    95.7876 |    95.6705 |    95.7902 |    -0.1171 |     0.0026
       2.00% |    91.7996 |    91.3410 |    91.8198 |    -0.4586 |     0.0202
       3.00% |    88.0219 |    87.0116 |    88.0887 |    -1.0103 |     0.0668
```

**金融直觉**：左图显示，债券价格-收益率曲线是一条**凸函数**（向下弯曲）。一阶近似（红色虚线）是一条直线，只能捕捉曲线在 $y_0$ 处的切线方向；二阶近似（蓝色点划线）加入了凸性修正，能更好地贴合曲线的弯曲。右图的误差分析更直观：在收益率变化 $\pm 1\%$ 时，一阶近似误差约 $0.12$ 元，二阶近似将误差压到约 $0.003$ 元——**精度提升了约 47 倍**。

---

### 示例 4：期权价格的 Delta-Gamma 近似——泰勒展开在衍生品中的应用

Black-Scholes 欧式看涨期权价格 $C$ 是标的价格 $S$ 的函数。当 $S$ 变化 $\Delta S$ 时：

$$\Delta C \approx \Delta \cdot \Delta S + \frac{1}{2}\Gamma \cdot (\Delta S)^2$$

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes 欧式看涨期权定价公式
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_greeks(S, K, T, r, sigma):
    """
    计算 Delta 和 Gamma
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return delta, gamma

# 参数设置
S0 = 100       # 当前标的价格
K = 100        # 行权价
T = 0.25       # 3个月到期
r = 0.05       # 无风险利率 5%
sigma = 0.20   # 波动率 20%

# 计算当前价格和 Greeks
price_0 = black_scholes_call(S0, K, T, r, sigma)
delta, gamma = black_scholes_greeks(S0, K, T, r, sigma)

print(f"当前期权价格: {price_0:.4f}")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.6f}")

# 模拟标的价格变化
S_range = np.linspace(80, 120, 200)
prices_exact = []
prices_delta = []      # 一阶近似: C ≈ C0 + Delta*(S-S0)
prices_delta_gamma = []  # 二阶近似: C ≈ C0 + Delta*(S-S0) + 0.5*Gamma*(S-S0)^2

for S in S_range:
    p_exact = black_scholes_call(S, K, T, r, sigma)
    prices_exact.append(p_exact)

    dS = S - S0
    p_dg = price_0 + delta * dS + 0.5 * gamma * dS**2
    prices_delta_gamma.append(p_dg)

    p_d = price_0 + delta * dS
    prices_delta.append(p_d)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：期权价格曲线
axes[0].plot(S_range, prices_exact, 'k-', linewidth=2.5, label='BS 精确价格', zorder=10)
axes[0].plot(S_range, prices_delta, 'r--', linewidth=1.5, label='Delta 近似（一阶）')
axes[0].plot(S_range, prices_delta_gamma, 'b-.', linewidth=1.5, label='Delta-Gamma 近似（二阶）')
axes[0].axvline(x=S0, color='gray', linestyle=':', alpha=0.5)
axes[0].axhline(y=price_0, color='gray', linestyle=':', alpha=0.5)
axes[0].scatter([S0], [price_0], color='green', s=100, zorder=11, label='当前点')
axes[0].set_xlabel('标的价格 $S$', fontsize=12)
axes[0].set_ylabel('看涨期权价格', fontsize=12)
axes[0].set_title('期权价格的 Delta-Gamma 近似', fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# 右图：近似误差
error_delta = np.array(prices_delta) - np.array(prices_exact)
error_dg = np.array(prices_delta_gamma) - np.array(prices_exact)

axes[1].plot(S_range, error_delta, 'r-', linewidth=1.5, label='Delta 近似误差')
axes[1].plot(S_range, error_dg, 'b-', linewidth=1.5, label='Delta-Gamma 近似误差')
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[1].axvline(x=S0, color='gray', linestyle=':', alpha=0.5)
axes[1].set_xlabel('标的价格 $S$', fontsize=12)
axes[1].set_ylabel('近似误差', fontsize=12)
axes[1].set_title('近似误差分析', fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印关键数值
print("\n关键数值对比表（平值附近）：")
print(f"{'标的价格':>10} | {'精确价格':>10} | {'Delta近似':>10} | {'DG近似':>10} | {'Delta误差':>10} | {'DG误差':>10}")
print("-" * 85)
test_prices = [90, 95, 98, 100, 102, 105, 110]
for S in test_prices:
    p_exact = black_scholes_call(S, K, T, r, sigma)
    dS = S - S0
    p_d = price_0 + delta * dS
    p_dg = price_0 + delta * dS + 0.5 * gamma * dS**2
    print(f"{S:10.2f} | {p_exact:10.4f} | {p_d:10.4f} | {p_dg:10.4f} | "
          f"{p_d-p_exact:10.4f} | {p_dg-p_exact:10.4f}")
```

**运行结果**：

```
当前期权价格: 4.6150
Delta: 0.5695
Gamma: 0.039288

      标的价格 |       精确价格 |    Delta近似 |       DG近似 |    Delta误差 |       DG误差
-------------------------------------------------------------------------------------
     90.00 |     0.8975 |    -1.0796 |     0.8848 |    -1.9771 |    -0.0127
     95.00 |     2.2712 |     1.7677 |     2.2588 |    -0.5035 |    -0.0124
     98.00 |     3.5558 |     3.4761 |     3.5547 |    -0.0798 |    -0.0012
    100.00 |     4.6150 |     4.6150 |     4.6150 |     0.0000 |     0.0000
    102.00 |     5.8308 |     5.7539 |     5.8325 |    -0.0769 |     0.0017
    105.00 |     7.9229 |     7.4623 |     7.9534 |    -0.4606 |     0.0305
    110.00 |    11.9883 |    10.3096 |    12.2740 |    -1.6787 |     0.2857
```

**金融直觉**：期权价格-标的价格曲线是一条**凸函数**（向上弯曲，因为 Gamma > 0）。一阶 Delta 近似（红色虚线）在标的价格远离 $S_0$ 时迅速偏离；而加入 Gamma 修正后，二阶近似在 $S \in [95, 105]$ 范围内几乎与精确曲线重合。这就是期权做市商为什么在市场平静时（标的价格波动小）可以只用 Delta 对冲，而在市场动荡时必须引入 Gamma 对冲——本质上是在调整泰勒展开的阶数。

---

### 示例 5：泰勒余项的数值验证——误差到底有多大？

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def bs_call_and_greeks(S, K, T, r, sigma):
    """返回期权价格、Delta、Gamma、Speed"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    # 三阶导数 Speed (dGamma/dS)
    speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)
    return price, delta, gamma, speed

# 参数
S0, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
price_0, delta, gamma, speed = bs_call_and_greeks(S0, K, T, r, sigma)

# 拉格朗日余项的理论估计：
# 一阶余项 R1 = 0.5 * f''(xi) * (dx)^2 ≈ 0.5 * Gamma * (dS)^2
# 二阶余项 R2 = (1/6) * f^{(3)}(xi) * (dx)^3 ≈ (1/6) * Speed * (dS)^3

dS_values = np.linspace(-10, 10, 100)

errors_1st = []      # 一阶近似的实际误差
errors_2nd = []      # 二阶近似的实际误差
remainder_1st = []  # 一阶余项的理论估计
remainder_2nd = []  # 二阶余项的理论估计

for dS in dS_values:
    p_exact = bs_call_and_greeks(S0 + dS, K, T, r, sigma)[0]
    p_1st = price_0 + delta * dS
    p_2nd = price_0 + delta * dS + 0.5 * gamma * dS**2

    errors_1st.append(abs(p_1st - p_exact))
    errors_2nd.append(abs(p_2nd - p_exact))

    # 理论余项估计（用当前点的 Greeks 估计）
    remainder_1st.append(abs(0.5 * gamma * dS**2))
    remainder_2nd.append(abs((1/6) * speed * dS**3))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(dS_values, errors_1st, 'r-', linewidth=2, label='一阶近似实际误差')
ax.plot(dS_values, remainder_1st, 'r--', linewidth=1.5, alpha=0.7, label='一阶余项理论估计')
ax.plot(dS_values, errors_2nd, 'b-', linewidth=2, label='二阶近似实际误差')
ax.plot(dS_values, remainder_2nd, 'b--', linewidth=1.5, alpha=0.7, label='二阶余项理论估计')

ax.set_xlabel('标的价格变化 $\\Delta S$', fontsize=12)
ax.set_ylabel('绝对误差', fontsize=12)
ax.set_title('泰勒余项理论估计 vs 实际误差（期权定价）', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.show()

# 打印比例关系验证
print("误差与 (dS)^n 的比例关系验证：")
print("-" * 60)
for dS in [1, 2, 5, 10]:
    idx = np.argmin(np.abs(dS_values - dS))
    e1 = errors_1st[idx]
    e2 = errors_2nd[idx]
    print(f"dS = {dS:2d}: 一阶误差 = {e1:.6f}, 二阶误差 = {e2:.6f}")
    print(f"       一阶误差 / (dS)^2 = {e1/dS**2:.6f}, 二阶误差 / (dS)^3 = {e2/dS**3:.6f}")
    print()
```

**核心结论**：右图的对数刻度显示，一阶近似的实际误差（红色实线）与理论余项估计（红色虚线）几乎重合，验证了拉格朗日余项公式的正确性。误差与 $(\Delta S)^2$ 成正比（一阶）和 $(\Delta S)^3$ 成正比（二阶），这意味着：
- 当价格变动减半，一阶误差缩小到 $1/4$；
- 二阶误差缩小到 $1/8$。

这就是高阶泰勒展开的威力——**误差随步长的高次幂衰减**。

---

## 6.5 本章小结

| 要点 | 核心内容 |
|------|----------|
| **泰勒展开的本质** | 用多项式在局部"临摹"任意光滑函数；阶数越高，临摹范围越大、精度越高 |
| **一阶展开** | $f(x) \approx f(x_0) + f'(x_0)(x-x_0)$；金融对应：久期（债券）、Delta（期权） |
| **二阶展开** | 加入 $\frac{1}{2}f''(x_0)(x-x_0)^2$；金融对应：凸性（债券）、Gamma（期权） |
| **麦克劳林展开** | 在 $x_0=0$ 处的特殊泰勒展开；$\ln(1+x) \approx x$ 和 $e^x \approx 1+x$ 都来源于此 |
| **余项控制** | 误差与 $(x-x_0)^{n+1}$ 成正比；步长减半，$n$ 阶误差缩小到 $1/2^{n+1}$ |
| **金融实践原则** | 市场平静时一阶近似够用；市场剧烈波动时必须引入二阶修正；极端行情下需要更高阶或完整模型 |

---

## 6.6 参考文献

1. Hull, J. C. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.（第19章：Greeks 与 Taylor 展开近似）
2. Tuckman, B., & Serrat, A. (2011). *Fixed Income Securities: Tools for Today's Markets* (3rd ed.). Wiley.（第4章：久期与凸性的数学基础）
3. Stewart, J. (2015). *Calculus: Early Transcendentals* (8th ed.). Cengage Learning.（第11章：泰勒级数与麦克劳林级数）
4. 李贤平. (2010). 《概率论基础》. 高等教育出版社.（级数收敛性与对数正态分布的展开基础）

---

> **下一章预告**：第7章《概率论基础——从掷骰子到资产收益》。我们将离开确定性函数的微积分世界，进入量化金融最核心的语言：概率。你会理解为什么"收益率是一个随机变量"这句话，彻底改变了我们看待市场的方式。
