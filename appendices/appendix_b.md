# 附录B: 数学符号表与希腊字母速查

> 量化金融中希腊字母无处不在——它们不仅是公式中的符号, 更是交易台和风控部门的通用语言. 本附录提供标准读音, 数学含义及金融应用.

---

## B.1 希腊字母速查表

| 字母 | 大写 | 小写 | 英文名 | 中文读音 | 数学中的常见含义 | 金融中的含义 |
|------|------|------|--------|---------|----------------|-------------|
| Alpha | A | $\alpha$ | alpha | 阿尔法 | 截距, 显著性水平 | **超额收益**(Jensen's Alpha); 第 I 类错误概率 |
| Beta | B | $\beta$ | beta | 贝塔 | 回归系数 | **市场敏感度**; $\beta>1$ 高弹性, $\beta<1$ 防御型 |
| Gamma | $\Gamma$ | $\gamma$ | gamma | 伽马 | Gamma 函数 | **期权 Gamma**(Delta 的变化率); 偏度系数 $\gamma_1$ |
| Delta | $\Delta$ | $\delta$ | delta | 德尔塔 | 微小变化量 | **期权 Delta**(价格对标的的敏感度); 差分算子 |
| Epsilon | E | $\epsilon,\varepsilon$ | epsilon | 艾普西龙 | 任意小的正数; 误差项 | 回归残差; $\varepsilon$-$\delta$ 极限定义 |
| Zeta | Z | $\zeta$ | zeta | 泽塔 | Riemann Zeta 函数 | 较少使用 |
| Eta | H | $\eta$ | eta | 伊塔 | 效率, 学习率 | 较少使用 |
| Theta | $\Theta$ | $\theta$ | theta | 西塔 | 角度; 参数向量 | **期权 Theta**(时间衰减) |
| Iota | I | $\iota$ | iota | 约塔 | 微小量 | 较少使用 |
| Kappa | K | $\kappa$ | kappa | 卡帕 | 曲率; 条件数 | 较少使用 |
| Lambda | $\Lambda$ | $\lambda$ | lambda | 兰姆达 | 特征值; 泊松强度 | **跳跃强度**(泊松过程); 风险价格 |
| Mu | M | $\mu$ | mu | 缪 | 均值(总体) | **期望收益**; 漂移率(GBM) |
| Nu | N | $\nu$ | nu | 纽 | 自由度 | **t 分布自由度**(控制尾部厚度) |
| Xi | $\Xi$ | $\xi$ | xi | 克西 | 随机变量 | 较少使用; 有时用于方差 |
| Omicron | O | $o$ | omicron | 奥密克戎 | 小o记号 | 较少使用 |
| Pi | $\Pi$ | $\pi$ | pi | 派 | 圆周率 $\approx 3.1416$ | 乘积符号 $\prod$; 状态概率 |
| Rho | P | $\rho$ | rho | 柔 | 相关系数 | **相关系数**; $\rho=1$ 完全正相关 |
| Sigma | $\Sigma$ | $\sigma$ | sigma | 西格玛 | 求和; 标准差 | **波动率**; $\Sigma$ 协方差矩阵; $\sigma_p$ 组合波动率 |
| Tau | T | $\tau$ | tau | 陶 | 时间变量 | 到期时间(期权); Kendall's $\tau$ 秩相关 |
| Upsilon | $\Upsilon$ | $\upsilon$ | upsilon | 宇普西龙 | 较少使用 | 较少使用 |
| Phi | $\Phi$ | $\phi,\varphi$ | phi | 斐 | 标准正态 CDF | $\Phi(\cdot)$ 正态 CDF; $\phi(\cdot)$ 正态 PDF |
| Chi | X | $\chi$ | chi | 卡 | 卡方分布 | $\chi^2$ 分布(波动率估计) |
| Psi | $\Psi$ | $\psi$ | psi | 普西 | 较少使用 | 较少使用 |
| Omega | $\Omega$ | $\omega$ | omega | 欧米伽 | 样本空间; 最后 | $\Omega$ 样本空间(第7章); GARCH 常数项 |

---

## B.2 量化中最常用的 10 个希腊字母

按使用频率排序:

| 排名 | 符号 | 每次出现大概率意味着 |
|------|------|-------------------|
| 1 | $\sigma$ | 波动率, 风险 |
| 2 | $\mu$ | 期望收益, 漂移 |
| 3 | $\beta$ | 市场敏感度, 因子暴露 |
| 4 | $\rho$ | 相关系数 |
| 5 | $\alpha$ | 超额收益, 截距 |
| 6 | $\Sigma$ | 协方差矩阵 |
| 7 | $\Delta$ | 期权 Delta(敏感度) |
| 8 | $\Gamma$ | 期权 Gamma(二阶敏感度) |
| 9 | $\lambda$ | 泊松强度, 风险价格 |
| 10 | $\nu$ | t 分布自由度 |

---

## B.3 常用数学符号

### 算子与函数

| 符号 | 含义 | 示例 |
|------|------|------|
| $\sum$ | 求和 | $\sum_{i=1}^{n} x_i$ |
| $\prod$ | 求积 | $\prod_{i=1}^{n} (1+r_i)$ |
| $\int$ | 积分 | $\int_a^b f(x)dx$ |
| $\lim$ | 极限 | $\lim_{n\to\infty}$ |
| $\frac{d}{dx}$ | 导数 | $\frac{d}{dx}x^2 = 2x$ |
| $\frac{\partial}{\partial x}$ | 偏导数 | $\frac{\partial f}{\partial S}$ = Delta |
| $\exp(x)$ 或 $e^x$ | 指数函数 | 连续复利 $e^{rt}$ |
| $\ln(x)$ | 自然对数 | 对数收益率 $\ln(P_t/P_{t-1})$ |
| $|\cdot|$ | 绝对值 | 绝对偏差 |
| $\|\cdot\|$ | 范数 | 向量/矩阵的"大小" |
| $\sqrt{\cdot}$ | 平方根 | 标准差 = $\sqrt{\text{Var}}$ |

### 概率与统计

| 符号 | 含义 | 示例 |
|------|------|------|
| $P(A)$ | 事件 $A$ 的概率 | $P(\text{涨}) = 0.55$ |
| $P(A \mid B)$ | 条件概率 | $P(\text{涨} \mid \text{加息})$ |
| $E[X]$ | 期望 | $E[R] = 8\%$/年 |
| $\text{Var}(X)$ | 方差 | $\text{Var}(R) = 0.04$ |
| $\text{Cov}(X,Y)$ | 协方差 | $\text{Cov}(R_i, R_j)$ |
| $\sigma_X$ | 标准差 | $\sigma = 20\%$/年 |
| $\rho_{XY}$ | 相关系数 | $\rho = 0.6$ |
| $X \sim \mathcal{N}(\mu,\sigma^2)$ | 服从正态分布 | $R \sim \mathcal{N}(0.001, 0.02^2)$ |
| $f(x)$ | 概率密度函数(PDF) | $f(r) = \frac{1}{\sigma\sqrt{2\pi}}e^{-(r-\mu)^2/2\sigma^2}$ |
| $F(x)$ | 累积分布函数(CDF) | $F(0) = P(R \leq 0)$ |
| $\Phi(z)$ | 标准正态 CDF | Black-Scholes 中的 $N(d_1)$ |
| $\hat{\theta}$ | 参数的估计量 | $\hat{\mu} = \bar{X}$ |

### 矩阵与向量

| 符号 | 含义 | 示例 |
|------|------|------|
| $w$ (粗体小写) | 向量 | 权重向量 $w = (w_1,\ldots,w_n)^T$ |
| $\Sigma$ (粗体大写) | 矩阵 | 协方差矩阵 |
| $A^T$ | 矩阵转置 | $w^T \Sigma w$ |
| $A^{-1}$ | 矩阵逆 | 最小方差组合权重 |
| $\mathbf{1}$ | 全 1 向量 | $\mathbf{1}^T w = 1$(权重和为 1) |

---

## B.4 快速查阅: "这个符号在量化中什么意思?"

当你看到一个不认识的符号, 按以下顺序排查:

1. **它是希腊字母吗?** → 查 B.1 表
2. **它有上下标吗?** → 下标通常是索引($i$ = 第 $i$ 个资产), 上标通常是幂或时间($t$ = 第 $t$ 期)
3. **它在公式的什么位置?** → 在分子通常是"收益端"的量, 在分母或根号内通常是"风险端"的量
4. **它出现在哪一章?** → 第 2-6 章(微积分)的符号以 $x, f, d/dx$ 为主; 第 7-12 章(概率统计)以 $P, E, \mu, \sigma, \beta$ 为主; 第 13-15 章(线性代数)以粗体向量/矩阵为主
