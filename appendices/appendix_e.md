# 附录E: 金融统计术语表

> 量化金融中的许多核心概念以"术语"而非"公式名"流传——IC、夏普比率、最大回撤这些词在研报和交易台上频繁出现。本附录将这些散落在各章节中的统计指标集中收录, 提供统一定义、计算公式和量化语境下的解读。每个术语标注了首次系统讲解的章节, 方便回溯学习。

---

## E.1 收益类指标

| 术语 | 英文 | 公式 | 说明 | 章节 |
|------|------|------|------|------|
| 单期收益率 | Simple Return | $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | 价格变动的百分比, 不可加 | Ch5 |
| 对数收益率 | Log Return | $r_t = \ln\frac{P_t}{P_{t-1}}$ | 时间可加: $r_{t_1 \to t_2} = \sum r_t$; 近似 $R_t$ 当 $R_t$ 很小时 | Ch5 |
| 累计收益 | Cumulative Return | $R_{\text{cum}} = \prod_{t=1}^{T}(1+R_t) - 1$ | 买入持有 $T$ 期的总收益 | Ch5 |
| 年化收益率 | Annualized Return | $\mu_{\text{ann}} = \bar{r} \times 252$ | 日收益均值×年交易日数; 用于跨频率比较 | Ch9 |
| 超额收益 | Excess Return | $r_t^{\text{ex}} = r_t - r_f$ | 超过无风险利率的部分——承担风险获得的补偿 | Ch9 |
| Alpha | Alpha ($\alpha$) | $\alpha = \bar{r}_p - [r_f + \beta(\bar{r}_m - r_f)]$ | Jensen's Alpha: 实际收益超过CAPM预测的部分——策略的"真本事" | Ch12 |

---

## E.2 风险类指标

| 术语 | 英文 | 公式 | 说明 | 章节 |
|------|------|------|------|------|
| 波动率 | Volatility ($\sigma$) | $\sigma = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(r_t - \bar{r})^2}$ | 收益率的标准差; 年化需 $\times\sqrt{252}$。对称度量——涨和跌的波动同等对待 | Ch9 |
| 下行波动率 | Downside Deviation | $\sigma_{\text{down}} = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}\min(r_t - \tau, 0)^2}$ | 只计低于目标收益 $\tau$(通常=0或无风险利率)的波动——惩罚下跌不惩罚上涨 | Ch9 |
| 最大回撤 | Maximum Drawdown (MDD) | $MDD = \min_t \left(\frac{P_t}{\max_{s \leq t} P_s} - 1\right)$ | 从历史最高点到后续最低点的最大跌幅; 最直观的风险度量——"最惨的时候亏了多少" | Ch14 |
| 回撤持续时间 | Drawdown Duration | $DD_{\text{dur}} = \text{从峰顶恢复到前高所需的交易日数}$ | MDD 不告诉你"在水下待了多久"——持续时间补充了这一信息 | — |
| 在险价值 | Value at Risk (VaR) | $P(r \leq -\text{VaR}_\alpha) = \alpha$ | 在置信水平 $1-\alpha$ 下, 单日最大可能亏损; 忽略了尾部极端损失 | Ch21 |
| 条件在险价值 | CVaR / Expected Shortfall | $\text{CVaR}_\alpha = E[-r \mid -r > \text{VaR}_\alpha]$ | 比 VaR 更保守——度量的是"超过 VaR 的那些极端损失的平均值" | — |

---

## E.3 风险调整收益类指标

| 术语 | 英文 | 公式 | 说明 | 章节 |
|------|------|------|------|------|
| 夏普比率 | Sharpe Ratio | $SR = \frac{\bar{r}_p - r_f}{\sigma_p}$ | 每单位总风险换取的超额收益; 年化: $SR_{\text{ann}} = SR_{\text{daily}} \times \sqrt{252}$; >1 为优秀, >2 为顶级; 不区分上行/下行波动 | Ch9 |
| 信息比率 | Information Ratio (IR) | $IR = \frac{\bar{r}_p - \bar{r}_b}{\sigma(r_p - r_b)}$ | 每单位主动风险(跟踪误差)换取的超额收益; 衡量基金经理的"选股能力"; >0.5 为良好 | Ch9 |
| Sortino 比率 | Sortino Ratio | $\text{Sortino} = \frac{\bar{r}_p - \tau}{\sigma_{\text{down}}}$ | 夏普比率的改进版——分母只用下行波动率。上涨的波动不被"惩罚" | — |
| Calmar 比率 | Calmar Ratio | $\text{Calmar} = \frac{\mu_{\text{ann}}}{|MDD|}$ | 年化收益÷最大回撤绝对值; 简单粗暴——"赚的够不够弥补最惨的时候"; >1 为良好 | — |
| 收益-回撤比 | Return/MDD | $\frac{\mu_{\text{ann}}}{\|MDD\|}$ | 同 Calmar, 但不强制用三年窗口 | Ch14 |
| Omega 比率 | Omega Ratio | $\Omega(\tau) = \frac{E[\max(r-\tau,0)]}{E[\max(\tau-r,0)]}$ | 目标收益 $\tau$ 以上的期望收益÷以下的期望损失; 考虑整个分布而非仅前两阶矩 | — |

---

## E.4 因子研究类指标

> 这些指标是量化股票多因子策略的核心语言, 将在第27章系统讲解。

| 术语 | 英文 | 公式 | 说明 | 章节 |
|------|------|------|------|------|
| 信息系数 | Information Coefficient (IC) | $\text{IC} = \text{Corr}(\text{factor}_t, r_{t+1})$ | 因子值 $t$ 期与下期收益的 Pearson 相关系数——因子预测能力的核心度量。\|IC\|>0.03 为有效, >0.05 为强因子 | Ch27 |
| Rank IC | Rank IC | 同上, 但用 Spearman 秩相关 | 比 Pearson IC 更稳健——受离群值和单调变换影响更小。量化行业的首选 | Ch27 |
| 信息比率(因子) | Information Ratio (IR) | $\text{IR} = \frac{\overline{\text{IC}}}{\sigma(\text{IC})}$ | IC的均值÷IC的标准差——因子预测能力的一致性和稳定性。IR>0.5 为良好 | Ch27 |
| IC 衰减 / 半衰期 | IC Decay / Half-Life | 计算IC在滞后 $k$ 期的衰减曲线, 找到IC降至峰值一半的滞后期 | 度量因子预测能力的"保质期"——半衰期越长, 调仓频率可以越低 | Ch27 |
| 胜率 | Hit Ratio / Win Rate | $\text{Win\%} = \frac{\text{预测方向正确的期数}}{\text{总期数}}$ | 简单直观——"猜对涨跌的比例"。二元指标, 不反映预测幅度 | Ch27 |
| 换手率 | Turnover | $\text{TO} = \frac{1}{2}\sum_i \|w_{i,t} - w_{i,t-1}\|$ | 单边换手——每期买入(或卖出)的权重比例。年化单边换手 < 200% 为低换手策略 | Ch27 |
| 分层回测 | Stratified Backtest | 按因子值将股票分5~10组, 计算每组等权收益; 多头组-空头组=因子溢价 | 不依赖具体权重方案——只看因子的"纯"区分能力; 比 IC 更贴近真实交易 | Ch27 |
| Fama-MacBeth | Fama-MacBeth Regression | 两步法: (1)逐期横截面回归得 $\hat{\beta}_{t}$; (2)时间序列上 $t$-检验 $\bar{\hat{\beta}}$ | 因子溢价显著性检验的标准方法——解决了残差截面相关导致的 t 统计量膨胀 | Ch27 |

---

## E.5 组合特征类指标

| 术语 | 英文 | 公式 | 说明 | 章节 |
|------|------|------|------|------|
| 跟踪误差 | Tracking Error (TE) | $\text{TE} = \sigma(r_p - r_b)$ | 组合收益与基准收益之差的标准差——"偏离基准的程度"。被动基金 TE<0.5%, 主动基金 3-8% | Ch9, Ch13 |
| 集中度 | Concentration (HHI) | $\text{HHI} = \sum_{i=1}^{N} w_i^2$ | Herfindahl-Hirschman Index——等权时最小(=1/N), 满仓一只时=1。$\sqrt{\text{HHI}}$ 的倒数 ≈"有效持仓数" | Ch13 |
| 有效持仓数 | Effective N | $N_{\text{eff}} = 1 / \sum w_i^2$ | 权重越不均匀, 有效持仓越少。$N=50$ 等权→ $N_{\text{eff}}=50$, HHI=0.02 | Ch13 |
| Beta | Beta ($\beta$) | $\beta_i = \frac{Cov(r_i, r_m)}{Var(r_m)}$ | 个股对市场变动的敏感度; $\beta>1$ 放大市场波动(进攻型), $\beta<1$ 缩小(防御型) | Ch9, Ch12 |
| 分散化比率 | Diversification Ratio | $\text{DR} = \frac{\sum w_i \sigma_i}{\sigma_p}$ | 加权平均波动率÷组合波动率; 越>1 说明分散化效果越好; =1 表示完全相关(无分散化收益) | Ch9 |
| 对冲比率 | Hedge Ratio | $\hat{\beta} = (\mathbf{x}^T\mathbf{x})^{-1}\mathbf{x}^T\mathbf{y}$ | OLS 斜率——需要多少单位的对冲工具来消除标的风险 | Ch12, Ch15 |

---

## E.6 统计推断类指标

| 术语 | 英文 | 公式 | 说明 | 章节 |
|------|------|------|------|------|
| t 统计量 | t-statistic | $t = \frac{\hat{\beta}}{\text{SE}(\hat{\beta})}$ | 估计值÷标准误——"信号/噪声比"。\|t\|>2  ≈ p<0.05 显著 | Ch11, Ch12 |
| p 值 | p-value | $P(\|T\| \geq \|t\| \mid H_0)$ | 在原假设下观察到如此极端统计量的概率; p<0.05 通常视为"显著"。不度量效应大小, 不度量 $H_0$ 为真的概率 | Ch11 |
| 置信区间 | Confidence Interval | $\hat{\theta} \pm z_{\alpha/2} \times \text{SE}(\hat{\theta})$ | 在重复抽样中, 有 $(1-\alpha)$ 的概率包含真实参数。95% CI 约 ±1.96×SE | Ch10 |
| $R^2$ | R-squared | $R^2 = 1 - \frac{RSS}{TSS} = \frac{ESS}{TSS}$ | 因变量总变异中被模型解释的比例; 在 CAPM 中 $R^2$ 约 0.2-0.5, 在 FF3 中约 0.3-0.6 | Ch12 |
| 调整 $R^2$ | Adjusted $R^2$ | $R^2_{\text{adj}} = 1 - (1-R^2)\frac{n-1}{n-k-1}$ | 惩罚多余变量——模型选择的信息准则 | Ch12 |
| 条件数 | Condition Number ($\kappa$) | $\kappa = \lambda_{\max} / \lambda_{\min}$ | 协方差矩阵的"病态程度"; $\kappa>30$ 需要收缩或降维, 否则组合优化不稳定 | Ch14 |
| 收缩强度 | Shrinkage Intensity ($\delta$) | $\mathbf{\Sigma}_{\text{LW}} = (1-\delta)\mathbf{S} + \delta\mathbf{F}$ | Ledoit-Wolf 自动选择; $\delta \approx 0.05\text{-}0.3$ 为典型范围 | Ch14 |

---

## E.7 术语索引（按中文拼音）

| 术语 | 英文 | 类别 |
|------|------|------|
| Alpha | Jensen's Alpha | 收益 |
| Beta | Market Beta | 组合特征 |
| Calmar 比率 | Calmar Ratio | 风险调整 |
| Fama-MacBeth 回归 | Fama-MacBeth Regression | 因子研究 |
| IC (信息系数) | Information Coefficient | 因子研究 |
| IC 半衰期 | IC Half-Life | 因子研究 |
| IR (信息比率) | Information Ratio | 风险调整/因子研究 |
| Omega 比率 | Omega Ratio | 风险调整 |
| p 值 | p-value | 统计推断 |
| Rank IC | Rank Information Coefficient | 因子研究 |
| Sortino 比率 | Sortino Ratio | 风险调整 |
| t 统计量 | t-statistic | 统计推断 |
| 波动率 | Volatility | 风险 |
| 超额收益 | Excess Return | 收益 |
| 对冲比率 | Hedge Ratio | 组合特征 |
| 对数收益率 | Log Return | 收益 |
| 分层回测 | Stratified Backtest | 因子研究 |
| 分散化比率 | Diversification Ratio | 组合特征 |
| 换手率 | Turnover | 因子研究 |
| 回撤持续时间 | Drawdown Duration | 风险 |
| 集中度 (HHI) | Concentration | 组合特征 |
| 年化收益率 | Annualized Return | 收益 |
| 胜率 | Hit Ratio / Win Rate | 因子研究 |
| 收益-回撤比 | Return/MDD | 风险调整 |
| 条件数 | Condition Number | 统计推断 |
| 调整 $R^2$ | Adjusted R-squared | 统计推断 |
| 跟踪误差 | Tracking Error | 组合特征 |
| 夏普比率 | Sharpe Ratio | 风险调整 |
| 下行波动率 | Downside Deviation | 风险 |
| 信息比率 | Information Ratio | 风险调整 |
| 有效持仓数 | Effective N | 组合特征 |
| 在险价值 | VaR | 风险 |
| 置信区间 | Confidence Interval | 统计推断 |
| 最大回撤 | Maximum Drawdown | 风险 |

---

> 本附录在不引入新章节的前提下, 将量化金融中最常用的统计术语集中整理。已系统讲解的指标标注了章节编号; 仅在此定义但未在正文展开的指标(如 Sortino、Calmar、Omega)可作为读者实践的参考起点。
