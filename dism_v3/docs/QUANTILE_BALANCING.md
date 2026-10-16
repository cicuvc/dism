# Quantile balancing：从全局分配到 DISM 的价格校正

本文推导当前固定 `direction=true` 的 QB 训练机制。数学推导按精确实数算术书写，
实现使用 BF16/FP32，数值与验证边界见
[SOFT_HARD_RELAXATION.md §53](SOFT_HARD_RELAXATION.md#53-fixed-true-qb-训练实现2026-10-16)。
配置方法见 [MODULE.md](MODULE.md#opt-in-quantile-balancing-fixed-true-direction)，
实现为 [`quantile_balance.py`](../python/flash_dism/quantile_balance.py)。

本文给出本实现所用优化问题的自包含推导，不将有限次坐标更新描述为精确最优分配。
Q/K 两侧各自求解同一种问题；以下先省略 layer、head、Q/K side 下标。

## 1. 把词频均衡写成全局分配问题

设有 N 个校准 token、V 个符号，原始选择分数为

\[
S_{ic}=\beta\,x_i^\top e_c,
\qquad i=1,\ldots,N,\quad c=1,\ldots,V.
\]

没有均衡时，每个 token 独立选择最高分符号。加入目标频率
\(\rho_c>0\)、\(\sum_c\rho_c=1\)，令容量 \(C_c=N\rho_c\)，得到

\[
\begin{aligned}
\max_X\quad &\sum_{i,c}S_{ic}X_{ic}\\
\text{s.t.}\quad
&\sum_cX_{ic}=1 &&\forall i,\\
&\sum_iX_{ic}=C_c &&\forall c,\\
&X_{ic}\in\{0,1\}.
\end{aligned}
\tag{1}
\]

它在满足全局使用量的分配中最大化保留的选择分数。均匀目标取
\(\rho_c=1/V\)。这不是“最大化语言模型质量”：S 是局部选择分数，容量是额外约束。

把最后一个条件放松为 \(X_{ic}\ge0\) 得到运输线性规划；行和已隐含
\(X_{ic}\le1\)。若各 C 为整数，该运输约束的整数性保证存在整数最优解，
LP 最优值与整数分配问题相同。若 C 非整数，精确整数分配不可行，但分数 LP
仍有意义。当前求解器允许 N/V 非整数；这不意味着 hard labels 能实现分数配额。

“全局”指同一层、同一 head、同一 side 的校准样本总体，不是把不同 head 的
code IDs 混在一起计数。Q 与 K 分别均衡，也不等价于联合优化它们的匹配结构。

## 2. 容量约束的对偶变量就是 code price

对列容量引入自由对偶变量 \(b_c\)，保留逐行 simplex 约束。拉格朗日函数为

\[
\begin{aligned}
\mathcal L(X,b)
&=\sum_{i,c}S_{ic}X_{ic}
 +\sum_cb_c\left(C_c-\sum_iX_{ic}\right)\\
&=\sum_{i,c}(S_{ic}-b_c)X_{ic}+\sum_cC_cb_c.
\end{aligned}
\]

固定 b 后，每行的最大值就是最高调整分数。因此 LP 的对偶为

\[
\boxed{
\min_b D(b),\qquad
D(b)=\sum_i\max_c(S_{ic}-b_c)+\sum_cC_cb_c.
}
\tag{2}
\]

归一化到每个 token：

\[
\bar D(b)=\frac1N\sum_i\max_c(S_{ic}-b_c)+\sum_c\rho_cb_c.
\tag{3}
\]

均匀目标的最后一项是 `mean(b)`，对应代码的 `best_dual`。价格不要求非负：
它是相对偏置。高价格压低某个 code 的竞争力，负价格提高它的竞争力。

若各行最大值唯一，记调整后 code 的负载为 \(n_c(b)\)，则

\[
\frac{\partial D}{\partial b_c}=C_c-n_c(b).
\tag{4}
\]

梯度下降会对过载 code 提价，对欠载 code 降价。有 ties 时应使用次梯度，
把每行质量在并列最大值之间分配；不能把任意 tie-breaking 后的整数计数
当作唯一梯度。

价格存在共同平移的自由度。因为 \(\sum_cC_c=N\)，

\[
D(b+t\mathbf1)=D(b).
\tag{5}
\]

实现每轮减去价格均值以固定这个自由度；这既不改变 argmax，也不改变 softmax。

## 3. 从 max 对偶转成可做 quantile 更新的 hinge 对偶

为获得简单的块最小化公式，在 primal 中显式保留冗余上界 \(0\le X_{ic}\le1\)，
然后同时对行和、列和引入自由乘子 \(\alpha_i,b_c\)。对每个 X 独立最大化：

\[
\max_{0\le x\le1}(S_{ic}-\alpha_i-b_c)x
=[S_{ic}-\alpha_i-b_c]_+.
\]

由此得到等价的扩展对偶

\[
\boxed{
F(\alpha,b)=\sum_i\alpha_i+\sum_cC_cb_c
 +\sum_{i,c}[S_{ic}-\alpha_i-b_c]_+,
\qquad \min_{\alpha,b}F(\alpha,b).
}
\tag{6}
\]

这个目标是凸的，但不光滑。它的每个行阈值、每个列价格都能通过排序位置求出
精确的一维最小值；“近乎闭式”指这些**块更新**，不是整个多 code 分配一次闭式求解。

### 3.1 固定价格，行阈值位于 top-1 与 top-2 之间

固定 b，令 \(r_{ic}=S_{ic}-b_c\)。第 i 行的子问题为

\[
\min_\alpha f_i(\alpha),\qquad
f_i(\alpha)=\alpha+\sum_c[r_{ic}-\alpha]_+.
\]

非折点处的导数为

\[
f_i'(\alpha)=1-\#\{c:r_{ic}>\alpha\}.
\tag{7}
\]

将 r 降序排列成 \(r_{i,(1)}\ge r_{i,(2)}\ge\cdots\)。恰好一个项高于阈值时
导数为零，因此任何

\[
\alpha_i\in[r_{i,(2)},r_{i,(1)}]
\tag{8}
\]

都是最小点，包括有 ties 的退化区间。实现取中点：

\[
\alpha_i=\frac{r_{i,(1)}+r_{i,(2)}}2.
\tag{9}
\]

在这个区间内 \(f_i(\alpha_i)=r_{i,(1)}\)，从而

\[
\min_\alpha F(\alpha,b)=D(b).
\tag{10}
\]

这也证明了 hinge 形式与前面的 max 对偶一致。

### 3.2 固定行阈值，列价格是容量对应的 quantile

固定 α，令 \(u_{ic}=S_{ic}-\alpha_i\)。第 c 列的子问题为

\[
\min_b g_c(b),\qquad g_c(b)=C_cb+\sum_i[u_{ic}-b]_+.
\]

非折点处

\[
g_c'(b)=C_c-\#\{i:u_{ic}>b\}.
\tag{11}
\]

在折点处，精确最优条件为

\[
\boxed{
\#\{i:u_{ic}>b_c\}\le C_c\le\#\{i:u_{ic}\ge b_c\}.
}
\tag{12}
\]

设 \(u_{(1)c}\ge\cdots\ge u_{(N)c}\)，并假设 \(0<C_c<N\)：

- 若 \(C_c=k\) 是整数，任何
  \(b_c\in[u_{(k+1)c},u_{(k)c}]\) 都最优。实现取两个端点的中点。
- 若 \(C_c\notin\mathbb Z\)，取
  \(b_c=u_{(\lceil C_c\rceil)c}\)。在此折点，次梯度区间包含零。

因此，对应的是上尾容量 \(C_c/N=\rho_c\)，也就是约 \(1-\rho_c\) 的分位位置。
用降序的第 k 大定义可以避免不同 quantile API 插值约定带来的歧义。
当前实现仅使用均匀 \(C=N/V\)，通过 `topk(ceil(C)+1, dim=token)` 取所需位置，
并在 N/V 非整数时使用单个 order statistic。

### 3.3 一轮更新和 reduced dual 的关系

一轮执行：

```text
r = S - b
alpha = midpoint(row_top1(r), row_top2(r))
u = S - alpha
b_new = column_capacity_quantile(u)
b_new -= mean(b_new)
```

忽略最后的平移，精确算术下有

\[
D(b^{t+1})
=\min_\alpha F(\alpha,b^{t+1})
\le F(\alpha^t,b^{t+1})
\le F(\alpha^t,b^t)
=D(b^t).
\tag{13}
\]

中心化不改变 D，所以也不影响这个结论。实现仍显式跟踪每个 head 的最小 D，
返回最优迭代，以应对有限精度并避免依赖“最后一轮必定最好”的假设。
这不提供有限轮数内到达全局最优点的证明；一般的非光滑坐标最小化也不能仅凭
凸性就声称一定收敛到全局最优。

## 4. 最优价格不等于任意 argmax 都满足容量

设 b 已达到对偶最优。互补条件要求 primal 的正质量只落在每行的
`argmax(S-b)` 支持集上，同时仍需满足列容量。

若每行都有唯一最大值，D 在该点可微，最优条件 (4) 给出负载恰好等于容量。
如果有 ties，可能需要在并列候选之间做额外的容量匹配，才能恢复整数 primal 解。
运输 LP 的整数性保证整数容量下存在整数最优分配，但不保证固定的
“取最小 code ID”规则会找出它。

例如所有 \(S_{ic}=0\)，取 b=0 已是对偶最优；然而普通 `argmax` 可以把所有 token
都分给第一个 code。中点阈值能减少某些边界 ties，无法消除这种结构性退化。
非整数容量则连精确满足容量的整数 assignment 都不存在。

当前实现直接使用调整后 logits 的 argmax，不执行额外的 flow/matching 修复。
应同时监测 active codes、全局频率熵/有效符号数、最大负载和容量残差，不能
仅凭 dual 下降或 active codes 增多宣称求得了精确最优配额。

二类问题有更直接的闭式说明。若 V=2、容量 \(C_1=k\) 为整数，定义 margin
\(d_i=S_{i1}-S_{i2}\)。把最大的 k 个 margin 分给 code 1，其余给 code 2 即最优；
价格差 \(b_1-b_2\) 位于第 k 与 k+1 大 margin 之间即可实现该划分（ties 仍需处理）。
V>2 时，各列的竞争通过行阈值耦合，不能简单对每列独立排序后就结束。

## 5. 强度、在线价格与 soft mass 是不同的选择

### 5.1 强度缩放不是新的精确配额解

实际使用

\[
a_c=\lambda b_c,\qquad z_{ic}=S_{ic}-a_c,
\quad 0\le\lambda\le1.
\tag{14}
\]

λ=0 不施加价格；λ=1 使用完整拟合价格。中间值是对分数偏置的插值，
**不是**把原始词频与均匀词频作线性插值，也没有每个 code 的使用量随 λ 单调变化
的保证。当前求解器拟合完整容量问题，再在应用时缩放价格。

### 5.2 当前训练是延迟价格控制，不是当前 batch 的强制分配

在线模型的分数会随参数、输入和 β 改变。实际采用：

\[
\widehat b_t=\operatorname{QB}(S_t,b_t),\qquad
b_{t+1}=\operatorname{center}\big((1-\eta)b_t+\eta\widehat b_t\big).
\tag{15}
\]

当前 forward 使用旧 b 的独立快照，新价格供后续 forward 使用；求解和更新都
stop-gradient。这样当前序列的 token 分配不会因为用它的未来 token 重新拟合
当前价格而产生额外的因果泄漏。eval 冻结价格，checkpoint 保存价格和更新计数。

分布式实现先汇总各 rank 的 sampled scores 再求解。平均各 rank 分别求得的
价格一般不等于全局问题的解，因为 max、quantile 都不是线性运算。
目前是每个非空 rank 等量采样；这对应各 rank 等权的采样分布，而不是在不同
有效 token 数之间自动实现精确的全 token 加权。

有限校准样本、有限迭代、EMA、延迟、λ<1，以及训练/验证分布变化，都会使
实际使用量偏离校准容量。它们是工程上的在线近似，不是精确配额保证。
价格按 S 的 logit 单位存储；改变 β 后旧价格不再自动对应新 S 的最优价格。

### 5.3 hard 容量均衡不保证 soft 概率质量均衡

当前目标 (2) 针对 top-1 分配。若希望直接均衡 soft mass，可以另定义熵正则 primal：

\[
\max_X\ \langle S,X\rangle
+\varepsilon\sum_i H(X_i),
\quad X_i\in\Delta_V,\quad\sum_iX_{ic}=C_c.
\]

它的价格对偶为

\[
D_\varepsilon(b)=
\varepsilon\sum_i\log\sum_c\exp((S_{ic}-b_c)/\varepsilon)
+\sum_cC_cb_c,
\tag{16}
\]

对应

\[
X_{ic}(b)=\operatorname{softmax}_c((S_{ic}-b_c)/\varepsilon),
\qquad
\partial_{b_c}D_\varepsilon=C_c-\sum_iX_{ic}(b).
\tag{17}
\]

其最优条件均衡的是概率质量。ε→0 时目标趋向 max 对偶，但有限 ε 下通常
不再有上面的硬 quantile 块更新公式。**这不是当前实现的求解目标**。
当前 soft path 使用 hard-capacity 拟合得到的同一价格，是为了与 hard 标签定义
一致；不能据此声称 soft mass 也均匀。

## 6. 接入 fixed-true DISM 的 Jensen 下界

分别对 Q/K 得到价格，并定义

\[
a_c^q=\lambda_qb_c^q,\qquad a_c^k=\lambda_kb_c^k,
\]
\[
z_{ic}^q=\beta q_i^\top e_c^q-a_c^q,\quad
z_{jc}^k=\beta k_j^\top e_c^k-a_c^k,
\quad p_i^q=\operatorname{softmax}(z_i^q),\quad
p_j^k=\operatorname{softmax}(z_j^k).
\tag{18}
\]

两个符号独立采样时的相等概率为
\(M_{ij}=\sum_cp_{ic}^qp_{jc}^k\)。对 log 使用 Jensen：

\[
\log M_{ij}
=\log\mathbb E_{c\sim p_j^k}[p_{ic}^q]
\ge\mathbb E_{c\sim p_j^k}[\log p_{ic}^q].
\]

展开价格后的 log probability：

\[
\begin{aligned}
L_i^q&=\log\sum_c\exp(z_{ic}^q),\\
u_j&=\sum_cp_{jc}^ke_c^q,\\
m_j&=\sum_cp_{jc}^ka_c^q,\\
\boxed{\ell_{ij}
&=\beta q_i^\top u_j-L_i^q-m_j
\le\log M_{ij}.}
\end{aligned}
\tag{19}
\]

因此原有“词表插值 + dot − LSE”仍可使用，但启用 Q 价格时必须额外减去 m。
K 价格通过 pk 改变插值和 m；展开 `E_K[log p_Q]` 时没有再显式减一项 K 价格，
也不应把 K 的 log-normalizer 加到这个 true-direction 下界中。

- K-only：\(a^q=0\)，所以 m=0；改变 K 插值权重、hard K labels 即可。
- Q-only：Q LSE、hard Q labels 发生变化，且 m 通常非零，即便 K 分布没有变。
- Q+K：两侧 logits 都校正，并保留相同的 m 修正。

soft core 的局部门使用 `logM_soft=tau+ell`；hard core 使用同一组 z 的 argmax
得到 labels，相等时门为 exp(tau)，不等时为零。Jensen 不等式约束的是 soft
分布相等概率，不是说 soft 门对每个确定性的 argmax 指示函数都逐点构成下界。

## 7. stop-gradient 价格不等于 detach 价格期望

一次 forward 内价格视为常数；但 \(m_j=\sum_cp_{jc}^ka_c^q\) 仍通过 pk 依赖 K。
设插值模块三个可微输出为 u、Lq、m，其上游梯度分别为向量 \(g_j\)、标量
\(h_i\)、标量 \(r_j\)。则

\[
v_{jc}=g_j^\top e_c^q+r_j a_c^q,
\]
\[
\boxed{
\delta^k_{jc}=\frac{\partial\mathcal L}{\partial z^k_{jc}}
=p^k_{jc}\left(v_{jc}-\sum_dp^k_{jd}v_{jd}\right),
\qquad
\delta^q_{ic}=h_i p^q_{ic}.
}
\tag{20}
\]

在插值模块内有

\[
\begin{aligned}
\nabla_{q_i}\mathcal L&=\beta\sum_c\delta^q_{ic}e_c^q,\\
\nabla_{k_j}\mathcal L&=\beta\sum_c\delta^k_{jc}e_c^k,\\
\nabla_{e_c^q}\mathcal L&=\sum_jp^k_{jc}g_j+
                         \beta\sum_i\delta^q_{ic}q_i,\\
\nabla_{e_c^k}\mathcal L&=\beta\sum_j\delta^k_{jc}k_j.
\end{aligned}
\tag{21}
\]

Q 作为 core dot 左操作数的直接梯度由外部计算图另外累加；(21) 只列插值模块
自身的贡献。这里没有对 b 的梯度，也没有隐式微分 quantile 解。
若把整个 m detach，就会遗漏 (20) 中 `r_j*a_qc` 导致的 K/codebook 梯度。

数值实现中 Q LSE 只预吸收一次 `-tau.detach()`，m 不吸收 tau；core 的显式 dτ
负责这一次 tau。价格快照需独立存储，不能在 backward 前原地改写被保存的价格。

## 8. 可以据此得出什么，仍需验证什么

该推导说明：容量问题可转化为 code price；hinge 对偶的两个块有 order-statistic
最优更新；价格可以同时进入 soft/hard 选择，而不破坏 true-direction 的 Jensen
下界形式。当前实现按这些公式计算，数值、状态恢复和配置限制另见 §53。

它不保证多 code 问题一次更新就最优，不保证有限迭代/ties 下的整数配额，不保证
独立均衡后的 Q/K 仍保留原有匹配关系，也不保证词频均匀提高最终语言模型质量。
与 MoE 的计算负载不同，符号的最优语义频率未必均匀；弱价格、非均匀目标或
仅防止死码是可单独研究的目标，当前代码只实现均匀容量价格及独立强度控制。
