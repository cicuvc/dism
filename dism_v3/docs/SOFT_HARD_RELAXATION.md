# soft / hard 分支：以匹配概率的 Jensen 下界实现可微训练

这份笔记整理 DISM 中「soft 分支」为什么可以作为「hard 分支（符号匹配）」
的可微近似。结论先给出：

> 硬分支把 pair (i,j) 的匹配权重视为离散的 `1[a_q(i) == a_k(j)]`；
> `M_ij = P(a_q(i) == a_k(j)) = <p_q(i), p_k(j)>` 才是它本来的“概率”语义。
> soft 分支实际计算的是 `log M_ij` 的 Jensen/ELBO 下界
> `S_ij = E_{c~p_k(j)}[log p_q(i)(c)]`，它对 q/k 与词表全部可微；
> 于是 `exp(S_ij) <= M_ij` 是匹配概率的一个保守、光滑的替代。
> 该下界恰好等于「插值嵌入双线性分数 − LSE」，正是 kernel 里那一步。
>
> 第二部分（§10–§16）回答另一半：概率化 suffix matching 为什么能近似
> 「纯离散 suffix matching 输出的期望」。对匹配计数可以严格用线性期望
> （递推本身线性、各偏移的匹配指示独立）；但对最终输出不行，因为输出
> 带了归一化分母 `1 + Σ G`。soft 前向算的是 `E[N]/E[D]`（annealed /
> mean-field），而真正的期望是 `E[N/D]`（quenched）。两者之差由分母的
> 相对涨落控制，在匹配多且置信时长上下文里很小，在稀疏/不确定时不可忽略。

代码依据：

- 硬/软分值与递推 oracle：`python/flash_dism/reference/dism_v3_ref.py`
  （`dism_ref` / `dism_ref_backward` / `dism_wrapper`）。
- “Jensen score = expectation of log probabilities” 的独立 oracle：
  `tests/test_dism_v3_ref.py::oracle`。
- 插值嵌入与 LSE-tau：`python/flash_dism/emb_kernel.py`。
- 训练期 `hard_prob` 调度：`docs/MODULE.md`、`docs/MODEL.md`。
- 预吸收 LSE / 硬覆盖：`docs/SUMMARY.md`。

---

## 1. 记号

每个 head 有两张有效码字表（可学习参数，本身就是码字，无运行时激活）：

```
W_q, W_k in R^{H x V x D}        # query / key vocabulary
q_i, k_j in R^D                  # 第 i 个 query / 第 j 个 key 的向量
```

词表 logits 与分布：

```
s_q(i,c) = q_i . W_q(c)          # c = 1..V
p_q(i)   = softmax_c s_q(i,.)
L_q(i)   = logsumexp_c s_q(i,c)  # natural-log LSE
a_q(i)   = argmax_c s_q(i,c)     # 离散符号（codeword 下标）

s_k(j,c) = k_j . W_k(c)
p_k(j)   = softmax_c s_k(j,.)
L_k(j)   = logsumexp_c s_k(j,c)
a_k(j)   = argmax_c s_k(j,c)
```

注意 `L_q(i)` 就是常说的 LSE；生产路径把 `tau` 预吸收进 LSE
（`q_lse = raw_q_lse - rtau`），本笔记为清晰起见把 `tau` 单独写出。

---

## 2. hard 分支：离散符号匹配

`dism_ref` 里：

```python
hard_score = zeros.masked_fill(idx_q != idx_k, -inf)   # 0 if a_q == a_k else -inf
mix_score  = where(hard, hard_score, norm_score)
logM       = mix_score + rtau
```

所以 hard 行每对的 log 权重是

```
logM_hard(i,j) = rtau + 0                       if a_q(i) == a_k(j)
               = rtau + (-inf)                  otherwise
```

即乘性门 `exp(logM_hard) = exp(tau) * 1[a_q(i) == a_k(j)]`，取值
`{0, exp(tau)}`。

问题：`a = argmax` 是关于 q/k/W 的分段常数函数，几乎处处梯度为 0。
直接对 hard 分支反传，匹配本身学不到任何东西；q/k/词表唯一的梯度来源
是 readout (`sq/sk/v`) 与 `tau`。

---

## 3. 匹配概率 `M_ij`：hard 分支的“期望”语义

把符号看成随机变量：对 logits 加独立 Gumbel 噪声后取 argmax
（Gumbel-max 恒等式），则

```
P(a_q(i) = c) = p_q(i)(c),   P(a_k(j) = c) = p_k(j)(c).
```

两侧独立时，两个符号相同的概率为

```
M_ij = P(a_q(i) == a_k(j))
     = sum_c p_q(i)(c) * p_k(j)(c)
     = <p_q(i), p_k(j)>
     = E_{c ~ p_k(j)} [ p_q(i)(c) ].
```

`M_ij` 才是“第 i 个 q 符号与第 j 个 k 符号相同的概率”。
hard 分支用的 `1[a_q == a_k]` 是它的 0/1 采样实现。
`M_ij in [0,1]` 对 q/k/W 光滑可微（它是两个 softmax 的内积），但直接
把它当 log 域门使用需要 `log M_ij`，而 `log sum_c p_q p_k` 不是我们想要
的“插值双线性 − LSE”形式。于是取 Jensen 下界。

---

## 4. 关键推导：`log M_ij` 的 Jensen 下界

对 `M_ij = E_{c~p_k(j)}[p_q(i)(c)]` 取对数。`log` 是凹函数，由 Jensen：

```
log M_ij = log E_{c~p_k(j)}[ p_q(i)(c) ]
        >= E_{c~p_k(j)}[ log p_q(i)(c) ]
         = S_ij.
```

把 `log p_q(i)(c) = q_i . W_q(c) - L_q(i)` 代入：

```
S_ij = sum_c p_k(j)(c) * ( q_i . W_q(c) - L_q(i) )
     = q_i . ( sum_c p_k(j)(c) W_q(c) ) - L_q(i)
     = q_i . q_emb(j) - L_q(i),          (direction = True)
```

其中 `q_emb(j) = sum_c p_k(j)(c) W_q(c)`、`L_q(i) = logsumexp_c(q_i . W_q(c))`。

这正好等于 kernel 中

```python
raw_scores = einsum('bnhc,bmhc->bhnm', q_vec, k_vec)
norm_score = raw_scores - where(direction, q_lse, k_lse)
```

所算的 `norm_score`：`q_vec = q_i`、`k_vec = q_emb(j)`、减 query LSE。
换句话说，soft 分支实现的是**匹配概率对数的变分下界（ELBO）**，
它天然可微、且完全由已有的插值嵌入 MMA 与 LSE 减法得到。

对称地，`direction = False` 时交换两侧：

```
S_ij = E_{c~p_q(i)}[ log p_k(j)(c) ]
     = k_emb(i) . k_j - L_k(j),          (direction = False)
```

两个方向都是同一个 `log M_ij` 的下界，只是 KL 朝向不同：

```
S_true  = -H(p_k) - KL(p_k || p_q)
S_false = -H(p_q) - KL(p_q || p_k)
```

因此 soft 分支在匹配的同时也在对齐两个码字分布。

---

## 5. soft 门与 hard 门的关系

soft 行：

```
logM_soft(i,j) = rtau + S_ij
exp(logM_soft) = exp(rtau) * exp(S_ij)
               <= exp(rtau) * M_ij.        (Jensen)
```

hard 行：

```
exp(logM_hard) = exp(rtau) * 1[a_q(i) == a_k(j)].
```

极限行为（`scale -> 无穷`，softmax 变 one-hot）：

- `a_q(i) == a_k(j)`：`M_ij -> 1`, `exp(S_ij) -> 1`；
- `a_q(i) != a_k(j)`：`M_ij -> 0`, `exp(S_ij) -> 0`。

即 soft 门 `exp(S)` 是 hard 指示函数的一个光滑下界替代，
在锐化极限下两者一致。数值上（见 §8）随 logits 放大，
`exp(S)` 与 `M` 的差距单调缩小。

与 oracle 对照，`tests/test_dism_v3_ref.py::oracle` 把这一点写得很直白：

```python
score = qs.log_softmax(-1) @ pk.T        # = E_{p_k}[log p_q], 即 S_ij
match = qs.argmax(-1)[:,None] == ks.argmax(-1)[None,:]
m = torch.where(hard, match, score.exp()) * tau.exp()
```

soft 分支就是用 `score.exp()` 替换离散的 `match`。

---

## 6. 为什么 soft 分支可微（梯度形式）

`S_ij` 同时依赖 q 侧（`q_i`, `W_q`, 以及 `L_q(i)`）和 k 侧（`p_k(j)`，
进而 `k_j`, `W_k`）。以 `direction = True` 为例，记

```
qbar_i = sum_c p_q(i)(c) W_q(c)      # query 用自身分布的插值
qtil_j = sum_c p_k(j)(c) W_q(c)      # query 词表用 key 分布的插值
```

则（标准 softmax/交叉熵残差形式）：

```
dS_ij / dq_i     = qtil_j - qbar_i
dS_ij / dW_q(c)  = ( p_k(j)(c) - p_q(i)(c) ) * q_i
dS_ij / dk_j     = sum_c p_k(j)(c) ( log p_q(i)(c) - E_{p_k}[log p_q] )
                          * ( W_k(c) - sum_u p_k(j)(u) W_k(u) )
dS_ij / dW_k(c)  = p_k(j)(c) ( log p_q(i)(c) - E_{p_k}[log p_q] ) * k_j
```

全部光滑、有限。对比 hard：`d 1[a_q == a_k] / d(theta) = 0`（a.e.），
没有可用的学习信号。

梯度最终经 log 域链式递推回传。`dism_ref_backward` 里：

```python
total = p * (s * e - correction)     # 匹配权重处的局部梯度
for i in reversed(range(N)):
    propagated = total[:, :, i] * alpha[:, :, i]   # alpha = sigmoid(softplus')
    if i:
        total[:, :, i-1, :-1] += propagated[..., 1:]
dtau  = total.sum((0, 2, 3))
dsoft = total.masked_fill(hard.unsqueeze(-1), 0.)  # hard 行不训练 q/k/词表
```

即：**hard 行仍然参与递推并训练 `tau`；只有 soft 行把梯度送给
q/k/词表去学习“匹配”本身。**

---

## 7. 接入 suffix-matching 递推

DISM 不是普通 attention，而是沿反对角线的符号串匹配递推
（`dism_ref`）：

```
W(i,j)   = logM(i,j) + softplus( W(i-1, j-1) )
```

等价地，在概率（乘性）域：

```
exp(W(i,j)) = exp(logM(i,j)) * ( 1 + exp(W(i-1, j-1)) ).
```

`1 + ...` 表示“重新起一段匹配”或“延长已有匹配链”。因此 `logM`
里的匹配门必须处在 log 域、且可微：

- hard 行 `logM in {tau, -inf}` → 二值门；
- soft 行 `logM = tau + S_ij`，`exp` 后是 `M_ij` 的下界估计。

两者的门都是 `exp(tau)` 的倍数，故 `tau` 对两种模式语义一致。

---

## 8. 数值验证

用 FP64 直接核对（不含 CUDA 依赖）：

```
identity:   max | S_ij - E_{p_k}[log p_q] |            = 1.8e-15
Jensen:     max ( exp(S_ij) - M_ij )                   < 0
soft grad:  ||dS/dq||, ||dS/dk||, ||dS/dWq||, ||dS/dWk||  > 0
hard:       1[a_q == a_k] 无 grad_fn
锐化极限:
    scale= 5:  mean|exp(S) - M| = 1.1e-2
    scale=20:  mean|exp(S) - M| = 8.5e-8
```

复现脚本：在 `dism_v3/` 下用 `PYTHONPATH=python:..` 运行 §4/§6 的公式即可
（只需 torch，不需要构建 CUDA 扩展）。

---

## 9. 训练策略与边界

- `hard_prob` 按优化步外部调度：`p=0` 全 soft（完全可微），
  逐渐退火到 `p=1` 逼近推理时的 hard 语义；`MODULE.md` 建议外部传入、
  不要依赖 forward 内部计数器。省略时训练默认 soft、评估默认 hard。
- 每行独立采样 hard/soft（packed 训练按行而非文档）。期望梯度为
  `(1-p) * grad_soft + p * grad_hard`，其中 `grad_hard` 只含 `tau` 与
  readout 通路。
- 硬标志在 backward 复用、不重新采样，保证同一步前后向一致。

必须保留的数值/语义边界：

- `exp(S) <= M` 只是下界，`S` 并非 `log M` 的无偏估计；分布越平坦
  下界越松，锐化极限才取等。因此不要把 soft 当作精确匹配概率。
- 方向选择改变 KL 朝向；true 用 query LSE，false 用 key LSE，
  不能混用。
- hard 分支本身在匹配项上不可训，这是设计取舍而非缺陷：离散匹配用于
  推理与精确语义，可微训练由 soft 下界承担。
- 不要为“让它更像 hard”而改成 straight-through 或硬性阈值，那会改变
  `logM` 数学并破坏现有 oracle / 梯度测试。

---

# 第二部分：概率化 suffix matching 与离散输出的期望

## 10. 纯离散 suffix matching 的输出

把每对 (i,j) 的符号匹配写成离散指示 `m_ij = 1[a_q(i) == a_k(j)] in {0,1}`。
沿反对角线的匹配计数（`dism_ref` 的 `rows` 展开）：

```
Gdisc_ij = m_ij * ( 1 + Gdisc_{i-1,j-1} )
         = sum_{l>=1} prod_{t=0}^{l-1} m_{i-t, j-t}    # 公共后缀长度之和
```

`Gdisc_ij` 是一个非负整数：q 的前缀（截止 i）与 k 的前缀（截止 j）
**公共后缀的条数**（长度 l=1,2,...）。离散输出为

```
Odisc_i = ( sum_j Gdisc_ij * a_ij ) / ( 1 + sum_j Gdisc_ij ),
   a_ij = ( sq_i . sk_j ) * v_j.
```

分母里的 `1` 是“空后缀/无匹配”基项（reference 中的 `fallback_weight`）。
它也是一条归一化：输出是 `a_ij` 以 `Gdisc_ij` 为权重、再加一个空匹配项
的加权平均。

## 11. 匹配计数：线性期望是严格成立的

设符号随机（例如对 logits 加 Gumbel 噪声后取 argmax），使
`P(m_ij = 1) = M_ij = <p_q(i), p_k(j)>`。沿一条反对角线各偏移用到的
token 互不相同，故不同偏移的匹配指示独立。对递推取期望：

```
E[Gdisc_ij] = E[m_ij (1 + Gdisc_{i-1,j-1})]
            = M_ij (1 + E[Gdisc_{i-1,j-1}])       # 独立 => 期望可拆
```

这正是 soft 递推，只是把 `g_ij` 换成真匹配概率 `M_ij`。所以
**匹配计数层面的“概率化”能精确算出 `E[Gdisc]`**，这一步线性期望完全
合法；进而 `E[N_i] = sum_j E[Gdisc_ij] a_ij`、`E[D_i] = 1 + sum_j E[Gdisc_ij]`
都可以逐步线性地算出来。

两个偏差在此已经埋下：

1. 代码实际用的是 `g_ij = exp(S_ij) <= M_ij`（第一部分的 Jensen），
   所以算的是 `E[Gdisc]` 的**下界**，不是 `E[Gdisc]` 本身；
2. 输出是比值，不是计数本身（见下）。

## 12. 分母为什么破坏线性：annealed vs quenched

把权重写成 `w_ij = Gdisc_ij / D_i`，`D_i = 1 + sum_j Gdisc_ij`，则

```
Odisc_i = sum_j w_ij a_ij = E_{J ~ w_i}[ a_{iJ} ]      # 条件期望
```

`w_i` 本身是随机的（依赖匹配配置），所以真正的期望是**先归一化再平均**
（quenched）：

```
E[Odisc_i] = E[ N_i / D_i ]        # 正确，但不可线性拆分
```

而概率化前向是**先平均再归一化**（annealed / mean-field）：

```
Osoft_i = E[N_i] / E[D_i] = ( sum_j E[Gdisc_ij] a_ij ) / ( 1 + sum_j E[Gdisc_ij] ).
```

一般

```
E[ N / D ]  !=  E[N] / E[D].
```

这就是用户指出的“不能用线性期望算”的地方：递推、分子可以线性取期望，
但归一化（softmax / `1+Σ`）是非线性的，期望与比值不交换。等价地说，
soft 前向把「对 softmax 权重取平均」换成了「对 logits/计数取平均后的
softmax」，这正是统计物理里的 annealed 与 quenched 平均之别（也是
“hard attention 的期望”与“soft attention”之间的经典差距）。

## 13. 偏差的定量（delta / 累积量展开）

记 `D = Dbar (1+eps)`，`N = Nbar (1+eta)`，`E[eps]=E[eta]=0`。展开：

```
E[N/D] = Nbar/Dbar - Cov(N,D)/Dbar^2 + E[N (D-Dbar)^2]/Dbar^3 - ...
```

相对偏差的首阶是

```
( E[N/D] - Nbar/Dbar ) / ( Nbar/Dbar )
   = - Cov(N,D)/(Nbar Dbar) + O( (D-Dbar)^2 / Dbar^2 ).
```

因为 `N = sum_j G_ij a_ij` 与 `D = 1 + sum_j G_ij` 共用同一批权重，二者
强正相关，协方差项一般不为零；其量级由 `D` 的相对涨落
`std(D)/Dbar ~ 1/sqrt(D_eff)` 控制，`D_eff` 是有效公共后缀计数。
特例（readout 近似常数 `a_ij ≈ ā`）下 `N ≈ ā(D-1)`，协方差项与二阶项
部分抵消，剩下 `E[O] - Osoft ≈ -ā Var(D)/Dbar^3`，相对偏差仍为
`O(Var(D)/Dbar^2) = O(1/D_eff)`。

## 14. 适用区间与两个误差来源

综合两部分，`Osoft` 相对“离散输出的真期望”有两级误差：

1. **门的下界误差（§3–§5）**：`g = exp(S) <= M`，使 `Gsoft <= E[Gdisc]`。
   随 softmax 锐化（匹配置信）单调缩小，one-hot 极限取等。
2. **归一化/mean-field 误差（§12–§13）**：`E[N]/E[D]` 对 `E[N/D]` 的
   替代，量级 `O(1/D_eff)`，随有效公共后缀数增大而缩小。分母中的 `+1`
   决定 `Dbar` 的尺度：`Dbar` 小（短上下文、稀疏匹配、`tau` 压低门值）
   时偏差大。

因此概率化 suffix matching 是「匹配置信 + 上下文足够长」区间里的可控近似，
而这恰是 DISM 面向的长程场景；在稀疏或不确定区间，它不能当作精确期望。
这也是为什么：

- 训练用 soft 作为可微代理、推理用 hard 的离散语义是合理取舍；
- `hard_prob` 退火让训练分布逐步逼近推理分布，但不能消除上面两级
  近似，只是让参数落在使两者都接近的区域；
- 不应把 soft 前向宣称为“hard 输出的无偏期望估计”，文献/文档里应保留
  “annealed 下界近似”的措辞。

数值示例（FP64，`m_ij ~ Bernoulli(M_ij)`，`M_ij = p` 于因果区，
每条 query 行独立 MC `3e5` 次，比较 quenched `E[N/D]` 与 annealed
`E[N]/E[D]`）：

```
N= 4 p=0.15  rel.gap=7.2e-01   |E[G]-Gbar|=8.3e-04
N= 4 p=0.50  rel.gap=3.1e-01   |E[G]-Gbar|=3.6e-03
N= 4 p=0.85  rel.gap=7.0e-02   |E[G]-Gbar|=4.2e-03
N=16 p=0.15  rel.gap=3.2e-01   |E[G]-Gbar|=1.9e-03
N=16 p=0.50  rel.gap=1.5e-01   |E[G]-Gbar|=5.6e-03
N=16 p=0.85  rel.gap=4.9e-02   |E[G]-Gbar|=1.0e-02
```

`|E[G]-Gbar|` 仅为 MC 噪声量级，印证 §11 的线性期望是精确的；
而 quenched 与 annealed 的差距在稀疏匹配（`p` 小）时可达数十个百分点，
随匹配变密/上下文变长而下降。

## 15. 精确表示：Laplace 变换 + mgf 线性递推

把每个 pair 的匹配看成 Bernoulli 门 `m_ij ~ Bernoulli(g_ij)`（理想情形
`g_ij = M_ij`；代码里 `g_ij = exp(S_ij) <= M_ij`，见第一部分），则可以用
一个恒等式把 quenched 平均写成没有除法的一维积分。对 `D > 0`：

```
1/D = integral_0^inf e^{-t D} dt
=>  E[N/D] = integral_0^inf E[ N e^{-t D} ] dt.
```

这给出 quenched 期望的**精确表达式**（仅依赖“不同 key j 的门相互独立”
这一层 mean-field）。引入 `G_ij` 的矩母函数

```
phi_ij(t) = E[ exp(-t G_ij) ],
```

由 `G_ij = m_ij (1 + G_{i-1,j-1})` 且 `m_ij` 与历史独立，得**线性递推**
（`phi = 1`, `phi' = 0` 为边界）：

```
phi_ij(t)   = (1 - g_ij) + g_ij e^{-t} phi_{i-1,j-1}(t)
phi'_ij(t)  = g_ij e^{-t} ( phi'_{i-1,j-1}(t) - phi_{i-1,j-1}(t) )
```

记 `m_ij(t) = E[G_ij e^{-t G_ij}]/phi_ij(t) = -phi'_ij(t)/phi_ij(t)`。
在 key 间独立的假设下 `D_i = 1 + sum_j G_ij` 的 mgf 可因子化：

```
E[N_i e^{-t D_i}]
  = e^{-t} ( prod_j phi_ij(t) ) * ( sum_j a_ij m_ij(t) ),
```

所以

```
E[O_i] = integral_0^inf e^{-t} ( prod_j phi_ij(t) ) ( sum_j a_ij m_ij(t) ) dt.
```

两点解释：

- **annealed 前向 = 该积分的“确定性 D”极限**。若把 `G_ij` 取成确定值
  `Gbar_ij`，则 `phi_ij(t) = e^{-t Gbar_ij}`、`m_ij(t) = Gbar_ij`，积分化为
  `sum_j a_ij Gbar_ij / (1 + sum_j Gbar_ij) = Osoft_i`。也就是说
  `E[N]/E[D]` 丢掉的正是 `D` 的涨落（`phi` 与 `m(t)` 的 `t` 依赖）。
- `phi_ij` 的递推是 O(1) 的，但要在 `Q` 个求积节点上得到
  `prod_j phi_ij(t)` 与 `sum_j a_ij m_ij(t)`，需要对每行做 O(N²)（与
  前向同阶）再乘 Q，因此**适合做 oracle/诊断，不适合训练内核**。

## 16. 有没有办法改善这个 bias？

先明确：问题只出在**分母**。分子 `E[N]` 与计数递推的线性期望是精确的
（§11），所以一切改进都围绕 `N/D` 的归一化非线性。下面按“是否改前向
数学 / 成本 / 适用区间”分类。

### (a) 不改变前向：把 `D` 的相对涨落压下去

bias 的相对量级是 `O(1/D_eff)`（§13）。任何增大有效匹配数 `D_eff`、
降低 `Var(D)` 的手段都会缩小它：更低的 softmax 温度 / 更大的 `tau`
（门更易开）→ 命中更多；更长的上下文（更大的 N）→ 每行累计更多后缀；
更大的 D / 词表 V → 匹配概率与计数尺度变化。

- 优点：零额外算子、零内核改动。
- 缺点：**只缩小、不消除** bias，而且会改变模型本身的匹配统计，
  不是无偏修正；温度/`tau` 本来是被训练的对象，不能当校正旋钮。

### (b) 确定性二阶（delta/cumulant）校正——最实用

不采样，只多跑一遍**二阶矩递推**。由 `G_ij = m_ij(1+G_prev)`、门独立：

```
E[G_ij]   = g_ij ( 1 + E[G_{i-1,j-1}] )
E[G²_ij]  = g_ij ( 1 + 2 E[G_{i-1,j-1}] + E[G²_{i-1,j-1}] )
V_ij      = E[G²_ij] - E[G_ij]²
```

再利用 §13 的二阶展开（key 间独立近似）：

```
Ocorrect_i = Nbar_i/Dbar_i
           - ( sum_j a_ij V_ij ) / Dbar_i²
           + Nbar_i ( sum_j V_ij ) / Dbar_i³  + O(third).
```

- 成本：只新增一阶矩递推同阶的标量递推（重 MMA 不用重算），
  依旧是前馈、确定性、可与现有行归约合并。
- 局限：扰动式，`Var(D)` 大（稀疏匹配）时展开不收敛，甚至可能给出
  越界值；且忽略了不同 `j` 的 `G_ij` 相关性（它们共享 q token），
  所以是同一层 mean-field 内的改进。
- 若采纳：必须是 **opt-in 模式**，与 FP64 oracle 对拍，默认不改。

### (c) 数值积分精确 quenched（不是近似）

用 §15 的公式，对 `integral_0^inf e^{-t} f(t) dt` 做 Gauss–Laguerre
求积：在每个节点 `t_q` 上算 `phi_ij(t_q)`、`m_ij(t_q)`，组装
`prod_j phi` 与 `sum_j a m`，再加权求和。

- 优点：在“key 间独立”这一层假设内，**精确捕捉 `D` 的涨落**，
  不再是逐阶近似；可作为检验 (a)/(b)/annealed 的 ground truth。
- 缺点：`Q × O(N²)`，比前向贵一个数量级；只适合离线 oracle / 小 N
  诊断。另外真实的跨 key 相关仍被忽略，所以它是“因子化模型的精确
  期望”，不等于原始离散过程的精确期望。

### (d) 随机硬采样 / REINFORCE——无偏但高方差

直接估 `E[O(m)]`：采 `S` 组 Bernoulli(g) 门，各跑一次硬前向，得
`(1/S) sum_s O(m^s)`。对参数 `theta` 的梯度用 score-function：

```
grad_theta E[O] = E[ O(m) * grad_theta log P_theta(m) ].
```

- 优点：期望无偏（对 `g` 定义的随机模型）；且可以把
deterministic 前向 `Osoft` 当作 **baseline / control variate**，
以降低方差（`Osoft ≈ E[O]`，是天然的低方差基线）。
- 缺点：需要 `S` 次完整 (硬) 前向，方差大、吞吐差；`hard_prob` 的
  行级混合**不是**这个估计量（它是前向模式的混合，不是同一样本上
  的平均），不能拿它冒充 quenched 期望。
- 适用：离线校准、小规模实验，不适合作为大规模预训练内核。

### (e) 直接改“目标”，而不是改估计量

- 把 annealed 前向**定义**为模型（训练就是在优化这个确定性函数），
  推理时用 hard，并把 soft/hard 的输出差当成已知的 train/inference
  gap 去度量，而不是声称训练在逼近 hard 期望。
- 用 `hard_prob` 退火：它让训练分布逐步靠向推理分布，但如 §14 所述
  不消除两级 bias，只调节参数落点。
- 可以在损失里加 `Var(D)` 或匹配熵的正则，鼓励集中匹配（变相实现 (a)），
  但会改变优化目标本身。

### 小结与建议

| 方法 | 是否改前向数学 | 成本 | 能否消除 bias | 适合场景 |
| --- | --- | --- | --- | --- |
| (a) 压涨落（温度/`tau`/长度） | 否 | 0 | 否，只缩小 | 常规训练 |
| (b) 二阶 delta 校正 | 是(opt-in) | ~2× 标量递推 | 仅小 `Var(D)` | 可选的训练修正 |
| (c) 数值积分 | 是 | `Q × O(N²)` | 在因子化假设内是精确 | oracle / 诊断 |
| (d) 硬采样 + REINFORCE(baseline `Osoft`) | 是 | `S` × 前向 | 期望无偏 | 离线校准 |
| (e) 重定义目标 | 否 | 0 | 不接受该 bias | 文档/评估口径 |

工程上的现实建议：训练继续用 annealed 前向作为“模型定义”，把 bias 当作
已知近似；若要更接近离散语义，(b) 是最值得先试的确定性修正，(c) 作为
FP64 oracle 用来量化 (a)/(b) 到底改进了多少。任何 (b)/(c) 实现都必须
是 opt-in、与现有 oracle/梯度测试对拍，并且不得改动默认 kernel 数学、
`tau` 语义或 hard 分支（AGENTS 约束）。

### 数值校验（FP64，`N=6`，Bernoulli 门，`3e5` 次 MC）

同一组输入下比较 annealed、二阶 delta 校正、Gauss–Laguerre 积分
（64 节点）对 MC quenched 的相对误差：

```
p=0.2:  rel(annealed)=4.8e-01  rel(delta2)=1.4e-01  rel(quad)=3.8e-03
p=0.5:  rel(annealed)=2.7e-01  rel(delta2)=4.1e-02  rel(quad)=3.4e-03
p=0.8:  rel(annealed)=9.2e-02  rel(delta2)=3.0e-02  rel(quad)=8.3e-04
```

- 积分结果与 MC 只差 MC 噪声量级（~1e-3），证实 §15 的 mgf/Laplace
  表示在“key 间独立”下确实给出精确 quenched 期望（前提：该 Bernoulli
  模型里不同 `j` 的匹配链互不相交）。
- 二阶 delta 校正把 annealed 偏差缩小约 3–9 倍，但**不归零**：它只是
  展开的首两项，且真实跨 key 相关与高阶涨落均未计入。
- 因此 (c) 适合做 ground truth 校验，(b) 适合做在线修正，两者不能混淆。

---

## 17. 真实 checkpoint 诊断：gap 随上下文增长，主因是可达匹配

对象：`dism-runs/dism-fwedu-valres-72m-2500m` 的 final checkpoint
`seqtrue72m-fwedu-valres @19074`，在 `fwedu-100B/sample/100BT` 的 val shard
上单卡评估（脚本 `diagnostics/soft_hard_diag.py`，只读 checkpoint）。
模型是 **DISM–GDN hybrid**（不是 DISM–SWA）：层 0–5 为 `PureGdnAttention`，
层 6–11 为 `DismGdnAttention`（`_combine_cuda` 把 DISM 原始输出与
`gdn_raw` 相加）；配置里的 `window_size=128` 对这种 attention 类型不生效。
`hard_granularity=sequence`，`hard_prob_max=0.95`，`direction_policy=true`。

### 17.1 复现

```
hard_bpb = 1.1828539530999698    soft_bpb = 1.1770186922534687
gap_bpb  = 0.0058352608465011    gap_loss = 0.014108401177942653
valid_tokens = 273352
```

与训练日志 `dual_validation.jsonl` 的最后一行逐位一致，说明评估口径正确。

### 17.2 gap 随文档内位置（上下文长度）增长

全量统计（`pos` = 到本 packed document 起点的 token 距离）：

```
 position     count     hard     soft      gap      gap 占比
 [   0, 32)    7610  3.31978  3.31976  +0.00002    0.0%
 [  32,128)   22830  2.92350  2.91784  +0.00566    3.4%
 [ 128,512)   75093  2.84743  2.83709  +0.01034   20.1%
 [ 512,1024)  73089  2.83317  2.81785  +0.01532   29.0%
 [1024,2048)  94730  2.83809  2.81877  +0.01933   47.5%
```

gap 随距离单调上升，`pos>=512` 合计约 **77%**。

### 17.2b 排除「hard 完全失配 → 零输出」后趋势仍在

hard 分支的原始 DISM 输出（`voc_dism` 结果，在加 `gdn_raw` 之前）在
**没有任何匹配**时严格为 `0`（分子为 0、分母为 `1+0`），此时 soft/hard
无差别，gap 必为 0。用 `_combine_cuda` 钩子取**首个**（层 6）和**末个**
（层 11）DISM 层的原始输出，按位置细分：

首个 DISM 层（层 6；其输入在 soft/hard 两次前向中完全相同，最干净）：

```
     position   count   P(O=0)   gap_all  gap_nonzero  m||Oh||  m||Os||
[   0,   16)    1628   0.2832  -0.00115     +0.00014   0.3031   0.2572
[  16,   32)    1635   0.0575  -0.00049     -0.00055   0.6205   0.5700
[  32,   64)    3262   0.0172  +0.00589     +0.00578   0.7517   0.7491
[  64,  128)    6533   0.0058  +0.00680     +0.00674   0.8932   0.9048
[ 128,  256)   12784   0.0013  +0.00404     +0.00404   1.0024   1.0056
[ 256,  512)   18283   0.0001  +0.01232     +0.01232   1.1176   1.1350
[ 512, 1024)   36541   0.0001  +0.01613     +0.01613   1.1994   1.2216
[1024, 2048)   56994   0.0000  +0.01844     +0.01845   1.1858   1.2199
```

末个 DISM 层（层 11）：

```
     position   count   P(O=0)   gap_all  gap_nonzero  m||Oh||  m||Os||
[   0,   16)    1628   0.1241  -0.00115     -0.00187   0.3116   0.2514
[  16,   32)    1635   0.0446  -0.00049     -0.00009   0.3558   0.2910
[  64,  128)    6533   0.0058  +0.00680     +0.00677   0.3514   0.2908
[ 256,  512)   18283   0.0001  +0.01232     +0.01232   0.3507   0.3177
[1024, 2048)   56994   0.0000  +0.01844     +0.01844   0.3115   0.3145
```

- `P(O_hard=0)` 只在最前面显著：首个 DISM 层 [0,16) 高达 **28.3%**，
  [16,32) 5.8%，[64,128) 已 <0.6%，之后≈ 0；这与“position 很小时
  hard 很容易完全失配、零输出就没有 gap”一致；
- **但排除这些零输出 token 后，`gap_nonzero` 仍从 `+0.006` 升到 `+0.018`**。
  所以该机制只能解释最前面一小段的 flat，不是整条单调上升的主因；
- 真正随上下文增长的量是**匹配质量**：首个 DISM 层的
  `m||Oh||` 从 `0.30`（[0,16)）单调涨到 `1.2`（[512,2048)），
  即可达匹配随上下文变多、变远；gap 与该量的走势一致。
- 短程 `m||Os|| < m||Oh||`（soft 匹配质量偏小，符合 `exp(S) <= M` 的
  Jensen 下界），长程两者趋同。

### 17.3 只出现在 DISM 层，且末层直接放大

逐层相对差异 `||soft-hard||_rms / ||soft||_rms`：

```
 layer   block     attn
  0-5    0.00000   0.00000     # GDN，与 soft/hard 无关
  6      0.02346   0.07893
  7      0.02628   0.07170
  8      0.01469   0.08606
  9      0.01246   0.06208
 10      0.02121   0.04733
 11      0.08601   0.08046     # 末层 attention 差异直接进 norm/LM head
```

GDN 层严格为 0；DISM 层 attention 输出差 ~5–9%，残差流 ~1–8.6%。
层 11 的 block 差异最大（8.6%），因为它之后只剩 norm+LM head，
attention 的 soft/hard 差不再被后续层“洗掉”。

### 17.5 匹配可达性分析：position 只是 DISM 活跃度的代理

从每次 `voc_dism` 调用捕获真实 `(q,k,q_vocab,k_vocab)`，重算
`idx_q=argmax(q·q_vocab)`、`idx_k=argmax(k·k_vocab)`，再用
`G_ij = m_ij(1+G_{i-1,j-1})`（`m_ij=1[idx_q(i)==idx_k(j)]`，同文档、因果）
得到每个 query 的**最大匹配跨距** reach 与**被匹配 key 数** nkeys，
对 6 个 DISM 层取最大值。（`diagnostics/soft_hard_diag_reach.py`）

```
 reach        count   mean_gap  mean_pos  mean_reach
[   0,  16)     212   -0.00338     8.4      6.9
[  16,  32)     190   +0.00758    25.1     23.4
[  32,  64)     386   +0.00465    49.3     47.6
[  64, 128)     763   +0.00645    97.4     95.6
[ 128, 256)    1531   +0.00577   193.3    191.4
[ 256, 512)    3055   +0.00758   385.1    383.2
[ 512,1024)    6090   +0.01673   769.4    767.4
[1024,2048)   11568   +0.01829  1514.0   1512.1
```

按被匹配 key 数分桶（同样 max over layers/heads）：

```
 nkeys        count   mean_gap  mean_pos  mean_nkeys
[ 0,  1)        94   -0.00191     4.5      0.43
[ 1,  2)       125   -0.00332    15.1      1.45
[ 2,  4)       258   +0.00398    36.7      2.93
[ 4,  8)       523   +0.00997    85.6      5.97
[ 8, 16)      1294   +0.00126   189.8     11.95
[16, 32)      2726   +0.01062   412.9     24.04
[32, 64)      5583   +0.01642   781.6     47.57
[>=64)       13192   +0.01706  1362.9    131.13
```

二维表（position × reach 组）：

```
      position       <64    64-255   256-511     >=512
[   0,  64)  +0.00329        --        --        --
[  64, 256)  -0.00026  +0.00635        --        --
[ 256, 512)        --  -0.03025  +0.00773        --     # 稀疏格，噪声大
[ 512,1024)        --        --  -0.01331  +0.01676
[1024,2048)        --        --        --  +0.01828
```

结论：

1. **DISM 几乎不活跃时 gap≈0**：`nkeys<2` 或 `reach<16` 的 token
   （典型就是文档开头）gap 为 `-0.002 ~ -0.003`（hard 略好）。
2. 随 `reach`/`nkeys` 增长，gap 单调上升到 `+0.018`。
3. `reach <= position` 是恒等式，且 `nkeys` 也随 position 增长，
   所以三个量高度共线，分桶均值几乎重合。二维表说明：固定 position
   时 gap 仍随 reach/nkeys 上升，固定 reach 时 gap 仍随 position 上升——
   二者都只是同一个潜在量（**DISM 实际做了多少匹配/匹配质量**）
   的不同侧面，而不是各自独立的因。
4. 结合 §17.2b 的 `m||Oh||`（层 6 从 0.30 涨到 1.2），可以认为：
   **gap ≈ DISM 分支的活跃度**；位置小只是活跃度低（候选少/无匹配）的
   常见情形，你的“完全失配→无 gap”是它在低端的极端表现。

### 17.6 对缓解方案的含义

1. gap 只在 DISM 实际做匹配处出现，因此**锐化 soft 门**（§1 温度退火 /
   §2 分布锐化）正好作用在产生 gap 的地方；下游 hard 微调只能吸收残差。
2. 评估要按位置/活跃度分桶（并区分零匹配 token），总体均值会稀释。
3. 层 11 直接放大差异，参数分组实验优先看末层 DISM 的
   `q_vocab/k_vocab/log_sel_tau`。

> 说明：这是单一 checkpoint / 单一 val shard 的结果，用于定位偏差来源，
> 不代表其它规模或数据上的普适数值；二维表中 `--` 或样本极少的格噪声较大。

---

## 18. Q/K 词表 sharpness 与 gap 的关系

同一 checkpoint，捕获每个 DISM 层真实的 `(q,k,q_vocab,k_vocab)`，对每个 head 计算
`softmax(q·q_vocab)`、`softmax(k·k_vocab)` 的统计
（`diagnostics/soft_hard_diag_sharpness.py`）。V=512，lnV=6.238。

```
layer  normH_q  normH_k   pmax_q   pmax_k   marg_q   marg_k   effV_q   effV_k   amb_q   amb_k
    6   0.3206   0.1821   0.0988   0.2299    0.314    0.707     7.39     3.11  0.4001  0.2708
    7   0.3263   0.1644   0.1004   0.2509    0.358    0.736     7.66     2.79  0.3837  0.2533
    8   0.2970   0.1783   0.1119   0.2250    0.313    0.589     6.38     3.04  0.3973  0.2853
    9   0.2866   0.1632   0.1199   0.2484    0.364    0.700     5.98     2.77  0.3772  0.2557
   10   0.1809   0.1093   0.2194   0.2946    0.505    0.743     3.09     1.98  0.3190  0.2387
   11   0.1653   0.1607   0.2213   0.2162    0.508    0.493     2.80     2.73  0.3033  0.3170
```

- `normH = H/lnV`（0=one-hot，1=uniform），`effV = exp(mean H)`（有效词表），
  `marg = top1-top2 logit margin`，`amb` = margin<1 的 token 比例。
- 词表分布**远非 one-hot**：层 6 有效词表 Q≈7.4、K≈3.1，层 11 收敛到 ~2.8；
  `pmax` 只有 0.10–0.29，`marg` 只有 0.31–0.74 nat，30–40% 的 Q token
  是近似平手（margin<1）。
- Q 在**前几个 DISM 层明显比 K 更平**（effV 7.4 vs 3.1），到层 11 才趋同。
- 这意味着 `S = E_{p_k}[log p_q]` 里 `log p_q` 的取值散布很大，`S` 明显
  小于 0，soft 门 `exp(S)` 系统性低于 hard 的 1——与 §17 里短程
  `m||Os|| < m||Oh||` 一致。

### 18.1 gap 与 sharpness 几乎不相关

按**每 token 的 head 平均 top1 margin**（层 11，hard 前向）分桶看 gap：

```
 margin       cnt_q    gap_q     cnt_k    gap_k
[0.00,0.20)     93   -0.02546      98   +0.11382   # 极少，噪声
[0.25,0.50)   1908   +0.01832    2385   +0.00664
[0.50,1.00)  15083   +0.01465   15709   +0.01080
[1.00,2.00)  13845   +0.01199   12695   +0.01761
[2.00,4.00)    622   +0.01078     663   +0.00964
```

gap 在各 margin 桶近似常数（+0.010~+0.018），没有明显单调关系。
即：**gap 不是由个别 token 的 argmax 模糊驱动的，而是系统性的**——
所有 token 的码字分布都同样“宽”（effV 3–8），soft 门处处偏小，
在匹配活跃（§17）处累积成 gap。

### 18.2 对缓解方案的含义

- 有明确的锐化空间（effV 3–8 → 可往 1–2 压），且因为偏差是系统的，
  **全局**降温/锐化就能同时作用于所有 token，而不是只修个别模糊位置。
- 但 §18.1 说明不能用「per-token margin」当自适应锐化依据；应做成
  per-head/per-layer 的全局温度或尖锐度参数。
- 需要验证的风险：过度锐化会让 `q_vocab/k_vocab` 梯度饱和、词表坍缩，
  反而损失码字容量；建议慢退火并监控 hard NLL 与码字使用分布。

---

## 19. β 温度：实现与锐化微调方案

### 19.1 已接好的接口

- Triton 插值本来就接受 `sm_scale`（=温度），所以 **CUDA core 不变**。
  关键一步是把插值嵌入再乘 β，让 core 的 `raw - LSE` 变成
  `β·raw - LSE_β = E_{p_k^β}[log p_q^β]`。
- `DismAttention`/`DismConfig` 新增 `vocab_temperature`（默认 1.0），在
  nanochat `DismLM` 里是 **non-persistent buffer**（不进 state_dict，旧
  checkpoint 可直接加载），并有 `set_vocab_temperature(value)`。
- eager 路径：`voc.py`/`varlen.py` 走 `emb_kernel.interp_embeddings(..., beta)`，
  β 在外层自动微分里缩放。
- compiled 路径：`compiler.py::voc_forward/voc_backward` 把 β 作为 tensor
  输入，forward 保存未缩放的 `qfk/kfq`、core 用 `β·E`，backward 用
  `doq*β, dok*β` 并传 `sm_scale=β`。
- 参考实现 `dism_v3_ref.dism_wrapper(beta=...)`；测试见
  `tests/test_dism_v3_ref.py::test_beta_oracle`（β∈{1,2,5,20} 对拍 FP64，
  β=1 逐位不变，β 增大软门单调逼近 hard）。CPU 上 47 个 reference 测试通过。
- 数值恒等式：`β·raw - LSE_β` 与 `E_{p_k^β}[log p_q^β]` 误差 ~1e-15；
  match 时 `exp(S)→1`、mismatch 时 `→0`（β=20 已 1e-5/5e-39）。

### 19.2 第一步：zero-training β 扫描（已完成）

`diagnostics/beta_sweep.py`（batch16, 262144 valid tokens, peak ~1.0GB）：

```
hard: bpb=1.182854  loss=2.859886
 beta   soft_bpb  soft_loss    gap_bpb   gap_loss
 1.00   1.177019   2.845777  +0.005835  +0.014108   # 与训练 dual_validation 一致
 1.50   1.177559   2.847084  +0.005295  +0.012802
 2.00   1.178268   2.848799  +0.004585  +0.011087
 3.00   1.179323   2.851349  +0.003531  +0.008536
 5.00   1.180321   2.853760  +0.002533  +0.006125
10.00   1.181465   2.856526  +0.001389  +0.003359
20.00   1.182204   2.858314  +0.000650  +0.001571
```

逐位置 gap_loss（hard − soft）：

```
 position      b=1     b=2     b=5    b=20
 [   0,128)  +0.0043 +0.0035 +0.0017 +0.0003
 [ 128,512)  +0.0103 +0.0081 +0.0041 +0.0013
 [ 512,1024) +0.0153 +0.0123 +0.0070 +0.0019
 [1024,2048) +0.0193 +0.0149 +0.0084 +0.0020
```

结论：

- 锐化让 soft 单调逼近 hard，gap 单调下降；β=20 时总 gap 只剩原来的
  ~11%。这是“门 bias 被锐化消除”的直接验证。
- **但残余 gap 不归零，且仍随位置增长**（β=20 时 0.0003→0.0020）。
  这正是 §12–13 的 mean-field 归一化 bias：锐化固定了单对门，但
  annealed 分母的偏差随候选数（上下文长度）增长，与 β 无关。
- 微调目标温度：β\*≈5–10（gap 已降到 20–45%，梯度尚未饱和）。
- 注意扫描里 soft_bpb **变大**（soft 被拉向较差的 hard）；这只能证明机制，
  不能证明 hard 变好。hard 是否受益要看 §19.3 的微调。

### 19.2b 加载路径的坑（已修）

`build_model` 用 `to_empty() + load_state_dict(assign=True)`。non-persistent
buffer 不在 state_dict 里，`to_empty` 后是未初始化内存，被读成 `0.0`，
导致插值嵌入被乘 0。已在 `DismLM.init_weights` 里重置
`vocab_temperature`（模型与各 attention）。修复后 hard/soft/gap 与训练日志
逐位一致。

### 19.3 第二步：β 退火续训

从 checkpoint 续训一小段，让匹配参数在“更接近 hard 的代理梯度”下再学一次：

1. 新优化器（保留 no-decay 分组），LR ≈ 0.1–0.3×base，warmup~20、
   warmdown 到 0.1；继续 `--grad-clip 1.0` 或更紧。
2. `hard_prob` 暂时降到 0（全 soft，保证每行都有匹配梯度），
   β 从 1.0 缓慢退火到 β\*（几何），约 500–1000 步。
3. 到 β\* 后可冻结 `q_vocab/k_vocab`（此时 softmax 已接近饱和，梯度小且
   易坍缩），再用 `hard_prob→1` 跑 ~200 步让下游适应 near-hard 分布。
4. 每 50 步记录 hard_bpb / soft_bpb / gap，以及按位置的 gap（用 §17 的
   分桶）。目标是 hard_bpb 低于基线 1.18285，且 gap 缩小。

风险与监控：
- β 太大 → softmax 饱和、`q_vocab/k_vocab` 梯度消失或码字坍缩。监控
  vocab grad norm 与码字使用分布（§18 的 sharpness 脚本）。
- 匹配对梯度随 β 放大，grad norm 可能震荡；慢退火 + clipping + 更小 LR。
- 若 hard NLL 不降反升，说明 soft 的优势主要来自“隐式 ensemble 平均”，
  此时锐化只会把 soft 拉向较差的 hard，应考虑推理侧 ensemble 而非微调。

### 19.4 可选：显式一致性损失

CE 在 β 很大时退化为 hard（无匹配梯度）。若想让软/硬差异被直接惩罚，
可加一项 stop-grad 一致性损失：

```
L = CE(mix) + lambda * D( soft_beta_logits, stopgrad(hard_logits) )
```

`D` 用 KL 或对 attention 输出做 MSE；hard 侧只提供目标、不回传匹配梯度，
soft 侧提供可微对齐。可与 β 退火叠加，lambda 也退火到 0。

验证命令（等 GPU 空闲）：先跑 `diagnostics/beta_sweep.py`，再看
`tests/test_module_dynamo.py::test_opaque_voc_beta_matches_eager`
（eager vs compiled，β∈{1,2,5}，含 backward 对拍）。

### 19.5 续训脚本（已就绪，已小 batch 试跑）

`finetune_sharpen.py`（run 目录下；不写原 checkpoint，输出到
`<base_dir>/finetune_checkpoints/<out-tag>`）。调度：

```
beta(t)      = beta_start * (beta_end/beta_start) ** min(step/beta_warmup, 1)
hard_prob(t) = hp_start + (hp_end-hp_start) * min(step/hard_warmup, 1)
```

关键参数：`--beta-start 1 --beta-end 8 --beta-warmup 300`、
`--hard-prob-start 0 --hard-prob-end 0.5 --hard-warmup 300`、
`--lr 2e-4 --grad-clip 1.0`、`--freeze-vocab-frac 1.0`（设 <1 在那之后冻结
`q_vocab/k_vocab`）、`--eval-every 50 --eval-tokens 131072 --save`。
每步打印 `loss/gnorm/vocab_gnorm`，评估打印 `hard_bpb/soft_bpb/gap`。

试跑（6 步, batch2, 4.2s）正常：训练、调度、评估、梯度监控都通。
正式跑建议 `--batch 8 --steps 500`；若 hard_bpb 不降，说明 soft 优势
主要来自隐式 ensemble，而不是可训的匹配代理。

---

## 20. Warm-started REINFORCE 续训方案

### 20.1 之前两次失败的原因（已核对远程实验）

- **方案 A（soft-loss baseline，`dism-reinforce-code`）**：从 scratch、step0 起全
  categorical、**无 soft 梯度（`soft_gradient:false`）、无 clip、无 reward 归一化**。
  轨迹分析：采样熵 6.09(step0)→2.84(300)→0.43(700)→0.13(1000)→0.046(3000) nats；
  loss 在 300–700 反而升到 5.15→6.78；grad norm 中位数 11.5，step679 最大 82.6。
  最终 argmax NLL 3.915，而同规模同 token 的 soft→argmax 退火基线 3.098。
  即：**策略在 ~700 步内塌成近确定性，`∇log p→0`，selector 失去训练信号**；
  没有 soft pathwise 项兜底，码字只能靠 direct hard 梯度撑着。
- **方案 B（K-side residual control variate，`dism-residual-code`）**：RELAX/REBAR
  式 proxy `u=∂L_soft/∂h̄`，但第 7 步 `proxy_upstream_rms≈3e-10`、
  `cv_surrogate≈1e-6`，**control variate 实际是死的**，且实验被提前 cancel。
  所以“control variate 降方差”这条路其实没被真正测过。

结论：两次都**不是**“soft 训完再 REINFORCE”，而是从零、且缺少防塌机制。

### 20.2 最保险的设计（已实现）

```
L = L_soft(β)                                         # 主信号：soft pathwise（低方差）
  + λ · mean_symbol( log π(z) · Â )                   # REINFORCE 修正项
  + w · L_hard                                        # 可选：采样 hard 的 direct 均值
  + η · relu(H_floor − mean_symbol H(π))              # 熵下限，防塌
```

其中 `Â = standardize_per_document(causal_return(hard_NLL − soft_NLL))`，
`π = softmax(β·q@vocabᵀ)`（与 soft 分支同温度）。tricks：

1. **保留 soft pathwise 梯度**（与失败方案最本质的区别）；
2. **warm start**：从 soft 训练好的 checkpoint（熵健康，effV 3–8）；
3. λ 小且线性 ramp（`--lam-warmup`），REINFORCE 只做修正；
4. advantage **按文档标准化**（去均值/单位方差）；
5. **熵下限** `η·relu(H_floor − H)`，只在塌缩趋势时生效；
6. 梯度裁剪；监控熵与 policy 幅值。

### 20.3 移植与接口

- 新增 `dism_v3/python/flash_dism/reinforce.py`：`collect()` / `enabled()` /
  `Categorical`（采样符号，返回 logp 与 entropy，自定义 backward 给出
  `dlogp/dlogits=onehot−p` 与 `dH/dlogits=−p(logp+H)`）/ `sampled_voc`。
- `module.py::_forward_cuda`：`if enabled(): sampled_voc(...)` 分支；生产路径不受影响
  （`enabled()` 默认 False）。
- `nanochat/models/dism/model.py`：新增 `loss_reduction='reinforce'`，返回
  `(differentiable_mean_NLL, detached_per_token_NLL)`。
- 新增 `nanochat/nanochat/reinforce_loss.py`：`causal_returns` /
  `standardize_per_document` / `reinforce_loss`。
- `finetune_sharpen.py`：`--lam --lam-warmup --eta --entropy-floor --hard-weight`，
  默认 0 → 纯锐化；`--lam>0` 才走采样路径。

### 20.4 建议跑法

先纯锐化确定 β\*，再开 REINFORCE 修正：

```bash
# 纯锐化（先跑）
python finetune_sharpen.py --steps 500 --batch 8 \
  --beta-end 8 --beta-warmup 300 --hard-prob-end 0.5 --eval-every 50 --save
# 加 REINFORCE 修正（β 保持在中等，别推到饱和）
python finetune_sharpen.py --steps 300 --batch 4 \
  --beta-start 1 --beta-end 3 --beta-warmup 100 \
  --hard-prob-start 0 --hard-prob-end 0 --lam 0.05 --lam-warmup 100 \
  --eta 0.1 --entropy-floor 3.0 --grad-clip 1.0 --eval-every 25
```

监控红线：采样熵跌破 ~1 nat、grad norm 持续 >5、hard_bpb 上升 → 立即停。
若 pure-sharpen 已能降 hard_bpb，REINFORCE 只当锦上添花；反之才说明 soft 代理
的偏置是瓶颈。

---

## 21. 锐化微调 / warm-start REINFORCE 实验结果

设置：warm start = `seqtrue72m-fwedu-valres@19074`，batch8，val shard 用 batch8
packing。**同 packing baseline：hard=1.180755，soft(β=1)=1.174803，gap=0.005952**
（batch16 packing 下为 1.182854/1.177019，数值不同不可直接比）。

### 21.1 学习率是第一个坑

```
配置(100 steps)                       hard_bpb    解释
baseline(不训练)                       1.180755    —
β=1 (无锐化)        lr=2e-4           1.194992    纯漂移 +0.014
β1→5               lr=2e-4           1.194741    与上面相同 => 退化来自 LR
β1→5               lr=3e-5 (200步)    1.180353    基本不变
```
收敛后的 checkpoint 续训，`lr≥2e-4` 会明显漂移；必须用 `≤3e-5`。

### 21.2 纯锐化不改善 hard

```
配置                               hard_bpb   soft_bpb  gap
baseline                            1.180755   1.174803  0.005952
β1→4, lr5e-5, 600步                 1.181355   1.177839  0.003516
β1→4, λ=0.3 REINFORCE, 600步        1.184087   1.179594  0.004493
β1→8, lr3e-5, 200步                 1.180353   1.175887  0.004465
```
共同模式：**hard 基本不动（±0.003，接近 eval 分辨率），soft 变差，gap 被压小**。
即锐化是通过“把 soft 拉向已经较差的 hard”来闭合 gap，而不是抬高 hard。

### 21.3 warm-start REINFORCE：稳定了，但没有收益

关键正面结果：之前的塌缩没有再现。
```
step0   ent=2.649
step100 ent=3.133  (熵下限生效，policy=+0.13)
step300 ent=1.495  (β 升到 4 后仍下降，但未崩)
gnorm  ~0.55-0.67，无爆炸
```
即 warm-start + soft pathwise 主体 + 熵下限确实解决了从零 REINFORCE 的
熵塌缩/梯度爆炸问题。但 hard_bpb 在 1.1812–1.1841 间波动，没有超过 baseline。

### 21.4 结论

1. **hard/argmax 路径在预训练结束时已接近其最优**（预训练最后 hard_prob=0.95），
   在同一数据上继续训既不会靠锐化、也不会靠 score-function 改善它。
2. 现有 gap 是**内在的 soft↔hard 评估差**：soft 是更平滑/更接近 ensemble 的模型，
   泛化略好；不是“训练不足”。锐化只能把这个优势抹掉。
3. REINFORCE 的稳定性问题已经解决（§20 的方案），但**没有 hard 侧的提升空间**
   可挖，因此没有收益。
4. 想真正提升离散推理质量，方向应是**推理侧**：多视点 hard 的 ensemble/平均，
   或直接部署 soft（若效率允许）；或在**新数据**上继续训练。继续在同一数据上
   做选择器微调，收益已到顶。

> 说明：以上为单一 72M checkpoint / 单 val shard 的结果；eval 分辨率约
> 0.001 bpb，小于该量级的差异不应过度解读。

---

## 22. 推理侧 hard ensemble 实验

对同一 val shard（batch8, 262144 valid tokens）比较：

```
argmax-hard           bpb = 1.182854   (等于训练 val_bpb)
soft beta=1           bpb = 1.177038   gap vs hard = +0.005816
soft beta=4           bpb = 1.179925   gap vs hard = +0.002929

 beta   K   sampled-hard 平均logits bpb   gap vs hard
  1.0   1            1.189865             -0.007011
  1.0   4            1.185623             -0.002768
  1.0   8            1.185109             -0.002255
  1.0  16            1.184761             -0.001906
  4.0   1            1.183475             -0.000621
  4.0   4            1.181650             +0.001204
  4.0   8            1.181293             +0.001561
  4.0  16            1.181220             +0.001634
```

结论：

1. **β=1（按 soft 分布采样）时 ensemble 比 argmax 更差**（单次采样 -0.007，
   K=16 仍 -0.0019）：argmax 是众数，随机采样探索的是更差的配置，平均收不回来。
2. **β=4（更接近 argmax 但有多样性）时，K≥4 的 logit 平均超过 argmax**
   ~+0.0012~0.0016 bpb，且 K=8 后饱和。
3. 但即使最好的 hard ensemble（β=4, K=16 → 1.181220）也**明显不如
   soft β=1（1.177038）**，只拿回 ~28% 的 gap；而且比 soft β=4 本身还差。
4. 代价是 K 倍推理（每个 token 跑 K 次 hard 匹配），直接抵消 DISM 的效率优势。

**综合判断**：gap 里只有约 1/4 来自“hard 单次采样 vs soft 平均”的方差，可以用
K 倍推理的 ensemble 拿回；剩下约 3/4 是 soft 本身作为更平滑模型的
系统性优势，推理侧平均抹不平。若部署只在乎质量、不在乎 K 倍算力，用
hard ensemble(β≈4, K≈8) 可拿到 ~0.0016 bpb；否则 soft(β=1) 仍是质量上限，
而 argmax-hard 是效率/质量折中。

> 说明：单一 checkpoint / 单 val shard；eval 分辨率约 0.001 bpb。

---

## 23. 长时间 REINFORCE（3000 步）——趋势是对的

配置：warm start valres@19074，batch8，lr5e-5(warmup50, final0.1)，β 1→2，
hard_prob=0，λ=0.3(ramp200)，η=0.5（熵下限 2.5），3000 步，每 200 步 eval。
baseline：hard=1.180755，soft=1.174803。

```
 step   hard_bpb   soft_bpb   gap
  200   1.183177   1.175811   0.007366
  400   1.186958   1.179365   0.007593
  600   1.191404   1.182843   0.008561   <- 触底
  800   1.187777   1.179310   0.008467
 1000   1.185701   1.177681   0.008020
 1400   1.185112   1.177038   0.008074
 1800   1.184089   1.176160   0.007929
 2200   1.183574   1.175700   0.007874
 2600   1.182963   1.175343   0.007620
 3000   1.182860   1.174998   0.007862
```

关键更正：**不是单调退化，而是"初始 transient 下探 → 之后持续改善"**。
step600→3000 的 2400 步里 hard 单调降了 0.0085；soft 也回到 1.174998
（baseline 1.174803）。到 3000 步 hard 仍比 baseline 高 0.0021，但斜率稳定为负
（≈ −3.6e-6/step），趋势指向可能穿过 baseline。

熵稳定在 2.5–3.0（未塌缩），policy 活跃（+0.05~+0.58），gnorm ~0.55–0.64。
说明 warm-start 方案能长时间稳定训练，且确实在缓慢降低 hard-soft 差异。

下一步（进行中）：从 `finetune_checkpoints/rf-long-b2-lam03/model_003000.pt`
继续 3000 步（避开初始 dip），并跑同配置 λ=0 对照做归因。

### 23.1 对照：λ=0 才是最佳的（结论修正）

补跑同配置 λ=0 对照（β=2 常数, lr3e-5, 3000 步, 从原始 checkpoint）：

```
配方(3000步)                       hard_bpb    soft_bpb    gap
baseline                          1.180755    1.174803    0.005952
λ=0    β=2, lr3e-5                1.179500    1.173022    0.006478   <- 最佳
λ=0.3  β1→2, lr5e-5, η0.5         1.182860    1.174998    0.007862
λ=0.3  续训(从上一行恢复点), lr3e-5 1.184882    1.176452    0.008431
```

要点：

1. **λ=0 对照从头单调改善**：1.180755 → 1.179500（−0.0013），无 transient，
   soft 同步改善。之前"锐化不改善 hard"的结论是短步数/高 LR 的假象。
2. **第一个长 REINFORCE run 的下降趋势是"恢复"而非"学习"**：从它的恢复点用
   同样 λ=0.3 重启，又掉到 1.185 并停住，并未继续下降 → 之前是在抹掉
   REINFORCE 造成的扰动。
3. **所有 λ>0 的 run 都差于同配置 λ=0**（+0.003~+0.005），且 gap 更大
   （0.008 vs 0.006）。
4. 结论：**要提升 hard，用纯 soft 温度微调（β≈2、lr≈3e-5、hard_prob=0、
   数千步），不要 REINFORCE**。收益 ~0.0013 bpb，单调且稳定；gap 不变，
   说明是整体质量提升而非专门闭合 soft/hard 差。

### 23.2 拉长 + 继续锐化（6000 步，λ=0）

从原始 checkpoint 一次跑（避免 restart 混杂），batch8，lr3e-5，hard_prob=0：

```
配方(6000步)                  hard_bpb   soft_bpb   gap
baseline                     1.180755   1.174803   0.005952
flat   β=2 常数              1.179970   1.173164   0.006806
ramp   β 2→4 渐进            1.179137   1.175029   0.004109
```

ramp 内部 β 2.83→4.0 的轨迹：hard 1.179952→1.179512→1.179415→1.179137
（单调降），gap 0.0053→0.0047→0.0044→0.0041（单调收）；flat 在 β=2 停在
~1.180 平台。

**结论：继续锐化（β 2→4）比单纯延长训练更有效** —— 额外带来 ~0.0008 bpb
的 hard 改善，并把 gap 从 0.0068 收到 0.0041。这与"锐化降低了 soft 的偏差、
使 soft 代理更接近 hard，从而 soft 梯度更好地改善 hard"的假设一致。

### 23.3 继续锐化到 β=6（9000 步，λ=0）

从原始 checkpoint 一次跑，β 2→6 线性（几何）升温，batch8，lr3e-5：

```
 step   beta   hard_bpb   soft_bpb   gap
  500   2.13   1.180282   1.174702   0.005580
 3000   2.88   1.180329   1.175101   0.005228
 5000   3.68   1.179976   1.175879   0.004097
 6000   4.16   1.179770   1.175968   0.003802
 7000   4.70   1.179383   1.176030   0.003354
 8000   5.31   1.179307   1.176212   0.003095
 9000   6.00   1.179220   1.176528   0.002692
```

- β 从 ~3 以后，hard 随 β 单调下降：1.1803(β2.9) → 1.1792(β6.0)；
  gap 从 0.0052 单调收到 **0.00269（腰斩）**。
- 对比 6000 步 β2→4：final hard 1.179137、gap 0.0041。两者 hard 相当
  （~1.1791–1.1792，差在 run 间 ~0.0005 波动），β=4 已经拿到大部分收益，
  β 4→6 主要在继续收 gap。
- 注意：这是在**训练中升温**。直接在原始模型上评估高 β 只会让 soft 变差
  （§19.2 β=20 soft=1.1822）；训练会适配更尖锐的温度。

### 23.4 最终结论（锐化 vs REINFORCE）

| 方案 | hard_bpb | gap | 备注 |
| --- | --- | --- | --- |
| baseline | 1.180755 | 0.005952 | |
| REINFORCE λ=0.3（3000–6000 步） | 1.183–1.185 | 0.008 | 稳定但有害 |
| 纯 soft β=2 常数 6000 步 | 1.179970 | 0.006806 | 小改善 |
| **纯 soft β 2→4 6000 步** | **1.179137** | 0.004109 | 最佳 hard |
| **纯 soft β 2→6 9000 步** | **1.179220** | **0.002692** | gap 腰斩 |

1. **要提升 hard：纯 soft 温度微调**（β 2→4~6、lr≈3e-5、hard_prob=0、
   数千步），收益 ~0.0015 bpb，并显著收窄 soft/hard gap。
2. **不要 REINFORCE**：warm-start 后它稳定（不塌缩）但一致伤害 hard；长跑只是
   在恢复它造成的扰动。
3. **不要从微调中途 checkpoint restart 续训**：fresh optimizer 会扰动模型
   （~+0.002），长跑也难恢复；要"更长"就在同一个 run 里跑更长。
4. 机制解释：升温让 soft 门更接近二值、soft 的 Jensen/mean-field bias 变小，
   于是 soft 代理梯度更接近 hard 的真实方向，训练出的参数对 argmax 也更好；
   gap 的收窄正是 bias 被缓解的直接证据。

---

## 24. 从头预训练：β 余弦 + hard_prob 两段调度（对照 valres）

对象：`dism-runs/dism-fwedu-valsched-72m-2500m`，配置与
`dism-fwedu-valres-72m-2500m` 完全一致（72M、12 层、19074 步、batch16、
total_batch131072、lr1.5e-3、warmup500、warmdown_ratio0.9738、WD0.01、
direction=true、rear_half_dism、post_norm、value_residual、DataLoader
shuffle seed1337），只改调度：

```json
"vocab_temperature_start": 1.0, "vocab_temperature_end": 4.0,   // beta 余弦
"hard_prob_start": 0.1, "hard_prob_max": 0.95,
"hard_start_fraction": 0.8, "hard_end_fraction": 1.0
```

`DismLM.set_training_step` 实现：

```
progress      = step/(anneal_steps-1)
hard_progress = clamp((progress - 0.8)/0.2, 0, 1)
hard_prob     = 0.1 + (0.95-0.1)*hard_progress          // 前80%固定0.1，后20%线性
beta          = 1 + 0.5*(4-1)*(1 - cos(pi*progress))    // 余弦 1->4
```

验证（配置实例化）：

```
step      0  hard_prob=0.100  beta=1.0000
step   9537  hard_prob=0.100  beta=2.5001
step  15259  hard_prob=0.100  beta=3.7136
step  17167  hard_prob=0.525  beta=3.9267
step  19073  hard_prob=0.950  beta=4.0000
```

设计依据（§23）：**β 领跑、hard_prob 收尾**。前 80% 保留 0.1 的 hard 曝光让
readout 不完全脱离离散路径，但主体是 β 余弦升温（可微、selector 有梯度）；
最后 20% 才把 hard_prob 拉到 0.95 做精确对齐。

产物：`train.sh`、`valsched_config.json`、wandb
`dism-simlpajama/seqtrue72m-fwedu-valsched-2500m`（run `1bbsmlo9`）；
`dual_validation.jsonl` 每 500 步记录 hard/soft/gap。对照基线
valres final：hard=1.182854, soft=1.177019, gap=0.005835。

---

## 25. 计划中的对照组：全程同步退火（valsync）

`dism-fwedu-valsync-72m-2500m`，与 valres/valsched 同架构同超参，调度改为
**hard_prob 与 beta 全程并行退火**：

```
hard_prob(p) = 0.95 * p                          // 前 100% 线性，等同 valres
beta(p)      = 1 + 0.5*(4-1)*(1 - cos(pi*p))     // 全程余弦 1->4
```

验证：step0 (hp0,β1)、9537 (hp0.475,β2.50)、15259 (hp0.760,β3.71)、
19073 (hp0.950,β4.00)。

启动方式：`queue_wait.sh` 轮询 valsched 的 PID，等它退出后自动
`bash train.sh`（wandb run `seqtrue72m-fwedu-valsync-2500m`）。

### 25.1 valsched 中期诊断（step 11000–11500）

```
step   valres gap   valsched gap   hard_prob    beta
10000   0.0126        0.0398       0.50/0.10    2.61
11000   0.0112        0.0419       0.55/0.10    2.86
11500   ~0.011        0.0401       0.57/0.10    3.00
```

- valsched gap 在 step~11000、β 越过 3 时**见顶回落**（0.0419→0.0401），
  与"β 变尖能压 gap"一致；但相对 valres 仍高 3–4 倍。
- 根因是 **hard 路径只拿 10% 信号、欠训**：同 step 下 valsched hard
  (1.258) 远差于 valres (1.236)，soft 略好 (1.218 vs 1.225) —— 用 hard
  质量换了 soft 质量。
- 结论：**hard_prob 不能推迟**；从头训练必须全程给 hard 曝光，β 只应作为
  叠加的"软化偏置修正"。valsync 正是验证这一点。

---

## 26. 优化器/调度实验汇总（预训练 + 微调 + PG19 长文本）

### 26.1 预训练四组（batch16 dual eval, step 19074）

同架构、同超参、同数据顺序，只改调度/优化器：

```
run                                       hard_bpb    soft_bpb    gap
valres  (β=1, hard 0→.95)                 1.182854    1.177019    0.005835
valsched(hard 推迟到后20%, β 余弦→4)       1.189378    1.177680    0.011698
valsync A (hard 全程, β 余弦 1→4)          1.183907    1.176767    0.007140
valsyncB (同 A + selector AdamW β1=.95)   1.182748    1.177316    0.005432
```

- 预训练期 β 退火**单独看是负收益**（A 相对 valres hard +0.001、gap +0.0013）。
- **selector β1=0.95 是真正增益**：B 相对 A，hard −0.0012、gap −0.0017；B 相对
  valres 也最终赢了 hard 与 gap。
- `valsched` 再次确认 hard_prob 不能推迟。

### 26.2 微调（warm-start β 锐化，batch8）

```
配方                                     hard_bpb   soft_bpb   gap
valres 基座 (β=1)                        1.180755   1.174803   0.005952
valsyncB 基座 (预训练 β=4)                1.181171   1.196474  -0.015303  ← β=1 soft 崩
E1 = valres  + β 1→6, 6000 步            1.178955   1.176066   0.002890
E2 = valsyncB+ β 4→6, 6000 步            1.181929   1.173447   0.008482
```

- **E1 相对 valres 基座 −0.0018 hard**；**E2 相对 valsyncB 基座 +0.0008（不升反降）**。
- `valsyncB` 在 β=1 下 soft 从 1.175 崩到 1.196，直接证明**模型缩了 logit 尺度去
  适应 β=4**（温度与尺度冗余）。预训练 β 退火把锐化潜力提前吃掉了 →
  **最佳 pipeline 是：预训练 β=1（或 + selector β1），锐化留给微调。**

### 26.3 PG19 长文本（downstream-eval，外推到 16k）

`bits/byte`（当前 adapter；`fwedu_valres` ctx2048 重跑修正，旧值 1.5013 是 stale）：

```
model              ctx2048   ctx4096   ctx8192   ctx16384
fwedu_valres       1.4374    1.4204    1.4104    1.4047   ← 最好
fwedu_valsyncB     1.4399    1.4227    1.4122    1.4063
E1_valres          1.4406    1.4231    1.4128    1.4069
E2_valsyncB        1.4431    1.4262    1.4161    1.4105
(slimpajama/fwedu baseline 更差，见 downstream-eval/results/PG19_SUMMARY.md)
```

**所有 context 上 valres 基座最好，β 锐化的模型依次变差**（~0.003–0.006 bpb）。
即**锐化在域内 val_bpb 上更好、在域外长文本上略差**——是个 trade-off，不是纯增益。

### 26.4 锐化到底做了什么：DISM 启用度统计

对 valres / E1 / valsyncB / E2 在 val 集上统计（`_combine_cuda` 抓原始 DISM 输出）：

```
              zero_frac  |DISM|  |GDN|  |D|/|G|  D<0.1G   match count (L6/L9/L10/L11)  reach
valres_base    0.31%     0.367   0.698   0.459    30.0%    29.5/51.1/118.8/26.6        ~1010
E1 (val+β→6)   0.28%     0.406   0.683   0.520    24.3%    32.3/62.2/193.8/30.7        ~1010
valsyncB_base  0.68%     0.379   0.716   0.454    38.4%    —                         —
E2             0.69%     0.415   0.693   0.512    31.4%    —                         —
```

**关键结论（否定"锐化让 DISM 更少启用"的假设）**：

- 零输出比例本身只有 0.3–0.7%，微调锐化（E1）反而**降低**它（0.31%→0.28%）；
- `|DISM|/|GDN|` 上升（0.459→0.520），DISM 可忽略比例下降（30%→24%）；
- **匹配数上升**（layer10：118.8→193.8，+63%），reach 基本不变（~1000，长程）。
- 即锐化让 DISM **匹配得更多、更激进**，不是更少。

因此 PG19 域外变差**不是**因为 DISM 被关掉（DISM 仍在做长程匹配），更可能是锐化让
码字匹配**过度拟合训练分布的匹配统计**（匹配数 +63%）：域内更好、域外长程更差。

### 26.5 结论

1. 用户担心的"锐化把 DISM 关掉 / 长程失效"**没有发生**：DISM 依然在长程匹配，
   且匹配更多；PG19 也只是小幅变差（~0.003 bpb）。
2. 最优组合：**预训练 β=1 + selector β1=0.95，锐化全部放到 warm-start 微调**。
3. 域内 (val_bpb) 与域外 (PG19) 存在小 trade-off；若要"提高匹配选择性而不增加
   匹配数"，需要固定 logit 尺度（q/k + vocab normalization + 固定 τ）那条线。

---

## 27. shareqk/TransVQ 实验的"锐化不足"分析

背景：`shareqk` 系列为简化插值 kernel 共用 QK 词表，但词表易塌，于是对词表做了一系列
重参数化（TransVQ map：rank / diagonal / mean_shift / frozen_random…）。这些实验的
词表都经过 `out_norm=RMSNorm(dim)`（码字 RMS 归一）。

### 27.1 sharpness 测量（β=1，归一化 entropy，1=均匀）

```
layer:      6      7      8      9      10     11    | Q均   K均
valres    Hq .634  .651  .581  .556  .355  .334   | .519  .317
          Hk .354  .331  .356  .320  .219  .323   |
sep-rank1 Hq .590  .620  .557  .363  .432  .297   | .476  .356
          Hk .463  .448  .370  .255  .326  .271   |
lite      Hq .538  .583  .512  .335  .445  .300   | .452  .371
          Hk .490  .459  .365  .282  .350  .281   |
```

logit std（β=1）：valres Q2.0–4.3/K3.0–4.3；sep-rank1 Q3.1–8.8/K3.9–8.4；lite Q3.3–9.8/K4.4–8.9。

- **不是全局锐化不足**：norm 带可学 weight，transvq 的 logit std 反而**高于** valres；
  Q 侧整体更尖（Q 均 0.45–0.48 vs 0.52）。
- **真正偏软的是 K 侧前两个 DISM 层（6–7）**：有效词表 `V^H`
  - layer6 K：valres **9.1** → sep-rank1 **17.9** → lite **21.3**（≈2× 更平）；
  - layer7 K：valres 8.0 → 16.4 / 17.3；
  - layer9–11 的 K 反而比 valres 尖。

### 27.2 warm-start β 锐化微调（batch8，β 1→6，6000 步）

```
基座            base hard   base gap   → 微调后 hard   gap      Δhard    gap 降幅
valres (E1)     1.180755    0.005952      1.178955    0.002890  -0.0018  -51%
sep-rank1 (E3)  1.181896    0.007309      1.180378    0.004601  -0.0015  -37%
lite (E4)       1.198845    0.009312      1.189787    0.005581  -0.0091  -40%
```

**结论：**

1. **shareqk/TransVQ 实验存在真实的"锐化不足"**：β 微调对三者都有效，且相对 gap
   降幅 37–51%，与 valres 同量级。这与此前测到的 K 早期层偏软一致。
2. 因此这些实验过去"词表易塌 / 重参数化"的结论**被锐化不足这一混杂因素污染**：
   它们的 gap 里有一大块本来用 β 锐化就能拿走（37–40%）。
3. **但锐化不是全部**：微调后 transvq 仍差于 valres（E3 1.180378 vs E1 1.178955），
   差额约 0.0014 来自 shareqk + rank1 map 的几何限制，锐化补不回来。
4. 公平评估建议：在**锐化后的 gap/hard** 上比较不同 map，而不是 β=1 的原始值。

### 27.3 PG19 长文本（downstream-eval，hard eval，外推到 16k）

`bits/byte`：

```
model             ctx2048   ctx4096   ctx8192   ctx16384
fwedu_valres       1.4374    1.4204    1.4104    1.4047
E1_valres          1.4406    1.4231    1.4128    1.4069   (+β锐化)
fwedu_seprank1     1.4669    1.4516    1.4414    1.4344
E3_seprank1        1.4712    1.4560    1.4458    1.4390   (+β锐化)
fwedu_lite         1.4895    1.4749    1.4647    1.4569   (step16000)
E4_lite            1.4846    1.4700    1.4600    1.4527   (+β锐化)
```

两个要点：

1. **shareqk/TransVQ 架构有显著的 long-text 泛化赤字**：sep-rank1 在 PG19 上比
   valres 差 ~0.03 bpb（1.4669 vs 1.4374 @2048），lite 差 ~0.05；而它们的
   in-domain val_bpb 只差 0.001（sep-rank1）/ 0.019（lite）。即**域内几乎打平、
   域外长文本大幅落后**——这个赤字比"锐化"大一个数量级。
2. **锐化是"域内换域外"的小 trade-off**：完全训练过的 sep-rank1，β 微调后
   val_bpb 降了（1.1819→1.1804），但 PG19 反而变差（1.4669→1.4712 @2048）；
   与 valres 的 E1 同向。lite 的 E4 在 PG19 上更好，但它的基座只到 step16000
   （欠训），改善里混了继续训练的成分，不能单独归因于锐化。

**结论**：transvq 的锐化不足（§27.2）确实存在且可用 β 微调缓解 val/gap，但
**它补不上 shareqk/TransVQ 的长文本泛化赤字**；且锐化本身在长文本上有小幅负收益。
若要推进 shareqk/transvq 这条线，瓶颈在架构（map/共享底表的 long-range 表达），
不是温度调度。

### 27.4 16k 上下文下的 reach / 匹配数：是过度匹配，不是长程退化

对一个 PG19 16k 窗口（doc0），逐层重算后缀匹配递推并按位置分桶：

```
reach（最大匹配跨距）—— 两者逐格几乎相同，都顶到位置上限：
valres    L6: 1003/3054/6123/10219/14313
sep-rank1 L6:  999/3046/6118/10215/14311

match count —— sep-rank1 高 2–6×，连 [0,2k) 就高：
L6 : valres   28/ 85/ 166/ 266/ 387 | sep-rank1  45/ 138/ 290/ 478/ 667
L8 : valres   33/ 99/ 196/ 323/ 455 | sep-rank1  79/ 255/ 495/ 845/1260
L9 : valres   40/107/ 210/ 336/ 469 | sep-rank1 211/ 649/1302/2110/2916
L11: valres   26/ 77/ 160/ 254/ 388 | sep-rank1  40/ 134/ 269/ 452/ 615
```

`|D|/|G|` 逐层不同（sep-rank1 L9 达 1.0–1.45，valres 0.32–0.39）。

**结论：shareqk/TransVQ 的 long-text 赤字不是长程匹配退化，而是过度匹配。**

1. **reach 没有退化**：两者的最大匹配跨距在每个位置桶一致，都顶到位置极限。
   "DISM 长程匹配坏掉"被否定。
2. **匹配数高 2–6×，且 [0,2k) 就高**：即它在所有距离上不加选择地匹配，不是长程特有。
3. 与 §27.1 一致：sep-rank1 的 K 早期层有效词表大 2×（更平）→ 更多 token 落到相同/相近
   码字 → 匹配数暴涨 → 伪匹配。
4. 这些伪匹配在训练域内可拟合、到域外（PG19）变噪声 → 均匀的域外赤字。

因此根因是 **shareqk + rank1 map 造成码字利用率下降/塌缩**，与温度调度无关，锐化治不了。
修复方向：提高码字利用率、抑制伪匹配（更强 map/更高 rank/cross-codeword mixing、码字使用
正则、或不共享 qk）；评估应加入 **codebook 利用率 / 匹配数**指标。

---

## 28. REINFORCE 方案汇总与问题（为后训练准备）

### 28.0 为什么需要 REINFORCE（动机）

DISM 的 selector = 每个 (layer, row, head) 的 Q/K 符号 `argmax_c(q·W_c)`。
在 hard row 上 selector 分支的梯度**恰为 0**（§20 的 `dsoft` mask；§26 的 gating
统计）。因此任何**离散路径上的下游目标**（hard LM loss，或后训练的 RLHF/GRPO
reward）对 matching 参数（`q_vocab`/`k_vocab`/DISM `q/k_conv.conv_weight`/共享
linear）的梯度都是 0。要训练 matching，只能靠
soft relaxation（β pathwise）、STE 或 score-function —— 后两者都试过，见下。

### 28.1 方案 A：从零 categorical + deterministic soft baseline（远程 `dism-reinforce`）

- **Estimator**：每步对所有 Q/K 符号独立 categorical 采样（温度 1），
  `L = direct_hard + Σ_i a_i · log p_i(z_i)`，
  `a_i = causal_return(hard_NLL − soft_NLL)/count`，soft baseline 为 no-grad 的
  deterministic full-soft per-token NLL。
- **配置**：40M、从 scratch、step0 起全 categorical、**无 soft 梯度、无 STE、
  无 clip、无 reward 归一化**、无 hard-prob 退火。
- **结果**：跑满 6000 步，final argmax NLL **3.915**，而 soft→argmax 退火基线
  **3.098**（差 26%）。
- **失败机制**（分析报告）：采样熵 `6.09(step0)→2.84(300)→0.43(700)→0.13(1000)
  →0.046(3000)` nats；loss 在 300–700 反而升到 5.15→6.78；grad norm 中位数 11.5、
  step679 峰值 82.6。
  → **策略在 ~700 步内塌成近确定性，`∇log p→0`，selector 失去信号**；没有 soft
  pathwise 项兜底，码字只能靠 direct hard 梯度硬撑。

### 28.2 方案 B：K-side soft linear control variate（远程 `dism-residual`）

- **设计**（RELAX/REBAR 式）：对每个 K 节点先跑 deterministic full-soft，
  取 `u_i = ∂L_soft/∂h̄_i`（冻结），构造局部线性 proxy `v_i(s) = u_i·E_Q(s)`，
  加零均值修正 `J_p v − (v(z)−μ_i)∇log p_i(z)`；Q 侧保留 A 的 scalar baseline。
- **结果**：**只跑了 7 步就被 cancel**；`proxy_upstream_rms ≈ 3e-10`、
  `cv_surrogate ≈ 1e-6` → **control variate 实际是死的**，退化成方案 A。
- **结论**：**"control variate 降方差"这条线从未被真正验证过**；需要先修
  `u_i` 的尺度/上游 detach 问题。

### 28.3 方案 C：warm-start 混合（本 repo，已实现并实测）

- **Estimator**：`L = L_soft(β) + λ·mean_symbol(logπ(z)·Â) + η·relu(H_floor − H(π))`，
  `Â` 按文档标准化；soft pathwise 为主体，score-function 只做小修正。
- **配置**：warm start 自 valres@19074（β=1, hard_prob=0.95），lr3e-5、
  hard_prob=0、λ0→0.3、η0.1、β1→2/4，3000–6000 步。
- **结果**：**稳定但有害**。熵 2.65→3.13→1.5（未塌缩）、gnorm ~0.55–0.67；
  但 hard_bpb 一致差于同配置 λ=0 对照：
  - 3000 步：λ=0 **1.1795** vs λ=0.3 **1.1849 / 1.1829**；
  - 1500 步恢复曲线 ≠ 学习，而是**在抹掉 REINFORCE 造成的扰动**
    （从恢复点再 restart、同 λ 又掉回 1.185）。
- 即：warm-start 解决了 A 的**塌缩/爆炸**，但 score-function 的噪声只是累积、
  没有带来 hard 侧收益。

### 28.4 问题分类汇总

| # | 问题 | 出现于 | 现状 |
|---|---|---|---|
| 1 | 冷启动 + 熵塌缩（`∇log p→0`） | A | warm-start + 熵下限可解决（C 不塌） |
| 2 | control variate 死（`u≈3e-10`） | B | 未验证/未修复 |
| 3 | 高方差：序列级奖励 × 多决策数；gating 使 SNR ↓√(1−h) | A/B/C | 只能靠更长动量/更大 batch/更低方差估计量 |
| 4 | 方向偏置：gated 梯度方向是 soft surrogate，非 hard；argmax ≠ categorical 采样 | 全部 | β 温度是正确杠杆；REINFORCE 改不了 |
| 5 | 与 soft pathwise 竞争：hard row 上 selector 无梯度，score term 只在小部分样本上起作用 | C | λ 小则无效、λ 大则有害 |
| 6 | warm-start 后 hard 已近最优，无 headroom | C | score-function 噪声净累积 → 有害 |
| 7 | 优化器噪声匹配：AdamW β1/β2、per-group β1 | Astra 分析 | 每层 4 个 selector 张量，建议统一 β1=0.95 |
| 8 | dropout 式梯度缩放（`1/(1−h)`） | 讨论 | 对 Adam scale-invariant，改不了 SNR，无效 |

### 28.5 对后训练的含义

1. **后训练的核心诉求**是让离散 matching 参数被 reward 训练；但 hard 路径对
   matching 的梯度为 0，所以**必须有可微或低方差的 estimator**。
2. 目前证据指向：**不要用 REINFORCE 直接训 matching**（高方差、方向偏、无 headroom）。
   - 最稳的是 **soft pathwise（β）作为主路径**，让 reward 通过软门回传；
   - REINFORCE 只能作为**极小的修正**，且需要 control variate 真正工作。
3. 若后训练必须走 RL（如 GRPO/PPO）：
   - 应在 **token 级策略**上做（标准 RLHF），matching 通过 β-soft 路径或 STE 直通；
   - 或把 REINFORCE 限制在**低维动作**（例如每行的 hard/soft 路由），而不是每个符号。
4. **未验证但优先**：修复方案 B 的 control variate（`u_i` 尺度）、并用
   REBAR/RELAX 的真实残差估计器重跑一次，确认它是否能把方差降下来——这是
   "REINFORCE 能不能用"的关键实验。

---

## 29. 全 hard 前向下的梯度诊断（Step 1）

**脚本**：`dism-fwedu-valres-72m-2500m/diagnostics/grad_diag.py`
**问题**：在推理时完全 hard 的前向上，能否用 score-function 得到可用的（无偏）梯度？
**设置**：valres@19074，packed val batch（T=2048），只在 **selector 参数**（24 张量：
12×`q/k_conv.conv_weight` + 12×`q_vocab/k_vocab`，2.38M 元素）上比较：

- `g_S`：soft pathwise（`hard_mode=False`, β），即"影子"梯度；
- `g_R = Σ_k adv_k ∇logp(z_k)`，`adv = hard_tokens − soft_tokens`（标准化）；
- `rand`：高斯随机方向对照。

### 29.1 主结果：`g_R` 是纯噪声

| K | `\|g_R\|` | `\|g_k\|`（单样本） | 比值 | `1/√K` | `cos(half)` | `cos(g_k,g_S)` |
|---|---|---|---|---|---|---|
| 8 | 3019.8 | 8597 | 0.3513 | 0.3536 | −0.050 | +0.007 |
| 32 | 1522.6 | 8528 | 0.1785 | 0.1768 | +0.006 | +0.018 |
| 64 | 1070.8 | 8509 | 0.1258 | 0.1250 | −0.025 | +0.016 |

**`|g_R| ≈ |g_k|/√K` 在 K=8/32/64 全部成立** → 均值方向的模完全等价于"K 个零均值向量的
平均"，即 **`E[g_k] ≈ 0`**。由 K=8 与 K=64 解出：信号 `s ≈ 87`，噪声 `σ ≈ 8538`，
**逐样本 SNR `s/σ ≈ 1%`**；两次独立一半估计的 cos ≈ 0（方向不可复现）。
旁证：`cos(g_k,g_S) ≈ 0.007–0.02`，单样本 score 梯度与 soft 方向几乎正交。

### 29.2 不是"处在最优"造成的

把 selector 从最优处推开（加 `0.01`、`0.05`×‖θ‖ 的噪声，hard loss
2.7060→2.7076→2.7098）：`cos(half)=+0.034 / −0.014`，仍是零信号。
β 扫描：β=0.5 / 2 / 4 → `cos(half)=+0.074 / −0.028 / +0.029`，也全是零。
无论是否在最优、无论温度，全 hard 的 score-function 梯度都 ≈ 噪声。

### 29.3 沿方向的 hard loss 变化

在 batch=8（避免小 batch 过拟合）、held-out 用另一批 val 数据：

```
lr     1e-05     3e-05     1e-04     3e-04     1e-03     3e-03
base   2.73120   2.73120   2.73120   2.73120   2.73120   2.73120
g_S   batch   (2.73163)(2.73283) 2.73110  2.72997   2.74333   2.77750
g_S   heldout (2.87503)(2.87487) 2.87538  2.87835   2.89167   2.91752
g_R   batch   (2.73229)(2.73248) 2.73243  2.73126   2.73449   2.75307
g_R   heldout (2.87502)(2.87486) 2.87478  2.87504   2.87878   2.89233
randx8 best-delta  batch mean=−0.00005 std=0.00032 | heldout mean=−0.00072 std=0.00046
```

- `g_R` 的 batch 最优 `+0.00005`（**完全无下降**），held-out `−0.00048`，落在随机方向
  分布内 → 与随机不可区分。
- `g_S` batch `−0.00123`、held-out `−0.00039`；在小步长下是对的方向，但幅度也只比
  随机略好。（更小 batch=4 时 `g_R` 曾给 −0.005，但换 batch=8 不复现 → 是小 batch
  过拟合。）
- 大 lr（≥3e-3）时所有方向都变差，是有限步过拟合，不是有效信号。

### 29.4 结论与机制

1. **在全 hard 前向上，score-function 的无偏梯度在训练好的 base 上信号 ≈ 0（SNR≈1%），
   不可用。** 与软/硬、是否在最优、温度 β 都无关。
2. **机制**：hard loss 对采样符号配置**几乎不敏感**——同一 batch 采样不同 `z` 的
   hard NLL 只有 `std ≈ 0.003`（相对 2.71 约 0.1%）；而 `∇logp` 的逐样本噪声极大，
   二者的协方差可忽略，所以"reward × score"里几乎只剩噪声。
   这是 DISM 设计的内在性质（匹配输出是近连续/选择不变的），不是估计器 bug。
3. **可用的方向只有 soft pathwise 代理** `g_S`（有偏、但可复现、能小幅降 hard loss）。
   这与 §26/§28 的结论一致：**REINFORCE 不能作为全 hard 后训练的梯度来源。**
4. **对后训练的直接含义**：不要把离散符号当作 RL 动作空间去做 policy gradient；
   离散动作的 reward 方差被符号扰动淹没。要让 reward 训练 matching，走
   **soft/coupled pathwise 通路**（β 软门或 STE），而不是 score-function。
   若必须用 RL，动作应选在**低维、对 loss 敏感的连续/结构变量**上（例如每行
   hard/soft 路由、温度、或直接对 token 策略），而不是每个符号。

---

## 30. 为什么 hard loss 对符号选择几乎不敏感

脚本：`dism-fwedu-valres-72m-2500m/diagnostics/sym_sens.py`、`sym_prop.py`
（直接覆写 `idx_q`/`idx_k` 后重跑全 hard 前向，测量 Δloss / ΔDISM / Δhidden）

### 30.1 测量

**单符号翻转**（40 次随机 layer/head/token/side，改成随机不同 label；valres@19074，
batch=4、T=8192、每层 49152 个 label、共 6 层 = 294912）：

```
mean|Δloss| = 0.00004   （相对 2.706 是 0.0015%）
frac|Δ|>1e-4 = 0.15
```

**翻转比例扫描**（ΔDISM=各 DISM 核心输出的相对变化均值；relΔh[11]=最后一层 hidden
的相对变化；full scramble = 全部 294912 个 label）：

```
   frac    nflip     Δloss    relΔDISM    relΔh[11]
  2e-5        6   +0.00003     0.0041      0.0032
  1e-4       29   +0.00014     0.0199      0.0078
  1e-3      295   +0.00030     0.0444      0.0155
  1e-2     2949   +0.00043     0.0921      0.0203
   0.1    29491   +0.00086     0.2679      0.0456
   1.0   294912   +0.01913     0.6303      0.1158
```

**随机初始化对照**（同一 config，重初始化权重，base loss=10.47）：

```
   frac    nflip     Δloss    relΔDISM    relΔh[11]
  1e-2     2949   -0.00002     0.2098      0.0037
   1.0   294912   -0.00011     1.2456      0.0221
```

### 30.2 结论

1. **单符号是"稠密冗余平均"里被稀释的一项（不是稀疏匹配）。**
   DISM 输出是对大量匹配项的加权平均；一个 label（每层 49152 个之一）只贡献约
   0.07% 的输出变化。所以单翻转 Δloss≈1e-5，几乎无作用。
   （§27.4 的 match count 30–400/query 也说明匹配是**稠密**的，不是稀疏。）

2. **主要原因是下游压缩。** 即使把**全部** label 打乱，DISM 核心输出变
   `relΔDISM=0.63`，但残差流只变 `relΔh=0.12`、loss 只变 `0.94%`。
   衰减：DISM→hidden ≈5×，hidden→loss ≈17×。
   即 gate(`silu`)/RMSNorm/`o_proj` 把 DISM 分支压缩到残差流的几个百分点。

3. **这不是训练学出来的鲁棒性。** 随机初始化模型衰减**更强**
   （DISM→hidden 57×，`Δloss/L`~1e-5），而训练后的模型是 5×。
   也就是说训练实际上**增强**了模型对 DISM 匹配的依赖，只是绝对值仍然很小。
   故"随机 soft/hard 切换让模型对符号不敏感"（H1 的字面版本）**不成立**。

4. **对 §29 的意义**：REINFORCE 估计 `∇E_z[L_hard(z)]`，而 `L_hard` 对 `z` 的敏感度
   本身只有 ~1%（全乱变）。SNR 必然低——这不是估计器缺陷，是**目标本身**的性质。

### 30.3 对后训练的含义

- **不要指望用"离散符号"作为动作信号去训练 matching**：单个符号的信息量太低
  （0.07% 输出变化），任何 per-symbol 的 policy gradient 都会被噪声淹没。
- 若要让 reward 影响 matching，必须作用于**能大幅改变 DISM 输出**的变量：
  整段 hard/soft 路由、τ、或将 reward 经 **pathwise 通路**直接作用于连续参数。
- 或者**改架构**，让单个符号有更大的杠杆：减少匹配冗余（更硬/更稀疏的聚合、
  top-k 匹配、可学习的匹配温度），使符号选择重新成为有效自由度。

---

## 31. 16k 尺度：DISM 消融 + 符号翻转

脚本：`dism-fwedu-valres-72m-2500m/diagnostics/pg19_ablate.py`、`sym_flip16k.py`
（PG19 走原 harness：`adapter.py` + `pg19.yaml`，hard eval，disjoint 16k 窗口）

### 31.1 DISM 消融（valres@19074）

| mode | ctx2048 | ctx8192 | ctx16384 |
|---|---|---|---|
| none（基线） | 1.437438 | 1.410405 | 1.404733 |
| `flip`（全部符号随机） | 1.484134 | 1.469408 | 1.463485 |
| `dism_core`（DISM 输出置零，保留 GDN） | 1.473630 | 1.460349 | 1.459243 |
| `dism_layer`（整个 DISM 层 attention 置零） | — | — | 3.160068 |

Δ vs 基线：

| ctx | flip | dism_core |
|---|---|---|
| 2048 | **+0.0467** | +0.0362 |
| 8192 | **+0.0590** | +0.0499 |
| 16384 | **+0.0588** | +0.0545 |

- **DISM matching 的贡献随上下文增长**：+0.036（2k）→ +0.055（16k）bpb。
- **符号信息几乎就是全部贡献**：全符号随机（+0.059）≈ 或略大于 DISM 输出置零（+0.055）。
  也就是说"错的匹配"比"没有匹配"还差一点 → 匹配确实在做实质工作。
- `dism_layer`（去掉 6 层的 attention）= 3.16 bpb，说明那 6 层整体仍是必需的，只是
  DISM matching 那一部分单独占 ~0.055 bpb。

### 31.2 16k 单序列上的符号翻转（valres, PG19 doc0, T=16384, base loss=3.235）

```
   frac    nflip      dLoss   relDISM   dLoss/base
  2e-5       12    -0.00000     0.0295     -0.000%
  1e-4       59    +0.00041     0.0396      0.013%
  1e-3      590    +0.00020     0.0553      0.006%
  1e-2     5898    +0.00179     0.0755      0.055%
   0.1    58982    +0.01295     0.1609      0.400%
   1.0   589824    +0.05914     0.3929      1.828%
```

按位置分桶全翻转（每个桶 = 25% 的 label）：

```
        bucket    nflip      dLoss   relDISM   dLoss/base
      [0k,4k)   147456    +0.02223    0.2581     0.687%
      [4k,8k)   147456    +0.01696    0.2129     0.524%
     [8k,12k)   147456    +0.01409    0.2083     0.435%
    [12k,16k)   147456    +0.01923    0.2014     0.594%
```

- 敏感度**随总上下文增大**：2k 全翻转是 0.94%，16k 是 1.83%（相对 loss）。
- 但**在 16k 内部不随距离增长**：四个桶的 Δloss 基本持平（0.014–0.022）。
  → DISM 的收益是"整体上下文越长越重要"，不是"越远的匹配越关键"。
- 注意 `relDISM` 在 16k 反而更小（0.39 vs 2k 的 0.63），而 Δloss/base 更大：
  16k 时 DISM 输出的**每单位变化对 loss 的放大更高**（~3×）。

### 31.3 与 §29/§30 的调和

- §30 说的是"loss 对符号的**相对**敏感度只有 ~1–2%（nats）"；放到 PG19 上，
  这就是 **+0.059 bpb（相对 4.2%）**——并不小。§30 的措辞（"几乎装饰性"）应弱化为
  "符号信息是 DISM 贡献的主体，但单符号的边际信息量很低（~0.07%/符号）"。
- §29 的"score-function 信号 ≈ 0"是**局部**结论：base 已处于
  `E_z[L_hard(z)]` 的（近）驻点 → 一阶梯度为 0，但把匹配整体扰动到随机会掉 0.06 bpb。
  两者不矛盾：驻点 + 大的有限扰动代价。
- 对后训练：**per-symbol 的动作仍不可用**（单符号 0.07% 输出变化、局部梯度为 0），
  但"匹配结构整体"是高价值变量 → 后训练应作用于**段级/结构级**变量
  （整行 hard/soft、τ、或经 pathwise 的连续参数），而不是逐个符号。
