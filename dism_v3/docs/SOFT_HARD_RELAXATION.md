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

---

## 32. NIAH：DISM 是长程 recall 的必需机制

**动机**：自然语言里绝大多数 token 由局部依赖决定，长程信息只在极少数 token 上有用，
bpb 会把长程失败平均掉。所以用合成的单针检索（NIAH）直接测长程能力。

脚本：`dism-fwedu-valres-72m-2500m/diagnostics/niah.py`
（PG19 文本作 haystack，插入 `The special magic number is <N>.`，query
`The special magic number is`，对 number token 算 NLL / greedy。
`absent` 对照把针替换成等长 filler。指标 `gain = NLL_absent − NLL_present`，越大 recall 越好。）

### 32.1 结果（valres@19074，8 trials/cell）

`gain`：
| ctx | depth | none | dism_core | flip |
|---|---|---|---|---|
| 2048 | 0.05 | +0.91 | +0.48 | +0.50 |
| 2048 | 0.50 | +1.41 | +0.73 | +0.82 |
| 2048 | 0.95 | +1.87 | +1.48 | +1.47 |
| 4096 | 0.05 | +0.55 | **−0.07** | **−0.06** |
| 4096 | 0.50 | +0.89 | +0.33 | +0.24 |
| 4096 | 0.95 | +1.66 | +1.34 | +1.31 |
| 8192 | 0.05 | +0.68 | **+0.01** | +0.11 |
| 8192 | 0.50 | +0.79 | **+0.07** | +0.16 |
| 8192 | 0.95 | +1.58 | +1.35 | +1.33 |
| 16384 | 0.05 | +0.44 | **+0.10** | +0.12 |
| 16384 | 0.50 | +0.59 | **+0.06** | +0.09 |
| 16384 | 0.95 | +1.08 | +0.72 | +0.77 |

mean：2048 `1.397→0.897`；4096 `1.035→0.535`；8192 `1.015→0.479`；16384 `0.704→0.291`。

exact-match（trial_all）：2048/d0.95 `0.12→0.00`；4096/d0.95 `0.25→0.00`。

### 32.2 结论

1. **72M 模型的 NIAH 能力确实存在**（gain 最高 +1.9 nats），但**强 recency 偏置**：
   depth 0.95（针靠近 query）远好于 depth 0.05（针在开头）。
2. **DISM 承担了几乎全部"远端"recall**：
   - 4k/d0.05：`+0.55 → −0.07`（完全崩塌）；8k/d0.05：`+0.68 → +0.01`；
     16k/d0.05：`+0.44 → +0.10`。
   - 近端（d0.95）DISM 置零只掉 ~0.2–0.35，因为局部/GDN 通路足以处理。
   - **符号随机 ≈ DISM 置零** → 符号匹配就是这整个机制本身（与 §31 一致）。
3. **用户判断被证实：bpb 严重稀释长程信号。**
   同一模型同一消融：bpb 只动 **+0.036 ~ +0.055**，而 NIAH 远端 gain 从 +0.55
   直接掉到 ≈0。长程能力的变化被海量短程可决定的 token 平均掉了。
4. 与 §29/§30/§31 调和：base 处在 `E_z[L_hard]` 驻点 → per-symbol 一阶梯度≈0；
   但**匹配结构整体**承载了全部 NIAH 长程能力。所以后训练应作用于段级/结构级变量，
   且评估必须包含 NIAH 类任务，而不只是 bpb。

---

## 33. NIAH：10 个模型的对比

脚本：`dism-fwedu-valres-72m-2500m/diagnostics/niah.py`
结果：`dism-fwedu-valres-72m-2500m/diagnostics/niah_results/*.json`
设置：`ctx ∈ {2048, 8192, 16384}`、`depth ∈ {0.05, 0.5, 0.95}`、16 trials，
**所有模型共用同一组 trial（固定 seed，配对比较）**。指标 = `NLL_absent − NLL_present`。

### 33.1 每格 gain

| model | c2048/d.05 | c2048/d.5 | c2048/d.95 | c8192/d.05 | c8192/d.5 | c8192/d.95 | c16384/d.05 | c16384/d.5 | c16384/d.95 |
|---|---|---|---|---|---|---|---|---|---|
| slimpajama_baseline | +1.39 | +1.57 | +2.14 | +0.84 | +1.08 | +1.82 | +0.77 | +0.75 | +1.06 |
| fwedu_baseline | +0.69 | +1.06 | +2.03 | +0.07 | +0.11 | +1.80 | **−0.01** | +0.11 | +1.38 |
| fwedu_valres | +0.82 | +1.26 | +1.74 | +0.69 | +0.56 | +1.43 | +0.28 | +0.67 | +1.10 |
| E1_valres | +0.88 | +1.30 | +1.92 | +0.77 | +0.48 | +1.45 | +0.45 | +0.66 | +1.23 |
| fwedu_valsyncB | +0.55 | +1.11 | +1.73 | +0.36 | +0.36 | +1.36 | +0.36 | +0.20 | +0.93 |
| E2_valsyncB | +0.62 | +1.02 | +1.70 | +0.35 | +0.36 | +1.42 | +0.30 | +0.15 | +0.92 |
| fwedu_seprank1 | +0.28 | +0.73 | +1.52 | +0.12 | +0.12 | +1.08 | +0.18 | +0.12 | +0.66 |
| E3_seprank1 | +0.36 | +0.67 | +1.61 | +0.22 | +0.15 | +0.98 | +0.25 | +0.20 | +0.60 |
| fwedu_lite | +0.63 | +1.13 | +1.68 | +0.51 | +0.46 | +1.31 | +0.47 | +0.20 | +0.95 |
| E4_lite | +0.64 | +1.01 | +1.54 | +0.62 | +0.41 | +1.21 | +0.56 | +0.29 | +0.85 |

depths 平均：
| model | ctx2048 | ctx8192 | ctx16384 |
|---|---|---|---|
| slimpajama_baseline | +1.702 | +1.249 | **+0.857** |
| E1_valres | +1.364 | +0.903 | +0.780 |
| fwedu_valres | +1.272 | +0.893 | +0.686 |
| E4_lite | +1.062 | +0.748 | +0.569 |
| fwedu_lite | +1.145 | +0.759 | +0.539 |
| fwedu_valsyncB | +1.130 | +0.695 | +0.495 |
| fwedu_baseline | +1.263 | +0.662 | +0.491 |
| E2_valsyncB | +1.113 | +0.709 | +0.457 |
| E3_seprank1 | +0.878 | +0.453 | +0.349 |
| fwedu_seprank1 | +0.843 | +0.443 | **+0.320** |

### 33.2 结论

1. **bpb 严重低估长程损伤（用户判断再次被证实）。**
   `fwedu_seprank1` vs `fwedu_valres`：
   - bpb@16k：1.4344 vs 1.4047 → 只差 **+0.030（2%）**
   - NIAH@16k：0.320 vs 0.686 → **差 53%**；@8192：0.443 vs 0.893 → **差 50%**
   → shareqk/TransVQ 砍掉了一半的长程 recall，而 bpb 几乎看不出来。

2. **数据本身对长程 recall 影响巨大，bpb 也看不出来。**
   `slimpajama_baseline` vs `fwedu_baseline`：bpb@16k 1.4368 vs 1.4410（几乎相同），
   但 NIAH@16k 0.857 vs 0.491（**差 43%**）。fwedu 的 far-needle（d0.05）在
   8192/16384 上 gain ≈ 0（**完全没有远端 recall**）。

3. **`value_residual` 对长程 recall 是关键**：
   `fwedu_baseline`（无 value residual）d0.05@8192 = +0.07、@16384 = −0.01；
   加上 value residual 的 `fwedu_valres` 变成 +0.69 / +0.28。

4. **β 锐化微调对 recall 有（小幅）帮助，方向一致**：
   E1 1.272→1.364（2k）、0.893→0.903（8k）、0.686→**0.780**（16k）；
   E4 0.539→0.569（16k）；E3 0.320→0.349（16k）。
   （16 trials 下 E1/E3 的改善接近噪声，但方向与 bpb 一致。）

5. **recency 在所有模型上都存在**：d0.95 ≫ d0.05，且 gap 随 ctx 增大。

### 33.3 评估建议

**NIAH 应作为常备指标**（至少 ctx 8k/16k + 远端 depth）。bpb 会把长程能力的
大幅退化平均掉（本节所有大差异在 bpb 上都 ≤0.03）。任何伤到 matching 结构或
value 通路的改动（shareqk/TransVQ、量化、剪枝、后训练）都必须看 NIAH。

---

## 34. 为什么 SlimPajama 的长程能力比 FineWeb-Edu 强

§33 里 `slimpajama_baseline` 的 NIAH 明显好于 `fwedu_baseline`（16k：0.857 vs 0.491）。
先排除 confound：两个 run 的 `model_spec`、`sequence_len`(2048)、`total_batch_size`(131072)、
steps(19074)、种子(1729)、**tokenizer 指纹完全相同**，总训练量都是 2.5B tokens；
唯一差别是数据目录。**所以这是纯数据效应。**

### 34.1 语料统计（各采样 800 篇）

| 语料 | tokens/doc 中位 | p90 tokens | gzip bytes/char | 8-gram 在 >128 token 后重复率 |
|---|---|---|---|---|
| slimpajama | 528 | 2092 | 0.523 | **1.93%** |
| fwedu | 679 | 2382 | 0.497 | **0.68%** |

- **fwedu 文档更长**（中位 679 vs 528，p90 2382 vs 2092，max 92055 vs 62370）
  → "fwedu 文档短导致长程信号弱"这个猜想**被否定**。
- **slimpajama 的长程重复率是 fwedu 的 ~3×**（1.93% vs 0.68%）
  → 同一条 8-gram 在 >128 token 后再次出现的概率，slimpajama 高得多
  → 提供了大量**长程 copy / induction 的训练信号**。
- **fwedu 更可压缩**（0.497 vs 0.523 bytes/char）→ 局部更可预测、熵更低。

### 34.2 模型侧：上下文增益曲线（PG19，同一批目标 token，只变前文长度）

脚本 `diagnostics/ctx_gain.py`（固定 window，目标 = 最后 256 token）

| model | p256 | p512 | p1024 | p2048 | gain(256→2048) |
|---|---|---|---|---|---|
| slimpajama | 3.996 | 3.659 | 3.610 | 3.593 | **+0.403** |
| fwedu | 3.986 | 3.689 | 3.647 | 3.636 | +0.350 |
| fwedu_valres | 3.934 | 3.595 | 3.534 | 3.504 | **+0.429** |
| fwedu_seprank1 | 4.015 | 3.721 | 3.674 | 3.652 | +0.363 |

- slimpajama 的模型确实**更依赖长上下文**（+0.403 vs +0.350），方向与 NIAH 一致。
- 但注意：**sep-rank1 的 context gain 正常（+0.363），NIAH 却很差**。
  → 自然文本上的"上下文增益"主要来自近/中程效应，**测不出 TransVQ 破坏的稀疏长程通路**。
  → 这进一步说明 NIAH 是不可替代的指标。

### 34.3 结论

用户的直觉其实**是反的**：FineWeb-Edu 的"干净、教育性"恰恰意味着**局部可预测性高**
（gzip 更可压缩、域内 val_bpb 1.19 vs 1.32），于是**需要长程信息的 token 更少**，
模型没有动力/压力去长出长程电路。SlimPajama 这种混杂语料（代码、书籍、论坛，
高重复、高熵）反而**逼出**了长程能力。

含义：
- **长程能力是可被数据混合调节的量**；更"高质量/干净"的过滤可能**主动损害**长程。
- fwedu 在域内 bpb 上大幅领先（1.19 vs 1.32）却长程更弱 → 这是一个**明确的 trade-off**，
  评估必须两者都看。
- 若要补长程：混入高重复/长文档/高熵数据（代码、书籍）。

---

## 35. MAD-lab MQAR：DISM / GDN / 混合布局的隔离测试

**动机**：§32–§34 表明自然语言 bpb 会稀释长程信号。为了把"机制本身能不能做
associative recall"和"自然语言里学不学得到"分开，接入了
`~/cs/project/mad-lab`（MAD-lite 的合成任务套件），用它的 `in-context-recall`
（多 query MQAR）。

### 35.1 接线（新增在 mad-lab，不在本仓库）

- `mad/model/layers/dism.py`：`_DismPacked` 把 MAD 的 `[B,T,dim]` 打包成
  `[1,B·Lp,dim]`（每个 batch 行 = 一个文档，**无跨行匹配**），并要求每个文档
  长度 256 对齐。MAD 的生成器返回 `inputs[:-1]`，所以模型输入 `T = seq_len-1`
  是**奇数**（255/511/1023）；因此**每个文档尾部补齐到 256 的倍数**
  （255→256，只多 1 个 token）。补齐安全：short conv 与后缀匹配递推都是因果的，
  尾部 pad 不影响任何真实位置，输出再裁掉。
  验证：同输入两次输出 bit-identical；只改 row 0 时 row 1–3 的 diff = 0.000e+00。
- 变体：`DismHybrid`（DISM+GDN）、`DismOnly`（只 DISM）、`GdnOnly`
  （参数与 DismHybrid 相同，只在 `_combine_cuda` 里丢弃 DISM 输出）、
  `DismHybridUntied`（解开 Q/K 线性共享）、`DismHybridNormed`（相加前各分支
  独立 RMS 归一化）、`DismHybridDeltaLP`（tiny_lm 的 rank-1 delta 写入）。
- 训练用软门 + `hard_prob=0.5`（随机 hard 行），eval 用 hard argmax。
- **`direction=None`**（保持用户设定，勿改）：固定 direction 在这类合成任务上
  会造成严重的 train/infer gap。

### 35.2 结果（vocab=128，best-epoch test accuracy）

| 布局 | seq=256 | seq=1024 |
|---|---|---|
| mh-attention（n_heads=16, head_dim=8） | 0.0645 | 0.0548 |
| mh-attention-4h（n_heads=4, head_dim=32） | 0.0640 | 双峰，见下 |
| gdn-only ×2 | 0.9973 | 0.9970 |
| **dism-only ×2** | **0.9992** | **0.9991** |
| dism+GDN 相加（生产 `_combine_cuda`） | 0.9950 (0.987–0.999) | 0.9638 (0.964–0.994) |
| dism+GDN 解绑 Q/K 线性 | 0.9994 | 0.9919 |
| dism+GDN per-branch RMS 归一化 | 0.9856 | 0.9930 |
| **dism+GDN rank-1 delta 写入（del_k/del_v）** | **0.9921** | **0.9918** |
| **专用层：GDN + 2×DISM** | **0.9992** | **0.9991** |
| 专用层：GDN + 1×DISM | 0.9921 | — |

### 35.3 结论

1. **层内 DISM+GDN 混合在两个任务档上都劣于 dism-only**，而且方差大得多
   （hybrid 跨 run 0.964–0.999，dism-only 0.989–0.999 且多数 ≥0.999）。
2. **三种混合机制都救不了**：相加、per-branch RMS 归一化、rank-1 delta 写入
   （tiny_lm 的 `del_k/del_v`）全部落在 0.986–0.995。delta 版本探针确认机制
   **确实在用**（`||inject||/||GDN|| = 0.07–0.12`，是相加版 4% 的两倍），
   即不是机制退化，而是**这个任务上 GDN 与 DISM 没有互补性**。
3. **"专用层"布局是唯一有效的做法**：`GDN + 2×DISM` 直接回到 0.9992/0.9991，
   和纯 dism-only 相同，且 17 个 epoch 就收敛。这支持把生产架构从"层内混合"
   改成"前半 GDN、后半 DISM+SWA 与 GDN 交错"的专用层方案。
4. **至少保留 2 个 DISM 层**：`1 GDN + 1 DISM` = 0.9921 < `2 DISM` = 0.999。
5. **分支塌缩现象**：训练好的 hybrid 里 DISM 只占合并输出的 4–12%
   （`||DISM||/||GDN||` 在不同 run 之间从 0.043 到 0.403）——即
   **winner-take-all，且胜者是随机的**。对照 nanochat 预训练的 valres：
   `|DISM|/|GDN| = 0.18–1.59`（两个分支都在实质工作），所以这个塌缩是
   小规模/短训练下的优化现象，不能直接外推到生产规模。

### 35.4 方法学教训（这些坑都实际踩过）

- **`results.csv` 用的是最后一个 epoch**，而所有 run 都会偶发 divergence spike
  → 必须用 best-epoch。
- **`--early-stop` 必须关掉**（MAD 默认 `early_stop=False`）：attention 基线在
  (128,256) 上需要 **226 个 epoch** 才起飞，patience 30 会把它提前杀掉。
- **`mh-attention.yml` 是 `n_heads=16 / dim=128` → head_dim=8，过小**；改成
  4 头（head_dim 32）后 attention 在 (16,1024) 从 0.318 → 0.9993。
- **run-to-run 方差极大**：完全相同的配置能给出 0.964 vs 0.994（hybrid），
  attention 甚至能 0.059 vs 0.999（induction circuit 形成是随机相变）。
  单 run 结果只能当趋势，定方案需要 ≥3 seeds 取 median。
- 结构上 MHA 需要**两层组合**才能做 induction（层1 用 RoPE 造 key 后移一位，
  层2 做 qk 等值召回），而 GDN/DISM 有 short conv、**层内**即可完成移位，
  所以 `[mixer,swiglu]×2` 对 MHA 本身不公平——比较时需给 MHA 补深度或补 dwconv。

### 35.5 极限档：vocab=512 / seq=2048（256 keys, ctx 2048）

配置：`in-context-recall`，`vocab_size=512`（256 keys / 256 values）、`seq_len=2048`
（模型输入 T=2047）、`num_train_examples=12800`、`batch=128`、200 epochs、lr 5e-4。
**注意**：examples 必须保持 12800，**不要**按 seq_len 缩放——先用 `1600`
（= 12800·256/2048）跑时两个架构都锁死在 chance（0.004），那是**数据量不足**，
不是机制失败。这是本次最重要的方法学教训。

| arch | best acc | test loss (min) | 收敛行为 |
|---|---|---|---|
| **dism-only ×2** | **0.8753** | **0.588** | 到 200 epoch 仍在涨（0.838@180→0.875@199） |
| gdn-only ×2 | 0.5791 | 2.079 | epoch 60–80 就饱和（~0.51→0.58） |

chance = 1/256 = 0.0039；dism 是 chance 的 224×，gdn 是 148×。

轨迹（acc）：
```
dism-only  0.004 0.022 0.432 0.556 0.629 0.707 0.706 0.801 0.813 0.838 0.875
gdn-only   0.004 0.008 0.146 0.440 0.490 0.510 0.524 0.535 0.546 0.568 0.577
```

**结论**：任务难度越高，DISM 相对 GDN 的优势越大——容易档（§35.2）两者都能解、
DISM 略好；极限档 **DISM 是唯一有效的**（0.875 vs 0.579，loss 差 3.5×）。
这是"固定大小循环状态 vs 匹配式检索"的容量差异，也说明**专用 DISM 层是把长程
能力做强的正确方向**（前提是 gap 被单独解决，见 35.6）。

---

## 36. 因子化（product-quantised）码表

### 36.1 动机

现在一个 V=512 的 qk 词表相当于 512 个 double-arm prototype，全部塞进同一个
D=64 维空间，靠插值压成一个向量，用点积算匹配 log-prob 的 Jensen 下界。
要建立匹配就得协调几百个 embedding 在**同一个**空间里的几何形态。

因子化：把 V 拆成若干组 `V = Π S_g`，每组 S_g 个 prototype 各自占用
`dg = D/g` 个**不相交**维度。因为

```
q · E_v          = Σ_g q_g · E^g_{v_g}
logsumexp_v(...) = Σ_g logsumexp_{v_g}(...)
```

于是插值输出 = 各组结果**拼接**、匹配 LSE = 各组 LSE **相加**，而 `q_vec·k_vec - LSE`
里的点积本身就在拼接维度上求和——**DISM core 与 kernel 一行不改**。硬匹配用混合进制
双射 `v = Σ_g v_g · Π_{h>g} S_h`，`idx_q == idx_k` 仍等价于"每组都相等"，
所以**硬匹配是精确等价的**；只有软分布被限制成 product-of-marginals。

实现：`dism_v3/python/flash_dism/pq_vocab.py`（`factorized_interp` /
`expand_codebook` / `split_codebook` / `strides`），
自检 `python -m flash_dism.pq_vocab` 用仓库自带 fp32 参考验证 block 结构与
因子化路径等价：**out rel-err ~1e-06、lse abs-err ~4e-06（scale 35–48）、
idx 全精确**（覆盖 (2,2)/(16,16)/(32,16)/(64,64)）。

**效率**：插值开销 `Σ_g S_g·(D/g) = S·D`，**与组数和 V 无关**；码表参数量同理。
V=256 的 16×16 比 V=256 的单表省 **16×** 插值、省 30% 参数。

**两个数值约束**（实测）：
- 每组维度 `dg` 必须是 **32 或 64**（`invalid vocabulary/channel size`），所以
  `head_dim=64` 只支持 g=2，`head_dim=32` 无法因子化；
- bf16 kernel 里"一次 256-way logsumexp"与"两次 16-way 相加"相差 **~1.0（2.6%）**
  （纯累加顺序舍入）→ **必须从头用因子化路径训练**，不能续现有 checkpoint；
  且分数基线平移 `−g·log S`，`rtau`/softcap 需按组数重标定。

### 36.2 小配置验证（MAD in-context-recall, vocab=128/seq=256/ntr=12800）

| 变体 | 容量 | 插值开销 | 参数量 | best acc | min test loss |
|---|---|---|---|---|---|
| 未因子化 V=256 | 256 | 16384 | 810k | 0.9971 | 0.0096 |
| **16×16** | 256 | **1024** | 564k | **0.9991（28 epoch 就到）** | **0.0047** |
| 64×64 | 4096 | 4096 | 588k | 0.8969 | — |

16×16 **同容量下三项全赢**（精度、16× 算力、30% 参数），且收敛快得多。
4096 在这档反而更差——任务只有 128 个 token，4096 码字是严重过参数化。

### 36.3 深度判据：几何协调压力确实被推到前面的层

MAD `--layers <mixer> swiglu`（1 个 mixer，DISM 直接读 embedding，"压力无处可去"）
vs 2 个 mixer：

| depth | 未因子化 V=256 | 16×16 | Δ(test loss) |
|---|---|---|---|
| **1 mixer** | 0.9973 / loss **0.0110** | 0.9915 / loss **0.0369** | **+0.026（3.4× 差）** |
| **2 mixers** | 0.9971 / loss 0.0096 | **0.9991 / loss 0.0047** | **−0.005（2× 好）** |

**干净的交叉**：因子化在没有前驱层时明显吃亏、有前驱层时明显占优。
即因子化把 (a) 码表 packing 的难度降下来，却把 (b) 表征落点变成
**g 个固定不相交子空间里的合取**，而 (b) 由 DISM 之前的层承担。
（附带代价：软分布被限制成 product-of-marginals，无法表达相关不确定，
会把质量放到部分匹配上 → 可能放大 gap。）

### 36.4 极限档（vocab=512/seq=2048/ntr=12800）

| 变体 | 容量 | head_dim | best acc | min test loss |
|---|---|---|---|---|
| 旧 `dism-only` | 512 | 32 | 0.8753 | 2.079 |
| A `dism-only-h64-v512` 未因子化 | 512 | 64 | **0.9940** | **0.030** |
| B **32×16**（同容量） | 512 | 64 | 0.9854 | 0.135 |
| C 64×64 | 4096 | 64 | 0.9572+（ep73 仍在爬） | — |

**同容量下因子化在长上下文反而更差**（0.9854 vs 0.9940，loss 4.5×）——
与小配置相反。合理解释：单点匹配率从 `a` 变成 `Π_g a_g`（合取），
而 reach 来自连续匹配链的 `softplus` 累加，期望链长 ~`1/(1−Π a_g)`。
seq=256 只需短链（惩罚轻），seq=2048 需要长链（惩罚指数放大）。
**这正是"组数越多越容易梯度消失/gap"的另一种表现：长程 reach 收缩。**

另注意 **head_dim 在极限档是巨大变量**：32→64 让 A 从 0.8753 跳到 0.9940。

### 36.5 nanochat 四格对照（专用层 vs 层内混合 × direction）

| run | 布局 | direction | val_bpb | soft_bpb | gap_bpb |
|---|---|---|---|---|---|
| **A** valres | 6 纯GDN + 6 混合 | true | **1.182854** | 1.177019 | 0.005835 |
| **C** spec2d1g | `ggggggddgddg` 专用 | random | 1.201140 | 1.193274 | 0.007866 |
| **D** spec2d1g-dirtrue | `ggggggddgddg` 专用 | true | **1.204047** | 1.192999 | 0.011047 |

**纯布局效应（D − A）：+0.0212 bpb**，其中 **soft +0.0160、gap +0.0052（1.9×）**。
**纯 direction 效应（C − D）：−0.0029 bpb（random 更好）、gap −0.0032（减半）。**

→ 之前 C vs A 的 +0.018 里 direction 是**负贡献**，布局真实代价是 **+0.021**。
`direction=random` 是独立的正收益。

NIAH 的 present NLL（越低越好）：

| cell | A 混合 true | C 专用 random | D 专用 true |
|---|---|---|---|
| 2048/d0.05 | 2.158 | **1.516** | 2.012 |
| 8192/d0.05 | 2.480 | **2.113** | 2.509 |
| 16384/d0.05 | 2.811 | 2.611 | **2.548** |
| 8192/d0.95 | **1.725** | 1.832 | 2.095 |
| 16384/d0.95 | **2.051** | 2.179 | 2.440 |

- **近端（d0.95）A 全面最好**（A 有 12 个 GDN vs D 的 8 个，短程/近因容量更多）；
- **远端、2k/8k 上下文 C 最好**；16k 远端 D 略好但 16k 近端 D 最差；
- `direction=random` 的收益集中在 2k/8k 远端。

### 36.6 结论 / 待办

1. 因子化的**数学与实现是正确的**，效率收益是真实的（V 从 256 扩到 4096 不再增加
   插值开销），小配置同容量下也确实更好。
2. 但它把几何协调压力推给前驱层（36.3 的交叉），且在**长上下文**上因合取导致
   reach 收缩（36.4）。
3. **专用层重构不是免费收益**：val_bpb +0.021、近端 recall 变差。
   注意 confound：`ggggggddgddg` 只有 **4 个 DISM 层**（valres 有 6 个），
   所以这 +0.021 里混了"DISM 层变少"。要干净隔离"专用 vs 层内混合"，
   应测 **`ggggggdddddd`**（6 个纯 DISM，rear half 无 GDN）。
4. **`direction=random` 应当保留**：bpb −0.003、gap 减半、远端 recall 提升。
5. 因子化的下一步：在**有前驱层**的前提下测，并优先测
   **少而大的组**（64×64 而不是 4×16），因为合取惩罚随组数增长；
   同时量 **reach / 匹配数**直接验证 36.4 的机制。

---

## 36b. 因子化（product-quantised）qk 码表（续：实现与验证）

**想法**（用户提出）：把 V 个 prototype 拆成若干组因子，`V = Π S_g`。因为
`q·E_v = Σ_g q_g·E^g_{v_g}` 且 `logsumexp_v` 对组可分解，

```
插值输出 = 各组插值结果沿 head dim 拼接
LSE      = Σ_g LSE_g
硬标签    = 混合进制双射 v = Σ_g v_g · Π_{h>g} S_h
```

于是 **DISM core 与 kernel 零改动**（点积天然对拼接维求和），硬匹配仍精确等价
（`idx_q == idx_k` ⟺ 每组都相等），只有软分布被限制成 product-of-marginals。

**实现**：`dism_v3/python/flash_dism/pq_vocab.py`（`factorized_interp` /
`expand_codebook` / `split_codebook` / `strides`）。组大小可不同，码表存
`[H, g, max(S), dg]`。kernel 约束：**dg 只能是 32 或 64**，所以 head_dim=64 时
最多 2 组；head_dim=32 无法因子化。

**精确性验证**（`python -m flash_dism.pq_vocab`，与 repo 自带 fp32 参考对比）：
(2,2)/(16,16)/(32,16)/(64,64) 全部 out rel-err ~1e-06、lse abs-err ~4e-06
（scale 35–48）、idx 精确。bf16 Triton kernel 内部因 logsumexp 累加顺序会有 ~1e0
的 LSE 偏差 → **必须用因子化路径从头训练**，且 `rtau`/softcap 基线要按组数重标定。

效率：插值开销 `Σ_g S·(D/g) = S·D`，**与 g、V 无关**；码表参数同理。16×16 相对
V=256 省 16× 算力、30% 参数；64×64 (V=4096) 与 V=512 的未因子化相比省 8× 算力。

### 36.1 小配置（MAD in-context-recall, vocab=128 / seq=256 / ntr=12800）

| depth | 变体 | best acc | min test loss |
|---|---|---|---|
| 2 mixers | 未因子化 V=256 | 0.9971 | 0.0096 |
| 2 mixers | **16×16** | **0.9991**（28 epoch 就到） | **0.0047** |
| 1 mixer | 未因子化 V=256 | 0.9973 | 0.0110 |
| 1 mixer | 16×16 | 0.9915 | 0.0369 |

**干净交叉**：没有前驱层（1 mixer，DISM 直接读 embedding）时因子化明显吃亏；
有前驱层时明显占优。→ 因子化**把码表几何协调的压力摊到了 DISM 之前的层**：码表侧
变简单（每组只在自己 32 维里排 S 个点、组间不共享维度），但表征侧要同时满足 g 个
**固定且不相交子空间**里的对齐条件（合取）。

### 36.2 极限档（vocab=512 / seq=2048，256 keys，ntr=12800，2 mixers）

| 变体 | V | head_dim | best acc | min test loss | 插值开销 |
|---|---|---|---|---|---|
| 未因子化 | 512 | 64 | **0.9940** | **0.030** | 32768 |
| 32×16 | 512 | 64 | 0.9854 | 0.074 | 2048 |
| 64×64 | 4096 | 64 | 0.9647 | 0.199 | 4096 |
| （参考：未因子化 V=512, head_dim=32） | 512 | 32 | 0.8753 | 0.588 | 16384 |

结论：**极限档因子化没赢**——同容量略差；**放大到 4096 反而更差**（和小配置一致，
瓶颈是码字排布本身而不是容量）。另外注意 **head_dim 32→64 在这个任务上是巨大
变量**（0.8753 → 0.9940），任何跨 head_dim 的对比都无效。

### 36.3 结论

- 数学与实现正确（精确等价），工程约束清楚（dg ∈ {32,64}）。
- **收益出现在"码表排布是瓶颈、且有足够前驱层"的场景**（小配置 2 mixers：更准、
  收敛快 7×、省 16× 算力、省 30% 参数）。
- **在难任务上不成立**，且**不能用它来放大容量**。
- 若要继续，方向应是：给因子化配**更深/更宽的前驱表征**（或专门的 symbol-forming
  层），而不是单纯扩组数。

---

## 37. 专用层布局 / direction 的 2x2 对照（含一次归因修正）

四个格子：布局 {valres `gggggghhhhhh` / 专用 `ggggggddgddg`} ×
`direction_policy` {true / random}。其余配置（数据、步数 19074、bs、lr、seed、
`value_residual`）完全一致。专用层由新加的 `DismConfig.layer_layout` 指定
（`configs/sequence_true_72m_valres_spec2d1g[_dirtrue].json`）。

| | 布局 | direction | val_bpb | PG19 16k | NIAH 16k mean-present-NLL |
|---|---|---|---|---|---|
| **A** | valres | true | **1.182854** | **1.4047** | **2.459** |
| B | valres | random | — 缺 — | — | — |
| **C** | 专用 | random | 1.201140 | 1.4488 | **2.454** |
| **D** | 专用 | true | 1.204047 | — | 2.555 |

### 37.1 修正：长程 recall 的提升来自 direction，不是布局

先前用 `gain = NLL_absent − NLL_present` 比较时得出结论"专用层带来长程提升"。
补充 D 组后**该结论被推翻**：用 present NLL（recall 的直接度量，不受 absent 基线
漂移影响）：

| ctx | A valres/true | C 专用/random | D 专用/true |
|---|---|---|---|
| 2048 | 1.724 | **1.433** | 1.804 |
| 8192 | 2.230 | **2.091** | 2.351 |
| 16384 | **2.459** | 2.454 | 2.555 |

**D（专用层 + 固定方向）每一档都最差，甚至不如 valres。** 所以：
- 长程收益来自 **`direction_policy="random"`**（C），不是专用层布局；
- `random` 同时改善 bpb（C 1.2011 < D 1.2040）和 NIAH；
- **专用层布局本身在 bpb 和 NIAH 上都不如 valres**，之前"改成专用层"的动机被削弱。

（方法学教训：`gain` 会被 absent 基线漂移污染；跨模型比较长程能力必须看
**present NLL**。）

### 37.2 train/infer gap

同 step 的 `hard_bpb − soft_bpb`：

| step | A valres/true | C 专用/random | D 专用/true |
|---|---|---|---|
| 1000 | +0.0226 | +0.0331 | **+0.0891** |
| 8000 | +0.0162 | +0.0305 | **+0.0451** |
| 19074 | +0.0058 | +0.0079 | — |

- **专用层 + 固定方向**的 gap 是 valres 的 2.8–4×（step 8000：0.0451 vs 0.0162），
  且分解显示 **soft 侧几乎持平、hard 侧明显更差** → 是 hard 读出路径的问题，不是学不动。
- 机制：hybrid 里 `_combine_cuda` 是 `dism + gdn` 后过**一套共享 RMSNorm**，DISM 的
  soft/hard 差异 ΔDISM 被整个和的尺度除掉（valres 实测 |DISM|/|GDN| = 0.18–1.59
  → 稀释 1.7–6.7×，与观测到的 2.8× gap 比吻合）。专用层没有这个分母，gap 全额暴露。
- **同一个共享 RMSNorm 既是 §35 分支塌缩的起因，也是 gap 被压住的原因。**
  → 三难：稀释⇒gap 小但 DISM 被压制；专用层⇒DISM 全额发挥但 gap 全额暴露。
- `direction=random` 把 gap 减半（D 0.0451 → C 0.0305），与用户此前"固定 direction
  在合成任务上推训 gap 极严重"的经验一致。

### 37.3 下一步

缺格子 **B（valres 布局 + random）** 是关键：若 B 在 bpb 与 NIAH 上都不差于 A，
则结论是"**只需把 valres 的 direction 改成 random，不必改布局**"——代价远小于重构。
另有 gap 侧的两个候选：加快 `hard_prob` 退火（`hard_start_fraction` 0→0.3~0.5，
或 `hard_prob_max` 直接 1.0）与 β 锐化（§19–§26 已验证能把 valres gap 0.0058→0.0029）。

---

## 38. 汇总：预训练评估 + MAD 合成任务评估

所有数字的来源：`val_bpb` 取自各 run 的 `meta_019074.json`；PG19 取自 harness
（`adapter.py` + `pg19.yaml`，hard eval，disjoint 窗口）；NIAH 取自
`dism-fwedu-valres-72m-2500m/diagnostics/niah_results/*.json`（16 trials，
所有模型共用同一组 trial）；train/infer gap 取自 `dual_validation.jsonl`；
MAD 取自 `/tmp/dism_diag/mad_*/` 的 `logs/metrics.csv`（best-epoch，§35.4）。

### 38.1 预训练：val_bpb / PG19 / NIAH

共同前提：fwedu-100B sample、seq 2048、bs 131072、19074 步（2.5B tokens）、
`rear_half_dism`、`value_residual`、seed 1729。**除注明外 `direction_policy="true"`。**

| run | 改了什么 | val_bpb | PG19 2k | PG19 16k |
|---|---|---|---|---|
| slimpajama（换数据） | slimpajama 语料 | 1.318086 | 1.5028 | 1.4368 |
| fwedu（无 value residual） | — | 1.190988 | 1.5272 | 1.4410 |
| **valres（A，基线）** | value_residual | **1.182854** | **1.4374** | **1.4047** |
| valsyncB | β 退火 + hard 调度 + selector β1 | 1.182748 | 1.4399 | 1.4063 |
| valsync | β 退火 + hard 调度 | 1.183907 | — | — |
| sep-rank1 | shareqk + TransVQ rank1 | 1.183828 | 1.4669 | 1.4344 |
| shareqk | 共享 qk 线性 | 1.185901 | — | — |
| valsched | hard_prob 推迟 | 1.189378 | — | — |
| **C = spec2d1g** | 专用层 `ggggggddgddg` + **dir=random** | 1.201140 | 1.4853 | 1.4488 |
| **D = spec2d1g-dirtrue** | 专用层 + dir=true | 1.204047 | 1.4947 | 1.4653 |
| **B = valres + dir=random** | 只改 direction | 🔄 进行中 | | |
| E1 | valres + warm-start β 1→6 | — | — | 1.4069 |
| E2 | valsyncB + β 4→6 | — | — | 1.4105 |
| E3 | sep-rank1 + β 1→6 | — | — | 1.4390 |
| E4 | lite + β 1→6 | — | — | 1.4527 |
| lite | shareqk + TransVQ lite | 1.4895→? | — | 1.4569 |

**NIAH present-NLL**（越低越好；`mean` = depth 0.05/0.5/0.95 平均，`d.05` = 最远端针）：

| model | 2k | 8k | 16k | 2k/d.05 | 8k/d.05 | 16k/d.05 |
|---|---|---|---|---|---|---|
| **fwedu_valres (A)** | **1.724** | 2.230 | **2.459** | 2.158 | 2.480 | 2.811 |
| E1_valres | 1.745 | 2.318 | 2.475 | 2.198 | 2.484 | 2.757 |
| E4_lite | 1.961 | 2.261 | 2.456 | 2.385 | **2.361** | **2.495** |
| fwedu_lite | 1.966 | 2.339 | 2.586 | 2.501 | 2.537 | 2.710 |
| slimpajama_baseline | 1.803 | 2.314 | 2.704 | 2.118 | 2.719 | 2.724 |
| **C = spec2d1g (dir=random)** | **1.433** | **2.091** | **2.454** | **1.516** | **2.113** | 2.611 |
| **D = spec2d1g-dirtrue** | 1.804 | 2.351 | 2.555 | 2.012 | 2.509 | 2.548 |
| fwedu_baseline | 1.858 | 2.529 | 2.852 | 2.410 | 3.116 | 3.262 |
| fwedu_valsyncB | 2.197 | 2.700 | 2.927 | 2.758 | 3.028 | 3.064 |
| E2_valsyncB | 2.350 | 2.801 | 3.081 | 2.841 | 3.138 | 3.216 |
| fwedu_seprank1 | 2.187 | 2.606 | 2.795 | 2.714 | 2.891 | 2.940 |
| E3_seprank1 | 2.285 | 2.705 | 2.896 | 2.750 | 2.921 | 2.973 |

**train/infer gap**（`hard_bpb − soft_bpb`，同 step）：

| step | A valres/true | C 专用/random | D 专用/true |
|---|---|---|---|
| 1000 | +0.0226 | +0.0331 | +0.0891 |
| 8000 | +0.0162 | +0.0305 | +0.0451 |
| 19074 | +0.0058 | +0.0079 | — |

**要点**
1. **bpb 排序：valres(A) < valsyncB < … < C < D**；专用层布局代价 ~0.02 bpb，
   `direction=random` 在专用层上反而更好（C<D）。
2. **NIAH 排序完全不同**：C 在 2k/8k 明显最好（1.433/2.091），A 与 C 在 16k 并列
   （2.459/2.454），**D 每档都最差**。→ 长程收益来自 `direction=random`，不是专用层。
3. **seprank1/valsyncB 家族在 NIAH 上明显差**（16k 2.795–3.081 vs 2.459），
   而它们的 bpb 只差 0.001–0.02 → **bpb 严重低估长程损伤**（§33）。
4. **gap 与"是否被 GDN 稀释"直接相关**：专用层+D 是 valres 的 2.8–4×。

### 38.2 MAD 合成任务（in-context-recall = MQAR）

`best acc / min test loss`。`chance = 1/(V/2)`。

| sl | vocab | ntr | attention | gdn-only | dism-only | **dism+GDN(hybrid)** | 其他 |
|---|---|---|---|---|---|---|---|
| 256 | 16 | 12800 | 0.9995 | 0.9998 | 0.9993 | 0.9997 | — |
| 256 | 128 | 12800 | **0.0645**¹ | 0.9973 | **0.9992** | 0.9950–0.9972 | 解绑 0.9994；per-branch-RMS 0.9856；delta-LP 0.9921；专用 G+2D **0.9992** |
| 1024 | 16 | 3200 | 0.3161 | 0.9993 | 0.9990 | 0.9994 | — |
| 1024 | 128 | 3200 | 0.0548 | 0.9970 | **0.9991** | 0.9638–0.9945 | 解绑 0.9919；per-branch-RMS 0.9930；delta-LP 0.9918；专用 G+2D **0.9991** |
| 2048 | 128 | 1600 | 0.0533 | — | — | 0.9794 | — |
| **2048** | **512** | **1600** | — | 0.0043 | 0.0047 | — | ← **数据量不足，全部 chance** |
| **2048** | **512** | **12800** | — | 0.5791 | **0.8753** | 0.3372² | A(h64,V512) **0.9940**；32×16 0.9854；64×64 0.9647 |

¹ attention 是**双峰**的：同配置另一次跑 226 epoch 后到达 0.9991（induction 电路形成是随机相变）。
² hybrid 在极限档 ep72 只有 0.337 且已饱和 → 同时低于两个单分支。

**要点**
1. **极限档（256 keys, ctx 2048）DISM 大幅胜 GDN**：0.8753 vs 0.5791（loss 0.59 vs 2.08），
   且 DISM 到 200 epoch 仍在涨、GDN 在 60–80 epoch 就饱和。
2. **层内 DISM+GDN 混合一致地不优于任一单分支**，且方差更大
   （hybrid 0.964–0.997 vs dism-only 0.999）；三种混合机制（相加 / per-branch RMS /
   rank-1 delta 写入）都救不了。**专用层布局（GDN + 2×DISM）能回到 0.9992/0.9991。**
3. **`--num-train-examples` 不能随 seq_len 缩放**：缩到 1600 时两个架构都锁死在 chance，
   那是数据量不足而非机制失败（这是最重要的方法学教训）。
4. **head_dim 32→64 在极限档是巨大变量**：0.8753 → 0.9940。
5. 因子化码表见 §36：小配置 2 mixers 更好（0.9991/0.0047 vs 0.9971/0.0096，收敛快 7×、
   省 16× 插值算力），1 mixer 更差（0.9915 vs 0.9973）→ 压力转移到前驱层。

### 38.3 新增任务 `mqmvar`（in-context 值分布 recall）

**任务**：每个 key 拥有一个**每序列随机的值分布**（`n_support`、`n_obs` 都随机），
历史形如 `A B A C A B`，下一个 `A` 应输出 `B=2/3, C=1/3, other=0`。
生成器返回**第三个数组** `dist[T, V]`（答案位置上的计数），dataset 支持 3-tuple，
`PLModelWrap` 见到 3-tuple 就用 **KL(D‖softmax(logits))** 作 loss，并记录
CE / TargetEntropy / Accuracy / Perplexity。位置：`mad/data/instances.py`、
`mad/data/dataset.py`、`mad/model/pl_model_wrapper.py`、`configs/tasks/mqmvar.yml`。

原语：keys = `[0,V/2)`、values = `[V/2,V)`；每 key 随机取 `n_support∈[2,4]` 个值作为
支撑集；历史值在该支撑集上自由采样；答案位置的**目标分布 = 该 key 历史值的经验归一化计数**
（要求历史长度 ≥ `min_history`），答案 token 本身**不需要是该分布的一个样本**
（这正是 KL loss 带来的自由度）。

**结果**（vocab=16 / seq=128 / ntr=12800 / 2 mixers / 200 epoch）：

| loss | arch | best acc | KL | test loss |
|---|---|---|---|---|
| **NTP token CE** | gdn-only | 0.5515 | — | 0.9107 |
| NTP token CE | dism-only | 0.5432 | — | 0.9287 |
| **KL(D‖Q)** | **gdn-only** | **0.9880** | **0.0001** | **0.0001** |
| KL | **dism-only** | 0.9822 | 0.0001 | 0.0001 |
| KL | attention-4h | 0.9633 | 0.0002 | 0.0002 |

参照量：目标分布熵均值 0.7490 nats；history-only plug-in 计数器在 history≥3 的位置
CE ≈ 0.6713。

**要点**
1. **KL loss 彻底解决信噪比**：同一模型从 loss 0.91（比 counting 最优高 ~0.25 nats 且不动）
   变成 **KL = 0.0001、CE = 目标熵**（即 Q ≡ D）。NTP-CE 在同样本量下信噪比太低。
2. 这一档**三个架构都饱和**（KL≈0）→ 无法区分。原因可解释：**计数目标是"值的加性和"**，
   正好是 GDN 状态（外积累加）的原语，所以 GDN 天然会做（与 §38.2 的极限 MQAR 对照，
   那里是精确 recall，DISM 赢）。
3. 要让它有区分度需要往"非加性聚合 / 更长历史 / 更大支撑集"方向加难度。

---

## 39. RegBench（probabilistic DFA）上的 DISM

**仓库**：`~/cs/project/seq_icl`（Akyürek et al., *In-context Language Learning: Architectures
and Algorithms*）。接入见该仓库 commit `49e6d58` / `3f9dbca` / `fac8455`。

### 39.1 数据管线（重要）

`ICLDFADataModule.setup()` **一次性生成全部数据并常驻内存**，之后 `trainer.fit` 就在这
固定一批上训 200 个 epoch（每条被看 200 次）：

| split | 条数 | 真实 token 长度 | 备注 |
|---|---|---|---|
| train | **5000** | min 129 / **mean 375** / p90 501 / max 662 | 每条 = 1 个 DFA + 10–20 个 in-context 例子 |
| val | 250 | mean 373 | |
| test | 250 | mean 375 | |

- **train/val/test 的 DFA 互不相交**（`DFAs[:5000]` → val 取接下来 250 → test 再 250），
  所以任务要求 in-context 推断 DFA，不能靠背 DFA。
- 但**唯一数据只有 ~1.88M token/epoch**，200 epoch ⇒ 对 1.4M 参数模型是 **~1.3 token/参数**
  （Chinchilla 20）→ **任何架构都必然过拟合**。实测 DISM+GDN 在 epoch 40 见顶
  （val/loss 1.0594 / dfa_acc 0.9048），到 epoch 100 退化到 1.3168 / 0.8718；
  而 `gla` 到 100 epoch 仍在缓慢改善。checkpoint 按 best val/loss 保存，所以最终
  `final_val/final_test` 用的是峰值权重，那些数字仍可用。

### 39.2 接入方式

`src/models/sequence/rnn/dism.py`：`_DismMixerBase`（把 `[B,L,D]` 打包成
`[1,B·Lp,D]`，每行一个文档、无跨行匹配，每个文档尾对齐到 256；BF16 autocast；
`direction=None` 保持随机方向——固定方向在合成任务上推训 gap 极大）+
`DismMixer`（DISM+GDN）/ `DismOnlyMixer`（纯 DISM）。注册为 `dism` / `dism-only`，
配置 `experiment=dfa/dism{,-only}`。除 mixer 外与 `dfa/gla`、`dfa/transformer+` 完全一致
（d_model 128 / n_layer 4 / vocab 20 / seq 4096 / bs 32 / 200 epoch / lr 5e-4 /
cosine warmup / seed 0）。

**为在 PL 2.6.1 上跑通所做的两处最小改动**（repo pin 的是 1.8.6）：
1. 删掉 `train.py` 里三个**纯空**的 `training/validation/test_epoch_end`
   （PL2 已移除；原方法只调 `super()`，日志体全被注释掉）；
2. `torch.load` 默认 `weights_only=False`（Lightning checkpoint 内嵌 omegaconf 对象，
   PyTorch≥2.6 的安全白名单会逐个全局级联报错）。

wandb：project `seq-icl-data`，group `dfa-dism` / `dfa-baselines`。

### 39.3 结果（seed 0，各自 best val/loss 权重）

| run | mixer | 参数量 | best val/loss | best val/dfa_acc | **test/dfa_acc** | model_dfa_diff↓ | model_ngram_diff↓ |
|---|---|---|---|---|---|---|---|
| **dfa_dism_only_s0_fixed** | **纯 DISM** | 1,407,440 | 1.0431 | **0.9443** | **0.9465** | 0.487 | 0.407 |
| dfa_transformerp_s0 | transformer+ | **399,360** | **0.9839** | 0.9289 | 0.9222 | **0.443** | **0.327** |
| dfa_dism_s0 | DISM+GDN | 1,483,248 | 1.0545 | 0.9067 | 0.9028 | 0.477 | 0.415 |
| dfa_gla_s0 | GLA（GDN） | 806,912 | 1.2501 | 0.7823 | 0.7910 | 0.705 | 0.528 |
| dfa_lstm_s0 | LSTM | 17,536 | 1.9323（训练中） | 0.5124 | 0.5307 | — | — |

（`n_gram_dfa_acc ≈ 0.93` 是**参考 n-gram 模型**的 DFA 准确率，不是被测模型。）

### 39.4 结论

1. **纯 DISM 拿到最高 DFA 准确率**：test 0.9465 / val 0.9443，高于 transformer+
   (0.9222/0.9289)、远高于 GLA (0.7910/0.7823) 与 LSTM (0.53)。**这支持"类 DISM
   架构在 regbench 上强"的印象。**
2. **DISM+GDN 再次劣于纯 DISM**（0.9028 vs 0.9465）——这是**第三次独立复现**
   （MAD §35 的层内混合、nanochat 的 §37、以及这里的 regbench）。同时它过拟合更早
   （ep40 见顶后退化）。
3. **指标排序不一致**：val/loss 上 transformer+ 最好（0.9839 vs 1.0431），但
   dfa_accuracy 上 DISM 最好。DFA loss 里含大量"复制例子"的 token，会被更容易的部分
   稀释。
4. **算法探针**：`model_dfa_diff` transformer+ 0.443 < DISM+GDN 0.477 < 纯 DISM 0.487
   < GLA 0.705；`model_ngram_diff` transformer+ 0.327 < 纯 DISM 0.407 < DISM+GDN 0.415
   < GLA 0.528。即 **DISM 的 DFA 准确率最高、但离"参考 DFA 算法"和"n-gram 算法"都更远**
   ——它用的是一个不同的机制（与其"最近匹配/后缀匹配"的原语一致），这点值得单独分析。
5. **必须注意的 confound**：DISM 是 **1.41M 参数**，而 transformer+ 只有 **0.40M**
   （3.5×）、GLA 0.81M（1.75×）。所以"准确率最高"里**包含参数量优势**，要下结论需要
   参数对齐的对照。另外**同配置 run 间方差不可忽略**（同一份 hybrid 配置的两次 run，
   一次 ep40 见顶后恶化、一次到 ep54 仍在改善）→ 需要 ≥2–3 seed。

### 39.5 iso-parameter 对照（缩参后的 dism-only）

§39.4 的第 5 条指出的 confound（DISM 1.41M vs transformer+ 0.40M / GLA 0.81M）已通过缩参
消除。DISM mixer 的参数高度集中在 **`q_vocab`+`k_vocab`**（V=512 时 131,072 / 219k = **60%**），
而 DFA 只有 20 个 token，512 的码表纯属浪费——所以 `qk_vocab_size` 是零成本杠杆。
（`n_heads=4` 时 mixer 有 ~96k 的地板：head 投影 conv + o_proj 不随码表缩小，所以到
0.4M 必须减 `n_heads`。）

新增配置（`configs/experiment/dfa/dism-only-{817k,416k}.yaml`）：

| 配置 | 设置 | 参数量 | 对齐基线 |
|---|---|---|---|
| `dism-only-817k` | `qk_vocab 64`, `d_inner 384`, n_heads 4 | 817,104 | `gla` 806,912（+1.3%）|
| `dism-only-416k` | `n_heads 1, hd/vd 64, V 32`, `d_inner 224` | 415,940 | `transformer+` 399,360（+4.2%）|

结果（seed 0）：

| run | 参数量 | 峰值 epoch | **val/loss** | val/dfa_acc | **test/dfa_acc** | model_dfa_diff↓ | model_ngram_diff↓ |
|---|---|---|---|---|---|---|---|
| **dism-only-416k** | 415,940 | 73 | **0.9368** | 0.9393 | **0.9434** | **0.377** | 0.399 |
| transformer+ | 399,360 | 63 | 0.9839 | 0.9289 | 0.9222 | 0.443 | **0.327** |
| dism-only-817k | 817,104 | 59 | 0.9977 | **0.9444** | **0.9440** | 0.411 | 0.399 |
| gla | 806,912 | 136 | 1.2501 | 0.7823 | 0.7910 | 0.705 | 0.528 |
| dism-only（原） | 1,407,440 | 89 | 1.0431 | 0.9443 | 0.9465 | 0.487 | 0.407 |
| dism+GDN | 1,483,248 | 41 | 1.0545 | 0.9067 | 0.9028 | 0.477 | 0.415 |
| lstm | 17,536 | 199 | 1.7958 | 0.5207 | 0.5367 | 1.252 | 0.954 |

**结论（修正 39.4 的保留意见）**

1. **同参数量预算下 DISM 仍然更强**：415,940 参数的 dism-only 在 val/loss（0.9368 vs
   0.9839）**和** test/dfa_acc（0.9434 vs 0.9222）**两项上都优于同预算的 transformer+**；
   对 GLA 则是大幅领先（0.9440 vs 0.7910）。所以 §39.4 里"准确率最高含参数量优势"
   这个保留意见**被消除**。
2. **缩参没有损失，反而改善 loss**：1.41M → 416k，test/dfa_acc 0.9465 → 0.9434
   （噪声级），而 val/loss 1.0431 → **0.9368**。即**多出来的参数纯粹是过拟合容量**。
   DFA 准确率在 **0.42M–1.41M 的 3.4× 参数区间内基本不动**（0.9434–0.9465），
   说明这个任务的瓶颈不在参数预算，而在机制。
3. **算法距离也反过来**：`model_dfa_diff` 在 416k 时是 **0.377，全场最优**（优于
   transformer+ 0.443、GLA 0.705）——缩参后 DISM 的预测分布反而**更接近参考 DFA 算法**。
   `model_ngram_diff` 则是 transformer+ 最低（0.327），即它更接近 n-gram 算法。
4. **DISM+GDN 第四次劣于纯 DISM**（0.9028 vs 0.9465），且过拟合最早（峰值 epoch 41，
   其它为 59–199）。
5. 仍然是 **n=1 seed**，且 416k 配置把 `n_heads` 4→1（结构性代价）。若要更严格，可补
   2–3 seed 和一个"保持 n_heads=4、改用 n_layer=2"的对照（后者对算法性质是更重的干扰）。

---

## 40. 推理复杂度与简化路线（direction / 递归 / 前向分离）

### 40.1 direction 在纯 hard 情形下是**可证明无关**的

```python
hard_score = zeros.masked_fill(idx_q != idx_k, -inf)
mix_score  = torch.where(hard[..., None], hard_score, norm_score)   # hard 行丢掉 norm_score
```

`direction` 只通过 `norm_score`（q-LSE vs k-LSE）和 `select_embedding`（q vs q_from_k）进入
计算，而两者**都只喂给 `norm_score`**。所以 **hard 行上 `q_vec`/`k_vec`/两个 LSE/两个插值
嵌入全部不被使用，只有标签 `idx_q/idx_k` 有效**。

实测（valres@19074，同一权重同一 batch，只改 direction 抽样粒度）：

```
hard loss:  true / false / perhead / allhead / global  全部 = 2.731204   (std = 0.000000)
soft loss:  true 2.717295 | false 2.722826 | perhead 2.718989 (std .00113)
            allhead 2.720302 (std .00137) | global 2.720752 (std .00286)
```

- **推理侧（全 hard）：direction 完全无关** ⇒ 部署 kernel 不需要 direction，也不需要 LSE
  和插值嵌入，只剩「码表 argmax → 标签匹配 → 递推 → sq·sk·v 读出」。这正是 SAM 能把硬
  前向归约成 event stream 的原因。
- **训练侧（soft gate）：direction 才是旋钮**。方向本身有偏好（true 2.7173 < false 2.7228），
  而**粒度只影响方差**：perhead → global 的 per-forward std 涨 2.5×，即粗化粒度 ≈ 在 soft
  路径上把有效 batch 缩小 2.5×。这也解释了 §37 里 random 能压 gap（0.045→0.030）的机制。
- `cycle8`（确定性逐层）此前实测**效果不好**，仍建议 random。

### 40.2 待做（已记录，未执行）

1. **分离 hard / soft 前向**（收益最大，必做）。因为 hard 行不需要 `raw_scores`，而
   `hard_prob` 从 0 爬到 0.95 ⇒ 训练后期只有 ~5% 的平面是 soft ⇒ 插值/LSE 开销可降 ~20×。
   前提是 per-plane 特化（4 桶 = {hard,soft}×{q-LSE,k-LSE}）。
2. **direction 粒度的 3 格配对短训**：`perhead-random`（现状）/ `cycle8` / `global-step`，
   判据 `val_bpb + NIAH(2k/8k/16k) + hard/soft gap`。因 hard 已证明不变，这三格只在测
   "soft 正则化强度是否够"，是干净的单变量对照。
3. **去掉对角递推（后缀匹配）的消融**：见 §41。

## 41. Opt-in ablation: disable diagonal recurrence

`diag_recurrence: bool = True` is the default in flash_dism `DismConfig`,
`DismAttention`, the public core/vocabulary wrappers, and nanochat `DismConfig`.
Set the nanochat model configuration field `diag_recurrence=False` for this
ablation. Configuration serialization preserves it; legacy configurations omit
it and retain `True`. It is a configuration value, not a parameter or buffer:
RNG counters, annealing state, optimizer groups/schema and consumed input
position are unchanged. No training run or checkpoint was modified for this work.

### 41.1 Algebra and exact mathematical semantics

The log-affine composition is

```
(a,b) * (c,d) = (a+c, logaddexp(b+c,d)).
```

Replacing each local `(logM,logM)` by `(-inf,logM)` gives
`(a,b) * (-inf,logM) = (-inf,logM)` for finite incoming scores. Thus the
inclusive scan's second component at every actual element is its own `logM`;
no predecessor, suffix length, warp boundary or chunk boundary contributes.
This applies to both hard and soft rows, without changing their selection:

```
W_ij = rtau                         hard row, idx_q[i] == idx_k[j]
     = -inf                         hard row, labels differ
     = dot(q_vec[i],k_vec[j])-LSE+rtau   soft row (existing direction convention)
```

Only causal keys `j<=i` in the same sequence participate. Let
`z_i=max(0,max_j W_ij)`, `w_ij=exp(W_ij-z_i)` and `f_i=exp(-z_i)`. Then

```
O_i = sum_j w_ij * dot(SQ_i,SK_j) * V_j / (f_i + sum_j w_ij).
```

For a hard row with `m_i` matching causal keys, equivalently
`O_i = exp(rtau) * sum_matching dot(SQ_i,SK_j)*V_j / (1+m_i*exp(rtau))`.
No matches gives exactly zero. This is **not** an ordinary average of V:
SQ/SK remain signed numerator-only readout features, and the unit zero-value
fallback remains in the denominator. Equal labels receive equal match weights,
regardless of the length of their matching suffix. Soft rows retain their local
soft/Jensen score; disabling recurrence does not turn soft rows into hard rows.

### 41.2 Implementation and boundaries

The fixed and packed output kernels select a compile-time `DiagRecurrence`
specialization at host dispatch. The existing tuple duplication is unchanged
when enabled; only the first component becomes literal `-INFINITY` when disabled.
There is no runtime flag branch in the tile loop, no new shared storage, and
no change to synchronization, launch defaults, masks or tolerances.

The existing `LogAffineOp` is unchanged. Its hard-mismatch second component
remains finite `LOG_ZERO=-1e6`; the ablation does not feed `(-inf,-inf)` to the
finite tanh approximation. Masked weights underflow to zero in the supported
score regime. The mathematical reference retains exact infinity semantics;
CUDA still has the existing BF16 MMA/readout and floating-point/tanh rounding.
The exact algebra above is not a claim of arbitrary infinite-input support or
bitwise agreement between CUDA and the FP64 reference.

Backward has matching fixed/packed specializations: recomputation returns the
local score, and the reverse scan's valid-element propagation coefficient is
zero. Local readout, score, LSE and tau gradients remain. The flag is saved in
autograd state and transported through the Dynamo custom operator, so compiled
pretraining cannot silently use the enabled backward. Existing summary/chunk
workspaces are still computed and allocated; the disabled specialization
annihilates their incoming state. Removing that redundant work is not part of
this change and no speedup is claimed.

Torch dense and cached references honor the flag. Keep it fixed for a cache's
lifetime, as with the model parameters. The separately invoked native SAM
`HardDismDecoder` remains a suffix-matching API and has no ablation switch.
The separate opt-in REINFORCE collector explicitly rejects recurrence-off;
it must not silently select the original suffix operator.

### 41.3 Validation

Validation uses conda `blkw`, this worktree's Python package and an isolated
extension under `build/diag_recurrence/`; the original extension is not replaced
while live training jobs are running. Builds use `MAX_JOBS=4` and
`DISM_BUILD_JOBS=4` with this version's `build.py`. GPU inputs are small unit
fixtures (N256/N512 boundary and regression cases), with no optimizer steps,
benchmarks, downloads or quality evaluations.

New tests in `tests/test_diag_recurrence.py` (27 parametrized cases, all passed):

| Test | Cases | What it checks |
| --- | ---: | --- |
| `test_reference_off_closed_form` | 3 | CPU FP64 independent matching-index sum, negative/zero/positive tau, fallback and suffix counterexample |
| `test_reference_default_and_backward` | 3 | Bitwise default/True reference and explicit off backward versus autograd |
| `test_cuda_reference_and_default` | 12 | Both modes, hard/soft/mixed, both directions, fixed/packed, existing `.008/.025` and relative-L2 `<.01` forward acceptance |
| `test_cuda_off_backward` | 2 | Fixed/packed gradients using unchanged `gradient_acceptance.py` thresholds |
| `test_decode_off_matches_dense` | 1 | Torch inference versus dense off reference |
| `test_config_switch_roundtrip` | 1 | Flash configuration serialization and attention construction |
| `test_off_packed_boundaries_and_fallback` | 1 | N512 continuous input, separate documents, empty document, no-match zero output/normalizer |
| `test_compiled_off_vocabulary_autograd` | 2 | Fixed/packed Dynamo+AOT-eager vocabulary forward/backward versus eager |
| `test_nanochat_switch_and_state_schema` | 1 | Nanochat field reaches attention; strict state loading and optimizer schema unchanged, no optimizer step |
| `test_off_supported_shapes` | 1 | One N256 off forward for each of all eight R/D/DV shapes |

Additional checks:

- Existing forward oracle/sign/fallback cases: **14 passed**, optional FP32
  cases **12 skipped** because this production build disables FP32 instances.
- Existing CPU reference tests: **47 passed**.
- Checkpoint serialization/registry-load tests: **3 passed**; four interrupted
  toy-training cases deselected to avoid optimizer-step/training work.
- Forward and multi-configuration codegen tests: **9 passed**. Exact instance
  counts now include both recurrence specializations, retaining the no-CALL,
  register-control and Torch-free dependency checks. They inspect the loaded
  extension's build directory, not stale artifacts from a different build.
- All **48** enabled instruction streams (six changed kernels × eight shapes)
  are identical to the pre-change worktree artifacts. Both specializations
  have no CALL. One small fixed and packed mixed input also gives bitwise-equal
  enabled output **and normalizer** against the pre-change extension.
- `memcheck` and `synccheck`: one N256 fixed hard/off forward each, **zero
  reported errors**. This is not whole-matrix or backward sanitizer coverage.
  Racecheck was not run; existing ARRIVES-immediate and BF16 delta/tau numerical
  limitations from `VARLEN_IMPLEMENTATION.md` / `BACKWARD_DIMS.md` remain.

Representative R32/D64/DV64 compiler resources, in bytes
`stack / spill stores / spill loads` (not runtime performance measurements):

| Kernel | Enabled | Disabled |
| --- | --- | --- |
| fixed forward | 8 / 16 / 24 | 8 / 16 / 24 |
| packed forward | 16 / 16 / 48 | 16 / 16 / 48 |
| fixed backward summary | 336 / 344 / 412 | 200 / 216 / 284 |
| fixed backward QK | 344 / 356 / 388 | 184 / 200 / 228 |
| packed backward summary | 304 / 308 / 340 | 208 / 220 / 252 |
| packed backward QK | 192 / 196 / 232 | 152 / 156 / 192 |

Logs/artifacts are ignored under `build/diag_recurrence/`: `build_all.log`,
`build_final.log`, `final_tests.log`, `followup.log`, `cpu_final.log`,
`sass_audit.log`, `baseline.log`, `memcheck.log`, `synccheck.log`.
The tested new extension is in `build/diag_recurrence/python`; select that
worktree-local directory on `PYTHONPATH` to use it. The normal
`python/cu_flash_dism*.so` was deliberately left untouched for the live jobs.
There are no measured loss, quality, convergence or throughput claims.

The interpretation requiring care is "label-matching positions only": it
applies to **hard rows**. Applying the requested tuple change to the existing
mixed operator also removes recurrence on soft rows, without replacing their
soft score by a label test. There is no remaining algebra uncertainty; the
separate SAM/REINFORCE restrictions and finite-precision limits are stated above.

### 41.4 Changed file:line inventory

Paths below are relative to the repository root. Line numbers identify each
changed hunk's start in the resulting source; tests list their entry points.

- `dism_v3/include/backward/recompute.cuh:36, 81`
- `dism_v3/include/forward/scan.cuh:35, 51, 53`
- `dism_v3/python/flash_dism/__init__.py:21, 31, 35, 45`
- `dism_v3/python/flash_dism/backward.py:16, 22, 24, 28, 36, 39, 42, 49`
- `dism_v3/python/flash_dism/compiler.py:99, 110, 118, 137, 151, 164, 169, 182, 185`
- `dism_v3/python/flash_dism/configuration_dism.py:22, 48`
- `dism_v3/python/flash_dism/decoding.py:89`
- `dism_v3/python/flash_dism/forward.py:6, 14`
- `dism_v3/python/flash_dism/modeling_dism.py:45`
- `dism_v3/python/flash_dism/module.py:128, 144, 384, 391, 394`
- `dism_v3/python/flash_dism/reference/dism_decode_ref.py:33, 101, 117, 152`
- `dism_v3/python/flash_dism/reference/dism_v3_ref.py:8, 19, 49, 76, 88, 106, 127, 129, 139, 183`
- `dism_v3/python/flash_dism/varlen.py:50, 68, 83, 89, 91, 93, 102, 105, 109, 117, 120, 140, 146`
- `dism_v3/python/flash_dism/voc.py:7, 24`
- `dism_v3/src/backward_qk.cu:8, 47, 76, 193, 199`
- `dism_v3/src/backward_summary.cu:8, 46, 113, 181, 187`
- `dism_v3/src/forward.cu:9, 85, 164, 170`
- `dism_v3/src/frontend.cpp:18, 20, 22, 24, 27, 35, 37, 39, 41, 111, 128, 132, 137, 144, 156, 165, 221`
- `dism_v3/src/host/backward_qk.cpp:10, 13, 45, 50, 57`
- `dism_v3/src/host/backward_summary.cpp:10, 13, 40, 45, 59`
- `dism_v3/src/host/forward.cpp:10, 15, 77, 79`
- `dism_v3/src/host/varlen_backward_qk.cpp:11, 14, 44, 49`
- `dism_v3/src/host/varlen_backward_summary.cpp:11, 14, 35, 40`
- `dism_v3/src/host/varlen_forward.cpp:11, 13, 57`
- `dism_v3/src/interface.cpp:20, 35, 39, 75, 80, 93`
- `dism_v3/src/varlen_backward_qk.cu:9, 48, 77, 195, 201`
- `dism_v3/src/varlen_backward_summary.cu:9, 47, 114, 182, 188`
- `dism_v3/src/varlen_forward.cu:10, 85, 164, 170`
- `dism_v3/src/varlen_interface.cpp:16, 20, 22, 35, 40`
- `dism_v3/tests/test_forward.py:82, 85`
- `dism_v3/tests/test_multi_codegen.py:7, 11, 37, 49`
- `nanochat/nanochat/models/dism/__init__.py:59`
- `nanochat/nanochat/models/dism/model.py:89`
- `dism_v3/tests/test_diag_recurrence.py:53, 63, 81, 103, 122, 131, 141, 165, 188, 212` (new test file; helpers at lines 12, 33, 38).
- `dism_v3/docs/SOFT_HARD_RELAXATION.md:2622` (§41, this report).

### 40.3 四格补全：direction × 布局 存在交互（修正 §37.1）

B 组（valres 布局 + `direction=random`）完成后四格齐了。**present-NLL**（越低越好，方括号为
最远端 d0.05）：

| cell | 2048 | 8192 | **16384** | 远端@16k | val_bpb |
|---|---|---|---|---|---|
| **A** valres / true | 1.724 | 2.230 | **2.459** | 2.811 | 1.182854 |
| **B** valres / **random** | 1.870 | 2.397 | 2.614 | 2.901 | **1.180388** |
| **C** 专用 / random | **1.433** | **2.091** | **2.454** | 2.611 | 1.201140 |
| **D** 专用 / true | 1.804 | 2.351 | 2.555 | 2.548 | 1.204047 |

**做错过的推断**：§37.1 由 C vs D 得出"长程收益来自 `direction=random`、不是专用层布局"。
补上 B 后该结论**只在专用层布局下成立**：

- **valres 布局**：random **伤** NIAH（16k present-NLL 2.230→2.397、远端 2.480→2.712），
  但**改善** bpb（1.182854→**1.180388**，全部 run 里最好）。
- **专用层布局**：random 同时改善 bpb 和 NIAH（C vs D：1.201140<1.204047，2.454<2.555）。
- ⇒ **direction 与布局存在交互**，不能当成独立可加的两个开关。之前的"通用规律"是过度概括。

**四格没有支配者**：bpb 排序 `B < A < C < D`；NIAH(2k/8k) 排序 `C < A < D < B`；
16k NIAH `C ≈ A < D < B`。A 是折中（bpb 次优、16k NIAH 并列最优）。

**教训（与 §33/§38 一致）**：bpb 与长程 recall 是两个独立轴，**必须同时看**；本表里 bpb 最优的
B 恰好是远端 recall 最差的之一。

### 40.4 塌缩的定位：`direction=random` + valres 布局 会打塌 qk 码表

用训练期 `--qk-hist-every 500` 记录的 `probe_qk_vocab_usage` 统计（6 个 DISM 层取平均，
末次记录）：

| run | q_coverage | **q_coverage_pooled** | q_max_share | k_coverage | k_coverage_pooled |
|---|---|---|---|---|---|
| A valres / true | 0.833 | (旧探针未记录) | 0.317 | 0.843 | — |
| **B valres / random** | **0.128** | **0.577** | **0.525** | 0.180 | 0.689 |
| C 专用 / random | 0.576 | — | — | 0.636 | — |
| D 专用 / true | 0.572 | **0.981** | 0.445 | 0.498 | 0.990 |

B 的 layer08 逐头 q_coverage = `[0.07, 0.404, 0.018, 0.039, 0.016, 0.049]`：一个头占 40%，
其余 1.6–7%。

**区分两种形态**（这是 `coverage` 与 `coverage_pooled` 同时记录的意义）：

- **D（专用 / true）不是塌缩**：per-head 0.572，但 **pooled 0.981** ⇒ **健康的 head 分工**
  （每头用自己的子集，合起来覆盖 98%）。
- **B（valres / random）是真塌缩**：连 pooled 都只有 **0.577**，且单码字占 **52.5%** 的位置。

**机制解释**：`direction=random` 让同一层的共享码表在**两种 LSE 归一化**（q-LSE / k-LSE）
之间随机切换，两侧几何需求冲突 ⇒ 码表收敛到少数"安全"码字。在 **valres 布局**上 DISM 分支
本身已被 GDN 稀释（§37），梯度不足以解开这个冲突，于是塌缩；在**专用层布局**上 DISM 拿到
完整梯度，同样的随机化只产生健康的分工（pooled 0.98）。

**这解释了 §40.3 的符号翻转**：塌缩后的码表是"更粗的匹配器" ⇒ 平均 bpb 略好
（B 的 1.180388 是全部 run 最优），但精确 recall 变差（远端 NIAH 2.901，最差之一）。
所以 §40.3 里"random 在 valres 布局伤 NIAH、在专用层布局帮 NIAH"不是玄学，而是有没有塌缩。

**行动含义**：`direction=random` **不能**与"层内稀释"共存；若要保留 in-layer hybrid，就得用
固定的/确定性的 direction（A/C/D 都是这个象限）。方向随机化的收益只在专用层布局上成立。
既然 §40.1 已证明 direction 对 hard 前向无关，它纯粹是 **soft 路径的正则化**，而它带来的
码表塌缩是一个**训练动力学**副作用，可在训练中直接用 `coverage_pooled` 监控。

### 41.3 Ablation run

`diag_recurrence=False` 的预训练 run（**唯一变量**，其余与 A 组逐项一致）：

| 项 | 值 |
|---|---|
| run dir | `dism-runs/dism-fwedu-valres-norec-72m-2500m` |
| model-tag | `seqtrue72m-fwedu-valres-norec` |
| config | `nanochat/configs/sequence_true_72m_valres_norec.json`（= valres + `diag_recurrence:false`） |
| 参数量 | **72,035,808（与 A 完全相同）** |
| 其余 | `rear_half_dism`、`value_residual`、`direction_policy=true`、rng_seed 1729、`--data-shuffle --shuffle-seed 1337`、19074 步、bs 131072、lr 1.5e-3 —— 全部同 A |
| 对照 | **A = `seqtrue72m-fwedu-valres`**（val_bpb 1.182854；NIAH present-NLL 1.724/2.230/2.459；gap@8k 0.0162） |

**必须用新扩展**：`PYTHONPATH=dism_v3/build/diag_recurrence/python:dism_v3/python:nanochat`
（默认位置的 `cu_flash_dism` 仍是改动前的，会报 `core_forward(): incompatible function arguments`）。
已在 `train.sh` 内固定。

**判据集合**（缺一不可）：`val_bpb` + **NIAH present-NLL(2k/8k/16k)** + `hard−soft gap` +
`qk/*_coverage_pooled`（§40.4：塌缩的直接指标）。只看 bpb 会漏掉 recall 轴（§33/§38）。

### 41.4 消融结果：递推承载长程精确 recall，而 bpb 看不见

配置见 41.3（唯一变量 `diag_recurrence: false`，其余与 A 逐项一致）。

| | **val_bpb** | NIAH 2k | NIAH 8k | **NIAH 16k** | **远端针 d0.05 @16k** | 近端 d0.95 @16k |
|---|---|---|---|---|---|---|
| **A 有递推** | **1.182854** | **1.724** | **2.230** | **2.459** | **2.811** | 2.051 |
| **ABL 无递推** | 1.190168 | 1.979 | 2.764 | **2.929** | **3.171** | 2.179 |
| Δ | **+0.0073 (0.6%)** | +0.255 | **+0.534** | **+0.470** | **+0.360** | +0.128 |

gain（present-NLL 反推）：远端 @16k **A `+0.28` → ABL `+0.06`**（recall 基本消失）；
近端 @16k **A `+1.10` → ABL `+1.02`**（几乎不受影响）。

码表（6 层平均，末次）：ABL `q_cov 0.31–0.50 / pooled 0.87–0.92 / max_share 0.21–0.33`
vs A `q_cov 0.833 / max_share 0.317` ⇒ **ABL 没有塌缩，是 head 分工**（pooled 0.87–0.92）。

**结论**

1. **递推（后缀匹配）是长程精确 recall 的承重件。** 去掉后远端 recall 从 +0.28 掉到 +0.06，
   而**近端几乎不变** ⇒ 局部通路（SWA/GDN）能处理近端，长程只能靠递推。这也解释了用户
   之前"DISM+SWA 退化成纯 SWA 水平"的经验——**但那是在 bpb 上看不出来的**。
2. **bpb 再次被证明对这条轴是盲的**：长程能力几乎归零，val_bpb 只差 **0.0073（0.6%）**。
   与 §33/§38/§40.3 的结论一致：**必须同时看 NIAH**。
3. **退化不是码表问题**：ABL 的 `coverage_pooled` 0.87–0.92 健康（head 分工），
   所以退化来自**寻址结构**而非码本塌缩。这与 MoM 框架吻合：
   **无递推 = 固定 slot（V 个）路由；有递推 = 变阶 n-gram 寻址（有效 `V_eff^n`）。**
   消融把前者和后者的差距量化出来了，而且**差距随上下文长度增大**（2k +0.26 → 8k +0.53）。
4. **对"指数 slot"主张的意义**：递推就是那个把地址空间从 `V` 抬到 `V_eff^n` 的机制，
   代价是 O(1) 级的扫描状态。这条消融是该主张的**核心正对照**。
5. 仍欠：**长度外推**（train@2k → 8k/16k/64k，hard 路径）与 MoM/GLA 的**等状态量四行对照**。

### 41.5 待做（已记录，未执行）

1. **长度外推**：固定权重（train@2k）在 8k/16k/64k/256k 上评 **hard 路径**的 recall。
   §41.6 分析给出：外推性质来自**绝对位置不进入计算**（唯一的"位置"是因果掩码与对角偏移
   算术），归一化是 `exp(score − zmax)` 的相对形式（不是除以 N），且匹配是离散精确等值 ⇒
   排序与 N 无关。**因此只评 hard**；软路径的锐度随 N 变化，用 soft loss 评会把"软锐度漂移"
   误读成"外推失败"。同时记录**累加器幅值 vs N**（每桶状态是事件求和，幅值 ~O(N)，
   是"无条件"最可能的失效点）与 **GDN 门控累积**（经典 SSM 长度漂移）。
   判据：hard 路径精确 recall 应无退化；任何退化须能归因到上述两条而非"位置超范围"。
2. **等状态量四行对照**：`GLA / MoM(K slots) / 无递推 DISM(= V slots) / DISM(+递推)`
   在同一 **M-scan**（按上下文基数 M = V, V², V³… 扫）任务上。
   预期形态：GLA 与 MoM 随 M 饱和（slot 数固定），DISM 到 `M ~ V_eff^n` 才饱和。
   这一张表同时回答"递推值多少"（对照 §41.4）与"离散量化值多少"（对照 GLA）。
   §41.4 已量化：递推把地址空间从 `V` 抬到 `V_eff^n`，而消融的差距**随上下文增大**
   （2k +0.26 → 8k +0.53）。

### 41.6 递推效率优化：bf16x2 状态二元组（template 特化）

现状：前向 log-domain 递推与反向普通仿射递推的状态二元组都是 **fp32x2**。
半年前实验显示 **bf16x2 几乎不掉点（甚至略强）**。计划做成**两个独立可调的格式开关**，
放在 **template 参数**里，通过 template instantiation 生成特化 kernel：

| 路径 | 算子 | 定义位置 |
|---|---|---|
| **前向 log-domain affine** | `pscore::LogAffineOp`：`(a,b)*(c,d) = (a+c, softplus_approx(b+c,d))`，`identity()={0, LOG_ZERO}` | `include/summary/primitives.cuh:407` |
| **反向普通 affine** | `pscore::AffineComposeOp`：`(a,b)*(c,d) = (a·c, b·c+d)`，`identity()={1,0}` | `include/ds_alt.cuh:63`（被 `backward/recompute.cuh:18` 使用） |

存储：`float *SummaryA/SummaryB`（`summary/primitives.cuh:312`、`varlen/summary.cuh:16`），
写入点 `summary/kernel_common.cuh:76`（`value.first.u0/u1`，即 `PackedF2`），
另有谓词化的 `AffineComposeOp::combine_either`
（`ds_alt.cuh:469`）与跨 warpgroup 边界的反向仿射交换（`backward/types.cuh:72`）。

**必须注意的坑**：前向的 `identity()/`哨兵是 `LOG_ZERO = -1e6f`，**bf16 表示不了 -1e6**
（bf16 指数范围仅 ~±3.4e38 但尾数 8 bit；-1e6 可表示但 `softplus_approx` 的
`fmaxf`/`tanh.approx` 分支与掩码下溢行为会变）⇒ 需要重新定义 bf16 可表示的哨兵，
并证明掩码/下溢语义在支持区间内等价，而不是简单把 `float` 换成半精度。

要求：**两个开关独立**、默认仍是 fp32x2 且**逐位不变**、**template 特化（tile 循环内无
运行时分支）**、边界交换处的格式一致。

**交接文件**：`dism_v3/docs/BF16_TUPLE_PRIOR_ART.md` 汇总了 `~/cs/projects/glx/experiments`
里 H100 上的先验实测。**要点（会改变本次优化的预期）**：

1. **bf16x2 不是速度优化，应作为格式/容量选项。** H100 上实测**更慢**：
   `full_forward MMA F32-tanh 1285.9 → BF16-tanh 2194.2 周期（1.7×）`，
   尽管指令更少（870→781）且 **SHFL 减半**（124→62）。原因是打包 bf16 MUFU 展开成
   2 条标量 `MUFU.*.BF16` + 1 条 `PRMT`。
2. **精度契约**：bf16x2 SoA 布局实测 **`max_abs_error = 3.10e-02`**（来自 `a+c` 的舍入）。
   `LOG_ZERO = -1e6f` 这类哨兵在 bf16 下语义会变，必须**重新定义并测试**，不能只换类型。
3. **EX2 多项式（n=2，err 2.2e-3）无速度优势**：`scan_only` 持平，`full_forward` +2.9%，
   带 broadcast 的 `special_full` **+24.8%**。
4. **真正的瓶颈是 wgmma/MUFU 争用**：密集 wgmma 下扫描退化 **1.4–2.5×**。所以调 op 的
   指令配比不是收益所在。
5. **架构差异**：`MMABuffer` 的 roll/unroll 在 **H100 占 ~403/1286 周期，RTX 5090（sm_120）
   占 ~1724/2527**。H100 的结论**不能照搬**，必须在目标 GPU 上重测。

### 41.7 Opt-in BF16 recurrence state pairs (2026-10-13)

**Experimental; numerical acceptance is incomplete.** The component-isolation
follow-up in §41.8 rejects all tested BF16 replacements across the sampled
frontier; the finite-state range below is not an attention-accuracy guarantee.
Two independent configuration
fields, `forward_diag_bf16x2=False` and `backward_diag_bf16x2=False`, now propagate
from flash_dism/nanochat configuration through attention, eager/compiled vocabulary
operators, fixed/packed core autograd and native host dispatch. Both default to
FP32x2. They are configuration values, not Parameters or buffers; RNG counters,
annealing state, no-decay groups, optimizer schema and consumed input position
are not reset. No training run or existing checkpoint was touched.

`forward_diag_bf16x2=True` selects BF16 pairs for summary, output scan and backward
forward-state recomputation. `backward_diag_bf16x2=True` selects BF16 reverse pairs
in B1 and B3. Each composition evaluates in FP32 and rounds the two resulting
components to BF16 RN. Two adjacent strip elements are packed per component, so
one diagonal's affine pair occupies two BF16 values. This is actual packed register
storage, not FP32 storage with a runtime quantization branch. The two scalar chunk
passing kernels remain FP32. SoA summaries/checkpoints retain their FP32 external
layout. The reverse WG mailbox retains float4 layout and expands already-rounded
BF16 pairs on send, then repacks them on receipt; this round-trip is exact and
preserves mailbox offsets, shared-memory size, synchronization and launch defaults.
No shared FP32 atomics were added.

Host dispatch selects template instantiations; format flags are absent from device
argument structures. `diag_recurrence=False` plus either BF16 format is rejected
by both the core and low-level host APIs. Cached/Torch inference and the separate
REINFORCE collector reject BF16 selection, rather than silently ignoring it.
The mathematical references retain their exact infinity semantics unchanged.

#### Formats and finite score contract

BF16 has the same eight exponent bits as FP32. The problem in §41.6 is precision,
not an exponent range too small for one million: `-1e6` rounds to `-999424` in BF16.
The BF16 log operator uses exactly representable `-2^20` instead. Supported reachable
**log2 state components and input scores must lie in `[-65536,65536]` at every
composition/boundary**, with finite input scores; this is a masking/range contract,
not a guarantee of small attention/gradient error throughout that interval.
Legacy masked checkpoint values below `-65536` are treated as the sentinel. Such
values cannot also be used as reachable scores in this format. A sentinel scale
annihilates its incoming state, and sentinel offsets are handled explicitly.
The finite tanh approximation receives only a clamped gap in `[0,32]`; a gap of
32 or greater uses the maximum. No BF16-path sentinel or infinity enters tanh.
The default FP32 LogAffineOp and its finite `-1e6` arithmetic are unchanged: its
existing bounded-score/context contract in SUMMARY.md still applies, without a
new universal input bound or a promise for arbitrary infinite inputs.

For BF16 ordinary affine, scales are in `[0,1]` and finite offsets/states must
remain within `[-65536,65536]` at every boundary. Identity `{1,0}` and zero scale
are exact. FP32 ordinary affine retains the pre-existing finite arithmetic
contract. BF16 subnormal and zero behavior is checked by the device probes;
there is no oracle substitution or relaxation of existing tests.

The forward range protects masking: an unreachable incoming term never becomes
reachable after adding a supported positive score, and an unreachable output
has exactly zero exp2 weight against the nonnegative fallback normalizer.
The probe enumerates every finite BF16 value in the stated range and checks both
identity orders and hard-mismatch annihilation. Padding is the same identity.
Training API lengths and packed boundaries still must be 256-aligned; nonaligned
inputs reject, including with the new flags. This change does not add padding.

#### Instantiations and selection

For each `KernelConfig<R,D,DV>`, R=16/32, D=32/64, DV=32/64:

- Fixed/packed summary add `ForwardDiagBF16=true` (two new kernels per shape).
- Fixed/packed forward add `(DiagRecurrence,ForwardDiagBF16)=(true,true)`.
- Each fixed/packed backward B1/B3 adds
  `(DiagRecurrence,ForwardDiagBF16,BackwardDiagBF16)` equal to
  `(true,false,true)`, `(true,true,false)`, `(true,true,true)`.
- Existing FP32 on/off kernels remain. Backward BF16 instantiations live in
  separate `*_format.cu` translation units to preserve the original FP32 kernel
  body and code generation. The default FP32 backward recompute helper is retained
  verbatim; the format helper is separate.

With production BF16 output this is **16 new primary kernels per shape, 128 over
all eight shapes**; there are 34 primary instances per shape including the previous
on/off variants and chunk kernels. Optional `DISM_ENABLE_FP32=1` also instantiates
FP32 output variants, giving `2 + 14*2 = 30` new primary kernels per shape; those
optional output variants were not validated here. Probes are separately opt-in.
CUDA device compilation remains Torch-free.

Builds use this worktree's `build.py`, conda `blkw`, `MAX_JOBS=4` and
`DISM_BUILD_JOBS=4`. The isolated directory is `dism_v3/build/bf16x2`, with `src`
and `include` symlinks into this worktree and a copy of this version's build.py.
From that directory:

```bash
PATH=/usr/local/cuda/bin:/home/cicuvc/miniconda3/envs/blkw/bin:$PATH \
MAX_JOBS=4 DISM_BUILD_JOBS=4 DISM_BUILD_PROBES=1 DISM_LINEINFO=1 \
/home/cicuvc/miniconda3/envs/blkw/bin/python build.py
```

From the repository root, select the new extension and this worktree's Python:

```bash
export PYTHONPATH="$PWD/dism_v3/build/bf16x2/python:$PWD/dism_v3/python:$PWD/dism_v3/tests:$PWD/nanochat"
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q dism_v3/tests/test_diag_format.py
```

Both flags can be set on `nanochat.models.dism.DismConfig` or passed to
`flash_dism.dism_core` / `voc_dism`. This provides an experimental A/B configuration,
**not evidence that positive-tau training is numerically accepted**. Default-false
Python calls retain the previous extension argument list; enabling a format with
an older extension raises an explicit error. The working-tree `python/cu_flash_dism`
binary is untouched (SHA256
`e9e7bd6895c5f841f82594967f5afcc761d502acc77e65a34e838e43dd098a1e`). The pre-existing
Python `diag_recurrence` interface still requires a compatible extension; the older
isolated `build/diag_recurrence/python` can serve default-format calls only.

#### Numerical acceptance and limitations

The new accuracy matrix compares against the FP32-pair CUDA path using identical
inputs and upstream derivatives: all three opt-in combinations, fixed/packed,
N256/512, soft/random-label/repeated-hard/mismatch, tau=0/0.3/ln4, both LSE
directions. Packed tests include an empty document. Forward gates are the unchanged
`.008` absolute / `.025` relative elementwise limits plus a separate new 3% vector
relative-L2 gate (a BF16 accumulation allowance, not a change to the existing 1%
forward tests). Gradient gates reuse `gradient_acceptance.py`: vector cosine .999,
relative L2 .03, absolute norm allowance 1e-4; LSE .995/.10/.01; tau .95/.50/.02.
These are the existing accepted BF16-output cancellation allowances, not a new
justification for large recurrence error. All tests retain finite and exact-zero
checks. No existing tolerance was changed.

The first R32/D64/DV64 matrix has **22 ordinary numerical failures**, all on
repeated hard-label chains with positive tau. Forward BF16 with tau=.3/N256 has
about **12.06% output relative L2 error**; BF16 backward alone leaves forward output
and normalizer bitwise unchanged but fails the tau direction gate (one case
cosine **0.87987**, required .95). A separate diagnostic sweep observes forward
BF16 gradient errors as large as ~25% in dV/dSQ/dSK and much larger relative tau
errors. These results are not covered by the passing zero-mask proof.

Long positive chains expose coarse spacing of the log state. Moreover, backward
recomputes the forward state with a transposed scan/tile partition: BF16 rounding
is not associative, so this need not reproduce the forward rounded values. The
FP32 implementation's much smaller version of this effect cannot justify treating
BF16 recomputation as exact. This is a numerical limitation, not a sanitizer
finding. No claim of “almost no loss”, throughput improvement, training convergence,
or correctness for long positive repeated chains is made. Fixing that discrepancy
without extra state or a different recurrence evaluation remains unresolved.
The switches remain opt-in and the failing tests are neither xfailed nor hidden.

#### Validation results and resource report

The full new suite (excluding the two separately run baseline checks) reports
**195 passed, 22 failed**. Its 48 all-shape fixed/packed format cases passed;
the 22 failures are the repeated-hard positive-tau accuracy cases described above.
The earlier 163-case run was 141 passed / 22 failed. Compiled eager/AOT comparisons,
configuration serialization, nanochat attention propagation and strict state/schema
comparisons all passed. Existing selected regressions passed 250 cases with 22
optional skips; one stale-artifact codegen test initially failed because it read
the working-tree build instead of the loaded extension. It now reads the isolated
extension and checks both new packed backward format kernels too; its rerun passed.
Three checkpoint serialization/registry-load tests passed (four toy interrupted
training cases deselected; no optimizer steps or training runs were launched).

Both memcheck and synccheck passed **48 all-shape cases each, zero errors**. This
executes every one of the 128 new primary BF16-output instantiations through all
three opt-in format combinations, fixed and packed. An additional eight-case
R32/D64/DV64 N512 boundary/identity/predicated-alias selection passed each sanitizer
with zero errors. This is small-fixture coverage, not exhaustive sanitizer coverage.
Racecheck, sm90, FP32-output builds and long-context quality/performance were not
validated. Existing saved-BF16-output delta/tau cancellation limitations remain.
The source-mapped mask checks, no-CALL/register-control checks and Torch-free device
dependency checks passed for all eight shapes and all 34 primary instances/shape.

Compiler resource numbers below are **stack / spill-store / spill-load bytes**,
from `ptxas -v`, not measured dynamic spill traffic. They enumerate every new
primary BF16-output instantiation. Register budgets and launches were not tuned.
Zero spill is not claimed for these variants.

| R/D/DV | Summary fixed F=BF16 | Summary packed F=BF16 | Forward fixed F=BF16 | Forward packed F=BF16 |
| --- | ---: | ---: | ---: | ---: |
| 16/32/32 | 0/0/0 | 0/0/0 | 0/0/0 | 24/20/64 |
| 16/32/64 | 0/0/0 | 0/0/0 | 0/0/0 | 16/16/48 |
| 16/64/32 | 0/0/0 | 16/16/40 | 0/0/0 | 16/16/48 |
| 16/64/64 | 0/0/0 | 16/16/40 | 0/0/0 | 16/16/48 |
| 32/32/32 | 0/0/0 | 0/0/0 | 0/0/0 | 16/16/48 |
| 32/32/64 | 0/0/0 | 0/0/0 | 0/0/0 | 16/16/48 |
| 32/64/32 | 0/0/0 | 16/16/40 | 0/0/0 | 16/16/48 |
| 32/64/64 | 0/0/0 | 16/16/40 | 8/16/24 | 16/16/48 |

Fixed B1 (F/B denote forward/reverse pair format):

| R/D/DV | F=FP32, B=BF16 | F=BF16, B=FP32 | F=BF16, B=BF16 |
| --- | ---: | ---: | ---: |
| 16/32/32 | 128/140/196 | 128/140/196 | 144/156/212 |
| 16/32/64 | 208/224/316 | 216/232/324 | 200/212/304 |
| 16/64/32 | 160/172/236 | 160/172/236 | 160/172/236 |
| 16/64/64 | 224/236/308 | 192/204/288 | 192/204/276 |
| 32/32/32 | 152/164/220 | 152/164/220 | 152/164/220 |
| 32/32/64 | 264/280/352 | 288/300/372 | 264/280/352 |
| 32/64/32 | 176/192/260 | 160/172/228 | 176/188/252 |
| 32/64/64 | 336/356/428 | 312/336/408 | 272/296/368 |

Fixed B3 (F/B denote forward/reverse pair format):

| R/D/DV | F=FP32, B=BF16 | F=BF16, B=FP32 | F=BF16, B=BF16 |
| --- | ---: | ---: | ---: |
| 16/32/32 | 88/104/120 | 152/164/180 | 96/108/124 |
| 16/32/64 | 168/184/204 | 192/208/240 | 168/184/204 |
| 16/64/32 | 200/216/248 | 208/228/260 | 200/216/248 |
| 16/64/64 | 184/200/232 | 240/256/288 | 160/176/196 |
| 32/32/32 | 112/124/140 | 144/156/176 | 112/124/140 |
| 32/32/64 | 168/184/204 | 208/224/256 | 168/184/204 |
| 32/64/32 | 168/180/204 | 240/256/288 | 168/188/212 |
| 32/64/64 | 200/216/248 | 360/376/408 | 184/204/236 |

Packed B1 (F/B denote forward/reverse pair format):

| R/D/DV | F=FP32, B=BF16 | F=BF16, B=FP32 | F=BF16, B=BF16 |
| --- | ---: | ---: | ---: |
| 16/32/32 | 104/112/144 | 88/96/128 | 112/116/148 |
| 16/32/64 | 192/196/228 | 256/268/296 | 192/196/228 |
| 16/64/32 | 152/156/188 | 136/144/176 | 152/160/192 |
| 16/64/64 | 240/244/276 | 208/208/244 | 216/220/252 |
| 32/32/32 | 144/148/180 | 144/152/184 | 160/164/196 |
| 32/32/64 | 264/268/300 | 272/284/312 | 264/268/300 |
| 32/64/32 | 176/180/212 | 168/172/204 | 176/180/212 |
| 32/64/64 | 320/328/364 | 304/316/344 | 280/280/312 |

Packed B3 (F/B denote forward/reverse pair format):

| R/D/DV | F=FP32, B=BF16 | F=BF16, B=FP32 | F=BF16, B=BF16 |
| --- | ---: | ---: | ---: |
| 16/32/32 | 16/16/52 | 56/64/92 | 16/16/52 |
| 16/32/64 | 96/104/140 | 112/116/152 | 88/92/128 |
| 16/64/32 | 120/124/160 | 136/144/180 | 120/120/156 |
| 16/64/64 | 144/144/180 | 176/180/216 | 136/136/172 |
| 32/32/32 | 48/52/88 | 88/96/132 | 32/40/76 |
| 32/32/64 | 72/80/116 | 112/116/152 | 64/72/108 |
| 32/64/32 | 144/144/180 | 176/180/216 | 144/148/184 |
| 32/64/64 | 168/168/204 | 192/196/232 | 168/168/204 |

The two fresh-baseline checks also passed: **112 pre-existing instruction streams**
(64 recurrence-enabled/summary streams plus 48 recurrence-off streams) are identical
across all eight shapes. Fixed and packed R32/D64/DV64 random, repeated-hard and
mismatched N512 fixtures have bitwise-identical default output, normalizer, vertical
checkpoints and horizontal boundaries versus the pre-change build. All eight
mismatched-input gradients are bitwise identical too. Nonzero FP32 atomic gradient
sums are not asserted bitwise across independent executions; their kernel streams
are identical and their existing numerical regressions pass. There is no claim of
making the pre-existing atomics deterministic. Reproduce these checks with
`DISM_DIAG_BASELINE=$PWD/dism_v3/build/bf16x2/before` and
`pytest -q dism_v3/tests/test_diag_format.py -k 'default_instruction_streams or default_bitwise_prechange'`.

Artifacts stay ignored under `build/bf16x2/`: `build_all.log`,
`build_isolated.log`, `build_baseline_all.log`, `tests_all.log`, `default_tests.log`,
`regression.log`, `regression_followup.log`, `checkpoint_tests.log`,
`memcheck{,_all}.log`, `synccheck{,_all}.log`, `numerics.json`, and `resources.json`.
`before/` is a snapshot of the starting worktree source and a fresh pre-change
build with the same toolchain. It includes the user's uncommitted diag_recurrence
changes; historical diag_recurrence cubins alone are not a reproducible baseline
for the current source/toolchain. No generated artifact is staged for Git.

#### Changed file:line inventory

Each number is the start of a changed hunk in the resulting source, relative to
this task's starting worktree (not HEAD, which already had the user's recurrence
ablation changes). New files are listed at line 1. Existing reference edits from
the earlier task were preserved byte-for-byte; none are part of this change.

- `dism_v3/include/backward/recompute.cuh:15, 120`
- `dism_v3/include/backward/types.cuh:73`
- `dism_v3/include/diag_format.cuh:1` (new file)
- `dism_v3/include/forward/scan.cuh:11, 36, 48, 53, 74`
- `dism_v3/include/summary/kernel_common.cuh:76, 81`
- `dism_v3/include/summary/primitives.cuh:10, 444, 509, 515`
- `dism_v3/src/backward_qk.cu:193, 200, 202`
- `dism_v3/src/backward_qk_format.cu:1` (new file)
- `dism_v3/src/backward_summary.cu:181, 188, 190`
- `dism_v3/src/backward_summary_format.cu:1` (new file)
- `dism_v3/src/forward.cu:9, 12, 63, 86, 109, 115, 170, 175, 177`
- `dism_v3/src/frontend.cpp:16, 18, 20, 22, 27, 33, 35, 37, 39, 111, 114, 122, 125, 130, 134, 139, 147, 150, 161, 170, 227, 230, 251`
- `dism_v3/src/host/backward_qk.cpp:10, 13, 45, 50, 57`
- `dism_v3/src/host/backward_summary.cpp:10, 13, 40, 45, 59`
- `dism_v3/src/host/forward.cpp:10, 15, 78, 80`
- `dism_v3/src/host/varlen_backward_qk.cpp:11, 14, 45, 50`
- `dism_v3/src/host/varlen_backward_summary.cpp:11, 14, 36, 41`
- `dism_v3/src/host/varlen_forward.cpp:11, 13, 58`
- `dism_v3/src/host/varlen_summary.cpp:10, 12, 39`
- `dism_v3/src/host/wmma_tma_preprocess.cpp:10, 14, 17`
- `dism_v3/src/interface.cpp:15, 20, 35, 39, 64, 69, 76, 90, 104, 106`
- `dism_v3/src/probe/diag_format.cu:1` (new file)
- `dism_v3/src/probe/host/diag_format.cpp:1` (new file)
- `dism_v3/src/varlen_backward_qk.cu:195, 202, 204`
- `dism_v3/src/varlen_backward_qk_format.cu:1` (new file)
- `dism_v3/src/varlen_backward_summary.cu:182, 189, 191`
- `dism_v3/src/varlen_backward_summary_format.cu:1` (new file)
- `dism_v3/src/varlen_forward.cu:10, 13, 63, 86, 109, 115, 170, 175, 177`
- `dism_v3/src/varlen_interface.cpp:13, 16, 20, 22, 31, 35, 48, 58`
- `dism_v3/src/varlen_summary.cu:9, 39, 108, 133`
- `dism_v3/src/wmma_tma_preprocess.cu:7, 34, 105, 130`
- `dism_v3/python/flash_dism/__init__.py:21, 31, 35, 45`
- `dism_v3/python/flash_dism/backward.py:6, 17, 23, 25, 30, 39, 42, 45, 52`
- `dism_v3/python/flash_dism/compiler.py:7, 100, 111, 119, 138, 152, 165, 171, 185, 188`
- `dism_v3/python/flash_dism/configuration_dism.py:22, 49`
- `dism_v3/python/flash_dism/diag_format.py:1` (new file)
- `dism_v3/python/flash_dism/forward.py:2, 7, 15`
- `dism_v3/python/flash_dism/modeling_dism.py:46`
- `dism_v3/python/flash_dism/module.py:128, 145, 254, 388, 395, 398`
- `dism_v3/python/flash_dism/varlen.py:2, 51, 69, 84, 90, 92, 95, 105, 108, 112, 120, 123, 143, 149`
- `dism_v3/python/flash_dism/voc.py:7, 24`
- `dism_v3/tests/test_diag_format.py:1` (new file)
- `dism_v3/tests/test_forward.py:86`
- `dism_v3/tests/test_multi_codegen.py:24, 43, 58, 66`
- `dism_v3/tests/test_summary_codegen.py:14, 23, 30, 32`
- `dism_v3/tests/test_varlen_execution.py:115, 127`
- `nanochat/nanochat/models/dism/__init__.py:60, 61`
- `nanochat/nanochat/models/dism/model.py:90, 91`
- `dism_v3/docs/SOFT_HARD_RELAXATION.md:2977` (§41.7, including this report)

### 41.8 Component frontier and sm_120 follow-up (2026-10-13)

#### Earlier plan, retained with corrections

以下是本轮开始前的计划记录，不是本轮测量结论。下文实测推翻了其中
“非对称格式消除反向重算不一致”的判断：`second` 也随正链增长，单独
将其量化仍失败。25% 仅指持久 tuple 的容量减少，不能等同整个 kernel
寄存器减少。半年前的训练记录尚未找到；没有启动这里计划的训练 A/B。

§41.7 的精度矩阵是**对抗性最坏情形**，不是训练相关的估计量。必须区分两个独立问题：

| 问题 | 判据 | 状态 |
|---|---|---|
| **质量**：bf16x2 训练是否掉点 | 短预训练 A/B 的 `val_bpb + NIAH(2k/8k/16k) + gap + qk coverage_pooled` | 待跑 |
| **资源**：是否真的降低寄存器/SHFL 压力 | sm_120 上 `regs/thread + spill`（主）、SHFL/LSU 指令数（辅）、占用率 | Astra P2 |

判据：**两者都成立才值得采用**；若质量成立而寄存器/SHFL 没降，则本次改动无意义
（H100 上 bf16x2 反而慢 1.7×，而本机 sm_120 的 LSU 是 Hopper 的一半，需实测）。

支持"训练可能没问题"的证据：§41.7 的 **22 个失败全部集中在"重复 hard-label 链 + 正 tau"
这一族退化序列**上，而不是分散到各类输入；且有半年前"bf16x2 几乎不掉点（甚至略强）"的
训练经验。反对的证据是前向 L2 12% / 梯度 25% 的最坏值，以及**反向重算与前向舍入不一致**
（bf16 舍入不满足结合律 + 转置划分），后者是结构性风险而非仅精度问题。

**非对称格式（fp32 `first` + bf16 `second`，1.5 regs）作为对冲保留**：它把**随链长增长**的
第一个分量留在 fp32，因此即使训练对纯 bf16x2 不容忍，也能拿到 25% 的寄存器节省，
并顺带消除反向重算不一致。Astra 当前轮已按此实现。

**计划**：等当前 Astra 轮结束（它会改同一批源码，期间不启动训练）后，跑短预训练 A/B：
`fp32x2`（基线，已有 A 组可复用）/ `bf16x2` / `非对称`，同 seed 同数据，配合 sm_120 资源报告一起判。

#### Measured follow-up

**No tested BF16 combination passes all unchanged numerical gates.** Both log
components matter, and reverse-affine-only BF16 is also rejected. This is a
negative numerical result, not evidence about training loss, structural-task
quality, or end-to-end throughput. The microbench/resource report is in
[BF16_TUPLE_PRIOR_ART.md §5](BF16_TUPLE_PRIOR_ART.md#5-关键更正h100-的结论不能外推到-sm_120本节优先级最高).
The historical experiment reconciliation is in that document's §6.

#### Component isolation and the original fixture

The diagnostic copies specialize `BinaryElement<BF16x1/BF16x2>` with independent
FP32/BF16 fields. A pair of diagonals occupies 4/3/2 32-bit words for FP32/mixed/
both-BF16 persistent tuple storage. Arithmetic and RN conversions follow §41.7;
the mixed fields round independently on ingress, composition, and boundary
exchange. Splitting chained `first=second=value` assignments is essential:
otherwise a nominal FP32 field can inherit the other field's rounding.
Production headers, defaults, launch settings, reference, and tests were not edited.

At N256/tau=.3, the unchanged original `fixture` (seed410, two heads with opposite
LSE directions, one repeated chain and one periodic-label chain) gives:

| Format | Output rel-L2 % | max abs Δ log2 Z | max vector grad rel-L2 % | tau grad rel-L2 (ratio) | tau cosine |
| --- | --- | --- | --- | --- | --- |
| Forward first only | 5.587 | 0.3113 | 6.072 | 0.9796 | 0.2802 |
| Forward second only | 9.305 | 0.4199 | 12.04 | 12.71 | 0.7187 |
| Forward both | 12.06 | 0.7274 | 13.42 | 12.6 | 0.7585 |
| Reverse both only | 0 | 0 | 0 | 0.4945 | 0.8799 |

Thus `second` is the larger contributor in this fixture, especially for tau,
but `first` alone already fails. Errors are not additive. The reproduced vector
error here is 13.42%, not 25%; the earlier “~25%” was a different sweep, not a
bound or this exact fixture. Repeating the original N512 fixture gives larger
errors still:

| N | tau | Forward both output % | max vector gradient % |
| --- | --- | --- | --- |
| 256 | 0.3 | 12.06 | 13.42 |
| 256 | 1.386 | 0.6869 | 0.753 |
| 512 | 0.3 | 26.67 | 31.01 |
| 512 | 1.386 | 23.61 | 34.75 |

#### Frontier: actual chain breaks, not merely changing both labels

`tools/bf16_frontier.py` uses R32/D64/DV64, B1/H2, seed410 BF16 Q/K/SQ/SK/V,
upstream derivative .125, and fixed CUDA execution (`ctas=1`). The reference is
the same extension's default FP32-pair path, asserted finite for every fixture.
For controlled lengths, all K labels and matching Q labels are zero; one fully
mismatched Q row follows each L matching rows. Changing Q and K labels together
would **not** break the main diagonal and is not a valid chain-length control.
N=max(256,L) remains training-aligned. Both LSE directions are present.

Measured L=16,32,64,128,256,512,1024,2048 and natural-log tau=
-8,-2,-.3,0,.001,.01,.03,.1,.3,.7,ln4,2,4,8,16. Extra signed-range probes at
L=16,256,2048 set `L*tau/log(2)` to ±64,±256,±1024,±8192,±60000.
These reach near both ends of the stated finite log-state interval; masked
sentinels are excluded from reachable-state bounds. Six original fixtures
(N256/512 × tau0/.3/ln4) are also retained. This is **156 fixtures × 3 switch
combinations × 3 field formats = 1,404 measurements**, not an exhaustive proof
for every real-valued score, context length, or tensor shape.

Raw `{first,second,both}.json` under ignored `build/bf16_followup/` records, for
**every** sample: output and log2-normalizer relative/absolute L2 and max error;
all eight gradient relative/absolute L2, max error, reference norm, and cosine;
failed unmodified gates; extension path, tau, L and N. Nonfinite metrics are
retained as Infinity/NaN in Python JSON output, not dropped. Console logs retain
all failure lists. “Vector” below is max(dV,dSQ,dSK) relative L2; hard-mode Q/K/LSE
gradients are structural zeros. Tau columns are gradient errors, not errors in
the input tau. `log2 Z` is the returned normalizer, not exponentiated Z.

At tau=.3, with forward format selected and ordinary reverse recurrence FP32:

| BF16 field(s) | L | Output % | max abs Δ log2 Z | Vector grad % | tau grad rel-L2 | tau cosine |
| --- | --- | --- | --- | --- | --- | --- |
| first | 16 | 0.1994 | 0.008957 | 0.2205 | 0.112 | 0.9981 |
| first | 32 | 0.2964 | 0.02047 | 0.33 | 0.01827 | 0.9998 |
| first | 64 | 0.3976 | 0.05775 | 0.4394 | 0.03422 | 0.9998 |
| first | 128 | 1.533 | 0.1156 | 1.804 | 0.2487 | 0.9784 |
| first | 256 | 4.706 | 0.2726 | 5.023 | 0.4421 | 0.9354 |
| first | 512 | 12.81 | 0.6111 | 12.57 | 0.1373 | 0.9917 |
| first | 1024 | 29.95 | 1.403 | 32.49 | 1.675 | -0.4455 |
| first | 2048 | 69.69 | 3.483 | 70.73 | 2.638 | -0.8302 |
| second | 16 | 0.5483 | 0.01504 | 0.6656 | 0.2369 | 0.9949 |
| second | 32 | 0.8504 | 0.05861 | 0.9256 | 0.07332 | 0.9984 |
| second | 64 | 3.328 | 0.07269 | 4.022 | 0.3728 | 0.9817 |
| second | 128 | 4.965 | 0.1203 | 5.909 | 5.397 | -0.6255 |
| second | 256 | 10.72 | 0.282 | 15.46 | 21.8 | 0.8291 |
| second | 512 | 30.26 | 1.091 | 39.31 | 11.63 | 0.3544 |
| second | 1024 | 45.89 | 1.944 | 50.9 | 390.7 | 0.9854 |
| second | 2048 | 76.43 | 2.468 | 71.01 | 84.99 | 0.4471 |
| both | 16 | 0.5597 | 0.01973 | 0.6323 | 0.4025 | 0.9935 |
| both | 32 | 0.6584 | 0.07612 | 0.7163 | 0.03938 | 0.9995 |
| both | 64 | 2.83 | 0.1368 | 3.763 | 0.1781 | 0.9936 |
| both | 128 | 5.749 | 0.3012 | 7.251 | 6.238 | -0.7618 |
| both | 256 | 10.95 | 0.6639 | 14.18 | 21.3 | 0.8248 |
| both | 512 | 30.37 | 0.7725 | 37.45 | 13.06 | 0.7168 |
| both | 1024 | 62.79 | 3.329 | 60 | 251.2 | 0.9777 |
| both | 2048 | 110.4 | 6.537 | 92.05 | 205.7 | 0.596 |

Tau dependence at L256 (same controlled fixture and forward-only selection):

| BF16 field(s) | tau | Output % | max abs Δ log2 Z | Vector grad % | tau grad rel-L2 | tau cosine |
| --- | --- | --- | --- | --- | --- | --- |
| first | -8 | 0.05321 | 6.482e-07 | 0.04217 | 1.986e-05 | 1 |
| first | 0 | 0 | 9.537e-07 | 0 | 1.207e-06 | 1 |
| first | 0.001 | 0.08018 | 0.0001736 | 0.09675 | 0.00748 | 1 |
| first | 0.01 | 0.1393 | 0.003109 | 0.1419 | 0.04919 | 0.9995 |
| first | 0.1 | 1.545 | 0.08003 | 1.646 | 0.05266 | 0.9999 |
| first | 0.3 | 4.706 | 0.2726 | 5.023 | 0.4421 | 0.9354 |
| first | 0.7 | 0.639 | 0.5166 | 0.7295 | 0.3047 | 0.9846 |
| first | 1.386 | 0.01228 | 3.052e-05 | 0.003946 | 0.02102 | 0.9999 |
| first | 4 | 35.16 | 3.791 | 36.61 | 4.173 | -0.9458 |
| first | 16 | 33.45 | 12.35 | 39.32 | 5.36 | -0.06175 |
| second | -8 | 1.433 | 0.001686 | 1.442 | 0.0132 | 1 |
| second | 0 | 0.9555 | 0.01159 | 0.9721 | 0.1525 | 0.9997 |
| second | 0.001 | 1.035 | 0.01363 | 1.021 | 0.1323 | 0.9992 |
| second | 0.01 | 1.296 | 0.01528 | 1.431 | 1.345 | 0.7726 |
| second | 0.1 | 4.641 | 0.2237 | 5.786 | 1.17 | 0.1425 |
| second | 0.3 | 10.72 | 0.282 | 15.46 | 21.8 | 0.8291 |
| second | 0.7 | 4.353 | 0.9579 | 3.143 | 3.881 | 0.9807 |
| second | 1.386 | 1.121 | 0.4152 | 1.057 | 0.01056 | 1 |
| second | 4 | 16.89 | 12.61 | 41.03 | 167.2 | -0.5056 |
| second | 16 | 17.08 | 50.63 | 3279 | 85.47 | -0.9986 |
| both | -8 | 1.433 | 0.001686 | 1.442 | 0.0132 | 1 |
| both | 0 | 0.9555 | 0.01159 | 0.9721 | 0.1525 | 0.9997 |
| both | 0.001 | 1.037 | 0.01363 | 1.02 | 0.126 | 0.9994 |
| both | 0.01 | 1.3 | 0.01755 | 1.452 | 1.298 | 0.7582 |
| both | 0.1 | 5.327 | 0.3051 | 6.413 | 1.063 | 0.223 |
| both | 0.3 | 10.95 | 0.6639 | 14.18 | 21.3 | 0.8248 |
| both | 0.7 | 12.7 | 0.9579 | 11.68 | 3.865 | 0.9908 |
| both | 1.386 | 1.121 | 0.4152 | 1.057 | 0.01056 | 1 |
| both | 4 | 48.95 | 14.84 | 1837 | 1.876e+04 | -0.6616 |
| both | 16 | 49.36 | 59.55 | 3.03e+10 | 1.348e+11 | -0.9445 |

The first **sampled** failing chain length at each tau is below. A dash means
all eight sampled lengths passed; it does not certify longer lengths or intervening
tau. Failure is any existing forward/gradient gate, not only the 3% output gate.
This frontier is nonmonotonic: e.g. tau=ln4 makes the hard log2 step exactly2,
so first-component sums can be exact while second-component rounding persists.

| tau | F first | F second | F both | B first | B second | B both |
| --- | --- | --- | --- | --- | --- | --- |
| -8 | — | — | — | — | — | — |
| -2 | — | — | — | — | — | — |
| -0.3 | — | — | — | — | — | — |
| 0 | — | — | — | 2048 | — | 2048 |
| 0.001 | — | — | — | — | — | — |
| 0.01 | — | 256 | 256 | 256 | — | 256 |
| 0.03 | 2048 | 256 | 256 | — | — | — |
| 0.1 | 512 | 128 | 128 | — | 2048 | 2048 |
| 0.3 | 256 | 64 | 64 | — | 128 | 128 |
| 0.7 | 512 | 128 | 128 | — | 1024 | 1024 |
| 1.386 | — | 64 | 64 | — | 256 | 256 |
| 2 | 128 | 16 | 16 | — | 1024 | 1024 |
| 4 | 32 | 16 | 16 | — | — | — |
| 8 | 32 | 16 | 16 | — | 64 | 64 |
| 16 | 16 | 16 | 16 | — | — | — |

#### What fails, and why the range statement is insufficient

For a matching positive chain, `first≈L*tau/log(2)` and `second` grows on the
same scale. BF16 spacing in [64,128) is .5 in log2 units, and in [128,256) it is1.
Even sub-unit log error matters after `exp2(W-log2 Z)`. Only a perfectly common
shift would cancel through normalization; the scan creates different rounding
errors for different keys and chain histories. Keeping `second` FP32 cannot
undo erroneous `first` additions; keeping `first` FP32 cannot undo repeated
rounding of the chain value. At tau=.3, the original N256 chain reaches only
about111 log2 units, far inside ±65536, yet output/vector/tau gates already fail.

Backward recomputes W using transposed tiles and a different composition tree.
Rounding is non-associative. This produces a different W from the forward pass
that supplied log2 Z, and consequently inconsistent probability and gradient
weights. The component ablations measure this complete production effect;
they do not separately quantify how much of each gradient error is due to
forward probability error versus recomputation mismatch. The latter mechanism
is source-derived, not a measured attribution percentage.

L1024/tau16 and L2048/tau16 yield **nonfinite vector/tau gradients** with
forward-second or forward-both BF16, despite finite FP32 baselines and reachable
log states inside the advertised interval. Some tau gradients already become
nonfinite at tau8. Large within-range log discrepancies can overflow backward
`exp2(recomputed_W - forward_log2_Z)`; the normalizer error at L2048/tau16 is
583.3 log2 units for both-BF16. These are ordinary numerical failures.

**Precisely:** §41.7's ±65536 statement explicitly promises sentinel/masking
separation, not output/gradient accuracy. This experiment does not refute that
limited sentinel proof. It refutes interpreting that interval as a numerically
supported attention/training contract: unchanged accuracy and even finite-gradient
requirements fail well inside it. The .031 one-composition prior-art error is
not a bound on a composed chain or its gradients. No tolerance was changed.

#### Variants, reverse affine, and rescaling cost

The original 144-case fixed/packed accuracy matrix remains unmodified:

| BF16 field(s) in selected path(s) | Passed | Failed | Meaning |
| --- | --- | --- | --- |
| both | 122 | 22 | Reproduces §41.7 |
| first | 136 | 8 | All failures select forward BF16 at positive tau |
| second | 122 | 22 | No cure from FP32 first |

The larger frontier gives the following numbers of failing fixtures out of156
per column; all unlisted fixtures passed their unchanged gates:

| BF16 field(s) | Forward only | Reverse only | Both directions |
| --- | --- | --- | --- |
| first | 42 | 2 | 44 |
| second | 74 | 18 | 74 |
| both | 74 | 21 | 76 |

Reverse-only both-BF16 reproduces tau cosine .8798739 < .95 on the original
N256/tau=.3 fixture, with output, normalizer, dV/dSQ/dSK unchanged. Scales in
[0,1] prevent explosive scale products, but they do not protect the precision
of the **signed offset sum**. Tau differentiates a cancellation-sensitive sum;
rounding the offset dominates the observed original failure. Keeping that offset
FP32 (reverse-first-only) passes the original matrix but fails the extended
L256/tau=.01 and L2048/tau=0 cases. Thus even that narrower option is not a
universally accepted replacement. Passing individual fixtures is reported, not
promoted to a quality guarantee.

For a candidate rescaling `(a,b)=(r+â,r+b̂)`, composition can use
`r_out=r_l+r_r`, `â_out=â_l+â_r`,
`b̂_out=lse2(b̂_l+â_r,b̂_r-r_l)`. This algebra explains both the cost and the
limitation: a growing offset reappears in the second term, and bounding both
residuals requires rebasing. Arbitrary chain breaks give different references
on different diagonals; one common tile/strip scalar cannot guarantee small
residuals for simultaneous reset and long-growing chains. One FP32 reference
**per diagonal** plus two BF16 residuals costs4 32-bit words for two diagonals,
versus4 for the original FP32 pair and2 for both-BF16. Shuffle payload returns
to128 bits/pair of diagonals, plus rebase temporaries and reference boundary
handling (8 extra bytes per pair if explicitly stored). A mixed residual/FP32
field plus that reference costs5 words. Materializing references at existing
FP32 checkpoints avoids new external storage but requires reconstructing/rebasing
them, and does not prove forward/backward rounding agreement. Even one reference per
diagonal is not a guarantee that both residuals stay small: on a negative chain
`first` decreases while `second` saturates, so their separation itself grows.
Two independent FP32 references plus two BF16 residuals would cost6 words per
pair of diagonals. This variant was
**not implemented**, as requested for an extra-state proposal. No branch or
extra state was inserted into production tile loops; bounded residual accuracy
and a cheaper shared-reference scheme remain unverified.

#### Isolation, reproduction, validation, and changed files

No active training process was visible before work; GPU processes were desktop
applications only. No run directory, checkpoint, optimizer, dataloader position,
RNG counter, or glx file was written. No downloads, external jobs, or training
were launched. The working-tree extension SHA256 was unchanged:
`e9e7bd6895c5f841f82594967f5afcc761d502acc77e65a34e838e43dd098a1e`.

Use conda `blkw` (`/home/cicuvc/miniconda3/envs/blkw/bin/python`). From repo root:

```bash
python dism_v3/tools/bf16_component_build.py first dism_v3/build/NEW_first
# Repeat for second with a distinct directory; the helper refuses overwrite.
# Within the created directory, run this version's copied build.py:
PATH=/usr/local/cuda/bin:/home/cicuvc/miniconda3/envs/blkw/bin:$PATH MAX_JOBS=4 DISM_BUILD_JOBS=4 DISM_BUILD_CONFIGS=32,64,64 DISM_LINEINFO=1 python build.py
```

The actual mixed roots were `dism_v3/build/bf16_followup/{first,second}`.
The existing compatible `dism_v3/build/bf16x2` extension supplies the unchanged
both-BF16 reference. No other worktree's flash_dism/cu_flash_dism is imported.
Exact Python selection (ROOT means `/home/cicuvc/cs/project/dism-core`):

```text
PYTHONPATH=ROOT/dism_v3/build/bf16_followup/first/python:ROOT/dism_v3/python:ROOT/dism_v3/tests
PYTHONPATH=ROOT/dism_v3/build/bf16_followup/second/python:ROOT/dism_v3/python:ROOT/dism_v3/tests
PYTHONPATH=ROOT/dism_v3/build/bf16x2/python:ROOT/dism_v3/python:ROOT/dism_v3/tests
```

With the corresponding path, run `dism_v3/tools/bf16_frontier.py --format first|second|both
--output <ignored-build-path>.json` and `python -m pytest
 dism_v3/tests/test_diag_format.py -k test_format_accuracy -q` from repo root.
The `.cu.d` dependencies in both new builds contain no Torch/ATen/c10 headers.
The two mixed builds each contain39 device entries, all audited for CALL and
stack/spills; the resource inventory also includes the corresponding39 entries
from the existing both-format extension. Other shapes were not rebuilt.

`bf16_component_smoke.py`: six N512 fixed/packed format combinations passed
**memcheck and synccheck separately for each mixed extension**, each with zero
reported errors. The second-field standalone benchmark also passed memcheck.
The initial attempt to reuse `test_boundary_exchange_smoke` under first-field
memcheck returned four assertion failures/two passes and zero sanitizer errors:
that test asserts **both** reverse fields are BF16, which intentionally does not
describe the mixed format. The test was left unchanged; the separate component
smoke retains the numerical gates and checks the applicable boundary executions.
No numerical failure is waived. Racecheck, all-shape mixed variants, FP32-output
variants, training/structural quality, concurrent tensor-core contention, and
end-to-end attention throughput were not measured.

Only these source/documentation files were changed in this follow-up (line1 for
new files; the cited sections identify edits to existing files):

- `dism_v3/tools/bf16_component_build.py:1` — isolated mixed-field source builder.
- `dism_v3/tools/bf16_frontier.py:1` — chain/tau/output/normalizer/gradient sweep.
- `dism_v3/tools/bf16_component_smoke.py:1` — mixed-format sanitizer fixture.
- `dism_v3/tools/bf16_scan_bench.cu:1` — Torch-free16x64 scan/MUFU benchmark.
- `dism_v3/tools/bf16_bench_build.py:1` — literal sm_120 compile/disassembly helper.
- `dism_v3/tools/bf16_sass_report.py:1` — static loop/category inventory.
- `dism_v3/tools/bf16_resource_inventory.py:1` — every-entry resource/CALL audit.
- `dism_v3/docs/SOFT_HARD_RELAXATION.md:2979` — §41.7 status cross-reference; §41.8 at line3290.
- `dism_v3/docs/BF16_TUPLE_PRIOR_ART.md:102` — §5 status/measurements and §6 reconciliation at line364.

Raw binaries, copies, logs, timing CSVs, JSON, and SASS remain ignored under
`dism_v3/build/bf16_followup/`; no generated artifacts were added to Git.

### 41.9 更正 Astra 的解读：主指标是 spill，且收益只在反向

Astra §41.8 / `ASYMMETRIC_SM120_REPORT.md` 把 `regs=168`、1 CTA/SM 当作"没有寄存器收益"的证据，
**这两项都不该作为判据**：

- `168 regs/thread` 是 **12-warp 配置 + `setmaxnreg` 重分配下的上限**；`setmaxnreg` 之后
  ptxas 只按 **warp 数**分配，所以这个数是天花板而非可优化量。
- **1 CTA/SM 是固定设计**，不是待优化的 occupancy（post-Ampere 后 occupancy 已不再是主要矛盾）。

**真正的指标是 spill。** 实测（`stack / static spill store / load` 字节，ptxas）：

| kernel | 格式 (fwd/bwd) | **spill B** | hot SHFL |
|---|---|---|---|
| forward fixed | P/any, B/any, M/any | **8/16/24（三者相同）** | 65–66 |
| B3 packed | P/**P** | 192/196/232 | 130 |
| B3 packed | P/**B** | **168/168/204** | **108** |
| B3 packed | P/M | 176/180/216 | 119 |
| B3 packed | **B**/P | 192/196/232 | 125 |
| B3 packed | B/B | **168/168/204** | 103 |
| **B3 fixed** | P/P | **344/356/388** | 130 |
| **B1 fixed** | P/P | **336/356/428** | 90 |

**结论**

1. **spill 严重的地方全在反向**（B3/B1 达 168–428 B），**前向只有 8 B 且三格式完全相同**
   ⇒ **前向改低精度零收益，前向应保持 FP32x2**。
2. **收益只能来自反向格式**：反向切 bf16 ⇒ B3 spill **192→168（−12.5%）**、hot SHFL
   **130→108（−17%）**；非对称反向小一些（192→176，−8%）。改前向格式对反向 spill 无影响
   （P/P 与 B/P 相同）。
3. 因此要做的 A/B 是 **`forward=FP32x2` 固定 + `backward∈{FP32x2, BF16x2, 非对称}`**，
   而不是 fwd×bwd 全组合。
4. **"只保留增长分量就行"的假设被否定**：非对称格式也失败，且 **N512 误差 25.53% 大于
   N256 的 9.31%** ⇒ 反向重算与前向舍入不一致（bf16 舍入不满足结合律 + 转置划分）很可能
   才是误差主因，而不是第一个分量的精度。
5. Astra 的"保留 FP32x2、优先调度与寄存器寿命"这一建议**在质量未知的前提下过早**；
   按 §41.8 的两问框架，仍应跑短预训练 A/B（反向三格式）来裁质量，配 spill/SHFL 数据判资源。

### 41.10 状态收尾（bf16 tuple 方向结论：无收益，已回退）

**结论**：**没有测到收益**，该方向到此为止。为此把三轮（§41.7–§41.9）的**全部未提交代码改动已回退**，
工作树在 `d9f04db` 干净，**HEAD 上不存在任何 bf16 tuple / 非对称格式的代码或配置字段**。

**HEAD 的实际状态**

| 项 | 状态 |
|---|---|
| 前向/反向状态二元组 | **均为 FP32x2**（默认，未变） |
| `*_bf16x2` / 非对称格式配置字段 | **不存在**（已回退） |
| `diag_recurrence`（§41.6 的无递推开关） | **存在于 HEAD 且可用**；默认 PYTHONPATH 下 seq_icl 的 DISM smoke test 端到端通过，**无需重建扩展** |
| 无递推消融 run | `dism-runs/dism-fwedu-valres-norec-72m-2500m`（val_bpb 1.190168；§41.4 有完整对照） |

**证据保存在仓库外**（避免污染工作树）：
`dism-runs/dism-fwedu-valres-72m-2500m/diagnostics/bf16_tuple_rounds/`
— 第三轮 `ROUND3_PACKED_BF16_SASS.md`、第二轮 `ASYMMETRIC_SM120_REPORT.md`、三轮 Astra 日志。

**判据与结论（供后续不再重复踩坑）**

1. **主指标是 ptxas spill，不是寄存器数、不是占用率**：`168 regs/thread` 是 12-warp +
   `setmaxnreg` 的**天花板**（之后 ptxas 只按 warp 数分配），`1 CTA/SM` 是**固定设计**。
2. **spill 全部集中在反向**：前向 `8/16/24 B` 且三种格式相同（改前向零收益，且不影响反向 spill）；
   B3 packed 192–232 B、B3 fixed 344–388 B、B1 fixed 336–428 B。
3. **唯一的（微小）资源收益来自反向格式**：反向切 bf16 ⇒ B3 spill **192→168（−12.5%）**、
   hot SHFL **130→108（−17%）**。但**没有测到吞吐改善**（参考 §1/§5：H100 上反而慢 1.7×；
   sm_120 上 packed bf16 MUFU 仍展开为 2 标量 MUFU + PRMT）。
4. **数值上失败**：bf16x2 与**非对称**格式**都**在"重复 hard-label 长链 + 正 tau"上失败
   （前向 worst 12.06%；非对称 N256 9.31% / **N512 25.53%**，即随 N 恶化）。"只保留增长分量
   就能救"的假设**被否定**，主因更可能是**反向重算与前向舍入不一致**（bf16 舍入非结合 +
   转置划分）。**不要**据此声称"训练应该没问题"——那需要独立的预训练 A/B，而本次未做。
5. **不要重复的**：非对称格式、前向格式改动、以及"把整个二元组换成 bf16"。
   若将来仍要降低反向 spill，方向应是**调度与寄存器寿命**，或把重算与扫描的划分对齐
   （消除非结合舍入导致的前后向不一致），而不是换存储精度。

---

## 42. 本轮收尾（2026-10-13）

### 42.1 文档结构说明（有历史遗留）

正文按时间追加，所以**编号顺序不严格单调**：
- §40 的 `40.3` / `40.4` 在 §41 之后（追加时插不进去了）；
- §41 的 `41.5`–`41.10` 是 Astra 在 `diag_recurrence` 之后追加的（bf16x2 递推状态方向，
  §41.10 结论是**无收益已回退**）；
- `§36` 有两条同号标题（第二条已改标 **36b**）。

按主题读：**§1–§28** 理论/软硬 gap/REINFORCE；**§29–§30** 全硬梯度诊断；
**§31–§34** 长文本与 NIAH；**§35–§36b** MAD 与因子化；**§37–§41** 布局×direction 四格、
码表塌缩、简化路线、无递推消融。

### 42.2 本轮确立的结论

**A 级（多次独立复现，可写进论文）**

1. **bpb 与长程 recall 是两个独立的轴**；只看 bpb 会**系统性漏掉长程损伤**。四次独立复现：
   §33（sep-rank1 家族 bpb 差 0.001–0.02，NIAH 差 0.34–0.62 nats）、§38（10 模型总表）、
   §40.3（B 组 bpb 最好而远端最差）、§41.4（递推消融 bpb 只差 0.6%，远端 recall 归零）。
2. **层内 DISM+GDN 混合一致地不优于任一单分支**，且 run 间方差更大；专用层布局
   （GDN + 2×DISM）能回到纯 DISM 水平。四次独立：§35（MAD）、§37（nanochat 2×2）、
   §39.5（RegBench）、§35.5（极限 MQAR）。
3. **后缀匹配递推承载长程精确 recall**（§41.4）：去掉后退化**随上下文增大**
   （2k +0.26 → 8k +0.53），远端 gain +0.28→+0.06 而近端几乎不变；且**码表未塌缩**
   （pooled 0.87–0.92）⇒ 是**寻址结构**而非码本问题。
4. **DISM 的优势精确落在"离散/算法式 in-context 推断"这条轴上**：RegBench **iso-param**
   0.9434 vs transformer+ 0.9222（val/loss 也更低）vs GLA 0.7910；极限 MQAR 0.875 vs
   0.579（loss 差 3.5×）。而在加性聚合（mqmvar）、通用 bpb、NC¹ 可达的容易档上**不体现**。

**B 级（单次复现或仅有理论论证，需再验）**

5. **direction 对 hard 前向可证明无关**（§40.1：代码级 + 实测 5 种粒度 std=0），
   纯粹是 soft 路径的正则化。⇒ 推理 kernel 可完全不实现 direction/LSE/插值嵌入。
6. **direction × 布局存在交互**（§40.3）；`direction=random` 叠加**层内稀释**会**打塌码表**
   （§40.4：pooled 0.577、单码字占 52.5%）。这与 §40.3 的符号翻转是同一件事。
7. **PQ 因子化精确**（§36，1e-06 级；§36.1 的 1-vs-2-mixer 交叉说明压力转移到前驱层）。
8. **MoM 对应与"指数 slot"假设**（§41.5 起）：无递推 = 固定 `V` slot 路由；
   有递推 = 变阶 n-gram 寻址，有效地址空间 `V_eff^n`，而携带状态 `O(R·DV)`/stream。
   **尚无直接测量**——这是 scale 前最该补的一项。

### 42.3 规模前提：72M 说服力不足

本轮**全部**结论都在 72M / 2.5B tokens 上取得。需要 scale 才能判断：
- "指数 slot" 是否随参数/深度保持（现在 n=1、小模型）；
- 递推的收益是否随规模保持（小模型上它已决定长程成败）；
- **层内混合的负结论是否在更大规模翻转**（混合方法常在小规模不利、大规模反转）。
所以 **scale 前必须先做效率优化**，否则同等 GPU 时间能覆盖的规模小一圈。

### 42.4 效率优化清单（按收益排序）

| # | 项 | 收益 | 风险/前提 | 状态 |
|---|---|---|---|---|
| 1 | **分离 hard/soft 前向** | 插值/LSE 开销后期可降 **~20×**（hard_prob→0.95 ⇒ 只有 ~5% soft 平面） | 前提是 per-plane 特化（4 桶 = {hard,soft}×{q-LSE,k-LSE}） | 必做，未做 |
| 2 | **推理删 direction/LSE/插值嵌入** | 部署 kernel 只剩 量化器+标签匹配+递推+读出 | **有证明**（§40.1） | 零风险，未做 |
| 3 | **PQ 因子化** | 插值开销对 V 变常数；16×16 省 16× | 压力转移到前驱层（需深度配合） | 已实现且精确 |
| 4 | 免费项：删 `top1_q/top1_k`；删 4 个死配置 + `direction_antithetic` | 死工作量 + 死代码 | 逐位不变 | 已定位，未做 |
| 5 | direction 粒度 3 格配对短训 | 可能省一个插值侧（~2×） | 需看 `coverage_pooled` | 待做 |
| 6 | bf16x2 递推状态二元组 | — | — | **已试，无收益，已回退**（§41.6–41.10） |

### 42.5 scale 前的 gate（必须先过）

1. **长度外推**：train@2k → 8k/16k/64k/256k，**只评 hard 路径**（软路径锐度随 N 变，会误读），
   并记录累加器幅值 vs N（§40 里"无条件外推"最可能的失效点）。
2. **M-scan 等状态量四行对照**：`GLA / MoM(K slots) / no-recur DISM(V slots) / DISM(+递推)`。
   一张表同时回答"递推值多少""离散量化值多少""指数 slot 是否成立"。
3. **≥2–3 seed 的 iso-param RegBench + 极限 MQAR**：现在 n=1，`0.9434 vs 0.9222` 要钉成显著；
   建议**配对设计**（同初始化/同数据顺序），比独立 seed 省一个量级。
4. **`coverage_pooled` 进常规监控**：它是"指数地址的底数"（§41.5 的 `6.5^n` 论证）；
   跌破 ~0.8 即预警（§40.4）。

### 42.6 工程状态与风险

- **扩展双轨**：Astra 把新扩展隔离在 `dism_v3/build/diag_recurrence/python`，默认位置仍是旧的
  ⇒ 任何新启动的 DISM 训练都必须显式把隔离路径放进 PYTHONPATH，否则报
  `core_forward(): incompatible function arguments`（seq_icl 的 817k 就是这样挂的）。
  **建议收敛为一条**（原地重建默认扩展；已加载旧扩展的进程不受影响）。
- **`/` 分区 97% 满（剩 ~15G）**；wandb 写在 `dism-core/wandb/`（已在 `/`，1.2G / 47 runs）。
- **`/tmp` 被清过一次**，诊断脚本丢失；关键脚本已复制到
  `dism-runs/dism-fwedu-valres-72m-2500m/diagnostics/`（`niah.py`、`pg19_ablate.py`、
  `run_niah_models.sh`、`niah_results/`）。
- **一次异常退出**（norec run，step 5594）：无 OOM、无 Xid、无 kill 记录；wandb 心跳先断
  12 分钟且同期网络故障（natfrp i/o timeout + SSH 暴力破解流量）⇒ 疑似 wandb online
  在断网下抛异常。**建议长跑改 offline 模式**，训练后统一 sync。

### 42.7 产物清单

**dism-core**（103 commits）：
- 记录：`dism_v3/docs/SOFT_HARD_RELAXATION.md` §1–§42
- 实现：`flash_dism/pq_vocab.py`（因子化码表 + 精确性自检）；`diag_recurrence` 开关
  （config→attention→compiled→fixed/packed CUDA）；`DismConfig.layer_layout`（专用层布局）；
  `reinforce.py` + `reinforce_loss.py`（warm-start score-function，默认关闭）
- 配置：`nanochat/configs/sequence_true_72m_valres{,_dirrandom,_norec,_spec2d1g,_spec2d1g_dirtrue}.json`

**mad-lab**：`mqmvar` 任务（随机 support + 分布目标 + KL loss + 3-tuple 数据管线）；
DISM/GDN 层与布局变体（hybrid / only / untied / normed / delta-LP / PQ）

**seq_icl**：DISM mixer（`DismMixer`/`DismOnlyMixer`）+ iso-param DFA 配置；PL2 兼容补丁

**run dirs**（均在 `dism-runs/`）：`dism-fwedu-valres-72m-2500m`（A，基线）、
`-dirrandom`（B）、`-spec2d1g`（C）、`-spec2d1g-dirtrue`（D）、`-norec`（递推消融）

### 42.8 未尽事项

见 §40.2、§41.5，以及 42.4 / 42.5。

### 42.9 消融解读的更正与"距离 × 分支"分工假设

**事实更正：valres 里没有 SWA。** `attention_type='hybrid_gdn'` → `DismGdnAttention` =
**DISM + GDN**；整个模型是 6 层纯 GDN + 6 层 (DISM+GDN)。`window_size` /
`DismSwaAttention` 只走 `attention_type='hybrid'` 那条路径（此前"DISM+SWA 去递推后退化成
纯 SWA"的经验来自**那个**配置，不是 valres）。

**因此 §41.4 的结论更强**：递推消融**完整保留了每层的 GDN 分支**，而 GDN 的 recall 能力
实测并不弱（极限 MQAR 256 keys/seq2048：**GDN 0.579** vs DISM 0.875）。但远端针 NIAH 仍然
从 gain +0.28 掉到 +0.06。即：

> 去掉递推后剩下的模型 = **GDN + 一个"分桶线性注意力"（无递推 DISM ＝ MoM 形态）**。
> **二者叠加仍然够不到远端针。**

**按"距离"重读 §41.4**（当前 harness 是按 depth＝上下文比例分桶，所以 depth 同时编码了距离）：

| 距离 | 有递推 | 无递推 | 谁在支撑 |
|---|---|---|---|
| ~800 tok（depth 0.95 @16k） | gain **1.10** | gain **1.02** | **GDN 顶住** |
| ~15k tok（depth 0.05 @16k） | gain **+0.28** | gain **+0.06** | **只有递推** |

边界落在 800–15k 之间 ⇒ 与「**GDN 的精确 recall 视界由状态容量决定（~O(1k)）**，而 DISM 的
寻址位置无关（无界）」一致。

**三分工假设（可一次性测死）**

| 分支 | 覆盖范围 | 机制 |
|---|---|---|
| SWA（仅 `hybrid` 配置） | 严格局部（窗口 128） | 硬窗口 |
| GDN（含 short conv） | 局部 + **中程 ~O(1k)** | 固定大小状态、容量受限、有干涉 |
| **DISM（带递推）** | **无界精确** | 位置无关的离散变阶寻址（`V_eff^n`） |

**待做实验：距离扫描 × 分支消融。** 现在的 NIAH 按 depth 分桶**把距离与绝对位置混在一起**；
改成**固定上下文、把针放在距 query 精确 D 处**：

```
D ∈ {64, 128, 256, 512, 1k, 2k, 4k, 8k, 15k}  ×  {full, recurrence-off, GDN-off, SWA-off}
```

得到一张 **距离 × 分支贡献矩阵**，预期（可证伪）：

- **SWA-off** 只在 D > 128 掉；
- **GDN-off** 从 D ≳ 1k 开始掉；
- **recurrence-off** 只在 D ≳ 4k 掉，且 **D 越大掉得越多**；
- 三条曲线在**距离轴上分层** —— 这就是"分工"的直接证据。

实现：给 `diagnostics/niah.py` 加 `--distance D`（把针插在距 query 精确 D 处，而不是按比例），
既有的 `--depths` 保留。**注意当前所有 NIAH 数字都是 depth 版本，不能当距离用。**

**可正面写进论文的一句**：

> 无论**门控线性记忆（GDN）**还是**固定 slot 的离散分桶记忆（MoM）**，单独或**联合**都够不到
> 远端精确 recall；只有带后缀结构的**变阶寻址**做到了。

这把"指数 slot"的主张从"我们比 MoM 好"升级为"**两类已知的固定状态记忆叠起来也不够**"。

## 43. 硬反向的两条派发路径梯度不一致（2026-10-15，已实测）

`_voc_forward_impl` 的前向只有一份实现（都用 `forward_hard`），但反向按 ticket 的
`route` 分岔成两套**独立内核**：

- `event`：planner 编译出的 `Program`，由 `CompactFusedHardPrefillBackward` /
  `FusedHardPrefillBackward` 消费；
- `quadratic`（回落）：`backward_hard_recompute` 重算 `forward_hard(save_state=True)`
  后调 `cu.hard_backward`。

现有测试把两条路径**各自**钉在一个 FP64 oracle 上，容差是
`gradient_acceptance` 的 `VECTOR`（cosine≥0.999、rel_l2≤3%）与 `TAU`
（cosine≥0.95、rel_l2≤50%），**没有任何测试把两条路径互相比较**。新增的
`dism_v3/tests/test_hard_route_consistency.py` 做了这件事。

**结论：偏差只出现在 τ 梯度上，而且错的是 event 路径。**

同一批输入、只改 planner 的 `backward_policy`，τ 梯度对 FP64 oracle 的相对 L2：

| 输入形态 | event | quadratic |
| --- | --- | --- |
| 随机符号，8 组 | 8e-6 … 9e-5 | 5e-6 … 5e-5 |
| 所有符号相同（最大匹配） | **2.6e-3 … 6.2e-2** | 2.5e-6 … 7.5e-4 |

`sq/sk/v` 两条路径在所有形态下互差 ≤6.5e-4（bf16 舍入水平），所以是 τ 独有的。

**触发条件是每条流跨多少个 planner chunk**（`dism_v3/tools/route_tau_sweep.py`
扫出来的，τ 梯度 event-vs-quadratic 相对 L2）：

| run/chunk | 误差 |
| --- | --- |
| 0.2 / 0.5 / 1.0 | 2e-7 … 7e-7（精确） |
| 2.0 | 2.8e-6 … 8.4e-4 |
| 4.0 | 3.8e-4 … 1.5e-2 |
| 8.0 / 16 / 32 | 4.4e-3 … 1.6e-1 |

同一个 run 长度在不同 `chunk_size` 下按比例复现（block=256 在 chunk 16/32/64 下都坏；
block=64 在 chunk 64 下等于 1 chunk，精确），所以故障在**多 chunk 流的导数路径**——
即 `prefill.hpp` 的 `logweights/derivatives` 表（存的是 `d log F/dτ`，不是 `dF/dτ`）、
`dlogden` 的跨流反向累积（`prefill.hpp:412`），以及消费端 `_query_backward` /
`_key_backward` + summary scan 的导数分量。

生产统计量 `single_blocks 56518 / multi_streams 2547 / multi_chunks 6925`（平均
2.7 chunk）说明约 4% 的流命中该路径，这就是为什么所有旧测试（标签随机、run 很短、
全是单 chunk）一直是绿的。

**影响与不是影响：**

- τ 梯度天然由长 run 主导（长度 s 的 run 对 dτ 的贡献约为其对权重贡献的 s 倍），
  而长 run 恰好就是多 chunk 的那些，所以偏差落在 dτ 权重最大的部分。
- 它**不能**解释 r4 里 L10/L11 的码表集中度：回落路径服务的是 L10/L11（恒回落），
  而那正是 τ 梯度**准确**的层；走 event 的 L06–L09 反而 coverage 更好。
- r4 不需要因此中断：偏差量级小、且切到 `DISM_HARD_BACKWARD=quadratic` 要付
  631 vs 600 ms/step 的代价。先离线修。

## 44. 码表集中度不是回归：旧基线同样塌，是探针掩盖了（2026-10-15）

起因是怀疑固定 direction=true 后替换的插值/LSE 实现有问题。做了三件事：

**一、提交审计（只看能动到码表几何 / soft 分数 / 温度 / 初始化的改动）。**

| 检查项 | 结论 |
| --- | --- |
| `dism_v3_ref.py::dism_ref` | v3 树里**从未被修改**（只有 `dism_wrapper` 在 `238fe06` 加了 β 参数），所以所有 oracle 对比是独立的 |
| 词表温度 β（`238fe06`, `7b18d14`） | valres 全程 β=1；`dism_wrapper` 与 `interp_embeddings` 的 `β·raw − LSE_β` 缩放自洽 |
| 固定 true 特化（`9fe61b8`, `7a3a25c`） | 与两条目路径**逐位等价**：标签零差异、`q_from_k` 位相等、LSE 差 7.2e-7、输出 bf16 量级 |
| 原子插值反向（`3fb6269`） | 文档标注 nondeterministic；每码表条目每步只累加约 64 次，FP32 原子序偏差 ~1e-6 |
| DISM/GDN 共享 q/k 线性（`e0afd36`） | tie 早于该提交，它只是避免把同一个确定性 `F.linear` 算两遍 |

未发现能解释集中度的实现错误。守卫强度也够：`test_interp_bwd_split.py` 有 1e-4 相互对比与逐位断言，码表梯度对 FP64 oracle 是 3%。

**二、量到 soft 路径 τ 梯度的精度上限**（`tools/check_tau_gradient.py`）：向量梯度在 soft/hard/mixed 三种模式下一致为 2.35e-3（bf16 底），而 τ 梯度在全 soft 行上是 1.0e-2（max 2.1e-2），全 hard 行是 6.3e-5。随 q/k 缩放（soft logit 幅度）1→3→6，soft 的 τ 误差 5.1e-3→8.9e-3→8.3e-2，而 hard 恒为 4.3e-5。方向保持（cosine ≥0.994），所以是精度/条件化而非逻辑错误。注意 `gradient_acceptance.TAU` 的门是 rel_l2 50%、cosine 0.95，这个量级本来也测不出来。

**三、决定性对照：旧基线自己也塌了。**

`tools/probe_old_valres_coverage.py` 用**修好的**探针测原始 `dism-fwedu-valres-72m-2500m` 的最终 checkpoint（step 19074，fused conv、旧目标函数、早于本会话所有改动）：

```
layer  q_cov  q_pool  q_maxshare  q_entropy | k_cov  k_pool  k_maxshare
    6  0.406   0.971     0.207      0.622   | 0.451   0.977     0.469
    7  0.386   0.959     0.208      0.648   | 0.412   0.963     0.517
    8  0.337   0.932     0.249      0.569   | 0.403   0.973     0.462
    9  0.240   0.820     0.278      0.544   | 0.306   0.887     0.498
   10  0.134   0.574     0.466      0.343   | 0.117   0.525     0.597
   11  0.236   0.799     0.454      0.323   | 0.202   0.750     0.448
```

而同一组权重当年被旧探针报成 `0.98 / 0.97 / 0.92 / 0.75 / 0.53 / 0.80`（`qk/q_coverage` 的历史曲线）。**旧探针把 L06 从 0.41 报成 0.98、L10 从 0.134 报成 0.53**，所以旧基线的"平稳高覆盖"是探针假象，不是它没塌。

当前 r4（step ≈8000，42% 训练）的 per-head：q = 0.579/0.627/0.366/0.223/0.256/0.247，k = 0.566/0.600/0.374/0.269/0.192/0.186。与旧基线最终值同量级，早层甚至更高。

**结论与推论：**

1. 集中度**不是回归**，也没有实现错误解释它——旧基线与当前代码同样集中，层级模式（L10/L11 最差）也一致。要归因必须先修探针，再谈配置差异。
2. 因此这是**方法/配置本身不够 robust**，防坍缩需要专门研究，而不是继续找 bug。
3. **跨探针修复的覆盖率比较全部作废**：任何把旧 run 的 `qk/*_coverage*` 与修复后的数字并列的结论都不可信，需要重新测。
4. 对防坍缩设计的一个直接输入：**k 侧与 q 侧塌得一样重**（旧基线 L10：q 0.134 / k 0.117，max_share 0.466 / 0.597）。所以"每个 kv head 分裂成 G 个 q 通道、k 侧不分裂"这类方案只保护 q 侧，k 侧仍然暴露。

## 45. Norm 位置与 selector 梯度的信噪比（2026-10-15）

假设（来自 forked session）：架构搜索里 postnorm 优于 prenorm，而 postnorm 的梯度按
O(2^{-L/2}) 随深度衰减；会不会这两件事其实是通过 selector 的 β1 耦合起来的——
即 postnorm 的优势只是 β1/噪声设定的副产品？

**先掐掉朴素版本。** Adam 的每参数更新是 `m̂/(√v̂+ε)`；对某个参数把梯度乘一个常数 c，
则 m 与 √v̂ 同比例变化，更新不变。所以"某层梯度整体变小"这件事 Adam **看不见**。
要产生真实的耦合，必须是衰减**改变了参数间的尺度构成**或**信噪比**，而不是纯尺度。

**测量**（`tools/check_norm_gradient_profile.py`）：`post_norm` 是 `DismBlock` 每次
forward 读取的属性（`modeling_dism.py:58`），所以可以直接在**同一组权重**上翻转它，
把"布线"与"训练出来的参数"分开。取 valres 最终 checkpoint（19074），固定一个
4×2048 的 packed batch，按 `rng_counter` 偏移抽 16 组独立 hard flag，对每个 selector
张量算 `μ`、`Σ`，报 `r = tr Σ/‖μ‖²`（Adam 实际看到的是 `‖m̂‖/√v̂ ≈ 1/√r`）。

| layer | r post@h=.95 | r pre@h=.95 | 比 | r post@h=.5 | r pre@h=.5 | ‖μ‖ post | ‖μ‖ pre | μ 比 | ‖g‖ post | ‖g‖ pre |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 10.36 | 5.45 | 1.90 | 0.88 | 0.56 | 3.83e-3 | 2.66e-5 | **144×** | 11.6 | 56.3 |
| 7 | 6.76 | 4.63 | 1.46 | 0.76 | 0.19 | 5.95e-3 | 1.29e-4 | 46× | 13.7 | 49.5 |
| 8 | 12.83 | 2.38 | 5.38 | 0.85 | 0.37 | 5.72e-3 | 6.53e-5 | 88× | 14.5 | 43.0 |
| 9 | 10.53 | 9.23 | 1.14 | 1.01 | 0.82 | 5.67e-3 | 2.30e-4 | 25× | 18.1 | 42.1 |
| 10 | 8.90 | 7.80 | 1.14 | 1.07 | 0.77 | 3.46e-2 | 3.49e-3 | 10× | 26.5 | 68.9 |
| 11 | 7.61 | 10.80 | **0.70** | 1.20 | 0.87 | 1.86e-2 | 4.53e-4 | 41× | 24.3 | 60.0 |

**三条结论：**

1. **β1 耦合假设不成立。** Adam 唯一能看到的统计量 r 在两种 norm 之间只差
   1.1–5.4×，而且方向不一致（layer 11 反而 prenorm 更小）。相反，selector 梯度的
   *幅度* 差 10–144×、全体参数梯度差 3–4×——两者都被 Adam 的逐参数归一化约掉。
   所以 postnorm 的优势**不可能**是通过 selector 的 beta1 传过来的。
2. **β1 那份分析拿到了新数字。** r 强烈依赖 hard_prob：h=0.5 时 r≈0.76–1.20，
   h=0.95（训练终值）时 r≈6.8–12.8，**噪声高约 10×**，对应 SNR ≈ 1/3。这独立确认了
   "gated selector 的梯度更噪、需要更长 momentum" 这条，`selector_beta1=0.95` 的实测
   增益也因此有了量级依据。
3. **更可能的解释是经典的 pre/post-LN 调参混淆。** post-LN 的逐层 update 幅度按
   O(1/√L) 衰减而 pre-LN 均匀，所以在**单一固定 LR/warmup** 下两种 norm 需要不同的
   超参——架构搜索若只按一套超参比，结论就是被调参决定的。这属于"需要重调"的混淆，
   而不是 β1 的混淆。

**限制与下一步：** 本测量在**同一组 postnorm 训练的权重**上比较布线，回答的是"布线如何
改变梯度统计"，不能替代真正训练一个 prenorm 模型（后者权重不同、r 可能不同）。而且
**本仓库从未跑过 prenorm**（没有任何 `post_norm: false` 的 config 或 run），所以那个
"postnorm 更好"的结论在本树里没被复现过。要判定它是不是调参伪影，最省的是
norm × LR 的小规模交互实验（例如 2000–4000 步、LR ∈ {2/3×, 1×, 1.5×}），
以及 norm × selector_beta1 作为对照。

## 46. 待办（2026-10-15 记录，未执行）

按"需要什么才能判定"排序，不表示优先级。

1. **norm × LR 交互实验（§45 的结论待判）。** 本仓库从未跑过 prenorm（没有任何
   `post_norm: false` 的 config 或 run），而"postnorm 更好"是从别处的架构搜索带过来的。
   §45 已经排除 β1 作为通道（Adam 逐参数尺度不变，且 Adam 可见的 `r` 在两 norm 间只差
   1.1–5.4×、方向还不一致），所以最可能的是经典调参混淆：post-LN 逐层 update 按
   O(1/√L) 衰减、pre-LN 均匀，单一 LR/warmup 不可能同时最优。**待办**：短程
   （2000–4000 步）norm × LR(2/3×, 1×, 1.5×) 交互，外加 norm × `selector_beta1` 作对照。
   判定标准：若 prenorm 在某 LR 下追平/反超，则原结论是调参伪影。
2. **event 路由多 chunk τ 梯度缺陷（§43）。** 已定位到"流跨过 1 个 planner chunk 就开始偏"，
   并已用 4 个 strict xfail 钉住。**待办**：修好后重测——若修好导致训练结果改变，按既定
   判断那就是方法本身不够 robust，需要专门解决稳定性，而不是回头怪缺陷。
   当前实验全部固定在 n² 路径（`DISM_HARD_PLANNER=0`）绕开它。
3. **2×2（direction × layout）覆盖率重测。** §44 证明旧探针把覆盖率高报 2–2.4×（同一组权重：
   L06 报 0.98 实为 0.41，L10 报 0.53 实为 0.134），因此**任何跨越探针修复的覆盖率比较作废**，
   包括文档里"healthy vs collapsed"那些格子。**待办**：在同一(修好的)探针下重测四种组合。
4. **GQA G=2（query 侧多通道，词表不新增）。** 实现已定稿（追加一条查询通道、不改动任何
   现有参数化，避开 q/k tie 的混淆），运行目录 `…-tokenbal-gqa2-20261015` 已备，对照是
   `…-tokenbal-quadforce-20261015`（同为 n² 路径）。预期参数量 73,372,200。
   **待办**：实现落地并跑 §"验证"五项（G=1 逐位一致 / G=2 参数量 / 通道独立 / 混合契约 / 回归测试）。
   注意 §44 的第 4 条：k 侧与 q 侧塌得一样重，本设计只保护 q 侧。
5. **监控口径固化。** 覆盖率只读逐 head 的 `qk/{q,k}_coverage/layerNN`；`coverage_pooled`
   在 `vocab_share_heads=false` 时是把 6 张独立码表取并集，必然虚高，不能用来判断塌缩。

## 47. NIAH circuit：层级依赖重分布与 L10 head 3（2026-10-16）

按 `NIAH_CIRCUIT_PROBE.md` 开始第一轮分析。完成 P1（单层必要性）、P2/P3
（真实内核输入的查询行和符号链），并针对 L8/L10 补做逐 slot 消融。
没有修改生产 forward、kernel、checkpoint 或训练进程。代码基线 `4259eb6`；
每份实验 JSON 记录实际探针源码 SHA256，新增脚本当时尚未提交。

### 47.1 输入、指标和 hook 的实际口径

- 三臂均读取各自 step 19074 的 `model_spec`，保留 old 的 fused conv 和另两臂的
  linear_fla。固定 hard、direction=true，保留 value residual 和完整 GDN 分支。
- 原 `niah.py` **只在背景插入数字尾部，不插入完整提示句**。实际 tokenizer 将尾部
  编为「空白 token + 五个数字 + 句号」共 7 tokens；答案预算 6，因而评分包含空白和
  五个数字，末尾句号被截掉。查询与答案之间没有 EOS 填充。P1 保留这个任务定义，
  P2/P3 的表格只聚合五个数字预测，排除空白目标；原始 JSON 包含全部六个查询行。
- 原 harness 的 present/absent 连续消耗 RNG，不是严格成对背景。此次新建固定 suite：
  每个 `(seed=1234, context, depth)` 独立确定随机序列，present/absent 只在 needle span
  不同；各臂、各消融完全复用同一 token 序列。每格 16 trials；8k/16k × depth .05/.95。
  **这是新样本集，不能与历史表格逐格相减。** 全部 64 对输入已检查长度、目标及差异范围。
- P1 在 `_combine_cuda` 输入处置零 DISM core，原样转发 `conv_seq_idx` 等参数，保留
  GDN 和共同的 norm/gate/o_proj。逐头干预在 `compiler.voc_forward` 输出处、GQA mix
  之前置零一个 slot；slot `2h` 是 base，`2h+1` 是新增 query。所有 hook 均核验调用次数。
- 固定 true 分支实际走 `compiler.voc_forward → emb_fwd_true`，不是普通
  `flash_dism.voc_dism → interp_embeddings`。P2/P3 截取真实 `iq/ik/sq/sk/v/tau`，
  以及 mix 前的 core 输出，不重新执行 q_conv/k_conv 来近似输入。
- 每格保存逐 trial、逐 token NLL/hit。置信区间为 **paired trial bootstrap，10000 次，
  seed 1729，pointwise 95%**；没有做多重比较校正，属于探索性定位。单头效应不能相加。

### 47.2 P1：主要依赖层不是统一衰弱

下面每个三元组为 `present NLL / absent NLL / present token accuracy`；最后一项为百分比。
原始：`dism_v3/build/niah_circuit_p1_20261016/{old,restored,gqa2}.json`；
配对区间：同目录 `summary.json`；固定输入：同目录 `inputs.json`。

| ctx / depth | old | restored | gqa2 |
|---|---|---|---|
| 8192 / .05 | 2.5061 / 3.0810 / 13.54 | 2.8353 / 3.3730 / 16.67 | 2.4309 / 3.2116 / 32.29 |
| 8192 / .95 | 1.3663 / 3.0376 / 65.63 | 1.4454 / 3.4690 / 62.50 | 1.7074 / 3.3036 / 57.29 |
| 16384 / .05 | 2.6977 / 3.0968 / 12.50 | 3.0416 / 3.4710 / 12.50 | 2.6395 / 3.2784 / 16.67 |
| 16384 / .95 | 2.0163 / 3.0543 / 37.50 | 2.1162 / 3.4706 / 38.54 | 2.3566 / 3.1696 / 26.04 |

16k 远端 restored−old present NLL = **+0.3439 [0.2217, 0.4714]**；
gqa2−old = **−0.0582 [−0.2728, 0.1543]**。因此复现了 restored 的远端劣势，
但本 suite 没有确立远端 old/gqa2 排序，也没有证明历史三个 depth 平均的完整排序。
不能把 across-arm `absent−present` 当成同一量尺。

移除对应层 DISM core 后的 **present NLL 增量**（16k / depth .05）：

| 模型 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---:|---:|---:|---:|---:|---:|
| old | −0.0061 | +0.0104 | −0.0322 | −0.0068 | **+0.3127** | +0.0970 |
| restored | +0.0020 | +0.0491 | **+0.2038** | −0.0035 | **+0.1813** | +0.0949 |
| gqa2 | +0.0427 | +0.2306 | **+0.5395** | +0.0152 | +0.0783 | +0.0172 |

restored−old 的消融增量差：L10 = **−0.1315 [−0.2490, −0.0224]**，
L8 = **+0.2359 [0.1120, 0.3657]**。8k 远端同样呈 L8 从 old 的 −0.0561
变为 restored 的 +0.3142、gqa2 的 +0.6553。支持**依赖层重分布**，不支持
「所有层均匀变弱」；L10 的必要性降低也不能单独解释全部性能差距。

完整四格图：`dism_v3/build/niah_circuit_p1_20261016/layer_necessity.png`。

### 47.3 P2/P3：old L10 head 3 的读路径，在 restored 同索引头不再出现

全 hard 下，长度为 `l` 的匹配后缀有 `exp(score)=sum(exp(k*tau), k=1..l)`。
因此仅沿请求行的匹配对角线计数，即可用 O(N) 存储重建归一化读权重，包含 fallback；
再乘真实 `sq·sk` 和 `v` 得到各 key 对 core 的向量贡献。没有分配 N×N 矩阵。
各 target 的 query 位置均为其前一 token，不能把六个预测视为同一个 query。

以下为 **16k / .05、16 trials × 五个数字 = 80 个 digit queries** 的均值。
「源数字」指 needle 中与当前目标同一 offset 的数字位置，不假设它是唯一有效 key；
另报整个七 token needle span 的 mass。chain 长度在 gate 关闭时计零。

| 模型 / 层 / slot | τ | 源数字 gate 开启率 | 源数字 chain | 源数字归一化 mass | 全 needle mass |
|---|---:|---:|---:|---:|---:|
| old L10 h0 | 1.39210 | 73.75% | 1.5375 | 0.13674 | 0.23047 |
| old L10 h3 | 1.16930 | **87.50%** | **2.2875** | **0.06506** | 0.07062 |
| restored L10 h0 | 1.60855 | **85.00%** | **2.1750** | **0.21099** | 0.31556 |
| restored L10 h3 | ≈0.00042 | **6.25%** | **0.0750** | ≈0.0002 | ≈0.0005 |
| restored L8 h5 | 2.94709 | 48.75% | 0.9375 | 0.10925 | 0.11386 |
| gqa2 L8 slot1 = h0/新增 | 1.36193 | 92.50% | 2.4250 | 0.20157 | 0.26665 |
| gqa2 L8 slot5 = h2/新增 | 1.42110 | 86.25% | 2.3625 | 0.04006 | 0.04145 |

old L10 h3 在源数字位置的 signed readout 平均 **+0.67637**，负值比例 **3.75%**；
restored 同索引头平均 **−0.04556**。但 gate、chain、τ、readout 同时变化，
不能据此单独归因于 readout cancellation 或仅归因于 τ。

逐头消融进一步给出 16k 远端 **present NLL 增量 [95% CI]**：

| 模型 / 层 / slot | Δ present NLL |
|---|---:|
| old L10 h0 | +0.0878 [0.0316, 0.1429] |
| **old L10 h3** | **+0.2458 [0.1599, 0.3323]** |
| restored L10 h0 | +0.2119 [0.1294, 0.3222] |
| restored L10 h3 | −0.0312 [−0.0430, −0.0203] |
| restored L8 h5 | +0.1396 [0.0241, 0.2598] |
| gqa2 L8 slot5 = h2/新增 | +0.3859 [0.2210, 0.5578] |

8k 远端 old L10 h3 同样必要：**+0.2260 [0.1207, 0.3357]**。
这建立了一个具体的工作假说：**old 在 L10 h3 用符号后缀匹配读取针中的数字，
其 core 对答案有实测贡献；restored 同索引头的匹配、链增强及读出特征没有保留，
但 L10 h0 和 L8 h5 仍承担读取。** 这比「restored 完全丢失针」更符合观测。
不同训练模型的相同 head 索引不是已经证明的语义对应，故不能把这一观察当成
训练过程中的同一 circuit 消亡证据，也不能把整头消融等同于仅移除针贡献。

gqa2 的额外 query 通道确实承载有用读取，而不是只扩大无效 slot 数量。
注意以上 mass 与 core norm 都是 **mix 前**；例如 L8 slot1 和 slot5 的 learned mix
分别为 **0.00749535** 和 **0.31577924**。不能仅按 pre-mix mass/norm 排序最终贡献，
也不能把不同模型不同 head 的向量范数直接当成质量指标。

原始查询行（所有 L6–11、全部 head/slot、全部目标位置）：

- `dism_v3/build/niah_circuit_p23_20261016/{old,restored,gqa2}.json`：trials 0–3。
- `dism_v3/build/niah_circuit_p23_remaining_20261016/{old,restored,gqa2}.json`：trials 4–15。
- 聚合：`dism_v3/build/niah_circuit_p23_summary_20261016.json`。
- 逐头消融：`dism_v3/build/niah_circuit_heads_20261016/{old,restored,gqa2}.json`，
  包含 8k/16k 远端 L8/L10 所有 slot 的逐 trial present 分数和 paired CI。
- GQA mixing 权重：`dism_v3/build/niah_circuit_gqa_mix_20261016.json`。

### 47.4 验证、资源与可复现入口

- P2/P3 的 baseline NLL 与 P1 固定输入的结果 **全部相等（最大漂移 0）**。
  查询行 CPU FP64 重建与实际 BF16 core 输出的 relative L2（median / p95 / max）：
  old **0.153% / 0.284% / 0.894%**，restored **0.155% / 0.274% / 0.766%**，
  gqa2 **0.151% / 0.281% / 1.319%**。保留这一数值限制，不称逐位重建。
- CPU 回归测试覆盖：单层消融保留 GDN/kwargs 且异常时恢复；单 slot 干预不影响其他
  层或 slot、不修改原输出；低内存查询行与完整 FP64 oracle 对照（长匹配、无匹配、
  τ=0/近零/正/负）。三项通过。没有修改 CUDA kernel；本轮未运行 sanitizer。
- P1 最大 allocated 显存 old/restored/gqa2 = **1.44 / 1.48 / 2.54 GiB**，chunk=1；
  运行前检查共享 RTX 5090 余量，allocator 上限 30%。所有 probe 完成时检查 buffers
  未变。缓存放外盘 `cicuvc/eval-cache/niah-{triton,inductor}`，结果在 gitignored build。
- 探针入口：`tools/niah_circuit_probe.py`、`tools/niah_read_rows.py`、
  `tools/niah_head_ablation.py`；汇总入口 `tools/summarize_niah_circuit.py`、
  `tools/summarize_niah_rows.py`。路径相对 `dism_v3/`。输出目录必须新建，避免覆盖结果。

```bash
# 在工作树根目录；PYTHONPATH 只指向本树
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export FLA_FLASH_KDA=0 DISM_HARD_PLANNER=0 CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR 指向外盘可写缓存目录
PY=/home/cicuvc/miniconda3/envs/blkw/bin/python
$PY dism_v3/tools/niah_circuit_probe.py --out dism_v3/build/NEW_p1
$PY dism_v3/tools/niah_read_rows.py \
  --inputs dism_v3/build/NEW_p1/inputs.json --out dism_v3/build/NEW_rows --trials 16
$PY dism_v3/tools/niah_head_ablation.py \
  --p1 dism_v3/build/NEW_p1 --out dism_v3/build/NEW_heads
$PY dism_v3/tools/summarize_niah_circuit.py dism_v3/build/NEW_p1
$PY dism_v3/tools/summarize_niah_rows.py --p1 dism_v3/build/NEW_p1 \
  --directories dism_v3/build/NEW_rows --out dism_v3/build/NEW_rows/summary.json
$PY dism_v3/tests/test_niah_circuit_probe.py
```

**尚未解决：** 尚无跨模型 circuit rescue，未分离 symbol、τ、readout 的独立因果效应；
P4 的 value 可解码性和 P5 的 residual/loss 方向尚未测。下一步优先对已定位的
old L10 h3 / restored L10 h0、L8 h5 做针贡献定向干预，再决定是否训练 value probe。
old/restored 同时有 conv frontend 与目标函数差异，不能把 circuit 差异直接归因于
loss normalization；还需完成后的 legacy/kvg2 对照。本轮没有重做全层 zero/flip/norec
等已确立实验，也没有将仅四格的结果推广为全部 NIAH 条件结论。

### 47.5 逐头 absent 对照：L10 h3 的差异不只是一般数字预测

对 16k/.05 的 L8/L10 slots `{0,1,3,5,9}`（超出该模型 slot 数的跳过）补做
相同 16 trials 的 absent 干预。每个模型先复算一个 baseline，均与 P1 相等；
所有模型 buffers 检查通过。原始：
`dism_v3/build/niah_circuit_heads_absent_20261016/{old,restored,gqa2}.json`；
汇总：`dism_v3/build/niah_circuit_head_contrasts_20261016.json`。

这里比较**同一模型内** `Δpresent − Δabsent` 的 paired 干预交互，
不是把不同模型的 baseline `absent−present` 当成可比较的 recall 分数。

| 模型 / 层 / slot | Δpresent | Δabsent | 差值及 paired 95% CI |
|---|---:|---:|---:|
| old L10 h0 | +0.0878 | +0.0483 | +0.0395 [−0.0074, 0.0823] |
| **old L10 h3** | **+0.2458** | **+0.0134** | **+0.2324 [0.1534, 0.3104]** |
| restored L10 h0 | +0.2119 | +0.0713 | +0.1406 [0.0516, 0.2258] |
| restored L10 h3 | −0.0312 | −0.0292 | −0.0020 [−0.0110, 0.0055] |
| restored L8 h5 | +0.1396 | −0.0680 | +0.2076 [0.0906, 0.3296] |
| gqa2 L8 slot1（新增） | +0.1402 | −0.0205 | +0.1606 [0.1046, 0.2219] |
| gqa2 L8 slot5（新增） | +0.3859 | −0.0322 | +0.4182 [0.2487, 0.5940] |

因此 old L10 h3 的必要性确实主要出现在针存在时；restored 同索引头的微小负效应
在 present/absent 都出现，不能描述为「有害的针读取」。这与 §47.3 中其源数字 gate、
chain 和 mass 都低的观测一致。restored 的 L10 h0、L8 h5 和 gqa2 新增查询通道仍有
明显的针条件效应。当前结果支持**读取路径重分布，old 的一条有用 L10 h3 路径在
restored 同索引头缺失**；尚未证明这是 restored 总体 NLL 劣势的充分原因。

复现该补充对照：

```bash
$PY dism_v3/tools/niah_head_ablation.py --p1 dism_v3/build/NEW_p1 \
  --out dism_v3/build/NEW_heads_absent --condition absent \
  --contexts 16384 --slots 0 1 3 5 9
```

## 48. old L10h3 的局部符号流：错一 token 的后缀续写（2026-10-16）

按用户要求直接查看 old，不做新的跨模型比较。复用 §47 的 16k/.05 trials 0–3，
hook 实际 `emb_fwd_true` 的 q/k labels，保存针前后及查询附近的逐 token 窗口。
脚本 `tools/niah_symbol_windows.py`；原始数据
`dism_v3/build/niah_symbol_windows_old_l10h3_20261016.json`。
四条样本的 NLL、各源数字 chain 和归一化 mass 均与已存 P2/P3 相等；buffers 未变。

trial 0 的针是 `88897`。`▁` 为独立空白 token，`↵` 表示换行。
下面所有数字都是该 head 的离散码表 ID，不是词表 token ID。

```text
位置        817  818  819  820  821  822  823  824  825
token        ↵    ▁    8    8    8    9    7    .  with
q           241  353   82  422  422  422  390  101  130
k           423  241  353   82  422  422  422  390  101

位置      16373 16374 16375 16376 16377 16378 16379 16380 16381 16382 16383
token       The special magic number    is     ▁     8     8     8     9     7
q           469   239   184    69   106   353    82   422   422   422   422
k           455   469   239   184   196   106   353    82   422   422   422
```

局部可见 `k[j] ≈ q[j−1]`，尤其整个针的空白至句号区间严格成立。
这不是同位置 q=k，而是 key 更像对前一个 token 的状态作索引。预测下一位时：

```text
目标数字                 8    8    8    9    7
查询前一 token 的 q     353   82  422  422  422
针内目标位置的 k        353   82  422  422  422
匹配后缀长度              1    2    3    4    5
目标源位置归一化 mass  0.39% 3.52% 6.32% 13.74% 33.00%
```

这是 **teacher-forced** 评估：后续查询已经看到真实数字前缀，不是模型自由生成了
`88897` 的证据。预测最前面的空白时，该 head 在整个 needle span 上 mass=0；
预测首位数字的 mass 也很小，随着正确前缀增长才强化。故此例更接近**前缀匹配后续写**，
不应解释为仅凭 `The special magic number is` 就精确取出整个五位数。

符号不是数字的一一编码：trial 0 `88897` 与 trial 2 `95257` 在针内五个数字的
k 流均为 `[353,82,422,422,422]`，查询数字预测 q 流也相同。
它提供的是有碰撞的粗粒度路由/后缀特征；具体数字内容需要从 value/readout 通路读出，
仅凭此观察不能确定完整 value 解码机制。

也存在断链：trial 3 `96277` 预测第二位 `6` 时，查询 q=82，源数字 k=390，
该位置 mass=0，后缀长度为 0。五位数字对应 chain 为 `[1,0,1,2,3]`，
最后一位源数字 mass 仅 4.27%。因此上面的连续增长是一个实测例子，不是所有针均成立。

## 49. 三 checkpoint 的有害匹配筛查：短链背景竞争（2026-10-16）

### 49.1 范围与干预定义

复用 §47 同一输入 hash 的 **16k、depth=.05、16 trials**，只评估五位数字的
teacher-forced token NLL。统计覆盖 L6–L11 全部 query slots；因果干预限定为此前
逐头实验定位出的六条路径：old L10 h0/h3、restored L8 h5/L10 h0、gqa2 L8
slot1/slot5。GQA 的两个 slot 都是新增 query 通道，在 mix 之前干预。
这不是全部 heads 的因果穷举，也不是对齐同索引头的跨模型语义比较。

每次只改一个 head 的一个 query 输出，随后重算下游，仅计该 query 对应数字的
NLL。五个数字先在 trial 内平均，再对 16 trials 做 paired bootstrap（10,000 次）；
CI 为 pointwise 95%，未做多重比较校正。后 8 trials 另报一致性检查，但这些输入
此前已用于 §47，不称为全新盲测。不同位置、不同 heads 的收益不能直接相加，也
不能作为整模型同时启用规则后的生成成绩。

设 needle 完整七 token span 为 S。候选背景 D 排除针及前后各 32 tokens，也排除
query 最近 256 tokens。背景只是候选干扰，不能先验断言其没有来源信息。
实际离散标签从 `emb_fwd_true` 捕获，权重为
`w(l)=sum(exp(k*tau), k=1..l)`，`a_j=w_j/(1+sum(w))`，单位 fallback 保留。
定义 `u_j=(sq_i·sk_j)v_j`、`o=sum(a_j*u_j)`、`A_D=sum_D(a_j)`、
`U_D=sum_D(a_j*u_j)`，比较：

| 干预 | 理想输出 | 含义 |
|---|---|---|
| content | `o-U_D` | 只去掉候选背景的内容贡献 |
| denominator | `o/(1-A_D)` | 只去掉其分母贡献，所有内容仍保留 |
| full | `(o-U_D)/(1-A_D)` | 同时删除内容与分母贡献 |
| half | `(o-.5*U_D)/(1-.5*A_D)` | 背景权重减半，检查较小剂量 |
| source | 删除 S 后重新归一化 | 正确来源必要性对照 |
| random0/1 | 随机删边并重新归一化 | 每个 log2 距离 bin 删除相同质量 |

随机对照使用两个固定种子，允许末条边被部分删除；匹配的是质量和距离 bin，
没有同时匹配边数、链长。其采样池包含针，因此可能删掉有效来源。
实际实现用稳定 log 权重重算 retained 分布，避免 `1-A_D` 消减和旧分布下溢。
FP64 计算输出差量，叠加在真实 CUDA BF16 core 输出上，保留 baseline 数值残差；
sham 是严格零差量。这里的指标统一为 **收益 B=baseline NLL−干预 NLL，正数为改善**，
与 §47 ablation 的 ΔNLL 符号相反。

### 49.2 六条路径均存在可干预的背景竞争

| 模型 / 路径 | 针质量 | 候选背景质量 | full 收益 [95% CI] | 相对随机删除的额外收益 [95% CI] | source 删除收益 |
|---|---:|---:|---:|---:|---:|
| old L10 h0 | 23.05% | 70.74% | +0.168 [0.106, 0.231] | +0.115 [0.059, 0.171] | −0.099 |
| **old L10 h3** | **7.06%** | **90.54%** | **+0.751 [0.578, 0.925]** | **+0.776 [0.583, 0.972]** | **−0.268** |
| restored L8 h5 | 11.39% | 83.85% | +0.246 [0.180, 0.316] | +0.172 [0.108, 0.241] | −0.156 |
| restored L10 h0 | 31.56% | 63.77% | +0.253 [0.178, 0.334] | +0.197 [0.128, 0.278] | −0.181 |
| gqa2 L8 slot1 | 26.66% | 70.42% | +0.160 [0.121, 0.206] | +0.137 [0.089, 0.190] | −0.152 |
| **gqa2 L8 slot5** | **4.14%** | **93.76%** | **+0.765 [0.605, 0.914]** | **+1.094 [0.854, 1.311]** | **−0.461** |

六条路径的 source 删除均有负收益，CI 全在零以下；full、half 和 full−random
均为正，CI 全在零以上。后 8 trials 的 full 和 full−random CI 仍全部大于零。
old L10 h3 / gqa2 slot5 的 full 数字准确率分别提高 38.75 / 33.75 个百分点；
这仍是逐位置单独干预下的 teacher-forced 准确率，不是五位数自由生成 exact match。

分解显示不同路径的失效方式并不一致：

| 路径 | content 收益 | denominator 收益 | half 收益 |
|---|---:|---:|---:|
| old L10 h0 | +0.039 | −0.051 | +0.025 |
| old L10 h3 | −0.030 | +0.452 | +0.120 |
| restored L8 h5 | +0.008 | +0.067 | +0.043 |
| restored L10 h0 | +0.029 | +0.052 | +0.055 |
| gqa2 L8 slot1 | −0.005 | +0.093 | +0.035 |
| gqa2 L8 slot5 | −0.001 | +0.429 | +0.167 |

old L10 h3 的 content 删除反而小幅恶化（CI [−0.056,−0.008]），而 denominator
收益 CI [0.334,0.568]；gqa2 slot5 的 content CI 跨零，denominator CI [0.284,0.578]。
支持**背景匹配占据分母、稀释有效读取**这一机制，但 denominator 同时是输出增益
干预，不能单独证明码本训练时发生了有害合并。下游非线性使 content 与 denominator
的 NLL 收益不可相加。restored L8 h5 的 denominator CI 跨零，不能将该分解结论
泛化到所有路径。某些 query 的全删放大很大（restored L8 h5 最大约 1.41e5）；
half 的一致正收益提供较小剂量证据，但全删仍应视作使用来源标注的诊断。

### 49.3 长链有区分力，短链总质量仍占优

old L10 h3 的候选背景质量中 **96.73% 来自长度 1–2 的链**；gqa2 slot5 为
91.84%。其余四条分别为 old h0 82.74%、restored h5 86.86%、restored h0 88.79%、
gqa2 slot1 76.51%。以下为 16×5 个 queries 的 pooled 生存计数（S / D）：

| 路径 | L≥1 | L≥2 | L≥3 | L≥4 | L≥5 |
|---|---:|---:|---:|---:|---:|
| old L10 h3 | 161 / 21190 | 69 / 1369 | 35 / 72 | 20 / 3 | 10 / 0 |
| gqa2 L8 slot5 | 166 / 81166 | 72 / 5537 | 37 / 346 | 22 / 14 | 11 / 1 |

因此不能只用“长链仍无法区分来源”解释：这两条路径长链相当有选择性，但短链
数量巨大，在归一化分母中仍占优势。码本频率统计仅作背景，不作有害性的判据：
old h3 的 K entropy-effective codes=42.91、最高频占比 6.33%；gqa2 slot5 为
16.66、10.45%。本轮只统计这一 NIAH 子集，不能替代大验证集的长尾利用率评估；
GQA 共享 K 的重复 slots 也不能当成独立码本合并统计。

### 49.4 后 8 trials 的短链过滤补测

主实验后固定阈值 3，未搜索其他阈值，复用 trials 8–15。`short_candidate` 只删除
D 中链长 1–2 的边，仍用针位置保护来源；`min_chain3` 删除所有链长 1–2 的边，
包括针和局部上下文，因此该规则本身不读取来源位置。仍然一次只干预一个 query。

| 路径 | 保护来源、删背景短链的收益 [95% CI] | 不看来源、要求 L≥3 的收益 [95% CI] |
|---|---:|---:|
| old L10 h0 | +0.125 [0.028, 0.249] | −0.025 [−0.068, 0.017] |
| **old L10 h3** | **+0.681 [0.388, 0.974]** | **+0.432 [0.097, 0.735]** |
| restored L8 h5 | +0.218 [0.135, 0.296] | −0.030 [−0.081, 0.019] |
| restored L10 h0 | +0.219 [0.129, 0.324] | +0.002 [−0.047, 0.049] |
| gqa2 L8 slot1 | +0.091 [0.047, 0.139] | +0.053 [0.018, 0.090] |
| **gqa2 L8 slot5** | **+0.478 [0.273, 0.656]** | **+0.290 [0.135, 0.424]** |

old h3 的 L≥3 收益主要来自第 3/4/5 位（+0.698/+1.185/+0.469），第 2 位反而
−0.221；gqa2 slot5 第 3/4 位 +1.024/+0.463，第 2 位 −0.092。这符合早期正确
来源也只有短链的情况，不能统一把短链当错误。即使同一 head，也应区分查询阶段。
三条路径 L≥3 的 NLL CI 为正，其余三条跨零；不支持所有头统一设硬阈值。

**当前结论：** 三个 checkpoint 均能筛到“正确来源有用，但背景匹配竞争压低读取”
的路径，最强候选为 old L10 h3 和 gqa2 L8 slot5。它不是 restored 特有现象，
也尚未解释模型总体性能排序。短链背景质量与删边收益，比 coverage 更直接地反映
该问题；保留这些指标和 source 必要性、随机删边对照联合判断。
下一步可检验按 head/query 状态调节短链权重，在新输入、其他长度/深度及自由生成
上同时启用干预。要把原因进一步归于**码本有害合并**，仍需保持其他条件的符号
拆分/重路由 rescue；当前实验未拆分码本，也未证明用户提出的 q/k 训练正反馈。

### 49.5 实现、产物与验证

新增 `tools/niah_merge_screen.py`、`tools/niah_short_chain_screen.py`、
`tools/summarize_niah_merge.py` 和 CPU 测试 `tests/test_niah_merge_screen.py`。
这些都是 opt-in 诊断，未修改模型、训练配置、kernel 或 checkpoint。

原始输出均在忽略目录：

- 主实验 trial 0：`build/niah_merge_smoke_20261016/{old,restored,gqa2}.json`。
- 主实验 trials 1–15：`build/niah_merge_screen_20261016/{old,restored,gqa2}.json`。
- 主汇总：`build/niah_merge_summary_20261016.json`；图：`build/niah_merge_effects_20261016.png`。
- 短链补测：`build/niah_short_chain_20261016/{old,restored,gqa2}.json`，附 wrapper hash manifest。
- 补测汇总：`build/niah_short_chain_summary_20261016.json`。

所有 baseline 与 §47 六 token NLL 逐项完全一致；所有 sham 完全一致；每次干预
前的实际 core row 与捕获值完全一致；所有模型 buffers 保持不变，零非有限干预。
FP64 重建相对真实 BF16 core 的最大相对 L2 残差 old/restored/gqa2 分别
0.00894/0.00608/0.01052（覆盖全部统计行），没有将重建值替换成 CUDA baseline。
随机对照的各行删除质量误差最大 1.13e−14。四项独立 CPU 测试通过：直接几何和、
下溢后 survivor/fallback、逐距离 bin 等质量随机删除、带/不带来源保护的短链过滤。
本轮无 kernel 改动，未运行 CUDA sanitizer。主实验进程峰值 allocated
1.43/1.46/2.53 GiB，在共享 GPU 上顺序执行，未启动或重启训练。

复现（沿用 §47 的 `$PY`、缓存环境和 `NEW_p1`；输出目录必须尚不存在）：

```bash
$PY dism_v3/tools/niah_merge_screen.py --p1 dism_v3/build/NEW_p1 \
  --out dism_v3/build/NEW_merge --trials 16
$PY dism_v3/tools/summarize_niah_merge.py \
  --directories dism_v3/build/NEW_merge --out dism_v3/build/NEW_merge_summary.json
$PY dism_v3/tools/niah_short_chain_screen.py --p1 dism_v3/build/NEW_p1 \
  --out dism_v3/build/NEW_short --start-trial 8 --trials 8 \
  --modes sham short_candidate min_chain3
$PY dism_v3/tools/summarize_niah_merge.py \
  --directories dism_v3/build/NEW_short --out dism_v3/build/NEW_short_summary.json \
  --expected-trials 8 9 10 11 12 13 14 15
$PY dism_v3/tests/test_niah_merge_screen.py
```

汇总可额外传 `--row-summary` 指向同 suite 的 P2/P3 汇总，以计算使用实际 τ 的
短链质量占比；不传时仍汇总实际干预、mass、生存计数与码本频率。

## 50. 按上下文长度缩放 τ：质量试验与 SAM 复杂度边界（2026-10-16）

### 50.1 方案与实验范围

按用户建议测试长度补偿。这里 τ 是 `softplus(log_sel_tau)` 后的正值，增大它
提高长链相对短链的权重；不直接缩放未经 softplus 的参数。
`W_tau(l)=sum(exp(r*tau), r=1..l)`。若一条有效链长为 ls、M 条背景链长约为 lb，
末项主导时背景/来源比约为 `M*exp(-tau*(ls-lb))`；M 随 N 线性增长且链长差固定
时，加法 `delta_tau≈log(N/N0)/(ls-lb)` 可以补偿。这是近似动机，不是已验证的
entropy invariance，也无法区分同链长的有害碰撞。

新增 `tools/niah_tau_scaling.py`，使用本地 tokenizer/PG19，**新 seed=20261017，
每格 8 trials**，长度 2048/8192/16384、深度 .05/.95，三个既有 checkpoint。
每格 present/absent 配对，三个模型共用同一输入；本轮输入不是 §47–49 的旧样本。
固定 N0=2048，比较四种规则：

| 规则 | 实际 τ | 2k / 8k / 16k 系数或增量 |
|---|---|---|
| log_mul | `tau * max(1, log(N)/log(2048))` | ×1 / ×1.1818 / ×1.2727 |
| log_add | `tau + .5*max(0, log(N/2048))` | +0 / +0.6931 / +1.0397 |
| fixed_mul | `tau * log(16384)/log(2048)` | 始终 ×1.2727 |
| fixed_add | `tau + .5*log(8)` | 始终 +1.0397 |

每种规则分别作用于全部 L6–L11 slots（all），或 §49 每模型的两个已定位 slots
（selected）；加 baseline 共 9 cases。等价配置只运行一次并记录 `equivalent_to`：
2k 两个 log 规则等于 baseline；16k 每个 log 规则与对应 fixed 规则相等。

**整段冻结 τ**：N 是本次请求已知的上下文预算，所有位置使用同一组每头 τ，
通过临时 `compiler.voc_forward` wrapper 注入。全部位置和下游层真正重算，
不是 §49 的单 query 输出差量。因此会包含后续层符号变化和多位置相互作用。
这是兼容当前固定 τ 算法的质量实验，没有实现逐 query τ，也没有调用 SAM 做
性能测试。指标为五位数字平均 `baseline NLL−modified NLL`，正数改善；先在
trial 内平均，再做 10,000 次 paired bootstrap，pointwise 95% CI。另保存包含
前导空白的六 token NLL、准确率、逐数字收益和 present−absent 干预交互。

### 50.2 结果：锐化有收益，尚未证明长度依赖的必要性

16k/.05 全头干预：

| checkpoint | baseline 数字 NLL | log_mul 收益 [95% CI] | log_add 收益 [95% CI] | log_add absent 收益 |
|---|---:|---:|---:|---:|
| old | 2.764 | +0.105 [0.059, 0.149] | **+0.337 [0.213, 0.455]** | +0.016 |
| restored | 3.065 | +0.162 [0.027, 0.306] | **+0.421 [0.177, 0.672]** | −0.026 |
| gqa2 | 2.786 | +0.317 [0.135, 0.546] | **+0.460 [0.209, 0.738]** | −0.055 |

log_add 的 present−absent 交互分别 +0.322 [0.197,0.431]、+0.447 [0.210,0.695]、
+0.516 [0.264,0.791]；改善与针存在有关，不能简单解释为普遍提高数字预测概率。
16k/.95 的 all log_add 收益也都为正：old +0.334 [0.211,0.453]，restored
+0.270 [0.113,0.428]，gqa2 +0.304 [0.117,0.494]。

但固定增大 τ 是关键对照。远针 .05 全头结果：

| checkpoint | 2k fixed_add 收益 | 8k log_add 收益 | 8k fixed_add 收益 | 16k 二者共同收益 |
|---|---:|---:|---:|---:|
| old | +0.294 | +0.203 | +0.287 | +0.337 |
| restored | +0.334 | +0.450 | +0.543 | +0.421 |
| gqa2 | +0.354 | +0.223 | +0.301 | +0.460 |

2k 远针 fixed_add 的三个 CI 都大于零。8k 远针上，fixed_add 超过 log_add
的收益分别为 +0.084 [0.067,0.101]、+0.093 [0.039,0.151]、+0.078 [0.037,0.139]。
因此当前实验更直接支持**这些 checkpoint 在此 NIAH 条件下受益于增大 τ**，
尚不能支持“必须采用 log(N) 规则”或“已保持 entropy 不变”。系数 .5 未搜索，
乘法和加法的剂量也不等价，不能把加法较好解释为其函数形式已胜出。

局部路径与全模型结果不同：16k/.05 selected log_add 的 old/restored/gqa2 收益
分别 +0.265 [0.170,0.348]、+0.041 [−0.088,0.179]、+0.459 [0.230,0.715]。
restored 的全头收益明显，但仅改两个已定位头不能确认有效；16k/.95 selected
收益 −0.051，CI [−0.347,0.130]。不能把全头改善都归因于 §49 的六条路径。

16k 远针 all log_add 的第 1 位收益 old/restored/gqa2 分别 +0.002/+0.014/−0.016，
主要改善仍发生在后续数字，符合正确前缀使有效链增长的情况。尚未测自由生成、
普通语言建模 perplexity、其他任务或更长上下文；本轮不证明修复了码本合并。

### 50.3 固定 τ 与逐 query τ 对算法的影响

根据当前 `README.md` 的精确界限及 `inference/csrc` 实现：

1. **每个请求固定一组缩放后的 τ：保留现有算法结构。** SAM topology 及
   prefill 的事件分解不因 τ 数值改变；改变的是事件系数。固定 head/readout/value
   维数时，prefill 仍为 `O(N log N)`。decode 从 prime 开始使用同一组 τ，
   普通步与重建的原有复杂度不变。可按 prompt 长度或声明的上下文预算选 τ，
   之后冻结；若后来延长请求，它就不再是按实时长度精确更新的规则。
2. **每个 query 使用 tau_i=f(i)：当前缓存不能直接支持。**
   `prefill.hpp::emit` 的 mode 2 把 `W_tau(lca_length(key,centroid))` 放在 key
   侧并共享前缀聚合。tau_i 改变后，此权重也依赖 query，原有分离失效。
   decode 的 `Planner::tau` 是 const；`edge_weight/edge_decay/prefix_weight`
   和 GPU sampled-prefix 矩阵都依赖它。不同链长的权重比会改变，不能靠一个
   输出缩放修复全部历史。无权的 subtree 聚合和 SAM topology 可重用，但加权
   摘要需重算或更换表示。朴素每 token 重扫/重建会失去原来的效率；这不是对
   任何其他精确次二次算法的不可能性证明。
3. **几何长度分档是可行的待实现折中。** 例如按 2k/4k/8k/16k 分档，档内
   冻结 τ，按当前位置选档，不依赖未来 token。对一个层已有的 q/k/SQ/SK/V，
   每档在该档末端的完整 prefix 上运行原固定 τ prefill，只保留属于本档的
   query 输出，再拼起来传给下一层。若 prefix 长度为几何级数，则总工作
   `sum_b O(2^b log(2^b)) = O(N log N)`；不需要每档都运行长度 N 的完整前缀。
   这精确实现分档规则，而非连续 log 规则。decode 跨档时可用已存完整标签和
   SK/V 历史重建新 τ 的摘要；不能用仅最近一段历史替代完整 prime。
   此调度尚未实现或做数值/性能验证。

还需纠正“现在 decode 有端到端 subquadratic 保证”的前提：README 明确指出
当前 snapshot 重建最坏 `O(T R DV log T)`，固定每 b=128 tokens 重建的摊销项为
`O(T R DV log T / b)`；**固定 b 不提供端到端次二次渐近保证**。普通 decode 步
不扫全历史，但重建不能忽略。长度分档的额外 tau 重建只在倍增边界发生，不等于
消除了既有周期重建成本；要取得整体界限，仍需设计/验证随 T 增长的重建调度。
此外，当前 `HardDismDecoder` 显式拒绝 GQA G>1；GQA2 的质量结果不代表它已接入
SAM 快速推理。以上缩放本身的算法分析与 GQA 接口适配是两项独立工作。

代码依据：`python/flash_dism/inference/csrc/prefill.hpp` 的 `emit/finish`，
`csrc/planner.hpp` 的 `rebuild/append`，`csrc/cuda_kernels.cu` 的 sampled-prefix
构造，`inference/api.py` 的固定 τ/GQA 检查，以及 `README.md` 的 Prefill/Decoding。

### 50.4 产物与验证

- 原始输入和分格结果：`build/niah_tau_scaling_20261016/{inputs,old,restored,gqa2}.json`。
- 汇总：`build/niah_tau_scaling_summary_20261016.json`。
- 图：`build/niah_tau_scaling_effects_20261016.png`。
- 汇总脚本：`tools/summarize_niah_tau.py`；输出包含全部 18 格、9 cases 的
  present/absent 及固定增益对照，不只保存显著/正收益结果。

全部模型加载自身 checkpoint spec；只使用本 worktree 的 flash_dism/cu_flash_dism。
每格干预后重跑首条 baseline，六 token NLL/准确率等字段与原 baseline 完全一致；
全部 buffers 未变，全部评分有限。实际 τ 注入位于原 CUDA forward 入口，无 FP64
输出替代或行补丁。主进程峰值 allocated old/restored/gqa2 为 1.44/1.48/2.54 GiB。
无 checkpoint 写入、训练重启、kernel 改动或新数据下载。共享 GPU 上的运行耗时
不作为性能基准，未新增 sanitizer/吞吐结论。

复现（缓存环境同 §47，输出目录必须不存在）：

```bash
$PY dism_v3/tools/niah_tau_scaling.py --out dism_v3/build/NEW_tau \
  --seed 20261017 --trials 8 --contexts 2048 8192 16384 --depths .05 .95
$PY dism_v3/tools/summarize_niah_tau.py --directory dism_v3/build/NEW_tau \
  --out dism_v3/build/NEW_tau_summary.json
```

## 51. τ 梯度：数值抵消、独立切线路径与 hard/soft 目标冲突（2026-10-16）

本节为代码审计、数学推导和 CPU 数值诊断；没有修改训练 kernel、optimizer、
checkpoint 或启动训练。用户补充：先前训练中 hard/soft 对 τ 通常给出相反方向。
本轮未重新测实际训练 batch 的两支方向，不预设哪支一定推大/推小。

### 51.1 三种问题必须分开

1. **分支内部数值误差。** 当前 soft `backward_qk.cu` 由 `P*(S*E-delta)`
   启动反向对角递推，最后累加全部 gate 梯度得到 dτ；delta 来自保存的 BF16 O。
   §44 已测 soft τ 相对误差约 1%（最大 2.1%），放大 logits 后可达约 8.3%。
   `BACKWARD_DIMS.md` 的历史 hard/统一反向路径中，仅以高精度 O 替换 delta
   就使部分误差从 0.03–0.06 降到 1e−7–1e−6，说明输出舍入可以是主要来源。
2. **hard 路由差异。** 当前 `src/hard_backward.cu` 已使用解析 g_L、移位的四矩
   和显式质量归一化，避免 BF16 O/delta；不能把上述旧路径缺陷等同于当前所有
   hard 反向。§43 测到 event 多 chunk 路径偏差，而 quadratic 通常准确得多。
   `_chunk_backward` 的单 chunk tau 用 FP32 pair 在 BF16 vector cast 前收缩；
   跨 chunk 的 hq/hk 则在 BF16 模式下先把矩阵摘要转 BF16，再形成各自 τ 项，
   是进一步审计的候选位置，不能仅凭静态代码把它断定为唯一根因。
3. **真实 hard/soft 目标冲突。** 若两支在 FP64 下仍相反，精度修复不会消除
   冲突。两支梯度相加后的净值很小，会把各自误差放大成净方向偏差。用已经按
   实际采样/有效 token 归一化的贡献定义
   `kappa_mix=(abs(g_h)+abs(g_s))/(abs(g_h+g_s)+eps)`；它与分支内部求和条件数
   是两个不同指标。不要先对每支重新归一化再计算本来没有的冲突。

§50 证实改变 τ 可改善固定 checkpoint 的 NIAH，但没有证明数值误差导致训练
最终 τ 不合适。旧 NIAH 记录中全头 mean/max τ 与 recall 并没有给出强的预期排序；
这也不能排除某条具体读取路径的 τ 敏感性。训练短上下文的 LM 目标与长程检索
目标不同，即使梯度完全准确也可能偏好不同 τ。

### 51.2 等价的协方差形式与误差来源

对每个 query，设递推后未归一化权重为 w_j，`p_j=w_j/(1+sum(w))`，
`u_j=(SQ_i·SK_j)V_j`，`x_j=dO_i·u_j`，`h_j=∂log(w_j)/∂tau`。
加入 fallback 作为额外 j=0：w0=1、x0=h0=0。则

```
mu_x = sum_j p_j*x_j
mu_h = sum_j p_j*h_j
g_i  = sum_j p_j*(x_j-mu_x)*(h_j-mu_h)
     = Cov_p(x,h)
```

若原反向使用 `delta=mu_x+epsilon_i`，权重/切线暂视为精确，则其 τ 行误差为
`−epsilon_i*mu_h`。因此 BF16 输出引入的小 delta 误差会被有效链长度放大。
提高最终 head 标量归约精度无法恢复此前已经损坏的局部残差。

优先方案是**先计算真实加权均值，再累加中心化乘积**，或以分块 weighted
covariance 合并，避免最后才相减两个大 raw moments。两个块的权重 W、均值
mx/mh、未归一化协方差 C 可按以下方式合并：

```
C = C_A + C_B + (W_A*W_B/(W_A+W_B))*(mx_A-mx_B)*(mh_A-mh_B)
g = C/W
```

跨块不同 log-weight pivot 必须先统一质量尺度，C 与 W 一样缩放。fallback 也要
参与，不能当成归一化误差忽略。使用 FP32 局部 MMA/矩、固定树形 partial reduction，
必要时仅把少量最终 scalar partials 用 FP64 归约；不采用 shared FP32 atomicAdd。
这减少数值引入的抵消，但不能消除真实正负梯度相抵。

当前 hard quadratic 选 `c_i=g_(i+1)`，然后累加 g_L−c_i 的四矩。这对长链共模
有帮助，但晚位置只有短链时 c_i 可能远离全部有效 g_L，形成新的大数相减。
可比较真实加权均值中心、实际最大匹配长度中心以及分块 covariance；不是说
现有移位无效，也不是仅把最后 reduce 改 FP64 就解决所有条件数问题。

### 51.3 独立 τ-only JVP：保持原数学目标

soft 的自然对数递推为

```
s_ij = mix_ij + tau + softplus(s_(i-1,j-1))
h_ij = 1 + softplus'(s_(i-1,j-1))*h_(i-1,j-1)
```

因此可以在前向重算时多传播一个标量切线 h，再用上述 covariance 与实际 dO 收缩，
只返回 dτ。数学上与完整反向相同，不需要再把 τ 信号沿漫长的反向 gate 链传播。
soft 主计算本来为二次，此路径保持阶数，但额外扫描/边界/带宽和寄存器开销需要
实测。若只在 backward 重算，需要保存或重建**切线边界**；不能从 primal 边界
凭空恢复切线。纯 hard 则直接用 `h=g_L` 的解析/稳定递推式即可。

hard event 还可使用 `hard_prefill_grad_reference.py` 已有的前向 tangent-matrix
思路，从原 event 计划传播 ∂M/∂τ，再收缩真实 dO，避免依赖有争议的跨 chunk
backward 分解；保留 near-linear event 阶数，但 tangent 状态和归约仍须中心化及
精度检查。该 CPU helper 的存在不代表生产 GPU JVP 已实现或已经稳定。

实现时保持以下语义：

- 局部 dO 必须来自该次真实 forward，不能用 soft forward 的 dO 配 hard shadow
  输出然后声称得到全 hard loss 的梯度。
- 只**替换**既有 dτ，其他向量/词表梯度保持原路径；不能额外叠加一份完整 dτ。
- 当前 interpolation 已以 detached tau 做 `LSE−tau` 吸收。JVP 给 logM 的 +1
  只计一次；不能又从吸收后的 LSE 回传同一份 τ 梯度。
- `g_rho=sigmoid(rho)*g_tau`，rho=`log_sel_tau`，外层 softplus 链式法则仅一次。
- 生产 softplus/tanh 近似、log2 单位、有限 mask、packed 边界及 sequence-level
  hard sampling 都需保留。这里的 CPU 检查对应维护中的 thresholded-softplus
  oracle，不是对所有 CUDA 浮点近似的逐位导数声明。

可先抽样完整 sequence/head，在相同 operands/dO 上算高精度参考。若以概率 π
独立抽样，`g_corr=g_fast+I/pi*(g_precise-g_fast)` 对当步 precise 梯度无偏，
但低 π 会增加方差；每 K 步简单替换或对误差做 EMA 不具有同样的无偏保证。
抽样 queries 时仍须保留完整匹配历史，截短上下文会改变目标。
有限差分宜用作独立核对：多个步长、共同随机采样、FP64/FP32 readout，并正确
重算 tau-absorbed LSE。直接对 BF16 输出作很小步长差分容易得到零或舍入噪声，
不推荐作为主要训练估计器。

### 51.4 若冲突真实存在，需要单独设计 τ 的训练目标

soft `logM=tau+S`，其中 S 是负交叉熵式匹配分数；hard 匹配的 `logM=tau`。
同一个 τ 在 soft 分支还承担补偿模糊匹配负偏置的作用，因此反向诉求可以是结构性的。
如果 soft 想推大、hard 想推小，这提供一个可能解释；相反方向也不能仅靠这条近似
排除，最终要测实际 batch 的高精度梯度。

在数值路径对齐之后，建议用对照区分以下策略，均属于改变训练设计而非精度修复：

| 策略 | 目的与限制 |
|---|---|
| 共享 τ、精确 mixed 梯度 | 基准，检验单纯数值修复的贡献 |
| τ 只收实际 hard 样本的梯度 | 减少 soft 拉扯；按 hard 概率和有效 token 做正确估计，早期 hard 很少时方差大；其他层仍混合采样，不等价于全 hard 模型目标 |
| 间歇全 hard batch，只更新 τ | 直接校准部署目标；必须真实重算全 hard forward/dO，其他参数冻结但下游反传仍需保留；额外计算不能视为免费 |
| tau_h 与 soft 校准偏移 | 例如 rho_s=stopgrad(rho_h)+delta、tau_s=softplus(rho_s)，hard 更新 rho_h，soft 更新 delta；独立参数可减轻冲突，也可能增加 soft/hard gap，需约束/退火 delta 并验证 |

不能靠相反梯度各自 normalize、取绝对值或直接删掉某符号来“修复 cancellation”。
先分别记录 g_h/g_s、生产−参考的 signed bias、有效梯度 sign agreement、内部和
分支间 cancellation ratio，以及 rho/tau、sigmoid(rho)、Adam 更新后的实际 Δtau。
接近零的参考梯度单独分组，避免相对误差和 sign rate 被微小值主导。若精度修复
后两支仍稳定相反，就有证据进入上述目标分离对照；若参考和生产经常异号，应先
修数值路径。两者都不能通过重置已有 Adam/RNG/annealing 状态做混杂比较。

### 51.5 本轮 CPU 数值检查

新增 opt-in `tools/tau_jvp_analysis.py`。16 个 CPU cases 覆盖 soft/尖锐 soft/
随机 hard/全重复 hard、N65/257、tau=.3/2，采用实际形式的 signed SQ/SK readout
与独立 dO。对每个 case 比较独立 autograd、前向切线 covariance 和中心差分。
FP64 JVP 对 autograd 最大绝对误差 **1.47e−14**，全部断言通过；FP32 JVP 的
最大绝对误差 **1.61e−7**。只把输出舍入 BF16、其余仍使用 FP64 的 delta 路径，
在重复 hard fixture 上最大绝对误差达到 **0.302**。这隔离了 saved-O 舍入机制，
不是说当前 hard quadratic 仍然有该缺陷。

另一个“晚 query、全是短匹配”的行级算术例子，真实梯度为 0.00735338012；
模拟当前最大位置中心的 FP32 四矩为 0.00878906436，均值中心的 FP32 covariance
为 0.00735337986。它显示改中心值得测试，但没有复制 GPU reduction 顺序或 CUDA
近似，不能作为生产 kernel 改善倍数。均匀长链例子的真实零梯度，两种中心都为零。

原始结果：`build/tau_jvp_analysis_fp64_20261016.json`。初版辅助行的几何权重常数
误用了默认 FP32 scalar，已修为 FP64 并重跑；原始文件保留在
`build/tau_jvp_analysis_20261016.json`，不用于本节精度结论。
本轮未做实际 checkpoint 梯度分支审计、GPU 吞吐/数值/sanitizer 或训练 A/B。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY dism_v3/tools/tau_jvp_analysis.py \
  --out dism_v3/build/NEW_tau_jvp.json
```
