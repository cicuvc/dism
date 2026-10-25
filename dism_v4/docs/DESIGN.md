# DISM v4：单一 Q 主线与全符号候选读取

日期：2026-10-25。

状态：**设计提案，尚无 v4 生产实现或训练结果。** 用户明确将本方案命名为
v4，并要求先记录完整方案。本次工作仅新增设计文档；v3 算子、默认配置、已有
checkpoint 和运行中的训练不因此改变。旧 GQA 实验仅提供多查询共享 K/V 的
启发，本文的公式才是 v4 的语义定义。

后续讨论更新：第 12 节将训练/推理均使用 hard 历史列为主方向；第 3 节的 soft
历史保留为参考。第 13 节记录用户新提出的“候选分子混合后做 RMSNorm”变体，
它主动修改原来的独立候选归一化，因此可移除第 6 节的动态分母瓶颈。目前尚未
选择最终归一化变体或启动实验。

第 14 节进一步澄清工程约束：允许词表相关计算位于独立 GEMM/kernel，只要求
复杂的对角递推/readout 前向和反向不承担 V 维收缩。原版独立候选归一化可以
用分阶段、分 query block 的精确算法实现，**无需落地完整 N x N 中间张量**。
因此，第 6 节的动态分母是计算依赖和总成本问题，并不构成采用 RMSNorm 的
必要理由；两种归一化应作为不同的模型语义比较。

## 1. 动机与已经确定的要求

v3 用可微 soft 路径学习符号分配，再使用离散后缀匹配实现 hard 推理。v4
考虑让推理承担更多工作：保留离散历史索引，但在每个位置精确枚举当前 Q 的
所有符号候选，让概率分布直接决定实际读取结果。

确定的构造：

1. 每个 token/head 只提交一个 Q 符号，构成唯一的 Q 主线；K 也保留单一硬符号串。
2. 输出时，从同一个 Q 主线前缀出发，枚举当前 Q 的全部 V 个符号。
3. 候选共享当前 head 的 SQ、SK、V、tau 和 K 历史，不新增独立 Q/SQ 投影。
4. 每个候选独立完成后缀匹配、读取和归一化，然后按当前 Q 的符号概率混合输出。
5. 只有主线符号进入后续匹配状态。候选状态不延续，不组成指数数量的历史串。
6. 枚举全部符号，不将 top-k 截断当成该构造的精确实现。
7. 保留 numerator-only 的 signed SQ/SK readout，以及分母中的单位 fallback。
8. 设计重点还包括大词表下的成本：避免让复杂对角递推/readout kernel 的
   收缩维度膨胀到词表大小 V。**允许独立的词表相关 GEMM/kernel 承担 V 维运算**，
   不要求全程序完全没有 O(N^2 V) 计算。分阶段实现见第 14 节。

输出相对于候选的混合是概率加权；每个候选内部由于 signed SQ/SK，输出本身
不必是 value 向量的凸组合。

## 2. 记号和精确 hard 定义

以下省略 batch/head。N 为长度，V 为符号词表大小，D 为 selector embedding
宽度，R 为 SQ/SK 宽度，D_v 为 value 宽度。token 下标从 1 开始。

令

\[
z_i^Q(c)=q_i^\top E_Q(c),\quad p_i(c)=\operatorname{softmax}_c z_i^Q(c),
\qquad z_j^K(c)=k_j^\top E_K(c).
\]

主线符号是 a_i = argmax_c z_i^Q(c)，K 符号是 b_j = argmax_c z_j^K(c)。
argmax tie-breaking 必须明确定义并在 prefill/decode 中一致。令 alpha = exp(tau)。
tau 当前按 head 固定；token-dependent tau 不属于已分析的快速算法契约。

主线未归一化匹配状态：

\[
H^H_{ij}=\alpha\,\mathbf1[a_i=b_j](1+H^H_{i-1,j-1}).
\]

下标越界时 H=0；仅 j<=i 的项参与当前输出。查询 reset 清空相应匹配历史，
序列边界、因果性和 packed-row 语义需要显式传递，不能靠拼接候选伪造 token。

当前位置尝试符号 c 时：

\[
W_{ij}^{(c)}=\alpha\,\mathbf1[c=b_j](1+H^H_{i-1,j-1}),
\quad Z_i(c)=\sum_{j\le i}W_{ij}^{(c)},
\]

\[
o_i^{(c)}=
\frac{\sum_{j\le i}W_{ij}^{(c)}\langle sq_i,sk_j\rangle v_j}{1+Z_i(c)},
\qquad \boxed{o_i=\sum_{c=1}^V p_i(c)o_i^{(c)}}.
\]

下一步保留 H^H_i = W_i^(a_i)，而不是候选状态的平均。
模型的后续层正常使用混合输出 o_i，因此混合读取仍可通过模型计算影响未来的
Q/K；“不延续候选”专指同一 DISM 层的符号匹配状态。

例：主线前缀为 ABC，当前读取尝试 ABCA、ABCB、ABCC、……，但若主线当前
选了 D，下一步共同历史就是 ABCD。

必须区分：

\[
\sum_c p_i(c)\frac{N_i(c)}{1+Z_i(c)}
\ne \frac{\sum_c p_i(c)N_i(c)}{1+\sum_c p_i(c)Z_i(c)}.
\]

右式不是本方案。某符号没有任何因果 K 端点时其候选输出为零，可精确跳过；
其概率质量不能重新分配给其他符号，否则改变语义。

## 3. Soft 参考定义：软历史、独立候选归一化

区分读取概率 p_i(c) 和历史路由分布 rho_i^Q(c)、rho_j^K(c)。例如：

\[
\rho_i^Q=\operatorname{softmax}(z_i^Q/T_Q),\qquad
\rho_j^K=\operatorname{softmax}(z_j^K/T_K).
\]

p_i 仍使用原始读取温度，不能因历史硬化而一起退化成 argmax。
温度趋零的表述需假设唯一最大值；精确硬端直接使用约定 tie-breaking 的 one-hot。
另一种路径是概率与 argmax one-hot 的显式插值，它不是处处光滑的温度松弛。

算术匹配的语义参考定义为：

\[
m^A_{ij}=\sum_c\rho_i^Q(c)\rho_j^K(c),\qquad
H^A_{ij}=\alpha m^A_{ij}(1+H^A_{i-1,j-1}).
\]

在当前 Q 候选条件下：

\[
A_{ij}=\alpha(1+H^A_{i-1,j-1})\mathbf1[j\le i],\qquad
W_{ij}^{(c)}=A_{ij}\rho_j^K(c),
\]

再按第 2 节逐候选归一化、按 p_i 混合。这里的 A 不是整个历史匹配权重 H，
而是尚未施加当前符号兼容性的延续因子。

当 rho^Q、rho^K 都取主线 one-hot 时，本节严格退化到第 2 节的 hard 算子，
即使 p_i 仍是非退化分布。仅 rho^K 为 one-hot 时，当前候选仍是精确的 K 符号桶
读取，但历史 Q 可以保持软化。

这是未归一化历史匹配状态的均值松弛，不是全部历史符号串输出的精确边缘化。
固定连续操作数、独立采样 Q/K 符号时，算术递推可解释为匹配状态的期望；
最终的归一化输出不能与期望交换。该参考定义不承诺逼近误差随硬化单调下降。

## 4. 梯度的新增能力和剩余缺口

固定主线历史及其他操作数，令 g_i = dL/do_i，当前 Q 读取 logits 的直接梯度为：

\[
\frac{\partial L}{\partial z_i^Q(c)}\bigg|_{\text{direct readout}}
=p_i(c)\langle g_i,o_i^{(c)}-o_i\rangle.
\]

全部候选都计算时，这个局部直接梯度无需 REINFORCE 或 STE，没有候选采样方差。
这不是“所有训练噪声消失”的声明，也不是只看当前 token 的 LM loss：g_i 可含
后续网络和未来 loss 的连续反传贡献。

在 soft 历史下，还应加上通过 H 和 K 路由的梯度。硬化后直接读取梯度仍保留，
但 a_i 对未来同层匹配路由的离散作用，以及 b_j 的离散选择，仍没有普通梯度。
共享 Q/K 在 GDN 等组件中的其他梯度必须另外计入。不能把上述偏导当成完整
共享参数更新，也不能据此宣称 K selector 已经获得可用学习信号。

## 5. Jensen：历史匹配可以继续低成本计算

算术匹配概率的 Jensen 下界仍成立：

\[
\log m^A_{ij}
=\log\mathbb E_{c\sim\rho_j^K}[\rho_i^Q(c)]
\ge \mathbb E_{c\sim\rho_j^K}[\log\rho_i^Q(c)],
\]

\[
m^J_{ij}=\exp\!\left(\sum_c\rho_j^K(c)\log\rho_i^Q(c)\right).
\]

数学依据是 log 的凹性，见 Boyd/Vandenberghe 的
[Jensen 不等式说明](https://see.stanford.edu/materials/lsocoee364a/03ConvexFunctions.pdf)。
它是确定性下界／surrogate，不是无偏的匹配概率估计。

由于 log rho^Q(c) = q_i^T E_Q(c)/T_Q - L_i^Q：

\[
\log m^J_{ij}
=\frac{q_i^\top}{T_Q}\underbrace{\sum_c\rho_j^K(c)E_Q(c)}_{\bar E_j}
-L_i^Q.
\]

这是 v3 direction=true 的同一个低维恒等式。词表工作留在 O(NVD) 的插值和
LSE 中，pairwise score 收缩宽度保持 D，而不是 V。

因此可定义另一条 soft surrogate：

\[
H^J_{ij}=\alpha m^J_{ij}(1+H^J_{i-1,j-1}),
\]

并用 H^J 代替第 3 节候选权重里的 H^A。精确 one-hot 路由下它仍对应第 2 节。
实现必须避免 0 * log(0) 的 NaN；硬端用明确的离散定义。

由非负递推可证明 H^J <= H^A，但归一化和 signed readout 破坏了简单的输出
排序。因此这不是输出误差界，也不是 loss 下界。Jensen gap 是否可接受要测量。

**历史使用 Jensen，并没有自动解决完整候选读取的词表维度成本。**

## 6. 真正新增的瓶颈：每个符号自己的分母

对任一选定的历史 H（算术或 Jensen），令 K_{jc}=rho_j^K(c)。候选分母是：

\[
Z=AK,\qquad Z_i(c)=\sum_j A_{ij}\rho_j^K(c).
\]

无需显式保存 N x N x V 权重。定义修正后的读取概率：

\[
u_i(c)=\frac{p_i(c)}{1+Z_i(c)},\qquad
B_{ij}=\sum_c u_i(c)\rho_j^K(c),\qquad M=A\odot B.
\]

最终输出可写成：

\[
o_i=\sum_j M_{ij}\langle sq_i,sk_j\rangle v_j.
\]

这里不能再加一次全行归一化。候选分母已经包含 fallback，并有
sum_j M_ij = sum_c p_i(c) Z_i(c)/(1+Z_i(c)) <= 1。

这个因式分解避免了 N x N x V 张量，却仍有 Z=AK 和 B=uK^T 两次 V 相关 dense
收缩，通用实现约 O(N^2 V)。算术历史另有 rho^Q (rho^K)^T 的同量级收缩；
Jensen 历史可消掉历史这一次 V 收缩，不能消掉另外两次。

### 6.1 对完整读取系数继续使用 Jensen

仍可合法地写：

\[
B_{ij}=\mathbb E_{c\sim\rho_j^K}
\left[\frac{p_i(c)}{1+Z_i(c)}\right]
\ge B^J_{ij}
=\exp\left(
\mathbb E_{\rho_j^K}\log p_i(c)
-\mathbb E_{\rho_j^K}\log(1+Z_i(c))\right).
\]

第一项依然等于 q_i^T bar(E)_j - LSE_i，只有 D 维。
第二项涉及动态的 V 维向量 ell_i(c)=log(1+Z_i(c))，一般不落在原 E_Q 的仿射
张成空间里。它不能被原来的一个 query LSE 标量吸收，也没有已知的同一低维
恒等式；甚至得到 Z 本身已经包含上面的 dense 收缩。

K 为 one-hot 时，这个读取层面的 Jensen 等号成立。这很有价值，但不意味着
稠密 soft K 时也能便宜计算。若近似 ell_i(c) 为低维特征展开，必须单独验证
近似误差；未经控制的拟合不保证 Jensen 的下界方向，不能当作精确实现。

B^J<=B 是同一 A/Z 下逐 pair 的非负系数界。因为 readout 是 signed 的，不能
将其解释为输出或 loss 的界，也不能据此断言更接近 v3 soft。

### 6.2 为什么不能靠换归一化“免费解决”

先混合权重再共用一个分母，会移除一部分候选特有结构，但违反第 2 节定义。
省略动态分母修正也不行。例如单 token、one-hot K、readout/value=1，alpha=10，
匹配符号的 Q 概率为 0.9：旧 hard=10/11，旧 soft=9/10，新 hard=0.9*10/11。
三者分别约 0.9091、0.9000、0.8182。新 hard 不保证更接近旧 soft。

可以构造更松的廉价界，例如用 sum_c Z_i(c)=sum_j A_ij 控制全部候选分母，
但通常丢失所需的硬端归一化。存在廉价下界与存在合适的 v4 soft 路径不是一回事。
本文没有证明任何通用低维算法都不可能；这里只指出原有恒等式不能直接完成任务。

## 7. 兼顾大词表的候选路线

| 路线 | 是否保留全 Q 候选 | 主要 pairwise 成本 | 关键限制 |
|---|---|---|---|
| 全 soft K，算术历史 | 是 | 通用 dense O(N^2 V) | 语义参考；不具备 v3 的低维优势 |
| 全 soft K，Jensen 历史 | 是 | 历史 O(N^2 D)，读取仍 O(N^2 V) | 只解决历史匹配部分 |
| hard K，soft Q 历史 | 是 | 低维匹配加符号桶归约 | K 符号选择没有直接可微路径 |
| 每个 K 支持 s 个符号 | 是 | 候选读取约 O(N^2 s)，另计词表投影 | 改变 soft K；支持集与梯度需定义 |
| 压缩动态分母场 ell_i(c) | 是或近似 | 取决于压缩秩和生成方法 | 尚无误差、端点或效率保证 |

hard K 时，Jensen 与算术历史匹配相等：m_ij=rho_i^Q(b_j)。候选读取可直接用：

\[
Z_i(c)=\sum_{j:b_j=c}A_{ij},\qquad
M_{ij}=A_{ij}\frac{p_i(b_j)}{1+Z_i(b_j)}.
\]

这是按符号的 segmented reduction 和查表，不需要 N^2 V 矩阵乘法。Q 的全词表
读取仍保持原定义，但它不是同时可微训练 Q/K 的完整解决方案。

稀疏 K 也可用类似的 scatter/gather；这里截断的是 soft K 的支持，不是用户要求
枚举的 Q 候选。若采用 top-s，其概率重归一化、尾部质量、离散支持选择的梯度
必须单独定义；不能将其等同于原始稠密 soft K。

目前不将某一条性能路线默认定为生产 soft。先保留第 3 节 oracle，并验证第 5 节
Jensen 历史版本，接着量化第 6 节的动态分母成本与可压缩性。

## 8. Hard prefill 与 decode 的算法路线

### 8.1 Prefill

K 串的 SAM 和 payload 索引共享。对主线前缀的状态/匹配长度 (s,l)，为所有 c
求临时转移 (s_c,l_c)，记录候选 query，最后只提交 c=a_i 的状态。
用当前 token 的时间戳约束全部候选 j<=i，不能将 V 条候选误作 V 个时间步。

完整 K 串建立的离线 SAM 可以包含未来符号，但向量聚合和分母必须过滤未来
端点；仅有 SAM 匹配不自动保证因果性。

已有 v3 centroid/event prefill 可作为结构参考：将 N 个 query 扩展到 NV 个
候选 query，共享 N 个 key。若转移查询被加速、事件构建推广到独立的 query ID
与时间戳，目标事件量为 O((N+NV) log N)，固定向量维度下是 O(NV log N)。
这是算法设计推导，尚未实现或验证 wall-clock/内存常数。

不能将每个候选直接回滚后再次逐条走 suffix links，并套用单一字符串扫描的
摊销线性界。静态 SAM 可预计算 O(NV) 的完整转移表，或用沿 suffix links 继承
的持久化映射存储最近有该符号出边的祖先。后一种方案可用 O(N log V) 级别的
额外预处理/存储和 O(log V) 查询；需要保留出边来源以更新匹配长度。

完整转移表的 NV 内存可能很贵；V=512 虽在渐近上是常数，不能因此忽略其实际
开销。候选归一化可以先做标量阶段，然后将 p_i(c)/(1+Z_i(c)) 合入 query 事件
系数，向原输出行累加；不必持久保存所有候选的 R x D_v 矩阵。

### 8.2 Decode

保留一套 K snapshot/summary、历史 SK/V、一条 Q 主线状态和主线 tail 匹配。
候选只产生临时查询；rebuild 与 payload 存储不按 V 复制。

raw tail 可一次按 b_j 分桶：对当前候选，其长度是前一条主线在 j-1 处的长度
加一，且只有 c=b_j 的候选接收该 key。逐候选归一化的标量/向量累积必须使用
一致的数值缩放。随后仅为主线 a_i 提交新的 tail 状态。

已有 decoder 固定 b=128 的周期性全量 rebuild **本身没有渐近 subquadratic
保证**，详见 [v3 decode 复杂度](../../dism_v3/README.md#decoding)。新增候选
不会自动修复这个问题。需要结合候选转移索引、summary 查询界和随历史长度
增长的 rebuild 间隔分析总成本。

例如，在固定向量维度/summary 参数、已加速候选转移的设计下，保守地把普通
步骤候选/tail 代价计作 O(Vb)，rebuild 计作 O(T log T + TV)，则 b 随 sqrt(T)
增长给出次二次的总工作量级 O(N^(3/2)(V+log N))。这是可行路线的条件性上界，
不是当前实现性能声明；共享 tail 分桶可进一步减少常数和重复工作。

需要同时保留 rebuild 峰值延迟、主线恢复、GPU 显存、CPU/GPU 通信、graph
失效/重捕获成本，不能只报告普通 decode step 的用时。

## 9. 数值和训练契约

- H/W/A 的直接概率域公式用于解释与小型 oracle。长链、正 tau 必须使用
  log-domain 或等价缩放，候选之间混合时处理各自的 scale/fallback。
- SQ/SK 只进入分子，不将其绝对值或正值偷偷放进分母。
- 所有 V 个候选有贡献的项都保留。概率截断、低秩近似、稀疏 K 都是独立变体。
- 保持 GDN、norm/gate/o_proj 的位置作为独立模型选择；v4 不是授权同时改动它们。
- 新 checkpoint 需要显式算子版本、历史路由类型、读取温度、归一化方式等
  model identity。不能把 v3 权重按新语义静默恢复训练。
- 若未来采用混合 soft/hard 训练，保留 packed-row sampling、RNG replay 和
  resume 状态完整性。历史路由温度与 hard-path 采样概率是不同参数。
- hard Q 的直接读取梯度与 soft/Jensen 历史梯度分别统计；K selector 梯度缺口
  必须单独诊断。历史 REINFORCE/STE 的失败不能直接推断该新算子的成败。

## 10. 验证阶段与当前证据

1. **语义 oracle。** 固定 hard 定义，实现独立后缀枚举、dense 主线递推、
   全候选读取；验证 one-hot 读取概率退回旧 hard，空匹配、reset、因果性和
   分母单位 fallback。算术历史、Jensen 历史作为两个命名清楚的 soft 版本。
2. **梯度与极限。** 对 Q/K/codebook/SQ/SK/V/tau 做 FP64 梯度检查；区分直接
   读取与历史梯度；检查硬端一致性、tie、尖锐分布、正负 tau 和长链缩放。
3. **真实权重诊断。** 固定同一语料和 checkpoint，比较旧 hard、旧 soft、新
   hard、新 soft 的输出/LM loss/长程读取；测试收益是否来自某类链长或符号桶。
   量化 Z 和 log(1+Z) 的动态范围、可压缩性与候选概率质量，先不启动完整训练。
4. **算法原型。** 验证全候选 SAM 转移、因果事件计划、共享候选分母；分别
   benchmark V 和 N 的缩放。decode 必须计入 rebuild 的总成本。
5. **训练与高性能实现。** 只有前述质量和成本成立后，再决定 soft K 路线、
   小规模训练、CUDA kernel、planner 和 CUDA graph 适配。不得将参考公式当成
   已完成的低维 kernel 设计。

截至本记录，仅完成临时 CPU/FP64 数学检查：

- 全候选 SAM 转移/LCA 后缀权重与 dense hard 主线递推：24 个小例子，包含
  tau<0、tau=0、tau>0，混合输出最大误差约 2.22e-16。
- soft 的候选输出枚举与 A * [p/(1+Z) K^T] 因式分解：最大误差约 2.78e-16；
  小例子的 Q/K logits autograd 梯度有限且非零。
- Jensen 历史下界、D 维恒等式、H^J<=H^A、读取系数下界、one-hot K 下读取
  Jensen 等号均通过小例子；D 维恒等式误差约 1.78e-15，硬 K 等号误差约 4.16e-17。

这些是会话中的小型数学核对，没有保存为正式回归测试，不构成大规模梯度、
数值稳定性或性能验证。尚无 v4 GPU kernel、sanitizer、训练或下游评估结果。

## 11. 参考与仍未解决的问题

- [v3 soft/hard 与 Jensen 推导](../../dism_v3/docs/SOFT_HARD_RELAXATION.md)。
- [v3 数学 oracle](../../dism_v3/python/flash_dism/reference/dism_v3_ref.py)。
- [v3 prefill builder](../../dism_v3/python/flash_dism/inference/csrc/prefill.hpp)。
- [v3 decode planner](../../dism_v3/python/flash_dism/inference/csrc/planner.hpp)。

在保留原始归一化设计的前提下，动态候选分母的计算量和数据流需要专门设计。
第 14 节给出满足澄清后的核心 kernel 约束的精确分阶段方案；其实际速度、稳定
缩放和 GPU 调度尚未验证。不能将“仍存在 V 相关计算”误写成“无法保持核心 kernel
低维”，也不能将它作为必须修改归一化语义的证明。

## 12. 后续决策：训练与推理均保留 hard 历史

用户追问 soft Q 历史在推理中如何处理后，明确区分历史与当前读取：

- 推理始终只保留 hard Q 主线和 hard K 符号串。
- 当前 Q 输出仍枚举全部符号并按完整概率混合。
- soft Q 历史不能用现有单一 SAM 查询状态表示；本文没有建立其精确快速推理。
- 主方向改为训练/推理都保留 hard 历史，避免默认重新引入历史路由的训练/推理
  偏差。第 3 节作为可选训练辅助和诊断 oracle，不是主方向的推理状态。
- hard 历史在训练中仍依赖当前权重算出的 argmax；普通梯度不穿过这些符号决策。
  当前 Q 的直接读取梯度仍可存在。K 选择与历史决策对未来的作用仍是学习缺口。

“训练中使用 soft K 读取”可以只软化当前匹配端点的兼容性，而不软化存储和延续
历史的 K 串。这两个作用必须分别命名，不能将这种 soft 读取误称为完整 soft 历史。

## 13. 新候选：去除候选分母，混合分子后做 RMSNorm

本节为用户提出的候选变体，主动改变第 2 节的归一化语义，尚未确认为默认方案。
按 **每个 head 内**、在加到 GDN 之前做 RMSNorm 来分析；归一化在所有候选混合
之后。若改成各候选先做 RMSNorm 再混合，会重新引入候选特有的非线性分母，
不具有本节的代数化简。

### 13.1 精确 hard 读取

两端都使用第 12 节的 hard 历史 H^H，令

\[
A_{ij}=\alpha(1+H^H_{i-1,j-1})\mathbf1[j\le i],\qquad
x_{ij}=\langle sq_i,sk_j\rangle v_j.
\]

先定义候选分子，不计算候选分母：

\[
N_i(c)=\sum_{j:b_j=c}A_{ij}x_{ij},\quad
U_i=\sum_c p_i(c)N_i(c)
=\sum_{j\le i}A_{ij}p_i(b_j)x_{ij}.
\]

\[
\boxed{o_i=\operatorname{RMSNorm}_{\epsilon}(U_i)}.
\]

因此全部 Q 候选的混合精确折叠成每个 pair 的 p_i(b_j) 因子，不再有动态的
Z_i(c) 或 E_K log(1+Z_i(c))。仍然只提交主线符号 a_i。

这不等于旧候选独立归一化后再 RMSNorm：后者相对重加权了各符号的贡献，本节
没有这种重加权。新算子也不承诺更接近 v3 soft。

### 13.2 对当前读取使用低维 Jensen，历史始终 hard

若训练时只软化当前 K 兼容性，保持同一份 hard 历史 A，则算术参考是：

\[
U_i^A=\sum_j A_{ij}
\underbrace{\left(\sum_c p_i(c)\rho_j^K(c)\right)}_{m^A_{ij}}
x_{ij}.
\]

可用 direction=true 的 Jensen surrogate：

\[
U_i^J=\sum_j A_{ij}m^J_{ij}x_{ij},\qquad
m^J_{ij}=\exp\left(\mathbb E_{\rho_j^K}\log p_i(c)\right)
=\exp(q_i^\top\bar E_j-L_i),
\quad \bar E_j=\sum_c\rho_j^K(c)E_Q(c).
\]

最终均对 U 做同样的 RMSNorm。K 兼容性取 one-hot 时 m^J=m^A=p_i(b_j)，
严格回到本节的 hard 读取；不要求将 Q 读取概率 p_i 一起硬化。

这次 **完整读取** 的低维化成立：词表相关工作留在 O(NVD) 的投影/插值，dense
pairwise score 宽度为 D，另计 SQ/SK/V 的向量维度和 quadratic hard 历史递推。
不能让 m^J 进入持续历史递推，否则又变成第 3/5 节的 soft 历史。

Jensen 仍是有偏的当前 K 读取 surrogate，且 RMSNorm 后不存在简单的逐分量
输出下界。它能为当前 K 兼容性提供梯度，但不解决历史 K 符号的离散 credit。

### 13.3 快速 hard 推理与数值稳定性

第 8 节全候选 SAM/event 结构仍可使用，候选 query 的输出系数现在直接乘 p_i(c)，
不再需要逐候选分母阶段。候选混合后再做每 token/head 的 RMSNorm。
这简化计划和读取，但没有证明候选数量开销可以完全消除；decode 的 rebuild
复杂度问题也仍然存在。

RMSNorm 的尺度不变性使公共正缩放在 epsilon=0、非零输入时可以抵消，包括
当前行的 exp(-L_i) 和当前步的 alpha。**历史 H^H 内部的 tau 仍改变各匹配链
的相对权重，不能因此删除 tau。** 词表投影仍用于符号选择和概率/嵌入计算。

有固定 epsilon 时不能随意丢弃整体尺度：若 U=s*U_tilde、s>0，则

\[
\operatorname{RMSNorm}_{\epsilon}(U)
=\frac{U_{\rm tilde}}{\sqrt{\operatorname{mean}(U_{\rm tilde}^2)+\epsilon/s^2}}
\]

（另乘相同的 learned RMS weight）。需跟踪 log-scale 或等价处理有效 epsilon，
不能先任意归一化 U 再套固定 epsilon 并声称等价。长链仍需要稳定的 log-domain
系数；RMSNorm 不能修复已经发生的溢出。signed readout 的抵消和接近零的 U
也需要单独检查，epsilon 控制该区域的输出和梯度。

上述按 head 抵消只适用于独立的 head 归一化。若只依赖现有的
RMSNorm(GDN + DISM)，DISM 的尺度会改变两条 mixer 的相对贡献，不能套用
RMSNorm(DISM) 的化简。norm 位置、是否保留外部 gate 和 head scale 需成为
显式模型配置。

### 13.4 质量风险：归一化移除的是信息，不只是计算

1. **候选之间的频率/链长平衡改变。** 原来的 1+Z_i(c) 会单独缩放每个符号的
   读取；RMSNorm 只缩放合并向量，不能恢复候选之间的比例。例：两个候选
   p=0.5，N_1=100 e_1、Z_1=100，N_2=e_2、Z_2=1；旧混合两方向比例约 1.98，
   新混合为 100。之后做 RMSNorm 也不会改变方向比例。
2. **整体匹配强度信号减弱。** 理想 RMSNorm 对公共正缩放不敏感。若某符号
   没有因果 K 端点，其 N_i(c)=0，增加它的 logit 只会通过 softmax 的分母统一
   缩小其他 p_i，可能让归一化输出不变。忽略 epsilon 时，该符号的直接读取
   梯度为零；但它仍可能赢得主线 argmax，破坏未来匹配。
3. **弱读取可能被放大。** 完全零的 U 保持零，但很小的非零读取会受到 norm
   与 epsilon 的强烈影响。RMSNorm 本身不能表达“这次匹配很弱所以少输出”。

这些是候选模型需要测量的行为，不是断言它一定失败。尺度不变性背景见
[RMSNorm 原论文](https://arxiv.org/abs/1910.07467)；符号梯度结论由本节算子推导。

一种可选对照是保留 **单个 head 标量** 的匹配强度 gate，而不是 V 个分母：

\[
M_i=\sum_j A_{ij}m_{ij},\qquad
o_i=\frac{M_i}{1+M_i}\operatorname{RMSNorm}(U_i).
\]

m 取对应 hard 或 Jensen 版本。该 gate 放在 RMSNorm 外，计算 M 可与读取融合，
稳定实现可用 sigmoid(log M)。它保留一部分整体匹配强度反馈，但不能恢复
每个符号内部的独立归一化，也不保证解决离散历史 credit。已有 learned gate
与这种基于匹配量的 gate 是两个不同选择，尚未决定采用哪一个。

### 13.5 此变体的最小验证集合

- 全候选分子枚举与按 pair 的 p_i(b_j) 加权逐项一致；one-hot K 的 Jensen 等号。
- FP64 梯度检查，包括接近零的 U、epsilon、无匹配符号、概率质量移出已出现符号。
- 与原候选归一化版本比较：符号频率偏置、不同链长贡献、长程读取、LM loss，
  不能只用输出 cosine 或单个 norm 指标决策。
- 原始 RMSNorm 与单标量匹配 gate 的对照；明确 norm 在 DISM/GDN 合并前后的位置。
- 现阶段只完成临时小型 CPU 数学检查：分子枚举与折叠公式最大误差约 1.33e-15；
  hard K Jensen 等号、soft K Jensen 下界通过；无匹配候选的理想 RMSNorm
  直接 logit 梯度约 -2.15e-16。未进行新训练、GPU 实现或实际吞吐测试。

## 14. 工程约束澄清：独立归一化可以精确分阶段、分块处理

用户指出，目标不是排除全部 V 维计算，而是将其移出复杂的对角递推/readout
前向与反向。按这个约束，原版候选独立归一化不存在之前暗示的架构障碍。
甚至可以先采用精确的算术读取，不必为了核心 kernel 的宽度而引入读取层面的
Jensen 近似。总 FLOPs、访存和调度是否合算是接下来要测量的问题。

### 14.1 前向依赖和独立 kernel 边界

保持 hard 历史，令 K_{jc}=rho_j^K(c)，P_{ic}=p_i(c)。K 可以是当前读取使用的
soft 分布；历史仍由固定 argmax Q/K 串决定。对 query block I：

1. 匹配阶段生成 A_I = alpha * (1+H^H_shift)，并施加因果 mask。
2. 独立词表 GEMM：Z_I = A_I K。
3. 逐元素计算 C_I = P_I / (1+Z_I)。
4. 独立词表 GEMM：B_I = C_I K^T。
5. readout 核心：O_i = sum_j A_ij B_ij <sq_i,sk_j> v_j。

步骤 2–4 承担词表维度；步骤 1 和 5 不需要 V 维向量。A/B 是传给核心的
标量 pair 系数，readout 可在内部形成 A*B，不必额外保存 M=A*B。

归一化依赖历史 A，因此不能全部放到递推之前。若已经把所有候选合成单个
D_v 维未归一化输出，候选分子的信息就丢失了，不能靠一个简单后处理恢复
逐候选分母。把归一化放到 readout 之后也可以，但那要求保留 N_i(c) 等候选
输出，代价可达 N V D_v；以上“递推后、最终 readout 前”分阶段更节省中间量。

现有融合的递推/readout 需要重新划分边界；本设计不是一个无需改 kernel 的
外部 wrapper。

### 14.2 无需完整 N x N 张量

令 query block 大小为 B_q。最直接的实现仅保留：

- A_I、B_I：各 B_q x N，可分配复用 scratch。
- Z_I、C_I：各 B_q x V。
- 输出块：B_q x D_v。
- K/P 概率及可选择保存的 Z：N x V，属于词表侧数据，不是 N x N pair 张量。
- hard 路由元数据、原始 SQ/SK/V 及必要的梯度累积。

因此 pair scratch 是 O(B_q N)，固定 B_q 时对 N 线性，而不是 O(N^2)。
例：B=1、H=12、N=8192、B_q=64，FP32 的一个 A_I 或 B_I buffer 为 24 MiB；
同条件完整 N x N buffer 则为 3 GiB。全序列 Z 在 V=512 时为 192 MiB，可
选择保存或重算。上述数字不包括模型、梯度、其他层和 kernel workspace。

还可以进一步做 key tiling：先遍历 key tiles 累计 Z_I，得到 C_I 后，再遍历
key tiles 重算 A 的 tile、计算 B 的 tile 并累计输出。这样 pair scratch 可降到
O(B_q B_k)，代价是重算和更多调度。不能声称因此去掉了全历史概率/元数据存储。

关键依赖是**完整的候选分母尚未累计完之前，不能最终确定该 query 的读取
系数**。它要求多阶段/多遍处理或缓存当前 query block，但不要求保存整张矩阵。

### 14.3 反向也可以保持相同边界

令 S_ij=<sq_i,sk_j>，E_ij=<g_i,v_j>，M=A*B，T=S*E。对 block I，readout
反向先给出：

\[
dA_{\rm read}=T\odot B,\qquad dB=T\odot A,
\]

以及 SQ/SK/V 的普通梯度。词表阶段的反向为：

\[
dC=dB K,\quad dK_B=dB^T C,\quad
dP=\frac{dC}{1+Z},\quad dZ=-\frac{dC\odot P}{(1+Z)^2},
\]

\[
dA=dA_{\rm read}+dZ K^T,\qquad dK=dK_B+A^T dZ.
\]

所有 V 维收缩仍是独立 GEMM。核心的 reverse recurrence 若被采用，只接收
标量 dA，不接收 V 维梯度向量。词表投影和 softmax 的反向同样放在独立阶段。

每个 query block 可重算 A/Z/C/B，消费该 block 后释放 scratch，向全局 dK、
dSK、dV 累加。更细 key tiling 时，先累计 dC，得到完整 dZ 后再遍历 key tiles
计算分母对 dA/dK 的贡献；dA_read 可重算，不必为所有 query 留存。

普通 autograd 若自动保存每个 block 的所有 pair 张量，总量仍可能回到 N^2。
生产实现必须明确 custom backward/checkpoint/recompute 策略，不能仅凭前向
写了 block 循环就宣称训练显存线性。同理，保存每个 query block 的 N 长度
递推边界，总量也有 O(N^2/B_q)；这不是严格的线性存储方案。

### 14.4 Hard 历史允许更直接的重算和 tau 梯度

本方案主方向是 hard 历史，tau 按 head 固定。令 ell 为 Q 主线前缀 i-1 与
K 前缀 j-1 的最长公共后缀长度，则：

\[
A_{ij}=\sum_{t=1}^{\ell+1}e^{t\tau},\qquad
\frac{\partial A_{ij}}{\partial\tau}
=\sum_{t=1}^{\ell+1}t e^{t\tau},\qquad j\le i.
\]

所以 A 不必依赖保存的 dense 递推轨迹：可从 hard 后缀元数据按块重建，用
dA*dA/dtau 累加 tau 梯度，无需为 Q/K 硬标签反传。SAM query 状态、匹配长度、
K endpoint 和 suffix-link-tree LCA 可支持这种重建；binary lifting 等方案的
元数据可为 O(N log N)，不会落地 N^2 状态。

这是数学上的重算入口，不是已实现的高效 GPU pair 查询。随机树访问、LCA
成本、重算调度与原对角扫描的性能需要比较。若将来重新采用 soft 历史，这个
硬后缀闭式不能沿用，需要真正的 reverse recurrence 和检查点策略。

### 14.5 数值边界与验证状态

分块解决存储，不自动解决指数动态范围。各候选独立归一化时，不同符号的
匹配量可能相差极大，却在归一化后都有重要贡献。不能以同一个全行最大值
缩放 A 后，允许弱符号桶下溢为零，再声称归一化会恢复它。应设计符号/块尺度、
稳定累计或 log-domain 处理，并验证 fallback=1 的绝对尺度。

本节公式是在实数域的精确分解；尚未选择最终的 BF16/FP32 稳定表示。中间
词表运算也不能仅因从核心 kernel 移出，就忽略 FLOPs、访存、kernel 启动或
CUDA graph 编排成本。

临时 CPU/FP64 检查已经验证：

- 全候选独立归一化与分阶段前向误差约 2.22e-16。
- 分阶段 autograd，以及显式 readout -> 词表运算 -> reverse recurrence 的
  Q/K/codebook/SQ/SK/V/tau 梯度，与候选 oracle 的最大误差约 8.88e-16。
- query-block 前向及显式反向（用独立后缀枚举重建 A 和 tau 导数），最大
  前向误差约 2.22e-16，梯度误差约 1.11e-15；block pair workspace 为 4 x 13，
  未在被检验算法中分配 13 x 13 pair 状态。参考/验证代码不作为生产存储承诺。

这些检查不包含 GPU 实现或吞吐测量。本节恢复了原版独立归一化作为可行的
工程候选；RMSNorm 变体仍值得比较，但不再以“无法隔离 V 维计算”作为其必要性。

## 15. 基于 v3 的改动范围与去除 SQ/SK 的候选

### 15.1 已检查的 v3 结构

对照 `dism_v3/src/forward.cu`、`include/forward/readout.cuh` 和
`python/flash_dism/compiler.py`：当前前向在一次流式遍历内完成 score MMA、
对角扫描、SQ/SK MMA、在线单行归一化和 value MMA。backward 则依赖保存的
row normalizer、delta、扫描边界，再接 vocabulary interpolation backward。

因此第 14 节不是在现有 fused kernel 后面加一个除法就能完成的修改。单行
normalizer/delta 不能直接替代 v4 的逐候选 dZ；需要重新划分前后向和保存契约。

| 部分 | 可复用基础 | v4 必须处理的变化 |
|---|---|---|
| 词表投影/插值 | 投影、LSE/argmax、tile 和精度经验 | 分块概率的导出/重算、直接概率梯度；旧接口主要返回插值向量而非完整概率 |
| hard 匹配 | label/reset 处理、扫描原语、SAM/后缀元数据 | 按 block 提供历史延续 A，或从后缀长度重建；不把当前软读取写回历史 |
| dense readout | value MMA、TMA/调度基础 | 接收外部 pair 系数，不再执行旧的单行归一化；有/无 SQSK 分开评估 |
| backward | 局部向量梯度的数学和基础代码 | 词表 dZ/dC 路径、block 重算、tau 与路由梯度边界、custom autograd |
| hard prefill/decode | K 索引、SAM 拓扑、历史缓存框架 | 每位置多候选 query、共用时间戳、候选混合与分母、主线提交接口 |
| planner/graph | 已验证的同步协议和缓冲管理经验 | 新任务与 buffer 形状、候选统计、capture/replay 和生命周期验证 |

范围判断：小规模分块正确性/质量原型可复用现有基础，用普通 GEMM 实现词表侧，
无需先重写所有低层流水线；达到当前 v3 水平的性能、完整反向、varlen、planner
和 CUDA graph 支持则属于较大改动。没有实测之前不提供工期或性能倍数承诺。
v4 应单独实现/验证，不能通过修改 v3 默认数学来“兼容”两种语义。

### 15.2 去除 SQ/SK 的数学含义

对于原版候选独立归一化，去掉 SQ/SK 等价于将 <sq_i,sk_j> 固定为 1：

\[
\mu_i(c)=\frac{\sum_{j:b_j=c}A_{ij}v_j}{1+Z_i(c)},\qquad
o_i=\sum_c p_i(c)\mu_i(c).
\]

这成为对“当前 hard 后缀历史条件下，各符号的 value 聚合”的概率读取。Q 的
全候选分布已经提供当前 token 的连续选择能力，所以 SQ/SK 不再是让当前读取
具有连续选择能力的唯一通路；去除它们是有实质依据的候选，不是只为省参数。

但两者不完全重复。对 hard K 中同属符号 c 的位置 j、l，固定历史时：

\[
\frac{M_{ij}}{M_{il}}=\frac{A_{ij}}{A_{il}},
\qquad b_j=b_l=c.
\]

当前 p_i(c) 在比例中消去。SQ/SK 能额外根据当前连续内容区分同一符号内的
key，且允许 signed token 权重。移除后仍有后缀链长的区分，但没有这项额外
的当前内容细分能力。大词表减少碰撞、多 head、GDN 的连续混合都可能补偿，
需要实际验证，不能由 soft Q 的存在推出完全等价。

原版归一化下无 SQ/SK 的有效 token 权重非负、总和不超过 1，因此输出是 values
与零 fallback 的凸组合；value 和后续输出投影本身仍可以带符号。这种有界读取
与第 13 节的 RMSNorm 输出是不同模型选择，应分别做对照。

### 15.3 去除 SQ/SK 对实现的影响

dense 路径变为 O=M V，反向主要是 dM=dO V^T、dV=M^T dO，删除 SQ/SK MMA、
dSQ/dSK 和相应投影/激活/缓存。词表归一化分阶段和 dZ 的链路仍然需要，不能
因为 readout 简化就沿用 v3 的旧 delta 公式。

hard event 路径的收益更大。v3 的状态更新为：

```
state[R,Dv] = decay * state + weight * outer(SK[j], V[j])
output[i] += coefficient * SQ[i] @ state
```

无 SQ/SK 时变为：

```
state[Dv] = decay * state + weight * V[j]
output[i] += coefficient * state
```

反向也从矩阵外积状态变成向量状态。R=32 时，每个聚合状态的元素数下降 32 倍，
大量 outer-product/MMA 工作、寄存器/summary 存储随之消失。这不是端到端 32 倍
提速：候选数、词表工作、scalar planning、访存、其他模型层和调度仍有成本；
向量执行器也需要新 specialization，不能直接把现有只支持 R16/32 的 kernel
参数改成 R=1。

以 width=768、12 heads、R=32、无 bias 的独立 SQ/SK 投影为例，可去掉每层
589824 个参数；6 个 hybrid 层共 3538944 个。这个数字不包括其他可能的 readout
前端结构，也不授权自动将节省参数补到 FFN 中。

建议先以“去除 SQ/SK、保留候选独立归一化”的 v4 oracle/分块原型作为简化
基线，并将完整 SQ/SK 版本作为表达能力对照；质量结果决定最终 kernel 形态。
本节是设计阶段记录；第 17 节给出后续分块原型的实现与验证边界。

## 16. 第一版 PyTorch reference

设计第 12–15 节的更新先提交为 `069d5af`，随后新增独立包
`dism_v4/python/flash_dism_v4`。使用方式和精确接口见 [v4 README](../README.md)。

- `dism_v4_ref` 接收 BNHC 符号 logits；默认无 SQ/SK，采用原版逐候选归一化。
- `dism_v4_from_embeddings` 提供 per-head Q/K codebook 适配，不隐式缩放 logits。
- `hard`、`soft` 和 `jensen` 模式仅改变当前 K 读取，历史始终提交 argmax Q/K。
  Jensen 对完整归一化系数作下界，保留动态分母，不软化历史。
- 支持可选 SQ/SK 对照、batch/head、多段 B=1 packed 输入、独立 query reset。
- 使用 log-domain 历史和逐候选 fallback 归一化，避免空候选 backward NaN。
- CPU 37 项测试通过，包括独立后缀枚举与梯度、FP64 gradcheck、因果/边界、
  one-hot/Jensen 界、长链极端动态范围、dtype 和 RNG 检查。

这是小规模 correctness oracle，普通 autograd 的候选保存量可达 O(B H N^2 V)。
不能把此 reference 的执行方式或性能当成分块 kernel 的内存/复杂度承诺。
reference 完成时尚未接入模型或运行 GPU kernel；后续实现如下。

## 17. 2K 分块训练原型

实现文件为 `flash_dism_v4/history.py`、`blocked.py`、`module.py`；
对应 nanochat 架构单独注册为 `dism_v4_gdn_hybrid@1`。具体命令、数值/恢复限制、
benchmark 记录见 [README](../README.md)。这是训练与 loss evaluation 原型，
不包含快速推理、planner 或 CUDA graph。

- history 的前缀扫描只读取 hard Q/K 符号与文档边界，不读取 C 维概率向量。
  利用第 14.4 节的几何和公式，同时生成 log A 和 tau 导数。
- 当前 soft 读取采用精确的全 K 概率、候选独立分母；当前 hard 读取按 hard K
  符号分桶。两者都混合全部 Q 候选，都只提交 argmax Q/K 历史。
- 词表归一化位于 history kernel 外部。无 SQ/SK，value readout 是 M V。
  默认保留原版独立分母，不采用第 13 节的无分母 RMSNorm 变体。
- 每个 block 只保留 Bq×N 成对 scratch。反向按前向顺序重新生成 history，
  显式累积 Q/K/V/tau 梯度，无 N² 激活或 N²/Bq 边界 checkpoint。
- hard 逐符号 log 缩放；soft 共同缩放 GEMM 有极端 logits 的逐行 log-domain
  fallback。fallback 保证不因丢弃弱符号桶而改变公式，但会明显降低吞吐。
- nanochat 保留 pure GDN、后归一化、gate、value residual、FFN/CE 和训练状态
  机制，仅替换 hybrid DISM 并移除 SQ/SK。训练 row 内多个文档共享该 row 的
  hard flags，各文档仍单独隔离匹配历史。

默认 12 层、width=768、H=12、C=512、`gggddddddggg`、无 SQ/SK 时为
119934144 参数，MXFP8 FFN/optimizer、BF16 CE。RTX 5090 上 B=4、N=2048、
block=64、FP32 readout 的短程合成训练测得 soft 10.42k / quadratic hard
28.99k / 50% 混合 15.04k tok/s，峰值 allocated 不超过 12.18 GB；
soft/hard loss eval 为 52.20k / 62.98k tok/s。没有据此推断完整预训练质量。

GPU 前向/反向对 FP64 oracle、跨 block 长匹配链、packed 文档与 row 采样、
自动混合精度、极端候选缩放均有测试。新 history kernel 的 memcheck、
racecheck、synccheck 均通过，所测三个 block specialization 无 CALL/spill。
这不覆盖完整 FLA/MXFP8 模型的 sanitizer。模型恢复仍有 BF16/atomic 数值
非确定性；serialization/schema/alias 与相同梯度下的 optimizer 更新另行验证，
不宣称 bitwise 训练轨迹一致。
