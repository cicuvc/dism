# CUDA embedding插值前向计划

用户随后授权实现：E1/E2/E3已完成首版，E4已有初步API及CUPTI计时，实测与剩余项见
[EMBEDDING_CUDA.md](EMBEDDING_CUDA.md)。端到端主线此前提交8a90cba，
现有Triton前后向仍为默认路径及对照。优先sm120，之后独立配置/验证sm90。
后续追加D32/64的B_V128已实现并验证零spill，已有热cache CUPTI kernel计时；
完整V场景较64有收益但未反超融合Triton，短V尾部变慢，保留为显式低层选项。

## 1. 数学与接口

用E_q/E_k表示每head词表[H,V,D]，避免与token q/k混淆：

- Q=sm_scale*q*E_q^T，P_q=softmax(Q)，k_from_q=P_q*E_k。
- K=sm_scale*k*E_k^T，P_k=softmax(K)，q_from_k=P_k*E_q。
- 输出两个BF16插值、两个FP32自然对数LSE、两个FP32 top1概率、
  两个int32 argmax索引，顺序与emb_fwd_wrapper/InterpolationResult一致：
  (q_from_k,k_from_q,k_lse,q_lse,k_top,q_top,k_index,q_index)。
- 不使用Dism的零value fallback：embedding分母只有词表softmax的sum。
  不引入direction/hard RNG；一次调用计算两路，供后续core选择全局方向。
- q/k和词表BF16，D=32/64/128；embedding自身不依赖DV。V与N支持非tile整数倍。
  FP32 score、online max/denominator、output accumulator，第一版PV权重BF16。
  输入sm_scale自然域，只在score处乘一次LOG2E；最终LSE乘LN2返回自然域。
- argmax ties取最小词表索引，包括tile内和跨tile；padding score为-inf，
  不沿用Triton的有限-1e5哨兵。V>0，无效token行不写回但完整参与同步。

## 2. CTA与warp角色

每CTA处理一个(batch,head)的同一64个token位置，不是两组共128个token：

| 物理warp | 常驻token操作数 | 16行FP32 accumulator | score所读词表 | PV所读词表 |
|---|---|---|---|---|
| 0–3 / WG0 | k，各16行 | acc_qo / q_from_k | E_k | E_q |
| 4–7 / WG1 | q，各16行 | acc_ko / k_from_q | E_q | E_k |
| 8 | 不持有计算tile | 无 | producer：加载E_q、E_k | 两路共享加载结果 |
| 9–11 | 无 | 无 | 参与producer组setmaxnreg | 不做冗余加载 |

warp w与w+4对应同一16个token位置。8个compute warps完全独立，无scan或组间
数值边界passing；只共享词表输入ring的ready/free协议。
12 warps/CTA，producer group整体dec<40>，两个compute groups各inc<232>，
放在各自长期角色分支内。40/232是初始配置，不作为已达成的资源指标。
初始化q/k staging转入寄存器后统一同步，再复用shared staging为词表ring。

## 3. Tile与词表布局复用

起点B_N=64个token/CTA，warp score tile=16xB_V，B_V=64；32作为回退，
128只在完整资源测量后探索。沿V串行online softmax，不做split-V初版。
每槽只有两个BF16 shared tile：E_q[B_V,D]和E_k[B_V,D]。
同一E_q给WG1的score及WG0的PV，同一E_k给WG0的score及WG1的PV。

沿用TK MMA/TMA；普通逻辑布局，不需要Dism的GLX列permutation或row-dependent roll。
score通过row register tile做AB^T，PV通过col-layout读同一shared tile做AB；
先验证同一shared存储的两种register视图，无额外global转置/重排。
TMA描述符保留独立V/head边界，不能将词表tail越界当成下一head的合法数据。
可按32/64-feature宽度分段TMA填入TK shared tile，以匹配64B/128B swizzle。
首版词表tail可由producer guarded load清零并由compute mask score为-inf；
完整tile用原生shared::cta TMA。未padding分配需sanitizer验证。

带宽预算针对“两个独立方向分别执行FA”：
每个词表step由4个BF16 tile读取变为2个，即8*B_V*D bytes→4*B_V*D bytes。
这是词表请求流量的二分之一，不是整个kernel/HBM实测带宽或时间保证。
现有Triton源码本来就融合两方向并各load一次词表，不能宣称相对它还能凭此再减半；
与它比较必须测L2/HBM流量、spill和实际时延。

## 4. 双缓冲流水线

producer warp8负责两槽交替预取，每槽包含一对词表tile：

1. slot可复用后，producer登记两张词表总TMA字节数并发出异步加载。
2. compute warps等待同一个ready phase，分别做自己的score、online更新与PV。
3. 每个compute线程仅在最后一次shared词表读取结束后arrive free；
   不能在score结束时提前释放，因为相反方向的PV仍会读同一对词表。
4. producer等待256个compute线程的free，再覆盖该槽。尾部同样发布ready，
   不足64行的CTA中无效compute warps也执行free，不提前return。
5. 退出前确保所有预取和shared读者完成，再复用空间或结束CTA。

按现有已验证风格，ready初始计数32（producer warp），free为256；
lane0发TMA/expect，其余producer lanes参与arrival，tail所有writer发布各自写入。
精确barrier phase与transaction byte数需在流水probe独立验证。
两组compute不要求锁步做相同GEMM阶段；无额外warpgroup串行依赖。
初版不把score/PV分给不同计算warp，以免引入P的shared交接与角色不均衡。

双缓冲词表数据占用=2槽*2词表*B_V*D*2 bytes。
B_V64时D32/64/128分别16/32/64 KiB；barrier/对齐另计。
初始64行q/k staging占8/16/32 KiB，与ring做生命周期复用，不永久相加。
输出优先warp coalesced store，不为epilogue保留两份完整输出shared tile。
完整CTA的实际shared大小、寄存器分配和spill以编译结果为准。

## 5. Online softmax与寄存器生命周期

每warp常驻一个BF16 token tile、一个FP32输出tile，以及16行m/l/argmax状态。
它们的纯数据寄存器下限（不含复制和临时值）分别约D/4、D/2个32-bit寄存器/thread。

每个词表tile：

```
S = score_mma(token, key_vocab) * sm_scale * LOG2E
tile_max, tile_argmax = row_max_argmax(S)
m_new = max(m, tile_max)
alpha = exp2(m - m_new)
p = exp2(S - m_new)
l = alpha*l + row_sum(p)
acc = alpha*acc + mma(bf16(p), value_vocab)
m = m_new
更新argmax（分数相等时保留更小的全局索引）
```

初始化m=-inf、l=0、acc=0；对无效行显式保护-inf减-inf，p置零。
最后out=acc/l，LSE=(m+log2(l))*LN2，top1_prob=1/l。
不保存全局logits或P，也不为argmax长期保存整个原score tile。

D128不将64x128词表两种operand同时长驻寄存器：
先完成score后释放key fragments，再读PV value fragments；必要时score沿D按32/64分段，
PV沿输出D按32/64分段，只让一个fragment短暂存活。
这只是初始实现结构，不将资源下限外推成实际零spill承诺。
第一版BF16 P一次MMA，与现Triton量化位置相近；不默认加入高位+残差双MMA。
如有精度失败先单独量化定位，保留FP32 oracle与实际Triton对照，不能改数学掩盖。

## 6. 执行顺序与验收

E1：单warp score→online softmax→PV，以及argmax/自然域LSE/top1输出。
验证D三种、V=1/31/32/63/64/65/127/128/129/257、N尾部、跨tile相等最大值，
并证明共享词表的两种加载视图与逻辑元素映射正确。

E2：按上述12-warp方案融合两方向、接TMA双缓冲及寄存器重分配；
单缓冲只保留作正确性/性能对照。检查所有SASS CALL/UTMALDG/USETMAXREG，
记录每D/B_V的寄存器、spill、shared以及尾部协议的三类sanitizer。
出现超出既有授权范围的新资源问题先报告，不自行展开无界调参。

E3：接口替换验证。先提供显式CUDA embedding forward选项，backward继续用现有
Triton wrapper，确保保存的是本次CUDA前向的插值/LSE，不混用Triton前向状态。
八项输出顺序保持不变；随后重跑六输入autograd、rtau符号及既有精度测试。
精度通过范围明确后再讨论默认切换，不在验证前移除Triton基线。

E4：完整embedding前向benchmark，与当前融合Triton以及双独立方向基线分别比较。
记录B/H/N/V/D、计时范围、冷/热cache、显存流量、资源及量化误差；
embedding不含DV，接入core后再覆盖九种D/DV端到端组合。
不把“加载请求减半”直接当作2倍加速，也不把sm120结果外推至sm90。

CUDA embedding backward不在本轮范围内；先完成前向正确性与资源/性能报告。
