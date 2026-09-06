# Dism v2 kernel 执行计划

状态：已开始阶段 1 的独立 GLX 兼容性与寄存器实验；尚未实现 attention kernel。结果见 `experiments/glx_scan/README.md`。

## 目标基线

优先平台为 RTX 5090 / sm120。固定长度 BF16 输入，D、DV 独立取 32/64/128，支持非 tile 对齐 N。前向融合打分、对角递推、online softmax 和 PV，不物化 NxN 中间矩阵。生产路径在 warp 内生成可重放的逐行随机决策。最终扩展 varlen 和 sm90。

暂定使用 ThunderKittens warp MMA/TMA + GLX，保留三阶段前向和三阶段反向的 checkpoint 框架。选择该框架是为了并行 query chunks 并复用既有推导；checkpoint 长度和 CTA 布局仍需实测决定。辅助状态空间预计随 BH*N*ceil(N/checkpoint) 增长，必须记录实际峰值，不能称为线性空间。

## 阶段 0：环境、接口与可重放语义

- 检查本地 PyTorch、Triton、CUDA 编译器、ThunderKittens 与 GLX 的可用版本，建立可复现的 sm120 构建入口。
- 固定 core 接口：选定方向的左右 GEMM 输入、对应 row/column LSE、q/k labels、rtau、hard_prob、v 及 RNG 元数据。列出自然对数与 log2 的转换边界。
- 定义 forward 保存项与 backward 输入项：O、逐行 log-normalizer、扫描 checkpoints、direction 和 seed/offset；估算九种维度下的临时内存。
- 定义 warp 内 RNG 的逻辑行 counter 与 PyTorch generator 消费方式，保证跨 pass、key tiles、重算及后续 varlen 的一致性。检查 generator/CUDA Graph 兼容需求，记录首版边界。
- 定义生产接口对 hard_prob 广播形状和输入 stride 的支持范围；不静默缩减 reference 语义。

验收：构建入口可用，接口/状态/RNG 约定明确。当前只确认设备可见为 RTX 5090、compute capability 12.0，未完成工具链验证。

## 阶段 1：MMA–TMA–GLX 布局和数值原语

- 为候选 16x64、16x32、32x32、16x16 warp tiles 建立列布局表，验证 TMA 随路置换、MMA accumulator 到 GLX 的映射及 PV 所需逆映射。优先争取 warp_k_size=64，32 可回退，128 后续按资源结果探索。
- 验证 roll/scan/unroll、reverse scan 和 summary-only reduce，覆盖左右/上下边界、多 warp 拼接与因果/tail masks。
- 提供语义正确的 FP32 log-affine op 和实数域 reverse-affine op；验证 identity、负无穷、hard 不匹配的零映射。
- 初版 logM 和 `(logM,logM)` 均用 FP32。使用原生 inclusive scan 缩短原 score 生命周期，分别记录独立扫描与完整融合 kernel 的寄存器占用；后续再评估 BF16。
- 建立精确或高精度基线后，单独接入 lse.cu 候选近似；不要让近似误差干扰布局调试。

验收：小 tile 与跨 tile oracle 一致，mask 和边界无 NaN 污染，必要的 sanitizer 检查通过；确认生成代码中的布局转换与 spill 情况。

## 阶段 2：fixed-length 核心前向

1. 摘要 pass：并行 query checkpoints，计算 GEMM + score，warp 内生成行决策，GLX reduce 输出对角边界摘要。
2. 边界传递 pass：合成各 checkpoint 的入边界；先采用独立 diagonal groups 内顺序传递的基线。
3. 输出 pass：重算 score 和行决策，注入边界做 GLX inclusive scan，融合 online softmax 与 PV。初始化 max=0、denominator=1、numerator=0 以包含固定 fallback。

- 先固定 direction 和 hard_prob=0 排查主路径，再加入 hard_prob=1、混合和随机 direction。
- 不保存 logM、W 或行 mask；检查不同 pass 的 RNG 选择完全一致。
- 支持全部九种 D/DV 组合及非对齐 N，记录临时存储和输出 dtype。

验收：对照相同 interpolation、direction 和行决策的 reference，所有目标形状前向正确；完全不匹配行输出为零，长匹配链数值稳定。

## 阶段 3：核心反向

数学基线（自然对数语义）：

```
P[i,j] = exp(W[i,j] - L[i])
delta[i] = dot(dO[i], O[i])
E[i,j] = P[i,j] * (dot(dO[i], v[j]) - delta[i])
G[i,j] = E[i,j] + sigmoid(W[i,j]) * G[i+1,j+1]
```

- 预处理 delta，重算前向 tile 并计算 dV 和反向 affine 摘要。
- 反向传递 checkpoint 边界，再重算并 reverse scan，得到 score 梯度。
- 计算左右 GEMM 输入梯度、对应 LSE 的 row/column reduction 和 rtau 梯度；soft mask 只作用在 soft score 的局部导数，不屏蔽递推链本身。
- 初始跨 CTA 累加可使用 FP32 atomics 或显式 partial buffers，按实测选择并说明确定性；不使用 BF16 原子累加作为梯度精度基线。

验收：core 全部可微输入对照 autograd；覆盖混合 hard/soft 跨 checkpoint 梯度传播、所有维度组合、tail 和两种方向。区分近似算子反向策略与精确数学梯度。

## 阶段 4：voc_dism 全链路

- 接入现有 EmbInterpFunction，先复现并解决 embedding backward 的重复 dq/dk 写入问题。
- 正确传递插值 embedding 与 LSE 梯度，合并 score 的直接 q/k 梯度和 embedding 阶段梯度，完成六个输入的 autograd 接口。
- 检查包导入、stride、dtype、sm_scale、词表 tail、top-1 tie 和随机调用语义。
- 分别对照纯 torch interpolation 和 kernel interpolation，报告前级量化与 core 误差。

验收：voc_dism 前向和 q/k/v/rtau/q_voc/k_voc 梯度全链路通过；随机前向可重放，backward 不额外消费随机数。

## 阶段 5：sm120 性能迭代

- 建立按 B/H/N/V/D/DV 分组的基准，序列长度覆盖短序列到显存预算内的长序列；具体训练代表形状由用户场景补充。
- 分别测量 embedding、摘要、传递、输出、各反向阶段以及端到端时延，包含临时缓冲和 RNG 的实际路径。
- 调整 GLX shape、checkpoint 长度、warps/CTA、TMA stages、producer/consumer 分工和寄存器存活区间。
- 评估 LSE 候选近似的输出/梯度误差、长链稳定性及端到端收益。SASS patch 如有必要，作为独立、可验证的实验路径。
- 根据瓶颈再决定并行化边界传递、减少 GEMM 重算、消除原子竞争或采用 CUTLASS/CuTe；不预设必须重写框架。

验收：发布可复现的 correctness/performance 表与显存开销，不设未经测量的性能承诺；优化后的必要回归检查通过。

## 阶段 6：varlen

- 增加 packed tokens 与 sequence offsets 接口，定义与 fixed-length 逐序列调用等价的数学结果。
- checkpoint、因果 mask、scan 边界与 RNG counter 严格隔离不同序列，处理零/短序列及混合长度。
- 避免按最大长度建立所有序列的 checkpoint 缓冲；按实际序列长度规划存储和调度。

验收：对照逐序列 reference，前向/反向无跨序列状态或梯度泄漏；覆盖九种维度组合和长度边界。

## 阶段 7：sm90

- 共享数学、RNG、GLX 和测试，建立独立 sm90 构建及配置。
- 先运行兼容的 warp MMA 路径，再评估 Hopper 上的 tile、TMA、warpgroup 调度以及原生 WGMMA 的实际收益和布局代价。
- varlen 与 fixed-length 都需实际设备验证；设备暂不可用时只报告编译结果，不称为 sm90 已验证。

验收：sm90 实机正确性、sanitizer 和性能记录齐全。若 sm90 设备提前可用，可在阶段 5 后提前进行 fixed-length 移植，再共同补齐 varlen。
