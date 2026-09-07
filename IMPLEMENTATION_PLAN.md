# Dism v2 kernel 执行计划

状态：阶段 1 的独立 GLX、TMA 列置换、融合单 stripe 和三阶段 checkpoint 实验已通过；当前更新主方案并准备阶段 2 实现，尚未实现正式 attention kernel。结果见 `experiments/glx_scan/README.md`、`experiments/glx_tma_permute/README.md` 和 `experiments/glx_fused/README.md`。

## 目标基线

优先平台为 RTX 5090 / sm120。固定长度 BF16 输入，D、DV 独立取 32/64/128，支持非 tile 对齐 N。前向融合打分、对角递推、online softmax 和 PV，不物化 NxN 中间矩阵。生产路径在 warp 内生成可重放的逐行随机决策。最终扩展 varlen 和 sm90。

使用 ThunderKittens warp MMA/TMA + GLX，保留三阶段前向和三阶段反向的 checkpoint 框架。当前前向主方案为32行 checkpoint、128行 CTA、双交错 compute warpgroup，不采用四个连续 warp 的逐级串行链；暂不拆 upsweep/downsweep。辅助状态空间随 BH*N*ceil(N/checkpoint) 增长，必须记录实际峰值，不能称为线性空间。

不兼容仓库内旧接口；为 v2 建立独立入口、扩展和测试。旧 CUDA/Triton 仅复用有验证依据的组件，不接入其 RMSNorm、旧布局或 Triton passing。现有 `dism_v2/dism_ref.py` 和 `emb_kernel.py` 的用户改动保留。

## 阶段 0：环境、接口与可重放语义

- 检查本地 PyTorch、Triton、CUDA 编译器、ThunderKittens 与 GLX 的可用版本，建立可复现的 sm120 构建入口。
- 固定 core 接口：选定方向的左右 GEMM 输入、对应 row/column LSE、q/k labels、rtau、hard_prob、v 及 RNG 元数据。列出自然对数与 log2 的转换边界。
- 定义 forward 保存项与 backward 输入项：O、逐行 log-normalizer、扫描 checkpoints、direction 和 seed/offset；估算九种维度下的临时内存。
- 定义 warp 内 RNG 的逻辑行 counter 与 PyTorch generator 消费方式，保证跨 pass、key tiles、重算及后续 varlen 的一致性。检查 generator/CUDA Graph 兼容需求，记录首版边界。
- 定义生产接口对 hard_prob 广播形状和输入 stride 的支持范围；不静默缩减 reference 语义。

验收：构建入口可用，接口/状态/RNG 约定明确。当前已实测 RTX 5090 / sm120、CUDA 13.1 的 nvcc 与三类 sanitizer；生产接口与 PyTorch generator seed/offset 消费约定尚未完成。

2026-09-07 环境准备：使用 conda `blkw`，解释器 `/home/cicuvc/miniconda3/envs/blkw/bin/python`；Python 3.12.12、PyTorch 2.13.0+cu130、Triton 3.8.0、pytest 9.0.2，ninja 可导入，GPU BF16 matmul 已实测。PyTorch CUDA 13.0 与系统 nvcc 13.1 不完全一致；现已在该组合完成最小 sm120 PyTorch CUDA 扩展的编译、加载、执行：BF16 bit-copy、137元素尾部、空输入和非默认 stream 通过。这不替代正式 TK/GLX 扩展的验证。

准备命令（仓库根目录；也可 `conda run -n blkw ...`）：

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v2/dism_ref.py
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_reference.py
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_build.py
```

实测 reference 自带 smoke test、80个新增 oracle 准备用例和1个扩展构建用例通过，扩展 bit-copy probe 的 memcheck 为0 errors。oracle 用例覆盖九种 D/DV、两个 direction、显式 soft/hard/mixed mask、16/32/64/128边界邻近长度的零输出 fallback；固定 direction + 显式 mask 不推进 generator 状态。这些测试仅验证 oracle/输入场景和工具链准备，不代表新 attention kernel 正确性，也未验证正式 RNG 契约或 embedding kernel。

## 阶段 1：MMA–TMA–GLX 布局和数值原语

- 为候选 16x64、16x32、32x32、16x16 warp tiles 建立列布局表，验证 TMA 随路置换、MMA accumulator 到 GLX 的映射及 PV 所需逆映射。优先争取 warp_k_size=64，32 可回退，128 后续按资源结果探索。
- 验证 roll/scan/unroll、reverse scan 和 summary-only reduce，覆盖左右/上下边界、多 warp 拼接与因果/tail masks。
- 提供语义正确的 FP32 log-affine op 和实数域 reverse-affine op；验证 identity、负无穷、hard 不匹配的零映射。
- 初版 logM 和 `(logM,logM)` 均用 FP32。使用原生 inclusive scan 缩短原 score 生命周期，分别记录独立扫描与完整融合 kernel 的寄存器占用；后续再评估 BF16。
- 建立精确或高精度基线后，单独接入 lse.cu 候选近似；不要让近似误差干扰布局调试。

验收：小 tile 与跨 tile oracle 一致，mask 和边界无 NaN 污染，必要的 sanitizer 检查通过；确认生成代码中的布局转换与 spill 情况。

进展：score GEMM 的 B-TMA 列置换及 TK accumulator→GLX 解释已在 sm120
覆盖 warp_k_size=32/64、D=32/64/128，六种组合 bit-exact 且 memcheck 通过。
16x64 的 PV 逆映射、三个 key tiles 的 TMA barrier/parity 循环和融合资源已验证：
九种 D/DV、72 个合成用例，111–166 registers/thread、0 spill、8-byte stack/local。
另有 20 个独立摘要→对角边界合成→重算用例，最多 12 个 checkpoints，
验证 forward HState 的 -1 列偏移、padding identity、hard break 与不同 CTA 分组下的 RNG 重放。
两程序均通过 memcheck/racecheck/synccheck。
未 padding 输入尾加载、生产 RNG 契约、多 warp 同 tile 协作、重叠流水和完整 attention 对照仍未完成；
不能将独立 probes 视为阶段 2 已完成。

## 阶段 2：fixed-length 核心前向

### 主调度与三个 kernel

每 CTA 覆盖128个 query 行，warp tile 为16x64，4个独立32行 checkpoints。compute group A 是 warps 0–3，group B 是 warps 4–7；producer 候选为 warp 8（288线程/CTA，不为凑齐 producer warpgroup 自动增加三个空闲 warp）。

| checkpoint | 前16行 / group A | 后16行 / group B |
|---|---|---|
| 0 | warp 0 | warp 4 |
| 1 | warp 1 | warp 5 |
| 2 | warp 2 | warp 6 |
| 3 | warp 3 | warp 7 |

1. **摘要 pass**：A 的四个 warp 从 top identity 出发，并行计算 score + scalar roll + FP32 pair reduce；B 各 warp 从配对 A 接收 bottom state，再 reduce 并导出32行摘要。各 warp 沿 key tiles 保留自己的 right state；B0 不向 A1 传递，不形成128行的串行链。不加载 V，不物化 score。
2. **边界传递 pass**：普通128/256-thread CTA，不做 warp specialization/TMA。每线程固定 checkpoint 对角坐标 `j-s*32`，沿 checkpoints 顺序合成 `X[s,j]=LSE(X[s-1,j-32]+a[s,j],b[s,j])`，缺失前驱为负无穷。只读摘要，写真实底边 W；不重新计算 score/RNG。相邻线程在同一 checkpoint 访问相邻列。
3. **输出 pass**：每个 A warp 独立加载对应 checkpoint 的真实入边界，B 从配对 A 接收状态；完整 inclusive scan 后发布边界，随后 unroll W、online softmax、PV。暂不拆 upsweep/downsweep，不重复执行 reduce+scan。初始化 max=0、denominator=1、numerator=0；输出 BF16 O，保存 FP32 normalization statistics。A 下一 key tile 与 B 当前 tile 的后半段允许重叠，不强制全 CTA 锁步。

### 数据、接口和缓冲

- 新文件按职责分为 v2 Python 入口/扩展构建、core CUDA、公共 log-affine/布局/RNG helpers、测试；不引入旧接口兼容层，也不把实验中的 `.cu`/main 重命名 include 方式带入正式代码。
- 两个 direction 使用统一 A/B GEMM 接口并分别实例化：`q_from_k` 为 A=q、B=q_from_k、减 row q_lse；`k_from_q` 为 A=k_from_q、B=k、减 column k_lse。rtau/LSE 对外自然对数，score 进入 scan 时统一换成 log2，不能重复乘 sm_scale。
- 输出保存 `L2=max+log2(denominator)`，文档明确为 log2；未来 backward 使用自然对数 L 时乘 ln(2)。正常 API 不返回完整 logM/W/P，不用 torch/Triton fallback 冒充 CUDA 前向。尚未实现 backward 时显式拒绝需要梯度的生产调用，而不是静默断梯度；测试可使用 no_grad。
- 首版用 contiguous BHND/BHNDV 作为 fast path。其他 stride 在入口显式校验或显式 contiguous 化，记录额外拷贝。hard_prob 支持范围及广播行为要显式测试，不能静默缩减 reference 语义。
- 以 K/V 双缓冲起步，保留单缓冲对照；摘要 pass 只有 K。每 slot 有 ready/free 和 phase，回收计数覆盖两个 compute warpgroup。metadata 与对应 tile 的 ready 协议一致。
- A 初始 shared staging 转入寄存器后可与后续 ring storage 复用。D=DV=128 时，不永久保留32 KiB的128行 A 再叠加64 KiB双缓冲；为边界和 metadata 留出预算。检查 A 的长存活是否导致寄存器/spill 问题，按实际资源调整，不预设双缓冲优于单缓冲。
- 配对 bottom mailbox 独立双缓冲，ready/free/phase 与 key tile 身份关联。先完整 scan 后发布；在正式主循环中不使用要求 producer 和 consumers 同步到达的全 CTA barrier。初始 staging 复用和最终退出可使用全 CTA barrier，但必须保证所有线程参加。
- 尾部采用有效坐标检查的安全加载路径，再进入同一 swizzled/permuted shared view；完整 tile 使用 TMA。不得直接用未 padding 输入的现有5D map越界取尾 tile。若后续采用显式 padded workspace，必须计入分配、拷贝和计时。
- 无效 query warp/checkpoint 不执行有效输出写回，但仍履行所有预定的 buffer/mailbox 协议；padding 为 affine identity，合法 hard mismatch 为零映射。明确因果右上方不生产/不读取的摘要区域。
- CTA 内按共同的 key tile 遍历范围驱动 producer 和所有 consumers，各 warp 用逻辑 mask 处理自己不需要的部分，不能因各行 causal 范围不同而漏掉回收确认。具体的跳过策略必须同时调整生产、消费和边界协议后再优化。
- 首版可采用矩形摘要 `(BH,ceil(N/32),padded_N,2)` 和底边 `(BH,ceil(N/32),padded_N)`；FP32总预算为 `12*BH*ceil(N/32)*padded_N` bytes。随后评估因果三角存储；O、L2 和其他 staging 另外统计。
- checkpoint、CTA 行数、compute group 数、通信方式分开配置。仅后续在实测 memory bound 且目标设备验证通过后，评估 CTA cluster/DSM 内四个等价 compute warpgroup、64行摘要。不在当前实现中引入 cluster，也不先拆 GLX scan。

### 实现顺序与检查点

1. **扩展入口与公共组件**：在 blkw 编译/加载最小 CUDA 扩展；整理现有经过验证的 FP32 log-affine、16x64布局、TMA helper；建立两个 direction 的数据参数。固定 RNG seed/offset/全局 direction 契约，首个数值路径用 hard_prob=0、固定 direction。
2. **双 group 摘要及 passing**：先完成32行摘要，检查与 CPU/torch oracle 的两个 affine 分量及已解析边界。验证配对关系、右边界、HState -1列、多个128行 CTA 和尾部无效 warps；memcheck/racecheck/synccheck。
3. **输出 kernel**：接入同一调度的完整 inclusive scan、online softmax 和 PV，与相同 interpolation 对照；先完成 D=DV=64，再覆盖九种组合。V BF16、概率 MMA 前 BF16 转换、FP32 denominator 分别记录误差来源。
4. **随机和场景补齐**：加入 hard=1/混合、两个 direction、可重放随机 direction、不同 B/H 和非对齐 N；测试摘要与重算行决策、generator 状态推进及不同执行配置一致。调试 mask 仅用于 oracle，不成为正式 global mask 输入。
5. **流水验证与初测**：同一主调度下比较单/双缓冲，检查所有9种 D/DV 的寄存器、local、spill、shared以及 kernel时延。包含尾加载、摘要及 passing 开销，分别报告 core 三阶段和含 embedding 端到端；先测瓶颈，再决定是否增加深度/减少摘要。

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
