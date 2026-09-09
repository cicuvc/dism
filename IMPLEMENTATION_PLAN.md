# Dism v2 kernel 执行计划

状态：阶段 1 的独立 GLX、TMA 列置换、融合单 stripe 和三阶段 checkpoint 实验已通过。阶段 2 已实现固定/每次调用全局随机 direction、标量 hard_prob（含混合 RNG）的三阶段 CUDA core 前向；完整接口、反向与性能基准仍未完成。新代码、实测和边界见 `dism_v2/README.md`；不能称为完整 voc_dism 已完成。

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

验收：构建入口可用，接口/状态/RNG 约定明确。当前已实测 RTX 5090 / sm120、CUDA 13.1 的 nvcc 与三类 sanitizer；fixed-direction core 的标量概率与 Philox seed/offset 契约已实现。完整 voc_dism 接口、概率广播及 graph-safe RNG 尚未完成。

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

每 CTA 覆盖128个 query 行，warp tile 为16x64，4个独立32行 checkpoints。compute group A 是 warps 0–3，group B 是 warps 4–7；当前 producer group 是 warps 8–11，总计384线程。早期288线程版本有 spill；按用户建议改为完整 producer group，通过 sm120a setmaxnreg 将其预算降至40、consumer 升至232。只有 warp8 实际加载，其余 producer warps 参与重分配及 CTA 同步。

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
- inc/dec 在 staging 结束后、producer/consumer 长期分支内执行，不能在降额后紧接着汇合并期望编译器仍分别按两套预算优化。预算 `128*40+256*232=64512` registers/CTA；初始 metadata 报168 registers/thread，不代表 consumer 的实际预算只有168。最终 SASS 检查寄存器重分配、原生 TMA、所有 CALL 和 local load/store。
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

前一版实测：固定方向 core 的49个用例通过，覆盖九种 D/DV、两个 direction、soft/hard、B=H=2、N最长513和多个尾部边界。检查 O、L2、局部摘要两个分量及所有 resolved checkpoints。12-warp/40→232版的3个摘要和9个输出实例均0 stack/0 spill；包括 passing 在内的所有 SASS 均无 CALL。输出除法改为每行一次硬件 reciprocal + FP32 Newton 修正，未修改 LSE 算法。尾部 ready barrier 改为每个 producer 线程发布自己的写入，修复单 leader 发布时的 racecheck 报告。49个 core 用例分别通过 memcheck/racecheck/synccheck，0 errors，racecheck 0 hazards/0 warnings。

行 RNG 更新：warp 内 Philox4x32-10、标量混合 hard_prob、PyTorch 默认/显式 CUDA generator 和 RowRNGState 重放已实现，不生成 global mask。固定方向混合调用 offset+4，端点/重放不消费；逻辑行 counter 不依赖物理 warp。新增24个混合数值用例、5个 generator/实际行选择用例和1个已知向量检查；含 codegen、环境/oracle 共161项通过。SASS 无 CALL/LDL/STL，LOCAL=0，TMA 和 setmaxnreg 断言仍通过。不同 CTA 配置、概率广播和 graph-safe RNG 尚未验收。尚未做吞吐测量，原始 reference/embedding 文件未修改。复现脚本 `scripts/check_dism_v2.sh`。

全局 direction 更新：`direction=random` 每次调用在 host 上计算预留 Philox block 的一个方向位，统一选择 A/B/LSE，无 GPU→CPU 同步。前置方向 block 与后续行 RNG block 隔离；随机方向端点/混合分别消费4/8 words，保存已选方向与行 offset，重放消费0。新增 `forward_interpolated` 命名入口（要求预先准备 BF16 contiguous interpolation，不运行 embedding）。增加27项随机方向测试，覆盖两个随机结果、九种 D/DV、默认/显式 generator、重设 seed、重放及端点；总计188项通过，106项 core 测试分别通过 memcheck/racecheck/synccheck，0 errors、0 hazards/0 warnings。SASS codegen 断言仍通过，日志 `/tmp/dism-v2-check.x6Ly3n`。chunk passing 后续保留完整 log1p/exp2 LSE，避免 tile 内近似扩展到长程传递。

## 阶段 3：核心反向

实际 embedding 前向诊断：新增64项 `test_dism_v2_embedding_precision.py`，48通过、16个相对未量化FP32插值的输出阈值失败；同emb输入core输出/L2全部通过，所有随机样本标签一致，LSE最大误差9.54e-7。包含九种D/DV、N至8193、V=31/64/65/129/257及多BH。只改输出缓冲dtype的诊断调用显示内部BF16 softmax权重和最终BF16写回都贡献误差；尚未修改生产embedding或CUDA core精度。详细对照见PRECISION.md。V65/N257尾部emb_fwd memcheck为0 errors；embedding backward仍未验证。

前向精度压力回归（新增141项）：N最长8193，rtau自然对数上限ln(D)，九种D/DV、两方向、长匹配链与FP32插值对照，详见 `dism_v2/PRECISION.md`。完整329项中301通过、28失败；14个失败位于rtau上限内，全部为纯soft输出对未量化FP32插值的阈值失败。同BF16插值的core检查通过，最差范围内长序列用例的reference间插值量化误差已接近端到端偏差。输出余弦也已固化到JUnit：范围内同BF16插值的整体余弦最低0.999996730，FP32插值对照最低0.999926263、最差行0.998288881；不以高余弦替代逐元素误差验收。保留普通失败、不放宽阈值；后续需与用户明确是否接受此量化误差或探索更高精度插值路径。此次未修改kernel数学或数值实现。

进入后续工作前的 RNG 回归：79项 core 测试在 memcheck、racecheck、synccheck 下分别通过；0 errors、0 hazards/0 warnings。记录 `/tmp/dism-v2-check.sPZ4nb`，12个 core 实例 STACK=0、LOCAL=0，初始 REG=168，producer/consumer 预算仍为40/232。

数学基线（自然对数语义）：

```
P[i,j] = exp(W[i,j] - L[i])
delta[i] = dot(dO[i], O[i])
E[i,j] = P[i,j] * (dot(dO[i], v[j]) - delta[i])
G[i,j] = E[i,j] + sigmoid(W[i,j]) * G[i+1,j+1]
```

### 当前主方案：按 key 分块的转置反向

用户接受当前前向精度作为初版基线，训练稳定性出现问题后再处理量化误差；测试中的失败与数值报告继续保留，
不作为当前反向布局验证的阻塞条件，也不通过放宽阈值伪称精度已全部验收。

- 前向训练保存 **每16列 key 的竖边 W₂** 和 **每64行 query 的横边 W₂**，仅 FP32 second 分量。
  替代此前讨论的横16/竖64方案；前向 score/scan/online softmax 的16×64计算 tile不变。
  32行 affine 摘要仅用于前向 passing，生命周期结束后可复用其空间，但需保证异步 stream/调试返回值语义。
- 反向每个 warp 持有16个 key 的 B 操作数，流式加载64个 query 的 A/dO，使用转置视图 `[16 key,64 query]`。
  原坐标的竖16边界对应转置视图的 top，横64对应 left，允许按 query tile 逆序独立重算 W。
  不沿用前向 warpgroup0→1 的重算边界传递；仅在反向 add-mul scan 中配对 `4→0,5→1,6→2,7→3`。
- 128个 key/CTA，逻辑16-key块仍交错分给0,4,1,5,2,6,3,7；反向摘要粒度32个 key。
  producer沿query逆序提供双缓冲，12-warps、40/232为初始候选，完整反向寄存器和流水须重新编译验证。
- B1：独立重算W/P/E/alpha，warp内累积dV并直接写回，生成32-key反向affine摘要。
  B2：沿对角线反向passing得到key chunks的真实梯度入边界。
  B3：重算并reverse scan得到G，warp内累积dB并直接写回，对dA做FP32 atomic，归约dLSE/drtau。
  delta可先由独立小kernel生成；不保留完整G/P/W矩阵。
- 所有query tiles由同一key warp遍历，因此dV与dB无跨CTA写竞争，不做split-query；dA存在跨key的累加。
  q_from_k方向B=q_from_k，k_from_q方向B=K；无atomic的是core的dB，不应无条件称为原始dK。
- 梯度定义在自然对数logits域，P=`exp2(W₂-L₂)`、alpha=`1/(1+exp2(-W₂))`（用稳定形式计算）。
  affine `(a,b)*(c,d)=(a*c,b*c+d)`，初始`(alpha,E)`；真实hard不匹配`(0,0)`，padding identity`(1,0)`。
  drtau包括soft与hard匹配项，不跳过hard行递推；后续dA/dB不重复乘LOG2E。
- RNG逻辑身份不变；转置tile中64个query对应列而不是行，每个query的hard/soft选择沿16个key广播。
- 忽略尾部取整，边界矩形存储约`4*BH*N²*(1/16+1/64)` bytes；N8192/BH1约20MiB。
  这是保存边界的空间，不是整个forward/backward的峰值临时内存；后续仍需生命周期/因果裁剪预算。

首个验证：从前向16×64 inclusive scan导出逻辑列15/31/47/63的second分量；
检查GLX roll前后映射、tail/identity、负无穷与跨tile状态，再用竖16/横64边界独立恢复转置tile。
实验目录 `experiments/glx_boundaries`。下一阶段才实现真实前向checkpoint写回和reverse add-mul scan，
随后融合dV/dB/dA GEMM；本布局probe不代表反向已实现。

2026-09-07验证结果：16×64的`data[r][7].second`在roll后仍对应原行，g=1/3直接输出15/31/47/63列，
无新增shuffle。40个跨tile/尾部case中，竖16/横64独立恢复转置tile成功，double最大误差6.26467e-6，
导出边界与完整W bit-exact。另验证16×16、16×32、32×16、32×32、16×64共25个单tile case；
全部默认实例无CALL/stack/local/spill，导出前后SHFL相同，三类sanitizer通过。32×64导出映射仍正确，
但dense scan对oracle有未定位偏差；当时16×128受GLX静态断言限制（更新后结果见下）。详情见实验README，
回归入口`tests/test_dism_v2_boundaries.py`。生产kernel尚未改动。

GLX更新复验：旧前向/布局189项通过；完整前向350通过、44个既有精度失败，失败集合和203条精度报告与
更新前完全一致。106项core用例三类sanitizer均通过。旧scan/reduce/log-affine探针也全部通过。
16×128现已支持并加入默认边界回归：5例log-affine数值通过、边界bit-exact，baseline/导出均128寄存器、
124条SHFL、零spill；上游14项正/反scan和12项reduce等价检查通过，memcheck零错误。
32×128的5例数值/导出也通过，但probe占255寄存器并有152/144 bytes stack及LDL/STL，保留为探索项。
32×64的dense数值偏差仍可复现（max_abs=0.1004318），不作为已验收shape。尚未扩大生产kernel的tile。

- 预处理 delta，重算前向 tile 并计算 dV 和反向 affine 摘要。
- 反向传递 checkpoint 边界，再重算并 reverse scan，得到 score 梯度。
- 计算左右 GEMM 输入梯度、对应 LSE 的 row/column reduction 和 rtau 梯度；soft mask 只作用在 soft score 的局部导数，不屏蔽递推链本身。
- 初始跨 CTA 累加可使用 FP32 atomics 或显式 partial buffers，按实测选择并说明确定性；不使用 BF16 原子累加作为梯度精度基线。

验收：core 全部可微输入对照 autograd；覆盖混合 hard/soft 跨 checkpoint 梯度传播、所有维度组合、tail 和两种方向。区分近似算子反向策略与精确数学梯度。

### 反向实现前的准备项（当前执行顺序）

1. 实际12-warp前向导出竖16/横64：已接入可选`save_boundaries`，返回FP32 W₂边界，
   不物化全矩阵。106项core测试覆盖全部边界与FP64递推、保存开关逐位输出一致性，
   加codegen共107项通过。尾部额外padding checkpoint承接最后一个已有32行边界。
   初次集成在D64/DV64出现8-byte stack；移除跨角色分支存活的tiles循环上界后恢复
   12实例STACK/LOCAL=0，无CALL/LDL/STL，原生TMA与40/232重分配保留。
   当前保存状态单独分配，尚未做临时空间复用或性能计时。
   106项core回归分别通过memcheck/racecheck/synccheck，零错误/零hazards；
   日志`/tmp/dism-saved-edges.U5afDy`。
2. 用真实BF16 MMA score、转置query列RNG、方向LSE和mask，验证按query逆序的独立重算。
   仅允许读上述边界，不能读取完整W。真实MMA独立probe已实现于`experiments/glx_recompute`：
   96数值用例+1codegen通过，FP64 GEMM/递推对照所有padded元素；D32/64/128资源125/124/193
   registers、STACK/LOCAL=0，无CALL/LDL/STL。生产反向尚未接入，未外推为融合反向资源。
   合并reference/build/core/codegen/稀疏边界/真实重算回归共286项通过；本轮未重跑长序列
   precision与embedding_precision，先前44个量化相关失败仍保留，不声称完整精度套件全通过。
   真实重算97项分别通过三类sanitizer，零错误/零hazards；日志`/tmp/dism-recompute-check.3YaK76`。
3. 构建Dism专用reverse add-mul验证：signed E、稳定alpha、hard break/identity、
   4→0配对mailbox、32-key摘要与reverse passing，对照独立FP64递推。
   已实现`experiments/glx_reverse`三步独立probe，45例全部通过，摘要first/second、passing边界及G
   最大绝对误差8.11646005e-7。N最长513，含非对齐尾部、混合break、全不匹配和ln64长链。
   reverse HState逻辑列1…64，列0由VState补齐；不能沿用前向-1…62编码。
   摘要/scan/passing分别116/138/36寄存器，STACK/LOCAL=0，无CALL/LDL/STL。
   三类sanitizer零错误/零hazards，日志`/tmp/dism-reverse-check.ZXq7hw`。
   E中的signed_dP为测试信号，尚未融合真实dO/V/delta、MMA或12-warp producer流水。
4. 固定core梯度接口与归约归属，再实现B1/B2/B3；最后处理embedding backward写竞争及全链路。

### Core反向接口与初版精度约定

- 反向消费已选方向的A/B/V/LSE/rtau/labels、保存的BF16 O、FP32 L₂和ScanBoundaries，
  以及RowRNGState；dO首版为contiguous BF16，与O同形状。不得重选direction或消费新offset。
- 返回core语义的FP32 `(dA,dB,dV,dLSE,drtau)`；完整autograd包装接入时再按输入dtype回传，
  不在跨CTA累计时降为BF16。不提供尚未完成的伪backward入口。
- dA为FP32 zero-init缓冲，B3按key tiles atomic累加；dB/dV每key warp遍历全部query，独占写回。
  q_from_k的dLSE按query归约，使用FP32跨key累加；k_from_q的dLSE按key归约，key warp独占写回。
  drtau包含soft及hard匹配的G，以每CTA/head partial再按head归约，避免BF16 atomics。
- `G=d loss/d natural_logM`；dA/dB仅乘sm_scale，不再乘LOG2E。
  soft局部导数使用`G_soft=G*(~hard_row)`，dLSE为负的row/column sum；drtau为所有G之和。
- delta使用保存的BF16 O：`sum(float(dO)*float(O))`，FP32累加输出；不是对BF16舍入严格求导。
  后续梯度误差测试需要区分理想FP32 core oracle和保存O量化引入的误差。
- 新增54项CPU FP64 autograd公式检查：全部九种D/DV、两方向、soft/mixed/hard，覆盖五项core梯度，
  atol/rtol=2e-12全部通过。该测试验证数学与归约契约，不等于CUDA B1/B2/B3验收。
- 已实现独立CUDA delta预处理（`dism_v2/backward.py::delta`），8行/CTA、一warp一行，支持DV32/64/128、
  非对齐行数、非默认stream；21数值/接口用例+1codegen通过。三实例14/16/19寄存器，
  STACK/LOCAL/SHARED均0，无CALL/LDL/STL。不与前向扩展混编，避免其codegen回归计数混淆。
  22项GPU测试分别通过memcheck/racecheck/synccheck，零错误/零hazards；日志`/tmp/dism-delta-check.7dxkGq`。
- 下一步接入B1：复用已验证的独立转置W重算，计算真实dO/V的dP、E与alpha，并融合dV和32-key摘要；
  随后接B2和B3。完整反向12-warp寄存器/双缓冲流水仍需在融合后实测。

### dV正确性里程碑

已实现独立`backward.value_gradient`：单warp CTA持有16key，64query逆序流式重算W/P，
TMA加载A/dO并在warp内累积FP32 dV，无atomic/global W/P；暂未融合E和32-key reverse summary，
不是完整B1。当前key先保存在shared、MMA前重新加载以缩短寄存器生命周期，尚无producer流水。
输入、RNG、精度修正与资源详情见`dism_v2/DV.md`。

初版P单次BF16转换在8193行两方向pure/mixed共4例超阈值；未放宽测试，改用P的BF16高位+
BF16残差两次MMA。79项dV与76项已有backward测试合跑155项通过。
随后将全匹配链/全不匹配等也扩至8193行，最终85项dV测试通过，最差相对L2=4.09694e-4，
最低cosine=0.999999928；覆盖九种D/DV、N至8193、rtau≤lnD。扩充前合并非precision回归442项通过；
本轮未重跑已有长序列forward precision/embedding precision套件，保留既有44个量化失败。
所有dV实例无CALL/stack/local/spill/atomic，寄存器141–254；尚未测性能，不能外推12-warp资源预算。
85项均已覆盖三类sanitizer，零错误/零hazards；日志`/tmp/dism-dv-corrected-check.9ZhQaG`
及`/tmp/dism-dv-extra-check.afHEMu`。dV正确性里程碑完成，详细边界见DV.md。
下一阶段才继续融合dP/E与反向摘要，然后B2/B3和完整core autograd。

### dV融合摘要与B2 passing

已在dV kernel内接入真实dP/E、稳定alpha和16-key add-mul reduce；后续一个CUDA passing kernel
组合16→32-key摘要并逆序传递G边界。接口`value_gradient(...,v=v,delta=delta)`返回dV、summary32、boundary32，
不提供可选参数时仍走独立dV。详见`dism_v2/BACKWARD_SUMMARY.md`。
这是保持单warp CTA的功能实现：没有启用12-warp producer流水/配对mailbox，暂时多保留16-key local。
该临时buffer和32-key输出共0.875*BH*np² bytes；np8192/BH1为56MiB，不能称为全路径峰值。

为消除融合spill，dV累加值在阶段间暂存到本warp的shared数组，dP按32维分段加载。
最终所有dV实例及passing无CALL/stack/local/spill/atomic，融合版170–255寄存器，passing36。
新增78项summary/边界检查与85项dV和76项backward合跑239项通过。
同重算W/L₂下FP64 oracle的摘要second/边界最大误差2.223e-5/6.315e-5；
reference W的长程舍入差异单独记录，不把这项测试当成完整G对autograd验收。
78项摘要用例三类sanitizer均通过，零错误/零hazards；日志`/tmp/dism-summary-final-check.3J8bU2`。
合并reference/build/core/codegen/布局/反向原语/dV/摘要回归526项通过，日志`/tmp/dism-summary-all.xml`。
本轮没有重跑已有forward precision/embedding precision，既有量化失败继续保留。
下一项为B3 reverse scan和dA/dB/dLSE/drtau，性能流水及状态buffer复用待后续完成。

### WS双缓冲接入：编译后因spill暂停

上一版已提交`1398426`。新增可选`warp_specialized=True`（需要v/delta），源文件`core_dv_ws.cu`；
默认仍为已验证的单warp路径。12warps/CTA、producer group40、consumer groups232，
只有warp8加载A/dO双槽；key块按0,4,1,5,2,6,3,7交错，score/W/dV/E独立重算，
只在reverse reduce前由0–3等待4–7。双槽mailbox直接输出32-key摘要，passing不再组合16-key local。
初始B/V shared staging与A/dO ring复用空间；B/V和dV跨tile保留寄存器，W跨PV/dP保留寄存器，
没有将单warp版本的per-warp shared暂存直接扩充到8份。

九种D/DV均编译成功，但全部出现spill。按用户最新指令已停止实现/调参，未执行WS数值测试或sanitizer。
SASS无CALL，9实例各自保留原生UTMALDG.5D和USETMAXREG inc/dec；初始REG metadata=168，
实际角色预算为40/232。32-key passing REG34、无spill。下表为ptxas编译报告，不是运行期累计访存量：

| D | DV | stack bytes | spill stores bytes | spill loads bytes |
|---|---|---:|---:|---:|
|32|32|40|40|48|
|32|64|16|16|16|
|32|128|208|220|224|
|64|32|32|32|32|
|64|64|8|4|4|
|64|128|200|236|228|
|128|32|56|48|56|
|128|64|32|32|40|
|128|128|264|404|344|

构建日志`/tmp/dism-ws-first-build.log`。新代码未提交；未修改spill策略、未放宽codegen断言。
由于WS实例已编入backward扩展，原有全扩展零LDL/STL检查目前会因这些实验实例失败；
不能把此前526项通过描述为新版WS已通过。以上为双MMA版本的历史结果。

### WS单次BF16 P MMA实验（用户授权）

仅WS路径取消P残差及第二次dV MMA；单warp基线保留高位+残差。
不改score/W/E/反向affine、40/232预算或mailbox流水。新的ptxas结果：

| D | DV | stack bytes | spill stores bytes | spill loads bytes |
|---|---|---:|---:|---:|
|32|32|8|8|16|
|32|64|0|0|0|
|32|128|48|68|72|
|64|32|8|8|16|
|64|64|0|0|0|
|64|128|64|72|68|
|128|32|24|20|28|
|128|64|0|0|0|
|128|128|144|224|180|

三个DV64实例已零spill，其余不继续自行优化。32/32原先P构造/PV/dP交叠区域的
8个寄存器spill已消除，只剩chunk索引及符号扩展的8-byte stack：入口STL、
摘要输出分支两条LDL.64及一条STL。全部WS仍无CALL，原生TMA和40/232重分配保留。
日志`/tmp/dism-ws-single-p-build.log`，SASS `/tmp/dism-ws-single-p.sass`。

新增`tests/test_dism_v2_dv_ws.py`复用既有oracle及阈值：85项79通过、6项精度失败
（4种不同长序列配置，纯soft在两个测试入口重复）；九维度和tails的65项还验证真实
reverse摘要/边界，均通过。原单warp163项全部通过，合计242通过/6失败，
XML `/tmp/dism-ws-single-p-tests.xml`。精度失败保留为普通失败，不改阈值或xfail；
具体指标见`dism_v2/DV.md`。全扩展零spill codegen验收仍未通过。
新增WS/passing的定向memcheck、racecheck、synccheck各65项通过，0 errors/hazards；
过滤条件`--kernel-name kns=_ZN7dism_v22ws`，日志
`/tmp/dism-ws-single-p-filtered-{memcheck,racecheck,synccheck}.log`。
此前全进程插桩因过慢主动停止，未计入通过；本轮未做性能测量或完整前向精度回归。

### WS D→C顺序与tanh sigmoid实验（历史对照）

按用户要求先构造dP/E及reverse二元组，再做单次BF16 P的dV MMA；
接着把sigmoid改为`fma(0.5,tanh.approx(W2*ln(2)/2),0.5)`。
保持-inf返回(0,0)、padding返回(1,0)，默认单warp基线不改。
仅调换顺序使32/32零spill，但大尺寸恶化；reverse二元组现需跨PV存活。
下表前两列为stack bytes比较，后三列为当前tanh版ptxas报告：

| D | DV | C→D stack | D→C旧sigmoid stack | 当前stack | 当前stores | 当前loads |
|---|---|---:|---:|---:|---:|---:|
|32|32|8|0|0|0|0|
|32|64|0|8|8|8|8|
|32|128|48|360|376|616|428|
|64|32|8|8|8|8|16|
|64|64|0|8|8|4|4|
|64|128|64|448|408|704|496|
|128|32|24|8|8|8|16|
|128|64|0|128|104|112|112|
|128|128|144|536|512|928|640|

当前9实例均无CALL，各32条MUFU.TANH，sigmoid算术主链为FMUL/TANH/FFMA
（常量MOV、分支和E运算另计）；TMA与40/232寄存器重分配保留。未继续调spill。
构建/SASS：`/tmp/dism-ws-{d-before-c,tanh}-build.log`及同前缀`.sass`。
两个版本原85项均79通过/6已知dV精度失败。
65项同W摘要oracle最大(first,second,boundary)误差：旧sigmoid
(5.07545e-8,2.49989e-6,2.49989e-6)，tanh
(3.65359e-6,1.34495e-5,1.34495e-5)，均通过原阈值。
D→C旧sigmoid版定向三类sanitizer各65项通过、零errors/hazards，
日志`/tmp/dism-ws-d-before-c-{memcheck,racecheck,synccheck}.log`；
tanh纯数学替换后尚未重跑sanitizer，不把上述结果冒充最新二进制验收。
新增12项N1025/2049、双方向、chain/break/bounded_soft长程摘要回归全部通过，
最大(first,second,boundary)误差为(6.50232e-6,7.82838e-5,2.90753e-4)。
XML `/tmp/dism-ws-tanh-long-summary.xml`。当前97项合计91通过/6已知失败；
尚未验证更长序列的tanh梯度passing误差或完整backward。

### WS C→D + tanh（当前源码）

按用户要求切回先dV后dP/E，保留单次BF16 P和tanh sigmoid，其他调度/预算不变。
当前ptxas资源（bytes）：

| D | DV | stack | spill stores | spill loads |
|---|---|---:|---:|---:|
|32|32|0|0|0|
|32|64|0|0|0|
|32|128|64|72|68|
|64|32|8|8|16|
|64|64|0|0|0|
|64|128|104|112|108|
|128|32|16|16|24|
|128|64|0|0|0|
|128|128|152|160|156|

32/32和全部DV64实例零spill，其余五种仍有spill，未继续优化。
相比D→C+tanh大尺寸明显改善；相比C→D旧sigmoid，DV128的stack增加，不能称全面胜出。
九实例全部无CALL、各32条MUFU.TANH，原生TMA及40/232重分配保留。
当前97项91通过/6原有P量化精度失败，包含12项长程摘要检查，未放宽阈值。
日志`/tmp/dism-ws-c-before-d-tanh-build.log`、同前缀`.sass`、
`/tmp/dism-ws-c-before-d-tanh-tests.{log,xml}`。
本轮未重跑sanitizer或性能测试；源码保留此版本，未提交。

## 阶段 4：voc_dism 全链路

前置B3布局probe：已按用户要求验证Gsoft经shared的转置读取、dA16x32分块MMA、
FP32 shared双槽和原生TMA异步reduce-add。见`experiments/glx_da_tma/README.md`。
63数值+1codegen通过，三类sanitizer通过；六个D/warp实例72/80寄存器，零CALL/spill。
输出不padding，TMA边界丢弃和多warp/CTA累加已验证。此probe无DV维度、无真实G重建，
不代表完整dA/dB或12-warp融合资源验收。当前没有改动生产kernel。

进一步的B3真实输入单warp验证见`experiments/glx_g_recompute/README.md`：
每warp顺序处理32key的高/低16半块，用真实score/dP/E、前向稀疏W边界和生产G32
边界恢复完整G（仅诊断输出）；随后测试Gsoft接入dA TMA及新dB独占累加probe。
新增G/梯度GEMM85项与原dA64项共149通过；合并dV/WS回归403通过/6旧P量化失败。
G九实例168–250registers、dB40/64/154，均零spill。尚未融合成不物化G的B3生产kernel，
不能把诊断kernel的资源或量化对照当作完整反向验收；未实现dLSE/drtau归约。
三类sanitizer各149项通过，零errors/hazards。G最大绝对误差2.47260e-4，
同BF16 Gsoft下dA/dB误差2.84749e-6/1.89835e-6；FP32 Gsoft量化对照最大relative L2
约0.00293/0.00295。具体口径及日志见实验README。下一步为不物化G的B3融合及资源验证。

### B3单warp融合：D128 spill，按约定暂停

验证阶段已提交`450168b`。新增`dism_v2/csrc/core_ab.cu`和`ab_rescan.cuh`，
编入backward扩展，但尚未提供Python/C++ binding调用入口或运行测试。
每CTA一个warp负责32key，query64逆序；真实W/dP/E→reverse scan→shared BF16 Gsoft，
不在global物化G/W/P/E。两个16key dB FP32 accumulator跨循环保留；
dB按feature32读取逻辑顺序query shared临时块计算，原permuted query staging保持不变。
dA按16x32分块，经双槽FP32 shared向3D[N,D,batch-head] tensor map异步TMA reduce-add；
最终dB独占写回。这里3D描述符按CUDA坐标实际为[D,N,batch-head]，
与已验证的2D probe不同，尚未验证实际运行的多batch/head或tail行为。
当前不含dLSE/drtau归约，不是完整autograd；不含12-warp producer流水。

编译成功，资源如下（spill统计单位bytes，非动态访存量）：

| D | DV | registers | stack | spill stores | spill loads |
|---|---|---:|---:|---:|---:|
|32|32|235|0|0|0|
|32|64|224|0|0|0|
|32|128|218|0|0|0|
|64|32|242|0|0|0|
|64|64|238|0|0|0|
|64|128|242|0|0|0|
|128|32|255|24|32|28|
|128|64|255|24|24|24|
|128|128|255|32|40|32|

全部实例无CALL，原生UTMALDG.5D/UTMAREDG.3D.ADD/MUFU.TANH保留。
按用户要求出现spill后停止，没有继续调整accumulator存储、寄存器预算或调度。
尚未数值/sanitizer/性能测试，不能用诊断probe的通过替代融合版验收。
日志`/tmp/dism-ab-first-build.log`，SASS `/tmp/dism-ab-first.sass`。本轮融合改动未提交。

### B3继续验证（D128 spill已获授权保留）

用户允许暂不处理D128 spill。已接入`backward.operand_gradient`及严格输入校验，
返回FP32 dA/dB，dA zero-init后使用3D TMA reduce-add，dB独占写回，无global G。
新增B3专属codegen允许D128 spill，仍要求其他六种零spill及所有实例无CALL/原生TMA。
87个同独立G/GEMM数值用例加codegen通过，dA/dB误差约5.22e-6/1.61e-5；
另有API参数、零dO、禁止higher-order/deterministic契约检查。
直接reference autograd的54常规维度项通过，6个D64/DV128、N17/65/139、两方向
pure soft、rtau=ln64逐元素精度失败，保留普通失败。reference FP32 O的delta诊断
仅部分降低误差，没有改变生产delta语义。详见dism_v2/AB.md。
当前未做性能优化/计时，未融合12-warp流水，未实现dLSE/drtau归约或embedding autograd。
最终AB149项143通过/6 reference精度失败；与dV/WS/G/梯度probe合并558项，
546通过/12失败（另6项是旧WS P量化）。三类sanitizer各88项通过、零errors/hazards，
仅插桩B3 namespace；API契约检查在普通回归中通过。日志见dism_v2/AB.md。
融合实现及上述验证已提交：17f5674。用户接受当前精度，普通失败保持可见。

### B3 warp specialization（当前执行）

最新进度：dLSE/drtau已接入WS与单warp，两路径接口改为四输出。
用户已授权保留WS D64/DV128新增56B stack及D128 spill，继续正确性验证，不优化。
同G检查通过，直接reference暴露新增drtau精度失败；诊断指向saved BF16 O的delta误差。
资源、数值与sanitizer范围见dism_v2/AB.md末尾。
本轮四输出299项257通过/42普通reference失败；非reference的176数值/契约与
3项codegen全部通过，三类sanitizer各179通过、零errors/hazards。
8193闭式对照8项通过，并完成独立记录的sanitizer范围；
旧路径/probe409项403通过/6原失败，delta/FP64公式75项通过。
不将同G通过视为原始reference精度达标；生产delta保持不变。
下方WS通过记录均为新增标量梯度之前的版本。

- 独立 core_ab_ws.cu，按用户后续要求operand_gradient默认启用WS；
  warp_specialized=False显式选择单warp诊断基线。dV入口默认策略未改。
- 12 warps/CTA，8 compute各持16key，4 producer-group，40/232寄存器重分配。
  128key交错分配0,4,1,5,2,6,3,7；A/dO双缓冲流式倒序加载64query。
- W/dP/E独立重算；warp4–7载入下一32key的G32边界，inclusive reverse scan后
  双槽mailbox传给0–3。边界在梯度GEMM前发布；不增加组内串行链。
- dB每warp一个FP32 accumulator独占写回，G寄存器保留query物理permutation，
  直接与TMA加载的A相乘。必须完成dB读取后才arrive input-free。
- Gsoft同时以逻辑query列写shared，再col-layout转置读入寄存器供dA。
  每warp2KiB union复用为Gsoft或两个16x16 FP32 dA输出槽；全CTA16KiB。
  采用16x16而非单warp版16x32 TMA输出块，以限制shared占用。
  每次union写G前wait_group.read0，输出槽复用wait_group.read1，
  所有writer发布fence/syncwarp，退出wait_group0。
- 首轮编译九实例通过，D32/64六实例零spill；D128对应DV32/64/128：
  stack32/144/216 B，spill stores32/192/228 B，loads32/144/212 B。
  ptxas CTA metadata168寄存器，不能误报为consumer角色寄存器数。
  D128沿用用户授权，暂不优化。初版日志 /tmp/dism-ab-ws-build.log。
- WS完整149项143通过/6原bounded-soft reference失败，同G/GEMM最大绝对误差
  dA=5.22108e-6、dB=2.40658e-5。三类sanitizer各89项通过，零errors/hazards。
  codegen强化检查40/232立即数后单独复跑通过。旧路径/probes558项546通过/12原失败，
  无新增失败；未运行全套前向/embedding测试，未测性能。日志见dism_v2/AB.md。

- 接入现有 EmbInterpFunction，先复现并解决 embedding backward 的重复 dq/dk 写入问题。
  最新接入已完成：dism_v2.autograd.voc_dism直接调用现有embedding forward/backward wrappers，
  避免core FP32梯度在中间autograd BF16节点提前舍入，最后合并直接梯度后统一cast。
  Phase B加pid_v==0归属保护；D128的num_stages=1避免前后向shared超限。
  89项端到端接线/契约及三类sanitizer通过；另三步loss.backward/update smoke通过。
  全部新增测试232项209通过/23普通精度失败（20个rtau幅值+3个独立V1 embedding）。
  两oracle的62场景中rtau未观察反号，不能外推长期训练稳定性。
  实测、资源、原始日志与限制见dism_v2/AUTOGRAD.md；性能/varlen/sm90仍未完成。
- 正确传递插值 embedding 与 LSE 梯度，合并 score 的直接 q/k 梯度和 embedding 阶段梯度，完成六个输入的 autograd 接口。
- 检查包导入、stride、dtype、sm_scale、词表 tail、top-1 tie 和随机调用语义。
- 分别对照纯 torch interpolation 和 kernel interpolation，报告前级量化与 core 误差。

验收：voc_dism 前向和 q/k/v/rtau/q_voc/k_voc 梯度全链路通过；随机前向可重放，backward 不额外消费随机数。

## 阶段 5：sm120 性能迭代

### 下一主项：CUDA embedding插值前向

用户确认端到端主线完成，提交8a90cba；随后授权实现CUDA embedding前向，
暂不实现其backward。详细执行与验收见[dism_v2/EMBEDDING_CUDA_PLAN.md](dism_v2/EMBEDDING_CUDA_PLAN.md)。
12 warps/CTA：WG0四warp各16行acc_qo、WG1四warp各16行acc_ko；
producer group8–11参与setmaxnreg，只有warp8实际加载。两个计算组处理相同64个token位置，
共享E_q/E_k双缓冲，分别交换词表的score-key/PV-value角色。初始B_V64、dec40/inc232，
以编译实测为准；只有最后一次PV输入读取完成后才释放词表slot。
E1单warp基线、E2融合12-warp/TMA双缓冲、E3显式端到端backend已实现。
D32/64/128均零spill，无SASS CALL，原生TMA和40/232寄存器重分配已验证。
124项embedding前向/布局/tie/codegen/Triton对照通过；三类sanitizer各124项零错误。
CUDA端到端81项接线通过；108项reference中100通过、8项既有类型rtau幅值失败，
符号无反转。与原套件一起514通过/31普通失败，其中旧套件23项失败保留。
E4已提供热cache前向API计时脚本；尚未完成纯kernel、冷cache/L2/HBM采样、
相同CTA划分的独立FA流量消融及单缓冲对照。D32/64部分形状仍慢于融合Triton，
因此默认Triton不变。详见[dism_v2/EMBEDDING_CUDA.md](dism_v2/EMBEDDING_CUDA.md)。
追加B_V128：D32/64两个WS实例均零spill，48项新增数值/tail测试通过。
已补CUPTI实际kernel计时：N4096/V1024、B1/H4时D32为55.94→54.21µs，
D64为78.03→71.02µs，融合Triton为29.63/56.35µs；V129尾部变慢。
暂保留block_v=128为低层显式实验选项，不自动改变高层64步长。
因此上述E4剩余项中纯kernel计时已有首批数据，冷cache/L2/HBM/消融仍待做。

- 建立按 B/H/N/V/D/DV 分组的基准，序列长度覆盖短序列到显存预算内的长序列；具体训练代表形状由用户场景补充。
- 分别测量 embedding、摘要、传递、输出、各反向阶段以及端到端时延，包含临时缓冲和 RNG 的实际路径。
- 调整 GLX shape、checkpoint 长度、warps/CTA、TMA stages、producer/consumer 分工和寄存器存活区间。
- 评估 LSE 候选近似的输出/梯度误差、长链稳定性及端到端收益。SASS patch 如有必要，作为独立、可验证的实验路径。
- 根据瓶颈再决定并行化边界传递、减少 GEMM 重算、消除原子竞争或采用 CUTLASS/CuTe；不预设必须重写框架。

验收：发布可复现的 correctness/performance 表与显存开销，不设未经测量的性能承诺；优化后的必要回归检查通过。

### CUDA embedding反向新增里程碑（用户已批准）

单warp稀疏反向基线已按用户授权保留spill并验证：48项同状态检查及三类sanitizer通过。
另18项与Triton反向一致，其中6项V1对FP32 oracle保留量化失败。
WS/P mailbox已写入，编译新增D64 vocab 8B stack、D128 token/vocab 16/64B stack；
全部无CALL、原生TMA/setmaxnreg生效，按新增spill先报约定暂停，未调参、未测试WS。
未接入autograd；详细资源及待确认项见dism_v2/EMBEDDING_BACKWARD.md。
随后按用户建议比较词表WG0/WG1=248/216与232/232：拆成两个长期consumer分支后，
两种预算都使D128 vocab降为8B stack（store8/load32B）；D64仍8B，D32零spill。
因此目前保留长期分支+232/232，不能把收益归因于非对称预算；未继续压低WG1预算。
此次只做编译/SASS对照，无CALL；WS运行验证仍待进行。
另按用户建议预乘scale/LSE的LOG2E，P重算仿射部分使用单FFMA。
LSE2临时行缓冲额外8B×BHN，梯度scale保持自然域；WS FMUL静态数量减少，spill不变。
单warp数值回归仍60通过/6项既有V1失败，详见embedding反向文档。
按用户后续要求新增对称词表WS：每组独占64词表项、每warp同时累积两张词表梯度，
取消P mailbox及配对通信，仅保留共享输入ring同步。D32/64/128编译新增stack32/152/248B，
无CALL、TMA/setmaxnreg生效；按约定暂停，未调参、未测试或计时。旧配对版保留作对照。
后续用户授权缩小token步长：对称版D32=32、D64/128=16，stack分别0/8/208B。
用户明确暂不处理D128；仅D64将两张词表常驻shared、按32-feature读取，
最大live GPR实测198→188，动态shared32896→45184B，8B标量spill仍在，未计时。
D32/64共36项同状态检查通过；D64三类sanitizer各18项通过。详见embedding反向文档。

最新：用户授权放弃spill修复，已完成20个可运行WS配置的数值/性能搜索。
embedding反向套件394通过、6个既有V1 FP32 oracle普通失败；全配置三类sanitizer各20项通过。
七组B/H/N/V、D32/64/128、每配置三轮20次CUPTI计时已固化CSV及可复现脚本。
小CTA网格配对更优；大CTA网格D32/64对称可胜出；D128本批仍配对最优。
D64 reg/T32与shared/T16对称候选随形状互有胜负，spill不作为淘汰依据。
完整方法、资源、误差与结果见dism_v2/EMBEDDING_BACKWARD.md及benchmarks/embedding_backward_sm120a.csv。
暂保留显式配置，不更改低层默认或把CUDA embedding backward接入autograd。
随后按用户最终全链路验证要求，已增加embedding_backward_backend="cuda"/"cuda_symmetric"
显式接入配对/对称WS反向；默认Triton及低层配置不变。最终验证结果见dism_v2/AUTOGRAD.md。
最终联合回归1508通过/77个既有类别精度失败（68项rtau幅值、9项V1词表量化）；
486项六后端组合接线全通过，三类sanitizer各69项通过，480个reference用例无rtau反号。

### 阶段5补充：前向裁剪/元数据缓存及tanh LSE训练评估（2026-09-08）

- 已实现CTA因果key上界、query行元数据缓存，默认full LSE不改数学。
  padding摘要保留identity；新增N191/255/1025回归。前向/codegen/原语111项通过，
  full端到端相关513项通过，三类sanitizer各27项通过。
- 显式`DISM_TILE_LSE=tanh`统一前向/反向重算，使用rl/lse.cu的approx（非approx2），
  passing完整语义保留。前向全实例零spill/无CALL；B1 D64/DV64新增8B stack，未优化。
- 3层D=DV64、V512、N1024、B64的3种子×1000步训练，full/tanh六次均完成且梯度有限。
  本批tanh没有妨碍收敛，但D128/DV32纯soft固定输入发现tau相对FP32 oracle反号，
  已加入实验模式普通失败回归，故不把tanh设为默认，也不宣称全形状精度验收。
- full裁剪+缓存前向三kernel合计1.914ms，tanh为1.594ms（同配置CUPTI每kernel60样本）。
  完整方法、精度/训练曲线、资源变化与性能见dism_v2/FORWARD_OPTIMIZATION.md。
- 用户要求前后向同时近似后再次核对：原开关已覆盖两边，SASS确认B1/B3各新增60处TANH。
  同配置seed0复跑1000步正常完成、hard准确率98.155%。重新计时full/tanh前向为
  1.923/1.602ms、反向6.393/5.543ms；完整训练step68.777/65.455ms（30次采样），
  输入吞吐提升5.1%。默认full及D128 tau反号已知问题不变；复测数据已固化。

### 前向摘要性能诊断后的约束

- NCU结果及旧版组织对照见dism_v2/SUMMARY_NCU.md；当前摘要SM吞吐约19.6%，
  热点为Q标量搬运、key标签load-use、控制流等待，不能把低occupancy当成算法上限。
- 用户新增shared使用标准：计算中间量仅计划内防spill staging、必须warp通信、输出布局整理
  可写shared；其他能在寄存器完成的必须在寄存器完成。不默认新增key元数据shared缓存。
- 用户最新指定摘要优化采用persistent CTA、每SM一个CTA；取消Q直载寄存器和提高occupancy
  的建议。保留Q/K异步输入槽，在当前workload执行期间预取下一workload的Q。

后续实施与验收顺序（本轮仅更新计划，尚未改kernel）：

1. persistent调度保留逻辑(batch,head,query-block)身份与输出索引，平衡不同因果长度任务；
   不把RNG/checkpoint身份绑定物理CTA或warp。以每SM一个CTA为主配置。
2. 将Q改为TMA异步预取。先启动初始Q/K，在当前Q进入寄存器、其输入槽释放后，
   producer预取下一workload Q并与当前任务的HMMA/scan重叠。
   Q槽不能覆盖仍在使用的K ring；明确输入槽释放、ready、跨workload phase及退出drain协议。
3. 保持两个compute warpgroup配对边界依赖，安排稳定阶段WG0 scan/reduce对应WG1 HMMA，
   随后交换。细化数据/边界ready与K槽释放时机，不在不必要的全CTA barrier上锁步。
4. key标签/column-LSE元数据用寄存器预取和复用，减少load-use串行链；
   不为实现方便添加shared中间缓存。K多缓冲深度据重叠效果决定，不以stage数量验收。
5. 先验证跨workload的边界/RNG、尾部及sanitizer，再比较NCU发射效率、pipeline利用和稳定阶段
   吞吐；检查SASS无CALL及资源变化。不将warp stall占比直接解释为可消除的墙钟时间。

### 前向摘要 persistent 计划 P0–P5（2026-09-08，待实施）

本轮仅制定计划；基线提交df561d7，NCU报告保留。范围仅摘要，output/passing/backward不改。

**P0：固定基线与协议。** 参考src/dism_fwd_nope.cu的FixedLengthScheduler（763行起）、
LoadSharedMemoryLayouts（875行起）、摘要producer/consumer（947–1088行）。
沿用静态persistent推进、Q+首K预取、K ring、elect_leader及主循环unroll 1组织。
旧版Prefetch/Default是union且task末有CTA同步，不能直接照搬为跨task重叠。
先列清各槽owner、arrival计数、phase、释放点及退出drain；固定现有full/tanh性能基线。

**P1：最优先完成persistent及Q/首K预取。** 12warp，dec40/inc232，每SM一个CTA。
核心验收是上一workload收尾与下一workload的Q/K0加载重叠，不是task0启动预取。
producer在当前任务最后若干tile仍进行HMMA/reduce/摘要写回时，提交下一任务Q及K0；
不经过task结束的全CTA barrier才开始预取。Q可更早发射，但不能挤占当前K流水供给。
任务切换时consumer等待的是此前已在途的Q/K0；task0仅是不可避免的pipeline prologue。
warp8加载，9–11保留寄存器重分配角色；compute保持0,4,1,5,2,6,3,7逻辑行顺序。
调度从旧版grid-stride开始，检查三角任务负载，必要时确定性长短交错；不默认增加atomic队列。
RNG/checkpoint始终按逻辑batch/head/query-block索引。

- 启动异步提交task0的Q和K0，并填充可用K槽。Q通过TMA进入shared输入槽再载入寄存器。
- Q槽释放后，在当前task计算期间预取下一task Q；当前task所有K提交后，
  利用已释放ring槽预取下一task K0，不等consumer开始新task才加载。
  首K直接使用下一逻辑K slot，不默认增加复制槽；短task重叠窗口需实测。
- K和mailbox按跨task单调tile序号推进phase，Q ready/free独立推进；
  不能每task直接重置phase而复用未释放槽。无效warp仍参与协议，最后明确drain。
- 两WG仅四对边界依赖，K读取结束及时释放，不增加逐task全CTA锁步；
  安排一组reduce与另一组HMMA错位重叠。Q预取不能覆盖仍在使用的K。
- 单独验证交错Q行映射和未padding尾部安全，不让TMA跨batch/head读入有效数据。

**P2：访存流水与score简化。** 在P1正确基线上分别测量增量。

- 审查Q/K、tau、query/key标签、行/列LSE、摘要写出及配对边界。
  大块输入TMA；元数据提前发射寄存器load、延迟消费并shuffle复用，
  区别于硬件异步copy，不为名义异步增加shared中转。tau等小标量允许普通load。
  摘要先保留合并global store，输出异步staging只在必要且有收益时采用；通信仍用mailbox。
- 加载后预算tau2=tau*LOG2E、scale2=scale*LOG2E；行LSE方向在Q元数据阶段算
  bias2=(tau-lse_q)*LOG2E，列LSE方向在每K元数据阶段算bias2=(tau-lse_k)*LOG2E。
  soft为fmaf(dot,scale2,bias2)，hard匹配为tau2，否则负无穷。
  不重复换底，不混淆行列LSE；重排改变FP32舍入，检查边界及反向重算误差。
- 方向、纯soft/纯hard用host dispatch/模板；混合行保留warp内RNG及必要谓词选择。
  纯hard省去无用score MMA和Q/K数值加载作为独立优化，不改变标签匹配及递推。
  完整严格下三角、对角及尾tiles分路径，避免每元素动态if链；控制模板及代码体积。
  不删除负无穷/identity guard以换取少分支。

**P3：发射及循环codegen。** P1起即遵守，随后系统检查。
TMA/expect单线程操作用elect.sync或验证后的TK封装，在收敛warp处选举；
保持正确active mask和arrival计数，不能只换lane0而漏掉其他lane的arrive。
布局指定的owner（如lane31导出边界）不是任意单线程发射，不随意替换。
编译期定长循环显式unroll，动态task/key主循环显式unroll 1。
SASS检查单FFMA score、无冗余选举/重复换底、无CALL、原生TMA/USETMAXREG，
记录分支、代码体积、寄存器/spill；新增spill先汇报，不自行扩大修复范围。

**P4：扩大stage及shared配置。** K stage模板化，D64先比较2/3/4stage，再覆盖D32/D128。
设置MaxDynamicSharedMemorySize及必要的PreferredSharedMemoryCarveout，查询实际opt-in上限；
完整sizeof计入通信、barrier、对齐及硬件限制，不假设整64KiB均可分配。
128行Q槽为256*D bytes，每个64行K槽为128*D bytes。
D64的Q+4K=48KiB，D32的Q+4K=24KiB，均另计通信；
D128的Q+2K=64KiB未计通信已无余量，不能直接套用。
D128验证64行分段Q输入槽的消费/释放（下一task Q分段预取），
使16KiB Q+2K=48KiB；若保留完整Q槽则减少K stage并说明回退原因。
任何布局复用都须证明旧读者已结束；不为occupancy减预算，也不要求统一stage数。

用户追加同步要求：上述四对payload仍独立存放，但mail ready/free改为WG级共享，
每slot各128线程arrival，不再每对32线程独立phase。全组发布/消费后推进，
尽量保持四warp步调一致，降低访存closure时间方差；不增加两组间的全CTA锁步。

**P5：验收与归因。** 分别记录P1 persistent+预取、P2 score/元数据/分支、P4深stage收益。
覆盖D32/64/128、两方向、hard端点/混合、尾部、长链、summary及保存W边界。
专门强制每CTA多个task，覆盖跨head/batch、奇偶tile数、最后task及无效warps；
运行memcheck/racecheck/synccheck，再跑九种D/DV全链路前后向回归。
full/tanh分别比较同输入oracle，既有误差不放宽；检查score重排与未改反向重算的一致性。
报告热身后kernel计时及同stream连续吞吐，重新采NCU比较发射效率、tensor利用率、
load-use和barrier位置，结合SASS判断重叠；stall样本占比不是可消除的墙钟时间，
仅存在双缓冲/TMA不足以证明HMMA与scan有效重叠。

P1首版实测：已接入默认summary dispatch，Q独立TMA输入槽及跨任务Q/K0预取，
任务间不再CTA同步；TK elect/TMA atom/寄存器分配/warp load与MMA。
按用户追加要求mail ready/free已改WG粒度128 arrivals。D32/64三K槽，D128暂一K槽。
full前向+codegen122项通过；首批D64混合tanh摘要CUPTI中位673.277→522.717us，
约1.29x；这不是persistent与stage的单变量消融，也尚未证明HMMA/scan实际重叠。
P2–P4、D128分段Q、多stage搜索和进一步NCU仍待做，原score语义未改。
完整协议、测量条件、原始数据及最终验证见dism_v2/PERSISTENT_SUMMARY.md。

free2/ready1实验已按用户授权撤回：D64/N257卡住，producer前后补syncwarp均未解决；
CUDA调试器看到producer已到末尾CTA barrier、consumer在等Q ready。根因未确定。
当前恢复ready32/free256及score后释放，WG mailbox128不变；baseline122项重新通过。
随后Q改为128行TK tile+subtile读取，Q/K tensor-map外维直接覆盖所有swizzle panels，
每workload Q及完整K tile各一次TMA。新版本122项通过，full/tanh各三个summary实例
恰有两个TMA发射位置，零spill/无CALL。D64首轮523.101→522.718us，未建立显著收益。
尾K仍标量安全加载；单线程发布同步优化暂搁置，详细验证记录见PERSISTENT_SUMMARY.md。

P2首版：仅摘要接入方向特化、寄存器分布式key元数据预取、bias2/scale2/tau2预计算，
score使用FFMA+显式selp；hard端点与tile特化暂未做，output/backward不改。
full原121项及新增12项N513/rtau=ln(D)检查通过，full/tanh均无spill/CALL。
D64混合tanh首轮两方向分别523.261→327.887us、512.237→337.854us；
这是整组score优化的收益，未拆分归因。详情与最终验证见dism_v2/PREDICATED_SCORE.md。

前向摘要补充实验：用户授权 `DISM_TILE_LSE=tanh_finite`，core 前后向使用
`-1e6` log-zero 并移除 tile tanh 的 infinity 分支，完整 chunk passing 不变。
默认配置未切换。66 个端到端相对 guarded tanh 的对照全部有限、tau 无新增符号反转；
6 个长链输出相同，严格 analytic normalization 测试仍有原 tanh 近似误差失败。
19 项 finite memcheck 零错误，27 项训练冒烟/重放检查通过。摘要两方向首测
328/337us→227/247us；范围、限制与复现见 `dism_v2/FINITE_SENTINEL.md`。

Row bitset 实验已按授权接入：CUDA embedding 收尾生成 packed 行决策，host RNG
预留前移，core 前后向直接复用；默认关闭，`DISM_ROW_BITSET=1` 启用。
62 项 bitset 测试及前向 codegen 通过，memcheck 零错误；详细协议、资源和实测见
`dism_v2/ROW_BITSET.md`。首测 dV+摘要约1–1.5%收益，端到端尚无稳定收益，
暂不默认切换、不根据指令数预设加速比。

摘要元数据阶段：int32 标签与 soft/hard/mixed 编译期分发已接入，int64 诊断回退保留。
full251项通过，finite118项和lineinfo memcheck通过；首轮D64混合摘要int64→int32
236.24→221.15us。下一tile元数据寄存器预取及摘要store重排等待新NCU再决定。
详细边界、资源与未测项目见 `dism_v2/LABEL_METADATA.md`。

### 摘要输出寄存器重排实验（2026-09-09）

已比较lane28补列31/63、两条完整warp store的方案。理论sector足迹18→16，
但D64混合int32/tanh_finite三轮CUPTI中q_from_k约慢2.13%，k_from_q约快0.44%，
无一致收益，默认恢复原写回。新增shuffle带来两处collective慢路径与一处BRA.DIV；
full251项通过，finite54组新旧隔离进程逐位一致，memcheck零错误，无spill/CALL。
未测试shared/TMA批量写回或新增NCU。候选与实测见experiments/summary_store/README.md。

### 三级K流水显式K0/K1前导预取实验（2026-09-09）

保持原槽数/同步，仅剥离展开producer前两步。finite117项通过，但D64六实例
出现STACK8；大配置B64/H4/N1024候选超过70秒未返回，终止测试并恢复原路径。
未得到候选耗时，不声称跨任务预取已验证，未继续修spill或猜测卡住根因。
补丁与资源记录见experiments/summary_k01/README.md。

### 摘要退出同步缩小为warpgroup（2026-09-09）

直接删除最终CTA barrier导致full模式N257/q_from_k/D32跨workload测试卡住；
恢复CTA barrier后12项通过。按用户建议改用TK warpgroup::sync(1+warp/4)，
三个WG分别128线程同步退出，保留初始化CTA barrier，组间无需同时退出。
full251项、独立persistent12项通过；新增kernel-only退出/重放12项三类sanitizer
均零错误/hazard。SASS保留原生TMA、inc232/dec40，无spill/CALL。
NCU报告/tmp/dism-summary-wg-exit-lineinfo-q.ncu-rep：229.44us，SM39.74%，
tensor38.54%；旧报告230.40us，未锁频单次profile不声称稳定提速。
未定位无退出同步卡住的硬件/编译器根因，详见experiments/summary_final_barrier/README.md。

### 单线程producer实验（2026-09-09）

先elect再判断warp8/leader，仅该线程运行producer循环，ready32→1；其余producer
线程仅参加WG退出同步，free256/mail128不变。full263项通过，finite codegen通过，
无spill/CALL。D64/N1024三轮q方向222.415→220.6545us，k方向基本持平；
N65/257的leader串行尾块分别约13→35us、37→59us，回退明显。
用户已接受并授权提交为当前summary基线，保留尾部性能回退记录，不能视为所有shape的
通用性能优化；详见experiments/summary_leader_only/README.md。
同配置Triton summary比较：CUDA纯soft约2.69–3.09倍、mixed约2.60–2.75倍相对吞吐；
只比较summary GPU时间，不是端到端训练吞吐，见dism_v2/SUMMARY_VS_TRITON.md。

### OUTPUT主kernel同类优化（本轮完成，2026-09-09）

按summary记录建立完整审计与实施范围，见dism_v2/OUTPUT_OPTIMIZATION.md。
第一步候选已做score/方向/模式/标签特化、寄存器元数据、单FFMA/selp，以及
完整K/V单次TMA与K的LDSM布局修正；尚未改persistent Q/K/V或mailbox同步。
finite codegen无CALL，但纯soft q方向D64/DV32、D128/DV128各新增STACK8，
最初按约定暂停，随后用户已明确授权记录spill并继续。第一步full200项通过。
第二步已接入persistent Q/K/V、按shared预算分配槽数、D128/DV128双段Q，
WG mailbox/退出同步及单线程producer；finite36项多workload全D/DV重放和codegen通过。
协作tail-producer候选有停滞/不一致，已移除；保留安全串行tail并记录性能风险。
最终增加BF16x2成对O写回，消除STG.U16；full170项、finite154项通过，
最终full/finite各507项端到端选定回归通过，反向重算257项通过。
三类sanitizer各36项零错误，full/tanh/finite的CALL/TMA/寄存器重分配检查通过。
OUTPUT只有D128/DV128仍有spill，按用户授权保留；摘要零spill要求不变。
tanh严格core oracle在基线与最终版均120失败/13通过、失败集合相同，未放宽容差；
既有backward全扩展零spill断言失败仍记录，不作为本轮数值失败隐藏。
D64/DV64/B64/H4/N1024/V512、mixed/int32/finite+lineinfo，三轮CUPTI：
q方向797.707→450.494us（1.77x），k方向837.867→455.806us（1.84x）。
九种维度的N1024单轮对照均提升；N65约40→65us回退，N257约124–129→114–115us。
NCU报告/tmp/dism-output-persistent-lineinfo-q.ncu-rep，467.58us、SM37.32%、
tensor37.08%。约95.3%的剩余excessive global sectors来自横边/O写回，
不再是重复score元数据加载；具体WG重叠尚未由时间线证明。
本轮要求的实施/验证/性能审计已完成，未提交；剩余store布局、任务负载均衡、
短序列尾加载和spill优化不声称已完成。完整记录见dism_v2/OUTPUT_OPTIMIZATION.md。

### 三个前向kernel与旧Triton对照（2026-09-09）

OUTPUT优化已提交82f9305；新增独立benchmark_forward_comparison统计正常forward
stream中的summary/passing/output，不含embedding、初始化kernel及CPU间隙。
B64/H4/N1024/D=DV64，CUDA词表512/finite，旧Triton N_VOCAB=N_HEADDIM64。
三轮各30次，保留所有样本：按90样本平均总时间，Triton1906.8us，
CUDA纯soft728.3/744.4us、mixed739.6/752.9us（q/k方向），相对吞吐2.53–2.62x。
单kernel中位数比值summary2.63–3.10x、passing1.72–1.76x、output2.39–2.48x。
不同数学语义，未锁频且Triton有明显样本波动；不作等价算法或训练TPS结论。
逐次总时间与分阶段中位数不能直接相加，见dism_v2/FORWARD_VS_TRITON.md。

### OUTPUT Q/KV存储复用验证（已完成，2026-09-09）

用户授权保守/激进两版。D64/DV64保留默认独立Q基线，新增进程级
DISM_OUTPUT_Q_ALIAS=kv/k隔离构建；均三级流水。
kv版Q覆盖KV2，最后PV后允许下一workload预取；k版SoA的Q覆盖K1/K2，
K读取后经WG同步再释放，最后K消费后允许下一Q预取，V仍由独立ready/free保护。
每任务从slot0开始，phase按物理槽维护；包含少于三级的短任务。
两版full core/replay/codegen各146项、finite labels/bitset/replay/codegen各130项通过。
两版full/finite的D64/DV64各10实例零spill、无CALL，三类sanitizer各10项零错误，
最终k版另重复三轮N129 memcheck各2项通过。最初紧随LDSM释放曾出现一行重放
不一致，已淘汰；仅将源码释放移到MMA后不能约束ptxas调度，最终加WG barrier4/5，
并验证无LDSM跨过该同步后才发布K-free。反向重算257项、full/finite选定端到端
各507项通过，保留既有精度问题及backward零spill断言冲突，不改容差。
动态shared：kv55424B/k55552B，均实现三级（原独立Q两级53376B）。
B64/H4/N1024/D64/DV64/mixed/finite三轮OUTPUT中位数：q方向基线451.556us、
kv417.373us、k462.126us；k方向456.431/447.421/456.462us。
kv在mixed有收益但soft及N65/257回退，k无稳定净收益；用户随后基于mixed主工况
选择默认kv，保留显式none/k，不自动dispatch。
实测、SASS与两份新NCU见dism_v2/OUTPUT_Q_ALIAS.md；本轮未提交。

### 前向scalar与hard_bits复测（已完成，2026-09-09）

前向scalar收尾和hard_bits复测已完成：默认保守kv，混合row_hard删除端点检查，
显式FTZ EX2消除subnormal处理。full263/finite130、标量probe3、反向257、
选定端到端full/finite各507、三类sanitizer各10通过；不宣称既有精度问题修复。
summary零spill，OUTPUT已接受spill未变化。hard_bits含embedding前向总时间仅改善
0.14%/0.07%，全前反向0.56%/0.23%，未锁频且存在波动，保留默认0。
数据、边界语义及复现命令见dism_v2/FORWARD_SCALAR_BITSET.md；未提交。

### OUTPUT BF16 O TMA实验（已完成，2026-09-09）

前置OUTPUT O TMA实验已完成：D64/DV64复用KV1存BF16 O，warp9异步输出，
next Q/KV0与O共用48KiB数据区但地址互不重叠；next KV1等待旧O TMA read完成。
full core133、重放12、finite重放/bitset/标签129、full/finite直接写回对照各60、
选定端到端各507、codegen及三类sanitizer各10通过；D64/DV64无新增spill/CALL。
mixed N1024三轮90样本中位数q406.222→410.576us、k447.198→451.038us，
因此保留DISM_OUTPUT_TMA=0默认，不基于短N单轮收益自动dispatch。
实现、协议与实测见dism_v2/OUTPUT_TMA.md；本实验未提交。

### 反向WS kernel优化（进行中，2026-09-09）

新增反向优化goal进行中，完整阶段见dism_v2/BACKWARD_OPTIMIZATION.md。
已冻结两WS kernel基线并采集NCU；DISM_BWD_OPT=1实验接入寄存器元数据缓存、
selp及FTZ EX2。单FFMA score产生两项新增BF16 GEMM对照失败，恢复旧算术顺序
后回到基线192通过/14既有失败；full/finite选定端到端各507通过。
D64/DV64新增spill已记录并继续。主工况三轮中位数B1加速1.73–1.78x，
B3加速1.14–1.16x，但尚未实现因果裁剪/persistent/跨任务预取/WG mail与stage实验，
目标仍active，暂不切默认或宣称完成。

随后已验证OPT2 CTA因果裁剪（dense摘要补零/identity）、OPT3 persistent及held B/V
TMA预取、完整A/dO单TMA、OPT4 WG mail128。full331通过/14相同已知失败，
finite139通过；含九维度108项相等性和30项多workload重放，端到端各507通过。
两轮主工况OPT4 B1约685us，B3约1443–1562us；NCU报告和spill表见上述文档。
第一块A/dO尚未跨workload预取，阶段数/首块预取、布局/特化和最终全范围验收仍待推进。

OPT5已加入score/dP直接RHS LDSM加载；新增DISM_BWD_STAGES=1/2/3实验入口，
按shared预算回退双槽，dA输出仍独立双槽。finite默认双槽及请求三槽各139通过；
随后finite请求单槽和full双槽各139通过。主工况两轮计时：OPT5改善B3，
三级未改善B1，单槽B1略快但未证明整体更优；轮间波动和各实例spill均记录。
stage1/3各三项三类sanitizer及stage3额外N385两项三类sanitizer均通过，零错误/hazard。
OPT6随后接入首块A/dO跨任务预取：B1独立预算内输入槽，B3复用既有scratch，
首块dB读完后256-reader barrier才允许Gsoft覆盖；输入ring与mail独立计数。
finite/full最终源码各139通过（含18实例noCALL/原生TMA断言），三类sanitizer各5项通过。
严格reference仍192通过/14相同已知失败。实测OPT6 D64 B1持平、B3约5%回退，
D32两kernel回退，保留实验但不切默认。NCU仍显示B3约4路shared-store bank conflict；
OPT7回到OPT5输入流水，仅验证dA输出槽TK FP32 64B swizzle及匹配TMA描述符。
OPT7 finite/full各139通过，三类sanitizer各3通过，但本批性能回退：
NCU显示平均bank conflict下降同时store请求数增加，总wavefront未下降。
OPT8使用TK float2成对输出store，不包含OPT6预取；finite/full各139通过，
恢复请求数并消除主工况新增spill，但整体吞吐仍接近OPT5、bank conflict未实质减少。
三类sanitizer各3通过，零错误/hazard；后续可独立验证float4输出线程归属，
模式特化/全维度筛选仍待推进。
OPT9已接入无swizzle的float4输出归属：相邻lane交换后四lane覆盖完整16列，
不增加shared。finite140通过（新增CPU精确归属/bank检查及STS.128断言），
full也140通过，三类sanitizer各3通过；NCU shared-store wavefront下降约32%，
但指令数增加约3.4%、实测B3慢约2–3%，保留实验、不切默认。
输出布局探索暂收敛，继续方向/标签类型/软硬模式特化和全维度候选筛选。
OPT10已在OPT5布局上接入10种score策略（方向×soft/hard32/hard64/mixed32/mixed64），
两WS kernel共180实例；hard只跳过score MMA，保留递推及所有梯度流程。
finite139通过、额外54项int64高位标签通过，180实例codegen通过；
full/sanitizer/分模式性能验证进行中，不含OPT6–9实验，默认仍OPT0。
后续OPT11对纯hard B3省去零operand/LSE梯度计算，保留G递推及rtau；
finite193项通过，full扩展套件414通过/26项与冻结OPT5相同的已知精度失败。
full/finite全扩展noCALL检查通过，选定端到端各507通过，finite bitset62通过。
纯hard主工况B3降至约556us，mixed基本不变；最终九维度性能及完整计时验收
尚未完成，默认仍OPT0。
详见反向优化文档及backward_rhs_{timing,codegen}_sm120a.json。

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
