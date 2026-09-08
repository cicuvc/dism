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
