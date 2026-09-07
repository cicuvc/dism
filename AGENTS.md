# Dism CUDA kernel 开发约定

## 目标与范围

- 实现 `dism_v2/dism_ref.py::voc_dism_ref` 对应的前向和反向 CUDA kernel，最终覆盖 sm120 与 sm90；优先在本地 RTX 5090（sm120）完成正确性和性能迭代。
- 当前主场景：q/k `[B,H,N,D]`、v `[B,H,N,DV]`，均为 BF16。D 与 DV 分别支持 32、64、128，二者独立，必须覆盖全部九种组合。
- 词表为每 head 的 `[H,V,D]`，rtau 为 `[H]`。明确区分词表大小 V、value 张量 v 与扫描边界状态。
- 首版支持 fixed-length，包括非 tile 整数倍的序列长度；最终支持 varlen。接口、RNG 索引与状态边界设计应为 varlen 留出空间。
- Python、测试及扩展构建使用 conda 环境 `blkw`（本机解释器 `/home/cicuvc/miniconda3/envs/blkw/bin/python`）。无需兼容仓库内旧 Python/CUDA 接口、旧布局或旧构建入口；新实现使用独立 v2 接口，不修改 reference 的数学语义。
- 用户最新明确指令优先于本文。实现、基准测试和算法变更按当前会话授权范围执行；本文件本身不授权启动尚未要求的工作。

## 数学语义

- 唯一算法 oracle 是 `dism_v2/dism_ref.py`。旧 `tt_dism.py` 和 `src/dism_fwd_nope.cu` 仅供参考。
- soft score 使用 reference 的 embedding 插值 Jensen 下界。`direction=random` 每次调用选择一个全局方向，不能改为逐行方向或双向平均。
- 用户目标场景的 rtau 上限为自然对数域 `ln(D)`。测试将范围内与超范围压力结果分开；不要误写成自然对数接口数值上限 `log2(D)`。kernel 当前不替调用方 clamp rtau。
- hard/soft 决策按 `(batch, head, query row)` 生成，一个 query 行的所有 key 共享决策。hard 匹配分数为 rtau，不匹配为负无穷。
- 因果递推为 `W[i,j] = logM[i,j] + softplus(W[i-1,j-1])`，缺失前驱为负无穷。
- 输出分母为 `1 + sum(exp(W))`，固定 fallback 的 log-score/value 都是 0。使用 FlashAttention 风格 online softmax；不得沿用旧 CUDA 的 RMSNorm 输出。
- 不在 global memory 物化 logM、W 或完整注意力矩阵。允许存储经过明确空间预算的 tile/checkpoint 边界和逐行 normalization statistics。
- reference 对外使用自然对数；kernel 可使用 log2/exp2，须明确接口、保存状态和梯度中的换底关系，避免重复乘 scale。
- 反向必须覆盖 q、k、v、rtau、q_voc、k_voc。hard 标签选择不求导，但匹配项仍贡献 rtau 梯度并可沿递推传播梯度；不能跳过 hard 行的递推反向。
- 近似 LSE 必须正确处理负无穷、零映射和 scan identity。禁止用有限负数悄悄替代 hard 不匹配语义。

## 随机数与重算

- 生产路径在 warp 内生成行决策所需随机数，不预先生成或保存 global-memory 行随机数/行 mask 数组。
- 使用可重放的 counter-based RNG 或等价方案，将逻辑 `(sequence/batch, head, query row)` 映射到随机数。不得使用会随 CTA 调度、warp 所属或 key tile 改变的随机身份。
- 同一行在不同 key tiles、前向摘要、前向重算和反向重算中的决策必须一致。只保存 seed、offset、选定的全局 direction 等少量元数据；反向不再消耗新的随机数。
- 与 PyTorch generator 的 seed/offset 管理方式、每次调用的随机数消费约定必须显式记录并测试。
- 当前 core 使用 Philox4x32-10，逻辑行 `(batch*H+head)*N+row` 为 subsequence，offset/4 为 block counter；取第一个 word 的高24位生成 [0,1) uniform，与 FP32 概率比较。固定方向的混合标量概率调用在 generator mutex 内预留4 words，0/1不消费；全局 random direction 额外预留前置4 words，host 计算该 block 的 subsequence0、第一个 word 的最低位，选择本次调用的统一方向，无 GPU→CPU 同步。RowRNGState 保存后一个 block 的行 offset 和已选方向，重放不消费。每个 compute warp 在 key 循环前由16个 lane 各生成一行，shuffle 分发，摘要/重算共享同一身份。当前不支持 CUDA Graph capture 和概率广播；详见 `dism_v2/README.md`。
- reference 的显式 `hard_mask`/`interpolation` 可用于调试对照，但不是生产路径的预计算 mask 方案。调试导出的 mask 不得进入正式性能路径。

## 实现组件与布局

- 当前前向主方案：warp tile 16x64，32行 checkpoint，128行/CTA，compute warps 0–3 与 4–7 组成两个交错 warpgroup，连续16行块依次交给 0,4,1,5,2,6,3,7。只保留 0→4、1→5、2→6、3→7 的配对边界依赖，各 compute warpgroup 内四个 warp 独立。暂不拆 GLX upsweep/downsweep。
- 当前反向主方案按key转置分块：每warp持有16个key，流式加载64个query，dV/dB在warp内累积直接写回，dA使用FP32 atomic。前向已通过可选save_boundaries接入原W坐标下竖16/横64粒度的FP32标量W₂边界；真实转置MMA/TMA、query列RNG和独立重算已在experiments/glx_recompute验证。warpgroup配对通信计划只用于reverse add-mul scan（4→0等）。q_from_k方向的dB是插值梯度，不能无条件称为dK。具体阶段与存储预算见IMPLEMENTATION_PLAN.md；生产反向尚未实现。
- 当前采用12 warps/CTA：8 compute warps + 4 producer-group warps。sm120a 上整个 producer group 执行 setmaxnreg.dec<40>，两个 compute groups 执行 inc<232>；producer group 中仅 warp8 实际加载，其余参与重分配和必要的 CTA 同步。inc/dec 放在各自长期角色分支内，避免立即汇合导致编译器按低预算分配。K/V 双缓冲，单缓冲对照仍待做。query 初始 shared staging 转入寄存器后复用；资源以完整 CTA 编译结果为准。摘要、入边界及 RNG 均按逻辑 checkpoint/行索引，不能绑定物理 warp 编号。
- sm120 当前 TMA 使用 shared::cta：本地工具链下 shared::cluster 的5D加载曾生成外部调用，使 setmaxnreg 被忽略。检查 SASS 的 UTMALDG 和 USETMAXREG，不能仅凭 PTX 或源代码判断指令已生效。实际验证见 `dism_v2/README.md`。
- CTA 行数、checkpoint 高度、warpgroup 数和边界通信方式分别配置。若实测 memory bound，可后续验证 CTA cluster/DSM 内四个等价 compute warpgroup、64行摘要方案；当前不实现 cluster，不预设目标设备支持或性能收益。优先优化计算并行度，不为减少摘要空间引入长串行链。
- 暂定沿用 ThunderKittens 的 warp MMA/TMA 组件，结合 `/home/cicuvc/cs/projects/glx/include/glx/diagonal_scan.cuh`；若实际代码或性能证据支持，可选 CUTLASS/CuTe。避免无依据地混用多个布局系统。
- GLX 与旧 sxdiag 的列布局不同。接入前明确逻辑 `(row,col)`、MMA accumulator `(lane,register,element)`、TMA/shared-memory 地址的映射，并验证正向与逆向路径。
- score GEMM 的 TK accumulator→GLX 列置换已验证可吸收到 B 的 TMA 行加载中，对 warp_k_size=32/64 与 D=32/64/128 均无需 MMA 后 shuffle；公式、5D map 和结果见 `experiments/glx_tma_permute/README.md`。该统一列置换不同于 GLX 的 row-dependent roll，后者仍在寄存器中执行。
- 新旧列布局都可以通过正确配置 TMA 完成随路转换；不要假设必须物化重排。但 GLX 的 row-dependent roll/skew 仍是单独的寄存器操作，不能与统一列置换混淆。
- warp_k_size 优先争取 64，32 可作为回退；128 是后续探索目标，须验证寄存器压力和 spill，不能预设不可行。
- GLX 原公开测试列出 16x64、32x32、16x16；本仓库实验已验证现有模板无需修改即可运行 16x32，具体覆盖及结果见 `experiments/glx_scan/README.md`。多 warp tile 的边界交换和同步由调用方负责。
- 竖16导出验证见`experiments/glx_boundaries/README.md`：16×64的最后register-column c=7不做roll，g=1/3从second直接导出逻辑列15/31/47/63，无额外shuffle；竖16/横64可恢复独立转置tile。更新后的GLX已验证16×128 scan/竖边导出（128 registers、零spill、无新增shuffle），包含它在内的六种默认shape覆盖见实验；32×64的导出映射正确但dense scan数值仍有未定位偏差。32×128数值通过但probe存在spill，不能外推为生产kernel资源已达标。
- 16x64 融合 score→scan→online softmax→PV 及独立 checkpoint 摘要/合成/重算已通过实验，见 `experiments/glx_fused/README.md`。该 shape 的 forward HState 编码底行列 -1…62，列 63 在 VState；不能按普通底行数组加载。现有 TMA tail 实验使用 padded allocation，尚未证明未 padding 输入的安全尾加载。
- 初始 logM 及二元组 `(logM,logM)` 明确使用 FP32，accumulator、扫描状态和归约也使用 FP32；BF16 输入及 Tensor Core 路径中的转换位置需要记录。后续可评估 BF16 logM/二元组，但需单独验证误差。
- GLX 原生 inclusive scan 直接产出 W，不沿用旧 exclusive scan 保存原 score tile、最后再合成 inclusive 结果的做法。让原 score 和不再需要的 affine first 分量尽早结束生命周期；寄存器收益以编译结果为准。
- 前向 scan 前先对标量 FP32 logM 做 roll，再将已 roll 的结果在寄存器中原地 duplicate 为 `(logM,logM)`。不要先 duplicate 再对两个相同分量分别 shuffle，以降低 roll 的 LSU/MIO 压力。保证展开后标量临时值不再独立存活，检查生成代码的 shuffle 数量与寄存器占用；此优化只适用于两个初始分量相同的前向 log-affine 输入。
- `/home/cicuvc/cs/projects/rl/lse.cu` 是近似运算候选。其 approx2 源码包含等待 SASS 修改的 EX2 占位表达式，不得原样当作正确实现使用。先保证源码语义正确，再独立评估指令优化。
- 即使 tile 内 affine 后续使用快速近似，跨 chunk passing 保留完整 `max + log1p(exp2(-abs)) * LOG2E` 语义，控制长程累积误差。
- 不将 sm120 的性能结论外推到 sm90；两者共享数学和扫描组件，分别配置 tile、流水和调度。
- reverse add-mul三步probe见`experiments/glx_reverse`：32-key摘要、逆向passing、4→0配对双槽通信已通过FP64/三类sanitizer验证。reverse HState编码列1…64，列0由VState补齐；与forward的-1…62不同。当前尚未融合梯度GEMM或反向producer流水。

## 已知旧实现问题

- 旧 CUDA 最终 bwd kernel 为空；跨 checkpoint 的前向传递仍有 Triton 阶段；输出尾部为 RMSNorm，不能直接作为 v2 的正确性基线。
- 旧 CUDA 使用 `[B,N,H,D]`，而 v2 使用 `[B,H,N,D]`，不能混用 stride 或接口。
- `dism_v2/emb_kernel.py::_interp_bwd` 的 Phase B 未按 pid_v 限制执行，多个 vocab blocks 会重复写 dq/dk；接入时先复现检查并修复写竞争。
- embedding 返回值顺序遵循 `InterpolationResult` 的定义，不按旧局部变量名猜测归属；直接 score 梯度与 embedding backward 梯度需要相加。

## 验证与协作

- 保留已有用户改动，不回滚或覆盖无关工作。外部 GLX、rl 源码默认作为依赖阅读，不因本仓库任务顺便修改它们。
- 验证按阶段进行：布局/边界 → fixed-length 前向 → 核心反向 → embedding 全链路 → 性能 → varlen → sm90。
- 覆盖九种 D/DV 组合、两个固定方向、hard_prob=0/1/混合、非对齐 N、跨 tile 对角线、全不匹配行、长匹配链及 rtau 不同取值。
- 正确性对照区分 FP32 torch 插值 oracle 和 BF16 embedding-kernel 插值 oracle，分别报告误差，避免把前级量化误差误判为扫描错误。
- 长序列精度测试与实测见 `dism_v2/PRECISION.md`；当前相同 BF16 插值输入的 core 检查通过，但纯 soft 对未量化 FP32 插值存在可复现失败（包括 rtau≤ln(D)）。保持失败可见，不把旧188项通过当成端到端精度验收。
- 实际 emb_kernel 前向接入测试也已固化：同 emb 输出的 core 检查通过，FP32 插值对照仍有失败（含一个混合行用例）。内部 softmax 权重转 BF16 与最终输出 BF16 写回是两处独立量化；FP32 输出缓冲变体仅作测试诊断，不等于生产 core 支持 FP32 插值。embedding backward 未验证。
- 对新增 kernel 进行必要的 Compute Sanitizer 检查；性能记录形状、dtype、GPU、工具链、计时范围、寄存器/spill 和临时存储大小。只声称实际执行过的验证。
- 检查所有最终 SASS 中的 CALL（含 CALL.REL，不只 CALL.ABS），定位并消除未内联或后端生成的调用，建立 codegen 回归断言；同时确保原生 TMA、寄存器重分配与零 spill 不退化。不能用改变 hard/identity 数学语义的方法消除调用。
- 部分有效的128行 CTA 中，无效 compute warps 仍须执行约定的 buffer/barrier 协议；mask 计算和写回，不以提前 return 破坏 producer/consumer 计数。双缓冲的复用、配对边界 mailbox 和 CTA 结束条件都要覆盖非对齐 N 的同步测试。
- 性能优化以完整路径为准，不只比较扫描 microbenchmark。近似计算须报告输出和各输入梯度的误差及长序列行为。
- 不主动启用子 agent，除非用户明确要求并行 agent 工作。
- 执行阶段和验收条件见 `IMPLEMENTATION_PLAN.md`；完成阶段后更新实测结果与剩余问题。
