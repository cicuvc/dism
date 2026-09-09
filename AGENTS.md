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
- 用户最新授权独立 `DISM_TILE_LSE=tanh_finite` 实验：core 前后向以 FP32 `-1e6` 表示不可达状态，tile tanh LSE 不检查 infinity；chunk passing 保留完整 LSE 公式。默认 full/tanh 不变。此为有界数值近似而非精确 affine identity，须显式记录上下文/score 范围、长链误差与对照结果，不将此结论外推到无界低层输入。

## 随机数与重算

- 前向summary/OUTPUT混合特化使用row_hard<true>省掉概率端点检查，其他未特化
  调用保留端点快速路径。online softmax及共享log-affine使用显式ex2.approx.ftz.f32，
  有意将subnormal指数输出清零；chunk passing仍是完整log1p公式，不换tanh。
  最新hard_bits A/B未见稳定净收益，默认仍0；实测见dism_v2/FORWARD_SCALAR_BITSET.md。

- 生产路径在 warp 内生成行决策所需随机数，不预先生成或保存 global-memory 行随机数/行 mask 数组。
- 用户最新授权 bitset A/B 实验：`DISM_ROW_BITSET=1` 在 CUDA embedding 收尾生成每32行一个 uint32，core 前后向复用；RNG 身份和消费约定不变。此为上述“不保存行 mask”约定的明确实验例外，默认保留重算路径。
- 使用可重放的 counter-based RNG 或等价方案，将逻辑 `(sequence/batch, head, query row)` 映射到随机数。不得使用会随 CTA 调度、warp 所属或 key tile 改变的随机身份。
- 同一行在不同 key tiles、前向摘要、前向重算和反向重算中的决策必须一致。只保存 seed、offset、选定的全局 direction 等少量元数据；反向不再消耗新的随机数。
- 与 PyTorch generator 的 seed/offset 管理方式、每次调用的随机数消费约定必须显式记录并测试。
- 当前 core 使用 Philox4x32-10，逻辑行 `(batch*H+head)*N+row` 为 subsequence，offset/4 为 block counter；取第一个 word 的高24位生成 [0,1) uniform，与 FP32 概率比较。固定方向的混合标量概率调用在 generator mutex 内预留4 words，0/1不消费；全局 random direction 额外预留前置4 words，host 计算该 block 的 subsequence0、第一个 word 的最低位，选择本次调用的统一方向，无 GPU→CPU 同步。RowRNGState 保存后一个 block 的行 offset 和已选方向，重放不消费。每个 compute warp 在 key 循环前由16个 lane 各生成一行，shuffle 分发，摘要/重算共享同一身份。当前不支持 CUDA Graph capture 和概率广播；详见 `dism_v2/README.md`。
- reference 的显式 `hard_mask`/`interpolation` 可用于调试对照，但不是生产路径的预计算 mask 方案。调试导出的 mask 不得进入正式性能路径。

## 实现组件与布局

- OUTPUT新增D64/DV64的Q存储复用实验，进程首次构建前设置DISM_OUTPUT_Q_ALIAS=kv/k，
  用户因主要工况为mixed选择默认kv；none/k仍可显式选择，其他维度保留原实现。kv为Q/KV2 union三级流水、最后PV后预取下一任务；
  k为Q覆盖SoA的K1/K2，V独立，K读取经WG同步后释放并允许下一Q预取。
  最初紧接LDSM释放曾有memcheck执行下重放失败；单纯把源码arrive移到MMA之后
  不能约束ptxas调度，最终k版使用WG barrier4/5保证共享读完成，codegen验证释放前
  有该同步且没有LDSM跨过它。两版最终三类sanitizer各10项通过，full/finite D64/DV64
  零spill、无CALL。kv仅mixed有实测收益，soft/短N回退；k无稳定收益，仍不默认启用。
  详见dism_v2/OUTPUT_Q_ALIAS.md；本条不授权继续扩大实验范围或自动切换dispatch。

- OUTPUT主kernel已完成summary同类优化（2026-09-09）：90个D×DV×方向×模式/标签
  特化，与30个summary合计120处inc/dec；score元数据寄存器缓存、单FFMA/selp。
  每SM一个persistent CTA，独立Q TMA槽与跨任务Q/K0/V0预取，K/V完整tile各一条TMA；
  D128/DV128为两个64行Q phase，先发K0/V0再等待Q0释放以装Q1。
  K/V槽数按D行、DV列32/64/128分别(3,3,2)/(3,2,1)/(1,1,1)，
  ready1/free256、WG mail128、WG独立退出；非对齐K/V尾部保留安全leader串行加载。
  O为BF16x2成对写回，未新增shared中间布局；summary及外部GLX/TK未改。
  用户允许继续spill，当前OUTPUT只在D128/DV128部分实例有spill；摘要仍须零spill。
  full/finite选定端到端各507项、三类sanitizer各36项通过，九维度N1024性能提升，
  N65串行尾加载明显回退。tanh严格oracle基线与新版均120失败/13通过，保持失败可见。
  本条覆盖旧OUTPUT非persistent/双槽及39处重分配计数；详见dism_v2/OUTPUT_OPTIMIZATION.md。

- 用户最新授权：OUTPUT core同类优化遇到spill记录并汇报，但不再因此暂停；
  继续正确性、流水线及性能验证，不放宽数值容差。覆盖下文历史spill停报约定。

- 当前已获用户接受的summary基线采用leader-only producer：producer WG dec40后先elect，仅warp8的
  elected线程进入task/key循环；其余线程走WG退出同步，不直接return。K-ready1，
  K-free256、mail128不变。尾块暂由leader串行安全搬运/补零，因此N65/257有明显
  性能回退，不视为通用性能优化。详见experiments/summary_leader_only/README.md；
  下文ready32为此前基线，free2/ready1历史卡住实验仍非本方案的根因证明。

- 摘要退出同步改为三个warpgroup分别执行TK warpgroup::sync(1+warp/4)，
  named barrier1/2/3各128线程；仅初始化保留CTA barrier0。直接删除所有退出同步
  曾导致N257跨workload测试卡住，WG同步版本通过full251项及kernel-only12项三类
  sanitizer；不能据此断言具体硬件根因。详见experiments/summary_final_barrier/README.md。

- 摘要元数据新版：int32/int64 标签由 host 分发，soft/hard/mixed 编译期特化；
  soft 不加载标签，hard 不加载 LSE，int64 诊断输入不截断。D×方向×5 共30个摘要实例，
  加9个output共39处寄存器重分配。高层保留 embedding 的 int32 标签，不再转 long；
  output/backward 通过类型标记读取，摘要逐元素无标签宽度判断。此条覆盖下述六实例计数。

- 当前摘要score已按行/列LSE编译期特化（D×方向共六实例），tau2/scale2及行bias2预计算，
  每lane预取两个key标签、列方向再预取两个bias2，用shuffle分发，无shared元数据缓存。
  逐元素为单FFMA+显式selp，保留hard/因果/identity的负无穷；未做hard端点/tile特化。
  output与backward暂保留旧求值顺序，rtau=ln(D)重排回归不放宽容差。
  full/tanh六摘要实例零spill、无CALL，各两处TMA/四处LDG.E.64；详见PREDICATED_SCORE.md。

- 当前已按用户授权回退未定位的free2/ready1实验：K ready32/free256、score后释放，
  WG mailbox各128 arrivals不变；不保留试探性named barrier及producer额外syncwarp。
  单线程代表WG释放的后续实验仍必须先同步整组，不能漏掉未完成的读者。
  D64/N257卡住的根因尚未确定，增加producer同步未解决；不把phase推断记为已证实原因。
- 摘要Q改为TK st_bf<128,D>，每warp通过subtile读取对应的交错16行；
  Q tensor-map box为[S,128,1,D/S,1]，K box为[S,2,4,8,D/S]，S=min(D,64)。
  每个Q workload及完整K tile各一条TMA，包括D128；不再按warp或swizzle panel循环发射。
  未padding K尾部暂仍走安全标量加载。full/tanh三shape均恢复零spill、无CALL，
  codegen断言每个summary实例恰有两个UTMALDG.5D发射位置。

- 优先使用ThunderKittens已有primitive（用户要求），自定义PTX/封装仅保留布局、工具链
  或精确协议确有需要的部分，检查生成代码而不假设封装必然高效。
- 当前摘要P1已改为summary_persistent：每SM至多一CTA，独立Q TMA输入槽，
  当前任务收尾时producer提交下一任务Q/K0，任务间无CTA barrier。
  D32/64三K槽，D128暂一K槽；WG级mail ready/free跨任务按累计tile计数推进。
  此条覆盖旧摘要Q/K union staging和非persistent调度描述；output仍保留原实现。
  score FFMA、元数据流水、分支特化及D128分段Q仍待做，详见dism_v2/PERSISTENT_SUMMARY.md。

- 用户最新要求：摘要mail同步以两个warpgroup为粒度，不再四对warp各自同步。
  四对边界payload布局不变，但每slot共用WG级ready/free（各128线程arrival）；
  全组写完再发布，全组读完才可复用。组内四warp尽量步调一致，减少访存请求/响应收尾方差。
  此条覆盖历史的逐对独立mailbarrier设计；不因此增加两WG间的全CTA锁步。

- 摘要优化新增约定（2026-09-08）：访存尽量异步并排流水级，小尺寸例外需说明。
  persistent及跨workload预取优先，参照src/dism_fwd_nope.cu：重点是上一workload收尾时
  预取下一workload的Q和首块K，不是只优化task0启动加载。
  预先计算scale2=scale*LOG2E、tau2=tau*LOG2E、bias2=(tau-lse)*LOG2E，
  soft逐元素仅fmaf(dot,scale2,bias2)，hard匹配直接tau2；行/列LSE按各自复用范围预处理。
  将方向、hard端点及完整tile判断移出逐元素动态分支；不删除负无穷、identity或尾部安全语义。
  单线程发射用elect.sync或经SASS验证的封装，不固定lane再补选举；布局指定的owner lane除外。
  编译期定长循环显式unroll，动态workload/key主循环显式#pragma unroll 1。
  K stage模板化，按SM120约64KiB总shared预算并通过attribute启用较大配置，
  必须计入Q预取、K ring、通信及对齐。实施顺序与验收见IMPLEMENTATION_PLAN.md的P0–P5。

- **Shared memory使用标准（用户最新明确要求）**：能在寄存器中完成的操作必须在寄存器中完成。
  除以下三类外，其他操作不得写shared memory：
  1. 计划内、为了防止spill的shared memory staging；
  2. 设计中必须的warp间通信；
  3. 输出结果的布局整理。
  不得为了编码方便、统一接口或便利的中间布局而增加shared中转。
  新增或保留一处shared写入时须说明其属于哪一类、为何必要、生命周期和同步协议；
  “可能更快”或“方便TMA/布局”本身不是例外。已有实现不因本条自动获得改写授权，
  按当前任务范围逐处审查；不能把本条理解为允许未经验证地删除必要通信或既定staging。
  用户随后明确要求的Q/K异步输入流水按指定设计保留：Q不直接global→寄存器，
  使用persistent CTA在当前workload计算期间预取下一workload的Q到TMA输入槽。
  这是明确指定的异步数据入口，不据此允许额外计算中间量写shared。
  key元数据优先寄存器预取/复用及lane shuffle，不默认给key标签增加shared缓存。

- 当前摘要性能主方案为persistent kernel、每SM一个CTA，不以提高occupancy为优化目标，
  不为增加resident CTA数压缩寄存器预算或牺牲tile/指令级并行。
  尽量异步加载并将延迟藏在计算后面；下一workload的Q应在开始计算前预取就绪，
  不沿用每个workload开头compute warp标量搬Q后全CTA等待的路径。
  双compute warpgroup有意错开阶段：尽量使一组CUDA-core scan/reduce与另一组HMMA重叠，
  不能只证明“用了双缓冲/warp specialization”就认为发生了实际重叠。
  以稳定阶段的吞吐、发射效率和时间线验证；NCU低occupancy是描述，不是当前优化目标。

- CUDA embedding前向已实现（计划dism_v2/EMBEDDING_CUDA_PLAN.md，实测dism_v2/EMBEDDING_CUDA.md）：
  每CTA同一64个token，WG0四warp各16行acc_qo=softmax(k E_k^T) E_q，
  WG1四warp各16行acc_ko=softmax(q E_q^T) E_k。8compute+4producer组成12warp，
  仅warp8加载E_q/E_k共享双缓冲，producer group整体释放寄存器；初始dec40/inc232。
  词表槽须等两个组的score与PV读取全部结束才复用，无scan/组间边界依赖。
  D32/64/128完整CTA均零spill、无CALL，原生TMA与dec40/inc232已验证。
  voc_dism通过embedding_backend="cuda"显式选择前向，默认仍为Triton。
  embedding_backward_backend="cuda"/"cuda_symmetric"分别选择配对/对称CUDA WS反向，默认Triton。
  八项输出和反向保存状态必须来自同一次被选中的前向，不混用两个backend的LSE/插值。
  保留单warp双独立FA作数值基线，但它与WS的CTA行数不同，不能作为流量减半的受控性能证明。
  D32/64另有block_v=128低层实验选项，两个实例零spill；完整V较64略快但本批仍慢于Triton，
  短V尾部可能变慢，不据此统一更改默认。高层CUDA embedding暂保持64步长。

- 当前前向主方案：warp tile 16x64，32行 checkpoint，128行/CTA，compute warps 0–3 与 4–7 组成两个交错 warpgroup，连续16行块依次交给 0,4,1,5,2,6,3,7。只保留 0→4、1→5、2→6、3→7 的配对边界依赖，各 compute warpgroup 内四个 warp 独立。暂不拆 GLX upsweep/downsweep。
- 当前反向主方案按key转置分块：每warp持有16个key，流式加载64个query，dV/dB在warp内累积直接写回，dA使用FP32 atomic。前向已通过可选save_boundaries接入原W坐标下竖16/横64粒度的FP32标量W₂边界；真实转置MMA/TMA、query列RNG和独立重算已在experiments/glx_recompute验证。warpgroup配对通信计划只用于reverse add-mul scan（4→0等）。q_from_k方向的dB是插值梯度，不能无条件称为dK。具体阶段与存储预算见IMPLEMENTATION_PLAN.md；dV已独立接入，完整反向尚未实现。
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
- dV融合摘要路径已接入真实dP/E和16-key reverse reduce，单个passing kernel组合16→32并传递G边界；暂未合入配对warp调度，额外16-key local是阶段性空间开销。融合路径为压缩寄存器生命周期在shared中暂存dV累加值和W，显式数组lane为最内层，不使用曾越界的TK FP32 shared读回。详见dism_v2/BACKWARD_SUMMARY.md；摘要oracle区分同重算W状态与理想reference梯度。
- dV已作为独立正确性路径接入backward.value_gradient：单warp CTA、16key×64query、FP32 warp累积无atomic，重放前向状态/RNG；P使用BF16高位+残差两次MMA以避免长序列单次BF16转换的精度失败。当前未合入12-warp producer流水或反向摘要，资源141–254寄存器、零spill；不得直接沿用前向232预算。详见dism_v2/DV.md。
- reverse add-mul三步probe见`experiments/glx_reverse`：32-key摘要、逆向passing、4→0配对双槽通信已通过FP64/三类sanitizer验证。reverse HState编码列1…64，列0由VState补齐；与forward的-1…62不同。当前尚未融合梯度GEMM或反向producer流水。

- dA shared转置/TMA输出probe已通过：`experiments/glx_da_tma`。Gsoft写shared时恢复逻辑query列，
  TK col-layout加载后按16x32分块MMA，FP32 shared双槽经原生UTMAREDG.2D.ADD异步累加。
  每writer fence+syncwarp发布，leader wait_group.read后方可复用输出槽；退出前wait_group0。
  未padding尾部和多warp/CTA冲突累加通过；此独立probe资源不能外推到完整12-warp反向。

- B3真实G前置验证见`experiments/glx_g_recompute`：单warp32key顺序处理高/低16半块，
  从真实score/dP/E、前向W边界及生产G32边界恢复G；之后独立验证dA TMA和dB独占GEMM。
  149项及三类sanitizer通过，G probe168–250寄存器、零spill。完整G仅作诊断输出，
  生产B3不得物化；尚未完成dA/dB融合或dLSE/drtau归约，不外推12-warp资源。
- B3融合`core_ab.cu`已接入`backward.operand_gradient`：单warp32key，Gsoft仅在shared交接，
  两个dB accumulator常驻寄存器，dA用3D TMA FP32 reduce-add。D32/64六实例零spill，
  D128三实例255寄存器、24–32B stack spill已获用户授权保留并继续正确性验证，不自行优化。
  同独立G/GEMM对照通过；直接reference在6个rtau=ln64纯soft用例有逐元素精度失败，
  保留普通失败，详见dism_v2/AB.md。当前无dLSE/drtau归约和autograd，不外推12-warp资源。

- B3 WS已新增core_ab_ws.cu，operand_gradient默认启用WS；warp_specialized=False保留单warp诊断。
  每warp16key，一个dB accumulator；A/dO双缓冲需到dB读取后释放。
  reverse inclusive scan按4→0配对双槽传G；Gsoft shared转置加载后，
  每warp2KiB union改作两个16x16 FP32 dA TMA输出槽。改写G前drain旧TMA读取。
  初版D32/64零spill，D128 stack32/144/216 B（DV32/64/128），按用户授权保留，
  不自行优化。完整WS验证状态见IMPLEMENTATION_PLAN.md和dism_v2/AB.md。

- operand_gradient返回FP32(dA,dB,dLSE,drtau)，两个B3路径共享FP32 G归约，
  dLQ atomic、dLK独占、drtau warp partial后按head归约。用户已授权暂时保留
  WS D64/DV128新增56B stack及D128 spill并继续验证，不自行优化。
  同G数值检查通过，但直接reference新增drtau精度失败；reference FP32 O的delta诊断
  大幅降低误差，不据此替换生产delta或放宽容差。实测与sanitizer范围见dism_v2/AB.md。

## 已知旧实现问题

- 旧 CUDA 最终 bwd kernel 为空；跨 checkpoint 的前向传递仍有 Triton 阶段；输出尾部为 RMSNorm，不能直接作为 v2 的正确性基线。
- 旧 CUDA 使用 `[B,N,H,D]`，而 v2 使用 `[B,H,N,D]`，不能混用 stride 或接口。
- `dism_v2/emb_kernel.py::_interp_bwd` 原Phase B多个vocab blocks重复写dq/dk，
  端到端接入时已增加pid_v==0保护（原归属测试失败、修复后通过）。
  D128前后向num_stages改为1以满足sm120 shared上限，没有重写插值算法。
- embedding 返回值顺序遵循 `InterpolationResult` 的定义，不按旧局部变量名猜测归属；直接 score 梯度与 embedding backward 梯度需要相加。

## 验证与协作

- 前向已按CTA因果范围裁剪key循环，并提前加载tau/query标签/query行LSE；
  column-LSE方向仍按key索引。所有角色共享循环上界，无效warp继续参与协议。
  被裁剪摘要在全padding的32行对角线写identity，其余上三角写零映射；保存W边界写负无穷。
- `DISM_TILE_LSE=tanh`为进程启动前设置的实验开关，默认`full`。
  前向与反向重算共用`rl/lse.cu::approx`的tanh系数及FMA顺序，明确处理负无穷；
  不是`approx2`或旧版多项式。跨chunk passing仍调用完整logadd2。
  禁止在同一进程中切换模式或混用不同模式的前向状态；反向仍用原递推梯度，
  不对tanh拟合公式本身求导。训练对照与局限见`dism_v2/FORWARD_OPTIMIZATION.md`。

- 用户最新授权：CUDA embedding反向不再修复或因spill暂停，先验证数值并按实测性能筛选
  配对/对称、token步长16/32/64、D64词表register/shared配置。记录spill但不以零spill
  作为候选准入条件；shared容量超限仍须排除。此约定覆盖下述历史“spill先停报”要求。
  已完成20个配置及七组形状搜索：小CTA网格配对更优，大CTA网格D32/64对称可胜出，
  D128本批仍配对领先；不要把CTA数量分界当作已确定的自动dispatch阈值。
  embedding反向套件394通过/6个既有V1失败，全配置三类sanitizer各20项通过。
  结果及资源见dism_v2/EMBEDDING_BACKWARD.md；随后已显式接入autograd，最终验证见AUTOGRAD.md。

- 完整一阶autograd入口dism_v2.autograd.voc_dism，沿用Triton embedding wrappers，
  core梯度与embedding梯度FP32相加后才转输入dtype。实际输出顺序与LSE路由不能猜测。
  六输入端到端测试仍保留rtau幅值和独立V1 embedding精度失败；rtau单独检查非近零
  oracle的符号（阈值1e-5），本批未反转不代表长期保证。见dism_v2/AUTOGRAD.md。
  此前“embedding backward未接入/未验证”的条目为历史里程碑，不代表最新接口状态。

- 本轮WS/双缓冲工作若出现spill，按用户明确要求先停止并汇报，不自行处理。用户授权比较单次BF16 P MMA、C/D顺序与tanh.approx sigmoid；当前源码为C→D+tanh，32/32和全部DV64四个实例零spill，其余五种仍有spill，不自行继续调参。单warp高位+残差基线不变。WS97项91通过/6已知P量化精度失败，保持普通失败；详情见IMPLEMENTATION_PLAN和dism_v2/DV.md。

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
