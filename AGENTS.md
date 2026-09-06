# Dism CUDA kernel 开发约定

## 目标与范围

- 实现 `dism_v2/dism_ref.py::voc_dism_ref` 对应的前向和反向 CUDA kernel，最终覆盖 sm120 与 sm90；优先在本地 RTX 5090（sm120）完成正确性和性能迭代。
- 当前主场景：q/k `[B,H,N,D]`、v `[B,H,N,DV]`，均为 BF16。D 与 DV 分别支持 32、64、128，二者独立，必须覆盖全部九种组合。
- 词表为每 head 的 `[H,V,D]`，rtau 为 `[H]`。明确区分词表大小 V、value 张量 v 与扫描边界状态。
- 首版支持 fixed-length，包括非 tile 整数倍的序列长度；最终支持 varlen。接口、RNG 索引与状态边界设计应为 varlen 留出空间。
- 用户最新明确指令优先于本文。实现、基准测试和算法变更按当前会话授权范围执行；本文件本身不授权启动尚未要求的工作。

## 数学语义

- 唯一算法 oracle 是 `dism_v2/dism_ref.py`。旧 `tt_dism.py` 和 `src/dism_fwd_nope.cu` 仅供参考。
- soft score 使用 reference 的 embedding 插值 Jensen 下界。`direction=random` 每次调用选择一个全局方向，不能改为逐行方向或双向平均。
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
- reference 的显式 `hard_mask`/`interpolation` 可用于调试对照，但不是生产路径的预计算 mask 方案。调试导出的 mask 不得进入正式性能路径。

## 实现组件与布局

- 暂定沿用 ThunderKittens 的 warp MMA/TMA 组件，结合 `/home/cicuvc/cs/projects/glx/include/glx/diagonal_scan.cuh`；若实际代码或性能证据支持，可选 CUTLASS/CuTe。避免无依据地混用多个布局系统。
- GLX 与旧 sxdiag 的列布局不同。接入前明确逻辑 `(row,col)`、MMA accumulator `(lane,register,element)`、TMA/shared-memory 地址的映射，并验证正向与逆向路径。
- 新旧列布局都可以通过正确配置 TMA 完成随路转换；不要假设必须物化重排。但 GLX 的 row-dependent roll/skew 仍是单独的寄存器操作，不能与统一列置换混淆。
- warp_k_size 优先争取 64，32 可作为回退；128 是后续探索目标，须验证寄存器压力和 spill，不能预设不可行。
- GLX 原公开测试列出 16x64、32x32、16x16；本仓库实验已验证现有模板无需修改即可运行 16x32，具体覆盖及结果见 `experiments/glx_scan/README.md`。多 warp tile 的边界交换和同步由调用方负责。
- 初始 logM 及二元组 `(logM,logM)` 明确使用 FP32，accumulator、扫描状态和归约也使用 FP32；BF16 输入及 Tensor Core 路径中的转换位置需要记录。后续可评估 BF16 logM/二元组，但需单独验证误差。
- GLX 原生 inclusive scan 直接产出 W，不沿用旧 exclusive scan 保存原 score tile、最后再合成 inclusive 结果的做法。让原 score 和不再需要的 affine first 分量尽早结束生命周期；寄存器收益以编译结果为准。
- 前向 scan 前先对标量 FP32 logM 做 roll，再将已 roll 的结果在寄存器中原地 duplicate 为 `(logM,logM)`。不要先 duplicate 再对两个相同分量分别 shuffle，以降低 roll 的 LSU/MIO 压力。保证展开后标量临时值不再独立存活，检查生成代码的 shuffle 数量与寄存器占用；此优化只适用于两个初始分量相同的前向 log-affine 输入。
- `/home/cicuvc/cs/projects/rl/lse.cu` 是近似运算候选。其 approx2 源码包含等待 SASS 修改的 EX2 占位表达式，不得原样当作正确实现使用。先保证源码语义正确，再独立评估指令优化。
- 不将 sm120 的性能结论外推到 sm90；两者共享数学和扫描组件，分别配置 tile、流水和调度。

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
- 对新增 kernel 进行必要的 Compute Sanitizer 检查；性能记录形状、dtype、GPU、工具链、计时范围、寄存器/spill 和临时存储大小。只声称实际执行过的验证。
- 性能优化以完整路径为准，不只比较扫描 microbenchmark。近似计算须报告输出和各输入梯度的误差及长序列行为。
- 不主动启用子 agent，除非用户明确要求并行 agent 工作。
- 执行阶段和验收条件见 `IMPLEMENTATION_PLAN.md`；完成阶段后更新实测结果与剩余问题。
