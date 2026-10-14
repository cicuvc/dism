# 任务书：Value residual 前向与反向融合 kernel

## 目标与范围

在 `/home/cicuvc/cs/project/dism-core` 实现仓库自有的 Triton value residual 融合算子，面向 RTX 5090 / RTX PRO 6000 的 sm120。融合当前 FP32 转换、sigmoid、逐元素混合及其反向，减少大激活的中间张量读写和 launch。

基线提交：`68363f6`。本轮交付独立算子、autograd 接口、测试、benchmark 和集成结果；**不修改模型数学语义，不重启训练，不修改 checkpoint**。先保留旧函数作为数值和性能参考，通过验收后接入共同 helper；不通过则保留 opt-in 实验入口，不强行切换。

先阅读根目录及 `dism_v3/AGENTS.md`、`dism_v3/docs/AGENTS.md`。涉及模型/编译接入时阅读后者指定的 MODULE、MODEL、DYNAMO 文档。保留用户改动。本文路径相对工作树根目录。

不在范围内：Linear/短卷积融合、GDN 递推、DISM 核心、norm/output gate 融合、CE、hard 路由策略修改、FP32 residual 改 BF16、CPU planner 调优。

## 现有入口与调用关系

- `dism_v3/python/flash_dism/module.py::value_residual_mix`：唯一共同 helper，也是语义参考。
- `dism_v3/python/flash_dism/hybrid_gdn.py::DismGdnAttention.gdn_raw`：GDN 分支调用；被 `torch.compiler.disable` 隔开，目前 helper 在这里以 eager 执行。
- `dism_v3/python/flash_dism/pure_gdn.py::PureGdnAttention`：复用上述 GDN 路径。
- `dism_v3/python/flash_dism/module.py::DismAttention._forward_cuda` 附近的 value residual 调用：DISM 分支，核对实际函数名和编译路径。
- `nanochat/nanochat/models/dism/model.py`：捕获首层 `v_first` 并传给后续层；保持其计算图和共享关系。

生产布局为 `v, v_first: [B,N,H,DV]`，`gate: [B,N,H]`，gate 是 sigmoid 之前的 logits。实际 packed 72M 路径为 `[1,32768,6,64]` 和 `[1,32768,6]`；逻辑训练 batch 为 16×2048。此操作逐 token/head，无需 document 边界或 hard mask。

## 数学及精度契约

现有前向：

```python
g = torch.sigmoid(gate.float()).unsqueeze(-1)
mixed = v_first.float() + (v.float() - v_first.float()) * g
out = mixed.to(v.dtype)
```

令 `a=FP32(v)`、`b=FP32(v_first)`、`s=sigmoid(FP32(gate))`，上游梯度升为 FP32 后记为 `dy`：

```text
delta   = a - b
out     = cast_v_dtype(b + delta * s)
dv      = dy * s
dvfirst = dy - dy * s
dgate   = sum_DV(dy * delta) * s * (1 - s)
```

要求：

- 前向内部混合、反向乘法、DV 归约均使用 FP32，不引入 BF16 中间乘积/归约。
- 各输入梯度分别返回各自输入 dtype，不假设 gate 与 v 总是同 dtype。
- `dvfirst = dy * (1-s)` 数学等价，但不一定与当前 autograd 的舍入顺序一致；先按现有图确认实际运算顺序。sigmoid 实现、FMA 收缩、乘法结合顺序及归约顺序造成的差异均需报告，不能仅以公式等价替代数值验证。
- `dgate` 只沿 DV 归约，是 per-token/head 梯度，**不沿 token/head 求和**。gate Linear 的参数梯度仍由原来的 Linear backward 负责。
- `v_first` 必须参与反向，不能 detach。跨层共享的梯度累加交给 autograd，kernel 不得写入输入或参数 `.grad`。
- 不原地修改输入、上游梯度或 saved tensors。支持 retained graph 重复 backward，不能重复乘缩放或重复累加到保存的 buffer。
- 同一张量同时作为 `v` 与 `v_first` 时，两个输入分支的梯度必须由 autograd 正确合并。
- 首版支持训练需要的 BF16 和 FP32；不支持的 dtype/布局明确 fallback 或报错。CPU 及现有非 CUDA 使用保留参考路径。不要为了验证声称支持 FP64 却在内部静默转 FP32；FP64 仅作为数学误差参考。

## 实现建议

建议新文件 `dism_v3/python/flash_dism/kernels/value_residual.py`，不修改 site-packages。

1. 前向一个 kernel：按 token/head 行加载 gate、v、v_first，在寄存器中完成 FP32 运算，直接写最终 dtype。
2. 反向一个 kernel：同时输出 dv、dvfirst、dgate；DV 内归约放在 CTA 内，不需要跨 CTA atomic。
3. 可由一个 CTA 处理多行，比较行数和 num_warps；DV64 是首要目标，同时覆盖 DV32 与尾部掩码。
4. 优先保存原输入并在反向重算 sigmoid；若比较保存 FP32 sigmoid 的方案，报告额外显存与性能，禁止保存完整 FP32 delta/mixed 作为默认方案。
5. 不引入 D2H、`.item()`、设备同步或数据相关 host 分支。避免为了统一布局先调用大张量 `.contiguous()`；训练实际 stride 应直接支持，其他 stride 明确说明处理策略和拷贝开销。
6. 保持共同 helper 的调用签名。eager、Dynamo/AOT 与真实 Inductor 路径都要测试；按现有仓库习惯选择 autograd.Function 或 custom op，不为此把整个 GDN 路径纳入编译。
7. 无需原子操作；禁止 shared FP32 atomicAdd。不要为这一逐元素操作引入无必要的 shared memory。

本轮只要求一阶训练梯度；二阶梯度若未支持，应显式说明，不能误报为已验证。

## 正确性验收

以原 helper 的 eager autograd 为主要参考，另用 FP64 数学参考区分输入舍入和实现误差。测试 loss 不限于简单 sum，使用随机、非均匀的 upstream gradient。

至少覆盖：

- BF16、FP32；实际 gate dtype，以及声明支持的混合 dtype 组合。
- 真实 `[1,32768,6,64]`，DV32/64，小尺寸、不同 B/H、非整 CTA 行数；若声明支持任意 DV，增加非 2 的幂维度。
- gate 为零、正负大值及饱和区；v≈v_first、大幅值、梯度正负抵消。
- 连续、真实训练 stride、切片形成的非连续张量和非连续 dout；不支持情况须验证 fallback/异常。
- 分别冻结各输入、只计算部分梯度；同一 v_first 供多层/多个分支共享。
- v 与 v_first alias、retained graph 两次不同 upstream backward。
- eager、编译 helper、实际调用方在 torch.compile 下的前反向。检查没有新增 graph break 或每步重编译。
- 小模型同权重同输入的前向和全参数梯度对照，尤其首层 value projection、后续 gate Linear 的梯度。

报告 out/dv/dvfirst/dgate 的最大绝对误差、相对 L2、cosine、norm ratio；近零梯度额外看绝对误差。新增容差要有 BF16/FP32 舍入依据，不修改现有测试容差以通过。不能只检查 loss 或只检查非抵消样本。

检查边界和同步，运行适用的 compute-sanitizer memcheck/racecheck/synccheck；记录不支持或未覆盖项。检查编译结果的 CALL、寄存器和 spill，说明发现及影响。

## 性能验收

### 独立算子

在空闲 GPU 上比较：原 eager helper、可编译的原 helper、新融合实现。编译基线只是补充：真实 `gdn_raw` 当前禁用编译，不能将孤立编译结果冒充已接入收益。

主要配置：32768 tokens、H6、DV64、BF16，另测 DV32、小序列及真实 stride。先预热编译/autotune，再多次测量前向、反向、完整 autograd；固定输入、相同梯度累加/清零方式。分别报告 CUDA event 和 host wall 时间、分位数、峰值显存、kernel launch 数。CUDA graphs 默认关闭，与既有模型基准一致。

目前 profile 中 `[1,32768,6,64]` 的独立 FP32/BF16 转换合计约 3.67 ms/microbatch，但这包含多个模块，**不是 value residual 可全部消除的耗时，也不是性能承诺**。融合还可能减少对应逐元素 kernel；应按调用范围重新测量，而不是直接引用全局 copy 数字。

### 72M 模型验证

复用 `dism_v3/tools/bench_model_training.py` 的 scratch/resident-batch replay；不得启动真实训练或继续消费 dataloader。对照保持：

```text
checkpoint step: 19074
parameters: 72,035,808
logical B=16, N=2048, H=6, DV=64
accumulation=4, 131072 tokens/step
conv=linear_fla
hard-backward-policy=ready
planner workers=16，保持相同 affinity
CE=chunked document-balanced（与基线相同的 process-local review 接法）
```

只读 checkpoint 目录：

```text
/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres
```

已有 scratch CE wrapper：`dism_v3/build/ce_review_model.py`，属于本地 ignored 工件，不保证其他 checkout 存在；先检查并复用。若缺失，明确报告并准备等价的审计过的接入，不得悄悄退回旧 CE/token mean。4 个 resident batch 的有效 document 数为 34/30/32/43，step 归一化按总数 139。保持 checkpoint RNG、annealing、optimizer 状态和 dataloader 位置；只在内存 scratch 副本上测试。

现有 ready 基线为约 792.1 ms/step、165.5k tokens/s；独立阶段平均前向 391.5 ms/step、反向 393.7 ms/step。这些是历史参考，不代替本轮相同环境下的 A/B。建议至少 4 warmup、3 个 8-step 窗口，使用旧→新→旧顺序校验漂移。

报告完整 step、持续吞吐、前反向、峰值显存及 event/quadratic 路由比例。融合可能改变 planner 的可用时间和 fallback 比例，必须同时记录，不能仅从微基准推导端到端收益。profile 单独采集，不混入吞吐结果；若使用 nsys，保留 `.nsys-rep`。

基线工件：`dism_v3/build/hard_dispatch_ready_idle.json`、`hard_dispatch_ready_profile.trace.json`、`hard_dispatch_ready_profile.breakdown.json`。不要覆盖旧结果。外部盘缺失或 GPU 有其他任务时报告并停止相关测量，不停止其他人的作业、不下载数据。

## 环境、交付与完成标准

使用 conda `blkw`，Python 为 `/home/cicuvc/miniconda3/envs/blkw/bin/python`，只加载本工作树包：

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/dism_v3/python/flash_dism/kernels:$PWD/nanochat"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1
```

交付：

- 融合前向、反向、autograd/compile 接口，以及旧参考入口。
- 正确性回归测试和可复现微基准。
- 共同 helper 的受控接入，无参数/state_dict/optimizer schema 变化。
- 数值、sanitizer、CALL/spill、微基准和完整模型结果文档；列出限制和未完成验证。

结果 JSON、日志、trace、编译产物均放 ignored `dism_v3/build/`，不得进 Git。不可更改原模型 FP32 residual 语义、hard/direction/RNG/CE 目标函数或其他 kernel 参数来制造收益。若独立算子更快但训练不改善，报告真实结果和路由/调度原因，不扩大改动范围。
