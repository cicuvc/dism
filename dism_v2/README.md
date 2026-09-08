# 当前 CUDA core 状态

完整六输入的一阶autograd入口现为 `dism_v2.autograd.voc_dism`：
直接使用现有Triton embedding前后向及CUDA WS core。
已进行端到端接线、梯度/reference和sanitizer检查，仍有精度失败，
范围、接口和已知限制见[AUTOGRAD.md](AUTOGRAD.md)。
下文的`core.forward`仍是底层不接autograd的诊断入口。

本目录保留用户的 `dism_ref.py` / `emb_kernel.py`，新 CUDA 实现在 `csrc/`，入口为 `core.forward`。
这是 **支持固定/全局随机方向的底层 core 前向接口**；六输入训练接口见上述autograd入口，不兼容旧接口。

## 已实现

- dV可选融合真实dP/E和add-mul摘要；CUDA passing组合为32-key摘要并逆向传递状态。
  调用和验证范围见[BACKWARD_SUMMARY.md](BACKWARD_SUMMARY.md)，B3与训练接线另见[AB.md](AB.md)及[AUTOGRAD.md](AUTOGRAD.md)。

- 独立FP32 dV入口`backward.value_gradient`：同前向状态的转置重算、key warp独占累积；
  BF16 P高位+残差两次MMA。正确性/精度/范围见[DV.md](DV.md)。

- B3 `backward.operand_gradient`已实现dA/dB，默认12-warp specialization，
  可显式传入`warp_specialized=False`使用单warp诊断基线。验证见[AB.md](AB.md)。
  dLSE/drtau已接入（统一返回四输出），同G归约检查通过，直接reference仍有精度失败；
  用户已允许保留新增spill。embedding backward和六输入autograd已通过独立入口接入；
  已测core分阶段串联不等于完整可训练链路已验证。

- 三个 CUDA kernels：32行 affine 摘要、对角线 passing、重算 scan + online softmax + PV。
- 16x64 warp tile，128行/CTA，compute warps 按 `0,4,1,5,2,6,3,7` 交错。
- 每 CTA 384线程：两个 compute warpgroups + 一个 producer warpgroup。后者仅 warp8 执行加载，另外三个 warp 参与寄存器释放和必要的 CTA 同步。
- K/V 双缓冲，配对 bottom mailbox 双缓冲；consumer A 完整 scan/reduce 后传出边界，未拆 upsweep/downsweep。
- 输入 A/B/V 是 BF16，D/DV 独立取32/64/128；LSE、rtau、logM、affine pair、normalizer 和累加器为 FP32。
- 支持固定 `q_from_k` / `k_from_q` 以及每次调用统一选择的 `random`；hard_prob 支持 [0,1] 标量，混合行决策由 warp 内 Philox 生成，不生成 global mask。
- 安全的非对齐尾加载：完整 K/V tile 用 TMA，最后不足64行由 producer 检查逻辑行并填零到同一置换 shared view。无效 query warps 不提前退出。
- 输出 BF16 O、FP32 log2 normalizer；可选返回32行摘要与底边用于调试，绝不物化完整 logM/W/P。

接口参数是选定 direction 的操作数：

```python
from dism_v2.core import forward
# q_from_k: a=q, b=interp.q_from_k.bfloat16(), lse=interp.q_lse
# k_from_q: a=interp.k_from_q.bfloat16(), b=k, lse=interp.k_lse
out, log_normalizer_2 = forward(
    a, b, v, lse, rtau, q_index, k_index,
    sm_scale=scale, direction="q_from_k", hard_prob=0.0,
)
```

所有输入要求 contiguous、同一 CUDA device；labels=int64、LSE/rtau=FP32。
BF16 interpolation 转换由调用方显式进行，测试 oracle 使用相同转换结果。
有梯度需求时入口明确拒绝，不能将本版当作 autograd 实现。

## 行选择 RNG 契约

使用 Philox4x32-10，逻辑 row=`(batch*H+head)*N+query_row`；counter 四个 word
为 `(low32(offset/4), high32(offset/4), low32(row), high32(row))`，key 为 seed 的低/高32位。
取输出第一个 word 的高24位乘 `2^-24`，与 FP32 hard_prob 比较，得到 hard 决策。
概率先转换为 FP32，因此有相应量化；不承诺与 `torch.rand` 的线程映射或 reference 默认随机 mask 逐位相同。
生产路径每个 compute warp 的 lane0–15 各生成一行，再用两次 shuffle 分发给该行的 MMA lanes；
两枚决策保留在寄存器中，位于 key 循环之外。摘要和重算独立生成同一决策；passing 不生成 RNG。

默认使用输入设备的 PyTorch CUDA generator，也可传 `generator=`。在 generator mutex 内预留
4 个 Philox word（offset 按每个 subsequence 的 word 数计，不是全网格样本总数），
每个逻辑行使用该 block 的第一个 word。固定方向的一次混合调用只推进 offset 4，三个 pass 不分别消费。
固定方向且 hard_prob=0/1 不推进状态。无索引的 `torch.Generator(device="cuda")` 也接受；显式设备须匹配输入。

```python
out, l2, state = forward(
    a, b, v, lse, rtau, q_index, k_index, sm_scale=scale,
    direction="q_from_k", hard_prob=0.37, return_rng_state=True,
)
out2, l2_again = forward(
    a, b, v, lse, rtau, q_index, k_index, sm_scale=scale,
    direction="q_from_k", hard_prob=0.37, rng_state=state,
)
```

`RowRNGState` 只含 seed、offset、逻辑 [B,H,N]、已选 direction 和概率；不含行数组。
显式重放不访问/推进 generator，要求 shape、概率一致，固定 direction 必须与已选方向一致；
`direction="random"` 重放直接采用状态中保存的方向，不再抽样。不能同时传 generator。
`return_rng_state=True` 在通常返回值（或四项 debug 返回值）末尾追加状态。
跨 tile/摘要/重算不依赖 CTA、warp 号；未来 varlen 应把 row 身份替换为明确的 sequence/token 索引契约，
不能直接依赖 padding 或 CTA 排布。当前仅验证既有交错配置，不声称测试了其他 CTA 配置。
CUDA Graph capture 当前明确拒绝；尚未实现 graph-safe generator 状态。

## 全局随机 direction

每次调用/step 选择一个方向，全体 batch、head、query 行及三个 pass 共享，不是逐行随机方向。
随机方向调用在 generator mutex 内一次性预留：前4 words 用于方向，若概率混合则再预留4 words 用于行选择。
host 用相同 Philox4x32-10 实现计算前一 block 的 subsequence0、word0 的最低位：
0→q_from_k，1→k_from_q。没有额外 CUDA kernel、GPU scalar、`.item()` 或 GPU→CPU 同步。
方向和行 RNG 不复用同一 block；endpoint 概率也选择方向，符合 reference 的调用顺序语义。
返回状态的 offset 始终是方向 block 之后的行 offset；端点不使用该行 block，也不为它消费 generator。
因此固定方向消费0/4、随机方向消费4/8 words（分别为端点/混合）；所有显式重放消费0。
不承诺与 reference 的 `torch.randint` 使用相同的随机序列映射，但每次调用的方向是一个公平随机位。

随机选择前必须提供两套操作数。推荐用命名包装入口：

```python
from dism_v2.core import forward_interpolated
out, l2, state = forward_interpolated(
    q, k, v, rtau, interp, sm_scale=scale,
    direction="random", hard_prob=0.37, return_rng_state=True,
)
```

`interp` 是预先准备的 InterpolationResult，两个 interpolation 张量要求 BF16，所有输入 contiguous；
包装入口不运行 embedding、不隐式转换 dtype 或复制布局。底层也可直接调用 `forward`，传
`a=(q,k_from_q), b=(q_from_k,k), lse=(q_lse,k_lse), direction="random"`。
host 先验证两套输入，然后抽样并选择该方向的 A/B/LSE，原有 CUDA pipeline 和 TMA descriptor 仅处理选中的一套。

## setmaxnreg 与 TMA 实测

2026-09-07，RTX 5090、CUDA13.1，conda blkw / torch2.13.0+cu130。
构建目标 `compute_120a/sm_120a`。初始静态预算168 registers/thread；producer group 降至40，consumer groups 升至232：

```
128 * 40 + 256 * 232 = 64512 registers / CTA
```

`setmaxnreg` 放在 producer/consumer 各自长期执行的分支内，之前完成初始 A staging 及其复用同步。
若在分支前重分配后立刻汇合，编译器会保守采用低预算并产生大量 spill；不要按相同行为重构。
producer 的 dec 由全部128线程执行，然后只有 warp8 执行加载；不能仅让 warp8 执行 dec。

另一个实测问题：这里的 `cp.async.bulk.tensor.5d.shared::cluster` 在本地工具链生成外部调用，
ptxas 报 C7506 并忽略 setmaxnreg。改成 `shared::cta` 后生成原生 `UTMALDG.5D`，外部调用消失。
最初24/240分配仍收到 minimum-register 警告；当前40/232且位于角色分支内的配置通过。
没有用 NDEBUG 去屏蔽检查；host 输入校验保留。

最终12个 core 实例（3摘要 + 9输出）全部：ptxas 0 stack、0 spill stores/loads；
SASS 存在 `USETMAXREG.DEALLOC ... 0x28`、`TRY_ALLOC ... 0xe8` 和 `UTMALDG.5D`，无 local load/store。
最终 SASS 中全部 CALL（包括 CALL.REL）也已消除。除 TMA 外，残留调用来自输出除法的 IEEE
特殊值处理慢路径；只加 forceinline 或改用 rcp.rn 仍不能消除。当前每行只求一次
`rcp.approx.ftz.f32` 并以两次 FMA 做一次 FP32 Newton 修正，再乘 numerator。
对扫描最大值保持有限的输入，online denominator 在 [1,N+1] 范围，倒数为 normal FP32；
这不改 log-affine LSE 的计算方式，也未启用全局 fast-math。输出仍通过同一 BF16 oracle 容差。
168是初始分配预算，不应解释成 consumer 只能使用168个寄存器。
这是资源/正确性结论，尚未建立性能对照，不能声称12-warps版已经更快。

## 验证

新增长序列/rtau 上限精度回归及余弦指标验证后、接入实际 embedding 之前，套件为 **301通过、28失败**；失败是 FP32 插值对照的输出精度，
不是相同 BF16 插值输入的 core 检查。包括 rtau≤ln(D) 范围内的14例。实测误差、范围与复现见
[`PRECISION.md`](PRECISION.md)。下文188项全通过指新增压力测试之前的基础回归。

随后接入真实 emb_kernel 的64项测试，单模块48通过、16个 FP32 插值输出阈值失败；
同 embedding 输入的 core 输出/L2 全部通过。实际 embedding 的内部 BF16 权重与输出写回量化分离诊断、
余弦/误差结果同样见 PRECISION.md；尚未接入生产 embedding 自动调用或反向。
最新完整套件为 **349通过、44失败**，全部失败均为 FP32 插值对照输出阈值；没有放宽容差。
更新GLX并加入稀疏边界回归后为350通过、同样44失败，203条已保存精度报告与更新前完全一致。
生产16×64的106项core三类sanitizer复验通过；宽shape验证见 `experiments/glx_boundaries/README.md`。

在仓库根目录使用 blkw：

```bash
MAX_JOBS=2 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_core.py tests/test_dism_v2_codegen.py
```

`DISM_VERBOSE_BUILD=1` 可显示首次构建的 ptxas 资源，GLX 路径由 `GLX_ROOT` 覆盖。
原有 core 数值用例49个：九种 D/DV × 两方向 × soft/hard 共36个，另有13种尾部长度，最大N=513，B=H=2。
检查 O、log2 normalizer、所有局部摘要的两个分量以及所有已解析 checkpoint（含 padding）。
O 对 BF16 interpolation reference 的断言容差是 atol=0.008/rtol=0.012，
L2 是2e-5，double affine/boundary 对照是3e-5；这些是验收阈值，不是实测最大误差。
`test_dism_v2_codegen.py` 防止 setmaxnreg 被静默忽略、任何 CALL 或 local load/store 回归。
RNG 增加九种 D/DV × 两方向的18个混合用例、6个混合尾部用例，以及5个默认 generator / 实际行身份用例；
另有 Philox 零 counter/零 key 的已知向量检查。CPU 整数 Philox 生成 oracle mask，核对摘要、边界、输出和重放。
覆盖64位 seed 高位、offset 超32位、显式/默认 generator、重设 seed、端点不消费及连续调用 offset+4；
全不匹配 labels 的 L2==0 精确标识 hard 行，用于直接核对实际 GPU 决策而不导出 mask。
随机 direction 又增加27项：九种 D/DV × 两个随机结果共18项、两个端点 × 两个方向4项，
以及默认 generator 的5种概率。检查方向位的 CPU Philox oracle、统一方向的 attention reference、
两套操作数接口和命名包装入口的逐位重放、重设 seed 和状态消费。
core 106项、codegen 1项、环境/oracle 81项，总计188项通过。
尾加载的 ready barrier 使用32个 producer 线程各自 arrive，而非单个 leader 代为发布；
这修复了 racecheck 在普通 shared 写入与 consumer ldmatrix 之间报告的竞争。
接入全局方向与行 RNG 后，106项 core 测试均完成 memcheck、racecheck、synccheck：0 errors，racecheck 也是0 hazards/0 warnings。
本次完整日志位于 `/tmp/dism-v2-check.x6Ly3n`（临时目录）；12个 core 实例的寄存器重分配及零 local memory 的 codegen 断言仍通过。
复现完整检查用 `bash scripts/check_dism_v2.sh`，脚本只对 mangled name 含 `_ZN7dism_v2`
的新 kernel 做 sanitizer instrumentation，不检查用于对照的 PyTorch kernels。

## 未完成

- 广播 hard_prob、一般 stride、varlen、sm90、backward、embedding 全链路集成。
- 当前遍历全部 key tiles（包括因果上三角的 masked 工作），尚未进行因果裁剪、单/双缓冲比较或性能测量。
- 当前每次调用建立 TMA descriptors；CUDA Graph capture 明确拒绝，graph-safe RNG 尚未实现。
- 摘要和边界是矩形存储，空间 `12*B*H*ceil(N/32)*padded_N` bytes；仍是二次增长。
# 反向准备：可选扫描边界保存

反向当前进度：`dism_v2.backward.delta(dout, out)` 已实现独立CUDA预处理，BF16输入、FP32逐行dot输出。
core五项梯度的自然对数语义/归约公式已有FP64 autograd测试；独立dV已实现，完整B1/B2/B3和autograd尚未接入。
delta使用已保存的BF16 O，不是对BF16舍入严格求导；详见IMPLEMENTATION_PLAN的core反向接口约定。

`forward(..., save_boundaries=True)` 在原返回项之后追加 `ScanBoundaries`，
如果同时指定 `return_rng_state=True`，RNG state 仍是最后一项。尚无 autograd。
设 `np=ceil(N/64)*64`，两个 FP32 log2 W₂ 张量为：

- `vertical[B,H,np/16,np]`：`vertical[...,e,i]=W₂[i,16*e+15]`。
- `horizontal[B,H,np/64,np]`：`horizontal[...,e,j]=W₂[64*e+63,j]`。

Padding 使用 affine identity 继续传递状态，不是有效注意力权重；包含完整 padded 方阵的边界，
用于后续独立转置重算。未保存 full W/logM。关闭选项不分配边界存储；打开时额外占
`4*B*H*np²*(1/16+1/64)` bytes（N8192、BH1 为20MiB）。目前不复用旧 summary/boundary，
因此这20MiB是额外开销，不是总峰值；异步生命周期与复用后续再处理。

边界由现有12-warp输出kernel直接保存：竖边在scalar unroll前取c=7，横边在现有unroll后取最后一行。
无新增扫描/重排kernel，未改40/232预算。core测试增加所有保存边界的FP64递推对照、
保存开关O/L2/旧摘要的逐位一致性检查。106项core测试三类sanitizer均通过，零错误/零hazards。
真实转置MMA重算的独立验证见`experiments/glx_recompute/README.md`；独立dV已接入，其他梯度尚未实现。
