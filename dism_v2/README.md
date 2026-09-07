# 当前 CUDA core 状态

本目录保留用户的 `dism_ref.py` / `emb_kernel.py`，新 CUDA 实现在 `csrc/`，入口为 `core.forward`。
这是 **fixed-direction core 前向初版**，不是完整可训练的 voc_dism；不兼容旧接口。

## 已实现

- 三个 CUDA kernels：32行 affine 摘要、对角线 passing、重算 scan + online softmax + PV。
- 16x64 warp tile，128行/CTA，compute warps 按 `0,4,1,5,2,6,3,7` 交错。
- 每 CTA 384线程：两个 compute warpgroups + 一个 producer warpgroup。后者仅 warp8 执行加载，另外三个 warp 参与寄存器释放和必要的 CTA 同步。
- K/V 双缓冲，配对 bottom mailbox 双缓冲；consumer A 完整 scan/reduce 后传出边界，未拆 upsweep/downsweep。
- 输入 A/B/V 是 BF16，D/DV 独立取32/64/128；LSE、rtau、logM、affine pair、normalizer 和累加器为 FP32。
- 固定 `q_from_k` / `k_from_q`；hard_prob 仅支持标量0或1。未实现混合 RNG，接口明确拒绝而非生成 global mask。
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

在仓库根目录使用 blkw：

```bash
MAX_JOBS=2 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_core.py tests/test_dism_v2_codegen.py
```

`DISM_VERBOSE_BUILD=1` 可显示首次构建的 ptxas 资源，GLX 路径由 `GLX_ROOT` 覆盖。
core 数值用例49个：九种 D/DV × 两方向 × soft/hard 共36个，另有13种尾部长度，最大N=513，B=H=2。
检查 O、log2 normalizer、所有局部摘要的两个分量以及所有已解析 checkpoint（含 padding）。
O 对 BF16 interpolation reference 的断言容差是 atol=0.008/rtol=0.012，
L2 是2e-5，double affine/boundary 对照是3e-5；这些是验收阈值，不是实测最大误差。
`test_dism_v2_codegen.py` 防止 setmaxnreg 被静默忽略、任何 CALL 或 local load/store 回归。
49个数值用例加 codegen 检查共50项通过，连同环境/oracle 准备测试总计131项。
尾加载的 ready barrier 使用32个 producer 线程各自 arrive，而非单个 leader 代为发布；
这修复了 racecheck 在普通 shared 写入与 consumer ldmatrix 之间报告的竞争。
最终49个 core 用例均完成 memcheck、racecheck、synccheck：0 errors，racecheck 也是0 hazards/0 warnings。
复现完整检查用 `bash scripts/check_dism_v2.sh`，脚本只对 mangled name 含 `_ZN7dism_v2`
的新 kernel 做 sanitizer instrumentation，不检查用于对照的 PyTorch kernels。

## 未完成

- mixed hard_prob、PyTorch generator seed/offset 契约和全局 random direction。
- 广播 hard_prob、一般 stride、varlen、sm90、backward、embedding 全链路集成。
- 当前遍历全部 key tiles（包括因果上三角的 masked 工作），尚未进行因果裁剪、单/双缓冲比较或性能测量。
- 当前每次调用建立 TMA descriptors；CUDA Graph 支持尚未验证。
- 摘要和边界是矩形存储，空间 `12*B*H*ceil(N/32)*padded_N` bytes；仍是二次增长。
