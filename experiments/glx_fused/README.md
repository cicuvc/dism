# GLX 融合路径与 checkpoint 验证

2026-09-07，RTX 5090 / sm120，driver 590.48.01，CUDA 13.1，nvcc `-O3 -arch=sm_120`。
运行方式：在仓库根目录执行 `bash experiments/glx_fused/run.sh`。
脚本打印临时结果目录，保存编译、运行、资源和 sanitizer 日志。
依赖本仓库 TK 与外部 GLX；复用前两个实验的 helper，未修改外部依赖。

这是两个独立 integration probes，**不是 voc_dism 前向实现**，没有 embedding、反向或性能计时。
使用可独立计算的合成 log2 score；double CPU 递推是这些合成输入的数值对照，
不替代正式实现必须遵循的 `dism_v2/dism_ref.py`。

## 1. Score MMA → scalar roll → inclusive scan → online softmax → PV

`fused.cu` 一个 warp 处理 16 个 query 行，顺序遍历三个 64-key tiles。
q/k/v 输入 BF16，score、log-affine 两个分量、扫描边界、max/denominator/output accumulator 全部 FP32。
先 roll 标量 logM，再 duplicate；scan 后丢弃 affine first，只 unroll 标量 W。
不在 global memory 存储 logM/W/P，最终诊断输出 FP32。

布局链路：

- K、V 使用相同 physical-row → logical-row TMA permutation。
- TK score accumulator 对应 GLX `data[r][c]` 的位置是 `tiles[0][c/2].data[r+2*(c&1)]`。
- 权重按相同位置放回 TK MMA A operand；V 用 column-layout register tile，从置换后的 shared view 加载。
- 因而第二个 MMA 计算的是 `P_physical V_physical = P_logical V_logical`，无需权重列 shuffle。
- 行归约每线程聚合 register/elements，再 `shfl_xor(1)`、`shfl_xor(2)`；每线程有两个 query 行的统计量。

初始化 `max=0, denominator=1, numerator=0`，保留固定 fallback。
每个 tile 更新 max，并同时 rescale 原 denominator 与 numerator。
概率 `exp2(W-new_max)` 在第二次 MMA 前才转 BF16，denominator 使用未量化的 FP32 概率。
独立 CPU 对照既计算不量化概率的 double 输出，也模拟相同 online 分块和 BF16 权重量化。
测试显式要求第二、第三个 tile 都发生 max 增大；全不匹配输出必须精确为零。

覆盖九种 D/DV 组合，每种 8 个用例：key 有效长度 139/192 × soft/hard/mixed/全不匹配。
包含逐逻辑列 modifier、labels、causal mask，在 roll **之前**应用。
这条 probe 将 query stripe 放在逻辑行 128…143，但首行 top boundary 设为缺失，
因此它验证的是隔离 stripe，而非完整 192×192 因果注意力。
长度 139 是 key-tail 测试，不是 query-tail load 测试。

实测：72/72 通过。对未量化 double 输出最大绝对误差 `1.1398217e-3`；
对模拟 BF16 权重量化的 double online 输出最大绝对误差 `6.259903e-7`。
二者差异表明本组输入的主要输出误差来自概率 BF16 转换。

| D | DV | 寄存器/thread | dynamic shared bytes |
|---|---|---|---|
| 32 | 32 | 122 | 9224 |
| 32 | 64 | 118 | 13320 |
| 32 | 128 | 166 | 21512 |
| 64 | 32 | 111 | 14344 |
| 64 | 64 | 116 | 18440 |
| 64 | 128 | 166 | 26632 |
| 128 | 32 | 156 | 24584 |
| 128 | 64 | 159 | 28680 |
| 128 | 128 | 157 | 36872 |

所有 fused 实例：ptxas spill stores/loads 都为 0，stack frame 为 8 bytes；
`cudaFuncGetAttributes` 报告 local=8 bytes、static shared=0。不要表述成零 local memory。
这是含 score MMA、scan、online statistics、PV accumulator 的完整单 stripe 融合资源，
不是单独 scan 资源；但不含多 warp 协作、double buffering 或 embedding。
尚未测量吞吐/occupancy，不能据此判断最终 kernel 性能。

TMA 使用可复用 mbarrier/parity，初始化后有 async proxy fence；加载完成和覆盖 shared 前均同步。
K/V 实际分配并初始化 192 行，139-tail 由 score mask 处理。
**这不证明现有 5D map 对未 padding 的 N=139 分配可安全直接加载**：
展开后的高维坐标界限并不等于最终逻辑 token 界限。
正式实现须给最后 tile 单独的安全加载路径/descriptor 或明确 padded workspace。

## 2. 独立摘要 → 对角线边界合成 → 重算

`checkpoint.cu` 使用 16-row checkpoints、64-key tiles、192 列 padded 空间。
三个独立 GPU kernels：

1. `summarize`：每个 stripe 一个 CTA/warp，top identity，沿 key tiles 传递 right boundary，
   只运行 GLX `reduce_forward`，保存底行的 FP32 `(a,b)` 摘要。
2. `propagate`：每线程一条 checkpoint 对角线，不计算 score、不重算 scan。
   `X[s,j] = LSE(X[s-1,j-16] + a[s,j], b[s,j])`，缺失 predecessor 为负无穷。
3. `recompute`：每 CTA 四个独立 warp/stripes，从 checkpoint 注入 top boundary，
   重算 score 和行 RNG，运行 inclusive scan。写 W 仅用于逐元素诊断，不是正式输出路径。

关键 HState 编码（**仅本文验证的 16×64 forward**）：
lane=`4*l+g`、packed element=`e` 的状态是底行逻辑列 `8*g+32*e+6-l`。
整体覆盖列 `-1…62`，其中 `-1` 来自左侧 tile 的底行末列；列 63 则在 VState 中。
摘要导出丢弃本 tile 的 `-1`，从 VState lane31/row-block1 补上 63。
重算注入保留这个 `-1` 偏移，从前一个 query checkpoint 的相应全局列取值。
不能将 HState 当作普通、未偏移的 64 元素底行数组直接加载。

有效 hard 不匹配是 `(-inf,-inf)`，会切断递推；padding 是 `(0,-inf)`，保持输入状态。
实验在 scalar roll/duplicate 后根据 rolled 坐标将 padding 改为 identity，未将两者混用。
CPU 独立检查每个局部摘要的两个分量、每个全局 checkpoint 和重算后的每个 W，包含 padding 状态。

覆盖 N=1/17/65/139/192 × soft/hard/mixed/全不匹配，共 20/20 通过；
最多 12 个 checkpoints，长匹配链跨多个 checkpoints，最大 log2 绝对误差 `4.5054833e-6`。
摘要/重算用同一逻辑行 counter，1-warp 与 4-warp CTA 分组下生成值逐行 bit-exact。
导出 RNG 数组仅作为测试断言，kernel 从不读取这些数组生成决策。
当前 hash 是实验占位，**不代表已经定义 PyTorch seed/offset、全局 direction 或 varlen 随机身份契约**。

不含诊断矩阵/RNG 的临时空间：`ceil(N/16) * 192 * (8+4)` bytes，
N=192 为 27648 bytes；推广到一般 N 是 `12*ceil(N/16)*padded_N`，不是线性空间。

## 检查结论及边界

两个程序均完成 memcheck、racecheck、synccheck：0 errors，racecheck 也没有 warnings/hazards。
这些结果支持以 **warp_k_size=64、FP32 log-affine** 开始 fixed-length 前向基线。
两项实验尚未合并成接收真实 embedding/LSE 的完整三阶段 attention。
下一步仍需落实：未 padding 的真实输入尾加载、PyTorch RNG 消费契约、两种 direction 的接口，
再对照 reference；sm90、varlen、反向、近似 LSE 与性能调优都未在本实验覆盖。
