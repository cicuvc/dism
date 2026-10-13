# 前作：bf16 下 log-domain affine 递推的近似方案性能（`~/cs/projects/glx/experiments`）

这些是在 **H100**（Modal，132 SM，CUDA 12.8.1，`-O2 -arch=sm_90`）上对 16x64 正向对角扫描
（`glx::BinaryElement` + `ApproxTupleOp`）的实测。**训练目标是 RTX 5090（sm_120），结论不能直接照搬。**

## 1. BF16x2 存储 + MUFU.BF16（`h100_bf16_scan_results.md`）

- 布局是 **SoA**：`BinaryElement<BF16x2>` 的 `first`/`second` 各是一个 packed bf16x2，
  即 `first=(a0,a1)`、`second=(b0,b1)`。`bf16_soa_layout_check` 在 H100 上
  **`max_abs_error = 3.10e-02, PASS`**（误差主要来自 `a+c` 的 bf16 舍入）。
  → **这就是 bf16x2 的实测精度代价，写实现时要把它当作"支持区间"来声明。**
- `tanh.approx.bf16x2` / `ex2.approx.ftz.bf16x2` 存在（PTX ISA 7.8，sm_90+），ptxas 自己
  生成 `MUFU.TANH.BF16` / `MUFU.EX2.BF16`；打包的 bf16x2 MUFU 在 SASS 里是
  **2 条标量 MUFU + 1 条 PRMT** 拼回 packed。
- 指令数下降（MMA）：`scan_only 626 → 516`、`full_forward 870 → 781`、
  `special_full 727 → 618`。
- **SHFL 减半**：`full_forward MMA 124 → 62`、`split 136 → 68`、`special_full 66 → 33`。
- **但 H100 上计时更慢**：`MMA F32-tanh full_forward 1285.9` vs
  **`MMA BF16-tanh full_forward 2194.2` 周期（≈1.7× 慢）**，`scan_only 883 → 1426`。
  → **指令更少、shuffle 减半，却更慢**（打包 bf16 MUFU 展开成 2 条标量 MUFU + PRMT）。

## 2. EX2 多项式替换 tanh 版（`h100_ex2_scan_results.md`）

新 op `lse2(x,y) = max + t·P(t)`, `t = 2^(-|x|)`，n=2，**max error 2.227e-3**（与文档一致）。

| op | 实现 | full_forward | scan_only | special_full |
|---|---|---:|---:|---:|
| TANH | MMA | 1285.8 | 883.1 | 1291.1 |
| **EX2 n=2** | **MMA** | **1323.1** (+2.9%) | **865.1** (−2.0%) | **1610.8** (**+24.8%**) |
| TANH | split | 1324.4 | 871.0 | 1503.1 |
| **EX2 n=2** | **split** | **1368.1** (+3.3%) | **924.0** (+6.1%) | **1900.1** (**+26.4%**) |

**结论：EX2 n=2 精度达标但无速度优势**；单独扫描基本持平，带 roll/unroll 或 broadcast
特殊路径时明显更慢（串行 FFMA 链更长）。

## 3. 真正的瓶颈是 wgmma 争用（`h100_wgmma_contention_results.md`）

| op | 实现 | 模式 | 4-warp 无干扰 | 密集 wgmma | 倍数 |
|---|---|---:|---:|---:|---:|
| tanh | MMA | full_forward | 1285.8 | 2441.4 | **1.90×** |
| tanh | MMA | scan_only | 883.1 | 1242.5 | 1.41× |
| tanh | split | full_forward | 1324.4 | 2647.2 | **2.00×** |
| ex2 | split | full_forward | 1368.1 | 3423.3 | **2.50×** |

**扫描在真实训练（张量核忙碌）下退化 1.4–2.5×**，所以**微调 op 的指令配比不是收益所在**；
按墙钟算退化更大（ex2 split ≈686 ns → ≈1835 ns，2.67×）。

## 4. 架构差异（重要）

`MMABuffer` 的 roll/unroll：**H100 上只占 ~403 周期（1286−883），RTX 5090（sm_120）上占
~1724 周期（2527−803）**。H100 的 shuffle 吞吐/延迟便宜得多，所以原版 roll 在 H100 上不是
瓶颈，**在 5090 上是**。

## 对本次 bf16x2 任务的直接含义

1. **bf16x2x2 应被当作"格式/容量选项"，而不是速度优化**：H100 上实测更慢；官方理由是
   打包 bf16 MUFU 的展开。**潜在收益在别处**：寄存器/共享内存压力、shuffle 通信减半
   （在 5090 上 roll/unroll 占比大，减半可能真正有用）、以及半年前训练实验观察到的
   "不掉点甚至略强"。
2. **精度契约要写清**：bf16x2 的实测 `max_abs_error ≈ 3.1e-2`，来自 `a+c` 的舍入；
   `LOG_ZERO = -1e6f` 这类哨兵在 bf16 下语义会变，必须重新定义并测试，而不是换类型。
3. **必须在目标 GPU（RTX 5090 / sm_120）上重测**，不要用 H100 的结论下判断——
   尤其因为 5090 的 roll/unroll 成本是 H100 的 4×。
4. **EX2 方案可以不优先**：精度虽好（2.2e-3）但在带 roll/broadcast 的路径上更慢。
5. 若目标是**吞吐**，优先考虑的方向应是**减少 wgmma/MUFU 争用**（调度/占用/重叠），
   而不是把 fp32 换成 bf16。
