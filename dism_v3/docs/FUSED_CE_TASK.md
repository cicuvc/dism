# 任务书：移植 FLA 分块 Linear CE，支持 softcap 和 document-balanced

## 目标与范围

在 `/home/cicuvc/cs/project/dism-core` 中，参考本机 FLA 的 fused-linear cross entropy，实现仓库自有的分块 CE 算子，支持正确的 softcap 和 document-balanced loss。

本轮交付算子、测试、benchmark 和实验记录。**不切换生产训练默认入口，不重启训练，不修改 checkpoint。** 不开展 DISM、卷积或模型架构优化。

先阅读根目录、`dism_v3`、`dism_v3/docs` 的 `AGENTS.md`。保留现有未提交修改。本文路径除特别说明外，均相对于工作树根目录。

## 参考实现

FLA 源码：

```text
/home/cicuvc/miniconda3/envs/blkw/lib/python3.12/site-packages/fla/modules/fused_linear_cross_entropy.py
```

仓库相关文件：

```text
dism_v3/python/flash_dism/kernels/fused_cross_entropy.py
dism_v3/python/flash_dism/kernels/test_fused_cross_entropy.py
dism_v3/tools/bench_cross_entropy.py
dism_v3/tools/bench_fla_cross_entropy.py
dism_v3/docs/TRAINING_BENCHMARK.md
```

移植时保留适用的版权与许可证声明，记录来源。不要修改 site-packages，也不要让新实现依赖 FLA 私有函数。

## 实现路线

采用 FLA 的基本结构：

1. 沿 token 维分块，用库 GEMM 计算 logits。
2. 计算 CE，并将该块 logits 缓冲区复用为 dlogits。
3. 用两个 GEMM 计算 dHidden、dWeight。
4. FP32 累加各块 dWeight，最终按输入 dtype 返回梯度。

允许在 autograd forward 中预计算梯度，但完整 forward＋backward 才是性能比较单位。

要求：

- 不物化完整 `[tokens, vocab]` logits。
- 不在 backward 再做一遍 logits GEMM。
- chunk 大小可配置，优先测试 1024、2048、4096 行。
- 保留旧实现作为对照，新实现用独立入口。
- 兼容现有 hidden、weight、labels 形状及 BF16 训练路径。
- 不引入逐块 `.item()`、D2H 或 host-device 同步；计数和归约尽量留在 GPU。
- arbitrary upstream gradient 必须正确；不要照搬 FLA 原地缩放 saved gradients 的行为，导致重复 backward 累乘。
- 明确记录 logits、dlogits、GEMM 输出和梯度累加的 dtype。注意“BF16 chunk dWeight 输出再加到 FP32 buffer”不等于直接获得 FP32 GEMM 输出。

## 数学要求

### Softcap

令原始 logits 为 `z = XWᵀ`：

```text
u = c * tanh(z / c)
loss_i = logsumexp(u_i) - u_i[target_i]
```

反向必须包含：

```text
dL/dz = dL/du * (1 - tanh(z/c)^2)
```

要求：

- `softcap=None` 禁用，生产对照值为 `30`。
- LSE、目标 logit 和 softmax 梯度必须使用同一套 capped logits。
- 不能只给 loss 加 softcap，却仍使用普通 CE 梯度。
- 复用 logits 缓冲区时，不得提前覆盖计算导数所需的信息。

### Document-balanced

`cu_seqlens` 描述展平 token 序列中的 document 边界。设 document `d` 中有效 label 数为 `n_d`，有效 document 数为 `D`：

```text
L = sum_{d: n_d > 0} [sum_{i in d, valid} loss_i / n_d] / D
```

因此有效 token 的梯度权重为：

```text
1 / (D * n_d)
```

要求：

- `ignore_index=-1` 的 token 不计入 loss，也不计入 `n_d`。
- 全 ignored 的 document 不计入 `D`。
- 全部 ignored 时 loss 和梯度均为零，不能产生 NaN。
- document 可以跨 GEMM chunk，不能按 chunk 独立归一化。
- 不得用 document 总长度代替有效 label 数。
- 无 `cu_seqlens` 时保留有效 token mean，便于兼容和对照。
- **loss 的 document 边界与 hard mask 无关。hard sampling 仍以 packed training row 为单位。**

为后续训练集成提供清晰的归约接口：至少能够取得“document mean 的和”及“有效 document 数”，或提供等价的外部归一化机制。

不要把“各 microbatch 的 document mean 再平均”宣称为整个 optimizer step 的 document mean。后续全局归约应为：

```text
L_step = 所有 rank、所有 microbatch 的 document loss 总和
         / 对应的有效 document 总数
```

这里的 document loss 指该 document 内有效 token 的平均 loss，而不是 token loss 的和。

本轮无需修改训练循环。尤其不要假设仓库使用普通 DDP：当前分布式梯度平均发生在 optimizer 中，后续集成需单独核对。

## 正确性验收

使用小规模 FP32 PyTorch 显式参考，以及现有 varlen CE 参考，比较 loss、dHidden、dWeight。

覆盖：

- 无 softcap、softcap30，以及 logits 饱和情形。
- 长短不等的 documents、部分 ignored、整篇 ignored、全部 ignored。
- document 跨 chunk、尾块不满、不同 chunk 大小。
- 上游梯度 `1`、`1/4` 和非二次幂缩放值。
- 同一图重复 backward，验证保存的梯度没有被修改。
- 不等 document 数的 microbatch 分拆：正确加权后应等价于一次整体计算。
- eager 和实际训练所需的 `torch.compile` 调用方式。

报告最大绝对误差、相对 L2、cosine、norm ratio。不得放宽现有测试容差来隐藏误差；新增 BF16 中间舍入的影响要明确说明。

修改 Triton kernel 后检查边界、同步、CALL、寄存器和 spill；运行适用的 sanitizer，未覆盖项如实记录。遵守仓库禁止 shared FP32 atomicAdd 的要求。

## 性能验收

复用现有脚本的真实 checkpoint fixture 和计时方法：

```text
GPU: RTX 5090 / sm120
tokens: 32768
hidden: 384
vocab: 32000
dtype: BF16
warmup: 5
samples: 5
repeats: 10
CUDA graphs: disabled
```

只读 checkpoint 目录与 step：

```text
/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres
step: 19074
```

fixture 由已有脚本从保存的第一份 prefetched batch 捕获，不启动训练、不继续消费 dataloader、不创建 optimizer。若外部盘未挂载，明确报告；不得擅自下载数据或改用随机输入冒充真实 fixture。

至少比较：

| 路径 | 已测完整前向＋反向 |
|---|---:|
| 旧实现，无 softcap，token mean | 39.02 ms |
| 旧实现，softcap30，document mean | 40.16 ms |
| 原版 FLA，默认 8 块，无 softcap，token mean | 17.05 ms |
| 原版 FLA，32 块，无 softcap，token mean | 16.84 ms |

原版 FLA 不支持 softcap/document-balanced，这些结果不是完整目标函数的等价替代。此前数据和 profiler 归因见 `dism_v3/build/ce72m_fla_retry_20261014.json` 及实验记录。

新实现分别测无 softcap/token mean 和 softcap30/document mean，扫描上述 chunk 大小。报告：

- 完整 autograd 时间，包含接口内的同步。
- CUDA event 和 host wall 时间。
- 上游梯度 `1`、`1/4`。
- 额外峰值显存。
- 独立 profiler 的 GEMM、CE、归约和梯度累加耗时。

目标是在支持完整数学语义后明显优于旧实现，并尽量接近 FLA 的约 17 ms；这是性能目标，不可通过省略计算或更改目标函数达成。不得用单独的 forward/backward 列掩盖预计算梯度的工作转移，也不得把 profiler 的 kernel 时间和当作端到端延迟。

## 交付与约束

- 独立的新算子及测试。
- 可复现的 benchmark。
- 文档中的结果、数值差异、剩余限制。
- 简短的后续训练接入说明，指出 accumulation 和跨 rank 归约问题。

使用 conda `blkw`，Python 为 `/home/cicuvc/miniconda3/envs/blkw/bin/python`；只使用本工作树的 `flash_dism` 和 `cu_flash_dism`。从工作树根目录运行时设置：

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1
```

结果 JSON、日志、profile、编译产物留在 ignored `dism_v3/build/`，不得加入 Git。不要修改旧 kernel 默认参数、模型配置、优化器 schema、RNG 或 dataloader 状态。不要删除或覆盖之前的 benchmark 结果。未获额外授权，不重启训练或修改 checkpoint。
