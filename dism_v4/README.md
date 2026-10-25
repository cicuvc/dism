# DISM v4 reference 与分块训练原型

设计见 [docs/DESIGN.md](docs/DESIGN.md)。包含独立 PyTorch correctness reference、
CUDA 分块前反向，以及单独注册的 nanochat v4 模型。CPU reference 不依赖 v3
extension；模型复用本 worktree 的 GDN、投影、FFN、CE 和 optimizer。
v3 默认结构和算子不变，不迁移或修改已有训练 checkpoint。

默认算子保留 **hard Q/K 历史、全 Q 候选、候选独立归一化、零值单位 fallback**，
不使用 SQ/SK，不额外做 RMSNorm/gate/output projection。每个位置的候选只影响
当前读取，只有 argmax Q 符号进入下一步匹配历史。

## 使用

在仓库根目录、conda `blkw` 环境下，将本目录的 `python` 加到 `PYTHONPATH`：

```bash
conda activate blkw
export PYTHONPATH="$PWD/dism_v4/python${PYTHONPATH:+:$PYTHONPATH}"
```

使用预先计算的符号 logits：

```python
import torch
from flash_dism_v4 import dism_v4_ref

B, N, H, C, Dv = 1, 8, 2, 5, 4
q_logits = torch.randn(B, N, H, C, dtype=torch.float64, requires_grad=True)
k_logits = torch.randn_like(q_logits, requires_grad=True)
values = torch.randn(B, N, H, Dv, dtype=torch.float64, requires_grad=True)
tau = torch.zeros(H, dtype=torch.float64, requires_grad=True)

output = dism_v4_ref(q_logits, k_logits, values, tau, key_mode="soft")
output.square().mean().backward()
assert output.shape == (B, N, H, Dv)
```

也可直接从 per-head codebook 计算 logits：

```python
from flash_dism_v4 import dism_v4_from_embeddings

D = 3
q = torch.randn(B, N, H, D, dtype=torch.float64, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
q_weight = torch.randn(H, C, D, dtype=torch.float64, requires_grad=True)
k_weight = torch.randn_like(q_weight, requires_grad=True)

output = dism_v4_from_embeddings(
    q, k, q_weight, k_weight, values, tau, key_mode="soft",
)
```

不隐式除以 sqrt(D)，不添加激活函数或采样。所有浮点输入中只要有 float64，
计算和输出就使用 float64，否则将 FP16/BF16/FP32 输入提升到 float32 计算。
所有 logits 和连续输入必须有限；需要 one-hot 极限测试时可使用足够尖锐的
有限 logits。argmax tie 选择最小符号索引。

## 当前读取模式与历史

| `key_mode` | 当前读取 | 持续历史 |
|---|---|---|
| `"hard"`（默认） | K 取 argmax；Q 枚举全部符号并按 softmax 加权 | Q/K 都取 argmax |
| `"soft"` | K 使用完整 softmax，精确计算逐候选归一化后的 Q 混合 | Q/K 都取 argmax |
| `"jensen"` | 对完整归一化读取系数使用 Jensen 下界 | Q/K 都取 argmax |

`jensen` 对应设计第 6.1 节，包含动态分母修正：

```
log B[i,j] = sum_c pK[j,c] * (log pQ[i,c] - log(1 + Z[i,c]))
```

它不是“只替换匹配分数而省掉候选分母”，也不会将 soft 匹配状态写回历史。
这是研究对照，不是选择它就自动获得低维高性能实现。

所有模式的梯度都不穿过主线 argmax。`hard` 模式下 K logits/codebook 没有
可微用途，autograd 可能返回 `None` 梯度；Q 的当前概率读取仍然有梯度。

可选的 `sq=...`、`sk=...` 必须一起提供，形状均为 `[B,N,H,R]`，直接使用已经
激活的 signed 特征点积，只作用于分子，用于与无 SQ/SK 基线比较。

## 边界与数值

- 普通 `[B,N,...]` 输入的各 batch row 相互隔离；没有 256 对齐限制。
- `cu_seqlens` 可为 CPU 或设备上的 int32/int64 向量，仅支持 `B=1` 的 packed
  输入，严格递增，从 0 到 N；不同 segment 的 Q/K 历史完全隔离。
- `query_reset` 是同设备的 bool `[B,N]`。它在当前行前清空 Q 匹配历史，仍允许
  读取之前的 K；不能用它代替文档隔离。没有隐式文档边界推断。
- log 域递推使用 `logaddexp`；每个候选将 log-weight=0 的 fallback 一并纳入
  `logsumexp`，因此无匹配候选的分母为 1，反向不会遇到全 `-inf` 的归一化。
- 独立候选的缩放保留到归一化后，避免全行共同缩放导致弱符号桶下溢后丢失。
- 不使用或修改 RNG、hard-probability 调度、模型 buffer 或 optimizer。

上述 reference 是用于小输入的 **dense eager oracle**。逐行临时量约 `[B,H,i+1,C]`，普通
autograd 会保留候选中间量，训练显存可达 `O(B H N² C)`。它不是设计第 14 节的
分块重算实现，不应据此测训练吞吐或宣称显存线性。Python 控制流和验证有设备
同步，当前也不支持 CUDA graph capture。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD/dism_v4/python" \
  python -m pytest -q dism_v4/tests/test_reference.py
```

本阶段 CPU 测试：37 项通过。覆盖独立符号串后缀枚举及梯度对比、embedding
adapter 的 FP64 gradcheck（包括 codebook/SQ/SK/tau）、负/零/正 tau、因果前缀
与梯度、packed segment 隔离、Q reset、one-hot 极限、Jensen 系数界、输入
契约、低精度输入提升以及 RNG 不变性。

还验证了候选匹配量约为 exp(3000) 和 exp(4000) 时，两者归一化后的贡献都能
保留，反向有限。该用例不是所有极端浮点范围的保证。分块实现的 GPU 验证见下文；
尚未进行完整预训练或收敛质量验证。

## 分块 CUDA 算子

```python
from flash_dism_v4 import dism_v4_blocked

output = dism_v4_blocked(
    q_logits, k_logits, values, tau,
    key_mode="soft", block_size=64, precision="fp32",
)
output.square().mean().backward()
```

输入为 CUDA FP32/BF16/FP16，输出 FP32。`block_size` 可选 32/64/128；
`precision="bf16"` 显式允许 GEMM 输入量化成 BF16、FP32 累加。默认 `fp32`
使用原生 FP32 `bmm`，遵循 PyTorch 全局 TF32 配置，不修改进程精度设置。
BF16 MMA 是有误差的加速选项，不是与 FP32 严格等价的路径。
`dism_v4_blocked_from_embeddings` 提供 Q/K codebook 适配，词表投影使用 FP32。

除 reference 已说明的边界参数外，可提供：

- `hard: bool[B,H]`：覆盖 `key_mode`，按 **训练 row/head** 分组计算 hard/soft。
- `segment_ids: int[B,N]`：同 row 内非递减文档编号，与 `cu_seqlens` 二选一。
  文档隔离与 row-level hard 选择是两件独立的事。

当前分块版仅支持无 SQ/SK 的 `hard` / `soft`，不支持 Jensen、二阶梯度、
cache/decode、planner、CUDA graph 或 Dynamo compile。

实现边界：

1. Triton 沿对角方向做前缀最大值扫描，得到 hard 后缀长度。只传一行边界。
2. 长度 `l` 对应 `A = sum(exp(t*tau), t=1..l+1)`，直接计算 `log A` 和
   `d log A / d tau`；tau 接近 0 时用级数避免消减误差。
3. 词表归一化与 value 读取在外部完成。soft 使用 GEMM；hard 使用逐符号
   max/sum scatter，保留各候选独立缩放，不构造完整候选输出 `[N,C,Dv]`。
4. 自定义反向重新生成同样的 blocks，显式计算 Q/K 概率、V、tau 的梯度。
   不保存 `[N,N]` 激活，也不保存每个 block 的边界 checkpoint。

常驻/保存张量为 `O(B H N (C+Dv))`，成对 scratch 为 `O(B H block_size N)`。
soft 计算仍是 `O(B H N² C)`，quadratic hard 为 `O(B H N² Dv)` 加 scatter；
这解决的是 2K 下的可用性与显存，不是 subquadratic 算法。

数值保护：hard 始终逐候选缩放。soft 在最小 `log pK < -40` 时切换到逐行
log-domain 候选归一化，防止某个符号桶因共同缩放而丢失。该 fallback 很慢，
但不截断 logits、不隐式修改概率、不物化 `N²C` 张量。若训练后出现尖锐的
K 分布，需要重新检查这一路的触发率与吞吐；目前性能数字不能外推至该状态。
输入应有限；只验证了文档列出的数值范围，不保证所有 FP32 极端值。

## Nanochat 模型与运行

配置：[dism_v4_2k_prototype.json](../nanochat/configs/dism_v4_2k_prototype.json)。
架构名为 `dism_v4_gdn_hybrid@1`，与 v3 checkpoint 身份不同；这是新模型，
不能直接恢复 v3 checkpoint。保留 postnorm、value residual、`gggddddddggg`、
固定 true direction、packed-row hard 采样和现有 hard-probability/tau 调度。
仅 hybrid 层替换 DISM 核心并删除 SQ/SK 参数，其余 GDN、gate、FFN/CE 不变。
`readout_dim` 在继承配置中仅为 scaffold 兼容字段，不产生 v4 SQ/SK 参数。
默认配置约 119.934M 参数，MXFP8 FFN/optimizer 开启，MXFP8 CE 关闭。

```bash
conda activate blkw
export PYTHONPATH="$PWD/dism_v4/python:$PWD/dism_v3/python:$PWD/nanochat"

# 合成数据短测：实际 forward/backward、clip、optimizer step，以及 soft/hard eval。
python dism_v4/tools/benchmark_model.py --batch 1 --steps 5 \
  --output dism_v4/build/blocked/model_2k.json

# 独立算子计时与该形状下 BF16/FP32 前反向误差。
python dism_v4/tools/benchmark_operator.py --check-bf16 \
  --output dism_v4/build/blocked/operator_2k.json
```

语料训练复用现有 `scripts.base_train`，选这个 `--model-spec`，先从
`--device-batch-size 1` 开始。保留已有 tokenizer、data-dir、seq-align=256、
学习率/总 batch 等显式训练参数，使用新的 model tag 和 run 目录。
不要加 `--cuda-graph` / `--cuda-graph-planner`；注册能力使训练循环保持 eager。
没有训练数据或 tokenizer 的自动下载步骤。
`estimate_flops` 是包含重算的 soft GEMM 估计，不能当作混合 hard 路径的精确 MFU。

评估沿用 packed CE / unreduced token loss：`model.eval_hard=True` 为 hard，
`False` 为 soft；不会推进 RNG counter。可以使用现有 hard/soft loss evaluator。
本阶段不支持 generation/cache，因此不提供生成式下游任务路径。

训练 step、hard_probability、rng_counter、no-decay 分组、Q/K 共享参数和
optimizer schema 都复用 nanochat；原有 dataloader consumed-position 机制不变。
resume 仍必须同时恢复模型、optimizer、rank RNG 与 dataloader 状态，不能只
加载模型权重就视作精确恢复。

## GPU 验证与性能记录（2026-10-25）

RTX 5090，PyTorch 2.13.0+cu130，Triton 3.7.1。测量包括主机提交与同步，
无 CUDA graph。算子测试用 FP64 oracle；性能测试用合成数据，不能代表质量。
验证通过：reference 37 项、分块算子 26 项、模型 3 项，以及 nanochat 的
DISM 集成 9 项与 checkpoint/dataloader resume 7 项。模型测试包含普通 AdamW
和 MXFP8 optimizer 的 2K 训练/恢复。单机单卡验证，不包含 DDP 验证。

2K、B=1、H=12、C=512、Dv=64、block=64，单层算子 FP32 前向+反向约
30–34 ms（soft）、15.9–16.6 ms（hard）。保存激活测试确认没有成对矩阵。
BF16 在同一 2K 随机输入上的 output/Q/K/V/tau 相对 L2 误差为 0.21%–0.37%，
cosine ≥ 0.999993；这是有限样本证据，不是全程误差保证。

完整 119.934M 模型、B=1、2K、预热 2 次后测 3 次：

| readout | 纯 soft 训练 | 纯 hard 训练 | 50% 混合训练 | soft eval | hard eval |
|---|---:|---:|---:|---:|---:|
| FP32 | 7.90k tok/s | 14.07k tok/s | 9.27k tok/s | 30.52k tok/s | 31.37k tok/s |
| BF16 MMA | 9.69k tok/s | 14.14k tok/s | 8.64k tok/s | 32.38k tok/s | 33.87k tok/s |

FP32 训练峰值 allocated 约 3.92 GB，eval 约 2.07 GB（仍保留 optimizer 状态）。
短测的 mixed 结果有主机调度/分组开销，BF16 不保证在每个模式下更快，默认仍为 FP32。
各模式依次执行 optimizer step，表中的测试 loss 不能用来比较模式的质量。

同一配置 B=4、预热 2 次后测 5 次，FP32 训练吞吐为 soft **10.42k**、hard
**28.99k**、混合 **15.04k tok/s**，eval 为 soft **52.20k** / hard **62.98k tok/s**。
训练峰值 allocated 为 12.18 / 11.61 / 11.46 GB。batch 增大可以摊薄提交开销，
但 soft 词表 GEMM 的计算量仍然较大。

```bash
OMP_NUM_THREADS=2 python -m pytest -q dism_v4/tests
python -m pytest -q nanochat/tests/test_checkpoint_resume.py nanochat/tests/test_dism_integration.py

compute-sanitizer --tool memcheck --error-exitcode 9 \
  python dism_v4/tools/check_history.py
# 可将 memcheck 换成 racecheck / synccheck。
```

新增 history kernel 的 memcheck/racecheck/synccheck 检查覆盖 N=1/31/73/129/2048、
block=32/64/128、边界与 reset；均通过。三个 specialization 的寄存器数
40/97/181，spill=0，SASS 未发现 CALL、LDL/STL 或 atomic 指令；shared memory
用于跨 warp 扫描/归约，非 FP32 atomicAdd。
sanitizer 范围仅为新 history kernel，不是完整 FLA/MXFP8/模型的 sanitizer 认证。
早期用 PyTorch GPU 随机数/验证算子的检查还触发过
`CUDA API Warning: Selective device code recompilation in progress`，工具将其
计为 1 error；改为 CPU 准备/检查数据的独立 history harness 后零错误。

恢复验证分别检查前反向数值一致、buffer/schema/alias 完整恢复，以及使用同一份
梯度时 optimizer 更新精确一致。一次完整模型 replay 的 Adam 更新曾出现单元素
约 1.56e-4 差异，虽然 loss/梯度通过比较；单独重测没有复现。BF16 与 atomic
归约不保证 bitwise replay，不能把 serialization 通过解读为训练轨迹逐 bit 相同。
