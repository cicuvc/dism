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

当前分块版支持无 SQ/SK 的 `hard` / `soft` / `jensen`。`soft_readout="jensen"`
控制 mixed hard flags 中 soft heads 的读取方式，默认 arithmetic 保持旧配置身份。
不支持二阶梯度、cache/decode、planner、CUDA graph 或 Dynamo compile。

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

数值范围：hard 始终逐候选缩放。按用户随后要求，算术 soft 和 Jensen soft 的
慢速 log-domain 保护均已**无条件移除**，不再检查 `log pK < -40`，始终走 GEMM。
不截断 logits、不添加分母 epsilon。有限输入仍可能产生下溢、上溢或 NaN，尤其
是长匹配链与极尖锐 K 分布叠加时。FP64 reference 保留稳定计算，供诊断使用。

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

## MQAR 短测（2026-10-25，固定 hard probability=0.5）

用户暂停长预训练后，使用本机 `~/cs/project/mad-lab` 的
`generate_in_context_recall_instance`、`LanguageModel` 与 `SwiGLU` 做短测。
工具为 [mqar_quick.py](tools/mqar_quick.py)，不修改 MAD 仓库。

- MAD 的 LayerNorm prenorm backbone，width=128，`[v4-only, SwiGLU] × 2`；
  每层 4 heads、D=Dv=32、符号词表 256，无 GDN、SQ/SK，direction 固定 true。
  这是独立小模型从头训练，不是对 123M 语料预训练权重作下游评估。
- **训练 hard_prob 固定 0.5，不调度**，按样本/head 采样；分别全 soft/hard eval。
- 训练 12800 条、测试 1280 条，batch=128、seed=12345、AdamW lr=5e-4、
  weight decay=0、cosine 到 1e-6。训练预测全部 next tokens；测试只计重复 key
  的 value 查询。输入已由原生成器 shift，实际长度分别为 127、255。
- 检查 train/test 完整序列无重叠；跨样本扰动与未来 token 扰动隔离检查通过。

| 词表 / 生成长度 | 完成 epoch | 最后 soft acc | 最后 hard acc | 最后 soft CE | 最后 hard CE |
|---|---:|---:|---:|---:|---:|
| 16 / 128 | 10 | 100% | 99.9721% | 0.000806 | 0.002007 |
| 128 / 256 | 10 | 42.6376% | 41.9854% | 2.141275 | 2.194077 |

基础档 cosine horizon=10；较难档原定 horizon=20，因为数值 fallback 导致速度
下降，在完成第 10 个 epoch 后结束短测，**尚未收敛**。单 seed、短预算，不能与
历史 200-epoch/best-epoch 的 v3 表格直接比较，也未运行 2K 极限档或完整 MAD suite。
复现停止预算时可用以下命令，保持较难档的 cosine horizon=20：

```bash
python dism_v4/tools/mqar_quick.py --epochs 10 --hard-prob .5 \
  --output dism_v4/build/mqar/new_v16_n128
python dism_v4/tools/mqar_quick.py --vocab 128 --length 256 --epochs 20 \
  --stop-after-epochs 10 --hard-prob .5 --output dism_v4/build/mqar/new_v128_n256
```

结果、模型 checkpoint、执行脚本快照和数据 hash 位于忽略目录
`dism_v4/build/mqar/{v16_n128,v128_n256}_h05_seed12345/`。

### 本次发现的性能问题

较难档每 epoch 从约 3.5s 升到第 9/10 epoch 的 73/83s。第 10 epoch checkpoint
在一个测试 batch 上，两层最小 log pK 分别约 -45.04/-46.17；约 23.0%/46.1%
的 row/head 含有低于 -40 的值。当前实现只要任一 soft group 元素低于该阈值，
就将该层这次调用的全部 soft heads、全部 query blocks 切到逐行 log-domain
广播归约。这个策略很保守，而且 fallback 放弃了高效 GEMM。

隔离数值诊断：取同一 checkpoint 第一层一个触发条件的 head（min log pK=-42.49，
tau=0.6045），仅在诊断进程中强制普通 FP32 GEMM 公式，对比 FP64 oracle。
输出/Q/K/V 的相对 L2 误差均不超过 6.3e-7，tau 为 6.6e-6，全部有限。
说明 **min log pK<-40 并不充分意味着普通路径数值失效**，不是取消保护的通用证明。
上述诊断时生产阈值尚未修改；随后用户明确要求无条件禁用慢速保护，见下一节。

常规精确 soft 读取仍有 Z=AK 与 B=CK^T 的词表维收缩，当前 dense 算法是
O(N²C)，分块只降低存储，不消除 FLOPs。这是需要单独评估的算法成本；不能由
当前 Python fallback 的约 24 倍减速推断 v4 方法本身无法高效实现。

## 保留候选分母的 Jensen 读取与禁用慢速保护

这次仅改变 soft 读取系数：

```
Z[i,c] = sum_j A[i,j] * pK[j,c]
R[i,c] = log pQ[i,c] - log(1 + Z[i,c])
M[i,j] = A[i,j] * exp(sum_c R[i,c] * pK[j,c])
output = M @ values
```

hard 历史、hard 读取、单位 fallback 分母、输出 norm/gate 均不变，未采用
去分母再 RMSNorm 的变体。这里的 **单位 fallback 是数学分母中的 1**，与被删除的
**慢速数值保护路径**不同。Jensen 对完整归一化系数做近似，因此仍有两次词表维
收缩；不会获得无分母 Jensen 的 O(NVD)+O(N²D) 计算结构。

启用接口：`dism_v4_blocked(..., soft_readout="jensen", hard=flags)`；
全 soft 时也可 `key_mode="jensen"`。nanochat 配置使用同名 `soft_readout` 字段，
新预设为 [参数对齐的 Jensen 配置](../nanochat/configs/dism_v4_123m_ggggggdddddd_slimpajama_3b_jensen.json)。
字段进入 ModelSpec 指纹，不会把原始 arithmetic checkpoint 默认为 Jensen。
MQAR 脚本增加 `--soft-readout jensen`，`hard_prob` 仍可固定 0.5。

FP32 前反向已对现有 FP64 `key_mode="jensen"` oracle 验证，包括跨 block、
packed/reset、尖锐 K、one-hot hard 端点、非负系数下界、mixed flags、MXFP8 optimizer
恢复。84 项测试通过，**其中两项明确记录禁用保护后极端 soft 输入会产生非有限值**：
tau=1000、K logits=±1000 的原用例仍有有限 FP64 输出，但算术/Jensen GEMM 输出
均非有限。没有放宽普通数值测试容差；没有 CUDA kernel 修改或新 sanitizer 声明。

### 配对效率测量

工具：[benchmark_readouts.py](tools/benchmark_readouts.py)。RTX 5090，FP32 readout、
BF16 模型 autocast，TF32 关闭。同权重、同输入、同 hard flags；只测 forward+backward，
不更新 optimizer，不启动长训练。旧保护实现从 Git `50bec9f` 加载到诊断进程，
不恢复生产保护。预热 2 次，取 3 组计时中位数；每组 5 次，MQAR 每组 3 次。

| 测试 | hard prob | 旧算术＋保护 | 算术、无保护 | Jensen、无保护 |
|---|---:|---:|---:|---:|
| 2K 单算子，B=1/H=12/C=512/Dv=64 | 0 | 30.36 ms | 30.04 ms | 32.78 ms |
| 同上 | 0.5 | 28.16 ms | 28.13 ms | 28.14 ms |
| 123.47M 完整模型，B=4/N=2048 | 0 | 743.68 ms | 744.63 ms | 740.53 ms |
| 同上 | 0.5 | 552.68 ms | 552.23 ms | 551.70 ms |
| MQAR epoch-10 checkpoint，B=128/N=255 | 0 | 2032.80 ms | 39.54 ms | 38.25 ms |
| 同上 | 0.5 | 776.25 ms | 32.92 ms | 34.33 ms |

当前主要收益来自禁用保护：MQAR 的 50% hard 情形算术路径加速约 23.6 倍。
Jensen 相对无保护算术没有稳定的吞吐收益；保留分母时计算量同阶。
以上是前反向速度，不包含 optimizer/数据加载/日志，不能直接当作完整预训练吞吐。
结果位于 `dism_v4/build/jensen/{operator,model,mqar}.json`；显存记录区分保留的
诊断参考梯度 buffer 与算子/模型本身。

MQAR 同 checkpoint 的 50% hard 冻结权重回放中，移除保护后的 CE 差约 -1.3e-5、
整体参数梯度 cosine≈0.999995。换成 Jensen 后 CE 差约 +0.02164、梯度 cosine≈0.315。
这是训练 surrogate 的实际改变，不是等价优化，质量需要重新训练验证。完整模型
BF16/atomic 回放仍存在非 bitwise 梯度差异；本轮没有宣称恢复轨迹精确一致。

另测 N=2048 随机输入的 BF16 readout 对 FP32：Jensen 输出相对 L2 为
0.00259，Q/K/V/tau 梯度为 0.00336/0.01029/0.00256/0.00513，均有限。
这只是该输入上的精度检查，不恢复上述极端动态范围保证。

## 无分母 RMSNorm 变体：效率诊断

首轮使用独立诊断实现和配对测量工具
[benchmark_rmsnorm.py](tools/benchmark_rmsnorm.py)，当时未接入生产配置。
实现现已迁入 [rmsnorm.py](python/flash_dism_v4/rmsnorm.py)，
[rmsnorm_probe.py](tools/rmsnorm_probe.py) 保留独立 FP64 oracle；后续 opt-in 接入见下一节。
实现第 13 节（去掉 SQ/SK 后）的变体：hard/soft 均移除候选分母，对混合后的
DISM 输出逐 head 做 RMSNorm，再与 GDN 合并；外部 norm/gate 不变。
新增 RMSNorm 的 gain 固定为 1、epsilon=1e-6，不增加参数。soft 使用
`Ebar=softmax(K logits) @ E_Q`，`log m=q @ Ebar^T-logsumexp(Q logits)`；
hard 使用所选符号的 E_Q 向量。两者仍使用同一条 argmax 历史。

该变体将两次 N²C 词表收缩替换成 NCD 插值和 N²D pair score；C=512、D=64。
复用当前 hard-history kernel，分块重算反向，不保存 N² 激活。
行缩放后同步调整 epsilon，未利用 RMSNorm 近似尺度不变性丢弃有限 epsilon。
4 组独立 FP64 dense oracle 前向/反向检查通过，覆盖 mixed flags、packed 边界、
跨块同符号长链，以及 epsilon=1e-6/0.1。所有计时输入的输出和参数梯度均有限；
这不是任意极端输入的数值保证，也未验证训练质量。

RTX 5090，FP32 readout、TF32 关闭；整模型沿用 BF16 autocast 和 MXFP8 FFN。
同输入、同初始化权重、同 hard flags，预热 2 次，每组 5 步，取 3 组中位数。
只测前向＋反向，不含 optimizer 更新、数据加载和日志。模型为 123,473,088 参数、
ggggggdddddd、N=2048、B=4。基线全部已禁用慢速保护。

| 整模型 hard prob | 算术＋候选分母 | Jensen＋候选分母 | 无分母 Jensen＋RMSNorm | 相对有分母 Jensen 加速 |
|---|---:|---:|---:|---:|
| 0 | 755.3 ms | 743.3 ms | 227.9 ms | 3.26× |
| 0.5 | 551.2 ms | 544.8 ms | 227.6 ms | 2.39× |
| 1 | 288.5 ms | 282.6 ms | 211.0 ms | 1.34× |

50% hard 下整模型耗时下降约 58.2%，峰值 allocated 显存由 10.83 GiB 降至
8.20 GiB（均未分配 optimizer state）。单算子计时包含 Q/K 词表投影和反向：
同为 N=2048/H=12/C=512/D=Dv=64，B=1 的 Jensen 基线 32.16 ms、变体
19.75 ms（1.63×）；B=4 分别为 81.28 ms、28.82 ms（2.82×）。收益随 batch
明显变化，不能把 B=1 算子比例直接外推整模型。此原型也合并了 hard/soft 的
pair readout 调度，结果包含这部分实现差别；没有单独剥离每项贡献。
B=1 全 hard 时本原型为 19.48 ms、旧 hard 为 16.62 ms，尚不能断言所有形状
或专用 hard 推理 kernel 都会加速。

结果位于 `dism_v4/build/rmsnorm/{operator,operator_b4,model}.json`。
这是不同归一化语义的模型候选，不是与原输出等价的优化；最终是否采用仍取决于
训练效果。未启动或恢复长训练。

## 分支各自归一化的融合及 3B 训练时长

用户选择无分母 RMSNorm 变体后，将 hybrid 融合改为
`(RMSNorm(gdn_raw) + RMSNorm(dism_raw)) * SiLU(gate)`，之后做原来的输出投影。
每个 head 分别沿 Dv 归一化，epsilon 均为 1e-6；DISM 的 RMSNorm 已包含在
readout 中，不重复归一化。相加后不再做 norm。两路共享原有 `norm.weight`
可学习尺度，所以参数量仍为 123,473,088；pure GDN 层保持原样。

新增显式配置 `soft_readout="rmsnorm_jensen"`、`hybrid_fusion="separate_rmsnorm"`，
默认仍为旧 arithmetic/sum。新预设
[dism_v4_123m_ggggggdddddd_slimpajama_3b_rmsnorm_h05.json](../nanochat/configs/dism_v4_123m_ggggggdddddd_slimpajama_3b_rmsnorm_h05.json)
使用旧布局 ggggggdddddd、固定 hard_prob=0.5、MXFP8 FFN/optimizer 开启、MXFP8 CE
关闭。该 opt-in 目前只支持 FP32 readout 和旧 SiLU gate；planner/CUDA graph 仍不支持。
不修改旧 checkpoint，不将旧 checkpoint 自动迁移到新融合。

86 项 pytest 通过，包括融合公式的逐项检查（禁止相加后的 norm）、新变体的
2K 训练、MXFP8 optimizer/schema、共享参数别名及 RNG/annealing 状态恢复。
RMSNorm 算子的 4 组独立 FP64 前反向对照另外由 `rmsnorm_probe.py` 执行。
此前保留的两个极端 soft 非有限值反例仍然存在。

完整更新步测量工具：[benchmark_training_step.py](tools/benchmark_training_step.py)。
使用新初始化模型和固定合成输入，单卡 RTX 5090、N=2048、B=8、累积 8 次，
每步 131,072 token；每 1024 token 设置 document 边界。与旧长训练一致采用
`float32_matmul_precision="high"`（允许 TF32）、document-loss 累积归一化、
grad clip=1、MXFP8 AdamW、hidden/update RMS 诊断。没有语料加载、W&B 网络日志、
定期验证或 checkpoint 写盘。运行 2 个预热更新和 5 个计时更新，无后台长训练。

- 完整步中位数：3.3286 s；范围 3.3181–3.4087 s。
- 吞吐：39,378 token/s；峰值 allocated 约 17.68 GiB，包含 optimizer/诊断。
- 原预算 22,888 步、2,999,975,936 token，纯训练约 21.16 小时。
- 加上数据、500 步验证/8K NIAH、2000 步存盘等未实测开销，建议预留 22–24 小时。

这是短合成测试的时间外推，非完整语料训练实测；不据其 loss 推断训练质量。
结果保存在 `dism_v4/build/rmsnorm/training_step.json`。长训练保持停止。
