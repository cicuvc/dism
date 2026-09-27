# 3层Copy task训练吞吐量（2026-09-08）

RTX5090 / sm120，conda blkw（torch2.13.0+cu130、CUDA13.1编译工具链）。
沿用copy_task模型与训练步，设置3层、4 heads、D=DV=64、d_model256、QK词表512、
序列长度1024；batch沿用64，token词表128，总参数4,068,844。
所有后端都成功运行batch64，没有OOM或降低batch；测量步骤loss/梯度norm均有限。

## 实测

| Dism实现 | 完整训练步中位数 | 输入token/s | 受监督token/s | 峰值allocated显存 |
|---|---:|---:|---:|---:|
| CUDA core + CUDA embedding前后向（配对WS） | 71.76 ms | 913,265 | 456,633 | 3.71 GiB |
| 原copy_task：eager Torch递推 + Triton embedding | 8,874.25 ms | 7,385 | 3,692 | 16.33 GiB |
| dism_ref：纯eager Torch Dism | 8,933.30 ms | 7,336 | 3,668 | 24.11 GiB |

CUDA相对原copy_task约**123.7倍**，相对纯Torch reference约**124.5倍**。
五次样本范围：CUDA71.687…71.878ms，原copy_task8.663…8.927s，reference8.766…9.096s。
输入token计数每步64×1024=65536；只计复制段loss位置则为32768，吞吐减半但倍率不变。

## 计时与基线定义

- 每个后端独立进程，串行运行；3个完整训练步预热，随后5步取中位数。
  使用CUDA同步边界的wall-clock时间，不用仅GPU kernel之和替代实际训练吞吐。
- 包含随机数据生成、zero_grad、完整3层前向、CE、反向、梯度裁剪、AdamW、LR scheduler。
  不含模块导入、模型构建、编译预热、评估和计时边界外的日志输出。
- 共同部分保留原short convolution、门控归一化、FFN和优化器；原SwiGLU的torch.compile保留，
  Torch Dism递推没有compile/fusion，不是与经过优化的compiled scan比较。
- FP32 master参数、BF16 autocast、sm_scale=1；hard_prob固定0.5表示训练中间阶段，
  direction每层每次调用随机；不要求不同RNG实现逐bit匹配。未把hard_prob随时间调度纳入本次计时。
- 原copy_task baseline直接调用其voc_dism：BF16 score GEMM、FP32逐行递推，embedding已有Triton。
  因而额外报告纯Torch reference，避免将其误称为全部由Torch实现。
- reference调用dism_ref.voc_dism_ref：禁用Dism内部autocast，Torch interpolation/score/scan使用FP32，
  PV遵循reference的BF16权重/value GEMM；输出转回BF16进入共同模型后半段。
  它还保留reference先计算两种direction score再选择的实现，未为性能修改oracle。
- 三种路径不是逐bit/同精度实现；CUDA的既有近似和BF16量化差异保留。
  两种Torch路径都物化N×N状态并依赖逐行eager autograd，倍率仅针对这些朴素基线。
  没有测同规模长期训练收敛，不能用本次吞吐测量替代训练质量验收。
- 峰值显存是预热后测量区间torch.cuda.max_memory_allocated，包含模型、梯度、optimizer状态
  和激活/临时缓冲，不是仅kernel scratch；reserved峰值另记录在JSON。

## 复现

原copy_task.py及dism_ref.py不修改；独立入口替换attention调用，并把原构造器的
QK词表参数从256改为512，保留其余模型构造顺序。

```bash
python -m dism_v2.benchmark_copy_training --backend cuda --batch 64
python -m dism_v2.benchmark_copy_training --backend torch --batch 64
python -m dism_v2.benchmark_copy_training --backend torch_ref --batch 64
```

可用`--warmup`、`--steps`增加样本，`--hard-prob`切换训练阶段；不同后端勿并发计时。
完整参数、预热loss、五次训练样本和结果见
[`benchmarks/copy_training_throughput_sm120a.json`](benchmarks/copy_training_throughput_sm120a.json)。
原始日志`/tmp/dism-throughput-{cuda,torch,torch-ref}-b64.jsonl`及对应stderr。

## 六个Dism主kernel逐launch时间

同一配置、CUDA配对WS路径，3步预热后用CUPTI/torch.profiler采样20个完整训练步。
每种kernel每步3次，共60次launch；下表为真实GPU执行时间，排除host发射间隙，
不能直接当作上一节wall-clock训练步时间。三层从输入端起编号，前向按1→2→3、
反向按3→2→1发射；分层统计按同名kernel每步的出现顺序还原。

| 主kernel | 每次launch中位数（ms） | 第1层（ms） | 第2层（ms） | 第3层（ms） |
|---|---:|---:|---:|---:|
| 前向摘要 | 1.1611 | 1.1636 | 1.1616 | 1.1600 |
| 前向 passing | 0.0331 | 0.0324 | 0.0334 | 0.0334 |
| 前向输出 | 1.8454 | 1.8475 | 1.8450 | 1.8452 |
| 反向 dV＋add-mul 摘要 | 2.5852 | 2.5860 | 2.5884 | 2.5668 |
| 反向 passing | 0.0485 | 0.0463 | 0.0496 | 0.0487 |
| 反向 dA/dB/dLSE/drtau partial | 3.7151 | 3.7220 | 3.7178 | 3.6099 |

六主kernel在每个完整3层训练步共18次launch，GPU时间合计中位数为**28.048ms**。
此外，每层仍有独立delta和rtau最终归约，单launch中位数分别0.03024ms与0.02506ms；
它们不计入“六个主kernel”。embedding也单列：前向0.72070ms，反向preprocess0.09333ms、
token梯度0.93828ms、词表梯度5.04090ms。实际每层共有12次Dism命名空间kernel launch，
不能把六主kernel的时间之和当作完整voc_dism或训练步时延。

本次未使用sudo或硬件计数器，只读取CUPTI计时。所有60次原始耗时及完整kernel名见
[`benchmarks/copy_training_kernel_profile_sm120a.json`](benchmarks/copy_training_kernel_profile_sm120a.json)。

```bash
python -m dism_v2.benchmark_copy_training --backend cuda --batch 64 --profile --steps 20
```

原始日志`/tmp/dism-copy-kernel-profile-final.jsonl`。脚本会断言每个Dism kernel恰好出现3×steps次，
并输出合并统计、三个出现位置的中位数与逐launch样本。

## 与tt_dism.py六kernel的对比

同一RTX5090、B64/H4/N1024，旧版N_HEADDIM=N_VOCAB=64、原Q_CHUNK_SIZE=D_CHUNK_SIZE=32。
直接调用未修改的ParallelSoftDiscreteAttention，三个独立输入组依次前向，再逆序反向。
logits和value为BF16 randn，beta为FP32 randn、tau=3，保持原SEPS=1e-4和所有launch参数。
首轮预热3次、复测预热10次，两次各20个前后向循环；每种kernel均60个GPU样本。
下表使用复测CUPTI中位数，v2列沿用上节完整3层训练上下文的测量。

| 阶段 | tt_dism.py kernel | 旧版（ms） | v2 CUDA（ms） |
|---|---|---:|---:|
| 前向摘要 | perprocess_kernel_hh | 0.6537 | 1.1611 |
| 前向passing | chunk_passing_kernel | 0.0567 | 0.0331 |
| 前向输出 | attn_fwd_kernel_hh | 1.1763 | 1.8454 |
| 反向dV＋摘要 | attn_bwd_kernel_hh | 2.1700 | 2.5852 |
| 反向passing | chunk_passing_kernel_bwd | 0.0784 | 0.0485 |
| 反向操作数梯度 | attn_bwd_kernel_post_hh | 4.4948 | 3.7151 |

六项launch中位数之和：旧版**8.630ms**，v2 **9.388ms**。
这不是端到端模型训练步时间，也不是六项实际总时间的中位数。
按这些阶段中位数相加，前向旧版1.887ms、v2 3.040ms；反向旧版6.743ms、v2 6.349ms。
旧版在摘要/前向输出/dV上更快；v2在两次passing和最终操作数梯度阶段更快。
首轮六项中位数之和约8.505ms，与复测有约1.5%的差异，原始样本及范围均保留。

必须区分算法与测量范围：

- 旧N_VOCAB=64是q/k概率向量宽度，与新版core操作数D=64对应；不能直接等同于新版
  embedding词表512。旧版softmax及新版embedding都不计入此六kernel对照。
- 旧score为tau+log(eps+(1-64eps) dot(softmax(q),softmax(k)))，可学习逐行beta控制fallback；
  新版是随机方向Jensen score、行hard_prob=0.5、固定零fallback。不是同数学/同输入对照。
- 旧后处理直接归约tau，并计算logits梯度；新后处理还输出dLSE与tau partial，最终tau归约另发射。
  原子累加归属也不同，不能把对应阶段视为等量GEMM工作。
- 旧版为独立attention前后向、重复输入，无模型/optimizer；v2为真实训练循环。
  输入分布、缓存及调度上下文不完全相同。只报告每launch GPU时间，不宣称某版本整体优劣。
- 首个预热检查旧版输出与五类输入梯度均有限；本次没有重新做旧版数值精度或训练收敛验收。
  未修复、调参或修改tt_dism.py，也没有把旧版当成v2数学oracle。

```bash
python -m dism_v2.benchmark_legacy_launches --warmup 10 --steps 20
```

使用conda blkw、Triton3.8.0。脚本排除softmax、delta、beta梯度、分配/清零等辅助GPU工作，
严格检查六个名字各出现60次。记录见
[`benchmarks/legacy_kernel_profile_sm120a.json`](benchmarks/legacy_kernel_profile_sm120a.json)，
原日志`/tmp/dism-legacy-launch-profile{,-final}.jsonl`。
