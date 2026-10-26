# K 状态块与 Q 读取块解耦原型（2026-10-26）

2026-10-27 更新：解耦反向和直接打包已经接入 top-k 训练及 planner CUDA Graph，
模型支持 K=1/2/4（K1 保留原生路径）。已加入 packed 文档、原始行 hard mask、
selector 梯度和异步 ready/fallback 协议，见
[当前训练支持及验证](TOPK_HARD_TRAINING.md)。模型 hard forward/prefill 尚未接入。
以下保留 2026-10-26 隔离原型的实现与测量记录。

已实现独立 CPU planner、GPU prefill 和解析反向，并用真实 checkpoint 的
q/k 序列完成性能测试。首版8K top4六层GPU prefill/反向耗时分别降低42%/39%。
后续直接打包版已消除额外CPU转换开销；即使旧格式也应用相同打包优化，
8K top4完整串行算子流程仍改善约3.0%/2.2%。这不是整模型训练吞吐测试。
该阶段实现均位于tools，当时未接入生产dispatch、planner graph协议或训练配置。

## 后续：直接计数和写入最终 arena

`tools/decoupled_readout_direct.hpp`实现新路径，保留原`prepare`作为对照，
通过隔离扩展的`prepare_direct`显式使用。SAM/分解、权重、GPU kernel和
launch参数不变，改动仅在CPU输出组织及上传适配：

1. 每个worker在`Builder.finish()`后统计最终K块、Q tile和保留旧路径的数量。
   不再构造旧`TrainingBatchProgram`的boundary数组。长stream仅凭事件数
   即可判定路由；短stream做reset-aware计数，不生成边界列表。
2. 对job的计数做前缀和，一次分配未预清零的连续arena。worker按不重叠区间
   直接写入；padding逐项写零，避免全缓冲区先清零再覆盖。
3. 按事件顺序直接形成K/Q字段，不再构造每个K块的keys/queries向量、每head
   临时Plan，也不再串行拼接或修正已生成的block索引。
4. prefill的长/短stream路由索引也直接写入；Program临时数组在worker上释放。
5. 可选allocator回调由Python提供pinned uint8 arena，C++保持Torch-free。
   Executor直接使用该arena的GPU views，一次H2D，不经第二套pinned字段打包。
   无allocator时仍可返回普通CPU arena，用于独立CPU对照。

输出语义字段与首版逐字节一致，新增的prefill路由索引与独立NumPy构造一致。
仍保留标量Program，未将SAM/分解和向量执行融合，也未实现生产异步ready协议。

### CPU结果

相同真实输入，8 workers，3轮预热、11轮随机交错；计时包含普通CPU内存
分配、prepare和释放，不含pinned/H2D。数值为六层中位数之和，单位ms。
两个旧格式对照也来自同一topk原型，不是生产top1 planner的绝对基线。

| N | K | 阶段 | 原旧格式 | 首版解耦转换 | 直接旧格式 | 直接解耦 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2K | 1 | prefill | 28.80 | 28.67 | 25.00 | 24.87 |
| 2K | 1 | backward | 29.50 | 31.22 | 25.39 | 25.31 |
| 2K | 4 | prefill | 47.95 | 47.61 | 41.53 | 41.49 |
| 2K | 4 | backward | 49.99 | 52.48 | 43.58 | 43.18 |
| 8K | 1 | prefill | 127.26 | 128.24 | 107.47 | 108.06 |
| 8K | 1 | backward | 142.38 | 153.74 | 109.80 | 109.81 |
| 8K | 4 | prefill | 232.14 | 239.02 | 182.76 | 181.98 |
| 8K | 4 | backward | 254.61 | 280.50 | 190.04 | 188.51 |

8K top4解耦路径中，pack阶段prefill从81.92降到25.59ms，反向112.90降到
24.86ms。不同路径pack计时所含工作略有变化：新路径计数已并入build，
pack包含arena分配、并行写入和Program释放。因此最终判断以完整CPU wall为准。

### 完整串行算子流程

另外实测prepare → pinned上传/对象初始化 → eager GPU执行并同步 → 释放。
无CPU/GPU重叠，不包括标签D2H、selector及模型其他层；不是生产训练步。
同样按六层中位数求和。9轮四路对照中，8K top4首版解耦转换与直接解耦分别为：
prefill 293.78→190.38ms，反向321.07→207.48ms，约减少35%。
这里大部分是通用打包优化，不能全部归为K/Q解耦收益。

为剥离这项通用收益，对**双方都直接写pinned arena**的旧格式和解耦格式
另做31轮随机交错复测，3轮预热，结果如下：

| N | K | 阶段 | 直接旧格式 ms | 直接解耦 ms | 降低 |
| --- | ---: | --- | ---: | ---: | ---: |
| 2K | 4 | prefill | 48.790 | 48.444 | 0.7% |
| 2K | 4 | backward | 52.419 | 51.287 | 2.2% |
| 8K | 4 | prefill | 194.242 | 188.493 | 3.0% |
| 8K | 4 | backward | 206.382 | 201.765 | 2.2% |

8K top4各阶段分别求和，旧→新：

| 阶段 | prefill ms | backward ms |
| --- | ---: | ---: |
| CPU prepare，直接写pinned | 175.06 → 173.43 | 182.77 → 183.02 |
| H2D及对象初始化/同步 | 7.79 → 8.11 | 8.98 → 9.48 |
| eager GPU执行及同步 | 10.88 → 6.49 | 14.29 → 9.00 |

阶段中位数之和不必严格等于整体中位数。新格式metadata更大，H2D阶段略慢，
但不再抵消GPU收益。CPU建图/标量Program形成仍占整体大部分时间，因此
GPU约40%的收益只转化为完整串行流程约2%～3%的收益。

按每层31个配对样本的差值中位数求和，并独立重采样每层配对样本10000次
（seed729），8K prefill/反向节省时间的95% bootstrap区间为
4.88～5.93ms / 4.07～5.20ms。该区间仅描述本批样本波动，不代表跨语料、
机器或训练过程的保证。2K prefill收益很小：9轮首测约1%回退，31轮复测约
0.7%改善；不据此设置全局默认dispatch。

### 直接路径验证和复现

- 576组Python并行元数据逐字节对照，worker1/8，B2H3，K1/2/4，随机/重复/
  无匹配，N1/65/129/257，前向/反向及四种路由阈值；真实输入48组也通过。
- 972组C++场景，每组增加8种格式对照，共7776组direct/旧格式检查，覆盖
  内部reset和BK/BQ16/32/64。用0xa5填充arena后逐字节检查全部字段及padding，
  ASAN/UBSAN通过；保留原系数/导数oracle。额外验证int32索引溢出拒绝。
- pinned直写的R32/DV64 GPU前向及反向quick suite通过原FP64数值门限，
  CUDA memcheck为0错误。未改变kernel或数值容差；原型首版的同步/race检查
  见下文，不把本轮memcheck表述为重做了全部sanitizer。
- 验证allocator拒绝错尺寸、错dtype及只读arena，输出views共享返回arena。
  pinned路径额外断言上传源地址与C++输出arena一致。
- 本轮实际产生的30个cubin重新检查PTX/SASS，无CALL；64×64 Q反向仍为
  255 registers、8 spill，SASS为8处LDL/8处STL，与首版限制一致。

在原复现环境中执行：

```bash
python dism_v3/tools/build_decoupled_readout.py --output dism_v3/build/decoupled_readout/direct_native
python dism_v3/tools/check_decoupled_direct.py \
  --native dism_v3/build/decoupled_readout/direct_native/_dism_decoupled.so
python dism_v3/tools/check_decoupled_readout.py --directory dism_v3/build/decoupled_readout \
  --native dism_v3/build/decoupled_readout/direct_native/_dism_decoupled.so \
  --direct --quick --rank 32 --value-dim 64
python dism_v3/tools/bench_decoupled_cpu.py --directory dism_v3/build/decoupled_readout \
  --native dism_v3/build/decoupled_readout/direct_native/_dism_decoupled.so \
  --inputs dism_v3/build/top4_event_probe/inputs --direct \
  --output dism_v3/build/decoupled_readout/results/direct_cpu_full.json
python dism_v3/tools/bench_decoupled_pipeline.py --directory dism_v3/build/decoupled_readout \
  --native dism_v3/build/decoupled_readout/direct_native/_dism_decoupled.so \
  --inputs dism_v3/build/top4_event_probe/inputs --direct-only --topk 4 --repeats 31 \
  --output dism_v3/build/decoupled_readout/results/direct_pipeline_paired31.json
```

原始数据位于`results/direct_cpu_full.json`、`direct_pipeline_full.json`和
`direct_pipeline_paired31.json`。以下各节保留首版转换原型的结构、数值及GPU
结果；其中CPU额外开销的结论已被上述直接打包实验更新。

这次实现的是 [此前结构探测](PLANNER_CPU_OPTIMIZATION.md) 中尚未实现的
K 状态块/Q tile 解耦；不是之前只重放少量 K 的 sparse-read 实验。

## 执行格式

SAM、centroid 分解、每候选的归一化分母及 tau 导数保持原算法。以相同
Program 为输入，按每 BK 次 K 更新切状态块，再将其间的 Q 读取切成 BQ 大小
的独立 tile。每个 Q tile 记录状态块编号，每个 Q 事件记录该块内因果可见的
K 数量。不同 Q tile 共享一个 FP32 R×DV 的入块状态。

- prefill：K block summary → 按 stream 扫描入块状态 → 并行 Q tile 读出。
- backward：K summary 和每块的 Q adjoint summary → 原有双向扫描 →
  Q tile 内局部梯度及历史 dSQ → K block 的未来 dSK/dV，另累加 tau。
- Q adjoint 由一个 CTA 顺序遍历该 K 块的 Q tiles 归并，避免每个 Q tile
  保存矩阵 partial，也不使用共享内存 FP32 atomicAdd。向量输出/梯度仍使用
  原框架允许的 global FP32 atomics，不保证 bitwise/batch invariance。
- 内部 decay=0 将 stream 分成独立段。第一个状态块的入状态为零；跨块首 K
  的 decay 仍计入该块 prefix。局部读取严格按每个 Q 的可见 K 数量掩码。

设入块状态为 H，块内 K 的累计 log decay 为 p_j，Q 读取时的 prefix 为 p_q，
Program 的 K/Q 权重分别为 u_j、w_q，则读取为：

```text
o_q += w_q [ exp(p_q) sq_qᵀ H
             + Σ_{j < visible(q)} u_j exp(p_q − p_j) (sq_qᵀ sk_j) v_j ]
```

K 块 summary 使用末尾 prefix 将局部更新折算到块末；每块 Q summary 汇总
`w_q exp(p_q) sq_q ⊗ dO_q`，提供对入块状态的 adjoint。逐事件 dlogweight
仍来自原 planner，没有改变 per-candidate 归一化或 selector 定义。

保留原短 stream 路径：反向只改原本超过一个 C64 chunk 的 stream，prefill
只改原本超过64个 C16 chunk 的长 stream。BK/BQ 分别测试32/32、64/32、64/64，
本轮总体以64/64最佳。单块、极少匹配的层几乎没有收益，新增 launch 还可能
造成微小回退。本原型没有跨 stream 合并相同 K 前缀，也不跨 head 合并状态。
固定 BK/BQ 后工作量仍随原分解事件量增长，不构造 N² 中间矩阵。

## 测量范围

RTX 5090，driver 615.71.09，B4/H12/R32/DV64，BF16 向量、FP32累加。
输入是 top4 最终 step22888 checkpoint、PG19 前四篇书、BOS+正文、六个
DISM 层捕获的真实标签、候选与 tau。top1 只取相同捕获的候选0，不是独立
top1 模型前向。每行是独立完整序列，全部 hard。

真实符号决定 planner 拓扑；SQ/SK/V 和每候选 dO 使用固定随机 BF16 payload。
此微测不包含 selector、候选输出加权、CPU↔GPU传输或整模型执行。
旧 GPU kernel 保持不变，只在隔离 Python adapter 中允许 Q 的 N×K 虚拟轴
和 K/V 的 N 物理轴。top1 旧格式另与生产 planner 的数组逐项核对。

GPU 时间为上传完成后的 CUDA graph replay，5轮预热、21轮随机交错测量，
逐层取中位数再求六层之和。不是一个训练步的实测时间。

| 上下文 | 候选数 | prefill 旧→新 ms | 降低 | 反向旧→新 ms | 降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K | 1 | 0.933 → 0.822 | 11.9% | 1.035 → 0.914 | 11.8% |
| 2K | 4 | 1.742 → 1.554 | 10.8% | 1.991 → 1.443 | 27.5% |
| 8K | 1 | 5.408 → 3.336 | 38.3% | 7.999 → 5.217 | 34.8% |
| 8K | 4 | 10.378 → 5.984 | 42.3% | 13.826 → 8.441 | 38.9% |

8K 各层 GPU 时间，DISM 层按0～5编号，单位 ms：

| 层 | top1 prefill | top1 backward | top4 prefill | top4 backward |
| ---: | ---: | ---: | ---: | ---: |
| 0 | .762 → .382 | 1.297 → .609 | 1.678 → .730 | 2.567 → 1.280 |
| 1 | .862 → .786 | 1.239 → .824 | 2.363 → 1.597 | 3.031 → 1.773 |
| 2 | .565 → .466 | .880 → .521 | 1.381 → .902 | 1.882 → 1.097 |
| 3 | 1.975 → 1.232 | 2.593 → 1.868 | 3.168 → 1.873 | 3.933 → 2.605 |
| 4 | 1.126 → .352 | 1.839 → 1.241 | 1.475 → .569 | 2.149 → 1.419 |
| 5 | .118 → .118 | .151 → .155 | .314 → .314 | .264 → .267 |

## 状态内存和 CPU 代价

以下为8K六层逐层矩阵临时工作集之和，不是同时存活量或整模型峰值；包含
保留的旧路径与新路径，排除共同向量输入/输出和独立统计的 metadata。

| 候选数 | prefill 矩阵 MiB | backward 矩阵 MiB | prefill metadata MiB | backward metadata MiB |
| --- | ---: | ---: | ---: | ---: |
| top1 | 1364.6 → 86.7 | 1066.5 → 554.1 | 165.5 → 174.4 | 158.4 → 190.2 |
| top4 | 2908.3 → 108.5 | 2002.0 → 726.0 | 266.4 → 280.7 | 257.0 → 309.2 |

top4 单层矩阵工作集最大值：prefill 877.8→37.2 MiB，反向628.9→252.3 MiB。
反向 K 状态块数确为之前预测的 top1 35,452、top4 46,451；64/64 下对应
Q tile 数为51,861、107,546。减少状态快照没有消除 Q 读取工作。

CPU 格式转换按 job/head 并行，随后串行拼接各 job 的输出。当前仍先构造
旧 Program 及旧边界，还会生成临时 vector 和额外 Q 索引，没有直接写入
生产 pinned arena。这是额外成本，不能把 GPU 结果当作 planner 全流程结果。

单独 CPU 复测使用8 workers、3轮预热、11轮随机交错 A/B；计时包含 native
prepare、普通内存分配和输出释放，不含 pinned 分配/H2D。对照也使用相同
topk Builder、系数缓存和本实验的旧格式 pack；不是生产 top1 prepare 的
绝对性能基线。以下同样是逐层中位数之和：

| 上下文 | 候选数 | prefill CPU 旧→新 ms | backward CPU 旧→新 ms |
| --- | ---: | ---: | ---: |
| 2K | 1 | 32.77 → 33.46 | 33.82 → 36.96 |
| 2K | 4 | 61.00 → 61.32 | 67.66 → 71.91 |
| 8K | 1 | 154.78 → 161.22 | 167.44 → 178.85 |
| 8K | 4 | 252.55 → 266.53 | 276.74 → 288.98 |

因此 CPU/GPU 完全串行时本版可能更慢。适合继续推进的方向是直接生成
K/Q 格式、计数后并行写入最终 arena、复用工作内存，避免先生成旧边界和
逐 job 临时数组后的再次复制。是否提高训练吞吐还取决于 CPU 工作的隐藏
程度、H2D metadata 增长、原 hard event/fallback 的触发率，需要接入后测量。

## 验证与限制

- CPU：972组标量系数/导数对照，包含K1/2/4、负/零/正tau、重复/随机/稀疏
  标签、内部 reset、BK/BQ16/32/64；ASAN/UBSAN通过。FP32 metadata 对FP64
  Program 的最大系数/导数绝对误差为2.75e-7/1.30e-6。
- GPU：80组 dense FP64 oracle 对照，BF16与TF32x3、全路由与混合路由，
  前向及dSQ/dSK/dV/tau全部通过现有梯度门限。另要求tau rtol2e-4、atol2e-6，
  未放宽仓库容差。BF16最大向量/输出相对L2误差0.244%，tau为2.69e-6；
  TF32x3向量/输出误差小于3.1e-7。最终补测R32/DV64及完全无匹配空计划通过。
- 48组真实输入/候选数/阶段、每组3个tile变体全部通过旧GPU基线对照。
  最大向量/输出相对L2误差0.2001%，tau为1.99e-5；最小向量cosine>0.999997。
  这只验证随机 payload 上的算子误差，不代替真实模型 loss/训练检查。
- BF16 R32/DV64、top1/top4、部分tile与多状态块：memcheck、synccheck、
  racecheck均通过，无错误/警告；不是所有shape和精度的sanitizer穷举。
- 实测BF16的41个cubin检查无PTX/SASS CALL。新64×64反向Q kernel占255
  registers，Triton报告8 spill，并在SASS看到8处LDL/8处STL；64×32新kernel
  无spill但总体更慢。旧baseline部分kernel也有spill。本轮不掩盖这一限制。
- 目前支持独立完整行、所有行hard、head级tau、K≤4；没有实现生产异步
  ready/flag协议、动态hard子集、图内运行时planner dispatch或selector加权。
  离线准备后的GPU执行可capture/replay，不代表完整planner graph已接通。

## 复现入口

源码：`tools/decoupled_readout_plan.hpp`、`decoupled_readout_bind.cpp`、
`decoupled_readout.py`。`decoupled_topk.patch`保存旧topk CPU实验对当前生产
头文件的差分；构建仅应用于输出目录，使用本版本inference/build.py、MAX_JOBS=2。
默认构建不再依赖先前忽略的header快照，重建后五个依赖头文件逐字节一致。

在仓库根目录，使用conda blkw并设置：

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
python dism_v3/tools/build_decoupled_readout.py --output dism_v3/build/decoupled_readout/native
python dism_v3/tools/check_decoupled_readout.py --directory dism_v3/build/decoupled_readout
python dism_v3/tools/bench_decoupled_readout.py --directory dism_v3/build/decoupled_readout \
  --inputs dism_v3/build/top4_event_probe/inputs \
  --output dism_v3/build/decoupled_readout/results/full.json
python dism_v3/tools/bench_decoupled_cpu.py --directory dism_v3/build/decoupled_readout \
  --inputs dism_v3/build/top4_event_probe/inputs \
  --output dism_v3/build/decoupled_readout/results/cpu_paired.json
```

CPU sanitizer入口为`tools/check_decoupled_readout.cpp`，include构建输出的
`native/inference/csrc`及`dism_v3/tools`，使用`-fsanitize=address,undefined`。
GPU sanitizer使用`check_decoupled_readout.py --quick --rank 32 --value-dim 64`。

输入/checkpoint哈希在`build/top4_event_probe/inputs/manifest.json`。
本轮JSON、日志、PTX/cubin/SASS及编译产物位于忽略的
`build/decoupled_readout`，实际存储于数据盘
`dism-runs/diagnostics/decoupled-readout-20261026`。结果不写入Git。
