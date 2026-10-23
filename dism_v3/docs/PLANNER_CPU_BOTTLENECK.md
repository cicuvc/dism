# 高 hard probability 下 planner 迟到：CPU 诊断（2026-10-23）

本次只执行 CPU 探针。按用户要求不暂停共享 GPU 上的下游评估，使用
`CUDA_VISIBLE_DEVICES=''`，未运行 GPU benchmark、未修改生产实现或训练状态。

当前证据首先指向 **CPU 规划工作量与共享池的层间串行排队**。尚未测得当前
graph replay 的完整时间线，不能给出真实训练中 CPU/DMA/唤醒延迟的精确占比。
GPU 等待 event 的 host 时间不等于通信时间：它还包括等待前面的 GPU 前向。
后续扩展性测试进一步表明：16-worker 池已经较忙，不能把层间等待理解成
16 个 worker 都在闲等；同样 16 个 worker 拆池仅改善约 4%。详见末节。
随后对 planner 内部的阶段、分配和硬件计数器诊断见
[PLANNER_CPU_PROFILE.md](PLANNER_CPU_PROFILE.md)。

## 1. 已确认的实现约束

- `GraphPlannerRuntime` 只有一个 `NativeTrainingPlanner(workers=16)`；各层
  coordinator 的 Python future 都调用同一个 native pool。
- `PlanningWorkers::run()` 在提交任务前持有 `invocation` mutex，并一直持有到
  本次所有 document/head jobs 完成。**层内并行，层与层的 run 调用串行。**
  不是六层的 document/head jobs 一起在 16 个 worker 上公平交错。
- 每层 native plan 调一次 `run()`，compact pack 再调一次。两次之间会回到
  Python，创建输出 storage/views。先完成规划的层可能在 pack 阶段等待其他层
  的整次 plan；`pack_ms` 因此包含锁等待，不能全部解释为内存带宽开销。
- mutex 无 layer/deadline 优先级。后面的 DISM 产生输入较晚，却更早进入反向，
  对应的可用时间窗口更短；不能按“每层单独只需几 ms”判断是否赶得上。
- route sample 只采样一次。迟到的计划仍会继续 plan/pack/upload，并在下一次
  microbatch 前 drain；当前 `PackedTrainingGraph` 未在发现 fallback 后取消
  这次 CPU 工作。若 GPU 已完成，尚未完成的 host 工作还能形成 drain 尾延迟。
  这是代码路径上的可能性，本次没有实测这段尾延迟。

源码入口：`python/flash_dism/planner_graph.py` 的 runtime 初始化、`_produce()`、
`drain()`；`python/flash_dism/inference/csrc/prefill_training.hpp` 的
`PlanningWorkers::run()`；`python/flash_dism/inference/training_planner.py` 的
`prepare()`；`nanochat/nanochat/cuda_graph.py` 的 `forward_backward()`/`drain()`。

## 2. CPU-only 复现：没有任何 host-device 通信，也会积压

使用生产 native plan/pack，compact C64，16 workers。几何为 B8/H12/N2048，
文档边界取新布局 step 22888 保存的 prefetched batch，共 26 个非空 packed
segment（包含 padding segment；不是按有效 target 统计的文档数）。labels 为
独立 q/k 的 iid 合成符号，tau=.5，不是 checkpoint 的实际 labels。
hard mask 按原训练行/head 采样，六层固定种子、共享池，h=.95 时分别激活
92/88/90/92/88/91 个 row-head。两次单层 warmup，七次测量，以下均为中位数。

CPU 输出用普通内存，因此不含 pinned allocation、D2H、H2D、CUDA event、
descriptor/device workspace 分配。未改 kernel/系数数学，native plan/pack 是
当前 worktree 的实现。共享机器上的下游评估仍活跃，CPU/内存压力不是完全隔离的。

| 合成符号数 | 单层 plan | 单层 pack | 单层总计 | 六层完成总时间，间隔 3 ms 提交 | 单层 metadata |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 6.30 ms | .34 ms | 7.06 ms | 42.28 ms | 20.11 MB |
| 32 | 11.86 ms | 3.81 ms | 16.48 ms | 92.18 ms | 55.37 MB |
| 8 | 16.52 ms | 6.23 ms | 23.51 ms | 132.00 ms | 83.47 MB |
| 1 | 14.37 ms | 8.43 ms | 23.68 ms | 136.61 ms | 146.56 MB |

六层总时间从首层提交计到最后一个完成；最后一层在约 15 ms 时提交。
这是可重复的 CPU 压力测试，不是实际 GPU 前向的提交间隔或 training deadline。
没有施加生产 metadata/workspace budget；尤其重复符号的极端计划可能在生产
中触发预算 fallback，不能把它的 CPU 完成时间当成真实 route 预测。

以 512 符号、h=.95、间隔 3 ms 为例：

- 第 0 层 plan 中位数 6.32 ms，pack 12.77 ms，而独跑 pack 只有 .34 ms。
- 第 2 层 plan 13.04 ms，pack 20.68 ms；包含等待其他层，而非纯算法耗时。
- 第 5 层从自身提交到完成 27.22 ms，而非独跑的 7.06 ms。
- 一次实测的六层完成时刻为 19.19、32.93、39.39、40.15、41.05、41.63 ms。

h=.475 时，同一个 512 符号设置的单层总计约 3.15 ms、六层约 23.19 ms；
h=.95 则为 7.06/42.28 ms。两档实际激活的 jobs 数也不同，不能把它当成严格
线性缩放定律。高 h 增加 CPU 工作量，同时 GPU soft 路径减少，可能缩短其窗口。

因此三层尾部 GDN+FFN 提供的是有限 slack，不能保证覆盖六层共享 CPU 服务的
积压。优化目标应包含排队延迟和计划发布顺序，而不仅是 planner 单层吞吐。

## 3. 真实 checkpoint 与历史传输测量提供的交叉证据

此前同一新布局 checkpoint 的 `layout_diagnosis/middle_symbols.json` 保存了
真实 labels 的单次 native planner statistics。B4/H12/N2048、h=.95：

| 层（零基） | native plan 时间 | metadata |
| --- | ---: | ---: |
| L3 | 5.55 ms | .11 MB |
| L4 | 3.92 ms | 2.35 MB |
| L5 | 3.87 ms | 2.21 MB |
| L6 | 4.66 ms | 6.05 MB |
| L7 | 5.28 ms | 9.53 MB |
| L8 | 5.36 ms | 12.56 MB |

这些是之前单次 standalone 规划的统计，未包括 pack，不是本次七次中位数；
也不能将 B4 时间直接乘二当作 B8。它们说明真实 labels 下 CPU 规划本身已有
毫秒量级，而传回的 plan 大小并不等同于最极端合成场景。

同批次此前完整模型 graph 的两次后续 replay，按前向层序记录的 routes 都是
`e,e,e,e,q,q`，所有 reason 均为 None：这两个探针中 L7/L8 的 fallback 不是
预算拒绝，而是 sample 时计划尚未 ready。首轮全 q，不能用首轮评估稳态性能。

历史 2026-10-14 同机 W16/B16/H6/N2048 的传输测试（相同 B×H×N，但为旧 C16
classic metadata、uniform docs，不是当前 compact graph）给出：

| 输入 | CPU total | D2H 完成墙钟 | H2D 完成墙钟 |
| --- | ---: | ---: | ---: |
| iid512 | 10.76 ms | .071 ms | .594 ms |
| 全重复 | 30.72 ms | .072 ms | 3.637 ms |

对应输入 labels 约 1.57 MB；当前 B8/H12/N2048 的每层 labels/hard/tau D2H
总共 1,573,008 bytes。历史结果只用于判断数量级，不可替代当前 graph 的 DMA
或 event 唤醒计时，也不可用作准确性能预测。

当前 `_upload_plan()` 还包含动态 pinned/device allocation、Python views、
multi-chunk IDs 和 descriptor 构造。这些开销不是单次 bulk H2D DMA，可以放大
CPU 调度/GIL 长尾，尚未在当前训练场景下分离。`upload.synchronize()` 在该层
publish 入队之后调用，不能直接把它的全部等待算成 publish 前阻塞。

近期 MXFP8 CE 吞吐对照也符合窗口敏感性：h=.95 时 BF16 CE 为 e1452/q84，
MXFP8 CE 为 e1312/q224，而整体仍快约 3.1%。它同时改变计算时间和量化训练
轨迹，不能据此断言所有额外 fallback 都由 CE 缩短窗口造成。

## 4. 下一步优先级与尚未完成的 GPU 诊断

优先检查 CPU 调度：让已经完成 plan 的 pack 尽快完成，研究按反向 deadline
安排层间任务，并测量更细粒度的 bounded job interleaving。不能直接删掉 mutex
或给六层各开 16 workers；前者需重新验证异常/队列/生命周期同步，后者会过度
订阅 CPU。再考虑重用容量合适的 pinned/device buffers，减少分配与 Python
构造；保留原 generation 生命周期和上传后发布语义。上述改动均未实施。

GPU 空闲后，应在真实 checkpoint、相同 hard mask 下记录以下节点：

1. GPU snapshot/D2H 开始和完成、该层 backward sample 时刻。
2. host event wait 返回、native invocation 锁等待、实际 plan、pack 锁等待、
   实际 pack，以及 pinned allocation/views 时间。
3. upload host enqueue、H2D 完成、descriptor/ready 发布完成、全图完成和 drain
   返回时刻。

各层的 `ready - sample` 才是赶不上 deadline 的直接指标；不能用
`input_ready.synchronize()` 的总 host 等待替代 D2H 延迟。同步对齐 CPU/GPU
时钟也不能靠 Python wall time 与 CUDA elapsed time 直接相减。

本阶段结论：**CPU 规划与层间排队已实证为大开销，D2H 带宽不是首要嫌疑；
当前 host 唤醒、allocation、上传和实际 deadline 的精确份额仍未测量。**

## 复现

工具：`tools/diagnose_planner_cpu.py`。在仓库根目录、conda blkw 下运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" \
python dism_v3/tools/diagnose_planner_cpu.py \
  --train-state /path/to/train_state_022888_rank0.pt \
  --output dism_v3/build/planner_cpu_20261023/new_results.json
```

本次完整原始结果为 ignored `build/planner_cpu_20261023/results.json`，含每次
分阶段时间、各层提交/完成时刻、metadata 统计、native binary hash；输出完成后
断言 `torch.cuda.is_initialized()` 为 false。没有生产代码/算子变更，未做新
sanitizer 或数值 oracle 验证。

## 5. 调度浪费还是 24 核饱和：worker 扩展性对照

进一步使用 `tools/diagnose_planner_scaling.py` 测量进程所有线程的 CPU time，
以 `CPU seconds / wall seconds` 计算平均运行核数。它不表示相同 IPC 的核数，
不区分有效计算与运行态开销，也不等同于内存带宽利用率。机器为 285K，8P+16E、
24 个物理核且无 SMT。当前生产 runtime 只开 16 个 native workers；六个 host
coordinator 大部分时间等待，不能简单把线程数加起来当作 22 个计算核。

本轮仍为 CPU-only、B8/H12/N2048、h=.95、同一组真实文档边界和合成 labels，
普通 CPU 输出存储。初测每配置九次，随后随机化配置顺序做十五次重点复测。
运行前发现下游 PG19 8K 已因 CUDA OOM 退出，因此本轮没有下游评估进程并发，
未重启或修改该任务。与上一轮存在机器负载差别；以下只比较本轮内部配置。

### 单层 scaling（九次中位数）

| workers | iid512 单层 | 平均运行核数 | iid32 单层 | 平均运行核数 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 90.53 ms | 1.00 | 184.16 ms | 1.00 |
| 4 | 23.14 ms | 3.86 | 49.05 ms | 3.90 |
| 8 | 13.46 ms | 7.42 | 26.82 ms | 7.56 |
| 12 | 9.20 ms | 10.79 | 22.16 ms | 11.05 |
| 16 | 6.84 ms | 13.86 | 19.21 ms | 14.41 |
| 20 | 5.70 ms | 16.80 | 16.88 ms | 17.20 |
| 24 | 5.70 ms | 18.33 | 16.12 ms | 19.31 |

16 个 worker 平均约 14 个核在运行，层内并行并未失效。20–24 的边际收益已经
降低；不能只因没有达到 24.0 就认为这些空闲比例都能通过删除锁变为加速。
串行收尾、阶段间切换、长任务尾部及异构核负载都会影响这个数。未做硬件计数器
采样，不能进一步断言算术执行单元或内存带宽已经饱和。

### 六层对照（间隔 3 ms 提交；十五次复测中位数）

以下 native workers 总数有明确上限，不是六层各开 16 个线程。

| 调度 | 总 workers | iid512 总时间 | 平均运行核数 | iid32 总时间 | 平均运行核数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 共享 1×16 | 16 | 43.25 ms | 14.10 | 93.32 ms | 14.82 |
| 分 2×8 | 16 | 41.58 ms | 14.08 | 89.47 ms | 14.35 |
| 共享 1×20 | 20 | 35.77 ms | 17.38 | 83.82 ms | 18.26 |
| 共享 1×24 | 24 | 33.91 ms | 19.03 | 83.59 ms | 20.18 |
| 分 3×8 | 24 | 36.32 ms | 16.69 | 73.17 ms | 19.11 |
| 分 6×4 | 24 | 42.67 ms | 13.70 | 73.91 ms | 17.06 |

解释：

- **同样 16 workers，分池仅缩短约 3.9–4.1%。** 不能把全部 43/93 ms 延迟
  都归咎于 invocation 锁；等待该锁时通常是其他层在使用 worker。
- 共享池从 16 增至 24，iid512 缩短约 21.6%，iid32 约 10.4%，不是线性
  的 1.5 倍吞吐。iid32 的 20→24 基本没有收益，但进程 CPU time 从约
  1.53 CPU-s 增为 1.67 CPU-s；更多 CPU 占用不等于更高效率。
- **同样 24 workers，分 3×8 对 iid32 再缩短约 12.5%，对 iid512 却慢约
  7.1%。** 分层并发确实有改进空间，但固定切分会降低单层可用并行度，不能
  作为普适修复。六层各 4 workers 在较轻的分批到达场景几乎没有总时间收益。
- 上一节的 pack 等待依然真实，但它主要体现 **完成顺序/deadline 问题**。
  一层规划完却迟迟不能 pack/publish，并不意味着整机期间没有做有效工作。

因此更准确的结论是：**当前没有把全部 24 核用满，但已较充分利用配置的
16-worker 计算预算；工作量本身很大，调度与剩余核资源提供有限且依赖输入的
改善空间。** 不能期待只去掉层间锁就获得数量级收益，也不能声称机器已经
达到不可改善的硬件极限。

下一步应保持全局 worker 预算，联合优化按 backward deadline 的任务调度、
plan→pack 的及时衔接和每层的并行粒度；在真实 labels 与实际 GPU 截止时间
上评估收益。单纯让每层同时跑、或者盲目加线程，都不够。这里只改诊断工具，
没有修改默认 workers、生产 affinity、线程池或 kernel。

产物：`build/planner_cpu_20261023/scaling.json`、
`build/planner_cpu_20261023/scaling_equal_workers.json`，包含 CPU/wall 时间、
context switches、每层完成时间及原始样本。复现后者：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" \
python dism_v3/tools/diagnose_planner_scaling.py \
  --geometry-json dism_v3/build/planner_cpu_20261023/results.json \
  --output dism_v3/build/planner_cpu_20261023/new_scaling.json \
  --focus --repeats 15
```
