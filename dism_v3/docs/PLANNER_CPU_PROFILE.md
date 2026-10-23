# CPU planner 常数与微架构诊断（2026-10-23）

结论：**已确认大量小对象分配/释放是可观的常数开销；更密集的匹配还增加了
centroid 分解、标量归一化/导数计算及 compact packing 的成本。当前证据不支持
把主要问题归为 DRAM 读带宽或 LLC miss。** 指针依赖链和分支压力仍存在，但
没有可靠的 Top-down 百分比可将它们单独定量归因。

前一阶段的线程池/排队分析见 [PLANNER_CPU_BOTTLENECK.md](PLANNER_CPU_BOTTLENECK.md)。
本轮不修改生产 planner、数学、线程数、allocator、affinity 或训练状态。
allocator/affinity 的变更仅限新启动的独立探针进程。

## 1. 工作负载和验证

- CPU 为 Core Ultra 9 285K，8P+16E。单 worker 探针绑定 CPU 0；16-worker 探针
  沿用 OS 调度。无 GPU 操作，没有重启下游评估。
- B8/H12/N2048、真实 checkpoint prefetched batch 的 26 个 packed segments，
  hard probability .95，92/96 个 row-head 激活，共 299 个 document/head jobs。
- q/k labels 为 iid512 或 iid32，tau=.5、compact C64，**不是训练 labels**。
- 基线独立 C++ target 直接包含当前 worktree 的生产 headers，`-O3 -g`、无
  fast-math。输出为普通 CPU 内存；不含 pinned allocation、Python、D2H/H2D。
- 插桩版仅在 ignored build 副本加入阶段时钟和 C++ new/delete 计数。iid512、
  iid32 的完整 compact bytes 均与基线及当前 Python native binding **逐字节
  相等**。allocator A/B 的两种输入、三个 allocator 设置也逐字节相等。
- 本次没有生产数值实现变更，没有新增梯度 oracle/sanitizer 覆盖；字节相等
  只覆盖选取的两个计划，不能替代后续优化的完整回归。

## 2. 阶段计时：固定容器成本很大，密集匹配的算法成本继续上升

单 worker、CPU 0，约五秒重复执行。以下为插桩平均每个 B8 计划的时间，包含
299 个 job 的累计工作，不是一个 document/head 的时间。基线单 worker 的
中位耗时约 92.98/184.65 ms，插桩总耗时约 94.50/185.04 ms；插桩不是零扰动。

| 阶段 | iid512 | iid32 |
| --- | ---: | ---: |
| τ/logweight/导数预计算 | 4.76 ms | 4.59 ms |
| SAM 构建 | 24.40 ms | 25.13 ms |
| adjacency 和相关数组构造 | 7.57 ms | 8.48 ms |
| query matching | 2.64 ms | 5.66 ms |
| centroid decomposition / emit | 23.66 ms | 56.69 ms |
| 标量 denominator / 导数 pass | 4.40 ms | 27.35 ms |
| coefficient 输出 pass | 2.03 ms | 12.32 ms |
| Builder 销毁 | 17.99 ms | 18.22 ms |
| chunk 边界统计 | .66 ms | 4.68 ms |
| compact pack | 4.82 ms | 19.21 ms |

其余为输出分配/释放、输入切片等。iid512 的 SAM+adjacency+Builder 销毁约
占阶段时间 **53%**，decompose/emit 约 **25%**；iid32 中前者约 **28%**，
decompose/emit 约 **31%**，两段 scalar/coefficient pass 合计约 **22%**。
因此“planner 算法同样复杂，所以全都只能靠更多核”并不充分：其中有很大的
数据结构与生命周期常数，并非全部是必需的匹配数学。

插桩统计每个 B8 计划约 **109.1 万 / 111.8 万次 C++ 分配**，累计请求字节约
147.8/291.6 MB。它们不是 peak RSS，也不是实测 DRAM 流量。单是 SAM 阶段就有
约 67 万次分配、约 61–62 MB 请求；adjacency 阶段约 36 万次分配。这些计数
涵盖被插桩阶段的 new/new[]，不等于所有进程级 malloc 调用。

## 3. VTune 热点与 allocator 因果对照

VTune 2026.6 的 iid512 单 worker 热点中，glibc 的 malloc/free、`_int_malloc`、
`_int_free`、`malloc_consolidate`、`unlink_chunk` 合计约 **38%** 自耗 CPU
采样时间。这与分阶段计时相符。

iid32、16 workers 的软件热点中，`pack_into_compact` 的 job lambda 约
**19.9%**，`__GI___exp` 约 **10.7%**，`__log1p_fma` 约 **4.0%**，
`__ieee754_log_fma` 约 **1.6%**。这些是函数自耗采样时间，不是互斥的源码
高层阶段占比；例如 pack 内调用 log 的时间会归到 libm。

保持相同二进制、输入和输出，只改变子进程 allocator。配置顺序随机化，
每个配置三次、每次至少 1.5 秒，以下为耗时中位数：

| 输入 / workers | glibc 默认 | glibc tcache_count=64 | jemalloc |
| --- | ---: | ---: | ---: |
| iid512 / 1 | 92.98 ms | 93.05 ms | **64.29 ms** |
| iid512 / 16 | 8.76 ms | 8.92 ms | **5.33 ms** |
| iid32 / 1 | 184.65 ms | 181.98 ms | **153.24 ms** |
| iid32 / 16 | 17.51 ms | **14.14 ms** | 15.47 ms |

稀疏输入换 jemalloc 降低约 31%/39% 耗时；密集输入并不存在一个设置对所有
并行度都最好。因此这是 **分配器与对象生命周期成本的证据**，不是建议给
生产训练直接设置全局 LD_PRELOAD，也不是端到端训练吞吐收益。

## 4. 硬件计数器：指令量大，读取 miss 并不突出

经用户授权临时将 `kernel.perf_event_paranoid` 从 3 调到 0；采样结束后已恢复
为 3。VTune 使用 driverless Perf。除 VTune 的采样事件外，还用独立
`perf_event_open` 读取子进程及其 worker 的完整计数。CPU 0、单 worker、排除
kernel/hypervisor，每个计数器的 running/enabled 均为 **1.0**，没有 multiplex。

| Linux perf 指标 | iid512 / glibc | iid512 / jemalloc | iid32 / glibc | iid32 / jemalloc |
| --- | ---: | ---: | ---: | ---: |
| 每个计划退休指令 | 10.68 亿 | 7.89 亿 | 17.13 亿 | 14.39 亿 |
| IPC | 3.09 | 3.20 | 2.55 | 2.56 |
| branch miss / retired branches | 1.31% | 1.37% | 2.67% | 3.01% |
| L1D read miss / 千条退休指令 | 1.49 | .66 | 1.20 | .71 |
| LLC read miss / 千条退休指令 | .0108 | .0106 | .0116 | .0128 |

换 allocator 的退休指令数下降约 **26%/16%**，IPC 基本不变；近端访存也改善。
这支持优先减少管理指令、对象分配与指针结构，而不是先假设 DRAM 带宽已经耗尽。
它不排除 L1 hit 的 load-use 依赖链、store/RFO 流量、TLB 或多线程下更大的
工作集成本；这里只测了列出的读事件，不能把 miss 数直接换算成带宽饱和度。

VTune 的窄事件集合也给出较低的 retired-load miss：iid512 单 P 核采到 L1
miss 1.685 亿、L2 miss 47 万、L3 miss 14 万，对应 1391.92 亿退休指令。
L3 只有 14 个采样，精度有限。iid32/W16 的 **P 核部分**采到 5192.12 亿
指令、L2 miss 1941 万、L3 miss 348 万。该配置没有对应 E 核 load-miss
事件，不拿 P 核分子除以 P+E 总指令。Linux 通用 cache 事件与 VTune
`MEM_LOAD_RETIRED` 的定义不同，不将两者当成完全相同指标。

### Top-down 限制

广事件 uarch-exploration 的四大类出现内部不一致：例如 iid512 显示 retiring
74.1%、front-end 34.7%、backend 57.1%，合计超过 100%。缩小到顶层事件后，
四类合计恢复 100%，但 iid32 的退休指令计数约 74.7 亿/12 秒，仍与窄事件
采样及独立非采样计数的每秒约 90–100 亿指令矛盾，差约 15 倍。**这两组 Top-down 类别
百分比全部不作为结论依据**，保留原始结果，不用它们宣称 frontend-bound
或 memory-bound 的精确份额。当前尚未隔离 profiler/PMU 指标异常的具体原因。

分支 miss 和源码中的串行依赖是确实存在的，但这轮不能可靠地拆出它们各自
造成多少停顿周期。原始计数、软件热点、阶段插桩和 allocator A/B 是相互独立
的证据；单一失效的 Top-down 报告不覆盖这些有效测量。

## 5. 对应源码和优化优先级

### 优先：减少 SAM/adjacency 的对象数量和跨 job 分配

`planner.hpp` 的每个 `Node` 含一个 `std::unordered_map<int,int>`，桶与 entry
独立分配；clone 还会复制 map。`extend()` 中有 `count()` 后 `operator[]`、
`at()` 的重复查找；query matching 也有 `count()` 后 `find()`。反汇编确认
hash bucket 路径存在整数 `div`，但没有给出其动态周期占比。

`prefill.hpp` 的 adjacency 是 `vector<vector<int>>`，大量低度节点形成小
vector 分配。`Scratch` 虽然在一次 Builder 的递归中复用，仍是 Builder 成员；
`TrainingBatchProgram` 每个 job 都新建并销毁 Builder，**并未跨 job 复用**。

优先候选是 worker-local 可复用 arena/scratch、连续 adjacency、小度数 inline
transition 或适合稀疏转移的 flat 表，以及合并重复查找。不要为每个 SAM 节点
直接展开整个 512-symbol dense 表，否则可能以更大的工作集换掉分配开销。
这些改动需要保证 clone、寿命和索引语义，不在本轮实施。

### 其次：减少 decompose 与 packing 的重复遍历/初始化

decompose 多次遍历同一组件、构造计数排序和 merge-tree ranges；部分 vector
resize 后又覆盖写入。现有 flat arena 已减少了旧版 per-stream 对象，但仍有
工作量和容量管理成本。应该先复用/预分配、再按热点验证是否需要改变算法。

训练构造阶段统计 chunk boundaries；compact pack 再统计 stream 的 chunk
数，多 chunk 分支还重新寻找边界。可以评估一次生成连续的分类/边界描述，
供 packing 直接使用，注意别重新引入大量中间小对象。padding 写入和 per-key
`log(decay)` 也在热路径，但输出布局受 GPU consumer 合约约束。
GCC 的 vectorization report 显示，部分 padding loop 已生成 16-byte SIMD，
并因潜在 aliasing 生成 loop versioning；实际 event loop 则因控制流未向量化。
因此不能笼统声称整个 packing 没有 SIMD，也不能只加一个编译开关解决依赖。

### 数学相关优化最后单独验证

scalar pass 的 `mass`/`dmass`、`logden[i]`/`dlogden[i]` 有 loop-carried
依赖；`exp/log/log1p` 增加延迟并限制直接 SIMD。τ 表也按 job 重复计算。
可以先研究同 generation、相同 τ/长度下的精确表复用。

不要为了性能直接启用 fast-math、改变求和顺序，或把 `log(exp(x))` 换成 x：
下溢、舍入和 τ 梯度相消都可能改变。标量系数改动必须单独通过现有严格数值
诊断。当前高置信度收益优先来自数据结构与生命周期，尚不需要先动数学。

## 6. 工具和产物

- `tools/profile_planner_native.cpp`：当前生产 headers 的独立 C++ target。
- `tools/build_planner_cpu_profile.py`：顺序构建基线/插桩版及计数器 wrapper，
  复制 headers 只写到指定 build 目录，保存源码 hashes 和编译命令。
- `tools/count_planner_events.cpp`：非采样计数器 wrapper，记录 raw counts 和
  time_enabled/time_running，继承到子进程 worker。
- `tools/bench_planner_allocator.py`：进程局部 allocator A/B、字节相等检查。

全部原始产物在 ignored `build/planner_uarch_20261023/`：
`phases{512,32}.csv`、`allocator_ab.json`、`counters*.jsonl`、
`byte_checks.json`、`build_manifest.json`、VTune result dirs/logs/hotspots CSV、
`integer_division_sites.json`、`vectorization.txt`。分配/计数器探针与原生 binding
的字节对照覆盖 20,113,376 和 55,374,776 bytes。没有提交生成文件。

在仓库根目录可重建独立目标和合成输入：

```bash
python dism_v3/tools/build_planner_cpu_profile.py \
  --output dism_v3/build/new_planner_profile \
  --geometry-json dism_v3/build/planner_cpu_20261023/results.json
taskset -c 0 dism_v3/build/new_planner_profile/phases \
  dism_v3/build/new_planner_profile/iid512.bin 5 1
```

所有时序是这些本地合成输入的诊断结果。真实 checkpoint 的 labels、τ 分布、
连续六层调度及 pinned/device allocation 可能改变热点排序；未宣称训练端到端
获得相同加速。
