# 归一化优化后的 CPU planner profile（2026-10-24）

当前提交 `ad43351`，归一化前基线 `a4d131e`。结论：**超越函数成本下降后，
相对热点转向 centroid 分解/事件生成、分支和 packing；当前 N2048 主要不是
DRAM 带宽饱和。N8192 的近端缓存和内存等待增加，仍有整体带宽余量。**

本轮没有修改生产实现，没有启动训练或加载 checkpoint。只重放保存的输入。
代码的数学/误差验证见 [归一化重写](PLANNER_NORMALIZATION.md)。

## 测量和计数有效性

- Core Ultra 9 285K，CPU0–7 为 P 核、8–23 为 E 核。单 P 核绑 CPU0，单 E 核
  绑 CPU8；W16 由 OS 调度。保持相同 `-O3 -g -fno-omit-frame-pointer` 编译选项。
- 主要真实输入为 model_022888/PG19 已保存的 `n2048_layer5.bin` 和
  `n8192_layer5.bin`，B4/H12、完整行、all-hard，在 gradient-enabled compact C64
  路径重放。并测 iid32 和 iid512；不是训练整步或六层同时竞争的调度轨迹。
- 独立 C++ 目标含 plan、普通 CPU buffer 分配、pack、释放；不含 Python/GIL、
  pinned allocator、PCIe 或 GPU。所有负载顺序执行，计数、软件采样和阶段插桩分开。
- VTune 2026.6 新旧软件采样各 10 秒；阶段插桩各 3 秒；每组独立 PMU 计数
  约 3 秒。没有用采样运行的 wall time 代替上一轮无插桩 A/B 吞吐结果。
- `perf_event_open` 继承到所有 worker，核心事件排除 kernel/hypervisor。
  P/E 使用各自 PMU type 和 Intel 对应原始事件编码，禁止把 P 核分子除以
  P+E 核总指令。W1 另用 Linux 通用事件作独立交叉验证。
- 这次使用**非采样、窄集合的可编程 Top-down 事件**，不复用上一轮失效的
  VTune 广事件配置或 Linux synthetic perf-metrics 百分比。P 核 slots/cycles=8，
  四类合计约 100%；E 核按 8*cycles 计算，四类合计约 100.3–100.4%，保留原值，
  不强制归一成 100%。E 核比例是近似分类，不能解释百分之零点几的差异。
- 单 P 核 running/enabled 约 .9998，单 E 核约 1。W16 混合核计数中，各 PMU
  的运行时间只覆盖相应核类型，比例低于 1；P/E 周期计数的运行比例相加约为 1。
  不按各自运行比例把原始计数外推到全 CPU 时间。固定 affinity 对照、同组时间
  一致性和每 plan 指令量一起用于校验，不能把混合核的比例直接当成 multiplex。
- N2048 单 P 核，新版 raw 指令计数为 **378.409M/plan**，Linux 通用事件为
  **378.404M/plan**，差约 .0013%；旧版为 429.908M/plan，新版下降约 **12.0%**。
  W16 的 P+E 指令为 376.961M/plan，接近单核；不同 job/tau 缓存命中和调度使
  并行版本不要求完全相等。没有再次出现此前约 15 倍的指令计数矛盾。
- 事件定义依据 Intel 的 [P-core](https://perfmon-events.intel.com/platforms/arrowlake/core-events/p-core/)
  和 [E-core](https://perfmon-events.intel.com/platforms/arrowlake/core-events/e-core/)
  文档；实际使用的 ARL JSON 为 2026-08-21、v1.21，已保存 SHA256。

## 热点转移

N2048 真实最后一个 DISM 层、W16，VTune 函数自耗采样：

| 热点 | 旧版 CPU 占比 | 新版 CPU 占比 | 旧→新累计 CPU ms/plan |
| --- | ---: | ---: | ---: |
| exp | 15.17% | 10.27% | 8.70 → 5.39 |
| log1p | 5.46% | 未采到 | 3.13 → 约 0 |
| decompose（同名实例合计） | 11.84% | 12.47% | 6.79 → 6.54 |
| compact pack lambda | 10.75% | 11.97% | 6.17 → 6.28 |
| finish 自身 | 4.44% | 7.89% | 2.55 → 4.14 |
| merge_into | 3.66% | 3.91% | 2.10 → 2.05 |

**占比上升不等于绝对耗时上升。** decompose 每 plan 自耗大体不变，exp/log1p
减少使其占比上升；finish 新增缩放和数组运算，所以自身采样时间增加，但它
调用的 libm 总成本明显下降。函数自耗不含独立归属的 STL、emit、merge 等，
不能把 decompose 自耗直接解释为整个分解阶段占比。

iid32/W1 上，exp 从 14.43% 降到 8.13%，log1p 从 6.12% 降至本轮未采到，
compact pack 约从 12.34% 到 13.74%。未采到不代表所有输入绝对没有 log1p 调用。

新版单 P 核的粗阶段计时：

| 阶段 | 真实 N2048 ms | iid32 ms |
| --- | ---: | ---: |
| tau 表 | 2.32 | .007 |
| SAM | 3.50 | 5.16 |
| adjacency / query matching | .38 / 1.10 | .61 / 3.98 |
| decompose/emit | **20.54** | **48.46** |
| scalar denominator | **4.96** | **16.09** |
| coefficients | 3.65 | 20.96 |
| boundary count | 1.45 | 5.70 |
| compact pack | 5.90 | 16.47 |
| 总计 | 44.26 | 120.15 |

旧阶段记录分别为 scalar 8.44 / 28.02 ms，decompose 20.40 / 49.28 ms。
这些旧阶段时间来自前一轮，未作为同轮严格 A/B；本轮新旧软件采样和指令计数
独立支持归一化成本下降的结论。新版 decompose/emit 已占真实输入总时间约 46%。

## 前端与分支压力

下表为原始事件计数得到的 Top-down slot 比例，**不是墙钟时间百分比**。

| 输入 / 并行 | 核类型 | Retiring | Frontend bound | Bad speculation | Backend bound |
| --- | --- | ---: | ---: | ---: | ---: |
| 旧 N2048 / W1 | P | 29.55% | 20.68% | 28.56% | 21.20% |
| 新 N2048 / W1 | P | 27.94% | 22.22% | 31.00% | 18.83% |
| 新 N2048 / W1 | E | 27.84% | 18.57% | 37.03% | 16.92% |
| 旧 N2048 / W16 | P | 29.37% | 21.04% | 28.43% | 21.14% |
| 旧 N2048 / W16 | E | 27.88% | 16.06% | 34.62% | 21.77% |
| 新 N2048 / W16 | P | 27.70% | 22.35% | 30.92% | 19.02% |
| 新 N2048 / W16 | E | 27.29% | 18.55% | 37.23% | 17.30% |
| 新 N8192 / W16 | P | 26.60% | 19.16% | 27.70% | 26.54% |
| 新 N8192 / W16 | E | 25.27% | 16.07% | 33.05% | 25.94% |

新版确实存在前端供给和分支压力，但**没有证据表明主要是指令缓存容量不够**：

- N2048/W16，P 核取指 tag stall 约 .70% cycles，data stall 约 .16% cycles。
  E 核由 I-cache 解释的前端空槽仅约 .14% 总 slots。
- E 核 frontend bandwidth 约 16.9% slots，其中 decode 约 14.0%；frontend
  latency 约 1.6%。按该事件模型，主要是供给/解码吞吐，而非代码 cache miss。
- P 核无前端供给持续至少两周期的 fetch-latency 事件约 7.45% cycles；分支或
  machine clear 后等待重新供给约 9.38% cycles。这些周期事件可能重叠，不能
  与 Top-down 分类直接相加。尚未进一步分离 P 核 DSB/MITE/解码的各自贡献。
- 新 N2048/W16，P/E branch miss 分别为 **6.76 / 6.46 MPKI**，约占退休分支
  的 **4.16% / 3.97%**。单 P 核旧→新 branch miss 为 5.86→6.71 MPKI，但每
  plan 的 miss 约 2.518M→2.540M，仅增约 .9%；比例上升主要伴随总指令减少。

另用 VTune 在单 P 核精确 branch-miss 采样 10 秒：约 **30.3%** 的 miss 归属
到 decompose，**9.4%** 到 merge_into，**15.2%** 到 finish，**10.8%** 到
PrefillMatcher，**7.3%** 到 compact pack。这是 miss 事件分布，不是各函数
浪费的周期百分比；但足以把分支压力定位到实际 planner 路径，而非仅归咎 libm。

## 数据缓存和内存带宽

W16 核心侧指标，各自用本核类型指令/周期作分母：

| 指标 | N2048 P | N2048 E | N8192 P | N8192 E |
| --- | ---: | ---: | ---: | ---: |
| L2 retired-load miss / 千指令 | .0144 | .0287 | .0265 | .0851 |
| P: L3 retired-load miss；E: local-DRAM load / 千指令 | .0013 | .0028 | .0111 | .0269 |
| P: DRAM load stall；E: LLC-miss load stall / cycles | .25% | .50% | **3.66%** | **5.27%** |

P/E 的 retired-load / retired-uop 事件定义并不完全相同，因此横向看趋势，不把
两者当同一事件合并求平均。低 retired-load miss 不代表总流量低：预取、RFO、
写回和输出写入都不能从该计数直接还原。

近端压力仍然存在。P 核 N2048 约 12.7% cycles 是“有 load 在途、没有 L1 miss、
没有执行”的 MEMORY_STALLS.L1；N8192 约 11.6%。E 核 L1-miss load stall 从
6.4% 增至 13.8%，store-buffer-full 从约 1.0% 增至 2.6%。这些指标说明不能把
“DRAM 不饱和”误写成“访存没有成本”，但也不足以单独证明 L1/L2 带宽饱和。

直接读取两组 `uncore_imc_free_running` 的 read/write，按 sysfs 单位转换为
64 bytes/count；这是**整机内存控制器流量**，不是只归属 planner 的流量。
无 GPU 任务；同期空载背景约 .54 GB/s。以下为未扣背景的十进制 GB/s：

| 负载 | 读 | 写 | 合计 |
| --- | ---: | ---: | ---: |
| 空载 | .38 | .16 | .54 |
| 旧 N2048 / W16 | 6.12 | 4.65 | 10.77 |
| 新 N2048 / W16 | 7.38 | 5.47 | **12.85** |
| 新 N8192 / W16 | 17.26 | 12.95 | **30.20** |
| 本机 16-thread 流式 triad 对照 | 47.99 | 17.15 | **65.14** |

流式对照使用三个各 256 MiB 数组，运行约 6 秒；其逻辑读写吞吐为 51.21 GB/s，
包含额外内存事务的 IMC 物理流量为 65.14 GB/s。这不是理论峰值或完整 STREAM
认证测试，而是同一采集链路、同机 16 线程可达到的参考下界。

因此 N2048/N8192 的总流量约为该参考的 **20% / 46%**。结合核心侧 DRAM 等待，
没有证据认为现有输入已撞上整机 DRAM 带宽上限。N8192 的后端占比和内存等待
明显增长，未来更长序列、更多在途计划仍可能改变瓶颈。平均带宽也不能排除
某些短阶段或缓存层的局部带宽限制。

## 下一步优先级

1. 优先评估 centroid/merge 的工作量与分支：前一轮因果剪枝、减少邻接扫描，
   再到按有效事件成本构建合并树。不要仅因 branch miss 高就把所有分支改为
   无条件计算；那可能增加指令和 cache 写入，需继续 A/B。
2. 减少 Program 中间物化、事件遍历和 packing 的复制/转换，特别关注长序列。
   这会同时降低指令和流量，但本轮没有测得这种重构的收益。
3. E 核解码供给压力值得通过更紧凑的循环、减少每事件管理指令来处理；P 核
   尚需更细的供给来源分析。当前不支持优先扩大指令缓存或盲目软件预取。
4. 多 tau 表复用仍有独立空间：真实 N2048 单 P 核约 2.32 ms；exp 仍有约 10%
   软件自耗，但已经不再是唯一优先方向。

## 产物和限制

产物全部在 ignored `dism_v3/build/planner_postnorm_profile/`：

- `build_manifest.json`、`collection_manifest.json`：源文件/二进制/事件表 SHA256。
- `sw_{old,new}_{real,iid32}/`、`hotspots_*.tsv`：VTune 软件热点原件和导出。
- `branch_real_p/`、`branch_functions.tsv`：branch-miss 硬件采样。
- `count_raw.cpp`、`*.events`、`counters.json`、各配置 JSONL：原始非采样计数、
  enabled/running 时间和 workload 完成次数。`generic_validation.jsonl` 为独立对照。
- `prepare_events.py`、`collect.py`：事件生成和串行采集命令；使用已保存的 Intel JSON。
- `stream.cpp`、`stream_w16.jsonl`、`idle_bandwidth.jsonl`：带宽对照与空载流量。
- `phases_*.csv`：阶段计时和分配计数。

硬件指标是短时间诊断，不是多次置信区间测量。没有锁 CPU 频率，没有证明所有
labels 分布、N 或训练并发状态都有相同瓶颈。特别是 Top-down、stall cycles、
函数 CPU 自耗和 IMC 流量是不同口径，不相加换算训练耗时。

使用会话中已有授权临时将 `kernel.perf_event_paranoid` 从 3 调至 0；已通过
退出 trap 恢复并读回 **3**。没有安装系统工具、调整训练 affinity 或改模型状态。
