# CPU planner 算法再评估（2026-10-24）

结论：**仍有优化空间，但上一轮消除小对象分配后的收益已不能照搬。**
本轮两个保持事件顺序和浮点计算不变的隔离候选，在保存的真实 labels/tau
上将 W16 CPU plan+pack 耗时降低约 4–12%；这不是训练吞吐提升。
更大的候选方向是减少事件复制和中间 Program 物化，而非继续更换 allocator。

生产 planner 没有修改。本轮只修复 profiling 工具的头文件依赖并记录评估。
候选代码、二进制、完整测量和 VTune 数据全部位于 ignored
`dism_v3/build/planner_reassessment/`。没有加载或改写 checkpoint，没有启动训练、
下载数据或执行 GPU 工作。原有用户文件保持不变。

## 当前实现和测量口径

源码基于 `2e98d44` 的当前 worktree。已包含 flat SAM、CSR、worker-local
Builder、tau 表复用、边界复用、pack 优先调度，以及 `finish()` 中 key 的
exp 结果跨两次 pass 复用。不能把已经存在的这些改进再次作为新建议。

Core Ultra 9 285K，GCC `-O3 -g -fno-omit-frame-pointer -std=c++17 -pthread`，
无 fast-math。使用 conda blkw 运行诊断脚本，独立 C++ target 直接包含当前
worktree headers，不导入其他 worktree 的 Python/native 模块。
W1 绑 CPU0，W16 沿用 OS 调度；不改变生产 affinity 或 worker 默认值。

无插桩计时包含构造计划、普通 CPU 输出分配、compact C64 packing 和释放；
不含 Python/GIL、pinned allocator、PCIe、GPU 等待或训练 deadline。每个进程
从第一次调用起计时，包含冷启动；每轮至少运行指定时长，不是逐调用延迟分位数。
配置顺序用固定 seed 打乱，各三轮，报告各进程平均耗时的中位数。CPU 频率未锁定，
小幅差异仍应视为噪声敏感，尤其不能把约 1% 的差异当作确定收益。

合成输入：B8/H12/N2048，沿用保存的 26 个 packed segments，299 个
segment/head jobs，固定 row/head hard flags，tau=.5；iid512/32/8 逐渐增加匹配密度。
每轮至少 1 秒。hard flags 没有改成按文档采样。

真实输入：复用 `build/prefill_planner_compare/inputs/manifest.json` 中已捕获的
model_022888、PG19 prefill labels/tau，并核对文件 SHA256。B4/H12，四个完整行，
全部 hard；N2048 的六个 DISM 层，以及 N8192 的第一个和最后一个 DISM 层。
每轮至少 .6 秒。这些是**真实 inference 捕获输入，在 training/gradient-enabled
compact planner 路径重放**，不是实际 packed training batch，也不是 inference
API 端到端测量。没有重新捕获或访问 checkpoint 内容。

## 当前热点

单 P 核阶段插桩，单位为每批累计 ms；插桩不参与 A/B 性能结论。

| 阶段 | iid512 | iid32 | iid8 |
| --- | ---: | ---: | ---: |
| SAM | 4.49 | 5.33 | 7.60 |
| adjacency | .69 | .82 | .72 |
| query matching | 3.59 | 4.06 | 4.58 |
| decompose/emit | 19.39 | 49.28 | 73.75 |
| scalar denominator | 4.35 | 28.02 | 43.66 |
| coefficients | 1.99 | 21.39 | 33.65 |
| boundary count | .76 | 5.72 | 9.80 |
| compact pack | 3.96 | 16.70 | 29.08 |

事件数分别为 201,476 / 1,185,146 / 1,996,014；最终 metadata 为
20.11 / 55.37 / 83.47 MB（十进制）。事件越密集，数学 pass、边界和 packing
成本越明显。iid512 的 decompose/emit 约占总耗时 48%，iid32/8 约为 36–37%。
单 tau 稳态下 tau 表预计算不足 .01 ms，不能由此推断不同 head tau 的成本也如此低。

VTune 2026.6，iid32/W1，12 秒软件热点采样：`exp` 自耗约 **13.2%**，
`log1p` **5.9%**，`log` **1.0%**，compact pack lambda **12.8%**。
函数自耗与高层阶段不是同一口径：例如 pack 内的 log 归入 libm，decompose
的 STL、emit、merge 等也有单独归属，不能只看 `decompose` 符号的自耗。

使用 user-mode sampling，不需要 sudo，没有修改 perf_event_paranoid（仍为 3）。
本轮没有有效 PMU/带宽/Top-down 测量，不能宣称 memory-bound 或给出 stall
比例，也不引用上一轮已经被判定异常的 Top-down 百分比。

## 两个保持输出不变的候选

**A：在 decompose 入口提前判定因果空集。** query/key 列表始终按 token
下标排序，若 `ks[kr.b] > qs[qr.e-1]`，该组件和全部子组件不可能有 `key<=query`
的配对。当前只在后面的 cross/emit 剪枝，前面的树扫描、centroid、分组等仍执行。
将同样条件提升到递归入口能直接省去这些工作。严格用 `>`，不能错误地排除
同一位置的 key/query。各兄弟组件已由被移除的 centroid 分隔，跳过无贡献组件
不会改变其他组件的事件顺序。

**B：在子树 size 的逆序归约中同时计算最大子树。** 当前先计算 size，再为
每个节点扫描邻接表求最大子树。候选把最大子树的归约合入前一次逆序循环，
避免第二次邻接表遍历；仍按原 `order` 顺序选择 centroid，保留 `<best` 的
平局规则。实验复用 cursor scratch，并暂借 owner 作为 order 索引，随后原流程
重新赋予 owner。额外 scratch 初始化/索引写入会抵消部分收益，因此需要实测。

| 输入 | workers | 当前 ms | B ms | A ms | A+B ms | A+B 耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iid512 | 1 | 40.740 | 39.922 | 38.089 | 37.336 | 8.4% |
| iid512 | 16 | 5.418 | 5.159 | 5.067 | 4.962 | 8.4% |
| iid32 | 1 | 134.791 | 132.791 | 133.029 | 130.489 | 3.2% |
| iid32 | 16 | 13.343 | 13.626 | 13.048 | 13.202 | 1.1% |
| iid8 | 1 | 212.372 | 191.976 | 208.616 | 189.047 | 11.0% |
| iid8 | 16 | 21.128 | 21.301 | 20.872 | 20.000 | 5.3% |

B 单独在 iid32/8 W16 略退步，组合也不总优于 A。不能认为每个局部少一次遍历
都必然加速多线程任务。

真实捕获输入，W16（层编号为零基 DISM 层序号）：

| 输入 | 当前 ms | A ms | A+B ms | A+B 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| n2048_layer0 | 1.638 | 1.555 | 1.440 | 12.1% |
| n2048_layer1 | 2.686 | 2.596 | 2.486 | 7.5% |
| n2048_layer2 | 2.724 | 2.673 | 2.576 | 5.4% |
| n2048_layer3 | 5.308 | 5.273 | 5.073 | 4.4% |
| n2048_layer4 | 6.558 | 6.333 | 6.146 | 6.3% |
| n2048_layer5 | 6.439 | 6.207 | 6.134 | 4.7% |
| n8192_layer0 | 6.830 | 6.740 | 6.167 | 9.7% |
| n8192_layer5 | 28.929 | 28.046 | 26.745 | 7.5% |

真实 n2048_layer5 的补充 W1 插桩：decompose/emit 20.40 ms，scalar 8.44 ms，
coefficients 3.62 ms，pack 6.01 ms，tau 预计算 **2.32 ms / 总 47.76 ms**。
这说明多 tau 下单一 `cached_tau` 会反复失效；该测量中约 4.8% 的时间值得
通过有界多 tau 表或同批共享表进一步消融，不能外推为 W16 的同等百分比。
该真实输入 W16 的另一轮 VTune 软件采样（10 秒）中，exp 自耗 **16.3%**，
log1p **5.3%**，compact pack lambda **9.3%**，支持热点判断不只适用于 iid。

## 后续优化优先级和算法边界

1. **先考虑 A，再独立评估 B。** A 的证明和改动都小，避免的是确实没有输出
   的递归；B 保留 centroid/tie-break，不改变树分解结构。当前实验足以支持
   开发候选，但三轮 CPU 计时不足以宣称生产训练收益。落地时仍需训练 planner、
   graph 生命周期和多 coordinator 的正式回归。
2. **多 tau 的精确表复用。** 当前 worker 只记一个 tau。同批不同 head、连续层
   交错时可能不断重算。可按完全相等的 tau 和有效长度建立有界缓存，或为同批
   构造只读表供 jobs 共享。必须保留原递推、有效长度、异常恢复和内存上界；
   不按近似 tau 合并，不无限累积每步变化的 tau。需要单独测缓存收益和调度成本。
3. **减少中间表示和遍历，而非直接删掉数学。** 当前顺序是 stream arenas →
   scalar normalization → FP64 Program → boundaries → compact FP32 buffers。
   生产路径有每事件 FP64 数组写入，再由 pack 读回、转换、重排和 padding。
   若先由整数事件拓扑确定大小/边界，可研究 coefficient pass 直接写最终布局。
   全部 stream 的 logden/dlogden 完成后才能生成归一化 query coefficients，
   因而不能简单把两个数学 pass 合成单次顺序扫描。还必须解决 output 分配时点、
   plan/pack 优先调度、Program 所有权和错误传播。当前阶段数据只证明有成本，
   **没有测得这个重构的收益**。
4. **更激进地减少事件量。** 当前 Huffman 式合并用组件节点数作权重，实际代价
   则受 query/key 数、因果重叠和 bucket padding 影响。可研究按有效事件工作量
   的合并策略、避免无用 merge 尾部，或不同分解表示。centroid 树深度有对数界，
   但不代表每个 stream 只有对数长度；还要计算 merge 复制量、事件 E 和 padded
   slots P。保持现有 GPU 布局时至少需要写出 P 个槽位，不能只优化 SAM 的近线性
   部分就期待整个 planner 同比例变快。改变合并顺序会改变事件顺序和 FP64
   求和路径，也可能改变 FP32 chunking 和 GPU 归约，因此这是需要严格数值与
   GPU 验证的算法分支，不能以字节等价的小优化名义合入。

不建议重复尝试已经失败的 exp(0)/log(1) 小特判作为首要路线，不启用 fast-math，
也不把 `log(exp(x))` 直接换成 x；下溢、舍入和 tau 梯度仍受原有数值合同约束。
CPU 更快只有在计划落后于 GPU deadline 或 CPU 吞吐成为限制时才能改善训练步耗时。
本轮没有测 route/fallback、H2D、GPU 或整步训练时间。

## 验证、产物和复现

- A、B、A+B 在三种 iid、W1/W16 的所有轮次，完整 compact 输出 SHA256 一致。
  A、A+B 在八个真实输入、W16 的所有轮次也一致。
- A+B 通过现有 `tests/native/test_prefill_workspace.cpp` 的 **1,400 组**比较：
  当前 production Builder 作为独立命名空间基线，完整 Program 和 C16/32/64
  chunk 逐字节一致。覆盖正负/零及近零 tau、变长、reset、gradients on/off、
  reference LCA、重复/周期/负值及 INT_MIN token。**ASan+UBSan 通过**。
- 两份相同头文件使用 `#pragma once` 时，GCC 可能按相同内容/时间戳跳过第二份；
  隔离 matcher/planner 副本添加无语义注释，确保 dual-namespace checker 实际
  编译两份实现。没有改生产头文件。
- 未执行独立数值 oracle、GPU 回归、CUDA sanitizer 或新的并发 TSan；不把
  CPU 字节对照宣称为覆盖了这些测试。没有 CUDA 改动，无新增 CALL/spill 检查。
- 旧 `build_planner_cpu_profile.py` 漏掉拆出的 `planning_workers.hpp`，导致
  instrumented target 无法构建。本轮补上隔离复制及 manifest hash，重建成功。

构建并复测当前源码（仓库根目录）：

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/build_planner_cpu_profile.py \
  --output dism_v3/build/planner_reassessment_reproduce \
  --geometry-json dism_v3/build/planner_cpu_20261023/results.json
taskset -c 0 dism_v3/build/planner_reassessment_reproduce/phases \
  dism_v3/build/planner_reassessment_reproduce/iid32.bin 3 1
```

原始产物在 `build/planner_reassessment/`：

- `build_manifest.json`：当前 headers/source hashes 与编译命令。
- `experiment.py`、`centroid/`、`causal/`、`combined/`：完整隔离候选及生成脚本。
- `measure.py`、`measurements.json`：合成 A/B、完整输出 hashes、事件和字节数。
- `real_measure.py`、`real_measurements.json`：真实输入重放及输出 hashes。
- `phases_*.csv`、`phases_real_layer5.csv`：阶段时间与分配计数。
- `vtune_iid32/`、`vtune_real_layer5_w16/`、对应日志和 hotspots CSV：软件采样原件。
- `check_combined`、`check_combined.log`：ASan/UBSan dual-namespace 回归。

这些临时候选尚未接入生产；本次结论是“存在已验证的小幅优化，以及需要进一步
实验的结构性方向”，不是承诺再获得上一轮 20–40% 的并行 CPU 收益。
