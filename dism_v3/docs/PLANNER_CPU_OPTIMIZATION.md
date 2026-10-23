# CPU planner 与线程池优化（2026-10-23）

实现最初在 `opt/planner-cpu-20261023` / `dism-core-planner-opt` 隔离验证，
随后合入主开发分支 `core/mxfp8-training`，基线为 `54e67e3`。
没有修改运行中的评估、训练或 checkpoint。
前置诊断见 [CPU 瓶颈](PLANNER_CPU_BOTTLENECK.md) 和
[阶段/微架构 profile](PLANNER_CPU_PROFILE.md)。

## 实现和边界

1. `prefill_matcher.hpp`：offline SAM 使用连续 Node/Edge 数组和一个 flat hash
   table，取消逐节点 unordered_map。低出度节点先检查内联链头，clone 复制
   transition 值；扩容时不保留 vector 引用。节点创建顺序、suffix link 和
   token 的完整有符号 int32 取值保持一致。online decode Planner 不变。
2. `prefill.hpp`：suffix-link 邻接表改为 CSR，严格保留原来逐 v 插入的邻居
   顺序（包括父节点编号晚于子节点的 clone 情况），从而保留 centroid 和事件
   顺序。Builder 增加可复用的工作区，训练 worker 通过 thread_local 保留它；
   q/k 直接借用输入，reset=nullptr 表示当前 document 内没有 reset。
3. 长度相关 logweights/derivatives 只在 tau 完全相等时复用，长度增加时继续
   原递推。不同 tau 重算；没有近似、fast-math、求和重排或容差变化。缓存有效
   长度单独记录，异常后的下一次 build 不把未计算的表项当作有效数据。
4. `prefill_training.hpp`：第一次边界发现保存 chunk ends/stream chunk offsets，
   sizing 与 classic/compact packing 共用，不再重复扫描 rows/decay。Program
   输出仍独立拥有数组，不借用 worker scratch。
5. 线程池取消整次 run 的 invocation mutex；每次提交一个 Batch，worker 按
   job 领取任务。pack 优先，阶段内 FIFO。任务执行、完成计数和等待者生命周期
   由同一个队列 mutex 协调；成功提交后，无论 worker 是否抛异常都排空后返回。
   队列插入失败时没有暴露任何 job。不增加 worker 数、不改变 affinity。
6. Python compact prepare 不再额外构造一套随即弃用的 NumPy buffer views。

并行线程数仍有界。并发 plan 的个数继续由 graph/eager coordinator 约束；这
不是无限任务队列的内存预算机制，也不是按 CUDA backward deadline 排序的
EDF scheduler。pack 优先让已经完成的 plan 尽早进入上传，避免被其他层的
完整 plan 调用阻塞。worker scratch 按处理过的最大 document 保留容量，在线程
退出时销毁；这用常驻工作区换取分配/释放开销，需要与临时 Program 内存区分。

没有修改 CUDA kernel、graph flag 协议、D2H/H2D、ready/fallback 语义、数学、
布局、默认 worker 数或 allocator。输出 metadata 大小保持一致。

## 测量口径

- Core Ultra 9 285K（8P+16E），B8/H12/N2048，保存的真实 26 个 packed segments，
  h=.95，299 个 document/head jobs。labels 为合成 iid512/iid32；不是训练轨迹。
- 单 worker 绑 CPU0；W16 沿用 OS 调度。生产下游评估仍运行，没有暂停它。
- 独立 C++ target 直接包含当前 headers，普通 CPU 输出内存；包含 plan、分配、
  pack、释放，不含 Python/GIL、pinned allocation、PCIe 或 CUDA 同步。
- 基线/候选交替，各配置三次、每次至少一秒，报告耗时中位数。每次输出 SHA
  相同，并对完整 buffer 做比较；profile 单独执行，不把插桩时间当性能数据。
- tau=.5 与逐 head 不同 tau 分开测：后者为 `.15 + .073*head`，避免把统一
  tau 的特殊缓存命中误当作一般训练收益。真实训练 tau 的分布仍需后续采样。

| 输入 | workers | tau | 基线 ms | 优化 ms | 耗时降低 |
| --- | ---: | --- | ---: | ---: | ---: |
| iid512 | 1 | 全部 .5 | 95.64 | 41.72 | 56.4% |
| iid512 | 16 | 全部 .5 | 9.08 | 5.37 | 40.8% |
| iid32 | 1 | 全部 .5 | 196.93 | 141.40 | 28.2% |
| iid32 | 16 | 全部 .5 | 18.66 | 14.34 | 23.1% |
| iid512 | 1 | 逐 head 不同 | 94.15 | 45.75 | 51.4% |
| iid512 | 16 | 逐 head 不同 | 8.82 | 5.54 | 37.2% |
| iid32 | 1 | 逐 head 不同 | 192.92 | 144.27 | 25.2% |
| iid32 | 16 | 逐 head 不同 | 18.19 | 14.43 | 20.7% |

初步更密集 iid8/W16 为 25.85→21.83 ms；该轮在最终 reserve/异常缓存保护前，
不与最终表混为同一二进制。高密度输入的剩余主要成本是事件数、centroid
分解、标量归一化及系数生成，不能承诺所有输入都有稀疏输入的加速比。

## 调度消融与内存

六个 coordinator 以 3 ms 间隔提交，复用相同的输入/flags，W16，共 30 个有效
round（两种反向配置顺序、各自三轮 warmup）。以下为六个请求全部完成的中位
时间；这是有控制的 CPU 排队模型，不是真实逐层 labels 或 backward deadline。

| 输入 | 基线 | 仅 planner 优化、原串行调度 | planner + 新调度 |
| --- | ---: | ---: | ---: |
| iid512 | 49.32 ms | 29.93 ms | **23.15 ms** |
| iid32 | 95.58 ms | 75.66 ms | **66.23 ms** |

新调度相对同算法/旧调度再降低 22.7% / 12.5% makespan。完成时间分布也更
有序：iid512 的六个请求中位完成时刻约为 5.01、8.89、12.56、16.11、19.65、
23.15 ms；iid32 为 13.54、24.19、34.80、45.32、56.02、66.23 ms。
这消除了 invocation mutex 竞争造成的随机整层超越，并允许 plan 尾部与其他
请求、pack 衔接；它不保证后提交而 backward 更早的层有最优 deadline。

单独 W16、一秒循环进程的峰值 RSS：iid512 从 40.65→43.22 MiB，iid32 从
110.90→113.56 MiB。这只是该输入的整个独立进程 RSS，不是生产六层同时在途
的 peak，也不是固定 workspace 上界；更长序列的保留容量需重新测量。

## 分配和阶段 profile

单 P 核、tau=.5 的粗粒度插桩。表中为每个 B8 plan 的 worker 累计 ms。
旧版来自前置诊断，具体 wall time 受主机会话和频率影响；新旧直接 A/B 以上表为准。

| 阶段 | iid512 旧→新 ms | iid32 旧→新 ms |
| --- | ---: | ---: |
| tau 预计算 | 4.76→.005 | 4.59→.008 |
| SAM 构建 | 24.40→4.50 | 25.13→5.36 |
| adjacency | 7.57→.66 | 8.48→.68 |
| query matching | 2.64→3.59 | 5.66→4.06 |
| decompose/emit | 23.66→19.21 | 56.69→48.89 |
| scalar denominator | 4.40→4.33 | 27.35→27.91 |
| coefficients | 2.03→1.96 | 12.32→21.41 |
| Builder 销毁 | 17.99→约 0 | 18.22→约 0 |
| boundary count | .66→.76 | 4.68→5.70 |
| compact pack | 4.82→5.10 | 19.21→16.25 |

插桩覆盖范围内每个 plan 的 new/new[] 从约 **109.1万/111.8万** 降为约
**3,036/3,056**，下降约 99.7%。它不包含所有进程级 malloc。Builder 的销毁
移到 worker 退出，不是内存凭空消失；tau=.5 的稳态缓存效果也不能套用到
不同 head 的 tau。CSR/flat SAM 本身就取消了原来的约 103 万次零散分配。

没有隐去局部退步：稀疏 query matching 和稠密 coefficients 的单阶段计时
没有一起改善；arena 高水位、内存布局/分配行为以及粗粒度采样会影响这些
阶段。端到端收益已由无插桩 A/B 确认，但这些局部变化尚不能独立定因。

## 非采样 PMU 对照

`count_planner_events.cpp` 对子进程及 worker 计数，CPU0/W1，排除 kernel。
四次运行的 running/enabled 均 >= .9997；不使用此前失效的 VTune Top-down
百分比。临时调整的 `kernel.perf_event_paranoid` 已恢复为 3。

| 每计划指标 | iid512 旧→新 | iid32 旧→新 |
| --- | ---: | ---: |
| 退休指令 | 10.67亿→4.71亿 | 17.13亿→11.06亿 |
| IPC | 3.09→3.27 | 2.55→2.53 |
| branch miss 次数 | 264.7万→130.3万 | 857.8万→641.9万 |
| L1D read miss 次数 | 158.3万→5.84万 | 205.7万→24.10万 |
| LLC read miss / 千指令 | .0116→.0065 | .0115→.0094 |

主要改变是指令量与近端读 miss 次数下降，不是 IPC 大幅提高。分支 miss 的
绝对次数下降，即使分支 miss 比例可能因总分支数下降而上升。没有证据把
当前瓶颈描述为 DRAM 读带宽饱和；这些指标也不能排除 store/RFO 或依赖链。

## 正确性和限制

- 1,400 组 Builder 比较：基线 header 独立命名空间，完整 Program 与 C16/32/64
  chunks 逐字节相等；覆盖变长、正负/零 tau、同 tau 复用与增长、reset、
  inference/training、reference LCA、负值/INT_MIN token。ASan+UBSan 通过。
- 84 组独立 SymbolicPlanner 数值 oracle 通过，沿用原来的 rtol=2e-9、
  atol=2e-10；包含 random/repeat/mismatch/periodic，N1/65/513。
- 96 组六 coordinator 并发 classic/compact 打包与独立 document plan 逐
  buffer 字节相等；包含不同 chunk 宽度、ragged document、空 hard、每行 tau。
- 线程池 100,000 次小任务、并发提交、跨 invocation 异常排空与恢复均通过
  TSan。新增测试证明一个调用中执行的 job 不阻塞其他调用向空闲 worker 提交。
- 大输入 iid512/iid32 统一/不同 tau 的所有性能对照，完整输出均一致。
- 隔离目录的 prefill native 用版本 `inference/build.py` 构建，JIT cache 也隔离。
  未导入其他 worktree 的 flash_dism/cu_flash_dism。
- CPU 诊断阶段未构建 cu_flash_dism，CPU checker 直接加载本目录 prefill 模块
  和独立 oracle，并确认 CUDA 未初始化。GPU 空闲后的补充验证见下节；没有
  替代、屏蔽或改动 GPU 测试。没有 CUDA 源码变更，无新增 CALL/spill 检查。

真实 checkpoint labels/tau 的逐层 plan/pack/H2D/flag 时间仍未测量。下节的
合成训练吞吐/ready 率不能外推为所有实际训练状态都能消除 deadline miss。

## 复现与产物

生成文件均在 ignored `dism_v3/build/planner_opt/`，不入 Git：

- `baseline_headers/`：54e67e3 原始 csrc snapshot；`baseline`：旧版无插桩二进制。
- `final_profile/build_manifest.json`：最终源码 SHA256 和完整编译命令。
- `validated_final_bench.json`、`pmu_*.txt`、`final_phases_iid*.txt`：A/B 和 profile。
- `final_schedule_bench.json`、`rss.json`：调度消融和峰值 RSS。
- `identity_bench.json`：未采纳的实验；仅特判 exp(0)/log(1)，W16/iid32 无稳定
  收益，未带入实现。未改 libm、编译浮点模式或默认 allocator。

CPU native 的推荐构建/检查（从仓库根目录执行）：

```bash
CUDA_VISIBLE_DEVICES='' MAX_JOBS=2 \
TORCH_EXTENSIONS_DIR="$PWD/dism_v3/build/torch_extensions" \
/home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/python/flash_dism/inference/build.py --component prefill \
  --output dism_v3/build/planner_opt/_dism_prefill.so
CUDA_VISIBLE_DEVICES='' /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/check_planner_cpu_equivalence.py \
  --native dism_v3/build/planner_opt/_dism_prefill.so
/home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/build_planner_cpu_profile.py \
  --output dism_v3/build/planner_opt/recheck
```

C++ 回归：`tests/native/test_prefill_workspace.cpp` 与
`tests/native/test_planning_workers.cpp`，使用 C++17、pthread、本目录 csrc include。
前者可定义 `DISM_BASELINE_PREFILL_HEADER` 为隔离基线 prefill.hpp 的绝对路径；
未定义时使用当前 fresh/reference-LCA Builder 与 reused Builder 对比。
ASan/UBSan 和 TSan 分别构建，不能合在同一个二进制。

## GPU 空闲后的训练吞吐（2026-10-23 晚）

用户确认下游评估结束后测量。使用本 worktree 的 `build.py`、两路有界编译，
构建训练需要的 R32/D64/DV64 CUDA 扩展及 MXFP8 libraries。新旧 CPU planner
在同一 worktree、相同 CUDA libraries、模型代码和配置下执行；旧版 native
从 54e67e3 snapshot 单独构建，并恢复旧版 Python prepare 方法。两种 native
模块的实际路径与 SHA256 都写入结果，避免误用缓存。

配置：123,473,088 参数、`gggddddddggg`、postnorm、MXFP8 FFN/optimizer/CE、
softcap=30、CUDA Graph、16 native workers，8 rows × N2048 × 8 microbatches，
每优化器步 131,072 tokens，changing packed boundaries。使用现有
`bench_training_cuda_graph.py` 的固定 seed、随机模型初始化和合成驻留 batch；
每个新进程 warmup 8 步，测量 32 步。计入 input H2D、前后向、planner drain、
文档归一化、clip 和 optimizer；不含数据读取、capture/编译、评估或保存。
没有加载/改写 checkpoint，也没有恢复或新启动预训练任务。

h=.95 以旧→新、新→旧两种顺序各测一轮，每个版本合计 64 步；h≈.475 各一轮。
GPU clocks 未锁定。新增逐层 route 计数只读取已有的 route tuple，不增加 GPU
同步；两种版本使用同一份测量代码。

| hard probability | 旧 planner ms/step | 新 planner ms/step | 旧 tokens/s | 新 tokens/s | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≈.475 | 835.43 | 836.65 | 156,892 | 156,662 | −0.15%，无明显收益 |
| .95 | 807.49 | **794.29** | 162,320 | **165,019** | **+1.66%** |

h=.95 旧版两轮为 804.88/810.10 ms，新版为 795.01/793.56 ms；两个顺序均显示
小幅收益。不能把这个约 1–2% 的收益当成所有 workload 的固定比例。h≈.475
的新旧所有 1536 次均走 event；差异处于本轮运行波动范围。

h=.95 的 route 变化很明确：旧版总计 e=2602/q=470，fallback **15.30%**；
新版 e=3072/q=0，测量期间 **没有 fallback**。前五个 DISM 层旧版也全部
ready；差异全部在最后一个 DISM（模型第 9 层，零基索引 8）：旧版只赶上
42/512 次（8.20%），新版赶上 512/512 次（100%）。这是该合成工作负载上的
结果，不是对真实 checkpoint 或更长 sequence 的 deadline 保证。

另测 h=.95 关闭 planner：**859.37 ms / 152,522 tokens/s**。新 planner 相对
关闭时吞吐提高 **8.19%**；该控制组只有一轮。与旧 planner 相比收益较小，
符合“其余五层的 CPU 规划原来就已被 GPU 计算隐藏”的观察，但没有据此给
D2H/H2D 或 GPU kernel 单独分配时间比例。

GPU allocated peak 在 h=.95 约 5716.1 MiB，新旧近似相同，reserved peak 均为
13734 MiB。CPU workspace 的保留容量见前面的 RSS 测试，不包含在这些 GPU 数字中。

补充 GPU 回归：`test_native_training_planner.py` + `test_planner_graph.py`
共 **117 passed**；5 个 `strict_gradient` 项未选择，原有 BF16 tau 数值限制
不变。没有运行 CUDA sanitizer，也没有宣称该验证覆盖所有模型 shape。

全部七组日志、命令、模块身份及汇总在 ignored
`dism_v3/build/planner_throughput/`：`results.json`、`summary.json`、`run_*.log`、
`run_suite.py`；回归日志为 `gpu_tests.log`。测量结束后进程已退出，GPU 已释放。
