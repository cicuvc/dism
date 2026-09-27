# 前向摘要 NCU profile（2026-09-08）

## 结论

当前摘要并未达到compute/bandwidth吞吐上限，而是**依赖延迟与控制流等待未被流水隐藏**。
按用户最新要求，优先恢复persistent任务调度、下一任务Q异步预取与跨warpgroup计算重叠，
同时优化key元数据load-use链；每SM一个CTA，不把提高occupancy作为目标。
本轮未修改kernel或做优化ablation；下面的优先级是profile支持的候选，不是已验证加速。

## 方法

RTX5090，NCU2025.4.1，sm120a，B64/H4/N1024/D=DV64/V512，scale1、rtau3。
开启`DISM_TILE_LSE=tanh`；实际CUDA embedding先生成输入，不在被采集kernel内。
固定并重放同一行RNG，跳过前10次匹配launch，采第11次`core<64,32,false>`。
grid=(8,256,1)，block=384，每次full集合40 passes。
cache-control=none、clock-control=none，用于保留热缓存/正常运行频率；
NCU会警告不锁频且不清缓存，跨pass存在波动，不能把profile耗时当严格benchmark。

复现（本机硬件counter要求管理员权限，只提升本次profiler进程，不修改驱动策略）：

```bash
sudo env DISM_TILE_LSE=tanh TORCH_EXTENSIONS_DIR=/home/cicuvc/.cache/torch_extensions/py312_cu130 \
  /usr/local/cuda/bin/ncu --set full --kernel-name-base mangled \
  --kernel-name 'regex:.*coreILi64ELi32ELb0.*' --launch-skip 10 --launch-count 1 \
  --cache-control none --clock-control none --export /tmp/dism-summary-tanh-q \
  /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.profile_forward_summary
```

追加`--hard-prob 0`、`--hard-prob 1`或`--direction k_from_q`可复现另外三个采样条件；
每次用不同export路径。未改变前向/反向数学、缓冲、register配置。

## 主场景：q_from_k / hard_prob=.5

| 指标 | 实测 |
|---|---:|
| Duration | 675.584 us |
| SM吞吐 | 19.56% |
| Tensor pipe利用率（elapsed cycles） | 11.36% |
| DRAM带宽 | 194.76 GB/s，峰值约11.04% |
| Scheduler无eligible warp | 78.72% |
| Active / eligible warps per scheduler | 3.00 / 0.242 |
| Achieved occupancy | 24.96% |
| 寄存器/线程（launch静态平均） | 168 |
| Dynamic shared / 当前shared配置 | 20.640 / 32.768 KB |
| Register / shared block limit | 1 / 1 CTA per SM |
| Local spill requests | 0 |

仍有原生TMA和setmaxnreg，consumer232/producer40预算未变。
共享内存的1-CTA限制是当前32KB carveout下的结果，不代表硬件SM只有32KB shared；
寄存器也独立限制1CTA，所以单独增大carveout不能解决并发问题。

### Warp stall与源位置

每条issued instruction对应平均14.08 warp cycles：

| 类型 | cycles/issued instruction | 占比约 |
|---|---:|---:|
| Barrier | 4.608 | 32.7% |
| Long scoreboard | 3.554 | 25.2% |
| Wait（固定延迟依赖） | 2.481 | 17.6% |
| Branch resolving | 1.336 | 9.5% |
| Short scoreboard | 0.256 | 1.8% |
| MIO throttle | 0.0086 | 0.061% |

这些是warp统计，**不是kernel墙钟比例，也不能作为可相加的加速空间**。

1. **Q初始搬运**（`core_fwd.cu:60–68`）
   是标量generic `LD.E.U16 → ST.E.U16`，不是TMA/cp.async/vectorized搬运。
   四个ST.E.U16指令位置等待其输入加载，占9127个long-scoreboard样本，
   为该类的47.9%（全部75808样本的12.0%）。随后Q-staging的CTA barrier占5961样本。
   两者形成明显的启动阶段等待，Q搬运值得最先单独验证。
2. **hard key label读取**（`core_fwd.cu:37`）
   `LDG.E.64 → ISETP.NE.S64`是另一条直接load-use链，占7105个long-scoreboard样本，
   为该类37.3%（全部样本9.4%）。query-row标签虽然已缓存，key标签仍在逐元素score中加载。
   同一个key tile在warp内不同query行及多个compute warp间复用，候选是按tile预取/暂存。
3. **Barrier最高，但不能误判为配对通信占32.7%**
   24806个barrier样本中18828（75.9%）落在末尾CTA同步后的EXIT PC附近。
   结合源码，warp9–11不承担加载工作，在角色分支后直接等待CTA结束，
   是该统计的重要来源；没有按物理warp分别采样，不能精确分摊各warp的贡献。
   初始Q barrier另外占5961，第二个初始化barrier仅17。
   配对mailbox和producer等待属于mbarrier轮询，也可表现为branch/scoreboard等待，
   不能等同于这里的CTA-barrier指标。
4. **分支与依赖，而非scan的MIO吞吐饱和**
   score因果/hard选择、LSE负无穷guard和mbarrier轮询都有热点。
   分支效率75.70%，MIO throttle却很低；没有证据支持当前首先优化roll shuffle吞吐。
   producer free-slot和mailbox轮询的TRYWAIT处也有branch stall，后续应和计算侧负载一起看。

## 数据条件对照（同一kernel，未改代码）

| direction / hard_prob | duration us | SM吞吐 | eligible/scheduler |
|---|---:|---:|---:|
| q_from_k / 0 | 510.432 | 24.77% | 0.320 |
| q_from_k / .5 | 675.584 | 19.56% | 0.242 |
| q_from_k / 1 | 601.760 | 17.72% | 0.221 |
| k_from_q / .5 | 686.048 | 19.22% | 0.239 |

纯hard依然做score MMA，却没有纯soft快；混合又比两端都慢。
这与key-label访存和混合控制流成本一致，不能简单把瓶颈归因于LSE算术。
改变hard比例同时改变了W/scan分支，故这不是严格隔离某一条指令的ablation。

## 建议的验证顺序（尚未实施）

1. persistent kernel，每SM一个CTA；当前workload计算期间由producer异步预取下一workload的Q。
   不采用Q直接global→寄存器；不能让每个任务都在开头重新支付未隐藏的Q加载和CTA等待。
   Q预取槽是用户明确指定的输入流水，不是允许任意计算中间量写shared。
2. 两个compute warpgroup错开阶段，使一组scan/reduce与另一组HMMA尽量重叠。
   保留必须的配对边界依赖，优化ready/release协议而非删除必要同步。
3. key label（以及column-LSE方向的key LSE）按tile寄存器预取/复用或lane shuffle广播，
   降低重复标量load-use依赖；不能为方便实现而增加shared缓存。
4. 基于源级采样优化guard/poll控制流；保留负无穷和identity，不能删guard换速度。
   不为提高occupancy压缩寄存器预算；用实际发射效率与重叠效果评估流水。

精简数据：[summary_ncu_sm120a.json](benchmarks/summary_ncu_sm120a.json)。
完整报告保留在`/tmp/dism-summary-tanh-{q,soft,hard,k}.ncu-rep`（每个15–18MB，未加入git）。
主场景源码/SASS采样导出为`/tmp/dism-summary-tanh-q-source.txt`与`...-sass.txt`。

## 与旧CUDA组织的对照

用户补充旧`src/dism_fwd_nope.cu`曾测得70%以上compute throughput。
这是用户提供的历史结果，本轮没有重跑旧二进制；它应作为重要优化参照，
不能把当前25% occupancy当作算法吞吐上限，也不应优先为提高occupancy而牺牲tile/ILP。
NCU的SM throughput是管线峰值利用率指标，与有效算法FLOPS不是同一概念；历史对比应统一指标。

已从源码确认的组织差异：

- 旧摘要在977–978行用TMA一起预取Q和首个K；新版先标量搬Q、CTA同步、读Q寄存器，
  再同步并开始K流水。新版本的Q load-use及其barrier已有直接profile热点证据。
- 旧摘要K ring为3 stages（DEFAULT_CONFIG第696行），新版2 stages。
  这只是待验证因素；当前K-ready等待没有显示为最主要热点，不能断言加第三级就解决问题。
- 旧score主要是dot后的log2变换与预加载rtau，不存在新版逐元素hard key-label读取和混合分支。
  新版纯soft摘要为510us、混合为676us，元数据和控制流成本必须单独优化。
- 旧摘要是8个compute warp加1个producer warp（第946行），新版本12warp含3个不加载的warp。
  但新版setmaxnreg以完整warpgroup工作，不能直接改回9warp并假定寄存器/同步仍合法。
- 旧`emulated_wgmma`名字并不意味着实际Hopper WGMMA：当前helper会把B读入寄存器，
  再调用warp MMA；不能仅凭函数名把历史差距解释成WGMMA硬件优势。
- 旧scheduler支持persistent任务循环；新版每CTA只做一个128行任务。历史launch配置未复现，
  不能据此给persistent策略归因一个确定的收益。

因此第一优先级是用persistent跨任务Q预取和跨warpgroup HMMA/scan重叠隐藏延迟，
消除key metadata串行load-use并降低热路径分支，不是压低寄存器预算以提高occupancy。

用户随后明确了shared写入的三类例外：计划内防spill staging、必须的warp通信、输出布局整理。
随后用户明确指定下一任务Q的TMA预取，因此保留这一异步输入流水；
这不放宽对其他计算中间量shared写入的限制。此前Q直载寄存器建议已撤回。
