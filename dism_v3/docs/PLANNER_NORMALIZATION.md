# CPU planner 归一化重写（2026-10-24）

基线为已提交的 `a4d131e`（上一轮评估和 profiling 工具修复）。本轮采用
**逐 query 动态缩放累加**，接入 `inference/csrc/prefill.hpp::Builder::finish()`。
不采用缓存全部 query 贡献的两遍方案。没有引入近似 exp/log 或 FP32 标量累加。
数学含义不变，FP64 求和顺序改变，输出不再承诺与旧版逐字节一致。

用户允许 CPU planner 数值误差约 1e-8～1e-7；本轮仍沿用已有更严格 oracle
阈值，未修改任何测试容差。没有改变 CUDA kernel、event/chunk 布局、worker
默认值、调度、RNG、checkpoint 或训练状态，也未把上一轮 A/B 的 centroid/因果
剪枝候选同时混入，因此表中收益单独来自归一化变化。

## 数学和实现

某个 stream 在 query i 处提供：query logweight a、query 导数 da、key pivot p、
缩放 key mass m 和缩放 key 导数 dm。对原始分母及其 tau 导数的贡献分别为：

```
exp(a+p) * m
exp(a+p) * (da*m + dm)
```

每个 query 保存 P、Z、D，满足 den=exp(P)*Z，真实分母导数=exp(P)*D。
P 是数值尺度，不是需要求导的模型参数。初始化 P=0、Z=1、D=0，保留原来的
单位分母项。令 t=a+p、d=da*m+dm：

```
t > P: s=exp(P-t); Z=Z*s+m; D=D*s+d; P=t
否则:  s=exp(t-P); Z+=m*s; D+=d*s
最终:  logden=P+log(Z); dlogden=D/Z
```

原实现每个有效 query 贡献需要 log(m)、logadd 的 exp/log1p 和导数混合的
两个 exp。新实现只需要一次缩放 exp，log/除法推迟到每个 token 一次。
key mass、key decay/weight、系数生成、event 顺序和 packing 过程保持原算法。
不使用 `log(exp(x))=x` 替代，也没有改动 tau 权重表递推。

增加三个 worker-local FP64 向量，容量按该线程处理过的最大 N 保留；梯度关闭
时只初始化其中两个。N8192、16 workers 时三个数组的元素容量约 3 MiB，
这是额外 workspace 容量估算，不是测得的进程 RSS。每次 finish 重新初始化
有效范围，输出仍独立拥有内存，不引用这些临时向量。

两遍候选先缓存每个 query 的 t/m/d，再求 P、累计 Z/D，并复用 exp(t-P)
生成 query 系数。它进一步省掉系数 pass 的 exp，但增加 O(query events) 的
多个 FP64 数组和遍历；实测通常不如只增加 O(N) 工作区的动态缩放版。

## CPU 性能

Core Ultra 9 285K，独立 C++ `-O3 -g -fno-omit-frame-pointer -std=c++17 -pthread`，
无 fast-math。W1 绑 CPU0，W16 沿用 OS 调度。基线/动态缩放/两遍方案打乱顺序，
每个配置三个进程、每次至少 .8 秒，报告进程平均耗时的中位数。
测量包含 plan、普通 CPU buffer 分配、compact C64 pack 和释放，包含首轮调用；
不含 Python、pinned allocator、PCIe 和 GPU。计时不包含 profile 或测试运行。

合成输入沿用 B8/H12/N2048、26 packed segments、299 jobs、tau=.5 的 iid
labels。真实输入直接重放已保存的 model_022888/PG19 labels 和 tau，B4/H12、
每行完整序列、all-hard，在 gradient-enabled compact planner 上执行。
没有加载 checkpoint，没有新启动训练或重新捕获输入。

| 输入 | workers | 基线 ms | 动态缩放 ms | 两遍 ms | 动态缩放耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| iid512 | 1 | 40.748 | 39.527 | 41.605 | 3.00% |
| iid512 | 16 | 5.135 | 5.004 | 5.282 | 2.54% |
| iid32 | 1 | 132.805 | 119.559 | 122.531 | 9.97% |
| iid32 | 16 | 13.310 | 12.337 | 13.847 | 7.31% |
| iid8 | 1 | 206.694 | 176.509 | 193.449 | 14.60% |
| iid8 | 16 | 20.907 | 18.825 | 19.012 | 9.96% |
| n2048_layer0 | 16 | 1.599 | 1.575 | 1.611 | 1.48% |
| n2048_layer1 | 16 | 2.667 | 2.454 | 2.524 | 7.99% |
| n2048_layer2 | 16 | 2.651 | 2.450 | 2.610 | 7.61% |
| n2048_layer3 | 16 | 5.348 | 4.989 | 5.021 | 6.71% |
| n2048_layer4 | 16 | 6.405 | 5.894 | 5.997 | 7.99% |
| n2048_layer5 | 16 | 6.416 | 5.846 | 5.914 | 8.89% |
| n8192_layer0 | 16 | 6.823 | 6.833 | 6.874 | -0.15% |
| n8192_layer5 | 16 | 27.733 | 25.290 | 26.741 | 8.81% |

N2048 真实六层中后五层降低约 6.7–8.9%，第一层仅 1.5%；N8192 第一层无
明显收益，最后一层降低 8.8%。这符合事件较密集时归一化占比更大的判断，
不能宣称所有输入有同样收益。没有测训练整步时间、ready/fallback 或 GPU
吞吐，CPU 改善不能直接换算成训练加速。

## 数值检查与限制

- 两个候选均通过既有 CPU checker：**84 组**独立 SymbolicPlanner oracle，
  保持 `rtol=2e-9, atol=2e-10`；**96 组**并发 classic/compact 打包与候选的
  独立 document plans 逐字节一致。该字节一致性不是与旧归一化系数相等。
- 动态缩放与旧实现的 1,400 组 Builder/Program/chunk 扩展对照，整数拓扑一致，
  最大浮点绝对差 **2.32831e-10**，低于用户允许量级。ASan/UBSan 无内存或
  未定义行为报告，但旧阈值的双实现数值比较有 **15 个 dlogweight 元素超限**，
  不能把该诊断的进程退出成功解释为所有数值比较通过。
- 这 15 个元素来自同一个 N1542、tau=16、重复 token、带 query reset 的 case。
  独立 long double 闭式计数表明，旧版本身在近零导数处有误差；此处候选更准确：

| 对 long double 的最大绝对误差 | 旧版 | 动态缩放 |
| --- | ---: | ---: |
| logden | 1.38e-12 | 1.16e-13 |
| dlogden | 2.66e-10 | 4.64e-11 |
| 近零 query dlogweight | 2.21e-10 | 8.78e-14 |

- 新增独立诊断 `tools/check_planner_normalization.cpp`：重复 labels 时恰有
  `i-r+2` 个因果 key 匹配至少 r 个 token，直接按 r 计数构造分母和导数，不调用
  centroid/SAM 求真值。N1/65/513/1542/4096，正负/近零/零 tau，reset on/off
  共 **80 组通过**，沿用 `rtol=2e-9, atol=2e-10`；最大误差只占阈值的 .00261。
  ASan/UBSan 通过。它专门验证归一化，不是任意 labels 的完整数学证明。
- 候选 native/compact/graph 常规 GPU 回归 **249 passed**，默认排除的 5 项
  strict-gradient 另行运行，得到 **4 passed、1 failed**。旧版在同一个
  `test_event_matches_existing_fp64_contract[boundary-True]` 失败，索引和误差
  完全相同：最大绝对误差 .015784588、相对误差 .093798036；原阈值为 .008/.025。
  这是既有 GPU 严格数值限制，没有隐藏或放宽它，也不能算作所有严格检查通过。
- 接入生产源码后通过版本 `inference/build.py` 重新构建到独立目录，再运行
  native training、compact、graph、native inference prefill 和 SAM inference：
  **382 passed、5 deselected**。strict 项结果见上；没有修改 pytest 默认配置。
- 最终源码的现有 workspace 回归 **1,400 组通过 ASan/UBSan**，fresh/reference-LCA
  与 reused Builder 逐字节相等。这里比较的是同一新算法的生命周期行为。
- 未运行 CUDA sanitizer，没有 CUDA 源码改动，无新增 CALL/spill 检查。
  也没有测试极端 FP64 上限附近的所有 tau；本轮结果不能外推到任意有限输入。

## 复现和产物

维护工具可从固定 Git 基线重建两个候选和基线，不依赖其他 worktree：

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/build_planner_normalization_experiment.py \
  --output dism_v3/build/normalization_new --native

# CPU A/B：将 baseline 替换为 online 或 twopass；不要并行执行计时。
dism_v3/build/normalization_new/baseline/native \
  dism_v3/build/planner_reassessment/iid32.bin 3 16

# 对指定独立 native module 运行既有数学/并发打包检查。
CUDA_VISIBLE_DEVICES='' /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/check_planner_cpu_equivalence.py \
  --native dism_v3/build/normalization_new/online/_dism_prefill.so

# 独立 long double 数值诊断；也可将 include 指向生产 csrc。
g++ -O1 -g -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I dism_v3/build/normalization_new/online/inference/csrc \
  dism_v3/tools/check_planner_normalization.cpp \
  -o dism_v3/build/normalization_new/check
./dism_v3/build/normalization_new/check
```

工具默认基线 `a4d131e`，保存全部基线 headers、修改后的 headers、source hashes
和编译命令。输出必须在本 worktree ignored build 下且使用新目录；独立 extension
使用版本 build.py，MAX_JOBS=2，不安装或覆盖生产 binary/cache。

本轮原始产物位于 `build/planner_normalization/`：`measurements.json`、
`measure.py`、两个候选和 baseline headers/binaries、`oracle.log`、
`check_extended.log`、`long_double.log`、`gpu_tests.log`、`strict_gpu_tests.log`、
`run_gpu_tests.py`、`workspace.log`。最终版本 native 在 `production/`，测试 runner
显式加载该模块并确认 flash_dism/cu_flash_dism 来自本 worktree。
维护构建工具的固定基线复现已在 `build/planner_normalization_pinned/` 验证。
生成文件不入 Git，原有用户文件未修改。训练进程下次加载/构建新版 native 才会
采用此变更；本轮没有重启任何训练。
