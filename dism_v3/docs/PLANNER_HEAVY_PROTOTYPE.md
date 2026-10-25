# CPU planner 重链分解原型对照

本轮实现两个隔离的 C++ CPU planner 原型，生产默认仍为 `3bfd918` 的
centroid/merge 分解。FP64、SAM/query 匹配、归一化、系数生成、普通/compact
packing 和 CUDA kernel 均沿用当前实现。本轮比较 CPU 耗时，不以事件数下降
作为候选成功的条件。

## 两个原型

`weighted`：suffix-link 树重链分解，链内按附属子树规模做带权三路分组；
规模比沿轻边递归可累加抵消，路由/事件规模为 O(N log N)。

`binary`：相同的重链分解，链内按位置中点二分；每个 token 最多经过
O(log N) 条重链，每条链 O(log N) 层分组，界为 O(N log² N)。

两版都用同一套实现：

- 一次遍历计算子树大小和重儿子，构造静态二叉合并拓扑。链内 join 表示
  浅/深两个区间之间的关系；同一 attachment 节点上的轻子树使用按节点数
  加权的 heap 合并，LCA 固定为 attachment。
- query/key 沿叶到根路径访问所属 stream；进入同一 attachment 的 join 时
  更新 attachment，沿链 join 时保留 attachment。浅侧 query 或 key 承担
  权重，另一侧因子为 1，全部流仍是正权重。
- 先统计 query 数及最后一个 query，再统计因果有效 key 数，精确分配扁平
  stream 数组。第二遍按原 token 顺序填充，所以不需要合并有序事件列表。
- SAM 根节点匹配长度为零，根处轻子树之间的所有贡献为零；这些重链树保持
  不连通，省掉无贡献的 heap 合并与后续路由。这对应生产 centroid 版已有的
  根 merge-tree 剪枝。
- worker-local 拓扑和计数数组复用容量，没有为每个节点/stream 分配独立对象。
  当前实现仍对 query/key 各做计数、填充两次路径遍历。

原型文件 `tools/planner_heavy_prototype.hpp` 是注入 Builder 的成员片段；
`tools/build_planner_heavy_experiment.py` 在忽略的 build 目录复制当前源码，
替换 decompose，构建 baseline/weighted/binary 三个版本。生产头文件没有
新增实验开关，也没有改变默认加载路径。

## 测量

Core Ultra 9 285K，单 worker 绑定 CPU 0，16 workers 使用 OS 调度。
standalone C++ 使用 `-O3 -g -fno-omit-frame-pointer -std=c++17 -pthread`。
各变体先在独立进程预运行 0.2 秒，再交错运行 5 轮，每轮至少 1.2 秒，取每轮
平均 ms/batch 的中位数。正式运行的进程内循环复用 worker 和 scratch；首个
batch 的初始化仍计入该轮时间。编译、测试、插桩测量和吞吐测量分开执行。

测量包含 planning、普通 CPU 输出 buffer 分配、compact C64 packing 和释放；
不包含 Python/GIL、pinned allocator、H2D、GPU 或训练整步。真实输入来自保存的
`prefill_planner_compare/inputs/`，B4/H12，使用全部 batch/head；合成 iid 输入为
B8/H12/N2048，保留已有 packed document 边界、hard mask 和 tau。repeat/periodic
仅替换 iid32 的标签为全零/周期 7，其他字段不变。

完整 CPU 路径耗时（ms/batch，五轮中位数；正百分比表示变慢）：

| 输入 | workers | centroid | weighted | 变化 | binary | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| n2048_layer5 | 1 | 40.967 | 50.425 | +23.1% | 52.707 | +28.7% |
| n2048_layer5 | 16 | 5.513 | 6.308 | +14.4% | 6.689 | +21.3% |
| n8192_layer5 | 1 | 182.969 | 230.334 | +25.9% | 245.699 | +34.3% |
| n8192_layer5 | 16 | 24.065 | 27.446 | +14.1% | 29.408 | +22.2% |
| iid512 | 1 | 37.296 | 42.332 | +13.5% | 42.229 | +13.2% |
| iid512 | 16 | 4.951 | 5.427 | +9.6% | 5.337 | +7.8% |
| iid32 | 1 | 102.398 | 110.718 | +8.1% | 111.565 | +9.0% |
| iid32 | 16 | 12.430 | 13.117 | +5.5% | 13.405 | +7.8% |
| repeat | 1 | 154.504 | 130.891 | -15.3% | 116.054 | -24.9% |
| repeat | 16 | 20.376 | 17.114 | -16.0% | 15.925 | -21.8% |
| periodic | 1 | 120.282 | 96.910 | -19.4% | 74.828 | -37.8% |
| periodic | 16 | 17.483 | 14.486 | -17.1% | 13.188 | -24.6% |

**结论：当前真实输入不建议替换 centroid。** weighted 在真实 N2048/N8192 的
16-worker 总耗时增加约 14%；binary 增加约 21%–22%。iid 输入也回退。
长链形态的全重复/周期输入则受益，binary 的单核总耗时降低约 25%/38%，
16-worker 降低约 22%/25%。这些合成结果不能作为真实模型输入上的加速承诺。

W1/CPU0 独立阶段插桩，每变体三轮、每轮至少 1 秒，阶段中位数如下。
插桩数据用于定位额外成本；不代替上表无插桩吞吐。

| 输入 | 阶段 | centroid ms | weighted ms | binary ms |
|---|---|---:|---:|---:|
| n2048_layer5 | decompose_emit | 17.920 | 25.661 | 26.680 |
| n2048_layer5 | scalar_denominator | 4.968 | 5.867 | 6.424 |
| n2048_layer5 | coefficients | 3.747 | 4.230 | 4.635 |
| n2048_layer5 | boundary_count | 1.535 | 1.594 | 1.648 |
| n2048_layer5 | compact_pack | 4.985 | 4.995 | 5.512 |
| n8192_layer5 | decompose_emit | 77.042 | 122.484 | 129.005 |
| n8192_layer5 | scalar_denominator | 24.133 | 26.812 | 29.543 |
| n8192_layer5 | coefficients | 24.595 | 24.149 | 27.217 |
| n8192_layer5 | boundary_count | 7.382 | 7.368 | 7.667 |
| n8192_layer5 | compact_pack | 17.522 | 17.125 | 19.222 |
| repeat | decompose_emit | 68.379 | 52.059 | 51.286 |
| repeat | scalar_denominator | 20.210 | 16.156 | 14.044 |
| repeat | coefficients | 21.127 | 24.361 | 16.263 |
| repeat | boundary_count | 3.809 | 3.631 | 2.992 |
| repeat | compact_pack | 36.181 | 29.827 | 26.942 |

真实输入的主要回退发生在 decompose/emit 本身。原型虽然删除了重心搜索和
事件列表 merge，但新增了合并拓扑构造及 query/key 各两次叶到根路由。
阶段计时证实整体分解阶段变慢；尚未单独测量这两项内部成本，不能把差值
全部归因于其中某个循环。分母和系数生成也有额外成本。

特别是 weighted：真实 N2048 事件数只增加 1.9%，输出元数据减少 0.8%；
N8192 事件数增加 2.2%，输出元数据基本持平。这表明回退不只是事件数或
packing 流量变大，CPU 分解实现的常数已经足以抵消理论上的结构简化。

下一步若继续这一路线，应优先验证按节点预展开的连续 route 表，减少四遍
parent 指针追踪，而不是先调整 GPU 或进一步牺牲数值精度。目前尚未实现或
测得该改进的收益。

## 验证与边界

- baseline 和两版候选均通过 500 组独立 long-double 检查。oracle 直接计算
  query/key 的连续匹配长度及几何权重，不使用 SAM、LCA 或任何树分解。
  比较每个归一化 pair weight、其 tau 导数、logden 和 dlogden，覆盖
  N1–128、query reset、梯度开关、正负/近零/零 tau、重复/周期/不匹配标签，
  以及复用 Builder。最终候选最大绝对误差约 1.2e-12。
- 两版各通过 100 组并发 inference packing/routing 检查，C16/32/64、无梯度和
  query reset 路径均覆盖。以上 C++ 检查使用 ASan/UBSan。
- 最终候选各通过 **136 passed，90 deselected**：native training 的行为/并发/
  GPU backward 测试及全部 compact layout/GPU/stream 测试。90 项排除的是
  强制要求与旧 centroid 完全相同 rows/offsets 的 84+6 项布局测试；这项合同
  不适用于算法替换。没有修改原测试或放宽数值容差。
- 新拓扑改变 stream 顺序、浮点求和次序和 chunk 边界，因此不声称与旧版逐字节
  一致。C++ 独立 dense oracle 是数学对照，packing 检查比较候选自身的独立
  逐文档计划与 batch 路径。
- 本轮不是完整部署验收；没有重新运行 strict-gradient、CUDA sanitizer、训练
  或 checkpoint 恢复。GPU kernel 未改动，未重新检查 device CALL/spill。

## 复现

使用 conda blkw；扩展通过复制到隔离目录的本版本 inference/build.py 构建，
MAX_JOBS=2，独立 TORCH_EXTENSIONS_DIR。只使用本 worktree 的 Python 包。

```bash
python dism_v3/tools/build_planner_heavy_experiment.py \
  --output dism_v3/build/planner_heavy_pruned --native
python dism_v3/tools/benchmark_planner_heavy.py \
  --experiment dism_v3/build/planner_heavy_pruned \
  --inputs dism_v3/build/prefill_planner_compare/inputs/n2048_layer5.bin \
           dism_v3/build/prefill_planner_compare/inputs/n8192_layer5.bin
```

构建和 benchmark 要求新的输出目录/结果文件，避免覆盖既有实验。
`tools/check_planner_heavy.cpp` 用 `-I <variant>/inference/csrc` 编译即可运行
独立 oracle；`tests/native/test_inference_packing.cpp` 使用同样 include 路径。
加入 `-fsanitize=address,undefined -fno-omit-frame-pointer` 可复现 sanitizer 检查。

最终结果及 build manifest 位于忽略的 `build/planner_heavy_pruned/`：
`benchmark.json`、`benchmark.log`、`<variant>/phases_*.csv`、`validation.log`、
`<variant>/gpu_tests.log`。第一版未剪枝的隔离产物在 `build/planner_heavy_cpu/`，
不作为最终成绩；其初轮真实 N2048/W1 约 53.6/56.4 ms，对照 40.8 ms，暴露了
根处无贡献路由的额外成本。所有二进制、缓存、结果 JSON 均不入 Git。
