# CPU planner：SAM 根分支森林与组件遍历复用

生产入口 `python/flash_dism/inference/csrc/prefill.hpp` 采用根分支森林分解。
FP64 数学、inclusive causal 规则、query reset、梯度公式、packing、chunk 宽度
和 worker 默认值不变。分解顺序和事件数可以变化，不再要求与旧 whole-tree
centroid oracle 的 Program 数组逐字节相同。

## 实现与等价性

suffix-link 树根的匹配长度为零，F(0)=0。不同根分支之间的 LCA 为根，
因此跨分支的 query/key 配对没有贡献，各分支可以独立处理。

`prefill_matcher.hpp` 在 extend 时维护稳定分支编号及按时间有序的 key 链表。
clone 继承被拆分节点的分支编号；若 clone 的 link 指向根，只更新实际分支根，
既有节点不需要重新标记。clone 没有独立 key。SAM Node 仍为四个 int，新增
整数元数据单独存储；没有在线维护动态 centroid。

query 匹配时构造相应分支的 query 链表。无查询或最早 key 晚于最后 query 的
分支直接跳过；分支内丢弃早于最早 key 的 query 和晚于最后 query 的 key。
保留同位置配对，query 无贡献时归一化分母仍为 1。

每个活跃分支只做一次初始 BFS/子树大小统计。centroid 的分组遍历已产生下一层
的节点顺序和父子方向，递归直接借用父层 arena。移除 centroid 后，子侧的
子树大小保持不变；仅沿旧 root→centroid 路径修复父侧大小。父层 arena 在
子调用结束前不改动，分层 scratch 使用 deque 保证增长时引用有效。

新增分支维护与分桶为 O(N)，分解仍为 O(N log N)，不引入近似。SAM 和邻接表
仍完整构造。所有分布都走同一路径，按用户要求不添加大量重复标签的回退或阈值。

## 原型性能依据

Core Ultra 9 285K；B4/H12 保存输入；同编译选项的 C++ 原生 benchmark。
包含 planning、输出分配、compact packing 与释放，不含 Python/H2D/GPU。
W1 固定 CPU0，W16 由系统调度；五轮交错运行，每次至少 1.5 秒，取中位数。
后台存在用户训练/评估，未中止；没有与本实验编译和测试重叠。

| 输入 | W1 基线 → 组合原型 ms | W16 基线 → 组合原型 ms | W16 累计 CPU 时间降低 |
|---|---:|---:|---:|
| N2048 layer0 | 16.206 → 8.225 | 1.708 → 0.972 | 46.0% |
| N2048 layer3 | 32.203 → 26.941 | 4.874 → 4.396 | 14.6% |
| N2048 layer5 | 41.146 → 36.539 | 5.759 → 5.318 | 10.9% |
| N8192 layer5 | 189.493 → 163.731 | 25.068 → 22.703 | 12.0% |

独立线程 CPU 分阶段计时：layer0 的 decompose 8.638→0.268 ms，SAM
3.352→3.799 ms；N8192 layer5 的 decompose 77.153→51.969 ms，SAM
15.619→17.438 ms。额外维护成本被省下的分解工作覆盖，packing 基本不变。

layer3 事件数增加 3.03%，两个 layer5 分别减少 0.44%/0.80%，layer0 不变。
全重复标签没有过滤收益，原型 W1 慢 6.3%、W16 基本持平；不针对该分布添加
特殊路径。上述数字为已测原型，不能直接解释为训练吞吐或 GPU 加速。

隔离原型及原始结果位于忽略目录 `build/planner_root_forest/`。生产集成移除了
未使用的分支节点计数和整段 ids 初始化，直接进入森林分解；保留相同的数学和
事件布局。集成验证产物位于 `build/planner_root_forest_merge/`，不入 Git。

## 回归契约

- `tests/native/test_prefill_branches.cpp`：200 组逐前缀 suffix-link 根归属与
  key 链表唯一性检查，覆盖 clone、重复、周期及不同词表。
- `tests/native/test_prefill_math.cpp`：500 组独立 long-double dense oracle，
  检查所有 pair 权重及 tau 导数、logden/dlogden，含 reset 与 tau ±16。
- `tests/native/test_prefill_workspace.cpp`：1,400 组 Builder 复用、reference LCA、
  长度/tau/梯度切换，检查 Program 与 C16/32/64 chunk 一致性。
- `tests/native/test_inference_packing.cpp`：100 组并发 packing/routing 对照。
- Python symbolic planner 保留原有 whole-tree 分解，通过执行结果、全部向量梯度、
  tau 梯度和 tangent 验证等价，保持 rtol=2e-9、atol=2e-10。
  native chunk/batch packing 则继续与独立 Python packing 和逐文档计划对照。
- 同一森林布局的 reference LCA 与快速 LCA 仍要求数组逐字节一致。

集成后的验证结果：

- 上述四个 C++ 程序均开启 ASan/UBSan 并通过，未禁用 leak 检查；dense oracle
  最大绝对误差 1.15879e-12。
- 额外以隔离 forest_reuse 原型作基线，1,400 组 Program/C16/32/64 chunk 数组
  逐字节相同。N2048 layer0 与 N8192 layer5 的完整训练 compact buffer SHA256
  也与原型一致，包含所有 batch/head、document 与 hard mask 路由。
- 重新用本版本 inference/build.py 构建生产 CPU 扩展：84 组 symbolic 数值等价
  和 96 组并发普通/compact batch packing 检查通过，CPU-only 检查未初始化 CUDA。
- Python/GPU：native training、compact training、native inference、planner graph、
  event backward 五个测试文件 325 passed；SAM prefill 相关测试 86 passed。
  另验证共享诊断 helper 的 fused 调用路径，56 passed（14 项无关测试未选中）。
  合计 **467 passed**。测试显式加载本次构建的扩展及本 worktree 的
  flash_dism/cu_flash_dism。严格梯度的 5 项按既有配置排除，另 16 项 decode
  测试未选中，本次不改变 decode 算法。
- profiler 新入口编译与分阶段计时 smoke 通过；git diff --check 通过。

本次未修改 device kernel，未重跑 CUDA sanitizer 或 device CALL/spill 检查。
BF16 严格梯度限制仍见 [PLANNER_NORMALIZATION.md](PLANNER_NORMALIZATION.md)，
没有修改容差或以新的参考实现覆盖该限制。GPU 测试验证正确性与调度，未测量
训练端到端吞吐，也未修改 checkpoint 或重启现有训练。

构建与 CPU 扩展回归（使用 conda blkw）：

```bash
MAX_JOBS=2 TORCH_EXTENSIONS_DIR="$PWD/dism_v3/build/planner_root_forest_merge/cache" \
  python dism_v3/python/flash_dism/inference/build.py --component prefill \
  --output dism_v3/build/planner_root_forest_merge/_dism_prefill.so
CUDA_VISIBLE_DEVICES='' python dism_v3/tools/check_planner_cpu_equivalence.py \
  --native dism_v3/build/planner_root_forest_merge/_dism_prefill.so
```

原生 sanitizer 测试使用 g++ -O1 -g -std=c++17 -pthread
-fsanitize=address,undefined -fno-omit-frame-pointer，并将本工作目录的
python/flash_dism/inference/csrc 加入 include path。原始日志、完整输出 hash
和源码/二进制校验记录保存在忽略的 build/planner_root_forest_merge/ 中。
