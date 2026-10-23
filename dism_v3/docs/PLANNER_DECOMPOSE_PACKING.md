# CPU planner：decompose 与 packing 的精确优化

本轮以 `ad43351` 的 FP64 动态缩放 normalizer 为基线。FP32 改动已按要求
暂缓；本轮只删除冗余工作，不改变数学、浮点类型、容差、stream/event 顺序、
chunk 边界、padding、worker 默认值或 CUDA kernel。

## 实现

`python/flash_dism/inference/csrc/prefill.hpp`：

- 输入 query/key 列表按位置有序。若最早 key 晚于最晚 query，整个递归子树
  都没有因果配对，直接返回；保留同位置 key/query 的配对。
- merge tree 中最多一个活跃 group 包含原始 parent component。跟踪其 group
  ID，`cross()` 只分区这一侧；两侧均不含 parent 时直接 emit。保留原有 heap
  权重、tie-break、centroid 选择和 emit 顺序。
- arena 里已有列表不可变。与空列表 merge 时直接复用非空 range，省去复制。
- 最后两个 group 的 cross 完成后，合并结果不再被使用。递归使用的是原始
  component range，因此不再构造最终 merged query/key 列表。

`python/flash_dism/inference/csrc/prefill_training.hpp`：

- 在已有 reset/chunk 边界扫描中，检查非 reset key 的 decay 是否全部**精确等于
  1**；首项 reset 不参与 prefix，query 不贡献 decay。不使用近似阈值。
- 每个 chunk 保存一个字节的零 prefix 标记。满足条件时，直接填充 +0 prefix，
  做原有 signed row 映射和 FP64→FP32 输出转换，省去逐项 log 和串行 prefix
  累加。dlog 转换独立成循环；padding 字节保持原样。
- 普通 layout、compact single bucket 和 compact multi chunk 共用这个快路径。
  不满足条件的 chunk 使用原来的 FP64 prefix 循环。单独的 `chunk_program()`
  保持原实现，也作为 batch packing 的独立回归对照。

代价：每个 job 多一个小 vector，逻辑长度为 chunk 数量；边界扫描多一次
unit-decay 判断。输出元数据大小不变。稀疏、短 chunk 的收益较小，packing
单独启用不保证更快。

## 测量方法

本机为 Core Ultra 9 285K（8P+16E）。使用已有输入，不运行训练或生成新的训练数据。真实输入来自
`build/prefill_planner_compare/inputs/`（B4/H12）；合成输入来自
`build/planner_reassessment/iid{512,32,8}.bin`（B8/H12/N2048，已有 packed
文档边界）。standalone C++ benchmark 包含 planning、输出分配、compact C64
packing 和释放，关闭结果 dump 的计时才用于性能比较；不含 CUDA 执行/H2D。

编译为 `-O3 -g -fno-omit-frame-pointer -std=c++17 -pthread`。单 worker 固定
CPU 0；16 workers 使用 OS 调度。每组变体交错运行，不与编译/测试重叠。
短测为每变体 3 次、每次至少 0.8 秒；确认测为每变体 5 次、每次至少 2 秒，
报告各轮平均每 batch 耗时的中位数。异构核心调度/频率会产生波动，不能把
小于几个百分点的单次变化解释为稳定收益。

确认测结果（ms/batch，五轮中位数）：

| 输入 | workers | 基线 | 仅 decompose | 两项组合 | 组合耗时降低 |
|---|---:|---:|---:|---:|---:|
| iid512 | 16 | 5.117 | 4.890 | 4.947 | 3.3% |
| iid32 | 1 | 119.466 | 114.691 | 102.050 | 14.6% |
| n2048_layer5 | 16 | 5.887 | 5.572 | 5.573 | 5.3% |
| n8192_layer5 | 16 | 24.726 | 23.913 | 23.892 | 3.4% |

真实 W16 输入本轮主要受益于 decompose，packing 的额外端到端收益较小；
iid32 单核则明显受益于两项组合。初轮较短测量中，真实 N2048 六层组合
耗时降低 3.0%–7.6%，N8192 第 5 层为 11.6%；延长复测后后者只有 3.4%。
因此不把初轮最大值当作可复现的固定加速。N8192 确认测单轮基线范围
24.10–25.85 ms，组合范围 23.23–24.92 ms，区间有重叠。

阶段插桩独立运行，W1/CPU0，每项 3 次、每次至少 1.5 秒。以下是阶段耗时
中位数（ms/batch）；它包含计时/分配计数开销，不能代替上表的无插桩吞吐。

| 输入 | 阶段 | 基线 | 组合 | 耗时降低 |
|---|---|---:|---:|---:|
| N2048 layer5 | decompose_emit | 20.511 | 17.907 | 12.7% |
| N2048 layer5 | boundary_count | 1.450 | 1.526 | -5.3% |
| N2048 layer5 | compact_pack | 5.833 | 5.000 | 14.3% |
| iid32 | decompose_emit | 48.618 | 43.840 | 9.8% |
| iid32 | boundary_count | 5.710 | 5.093 | 10.8% |
| iid32 | compact_pack | 16.390 | 15.233 | 7.1% |

真实输入的 decompose/emit 降低约 12.7%，packing 降低约 14.3%；边界扫描
反而增加约 5.3%，符合分类带来的开销。packing 的改进不等于消除了边界
扫描或内存流量。decompose 仍是主要阶段热点。本轮没有重新采集 PMU/IMC，
不能据此宣称 frontend 或内存带宽压力下降了某个百分比。

另测了仅在通用循环里跳过 `log(1)` 的方案。它不需要分类标记，但合成
密集输入 iid32 的单核组合耗时约 113.76 ms，零 prefix chunk 方案约
102.95 ms，同轮基线 119.84 ms；16 workers 下差异较小且随输入变化。
本轮保留 chunk 专用循环，未加入基于特定输入的切换阈值。

## 验证

- 1,400 组复用 Builder 测试与基线完整 Program、C16/32/64 chunk 数组逐字节
  一致；覆盖参考 LCA、梯度开关、reset、长度/tau 切换、随机/重复/周期标签。
  decompose 候选通过 ASan/UBSan。
- 100 组并发 inference plan/packed/routing buffer 对照逐字节一致；组合候选
  通过 ASan/UBSan，覆盖无梯度和 query reset 路径。
- 最终生产扩展：84 组 symbolic oracle 数值检查通过，沿用
  `rtol=2e-9, atol=2e-10`；96 组并发 batch packing 与独立逐文档计划逐字节
  一致，覆盖 C16/32/64、普通/compact、空 hard mask、不同 tau 广播方式。
- 初轮 14 组真实/合成输入与 worker 配置，各变体输出的完整 compact buffer
  SHA256 相同；确认测再次比较生产 binary 的完整输出。
- 最终生产扩展 GPU/接口回归：`test_native_training_planner.py`、
  `test_compact_training_planner.py`、`test_planner_graph.py`、
  `test_native_inference_prefill.py`、`test_sam_inference.py`，结果
  **382 passed，5 deselected**。
- 默认排除的 strict-gradient 项本轮未重跑。上一轮已有的 BF16 严格梯度限制
  仍见 [normalizer 报告](PLANNER_NORMALIZATION.md)，不因本轮优化而消失。
  本轮未修改 device kernel，未重新运行 CUDA sanitizer 或检查 device CALL/spill。

## 复现与产物

使用 conda `blkw`，本 worktree 的 `dism_v3/python` 和 `dism_v3/tools`。
生产扩展由版本内 build.py 构建：

```bash
MAX_JOBS=2 TORCH_EXTENSIONS_DIR="$PWD/dism_v3/build/planner_decompose_packing/cache" \
  python dism_v3/python/flash_dism/inference/build.py --component prefill \
  --output dism_v3/build/planner_decompose_packing/_dism_prefill.so
CUDA_VISIBLE_DEVICES='' python dism_v3/tools/check_planner_cpu_equivalence.py \
  --native dism_v3/build/planner_decompose_packing/_dism_prefill.so
python dism_v3/tools/build_planner_cpu_profile.py \
  --output dism_v3/build/planner_decompose_packing
```

忽略的本地产物：`build/planner_decompose_packing/` 中的 `benchmark.json`、
`bytes.json`、`phases_*.csv`、`cpu_equivalence.log`、`gpu_tests.log` 和
`build_manifest.json`。初轮隔离变体和交错测量脚本在
`build/planner_fp32_opt/`；该目录名沿用已暂停的精度实验，选用的
`decompose`、`packing`、`combined` 变体均为 FP64。它们的 baseline source
快照对应 `ad43351`。所有二进制、profile 和 JSON 均不入 Git。
