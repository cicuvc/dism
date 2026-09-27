# Copy task训练验证（2026-09-08）

结论：当前CUDA Dism前后向能在`copy_task.py`的任务上连续训练并显著降低loss，
配对WS三个seed、对称WS一个seed、Triton embedding反向对照均完成1000步。
本批没有NaN/Inf或非有限梯度，但尚未达到整段64-token稳定全对。

入口`python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 0`。
原`copy_task.py`未改动，独立入口复用其DismTransformerBlock、SwiGLU、卷积、门控归一化
及调度器，只替换其局部voc_dism为当前autograd，添加指标和梯度有限性检查。
未使用W&B或上传数据，未保存checkpoint。

## 保持的任务与参数

- B64、N128、token vocab128（采样0…126）；随机64-token串重复两遍，预测第二遍。
  Teacher forcing与原loss位置切片不变，不是自回归生成测试。
- 一层，d_model128，4 heads，D=DV=32，Dism vocab256；原模型含short convolution和门控。
- FP32 master参数，autocast BF16；进入Dism的q/k/v/vocab为BF16，rtau为FP32。
  CE继续按原脚本在autocast外对BF16 logits计算；没有顺便改为FP32 CE。
- AdamW lr5e-3、原no-weight-decay分组、weight decay1e-2、clip norm1、50步warmup、cosine到0.1倍LR。
- sm_scale=1；rtau=softplus(log_sel_tau)，未添加clamp。hard_prob=step/1000，
  训练每次调用选择一个全局random direction；评估hard_prob=1、固定方向（hard数学与方向无关）。
- data seed12345、row RNG seed777；hard评估seed424242、每次10个batch共640条序列。
  RNG使用当前可重放Philox设施，不与原torch.rand随机mask逐bit相同。
- 每100步评估，并补充训练前和最后一步评估；训练eval不推进step或行RNG。
  忽略原脚本未用于实际voc_dism的旧leak模式标题/配置，不改变实际where软硬切换语义。

## 完整1000步结果

所有运行使用CUDA core和CUDA embedding前向；表中backend只表示embedding反向。

| 反向 | seed | 初始hard loss | 最终hard loss | token准确率 | 整段全对率 | 第5…64 token准确率 |
|---|---:|---:|---:|---:|---:|---:|
| CUDA配对 | 0 | 5.0250 | 0.1516 | 97.34% | 2.97% | 99.11% |
| CUDA配对 | 1 | 5.0219 | 0.1545 | 97.26% | 0.625% | 99.66% |
| CUDA配对 | 2 | 5.0063 | 0.1917 | 96.62% | 0.3125% | 98.91% |
| CUDA对称 | 0 | 5.0250 | 0.1402 | 97.15% | 11.25% | 98.90% |
| Triton | 0 | 5.0250 | 0.1383 | 97.09% | 11.875% | 98.70% |

配对三个seed的首个复制token准确率分别5.31%、1.41%、1.56%，第二个为84.84%、
52.19%、55.47%。因此全段全对率较低与复制段开头错误有关。
用户确认：当前模型没有位置编码，依赖基于语义的接续复制，需要一定数量的启动token
消除歧义后才能连续匹配；开头段错误符合该任务设定的预期，不作为kernel训练失败的判据。
整段全对率仍保留为原始指标，评价连续复制能力时应同时查看启动段之后的逐位置准确率。

各运行梯度裁剪前最大norm为1.43…2.67，clip调用启用error_if_nonfinite，未触发。
rtau最大0.903…2.073，均低于ln32≈3.466。soft/hard概率升高过程中未观察到数值崩溃。
三种反向都能学习，本批不足以断言某一后端训练质量更好或量化误差已无影响。

首轮未记录逐位置准确率的seed0试跑也成功：hard loss5.025→0.2182、token96.05%、
整段1.25%。补齐指标后重新运行得到上表seed0结果，不能声称固定seed逐bit复现；
当前core反向包含FP32 atomic且不支持deterministic backward，未定位这次训练轨迹差异的具体来源。
所有结果均为1000步有限验证，不代表长期训练稳定性或完全解决任务。

RTX5090、conda blkw。暖编译缓存的五次训练循环约8.2…9.5秒，含评估/日志但不含导入和模型构建，
不是受控kernel或端到端性能基准；初次试跑约13.7秒。训练峰值显存约156MiB（初次试跑约369MiB），
受编译/autotune缓存影响，不作为固定模型内存需求保证。

原始日志`/tmp/dism-copy-{cuda,cuda_symmetric,triton}-seed*.jsonl`及对应stderr。
配置、评估曲线和逐位置最终准确率固化于
[`benchmarks/copy_task_sm120a.json`](benchmarks/copy_task_sm120a.json)。

```bash
python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 0
python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 1
python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 2
python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 0 --embedding-backward cuda_symmetric
python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 0 --embedding-backward triton
```
