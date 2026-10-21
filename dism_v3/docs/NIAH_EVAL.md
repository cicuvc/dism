# 训练 eval step 的短 NIAH 指标

`nanochat/scripts/base_train.py` 默认在 validation step 后运行短 NIAH，
包括 step 0 和最后一步；沿用 `--eval-every`，恢复训练的入口 step 与原有
validation 一样跳过。`--no-niah-eval` 可关闭，`--eval-every -1` 也会一起关闭。
实现位于 `nanochat/nanochat/niah_eval.py`。

## 默认配置与本地数据

```bash
--niah-data ~/downstream-eval/data/pg19/test.parquet \
--niah-contexts 8192 \
--niah-depths 0.05 0.5 0.95 \
--niah-trials 2 \
--niah-seed 1234
```

每个 context/depth 使用 2 对 present/absent 样本，默认共 12 次单样本
forward。context 必须为正的 256 倍数。这里只读取本地 Parquet 的前 6 篇
`text`，使用当前训练 tokenizer，不下载数据。缺文件、缺依赖、损坏文件、
背景文本不足或 tokenizer 不兼容时记录 warning，返回 `niah/skipped=1`，
训练继续。同一失败原因只警告一次，下一个 eval 仍会重试。
模型执行错误不能伪装成数据缺失，会在恢复状态后报错。

## 任务与指标

沿用早期 NIAH circuit probe 的历史协议：
插入的是 `\nThe special magic number is {number}.` 编码后去掉 query 前缀的
**数字尾部 tokens**，不是完整提示句；query 是 `\nThe special magic number is`。
每个 present/absent 对共享背景、位置和答案；absent 用等长背景片段替换插入尾部。
使用 5 位数，答案最多评分 6 tokens（具体 tokens 由训练 tokenizer 决定）。

每个 cell 使用私有 `random.Random(f'{seed}:{context}:{depth}')`，CPU 样本缓存
在 evaluator 中；文件大小/mtime 或 tokenizer 实例变化时重建。
结果包含实际输入 tokens 的 SHA256，便于确认跨 eval 使用同一套样本。

- `niah/present_nll`、`niah/absent_nll`：每个 trial 的答案平均 NLL，再对 trials 平均。
- `niah/gain = absent_nll - present_nll`：正值表示 needle 降低答案 NLL。
- `niah/{present,absent}_token_acc`：teacher-forced token 命中率。
- `niah/{present,absent}_trial_all`：全部评分 tokens 都命中的 trial 比例。
- `niah/skipped`：正常为 0，跳过为 1；跳过时不伪造其他指标为零。

另有 `niah/ctx_<length>/depth_<depth>/...` 分组指标。指标写入同一次 W&B
validation log；完整记录追加到 run base directory 的 `niah_validation.jsonl`，
并打印 `NIAH_VALIDATION`。NLL 单位为自然对数。

这是小样本、teacher-forced 的检索诊断，不能当作自由生成 exact match。
沿用独立 probe 的最终 norm hook，仅提取答案位置，再用 FP32 lm head 与
softcap 计算 logits，避免为整个长 context 分配完整词表 logits。

## 状态隔离

- 独立准备输入，不接收、不创建、不推进训练/验证 dataloader，不替换训练预取 batch。
- Python、NumPy、Torch CPU 和执行设备的 CUDA RNG 在整个数据准备与评估期间保存并恢复。
- 使用 eager `orig_model`、`no_grad`、临时 eval/full-hard 模式；不经过训练编译图。
  临时扩展 `config.sequence_len` 以容纳长 context，退出时恢复。
- 保存并恢复每个子模块的原始 training flag 和 `eval_hard`。
  正常路径不写 model buffers，RNG counter、annealing 状态保持不变，
  连 buffer tensor version 也不增加。若发现 buffer 被意外修改，恢复其数值并报错。
  不运行 optimizer，不清空或更改已有 gradients。
- 分布式只有 rank 0 执行 probe，其余 ranks 等待结果广播；成功、跳过或错误均广播，
  防止某个 rank 单独抛错而让其他 ranks 卡在该广播上。
- 仅支持当前 DISM packed 模型。其他模型 warning 后跳过。

回归测试：`nanochat/tests/test_niah_eval.py` 覆盖配对构造、全局 RNG、
模型模式/buffers/gradients、异常恢复、缺数据重试、live loader 和预取 batch、
双进程广播及真实 CUDA DISM 状态隔离；同时运行 checkpoint resume/prefetch 测试。
