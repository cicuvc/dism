# 中段 DISM 布局诊断（2026-10-23）

目标：保留最后一个 DISM 后面的 GPU 计算，以覆盖 planner 的 D2H、CPU 规划与
通知延迟，同时定位 `gggddddddggg` 相对 `ggggggdddddd` 的语言建模回退。
本次仅加载 checkpoint 做诊断；没有更新权重、恢复训练或修改 checkpoint。
生产架构、算子和默认配置未改。

## 数据与比较边界

两个 run 均位于 `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/`：

- old：`dism-slimpajama-125m-h768-h12-i1152-r32-mxfp8-ffnonly-3b-docce-niah-clip1-lr8e4-20261022`
- middle：`dism-slimpajama-125m-h768-h12-i1152-r32-mxfp8-ffnonly-3b-docce-niah-clip1-lr8e4-gggddddddggg-planner-20261023-042848`

均使用最终 step 22888。旧 run 未启用 graph/planner，新 run 启用；两者都未使用新
MXFP8 fused CE。每个布局只有一个训练样本，不能把差异全部归因于布局。
相同 seed 也不意味着移动后的 DISM 层具有相同初始化或原训练 hard mask。

两份 train state 保存的 prefetched inputs、targets、cu_seqlens、segment ids
完全一致。固定输入探针将 `[8,2048]` 拆为两个 `[4,2048]` 批次，保留原 packed
行边界，共 12,231 个有效 target、26 个有效文档。hard mask 按 DISM 序号而非
绝对层号对齐，每 packed 行、每 head 采样，不按文档采样。

另用本地验证集前 8 个 `[8,2048]` 批次，共 123,184 个有效 token、109 个文档，
shuffle=False、seq_align=256、aligned_segment_ends=True。缓存时 clone loader
张量，避免复用 staging buffer。以下 NLL 是 token 平均自然对数损失，不是训练
日志的 document loss，也不是 BPB。NIAH 使用原有本地协议：8192 context、
depth .05/.5/.95、各 2 trials、seed 1234；只有 6 trials。

## 1. 回退主要表现为 DISM 带来的 LM 增益减少

| 验证 NLL | old | middle |
| --- | ---: | ---: |
| 全 hard | 2.96630 | 3.01024 |
| 全 soft | 2.94869 | 3.00055 |
| 所有 DISM 输出置零，保留 GDN | 3.14703 | 3.14031 |
| all-off NLL − full-hard NLL | 0.18073 | 0.13007 |

新布局 full-hard 回退约 0.04394，但关掉 DISM 后没有同样的差距。相对 all-off
消融的收益减少约 28%。这支持“已训练模型对 DISM 的有效利用发生变化”，不能
解释为独立训练的纯 GDN 基线，也不是严格的加性因果分解：分支共享 norm/o_proj，
消融会改变混合表示的分布。

逐个关闭第 0–5 个 DISM 的 ΔNLL（层号从零开始）：

| DISM 序号 | old 层 | old ΔNLL | middle 层 | middle ΔNLL |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6 | .00003 | 3 | .00020 |
| 1 | 7 | .00031 | 4 | .00033 |
| 2 | 8 | .00094 | 5 | .00051 |
| 3 | 9 | .01928 | 6 | .00035 |
| 4 | 10 | .05897 | 7 | .01439 |
| 5 | 11 | .00053 | 8 | .06961 |

旧布局主要贡献来自 L9/L10，新布局集中到 L7/L8，尤其最后一个 DISM L8。
这些单层消融不能相加；共同关闭的损害远大于若干单层关闭。
一种待验证解释是：前置 GDN 从六层减为三层，早期 DISM 的输入表征还不够适合
LM 匹配；另一种是尾部重新加工改变了 DISM 获得的监督。当前探针不能区分二者。

## 2. 不支持“新布局普遍增加 selector 切换噪声”

固定每个批次，h=.95，128 次匹配 mask 采样；统计
`r = tr(sample_cov(g)) / ||sample_mean(g)||²`。六层、两个批次的中位数：

| 梯度组 | old r | middle r |
| --- | ---: | ---: |
| q_vocab | 20.53 | 17.60 |
| k_vocab | 18.48 | 15.82 |
| selector_conv | 19.89 | 18.58 |

个别层差异大，但没有系统升高。shared_qk 的 r 约 .01，不能用它代表码表梯度：
Q/K 投影与 GDN 共享，还收到 GDN 梯度。h=.5 的 32-draw 探针则多在低个位数。
这复现了高 hard probability 下门控噪声增加，却不支持它单独解释布局回退。
这是固定输入下的条件统计，不包含数据噪声；有限 draws 的比值有偏，JSON 同时
保存均值平方的有限样本修正量，未把中位数当成显著性检验。

对仅 soft 时激活的梯度，q=1−h，若 soft 梯度均值/协方差为 μ/Σ，则单个门控项
有 `E[g]=qμ`、`Cov(g)=qΣ+q(1−q)μμᵀ`。该公式不能不加区分地应用于整批聚合、
共享 Q/K 或 hard 路径也有梯度的 τ。

参照 [SOFT_HARD_RELAXATION.md](SOFT_HARD_RELAXATION.md) §45 的历史分析时，
也需收紧“Adam 抵消缩放，所以 postnorm 不可能与 β1 耦合”的推论：标量缩放
抵消不涵盖随输入、mask 变化的各向异性 Jacobian。当前 optimizer 的 betas
仍为 (.9,.95)，并未启用历史选择性 β1=.95 实验。

## 3. 尾部不是简单的扰动滤波器，postnorm 不能只看绝对尺度

把最后一个 DISM 从全 hard 切为全 soft，其他层保持 hard。middle 在两个固定
批次上，L8 输出的相对扰动 RMS 约 1.48%，经 L9/L10 到 L11 变为约 9.7–9.8%；
绝对扰动 RMS 从约 .0334 增为 .1106。尾部对该扰动敏感，未将其简单压掉。
这是有限 hard/soft 切换，不是无穷小 Jacobian、梯度 SNR 或纯噪声传播测量。

例如 batch 0 的 L11 attention norm 前后：绝对扰动 .04706 → .01044，
相对扰动 2.21% → 10.31%；MLP norm 前后：绝对扰动 .01598 → .11051，
相对扰动 11.14% → 9.72%。norm 改变基线尺度，绝对和相对指标会给出不同印象。
不能据此单独宣布某个 norm 放大噪声；需要连同残差、gamma 和任务相关方向看。
hidden 扰动更大也不等于 loss 更差：新布局 hard/soft loss gap 反而更小。

冻结权重，仅将 L11 或 L9–11 的 attention/MLP 残差分支分别乘 .75/1.25，
所有 12 个验证消融均变差。例如 L11 attention 为 3.02867/3.02617，L11 MLP
为 3.06086/3.05257，基线 3.01024。因此没有发现简单的推理期缩放修复。
这不排除重新训练的可学习缩放，但不支持直接修改 postnorm 或固定尾部增益。

## 4. 检索功能仍然有效，未见全局符号塌缩

NIAH baseline：old present NLL 2.07573、margin .87996、gain 1.01250；
middle 为 1.90347、1.44818、1.55202。关闭 middle L7/L8，present NLL 分别
升到 2.66719/2.57657，gain 降到 .71460/.88340。尾部并未使 DISM 失效；
当前更像任务收益分配变化，而不是统一的记忆能力衰退。6 trials 不足以推广到
所有检索任务。

有效 target 位置的逐 head 标签熵统计未见普遍塌缩。middle 最早 DISM 的 Q
平均有效符号数较少（9.7 vs old 首个 DISM 22.0），其他层并不一致下降。
主要 LM 贡献层的 τ 中位数也不同（old L10 .033，middle L8 .181），但 τ 为
匹配链的加性 score，不能只凭中位数把差异解释为过度检索或确定修复方向。

## 5. planner 数值轨迹仍是混杂因素

复制 graph 捕获的不可变输入，以相同 hard mask 比较 event 与 quadratic
单算子反向：两模型、各六层的最大 relative L2，sq/sk/v 不超过 .002383
（约 .24%），τ 不超过 .000323（约 .033%）。没有发现该输入上的大幅局部偏差。
这不是 FP64 oracle，也不能排除长期累积效应。

整模型同输入、同 mask 的 eager 反向重复执行，本身就有可见差异：旧模型某些
梯度组约 1–2%，新模型部分 selector 组最高约 9%；较小的梯度范数会放大相对
指标。graph/planner 对照也存在百分比级组差异，前向 loss 相同。量化链式反向
及累加顺序的敏感性意味着不能仅凭一次全模型差值认定 planner 错误，也不能
声明后端因素已经排除。本次未改容差、kernel 或数值验收，也未新增 sanitizer
覆盖。

## 后续实验顺序：保留尾部计算窗口

1. **先隔离训练后端因素。** 同布局、同初始化和 dataloader、匹配采样计数，做
   planner off/on 的受控短训练；必要时先比较固定 batch 多步更新，记录每层
   梯度方向与参数偏移。已有 run 同时改了布局和后端，不能替代这个对照。
2. **优先测试 DISM 的直接 LM 监督。** 在 L8 输出加入小权重的辅助 LM loss，
   最终 head、三层尾部 GDN 和 postnorm 保留。检验早期 DISM 消融贡献是否增大、
   最终 hard/soft NLL 是否改善，同时监测 NIAH。辅助 head 是训练假设，可能
   引入额外 CE 成本及改变反向调度；必须重新测 planner 的 deadline/吞吐，不能
   预先承诺收益。不要直接用末层 readout 做“logit lens”并将未校准 loss 当成
   中间表征质量结论。
3. **再区分输入成熟度与尾部监督。** 若辅助监督无效，对照前置计算更充分且仍
   有尾部窗口的布局，或在固定布局上研究从 DISM 区域到末端的可学习 skip。
   前者会改变可用 planner 窗口，后者会改变梯度依赖；都需要质量/吞吐双测。
   不同时改 β1、τ、annealing、norm，否则难以定位原因。

这些实验尚未执行。当前证据优先支持改善 DISM 的 LM 利用与监督，不支持以
“新布局全局切换噪声更大”为理由直接提高 β1，也不支持移除尾部作为唯一修复。

## 复现与产物

四个诊断工具位于 `../tools/diagnose_layer_layout.py`、
`../tools/diagnose_layout_tail.py`、`../tools/diagnose_layout_norm.py`、
`../tools/diagnose_layout_routes.py`。使用 conda blkw，在仓库根目录设置
`PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" OMP_NUM_THREADS=1 DISM_HARD_PLANNER=0`。
所有输出限定在 gitignored `dism_v3/build/`；参数通过 `--help` 查看。
tail/norm/routes 针对此次 step 22888 和 12 层配置，不是通用评估入口。

实际产物位于 `dism_v3/build/layout_diagnosis/`：

- `paths.json`：两个完整 checkpoint 目录。
- `{old,middle}.json`：forward、32-draw 噪声、backend 对照；通过
  `--noise-only --draws 128` 增补 h=.95 的 `noise_128`。
- `{old,middle}_tail.json`：本地验证消融与 NIAH。
- `{old,middle}_norm.json`：norm 前后有限切换扰动。
- `{old,middle}_routes.json`、`{old,middle}_symbols.json`：同输入反向对照，
  后者还包括标签分布和 τ。

主探针需传 `--checkpoint-dir`、`--batch-dir`、`--output`；两模型用同一
`--batch-dir`，默认 rows=4、step=22888。首次 full run 后可用
`--backend-only` 重测后端。norm/routes 同样接受上述三个路径参数；tail 只需
checkpoint/output，默认 validation-batches=8。新结果使用新文件名，避免覆盖。
初次 graph 探针曾保留 autograd 引用，初次验证探针曾遇到 loader buffer 复用；
分别释放引用、clone 数据后完整重跑。这些是探针问题，不是生产训练故障。

## 下游扩展评估（2026-10-23，28/28 完成）

使用 `~/downstream-eval` 的本地任务和上述两份最终 checkpoint：ARC-Challenge、
ARC-Easy、HellaSwag、WinoGrande、PIQA、LAMBADA、SWDE、FDA、GovReport、PG19。
按用户要求跳过 Qasper、SQuAD。所有任务比较 2048/4096 上下文，GovReport 和
PG19 另加 8192。全量样本、零样本提示、全 hard、固定 true direction。
GovReport 在这里是长文 rolling perplexity，不是摘要生成任务；rolling windows
不跨窗口传递状态，每个 token 只计分一次。

入口为 `~/downstream-eval/run_layout_comparison.py`，输出隔离在
`~/downstream-eval/results/layout_20261023/`。`state.json`/`sweep.log` 跟踪进度，
`comparison.md`/`comparison.json` 随配对任务完成而更新。每个任务保存样本结果、
checkpoint/adapter/task hashes 和运行协议。汇总器核对 doc/prompt/target hashes，
用 2,000 次配对文档 bootstrap 给出差值区间；不涵盖训练 seed 不确定性，也未做
多重比较校正。runner 不启动训练、不写 checkpoint、不下载数据。

独立的 `adapter_layout.py` 补充 strict assign=True 权重加载、MXFP8 cache 刷新
与 tokenizer fingerprint 校验，保留历史 adapter。原 smoke 的单样本/双样本
对照在两模型上都超过原 .08 LL 门限（约 .14）；直接模型 CE 精确复现了该差异，
即观察到模型对物理 batch shape 的敏感性，其 kernel 级来源尚未隔离。新协议固定
物理 batch=4，末批补独立且不计分的 packed 行，两个模型均通过原门限的 smoke，
没有放宽容差。这不证明所有 padded length/GEMM shape 下都不变；与历史评估结果
横比时必须考虑协议差异。详细原始数值记录在结果目录的 `PROTOCOL.md`。

PG19 初次 8K 执行因多形状 MXFP8 工作区累积 OOM。续跑使用独立的
`adapter_layout_bounded.py`，仅在 eager inference 输入形状变化时清除
FFN `_forward_states` 和 fused RMSNorm `_forward_plans`，保持权重缓存、
batch=4、上下文和计分不变。切换长度后回到 8K，似然完全一致，显存分配返回
约 7.77 GB。已完成的 2K MC/LAMBADA 保留原适配器及 provenance，其余任务记录
新适配器 hash。所有 28 项现已完成，结果协议均核对为 full dataset、全 hard、
physical batch=4；没有降到短上下文或跳过样本来规避 OOM。

### 下游结果

以下 old=`ggggggdddddd`，middle=`gggddddddggg`；选择题/contains 为百分数。
短任务 2K 与 4K 得分完全相同，不能作为两次独立重复实验。

| 任务 / 指标 | old | middle | 差值（百分点） |
| --- | ---: | ---: | ---: |
| ARC-Challenge / acc_norm | 21.08 | 22.01 | +.94 |
| ARC-Easy / acc | 35.61 | 36.20 | +.59 |
| HellaSwag / acc_norm | 28.50 | 28.59 | +.09 |
| WinoGrande / acc | 50.36 | 49.88 | −.47 |
| PIQA / acc | 60.66 | 60.17 | −.49 |
| LAMBADA / acc | 16.30 | 12.73 | −3.57 |
| SWDE 2K / contains | 15.03 | 13.86 | −1.17 |
| SWDE 4K / contains | 15.30 | 13.95 | −1.35 |
| FDA 2K / contains | 7.44 | 8.89 | +1.45 |
| FDA 4K / contains | 11.98 | 13.52 | +1.54 |

LAMBADA 有 5,153 个样本，old-only correct=324、middle-only correct=140、
both correct=516；accuracy 差值的配对 95% 区间为 [−4.41,−2.72] pp。
PPL 为 284.54 → 346.23，目标续写 NLL 差值 +.19624，区间 [.15641,.23726]。
上述其他表格项的配对区间均跨零。HellaSwag 未归一化 acc 有小幅下降
（−.44 pp，区间 [−.86,−.04]），但 acc_norm 持平，且未做多重比较校正；
不能概括为常识任务普遍回退。FDA 上升、SWDE 下降目前也只宜视为趋势。

| 长文任务 | context | old token PPL | middle token PPL | middle−old NLL/token |
| --- | ---: | ---: | ---: | ---: |
| PG19 | 2048 | 39.41582 | 40.79366 | +.034359 |
| PG19 | 4096 | 37.90888 | 39.24302 | +.034588 |
| PG19 | 8192 | 36.98868 | 38.34169 | +.035926 |
| GovReport | 2048 | 19.02828 | 19.65665 | +.032489 |
| GovReport | 4096 | 17.83840 | 18.46968 | +.034777 |
| GovReport | 8192 | 17.16678 | 17.81511 | +.037071 |

PG19 共 100 篇、11,337,655 tokens，GovReport 共 973 篇、12,567,429 tokens；
上下文长度之间也核对了相同文档及 token 计数。所有长文 NLL 差值的配对区间
均高于零。8K 下新布局在 PG19 的 99/100 篇、GovReport 的 960/973 篇上更差，
并非由极少数长文拖累。

另做 5,000 次配对文档 bootstrap（seed 20261023），比较 2K→8K 的 NLL 降幅：

| 任务 | old 降幅 | middle 降幅 | middle−old 降幅 | 配对 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| PG19 | .063555 | .061989 | −.001567 | [−.002457,−.000707] |
| GovReport | .102950 | .098369 | −.004581 | [−.005098,−.004050] |

新布局仍保留约 97.5% / 95.5% 的上下文扩展收益。相较整体 .03–.04 NLL 的
回退，长度增加带来的额外差距较小；表现是整体预测偏弱、长上下文收益稍弱，
而非扩到 8K 后突然崩溃。补充统计在结果目录 `context_gain_analysis.json`。

结合前述消融，这进一步支持优先检查“DISM 表征如何被用于最终词预测”：
合成 NIAH 更好并未转化为 LAMBADA 或自然长文 LM 更好，不能直接推广为一般
长上下文能力增强。仍不能仅凭这两次训练确定是布局、planner 数值轨迹还是
训练随机性导致；bootstrap 只反映这些固定 checkpoint 的评估样本不确定性。
