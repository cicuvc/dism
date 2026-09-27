# 前向精度回归：2026-09-07

平台：RTX 5090 / sm120a，CUDA 13.1、PyTorch 2.13.0+cu130，conda blkw。
本次仅添加测试与报告工具，未修改 CUDA kernel 或用户 reference。rtau 上限按自然对数域 `ln(D)`，
不是 `log2(D)`；测试将上限以内与超上限压力结果分开统计。

## 结论

接入实际 embedding 之前完整运行 **301 passed / 28 failed**（329项）：原有188项全部通过；新增141项中113通过、28失败。
失败全部是纯 soft 输出对**未量化的 FP32 插值 oracle**未达到原输出阈值：
14个在 rtau≤ln(D) 范围内，14个在超上限压力范围。相同 BF16 插值输入的 core 输出/L2 检查全部通过。
这不是全链路精度已验收：这些失败保留为普通失败，未 xfail、跳过或放宽容差。
因此 `scripts/check_dism_v2.sh` 当前会在数值阶段返回非零，并不会继续执行其后的 sanitizer 阶段。

### 上限内最差的 FP32 插值对照用例

N=8193，D=64，DV=128，B=H=1，rtau=ln(64)，hard_prob=0，随机选择结果为 k_from_q。

| 对照 | 输出最大绝对误差 | 输出 RMSE | 输出相对 L2 |
| --- | ---: | ---: | ---: |
| CUDA vs 相同 BF16 插值的 reference | 0.0154972 | 0.00136812 | 0.00152166 |
| CUDA vs FP32 插值的 reference | 0.132521 | 0.0109385 | 0.0121738 |
| BF16 插值 reference vs FP32 插值 reference | 0.132779 | 0.0109260 | 0.0121599 |

最后一行没有执行 CUDA core，仅比较两次 reference；结果说明该用例中插值转 BF16 后的误差累积
已足以解释大部分偏差。不能将其归因为 warp scan、随机方向或近似 LSE；当前 affine 仍用完整 log1p/exp2。
本节尚未包含实际 embedding kernel（后续接入结果见文末），也没有证明所有训练输入分布都会出现同等偏差。

上限内全部随机用例的 core 输出最大绝对误差为0.0155563，最大 case RMSE为0.00148185；
二者都仍满足 `abs_error ≤ 0.008 + 0.012*abs(reference)`，不是“最大绝对误差≤0.008”。
core L2 对 FP32 reference 的最大绝对误差0.00415039；这里 reference 的逐行 FP32 递推本身也有累计舍入，
不能将该差值完全视为 CUDA 相对精确实数运算的误差。

### 输出余弦相似度

rtau≤ln(D) 的所有用例中，各指标的最小值如下；同一行不同列的最差值不一定来自同一个用例。

| 对照 | 整体余弦 | 逐行余弦均值 | 最差行余弦 | 行余弦1%分位数 |
| --- | ---: | ---: | ---: | ---: |
| CUDA vs 相同 BF16 插值 reference | 0.999996730 | 0.999996844 | 0.999992637 | 0.999994165 |
| CUDA vs FP32 插值 reference | 0.999926263 | 0.999901481 | 0.998288881 | 0.999063587 |
| BF16 插值 reference vs FP32 插值 reference | 0.999926313 | 0.999901399 | 0.998284808 | 0.999060935 |
| 长匹配链 CUDA vs FP64 闭式解 | 0.999997907 | 0.999997951 | 0.999995501 | 0.999996776 |

FP32 插值对照四项最差值均来自上述 N=8193、D=64、rtau=ln(64) 用例。
余弦由 FP64 计算：整体先展平所有行，逐行以 DV 为向量维；两边均为零的 fallback 行定义为1，
仅一边为零定义为0，有限非零小向量不使用固定 epsilon 扭曲方向。
加入零向量、反向向量及极小非零向量的指标单元测试。
余弦目前作为报告指标，不另设验收阈值；较高余弦不意味着原逐元素误差断言已通过，也不检验幅值一致性。

### 长匹配链

固定同一标签、hard_prob=1，使用 reference 递推的 FP64 闭式解：
`exp(W[i,j]) = sum(exp(t*rtau), t=1..j+1)`，j≤i。
CPU FP64 稳定 prefix softmax 包含零 fallback；小 N 与 dism_ref 的直接递推交叉验证。
这样长链对照不依赖 FP32 逐行 reference 的舍入，也不复用 GLX 的 scan/checkpoint 算法。

- N=1025/8193，D=32、DV=64，rtau=-16/-1/-0.001/0/0.001/1/ln(32)/16；全部16例通过。
- 上限内：输出最差 max_abs=0.0142835、最差 case RMSE=0.00161150；
  L2 最差 max_abs=0.0368173，已解析底边最差 max_abs=0.0366138（后二者出现在 rtau=1）。
- 超上限 rtau=16、N=8193：L2 max_abs=0.586392，底边 max_abs=0.587559；虽满足相对容差，
  仍应记录长程绝对误差，不解释为高绝对精度。跨 chunk passing 将继续保留完整 LSE。
- 很负的 rtau 会令 normalizer 接近0，其相对误差可能显得较大，应结合绝对误差解释。

## 测试覆盖与方法

新增 `tests/test_dism_v2_precision.py`：

- 123个随机输入用例：N=1025覆盖全部九种 D/DV、两个固定方向、soft/混合、rtau=-8/8/ln(D)；
  N=2049/4097/8193覆盖随机方向、若干 soft/hard/混合场景及上限/压力值。长 N 不覆盖全部九种维度。
- 16个长匹配链用例，另加1个小 N 闭式解对 reference 的验证、1个余弦指标单元测试。
- 随机数据固定可重现种子；未做大规模多种子分布统计。
- 三路对照分别记录每例 max_abs、RMSE、相对 L2，以及输出整体/逐行余弦；RNG 行 mask 由独立 CPU Philox oracle 生成，仅用于测试。
- FP32 torch 插值与显式 BF16 转换的插值分别喂给 reference。两路的 v 均转 FP32，
  因而 oracle 的 PV/输出是 FP32，不让 reference 自身 BF16 weight/PV 量化掩盖 core 的输出量化误差。
  q/k/v 的原始输入仍是 BF16。禁用 oracle 的 TF32，测试后恢复配置。
- 输出阈值保持 atol=.008、rtol=.012；core L2 保持2e-5；长链边界保持3e-5。
  未为 FP32 插值端到端误差另设较宽预算。FP32 插值 L2 目前记录数值，不设独立验收断言。
- 与已有 core 测试互补：本模块的长链只检查完整 checkpoint 底边；部分尾 checkpoint 的
  identity 及局部 affine 两分量由原有 double 递推测试检查。

## 复现与机器可读报告

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q \
    tests/test_dism_v2_precision.py --junitxml=/tmp/dism-precision.xml -o junit_family=legacy
/home/cicuvc/miniconda3/envs/blkw/bin/python scripts/summarize_dism_precision.py \
    /tmp/dism-precision.xml --scope bounded
```

`--scope stress` 查看超上限压力结果，默认 all。JUnit 中每例 `precision` 属性保存数值与各检查是否通过，
即使测试失败也能汇总。完整校验脚本同样写出 JUnit；本次完整运行原始报告为 `/tmp/dism-precision-cosine-final.xml`，
临时文件可能被清理，以上数值与生成方法已在仓库内保留。

本次未重新对所有新增压力用例执行 racecheck/synccheck；原有106项 core 的三类 sanitizer 记录仍见 README。
另对新增 N=8193、rtau=ln(32) 长匹配链运行 memcheck：1 passed、0 errors（仅 instrument Dism kernels）。
不将“已有 kernel sanitizer 通过”表述成“所有新增长序列压力用例 sanitizer 均通过”。

## 实际 emb_kernel 接入

新增 `tests/test_dism_v2_embedding_precision.py`，直接导入
`dism_v2.emb_kernel.EmbInterpFunction`，使用 InterpolationResult 的实际八项顺序；
仅将返回的 int32 labels 无损转 int64 以适配 core。没有修改用户 emb_kernel.py 或 dism_ref.py。
本次只验证前向，不触发或修复已知的 embedding backward 多 vocab block 写竞争。

64项：N=1025、全部九种 D/DV、两方向、hard_prob=0/.37/1、V=31，共54项；
N=2049/4097/8193、D=64/DV=128、随机方向、soft/混合，共6项；
V=64/65/129/257、N=257、B=H=2 的多 vocab block/尾部共4项。所有 rtau=ln(D)。
结果48通过、16失败；失败仍全部是对原始 FP32 插值的输出容差，不是同 embedding 输入的 core 输出/L2。
其中15个纯 soft、1个混合用例（N=1025,D=128,DV=64,q_from_k,p=.37）；因此混合行只是本组测试更稳，
不是精度保证。所有样本的 q/k label mismatch 为0，q/k LSE 最大绝对误差均9.53674e-7。

### 与此前最差长序列样本直接比较

相同输入种子、N=8193、D=64、DV=128、V=31、rtau=ln64、p=0、k_from_q。
各行都对照 torch FP32 插值 + FP32 reference PV/输出：

| 路径 | 输出 max_abs | RMSE | 整体余弦 | 最差行余弦 |
| --- | ---: | ---: | ---: | ---: |
| torch 插值转 BF16 + CUDA core（此前） | 0.132521 | 0.0109385 | 0.999926263 | 0.998288881 |
| 实际 emb_kernel + CUDA core | 0.120802 | 0.00672747 | 0.999972096 | 0.998597008 |
| 实际 emb_kernel + reference | 0.124151 | 0.00685661 | 0.999970946 | 0.998502997 |
| emb_kernel 仅改 FP32 输出缓冲 + reference（诊断） | 0.0360265 | 0.00241493 | 0.999996391 | 0.999862454 |

实际 emb_kernel 的两个插值输出都是 BF16，但并不等于 torch FP32 插值再转 BF16：
`emb_fwd` 内部也有 `tl.dot(s_qs.to(k_embs.dtype), k_embs)` 与对应 q 路径，
即 softmax 权重先转 BF16 再做 Tensor Core 插值 GEMM。最终输出 BF16 写回是另一处量化。
本次样本的内部误差与最终舍入可能部分抵消；不能由这一例推断真实 embedding kernel 总是更准。

诊断调用复用未修改的 `emb_fwd` Triton 函数，仅分配 FP32 输出缓冲，其他 q/k/vocab dtype、
tile/参数不变；内部 softmax 权重仍转 BF16。它的输出只送入 reference，不送入当前只接受 BF16 的 core。
由此可见，提高输出存储精度可明显减小本例误差，但不能消除内部插值计算误差；这不是生产 FP32 插值支持。
没有将两段 RMSE 当成可线性相加的误差预算。

实际 emb 输入的同输入 core 对照在全部64项中，输出 max_abs 最大0.0155821，最大 case RMSE=0.00148090，
整体余弦最低0.999997285、最差行余弦0.999992526。对未量化 FP32 插值的完整路径，
整体余弦最低0.999943652，最差行0.998597008；这些指标的极值不全来自上表同一个样本。

报告复现：

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q \
    tests/test_dism_v2_embedding_precision.py --junitxml=/tmp/emb-precision.xml -o junit_family=legacy
/home/cicuvc/miniconda3/envs/blkw/bin/python scripts/summarize_dism_precision.py \
    /tmp/emb-precision.xml --property embedding_precision
```

单模块原始报告 `/tmp/dism-embedding-precision-final.xml`。
连同此前套件完整运行393项：349通过、44失败，报告 `/tmp/dism-all-embedding-precision.xml`。
另对 B=H=2、N=257、V=65 尾部测试执行 memcheck（只 instrument emb_fwd，包括诊断变体）：
1 passed、0 errors。未对所有 embedding 用例执行 sanitizer，未验证其反向、性能或 top-1 人工构造 tie。
