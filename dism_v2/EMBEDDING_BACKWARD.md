# CUDA embedding backward：基线已验证，WS因新增spill暂停

用户批准继续后，单warp基线已完成数值/sanitizer验证。随后写入WS和P mailbox，
编译出现新的D64/D128 spill，按约定再次暂停；未进行资源调优。
WS尚未运行数值或同步测试；生产embedding backward仍为Triton，未接入autograd。

接口`dism_v2.embedding.backward`仅支持Dism的实际稀疏上游组合：
选定插值的FP32 U、另一分支LSE的FP32梯度。返回embedding贡献的FP32
dq/dk/dq_voc/dk_voc；方向已由调用方重放确定，不产生RNG。

当前五次launch：FP32 delta/BF16 U packing、完整分支token梯度、LSE-only token梯度、
key词表梯度、value词表梯度。后两者每warp独占16个词表项，循环所有batch/token，无atomic。
value梯度分支暂时独立重算完整分支的P；后续计划通过配对warp mailbox消除此重算。
所有MMA按32-feature分段，D32/64扫描步长64、D128扫描步长32。
delta用保存的BF16插值和原始FP32 U计算，之后U才转换BF16。P/G转换BF16供MMA，
GEMM乘sm_scale在FP32累积时进行，不重复乘LOG2E。

RTX5090、sm120a、CUDA13.1、blkw编译结果：

| D | token full | token LSE-only | vocab key | vocab value |
|---|---:|---:|---:|---:|
| 32 | 107 | 102 | 116 | 122 |
| 64 | 166 | 156 | 168 | 185 |
| 128 | 245 | 174 | 245 | 255 + spill |

表内为寄存器/thread。D128 vocab value实例：64B stack、60B spill stores、60B spill loads。
其他实例和preprocess零spill；SASS全部无CALL。D128两个245寄存器实例即使单warp零spill，
也不能声称接入WS的232预算后仍零spill。

构建日志`/tmp/dism-emb-bwd-build.log`，SASS`/tmp/dism-emb-bwd.sass`。
`tests/test_dism_v2_embedding_backward.py`：在单warp基线版本运行67项，61通过/6普通失败。
48项同状态BF16 MMA oracle全部通过；另18项全部与Triton反向一致，
但其中6项V1对纯FP32 oracle有量化精度失败（D三种、两个方向），不设xfail或放宽容差。
单warpcodegen无CALL；用户接受的D128 value实例允许local，其余仍要求零local。
memcheck/racecheck/synccheck分别运行48项同状态测试，全部通过，零错误/零hazard。
日志`/tmp/dism-emb-bwd-baseline.{log,xml}`、`/tmp/dism-emb-bwd-{memcheck,racecheck,synccheck}.log`。

## WS新增实现与待确认资源

低层backward增加显式`warp_specialized=True`，默认仍是单warp基线。
WS三次launch：preprocess、token_ws、vocabulary_ws。12warp、dec40/inc232，
TMA双缓冲和guarded尾部。vocabulary每CTA处理64个词表项，warp0–3积累dkey，
warp4–7积累dvalue；每对16词表项的BF16 P通过双槽shared mailbox传递。
完整分支先发布P，再原地计算G；LSE-only组先算自己的key贡献，再消费P累积value项。
所有batch都由同一CTA顺序处理，当前无split或atomic。没有重复score来代替P传递。

完整WS编译资源（静态寄存器均168、动态角色预算40/232）：

| D | token stack / spill store / load B | vocab stack / spill store / load B |
|---|---|---|
| 32 | 0 / 0 / 0 | 0 / 0 / 0 |
| 64 | 0 / 0 / 0 | 8 / 8 / 16 |
| 128 | 16 / 20 / 24 | 64 / 72 / 92 |

全部SASS无CALL，存在原生UTMALDG和USETMAXREG。新增spill未加入测试豁免，
当前codegen零local断言会报告它们。尚未运行WS，不能声称P mailbox布局/同步已验证。
构建`/tmp/dism-emb-bwd-ws-build.log`，SASS`/tmp/dism-emb-bwd-ws.sass`。
等待用户决定保留这些新增spill继续验证，或先处理资源；尚未完成CUDA embedding autograd。

## WG预算转移对照

按用户建议，仅调整词表WS：尝试WG0=248、WG1=216、producer=40，
CTA总预算仍为128×(248+216+40)=64512。为避免低预算汇合影响编译，
将两个consumer分别放入长期分支，强制内联各自的`vocab_consume<FULL>`。
随后保持同样的长期分支结构，恢复232/232作为受控对照。

| consumer结构 / 预算 | D32 stack/store/load B | D64 | D128 |
|---|---|---|---|
| 原共享循环，232/232 | 0/0/0 | 8/8/16 | 64/72/92 |
| 长期分支，248/216 | 0/0/0 | 8/8/32 | 8/8/32 |
| 长期分支，232/232 | 0/0/0 | 8/8/32 | 8/8/32 |

两种长期分支版本资源相同，不能将D128改善归因为非对称预算。
当前保留长期分支+232/232；token WS未修改。248/216版本SASS确有
USETMAXREG的0xf8/0xd8申请；两种版本都无CALL。
248/216版本的剩余8B local在进入角色重分配之前就写入，两个consumer分支都有读回；
尚未追踪对应的源码标量，也未证明运行时开销，不能直接归因于G/P矩阵tile过大。
WS仍未进行数值/sanitizer/性能测试。

日志`/tmp/dism-emb-bwd-asym-build.log`、`/tmp/dism-emb-bwd-balanced-split-build.log`，
SASS对应`/tmp/dism-emb-bwd-asym.sass`和`/tmp/dism-emb-bwd-balanced-split.sass`。

## Score换底预乘

按用户建议，所有单warp/WS的P重算改为
`exp2(fmaf(score, scale2, -lse2))`，其中scale2在host按FP32 scale乘LOG2E，
两路lse2在现有preprocess中从保存的自然域LSE转换，额外FP32临时存储8×B×H×N字节。
公开LSE、delta和梯度G仍使用原自然域约定；梯度GEMM继续乘原scale，不乘scale2。

SASS受控对照：每个WS token/vocab kernel，D32/64的静态FMUL从256降到192，
D128从128降到96，FFMA数量不变。全部无CALL、原生TMA/setmaxnreg保留，
WS已有stack/spill未变；这不是性能测量。`test_lse_fma_codegen`固化FMUL上限回归。
单warp66项数值检查60通过/6个既有V1 FP32 oracle失败，48项同状态检查全通过；
WS仍未运行验证。日志`/tmp/dism-emb-bwd-fma-{build,tests}.log`、
`/tmp/dism-emb-bwd-fma.sass`。
