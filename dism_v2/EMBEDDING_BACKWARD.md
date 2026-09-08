# CUDA embedding backward：WS数值验证与配置搜索

最新状态（2026-09-08）：用户授权不再修复spill，20个可运行WS配置已完成数值与
sanitizer验证，并按实测GPU时间比较。最新结果见文末“接受spill后的配置搜索”。
低层默认仍为单warp诊断路径，未自动选择配置；autograd反向默认Triton，现可显式选择CUDA
配对或对称WS，最终端到端验收见`AUTOGRAD.md`。
以下暂停记录是开发历史，不代表当前阻塞。

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

## 对称双warpgroup试验（编译后因新spill暂停）

按用户要求新增`vocab_symmetric=True`（须同时warp_specialized=True），旧配对版本保留。
WG0/1分别负责不同的64个词表项，CTA共128项，每warp负责16项的dkey和dvalue，
两组运行完全相同的代码。无P/G shared mailbox、无配对warp ready/free；
仍有三个输入tile双缓冲的producer/consumer同步，以及初始化/退出CTA同步。
扫描步长D32/64为64，D128为32，producer40、两个compute组各232预算。

每warp常驻两个FP32 16×D accumulator与两张BF16词表片段；
每步先重算完整分支P，用BF16 P累积value贡献，再原地形成G、累积key梯度，
最后重算LSE-only分支并累积value词表的key贡献。P完全留在本warp，不重复完整分支score。
无atomic、无split，仍逐batch/token扫描后独占写回两个梯度。

首轮完整CTA编译：

| D / token步长 | stack B | spill store B | spill load B |
|---|---:|---:|---:|
| 32 / 64 | 32 | 68 | 56 |
| 64 / 64 | 152 | 444 | 296 |
| 128 / 32 | 248 | 732 | 488 |

三个实例均静态168寄存器，原生TMA/setmaxnreg生效，SASS无CALL。
共享初始化空间按8warp各两张词表tile分配，转入寄存器后复用为输入ring；
D128时初始化比输入ring更大，仍在shared容量内。
按新增spill先汇报约定暂停，未调参，也未做数值/sanitizer/性能验证；不能声称比配对版慢或快。
新增spill未加入codegen豁免。实现见`csrc/embedding_bwd_symmetric.cuh`，
日志`/tmp/dism-emb-bwd-symmetric-build.log`、`/tmp/dism-emb-bwd-symmetric.sass`。

## 缩小token步长与D64 shared常驻词表

后续按用户要求缩小对称词表扫描步长；token梯度kernel及配对词表版本不改：

| D | 对称词表token步长 | stack/store/load B |
|---|---|---|
| 32 | 64→32 | 32/68/56→0/0/0 |
| 64 | 64→32→16 | 152/444/296→32/80/64→8/8/16 |
| 128 | 32→16 | 248/732/488→208/624/416 |

当前步长为D32=32、D64/128=16。D32/64两个方向、N1/17/65、V1/65/129、B2/H2
共36项同状态检查通过。用户已明确D128 spill暂不处理；此次不验证/优化D128。

随后仅D64改为shared常驻两张词表。8warp各16行的E_key/E_value初始化后不再被ring覆盖，
score/dP/LSE-only score通过`dot_shared_vocab`每次读取32-feature片段。
不再把两张完整16×64 BF16词表tile常驻寄存器；两个FP32梯度accumulator保持不变。
总动态shared从32896B增到45184B，另有1024B静态shared；数据部分为32KiB词表+12KiB输入ring。

结果：D64仍8B stack、spill store/load 8/16B，原来的64位vb标量保存/读回没有消除。
nvdisasm对已分配SASS的最大live GPR从198降到188（不是未spill时的理论需求或性能指标）。
静态LDSM从32条增到36条，HMMA仍48条；两次STL、两次LDL，原生TMA/setmaxnreg保留，
无CALL。D32和D128编译资源不变。未计时，不能称为性能提升。

shared版D32/64的36项同状态回归通过；D64的18项分别通过memcheck/racecheck/synccheck，
零错误/零hazard。新增`test_symmetric_shared64_codegen`约束已知local访问不增加。
这不是FP32端到端精度验收，CUDA embedding backward仍未接入autograd。
日志`/tmp/dism-emb-bwd-shared64-{build,tests,memcheck,racecheck,synccheck}.log`、
`/tmp/dism-emb-bwd-shared64.sass`；寄存器分析在`/tmp/dism-emb-shared64-jL8EcU`。

## 接受spill后的配置搜索（2026-09-08）

不再进行spill修复。新增显式`vocab_token_step=16/32/64`及`vocab_shared=True/False`，
只改变词表梯度扫描；token梯度kernel仍D32/64步长64、D128步长32。
配对和对称路径都保留，producer/consumer仍40/232/232，三个kernel完成四类梯度。
`vocab_shared=True`仅支持对称D64。默认参数保持上一轮选择，不添加未经训练场景验证的自动dispatch。

共20个WS配置：D32六个，D64九个，D128五个；另比较三个D的单warp基线及Triton wrapper。
唯一容量排除项是配对D128/T64：动态shared114944B，另有1024B静态shared，超过sm120
101376B opt-in容量，接口明确拒绝且不实例化。其余配置不因spill排除。
临时global scratch（不含输出）为`(2D+12)*B*H*N`字节：BF16 U、FP32 delta及两路LSE2。

### 数值与同步

- `tests/test_dism_v2_embedding_backward.py`：**394通过、6个既有普通失败**，没有放宽容差或xfail。
- 全20配置×两个方向×五组N/V（1/1、17/65、65/129、257/257、1025/65），
  B2/H2的200项同状态BF16舍入oracle检查通过。
- 全20配置×两个方向×V65/129，N65/B2/H2的80项Triton与独立FP32 autograd reference
  检查通过。四类梯度中最坏relative L2为0.002662，最小cosine为0.99999646，
  最大绝对误差0.006746；这些误差指标仅对应这80个常规用例，不包含V1或长序列。
- 新增6项N4096/V1024配对推荐配置及8项N1024/V8192候选配置的同状态检查，全通过。
- 6个失败仍是V1、三种D及两个方向对未量化FP32 reference的词表梯度误差；
  先与Triton的比较通过。量化路径保留，不归因于本轮布局或spill。
- 全20配置在B2/H2/N65/V129、q_from_k下分别通过memcheck/racecheck/synccheck，
  每类20项，零error/hazard。覆盖完整输入槽、尾槽、跨batch复用及部分有效词表CTA。
- 全部最终SASS无CALL（含REL/ABS），全部WS保留UTMALDG/USETMAXREG。
  codegen断言不再要求零spill，但仍严格检查CALL、原生TMA/寄存器重分配与换底FMUL上限。

日志`/tmp/dism-emb-bwd-sweep-{tests,memcheck,racecheck,synccheck}.log`，
误差属性`/tmp/dism-emb-bwd-sweep-results.xml`。本轮只验证embedding反向，不声称已接入CUDA全链路autograd。

### 测量方法与结果

RTX5090、170 SM，CUDA13.1编译、torch2.13.0+cu130、BF16输入/FP32上游及梯度。
CUPTI/torch.profiler GPU event时间，热输入、不锁频；每候选5次warmup、20次计时、
三轮乱序。表中完整反向时间是各kernel中位数之和再取三轮中位数，排除Python/分配/发射间隙，
不是API latency或端到端Dism时间。Triton列包含wrapper补零及preprocess/fused backward，
它仍执行通用双分支工作，不是控制相同指令量的消融。未作冷cache、HBM流量或因果瓶颈证明。

完整182行配置结果（包含三轮范围、词表阶段时间、资源及CTA数）保存在
[`benchmarks/embedding_backward_sm120a.csv`](benchmarks/embedding_backward_sm120a.csv)。
以下时间单位µs，配对/对称列均选各家词表阶段最好的候选；括号为token步长。

| B/H/N/V | D | 配对词表 | 对称词表 | 最佳CUDA完整反向 | Triton完整反向 |
|---|---:|---:|---:|---:|---:|
| 1/4/4096/1024 | 32 | 175.63 (64) | 286.88 (reg32) | 226.99 | 361.15 |
| 1/4/4096/1024 | 64 | 231.41 (32) | 369.53 (shared16) | 316.05 | 570.25 |
| 1/4/4096/1024 | 128 | 362.69 (32) | 801.31 (reg16) | 526.40 | 935.03 |
| 1/8/1024/8192 | 32 | 314.83 (64) | 295.50 (reg32) | 451.58 | 582.55 |
| 1/8/1024/8192 | 64 | 432.93 (32) | 402.88 (reg32) | 665.29 | 1271.82 |
| 1/8/1024/8192 | 128 | 749.36 (32) | 909.96 (reg32) | 1276.42 | 2442.21 |
| 1/32/1024/1024 | 32 | 181.53 (64) | 152.00 (reg32) | 251.71 | 244.27 |
| 1/32/1024/1024 | 64 | 247.65 (32) | 204.78 (shared16) | 370.08 | 608.32 |
| 1/32/1024/1024 | 128 | 413.69 (32) | 459.17 (reg32) | 758.43 | 1114.14 |

短形状B1/H4/N65/V129三种D均配对T16最好，完整CUDA为23.06/37.68/65.90µs，
仍慢于Triton13.18/26.02/45.38µs。不能统一宣称CUDA胜出。

配置建议仅限本批测量：

- CTA较少、长N：配对D32/T64，D64/T32或T64近似打平，D128/T32。
  短N尾块倾向T16；B2/N1025等非对齐场景可使较小步长稍有优势。
- CTA充足时，D32对称reg/T32、D64对称reg/T32或shared/T16更好。
  H4/V1024时对称仅32 CTA、配对64；H32/V1024时变为256/512。
  CTA数量差异是可能影响因素，不把上述对照当作已证实的occupancy瓶颈。
- D128本批全部形状仍配对领先，不因对称取消通信就默认它更快。
- D64 paired/T64有8B stack，而T32为0B；在H4/N4096/V1024词表阶段231.44 vs231.41µs，
  几乎没有差异。D64 symmetric reg/T32的32B stack在大V下仍能赢8B stack的shared/T16，
  402.88 vs404.72µs；H32/V1024则shared16小胜，204.78 vs206.57µs。保留两者，不按spill排序。

本批WS静态寄存器均168（运行时角色预算不同）；对称reg/T16,T32,T64的stack B分别为
D32:0/0/32、D64:8/32/152、D128:208/248/440；D64 shared为8/24/160。
配对D32全部0，D64为0/0/8，D128 T16/T32均8。stack大小不是每次迭代local流量，
不能单独推导性能。资源CSV的动态shared不含另外1024B静态shared。

复现（conda blkw；sanitizer和benchmark不要并发运行）：

```bash
python -m pytest -q tests/test_dism_v2_embedding_backward.py
compute-sanitizer --tool racecheck --error-exitcode 99 python -m pytest -q tests/test_dism_v2_embedding_backward.py -k 'test_configurations and q_from_k and 65-129'
python -m dism_v2.benchmark_embedding_backward --repeats 20 --rounds 3
python -m dism_v2.benchmark_embedding_backward --repeats 20 --rounds 3 --shapes 1,8,1024,8192 1,32,1024,1024
```

memcheck/synccheck替换同一sanitizer命令的tool参数。默认五组形状，补测两组大CTA形状；
`--dims`可缩小维度范围。原始CUPTI日志在`/tmp/dism-emb-bwd-sweep{,-wide}-bench.jsonl`。
