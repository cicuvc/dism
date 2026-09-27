# B3 dA/dB融合正确性路径（sm120a）

`backward.operand_gradient`当前接口返回已选方向core的FP32 `(dA,dB,dLSE,drtau)`。
新增dLSE/drtau已通过同G归约验证，直接reference仍有精度失败；见本文末尾。
不物化global W/P/E/G；本页为core低层接口，完整六输入autograd接线见[AUTOGRAD.md](AUTOGRAD.md)。
q_from_k时dB是插值操作数的梯度，不能直接称dK。

```python
from dism_v2.backward import delta, value_gradient, operand_gradient
dd = delta(dout, out)
dv, summary, g32 = value_gradient(
    a, b, dout, lse, tau, q_label, k_label, l2, edges,
    sm_scale=scale, rng_state=rng, v=v, delta=dd, warp_specialized=True,
)
da, db, dlse, dtau = operand_gradient(
    a, b, v, dout, lse, tau, q_label, k_label, l2, dd, edges, g32,
    sm_scale=scale, rng_state=rng,
)
```

A/B/V/LSE/labels/tau、edges、L2和rng必须来自同一次forward且未被修改。
G32的形状为`[B,H,paddedN/32,paddedN]`，来自同一delta的passing。
首版BF16 contiguous输入，D/DV独立32/64/128，fixed-length含尾部，sm120a。
无新RNG消费；不支持higher-order backward、CUDA Graph capture或deterministic模式。
dA的TMA atomic加法顺序依赖调度，不能要求逐位重放相同。

## 实现

每CTA一个warp负责32key，query64逆序，每tile依次处理高16、低16key。
高半块读下一G32边界，低半块接收高半块HState，分别维护right VState。
真实MMA→log-affine inclusive scan重建W₂；dP按DV32分段，tanh.approx计算sigmoid，
reverse inclusive scan重建自然logM梯度G。hard行参与scan，之后才屏蔽为Gsoft。

- Gsoft只写BF16 shared，恢复逻辑query列。
- dB的两个16-key FP32 accumulator跨循环保留，query按32feature段从permuted shared
  搬到逻辑顺序临时块再做MMA；结束后唯一写回。
- dA从Gsoft shared转置读取，16x32 MMA后写FP32 shared双槽，3D tensor map
  `[D,N,BH]`上的原生TMA reduce-add负责跨key CTA累加。输出真实N，不补齐尾行。
- dA缓冲由binding zero-init；所有writer fence+syncwarp发布，leader commit，
  复用槽前wait_group.read，退出前wait_group0。
- dA/dB仅乘sm_scale，不再乘LOG2E。

尚未接入12-warp WS/producer双缓冲；当前仅dA输出为双槽。D128 spill按用户授权保留，
资源表见IMPLEMENTATION_PLAN。D32/64零spill，所有实例无CALL；原生TMA/TANH保留。
不能用本版本资源直接推断12-warp版本可行性。

## 验证口径与已知精度问题

测试`tests/test_dism_v2_ab.py`分两层：

1. 87个融合数值用例：独立真实G重建及BF16 Gsoft的FP64 GEMM作为oracle，
   覆盖九D/DV、双方向、soft/mixed/hard、N1至2049、B=H=2的非对齐输入、重放和非默认stream。
   最大dA/dB绝对误差约5.22e-6/1.61e-5。另有codegen与API契约检查。
2. 60个直接reference autograd对照：保持BF16插值操作数值和LSE固定，将A/B作为独立变量，
   reference使用FP32 v和输出，不对BF16舍入求导。这不是q/k/vocab全链路。

54个常规reference维度用例通过；6个D64/DV128、N17/65/139、双方向、pure soft、
rtau=ln64用例在atol=.008/rtol=.02逐元素检查失败。失败保持普通测试失败，不放宽阈值。
六例dA最大relative L2=0.00548918，dB=0.00613192；最小cosine约0.99998509/0.99998121。
注意这些指标不能替代失败的逐元素检查。

额外delta诊断用reference FP32 O重算delta，再重新生成摘要/passing并运行B3（仅测试）：
六例dA/dB最大relative L2分别降至0.00225945/0.00260034，未完全消失。
因此saved BF16 O的delta是误差来源之一，但还存在Gsoft BF16转换、W/L2重算等差异，
不将全部误差归因于单个环节，不把诊断delta替换成生产语义。
具体逐例指标见`/tmp/dism-ab-delta-diagnostic.xml`。

初版资源日志`/tmp/dism-ab-first-build.log`及SASS `/tmp/dism-ab-first.sass`；
原全扩展零spill测试仍会因WS与D128 B3的已知spill失败，本轮没有弱化该旧断言。
新增B3专属codegen检查允许D128 spill，但仍要求其余六种零spill、所有实例无CALL和原生TMA。

## 本轮验收记录

当前AB共149项：143通过/6上述reference失败，其中同状态数值87项、codegen1项、
API契约1项、reference60项。合并dV/WS/G重建/dA probe共558项，546通过/12失败，
另外6项是原WS长序列P量化问题。XML及日志`/tmp/dism-ab-regression.{xml,log}`。
三类sanitizer分别覆盖融合数值87项+codegen，共88项，零errors/hazards；
仅对B3命名空间`kns=_ZN7dism_v22ab`插桩，日志
`/tmp/dism-ab-final-{memcheck,racecheck,synccheck}.log`。
API契约检查是随后新增并通过普通回归，未计入上述88项sanitizer覆盖。
早期sanitizer运行因新reference测试的Python变量遮蔽错误被停止，不计入通过；
修复测试后重新完成以上三类检查。未运行全套前向/embedding精度回归，未测性能。

## 12-warp specialization

单warp基线及上述结果已提交为17f5674；用户接受当前精度，原reference失败继续保留。
新增 `operand_gradient(...,warp_specialized=True)`，按用户后续要求默认启用WS。
显式传入 `warp_specialized=False` 可使用单warp诊断基线；dV入口默认策略未改。
代码在 `csrc/core_ab_ws.cu`，没有global W/P/E/G，不消耗额外RNG。

- 每CTA128key，8个compute warp各16key，逻辑顺序0,4,1,5,2,6,3,7；
  producer group8–11释放寄存器到40，compute groups提升至232，只有warp8加载。
- 初始B/V shared staging转入常驻寄存器，然后复用为A/dO两槽输入环，每次64query倒序。
  W/dP/E完全独立；4–7从下一32key G边界开始reverse inclusive scan，随后
  把HState经配对双槽mailbox传给0–3，立即继续自己的梯度GEMM。
- 每warp常驻一个dB FP32 accumulator。Gsoft寄存器保留物理query列排列，
  直接与TMA加载的A相乘；dB使用完输入后才释放输入槽，避免producer覆盖。
- Gsoft另写逻辑列shared，col-layout转置读出后用来计算dA。
  每warp2KiB union先存Gsoft16x64 BF16，再复用为两个16x16 FP32 dA输出槽。
  dA由3D TMA reduce-add异步写回，B在小register tile上转col布局。
  使用较小输出块限制shared占用，尚未测量TMA事务增多的性能影响。
- 下次改写Gsoft前leader执行wait_group.read0；输出双槽轮转使用wait_group.read1，
  writer fence+syncwarp发布，退出wait_group0。无效warp仍完整参与输入/mailbox协议。

九实例初版编译资源如下，D128 spill按用户授权保留、不做优化：

| D | DV | stack B | spill stores/loads B |
|---|---|---|---|
| 32 | 32/64/128 | 0 | 0/0 |
| 64 | 32/64/128 | 0 | 0/0 |
| 128 | 32 | 32 | 32/32 |
| 128 | 64 | 144 | 192/144 |
| 128 | 128 | 216 | 228/212 |

ptxas全CTA metadata为168 registers，不是compute角色上限。SASS九实例均无CALL，
保留UTMALDG.5D、UTMAREDG.3D.ADD、MUFU.TANH、LDSM转置，
以及USETMAXREG释放40/申请232；测试明确检查两种重分配指令及其立即数。
日志 `/tmp/dism-ab-ws-build.log`、SASS `/tmp/dism-ab-ws.sass`。

WS测试149项：143通过、6个原bounded-soft reference逐元素失败。
同G/GEMM对照最大绝对误差dA=5.22108e-6、dB=2.40658e-5。
reference最大relative L2为0.00548918/0.00613192，
最小cosine为0.99998509/0.99998121；精度表现与单warp基线相当。
覆盖全部九种维度、两方向/随机全局方向、软硬混合、N=1…2049尾部/长链、
非默认stream及重放。记录 `/tmp/dism-ab-ws.xml`、`/tmp/dism-ab-ws-full.log`。
三类sanitizer各89项（非reference数值+契约+codegen）通过：
memcheck 211.71s、racecheck 219.44s、synccheck 125.19s，零errors/hazards。
这些是并行运行时的测试总耗时，不是kernel性能数据。仅插桩B3 WS命名空间
`kns=_ZN7dism_v25ab_ws`，日志 `/tmp/dism-ab-ws-{memcheck,racecheck,synccheck}.log`。
codegen随后强化为检查40/232立即数并单独复跑通过，kernel二进制未变。
旧AB/dV/WS/G/probe回归558项：546通过/12原失败（6单warp AB reference、
6旧dV WS P量化），无新增失败；日志/XML `/tmp/dism-ab-ws-regression.{log,xml}`。
未运行全套前向/embedding回归，未计时，不声称已有性能收益。

## dLSE/drtau接入

当前未提交实现为两个B3路径统一新增FP32标量梯度，返回四项：
dA/dB保持原含义，dLSE形状[B,H,N]，drtau形状[H]。
scalar_grad.cuh在G反向unroll后、BF16转换前求和，不额外乘scale或LOG2E。
q_from_k按warp归约16key后atomic add到dLQ；k_from_q每warp持有16key的
dLK局部和，结束时独占写回。drtau包含hard匹配项，每warp partial写回后
由core_tau.cu按head合并batch；partial为FP32，无全局G。
WS partial大小为B*H*ceil(paddedN/128)*8*4 bytes，单warp为B*H*(paddedN/32)*4 bytes。

首次编译日志 /tmp/dism-scalar-build.log。WS新增D64/DV128 spill：
56B stack、64B spill stores、56B spill loads（此前零）。
其他五种D32/64实例仍零spill；D128对应DV32/64/128的stack为64/152/288B，
stores为88/216/424B，loads为68/168/340B。单warpD32/64六种仍零spill，
D128 stack为64/40/40B。
首次编译后按约定暂停；用户随后明确授权保留上述spill继续验证，未做spill优化。
WS专属codegen仅新增D64/DV128例外，其余五种D32/64仍要求零spill；
单warp仍要求D32/64六种零spill，全扩展旧严格零spill断言未改。
上节通过数量对应新增标量梯度之前的二输出版本，本次实测见下方。
SASS检查：两个B3的18实例无CALL。独立dtau归约初版64位除法产生CALL，
已改为显式batch循环消除，未调整B3 spill。dtau归约38 registers、32B shared、
零spill，专属无CALL/LDL/STL codegen测试1项通过；日志 /tmp/dism-scalar-tau-build.log。

### 四输出数值验证

两个B3测试文件合计299项：257通过、42普通失败。非reference的176项数值/契约
全部通过，另3项codegen通过；两个路径各60 reference项中39通过/21失败。
每路径6项原bounded-soft dA/dB精度失败仍在，新增15项常规case的drtau精度失败；
不xfail、不放宽数值阈值。原始日志/XML：/tmp/dism-scalar-full.{log,xml}。

同独立重算G的FP64归约对照，WS最大绝对误差dLSE=3.06602e-5、
drtau=1.79582e-3；单warp分别3.06602e-5、1.46013e-3，
全部满足现有atol=5e-4、rtol=2e-3组合阈值。
这些数值仅验证归约对同G正确，不代表对原始reference的精度达标。
覆盖九种D/DV、两方向、soft/hard/mixed、N=1…2049、RNG重放、
非默认stream，全hard场景的dLSE严格为零，全hard不匹配链的drtau严格为零。

新增test_dism_v2_scalar_grad.py的8个8193长度闭式对照用例全部通过，
覆盖两路径、两方向、全soft/全hard、多batch/head，两个路径均超过256个tau partial。
logM=0时exp(W[i,j])=j+1，其tau导数为(j+1)(j+2)/2，
用FP64前缀和及保存的normalizer/delta构造O(N*DV) oracle，不物化全矩阵。
soft用例还检查drtau=-sum(dLSE)，hard用例检查dLSE=0。
日志/XML /tmp/dism-scalar-long.{log,xml}。

### 标量梯度精度诊断（不是生产路径变更）

WS60个reference用例的最大相对L2：dLSE=0.0597281、drtau=0.283235。
仅把delta替换为reference FP32 O与dO的FP32 dot，并重新生成B1/G32和B3，
最大相对L2分别降至0.000246878、0.00172901。
最大绝对误差由0.377956/23.1400降至0.00163828/0.149010。
这支持saved BF16 O的delta误差是这些标量梯度偏差的主要来源；
不是FP32乘加精度不足，也不是把Gsoft转BF16再求和导致。
残余仍包含W/normalizer重算近似，不能声称已完全消除误差。
所有60例诊断已固化到reference测试helper，生产delta语义保持不变；
日志/XML /tmp/dism-scalar-delta.{log,xml}。

### 其他回归与检查范围

常规N≤2049套件三类sanitizer各179项通过（176数值/契约+3 codegen），
零errors/hazards；过滤器regex=_ZN7dism_v2(2ab3run|5ab_ws3run|10scalar_bwd)，
覆盖两种B3和新dtau归约，不把未插桩的其他kernel计为本轮sanitizer覆盖。
memcheck/racecheck/synccheck测试耗时456.93/514.97/226.64秒，
为多进程并发下的测试总时长而非kernel基准。日志 /tmp/dism-scalar-{memcheck,racecheck,synccheck}.log。

dV/原WS/G重建/dA probe回归409项：403通过、6个原dV WS P量化失败，
没有新增失败，日志/XML /tmp/dism-scalar-regression.{log,xml}。
独立delta与CPU FP64五梯度公式共75项通过，日志 /tmp/dism-scalar-contract.log。
旧全扩展严格零spill codegen未计入这75项，仍不满足既有spill现状。

8193长序列8项：memcheck与synccheck对两个B3及dtau归约全部插桩、零errors；
额外racecheck仅插桩dtau归约，同样8通过、零hazards，覆盖每线程读取多个partial。
日志 /tmp/dism-scalar-long-{memcheck,synccheck}.log、
/tmp/dism-scalar-long-tau-racecheck.log。
8193全B3 racecheck开销较大，启动后主动中止，不计为通过；
完整B3 racecheck覆盖由常规N≤2049套件承担。没有修改任何kernel以绕过插桩。
本轮未测性能，未做spill优化，未验证sm90或embedding backward。
