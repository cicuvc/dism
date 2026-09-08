# B3 dA/dB融合正确性路径（sm120a）

`backward.operand_gradient`计算已选方向core操作数的FP32 `(dA,dB)`。
不物化global W/P/E/G；不是完整backward/autograd，尚无dLSE/drtau归约或embedding backward。
q_from_k时dB是插值操作数的梯度，不能直接称dK。

```python
from dism_v2.backward import delta, value_gradient, operand_gradient
dd = delta(dout, out)
dv, summary, g32 = value_gradient(
    a, b, dout, lse, tau, q_label, k_label, l2, edges,
    sm_scale=scale, rng_state=rng, v=v, delta=dd, warp_specialized=True,
)
da, db = operand_gradient(
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
