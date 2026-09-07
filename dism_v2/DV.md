# dV 正确性里程碑（sm120a）

本页记录独立dV基线。随后新增的可选dP/E摘要与passing见[BACKWARD_SUMMARY.md](BACKWARD_SUMMARY.md)；
融合路径会在阶段间把dV累加值暂存shared，与本页寄存器常驻基线的资源策略不同。

入口：`dism_v2.backward.value_gradient(a,b,dout,lse,tau,q_label,k_label,normalizer,boundaries,sm_scale=...,rng_state=...)`。
参数是同一次forward的已选方向操作数、FP32 L₂、`save_boundaries=True`返回的边界及RowRNGState。
输入BF16 contiguous，D/DV独立32/64/128，返回FP32 `[B,H,N,DV]`。
调用方不得修改保存的操作数/状态或更换scale；当前尚无自动保存/version-check的autograd包装。
不消费新RNG，随机direction使用前向已选结果。

固定方向的最小调用（A/B/LSE已按方向选好，所有操作数不变）：

```python
from dism_v2.core import forward
from dism_v2.backward import value_gradient

out, l2, edges, rng = forward(
    a, b, v, lse, tau, q_label, k_label,
    sm_scale=scale, direction=direction, hard_prob=hard_prob,
    save_boundaries=True, return_rng_state=True,
)
dv = value_gradient(
    a, b, dout, lse, tau, q_label, k_label, l2, edges,
    sm_scale=scale, rng_state=rng,
)
```

全局random方向调用后，须按`rng.direction`选择A/B/LSE再传入此入口，不能把两个方向的tuple直接传入。

## 当前实现

每CTA一个warp，负责16个key，逆序流式加载64-query tiles。key先保留在shared，
每次MMA前加载到寄存器，并允许其生命周期在scan前结束；dV accumulator长期保留在warp内。
完整A/dO tile使用原生5D TMA置换加载，非对齐尾部guarded copy。两个query-column RNG ballots
沿16个key共用。重算W仅依赖竖16/横64边界，不读取global W/P或mask。

P=`exp2(W₂-L₂)`，计算`dV=P.T @ dO`，输出由key warp独占，无atomic。
当前单缓冲、单warp CTA是正确性基线，尚未接入12-warp producer/consumer流水，
也未计算E、reverse summary、dA/dB/dLSE/drtau。不能称为完整B1或完整backward。
dV不依赖delta，后续score梯度才使用它。

## P 精度修正

首版仅将P转为一次BF16 MMA输入。N≤2049测试通过，但N8193、rtau=ln64、两方向的
纯soft与混合共4例逐元素阈值失败。参考P→BF16的诊断说明混合场景主要受该转换影响：
q_from_k/mixed的量化RMSE=9.46572e-4，kernel总RMSE=9.46272e-4。

当前使用 `P_hi=bf16(P)`、`P_lo=bf16(P-float(P_hi))`，两次BF16 MMA累加FP32 dV。
没有改score、scan、L₂、输入dtype或测试阈值；没有修改前向的PV量化策略。
这增加了dV乘法的MMA工作量，尚未测量总性能，不预设为最终性能方案。

## 数值回归

`tests/test_dism_v2_dv.py`的85例，oracle为`voc_dism_ref`对v的autograd，
双方使用相同BF16 interpolation、hard决策与direction；oracle的v使用FP32以避免输出舍入。
这不是未量化FP32 embedding的端到端验收，也不代表严格求导BF16舍入。

- 九种D/DV、两固定方向、hard_prob=0/.37/1，N139、B=H=2。
- N=1/17/31/32/63/64/65/128/129/257/513，随机direction、混合概率、非默认stream、逐位重放。
- N1025/8193的ln64长匹配链、全不匹配和纯soft。
- D64/DV128、N2049/8193、rtau=ln64、两方向的pure/mixed。
- 检查无新RNG消费、全不匹配精确零、返回dtype和非法状态形状/offset拒绝。

全部通过既定atol=.008、rtol=.012；相对L2<.004、整体cosine>.99998。
修正版85例最大abs=0.0117605、最大RMSE=4.03614e-4、最大相对L2=4.09694e-4，
最低cosine=0.999999928。max_abs是所有元素最大差值，判定仍按abs+relative组合，
不应误读为所有元素绝对误差都小于.008。
N8193/q_from_k/mixed的修正版RMSE=5.52908e-6，max_abs=1.05858e-4。
纯soft仍有FP32长程scan/normalizer等误差；本实验未进一步拆分其来源。
初版诊断`/tmp/dism-dv-diagnostic.xml`，最终85例报告`/tmp/dism-dv-complete.xml`。
新增8193行全匹配链max_abs=0.00249720、relative_L2=4.09694e-4，仍通过原阈值。

## 资源（RTX5090，CUDA13.1，blkw）

| D \\ DV | 32 | 64 | 128 |
|---|---:|---:|---:|
| 32 | 141 | 254 | 244 |
| 64 | 242 | 254 | 250 |
| 128 | 206 | 200 | 236 |

表为registers/thread，所有实例STACK/LOCAL=0，无CALL（含REL）、LDL/STL、ATOM/RED，
有原生UTMALDG.5D。shared为10,256至37,904 bytes/CTA（含工具报告开销）。
部分实例超过前向consumer的232预算，合入12-warp流水时必须重新评估，不能照搬预算。
除保存的前向状态与输出dV（`4*BH*N*DV` bytes）外，当前dV kernel没有global临时缓冲；
shared不随N增长。无全矩阵物化，未做性能计时。

## Sanitizer与状态

dV正确性里程碑已完成：全部85项通过数值测试，全部用例覆盖memcheck/racecheck/synccheck，
零错误、racecheck零hazards/零warnings。原79项memcheck/racecheck和最终85项synccheck日志在
`/tmp/dism-dv-corrected-check.9ZhQaG`；新增8193行用例的补充memcheck/racecheck在
`/tmp/dism-dv-extra-check.afHEMu`（10例含4个重叠用例）。
扩充前合并reference/build/core/codegen/布局/反向原语/dV共442项通过。
没有重跑forward长序列precision和embedding_precision；其中既有量化失败继续保留。
