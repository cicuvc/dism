# 融合dV的reverse affine摘要与passing

当前接口在原`value_gradient`上增加成对可选参数`v`和`delta`：

```python
from dism_v2.backward import delta, value_gradient
row_delta = delta(dout, out)
dv, summary, boundary = value_gradient(
    a, b, dout, lse, tau, q_label, k_label, l2, edges,
    sm_scale=scale, rng_state=rng, v=v, delta=row_delta,
)
```

不提供v/delta时仍返回独立dV。提供两者时在同一个CUDA kernel内重算W/P、累积dV、
计算真实`dP=V_tile @ dO_tile.T`及`E=P*(dP-delta)`，并做add-mul reduce。
delta为保存的BF16 O与dO的FP32点积，与已有预处理约定一致；不重新消费RNG。

## 布局与passing

当前仍是单warp CTA、16key×64query的正确性基线。每warp独立生成16-key局部摘要，
query tiles逆序遍历时携带reverse VState。没有新warp间串行依赖。
reverse HState编码列1…64，列0由VState补齐，局部状态保存在矩形FP32二元组buffer中。

随后一个passing kernel沿对角线执行两件事：

1. 组合相邻两份16-key摘要：若top=(a,b)、bottom=(c,d)，组合为`(a*c,a*d+b)`。
   bottom需读query偏移+16，超出paddedN时为identity。
2. 按32-key块逆序传递`state=a*state+b`，缺失后继state=0。

设np=ceil(N/64)*64，返回：

- `summary[B,H,np/32,np,2]`：每32-key块从其后一块得到G的affine映射，最后两轴为query和二元组。
- `boundary[B,H,np/32,np]`：`boundary[...,c,q]=G[q,32*c]`，供后续B3作为块下边界使用。

所有summary/边界均为FP32，自然logits梯度语义。alpha用稳定sigmoid(W₂*ln2)，
hard不匹配为(0,0)，padding为(1,0)，hard匹配不跳过E或递推。
E中的P保留FP32；仅dV GEMM使用BF16高位+残差。passing使用FP32 FMA，不降BF16。
未物化global W/P/E/G或行mask；调试probe中完整W仅供测试。

临时16-key local大小`8*BH*(np/16)*np` bytes，passing后释放；输出summary+boundary为
`12*BH*(np/32)*np` bytes。两者共占`0.875*BH*np²` bytes（np8192/BH1=56MiB），
不含前向保存状态、dV、delta等；不是全流程峰值。输出保留24MiB，local暂占32MiB。
后续合入配对warp调度时，可直接生成32-key摘要，移除这份16-key临时buffer。

## 寄存器与shared暂存

直接融合最初出现spill。当前融合版在PV阶段前后从shared读回/写回本warp的dV累加值，
从而不让整块累加器在score/scan/reverse reduce期间长期占用寄存器。
W也仅在shared中跨PV/dP阶段暂存；数组以lane为最内层，连续lane访问连续float2。
dP GEMM按DV的32维分段加载；score/PV保留原计算顺序。

暂存使用显式FP32数组，不使用TK FP32 shared tile读回：该尝试曾被memcheck捕获越界，
已经移除相关读写，没有修改外部TK/GLX代码。使用动态shared并为大实例显式设置opt-in容量。
当前单warp融合版寄存器矩阵（行D、列DV32/64/128）：

| D | DV32 | DV64 | DV128 |
|---|---:|---:|---:|
| 32 | 255 | 223 | 174 |
| 64 | 255 | 227 | 174 |
| 128 | 214 | 210 | 170 |

passing为36寄存器。所有实例无CALL/LDL/STL、STACK/LOCAL=0，有原生UTMALDG.5D，无atomic。
shared暂存和重复加载有成本，尚未计时；此版本不是12-warp多缓冲流水的性能验收。

## 验证范围与误差分层

新增78项summary回归，覆盖九种D/DV、两固定方向、随机direction、soft/hard/mixed，
N=1/17/64/65/129/139/513/1025/2049、ln64匹配链、全不匹配和尾部identity。
同时检查开启摘要后的dV与独立dV一致（atol/rtol=2e-6）、非默认stream和无新RNG消费。
与85项dV及76项backward原语测试合跑239项通过。

摘要/状态oracle用独立转置诊断probe取得同一保存边界下的重算W₂，
在FP64中计算真实dO/V点积、E/alpha及逐key递推。并不是读取本kernel的E或摘要作为oracle。
最大误差：summary first=5.966e-8、second=2.223e-5、boundary=6.315e-5。
阈值分别为(2e-5,2e-4)、(2e-4,5e-4)、(5e-4,1e-3)的atol/rtol。

不能把上述误差称为相对理想reference的完整G梯度误差：初次直接使用reference FP32 W时，
513行ln64匹配链曾有summary second差异约0.002416，而改用同重算状态后通过紧阈值。
测试以`summary_w_max_abs`单独记录reference W与重算W差异，不通过修改hard语义或放宽上述阈值掩盖。
完整core梯度对autograd的精度验收仍属于后续B3工作；既有前向量化失败继续保留。
数值日志`/tmp/dism-summary-final.xml`。

最终78项摘要用例分别通过memcheck/racecheck/synccheck，零错误、racecheck零hazards/零warnings；
日志`/tmp/dism-summary-final-check.3J8bU2`。此前TK shared读回的失败已由修正版全量重跑覆盖。
合并非precision回归526项通过（`/tmp/dism-summary-all.xml`）；未重跑已有forward/embedding精度套件。
