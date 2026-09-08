# sm120端到端前向与反向

最新（2026-09-08）：CUDA embedding前向和配对/对称WS反向均已显式接入。
设置`embedding_backend="cuda", embedding_backward_backend="cuda"`运行全CUDA链路；
反向选`"cuda_symmetric"`使用对称词表方案。默认前后向仍为Triton embedding，
CUDA core不变。两个后端开关独立，全部六种组合均有接线/RNG回归。
CUDA配对反向的token扫描步长D32=64、D64/128=32；对称为D32 reg/T32、
D64 shared/T16、D128 reg/T16。它们是显式固定选项，不是自动性能调度器。
以下原Triton验证记录保留；最终本轮结果见文末。

入口：

```python
from dism_v2.autograd import voc_dism
out, state = voc_dism(
    q, k, v, rtau, q_voc, k_voc,
    sm_scale=q.shape[-1] ** -0.5, direction="random", hard_prob=0.37,
    return_rng_state=True,
)
out.float().square().mean().backward()
# q.grad / k.grad / v.grad / rtau.grad / q_voc.grad / k_voc.grad
```

q/k/v及词表均为contiguous BF16，rtau为FP32 [H]。D/DV独立32/64/128，
词表[H,V,D]，V>0；支持fixed-length尾部和一次调用的全局随机方向。
显式rng_state重放不消费RNG，backward不消费RNG。可只请求部分输入梯度。
仅sm120，不支持varlen、CUDA Graph、deterministic backward、高阶梯度或概率广播。
返回O及各输入梯度的dtype与对应输入一致；rtau梯度FP32。

## 实现与梯度映射

使用现有emb_kernel.py的emb_fwd_wrapper/emb_bwd_wrapper，不重写插值算法。
前向：Triton embedding → CUDA core forward并保存边界。
反向：FP32 delta → WS dV/summary → G32 passing → WS dA/dB/dLSE/drtau
→ Triton embedding backward → FP32合并直接与插值梯度 → 最终BF16写回。
直接调用wrapper是为了让core产生的FP32梯度传入embedding preprocessing，
并在直接/插值分支相加前不经PyTorch BF16中间节点额外舍入。

原embedding输出顺序为
(q_from_k,k_from_q,k_lse,q_lse,k_top,q_top,k_index,q_index)。
因此传给embedding backward的位置如下：

| 全局direction | doq | dok | dlq | dlk | 最后合并 |
|---|---|---|---|---|---|
| q_from_k | dB | 0 | 0 | dLSE | dq += dA |
| k_from_q | 0 | dA | dLSE | 0 | dk += dB |

hard不对标签求导；全hard时q/k/词表梯度严格为零，但dV/drtau仍计算。
生产delta继续使用保存的BF16 O，没有采用reference delta诊断替换。
没有global logM/W/P/G或行随机数数组。

## 对原embedding文件的最小调整

1. Phase B仅允许pid_v==0计算/写dq和dk。原实现每个vocab block都重复写相同地址，
   V=129时每个row block有3个writer；AST归属测试在原实现失败，修复后通过。
   这属于写入归属修复，不改变数学计算。
2. D128前向及反向的num_stages设为1，其余维度保留3。原默认配置在多vocab block下，
   前向需要106496B、反向136704B shared，超过本机101376B上限。
   调整后测试shape的前向/反向分别73728B/90112B，可启动；未做性能优化。

未改dism_ref.py、旧tt_dism.py或外部依赖。emb_kernel.py本来是用户未跟踪文件；
本轮只修改以上保护和launch配置，不把其余内容视为本轮编写。

## 数值与接线测试

tests/test_dism_v2_autograd.py：

- 81项九维度组合×三方向×三种概率的完整六输入autograd接线，对照手动kernel串联。
- 6项尾部/词表/重放/非默认stream/只求tau梯度，N=1…1025、V=1…257。
- 输入契约、高阶/确定性拒绝与Phase B归属检查共2项；上述89项全部通过。
- 三步真实loss.backward与参数更新smoke通过，六输入grad均存在且有限。
- 15项独立embedding backward对FP32 torch：12通过，3个V=1词表梯度精度失败。
  D32/V1在修改前已复现；全部D的V1均保留普通失败，没有修改数值容差或插值算法。
- 108项端到端reference：54场景分别对纯FP32 torch插值与相同embedding值的oracle比较。
  后者用实际embedding值/标签，但保留FP32 torch embedding Jacobian，不能称为
  对BF16舍入求精确导数。100通过、8个rtau幅值容差失败，未出现其他梯度失败。
- 16项rtau=ln64对照，N=17/139/513，pure soft/mixed，两方向及两oracle：
  4通过、12个rtau幅值失败，其他梯度和输出通过。
- 三项embedding codegen通过，无CALL（包括CALL.REL）。

总计232项，209通过、23普通失败。23项=20个rtau幅值测试+3个独立embedding
V1精度测试。没有xfail、跳过失败或放宽数值容差。两组执行记录分别为
/tmp/dism-e2e-verified.{log,xml}（215项）、
/tmp/dism-e2e-bound.{log,xml}（17项）。
最终整套复跑同样209通过/23失败，已包含rtau非近零反号显式断言：
/tmp/dism-e2e-complete.{log,xml}。

纯FP32 torch oracle的62个场景中，相对L2最大值：

| 输出/梯度 | 最大relative L2 |
|---|---|
| O | 0.00240340 |
| dq | 0.00496129 |
| dk | 0.00388064 |
| dv | 0.00297049 |
| drtau | 0.0372084 |
| dq_voc | 0.00501286 |
| dk_voc | 0.00411408 |

两种oracle均未观察到rtau反号。每种oracle覆盖62场景×2head=124个符号比较，
本次数据没有|reference drtau|≤1e-5的head；测试同时记录近零项与实际/参考梯度值。
无反号只是本次观测，不保证未来训练中不会反号；原有幅值失败继续可见。

## Codegen与检查

采样B=2/H=2/N=65/V=129，D=32/64/128，BF16输入、FP32上游梯度；
记录为Triton n_regs/n_spills/metadata.shared，n_spills不是ptxas的spill bytes。

| D | forward regs / n_spills / shared B | backward regs / n_spills / shared B |
|---|---|---|
| 32 | 238 / 0 / 24576 | 255 / 30 / 38400 |
| 64 | 255 / 24 / 49152 | 255 / 150 / 71168 |
| 128 | 255 / 170 / 73728 | 255 / 148 / 90112 |

embedding preprocess三个实例零spill。原embedding仍有上述spill，未优化；
本轮没有改变CUDA core的代码或资源。日志/XML /tmp/dism-emb-codegen.{log,xml}。

Triton consan尝试在原embedding前向编译阶段失败（shape/order rank 5 vs 7），
不能据此声称consan通过，记录/tmp/dism-emb-original-consan.log。
Compute Sanitizer的racecheck主要检查shared memory，不将其通过当作原跨CTA
global重复写入已被工具检测；归属修复另由源代码结构与测试明确保证。

三类Compute Sanitizer各89项接线/契约/尾部测试通过、零errors/hazards：
过滤器regex=(emb_fwd|_interp_bwd|_ZN7dism_v2)，覆盖实际embedding前后向及所有Dism
CUDA kernels；不把PyTorch oracle kernels计入插桩范围。
memcheck/racecheck/synccheck测试耗时9.41/160.29/11.78秒，不是性能基准。
日志 /tmp/dism-e2e-{memcheck,racecheck,synccheck}.log。
测试刻意跨stream复用leaf，PyTorch提示AccumulateGrad stream mismatch，
已显式wait_stream并通过输出/梯度重放，未隐藏该提示；不支持CUDA Graph。

原实际embedding前向精度套件64项重跑：48通过、16个原FP32插值对照失败，
没有新增失败，记录/tmp/dism-e2e-forward-regression.{log,xml}。
本轮没有改变既有CUDA core二进制，未重跑所有core微基准或长序列反向压力集；
不将本批有限场景视为完整训练稳定性/性能验收。

## 全CUDA embedding最终端到端验证（2026-09-08）

本轮新增显式CUDA embedding反向接线；保存本次实际前向的插值/LSE，
反向按保存的全局direction将db或da和另一支dlse传给CUDA sparse embedding backward，
之后与core直接梯度FP32相加，再转回输入dtype。反向不重新抽样，不混用后端保存状态。

```python
out = voc_dism(q, k, v, rtau, q_voc, k_voc,
               sm_scale=D**-0.5, hard_prob=0.37,
               embedding_backend="cuda",
               embedding_backward_backend="cuda")  # 或 cuda_symmetric
out.float().square().mean().backward()
```

验证范围：九种D/DV、两固定方向及random、hard_prob=0/.37/1，
两种embedding前向×三种反向共六种组合。486项接线测试以同一次embedding前向值，
手动CUDA core+原Triton embedding反向为对照，全部通过。额外6项测试禁止调用
Triton embedding wrappers，确保全CUDA选项没有fallback。
三步训练、跨stream尾部N1/17/63/129/257/1025、RNG重放和只求rtau梯度检查通过。

联合`test_dism_v2_autograd.py`、`test_dism_v2_embedding_cuda.py`、
`test_dism_v2_embedding_backward.py`最终结果：**1508通过、77个普通精度失败**。
autograd为842通过/63失败，embedding CUDA前向套件272/8，独立CUDA embedding反向394/6。
失败构成为68项rtau幅值与9项V1词表梯度量化；没有放宽容差或添加xfail。
新增两种CUDA反向的40项reference失败均是原有rtau幅值场景的重复覆盖，
没有出现接线、NaN、RNG或q/k/v/词表梯度的新失败。
480个reference用例记录的rtau符号检查均无反号（近零阈值1e-5）；不外推训练稳定性。
完整误差属性与失败列表：`/tmp/dism-final-e2e.xml`、`/tmp/dism-final-e2e.log`。

memcheck/racecheck/synccheck各69项端到端选例全部通过，零errors/hazards。
选例覆盖九种D/DV的混合行接线、全部后端的尾部/跨stream重放、全CUDA无fallback。
kernel过滤器`regex=(emb_fwd|_interp_bwd|_ZN7dism_v2)`，覆盖embedding及CUDA core，
不对PyTorch oracle kernels插桩。racecheck用时176.66秒，不是性能数据。
PyTorch已有的跨stream AccumulateGrad提示仍保留，显式wait_stream及重放检查通过。
日志`/tmp/dism-final-e2e-{memcheck,racecheck,synccheck}.log`。

复现（conda blkw）：

```bash
python -m pytest -q tests/test_dism_v2_autograd.py tests/test_dism_v2_embedding_cuda.py tests/test_dism_v2_embedding_backward.py
compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name 'regex=(emb_fwd|_interp_bwd|_ZN7dism_v2)' python -m pytest -q tests/test_dism_v2_autograd.py -k '(test_autograd_wiring and cuda and q_from_k and 0.37) or test_replay_tails_and_partial_grad or test_all_cuda_no_embedding_fallback'
```

默认后端不变；没有移植sm90、增加varlen、修复spill或改变此前接受的BF16量化语义。
