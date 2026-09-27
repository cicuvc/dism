# CUDA embedding前向：首版实测

RTX 5090，sm120a，CUDA 13.1，torch 2.13.0+cu130，conda blkw。
只实现CUDA前向；embedding backward仍调用现有Triton wrapper。

## 接口与实现

```python
from dism_v2.embedding import forward
raw = forward(q, k, q_voc, k_voc, sm_scale)  # 默认12-warp双缓冲
baseline = forward(q, k, q_voc, k_voc, sm_scale, warp_specialized=False)
wide = forward(q, k, q_voc, k_voc, sm_scale, block_v=128)  # 仅D32/64

from dism_v2.autograd import voc_dism
out = voc_dism(q, k, v, rtau, q_voc, k_voc,
               sm_scale=sm_scale, embedding_backend="cuda")
```

低层forward无autograd；高层六输入autograd支持显式CUDA embedding前向。
高层默认仍为`embedding_backend="triton"`。两条路径都保存本次实际插值/LSE给Triton反向。
输出顺序严格为`q_from_k, k_from_q, k_lse, q_lse, k_top, q_top, k_index, q_index`。
BF16输入/插值输出，FP32 LSE/top，int32索引。fixed-length、D32/64/128、N/V尾部。

每CTA同一64个token；warp0–3各负责16行q_from_k，warp4–7负责相同位置的k_from_q。
warp8加载两张词表，warp8–11整体dec40，compute groups inc232。
每槽E_q/E_k各64×D BF16，两个槽复用初始q/k staging。
完整词表块使用普通3D `[D,V,H]` TMA，D128分两个64-feature段；尾部guarded加载。
同一shared词表由score的row-layout和PV的col-layout读取，没有GLX置换或global转置。
所有warp完成最后的PV shared读取后才释放slot；无效token warp完整参与协议，P置零、不写回。

score和PV按32-feature片段读取，FP32 online max/sum/acc，BF16 P单次MMA。
没有Dism零fallback，没有global logits/P，没有RNG。相等最大值取最小词表索引。
输出除法最初产生CALL慢路径，已改为`rcp.approx.ftz`乘法，与core做法一致。
其适用分母为正且≥1；没有以有限负数替代词表tail的负无穷。

## 资源与正确性

| D | 单warp寄存器/thread | WS静态寄存器/thread | WS动态shared bytes | spill / CALL |
|---|---:|---:|---:|---|
| 32 | 96 | 168 | 16512 | 0 / 0 |
| 64 | 128 | 168 | 32896 | 0 / 0 |
| 128 | 255 | 168 | 65664 | 0 / 0 |

动态shared包含32B barrier和128B对齐；另外cuobjdump报告每kernel有1024B静态shared。
WS的168为初始CTA分配，不是compute运行时预算；SASS实际有40释放/232申请。
六个kernel全部零stack/零local；三个WS均有原生UTMALDG和USETMAXREG。
`test_codegen`断言无CALL（包括CALL.REL）、无LDL/STL，WS原生指令不退化。

`tests/test_dism_v2_embedding_cuda.py`：

- 124项非autograd测试全部通过：D三种，V=1/31/32/63/64/65/127/128/129/257，
  N=1/15/16/17/63/64/65/129；FP32输出/statistics/argmax oracle，跨tile ties，
  scale=-1/0/1，单warp/WS精确对比，以及融合Triton对照。
- FP32插值对照最大relative L2为0.00220246，最低余弦0.9999975766；包含BF16 P/输出量化。
- Compute Sanitizer memcheck/racecheck/synccheck各124项通过，零error/零race hazard。
  范围为embedding CUDA kernels，输入不做padding；racecheck不证明global写竞争不存在。
- CUDA backend的108项端到端reference测试100通过、8项rtau幅值失败；形状与原Triton
  路径失败一致，只有D64/DV128混合行、D128/DV128纯hard，两个方向/两种oracle。
  其余输出和q/k/v/词表梯度通过；两个oracle合计216个head梯度比较，无rtau符号反转。
- `test_dism_v2_autograd.py`新增81项CUDA backend接线测试全部通过：九种D/DV、
  两固定方向/random、hard_prob=0/.37/1，反向不额外消费RNG。

两个文件合计545项：514通过、31普通失败。31=上述8项+原套件23项
（20项rtau幅值、3项独立Triton V1 embedding backward量化误差），没有xfail或放宽容差。
CUDA backend纯torch oracle的54组最大relative L2：O .00240340、dq .00392638、
dk .00373591、dV .00297049、drtau .01884995、dq_voc .00389517、dk_voc .00378310。
这里只覆盖该批形状，不能外推长期训练或其他rtau值的符号保证。

复现：

```bash
MAX_JOBS=2 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest tests/test_dism_v2_embedding_cuda.py tests/test_dism_v2_autograd.py -q
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 --kernel-name regex=_ZN7dism_v29embedding /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest tests/test_dism_v2_embedding_cuda.py -q -k 'not autograd'
```

本次日志：`/tmp/dism-embedding-final.{log,xml}`、`/tmp/dism-embedding-final-build.log`、
`/tmp/dism-embedding-final-{memcheck,racecheck,synccheck}.log`。

## 初步计时及剩余工作

`python -m dism_v2.benchmark_embedding`：重复相同输入热cache，10次warmup，
7组×30次CUDA event计时取中位数；包含output分配与Python/host发射间隙，**不是纯kernel计时**。
B=1、H=4；数值单位µs。测试不与其他验证并发运行。

| N / V | D | CUDA WS | 两次单warp FA | 融合Triton |
|---|---:|---:|---:|---:|
| 65 / 129 | 32 | 13.79 | 90.57 | 43.40 |
| 65 / 129 | 64 | 16.95 | 168.43 | 43.34 |
| 65 / 129 | 128 | 31.17 | 332.35 | 46.44 |
| 1024 / 1024 | 32 | 29.37 | 602.37 | 44.83 |
| 1024 / 1024 | 64 | 41.42 | 1163.82 | 48.51 |
| 1024 / 1024 | 128 | 70.11 | 2308.04 | 120.44 |
| 4096 / 1024 | 32 | 57.81 | 605.51 | 44.77 |
| 4096 / 1024 | 64 | 79.45 | 2070.31 | 59.05 |
| 4096 / 1024 | 128 | 145.88 | 7815.21 | 238.85 |

单warp数值基线只有16行/CTA、无TMA且两次launch；它不是相同CTA划分的带宽消融。
因此此处巨大差距不能归因为“共享词表流量减半”。与融合Triton也没有额外减半承诺。
D32/64大N场景仍落后Triton；目前不切换高层默认backend。

尚待：冷cache与L2/HBM实际流量、相同CTA划分的独立FA对照、
单缓冲对照，以及profile后有针对性的性能优化。sm90未编译/实机验证。
CUDA embedding backward不在本轮范围内。

## 追加实验：D32/64、B_V128

按用户要求增加`block_v=128`，仍是12-warp、两个槽、dec40/inc232。
D32/64动态shared分别32896/65664B（另外1024B静态shared）；两个实例均静态168寄存器、
零stack/零spill，无CALL，TMA和寄存器转移生效。D128不生成B_V128实例，避免shared容量超限。
低层CUDA forward可显式选择128；高层autograd CUDA选项仍用64，本轮不自动选择形状策略。

新增48项：D32/64×V=1/31/32/63/64/65/127/128/129/255/256/257/513/1025，
加N尾部、全相等与跨tile argmax测试，全部通过。共172项非autograd测试通过。
八个kernel（3单warp、3个WS64、2个WS128）的codegen回归均通过。
三类sanitizer分别运行新增48项和codegen共49项，均零错误/零hazard。
追加后的两个完整测试文件593项：562通过、31项上述既有精度失败，未新增失败。

新增`python -m dism_v2.benchmark_embedding --device-only`：10次warmup后，
用torch.profiler/CUPTI取30次实际kernel duration的中位数，验证每调用只产生一个kernel。
热cache，包含profiling instrumentation，排除Python/分配/发射间隙；不是冷cache带宽测量。
这也说明前面的API计时不能作为纯GPU性能排序依据，尤其是小kernel。

| B/H | N / V | D | WS64 µs | WS128 µs | 融合Triton µs |
|---|---|---:|---:|---:|---:|
| 1/4 | 65 / 129 | 32 | 9.44 | 11.10 | 4.61 |
| 1/4 | 65 / 129 | 64 | 15.97 | 18.94 | 7.90 |
| 1/4 | 1024 / 1024 | 32 | 28.32 | 27.42 | 16.99 |
| 1/4 | 1024 / 1024 | 64 | 40.29 | 35.78 | 31.58 |
| 1/4 | 4096 / 1024 | 32 | 55.94 | 54.21 | 29.63 |
| 1/4 | 4096 / 1024 | 64 | 78.03 | 71.02 | 56.35 |

128步长对完整大V有收益，D64本批约9–11%时延降低，D32约3%；V129尾部则变慢。
没有反超融合Triton，不能仅由零spill推断已达到性能目标；不统一切换默认步长。
D128保持64，同次CUPTI在N1024/4096、V1024为69.30/144.38µs，Triton118.10/236.64µs。
更进一步优化应先定位score/PV、softmax/argmax、producer等待和共享读取的瓶颈，不盲目加宽。

日志：`/tmp/dism-embedding-bv128-build.log`、`/tmp/dism-embedding-bv128.{log,xml}`、
`/tmp/dism-embedding-bv128-benchmark.jsonl`（API）、`/tmp/dism-embedding-bv128-device.jsonl`（CUPTI）。
完整回归`/tmp/dism-embedding-bv128-full.{log,xml}`；sanitizer为
`/tmp/dism-embedding-bv128-{memcheck,racecheck,synccheck}.log`。
