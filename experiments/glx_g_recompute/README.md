# B3单warp真实G重建与梯度GEMM前置验证

sm120a诊断probe，尚未融合生产dA/dB kernel。与合成reverse probe不同，
本probe读取真实A/B/V/dO、LSE/rtau/labels、保存的forward竖16/横64 W₂边界、
L₂/delta、RowRNGState及生产WS passing的32-key G边界，不读取global W/P/E。
唯一完整矩阵输出是测试用G；后续GEMM诊断消费该输出，不能作为生产物化方案。

## 调度与数学

每CTA一个warp负责32个key，query按64逆序遍历。每个query tile先处理高16key，
再处理低16key；两个半块分别保留right VState。高半块从下一32-key checkpoint
读取实际G bottom边界，低半块读取刚生成的高半块HState。末端G=0。
HState仍编码逻辑query列1..64，不能直接按普通底行加载。

- A/dO完整tile走原生permuted TMA，尾部guarded shared copy，单缓冲。
- 行hard决策在warp内以两个ballot重放，不新增global mask或消费generator。
- `rescan.cuh`为已验证GLX重算的局部诊断副本，真实MMA→标量roll→duplicate→inclusive scan。
- dP按DV32维分段MMA，使用当前tanh sigmoid计算alpha，E保持FP32 exp2。
- `reverse_inclusive_scan`恢复全部G，包含hard行；测试仅在此后屏蔽hard列生成Gsoft。
- G为自然logM梯度，梯度GEMM只乘sm_scale，无额外LOG2E。

这里B/V每半块重新staging，未实现寄存器长期驻留、producer组、双缓冲或mailbox。
这些简化是为了隔离边界与布局；不是完整WS实现的性能模型。

## 后续GEMM诊断

同一个用例把真实重建的Gsoft按16-key groups传给`glx_da_tma`：

- dA：shared转置读取→16x32 MMA→原生TMA FP32 add，多warp/CTA累加。
- dB：新单warp独占16keys的probe，遍历query64块，FP32累加后唯一写回，无atomic。

两者先将Gsoft转为BF16一次，独立核对相同量化输入的FP64 GEMM；另记录与FP32
重建Gsoft执行GEMM的relative L2/cosine差异。后者是量化诊断，不是完整reference
autograd验收。真实G核对采用相同重建W、FP64 dP、精确sigmoid的独立逐元素递推，
包含padding。未验证embedding backward或dLSE/drtau的CUDA归约。

## 资源

G probe九实例全零CALL/stack/spill，原生TMA/TANH保留，registers/thread：

| D \ DV | 32 | 64 | 128 |
|---|---:|---:|---:|
|32|227|216|250|
|64|230|234|232|
|128|168|192|202|

单warp无需setmaxnreg，不能直接套用生产232预算。shared为10–40 KiB加16 B。
dB probe D32/64/128为40/64/154 registers、shared6144/10240/18432 B，零spill/CALL。
未测性能，未做12-warp融合，未测sm90。

## 测试

`tests/test_dism_v2_g_recompute.py`：54项九维度×两方向×soft/mixed/hard，
30项N1/17/31/32/63/64/65/129/513/1025的chain/break/bounded_soft及random direction，
另1项codegen；所有数值项均检查G和dA/dB GEMM，并检查RNG状态未消费。
`test_dism_v2_da_tma.py`继续保留63项合成Gsoft/输出canary测试及codegen。

```bash
MAX_JOBS=2 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest tests/test_dism_v2_g_recompute.py tests/test_dism_v2_da_tma.py -q
```

上述149项通过；连同单warp dV和WS共409项，403通过/6原有P量化精度失败。
日志`/tmp/dism-g-gemm-tests.{log,xml}`；资源日志`/tmp/dism-g-recompute-build.log`、
`/tmp/dism-db-probe-build.log`，G SASS `/tmp/dism-g-recompute.sass`。

最终85项重跑指标（`/tmp/dism-g-gemm-final.xml`）：G最大绝对误差2.47260e-4；
同BF16 Gsoft的dA/dB GEMM误差分别2.84749e-6/1.89835e-6。
相对FP32重建Gsoft的量化诊断：dA/dB最大relative L2分别0.00292798/0.00295391，
最小cosine分别0.999995866/0.999995815；尚未据此定义完整训练梯度验收阈值。
三类sanitizer各149项通过、零errors/hazards，插桩过滤为g_probe及da_probe命名空间，
日志`/tmp/dism-g-gemm-{memcheck,racecheck,synccheck}.log`。
