# 反向维度支持扫描

前一阶段提交`54b3fdb`，16-key摘要默认关闭（默认32-key），B2仍默认四级流水。
用户接受反向初版完成；本轮优先8组R16/32×D32/64×DV32/64，再额外尝试
R16/D128/DV64。尚未进行sm90验证。

## 参数化

- `DISM_READOUT_DIM=16/32`（默认32）控制前后向soft Q/K的R。
- `DISM_KC_KEY_DIM=32/64/128`（默认64）控制前后向score的D。
- `DISM_KC_HEAD_DIM=32/64/128`（默认64）控制V/O/反向delta的DV。
- 每个扩展一次只编译一组维度，Python和CUDA拒绝不匹配输入；不是runtime dispatch。
  `forward_readout_dim()`、`summary_key_dim()`、`forward_head_dim()`报告当前配置。
- 前向、反向及保存的operand/边界使用同一构建；Triton词表插值保持不变。

必要适配：R16的private BF16输出使用16通道panel，其他使用32通道；都复用
既有transpose slot。dQ/dsq的TMA选择改为明确的用途bit，不用Channels==R/D，
避免R=D32混淆。D128的16个query-output tiles按owner,owner+8分两轮输出，
每warp仍只有一个TMA共享槽，覆盖前read-wait。

主8组都保留双输入缓冲。额外D128组原布局需104KiB，超过本机99KiB，因此仅
该D128路径使用单输入槽（91KiB）。未为此调整主配置的scan tile或寄存器预算。

## 验证定义

`tests/test_backward_dims.py`每组39项：36组真实V512插值输入，N1/17/33/65/
129/257，soft/mixed/hard、tau=2/ln(D)，4head同时覆盖两方向；比较FP64输出
和显式反向oracle，并检查BF16私有梯度是否逐位等于FP32结果的RN转换、delta
的FP32归约。另3项覆盖Triton插值和8种输入/词表梯度的autograd链路。
使用原默认余弦/模长/误差容差，不为维度扫描放宽。

记录cosine、norm ratio、relative L2（包含tau）；这是固定seed的有限输入扫描，
不能证明长期训练无系统性偏差。新增失败会保持pytest失败，不自动加入skip/xfail。

脚本`tools/check_backward_dims.py`按主8组→额外组执行，保存编译资源、SASS、测试和
sanitizer日志到`build/backward_dims`，结束恢复R32/D64/DV64、32-key摘要。
使用conda blkw执行，默认TMA7、shared BF16写回、tanh、forward WarpKSize64。

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python tools/check_backward_dims.py
```

`tools/check_dimension_delta.py`仅作诊断：在相同kernel/输入/normalizer下替换为FP64
oracle输出计算的delta，检查tau误差是否来自生产BF16 O/delta；不修改生产路径。

## 扫描结果

以下为B1/B3的每CTA shared、输入槽数和静态stack字节；不是动态spill流量。

| R | D | DV | shared KiB | 槽数 | B1 stack B | B3 stack B | 数值测试 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 32 | 32 | 56 | 2 | 160 | 160 | 39/39 |
| 16 | 32 | 64 | 68 | 2 | 288 | 192 | 38/39，tau |
| 16 | 64 | 32 | 68 | 2 | 184 | 208 | 39/39 |
| 16 | 64 | 64 | 80 | 2 | 216 | 224 | 38/39，tau |
| 32 | 32 | 32 | 62 | 2 | 168 | 168 | 39/39 |
| 32 | 32 | 64 | 74 | 2 | 336 | 208 | 39/39 |
| 32 | 64 | 32 | 74 | 2 | 176 | 264 | 39/39 |
| 32 | 64 | 64 | 86 | 2 | 360 | 360 | 39/39 |
| 16 | 128 | 64 | 91 | 1 | 360 | 480 | 38/39，tau |

所有9组编译/运行成功，forward/B1/B3 SASS均无CALL。B2与dimension无关，仍4级。
每组memcheck/synccheck/racecheck各选定4项（mixed、N17/129、两个tau值）通过，
无错误或hazard；racecheck只过滤反向kernel。初扫因数值失败未跑的3组已通过
followup补齐；最终状态见`build/backward_dims/results.json`。
R16/D32/DV32首轮命中构建缓存，其资源报告来自同一源码构建的
`build/backward_dims_smoke_build.log`；其余见各组build日志。

| R/D/DV | B1 spill store/load B | B3 spill store/load B |
| --- | --- | --- |
| 16/32/32 | 176/256 | 172/204 |
| 16/32/64 | 300/424 | 204/248 |
| 16/64/32 | 196/280 | 220/264 |
| 16/64/64 | 224/336 | 228/272 |
| 32/32/32 | 180/284 | 176/216 |
| 32/32/64 | 348/496 | 216/260 |
| 32/64/32 | 192/296 | 268/312 |
| 32/64/64 | 384/532 | 368/412 |
| 16/128/64 | 384/516 | 548/592 |

9组主要向量梯度（Q/K/sq/sk/V）的最低cosine约0.99999235，模长比在
0.99636–1.00569之间。所有词表插值autograd检查通过；错误仅来自3个纯hard、
tau=ln(D)的core tau测试，cosine分别为0.907519、0.945712、0.930558。
三例中每head tau符号都与oracle一致，但仍不豁免现有0.95 cosine测试。

### delta归因诊断

保持同一生产kernel、score/normalizer、dO和全部输入，只替换传入的delta：

| R/D/DV、N（纯hard，tau=ln D） | 生产tau误差范数 | oracle-delta误差范数 | oracle-delta cosine |
| --- | ---: | ---: | ---: |
| 16/32/64、33 | 0.03414 | 1.09e-7 | 0.9999999999998 |
| 16/64/64、129 | 0.06392 | 1.25e-6 | 0.9999999999912 |
| 16/128/64、129 | 0.04527 | 1.94e-6 | 0.9999999999611 |

说明这些case的主要偏差来自生产BF16 saved O所形成的delta，而不是维度扩展
造成的G边界或GEMM布局错误。诊断不进入生产；不会以此替代真实oracle失败。
`tools/check_backward_dims.py --followup`重跑失败组、记录上述诊断并补齐sanitizer。

扫描结束已恢复R32/D64/DV64、32-key摘要、TMA7、shared BF16写回、B2四级流水。
恢复后的完整默认回归：950 passed、24 skipped、78 deselected；原严格梯度测试
仍按既有规则单独运行。维度扩展代码尚未提交，前一阶段提交为`54b3fdb`。
