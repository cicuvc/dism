# Gsoft shared转置 → dA MMA → TMA FP32 reduction（sm120a）

独立布局/同步probe，不是完整dA/dB backward。输入为诊断用FP32
`Gsoft[groups,16,N]`及BF16 `B[groups,16,D]`；多个group累加同一个FP32 `[N,D]`
输出。正式路径的Gsoft应来自寄存器reverse scan，不应物化本probe的global G输入。

## 数据流

1. 按unrolled GLX16x64的lane/register/element坐标访问Gsoft：
   `key=8*r+lane/4, query=c+8*(lane%4)+32*element`。
   FP32→BF16并写TK `st_bf<16,64>` 的逻辑query列；在这里消除原score MMA的统一列置换。
   本probe不在MMA后用shuffle转置，也不验证真实reverse scan本身。
2. 对每个16-query子块，用TK col-layout读取Gsoft和16x32的B feature子块，
   `mma_AtB`得到FP32 `dA[16,32]`。SASS为`LDSM.16.MT88.4`及`HMMA`。
3. 将scale后的accumulator写到无swizzle的FP32 shared `[16,32]`双槽之一。
   显式lane映射写出，不调用曾存在问题的TK FP32 shared读回。
4. 每个writer执行async proxy fence，再syncwarp，leader发出
   `cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group`并commit。
   实际SASS为`UTMAREDG.2D.ADD`，没有标量atomic回退。
5. 第三次使用及以后，leader先`wait_group.read 1`再syncwarp，保证旧TMA已读完待复用槽。
   退出前`wait_group 0`并syncwarp。全程无跨warp共享scratch，无CTA barrier。

输出tensor map为真实未padding的`[N,D]`，box `[16,32]`（CUDA维度顺序相反），
原生TMA丢弃越界的尾行。N=1/17/63/65等用例检查有效结果及输出末尾256 floats canary，
memcheck也通过；无需给此probe的输出补齐64行或使用scalar atomic尾部回退。

## 编译资源与边界

RTX5090，CUDA13.1，sm120a，blkw；6实例均无CALL/LDL/STL，stack/spill为0。

| D | registers/thread（1或8warps） | shared/warp | shared/8warps |
|---|---:|---:|---:|
|32|72|7168 B|57344 B|
|64|80|8192 B|65536 B|
|128|72|10240 B|81920 B|

shared包含Gsoft、B staging和两个dA输出槽。这里只验证1/8-warp计算probe，
没有12-warp producer组、A/dO ring、dB accumulator、真实W/G重算或配对mailbox。
**不能把这些资源与当前WS kernel简单相加后宣称可用**；融合时需按生命周期规划复用，
特别是TMA尚未读完的输出槽不能提前被producer覆盖。未测性能，未测sm90。

## 回归

`tests/test_dism_v2_da_tma.py`：63个数值用例+1个codegen检查。
覆盖D32/64/128、N1/17/63/64/65/139/257，(warps,groups)=(1,1)/(1,9)/(8,19)，
signed Gsoft、每7行hard-style零列、非零初始输出、多warp/CTA冲突累加、非默认stream及双槽复用。
oracle为BF16量化后的Gsoft和B执行FP64 GEMM，独立报告未量化Gsoft对照误差。
这不代表完整dA训练梯度精度验收；DV在此子路径中不出现，九种D/DV融合验证尚待完成。

```bash
MAX_JOBS=2 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest tests/test_dism_v2_da_tma.py -q
```

64项通过；三类sanitizer分别64项通过、0errors/hazards，插桩过滤`kns=_ZN8da_probe`。
最终数值回归相同BF16 Gsoft oracle最大绝对误差2.37535e-6；未量化FP32 Gsoft
对照最大绝对误差0.0205204、最大relative L2=0.00174365、最小cosine=0.999998480。
后者主要反映本probe主动采用的Gsoft→BF16转换，不应计入布局/atomic实现误差。
日志`/tmp/dism-da-tma-{build,memcheck,racecheck,synccheck,final}.log`，
资源/SASS见build日志和`/tmp/dism-da-tma.sass`，数值指标XML `/tmp/dism-da-tma-final.xml`。
