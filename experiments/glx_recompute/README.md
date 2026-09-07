# 真实转置 score 重算验证（sm120a）

本实验验证生产前向保存的竖16/横64标量W₂是否足以支持key-owned反向重算，
不是生产反向kernel。私有测试binding仅接受测试构造的、经前向验证的输入。
仅诊断输出物化padded W用于逐元素断言；生产forward仍不物化W/logM。

每CTA一个warp，持有16个BF16 key（实际core B）；按query tile逆序流式加载64行A。
完整query tile沿用生产5D TMA统一列置换，尾部guarded copy到相同shared布局。
MMA计算`B_tile @ A_tile.T`，accumulator直接按GLX列布局解释。
两个ballot收集每lane生成的两个query决策，在16个key间共用；Philox身份、seed/offset
和已选direction全部重放，不消费新的generator offset，不读取global mask。

计算自然对数soft/hard logits后统一乘LOG2E；先scalar roll再duplicate，padding用identity。
每个query tile重新加载top/left：

- top来自原竖边`vertical[...,kb/16-1,q]`；HState的query索引为`qb+8*g+6-l`，包括-1角点。
- left来自原横边`horizontal[...,qb/64-1,kb+8*r+l]`，仅g=3的VState有效分量需要加载。

无需前一个query tile或者另一个warp传递重算状态。当前实验只验证这一独立性，
未实现12-warp反向流水、反向add-mul、dV/dB累积或dA atomic。

回归：

```bash
MAX_JOBS=2 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_recompute.py
```

RTX5090、CUDA13.1、blkw：96数值用例+1codegen通过。
D=32/64/128、两种固定方向、soft/mixed/break/chain、N=17/65/139/257、B=H=2。
每个padded元素对照FP64 GEMM和独立对角递推，atol=5e-5、rtol=3e-5；
重算竖/横边也与生产导出边界对照。长匹配链rtau=ln(D)。
此实验不依赖DV；输入v固定DV32，生产边界导出另由core测试覆盖全部九种D/DV。

| D | registers/thread | shared bytes（含工具报告的静态开销） | stack/local |
|---|---:|---:|---:|
| 32 | 125 | 6160 | 0/0 |
| 64 | 124 | 11280 | 0/0 |
| 128 | 193 | 21520 | 0/0 |

SASS含原生UTMALDG.5D，无CALL（含REL）、LDL/STL。这里没有setmaxnreg，也没有梯度
accumulator，不能用这些寄存器数字代替完整反向资源评估。尚未做性能测量。

97项测试分别通过memcheck/racecheck/synccheck，零错误、racecheck零hazards/零warnings。
日志`/tmp/dism-recompute-check.3YaK76`。与reference/build/core/codegen及合成边界probe合跑286项通过。
本轮未重跑长序列/embedding精度套件；已有量化相关失败未删除或放宽。
