# Dism reverse add-mul 验证

独立sm120a实验，不是生产backward。使用诊断用完整W₂与G缓冲；生产路径仍不物化全矩阵。
CPU生成因果Dism W₂和包含fallback的L₂，然后GPU从相同FP32输入计算：

```
alpha = sigmoid(W₂ * ln(2))
E = exp2(W₂ - L₂) * signed_dP
G[q,k] = E[q,k] + alpha[q,k] * G[q+1,k+1]
```

signed_dP是正负交错的测试信号，不是完整dO/V/delta GEMM；此处只验证reverse affine子问题。
alpha使用`exp2(-abs(W₂))`和一次Newton修正的硬件reciprocal，避免溢出与CALL慢路径。
hard不匹配/因果无效为(0,0)，padding为(1,0)。两分量不同，均须reverse roll，不能套用前向duplicate优化。
GLX AffineComposeOp `(a,b)*(c,d)=(a*c,b*c+d)`，保持自然logits梯度语义，不额外乘LOG2E。

三步：

1. 每CTA八个warp、每warp16key×64query，128key按0,4,1,5,2,6,3,7交错。
   query逆序流式遍历；4→0、5→1、6→2、7→3双槽mailbox传递bottom状态，生成32-key摘要。
2. 每线程沿一条对角线逆序passing，保存各32-key块首行的真实G边界。
3. 相同配对调度重算系数，从passing边界reverse inclusive scan，诊断输出完整G。

reverse HState编码逻辑列1…64：lane=4*l+g对应u0列`8*g+8-l`、u1再加32；
列0从VState的lane0/r0补齐。这与前向HState的-1…62偏移不同。
无效尾warp仍执行全部mailbox协议，只有计算输入变identity、输出受mask保护。

45例：N=1/17/31/64/65/129/139/257/513，soft、混合hard break、全不匹配、rtau=ln64长匹配链、
低score(-16)场景。独立FP64标量递推同时检查摘要first/second、passing和G，最大绝对误差
8.11646005e-7（不是完整attention梯度误差）。摘要/scan/passing分别116/138/36寄存器，
STACK/LOCAL=0，无CALL/LDL/STL。未融合MMA、dV/dB/dA或producer流水，不外推完整反向资源。
45例分别通过memcheck/racecheck/synccheck，零错误，racecheck零hazards/零warnings；
日志`/tmp/dism-reverse-check.ZXq7hw`。

```bash
bash experiments/glx_reverse/run.sh
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_reverse.py
```
