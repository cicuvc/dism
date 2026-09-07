# Inclusive scan 的竖16边界与转置重算

2026-09-07，RTX 5090 / sm120a，CUDA13.1，外部 GLX 源码未修改。
这是独立布局验证，不是生产 checkpoint 写回或反向 kernel 实现；合成 log2 score 只用于测试。

## 16×64 的直接导出公式

lane=`4*l+g`，l=0…7，g=0…3，r=0/1。
前向 scalar roll 中 register-column `c=7` 的位移为0，因此在
`data.inclusive_scan(left,top)` 后、不进行任何 unroll，就有：

```
data[r][7].second.u0 = W[8*r+l,  7+8*g]
data[r][7].second.u1 = W[8*r+l, 39+8*g]
```

| g | u0列 | u1列 |
| --- | ---: | ---: |
| 1 | 15 | 47 |
| 3 | 31 | 63 |

因而只需 `if (lane & 1)` 的 store，就能取得4条完整竖边，每条16个 FP32 元素。
不需要额外 shuffle，不读取 first 分量，也不从 HState 推断内部竖边。
第63列与返回 VState 的有效 second 分量一致，但15/31/47列必须从 scanned accumulator 抽取。
probe 使用竖边数组 `[key_boundary_index, query_row]`，横边数组 `[query_boundary_index, key_col]`。
这里证明寄存器导出不需 shuffle，不声称存储事务数或性能已经最优。

## 其他 shape

同一规则推广到 C 列的布局，CB=C/8：最后一个 register-column `CB-1` 不做 roll，
对应逻辑列 `(g+1)*CB-1 + e*(C/2)`，e=0/1，逻辑行为`8*r+l`。
选择其中列号模16等于15的位置，即得到竖16边界。R 只改变 r 的范围，不改变选列公式。

| shape | 导出与完整 W | CPU double scan | SHFL：baseline / 导出 | registers：baseline / 导出 |
| --- | --- | --- | ---: | ---: |
| 16×16 | bit-exact | 5/5通过 | 28 / 28 | 37 / 38 |
| 16×32 | bit-exact | 5/5通过 | 38 / 38 | 47 / 48 |
| 32×16 | bit-exact | 5/5通过 | 56 / 56 | 56 / 56 |
| 32×32 | bit-exact | 5/5通过 | 82 / 82 | 80 / 79 |
| 16×64 | bit-exact | 5/5通过 | 62 / 62 | 64 / 64 |
| 16×128 | bit-exact | 5/5通过 | 124 / 124 | 128 / 128 |

六种默认 shape 的最差 double 对照误差4.34e-7。资源数字只是含诊断写回的 scan probe，
不能外推到完整 MMA/softmax/梯度 kernel。baseline 也写出完整 W，从而对照所有相同的 scan 输出。

更新 GLX 后的额外探索结果：

- **32×64**：竖边导出仍逐位等于该 kernel 的完整 W，但原有 inclusive scan probe 在 dense
  非因果用例中与 CPU oracle 不一致，最大绝对误差0.1004318；未定位/修复，不称为数值已支持。
  `-DPROBE_32X64` 保留此可复现实例，返回非零；更新前后最大误差均0.1004318，尚未定位。
- **16×128**：此前受静态断言限制；更新后的 GLX 已允许实例化。现在5/5数值通过，max_abs=4.03883e-7，
  导出边界 bit-exact，无新增shuffle，128 registers、STACK=LOCAL=0，已加入默认回归。
- **32×128**：`-DPROBE_32X128` 的5例数值通过，max_abs=6.38806e-7，边界 bit-exact。
  但此probe baseline/导出均REG=255，STACK=152/144 bytes；SASS分别有40/40条LDL、41/40条STL、无CALL。
  属于有spill的数值探索，未加入默认的零spill codegen gate，也未作为生产tile选择。
  其baseline/导出SHFL分别1948/1947，编译调度并非完全一致，但没有因导出增加shuffle。

## 竖16 / 横64 恢复转置 tile

第二组 probe 先按原坐标16-query×64-key完整 forward scan，导出竖16与横64标量边界。
随后每个独立 CTA/warp 重算一个 `[16 key,64 query]` 转置 tile，query tile 索引逆序：

- top：原矩阵第`key_base-1`列，从竖16数组加载。
- left：原矩阵第`query_base-1`行，从横64数组加载。
- top 仍须按 GLX HState 的 `8*g+6-l+32*e` 编码加载，包含局部列-1的corner。
- left 只在g=3加载 VState，行号为`8*r+l`，first置0，second为真实W₂。
- 每个转置 CTA 只读取稀疏边界与重算的合成score，不读取完整 forward W，不与其他CTA通信。

测试 N=1/17/31/64/65/129/139/257，5种 score：causal soft、含 hard break、全不匹配、
长匹配链、非因果 dense（额外布局压力）。共40例，包含未对齐的有效长度和 padded identity 传播。
forward 输出、独立转置重算均对照 CPU double 每元素递推；所有导出边界也与完整 W 逐位对照。
结果40/40通过，最大绝对误差均为 **6.26466903e-6**，没有因导出改变任何 forward W。

前向循环 baseline/导出版均62条静态SHFL，寄存器分别111/109；转置重算94寄存器。
所有默认 probe 实例无 CALL、无 LDL/STL，STACK=LOCAL=0，ptxas零spill。
更新后的30个单tile用例＋40个跨tile用例均运行 memcheck/racecheck/synccheck：0 errors，racecheck 0 hazards/0 warnings。
未测试 TMA/MMA/多warp流水或性能；这些已有组件将在生产接入后再联合验证。

## 复现

```bash
bash experiments/glx_boundaries/run.sh
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_boundaries.py
```

默认脚本完成编译、数值、三类sanitizer及SASS断言。更新后日志目录 `/tmp/dism-glx-boundaries.59eNHq`。
pytest 固化编译/数值/SASS检查，不在每次 pytest 中重复启动sanitizer。
默认实例不包含32×64失败案例和32×128有spill案例；探索复现可在脚本同样的nvcc参数上增加对应 `-D` 宏。
本次32×128探索程序 `/tmp/dism-glx-new-32x128`；临时文件可能被清理。

## 更新后前向兼容性回归

本次GLX header SHA256：`e232c615b0e96bc478aa2e10d29211ed00142a2f3eb7d0c802596450b0262da8`。
确认扩展的core_fwd.cuda.o已在header修改后重新生成，不是复用旧二进制。

- 首先跑旧生产前向/布局：189项通过，包括九种维度、RNG、summary/boundary、codegen与稀疏边界probe。
- 完整前向套件350通过、44个已知精度失败，报告 `/tmp/dism-updated-glx-forward.xml`。
  对比更新前 `/tmp/dism-all-embedding-precision.xml`：失败集合完全一致，203条精度属性逐条完全一致。
- 106项core用例分别重跑memcheck/racecheck/synccheck，均0 errors，racecheck 0 hazards/0 warnings；
  日志 `/tmp/dism-glx-core-sanitize.QT4goW`。现有16×64布局、HState偏移与roll/duplicate契约没有发现回归。
- `experiments/glx_scan/run.sh` 的旧16×32正/反scan、reduce、16×32/64 log-affine及资源probe全部通过，
  各程序memcheck均0 errors，日志 `/tmp/dism-glx-scan.5IWSMa`。
- `bash experiments/glx_boundaries/run_wide.sh` 复现上游16×128的14项scan（含正/反向、F32/BF16、
  Add/Mul/Affine/predicated affine）及12项reduce检查，全部通过、两个程序memcheck均0 errors；
  日志 `/tmp/dism-glx-wide.zTiXWw`。reduce检查边界与inclusive scan相同且不修改原tile，非独立CPU oracle。

尚未将16×128接入生产MMA/TMA流水，其完整CTA资源需另测；不能根据单独scan的128寄存器断言最终不会spill。
