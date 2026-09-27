# 前向因果裁剪、行元数据缓存与 tile LSE 实验

## 实现

前向 producer/consumer 共用 `min(padded_N, CTA_query_base+128)` key 上界。
不改变双缓冲和配对 mailbox 协议；无效 compute warp 不提前退出。
矩形摘要/边界存储格式不变，被跳过区域仍显式初始化：

- 有效域上三角的 affine summary 为 `(-inf,-inf)`。
- 一个32行对角线完全落在列 padding 中（摘要终点 `j>=N+31`）时为 `(0,-inf)`。
- 被跳过的 W 竖/横边界为 `-inf`。

tau、query label 和 query-row LSE 提前缓存至 key 循环外。
`column_lse` 方向仍按 key 读取 LSE，不误用 query-row 值。
score 的 FP32 运算顺序及 BF16 转换位置不变。

## 近似开关

默认 `DISM_TILE_LSE=full`。实验使用 `DISM_TILE_LSE=tanh`，必须在启动进程前设置。
配置由前后向共享、在模块导入时固定，构建使用独立 `_tanh` 扩展名；不支持进程内切换。
不要将不同模式的低层 forward 状态混用。embedding kernel 不受此开关影响。

严格采用 `/home/cicuvc/cs/projects/rl/lse.cu::approx` 的常数和计算顺序：

```text
x = b+c; y = d
p = fma(abs(x-y), 0.34114549, 0.48232999)
t = tanh.approx(p)
base = max(x,y) + 1.81089463
(a,b) * (c,d) = (a+c, fma(t, -1.81089463, base))
```

为保证 hard 不匹配、零映射和 identity，先显式处理任一 LSE 输入为 `-inf`。
不使用 `approx2` 的 EX2 占位代码，也不使用旧 CUDA 的多项式。
前向摘要/scan 与反向 W 重算共用此近似；跨 chunk `passing` 仍调用完整
`max + log1p(exp2(-abs)) * LOG2E`。反向保留原递推的 sigmoid 梯度，
不是对 tanh 拟合公式逐指令求导；近似非严格结合，转置重算也不保证逐位恢复前向。

## 可复现验证

Python 使用 conda `blkw`。原有 reference 容差不放宽；近似比较单独运行。

```bash
python -m pytest -q tests/test_dism_v2_core.py tests/test_dism_v2_codegen.py tests/test_dism_v2_lse.py
python -m pytest -q tests/test_dism_v2_autograd.py -k 'autograd_wiring or replay_tails or training_backward or all_cuda_no_embedding'
DISM_TILE_LSE=full python -m dism_v2.compare_tile_lse --baseline /tmp/lse-full.pt
DISM_TILE_LSE=tanh python -m dism_v2.compare_tile_lse --baseline /tmp/lse-full.pt
```

训练每种模式、每个 seed 使用独立进程，seed 为 0/1/2：

```bash
DISM_TILE_LSE=full python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 0 --layers 3 --head-dim 64 --qk-vocab 512 --seq-len 1024 --log-every 100
DISM_TILE_LSE=tanh python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 0 --layers 3 --head-dim 64 --qk-vocab 512 --seq-len 1024 --log-every 100
```

沿用 copy_task 模型、AdamW、学习率和 hard 概率日程；BF16 compute、FP32 master，
完整 CUDA embedding 前后向。batch64/H4/D=DV64，固定数据 seed12345、direction/row RNG seed777。
不添加位置编码，不 clamp rtau。CUDA atomic 与训练的离散 label 决策意味着
相同种子也不保证训练逐位确定；三种子短期结果不能证明普遍训练质量或长期稳定性。

## 2026-09-08 实测

RTX 5090 / sm120a，PyTorch 2.13.0+cu130，CUDA编译工具链13.1。

### 正确性与资源

- full 前向/代码生成/近似函数独立probe：111项通过；tanh专用tau反号回归在默认模式跳过。
- full 端到端接线、尾部重放、训练反向与无fallback：513项通过。
- tanh 前向代码生成、尾部重放、训练反向与无fallback：28项通过。
- full 九种D/DV、双方向、混合RNG与N尾部：memcheck/racecheck/synccheck各27项通过，
  0 errors，racecheck 0 hazards/0 warnings。
- tanh CUDA端到端尾部重放：memcheck 12项通过，0 errors。
- 所有前向实例 full/tanh 均无CALL、零stack/local，原生TMA和setmaxnreg保留。
  tanh反向无CALL，但寄存器调度变化：B1 D64/DV64 stack从0增到8B，未调参。
  其他stack变化（full→tanh，字节）：B1 128/32 16→32、64/128 104→128、
  32/128 64→32；单warp B3 128/32 64→72；WS B3 128/32 64→120、64/128 56→24。

新加入N191/255/1025的混合尾部回归，曾检出裁剪区padding summary的first分量错误；
已按上面的identity规则修复，没有放宽原有reference容差。

原语probe扫描abs差值0…32，步长0.001：最大log2 LSE误差0.000591790，
相对CPU tanh拟合公式误差0.0000136184（包含MUFU.TANH自身近似），
完整passing LSE误差0.000000116544。负无穷/identity检查通过。

### 固定输入误差

66项对照包含九种D/DV组合（N139）及D64/DV64的N513/1024，
双方向、hard_prob=0/.37/1、tau=ln(D)、B1/H2/V65/scale=D^-0.5。
比较对象是**相同输入的full CUDA**，不是FP32算法oracle；结果含atomic归约噪声。

| 项 | 最大relative L2 | 最低cosine |
|---|---:|---:|
| O | 0.2095% | 0.9999978 |
| dQ | 0.5987% | 0.9999821 |
| dK | 0.8146% | 0.9999669 |
| dV | 0.2165% | 0.9999977 |
| dQ_vocab | 0.6836% | 0.9999766 |
| dK_vocab | 0.8748% | 0.9999618 |
| drtau | 22.696% | 0.9748284 |

全部输出/梯度有限，但132个非近零head梯度中有一次tau反号。
进一步对照相同CUDA embedding值的FP32 recurrence oracle确认：
`D128/DV32/N139/q_from_k/hard_prob=0/rtau=ln128` 的head0，
full=+2.142464，tanh=-2.066700，oracle=+2.857761。
这不是只相对full版本的符号变化，确实与oracle反号。
已固化为`tests/test_dism_v2_lse.py::test_tanh_tau_sign_at_dimension_bound`，
`DISM_TILE_LSE=tanh`下保持普通失败（该文件1通过/1失败），没有xfail或放宽符号要求。
因此保留full默认，不能宣称近似已经满足全形状梯度精度要求。

完整66项：[tile_lse_numerics_sm120a.json](benchmarks/tile_lse_numerics_sm120a.json)。

### 训练结果

每次1000步，hard-eval使用同一固定数据集（10 batches）：

| Seed | full末100步loss | tanh末100步loss | full最终hard准确率 | tanh最终hard准确率 |
|---|---:|---:|---:|---:|
| 0 | 0.43893 | 0.04609 | 90.330% | 99.147% |
| 1 | 0.17087 | 0.05220 | 97.183% | 99.066% |
| 2 | 0.43178 | 0.25654 | 90.829% | 95.280% |

六次均完成，loss/梯度有限；所有运行max rtau低于ln64。
本批没有看到近似妨碍该D64训练场景，三种子结果反而都更好，
但不能把小样本、非逐位确定的离散训练轨迹解释为稳定的质量提升。
这也不能覆盖上述D128 tau反号问题。
前段没有位置编码的消歧义错误仍计入总准确率；另存跳过前32个预测位置的准确率。

配置、曲线与最终统计：[tile_lse_training_sm120a.json](benchmarks/tile_lse_training_sm120a.json)。

### 性能

上述3层B64/H4/N1024/D=DV64/V512配置，hard_prob=.5，
5步warmup、20步CUPTI采样，每kernel60次launch，表为单次GPU耗时中位数（ms）：

| Kernel | 裁剪+缓存，full | 再启用tanh |
|---|---:|---:|
| 前向摘要 | 0.7445 | 0.6421 |
| 前向passing | 0.0332 | 0.0324 |
| 前向输出 | 1.1364 | 0.9195 |
| dV+反向摘要 | 2.5871 | 2.2856 |
| 反向passing | 0.0487 | 0.0488 |
| dA/dB/dLSE/drtau | 3.7157 | 3.1952 |
| 前向三项之和 | **1.9141** | **1.5940** |
| 六项之和 | **8.2656** | **7.1236** |

此前未裁剪/缓存的同配置前向记录约3.040ms。
本次full优化约降低37%前向GPU时间，tanh在此基础上再降低约17%。
这不是完整训练step延迟（未计embedding、其余模型、optimizer、host间隙）；
中位数之和也不是完整step的中位数。
性能日志：[tile_lse_kernel_profile_sm120a.json](benchmarks/tile_lse_kernel_profile_sm120a.json)。

## 前后向同时启用近似：复核与复测

上一节的tanh实验已经同时启用前向和反向W重算近似，并非仅前向。
本轮再次核对前后向扩展的构建宏、共享LogAffine路径和实际SASS；未改变kernel源码。
D64/DV64实例的静态指令位置数如下（不是动态执行次数）：

| Kernel | full TANH/FFMA | tanh TANH/FFMA |
|---|---:|---:|
| 前向输出 | 0 / 875 | 60 / 215 |
| dV+反向摘要 | 32 / 882 | 92 / 222 |
| dA/dB/dLSE/drtau | 32 / 904 | 92 / 244 |

反向原有32处TANH用于sigmoid；新增加的60处用于W重算中的LSE。
反向add-mul scan本身没有LSE可替换。前向跨chunk log-affine passing仍保留完整LSE，
反向passing仍为add-mul。两种扩展均无CALL；tanh B1 D64/DV64的8B stack spill仍保留。

同配置seed0再次独立训练1000步：末100步平均loss **0.09603**，
最终hard-eval token accuracy **98.155%**，训练完成且梯度有限。
这次结果与上一次相同seed的99.147%不同，进一步说明训练轨迹非逐位确定；
仍没有看到本配置训练发散，但D128 tau反号已知问题并未因此解决。

重新计时使用10步warmup；CUPTI采20个完整训练step，每个主kernel60次launch。
另用独立进程采30个同步完整训练step，包含data/forward/backward/clip/AdamW/scheduler，
不含eval、编译和日志时间；两种模式均hard_prob=.5，embedding前后向均CUDA。

| 单次launch GPU中位数（ms） | full | 前后向tanh |
|---|---:|---:|
| 前向摘要 | 0.7471 | 0.6467 |
| 前向passing | 0.0327 | 0.0329 |
| 前向输出 | 1.1436 | 0.9220 |
| dV+反向摘要 | 2.6014 | 2.2912 |
| 反向passing | 0.0484 | 0.0479 |
| dA/dB/dLSE/drtau | 3.7430 | 3.2035 |
| 前向三项之和 | 1.9234 | 1.6016 |
| 反向三项之和 | 6.3928 | 5.5426 |
| 六项之和 | 8.3162 | 7.1442 |

六个主kernel单层GPU时间之和减少14.1%，不含embedding、delta、tau最终归约等辅助kernel。
整个3层模型训练step同步wall-clock中位数：**68.777ms → 65.455ms**，
input吞吐 **952,874 → 1,001,239 tokens/s**，提升约**5.1%**。
这是本轮两个独立进程各30步的结果，不混用CUPTI时间之和与wall-clock step时间。
原始launch/step样本、训练曲线与配置：
[tile_lse_both_recheck_sm120a.json](benchmarks/tile_lse_both_recheck_sm120a.json)。

## tt_dism_contrun.py 同stream连续填充吞吐

原脚本不修改，B64/H4/N1024/N_HEADDIM64/N_VOCAB64，默认stream0连续10000次
`forward → randn_like(dO) → backward`，叶子`.grad`保持累积，无optimizer/CUDA Graph。
用`python -m dism_v2.benchmark_legacy_contrun`包裹原脚本，先在计时外预热20次，
仅在循环首尾计时和同步，不额外增加逐次同步。保留原tqdm，排除导入、输入构造和JIT。

RTX5090本次实测：总墙钟**97.3292秒**，CUDA event区间**97.3296秒**；
平均每次前后向**9.7329ms**，持续**102.744次/秒**，即**6,733,439 input tokens/s**。
CPU提交循环用97.0301秒，结束后等待GPU排空0.2991秒，吞吐已计入这段drain时间。
事件区间包含可能的GPU空闲间隙，不是六个kernel独立计时之和。

这一数字包含旧wrapper的softmax、随机dO、辅助操作和grad累积，但不包含三层模型、
新版词表512的embedding插值、optimizer，不能直接与前面的完整训练tokens/s相除比较。
原始计时与脚本hash：[legacy_contrun_sm120a.json](benchmarks/legacy_contrun_sm120a.json)。

### CUDA voc_dism 按相同方法连续填充

用户澄清要测CUDA实现后，使用
`DISM_TILE_LSE=tanh python -m dism_v2.benchmark_cuda_contrun`：
B64/H4/N1024/D=DV64/V512，scale1、rtau3、hard_prob=.5、direction=random；
CUDA embedding前向和配对WS反向，前后向W重算均tanh LSE。
20次计时外预热，默认stream0连续10000次前向/随机dO/反向，累积六个叶子的grad；
无逐次同步、CUDA Graph、三层模型或optimizer。

总墙钟**147.2353秒**，CUDA event区间**147.2359秒**，平均**14.7235ms/次**；
持续**67.9185次/秒**、**4,451,106 input tokens/s**。
CPU提交146.7307秒，最终drain0.5046秒已计入；最终输出及累积梯度均有限。
这里包含embedding和全部辅助kernel，不是仅六个主kernel的GPU时间之和；
旧Triton的N_VOCAB64也不同于这里的embedding词表512。
原始数据：[cuda_contrun_sm120a.json](benchmarks/cuda_contrun_sm120a.json)。
