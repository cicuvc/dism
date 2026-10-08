# 反向测试分层

2026-09-29 用户根据余弦/模长/误差向量检查接受初版正确性，明确要求原严格容差
不改，另外建立较宽松的默认测试。kernel及其默认数值路径没有变化。

## 默认验收

后续性能清理默认`DISM_BACKWARD_DEBUG=0`：dense ca/cb/G输出编译移除，
`test_backward_components.py`的21项因此跳过。需要这些诊断时设置
`DISM_BACKWARD_DEBUG=1`重新运行build.py再测试；测试本身与容差没有删除或修改。
普通梯度验收、重算probe及BF16写回检查不依赖该开关。详见BACKWARD_KEY_STORE.md。

在dism_v3目录运行：

```bash
PYTHONPATH=python:.. /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests
```

`pytest.ini` 默认使用 `-m "not strict_gradient"`。
新的 `tests/test_backward_acceptance.py` 包含：

- 60项真实V512插值输入的core梯度检查：N1/17/65/257/513，soft/mixed/hard，
  tau=2/ln64，seed0/7，四head同时覆盖两个direction。使用默认BF16私有梯度输出，
  对照FP64显式reference。
- 18项B1及18项B3分层检查，沿用原独立LSE压力输入；仅放宽梯度GEMM的
  逐元素容差至atol=.003、rtol=.02，并增加方向/整体误差检查。
  affine摘要、G边界、同状态标量梯度仍使用原阈值。
- 6项完整Triton插值+autograd梯度检查，与原严格端到端测试使用相同fixture。

`tests/gradient_acceptance.py` 集中定义默认全向量判据：

| 梯度 | cosine下限 | relative L2上限 | absolute L2余量 |
| --- | ---: | ---: | ---: |
| Q/K、sq/sk、V、词表 | .999 | .03 | .0001 |
| q_lse/k_lse | .995 | .10 | .01 |
| tau | .95 | .50 | .02 |

实际断言为 `norm(actual-reference) <= absolute + relative*norm(reference)`，
并显式检查模长差；参考模长大于absolute余量时再检查cosine。
理论零梯度仍要求实际模长<1e-6；shape、NaN/Inf检查不放宽。
tau的较大余量用于已接受的BF16 saved-O/delta抵消误差，不表示其它梯度也允许50%。
策略单测另外验证翻转方向、置零、放大两倍、非有限值等错误仍会失败。

此前probe、边界重算、布局、同步、delta、autograd接线、训练smoke等测试仍默认运行，
没有为了得到绿色默认测试而对它们新增skip或xfail。

## 严格诊断（原阈值原样保留）

```bash
# 原四组严格梯度测试，共78项。
PYTHONPATH=python:.. /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests -m strict_gradient
# 包括默认验收与所有严格诊断。
PYTHONPATH=python:.. /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests -o addopts=
```

仅给以下原测试添加marker，没有改函数体或任何阈值：
`test_backward_reference`、`test_summary_readout_gradients`、`test_qk_gradients`、
`test_triton_embedding_end_to_end`。严格失败保持普通FAIL，不改成xfail。

须区分输入域：原ideal-reference压力fixture的Q/K与LSE独立随机生成，不一定是
某个词表插值的联合输出；高tau时能产生持续正的logM和很长的高分递推链。
它仍有诊断价值，不能用新真实插值fixture的通过声称这组压力精度也通过。
补查原fixture的N257/soft/query发现tau cosine约-0.773；N513/soft/key的tau
模长比约2.05。这些不靠进一步扩大默认tau阈值遮蔽，仍由原严格测试明确报错。
实际V512输入的独立/轨迹误差范围见GRADIENT_BIAS.md。

本次验收是用户接受有记录的有限精度初版，不是删除历史误差或保证任意输入的严格精度。

默认全套实测：**757 passed、3 skipped、78 deselected**，16.52秒。
日志：`build/backward_default_acceptance.log`。未启动正式训练，未提交本轮代码。
显式严格组实测：**30 failed、48 passed、760 deselected**，4.37秒。
日志：`build/backward_strict_gradient.log`；marker选择没有隐藏其普通失败。
