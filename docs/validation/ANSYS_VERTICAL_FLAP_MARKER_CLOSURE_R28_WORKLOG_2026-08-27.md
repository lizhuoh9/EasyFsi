# ANSYS 竖直薄板 marker closure r28 工作日志

日期：2026-08-27
状态：可供代码与证据审查的 WIP；exact FSI50 尚未通过

本日志记录 `r23`--`r28` 的 marker-closure 诊断、最小修复、真实运行耗时和
新的 fail-closed 边界。它是
[`ANSYS_VERTICAL_FLAP_IQN_KALMAN_ACCELERATION_PROTOCOL_2026-08-26.md`](ANSYS_VERTICAL_FLAP_IQN_KALMAN_ACCELERATION_PROTOCOL_2026-08-26.md)
之前的鲁棒性门槛，不构成 Fluent parity、正式加速或 FSI5000 结论。

## 源代码与不变量

- 权威工作树：`/home/zhuohengli/worktrees/HIBM-MPM-r21-validation`。
- branch：`codex/closure-diagnostic-r23`。
- r27/r28 solver source：`c80cb1fba7cc550f58699dcaab9ae9d126dda6a7`。
- 关键提交：
  - `f000c26`：closure trajectory diagnostic 与 RED 测试；
  - `0327749`：恢复 diagnostic wrapper 的 repo import path；
  - `c80cb1f`：按初始 adjustable constraint 数量给 closure recovery
    提供有限预算。
- `dt_s=5e-4 s` 未缩短；每个 accepted macro step 中 fluid 与 solid 都必须各自
  接受完整 `5e-4 s`，remaining time 必须为零。
- closure tolerance 保持 `1.1e-6 m/s`，每 batch 仍为 64 sweeps；修复没有放宽
  tolerance，也没有关闭 invalid/failure/nonfinite/budget fail-closed。

## 为什么原先的耗时判断错误

“两步约 28.7 分钟，因此 50 步是小时级或十小时级”不是当前 profile 支持的
结论。短跑把一次性 Taichi 编译计入了极少数步，不能线性外推。

| run | 结果 | runner/solver elapsed | launch-to-exit | 证据边界 |
|---|---:|---:|---:|---|
| r27 | 36/36 | `456.5719823 s` | `460.2890045 s` | exact 36-step PASS |
| r28 | 41/50 后失败 | `510.3919388 s` | `518.23 s` | exact 50-step FAIL |

r28 冷启动到第一个 accepted step 为约 `208.26 s`；step 2 到 step 10 增量约
`48.82 s`，即该段约 `6.1 s/step`。后续拓扑变化使部分单步增至约 8--12 秒，
但整个 41-step prefix 仍是约 8.5 分钟。当前证据只支持“分钟级到十几分钟级”，
不支持“十小时级”。因为 r28 在 step 42 前失败，本日志不把估算值冒充完整
FSI50 实测。

## r24 closure 轨迹与 r27 通过

r24 请求 36 步，完成 35 步后在 step 36 的 topology rebuild 中失败：

```text
HIBM-owned hard target marker compatibility closure did not converge before canonical commit
```

该调用的 adjustable residual 轨迹为：

| completed sweeps | residual (m/s) |
|---:|---:|
| 0 | `1.5580164268612862e-2` |
| 64 | `9.85415535978973e-4` |
| 128 | `6.026035407558084e-4` |
| 192 | `3.6848464515060186e-4` |
| 256 | `2.2530098794959486e-4` |

首批之后的比值约为 `0.6115`，属于稳定的慢收敛，而非 nonfinite、invalid、
failure 或停滞。固定四个 batch 是提前截断的预算墙。`c80cb1f` 将最大 batch
数量改为 `max(4, initial_adjustable_constraint_count)`；达到原 tolerance 后仍立即
退出，预算耗尽仍 fail-closed。runner health gate 使用同一上限。

验证证据：

- 两个针对旧上限的 RED 场景均能击穿旧实现；
- focused GREEN：`2 passed, 13 subtests passed`；
- 受影响 contract 文件：`52 passed, 182 subtests passed in 8.37s`；
- core/runner `py_compile` 通过；
- 独立审查无 P0/P1/P2 阻塞项；
- r27 strict-CUDA/f32/seed0 完成 exact 36/36，final time `0.018 s`。

r27 step 36 的 final closure residual 为 `9.55791051637789e-7 m/s`，低于
`1.1e-6 m/s`；fluid/solid accepted time 均为 `5e-4 s`，remaining 均为零。
该结果验证了当前故障点，但不是 FSI5000 收敛证明。

## r28 exact FSI50 的新 fail-closed 边界

r28 使用与 r27 相同的 source-matched r26 preflow snapshot、strict CUDA/f32、
seed 0 和 A0 (`carry_forward`, IQN reuse off, Kalman off) 配置，只把请求步数改为
50。结果为：

- `FORMAL_EXIT_CODE=1`；
- 41 个 `step_fields/*.npz` 和 41 个 `step_history/*.json`；
- final accepted physical time `0.0205 s`；
- 没有 completed summary，也没有第 42 步 artifact；
- r28 closure trace 不存在，因为本次失败不是 closure convergence error。

最后 accepted 的 step 41 仍满足：

- fluid accepted/remaining：`5e-4 / 0 s`；
- solid accepted/remaining：`5e-4 / 0 s`；
- pressure CG converged，64 iterations，relative residual
  `4.628446864280993e-7`，`pressure_solve_failed=false`；
- marker closure final residual `9.55530367718893e-7 < 1.1e-6 m/s`；
- closure constraint/adjustable count `77/77`，solve count `2`；
- solid reject/retry/deformation-clamp count 均为零。

下一 trial 在 topology rebuild 的 canonical ledger 预提交处正确 fail-closed：

```text
conflict_source = prepare_author_cardinality
component_face = (0, 9, 32)
component_axis = 2
claim_count = 3
author source rows = (0, 9, 31), (0, 8, 31), (0, 9, 32)
author routes = direct, topology relocation-shadow, direct
relocation geometry base = (0, 9, 31)
projection segments = (5, 6), (4, 5), (4, 5)
serialized z targets ~= -0.183596, -0.143864, -0.141830 m/s
```

三个 author 来自同一 region 101 的连续 marker 折线，但它们不是三个独立 direct
rows：中间项是第一个 direct row 的 topology relocation shadow。当前已证明的
多作者模型只接受两作者 pair，或严格的 inactive-axis extrusion copy cohort；尚未
证明 active component 方向的 direct/shadow/direct cohort。该 shadow 不能仅凭
“看起来属于同一表面”获得有限几何 authority，因此现有拒绝行为应保留，直到
RED 测试和明确的 segment-aggregation/reconstruction 模型成立。exact relocation
identity 只证明 storage transport path，不证明 shadow 与同段 direct row 是同一
physical owner。`3666637` 的 seam helper 只在
既有 `claim_count==2` 且 exact topology 成立时重建 target，没有扩大本次 author
集合。

## 下一步与发布边界

1. 用 r28 的三 author witness 写最小 RED：先保留三个 raw claims，再按已注册
   finite segment 聚合；必须区分“exact relocation transport path 加 adjacent
   direct author”和真正的第三 segment、region drift、malformed weights、
   stale/unowned relocation-shadow 注入。
2. 若数学模型成立，canonical target 必须只由 MAC face 与已验证的 piecewise-linear
   marker geometry 求值，并且与 author 扫描顺序无关；禁止目标平均或放宽
   `1e-6` conflict tolerance。
3. 源码改变后重新生成 source-matched preflow，再按 focused tests -> FSI1/2/8 ->
   exact FSI50 的顺序验证。
4. 只有鲁棒性门槛通过后，才继续协议中的 A0--A3 50-step 研究矩阵。FSI5000 需要
   后续分段 soak、守恒/pressure/closure/physical-time 在线审计和可恢复 checkpoint；
   当前 36-step PASS 与 41-step prefix 都不能替代它。

原始数值产物保留在本地 `validation_runs/` campaign 路径；Git 仅发布可审查代码、
测试和本工作日志，避免把大型 NPZ 当成代码审查接口。
