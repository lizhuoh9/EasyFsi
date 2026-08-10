# Turek–Hron FSI1 线程交接（2026-07-13，v25 后）

## 0. 当前结论

本线程没有完成 FSI1 的正式 220 步门禁，因此 **禁止启动 1600 步正式运行**。

当前最好的正式证据是 v25：

- 完成 183 个物理步；
- 第 184 步使用完 24/24 个合法强耦合 trial 后 fail-closed；
- 最佳真实 interface velocity RMS：`1.0659957986893416e-4 m/s`；
- 正式绝对门：`1.0e-4 m/s`；
- 还差 `6.59957986893416e-6 m/s`，即约门槛的 `6.60%`；
- `physical_state_restored=true`；
- 没有 summary，不能续跑，也不能称为正式 FSI1 结果。

v25 已证明最新的 near-band strict-decrease 修复按设计生效。现在的瓶颈不再是“真实小幅改善被停滞守卫拒绝”，而是：每次微小接受后，新的 full-history IQN 方向仍发生约 `2.56×` 残差回归，随后 Picard 又要花完整回溯梯级才得到约 `0.3%–0.8%` 的改善，24 个合法 trial 不够继续推进到 `1e-4`。

下一线程不应再围绕“第 184 步”硬编码，也不能放宽容差或增加预算。首要可证伪任务见第 9 节。

---

## 1. 工作目录、解释器与保护规则

工作目录：

```text
D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored
```

可靠解释器：

```text
D:\working\taichi\env\python.exe
```

重要规则：

1. 工作树很脏，包含大量与 FSI1 无关的用户改动、未跟踪文件和验证产物。不要 `git clean`、不要 reset、不要 checkout 覆盖。
2. 同时只能运行一个 Turek–Hron Python 进程。
3. 每次正式运行必须使用新的输出目录；失败目录不得续跑或覆盖。
4. checkpoint/resume 只能作为诊断 probe。此前确认 scratch state 会使恢复轨迹与不间断正式运行分叉，因此正式门必须从 step 1 uninterrupted 运行。
5. 只有严格 220/220 通过后才能启动 1600 步；当前仍禁止。

---

## 2. 不可更改的严格 FSI1 配置

```text
preset                         fsi1
steps                          220（门禁）
dt                             0.005 s
grid                           4,48,288
markers                        100
projection iterations          4000
predictor substeps             1
pressure preconditioner        fv_multigrid
coupling accelerator           iqn_ils
base coupling trials           16
near-band factor               1.25
near-band extra trials         8
maximum legal trials           24
relative tolerance             1e-3
absolute RMS tolerance         1e-4 m/s
require coupling convergence   true
initial relaxation             0.5
```

严禁：

- 放宽 `1e-4 m/s`；
- 把 near band 当收敛；
- 增加 16/24 trial；
- 只看 relative residual；
- 硬编码第 184 步、某个 beta、位移、压力或速度；
- 用 1/2/200 步 probe 冒充 220 步门禁；
- 跳过 220 直接跑 1600；
- 转去 FSI2/FSI3。

---

## 3. 为什么问题经常出现在约第 184 步

“184”不是固定的 off-by-one 或固定代码分支。

证据：

- v19 通过 step 184/185，失败于 186；
- v21/v22 失败于 184；
- v23/v24 在 step 183 就进入多 trial，之后失败于 184；
- v25 的 step 183 又恢复为 1 trial，但 step 184 失败。

相同代码和相同命令从 step 1 就会出现微小 CUDA/f32 归约分叉，早期误差随后在 added-mass 强耦合敏感区被放大。因此应修复“对微扰不鲁棒的通用方向/接受策略”，不能修补固定步号。

---

## 4. 为什么没有把整个求解器直接改成 f64

当前采用：

- Taichi/GPU 主状态仍为 f32；
- 位置增量和误差反馈使用 f64 host/intermediate 计算；
- f32 舍入余量保存为 compensation，在后续子步回灌。

实现位置：

```text
simulation_core/solids/neo_hookean_mpm.py
```

原因：

1. 整机 f64 会改变全部 Taichi field、原子操作、kernel、内存占用和 GPU 吞吐，不是局部修复。
2. 主要风险是长期小位移增量被 f32 状态吞掉，不要求所有流体/固体数组永久 f64。
3. 全局 f64 会显著拉长 220/1600 运行时间，但不能自动修复 IQN 方向质量或状态机逻辑。
4. 当前补偿方案保留 GPU resident f32 数据布局，同时把最敏感的累计误差留在 f64 中间量。

这不意味着 f64 没价值；若未来要做全局 f64，应作为独立迁移项目验证性能、Taichi kernel/atomic 支持和 f32/f64 基线，而不是作为第 184 步临时开关。

---

## 5. 本线程保留的实现

主要文件：

```text
cases/turek_hron_fsi.py
simulation_core/solids/neo_hookean_mpm.py
tests/cases/test_turek_hron_fsi.py
tests/cases/test_turek_hron_fsi_strong_coupling_contracts.py
tests/solvers/test_neo_hookean_mpm.py
```

### 5.1 固体位置误差反馈

- f32 particle state 保持不变；
- f64 计算 `dt * velocity` 和 compensated accumulation；
- 保存无法写入 f32 position 的剩余误差；
- 有 CPU/solver focused tests。

### 5.2 IQN/Picard 状态机与诊断

当前代码包含并测试：

- scale-aware IQN initial beta；
- 真实 residual evaluated line search；
- rejected trial 不进入 accepted secant history；
- Picard accepted relaxation memory 与 `0.00625` floor；
- 保留干净 accepted secants；
- velocity/Neumann-gradient 成对更新；
- bounded near-band continuation 与 global-best cold recovery；
- transition checkpoint/diagnostic（只用于 probe）；
- failure artifact 持久化与 physical-state restore；
- late-budget leave-newest-out（LNO）候选、selector 与 counterfactual diagnostics；
- near-band strict-decrease policy。

### 5.3 automatic backtracked-IQN newest-column rollback 已撤回

v23 中旧 automatic rollback 真正触发后，下一 IQN 的所有 beta 都没有产生真实下降。按照预设判废标准，本线程已从 runtime 撤回该自动策略。

当前 runtime：

```text
exclude_newest_secant_for_proposal = False
```

只有窄 LNO selector 真正选择 alternate 时才会置为 true。旧 `_iqn_ils_newest_secant_rollback_report` 纯 helper/单测仍存在，是 P2 清理债，但 runtime 不调用。

### 5.4 near-band strict-decrease policy（v25 最新）

实现位置约为：

```text
cases/turek_hron_fsi.py
  _iqn_ils_stagnation_rejection_policy_report
```

规则：

- 仅当 `phase == iqn`；
- 且此前已真实登记的 best 满足 `tol < best <= 1.25 * tol`；
- 才关闭 tiny-step 的“material reduction”否决；
- evaluator 仍要求 `observed < source`，持平/回归仍拒绝；
- absolute gate 仍独立要求 `observed <= 1e-4`。

v25 已实证该策略生效，不是只通过单元测试。

---

## 6. 测试状态

最新绿结果：

```text
D:\working\taichi\env\python.exe -m pytest -q tests/cases/test_turek_hron_fsi_strong_coupling_contracts.py
186 passed
```

```text
D:\working\taichi\env\python.exe -m pytest -q \
  tests/cases/test_turek_hron_fsi.py \
  tests/contracts/test_generic_fsi_solver_architecture.py \
  tests/validation/test_turek_hron_acceptance.py \
  tests/cases/test_turek_hron_fsi_coupling_guards.py \
  tests/cases/test_turek_hron_fsi_mechanism_probe.py
177 passed, 70 subtests passed
```

另外：

- `python -m py_compile cases/turek_hron_fsi.py` 通过；
- `git diff --check` 通过，仅有 LF/CRLF warning；
- 两次独立只读 review 均报告 P0=0、P1=0。

pytest cache 因工作目录权限产生 WinError 5 warning，不是测试失败。

---

## 7. v19–v25 正式运行证据

| 运行 | 完成步 | 失败步 | trials | 最佳绝对 RMS | 关键结论 |
|---|---:|---:|---:|---:|---|
| v19 | 185 | 186 | 16 | `1.618884301965e-4` | step184/185 可过；定位到 backtracked IQN 后 enriched model 失败 |
| v20 | 0 | — | — | — | 直接脚本启动导致 `ModuleNotFoundError`；无效启动，不得引用 |
| v21 | 183 | 184 | 16 | `1.331155388942e-4` | newest rollback 从未触发；证明 CUDA 轨迹分叉 |
| v22 | 183 | 184 | 16 | `1.311283143238e-4` | 同代码第二次失败；sub-min IQN 多次回归 |
| v23 | 183 | 184 | 24 | `1.212989347263e-4` | automatic rollback 触发但无下降；cold recovery 唯一改善 |
| v24 | 183 | 184 | 21 | `1.191774347785e-4` | full history 更好；真实 2.31% tiny decrease被停滞守卫拒绝 |
| v25 | 183 | 184 | 24 | `1.065995798689e-4` | near strict-decrease 修复实证有效；还差约 6.60% |

重要目录：

```text
validation_runs/turek_hron_fsi/repair_20260713/
  fsi1_220step_mg_iqn16_recovery_memory_uninterrupted_formal_v19/
  fsi1_220step_mg_iqn16_newest_secant_rollback_uninterrupted_formal_v21/
  fsi1_220step_mg_iqn16_newest_secant_rollback_repeat_uninterrupted_formal_v22/
  fsi1_220step_mg_iqn16_late_budget_lno_uninterrupted_formal_v23/
  fsi1_220step_mg_iqn16_full_history_lno_uninterrupted_formal_v24/
  fsi1_220step_mg_iqn16_near_strict_decrease_full_history_uninterrupted_formal_v25/
```

每个失败目录以 `turek_hron_fsi_coupling_failure.json` 和 `turek_hron_fsi_history.csv` 为准，不要根据目录名推断某策略一定触发。

---

## 8. v25 第 184 步精确证据

v25 前 183 步全部 1 trial。第 184 步：

| Trial | Phase | beta | absolute RMS | 结果 |
|---:|---|---:|---:|---|
| 1 | initial | — | `6.899758726925e-4` | source |
| 2 | Picard | 1 | `1.667304544866e-4` | accept |
| 3 | Picard | 1 | `2.784991745597e-4` | reject |
| 4 | Picard | .5 | `1.263195226211e-4` | accept |
| 5–7 | IQN | 1/.5/.25 | `3.154e-4` → `2.758e-4` | reject |
| 8 | IQN | .125 | `1.118053011556e-4` | accept, near band |
| 9–12 | IQN | 1/.5/.25/.125 | `3.178e-4` → `2.751e-4` | reject |
| 13–15 | Picard | 1/.5/.25 | `2.794e-4` → `2.738e-4` | reject |
| 16 | Picard | .125 | `1.077750089215e-4` | accept |
| 17 | IQN | .015625 | `1.069573056905e-4` | **accept，tiny strict decrease policy 生效** |
| 18 | IQN | .015625 | `2.736687004294e-4` | reject |
| 19–21 | Picard | 1/.5/.25 | `2.737e-4` → `2.731e-4` | reject |
| 22 | Picard | .125 | `1.065995798689e-4` | accept，global best |
| 23 | IQN | .00390625 | `2.731467112332e-4` | reject |
| 24 | Picard | 1 | `2.733330864343e-4` | reject，budget exhausted |

### 8.1 最新修复的实证

T17：

```text
near_tolerance_refinement = true
enforce_stagnation_rejection = false
strict_residual_decrease = true
stalled_model_detected = true
stalled_model_rejected = false
accepted = true
```

因此 v24 的“真实小幅改善被拒绝”问题已经修复。

### 8.2 LNO 为什么没有执行

在 T9–T16 链中：

- 完整 IQN 注册 beta 全拒绝；
- Picard 回溯后 T16 接受；
- `latch_candidate=true`；
- 下一 full-history IQN initial beta 为 `.015625 < .125`。

但是现有 LNO gate 报告：

```text
action = keep_full_history_proposal
completed_trials = 16
remaining_base_trial_slots = 0
late_budget_slot_limit = 5
best = 1.077750089215e-4
```

它只按 base slots 和 near-band 外条件触发；此时 near extra 已合法授权，但 gate 没把这些 legal slots 纳入，因此 LNO alternate 没有计算、没有 counterfactual、没有 exclusion、没有消费 one-shot。

v25 所有 `newest_secant_excluded_count=0`，automatic rollback 也确实不存在。

---

## 9. 下一线程首要任务

### 9.1 推荐的第一项 TDD：修正 LNO gate 的 legal-slot/near-band 语义

这是 v25 留下的最窄、尚未实测的既有 alternate，不保证通过，但比继续增加 trial 或盲目缩 beta 更有证据。

目标：

1. LNO 已经 latch；
2. 当前 full-history IQN 需要 sub-min beta；
3. best 已经满足 `tol < best <= 1.25*tol`；
4. base 16 已用完，但 near extra 8 已合法授权；
5. selector 只在 alternate 仍为正常 IQN、`excluded_count==1` 时采用；否则零 trial 成本 fail-closed 回 full history。

当前 `_iqn_ils_late_budget_leave_newest_out_report` 不应只看：

```text
remaining_base_trial_slots = base_budget - completed
best > near_limit
```

下一线程应先写 RED，使用 v25 T16 snapshot：

```text
prior_transition_eligible = true
current_update_mode = iqn_ils
full_history_initial_beta = 0.015625
retained_secant_column_count >= 3
completed_trials = 16
base_iteration_budget = 16
best_absolute_residual_mps = 1.077750089215e-4
absolute_tolerance_mps = 1e-4
already_attempted = false
```

期望：

```text
near extra authorized = true
legal trial limit = 24
remaining legal slots = 8
action = recompute_without_newest_secant
```

必须保留的负边界：

- best 未进入 near band且没有合法剩余槽；
- best 已正式 `<= tol`；
- latch=false；
- current mode 不是 IQN；
- retained columns 不足；
- one-shot 已消费；
- alternate rank/trust/recovery fallback；
- alternate 没真正排除一列。

必须证明：

- LNO 重算本身不消费 physical trial；
- history list 不切片、不改写；
- source 仍是当前 accepted Picard 的 velocity/candidate/residual；
- gradient 使用 selected LNO diagnostic 重新配对；
- selected ratio/scale beta 来自 LNO，不是 full-history counterfactual；
- 16/24/`1e-4` 完全不变。

### 9.2 如果 near-band LNO 正式实测仍失败

不要再叠加 line-search 调度 heuristic。应转向新的方向生成证据，例如：

- 保存 full residual vector、secant matrices、SVD spectrum 和 selected coefficients；
- 比较 full-history、LNO、regularized IQN 的 predicted step/correlation；
- 设计带 Tikhonov/LM 或 residual-vector trust-region 的可证伪方向；
- 仍由下一次真实 nonlinear map RMS 决定接受，不能按模型预测宣布收敛。

### 9.3 不推荐的下一步

- recovery-first：v23 已表明它只平移已观测路径，不创造下降方向；
- 继续更小 Picard beta：会趋回 source，v25 每轮改善已降到 0.33%；
- 恢复 automatic backtracked-IQN rollback：v23 实测没有下降；
- 全局 f64 临时开关：不能修复方向质量，且改动/成本过大；
- 增加到 25/32 trials：违反门禁；
- 因 best 只有 `1.066e-4` 就宣称通过：不允许。

---

## 10. 下一次正式 220 步命令模板

必须使用 module entry；不要直接运行脚本路径。

```powershell
& 'D:\working\taichi\env\python.exe' -u -m cases.turek_hron_fsi `
  --preset fsi1 `
  --steps 220 `
  --dt-s 0.005 `
  --grid-nodes 4,48,288 `
  --output-dir validation_runs\turek_hron_fsi\repair_YYYYMMDD\NEW_UNIQUE_DIR `
  --projection-iterations 4000 `
  --fsi-coupling-iterations 16 `
  --fsi-coupling-tolerance 0.001 `
  --fsi-coupling-absolute-tolerance-mps 0.0001 `
  --fsi-coupling-accelerator iqn_ils `
  --require-coupling-convergence `
  --fsi-aitken-initial-relaxation 0.5 `
  --flow-predictor-substeps 1 `
  --flow-cg-preconditioner fv_multigrid `
  --history-flush-interval-steps 1 `
  --transition-checkpoint-step 185
```

运行前：

1. 检查没有另一个 `cases.turek_hron_fsi` 进程；
2. 确认输出目录不存在；
3. 核对完整 CommandLine；
4. 不要 silent restart；
5. 失败后先审计 artifact，再改代码。

---

## 11. 220 步通过后的唯一下一步

只有同时满足以下条件，下一线程才能启动 1600：

- history 恰有 220 行；
- 没有 coupling failure artifact；
- 每步 `fsi_coupling_converged=True`；
- require-absolute 模式下每步 absolute RMS `<=1e-4`；
- projection CG、marker validity、OOB、fixed root、flux/physical guards 均通过；
- summary 与 history 一致；
- 运行是不间断正式运行，不是 checkpoint probe。

在此之前，FSI1 状态仍是：**未完成，1600 禁止。**
