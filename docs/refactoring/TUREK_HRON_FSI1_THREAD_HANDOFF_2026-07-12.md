# Turek–Hron FSI1 新线程交接（2026-07-12）

## 0. 交接结论先读

FSI1 **尚未成功**。当前没有仿真进程，旧心跳监控已经删除；不要续跑任何旧目录，也不要启动 FSI2/FSI3。

本线程已经把最初的单次、不可测耦合推进到严格的强耦合、双投影、完整 marker/守卫证书，并修复了 embedded-boundary velocity reconstruction。140 步严格门曾经 140/140 全部通过。但后续正式运行和多代 220 步门禁都在约第 184 步附近因真实 interface velocity RMS 未达到 `1e-4 m/s` 而 fail-closed。

最新 v4 不是容差、CG、marker、根部夹持或物理守卫失败。它在第 184 步的第 8 次实测试探取得全局最佳 `1.0911772079218964e-4 m/s`，已经进入 `1.25 × tolerance` near-band，但仍高于正式门。之后 IQN 与 Picard 两个方向都被真实 map 判为上升方向；第 16 次后代码把 `line_search_exhausted` 无条件当作终止，覆盖了已经批准的 near-band continuation，因此预留的 8 次额外试探完全没有使用，并把失败误写成 `iteration_budget_exhausted`。

下一线程的首要任务不是“多给几次”或放宽门，而是按 TDD 修复这个状态机接线缺口：在 near-band 且仍有合法 extra slots 时，从 immutable global-best velocity/gradient pair 发起一次明确、全新、受真实残差过滤的冷却 recovery；不能让被拒 trial 污染 secant history，不能回写 normal Picard memory，不能把 near-band 当成收敛。

## 1. 工作区与当前终态

- 工作区：`D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`
- 可靠 Python：`D:\working\taichi\env\python.exe`
- 当前 Python 仿真进程：**无**（本交接生成前已检查）
- 已删除的旧监控：`turek-hron-fsi1-v4`
- 当前最新门禁：
  `validation_runs\turek_hron_fsi\repair_20260712\fsi1_220step_mg_iqn16_velocity_linesearch_picardfloor_reference_reconstructed_v4`
- v4 终态：第 184 步 fail-closed，183 步已完成，`physical_state_restored=true`，无 summary
- 不得续跑的旧正式目录：
  `validation_runs\turek_hron_fsi\formal_20260712\fsi1_1600step_mg_iqn16_reconstructed_snapshot20`
- 可视化最终副本目录（仅正式完成后使用）：
  `C:\Users\lizhu\.codex\visualizations\2026\07\09\019f48d5-def2-74b3-bda2-ac3051455c17`

## 2. 不可改变的硬约束

1. 任意时刻最多一个仿真进程。
2. 固定 `dt=0.005 s`。
3. 固定 base coupling budget `16`。
4. 固定正式 absolute interface velocity RMS tolerance `1e-4 m/s`。
5. near-band 只允许在 `best <= 1.25e-4 m/s` 时使用最多 8 次额外真实试探；成功仍必须实际 `RMS <= 1e-4 m/s`。
6. 不得用 relative tolerance、best point、启动点、单点或未测 residual 冒充正式收敛。
7. 不改物理方程、网格、marker 数、near-band 规则或正式验收参考来“做绿”。
8. `interpolate_velocity_dirichlet_with_interior=True` 是已验证修复，不得回退。
9. `post_solid_no_slip_*` 是截断的一侧近壁流体采样；其幅值不是正式 sharp no-slip gate。只门控 report available、100 valid、0 invalid；正式 sharp 边界由重构 Dirichlet row contract 决定。
10. 未通过 FSI1 前不启动 FSI2/FSI3。
11. 所有终态判断以运行目录 artifact 为准，PID 仅作辅助。

## 3. 最终成功合同

### 3.1 下一轮 220 步严格门禁

- summary 存在且 `completed_steps=220`；
- CSV 恰好 220 个连续、有限数据行；
- 每步 `100 valid / 0 invalid` marker；
- 每步 `stress_one_sided_pressure_marker_count=100`；
- 每步 coupling `measured=true`、`converged=true`、实测 absolute velocity RMS `<=1e-4 m/s`；
- 主投影与 post-solid 投影 CG 全部通过；
- flux、fixed root、scatter、particle bounds、projection、mechanism/physical guards 全部通过；
- 失败时必须回滚 fluid/solid/marker/gradient 并写 failure artifact；不能静默重启。

### 3.2 正式 1600 步

只有新的 220 步门禁严格通过、artifact 复核通过后，才允许创建一个**全新**正式目录并启动唯一一个 1600 步进程。正式运行增加每 20 步一个完整 flow snapshot。

完成后必须用 `src.refactored.validation.turek_hron_fsi.acceptance` 分别验收 `t=4–6 s` 与 `t=6–8 s` 相邻窗口，检查均值、跨度、趋势和相邻窗口漂移。不得用启动阶段或单一末端点当稳态。

## 4. 本线程完成了什么

### 4.1 初始 700 步正式运行的诚实失败分析

最初目录：

`validation_runs\turek_hron_fsi\formal_20260711\fsi1_700step_auto_ramp2_snapshot10`

该运行只完成 400/700 步，正好到 `t=2.0 s` ramp 结束；没有 summary，也没有 post-ramp 稳态窗口。下一结构 trial 触发 MPM out-of-bounds fail-closed guard（1120 个粒子中 3 个越界）。根因不是“网格不够大”，而是此前已出现大幅流固反馈失稳；该运行使用 one-pass coupling，residual 实际未测。

已有完整分析：

`validation_runs\turek_hron_fsi\formal_20260711\fsi1_700step_auto_ramp2_snapshot10\FSI1_FORMAL_FAILURE_ANALYSIS.md`

该文档还包含固定色标、无白色 marker/rest 点的失败诊断 GIF 和 canonical / LS-DYNA 分账。

### 4.2 严格强耦合与 crash-surviving 证书

本线程把 FSI1 推进为每个物理步都必须有真实 measured/converged coupling certificate 的 fail-closed 路径，并持续保留：

- absolute velocity residual 使用 per-marker RMS；
- marker candidate geometry 锚定一个物理时间步；
- rejected trial 不进入 secant history；
- state-machine 异常先回滚 fluid、solid、marker 和 gradient，再持久化 failure artifact；
- 失败前已完成的 CSV 行先 flush；
- 主/后投影 CG、flux、fixed root、scatter、marker completeness、field finiteness 都有显式证据。

### 4.3 embedded-boundary velocity reconstruction 修复

恢复 FSI1 的：

`interpolate_velocity_dirichlet_with_interior=True`

此前旧第 133 步 coupling RMS 约为 `4.88e-4 m/s`；恢复重构行后，第 133 步降到：

`6.717924046138433e-05 m/s`

对应门禁目录：

`validation_runs\turek_hron_fsi\repair_20260711\fsi1_140step_mg_iqn16_reconstructed_rows_schema3`

已核实：summary `completed_steps=140`、CSV 140 行、100/0 marker、全部 measured/converged、最大 coupling RMS `7.00048834885846e-05 m/s`。这是本线程最重要的已通过机制证据，不能回退。

### 4.4 旧正式 1600 步运行

目录：

`validation_runs\turek_hron_fsi\formal_20260712\fsi1_1600step_mg_iqn16_reconstructed_snapshot20`

它在第 184 步失败，只完成 183 行，`physical_state_restored=true`；第 184 步 16 次 trial 的最佳值为 `1.02312153389159e-4 m/s`，仍未达到 `1e-4`。该目录不可续跑，也不能误判成正式结果。

### 4.5 IQN/line-search 修复演进与 artifact

以下均为严格 fail-closed，不是成功：

| 门禁目录（`repair_20260712/` 下） | 失败步 | 完成行 | trials | 失败步最佳 RMS (m/s) | 主要目的/结论 |
|---|---:|---:|---:|---:|---|
| `fsi1_220step_mg_iqn16_globalized_reconstructed` | 185 | 184 | 24 | `1.15764820125388e-4` | globalized IQN 与 near-band extra trial 基线 |
| `fsi1_220step_mg_iqn16_relaxedmap_stallguard_reconstructed` | 183 | 182 | 16 | `1.65069470093287e-4` | relaxed-map secant 与 evaluated stagnation guard |
| `fsi1_220step_mg_iqn16_relaxedmap_stallguard_pairedgradient_reconstructed` | 184 | 183 | 16 | `1.362494599542e-4` | recovery 同步 velocity/Neumann-gradient pair |
| `fsi1_220step_mg_iqn16_relaxedmap_pairedgradient_complement_reconstructed` | 187 | 186 | 16 | `1.97948832340891e-4` | normal 路径 joint relaxed complement |
| `fsi1_220step_mg_iqn16_velocity_linesearch_reconstructed` | 0 | 0 | — | — | 被桌面沙箱回收的零步启动尝试；只有空日志，不是有效运行 |
| `fsi1_220step_mg_iqn16_velocity_linesearch_reconstructed_v2` | 184 | 183 | 16 | `1.37363674978918e-4` | velocity-only evaluated line search；暴露 accepted Picard omega 未记忆 |
| `fsi1_220step_mg_iqn16_velocity_linesearch_picardmemory_reconstructed_v3` | 184 | 183 | 16 | `1.32479111678884e-4` | 同物理步 Picard memory；暴露 omega 可连续缩到 0.05 以下和参考比例漂移 |
| `fsi1_220step_mg_iqn16_velocity_linesearch_picardfloor_reference_reconstructed_v4` | 184 | 183 | 16 | `1.09117720792190e-4` | floor/reference 修复有效；暴露 near-band continuation 被 exhaustion 提前截断 |

### 4.6 v2：evaluated velocity-only line search

实现/验证的合同包括：

- physical fixed-point state 保持 velocity-only；
- 每个 IQN proposal 必须由下一次真实 map RMS 评估；
- 只有严格下降才接受；
- 沿固定原方向使用 `beta=1/2,1/4,1/8` 回溯；
- rejected trial 不进入 secant；
- IQN 方向耗尽后从包含当前观测的 global-best pair 重启 relaxed Picard；
- formal convergence 仍只看原始 measured relative/absolute velocity residual；
- 内部状态机异常完整回滚并原样抛出。

v2 在第 184 步的接受路径为：

`6.9475e-4 → 1.6665e-4 → 1.4638e-4 → 1.3736e-4`

但 accepted Picard effective omega（从 0.5 降到 0.125/0.0625）没有在同一物理步内保留，下一 fallback 又从 0.5 重启，浪费固定预算。

### 4.7 v3：Picard memory

按 TDD 加入：

- pending 显式记录 full Picard omega；
- 真实接受后保存 `effective omega = full_omega × beta`；
- 同一物理步的后续 fallback 与 IQN unmodeled complement 复用；
- 下一物理步从配置 0.5 重置；
- best-recovery relaxation 与 normal memory 独立；
- diagnostics 记录 global trial index、full/effective/accepted omega 和评估后 best residual。

v3 证明 memory 被复用，但 accepted omega 继续几何缩小到 `0.03125`，同时部分 stagnation 比例相对于“当前 memory”而不是固定配置参考值，导致判据语义漂移。

### 4.8 v4：Picard floor 与 configured reference

按 RED→GREEN 完成：

- 真实 accepted Picard omega 原样记录；
- 下一 normal memory 为 `max(0.05, raw_accepted_omega)`；
- 每物理步 immutable configured Picard reference 固定为 0.5；
- normal Picard 和 normal IQN 的步长/停滞比例都相对此配置参考；
- recovery phase 与 normal memory/reference 独立；
- IQN 配置初始 relaxation `<0.05` fail-fast；Aitken legacy zero 保持兼容；
- diagnostics 明确区分 raw/stored/floor/reference ratio。

实现后完成的本地验证：

- strong-coupling contract 全文件；
- 相关 6 文件测试套件；
- `py_compile`；
- diff-check；
- 代码、数值和测试三路只读审查；
- 汇总结果：`213 tests + 33 subtests` 通过。

没有测得覆盖率百分比，因此不要把上述结果改写成“80% coverage 已证明”。

## 5. 最新 v4 artifact 的精确终态

### 5.1 文件

运行目录只有：

- `turek_hron_fsi_history.csv`：183 行；
- `turek_hron_fsi_coupling_failure.json`：第 184 步 failure evidence；
- `stdout.log`；
- `stderr.log`；
- 无 `turek_hron_fsi_summary.json`。

### 5.2 已完成的 1–183 步

已重新核实：

- steps 1–183 连续；
- marker 始终 100 valid / 0 invalid；
- one-sided pressure marker 始终 100；
- coupling 每步 measured/converged，最大 absolute RMS `6.0815596571522e-05 m/s`；
- 主 CG false 数 0；post-solid CG false 数 0；
- fixed-root 最大位移 0；
- scatter invalid 最大值 0；
- post-solid physical failure 数 0。

第 183 步（仅是最后完成点，不是稳态结论）：

- `t=0.915 s`，ramp `0.4334393307367238`；
- tip `ux=2.223253420656556e-05 m`；
- tip `uy=1.5621411415473983e-03 m`；
- total drag `17.243711231283054 N/m`；
- total lift `-2.5078216706531498 N/m`；
- coupling RMS `2.303266583059407e-05 m/s`，1 iteration；
- flux relative imbalance `1.7159861277687244e-07`；
- scatter action-reaction residual `5.40775654666155e-08 N`。

### 5.3 第 184 步 trial 序列

failure JSON：

- `failed_step=184`；
- `failed_time_s=0.92`；
- `completed_steps=183`；
- `completed_history_rows_flushed=183`；
- `physical_state_restored=true`；
- accelerator `iqn_ils`；
- base budget 16；
- maximum trial limit 24；
- near-tolerance extra limit 8；
- relative tolerance `1e-3`；
- absolute tolerance `1e-4 m/s`；
- actual trials used 16。

精确 RMS 路径：

| Trial | proposal | beta / raw omega | RMS (m/s) | 判定 |
|---:|---|---:|---:|---|
| 1 | initial observation | — | `6.887119119476754e-4` | initial |
| 2 | Picard | `omega=0.5` | `1.6248462952410723e-4` | accept |
| 3 | Picard | `omega=0.5` | `2.801836999733742e-4` | reject |
| 4 | Picard backtrack | `omega=0.25` | `1.2378696148313093e-4` | accept, enters near-band |
| 5 | IQN | `beta=1` | `3.173000501838806e-4` | reject |
| 6 | same IQN direction | `beta=0.5` | `2.881314415638371e-4` | reject |
| 7 | same IQN direction | `beta=0.25` | `2.779348433225945e-4` | reject |
| 8 | same IQN direction | `beta=0.125` | `1.0911772079218964e-4` | accept, global best |
| 9 | enriched rank-3 IQN | `beta=1` | `3.1907758241847713e-4` | reject |
| 10 | same IQN direction | `beta=0.5` | `2.9232707941041514e-4` | reject |
| 11 | same IQN direction | `beta=0.25` | `2.8183247679080777e-4` | reject |
| 12 | same IQN direction | `beta=0.125` | `2.775018576907523e-4` | reject |
| 13 | global-best Picard restart | `omega=0.25` | `2.8172251488145495e-4` | reject |
| 14 | same Picard direction | `omega=0.125` | `2.7713415786739955e-4` | reject |
| 15 | same Picard direction | `omega=0.0625` | `2.7531518929750106e-4` | reject |
| 16 | same Picard direction | `omega=0.03125` | `2.7459686374005517e-4` | reject, line search exhausted |

正式 certificate 最终写的是：

- measured `true`；
- converged `false`；
- absolute residual `2.7459686374005517e-4 m/s`；
- reason `iteration_budget_exhausted`。

这个 reason 不够准确：循环只用了 16/24 次。真实原因是 `line_search_exhausted` 的无条件 break 覆盖 near-band continuation。

## 6. 当前明确根因

当前源代码的关键控制流（行号是交接生成时的工作树）：

- 约 `cases/turek_hron_fsi.py:3844–3847`：Picard 回溯耗尽后清空 pending 并置 `iqn_line_search_exhausted=True`；
- 约 `:3872–3885`：near-band continuation 对 v4 trial16 明确返回 true；
- 约 `:3890–3899`：随后因为 `iqn_line_search_exhausted` 且无 forced next，无条件 break。

因此这是状态机接线 bug，而不是容差策略本身失败。

但要特别注意：**不能只删除 break**。如果让 trial16 的 rejected state 落入普通 history append，它会被错误当成 accepted observation，污染 IQN secant history。也不能简单重放 global-best point；那不会产生新信息。

数值上，trial9–12 和 trial13–16 随 `beta→0` 都从高 residual 向 best 渐近、但始终高于 best，说明这两个方向在当前 best 处都是上升方向。仅继续缩同一 beta 或机械增加预算，不能证明会下降。

## 7. 尚未实现的推荐 v5 修复（新线程任务）

### 7.1 推荐状态转换

当且仅当以下全部成立：

- 当前 direction 已 line-search exhausted；
- best residual `<=1.25 × tolerance` 但 `> tolerance`；
- completed trials `<24`；
- 本物理步尚未执行过 near-band cold recovery；

执行一次显式 `recover_from_global_best`：

1. 从 immutable global-best velocity guess/candidate 与配对 Neumann-gradient guess/candidate 恢复 source；
2. history 重置为该 global-best pair 的深拷贝，只保留这一 pair；
3. 构造 `phase="recovery"` 的全新、未测试 proposal；
4. velocity 与 gradient 使用同一个 recovery omega，后续 backtracking 使用同一个 beta；
5. forced-next 设置后直接 `continue`，绝不落入 ordinary history append；
6. recovery 的接受/拒绝仍由下一次真实 velocity RMS 决定；
7. recovery 不得改写 normal Picard memory；其 `full_picard_relaxation` 与 configured Picard reference 应为 `None`；
8. 所有 recovery trials 仍计入 24 的硬上限；
9. 如果无法构造严格新颖的 proposal，则明确以 `line_search_exhausted` fail-closed，不得伪装为 budget exhaustion。

候选 recovery scale policy（必须先用测试证明，而不是直接硬编码到物理 case）：

- 从 recovery 专属 configured scale `0.05` 开始；
- 已测试则按 `0.5` 几何缩小：`0.025, 0.0125, 0.00625`；
- 只允许一次 cold-recovery generation，防止重复循环；
- recovery scale 可以低于 normal-memory floor，但不得回写 normal memory。

这只是推荐的通用 policy，不保证 v4 的真实 GPU 轨迹一定在 8 次内成功；必须由新门禁 artifact 判断。

### 7.2 必须先写的 RED 合同

至少包括：

1. `completed=16`、best in near-band、Picard exhausted 时返回 `schedule_best_recovery`，而不是 stop/converged。
2. best 超出 near-band、completed `>=24`、未 exhausted 或 cooldown 已尝试时不得触发。
3. formal tolerance、base budget、extra limit 原样不变。
4. recovery pending phase、source、velocity/gradient omega 与 beta 配对正确。
5. rejected exhausted trial 不进入 history；reset 后只含 copied global-best pair，且无 ndarray alias。
6. recovery proposal 不得等于任何已耗尽 proposal；若重复则明确终止。
7. recovery 接受也不更新 normal Picard memory/reference。
8. transition/helper 必须继续经 `_state_machine_call`；注入异常时验证 fluid/solid/marker/gradient 全回滚且只持久化一次失败证据。
9. 用 v4 trial1–16 replay：trial16 后必须调度 trial17，不能报告 `iteration_budget_exhausted`。
10. 真正 trial24 未收敛时必须严格 fail-closed。

建议非线性代理测试（仅表达局部非单调，不含 FSI1 step/残差硬编码）：

`F(x) = x + tol × (1.1 - 48x + 1920x²)`

其 residual 可构造出 near-band best、旧较大尺度全部回归、较小未测试 recovery scale 最终通过的路径。代理必须要求“只有真实评估后才通过”。

## 8. GPU 路径敏感性警告

v3 与 v4 即使在理论上相同的早期 Picard proposal 上，也出现明显不同的第 184 步 residual 路径；对 1–183 步 CSV 的只读比较还显示从很早阶段就有小幅分叉，随后在敏感点被放大。

这可能来自 CUDA 浮点归约/执行顺序的非确定性，也可能是隐藏状态差异。不要用单次 GPU trial 的某一个 beta/残差硬编码 policy，也不要宣称 v5 必然复现 v4 的精确数值。正确做法是：

- policy 只依赖通用状态（near-band、真实下降、novel proposal、剩余预算）；
- surrogate/contract tests 必须 deterministic；
- 真实门禁仍以 artifact 为唯一事实来源；
- 如需做可重复性检查，先设计一个短、只读且不与门禁并发的控制实验，不要静默启动完整 220 步。

## 9. 当前代码与测试文件

本轮 FSI1 修复应继续严格限制在：

- `cases/turek_hron_fsi.py`
- `tests/cases/test_turek_hron_fsi_strong_coupling_contracts.py`

交接时：

- `cases/turek_hron_fsi.py` 相对 HEAD 是大型未提交 diff（约 `+2711/-136`）；
- `tests/cases/test_turek_hron_fsi_strong_coupling_contracts.py` 是 untracked 文件；
- 整个 worktree 还有大量与本任务无关的用户修改和 artifact。

不要 `git reset --hard`、不要 checkout 覆盖、不要清理整个 worktree。先用 `git status --short` 与 scoped diff 审查，只编辑上述两个文件，除非 artifact 明确证明必须扩大范围并先说明理由。

当前 strong-coupling contract 文件已经覆盖：absolute RMS、marker invariance、candidate anchoring、paired-gradient recovery、relaxed complement、SVD/rank filtering、evaluated IQN line search、Picard memory/floor/reference、rollback、near-band bounded continuation、failure artifact persistence 等。

## 10. 新线程推荐执行顺序

1. 完整阅读本交接、v4 failure JSON、v4 CSV 和当前 scoped diff。
2. 确认没有 Python 仿真进程。
3. 只读复核 `line_search_exhausted`、near-band continuation 和 history append 的控制流。
4. 按第 7.2 节先写 RED tests，运行并保存明确失败证据。
5. 最小实现一次性 global-best cold recovery；不要顺手重构整段 solver。
6. 跑 strong-coupling 全文件，然后相关 FSI1 六文件套件、`py_compile`、diff-check。
7. 做代码、数值、测试三路只读 review；修复所有 Critical/High。
8. 明确通知后，才启动唯一一个全新 v5 220 步门禁；不要复用 v4 目录。
9. v5 失败：立即分析 artifact，不静默重启。
10. v5 严格通过：先做 220 行全量证书审计，再启动唯一正式 1600 步。

建议新门禁目录（名称可调整，但必须全新）：

`validation_runs\turek_hron_fsi\repair_20260712\fsi1_220step_mg_iqn16_velocity_linesearch_nearband_recovery_reconstructed_v5`

建议正式目录（只有 v5 通过后）：

`validation_runs\turek_hron_fsi\formal_20260712\fsi1_1600step_mg_iqn16_velocity_linesearch_nearband_recovery_reconstructed_snapshot20`

固定运行配置：

- preset `fsi1`；
- steps 220（门禁）/1600（正式）；
- `dt=0.005 s`；
- grid `(4,48,288)`；
- 100 markers；
- FV multigrid；
- predictor substeps 1；
- projection budget 4000；
- coupling accelerator IQN-ILS；
- base budget 16；
- relative tolerance `1e-3`；
- absolute RMS `1e-4 m/s`；
- 原 near-band extra 8 规则；
- 正式 flow snapshot interval 20。

## 11. 正式物理验收与两个独立误差账本

### Canonical Turek–Hron FSI1

- tip `ux = 2.27e-5 m`
- tip `uy = 8.209e-4 m`
- drag `14.295 N/m`
- lift `0.7638 N/m`

### 本地 LS-DYNA ICFD 文献（独立账本）

- tip `ux = 1.7e-5 m`
- tip `uy = 8.6e-4 m`
- drag `14.26 N/m`
- lift `0.73 ± 0.30 N/m`

这两个账本不得混合。5% 误差门必须如实计算；如果稳态、趋势、跨度或某个量不通过，就报告不通过并继续定位一般性求解器根因，不能选择性换参考。

正式分析还要检查：固定根部、流量平衡、主/后投影、耦合 residual、marker 完整性、scatter closure、字段有限性和假收敛风险。

## 12. 正式渲染要求

正式 1600 步且物理验收完成后，使用现有：

`src.refactored.validation.turek_hron_fsi.rendering`

从完整 `flow_snapshots` 渲染速度云图 GIF：

- 固定全局色标；
- 不显示白色 marker/rest 点；
- 使用物理固体形状；
- 复制可查看副本到指定 visualization 目录；
- 在任务中展示；
- 把正式比较报告写入正式运行目录。

## 13. 明确不要做

- 不续跑 v4、旧 formal、v2/v3 或任何 fail-closed 目录；
- 不把 `1.091177e-4` 判成通过；
- 不提高 `1e-4` 容差；
- 不提高 base budget 16；
- 不把 extra 8 变成无限迭代；
- 不删除 `line_search_exhausted` break 后让 rejected state 落入普通 history；
- 不把 rejected IQN/Picard trial 进入 secant；
- 不用 trial16 的 gradient candidate 做 recovery source；
- 不让 recovery 污染 normal Picard memory；
- 不为绕过 out-of-bounds 或物理守卫扩大网格/关闭守卫；
- 不把启动阶段、最后一点或力的偶然抵消称作稳态；
- 不伪造 5% 结论；
- 不启动 FSI2/FSI3。

## 14. 可直接发给新线程的首条消息

```text
请接手并继续完成 Turek–Hron FSI1。工作区：
D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored

先完整阅读：
docs/refactoring/TUREK_HRON_FSI1_THREAD_HANDOFF_2026-07-12.md

以 v4 artifact 为当前事实来源：
validation_runs/turek_hron_fsi/repair_20260712/fsi1_220step_mg_iqn16_velocity_linesearch_picardfloor_reference_reconstructed_v4

当前没有仿真进程。不要续跑旧目录。固定 dt=0.005 s、absolute velocity RMS tolerance=1e-4 m/s、base budget=16、near-band extra最多8、grid=(4,48,288)、100 markers；不放宽、不伪造通过。

先按 handoff 第7节 TDD 修复 near-band continuation 被 line_search_exhausted 提前截断的状态机缺口。只编辑 cases/turek_hron_fsi.py 和 tests/cases/test_turek_hron_fsi_strong_coupling_contracts.py，保护脏工作树其他用户修改。必须 RED→最小 GREEN→focused suites/py_compile/diff-check→三路只读 review，全部通过并明确通知后才启动唯一全新 v5 220步门禁。

如果 v5 提前失败，立即分析 artifact，绝不静默重启。只有 v5 220/220 严格通过，才启动唯一正式1600步，做 t=4–6 与6–8 s acceptance，canonical和本地LS-DYNA分账，并生成固定色标、无白色marker/rest点的速度GIF。不启动FSI2/3。
```

## 15. 交接时状态声明

- 本交接只创建此 Markdown 文档，没有继续修改求解器或测试。
- v4 失败后没有启动任何新仿真。
- 旧 `turek-hron-fsi1-v4` heartbeat 已删除。
- 当前 FSI1 仍是 **未通过**；下一线程必须从 TDD 状态机修复开始。
