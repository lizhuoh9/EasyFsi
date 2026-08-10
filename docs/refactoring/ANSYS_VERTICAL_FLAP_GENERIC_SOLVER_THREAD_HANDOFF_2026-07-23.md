# ANSYS/Fluent 官方竖直薄板：通用求解器继续验证交接（2026-07-23）

## 0. 当前硬结论

本线程仍未证明我们的通用求解器复现了 ANSYS/Fluent 官方竖直薄板案例。

当前最准确的状态是：

- `vf48c` 的旧 current-source 稳态预流、2-step gate 曾通过，但对应源码已经过期，快照不得复用；
- `vf48c` 独立 50 步在完成 3 步、准备第 4 步时暴露新的有限线段 union claim 冲突；
- 本线程已实现并聚焦验证统一 finite-segment-union 几何合同；
- 第一次全新 current-source 预流 `vf48d` 在第 1 个预流步进入 production reconstruct kernel 编译时发现一个真实作用域错误；
- 该作用域错误已经用 RED/GREEN 最小修复，但修复后尚未重新完成 production preflow；
- 因此当前只能写成“聚焦合同绿色、production preflow 待重跑”，不能写成 solver green、50/50 或 Fluent parity。

## 1. 用户锁定的任务边界

本任务只能用我们的通用求解器复现 **ANSYS/Fluent 官方竖直薄板案例**，不得转去 Turek-Hron FSI1 或其他相邻案例。

为避免继续陷入局部补丁循环，后续必须保持以下顺序：

1. 全新 current-source 稳态预流和严格快照；
2. 从该快照启动全新 2-step gate；
3. 从同一快照独立启动 50-step gate；
4. 只有 50/50、连续工件和全部物理门通过后，才运行锁定 Fluent strict comparison；
5. 若 2-step/50-step 出现不在现有 RED 覆盖内的新几何类，立即停止并做架构复审，不得追加第三个局部例外。

继续禁止：

- author target 平均；
- 放宽容差掩盖冲突；
- face-axis 外推；
- case/marker/region 硬编码；
- 用短探针、残差或脚本完成状态冒充 Fluent parity。

旧总交接仍需阅读：

`docs/refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-16.md`，重点是第 13 节。

## 2. 本线程从 production 工件得到的几何证据

### 2.1 旧独立 50 步失败

运行：

`validation_runs/solver_soaks/vf48c_fsi50_chordnormal_20260723_a`

真实状态：完成 3 步，准备第 4 步时 fail closed，claim-conflict count 为 72。

### 2.2 bounded 全量捕获

完整捕获：

`validation_runs/solver_soaks/vf48c_diag_full_afterstep3_saved_chordnormal_20260723_a_claims.json`

捕获结果：

- 68/68 bounded duplicate witnesses 完整保存；
- diagnostic path 0 为 64 条，path 2 为 4 条；
- 60 条为 adjacent-segment pair，8 条为 same-segment pair；
- 捕获证明需要统一的 face-first finite-segment-union owner 几何，而不是只修某一个 marker 或某一个 face。

## 3. 当前实现内容

主要生产文件：

`simulation_core/coupling/hibm_mpm/core.py`

实现要点：

- 两个 finite primitives 都先投影到 physical MAC face；
- 用 F64 几何距离选择唯一 strict-nearest owner；
- loser 的 endpoint support 不得否决 strict interior winner；
- 只有 winner 确实 endpoint-clamped 时才检查 endpoint support；
- endpoint-clamped winner 必须在 authoritative projection-segment topology 中满足：owner edge 恰出现一次，clamped endpoint 的全局 incident degree 恰为 1；
- field-only 调用没有 authoritative topology 时，对 endpoint clamp 必须 fail closed；
- prepare 在 active plane 统一调用该 helper 并写入 `FACE_FIRST_FINITE_SEGMENT_PAIR` mode；
- reconstruct 读取同一 mode、再次调用同一 helper，并同时要求 admission/full valid；
- helper 失败后不得回落到旧 author-local target、平均或外推路径。

相关聚焦合同：

- `tests/solvers/test_hibm_segment_pair_geometry.py`（当前为未跟踪文件，必须保留）；
- `tests/solvers/_hibm_component_face_ledger_contracts.py`；
- `tests/solvers/test_hibm.py`；
- `tools/diagnose_hibm_component_claim_conflict.py`；
- `tests/tools/test_diagnose_hibm_component_claim_conflict.py`。

## 4. 已完成的 RED/GREEN 与审查

### 4.1 有效 review RED

以下三个缺口先得到真实 RED，再由统一实现转绿：

1. active-plane component-axis pair 不得绕过统一 helper；
2. same-region internal degree-2 endpoint 不得冒充 terminal endpoint；
3. loser endpoint over-support 不得拒绝 strict-nearest interior winner。

### 4.2 最终当前源码验证

命令：

```powershell
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.solvers.test_hibm_segment_pair_geometry
```

结果：

- `Ran 6 tests in 51.505s`；
- `OK`。

命令：

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  simulation_core\coupling\hibm_mpm\core.py `
  tests\solvers\test_hibm_segment_pair_geometry.py `
  tests\solvers\_hibm_component_face_ledger_contracts.py
```

结果：exit 0。

最终只读审查结论：

- High = 0；
- Medium = 0；
- 作用域修复是语义中性的，只修 Taichi name resolution；
- 保留的静态作用域回归准确覆盖本次 production RED；
- 修复后尚无成功 production preflow，是唯一明确的残余风险。

### 4.3 不得误报为绿色的测试

旧 full-fluid ledger 集成夹具在 20 分钟内仍停留于完整 `CartesianFluidSolver` JIT，未取得 prepare/reconstruct 结果。

曾尝试一个 minimal-field production-kernel 测试：

- 单独运行 1809 秒；
- 仅显示测试开始，无 PASS/FAIL；
- 峰值约 24 GB；
- 还缺 topology-present reconstruct 正控，存在测试因果缺口。

该昂贵且未验证的测试已经整段撤除，不能在新线程中恢复为默认发现测试，也不能写成通过。

## 5. `vf48d` production preflow 的真实 RED

### 5.1 配置身份 dry-run

审计目录：

- 初次 dry-run：`validation_runs/solver_soaks/vf48d_pf_finiteunion_20260723_a_dryrun`；
- 锁定 dry-run：`validation_runs/solver_soaks/vf48d_pf_finiteunion_20260723_a_dryrun2`。

第一次 dry-run 发现 x search radius 的 JSON 浮点序列化为 `0.0012000000000000001`；第二次显式传入：

`--hibm-search-radius-xyz-m 0.0012 0.000390625 0.00046875`

第二次 dry-run 与上一份通过的 `vf48c_pf_chordnormal_20260723_a/our_solver_config.json` 逐字段比较，唯一配置差异是新的 `preflow_snapshot_output_path`。

当时 source manifest 与 `vf48c` 比较，唯一 source hash 变化是 `simulation_core/coupling/hibm_mpm/core.py`。

### 5.2 正式失败目录

正式目录：

`validation_runs/solver_soaks/vf48d_pf_finiteunion_20260723_a`

证据：

`validation_runs/solver_soaks/vf48d_pf_finiteunion_20260723_a/failure.json`

真实结果：

- `status=failed`；
- `error_type=TaichiNameError`；
- `elapsed_s=6379.144800900016`；
- 没有 `preflow_progress.json`；
- 没有 `our_solver_summary.json`；
- 没有生成 `vf48d_snap_finiteunion_20260723_a` 快照目录。

错误位于 production reconstruct kernel：

```text
Name "authors_are_component_axis_pair" is not defined
```

原因：该变量原来只在一个动态 `if reconstruction_valid != 0` 内定义，随后在另一个动态分支读取。Python 语法允许，Taichi kernel lowering 不允许该条件作用域逃逸。

### 5.3 最小修复

在 reconstruct kernel 的外层局部初始化区加入：

```python
authors_are_component_axis_pair = 0
```

后续有效几何路径仍用原表达式完整覆盖该值；没有改变任何 admission、topology、normal、probe、target 或 fail-closed 条件。

对应静态合同先得到 `0 != 1` 的 RED，再转绿。

当前修复后 `core.py` SHA256：

`59a61bd463331cb45e195516dbc2240fd77fbb558c730b28127cdd81da4b40ce`

注意：`vf48d` manifest 属于作用域修复前源码。下一次必须重新 dry-run 并生成新的 manifest，绝不能在 `vf48d` 目录续跑。

## 6. 当前工作树冻结状态

冻结检查：

- Python/PythonW process count = 0；
- 没有仿真或测试进程需要接管；
- 没有 commit、push、reset 或 clean；
- 相关 `git diff --check` 通过，仅有 LF/CRLF 警告；
- 工作树仍包含大量本任务之前和其他线程留下的用户修改，必须全部保留。

本线程直接新增/触达的关键状态：

- `simulation_core/coupling/hibm_mpm/core.py`：统一 finite-segment-union + terminal topology + 最小作用域修复；
- `tests/solvers/test_hibm_segment_pair_geometry.py`：未跟踪，6-test 聚焦合同；
- `tests/solvers/_hibm_component_face_ledger_contracts.py`：captured production fixtures/合同；
- `tools/diagnose_hibm_component_claim_conflict.py` 与对应测试：bounded conflict capture；
- 本交接文档；
- `vf48c_*` 捕获/验证工件；
- `vf48d_*_dryrun*` 配置审计工件；
- `vf48d_pf_finiteunion_20260723_a/failure.json` production RED。

不要因为 `git status --short` 很长而执行 `git reset --hard`、`git clean`、批量删除或覆盖无关文件。

## 7. 新线程唯一允许的第一步

### 7.1 先复核冻结状态

```powershell
Get-Process python,pythonw -ErrorAction SilentlyContinue
git status --short
```

重读：

1. 本文；
2. `docs/refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-16.md` 第 13 节；
3. `validation_runs/solver_soaks/vf48d_pf_finiteunion_20260723_a/failure.json`；
4. 当前 `core.py` 中 outer flag 初始化和统一 helper 的 prepare/reconstruct 调用点。

### 7.2 使用全新 `vf48e` 身份做 dry-run

建议目录（启动前必须确认都不存在）：

- dry-run：`validation_runs/solver_soaks/vf48e_pf_finiteunion_scopefix_20260723_a_dryrun`；
- 正式预流：`validation_runs/solver_soaks/vf48e_pf_finiteunion_scopefix_20260723_a`；
- 快照：`validation_runs/solver_soaks/vf48e_snap_finiteunion_scopefix_20260723_a/preflow_state`。

精确 dry-run 命令：

```powershell
& 'D:\working\taichi\env\python.exe' `
  'validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\scripts\run_our_solver_vertical_flap.py' `
  --output-dir 'validation_runs\solver_soaks\vf48e_pf_finiteunion_scopefix_20260723_a_dryrun' `
  --run-label 'vf48e_pf_finiteunion_scopefix_20260723_a' `
  --steps 0 `
  --preflow-steps 200 `
  --preflow-convergence-mode windowed_stationary `
  --preflow-stationary-min-steps 20 `
  --preflow-stationary-window-steps 10 `
  --preflow-stationary-consecutive-windows 3 `
  --preflow-stationary-tolerance 0.01 `
  --preflow-stationary-divergence-tolerance 0.05 `
  --preflow-stationary-no-slip-tolerance-fraction 0.05 `
  --preflow-snapshot-out 'validation_runs/solver_soaks/vf48e_snap_finiteunion_scopefix_20260723_a/preflow_state' `
  --grid-nodes 4 256 320 `
  --solid-particle-counts 1 256 20 `
  --marker-count 64 `
  --flow-projection-iterations 1080 `
  --flow-post-dirichlet-consistency-projections 1 `
  --flow-cg-preconditioner fv_multigrid `
  --flow-pressure-solve-failure-policy raise `
  --solid-substeps 1600 `
  --flow-predictor-substeps 2 `
  --hibm-search-radius-xyz-m 0.0012 0.000390625 0.00046875 `
  --dry-run
```

必须把新 config 与 `vf48c_pf_chordnormal_20260723_a/our_solver_config.json` 逐字段比较。除 snapshot path 外不得有配置漂移；新 manifest 必须明确记录当前 `core.py` hash。

### 7.3 全新 production preflow

dry-run 身份通过后，使用完全相同参数：

- 把 `--output-dir` 改为正式 `vf48e_pf_finiteunion_scopefix_20260723_a`；
- 删除 `--dry-run`；
- 不复用或续跑 `vf48d`；
- 运行期间保持只有一个 Python/Taichi 进程，不改代码。

预流完成后必须验证：

- process exit 0；
- summary completed；
- 连续有限的 preflow history/progress；
- pressure CG、no-slip、SST、物理门和 HIBM conflict 全部通过；
- snapshot manifest、generation NPZ、SHA256、config/geometry/source identity 全部通过当前 loader。

### 7.4 后续门禁

只有 `vf48e` preflow 和 snapshot 完整通过后：

1. 全新 2-step gate，从该快照启动，保留 `--preflow-steps 200`；
2. 全新 independent 50-step gate，从同一快照独立启动，并启用 `--save-step-fields`；
3. 若 50/50、50 个连续 step fields/history、全部物理门和身份审查通过，才运行锁定 Fluent strict comparison；
4. strict comparison 完成也只能先写 `diagnostic_complete`，压力、速度和位移合同全部达标前保持 `parity_claimed=false`。

锁定 Fluent 输入仍是：

`validation_runs/ansys_vertical_flap_fsi/official_fluent_fine_fsi_valid_2026-07-10/runs/fresh50_20260713_104843/postprocess_compare31_strict_pressure_20260719_142808_r2`

锁定比较入口仍是：

`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/scripts/postprocess_our_solver_vs_native_fluent.py`

## 8. 新线程开场一句话

> 继续 `docs/refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-23.md`：不要去 FSI1，不要复用 vf48c/vf48d 快照或目录；先复核无 Python 进程与 dirty worktree，再按第 7.2 节创建 vf48e dry-run，验证当前 scope-fix source identity 后启动全新 current-source preflow。

## 9. 2026-07-24 暂停冻结增量

本节覆盖第 7、8 节中已经被执行过的 `vf48e` 起跑说明。下一线程必须从本节继续，不得再次从旧的 `vf48e` 指令起跑。

### 9.1 用户要求暂停后的冻结状态

2026-07-24 用户明确要求“先暂停，写交接文档”。已执行：

- 强制停止仍在运行的 current-source preflow 进程 PID `59556`；
- 停止后再次检查，`python/pythonw` process count = `0`；
- 删除旧线程的 heartbeat automation `continue-vf48k-preflow`；
- 没有启动替代进程，没有重启 preflow、2-step 或 50-step；
- 没有 commit、stage、push、reset、clean，也没有清理任何旧失败目录；
- 保留了全部既有 dirty worktree 和未跟踪测试/验证工件。

暂停时的 `vf48k` 生产目录：

`validation_runs/solver_soaks/vf48k_pf_interiority_20260724_a`

生产目录本身只包含：

- `our_solver_config.json`；
- `run_manifest.json`；

同级外部日志为：

- `vf48k_pf_interiority_20260724_a.stdout.log`，`0` bytes；
- `vf48k_pf_interiority_20260724_a.stderr.log`，`274` bytes，仅有 Taichi `ticache.lock` warning。

它没有：

- `progress.json`；
- `failure.json`；
- `our_solver_summary.json`；
- `our_solver_history.csv`；
- 任何 preflow snapshot。

预定快照路径

`validation_runs/solver_soaks/vf48k_snap_interiority_20260724_a/preflow_state`

不存在。故 `vf48k` 是用户主动中止的冻结工件，不是 solver RED，也不是可续跑目录；下一线程禁止复用其生产目录或虚构其完成状态。

### 9.2 `vf48e → vf48h` 的 production 证据链

以下目录全部冻结，不复用、不覆盖：

1. `vf48e_pf_finiteunion_scopefix_20260723_a`
   - `elapsed_s = 405.4820415999857`；
   - preflow 早期 fail closed；
   - `target_conflict_count = 140`；
   - 首个 component-face 为 `(0, 1, 149)`；
   - 暴露 exact terminal endpoint 与 generic terminal topology 的未统一路径。
2. `vf48f_pf_finiteunion_exactendpoint_20260723_a`
   - `elapsed_s = 7487.277203000034`；
   - `target_conflict_count = 132`；
   - 首个 component-face 为 `(0, 1, 161)`；
   - 暴露 node-search provenance 与 segment-support provenance 不一致。
3. `vf48g_pf_searchsupport_20260723_a`
   - `elapsed_s = 7458.978681899956`；
   - `target_conflict_count = 96`；
   - 首个 component-face 为 `(0, 5, 149)`；
   - 两条相邻内部段在共享 marker 处出现 F32 one-ULP co-state 差异，暴露 internal degree-2 C0 canonical state 缺口。
4. `vf48h_pf_internalc0_20260723_c`
   - `status = completed`；
   - `elapsed_s = 32046.167627699964`；
   - 在 requested `200` 步中完成 `92` 步后达到 `windowed_stationary`；
   - `local_velocity_peak_mps = 40.293418884277344`；
   - 最终 preflow 记录的 `no_slip_projected_residual_after_projection_mps = 5.74328282709757e-07`；
   - snapshot 写入：
     `validation_runs/solver_soaks/vf48h_snap_internalc0_20260723_c/preflow_state.json`
     和
     `preflow_state.52217e73131b4b4f829fad2347f98857.npz`。

`vf48h` 是当时源码身份下的绿色 preflow/snapshot，但不是当前源码可加载的快照。其 manifest 中 `core.py` SHA256 为：

`f8526a84e5781da6bd6bf89ddbd348a49ff67b9ba4646c97bd2d9c9d82fdc5a0`

### 9.3 `vf48i` 2-step production RED 与通用根因

冻结目录：

`validation_runs/solver_soaks/vf48i_gate2step_internalc0_20260724_a`

结果：

- 从 `vf48h` snapshot 严格加载成功；
- 在第一个 FSI step 完成前 fail closed；
- `elapsed_s = 378.7405153000145`；
- `status = failed`；
- `step_completed = 0`，`time_s = 0.0`；
- `target_conflict_count = 8`；
- 首个 component-face 为 `(0, 93, 149)`，axis `y`，region `202`；
- 相邻 segments 为 `(109,110)` 和 `(110,111)`，共享 marker `110`。

精确 witness 在存储 F32 几何提升到 F64 后为：

- 第一段 raw parameter 约 `1.001289`，已经越过共享 endpoint；
- 第二段 raw parameter 约 `0.000930`，是严格 interior；
- distance-squared gap 约 `2.116e-14`，仍小于旧 broad tie band `4.6566e-14`；
- 旧实现因此把可以证明的唯一第二段 owner 错判为 ambiguity。

架构/TDD 复核结论：这不是新的 corner、seam、cap、T-junction 或 region geometry class，而是通用 finite-segment-union / Voronoi interiority invariant 没有完整实现。

实现修复位于：

`simulation_core/coupling/hibm_mpm/core.py`

当前文件 SHA256：

`acfca19002dd850e3d887d7ae58c5c09fe6aaa54d285e38ee240080ea5e2dadd`

修复合同：

- 只在原有 broad distance-tie band 内应用；
- authoritative projection topology 必须证明两条相邻段各恰好出现一次；
- 共享 vertex 的全局 incident segment degree 必须恰为 `2`；
- 一条候选必须在 `2e-6` 参数余量外严格 interior；
- 另一条必须在同一余量外明确越过共享 endpoint；
- 满足这些条件时选择严格 interior primitive 的 marker-shape interpolation；
- 不平均 author targets，不固定选择某一 author/segment side；
- 不扩大 tolerance，不按 case、marker、region 或 face axis 硬编码；
- 其他 tie 仍只能走已有 internal degree-2 C0 shared-state proof，否则 fail closed。

只读 code review 结论：High/Medium finding = `0`；所有权定理、topology guard、author-order/mirror symmetry 和 target provenance 均通过。

### 9.4 TDD/验证状态：哪些已经 GREEN，哪些没有

未跟踪的 focused contract 文件：

`tests/solvers/test_hibm_segment_pair_geometry.py`

已保存：

- exact `vf48i` witness；
- author-order reversal；
- mirrored chain / owner side reversal；
- missing topology / degree-3 bypass fail-closed；
- 原有 internal C0、near-C0、fail-closed 和 malformed/NaN 合同。

本线程实际得到的证据：

- RED：修复前 primitive valid 为 `(1,1)`，pair admission 为 `(0,0)`；两个新增断言按预期失败，`179.939 s`；
- GREEN：exact witness + mirror/order 两个测试，`Ran 2 tests in 234.899s — OK`；
- GREEN：已有 C0 / near-C0 / fail-closed / NaN 四测试，`Ran 4 tests in 675.522s — OK`；
- GREEN：独立的 missing-topology / degree-3 bypass 合同在组合运行中输出 `ok`；
- 因此相关 focused GREEN 共 `7` 个测试。

新增的完整 ledger 集成合同位于：

`tests/solvers/_hibm_component_face_ledger_contracts.py`

测试名：

`test_vf48i_strict_interior_owner_survives_ledger_reconstruction`

该测试语法有效，但本线程没有得到 PASS/FAIL 终态。三次尝试分别在约 `20 min`、`40 min`、`40 min` 外部上限被终止；`TI_OFFLINE_CACHE=0` 仍出现同一 Taichi `ticache.lock` warning。必须把它记录为“本地未完成/基础设施受阻”，绝不能写成 GREEN，也不能把超时写成物理 solver failure。

### 9.5 `vf48j`：配置审计与严格快照身份拒绝

以下目录全部冻结：

- 错误 dry-run：
  `validation_runs/solver_soaks/vf48j_gate2step_interiority_20260724_a_dryrun`
- 正确 dry-run：
  `validation_runs/solver_soaks/vf48j_gate2step_interiority_20260724_b_dryrun`
- production strict-load：
  `validation_runs/solver_soaks/vf48j_gate2step_interiority_20260724_b`

`vf48j_a_dryrun` 误传 legacy scalar `--hibm-interior-probe-distance-m`，导致 locked anisotropic probe vector 被清空；与 `vf48i` config 逐字段比较有 `7` 个 flattened differences。禁止复用。

`vf48j_b_dryrun` 与 `vf48i` locked config 的逐字段差异数为 `0`，manifest 中 `core.py` SHA256 与当前磁盘一致。

`vf48j_b` production 在 `54.99220350000542 s` 后由 strict snapshot loader 正确拒绝，尚未推进任何物理 step：

- `status = failed`；
- `step_completed = 0`；
- `time_s = 0.0`；
- snapshot stored aggregate source identity：
  `ce34dd4f40918167fb6f880422a691e398cb690f0601d127f2844b98f841c1fe`；
- current expected aggregate source identity：
  `16066ddd1c242807c024a9ba51bf3feb1f598013557b2e1f814d08b6288d350b`。

这是严格 provenance 守门的预期正确行为，不是 solver physics RED。它证明 `vf48h` snapshot 不得在 current source 下继续使用。

### 9.6 `vf48k` dry-run 身份审计

正确 dry-run：

`validation_runs/solver_soaks/vf48k_pf_interiority_20260724_a_dryrun`

相对最后绿色的 `vf48h_pf_internalc0_20260723_c_dryrun`，逐字段比较只有 `1` 个差异：

`preflow_snapshot_output_path`

production manifest 中 `core.py` SHA256 与当前磁盘完全一致：

`acfca19002dd850e3d887d7ae58c5c09fe6aaa54d285e38ee240080ea5e2dadd`

但 production 已按用户要求停止，未产生可审计终态或 snapshot。`vf48k` dry-run 可作为配置参考，`vf48k` production 不可续跑。

## 10. 下一线程的唯一续跑顺序

### 10.1 先复核冻结现场

```powershell
Get-Process python,pythonw -ErrorAction SilentlyContinue
git status --short
```

要求：

- Python/PythonW count 必须为 `0`；
- 保留全部 dirty/untracked 状态；
- 不清理 `vf48e` 至 `vf48k`；
- 确认拟用的新 `vf48l_*` 路径都不存在；
- 重读本文第 9、10 节和当前 `core.py`/focused tests；
- 不先跑 ledger 长测试，不并发启动多个 Python/Taichi 进程。

### 10.2 使用全新 `vf48l` 身份重新生成 current-source preflow

建议路径（启动前必须再次确认不存在）：

- dry-run：
  `validation_runs/solver_soaks/vf48l_pf_interiority_20260724_a_dryrun`
- production：
  `validation_runs/solver_soaks/vf48l_pf_interiority_20260724_a`
- snapshot：
  `validation_runs/solver_soaks/vf48l_snap_interiority_20260724_a/preflow_state`

精确 dry-run 命令：

```powershell
& 'D:\working\taichi\env\python.exe' `
  'validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\scripts\run_our_solver_vertical_flap.py' `
  --output-dir 'validation_runs\solver_soaks\vf48l_pf_interiority_20260724_a_dryrun' `
  --run-label 'vf48l_pf_interiority_20260724_a' `
  --steps 0 `
  --preflow-steps 200 `
  --preflow-convergence-mode windowed_stationary `
  --preflow-stationary-min-steps 20 `
  --preflow-stationary-window-steps 10 `
  --preflow-stationary-consecutive-windows 3 `
  --preflow-stationary-tolerance 0.01 `
  --preflow-stationary-divergence-tolerance 0.05 `
  --preflow-stationary-no-slip-tolerance-fraction 0.05 `
  --preflow-snapshot-out 'validation_runs/solver_soaks/vf48l_snap_interiority_20260724_a/preflow_state' `
  --grid-nodes 4 256 320 `
  --solid-particle-counts 1 256 20 `
  --marker-count 64 `
  --flow-projection-iterations 1080 `
  --flow-post-dirichlet-consistency-projections 1 `
  --flow-cg-preconditioner fv_multigrid `
  --flow-pressure-solve-failure-policy raise `
  --solid-substeps 1600 `
  --flow-predictor-substeps 2 `
  --hibm-search-radius-m 0.0017 `
  --hibm-search-radius-xyz-m 0.0012 0.000390625 0.00046875 `
  --young-modulus-pa 1000000 `
  --pressure-pair-provider-mode runtime_anchored_cell_pair `
  --span-reduction mean `
  --streamwise-velocity-sign -1.0 `
  --dry-run
```

关键禁止项：

- 不要传 `--hibm-interior-probe-distance-m`；
- 不要把 anisotropic probe/search vector 清空；
- 不要复用 `vf48h` snapshot；
- 不要把 run-label、output-dir 或 snapshot path 指回失败/中止目录。

dry-run 后，把 config 与

`validation_runs/solver_soaks/vf48k_pf_interiority_20260724_a_dryrun/our_solver_config.json`

逐字段比较。除新的 `preflow_snapshot_output_path` 外差异数必须为 `0`；manifest 中当前 `core.py` hash 必须为 `acfca190...e2dadd`。

确认 dry-run 绿色后，以同一 CLI：

- 把 output-dir 改为 `vf48l_pf_interiority_20260724_a`；
- 删除 `--dry-run`；
- 保持单一 Python/Taichi 进程；
- 运行期间不改源码或 config。

### 10.3 preflow 完成后的审计

只有 production 自然 exit `0` 后，才审计：

- summary `status=completed`；
- windowed-stationary stop reason、完成步数和连续窗口；
- 连续且全 finite 的 history；
- pressure CG convergence / breakdown count；
- divergence、no-slip、marker-target closure；
- SST nonfinite/cap/rejected-trial monitors；
- HIBM claim/target/region conflicts；
- 物理 pressure/velocity/outlet monitors；
- manifest 中每个 source hash；
- snapshot JSON/NPZ generation path 和 SHA256；
- config、geometry、source aggregate identity。

随后必须用当前 loader 对新 snapshot 做独立 strict-load 审计。strict-load 未通过就冻结工件，不进入 2-step。

### 10.4 2-step 与 independent 50-step

只有新 current-source snapshot 完整绿色后：

1. 创建全新的 2-step dry-run 和 production identity；
2. 从该新 snapshot 启动，保留 locked config、`--preflow-steps 200` 和单 Python 规则；
3. 审计 `2/2` history、step fields、身份和全部物理门；
4. 只有 2-step 全绿，才从同一个 snapshot 创建全新、独立的 50-step run；
5. 50-step 必须启用 `--save-step-fields`，不得从 2-step 目录续跑；
6. 只有 `50/50`、50 个连续 history/field frame 和全部物理门绿色，才进入锁定 Fluent strict comparison。

如果下一次失败暴露真正新的 geometry class，立即冻结 failure/config/manifest/log/diagnostic 工件并停止；先做 architecture + TDD review，不得继续加局部例外或自动重跑。

## 11. 下一线程开场一句话

> 继续 `docs/refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-23.md` 第 9、10 节：现场已暂停且 vf48k 未完成，禁止复用 vf48h/vf48j/vf48k 的 snapshot 或 production 目录；先确认无 Python 和 dirty worktree，再用全新 vf48l dry-run 复制 vf48k locked config，只改变新 snapshot path，随后重新生成 current-source preflow。

## 12. 2026-07-24 初始化/预流性能复核与继续门

本节覆盖第 10.2 节中“立即按旧命令启动 vf48l”的要求。正式 vf48l 仍未启动。

### 12.1 第 10.2 节的 predictor 身份已过期

第 10.2 节旧命令传入：

`--flow-predictor-substeps 2`

但当前 selected formulation 与 final strict identity 都锁定：

`flow_predictor_substeps = 1`

位置：

- `cases/ansys_vertical_flap_fsi.py`
- `src/refactored/validation/ansys_vertical_flap_fsi/native_fine_final_contracts.py`

因此旧 vf48k/vf48h 配置不能再作为“只改 snapshot path”的最终身份。后续 current-source preflow 必须使用 `1`；否则每个外层步会重复两套 predictor/HIBM/pressure-projection 工作，并生成不符合 final strict identity 的 snapshot。

### 12.2 ANSYS/Fluent 的 initialization 不是稳态预流

当前通用 runner 已经在启动时：

- 把全流场速度初始化为入口速度 `-10 m/s`；
- 把障碍内部速度设为零；
- 把压力设为零；
- 按入口湍流强度和黏度比初始化 SST `k/omega/mu_t`。

所以长时间等待不是“忘了给初始速度”。

锁定的 Fluent 工件也不是初始化后直接进入 FSI。其流程为：

1. hybrid initialization；
2. 继续执行 100 次 steady pseudo-transient SST iterations；
3. 再进入后续 FSI/比较。

均匀速度只是初猜，不能替代满足 no-slip、压力出口、不可压投影、SST 湍流和 HIBM 边界的一致稳态场。`vf48h` 的速度/压力窗口较早稳定，但 SST 联合窗口直到第 92 步才全部通过，因此不得跳过 SST 门或放宽 `0.01` 来制造快照。

### 12.3 本轮通用修复

已完成：

- `simulation_core/diagnostics/runtime.py`
  - 支持显式 Taichi offline-cache 开关和 cache path；
  - 用锁保护首次 `ti.init`；
  - 已初始化进程若请求冲突的 arch/fp/cache 身份会 fail closed。
- `validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/scripts/run_our_solver_vertical_flap.py`
  - 不再强制关闭 offline cache；
  - 默认使用项目隔离、可复用的 cache 目录；
  - manifest 明确把 cache 状态标为 `requested_before_taichi_init`；
  - 即使 `--steps 0` 也持续原子写入 initialization/preflow progress；
  - setup/config/hash/manifest 的早期失败也会生成 failure/progress；
  - failure 工件写入失败不得覆盖原始 solver 异常；
  - FSI step progress 合并已有字段，不再抹掉 cache/init/preflow 元数据；
  - `--profile-wall-time` 才启用同步 GPU 计时，正常 production 不增加这些 barrier。
- `benchmarks/official/solid_mpm_fsi_runner.py`
  - 增加 initialization phase、preflow step、SST、momentum predictor 计时；
  - 同步计时在异常路径也会关闭同步，同时保留主异常；
  - telemetry 文件写入时间不再计入被测 phase；
  - `preflow_flow_advance_wall_time_s` 在启用 profile 时有明确 pre/post sync。
- `.gitignore`
  - 忽略 `validation_runs/.taichi_cache/`，避免 cache 二进制污染 dirty worktree。

聚焦回归：

`37 passed, 10 subtests passed`

完整 `tests/cases/test_ansys_vertical_flap_fsi.py` 仍会在一个与本轮无关的历史合同处失败：缺少已被忽略的 `_codex_validation/official_ansys_fluent_vertical_flap_full_domain_solve_20260622/run_full_domain_two_flap_true_sharp_fsi.py`。不得把这个缺文件失败归到本轮 cache/progress 修复。

### 12.4 `vf48m` current-source 1-step 冷启动性能 RED

dry-run：

`validation_runs/solver_soaks/vf48m_init_profile_1step_20260724_a_dryrun`

实际 1-step profile：

`validation_runs/solver_soaks/vf48m_init_profile_1step_20260724_a`

配置要点：

- `steps = 0`
- `preflow_steps = 1`
- `flow_predictor_substeps = 1`
- grid `4 x 256 x 320`
- 单一 Python/Taichi 进程

现场观测：

- ordinary initialization 在 `45.5195863 s` 内完成并进入 `preflow_step=1`；
- 之后至少运行 `8775.8 s`，仍为 `preflow_steps_completed=0`；
- Python CPU time 基本与 wall time 1:1；
- GPU utilization 约 `2-3%`，显存约 `1.4/6.1 GB`；
- 编译阶段 private memory 峰值约 `42.7 GB`；
- cache lock warning 为 `0`；
- 因首步超过 2.4 小时仍未返回，手动终止该有界 profile；
- 终止后 Python/PythonW count 为 `0`；
- 因进程未正常退出，offline cache 文件数仍为 `0`。

冻结工件：

- `progress.json`
- `profile_termination.json`
- `run_manifest.json`
- `our_solver_config.json`

分类必须写成：

`cold_jit_performance_red`

不得写成物理 solver failure，也不得写成 preflow convergence failure。真正的稳态窗口根本还没有开始累计。该证据说明过长“初始化”的主体是巨型 Taichi/LLVM kernel 冷 JIT，而不是流场赋初速度，也不是 GPU 数组填充。

### 12.5 GPU 并行结论

当前求解器已经在单 GPU 上执行 Taichi kernel。不能通过同时启动多个 Python/Taichi 进程来“并行初始化”：

- CPU 侧 AST/LLVM JIT 不会被多 GPU 加速；
- 多进程会争用 cache lock、GPU 显存和调度；
- predictor、SST、pressure projection 在时间上有顺序依赖；
- 真正多 GPU 需要域分解、压力全局归约和新的数值合同，不是启动参数。

可把一次性的 NumPy uniform fill 改成 GPU kernel，但本次证据表明它最多对应 initialization 的几十秒，不解释 2.4 小时首步冷 JIT。

### 12.6 下一执行顺序

1. 正式 vf48l 继续冻结。
2. 先对 `simulation_core/coupling/hibm_mpm/core.py` 的巨型 topology/finite-segment-union Taichi 路径做通用拆分或按 geometry revision 预计算，减少 prepare/reconstruct 的重复内联与冷 JIT 规模。
3. 必须先写 exact witness、mirror/order、fail-closed 与 ledger reconstruction 合同，保证 owner/topology 语义不漂移。
4. 修复后用全新目录重新做 `1-step + --profile-wall-time` 冷启动；必须自然退出并产生 cache 文件。
5. 随后以相同 source/config/cache 做第二个全新目录的 warm 1-step A/B。
6. 只有 cold/warm 工件完整、首步可接受且 final strict identity 使用 `flow_predictor_substeps=1`，才生成新的 vf48l dry-run/production identity。
7. 之后仍按原顺序执行 windowed-stationary preflow、strict snapshot load、2-step、独立 50-step、Fluent strict comparison。

本轮没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作原样保留。

### 12.7 巨型 JIT 的最小拆分设计（设计阶段）

只读架构审计定位到：

- `_canonical_component_face_finite_segment_union_owner_geometry`
  - 位于 `simulation_core/coupling/hibm_mpm/core.py`；
  - 约 1193 行 `@ti.func`；
  - 被同时内联进约 1836 行 prepare kernel 与约 1050 行 reconstruct kernel；
  - prepare 内还有 `ti.static(range(3))` axis specialization。
- prepare 实际只消费 `admission_valid`；
- reconstruct 才消费完整九项 owner geometry 结果。

这会把同一套大型 owner/topology 证明重复放大到两个复杂 kernel IR，是当前冷 JIT 膨胀的最强结构证据。

建议的第一阶段最小设计：

1. 保持现有 1193 行 owner helper 语义完全不动。
2. 在 presample 后、prepare 前新增一个 runtime-axis precompute kernel。
3. 对每个 `(target, component_axis)` 只计算一次 fixed direct-author pair，并把九项结果写到 transaction-local device scratch：
   - `admission_valid`
   - `full_valid`
   - `endpoint_clamped`
   - `boundary_point`
   - `normal`
   - `nominal_probe`
   - `boundary_target`
   - `clamp_support_ratio`
   - `geometry_tolerance`
4. prepare 只读 scratch 的 `admission_valid`。
5. reconstruct 读同一 scratch 的完整结果。
6. author kind/linear key 不完全匹配 fixed direct pair 时继续 fail closed。
7. 第一版每次 ledger assembly 都重算 scratch，不做跨步 host cache，避免 stale geometry/obstacle 语义。

在 `4 x 256 x 320 x 3` lanes 上，scratch 约增加 59 MB，远小于本次冷 JIT 约 42.7 GB private-memory 峰值。

必须先建立：

- AST 结构 RED：giant helper 只能由新 precompute kernel 调用，prepare/reconstruct 不再直接引用；
- 九项结果 parity：strict-nearest adjacent、same-segment endpoint、terminal clamp、C0 shared vertex、author reversal；
- 现有 owner/topology/fail-closed ledger contracts 原样绿色；
- fresh-cache 子进程的 cold `canonical_ledger_build` wall/RSS 对照。

设计阶段不应直接修改 `core.py`，原因是该文件已有约 `+2790/-86` 未提交改动，且 owner、endpoint、C0 与 conflict witness 高度耦合；必须先冻结结构/parity RED，再做上述最小 device-scratch 拆分。第 12.8 节记录随后完成的受控 TDD 实现，并覆盖本段“尚未实现”的状态。

### 12.8 owner-geometry cold-JIT 拆分实现与当前验证门

已按第 12.7 节设计完成第一阶段最小实现：

- `simulation_core/coupling/hibm_mpm/core.py`
  - 新增 transaction-local direct-pair owner scratch；
  - 保存九项 owner geometry 结果以及 exact author linear keys/kinds；
  - 新增 runtime-axis precompute kernel；
  - 调用顺序为 `clear -> presample -> precompute -> prepare -> reconstruct`；
  - fixed pair 严格绑定为 `minus-face direct author -> target direct author`；
  - prepare/reconstruct 仅在 linear keys 和 kinds 完全匹配时读取 scratch，否则 fail closed；
  - 每次 precompute 先把全部 13 个字段清成 neutral，第一版不做跨代 host cache；
  - 1193 行 owner helper 的生产引用已从 prepare/reconstruct 移除，当前仅剩 helper definition 与 precompute kernel 中一次调用。
- `tests/solvers/test_hibm_segment_pair_geometry.py`
  - 新增 helper 只允许由 precompute 调用的 AST 结构合同；
  - 旧的“reconstruct 必须直接调用 helper”合同已先得到预期 RED，再改为 scratch/key/kind/fail-closed 合同；
  - 复用现有 vf48g direct-probe fixture，对九项结果逐项比较；
  - 有效代验证 exact keys `(2, 6)`、kinds `(0, 0)`；
  - 下一代关闭一个 direct author 后，再运行同一 precompute，验证 13 个 scratch 字段全部回到 neutral：
    - 3 个 i32 为 `0`；
    - 3 个 vec3 为零向量；
    - 3 个 float 为 `0`；
    - 2 个 author keys 与 2 个 author kinds 为 `-1`。

当前 source SHA256：

- `core.py`: `D47920B5EFD156A7755F1B1A251234CF098D8B45DCAA6FE21F1F2410367C3DDC`
- `test_hibm_segment_pair_geometry.py`: `7756579E5327188305BED9967FCB0814D46C8A97CBB13E73ECA382F3321479A6`

受控验证结果：

- 新旧两个结构合同：`2 passed`，`1.58 s`；
- vf48i strict-interior topology 窄测：`1 passed`，`87.38 s`；
- vf48g 九项 parity + exact binding + valid-to-invalid neutralization：`1 passed`，`93.23 s`；
- `git diff --check`：绿色；
- 整个 `test_hibm_segment_pair_geometry.py` 曾运行到 10 分钟上限并被停止，输出为 `..F.........`；其中已观测的 `F` 是要求 reconstruct 直接调用 helper 的过期结构合同，该合同随后已按 TDD 更新并单独绿色，但整文件没有重新宣称全绿。

复审结论：

- 未发现 Critical/High 代码问题；
- 新增 13 个 dense fields 约为 `84 B/face`，目标网格 `4 x 256 x 320` 增加约 `82.6 MB`（`78.75 MiB`）显存，并增加每代一次全域清写；
- 先前 profile 约占 `1.4/6.1 GB` 显存，因此这不是当前首要阻塞，但 full-grid profile 必须继续记录显存峰值；
- 仍缺一个诚实的、已有几何合同支撑的 micro-grid `precompute -> prepare -> reconstruct -> public ledger` 集成 fixture。

尝试补 public-ledger 集成 RED 时发现：现有 vf48g direct-probe 的 accepted probe 位于其 local test grid 之外。手工把它塞进 4³ reconstruct/commit 会引入未经证明的 author admission 和几何假设，形成伪绿。因此该无效测试未保留，不能用它给 full-grid profile 放行。

当前执行门：

1. 正式 vf48l 和新的 full-grid cold/warm profile 继续冻结。
2. 先从已有可发布的 component-face ledger fixture 中抽出一个最小 direct-pair 集成入口；不得手工制造只为通过测试的 geometry。
3. 该入口必须贯穿 precompute、prepare、reconstruct 与 public ledger，并验证 exact author binding、face-first mode、published owner payload 和 mismatch fail-closed。
4. 集成窄测绿色后，先用全新目录做 current-source 1-step cold profile，再用同 source/config/cache 做 warm A/B。
5. 只有冷/暖 profile 自然退出、cache 正常持久化、首步时间/RSS/显存可接受，才恢复 vf48l current-source preflow。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作原样保留。

### 12.9 `vf48p` owner-precompute full-grid 诊断结果

最终复审确认：

- valid-to-invalid 测试确实在同一个 boundary/scratch 实例上运行；
- 它没有重建对象或直接清 scratch；
- 第 12.8 节的 Medium stale-state 缺口已解除；
- public-ledger micro-grid 缺口降为 Low，可把 full-grid 当作首次真实集成诊断，但不能据此认可 production。

因此执行了一个有界 full-grid 诊断。dry-run：

`validation_runs/solver_soaks/vf48p_ownerprecompute_cold_20260724_a_dryrun`

dry-run 与 `vf48m` solver config 逐字段比较：

- changed fields: `0`
- missing fields: `0`
- `flow_predictor_substeps = 1`
- `profile_wall_time = true`
- current `core.py` SHA256 为 `d47920b5...67c3ddc`
- 使用新的隔离 cache：
  `validation_runs/.taichi_cache/vf48p_ownerprecompute_cuda_f32`

第一次后台启动目标：

`validation_runs/solver_soaks/vf48p_ownerprecompute_cold_20260724_a`

PowerShell `Start-Process` 因继承环境同时包含大小写重复的 `Path/PATH` 键而在创建 Python 前失败。该目录没有运行 solver，已写：

`launch_failure.json`

实际诊断目录：

`validation_runs/solver_soaks/vf48p_ownerprecompute_cold_20260724_b`

运行身份：

- `steps = 0`
- `preflow_steps = 1`
- `flow_predictor_substeps = 1`
- grid `4 x 256 x 320`
- 单一 Python/Taichi 进程
- `--profile-wall-time`
- 全新隔离 cache，初始文件数 `0`

现场结果：

- ordinary initialization 自然完成：`43.9297664 s`
  - fluid build: `16.9876914 s`
  - flow-field fill: `12.6063749 s`
  - marker build: `3.7948790 s`
  - solid build: `1.5486259 s`
  - interface initialization: `8.9921952 s`
- 之后进入 `preflow_step = 1`；
- 最后一次冻结观测总 wall time 下界：`947.2 s`；
- 对应首个 preflow step 等待下界：`903.2702336 s`；
- `preflow_steps_completed = 0`；
- Python CPU time 仍基本跟随 wall time；
- GPU utilization 最后观测约 `2%`；
- GPU memory 最后观测约 `1466 MiB / 6144 MiB`；
- private memory 观测峰值下界约 `15.74 GB`，随后回落到约 `4.11 GB`；
- 未见 cache-lock warning；
- 进程未自然退出，cache 持久化文件数仍为 `0`；
- 达到有界诊断门后主动停止，停止后 Python/PythonW count 为 `0`。

冻结工件：

- `our_solver_config.json`
- `run_manifest.json`
- `progress.json`
- `profile_termination.json`

分类：

`cold_jit_performance_red_after_owner_precompute_split`

物理解读：

- 普通数组/场初始化仍只有约 44 秒，继续证明“没有给初始速度”不是根因；
- owner-precompute 拆分把观测 private-memory 峰值从 `vf48m` 约 `42.7 GB` 降到至少 `15.74 GB`，说明它真实减少了冷 JIT 的 IR/RSS 压力；
- 但 15.8 分钟总 wall time 后仍没有完成一个物理 preflow step，因此该拆分只解决了部分编译膨胀，尚未解决可用性问题；
- 这不是物理 solver failure，也不是 preflow convergence failure；
- 因 cold run 没有自然退出并持久化 cache，不得启动 warm A/B，也不得恢复 vf48l production。

当前 `core.py` 中仍有多个大型 Taichi 编译单元：

- `_update_pressure_neumann_gradient_from_ib_nodes_kernel`: 约 3809 行；
- `_assemble_pressure_neumann_matrix_rows_kernel`: 约 1878 行；
- `_prepare_velocity_dirichlet_component_face_claims_kernel`: 约 1851 行；
- `_search_and_classify_grid_fields_kernel`: 约 1611 行；
- `_reconstruct_velocity_dirichlet_component_face_segment_claims_kernel`: 约 1033 行。

不能仅按行数盲拆。下一步应先利用已有 `HIBM_DEBUG_STAGE_PROGRESS=1` host boundary，在全新目录做同一 1-step、同一 15 分钟硬门的带日志诊断，定位最后一个已完成/开始的 host stage；如果现有 stage 粒度仍不足，再以 TDD 给 canonical-ledger build 内部的 kernel 边界接入原子 progress observer。只有定位到实际阻塞编译单元，才继续做第二轮通用拆分。

### 12.10 原子 preflow stage 追踪取代不可用的 stdout trace

先后尝试：

- `validation_runs/solver_soaks/vf48q_stage_trace_20260724_a`
- `validation_runs/solver_soaks/vf48q_stage_trace_20260724_b`

第一项受 stdout buffering 影响，第二项改成文件 trace 后仍没有得到 HIBM 内部阶段；这证明既有 `HIBM_DEBUG_STAGE_PROGRESS` hook 没有覆盖 runner 的 pre-predictor resource/search/build/velocity-row 路径。两项均已停止，不得据此判定 physics RED。

随后以 TDD 增加默认关闭、仅由 host 调用的原子 stage observer：

- runner 外层覆盖 HIBM resource allocate、search/classify、obstacle publish、boundary build、velocity-row assembly；
- flow advance 覆盖 pre-predictor HIBM、SST wall distance、SST transport、momentum predictor、projection HIBM、主压力投影及带一基索引的 consistency HIBM/pressure projection；
- canonical velocity-row transaction 内部覆盖 relocation clear/arbitrate/materialize、direct presample、segment-pair precompute、claim prepare、segment reconstruct、merge audit、marker closure、report；
- progress observer 写盘耗时从 flow/step timer 和 `canonical_ledger_build` stage timer 中分别、且只分别扣除一次；
- callback 失败使用专用 `PreflowStageObserverError`，不会被误分类为数值 `FloatingPointError/ValueError`；
- 默认 `None` 路径不创建 canonical wrapper/lambda，core 只保留直接 guard。

最终定向门：

- `16 passed`
- `36 subtests passed`
- `py_compile` 绿色
- `git diff --check` 绿色
- 两轮独立复审最终均为 Critical/High/Medium `0`

### 12.11 `vf48r` 与 `vf48s`：从外层 velocity-row 缩到 claim-prepare

第一层原子追踪：

`validation_runs/solver_soaks/vf48r_atomic_stage_trace_20260724_a`

结果：

- ordinary initialization: `42.7266198 s`
- resource allocate/search/classify/obstacle publish/boundary build 均已返回；
- `hibm_velocity_row_assembly_before` 后至少 `185.2 s` 无 `after`；
- 总 wall time 下界 `240.0 s`；
- `preflow_steps_completed = 0`
- 分类：`cold_jit_hibm_velocity_row_assembly_red`

第二层 transaction 追踪：

`validation_runs/solver_soaks/vf48s_inner_stage_trace_20260724_a`

结果：

- ordinary initialization: `42.6340277 s`
- relocation clear/arbitrate/materialize、direct presample 和 segment-pair precompute 均完成；
- segment-pair precompute 从约 `75.3 s` 的 before 走到约 `116.4 s` 的 claim-prepare before，约 `41 s` 内返回；
- `hibm_velocity_row_claim_prepare_before` 后至少 `201.9 s` 无 `after`；
- 总 wall time 下界 `318.4 s`；
- `preflow_steps_completed = 0`
- 分类：`cold_jit_hibm_velocity_row_claim_prepare_red`

因此 owner-precompute 拆分不是新的最终阻塞；实际剩余主编译单元是约 1851 行、三轴静态展开的 `_prepare_velocity_dirichlet_component_face_claims_kernel`。

### 12.12 claim-prepare runtime-axis TDD 修复

最小通用修复仅作用于 `_prepare_velocity_dirichlet_component_face_claims_kernel`：

- `for axis in ti.static(range(3))` 改为每个 target 内的 runtime `for axis in range(3)`；
- kernel 内五个 `ti.static(axis ==/!= ...)` 改为 runtime branch；
- 外层 `for target in ti.grouped(...)` 保持 GPU 并行；
- 每个 `(target, axis)` scratch/claim lane 仍由唯一 target thread 写入；
- 全局 report 仍使用既有 atomic add/min/max；
- 没有改 owner、segment、conflict、closure 或 commit 数学。

依据：

- Taichi pin 为 `1.7.4`，runtime vector/matrix indexing 自 1.4 起受支持；
- 新结构合同先 RED，再 GREEN；
- 真实 CUDA 三轴 ledger 夹具：
  - x/y sign 与 canonical storage；
  - negative-z forward storage；
  - 结果 `2 passed, 4 subtests passed`
  - 首次进程 wall time `384.22 s`
- 独立复审：Critical/High/Medium `0`；Low 为结构测试仍使用源码字符串，后续应升级为 AST outer-target/runtime-axis 合同。

### 12.13 `vf48t`：claim-prepare 已解阻，当前新阻塞是 marker closure

诊断目录：

`validation_runs/solver_soaks/vf48t_runtime_axis_trace_20260724_a`

运行身份仍为：

- `steps = 0`
- `preflow_steps = 1`
- `flow_predictor_substeps = 1`
- grid `4 x 256 x 320`
- 单 Python、单 CUDA、全新隔离 cache
- `--profile-wall-time`

冻结结果：

- ordinary initialization: `44.1341433 s`
- `claim_prepare_before` 约在总 elapsed `117.1151 s`
- `segment_reconstruct_before` 约在总 elapsed `237.0646 s`
- 因此 full-grid claim-prepare 已在约 `120 s` 内自然返回；
- `marker_closure_before` 约在总 elapsed `372.0576 s`
- 因此 segment reconstruct 已在约 `135 s` 内自然返回，merge audit 也已完成；
- marker closure 此后至少 `279.7 s` 无 `after`；
- 总 wall time 下界 `651.7 s`
- `preflow_steps_completed = 0`
- 停止后 Python count `0`，cache 文件仍为 `0`
- 分类：`cold_jit_hibm_velocity_row_marker_closure_red_after_runtime_axis`

这证明 runtime-axis 是有效的部分修复：它让先前确认卡住的 claim-prepare 在 full grid 上返回，并继续越过 reconstruct；但首个 preflow step 仍未完成，因此不得恢复 vf48l production，也不得启动 warm A/B。

当前下一执行门：

1. 正式 vf48l 继续冻结；
2. 在 `_close_owned_hard_targets_to_marker_constraints` 内增加同样的 host-only 原子阶段，至少区分 prospective sampling view、direct no-slip identity、fallback identity、初始 measure、Kaczmarz sweep 与最终 measure；
3. 用全新目录、相同 current source/config 做有界 1-step trace，先定位 marker-closure 内最后一个 before；
4. 仅对实际阻塞的 closure kernel 做通用 IR 拆分或 runtime-loop 化，不得跳过 marker closure、降低 closure tolerance 或硬编码本 case；
5. 只有 cold 1-step 自然退出、cache 正常持久化且物理门绿色后，才做同 source/config/cache 的 warm A/B；
6. 之后才恢复 windowed-stationary preflow、strict snapshot load、2-step、独立 50-step 与 Fluent strict comparison。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作原样保留。

### 12.14 `vf48u`：marker closure 内部定位到 Kaczmarz sweeps

在 `_close_owned_hard_targets_to_marker_constraints` 内增加默认关闭的六对 host-only stage：

- prospective sampling view
- direct no-slip identity
- conditional fallback identity
- initial residual measure
- conditional Kaczmarz sweeps
- final residual measure

fallback 与 sweeps 事件只在真实条件分支执行时发出；observer 异常不被吞掉，ledger rollback 后原样重抛。定向门为：

- `18 passed`
- `44 subtests passed`
- `py_compile` / `git diff --check` 绿色
- 独立复审 Critical/High/Medium `0`

诊断目录：

`validation_runs/solver_soaks/vf48u_closure_stage_trace_20260724_a`

冻结结果：

- ordinary initialization: `43.3563402 s`
- prospective sampling view 完成；
- direct no-slip identity 约 `14.4 s` 完成；
- fallback 未执行，说明 direct identity 已覆盖本次 marker；
- initial measure 约 `101.3 s` 完成；
- `hibm_marker_closure_kaczmarz_sweeps_before` 后至少 `281.5 s` 无 `after`；
- 总 wall time 下界 `754.4 s`
- `preflow_steps_completed = 0`
- 分类：`cold_jit_hibm_marker_closure_kaczmarz_sweeps_red`

因此 closure 的长时间不是 sampling fallback，也不是初始 residual measure；真实剩余热点是实际执行的 64-sweep Kaczmarz block。

### 12.15 Kaczmarz runtime-stencil 修复与 `vf48v`

`_marker_target_closure_kaczmarz_sweep_kernel` 原来同时静态展开：

- `3` 个 component axes；
- 第一轮 `2 x 2 x 2` gather stencil；
- 第二轮 `2 x 2 x 2` scatter stencil。

但该 kernel 已在 outer marker loop 前显式 `ti.loop_config(serialize=True)`，Kaczmarz/Gauss-Seidel marker 顺序本来就必须串行。最小修复因此只把上述三处 static loop 改成 runtime loop：

- marker 顺序不变；
- axis 顺序仍为 `0 -> 1 -> 2`；
- 八点 stencil lexical order 不变；
- 每轴四个局部累加器仍重新置零；
- 不同 axis 写不同 vector lane；
- 相邻 marker 的共享支撑写回仍由 outer serialize 顺序化；
- 64 次 host sweep、closure tolerance 和所有公式不变。

TDD/验证：

- 新结构合同先 RED，再 GREEN；
- `test_adjacent_moving_markers_publish_owned_hard_targets_as_a_j_fixed_point`
  - 真实 CUDA；
  - `1 passed`
  - `571.21 s`
- 独立复审 Critical/High/Medium `0`

full-grid 诊断：

`validation_runs/solver_soaks/vf48v_kaczmarz_runtime_trace_20260724_a`

结果：

- ordinary initialization: `42.3175053 s`
- claim-prepare 约 `110 s` 返回；
- segment reconstruct 返回；
- Kaczmarz sweeps、final measure、report 与整个 `hibm_velocity_row_assembly` 全部返回；
- `hibm_velocity_row_assembly_after` 在总 elapsed 约 `487.6877 s` 出现；
- 随后 canonical prepare/seal、pressure reachability/Neumann、pre-predictor HIBM 与 SST wall distance 均越过；
- 最后进入 `sst_transport_before`，总 elapsed 约 `685.5449 s`；
- `sst_transport_before` 后至少 `214.5 s` 无 `after`；
- 900.7 秒外部门超时，停止后 Python count `0`；
- `preflow_steps_completed = 0`，cache 文件仍为 `0`；
- 分类：`cold_jit_sst_transport_red_after_hibm_velocity_row_unblocked`。

这是一项明确的解阻结果：先前 `vf48m` 在 `8775.8 s` 后仍没有越过第一个 preflow step 的 HIBM 冷 JIT；当前 source 在约 `487.7 s` 已完成整个 HIBM velocity-row transaction，并继续推进到 SST transport。剩余长时间已经从“初始化/HIBM 黑箱”转移为明确的 SST transport 首次编译/执行。

当前最新执行门：

1. 正式 vf48l 仍冻结，因为 `preflow_steps_completed` 仍为 `0` 且 cache 未自然持久化；
2. 不得回退 runtime-axis/runtime-stencil 修复，也不得通过跳过 SST 或放宽 stationary tolerance 制造绿色；
3. 下一轮先在 SST transport 内增加/复用同步 wall-time 边界，区分 turbulence-property update、k transport、omega transport、Helmholtz/linear solve 与 field commit；
4. 只对实际阻塞的 SST kernel 做 runtime-loop/IR 拆分或通用 solver 优化；
5. 必须先让 cold 1-step 自然退出并持久化 cache，再做同 source/config/cache warm A/B；
6. warm A/B 后还要比较 steady-state step wall time，确认 runtime loop 没有不可接受的热运行回退；
7. 之后才恢复 windowed-stationary preflow、strict snapshot load、2-step、独立 50-step 与 Fluent strict comparison。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作原样保留。

### 12.16 `vf48w/vf48x`：cold 1-step 首次自然退出并完成 warm A/B

在 `vf48v` 已把 HIBM velocity-row transaction 解阻、最后阶段推进到 `sst_transport_before` 后，本轮继续以 TDD 增加 SST transport 内部的默认关闭 host observer。它覆盖：

- wall target guard、primal flux ledger、advection-rate reduction；
- 首个 transport slice 的 coefficient update、max diffusivity、base/previous copy；
- conditional MUSCL reconstruction、explicit transport、candidate diagnostics；
- state commit、x/y/z LOD、wall state、accepted-state diagnostics；
- positivity retry 的 transport-base restore；
- final coefficient/state diagnostics 与 volume moments。

在 `--profile-wall-time` 下，每个细粒度 callback 前先同步 Taichi，避免 callback 文件 I/O 与异步 GPU 工作重叠后被错误地从 SST compute wall time 中全额扣除；非 profile/default `None` 路径不增加同步或改变数值调用顺序。TDD 证据：

- 初始 RED：`21 failed, 10 passed, 10 subtests passed`；
- 第一轮 GREEN：`12 passed, 29 subtests passed`；
- 复审发现同步扣时与 retry-restore 两个 Medium 后再次 RED；
- 最终 GREEN：`14 passed, 29 subtests passed`；
- 一个真实 Taichi SST 默认路径回归：`1 passed`，`261.56 s`；
- `py_compile` 与 `git diff --check` 绿色。

#### cold：`vf48w_sst_inner_stage_trace_20260724_c`

运行身份与 `vf48v` 的 solver config 逐字段相同：

- config SHA256：`1C1F39FAC23922942F88024B2F0C55201D04687134361DC8EC727BCC6E921FFA`
- `steps = 0`
- `preflow_steps = 1`
- `flow_predictor_substeps = 1`
- grid `4 x 256 x 320`
- 单 Python、单 CUDA、全新隔离 cache
- `--profile-wall-time`

自然退出结果：

- ordinary initialization：`43.3919738 s`
- preflow step wall time：`1732.0842968 s`
- total elapsed：`1785.3276737 s`
- SST transport substeps：`22`
- momentum advection substeps：`72`
- `preflow_steps_completed = 1`
- status：`completed`
- 停止后 Python count：`0`
- cold 退出后即时观测 cache 已自然持久化：`782` files，约 `84.0 MB`

这推翻了先前“当前 source 的首步仍不能退出”的性能 RED：该 source 已首次在 full grid 上自然完成整个 cold 1-step，并把 cache 正常落盘。它仍只是单步执行性门，不是 windowed-stationary 预流收敛证明。

#### warm：`vf48x_sst_warm_trace_20260724_a`

warm run 使用相同 config、相同 source hashes 和 `vf48w` 的同一 cache：

- ordinary initialization：`15.9093105 s`
- preflow step wall time：`133.1271899 s`
- total elapsed：`154.6052229 s`
- SST transport substeps：`22`
- momentum advection substeps：`72`
- `preflow_steps_completed = 1`
- status：`completed`
- 停止后 Python count：`0`

cold/warm 对比：

- total elapsed 提速 `11.5477x`，减少 `91.34%`；
- ordinary initialization 提速 `2.7275x`；
- preflow step 提速 `13.0107x`；
- config 文件 SHA256 完全相同；
- cold/warm manifest 的完整 source-hash mapping 完全相同；
- NPZ 的 13 个 key/shape 相同；仅 `u/v/p/speed` 有 GPU/f32 顺序级差异，最大绝对差为 `9.5367431640625e-7`。

warm 自然退出并完成 cache 合并后，当前目录可复核为 `587` files（`586 .tic + 1 .tcb`），总计 `67,008,882 B`。因此 `782` 只保留为 cold 退出时的即时观测值，不作为当前目录计数。

因此，这组严格同 config/source/cache 路径的 A/B 强烈支持主差值来自多组大型 Taichi kernel 的 cold compilation/cache miss；单一 cold/warm 对不能把全部差值数学上唯一归因于 JIT。它明确否定了“没有赋初始速度”以及“需要多个 GPU 同时填场”这两个解释。当前单 GPU kernel 本来已并行；多 GPU 仍需要域分解、压力全局归约和新的数值合同，不能作为初始化开关。

cold/warm 完成后，observer 又接受了两项纯 host 诊断修复：profile callback 前同步、positivity restore before/after。它们不改变 kernel 数学，但会改变 source hash；因此上面的性能数字严格归属于 manifest 中锁定的 cold/warm source，最终 host-observer source 只宣称定向合同绿色，不伪称已经用新 hash 重跑 full-grid 性能 A/B。

当前最新执行门：

1. `vf48l` 不再因“cold 1-step 无法退出/cache 无法持久化”而阻塞；
2. 但不得把单步 completion 当作稳态预流或 Fluent parity 绿色；
3. 下一步先用 warm cache 恢复 windowed-stationary preflow，保留速度、压力、`k/omega/mu_t`、通量、no-slip/HIBM 与 CG 物理门；
4. 生成并严格加载 post-preflow snapshot 后，才恢复 2-step、独立 50-step 与 Fluent strict comparison；
5. 若还要继续降低 warm `133.1 s/step`，先分离 SST k/omega transport、SST momentum Helmholtz、PCG host reductions 与 pressure projection 的热运行占比，再做通用 solver 优化；不得跳过 SST、放宽 `0.01` 或硬编码本 case。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作原样保留。

### 12.17 代码正确性与初始化/预流热点定向修复

用户明确要求“解决问题优先，不要为了堆工作量去测试”。因此本轮没有跑全量测试、覆盖率或新的长时 production preflow，只对实际修改执行一一对应的 RED/GREEN，并给一个现成 CUDA 数值门设置硬超时。

确认并修复的正确性错误：

- SST transport 在隐式 LOD 结果违反 positivity 后，已经写入 live `k/omega`，随后先调用 `transport_base_restore_before` observer、再执行 rollback；若 observer/progress I/O 抛异常，原代码会跳过 rollback，留下部分提交或无效的湍流状态。
- 现在 observer-before 位于 `try`，`_restore_sst_state_from_transport_base_kernel()` 位于 `finally`；任何 `BaseException` 都不能绕过事务恢复，成功恢复之后才允许发送 observer-after。
- 对应结构性 RED 为 `0 != 1`，修复后单项 GREEN。

已落地的 warm-path 通用优化：

- `_compute_muscl_momentum_dual_geometry_kernel()` 只依赖当前 predictor transaction 内不变的 grid 与 obstacle mask。
- 旧路径在每次 MUSCL flux state 都重算全场 dual volume；当前路径在 `predict(..., advection_scheme="muscl_tvd")` 入口只准备一次，并用显式 `dual_geometry_prepared=True` 传给三处 flux evaluation 和 SST Helmholtz。flux helper 与直接 Helmholtz caller 的默认值仍为 `False`，所以现有独立调用者继续获得自备几何语义。
- 对锁定 warm artifact 的 `72` 个 accepted momentum slices、零 retry，旧路径在 predictor 内为 `72 x 3 = 216` 次几何 launch，加 Helmholtz 自备的一次为 `217`；当前实现为整个 predictor transaction 一次，结构上消除 `216` 次重复 full-grid launch。没有改变 flux、CFL、SSP-RK2、rollback、PCG 或 Helmholtz 方程。

被复审否决并已撤回的实验：

- 曾把 SST momentum Helmholtz 的 `component/axis_index` 从 `ti.template()` 改为 runtime selector，希望减少 cold JIT 特化。
- 独立复审指出 `axis_index` 已和三个不同 `edge_coefficient` template 字段一一固定；runtime axis 不减少该字段特化，反而失去静态折叠。真实 CUDA manufactured test 又在 `240.7 s` 上限内没有完成 cold compilation。
- 因此该实验和对应测试已经完整撤回；当前 production kernel 保持原有 template/static selector，不把未证明的改动留在工作树。

验证边界：

- 两个保留的新合同都先 RED 后 GREEN；只运行对应 host-only 文件，没有跑整套测试。
- 最终 host-only 定向文件为 `5/5` 通过，耗时 `1.512 s`；`git diff --check` 通过（仅保留既有 LF/CRLF 提示）。
- CUDA manufactured test 超时未产生 pass/fail 数值结果，终止后 Python/pythonw count 为 `0`；它发生在已撤回的 runtime-selector 实验上，不能作为当前 source 的验证证据。
- 因而本轮只宣称事务结构和重复 launch 结构已修复，不能宣称新 source 已完成 CUDA 数值回归，也不能宣称已经测得新的 full-grid wall-time 加速比。
- 撤回高风险实验并补齐 Helmholtz prepared wiring 后，独立复审 gate 为 `Critical 0 / High 0 / Medium 0`。

关于“ANSYS 初始化只需给初速度”的核对结论保持不变：

- 本 solver 已经给完整流场初始速度、`p=0` 和 inlet-derived `k/omega/mu_t`；慢的不是填数组。
- Fluent hybrid initialization 是初猜生成；锁定 Fluent reference 随后仍执行稳态 SST 求解。当前代码的一个 preflow step 则包含 HIBM、SST transport、MUSCL momentum、marker-nullspace pressure operator 与 FV-PCG，并非 initialization API。
- 当前 Taichi kernel 已在单个 CUDA GPU 上并行。增加第二个 GPU 不能并行消除 Python/LLVM cold JIT；真正 multi-GPU 需要流体域分解、halo exchange、压力/PCG 全局归约和新的数值/复现合同，不是一个安全的初始化开关。

下一项最高价值但本轮没有贸然实现的 warm 优化，是 fixed-solid preflow 的 SST wall-distance 缓存：当前每个固定几何 preflow step 都重新执行约 `327,680 cells x 129 segments` 的点-线段距离检查。实现时必须以 marker/topology generation、wall flags 和 moving-solid 阶段作严格失效，不能用“步号大于一”盲缓存。随后才值得处理 momentum PCG 的 device-resident reductions；不得以减少物理 slices、放宽 stationary tolerance 或跳过 SST 换取速度。

本轮仍没有 commit、stage、push、reset 或 clean；没有删除或覆盖既有 dirty/untracked 工作。

### 12.18 Squid sharp air-backed、H1 探针与 solid/case 合同修复

用户随后逐项核查了 squid sharp 路径。本轮按当前 refactored split source 重新定位并修复确认成立的问题；没有把旧 worktree 的行号或已修复旧实现直接套到当前代码。

流体 air-backed 状态合同：

- `pressure_outlet_zmin=False` 时不再只清 host 统计后提前返回；现在同时清空 device 上的 outlet reachable、next reachable、unreached component label 与 air-component selection，禁止闭出口运行复用旧分类。
- air-backed component 报告改为统计实际至少转换一个 cell 的 component，不再统计仅被选中但未转换的 slot。
- `write_hibm_air_backed_cell_pressures` 只对 `hibm_air_cell != 0 && obstacle != 0` 的 air obstacle 戳印，旧 air tag 不能污染已恢复为 active water 的 cell。
- FSI `save_state/restore_state` 同步保存、恢复 `hibm_air_cell`；restore 后清空所有由 obstacle topology 派生的 reachability/component 分类，避免 obstacle、air mask 与恢复速度处于不同事务版本。
- public conversion 合同明确为 caller-ordered 的 `apply -> fresh zmin reachability -> seed -> convert`，不再声称无前置重湿时可重复调用仍是 stateless。

H1 与最终压力顺序：

- pressure traction 与 split-viscous extended probe 共用 marker-to-rung 离散路径守卫；守卫读取与采样相同的 effective sampling obstacle view。
- 守卫允许探针先离开 marker 所在的连续 self-obstacle band；进入 fluid 后再遇 obstacle 才判定穿越，从而同时修掉 marker 边界单格过杀和 `marker_near_is_obstacle` 门控导致的穿薄特征欠杀。
- one-sided marker 只使用 `one_sided_probe_max_multiplier`，其他 marker 只使用 `two_sided_probe_max_multiplier`，两个旋钮不再通过 `max(...)` 耦合。
- post-solid 重 seed/convert 保留，因为它依赖 solid 更新后的几何与 reachability；最后一次可选 projection 返回后再戳印一次 `p_far`，零 active-row、未执行 projection 的路径也覆盖。
- generic assemble 与 squid case 都拒绝 `far_pressure_air_backed=True` 且 `pressure_outlet_zmin=False` 的不自洽组合。

solid/case 配置合同：

- Neo-Hookean 与 Tri-Mooney 都 fail-closed：`fixed_region_id` 不能等于 primary/secondary，且固定区域必须真实包含 surface faces。
- 新增显式 `--fixed-rim-region-id`，默认 `5`；同一值贯穿 fixed solid、reachability barrier、setup 与面积报告，移除生产路径的 region-5 魔法接线。
- air backing、far-inside/two-sided/one-sided probe multiplier 与 air-seed normal sign 都有 CLI 参数；默认保留当前 sharp 行为 `air-backed=True`、三个 multiplier `12.0`，但可用 `--no-far-pressure-air-backed` 恢复非 air-backed A/B。air seed sign 默认 `0.0` 双向扫描，不再复用 traction 的 region-level normal sign。
- sharp-only air-backing/probe 校验只在 `hibm_mpm_sharp` 模式执行；显式 legacy diagnostic 模式不会因未使用的默认 air-backed 参数而被错误拒绝。
- Neo fixed-node policy 显式接线；case 默认解析为 `pure_fixed_mass`，避免 any-fixed-particle 将共享 free/fixed node 整体锁死。库级默认仍保留原 API，Mooney 收到 Neo-only 显式选项时拒绝静默忽略。
- 新增物理/边界参数全部进入 checkpoint argument fingerprint；Neo 默认 `None` 在 fingerprint 中按实际执行值 `pure_fixed_mass` 归一化，因此显式等价值可以 resume，而真实策略变化仍被拒绝。旧 checkpoint 不能在参数变化后静默 resume。
- closure coverage 的完整 patience 窗口若缺必需字段现在 fail-closed；短于 patience 的运行仍保留 streak 语义。CSV 使用目标目录内唯一临时文件，并在序列化或 replace 失败后清理。
- 重力仍为零；本轮没有在缺少 benchmark 物理合同的情况下擅自加入重力。

定向 TDD/验证证据：

- fluid air-backed CUDA `4^3`：RED `5/5` 失败，修复后同文件 `5/5` 通过；另有两个现有 save/restore 聚焦回归通过。
- H1/final-pressure：host/AST `4/4` 通过；CUDA `16^3` runtime 当前四个单项均通过，覆盖 sampling-view self-obstacle 离开、foreign-obstacle re-entry 拒绝与 multiplier 双向解耦。
- case/solid 新合同：`16/16` 通过；同步更新的六个旧静态合同 `6/6` 通过；CSV/closure 邻近回归 `8/8`、真实 Taichi fixed-region `3/3`、mocked case integration `1/1` 通过。
- 第一轮独立复审发现 legacy sharp-only 校验与 Neo fingerprint 等价策略两个 Medium；两项聚焦测试先分别 ERROR/FAIL，修复后连同 sharp 对照 `3/3` 通过。
- 同轮收紧测试门：最终 `p_far` 戳印必须晚于该 advance 内全部 `project` 调用；新增 CUDA direct-helper 反例证明 path 从 self obstacle 进入 fluid 后遇 foreign obstacle 返回 crossed；one/two multiplier 双向隔离的新增 CUDA 单项通过。full-stress 与 split-viscous 两个生产 kernel 的共享 helper 接线继续由 AST 合同覆盖，没有为此重复两条较重冷路径。
- 修复后独立复审最终 gate：`Critical 0 / High 0 / Medium 0`。
- 定向 `py_compile` 与 `git diff --check` 绿色；只保留既有 LF/CRLF 提示。

验证边界：

- 没有跑全套、覆盖率、长时间 squid production、ANSYS windowed-stationary preflow 或 Fluent parity。
- 本节修复会改变 source hash；`vf48w/vf48x` 的 cold/warm 数字仍只属于各自 manifest 锁定的旧 source，不能当作本轮 source 的性能测量。
- 因此本轮只宣称上述正确性与配置合同在对应聚焦门内绿色，不宣称完整物理收敛、数值 parity 或新的 full-grid 加速比。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作原样保留。

### 12.19 第一轮高置信代码瘦身

用户指出当前代码过于臃肿、阅读困难。本轮遵守“只删除可证明无用或重复的实现”的边界，没有为了缩短文件而改写数值算法、公开 API、快照 schema、兼容输入或物理参数。

客观盘点显示主要阅读瓶颈仍是：

- `simulation_core/coupling/hibm_mpm/core.py`：本轮开始约 `30,012` 行；
- `simulation_core/fluids/solver.py`：本轮开始约 `27,789` 行；
- `cases/squid_soft_robot/runner.py`：本轮开始约 `1,876` 行。

已完成的高置信删除：

- `ruff F401` 只清除普通 split-module 中可由静态语义确认的 unused imports；没有触碰 `simulation_core.__all__` 的公开 facade exports。初次自动修复曾把 squid runner 的动态 context imports 误判为无用，独立复审发现后已完整恢复，并在文件头显式禁止该类 F401 自动删除。
- 全仓 AST `Name/Attribute/reflective-string` 引用扫描确认并删除 `16` 个零引用 private Taichi helper/kernel，包括旧 MUSCL vector slope/face flux、旧 SST trilinear/velocity derivative wrappers、reachability 的分离 expand/commit、旧 prospective-ledger audit 与未使用的 face-coordinate helper。
- 两套 HIBM node-search kernel 中删除六段 `nearest_external/global_external_seen` bookkeeping：这些分支计算最近 external marker、signed distance、boundary point、normal 与 projection weights，但结果从未被任何输出或后续条件读取；保留真实使用的 `nearest` 与 `nearest_global` 路径。
- 删除未消费的 coincident-probe compatibility 计算、segment-pair metadata loads、target-match 临时量与 solver 中的 dead locals；这些表达式没有写 field、atomic counter、report 或返回值。
- 删除 squid runner 中已经离开当前实现位置的旧 Windows atomic-write 事故注释；实际 CSV 原子替换与 cleanup 合同仍由 `history.py` 维护。
- 保留 squid runner 中静态看似未使用、但经 `{**globals(), **locals()}` 提供给 `step_loop.py`/`summary.py` 的全部依赖；这组导入也是 package 动态重导出面的一部分，不属于死代码。后续若要真正缩小它，应先用显式 typed context 对象替换当前隐式字典合同。

直接行数结果：

- HIBM core：`30,012 -> 29,631`，减少 `381` 行；
- fluid solver：`27,789 -> 27,475`，减少 `314` 行；
- squid runner：`1,876 -> 1,873`，减少 `3` 行；
- 三个最大直接目标合计减少 `698` 行；其他普通 squid 模块另删除少量静态可证的 unused-import 行。

验证：

- 目标范围 `ruff --select F401,F841`：`All checks passed`；
- 修改文件定向 `py_compile` 与 `git diff --check` 通过；
- 当前 squid sharp contract：`14/14` 通过；
- HIBM segment-projection/node-search CUDA 合同：`1/1` 通过，`12.307 s`。
- 独立复审确认 solver/core 的 16 个删除对象无精确代码、字符串、`getattr` 引用，也无 field/atomic/report/return 副作用；复审发现的 runner 动态-context Critical 已通过完整恢复导入消除。

验证边界与后续结构方向：

- 没有跑全套、覆盖率、长时间 preflow、squid production 或 Fluent parity；删除对象均为 private 且全仓零引用/零消费，不宣称新的数值或性能收益。
- 两个核心文件仍分别约 `29.6k/27.5k` 行，阅读负担尚未根治。下一阶段应按已有 package ownership 分批抽离“report/schema host logic”和彼此独立的 kernel family；不能一次性移动数千行 Taichi kernel，因为那会改变 JIT specialization/source hash、缓存行为和复审面。
- README 与 `docs/MODULE_MAP.md` 未修改：包结构、公开入口和 ownership 规则没有变化。本节是当前清理证据的唯一文档增量。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作原样保留。

### 12.20 Squid runner 显式 context 第一、二阶段

沿 12.19 的后续建议，本轮先处理 squid case 最妨碍静态分析的边界，没有进入数值循环或 Taichi kernel 拆分。

第一阶段消除了模块全局注入：

- `runner.py` 不再把 `{**globals(), **locals()}` 交给 `step_loop.py`/`summary.py`；三处调用现在只提供运行时 locals。
- `step_loop.py` 原有 `152` 个 hard context lookup 中，`55` 个函数、类型、模块和常量改由消费者直接 import，保留 `97` 个运行时值。
- `summary.py` 两个 builder 原有 `152` 个唯一 hard key 中，`36` 个静态依赖改由消费者直接 import，保留 `116` 个运行时值。
- 因此 `runner.py` 中仅为隐式注入保留的 `118` 个 imports 可以删除；这些名字全部仍由 package 的其他 canonical 子模块导出，旧 `cases.squid_soft_robot` 聚合导入面没有因此缩减。

第二阶段收紧了反向结果合同：

- `run_squid_step_loop()` 不再 `return dict(locals())`，避免把约数百个 loop-local 临时量泄漏回 runner 和 summary。
- 新增 frozen `StepLoopResult`，只返回 `rows`、`interface_reaction_state`、`sharp_coupling_state`、`partial_run_stopped`、`partial_run_reason` 五项真实跨边界状态。
- runner 改为显式属性读取；final/sharp summary 不再展开 `**step_loop_result`。
- checkpoint 的 source-pinned 静态合同同步改为识别 `StepLoopResult`，closing checkpoint 继续使用返回后的 interface/sharp state。

生产代码直接行数变化（相对 12.19 结束时）：

- `runner.py`：`1,873 -> 1,745`；
- `step_loop.py`：`3,347 -> 3,362`；
- `summary.py`：`4,477 -> 4,460`；
- `step_context.py`：`41 -> 55`；
- 四个生产文件合计净减少 `116` 行。更重要的是静态依赖和 loop 输出已变成可审计边界。

定向 TDD/验证：

- 新结构合同第一阶段 `3/3` RED，迁移后连同显式 result 合同最终 `5/5` GREEN；
- squid sharp 合同 `14/14`、checkpoint closing-state 合同 `1/1`、preflight-only 合同 `1/1`，合计定向 `21/21` 通过；
- repository architecture boundary 合同另有 `21/21` 通过；本轮实际执行的相关测试合计 `42/42`；
- 相关生产/新测试文件 `ruff F401,F841` 与 `py_compile` 通过；旧 package 大型测试模块可正常 import。

边界与下一步：

- 本轮没有改变 pressure/velocity/FSI/solid 算法、时间循环分支、参数或输出 schema，也没有运行长 production、full suite、覆盖率、preflow 或 Fluent parity。
- 输入端仍有 `97` 个运行时 hard keys 和大量历史 guarded hydration；下一阶段应按 `Settings / Resources / Callbacks / MutableState` 分组，并复用 `step_context.py`，不要创建百字段 mega-dataclass。
- 其中 production caller 实际只向 guarded hydration 提供六个初始状态；其余兼容入口在删除前必须单独决定 public compatibility，不得仅凭当前 runner 未使用就直接删除。
- 完成输入分组后，才进入 HIBM/fluid 独立 kernel family 的小批迁移。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作原样保留。

### 12.21 Squid step-loop typed input 第三阶段

按 12.20 的后续路线，`run_squid_step_loop()` 的输入边界已经从宽泛的 runner `locals()` 收敛为四个职责组：

- frozen `StepLoopSettings`：`78` 个已经解析和校验过的数值、算法与耦合配置；
- frozen `StepLoopResources`：`12` 个长生命周期 solver/material/spec/path/runtime 对象；
- frozen `StepLoopCallbacks`：`3` 个必须闭包捕获 live solver state 的 runner callback；
- `StepLoopMutableState`：`10` 个跨 step 变化的状态种子，包括原有 `first_step/rows/sharp_coupling_state/run_started_at_perf`，以及旧 guarded hydration 中 production caller 真正提供的六项状态。

四组由 frozen `StepLoopContext` 组合，runner 用同名关键字逐项构造；结构合同同时验证每组字段集合、字段总数 `103`、组间无重名，以及每个 constructor keyword 都来自同名 runner 局部值。因此，`run_squid_step_loop(locals())`、入口参数上的 `context: Mapping[str, object]`、字符串 `_required_context_value()` 查找均已删除。循环内部用于 payload/runtime 类型守卫的 `Mapping` 仍然保留。

旧入口还包含 `217` 个 `if '<local>' in context` hydration 分支。production caller 只真正提供其中六个状态，现已进入 `StepLoopMutableState`；其余 `211` 个名称只是函数拆分时泄漏出来的 loop-local 临时变量，没有仓内 caller，也没有作为返回合同使用，现已删除。数值循环主体继续使用原有局部变量名，未改写 pressure、velocity、FSI、solid、checkpoint 或诊断算法。

本阶段 TDD/验证：

- typed-context 合同先因四个类型不存在而 RED，完成后 `8/8` GREEN；
- sharp/case/checkpoint/preflight 定向合同通过；repository architecture boundary `21/21` 通过；package export/import `5/5` 通过；
- `ruff F401,F841`、`py_compile` 与 `git diff --check` 通过；
- 没有运行长时间 squid production、Taichi 数值回归、full suite、preflow 或 Fluent parity，不能据此宣称物理/性能 parity。

这个阶段改变了直接调用 `run_squid_step_loop()` 的内部 Python 合同：外部 caller 必须构造 `StepLoopContext`，不能再通过任意 mapping 注入 helper 或 loop-local。仓内唯一生产 caller 已迁移，package 的显式子模块和聚合导入合同保持可用。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作继续保留。

### 12.22 Typed-context 后续减量与 F821 审计

继续检查 12.21 后的样板代码时，明确保留了 runner 中四组 dataclass 的同名关键字装配：Python 3.10 没有 keyword shorthand；若用 `locals()`、反射投影或字符串 field list 缩短该块，会重新失去显式依赖和静态审计能力，代码虽然更短但合同更弱。

本轮只删除可证明没有复用价值的 pass-through：

- `step_loop.py` 中 `23` 个 immutable Settings/Callbacks 字段只在赋值后读取一次，现改为在唯一消费点直接读取 `settings.*` 或 `callbacks.*`；
- `first_step` 与 `run_started_at_perf` 两个只读 state pass-through 也改为直接读取 `state.*`；
- 数值、调用参数和值的来源均未改变，`step_loop.py` 从 `3,152` 行降至 `3,128` 行，净减少 `24` 行。

减量审计同时发现上一阶段的一个静态遗漏：typed input 不再需要 `Mapping` 参数后曾删除该 import，但循环内部仍有三处 `Mapping` payload/type guard。`py_compile` 不会解析未执行名称；新增的 F821 门发现问题后已经恢复 `collections.abc.Mapping`，没有移除任何运行时守卫。

新增单次 immutable alias 合同先 RED、精简后 GREEN。最终聚焦验证包含 typed-context、sharp、checkpoint、preflight、package export 和 architecture boundary；同时执行 `ruff F401/F821/F841`、`py_compile` 与 `git diff --check`。仍未运行长时间 squid production、Taichi 数值回归、full suite、preflow 或 Fluent parity。

本轮仍没有 commit、stage、push、reset 或 clean；既有 dirty/untracked 工作继续保留。

### 12.23 EasyFsi main 审计修复与有界性能收口（2026-08-10）

本节覆盖基线 `736a8ddebba353d55f2aaf559dab1ad3e631679d` 的只读审查后续。修复优先处理会让失败事务、旧缓存或非有限输入继续进入数值推进的真实缺陷；没有为了增加测试数量运行 full suite、长时间 preflow、50-step production 或 Fluent parity。

正确性与事务边界：

- HIBM→MPM 推进门现在要求 marker/load/stress/no-slip/projection 报告完整、计数一致、力与残差有限，任一缺失或失败均不再推进 solid。
- Marker–MAC 的新 prepare、失败 prepare 和 stale audit 会先退休旧 committed/prepared/pressure-nullspace 生命周期；marker/device loader 与 feedback 改为完整验证后再发布，失败不会留下可消费的半写几何。
- obstacle topology 的所有 writer 统一失效 pressure reachability、hard-face ledger/device mask 与 SST wall-distance identity；`save_state/restore_state` 同步覆盖 `fsi_pressure`、SST wall flags/mask/valid/cache key。
- `FluidDomainSpec` 与 fluid step API、TriSurface、hyperelastic、Mooney/Neo-Hookean、marker counted-field 和 segment endpoint 均在 device work 前拒绝 NaN/Inf、非法物理量、容量越界，以及 f32 溢出或非零值下溢为零；pressure RHS/metric 等派生系数也在任何投影状态写入前验证。
- Mooney 粒子在本步积分后越界会当步 fail closed，且“全部粒子越界但 tolerance 等于粒子数”不再被接受。
- squid checkpoint 升级为 v4，指纹加入 CAD/STEP、实际生效的 surface/volume cache、拓扑内容和遗漏的 projection 参数；runner 冻结初始化输入身份并在 source-dependent setup 后复核，运行中漂移 fail closed。restore 对版本、维度、步号和数组完整解码/严格校验后一次提交，避免半恢复。
- traction history 的 36 个字段、严格连续整数 temporal gate、fresh-clone canonical 测试、STEP part tag、`run_simulation.py --help` 正常退出和 checksum 目录闭包均已修复。
- IQN-ILS 先做全局缩放再 QR，范数/点积采用抗溢出实现；ReferenceCurve、loaded-force 与材料参数不再允许非有限值或负容差绕过。

性能与可读性：

- 固定几何 SST wall distance 现在使用具名严格 cache key，包含 geometry/device value digest、topology revision、wall flags、inactive axis、marker/segment count 与 endpoints。命中时跳过 full-grid/JFA/segment 内核；按当前目标网格每个固定步避免约 `327,680 x 129 = 42,270,720` 次点—线段检查。
- Marker–MAC 在 `marker_count <= 64` 时保留原小规模单核路径；更大 marker 集使用 device-resident exact-f32 open-address canonicalization，从 O(M²) 降为 expected O(M)，并保持最小 owner、`+0/-0` 和精确 target-conflict 语义。
- MPM bin cache 以 field identity、显式 position generation、count、radius 和 capacity 为键。official runner 在初始化和每个 solid position write 后单调推进 generation，使 post-solid feedback 的 bins 可被下一步 scatter 安全复用；scatter→solid→feedback 之间仍因位置真实变化而重建。
- graded-grid side/bridge 从 O(N²) 改为二分最小可行格数加线性包络；TriSurface 报告合并为一次 f64 snapshot；Turek-Hron NPZ 与 GIF 均逐帧流式处理，内存不再随 `frames x cells/pixels` 增长。
- 没有机械拆分 2.8 万行 Taichi 类。新增逻辑集中在统一 topology invalidator、具名 cache key、边界验证 helper 和 runner generation observer；大类按 kernel family 迁移仍应是独立 source-hash/JIT 任务。

保留的性能边界：

- particle-bin exclusive prefix scan 仍是 device serial。Taichi 1.7.4 的 `PrefixSumExecutor` 只接受具备 SNode 的固定 field，不能直接替换当前按粒子数惰性分配的 `ti.ndarray`；自写 Blelloch scan 会增加 `2 log2(H)` 次 launch，必须先做阈值 A/B，不能把理论并行当成已证实加速。
- 因此本轮解决了重复 rebuild 和大 marker 的降阶问题，但不宣称 prefix scan 已并行化，也不宣称新 source 已取得 full-grid wall-time 加速比。

验证边界：

- coupling 新合同 `19/19`、fluid 新合同 `14/14`，并完成有效 pressure projection、TriSurface 抵消哈希和各自聚焦旧 CUDA 兼容性检查；
- checkpoint/validation/STEP/CLI/artifact 聚焦反例全部由 RED 转 GREEN；
- numeric/solid/grid/runner/CI/renderer 聚合检查 `27 passed`；发布前 Linux-compatible fast unittest `94/94` 与 Turek renderer pytest `1/1` 通过；聚焦 CUDA state contracts 与 TriSurface cancellation 合计 `39/39` 通过；
- 51 个当前改动/新增 Python 文件 `py_compile` 通过，仓库 Ruff `F601/F821` 与 scoped `git diff --check` 通过；
- 未运行 full suite、覆盖率、长 preflow、50-step production、squid production 或 Fluent parity，因此不能据此宣称完整数值 parity。

CI 现在从 Python 3.10 兼容的锁定 `requirements.txt` 安装依赖（SciPy 固定为 `1.15.3`），并分为 Linux 快速静态/纯合同、Windows 完整 ANSYS 合同、以及仅 weekly/manual 触发的 self-hosted Windows CUDA 层；该 CUDA 层先运行聚焦 state contracts，再进入 10/30/50-step 验证。step50 CLI 只有真实通过状态返回零；远端 Actions 是否绿色仍必须以真实 run URL/status 为证据。
