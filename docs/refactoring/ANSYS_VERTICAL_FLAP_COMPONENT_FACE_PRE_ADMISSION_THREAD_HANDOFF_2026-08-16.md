---
status: paused-in-progress
branch: main
timestamp: 2026-08-16T22:57:38+09:00
stop_reason: user-requested pause while PRE failure-only diagnostic GREEN was unverified
authoritative_checkout: /home/zhuohengli/work/squid-robot/HIBM-MPM-refactored
host_workspace_mirror: D:/working/squid robot/simulation/src/reference/papers/HIBM-MPM/refactored
files_modified_at_pause:
  - benchmarks/official/solid_mpm_fsi_runner.py
  - docs/MODULE_MAP.md
  - simulation_core/coupling/hibm_mpm/constants.py
  - simulation_core/coupling/hibm_mpm/core.py
  - simulation_core/coupling/hibm_mpm/marker_mac_constraint.py
  - simulation_core/coupling/pressure_interface.py
  - simulation_core/fluids/solver.py
  - tests/benchmarks/test_canonical_production_runner_boundary_ledger.py
  - tests/cases/test_ansys_vertical_flap_fsi.py
  - tests/integration/test_ansys_vertical_flap_component_face_probe.py
  - tests/integration/test_ansys_vertical_flap_snapshot_diagnostic_replay.py
  - tests/solvers/_hibm_component_face_ledger_contracts.py
  - tests/solvers/test_core_fluid.py
  - tests/solvers/test_hibm_component_face_geometry.py
  - tests/solvers/test_hibm_marker_mac_pcg_work_elision_contract.py
  - tests/solvers/test_hibm_relocation_transaction_static.py
  - tests/solvers/test_hibm_runner_reachability_cache.py
  - tests/solvers/test_hibm_segment_pair_geometry.py
  - tests/solvers/test_hibm_shared_marker_sampling_identity.py
  - tests/solvers/test_topology_cache_invalidation_contracts.py
  - validation_runs/ansys_vertical_flap_fsi/scripts/run_preflow_snapshot_one_step_diagnostic.py
untracked_at_pause:
  - docs/MODULE_MAP.md.orig
  - simulation_core/coupling/hibm_mpm/core.py.orig
  - tests/solvers/test_hibm_marker_mac_reliable_residual.py
---

# ANSYS vertical-flap component-face PRE admission 交接

## 0. 暂停边界

这是用户要求的暂停点，不是完成报告。

暂停时已经执行的动作：

- 所有子代理均已停止或完成；
- 没有 `run_our_solver_vertical_flap.py`、pytest、Taichi 或项目 `.venv` Python 进程；
- 没有 commit、push、reset、checkout、clean 或批量删除；
- r27 snapshot、失败工件和已有 dirty worktree 全部保留；
- `simulation_core/coupling/hibm_mpm/core.py` 的 PRE failure-only 诊断补丁已落盘，但在用户喊停前尚未完成 post-core 测试和审查；
- `tests/integration/test_ansys_vertical_flap_component_face_probe.py` 的 refined RED 合同已落盘；
- 当前仅对这两个文件执行过一次 scoped `git diff --check`，结果为空；这不等于 py_compile、Ruff、pytest 或 CUDA GREEN。

暂停时的关键 SHA256：

```text
5df6507bf9e7e12319861730877e10145267b9f18dc419398912a8615d12e937  simulation_core/coupling/hibm_mpm/core.py
aca04901cef3be0ffe85839f0814b3450652679557e784b02bb9ade3d92fb283  simulation_core/coupling/hibm_mpm/marker_mac_constraint.py
1b393d0caa5b1eb32c5b0cc7da7792ae71bfcdc2dc2b73ac35e5d21298b71bfb  tests/integration/test_ansys_vertical_flap_component_face_probe.py
4f0c10508e823541ac183d1090ca253e5f495c5c3ae46be91b7acd86e0e42887  tests/solvers/test_hibm_component_face_geometry.py
b28e5d4762b9145ec52ae1f68bd7b95ed8b66ea61f8e23d02d00a6416d364e0a  tests/solvers/_hibm_component_face_ledger_contracts.py
a4b387afd63e5a4653a1c9584348af41ca8450a46af4de34a445a2bc6772b96c  tests/solvers/test_hibm_marker_mac_reliable_residual.py
```

`git diff --stat` 显示的五万多行变化主要包含工作树既有改动和 LF/CRLF 重写噪声，不能用它判断本线程的真实修改规模。必须按下面列出的文件 SHA、patch 文件和符号位置审查。

## 1. 用户目标与当前结论

用户提出的工作方式是：先用简单、结果已知的场景证明修复，再运行昂贵的 50 步正式仿真，以减少无效等待。

本线程严格按这个顺序推进：

1. 纯 host / 小型已知结果 RED；
2. 最小 production GREEN；
3. 独立只读审查；
4. 单个 focused CUDA 节点；
5. fresh source-matched preflow；
6. strict-load；
7. smoke8；
8. 独立 formal50；
9. 只有 formal50 完成后才允许 Fluent comparison。

当前最终结论仍是 **NO-GO**：

- r27 fresh preflow、strict-load 和 smoke8 全部通过；
- r27 formal50 只接受 4 步；
- 物理第 5 步在 canonical component-face PRE admission 阶段 fail closed；
- 尚未完成 50/50；
- 尚未运行新的 Fluent strict comparison；
- 不能声称 solver green 或 Fluent parity。

## 2. 为什么“已知结果优先”确实节省了时间

这条路线避免了至少三次盲目长跑：

- marker-MAC PCG 的失败诊断先在 2-marker 已知矩阵上证明，不先重跑 50 步；
- storage selector 的跨 kernel f32 near-tie 先由一个 exact CUDA 节点复现并转绿，再启动 fresh r27 preflow；
- reconstruction 的 strict-owner 和 live-topology fail-closed 又由一个 bounded CUDA 节点验证，避免 formal run 才发现 reconstruct 回归。

昂贵部分仍然存在：production 网格为 `4 x 256 x 320`，每个物理步包含大量流体/SST、压力投影、1600 MPM substeps 和 3 次 FSI coupling trial。冷 Taichi JIT 也很贵，所以全程坚持同一时刻只运行一个 Taichi/CUDA 任务。

## 3. 本线程完成了什么

### 3.1 r25：先处理 component-face endpoint/strict-owner 类问题

完成内容：

- 修正 cap endpoint 测试夹具的 marker 时序和 target 语义；
- 扩充 target-conflict failure diagnostic，使其保存 pair cache、author rows、marker state 和 post-admission raw fields；
- 对 diagnostic enrichment 加入局部异常隔离，保证新诊断失败时仍保留原 target-conflict；
- 修复 host normal 重建的 inactive-axis 投影和单位化顺序；
- 加入 JSON-safe、read-failure-preserves-base、full-valid=0 仍读 raw rows 等纯 host 合同。

r25 运行结果：

- fresh preflow 在 77/200 达到 stationary early stop；
- strict-load 通过；
- smoke8 通过；
- formal50 接受 44 步，在物理第 45 步发生 `prepare_pair_arbitration` target conflict；
- r25 formal failure 不是压力、no-slip、MPM 或 marker-MAC failure。

r25 snapshot 关键身份：

```text
NPZ SHA256 = 86151af1b034626cdd99b0261f91a82b9664fde48c74139a1496fa0f39dcb3cd
config      = 4356b023...1b30
source      = b4d1b51b...c63cc
geometry    = f300eeb9...72bd4
```

### 3.2 r26：marker-MAC PCG 失败先做已知矩阵诊断

r26 fresh source 重新完成：

- preflow 77/200 stationary；
- strict-load 通过；
- smoke8 8/8 通过；
- formal50 接受 24 步，物理第 25 步因 marker-MAC PCG exhausted 64/64 失败。

正式失败值：

```text
exact_residual_mps = 3334.932373046875
confirmation_count = 9
restart_count = 8
```

问题：旧异常只是普通 `RuntimeError`，`failure.json` 没有 rhs、diagonal、argmax row、support stencil 或 exact-refresh history，无法诚实判断是坏条件数、拓扑突变还是 recursive/exact residual 漂移。

解决：

- 在 `marker_mac_constraint.py` 增加专用、JSON-safe `.diagnostics`；
- 只在 max-iteration failure 路径下载 row/stencil 数据，成功路径不做全量 host transfer；
- `A_lambda` 使用 `rhs - exact_residual`，不读取已被 PCG 重用的 Ap buffer；
- tie 的 argmax 固定为最小 active row；
- 保存 confirmation history、row statistics 和 8-slot provenance；
- failed solve 保持不可 commit，fluid velocity bitwise unchanged。

已知结果 CUDA 合同：

```text
tests/solvers/test_hibm_marker_mac_reliable_residual.py::
HibmMarkerMacReliableResidualTests::
test_exact_residual_restart_commits_a_physically_converged_candidate

1 passed in 52.52s
```

同一节点先用 `max_iterations=1` 验证 failure diagnostics，再验证原 32-iteration reliable-residual/restart/commit 合同。

### 3.3 将 one-step replay 扩为 diagnostic-only formal-step-count replay

旧 replay 工具把 `step_count` 硬编码为 1，不能追到 r26 第 25 步。

解决：

- `run_preflow_snapshot_one_step_diagnostic.py` 增加 `--preserve-config-step-count`；
- 仅允许 paired config 的原始 `step_count` 严格为 int 50；
- observer 只记录 commit/publish 后的 accepted steps；
- failure metadata 保存 completed、last、attempted step 和原生 `.diagnostics`；
- snapshot before/after hash 必须不变；
- 一直标记 `diagnostic_only`，不得冒充 formal validation。

纯 host 证据：

```text
精确新增+既有节点：9 passed
整个 replay host 文件：24 passed
```

r26 replay 没有复现 marker-MAC step25。它先暴露了 component-face 冲突：第一次在 step8，补诊断后在 step10。这个差异被保留为“诊断轨迹数值敏感”，没有冒充正式轨迹。

### 3.4 证明并修复 storage selector 跨 kernel near-tie

真实证据表明，同一 direct row 在两个 force-inline kernel context 中得到不同选择：

- classifier cache 选择 storage offset 1；
- prepare live selector 选择 offset 0；
- pair cache 因此认为 route invalid，但 prepare 又把两作者写到同一 target，最终冲突。

根因：旧 tangential distance tie tolerance 只有

```text
max(1e-24, 1e-12 * ray_length_squared)
```

在 production f32 尺度上远低于一个 t² ULP，不能吸收 CUDA/FMA callsite 差异。

最小修复位于 `_select_canonical_component_face_storage_device`：

- 保留旧 absolute floor；
- raw t² 必须 finite；
- 首个 finite candidate 明确接受；
- 后续比较使用：

```text
max(
    old_floor,
    4 * eps_f32 * max(current_t2, best_t2, 1e-24_f32),
)
```

- 仍按 distance-first、smaller-progress true-tie 规则选择；
- 不改变 topology、route 或 target 语义。

可信 RED：

```text
classifier offset=1, prepare lower-face claim=1, upper-face claim=0
expected=(1,0,1)
```

GREEN：

```text
test_near_tie_storage_selector_keeps_cached_and_live_route_consistent
1 passed in 228.61s
```

负例将 boundary.z 移动一个 f32 ULP，断言真实可分 metric 仍选择 offset0，证明 C4 没有过宽。

reconstruction/fail-closed 回归：

```text
test_vf48i_strict_interior_owner_survives_ledger_reconstruction
1 passed, 2 subtests passed in 729.41s
```

覆盖 forward/reverse authors、reconstruct selector、live topology mutation、path2 error 和 ledger atomic rollback。

## 4. r27 fresh production 验证

### 4.1 fresh preflow

目录：

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/our_solver/unified_core_preflow200_interp_storage_tie_20260816_r27
```

snapshot：

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflow_snapshots/unified_core_preflow200_interp_storage_tie_20260816_r27/preflow_state
```

结果：

- exit 0；
- 77/200 stationary early stop；
- history 1..77 连续；
- 3 个连续 stationary windows；
- union 最大相对 span `0.00940051 <= 0.01`；
- 77 步 canonical danger counts 全 0；
- QP/CG、pressure、no-slip、MPM 全部门通过；
- manifest 98 个 source 文件与当时源码逐项匹配。

snapshot hash：

```text
JSON = 3c852fb3f05392c85f14277ad133e314ef3eab69ef786a610727df651d10f14e
NPZ  = d4d72df1649a8eae0174bb48561dbe866dd630b18f7ee446ff7a848eaacba2c8
config   = 4356b023...1b30
geometry = f300eeb9...bd4
source   = 4acc4d11...bbea5
```

### 4.2 strict-load

目录：

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflight/unified_core_preflow200_interp_storage_tie_20260816_r27_strict_load
```

结果：exit 0、0/0、`preflow_snapshot_loaded=true`、identity 三元完全匹配。

### 4.3 smoke8

目录：

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/unified_core_fsi8_interp_storage_tie_r27_20260816_r01
```

结果 8/8 GO：

- history/fields 1..8 连续；
- FSI 全部 3 iterations；
- abs residual max `5.4703643e-6 < 1e-5`；
- relative max `2.2094188e-4 < 1e-3`；
- canonical/observer danger counts 全 0；
- closure max `5.0595094e-7 < 1e-6`；
- QP/CG exact relative max `7.3317230e-7 < 1e-6`；
- null invalid=0；
- no-slip invalid=0，residual max `2.0225521e-6 < 1e-4`；
- MPM OOB/clamp=0。

### 4.4 formal50

目录：

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/unified_core_fsi50_interp_storage_tie_r27_20260816_r01
```

终态：

- accepted boundary=4；
- 物理 step5 在 commit/publish 前 fail closed；
- step5 history/NPZ 不存在；
- step1..4 全部硬门通过；
- 无残留进程。

工件：

```text
failure.json SHA256 = 636ac1df86252a2b7548a0f4d3d537219438f94fdae39a833357f8642054fe80
progress.json SHA256 = 481249bafe2c1a38164db7798dbe119ad43eb7a8299c699e6ed8022d53d97aec
```

## 5. r27 step5 新问题

异常：

```text
CanonicalVelocityBoundaryTopologyIncompatibilityError
reason_code=target_conflict
count=4
```

首冲突：

```text
component_face = (0,98,149)
component_axis = 2
region = 202
source = prepare_pair_arbitration
path = 0
authors = (0,98,148) / (0,98,149)
segments = (113,114) / (112,113)
shared marker = 113
selector offsets = (1,0)
adjacent_direct_pair_target_valid = 1
pair admission/full = false/false
```

这证明 selector 修复已生效。旧问题是 cache/live 路由分裂；r27 两作者现在一致路由到同一 target。

由持久化 f32 marker/face 几何重算：

```text
first raw t  = 3.704267491199071e-4
second raw t = 0.9999744507382295
first d2     = 8.283520040981581e-8
second d2    = 8.283520373789444e-8
delta d2     = 3.328078635364529e-15
tie band     = 4.656603991297431e-14
delta/band   = 0.07147
```

两个投影都严格位于段内，没有 endpoint clamp，也没有 strict owner。

共享 marker 113 捕获：

```text
first closest -> shared delta^2  = 3.343987594511111e-15
second closest -> shared delta^2 = 1.590796917645223e-17
geometry tolerance^2             = 1.247882474131663e-16
```

第一段超出当前 shared-vertex capture tolerance `26.7973x`，第二段在 tolerance 内。C0 tangent alignment 约 `-0.9999999769`。

当前最早能从保存工件证明的失败谓词是：

```text
adjacent_distance_tie = true
internal_shared_vertex_coownership = false
```

随后 PRE 保持 `valid=0`，admission/full 都为 false，prepare 无法启用 finite-pair reconstruction，最终写入两个不同 author targets 并触发 path0 conflict。

### 5.1 证据仍缺什么

旧 failure payload 在 admission=false 时没有保存：

- raw author normals；
- nominal/actual sample points；
- actual sample velocity/valid；
- exact search-support flags/radii；
- projection registry counts 和 shared degree；
- ordered PRE predicate results；
- prepare-time selector alpha 和实际 interpolated targets。

因此当前几何能证明 shared capture 失败，但不能严格排除更早的 topology、normal、anchor、face-support 或 search-support 谓词同时失败。

这就是为什么没有直接放宽算法。

## 6. 用户喊停时的未完成诊断补丁

### 6.1 已完成的 RED

`tests/integration/test_ansys_vertical_flap_component_face_probe.py` 增加了两个 pure-host 合同：

```text
test_production_conflict_diagnostic_reports_pre_admission_pair_host_metrics
test_production_conflict_diagnostic_preserves_base_when_pre_admission_read_fails
```

test-only 阶段结果：

```text
2 failed
```

失败点分别是缺少：

```text
pre_admission_pair
pre_admission_pair_capture_error
```

这是可信 RED。test-only py_compile 已通过。随后 schema 又补充了：

- `source=raw_transaction_fields_and_context`；
- `source_precision=f32_field_read_promoted_to_python_float_recomputed_f64`；
- `kernel_predicate_parity_guaranteed=false`；
- route guard、raw authors、topology、tie/shared metrics；
- JSON-safe；
- read failure 必须保留 face/authors/markers/pair cache 和原 target-conflict。

### 6.2 已落盘但未验证的 core 中间态

用户喊停前已经落盘：

- diagnostic context 新增 cell-face、projection registry 和 source-search support 的 host references/scalars；
- 新 helper：`_canonical_velocity_dirichlet_pre_admission_pair_diagnostic`，当前约在 `core.py:29136`；
- failure diagnostic callsite，当前约在 `core.py:29478`；
- 任意新字段读取失败写 `pre_admission_pair={available:false, reason:raw_field_read_failed}` 和局部 capture error。

对应补丁文件：

```text
C:\Users\lizhu\.codex\visualizations\2026\08\14\019ffed4-abf3-7c82-9e17-bdd2278410e1\r27_pre_admission_core_green.patch
C:\Users\lizhu\.codex\visualizations\2026\08\14\019ffed4-abf3-7c82-9e17-bdd2278410e1\r27_pre_admission_test_red.patch
C:\Users\lizhu\.codex\visualizations\2026\08\14\019ffed4-abf3-7c82-9e17-bdd2278410e1\r27_pre_admission_test_schema_red_v2.patch
```

重要：当前 helper 仍把下面一组 kernel branches 明确标为 unsupported：

```text
normal_cone_anchor_face_support_endpoint_transport_and_derived_terminal
```

也就是说，用户喊停时它还没有达到 reviewer 要求的完整 PRE 证据面。它只重建了 raw authors、search support、registry、segment t/d2、tie 和 shared-vertex 基本谓词。

当前 **没有** post-core py_compile、Ruff、host GREEN、完整 host 文件回归、独立 post-diff review、replay 或 CUDA 证据。继续工作时必须先审查并补全/收窄该中间态，不能把 SHA `5df6507b...` 写成已验证 GREEN。

## 7. 遇到的问题与处理方式

| 问题 | 证据 | 处理方式 | 当前状态 |
|---|---|---|---|
| 长仿真成本高 | 冷 JIT 数分钟，focused reconstruct 节点约 12 分钟 | known-result RED -> focused CUDA -> preflow -> smoke -> formal | 流程有效，继续坚持 |
| source 变化后旧 snapshot 失效 | 每次 manifest/source SHA 改变 | fresh source-matched preflow，strict-load 三元校验 | 已执行 r25/r26/r27 |
| FSI residual 语义被误读为 AND | production 使用 abs OR relative | 审计源码，单项越界只作 advisory，runner converged 为硬门 | 已澄清 |
| r26 marker-PCG failure 缺证 | 普通 RuntimeError 只写最终 residual | failure-only native diagnostics + 2-marker known result | focused GREEN |
| replay 固定一步 | 工具硬编码 step_count=1 | 增加 strict preserve-config-step-count=50 diagnostic mode | host GREEN |
| storage selector 跨 kernel 翻转 | classifier offset1、prepare offset0 | C4 f32 relative t2 band，保留 distance/progress 语义 | focused CUDA GREEN |
| r27 新 PRE ambiguity | route bit1、admission0、distance tie | 先补 failure-only diagnostic，不先放宽算法 | 进行中，已暂停 |
| Windows/WSL 编辑通道混乱 | UNC/apply_patch 失败；raw SHA 与 git blob SHA 混淆 | 只在 visualizations 生成标准 diff，再用 WSL `git apply` | 后续沿用 |
| 行尾造成巨大 diff stat | 21 文件显示数万行改动 | 用 scoped SHA、symbol diff 和 patch 文件审计 | 未清理，必须保留 |

## 8. 已锁定的设计原则

后续不得违反：

- 不平均两个 author targets；
- 不靠扩大 `1e-6` conflict tolerance 掩盖几何冲突；
- 不做 face-axis 外推；
- 不写 ANSYS marker/row/region 硬编码；
- 不从 `step_fields/*.npz` 恢复 solver；它们不是 transaction checkpoint；
- source 改变后必须 fresh preflow；旧 snapshot 仅允许显式 source-diff 的 diagnostic-only replay；
- 同一时刻只跑一个 Taichi/CUDA 任务；
- focused test 不能冒充 formal50；
- diagnostic replay 不能冒充 production identity 或 Fluent parity；
- 所有 component-face 放宽必须在 PRE、live cache reproof 和 reconstruction 同步，不能只改一个阶段；
- shared-vertex topology、degree2、nearest、weight、region、finite、normal cone 和 live target ULP guard 必须保留。

## 9. 候选算法方向，尚未授权

当前 PRE shared-vertex capture 只用 `geometry_tolerance^2`，而 reconstruction 已允许 tie-derived shared proximity。这是一个跨阶段不一致候选。

不能直接使用完整 grid-derived tie band，因为强各向异性网格上可能沿短 segment 方向过宽。当前更窄候选为：

```text
capture^2 = max(
    geometry_tolerance^2,
    min(
        tie_tolerance_squared,
        4 * eps_f32 * min(first_segment_length^2, second_segment_length^2),
    ),
)
```

r27 尺度：

```text
current capture radius       = 0.01117 um
grid tie-derived radius      = 0.21579 um
segment-length-limited radius = about 0.1078 um
r27 first shared miss        = about 0.05781 um
```

该候选在 r27 尺度上足够，但仍是 **NO-GO**，直到真实 failed-trial normals、registry、anchor、face support 和 actual samples 被新诊断 replay 保存并全部核验。

不要新增两个 author target 的绝对相等门槛。现有 shared-vertex 语义允许连续速度梯度；唯一 canonical target 应来自 shared marker，并由 live ULP-scale target reproof 验证。

## 10. 继续工作的精确顺序

### 10.1 先重建暂停边界

只读执行：

```bash
cd /home/zhuohengli/work/squid-robot/HIBM-MPM-refactored
git status --short
sha256sum \
  simulation_core/coupling/hibm_mpm/core.py \
  simulation_core/coupling/hibm_mpm/marker_mac_constraint.py \
  tests/integration/test_ansys_vertical_flap_component_face_probe.py
ps -eo pid,args
```

期望核心 SHA 就是第 0 节记录的值，且没有 solver/pytest/Taichi workload。若不同，先查来源，不要 reset。

### 10.2 完成 failure-only PRE diagnostic

1. 只读审查 `r27_pre_admission_core_green.patch` 与当前 `core.py`；
2. 保留 test-only RED；
3. 将 r27 算法决策所需的 raw normal、anchor residual、face-support bracket、normal/chord alignments、face-ray progress/smooth cone 和 threshold margin 加入诊断；
4. 对无关 endpoint/transport branch 可返回明确 unsupported，但不能在 r27 路径上缺证；
5. 添加一个使用 r27 float32 几何的 negative host fixture，断言 `first_failed_predicate` 是 shared-vertex capture；当前正例只证明 schema 和 all-green 重建；
6. 任何读取错误仍只写 local capture error，原 target-conflict 必须保留；
7. 先跑 py_compile、Ruff E9/F、scoped diff-check；
8. 只跑两个 exact pure-host nodes，再跑整个 component-face probe host 文件；
9. 请求独立 post-diff review；
10. 在 review GO 前不运行 replay。

### 10.3 diagnostic-only replay

新 output 必须不存在。建议名称：

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflight/unified_core_fsi50_r27_pre_admission_diag_20260816_v1
```

建议命令：

```bash
cd /home/zhuohengli/work/squid-robot/HIBM-MPM-refactored
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
/usr/bin/timeout 3600s .venv/bin/python \
  validation_runs/ansys_vertical_flap_fsi/scripts/run_preflow_snapshot_one_step_diagnostic.py \
  --snapshot validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflow_snapshots/unified_core_preflow200_interp_storage_tie_20260816_r27/preflow_state \
  --config-json validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/unified_core_fsi50_interp_storage_tie_r27_20260816_r01/our_solver_config.json \
  --source-manifest-json validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/unified_core_fsi50_interp_storage_tie_r27_20260816_r01/run_manifest.json \
  --output-dir validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflight/unified_core_fsi50_r27_pre_admission_diag_20260816_v1 \
  --preserve-config-step-count \
  --allow-source-diff simulation_core/coupling/hibm_mpm/core.py
```

接受边界：

- snapshot before/after JSON 和 NPZ hash 必须不变；
- source diff 必须精确只有 `core.py`；
- 若复现同一 r27 step5 witness，读取 `pre_admission_pair` 全部 metrics；
- 若在不同步数或不同异常先失败，只把它当 diagnostic divergence，不允许据此修改 shared-vertex 算法；
- 一直保持 `diagnostic_only`、`formal_validation_eligible=false`。

### 10.4 若诊断证明 r27 其余 PRE 谓词全绿

先写 RED，不先改 production：

- exact r27 float32 witness；
- author order 反转、rerun byte stability；
- PRE admission、live cache reproof、reconstruction 三阶段一致；
- segment-length-limited band 的 just-outside negative；
- 强各向异性 negative；
- closest-pair delta 超限；
- nearest/weight/region/topology/degree/T-junction negatives；
- normal flip、sharp corner、face-ray cone negatives；
- live marker target mutation path2 atomic failure；
- high continuous velocity gradient 正例，不增加 cross-author absolute target gate。

然后做最小同步实现、独立 review、单个 focused CUDA 节点和 reconstruction 节点。

### 10.5 fresh r28 production chain

任何算法或 kernel/source 改动都使 r27 snapshot 失去正式身份。必须新建 r28：

1. fresh preflow，最多 200，windowed stationary `20/10/3`；
2. strict-load；
3. smoke8；
4. 独立 formal50；
5. 只有 50/50 且工件连续、全部物理门通过，才执行 locked Fluent comparison；
6. 用户要求的每个 Fluent comparison error 必须 `<=10%`。

## 11. 恢复时优先阅读的文件

按顺序：

1. 本文档；
2. `docs/README.md`；
3. `docs/MODULE_MAP.md`；
4. `docs/refactoring/ANSYS_VERTICAL_FLAP_50_STEP_CLOSURE_PERFORMANCE_THREAD_HANDOFF_2026-08-14.md`；
5. `simulation_core/coupling/hibm_mpm/core.py` 中：
   - `_select_canonical_component_face_storage_device`；
   - `_canonical_component_face_finite_segment_union_owner_geometry`；
   - `_canonical_component_face_smooth_shared_vertex_pair_cache_is_current`；
   - `_canonical_velocity_dirichlet_pre_admission_pair_diagnostic`；
   - `_canonical_velocity_dirichlet_first_target_conflict_diagnostic`；
6. `tests/integration/test_ansys_vertical_flap_component_face_probe.py`；
7. `tests/solvers/test_hibm_component_face_geometry.py`；
8. `tests/solvers/_hibm_component_face_ledger_contracts.py`；
9. r27 formal `failure.json`。

## 12. 不要误报的内容

- test-only RED 不是 GREEN；
- 当前 core SHA `5df6507b...` 尚未验证；
- r27 smoke8 通过不等于 formal50 通过；
- r27 formal 4/50 不等于可外推 50 步；
- diagnostic replay 不等于 formal validation；
- focused CUDA 节点不等于 full suite；
- 没有新的 Fluent comparison；
- 当前没有 commit 或 PR。

下一线程的第一件事不是启动 r28，也不是放宽 tolerance。先把暂停中的 PRE 诊断补丁变成可信 GREEN，并用 diagnostic-only replay 获取真实 failed-trial 输入。只有这样，下一次昂贵 preflow 才有意义。
