# ANSYS 竖直薄板固体子步 A/B 先验门槛

日期：2026-08-25
状态：数值门槛已在任何 fixed1600/adaptive FSI 结果前锁定；r01 preflow 失败，下一轮必须使用 r02
适用源码：`codex/perf-r114-hotpath` 当前未提交 source-matched 工作树

## 1. 目的与证据边界

本文锁定 fixed1600 reference override 与 production adaptive solid substeps
的精度、稳定性、物理时间和性能晋级门槛。不得在看到 FSI1、FSI2 或 FSI8
结果后修改门槛来迁就 adaptive 方案。若门槛失败，应修改通用 controller
（accuracy/deformation guard、step-doubling 或 fail-closed retry），随后更换 run
identity 并从 fresh source-matched preflow 重新开始。

已完成的小型 strict-CUDA 单宏步测试只验证真实 device/controller 契约：adaptive
选择 2 步、fixed override 选择 4 步，两者均完整消费 `dt_s`、保持宏步 damping、
无 retry/OOB/clamp，且状态有限。它不属于下述 fixed1600 精度结果。

### 1.1 r01 已消耗且失败，禁止复用

- r01 dry-run 成功，manifest 记录 `dry_run=true`、strict CUDA request、`f32`、seed 0
  和 96-file source surface；dry-run 没有生成 snapshot 或 FSI target。
- r01 production preflow 在第 `12/200` 步失败，完成 `11` 步，总 elapsed
  `2330.9217 s`。失败是 SST accepted-time ledger 的固定 32-ULP 审计误报：
  requested `5e-4 s`，报告的 reconstruction tail `3.79471e-18 s`，固定容差
  `3.46945e-18 s`。失败目录必须原样保留：
  `validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/ansys_vf__solid_substep_ab__20260825__r01__preflow/`。
- r01 没有生成 snapshot root、manifest 或 generation NPZ，因此没有任何 r01 FSI
  可以合法启动。`progress.json` 中的 SST/MUSCL totals 来自最后一个成功步的累计状态，
  不能反推失败第 12 步的 exact accepted slice count。
- 现场最小重现得到 `257` 个 accepted slice、loop-tracked remaining 精确为 `0`，但
  `math.fsum` 重建账本留下 `33 ULP` 尾差，证明固定 32 ULP 与 accepted operation count
  无关，会误拒绝合法闭合；删除一个真实 accepted slice 仍远超新的动态容差。
- 修复保持物理积分路径不变：SST/MUSCL 首个和 retry trial 都在物理 kernel 前拒绝
  `dt <= shared floor`，循环自身必须先证明 remaining 精确为 `0`，之后才允许按 accepted
  slice 数量推导的 ULP 容差处理 `fsum` reconstruction tail。runner 的 accepted-substep
  count 同时改为必填，缺失、零、负数均 fail closed。
- 聚焦 TDD 证据：host RED `3 failed, 2 passed, 8 subtests passed`，GREEN
  `5 passed, 13 subtests passed`；有效 CUDA RED 两项都触达禁止的 SST/MUSCL kernel，
  GREEN `2 passed in 53.61s`；直接相关 host 扩大门槛
  `82 passed, 1 skipped, 7 subtests passed`；SST/MUSCL/Helmholtz retry Taichi 门槛
  `3 passed in 377.17s`。这些证据不是 fresh preflow、FSI A/B 或 strict50。
- 数学复审、集成复审和全新独立 reviewer 均给出进入 fresh source-matched preflow
  的接受结论，P0/P1/P2 均为 `0/0/0`；三者都明确限定为本轮五文件修复，不能外推为
  strict preflow、FSI1/2/8/50 或发布证据。

## 2. 完全相同的实验身份

preflow producer、fixed、adaptive 和执行比较器时的当前工作树必须满足：

- 完全相同的 `source_sha256`，且记录的 source map 必须与比较器现场重算的当前
  source map 精确相等；仅 fixed/adaptive 彼此相等不足以排除 stale-source evidence；
- 同一 strict CUDA backend、Taichi `f32`、random seed 和 offline cache 配置；
  preflow/fixed/adaptive 的 `run_manifest.taichi_runtime` 必须记录同一 request，最终
  compact/summary 的
  `taichi_runtime_identity` 必须记录并闭合实际 `cuda`、`f32`、seed `0`、
  `strict_arch_verified=True` 和实际 cache identity；
- 同一个 fresh strict preflow snapshot；必须定位本 campaign 的 `__preflow`
  producer，并闭合 producer manifest/config/compact/summary、snapshot manifest、
  唯一 generation NPZ 及其 SHA256；
- 同一 `dt_s=5.0e-4 s`、步数、网格 `4 x 256 x 320`、固体粒子
  `1 x 256 x 20`、每个 physical face 的 marker `64`；正式
  `traction_marker_layout=dual_physical_faces`，所以 face count 为 `2`、实际 marker
  count 为 `128`；
- 同一材料、边界条件、HIBM 几何、流体预算、收敛容差、observer/export 和
  wall-time profiling 设置；正式 A/B 必须同时设置
  `flow_report_include_percentiles=True`、`profile_wall_time=True` 和
  `save_step_fields=True`；
- config 中唯一允许的求解差异是：fixed 为 `solid_substeps=1600`，adaptive
  为 `solid_substeps=None`；
- preflow config 必须按 runner 的 FSI-only snapshot 字段表投影后与 fixed/adaptive
  config 相等；producer 为 `step_count=0`、snapshot-out，FSI 为对应 step count、
  同一路径 snapshot-in，输入/输出方向必须互为预期，不能用手写字段子集近似；
- 每组 `run_manifest.config == our_solver_config.json == compact.config`；
  `our_solver_history.csv` 必须与 compact history 闭合，summary 的
  `step_artifact_validation` 必须等于实际 step frame/history 序列和计数；runner
  原始 history 中并存的 `_n`/`_N` force-unit aliases 必须在生成 CSV fieldnames 和行
  之前，使用与 compact JSON 相同的 canonical alias 规则折叠，不能让 CSV 保留额外列；
- output directory 和 run label 必须不同且从未使用；directory basename、manifest
  run label、summary run label、summary output path、final NPZ summary path 都必须闭合。

`solid_substeps`、solid CFL 和 damping 是 FSI-only snapshot identity 字段，所以
一份 fresh preflow 可以安全共享；禁止从 `step_fields/*.npz` 重启。

## 3. 比较定义

标量 `a`（adaptive）与 `f`（fixed）使用：

```text
close(a, f; r, A, S) :=
    abs(a - f) <= max(A, r * max(abs(a), abs(f), S))
```

其中 `r` 是相对容差、`A` 是量纲相关绝对容差、`S` 是防止近零值产生无意义
相对误差的比较尺度。向量先比较各分量，再比较 2-范数。时间序列必须同时满足
每一步 `close` 和：

```text
NRMSE(a, f; S) :=
    sqrt(mean((a - f)^2)) / max(sqrt(mean(f^2)), S)
```

所有字段必须存在且 finite；缺字段不是“不可用”，而是 gate failure。

## 4. 不可放宽的精确门槛

每一个 accepted FSI 宏步必须满足：

- `requested_macro_dt_s == dt_s`；
- `fluid_accepted_time_s == dt_s`，且
  `fluid_remaining_unadvanced_time_s == 0`；
- `solid_accepted_time_s == dt_s`，且
  `solid_remaining_unadvanced_time_s == 0`；
- 时间相等只允许 operation-count 推导出的 `fsum` reconstruction 浮点尾差；在应用
  该容差前，SST/MUSCL loop-tracked remaining 必须精确为零，且容差上限严格小于
  shared minimum admissible physical slice 的一半，不能掩盖任何真实少推进的 slice；
- fixed 的 `solid_substeps_selected == 1600`、
  `solid_accepted_substep_count == 1600`，且无 retry 时
  `solid_substeps_executed_total == 1600`；
- adaptive 的 selected、accepted 和实际执行计数与生产 report 一致；
- rejected trial 不增加 accepted time、feedback、history 或 checkpoint；
- 每步 `solid_step_kernel_launch_count == solid_substeps_executed_total`，
  `solid_guard_batch_count == solid_rejected_trial_count + 1`，且 selector
  accepted-state scalar read 次数与 selector evaluation 次数一致；
- packed report transfer 只统计实际进入
  `end_out_of_bounds_guard_batch()` 的 device-to-host read attempt，因此每步必须
  `1 <= packed transfer count <= guard batch count`；在首个 `solid.step()` 内提前
  reject 的 trial 不产生 packed read，不能用 `rejected + 1` 伪造；
- 正式 profiling run 每步 `solid_wall_time_synchronized == True`，顶层 solid
  wall time 必须等于逐步值之和；
- `mpm_grid_out_of_bounds_particle_count == 0`；
- `mpm_deformation_clamp_count == 0`；
- pressure/CG/PCG breakdown count 为零，所有 pressure、divergence、no-slip、
  CG/PCG residual finite；
- 两组每一步均 accepted，history 长度严格等于请求步数。

Adaptive 可以在稳定性诊断中发生一次明确的 typed fail-closed retry，但任何
`solid_rejected_trial_count > 0` 的运行都不得晋级为性能 baseline；正式 FSI8
性能晋级要求两组均 zero retry。

FSI2 还必须逐项满足：

- `fluid_projection_consumed_feedback == [False, True]`；
- consumed count 为 `1`；
- 第二步 feedback marker count 大于零；
- mode 为 `hibm_sharp_reconstructed_rows`；
- `hibm_observer_topology_refreshed == True`；
- valid marker 大于零、invalid marker 为零；
- projected/no-slip residual finite；
- 两步均 accepted，无 pressure/PCG breakdown。

## 5. 预先锁定的数值门槛

以下 absolute floor 只处理近零比较，不替代相对门槛。

| history/final 字段 | FSI1/FSI2 每步门槛 | FSI8 每步门槛 | FSI8 NRMSE |
|---|---:|---:|---:|
| `tip_mean_displacement_m`, `max_displacement_m` | `r=0.02, A=2e-8 m, S=1e-6 m` | `r=0.05, A=5e-7 m, S=1e-5 m` | `<=0.03` |
| `mpm_max_speed_mps`, primary/secondary mean velocity | `r=0.02, A=1e-4 m/s, S=0.01 m/s` | `r=0.05, A=1e-3 m/s, S=0.05 m/s` | `<=0.03` |
| `total_marker_force_n`, `marker_force_z_N` | `r=0.05, A=2e-5 N, S=1e-4 N` | `r=0.075, A=1e-4 N, S=1e-3 N` | `<=0.05` |
| `max_abs_traction_pa` | `r=0.05, A=0.1 Pa, S=1 Pa` | `r=0.075, A=5 Pa, S=10 Pa` | `<=0.05` |
| `local_velocity_peak_mps`, p99, p999 | `r=0.02, A=0.05 m/s, S=1 m/s` | `r=0.03, A=0.1 m/s, S=1 m/s` | `<=0.02` |
| final exported velocity peak | `r=0.02, A=0.05 m/s, S=1 m/s` | `r=0.03, A=0.1 m/s, S=1 m/s` | final only |
| `pressure_min_pa`, `pressure_max_pa`, pressure range | `r=0.03, A=5 Pa, S=100 Pa` | `r=0.05, A=10 Pa, S=100 Pa` | `<=0.03` |

final/step NPZ 必须使用正式 cell-center 坐标、正式物理 solid mask、非空且内部一致的
fluid/boundary/display masks，以及精确的 `static_gauge_pressure_pa` / `outlet_0_pa`
语义；summary 的 shape、mask counts、坐标 extrema 和 speed peak 必须与 NPZ 闭合。
final 的数值网格字段必须精确为 `float64`、masks 为 `bool`，并在整个网格上 finite；
step 的 solid/marker 浮点字段必须为 `float32`、marker/grid index 字段为 `int32`、masks
为 `bool`。step NPZ 不允许额外或重复字段，且 stage 必须精确为
`flow_solution_stage=boundary_topology_stage=pre_solid_projection`、
`structure_geometry_stage=post_solid_observer`、同步标志为 scalar `True`。每个运行最后
一个 step frame 与 final NPZ 的 13 个 parity fields 必须逐元素精确闭合，防止混入
stale final artifact。
`solid_mask`、`s/y` 和 pressure metadata 是成对静态身份，必须逐元素相同；其余 masks
来自各自 accepted obstacle/Dirichlet 状态，允许因容差内轨迹差异而不同，不做成对
逐元素相等。final exported velocity peak 固定在两组 `fluid_mask` 交集上测量，并报告
每类动态 mask mismatch count；交集为空时 fail closed。

对“越小越好”的 residual，除两组都必须通过现有 formal health gate 外，adaptive
还必须满足：

| residual | FSI1/FSI2 adaptive 上限 | FSI8 adaptive 上限 |
|---|---:|---:|
| `flow_projection_l2`, `flow_projection_max_abs` | `max(1.25 * fixed, 1e-4 s^-1)` | `max(1.50 * fixed, 1e-4 s^-1)` |
| `flow_projection_cg_relative_residual_max` | `max(1.25 * fixed, 1e-5)` | `max(1.50 * fixed, 1e-5)` |
| `hibm_no_slip_max_residual_mps` | `max(1.25 * fixed, 1e-5 m/s)` | `max(1.50 * fixed, 1e-5 m/s)` |
| `no_slip_projected_residual_after_projection_mps` | `max(1.25 * fixed, 1e-5 m/s)` | `max(1.50 * fixed, 1e-5 m/s)` |

FSI1/2 的位移、固体速度和流体速度 NRMSE 各不得超过 `0.02`，界面力和 traction
不得超过 `0.05`。短序列仍必须逐步通过上表，不能只比较最后一步。

Fluent reference 只作为外部误差报告，不用来放宽 fixed/adaptive 内部 A/B 门槛。
FSI8 必须同时报告相对 Fluent 的位移和速度误差，但不得声称 parity。当前已知
native Fluent field 最大速度约 `30.4267 m/s`，case monitor reference 为
`28.1 m/s`；两者的测量定义必须分列，不能混用。

## 6. 性能晋级门槛

FSI1 和 FSI2 用于正确性、feedback 与 cache warm-up，不据此声称加速。FSI8
在两组都通过第 4、5 节后，才检查：

- fixed 实际 solid step-kernel launch 为 `1600 * 8 = 12800`；
- adaptive `solid_substeps_executed_total` 总和必须小于 `12800`；该字段与
  `_step_kernel` 的实际调用一一对应；
- 两组 selector scalar read、packed report read、retry 和 host/device transfer
  计数必须报告；缺少全路径计数时明确写成未测，不得估算成实际值；
- synchronized `solid_wall_time_s` speedup `fixed/adaptive >= 1.05x`；
- adaptive 的 `pre_summary_artifact_elapsed_s <= 1.05 * fixed`；
- `elapsed_s` 是兼容旧报告的 setup + solver 边界，不能替代正式 pre-summary
  artifact 边界；
- `solver_elapsed_s` 只覆盖 solver call（其中包含逐步 observer callback），
  `post_solver_artifact_export_wall_time_s` 覆盖 solver 返回后的 history、compact、
  final NPZ 和 step-artifact validation；`pre_summary_artifact_elapsed_s` 从 setup
  开始，到 summary 发布前上述正式 solver artifacts 已验证为止；summary 发布后
  重新采样的 terminal elapsed 只写入 `progress.json.elapsed_s`，两者不得混名；
- `flow_wall_time_s` 是每个完整 flow macro call 的同步计时；HIBM 的
  pre-predictor、projection-cycle 和 post-solid-observer 三桶是它及后续 observer
  路径的细分，其中 projection-cycle 只表示 HIBM boundary assembly，不含
  pressure/CG/PCG wall time；三桶不得再与 whole-flow 相加冒充总耗时；
- `snapshot_capture_wall_time_s` 只计 device/host snapshot capture，
  `step_artifact_export_wall_time_s` 只计 callback serialization。callback 返回后才能
  得到自身耗时，所以正式比较读取最终 compact/report history；callback 当场写出的
  单步 history JSON 中该字段仍是调用前的 `0`，不得作为该指标来源；
- final summary/report 中的 profile totals 必须与逐步 history 精确求和闭合；
- flow、HIBM、solid、snapshot/export 时间分列，不能把 pytest 时间或
  JIT/cache 首次编译时间算成仿真加速；
- 所有性能结论使用同一 offline cache 配置下的成对正式运行；首次编译是否仍落入
  `solver_elapsed_s` 必须原样披露，不能事后从单组中扣除；
- adaptive 的 total `elapsed_s <= 1.05 * fixed elapsed_s` 是旧兼容字段的附加
  sanity check，不是正式 total speedup 定义；
- 两组都使用 `--profile-wall-time --save-step-fields`，因此该组 wall time 只与
  同配置诊断 run 比较，不外推为未 profiling 的 production wall time。

若精度通过但 wall-time 门槛不通过，自适应数值实现可以保留为稳定基线，但不得
报告性能胜出；应进入 profile-first 的性能优化阶段。

## 7. fresh snapshot 与 run identity

已消耗失败的 identity 是 `r01`，其 dry-run、preflow failure 和 cache 都必须保留，
不得删除或复用。源码已经改变，下一轮首选 identity：

```text
ansys_vf__solid_substep_ab__20260825__r02
```

输出目录：

```text
validation_runs/ansys_vertical_flap_fsi/
  our_solver_vs_native_fluent_fine_2026-07-10/runs/
    ansys_vf__solid_substep_ab__20260825__r02__preflow_dryrun/
    ansys_vf__solid_substep_ab__20260825__r02__preflow/
    ansys_vf__solid_substep_ab__20260825__r02__fixed1600__fsi01/
    ansys_vf__solid_substep_ab__20260825__r02__adaptive__fsi01/
    ansys_vf__solid_substep_ab__20260825__r02__fixed1600__fsi02/
    ansys_vf__solid_substep_ab__20260825__r02__adaptive__fsi02/
    ansys_vf__solid_substep_ab__20260825__r02__fixed1600__fsi08/
    ansys_vf__solid_substep_ab__20260825__r02__adaptive__fsi08/
    ansys_vf__solid_substep_ab__20260825__r02__comparison__fsi01/
    ansys_vf__solid_substep_ab__20260825__r02__comparison__fsi02/
    ansys_vf__solid_substep_ab__20260825__r02__comparison__fsi08/
validation_runs/solver_soaks/
  ansys_vf__solid_substep_ab__20260825__r02__snapshot/preflow_state
validation_runs/.taichi_cache/
  ansys_vf_solid_substep_ab_20260825_r02/
```

首次执行 r02 的任何命令前，如果上述任一目标已存在，禁止删除或复用，
整体递增到 `r03`。dry-run 成功后，它自己的 output 和共享 cache 已存在是本次
campaign 的预期状态；同一冻结源码、同一 runtime 配置下后续命令允许复用该 cache，
但其余尚未运行的 output/snapshot 目标仍必须不存在。

## 8. 经过当前 parser 核对的命令

从权威 WSL UNC 根目录运行：

```powershell
$repo = '\\wsl.localhost\Ubuntu-22.04\home\zhuohengli\worktrees\HIBM-MPM-v46e2-main-integration'
$python = 'D:\working\taichi\env\python.exe'
$script = 'validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\scripts\run_our_solver_vertical_flap.py'
$runs = 'validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs'
$identity = 'ansys_vf__solid_substep_ab__20260825__r02'
$snapshot = "validation_runs\solver_soaks\${identity}__snapshot\preflow_state"
$cache = 'validation_runs\.taichi_cache\ansys_vf_solid_substep_ab_20260825_r02'
Set-Location -LiteralPath $repo

$snapshotRoot = Split-Path -Parent $snapshot
$reservedTargets = @(
  "$runs\${identity}__preflow_dryrun",
  "$runs\${identity}__preflow",
  "$runs\${identity}__fixed1600__fsi01",
  "$runs\${identity}__adaptive__fsi01",
  "$runs\${identity}__fixed1600__fsi02",
  "$runs\${identity}__adaptive__fsi02",
  "$runs\${identity}__fixed1600__fsi08",
  "$runs\${identity}__adaptive__fsi08",
  "$runs\${identity}__comparison__fsi01",
  "$runs\${identity}__comparison__fsi02",
  "$runs\${identity}__comparison__fsi08",
  $snapshotRoot,
  $cache
)
$existingTargets = @($reservedTargets | Where-Object {
  Test-Path -LiteralPath $_
})
if ($existingTargets.Count -ne 0) {
  throw "r02 target already exists; increment the whole campaign identity: $($existingTargets -join ', ')"
}

function Assert-JsonEqual {
  param(
    [Parameter(Mandatory)]$Actual,
    [Parameter(Mandatory)]$Expected,
    [Parameter(Mandatory)][string]$Label
  )
  $actualJson = ConvertTo-Json $Actual -Depth 64 -Compress
  $expectedJson = ConvertTo-Json $Expected -Depth 64 -Compress
  if ($actualJson -cne $expectedJson) {
    throw "$Label mismatch: actual=$actualJson expected=$expectedJson"
  }
}

$physicsArgs = @(
  '--preflow-steps', '200',
  '--preflow-convergence-mode', 'windowed_stationary',
  '--preflow-stationary-min-steps', '20',
  '--preflow-stationary-window-steps', '10',
  '--preflow-stationary-consecutive-windows', '3',
  '--preflow-stationary-tolerance', '0.01',
  '--preflow-stationary-divergence-tolerance', '0.05',
  '--preflow-stationary-no-slip-tolerance-fraction', '0.05',
  '--grid-nodes', '4', '256', '320',
  '--solid-particle-counts', '1', '256', '20',
  '--marker-count', '64',
  '--flow-projection-iterations', '1080',
  '--flow-post-dirichlet-consistency-projections', '1',
  '--flow-cg-preconditioner', 'fv_multigrid',
  '--flow-pressure-solve-failure-policy', 'raise',
  '--flow-report-percentiles',
  '--hibm-search-radius-m', '0.0017',
  '--hibm-search-radius-xyz-m', '0.0012', '0.000390625', '0.00046875',
  '--young-modulus-pa', '1000000',
  '--taichi-offline-cache-dir', $cache
)

$dryRunArgs = @(
  '--output-dir', "$runs\${identity}__preflow_dryrun",
  '--run-label', "${identity}__preflow_dryrun",
  '--steps', '0',
  '--preflow-snapshot-out', $snapshot,
  '--profile-wall-time',
  '--save-step-fields',
  '--dry-run'
) + $physicsArgs
& $python $script @dryRunArgs
if ($LASTEXITCODE -ne 0) {
  throw "dry-run failed with exit code $LASTEXITCODE"
}

$dryOutput = "$runs\${identity}__preflow_dryrun"
$dryManifest = Get-Content -Raw -LiteralPath "$dryOutput\run_manifest.json" |
  ConvertFrom-Json
$dryConfig = Get-Content -Raw -LiteralPath "$dryOutput\our_solver_config.json" |
  ConvertFrom-Json
Assert-JsonEqual $dryManifest.config $dryConfig 'dry-run manifest/config'
if (
  $dryManifest.dry_run -ne $true -or
  $dryManifest.profile_wall_time -ne $true -or
  $dryManifest.save_step_fields -ne $true
) {
  throw 'dry-run manifest mode flags mismatch'
}
$lockedDryConfig = [ordered]@{
  duct_length_m = 0.10
  duct_height_m = 0.04
  flap_height_m = 0.01
  flap_streamwise_min_m = 0.050
  flap_streamwise_max_m = 0.053
  dt_s = 5.0e-4
  step_count = 0
  grid_nodes = @(4, 256, 320)
  solid_particle_counts = @(1, 256, 20)
  marker_count = 64
  preflow_steps = 200
  preflow_convergence_mode = 'windowed_stationary'
  preflow_stationary_min_steps = 20
  preflow_stationary_window_steps = 10
  preflow_stationary_consecutive_windows = 3
  preflow_stationary_tolerance = 0.01
  preflow_stationary_divergence_tolerance = 0.05
  preflow_stationary_no_slip_tolerance_fraction = 0.05
  flow_projection_iterations = 1080
  flow_post_dirichlet_consistency_projection_iterations = 1
  flow_cg_preconditioner = 'fv_multigrid'
  flow_pressure_solve_failure_policy = 'raise'
  flow_solid_boundary_mode = 'hibm_sharp_marker_rows'
  flow_report_include_percentiles = $true
  flow_hibm_sharp_search_radius_m = 0.0017
  flow_hibm_sharp_search_radius_xyz_m = @(0.0012, 0.000390625, 0.00046875)
  young_modulus_pa = 1000000.0
  solid_substeps = $null
}
foreach ($field in $lockedDryConfig.Keys) {
  if ($null -eq $dryConfig.PSObject.Properties[$field]) {
    throw "dry-run config is missing $field"
  }
  Assert-JsonEqual $dryConfig.$field $lockedDryConfig[$field] "dry-run config.$field"
}
$runtime = $dryManifest.taichi_runtime
if (
  $runtime.requested_arch -cne 'cuda' -or
  $runtime.default_fp -cne 'f32' -or
  $runtime.random_seed -ne 0 -or
  $runtime.strict_arch -ne $true -or
  $runtime.offline_cache_enabled -ne $true -or
  (Resolve-Path -LiteralPath $runtime.offline_cache_file_path).Path -cne
    (Resolve-Path -LiteralPath $cache).Path
) {
  throw 'dry-run Taichi runtime/cache request mismatch'
}
$requiredSources = @(
  'cases/ansys_vertical_flap_fsi.py',
  'benchmarks/official/solid_mpm_fsi_runner.py',
  'simulation_core/fluids/solver.py',
  'simulation_core/solids/neo_hookean_mpm.py',
  'tools/validation/compare_solid_substep_ab.py',
  'validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/scripts/run_our_solver_vertical_flap.py'
)
foreach ($source in $requiredSources) {
  $digest = $dryManifest.source_sha256.$source
  if ($digest -cnotmatch '^[0-9a-f]{64}$') {
    throw "dry-run source hash mismatch: $source"
  }
}
if (Test-Path -LiteralPath $snapshotRoot) {
  throw 'dry-run unexpectedly created the preflow snapshot target'
}

$preflowArgs = @(
  '--output-dir', "$runs\${identity}__preflow",
  '--run-label', "${identity}__preflow",
  '--steps', '0',
  '--preflow-snapshot-out', $snapshot
) + $physicsArgs
& $python $script @preflowArgs
if ($LASTEXITCODE -ne 0) {
  throw "production preflow failed with exit code $LASTEXITCODE"
}

$preflowOutput = "$runs\${identity}__preflow"
$preflowManifest = Get-Content -Raw -LiteralPath "$preflowOutput\run_manifest.json" |
  ConvertFrom-Json
Assert-JsonEqual $preflowManifest.config $dryManifest.config 'preflow/dry-run config'
Assert-JsonEqual $preflowManifest.source_sha256 $dryManifest.source_sha256 'preflow/dry-run source hashes'
Assert-JsonEqual $preflowManifest.taichi_runtime $dryManifest.taichi_runtime 'preflow/dry-run Taichi runtime request'

$snapshotValidator = @'
import json
import re
import sys
from pathlib import Path

from simulation_core.fluids.preflow_snapshot import (
    PreflowSnapshotIdentity,
    load_preflow_snapshot,
)

base = Path(sys.argv[1])
preflow_output = Path(sys.argv[2])
manifest_path = base.with_suffix(".json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
identity = manifest.get("identity")
if not isinstance(identity, dict) or set(identity) != {
    "config_sha256", "source_sha256", "geometry_sha256"
}:
    raise SystemExit("snapshot identity schema mismatch")
if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in identity.values()):
    raise SystemExit("snapshot identity digest mismatch")
npz_name = manifest.get("npz_file")
if not isinstance(npz_name, str) or re.fullmatch(r"preflow_state\.[0-9a-f]{32}\.npz", npz_name) is None:
    raise SystemExit("snapshot generation name mismatch")
generation = manifest_path.parent / npz_name
if not generation.is_file() or list(manifest_path.parent.glob("preflow_state.*.npz")) != [generation]:
    raise SystemExit("snapshot generation set mismatch")
compact = json.loads((preflow_output / "our_solver_report_compact.json").read_text(encoding="utf-8"))
if compact.get("preflow_snapshot_identity") != identity:
    raise SystemExit("snapshot/report identity mismatch")
load_preflow_snapshot(
    base,
    expected_identity=PreflowSnapshotIdentity(**identity),
    expected_velocity_dirichlet_boundary_authority=manifest.get(
        "velocity_dirichlet_boundary_authority"
    ),
)
print(json.dumps({"status": "passed", "identity": identity, "npz_file": npz_name}, sort_keys=True))
'@
& $python -c $snapshotValidator $snapshot $preflowOutput
if ($LASTEXITCODE -ne 0) {
  throw "preflow snapshot validation failed with exit code $LASTEXITCODE"
}
```

上述命令已把 dry-run manifest/config/runtime/source 审计、退出码检查，以及 production
preflow 的 snapshot manifest、唯一 generation NPZ、SHA256/字段完整性和
geometry/config/source identity 闭合写成可执行 fail-closed 门槛。任一命令抛错即停止，
不得继续运行 FSI。

正式比较器还会自动重新计算当前 launcher source surface（其中明确包含比较器本身），
定位同 identity 的 sibling `__preflow` producer，并验证 producer/fixed/adaptive 的
source/runtime、runner 定义的 FSI-only config 投影、snapshot I/O 路径方向、snapshot
manifest/generation/hash、run label/directory、summary output path 和 final NPZ path。
它同时验证 dual-face `64 x 2 = 128` marker report/array closure、canonical CSV 字段集合
与逐字段内容、exact NPZ schema/dtype/stage、全网格 finite，以及同一运行最后 step 与 final 的
13-field parity closure。这些检查不是可选的人工审阅项；任一失败都必须使 comparison
非零退出并停止晋级。

```powershell
function Invoke-SolidSubstepAb {
  param(
    [Parameter(Mandatory)][string]$Mode,
    [Parameter(Mandatory)][int]$Steps,
    [Nullable[int]]$FixedSubsteps
  )

  $label = "${identity}__${Mode}"
  $output = "$runs\$label"
  if (Test-Path -LiteralPath $output) {
    throw "Refusing to reuse output directory: $output"
  }
  $solidArgs = if ($null -eq $FixedSubsteps) {
    @()
  } else {
    @('--solid-substeps', "$FixedSubsteps")
  }
  $runArgs = @(
    '--output-dir', $output,
    '--run-label', $label,
    '--steps', "$Steps",
    '--preflow-snapshot-in', $snapshot,
    '--profile-wall-time',
    '--span-reduction', 'mean',
    '--streamwise-velocity-sign', '-1.0',
    '--save-step-fields'
  ) + $physicsArgs + $solidArgs
  & $python $script @runArgs
  if ($LASTEXITCODE -ne 0) {
    throw "$label failed with exit code $LASTEXITCODE"
  }
}
```

再定义冻结的比较器入口；comparison output 也必须是全新目录：

```powershell
$compareScript = 'tools\validation\compare_solid_substep_ab.py'
function Compare-SolidSubstepAb {
  param([Parameter(Mandatory)][string]$Gate)

  $fixed = "$runs\${identity}__fixed1600__${Gate}"
  $adaptive = "$runs\${identity}__adaptive__${Gate}"
  $comparison = "$runs\${identity}__comparison__${Gate}"
  if (Test-Path -LiteralPath $comparison) {
    throw "Refusing to reuse comparison directory: $comparison"
  }
  & $python $compareScript `
    --fixed-dir $fixed `
    --adaptive-dir $adaptive `
    --output-dir $comparison
  if ($LASTEXITCODE -ne 0) {
    throw "${Gate} comparison failed with exit code $LASTEXITCODE"
  }
}

```

最后严格按下列顺序逐个执行，每一对 comparison 通过后才进入下一对：

```powershell
Invoke-SolidSubstepAb 'fixed1600__fsi01' 1 1600
Invoke-SolidSubstepAb 'adaptive__fsi01' 1 $null
Compare-SolidSubstepAb 'fsi01'

Invoke-SolidSubstepAb 'fixed1600__fsi02' 2 1600
Invoke-SolidSubstepAb 'adaptive__fsi02' 2 $null
Compare-SolidSubstepAb 'fsi02'

Invoke-SolidSubstepAb 'fixed1600__fsi08' 8 1600
Invoke-SolidSubstepAb 'adaptive__fsi08' 8 $null
Compare-SolidSubstepAb 'fsi08'
```

比较器必须从最终 compact history 读取 callback serialization 时间；逐步 callback
当场写出的 history JSON 中该字段仍为调用前的 `0`。除这一个有文档说明的字段外，
逐步 history 必须与最终 compact history 闭合。

当前正式 parser 不再提供 `--pressure-pair-provider-mode` 或
`--disable-hibm-interpolate-velocity-rows`；旧 handoff 中含这些参数的命令不得复用。
`flow_predictor_substeps` 也不做 diagnostic override，使用当前 production 默认 `1`。

## 9. 晋级与失败处理

- FSI1 任一精确或数值门槛失败：停止，不运行 FSI2；
- FSI2 feedback 或 A/B 门槛失败：停止，不运行 FSI8；
- FSI8 精度失败：修改通用 adaptive controller，不恢复固定 1600 production 默认；
- FSI8 只在精度、稳定性和 residual 全通过后评估性能；
- 任何源码修改都会使当前 snapshot 失效，必须使用新 identity 生成 fresh snapshot；
- 所有结果必须保留原始 manifest、config、history、compact report、summary、
  step fields 和 comparison report；
- 通过 FSI8 仍不等于 strict FSI50、Fluent parity 或发布就绪。
