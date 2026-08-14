---
status: in-progress
branch: codex/fix-main-audit-2026-08-10
timestamp: 2026-08-14T14:38:05+09:00
stop_reason: user-requested pause after writing this handoff
files_modified_in_this_slice:
  - benchmarks/official/solid_mpm_fsi_runner.py
  - cases/ansys_vertical_flap_fsi.py
  - simulation_core/coupling/hibm_mpm/core.py
  - simulation_core/coupling/hibm_mpm/marker_mac_constraint.py
  - simulation_core/coupling/hibm_mpm/marker_target_closure.py
  - tests/benchmarks/test_canonical_production_runner_boundary_ledger.py
  - tests/cases/test_ansys_vertical_flap_fsi.py
  - tests/contracts/test_unified_fsi_solver_core.py
  - tests/integration/test_ansys_vertical_flap_preflow_snapshot.py
  - tests/solvers/_hibm_component_face_ledger_contracts.py
  - tests/solvers/test_hibm_marker_mac_pcg_work_elision_contract.py
  - tests/solvers/test_marker_target_closure.py
artifacts_created:
  - validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflight/unified_core_preflow200_20260814_r07_dryrun
  - validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflight/unified_core_preflow200_20260814_r07_production_dryrun
  - validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflight/unified_core_preflow200_20260814_r07_exact_dryrun
---

# ANSYS vertical-flap 50-step closure/performance handoff

## Stop boundary

Work stopped at the user's request immediately after starting the final broad case
regression. The regression process was terminated, its reviewer agent was closed,
and a final process check returned `NO_PYTHON_PROCESSES`.

Do not treat the interrupted broad regression as a pass. It did not produce the
requested JUnit XML. No r07 production preflow, r07 snapshot, r07 formal 50-step
run, or new Fluent comparison was started.

The worktree was already intentionally very dirty before this slice. Preserve all
unrelated modified and untracked files. Do not run `git clean`, `git reset`, or
bulk-delete validation artifacts.

## User goal and current verdict

The required end state is:

1. all ANSYS vertical-flap FSI steps use the common generic FSI solver;
2. run a fresh, source-matched fixed-solid preflow;
3. complete 50 accepted FSI physical steps;
4. compare the completed run with the locked native Fluent 50-step reference;
5. every user-required comparison error must be no greater than 10%.

That goal is not complete. The latest formal run accepted 8 steps and failed at
the initial trial of step 9. The closure/performance implementation below has
focused evidence, but it has not yet been exercised by a fresh production preflow
or formal 50-step run.

## Last production evidence before the current fix

### Strict preflow r06

The last completed strict preflow is:

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/our_solver/unified_core_preflow200_20260814_r06
```

Its snapshot prefix is:

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/preflow_snapshots/unified_core_preflow200_20260814_r06/preflow_state
```

It reached the windowed-stationary gate after 75 of the allowed 200 preflow
windows/steps and took approximately `2778.99 s`. It is valid evidence for the
source that created it, not for the current source after the closure changes.

### Formal 50-step r01 from r06

The failed run is:

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/unified_core_fsi50_from_preflow_r06_20260814_r01
```

Its `progress.json` records:

```text
status=failed
step_completed=8
time_s=0.004
elapsed_s=5279.468408500019
```

The step-9 initial trial failed with:

```text
HIBM-owned hard target marker compatibility closure did not converge before
canonical commit: adjustable_residual_mps=0.0003476178681012243,
closure_tolerance_mps=1e-06, invalid_count=0, failure_code=0
```

The eight `step_fields/step_000N.npz` files are reduced parity/output fields. They
are not complete fluid/solid/HIBM transaction checkpoints. There is no supported
way to resume the current solver from accepted step 8, so the next formal run must
start from a source-matched preflow snapshot.

## Why it was slow

The slow runtime had both real physics cost and avoidable code cost.

- Production uses `4 x 256 x 320 = 327,680` cells. A coupled physical step has
  hundreds of fluid/SST substeps, `1600` MPM substeps per trial, and normally
  three generic-solver coupling trials. This part is genuine work.
- The old hard-target closure always dispatched its full `4096` serialized
  Kaczmarz sweeps. Every accepted report showed `4096`, even when additional
  sweeps were no longer useful.
- Marker-MAC PCG could report convergence after one iteration while the Python
  host still launched the configured loop up to `4096`; device guards skipped
  arithmetic but did not avoid thousands of kernel dispatches.
- Cold Taichi compilation is independently expensive. `core.py` contains large
  kernels; a small CUDA ledger test took 13-18 minutes cold and about 1.5 minutes
  warm. A production process pays the corresponding cold specialization cost
  once, so do not run competing Taichi jobs in parallel.
- Deformation also increases real canonical-ledger work. In the failed run,
  reconstructed components rose roughly from 259 to 615, duplicate claims from
  569 to 1041, active components from 3103 to 3827, rows from 1729 to 2006, and
  relocations from 36 to 260. A step-8 canonical ledger build was about `68.73 s`.

## Current implementation

### One weighted direct hard-target closure

`simulation_core/coupling/hibm_mpm/marker_target_closure.py` now owns a small pure
NumPy solve for

```text
A * delta = residual
```

using the inverse-mass weighted minimum norm. The solve is f64, but its numerical
rank cutoff is based on f32 source precision because the Taichi operator fields
were assembled in f32. It fails closed when the row system is incompatible.

`core.py` now performs:

1. one device measurement of active marker rows and their exact support;
2. construction of the compact host matrix;
3. one weighted least-squares solve when needed;
4. f32 materialization precheck;
5. packed gather/scatter of only adjustable component DOFs;
6. a final device measurement before the canonical target is accepted.

The old Kaczmarz kernel, iteration argument, and compatibility report fields were
deleted. There is no old closure fallback.

The report now uses:

```text
solver=weighted_minimum_norm_lstsq
solve_count
matrix_rank
adjustable_dof_count
least_squares_max_residual_mps
materialized_max_residual_mps
max_abs_correction_mps
```

The canonical parent report schema is now `5`. The official runner validates the
new report exactly and rejects old/mixed schemas.

### Marker-MAC PCG early stop

`MarkerMacConstraint.solve_device()` now gives an initially converged system a
zero iteration budget. Otherwise it checks device convergence/failure every eight
iterations and exits the host loop. This preserves the common solver while
removing post-convergence dispatches.

The ANSYS case no longer overrides the common budget to `4096` and now uses the
existing case/default value `64`. There is no ANSYS-only compatibility path.

### Relevant current source locations

```text
simulation_core/coupling/hibm_mpm/marker_target_closure.py:24
simulation_core/coupling/hibm_mpm/core.py:23438
simulation_core/coupling/hibm_mpm/core.py:23510
simulation_core/coupling/hibm_mpm/core.py:23597
simulation_core/coupling/hibm_mpm/marker_mac_constraint.py:3137
benchmarks/official/solid_mpm_fsi_runner.py:5213
cases/ansys_vertical_flap_fsi.py:269
```

## Review status

The first independent review found six concrete issues:

1. producer/runner report schema mismatch;
2. f64 rank threshold was too optimistic for f32 source data;
3. closure candidate transfer copied full-grid fields to/from the host;
4. the ANSYS failure budget still allowed 4096 PCG iterations;
5. the report mislabeled the post-f32 residual as the least-squares residual;
6. the inverse-mass weighting direction lacked a direct test.

All six were addressed. The follow-up reviewer was still running when the user
requested a pause and was deliberately closed. Therefore the current diff does
not have a completed post-fix independent review.

## Validation evidence and exact boundary

Completed evidence observed in this work session:

- pure weighted-closure tests: `4 passed`;
- packed CUDA candidate gather/scatter: `1 passed in 15.72s`;
- successful CUDA core producer through runner health validation:
  `1 passed in 1073.67s` on a cold compile;
- immutable-row rollback: `1 passed in 33.50s` warm;
- initially converged Marker-MAC with `max_iterations=4096`:
  `1 passed in 181.25s` cold and reported zero iterations;
- nonzero exact/transactional Marker-MAC path: `1 passed in 29.11s`;
- focused host/report group: `68 passed, 338 subtests`;
- preflow snapshot integration group: `122 passed`;
- an earlier focused group: `24 passed, 203 subtests`;
- selected ANSYS config/tip-cap group: `4 passed`;
- `py_compile` and focused Ruff F/E9 checks passed before the last test-fixture-only
  edit.

The broad case gate is not green yet:

- a combined run first found an old fixture still publishing the removed
  `iterations` closure field;
- that fixture was updated and its individual test then passed;
- one full case rerun completed but its output was truncated, so it was not
  counted;
- a second full case rerun was interrupted for this handoff and produced no
  JUnit XML.

Do not claim fresh production performance, 50-step completion, or Fluent parity
from the focused evidence above.

## Current file hashes

These hashes identify the exact relevant files at the pause boundary:

```text
F4F4E2AE2E29166CD87468E476C806386E79256DC605730EA0A44D25E4F8D536  simulation_core/coupling/hibm_mpm/marker_target_closure.py
D3F8825CE7638112C023FB570ECB59CD6FD61674485712C7AFB037BC1CB944D3  simulation_core/coupling/hibm_mpm/core.py
BF9849C22067E108839E544452C0E461168CEC1A3125D04403EF348151E7B127  simulation_core/coupling/hibm_mpm/marker_mac_constraint.py
20BE8D1004E1363513CE29084B35C56CC16DB8FEEF0FA35BD9344F955414DFE9  benchmarks/official/solid_mpm_fsi_runner.py
089189394C673C63C72027233083D525D25F074EF9CF9403A7D3B5F4C0C441BF  cases/ansys_vertical_flap_fsi.py
```

## Exact continuation order

Use the trusted interpreter:

```powershell
$python = 'D:\working\taichi\env\python.exe'
```

### 1. Re-establish the pause boundary

```powershell
git status --short
Get-Process python -ErrorAction SilentlyContinue
```

Preserve the dirty tree. Confirm that no production run was started after this
handoff.

### 2. Finish the interrupted gates in separate commands

Run small groups separately so one truncated output cannot erase the result:

```powershell
& $python -m pytest -q tests\solvers\test_marker_target_closure.py -p no:cacheprovider
& $python -m pytest -q tests\solvers\test_hibm_marker_mac_pcg_work_elision_contract.py -p no:cacheprovider
& $python -m pytest -q tests\cases\test_ansys_vertical_flap_fsi.py -p no:cacheprovider
& $python -m pytest -q tests\benchmarks\test_vertical_flap_sst_runner_contract.py -p no:cacheprovider
& $python -m pytest -q tests\integration\test_ansys_vertical_flap_step50_cli.py -p no:cacheprovider
& $python -m pytest -q tests\contracts\test_unified_fsi_solver_core.py -p no:cacheprovider
```

Then rerun syntax/static checks for the exact current files. If a Taichi test is
cold, allow for compilation and monitor CPU instead of assuming a hang.

Request one final read-only review of the focused closure/PCG diff. Address only
real correctness/performance findings; do not broaden into splitting every giant
Taichi kernel before the production run.

### 3. Launch the fresh strict preflow r07

The three planned target directories did not exist at the pause boundary. The
exact dry run already records `grid=4,256,320`, particles `1,256,20`, 64 markers,
1080 projection iterations, 1600 solid substeps, and Marker-MAC PCG budget 64.

Do not pass `--hibm-interior-probe-distance-m`. Passing that scalar erases the
required anisotropic probe tuple. The command below reproduces the r06 physical
configuration while using the current source:

```powershell
& $python 'validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\scripts\run_our_solver_vertical_flap.py' `
  --output-dir 'validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\our_solver\unified_core_preflow200_20260814_r07' `
  --run-label 'unified_core_preflow200_20260814_r07' `
  --steps 0 `
  --preflow-steps 200 `
  --preflow-convergence-mode windowed_stationary `
  --preflow-stationary-min-steps 20 `
  --preflow-stationary-window-steps 10 `
  --preflow-stationary-consecutive-windows 3 `
  --preflow-stationary-tolerance 0.01 `
  --preflow-stationary-divergence-tolerance 0.05 `
  --preflow-stationary-no-slip-tolerance-fraction 0.05 `
  --preflow-snapshot-out 'validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\preflow_snapshots\unified_core_preflow200_20260814_r07\preflow_state' `
  --grid-nodes 4 256 320 `
  --solid-particle-counts 1 256 20 `
  --marker-count 64 `
  --flow-projection-iterations 1080 `
  --flow-post-dirichlet-consistency-projections 3 `
  --flow-cg-preconditioner fv_multigrid `
  --flow-pressure-solve-failure-policy raise `
  --solid-substeps 1600 `
  --flow-predictor-substeps 1 `
  --hibm-search-radius-m 0.0017 `
  --disable-hibm-interpolate-velocity-rows `
  --hibm-search-radius-xyz-m 0.0012 0.000390625 0.00046875 `
  --young-modulus-pa 1000000 `
  --pressure-pair-provider-mode runtime_anchored_cell_pair `
  --taichi-offline-cache-dir 'validation_runs\.taichi_cache\ansys_vertical_flap_cuda_f32'
```

Do not run another Taichi job concurrently. Monitor the r07 `progress.json` and
process CPU at intervals. A process using CPU with slowly changing progress is not
necessarily stuck because the progress file advances only at reporting boundaries.

### 4. Validate r07 before formal FSI

Require all of the following:

- run status is completed, not merely a created directory;
- windowed stationary gate passed;
- snapshot JSON and atomically named NPZ both exist;
- manifest/source hashes match the current source;
- no pressure, no-slip, topology, or reporting failure is hidden in the compact
  report.

Then run a zero-step strict snapshot-load preflight with the same explicit physical
configuration and `--preflow-snapshot-in` pointing to r07. Do not override snapshot
identity checks. The r06 snapshot may be used only for diagnostics now; it is
source-stale for a formal current-source run.

### 5. Launch the fresh formal 50-step run

After the strict load preflight passes:

```powershell
& $python 'validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\scripts\run_our_solver_vertical_flap.py' `
  --output-dir 'validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs\unified_core_fsi50_from_preflow_r07_20260814_r01' `
  --run-label 'unified_core_fsi50_from_preflow_r07_20260814_r01' `
  --steps 50 `
  --preflow-steps 200 `
  --preflow-convergence-mode windowed_stationary `
  --preflow-stationary-min-steps 20 `
  --preflow-stationary-window-steps 10 `
  --preflow-stationary-consecutive-windows 3 `
  --preflow-stationary-tolerance 0.01 `
  --preflow-stationary-divergence-tolerance 0.05 `
  --preflow-stationary-no-slip-tolerance-fraction 0.05 `
  --preflow-snapshot-in 'validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\preflow_snapshots\unified_core_preflow200_20260814_r07\preflow_state' `
  --grid-nodes 4 256 320 `
  --solid-particle-counts 1 256 20 `
  --marker-count 64 `
  --flow-projection-iterations 1080 `
  --flow-post-dirichlet-consistency-projections 3 `
  --flow-cg-preconditioner fv_multigrid `
  --flow-pressure-solve-failure-policy raise `
  --solid-substeps 1600 `
  --flow-predictor-substeps 1 `
  --hibm-search-radius-m 0.0017 `
  --disable-hibm-interpolate-velocity-rows `
  --hibm-search-radius-xyz-m 0.0012 0.000390625 0.00046875 `
  --young-modulus-pa 1000000 `
  --pressure-pair-provider-mode runtime_anchored_cell_pair `
  --save-step-fields `
  --taichi-offline-cache-dir 'validation_runs\.taichi_cache\ansys_vertical_flap_cuda_f32'
```

The new direct closure should either satisfy the marker rows once or fail quickly
with matrix rank, row, marker, axis, region, and residual evidence. Do not restore
the 4096-sweep compatibility loop if a new incompatibility appears.

### 6. Compare all 50 steps with Fluent

The locked reference run is:

```text
validation_runs/ansys_vertical_flap_fsi/official_fluent_fine_fsi_valid_2026-07-10/runs/fresh50_20260713_104843
```

The comparison entry point is:

```text
validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/scripts/postprocess_our_solver_vs_native_fluent.py
```

It takes `--our-run-dir` and a new `--output-dir`; its default Fluent input is the
locked native reference. Run it only after the formal run reports 50 accepted
steps. Enforce the user's 10% limit across every required reported metric. Do not
drop a metric, change the reference, or relabel a partial run as parity. If any
metric exceeds 10%, diagnose the first physical/numerical divergence before making
another code change.

### 7. Final cleanup only after production evidence

Current documentation still contains stale Kaczmarz wording:

- `docs/MODULE_MAP.md` around line 110;
- `docs/refactoring/UNIFIED_FSI_SOLVER_CORE_GOAL_2026-08-13.md` around line 65.

After the 50-step/parity result is known, update current docs to the weighted direct
closure and early-stop PCG behavior. Preserve historical handoffs as historical
evidence. Then run the `simplify` pass on this focused implementation only.

## Non-negotiable evidence rules

- Focused tests are not a production-run pass.
- Eight accepted steps are not a completed 50-step run.
- A source-stale snapshot is not formal current-source provenance.
- Reduced step NPZs are not restart checkpoints.
- An interrupted or output-truncated test is not green.
- A comparison is successful only if the fresh run completed all 50 steps and all
  required errors are at most 10%.
