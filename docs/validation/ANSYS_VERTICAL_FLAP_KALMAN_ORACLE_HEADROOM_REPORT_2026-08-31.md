# ANSYS Vertical-Flap Kalman Oracle Headroom Report

Date: 2026-08-31
Status: COMPLETE; `PASS_ORACLE_HEADROOM`, alpha response `COMPLETED`

## 1. Decision

The same-step accepted-state oracle materially reduces real coupled work while
preserving the accepted physical result.  All five predeclared Q0/Q3 gates
passed, so the formal R24B classification is:

    PASS_ORACLE_HEADROOM

This is a causal upper bound, not a deployable predictor or production
acceleration result.  The conditional alpha response shows no work reduction
at alpha 0.25, 0.50, or 0.75; only the exact alpha-1 oracle crosses the
discrete IQN work threshold.  R24's `FAIL_NO_KALMAN_PREDICTIVE_VALUE` result
for K0/K1/K2 therefore remains in force, and this report does not authorize
K3, adaptive covariance, GRU, exact50, Fluent, or long-horizon work.

## 2. Scope and frozen source boundary

The work was performed only in the authoritative WSL Ubuntu-22.04 worktree:

    /home/zhuohengli/worktrees/HIBM-MPM-r21-validation

Branch:

    codex/kalman-oracle-headroom-r24b

Starting commit:

    0ab5157ae7451d7af0082379d84356cb6b7fc0ea

R24's `FAIL_NO_KALMAN_PREDICTIVE_VALUE` result is preserved.  R24B asks a
different causal upper-bound question: whether a same-step accepted-state
iteration-zero guess reduces real IQN/CG/fluid/solid work.

No production predictor, controller, IQN, runner, case, fluid, solid, pressure,
closure, no-slip, checkpoint, or rollback source was changed.

## 3. Pre-run findings

### 3.1 Configuration lock

A Q0 dry-run was compared field by field with the r47 fresh-step manifest.
There were exactly two differences:

- `step_count`: r47 fresh prefix 1, R24B exact8 8;
- `fsi_checkpoint_output_path`: r47 local checkpoint destination versus
  `None` for both fresh R24B arms.

All other numerical and physical fields matched, including fine grid, material
particles, markers, adaptive solid substeps, pressure/CG, IQN, tolerances,
closure, geometry, boundary conditions, SST, and preflow settings.

### 3.2 WSL CUDA bridge

The first strict-CUDA initialization failed before field construction because
the WSL dynamic loader did not search `/usr/lib/wsl/lib`.  The zero-step
failure was retained at:

    validation_runs/solver_soaks/
      ansys_vf__oracle_q0__material_fine__20260831__r24b__failed_libcuda

The GPU bridge itself was present.  A minimal strict probe succeeded after
setting `LD_LIBRARY_PATH=/usr/lib/wsl/lib` and reported `Arch.cuda`; no CPU
fallback was used.

### 3.3 Source-matched preflow

The old r47 snapshot was then rejected before an accepted step because its
executable-source identity did not match the current frozen solver dependency
surface:

    stored:   69ce29b3be3379267acb1e7386bf13e15e1806b6c0528c40b5b2cdab192c5495
    expected: a548b0f351ebaf2aaa5270d91ffd8b7b784d8a95db54dbd8f5093dd8a74f09d323

That zero-step failure was retained at:

    validation_runs/solver_soaks/
      ansys_vf__oracle_q0__material_fine__20260831__r24b__failed_snapshot_identity

A fresh fine-grid strict-CUDA preflow completed with exit 0 after 79 steps when
three consecutive stationary windows passed.  It completed no FSI macro step
and remained at physical time 0.  After the first Sol/Ultra review strengthened
the audit source, and a later fail-closed audit bound the executable-preflow
surface and array shapes independently, a new source-map-matched preflow was
generated for the final evidence rather than re-signing any earlier runs.

    root:
      validation_runs/solver_soaks/
        ansys_vf__preflow__material_fine__20260831__r24b_final_contract
    solver elapsed:
      1079.0223679970004 s
    state JSON file SHA256:
      a97a737d3a5ff3bb949a7e37fe4eea15177dbb2b1c918bd5aaaca4ee92abc904
    state NPZ SHA256:
      4a3ad74a0a2e6f690a77d358e2ebba3e7d919b2e20a624d5b6a2800ddd20e932
    config identity:
      16298a97d45eb03639e7f3d1c7f4048386d66bd573af55c7e485ceedabc4783f
    geometry identity:
      eb28d5616a3ac1f270c2ceae286534a387c5176873086cbd5105d06dae568bbb
    source identity:
      a548b0f351ebaf2aaa5270d91ffd8b7b784d8a95db54dbd8f5093dd8a74f09d323

## 4. Evidence implementation

The new Taichi-free audit surface:

- loads exact8 Q0/Q3 manifests, summaries, progress, step fields, histories,
  IQN trial vectors, and shared preflow artifacts fail-closed;
- requires identical source, non-control config, layout, and preflow identity;
- proves every Q3 trial-zero vector is exactly the Q0 same-step accepted marker
  velocity;
- checks full fluid and solid accepted `dt_s`, zero remaining time, pressure,
  coupling, OOB, deformation clamp, retry, no-slip, and closure health;
- records coupling/rejected trials, first residuals, CG, fluid/solid solve and
  substep work, synchronized component walls, accepted-state NRMSE, and
  per-step frame/history SHA values;
- produces deterministic, self- and cross-fingerprinted source, CSV, summary,
  and blend-response artifacts;
- can generate and audit transparent non-deployable alpha 0.25/0.50/0.75
  producer trajectories only after the oracle gate passes.
- follows manifest roots back to the live Q0/Q3 and alpha producer/consumer
  artifacts and recomputes all four final files bottom-up, including producer
  trajectory and consumer identities;
- independently reconstructs the current preflow executable-source identity
  from the runner plus the complete `simulation_core/**/*.py` surface instead
  of trusting a run-declared source list;
- locks actual marker, solid, and flow-field array shapes, requires the frozen
  `1.1e-6` closure tolerance and invalid-axis diagnostic, and requires Q0's
  oracle path to be explicitly null.

The audit implementation is split into a small public facade plus focused
analysis, artifact, contract, integrity, and verification modules.  Every
implementation file remains below the project's 800-line limit without
changing the public API.

## 5. Investigation and verification

The first real alpha producer exposed nine `NaN` values in optional
empty-set coordinate diagnostics inside Q0's full terminal summary.  A RED
test reproduced the failure, and the producer was changed to write only the
finite completion fields that its consumer contract actually requires.
Decision-required values remain fail-closed and finite.  Optional nondecision
NaNs may remain in the original solver summary; they are neither sanitized nor
copied into the strict producer manifest.

The first fresh Sol/Ultra review returned `fix-first` for two P1 evidence
holes: the contract did not absolutely freeze both requested and actual
runtime/config values, and the final verifier trusted bundle-internal SHA
links instead of recomputing from the run roots.  RED/GREEN fixes added the
absolute contract and a dedicated bottom-up verification module.  A subsequent
review found and closed runtime arm/IQN-reuse and relative-path/CWD gaps.  The
new tests also cover synchronized re-sign attacks, consumer tampering, common
configuration drift, and runtime reuse drift.

A final fail-closed test pass first produced seven expected RED failures for
preflow executable-source drift, three forged array shapes, closure-tolerance
drift, a missing invalid-axis field, and a non-null Q0 oracle path.  The
integrity helper and contract changes closed all seven before the final
source-matched campaign was generated.

Final verification:

- focused R24B tests: 36 passed in 6.37 s;
- the predeclared Kalman/predictor/writeback/statistical boundary matrix:
  242 passed in 84.42 s, with one pre-existing NumPy
  `VisibleDeprecationWarning` at `interface_kalman_predictor.py:665`;
- focused plus the accepted-checkpoint, controller restart, preflow snapshot,
  IQN reuse, adaptive-contract, and runner boundary matrix: 388 passed in
  111.44 s;
- `py_compile`: facade, analysis, artifacts, contracts, integrity,
  verification, CLI, and both focused test files passed;
- CLI `--help`: passed without Taichi initialization;
- final artifact verifier: three passes with identical SHA256 output.

An additional exploratory matrix added the broad legacy vertical-flap case and
SST-runner files.  It ended with 474 passed, 1 xfailed, and 11 failed in
325.41 s.  Every failure was confined to
`tests/cases/test_ansys_vertical_flap_fsi.py` or
`tests/benchmarks/test_vertical_flap_sst_runner_contract.py`; their associated
test, case, and production runner files are unchanged relative to the starting
commit.  Those existing branch-drift failures are reported, not counted as a
green R24B gate and not repaired outside this Goal.

These are focused and related gates, not a claim that the entire repository
test suite passed.

The exact final related-test and evidence-verification commands were:

    /usr/bin/python3 -m pytest -q \
      tests/validation/test_kalman_oracle_headroom.py \
      tests/validation/test_kalman_oracle_headroom_fail_closed.py \
      tests/validation/test_our_solver_vertical_flap_runner.py \
      tests/coupling/test_interface_kalman_predictor.py \
      tests/coupling/test_active_kalman_writeback.py \
      tests/benchmarks/test_modified_physics_kalman_contract.py \
      tests/validation/test_kalman_statistical_calibration.py \
      tests/validation/test_kalman_iqn_reuse_fine_contracts.py \
      tests/tools/test_calibrate_iqn_kalman_qr.py

    /usr/bin/python3 -m pytest -q \
      tests/validation/test_kalman_oracle_headroom.py \
      tests/validation/test_kalman_oracle_headroom_fail_closed.py \
      tests/coupling/test_accepted_fsi_checkpoint.py \
      tests/coupling/test_interface_controller_restart.py \
      tests/coupling/test_interface_initial_guess_controller.py \
      tests/integration/test_ansys_vertical_flap_fsi_checkpoint.py \
      tests/integration/test_ansys_vertical_flap_preflow_snapshot.py \
      tests/validation/test_current_iqn_adaptive_fine_contracts.py \
      tests/validation/test_kalman_iqn_reuse_fine_contracts.py \
      tests/validation/test_our_solver_vertical_flap_runner.py

    /home/zhuohengli/vibeflow_env/bin/python \
      tools/audit_ansys_vertical_flap_oracle_headroom.py verify \
      --output-dir \
      validation_runs/kalman_oracle_headroom/ansys_vf__oracle_headroom__20260831__r24b_final_contract


## 6. Final Q0/Q3 numerical result

Both arms completed exact8 from the same source-matched preflow snapshot with
strict CUDA and physical time 0.004 s.

| Metric | Q0 carry-forward | Q3 exact oracle | Reduction |
| --- | ---: | ---: | ---: |
| coupling trials | 24 | 8 | 66.6667% |
| rejected trials | 16 | 0 | 100% |
| first absolute residual mean, m/s | 1.7711701e-2 | 3.6073550e-8 | 99.9998% |
| first relative residual mean | 5.0467000e-1 | 8.3654278e-7 | 99.9998% |
| pressure CG iterations | 5728 | 1920 | 66.4804% |
| fluid solves | 24 | 8 | 66.6667% |
| solid macro solves | 24 | 8 | 66.6667% |
| momentum substeps | 7581 | 2527 | 66.6667% |
| SST transport substeps | 6873 | 2291 | 66.6667% |
| solid substeps | 30825 | 10275 | 66.6667% |
| warm component wall time, s | 255.119312 | 80.315158 | 68.5186% |
| raw runner elapsed, s | 376.506976 | 157.394711 | diagnostic only |

The directly serialized pressure metric is CG iterations; a complete pressure
matvec total is not available in this artifact schema.

Q3 versus Q0 maximum accepted-state NRMSE was
`2.0743282829153763e-06` for marker/solid state and
`5.7024519828282175e-06` for flow fields.  Both are orders of magnitude
inside the predeclared 0.5% and 1% limits.  Every step passed full fluid and
solid `dt_s`, zero remaining time, pressure/CG, coupling, OOB,
deformation-clamp, retry, no-slip, and canonical marker-closure checks.

## 7. Conditional alpha response

Q0 supplies alpha 0, Q3 supplies alpha 1, and three transparent sealed
producers supplied the intermediate initial guesses.

| alpha | first abs residual mean, m/s | trials | rejected | CG | warm component wall, s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1.7711701e-2 | 24 | 16 | 5728 | 255.119312 |
| 0.25 | 1.3283766e-2 | 24 | 16 | 5744 | 238.070117 |
| 0.50 | 8.8558418e-3 | 24 | 16 | 5760 | 235.426672 |
| 0.75 | 4.4279256e-3 | 24 | 16 | 5760 | 240.400354 |
| 1.00 | 3.6073550e-8 | 8 | 0 | 1920 | 80.315158 |

The first residual decreases monotonically and the coupling-trial count is
nonincreasing.  CG iterations and warm wall time are not monotonically
nonincreasing: all three imperfect oracle runs retain the same three-trial
work and show small CG/timing variation.  All five runs nevertheless pass
accepted-state and physics health, so the response artifact records
`curve_health: true` while preserving the two non-monotonic work flags.

The measured work transition lies somewhere in `(0.75, 1.0]`; exact8 does
not resolve it more narrowly.  A deployable predictor would have to operate
near that region before another expensive causal experiment is justified.

## 8. Evidence identities and paths

Final evidence root:

    validation_runs/kalman_oracle_headroom/
      ansys_vf__oracle_headroom__20260831__r24b_final_contract

Final file SHA256 values, verified twice:

- `oracle_source_manifest.json`:
  `f72448a488061d5950ecf7b7f30be18b26d8a75bdd8252663ccff52916fbd2a7`;
- `oracle_step_metrics.csv`:
  `b74307f5196bbc814ab28308e104017c82b69a222ad38fb989c390ed7b9e56f2`;
- `oracle_headroom_summary.json`:
  `30570a662ab8b9586584934e90561948c56c63ab3f439cac7dd948419f62f763`;
- `oracle_blend_response.json`:
  `832e52d72ea81853648eb2350a3dd4ec4af7a1f9fb920dd83cab44108aaaa96d`.

The summary canonical self SHA is
`0518e15e7d03964a4bfcace0191380602fe227f99e9c9a323de33ed6875eea68`;
the completed blend-response self SHA is
`ffa5a9c7e90b849c37b3df197fc89e0daee363b489166d2623b0b694e0665bde`.

The final Q0, Q3, and alpha consumer roots are respectively:

    validation_runs/solver_soaks/
      ansys_vf__oracle_q0__material_fine__20260831__r24b_final_contract
      ansys_vf__oracle_q3__material_fine__20260831__r24b_final_contract
      ansys_vf__oracle_alpha025__material_fine__20260831__r24b_final_contract
      ansys_vf__oracle_alpha050__material_fine__20260831__r24b_final_contract
      ansys_vf__oracle_alpha075__material_fine__20260831__r24b_final_contract

Earlier dry-run, fail-closed, and pre-review evidence roots were preserved, not
deleted or overwritten.

The bottom-up verifier is materially stronger than the first version, but its
trust boundary is explicit.  It anchors current source files through each
manifest's `repo_root`, independently reconstructs the complete executable
preflow source surface, and then recomputes from the underlying roots.  An
attacker able to forge those root pointers and every underlying artifact still
requires an external trusted-root or signed-hash anchor; R24B does not claim to
solve that external provenance problem.

## 9. Interpretation and next boundary

R24B proves that iteration-zero prediction quality can dominate real
FSI/IQN/CG/fluid/solid work in this exact8 prefix.  It also proves that a large
but imperfect correction, up to the tested 75%, is insufficient to reduce the
discrete work count.

The next step is not a 5000-step run and not immediate K3/GRU/adaptive
implementation.  A separately authorized offline feasibility goal should
first test whether a modal/load-aware deployable predictor can approach the
measured `(0.75, 1.0]` region on source-matched accepted states.  Only a
candidate that clears that prediction gate should return to the same exact8
causal work gate.  Exact50, Fluent comparison, and durability remain blocked
until then.

No commit, push, merge, or remote publication was performed by this Goal.
