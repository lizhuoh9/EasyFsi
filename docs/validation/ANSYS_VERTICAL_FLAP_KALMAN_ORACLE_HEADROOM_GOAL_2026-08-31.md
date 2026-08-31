# ANSYS Vertical-Flap Kalman Oracle Headroom Goal

Date: 2026-08-31
Status: COMPLETE; `PASS_ORACLE_HEADROOM`, conditional alpha response completed

## 1. Decision question

R24 rejected the tested K0/K1/K2 local independent-marker Kalman family.  It
did not prove that every Kalman model is useless.  Before proposing a modal
model, adaptive covariance, GRU, exact50, or a 5000-step run, R24B must answer:

> If iteration zero is given the accepted marker velocity from the same
> physical step, does the real IQN-ILS solve perform materially less work
> without changing the accepted physical result?

This is a causal upper-bound experiment.  It is not deployable, predictive, or
a production speedup claim.

## 2. Authoritative state and publication boundary

All Git operations, edits, tests, artifact reads, and numerical runs use WSL
Ubuntu-22.04 at:

    /home/zhuohengli/worktrees/HIBM-MPM-r21-validation

R24B branch and starting state:

    branch: codex/kalman-oracle-headroom-r24b
    starting commit: 0ab5157ae7451d7af0082379d84356cb6b7fc0ea
    remote-equivalent content commit: a901948

The older WSL checkout under
`/home/zhuohengli/work/squid-robot/HIBM-MPM-refactored` and the Windows
project mirror are not sources of truth.  Preserve their dirty state.

Do not reset, clean, delete unrelated files, commit, push, merge, or alter a
remote without separate authorization.  Numerical artifacts remain local
under the ignored `validation_runs/` tree.

## 3. Analysis incorporated from the R24 review

The review established these evidence boundaries:

1. R24 is internally consistent and has no known P0/P1 defect.
2. Its normalized accepted-marker-velocity RMSE is an offline prediction
   metric; it does not by itself establish a causal link to the first
   fixed-point residual, IQN trial count, CG work, or wall time.
3. The r47 baseline used three IQN trials on every accepted step.  Existing
   evidence does not show whether that floor is algorithmic or caused by the
   iteration-zero guess.
4. K1/K2 are local independent-marker models.  They do not represent spatial
   interface modes, load/pressure inputs, or a defensible measurement-noise
   model.
5. The repository already contains an accepted-only, non-deployable
   `oracle_replay` route.  R24B must reuse it and must not modify the
   production predictor, initial-guess controller, IQN equations, solver, case,
   or runner.
6. An older r06 quick oracle pilot reduced work, but it covered a different
   5 ms prefix and cannot replace a fresh source-matched r47-family experiment.

Therefore the next authorized question is oracle headroom only.  Adaptive
Kalman, GRU, modal K3, exact50, Fluent comparison, and long-horizon work remain
outside this goal.

## 4. Frozen numerical and transactional invariants

Q0 and Q3 must be fresh runs from the same strict post-preflow snapshot.  They
must differ only in the initial-guess control surface.

- strict CUDA, Taichi 1.7.4, float32 solver fields, random seed 0;
- exact 8 accepted macro steps;
- `dt_s = 5e-4 s`;
- grid nodes `[4, 256, 320]`;
- solid particles `[1, 256, 20]`;
- 64 interface markers;
- adaptive solid substeps with CFL target 0.14;
- `iqn_ils`, maximum 16 trials, relative tolerance `1e-3`;
- no IQN history reuse;
- no Kalman modified-physics writeback;
- FV-Jacobi pressure route with FV multigrid preconditioner and unchanged
  `1e-6` CG tolerance;
- unchanged `1.1e-6 m/s` marker compatibility closure;
- step fields, IQN trial vectors, step histories, and synchronized component
  wall timing enabled;
- source SHA map, layout SHA, config, preflow snapshot identity, and every
  accepted frame/history file bound into the evidence manifest.

Every accepted fluid advance and every accepted solid advance must consume the
full macro `dt_s`.  A converged residual ends algebraic iterations at the same
physical time; rejected trials and rollback never advance physical time.

The oracle affects only the iteration-zero trial guess.  Q3 step `n` trial
zero must equal Q0 step `n` accepted marker velocity exactly.  Q3 may not
commit future information to predictor state, checkpoints, or physical fields.

## 5. Locked experiment matrix

| ID | Initial guess | IQN reuse | Purpose |
| --- | --- | --- | --- |
| Q0 | carry-forward | off | source-matched baseline and oracle producer |
| Q3 | same-step Q0 accepted velocity | off | non-deployable upper bound |
| A25 | Q0 trial zero + 0.25 x oracle correction | off | conditional response curve |
| A50 | Q0 trial zero + 0.50 x oracle correction | off | conditional response curve |
| A75 | Q0 trial zero + 0.75 x oracle correction | off | conditional response curve |

Q0 is also alpha 0 and Q3 is alpha 1.  A25/A50/A75 are permitted only if the
joint Q0/Q3 oracle gate passes.  Each intermediate producer is a transparent
derived trajectory with its own frame and derivation SHA; it remains
non-deployable.

## 6. Predeclared joint acceptance gate

The oracle has actionable headroom only when all five gates pass:

1. total coupling trials decrease by at least 10 percent;
2. total pressure CG iterations or a directly serialized pressure-matvec count
   decreases by at least 10 percent;
3. synchronized warm component wall time, summed over flow, HIBM, and solid for
   steps 2--8, decreases by at least 5 percent;
4. both runs pass pressure, coupling, full physical-time, OOB, deformation
   clamp, retry, no-slip, and canonical marker-closure health checks;
5. Q3 versus Q0 accepted-state NRMSE remains at most 0.5 percent for marker
   velocity/position and solid position, and at most 1 percent for
   `u`, `v`, pressure, and speed fields.

The wall metric is synchronized profiled component work, not raw process
elapsed time.  The current artifact schema directly serializes pressure CG
iterations but not a complete pressure-matvec total, so the first R24B decision
uses the CG branch of gate 2 and records that limitation explicitly.

Classification is fail-closed:

- all five pass: `PASS_ORACLE_HEADROOM`;
- any gate fails: `STOP_KALMAN_ACCELERATION`;
- missing, malformed, non-finite, non-exact8, source-drifted, config-drifted,
  layout-drifted, or oracle-mismatched evidence: contract error, with no
  performance classification.

No favorable single metric can override the joint gate.

## 7. Ordered execution and stop tree

1. Add failing focused tests for exact8 identity, config/source drift, same-step
   oracle equality, work gates, physical-time health, deterministic alpha
   derivation, and self-fingerprinted artifacts.
2. Implement the smallest Taichi-free analyzer and CLI needed to make those
   tests pass.  Do not touch production solver files.
3. Use a Q0 dry-run to compare its complete config with r47 before any CUDA
   advance.  Only `step_count` and the unused checkpoint-output destination
   may differ from the r47 fresh-step manifest.  Attempt to load the r47
   preflow snapshot strictly; if its executable-source identity differs, retain
   the zero-step failure and generate a fresh snapshot under the frozen R24B
   executable source.
4. Confirm no other expensive CUDA task is active.
5. Run Q0 strict-CUDA exact8.  Verify terminal completion and all artifacts
   before starting Q3.
6. Run Q3 strict-CUDA exact8 from Q0's sealed trajectory.  Run only one
   expensive CUDA job at a time.
7. Generate and independently verify the four decision artifacts.
8. If classification is `STOP_KALMAN_ACCELERATION`, stop.  Do not generate
   alpha runs, K3, adaptive, GRU, exact50, Fluent, or long-run evidence.
9. If classification is `PASS_ORACLE_HEADROOM`, generate sealed A25/A50/A75
   producers and run the three exact8 consumers serially.  Reuse Q0/Q3 for
   alpha 0/1.  Report monotonicity, trial/CG/wall response, first residuals,
   accepted-state deltas, and all health gates.
10. Only a useful, stable response curve may authorize a separately scoped K3
    modal/load-aware model goal.  It does not authorize implementation here.
11. Run focused and related regression tests, `git diff --check`, source
    boundary checks, documentation consistency, and a fresh read-only
    Sol/Ultra review.

## 8. Required implementation and evidence

Tracked files:

- `src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom.py`;
- `src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_analysis.py`;
- `src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_artifacts.py`;
- `src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_contracts.py`;
- `src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_integrity.py`;
- `src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_verification.py`;
- `tools/audit_ansys_vertical_flap_oracle_headroom.py`;
- `tests/validation/test_kalman_oracle_headroom.py`;
- `tests/validation/test_kalman_oracle_headroom_fail_closed.py`;
- this Goal;
- `ANSYS_VERTICAL_FLAP_KALMAN_ORACLE_HEADROOM_REPORT_2026-08-31.md`;
- the targeted entry in `docs/README.md`.

Local evidence directory:

    validation_runs/kalman_oracle_headroom/
      ansys_vf__oracle_headroom__20260831__r24b_final_contract/

Required decision artifacts:

- `oracle_source_manifest.json`;
- `oracle_step_metrics.csv`;
- `oracle_headroom_summary.json`;
- `oracle_blend_response.json`.

The source manifest binds both run roots, source/config/layout identities, and
every exact8 frame/history SHA.  The summary and blend response carry canonical
self-SHA values.  The summary binds the manifest and CSV file SHA values.  The
final verifier does not stop at bundle-internal SHA checks: it follows the
manifest roots, reloads Q0/Q3 and all alpha producers/consumers, revalidates
their frozen runtime/config/source/preflow/layout contracts, and recomputes the
CSV, summary, response curve, producer identities, and consumer identities
bottom-up.

## 9. Locked run roots

The r47 preflow state was tested first and rejected before any accepted step:

    stored source identity:
      69ce29b3be3379267acb1e7386bf13e15e1806b6c0528c40b5b2cdab192c5495
    current expected source identity:
      a548b0f351ebaf2aaa5270d91ffd8b7b784d8a95db54dbd8f5093dd8a74f09d323

The retained zero-step failure root is:

    validation_runs/solver_soaks/
      ansys_vf__oracle_q0__material_fine__20260831__r24b__failed_snapshot_identity

The first source-matched R24B preflow and exact8 pair were retained.  A
Sol/Ultra fix-first review then required stronger frozen runtime/config checks
and bottom-up artifact verification.  A later fail-closed audit required
independent executable-preflow identity reconstruction, absolute array shapes,
an explicit closure tolerance and invalid-axis field, and a null Q0 oracle
path.  Because this final hardening changed the sealed audit source map again,
the final decision evidence uses another fresh preflow instead of re-signing
earlier numerical runs:

    validation_runs/solver_soaks/
      ansys_vf__preflow__material_fine__20260831__r24b_final_contract/state

It completed 79 stationary preflow steps, zero FSI steps, and physical time 0.
Its state JSON and NPZ file SHA256 values are respectively:

    a97a737d3a5ff3bb949a7e37fe4eea15177dbb2b1c918bd5aaaca4ee92abc904
    4a3ad74a0a2e6f690a77d358e2ebba3e7d919b2e20a624d5b6a2800ddd20e932

Q0:

    validation_runs/solver_soaks/
      ansys_vf__oracle_q0__material_fine__20260831__r24b_final_contract

Q3:

    validation_runs/solver_soaks/
      ansys_vf__oracle_q3__material_fine__20260831__r24b_final_contract

The authorized intermediate producers and consumers use explicit
`alpha025`, `alpha050`, and `alpha075` labels with the same
`__r24b_final_contract` suffix.  They do not overwrite Q0 or Q3.  Earlier
`__r24b`, dry-run, and fail-closed roots remain local historical evidence,
not inputs to the final classification.

## 10. Evidence claims that remain prohibited

R24B cannot claim:

- a deployable oracle, Kalman, modal, adaptive, or GRU predictor;
- production acceleration from a failed or partial joint gate;
- exact50, 5000-step durability, Fluent parity, or truth;
- statistical calibration of Q/R;
- broad regression health from focused tests;
- raw end-to-end wall-time acceleration from synchronized component timing.

The final report must state the exact classification, exact commands, exact
passed checks, missing metrics, artifact SHA values, local evidence paths, and
the resulting stop/continue branch.

## 11. Completion evidence and resulting branch

The final source-matched strict-CUDA exact8 pair passed every predeclared joint
gate:

| Metric | Q0 | Q3 | Reduction |
| --- | ---: | ---: | ---: |
| coupling trials | 24 | 8 | 66.6667% |
| rejected trials | 16 | 0 | 100% |
| pressure CG iterations | 5728 | 1920 | 66.4804% |
| warm component wall time | 255.119312 s | 80.315158 s | 68.5186% |

Q3 versus Q0 maximum accepted-state NRMSE was
`2.0743282829153763e-06` on marker/solid state and
`5.7024519828282175e-06` on flow fields.  Both arms passed exact8,
strict-CUDA, full fluid/solid physical-time, pressure, coupling, OOB,
deformation-clamp, retry, no-slip, and marker-closure checks.

The conditional alpha response was also completed.  Alpha 0.25, 0.50, and
0.75 each still required 24 coupling trials and 16 rejected trials.  Their
first residuals decreased monotonically, while their CG totals were 5744,
5760, and 5760 rather than Q0's 5728.  Warm component timing varied and was
5.77--7.72% lower than Q0, but without any trial or CG reduction that timing
alone is not causal acceleration.  Only the exact alpha-1 oracle crossed the
discrete work threshold.

Therefore:

- the causal upper-bound question is answered yes and the formal
  classification is `PASS_ORACLE_HEADROOM`;
- R24's `FAIL_NO_KALMAN_PREDICTIVE_VALUE` conclusion for K0/K1/K2 remains
  unchanged;
- this result does not authorize enabling the current Kalman family,
  implementing K3/GRU/adaptive covariance, or starting exact50/5000-step work;
- any next model goal must first show, offline and source-matched, that a
  deployable predictor can approach the narrow `(0.75, 1.0]` correction
  region and then pass the same causal exact8 work gates.

The final schema-2 artifacts were verified twice from the underlying run roots
with identical file SHA256 values:

- `oracle_source_manifest.json`:
  `f72448a488061d5950ecf7b7f30be18b26d8a75bdd8252663ccff52916fbd2a7`;
- `oracle_step_metrics.csv`:
  `b74307f5196bbc814ab28308e104017c82b69a222ad38fb989c390ed7b9e56f2`;
- `oracle_headroom_summary.json`:
  `30570a662ab8b9586584934e90561948c56c63ab3f439cac7dd948419f62f763`;
- `oracle_blend_response.json`:
  `832e52d72ea81853648eb2350a3dd4ec4af7a1f9fb920dd83cab44108aaaa96d`.

The review-driven hardening also freezes absolute requested and actual
runtime/config values, rejects runtime arm or IQN-reuse drift, resolves
relative oracle paths from each manifest's repository root, recomputes producer
trajectory SHA values and exact frame-key sets, and seals alpha consumer
identity.  It also reconstructs the current preflow executable-source identity
independently of the run manifest, requires the complete executable source
surface, locks real marker/solid/flow array shapes, and rejects missing or
drifted closure and Q0-oracle fields.  Required decision values must be finite;
optional empty-set
diagnostics in the underlying solver summary may remain NaN but are not used or
copied into strict producer manifests.  The remaining trust boundary is
explicit: a party able to forge the manifest's repository/run roots and every
underlying run artifact still requires an external trusted root or signed hash
anchor.

Final verification includes the focused and related contract suites,
`py_compile`, CLI help without Taichi initialization, whitespace checks, a
double bottom-up artifact verification, and a fresh read-only Sol/Ultra review.
Exact final test counts and review disposition are recorded in the companion
report and final handoff.  Publication remains outside this Goal.
