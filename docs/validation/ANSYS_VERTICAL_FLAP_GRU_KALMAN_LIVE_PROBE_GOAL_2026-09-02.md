# ANSYS Vertical Flap R25B GRU/Kalman Live No-Commit Probe Goal

Status: complete. The design below was frozen before implementation. The r2 sweep
reached `FAIL_NO_LIVE_SOLVER_WORK_REDUCTION`, but a post-run audit found missing
OOD, explicit rejected-trial, pressure-matvec, and anchor-counter evidence fields.
The frozen gates were not changed; the fresh source-matched r3 chain is complete,
and the required fresh read-only Sol review returned `SHIP`. The implementation
started from R25A commit
`adb2a0470085ecca1f772bae14d292df76c963d9` on branch
`codex/gru-kalman-live-probe-r25b`.

## 1. Purpose and evidence boundary

R25B answers four narrow questions on the real strict-CUDA FSI solver:

1. Does a state-only GRU reduce coupling trials or pressure-CG work relative to
   carry-forward?
2. Does frozen calibrated K1 reduce that work by itself?
3. Does GK1 outperform both K1 and a capacity-matched non-Kalman GRU, so that
   Kalman-specific information has live incremental value?
4. Does the very low offline error of train-only POD-AR translate into solver work
   reduction?

The primary outcome is discrete solver work, not offline NRMSE or wall time. R25B is
a research-only, no-commit target-step experiment. It does not authorize a production
initial-guess controller, online training, adaptive Kalman, exact20/exact50, Fluent,
deployment, a PR, or a push.

R25A remains a valid offline result. Its formal classifications are not reopened:

- G0: `FAIL_OFFLINE_GRU_VALUE`, despite an approximately 11.5% median NRMSE
  improvement relative to carry-forward.
- GK0: `FAIL_OFFLINE_KALMAN_GRU_VALUE`.
- GK1: `PASS_OFFLINE_KALMAN_GRU_VALUE` relative to the frozen neural-family gates.
- POD-AR: NRMSE `0.0200133763284`, approximately 34.9 times smaller than median
  GK1 error.

R25A's historical runtime identity remains the reported
`fbf4b729a68fab4c69316568cadcf46f234202d9` dirty run. The committed R25A
implementation is `adb2a047...`. Commit `00e16d6` separates those contracts: the
current runtime test accepts a valid clean or dirty repository identity, while a
dedicated report test binds the historical JSON and all 11 source hashes to the R25A
commit blobs. The focused R25A suite is `36 passed` when its unchanged local sealed
manifest is present.

## 2. Frozen model controls

The R25A selected G0 and GK1 architectures differ in window length, and GK1 also has
three times the input width. R25B therefore adds two post-hoc mechanism controls. They
do not reopen architecture selection and are never called holdout-optimal models.

### G0-M: architecture-matched state-only GRU

- rank: 8
- window: 4
- hidden size: 16
- seeds: 0, 1, 2, all retained
- input at each history position: accepted modal coefficient `a_i`
- output: carry-forward modal coefficient plus the frozen GRU correction

### GDelta-M: capacity-matched non-Kalman GRU

- rank: 8
- window: 4
- hidden size: 16
- seeds: 0, 1, 2, all retained
- input width: `3 * rank`, equal to GK1
- input at each history position: `[a_i, a_i - a_(i-1), a_(n-1)]`
- output: carry-forward modal coefficient plus the frozen GRU correction
- forbidden inputs: Kalman state, Kalman innovation, Kalman prediction, target-step
  accepted values, rejected trials, or future provenance

Both controls use only D0 steps 1--100 for POD, normalization, and fitting. D0 steps
101--200 are used only for the same early-stopping rule. Architectures are fixed and
there is no search. D1 never selects a seed. The R25A deterministic float64 CPU
training contract, optimizer, loss, zero output-head initialization, and three-seed
retention remain unchanged.

## 3. Causal candidate construction

R25A's saved `d1_predictions.npz` is forbidden as a live input. Every candidate must
be recomputed from the current source-matched accepted prefix:

- target step 7 uses accepted steps 1--6;
- target step 8 uses accepted steps 1--7;
- every manifest records `max_source_step = target_step - 1`;
- C0 is `v_(n-1)`;
- K1 cold-starts from zero, assimilates accepted steps 1 through `n-1`, then predicts
  step `n`;
- G0-M uses the latest four accepted modal states;
- GDelta-M uses the latest four accepted modal states, their causal accepted
  increments, and the repeated current carry baseline;
- GK1 advances the frozen K1 transaction causally and uses the latest four accepted
  modal states, four already-observed K1 innovations, and the current K1 prediction;
- POD-AR uses the latest four accepted modal coefficients;
- Q uses the source-matched target accepted state only as a noncausal upper bound.

The generator records, but does not gate on, maximum normalized POD coefficient,
fraction outside the D0 training range, and maximum normalized innovation.

## 4. Candidate bundle contract

CPU research code generates an immutable manifest plus NPZ. The CUDA solver never
imports PyTorch or loads neural model weights.

The manifest and loader fail closed unless all of the following match:

- schema version;
- target step 7 or 8 and `max_source_step = target_step - 1`;
- marker velocity shape `(128, 3)` for every arm;
- `dt_s = 0.0005`;
- layout SHA256
  `373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164`;
- axis order `x, y, z` and units `marker_velocity_mps`;
- bitwise-zero x velocity;
- identical marker region IDs, reference positions, ordering, and their hashes;
- finite C-contiguous float64 candidate values;
- unique arm IDs and the exact frozen arm matrix;
- whole-NPZ SHA256 and per-candidate SHA256;
- schema-2 causal-input OOD provenance, using D0 fit steps 1--100 and the latest
  four accepted source steps for normalized POD coefficients and K1 innovations;
- finite nonnegative OOD values with fractions constrained to `[0, 1]` and all
  source steps bounded by `target_step - 1`.

The manifest also records generator-source SHA256 provenance. Those hashes are
separately audited against the generator files; because they are authored by the
same generator, the loader does not treat them as an external trust root. Loader
acceptance is enforced by the schema, source bounds, layout/marker identities,
whole-NPZ hash, per-candidate hashes, and OOD contract above.

Any layout, ordering, region, shape, dtype, identity, or hash mismatch is classified
`BLOCKED_MODEL_LAYOUT_MISMATCH`. R25B does not implement marker remapping.

## 5. Frozen step-7/8 arm matrix

Each target evaluates exactly 13 no-commit arms:

| arm | variants | role |
| --- | ---: | --- |
| C0 | 1 | carry-forward reference |
| K1 | 1 | frozen calibrated Kalman alone |
| G0-M | seeds 0, 1, 2 | architecture-matched state-only GRU |
| GDelta-M | seeds 0, 1, 2 | capacity-matched non-Kalman GRU |
| GK1 | seeds 0, 1, 2 | frozen Kalman-residual GRU |
| AR | 1 | train-only POD-AR |
| Q | 1 | exact accepted noncausal upper bound |

The first live matrix is therefore 13 arms times 2 target steps, or 26 solver
evaluations. GK0 is excluded because K0 and GK0 both failed offline and do not answer
the GK1 mechanism question.

## 6. Thin solver integration and transaction invariant

`VerticalFlapFsiConfig` gains only the research-only path
`research_initial_guess_candidate_matrix_path`. It is mutually exclusive with the
existing baseline-to-Oracle alpha sweep.

For a target-step candidate sweep, the official runner must:

1. load and validate the precomputed bundle before the target solve;
2. capture one accepted `HostMacroStepState` and marker interface base;
3. restore that exact base before every arm;
4. clone the existing IQN runtime and replace only the iteration-zero marker velocity;
5. run the existing `solve_fsi_step` with trial-vector recording enabled;
6. record the requested candidate SHA, actual first-IQN-guess SHA, and exact array
   equality;
7. always call rollback in `finally`;
8. recapture and compare the complete `HostMacroStepState` after every arm;
9. restore and compare the complete state again after the whole sweep;
10. return a terminal research report without committing the target step, advancing
    accepted time, or updating GRU/Kalman state.

The existing R24C capture/restore, mismatch-field, residual, IQN, and work-counter
logic is authoritative. R25B extends that research path; it does not create a second
production controller.

Immediate fail-closed conditions are non-finite results, candidate/actual first-guess
mismatch, rollback mismatch, changed accepted time, committed target step, or any
state mismatch after the sweep. Non-convergence stops and classifies the affected arm;
it is never counted as acceleration.

## 7. Frozen solver configuration

All arms use exactly:

- initial Picard omega: 0.75
- IQN previous-step reuse: off
- active Kalman writeback: off
- FSI maximum iterations: 16
- relative tolerance: `1e-3`
- absolute tolerance: 0
- `dt_s`: `5e-4`
- backend: strict CUDA
- solver state: f32
- Taichi: 1.7.4
- random seed: 0
- identical traction, material, geometry, marker layout, and preflow

Every accepted fluid and solid physical step must independently consume the complete
macro `dt_s`. Algebraic convergence may stop same-time pressure, Helmholtz, or FSI
iterations; it must never truncate physical-time consumption.

## 8. Source-matched numerical chain

Runner changes invalidate the R24C.1 executable source identity unless the generated
source map proves otherwise. Old artifacts are never re-signed.

The maximum authorized first-stage numerical chain is:

1. one fresh stationary preflow and strict snapshot, if the runner/source map changed;
2. one source-matched `omega=0.75`, reuse-off, carry exact8 baseline;
3. one target-step-7 candidate sweep;
4. one target-step-8 candidate sweep.

Only one expensive CUDA process may run at a time. The preflow, baseline, and probes
must share executable-source, config, geometry, traction, material, marker-layout,
runtime, and snapshot lineage. Reduced `step_fields/*.npz` files are not restart
states.

## 9. Required live evidence

Prediction diagnostics are frozen before the CUDA run and do not enter any work
gate. They use only active y/z velocity components:

- RMSE is the unweighted component-wise root mean square error;
- area-weighted RMSE first averages squared y/z error per marker, then weights by
  fixed reference `marker_area_m2` normalized by total area;
- the tip region is every marker in the highest 10% of reference y height, with the
  exact Boolean-mask SHA and marker count recorded;
- `alpha_parallel` and `r_perp` project `candidate - C0` onto and perpendicular to
  `Q - C0`, normalized by the squared `Q - C0` norm.

The candidate generator records these definitions in each arm's manifest diagnostics.
The fresh exact8 accepted-step export supplies the immutable marker reference positions,
region ordering, and fixed areas. The CUDA runner imports only the NumPy/stdlib bundle
reader; PyTorch remains confined to the CPU generator process.

Each arm records at least:

- convergence, coupling trials, and rejected trials;
- first/second absolute and relative residuals, maximum-marker residual, and complete
  residual histories;
- pressure CG iterations and candidate-only pressure-operator matvec work;
- fluid solves, momentum/SST substeps, solid macro solves, and MPM substeps;
- the exact invariant
  `coupling_rejected_trial_count = iterations - int(converged)`;
- a target-level anchor-refresh delta equal to the sum of all 13 row trial counts;
- IQN update modes, rank, condition number, fallback, and update limiting;
- per-arm and final rollback equality plus mismatch fields;
- accepted step/time before and after the probe;
- prediction RMSE, area-weighted RMSE, tip-region error, alpha-parallel, and
  orthogonal residual;
- requested candidate SHA, actual first-guess SHA, and exact requested/actual equality;
- CPU inference time and solver wall time as diagnostics only.

## 10. Frozen factor analysis

For target set `T = {7, 8}`, let `W_X` be total coupling trials and `CG_X` total
pressure-CG iterations. For each matched neural seed `s`, report both work measures:

- standalone GRU: `Delta_G(s) = W_C0 - W_GDelta(s)`;
- Kalman alone: `Delta_K = W_C0 - W_K1`;
- GRU conditional on Kalman: `Delta_G_given_K(s) = W_K1 - W_GK1(s)`;
- Kalman-specific information: `Delta_Kinfo(s) = W_GDelta(s) - W_GK1(s)`;
- interaction: `S(s) = W_GDelta(s) + W_K1 - W_C0 - W_GK1(s)`.

The same equations are computed for pressure CG. Positive `S` means positive
interaction; zero means approximately additive work reduction; negative means
overlap or interference.

## 11. Frozen step-7/8 classifications

`PASS_G0_MATCHED_LIVE_VALUE` requires at least two seeds with either:

- at least one fewer total trial than C0 across steps 7 and 8; or
- equal trials, at least 10% lower pressure CG, and no target with more trials.

`PASS_GK1_INCREMENTAL_LIVE_VALUE` requires at least two matched seeds for which GK1
has fewer total trials than both K1 and GDelta-M, or equal trials with at least 10%
lower pressure CG than both.

`PASS_POD_AR_LIVE_VALUE` requires at least one fewer total trial than C0.

If first residual improves but trials are unchanged and CG improves by less than 10%,
classify `PREDICTION_IMPROVED_NO_SOLVER_WORK_REDUCTION`.

If only POD-AR passes, classify `FAIL_NEURAL_LIVE_VALUE_POD_AR_PASS` and stop all
neural expansion. If no arm reduces trials or sufficient CG work, stop the
accepted-state-predictor route. No post-hoc threshold changes are allowed.

## 12. Conditional continuation and hard stop tree

Only a G0-M, GDelta-M, or GK1 live signal authorizes endpoints 9 and 12. That stage
adds the formally selected R25A G0 `8:8:16`. If endpoints retain a signal, steps 10
and 11 may be filled in. The four-step gate is at least two fewer total trials, or
equal trials with at least 15% lower pressure CG and no worsening in at least three
of four targets.

R25C is separate and remains unauthorized until the four-step live matrix passes. Its
first committed causal run would be exact12, not exact20. Exact20 is considered only
after steps 9--12 reduce both trials and pressure CG by at least 10%, pass every
physical-health and accepted-solution-equivalence gate, and show no continuing
degradation. Exact50 remains out of scope.

## 13. Files and artifacts

Intended committed scope:

- `tools/validation/gru_kalman_live/controls.py`
- `tools/validation/gru_kalman_live/candidate_bundle.py`
- `tools/validation/gru_kalman_live/live_analysis.py`
- `tools/run_ansys_vertical_flap_gru_live_probe.py`
- the minimum case/official-runner fields needed for the research sweep
- `tests/validation/test_gru_kalman_live_probe.py`
- focused existing runner contract tests where needed
- this goal, the dated report, and the docs index

Ignored local artifacts:

- `r25a_commit_binding.json`
- `matched_control_training.json`
- `candidate_manifest.json`
- `candidate_predictions.npz`
- `live_probe_raw.json`
- `live_probe_summary.csv`
- `factor_effects.json`
- `artifact_sha256.json`

No new attestation framework, production checkpoint schema, PyTorch CI job, marker
remapping, full-suite cleanup, or release/deployable marker is part of R25B.

## 14. TDD and completion evidence

Implementation order is fixed:

1. RED tests for candidate schema, hashes, layout/order/region mismatch, causal source
   bounds, schema-2 OOD provenance, exact arm matrix, analysis formulas, and stop
   gates;
2. RED runner contracts for mutual exclusion, forced trial-vector recording,
   requested/actual first-guess identity, per-arm rollback, final sweep equality, and
   terminal accepted step/time, plus rejected-trial, pressure-matvec, and anchor
   conservation invariants;
3. minimal GREEN implementation;
4. focused R25A regression, new R25B tests, existing official-runner probe tests,
   `py_compile`, `git diff --check`, source-map review, and actual-diff review;
5. only after those gates, the bounded source-matched CUDA chain.

R25B is complete only when the code and local evidence support one frozen
classification, the dated report distinguishes CPU contracts from live CUDA evidence,
all requested-vs-actual and rollback invariants pass, the final accepted prefix remains
unchanged, and a fresh read-only reviewer returns `ship`. Focused tests or CPU-only
training are never reported as solver acceleration, full-suite validation, Fluent
parity, or production readiness.

## 15. r3 completion evidence

Implementation commit `f833439bf5fc15c6f04923410c8f804e5a394fe8` completed
the schema-2 OOD, explicit rejected-trial, candidate-only pressure-matvec, and anchor
contracts without changing the frozen work gates or production/oracle checkpoint
schemas. The authoritative r3 chain contains six equal 141-entry formal source maps;
their canonical sorted compact-JSON digest is
`e0b643d2f8ec8d36935148b44c8139125cb9499798bbb9109ccb6e1bb6f4e28b`.
The fresh preflow reached stationary step 79 and produced the shared strict snapshot
`state.4237fbac9c384f6fb0fb3427c7ce2f84.npz`, SHA256
`7c4cb847bbc8b09a10049f73c5fd6c9589ddf9413e59a6634e043ed2ef2dccc3`.
Exact8 accepted 8/8 steps; its no-op resume restored accepted step 8 and executed no
new physical step.

Review-fix commit `4349f63` adds the previously missing G0-M-only continuation
branch to the bottom-up classifier. `live_analysis.py` is outside the six formal CUDA
source maps, so the raw r3 runs remain source matched; reanalysis leaves every G0-M
passing-seed set empty and the frozen scientific classification unchanged.

Both target bundles are immutable schema-2 `(13, 128, 3)` float64 arrays. Their OOD
source windows are steps 3--6 and 4--7, respectively; those diagnostics do not enter
the work gates. Probe 7 and probe 8 each preserve the accepted prefix, have 13/13
exact requested/actual first guesses, 13/13 equal per-arm rollbacks, equal final
sweep state, no target artifact, and anchor delta 37 equal to the target's summed
trial count.

Every causal arm totals 6 trials, 4 rejected trials, 1,440 pressure-CG iterations,
and 1,452 pressure matvecs across the two targets. Noncausal Q totals 2 trials, 0
rejected trials, 480 CG iterations, and 484 matvecs. The full 26-row matrix sums to
74 trials, 48 rejected trials, 17,760 CG iterations, and 17,908 matvecs. The frozen
classifications are:

- `FAIL_G0_MATCHED_LIVE_VALUE`;
- `FAIL_GK1_INCREMENTAL_LIVE_VALUE`;
- `FAIL_POD_AR_LIVE_VALUE`;
- `FAIL_NO_LIVE_SOLVER_WORK_REDUCTION`.

The focused R25B suite passed 14 tests with 3 skipped; the six-file formal
CLI/checkpoint/resume regression passed 141 tests. Selected independent runner and
pressure-nullspace contracts passed. The adjacent full SST runner suite still has
one pre-existing, out-of-scope AST-kwargs expectation failure, so no full-suite-green
claim is made. No exact9/12 expansion, step10/11 fill, R25C, exact12/20/50, Fluent,
production controller, online training, deployment, push, or PR was authorized.
The dated report contains the exact evidence. The required fresh read-only Sol
reviewer returned `SHIP`; this Goal is complete on its bounded evidence and stop tree.
