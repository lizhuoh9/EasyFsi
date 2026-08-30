# ANSYS Vertical-Flap Kalman Statistical Diagnosis and Calibration Goal

Date: 2026-08-31
Status: COMPLETED for R24 with `FAIL_NO_KALMAN_PREDICTIVE_VALUE`; R25 through
R28 remain gated and are not authorized

## 1. Active contract

Complete a solver-independent R24 Kalman statistical diagnosis and calibration
campaign before any new shadow, active, adaptive, GRU, or long-horizon run.

This goal must answer one decision question:

> Is the current Kalman failure caused by a statistically mismatched model,
> scale, covariance, or time alignment, or does a more accurate initial guess
> still fail to improve an IQN-dominated FSI convergence path?

The R24 deliverable is an auditable offline result and a single explicit exit
classification. It is not a production speedup claim.

## 2. Authoritative environment and baseline

All Git operations, edits, tests, artifact reads, and analysis commands must run
through WSL Ubuntu-22.04.

Authoritative worktree for this goal:

    /home/zhuohengli/worktrees/HIBM-MPM-r21-validation

Starting branch and immutable validation baseline:

    branch: codex/closure-diagnostic-r23
    commit: f916f80afac5f5ca6d6558e4c3e87fba40831626

Create the implementation branch only after rechecking a clean status:

    codex/kalman-calibration-r24

Do not use the older
/home/zhuohengli/work/squid-robot/HIBM-MPM-refactored worktree as the source of
truth for this campaign. It is an earlier dirty WIP at 952332e. Do not copy
changes from the Windows mirror into WSL.

Do not reset, clean, delete, commit, push, merge, or modify a remote unless the
user separately authorizes that action. Preserve all unrelated evidence.

## 3. Evidence boundary already confirmed

The following facts were verified in the authoritative WSL worktree:

1. f916f80 locks accepted-state provenance, raw IQN secant replay, lstsq, rank,
   condition, fallback, reset, next-guess, and closure evidence. It proves an
   auditable H3 chain, not Kalman acceleration.
2. The current InterfaceKalmanPredictor is already an independent-DOF
   constant-rate state model with state [interface velocity, interface
   acceleration]. It is not a simple random-walk model.
3. The current predictor uses float64 state/covariance arithmetic, a Joseph-form
   covariance update, positive-finite innovation variance checks, symmetric/PSD
   validation, accepted-step commit, and rejected-step discard.
4. The locked H3 configuration is axis-specific, but it is not expressed in a
   normalized dimensionless state space.
5. The r51 H3 evidence root exists and completed strict-CUDA exact50 at 0.025 s:

       /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3

   Its completed resume attempt is:

       /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3__resume50__attempt1

6. r51 passed the locked physical-time, pressure, closure, no-slip, marker,
   checkpoint, and exact50 evidence gates. It does not prove 5000-step
   durability, Fluent parity, or speedup.
7. r51 reports mean initial-guess RMSE near 0.0198655 m/s and NIS mean near
   73.2825 with range 4.1025 to 225.0764. This is not statistically calibrated.
8. Current per-step public JSON records only aggregate NIS mean, prediction
   bias, and prediction RMS. It does not expose the per-axis innovation, S,
   gain, prior/posterior covariance, or the exact Q/R values needed to diagnose
   the failure.
9. The existing r48 offline pilot used the r47 accepted trajectory, steps 1-20
   for calibration and 21-50 for frozen evaluation. Its RMSE-only selection
   chose q_multiplier 0.1, which produced frozen-evaluation NIS near 13.8.
   The same frozen sensitivity table reports q_multiplier 1.0 with slightly
   lower RMSE and NIS near 1.38. This confirms that RMSE-only selection without
   a statistical-consistency and non-degeneracy gate is a concrete failure
   mode to audit, not yet the final calibrated answer.

## 4. User-supplied claims that remain unverified

The supplied analysis reports:

- r38 versus no-Kalman r44: FSI work +11.37 percent, CG +15.10 percent,
  wall time +14.57 percent.
- r43 versus r49: FSI work +17.76 percent, CG +33.46 percent,
  wall time +14.48 percent.
- r38: 45 NIS alerts in 49 samples, with axis means near
  3.4e6, 7.3e24, and 1.8e26.

These figures are valid motivation, but the matching r44 artifact and an
unambiguous mapping from those run labels to current WSL manifests were not
located during goal creation. Do not confuse unrelated material-surface r38/r43
runs with the claimed Kalman comparison, and do not use the figures as a live
acceptance baseline until their manifests are found.

A stronger live calibration source was verified after goal creation: the r47
canonical trajectory contains 200 accepted frames, and its completed resume200
attempt binds those frames to a completed strict-CUDA 200/200 run:

    canonical root: /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__fresh01__material_fine__20260830__r47

    completed attempt: /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__resume200__material_fine__20260830__r47

r47 and r51 share dt_s 5e-4 s, marker_velocity_mps shape 128 by 3, layout SHA256
373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164,
and identical production solver/predictor source hashes. Their four differing
source files are atomic publication and validation control-plane files, which
must remain listed in the split manifest.

If the r47 canonical/attempt dual-root provenance cannot be locked, exit
BLOCKED_MISSING_CALIBRATION_EVIDENCE. Do not silently tune and score on r51.

## 5. Research hypotheses

### H0-K: statistical consistency

Determine whether prediction error agrees with the filter uncertainty. Audit:

- per-axis innovation bias and scale;
- per-axis NIS distribution and 95 percent exceedance rate;
- unit and axis scaling;
- accepted-state time indexing;
- Q, R, and P0 floors and collapse;
- prior/posterior covariance evolution;
- gain collapse or saturation;
- lag-1, lag-2, and lag-3 innovation autocorrelation;
- Ljung-Box evidence and same-sign run lengths;
- behavior around steps 5, 7, 24, 32, 42, and 49.

H0-K must pass before any online acceleration claim.

### H1-P: offline prediction value

On held-out accepted-state data, determine whether a calibrated Kalman
candidate predicts the next accepted interface state better than both
carry-forward and predeclared linear extrapolation.

### H1-A: online solver acceleration

This is not part of R24. It begins only in a later goal after H0-K and H1-P
pass.

## 6. Frozen solver and validation contracts

R24 must not change numerical solver behavior. In particular, do not modify:

- simulation_core/coupling/interface_kalman_predictor.py;
- simulation_core/coupling/interface_initial_guess_controller.py;
- IQN equations, history, rank, condition, fallback, or reset rules;
- accepted-state ownership, checkpoint, retry, or rollback semantics;
- runner, case, pressure, CG, closure, no-slip, marker, or Fluent gates;
- exact50, strict-pressure, absolute 1e-4 m/s, closure 1.1e-6 m/s, or
  invalid-axis-equals-zero contracts;
- fluid or solid physical substep policies.

For every later online trial, fluid accepted dt and solid accepted dt must each
sum to the full macro dt_s. Rejected trials advance zero accepted time.
Same-time algebraic convergence may end IQN/pressure/CG work, never physical
time advancement.

R24 must not launch Taichi/CUDA simulation, fresh preflow, Fluent, 200-step,
500-step, 1000-step, or 5000-step work. It must not implement adaptive Kalman,
adaptive fluid/solid substeps, or GRU.

## 7. Data split and leakage prevention

Use three strictly separated datasets:

1. Calibration/diagnosis dataset D0:
   the r47 canonical accepted-state trajectory, bound to the completed
   resume200 attempt. Candidate selection and normalization may use only a
   predeclared prefix/split within r47. The existing r48 pilot is historical
   evidence and must be independently replayed rather than trusted by copy.
2. Held-out dataset D1:
   r51 zero-preflow exact50 accepted-state source replay. It may score frozen
   candidates, but it may not select scales, Q, R, P0, beta, warmup, or
   thresholds.
3. Future blind dataset D2:
   a new shadow/no-op exact50 run. It belongs to R25 and must not be generated
   in this goal.

Produce a machine-readable split manifest containing artifact roots, source
SHA256 values, step range, dt_s, state layout identity, axis order, units,
normalization statistics, and a fingerprint over every tuning choice.

Reduced step_fields NPZ files are not restart checkpoints. If they are used as
offline observations, label that use explicitly and bind every observation to
the accepted checkpoint/journal and step-history provenance.

## 8. Candidate matrix

Freeze and score all of the following:

| ID | Predictor | Purpose |
| --- | --- | --- |
| C0 | Previous accepted state | Primary non-Kalman baseline |
| C1a | Linear extrapolation, beta 0.5 | Conservative trend baseline |
| C1b | Linear extrapolation, beta 0.8 | Intermediate trend baseline |
| C1c | Linear extrapolation, beta 1.0 | Full first-order trend baseline |
| K0 | Exact current constant-rate implementation and locked old config | Historical failure control; never tune |
| K1 | Dimensionless, per-axis random-walk Kalman | Isolate scale/covariance effects |
| K2 | Dimensionless, per-axis constant-rate Kalman | Calibrated version of the current model family |

K2 is not a newly discovered model: the live K0 implementation is already
constant-rate. K2 differs through explicit normalization and held-out-safe
calibration. Do not add acceleration as a third state in R24.

At most two Kalman candidates may be recommended for R25. If C1 is as good as
or better than every non-degenerate Kalman candidate, report that result and
stop the Kalman promotion path.

## 9. Required implementation surface

Prefer the existing validation package over new solver code:

- src/refactored/validation/ansys_vertical_flap_fsi/
  kalman_statistical_calibration.py
- tools/audit_ansys_vertical_flap_kalman.py
- tests/validation/test_kalman_statistical_calibration.py
- docs/validation/
  ANSYS_VERTICAL_FLAP_KALMAN_CALIBRATION_REPORT_2026-08-31.md

Generated evidence belongs under a validation_runs campaign named according to
validation_runs/README.md, for example:

    validation_runs/solver_soaks/
    ansys_vf__kalman__offline_calibration__20260831__r24

Do not import implementation from validation_runs or archive. Keep the core
analysis solver-independent and free of Taichi/GPU initialization. Reuse the
production predictor only to reproduce K0 exactly; keep K1/K2 experimental
calibration logic in the validation surface until a later promotion decision.

Avoid adding a dependency solely for this analysis when NumPy and the standard
library are sufficient. Any statistical approximation must be documented and
covered by a deterministic test.

## 10. Required row-level telemetry

For every physical step and axis, emit at least:

- physical_step;
- accepted_state_source_step;
- accepted_state_source_sha256;
- layout_id and axis;
- model_name and candidate fingerprint;
- prediction and measurement summaries;
- innovation;
- innovation covariance S;
- normalized innovation squared;
- Kalman gain;
- P prior and P posterior;
- Q, R, and P0 identity;
- fallback reason and reset reason;
- normalized RMS, maximum-component, and representative-DOF error;
- aligned FSI iterations, CG iterations, and matvec count when provenance
  supplies them.

For a high-dimensional interface field, the public CSV may contain stable
per-axis norms and representative DOFs instead of every matrix entry, but the
audit calculation must use the actual S and must retain sufficient
machine-readable evidence to recompute every aggregate.

Required outputs:

- kalman_innovation_audit.json;
- kalman_candidate_ranking.json;
- kalman_candidate_step_metrics.csv;
- kalman_data_split_manifest.json;
- the dated Markdown calibration report.

All JSON/CSV outputs must be deterministic, finite, schema-versioned, and
fingerprinted.

## 11. TDD and verification sequence

Follow RED, GREEN, IMPROVE:

1. Write focused failing contracts before implementation.
2. Implement the minimum offline replay and audit behavior.
3. Refactor only after focused tests are green.
4. Run only bounded WSL host tests for R24.
5. Review the final diff and generated evidence for source leakage and claim
   inflation.

Required focused tests:

1. deterministic replay;
2. unit-scaling invariance for state and covariance;
3. axis-permutation/schema rejection;
4. stale n-2 source rejection;
5. rejected-trial isolation;
6. covariance finite, symmetric, and PSD checks;
7. exact K0 parity with the current production predictor;
8. calibration/held-out split isolation;
9. bad source/config/layout fingerprint rejection;
10. checkpoint/restart replay continuity;
11. deterministic ranking and tie handling;
12. negative test that a huge R or near-zero gain cannot win only by making NIS
    look benign;
13. missing or inconsistent r47 canonical/attempt provenance exits blocked
    instead of substituting r51;
14. no Taichi import or solver-state mutation from the R24 CLI.

Use the WSL interpreter currently available at /usr/bin/python3. Start with:

    python3 -m pytest -q       tests/validation/test_kalman_statistical_calibration.py

Then run the smallest existing predictor/controller contracts affected by
imports. Do not call focused tests a full-suite or physical validation.

## 12. Diagnostics and ranking

For each axis and candidate, report:

- normalized RMSE versus C0 and versus the best C1;
- maximum one-step error;
- innovation mean, standard deviation, median, and robust scale;
- NIS mean, median, quantiles, and 95 percent exceedance fraction;
- lag-1 through lag-3 autocorrelation and Ljung-Box statistic;
- positive/negative same-sign run lengths;
- covariance minimum eigenvalue, symmetry error, and finite status;
- gain distribution;
- fallback/reset count;
- segmented results for steps 1-5, 6-15, 16-31, 32-41, 42, and 43-49.

The report must separate:

1. large innovation;
2. too-small S;
3. unit/axis scaling;
4. time-index mismatch;
5. systematic model lag;
6. combinations of the above.

Do not increase the NIS alarm threshold to hide a mismatch. Do not obtain a
passing NIS result by making R so large that the Kalman gain collapses and the
candidate degenerates to C0.

## 13. R24 acceptance and exit classifications

R24 may end with exactly one of:

- PASS_ADVANCE_TO_R25;
- FAIL_NO_KALMAN_PREDICTIVE_VALUE;
- FAIL_STATISTICAL_MODEL;
- BLOCKED_MISSING_CALIBRATION_EVIDENCE;
- FAIL_EVIDENCE_OR_IMPLEMENTATION_CONTRACT.

PASS_ADVANCE_TO_R25 requires all of the following:

1. D0 and D1 provenance and split fingerprints are complete.
2. K0 offline replay reproduces the locked implementation and existing
   aggregate telemetry within a predeclared numerical tolerance.
3. The catastrophic or elevated NIS is attributed with direct innovation/S/
   scale/time/model evidence, not speculation.
4. At least one non-degenerate K1/K2 candidate reduces held-out normalized RMSE
   by at least 5 percent relative to C0 and is better than the best frozen C1.
5. Held-out NIS is brought back to the order of unity rather than 1e6 to 1e26,
   its exceedance is not systematically near 100 percent, and the report does
   not hide axis-specific failure in an aggregate mean.
6. Innovation has no unexplained persistent bias or serial pattern.
7. Covariance remains finite, symmetric, and PSD with no NaN, Inf, or negative
   variance.
8. The winning candidate does not rely on gain collapse, fallback, or reset to
   imitate C0.
9. Every required test and deterministic artifact gate passes.
10. The report states the exact evidence boundary and names the next permitted
    stage without running it.

If prediction improves but no trustworthy NIS calibration exists, use
FAIL_STATISTICAL_MODEL. If no Kalman candidate beats the best simple
extrapolation, use FAIL_NO_KALMAN_PREDICTIVE_VALUE. Both are scientifically
useful results and must stop promotion.

## 14. Locked later-stage gates

These stages are documented here to preserve order, but they are not executable
under this R24 goal.

### R25: shadow/no-op exact50

Only after PASS_ADVANCE_TO_R25:

- production initial guess remains C0;
- all predictors run read-only;
- three independent strict-CUDA exact50 runs;
- trajectory identity and accepted-state source match;
- no-op wall overhead at most 3 percent;
- no solver state mutation;
- new D2 is blind until candidates are frozen.

### R26: active H1 screening

Only after R25 passes:

- change only iteration-0 initial guess;
- keep IQN, tolerances, substeps, checkpoint, logging, and CUDA identity fixed;
- at least three paired runs per candidate with balanced/randomized order;
- stop on any correctness failure;
- stop a candidate after two pairs with FSI work worse by more than 5 percent
  or repeated CG/matvec regression.

### R27: H1 confirmation

Keep only one candidate and run five new randomized pairs. A production
acceleration claim requires:

- total FSI iterations down at least 10 percent;
- wall time down at least 5 percent;
- alpha 0.01 paired significance;
- at least four of five pairs in the same direction;
- CG and matvec regression no worse than about 3 percent;
- every pressure, closure, accepted-state, rollback, and physical-time contract
  passing;
- no benefit confined only to steps 1-5 and no concentrated regression around
  difficult steps 32, 42, or 49.

Fluent comparison is not a Kalman acceptance metric.

### R28: long-horizon ladder

Only after R27 passes, run matched control/candidate pairs in order:

    repeat exact50 -> 200 -> 500 -> 1000 -> 5000

Each level must pass before the next. Monitor cumulative FSI/CG/matvec work,
covariance drift, NIS drift, fallback clusters, restart identity, pressure,
closure, state drift, and actual wall reduction.

Adaptive Kalman is considered only if fixed calibrated models work in stable
segments but NIS rises systematically in rapid-deformation segments. GRU is
considered only after multiple independent operating conditions provide a
proper train/validation/test split and simple predictors have reached a proven
limit.

## 15. Completion handoff

At R24 completion, update this document or the dated calibration report with:

- exact branch and commit or dirty-diff boundary;
- all changed files;
- exact WSL commands and results;
- artifact paths and fingerprints;
- confirmed root cause;
- candidate ranking;
- the single exit classification;
- remaining blockers;
- the next permitted goal.

Do not commit, push, merge, start R25, or claim production acceleration without
a separate authorization boundary.

## 16. R24 completion result

R24 analysis completed on branch `codex/kalman-calibration-r24` from immutable
HEAD `f916f80afac5f5ca6d6558e4c3e87fba40831626`. At that analysis-completion
boundary, before later publication authorization, the worktree was intentionally
dirty and uncommitted; no production solver file, remote, commit, merge, Taichi/
CUDA/Fluent job, shadow run, active run, or R25 stage was changed or launched.

### 16.1 Locked evidence and model boundary

- D0 r47 fingerprint:
  `fa26a8b864ef9fbdb97bf6ad4040f326b6a8cd359b1d751f3a2405be967998bf`.
- D1 r51 fingerprint:
  `dced7627ea2a648c1434644f1b5c955b3f3b1119ce9c8eb1fd977e5a451fa46b`.
- Both traces use `dt_s=0.0005`, layout
  `373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164`,
  and 128-by-3 marker-velocity observations.
- The x component is an identically-zero plane-strain axis. It is preserved as
  zero, recorded as inactive, and excluded from normalized RMSE, NIS, gain, and
  root-cause aggregation. The statistical axes are y and z.
- Production K0 replay matched all 50 D1 history rows step-exactly: maximum
  absolute RMSE, bias, and NIS differences are all `0.0`. The source-only
  production adapter did not import Taichi.

### 16.2 Held-out decision

| Candidate | normalized RMSE | NIS mean | gain mean | result |
| --- | ---: | ---: | ---: | --- |
| C0 | 0.924019283 | n/a | n/a | best overall |
| K1 | 0.927728290 | 2.87706512 | 0.840214420 | statistically consistent, but 0.4014% worse than C0 and serially biased |
| C1a | 1.01103046 | n/a | n/a | best simple extrapolation |
| C1b | 1.15131676 | n/a | n/a | worse than C0 |
| C1c | 1.26967091 | n/a | n/a | worse than C0 |
| K2 | 1.04713552 | 5.59723468 | 0.755989048 | statistically inconsistent and 13.3240% worse than C0 |
| K0 | 1.27520848 | 121.007428 | 0.999958472 | degenerate gain and statistically inconsistent |

The production time index is reproduced exactly, so an off-by-one
implementation mismatch is not supported. The confirmed K0 failure is
under-dispersed innovation covariance on y/z plus persistent bias/serial model
lag on y/z. K1 repairs aggregate NIS to order unity but does not create held-out
predictive value relative to C0 and retains serial structure. No K1/K2 candidate
passes the promotion conjunction.

The single R24 exit classification is:

    FAIL_NO_KALMAN_PREDICTIVE_VALUE

This result stops the Kalman promotion path. R25 is not authorized. A future
non-Kalman C0/IQN convergence diagnosis would require a separate user-approved
goal; it is not implied or started here.

### 16.3 Final deterministic artifacts

Directory:

    validation_runs/kalman_statistical_calibration/
    ansys_vf__kalman__offline_calibration__20260831__r24

- `kalman_candidate_ranking.json`:
  `01ebe5b652a210e1e3ed347d28551136c7d782c232dab1e6681a8f127695f734`.
- `kalman_candidate_step_metrics.csv`:
  `d4fa3d6ad6ee629fde294b20e4ecc7afa1374f19910b9aa2bafa78206e20d8b5`.
- `kalman_data_split_manifest.json`:
  `81c7613c1e3a12e0d8f4294b39a22bff9afc6ea3bd49e4abcb16b6dfc830c35d`.
- `kalman_innovation_audit.json`:
  `729a3a488be08bdf839ffe87c8c4a85e5211997ff82cd81eb4b503111374a234`.

Two consecutive final-schema campaign invocations produced the same
classification and the same four SHA256 values. The CSV has 5250 data rows plus
one header; it separates the n-1 source SHA from the accepted measurement SHA
and records both candidate ID and model family. Each JSON artifact is finite,
schema-versioned, and self-fingerprinted. The exact campaign command and dirty
boundary are preserved in
`ANSYS_VERTICAL_FLAP_KALMAN_CALIBRATION_REPORT_2026-08-31.md`.

### 16.4 Verification and changed-file boundary

- `python3 -m pytest -q tests/validation/test_kalman_statistical_calibration.py`:
  23 passed.
- `python3 -m pytest -q tests/coupling/test_interface_kalman_predictor.py
  tests/coupling/test_interface_initial_guess_controller.py
  tests/coupling/test_active_kalman_writeback.py
  tests/validation/test_kalman_iqn_reuse_fine_contracts.py`: 111 passed; one
  pre-existing ragged-array deprecation warning.
- `python3 -m py_compile tools/audit_ansys_vertical_flap_kalman.py
  src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_*.py`:
  passed.
- The R24 CLI no-Taichi-import contract, restart equivalence, trial discard,
  provenance failure, executed-source SHA binding, axis schema, per-axis NIS
  rejection, K0 parity, source/measurement-row alignment, finite artifact, and
  deterministic replay gates are included in the 23-test focused suite.

Changed/untracked implementation files are the seven
`kalman_statistical_{campaign,evidence,filter,replay,reporting,selection,types}`
modules plus the `kalman_statistical_calibration.py` facade,
`tools/audit_ansys_vertical_flap_kalman.py`, the focused test file, this Goal,
the dated report, and `docs/README.md`. At the R24 analysis-completion
boundary, all were uncommitted; a later, separately user-authorized publication
may supersede that Git state without changing the R24 scientific evidence.
