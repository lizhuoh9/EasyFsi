# ANSYS Vertical-Flap Oracle Threshold and IQN First-Update Report

Date: 2026-09-01
Campaign: R24C
Status: COMPLETE

## 1. Decision

R24C reached all three predeclared terminal classifications:

- accepted displacement audit: `PASS_ACCEPTED_DISPLACEMENT_AUDIT`;
- discrete carry-to-oracle threshold matrix: `PASS_ORACLE_THRESHOLD_MATRIX`;
- conditional carry/oracle by IQN-reuse factor matrix:
  `PASS_IQN_REUSE_FACTOR_MATRIX`.

The best safe first Picard relaxation is `omega=0.75`. Carry-forward IQN
reuse reduced the exact-8 workload from 24 to 19 coupling trials, while the
same-step oracle reduced it to 8. This is mechanism evidence, not a
deployable predictor result. The terminal predictor decision is
`academic_offline_feasibility_only`, and the publication projection keeps
`deployable=false`.

The result prioritizes IQN first-update relaxation and accepted-history reuse
over a new full-state predictor. It does not authorize GRU, adaptive Kalman,
POD/DMD, active predictor writeback, exact50, Fluent comparison, or
200/500/5000-step continuation.

## 2. Scope and source boundary

All edits, Git reads, tests, audits, and numerical runs used WSL Ubuntu-22.04
at:

    /home/zhuohengli/worktrees/HIBM-MPM-r21-validation

The branch remained `codex/oracle-threshold-iqn-first-update-r24c`, with Git
HEAD at `b18bddadab384aec931328ebbd227e1368023a59`. R24C is a dirty
working-tree result; it is not an immutable R24C commit.

Two source identities are intentionally distinct:

| evidence boundary | source count | source-map SHA256 |
|---|---:|---|
| sealed R24B at `b18bddad...` | 129 | `de1f585bcbee9a2b176684d74c5b5c4c2c3100c602101a751d13d0f386985c0c` |
| R24C current source, dirty working tree | 139 | `84afaa15c7c4cc07ebceadbc141cb087c53a00efb71f6ca2e81b787649f350d8` |

The accepted-displacement artifact retains its earlier `reusefix`-named path,
but it is explicitly sealed to the immutable R24B `129/de1f...` identity and
passes commit-bound bottom-up verification. All formal R24C Q0, threshold,
probe, and reuse-factor roots use `r24c_normalfix`. Earlier R24C `reusefix`
and `rematch` numerical roots are diagnostic only because they predate the
current audit-source identity.

The formal R24C audit CLI SHA256 is
`9fb5672fe7c3309cc4cf0a913f6e9a949070bd0eed90fa526e58857aee760c93`.

## 3. Implemented audit and transaction contracts

The implementation added the following bounded research surfaces without
changing the production FSI equations:

- displacement-relative marker and solid comparisons, including frozen
  marker-reference reconstruction, tip displacement, identity, and absolute
  error checks;
- a no-commit target-step probe that restores the accepted host state in
  `finally`, uses a fresh IQN runtime per alpha, and proves equal rollback
  state after each trial and after the sweep;
- source-, runtime-, preflow-, configuration-, and lineage-bound threshold
  evidence for three Q0 arms and nine terminal probes;
- conditional carry/oracle by reuse off/on exact-8 evidence, including source
  step, reset, fallback, accepted-time, and work counters;
- a path-free publication projection that excludes local paths and
  credential-like content while declaring `bottom_up_reverification=false`;
- CPU-only workflow gates for compile, focused tests, audit CLI help, and
  committed-range whitespace checking.

A factor-arm audit initially compared the deformed `marker_normal` array
bitwise across arms. The solver recomputes those normals from deformed
geometry, so this was not a valid immutable-identity field. RED tests captured
the failure. The corrected contract keeps layout, region, and area exact and
checks normals with both NRMSE and maximum-absolute gates. The observed worst
normal difference was NRMSE `5.961969690262958e-06` and max absolute
`2.4123117327690125e-05`, at carry/reuse-on step 5, within
the existing `5e-5` dual bound. All current-source CUDA roots were rerun under
new `normalfix` names after this source change; no numerical artifact was
re-signed or overwritten.

## 4. Accepted displacement audit

Formal artifact:

    validation_runs/kalman_oracle_headroom/
      ansys_vf__oracle_displacement__r24c_reusefix__20260901__r01.json

Its sealed producer roots are:

    validation_runs/solver_soaks/
      ansys_vf__oracle_q0__material_fine__20260831__r24b_final_contract
      ansys_vf__oracle_q3__material_fine__20260831__r24b_final_contract
    validation_runs/kalman_oracle_headroom/
      ansys_vf__oracle_headroom__20260831__r24b_final_contract

The artifact binds 64 physical markers per face, 128 physical markers total,
and 128 interface rows. Its `5e-3` displacement-NRMSE gate passed:

| metric | maximum |
|---|---:|
| marker displacement NRMSE | `6.873413338857167e-06` |
| solid displacement NRMSE | `5.317370943492224e-06` |
| marker position absolute error | `3.725290298461914e-09 m` |
| solid position absolute error | `3.725290298461914e-09 m` |
| mean-tip displacement-vector error | `9.313225746155579e-11 m` |
| tip-displacement amplitude error | `2.6324233591557e-13 m` |

File SHA256:
`84ee846dc09ec30607eedd21e2f2ecbd0206e594cf17d79d28451b78d219f98f`.

## 5. Frozen strict-CUDA matrix

The shared fresh preflow is:

    validation_runs/solver_soaks/
      ansys_vf__preflow__material_fine__20260901__r24c_final

It has zero accepted steps and physical time zero. Its identities are:

- executable source: `9472a46e1424ece008ab53935b8e6468e703dd10e27e4b78ea1d98ce50736f7e`;
- config: `16298a97d45eb03639e7f3d1c7f4048386d66bd573af55c7e485ceedabc4783f`;
- geometry: `eb28d5616a3ac1f270c2ceae286534a387c5176873086cbd5105d06dae568bbb`;
- preflow NPZ: `31bd9a970854eb2c820cb325a754bca1f4a814db6dbfce00d82f6d934ce3a9cd`.

Every formal numerical arm used actual strict CUDA, Taichi 1.7.4, float32,
seed 0, and the frozen fine material configuration. Only one expensive CUDA
job ran at a time. Each accepted physical step consumed the full
`dt_s=5e-4 s` in both fluid and solid; each exact-8 arm ended at `0.004 s`.

The three formal Q0 roots are:

    ansys_vf__threshold_q0_o050__r24c_normalfix__20260901__r01
    ansys_vf__threshold_q0_o075__r24c_normalfix__20260901__r01
    ansys_vf__threshold_q0_o100__r24c_normalfix__20260901__r01

| omega | trials | rejected | pressure CG | trials by accepted step |
|---:|---:|---:|---:|---|
| 0.50 | 24 | 16 | 5728 | `3,3,3,3,3,3,3,3` |
| 0.75 | 24 | 16 | 5728 | `3,3,3,3,3,3,3,3` |
| 1.00 | 19 | 11 | 4528 | `3,3,2,2,2,2,2,3` |

## 6. Nine no-commit probes and discrete thresholds

The ordered alpha samples were:

    0.9, 0.95, 0.975, 0.99, 0.995,
    0.996, 0.9975, 0.998, 0.999, 1.0

All 90 alpha trials converged. Every per-alpha rollback digest and final sweep
state matched its accepted prefix. Target steps 2, 5, and 8 retained accepted
prefixes of 1, 4, and 7 steps and times `0.0005`, `0.002`, and `0.0035 s`;
none committed the target step.

| omega | target step | trials over ordered alpha samples | first 3-to-2 | first 2-to-1 |
|---:|---:|---|---:|---:|
| 0.50 | 2 | `3,3,3,3,3,3,2,2,1,1` | 0.9975 | 0.999 |
| 0.50 | 5 | `3,3,2,1,1,1,1,1,1,1` | 0.975 | 0.99 |
| 0.50 | 8 | `3,3,3,3,3,3,3,3,2,1` | 0.999 | 1.0 |
| 0.75 | 2 | `3,3,3,3,2,2,2,2,1,1` | 0.995 | 0.999 |
| 0.75 | 5 | `3,2,2,1,1,1,1,1,1,1` | 0.95 | 0.99 |
| 0.75 | 8 | `3,3,3,3,3,3,2,2,2,1` | 0.9975 | 1.0 |
| 1.00 | 2 | `2,2,2,2,2,2,2,2,1,1` | 0.9 | 0.999 |
| 1.00 | 5 | `2,2,2,1,1,1,1,1,1,1` | 0.9 | 0.99 |
| 1.00 | 8 | `2,2,2,2,2,2,2,2,2,1` | 0.9 | 1.0 |

These are discrete sampled thresholds. No interpolation is claimed. All three
omega values met the frozen safety criteria; omega 0.75 was selected because
it reduced all three targets and had a lower worst first 3-to-2 threshold
(`0.9975`) than omega 0.5, without choosing the larger relaxation of 1.0.

The nine formal probe roots are the Cartesian set
`ansys_vf__threshold_probe_o{050,075,100}_s{02,05,08}` with suffix
`__r24c_normalfix__20260901__r01`, under `validation_runs/solver_soaks/`.

Threshold evidence root:

    validation_runs/kalman_oracle_headroom/
      ansys_vf__oracle_threshold__r24c_normalfix__20260901__r01

Artifact SHA256 values:

- response: `f61224f5a110dcf93de5fd71e99e7bc3adb947a63c0f7abcfa77464d6e8af6d5`;
- source manifest: `3f2b05f4918a84052480e85ebc9495274ed042e38c516dc63619a8cac6d859c0`;
- summary: `92a77a62cc4d9e9b48756ae58779cc31a8fffe80c6d484ee83e4fdca4b2487ba`.

## 7. Conditional IQN-reuse factor

The conditional branch was authorized because omega 0.75 was safe, reduced
all three targets, and therefore met `safe_higher_first_picard_relaxation`.
Carry/reuse-off reuses the omega-0.75 Q0 root. The additional roots are:

    ansys_vf__reuse_carry_on_o075__r24c_normalfix__20260901__r01
    ansys_vf__reuse_oracle_off_o075__r24c_normalfix__20260901__r01
    ansys_vf__reuse_oracle_on_o075__r24c_normalfix__20260901__r01

| arm | trials | rejected | pressure CG | actual reuse steps |
|---|---:|---:|---:|---|
| carry, reuse off | 24 | 16 | 5728 | none |
| carry, reuse on | 19 | 11 | 4528 | 2 through 8 |
| oracle, reuse off | 8 | 0 | 1920 | none |
| oracle, reuse on | 8 | 0 | 1920 | none |

Carry reuse reduced trials by `20.8333%`, rejected trials by `31.25%`, and
pressure CG work by `20.9497%`. It also reduced momentum, SST, and solid
substep work. The oracle arms converged on their first trial at every step, so
the reuse-on oracle arm had no accepted secant pair to consume. No arm reported
a reset or fallback.

Raw solver elapsed times were `348.907/313.913 s` for carry off/on and
`173.109/173.312 s` for oracle off/on. They are recorded only as diagnostics;
they do not establish production or wall-time acceleration.

Reuse artifact:

    validation_runs/kalman_oracle_headroom/
      ansys_vf__iqn_reuse_factor__r24c_normalfix__20260901__r01.json

File SHA256:
`43ec674f82c8e7e09463b31fb91204b449f9ea26a5e2751ff4f59efa26e6772d`.
The path-free publication projection does not embed this reuse artifact, so
this SHA is a separate formal binding.

## 8. Publication projection

The path-free projection is:

    validation_runs/kalman_oracle_headroom/
      ansys_vf__oracle_threshold_publication__r24c_normalfix__20260901__r01.json

Its SHA256 is
`940f3e42cc6eeb2e3c4ae88c48b1108dc7f137714a8775920471b7a1b1fbfffd`.
It binds the displacement, response, source-manifest, and threshold-summary
hashes above, contains no absolute user path, declares portability, preserves
`deployable=false`, and correctly declares `bottom_up_reverification=false`.
The original local artifacts, not this projection, remain the bottom-up
verification source.

## 9. Engineering verification

The final focused command used `/usr/bin/python3` in the authoritative WSL
working tree:

```bash
python3 -B -m pytest -q \
  tests/validation/test_kalman_statistical_calibration.py \
  tests/validation/test_kalman_oracle_headroom.py \
  tests/validation/test_kalman_oracle_headroom_fail_closed.py \
  tests/validation/test_kalman_iqn_reuse_fine_contracts.py \
  tests/validation/test_oracle_threshold_iqn_first_update.py \
  tests/validation/test_oracle_threshold_lineage.py \
  tests/validation/test_oracle_threshold_publication.py \
  tests/validation/test_oracle_threshold_reuse_evidence.py \
  tests/validation/test_our_solver_vertical_flap_runner.py \
  tests/integration/test_ansys_vertical_flap_runner_loop_contract.py::AnsysVerticalFlapRunnerLoopContractTests::test_iqn_runner_maps_generic_threshold_audit_histories \
  tests/integration/test_ansys_vertical_flap_runner_loop_contract.py::AnsysVerticalFlapRunnerLoopContractTests::test_research_probe_recaptures_and_compares_after_each_rollback \
  tests/integration/test_ansys_vertical_flap_runner_loop_contract.py::AnsysVerticalFlapRunnerLoopContractTests::test_research_probe_rejects_iqn_history_reuse \
  tests/integration/test_ansys_vertical_flap_runner_loop_contract.py::AnsysVerticalFlapRunnerLoopContractTests::test_research_probe_terminal_satisfies_official_report_contract \
  tests/tools/test_calibrate_iqn_kalman_qr.py \
  tests/coupling/test_interface_kalman_predictor.py \
  tests/coupling/test_active_kalman_writeback.py \
  tests/benchmarks/test_modified_physics_kalman_contract.py
```

Result: `336 passed, 1 warning in 91.20 s`. The warning is the existing NumPy
`VisibleDeprecationWarning` for a deliberately ragged invalid-input test at
`simulation_core/coupling/interface_kalman_predictor.py:665`.

Additional final gates:

- the workflow's explicit R24-through-R24C `py_compile` list exited 0;
- `python3 -B -m unittest -v tests.integration.test_validation_ci_workflow`
  passed 5/5 tests;
- all three audit CLIs returned help with exit 0 and no Taichi initialization;
- bottom-up displacement, threshold, and reuse verification returned the three
  PASS classifications and the exact hashes above;
- publication portability and raw-hash binding passed focused tests.

The exact compile, workflow-contract, help, and bottom-up verification
commands were:

```bash
PYTHONPYCACHEPREFIX=/tmp/r24c-normalfix-pycache \
/usr/bin/python3 -B -m py_compile \
  benchmarks/official/solid_mpm_fsi_runner.py \
  src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_calibration.py \
  src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom.py \
  src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_analysis.py \
  src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_artifacts.py \
  src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_contracts.py \
  src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_integrity.py \
  src/refactored/validation/ansys_vertical_flap_fsi/kalman_oracle_headroom_verification.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_common.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_probe_contracts.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_iqn_first_update.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_displacement_evidence.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_evidence.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_lineage.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_prefix_decisions.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_publication.py \
  src/refactored/validation/ansys_vertical_flap_fsi/oracle_threshold_reuse_evidence.py \
  validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/scripts/run_our_solver_vertical_flap.py \
  tools/audit_ansys_vertical_flap_kalman.py \
  tools/audit_ansys_vertical_flap_oracle_headroom.py \
  tools/audit_ansys_vertical_flap_oracle_threshold.py \
  tests/validation/test_kalman_statistical_calibration.py \
  tests/validation/test_kalman_oracle_headroom.py \
  tests/validation/test_kalman_oracle_headroom_fail_closed.py \
  tests/validation/test_oracle_threshold_iqn_first_update.py \
  tests/validation/test_oracle_threshold_lineage.py \
  tests/validation/test_oracle_threshold_publication.py \
  tests/validation/test_oracle_threshold_reuse_evidence.py \
  tests/validation/test_our_solver_vertical_flap_runner.py \
  tests/integration/test_ansys_vertical_flap_runner_loop_contract.py

/usr/bin/python3 -B -m unittest -v \
  tests.integration.test_validation_ci_workflow

/usr/bin/python3 -B tools/audit_ansys_vertical_flap_kalman.py --help
/usr/bin/python3 -B tools/audit_ansys_vertical_flap_oracle_headroom.py --help
/usr/bin/python3 -B tools/audit_ansys_vertical_flap_oracle_threshold.py --help

/usr/bin/python3 -B tools/audit_ansys_vertical_flap_oracle_threshold.py \
  verify-displacement \
  --output validation_runs/kalman_oracle_headroom/ansys_vf__oracle_displacement__r24c_reusefix__20260901__r01.json
/usr/bin/python3 -B tools/audit_ansys_vertical_flap_oracle_threshold.py \
  verify-threshold \
  --output-dir validation_runs/kalman_oracle_headroom/ansys_vf__oracle_threshold__r24c_normalfix__20260901__r01
/usr/bin/python3 -B tools/audit_ansys_vertical_flap_oracle_threshold.py \
  verify-reuse \
  --output validation_runs/kalman_oracle_headroom/ansys_vf__iqn_reuse_factor__r24c_normalfix__20260901__r01.json
```

Running the old R24B verifier directly from the dirty R24C source tree fails
closed on the current runner-source SHA, as designed. This is not a numerical
R24B failure. The R24C displacement verifier instead checks the sealed R24B
bundle and source map at immutable commit `b18bddad...`; that commit-bound
bottom-up verification passed.

The broad legacy comparison used a `git archive` snapshot of immutable base
`b18bddad...`, the dirty R24C working tree, `/usr/bin/python3`, pytest 9.1.1,
the same 12 test files, and the node list collected only from the base. It
compared `PASSED`, `FAILED`, `ERROR`, `SKIPPED`, `XFAIL`, `XPASS`, and
`NOT_COLLECTED` by node ID.

The exact 12-file list was:

```text
tests/validation/test_kalman_oracle_headroom.py
tests/validation/test_kalman_oracle_headroom_fail_closed.py
tests/coupling/test_accepted_fsi_checkpoint.py
tests/coupling/test_interface_controller_restart.py
tests/coupling/test_interface_initial_guess_controller.py
tests/integration/test_ansys_vertical_flap_fsi_checkpoint.py
tests/integration/test_ansys_vertical_flap_preflow_snapshot.py
tests/validation/test_current_iqn_adaptive_fine_contracts.py
tests/validation/test_kalman_iqn_reuse_fine_contracts.py
tests/validation/test_our_solver_vertical_flap_runner.py
tests/cases/test_ansys_vertical_flap_fsi.py
tests/benchmarks/test_vertical_flap_sst_runner_contract.py
```

The exact comparison commands were:

```bash
/usr/bin/python3 -B \
  /tmp/r24c-broad-base.xyyFID/harness/broad_matrix_harness.py collect \
  --root /tmp/r24c-broad-base.xyyFID/source \
  --output /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/base-nodeids.json \
  --log /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/base-collect.log \
  --tmpdir /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/tmp-base-collect

/usr/bin/python3 -B \
  /tmp/r24c-broad-base.xyyFID/harness/broad_matrix_harness.py run \
  --root /tmp/r24c-broad-base.xyyFID/source \
  --nodes /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/base-nodeids.json \
  --output /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/base-outcomes.json \
  --log /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/base.pytest.log \
  --junit /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/base.junit.xml \
  --tmpdir /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/tmp-base

/usr/bin/python3 -B \
  /tmp/r24c-broad-base.xyyFID/harness/broad_matrix_harness.py run \
  --root /home/zhuohengli/worktrees/HIBM-MPM-r21-validation \
  --nodes /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/base-nodeids.json \
  --output /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/current-working-tree-outcomes.json \
  --log /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/current-working-tree.pytest.log \
  --junit /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/current-working-tree.junit.xml \
  --tmpdir /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/tmp-current

/usr/bin/python3 -B \
  /tmp/r24c-broad-base.xyyFID/harness/broad_matrix_harness.py compare \
  --base /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/base-outcomes.json \
  --current /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/current-working-tree-outcomes.json \
  --output /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_oracle_headroom/ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01/comparison.json
```

Evidence root:

    validation_runs/kalman_oracle_headroom/
      ansys_vf__broad_legacy_matrix__r24c_normalfix__20260901__r01

Result: `PASS_IDENTICAL_NODE_OUTCOMES`, 486 requested nodes, no differences,
no uncollected nodes, and identical counts on both sides:
`474 PASSED / 11 FAILED / 1 XFAIL`. The 11 failures are unchanged legacy
outcomes, so this is a scoped non-regression statement, not a full-suite-pass
claim. Node-list SHA256 is
`c5397c48b34cc99f159497952f9396198fb861edae19b495fe4e2824d48feadc`;
comparison JSON SHA256 is
`2dae1a6f6980f7f9a06e6a1f53a5b05496efbf46a0429716efd9669584a7965d`.

Final `git diff --check` produced no errors. All 16 untracked files were also
checked individually with `git diff --no-index --check` and produced no
whitespace diagnostics. The exact branch, HEAD, tracked modifications, and
untracked files were audited without staging. A fresh read-only Sol/Ultra
review inspected the complete change set, found no P0, P1, or P2 actionable
issue, and returned `ship`.

## 10. Interpretation and stop boundary

The measured threshold surface shows that a safer first update and accepted
IQN history can remove one trial on most accepted steps without inventing a
new state predictor. The perfect same-step oracle still has large academic
headroom, but it is non-causal and non-deployable. R24C therefore stops at
`academic_offline_feasibility_only` for any later modal or learned model.

This report does not claim:

- a deployable Kalman, GRU, POD/DMD, or other predictor;
- production, end-to-end, or wall-time acceleration;
- exact50 success, Fluent parity, or a 200/500/5000-step result;
- a full repository test pass, remote CI pass, or GitHub required check.

## 11. Git publication state

At the validation-completion gate, no commit, push, merge, pull request, or
remote CI run had been performed. The user subsequently gave explicit current
authorization to publish the completed code and documentation. The authorized
destination is a non-force push to
`origin/codex/oracle-threshold-iqn-first-update-r24c`. Ignored local numerical
roots remain local and are not part of the Git commit. This report does not
treat the push itself as numerical validation or remote CI evidence; the final
handoff must verify the remote commit before claiming publication success.

## 12. R24C.1 CI attestation closure (2026-09-01)

R24C.1 reviewed baseline aef06288d098ac5e674cdec381a5a41df13eff8b in the
authoritative WSL checkout

    /home/zhuohengli/worktrees/HIBM-MPM-r21-validation

on branch codex/r24c1-ci-attestation. Historical GitHub run 33465020431 had
conclusion=failure. Its four exact local CI root-cause categories were stale
particle-bin consumer/generation assertions, missing preflow runtime
generation/support-radius capture, a solid-substep test targeting the retired
batched helper instead of the selected macro-step report, and an incomplete
removed-backend test namespace that failed before the intended full-config
validation.

The local TDD sequence began RED with 3 failures and 1 error. The corrected
five-node semantic surface is now OK: the two original particle-bin generation
tests, the preflow runtime-capture test, the solid-substeps runtime test, and
the removed-backend full-config test all passed. The original focused sealer
gate passed 19 tests; the focused suite is now expanded and passes 21 tests.
The core, contracts helper, CLI, and focused test all pass py_compile. The
current WSL environment has no ruff module, so no local ruff result is claimed.

The real read-only CPU bottom-up helper returned:

- displacement: PASS_ACCEPTED_DISPLACEMENT_AUDIT;
- threshold: PASS_ORACLE_THRESHOLD_MATRIX;
- reuse: PASS_IQN_REUSE_FACTOR_MATRIX with status reuse_matrix_authorized.

The six verified artifact hashes are:

- displacement: 84ee846dc09ec30607eedd21e2f2ecbd0206e594cf17d79d28451b78d219f98f;
- threshold response: f61224f5a110dcf93de5fd71e99e7bc3adb947a63c0f7abcfa77464d6e8af6d5;
- threshold source manifest: 3f2b05f4918a84052480e85ebc9495274ed042e38c516dc63619a8cac6d859c0;
- threshold summary: 92a77a62cc4d9e9b48756ae58779cc31a8fffe80c6d484ee83e4fdca4b2487ba;
- IQN reuse: 43ec674f82c8e7e09463b31fb91204b449f9ea26a5e2751ff4f59efa26e6772d;
- legacy projection: 940f3e42cc6eeb2e3c4ae88c48b1108dc7f137714a8775920471b7a1b1fbfffd.

The formal source map remains exactly 139 entries with canonical SHA256
84afaa15c7c4cc07ebceadbc141cb087c53a00efb71f6ca2e81b787649f350d8, and every
current-checkout byte was checked. The schema-2 legacy projection was read
only and never overwritten. The future schema-3 projection keeps
deployable=false and bottom_up_reverification=false; its terminal attestation
has not been generated.

The remote Ubuntu/Windows green result and clean-final-commit acceptance are
pending. Once authorized on a clean checkout, the seal must produce and
immediately verify the schema-3 projection plus terminal attestation. No
attestation is claimed in this report. R24D, R24E, adaptive Kalman, GRU,
CUDA/numerical reruns, Fluent parity, exact50, and long-run claims remain
explicitly excluded.
