# ANSYS Vertical-Flap Oracle Threshold and IQN First-Update Goal

Date: 2026-08-31
Status: COMPLETE
Campaign: R24C

## 1. Decision questions

R24B proved that a non-deployable same-step oracle can reduce the exact-8
coupling workload from 24 trials to 8. It also showed that alpha values up to
0.75 reduce the first residual without reducing the discrete trial count.
R24C must answer three narrower questions before any new predictor is designed:

1. Does the Q3 accepted structure remain close to Q0 when position error is
   normalized by displacement rather than by absolute coordinates?
2. Along the favorable carry-to-oracle direction, where are the measured
   3-to-2 and 2-to-1 trial transitions?
3. Is the current first Picard relaxation of 0.5 a larger opportunity than a
   learned or adaptive iteration-zero predictor?

This is a mechanism and threshold campaign. It does not authorize a GRU,
adaptive covariance, a modal predictor, active predictor writeback, exact50,
Fluent comparison, or a long-horizon run.

## 2. Authoritative state and publication boundary

All Git operations, edits, tests, artifact reads, and numerical runs use WSL
Ubuntu-22.04 at:

    /home/zhuohengli/worktrees/HIBM-MPM-r21-validation

R24C starts from:

    branch: codex/oracle-threshold-iqn-first-update-r24c
    starting commit: b18bddadab384aec931328ebbd227e1368023a59

The Windows checkout and other WSL checkouts are not sources of truth. The
starting worktree was clean. Preserve unrelated files and every historical
validation root. Do not reset, clean, overwrite an output root, or restart
from reduced step fields.

Commit, push, merge, PR creation, and remote CI claims remain outside this Goal
unless the user gives separate current authorization. Local numerical
artifacts remain ignored under `validation_runs/`.

## 3. Inherited evidence and source boundary

The immutable R24B result at the starting commit remains:

- R24: `FAIL_NO_KALMAN_PREDICTIVE_VALUE` for K0/K1/K2;
- R24B: `PASS_ORACLE_HEADROOM` for a non-deployable exact oracle;
- Q0/Q3 total trials: 24/8;
- Q0/Q3 pressure CG iterations: 5728/1920;
- alpha 0.25, 0.50, and 0.75: 24 trials each.

The final local R24B pair used for the first CPU-only audit is:

    validation_runs/solver_soaks/
      ansys_vf__oracle_q0__material_fine__20260831__r24b_final_contract
      ansys_vf__oracle_q3__material_fine__20260831__r24b_final_contract

R24C must not rewrite or re-sign the R24B four-file bundle. New R24C modules
first call the existing bottom-up R24B verifier, then derive separately named
displacement and threshold artifacts. This keeps the sealed R24B audit source
unchanged. A change to the formal runner, however, changes the executable
preflow source identity and therefore requires a fresh source-matched preflow
before any R24C CUDA probe.

## 4. Stage A: accepted displacement audit

Write failing synthetic tests before implementation. The analysis is
Taichi-free and must use the existing exact-8 step fields and histories without
running CUDA.

For solid particles at each accepted step:

    d0 = x_Q0 - x_rest_Q0
    d3 = x_Q3 - x_rest_Q3
    displacement NRMSE = RMSE(d3 - d0) / max(RMS(d0), 1e-12 m)

The same calculation is reported per axis together with each reference-axis
RMS. A near-zero axis must also report absolute RMSE and may not be made to
look accurate solely by the denominator floor.

Required per-step metrics are:

- legacy absolute-coordinate marker and solid position NRMSE;
- solid displacement-relative NRMSE and per-axis values;
- marker displacement-relative NRMSE and per-axis values when a rigorously
  bound marker reference is available;
- maximum marker and solid position absolute error in metres;
- Q0/Q3 mean tip displacement vectors;
- tip displacement vector error and amplitude error in metres;
- physical marker count and interface-state row count.

`solid_rest_position_m` and `solid_tip_mask` are mandatory bound arrays.
The marker reference may only be reconstructed from the frozen dual-face
geometry, frozen config, and verified executable/preflow identity, or loaded
from a real bound artifact. Step 1 may not stand in for the preflow reference.
If this reference cannot be proved, marker displacement NRMSE must fail closed
or be explicitly unavailable while the absolute marker error remains reported.

The accepted displacement gate is `max solid displacement NRMSE <= 5e-3`.
If marker displacement NRMSE is available, the same limit applies. Maximum
absolute and tip errors are reported without inventing a new physics tolerance.
The existing R24B accepted-state and health gates remain unchanged.

The frozen configuration has 64 markers per physical face and 128 physical
interface rows for the dual-face layout. R24C artifacts must name these
`physical_marker_count_per_face` and `interface_state_row_count`, validate
the actual array shape, and never use one value as the other.

## 5. Stage B: no-commit probe contract

Add RED tests, then make the smallest research-only extension to the existing
oracle-interpolation probe. Do not change the production FSI equations,
same-time IQN mathematics, fluid, solid, pressure, closure, checkpoint format,
or any active predictor.

The R24C control endpoint is the accepted carry-forward marker velocity. For a
target step and its Q0 same-step accepted oracle:

    x0(alpha) = x_carry + alpha * (x_oracle - x_carry)

Every alpha starts from the same captured accepted `HostMacroStepState`.
Each uses a fresh copied IQN runtime. The research path must suppress step
observers, checkpoints, accepted history, and controller acceptance. It must
rollback in `finally`, restore the accepted base after the whole sweep, and
serialize a before/after equality digest or equivalent complete equality
check.

The terminal result must prove:

- `accepted_step_count == target_step - 1`;
- `accepted_time_s == (target_step - 1) * dt_s`;
- no target-step accepted commit;
- equal accepted state before and after every probe and after the sweep.

For each alpha, record coupling iterations, the first two absolute and relative
residuals, effective tolerance and residual/tolerance ratios, update modes,
IQN rank and condition number, fallback and update-limited state, pressure CG,
fluid and solid solves, momentum/SST/solid substeps, solid trial reports, and
synchronized component wall time as a diagnostic only. Missing or non-finite
decision fields, convergence failure, rollback inequality, or accepted-base
drift fail closed.

Dry-run config, CLI, and run manifests must bind the target step, oracle root,
ordered alpha list, baseline mode, omega, reuse flag, source, runtime, preflow,
layout, and `deployable=false`.

## 6. Frozen strict-CUDA matrix

The numerical identity is the R24B fine material case:

- actual strict CUDA, Taichi 1.7.4, float32, seed 0;
- `dt_s=5e-4`, grid `[4,256,320]`, solid `[1,256,20]`;
- 64 markers per face and 128 interface rows;
- adaptive solid substeps, CFL target 0.14;
- IQN maximum 16, relative tolerance `1e-3`, absolute tolerance 0;
- FV-Jacobi pressure, FV multigrid preconditioner, CG tolerance `1e-6`;
- marker compatibility closure `1.1e-6 m/s`;
- predictor writeback off and previous-step IQN reuse off.

If the old preflow executable identity differs after the research runner
change, preserve the zero-step rejection and generate a fresh preflow. Never
weaken the source check.

Target steps:

    2, 5, 8

Ordered alphas:

    0.9000, 0.9500, 0.9750, 0.9900, 0.9950,
    0.9960, 0.9975, 0.9980, 0.9990, 1.0000

First Picard relaxation values:

    0.50, 0.75, 1.00

The matrix contains nine terminal runs. Each run normally accepts the prefix
through `target-1`, then evaluates all ten alpha values from one accepted
base without commit. Use unique non-overwriting roots and run exactly one
expensive CUDA job at a time.

For each target/omega pair:

- `alpha_3_to_2` is the smallest sampled alpha with at most two trials;
- `alpha_2_to_1` is the smallest sampled alpha with one trial;
- the value is null when no sampled point qualifies.

These are discrete sampled thresholds. Do not interpolate a more precise
number. An omega is called safe only when all probes converge with complete
rollback/physics/work evidence, alpha 0 is not worse than omega 0.5, and at
least two of three targets reduce carry-forward from three trials to at most
two.

## 7. Conditional reuse branch and stop tree

A carry/oracle by IQN-reuse off/on small source-matched factor experiment is
authorized only when:

- omega 0.75 or 1.0 is safe and reduces carry-forward on most targets; or
- the best safe omega has `alpha_3_to_2 <= 0.9900`.

Otherwise report `reuse_matrix_not_authorized` and stop that branch.

If safe first-update or reuse reduces most steps from three to two and leaves
little oracle headroom, prioritize IQN first-update/reuse and stop full-state
predictor acceleration. If the best-safe 3-to-2 threshold remains at or above
0.995, or most targets have no transition, stop expectations of an active GRU
or full-state predictor. POD-AR, DMD, modal Kalman, and a tiny POD-GRU may only
be considered later as a separately authorized offline academic feasibility
Goal using the measured R24C threshold.

Any source, config, runtime, preflow, layout, shape, accepted-time, rollback, or
health failure produces an evidence error or STOP. Favorable subsets may not
override the frozen matrix.

## 8. Engineering gates and evidence

Add a CPU-only R24/R24B/R24C step to the existing Ubuntu fast workflow:

- explicit `py_compile` for the audit modules, CLIs, and focused tests;
- focused R24/R24B/R24C pytest files;
- all audit CLIs with `--help` and no Taichi initialization;
- a committed-range `git diff --check`.

Update the workflow contract test and keep this work out of the scheduled CUDA
job. A local workflow file is not a claim that GitHub required checks exist.

Reproduce the broad legacy matrix on base `b18bdda` and the R24C working tree
with the same interpreter and test list. Only identical node outcomes support
a scoped non-regression statement. Without an immutable R24C commit, label the
head side as a working-tree comparison rather than an immutable base/head
proof. Do not repair unrelated legacy failures.

Keep the original absolute-root, bottom-up-verifiable evidence local. Create a
separate path-free publication projection with logical arm IDs, source commit,
raw artifact SHA256 values, metrics, gates, definitions,
`deployable=false`, and a portability declaration. It must say
`bottom_up_reverification=false` unless the full approximately 149 MiB
dependency closure is separately packaged. Tests must reject absolute user
paths and credential-like content.

Required tracked documentation:

- this Goal;
- `ANSYS_VERTICAL_FLAP_ORACLE_THRESHOLD_IQN_FIRST_UPDATE_REPORT_2026-08-31.md`;
- the targeted entry in `docs/README.md`.

Required local machine evidence:

- R24C source manifest;
- displacement step metrics;
- threshold/omega summary;
- complete threshold/omega response.

## 9. Verification and completion

Run, in order:

1. focused RED/GREEN tests;
2. R24/R24B/R24C and transaction-related tests;
3. workflow contract tests;
4. CPU-only compile and CLI help;
5. bottom-up R24B plus R24C evidence verification;
6. base versus working-tree broad outcome comparison;
7. `git diff --check` and exact status/diff audit;
8. code review with no P0/P1;
9. a fresh read-only Sol/Ultra review.

A `fix-first` review requires RED/GREEN repair and a different fresh
Sol/Ultra reviewer. Mark this Goal complete only after every predeclared
branch has a terminal result and the final reviewer says `ship`.

The final report must include exact displacement results, each target/omega
threshold, whether the reuse branch ran and why, exact commands and test
counts, artifact hashes and roots, prohibited claims, and Git publication
state.

## 10. Completion evidence

All predeclared implementation and numerical branches have terminal results:

- displacement: `PASS_ACCEPTED_DISPLACEMENT_AUDIT`;
- threshold: `PASS_ORACLE_THRESHOLD_MATRIX`, best safe `omega=0.75`;
- conditional reuse: authorized by `safe_higher_first_picard_relaxation` and
  completed as `PASS_IQN_REUSE_FACTOR_MATRIX`;
- predictor decision: `academic_offline_feasibility_only`,
  `deployable=false`.

The sealed R24B identity remains 129 source files with source-map SHA256
`de1f585bcbee9a2b176684d74c5b5c4c2c3100c602101a751d13d0f386985c0c`.
Formal current-source R24C evidence is bound to 139 source files with
source-map SHA256
`84afaa15c7c4cc07ebceadbc141cb087c53a00efb71f6ca2e81b787649f350d8`.
R24C remains a dirty working-tree result at Git HEAD `b18bddad...`, not an
immutable R24C commit.

The formal strict-CUDA campaign completed three exact-8 Q0 arms, nine
no-commit probes, and the four carry/oracle by reuse off/on arms. All accepted
fluid and solid physical times consumed the full macro `dt_s`; all source,
preflow, configuration, rollback, lineage, and health gates passed. Formal
current-source numerical roots use the `r24c_normalfix` suffix. Earlier
current-source-mismatched R24C roots are diagnostic only.

Formal artifact SHA256 values are:

- displacement: `84ee846dc09ec30607eedd21e2f2ecbd0206e594cf17d79d28451b78d219f98f`;
- threshold response: `f61224f5a110dcf93de5fd71e99e7bc3adb947a63c0f7abcfa77464d6e8af6d5`;
- threshold source manifest: `3f2b05f4918a84052480e85ebc9495274ed042e38c516dc63619a8cac6d859c0`;
- threshold summary: `92a77a62cc4d9e9b48756ae58779cc31a8fffe80c6d484ee83e4fdca4b2487ba`;
- IQN reuse factor: `43ec674f82c8e7e09463b31fb91204b449f9ea26a5e2751ff4f59efa26e6772d`;
- path-free publication: `940f3e42cc6eeb2e3c4ae88c48b1108dc7f137714a8775920471b7a1b1fbfffd`.

Final engineering evidence is:

- focused pytest: `336 passed, 1 warning`;
- explicit `py_compile`: exit 0;
- workflow contract: 5/5 passed;
- all three audit CLI help commands: exit 0 without Taichi initialization;
- bottom-up displacement, threshold, and reuse verification: PASS;
- base versus dirty working-tree broad matrix:
  `PASS_IDENTICAL_NODE_OUTCOMES`, 486 nodes, no differences, with identical
  `474 PASSED / 11 FAILED / 1 XFAIL` outcomes.

At the validation-completion gate, no commit, push, merge, PR, or remote CI
action had been performed. The user subsequently gave explicit current
authorization to publish the completed code and documentation to GitHub. The
authorized destination is the non-force `origin` push of branch
`codex/oracle-threshold-iqn-first-update-r24c`; ignored local numerical roots
are outside the Git publication scope. Final `git diff --check` and the exact
status/diff audit passed. The fresh read-only Sol/Ultra review found no P0,
P1, or P2 actionable issue and returned `ship`. All predeclared branches
therefore have a terminal result and this Goal is complete.
