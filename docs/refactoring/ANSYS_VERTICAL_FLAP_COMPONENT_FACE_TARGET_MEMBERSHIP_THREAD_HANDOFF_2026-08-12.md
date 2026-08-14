---
status: completed
branch: codex/fix-main-audit-2026-08-10
timestamp: 2026-08-12T14:51:23+09:00
files_modified:
  - benchmarks/official/solid_mpm_fsi_runner.py
  - simulation_core/coupling/hibm_mpm/core.py
  - tests/benchmarks/test_canonical_production_runner_boundary_ledger.py
  - tests/benchmarks/test_post_solid_step_output_contract.py
  - tests/cases/test_ansys_vertical_flap_fsi.py
  - tests/integration/test_ansys_vertical_flap_feedback_conditioned_projection.py
  - tests/integration/test_ansys_vertical_flap_official_fluent_2way_reference.py
  - tests/integration/test_ansys_vertical_flap_preflow_snapshot.py
  - tests/solvers/test_hibm_runner_reachability_cache.py
  - tests/solvers/test_hibm_segment_pair_geometry.py
  - tests/solvers/_hibm_component_face_ledger_contracts.py
artifacts_created:
  - validation_runs/solver_soaks/diagnostic_replays/ansys_vf__diag__cf_fix__20260812__r01/diagnostic_replay.json
---

# ANSYS vertical-flap component-face target-membership handoff

## Canonical HIBM core cutover

The later cleanup in this worktree supersedes the earlier statement that the old
HIBM reconstructed cell-row API remained available. HIBM now has one production
velocity-boundary core: the schema-4 canonical component-face ledger.

- The old reconstructed-row writer, its row report type, relocation rollback
  image, shadow-face diagnostics, region-row counters, and schema-2/3 report
  adapters were deleted rather than retained as compatibility code.
- Generic HIBM load and post-solid assembly share one component-face builder and
  require canonical authority. The official runner rejects missing or historical
  HIBM reports instead of falling back to a row report.
- ANSYS vertical flap, Squid sharp HIBM-MPM, and Turek-Hron now write component
  masks and directed external faces, prepare every canonical consumer, and seal
  the ledger before a solve.
- The ordinary non-HIBM fluid boundary implementation remains for simulations
  that do not use HIBM. It is not an alternate HIBM core and no HIBM entry point
  dispatches to it.
- Tests dedicated to the deleted row implementation were removed; canonical
  relocation, component-face report, runner, snapshot, Turek, and migration
  contracts remain the supported verification surface.

Focused cutover evidence includes the canonical migration contracts, runner
schema-4 contracts, preflow snapshot contracts, all Turek case tests, and a real
CUDA small-grid Turek boundary build/seal. This is implementation and focused
contract evidence, not a new production preflow or Fluent-parity claim.

Latest cutover verification:

- the production old-symbol scan found no reconstructed-row writer, row report,
  row-count helper, shadow-face field, or schema-2/3 adapter under
  `simulation_core/`, `cases/`, or `benchmarks/`;
- migration contracts: `7 passed`; relocation transaction contracts: `18
  passed` with `307` subtests; canonical report contracts: `13 passed` with
  `177` subtests;
- a focused CUDA probe compiled the relocation source-key helper as `@ti.func`
  and passed in `11.68s`;
- the full canonical obstacle-relocation node exceeded the bounded `300s` cold
  JIT budget without an assertion result. It is a timeout, not a pass; no Python
  process remained afterward.

## Continuation result

This section supersedes the original stop boundary below.

The component-face target-membership slice is complete through one diagnostic FSI
step. The new predicate admits only the precomputed forward direct/shadow pair when
all existing surface topology gates pass, the inactive axis equals the component
axis, and the live shadow still resolves to the direct author's storage row. It
copies the cached admission into the existing finite-segment path; it does not
relax the denominator gate, write a mode directly, accept reverse ordering, or add
a compatibility path.

The exact test was simplified. Its temporary `solver_failure` /
`old_bug_precompute` branch and duplicate cleanup layer were deleted. The test now
states only the current successful contract and lets any `RuntimeError` fail
directly. Its global reconstruction-count oracle is `3`: this fixture reconstructs
the target x pair plus its existing auxiliary y and z pairs.

Current relevant file hashes:

```text
81EFF51B43E6E13C8D44BE4D6770778C3050CD974197BFB5C69678618DCFF1F4  benchmarks/official/solid_mpm_fsi_runner.py
AD8EC269385F20E8BFD04A294B4D8880E184E56CCF1A558A752401C6D73F0C8E  simulation_core/coupling/hibm_mpm/core.py
4BD0823A044B77675D483F9B8179E0E8EF72267575EE039CB86E04B78540B7C3  tests/solvers/test_hibm_segment_pair_geometry.py
30E7F99206652CAE75BF0A9AAAAFFFBD58391FAA7A0D2EB47878F770AEB13E0D  tests/solvers/_hibm_component_face_ledger_contracts.py
```

Current validation evidence:

- `py_compile`: passed for the three relevant Python files;
- scoped `git diff --check`: passed, with only the existing LF/CRLF warnings;
- the three micro nodes: `3 passed in 10.62s`, then `3 passed in 16.14s` after
  test simplification;
- the three low-level same-storage admission/anchor contracts: `3 passed in
  65.92s`;
- a faulthandler run confirmed the earlier 300-second timeouts were inside Taichi's
  prepare-kernel launch/compilation path; after one complete cold compile, the exact
  4x4x4 node passed: `1 passed in 44.57s`;
- the focused same-storage/bit64 matrix is GREEN: five ordered nodes emitted pass
  markers before the first 900-second bound, and the four remaining nodes completed
  as `4 passed in 776.25s`;
- the production snapshot replay completed its one requested FSI step in `1598.8s`:
  `validation_runs/solver_soaks/diagnostic_replays/ansys_vf__diag__cf_fix__20260812__r01/diagnostic_replay.json`;
- the replay loaded the canonical snapshot once, preserved both snapshot hashes,
  and all 79 recorded canonical/observer component-face reports had conflict counts
  `(target, region, alpha) = (0, 0, 0)`;
- the replay JSON SHA-256 is
  `2A65B7A9C0986F5F5F1D3E6B6E4CD004139042D56CFDB14BE2F9236EEDD47014`;
- focused post-change review found C/H/M/L = `0/0/0/0`;
- all completed and bounded runs ended with zero `python`/`pythonw` processes.

The production replay remains diagnostic-only: it reused a source-stale preflow
snapshot, reports `production_identity_valid=false`, and establishes neither a
fresh preflow nor formal or Fluent parity.

## Post-completion simplification

The follow-up cleanup removed compatibility code from the ANSYS production runner
without changing the component-face physics:

- sharp HIBM assembly now accepts only canonical component-face rows; the
  unreachable legacy reconstructed-row dispatch and its report adapter are gone;
- non-sharp marker feedback calls the concrete solver device API directly; the
  host NumPy fallback, its helper functions, and the cross-step
  `previous_feedback_constraint_cells` bookkeeping are gone;
- canonical inlet initialization and zmax refresh read the concrete solver
  authority/writer directly, so test doubles now state their authority explicitly;
- the equal-axis precompute path no longer uses a redundant selection helper, and
  prepare no longer rereads admission already proved by the cached author match;
- obsolete compatibility tests, synthetic impossible-state probes, duplicate
  source assertions, and four dead test locals were deleted.

The generic fluid solver's non-HIBM boundary authority remains for ordinary fluid
cases. The older HIBM core assembly API described here was subsequently removed by
the canonical-core cutover above.

Follow-up validation:

- `py_compile`, focused Ruff undefined/unused checks, and scoped
  `git diff --check`: passed;
- runner/authority/feedback/reachability contracts: `59 passed, 131 subtests
  passed`;
- preflow snapshot contracts: `139 passed`;
- official Fluent-reference integration helpers: `22 passed, 1 skipped`;
- the concrete solver's device marker-feedback contract: `1 passed`;
- affected boundary-predictor contract: `1 passed`;
- zmax/init authority and external-face provenance contracts: `4 passed`;
- direct same-storage geometry contracts: `4 passed`;
- independent post-cleanup review: no actionable findings;
- the large exact 4x4x4 ledger node reached the 300-second bound without an
  assertion result, so it is recorded as a timeout, not a pass or failure;
- the broad case file was not counted as GREEN: it stops on current-worktree
  expectations in tip-cap force reporting and top-face inlet reporting, neither
  of whose implementation functions was changed by this cleanup;
- all bounded runs ended with no remaining `python` or `pythonw` process.

No new production replay was started for this cleanup. The prior diagnostic-only
evidence boundary above remains unchanged.

## Original stop boundary (historical)

The user explicitly asked to pause and hand this work to a new session. No more
implementation or CUDA runs were started after that request. All child agents were
interrupted. At handoff time there are no `python` or `pythonw` processes.

Do not treat this document as a claim that the goal is complete. The current small
known-answer test is RED at a newly isolated prepare-stage gate. Production has not
been rerun after the target-aware precompute change, and there is no fresh/full/formal
Fluent parity evidence.

## Workspace warning

This is a heavily shared dirty worktree. Preserve it exactly.

- Branch: `codex/fix-main-audit-2026-08-10`
- Current HEAD: `c77b614 fix: stabilize validation artifact provenance`
- Do not run `git reset`, `git checkout --`, broad formatters, broad cleanup, or an
  entire-file rewrite.
- The three relevant files already contain large cumulative edits from this and
  earlier work. `git diff --stat` reports 8,529 insertions and 466 deletions across
  them. That is not the size of the final unresolved fix.
- No commit or push was made in this session.

Current file hashes at the stop boundary:

```text
CF02FA6A13249A100F5BA6FD8B37E7A6386859EA82D1C0E4000BDD94BD7DA356  simulation_core/coupling/hibm_mpm/core.py
AFAC090E3B0E9F3FF65E4C902A2F458BD75F37063D73E22F691DCFF2E8F9129D  tests/solvers/test_hibm_segment_pair_geometry.py
1D9CA9DC2D1CBE0D4BA0042EE7E46BF45567C20FF0EE6AF9C2222EBF58FBB958  tests/solvers/_hibm_component_face_ledger_contracts.py
```

Recheck these before editing. A mismatch can mean another session changed shared
files and must be reconciled, not reverted.

## Goal and debugging strategy

The user wants small, known-result simulations to expose narrow HIBM-MPM defects
without waiting hours for the production vertical-flap preflow. The current slice is
the canonical component-face arbitration used by the post-solid HIBM assembly.

The working protocol is:

1. Reproduce one production witness.
2. Build a 4x4x4 known-answer fixture with an exact numerical oracle.
3. Capture RED before implementation.
4. Make the smallest fail-closed production change.
5. Run micro kernels first, then one exact ledger node.
6. Only after those pass, run a bounded production diagnostic checkpoint.

Keep evidence labels honest. A micro or 4x4x4 pass is not production vertical-flap
coverage, one diagnostic snapshot replay is not a completed FSI step, and neither is
formal/fresh/Fluent parity.

## Production root cause that started this slice

Production diagnostic AB is here:

```text
validation_runs/solver_soaks/diagnostic_replays/
  vf50_goal_replay_20260812_ab_component_face_same_storage_capture/
```

It completed diagnostically with a solver conflict, zero completed FSI steps, one
loader invocation, and unchanged snapshot hashes:

```text
metadata f5bceb17c828a850e08e26cc4fe531eeca918dacb08761380afdf30cdcc2e634
npz      f51be25d565be3928a0fa617f45a400db228f6909fd83e31baa2c8bbe866ca8c
```

First conflict:

- target component face `(2,130,156)`, axis/inactive axis `0`
- prepare raw authors: direct key `205596`, row `(2,130,156)`, and relocation
  shadow source key `205276`, row `(2,129,156)`
- precompute cache before the current fix: direct/direct keys
  `(123676,205596)`, kinds `(0,0)`, admission/full `(1,1)`
- conflict source `prepare_pair_arbitration`, path `0`, claim count `2`

The deterministic cause was a precompute/prepare cohort mismatch:

- Precompute gave direct/direct priority after checking only active/fluid/actual
  sample availability for `target-e_axis` and `target`.
- Prepare independently called
  `_select_canonical_component_face_storage_device` for every raw author and only
  accepted authors whose ray selected the current target storage.
- Key `123676` belongs to the neighboring lower x face, so prepare excluded it.
- The target direct plus its same-storage transverse relocation shadow were the real
  target-local pair, but precompute had already cached the wrong authors.

## Implemented and verified in this session

### 1. Target-local pair selection contract

`core.py` now contains
`_canonical_component_face_equal_axis_pair_selection_kind` near line 16564.

It returns:

- `1` only when both direct rays select the current target face;
- `2` only when there is exactly one target-local same-storage direct/shadow pair;
- `0` for ambiguous or nonlocal cohorts.

The truth-table test is
`test_equal_axis_pair_selection_uses_only_target_local_authors` near
`test_hibm_segment_pair_geometry.py:443`.

### 2. Low-IR target-membership classifier

An initial correct implementation called the storage selector three times inside
the already-large precompute kernel. The exact 4x4x4 node then hit a 900-second hard
timeout with no pytest terminal. That approach was abandoned because the repeated
`@ti.func` inlining enlarged cold CUDA JIT too much.

The current implementation isolates selection in one small kernel:

- Two scalar `i32` node fields near `core.py:12263`:
  - `velocity_dirichlet_component_face_direct_selected_storage_offset`
  - `velocity_dirichlet_relocation_shadow_selected_storage_offset`
- Sentinel `-1`; legal offsets `0` or `1` relative to the direct source or shadow
  storage base along the one global inactive axis.
- `_classify_canonical_component_face_inactive_axis_storage_kernel` near
  `core.py:16596`.
- Exactly one syntax call site to
  `_select_canonical_component_face_storage_device`, shared by a runtime
  `author_kind` loop.
- Host launch only when `inactive_axis >= 0`, after the
  `hibm_velocity_row_segment_pair_precompute_before` observer and immediately
  before precompute, near `core.py:24002`.
- The large precompute kernel now contains zero selector call sites and only reads
  cached offsets.
- Commit and transaction clear/rollback reset both fields to `-1`.
- `_assert_component_face_relocation_transient_neutral` near ledger contracts line
  3933 checks both fields.

Independent static review of this low-IR hunk was C/H/M/L = `0/0/0/0`.

Runtime micro evidence, one CUDA process, exact nodes, same cache:

```text
3 passed in 12.17s
PYTHON_PROCESS_COUNT_AFTER=0
```

The three nodes were:

- `test_inactive_axis_storage_classifier_caches_direct_and_shadow_offsets`
- `test_equal_axis_pair_selection_uses_only_target_local_authors`
- `test_same_storage_segment_mode_accepts_only_exact_bit8_pair`

The classifier micro locks direct offset `1`, direct offset `0`, shadow offset `0`,
and untouched invalid values `-1/-1`.

### 3. Exact 4x4x4 target-membership fixture

The RED/GREEN integration node is:

```text
tests/solvers/test_hibm_component_face_geometry.py::
HibmComponentFaceGeometryTests::
test_same_storage_candidate_precedes_unused_opposite_direct_pair
```

The implementation lives near
`tests/solvers/_hibm_component_face_ledger_contracts.py:13574`.

Fixture:

- target `(2,2,2)`, component/inactive axis `0`
- opposite direct `(1,2,2)`, key `26`
- target direct `(2,2,2)`, key `42`
- transverse shadow source `(2,1,2)`, key `38`, stored at target direct
- x faces `(0,.1,.4,.7,1)`
- opposite direct ray remains at x `.25` and selects the neighboring face
- target direct and shadow rays use x `.54` and both select target face x `.4`

Desired precompute result is admission/full `1/1`, keys `42/38`, kinds `0/1`.
That is now achieved at runtime.

## Latest runtime evidence and current RED

### Cold-ish exact run after the low-IR refactor

```text
1 failed in 824.16s
shell wall 827.413s
exit 1
PYTHON_PROCESS_COUNT_AFTER=0
```

This did not time out. It reached pytest and proved the precompute result was already
the desired `(1,1,42,38,0,1)`. The test's old RED branch then obscured the downstream
RuntimeError by asserting the obsolete cache `(26,42,0,0)`.

The test was changed only to stop obscuring that error. If the old cache is seen it
still checks the original RED; otherwise it prints the later stage evidence and the
original RuntimeError.

### Warm diagnostic rerun after that tests-only fix

```text
1 failed in 47.19s
shell wall 48.319s
exit 1
PYTHON_PROCESS_COUNT_AFTER=0
```

Exact observation:

```text
materialized = (1, shadow_source=(2,1,2), storage_base=(2,2,2))
selected_storage_offsets = (opposite_direct=0, target_direct=0, target_shadow=0)
precompute = (
  admission=1,
  full=1,
  first_key=42,
  second_key=38,
  first_kind=0,
  second_kind=1,
  boundary=(0.4000000059604645, 0.375, 0.625),
  normal=(0.0, 1.0, 0.0),
  probe=(0.4000000059604645, 0.75, 0.625),
  boundary_target=2.0,
)
x_claim_rows = {(1,2,2): 1, (2,2,2): 2}
prepare = (claim_count=2, mode=0, witness1=-215, witness2=-219, conflicts=1)
first_conflict = face(2,2,2), axis0, prepare_pair_arbitration/path0,
                 raw author keys(42,38)
reconstruct = (mode=0, alpha=0.6666666865, target=6.8800005913)
precommit claim = same alpha/target
original error = conflicting canonical component-face claims (target): count=1
```

This is the current RED. Do not change the fixture or weaken the error oracle to make
it pass.

## Newly isolated prepare-stage cause, not yet fixed

The current precompute cache and author identity are correct. Prepare still does not
promote the equal-axis same-storage direct/shadow pair to mode `12` (`4|8`).

Relevant current code:

- `precomputed_finite_segment_pair_author_match`, around `core.py:19360`, is true.
- `claim_pair_is_same_storage_transverse` is true for direct `(2,2,2)` and shadow
  `(2,1,2)`.
- `compatible_interpolated_surface_segment_pair`, around `core.py:19523`, uses the
  old bracket rule. It requires a nonzero component-axis denominator between the two
  raw boundary points.
- Here both raw boundary x coordinates are `.54`, so the denominator is zero.
- `compatible_interpolated_distinct_finite_segment_pair`, around
  `core.py:19564`, is populated from the cached admission only when
  `surface_projection_inactive_axis != axis` (or through the separate bit64 double
  relocation path).
- Here inactive axis equals component axis (`0 == 0`), so this path remains zero.
- Therefore `compatible_interpolated_face_first_finite_segment_pair`, around
  `core.py:19675`, is false and prepare records path0 instead of mode12.

The next fix belongs in prepare, not in the classifier or geometry helper.

## Recommended next change

Keep it narrow and fail-closed:

1. Add an equal-axis same-storage direct/shadow prepare-topology predicate only after
   `precomputed_finite_segment_pair_author_match`, claim kinds, raw author rows, and
   live relocation identity are known.
2. Require at least:
   - inactive axis equals component axis;
   - exact cached precompute author match and admission;
   - cached kinds `(0,1)` in the same order as the live pair, or a deliberately
     handled reverse order;
   - `claim_pair_is_same_storage_transverse`;
   - live shadow valid at the direct storage row;
   - live shadow storage base equals the direct row;
   - live shadow source equals the relocation author;
   - raw author count at this comparison is exactly two-author progression
     (`claim_count == 1` for the second author);
   - the existing registered-segment, region, normal, actual-sample, and author
     validity gates remain true.
3. Only then copy the cached admission into
   `compatible_interpolated_distinct_finite_segment_pair`, so the existing face-first
   branch publishes exact mode12 and defers one canonical sample to reconstruct.
4. Do not remove the old denominator gate globally. It protects other pair modes.
5. Do not reuse bit16 or bit64 flags. This is the existing bit8 same-storage cohort.

Before editing, inspect the static conclusions returned by any new reviewer. The
prior reviewer tasks were interrupted at the user stop boundary and must not be
assumed complete.

## Tests that must remain fail-closed

The new prepare predicate must not bypass post-precompute stale-state checks. In
particular preserve or rerun:

- `test_transverse_same_storage_cap_face_rejects_contaminated_pair_mode_atomically`
- `test_transverse_same_storage_cap_face_requires_search_envelope`
- `test_transverse_same_storage_cap_face_requires_registered_segment`
- `test_transverse_same_storage_cap_face_rejects_stale_segment_anchor`
- `test_transverse_same_storage_cap_face_rejects_malformed_segment_indices`
- `test_transverse_same_storage_cap_face_rejects_third_author_atomically`
- the bit64 double-relocation negative matrix inside
  `test_shifted_inactive_axis_double_relocation_reconstructs_one_face_ray`

Also preserve these focused positives/micro contracts:

- `test_same_storage_direct_relocation_transverse_axis_remains_admitted`
- `test_same_storage_direct_relocation_component_axis_is_admitted`
- `test_same_storage_component_axis_rejects_anchor_beyond_four_tolerances`
- `test_same_storage_segment_mode_accepts_only_exact_bit8_pair`
- `test_equal_axis_pair_selection_uses_only_target_local_authors`
- `test_inactive_axis_storage_classifier_caches_direct_and_shadow_offsets`

## Suggested continuation sequence

1. Confirm no Python process and confirm the three hashes or reconcile any shared-file
   changes.
2. Read the current prepare block around `core.py:19360-19680` and the exact RED test
   around ledger contracts `13574-13890`.
3. Get a read-only static review of the proposed narrow predicate before editing.
4. Apply only the prepare predicate and any necessary focused assertions.
5. Run `py_compile` and scoped `git diff --check`.
6. Run the three micro nodes. They were 12 seconds with the warm cache.
7. Run the exact 4x4x4 node. With the current warm cache the diagnostic rerun was 47
   seconds. Do not launch a duplicate if one Python process already exists.
8. If exact GREEN, run the smallest same-storage/bit64 regression set in one process.
9. Only after those pass, run a new bounded production diagnostic checkpoint. Reuse
   the exact snapshot/config/manifest and the production cache; write to a new output
   directory. Do not overwrite AB.
10. Report production evidence as diagnostic-only unless a complete requested FSI
    step and all formal/fresh/parity gates actually finish.

## Commands

Trusted Python:

```text
D:\working\taichi\env\python.exe
```

Warm small-grid cache:

```text
validation_runs\.taichi_cache\cap_z_goal_20260811_a_cuda_f32
```

Before every CUDA command, use a single foreground process and clear Taichi variables
with .NET calls. `Remove-Item Env:...` has produced a duplicate-key warning in this
PowerShell environment.

```powershell
$p = @(Get-Process -Name python,pythonw -ErrorAction SilentlyContinue)
if ($p.Count -ne 0) { $p | Select-Object Id,ProcessName,StartTime,CPU; throw 'Python already running' }

[System.Environment]::SetEnvironmentVariable('TI_OFFLINE_CACHE',$null,'Process')
[System.Environment]::SetEnvironmentVariable('TI_OFFLINE_CACHE_FILE_PATH',$null,'Process')
[System.Environment]::SetEnvironmentVariable('TI_ARCH',$null,'Process')
[System.Environment]::SetEnvironmentVariable('TI_DEFAULT_FP',$null,'Process')
[System.Environment]::SetEnvironmentVariable('TI_DEVICE_MEMORY_FRACTION',$null,'Process')
$env:SIMULATION_TAICHI_OFFLINE_CACHE='1'
$env:SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH=(Resolve-Path 'validation_runs\.taichi_cache\cap_z_goal_20260811_a_cuda_f32').Path
```

Static gate:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  'simulation_core\coupling\hibm_mpm\core.py' `
  'tests\solvers\test_hibm_segment_pair_geometry.py' `
  'tests\solvers\_hibm_component_face_ledger_contracts.py'

git diff --check -- `
  'simulation_core/coupling/hibm_mpm/core.py' `
  'tests/solvers/test_hibm_segment_pair_geometry.py' `
  'tests/solvers/_hibm_component_face_ledger_contracts.py'
```

Micro gate:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest -q -x -p no:cacheprovider `
  'tests/solvers/test_hibm_segment_pair_geometry.py::HibmMpmSegmentPairGeometryTests::test_inactive_axis_storage_classifier_caches_direct_and_shadow_offsets' `
  'tests/solvers/test_hibm_segment_pair_geometry.py::HibmMpmSegmentPairGeometryTests::test_equal_axis_pair_selection_uses_only_target_local_authors' `
  'tests/solvers/test_hibm_segment_pair_geometry.py::HibmMpmSegmentPairGeometryTests::test_same_storage_segment_mode_accepts_only_exact_bit8_pair'
```

Exact integration RED/GREEN:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest -q -x -p no:cacheprovider `
  'tests/solvers/test_hibm_component_face_geometry.py::HibmComponentFaceGeometryTests::test_same_storage_candidate_precedes_unused_opposite_direct_pair'
```

Use a 300-second hard bound for the warm diagnostic. If the core changes cause a new
cold specialization, the first run may take longer; record a timeout honestly and do
not start a second process in parallel.

## Validation evidence boundaries

What is established:

- production AB exposed a deterministic precompute/prepare author-cohort mismatch;
- the low-IR classifier runs correctly on CUDA micro fixtures;
- the exact 4x4x4 precompute now selects target-local keys `42/38`, kinds `0/1`, with
  admission/full `1/1`;
- the previous RED was specifically prepare's equal-axis pair admission, not the
  classifier or cached geometry helper;
- the prepare source now has the narrow forward direct/shadow topology admission,
  and its static and micro gates pass;
- the exact 4x4x4 ledger node is GREEN, including prepare mode `12`, cached authors
  `42/38`, the expected reconstruction weight/value, and zero target conflicts;
- all requested focused same-storage and bit64 positive/fail-closed regressions are
  GREEN;
- one diagnostic post-snapshot FSI step completed with all recorded component-face
  conflict triples equal to zero and unchanged snapshot hashes;
- all runs ended with zero Python orphans.

What is not established:

- a fresh current-source preflow;
- a production-identity-valid run from a current-source snapshot;
- more than the one requested diagnostic FSI step;
- formal validation, Fluent parity, or fluent-parity evidence.

That distinction matters. Do not compress it in the next status report.
