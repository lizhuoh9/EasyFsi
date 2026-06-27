# ANSYS Vertical-Flap Selected Formulation Coupled Anchor Injection Goal

Date: 2026-06-27

Source checkpoint: remote branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
at commit `e6947d21bae14f3599d5dc53f22f584e64ce68e2`.

Short goal reference for Codex `/goal`:

```text
Implement the detailed ANSYS vertical-flap selected-formulation coupled
anchor-injection contract in
docs/refactoring/ANSYS_VERTICAL_FLAP_SELECTED_FORMULATION_COUPLED_ANCHOR_INJECTION_GOAL_2026-06-27.md.
First lock the current coupled-smoke pending artifact as a negative checkpoint.
Then make the selected formulation coupled smoke path install baseline
pressure-pair anchor data into live markers, add source-level tests, refresh
the coupled-smoke artifact, keep all claims artifact-bounded, commit, push,
and verify remote Actions. Do not claim 50-step validation or Fluent parity.
```

## Current Checkpoint

The previous checkpoint is a correct fail-closed coupled-smoke entry point.
The selected formulation can enter the real coupled smoke path, the first
coupled preflight step executes, and the committed artifact remains honest:

```text
candidate_status = selected_formulation_coupled_smoke_pending
smoke_status = blocked_invalid_marker_sampling
requested_step_count = 5
completed_step_count = 1
invalid_marker_count_max = 24
one_sided_marker_count_min = 0
```

The artifact does not claim 5-step, 50-step, or Fluent parity success. The
active blockers are:

```text
coupled_fsi_validation_pending
no_fluent_parity_claim
blocked_invalid_marker_sampling
```

The scenario diagnostics locate the current failure in stress/traction
sampling rather than feedback or scatter bookkeeping:

```text
stress_invalid_marker_count = 24
stress_valid_marker_count = 0
primary_face_invalid_marker_count = 12
secondary_face_invalid_marker_count = 12
two_sided_pressure_marker_count = 0
scatter_invalid_marker_count = 0
feedback_invalid_marker_count = 0
surface_feedback_updated_marker_count = 24
```

The likely direct cause is that the selected formulation coupled smoke path
sets the anchored policies but does not install the per-marker baseline anchor
data into the live marker structure used by stress sampling.

## Objective

Move from a fail-closed selected-formulation coupled-smoke entry point to a
coupled smoke path that explicitly installs and verifies pressure-pair anchor
data for live markers.

This goal has five required outputs:

1. Harden the current committed coupled-smoke artifact test so the current
   pending state is a clear negative checkpoint.
2. Add a narrow selected-formulation anchor-installation path for the coupled
   smoke runner without allowing arbitrary non-default formulations to bypass
   the existing guard.
3. Add source-level tests for missing, mismatched, and valid anchor-map
   installation semantics.
4. Regenerate the selected-formulation coupled-smoke artifact from the updated
   runner and update the artifact test to lock the new honest evidence.
5. Verify locally, commit, push, and confirm the pushed GitHub Actions run.

The strongest allowed final claim is:

```text
selected formulation coupled smoke passed
```

Only if the committed artifact proves the requested 5-step smoke completed and
all smoke gates passed.

If only the first diagnostic step is repaired, the artifact must remain:

```text
candidate_status = selected_formulation_coupled_smoke_pending
```

with an explicit blocker such as:

```text
blocked_requested_5step_not_completed
```

That is still a valid improvement if:

```text
invalid_marker_count_max = 0
one_sided_marker_count_min >= 24
anchor_selected_marker_count_min >= 24
anchor_fallback_marker_count_max = 0
```

If the anchor install still cannot make stress sampling valid, keep the
artifact pending and record the exact failing diagnostics. Do not fake a pass.

## Non-Goals

Do not implement or claim:

- 30-step, 50-step, or longer coupled validation
- Fluent parity
- final displacement parity
- material parameter tuning
- geometry tuning
- force aggregation rewrites
- pressure, velocity, force, displacement, or marker-result hardcoding
- broad solver physics rewrites
- relaxing the selected-formulation coupled-smoke guard
- allowing arbitrary non-default traction formulations into coupled runs
- heavy coupled runner execution in GitHub Actions

Do not retire `no_fluent_parity_claim`.

Do not replace the current fail-closed artifact with a broad, ambiguous test
that accepts too many states. Each committed checkpoint must lock the current
artifact-backed evidence.

## Phase 0 - Lock Current Pending Diagnostics

Update:

```text
tests/integration/test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py
```

Before changing runtime behavior, add assertions that the current committed
artifact is a negative checkpoint:

```text
candidate_status == selected_formulation_coupled_smoke_pending
smoke_status == blocked_invalid_marker_sampling
completed_step_count == 1
requested_step_count == 5
invalid_marker_count_max == 24
one_sided_marker_count_min == 0
smoke_acceptance.accepted == false
smoke_acceptance.no_marker_invalid == false
smoke_acceptance.one_sided_complete == false
smoke_acceptance.finite_fields == true
smoke_acceptance.anchor_fallback_zero == true
```

Also assert the scenario diagnostics:

```text
stress_invalid_marker_count == 24
stress_valid_marker_count == 0
primary_face_invalid_marker_count == 12
secondary_face_invalid_marker_count == 12
scatter_invalid_marker_count == 0
feedback_invalid_marker_count == 0
surface_feedback_updated_marker_count == 24
```

This protects the diagnosis that the failure is in stress sampling, not
scatter/feedback, and not NaN/Inf.

## Phase 1 - Anchor Installation Design

Add a narrow mechanism that lets the selected-formulation coupled smoke path
install pressure-pair anchor data into live markers after marker construction
and before stress sampling.

Preferred shape:

```python
marker_post_build_callback(markers, config) -> None
```

or an equivalent runner-owned hook if a better local extension point already
exists.

The install path must verify:

```text
anchor marker_count == live marker_count
anchor_source_marker_geometry_sha256 == current marker_geometry_sha256
anchor_source_flow_snapshot_sha256 == source shared/fixed-solid snapshot SHA
pressure_pair_policy == baseline_anchored_cell_pair
one_sided_pressure_policy == per_face_mirrored
```

Any mismatch must fail closed with a clear error or artifact blocker. Silent
fallback is not allowed.

If no committed anchor-cell arrays are available in the existing artifacts,
implement the narrowest honest fallback:

```text
derive deterministic per-marker anchor cells from the same live marker geometry
and selected pressure-pair policy, record that the source is derived_live_marker_geometry,
and fail closed if the derived anchors cannot cover every selected smoke marker.
```

The artifact must clearly distinguish committed artifact anchors from derived
live anchors.

## Phase 2 - Runtime Diagnostics

Extend the coupled smoke artifact rows and history so each run records:

```text
pressure_pair_anchor_active_marker_count
pressure_pair_anchor_selected_marker_count
pressure_pair_anchor_fallback_marker_count
pressure_pair_anchor_map_sha256
pressure_pair_anchor_source_marker_geometry_sha256
pressure_pair_anchor_current_marker_geometry_sha256
pressure_pair_anchor_source
```

If the runner uses derived live anchors, record:

```text
pressure_pair_anchor_source = derived_live_marker_geometry
```

If it uses committed artifact anchors, record the artifact path and SHA.

The runner summary must keep the non-claim text:

```text
does not claim 50-step validation
does not claim Fluent parity
```

## Phase 3 - Source-Level Tests

Add or extend focused tests to cover:

```text
selected coupled smoke cannot run anchored policy without anchor map/install
anchor marker-count mismatch fails fast
anchor geometry SHA mismatch fails fast
valid anchor install marks all selected-smoke markers anchor-active
valid anchor install produces zero anchor fallback count
non-selected coupled path still cannot bypass diagnostics-only guard
```

Prefer existing test modules if they already cover ANSYS vertical-flap and
traction pressure-pair policy contracts:

```text
tests/cases/test_ansys_vertical_flap_fsi.py
tests/integration/test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py
```

Create a new solver-level test only if the relevant helper lives outside the
case runner and needs direct unit coverage.

## Phase 4 - Regenerate Coupled Smoke Artifact

Run:

```powershell
& 'D:\working\taichi\env\python.exe' `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_smoke.py
```

The regenerated artifact must choose one honest result:

### Result A - First-Step Sampling Fixed, 5-Step Still Pending

```text
candidate_status = selected_formulation_coupled_smoke_pending
smoke_status = blocked_requested_5step_not_completed
completed_step_count = 1
requested_step_count = 5
invalid_marker_count_max = 0
one_sided_marker_count_min >= 24
anchor_selected_marker_count_min >= 24
anchor_fallback_marker_count_max = 0
```

This is an acceptable completion for this goal because it removes
`blocked_invalid_marker_sampling` but keeps the requested 5-step blocker.

### Result B - Requested 5-Step Smoke Passed

```text
candidate_status = selected_formulation_coupled_smoke_passed
completed_step_count = 5
requested_step_count = 5
invalid_marker_count_max = 0
one_sided_marker_count_min >= 24
anchor_selected_marker_count_min >= 24
anchor_fallback_marker_count_max = 0
```

In this case retire:

```text
coupled_fsi_validation_pending
```

and keep:

```text
long_coupled_validation_pending
no_fluent_parity_claim
```

### Result C - Anchor Install Still Fails

```text
candidate_status = selected_formulation_coupled_smoke_pending
smoke_status = blocked_anchor_installation
```

or the exact failing diagnostic. Keep `coupled_fsi_validation_pending` active
and do not claim improvement beyond the new diagnostics.

## Phase 5 - Artifact Test Update

Update the artifact test to lock the regenerated evidence.

If Result A is produced, assert:

```text
candidate_status == selected_formulation_coupled_smoke_pending
smoke_status == blocked_requested_5step_not_completed
completed_step_count == 1
requested_step_count == 5
invalid_marker_count_max == 0
one_sided_marker_count_min >= 24
anchor_selected_marker_count_min >= 24
anchor_fallback_marker_count_max == 0
coupled_fsi_validation_pending remains active
no_fluent_parity_claim remains active
```

If Result B is produced, assert:

```text
candidate_status == selected_formulation_coupled_smoke_passed
completed_step_count == 5
requested_step_count == 5
long_coupled_validation_pending active
no_fluent_parity_claim active
coupled_fsi_validation_pending retired
```

If Result C is produced, assert the exact new blocker and diagnostics.

All artifact tests must continue to assert:

```text
summary contains no Fluent parity success claim
summary contains no 50-step success claim
checksums match committed bytes
source artifact SHA values match actual files
source_script is repo-relative
```

## Phase 6 - Workflow Scope

Do not add heavy runner execution to GitHub Actions.

Keep workflow coverage cheap:

```text
py_compile selected smoke runner
run committed artifact tests
run relevant source-level unit tests if already cheap
```

Only update `.github/workflows/ansys-vertical-flap-validation.yml` if the new
source-level tests are cheap enough and logically belong in the existing
contracts job.

## Phase 7 - Local Verification

Use the trusted local interpreter when available:

```powershell
& 'D:\working\taichi\env\python.exe' ...
```

Minimum verification:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  cases\ansys_vertical_flap_fsi.py `
  benchmarks\official\solid_mpm_fsi_runner.py `
  tests\cases\test_ansys_vertical_flap_fsi.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_smoke.py

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts `
  -v

git diff --check
```

Also verify the regenerated artifact checksums.

## Phase 8 - Completion Criteria

This goal is complete only when:

- This detailed goal file is committed.
- The current pending coupled-smoke artifact is locked as a negative checkpoint
  before runtime repair.
- The selected-formulation coupled smoke path has explicit anchor installation
  or an explicit fail-closed diagnostic for why anchor installation cannot be
  completed.
- Source-level tests cover missing/mismatched/valid anchor installation.
- The coupled-smoke artifact is regenerated and committed.
- The artifact test locks the new honest evidence.
- No 50-step validation claim is made.
- No Fluent parity claim is made.
- Local focused verification passes.
- The branch is committed and pushed.
- The pushed GitHub Actions run is green, or any remaining red status is
  explicitly diagnosed as unrelated and reported with evidence.
