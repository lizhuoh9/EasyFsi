# ANSYS Vertical-Flap Per-Face One-Sided Pressure Goal

Date: 2026-06-27

Source checkpoint: remote commit
`49005323d6e18204385c5255103131c8df3584ad` on branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

## Source Evidence

The previous reference-preselection checkpoint correctly promoted
`baseline_anchored_cell_pair` to a pressure-pair policy component candidate
while keeping complete reference-formulation selection deferred:

- `candidate_status = pressure_pair_policy_preselection_candidate_found`
- `pressure_pair_policy_candidate = baseline_anchored_cell_pair`
- `reference_formulation_candidate = None`
- `anchored_force_ratio_span.relative_span = 0.0`
- `absolute_baseline_bias = 0.003726805140894629`
- shared snapshot SHA:
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- `completed_formulation_count = 9`
- `unsupported_formulation_count = 1`

The remaining explicit blocker is the dual-face one-sided row:

```text
run_status = unsupported
formulation_status = unsupported
pressure_pair_policy = per_face_one_sided_pressure
unsupported_reason = dual-face one-sided pressure needs per-face one-sided region support
```

This goal starts from that honest blocker. It must not reinterpret the previous
artifact as full reference-formulation evidence.

## Objective

Implement a diagnostic-only per-face one-sided pressure policy for the ANSYS
vertical-flap shared-snapshot traction path so the dual-face one-sided pressure
case can complete as a sampled marker-traction row instead of remaining
unsupported.

The completed per-face one-sided artifact must still be constrained to
shared-snapshot, fixed-solid, marker-traction sampling. It must keep:

```text
reference_formulation_candidate = None
```

The pressure-pair candidate remains:

```text
pressure_pair_policy_candidate = baseline_anchored_cell_pair
```

This goal removes only the per-face one-sided support blocker. It does not
advance to complete reference formulation selection.

## Non-Goals

Do not implement or claim any of the following:

- coupled FSI execution
- 50-step coupled FSI
- fixed-solid regenerated preflow evidence
- Fluent parity
- complete reference formulation selection
- any non-`None` `reference_formulation_candidate`
- changes to marker force aggregation
- changes to fluid solver physics
- changes to solid solver physics
- changes to marker geometry generation
- overwriting previous reference-preselection artifacts
- hardcoded pressure, force, displacement, flow, or marker results

All new evidence must remain artifact-bounded and sampling-only.

## Phase 0 - Harden Existing Reference-Preselection History Semantics

Before adding the new per-face one-sided path, harden the existing
reference-preselection artifact semantics so completed history rows cannot keep
the older anchor-map wording.

Update:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_pressure_pair_reference_preselection_matrix.py
tests/integration/test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts.py
```

The runner should normalize completed histories with:

```python
def _normalize_completed_history(history, *, scenario):
    normalized = dict(history)
    normalized["flow_phase"] = "shared_snapshot_pressure_pair_reference_preselection"
    normalized["scenario"] = scenario
    return normalized
```

The artifact test must assert every committed history row records:

- `flow_phase == shared_snapshot_pressure_pair_reference_preselection`
- the expected shared snapshot SHA
- a scenario from the expected reference-preselection scenario set

If the committed history content changes, rerun the reference-preselection
runner and refresh checksums.

## Phase 1 - Public Per-Face One-Sided Config Surface

Add a narrow diagnostic config surface to the ANSYS vertical-flap case. The
default must preserve existing behavior.

Required public fields:

```python
traction_one_sided_pressure_policy: str = "disabled"
traction_one_sided_primary_fluid_side_normal_sign: float | None = None
traction_one_sided_secondary_fluid_side_normal_sign: float | None = None
traction_one_sided_primary_reference_pressure_pa: float = 0.0
traction_one_sided_secondary_reference_pressure_pa: float = 0.0
traction_one_sided_pressure_pair_policy: str = "baseline_anchored_cell_pair"
```

Supported policies:

```text
disabled
per_face_mirrored
```

`disabled` must be the default.

`per_face_mirrored` is allowed only for diagnostic sampling contexts. The
runner/config guard must reject attempts to use it as a normal positive-step
coupled-FSI setting. The intended allowed context is a shared snapshot or
preflow/fixed-solid diagnostic with no coupled advancement.

## Phase 2 - Core Per-Face One-Sided API

The current one-sided surface is too coarse because it is based on a single
one-sided region. Add explicit per-face inputs rather than overloading only:

```python
one_sided_pressure_region_id
```

The minimum acceptable API must distinguish primary and secondary face
selection and reference pressures:

```python
one_sided_pressure_primary_region_id: int = -1
one_sided_pressure_secondary_region_id: int = -1
one_sided_primary_reference_pressure_pa: float = 0.0
one_sided_secondary_reference_pressure_pa: float = 0.0
one_sided_primary_fluid_side_normal_sign: float = 0.0
one_sided_secondary_fluid_side_normal_sign: float = 0.0
```

Equivalent per-marker/per-region storage is acceptable if it is at least as
explicit and is covered by tests.

Required sampling semantics:

- primary face chooses inside or outside pressure from
  `one_sided_primary_fluid_side_normal_sign`
- secondary face chooses the mirrored/declared side from
  `one_sided_secondary_fluid_side_normal_sign`
- primary and secondary faces apply independent reference pressures
- missing required side data fails closed or records explicit fallback status
- no silent fallback to a two-sided policy is allowed for a declared
  per-face one-sided row

## Phase 3 - Core Diagnostics

Every per-face one-sided marker diagnostic must make the side selection
auditable. Required fields include:

```text
one_sided_policy
one_sided_region_id
one_sided_side_selected
one_sided_fluid_side_pressure_pa
one_sided_reference_pressure_pa
one_sided_pressure_pair_policy
one_sided_anchor_selected
one_sided_anchor_fallback_used
```

The diagnostics must be present in the committed per-face one-sided artifact
and must remain repo-relative where paths are recorded.

## Phase 4 - Solver Tests

Add focused solver/config coverage:

```text
tests/solvers/test_hibm_traction_per_face_one_sided_pressure.py
```

Required behavior:

- default one-sided disabled path is unchanged
- legacy single-region one-sided behavior remains unchanged if retained
- `per_face_mirrored` selects the primary and secondary sides explicitly
- primary and secondary reference pressures are applied independently
- missing anchor/pair/side data fails closed or records explicit fallback
- invalid policy/sign/reference values fail fast
- positive-step coupled config rejects the per-face diagnostic policy
- step-zero/preflow/shared-snapshot diagnostics allow the policy

These tests must not run coupled FSI.

## Phase 5 - Per-Face One-Sided Runner

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_per_face_one_sided_matrix.py
```

Output artifacts under:

```text
validation_runs/ansys_vertical_flap_fsi/traction_per_face_one_sided_diagnostics/
```

The runner must use the archived shared snapshot:

```text
3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968
```

It must reuse the reference-preselection anchor-map provenance:

```text
anchor_source_scenario = baseline_independent_ladder_probe0p51
pressure_pair_policy_candidate = baseline_anchored_cell_pair
```

Minimum matrix rows:

```text
baseline_anchored_two_sided_probe0p51
dual_one_sided_per_face_probe0p51
dual_one_sided_per_face_probe0p625
dual_one_sided_per_face_probe1p00
```

The runner must:

1. Load the committed shared snapshot.
2. Reuse the snapshot without fluid advance, solid advance, or feedback.
3. Sample marker tractions only.
4. Preserve the baseline anchored two-sided row as a reference row.
5. Complete the dual-face one-sided per-face rows.
6. Write matrix JSON/CSV, history JSON, marker diagnostics, summary Markdown,
   and checksums.
7. Keep recorded paths repo-relative.

## Phase 6 - Per-Face One-Sided Artifact Gate

The new artifact may report:

```text
candidate_status = per_face_one_sided_pressure_completed
```

only if all required per-face rows complete and every gate below passes.

Required gates:

- the previous unsupported row is no longer unsupported in the new artifact
- primary face marker count equals expected marker count
- secondary face marker count equals expected marker count
- one-sided pressure is complete on both faces
- invalid marker counts are zero
- anchor selected count equals marker count where anchors are applicable
- anchor fallback count is zero
- traction decomposition residual `<= 1.0e-8`
- every row uses the expected shared snapshot SHA
- no coupled FSI was advanced
- no marker feedback was applied
- no Fluent parity claim is made
- `reference_formulation_candidate is None`

The new artifact must keep candidate blockers for evidence still not available:

```text
reference_selection_deferred
sampling_only_no_coupled_fsi
no_fluent_parity_claim
```

The historical blocker:

```text
dual_face_one_sided_unsupported
```

may be removed from the new per-face artifact only after the dual-face
per-face rows complete. Previous reference-preselection artifacts must not be
rewritten to pretend the old unsupported row was completed.

## Phase 7 - Artifact Tests And Workflow Cheap Checks

Add:

```text
tests/integration/test_ansys_vertical_flap_traction_per_face_one_sided_artifacts.py
```

The test must assert:

- matrix, CSV, history, summary, marker diagnostics, and checksums exist
- `source_script` is repo-relative
- the expected shared snapshot SHA is used
- all per-face one-sided rows complete
- one-sided diagnostics are present for primary and secondary faces
- primary and secondary side policies are recorded
- anchors are selected for all applicable markers
- anchor fallback is zero
- `reference_formulation_candidate is None`
- blockers preserve no-coupled/no-Fluent/reference-deferred limits
- no local absolute paths are recorded

Update:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Only add cheap checks:

- `py_compile` for `run_traction_per_face_one_sided_matrix.py`
- unittest for
  `tests.integration.test_ansys_vertical_flap_traction_per_face_one_sided_artifacts`

Do not make CI run the GPU artifact-generation runner.

## Phase 8 - Later Reference Formulation Selection

This phase is documented for sequencing only and is not part of this
implementation unless a future goal explicitly authorizes it.

Only after per-face one-sided pressure completes may a later goal create:

```text
validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/
```

The later phase requires:

- `pressure_pair_policy_candidate = baseline_anchored_cell_pair`
- `absolute_baseline_bias <= 0.01`
- dual-face one-sided completed
- all required formulation rows completed
- offset sensitivity within tolerance
- the expected shared snapshot SHA
- complete diagnostics

Only that later phase may set:

```text
reference_formulation_candidate != None
```

Even then, it must not claim Fluent parity while evidence remains
shared-snapshot sampling-only.

## Phase 9 - Later Regenerated Fixed-Solid Evidence

After a future reference formulation candidate is selected, the next stage
should be regenerated fixed-solid evidence, not coupled FSI:

```text
fixed-solid regenerated preflow with selected policy
fixed-solid load consistency
fixed-solid temporal/load artifact
```

If the regenerated snapshot or marker geometry changes, the old anchor map
must not be blindly reused. Re-derive or revalidate anchor-map provenance.

## Phase 10 - Later Coupled Smoke And Fluent Parity

Only after fixed-solid evidence is complete should later goals run:

```text
short coupled smoke, 5-10 steps
STEP30 / STEP50 or 50-step coupled FSI
force / displacement / flow comparison
```

Only then may Fluent parity be discussed.

## Validation Plan

Use the reliable local interpreter:

```powershell
& "D:\working\taichi\env\python.exe" -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_pressure_pair_reference_preselection_matrix.py tests\integration\test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts.py
& "D:\working\taichi\env\python.exe" -m unittest tests.integration.test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts -v
& "D:\working\taichi\env\python.exe" -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_per_face_one_sided_matrix.py tests\integration\test_ansys_vertical_flap_traction_per_face_one_sided_artifacts.py
& "D:\working\taichi\env\python.exe" -m unittest tests.solvers.test_hibm_traction_per_face_one_sided_pressure -v
& "D:\working\taichi\env\python.exe" validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_per_face_one_sided_matrix.py
& "D:\working\taichi\env\python.exe" -m unittest tests.integration.test_ansys_vertical_flap_traction_per_face_one_sided_artifacts -v
& "D:\working\taichi\env\python.exe" -m unittest tests.integration.test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts tests.integration.test_ansys_vertical_flap_traction_per_face_one_sided_artifacts -v
git diff --check
```

If runtime cost is manageable, also run the workflow-equivalent artifact
consistency unittest block.

## Git And Push Requirements

After validation passes:

1. Confirm `git status --short`.
2. Stage only files belonging to this goal.
3. Commit with a conventional message.
4. Push the current branch to `origin`.
5. Verify the remote ref points at the pushed commit.

The final report must include:

- commit hash
- remote branch
- validation commands and outcomes
- whether README was checked and updated
- whether push succeeded
- remote-ref verification

## Done Criteria

This goal is complete only when:

- this detailed goal file exists
- the active Codex goal references this file
- reference-preselection history semantics are hardened
- default disabled one-sided behavior is preserved
- per-face one-sided diagnostic config/API exists
- per-face one-sided solver/config tests pass
- per-face one-sided runner exists and generates committed artifacts
- dual-face one-sided rows complete in the new artifact
- `reference_formulation_candidate` remains `None`
- artifact tests protect candidate gates and non-claims
- workflow cheap checks include the new runner/test
- README has been checked for contract drift
- validation passes locally
- the commit is pushed to GitHub and the remote ref is verified
