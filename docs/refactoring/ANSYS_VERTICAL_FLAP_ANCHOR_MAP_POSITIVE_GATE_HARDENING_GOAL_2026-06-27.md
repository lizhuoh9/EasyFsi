# ANSYS Vertical-Flap Anchor-Map Positive Gate Hardening Goal

Date: 2026-06-27

Source request: harden the evidence produced after commit
`9980a9e4b8d85474ec1abfb8b2a27428823241bc` on branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

## Objective

Lock the first positive `baseline_anchored_cell_pair` shared-snapshot evidence
into the artifact test suite without changing solver physics, runner behavior,
or reference-formulation status.

The current anchor-map artifact proves that fixed baseline-derived
inside/outside pressure-pair cells remove the probe-origin sweep instability:
the anchored rows have force-ratio relative span `0.0`, all anchored rows select
24/24 anchors, and no anchored row uses fallback. The existing artifact test
still permits either a stable candidate or a no-stable-candidate state. That was
appropriate while the runner was being introduced, but it is now too weak: a
future regression could erase this first positive result while the test still
passes.

This task must convert the artifact gate from "schema accepts both outcomes" to
"this committed artifact is a positive stable anchor-map result" while preserving
the diagnostic-only boundary:

- `candidate_status` must remain
  `pressure_pair_anchor_map_stable_candidate_found`.
- `stable_pressure_pair_policy` must remain `baseline_anchored_cell_pair`.
- `reference_formulation_candidate` must remain `None`.
- No coupled FSI, Fluent parity, or reference formulation selection may be
  claimed by this hardening.

## Scope

Touch only the files needed to harden and verify the existing positive
anchor-map artifact evidence.

Expected code/test scope:

- `tests/integration/test_ansys_vertical_flap_traction_pressure_pair_anchor_map_artifacts.py`
  - Strengthen candidate-status assertions from a two-state allowance into a
    positive stable-candidate requirement.
  - Assert the stable pressure-pair policy.
  - Assert the acceptance gate is accepted.
  - Assert the force-ratio relative span is exactly `0.0` for the committed
    artifact.
  - Assert no anchor fallback is used.
  - Assert all anchors are selected for every anchored row.
  - Assert every anchored row uses `baseline_anchored_cell_pair`.
  - Assert every anchored row has 24 selected anchors, zero fallback markers,
    and a non-empty anchor-map hash.
  - Assert all anchored force ratios are identical to each other, without
    necessarily pinning the absolute floating-point value unless needed.
  - Keep the existing schema, path, checksum, marker-field, scope, and non-claim
    assertions.

Expected documentation scope:

- This goal file is the detailed contract for the task.
- README does not need an update because this task only strengthens tests
  around committed validation artifacts and does not change user-facing commands,
  solver behavior, or repository layout.

Expected artifact scope:

- Do not regenerate anchor-map artifacts unless a validation failure proves the
  checked-in artifacts are inconsistent with their tests.
- Do not modify generated artifact contents merely to make a test pass.

## Non-Goals

Do not implement or start any of these in this commit:

- A reference-preselection runner.
- A reference-formulation selection runner.
- Any change that makes `reference_formulation_candidate` non-`None`.
- Per-face one-sided pressure support.
- Dual-face one-sided artifact generation.
- Coupled FSI smoke, fixed-solid regenerated snapshot, or Fluent parity checks.
- Solver-core policy changes.
- Force aggregation changes.
- Changes to anchor-map runner behavior.
- New physics or numerical tuning.

These later phases remain important, but they must be handled in separate goals
and commits after this positive evidence is locked.

## Required Hardening Assertions

The artifact test must lock the top-level positive candidate gate:

```python
self.assertEqual(
    payload["candidate_status"],
    "pressure_pair_anchor_map_stable_candidate_found",
)
self.assertEqual(
    payload["stable_pressure_pair_policy"],
    "baseline_anchored_cell_pair",
)
self.assertIsNone(payload["reference_formulation_candidate"])
self.assertTrue(payload["anchor_map_acceptance"]["accepted"])
self.assertEqual(
    float(payload["anchor_map_acceptance"]["force_ratio_relative_span"]),
    0.0,
)
self.assertTrue(payload["anchor_map_acceptance"]["anchor_selected_all_markers"])
self.assertTrue(payload["anchor_map_acceptance"]["anchor_fallback_zero"])
self.assertLessEqual(
    float(
        payload["anchor_map_acceptance"][
            "max_face_traction_decomposition_residual_pa"
        ]
    ),
    1.0e-8,
)
```

The artifact test must also lock every anchored row:

```python
anchored_ratios = {
    round(float(by_scenario[scenario]["force_ratio_to_anchor_baseline"]), 12)
    for scenario in EXPECTED_ANCHORED_SCENARIOS
}
self.assertEqual(len(anchored_ratios), 1)

for scenario in EXPECTED_ANCHORED_SCENARIOS:
    row = by_scenario[scenario]
    self.assertEqual(row["pressure_pair_policy"], "baseline_anchored_cell_pair")
    self.assertEqual(row["anchor_source_scenario"], EXPECTED_BASELINE_SCENARIO)
    self.assertNotEqual(row["anchor_map_sha256"], "")
    self.assertEqual(int(row["pressure_pair_anchor_selected_marker_count"]), 24)
    self.assertEqual(int(row["pressure_pair_anchor_fallback_marker_count"]), 0)
```

It is acceptable to additionally assert the known current ratio
`0.9962731948591054` if the value proves stable in the committed artifact, but
the minimum required lock is that all anchored rows have the same ratio and the
relative span is exactly zero.

## Validation Plan

Use the reliable local interpreter:

```powershell
& "D:\working\taichi\env\python.exe" -m py_compile tests\integration\test_ansys_vertical_flap_traction_pressure_pair_anchor_map_artifacts.py
& "D:\working\taichi\env\python.exe" -m unittest tests.integration.test_ansys_vertical_flap_traction_pressure_pair_anchor_map_artifacts -v
git diff --check
```

Before pushing, also run the focused artifact pair that protects both the
previous symmetric negative evidence and this positive anchor-map evidence:

```powershell
& "D:\working\taichi\env\python.exe" -m unittest tests.integration.test_ansys_vertical_flap_traction_symmetric_pressure_pair_artifacts tests.integration.test_ansys_vertical_flap_traction_pressure_pair_anchor_map_artifacts -v
```

If the test fails because the artifact is no longer positive, do not weaken the
assertion. Stop and diagnose why the committed artifact no longer has the
positive gate.

## Git And Push Requirements

After validation passes:

1. Confirm `git status --short`.
2. Stage only files belonging to this task.
3. Commit with a conventional message, expected:
   `test: harden ANSYS pressure pair anchor-map gates`.
4. Push the current branch to `origin`.
5. Verify the remote branch with `git ls-remote`.

The final report must include:

- Commit hash.
- Remote branch.
- Validation commands and outcomes.
- Confirmation that README was checked and did not require changes.
- Confirmation that the pushed remote ref matches the local commit.

## Done Criteria

This task is complete only when all of the following are true:

- A detailed goal markdown file exists in `docs/refactoring/`.
- The active goal references this markdown file.
- The anchor-map artifact test requires the positive stable candidate state.
- Anchored rows are locked to `baseline_anchored_cell_pair`, 24/24 selected
  anchors, zero anchor fallback markers, and identical force ratios.
- `reference_formulation_candidate` remains `None`.
- No solver physics, runner behavior, or generated artifact contents are
  changed unless proven necessary by validation.
- The focused compile/test checks pass.
- `git diff --check` passes.
- The commit is pushed to GitHub and the remote ref is verified.
