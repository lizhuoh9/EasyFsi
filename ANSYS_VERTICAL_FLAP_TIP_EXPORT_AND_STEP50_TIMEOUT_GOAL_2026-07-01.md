# ANSYS Vertical-Flap Tip Export and Step50 Timeout Goal - 2026-07-01

## Starting Point

The remote branch is expected to start from:

```text
branch: codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01
head: a8251b4f2107ba506b3610934c0b8b998f671b90
message: validation: record vertical flap real solver evidence
```

The previous commit correctly stopped importer hardening and recorded real
solver validation status. The next change must stay in that same evidence-first
boundary: fix or document the EasyFsi-side export issue, add tests first, keep
Fluent blocked unless real exports exist, and push only after focused
verification passes.

## Problem Statement

The fresh generic solver selected-formulation artifacts record:

```text
candidate_status = generic_solver_selected_formulation_step50_passed
completed_step_count = 50
max_displacement_m = 2.298969411640428e-05
max_pressure_abs_pa = 488.50444618549903
max_velocity_mps = 31.764066696166992
fluent_parity_claimed = False
```

However, the committed `easyfsi_tip_displacement_history.csv` has
`tip_displacement_x_m`, `tip_displacement_y_m`, `tip_displacement_z_m`, and
`tip_displacement_norm_m` all equal to `0.0` through step 50 while
`max_displacement_m` is nonzero. That may be physically valid only if the
exported tip probe is intentionally rooted/fixed, but it is dangerous as a
future Fluent tip-displacement comparison input. The current artifacts do not
make that boundary obvious enough.

The selected-formulation coupled step50 rerun was also attempted but timed out
at the local 600 second window. That must remain explicit: committed step50
artifacts may satisfy their artifact contract, but they are not fresh evidence
from the latest rerun.

## Goals

1. Create this detailed goal document first and reference it from the active
   goal before modifying code or tests.
2. Diagnose the generic solver history fields that contain tip or displacement
   data, using the current committed artifacts.
3. Add a RED test that fails on the current ambiguous export boundary:
   nonzero `max_displacement_m` must not coexist with all-zero tip displacement
   columns unless the artifact explicitly records the fixed-tip/export-basis
   explanation and blocks future Fluent tip parity usage.
4. Implement the minimal fix in the generic solver artifact export surface:
   either map the true nonzero tip displacement if the runtime history provides
   it, or explicitly mark the tip export as a fixed-root probe / unavailable
   for Fluent tip parity while preserving `max_displacement_m` as the usable
   EasyFsi displacement metric.
5. Regenerate the generic solver selected-formulation artifacts only if the
   exporter or summary content changes.
6. Update the real solver validation status report so it states the resolved
   interpretation of the all-zero tip-displacement columns and the remaining
   selected step50 timeout boundary.
7. Keep `fluent_parity_claimed: false` and
   `fluent_parity_status: blocked_reference_incomplete`.
8. Do not promote or synthesize Fluent `source_exports`; real Fluent bundle
   import remains out of scope unless a provenance-backed bundle is actually
   present.
9. Verify focused tests and push the completed branch.

## Non-Goals

- Do not continue speculative importer hardening.
- Do not generate fake Fluent CSVs, fake metadata, or placeholder rows.
- Do not claim Fluent parity without real Fluent source exports.
- Do not rewrite tolerances to force a green parity claim.
- Do not treat selected-formulation coupled step50 as fresh evidence if the
  rerun remains blocked by runtime duration.
- Do not create a velocity contour from scalar history data or otherwise
  invent spatial fields.
- Do not change solver physics merely to make an artifact look better.

## Required Diagnostic Commands

Inspect generic solver displacement fields:

```powershell
@'
import json
from pathlib import Path

history_path = Path("validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/generic_solver_selected_formulation_history.json")
payload = json.loads(history_path.read_text(encoding="utf-8"))
history = payload["history"]

keys = sorted({key for row in history for key in row})
print("history_key_count =", len(keys))
for key in keys:
    if "tip" in key.lower() or "displacement" in key.lower():
        values = [row.get(key) for row in history[-5:]]
        print(key, "=>", values)
'@ | & 'D:\working\taichi\env\python.exe' -
```

If the history has a real nonzero tip displacement field, the exporter must map
that field into `easyfsi_tip_displacement_history.csv`.

If the history does not have a real tip displacement field, the exporter and
status report must explicitly state that the tip displacement columns are a
fixed-root or unavailable tip probe and that `max_displacement_m` is the
currently valid EasyFsi displacement metric. Future Fluent tip parity must stay
blocked until a real comparable tip probe is exported.

## Required Tests

At minimum, add or update focused tests so they prove:

```text
generic solver status remains step50 passed
max_displacement_m remains nonzero
tip displacement export does not silently look comparable when all zero
the artifact explains the tip export basis or blocks tip parity usage
fluent_parity_claimed remains false
source_exports is not modified or promoted
selected formulation step50 timeout remains documented as non-fresh evidence
```

The RED test must be run before production/artifact changes and fail for the
intended missing explanation or mapping.

## Verification Commands

Use the trusted interpreter:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_ansys_vertical_flap_generic_solver.py `
  tests\integration\test_ansys_vertical_flap_generic_solver_artifacts.py `
  tests\integration\test_ansys_vertical_flap_real_solver_validation_status.py
```

Focused tests:

```powershell
& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_generic_solver_artifacts `
  tests.integration.test_ansys_vertical_flap_real_solver_validation_status `
  -v
```

Boundary checks:

```powershell
git diff --name-only HEAD -- validation_runs\ansys_vertical_flap_fsi\fluent_reference\source_exports
git diff --check
```

The `source_exports` diff must be empty.

## Completion Criteria

The task is complete only when:

1. The active goal references this markdown file.
2. The RED test was observed before the fix.
3. The fix or documentation change is implemented with the narrowest practical
   artifact/code surface.
4. The generic solver artifacts and status report no longer silently imply
   comparable tip displacement when the CSV columns are all zero.
5. Focused tests pass locally with `D:\working\taichi\env\python.exe`.
6. `source_exports` remains untouched.
7. The final commit is pushed to the current branch.

