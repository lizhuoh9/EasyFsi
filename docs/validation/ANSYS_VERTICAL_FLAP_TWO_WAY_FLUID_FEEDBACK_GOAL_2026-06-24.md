# ANSYS Vertical-flap Two-way Fluid-feedback Goal

Objective:
Implement the first scoped solver-validation fix for the ANSYS Fluent
vertical-flap FSI case: the EasyFsi smoke runner must no longer behave like a
single precomputed fluid load followed by solid-only time stepping. Each FSI
time step must update the fluid obstacle/interface representation from the
current solid state after surface feedback, recompute/project the fluid field,
and record that recomputation in the report/history so later Fluent CSV
comparison can distinguish flow/interface/scatter/solid/time-coupling failures.

Base branch:
`validation/ansys-vertical-flap-fsi-2026-06-24`

Base commit:
`4149187c1705cb122d8c6d0454a206873971ee9c`

Working branch:
`solver/ansys-vertical-flap-two-way-fluid-feedback`

Primary case:
`ansys-vertical-flap-fsi`

Primary runner:
`benchmarks/official/solid_mpm_fsi_runner.py`

Primary diagnostic script:
`tools/validation/print_ansys_vertical_flap_diagnostics.py`

Reason for this branch:
The previous validation branch added diagnostic tooling only. It did not commit
real EasyFsi/Fluent outputs and did not change solver behavior. Review now
identifies the most suspicious physical mismatch: the current runner solves the
fluid once before the FSI loop, then repeatedly samples the same fluid field
while the solid advances and marker feedback is recorded. That is not
equivalent to Fluent two-way intrinsic FSI, where structure deformation updates
the interface/dynamic mesh and the next transient fluid solve sees the changed
boundary.

Hard boundaries:
- Do not tune `max_displacement_m` reference values.
- Do not loosen `displacement_tolerance` or `velocity_peak_tolerance`.
- Do not change ANSYS geometry, material, `dt_s`, step count, inlet velocity,
  outlet type, or boundary-condition metadata.
- Do not add pressure/velocity/displacement hardcoding to force agreement.
- Do not hide failures by changing tests to accept worse physics.
- Do not claim numerical agreement with Fluent until a real Fluent
  `fluent_tip_displacement.csv` is available and parsed.
- Do not run or commit long generated validation outputs unless they are small,
  deterministic, and explicitly used as test fixtures.

Allowed edit surface:
- `benchmarks/official/solid_mpm_fsi_runner.py`
- `cases/ansys_vertical_flap_fsi.py` only if a config/report knob is needed
  for this case.
- `tests/cases/test_ansys_vertical_flap_fsi.py`
- `tests/tools/test_ansys_vertical_flap_diagnostics.py`
- `tools/validation/print_ansys_vertical_flap_diagnostics.py`
- `docs/validation/`
- `docs/VALIDATION.md`

Required behavior:
1. Preserve the existing initial steady/projection fluid solve before the FSI
   loop.
2. After each solid advance and HIBM surface-feedback update, update the fluid
   obstacle/interface representation from the current solid state.
3. Recompute/project the fluid field after that feedback update so the next
   stress sampling uses the latest feedback-aware fluid state.
4. Record per-step evidence in `history`:
   - `fluid_recomputed_after_feedback`
   - `fluid_recompute_step`
   - `post_feedback_local_velocity_peak_mps`
   - `post_feedback_pressure_min_pa`
   - `post_feedback_pressure_max_pa`
   - `post_feedback_obstacle_cell_count`
   - `post_feedback_fluid_cell_count`
5. Record report-level evidence:
   - `fluid_recomputed_after_feedback`
   - `fluid_recompute_count`
   - `fluid_recompute_steps`
   - `initial_flow_projection_report`
   - `final_flow_projection_report`
   - `fluid_feedback_coupling_mode`
6. Keep the previous `flow_projection_report`, `computed_pressure_min_pa`,
   `computed_pressure_max_pa`, `local_velocity_peak_mps`, and cell-count fields
   meaningful by mapping them to the final feedback-aware fluid state.
7. Update the validation diagnostic script so `stage_check.md` no longer prints
   a hardcoded `fluid_recomputed_after_feedback = false`; it must read the
   report field and show recompute count/steps when present.

Implementation notes:
- Prefer a small helper that rebuilds or updates the rectangular solid obstacle
  from the current solid particle positions, while preserving the original
  domain, inlet, outlet, and projection setup.
- If the obstacle update cannot safely represent full Fluent dynamic mesh
  behavior, name the limitation honestly in report fields; do not overclaim.
- The first goal is feedback-aware fluid recomputation and observability, not
  final Fluent agreement.

Required tests:
1. Add/adjust a fast source-level or mocked test proving the FSI loop calls the
   fluid recompute path after surface feedback, not only before the loop.
2. Add/adjust a report-level test proving `fluid_recomputed_after_feedback` is
   true and `fluid_recompute_count == step_count` for a short ANSYS vertical
   flap run.
3. Add/adjust a history test proving every history row records the post-feedback
   recompute fields.
4. Add/adjust diagnostic-script tests proving `stage_check.md` reflects the
   report-provided feedback recompute state instead of a hardcoded false value.
5. Preserve existing checks that:
   - pressure/velocity are computed, not reference-assigned;
   - marker force streamwise sign is physical;
   - root displacement remains clamped;
   - invalid marker/scatter counters remain exposed.

Required validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  benchmarks\official\solid_mpm_fsi_runner.py `
  cases\ansys_vertical_flap_fsi.py `
  tools\validation\print_ansys_vertical_flap_diagnostics.py `
  tests\cases\test_ansys_vertical_flap_fsi.py `
  tests\tools\test_ansys_vertical_flap_diagnostics.py
& 'D:\working\taichi\env\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_diagnostics -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.cases.test_ansys_vertical_flap_fsi -v
& 'D:\working\taichi\env\python.exe' scripts\validate_structure.py
git diff --check
```

Known possible outcome:
The ANSYS case test may still fail final displacement magnitude or Fluent parity
until a real Fluent report CSV and deeper solver work exist. It should not fail
because the runner is still single-flow-load-only, because recompute evidence is
missing, or because diagnostics hardcode `fluid_recomputed_after_feedback`.

Acceptance:
- The runner recomputes/projects fluid after each feedback update.
- Report/history expose feedback-aware fluid recomputation evidence.
- Diagnostics consume and display that evidence.
- Tests prove the new feedback/recompute contract.
- No reference/tolerance/material/geometry shortcuts are introduced.
- Work is committed and pushed to GitHub on
  `solver/ansys-vertical-flap-two-way-fluid-feedback`.
