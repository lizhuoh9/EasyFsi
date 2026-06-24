# ANSYS Vertical-flap Solver-validation Goal

Objective:
Start the first solver-validation branch after the ten-step structure refactor
by adding stable diagnostic output for the ANSYS Fluent vertical-flap FSI smoke
case. This branch must make the existing EasyFsi JSON reports easier to inspect
and compare against future Fluent report files. It must not fix solver physics.

Base branch:
`docs/post-refactor-baseline`

Base commit:
`cc94367a083d24190c1316fa8a425178aa49b970`

Working branch:
`validation/ansys-vertical-flap-fsi-2026-06-24`

Primary case:
`ansys-vertical-flap-fsi`

Primary entry point:
```powershell
& 'D:\working\taichi\env\python.exe' run_simulation.py ansys-vertical-flap-fsi --steps 50 --json
```

Reason for this branch:
`docs/POST_REFACTOR_BASELINE.md` records the ANSYS vertical-flap displacement
tolerance as a known non-gating solver-validation failure. Future solver fixes
need a focused reproduction, scoped solver change, and physical validation
evidence. This branch creates the diagnostic output needed before any solver
formula change is attempted.

Hard boundaries:
- Do not change solver physics.
- Do not change Taichi kernel math.
- Do not change fluid projection formulas.
- Do not change HIBM/MPM coupling behavior.
- Do not change solid MPM formulas.
- Do not change material formulas.
- Do not change ANSYS case parameters.
- Do not change case defaults.
- Do not change benchmark formulas.
- Do not adjust displacement tolerance to hide the known failure.
- Do not claim physical validation against Fluent unless real Fluent report
  files are present and parsed.
- Do not commit generated long-run output unless it is small, deterministic, and
  explicitly part of this goal.

Allowed edit surface:
- `tools/validation/`
- `tests/tools/`
- `docs/validation/`
- `docs/VALIDATION.md` only if adding a link or command for the new diagnostic
  script.

Required implementation:
1. Add a reusable Python diagnostic module/script under `tools/validation/`.
2. The script must read one or more EasyFsi JSON reports produced by
   `run_simulation.py ansys-vertical-flap-fsi --steps N --json`.
3. The script must write deterministic files to an output directory:
   - `easyfsi_summary.csv`
   - `easyfsi_summary.json`
   - `easyfsi_history.csv`
   - `stage_check.md`
4. The script must optionally accept a Fluent tip-displacement CSV. If provided,
   it must also write:
   - `displacement_compare.csv`
5. The script must be safe to run from the repo root with:
   ```powershell
   & 'D:\working\taichi\env\python.exe' -m tools.validation.print_ansys_vertical_flap_diagnostics `
     --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json `
     --output-dir validation_runs\ansys_vertical_flap_fsi\compare
   ```
6. The script must return a nonzero exit code only for malformed inputs or I/O
   errors, not for physical FAIL_* diagnostic statuses. A failed physical
   status is an output fact, not a process failure.

Required summary columns:
- `case`
- `steps`
- `dt_s`
- `grid`
- `particles`
- `markers`
- `support_radius_m`
- `velocity_peak_mps`
- `velocity_peak_relerr`
- `max_disp_m`
- `ref_max_disp_m`
- `disp_relerr`
- `root_max_disp_m`
- `stress_invalid`
- `scatter_invalid`
- `feedback_invalid`
- `marker_force_z_N`
- `mpm_external_force_z_N`
- `scatter_action_reaction_residual_N`
- `status`

Required status order:
1. `FAIL_FLOW`: velocity peak is outside the official/reference range or its
   relative error exceeds the report tolerance.
2. `FAIL_INTERFACE`: invalid stress markers exist or the marker force has the
   wrong streamwise sign.
3. `FAIL_SCATTER`: invalid scatter markers exist or the action-reaction
   residual is too large.
4. `FAIL_SOLID_ROOT`: the clamped root displacement is visibly nonzero.
5. `FAIL_SOLID_SIGN`: tip displacement has the wrong streamwise sign.
6. `FAIL_MAGNITUDE`: direction is plausible but displacement magnitude exceeds
   tolerance.
7. `PASS_SMOKE`: all checks above pass.

Required history columns:
- `step`
- `time_s`
- `stress_valid_marker_count`
- `scatter_invalid_marker_count`
- `feedback_invalid_marker_count`
- `total_marker_force_x_N`
- `total_marker_force_y_N`
- `total_marker_force_z_N`
- `mpm_external_force_x_N`
- `mpm_external_force_y_N`
- `mpm_external_force_z_N`
- `tip_mean_dx_m`
- `tip_mean_dy_m`
- `tip_mean_dz_m`
- `tip_norm_m`
- `max_displacement_m`
- `root_max_displacement_m`
- `surface_feedback_max_marker_displacement_m`

Required `stage_check.md` sections:
- `[SETUP]`
- `[FLOW_ONLY]`
- `[INTERFACE_FORCE]`
- `[SOLID_RESPONSE]`
- `[FSI_FEEDBACK]`
- `[COORDINATE_MAPPING]`

Required coordinate mapping text:
```text
Fluent x <-> EasyFsi z
Fluent y <-> EasyFsi y
Fluent out-of-plane <-> EasyFsi x
```

Required Fluent comparison behavior:
- Accept a CSV with `step` or `time_s`.
- Accept `tip_total_displacement_m`, `tip_x_displacement_m`, and
  `tip_y_displacement_m` when present.
- Join by `step` when possible; otherwise join by nearest `time_s`.
- Write `displacement_compare.csv` with:
  - `step`
  - `time_s`
  - `fluent_tip_total_m`
  - `easyfsi_tip_total_m`
  - `abs_error`
  - `rel_error`
  - `fluent_tip_x_m`
  - `fluent_tip_y_m`
  - `easyfsi_tip_streamwise_m`
  - `easyfsi_tip_vertical_m`
- If no Fluent CSV is supplied, `stage_check.md` must clearly say that Fluent
  comparison was not run.

Required tests:
- Add focused `unittest` coverage under `tests/tools/`.
- Test deterministic summary/status generation from a small fixture report.
- Test history CSV rows and vector component extraction.
- Test `stage_check.md` contains setup, stage diagnoses, and coordinate mapping.
- Test optional Fluent comparison CSV generation.
- Test malformed JSON or missing required input fails with a clear exception or
  nonzero CLI exit.

Required validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  tools\validation\__init__.py `
  tools\validation\print_ansys_vertical_flap_diagnostics.py `
  tests\tools\test_ansys_vertical_flap_diagnostics.py
& 'D:\working\taichi\env\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_diagnostics -v
git diff --check
```

Known-red confirmation:
```powershell
& 'D:\working\taichi\env\python.exe' -m unittest tests.cases.test_ansys_vertical_flap_fsi -v
```

This command may fail on the existing ANSYS vertical-flap solver-validation
red light. Do not fix solver physics in this branch. The acceptable outcome for
this report-only branch is either a pass or a failure confined to the existing
vertical-flap physical chain/tolerance behavior.

Optional runtime probe:
```powershell
New-Item -ItemType Directory -Force validation_runs\ansys_vertical_flap_fsi\easyfsi
& 'D:\working\taichi\env\python.exe' run_simulation.py ansys-vertical-flap-fsi --steps 1 --json `
  > validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step001.json
& 'D:\working\taichi\env\python.exe' -m tools.validation.print_ansys_vertical_flap_diagnostics `
  --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step001.json `
  --output-dir validation_runs\ansys_vertical_flap_fsi\compare
```

Acceptance:
- The new validation script exists and can be run as a module.
- It writes summary, history, and stage diagnosis artifacts from EasyFsi JSON.
- It can generate a Fluent displacement comparison when a Fluent CSV is present.
- Tests cover the diagnostic extraction and comparison behavior.
- The branch contains no solver physics, kernel, case default, or benchmark
  formula changes.
- The completed work is committed and pushed to GitHub on
  `validation/ansys-vertical-flap-fsi-2026-06-24`.
