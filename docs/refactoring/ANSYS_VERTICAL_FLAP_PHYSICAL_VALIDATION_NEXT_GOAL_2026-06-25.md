# ANSYS vertical-flap physical-validation next goal - 2026-06-25

## Objective

Advance the ANSYS official half-domain vertical-flap work from the completed
artifact/schema repair at commit `10bdb7c8e2600c1f772c12509ffef3e2e4b5fa46`
into the next physical-validation baseline.

The previous commit fixed the official-half-domain evidence chain and formal
runner structure. This goal must keep that boundary honest: do not claim
pointwise Fluent parity, do not claim completed 50-step official validation
without fresh 50-step evidence, and do not treat the existing one-step archive
as a full FSI evolution result.

This goal is based on the review note attached on 2026-06-25. The review
accepted the schema/geometry/marker/CFL repairs and identified the remaining
work needed to make the branch reviewable and to start diagnosing flow/solid
physics rather than only metadata consistency.

## Current accepted baseline

The repository already has:

- official half-domain archive schema consistency:
  - `case = ansys-fluent-official-half-domain-single-flap`
  - `official_half_domain = true`
  - `full_domain_two_flap = false`
  - `flap_count_modeled = 1`
  - `flap_count_displayed_after_symmetry_mirror = 2`
  - `marker_count_actual = 168`
  - `flow_projection_iterations_actual = 4096`
- formal runner full-span flap geometry:
  - `x_min = 0.0`
  - `x_max = config.span_m`
  - streamwise flap extent from official mesh coordinates
    `z = 0.050 m` to `z = 0.053 m`
- formal runner two streamwise marker faces:
  - `+z` and `-z` marker normals
  - actual marker count is two times `config.marker_count`
- solid elastic-wave CFL substep selection and reporting.
- archive consistency test coverage across manifest, summary, report, process,
  history, render metadata, and `fields.npz`.

These facts are the baseline for this goal. Do not undo them.

## Required work

### 1. Add reviewable verification surface

Add a lightweight GitHub Actions workflow or an equivalent repository-visible
verification surface.

The workflow must not require a CUDA GPU for its required job. It should run
source-level and archive-level checks that are credible on GitHub-hosted
runners:

```powershell
python -m py_compile cases/ansys_vertical_flap_fsi.py benchmarks/official/solid_mpm_fsi_runner.py tools/validation/print_ansys_vertical_flap_diagnostics.py
python -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency -v
python -m unittest <non-runtime ANSYS vertical-flap source/contract tests> -v
```

If runtime-heavy Taichi/CUDA tests are not portable to CI, keep them out of the
required CI job or gate them behind an explicit environment condition. The CI
must still catch regressions in:

- official-half-domain archive schema.
- formal runner full-span geometry.
- dual streamwise marker faces.
- solid CFL substep helper.
- diagnostics/report schema expected by committed artifacts.

### 2. Commit local verification record for the prior repair

Add a verification record under `validation_runs/ansys_vertical_flap_fsi/`.
It must record:

- commit under review: `10bdb7c8e2600c1f772c12509ffef3e2e4b5fa46`
- commands run locally for that commit.
- pass counts for the focused archive and runner tests.
- the 200 second timeouts observed when full discovery reached runtime-heavy
  50-step / two-step GPU tests.
- the fact that `git diff --check` emitted Windows line-ending warnings but no
  whitespace errors.
- the fact that the previous push hook printed
  `No supported checks found in this repository. Skipping.`

This record exists so that the validation facts are not available only in the
chat transcript.

### 3. Run or attempt repaired 50-step coarse smoke

Run the repaired formal ANSYS vertical-flap case through the local EasyFsi
solver with the default coarse configuration:

```powershell
& 'D:\working\taichi\env\python.exe' run_simulation.py ansys-vertical-flap-fsi --steps 50 --json
```

Capture the output as:

```text
validation_runs/ansys_vertical_flap_fsi/easyfsi/easyfsi_step050_after_halfdomain_repair.json
```

Then run diagnostics:

```powershell
& 'D:\working\taichi\env\python.exe' -m tools.validation.print_ansys_vertical_flap_diagnostics `
  --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050_after_halfdomain_repair.json `
  --fluent-tip-csv validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv `
  --output-dir validation_runs\ansys_vertical_flap_fsi\compare_after_halfdomain_repair
```

If the run is too slow or environment-blocked, preserve the exact command,
timeout, partial output path, and reason in the verification record. Do not
invent success artifacts.

The primary metrics to preserve are:

- `velocity_peak_mps`
- `computed_pressure_min_pa`
- `computed_pressure_max_pa`
- `total_marker_force_z_N`
- `mpm_external_force_z_N`
- `scatter_action_reaction_residual_N`
- `tip_dz_final_m`
- `tip_dz_min_m`
- `tip_dz_max_m`
- `tip_dz_sign_violation_count`
- `tip_dz_monotonic_violation_count`
- `max_displacement_relative_error`
- `solid_substeps_selected`
- `solid_estimated_cfl`

The purpose is not to force a green physical result. The purpose is to learn
whether full-span geometry and two-face markers fixed the previous `FAIL_FLOW`
symptom.

### 4. Add fixed-solid preflow controls and diagnostics

Add explicit preflow controls to the formal ANSYS vertical-flap runner path.

Required behavior:

- `preflow_steps = 0` remains the default unless explicitly configured.
- During preflow, the solid/flap must remain fixed.
- The flow is projected around the fixed flap.
- Marker force/stress sampling can run to record the fixed-solid loading.
- MPM solid stepping must not run during preflow.
- Surface feedback must not be treated as a completed FSI step during preflow.
- The final preflow flow state is retained as the initial flow state for the
  subsequent FSI steps.

Report fields must include:

- `preflow_steps_requested`
- `preflow_steps_completed`
- `preflow_converged`
- `preflow_stop_reason`
- `preflow_history`
- per-preflow-step velocity peak, pressure min/max, projection diagnostics,
  marker force, valid/invalid stress marker counts, and optional convergence
  deltas.

The first implementation may be conservative. It should establish explicit
diagnostic evidence, not hide a pressure or force instability.

### 5. Add Gate A/B non-expected tests

Add non-expected tests for structural and qualitative physical gates. These
tests must not be hidden behind `@unittest.expectedFailure`.

Gate A: structural correctness:

- official half-domain is true.
- full-domain two-flap is false.
- modeled flap count is one.
- mirrored display flap count is two.
- full-span flap geometry is used.
- two streamwise marker faces are represented.
- invalid stress/scatter/feedback counts are zero for the checked artifact or
  light runtime report.

Gate B: qualitative physics:

- streamwise marker force sign is physically documented.
- tip streamwise displacement sign is physically documented.
- root displacement remains near zero.
- pressure, velocity, displacement, and force values are finite.
- if velocity peak remains outside the official web contour range, the test or
  artifact must report that as a diagnostic status rather than silently passing
  quantitative validation.

The existing 5 percent displacement tolerance may remain as expected failure.
Do not use it as the only physical gate for this stage.

### 6. Add preflow smoke verification

Run at least one short preflow-enabled smoke or source-level test that proves:

- `preflow_steps` is accepted by the CLI/config path.
- preflow records diagnostics.
- MPM substeps are not executed inside preflow.
- the subsequent FSI loop still records normal history.

If a full `preflow=5/10/20 + steps=50` sweep is too slow for this task, record
that it remains future work and keep the short smoke evidence honest.

## Non-goals

- Do not run or claim high-resolution L3 50-step validation in this goal.
- Do not claim ANSYS Fluent solve parity.
- Do not commit raw ANSYS tutorial assets.
- Do not hide expected failures by weakening physical assertions.
- Do not convert default coarse smoke settings into a validation-level claim.
- Do not add tolerance hacks to make the 50-step displacement test green.

## Validation requirements

Before commit and push:

1. Compile changed Python files.
2. Run the new archive/source/Gate A/B tests.
3. Run the CI-equivalent local command set.
4. Run or honestly record the repaired 50-step coarse smoke attempt.
5. Run `git diff --check`.
6. Confirm `git status --short` before staging.
7. Commit and push to the configured GitHub remote.

Expected minimum commands:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.cases.test_ansys_vertical_flap_fsi -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests -p '*ansys*vertical*flap*.py' -v
git diff --check
```

If full discovery is too slow, record the exact timeout and run the narrowest
tests that cover the changed behavior.

## Deliverables

- This detailed goal file.
- Short goal reference to this file.
- Lightweight CI or equivalent verification workflow.
- Committed verification record for commit `10bdb7c8...`.
- Repaired 50-step coarse smoke artifact or an honest timeout/failure record.
- Diagnostics output under
  `validation_runs/ansys_vertical_flap_fsi/compare_after_halfdomain_repair`
  when the repaired 50-step run completes.
- Fixed-solid preflow controls and report diagnostics.
- Gate A/B non-expected tests.
- Final commit hash and pushed branch.
