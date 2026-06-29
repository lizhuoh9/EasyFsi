# ANSYS Vertical Flap Fixed-Flow Step 3 Postprocess Goal - 2026-07-01

## Source Request

Implement the Step 3 postprocessing plan described in the user-provided review file:

`C:\Users\lizhu\.codex\attachments\3a220f75-e519-4fa0-9aa2-17933f392dbf\pasted-text.txt`

Current branch:

`codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`

Starting HEAD:

`2f617e1e5c7b2008b6221356f543087bc7eb5436`

## Objective

Add Step 3 for the ANSYS vertical-flap fixed-flow validation case: Fluent-style postprocessing, field plots, profile CSVs, numerical quality gates, and a validation report.

This step must consume the Step 2 fixed-flap projection-solver artifacts and produce visual/report artifacts that make the current solver state inspectable without overstating Fluent parity.

Step 3 must not modify the Step 2 solver numerics. It must not read old FSI shared-snapshot or traction diagnostics. It must not claim Fluent parity.

## Inputs

Step 3 must read:

- `validation_runs/ansys_vertical_flap_fixed_flow/fields/final_fields.npz`
- `validation_runs/ansys_vertical_flap_fixed_flow/logs/solver_history.csv`
- `validation_runs/ansys_vertical_flap_fixed_flow/logs/mass_balance.csv`
- `validation_runs/ansys_vertical_flap_fixed_flow/case_manifest_step2.json`

The `final_fields.npz` source is the Step 2 fixed-flap projection solver output. It is not a Fluent export and not an FSI shared snapshot.

## Required Outputs

Write all Step 3 artifacts under:

`validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/`

Required files:

- `speed_full_fluent_scale_0_28p1.png`
- `speed_full_autoscale.png`
- `streamwise_minus_Uz_fluent_scale_0_28p1.png`
- `streamwise_minus_Uz_autoscale.png`
- `Uy_full.png`
- `pressure_full.png`
- `geometry_overlay.png`
- `solver_history_plot.png`
- `mass_balance_plot.png`
- `centerline_streamwise_minus_Uz.csv`
- `throat_profile_streamwise_minus_Uz.csv`
- `downstream_profiles_streamwise_minus_Uz.csv`
- `validation_report.md`
- `case_manifest_step3.json`

## Required Source Files

Add:

- `src/refactored/validation/ansys_vertical_flap_fixed/plot_fields.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/profile_extractors.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/quality_gates.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/report_builder.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/postprocess_fluent_style.py`
- `validation_cases/ansys_vertical_flap_fixed_flow/run_fixed_flap_postprocess.py`
- `tests/integration/test_ansys_vertical_flap_fixed_flow_step3_postprocess.py`

Update:

- `src/refactored/validation/ansys_vertical_flap_fixed/__init__.py`

## Claims And Scope Boundary

Step 3 must explicitly state:

- `fluent_parity = not_claimed`
- `fsi = not_claimed`
- `source = step2_fixed_flap_projection_solver`
- `traction_shared_snapshot_diagnostics = not_used`

The report must contain the literal language:

- `No Fluent parity claim`
- `No FSI claim`
- `traction_shared_snapshot_diagnostics not used`
- `Step 2 solver output`
- `diagnostic_only_not_parity`

Do not use phrases that imply official Fluent agreement, such as `parity achieved`, `validated against Fluent`, or `official Fluent match`.

## Quality Gates

Add `quality_gates.py` with:

- `load_solver_history(path: str | Path) -> list[dict[str, float]]`
- `load_mass_balance(path: str | Path) -> list[dict[str, float]]`
- `evaluate_quality_gates(history_rows, mass_rows, final_summary, config=None) -> dict`

Quality gates must include:

- `visual_candidate`
- `mass_quality`
- `incompressibility_quality`
- `overall_status`

Default gate thresholds:

- `max_mass_imbalance_rel = 0.05`
- `max_divergence_l2_warn = 100.0`
- `max_divergence_linf_warn = 10000.0`
- `max_poisson_residual_linf_warn = 1.0e8`

Expected interpretation for the current Step 2 artifacts:

- Visual candidate should pass because a centerline jet exists.
- Mass quality may pass or warn depending on the final artifact value.
- Incompressibility quality must not be silently reported as pass if the divergence or Poisson residual values exceed thresholds.
- Overall status must be `diagnostic_only_not_parity` unless all numerical quality gates justify a stronger candidate label.

## Plotting Contract

Add `plot_fields.py` with pure local plotting support. Do not require network downloads or heavy new plotting dependencies. If `matplotlib` is unavailable, write valid PNGs with a local raster fallback.

Field plots must:

- Use display coordinates `s` and `y`.
- Use white for solid cells.
- Preserve full-domain geometry shape.
- Use fixed Fluent-style scale `0-28.1 m/s` for:
  - `speed_full_fluent_scale_0_28p1.png`
  - `streamwise_minus_Uz_fluent_scale_0_28p1.png`
- Also write autoscale versions for speed and streamwise velocity because Step 2 max speed may exceed `28.1 m/s`.
- Use symmetric color scaling for `Uy_full.png`.
- Use autoscale for `pressure_full.png`.
- Generate `geometry_overlay.png`.
- Generate simple history and mass-balance plot PNGs from CSV rows.

## Profile Extraction Contract

Add `profile_extractors.py` with:

- `extract_centerline_profile(fields) -> list[dict]`
- `extract_throat_profile(fields, flap_center_s=0.048) -> list[dict]`
- `extract_downstream_profiles(fields, offsets=(0.004, 0.010, 0.020, 0.040), flap_center_s=0.048) -> list[dict]`
- `write_profile_csv(path, rows) -> None`

Profile CSVs must include columns:

- `s`
- `y`
- `u`
- `Uz`
- `Uy`
- `speed`
- `fluid_mask`
- `near_solid_mask`

The report must summarize:

- centerline maximum `u`
- centerline maximum `s`
- throat maximum `u`
- throat mean `u`
- downstream profile peak values

## Report Contract

Add `report_builder.py` to generate `validation_report.md`.

The report must include these sections:

- `# ANSYS Vertical Flap Fixed-Flow Step 3 Validation Report`
- `## Scope`
- `## Field Summary`
- `## Visual Outputs`
- `## Quality Gates`
- `## Interpretation`
- `## Required Next Solver Improvement`

The field summary must report:

- `max_u`
- `max_speed`
- `centerline_max_u`
- `mass_imbalance_rel`
- `divergence_linf`
- `divergence_l2`
- `poisson_residual_linf`

The report must state that the current result is a Fluent-style visualization of the Step 2 solver output, not Fluent parity validation.

The report must not hide the current solver numerical issues. If `poisson_residual_linf` or `divergence_linf` is large, they must appear in the report and the quality table.

## Main API Contract

Add `postprocess_fluent_style.py` with:

```python
def run_fluent_style_postprocess(
    final_fields_path: str | Path,
    solver_history_path: str | Path,
    mass_balance_path: str | Path,
    step2_manifest_path: str | Path,
    output_root: str | Path,
    config: dict | None = None,
) -> dict:
    ...
```

Return shape:

```python
{
  "output_root": ".../rendered_results/step3_fluent_style",
  "figures": {...},
  "profiles": {...},
  "report": ".../validation_report.md",
  "manifest": ".../case_manifest_step3.json",
  "quality": {...},
  "claims": {
    "fluent_parity": "not_claimed",
    "fsi": "not_claimed"
  }
}
```

## Runner Contract

Add command:

`D:\working\taichi\env\python.exe validation_cases\ansys_vertical_flap_fixed_flow\run_fixed_flap_postprocess.py`

Runner behavior:

1. Check Step 2 artifacts.
2. If `final_fields.npz` is missing, run the Step 2 solver runner logic first.
3. Run Step 3 postprocess.
4. Print a JSON summary with case, step, output root, report, quality, and claims.

## Required Tests

Use test-first development.

Add:

`tests/integration/test_ansys_vertical_flap_fixed_flow_step3_postprocess.py`

Before implementation, run it and confirm RED.

Tests must verify:

1. Step 3 consumes Step 2 artifacts.
2. The postprocess API can be imported.
3. All required PNG, CSV, MD, and manifest outputs are generated.
4. PNG files have valid PNG magic bytes and are larger than 1 KB.
5. Profile CSV files have headers, at least 10 rows, and required columns.
6. `validation_report.md` contains no-parity/no-FSI language.
7. `validation_report.md` contains `poisson_residual_linf`, `divergence_linf`, `mass_imbalance_rel`, and `diagnostic_only_not_parity`.
8. `case_manifest_step3.json` states:
   - `claims.fluent_parity == not_claimed`
   - `claims.fsi == not_claimed`
   - Step 2 final fields/logs/manifest are the sources.
   - `forbidden_sources.traction_shared_snapshot_diagnostics == not_used`
9. The quality gates do not silently pass incompressibility when thresholds are exceeded.
10. Step 1 and Step 2 tests still pass.

## Verification Commands

RED:

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step3_postprocess -v`

GREEN:

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step1 -v`

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step2_solver -v`

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step3_postprocess -v`

`& 'D:\working\taichi\env\python.exe' validation_cases\ansys_vertical_flap_fixed_flow\run_fixed_flap_postprocess.py`

`git diff --check`

## Commit And Push Contract

Use the current branch:

`codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`

Required commits:

1. `test: reproduce fixed flap fluent-style postprocess contract`
2. `validation: add fixed flap fluent-style postprocess artifacts`

Push to:

`origin/codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`

## Completion Criteria

This goal is complete only when:

- The detailed goal file exists and is referenced by the active Codex goal.
- RED Step 3 test is committed before implementation.
- Step 3 source files and runner exist.
- Step 3 artifact bundle exists.
- Step 1 tests pass.
- Step 2 tests pass.
- Step 3 tests pass.
- The Step 3 runner succeeds.
- `git diff --check` passes.
- No old FSI rendered results or traction snapshot artifacts are committed.
- The branch is pushed to origin.
