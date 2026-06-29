# ANSYS Vertical Flap Fixed-Flow Step 1 Goal - 2026-07-01

## Source Request

Implement the Step 1 package described in the user-provided file:

`C:\Users\lizhu\.codex\attachments\021243b3-04e5-4ca3-b213-a0d541e1b7dd\pasted-text.txt`

Current branch:

`codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`

Starting HEAD:

`6010e7328fa306a7191fc262ef3c0607b4e71f89`

## Objective

Add a standalone ANSYS vertical-flap fixed-flap flow validation preprocessing package and case runner. This package must create a clean fixed-solid fluid-domain contract for the next solver step, so that later velocity-contour work is based on a true fluid field contract rather than on EasyFSI shared-snapshot or traction diagnostics.

The finished Step 1 must provide:

- A case config for the fixed-flap official flow preprocessing setup.
- A runnable command:
  `python validation_cases/ansys_vertical_flap_fixed_flow/run_fixed_flap_flow.py`
- Geometry mask generation for the full duct, walls, and fixed upper/lower flap blocks.
- Boundary-condition map generation for inlet, outlet, no-slip walls, and no-slip flap solids.
- Initial field generation with signed streamwise velocity convention.
- A geometry preview PNG artifact.
- A manifest that documents sign conventions, generated files, and explicit non-claims.
- Focused tests that fail before implementation and pass after implementation.

## Physical And Data Contract

This is a preprocessing and artifact-contract step only. It must not claim Fluent parity or generate a synthetic velocity contour pretending to be the official Fluent result.

Required coordinate convention:

- Arrays use shape `(ny, ns)`.
- Axis 0 is physical/display vertical coordinate `y`.
- Axis 1 is displayed streamwise coordinate `s`.
- Physical solver coordinate is `z = -s`.
- A left-to-right displayed flow has negative physical `Uz`.
- Any displayed streamwise velocity must use `streamwise_minus_Uz = -Uz`.

Required fixed-flap case values:

- `grid.ns = 360`
- `grid.ny = 128`
- `geometry.duct_length = 0.120`
- `geometry.duct_height = 0.040`
- `geometry.flap_center_s = 0.048`
- `geometry.flap_thickness = 0.0030`
- `geometry.gap_height = 0.0100`
- `geometry.wall_cells_are_solid = true`
- `fluid.rho = 1000.0`
- `fluid.mu = 0.001`
- `boundary_conditions.inlet_Uz = -7.0`
- `boundary_conditions.inlet_Uy = 0.0`
- `boundary_conditions.outlet_pressure = 0.0`
- `output.root = validation_runs/ansys_vertical_flap_fixed_flow`

## Required Files

Create the case and validation package under:

- `validation_cases/ansys_vertical_flap_fixed_flow/config.yaml`
- `validation_cases/ansys_vertical_flap_fixed_flow/run_fixed_flap_flow.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/__init__.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/geometry.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/bc.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/preprocess_fixed_flap.py`

Add focused integration tests under:

- `tests/integration/test_ansys_vertical_flap_fixed_flow_step1.py`

## Required Generated Artifacts

The default runner must generate and keep the following artifact bundle:

- `validation_runs/ansys_vertical_flap_fixed_flow/preprocess/geometry_mask.npz`
- `validation_runs/ansys_vertical_flap_fixed_flow/preprocess/bc_map.npz`
- `validation_runs/ansys_vertical_flap_fixed_flow/fields/initial_fields.npz`
- `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/geometry_preview.png`
- `validation_runs/ansys_vertical_flap_fixed_flow/case_manifest.json`

The manifest must explicitly record:

- Case name.
- Step scope as fixed-flap flow preprocessing only.
- No Fluent parity claim.
- No FSI claim.
- No solver-step claim.
- The sign convention `left_to_right_display_flow_has_Uz_negative`.
- The display helper `streamwise_display_velocity = -Uz`.
- Paths for every generated artifact.

## Required Tests

Use test-first development:

1. Add a failing integration test that imports the new package and asserts the Step 1 contract.
2. Run the test before implementation and preserve the RED result.
3. Implement the minimum code needed for the test to pass.
4. Generate the default artifact bundle.
5. Re-run the focused test and preserve the GREEN result.

The tests must verify:

- The runner/config path exists.
- The preprocessing API can generate an isolated temporary artifact bundle.
- Geometry arrays have shape `(128, 360)`.
- `s`, `y`, `S`, `Y`, and `Z` are present and consistent, with `Z == -S`.
- The top and bottom wall rows are solid.
- The upper/lower flap blocks are solid.
- The middle gap through the flap is fluid.
- The inlet mask occupies the left fluid boundary and is not applied to solid cells.
- The outlet mask occupies the right fluid boundary and is not applied to solid cells.
- Inlet `Uz` values are negative and inlet `Uy` values are zero.
- Initial fields include `Uz`, `Uy`, `p`, and `streamwise_minus_Uz`.
- `streamwise_minus_Uz == -Uz`.
- The PNG preview has a valid PNG header.
- The default committed artifacts match the manifest and preserve the non-claim language.

## Implementation Constraints

- Do not modify Fluent source-export importer behavior in this step.
- Do not modify existing `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics`.
- Do not use `traction_shared_snapshot_diagnostics/step020_fields.npz` as a velocity-contour parity source.
- Do not add solver logic for Step 2 in this change.
- Do not create synthetic red-jet velocity fields for visual matching.
- Do not edit `source_exports` artifacts unless the user explicitly requests it.
- Keep the implementation deterministic and small.
- Use `D:\working\taichi\env\python.exe` for local validation.
- If `matplotlib` is unavailable in that interpreter, provide a no-dependency PNG fallback rather than failing the runner.

## Verification Commands

Expected RED command:

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step1 -v`

Expected GREEN commands:

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step1 -v`

`& 'D:\working\taichi\env\python.exe' validation_cases\ansys_vertical_flap_fixed_flow\run_fixed_flap_flow.py`

`git diff --check`

## Completion Criteria

The goal is complete only when:

- The detailed goal file exists and is referenced by the active Codex goal.
- The RED test was run before implementation.
- The implementation files and case runner exist.
- The default artifact bundle exists.
- The focused GREEN test passes.
- The runner command succeeds.
- The diff is self-reviewed and does not include unrelated old rendered FSI artifacts.
- A RED test commit and a GREEN implementation/artifact commit are created.
- The branch is pushed to the configured remote.
