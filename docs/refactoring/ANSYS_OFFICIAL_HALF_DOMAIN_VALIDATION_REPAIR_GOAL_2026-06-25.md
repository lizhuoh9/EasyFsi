# ANSYS official half-domain validation repair goal - 2026-06-25

## Objective

Repair the ANSYS vertical-flap validation path so that the repository no longer
mixes the official Fluent half-domain case with the older full-domain two-flap
runner semantics. The end state must provide an honest, reproducible local
HIBM-MPM validation workflow for the official Fluent tutorial geometry, with
artifact consistency tests and no false Fluent parity claims.

This goal is driven by the review note attached on 2026-06-25. The review
accepted the previous archive direction, but identified that the current archive
and formal benchmark path are not yet credible as Fluent parity validation
because the evidence chain and runner geometry are inconsistent.

## Source case constraints

The source case is the ANSYS Fluent tutorial "Modeling Two-Way
Fluid-Structure Interaction (FSI) Within Fluent".

The implementation and reports must preserve these facts:

- The official case is a 2D planar duct vertical-flap FSI tutorial.
- Full duct length is `0.10 m`.
- Full duct height is `0.04 m`.
- Fluent models only the lower symmetry half-domain; modeled height is `0.02 m`.
- Display may be mirrored about the centerline to show two flaps, but the solve
  is one modeled lower flap.
- Flap height is `0.01 m`.
- Flap thickness is `0.003 m`.
- Flap streamwise location from the official mesh is `z = 0.050 m` to
  `z = 0.053 m`, not a centered thickness around `z = 0.050 m`.
- Inlet velocity is `10.0 m/s`.
- Outlet is a pressure outlet.
- Solid is silicone rubber with density `1600 kg/m^3`, Young's modulus
  `1e6 Pa`, and Poisson ratio `0.47`.
- Fluent uses a linear-elastic structural model; the local HIBM-MPM path uses
  Neo-Hookean MPM unless a later explicit task implements linear-elastic MPM.

## Non-goals and claim boundaries

Do not claim pointwise Fluent parity.

Do not claim a completed 50-step official validation unless a fresh 50-step run
with consistent metadata and physical gates is actually produced.

Do not continue the full-domain two-flap hand-built model as the main official
validation path. It may remain as a solver stress regression or failure evidence
only.

Do not commit ANSYS raw tutorial assets such as `fsi_2way.zip`, `flap.msh`, or
`steady_fluid_flow.jou`. Commit extracted parameters and local HIBM-MPM outputs
only.

Do not bypass pressure-solve guards or MPM out-of-bounds guards. Failed solves
must remain explicit evidence.

## Required repairs

### 1. Repair official half-domain evidence-chain consistency

Create or promote a real official half-domain runner/reporting path instead of
monkey-patching a full-domain two-flap runner whose report hardcodes
full-domain semantics.

At minimum, every official-half-domain manifest, summary, report, history, and
field metadata must agree on:

- `case = ansys-fluent-official-half-domain-single-flap`
- `official_half_domain = true`
- `full_domain_two_flap = false`
- `flap_count_modeled = 1`
- `flap_count_displayed_after_symmetry_mirror = 2`
- `marker_count_actual = 2 * markers_per_face`
- `flow_projection_iterations_actual = <actual runtime projection iterations>`
- `modeled_grid_nodes = [4, ny, nz]`
- `display_grid_after_symmetry_mirror = [4, 2*ny, nz]`

The nested config/report fields must not silently contradict the actual run. If
a copied legacy config field remains for compatibility, the archive README must
identify it clearly and tests must still expose the actual runtime count.

### 2. Repair the formal benchmark runner geometry

The formal ANSYS vertical-flap benchmark runner must not leave out-of-plane
side bypasses for a 2D planar official case.

Change the official vertical-flap solid box from:

```python
x_min = 0.15 * span
x_max = 0.85 * span
```

to:

```python
x_min = 0.0
x_max = span
```

This full-span solid/flap geometry must be covered by tests.

### 3. Repair formal benchmark marker coverage

The formal benchmark runner must place markers on both streamwise faces of the
thin flap:

- downstream/upstream plus-z face
- upstream/downstream minus-z face

The marker count must become `2 * config.marker_count` or the equivalent
actual marker-capacity field must be explicitly represented. Marker normals
must be opposite (`+z` and `-z`). Force and pressure diagnostics must make clear
that both faces are sampled.

This dual-face marker behavior must be covered by tests.

### 4. Add solid CFL-driven substep selection

The earlier full-domain high-resolution failure showed that
`solid_substeps=200` can push MPM particles out of the background grid. The
estimated P-wave CFL explains why `solid_substeps=1000` stabilized the run.

Implement a reusable solid-substep helper:

```python
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
cp = sqrt((lam + 2 * mu) / rho)
solid_substeps = ceil(cp * dt / (cfl_target * min_h))
```

Use a conservative default target such as `0.45`, while preserving explicit
user-provided `solid_substeps` when it is already higher. Report the selected
substep count, estimated wave speed, minimum spacing, target CFL, and estimated
CFL.

### 5. Add artifact consistency tests

Add tests that read the archived official-half-domain artifact set and enforce
cross-file consistency:

- manifest, summary, report, process, history, and render metadata refer to the
  same official half-domain case.
- no official-half artifact reports `full_domain_two_flap=true`.
- modeled/display flap counts are correct.
- grid and mirrored-grid metadata are consistent.
- `fields.npz` shapes match the reported modeled grid.
- marker count in the field snapshot matches two streamwise faces.
- actual projection iterations are represented as `4096` in the archive
  evidence chain.

This test should fail on the previous polluted report semantics and pass after
the repair.

### 6. Preserve failure evidence

Keep the two failed/superseded full-domain attempts as evidence:

- `p1080`: pressure solve did not converge before traction sampling.
- `p4096 + solid_substeps=200`: MPM particles left the background grid.

Do not reinterpret these failures as successful physical validation.

### 7. Documentation updates

Update the validation archive README and/or source metadata so that the archive
plainly states:

- It is a local HIBM-MPM reproduction of the official Fluent parameters.
- It is not an ANSYS Fluent solve.
- It is not pointwise Fluent parity.
- It is one-step evidence unless a later validated 50-step run is produced.
- The official modeling convention is half-domain solve plus mirrored display.
- The formal benchmark runner has been repaired to full-span flap geometry and
  two-face pressure markers.

## Validation requirements

Before commit and push:

1. Compile affected Python files.
2. Run focused ANSYS vertical-flap tests.
3. Run the artifact consistency test.
4. Run `git diff --check`.
5. Verify `git status --short` before staging.
6. Stage only files that belong to this goal.
7. Commit and push to the configured GitHub remote.

Expected minimum commands:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py
& 'D:\working\taichi\env\python.exe' -m unittest tests.cases.test_ansys_vertical_flap_fsi -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\integration -p '*ansys*vertical*flap*.py' -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests -p '*ansys*vertical*flap*.py' -v
git diff --check
```

If full test discovery is too slow or environment-blocked, record the exact
failure and still run the narrowest tests that cover this goal.

## Deliverables

- Detailed goal file: this document.
- Short `/goal` value referencing this document.
- Code changes for official half-domain evidence semantics.
- Code changes for full-span official flap geometry.
- Code changes for dual streamwise-face markers.
- Code changes for solid CFL-driven substep reporting/selection.
- Artifact consistency tests.
- Updated validation archive docs/metadata.
- Commit hash and pushed branch reported to the user.
