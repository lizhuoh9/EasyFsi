# ANSYS Fluent official vertical-flap case rerun with local HIBM-MPM

This directory archives the June 25, 2026 local HIBM-MPM rerun of the official
ANSYS Fluent two-way FSI vertical-flap tutorial.

## What was simulated

- Source case: ANSYS Fluent tutorial "Modeling Two-Way Fluid-Structure
  Interaction (FSI) Within Fluent".
- Official geometry: 2D duct, length `0.10 m`, full height `0.04 m`.
- Official modeling convention: only the lower symmetry half is modeled.
- Modeled half-domain in this rerun: `0.10 m x 0.02 m`.
- Flap: one lower-half flap, height `0.01 m`, thickness `0.003 m`, streamwise
  extent `z = 0.050 m` to `0.053 m`.
- Display convention: the half-domain result is mirrored about the duct
  centerline to match the Fluent tutorial's two-flap view.
- Boundary conditions: `10.0 m/s` velocity inlet at `zmax`, pressure outlet at
  `zmin`, symmetry at the half-domain centerline.
- Solver used: local HIBM-MPM `advance_hibm_mpm_sharp_mpm_step`, not ANSYS
  Fluent.

The raw ANSYS tutorial zip/mesh/journal was used only to confirm the source
case dimensions and journal settings. Those third-party tutorial assets are not
redistributed here.

## Main run

Command used from the repository root:

```powershell
& 'D:\working\taichi\env\python.exe' tmp\run_official_fluent_half_domain_hibm_mpm_4x320x640.py
```

Archived reproduction script:

```powershell
& 'D:\working\taichi\env\python.exe' validation\ansys-fluent-official-half-domain-hibm-mpm-2026-06-25\scripts\run_official_fluent_half_domain_hibm_mpm_4x320x640.py
```

Run parameters:

- Modeled grid: `4 x 320 x 640`.
- Mirrored display grid: `4 x 640 x 640`.
- Time steps: `1`.
- Time step size: `0.0005 s`.
- Pressure projection iterations: `4096`.
- Fluid substeps: `2`.
- Solid substeps: `1000`.
- Solid particles: `1 x 80 x 24`.
- Markers per face: `84`.

Result:

- Status: completed.
- Wall time: `738.2633276999986 s`.
- Stress markers: `168/168` valid.
- Marker force z: `-3.7221164064638377e-4 N`.
- Maximum displacement: `6.000609005241131e-7 m`.
- Flow speed p99: `25.523324451446538 m/s`.
- Flow speed p999: `28.5929762916567 m/s`.
- Flow speed max: `60.13129425048828 m/s`.

## Data inventory

- `data/official_half_grid4x320x640_step1_p4096_s1000_fields.npz`: compressed
  field snapshot with velocity, pressure, obstacle mask, solid particles, and
  markers.
- `data/official_half_grid4x320x640_step1_p4096_s1000_report.json`: full solver
  report.
- `data/official_half_grid4x320x640_step1_p4096_s1000_history.csv`: per-step
  summary.
- `data/official_half_grid4x320x640_step1_p4096_s1000_process.json`: process
  status.
- `data/official_half_grid4x320x640_step1_p4096_s1000_summary.json`: compact
  run summary.
- `data/official_half_grid4x320x640_step1_p4096_s1000_manifest.json`: run
  manifest.
- `data/official_half_grid4x320x640_step1_p4096_s1000_fields_mirrored_velocity_pipe_style.json`:
  render metadata.
- `figures/official_half_grid4x320x640_step1_p4096_s1000_fields_mirrored_velocity_pipe_style.png`:
  Fluent-style mirrored velocity magnitude image.
- `source_metadata/official_fluent_case_extracted_parameters.json`: extracted
  official-case parameters and local run settings.
- `failures/`: failed or superseded attempts retained as evidence.

## Render

The figure in `figures/` uses the modeled half-domain field, mirrors it about
the symmetry centerline, flips the streamwise axis for the same left-to-right
visual orientation as the provided Fluent-style reference image, and uses a
vertical scientific-notation color bar.

Render command:

```powershell
$env:FIELDS_NPZ='validation\ansys-fluent-official-half-domain-hibm-mpm-2026-06-25\data\official_half_grid4x320x640_step1_p4096_s1000_fields.npz'
& 'C:\Users\lizhu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' validation\ansys-fluent-official-half-domain-hibm-mpm-2026-06-25\scripts\render_official_half_domain_mirrored_pipe_style.py
```

## Errors and findings while running

1. The earlier full-domain two-flap visualization was not the official Fluent
   modeling convention. The official Fluent tutorial uses a lower symmetry
   half-domain and mirrors the display.
2. The local ANSYS installation under `C:\Program Files\ANSYS Inc\v251` contains
   only common files in this environment; no `fluent.exe` or Fluent launcher was
   found. Therefore this archive is a local HIBM-MPM rerun of the Fluent case,
   not an ANSYS Fluent solve.
3. A very fine full-domain `4 x 640 x 640` hand-built two-flap attempt with
   `1080` projection iterations failed before stress sampling because the
   pressure solve did not converge: `cg_relative_residual_max=0.015882`.
4. The same superseded full-domain attempt with `4096` projection iterations
   but only `200` solid substeps failed in the MPM solid step with
   `108 of 3840 MPM particles are outside the background grid`.
5. Increasing the superseded full-domain solid substeps to `1000` made that
   non-official geometry run complete, which motivated using `solid_substeps=1000`
   for the official half-domain rerun.
6. The copied base runner serializes `config.flow_projection_iterations` with
   the dataclass default (`1080`) inside the nested config report, while the
   actual sharp-step call used the explicit manifest/summary value (`4096`).
   For this archive, use `*_manifest.json` and `*_summary.json` as the source of
   truth for the actual projection-iteration count.

## Scope boundary

This is a one-step high-resolution HIBM-MPM rerun of the official Fluent
vertical-flap setup for artifact-backed visualization and diagnostics. It is not
a 50-step Fluent parity claim, and it does not include ANSYS Fluent-generated
case/data files.
