# ANSYS Vertical-Flap 2D to 3D Slab Equivalence

This note defines the comparison contract for the ANSYS Fluent vertical-flap
case and the local HIBM-MPM runner.

## Coordinate Models

The official Fluent tutorial is a conceptual 2D case. It models the lower
symmetry half of a `0.10 m x 0.04 m` duct and reports the flap response in that
2D setup.

The local HIBM-MPM implementation runs on a 3D Cartesian grid. The 2D case is
therefore represented as a thin slab:

| Quantity | Local Field | Meaning |
| --- | --- | --- |
| streamwise axis | `z` | duct direction and flap thickness direction |
| vertical axis | `y` | half-domain height direction |
| out-of-plane axis | `x` | extrusion direction used to make the 2D case 3D |
| extrusion depth | `VerticalFlapFsiConfig.span_m` | slab depth in `x` |
| flap streamwise thickness | `VerticalFlapFsiConfig.flap_thickness_m` | physical thickness from `z = 0.050 m` to `z = 0.053 m` |

`flap_thickness_m` is not the 2D-to-3D extrusion depth. The extrusion depth is
`span_m`.

## Direct vs Depth-Normalized Quantities

3D integrated quantities are totals over the slab depth. They should not be
compared directly against 2D force-like quantities unless the reference is also
known to use the same depth.

Compare these directly only when both sides use the same extrusion depth:

| 3D Total | Unit | Depth-Normalized Quantity | Unit |
| --- | --- | --- | --- |
| `interface_force_total_n` | `N` | `interface_force_per_depth_npm` | `N/m` |
| `pressure_force_total_n` | `N` | `pressure_force_per_depth_npm` | `N/m` |
| `solid_mass_total_kg` | `kg` | `solid_mass_per_depth_kgpm` | `kg/m` |
| `marker_total_area_m2` | `m^2` | `marker_total_area_per_depth_m` | `m` |

If `span_m` doubles under the same conceptual 2D physics, the expected scaling
is:

- total marker area doubles;
- total solid mass doubles;
- total integrated interface force doubles;
- marker area per depth stays fixed;
- solid mass per depth stays fixed;
- force per depth stays fixed;
- displacement should remain depth-invariant when both force and mass scale
  together.

## Diagnostics Added

`benchmarks/official/solid_mpm_fsi_runner.py` exposes
`slab_equivalence_diagnostics(...)`. The ANSYS vertical-flap case adapter and
archived validation runner report:

- `conceptual_coordinate_model`
- `runtime_discretization_model`
- `extrusion_depth_m`
- `extrusion_depth_source`
- `span_is_extrusion_depth`
- `flap_streamwise_thickness_m`
- `flap_thickness_is_streamwise_not_extrusion`
- `marker_total_area_m2`
- `marker_total_area_per_depth_m`
- `interface_force_total_n`
- `interface_force_per_depth_npm`
- `pressure_force_total_n`
- `pressure_force_per_depth_npm`
- `solid_mass_total_kg`
- `solid_mass_per_depth_kgpm`
- `out_of_plane_boundary_policy`
- `out_of_plane_boundary_residual_modeling_error`
- `fluent_parity_claimed`

The diagnostics are report-only. They do not change the fluid solver, HIBM-MPM
coupling, MPM solid integration, Fluent reference data, or benchmark acceptance
metrics.

## Boundary Caveat

The slab is currently reported with:

`out_of_plane_boundary_policy = finite_slab_x_faces_no_periodic_or_slip`

That means the local 3D run is a finite-thickness slab diagnostic, not a strict
2D periodic/slip extrusion. Reports set
`out_of_plane_boundary_residual_modeling_error = true` for this case. This is a
modeling-risk flag, not a tuned correction.

## Current Interpretation

A result mismatch can come from a 2D/3D comparison error if total `N`, `kg`, or
`m^2` quantities are compared without normalizing by `extrusion_depth_m`.

The formal benchmark runner now makes the intended scaling explicit:
`span_m` is the out-of-plane depth, marker area and solid mass scale with that
depth, and the report includes both total and per-depth quantities. The remaining
known risk is the out-of-plane boundary condition: until strict periodic/slip
extrusion is available, a 3D slab run should not be described as full Fluent
parity.
