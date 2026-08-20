# Simulation Core Module Map

This repository treats `simulation_core/` as the reusable solver core. Root-level
Python files are intentionally limited to `simulation_core/__init__.py`; real
implementation belongs in the functional packages below.

## Functional Packages

| Package | Responsibility | Modify Here When |
| --- | --- | --- |
| `simulation_core/fluids/` | Fluid grids, fluid solver, pressure projection, pressure outlet cleanup, fluid reports. | Changing flow discretization, pressure projection, velocity/pressure boundary rows, outlet mass balance, or fluid diagnostics. |
| `simulation_core/coupling/` | Generic FSI coupling primitives, IBM/projected IBM, interface reaction, pressure interface policies, pressure sample pairs, moving-boundary pair maps, triangle-surface diagnostics. | Changing FSI force balance, pressure interface semantics, IBM coupling, interface maps, or shared surface-force transfer logic. |
| `simulation_core/coupling/hibm_mpm/` | Sharp HIBM-MPM coupling, surface markers, IB-node search, velocity boundary rows, pressure Neumann rows, and fluid-to-solid load transfer. | Changing HIBM-MPM paper-aligned coupling, marker search/classification, pressure Neumann assembly, no-slip rows, full-stress sampling, or marker-to-MPM force scatter. |
| `simulation_core/solids/` | MPM solid solvers, including Neo-Hookean particles and Mooney shell implementation. | Changing solid time integration, particle/shell state, material force application, or MPM external-force consumption. |
| `simulation_core/materials/` | Constitutive material models and material conversion helpers. | Changing Neo-Hookean or Ecoflex material behavior, stress probes, or material-unit conversions. |
| `simulation_core/geometry_tools/` | CAD parsing, STEP tessellation, coordinate models, fluid-domain geometry, and reusable surface meshes. | Changing CAD/surface mesh handling, domain geometry, boundary-region descriptors, or coordinate-system models. |
| `simulation_core/diagnostics/` | Validation helpers, CFL/time-step controllers, field checks, and Taichi runtime bootstrap. | Changing validation/report helpers, CFL substep rules, or shared runtime initialization. |

## Removed Legacy Entry Points

The old root-level compatibility modules have been removed. Import from the
functional package paths below; `simulation_core/__init__.py` no longer
registers `sys.modules` aliases for these names.

| Legacy Import | Real Implementation |
| --- | --- |
| `simulation_core.fluid` | `simulation_core.fluids` |
| `simulation_core.fsi_coupling` | `simulation_core.coupling.fsi_coupling` |
| `simulation_core.hibm` | `simulation_core.coupling.hibm` |
| `simulation_core.hibm_mpm` | `simulation_core.coupling.hibm_mpm` |
| `simulation_core.interface_pair` | `simulation_core.coupling.interface_pair` |
| `simulation_core.moving_boundary` | `simulation_core.coupling.moving_boundary` |
| `simulation_core.pressure_interface` | `simulation_core.coupling.pressure_interface` |
| `simulation_core.pressure_sample_pairs` | `simulation_core.coupling.pressure_sample_pairs` |
| `simulation_core.projected_ibm` | `simulation_core.coupling.projected_ibm` |
| `simulation_core.tri_surface` | `simulation_core.coupling.tri_surface` |
| `simulation_core.runtime` | `simulation_core.diagnostics.runtime` |
| `simulation_core.neo_hookean_mpm` | `simulation_core.solids.neo_hookean_mpm` |
| `simulation_core.mooney_shell_mpm` | `simulation_core.solids.mooney_shell` |
| `simulation_core.geometry` | `simulation_core.geometry_tools.surface_mesh` |
| `simulation_core.coordinate_models` | `simulation_core.geometry_tools.coordinate_models` |
| `simulation_core.fluid_domain` | `simulation_core.geometry_tools.fluid_domain` |
| `simulation_core.cad_import` | `simulation_core.geometry_tools.cad_import` |
| `simulation_core.cad_tessellation` | `simulation_core.geometry_tools.cad_tessellation` |
| `simulation_core.hyperelastic` | `simulation_core.materials.hyperelastic` |
| `simulation_core.validation` | `simulation_core.diagnostics.validation` |
| `simulation_core.time_stepping` | `simulation_core.diagnostics.time_stepping` |

`simulation_core/__init__.py` remains the package-level public API facade. Its
`__all__` list is intentionally broader than current in-repository imports so
external callers can keep using stable package-level symbols; zero internal
users alone is not a deletion signal. New project code should import from the
functional packages directly, and new root exports should be added only when the
symbol is intentionally public.

The superseded `simulation_core.drivers` package has no replacement facade.
For the official ANSYS rectangular-solid benchmark, FSI execution is
consolidated into one numerical entry point:
`benchmarks.official.solid_mpm_fsi_runner.run_hibm_mpm_fsi`.
`cases.ansys_vertical_flap_fsi.run_ansys_vertical_flap_benchmark` adds only
case metadata and official-report validation around that solver. `FsiCaseSpec`
lives with the official benchmark contracts in
`benchmarks.official.official_benchmark_solver`.

This consolidation is not a claim that unlike physical cases share one runner.
The Squid, Turek-Hron, and COMSOL cases retain case-specific geometry and model
assembly. Where a case uses HIBM-MPM, it must use the canonical sharp
formulation rather than adding a parallel projected/reduced workflow.

## Migration Summary

Moved real implementations out of root compatibility modules and removed the
root wrapper files:

- `fsi_coupling.py` -> `coupling/fsi_coupling.py`
- `hibm.py` -> `coupling/hibm.py`
- `interface_pair.py` -> `coupling/interface_pair.py`
- `moving_boundary.py` -> `coupling/moving_boundary.py`
- `pressure_interface.py` -> `coupling/pressure_interface.py`
- `pressure_sample_pairs.py` -> `coupling/pressure_sample_pairs.py`
- `projected_ibm.py` -> `coupling/projected_ibm.py`
- `tri_surface.py` -> `coupling/tri_surface.py`
- `runtime.py` -> `diagnostics/runtime.py`

## Navigation Rules

- HIBM-MPM paper coupling fixes go in `simulation_core/coupling/hibm_mpm/`.
- Generic IBM/projected-IBM and pressure-interface fixes go in `simulation_core/coupling/`.
- Fluid pressure projection, outlet cleanup, and grid changes go in `simulation_core/fluids/`.
- Solid MPM behavior goes in `simulation_core/solids/`.
- Material laws go in `simulation_core/materials/`.
- CAD, surface mesh, coordinate, and domain geometry changes go in `simulation_core/geometry_tools/`.
- Validation helpers and runtime initialization go in `simulation_core/diagnostics/`.
- The reusable ANSYS rectangular-solid HIBM-MPM numerical loop goes only in
  `benchmarks/official/solid_mpm_fsi_runner.py`; do not add parallel driver or
  generic-solver entry points.
- ANSYS vertical-flap wrappers may add metadata and report validation, but must
  delegate their numerical work to `run_hibm_mpm_fsi`.
- Other physical cases may keep case-specific assembly, but must not expose a
  second coupling formulation for the same case workflow.
- Fluent benchmark/parity runners should use these package paths and must not introduce case-specific solver logic under `simulation_core/`.

Legacy module names are not installed. New project code and external migration
guides should use the functional package path.
