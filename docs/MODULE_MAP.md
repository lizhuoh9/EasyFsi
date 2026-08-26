# Simulation Core Module Map

This repository treats `simulation_core/` as the reusable solver core. Root-level
Python files are intentionally limited to `simulation_core/__init__.py`; real
implementation belongs in the functional packages below.

## Functional Packages

| Package | Responsibility | Modify Here When |
| --- | --- | --- |
| `simulation_core/fluids/` | Fluid grids, fluid solver, pressure projection, pressure outlet cleanup, fluid reports. | Changing flow discretization, pressure projection, velocity/pressure boundary rows, outlet mass balance, or fluid diagnostics. |
| `simulation_core/coupling/` | Generic FSI coupling primitives, IBM/projected IBM, interface reaction, pressure interface policies, pressure sample pairs, moving-boundary pair maps, triangle-surface diagnostics. | Changing FSI force balance, pressure interface semantics, IBM coupling, interface maps, or shared surface-force transfer logic. |
| `simulation_core/coupling/hibm_mpm/` | Sharp HIBM-MPM coupling, surface markers, IB-node search, the canonical velocity component-face ledger, pressure Neumann rows, and fluid-to-solid load transfer. | Changing HIBM-MPM paper-aligned coupling, marker search/classification, component-face no-slip assembly, pressure Neumann assembly, full-stress sampling, or marker-to-MPM force scatter. |
| `simulation_core/solids/` | MPM solid solvers, including Neo-Hookean particles and Mooney shell implementation. | Changing solid time integration, particle/shell state, material force application, or MPM external-force consumption. |
| `simulation_core/materials/` | Constitutive material models and material conversion helpers. | Changing Neo-Hookean or Ecoflex material behavior, stress probes, or material-unit conversions. |
| `simulation_core/geometry_tools/` | CAD parsing, STEP tessellation, coordinate models, fluid-domain geometry, and reusable surface meshes. | Changing CAD/surface mesh handling, domain geometry, boundary-region descriptors, or coordinate-system models. |
| `simulation_core/diagnostics/` | Validation helpers, CFL/time-step controllers, field checks, and Taichi runtime bootstrap. | Changing validation/report helpers, CFL substep rules, or shared runtime initialization. |
| `simulation_core/drivers/` | Shared runtime-adapter contracts and the case-agnostic FSI trial engine used by adapter-based cases. | Changing shared physical-step ownership, generic coupling convergence, runtime-adapter contracts, or driver result envelopes. |

## Removed Legacy Entry Points

The old root-level compatibility modules have been removed. Import from the
functional package paths below; `simulation_core/__init__.py` no longer
registers `sys.modules` aliases for these names.

| Legacy Import | Real Implementation |
| --- | --- |
| `simulation_core.fluid` | `simulation_core.fluids` |
| `simulation_core.fsi_coupling` | Removed; use `simulation_core.drivers.generic_fsi_solver` for FSI orchestration and `simulation_core.coupling.interface_forces` for force balance. |
| `simulation_core.generic_fsi_solver` | `simulation_core.drivers.generic_fsi_solver` |
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
`__all__` list contains the current deliberate public surface; deleted solver
modes and compatibility symbols are not re-exported. New project code should
import from the functional packages directly.

## Migration Summary

Moved real implementations out of root compatibility modules and removed the
root wrapper files:

- the old `fsi_coupling.py` implementations were deleted; orchestration now
  lives only in `drivers/generic_fsi_solver.py`, while stateless force helpers
  live in `coupling/interface_forces.py`
- `hibm.py` -> `coupling/hibm.py`
- `interface_pair.py` -> `coupling/interface_pair.py`
- `moving_boundary.py` -> `coupling/moving_boundary.py`
- `pressure_interface.py` -> `coupling/pressure_interface.py`
- `pressure_sample_pairs.py` -> `coupling/pressure_sample_pairs.py`
- `projected_ibm.py` -> `coupling/projected_ibm.py`
- `tri_surface.py` -> `coupling/tri_surface.py`
- `generic_fsi_solver.py` -> `drivers/generic_fsi_solver.py`
- `runtime.py` -> `diagnostics/runtime.py`

## Navigation Rules

- HIBM-MPM paper coupling fixes go in `simulation_core/coupling/hibm_mpm/`.
- Generic IBM/projected-IBM and pressure-interface fixes go in `simulation_core/coupling/`.
- Fluid pressure projection, outlet cleanup, and grid changes go in `simulation_core/fluids/`.
- Solid MPM behavior goes in `simulation_core/solids/`.
- Material laws go in `simulation_core/materials/`.
- CAD, surface mesh, coordinate, and domain geometry changes go in `simulation_core/geometry_tools/`.
- Validation helpers and runtime initialization go in `simulation_core/diagnostics/`.
- Case-agnostic FSI orchestration goes in `simulation_core/drivers/`.
- Fluent benchmark/parity runners should use these package paths and must not introduce case-specific solver logic under `simulation_core/`.

## FSI Execution Ownership

`simulation_core.drivers.generic_fsi_solver` is the single shared trial engine
for cases that implement its runtime-adapter protocol, including Turek-Hron.
For those callers it owns each committed physical step, rollback transaction,
and marker-velocity coupling trial. Component-local fluid CFL/RK substeps,
solid elastic-wave substeps, and HIBM/pressure algebraic iterations do not count
as extra committed physical steps.

The official ANSYS rectangular-solid benchmark deliberately uses the validated
direct sharp pipeline in
`benchmarks/official/solid_mpm_fsi_runner.run_hibm_mpm_fsi`. Its case wrapper
must delegate exactly once and may add only metadata/report validation. Squid
uses a typed `StepLoopContext` around its case-specific direct sharp fixed-point
assembly. These are different execution adapters around the same canonical
sharp HIBM-MPM formulation, not permission to restore a second
`legacy_projected_reduced` or cell-obstacle workflow for either case.

Shared snapshot and rollback state lives in
`coupling/hibm_mpm/interface_state.py`. Generic adapters must invalidate a
previous transaction before a new snapshot, arm rollback only after every
pre-mutation snapshot succeeds, and clear the transaction after commit or
rollback. The direct ANSYS path instead owns one accepted macro transaction per
FSI step. Within that transaction, accepted fluid and solid physical substeps
must each consume exactly `dt_s`; rejected CFL, positivity, Helmholtz, pressure,
or MPM trials contribute zero accepted time and restore the accepted state.
Pressure/PCG/Helmholtz and FSI coupling iterations may stop at residual
convergence because they are algebraic work at one physical time, but they must
not truncate either component's remaining physical time. The path retains its
own fail-closed pressure, ledger, no-slip, traction, MPM, and SST health gates.

Changing which adapter a validated case uses is a numerical behavior change.
It requires a fresh source-matched preflow snapshot and staged CUDA
FSI1/FSI8/FSI50 validation; host architecture tests alone are insufficient.

Legacy module names are not installed. New project code and external migration
guides should use the functional package path.
