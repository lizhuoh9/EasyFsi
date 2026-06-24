# Architecture

## Runtime Layers

- `simulation_core/`: reusable solver package.
- `simulation_core/fluids/`: Cartesian fluid implementation.
- `simulation_core/coupling/`: FSI, IBM, and HIBM coupling APIs.
- `simulation_core/coupling/hibm_mpm/`: sharp HIBM-MPM implementation.
- `simulation_core/solids/`: solid MPM implementations.
- `simulation_core/geometry_tools/`: mesh, CAD, coordinate, and domain helpers.
- `simulation_core/materials/`: material cards and stress probes.
- `simulation_core/diagnostics/`: validation and time-step helpers.
- `cases/`: runnable case adapters.
- `benchmarks/`: official/vendor benchmark adapters.
- `tools/`: diagnostics, rendering, and post-processing scripts.
- `tests/`: tests grouped by responsibility.
- `docs/`: architecture, validation, and refactoring records.
- `archive/`: historical one-shot maintenance scripts.

## Dependency Direction

Allowed:

- `cases -> benchmarks -> simulation_core`
- `cases -> simulation_core`
- `benchmarks -> simulation_core`
- `tools -> cases/benchmarks/simulation_core`
- `tests -> anything`

Forbidden:

- `simulation_core -> cases`
- `simulation_core -> benchmarks`
- `simulation_core -> tools`
- `benchmarks -> cases`
- `cases -> tools`

## Legacy Compatibility

Top-level modules such as `simulation_core.fluid`, `simulation_core.hibm_mpm`,
`simulation_core.neo_hookean_mpm`, `simulation_core.mooney_shell_mpm`,
`simulation_core.geometry`, `simulation_core.hyperelastic`,
`simulation_core.validation`, and `simulation_core.time_stepping` remain
compatibility shims during migration.

New code should prefer package-backed imports from `simulation_core.fluids`,
`simulation_core.coupling`, `simulation_core.solids`,
`simulation_core.geometry_tools`, `simulation_core.materials`, and
`simulation_core.diagnostics`.

The root `simulation_core` package preserves the public API by importing
package-backed objects from the layered packages/facades.
