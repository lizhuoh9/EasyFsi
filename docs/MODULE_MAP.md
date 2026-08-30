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

## Physical Exterior Normal Flux

`fluids/solver.py` owns one `_physical_exterior_normal_contract` returning
`[prescribed, absolute normal velocity]`. Its velocity wrapper is shared by
projection divergence, primal-Q, MUSCL normal states, and SST reconstruction;
SST strain/transpose and normal Helmholtz rows also consume the prescribed flag.
Unregistered exterior normals are closed; a maximum-side
exterior face must not borrow the last internal backward-MAC velocity. Exact
normal targets remain authoritative, including zero targets and partial masks.
Interior HIBM transport Q remains wall-relative, not the absolute wall velocity
used by projection.

`predict` and `advance_sst_transport` receive `pressure_outlet_zmin=False` and
`velocity_inlet_zmax=None` explicitly. The pressure outlet alone may use its
colocated minimum compact row. The zmax modes retain their projection meaning:
None derives per-face ownership, True permits the legacy whole-plane fallback,
and False rejects any exact zmax normal on a fluid-adjacent face. Conflicting
no-slip/open declarations are rejected before physical writes. Topology is
resolved per call and passed through all SSP sources/retries; no active-mode
cache is added to persistent state.

An extrapolated zero at an explicit z port is still free, unlike an exact or
default closed zero. Minimum normal compact rows are synchronized before fluxes
and after SSP/implicit stages. Maximum exterior normals have no compact owner:
their ghost state and matrix boundary term must not overwrite the last internal
MAC row. Normal closure alone does not activate tangential no-slip or SST wall
correlation friction. All three SST reconstruction paths receive the same
per-call topology and their own current/previous stage source.

The official runner shares one config parser across SST, prediction and
projection. The generic `hibm_mpm/core.py` sharp-load assembly also passes its
existing outlet/inlet settings to both predictor and projection. Standalone
throughflow callers must explicitly declare the openings they require.

Generic sharp-HIBM band and air sweeps invalidate their canonical ledger even
when the returned cell increment is zero. Rebuild/prepare/seal before testing an
early exit or invoking a reachability reader. Positive overflow/tiny cleanup
also reseals before its next reader, and nested helpers publish the report from
that same generation. No reader performs lazy repair or relaxes the sealed guard.

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

## Registered Segment Geometry and Accepted-Step Restart

The non-interpolated registered 2-D/extruded route is decomposed inside
`coupling/hibm_mpm/`: `component_face_segment_geometry.py` owns finite F64
projection and integer topology validation; `component_face_segment_assembly.py`
owns full raw-route scratch and independent global-nearest-owner selection;
`component_face_segment_audit.py` certifies every raw source and connected path.
Raw sources retain the original strict support. Geometric connectors use the
strict face-global Euclidean disk circumscribing the active-plane source box
(scalar support retains its original disk). Owner, corner and every actual
connector share this bounded domain; a unique qualified registered subarc is
still required. This supersedes the projection hull that rejected a legal
curved connector at r36 step 49; it does not widen raw-source permission.
`core.py` dispatches these passes and writes the canonical ledger only after
all certificates pass. Legacy 3-D/interpolated routes are not this contract.

`component_face_candidate_geometry.py` checks global-nearest-owner permission
for each possible MAC destination before source-progress ranking. This is a
candidate prepass only: it neither creates/drops raw authors nor replaces the
final route audit. Failure still prevents the sole public-ledger commit.

## Fixed Material Surface and Adjoint Loads

- `coupling/hibm_mpm/material_surface_binding.py` constructs immutable Cartesian
  reference W, its source identity, and finite conditioning/mass-gain diagnostics.
  It checks unity/affine reproduction and bounded signed half-cell extrapolation
  with particle and marker input-quantization accounting.
- `coupling/hibm_mpm/material_surface_transfer.py` owns device Wx/Wv geometry,
  edge-oriented normals, pressure probes and deterministic CSR W.T loads. Actual
  rounded f32 particle-force increments are staged, audited and then committed.
- `hibm_mpm/core.py` binds that map, guards immutable topology, and composes cap
  motion/load derivatives. `interface_state.py` includes binding identity while
  keeping an IQN trial velocity independent of accepted-material Wv.
- `solids/neo_hookean_mpm.py` reports direct fixed force (N, final substep), support
  and damping impulse (N s, accepted batch), and their angular impulses (N m s).
  The `pure_fixed_mass` policy includes the discarded fixed PIC/APIC share at
  unclamped grid nodes without double-counting the grid clamp. Persistent `F` and
  `saved_F` are f64; `C`, `v`, grid velocity and the existing P2G/APIC layout remain
  f32. The step uses explicit f64 deformation recurrence/constitutive locals and
  preserves an in-range raw `F` without SVD reconstruction; SVD projection is only
  for an actual singular-value bound violation or reversed determinant. These
  diagnostics have rollback state; they are not a global PIC/FLIP momentum or
  whole-FSI energy proof.
- The ANSYS case and official runner select `cartesian_reference_adjoint_v1`.
  Physical markers have zero face offset; pressure probes carry the separate
  offset. Fixed reference surface/cap areas remain the current convention.

The new `src/refactored/validation/ansys_vertical_flap_fsi/material_reference_fine_contracts.py`
adds material/reaction evidence to the existing strict IQN fine50 contract.
The validation CLI must preserve these fields through its canonical JSON `_N`
force-key serialization. Historical Fluent profiles remain separate.

## Complete Accepted-Step Persistence

Persistent restart is separate from an in-memory trial rollback:

- `coupling/accepted_fsi_checkpoint.py` validates complete accepted macro state,
  controller/IQN state, physical time, binding identity, and incremental
  report/outbox records. Bound material geometry is checked before runtime writes;
  unbound legacy marker metadata remains a separate supported schema. Its solid
  deformation checkpoint field accepts only f64 `F`; a legacy f32 deformation is
  rejected before owner writes rather than implicitly cast, so accepted save/restore
  retains f64 low bits while other declared f32 fields remain f32.
- `diagnostics/checkpoint_codec.py` is the non-pickle JSON/numeric-array codec.
- `diagnostics/checkpoint_store.py` publishes immutable NPZ generations and
  checksummed journals with a manifest-last single-writer transaction.
- `diagnostics/atomic_file.py` bounds retries of one already-prepared Windows
  publication when sharing conflicts temporarily deny rename. Mutable metadata
  uses atomic replacement; immutable accepted artifacts use create-only rename
  and still never overwrite an existing destination. Neither path regenerates
  payloads, retries solver work or advances physical time.
- `diagnostics/run_attempt.py` preserves old `failure.json` / `interruption.json`
  under a unique checksummed `attempts/` ledger after successful resume preflight.
  It rejects non-regular terminal entries, including Linux symlinks exposed as
  Windows/WSL reparse points. Completion requires both completed records and no
  active failure/interruption. Its single-writer rename contract preserves bytes
  across ordinary process errors; it does not certify power-loss durability.
  `validate_dual_root_attempt_provenance` also binds a completed attempt-v2 root
  to a distinct canonical artifact root, exact target step, checkpoint
  generation/identity, source hashes and matching manifest/summary provenance.
  This is a control-plane contract only. The comparison consumer separately
  validates the canonical checkpoint journal and exact field/history prefix,
  then uses `validate_dual_root_history_row_semantics` to bind every public
  journal field to the per-step JSON and aggregate CSV. CSV boolean spelling and
  empty values are normalized only at that serialization boundary; unexpected
  aliases, JSON type drift, nonfinite values and non-core field mismatches fail
  closed.
  Therefore a canonical head at K200 may supply a locked K1--K50 artifact prefix
  to a completed K50 attempt without treating the canonical historical summary
  as the current accepted-state pointer or silently merging the two roots.
- The official runner restores validated state, rebuilds derived caches, then
  continues from accepted step K. The validation CLI owns output-prefix/outbox
  checks. It checks the current production source identity before archiving or
  publishing a new running state, and reuses the same verified checkpoint head
  after checking the loaded generation and accepted step. Reduced
  `step_fields/*.npz` files are never restart checkpoints. For the exact
  `FsiCouplingConvergenceError` only, its failure artifact also preserves the
  complete context/report and raw IQN guess/candidate/residual histories; legacy
  pressure diagnostics retain their existing meaning. Failure reporting preserves
  the already accepted progress index/time, does not label non-FSI errors as FSI,
  and a new `running` event clears stale FSI failure diagnostics.

See the [continuous-execution design and measured validation boundary](refactoring/ANSYS_VERTICAL_FLAP_CONTINUOUS_EXECUTION_DESIGN_2026-08-28.md)
before changing these contracts or starting a long numerical run.
