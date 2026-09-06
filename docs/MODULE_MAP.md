# Simulation Core Module Map

## Latest result: focused gates passed; R16 stopped for publication

The strict-CUDA/f32 GREEN25 r03 passed all 25 tests in 81.096244 s on clean
source commit `efdb2527900ff455e2150fdef701cd15298be8fe`. The complete R12,
R13, R14 and R15 frozen replays also passed. Each replay's 21 snapshots of
155 fields and 41 cleanup fields were independently read back and recomputed.
These checks advance zero physical time.

The from-zero eight-step R16 diagnostic was stopped with SIGINT during Taichi
compilation at 1% quota remaining to complete the user's authorized GitHub
publication before exhaustion. Owner PID/PGID 410 exited -2 after 686.565025 s,
with zero accepted steps and `KeyboardInterrupt`. Source193, all 77 dependencies,
and host identities remained unchanged; wrapper audit errors are empty.
This is an administrative interruption, not a measured numerical failure.
The initial native `process.json` still says running; the final owner record
and execution audit establish the actual exit.

Continuous eight-step success, independent full rollback validation, and the
registered component/formal benchmark gates remain unproved for this source.
No reset credit was used. The old 10% threshold remains withdrawn.
The following older checkpoints are historical; this block is the current state.

Full evidence and exact continuation requirements: [R15 fallback report](validation/TUREK_HRON_R15_FALLBACK_VALIDATION_REPORT_2026-09-06.md).

Publication: [source and complete evidence](https://github.com/lizhuoh9/EasyFsi/releases/tag/r15-wip-20260906-1330).

## Historical explicit continuation checkpoint, 2026-09-06

The user resumed work after the archived quota stop, explicitly withdrew the
10% stop threshold, and authorized committing and pushing all current project
work to GitHub before quota exhaustion. The earlier pause below is historical.
No quota-reset credit has been used. The current core is still the reviewed
`26f37eb1` candidate; its 25-test native verification and continuous physical
validation remain incomplete at this source checkpoint.

The next focused run will use a fresh output label and a manifest that records
this commit's actual execution identity while preserving source and input
hashes. The old A50 baseline and all original captured identities remain intact.
The full-replay geometry-binding review correction is being prepared externally.

Publication destination: [R15 WIP source and complete diagnostic evidence](https://github.com/lizhuoh9/EasyFsi/releases/tag/r15-wip-20260906-1330).
The 141,016,737-byte archived snapshot predates this explicit continuation;
its SHA256 is `671728300c42b7e0a0d50a9169d39ab50854b66ef53d2f9d808ac2c2fa15b5c7`.
It preserves the stopped WIP and original evidence without claiming a native,
continuous, component or formal benchmark pass.

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

Generic sharp-HIBM band sweeps invalidate their canonical ledger even when the
returned cell increment is zero. `_stabilize_hibm_solid_band` uses the existing
assembler with `enable_marker_compatibility_closure=False` to rebuild geometric
claims and refresh the raw pressure-hard mask while topology changes. Only a
zero-increment sweep permits full marker closure and prepare/seal; a positive
last increment fails closed. Initial and air-conversion loops retain cap 8;
the post-solid path retains its first sweep plus eight more (cap 9), with cap 8
for the existing conditional disconnected path.

`_hibm_marker_compatibility_closure_pending` blocks full preparation, sealing,
public consumers and trial `save_state`. Invalidation preserves the pending
state; successful full row clearing or a complete trial restore releases it.
Intermediate geometry-only ledgers remain unsealed and carry no marker-health
qualification. Air conversion with zero added cells still rebuilds/closes before
its next reader. Positive overflow/tiny cleanup also reseals before its next
reader, and nested helpers publish that generation's report. No reader performs
lazy repair or relaxes the sealed guard. See the
[trace-space audit](validation/TUREK_HRON_TRACE_SPACE_AUDIT_2026-09-05.md) for the
bounded frozen-pose evidence and the outstanding coupled validation. The
common-face route repair at `a50b67f0` passes all 14 strict-CUDA contracts and
repaired R12/R13/R14 full-domain controls with complete artifact readback.
Fresh R15 exited 1 after five accepted steps to t = 0.025 s; step 6, assembly 99
reports four `prepare_pair_arbitration` conflicts, first face (0, 50, 342),
axis 2, path 0, claim_count 2. The complete 153-array precleanup capture and
207-field input manifest support diagnosis. The host terminal audit passes as
`FAILED_WITH_VALIDATED_ACCEPTED_PREFIX`, with fluid/solid time each 0.025 s
and fresh8 false; independent full physical rollback equality is unverified.
The original coarse S0 obstruction still needs a source-current zero-time
operator check at the R04 pose: 4x48x288, automatic112, solid100, nine post-solid
passes and external-boundary time 0.040 s, using rebuilt current topology.
Historical coefficient certificates and fine-grid control passes do not
qualify that coarse check or formal coupled acceptance.

## Canonical Component-Face Cohorts

`coupling/hibm_mpm/core.py` owns the interpolated segment-pair cache and
the actual source-consumption contract. `_precompute_canonical_component_face_common_trace_members_kernel`
proves original direct/shadow sources, materialized storage payloads, registered
unique finite owner and support against the cached trace. Seed and proved masks
are eligibility data. Native prepare tracks actual consumption locally and
publishes the consumed mask only for an admitted common cohort.

Prepare can select common mode256 only for an existing rejection whose actual
members differ from the cached seeds and whose entire consumed mask is proved.
Region/normal, count and identity checks remain mandatory. An unsuccessful
proof publishes the original pending health events; it never clears global
errors. Existing successful and inactive-axis routes retain their contracts.

`_reconstruct_canonical_component_face_common_trace` rechecks the exact mode,
membership and cache linkage, then consumes the existing cached B/N/Q and
shared canonical trace/target publication helpers. Every admitted common trace
requests the existing pair route at its certified physical face; it does not
switch back to a scalar primary face according to D/D or D/S seed kinds.
The shared route, alpha, geometry and finite-target checks remain mandatory.
Unique-owner indices and
the three masks are transaction state cleared by native commit and error
cleanup. Focused tests distinguish a valid unused geometry seed from actual
membership and check the eight canonical fields before any fixture reset.
The mixin is `tests/solvers/_hibm_common_trace_cohort_contracts.py`, integrated
by `tests/solvers/test_hibm_component_face_geometry.py`; its immutable geometry
fixture is `tests/solvers/fixtures/turek_hron_common_trace_cohorts.json`.
The additional immutable
`tests/solvers/fixtures/turek_hron_common_trace_r14_tilted.json` retains R14's
tilted D/S geometry at original face(0,50,305), axis2. The same mixin's
`test_systemic_r14_tilted_ds_common_cohort_reconstructs_affine_velocity`
checks actual native reconstruction against an affine known solution.
Its compact RED/GREEN result is separate from full-domain and coupled evidence.

## Candidate-pair fallback for the stopped R15 WIP

`_precompute_canonical_component_face_fallback_trace_geometry_kernel` in
`simulation_core/coupling/hibm_mpm/core.py` enumerates the available pairs among
up to four complete candidates when the old admission/full-valid cache is absent.
It keeps valid legacy caches, including C0, and requires one consistent unique
owner/B/Q with a direct-containing seed. Actual consumers still need registered
source and cached-trace membership proof; at least one actual direct member is
required. A fallback seed may equal the actual consumed mask. The original shape
rejection is forced only for fallback groups with at least two claims, preserving
zero/one-consumer behavior. Ambiguity and invalid membership retain rejection.

New temporary fields `velocity_dirichlet_component_face_common_trace_fallback_valid`
and `velocity_dirichlet_component_face_common_trace_fallback_prior_adjacent_direct`
are scalar i32 component-face fields. Both use zero initialization and native
commit/error cleanup; full replay inventory is155 arrays and41 cleanup fields.
The existing common-cohort mixin now has21 tests, with four legacy controls in the
external25-test gate, and uses `turek_hron_common_trace_r15_fallback.json`.

Current core26f37eb1 is applied WIP. Its25-test strict-CUDA gate was interrupted at
the user's10% quota boundary during first compilation, with zero completed tests.
Only the unchanged A50 consumer baseline2/2 has completed. Full R12-R15 replay
preparation is host-only and needs its independent query-binding review amendment.
See the existing trace-repair handoff for the exact source hashes, stopped process
and resume order. No continuous or formal benchmark pass is claimed.

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

## Turek-Hron Offline Validation Ownership

`src/refactored/validation/turek_hron_fsi/` owns the solver-free Turek-Hron
reference and assessment layer. `references.py` is the single immutable,
source-first catalog for canonical Featflow definitions, published table rows,
raw-series identities, and separately labelled LS-DYNA cross-checks.
`featflow.py` accepts only manifest fields and raw bytes that exactly match that
catalog. `limit_cycle.py` owns the preregistered rising-crossing, extrema,
stability, and FFT calculations without importing Taichi or solver code.

The stage layer is also solver-free: campaign_stages.py owns immutable stage
specifications, prerequisite order and cross-run research gates; acceptance.py
owns the shared typed physical-history and candidate-step health contract;
periodic_acceptance.py binds FSI2/3 policy, accepted interface records and the
existing limit-cycle analysis. The CLI in
tools/validation/run_turek_hron_fsi_campaign.py owns execution, provenance
replay, output claims and progress. A stage pass, exploratory pass and final
benchmark-quality pass are distinct statuses.

The upstream FSI2/FSI3 bytes and their exact manifests live under
`docs/validation/turek_hron_featflow/`; `.gitattributes` marks the `.point`
files non-text so line-ending conversion cannot invalidate their SHA256
identities. `cases/turek_hron_fsi.py` may expose a read-only compatibility
projection and a fresh JSON reporting copy, but must not own another mutable
reference table or silently fall back between sources.

The executable constituent gate is split by responsibility under
`tools/validation/`: `turek_hron_component_gate_contracts.py` owns the frozen
matrix, formulas, raw-evidence schemas, and runtime-identity validation;
`turek_hron_component_gate_runtimes.py` owns lazy construction of the three real
Taichi runtimes; and `run_turek_hron_component_gates.py` owns exclusive output
claims, artifact provenance, comparisons, and CLI exit status. Runtime instances
must be constructed only after the output claim. Comparison artifacts record
`not-applicable` for their own Taichi identity because comparison is solver-free;
their constituent parents retain complete measured CUDA identities.

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
