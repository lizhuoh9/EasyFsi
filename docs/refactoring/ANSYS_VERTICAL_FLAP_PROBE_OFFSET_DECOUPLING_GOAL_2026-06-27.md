# ANSYS Vertical Flap Probe Offset Decoupling Goal - 2026-06-27

## Source Context

- Repository: `lizhuoh9/EasyFsi`
- Branch at goal creation: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Prior checkpoint: `162403bb3cca8f4b00df05f0213d542a1c559fdd`
- Prior checkpoint status: shared-snapshot traction resampling contract hardening is complete.
- Prior shared snapshot SHA-256:
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- Prior shared snapshot source commit:
  `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`
- Prior resampling result: 5 completed formulations, 1 unsupported formulation.
- Current blocker: dual/two-sided rows remain strongly offset-sensitive even
  when every completed row samples the same archived flow snapshot.

## Objective

Implement the next ANSYS vertical-flap traction diagnostic stage:
**probe offset decoupling**.

The previous evidence chain proved that formulation rows can be resampled on one
shared velocity/pressure/obstacle snapshot, but it also showed that changing the
marker face offset changes force ratios substantially. This goal separates two
currently coupled concepts:

- force-integration marker position,
- pressure-probe ladder origin/start position.

The completed work must make pressure-probe origin independently configurable
while preserving existing default behavior. It must then add a shared-snapshot
diagnostic runner that can sweep probe origin while holding force marker geometry
fixed, and sweep marker geometry while holding probe origin fixed. The output is
diagnostic evidence about the source of offset sensitivity, not a reference
formulation selection.

## Required Scope

This task has three implementation tracks:

1. Move the prior root-level traction resampling goal into `docs/refactoring/`
   and remove local absolute working-directory wording from that goal file.
2. Add pressure-probe origin support to HIBM traction markers in a backward
   compatible way.
3. Add the ANSYS vertical-flap shared-snapshot probe-offset decoupling runner,
   artifacts, and tests.

## Explicit Non-Goals

- Do not implement dual-face one-sided pressure support.
- Do not select a reference formulation.
- Do not change pressure formulas.
- Do not change force aggregation formulas.
- Do not change fluid or solid physics.
- Do not change material parameters, grid dimensions, source schedules, support
  radii, damping, or ANSYS case constants.
- Do not run coupled 50-step FSI.
- Do not claim Fluent parity.
- Do not overwrite current `traction_snapshot_resampling_diagnostics` artifacts.
- Do not hide the unsupported one-sided dual-face scenario.
- Do not introduce a shortcut that treats dual-face one-sided pressure as a
  single-mid two-sided surrogate.

## Phase 0 - Repository Documentation Hygiene

Move:

```text
ANSYS_VERTICAL_FLAP_TRACTION_SNAPSHOT_RESAMPLING_CONTRACT_GOAL_2026-06-27.md
```

to:

```text
docs/refactoring/ANSYS_VERTICAL_FLAP_TRACTION_SNAPSHOT_RESAMPLING_CONTRACT_GOAL_2026-06-27.md
```

Then edit that moved file so any local absolute working directory is replaced by
a repo-relative description. This is documentation hygiene only; do not change
the already committed resampling runner or artifacts as part of this move.

## Phase 1 - Core Probe-Origin Data Model

Add per-marker pressure-probe origin storage to the HIBM marker system. The new
state should represent the physical point from which inside/outside pressure
probe ladder positions are offset.

Required behavior:

- Existing marker force position remains unchanged.
- Existing marker normal, area, region id, and one-sided/two-sided policy remain
  unchanged.
- Default pressure-probe origin equals the force marker position.
- Existing code paths that do not pass explicit probe origins remain contract
  compatible.
- Pressure-probe origin is available in marker diagnostics so artifacts can
  prove marker position and probe origin were decoupled.

Recommended implementation shape:

```python
pressure_probe_origin_m: per-marker 3D vector field
```

Default assignment:

```python
pressure_probe_origin_m[marker] = x_gamma_m[marker]
```

Recommended public interface:

```python
def set_pressure_probe_origins_m(self, origins_m: Sequence[Sequence[float]]) -> None
```

or an optional marker-loading parameter:

```python
pressure_probe_origins_m: Sequence[Sequence[float]] | None = None
```

Validation requirements:

- wrong marker count fails fast,
- wrong shape fails fast,
- non-finite values fail fast,
- default behavior does not require callers to know this new field exists.

## Phase 2 - Sampling Kernel Decoupling

Keep force integration anchored at the existing marker position:

```python
position = x_gamma_m[marker]
```

Use the new probe origin only for pressure probe ladder positions:

```python
probe_origin = pressure_probe_origin_m[marker]
outside_position = probe_origin + normal * probe_distance
inside_position = probe_origin - normal * probe_distance
```

The key invariant is:

```text
force marker position may differ from pressure probe origin
```

The kernel must still preserve the current inside/outside semantics, pressure
jump sign convention, one-sided fail-closed behavior, and traction decomposition
diagnostics.

## Phase 3 - Core Tests

Add solver-level tests, preferably:

```text
tests/solvers/test_hibm_traction_probe_origin_decoupling.py
```

Required test cases:

1. Default behavior remains equivalent:
   - load markers without explicit pressure-probe origins,
   - sample two-sided pressure,
   - assert diagnostics still use marker-position origins,
   - assert force and traction outputs match the old marker-origin behavior.

2. Probe origin can change pressure sampling without moving the force marker:
   - use the same marker position, normal, area, and pressure field,
   - use a different pressure-probe origin,
   - assert marker position diagnostics are unchanged,
   - assert probe origin diagnostics changed,
   - assert inside/outside probe grid coordinates or nearest cells changed,
   - assert pressure traction changes because the probe origin changed.

3. Invalid origins fail fast:
   - wrong marker count raises `ValueError`,
   - wrong coordinate shape raises `ValueError`,
   - non-finite coordinate raises `ValueError`.

4. Diagnostics expose decoupling evidence:
   - marker diagnostics include `pressure_probe_origin_m`,
   - marker diagnostics include `pressure_probe_origin_source` or equivalent,
   - diagnostics make it possible to compare marker position vs probe origin.

## Phase 4 - ANSYS Case/Runner Configuration

Add explicit controls to the ANSYS vertical-flap configuration or marker builder
path. Recommended fields:

```python
traction_pressure_probe_origin_offset_cells: float | None = None
traction_pressure_probe_origin_mode: str = "marker_position"
```

Supported modes:

```text
marker_position
physical_face_offset
```

Rules:

- Default `marker_position` keeps current behavior.
- The existing `traction_marker_face_offset_cells` remains the force marker
  position control.
- `physical_face_offset` computes probe origins from the physical flap face plus
  an explicit probe-origin offset along the marker normal.
- Only the new decoupling runner should opt into `physical_face_offset` during
  this task.
- Existing validation and smoke runners should retain old defaults.

## Phase 5 - Shared-Snapshot Probe-Offset Decoupling Runner

Add a new runner:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_offset_decoupling_matrix.py
```

Output directory:

```text
validation_runs/ansys_vertical_flap_fsi/traction_probe_offset_decoupling_diagnostics/
```

The runner must:

- load the same archived shared snapshot manifest and NPZ used by traction
  snapshot resampling,
- verify the shared snapshot SHA-256,
- restore velocity/pressure/obstacle/grid fields into the fluid solver,
- not advance the fluid,
- not advance the structure,
- not run a coupled FSI loop,
- not claim Fluent parity,
- write JSON, CSV, history, summary, checksums, and marker diagnostics,
- record repo-relative `source_script`,
- record top-level and row-level sampling-only scope,
- record candidate status as diagnostic-only,
- preserve the current unsupported/fail-closed one-sided boundary.

Minimum scenario matrix:

```text
fixed_marker0p51_probe0p00
fixed_marker0p51_probe0p25
fixed_marker0p51_probe0p51
fixed_marker0p51_probe1p00
fixed_probe0p51_marker0p00
fixed_probe0p51_marker0p25
fixed_probe0p51_marker0p51
fixed_probe0p51_marker1p00
```

These two groups answer different questions:

- fixed marker, swept probe origin: isolate pressure-probe ladder sensitivity,
- fixed probe origin, swept marker position: isolate force-marker geometry
  sensitivity.

The runner must keep the prior shared snapshot SHA:

```text
3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968
```

## Phase 6 - Decoupling Artifact Contract

The new matrix payload should include:

- `schema_version`,
- `case`,
- `purpose`,
- `scope_limit`,
- `source_script`,
- `flow_snapshot_sha256`,
- `flow_snapshot_source_commit`,
- `candidate_status`,
- `reference_formulation_candidate: null`,
- `candidate_blockers`,
- completed row count,
- scenario list,
- row-level marker offset,
- row-level probe-origin offset,
- row-level marker geometry hash,
- row-level probe origin hash,
- force ratio to baseline,
- group-level ratio spans for fixed-marker and fixed-probe sweeps,
- marker diagnostics path.

Suggested candidate status:

```text
probe_offset_decoupling_diagnostic_only
```

Suggested blockers:

```text
reference_selection_deferred
dual_face_one_sided_unsupported
probe_offset_decoupling_diagnostic_only
sampling_only_no_coupled_fsi
```

If the evidence shows sensitivity remains high, record it honestly. Do not turn
the diagnostic into a pass/fail ratio target unless the data justifies it.

## Phase 7 - Artifact Tests

Add:

```text
tests/integration/test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts.py
```

Required checks:

- matrix JSON exists,
- matrix CSV exists,
- history JSON exists,
- summary markdown exists,
- checksum file exists,
- marker diagnostics exist for completed rows,
- all completed rows share the same `flow_snapshot_sha256`,
- `source_script` is repo-relative,
- top-level and row-level scope are shared-snapshot sampling-only,
- marker position fields are present,
- pressure-probe origin fields are present,
- fixed-marker probe-origin sweep keeps marker geometry hash constant,
- fixed-probe marker sweep keeps probe-origin hash constant,
- probe offset ratios are reported,
- candidate status is diagnostic-only,
- `reference_formulation_candidate` is null,
- summary contains no Fluent parity claim,
- summary contains no coupled 50-step FSI claim,
- checksums match committed artifacts.

## Phase 8 - Verification Commands

Use the repository's trusted Python environment:

```powershell
python -m py_compile simulation_core/coupling/hibm_mpm/core.py simulation_core/hibm_mpm.py cases/ansys_vertical_flap_fsi.py benchmarks/official/solid_mpm_fsi_runner.py validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_offset_decoupling_matrix.py tests/solvers/test_hibm_traction_probe_origin_decoupling.py tests/integration/test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts.py
```

Run solver-level tests:

```powershell
python -m unittest tests.solvers.test_hibm_traction_probe_origin_decoupling tests.solvers.test_hibm_traction_probe_diagnostics -v
```

Regenerate the new diagnostic artifacts:

```powershell
python validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_offset_decoupling_matrix.py
```

Run artifact tests:

```powershell
python -m unittest tests.integration.test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts tests.integration.test_ansys_vertical_flap_traction_snapshot_resampling_artifacts tests.integration.test_ansys_vertical_flap_traction_shared_snapshot_artifacts tests.integration.test_ansys_vertical_flap_traction_probe_observability_artifacts -v
```

Run whitespace verification:

```powershell
git diff --check
```

## Acceptance Criteria

- Prior root-level resampling goal is moved into `docs/refactoring/`.
- New probe offset decoupling goal is committed in `docs/refactoring/`.
- Existing default HIBM marker pressure sampling behavior remains compatible.
- Pressure probe origin can be configured independently of marker force
  position.
- New diagnostics expose marker position and pressure-probe origin separately.
- New shared-snapshot decoupling runner writes a complete artifact set.
- Fixed-marker and fixed-probe sweep groups are both represented.
- All completed decoupling rows share the same archived snapshot SHA.
- No current resampling artifacts are overwritten by the new runner.
- Tests cover core decoupling and artifact contracts.
- Validation commands above pass, or any GPU/CUDA blocker is reported with the
  exact command and error.
- The final branch is committed and pushed to `origin`.

## Commit And Push Requirements

The user approved push after the modification is complete. Do not push before:

1. this detailed goal file exists,
2. the active short goal references this file,
3. implementation and artifacts are complete,
4. required tests and `git diff --check` pass,
5. the staged diff is reviewed for unrelated changes.

Recommended commit sequence:

```text
docs: add ANSYS traction probe offset decoupling goal
feat: add pressure probe origin support for HIBM markers
test: cover HIBM traction probe origin decoupling
validation: add ANSYS traction probe offset decoupling matrix
test: add ANSYS traction probe offset decoupling artifacts
```

It is acceptable to combine the work into fewer commits if the final diff stays
coherent and verification is complete. The final report must include:

- branch name,
- final commit hash or hashes,
- push target,
- validation commands and results,
- artifact directory,
- whether the evidence reduced or preserved offset sensitivity,
- explicit statement that the patch remains sampling-only and makes no Fluent
  parity or coupled FSI validation claim.
