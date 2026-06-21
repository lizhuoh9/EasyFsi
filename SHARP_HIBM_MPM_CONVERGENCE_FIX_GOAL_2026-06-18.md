# Sharp HIBM-MPM Convergence Fix Goal

## Source And Scope

This goal is derived from the static code audit of commit
`361617fbb5db020113af65a23ce9f82a07ef1974`.

The audit was not a completed CUDA/Taichi long-run validation. Treat it as an
implementation roadmap for confirmed code hazards and high-probability
structural causes of non-convergence:

- solid partial out-of-bounds particles continuing after leaving the MPM grid;
- stale or incorrectly preserved solid external forces across FSI trials;
- empty shell regions hidden by `max(count, 1)`;
- HIBM obstacle/topology cleanup happening inside `fluid.project()` after HIBM
  rows and pressure matrix terms may already be assembled;
- pressure-outlet reachability and source-flux diagnostics missing validity and
  cancellation-resistant ratios;
- FSI fixed-point rejection/trust-region paths committing unsafe state,
  especially zero force when all trials are rejected;
- non-reusable accepted FSI trials needing explicit re-advance;
- legacy projected/reduced mobility-ratio dt semantics being too aggressive if
  those optional modes are enabled.

The worktree is:

`D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`

Do not make claims about physical convergence, 2s readiness, or paper
equivalence from this goal alone. Every behavior change must be backed by a red
test first, a focused green implementation, and short-run evidence before any
long-run claim.

## Objective

Repair the sharp HIBM-MPM squid route so it fails fast on inconsistent physical
state, preserves fluid/solid/topology contracts across each FSI trial, and
produces trustworthy short-run evidence before any long simulation is attempted.

The first success target is not "make the long run green by damping". The first
success target is:

1. no silent solid MPM particle loss;
2. no stale solid external-force reuse across trials or physical steps;
3. no HIBM pressure projection with rows/matrix assembled against an obsolete
   obstacle topology;
4. no all-rejected FSI step committing zero interface force;
5. no accepted FSI result whose reported row diverges from the committed
   solid/fluid state;
6. pressure outlet diagnostics that distinguish valid reachability from stale
   labels and net-source cancellation;
7. short 8/32/128 step artifacts that expose real failures instead of hiding
   them behind solver-green bookkeeping.

## Hard Constraints

- Stay on the sharp HIBM-MPM / paper-route repair path. Do not replace the core
  issue with scalar damping, denominator tuning, or a diagnostic-only mask.
- Preserve existing user changes. Start with `git status --short` and do not
  revert unrelated dirty files.
- Use the reliable validation interpreter on this machine:
  `D:\working\taichi\env\python.exe`.
- Do not treat `pressure_projection_cg_converged_all=True`,
  low CG residual, or zero interior divergence as sufficient physical success.
  Also require bounded CFL, finite pressure/velocity, nonzero physically
  consistent membrane force, valid HIBM markers/stress rows, and honest FSI
  residual units.
- Do not claim full waveform readiness from 1-step or 2-step probes. After
  geometry/topology/FSI contract changes, require at least 8 and 32 step probes
  before considering 128 step probing.
- Keep explicit-step validation separate from full waveform readiness.
- Do not move topology cleanup inside pressure projection as a shortcut unless
  it fail-fast blocks stale HIBM rows/matrix and is documented as temporary.

## Primary Code Surfaces

- Solid membrane MPM:
  - `simulation_core/mooney_shell_mpm.py`
    - `TriMooneyShellMpmState`
    - `advance_with_external_forces()`
    - `report()`
  - `simulation_core/neo_hookean_mpm.py`
    - `NeoHookeanMpmState`
    - `report()`
- Fluid projection and HIBM pressure handling:
  - `simulation_core/fluid.py`
    - `CartesianFluidSolver.project()`
    - `pressure_outlet_fv_flux_report()`
    - `fill_hibm_converted_cell_pressures()`
    - HIBM reachability fields and converted-cell cleanup kernels
  - `simulation_core/hibm_mpm.py`
    - HIBM search/classify/row assembly and pressure-Neumann assembly paths
- FSI fixed-point coupling:
  - `simulation_core/fsi_coupling.py`
    - `InterfaceReactionFixedPointResult`
    - `solve_interface_reaction_fixed_point()`
    - `solve_and_apply_interface_reaction_step()`
- Squid case glue and diagnostics:
  - `cases/squid_soft_robot.py`
    - HIBM sharp driver flow
    - accepted-state reuse/re-advance logic
    - pressure outlet report consumption
    - `solid_response_constraint_force_mobility_ratio()`
    - summary/history rows
- Focused tests:
  - `tests/test_fsi_coupling.py`
  - `tests/test_core_fluid.py`
  - `tests/test_mooney_shell_mpm.py`
  - `tests/test_neo_hookean_mpm.py`
  - `tests/test_squid_latest_core_config.py`
  - add narrow new tests where no focused file currently exists.

## Phase 0 - Freeze A Short Reproducible Evidence Matrix

Before changing behavior, define the minimal evidence matrix to run after each
phase. Do not start with long blind runs.

Required per-step fields to preserve or add to `history.csv` / `summary.json`:

```text
solid_mpm_grid_out_of_bounds_particle_count
solid_mpm_max_speed_mps
solid_primary_particle_count
solid_secondary_particle_count
solid_primary_out_of_bounds_count
solid_secondary_out_of_bounds_count
pressure_projection_cg_converged_all
pressure_projection_cg_breakdown_count
pressure_projection_physical_failure
pressure_projection_physical_failure_reason
hibm_unreached_incompatible_component_count
hibm_unreached_component_rhs_mean_max_abs
hibm_projection_overflow_singleton_cleanup_cell_count
hibm_projection_tiny_unreached_cleanup_cell_count
pressure_outlet_reachable_source_volume_flux_m3s
pressure_outlet_unreached_source_volume_flux_m3s
pressure_outlet_positive_source_volume_flux_m3s
pressure_outlet_abs_source_volume_flux_m3s
pressure_outlet_reachability_valid
pressure_outlet_reachability_revision
fsi_coupling_convergence_measured
fsi_coupling_converged
fsi_coupling_residual_units
fsi_coupling_residual_norm_mps
fsi_coupling_residual_norm_n
accepted_fsi_trial_index
accepted_fsi_trial_state_reusable
accepted_fsi_trial_state_readvanced
accepted_fsi_trial_state_reused
fsi_all_trials_rejected
fsi_zero_force_commit_blocked
```

If a field is renamed to match existing naming conventions, document the mapping
inside the implementation notes and summary.

## Phase 1 - P0 Solid State Safety

Goal: make damaged solid state impossible to silently carry into the next HIBM
boundary state.

Tasks:

1. In `TriMooneyShellMpmState.report()`, make partial
   `grid_out_of_bounds_particle_count > tolerance` fail by default.
   Default tolerance should be zero unless a caller explicitly opts into a
   diagnostic tolerance.
2. Apply equivalent fail-fast behavior in `NeoHookeanMpmState.report()`.
3. Stop using `max(count, 1)` to hide empty primary/secondary shell regions.
   If `primary_count == 0` or `secondary_count == 0`, raise an explicit error.
4. Record primary/secondary particle counts and out-of-bounds counts in reports
   and case rows.
5. Add a preflight check that solid MPM background-grid padding covers at least:

   ```text
   solid displacement max + 3 * fluid max cell width + estimated marker support radius
   ```

6. Keep the existing case-layer out-of-bounds guard, but treat it as a second
   guard, not the first place where solid failure is detected.

Required red-to-green tests:

```text
test_tri_mooney_partial_out_of_bounds_raises
test_neo_hookean_partial_out_of_bounds_raises
test_shell_region_counts_must_be_nonzero
test_solid_grid_padding_preflight_rejects_undersized_grid
```

## Phase 2 - P0 HIBM Topology Cleanup Before Projection

Goal: ensure `CartesianFluidSolver.project()` solves a pressure system whose
obstacle mask, velocity Dirichlet rows, pressure Neumann rows, and pressure
matrix terms all describe the same topology.

Current risk:

`project()` can call HIBM pressure-outlet reachability and orphan/tiny-component
conversion after velocity boundary conditions and after policy/matrix reports
have been read. Those conversions can change `obstacle`, clear velocity/source
state, and invalidate preassembled HIBM rows.

Target flow:

```text
HIBM search/classify
assemble preliminary velocity rows if needed for reachability
mark outlet reachability
convert solid-band / orphan / singleton / tiny unreachable / air-backed cells
if topology changed:
  clear HIBM velocity Dirichlet rows
  clear HIBM pressure Neumann rows and pressure matrix terms
  re-search or revalidate anchors affected by the changed topology
  assemble final velocity Dirichlet rows
  assemble final pressure Neumann rows and pressure matrix terms
project()
fill converted-cell pressures before stress sampling
sample stress / traction
```

Implementation direction:

1. Move topology cleanup out of `CartesianFluidSolver.project()` into the sharp
   HIBM driver before final row/matrix assembly.
2. If a temporary compatibility path leaves cleanup inside `project()`, set
   `topology_mutated=True`. If topology mutated while pressure-interface matrix
   rows were active, raise immediately with a message requiring caller-side
   cleanup and reassembly.
3. Add explicit stale-state guards:
   - reachability labels must match the current obstacle/source revision;
   - pressure matrix terms must be assembled after the final topology revision;
   - velocity Dirichlet rows and pressure Neumann rows must not survive topology
     mutation unless they were rebuilt.
4. Preserve the intent of `fill_hibm_converted_cell_pressures()`: it must run
   after projection and before stress sampling whenever converted cells can be
   sampled by HIBM stress closure.
5. Write converted/fill counts into the same step row that sampled stress.

Required red-to-green tests:

```text
test_hibm_topology_cleanup_happens_before_pressure_matrix_assembly
test_project_does_not_mutate_obstacle_topology
test_project_raises_if_topology_mutates_with_active_pressure_matrix
test_pressure_matrix_report_recomputed_after_hibm_cleanup
test_converted_cell_pressure_fill_runs_before_hibm_stress_sampling
```

## Phase 3 - P0 FSI Rejection Must Not Commit Zero Force

Goal: all-rejected FSI trials must fail fast or commit a documented previous
force, never silently submit all-zero interface force by default.

Tasks:

1. In `solve_interface_reaction_fixed_point()`, when every evaluated trial is
   rejected and `rejected_trial_backtrack < 1.0`, default to raising
   `RuntimeError("all FSI trials rejected; refusing to commit zero interface force")`.
2. If a non-raising compatibility option is needed, make it explicit and commit
   `initial_force_n`, not zero force. Mark `converged=False`.
3. Add result/report fields:
   - `all_trials_rejected`;
   - `zero_force_commit_blocked`;
   - `fallback_force_source`;
   - `accepted_trial_index`.
4. Ensure case rows do not mark FSI coupling as completed-success when all
   trials were rejected.

Required red-to-green tests:

```text
test_all_rejected_trials_raise_instead_of_zero_force
test_all_rejected_trials_can_explicitly_fallback_to_initial_force
test_all_rejected_trials_are_not_reported_as_converged
```

## Phase 4 - P1 Accepted-State Replay And Trust-Region Semantics

Goal: accepted FSI force, committed state, and diagnostic rows must refer to the
same physical trial.

Tasks:

1. Treat `accepted_state_reusable=True` only when the accepted trial is the final
   evaluated trial and accepted force was not modified by passivity or fallback.
2. If `accepted_state_reusable=False`, require the caller to restore the saved
   pre-trial state, re-advance with `accepted_force_n`, and write
   `accepted_state_readvanced=True`.
3. Assert that the accepted report corresponds to `accepted_trial_index` and
   `accepted_force_n`; do not only apply force onto a later rejected state.
4. Change adaptive trust-region logic so accepted trials can trigger grow, while
   rejected trials can only trigger shrink or no update. Do not compare grow
   against the last evaluated rejected residual.
5. Track previous accepted residual or best accepted residual separately from
   previous evaluated residual.

Required red-to-green tests:

```text
test_nonlast_best_trial_is_readvanced_before_commit
test_accepted_state_reusable_false_requires_readvance
test_rejected_trial_does_not_grow_trust_region
test_trust_region_growth_uses_previous_accepted_residual
```

## Phase 5 - P1 Solid External-Force Contract

Goal: `advance_with_external_forces()` may preserve external force during solid
substeps, but no physical step or trial replay may inherit stale external force.

Required step order:

```text
clear solid.external_force_n
scatter marker forces -> solid.external_force_n
validate external_force_n finite and action-reaction consistent
solid substeps using preserve_existing_external_force=True
```

Tasks:

1. Add a step-local or trial-local marker indicating that external force was
   cleared and repopulated before `advance_with_external_forces()`.
2. Add optional assertions for:
   - finite external force;
   - external-force sum matching marker scatter report within tolerance;
   - no stale force after `restore_state()`.
3. Extend `save_state()` / `restore_state()` contract tests to lock down whether
   restored external force is zero or restored trial force. For trial replay,
   prefer clearing and forcing a fresh scatter.

Required red-to-green tests:

```text
test_advance_with_external_forces_requires_fresh_scatter
test_external_force_sum_matches_marker_scatter_report
test_restore_state_clears_or_restores_external_force_by_contract
test_replayed_trial_rescatters_external_force_before_solid_advance
```

## Phase 6 - P1 Pressure Outlet Diagnostics And Source Ratios

Goal: pressure outlet diagnostics must not hide stale reachability or source
cancellation.

Tasks:

1. Add reachability validity metadata:
   - `last_hibm_reachability_valid`;
   - `last_hibm_reachability_revision`;
   - optional `last_hibm_reachability_step`.
2. Whenever kernels mutate `obstacle`, velocity Dirichlet activity, or
   `volume_source_s`, mark reachability invalid.
3. `pressure_outlet_fv_flux_report()` must return:

   ```python
   "zmin_reachability_valid": bool(...)
   "zmin_reachability_revision": int(...)
   ```

4. Split outlet/source ratios into:

   ```text
   velocity_outlet_to_net_source_ratio
   velocity_outlet_to_positive_source_ratio
   velocity_outlet_to_abs_source_ratio
   ```

5. Validation gates should prefer positive or physically direction-consistent
   source ratio. Net-source ratio is diagnostic only when positive and negative
   sources cancel.
6. Only report unreached source centroid/bounds as valid when reachability is
   valid for the current topology revision.

Required red-to-green tests:

```text
test_pressure_outlet_report_marks_invalid_reachability
test_pressure_outlet_reachability_revision_changes_after_remark
test_pressure_outlet_ratio_uses_positive_source_without_cancellation
test_pressure_outlet_abs_source_ratio_tracks_cancelling_sources
test_unreached_source_centroid_reports_only_when_valid
```

## Phase 7 - P2 Stabilization Presets And Legacy Mobility Cleanup

Goal: make stabilization behavior auditable and prevent optional legacy
mobility features from using an overstrong dt scale.

Tasks:

1. Add:

   ```text
   --fsi-stabilization-preset off
   --fsi-stabilization-preset conservative
   --fsi-stabilization-preset aggressive
   ```

2. Keep `off` equivalent to current explicit defaults.
3. Define conservative/aggressive presets in code, then write the fully expanded
   effective parameters into preflight and summary.
4. User-provided explicit parameters should either override the preset with
   clear precedence or be rejected when ambiguous. Pick one policy and test it.
5. In legacy projected/reduced mobility paths, change the ratio dt to the actual
   solid response span:

   ```python
   solid_response_dt_s = spec.dt_s
   ```

   Alternatively rename the helper to make substep semantics explicit and pass
   true substep solid response, not full-step report divided by substep dt.

Required red-to-green tests:

```text
test_stabilization_presets_expand_to_expected_parameters
test_stabilization_preset_writes_effective_parameters_to_summary
test_stabilization_preset_conflict_policy_is_enforced
test_legacy_mobility_ratio_uses_solid_response_dt
```

## Validation Commands

Use focused tests after each phase:

```powershell
& 'D:\working\taichi\env\python.exe' -m unittest tests.test_fsi_coupling tests.test_core_fluid tests.test_mooney_shell_mpm tests.test_neo_hookean_mpm tests.test_squid_latest_core_config -v
```

Run short squid probes only after the focused unit/contract tests are green:

```powershell
& 'D:\working\taichi\env\python.exe' run_simulation.py squid-soft-robot --steps 8 --pressure-solve-failure-policy raise
& 'D:\working\taichi\env\python.exe' run_simulation.py squid-soft-robot --steps 32 --pressure-solve-failure-policy raise
```

Only after 8 and 32 steps produce honest, physically meaningful rows:

```powershell
& 'D:\working\taichi\env\python.exe' run_simulation.py squid-soft-robot --steps 128 --adaptive-fluid-substeps --pressure-solve-failure-policy raise
```

Before Phase 3/4 are complete, do not enable complex combinations such as:

```text
--fsi-coupling-trust-region-adaptive
--fsi-coupling-residual-continuation-*
--fsi-coupling-same-step-rerun-*
```

unless a specific test is proving their contract. Otherwise non-convergence can
become a superposition of multiple unfinished gates.

## Done Criteria

This goal is complete only when all of the following are true:

1. P0 tests for solid out-of-bounds, HIBM topology/projection consistency, and
   all-rejected FSI behavior fail before the fix and pass after the fix.
2. `fluid.project()` no longer silently mutates HIBM obstacle topology after
   rows/matrix are assembled, or it fail-fast blocks that state with a clear
   runtime error.
3. All-rejected FSI trials cannot commit zero force by default.
4. Non-reusable accepted FSI trials are re-advanced from the saved pre-trial
   state, and rows record whether reuse or re-advance happened.
5. Solid reports fail on partial out-of-bounds and empty primary/secondary
   shell regions instead of returning masked averages.
6. Pressure outlet reports expose reachability validity/revision and net,
   positive, and absolute source ratios.
7. The focused unittest command above passes.
8. At least 8-step and 32-step squid probes complete or fail for an explicitly
   physical reason, with `history.csv` and `summary.json` fields showing the
   relevant gate.
9. Any 128-step result is described strictly as a short-run regression probe,
   not as 2s/full waveform readiness.
10. A final implementation note records exact commands, artifact paths, and the
    first failing test / first passing test trail for each behavior change.
