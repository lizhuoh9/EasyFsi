# FSI Full Audit Repair Goal - 2026-07-02

## Objective

Repair all confirmed findings from the July 2026 FSI full-scope audit in the
current refactored HIBM-MPM checkout. The scope includes correctness bugs,
default-path performance waste, dead code, dead diagnostics, stale compatibility
surfaces, and tests that currently lock incorrect behavior.

This is an execution goal, not a validation claim. A finding is complete only
after the current code evidence is checked, a focused regression/static test is
added or updated where practical, the fix lands in the core module or proper
owner boundary, and targeted verification passes.

## Source Inputs

- User-provided audit synthesis summary, plus the full local report at
  `D:\User_Temp\lizhu\claude\D--working-squid-robot-simulation-src-reference-papers-HIBM-MPM-refactored\d88c2d82-89b1-4130-b3bb-9e135d549da4\scratchpad\audit_synthesis.md`.
- Current checkout:
  `D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`.
- Priority repair goal already started in
  `FSI_AUDIT_PRIORITY_REPAIR_GOAL_2026-07-02.md`.

## Non-Negotiable Constraints

- Keep solver behavior in `simulation_core`; do not hide numerical fixes in
  runner/case code.
- Preserve unrelated dirty-worktree edits. Do not revert or clean pre-existing
  modified/deleted/generated files.
- Do not claim full squid, ANSYS, Fluent, or publication validation from unit
  tests, static tests, or short probes.
- Prefer red-to-green tests for correctness bugs. For dead code and roundtrip
  reductions, static/source tests plus targeted behavior tests are acceptable.
- Treat host/device transfers in default hot paths as first-class failures.
- Keep compatibility shims only where active non-test users still exist.

## Already Completed In Priority Pass

These are still part of this full goal and must remain green:

1. `simulation_core/coupling/fsi_coupling.py`: Aitken step state stores the raw
   pre-relaxation residual for the next Aitken update.
2. `benchmarks/official/solid_mpm_fsi_runner.py`: flow pressure min/max no
   longer use NumPy `initial=0.0` on signed pressure fields.
3. `simulation_core/coupling/hibm_mpm/core.py`: stress marker diagnostics are
   opt-in for stress sampling reports.
4. `simulation_core/fluids/solver.py`: predictor backtrace obstacle guarding
   marches across multi-cell paths instead of only checking six adjacent cells.
5. `simulation_core/coupling/hibm_mpm/core.py`: two zero-call legacy viscous
   marker kernels were removed while active base/split-mode kernels remain.

## Additional Progress In Full Pass

The full goal remains active. These additional findings have been repaired with
focused tests:

1. `solid_mpm_fsi_runner.py`: preflow now measures marker no-slip residual after
   projection and `_preflow_only_report()` propagates the measured feedback and
   no-slip diagnostics instead of hardcoding perfect residuals.
2. `hibm_mpm/core.py`: nonpositive pressure-Neumann transmissibility rows are
   counted as invalid and skipped before owner-slot or row-list allocation, so
   zero-area/narrow-gap rows cannot block valid owners.
3. `solver.py`: closed-domain projections ignore stale HIBM outlet reachability
   failure state when evaluating physical projection gates.
4. `hibm_mpm/core.py`: marker aggregate `action_reaction_residual_n` is no
   longer computed as `total + (-total)`; it now reports primary/secondary force
   grouping closure while scatter reports preserve the marker-to-MPM force
   conservation residual.
5. `hibm_mpm/core.py`: symmetric pressure-pair diagnostics now have a real
   independent-ladder fallback path and can set `pressure_pair_fallback_used`.
6. `cases/ansys_vertical_flap_fsi.py`: HIBM sharp search/probe/interpolation
   controls are declared config fields instead of hidden `getattr` defaults.
7. `hibm_mpm/core.py`: two-sided stress nearest-cell spacing lookup uses
   centered rounding (`floor(coord + 0.5)`) instead of base-cell floor.
8. `hibm_mpm/core.py`: mirrored far-pressure closure mode now discards the dry
   side gradient explicitly instead of reusing stale inside-gradient state.
9. `hibm_mpm/core.py`: pressure-Neumann row-list overflow reports the actually
   materialized capped capacity rather than the raw over-capacity row count.
10. `solver.py`: `restore_state()` clears transient
    `velocity_dirichlet_boundary_marker_region_id` state along with the active
    and weight fields.
11. `neo_hookean_mpm.py`: host snapshot packed counters are read as f64, and
    transfer-error expected momentum/force excludes particles whose P2G stencil
    is out of bounds.
12. `hibm_mpm/core.py`: one-sided stress face diagnostics count configured
    one-sided policies, not only selected closure anchors.
13. `solid_mpm_fsi_runner.py`: solid substep CFL spacing is derived from the
    solid MPM grid/bounds rather than the fluid grid.
14. `projected_ibm.py`: Robin matrix final diagnostics are separated from the
    force report used for the applied host-fallback impulse.
15. `solver.py`: `project()` uses a lightweight pressure-interface policy probe
    when the caller does not request a full report.
16. `hibm_mpm/core.py`: post-solid reachability floods and HIBM grid-field
    classifications are skipped/reused when their inputs have not changed.
17. `tri_surface.py`: opt-in report-only active force-cell counting uses a
    linear hash aggregation instead of a serial O(n^2) duplicate scan.
18. `solver.py`: row-cloud orphan cleanup returns before grid downloads when
    no unreached cells/components make cleanup possible.
19. `hibm_mpm/core.py`: velocity-Dirichlet marker-region stamping is
    deterministic; concurrent region candidates are collected with atomic
    min/max and conflicting cells are marked unassigned.
20. `hibm_mpm/core.py`: velocity-Dirichlet region row counts are scalar
    device-side counters instead of two full-grid `to_numpy()` downloads.
21. `solver.py` and `solid_mpm_fsi_runner.py`: default flow state reporting
    reads scalar device-side counts/peak/pressure extrema; expensive speed
    percentiles require explicit `flow_report_include_percentiles`.
22. `solver.py` and `solid_mpm_fsi_runner.py`: zmax inlet boundary reporting
    uses a solver-side scalar report on the default path.
23. `solver.py`: solid-band marking now uses a two-stage candidate/apply sweep
    so one kernel does not make later cells see obstacles created earlier in the
    same sweep.
24. `solver.py` and `solid_mpm_fsi_runner.py`: default zmax inlet boundary
    refresh uses a device-side top-face kernel plus scalar report; legacy
    host fallback remains for explicit extra no-slip row/layer configurations.
25. `hibm_mpm/core.py`, `hibm_mpm/reports.py`, and squid history fields: the
    dead pressure-Neumann gradient limiter count was removed from production
    reports and field schemas instead of exporting an always-zero diagnostic.
26. `simulation_core/__init__.py` and tests: the legacy top-level module alias
    installer was removed after migrating remaining test imports to layered
    package paths; facade tests now assert the old alias modules are absent.
27. `hibm_mpm/core.py`: pressure-interface row-list duplicate compaction now
    stays device-side via a Taichi kernel, preserving overflow/capacity
    behavior without materializing the full row list on the host.
28. `fluids/solver.py`: HIBM row-cloud orphan cleanup no longer downloads full
    grids for `np.argwhere`/Python DFS/NumPy mask write-back. Compact component
    cleanup and raw-label singleton overflow cleanup now run through device
    kernels, with tests covering overflow singletons, protection radii, and
    uncompacted positive labels.
29. `fluids/solver.py`: unreached-component label compaction no longer uses
    host `np.unique` or label-grid `to_numpy/from_numpy`. Raw component counts,
    component-size distribution stats, compact label tables, and compact-label
    assignment now run device-side, with overflow/statistics regression tests.
30. `fluids/solver.py` and `solid_mpm_fsi_runner.py`: legacy cell-obstacle
    dynamic solid updates now prefer a solver device API that rasterizes flap
    rows from solid particle fields, applies candidate obstacles, zeros solid
    velocities, and reports added/removed counts without full-grid CPU
    roundtrips. The NumPy path remains only as a compatibility fallback.
31. `simulation_core/__init__.py`, facade tests, and `docs/MODULE_MAP.md`:
    public API cleanup is now explicit. Legacy top-level module aliases are
    removed and tested absent; the remaining root `__all__` is documented as an
    intentional package-level facade rather than dead code inferred from
    internal usage only.
32. `solid_mpm_fsi_runner.py`: the legacy
    `_solid_obstacle_from_mpm_particles()` fallback no longer rasterizes each
    rest-y row through Python row/grid loops. It now uses NumPy row
    accumulators and broadcast overlap masks, while the default runner path
    still prefers the solver device API.

## Current Open Items

No confirmed audit item from this goal remains open after the targeted repair
pass. Remaining limitations are validation-scope limits: the evidence here is
focused unit/static tests and short behavior probes, not a fresh full squid,
ANSYS, Fluent, or publication validation run.

## Phase 1 - Correctness Repairs

Complete all confirmed MEDIUM and LOW calculation findings, including:

1. `solver.py`: keep CFL>1 backtrace wall protection green; add a behavior test
   for a multi-cell jump through a one-cell obstacle if feasible.
2. `fsi_coupling.py`: keep Aitken residual storage aligned with the sibling
   fixed-point implementation; update tests that locked the old behavior.
3. `projected_ibm.py`: replace global-min-spacing legacy probe distance with an
   auto/local distance path for nonuniform grids.
4. `mooney_shell/core.py`: stop `preserve_external_force` body-force
   accumulation across substeps.
5. `mooney_shell/core.py`: add Uv NaN/out-of-bounds protection matching the Tri
   path.
6. `mooney_shell/core.py`: make `step()` honor partition/region thickness in
   pressure stiffness paths.
7. `solver.py`: make unreached-set mean add/subtract barrier predicates
   symmetric.
8. `solver.py`: preserve the pressure-interface RHS, or equivalent divergence
   state, across rejected cleanup rollback.
9. `solver.py`: reset or gate stale HIBM projection stats so closed-domain
   projection cannot inherit outlet-path HIBM failure flags.
10. `hibm_mpm/core.py`: fix invalid/narrow-gap relocated rows so `weight=0`
    no-op rows do not occupy cells and block valid owners. The
    pressure-Neumann row family was fixed in the prior repair round (see
    "Additional Progress In Full Pass" item 2). Velocity-Dirichlet lazy-row
    weight pinning and claim-after-sample ordering are completed in this
    repair round.
11. `solid_mpm_fsi_runner.py`: keep pressure extrema reduction green.
12. `solid_mpm_fsi_runner.py`: prevent ymin no-slip refresh from overwriting
    active marker feedback targets in sustained mode.
13. `hibm_mpm/core.py`: fix or remove `next_pressure_neumann_gradient`
    diagnostics that use full-step dt and are immediately overwritten.
14. LOW correctness items from the audit: floor `+0.5` consistency, mode-4
    gradient handling, split-viscous obstacle crossing guard parity,
    deterministic region stamping, marker-vs-particle scatter count naming or
    counting, capped active row count reporting, stale marker-region restore
    cleanup, solid-band deterministic marking if required, Neo-Hookean expected
    momentum in-bounds predicate, f64 packed counters, solid spacing reporting,
    and Robin post-impulse force capture.

Acceptance:

- Each fixed bug has a focused test or a documented reason why static/source
  verification is the right contract.
- Existing physical-report semantics are preserved unless explicitly corrected
  by the audit item.
- No new case-layer shortcut is introduced to mask a core solver defect.

## Phase 2 - Default-Path Performance Repairs

Complete all confirmed performance findings, prioritizing default-path host
traffic and Python loops:

1. `solid_mpm_fsi_runner.py::_apply_marker_feedback_to_fluid` and
   `_write_marker_velocity_constraints`: move marker feedback and constraint
   writing to device-side kernels or an existing device-resident contract,
   eliminating the per-step full-grid CPU roundtrips and Python marker loop.
2. `hibm_mpm/core.py::stress_marker_diagnostics`: keep default reports opt-in
   and ensure runner only requests final/archival diagnostics where consumed.
3. `hibm_mpm/core.py`: avoid disconnected-region report D2H work before early
   returns and reduce per-step repeated full-grid reads.
4. `fluids/solver.py`: replace row-cloud orphan host `np.argwhere`/Python DFS
   with existing or new device-side labeling/compaction where feasible.
5. `hibm_mpm/core.py`: avoid duplicate full-node/full-triangle classification
   when marker geometry has not changed.
6. `hibm_mpm/core.py`: remove or bypass the substep triple full-grid pass whose
   result is immediately overwritten.
7. `fluids/solver.py::project`: avoid unconditional full-grid reporting kernels
   when only a boolean is needed.
8. `tri_surface.py`: remove or restructure opt-in Robin serial O(n^2) diagnostic
   loops so large meshes do not run a serial billion-iteration path.
9. Remaining MEDIUM/LOW roundtrip findings from the audit table: compact label
   host unique, dynamic obstacle update D2H/H2D loops, no-slip residual scalar
   measurement, zmax inlet refresh/report, per-step flow reports and percentiles,
   velocity row counts, pressure row compaction, and particle-to-obstacle Python
   loops.

Acceptance:

- Hot-path tests or source contracts prove the expensive default path is gone.
- Any new device-side helper has a small behavior test on a toy grid/marker set.
- Reports still contain required scalar counters; full arrays/dicts are only
  materialized when a caller explicitly consumes them.

## Phase 3 - Dead Code And Dead Diagnostics

Complete all confirmed dead-code cleanup:

1. Keep the two removed legacy viscous kernels absent.
2. Remove or make truthful the seven "never alarms" diagnostics, including
   constant-false pressure pair fallback, action-reaction self-subtraction, and
   preflow history fields that hardcode perfect no-slip residuals.
3. Remove dead config `getattr` fields that have no config definitions, or add
   real config fields and tests if they are intentionally supported.
4. Migrate the last non-test users of compatibility aliases (`runtime`,
   `generic_fsi_solver`) and delete the stale alias table once safe.
5. Audit public `__all__` exports and remove zero-user exports unless the API is
   intentionally public and documented.
6. Keep "test-only but intentionally retained" modules documented separately
   from true dead code: CAD subsystem, `FsiDriver`, and uniform-spacing search
   variants must not be deleted just because production callers are absent.

Acceptance:

- Static tests verify deleted symbols/diagnostics do not silently return.
- Compatibility tests are updated after migrations.
- No public surface is removed without either migrating active users or writing a
  short compatibility note in existing docs.

## Phase 4 - Verification And Completion Criteria

Minimum local verification before declaring this full goal complete:

1. Targeted tests for every changed subsystem:
   - `tests/solvers/test_fsi_coupling.py`
   - `tests/cases/test_ansys_vertical_flap_fsi.py`
   - relevant HIBM-MPM stress/pressure tests
   - relevant fluid predictor/projection tests
   - relevant Mooney/Neo-Hookean/projected-IBM tests
2. `compileall` for touched Python modules.
3. A grep/static audit showing removed dead symbols and dead diagnostics are no
   longer present, or are replaced by truthful diagnostics.
4. A short final table mapping each audit item to one of:
   `fixed`, `already fixed before full goal`, `not applicable in current code`,
   or `blocked with evidence`.

Full-run validation is not automatically required for every code cleanup, but
any claim of squid, ANSYS, Fluent, or physical parity requires fresh artifact
evidence beyond this goal file and targeted tests.
