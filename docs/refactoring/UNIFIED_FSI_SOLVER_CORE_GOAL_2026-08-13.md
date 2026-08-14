# Unified FSI Solver Core Goal - 2026-08-13

## Status

Architecture implementation complete; numerical production validation pending.
A fresh production preflow is valid, but the first unified 5-step gate was
operator-aborted after one committed step while measured performance defects
were repaired. It is not 5-step or Fluent-parity evidence.

## Implemented architecture

The repository now has one shared FSI solver:

- `simulation_core/drivers/generic_fsi_solver.py::solve_fsi_runtime` owns the
  physical-step loop, marker-velocity residual, Picard bootstrap, IQN-ILS
  update, convergence decision, rollback, and single commit.
- ANSYS vertical flap, Turek-Hron FSI1/2/3, and Squid sharp HIBM-MPM expose
  runtime adapters to that core.
- fluid and solid substeps remain inside one adapter trial and are recomputed
  from the saved physical-step state for every coupling iteration.
- `runtime_executor`, `FsiDriver`, the old force fixed-point core, selectable
  coupling modes, and the Squid legacy coupling module have been deleted.

The earlier `GENERIC_FSI_SOLVER_BOUNDARY_GOAL_2026-06-28.md` explicitly allowed
an adapter-backed first stage. That transition boundary is now obsolete.

## Objective

Make `simulation_core` own one partitioned HIBM-MPM FSI algorithm. ANSYS
vertical flap, Turek-Hron FSI1/2/3, and Squid sharp HIBM-MPM must execute that
same solver. A case may select physical parameters and convergence tolerances;
it may not implement a different time loop, fixed-point loop, or interface
unknown.

## Canonical coupling contract

1. `solve_fsi()` owns the physical-step loop and the within-step coupling loop.
2. The canonical HIBM-MPM interface unknown is marker velocity in m/s.
3. Every trial restores the fluid, solid, and marker state saved at the start of
   the physical step, applies the current marker-velocity guess, and evaluates
   one complete fluid -> traction -> solid -> marker-feedback map.
   `begin_step()` must capture the pre-preparation rollback base before reseed,
   boundary, or predictor writes; adapters may then capture a post-preparation
   base used only for repeated trials. A new begin first invalidates the prior
   transaction, and rollback is armed only after every pre-mutation snapshot
   succeeds; commit and rollback both clear it.
4. Convergence uses marker-count-invariant RMS absolute and relative velocity
   residuals.
5. IQN-ILS is the canonical accelerator, with one ordinary relaxed Picard step
   until a secant is available. The implementation must stay standard and
   compact; case-specific recovery branches and historical byte-compatibility
   paths are not retained.
6. The accepted trial is committed exactly once. A non-converged required step
   fails closed and cannot be reported as completed.
7. Pressure-Neumann scratch state may be restored with a trial, but it is not a
   second independently selected interface unknown.
8. The marker transaction contains position, explicit pressure-probe origin,
   velocity, normal, area, marker count, projection counts, and tip-cap binding.
   A failed feedback write may retire live geometry, but rollback must republish
   this complete state before the original exception escapes.
9. Trial rollback invalidates classified HIBM topology metadata while retaining
   shape-stable Taichi search, boundary, and projection resources. Clearing the
   full resource cache changes template-field identity and forces a new kernel
   specialization for each coupling trial.
10. Marker-target Kaczmarz closure preserves its ordered Gauss-Seidel updates
    and full configured sweep budget. Sweeps execute in serialized batches of
    64 to reduce host kernel dispatches without changing row order or adding an
    early convergence exit.

## Time-step and substep hierarchy

One generic solver does not require every physics component to use the same
integration increment. The ownership hierarchy is fixed:

1. `simulation_core` owns each physical FSI step of size `dt_s`.
2. `simulation_core` owns the marker-velocity coupling trials inside that step.
3. One trial may ask the fluid integrator for CFL-limited transport/RK
   substeps, the MPM integrator for elastic-wave-limited solid substeps, and
   HIBM/pressure solvers for their internal algebraic iterations.
4. Those component substeps are recomputed after every trial rollback and are
   never interpreted as additional committed physical steps.
5. A case may configure tolerances and stability limits, but it may not own a
   substep scheduler, physical-step loop, or interface-coupling loop.

ANSYS, Turek-Hron FSI1/2/3, and Squid may therefore select different numbers of
fluid and solid substeps while still executing exactly the same FSI solver.

## Runtime boundary

Cases provide a runtime adapter with only these responsibilities:

- initialize case-specific fluid, solid, HIBM, and output state;
- save/restore one physical-step base state;
- evaluate one trial for a supplied marker-velocity guess;
- commit the accepted trial and return one history row;
- finalize case-specific diagnostics and artifacts.

The adapter must not contain a physical-step loop or a coupling-iteration loop.
`FsiProblem.runtime_executor` is removed rather than preserved as a compatibility
path.

## Required deletions

- the executor-only implementation of `solve_fsi`;
- the duplicate loose-only `FsiDriver` run loop once its callers use
  `solve_fsi`;
- the Turek-Hron case-local Picard/Aitken/IQN state machine and its single-pass
  legacy branch;
- the Squid case-local sharp-marker fixed-point loop after migration;
- tests whose only purpose is to preserve those deleted implementations.

Ordinary non-HIBM fluid boundaries are not an alternate HIBM solver and remain
outside this deletion.

## Acceptance gates

### Architecture

- a toy runtime proves that `solve_fsi` itself performs every requested physical
  step and every coupling trial;
- source contracts reject `runtime_executor` and case-owned coupling loops;
- ANSYS and Turek-Hron report the same solver identity, interface unknown, and
  accelerator family;
- no production HIBM case can select the deleted single-pass implementation.

### Focused numerical validation

1. generic fixed-point RED/GREEN tests, including convergence and required
   non-convergence failure;
2. focused ANSYS and Turek runtime-adapter tests;
3. one small CUDA step for each adapter through `solve_fsi`;
4. ANSYS 5-step gate from the approved preflow state;
5. independent ANSYS 50-step run;
6. locked comparison to the Fluent 50-step history with every required error
   not greater than 10 percent.

No micro test, diagnostic replay, partial history, or source-stale preflow may be
reported as 50-step or Fluent-parity evidence.

## Current validation state

- Unified ownership, rollback/commit, substep nesting, ANSYS/Turek/Squid
  migration, and deleted-path source contracts are green.
- The focused shared-core/runtime/checkpoint group is green (`54` tests and
  `111` subtests), including five injected Squid begin-snapshot failures,
  original-exception preservation, absolute resume-time validation, Turek's
  post-prepare trial base, and explicit probe-origin checkpoint roundtrip.
- Turek case guards are green (`89` tests and `22` subtests). The focused ANSYS
  group is green (`85` passed, `2` explicitly deselected long tests). A real
  CUDA marker test also restores an explicit probe origin after geometry
  retirement. Independent static re-review found no remaining blocker/P1/P2
  in this transaction-fix scope.
- The broad Squid latest-config suite still contains unrelated stale
  pressure-outlet source-string assertions and is not claimed green here.
- Fresh production preflow `unified_core_preflow200_20260813_r02` stopped
  naturally at `75/200` after three stationary windows; the union maximum
  stationary metric was about `0.00933 <= 0.01`. Its schema-v8 snapshot passed
  a strict zero-step identity/load audit.
- The first unified gate
  `unified_core_fsi5_from_preflow_r02_20260813_r01` committed step 1 with three
  IQN/Picard trials and an absolute marker-velocity residual of about
  `4.13e-6 m/s`, then was operator-aborted during uncommitted step 2 after
  `7256.1 s`. Logs showed six specializations of the large HIBM claim kernel;
  root cause was full sharp-boundary resource-cache clearing on every trial.
- The performance repair keeps HIBM resources across trial rollback and batches
  4096 ordered closure sweeps into 64-sweep launches. Focused source contracts,
  transaction/cache tests, and a direct CUDA `65 x 1` versus `64 + 1` sweep
  comparison are green; the two CUDA target fields were byte-identical. A
  broader 4x4x4 assembly test reached its 10-minute bound while compiling an
  unrelated segment-reconstruction kernel and is not claimed passed or failed.
- A new source-matched preflow snapshot, 5-step gate, independent 50-step run,
  and locked Fluent comparison remain required.

## Working-tree constraints

The checkout is intentionally dirty. Preserve unrelated edits and validation
artifacts. Do not reset, clean, or rewrite large files wholesale. Make the
solver migration in reviewable slices and run the smallest relevant test after
each slice.
