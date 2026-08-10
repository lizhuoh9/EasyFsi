# FSI Audit Priority Repair Goal - 2026-07-02

## Objective

Confirm and repair the highest-value findings from the July 2026 FSI audit in the
current refactored HIBM-MPM checkout, with regression coverage and bounded
claims. This goal covers confirmed correctness/performance/dead-code issues in
the core solver and official FSI runner, not full physical validation of every
audit row.

## Confirmed Findings In Scope

1. `benchmarks/official/solid_mpm_fsi_runner.py::_flow_state_report` clamps
   pressure extrema with NumPy `initial=0.0`, so all-positive pressure reports a
   false zero minimum and all-negative pressure reports a false zero maximum.
2. `simulation_core/coupling/fsi_coupling.py::update_interface_reaction_for_next_step`
   stores the post-relaxation committed residual as the previous Aitken residual.
   The sibling fixed-point path stores the raw residual, so the two Aitken paths
   are inconsistent.
3. `simulation_core/coupling/hibm_mpm/core.py::sample_fluid_stress_to_marker_tractions`
   always attaches full per-marker diagnostics to the stress report, forcing many
   device-to-host transfers even when callers only need scalar counters.
4. `simulation_core/fluids/solver.py::_backtrace_crosses_adjacent_obstacle_face`
   only checks the six adjacent cells, while predictor advection can tolerate
   local CFL values above one; a backtrace can therefore jump through a one-cell
   obstacle.
5. `simulation_core/coupling/hibm_mpm/core.py` still contains the legacy
   `_add_viscous_marker_tractions_kernel` and `_add_split_viscous_marker_tractions_kernel`
   definitions with no call sites, while newer base/split viscous kernels are the
   active implementation.

## Repair Contract

- Add narrow regression or static tests before the corresponding implementation
  changes where practical.
- Keep solver behavior in `simulation_core`; do not hide numerical fixes in case
  code.
- Do not rewrite unrelated validation artifacts or clean the pre-existing dirty
  worktree.
- Do not claim full squid/ANSYS physical validation from unit or static tests.

## Validation Plan

- Targeted unit tests for Aitken residual storage and pressure extrema.
- Static/source tests for optional marker diagnostics and dead kernel removal.
- A focused solver source/behavior test for multi-cell backtrace obstacle
  guarding.
- Run only targeted tests required by these changes unless a later failure
  indicates a broader suite is needed.
