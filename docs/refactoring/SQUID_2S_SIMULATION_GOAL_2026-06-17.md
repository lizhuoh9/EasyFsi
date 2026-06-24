# Squid 2s Simulation Goal

## Objective

Make the real-CAD squid case reach the 2s pressure waveform with physically meaningful jet evidence, not just solver-green bookkeeping.

The first executable route is:

1. keep the current non-pinning `legacy_projected_reduced` projected-IBM path;
2. make its step-internal interface-reaction loop stability-aware so high-CFL trials cannot be reported as converged success;
3. route this mode through the stable FV-CG pressure projection by default;
4. use short-step artifacts as the gate before attempting a 2s waveform.

## Constraints

- Do not retry keep-open, post-Dirichlet reproject, naive L2 source injection, or simply increasing `--ibm-correction-iterations`.
- Do not claim 2s readiness from 1-2 step probes.
- Treat `interior_divergence_l2=0` as necessary but not sufficient; also require bounded CFL, finite pressure/velocity, nonzero membrane force, and outlet flow comparable to membrane volume flux.
- Keep sharp HIBM-MPM evidence separate from legacy projected/reduced evidence.

## Red Baseline

Known artifact:

`_codex_validation/diffuse_mode_4step`

- mode: `legacy_projected_reduced`
- step 2 failure: `cfl=7.202000e+00 >= 5.000000e-01`
- `interior_divergence_l2=0.0`

Interpretation: the old diffuse/projected-IBM route can report zero interior divergence while the velocity field is already CFL-fatal.

## Implemented

### Stability-aware trial acceptance

Files:

- `simulation_core/fsi_coupling.py`
  - added `accept_evaluation` to `solve_interface_reaction_fixed_point`
  - added `rejected_trial_count` to `InterfaceReactionFixedPointResult`
  - prevents a converged-but-rejected trial from overriding the best acceptable trial
- `cases/squid_soft_robot.py`
  - added cheap trial CFL sampling through `ReducedSquidFSI.sample_cfl_report`
  - rejects legacy projected/reduced trial evaluations with `trial_cfl >= 0.5`
  - writes accepted/rejected trial CFL diagnostics into `history.csv` and `summary.json`
- `tests/test_fsi_coupling.py`
  - added coverage proving that a rejected converged trial is not accepted

### FV-CG auto route for legacy projected/reduced

Files:

- `cases/squid_soft_robot.py`
  - `resolve_pressure_solver("auto", fsi_coupling_mode="legacy_projected_reduced")` now resolves to `fv_cg`
  - summary keeps both `pressure_solver_requested` and resolved/actual solver fields
- `tests/test_squid_latest_core_config.py`
  - added/updated focused assertions for the legacy projected/reduced auto route

Rationale: the short probes showed the old automatic uniform-grid route was the CFL wall, while FV-CG keeps the same non-pinning projected-IBM path stable for at least 10 explicit steps.

### Configurable Aitken bounds

Files:

- `simulation_core/fsi_coupling.py`
  - added `aitken_lower_bound` and `aitken_upper_bound` to accepted-step and step-internal interface-reaction updates
  - defaults remain `[0.01, 1.5]`, preserving the previous Aitken clipping behavior
- `cases/squid_soft_robot.py`
  - added `--interface-reaction-aitken-lower-bound`
  - added `--interface-reaction-aitken-upper-bound`
  - records the configured bounds in preflight, row, and summary diagnostics
- `tests/test_fsi_coupling.py`
  - verifies accepted-step and fixed-point Aitken paths honor the configured lower and upper bounds
- `tests/test_squid_latest_core_config.py`
  - verifies the CLI default and explicit selection

Rationale: this was added as a controlled way to test whether the step-8 overshoot was caused by either the previous `0.01` floor or the `1.5` upper clip. The probes below show these scalar bounds are useful diagnostics but not sufficient as the main stabilization route.

### CFL-aware rejected-trial backtracking

Files:

- `simulation_core/fsi_coupling.py`
  - added `rejected_trial_backtrack` to `solve_interface_reaction_fixed_point` and `solve_and_apply_interface_reaction_step`
  - added `rejected_trial_backtrack_count` to `InterfaceReactionFixedPointResult`
  - rejected trials can now retry between the last accepted force and the rejected force
  - if the first trial in a step is rejected, the retry damps from zero force toward the rejected force
  - if every evaluated trial is rejected, the solver commits a zero-force failsafe instead of committing the unsafe rejected initial force
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-rejected-trial-backtrack`
  - records the configured value and per-step backtrack count in preflight, row, and summary diagnostics
- `tests/test_fsi_coupling.py`
  - verifies rejected converged trials are not accepted, rejected trials can be backtracked, first rejected trials backtrack toward zero, and all-rejected trials commit the zero-force failsafe
- `tests/test_squid_latest_core_config.py`
  - verifies the CLI default and explicit selection

Rationale: this is the first actual CFL-aware accepted-trial controller. It fixes the false-green/unsafe-commit behavior, but the runtime probes below show it still needs an adaptive explicit-step controller and still does not deliver strong-coupling convergence.

### Residual-growth rejected-trial gate

Files:

- `simulation_core/fsi_coupling.py`
  - added `residual_growth_rejection_factor` to `solve_interface_reaction_fixed_point` and `solve_and_apply_interface_reaction_step`
  - added `residual_growth_rejected_trial_count` to `InterfaceReactionFixedPointResult`
  - the gate is disabled by default with `math.inf`
  - when enabled, an otherwise stability-accepted trial is rejected if its physical residual norm exceeds the best accepted residual by the configured factor
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-residual-growth-rejection-factor`
  - records the configured factor and per-step residual-growth rejection count in preflight, row, and summary diagnostics
- `tests/test_fsi_coupling.py`
  - verifies parameter validation, default-disabled behavior, and residual-growth rejection/backtracking
- `tests/test_squid_latest_core_config.py`
  - verifies the default disabled state and explicit CLI selection

Rationale: this is a narrow trust gate for trials that improve CFL or pass the caller predicate while making the physical force residual worse. It is not a scalar target-map relaxation and it is disabled unless explicitly selected.

### Absolute residual rejected-trial gate

Files:

- `simulation_core/fsi_coupling.py`
  - added `max_accepted_residual_n` to `solve_interface_reaction_fixed_point` and `solve_and_apply_interface_reaction_step`
  - added `max_residual_rejected_trial_count` to `InterfaceReactionFixedPointResult`
  - the gate is disabled by default with `math.inf`
  - when enabled, an otherwise accepted trial is rejected if its physical force residual norm exceeds the configured Newton threshold
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-max-accepted-residual-n`
  - records the configured cap and per-step absolute-residual rejection count in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_fsi_coupling.py`
  - verifies parameter validation, default-disabled behavior, and absolute residual rejection/backtracking
- `tests/test_squid_latest_core_config.py`
  - verifies the default disabled state, explicit CLI selection, and fingerprint coverage

Rationale: this was added to test the suspected cross-step failure mode where an intra-step-improving trial can still leave the next physical step with a much larger residual. The probes below show that a hard absolute cap is too blunt for the current scaffold: it can prevent the large residual, but by rejecting every trial and falling back to `inf` residual rather than producing a usable strong-coupling update.

### Force-increment trust region

Files:

- `simulation_core/fsi_coupling.py`
  - added `trust_region_force_increment_n` to `solve_interface_reaction_fixed_point` and `solve_and_apply_interface_reaction_step`
  - added `trust_region_limited_update_count` to `InterfaceReactionFixedPointResult`
  - limits only the proposed next interface-reaction force update, `|F_next - F_current|`, and does not reject or hide the evaluated trial residual
  - the limiter is disabled by default with `math.inf`
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-trust-region-force-increment-n`
  - records the configured force-increment cap and per-step limited-update count in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_fsi_coupling.py`
  - verifies parameter validation and that the limiter changes the next trial force sequence without rejecting otherwise accepted trials
- `tests/test_squid_latest_core_config.py`
  - verifies the default disabled state, explicit CLI selection, and fingerprint coverage

Rationale: the hard residual cap showed the wrong failure mode: rejecting high-residual trials caused all-rejected failsafe steps. The trust-region limiter instead allows residuals to be measured while preventing the fixed-point map from jumping from small interface forces to very large force guesses in one iteration.

### Adaptive trust-radius feedback

Files:

- `simulation_core/fsi_coupling.py`
  - added `trust_region_adaptive`, `trust_region_shrink_factor`, and `trust_region_growth_factor`
  - added `trust_region_shrink_count`, `trust_region_growth_count`, and `trust_region_effective_force_increment_n` to `InterfaceReactionFixedPointResult`
  - when enabled, the effective force-increment radius shrinks after a trial residual grows relative to the previous trial and grows back after residual reduction
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-trust-region-adaptive`
  - added `--fsi-coupling-trust-region-shrink-factor`
  - added `--fsi-coupling-trust-region-growth-factor`
  - records adaptive trust-radius counts/configuration in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_fsi_coupling.py`
  - verifies adaptive shrink/grow behavior on a controlled fixed-point map
- `tests/test_squid_latest_core_config.py`
  - verifies explicit CLI selection and fingerprint coverage

Rationale: fixed `2N` trust radius was the best current short-step result, while fixed `1N` improved step 4 but not step 5. This adaptive mode tests whether residual feedback can combine both behaviors without hard rejecting trials.

### Residual-rebound trust backtrack

Files:

- `simulation_core/fsi_coupling.py`
  - added `trust_region_rebound_factor` and `trust_region_rebound_backtrack`
  - added `trust_region_rebound_backtrack_count` to `InterfaceReactionFixedPointResult`
  - when enabled, an otherwise accepted trial that rebounds above the best accepted residual by the configured factor places the next trial between the best accepted force and the rebounded force
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-trust-region-rebound-factor`
  - added `--fsi-coupling-trust-region-rebound-backtrack`
  - records rebound configuration/counts in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_fsi_coupling.py`
  - verifies validation and a controlled rebound sequence that returns from a worse trial toward the best trial
- `tests/test_squid_latest_core_config.py`
  - verifies default-disabled behavior, explicit CLI selection, and fingerprint coverage

Rationale: fixed `2N` step 5 improved dramatically by the fourth trial, then evaluated worse later trials. This policy tests a local backtrack around the best accepted trial without turning rebound into hard rejection.

### Best-trial rebound stop

Files:

- `simulation_core/fsi_coupling.py`
  - added `trust_region_rebound_stop_factor`
  - added `trust_region_rebound_stop_count` to `InterfaceReactionFixedPointResult`
  - when enabled, the fixed-point solve stops and commits the best accepted trial if a later otherwise accepted trial rebounds above the best accepted residual by the configured factor
  - the stop policy does not mark the rebounded trial as rejected and does not change the accepted force
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-trust-region-rebound-stop-factor`
  - records stop configuration/counts in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_fsi_coupling.py`
  - verifies validation and a controlled stop sequence that commits the best trial without rejecting the rebound trial
- `tests/test_squid_latest_core_config.py`
  - verifies default-disabled behavior, explicit CLI selection, and fingerprint coverage

Rationale: fixed `2N` and rebound-backtrack showed that after a good best trial, later exploratory trials can waste cost or slightly worsen the selected state. This policy is a conservative accept/stop rule around the current best force-increment limiter.

### Residual/CFL-triggered adaptive iteration budget

Files:

- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-adaptive-iterations-max`
  - added `--fsi-coupling-adaptive-iterations-residual-threshold-n`
  - added `--fsi-coupling-adaptive-iterations-cfl-threshold`
  - when enabled, a projected/reduced step can raise its step-internal interface-reaction iteration budget on the next step if the previous step's FSI residual or CFL exceeds the configured threshold
  - records base/requested iterations, trigger reason, previous residual/CFL, and summary trigger counts in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_squid_latest_core_config.py`
  - verifies default-disabled behavior, explicit CLI selection, and fingerprint coverage

Rationale: fixed 12-iteration probes showed that the old step-6 divergence failure is not a hard physics wall, but fixed 12 everywhere is too expensive. This opt-in budget controller tests whether difficult steps can receive more strong-coupling work without making every early/easy step pay the fixed high cost.

### High-residual rebound-stop ceiling

Files:

- `simulation_core/fsi_coupling.py`
  - added `trust_region_rebound_stop_max_residual_n` to `solve_interface_reaction_fixed_point` and `solve_and_apply_interface_reaction_step`
  - added `trust_region_rebound_stop_suppressed_count` to `InterfaceReactionFixedPointResult`
  - the ceiling is disabled by default with `math.inf`, preserving the previous stop2 behavior
  - when a rebound-stop condition occurs above the configured residual ceiling, the solver suppresses the stop and continues instead of committing a high-residual early stop
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-trust-region-rebound-stop-max-residual-n`
  - records the configured ceiling and suppressed-stop counts in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_fsi_coupling.py`
  - verifies parameter validation and a controlled high-residual sequence where rebound-stop is suppressed and the solve continues
- `tests/test_squid_latest_core_config.py`
  - verifies default-disabled behavior, explicit CLI selection, and fingerprint coverage

Rationale: the fixed 12-iteration positive probe crosses the old step-6 divergence guard partly because stop2 fires after a lower residual is found. This controller tests whether stop2 should be disallowed on high-residual steps. The runtime probe below shows the mechanism works, but simple high-residual suppression alone does not improve the step-6 trajectory.

### Same-step residual-triggered rerun

Files:

- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-same-step-rerun-iterations-max`
  - added `--fsi-coupling-same-step-rerun-residual-threshold-n`
  - after the first projected/reduced FSI fixed-point attempt, restores the current physical step's start state and reruns the same step with a larger iteration budget if the first attempt did not converge and its accepted residual exceeds the configured threshold
  - records trigger count, first-attempt residual, first-attempt iteration use, final requested iterations, and summary trigger totals in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_squid_latest_core_config.py`
  - verifies default-disabled behavior, explicit CLI selection, trigger predicate behavior, and checkpoint fingerprint coverage

Rationale: previous-row adaptive iteration triggers showed that step4/5-only budget raises are too late, while fixed 12 and early `0.0005N` triggers change the accepted-state branch from step 2 onward. Same-step rerun tests whether the runner can make that branch change within the current physical step without manually running fixed 12 from the start.

### Residual-quality fixed-point continuation

Files:

- `simulation_core/fsi_coupling.py`
  - added `residual_continuation_iterations_max`
  - added `residual_continuation_threshold_n`
  - added `residual_continuation_iteration_count` to `InterfaceReactionFixedPointResult`
  - when the base fixed-point budget ends without convergence and the best accepted residual is still above the configured threshold, the solver can append extra iterations inside the same fixed-point history instead of restarting the whole physical step
- `cases/squid_soft_robot.py`
  - added `--fsi-coupling-residual-continuation-iterations-max`
  - added `--fsi-coupling-residual-continuation-threshold-n`
  - records continuation configuration and per-step continuation counts in preflight, row, summary, and checkpoint fingerprint diagnostics
- `tests/test_fsi_coupling.py`
  - verifies continuation stops after the residual-quality threshold is reached rather than always exhausting the extra budget
- `tests/test_squid_latest_core_config.py`
  - verifies default-disabled behavior, explicit CLI selection, and checkpoint fingerprint coverage

Rationale: same-step rerun proved that the budget must rise by step 2, but rerunning a whole physical step duplicates work. Residual-quality continuation keeps the first six fixed-point trials and appends only the extra trials needed by the residual gate, preserving Aitken/history information and avoiding the 6+12 rerun pattern.

## Execution Evidence

### Strong coupling alone is insufficient

Artifact:

`_codex_validation/codex_goal_diffuse_fsi_iters6_4step_20260617_001`

Command used `--fsi-coupling-mode legacy_projected_reduced --fsi-coupling-iterations 6`.

Result:

- failed at step 2: `cfl=4.888498e+00 >= 5.000000e-01`
- `fsi_coupling_converged=True`
- step 2 residual: `7.759696184367069e-05`
- step 2 `interior_divergence_l2=0.0`

Interpretation: force fixed-point convergence was a false green for field stability.

### Trial acceptance gate fixes the false green, not the final instability

Artifact:

`_codex_validation/codex_goal_diffuse_stability_accept_4step_20260617_002`

Result:

- step 1 passed: `cfl=1.500852e-03`
- step 2 failed: `cfl=4.872431557519095`
- step 2 `fsi_coupling_rejected_trial_count=5`
- step 2 `fsi_coupling_converged=False`
- step 2 `fsi_coupling_residual_norm_n=inf`

Interpretation: the solver no longer accepts high-CFL trials as a successful coupled solve, but rejecting bad trials alone cannot stabilize the accepted re-advance.

### Excluded probes

- `_codex_validation/codex_goal_diffuse_stability_accept_passivity_4step_20260617_001`
  - `--interface-reaction-passivity-limit`
  - still failed at step 2 with `cfl=4.872431e+00`
- `_codex_validation/codex_goal_diffuse_relax0_4step_20260617_001`
  - `--interface-reaction-relaxation 0.0`
  - step 2 still failed with `cfl=4.421944e+00`
- `_codex_validation/codex_goal_diffuse_force_scale01_4step_20260617_001`
  - `--constraint-force-scale 0.1`
  - worsened step 2 to `cfl=8.279360e+00`
- `_codex_validation/codex_goal_diffuse_substeps2_4step_20260617_001`
  - `--fluid-substeps 2`
  - step 1 failed with `cfl=3.592050e+00`
- `_codex_validation/codex_goal_diffuse_stability_accept_substeps10_4step_20260617_001`
  - reached printed steps with `cfl=0`, but `main_displacement_z_m=0` and `solid_mpm_active_grid_nodes=0`
  - invalid static evidence
- `_codex_validation/codex_goal_diffuse_preserve_pressure_4step_20260617_001`
  - temporary pressure-preservation experiment, removed from code
  - worsened step 2 to `cfl=1.711223e+02`

### FV-CG explicit route passes short-step stability

Artifact:

`_codex_validation/codex_goal_diffuse_fvcg_10step_20260617_001`

Command used explicit `--pressure-solver fv_cg`.

Result:

- `completed_steps=10`
- `pressure_solver="fv_cg"`
- `max_cfl=0.003736858209595084`
- `max_interior_divergence_l2=0.0`
- `pressure_projection_cg_converged_all=True`
- `fsi_coupling_not_converged_count=0`
- `max_fsi_coupling_rejected_trial_count=0`
- final `main_displacement_z_m=-1.5471348888240755e-05`
- final `main_velocity_z_mps=-0.007886640727519989`
- final `solid_mpm_active_grid_nodes=9350`
- final `main_fsi_fluid_force_z_n=-0.0993008054792881`
- final `fsi_volume_source_m3s=1.100632107409183e-05`
- final `outlet_flow_negative_z_m3s=-0.0`

Interpretation: this is the first non-pinning projected-IBM route that materially improves the current failure mode. It keeps the coupled system moving and bounded for 10 steps, but it does not yet produce a jet.

### Auto route now selects FV-CG and reproduces the 10-step pass

Artifact:

`_codex_validation/codex_goal_diffuse_auto_fvcg_10step_20260617_001`

Command did not pass `--pressure-solver fv_cg`; it used the default `--pressure-solver auto`.

Result:

- `completed_steps=10`
- `pressure_solver_requested="auto"`
- `pressure_solver="fv_cg"`
- `max_cfl=0.003736851384331073`
- `max_interior_divergence_l2=0.0`
- `pressure_projection_cg_converged_all=True`
- `total_pressure_projection_cg_converged_all=True`
- `fsi_coupling_not_converged_count=0`
- `max_fsi_coupling_rejected_trial_count=0`
- `max_fsi_coupling_accepted_trial_cfl=0.003736852521875075`
- `max_fsi_coupling_trial_cfl=0.003994128573685885`
- final `main_displacement_z_m=-1.5471348888240755e-05`
- final `main_velocity_z_mps=-0.007886640727519989`
- final `solid_mpm_active_grid_nodes=9350`
- final `main_fsi_fluid_force_z_n=-0.09930071979761124`
- final `fsi_volume_source_m3s=1.1006300155713689e-05`
- final `outlet_flow_negative_z_m3s=-0.0`
- `validation_scope="explicit_step_count"`
- `validation_scope_complete=false`
- `completed_step_checks_passed=false`

Interpretation: the first stable short-step route is now available as the default auto path for this coupling mode. It is still not 2s-ready because outlet flow remains zero and the nozzle is under-resolved (`nozzle_diameter_cells_min=1`).

### Source reachability diagnostic identifies the outlet blocker

Implemented after the stable 10-step route:

- `simulation_core/fluid.py`
  - `pressure_outlet_fv_flux_report()` now splits source volume flux into:
    - `zmin_reachable_source_volume_flux_m3s`
    - `zmin_unreached_source_volume_flux_m3s`
- `cases/squid_soft_robot.py`
  - writes these as row fields:
    - `pressure_outlet_reachable_source_volume_flux_m3s`
    - `pressure_outlet_unreached_source_volume_flux_m3s`
  - writes final summary fields:
    - `final_pressure_outlet_reachable_source_volume_flux_m3s`
    - `final_pressure_outlet_unreached_source_volume_flux_m3s`
- `tests/test_core_fluid.py`
  - added a disconnected-component test proving the split is source-location aware

Artifact:

`_codex_validation/codex_goal_diffuse_auto_fvcg_source_reach_10step_20260617_001`

Result:

- `completed_steps=10`
- `pressure_solver_requested="auto"`
- `pressure_solver="fv_cg"`
- `max_cfl=0.003736846075792398`
- `fsi_coupling_not_converged_count=0`
- `solid_mpm_active_grid_nodes=9350`
- `active_water_connectivity.component_count=3`
- `active_water_connectivity.trapped_active_cell_count=9395`
- final `fsi_volume_source_m3s=1.1006281056324951e-05`
- final `pressure_outlet_source_volume_flux_m3s=1.1006284694303758e-05`
- final `pressure_outlet_reachable_source_volume_flux_m3s=0.0`
- final `pressure_outlet_unreached_source_volume_flux_m3s=1.1006288332282566e-05`
- final `pressure_outlet_velocity_flux_m3s=0.0`
- final `outlet_flow_negative_z_m3s=-0.0`

Interpretation: the current outlet failure is now localized. The FSI path creates nonzero volume source, but all of the pressure-outlet source lies in a z-min-unreachable active-water component. The pressure solver is not failing globally; it is correctly unable to discharge source from a trapped component.

### Reduced-water intersection is not sufficient

Artifact:

`_codex_validation/codex_goal_diffuse_intersect_reduced_water_10step_20260617_001`

Command added:

`--source-config-intersect-reduced-water-domain`

Result:

- `source_config_active_mask_intersected_with_reduced_water_domain=true`
- `reduced_water_intersection_added_obstacle_cell_count=115894`
- `completed_steps=10`
- `max_cfl=0.0037368559345070805`
- `active_water_connectivity_passed=false`
- final `pressure_outlet_reachable_source_volume_flux_m3s=0.0`
- final `pressure_outlet_unreached_source_volume_flux_m3s=1.100632107409183e-05`
- final `pressure_outlet_velocity_flux_m3s=0.0`
- final `outlet_flow_negative_z_m3s=-0.0`

Interpretation: simply intersecting the CAD active mask with the reduced analytic water domain does not connect the source to the z-min pressure outlet. This is a negative topology probe, not a viable propulsion route.

### Source location diagnostic identifies the trapped region

Implemented after the reachability split:

- `simulation_core/fluid.py`
  - adds reachable/unreached source cell counts
  - adds abs-flux-weighted centroid for the z-min-unreachable source
  - adds a source-cell bounding box for the z-min-unreachable source
- `cases/squid_soft_robot.py`
  - writes the new diagnostics to every row and final summary
- `tests/test_core_fluid.py`
  - verifies count, abs-flux, centroid, and bounding box on a disconnected 5x5x5 pressure-outlet case
- `tests/test_squid_latest_core_config.py`
  - verifies row propagation for the new diagnostics

Artifact:

`_codex_validation/codex_goal_diffuse_auto_fvcg_source_location_10step_20260617_001`

Result:

- `completed_steps=10`
- `pressure_solver_requested="auto"`
- `pressure_solver="fv_cg"`
- `max_cfl=0.003736854417781745`
- `fsi_coupling_not_converged_count=0`
- `solid_mpm_active_grid_nodes=9350`
- `active_water_connectivity.component_count=3`
- `active_water_connectivity.trapped_active_cell_count=9395`
- final `pressure_outlet_source_volume_flux_m3s=1.1006315617123619e-05`
- final `pressure_outlet_reachable_source_volume_flux_m3s=0.0`
- final `pressure_outlet_unreached_source_volume_flux_m3s=1.1006314707628917e-05`
- final `pressure_outlet_reachable_source_cell_count=0`
- final `pressure_outlet_unreached_source_cell_count=2704`
- final `pressure_outlet_unreached_source_abs_flux_m3s=3.6150991945760325e-05`
- final unreached-source centroid: `x=-0.03131044577452054`, `y=0.015906803540647486`, `z=1.0291377502249397`
- final unreached-source bbox:
  - `x=[-0.06396874785423279, 0.0004895833553746343]`
  - `y=[-0.017750000581145287, 0.04975000023841858]`
  - `z=[1.0068421363830566, 1.0387719869613647]`
- final `pressure_outlet_velocity_flux_m3s=0.0`
- final `outlet_flow_negative_z_m3s=-0.0`

Interpretation: the stable FV-CG projected-IBM route generates source in a compact trapped region near `z≈1.03m`. The immediate next implementation target is the topology/active-mask connection between this source region and the intended outlet path, not another CFL-only coupling tweak.

### Surface-seed to z-min connection repair creates outlet response

Implemented as an explicit opt-in diagnostic topology repair:

- `cases/squid_soft_robot.py`
  - adds `--source-config-connect-surface-seeds-to-zmin`
  - adds `--source-config-surface-seed-zmin-connection-max-carve-cells`
  - uses a host-side 0-1 BFS to minimally carve obstacle cells between z-min active water and surface-seeded active-water components
  - records the repair under `source_config_fluid_topology.fluid_active_mask_surface_seed_zmin_connection`
  - keeps the repair disabled by default because it changes the initial CAD-derived obstacle mask
- `tests/test_squid_latest_core_config.py`
  - verifies the repair carves a single-cell barrier
  - verifies the max-carve limit prevents unintended widening

Artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_1step_20260617_002`

Result:

- `completed_steps=1`
- topology repair:
  - initial unreachable surface-seeded cells: `9395`
  - initial unreachable surface-seeded components: `2`
  - connected paths: `2`
  - carved cells: `2`
  - final unreachable surface-seeded cells: `0`
- `active_water_connectivity.component_count=1`
- `active_water_connectivity.trapped_active_cell_count=0`
- `active_water_connectivity_passed=true`
- final `pressure_outlet_source_volume_flux_m3s=2.762720612281555e-07`
- final `pressure_outlet_reachable_source_volume_flux_m3s=2.7627208964986494e-07`
- final `pressure_outlet_unreached_source_volume_flux_m3s=0.0`
- final `pressure_outlet_velocity_flux_m3s=2.7627211807157437e-07`
- final `pressure_outlet_velocity_to_source_ratio=1.000000238418579`
- final `outlet_flow_negative_z_m3s=1.9314815025150978e-10`
- `max_cfl=0.009159265418670006`

Interpretation: this proves the previous outlet blocker was the two-cell topological disconnect between the surface-seeded water pockets and the z-min pressure-outlet component. Once connected, the pressure outlet can discharge the FSI source with a velocity/source ratio near 1.

### Six-step connected-route evidence

Artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_6step_20260617_001`

Command added:

`--source-config-connect-surface-seeds-to-zmin --source-config-surface-seed-zmin-connection-max-carve-cells 512`

Result:

- `completed_steps=6`
- `validation_scope="explicit_step_count"`
- `validation_scope_complete=false`
- topology repair:
  - initial unreachable surface-seeded cells: `9395`
  - initial unreachable surface-seeded components: `2`
  - connected paths: `2`
  - carved cells: `2`
  - final unreachable surface-seeded cells: `0`
- `active_water_connectivity_passed=true`
- `max_cfl=0.18202860206365584`
- final `pressure_outlet_source_volume_flux_m3s=4.8044853429018985e-06`
- final `pressure_outlet_reachable_source_volume_flux_m3s=4.8044853429018985e-06`
- final `pressure_outlet_unreached_source_volume_flux_m3s=0.0`
- final `pressure_outlet_velocity_flux_m3s=4.804483978659846e-06`
- final `pressure_outlet_velocity_to_source_ratio=0.9999997615814209`
- final `outlet_flow_negative_z_m3s=3.3583278380433512e-09`
- `max_fsi_coupling_rejected_trial_count=5`
- `total_fsi_coupling_rejected_trial_count=20`
- `fsi_coupling_not_converged_count=4`
- `completed_step_checks_passed=false`

Interpretation: the route has moved from "stable but no jet" to "short-step stable with reachable source and pressure-outlet velocity/source conservation." It is still not strong-coupling-complete and not 2s-ready, because several fixed-point iterations are rejected/not converged and the 10-step extension below still hits a CFL wall.

### Connected-route 10-step and stability negative probes

Artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_10step_20260617_001`

Result:

- failed at step 7
- step 7 `cfl=5.215436e-01 >= 5.000000e-01`
- source remained reachable before failure:
  - step 7 `pressure_outlet_reachable_source_volume_flux_m3s=1.572574910824187e-05`
  - step 7 `pressure_outlet_unreached_source_volume_flux_m3s=0.0`
  - step 7 `pressure_outlet_velocity_to_source_ratio=1.0000020265579224`
- FSI stability gate rejected high-CFL trials at the failure:
  - step 7 `fsi_coupling_rejected_trial_count=6`
  - step 7 `fsi_coupling_residual_norm_n=inf`

Adaptive probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_adaptive_10step_20260617_001`

- command used `--adaptive-fluid-substeps --adaptive-fluid-substeps-target-cfl 0.05 --adaptive-fluid-substeps-max 16`
- stopped manually after more than 17 minutes with no `history.csv`
- not usable evidence for stability

Target-map relaxation negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_targetrelax05_8step_20260617_001`

- command used `--fsi-coupling-target-map-relaxation 0.5`
- failed earlier than the unmodified connected route:
  - step 3 `cfl=0.49893279331071033`
  - step 4 `cfl=1.7607417379106793 >= 0.5`
  - step 4 `max_fluid_speed_mps=8.64925765991211`
- source reachability was still fixed at the failure:
  - step 4 `pressure_outlet_source_volume_flux_m3s=5.310135748004541e-05`
  - step 4 `pressure_outlet_reachable_source_volume_flux_m3s=5.310135748004541e-05`
  - step 4 `pressure_outlet_unreached_source_volume_flux_m3s=0.0`
  - step 4 `pressure_outlet_velocity_to_source_ratio=1.0000001192092896`
- FSI stability gate still saw unstable trials:
  - step 4 `fsi_coupling_rejected_trial_count=6`
  - step 4 `fsi_coupling_residual_norm_n=inf`
  - step 4 `fsi_coupling_trial_cfl_max=4.336910738054541`
- high-residual dump bounding box: `i=4..37`, `j=9..38`, `k=40..55`

Fixed `fluid_substeps=2` negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_substeps2_10step_20260617_001`

- command used `--fluid-substeps 2`
- failed earlier than the unmodified connected route:
  - step 3 `cfl=0.16134096596922193`
  - step 4 `cfl=0.5763948440551758 >= 0.5`
  - step 4 `max_fluid_speed_mps=5.6628265380859375`
- source reachability was still fixed at the failure:
  - step 4 `pressure_outlet_source_volume_flux_m3s=3.4676217183005065e-05`
  - step 4 `pressure_outlet_reachable_source_volume_flux_m3s=3.467622445896268e-05`
  - step 4 `pressure_outlet_unreached_source_volume_flux_m3s=0.0`
  - step 4 `pressure_outlet_velocity_to_source_ratio=1.0000001192092896`
- FSI stability gate still rejected unstable trials:
  - step 4 `fsi_coupling_rejected_trial_count=6`
  - step 4 `fsi_coupling_residual_norm_n=inf`
  - step 4 `fsi_coupling_trial_cfl_max=2.7570247854505263`
- high-residual dump bounding box: `i=4..38`, `j=9..38`, `k=39..55`

IQN-ILS negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_iqn_10step_20260617_001`

- command used `--fsi-coupling-solver iqn_ils`
- step 2 residual improved relative to Aitken:
  - IQN-ILS step 2 `fsi_coupling_residual_norm_n=2.4944742032448735e-06`
  - Aitken step 2 `fsi_coupling_residual_norm_n=0.00034431431783712817`
- it did not move the CFL wall:
  - step 7 `cfl=0.5217799186706543 >= 0.5`
  - step 7 `pressure_outlet_source_volume_flux_m3s=1.5732914107502438e-05`
  - step 7 `pressure_outlet_velocity_to_source_ratio=1.0000005960464478`
  - step 7 `fsi_coupling_residual_norm_n=inf`
- high-residual dump bounding box: `i=5..36`, `j=9..38`, `k=39..55`

Small pressure-matrix Robin cost probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_matrix4_10step_20260617_001`

- command used `--interface-reaction-robin-matrix-impedance-ns-m 4.0`
- stopped manually after no `history.csv` was produced in the observation window
- `run_process.json` was marked `status="stopped"` with `stopped_reason="manual_stop_after_no_history_csv"`
- not usable as stability evidence; current matrix-Robin implementation is too expensive for this short-turn validation path until profiled separately

Physical-mode explicit Robin impedance probes:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical75_10step_20260617_001`

- command used `--interface-reaction-robin-impedance-ns-m 75.0 --interface-reaction-robin-target-mode physical`
- moved the original step-7 CFL failure forward, but still failed at step 8:
  - step 7 `cfl=0.4390371884618487`
  - step 7 `pressure_outlet_velocity_to_source_ratio=1.0000003576278687`
  - step 8 `cfl=0.9315632854189192 >= 0.5`
  - step 8 `pressure_outlet_velocity_to_source_ratio=1.0000005960464478`

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_10step_20260617_001`

- command used `--interface-reaction-robin-impedance-ns-m 100.0 --interface-reaction-robin-target-mode physical`
- best current short-run stability improvement:
  - step 7 `cfl=0.42681024670600887`, down from baseline step 7 `0.5215435521943229`
  - step 7 `pressure_outlet_source_volume_flux_m3s=1.2825780686398502e-05`
  - step 7 `pressure_outlet_velocity_to_source_ratio=0.9999997615814209`
  - failed at step 8 with `cfl=0.8884603296007428 >= 0.5`
  - step 8 `pressure_outlet_velocity_to_source_ratio=1.0000003576278687`
- high-residual dump bounding box at failure: `i=5..36`, `j=9..38`, `k=40..55`

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical200_10step_20260617_001`

- command used `--interface-reaction-robin-impedance-ns-m 200.0 --interface-reaction-robin-target-mode physical`
- over-stabilized and worsened the route:
  - step 3 `cfl=0.48475085326603484`
  - step 4 `cfl=0.7075927989823477 >= 0.5`
  - step 4 `pressure_outlet_velocity_to_source_ratio=0.0`
- failure high-residual cells moved to the z-min boundary plane: `i=1..11`, `j=1..46`, `k=1..1`

No-Aitken fixed-relaxation probes:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_noaitken_relax005_10step_20260617_001`

- command used `--interface-reaction-robin-impedance-ns-m 100.0 --interface-reaction-robin-target-mode physical --no-interface-reaction-aitken --interface-reaction-relaxation 0.005`
- completed 10 explicit steps
- `max_cfl=0.3329745271376201`
- `pressure_projection_cg_converged_all=True`
- `total_pressure_projection_cg_converged_all=True`
- final `pressure_outlet_reachable_source_cell_count=2704`
- final `pressure_outlet_unreached_source_cell_count=0`
- final `pressure_outlet_velocity_to_source_ratio=1.0000028610229492`
- `fsi_coupling_not_converged_count=10`
- `max_fsi_coupling_residual_norm_n=1.862711332179945`
- `completed_step_checks_passed=false`

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_noaitken_relax01_10step_20260617_001`

- command used `--interface-reaction-robin-impedance-ns-m 100.0 --interface-reaction-robin-target-mode physical --no-interface-reaction-aitken --interface-reaction-relaxation 0.01`
- completed 10 explicit steps
- `max_cfl=0.42613643237522664`
- `pressure_projection_cg_converged_all=True`
- `total_pressure_projection_cg_converged_all=True`
- final `pressure_outlet_reachable_source_cell_count=2704`
- final `pressure_outlet_unreached_source_cell_count=0`
- `fsi_coupling_not_converged_count=10`
- `max_fsi_coupling_residual_norm_n=2.981702045861192`
- `total_fsi_coupling_rejected_trial_count=4`
- `completed_step_checks_passed=false`

Aitken lower-bound negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_aitkenfloor005_10step_20260617_001`

- command used `--interface-reaction-robin-impedance-ns-m 100.0 --interface-reaction-robin-target-mode physical --interface-reaction-aitken-lower-bound 0.005`
- failed at step 8:
  - step 7 `cfl=0.2464894652366638`
  - step 8 `cfl=0.6858818829059601 >= 0.5`
  - step 8 `fsi_coupling_residual_norm_n=inf`
  - step 8 `fsi_coupling_rejected_trial_count=6`
  - step 8 `fsi_coupling_trial_cfl_max=0.9512128625597271`
- source reachability stayed fixed:
  - step 8 `pressure_outlet_reachable_source_cell_count=2704`
  - step 8 `pressure_outlet_unreached_source_cell_count=0`
- high-residual dump moved to the z-min plane:
  - `max_abs_residual_s=0.27005088329315186`
  - bbox `i=1..11`, `j=1..46`, `k=1..1`

High-iteration budget negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_noaitken_relax01_iters30_3step_20260617_001`

- command used `--no-interface-reaction-aitken --interface-reaction-relaxation 0.01 --fsi-coupling-iterations 30 --steps 3`
- stopped manually after roughly 2.5 minutes with no `history.csv`
- not usable as a short-turn validation route

Time-step reduction negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_dt05_aitken_10step_20260617_001`

- command used `--time-step-scale 0.5` with default Aitken and physical Robin `Z=100`
- first two steps converged:
  - step 1 `fsi_coupling_residual_norm_n=0.00037285815816858234`
  - step 2 `fsi_coupling_residual_norm_n=0.0004173964294839195`
- failed at step 6:
  - step 5 `cfl=0.4693726658821106`
  - step 6 `cfl=0.5862415220056261 >= 0.5`
  - step 6 `fsi_coupling_residual_norm_n=inf`
  - step 6 `fsi_coupling_trial_cfl_max=4.295854997634888`
- source reachability stayed fixed at failure:
  - step 6 `pressure_outlet_velocity_to_source_ratio=1.0000020265579224`

Aitken upper-bound negative probes:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_aitkenupper025_10step_20260617_001`

- command used `--interface-reaction-aitken-upper-bound 0.25`
- failed at step 8:
  - step 7 `cfl=0.434682366677693`
  - step 8 `cfl=0.9400076355252948 >= 0.5`
  - step 8 `fsi_coupling_residual_norm_n=inf`
  - step 8 `fsi_coupling_trial_cfl_max=5.697209031241281`
  - step 8 `pressure_outlet_velocity_to_source_ratio=1.000001072883606`

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_aitkenupper010_10step_20260617_001`

- command used `--interface-reaction-aitken-upper-bound 0.1`
- failed at step 8:
  - step 7 `cfl=0.44624709401811874`
  - step 8 `cfl=1.0385705147470747 >= 0.5`
  - step 8 `fsi_coupling_residual_norm_n=inf`
  - step 8 `fsi_coupling_trial_cfl_max=6.079952267238072`
  - step 8 `pressure_outlet_velocity_to_source_ratio=0.9999997019767761`

Interpretation: after topology repair, the next blocker is no longer reachability. It is coupled field stability/cost once the outlet path is actually open. A scalar target-map relaxation of `0.5`, IQN-ILS alone, fixed `fluid_substeps=2`, high fixed-point iteration count, plain time-step halving, Aitken lower-bound reduction, and Aitken upper-bound clipping are excluded as primary stabilization routes. Pressure-matrix Robin needs separate profiling before it can be used as a validation path. Explicit physical-mode Robin impedance is the first non-pinning stabilizer that moves the failure boundary in the right direction, but the stable 10-step evidence currently comes only from fixed low relaxation with Aitken disabled. That 10-step route is useful as a scaffold, not as final strong-coupling evidence, because every step remains fixed-point-not-converged.

### CFL-aware rejected-trial backtracking probes

Initial backtracking implementation:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_backtrack05_10step_20260617_001`

- command used `--fsi-coupling-rejected-trial-backtrack 0.5`
- failed at step 4:
  - step 3 `cfl=0.41508375150816784`
  - step 4 `cfl=1.6101222242627824 >= 0.5`
  - step 4 `fsi_coupling_rejected_trial_count=6`
  - step 4 `fsi_coupling_rejected_trial_backtrack_count=0`

First-rejected damping enabled:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_backtrack05_10step_20260617_002`

- failed at step 4:
  - step 4 `cfl=1.6101237773895263 >= 0.5`
  - step 4 `fsi_coupling_rejected_trial_count=6`
  - step 4 `fsi_coupling_rejected_trial_backtrack_count=6`

All-rejected zero-force failsafe enabled:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_backtrack05_10step_20260617_003`

- failed at step 4, but the final-step CFL was reduced:
  - step 4 `cfl=1.112920148032052 >= 0.5`
  - step 4 `max_fluid_speed_mps=5.466976165771484`
  - step 4 `fsi_coupling_rejected_trial_count=6`
  - step 4 `fsi_coupling_rejected_trial_backtrack_count=6`
  - step 4 `pressure_outlet_velocity_to_source_ratio=1.0000004768371582`
  - step 4 `pressure_projection_cg_converged_all=True`

Interpretation: rejected-trial backtracking fixes unsafe trial commitment and reduces the step-4 CFL after all trials are rejected, but it is not sufficient by itself. Once the outlet source is connected, the explicit fluid step needs its own CFL control.

### Adaptive explicit-step CFL probe

Moderate adaptive target:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_backtrack05_adaptive_10step_20260617_001`

- command added `--adaptive-fluid-substeps --adaptive-fluid-substeps-target-cfl 0.25 --adaptive-fluid-substeps-max 16`
- crossed the old step-4 failure point, then failed at step 5:
  - step 4 `fluid_substeps=3`, `cfl=0.4625230857304164`
  - step 5 `fluid_substeps=7`, `cfl=0.5223048755100795 >= 0.5`

More conservative adaptive target:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_backtrack05_adaptive010_safety2_10step_20260617_001`

- command added `--adaptive-fluid-substeps-target-cfl 0.1 --adaptive-fluid-substeps-safety 2.0 --adaptive-fluid-substeps-max 64`
- manually stopped after step 5 because step 6 was still running with high adaptive substep cost
- stable recorded rows:
  - step 1 `fluid_substeps=1`, `cfl=0.0092768396916134`, `fsi_coupling_converged=True`
  - step 2 `fluid_substeps=1`, `cfl=0.07232754113418716`, `fsi_coupling_converged=False`
  - step 3 `fluid_substeps=2`, `cfl=0.2408395298889705`, `fsi_coupling_converged=False`
  - step 4 `fluid_substeps=10`, `cfl=0.10981260333742414`, `fsi_coupling_converged=False`
  - step 5 `fluid_substeps=22`, `cfl=0.14377311034636064`, `fsi_coupling_converged=False`, `fsi_coupling_residual_norm_n=10.051203345624042`

Interpretation: conservative adaptive substeps can keep the connected route CFL-stable through five recorded steps, but the cost grows immediately and the FSI fixed-point residual gets worse. This is not a 2s-ready route; it is evidence that the next blocker is coupled update quality and cost, not source reachability or pressure CG convergence.

### Accepted-trial reuse A/B on the conservative adaptive route

Reuse artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command added `--reuse-accepted-fsi-trial-state` to the conservative adaptive run and limited the comparison to `--steps 5`
- finished 5/5 requested explicit steps:
  - `completed_steps=5`
  - `full_pressure_waveform_steps=4000`
  - `validation_scope_complete=false`
  - `validation_scope_reason=explicit_steps_before_full_pressure_waveform`
- reuse was active in 4 of 5 rows:
  - `accepted_fsi_trial_state_reuse_count=4`
  - step 1 reused, `total_pressure_projection_cg_project_calls=8`
  - step 2 reused, `total_pressure_projection_cg_project_calls=12`
  - step 3 not reused, `total_pressure_projection_cg_project_calls=28`
  - step 4 reused, `total_pressure_projection_cg_project_calls=120`
  - step 5 reused, `total_pressure_projection_cg_project_calls=264`
- compared with the no-reuse conservative adaptive run:
  - step 4 `step_wall_time_s` improved from `124.72855880000861` to `97.80343329999596`
  - step 5 `step_wall_time_s` improved from `273.2922233999998` to `213.86397109998506`
  - step 5 `total_pressure_projection_cg_project_calls` improved from `308` to `264`
- physical/coupling status stayed incomplete:
  - `max_cfl=0.24084023364952634`
  - `fsi_coupling_not_converged_count=4`
  - `max_fsi_coupling_residual_norm_n=10.05155920462909`

Interpretation: accepted-trial reuse is a valid opt-in cost reduction for this route and should stay in future short probes. It does not solve the strong-coupling residual, because it removes only the final accepted re-advance layer, not the expensive and non-converged FSI trial loop.

Step-5 fixed-point history diagnosis from the reuse artifact:

- step 5 starts with a small trial force but a large physical target:
  - trial 0 `|F|=0.8864863659339395`
  - trial 0 `|T|=58.75422478367749`
  - trial 0 `|R|=59.29391075512634`
- the map overshoots during the six-trial budget:
  - trial 1 `|F|=29.11151715120191`
  - trial 1 `|T|=750.3662404696599`
  - trial 1 `|R|=775.2366324685057`
- the final accepted trial is CFL-safe but still not a strong-coupling solution:
  - trial 5 `|F|=2.7580413728552737`
  - trial 5 `|T|=12.517249828570938`
  - trial 5 `|R|=10.05155920462909`

Interpretation: the step-5 residual is caused by the interface map itself, not by the final accepted-state re-advance. Future work needs a better coupled update/trust-region model, not just more reuse.

### IQN-ILS A/B on the same scaffold

IQN artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_iqn_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command changed only the fixed-point solver from Aitken to `--fsi-coupling-solver iqn_ils`
- finished 5/5 requested explicit steps:
  - `completed_steps=5`
  - `full_pressure_waveform_steps=4000`
  - `validation_scope_complete=false`
- improved CFL/cost relative to Aitken+reuse:
  - Aitken+reuse `max_cfl=0.24084023364952634`
  - IQN+reuse `max_cfl=0.21076745305742536`
  - Aitken+reuse `mean_total_pressure_projection_cg_project_calls=86.4`
  - IQN+reuse `mean_total_pressure_projection_cg_project_calls=70.8`
  - Aitken+reuse step 5 `fluid_substeps=22`, `step_wall_time_s=213.86397109998506`
  - IQN+reuse step 5 `fluid_substeps=18`, `step_wall_time_s=199.57521079998696`
- worsened strong-coupling residual:
  - Aitken+reuse `max_fsi_coupling_residual_norm_n=10.05155920462909`
  - IQN+reuse `max_fsi_coupling_residual_norm_n=23.673683368019653`
  - IQN step 4 `fsi_coupling_residual_norm_n=17.842221192433282`
  - IQN step 5 `fsi_coupling_residual_norm_n=23.673683368019653`
  - `fsi_coupling_not_converged_count=4`

Interpretation: IQN-ILS reduces CFL and projection cost in this scaffold, but it does not solve the strong-coupling residual; it makes the residual worse by step 4-5. Do not use IQN-ILS as the next primary stabilization route without a residual-aware trust-region/acceptance criterion.

### Residual-growth gate A/B on the Aitken scaffold

Residual-growth gate artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_residgate2_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command added `--fsi-coupling-residual-growth-rejection-factor 2.0` to the Aitken+backtrack+adaptive010+safety2+reuse scaffold
- finished 5/5 requested explicit steps:
  - `completed_steps=5`
  - `full_pressure_waveform_steps=4000`
  - `validation_scope_complete=false`
- residual-growth gate did not trigger:
  - `max_fsi_coupling_residual_growth_rejected_trial_count=0`
  - `total_fsi_coupling_residual_growth_rejected_trial_count=0`
- final short-step state remained essentially the same as Aitken+reuse:
  - factor-2 gate `max_cfl=0.24084074326923913`
  - Aitken+reuse `max_cfl=0.24084023364952634`
  - factor-2 gate `max_fsi_coupling_residual_norm_n=10.05012913635335`
  - Aitken+reuse `max_fsi_coupling_residual_norm_n=10.05155920462909`
  - factor-2 gate `fsi_coupling_not_converged_count=4`
  - factor-2 gate `mean_total_pressure_projection_cg_project_calls=86.4`

Interpretation: the intra-step residual-growth gate is implemented and observable, but factor `2.0` does not improve the real scaffold because the large-residual trials are already being rejected by the CFL/stability predicate or the accepted residual decreases within the step. The remaining problem is cross-step/absolute residual growth: step 5 can be an intra-step improvement while still being far worse than the previous physical step.

### Absolute residual gate A/B on the Aitken scaffold

Strict cap artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_maxresid1_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command added `--fsi-coupling-max-accepted-residual-n 1.0` to the Aitken+backtrack+adaptive010+safety2+reuse scaffold
- finished 5/5 requested explicit steps:
  - `completed_steps=5`
  - `requested_steps=5`
  - `max_cfl=0.2408418110438756`
- the absolute residual gate triggered only after the early small-residual steps:
  - step 1 residual `0.0009911243848114177`, max-residual rejections `0`, converged
  - step 2 residual `0.003060784629250693`, max-residual rejections `0`, not converged
  - step 3 residual `0.19669777525963017`, max-residual rejections `0`, not converged
  - step 4 residual `inf`, max-residual rejections `6`, not converged
  - step 5 residual `inf`, max-residual rejections `6`, not converged
- summary:
  - `total_fsi_coupling_max_residual_rejected_trial_count=12`
  - `max_fsi_coupling_max_residual_rejected_trial_count=6`
  - `fsi_coupling_not_converged_count=4`
  - `accepted_fsi_trial_state_reuse_count=2`

Wider cap artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_maxresid5_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command changed only the absolute residual cap to `--fsi-coupling-max-accepted-residual-n 5.0`
- finished 5/5 requested explicit steps:
  - `completed_steps=5`
  - `requested_steps=5`
  - `max_cfl=0.24083977256502423`
- result remained effectively the same as the strict cap:
  - step 1 residual `0.0009911260872252422`, max-residual rejections `0`, converged
  - step 2 residual `0.0030607119349967847`, max-residual rejections `0`, not converged
  - step 3 residual `0.19669565459349758`, max-residual rejections `0`, not converged
  - step 4 residual `inf`, max-residual rejections `6`, not converged
  - step 5 residual `inf`, max-residual rejections `6`, not converged
- summary:
  - `total_fsi_coupling_max_residual_rejected_trial_count=12`
  - `max_fsi_coupling_max_residual_rejected_trial_count=6`
  - `fsi_coupling_not_converged_count=4`
  - `accepted_fsi_trial_state_reuse_count=2`

Interpretation: the hard absolute residual cap is useful as a diagnostic and prevents falsely accepting large-residual trials, but it is not the primary stabilization route. Both `1N` and `5N` caps force all trials to be rejected at step 4 and step 5, yielding `inf` residuals. The next implementation should not simply tighten rejection thresholds; it needs a residual-aware trust-region update that can produce a smaller accepted force/target update instead of turning the whole step into an all-rejected failsafe.

### Force-increment trust-region A/B on the Aitken scaffold

Baseline for comparison:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- no force-increment trust region
- completed 5/5 requested explicit steps
- `max_cfl=0.24084023364952634`
- `max_fsi_coupling_residual_norm_n=10.05155920462909`
- `fsi_coupling_not_converged_count=4`
- step 4:
  - residual `0.873073321201799`
  - rejected trials `2`
  - trial force norms: `[0.171660, 4.355071, 2.119455, 1.004634, 0.758156, 0.883283]`
  - physical residual norms: `[8.949603, 47.317736, 19.941273, 6.690961, 1.279102, 0.873073]`
- step 5:
  - residual `10.05155920462909`
  - rejected trials `3`
  - trial force norms: `[0.886486, 29.111517, 14.296771, 6.903469, 3.238580, 2.758041]`
  - physical residual norms: `[59.293911, 775.236632, 355.131436, 151.579968, 55.914281, 10.051559]`

Trust-region 1N artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc1_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command added `--fsi-coupling-trust-region-force-increment-n 1.0`
- completed 5/5 requested explicit steps
- `max_cfl=0.24083977256502423`
- `max_fsi_coupling_residual_norm_n=6.7939362932145295`
- `fsi_coupling_not_converged_count=4`
- `total_fsi_coupling_trust_region_limited_update_count=8`
- `total_fsi_coupling_rejected_trial_count=0`
- step 4 improved:
  - residual `0.519326740301165`
  - trust-region limited updates `2`
  - trial force norms: `[0.171660, 0.886919, 0.873413, 0.984231, 0.985441, 0.996903]`
  - physical residual norms: `[8.949635, 5.478660, 0.519327, 1.417739, 1.214004, 1.032580]`
- step 5 improved but did not converge:
  - residual `6.7939362932145295`
  - trust-region limited updates `6`
  - trial force norms: `[0.876484, 0.855294, 1.434163, 1.856753, 2.313280, 2.977596]`
  - physical residual norms: `[44.708105, 28.020732, 20.831923, 19.058536, 13.382040, 6.793936]`

Trust-region 2N artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command changed only the force-increment cap to `--fsi-coupling-trust-region-force-increment-n 2.0`
- completed 5/5 requested explicit steps
- `max_cfl=0.24085132394518172`
- `max_fsi_coupling_residual_norm_n=3.377682802933034`
- `fsi_coupling_not_converged_count=4`
- `total_fsi_coupling_trust_region_limited_update_count=4`
- `total_fsi_coupling_rejected_trial_count=1`
- step 4 was worse than 1N and slightly worse than no-cap:
  - residual `0.9249608332754756`
  - trust-region limited updates `1`
  - rejected trials `1`
  - physical residual norms: `[8.949884, 17.045290, 5.478588, 2.082443, 1.100494, 0.924961]`
- step 5 was the best current 5-step result:
  - residual `3.377682802933034`
  - trust-region limited updates `3`
  - rejected trials `0`
  - trial force norms: `[0.978574, 1.687578, 2.414861, 3.229249, 4.128292, 4.133666]`
  - physical residual norms: `[45.830080, 24.357160, 23.148113, 3.377683, 8.650132, 5.111125]`

Interpretation: the force-increment trust region is the first residual-aware update model in this sequence that improves the real connected scaffold without turning steps into all-rejected failsafes. It materially reduces the step-5 residual from `10.051559N` to `3.377683N` while preserving bounded CFL. It is still not strong-coupling-complete: 4 of 5 steps remain fixed-point-not-converged, and the best residual is still far above `1.0e-3N`. The next work should tune or adapt the trust radius and/or iteration budget around this model, not return to hard residual rejection.

### Trust-region iteration budget and adaptive-radius probes

Higher fixed iteration budget artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_iters10_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command changed fixed `trustinc2` only by increasing `--fsi-coupling-iterations` from `6` to `10`
- manually stopped after 4 recorded steps because step 4 had already degraded and step 5 runtime was high
- process state was corrected to `stopped_by_codex` in `run_process.json`
- recorded rows:
  - step 1 residual `0.0009911434817551762`, converged, iterations used `4`
  - step 2 residual `0.0015675025958265516`, not converged, iterations used `10`
  - step 3 residual `0.12033496239975783`, not converged, iterations used `10`
  - step 4 residual `1.9755538497868919`, not converged, iterations used `10`
- step 4 physical residual norms:
  - `[7.223710, 16.375898, 5.574663, 3.251415, 1.975554, 2.122942, 2.282806, 2.349898, 2.419003, 2.489902]`

Interpretation: more iterations under a fixed `2N` trust radius help early rows but make step 4 worse than the 6-iteration `trustinc2` run (`1.975554N` vs `0.924961N`) and much worse than `trustinc1` step 4 (`0.519327N`). Do not use higher fixed iteration count as the next primary route unless it is paired with a better accept/stop/adaptive policy.

Default adaptive-radius artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_adaptive_backtrack05_adaptive010_safety2_reuse_5step_20260617_001`

- command added `--fsi-coupling-trust-region-adaptive` to fixed `trustinc2`
- used default shrink/grow factors:
  - `--fsi-coupling-trust-region-shrink-factor 0.5`
  - `--fsi-coupling-trust-region-growth-factor 1.25`
- completed 5/5 requested explicit steps:
  - `completed_steps=5`
  - `max_cfl=0.24083967549460272`
  - `max_fsi_coupling_residual_norm_n=3.8482910839502975`
  - `fsi_coupling_not_converged_count=4`
  - `total_fsi_coupling_trust_region_limited_update_count=5`
  - `total_fsi_coupling_trust_region_shrink_count=5`
  - `total_fsi_coupling_trust_region_growth_count=5`
  - `total_fsi_coupling_rejected_trial_count=1`
- step 4 was slightly better than fixed `trustinc2`:
  - adaptive residual `0.8844559313195509`
  - fixed `trustinc2` residual `0.9249608332754756`
  - shrink/grow counts: `1` / `4`
- step 5 was worse than fixed `trustinc2`:
  - adaptive residual `3.8482910839502975`
  - fixed `trustinc2` residual `3.377682802933034`
  - shrink/grow counts: `1` / `1`
- step 5 physical residual norms:
  - `[48.706032, 25.529133, 24.891052, 3.848291, 8.490584, 4.425182]`

Interpretation: the adaptive-radius plumbing works and remains CFL-stable, but the default residual-to-previous-trial policy is not yet better than fixed `2N`. It slightly improves step 4 but loses the best step-5 residual. The current best 5-step scaffold remains fixed `--fsi-coupling-trust-region-force-increment-n 2.0` with 6 iterations.

### Residual-rebound trust-backtrack probe

Fair A/B artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_rebound2_backtrack05_adaptive010_safety2_reuse_5step_20260617_003`

- command added to fixed `trustinc2`:
  - `--fsi-coupling-trust-region-rebound-factor 2.0`
  - `--fsi-coupling-trust-region-rebound-backtrack 0.5`
  - `--projection-divergence-tolerance 0.1` to match the previous fixed `trustinc2` artifact
- completed 5/5 requested explicit steps:
  - `status=finished`
  - `max_cfl=0.24085244025502883`
  - `max_fsi_coupling_residual_norm_n=3.379246411726515`
  - `fsi_coupling_not_converged_count=4`
  - `total_fsi_coupling_rejected_trial_count=1`
  - `total_fsi_coupling_rejected_trial_backtrack_count=1`
  - `total_fsi_coupling_trust_region_limited_update_count=4`
  - `total_fsi_coupling_trust_region_rebound_backtrack_count=1`
  - `accepted_fsi_trial_state_reuse_count=3`
- fixed `trustinc2` comparison:
  - fixed max residual: `3.377682802933034`
  - rebound max residual: `3.379246411726515`
  - fixed max CFL: `0.24085132394518172`
  - rebound max CFL: `0.24085244025502883`
  - both left 4 of 5 steps not converged
- per-step residual comparison:
  - step 4 fixed/rebound: `0.9249608332754756` / `0.9251045064536456`
  - step 5 fixed/rebound: `3.377682802933034` / `3.379246411726515`
  - rebound triggered only on step 5, count `1`

Negative startup artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_rebound2_backtrack05_adaptive010_safety2_reuse_5step_20260617_002`

- same route but missing `--projection-divergence-tolerance 0.1`
- failed at step 4 under the current default `0.01` guard:
  - `interior_divergence_l2=0.04046643709492241 > 0.01`
- interpretation: this is a guard-configuration mismatch relative to the fixed `trustinc2` artifact, not a fair rebound physics comparison

Interpretation: residual-rebound backtracking is correctly instrumented and can trigger, but factor `2.0` / backtrack `0.5` is not a runtime improvement over fixed `2N`. It slightly worsens both step 4 and step 5 residuals while preserving essentially the same CFL. The current best 5-step scaffold remains fixed `--fsi-coupling-trust-region-force-increment-n 2.0` without rebound.

### Best-trial rebound-stop probes

5-step positive A/B artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_backtrack05_adaptive010_safety2_reuse_5step_20260618_001`

- command added to fixed `trustinc2`:
  - `--fsi-coupling-trust-region-rebound-stop-factor 2.0`
  - `--projection-divergence-tolerance 0.1` to match the fixed `trustinc2` artifact
- completed 5/5 requested explicit steps:
  - `status=finished`
  - `max_cfl=0.24084028218473708`
  - `max_fsi_coupling_residual_norm_n=3.3774420680770194`
  - `fsi_coupling_not_converged_count=4`
  - `total_fsi_coupling_trust_region_rebound_stop_count=1`
  - `accepted_fsi_trial_state_reuse_count=3`
  - `mean_step_wall_time_s=79.62112641999848`
- fixed `trustinc2` comparison:
  - fixed max residual: `3.377682802933034`
  - stop2 max residual: `3.3774420680770194`
  - fixed max CFL: `0.24085132394518172`
  - stop2 max CFL: `0.24084028218473708`
  - fixed mean step wall time: `106.82204185998998s`
  - stop2 mean step wall time: `79.62112641999848s`
- step 5 comparison:
  - fixed residual: `3.377682802933034`, iterations `6`, FSI wall time `214.96952700000838s`
  - stop2 residual: `3.3774420680770194`, iterations `5`, FSI wall time `146.97094120000838s`

Interpretation: best-trial rebound stop is the first accept/stop policy that improves the fixed `2N` scaffold without changing the accepted force model. It gives a small residual improvement and a meaningful short-run cost reduction. It is still not a 10-step stability proof.

10-step fixed `2N + stop2` artifact:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_backtrack05_adaptive010_safety2_reuse_10step_20260618_001`

- command used the same stop2 policy with `--steps 10`
- failed at step 6:
  - `status=failed`
  - `failed_step=6`
  - `error=step 6 numerical guard failed: interior_divergence_l2=3.100108e-01 > 1.000000e-01`
- recorded rows before failure:
  - step 5 residual `3.3775307630447977`, stop count `1`, fluid substeps `18`
  - step 6 residual `88.55802078108698`, stop count `0`, fluid substeps `54`
  - step 6 `cfl=0.12830266422695583`
  - step 6 `interior_divergence_l2=0.3100107910512836`
  - step 6 `fsi_coupling_trust_region_limited_update_count=6`
  - step 6 FSI wall time `497.460756400018s`
- step 6 residual history was large but did not rebound; it decreased from huge values to `88.558N`, so the rebound-stop policy did not trigger

Interpretation: stop2 improves the 5-step scaffold but the 10-step gate still fails at step 6. The next blocker is not post-best rebound; it is entry into a high-residual physical step where every limited trial remains far from a force fixed point and the pressure projection divergence guard trips.

10-step conservative-radius negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc1_stop2_backtrack05_adaptive010_safety2_reuse_10step_20260618_001`

- command changed the stop2 10-step run only by lowering:
  - `--fsi-coupling-trust-region-force-increment-n 1.0`
- failed at step 6:
  - `status=failed`
  - `failed_step=6`
  - `error=step 6 numerical guard failed: interior_divergence_l2=1.401305e-01 > 1.000000e-01`
- recorded rows:
  - step 4 residual `0.5193182180337962`, stop count `1`
  - step 5 residual `6.793998270468162`, stop count `0`
  - step 6 residual `154.1180477253874`, stop count `0`
  - step 6 `cfl=0.04983925175453935`
  - step 6 `interior_divergence_l2=0.14013050641121927`
  - step 6 fluid substeps hit the configured max `64`

Interpretation: lowering the force-increment radius from `2N` to `1N` lowers step-6 CFL and divergence but worsens the force residual and still fails the 10-step guard. The next route should not be simple scalar trust-radius reduction; it needs a step-6 high-residual entry policy, likely around cross-step force prediction/rollback or residual-aware accepted-step damping, with pressure-divergence evidence as the gate.

### Higher iteration and adaptive-budget probes

Fixed 12-iteration positive probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_iter12_backtrack05_adaptive010_safety2_reuse_7step_20260618_001`

- command changed the stop2 scaffold by raising `--fsi-coupling-iterations` from `6` to `12` and adding `--checkpoint-every-step`
- completed 7/7 requested explicit steps:
  - `status=finished`
  - `max_interior_divergence_l2=0.08921277693678552`
  - `max_cfl=0.21816191375255584`
  - `max_fsi_coupling_residual_norm_n=12.853885887268461`
  - `total_fsi_coupling_trust_region_rebound_stop_count=1`
- key rows:
  - step 5 residual `1.7749075654147994`, iterations `12`, CFL `0.21816191375255584`
  - step 6 residual `8.449486169843349`, iterations `9`, stop count `1`, CFL `0.057476333987956145`, `interior_divergence_l2=0.08921277693678552`
  - step 7 residual `12.853885887268461`, iterations `12`, CFL `0.13074439693411052`, `interior_divergence_l2=1.4898263799356254e-06`
- cost remains high:
  - step 6 wall time `542.8900764000136s`
  - step 7 wall time `809.9804937000154s`

Interpretation: the old step-6 divergence failure is not a hard non-pinning route wall. More step-internal strong-coupling work can push the connected projected-IBM route through step 6 and step 7, but residuals remain far from convergence and the cost is not 2s-ready. This is a positive stability-direction artifact, not a full waveform route.

Late residual-triggered adaptive-budget negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_adaptiter12_resgt1_backtrack05_adaptive010_safety2_reuse_6step_20260618_001`

- command kept base `--fsi-coupling-iterations 6` and added:
  - `--fsi-coupling-adaptive-iterations-max 12`
  - `--fsi-coupling-adaptive-iterations-residual-threshold-n 1.0`
- failed at step 6:
  - step 5 residual `3.3775092108493006`, requested `6`, used `5`
  - step 6 requested `12`, used `9`, residual `10.699421019007092`
  - step 6 `interior_divergence_l2=0.18092965398365624 > 0.1`

CFL + residual trigger negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_adaptiter12_resgt05_cflgt010_backtrack05_adaptive010_safety2_reuse_6step_20260618_001`

- command added:
  - `--fsi-coupling-adaptive-iterations-max 12`
  - `--fsi-coupling-adaptive-iterations-residual-threshold-n 0.5`
  - `--fsi-coupling-adaptive-iterations-cfl-threshold 0.1`
- failed at step 6:
  - step 4 requested `12` due CFL trigger but remained close to the base state: residual `0.9249282192660134`, `interior_divergence_l2=0.04047933532897851`
  - step 5 requested `12` due residual trigger but stop2 ended at 5 iterations: residual `3.384149836890215`, stop count `1`
  - step 6 requested `12`, used `9`, residual `10.8078611025893`, stop count `1`
  - step 6 `interior_divergence_l2=0.18121123994692284 > 0.1`

High-residual stop2 ceiling negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_stopmaxres1_adaptiter12_resgt05_cflgt010_backtrack05_adaptive010_safety2_reuse_6step_20260618_001`

- command added:
  - `--fsi-coupling-trust-region-rebound-stop-max-residual-n 1.0`
- failed at step 6:
  - step 5 requested/used `12`, suppressed stop count `1`, stop count `0`, residual `3.3775866863975437`, CFL `0.14740171546027775`
  - step 6 requested/used `12`, suppressed stop count `3`, stop count `0`, residual `10.705033462434281`
  - step 6 `cfl=0.07551487741016205`
  - step 6 `interior_divergence_l2=0.18095017980191166 > 0.1`
  - step 6 wall time `1059.5995318000205s`, fluid substeps `54`

Interpretation: the stop ceiling is instrumented correctly and does suppress high-residual early stops, but continuing those high-residual iterations did not recover the trajectory. It increased cost and still failed the same step-6 divergence guard. High-residual stop suppression is diagnostic plumbing, not the primary 2s route.

Early residual-triggered adaptive-budget mixed probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_adaptiter12_resgt0005_cflgt010_backtrack05_adaptive010_safety2_reuse_6step_20260618_001`

- command kept base `--fsi-coupling-iterations 6` and added:
  - `--fsi-coupling-adaptive-iterations-max 12`
  - `--fsi-coupling-adaptive-iterations-residual-threshold-n 0.0005`
  - `--fsi-coupling-adaptive-iterations-cfl-threshold 0.1`
- completed all 6 requested explicit rows, but `completed_step_checks_passed=False`
- crossed the old step-6 pressure-projection divergence guard:
  - `max_interior_divergence_l2=0.08916245938685781`
  - `max_cfl=0.2181534686258861`
  - step 6 requested `12`, used `9`, residual `8.45058524197137`
  - step 6 `interior_divergence_l2=0.08916245938685781`
- still not a valid readiness result:
  - `fsi_coupling_not_converged_count=5`
  - `max_fsi_coupling_residual_norm_n=8.45058524197137`
  - `checks.fsi_coupling_converged=False`
  - `checks.fsi_physical_interface_map_stable=False`
  - `diagnostic_checks.final_outlet_to_fsi_volume_source_ratio_physical=False`
  - `diagnostic_checks.boundary_drive_has_no_prescribed_driver=False`

Interpretation: simply raising the budget after step 4/5 indicators is too late, and high-residual stop2 suppression alone is not enough. A very early residual trigger (`0.0005N`) can reproduce the fixed-12 stability direction through step 6, which confirms that the accepted-state branch must be improved early. However, the mixed probe still fails completed-step checks because the force fixed point and flow diagnostics remain unphysical. The next controller must be step-internal quality aware, not just previous-row threshold based; likely candidates are residual-slope-based continuation with a convergence target, a two-stage rerun of the same step when accepted residual remains high, or a better predictor for the accepted interface force before the step enters the pressure projection.

Same-step rerun late-threshold negative probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_samestep12_resgt05_backtrack05_adaptive010_safety2_reuse_6step_20260618_002`

- command added:
  - `--fsi-coupling-same-step-rerun-iterations-max 12`
  - `--fsi-coupling-same-step-rerun-residual-threshold-n 0.5`
- failed at step 6:
  - `error=step 6 numerical guard failed: interior_divergence_l2=1.809437e-01 > 1.000000e-01`
  - step 4 triggered rerun: first residual `0.9250372361696464`, final residual `0.9250337638898329`
  - step 5 triggered rerun: first residual `3.3777422741657763`, final residual `3.377742965840757`
  - step 6 triggered rerun: first residual `88.55348769850674`, final residual `10.704072170619234`
  - step 6 `cfl=0.07551231043679373`
  - step 6 `interior_divergence_l2=0.18094373506461597`
  - step 6 wall time `1313.1393224000058s`

Interpretation: same-step rerun is wired correctly, but a `0.5N` threshold is still too late and expensive. It does not change the bad branch at step 4/5, and step 6 fails like the late previous-row trigger and stop-suppression probes.

Same-step rerun early-threshold mixed probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_samestep12_resgt0005_backtrack05_adaptive010_safety2_reuse_6step_20260618_001`

- command added:
  - `--fsi-coupling-same-step-rerun-iterations-max 12`
  - `--fsi-coupling-same-step-rerun-residual-threshold-n 0.0005`
- completed all 6 requested explicit rows, but `completed_step_checks_passed=False`
- crossed the old step-6 pressure-projection divergence guard:
  - `max_interior_divergence_l2=0.08922119723600104`
  - `max_cfl=0.2181969561747142`
  - step 6 first residual `8.451990349825907`, final residual `8.452008385567922`
  - step 6 `interior_divergence_l2=0.08922119723600104`
- useful branch evidence:
  - total same-step rerun triggers `5`
  - step 2 first residual `0.0030608262221115145`, final residual `0.001030337496026299`
  - step 5 first residual `3.617808438506736`, final residual `1.7757192056477087`
- still not a valid readiness result:
  - `fsi_coupling_not_converged_count=5`
  - `max_fsi_coupling_residual_norm_n=8.452008385567922`
  - `checks.fsi_coupling_converged=False`
  - `checks.fsi_physical_interface_map_stable=False`
  - `diagnostic_checks.final_outlet_to_fsi_volume_source_ratio_physical=False`

Interpretation: same-step rerun proves the branch-timing diagnosis. If the higher budget is applied from step 2, the route crosses the old step-6 divergence guard; if the trigger waits until `0.5N`, it does not. The downside is cost: this mode pays for a failed low-budget attempt before the high-budget solve, so it is a diagnostic bridge, not the final 2s strategy. The next implementation target should convert this finding into a cheaper pre-step budget predictor or a residual-quality continuation that improves convergence rather than simply duplicating work.

Residual-quality continuation mixed probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_cont6_resgt0005_backtrack05_adaptive010_safety2_reuse_6step_20260618_001`

- command added:
  - `--fsi-coupling-residual-continuation-iterations-max 6`
  - `--fsi-coupling-residual-continuation-threshold-n 0.0005`
- base requested budget remained `--fsi-coupling-iterations 6`
- completed all 6 requested explicit rows, but `completed_step_checks_passed=False`
- crossed the old step-6 pressure-projection divergence guard:
  - `max_interior_divergence_l2=0.08919535608276821`
  - `max_cfl=0.2181645831891469`
  - step 6 requested `6`, used `9`, continuation count `3`
  - step 6 residual `8.451413140045414`
  - step 6 `interior_divergence_l2=0.08919535608276821`
- useful cost/branch evidence:
  - total continuation iterations `27`
  - step 2 requested `6`, used `12`, residual `0.0010303552508775806`, wall `20.072870899981353s`
  - step 4 requested `6`, used `12`, residual `2.1535801437852147`, wall `160.6707391999953s`
  - step 5 requested `6`, used `12`, residual `1.7755194132977428`, wall `160.83719419999397s`
  - step 6 wall `530.9919811s`, lower than same-step rerun step 6 `844.268650300015s` and close to fixed 12 step 6 `542.8900764000136s`
- still not a valid readiness result:
  - `fsi_coupling_not_converged_count=5`
  - `max_fsi_coupling_residual_norm_n=8.451413140045414`
  - `checks.fsi_coupling_converged=False`
  - `checks.fsi_physical_interface_map_stable=False`
  - `diagnostic_checks.final_outlet_to_fsi_volume_source_ratio_physical=False`

Interpretation: residual-quality continuation is a better implementation of the early-budget finding than same-step rerun. It preserves the successful branch timing and avoids duplicate physical-step reruns. It still does not solve strong-coupling convergence: residuals remain high and the completed-step checks fail. The next target should improve convergence quality within the continued solve, likely by changing the update model after the continuation gate trips rather than simply adding more Aitken-limited trust-region iterations.

Residual-quality continuation plus high-residual rebound-stop suppression:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_cont6_resgt0005_stopmaxres05_backtrack05_adaptive010_safety2_reuse_6step_20260618_001`

- command added, relative to the residual-quality continuation probe:
  - `--fsi-coupling-trust-region-rebound-stop-max-residual-n 0.5`
- completed all 6 requested explicit rows, but `completed_step_checks_passed=False`
- diagnostic result:
  - step 6 requested `6`, used `12`, continuation count `6`
  - step 6 rebound-stop count `0`, suppressed count `4`
  - step 6 residual `8.450990099128687`
  - step 6 `interior_divergence_l2=0.08924468895152811`
  - step 6 wall `696.1441169999889s`
- comparison against the same continuation probe without the stop ceiling:
  - without stop ceiling, step 6 used `9`, continuation count `3`, residual `8.451413140045414`, `interior_divergence_l2=0.08919535608276821`, wall `530.9919811s`
  - with `0.5N` stop ceiling, the solve spends three more fixed-point iterations and suppresses four rebound-stop events, but the residual improvement is only `0.000423N` while divergence is slightly worse

Interpretation: the step-6 residual plateau is not primarily caused by rebound-stop early termination. Suppressing high-residual rebound-stop inside the residual-quality continuation branch converts the run into a more expensive fixed-12-style solve without a meaningful residual or divergence improvement. The next implementation should change the accepted-force update direction or local response model after the continuation gate trips; it should not keep spending effort on stop2 ceiling variants.

Residual-quality continuation with rebound secant-from-best:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_cont6_resgt0005_secantbest_backtrack05_adaptive010_safety2_reuse_6step_20260618_001`

- command added, relative to the residual-quality continuation probe:
  - `--fsi-coupling-residual-continuation-rebound-secant-from-best`
- implementation:
  - when residual continuation is active and a continuation trial rebounds away from the best accepted trial, compute a diagonal secant Newton-like force update from the best accepted force using the measured trial/residual history
  - limit the proposed secant update by the existing force-increment trust region
  - record `fsi_coupling_residual_continuation_rebound_secant_count` in rows and summary
- 6-step result:
  - completed all 6 requested explicit rows, but `completed_step_checks_passed=False`
  - step 6 requested `6`, used `12`, continuation count `6`
  - step 6 secant count `1`
  - step 6 residual `1.5511584433995331`
  - step 6 `interior_divergence_l2=0.0885088299023929`
  - step 6 `projected_ibm_residual_mps=0.16026180982589722`
  - step 6 wall `689.5849174999748s`
- comparison against residual-quality continuation without secant:
  - without secant, step 6 used `9`, continuation count `3`, residual `8.451413140045414`, `interior_divergence_l2=0.08919535608276821`, wall `530.9919811s`
  - with secant, step 6 used `12`, continuation count `6`, residual `1.5511584433995331`, `interior_divergence_l2=0.0885088299023929`, wall `689.5849174999748s`
  - step 6 residual therefore dropped by about `6.900255N` relative to the best continuation-only artifact
- 7-step checkpoint resume:
  - same output directory was resumed from checkpoint with `--steps 7`
  - row-level history contains 7 rows after resume
  - step 7 requested `6`, used `12`, continuation count `6`
  - step 7 secant count `1`
  - step 7 residual `10.220223352082508`
  - step 7 `interior_divergence_l2=1.3618989082978434e-06`
  - step 7 `projected_ibm_residual_mps=0.479605108499527`
  - step 7 wall `864.6723265000037s`
- comparison against the older fixed `2N + stop2 + 12 iterations` 7-step artifact:
  - older step 7 residual was `12.854N`
  - secant continuation step 7 residual is lower at `10.220223352082508N`, but still not converged
- reporting fix:
  - resume-loaded CSV booleans such as `"False"` were being counted with Python `bool("False")`, which made `summary.json` undercount FSI non-converged rows after resume
  - added `count_enabled_unconverged_fsi_rows` so future resume summaries use `_row_bool`
  - for this resumed artifact, row-level `history.csv` is the authoritative convergence evidence; each row's `fsi_coupling_converged` field shows steps 2-7 are still not fixed-point converged

Interpretation: rebound secant-from-best is the first update-model change in this goal that materially improves the high-residual step-6 branch instead of only lowering CFL or increasing iteration count. It is not 2s-ready: step 7 remains high residual, projected-IBM residual grows, and completed checks still fail. The route is nevertheless better defined: keep residual-quality continuation, keep the secant-from-best rebound update, and next target the remaining high-residual steps with a stronger multi-component response model rather than scalar thresholds.

Residual-quality continuation with explicit secant trigger factor and evaluation extension:

Implementation additions:

- `--fsi-coupling-residual-continuation-rebound-secant-factor`
  - default `inf` preserves the previous behavior by inheriting `--fsi-coupling-trust-region-rebound-stop-factor`
  - finite values let the secant-from-best update trigger before the stop2 rebound threshold
- `--fsi-coupling-residual-continuation-rebound-secant-evaluation-extensions-max`
  - default `0` preserves the strict configured continuation budget
  - finite values reserve extra same-step evaluations only when a secant candidate is generated at the end of the continuation budget
- rows and summary now record `fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count`

Extension-only probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_cont6_resgt0005_secantbest_evalext1_backtrack05_adaptive010_safety2_reuse_7step_20260618_001`

- command added, relative to secant-from-best:
  - `--fsi-coupling-residual-continuation-rebound-secant-evaluation-extensions-max 1`
- completed 7 rows, but `completed_step_checks_passed=False`
- step 6 residual `1.5266929534173712`, `interior_divergence_l2=0.08798510581805695`
- step 7 residual `10.855299787541638`, `interior_divergence_l2=1.4440045993605816e-06`
- `total_fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count=0`

Interpretation: extension-only is not the missing step-7 mechanism in this trajectory. It did not trigger, because step 7 did not cross the inherited stop2 rebound factor.

Aggressive factor-1.0 probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_cont6_resgt0005_secantbest_factor1_evalext1_backtrack05_adaptive010_safety2_reuse_7step_20260618_001`

- command added:
  - `--fsi-coupling-residual-continuation-rebound-secant-factor 1.0`
  - `--fsi-coupling-residual-continuation-rebound-secant-evaluation-extensions-max 1`
- failed at step 6:
  - `run_process.json` status `failed`
  - error `step 6 numerical guard failed: interior_divergence_l2=1.595311e-01 > 1.000000e-01`
- row-level result before failure:
  - step 3 used `13`, secant count `4`, extension count `1`, residual `0.0002494963560728531`
  - step 6 used `13`, secant count `3`, extension count `1`, residual `0.007022724101164457`
  - step 6 `interior_divergence_l2=0.15953109068398108`

Interpretation: triggering secant on every residual growth is too aggressive. It can drive force residual very low, but it breaks the pressure-projection divergence guard by step 6.

Moderate factor-1.5 probe:

`_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_cont6_resgt0005_secantbest_factor15_evalext1_backtrack05_adaptive010_safety2_reuse_7step_20260618_001`

- command added:
  - `--fsi-coupling-residual-continuation-rebound-secant-factor 1.5`
  - `--fsi-coupling-residual-continuation-rebound-secant-evaluation-extensions-max 1`
- completed 7 rows, then failed when resumed to step 8
- 7-step row-level result:
  - step 5 used `12`, secant count `1`, residual `0.034589908335776236`, `interior_divergence_l2=6.021591728847377e-07`
  - step 6 used `6`, secant count `0`, residual `3.882117693551039`, `interior_divergence_l2=0.08428825086107923`
  - step 7 used `12`, secant count `1`, residual `1.0590821778738138`, `interior_divergence_l2=1.4743874873185458e-06`
  - 7-step max residual `3.882117693551039`
  - 7-step max `interior_divergence_l2=0.08428825086107923`
- comparison:
  - inherited factor2 secant route had step 7 residual `10.220223352082508`
  - factor1.5 route reduced step 7 residual to `1.0590821778738138`
  - factor1.5 route still did not fixed-point converge rows 2-7
- step 8 resume failure:
  - `run_process.json` status `failed`
  - error `step 8 numerical guard failed: interior_divergence_l2=8.693900e-01 > 1.000000e-01`
  - step 8 row residual `340.50883256655555`
  - step 8 row `projected_ibm_residual_mps=1.3118536472320557`

Interpretation: factor1.5 is the best 7-step residual artifact so far and a real update-model improvement, but it is not an 8-step or 2s route. The next target is not a lower secant factor; it is a safety/acceptance model that prevents the step-8 divergence while preserving the factor1.5 residual gains through step 7.

### Trial-divergence gate and stricter adaptive-CFL probes

Hard trial-divergence gate on the factor1.5 secant scaffold:

- artifact: `_codex_validation/codex_goal_surface_seed_zmin_connect_robin_physical100_trustinc2_stop2_cont6_resgt0005_secantbest_factor15_trialdiv010_evalext1_backtrack05_adaptive010_safety2_reuse_8step_20260618_001`
- command delta from the factor1.5 scaffold: `--fsi-coupling-trial-interior-divergence-tolerance 0.1`
- result: failed at step 5, earlier than the ungated factor1.5 step-8 failure
- step 4: `interior_divergence_l2=0.04587170619095236`, `fsi_coupling_residual_norm_n=0.058207719191929595`, `fsi_coupling_rejected_trial_count=2`, `fsi_coupling_trial_interior_divergence_l2_max=0.28606202795709634`
- step 5: `interior_divergence_l2=0.12837298971335262`, `fsi_coupling_residual_norm_n=inf`, `fsi_coupling_rejected_trial_count=12`, `fsi_coupling_accepted_trial_interior_divergence_l2=nan`

Interpretation: tying acceptance to trial pressure-projection divergence is directionally correct because it rejects bad step-4 trial states, but a hard `0.1` gate alone can produce an all-rejected step and fall back to an unusable force/state. This is not the final safety model.

Stricter adaptive-CFL probe on the same scaffold:

- artifact: `_codex_validation/codex_goal_factor15_trialdiv010_targetcfl005_max128_5step_20260618_001`
- command delta: `--adaptive-fluid-substeps-target-cfl 0.05 --adaptive-fluid-substeps-max 128` with the same `--fsi-coupling-trial-interior-divergence-tolerance 0.1`
- 5-step result: finished 5/5; step 5 crossed the previous step-5 guard with `fluid_substeps=23`, `cfl=0.0583643759259526`, `interior_divergence_l2=4.112202926295898e-07`, and `fsi_coupling_rejected_trial_count=2`
- 6-step resume result: finished 6/6; step 6 used `fluid_substeps=54`, `cfl=0.03675459699025229`, `interior_divergence_l2=0.0884467991714194`, and `fsi_coupling_rejected_trial_count=8`
- residual regression: step 5 residual rose to `2.2100754878469546N`, and step 6 residual rose to `87.5176895355552N`

Interpretation: smaller adaptive CFL can keep the connected route below the divergence guard through step 6, but only by making the solve very expensive and allowing the force fixed-point residual to explode. This is useful negative evidence: the next route should not be "lower target CFL" alone. It should preserve the factor1.5 residual gains while using divergence-aware acceptance/backtracking only as a safety guard, with diagnostics that expose whether trials are rejected by CFL, divergence, or residual quality.

## Verification

Focused tests:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_carves_minimal_barrier `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_respects_carve_limit `
  tests.test_core_fluid.CoreCartesianFluidSolverTests.test_pressure_outlet_report_splits_reachable_and_unreached_source_flux `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_pressure_solver_auto_uses_fv_cg_for_graded_grid `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_sharp_case_row_uses_hibm_marker_fields_not_projected_ibm `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_fsi_coupling `
  tests.test_time_stepping -v
```

Result:

`Ran 50 tests ... OK`

Latest focused verification after the Robin probes:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_carves_minimal_barrier `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_respects_carve_limit `
  tests.test_core_fluid.CoreCartesianFluidSolverTests.test_pressure_outlet_report_splits_reachable_and_unreached_source_flux `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_pressure_solver_auto_uses_fv_cg_for_graded_grid `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_sharp_case_row_uses_hibm_marker_fields_not_projected_ibm `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_impedance_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_target_mode_selects_target_force `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_target_mode_can_be_selected_explicitly `
  tests.test_fsi_coupling `
  tests.test_time_stepping -v
```

Result:

`Ran 53 tests in 9.247s OK`

Latest focused verification after the Aitken bounds control:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 45 tests in 0.008s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_aitken_lower_bound_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_aitken_upper_bound_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_impedance_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_target_mode_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_pressure_solver_auto_uses_fv_cg_for_graded_grid `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_carves_minimal_barrier `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_respects_carve_limit `
  tests.test_core_fluid.CoreCartesianFluidSolverTests.test_pressure_outlet_report_splits_reachable_and_unreached_source_flux `
  tests.test_time_stepping -v
```

Result:

`Ran 12 tests in 8.793s OK`

Latest focused verification after rejected-trial backtracking and all-rejected failsafe:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 48 tests in 0.009s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_target_map_relaxation_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_rejected_trial_backtrack_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_aitken_lower_bound_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_aitken_upper_bound_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_impedance_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_target_mode_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_pressure_solver_auto_uses_fv_cg_for_graded_grid `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_carves_minimal_barrier `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_respects_carve_limit `
  tests.test_core_fluid.CoreCartesianFluidSolverTests.test_pressure_outlet_report_splits_reachable_and_unreached_source_flux `
  tests.test_time_stepping -v
```

Result:

`Ran 14 tests in 9.641s OK`

Latest focused verification after residual-growth gate:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 50 tests in 0.009s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_target_map_relaxation_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_rejected_trial_backtrack_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_residual_growth_rejection_factor_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_aitken_lower_bound_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_aitken_upper_bound_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_impedance_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_interface_reaction_robin_target_mode_can_be_selected_explicitly `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_pressure_solver_auto_uses_fv_cg_for_graded_grid `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_carves_minimal_barrier `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_connect_surface_seed_components_to_zmin_respects_carve_limit `
  tests.test_core_fluid.CoreCartesianFluidSolverTests.test_pressure_outlet_report_splits_reachable_and_unreached_source_flux `
  tests.test_time_stepping -v
```

Result:

`Ran 15 tests in 8.415s OK`

Latest focused verification after absolute residual trust gate:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 51 tests in 0.010s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_max_accepted_residual_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 3 tests in 0.006s OK`

Latest focused verification after force-increment trust region:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 52 tests in 0.007s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_max_accepted_residual_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_trust_region_force_increment_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 4 tests in 0.009s OK`

Latest focused verification after adaptive trust-radius plumbing:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 53 tests in 0.007s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_trust_region_force_increment_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_adaptive_trust_region_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 4 tests in 0.009s OK`

Latest focused verification after residual-rebound trust backtrack:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 54 tests in 0.007s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_trust_region_force_increment_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_adaptive_trust_region_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_trust_region_rebound_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 5 tests in 0.011s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  simulation_core/fsi_coupling.py `
  cases/squid_soft_robot.py `
  tests/test_fsi_coupling.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Compile check:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  simulation_core/fluid.py `
  simulation_core/fsi_coupling.py `
  cases/squid_soft_robot.py `
  tests/test_core_fluid.py `
  tests/test_fsi_coupling.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Latest focused verification after best-trial rebound stop:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 55 tests in 0.008s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_trust_region_rebound_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 3 tests in 0.006s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  simulation_core/fsi_coupling.py `
  cases/squid_soft_robot.py `
  tests/test_fsi_coupling.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Latest focused verification after residual/CFL-triggered adaptive iteration budget:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_adaptive_iterations_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 3 tests in 0.006s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  cases/squid_soft_robot.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Latest focused verification after high-residual rebound-stop ceiling:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 56 tests in 0.008s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_trust_region_rebound_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 3 tests in 0.007s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  simulation_core/fsi_coupling.py `
  cases/squid_soft_robot.py `
  tests/test_fsi_coupling.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Latest focused verification after same-step residual-triggered rerun:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_same_step_rerun_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_same_step_rerun_triggers_only_for_unconverged_high_residual `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 4 tests in 0.006s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  cases/squid_soft_robot.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Latest focused verification after residual-quality fixed-point continuation:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_fsi_coupling.InterfaceReactionFixedPointTests.test_residual_continuation_extends_only_until_quality_threshold `
  tests.test_fsi_coupling.InterfaceReactionFixedPointTests.test_fixed_point_rejects_nonfinite_scalar_controls `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_residual_continuation_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 5 tests in 0.008s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  simulation_core/fsi_coupling.py `
  cases/squid_soft_robot.py `
  tests/test_fsi_coupling.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Latest focused verification after residual-quality stop-ceiling negative probe:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 57 tests in 0.008s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_residual_continuation_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_same_step_rerun_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_same_step_rerun_triggers_only_for_unconverged_high_residual `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 5 tests in 0.009s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  simulation_core/fsi_coupling.py `
  cases/squid_soft_robot.py `
  tests/test_fsi_coupling.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Latest focused verification after residual-continuation secant factor/extension and resume boolean reporting:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest tests.test_fsi_coupling -v
```

Result:

`Ran 60 tests in 0.011s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m unittest `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_not_converged_count_parses_resume_csv_booleans `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_residual_continuation_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_coupling_same_step_rerun_cli_is_explicit `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_fsi_same_step_rerun_triggers_only_for_unconverged_high_residual `
  tests.test_squid_latest_core_config.SquidLatestCoreConfigTests.test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy -v
```

Result:

`Ran 6 tests in 0.009s OK`

```powershell
& 'D:/TOOL/Anaconda/python.exe' -m py_compile `
  simulation_core/fsi_coupling.py `
  cases/squid_soft_robot.py `
  tests/test_fsi_coupling.py `
  tests/test_squid_latest_core_config.py
```

Result: OK.

Whitespace check:

```powershell
git diff --check
```

Result: OK, with CRLF normalization warnings only.

## Current Conclusion

The full 2s goal is not complete. The first executable route has been narrowed to a physically checkable but not yet production-ready scaffold:

For a 2s squid simulation, the next step should be:

1. keep `legacy_projected_reduced + auto -> fv_cg + fsi_coupling_iterations=6` as the baseline runner;
2. keep the surface-seed-to-zmin connection repair explicit and audited; it carves only 2 cells in the current CAD mask and makes all source reachable;
3. keep `Z=100 N*s/m + physical Robin target + rejected-trial backtracking` as the current non-pinning coupled-update scaffold;
4. use conservative adaptive fluid substeps only as a diagnostic CFL guard, not as the final 2s strategy, because step 5 already required `fluid_substeps=22`;
5. enable `--reuse-accepted-fsi-trial-state` for future short probes to avoid re-advancing accepted current trials, but treat it only as a cost optimization;
6. do not switch the primary route to IQN-ILS alone; the matched 5-step A/B lowered CFL/cost but increased the max residual from `10.05155920462909` to `23.673683368019653`;
7. do not treat the intra-step residual-growth gate with factor `2.0` as a runtime improvement; it did not trigger in the 5-step scaffold because the large-residual trials were already rejected by the CFL/stability predicate or were still intra-step improvements;
8. do not treat a hard absolute residual cap as the next primary route; `1N` and `5N` caps both made step 4 and step 5 all-rejected failsafe steps with `inf` residual;
9. keep fixed force-increment trust-region limiting as the current primary residual-aware scaffold; fixed `2N` reduced the 5-step max force residual from `10.051559N` to `3.377683N` without hard all-rejected steps, but still left 4 of 5 steps not converged;
10. do not use fixed 10-iteration `trustinc2` as the next primary route; it improved early residuals but worsened step 4 to `1.975554N` and was manually stopped before step 5 completed;
11. do not use the default adaptive trust-radius policy as the next primary route; it remained CFL-stable but worsened the best 5-step residual from fixed `2N` `3.377683N` to `3.848291N`;
12. do not use residual-rebound trust backtracking with factor `2.0` / backtrack `0.5` as the next primary route; it triggered once but slightly worsened the best 5-step residual from fixed `2N` `3.377683N` to `3.379246N`;
13. best-trial rebound stop with factor `2.0` is useful as a short-step cost/accept-stop improvement; it reduced the fixed `2N` 5-step mean wall time from `106.822s` to `79.621s` and slightly lowered max residual from `3.377683N` to `3.377442N`;
14. do not treat stop2 as 10-step-ready: fixed `2N + stop2` failed at step 6 with `interior_divergence_l2=0.3100108` and residual `88.558N`;
15. do not treat smaller fixed radius as the next answer: fixed `1N + stop2` lowered step-6 CFL/divergence but still failed at step 6 with `interior_divergence_l2=0.1401305` and residual `154.118N`;
16. fixed `2N + stop2 + 12 iterations` is a positive stability-direction probe: it completed 7/7 steps and crossed the old step-6 divergence guard with `interior_divergence_l2=0.0892128`, but residuals stayed high (`8.449N` at step 6, `12.854N` at step 7) and cost reached `809.98s` for step 7;
17. do not treat late previous-row adaptive iteration triggers as the answer: residual-threshold `1.0N` failed step 6 at `interior_divergence_l2=0.1809297`, and residual `0.5N` plus CFL `0.1` also failed step 6 at `0.1812112`;
18. do not treat high-residual stop2 suppression as the answer: `--fsi-coupling-trust-region-rebound-stop-max-residual-n 1.0` correctly suppressed high-residual stop2 events but still failed step 6 at `interior_divergence_l2=0.1809502` with residual `10.705N`;
19. early residual-triggered adaptive budget is a useful branch-timing result: `residual-threshold=0.0005N` crossed the old step-6 divergence guard with `interior_divergence_l2=0.0891625`, but `completed_step_checks_passed=False` because 5 of 6 steps were not fixed-point converged and flow diagnostics remained unphysical;
20. same-step rerun confirms the branch-timing diagnosis: `0.5N` triggers too late and still fails step 6 at `interior_divergence_l2=0.1809437`, while `0.0005N` triggers from step 2 and crosses step 6 with `interior_divergence_l2=0.0892212`;
21. same-step rerun is not the final strategy because it duplicates cost: the `0.0005N` run triggered 5 reruns in 6 steps and still left `fsi_coupling_not_converged_count=5`;
22. residual-quality continuation is the better early-budget implementation: it keeps base requested iterations at `6`, appends 27 total continuation iterations only after the residual gate, crosses step 6 with `interior_divergence_l2=0.0891954`, and avoids same-step rerun's duplicated physical-step work;
23. do not treat residual-quality continuation plus high-residual rebound-stop suppression as the answer: `--fsi-coupling-trust-region-rebound-stop-max-residual-n 0.5` forced step 6 to use all 12 iterations and suppressed 4 rebound-stop events, but residual only changed from `8.451413N` to `8.450990N` while wall time rose from `530.99s` to `696.14s`;
24. the useful branch-timing finding is that the accepted-state branch must be improved by step 2, not step 4 or step 5. Fixed 12, previous-row `0.0005N`, same-step `0.0005N`, and residual continuation `0.0005N` all change the later trajectory; `0.5N` thresholds and high-residual stop suppression do not;
25. residual-quality continuation plus rebound secant-from-best is the first useful short-step update-model scaffold: it keeps the early continuation branch, triggers one secant reset at step 6, drops step-6 residual from `8.451413N` to `1.551158N`, and keeps `interior_divergence_l2` under the `0.1` guard at `0.0885088`;
26. secant evaluation extension alone is not the missing step-7 mechanism: `--fsi-coupling-residual-continuation-rebound-secant-evaluation-extensions-max 1` did not trigger in the inherited factor2 route, and step-7 residual remained high at `10.855300N`;
27. do not use secant factor `1.0` as the next route: it drove residuals very low by step 6 (`0.007023N`) but failed the divergence guard at step 6 with `interior_divergence_l2=0.1595311`;
28. secant factor `1.5` is the best 7-step residual artifact so far: step 5 residual `0.034590N`, step 7 residual `1.059082N`, and 7-step max divergence `0.0842883`; however it still leaves rows 2-7 not fixed-point converged and fails when resumed to step 8 with residual `340.508833N` and `interior_divergence_l2=0.869390`;
29. the next implementation target should keep the factor1.5 residual gains but add a safety/acceptance model that prevents the step-8 divergence, likely by tying secant acceptance to pressure-projection/divergence response rather than force residual alone;
30. only after 10+ connected steps are both CFL-stable and fixed-point-converged should the validation ladder extend to 100, 500, 1000, and finally the full 2s waveform.

Do not spend the next iteration on more CFL-only diffuse tuning, target-map scalar relaxation, IQN-ILS alone, fixed substep counts, high fixed-point iteration counts without an accept/stop policy, plain time-step halving, increasing scalar Robin impedance, lowering the Aitken floor, clipping the Aitken upper bound, hard absolute-residual rejection alone, the default previous-trial adaptive trust-radius policy, residual-rebound backtracking at factor `2.0`, simple scalar reduction from `2N` to `1N`, late previous-row adaptive iteration triggers alone, high-residual stop2 suppression alone, residual-quality continuation plus high-residual stop2 suppression, secant factor `1.0`, or same-step rerun with a late `0.5N` threshold. The current blocker has moved from "projected-IBM blows up at step 2" to "connected outlet flow can be made short-step CFL-stable through step 7 with factor1.5 secant-from-best and much lower residuals, but the route still fails at step 8 and is not fixed-point converged." Residual-quality continuation plus rebound secant-from-best with factor `1.5` is now the best 7-step residual scaffold; fixed `2N + stop2` remains the cheaper 5-step scaffold. None is full waveform proof.
