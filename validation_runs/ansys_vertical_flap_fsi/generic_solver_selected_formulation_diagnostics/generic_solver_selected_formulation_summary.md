# ANSYS vertical-flap generic solver selected formulation

## Scope

This artifact invokes the generic FSI solver boundary for the ANSYS vertical-flap selected formulation. It is EasyFsi generic solver validation and does not claim Fluent parity.

## Candidate decision

- candidate_status: `generic_solver_selected_formulation_step50_passed`
- completed_step_count: `50`
- pressure_pair_mode: `runtime_anchored_cell_pair`
- pressure_pair_runtime_generation_status: `runtime_generated`
- pressure_pair_runtime_generation_complete: `True`

## Gates

- invalid_marker_count_max: `0.0`
- sample_pair_fallback_count_max: `0.0`
- one_sided_marker_count_min: `24.0`
- force_action_reaction_residual_max_n: `0.0`

## Non-claims

- Does not claim Fluent parity.
- Does not complete Fluent reference exports.
- Pressure-pair cells are generated from runtime marker geometry through the generic pressure sample pair contract.
- Tip displacement CSV columns are mapped from the runtime `tip_mean_displacement_m` vector; `max_displacement_m` remains the whole-field displacement envelope.

## Files

- tip_displacement: `validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/easyfsi_tip_displacement_history.csv`
- force: `validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/easyfsi_force_history.csv`
- flow_balance: `validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/easyfsi_flow_balance_history.csv`
- pressure_summary: `validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/easyfsi_pressure_summary_history.csv`
- pressure_sample_pair_map: `validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/pressure_sample_pair_map.json`
