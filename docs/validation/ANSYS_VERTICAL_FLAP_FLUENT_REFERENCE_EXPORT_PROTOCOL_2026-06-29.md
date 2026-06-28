# ANSYS Vertical-Flap Fluent Reference Export Protocol

## Scope

This protocol defines how to collect provenance-backed ANSYS Fluent reference exports for the vertical-flap FSI case. It is a source-export protocol only. It does not run EasyFsi, does not run HIBM-MPM, does not tune tolerances, and does not claim Fluent parity.

The public ANSYS tutorial URL is tracked only as metadata evidence:

- https://ansyshelp.ansys.com/public/views/secured/corp/v251/en/flu_tg/flu_tg_fsi_2way.html

The public tutorial may be used to cross-check case identity, geometry, boundary conditions, and export setup. It must not be used as numeric parity truth.

## Required Run Setup

- Case: `ansys_vertical_flap_fsi`
- Step count: `50`
- Time step: `0.0005`
- Final time: `0.025`
- Reference contract schema: `ansys_vertical_flap_fluent_reference_contract_v1`
- Active manifest schema: `active_fluent_reference_contract_manifest_v1`

The Fluent run must be generated independently in ANSYS Fluent from the documented vertical-flap FSI setup. Repository solver outputs are comparison candidates only and must not be copied into the Fluent reference export files.

## Required Source Exports

Write the exports under:

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/`

Required CSV files and headers:

- `fluent_tip_displacement_history.csv`
  - `step,time_s,tip_displacement_x_m,tip_displacement_y_m,tip_displacement_z_m,tip_displacement_norm_m,max_displacement_m,source`
- `fluent_force_history.csv`
  - `step,time_s,force_x_N,force_y_N,force_z_N,primary_force_z_N,secondary_force_z_N,source`
- `fluent_flow_balance_history.csv`
  - `step,time_s,inlet_flow_rate_m3s,outlet_flow_rate_m3s,pressure_outlet_flux_m3s,velocity_outlet_flux_m3s,source`
- `fluent_pressure_summary_history.csv`
  - `step,time_s,pressure_min_pa,pressure_max_pa,pressure_range_pa,source`
- `fluent_metadata_2026-06-28.md`
  - Include source document, Fluent run id, export author, export date, Fluent version, mesh/domain source, geometry units, material model, boundary conditions, time step, number of steps, coupling settings, export procedure, force sign convention, flow sign convention, pressure reference, and displacement definition.

Every data CSV must include a row with `step=50`, `time_s=0.025`, non-empty `source`, and numeric reference values for the columns used by the schema validator.

## Validation Commands

After adding real Fluent exports, run:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py
python validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py
python -m unittest tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts -v
python -m unittest tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema -v
python -m unittest tests.integration.test_ansys_vertical_flap_fluent_source_export_schema -v
python -m unittest tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts -v
python -m unittest tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic -v
```

Promotion is allowed only when the collection validator reports `fluent_reference_complete`, the active manifest SHA checks pass, and the parity runner reports all gates passed from schema-validated Fluent reference metrics.
