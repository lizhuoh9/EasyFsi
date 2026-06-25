# ANSYS Vertical-Flap Source Candidate STEP20 Diagnostics

best_candidate = source_0p75_ramp5_step20 (compatibility diagnostic fallback when no promotion candidate exists)
best_final_gate_candidate = source_0p75_ramp5_step20
best_temporal_candidate = none
best_flow_temporal_candidate = source_0p80_ramp2_step20
best_combined_temporal_candidate = none
promotion_candidate = none
promotion_candidate_status = no_promotion_candidate
diagnostic_fallback_candidate = source_0p75_ramp5_step20
nearest_non_candidate = source_0p70_constant_step20
candidate_status = no_temporal_candidate
temporal_candidate_status = no_temporal_candidate
temporal_best_candidate_status = none
temporal_candidate_count = 0
flow_temporal_candidate_count = 4
best_candidate_history_csv = validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/best_candidate_step20_history.csv
best_final_gate_candidate_history_csv = validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/best_final_gate_candidate_history.csv
best_flow_temporal_candidate_history_csv = validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/best_flow_temporal_candidate_history.csv
best_combined_temporal_candidate_history_csv = validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/best_combined_temporal_candidate_history.csv
mass_balance_primary_metric = velocity_outlet_flux_ratio
pressure_outlet_flux_interpretation = diagnostic_only_until_pressure_outlet_model_reviewed
primary_observation = best_candidate=source_0p75_ramp5_step20; p999 range 11.175716391563588-25.300280584335876 m/s; peak range 12.719998359680176-32.88761520385742 m/s; velocity_outlet_flux_ratio range 0.0-1.0982373626194526
current_best_hypothesis = source/outlet flow can satisfy the STEP20 flow temporal gate, but coupled force/tip settling still blocks promotion
next_action = stop before 50-step; run fixed-solid STEP30 flow diagnostics and coupling-settling review

## STEP20 Matrix

| scenario | final status | temporal status | strength | profile | ramp | flow temporal | coupling settling | promotion | peak m/s | p999 m/s | max peak m/s | flow last-5 min p999 | flow last-5 outlet mean | force z N | tip dz m |
|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| projection_only_step20_baseline | below_p999_gate | temporal_failed | 1.0 | constant | 0 | flow_temporal_failed | coupling_settled_late | not_promotion_candidate | 12.719998359680176 | 11.175716391563588 | 31.764034271240234 | 11.175716391563588 | 0.0 | -0.0003492466824987069 | -9.343959391117096e-06 |
| diagnostic_reinitialize_step20_upper_bound | diagnostic_excluded | temporal_not_applicable | 1.0 | constant | 0 | flow_temporal_not_applicable | coupling_not_applicable | promotion_not_applicable | 29.445350646972656 | 22.984101106643706 | 29.445518493652344 |  |  | -0.002380845137920346 | -2.7056783437728882e-05 |
| source_0p70_constant_step20 | below_p999_gate | temporal_failed | 0.7 | constant | 0 | flow_temporal_failed | coupling_settled_late | not_promotion_candidate | 25.47308921813965 | 19.53290603828466 | 35.024253845214844 | 18.45301177978546 | 0.8051835684609234 | -6.954605826858572e-05 | -1.7741695046424866e-06 |
| source_0p75_constant_step20 | candidate | temporal_failed | 0.75 | constant | 0 | flow_temporal_strict | coupling_settled_late | not_promotion_candidate | 28.41701316833496 | 21.808535358429374 | 35.54386901855469 | 21.118317653656387 | 0.8707787607036579 | -4.5637700616408615e-05 | -1.1166557669639587e-06 |
| source_0p80_constant_step20 | candidate | temporal_failed | 0.8 | constant | 0 | flow_temporal_strict | coupling_settled_late | not_promotion_candidate | 31.359745025634766 | 24.08417959213305 | 36.0634880065918 | 23.783608764648896 | 0.9281774518903031 | -2.1731575131679863e-05 | -4.6659260988235474e-07 |
| source_0p75_reset_pressure_step20 | force_sign_failed | temporal_failed | 0.75 | constant | 0 | flow_temporal_strict | coupling_unsettled | not_promotion_candidate | 32.88761520385742 | 25.300280584335876 | 33.80222702026367 | 25.300280584335876 | 1.105092049890159 | 8.922783131377726e-06 | 1.5832483768463135e-08 |
| source_0p75_ramp2_step20 | candidate | temporal_failed | 0.75 | linear_ramp | 2 | flow_temporal_strict | coupling_settled_late | not_promotion_candidate | 29.146724700927734 | 22.53506886100811 | 33.390907287597656 | 21.085058422088977 | 0.888971469006338 | -6.574975560207086e-05 | -3.536231815814972e-06 |
| source_0p80_ramp2_step20 | candidate | temporal_failed | 0.8 | linear_ramp | 2 | flow_temporal_strict | coupling_settled_late | not_promotion_candidate | 32.39445877075195 | 24.901540102005523 | 33.69102096557617 | 23.744439439774027 | 0.9474913738967601 | -4.459377451209714e-05 | -2.9923394322395325e-06 |
| source_0p75_ramp5_step20 | candidate | temporal_failed | 0.75 | linear_ramp | 5 | flow_temporal_failed | coupling_settled_late | not_promotion_candidate | 32.097145080566406 | 24.732337203980045 | 32.15736770629883 | 19.449453424454106 | 0.9163411943186495 | -0.000198867892722342 | -6.66547566652298e-06 |
