# ANSYS Vertical-Flap Fixed-Solid Source Temporal STEP30 Diagnostics

best_fixed_solid_flow_candidate = fixed_source_0p75_constant_step30
candidate_status = candidate_found
fixed_solid_flow_candidate_count = 4
scope_limit = fixed-solid preflow-only diagnostic; not coupled FSI validation

## Matrix

| scenario | status | strength | profile | ramp | p999 m/s | peak m/s | last-10 min p999 | last-10 outlet mean |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fixed_source_0p75_constant_step30 | flow_temporal_strict | 0.75 | constant | 0 | 25.53406524658203 | 32.173492431640625 | 22.384153366088867 | 1.0261125242992655 |
| fixed_source_0p80_constant_step30 | flow_temporal_strict | 0.8 | constant | 0 | 26.374658584594727 | 33.23227310180664 | 24.4965763092041 | 1.0141823826859357 |
| fixed_source_0p75_ramp2_step30 | flow_temporal_strict | 0.75 | linear_ramp | 2 | 25.836065172195436 | 32.55419158935547 | 23.177108764648438 | 1.0666568446658549 |
| fixed_source_0p80_ramp2_step30 | flow_temporal_strict | 0.8 | linear_ramp | 2 | 26.71710956001282 | 33.6639404296875 | 25.392974853515625 | 1.0572379928953903 |
| fixed_source_0p75_ramp5_step30 | flow_temporal_failed | 0.75 | linear_ramp | 5 | 28.01066017150879 | 35.29524230957031 | 25.985532760620117 | 1.2531808310331072 |
| projection_only_step30_baseline | flow_temporal_failed | 1.0 | constant | 0 | 10.518691002845765 | 11.029960632324219 | 10.00255298614502 | 0.0 |
| diagnostic_reinitialize_step30_upper_bound | flow_temporal_not_applicable | 1.0 | constant | 0 | 22.381813049316406 | 28.14044952392578 |  |  |
