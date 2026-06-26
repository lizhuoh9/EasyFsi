# ANSYS Vertical-Flap Fixed-Solid Load Temporal Diagnostics

best_fixed_solid_load_candidate = fixed_load_0p80_ramp2_step60
fixed_solid_load_candidate_count = 1
candidate_status = candidate_found
scope_limit = fixed-solid load diagnostic only; no coupled 50-step or Fluent parity claim
candidate_rule = completed, non-diagnostic, flow_temporal_strict, and load_temporal_strict

## Matrix

| scenario | flow | load | last20 force mean N | negative fraction | zero crossings | marker residual max N | scatter residual max N |
|---|---|---|---|---|---|---|---|
| fixed_load_0p75_constant_step60 | flow_temporal_strict | load_temporal_failed | -2.400234020543406e-05 | 0.6 | 4 | 0.0 | 4.635111955711424e-12 |
| fixed_load_0p80_constant_step60 | flow_temporal_strict | load_temporal_failed | -1.3649956076023602e-05 | 0.6 | 4 | 0.0 | 3.442935213727842e-12 |
| fixed_load_0p75_ramp2_step60 | flow_temporal_strict | load_temporal_failed | -5.4307036238461634e-05 | 0.7 | 4 | 0.0 | 4.747400660521511e-12 |
| fixed_load_0p80_ramp2_step60 | flow_temporal_strict | load_temporal_strict | -4.7990171893160255e-05 | 0.8 | 4 | 0.0 | 3.983710984787475e-12 |
| projection_only_step60_baseline | flow_temporal_failed | load_temporal_failed | -5.073249590828436e-05 | 0.65 | 4 | 0.0 | 3.3813738701399254e-11 |
| diagnostic_reinitialize_step60_upper_bound | flow_temporal_failed | load_temporal_strict | -0.008651004931407375 | 1.0 | 0 | 0.0 | 2.5917877392866995e-10 |
