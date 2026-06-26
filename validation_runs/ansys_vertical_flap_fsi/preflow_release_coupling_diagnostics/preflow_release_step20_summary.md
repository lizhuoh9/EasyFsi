# ANSYS Vertical-Flap Preflow-Release STEP20 Diagnostics

best_preflow_release_candidate = none
best_release_flow_candidate = no_preflow_release20_source_0p80_ramp2
promotion_candidate_count = 0
candidate_status = no_promotion_candidate
scope_limit = coupled STEP20 diagnostic only; no 50-step or Fluent parity claim

## Matrix

| scenario | preflow | release flow | release combined | coupling | promotion | continuity | restart |
|---|---|---|---|---|---|---|---|
| no_preflow_release20_source_0p80_ramp2 | flow_temporal_not_applicable | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate |  | False |
| preflow10_release20_source_0p80_ramp2 | flow_temporal_failed | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate | True | False |
| preflow20_release20_source_0p80_ramp2 | flow_temporal_strict | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate | True | False |
| preflow30_release20_source_0p80_ramp2 | flow_temporal_strict | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate | True | False |
| preflow20_release20_source_0p75_constant | flow_temporal_strict | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate | True | False |
| preflow30_release20_source_0p75_constant | flow_temporal_strict | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate | True | False |
| preflow20_release20_source_0p75_ramp2 | flow_temporal_strict | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate | True | False |
| preflow20_release20_source_0p80_ramp2_feedback_off | flow_temporal_strict | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate | True | False |
| preflow20_release20_source_0p80_ramp2_phase_local | flow_temporal_strict | flow_temporal_strict | temporal_failed | coupling_unsettled | not_promotion_candidate | True | True |
