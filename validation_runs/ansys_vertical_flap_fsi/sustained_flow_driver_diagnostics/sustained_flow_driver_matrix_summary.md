# ANSYS Vertical-Flap Sustained Flow Driver Matrix

primary_observation = projection_only final p999=10.311924845695513 m/s; diagnostic_reinitialize final p999=22.983848978042634 m/s; best_sustained=sustained_inlet_predictor_feedback_off_step10 final p999=32.11051559448242 m/s
current_best_hypothesis = sustained flow driver restores p999 but over-accelerates; refine source strength, outlet compatibility, and predictor coupling before any 50-step run
next_action = refine source strength, outlet compatibility, and predictor coupling before any 50-step run

| scenario | mode | status | final p999 m/s | source flux m3/s | flow status |
|---|---|---:|---:|---:|---|
| projection_only_step10 | projection_only | completed | 10.311924845695513 | 0.0 | collapsed_after_initial_acceleration |
| reinitialize_inlet_each_step_step10 | reinitialize_inlet_each_step_diagnostic | completed | 22.983848978042634 | 0.0 | within_official_range |
| sustained_boundary_inlet_step10 | sustained_boundary_inlet | completed | 10.311926753044146 | 0.0 | collapsed_after_initial_acceleration |
| sustained_volume_source_inlet_step10 | sustained_volume_source_inlet | completed | 31.845914274216423 | 0.0006000002613291144 | above_official_range |
| sustained_inlet_predictor_step10 | sustained_inlet_predictor | completed | 31.853287534714475 | 0.0006000002613291144 | above_official_range |
| sustained_inlet_predictor_feedback_off_step10 | sustained_inlet_predictor | completed | 32.11051559448242 | 0.0006000002613291144 | above_official_range |
| reset_pressure_every_step_step10 | projection_only | completed | 17.799482412338477 | 0.0 | collapsed_after_initial_acceleration |
