# ANSYS Vertical-Flap Flow-Collapse Diagnostic Matrix

primary_observation = feedback_on final p999=10.311923771858233 m/s; feedback_off final p999=10.29703426361084 m/s
current_best_hypothesis = projection-only flow path is the primary suspect for flow collapse
next_action = prioritize flow predictor, inlet driving, outlet, and projection solver path

| scenario | status | final p999 m/s | max p999 m/s | flow status |
|---|---:|---:|---:|---|
| feedback_on_step10 | completed | 10.311923771858233 | 24.666577596664457 | collapsed_after_initial_acceleration |
| feedback_off_step10 | completed | 10.29703426361084 | 25.043361543655397 | collapsed_after_initial_acceleration |
| solver_fv_jacobi_1080_step10 | completed | 10.311923771858233 | 24.666577596664457 | collapsed_after_initial_acceleration |
| solver_fv_cg_1080_step10 | completed | 10.0 | 10.0 | below_official_range |
| solver_fv_cg_4096_step10 | completed | 10.0 | 10.0 | below_official_range |
| reset_pressure_first_only_step10 | completed | 10.311923771858233 | 24.666577596664457 | collapsed_after_initial_acceleration |
| reset_pressure_every_step_step10 | completed | 17.799482637405614 | 23.82972980308536 | collapsed_after_initial_acceleration |
| reinitialize_inlet_each_step_step10 | completed | 22.98385433959964 | 22.98582481193546 | within_official_range |
