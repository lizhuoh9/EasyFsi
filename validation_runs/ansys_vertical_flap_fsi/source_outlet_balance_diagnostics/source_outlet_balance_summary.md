# ANSYS Vertical-Flap Source/Outlet Balance Diagnostics

best_candidate = source_strength_0p75_step10
candidate_status = candidate_found
primary_observation = source_strength range 0.2-1.0; best_candidate=source_strength_0p75_step10; p999 range 10.023739116668702-31.845914394379385 m/s; peak range 10.116854667663574-41.35007095336914 m/s
current_best_hypothesis = a non-full-reset source strength candidate satisfies the 10-step flow gate
next_action = run a 20-step candidate check before any 50-step run

## Source Strength Sweep

| scenario | strength | status | peak m/s | p999 m/s | pressure ratio | velocity ratio |
|---|---:|---|---:|---:|---:|---:|
| source_strength_0p20_step10 | 0.2 | below_p999_gate | 10.116854667663574 | 10.023739116668702 | -1.251691705765342 | 0.0 |
| source_strength_0p30_step10 | 0.3 | below_p999_gate | 10.967827796936035 | 10.252599336624199 | -0.394025073719484 | 0.33531922203964987 |
| source_strength_0p40_step10 | 0.4 | below_p999_gate | 12.487918853759766 | 10.935854234695531 | -0.23800781110496763 | 0.5983510374576638 |
| source_strength_0p50_step10 | 0.5 | below_p999_gate | 16.69333267211914 | 12.904831772804334 | -0.14436468597077629 | 0.7562432539775703 |
| source_strength_0p60_step10 | 0.6 | below_p999_gate | 21.371919631958008 | 16.530649398803757 | -0.08141387805918174 | 0.8500109745523777 |
| source_strength_0p75_step10 | 0.75 | candidate | 28.874183654785156 | 22.175501684189303 | -0.02067205639259943 | 0.9603284515014241 |
| source_strength_1p00_step10 | 1.0 | over_accelerated | 41.35007095336914 | 31.845914394379385 | 0.04465573384985495 | 1.0667161366893787 |

## Outlet Balance Sweep

| scenario | strength | status | peak m/s | p999 m/s | pressure ratio | velocity ratio |
|---|---:|---|---:|---:|---:|---:|
| projection_only_baseline_step10 | 1.0 | below_p999_gate | 10.901412963867188 | 10.311924845695513 | 0.0 | 0.0 |
| diagnostic_reinitialize_upper_bound_step10 | 1.0 | diagnostic_excluded | 29.44530487060547 | 22.98385791397098 | 0.0 | 0.0 |
| selected_source_strength_step10 | 0.75 | candidate | 28.874183654785156 | 22.175501684189303 | -0.02067205639259943 | 0.9603284515014241 |
| selected_source_strength_reset_pressure_step10 | 0.75 | candidate | 33.63189697265625 | 25.899071310043922 | -0.006938480406692922 | 1.1417312776894566 |
| selected_source_strength_ramp5_step10 | 0.75 | below_p999_gate | 22.742055892944336 | 17.29636069870012 | 0.002629433263870057 | 0.5144381795055157 |
