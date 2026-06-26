# ANSYS Vertical-Flap Traction Formulation Diagnostics

reference_formulation_candidate = none
candidate_status = no_reference_formulation_candidate
supported_formulation_count = 5
unsupported_formulation_count = 1
pressure_mean_status = not_exposed_by_current_core; force and traction counters are archived
scope_limit = fixed-solid traction formulation diagnostic only; no coupled 50-step or Fluent parity claim

candidate_rule = all A/B/C rows supported, completed, conservative, and stable

## Matrix

| scenario | status | layout | pressure mode | viscous | total force z N | diff from ref N | face ratio | reason |
|---|---|---|---|---|---|---|---|---|
| dual_two_sided_offset0p51_pressure_only | completed | dual_physical_faces | two_sided_pressure_jump | False | -0.00017959052273141572 | 0.0 | 42.520268106203886 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
| dual_one_sided_offset0p51_pressure_only | unsupported | dual_physical_faces | one_sided_surface_pressure | False |  |  |  | dual-face one-sided pressure needs per-face one-sided region support; current core exposes one one_sided_pressure_region_id |
| single_mid_two_sided_offset0p00_pressure_only | completed | single_mid_surface | two_sided_pressure_jump | False | -0.0001726169401754124 | 6.973582556003317e-06 |  | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
| dual_two_sided_offset0p25_pressure_only | completed | dual_physical_faces | two_sided_pressure_jump | False | -0.00035033234658293674 | -0.00017074182385152102 | 0.9943280025877084 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
| dual_two_sided_offset1p00_pressure_only | completed | dual_physical_faces | two_sided_pressure_jump | False | -1.2125077444893233e-05 | 0.0001674654452865225 | 0.6766406208924867 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
| dual_two_sided_offset0p51_viscous_air | completed | dual_physical_faces | two_sided_pressure_jump | True | -0.00017616778972252313 | 3.422733008892593e-06 | 36.03241370167948 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
