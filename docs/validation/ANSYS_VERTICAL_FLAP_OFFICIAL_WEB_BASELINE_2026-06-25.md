# ANSYS Vertical Flap Official Web Baseline - 2026-06-25

## Source

ANSYS Fluent v242 tutorial:
Modeling Two-Way Fluid-Structure Interaction (FSI) Within Fluent.

Source URL:
https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_tg/flu_tg_fsi_2way.html

## Scope

Compared against ANSYS official tutorial web-published displacement scale.
Not yet compared against a Fluent exported time-history report.

This baseline does not download or run a Fluent case. It records the official web-published contour scales and compares them with a 50-step EasyFsi run.

## Official Web-Published Reference

Geometry:

- duct_length_m = 0.10
- duct_height_m = 0.04
- flap_height_m = 0.01
- flap_thickness_m = 0.003

Material:

- silicone rubber
- density_kgm3 = 1600
- young_modulus_pa = 1.0e6
- poisson_ratio = 0.47

Flow:

- inlet_velocity_mps = 10.0
- outlet = pressure outlet
- modeled domain = lower half by symmetry

Time:

- dt_s = 0.0005
- step_count = 50
- final_time_s = 0.025

Published result scale:

- displacement contour range = 0 to 5.1e-05 m
- velocity magnitude contour range = 20 to 29 m/s

Checked-in reference CSV:

```text
docs/validation/ansys_vertical_flap_official_web_reference_2026-06-25.csv
```

## EasyFsi Run

Command:

```powershell
& $python run_simulation.py ansys-vertical-flap-fsi --steps 50 --json `
  > validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json
```

The PowerShell redirected output was normalized to UTF-8 after capture because the redirected JSON body contained NUL bytes. The decoded report contains 50 history rows.

Runtime artifact:

```text
validation_runs/ansys_vertical_flap_fsi/easyfsi/easyfsi_step050.json
```

## Diagnostics

Command:

```powershell
& $python -m tools.validation.print_ansys_vertical_flap_diagnostics `
  --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json `
  --fluent-tip-csv validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv `
  --output-dir validation_runs\ansys_vertical_flap_fsi\compare
```

Generated artifacts:

```text
validation_runs/ansys_vertical_flap_fsi/compare/easyfsi_summary.csv
validation_runs/ansys_vertical_flap_fsi/compare/easyfsi_summary.json
validation_runs/ansys_vertical_flap_fsi/compare/easyfsi_history.csv
validation_runs/ansys_vertical_flap_fsi/compare/stage_check.md
validation_runs/ansys_vertical_flap_fsi/compare/displacement_compare.csv
```

## Result

EasyFsi summary:

- EasyFsi status = FAIL_SOLID_HISTORY
- EasyFsi velocity_peak_mps = 28.15654945373535
- EasyFsi velocity_peak_relerr = 0.002012436075991108
- official velocity range = 20 to 29 m/s
- EasyFsi max_disp_m = 2.826645868481137e-05
- EasyFsi ref_max_disp_m = 5.1e-05
- EasyFsi disp_relerr = 0.4457557120625222
- EasyFsi tip_dz_final_m = -2.560298889875412e-05
- EasyFsi tip_dz_min_m = -3.392063081264496e-05
- EasyFsi tip_dz_max_m = -7.167458534240723e-06
- tip_dz_monotonic_violation_count = 23
- first_tip_dz_violation_step = 5
- max_tip_dz_rebound_m = 5.472451448440552e-06
- tip_dz_sign_violation_count = 0
- feedback_closure_status = OPEN_LOOP_LOAD_REUSE

Official-web displacement comparison:

- Official web displacement scale = 5.1e-05
- EasyFsi final tip total displacement = 2.703296924203968e-05
- Final absolute error vs official web scale = 2.396703075796032e-05
- Final relative error vs official web scale = 0.4699417795678494
- EasyFsi final streamwise tip displacement = -2.560298889875412e-05
- EasyFsi final vertical tip displacement = 8.67573544383049e-06

## Interpretation

The flow gate passes against the web-published velocity range: EasyFsi reports `28.15654945373535 m/s`, which is inside the official `20 to 29 m/s` contour scale.

The current blocking status is `FAIL_SOLID_HISTORY`. The final tip streamwise displacement has the correct negative sign, but the history rebounds 23 times, with the first detected rebound at step 5 and a maximum rebound of `5.472451448440552e-06 m`.

The feedback section reports `OPEN_LOOP_LOAD_REUSE`, so this baseline still supports the next solver target: make solid/marker feedback affect a subsequent fluid solve instead of reusing the same open-loop load field.

This artifact set is sufficient for the first EasyFsi-vs-official-web comparison. It is not sufficient for pointwise Fluent validation; that still requires a Fluent-exported time-history report.

## Next Solver Target

1. Remove `OPEN_LOOP_LOAD_REUSE` by implementing closed-loop fluid feedback.
2. Eliminate `FAIL_SOLID_HISTORY` by checking time integration, load persistence, and feedback-loop coupling.
3. After history is physically credible, bring final displacement relative error into a controlled range against the official web scale and later against Fluent exported reports.
