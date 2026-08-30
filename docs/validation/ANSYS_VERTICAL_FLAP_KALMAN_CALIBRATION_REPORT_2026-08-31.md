# ANSYS Vertical-Flap Kalman Calibration Report

Date: 2026-08-31

Exit classification: **FAIL_NO_KALMAN_PREDICTIVE_VALUE**

## Outcome

R24 completed as a CPU-only, solver-independent replay. No Taichi/CUDA solver,
Fluent job, shadow run, active Kalman run, adaptive filter, GRU, or long-horizon
simulation was launched. Offline work metrics are aligned source telemetry, not
a causal acceleration measurement.

Stop the Kalman promotion path. R25 is not authorized by this result.

## Evidence boundary

- WSL branch: codex/kalman-calibration-r24
- immutable starting HEAD: f916f80afac5f5ca6d6558e4c3e87fba40831626
- D0: r47 accepted steps 1-100 fit, 101-200 frozen selection
- D1: r51 accepted steps 1-50 held out from every tuning choice
- dt_s: 0.0005
- layout: 373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164
- active axes: [False, True, True]; x is an identically-zero
  plane-strain axis and is excluded from normalized RMSE, NIS, gain, and
  root-cause aggregation
- K0 parity: True, max absolute differences
  {"bias": 0.0, "nis": 0.0, "rmse": 0.0}
- reduced step_fields were used only as observations and were bound to per-step
  history and checkpoint-journal SHA values.

## Held-out ranking

| Candidate | normalized RMSE | NIS mean | gain mean | eligible |
| --- | ---: | ---: | ---: | --- |
| C0 | 0.924019283 | n/a | n/a | yes |
| K1 | 0.92772829 | 2.87706512 | 0.84021442 | yes |
| C1a | 1.01103046 | n/a | n/a | yes |
| C1b | 1.15131676 | n/a | n/a | yes |
| C1c | 1.26967091 | n/a | n/a | yes |
| K2 | 1.04713552 | 5.59723468 | 0.755989048 | yes |
| K0 | 1.27520848 | 121.007428 | 0.999958472 | no (degenerate_gain) |

Best simple extrapolation: C1a.
Predictive Kalman candidates: [].
Statistically valid Kalman candidates: [].
Recommended for R25: [].

## Confirmed statistical diagnosis

- K0 innovation covariance S is under-dispersed relative to accepted-state innovation on axes y,z
- constant-rate innovations retain bias/serial structure on axes y,z

The production time index was reproduced step-by-step, so an implementation
off-by-one mismatch is not supported. This does not exclude model lag; the
innovation autocorrelation, Ljung-Box, bias, NIS, gain, and covariance evidence
is recorded in kalman_innovation_audit.json.

## Deterministic artifacts

- kalman_candidate_ranking.json: 01ebe5b652a210e1e3ed347d28551136c7d782c232dab1e6681a8f127695f734
- kalman_candidate_step_metrics.csv: d4fa3d6ad6ee629fde294b20e4ecc7afa1374f19910b9aa2bafa78206e20d8b5
- kalman_data_split_manifest.json: 81c7613c1e3a12e0d8f4294b39a22bff9afc6ea3bd49e4abcb16b6dfc830c35d
- kalman_innovation_audit.json: 729a3a488be08bdf839ffe87c8c4a85e5211997ff82cd81eb4b503111374a234

## Commands and dirty-diff boundary

Campaign command:

    python3 tools/audit_ansys_vertical_flap_kalman.py --d0-root /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__fresh01__material_fine__20260830__r47 --d0-attempt /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__resume200__material_fine__20260830__r47 --d1-root /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3 --d1-attempt /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3__resume50__attempt1 --output-dir /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/kalman_statistical_calibration/ansys_vf__kalman__offline_calibration__20260831__r24 --predictor-source /home/zhuohengli/worktrees/HIBM-MPM-r21-validation/simulation_core/coupling/interface_kalman_predictor.py --fit-stop 100

Focused host-test gate:

    python3 -m pytest -q tests/validation/test_kalman_statistical_calibration.py

Changed/untracked boundary at report generation:

- M docs/README.md
-  A docs/validation/ANSYS_VERTICAL_FLAP_KALMAN_CALIBRATION_REPORT_2026-08-31.md
-  A docs/validation/ANSYS_VERTICAL_FLAP_KALMAN_STATISTICAL_CALIBRATION_GOAL_2026-08-31.md
-  A src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_calibration.py
-  A src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_campaign.py
-  A src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_evidence.py
-  A src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_filter.py
-  A src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_replay.py
-  A src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_reporting.py
-  A src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_selection.py
-  A src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_types.py
-  A tests/validation/test_kalman_statistical_calibration.py
-  A tools/audit_ansys_vertical_flap_kalman.py

No commit, push, merge, remote mutation, or R25 work was performed.

## Final verification after artifact generation

- The focused R24 suite passed 23/23:

      python3 -m pytest -q tests/validation/test_kalman_statistical_calibration.py

- The existing production predictor, initial-guess controller,
  active-writeback, and r51 IQN-reuse contract suites passed 111/111:

      python3 -m pytest -q tests/coupling/test_interface_kalman_predictor.py tests/coupling/test_interface_initial_guess_controller.py tests/coupling/test_active_kalman_writeback.py tests/validation/test_kalman_iqn_reuse_fine_contracts.py

  The only warning was the pre-existing ragged-array
  `VisibleDeprecationWarning` in the production predictor input-validation
  test.
- All new Python entry points and modules passed:

      python3 -m py_compile tools/audit_ansys_vertical_flap_kalman.py src/refactored/validation/ansys_vertical_flap_fsi/kalman_statistical_*.py

- The final-schema campaign was executed twice consecutively with the same
  exit classification and the same four SHA256 values shown above.
- `kalman_candidate_step_metrics.csv` contains 5250 data rows plus one header.
  Its row schema distinguishes the n-1 accepted source SHA from the current
  measurement SHA and records both candidate ID and model family.
- The K0 source-only adapter reported `production_import_loaded_taichi=false`;
  the executed predictor bytes also matched the D0/D1 manifest SHA before
  import.

The report-generation status block above is a historical boundary captured
before the separately authorized commit and push. Later publication does not
change the artifact fingerprints or R24 scientific decision.
