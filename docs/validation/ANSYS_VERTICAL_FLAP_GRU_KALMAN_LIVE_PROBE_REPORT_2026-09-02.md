# ANSYS Vertical Flap R25B GRU/Kalman Live No-Commit Probe Report

Status: final. The fresh source-matched r3 chain and all bottom-up evidence contracts
pass. The required fresh read-only Sol review returned `SHIP`.

## 1. Decisive result

R25B answered the review questions on the real strict-CUDA FSI solver without
committing either target step:

- `FAIL_G0_MATCHED_LIVE_VALUE`;
- `FAIL_GK1_INCREMENTAL_LIVE_VALUE`;
- `FAIL_POD_AR_LIVE_VALUE`;
- overall: `FAIL_NO_LIVE_SOLVER_WORK_REDUCTION`.

Every causal neural, Kalman, and POD-AR arm converged, but every one consumed exactly
the same discrete work as carry-forward across target steps 7 and 8: 6 coupling
trials, 4 rejected trials, 1,440 pressure-CG iterations, and 1,452 pressure-operator
matvecs. The noncausal exact-accepted Q upper bound used 2 trials, 0 rejected trials,
480 CG iterations, and 484 matvecs. Q confirms very-near-exact initial-guess
headroom, but it is not a deployable predictor and does not rescue any causal arm.

The frozen stop tree is active. R25B does not authorize endpoints 9/12, filling steps
10/11, R25C, exact12, exact20, exact50, Fluent comparison, a production controller,
online training, deployment, a push, or a PR.

## 2. Review question, implementation identity, and evidence boundary

The parent review accepted the R25A offline science but identified two confounds: the
selected G0 and GK1 architectures had different history windows, and GK1 had three
times the input width. R25B therefore added fixed post-hoc G0-M and GDelta-M controls
without reopening architecture search or D1 selection.

The work started from R25A commit
`adb2a0470085ecca1f772bae14d292df76c963d9` on branch
`codex/gru-kalman-live-probe-r25b`. The implementation used by the authoritative r3
chain is `f833439bf5fc15c6f04923410c8f804e5a394fe8`:

- `00e16d6` separates current R25A runtime identity from the historical dirty-run
  identity;
- `a60e03c` adds the R25B CPU controls, immutable bundles, no-commit sweep, analysis,
  tests, and frozen Goal;
- `2273020` normalizes the Goal permissions;
- `8385136` includes the R25B bundle reader in the executable source map;
- `f36a87f` persists marker reference identity in accepted-step exports;
- `f833439` completes schema-2 OOD, explicit rejected-trial, pressure-matvec, and
  anchor-counter evidence contracts;
- `4349f63` adds the analysis-only G0-M continuation branch found by final review;
  it is outside the formal CUDA source map and does not alter any raw r3 solve.

The original un-suffixed roots and the complete `*_r2` chain remain untouched local
diagnostics. The r2 result was scientifically consistent with r3 but contract
incomplete; it omitted the four evidence families added in `f833439`. It is not
authoritative for this report. Only the `*_r3` chain below is authoritative.

## 3. Frozen model and candidate contract

G0-M and GDelta-M both use rank 8, window 4, hidden size 16, and seeds 0/1/2. G0-M
receives accepted modal state only. GDelta-M receives accepted state, accepted
increments, and repeated carry, giving the same `3 * rank` input width as GK1 without
Kalman state, innovation, prediction, target value, or rejected trials. POD,
normalization, and fitting use D0 steps 1--100; steps 101--200 are used only by the
fixed early-stopping rule. D1 is not used for tuning.

Matched-control training took 4.96083682400058 s in the CPU-only environment:

| family | seed | best epoch | best selection loss |
| --- | ---: | ---: | ---: |
| G0-M | 0 | 281 | 0.05613714735 |
| G0-M | 1 | 297 | 0.06678998589 |
| G0-M | 2 | 277 | 0.06303970633 |
| GDelta-M | 0 | 310 | 0.07817808250 |
| GDelta-M | 1 | 367 | 0.07737041898 |
| GDelta-M | 2 | 364 | 0.07368284006 |

Frozen identities:

- selection fingerprint:
  `5bb2a535b40e596a836e41201a2c682ee29ea670dc7300a26e3cb1613bb19f86`;
- POD fingerprint:
  `1469de8da1fd0ac2e45049da442587b3176525a1ca8e6581cb2e89450f640143`;
- normalization fingerprint:
  `8df51086d4eaf11d279665bf4a1894b185beb8b756794a4ccbdcf34d142c8e18`;
- exact8 source fingerprint:
  `5c3f7d2605252baced26f3d712cba711069c8017dc028704efac78b634d18c53`;
- marker layout:
  `373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164`;
- marker identity: float64 `(128, 3)` reference positions, 64 region-101 markers,
  64 region-202 markers, and fixed reference area
  `5.999999848427251e-05 m^2`.

Both schema-2 bundles loaded through the production bottom-up validator as finite,
read-only float64 `(13, 128, 3)` arrays with exact arm order, `xyz` axes,
`marker_velocity_mps` units, `dt_s = 0.0005`, and bitwise positive-zero x velocity.

| target | causal prefix | target frame SHA256 | manifest SHA256 | NPZ SHA256 |
| ---: | ---: | --- | --- | --- |
| 7 | 1--6 | `b891cd3a79b5dc3028794f3b7a46792bf80567ad7740090e6977c16b2fa5420b` | `9cd9e945ff9dbd194194b4a9ae816cfbc8158a28260275d2105348bbb05531cd` | `2caede374737ab46fbb2d134262e5cd6ffea4aece0b0204545b155137d6b4469` |
| 8 | 1--7 | `46a10202e6a1605688b62f8eee2a4f862ef57f421ff8853c8d181d4145532b28` | `b8aa5211b22ae343cfc9e5688f49ecb442f6512ee24c53c334fbde3884296cfb` | `c77303b24d0036c31a08b89ec27abacb0675c8005462ff68850fa0828776aa4c` |

For every causal arm, `max_source_step = target - 1`; Q alone is noncausal and uses
the accepted target frame. The schema-2 source identity also records the causal OOD
diagnostics once per bundle:

| target | history/innovation source steps | max abs normalized POD coefficient | outside D0 range fraction | max abs normalized K1 innovation |
| ---: | --- | ---: | ---: | ---: |
| 7 | 3--6 | 4.913577079 | 0.0625 | 6.355172888 |
| 8 | 4--7 | 3.047015997 | 0.03125 | 6.355172888 |

These OOD values are diagnostics, not work gates. The system Python used by the CUDA
runner has no Torch installation (`find_spec("torch") is None`); PyTorch was confined
to the CPU candidate-generation process.

The four generator-source hashes recorded by each manifest were separately
recomputed with zero mismatches. They are audited provenance, not a manifest-authored
external trust root for bundle acceptance; the loader fail-closes on the bundle
schema, source bounds, layout/marker identities, whole-NPZ and per-candidate hashes,
and the OOD contract.

## 4. Fresh source-matched numerical chain

The following six formal roots contain source maps that are equal key-for-key, not
merely by digest. Each contains 141 entries and has canonical digest
`e0b643d2f8ec8d36935148b44c8139125cb9499798bbb9109ccb6e1bb6f4e28b`:

- `ansys_vf__r25b__preflow_r3_dryrun`;
- `ansys_vf__r25b__preflow_r3`;
- `ansys_vf__r25b__exact8_r3`;
- `ansys_vf__r25b__exact8_r3__resume8__attempt1`;
- `ansys_vf__r25b__probe7_r3`;
- `ansys_vf__r25b__probe8_r3`.

The dry run passed the frozen configuration before GPU execution. Fresh preflow
reached three consecutive stationary windows at step 79; the largest metric across
their union was `0.008919564123750999 < 0.01`, and the final-window maximum SST eddy
viscosity span was `0.006253935580026151`. Its 31-field strict snapshot is
`state.4237fbac9c384f6fb0fb3427c7ce2f84.npz`, SHA256
`7c4cb847bbc8b09a10049f73c5fd6c9589ddf9413e59a6634e043ed2ef2dccc3`,
with manifest SHA256
`7bbf1a454209a8ce9b599b7ae2f8cac32eb0a9e7655377957d99e42eed6f9c8b`
and ledger generation 244. Exact8 and both probes bind the same snapshot artifact
identity.

The fresh carry exact8 canonical completed 8/8 accepted steps and `0.004 s`, with 24
coupling trials, 16 rejected trials, and 5,728 pressure-CG iterations. Per-step
trials were all 3; CG
counts were 640, 768, then 720 for steps 3--8. Adaptive solid substeps were 1,280,
1,282, 1,284, 1,286, 1,286, 1,286, 1,286, and 1,285. A no-op resume restored
generation `4018af5a76494881920f7a581ec411fa`, accepted step 8, and `0.004 s` without
executing another physical step or creating accepted artifacts in the attempt root.

Every accepted prefix step in both probes independently consumed the complete macro
time in both solvers. The largest fluid accepted-time error was
`4.336808689942018e-19 s`; fluid remaining time, solid accepted-time error, and solid
remaining time were all exactly zero. Residual convergence never substituted for
physical-time advancement.

### Independent-CUDA replay boundary

The runs are source/config/snapshot/layout matched, but independent f32 CUDA replays
are not bitwise-identical trajectories. Relative to exact8, the maximum componentwise
accepted marker-velocity drift over the probe-7 prefix was
`1.7136335372924805e-07 m/s`; over the probe-8 prefix it was
`1.2665987014770508e-07 m/s`. At each target base, external C0 differed from that
run's live carry by active-yz RMSE `3.4239297282863594e-08` and
`2.868156543245427e-08 m/s`. These are only `2.929222307763767e-06` and
`1.495171011846986e-06` of the corresponding C0-to-target RMSE.

This evidence is therefore source-matched, not a bitwise trajectory replay. Requested
candidate versus actual iteration-zero equality inside each live solve is still
exact.

## 5. Frozen prediction diagnostics

Prediction metrics were computed before the CUDA runs and never entered a work gate.
The tip mask selects 14 markers in the highest 10% of reference y at threshold
`0.008937499865714927 m`; its SHA256 is
`54fe1a6ddd2561c646f0e00f9bf76fe84e2a81dd318c5039e48f7befc7463603`.
Area-weighted RMSE equals unweighted RMSE at the shown precision because reference
marker areas are uniform.

| arm | RMSE7 | tip RMSE7 | alpha7 | r_perp7 | RMSE8 | tip RMSE8 | alpha8 | r_perp8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 1.16889e-2 | 1.78956e-2 | 0 | 0 | 1.91828e-2 | 4.26278e-2 | 0 | 0 |
| K1 | 1.23291e-2 | 1.80384e-2 | -0.0518654 | 0.0782986 | 2.02652e-2 | 4.44224e-2 | -0.0553101 | 0.0485898 |
| G0-M-seed0 | 4.39617e-3 | 2.81193e-3 | 0.728506 | 0.260272 | 1.44888e-2 | 3.18039e-2 | 0.246517 | 0.0523476 |
| G0-M-seed1 | 7.99182e-3 | 1.30116e-2 | 0.317969 | 0.0479198 | 9.45973e-3 | 2.09778e-2 | 0.507951 | 0.0327237 |
| G0-M-seed2 | 1.68392e-3 | 1.70678e-3 | 0.917663 | 0.118214 | 1.23147e-2 | 2.68112e-2 | 0.358903 | 0.0333977 |
| GDelta-M-seed0 | 7.08608e-3 | 1.12964e-2 | 0.396832 | 0.0608021 | 1.38654e-2 | 2.97053e-2 | 0.280838 | 0.0724675 |
| GDelta-M-seed1 | 5.84553e-3 | 8.51386e-3 | 0.505460 | 0.0743228 | 1.51507e-2 | 3.20728e-2 | 0.216697 | 0.101161 |
| GDelta-M-seed2 | 5.96474e-3 | 8.95324e-3 | 0.492643 | 0.0546582 | 1.03138e-2 | 2.27421e-2 | 0.465261 | 0.0559400 |
| GK1-seed0 | 5.51770e-3 | 9.55663e-3 | 0.534510 | 0.0784109 | 1.11293e-2 | 2.32024e-2 | 0.429108 | 0.103341 |
| GK1-seed1 | 5.85191e-3 | 1.01637e-2 | 0.508351 | 0.0944524 | 9.88236e-3 | 2.01959e-2 | 0.502737 | 0.134637 |
| GK1-seed2 | 6.50325e-3 | 8.58976e-3 | 0.455219 | 0.112928 | 5.61410e-3 | 1.26178e-2 | 0.717849 | 0.0777336 |
| AR | 1.33718e-4 | 1.74997e-4 | 1.00415 | 0.0106613 | 1.65853e-4 | 1.99747e-4 | 0.994966 | 0.00702960 |
| Q | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |

These diagnostics show substantial geometric improvement for many candidates,
especially POD-AR, but they do not establish solver acceleration.

## 6. Live transaction and discrete-work evidence

Both compact reports passed `validate_live_probe_report` from the bottom up; no
aggregate success Boolean was trusted alone.

| target | terminal accepted step/time | accepted artifacts | requested/actual SHA | per-arm rollback | final sweep | anchor delta | wall time |
| ---: | --- | ---: | --- | --- | --- | ---: | ---: |
| 7 | 6 / 0.003 s | 6 fields + 6 histories | 13/13 exact | 13/13 equal | equal; no mismatches | 37 | 467.410 s |
| 8 | 7 / 0.0035 s | 7 fields + 7 histories | 13/13 exact | 13/13 equal | equal; no mismatches | 37 | 429.308 s |

Neither target artifact exists. Every arm recorded equal requested and actual
first-IQN-guess SHA values, every `finally` rollback recaptured an equal complete
`HostMacroStepState`, and the post-sweep state equaled the pre-sweep base. Each anchor
delta is exactly the sum of its 13 raw-row trial counts. No target step, accepted
time, GRU state, or Kalman state was committed.

Runtime identity was the same in both probes: Taichi 1.7.4, actual/requested CUDA,
strict architecture verified, f32, seed 0, IQN reuse off, Kalman writeback off,
omega 0.75, maximum 16 trials, relative tolerance `1e-3`, and absolute tolerance 0.
Wall time is diagnostic only and is not acceleration evidence.

| arm | step7 trials/rejected/CG/matvec | step8 trials/rejected/CG/matvec | total trials/rejected/CG/matvec | first residual 7 | first residual 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 1.65591e-2 | 2.71483e-2 |
| K1 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 1.74691e-2 | 2.86817e-2 |
| G0-M-seed0 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 6.23330e-3 | 2.05026e-2 |
| G0-M-seed1 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 1.13210e-2 | 1.33864e-2 |
| G0-M-seed2 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 2.38554e-3 | 1.74281e-2 |
| GDelta-M-seed0 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 1.00404e-2 | 1.96247e-2 |
| GDelta-M-seed1 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 8.28458e-3 | 2.14445e-2 |
| GDelta-M-seed2 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 8.45283e-3 | 1.45937e-2 |
| GK1-seed0 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 7.81485e-3 | 1.57539e-2 |
| GK1-seed1 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 8.28679e-3 | 1.39928e-2 |
| GK1-seed2 | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 9.21777e-3 | 7.94269e-3 |
| AR | 3/2/720/726 | 3/2/720/726 | 6/4/1440/1452 | 1.89134e-4 | 2.34759e-4 |
| Q | 1/0/240/242 | 1/0/240/242 | 2/0/480/484 | 4.60927e-8 | 4.09546e-8 |

All 26 solves converged. K1 worsened the first residual at both targets. Every G0-M,
GDelta-M, GK1, and AR arm improved it at both targets, but none crossed a trial
boundary or reduced pressure CG. Those arms therefore have diagnostic classification
`PREDICTION_IMPROVED_NO_SOLVER_WORK_REDUCTION`. POD-AR is the clearest separation:
its target and first-residual errors are about two orders below C0, yet it still pays
3 trials and 720 CG at each target. Q alone reaches the one-trial floor.

Across all 26 rows, direct summation gives 74 trials, 48 rejected trials, 17,760 CG,
17,908 matvecs, 74 fluid solves, 74 solid macro solves, 23,125 momentum substeps,
20,350 SST substeps, and 95,127 MPM substeps. The anchor delta total is also 74.

## 7. Frozen factor analysis and classifications

For seeds 0, 1, and 2, all five frozen effects (`Delta_G`, `Delta_K`,
`Delta_G_given_K`, `Delta_Kinfo`, and interaction) are exactly zero for both trials
and pressure CG. The separately evaluated G0-M continuation set is also empty for
all three seeds. No capacity-matched standalone-GRU seed and no GK1-incremental seed
passes.

| question | frozen result | reason |
| --- | --- | --- |
| capacity-matched non-Kalman GRU vs C0 | `FAIL_G0_MATCHED_LIVE_VALUE` | 0/3 passing seeds; equal trials and 0% CG reduction |
| GK1 vs both K1 and paired GDelta-M | `FAIL_GK1_INCREMENTAL_LIVE_VALUE` | 0/3 passing seeds; all work equal |
| POD-AR vs C0 | `FAIL_POD_AR_LIVE_VALUE` | equal 6 trials, not at least one fewer |
| overall | `FAIL_NO_LIVE_SOLVER_WORK_REDUCTION` | no causal arm reduces trials or sufficient CG work |

This does not mean the predictors are numerically meaningless; several are much
closer to the accepted target. It establishes the narrower result requested by the
review: under the frozen real-solver conditions, that improvement is insufficient to
reduce paid discrete work. Kalman-specific information has no demonstrated live
incremental value.

## 8. Verification evidence and limitations

The implementation followed RED/GREEN contracts for bundle schema and identity,
causal source bounds, exact arm order, OOD provenance, analysis formulas, CLI mutual
exclusion, forced trial-vector recording, requested/actual first-guess identity,
explicit rejected trials, exact pressure matvec aggregation, per-arm/final rollback,
anchor conservation, and terminal accepted step/time.

Verified evidence:

- R25B focused CPU suite in its Torch environment: 14 passed, 3 skipped;
- formal CLI/checkpoint/resume in-scope regression: 141 passed;
- independent Taichi candidate-runner contract: 1 passed, 21 deselected;
- existing pressure-nullspace aggregation contract: 1 passed, 10 deselected;
- adjacent SST runner suite: 21 passed and 21 subtests passed, with one pre-existing
  failure because the old AST expectation omits the now-present
  `pressure_outlet_zmin` and `velocity_inlet_zmax` kwargs;
- fresh dry-run, preflow, exact8, no-op resume, CPU generation, both probes, and
  bottom-up r3 analysis completed on their stated boundaries;
- `py_compile` and `git diff --check` passed before r3 and after the review fix;
- a scoped pre-landing review found no material implementation issue;
- the required fresh read-only Sol review checked the implementation, raw r3 evidence,
  classifications, stop tree, and synchronized documentation and returned `SHIP`.

The analysis artifact is
`validation_runs/gru_kalman_live/ansys_vf__r25b__live_analysis_r3.json`, SHA256
`87eeb92088629ce5d8aa0adbff61731c0ca4ef499a786ff7c0cae916d680a089`.
The classifier source SHA256 is
`8bc19f383142927676e72170769c35daae844c943c00da40d97f184b3f043d96`.

These are focused code gates and bounded numerical evidence. They are not a
full-suite or coverage claim, production validation, Fluent parity result,
accepted-solution equivalence over a long run, deployment qualification, or measured
speedup. Numerical artifacts remain ignored local evidence and are not committed.

## 9. Artifact roots and hard stop

Authoritative local evidence:

- dry-run: `validation_runs/gru_kalman_live/ansys_vf__r25b__preflow_r3_dryrun`;
- preflow: `validation_runs/gru_kalman_live/ansys_vf__r25b__preflow_r3`;
- exact8: `validation_runs/gru_kalman_live/ansys_vf__r25b__exact8_r3`;
- no-op resume:
  `validation_runs/gru_kalman_live/ansys_vf__r25b__exact8_r3__resume8__attempt1`;
- candidates: `validation_runs/gru_kalman_live/ansys_vf__r25b__candidates_r3`;
- step 7: `validation_runs/gru_kalman_live/ansys_vf__r25b__probe7_r3`;
- step 8: `validation_runs/gru_kalman_live/ansys_vf__r25b__probe8_r3`;
- analysis: `validation_runs/gru_kalman_live/ansys_vf__r25b__live_analysis_r3.json`.

The accepted-state predictor expansion stops here. No causal arm earned permission
to add endpoints, fill intermediate targets, or start a committed run. The only
reduced-work arm is the noncausal exact Q oracle. The final result remains
`deployable=false`.
