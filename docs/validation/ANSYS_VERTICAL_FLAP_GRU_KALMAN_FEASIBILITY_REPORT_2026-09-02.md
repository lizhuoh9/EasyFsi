# ANSYS Vertical Flap R25A POD-GRU / Kalman-Residual-GRU Feasibility

This is a frozen R25A offline feasibility study. D1 was independently loaded after selection sealing, but it is the same operating condition already inspected by R24; it is not an unseen cross-condition/generalization test.

## Evidence boundary and frozen matrix

- Selection fingerprint: 5bb2a535b40e596a836e41201a2c682ee29ea670dc7300a26e3cb1613bb19f86
- D0 frames: 200 (fit 1-100; selection 101-200); D1 frames: 50 (cold start, score 9-50).
- Selected architectures: {"gru": "8:8:16", "kalman0_gru": "8:4:16", "kalman1_gru": "8:4:16", "pod_ar": "8:4:16"}.
- K0 fingerprint: 383f9fc10475449cd88ce4fbc9b0d3b7595b47e62e7ef4aa53a516dd0058e03e; K1 fingerprint: 603ec011922df847f61a0d8a91216ba2a2e3b2c60eb757092f910df37678d91e; predictor SHA256: 5bbb7735ba43493ce4b768a4c87008f86347192b2c71f12385dde97ee556856b.
- POD, normalization, AR, and neural fitting use D0 steps 1-100 only; D0 steps 101-200 are selection and early stopping.
- D1 opens only after the pre-D1 seal and immediate re-hash verification; no training or selection API is used after opening.

## D1 metrics and gate outcomes

| model | active-yz NRMSE | y RMSE | z RMSE | y bias | z bias | marker p95 | marker max |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.893859626583 | 0.0138718330327 | 0.0260842503904 | 7.02345884985e-06 | -0.000906019712008 | 0.0529862119802 | 0.112200325278 |
| C1 | 1.25539192427 | 0.0206148166596 | 0.028963779094 | -0.000104972164805 | 0.000513326351172 | 0.0624838568208 | 0.125317688064 |
| K0 | 1.30957695101 | 0.0215966484792 | 0.029477253411 | -0.000163815381844 | 0.000354647374147 | 0.0650365091575 | 0.139753329711 |
| K1 | 0.896535015657 | 0.0137156514326 | 0.0272946992691 | 2.61598889821e-05 | -0.0010787647599 | 0.054698158189 | 0.117369896918 |
| Q | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pod_ar | 0.0200133763284 | 0.0002763493104 | 0.000750858373788 | -7.49263901628e-06 | -0.000160573536527 | 0.00171751333317 | 0.00306575882528 |
| gru_seed0 | 0.797403751426 | 0.0122459466663 | 0.0240138272593 | 0.000448431396706 | -0.000554201668346 | 0.0513335837652 | 0.116813009902 |
| gru_seed1 | 0.790971099372 | 0.0119073612746 | 0.0251261089942 | 0.000179760989988 | -0.00172046711679 | 0.0565822329404 | 0.0988390717669 |
| gru_seed2 | 0.787098040379 | 0.0121042356537 | 0.0236096536531 | 0.000257787036236 | -0.00219465265388 | 0.0519417400043 | 0.113842639019 |
| kalman0_gru_seed0 | 1.16777914362 | 0.0192010346282 | 0.0267450957497 | -9.84569864203e-05 | 0.000343165867361 | 0.0600887126045 | 0.136345151248 |
| kalman0_gru_seed1 | 1.18214965517 | 0.0194419172992 | 0.0270376080185 | -0.000242304314987 | 0.000778588002566 | 0.0608210531926 | 0.133423305337 |
| kalman0_gru_seed2 | 1.18194408603 | 0.0194210368238 | 0.0271719369554 | -0.000147783259731 | 0.000898631611841 | 0.0610268486501 | 0.125999264764 |
| kalman1_gru_seed0 | 0.699300672725 | 0.0106781106432 | 0.0214017731038 | 0.000178196051759 | -0.000305197003289 | 0.0464305254699 | 0.0810256717006 |
| kalman1_gru_seed1 | 0.651931166576 | 0.0097761291293 | 0.0209081528488 | -0.000173766218736 | 4.43834563307e-05 | 0.0431872423551 | 0.0792758061533 |
| kalman1_gru_seed2 | 0.706387504049 | 0.0107109706054 | 0.0220299447761 | 0.000372910185197 | 0.00215664411436 | 0.0448325178974 | 0.0833189662644 |

| model | rho median | rho p95 | frac rho<1 | frac rho<0.1 | frac rho>2 | median alpha_parallel | median r_perp |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| C1 | 1.20202940146 | 2.82347680831 | 0.380952380952 | 0 | 0.119047619048 | 0.28733188914 | 0.662179263428 |
| K0 | 1.25566155566 | 3.00795556017 | 0.380952380952 | 0 | 0.119047619048 | 0.513085396707 | 0.816303707872 |
| K1 | 1.03738876992 | 1.15477111467 | 0.309523809524 | 0 | 0 | -0.0287286669341 | 0.100588176342 |
| Q | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| pod_ar | 0.0212975164384 | 0.0607395718994 | 1 | 1 | 0 | 0.999982825889 | 0.0151394443161 |
| gru_seed0 | 0.827060800702 | 1.42164381044 | 0.761904761905 | 0 | 0 | 0.205018917546 | 0.197851743754 |
| gru_seed1 | 0.87111170446 | 1.42549993008 | 0.738095238095 | 0 | 0 | 0.190413894058 | 0.259311823379 |
| gru_seed2 | 0.858656575897 | 1.23987570588 | 0.761904761905 | 0 | 0 | 0.19609284056 | 0.267580086668 |
| kalman0_gru_seed0 | 1.08449921145 | 2.53911938522 | 0.428571428571 | 0 | 0.0952380952381 | 0.537660879161 | 0.760856302534 |
| kalman0_gru_seed1 | 1.09464943277 | 2.79933505872 | 0.404761904762 | 0 | 0.0952380952381 | 0.58562953408 | 0.742815243821 |
| kalman0_gru_seed2 | 1.1321702031 | 2.68267979371 | 0.380952380952 | 0 | 0.119047619048 | 0.567967251792 | 0.746452707064 |
| kalman1_gru_seed0 | 0.771815399261 | 1.24779369161 | 0.785714285714 | 0 | 0 | 0.272980316767 | 0.164267149959 |
| kalman1_gru_seed1 | 0.734471724022 | 1.55543497997 | 0.809523809524 | 0 | 0.047619047619 | 0.342576017635 | 0.22257729237 |
| kalman1_gru_seed2 | 0.803014468264 | 1.45516558977 | 0.809523809524 | 0 | 0.0238095238095 | 0.263883831665 | 0.199821333013 |

- Gate outcomes: {"G0": "FAIL_OFFLINE_GRU_VALUE", "GK0": "FAIL_OFFLINE_KALMAN_GRU_VALUE", "GK1": "PASS_OFFLINE_KALMAN_GRU_VALUE", "overall": "FAIL_OFFLINE_GRU_VALUE_GK_PASS_REQUIRES_REVIEW"}
- Q is numerically reported but excluded from architecture selection, ranking, and every gate.
- G0: median(G0 seed NRMSE) <= 0.95*C0 and <= 0.98*selected-AR; median carry-beating fraction >= 0.60; every seed <= 1.10*C0.
- GK: median(GK seed NRMSE) <= 0.95*matching-Kalman and <= 0.98*median(G0 seed NRMSE); median paired-step fraction >= 0.55; median(GK carry-relative p95 rho) <= 1.10*median(G0 carry-relative p95 rho); at least two strict favorable seeds against both references.
- Gate limits are inclusive; ties are not favorable. AR approximately matching G0 means abs(selected_AR_NRMSE - median(G0 seed NRMSE)) <= 0.02*selected_AR_NRMSE.
- Stop classifications cover G0 fail/GK fail, G0 pass/GK fail, G0 fail/any-GK pass (review required), and G0 pass/any-GK pass.

## Parent review of the predeclared gates

This section records the parent-side review after the one-shot campaign. It does
not change the executed harness, reopen D1, or alter the formal classification.

- G0 median NRMSE is `0.790971099372`. It passes the C0 ratio (`0.884894 <=
  0.95`), carry-beating fraction (`0.761905 >= 0.60`), and worst-seed ratio
  (`0.892091 <= 1.10`), but fails the selected POD-AR comparison by a wide
  margin (`39.5221`, required `<= 0.98`).
- GK0 median NRMSE is `1.18194408603`. Although its ratio to K0 is `0.902539`,
  it is worse than paired G0 (`1.49429`), has paired-step fraction `0.380952`,
  has carry-relative p95 ratio `1.88703`, and has zero strictly favorable
  seeds. GK0 therefore fails.
- GK1 median NRMSE is `0.699300672725`. All frozen hybrid conditions pass:
  ratio to K1 `0.780004`, ratio to median G0 `0.884104`, paired-step fraction
  `0.619048`, carry-relative p95 ratio `1.02358`, and three strictly favorable
  seeds. The formal GK1 result remains `PASS_OFFLINE_KALMAN_GRU_VALUE`.
- The selected POD-AR NRMSE is `0.0200133763284`, only about `2.53%` of median
  G0 error and `2.86%` of median GK1 error. Thus the review-required overall
  result means that Kalman state/innovation helps within the tested neural
  family, while the neural models still have no demonstrated value over the
  simple train-only linear POD-AR baseline.

Parent verification rehashed all 12 final output artifacts with no mismatch,
checked all 15 prediction arrays as finite float64 `(50,128,3)` with exact-zero
x, verified 36 neural training combinations and 40 selection rows, checked all
630 model-step proxy rows for physical steps 9--50, loaded the selected
weights-only state bundle, and revalidated all 11 recorded harness-source
hashes against the post-run working tree.

The required R25A stop boundary is unchanged: no CUDA, R25B/C, no-commit
probe, exact20/exact50, IQN-reuse interaction, online learning, solver
integration, commit, or push is authorized by this result.

## Reproducibility and artifact identity

The base commit alone does not identify this intentionally uncommitted implementation; the exact runtime identity and harness-source SHA256 map are bound into model_config, the final data manifest, and this report.
RUNTIME_IDENTITY_JSON:
{
  "base_commit": "fbf4b729a68fab4c69316568cadcf46f234202d9",
  "base_commit_is_not_implementation_identity": true,
  "cpu_only": true,
  "cuda_available": false,
  "cuda_version": null,
  "dirty_state": "implementation_files_uncommitted",
  "harness_source_sha256": {
    "tools/run_ansys_vertical_flap_gru_study.py": "4b5b4980d0d7896dca07148d97740b19ee40fb3fe8b87d18e9849e47ca44b698",
    "tools/validation/gru_kalman/__init__.py": "123c9ea72169f5e8184882aa565f53fe8f44fd9931550a396fcb543f865a4091",
    "tools/validation/gru_kalman/artifacts.py": "4055be9ecf2db843e1a9e330d379c98fd05d863775e8c493ccdcb0947ddea434",
    "tools/validation/gru_kalman/baselines.py": "d2dddb3a68204ee54003c613dd4d657cd1742037d1d7389688868c80e9d8bae9",
    "tools/validation/gru_kalman/campaign.py": "07c9ab2e437112acca526e9cf324056832c12b1c7aeee26d1666dd052defe346",
    "tools/validation/gru_kalman/dataset.py": "376ce775c5eb1c54426d8c5d286eec5bf0939579cde6eb50b20799242f84c2ce",
    "tools/validation/gru_kalman/evaluation.py": "a07ebd5ddf5cd3d713896c3946e52cdbe6807f0aee221f14281219d00434cdf9",
    "tools/validation/gru_kalman/models.py": "71c0b72ff6f369df509354952ae0f6c7040808579fa1679e93916a1063036eff",
    "tools/validation/gru_kalman/pod.py": "583398ca89490c0e1412ef22709f5fe877176457ca6fe0cdb523f7fa77960fef",
    "tools/validation/gru_kalman/reporting.py": "d3a7988d566138380f7b12461a1a121c655830847b7b5a48260e8528cad6c778",
    "tools/validation/gru_kalman/training.py": "d2cf376947179dcb271a81dfe357af8090ab43af9ced5b75b4029b2ff8623c77"
  },
  "numpy_version": "2.1.2",
  "python_version": "3.10.12",
  "pytorch_version": "2.9.1+cpu",
  "working_tree_dirty": true
}
END_RUNTIME_IDENTITY_JSON

PRE_D1_ARTIFACT_SHA256_JSON:
{
  "model_config.json": "1f397fa7006d47b6384d2067b30dba2031f9feef19c1c9836cf01292cb65a047",
  "model_state.pt": "cd7ddc1b3bd641608b12d79d05ce41479c5e92f3deb3d1414c715ee55830ee59",
  "normalization.json": "d700987623e65b0fab4a4e40cdfe81e17fd0fb71a8422b5147a26dc31d887d4a",
  "pod_ar_state.json": "31dfa7cc8ee87b143877240d555c363721f55e4a0912a5579caf9ca82028d4e6",
  "pod_basis.npz": "c990d6c51ebe4e5271fc83207ae65b00fa90db19e9fc1769ac7a6f790aaf002d",
  "selection_metrics.csv": "e8e50accbdd37c82bfe1a11b96aea4236f93bb1843b1b69a12bc4d1c2568eec0",
  "training_history.csv": "5c8844b4382aa671ca151890e8c37501239aa4153a7e23281390377cb0d8af92"
}
END_PRE_D1_ARTIFACT_SHA256_JSON

FINAL_OUTPUT_ARTIFACT_SHA256_JSON:
{
  "d1_holdout_metrics.csv": "458fb4d72833a8e610dfaaf200775aa5b20ef886b5b644567636088bc080c91a",
  "d1_predictions.npz": "84404f46092513a614f4b1b43acbf23919c8f9341cf0ba1d708ec70b63de66ea",
  "data_split_manifest.json": "6c9818933b8fe24e253d138a91b77877a27a8cfeb61ac2dd29e838628eeabad0",
  "model_config.json": "1f397fa7006d47b6384d2067b30dba2031f9feef19c1c9836cf01292cb65a047",
  "model_state.pt": "cd7ddc1b3bd641608b12d79d05ce41479c5e92f3deb3d1414c715ee55830ee59",
  "normalization.json": "d700987623e65b0fab4a4e40cdfe81e17fd0fb71a8422b5147a26dc31d887d4a",
  "pod_ar_state.json": "31dfa7cc8ee87b143877240d555c363721f55e4a0912a5579caf9ca82028d4e6",
  "pod_basis.npz": "c990d6c51ebe4e5271fc83207ae65b00fa90db19e9fc1769ac7a6f790aaf002d",
  "selection_fingerprint.json": "4fd80ffbac2bf7c676424a38fba90b6dd6624610b77e68d1684d2c023c64b2c9",
  "selection_metrics.csv": "e8e50accbdd37c82bfe1a11b96aea4236f93bb1843b1b69a12bc4d1c2568eec0",
  "threshold_proxy.csv": "32e229dbd397f7a29b20cd5db3dda8ce3361cb88b26cf93873dd87e63a553fbc",
  "training_history.csv": "5c8844b4382aa671ca151890e8c37501239aa4153a7e23281390377cb0d8af92"
}
END_FINAL_OUTPUT_ARTIFACT_SHA256_JSON

## Limitations and stop boundary

Lower offline error is not solver acceleration. R24B oracle/IQN evidence is nondeployable, and K1 was not better than carry in R24.
D1 is the same operating condition already inspected by R24, so this is not unseen cross-condition/generalization evidence.
Out of scope: R25B/C, no-commit probes, exact20/exact50, CUDA, IQN-reuse interaction, online GRU, solver integration, commit, and push.
Initial focused RED: /home/zhuohengli/.venvs/hibm-mpm-r25a-cpu/bin/python -B -m pytest -q -p no:cacheprovider tests/validation/test_gru_kalman_feasibility.py; ModuleNotFoundError: No module named 'tools.validation.gru_kalman' (0 collected).
Review RED covered strict seals, gate fallbacks, missing selection data, incomplete evidence rows, identity binding, trace binding, and ordering.
Focused GREEN verification command: /home/zhuohengli/.venvs/hibm-mpm-r25a-cpu/bin/python -B -m pytest -q -p no:cacheprovider tests/validation/test_gru_kalman_feasibility.py; 35 passed.
Existing R24 Kalman CPU regression command: /home/zhuohengli/.venvs/hibm-mpm-r25a-cpu/bin/python -B -m pytest -q -p no:cacheprovider tests/validation/test_kalman_statistical_calibration.py; 23 passed.
Real D0-only preflight before holdout: sealed manifest and production predictor SHA verified; 200 finite accepted frames with shape (200,128,3) and all 200 frame/history/journal evidence rows matched. D1 was not opened by this preflight.
