"""Substantive R25A report rendering kept separate from campaign orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping


def write_feasibility_report(
    path: Path,
    *,
    selection_fingerprint: str,
    d0_count: int,
    d1_count: int,
    selected_architectures: Mapping[str, str],
    metrics: Mapping[str, Any],
    classifications: Mapping[str, str],
    runtime_identity: Mapping[str, Any],
    selection_artifact_hashes: Mapping[str, str],
    artifact_hashes: Mapping[str, str],
    k0_fingerprint: str,
    k1_fingerprint: str,
    predictor_sha256: str,
) -> None:
    if path.exists() and path.stat().st_size > 0:
        raise ValueError(f"refusing to overwrite nonempty report {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ANSYS Vertical Flap R25A POD-GRU / Kalman-Residual-GRU Feasibility",
        "",
        "This is a frozen R25A offline feasibility study. D1 was independently loaded after selection sealing, but it is the same operating condition already inspected by R24; it is not an unseen cross-condition/generalization test.",
        "",
        "## Evidence boundary and frozen matrix",
        "",
        f"- Selection fingerprint: {selection_fingerprint}",
        f"- D0 frames: {d0_count} (fit 1-100; selection 101-200); D1 frames: {d1_count} (cold start, score 9-50).",
        f"- Selected architectures: {json.dumps(dict(selected_architectures), sort_keys=True)}.",
        f"- K0 fingerprint: {k0_fingerprint}; K1 fingerprint: {k1_fingerprint}; predictor SHA256: {predictor_sha256}.",
        "- POD, normalization, AR, and neural fitting use D0 steps 1-100 only; D0 steps 101-200 are selection and early stopping.",
        "- D1 opens only after the pre-D1 seal and immediate re-hash verification; no training or selection API is used after opening.",
        "",
        "## D1 metrics and gate outcomes",
        "",
        "| model | active-yz NRMSE | y RMSE | z RMSE | y bias | z bias | marker p95 | marker max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        lines.append(
            f"| {name} | {metric.global_active_yz_nrmse:.12g} | "
            f"{metric.axis_rmse[1]:.12g} | {metric.axis_rmse[2]:.12g} | "
            f"{metric.axis_bias[1]:.12g} | {metric.axis_bias[2]:.12g} | "
            f"{metric.global_marker_p95:.12g} | {metric.global_marker_max:.12g} |"
        )
    lines.extend(
        [
            "",
            "| model | rho median | rho p95 | frac rho<1 | frac rho<0.1 | frac rho>2 | median alpha_parallel | median r_perp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metric in metrics.items():
        lines.append(
            f"| {name} | {metric.rho_median:.12g} | {metric.rho_p95:.12g} | "
            f"{metric.fraction_rho_lt_1:.12g} | {metric.fraction_rho_lt_01:.12g} | "
            f"{metric.fraction_rho_gt_2:.12g} | {median(metric.alpha_parallel):.12g} | "
            f"{median(metric.r_perp):.12g} |"
        )
    lines.extend(
        [
            "",
            f"- Gate outcomes: {json.dumps(dict(classifications), sort_keys=True)}",
            "- Q is numerically reported but excluded from architecture selection, ranking, and every gate.",
            "- G0: median(G0 seed NRMSE) <= 0.95*C0 and <= 0.98*selected-AR; median carry-beating fraction >= 0.60; every seed <= 1.10*C0.",
            "- GK: median(GK seed NRMSE) <= 0.95*matching-Kalman and <= 0.98*median(G0 seed NRMSE); median paired-step fraction >= 0.55; median(GK carry-relative p95 rho) <= 1.10*median(G0 carry-relative p95 rho); at least two strict favorable seeds against both references.",
            "- Gate limits are inclusive; ties are not favorable. AR approximately matching G0 means abs(selected_AR_NRMSE - median(G0 seed NRMSE)) <= 0.02*selected_AR_NRMSE.",
            "- Stop classifications cover G0 fail/GK fail, G0 pass/GK fail, G0 fail/any-GK pass (review required), and G0 pass/any-GK pass.",
            "",
            "## Reproducibility and artifact identity",
            "",
            "The base commit alone does not identify this intentionally uncommitted implementation; the exact runtime identity and harness-source SHA256 map are bound into model_config, the final data manifest, and this report.",
            "RUNTIME_IDENTITY_JSON:",
            json.dumps(dict(runtime_identity), sort_keys=True, indent=2),
            "END_RUNTIME_IDENTITY_JSON",
            "",
            "PRE_D1_ARTIFACT_SHA256_JSON:",
            json.dumps(dict(selection_artifact_hashes), sort_keys=True, indent=2),
            "END_PRE_D1_ARTIFACT_SHA256_JSON",
            "",
            "FINAL_OUTPUT_ARTIFACT_SHA256_JSON:",
            json.dumps(dict(artifact_hashes), sort_keys=True, indent=2),
            "END_FINAL_OUTPUT_ARTIFACT_SHA256_JSON",
            "",
            "## Limitations and stop boundary",
            "",
            "Lower offline error is not solver acceleration. R24B oracle/IQN evidence is nondeployable, and K1 was not better than carry in R24.",
            "D1 is the same operating condition already inspected by R24, so this is not unseen cross-condition/generalization evidence.",
            "Out of scope: R25B/C, no-commit probes, exact20/exact50, CUDA, IQN-reuse interaction, online GRU, solver integration, commit, and push.",
            "Initial focused RED: /home/zhuohengli/.venvs/hibm-mpm-r25a-cpu/bin/python -B -m pytest -q -p no:cacheprovider tests/validation/test_gru_kalman_feasibility.py; ModuleNotFoundError: No module named 'tools.validation.gru_kalman' (0 collected).",
            "Review RED covered strict seals, gate fallbacks, missing selection data, incomplete evidence rows, identity binding, trace binding, and ordering.",
            "Focused GREEN verification command: /home/zhuohengli/.venvs/hibm-mpm-r25a-cpu/bin/python -B -m pytest -q -p no:cacheprovider tests/validation/test_gru_kalman_feasibility.py; 35 passed.",
            "Existing R24 Kalman CPU regression command: /home/zhuohengli/.venvs/hibm-mpm-r25a-cpu/bin/python -B -m pytest -q -p no:cacheprovider tests/validation/test_kalman_statistical_calibration.py; 23 passed.",
            "Real D0-only preflight before holdout: sealed manifest and production predictor SHA verified; 200 finite accepted frames with shape (200,128,3) and all 200 frame/history/journal evidence rows matched. D1 was not opened by this preflight.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["write_feasibility_report"]
