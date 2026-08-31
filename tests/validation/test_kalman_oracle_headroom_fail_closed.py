"""Adversarial fail-closed checks for the R24B evidence boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.refactored.validation.ansys_vertical_flap_fsi import (
    kalman_oracle_headroom as subject,
)
from tests.validation.test_kalman_oracle_headroom import (
    _write_json,
    paired_runs,
)


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as frame:
        payload = {name: np.array(frame[name], copy=True) for name in frame.files}
    payload.update(updates)
    np.savez(path, **payload)


def test_preflow_source_identity_must_match_current_executable_surface(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    manifest = json.loads((q0 / "run_manifest.json").read_text(encoding="utf-8"))
    snapshot = Path(manifest["config"]["preflow_snapshot_input_path"] + ".json")
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["identity"]["source_sha256"] = "6" * 64
    _write_json(snapshot, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="preflow source"):
        subject.analyze_oracle_headroom(q0, q3)


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("marker_position_m", (2, 3)),
        ("solid_position_m", (3, 3)),
        ("u", (2, 2)),
    ],
)
def test_frozen_step_array_shapes_are_absolute(
    paired_runs: tuple[Path, Path],
    field: str,
    shape: tuple[int, ...],
) -> None:
    for root in paired_runs:
        path = root / "step_fields" / "step_0002.npz"
        _rewrite_npz(path, **{field: np.zeros(shape, dtype=np.float32)})

    with pytest.raises(subject.OracleHeadroomContractError, match="frozen shape"):
        subject.analyze_oracle_headroom(*paired_runs)


def test_closure_tolerance_cannot_be_raised_inside_evidence(
    paired_runs: tuple[Path, Path],
) -> None:
    for root in paired_runs:
        path = root / "step_history" / "step_0003.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload["history"]
        history["flow_hibm_marker_compatibility_closure_tolerance_mps"] = 0.1
        closure = history["canonical_velocity_dirichlet_report"][
            "marker_target_closure"
        ]
        closure["closure_tolerance_mps"] = 0.1
        closure["final_max_residual_mps"] = 0.05
        _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="closure tolerance"):
        subject.analyze_oracle_headroom(*paired_runs)


def test_missing_projection_invalid_axis_count_fails_closed(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    path = q3 / "step_history" / "step_0004.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["history"]["canonical_velocity_dirichlet_report"][
        "marker_target_closure"
    ]["projection_only_invalid_axis_count"]
    _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="invalid closure"):
        subject.analyze_oracle_headroom(q0, q3)


def test_q0_carry_forward_cannot_name_an_oracle_path(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    path = q0 / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["initial_guess_oracle_path"] = str(q3)
    _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="Q0 oracle path"):
        subject.analyze_oracle_headroom(q0, q3)
