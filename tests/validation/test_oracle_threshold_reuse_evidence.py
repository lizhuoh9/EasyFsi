"""Conditional exact8 IQN-reuse evidence contracts for R24C."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_common import (
    OracleThresholdContractError,
)
from tests.validation.test_kalman_oracle_headroom import _write_run


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as frame:
        payload = {name: np.array(frame[name], copy=True) for name in frame.files}
    payload.update(updates)
    np.savez(path, **payload)


def _factor_roots(tmp_path: Path) -> dict[str, Path]:
    c0 = _write_run(
        tmp_path / "carry_off",
        mode="carry_forward",
        iterations=3,
        cg_iterations=300,
        component_wall_s=1.0,
    )
    c1 = _write_run(
        tmp_path / "carry_on",
        mode="carry_forward",
        iterations=3,
        cg_iterations=300,
        component_wall_s=1.0,
    )
    o0 = _write_run(
        tmp_path / "oracle_off",
        mode="oracle_replay",
        oracle_path=c0,
        iterations=3,
        cg_iterations=300,
        component_wall_s=1.0,
    )
    o1 = _write_run(
        tmp_path / "oracle_on",
        mode="oracle_replay",
        oracle_path=c1,
        iterations=3,
        cg_iterations=300,
        component_wall_s=1.0,
    )
    roots = {
        "carry_reuse_off": c0,
        "carry_reuse_on": c1,
        "oracle_reuse_off": o0,
        "oracle_reuse_on": o1,
    }
    for name, root in roots.items():
        manifest_path = root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config"]["iqn_reuse_previous_step_history"] = name.endswith("_on")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        marker_normal = np.tile(
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            (128, 1),
        )
        marker_area = np.full(128, 1.0e-6, dtype=np.float64)
        marker_region = np.arange(128, dtype=np.int64)
        for step in range(1, 9):
            path = root / "step_fields" / f"step_{step:04d}.npz"
            with np.load(path, allow_pickle=False) as frame:
                payload = {name: np.array(frame[name], copy=True) for name in frame.files}
            accepted = np.zeros_like(payload["marker_velocity_mps"], dtype=np.float64)
            guesses = np.zeros_like(payload["iqn_trial_guess_mps"], dtype=np.float64)
            candidates = np.zeros_like(guesses)
            payload.update(
                marker_velocity_mps=accepted,
                iqn_trial_guess_mps=guesses,
                iqn_trial_candidate_mps=candidates,
                iqn_trial_residual_mps=candidates - guesses,
                marker_normal=marker_normal,
                marker_area_m2=marker_area,
                marker_region_id=marker_region,
            )
            np.savez(path, **payload)
    return roots


def test_factor_loader_retains_physical_marker_union_across_four_arms(
    tmp_path: Path,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    roots = _factor_roots(tmp_path)
    modes = {
        "carry_reuse_off": ("carry_forward", False),
        "carry_reuse_on": ("carry_forward", True),
        "oracle_reuse_off": ("oracle_replay", False),
        "oracle_reuse_on": ("oracle_replay", True),
    }
    runs = {
        name: subject._load_factor_run(
            root,
            expected_mode=modes[name][0],
            reuse_enabled=modes[name][1],
        )
        for name, root in roots.items()
    }
    expected = {
        "marker_position_m",
        "marker_velocity_mps",
        "marker_normal",
        "marker_area_m2",
        "marker_region_id",
        "iqn_trial_guess_mps",
        "iqn_trial_candidate_mps",
        "iqn_trial_residual_mps",
    }
    for run in runs.values():
        assert expected <= set(run.steps[0].arrays)
    subject._validate_factor_marker_consistency(tuple(runs.values()))


def test_factor_marker_consistency_allows_bounded_dynamic_normal_drift(
    tmp_path: Path,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    roots = _factor_roots(tmp_path)
    path = roots["carry_reuse_on"] / "step_fields" / "step_0004.npz"
    with np.load(path, allow_pickle=False) as frame:
        marker_normal = np.array(frame["marker_normal"], copy=True)
    marker_normal[:, 0] = 2.5e-5
    marker_normal[:, 1] = np.sqrt(1.0 - marker_normal[:, 0] ** 2)
    _rewrite_npz(path, marker_normal=marker_normal)

    runs = {
        name: subject._load_factor_run(
            root,
            expected_mode=(
                "carry_forward" if name.startswith("carry") else "oracle_replay"
            ),
            reuse_enabled=name.endswith("_on"),
        )
        for name, root in roots.items()
    }

    subject._validate_factor_marker_consistency(tuple(runs.values()))


@pytest.mark.parametrize("violated_gate", ("max_abs", "nrmse"))
def test_factor_marker_consistency_rejects_independent_normal_gate_violation(
    tmp_path: Path,
    violated_gate: str,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    roots = _factor_roots(tmp_path)
    path = roots["carry_reuse_on"] / "step_fields" / "step_0004.npz"
    with np.load(path, allow_pickle=False) as frame:
        marker_normal = np.array(frame["marker_normal"], copy=True)
    if violated_gate == "max_abs":
        marker_normal[0, 0] = 5.1e-5
        marker_normal[0, 1] = np.sqrt(1.0 - marker_normal[0, 0] ** 2)
    else:
        marker_normal[:, 0] = 4.0e-5
        marker_normal[:, 2] = 4.0e-5
        marker_normal[:, 1] = np.sqrt(
            1.0 - marker_normal[:, 0] ** 2 - marker_normal[:, 2] ** 2
        )
    reference_normal = np.tile(
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        (128, 1),
    )
    difference = marker_normal.astype(np.float64) - reference_normal.astype(
        np.float64
    )
    nrmse = float(np.sqrt(np.mean(np.square(difference)))) / float(
        np.sqrt(np.mean(np.square(reference_normal.astype(np.float64))))
    )
    max_abs = float(np.max(np.abs(difference)))
    if violated_gate == "max_abs":
        assert nrmse < subject.PREFIX_REPLAY_NRMSE_MAX
        assert max_abs > subject._PREFIX_MAX_ABS_BY_ARRAY["marker_normal"]
    else:
        assert nrmse > subject.PREFIX_REPLAY_NRMSE_MAX
        assert max_abs < subject._PREFIX_MAX_ABS_BY_ARRAY["marker_normal"]
    _rewrite_npz(path, marker_normal=marker_normal)

    runs = {
        name: subject._load_factor_run(
            root,
            expected_mode=(
                "carry_forward" if name.startswith("carry") else "oracle_replay"
            ),
            reuse_enabled=name.endswith("_on"),
        )
        for name, root in roots.items()
    }

    with pytest.raises(
        OracleThresholdContractError,
        match="marker_normal exceeds replay bounds",
    ):
        subject._validate_factor_marker_consistency(tuple(runs.values()))


@pytest.mark.parametrize(
    "tampered_key",
    ("marker_area_m2", "marker_region_id", "layout"),
)
def test_factor_marker_identity_tampering_fails_closed(
    tmp_path: Path,
    tampered_key: str,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    roots = _factor_roots(tmp_path)
    runs = {
        name: subject._load_factor_run(
            root,
            expected_mode=(
                "carry_forward" if name.startswith("carry") else "oracle_replay"
            ),
            reuse_enabled=name.endswith("_on"),
        )
        for name, root in roots.items()
    }
    path = roots["carry_reuse_on"] / "step_fields" / "step_0004.npz"
    if tampered_key == "layout":
        _rewrite_npz(
            path,
            iqn_trial_layout_sha256=np.asarray("2" * 64),
        )
        with pytest.raises(OracleThresholdContractError, match="layout identity"):
            subject._load_factor_run(
                roots["carry_reuse_on"],
                expected_mode="carry_forward",
                reuse_enabled=True,
            )
        return

    with np.load(path, allow_pickle=False) as frame:
        update = np.array(frame[tampered_key], copy=True)
    if tampered_key == "marker_area_m2":
        update[:] = 2.0e-6
    else:
        update[:] += 1000
    for step in range(1, 9):
        _rewrite_npz(
            roots["carry_reuse_on"] / "step_fields" / f"step_{step:04d}.npz",
            **{tampered_key: update},
        )
    runs["carry_reuse_on"] = subject._load_factor_run(
        roots["carry_reuse_on"],
        expected_mode="carry_forward",
        reuse_enabled=True,
    )
    with pytest.raises(
        OracleThresholdContractError,
        match=tampered_key,
    ):
        subject._validate_factor_marker_consistency(tuple(runs.values()))


def test_factor_marker_region_common_step_drift_fails_closed(
    tmp_path: Path,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    roots = _factor_roots(tmp_path)
    for root in roots.values():
        path = root / "step_fields" / "step_0004.npz"
        with np.load(path, allow_pickle=False) as frame:
            marker_region = np.array(frame["marker_region_id"], copy=True)
        _rewrite_npz(path, marker_region_id=marker_region + 1000)

    with pytest.raises(
        OracleThresholdContractError,
        match="marker_region_id",
    ):
        tuple(
            subject._load_factor_run(
                root,
                expected_mode=(
                    "carry_forward"
                    if name.startswith("carry")
                    else "oracle_replay"
                ),
                reuse_enabled=name.endswith("_on"),
            )
            for name, root in roots.items()
        )


def test_threshold_context_rejects_artifact_change_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    root = tmp_path / "threshold"
    root.mkdir()
    response = root / "oracle_threshold_response.json"
    response.write_text(
        json.dumps({"classification": "PASS_ORACLE_THRESHOLD_MATRIX"}),
        encoding="utf-8",
    )
    (root / "oracle_threshold_source_manifest.json").write_text(
        json.dumps({"execution_source": {}}),
        encoding="utf-8",
    )
    (root / "oracle_threshold_summary.json").write_text("{}", encoding="utf-8")

    def mutate_during_verify(_snapshot_root: Path) -> dict[str, object]:
        response.write_text(
            json.dumps(
                {"classification": "PASS_ORACLE_THRESHOLD_MATRIX", "changed": True}
            ),
            encoding="utf-8",
        )
        return {"artifact_sha256": {}}

    monkeypatch.setattr(subject, "verify_threshold_evidence", mutate_during_verify)
    with pytest.raises(
        OracleThresholdContractError,
        match="changed during verification",
    ):
        subject._load_threshold_context(root)


def _threshold_context(*, authorized: bool) -> dict[str, object]:
    return {
        "response": {
            "classification": "PASS_ORACLE_THRESHOLD_MATRIX",
            "reuse_branch": {
                "authorized": authorized,
                "status": (
                    "reuse_matrix_authorized"
                    if authorized
                    else "reuse_matrix_not_authorized"
                ),
                "reason": (
                    "safe_higher_first_picard_relaxation"
                    if authorized
                    else "no_safe_omega"
                ),
            },
            "best_safe_omega": 0.5 if not authorized else 0.75,
            "omega_summary": [
                {
                    "omega": 0.5,
                    "safe": not authorized,
                    "selection_rank_key": [0, 0, 0],
                },
                {
                    "omega": 0.75,
                    "safe": authorized,
                    "selection_rank_key": [-2, 1, 2],
                },
                {
                    "omega": 1.0,
                    "safe": False,
                    "selection_rank_key": [0, 9, 27],
                },
            ],
        },
        "manifest": {
            "execution_source": {
                "mode": "source_map_bound_working_tree",
                "git_head_commit": "a" * 40,
                "source_count": 140,
                "source_map_sha256": "b" * 64,
            },
            "q0_roots": {"0.5": "/q0-050", "0.75": "/q0-075", "1.0": "/q0-100"},
        },
        "artifact_sha256": {
            "oracle_threshold_response.json": "c" * 64,
            "oracle_threshold_source_manifest.json": "d" * 64,
            "oracle_threshold_summary.json": "e" * 64,
        },
    }


def test_reuse_evidence_records_terminal_not_authorized_without_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    context = _threshold_context(authorized=False)
    monkeypatch.setattr(subject, "_load_threshold_context", lambda _root: context)
    output = tmp_path / "oracle_reuse_response.json"

    artifact_sha = subject.write_reuse_evidence(
        tmp_path / "threshold",
        None,
        output,
    )
    verified = subject.verify_reuse_evidence(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "reuse_matrix_not_authorized"
    assert payload["reason"] == "no_safe_omega"
    assert payload["run_roots"] == {}
    assert payload["matrix"] is None
    assert verified["artifact_sha256"] == artifact_sha

    payload["reason"] = "tampered"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OracleThresholdContractError, match="self SHA"):
        subject.verify_reuse_evidence(output)


def test_authorized_reuse_matrix_requires_exact_four_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    monkeypatch.setattr(
        subject,
        "_load_threshold_context",
        lambda _root: _threshold_context(authorized=True),
    )
    with pytest.raises(
        OracleThresholdContractError,
        match="exactly four factor arms",
    ):
        subject.write_reuse_evidence(
            tmp_path / "threshold",
            {"carry_reuse_off": tmp_path / "c0"},
            tmp_path / "oracle_reuse_response.json",
        )


def test_authorized_reuse_matrix_is_recomputed_and_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    context = _threshold_context(authorized=True)
    roots = {
        "carry_reuse_off": (tmp_path / "c0").resolve(),
        "carry_reuse_on": (tmp_path / "c1").resolve(),
        "oracle_reuse_off": (tmp_path / "o0").resolve(),
        "oracle_reuse_on": (tmp_path / "o1").resolve(),
    }
    for root in roots.values():
        root.mkdir()
    matrix = {
        "classification": "PASS_IQN_REUSE_FACTOR_MATRIX",
        "selected_omega": 0.75,
        "arms": {
            name: {
                "coupling_iterations_total": 24 if name.endswith("off") else 16,
                "reuse_used_steps": [] if name.endswith("off") else [2, 3],
            }
            for name in roots
        },
    }
    monkeypatch.setattr(subject, "_load_threshold_context", lambda _root: context)
    calls: list[dict[str, Path]] = []

    def analyze(
        loaded_context: dict[str, object],
        loaded_roots: dict[str, Path],
    ) -> dict[str, object]:
        assert loaded_context is context
        calls.append(loaded_roots)
        return matrix

    monkeypatch.setattr(subject, "_analyze_authorized_matrix", analyze)
    output = tmp_path / "oracle_reuse_response.json"

    subject.write_reuse_evidence(tmp_path / "threshold", roots, output)
    verified = subject.verify_reuse_evidence(output)

    assert verified["classification"] == "PASS_IQN_REUSE_FACTOR_MATRIX"
    assert len(calls) == 2
    assert calls[0] == roots
    assert calls[1] == roots


def test_reuse_chain_keeps_private_secants_for_next_step_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_reuse_evidence as subject,
    )

    steps = []
    for step_index in (1, 2):
        frame = tmp_path / f"step_{step_index:04d}.npz"
        history = tmp_path / f"step_{step_index:04d}.json"
        frame.write_bytes(f"frame-{step_index}".encode())
        history.write_bytes(f"history-{step_index}".encode())
        steps.append(
            SimpleNamespace(
                step=step_index,
                arrays={},
                history={"hibm_iqn_reuse": {"first_update_mode": "iqn_ils_reuse"}},
                frame_path=frame,
                history_path=history,
            )
        )
    run = SimpleNamespace(steps=tuple(steps))
    monkeypatch.setattr(
        subject,
        "validate_iqn_trial_vector_frame",
        lambda arrays, *, step, marker_count, layout_sha256: {
            "layout": "a" * 64,
        },
    )

    def validate(
        history: dict[str, object],
        trace: dict[str, object],
        step: int,
        *,
        prior_reports: list[dict[str, object]],
        initial_picard_relaxation: float,
    ) -> dict[str, object]:
        assert initial_picard_relaxation == 0.75
        if step == 2:
            assert np.array_equal(
                prior_reports[0]["_delta_residual"],
                np.asarray([[1.0]]),
            )
            assert np.array_equal(
                prior_reports[0]["_delta_candidate"],
                np.asarray([[2.0]]),
            )
        return {
            "step": step,
            "used": step == 2,
            "source_step": None if step == 1 else 1,
            "_delta_residual": np.asarray([[1.0]]),
            "_delta_candidate": np.asarray([[2.0]]),
        }

    monkeypatch.setattr(subject, "_validate_reuse_report", validate)

    public = subject._validate_reuse_chain(
        run,
        enabled=True,
        initial_picard_relaxation=0.75,
    )

    assert len(public) == 2
    assert "_delta_residual" not in public[0]
    assert public[1]["source_step"] == 1
