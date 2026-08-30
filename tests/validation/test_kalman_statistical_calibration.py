"""R24 contracts for solver-independent Kalman diagnosis and calibration."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.refactored.validation.ansys_vertical_flap_fsi import (
    kalman_statistical_calibration as subject,
)


_LAYOUT = "a" * 64


def _trajectory(frame_count: int = 12, *, multiplier: float = 1.0) -> np.ndarray:
    step = np.arange(1, frame_count + 1, dtype=np.float64)[:, None, None]
    marker = np.arange(2, dtype=np.float64)[None, :, None]
    axis = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)[None, None, :]
    oscillation = np.sin(step * np.asarray([0.2, 0.3, 0.5])[None, None, :])
    return multiplier * axis * (0.01 * step * step + 0.002 * marker + oscillation)


def _trace(
    frame_count: int = 12,
    *,
    values: np.ndarray | None = None,
    name: str = "synthetic",
    layout_id: str = _LAYOUT,
    source_fingerprint: str = "b" * 64,
) -> subject.AcceptedTrace:
    return subject.AcceptedTrace.synthetic(
        _trajectory(frame_count) if values is None else values,
        name=name,
        dt_s=0.1,
        layout_id=layout_id,
        source_fingerprint=source_fingerprint,
    )


def _candidate(
    *,
    candidate_id: str = "K2",
    model: str = "constant_rate",
    scale_xyz: tuple[float, float, float] = (1.0, 2.0, 4.0),
    q_xyz: tuple[float, float, float] = (0.1, 0.2, 0.3),
    r_xyz: tuple[float, float, float] = (0.2, 0.3, 0.4),
    p0_value_xyz: tuple[float, float, float] | None = None,
    p0_rate_xyz: tuple[float, float, float] | None = None,
    warmup_accepted_states: int = 2,
    beta: float = 1.0,
) -> subject.CandidateSpec:
    return subject.CandidateSpec(
        candidate_id=candidate_id,
        model=model,
        axis_order=("x", "y", "z"),
        scale_xyz=scale_xyz,
        q_xyz=q_xyz,
        r_xyz=r_xyz,
        p0_value_xyz=(r_xyz if p0_value_xyz is None else p0_value_xyz),
        p0_rate_xyz=(
            tuple(value / 0.01 for value in r_xyz)
            if p0_rate_xyz is None
            else p0_rate_xyz
        ),
        warmup_accepted_states=warmup_accepted_states,
        beta=beta,
    )


def test_deterministic_replay_is_byte_stable() -> None:
    trace = _trace()
    candidate = _candidate()

    first = subject.replay_candidate(trace, candidate)
    second = subject.replay_candidate(trace, candidate)

    assert first.to_json_bytes() == second.to_json_bytes()
    assert first.candidate_fingerprint == candidate.fingerprint


def test_replay_source_hashes_align_with_the_n_minus_one_state() -> None:
    trace = _trace(frame_count=4)
    replay = subject.replay_candidate(trace, _candidate())

    assert replay.rows[0].accepted_state_source_step == 0
    assert len(replay.rows[0].accepted_state_source_sha256) == 64
    assert replay.rows[0].accepted_state_source_sha256 != trace.frame_sha256[0]
    assert replay.rows[0].accepted_measurement_sha256 == trace.frame_sha256[0]
    for index, row in enumerate(replay.rows[1:], start=1):
        assert row.accepted_state_source_step == trace.source_steps[index] - 1
        assert row.accepted_state_source_sha256 == trace.frame_sha256[index - 1]
        assert row.accepted_measurement_sha256 == trace.frame_sha256[index]


def test_dimensionless_calibration_is_invariant_to_axis_unit_scaling() -> None:
    original = _trace(frame_count=20)
    factors = np.asarray([1000.0, 0.01, 10.0])[None, None, :]
    rescaled = _trace(frame_count=20, values=original.values * factors)

    candidate_a = subject.calibrate_kalman_candidate(
        original, model="constant_rate", candidate_id="K2"
    )
    candidate_b = subject.calibrate_kalman_candidate(
        rescaled, model="constant_rate", candidate_id="K2"
    )
    score_a = subject.score_replay(subject.replay_candidate(original, candidate_a))
    score_b = subject.score_replay(subject.replay_candidate(rescaled, candidate_b))

    np.testing.assert_allclose(
        score_a.axis_normalized_rmse,
        score_b.axis_normalized_rmse,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        score_a.axis_nis_mean,
        score_b.axis_nis_mean,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_axis_permutation_and_layout_schema_are_rejected() -> None:
    with pytest.raises(subject.CalibrationContractError, match="axis_order"):
        subject.AcceptedTrace.synthetic(
            _trajectory(),
            name="permuted",
            dt_s=0.1,
            layout_id=_LAYOUT,
            axis_order=("y", "x", "z"),
            source_fingerprint="b" * 64,
        )

    trace = _trace()
    with pytest.raises(subject.CalibrationContractError, match="layout"):
        subject.replay_candidate(
            trace,
            replace(_candidate(), layout_id="c" * 64),
        )


def test_stale_n_minus_2_source_is_rejected() -> None:
    candidate = _candidate(warmup_accepted_states=1)
    engine = subject.KalmanTrialEngine(
        candidate,
        initial_values=np.zeros((2, 3)),
        committed_step=1,
        layout_id=_LAYOUT,
    )

    with pytest.raises(subject.CalibrationContractError, match="source_step"):
        engine.begin_step(
            target_step=2,
            accepted_state_source_step=0,
            dt_s=0.1,
            layout_id=_LAYOUT,
        )


def test_rejected_trial_does_not_mutate_committed_state() -> None:
    candidate = _candidate(warmup_accepted_states=1)
    engine = subject.KalmanTrialEngine(
        candidate,
        initial_values=np.zeros((2, 3)),
        committed_step=0,
        layout_id=_LAYOUT,
    )
    first = engine.begin_step(
        target_step=1,
        accepted_state_source_step=0,
        dt_s=0.1,
        layout_id=_LAYOUT,
    )
    engine.assimilate(np.full((2, 3), 999.0), accepted_step=1, layout_id=_LAYOUT)
    engine.discard_trial()

    repeated = engine.begin_step(
        target_step=1,
        accepted_state_source_step=0,
        dt_s=0.1,
        layout_id=_LAYOUT,
    )
    np.testing.assert_array_equal(first.values, repeated.values)


def test_covariance_remains_finite_symmetric_and_psd() -> None:
    replay = subject.replay_candidate(_trace(), _candidate())
    for row in replay.rows:
        assert row.covariance_finite
        assert row.covariance_symmetry_error <= 1.0e-12
        assert row.covariance_min_eigenvalue >= -1.0e-12
        assert np.isfinite(row.innovation_variance_mean)
        assert row.innovation_variance_mean > 0.0


def test_k0_replay_matches_the_production_predictor_without_importing_taichi() -> None:
    trace = _trace(frame_count=8)
    source = Path("simulation_core/coupling/interface_kalman_predictor.py").resolve()
    config = {
        "rate_process_noise_spectral_density": [0.0, 2.0, 4.0],
        "measurement_variance": [0.1, 0.2, 0.3],
        "initial_value_variance": [0.1, 0.2, 0.3],
        "initial_rate_variance": [1.0, 2.0, 3.0],
        "warmup_accepted_states": 3,
    }

    before = set(sys.modules)
    actual = subject.replay_production_k0(trace, config, source)

    spec = importlib.util.spec_from_file_location("_expected_r24_k0", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected_config = module.InterfaceKalmanConfig(
        rate_process_noise_spectral_density=tuple(
            config["rate_process_noise_spectral_density"]
        ),
        measurement_variance=tuple(config["measurement_variance"]),
        initial_value_variance=tuple(config["initial_value_variance"]),
        initial_rate_variance=tuple(config["initial_rate_variance"]),
        warmup_accepted_states=config["warmup_accepted_states"],
    )
    predictor = module.InterfaceKalmanPredictor(expected_config)
    predictor.initialize(np.zeros_like(trace.values[0]), layout_id=trace.layout_id)
    for row, measurement in zip(actual.rows, trace.values, strict=True):
        estimate = predictor.predict_trial(dt=trace.dt_s, layout_id=trace.layout_id)
        update = predictor.update_trial(measurement, layout_id=trace.layout_id)
        predictor.commit_trial()
        assert row.raw_prediction_rms_mps == pytest.approx(
            float(np.sqrt(np.mean(np.square(estimate.values - measurement)))),
            abs=1.0e-15,
        )
        assert row.nis_mean == pytest.approx(
            float(np.mean(update.normalized_innovation_squared)),
            abs=1.0e-12,
        )
    assert "taichi" not in set(sys.modules) - before


def test_held_out_data_cannot_change_frozen_candidate_fingerprints() -> None:
    d0 = _trace(frame_count=20, name="D0")
    frozen = subject.freeze_candidate_matrix(d0, fit_stop=10)
    d1_a = _trace(frame_count=8, name="D1-a")
    d1_b = _trace(frame_count=8, name="D1-b", values=_trajectory(8) * 1000.0)

    first = subject.evaluate_frozen_candidates(d1_a, frozen)
    second = subject.evaluate_frozen_candidates(d1_b, frozen)

    assert first.candidate_fingerprints == second.candidate_fingerprints
    assert first.candidate_fingerprints == tuple(
        candidate.fingerprint for candidate in frozen
    )


def test_source_config_and_layout_fingerprints_fail_closed() -> None:
    trace = _trace()
    candidate = _candidate()
    fingerprint = subject.analysis_fingerprint(trace, candidate)

    subject.verify_analysis_fingerprint(trace, candidate, fingerprint)
    with pytest.raises(subject.CalibrationContractError, match="fingerprint"):
        subject.verify_analysis_fingerprint(
            replace(trace, source_fingerprint="d" * 64),
            candidate,
            fingerprint,
        )
    with pytest.raises(subject.CalibrationContractError, match="fingerprint"):
        subject.verify_analysis_fingerprint(
            trace,
            replace(candidate, q_xyz=(9.0, 9.0, 9.0)),
            fingerprint,
        )


def test_checkpoint_restore_matches_uninterrupted_replay() -> None:
    trace = _trace(frame_count=12)
    candidate = _candidate()
    uninterrupted = subject.replay_candidate(trace, candidate)

    prefix = subject.replay_candidate(trace, candidate, stop_index=5)
    suffix = subject.replay_candidate(
        trace,
        candidate,
        snapshot=prefix.snapshot,
        start_index=5,
    )

    assert [row.to_payload() for row in uninterrupted.rows] == [
        row.to_payload() for row in prefix.rows + suffix.rows
    ]


def test_ranking_and_tie_breaking_are_deterministic() -> None:
    trace = _trace(frame_count=20)
    candidates = (
        _candidate(candidate_id="C1b", model="linear", beta=0.8),
        _candidate(candidate_id="C1a", model="linear", beta=0.8),
        _candidate(candidate_id="C0", model="carry"),
    )
    replays = tuple(subject.replay_candidate(trace, candidate) for candidate in candidates)

    first = subject.rank_candidates(replays)
    second = subject.rank_candidates(tuple(reversed(replays)))

    assert first.to_payload() == second.to_payload()
    tied = [
        row.candidate_id
        for row in first.rows
        if row.candidate_id in {"C1a", "C1b"}
    ]
    assert tied == ["C1a", "C1b"]


def test_huge_r_near_zero_gain_cannot_win_by_benign_nis() -> None:
    trace = _trace(frame_count=20)
    honest = _candidate(candidate_id="K2-honest", warmup_accepted_states=1)
    degenerate = _candidate(
        candidate_id="K2-huge-r",
        q_xyz=(1.0e-20, 1.0e-20, 1.0e-20),
        r_xyz=(1.0e20, 1.0e20, 1.0e20),
        p0_value_xyz=(1.0, 1.0, 1.0),
        p0_rate_xyz=(1.0, 1.0, 1.0),
        warmup_accepted_states=1,
    )
    ranking = subject.rank_candidates(
        (
            subject.replay_candidate(trace, degenerate),
            subject.replay_candidate(trace, honest),
        )
    )

    huge_r = next(row for row in ranking.rows if row.candidate_id == "K2-huge-r")
    assert not huge_r.eligible
    assert huge_r.exclusion_reason == "degenerate_gain"
    assert ranking.rows[0].candidate_id != "K2-huge-r"


def test_axis_specific_nis_failure_cannot_be_hidden_by_pooled_statistics() -> None:
    replay = subject.replay_candidate(
        _trace(frame_count=5),
        _candidate(warmup_accepted_states=1),
    )
    rows = []
    for index, row in enumerate(replay.rows):
        y_values = (21.0, 21.0) if index == 0 else (0.0, 0.0)
        rows.append(
            replace(
                row,
                active_axes=(False, True, True),
                nis_axis_mean=(0.0, float(np.mean(y_values)), 0.1),
                nis_axis_exceedance_fraction=(
                    0.0,
                    float(np.mean(np.asarray(y_values) > 3.841458820694124)),
                    0.0,
                ),
                nis_dof_by_axis=((0.0, 0.0), y_values, (0.1, 0.1)),
                gain_axis_mean=(0.0, 0.5, 0.5),
                gain_dof_by_axis=((0.0, 0.0), (0.5, 0.5), (0.5, 0.5)),
            )
        )
    masked = replace(replay, rows=tuple(rows))

    score = subject.score_replay(masked)

    assert score.nis_mean == pytest.approx(2.15)
    assert score.nis_exceedance_fraction == pytest.approx(0.1)
    assert score.axis_nis_mean[1] == pytest.approx(4.2)
    assert not score.statistically_consistent


def test_missing_or_inconsistent_canonical_attempt_provenance_is_blocked(
    tmp_path: Path,
) -> None:
    with pytest.raises(subject.EvidenceBlocked) as raised:
        subject.load_accepted_trace(
            tmp_path / "missing-canonical",
            tmp_path / "missing-attempt",
            name="D0",
            expected_steps=200,
        )

    assert raised.value.exit_classification == (
        "BLOCKED_MISSING_CALIBRATION_EVIDENCE"
    )


def test_r24_cli_help_imports_no_taichi_and_mutates_no_solver_source() -> None:
    source = Path("simulation_core/coupling/interface_kalman_predictor.py").resolve()
    before = (source.stat().st_mtime_ns, source.read_bytes())
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "tools/audit_ansys_vertical_flap_kalman.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    after = (source.stat().st_mtime_ns, source.read_bytes())

    assert completed.returncode == 0, completed.stderr
    assert "Taichi" not in completed.stdout + completed.stderr
    assert before == after

def test_reporting_diagnostics_are_finite_and_cover_serial_segments() -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        kalman_statistical_reporting as reporting,
    )

    replay = subject.replay_candidate(
        _trace(frame_count=50),
        _candidate(warmup_accepted_states=1),
    )
    diagnostics = reporting.summarize_candidate(replay)

    assert diagnostics["candidate_id"] == "K2"
    assert diagnostics["innovation"]["axis"]["x"]["lag_autocorrelation"].keys() == {
        "lag_1",
        "lag_2",
        "lag_3",
    }
    assert diagnostics["innovation"]["axis"]["x"]["ljung_box"]["lag_3"][
        "critical_5pct"
    ] == pytest.approx(7.814727903251179)
    assert diagnostics["segments"].keys() >= {
        "steps_1_5",
        "steps_6_15",
        "steps_16_31",
        "steps_32_41",
        "step_42",
        "steps_43_49",
        "step_50",
    }
    json.dumps(diagnostics, allow_nan=False, sort_keys=True)


def test_exit_classification_separates_prediction_and_statistical_failures() -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        kalman_statistical_reporting as reporting,
    )

    common = {
        "provenance_ok": True,
        "k0_parity_ok": True,
        "contracts_ok": True,
    }
    assert reporting.classify_r24(
        **common,
        kalman_predictive_value=False,
        kalman_statistically_valid=True,
    ) == "FAIL_NO_KALMAN_PREDICTIVE_VALUE"
    assert reporting.classify_r24(
        **common,
        kalman_predictive_value=True,
        kalman_statistically_valid=False,
    ) == "FAIL_STATISTICAL_MODEL"
    assert reporting.classify_r24(
        **common,
        kalman_predictive_value=True,
        kalman_statistically_valid=True,
    ) == "PASS_ADVANCE_TO_R25"
    assert reporting.classify_r24(
        provenance_ok=True,
        k0_parity_ok=False,
        contracts_ok=True,
        kalman_predictive_value=True,
        kalman_statistically_valid=True,
    ) == "FAIL_EVIDENCE_OR_IMPLEMENTATION_CONTRACT"


def test_artifact_bundle_is_deterministic_finite_and_self_fingerprinted(
    tmp_path: Path,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        kalman_statistical_reporting as reporting,
    )

    trace = _trace(frame_count=8)
    replays = (
        subject.replay_candidate(
            trace, _candidate(candidate_id="C0", model="carry")
        ),
        subject.replay_candidate(
            trace, _candidate(candidate_id="K2", warmup_accepted_states=1)
        ),
    )
    ranking = subject.rank_candidates(replays)
    kwargs = {
        "output_dir": tmp_path,
        "split_manifest": {
            "schema_version": 1,
            "D0": {"fingerprint": trace.source_fingerprint},
            "D1": {"fingerprint": trace.source_fingerprint},
        },
        "replays_by_split": {"D0": replays, "D1": replays},
        "ranking": ranking,
        "k0_parity": {"passed": True},
        "exit_classification": "FAIL_NO_KALMAN_PREDICTIVE_VALUE",
    }

    reporting.write_artifact_bundle(**kwargs)
    names = {
        "kalman_innovation_audit.json",
        "kalman_candidate_ranking.json",
        "kalman_candidate_step_metrics.csv",
        "kalman_data_split_manifest.json",
    }
    first = {name: (tmp_path / name).read_bytes() for name in names}
    reporting.write_artifact_bundle(**kwargs)
    second = {name: (tmp_path / name).read_bytes() for name in names}

    assert first == second
    for name in names - {"kalman_candidate_step_metrics.csv"}:
        payload = json.loads(first[name])
        assert len(payload["artifact_fingerprint"]) == 64
        json.dumps(payload, allow_nan=False, sort_keys=True)
    assert b"accepted_state_source_sha256" in first[
        "kalman_candidate_step_metrics.csv"
    ]
    csv_rows = list(
        csv.DictReader(
            first["kalman_candidate_step_metrics.csv"].decode().splitlines()
        )
    )
    assert csv_rows[0]["accepted_measurement_sha256"] == trace.frame_sha256[0]
    assert csv_rows[0]["candidate_id"] == "C0"
    assert csv_rows[0]["model_name"] == "carry"

def test_source_compatibility_allows_only_predeclared_control_plane_differences() -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        kalman_statistical_campaign as campaign,
    )

    common = (
        ("attempt:simulation_core/coupling/interface_kalman_predictor.py", "1" * 64),
        ("attempt:simulation_core/fluids/solver.py", "2" * 64),
    )
    d0 = replace(
        _trace(name="D0"),
        source_sha256=common
        + (("attempt:simulation_core/diagnostics/atomic_file.py", "3" * 64),),
    )
    d1 = replace(
        _trace(name="D1"),
        source_sha256=common
        + (("attempt:simulation_core/diagnostics/atomic_file.py", "4" * 64),),
    )

    report = campaign.validate_source_compatibility(d0, d1)
    assert report["unexpected_differences"] == []
    assert report["allowed_differences"] == [
        "simulation_core/diagnostics/atomic_file.py"
    ]

    bad = replace(
        d1,
        source_sha256=(
            ("attempt:simulation_core/coupling/interface_kalman_predictor.py", "9" * 64),
            ("attempt:simulation_core/fluids/solver.py", "2" * 64),
        ),
    )
    with pytest.raises(subject.CalibrationContractError, match="production source"):
        campaign.validate_source_compatibility(d0, bad)


def test_executed_predictor_source_must_match_the_evidence_sha(
    tmp_path: Path,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        kalman_statistical_campaign as campaign,
    )

    source = Path("simulation_core/coupling/interface_kalman_predictor.py")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    assert campaign.validate_predictor_source(source, expected) == expected
    tampered = tmp_path / "interface_kalman_predictor.py"
    tampered.write_bytes(source.read_bytes() + b"\n# behavior-preserving tamper\n")
    with pytest.raises(subject.CalibrationContractError, match="SHA256"):
        campaign.validate_predictor_source(tampered, expected)


def test_k0_parity_is_step_exact_and_fails_on_tampered_history(
    tmp_path: Path,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        kalman_statistical_campaign as campaign,
    )

    trace = _trace(frame_count=4)
    config = {
        "rate_process_noise_spectral_density": [0.0, 2.0, 4.0],
        "measurement_variance": [0.1, 0.2, 0.3],
        "initial_value_variance": [0.1, 0.2, 0.3],
        "initial_rate_variance": [1.0, 2.0, 3.0],
        "warmup_accepted_states": 2,
    }
    replay = subject.replay_production_k0(
        trace,
        config,
        Path("simulation_core/coupling/interface_kalman_predictor.py"),
    )
    history_dir = tmp_path / "step_history"
    history_dir.mkdir()
    for row in replay.rows:
        payload = {
            "history": {
                "initial_guess_prediction_rms_mps": (
                    row.effective_prediction_rms_mps
                ),
                "initial_guess_prediction_bias_mps": (
                    row.effective_prediction_bias_mps
                ),
                "initial_guess_kalman_nis_mean": row.nis_mean,
            }
        }
        (history_dir / f"step_{row.physical_step:04d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    report = campaign.validate_k0_parity(replay, tmp_path)
    assert report["passed"]
    tampered = json.loads((history_dir / "step_0004.json").read_text())
    tampered["history"]["initial_guess_kalman_nis_mean"] += 1.0
    (history_dir / "step_0004.json").write_text(json.dumps(tampered))

    with pytest.raises(subject.CalibrationContractError, match="K0 parity"):
        campaign.validate_k0_parity(replay, tmp_path)

def test_inactive_zero_axis_is_preserved_and_excluded_from_statistical_score() -> None:
    values = _trajectory(frame_count=20)
    values[:, :, 0] = 0.0
    trace = _trace(frame_count=20, values=values)
    candidate = subject.calibrate_kalman_candidate(
        trace, model="constant_rate", candidate_id="K2"
    )

    assert candidate.active_axes == (False, True, True)
    assert candidate.scale_xyz[0] == 1.0
    replay = subject.replay_candidate(trace, candidate)
    score = subject.score_replay(replay)
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        kalman_statistical_reporting as reporting,
    )
    diagnostics = reporting.summarize_candidate(replay)

    assert score.axis_normalized_rmse[0] == 0.0
    assert score.axis_nis_mean[0] == 0.0
    assert all(row.raw_prediction_axis_mean_mps[0] == 0.0 for row in replay.rows)
    assert all(row.innovation_axis_mean_mps[0] == 0.0 for row in replay.rows)
    assert diagnostics["innovation"]["axis"]["x"]["active"] is False
    assert diagnostics["prediction"]["normalized_rmse"] == score.normalized_rmse

    invalid_values = np.array(trace.values, copy=True)
    invalid_values[:, :, 0] = 1.0
    with pytest.raises(subject.CalibrationContractError, match="inactive axes"):
        subject.replay_candidate(
            _trace(frame_count=20, values=invalid_values), candidate
        )
