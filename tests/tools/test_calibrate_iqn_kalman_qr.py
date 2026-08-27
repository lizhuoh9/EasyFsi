from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_frame(
    directory: Path,
    *,
    step: int,
    dt_s: float = 0.1,
    layout_id: str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    marker_count: int = 2,
) -> None:
    time_s = step * dt_s
    marker_axis = np.arange(marker_count, dtype=np.float64)[:, None]
    axes = np.arange(3, dtype=np.float64)[None, :]
    accepted = float(step) + marker_axis + axes
    guesses = np.stack((accepted - 0.30, accepted - 0.10, accepted - 0.01))
    candidates = np.stack((accepted - 0.20, accepted, accepted))
    residuals = candidates - guesses
    np.savez(
        directory / f"step_{step:04d}.npz",
        marker_velocity_mps=accepted,
        iqn_trial_guess_mps=guesses,
        iqn_trial_candidate_mps=candidates,
        iqn_trial_residual_mps=residuals,
        iqn_trial_index=np.arange(3, dtype=np.int64),
        iqn_trial_layout_sha256=np.asarray(layout_id),
        iqn_trial_step=np.asarray(step, dtype=np.int64),
        iqn_trial_dt_s=np.asarray(dt_s),
        iqn_trial_time_s=np.asarray(time_s),
    )


def _write_study(directory: Path) -> None:
    directory.mkdir()
    for step in range(1, 251):
        _write_frame(directory, step=step)


def test_calibration_and_frozen_eval_are_causal_and_emit_machine_readable_outputs(
    tmp_path: Path,
) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    frames = tmp_path / "step_fields"
    _write_study(frames)
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "result.csv"

    result = subject.calibrate_and_evaluate(
        frames,
        scalar_q=0.2,
        scalar_r=0.4,
        json_path=output_json,
        csv_path=output_csv,
    )

    assert result["provenance"]["frame_count"] == 250
    assert result["provenance"]["layout_sha256"] == (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )
    assert result["ranges"] == {
        "calibration_steps": [1, 100],
        "frozen_evaluation_steps": [101, 250],
    }
    assert result["formula"]["measurement_variance"] == (
        "sample_var(last_trial_candidate - penultimate_trial_candidate, ddof=1)"
    )
    assert result["selection"]["q_multiplier_candidates"] == [
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
    ]
    assert result["selection"]["selection_used_frozen_eval"] is False
    assert result["trial_reduction_evaluated"] is False
    assert result["acceleration_claimed"] is False
    assert result["calibration"]["r_xyz_m2_per_s2"] == pytest.approx([0.0, 0.0, 0.0])
    assert result["calibration"]["q0_xyz_m2_per_s5"] == pytest.approx(
        [0.0, 0.0, 0.0], abs=1.0e-18
    )
    assert result["formula"]["units"] == {
        "r_xyz": "m^2/s^2",
        "q0_xyz": "m^2/s^5",
        "scalar_r": "m^2/s^2",
        "scalar_q": "m^2/s^5",
    }
    assert result["calibration"]["axis_status"] == ["inactive_zero_variance"] * 3
    assert result["selection"]["selected_q_multiplier"] == 0.1

    frozen = result["frozen_evaluation"]
    assert set(frozen["methods"]) == {
        "carry_forward",
        "linear_extrapolation",
        "scalar_kalman",
        "per_axis_kalman",
    }
    for method in frozen["methods"].values():
        assert method["sample_count"] == 150 * 2
        assert len(method["axis_rmse_mps"]) == 3
        assert len(method["axis_bias_mps"]) == 3
    assert frozen["methods"]["carry_forward"]["global_nis"] is None
    assert frozen["methods"]["linear_extrapolation"]["global_nis"] is None
    assert np.isfinite(frozen["methods"]["scalar_kalman"]["global_nis"])
    assert frozen["methods"]["per_axis_kalman"]["global_nis"] is None

    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    assert persisted == result
    csv_text = output_csv.read_text(encoding="utf-8")
    assert "selection_used_frozen_eval" in csv_text
    assert "measurement_variance_formula" in csv_text
    assert "frozen_evaluation_steps" in csv_text
    assert "per_axis_kalman" in csv_text


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _write_anisotropic_study(directory: Path) -> None:
    directory.mkdir()
    scales = np.asarray([1.0, 4.0, 9.0], dtype=np.float64)
    for step in range(1, 251):
        marker = np.arange(2, dtype=np.float64)[:, None]
        accepted = (step**4 * scales)[None, :] + marker
        penultimate = accepted - 0.2
        delta = (step + marker) * scales
        candidates = np.stack((accepted - 0.4, penultimate, penultimate + delta))
        guesses = candidates - 0.1
        np.savez(
            directory / f"step_{step:04d}.npz",
            marker_velocity_mps=accepted,
            iqn_trial_guess_mps=guesses,
            iqn_trial_candidate_mps=candidates,
            iqn_trial_residual_mps=candidates - guesses,
            iqn_trial_index=np.arange(3, dtype=np.int64),
            iqn_trial_layout_sha256=np.asarray("a" * 64),
            iqn_trial_step=np.asarray(step, dtype=np.int64),
            iqn_trial_time_s=np.asarray(step * 0.1),
            iqn_trial_dt_s=np.asarray(0.1),
        )


def test_calibration_isolation_never_selects_using_frozen_steps(tmp_path: Path) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    reference = tmp_path / "reference"
    perturbed = tmp_path / "perturbed"
    _write_study(reference)
    _write_study(perturbed)
    for step in range(101, 251):
        path = perturbed / f"step_{step:04d}.npz"
        payload = _load_payload(path)
        payload["marker_velocity_mps"] = payload["marker_velocity_mps"] + step**2
        payload["iqn_trial_guess_mps"] = payload["iqn_trial_guess_mps"] + step**2
        payload["iqn_trial_candidate_mps"] = payload["iqn_trial_candidate_mps"] + step**2
        np.savez(path, **payload)

    first = subject.calibrate_and_evaluate(reference, scalar_q=0.2, scalar_r=0.4)
    second = subject.calibrate_and_evaluate(perturbed, scalar_q=0.2, scalar_r=0.4)

    assert first["calibration"] == second["calibration"]
    assert first["selection"] == second["selection"]
    assert first["frozen_evaluation"]["methods"]["carry_forward"] != second[
        "frozen_evaluation"
    ]["methods"]["carry_forward"]


def test_anisotropic_calibration_recovers_distinct_nonzero_per_axis_q_and_r(
    tmp_path: Path,
) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    frames = tmp_path / "anisotropic"
    _write_anisotropic_study(frames)
    result = subject.calibrate_and_evaluate(frames, scalar_q=0.2, scalar_r=0.4)

    r_xyz = np.asarray(result["calibration"]["r_xyz_m2_per_s2"])
    q_xyz = np.asarray(result["calibration"]["q0_xyz_m2_per_s5"])
    study = subject.load_iqn_kalman_study(frames)
    expected_r = np.var(
        np.concatenate(
            [candidate[-1] - candidate[-2] for candidate in study.trial_candidates[:100]],
            axis=0,
        ),
        axis=0,
        ddof=1,
    )
    velocity = study.accepted[:100]
    expected_q = np.var(
        ((velocity[2:] - 2 * velocity[1:-1] + velocity[:-2]) / 0.1**2).reshape(-1, 3),
        axis=0,
        ddof=1,
    ) * 0.1
    np.testing.assert_allclose(r_xyz, expected_r)
    np.testing.assert_allclose(q_xyz, expected_q)
    assert np.all(r_xyz > 0.0)
    assert np.all(q_xyz > 0.0)
    assert np.all(np.diff(r_xyz) > 0.0)
    assert np.all(np.diff(q_xyz) > 0.0)
    assert result["calibration"]["axis_status"] == ["active", "active", "active"]


def test_calibration_rejects_calibration_frame_with_fewer_than_two_trials(
    tmp_path: Path,
) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    frames = tmp_path / "step_fields"
    _write_study(frames)
    path = frames / "step_0050.npz"
    payload = _load_payload(path)
    for key in ("iqn_trial_guess_mps", "iqn_trial_candidate_mps", "iqn_trial_residual_mps"):
        payload[key] = payload[key][:1]
    payload["iqn_trial_index"] = np.asarray([0], dtype=np.int64)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="at least two"):
        subject.calibrate_and_evaluate(frames, scalar_q=0.2, scalar_r=0.4)


def test_frozen_evaluation_allows_one_trial_after_calibration(tmp_path: Path) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    frames = tmp_path / "step_fields"
    _write_study(frames)
    path = frames / "step_0101.npz"
    payload = _load_payload(path)
    for key in ("iqn_trial_guess_mps", "iqn_trial_candidate_mps", "iqn_trial_residual_mps"):
        payload[key] = payload[key][:1]
    payload["iqn_trial_index"] = np.asarray([0], dtype=np.int64)
    np.savez(path, **payload)

    result = subject.calibrate_and_evaluate(frames, scalar_q=0.2, scalar_r=0.4)
    assert result["frozen_evaluation"]["methods"]["carry_forward"]["sample_count"] == 300


def test_zero_q_with_positive_r_remains_an_active_constant_rate_axis(
    tmp_path: Path,
) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    frames = tmp_path / "step_fields"
    _write_study(frames)
    for step in range(1, 101):
        path = frames / f"step_{step:04d}.npz"
        payload = _load_payload(path)
        candidate = payload["iqn_trial_candidate_mps"].copy()
        candidate[-1] = candidate[-2] + step * np.asarray([1.0, 2.0, 3.0])
        payload["iqn_trial_candidate_mps"] = candidate
        payload["iqn_trial_residual_mps"] = candidate - payload["iqn_trial_guess_mps"]
        np.savez(path, **payload)

    result = subject.calibrate_and_evaluate(frames, scalar_q=0.2, scalar_r=0.4)
    assert result["calibration"]["q0_xyz_m2_per_s5"] == [0.0, 0.0, 0.0]
    assert result["calibration"]["axis_status"] == [
        "active_zero_process_variance",
        "active_zero_process_variance",
        "active_zero_process_variance",
    ]
    assert all(
        np.isfinite(result["frozen_evaluation"]["methods"]["per_axis_kalman"]["axis_nis"])
    )


@pytest.mark.parametrize(
    ("frame", "key", "replacement", "message"),
    [
        (10, "iqn_trial_layout_sha256", np.asarray("f" * 64), "layout"),
        (20, "iqn_trial_dt_s", np.asarray(0.2), "dt_s"),
        (30, "iqn_trial_time_s", np.asarray(3.123), "time_s"),
        (
            40,
            "iqn_trial_index",
            np.asarray([0, 2, 3], dtype=np.int64),
            "trial indices",
        ),
        (
            50,
            "iqn_trial_candidate_mps",
            np.full((3, 2, 3), np.nan),
            "finite",
        ),
    ],
)
def test_calibration_rejects_noncontinuous_or_invalid_evidence(
    tmp_path: Path,
    frame: int,
    key: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    frames = tmp_path / "step_fields"
    _write_study(frames)
    path = frames / f"step_{frame:04d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload[key] = replacement
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=message):
        subject.load_iqn_kalman_study(frames)


def test_calibration_requires_exactly_250_contiguous_frames(tmp_path: Path) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    frames = tmp_path / "step_fields"
    _write_study(frames)
    (frames / "step_0250.npz").unlink()

    with pytest.raises(ValueError, match="exactly 250"):
        subject.load_iqn_kalman_study(frames)


@pytest.mark.parametrize(
    ("scalar_q", "scalar_r"),
    [(None, 0.1), (0.1, None), (0.0, 0.1), (0.1, float("inf"))],
)
def test_scalar_kalman_parameters_must_be_explicit_positive_finite(
    tmp_path: Path,
    scalar_q: float | None,
    scalar_r: float | None,
) -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    frames = tmp_path / "step_fields"
    _write_study(frames)

    with pytest.raises(ValueError, match="scalar"):
        subject.calibrate_and_evaluate(frames, scalar_q=scalar_q, scalar_r=scalar_r)


def test_cli_requires_explicit_scalar_q_and_r() -> None:
    from tools.validation import calibrate_iqn_kalman_qr as subject

    with pytest.raises(SystemExit):
        subject.main(
            [
                "--step-fields",
                "missing",
                "--scalar-r",
                "0.4",
                "--output-json",
                "result.json",
                "--output-csv",
                "result.csv",
            ]
        )
