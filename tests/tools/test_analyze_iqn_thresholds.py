from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def _report(*history: dict[str, object]) -> dict[str, object]:
    return {"history": list(history)}


def _step(
    step: int,
    *,
    absolute: list[float],
    relative: list[float],
    candidate: list[float],
    tolerance: list[float],
    ratio: list[float],
) -> dict[str, object]:
    return {
        "step": step,
        "hibm_fsi_coupling_absolute_residual_history_mps": absolute,
        "hibm_fsi_coupling_relative_residual_history": relative,
        "hibm_fsi_coupling_candidate_velocity_rms_history_mps": candidate,
        "hibm_fsi_coupling_effective_tolerance_history_mps": tolerance,
        "hibm_fsi_coupling_residual_to_effective_tolerance_history": ratio,
    }


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_reports_emits_per_trial_threshold_audit_and_optional_csv(
    tmp_path: Path,
) -> None:
    from tools.validation import analyze_iqn_thresholds as subject

    run = tmp_path / "q2"
    run.mkdir()
    _write_report(
        run / "our_solver_report_compact.json",
        _report(
            _step(
                7,
                absolute=[3.0, 1.2, 0.7],
                relative=[0.30, 0.12, 0.07],
                candidate=[10.0, 10.0, 10.0],
                tolerance=[1.0, 1.0, 1.0],
                ratio=[3.0, 1.2, 0.7],
            ),
            _step(
                8,
                absolute=[0.8],
                relative=[0.8 / 11.0],
                candidate=[11.0],
                tolerance=[1.0],
                ratio=[0.8],
            ),
        ),
    )

    csv_path = tmp_path / "trials.csv"
    result = subject.analyze_reports([run], csv_path=csv_path)

    assert result["step_count"] == 2
    assert result["trial_count_distribution"] == {"1": 1, "3": 1}
    assert result["trial_hit_rate"] == {"0": 0.5, "1": 0.0, "2": 1.0}
    assert result["trials"][0] == {
        "report": str(run / "our_solver_report_compact.json"),
        "step": 7,
        "trial": 0,
        "absolute_residual_mps": 3.0,
        "relative_residual": 0.3,
        "candidate_velocity_rms_mps": 10.0,
        "effective_tolerance_mps": 1.0,
        "residual_over_tolerance": 3.0,
        "hit": False,
        "r1_over_r0": None,
        "r2_over_r1": None,
    }
    assert result["trials"][1]["r1_over_r0"] == pytest.approx(0.4)
    assert result["trials"][2]["r2_over_r1"] == pytest.approx(0.7 / 1.2)
    assert result["trials"][3]["hit"] is True

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert float(rows[1]["r1_over_r0"]) == pytest.approx(0.4)
    assert rows[2]["r2_over_r1"] == str(0.7 / 1.2)


def test_analyze_reports_keeps_observed_four_trial_count(tmp_path: Path) -> None:
    from tools.validation import analyze_iqn_thresholds as subject

    report = tmp_path / "our_solver_report_compact.json"
    _write_report(
        report,
        _report(
            _step(
                1,
                absolute=[4.0, 3.0, 2.0, 0.5],
                relative=[0.4, 0.3, 0.2, 0.05],
                candidate=[10.0, 10.0, 10.0, 10.0],
                tolerance=[1.0, 1.0, 1.0, 1.0],
                ratio=[4.0, 3.0, 2.0, 0.5],
            )
        ),
    )

    assert subject.analyze_reports([report])["trial_count_distribution"] == {"4": 1}

def test_analyze_reports_rejects_legacy_report_without_new_audit_fields(
    tmp_path: Path,
) -> None:
    from tools.validation import analyze_iqn_thresholds as subject

    report = tmp_path / "our_solver_report_compact.json"
    _write_report(report, {"history": [{"step": 1}]})

    with pytest.raises(ValueError, match="source-matched rerun"):
        subject.analyze_reports([report])


def test_analyze_reports_rejects_tampered_residual_over_tolerance(
    tmp_path: Path,
) -> None:
    from tools.validation import analyze_iqn_thresholds as subject

    report = tmp_path / "our_solver_report_compact.json"
    _write_report(
        report,
        _report(
            _step(
                1,
                absolute=[3.0, 1.2],
                relative=[0.3, 0.12],
                candidate=[10.0, 10.0],
                tolerance=[1.0, 1.0],
                ratio=[3.0001, 1.2],
            )
        ),
    )

    with pytest.raises(ValueError, match="residual-over-tolerance mismatch"):
        subject.analyze_reports([report])


def test_analyze_reports_rejects_tampered_relative_residual(tmp_path: Path) -> None:
    from tools.validation import analyze_iqn_thresholds as subject

    report = tmp_path / "our_solver_report_compact.json"
    _write_report(
        report,
        _report(
            _step(
                1,
                absolute=[3.0, 1.2],
                relative=[0.3001, 0.12],
                candidate=[10.0, 10.0],
                tolerance=[1.0, 1.0],
                ratio=[3.0, 1.2],
            )

        )
    )

    with pytest.raises(ValueError, match="relative residual mismatch"):
        subject.analyze_reports([report])


def test_analyze_reports_uses_recomputed_near_threshold_metrics(
    tmp_path: Path,
) -> None:
    from tools.validation import analyze_iqn_thresholds as subject

    report = tmp_path / "our_solver_report_compact.json"
    recomputed_ratio = 1.0000000000004
    _write_report(
        report,
        _report(
            _step(
                1,
                absolute=[recomputed_ratio],
                relative=[0.9999999999996],
                candidate=[1.0],
                tolerance=[1.0],
                ratio=[0.9999999999996],
            )
        ),
    )

    trial = subject.analyze_reports([report])["trials"][0]

    assert trial["relative_residual"] == recomputed_ratio
    assert trial["residual_over_tolerance"] == recomputed_ratio
    assert trial["hit"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda row: row.__setitem__(
                "hibm_fsi_coupling_effective_tolerance_history_mps", [1.0, 0.0]
            ),
            "positive",
        ),
        (
            lambda row: row.__setitem__(
                "hibm_fsi_coupling_relative_residual_history", [0.2]
            ),
            "length",
        ),
        (
            lambda row: row.__setitem__(
                "hibm_fsi_coupling_residual_to_effective_tolerance_history",
                [float("nan"), 0.2],
            ),
            "finite",
        ),
    ],
)
def test_analyze_reports_rejects_invalid_trial_series(
    tmp_path: Path, mutator: object, message: str
) -> None:
    from tools.validation import analyze_iqn_thresholds as subject

    row = _step(
        1,
        absolute=[2.0, 0.2],
        relative=[0.2, 0.02],
        candidate=[10.0, 10.0],
        tolerance=[1.0, 1.0],
        ratio=[2.0, 0.2],
    )
    mutator(row)  # type: ignore[operator]
    report = tmp_path / "our_solver_report_compact.json"
    _write_report(report, _report(row))

    with pytest.raises(ValueError, match=message):
        subject.analyze_reports([report])
