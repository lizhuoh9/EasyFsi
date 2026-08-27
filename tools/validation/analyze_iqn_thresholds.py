"""Summarize fail-closed IQN-ILS threshold-audit reports without running a solver."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

_REQUIRED_HISTORY_FIELDS = {
    "absolute_residual_mps": "hibm_fsi_coupling_absolute_residual_history_mps",
    "relative_residual": "hibm_fsi_coupling_relative_residual_history",
    "candidate_velocity_rms_mps": (
        "hibm_fsi_coupling_candidate_velocity_rms_history_mps"
    ),
    "effective_tolerance_mps": (
        "hibm_fsi_coupling_effective_tolerance_history_mps"
    ),
    "residual_over_tolerance": (
        "hibm_fsi_coupling_residual_to_effective_tolerance_history"
    ),
}
_CSV_FIELDS = (
    "report",
    "step",
    "trial",
    "absolute_residual_mps",
    "relative_residual",
    "candidate_velocity_rms_mps",
    "effective_tolerance_mps",
    "residual_over_tolerance",
    "hit",
    "r1_over_r0",
    "r2_over_r1",
)


def _report_path(value: Path) -> Path:
    path = value / "our_solver_report_compact.json" if value.is_dir() else value
    if not path.is_file():
        raise ValueError(f"report not found: {path}")
    return path


def _load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed report: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"malformed report: {path}")
    return payload


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be finite")
    return numeric


def _history_series(
    row: dict[str, Any],
    *,
    report: Path,
    history_index: int,
) -> dict[str, list[float]]:
    missing = [
        source_name
        for source_name in _REQUIRED_HISTORY_FIELDS.values()
        if source_name not in row
    ]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(
            f"{report}: history[{history_index}] lacks threshold-audit fields "
            f"({fields}); old r06 artifacts require a source-matched rerun"
        )

    series: dict[str, list[float]] = {}
    for output_name, source_name in _REQUIRED_HISTORY_FIELDS.items():
        values = row[source_name]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"{report}: history[{history_index}].{source_name} must be a "
                "non-empty list"
            )
        series[output_name] = [
            _finite(
                value,
                context=f"{report}: history[{history_index}].{source_name}[{trial}]",
            )
            for trial, value in enumerate(values)
        ]

    expected_length = len(series["absolute_residual_mps"])
    for output_name, values in series.items():
        if len(values) != expected_length:
            raise ValueError(
                f"{report}: history[{history_index}] trial-history length mismatch "
                f"for {output_name}"
            )
    for trial, tolerance in enumerate(series["effective_tolerance_mps"]):
        if tolerance <= 0.0:
            raise ValueError(
                f"{report}: history[{history_index}] effective tolerance at "
                f"trial {trial} must be positive"
            )
    for trial, (absolute, candidate, tolerance, relative, ratio) in enumerate(
        zip(
            series["absolute_residual_mps"],
            series["candidate_velocity_rms_mps"],
            series["effective_tolerance_mps"],
            series["relative_residual"],
            series["residual_over_tolerance"],
        )
    ):
        expected_relative = absolute / max(candidate, 1.0e-30)
        if not math.isclose(relative, expected_relative, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError(
                f"{report}: history[{history_index}] relative residual mismatch "
                f"at trial {trial}"
            )
        series["relative_residual"][trial] = expected_relative
        expected_ratio = absolute / tolerance
        if not math.isclose(ratio, expected_ratio, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError(
                f"{report}: history[{history_index}] residual-over-tolerance "
                f"mismatch at trial {trial}"
            )
        series["residual_over_tolerance"][trial] = expected_ratio
    return series


def _contraction(
    residuals: list[float], numerator_trial: int, denominator_trial: int
) -> float | None:
    if len(residuals) <= numerator_trial:
        return None
    denominator = residuals[denominator_trial]
    return None if denominator == 0.0 else residuals[numerator_trial] / denominator


def analyze_reports(
    inputs: Sequence[Path | str], *, csv_path: Path | str | None = None
) -> dict[str, Any]:
    """Return per-trial threshold evidence from compact report files or run dirs."""

    reports = [_report_path(Path(value)) for value in inputs]
    if not reports:
        raise ValueError("at least one compact report or run directory is required")

    trials: list[dict[str, Any]] = []
    step_trial_counts: list[int] = []
    for report in reports:
        payload = _load_report(report)
        history = payload.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"{report}: history must be a non-empty list")
        for history_index, raw_row in enumerate(history):
            if not isinstance(raw_row, dict):
                raise ValueError(f"{report}: history[{history_index}] must be an object")
            if "step" not in raw_row:
                raise ValueError(f"{report}: history[{history_index}] lacks step")
            series = _history_series(
                raw_row, report=report, history_index=history_index
            )
            residuals = series["absolute_residual_mps"]
            step_trial_counts.append(len(residuals))
            for trial, residual in enumerate(residuals):
                trials.append(
                    {
                        "report": str(report),
                        "step": raw_row["step"],
                        "trial": trial,
                        "absolute_residual_mps": residual,
                        "relative_residual": series["relative_residual"][trial],
                        "candidate_velocity_rms_mps": (
                            series["candidate_velocity_rms_mps"][trial]
                        ),
                        "effective_tolerance_mps": (
                            series["effective_tolerance_mps"][trial]
                        ),
                        "residual_over_tolerance": (
                            series["residual_over_tolerance"][trial]
                        ),
                        "hit": (
                            residual
                            <= series["effective_tolerance_mps"][trial]
                        ),
                        "r1_over_r0": (
                            _contraction(residuals, 1, 0) if trial == 1 else None
                        ),
                        "r2_over_r1": (
                            _contraction(residuals, 2, 1) if trial == 2 else None
                        ),
                    }
                )

    max_trial = max(row["trial"] for row in trials)
    hit_rate: dict[str, float] = {}
    for trial in range(max_trial + 1):
        values = [row["hit"] for row in trials if row["trial"] == trial]
        hit_rate[str(trial)] = sum(values) / len(values)

    result = {
        "report_count": len(reports),
        "step_count": len(step_trial_counts),
        "trial_count_distribution": {
            str(count): sum(value == count for value in step_trial_counts)
            for count in sorted(set(step_trial_counts))
        },
        "trial_hit_rate": hit_rate,
        "trials": trials,
    }
    if csv_path is not None:
        _write_csv(Path(csv_path), trials)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize IQN-ILS residual-to-tolerance threshold evidence."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="our_solver_report_compact.json paths or their run directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON summary to this path instead of standard output",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional path for the per-trial CSV time series",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = analyze_reports(args.inputs, csv_path=args.csv)
    except ValueError as exc:
        print(f"threshold audit failed: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
