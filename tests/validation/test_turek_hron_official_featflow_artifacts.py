from __future__ import annotations

from pathlib import Path

import pytest

from src.refactored.validation.turek_hron_fsi.featflow import load_featflow_series
from src.refactored.validation.turek_hron_fsi.limit_cycle import (
    analyze_featflow_limit_cycle,
)
from src.refactored.validation.turek_hron_fsi.references import (
    FEATFLOW_RAW_FSI2_SOURCE_ID,
    FEATFLOW_RAW_FSI3_SOURCE_ID,
    case_reference_contract,
    reference_metric,
)


_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "validation"
    / "turek_hron_featflow"
)

_CASES = (
    (
        "fsi2",
        FEATFLOW_RAW_FSI2_SOURCE_ID,
        (
            12.6535123414151,
            13.1714752669647,
            13.6894430250074,
            14.2074128547141,
        ),
        "total_lift_not_strictly_increasing",
    ),
    (
        "fsi3",
        FEATFLOW_RAW_FSI3_SOURCE_ID,
        (
            5.86502510218353,
            6.04773603930534,
            6.23043694489322,
            6.41313339973496,
        ),
        "total_drag_not_strictly_increasing",
    ),
)


@pytest.mark.parametrize(
    ("case_id", "source_id", "crossings", "strict_trend_check"), _CASES
)
def test_official_featflow_artifacts_reproduce_frozen_periodic_anchors(
    case_id: str,
    source_id: str,
    crossings: tuple[float, ...],
    strict_trend_check: str,
) -> None:
    contract = case_reference_contract(source_id, case_id)
    identity = contract.raw_series
    assert identity is not None

    raw_path = _ARTIFACT_ROOT / identity.filename
    manifest_path = _ARTIFACT_ROOT / f"ref_{case_id}.manifest.json"
    series = load_featflow_series(raw_path, manifest_path)

    assert series.manifest.schema_version == identity.schema_version
    assert series.manifest.source_id == identity.source_id
    assert series.manifest.case_id == identity.case_id
    assert series.manifest.url == identity.url
    assert series.manifest.byte_count == identity.byte_count
    assert series.manifest.row_count == identity.row_count
    assert series.manifest.column_count == identity.column_count
    assert series.manifest.time_start_s == pytest.approx(identity.time_start_s)
    assert series.manifest.time_end_s == pytest.approx(identity.time_end_s)
    assert series.manifest.time_step_s == pytest.approx(identity.dt_s)
    assert series.manifest.sha256 == identity.sha256
    assert tuple(
        (int(column), meaning)
        for column, meaning in series.manifest.documented_columns.items()
    ) == identity.documented_columns
    assert series.manifest.opaque_columns == identity.opaque_columns
    assert tuple(series.manifest.units.items()) == identity.units
    assert series.manifest.force_scope == identity.force_scope
    assert series.manifest.coordinate_convention == identity.coordinate_sign_convention
    assert series.manifest.extraction_policy == identity.extraction_policy

    report = analyze_featflow_limit_cycle(series)
    assert report.crossing_times_s[-4:] == pytest.approx(crossings)
    assert report.primary_frequency_hz == pytest.approx(
        reference_metric(source_id, case_id, "point_a_uy", "frequency").value
    )
    assert report.cycles[-1].signals["point_a_uy_m"].amplitude == pytest.approx(
        reference_metric(source_id, case_id, "point_a_uy", "amplitude").value
    )
    assert report.cycles[-1].signals["total_drag_n"].midrange == pytest.approx(
        reference_metric(source_id, case_id, "total_drag", "midrange").value
    )
    assert report.cycles[-1].signals["total_lift_n"].amplitude == pytest.approx(
        reference_metric(source_id, case_id, "total_lift", "amplitude").value
    )

    # The source series defines reference anchors; it is not an R26A solver run.
    # Both traces satisfy every quantitative stability threshold but each has one
    # tiny monotone force-amplitude trend that the stricter local run gate rejects.
    assert report.stability.checks[strict_trend_check] is False
    assert all(
        passed
        for check, passed in report.stability.checks.items()
        if check != strict_trend_check
    )
