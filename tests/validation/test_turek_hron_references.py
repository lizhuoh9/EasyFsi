from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.refactored.validation.turek_hron_fsi.references import (
    FEATFLOW_RAW_FSI2_SOURCE_ID,
    FEATFLOW_RAW_FSI3_SOURCE_ID,
    FEATFLOW_WEB_SOURCE_ID,
    LSDYNA_CROSSCHECK_SOURCE_ID,
    REFERENCE_CATALOG,
    ReferenceContractError,
    canonical_fsi1_metric_values,
    case_reference_contract,
    legacy_case_reference_projection,
    published_result,
    raw_series_identity,
    reference_metric,
)


def test_complete_featflow_fsi123_contract_preserves_exact_anchors() -> None:
    fsi1 = case_reference_contract(FEATFLOW_WEB_SOURCE_ID, "fsi1")
    fsi2 = case_reference_contract(FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2")
    fsi3 = case_reference_contract(FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3")

    assert {metric.quantity for metric in fsi1.metrics} == {
        "point_a_ux", "point_a_uy", "total_drag", "total_lift",
    }
    assert reference_metric(
        FEATFLOW_WEB_SOURCE_ID, "fsi1", "point_a_ux", "steady_value"
    ).value == pytest.approx(2.270493e-5)
    assert reference_metric(
        FEATFLOW_WEB_SOURCE_ID, "fsi1", "total_lift", "steady_value"
    ).value == pytest.approx(0.7637460)
    assert reference_metric(
        FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2", "point_a_uy", "amplitude"
    ).value == pytest.approx(0.08165565385)
    assert reference_metric(
        FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2", "total_drag", "midrange"
    ).value == pytest.approx(215.088610865)
    assert reference_metric(
        FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3", "point_a_uy", "frequency"
    ).value == pytest.approx(5.47355995969522)
    assert reference_metric(
        FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3", "total_lift", "amplitude"
    ).value == pytest.approx(153.527757)
    assert fsi2.raw_series is not None
    assert fsi2.raw_series.sha256 == (
        "d4e192f5ae6aa493d36c472b68d8d734d54194aca6e9f0a70014816b6835dd75"
    )
    assert fsi3.raw_series is not None
    assert fsi3.raw_series.row_count == 5769


def test_sources_are_separate_and_unknown_lookups_fail_closed() -> None:
    assert REFERENCE_CATALOG.source(FEATFLOW_WEB_SOURCE_ID).role == "canonical"
    assert REFERENCE_CATALOG.source(FEATFLOW_RAW_FSI2_SOURCE_ID).role == "canonical"
    assert REFERENCE_CATALOG.source(FEATFLOW_RAW_FSI3_SOURCE_ID).role == "canonical"
    assert REFERENCE_CATALOG.source(LSDYNA_CROSSCHECK_SOURCE_ID).role == "cross_check"
    with pytest.raises(ReferenceContractError, match="unknown source"):
        REFERENCE_CATALOG.source("unknown")
    with pytest.raises(ReferenceContractError, match="unknown case"):
        case_reference_contract(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi2")
    with pytest.raises(ReferenceContractError, match="unknown metric"):
        reference_metric(FEATFLOW_WEB_SOURCE_ID, "fsi1", "total_drag", "amplitude")


def test_source_metadata_and_force_scope_are_exact() -> None:
    fsi1_drag = reference_metric(
        FEATFLOW_WEB_SOURCE_ID, "fsi1", "total_drag", "steady_value"
    )
    fsi2_frequency = reference_metric(
        FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2", "point_a_uy", "frequency"
    )

    assert fsi1_drag.unit == "N"
    assert fsi1_drag.force_scope == "whole_body"
    assert fsi2_frequency.unit == "Hz"
    assert fsi2_frequency.force_scope is None
    assert reference_metric(
        FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3", "total_lift", "amplitude"
    ).force_scope == "whole_body"
    assert len(REFERENCE_CATALOG.definition.primary_urls) == 4
    assert all(url.startswith("https://wwwold.mathematik.tu-dortmund.de/") for url in REFERENCE_CATALOG.definition.primary_urls)
    assert REFERENCE_CATALOG.definition.coordinate_sign_convention.startswith(
        "turek_hron_2d_v1:x_left_inflow_to_right_outflow"
    )
    assert REFERENCE_CATALOG.definition.shear_modulus_pa[-1] == ("fsi3", 2.0e6)


def test_contract_and_compatibility_projection_are_deeply_read_only() -> None:
    fsi1 = case_reference_contract(FEATFLOW_WEB_SOURCE_ID, "fsi1")
    with pytest.raises(FrozenInstanceError):
        fsi1.case_id = "fsi2"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        fsi1.metrics += ()  # type: ignore[misc]
    assert fsi1.metrics
    with pytest.raises(FrozenInstanceError):
        fsi1.metrics[0].value = 0.0  # type: ignore[misc]
    values = canonical_fsi1_metric_values()
    with pytest.raises(TypeError):
        values["tip_ux_turek_hron_m"] = 0.0  # type: ignore[index]
    projection = legacy_case_reference_projection()
    with pytest.raises(TypeError):
        projection["fsi1"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        projection["fsi1"]["ux_a_m"] = 0.0  # type: ignore[index]
    raw_fsi2 = case_reference_contract(FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2")
    assert raw_fsi2.raw_series is not None
    with pytest.raises(TypeError):
        raw_fsi2.raw_series.documented_columns[0] = (1, "other")  # type: ignore[index]
    assert reference_metric(
        FEATFLOW_WEB_SOURCE_ID, "fsi1", "point_a_ux", "steady_value"
    ).value == pytest.approx(2.270493e-5)


def test_raw_identities_are_complete_and_exact() -> None:
    fsi2 = raw_series_identity(FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2")
    fsi3 = raw_series_identity(FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3")

    assert fsi2.url.endswith("/fsi2/0p0005/ref_fsi2.point")
    assert fsi2.schema_version == "featflow_raw_series_manifest_v1"
    assert fsi2.units == (
        ("time", "s"), ("force", "N"), ("point_a_displacement", "m")
    )
    assert fsi2.coordinate_sign_convention == (
        REFERENCE_CATALOG.definition.coordinate_sign_convention
    )
    assert fsi3.url.endswith("/fsi3/0p00025/ref_fsi3.point")
    with pytest.raises(ReferenceContractError, match="no raw-series identity"):
        raw_series_identity(FEATFLOW_WEB_SOURCE_ID, "fsi1")


def test_exact_published_featflow_rows_cover_all_preregistered_identities() -> None:
    rows = REFERENCE_CATALOG.published_results
    assert len(rows) == 24
    assert [row.level for row in rows[:6]] == [
        "2+0", "3+0", "4+0", "5+0", "6+0", "7+0",
    ]
    assert [(row.nel, row.ndof) for row in rows[:6]] == [
        (992, 19488),
        (3968, 76672),
        (15872, 304128),
        (63488, 1211392),
        (253952, 4835328),
        (1015808, 19320832),
    ]
    assert all(row.timestep_s is None for row in rows[:6])
    assert published_result("fsi1", "7+0", None).total_lift[0] == pytest.approx(
        7.637460e-1
    )
    assert published_result("fsi2", "4+0", 0.0005).nel == 15872
    assert published_result("fsi3", "4+0", 0.00025).ndof == 304128
    assert published_result("fsi2", "4+0", 0.0005).point_a_uy == pytest.approx(
        (1.30e-3, 8.16e-2, 1.93)
    )
    assert published_result("fsi3", "2+0", 0.00025).total_lift == pytest.approx(
        (2.23, 1.4600e2, 5.33)
    )
    with pytest.raises(ReferenceContractError, match="unknown published Featflow row"):
        published_result("fsi2", "7+0", 0.0005)
