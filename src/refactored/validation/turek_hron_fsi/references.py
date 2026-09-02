"""Immutable source-first reference contracts for Turek--Hron FSI validation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


FEATFLOW_WEB_SOURCE_ID = "featflow_tu_dortmund_web"
FEATFLOW_RAW_FSI2_SOURCE_ID = "featflow_raw_fsi2_dt_0p0005"
FEATFLOW_RAW_FSI3_SOURCE_ID = "featflow_raw_fsi3_dt_0p00025"
LSDYNA_CROSSCHECK_SOURCE_ID = "lsdyna_2013_crosscheck"


class ReferenceContractError(ValueError):
    """Raised when a source, case, or metric is not explicitly contracted."""


@dataclass(frozen=True)
class ReferenceMetric:
    source_id: str
    case_id: str
    quantity: str
    statistic: str
    value: float
    unit: str
    force_scope: str | None = None
    uncertainty: float | None = None


@dataclass(frozen=True)
class RawSeriesIdentity:
    source_id: str
    case_id: str
    url: str
    filename: str
    schema_version: str
    byte_count: int
    row_count: int
    column_count: int
    time_start_s: float
    time_end_s: float
    dt_s: float
    sha256: str
    documented_columns: tuple[tuple[int, str], ...]
    opaque_columns: tuple[int, ...]
    units: tuple[tuple[str, str], ...]
    force_scope: str
    coordinate_sign_convention: str
    extraction_policy: str


@dataclass(frozen=True)
class CaseReferenceContract:
    case_id: str
    metrics: tuple[ReferenceMetric, ...]
    raw_series: RawSeriesIdentity | None = None


@dataclass(frozen=True)
class ReferenceSource:
    source_id: str
    role: str
    url: str
    cases: tuple[CaseReferenceContract, ...]


@dataclass(frozen=True)
class BenchmarkDefinition:
    """Immutable benchmark geometry, physics, and convention contract."""

    primary_urls: tuple[str, ...]
    channel_length_m: float
    channel_height_m: float
    cylinder_center_m: tuple[float, float]
    cylinder_radius_m: float
    beam_length_m: float
    beam_thickness_m: float
    point_a_initial_m: tuple[float, float]
    fluid_density_kg_m3: float
    fluid_kinematic_viscosity_m2_s: float
    poisson_ratio: float
    mean_inlet_mps: tuple[tuple[str, float], ...]
    solid_density_kg_m3: tuple[tuple[str, float], ...]
    youngs_modulus_pa: tuple[tuple[str, float], ...]
    shear_modulus_pa: tuple[tuple[str, float], ...]
    boundary_conditions: tuple[str, ...]
    coordinate_sign_convention: str


@dataclass(frozen=True)
class PublishedResultRow:
    """One exact Featflow table row; periodic values are midrange/amplitude/frequency."""

    case_id: str
    level: str
    nel: int
    ndof: int
    timestep_s: float | None
    point_a_ux: tuple[float, float | None, float | None]
    point_a_uy: tuple[float, float | None, float | None]
    total_drag: tuple[float, float | None, float | None]
    total_lift: tuple[float, float | None, float | None]


@dataclass(frozen=True)
class ReferenceCatalog:
    sources: tuple[ReferenceSource, ...]
    definition: BenchmarkDefinition
    published_results: tuple[PublishedResultRow, ...]

    def source(self, source_id: str) -> ReferenceSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise ReferenceContractError(f"unknown source {source_id!r}")

    def case(self, source_id: str, case_id: str) -> CaseReferenceContract:
        source = self.source(source_id)
        for case in source.cases:
            if case.case_id == case_id:
                return case
        raise ReferenceContractError(
            f"unknown case {case_id!r} for source {source_id!r}"
        )


_WHOLE_BODY = "whole_body"
FEATFLOW_RAW_MANIFEST_SCHEMA = "featflow_raw_series_manifest_v1"
_COORDINATE_SIGN_CONVENTION = (
    "turek_hron_2d_v1:x_left_inflow_to_right_outflow;"
    "y_bottom_to_top;point_a_ux_uy_on_xy_axes;"
    "whole_body_drag_plus_x;whole_body_lift_plus_y"
)
_FSI_TESTS_URL = (
    "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/"
    "cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_fsi_tests.html"
)
_FSI_REFERENCE_URL = (
    "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/"
    "cfdbenchmarking/fsi_benchmark/fsi_reference.html"
)
_FSI_DEFINITIONS_URL = (
    "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/"
    "cfdbenchmarking/fsi_benchmark/fsi_definitions.html"
)
_FSI_QUANTITIES_URL = (
    "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/"
    "cfdbenchmarking/fsi_benchmark/fsi_quantities.html"
)
_RAW_FSI2_URL = (
    "https://wwwold.mathematik.tu-dortmund.de/~featflow/media/fsi/data/"
    "fsi2/0p0005/ref_fsi2.point"
)
_RAW_FSI3_URL = (
    "https://wwwold.mathematik.tu-dortmund.de/~featflow/media/fsi/data/"
    "fsi3/0p00025/ref_fsi3.point"
)
_COLUMNS = (
    (1, "time"),
    (5, "beam_drag"),
    (6, "beam_lift"),
    (7, "cylinder_drag"),
    (8, "cylinder_lift"),
    (11, "point_a_ux"),
    (12, "point_a_uy"),
)
_EXTRACTION_POLICY = "last_complete_rising_crossing_bounded_cycle"
_MESH_IDENTITIES = MappingProxyType(
    {
        "2+0": (992, 19488),
        "3+0": (3968, 76672),
        "4+0": (15872, 304128),
        "5+0": (63488, 1211392),
        "6+0": (253952, 4835328),
        "7+0": (1015808, 19320832),
    }
)


def _published_row(
    case_id: str,
    level: str,
    timestep_s: float | None,
    point_a_ux: tuple[float, float | None, float | None],
    point_a_uy: tuple[float, float | None, float | None],
    total_drag: tuple[float, float | None, float | None],
    total_lift: tuple[float, float | None, float | None],
) -> PublishedResultRow:
    nel, ndof = _MESH_IDENTITIES[level]
    return PublishedResultRow(
        case_id, level, nel, ndof, timestep_s, point_a_ux, point_a_uy,
        total_drag, total_lift,
    )


REFERENCE_CATALOG = ReferenceCatalog(
    sources=(
        ReferenceSource(
            source_id=FEATFLOW_WEB_SOURCE_ID,
            role="canonical",
            url=_FSI_TESTS_URL,
            cases=(
                CaseReferenceContract(
                    case_id="fsi1",
                    metrics=(
                        ReferenceMetric(
                            FEATFLOW_WEB_SOURCE_ID, "fsi1", "point_a_ux",
                            "steady_value", 2.270493e-5, "m",
                        ),
                        ReferenceMetric(
                            FEATFLOW_WEB_SOURCE_ID, "fsi1", "point_a_uy",
                            "steady_value", 8.208773e-4, "m",
                        ),
                        ReferenceMetric(
                            FEATFLOW_WEB_SOURCE_ID, "fsi1", "total_drag",
                            "steady_value", 14.29426, "N", _WHOLE_BODY,
                        ),
                        ReferenceMetric(
                            FEATFLOW_WEB_SOURCE_ID, "fsi1", "total_lift",
                            "steady_value", 0.7637460, "N", _WHOLE_BODY,
                        ),
                    ),
                ),
            ),
        ),
        ReferenceSource(
            source_id=FEATFLOW_RAW_FSI2_SOURCE_ID,
            role="canonical",
            url=_RAW_FSI2_URL,
            cases=(
                CaseReferenceContract(
                    case_id="fsi2",
                    metrics=(
                        ReferenceMetric(
                            FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2", "point_a_uy",
                            "amplitude", 0.08165565385, "m",
                        ),
                        ReferenceMetric(
                            FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2", "point_a_uy",
                            "frequency", 1.93061437683808, "Hz",
                        ),
                        ReferenceMetric(
                            FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2", "total_drag",
                            "midrange", 215.088610865, "N", _WHOLE_BODY,
                        ),
                        ReferenceMetric(
                            FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2", "total_lift",
                            "amplitude", 237.661293, "N", _WHOLE_BODY,
                        ),
                    ),
                    raw_series=RawSeriesIdentity(
                        FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2",
                        _RAW_FSI2_URL, "ref_fsi2.point",
                        FEATFLOW_RAW_MANIFEST_SCHEMA, 1783320, 9240, 12,
                        10.0, 14.6195, 0.0005,
                        "d4e192f5ae6aa493d36c472b68d8d734d54194aca6e9f0a70014816b6835dd75",
                        _COLUMNS, (2, 3, 4, 9, 10),
                        (("time", "s"), ("force", "N"), ("point_a_displacement", "m")),
                        _WHOLE_BODY,
                        _COORDINATE_SIGN_CONVENTION,
                        _EXTRACTION_POLICY,
                    ),
                ),
            ),
        ),
        ReferenceSource(
            source_id=FEATFLOW_RAW_FSI3_SOURCE_ID,
            role="canonical",
            url=_RAW_FSI3_URL,
            cases=(
                CaseReferenceContract(
                    case_id="fsi3",
                    metrics=(
                        ReferenceMetric(
                            FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3", "point_a_uy",
                            "amplitude", 0.03491637475, "m",
                        ),
                        ReferenceMetric(
                            FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3", "point_a_uy",
                            "frequency", 5.47355995969522, "Hz",
                        ),
                        ReferenceMetric(
                            FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3", "total_drag",
                            "midrange", 460.31160355, "N", _WHOLE_BODY,
                        ),
                        ReferenceMetric(
                            FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3", "total_lift",
                            "amplitude", 153.527757, "N", _WHOLE_BODY,
                        ),
                    ),
                    raw_series=RawSeriesIdentity(
                        FEATFLOW_RAW_FSI3_SOURCE_ID, "fsi3",
                        _RAW_FSI3_URL, "ref_fsi3.point",
                        FEATFLOW_RAW_MANIFEST_SCHEMA, 1113417, 5769, 12,
                        5.0, 6.4420, 0.00025,
                        "c428bc3ff48c1698cd10aacf0941cba6aabdfcb9d61c8b7e42c343f381434e63",
                        _COLUMNS, (2, 3, 4, 9, 10),
                        (("time", "s"), ("force", "N"), ("point_a_displacement", "m")),
                        _WHOLE_BODY,
                        _COORDINATE_SIGN_CONVENTION,
                        _EXTRACTION_POLICY,
                    ),
                ),
            ),
        ),
        ReferenceSource(
            source_id=LSDYNA_CROSSCHECK_SOURCE_ID,
            role="cross_check",
            url="https://lsdyna.ansys.com/aerofsi1/",
            cases=(
                CaseReferenceContract(
                    case_id="fsi1",
                    metrics=(
                        ReferenceMetric(
                            LSDYNA_CROSSCHECK_SOURCE_ID, "fsi1", "point_a_ux",
                            "steady_value", 1.7e-5, "m",
                        ),
                        ReferenceMetric(
                            LSDYNA_CROSSCHECK_SOURCE_ID, "fsi1", "point_a_uy",
                            "steady_value", 8.6e-4, "m",
                        ),
                        ReferenceMetric(
                            LSDYNA_CROSSCHECK_SOURCE_ID, "fsi1", "total_drag",
                            "steady_value", 14.26, "N", _WHOLE_BODY,
                        ),
                        ReferenceMetric(
                            LSDYNA_CROSSCHECK_SOURCE_ID, "fsi1", "total_lift",
                            "steady_value", 0.73, "N", _WHOLE_BODY, 0.30,
                        ),
                    ),
                ),
                CaseReferenceContract(
                    case_id="fsi3",
                    metrics=(
                        ReferenceMetric(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", "point_a_ux", "midrange", -2.35e-3, "m"),
                        ReferenceMetric(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", "point_a_ux", "amplitude", 2.45e-3, "m"),
                        ReferenceMetric(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", "point_a_uy", "midrange", 1.5e-3, "m"),
                        ReferenceMetric(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", "point_a_uy", "amplitude", 33.5e-3, "m"),
                        ReferenceMetric(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", "total_drag", "midrange", 452.0, "N", _WHOLE_BODY),
                        ReferenceMetric(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", "total_drag", "amplitude", 31.0, "N", _WHOLE_BODY),
                        ReferenceMetric(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", "total_lift", "midrange", 3.3, "N", _WHOLE_BODY),
                        ReferenceMetric(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", "total_lift", "amplitude", 83.1, "N", _WHOLE_BODY),
                    ),
                ),
            ),
        ),
    ),
    definition=BenchmarkDefinition(
        primary_urls=(
            _FSI_TESTS_URL,
            _FSI_REFERENCE_URL,
            _FSI_DEFINITIONS_URL,
            _FSI_QUANTITIES_URL,
        ),
        channel_length_m=2.5,
        channel_height_m=0.41,
        cylinder_center_m=(0.2, 0.2),
        cylinder_radius_m=0.05,
        beam_length_m=0.35,
        beam_thickness_m=0.02,
        point_a_initial_m=(0.6, 0.2),
        fluid_density_kg_m3=1000.0,
        fluid_kinematic_viscosity_m2_s=1.0e-3,
        poisson_ratio=0.4,
        mean_inlet_mps=(("fsi1", 0.2), ("fsi2", 1.0), ("fsi3", 2.0)),
        solid_density_kg_m3=(("fsi1", 1000.0), ("fsi2", 10000.0), ("fsi3", 1000.0)),
        youngs_modulus_pa=(("fsi1", 1.4e6), ("fsi2", 1.4e6), ("fsi3", 5.6e6)),
        shear_modulus_pa=(("fsi1", 0.5e6), ("fsi2", 0.5e6), ("fsi3", 2.0e6)),
        boundary_conditions=(
            "parabolic_inlet_with_documented_2s_cosine_startup_ramp",
            "no_slip_walls_cylinder_and_interface",
            "zero_mean_outlet_pressure_reference",
            "st_venant_kirchhoff_local_plane_strain",
        ),
        coordinate_sign_convention=_COORDINATE_SIGN_CONVENTION,
    ),
    published_results=(
        _published_row("fsi1", "2+0", None, (2.287080e-5, None, None), (8.193038e-4, None, None), (1.427359e1, None, None), (7.617550e-1, None, None)),
        _published_row("fsi1", "3+0", None, (2.277423e-5, None, None), (8.204231e-4, None, None), (1.429177e1, None, None), (7.630484e-1, None, None)),
        _published_row("fsi1", "4+0", None, (2.273175e-5, None, None), (8.207084e-4, None, None), (1.429484e1, None, None), (7.635608e-1, None, None)),
        _published_row("fsi1", "5+0", None, (2.271553e-5, None, None), (8.208126e-4, None, None), (1.429486e1, None, None), (7.636992e-1, None, None)),
        _published_row("fsi1", "6+0", None, (2.270838e-5, None, None), (8.208548e-4, None, None), (1.429451e1, None, None), (7.637359e-1, None, None)),
        _published_row("fsi1", "7+0", None, (2.270493e-5, None, None), (8.208773e-4, None, None), (1.429426e1, None, None), (7.637460e-1, None, None)),
        _published_row("fsi2", "2+0", 0.02, (-1.402e-2, 1.203e-2, 3.85), (1.25e-3, 7.93e-2, 1.93), (2.1010e2, 7.262e1, 3.85), (2.5e-1, 2.279e2, 1.93)),
        _published_row("fsi2", "3+0", 0.02, (-1.454e-2, 1.250e-2, 3.86), (1.25e-3, 8.07e-2, 1.93), (2.1306e2, 7.576e1, 3.86), (8.5e-1, 2.344e2, 1.93)),
        _published_row("fsi2", "4+0", 0.02, (-1.487e-2, 1.273e-2, 3.86), (1.24e-3, 8.17e-2, 1.93), (2.1283e2, 7.589e1, 3.86), (9.2e-1, 2.343e2, 1.93)),
        _published_row("fsi2", "2+0", 0.01, (-1.401e-2, 1.204e-2, 3.86), (1.25e-3, 7.93e-2, 1.93), (2.1009e2, 7.282e1, 3.86), (5.2e-1, 2.286e2, 1.93)),
        _published_row("fsi2", "3+0", 0.01, (-1.454e-2, 1.248e-2, 3.86), (1.25e-3, 8.07e-2, 1.93), (2.1306e2, 7.576e1, 3.86), (8.5e-1, 2.344e2, 1.93)),
        _published_row("fsi2", "4+0", 0.01, (-1.487e-2, 1.273e-2, 3.86), (1.24e-3, 8.17e-2, 1.93), (2.1518e2, 7.778e1, 3.86), (8.7e-1, 2.380e2, 1.93)),
        _published_row("fsi2", "2+0", 0.0005, (-1.401e-2, 1.204e-2, 3.86), (1.28e-3, 7.92e-2, 1.93), (2.1014e2, 7.286e1, 3.86), (4.9e-1, 2.287e2, 1.93)),
        _published_row("fsi2", "3+0", 0.0005, (-1.448e-2, 1.245e-2, 3.86), (1.24e-3, 8.07e-2, 1.93), (2.1305e2, 7.574e1, 3.86), (8.4e-1, 2.348e2, 1.93)),
        _published_row("fsi2", "4+0", 0.0005, (-1.485e-2, 1.270e-2, 3.86), (1.30e-3, 8.16e-2, 1.93), (2.1506e2, 7.765e1, 3.86), (6.1e-1, 2.378e2, 1.93)),
        _published_row("fsi3", "2+0", 0.001, (-3.02e-3, 2.83e-3, 10.75), (1.41e-3, 3.547e-2, 5.37), (4.582e2, 2.832e1, 10.75), (2.41, 1.4558e2, 5.37)),
        _published_row("fsi3", "3+0", 0.001, (-2.78e-3, 2.62e-3, 10.93), (1.44e-3, 3.436e-2, 5.46), (4.591e2, 2.663e1, 10.93), (2.41, 1.5126e2, 5.46)),
        _published_row("fsi3", "4+0", 0.001, (-2.86e-3, 2.70e-3, 10.95), (1.45e-3, 3.493e-2, 5.47), (4.602e2, 2.765e1, 10.95), (2.47, 1.5487e2, 5.47)),
        _published_row("fsi3", "2+0", 0.0005, (-3.02e-3, 2.85e-3, 10.75), (1.42e-3, 3.563e-2, 5.37), (4.587e2, 2.878e1, 10.75), (2.23, 1.4602e2, 5.37)),
        _published_row("fsi3", "3+0", 0.0005, (-2.78e-3, 2.62e-3, 10.92), (1.44e-3, 3.435e-2, 5.46), (4.591e2, 2.662e1, 10.92), (2.39, 1.5068e2, 5.46)),
        _published_row("fsi3", "4+0", 0.0005, (-2.86e-3, 2.70e-3, 10.92), (1.45e-3, 3.490e-2, 5.46), (4.602e2, 2.747e1, 10.92), (2.37, 1.5375e2, 5.46)),
        _published_row("fsi3", "2+0", 0.00025, (-3.02e-3, 2.85e-3, 10.74), (1.32e-3, 3.573e-2, 5.36), (4.587e2, 2.880e1, 10.74), (2.23, 1.4600e2, 5.33)),
        _published_row("fsi3", "3+0", 0.00025, (-2.77e-3, 2.61e-3, 10.93), (1.43e-3, 3.443e-2, 5.46), (4.591e2, 2.650e1, 10.93), (2.36, 1.4991e2, 5.46)),
        _published_row("fsi3", "4+0", 0.00025, (-2.88e-3, 2.72e-3, 10.93), (1.47e-3, 3.499e-2, 5.46), (4.605e2, 2.774e1, 10.93), (2.50, 1.5391e2, 5.46)),
    ),
)


def case_reference_contract(source_id: str, case_id: str) -> CaseReferenceContract:
    return REFERENCE_CATALOG.case(source_id, case_id)


def raw_series_identity(source_id: str, case_id: str) -> RawSeriesIdentity:
    """Return the sole contracted raw identity, never a source fallback."""

    identity = case_reference_contract(source_id, case_id).raw_series
    if identity is None:
        raise ReferenceContractError(
            f"no raw-series identity for case {case_id!r} and source {source_id!r}"
        )
    return identity


def published_result(
    case_id: str, level: str, timestep_s: float | None
) -> PublishedResultRow:
    """Return one exact Featflow table row or fail closed."""

    for row in REFERENCE_CATALOG.published_results:
        if (
            row.case_id == case_id
            and row.level == level
            and row.timestep_s == timestep_s
        ):
            return row
    raise ReferenceContractError(
        "unknown published Featflow row "
        f"for case {case_id!r}, level {level!r}, timestep {timestep_s!r}"
    )


def reference_metric(
    source_id: str, case_id: str, quantity: str, statistic: str
) -> ReferenceMetric:
    case = case_reference_contract(source_id, case_id)
    for metric in case.metrics:
        if metric.quantity == quantity and metric.statistic == statistic:
            return metric
    raise ReferenceContractError(
        "unknown metric "
        f"{quantity!r}/{statistic!r} for case {case_id!r} and source {source_id!r}"
    )


def _metric_values(
    source_id: str, case_id: str, fields: tuple[tuple[str, str, str], ...]
) -> Mapping[str, float]:
    return MappingProxyType(
        {
            field: reference_metric(source_id, case_id, quantity, statistic).value
            for field, quantity, statistic in fields
        }
    )


_FSI1_FIELDS = (
    ("tip_ux_turek_hron_m", "point_a_ux", "steady_value"),
    ("tip_uy_turek_hron_m", "point_a_uy", "steady_value"),
    ("total_drag_per_span_n_per_m", "total_drag", "steady_value"),
    ("total_lift_per_span_n_per_m", "total_lift", "steady_value"),
)


def canonical_fsi1_metric_values() -> Mapping[str, float]:
    return _metric_values(FEATFLOW_WEB_SOURCE_ID, "fsi1", _FSI1_FIELDS)


def lsdyna_fsi1_metric_values() -> Mapping[str, float]:
    return _metric_values(LSDYNA_CROSSCHECK_SOURCE_ID, "fsi1", _FSI1_FIELDS)


def lsdyna_fsi1_uncertainty_values() -> Mapping[str, float]:
    return MappingProxyType(
        {
            field: float(metric.uncertainty)
            for field, quantity, statistic in _FSI1_FIELDS
            if (
                metric := reference_metric(
                    LSDYNA_CROSSCHECK_SOURCE_ID, "fsi1", quantity, statistic
                )
            ).uncertainty is not None
        }
    )


def legacy_case_reference_projection() -> Mapping[str, Mapping[str, object]]:
    fsi1 = canonical_fsi1_metric_values()
    fsi3 = lambda quantity, statistic: reference_metric(
        LSDYNA_CROSSCHECK_SOURCE_ID, "fsi3", quantity, statistic
    ).value
    return MappingProxyType(
        {
            "fsi1": MappingProxyType(
                {
                    "source": "Featflow TU Dortmund published level 7+0",
                    "source_id": FEATFLOW_WEB_SOURCE_ID,
                    "ux_a_m": fsi1["tip_ux_turek_hron_m"],
                    "uy_a_m": fsi1["tip_uy_turek_hron_m"],
                    "drag_n_per_m": fsi1["total_drag_per_span_n_per_m"],
                    "lift_n_per_m": fsi1["total_lift_per_span_n_per_m"],
                    "regime": "steady",
                }
            ),
            "fsi2": MappingProxyType(
                {
                    "source": "Featflow TU Dortmund raw FSI2 reference",
                    "source_id": FEATFLOW_RAW_FSI2_SOURCE_ID,
                    "regime": "periodic-large-amplitude",
                }
            ),
            "fsi3": MappingProxyType(
                {
                    "source": "LS-DYNA ICFD aerofsi1 report (mean +/- amplitude)",
                    "source_id": LSDYNA_CROSSCHECK_SOURCE_ID,
                    "ux_a_mean_m": fsi3("point_a_ux", "midrange"),
                    "ux_a_amplitude_m": fsi3("point_a_ux", "amplitude"),
                    "uy_a_mean_m": fsi3("point_a_uy", "midrange"),
                    "uy_a_amplitude_m": fsi3("point_a_uy", "amplitude"),
                    "drag_mean_n_per_m": fsi3("total_drag", "midrange"),
                    "drag_amplitude_n_per_m": fsi3("total_drag", "amplitude"),
                    "lift_mean_n_per_m": fsi3("total_lift", "midrange"),
                    "lift_amplitude_n_per_m": fsi3("total_lift", "amplitude"),
                    "regime": "periodic",
                }
            ),
        }
    )


__all__ = [
    "BenchmarkDefinition",
    "CaseReferenceContract",
    "FEATFLOW_RAW_MANIFEST_SCHEMA",
    "FEATFLOW_RAW_FSI2_SOURCE_ID",
    "FEATFLOW_RAW_FSI3_SOURCE_ID",
    "FEATFLOW_WEB_SOURCE_ID",
    "LSDYNA_CROSSCHECK_SOURCE_ID",
    "PublishedResultRow",
    "REFERENCE_CATALOG",
    "RawSeriesIdentity",
    "ReferenceCatalog",
    "ReferenceContractError",
    "ReferenceMetric",
    "ReferenceSource",
    "canonical_fsi1_metric_values",
    "case_reference_contract",
    "legacy_case_reference_projection",
    "lsdyna_fsi1_metric_values",
    "lsdyna_fsi1_uncertainty_values",
    "published_result",
    "raw_series_identity",
    "reference_metric",
]
