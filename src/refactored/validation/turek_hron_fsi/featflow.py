"""Fail-closed importer for immutable Featflow raw reference series."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .references import (
    FEATFLOW_RAW_MANIFEST_SCHEMA,
    RawSeriesIdentity,
    ReferenceContractError,
    raw_series_identity,
)


class FeatflowValidationError(ValueError):
    """Raised when a raw Featflow artifact fails its manifest contract."""


_FIELDS = frozenset({
    "schema_version", "source_id", "case_id", "url", "filename", "byte_count",
    "row_count", "column_count", "time_start_s", "time_end_s", "time_step_s",
    "sha256", "documented_columns", "opaque_columns", "units", "force_scope",
    "coordinate_convention", "extraction_policy",
})


@dataclass(frozen=True)
class FeatflowManifest:
    schema_version: str
    source_id: str
    case_id: str
    url: str
    filename: str
    byte_count: int
    row_count: int
    column_count: int
    time_start_s: float
    time_end_s: float
    time_step_s: float
    sha256: str
    documented_columns: Mapping[str, str]
    opaque_columns: tuple[int, ...]
    units: Mapping[str, str]
    force_scope: str
    coordinate_convention: str
    extraction_policy: str


@dataclass(frozen=True)
class FeatflowSeries:
    manifest: FeatflowManifest
    values: np.ndarray

    @property
    def case_id(self) -> str:
        return self.manifest.case_id

    @property
    def time_s(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def beam_drag_n(self) -> np.ndarray:
        return self.values[:, 4]

    @property
    def beam_lift_n(self) -> np.ndarray:
        return self.values[:, 5]

    @property
    def cylinder_drag_n(self) -> np.ndarray:
        return self.values[:, 6]

    @property
    def cylinder_lift_n(self) -> np.ndarray:
        return self.values[:, 7]

    def _immutable_sum(self, first: int, second: int) -> np.ndarray:
        total = self.values[:, first] + self.values[:, second]
        return np.frombuffer(total.tobytes(), dtype=np.float64)

    @property
    def total_drag_n(self) -> np.ndarray:
        return self._immutable_sum(4, 6)

    @property
    def total_lift_n(self) -> np.ndarray:
        return self._immutable_sum(5, 7)

    @property
    def point_a_ux_m(self) -> np.ndarray:
        return self.values[:, 10]

    @property
    def point_a_uy_m(self) -> np.ndarray:
        return self.values[:, 11]


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FeatflowValidationError(f"manifest repeats field {key!r}")
        result[key] = value
    return result


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FeatflowValidationError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FeatflowValidationError(f"{field} must be a non-negative integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatflowValidationError(f"{field} must be a finite numeric value")
    value = float(value)
    if not np.isfinite(value):
        raise FeatflowValidationError(f"{field} must be a finite numeric value")
    return value


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatflowValidationError(f"cannot parse manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FeatflowValidationError("manifest must be a JSON object")
    if set(data) != _FIELDS:
        raise FeatflowValidationError(
            "manifest fields must exactly match v1; "
            f"missing={sorted(_FIELDS - set(data))}, extra={sorted(set(data) - _FIELDS)}"
        )
    return data


def _manifest(data: Mapping[str, Any], raw_path: Path) -> FeatflowManifest:
    schema = _string(data["schema_version"], "schema_version")
    if schema != FEATFLOW_RAW_MANIFEST_SCHEMA:
        raise FeatflowValidationError(
            f"schema_version must be {FEATFLOW_RAW_MANIFEST_SCHEMA!r}"
        )
    case = _string(data["case_id"], "case_id")
    source = _string(data["source_id"], "source_id")
    try:
        identity = raw_series_identity(source, case)
    except ReferenceContractError as exc:
        raise FeatflowValidationError(
            f"source/case is not a contracted raw identity: {exc}"
        ) from exc
    _require_identity(data, raw_path, identity)
    return FeatflowManifest(
        schema, source, case, identity.url, identity.filename, identity.byte_count,
        identity.row_count, identity.column_count, identity.time_start_s,
        identity.time_end_s, identity.dt_s, identity.sha256,
        MappingProxyType(dict((str(column), meaning) for column, meaning in identity.documented_columns)),
        identity.opaque_columns, MappingProxyType(dict(identity.units)),
        identity.force_scope, identity.coordinate_sign_convention,
        identity.extraction_policy,
    )


def _require_identity(
    data: Mapping[str, Any], raw_path: Path, identity: RawSeriesIdentity
) -> None:
    filename = _string(data["filename"], "filename")
    if filename != identity.filename or filename != raw_path.name:
        raise FeatflowValidationError("filename does not bind the supplied raw file")
    url = _string(data["url"], "url")
    if url != identity.url:
        raise FeatflowValidationError("url is not the exact contracted raw source")
    expected_columns = {
        str(column): meaning for column, meaning in identity.documented_columns
    }
    if data["documented_columns"] != expected_columns:
        raise FeatflowValidationError("documented_columns must be the exact one-based map")
    if data["opaque_columns"] != list(identity.opaque_columns):
        raise FeatflowValidationError("opaque_columns are not the exact contracted columns")
    if data["units"] != dict(identity.units):
        raise FeatflowValidationError("units must be the exact documented mapping")
    if _string(data["force_scope"], "force_scope") != identity.force_scope:
        raise FeatflowValidationError("force_scope is not the exact contracted scope")
    if (
        _string(data["coordinate_convention"], "coordinate_convention")
        != identity.coordinate_sign_convention
    ):
        raise FeatflowValidationError("coordinate_convention is not frozen")
    if (
        _string(data["extraction_policy"], "extraction_policy")
        != identity.extraction_policy
    ):
        raise FeatflowValidationError("extraction_policy is not frozen")
    digest = _string(data["sha256"], "sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise FeatflowValidationError("SHA256 must be a 64-character lowercase hex digest")
    if digest != identity.sha256:
        raise FeatflowValidationError("sha256 is not the exact contracted digest")
    if _integer(data["byte_count"], "byte_count") != identity.byte_count:
        raise FeatflowValidationError("byte_count is not the exact contracted size")
    if _integer(data["row_count"], "row_count") != identity.row_count:
        raise FeatflowValidationError("row_count is not the exact contracted count")
    if _integer(data["column_count"], "column_count") != identity.column_count:
        raise FeatflowValidationError("column_count is not the exact contracted count")
    start = _number(data["time_start_s"], "time_start_s")
    end = _number(data["time_end_s"], "time_end_s")
    dt = _number(data["time_step_s"], "time_step_s")
    if (
        start != identity.time_start_s
        or end != identity.time_end_s
        or dt != identity.dt_s
    ):
        raise FeatflowValidationError("time contract is not the exact raw identity")


def _rows(raw: bytes) -> np.ndarray:
    result: list[list[float]] = []
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise FeatflowValidationError("raw series must be UTF-8 text") from exc
    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 12:
            raise FeatflowValidationError(
                f"raw row {line_number} must have exactly 12 columns; got {len(fields)}"
            )
        try:
            row = [float(field) for field in fields]
        except ValueError as exc:
            raise FeatflowValidationError(
                f"raw row {line_number} must contain finite numeric values"
            ) from exc
        if not np.all(np.isfinite(row)):
            raise FeatflowValidationError(
                f"raw row {line_number} must contain finite numeric values"
            )
        result.append(row)
    if not result:
        raise FeatflowValidationError("raw series has no data rows")
    return np.asarray(result, dtype=np.float64)


def _time(values: np.ndarray, manifest: FeatflowManifest) -> None:
    time = values[:, 0]
    differences = np.diff(time)
    if np.any(differences <= 0.0):
        raise FeatflowValidationError("raw time must be strictly increasing")
    tolerance = max(1.0e-12, abs(manifest.time_step_s) * 1.0e-9)
    if not np.allclose(differences, manifest.time_step_s, rtol=1.0e-9, atol=tolerance):
        raise FeatflowValidationError("raw time must be uniform at manifest time_step_s")
    if not np.isclose(time[0], manifest.time_start_s, rtol=1.0e-9, atol=tolerance):
        raise FeatflowValidationError("time_start_s does not match raw data")
    if not np.isclose(time[-1], manifest.time_end_s, rtol=1.0e-9, atol=tolerance):
        raise FeatflowValidationError("time_end_s does not match raw data")


def load_featflow_series(raw_path: str | Path, manifest_path: str | Path) -> FeatflowSeries:
    """Load raw data only after its identity and numerical contract verify."""

    raw_file = Path(raw_path)
    manifest = _manifest(_read_manifest(Path(manifest_path)), raw_file)
    try:
        raw = raw_file.read_bytes()
    except OSError as exc:
        raise FeatflowValidationError(f"cannot read raw series {raw_file}: {exc}") from exc
    if len(raw) != manifest.byte_count:
        raise FeatflowValidationError("byte_count does not match raw data")
    if hashlib.sha256(raw).hexdigest() != manifest.sha256:
        raise FeatflowValidationError("sha256 does not match raw data")
    values = _rows(raw)
    if values.shape != (manifest.row_count, 12):
        raise FeatflowValidationError(
            f"row_count does not match raw data; expected {manifest.row_count}, got {len(values)}"
        )
    _time(values, manifest)
    immutable = np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(values.shape)
    return FeatflowSeries(manifest, immutable)
