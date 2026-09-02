from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.refactored.validation.turek_hron_fsi.featflow import (
    FeatflowValidationError,
    _rows,
    load_featflow_series,
)
from src.refactored.validation.turek_hron_fsi.references import (
    FEATFLOW_RAW_FSI2_SOURCE_ID,
    raw_series_identity,
)


_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "validation"
    / "turek_hron_featflow"
)


def _manifest_data(raw: bytes) -> dict[str, object]:
    """A deliberately self-consistent five-row imposter, not a valid identity."""

    identity = raw_series_identity(FEATFLOW_RAW_FSI2_SOURCE_ID, "fsi2")
    return {
        "schema_version": identity.schema_version,
        "source_id": identity.source_id,
        "case_id": identity.case_id,
        "url": "https://example.invalid/fsi2/0p0005/ref_fsi2.point",
        "filename": identity.filename,
        "byte_count": len(raw),
        "row_count": 5,
        "column_count": identity.column_count,
        "time_start_s": 0.0,
        "time_end_s": 1.0,
        "time_step_s": 0.25,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "documented_columns": {
            str(column): meaning for column, meaning in identity.documented_columns
        },
        "opaque_columns": list(identity.opaque_columns),
        "units": dict(identity.units),
        "force_scope": identity.force_scope,
        "coordinate_convention": identity.coordinate_sign_convention,
        "extraction_policy": identity.extraction_policy,
    }


def test_self_consistent_synthetic_example_url_cannot_masquerade_as_fsi2(
    tmp_path: Path,
) -> None:
    raw = b"".join(
        (
            f"{time} 2 3 4 100 -25 -88 39 9 10 11 12\n".encode()
            for time in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
    )
    raw_path = tmp_path / "ref_fsi2.point"
    manifest_path = tmp_path / "ref_fsi2.manifest.json"
    raw_path.write_bytes(raw)
    manifest_path.write_text(json.dumps(_manifest_data(raw)), encoding="utf-8")

    with pytest.raises(FeatflowValidationError, match="exact contracted raw source"):
        load_featflow_series(raw_path, manifest_path)


@pytest.mark.parametrize(
    ("text", "message"),
    (
        ("0 1 2 3 4 5 6 7 8 9 10\n", "12 columns"),
        ("0 1 bad 3 4 5 6 7 8 9 10 11\n", "finite numeric"),
        ("0 1 2 3 4 nan 6 7 8 9 10 11\n", "finite numeric"),
        ("\n# comment\n", "no data rows"),
    ),
)
def test_raw_row_parser_rejects_malformed_or_nonfinite_rows(
    text: str, message: str
) -> None:
    with pytest.raises(FeatflowValidationError, match=message):
        _rows(text.encode())


def _write_canonical_manifest(tmp_path: Path, **changes: object) -> tuple[Path, Path]:
    raw_path = _ARTIFACT_ROOT / "ref_fsi2.point"
    data = json.loads((_ARTIFACT_ROOT / "ref_fsi2.manifest.json").read_text())
    data.update(changes)
    manifest_path = tmp_path / "ref_fsi2.manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    return raw_path, manifest_path


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"source_id": "unknown"}, "source/case"),
        ({"case_id": "fsi3"}, "source/case"),
        ({"url": "https://example.invalid/fsi2/0p0005/ref_fsi2.point"}, "exact contracted raw source"),
        ({"row_count": 5}, "exact contracted count"),
        ({"byte_count": 5}, "exact contracted size"),
        ({"column_count": 11}, "exact contracted count"),
        ({"time_step_s": 0.1}, "time contract"),
        ({"force_scope": "beam"}, "exact contracted scope"),
        ({"coordinate_convention": "unknown"}, "coordinate_convention"),
        ({"documented_columns": {"1": "time"}}, "documented_columns"),
        ({"opaque_columns": [2]}, "opaque_columns"),
        ({"units": {}}, "units"),
        ({"sha256": "0" * 64}, "exact contracted digest"),
    ),
)
def test_manifest_cannot_drift_from_central_identity(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    raw_path, manifest_path = _write_canonical_manifest(tmp_path, **changes)
    with pytest.raises(FeatflowValidationError, match=message):
        load_featflow_series(raw_path, manifest_path)


def test_manifest_fields_are_closed(tmp_path: Path) -> None:
    raw_path, manifest_path = _write_canonical_manifest(tmp_path)
    data = json.loads(manifest_path.read_text())
    data["unexpected"] = 1
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FeatflowValidationError, match="manifest fields"):
        load_featflow_series(raw_path, manifest_path)
