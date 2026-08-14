from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.refactored.validation.ansys_vertical_flap_fsi.native_fine_comparison import (
    NativeFineComparisonError,
    postprocess_native_fine_comparison,
)
from src.refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (
    save_solver_npz_from_flow_snapshot,
)
from src.refactored.validation.ansys_vertical_flap_fsi.official_fluent_reference import (
    FluentFieldBundle,
    _write_field_npz,
)

from .test_native_fine_comparison import (
    FLUENT_CHECKSUM_RELATIVE_PATHS,
    _synthetic_inputs,
    _write_checksums,
)


PRESSURE_QUANTITY = "static_gauge_pressure_pa"
PRESSURE_REFERENCE = "outlet_0_pa"
POSTPROCESS_CLI = (
    Path(__file__).resolve().parents[2]
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_vs_native_fluent_fine_2026-07-10"
    / "scripts"
    / "postprocess_our_solver_vs_native_fluent.py"
)


def _flow_snapshot() -> dict[str, np.ndarray]:
    shape = (2, 2, 3)
    velocity = np.zeros((*shape, 3), dtype=np.float64)
    velocity[..., 1] = 2.0
    velocity[..., 2] = -3.0
    center_y = np.zeros(shape, dtype=np.float64)
    center_z = np.zeros(shape, dtype=np.float64)
    center_y[:, 1, :] = 0.02
    center_z[:, :, 1] = 0.5
    center_z[:, :, 2] = 1.0
    return {
        "pressure": np.ones(shape, dtype=np.float64),
        "velocity": velocity,
        "obstacle": np.zeros(shape, dtype=np.int32),
        "cell_center_y_m": center_y,
        "cell_center_z_m": center_z,
    }


def _fluent_bundle(tmp_path: Path) -> FluentFieldBundle:
    values = np.asarray([0.0, 1.0], dtype=np.float64)
    return FluentFieldBundle(
        case_path=tmp_path / "reference.cas.h5",
        data_path=tmp_path / "reference.dat.h5",
        cell_ids=np.asarray([1, 2], dtype=np.int64),
        x=values,
        y=values,
        u=values,
        v=values,
        p=values,
        speed=np.sqrt(2.0) * values,
        mesh_summary={"cell_count": 2},
        field_summary={"cell_count": 2},
    )


def _assert_pressure_metadata(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        quantity = data["pressure_quantity"]
        reference = data["pressure_reference"]
        assert quantity.shape == ()
        assert reference.shape == ()
        assert quantity.dtype.kind == "U"
        assert reference.dtype.kind == "U"
        assert quantity.item() == PRESSURE_QUANTITY
        assert reference.item() == PRESSURE_REFERENCE


def _rewrite_npz(
    path: Path,
    *,
    pressure_quantity: object | None = None,
    pressure_reference: object | None = None,
) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.array(data[name], copy=True) for name in data.files}
    if pressure_quantity is not None:
        payload["pressure_quantity"] = np.asarray(pressure_quantity)
    if pressure_reference is not None:
        payload["pressure_reference"] = np.asarray(pressure_reference)
    np.savez_compressed(path, **payload)


def _comparison_field_paths(
    our_dir: Path,
    fluent_dir: Path,
    *,
    steps: int,
) -> tuple[Path, Path]:
    return (
        our_dir / "step_fields" / f"step_{steps:04d}.npz",
        fluent_dir / "fields" / "final_fields.npz",
    )


def _seal_fluent_bundle(fluent_dir: Path) -> None:
    _write_checksums(fluent_dir, FLUENT_CHECKSUM_RELATIVE_PATHS)


def test_our_solver_field_writer_declares_pressure_semantics(tmp_path: Path) -> None:
    path = tmp_path / "our_fields.npz"

    save_solver_npz_from_flow_snapshot(path, _flow_snapshot())

    _assert_pressure_metadata(path)


def test_fluent_field_writer_declares_pressure_semantics(tmp_path: Path) -> None:
    path = tmp_path / "fluent_fields.npz"

    _write_field_npz(path, _fluent_bundle(tmp_path))

    _assert_pressure_metadata(path)


def test_strict_pressure_semantics_rejects_fully_legacy_metadata(
    tmp_path: Path,
) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path, steps=3)
    output_dir = tmp_path / "strict_output"

    with pytest.raises(NativeFineComparisonError, match="pressure semantics"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            output_dir,
            expected_steps=3,
            pressure_semantics_mode="strict",
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("mode", ("legacy_compatible", "strict"))
@pytest.mark.parametrize("source", ("our_solver", "native_fluent"))
@pytest.mark.parametrize(
    ("quantity", "reference"),
    (
        (PRESSURE_QUANTITY, None),
        (None, PRESSURE_REFERENCE),
        ("total_pressure_pa", PRESSURE_REFERENCE),
        (PRESSURE_QUANTITY, "operating_pressure_pa"),
    ),
)
def test_partial_or_explicitly_wrong_pressure_semantics_always_fail_closed(
    tmp_path: Path,
    mode: str,
    source: str,
    quantity: str,
    reference: str | None,
) -> None:
    steps = 3
    our_dir, fluent_dir = _synthetic_inputs(tmp_path, steps=steps)
    our_fields, fluent_fields = _comparison_field_paths(
        our_dir,
        fluent_dir,
        steps=steps,
    )
    target = our_fields if source == "our_solver" else fluent_fields
    _rewrite_npz(
        target,
        pressure_quantity=quantity,
        pressure_reference=reference,
    )
    if source == "native_fluent":
        _seal_fluent_bundle(fluent_dir)

    with pytest.raises(NativeFineComparisonError, match="pressure semantics"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=steps,
            pressure_semantics_mode=mode,
        )


@pytest.mark.parametrize(
    ("malformed_quantity", "error_match"),
    (
        ([PRESSURE_QUANTITY], "scalar Unicode"),
        (PRESSURE_QUANTITY.encode("utf-8"), "scalar Unicode"),
        (
            np.asarray(PRESSURE_QUANTITY, dtype=object),
            "Object arrays cannot be loaded",
        ),
    ),
    ids=("non_scalar_unicode", "scalar_bytes", "object_dtype"),
)
def test_pressure_semantics_metadata_must_be_scalar_unicode(
    tmp_path: Path,
    malformed_quantity: object,
    error_match: str,
) -> None:
    steps = 3
    our_dir, fluent_dir = _synthetic_inputs(tmp_path, steps=steps)
    our_fields, _ = _comparison_field_paths(our_dir, fluent_dir, steps=steps)
    _rewrite_npz(
        our_fields,
        pressure_quantity=malformed_quantity,
        pressure_reference=PRESSURE_REFERENCE,
    )

    with pytest.raises(NativeFineComparisonError, match=error_match):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=steps,
            pressure_semantics_mode="strict",
        )


def test_exact_strict_pressure_semantics_preserves_all_numeric_comparison(
    tmp_path: Path,
) -> None:
    steps = 3
    legacy_our, legacy_fluent = _synthetic_inputs(
        tmp_path / "legacy",
        steps=steps,
    )
    legacy_report = postprocess_native_fine_comparison(
        legacy_our,
        legacy_fluent,
        tmp_path / "legacy_output",
        expected_steps=steps,
    )

    strict_our, strict_fluent = _synthetic_inputs(
        tmp_path / "strict",
        steps=steps,
    )
    strict_our_fields, strict_fluent_fields = _comparison_field_paths(
        strict_our,
        strict_fluent,
        steps=steps,
    )
    for path in (strict_our_fields, strict_fluent_fields):
        _rewrite_npz(
            path,
            pressure_quantity=PRESSURE_QUANTITY,
            pressure_reference=PRESSURE_REFERENCE,
        )
    _seal_fluent_bundle(strict_fluent)
    strict_report = postprocess_native_fine_comparison(
        strict_our,
        strict_fluent,
        tmp_path / "strict_output",
        expected_steps=steps,
        pressure_semantics_mode="strict",
    )

    assert strict_report["pressure_semantics_contract"]["status"] == "passed"
    assert strict_report["final_field_comparison"] == legacy_report[
        "final_field_comparison"
    ]


def test_official_postprocess_cli_requests_strict_pressure_semantics(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("pressure_semantics_cli", POSTPROCESS_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict[str, Any] = {}

    def fake_postprocess(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "status": "diagnostic_complete",
            "five_percent_diagnostic_gate": {
                "status": "passed",
                "all_metrics_within_tolerance": True,
            },
        }

    module.postprocess_native_fine_comparison = fake_postprocess
    exit_code = module.main(
        [
            "--our-run-dir",
            str(tmp_path / "our"),
            "--fluent-postprocess-dir",
            str(tmp_path / "fluent"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 0
    assert captured["pressure_semantics_mode"] == "strict"
