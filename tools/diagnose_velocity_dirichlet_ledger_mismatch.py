"""Capture an exact host-side diff when the device row ledger fails closed.

This is a diagnostic wrapper, not a production tolerance.  It delegates to the
solver's existing exact comparison first and only performs CPU readback after
that comparison reports a mismatch.  The wrapped runner therefore keeps the
same fail-closed exit and the preflow snapshot source identity stays unchanged.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_core.fluids.solver import CartesianFluidSolver


_CAPTURED = False
_COMPARISON_CALL_INDEX = 0


def _json_scalar(value: np.generic | int | float | bool | str) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _float_bits_hex(value: np.generic, dtype: np.dtype[Any]) -> str:
    scalar = np.asarray(value, dtype=dtype).reshape(())
    if dtype.itemsize == 4:
        return f"0x{int(scalar.view(np.uint32)):08x}"
    if dtype.itemsize == 8:
        return f"0x{int(scalar.view(np.uint64)):016x}"
    raise ValueError(f"unsupported floating dtype for exact bits: {dtype}")


def _array_difference_summary(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    component_axis: bool,
    max_examples: int = 32,
) -> dict[str, object]:
    """Summarize exact element and logical-row differences between arrays."""

    current_array = np.asarray(current)
    reference_array = np.asarray(reference)
    if current_array.shape != reference_array.shape:
        raise ValueError(
            "ledger diagnostic array shapes differ: "
            f"current={current_array.shape}, reference={reference_array.shape}"
        )
    if current_array.dtype != reference_array.dtype:
        raise ValueError(
            "ledger diagnostic array dtypes differ: "
            f"current={current_array.dtype}, reference={reference_array.dtype}"
        )
    if component_axis and (
        current_array.ndim < 1 or current_array.shape[-1] != 3
    ):
        raise ValueError(
            "component-axis ledger array must end in three components: "
            f"shape={current_array.shape}"
        )

    element_mismatch = np.not_equal(current_array, reference_array)
    row_mismatch = (
        np.any(element_mismatch, axis=-1)
        if component_axis
        else element_mismatch
    )
    examples: list[dict[str, object]] = []
    for raw_index in np.argwhere(element_mismatch)[: max(0, int(max_examples))]:
        index = tuple(int(value) for value in raw_index)
        current_value = current_array[index]
        reference_value = reference_array[index]
        example: dict[str, object] = {
            "index": list(index),
            "current": _json_scalar(current_value),
            "reference": _json_scalar(reference_value),
        }
        if np.issubdtype(current_array.dtype, np.floating):
            example["current_bits_hex"] = _float_bits_hex(
                current_value,
                current_array.dtype,
            )
            example["reference_bits_hex"] = _float_bits_hex(
                reference_value,
                reference_array.dtype,
            )
        examples.append(example)

    return {
        "shape": list(current_array.shape),
        "dtype": str(current_array.dtype),
        "mismatch_row_count": int(np.count_nonzero(row_mismatch)),
        "mismatch_element_count": int(np.count_nonzero(element_mismatch)),
        "examples": examples,
    }


def _array_pair(
    solver: CartesianFluidSolver,
    current_name: str,
    reference_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        getattr(solver, current_name).to_numpy(),
        getattr(solver, reference_name).to_numpy(),
    )


def _canonical_field_specs() -> tuple[tuple[str, str, str, bool], ...]:
    return (
        (
            "canonical.active_component_mask",
            "velocity_dirichlet_boundary_active_component_mask",
            "_velocity_dirichlet_component_ledger_reference_active_component_mask",
            False,
        ),
        (
            "canonical.value_mps",
            "velocity_dirichlet_boundary_value_mps",
            "_velocity_dirichlet_component_ledger_reference_value_mps",
            True,
        ),
        (
            "canonical.pressure_mobility",
            "velocity_dirichlet_boundary_pressure_mobility",
            "_velocity_dirichlet_component_ledger_reference_pressure_mobility",
            True,
        ),
        (
            "canonical.component_enforcement_weight",
            "velocity_dirichlet_boundary_component_enforcement_weight",
            "_velocity_dirichlet_component_ledger_reference_component_enforcement_weight",
            True,
        ),
        (
            "canonical.component_region_id",
            "velocity_dirichlet_boundary_component_region_id",
            "_velocity_dirichlet_component_ledger_reference_component_region_id",
            True,
        ),
        (
            "canonical.hard_fixed_component_mask",
            "velocity_dirichlet_boundary_hard_fixed_component_mask",
            "_velocity_dirichlet_component_ledger_reference_hard_fixed_component_mask",
            False,
        ),
        (
            "canonical.external_exact_component_mask",
            "velocity_dirichlet_boundary_external_exact_component_mask",
            "_velocity_dirichlet_component_ledger_reference_external_exact_component_mask",
            False,
        ),
        (
            "canonical.owned_component_mask",
            "velocity_dirichlet_boundary_owned_component_mask",
            "_velocity_dirichlet_component_ledger_reference_owned_component_mask",
            False,
        ),
        (
            "canonical.obstacle",
            "obstacle",
            "_velocity_dirichlet_ledger_reference_obstacle",
            False,
        ),
    )


def _external_field_specs() -> tuple[tuple[str, str, str, bool], ...]:
    specs: list[tuple[str, str, str, bool]] = []
    for axis in ("x", "y", "z"):
        specs.extend(
            (
                (
                    f"external_{axis}.active_component_mask",
                    f"external_velocity_boundary_{axis}_face_active_component_mask",
                    f"_velocity_dirichlet_ledger_reference_external_{axis}_face_active_component_mask",
                    False,
                ),
                (
                    f"external_{axis}.value_mps",
                    f"external_velocity_boundary_{axis}_face_value_mps",
                    f"_velocity_dirichlet_ledger_reference_external_{axis}_face_value_mps",
                    True,
                ),
            )
        )
    return tuple(specs)


def _row_mismatch_mask(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    component_axis: bool,
) -> np.ndarray:
    element_mismatch = np.not_equal(current, reference)
    return (
        np.any(element_mismatch, axis=-1)
        if component_axis
        else element_mismatch
    )


def _capture_mismatch(
    solver: CartesianFluidSolver,
    *,
    expected_generation: int,
    kernel_mismatch_rows: int,
    comparison_call_index: int,
) -> dict[str, object]:
    authority = solver._validated_velocity_dirichlet_boundary_authority(
        solver.velocity_dirichlet_boundary_authority
    )
    reference_authority = str(
        solver._velocity_dirichlet_ledger_reference_authority
    )
    metadata = {
        "authority": {
            "current": str(authority),
            "reference": reference_authority,
            "mismatch": str(authority) != reference_authority,
        },
        "component_ledger_generation": {
            "current": int(solver.velocity_dirichlet_component_ledger_generation),
            "reference": int(
                solver._velocity_dirichlet_ledger_reference_component_generation
            ),
            "mismatch": int(solver.velocity_dirichlet_component_ledger_generation)
            != int(solver._velocity_dirichlet_ledger_reference_component_generation),
        },
        "face_symmetric": {
            "current": int(solver.velocity_dirichlet_face_symmetric),
            "reference": int(
                solver._velocity_dirichlet_ledger_reference_face_symmetric
            ),
            "mismatch": int(solver.velocity_dirichlet_face_symmetric)
            != int(solver._velocity_dirichlet_ledger_reference_face_symmetric),
        },
    }

    field_summaries: dict[str, dict[str, object]] = {}
    row_groups: dict[str, np.ndarray] = {}
    field_specs = list(_external_field_specs())
    if authority == "canonical":
        field_specs = [*_canonical_field_specs(), *field_specs]

    first_mismatch_field: str | None = None
    for field_name, current_name, reference_name, component_axis in field_specs:
        current, reference = _array_pair(solver, current_name, reference_name)
        row_mask = _row_mismatch_mask(
            current,
            reference,
            component_axis=component_axis,
        )
        if not np.any(row_mask):
            continue
        summary = _array_difference_summary(
            current,
            reference,
            component_axis=component_axis,
        )
        field_summaries[field_name] = summary
        if first_mismatch_field is None:
            first_mismatch_field = field_name
        group_name = field_name.split(".", 1)[0]
        existing = row_groups.get(group_name)
        row_groups[group_name] = row_mask if existing is None else (existing | row_mask)

    for metadata_name, values in metadata.items():
        if bool(values["mismatch"]) and first_mismatch_field is None:
            first_mismatch_field = f"metadata.{metadata_name}"

    metadata_mismatch_count = sum(
        1 for values in metadata.values() if bool(values["mismatch"])
    )
    host_row_count = metadata_mismatch_count + sum(
        int(np.count_nonzero(mask)) for mask in row_groups.values()
    )
    return {
        "schema_version": 1,
        "comparison_call_index": int(comparison_call_index),
        "expected_reference_generation": int(expected_generation),
        "actual_reference_generation": int(
            solver._velocity_dirichlet_ledger_reference_generation
        ),
        "reference_valid": bool(
            solver._velocity_dirichlet_ledger_reference_valid
        ),
        "authority": str(authority),
        "kernel_reported_mismatch_rows": int(kernel_mismatch_rows),
        "host_reconstructed_mismatch_rows": int(host_row_count),
        "host_matches_kernel_count": int(host_row_count)
        == int(kernel_mismatch_rows),
        "first_mismatch_field": first_mismatch_field,
        "metadata": metadata,
        "field_differences": field_summaries,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-output", required=True)
    parser.add_argument("--runner-script", required=True)
    args, runner_args = parser.parse_known_args()
    output_path = Path(args.diagnostic_output).resolve()
    runner_script = Path(args.runner_script).resolve()
    original_compare = (
        CartesianFluidSolver.velocity_dirichlet_boundary_ledger_mismatch_rows
    )

    def compare_then_capture(
        self: CartesianFluidSolver,
        *,
        expected_generation: int,
    ) -> int:
        global _CAPTURED, _COMPARISON_CALL_INDEX
        _COMPARISON_CALL_INDEX += 1
        mismatch_rows = int(
            original_compare(
                self,
                expected_generation=int(expected_generation),
            )
        )
        if mismatch_rows > 0 and not _CAPTURED:
            _CAPTURED = True
            try:
                payload = _capture_mismatch(
                    self,
                    expected_generation=int(expected_generation),
                    kernel_mismatch_rows=mismatch_rows,
                    comparison_call_index=_COMPARISON_CALL_INDEX,
                )
            except BaseException as exc:  # preserve the solver's original failure
                payload = {
                    "schema_version": 1,
                    "comparison_call_index": int(_COMPARISON_CALL_INDEX),
                    "expected_reference_generation": int(expected_generation),
                    "kernel_reported_mismatch_rows": int(mismatch_rows),
                    "diagnostic_capture_error": repr(exc),
                    "diagnostic_capture_traceback": traceback.format_exc(),
                }
            _write_json_atomic(output_path, payload)
        return mismatch_rows

    CartesianFluidSolver.velocity_dirichlet_boundary_ledger_mismatch_rows = (  # type: ignore[method-assign]
        compare_then_capture
    )
    sys.argv = [str(runner_script), *runner_args]
    try:
        runpy.run_path(str(runner_script), run_name="__main__")
    finally:
        CartesianFluidSolver.velocity_dirichlet_boundary_ledger_mismatch_rows = (  # type: ignore[method-assign]
            original_compare
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
