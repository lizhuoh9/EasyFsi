"""Diagnostic wrapper for HIBM marker-closure convergence trajectories.

The normal runner does not import this file.  A diagnostic replay invokes this
sidecar, which patches only the marker-closure method for the lifetime of that
process.  It does not change geometry, velocities, tolerances, or sweep counts.
"""

from __future__ import annotations

import argparse
import functools
from hashlib import sha256
import json
import math
from pathlib import Path
import runpy
import sys
from typing import Any, Callable, Sequence


_CLOSURE_ERROR = (
    "HIBM-owned hard target marker compatibility closure did not converge "
    "before canonical commit"
)
_CLOSURE_ERROR_PREFIX = _CLOSURE_ERROR + ": adjustable_residual_mps="
_MEASUREMENT_AFTER_STAGES = {
    "hibm_marker_closure_initial_measure_after",
    "hibm_marker_closure_final_measure_after",
    "hibm_marker_closure_recovery_measure_after",
}
_RUNNER = Path(__file__).with_name("run_our_solver_vertical_flap.py")


def script_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _field_value(boundary: Any, name: str, cast: Callable[[Any], Any]) -> Any:
    field = getattr(boundary, name)
    return cast(field[None])


def _optional_field_value(
    boundary: Any,
    name: str,
    cast: Callable[[Any], Any],
) -> Any:
    field = getattr(boundary, name, None)
    return None if field is None else cast(field[None])


def _measurement_snapshot(
    boundary: Any,
    *,
    stage: str,
    measurement_index: int,
    iterations_per_batch: int,
) -> dict[str, Any]:
    completed_sweeps = 0
    if stage != "hibm_marker_closure_initial_measure_after":
        completed_sweeps = measurement_index * iterations_per_batch
    return {
        "stage": stage,
        "measurement_index": measurement_index,
        "completed_sweeps": completed_sweeps,
        "constraint_count": _field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_constraint_count",
            int,
        ),
        "adjustable_constraint_count": _field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_adjustable_count",
            int,
        ),
        "immutable_constraint_count": _field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_immutable_count",
            int,
        ),
        "invalid_count": _field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_invalid_count",
            int,
        ),
        "failure_code": _field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_failure_code",
            int,
        ),
        "max_residual_mps": _field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_max_residual_mps",
            float,
        ),
        "max_adjustable_residual_mps": _field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_max_adjustable_residual_mps",
            float,
        ),
        "max_immutable_residual_mps": _field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_max_immutable_residual_mps",
            float,
        ),
        "projection_only_constraint_count": _optional_field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_projection_only_constraint_count",
            int,
        ),
        "projection_only_max_residual_mps": _optional_field_value(
            boundary,
            "report_velocity_dirichlet_marker_target_closure_projection_only_max_residual_mps",
            float,
        ),
    }


def _json_safe(
    value: Any,
    *,
    path: str,
    nonfinite_fields: list[str],
) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite_fields.append(path)
        return None
    if isinstance(value, dict):
        return {
            key: _json_safe(
                item,
                path=f"{path}.{key}" if path else key,
                nonfinite_fields=nonfinite_fields,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _json_safe(
                item,
                path=f"{path}[{index}]",
                nonfinite_fields=nonfinite_fields,
            )
            for index, item in enumerate(value)
        ]
    return value


def _write_failure_trace(
    trace_path: Path,
    *,
    error: RuntimeError,
    kwargs: dict[str, Any],
    trajectory: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": 1,
        "capture_state": "post_sweeps_before_caller_rollback",
        "error_type": type(error).__name__,
        "error": str(error),
        "script_sha256": script_sha256(),
        "iterations_per_batch": int(kwargs["iterations_per_batch"]),
        "absolute_tolerance_mps": float(kwargs["absolute_tolerance_mps"]),
        "closure_tolerance_mps": float(kwargs["closure_tolerance_mps"]),
        "density_kgm3": float(kwargs["density_kgm3"]),
        "primary_region_id": int(kwargs["primary_region_id"]),
        "secondary_region_id": int(kwargs["secondary_region_id"]),
        "measurement_trajectory": trajectory,
    }
    nonfinite_fields: list[str] = []
    payload = _json_safe(
        payload,
        path="",
        nonfinite_fields=nonfinite_fields,
    )
    payload["nonfinite_fields"] = nonfinite_fields
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def make_traced_close(
    original_close: Callable[..., Any],
    trace_path: Path,
) -> Callable[..., Any]:
    """Return a closure wrapper that records only the failing call."""

    @functools.wraps(original_close)
    def traced_close(boundary: Any, **kwargs: Any) -> Any:
        original_observer = kwargs.get("stage_observer")
        iterations_per_batch = int(kwargs["iterations_per_batch"])
        trajectory: list[dict[str, Any]] = []

        def traced_observer(stage: str) -> None:
            if original_observer is not None:
                original_observer(stage)
            if stage in _MEASUREMENT_AFTER_STAGES:
                trajectory.append(
                    _measurement_snapshot(
                        boundary,
                        stage=stage,
                        measurement_index=len(trajectory),
                        iterations_per_batch=iterations_per_batch,
                    )
                )

        traced_kwargs = dict(kwargs)
        traced_kwargs["stage_observer"] = traced_observer
        try:
            return original_close(boundary, **traced_kwargs)
        except RuntimeError as error:
            if str(error).startswith(_CLOSURE_ERROR_PREFIX):
                try:
                    _write_failure_trace(
                        Path(trace_path),
                        error=error,
                        kwargs=traced_kwargs,
                        trajectory=trajectory,
                    )
                except Exception as diagnostic_error:
                    print(
                        "closure diagnostic write failed without replacing the "
                        f"solver error: {type(diagnostic_error).__name__}: "
                        f"{diagnostic_error}",
                        file=sys.stderr,
                        flush=True,
                    )
            raise

    return traced_close


def _parse_args(argv: Sequence[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(
        description="Replay the vertical-flap runner with closure trajectory capture."
    )
    parser.add_argument("--closure-trace-output", type=Path, required=True)
    args, runner_args = parser.parse_known_args(list(argv))
    if runner_args[:1] == ["--"]:
        runner_args = runner_args[1:]
    if not runner_args:
        parser.error("runner arguments are required after --")
    return args.closure_trace_output, runner_args


def main(argv: Sequence[str] | None = None) -> None:
    trace_path, runner_args = _parse_args(
        sys.argv[1:] if argv is None else argv
    )
    if trace_path.exists():
        raise FileExistsError(
            f"refusing to overwrite closure trace: {trace_path}"
        )
    from simulation_core.coupling.hibm_mpm.core import (
        HibmMpmIbBoundaryConditions,
    )

    method_name = "_close_owned_hard_targets_to_marker_constraints"
    original_close = getattr(HibmMpmIbBoundaryConditions, method_name)
    setattr(
        HibmMpmIbBoundaryConditions,
        method_name,
        make_traced_close(original_close, trace_path),
    )
    original_argv = sys.argv
    sys.argv = [str(_RUNNER), *runner_args]
    try:
        runpy.run_path(str(_RUNNER), run_name="__main__")
    finally:
        sys.argv = original_argv
        setattr(HibmMpmIbBoundaryConditions, method_name, original_close)


if __name__ == "__main__":
    main()
