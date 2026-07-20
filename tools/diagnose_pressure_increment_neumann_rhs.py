"""A/B diagnostic for repeated affine Neumann forcing in pressure increments.

The production runner assembles the HIBM pressure-interface operator before
every pressure solve.  A solve with ``accumulate_pressure_into_previous=True``
computes a pressure *increment*.  Its fixed Neumann jump must therefore be
homogeneous; replaying the full-pressure interface RHS applies the same affine
forcing again.

This wrapper changes no production source.  It clears only the assembled
pressure-interface RHS immediately before an accumulating ``project()`` call,
records the before/after operator report, and delegates every other operation
to the unmodified solver.  It is intended for a bounded A/B reproduction from
an already validated preflow snapshot.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path
from typing import Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_core.fluids.solver import CartesianFluidSolver


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _interface_rhs_summary(solver: CartesianFluidSolver) -> dict[str, object]:
    report = dict(solver.pressure_interface_matrix_terms_report())
    rhs_field = getattr(solver, "pressure_interface_matrix_rhs", None)
    to_numpy = getattr(rhs_field, "to_numpy", None)
    if callable(to_numpy):
        rhs = np.asarray(to_numpy(), dtype=np.float64)
        max_abs_rhs = float(np.max(np.abs(rhs), initial=0.0))
    else:
        # Keep the helper usable with narrow diagnostic fakes. Production
        # solvers always take the device-field branch above, so a cancelling
        # integral cannot be mistaken for a zero pointwise RHS.
        max_abs_rhs = float(report.get("max_abs_rhs", 0.0))
    return {
        "active_cells": int(report.get("active_cells", -1)),
        "row_count": int(report.get("row_count", -1)),
        "rhs_integral": float(report.get("rhs_integral", 0.0)),
        # The production report intentionally exposes the volume-weighted
        # integral but not a pointwise RHS maximum.  Read the device field in
        # this tools-only diagnostic so a cancelling owner/neighbor pair
        # cannot masquerade as an identically zero affine term.
        "max_abs_rhs": max_abs_rhs,
    }


def _homogenize_increment_rhs(
    solver: CartesianFluidSolver,
) -> tuple[dict[str, object], dict[str, object]]:
    before = _interface_rhs_summary(solver)
    solver._clear_pressure_interface_matrix_rhs_kernel()
    after = _interface_rhs_summary(solver)
    return before, after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-output", required=True)
    parser.add_argument("--runner-script", required=True)
    args, runner_args = parser.parse_known_args()
    output_path = Path(args.diagnostic_output).resolve()
    runner_script = Path(args.runner_script).resolve()
    records: list[dict[str, object]] = []
    original_project = CartesianFluidSolver.project

    def project_with_homogeneous_increment_rhs(
        self: CartesianFluidSolver,
        *project_args,
        **project_kwargs,
    ):
        accumulate = bool(
            project_kwargs.get("accumulate_pressure_into_previous", False)
        )
        record: dict[str, object] | None = None
        if accumulate:
            before, after = _homogenize_increment_rhs(self)
            context = project_kwargs.get("pressure_solve_context", {})
            record = {
                "call_index": len(records) + 1,
                "pressure_solve_context": (
                    dict(context) if isinstance(context, Mapping) else {}
                ),
                "before": before,
                "after": after,
                "returned": False,
            }
            records.append(record)
            _write_json_atomic(
                output_path,
                {"schema_version": 1, "records": records},
            )
        try:
            result = original_project(self, *project_args, **project_kwargs)
        except BaseException as exc:
            if record is not None:
                record["exception"] = repr(exc)
                record["traceback"] = traceback.format_exc()
                _write_json_atomic(
                    output_path,
                    {"schema_version": 1, "records": records},
                )
            raise
        if record is not None:
            projection_report = dict(result) if isinstance(result, Mapping) else {}
            record.update(
                {
                    "returned": True,
                    "projection_l2": float(projection_report.get("l2", 0.0)),
                    "projection_max_abs": float(
                        projection_report.get("max_abs", 0.0)
                    ),
                    "pre_projection_l2": float(
                        projection_report.get("pre_projection_l2", 0.0)
                    ),
                    "cg_exact_relative_residual": float(
                        projection_report.get("cg_exact_relative_residual_max", 0.0)
                    ),
                }
            )
            _write_json_atomic(
                output_path,
                {"schema_version": 1, "records": records},
            )
        return result

    CartesianFluidSolver.project = project_with_homogeneous_increment_rhs  # type: ignore[method-assign]
    sys.argv = [str(runner_script), *runner_args]
    try:
        runpy.run_path(str(runner_script), run_name="__main__")
    finally:
        CartesianFluidSolver.project = original_project  # type: ignore[method-assign]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
