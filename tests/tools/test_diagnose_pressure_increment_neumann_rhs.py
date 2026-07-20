from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.diagnose_pressure_increment_neumann_rhs import (
    _homogenize_increment_rhs,
    _write_json_atomic,
)


class _FakeSolver:
    def __init__(self) -> None:
        self.rhs = 7.0
        self.clear_calls = 0

    def pressure_interface_matrix_terms_report(self) -> dict[str, object]:
        return {
            "active_cells": 2,
            "row_count": 3,
            "rhs_integral": self.rhs / 10.0,
            "max_abs_rhs": abs(self.rhs),
        }

    def _clear_pressure_interface_matrix_rhs_kernel(self) -> None:
        self.clear_calls += 1
        self.rhs = 0.0


class PressureIncrementNeumannDiagnosticTests(unittest.TestCase):
    def test_homogenize_records_nonzero_rhs_then_clears_it_exactly_once(self) -> None:
        solver = _FakeSolver()

        before, after = _homogenize_increment_rhs(solver)  # type: ignore[arg-type]

        self.assertEqual(solver.clear_calls, 1)
        self.assertEqual(before["max_abs_rhs"], 7.0)
        self.assertEqual(after["max_abs_rhs"], 0.0)
        self.assertEqual(before["row_count"], after["row_count"])
        self.assertEqual(before["active_cells"], after["active_cells"])

    def test_atomic_writer_replaces_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic.json"

            _write_json_atomic(output, {"schema_version": 1, "value": 3})

            self.assertIn('"value": 3', output.read_text(encoding="utf-8"))
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
