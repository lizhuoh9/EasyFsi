from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "fluent_source_export_schema.py"
)

HEADER = ["step", "time_s", "force_z_N", "source"]
REFERENCE_COLUMNS = {"force_z_N": "force_z_N"}


def _load_schema():
    spec = importlib.util.spec_from_file_location("fluent_source_export_schema", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SCHEMA = _load_schema()


class AnsysVerticalFlapFluentSourceExportSchemaTests(unittest.TestCase):
    def test_header_only_is_schema_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(path, "step,time_s,force_z_N,source\n")

            result = _validate(path)

        self.assertEqual(result["file_status"], "schema_only")
        self.assertEqual(result["row_count"], 0)
        self.assertEqual(result["metric_status"], "missing")

    def test_wrong_header_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(path, "step,time_s,force_z_kN,source\n")

            result = _validate(path)

        self.assertEqual(result["file_status"], "present_header_mismatch")
        self.assertEqual(result["header_status"], "failed")

    def test_rows_without_final_step_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(path, "step,time_s,force_z_N,source\n49,0.0245,2.0,synthetic\n")

            result = _validate(path)

        self.assertEqual(result["file_status"], "present_missing_final_step")
        self.assertEqual(result["final_step_status"], "missing_final_step")

    def test_final_time_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(path, "step,time_s,force_z_N,source\n50,0.02,2.0,synthetic\n")

            result = _validate(path)

        self.assertEqual(result["file_status"], "present_final_time_mismatch")
        self.assertEqual(result["final_step_status"], "final_time_mismatch")

    def test_empty_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(path, "step,time_s,force_z_N,source\n50,0.025,2.0,\n")

            result = _validate(path)

        self.assertEqual(result["file_status"], "present_missing_source")

    def test_missing_metric_value_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(path, _complete_row(metric_value="", source="Fluent run export 2026-06-28"))

            result = _validate(path)

        self.assertEqual(result["file_status"], "present_missing_metric_value")

    def test_disallowed_source_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(
                path,
                _complete_row(
                    source="EasyFsi/HIBM-MPM validation_runs placeholder not Fluent truth"
                ),
            )

            result = _validate(path)

        self.assertEqual(result["file_status"], "present_disallowed_source_provenance")
        self.assertEqual(result["metric_status"], "missing")
        self.assertEqual(result["reference_values"], {})
        self.assertIn("disallowed_source_provenance", result["blockers"])

    def test_test_sources_require_explicit_allowance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(path, _complete_row(source="synthetic-test-only-not-fluent-truth"))

            blocked = _validate(path)
            allowed = _validate(path, allow_test_sources=True)

        self.assertEqual(blocked["file_status"], "present_disallowed_source_provenance")
        self.assertEqual(allowed["file_status"], "present_complete")
        self.assertEqual(allowed["reference_values"], {"force_z_N": 2.0})

    def test_complete_row_returns_reference_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.csv"
            _write(path, _complete_row(source="Fluent run export 2026-06-28"))

            result = _validate(path)

        self.assertEqual(result["file_status"], "present_complete")
        self.assertEqual(result["metric_status"], "available")
        self.assertEqual(result["reference_values"], {"force_z_N": 2.0})


def _validate(path: Path, *, allow_test_sources: bool = False):
    return SCHEMA.validate_source_export_csv(
        path,
        HEADER,
        reference_value_columns=REFERENCE_COLUMNS,
        allow_test_sources=allow_test_sources,
    )


def _complete_row(*, metric_value: object = 2.0, source: str) -> str:
    return f"step,time_s,force_z_N,source\n50,0.025,{metric_value},{source}\n"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
