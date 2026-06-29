from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi" / "scripts"
REFERENCE_ROOT = (
    Path("validation_runs") / "ansys_vertical_flap_fsi" / "fluent_reference"
)
SOURCE_EXPORTS_ROOT = REFERENCE_ROOT / "source_exports"
CURRENT_CONTRACT = REFERENCE_ROOT / "fluent_reference_contract_2026-06-27.json"
DIAG_ROOT = REFERENCE_ROOT / "validation_diagnostics"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
REQUIRE_REAL_FLUENT_EXPORTS_ENV = "EASYFSI_REQUIRE_REAL_FLUENT_EXPORTS"

EXPECTED_FINAL_STEP = 50
EXPECTED_FINAL_TIME_S = 0.025
EXPECTED_METRIC_COUNT = 5
EXPECTED_HEADERS = {
    "fluent_tip_displacement_history.csv": [
        "step",
        "time_s",
        "tip_displacement_x_m",
        "tip_displacement_y_m",
        "tip_displacement_z_m",
        "tip_displacement_norm_m",
        "max_displacement_m",
        "source",
    ],
    "fluent_force_history.csv": [
        "step",
        "time_s",
        "force_x_N",
        "force_y_N",
        "force_z_N",
        "primary_force_z_N",
        "secondary_force_z_N",
        "source",
    ],
    "fluent_flow_balance_history.csv": [
        "step",
        "time_s",
        "inlet_flow_rate_m3s",
        "outlet_flow_rate_m3s",
        "pressure_outlet_flux_m3s",
        "velocity_outlet_flux_m3s",
        "source",
    ],
    "fluent_pressure_summary_history.csv": [
        "step",
        "time_s",
        "pressure_min_pa",
        "pressure_max_pa",
        "pressure_range_pa",
        "source",
    ],
}
REQUIRED_METRIC_COLUMNS = {
    "fluent_tip_displacement_history.csv": [
        "tip_displacement_norm_m",
        "max_displacement_m",
    ],
    "fluent_force_history.csv": ["force_z_N"],
    "fluent_flow_balance_history.csv": ["outlet_flow_rate_m3s"],
    "fluent_pressure_summary_history.csv": ["pressure_range_pa"],
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COLLECTION = _load_module(
    "run_fluent_reference_collection_validation_for_real_artifacts",
    SCRIPT_ROOT / "run_fluent_reference_collection_validation.py",
)
DISALLOWED_TERMS = tuple(COLLECTION.DISALLOWED_METADATA_PROVENANCE_TERMS)


class AnsysVerticalFlapRealFluentSourceExportArtifactTests(unittest.TestCase):
    def test_committed_source_exports_are_schema_only_or_real_ready(self):
        payload = _run_collection()
        source_rows = _source_rows_by_artifact()

        if _all_schema_only(source_rows):
            _assert_schema_only_fail_closed(self, payload)
            return

        self.assertTrue(
            _all_exports_have_rows(source_rows),
            "Committed Fluent source exports are partial: either keep all four "
            "CSV files schema-only or commit a complete real Fluent export set.",
        )
        _assert_real_export_csv_rows(self, source_rows)
        _assert_real_collection_ready(self, payload)

    def test_strict_mode_requires_real_fluent_exports(self):
        if os.environ.get(REQUIRE_REAL_FLUENT_EXPORTS_ENV) != "1":
            self.skipTest(
                f"Set {REQUIRE_REAL_FLUENT_EXPORTS_ENV}=1 to require committed "
                "real Fluent report exports."
            )

        payload = _run_collection()
        source_rows = _source_rows_by_artifact()

        self.assertFalse(
            _all_schema_only(source_rows),
            "Strict mode requires real Fluent source rows; current source "
            "exports are schema-only.",
        )
        self.assertTrue(
            _all_exports_have_rows(source_rows),
            "Strict mode requires all four Fluent source export CSV files to "
            "contain final-step rows.",
        )
        _assert_real_export_csv_rows(self, source_rows)
        _assert_real_collection_ready(self, payload)

    def test_collection_diagnostic_checksums_match(self):
        checksum_rows = _read_checksums(CHECKSUMS)

        for artifact in (
            "ARTIFACT_MANIFEST.json",
            "fluent_reference_collection_candidate_contract.json",
            "fluent_reference_collection_matrix.csv",
            "fluent_reference_collection_matrix.json",
            "fluent_reference_collection_summary.md",
        ):
            self.assertIn(artifact, checksum_rows)
            self.assertEqual(checksum_rows[artifact], _sha256_file(DIAG_ROOT / artifact))


def _run_collection() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        return COLLECTION.run_with_paths(
            source_exports_root=SOURCE_EXPORTS_ROOT,
            current_contract_json=CURRENT_CONTRACT,
            output_dir=root / "diagnostics",
            active_manifest_json=root / "active_fluent_reference_contract.json",
        )


def _source_rows_by_artifact() -> dict[str, list[dict[str, str]]]:
    rows_by_artifact: dict[str, list[dict[str, str]]] = {}
    for artifact, expected_header in EXPECTED_HEADERS.items():
        path = SOURCE_EXPORTS_ROOT / artifact
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self_header = list(reader.fieldnames or [])
            if self_header != expected_header:
                raise AssertionError(
                    f"{artifact} header mismatch: {self_header} != {expected_header}"
                )
            rows_by_artifact[artifact] = list(reader)
    return rows_by_artifact


def _all_schema_only(source_rows: dict[str, list[dict[str, str]]]) -> bool:
    return all(len(rows) == 0 for rows in source_rows.values())


def _all_exports_have_rows(source_rows: dict[str, list[dict[str, str]]]) -> bool:
    return all(len(rows) > 0 for rows in source_rows.values())


def _assert_schema_only_fail_closed(
    test_case: unittest.TestCase,
    payload: dict[str, object],
) -> None:
    gate = payload["real_fluent_import_gate"]
    metadata = payload["metadata_check"]
    schema = payload["schema_validation"]

    test_case.assertEqual(payload["candidate_status"], "fluent_reference_collection_pending")
    test_case.assertEqual(
        payload["candidate_contract_status"],
        "fluent_reference_incomplete",
    )
    test_case.assertEqual(schema["validated_metric_count"], 0)
    test_case.assertEqual(schema["required_metric_count"], EXPECTED_METRIC_COUNT)
    test_case.assertEqual(gate["status"], "blocked_real_fluent_import_incomplete")
    test_case.assertFalse(gate["can_import_real_fluent_reference"])
    test_case.assertFalse(gate["can_run_solver_evaluation"])
    test_case.assertFalse(gate["fluent_parity_claimed"])
    test_case.assertEqual(gate["candidate_contract_status"], "fluent_reference_incomplete")
    test_case.assertEqual(gate["promotion_status"], "blocked_reference_incomplete")
    test_case.assertFalse(gate["metadata"]["ready"])
    test_case.assertEqual(metadata["provenance_status"], "incomplete")
    test_case.assertEqual(metadata["disallowed_provenance"], [])
    for check in payload["source_checks"]:
        test_case.assertEqual(check["row_count"], 0)
        test_case.assertEqual(check["file_status"], "schema_only")
        test_case.assertEqual(check["final_step_status"], "missing_final_step")
        test_case.assertEqual(check["metric_status"], "missing")
    for export in gate["source_exports"]:
        test_case.assertFalse(export["ready"])
        test_case.assertEqual(export["file_status"], "schema_only")
        test_case.assertEqual(export["final_step_status"], "missing_final_step")
        test_case.assertEqual(export["metric_status"], "missing")


def _assert_real_export_csv_rows(
    test_case: unittest.TestCase,
    source_rows: dict[str, list[dict[str, str]]],
) -> None:
    for artifact, rows in source_rows.items():
        final_rows = [
            row for row in rows if _int_value(row.get("step")) == EXPECTED_FINAL_STEP
        ]
        test_case.assertTrue(final_rows, f"{artifact} has no step 50 row")
        final_row = final_rows[-1]
        test_case.assertAlmostEqual(
            _float_value(final_row.get("time_s")),
            EXPECTED_FINAL_TIME_S,
            places=12,
            msg=f"{artifact} final time must be 0.025 s",
        )
        source = str(final_row.get("source", "")).strip()
        test_case.assertNotEqual(source, "", f"{artifact} source is empty")
        normalized_source = " ".join(source.lower().replace("\\", "/").split())
        for term in DISALLOWED_TERMS:
            test_case.assertNotIn(term, normalized_source, artifact)
        for metric in REQUIRED_METRIC_COLUMNS[artifact]:
            value = _float_value(final_row.get(metric))
            test_case.assertTrue(
                math.isfinite(value),
                f"{artifact} {metric} must be finite",
            )


def _assert_real_collection_ready(
    test_case: unittest.TestCase,
    payload: dict[str, object],
) -> None:
    gate = payload["real_fluent_import_gate"]
    metadata = payload["metadata_check"]
    schema = payload["schema_validation"]

    test_case.assertEqual(payload["candidate_status"], "fluent_reference_collection_complete")
    test_case.assertEqual(payload["candidate_contract_status"], "fluent_reference_complete")
    test_case.assertEqual(schema["validated_metric_count"], EXPECTED_METRIC_COUNT)
    test_case.assertEqual(schema["required_metric_count"], EXPECTED_METRIC_COUNT)
    test_case.assertEqual(schema["missing_required_metrics"], [])
    test_case.assertEqual(payload["promotion_status"], "ready_for_versioned_contract_promotion")
    test_case.assertEqual(gate["status"], "ready_for_real_fluent_import")
    test_case.assertTrue(gate["can_import_real_fluent_reference"])
    test_case.assertTrue(gate["can_run_solver_evaluation"])
    test_case.assertFalse(gate["fluent_parity_claimed"])
    test_case.assertEqual(gate["blockers"], [])
    test_case.assertTrue(gate["metadata"]["ready"])
    test_case.assertEqual(metadata["provenance_status"], "complete")
    test_case.assertEqual(metadata["disallowed_provenance"], [])
    for export in gate["source_exports"]:
        test_case.assertTrue(export["ready"])
        test_case.assertEqual(export["schema_blockers"], [])


def _read_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        rows[name] = digest
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _int_value(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _float_value(value: object) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    unittest.main()
