from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
REFERENCE_ROOT = ROOT / "fluent_reference"
SOURCE_EXPORTS_ROOT = REFERENCE_ROOT / "source_exports"
DIAG_ROOT = REFERENCE_ROOT / "validation_diagnostics"
CURRENT_CONTRACT = REFERENCE_ROOT / "fluent_reference_contract_2026-06-27.json"
MATRIX_JSON = DIAG_ROOT / "fluent_reference_collection_matrix.json"
MATRIX_CSV = DIAG_ROOT / "fluent_reference_collection_matrix.csv"
SUMMARY_MD = DIAG_ROOT / "fluent_reference_collection_summary.md"
CANDIDATE_CONTRACT = DIAG_ROOT / "fluent_reference_collection_candidate_contract.json"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
ACTIVE_CONTRACT_MANIFEST = REFERENCE_ROOT / "active_fluent_reference_contract.json"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_fluent_reference_collection_validation.py"
)
EXPECTED_CANDIDATE_STATUS = "fluent_reference_collection_pending"
EXPECTED_CONTRACT_STATUS = "fluent_reference_incomplete"
EXPECTED_BLOCKERS = {
    "fluent_displacement_reference_missing",
    "fluent_force_reference_missing",
    "fluent_flow_reference_missing",
    "fluent_pressure_reference_missing",
    "fluent_reference_provenance_incomplete",
}
EXPECTED_PROMOTION_BLOCKERS = EXPECTED_BLOCKERS | {
    "fluent_reference_comparison_metadata_incomplete",
    "fluent_reference_tolerances_incomplete",
}
EXPECTED_MISSING_METRICS = {
    "tip_displacement_m",
    "max_displacement_m",
    "force_z_N",
    "flow_rate_m3s",
    "pressure_range_pa",
}
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
EXPECTED_METADATA_FIELDS = {
    "Source document",
    "Fluent run id",
    "Export author",
    "Export date",
    "Fluent version",
    "mesh/domain source",
    "geometry units",
    "material model",
    "boundary conditions",
    "time step",
    "number of steps",
    "coupling settings if applicable",
    "export procedure",
    "who/when/how generated",
    "force_z_positive",
    "flow_rate_positive",
    "pressure_reference",
    "displacement_definition",
}


class AnsysVerticalFlapFluentReferenceCollectionArtifactTests(unittest.TestCase):
    def test_source_export_schemas_are_committed(self):
        for name, expected_header in EXPECTED_HEADERS.items():
            path = SOURCE_EXPORTS_ROOT / name
            self.assertTrue(path.exists(), path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], expected_header)
            self.assertEqual(len(rows), 1)

        metadata = SOURCE_EXPORTS_ROOT / "fluent_metadata_2026-06-28.md"
        self.assertTrue(metadata.exists(), metadata)
        text = metadata.read_text(encoding="utf-8")
        for field in EXPECTED_METADATA_FIELDS:
            self.assertIn(f"- {field}: MISSING", text)

    def test_collection_matrix_records_pending_reference_state(self):
        for path in (
            MATRIX_JSON,
            MATRIX_CSV,
            SUMMARY_MD,
            CANDIDATE_CONTRACT,
            ACTIVE_CONTRACT_MANIFEST,
            CHECKSUMS,
        ):
            self.assertTrue(path.exists(), path)

        payload = _read_json(MATRIX_JSON)
        blockers = {item["blocker"] for item in payload["candidate_blockers"]}
        promotion_blockers = {item["blocker"] for item in payload["promotion_blockers"]}
        rows_by_artifact = {row["artifact"]: row for row in payload["rows"]}

        self.assertEqual(payload["purpose"], "fluent_reference_collection_matrix")
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertEqual(payload["candidate_status"], EXPECTED_CANDIDATE_STATUS)
        self.assertEqual(payload["candidate_contract_status"], EXPECTED_CONTRACT_STATUS)
        self.assertEqual(blockers, EXPECTED_BLOCKERS)
        self.assertEqual(int(payload["expected_step_count"]), 50)
        self.assertEqual(float(payload["expected_time_step_s"]), 0.0005)
        self.assertEqual(float(payload["expected_total_time_s"]), 0.025)
        self.assertEqual(payload["current_reference_contract"], CURRENT_CONTRACT.as_posix())
        self.assertEqual(
            payload["current_reference_contract_sha256"],
            _sha256_file(CURRENT_CONTRACT),
        )
        self.assertEqual(payload["candidate_contract"], CANDIDATE_CONTRACT.as_posix())
        self.assertEqual(
            payload["candidate_contract_sha256"],
            _sha256_file(CANDIDATE_CONTRACT),
        )
        self.assertEqual(
            payload["active_fluent_reference_contract_manifest"],
            ACTIVE_CONTRACT_MANIFEST.as_posix(),
        )
        self.assertEqual(
            payload["active_fluent_reference_contract_manifest_sha256"],
            _sha256_file(ACTIVE_CONTRACT_MANIFEST),
        )
        self.assertEqual(payload["active_contract"], CURRENT_CONTRACT.as_posix())
        self.assertEqual(payload["active_contract_sha256"], _sha256_file(CURRENT_CONTRACT))
        self.assertEqual(payload["promotion_status"], "blocked_reference_incomplete")
        self.assertEqual(payload["recommended_action"], "keep_current_incomplete_contract")
        self.assertEqual(promotion_blockers, EXPECTED_PROMOTION_BLOCKERS)
        self.assertEqual(set(payload["missing_reference_metrics"]), EXPECTED_MISSING_METRICS)
        self.assertEqual(set(rows_by_artifact), set(EXPECTED_HEADERS) | {"fluent_metadata_2026-06-28.md"})

        for check in payload["source_checks"]:
            self.assertTrue(check["exists"])
            self.assertEqual(check["file_status"], "schema_only")
            self.assertEqual(check["header_status"], "passed")
            self.assertEqual(check["final_step_status"], "missing_final_step")
            self.assertEqual(check["metric_status"], "missing")
            self.assertEqual(int(check["row_count"]), 0)
            self.assertEqual(check["observed_columns"], EXPECTED_HEADERS[check["artifact"]])
            self.assertEqual(check["required_columns"], EXPECTED_HEADERS[check["artifact"]])

        metadata = payload["metadata_check"]
        self.assertEqual(metadata["file_status"], "present_incomplete")
        self.assertEqual(metadata["provenance_status"], "incomplete")
        self.assertEqual(metadata["blocker"], "fluent_reference_provenance_incomplete")
        self.assertEqual(set(metadata["missing_fields"]), EXPECTED_METADATA_FIELDS)
        self.assertEqual(metadata["semantic_mismatches"], [])
        self.assertEqual(metadata["source_provenance"]["status"], "missing")

    def test_candidate_contract_remains_incomplete_and_missing_reference_backed(self):
        current = _read_json(CURRENT_CONTRACT)
        candidate = _read_json(CANDIDATE_CONTRACT)
        manifest = _read_json(ACTIVE_CONTRACT_MANIFEST)

        self.assertEqual(current["contract_status"], EXPECTED_CONTRACT_STATUS)
        self.assertEqual(candidate["contract_status"], EXPECTED_CONTRACT_STATUS)
        self.assertEqual(candidate["provenance_status"], "incomplete")
        self.assertEqual(candidate["source_provenance"]["status"], "missing")
        self.assertEqual(candidate["displacement_definition"]["status"], "missing")
        self.assertEqual(candidate["sign_conventions"]["status"], "missing")
        self.assertEqual(
            candidate["active_contract_recommendation"]["recommended_action"],
            "keep_current_incomplete_contract",
        )
        self.assertIn(
            "comparison_metadata_incomplete",
            candidate["active_contract_recommendation"]["reason"],
        )
        self.assertEqual(candidate["collection_validator"]["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertEqual(
            candidate["collection_validator"]["current_contract_sha256"],
            _sha256_file(CURRENT_CONTRACT),
        )
        self.assertFalse(candidate["collection_validator"]["tolerances_complete"])
        self.assertFalse(
            candidate["collection_validator"]["comparison_metadata_complete"]
        )
        self.assertEqual(manifest["purpose"], "active_fluent_reference_contract_manifest")
        self.assertEqual(manifest["active_contract"], CURRENT_CONTRACT.as_posix())
        self.assertEqual(manifest["active_contract_sha256"], _sha256_file(CURRENT_CONTRACT))
        self.assertEqual(manifest["active_contract_status"], EXPECTED_CONTRACT_STATUS)
        self.assertEqual(manifest["candidate_contract"], CANDIDATE_CONTRACT.as_posix())
        self.assertEqual(
            manifest["candidate_contract_sha256"],
            _sha256_file(CANDIDATE_CONTRACT),
        )
        self.assertEqual(manifest["candidate_contract_status"], EXPECTED_CONTRACT_STATUS)
        self.assertEqual(manifest["promotion_status"], "blocked_reference_incomplete")
        self.assertEqual(manifest["recommended_action"], "keep_current_incomplete_contract")
        self.assertEqual(
            {item["blocker"] for item in manifest["promotion_blockers"]},
            EXPECTED_PROMOTION_BLOCKERS,
        )
        self.assertFalse(manifest["no_fluent_parity_claim_retired"])
        self.assertEqual(set(candidate["missing_reference_metrics"]), EXPECTED_MISSING_METRICS)
        for metric in EXPECTED_MISSING_METRICS:
            self.assertEqual(candidate["reference_metrics"][metric]["status"], "missing")
            self.assertIsNone(candidate["reference_metrics"][metric]["value"])
            self.assertEqual(candidate["reference_metrics"][metric]["source"], "not_collected")

    def test_summary_csv_and_checksums_are_consistent(self):
        payload = _read_json(MATRIX_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")

        self.assertIn("Fluent reference collection", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertNotIn("fluent_parity_validated", summary)
        self.assertNotIn("Fluent parity validated", summary)
        self.assertIn(EXPECTED_CANDIDATE_STATUS, summary)

        with MATRIX_CSV.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 5)
        self.assertEqual(
            {row["blocker"] for row in csv_rows},
            EXPECTED_BLOCKERS,
        )
        self.assertEqual(
            {row["artifact"] for row in csv_rows},
            set(EXPECTED_HEADERS) | {"fluent_metadata_2026-06-28.md"},
        )

        checksum_rows = _read_checksums(CHECKSUMS)
        for artifact in (
            MATRIX_JSON.name,
            MATRIX_CSV.name,
            SUMMARY_MD.name,
            CANDIDATE_CONTRACT.name,
        ):
            self.assertIn(artifact, checksum_rows)
            self.assertEqual(checksum_rows[artifact], _sha256_file(DIAG_ROOT / artifact))
        self.assertEqual(
            payload["candidate_contract_sha256"],
            checksum_rows[CANDIDATE_CONTRACT.name],
        )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        rows[name] = digest
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
