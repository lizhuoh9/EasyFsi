from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
SOURCE_STEP50_MATRIX = (
    ROOT
    / "traction_selected_formulation_coupled_step50_diagnostics"
    / "traction_selected_formulation_coupled_step50_matrix.json"
)
SOURCE_STEP50_HISTORY = (
    ROOT
    / "traction_selected_formulation_coupled_step50_diagnostics"
    / "traction_selected_formulation_coupled_step50_history.json"
)
FLUENT_REFERENCE_CONTRACT = (
    ROOT / "fluent_reference" / "fluent_reference_contract_2026-06-27.json"
)
ACTIVE_CONTRACT_MANIFEST = (
    ROOT / "fluent_reference" / "active_fluent_reference_contract.json"
)
DIAG_ROOT = ROOT / "traction_selected_formulation_fluent_parity_diagnostics"
SCENARIO_DIAGNOSTICS_ROOT = DIAG_ROOT / "scenario_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_selected_formulation_fluent_parity_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_selected_formulation_fluent_parity_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_selected_formulation_fluent_parity_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_selected_formulation_fluent_parity_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_selected_formulation_fluent_parity.py"
)
EXPECTED_CANDIDATE = "anchored_dual_face_pressure_pair_with_per_face_one_sided"
EXPECTED_SCENARIO = "selected_formulation_fluent_parity"
EXPECTED_CANDIDATE_STATUS = "fluent_parity_blocked_reference_incomplete"
EXPECTED_ACTIVE_BLOCKERS = {
    "fluent_reference_incomplete",
    "no_fluent_parity_claim",
}


class AnsysVerticalFlapSelectedFormulationFluentParityArtifactTests(
    unittest.TestCase
):
    def test_fluent_parity_matrix_records_reference_incomplete_boundary(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        blockers = {item["blocker"] for item in payload["candidate_blockers"]}

        self.assertEqual(payload["purpose"], "selected_formulation_fluent_parity_matrix")
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertEqual(payload["scenario_count"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scenario"], EXPECTED_SCENARIO)
        self.assertEqual(rows[0]["run_status"], "blocked")
        self.assertEqual(rows[0]["parity_status"], EXPECTED_CANDIDATE_STATUS)
        self.assertEqual(set(rows[0]["active_blockers"]), EXPECTED_ACTIVE_BLOCKERS)
        self.assertEqual(payload["candidate_status"], EXPECTED_CANDIDATE_STATUS)
        self.assertFalse(payload["fluent_parity_claimed"])
        self.assertFalse(rows[0]["fluent_parity_claimed"])
        self.assertEqual(blockers, EXPECTED_ACTIVE_BLOCKERS)
        self.assertEqual(payload["historical_blockers_retired"], [])
        self.assertEqual(
            payload["reference_contract_status"],
            "fluent_reference_incomplete",
        )
        self.assertEqual(
            payload["source_step50_candidate_status"],
            "selected_formulation_coupled_step50_passed",
        )
        self.assertEqual(payload["reference_formulation_candidate"], EXPECTED_CANDIDATE)

    def test_fluent_parity_sources_and_shas_are_locked(self):
        payload = _read_json(MATRIX_JSON)
        row = payload["rows"][0]

        self.assertEqual(payload["source_step50_matrix"], SOURCE_STEP50_MATRIX.as_posix())
        self.assertEqual(
            payload["source_step50_matrix_sha256"],
            _sha256_file(SOURCE_STEP50_MATRIX),
        )
        self.assertEqual(payload["source_step50_history"], SOURCE_STEP50_HISTORY.as_posix())
        self.assertEqual(
            payload["source_step50_history_sha256"],
            _sha256_file(SOURCE_STEP50_HISTORY),
        )
        self.assertEqual(
            payload["fluent_reference_contract"],
            FLUENT_REFERENCE_CONTRACT.as_posix(),
        )
        self.assertEqual(
            payload["fluent_reference_contract_sha256"],
            _sha256_file(FLUENT_REFERENCE_CONTRACT),
        )
        self.assertEqual(
            payload["active_fluent_reference_contract_manifest"],
            ACTIVE_CONTRACT_MANIFEST.as_posix(),
        )
        self.assertEqual(
            payload["active_fluent_reference_contract_manifest_sha256"],
            _sha256_file(ACTIVE_CONTRACT_MANIFEST),
        )
        self.assertEqual(
            payload["active_contract_status"],
            "fluent_reference_incomplete",
        )
        self.assertEqual(
            payload["active_contract_promotion_status"],
            "blocked_reference_incomplete",
        )
        self.assertEqual(
            row["fluent_reference_contract_sha256"],
            _sha256_file(FLUENT_REFERENCE_CONTRACT),
        )
        self.assertEqual(
            row["active_fluent_reference_contract_manifest_sha256"],
            _sha256_file(ACTIVE_CONTRACT_MANIFEST),
        )
        self.assertTrue(Path(row["scenario_diagnostics_json"]).exists())
        self.assertEqual(
            Path(row["scenario_diagnostics_json"]).parent,
            SCENARIO_DIAGNOSTICS_ROOT,
        )

    def test_fluent_reference_contract_is_explicitly_incomplete(self):
        contract = _read_json(FLUENT_REFERENCE_CONTRACT)

        self.assertEqual(contract["case"], "ansys_vertical_flap_fsi")
        self.assertEqual(contract["provenance_status"], "incomplete")
        self.assertEqual(contract["contract_status"], "fluent_reference_incomplete")
        self.assertIn("source_provenance", contract)
        self.assertIn("simulation", contract)
        self.assertEqual(
            set(contract["source_provenance"]),
            {"document", "run_id", "author", "date", "status"},
        )
        self.assertNotIn(
            contract["source_provenance"]["status"],
            {"complete", "validated"},
        )
        self.assertEqual(int(contract["simulation"]["step_count"]), 50)
        self.assertEqual(float(contract["simulation"]["time_step_s"]), 0.0005)
        self.assertEqual(float(contract["simulation"]["total_time_s"]), 0.025)
        self.assertEqual(int(contract["step_count"]), 50)
        self.assertEqual(float(contract["time_step_s"]), 0.0005)
        self.assertEqual(
            set(contract["missing_reference_metrics"]),
            {
                "tip_displacement_m",
                "max_displacement_m",
                "force_z_N",
                "flow_rate_m3s",
                "pressure_range_pa",
            },
        )
        for metric in contract["missing_reference_metrics"]:
            self.assertEqual(contract["reference_metrics"][metric]["status"], "missing")
            self.assertIsNone(contract["reference_metrics"][metric]["value"])

    def test_fluent_parity_metrics_are_present_and_blocked_by_reference(self):
        payload = _read_json(MATRIX_JSON)
        row = payload["rows"][0]
        metrics = payload["parity_metrics"]

        self.assertEqual(row["run_status"], "blocked")
        self.assertEqual(row["parity_status"], EXPECTED_CANDIDATE_STATUS)
        self.assertFalse(row["fluent_parity_claimed"])
        self.assertEqual(
            set(row["active_blockers"]),
            EXPECTED_ACTIVE_BLOCKERS,
        )
        for group in ("displacement", "force", "flow_outlet", "pressure", "metadata"):
            self.assertIn(group, metrics)
            self.assertIn("gate_status", metrics[group])
        self.assertEqual(
            metrics["displacement"]["gate_status"],
            "blocked_reference_missing",
        )
        self.assertEqual(metrics["force"]["gate_status"], "blocked_reference_missing")
        self.assertEqual(
            metrics["flow_outlet"]["gate_status"],
            "blocked_reference_missing",
        )
        self.assertEqual(metrics["pressure"]["gate_status"], "blocked_reference_missing")
        self.assertEqual(metrics["metadata"]["gate_status"], "passed")
        self.assertEqual(
            metrics["metadata"]["source_step50_candidate_status"],
            "selected_formulation_coupled_step50_passed",
        )
        self.assertEqual(
            metrics["metadata"]["reference_formulation_candidate"],
            EXPECTED_CANDIDATE,
        )
        self.assertEqual(int(metrics["metadata"]["contract_simulation"]["step_count"]), 50)
        self.assertEqual(
            float(metrics["metadata"]["contract_simulation"]["time_step_s"]),
            0.0005,
        )
        self.assertEqual(
            float(metrics["metadata"]["contract_simulation"]["total_time_s"]),
            0.025,
        )
        self.assertEqual(
            metrics["metadata"]["contract_source_provenance"]["status"],
            "missing",
        )
        self.assertEqual(
            metrics["metadata"]["contract_schema_validation"]["contract_status"],
            "fluent_reference_incomplete",
        )
        self.assertEqual(
            metrics["metadata"]["contract_schema_validation"][
                "validated_metric_count"
            ],
            0,
        )

    def test_history_summary_csv_and_checksums_are_consistent(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")

        self.assertEqual(set(history["histories"]), {EXPECTED_SCENARIO})
        self.assertEqual(
            history["histories"][EXPECTED_SCENARIO]["candidate_status"],
            EXPECTED_CANDIDATE_STATUS,
        )
        self.assertFalse(
            history["histories"][EXPECTED_SCENARIO]["fluent_parity_claimed"]
        )
        self.assertEqual(
            history["active_fluent_reference_contract_manifest"],
            ACTIVE_CONTRACT_MANIFEST.as_posix(),
        )
        self.assertEqual(
            history["active_fluent_reference_contract_manifest_sha256"],
            _sha256_file(ACTIVE_CONTRACT_MANIFEST),
        )
        self.assertIn("selected-formulation Fluent parity", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertNotIn("fluent_parity_validated", summary)
        self.assertNotIn("Fluent parity validated", summary)
        self.assertIn(EXPECTED_CANDIDATE_STATUS, summary)

        with MATRIX_CSV.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(csv_rows[0]["scenario"], EXPECTED_SCENARIO)
        self.assertEqual(csv_rows[0]["candidate_status"], EXPECTED_CANDIDATE_STATUS)

        checksum_rows = _read_checksums(CHECKSUMS)
        for artifact in (
            MATRIX_JSON.name,
            MATRIX_CSV.name,
            HISTORY_JSON.name,
            SUMMARY_MD.name,
        ):
            self.assertIn(artifact, checksum_rows)
            self.assertEqual(checksum_rows[artifact], _sha256_file(DIAG_ROOT / artifact))
        for row in payload["rows"]:
            diagnostics_rel = Path(row["scenario_diagnostics_json"]).relative_to(
                DIAG_ROOT
            )
            self.assertIn(diagnostics_rel.as_posix(), checksum_rows)
            self.assertEqual(
                checksum_rows[diagnostics_rel.as_posix()],
                _sha256_file(DIAG_ROOT / diagnostics_rel),
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
