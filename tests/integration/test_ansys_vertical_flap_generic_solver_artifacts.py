from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path

from simulation_core.pressure_sample_pairs import (
    PressureSamplePair,
    pressure_sample_pair_map_sha256,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "generic_solver_selected_formulation_diagnostics"
MATRIX_JSON = DIAG_ROOT / "generic_solver_selected_formulation_matrix.json"
HISTORY_JSON = DIAG_ROOT / "generic_solver_selected_formulation_history.json"
SUMMARY_MD = DIAG_ROOT / "generic_solver_selected_formulation_summary.md"
PRESSURE_PAIR_MAP_JSON = DIAG_ROOT / "pressure_sample_pair_map.json"
TIP_CSV = DIAG_ROOT / "easyfsi_tip_displacement_history.csv"
FORCE_CSV = DIAG_ROOT / "easyfsi_force_history.csv"
FLOW_CSV = DIAG_ROOT / "easyfsi_flow_balance_history.csv"
PRESSURE_CSV = DIAG_ROOT / "easyfsi_pressure_summary_history.csv"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_ansys_vertical_flap_generic_solver.py"
)
EXPECTED_EXPORT_HEADERS = {
    TIP_CSV.name: [
        "step",
        "time_s",
        "tip_displacement_x_m",
        "tip_displacement_y_m",
        "tip_displacement_z_m",
        "tip_displacement_norm_m",
        "max_displacement_m",
        "source",
    ],
    FORCE_CSV.name: [
        "step",
        "time_s",
        "force_x_N",
        "force_y_N",
        "force_z_N",
        "primary_force_z_N",
        "secondary_force_z_N",
        "source",
    ],
    FLOW_CSV.name: [
        "step",
        "time_s",
        "inlet_flow_rate_m3s",
        "outlet_flow_rate_m3s",
        "pressure_outlet_flux_m3s",
        "velocity_outlet_flux_m3s",
        "source",
    ],
    PRESSURE_CSV.name: [
        "step",
        "time_s",
        "pressure_min_pa",
        "pressure_max_pa",
        "pressure_range_pa",
        "source",
    ],
}


class AnsysVerticalFlapGenericSolverArtifactTests(unittest.TestCase):
    def test_generic_solver_matrix_records_adapter_boundary(self) -> None:
        for path in (
            MATRIX_JSON,
            HISTORY_JSON,
            SUMMARY_MD,
            PRESSURE_PAIR_MAP_JSON,
            TIP_CSV,
            FORCE_CSV,
            FLOW_CSV,
            PRESSURE_CSV,
            CHECKSUMS,
        ):
            self.assertTrue(path.exists(), path)

        payload = _read_json(MATRIX_JSON)
        row = payload["rows"][0]

        self.assertEqual(payload["purpose"], "generic_solver_selected_formulation_matrix")
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertTrue(payload["generic_api_invoked"])
        self.assertEqual(payload["generic_solver_entrypoint"], "solve_fsi")
        self.assertEqual(payload["adapter"], "AnsysVerticalFlapProblem")
        self.assertEqual(payload["validation_scope"], "easyfsi_generic_solver_only")
        self.assertFalse(payload["fluent_parity_claimed"])
        self.assertEqual(payload["fluent_parity_status"], "blocked_reference_incomplete")
        self.assertEqual(payload["scenario_count"], 1)
        self.assertEqual(row["scenario"], "generic_solver_selected_formulation_step50")
        self.assertEqual(row["requested_step_count"], 50)
        self.assertEqual(row["completed_step_count"], 50)
        self.assertEqual(row["run_status"], "completed")

    def test_pressure_pair_runtime_generation_is_official_path(self) -> None:
        payload = _read_json(MATRIX_JSON)
        row = payload["rows"][0]
        blockers = {item["blocker"] for item in payload["candidate_blockers"]}
        pressure_policy = payload["pressure_pair_policy"]

        self.assertEqual(
            pressure_policy["mode"],
            "runtime_anchored_cell_pair",
        )
        self.assertEqual(
            pressure_policy["pair_source_status"],
            "runtime_generated",
        )
        self.assertFalse(pressure_policy["transition_backed"])
        self.assertTrue(payload["pressure_pair_runtime_generation_complete"])
        self.assertEqual(
            payload["pressure_pair_runtime_generation_status"],
            "runtime_generated",
        )
        self.assertFalse(payload["transition_artifact_dependency"])
        self.assertEqual(
            payload["candidate_status"],
            "generic_solver_selected_formulation_step50_passed",
        )
        self.assertNotIn("runtime_pressure_pair_generation_pending", blockers)
        self.assertIn("no_fluent_parity_claim", blockers)
        self.assertEqual(row["pressure_pair_anchor_source"], "runtime_generated")
        self.assertTrue(row["pressure_pair_anchor_map_sha256"])
        self.assertTrue(row["pressure_pair_anchor_current_marker_geometry_sha256"])
        self.assertEqual(int(row["invalid_marker_count_max"]), 0)
        self.assertEqual(int(row["sample_pair_fallback_count_max"]), 0)
        self.assertGreaterEqual(int(row["one_sided_marker_count_min"]), 24)
        self.assertLessEqual(float(row["force_action_reaction_residual_max_n"]), 1.0e-8)

    def test_exports_are_fluent_comparable_but_easyfsi_sourced(self) -> None:
        for path in (TIP_CSV, FORCE_CSV, FLOW_CSV, PRESSURE_CSV):
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(handle.seekable(), True)
            self.assertEqual(list(rows[0].keys()), EXPECTED_EXPORT_HEADERS[path.name])
            self.assertEqual(len(rows), 50)
            self.assertEqual({row["source"] for row in rows}, {"easyfsi_generic_solver"})

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("EasyFsi generic solver", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertIn("runtime marker geometry", summary)
        self.assertNotIn("transition_seeded_from_anchor_artifact", summary)
        self.assertNotIn("Fluent parity validated", summary)

    def test_history_and_checksums_are_consistent(self) -> None:
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)

        self.assertEqual(history["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertEqual(history["generic_solver_entrypoint"], "solve_fsi")
        self.assertEqual(history["history_source"], "easyfsi_generic_solver_runtime")
        self.assertEqual(len(history["history"]), 50)
        self.assertEqual(
            history["pressure_pair_policy"]["pair_source_status"],
            "runtime_generated",
        )
        self.assertFalse(history["pressure_pair_policy"]["transition_backed"])
        self.assertEqual(
            set(payload["export_artifacts"]),
            {
                "tip_displacement",
                "force",
                "flow_balance",
                "pressure_summary",
                "pressure_sample_pair_map",
            },
        )

        checksum_rows = _read_checksums(CHECKSUMS)
        for path in (
            MATRIX_JSON,
            HISTORY_JSON,
            SUMMARY_MD,
            PRESSURE_PAIR_MAP_JSON,
            TIP_CSV,
            FORCE_CSV,
            FLOW_CSV,
            PRESSURE_CSV,
        ):
            rel_name = path.name
            self.assertIn(rel_name, checksum_rows)
            self.assertEqual(checksum_rows[rel_name], _sha256_file(path))

    def test_pressure_pair_sidecar_is_auditable(self) -> None:
        payload = _read_json(MATRIX_JSON)
        row = payload["rows"][0]
        pair_map = _read_json(PRESSURE_PAIR_MAP_JSON)
        pairs = pair_map["pairs"]

        self.assertEqual(pair_map["provider_mode"], "runtime_anchored_cell_pair")
        self.assertEqual(pair_map["fallback_count"], 0)
        self.assertEqual(pair_map["selected_count"], 24)
        self.assertEqual(len(pairs), 24)
        self.assertEqual(
            [pair["marker_index"] for pair in pairs],
            list(range(24)),
        )
        region_counts = {
            region_id: sum(1 for pair in pairs if pair["region_id"] == region_id)
            for region_id in {pair["region_id"] for pair in pairs}
        }
        self.assertEqual(region_counts, {"101": 12, "202": 12})
        for pair in pairs:
            self.assertEqual(pair["sample_status"], "runtime_generated")
            self.assertEqual(pair["fallback_status"], "no_fallback")
            self.assertEqual(pair["diagnostic_reason"], "runtime_anchored_cell_pair")
            inside_cell = tuple(pair["inside_cell"])
            outside_cell = tuple(pair["outside_cell"])
            self.assertNotEqual(inside_cell, outside_cell)
            for cell in (inside_cell, outside_cell):
                self.assertTrue(0 <= cell[0] < 4)
                self.assertTrue(0 <= cell[1] < 32)
                self.assertTrue(0 <= cell[2] < 64)

        recomputed_sha = pressure_sample_pair_map_sha256(
            tuple(
                PressureSamplePair(
                    marker_index=int(pair["marker_index"]),
                    region_id=str(pair["region_id"]),
                    inside_cell=tuple(pair["inside_cell"]),
                    outside_cell=tuple(pair["outside_cell"]),
                    sample_status=str(pair["sample_status"]),
                    fallback_status=str(pair["fallback_status"]),
                    diagnostic_reason=str(pair["diagnostic_reason"]),
                )
                for pair in pairs
            )
        )
        self.assertEqual(pair_map["pair_map_sha256"], recomputed_sha)
        self.assertEqual(row["pressure_pair_anchor_map_sha256"], recomputed_sha)
        self.assertEqual(
            row["pressure_pair_anchor_current_marker_geometry_sha256"],
            pair_map["marker_geometry_sha256"],
        )
        self.assertTrue(pair_map["marker_geometry_sha256"])

    def test_generated_artifacts_do_not_contain_transition_state_strings(self) -> None:
        forbidden = (
            "transition_seeded_from_anchor_artifact",
            "runtime_pressure_pair_generation_pending",
        )
        for path in DIAG_ROOT.glob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, path)


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
