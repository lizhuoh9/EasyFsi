from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_traction_selected_formulation_fluent_parity.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("fluent_parity_runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


class AnsysVerticalFlapFluentParityComparisonLogicTests(unittest.TestCase):
    def test_complete_contract_with_all_gates_passing_validates_parity(self):
        metrics = RUNNER._parity_metrics(
            source_matrix=_source_matrix(),
            source_row=_source_row(),
            source_history_rows=[_final_step()],
            reference_contract=_complete_contract(),
        )

        self.assertEqual(metrics["displacement"]["gate_status"], "passed")
        self.assertEqual(metrics["force"]["gate_status"], "passed")
        self.assertEqual(metrics["flow_outlet"]["gate_status"], "passed")
        self.assertEqual(metrics["pressure"]["gate_status"], "passed")
        self.assertEqual(metrics["metadata"]["gate_status"], "passed")
        self.assertEqual(metrics["force"]["force_sign_matches"], True)
        self.assertEqual(metrics["flow_outlet"]["flow_sign_matches"], True)
        self.assertAlmostEqual(
            metrics["pressure"]["absolute_error"],
            2.0,
            places=12,
        )
        self.assertEqual(
            RUNNER._candidate_status(_complete_contract(), metrics),
            "fluent_parity_validated",
        )
        self.assertEqual(
            RUNNER._candidate_blockers("fluent_parity_validated", metrics),
            [],
        )

    def test_complete_contract_with_force_mismatch_keeps_no_parity_claim(self):
        contract = _complete_contract()
        contract["reference_metrics"]["force_z_N"]["value"] = 10.0
        metrics = RUNNER._parity_metrics(
            source_matrix=_source_matrix(),
            source_row=_source_row(),
            source_history_rows=[_final_step()],
            reference_contract=contract,
        )

        status = RUNNER._candidate_status(contract, metrics)
        blockers = {
            item["blocker"] for item in RUNNER._candidate_blockers(status, metrics)
        }

        self.assertEqual(metrics["force"]["gate_status"], "failed")
        self.assertEqual(status, "fluent_parity_failed")
        self.assertIn("fluent_force_mismatch", blockers)
        self.assertIn("no_fluent_parity_claim", blockers)

    def test_near_zero_reference_uses_deterministic_denominator_floor(self):
        comparison = RUNNER._relative_comparison(
            source_value=1.0e-9,
            reference_value=0.0,
            tolerance=0.1,
        )

        self.assertEqual(comparison["gate_status"], "failed")
        self.assertAlmostEqual(comparison["relative_error"], 1000.0, places=9)


def _source_matrix():
    return {
        "candidate_status": "selected_formulation_coupled_step50_passed",
        "reference_formulation_candidate": (
            "anchored_dual_face_pressure_pair_with_per_face_one_sided"
        ),
    }


def _source_row():
    return {
        "max_displacement_m": 0.02,
        "marker_force_z_by_step": [2.0],
        "force_sign_flip_count": 0,
        "fluid_finite": True,
        "max_velocity_mps": 10.0,
        "pressure_finite": True,
        "max_pressure_pa": 160.0,
        "max_pressure_growth_ratio": 1.0,
        "completed_step_count": 50,
        "selected_anchor_markers_source": "synthetic-test-only",
        "selected_anchor_markers_source_sha256": "synthetic-test-only",
        "pressure_pair_anchor_map_sha256": "synthetic-test-only",
    }


def _final_step():
    return {
        "tip_mean_displacement_m": 0.01,
        "max_displacement_m": 0.02,
        "marker_force_z_N": 2.0,
        "primary_face_force_z_N": 1.0,
        "secondary_face_force_z_N": 1.0,
        "zmin_velocity_outlet_flux_m3s": 0.001,
        "pressure_min_pa": 100.0,
        "pressure_max_pa": 160.0,
    }


def _complete_contract():
    return {
        "contract_status": "fluent_reference_complete",
        "step_count": 50,
        "time_step_s": 0.0005,
        "simulation": {
            "step_count": 50,
            "time_step_s": 0.0005,
            "total_time_s": 0.025,
        },
        "source_provenance": {
            "document": "synthetic-test-only",
            "run_id": "synthetic-test-only",
            "author": "synthetic-test-only",
            "date": "2026-06-28",
            "status": "complete",
        },
        "reference_metrics": {
            "tip_displacement_m": {"status": "available", "value": 0.0105},
            "max_displacement_m": {"status": "available", "value": 0.0195},
            "force_z_N": {"status": "available", "value": 2.1},
            "flow_rate_m3s": {"status": "available", "value": 0.00105},
            "pressure_range_pa": {"status": "available", "value": 58.0},
        },
        "tolerances": {
            "tip_displacement_relative": {"status": "available", "value": 0.1},
            "max_displacement_relative": {"status": "available", "value": 0.1},
            "force_z_relative": {"status": "available", "value": 0.2},
            "flow_rate_relative": {"status": "available", "value": 0.1},
            "pressure_sanity_absolute": {"status": "available", "value": 5.0},
        },
        "displacement_definition": {
            "metric": "tip_displacement_norm_m",
            "source_step50_metric": "tip_mean_displacement_m",
            "point": "synthetic test fixture",
            "status": "complete",
        },
        "sign_conventions": {
            "force_z_positive": "synthetic positive z",
            "flow_rate_positive": "synthetic positive outlet",
            "pressure_reference": "synthetic gauge",
            "status": "complete",
        },
        "geometry": {},
        "material": {},
    }


if __name__ == "__main__":
    unittest.main()
