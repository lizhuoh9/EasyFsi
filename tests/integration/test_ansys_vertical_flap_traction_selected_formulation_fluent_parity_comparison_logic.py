from __future__ import annotations

import importlib.util
import json
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
        contract = _complete_contract()
        schema_validation = _schema_validation(contract)
        metrics = RUNNER._parity_metrics(
            source_matrix=_source_matrix(),
            source_row=_source_row(),
            source_history_rows=[_final_step()],
            reference_contract=contract,
            contract_schema_validation=schema_validation,
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
            RUNNER._candidate_status(contract, metrics),
            "fluent_parity_validated",
        )
        self.assertEqual(
            RUNNER._candidate_blockers("fluent_parity_validated", metrics),
            [],
        )

    def test_complete_contract_with_force_mismatch_keeps_no_parity_claim(self):
        contract = _complete_contract()
        contract["reference_metrics"]["force_z_N"]["value"] = 10.0
        schema_validation = _schema_validation(contract)
        metrics = RUNNER._parity_metrics(
            source_matrix=_source_matrix(),
            source_row=_source_row(),
            source_history_rows=[_final_step()],
            reference_contract=contract,
            contract_schema_validation=schema_validation,
        )

        status = RUNNER._candidate_status(contract, metrics)
        blockers = {
            item["blocker"] for item in RUNNER._candidate_blockers(status, metrics)
        }

        self.assertEqual(metrics["force"]["gate_status"], "failed")
        self.assertEqual(status, "fluent_parity_failed")
        self.assertIn("fluent_force_mismatch", blockers)
        self.assertIn("no_fluent_parity_claim", blockers)

    def test_complete_status_string_with_schema_incomplete_stays_blocked(self):
        contract = _complete_contract()
        contract["reference_metrics"]["force_z_N"]["status"] = "missing"
        contract["reference_metrics"]["force_z_N"]["value"] = None
        schema_validation = {
            "contract_status": "fluent_reference_incomplete",
            "blockers": [{"blocker": "force_z_N_missing", "detail": "synthetic"}],
            "validated_metric_count": 4,
            "required_metric_count": 5,
            "missing_required_metrics": ["force_z_N"],
        }

        metrics = RUNNER._parity_metrics(
            source_matrix=_source_matrix(),
            source_row=_source_row(),
            source_history_rows=[_final_step()],
            reference_contract=contract,
            contract_schema_validation=schema_validation,
        )
        status = RUNNER._candidate_status(contract, metrics)
        blockers = {
            item["blocker"] for item in RUNNER._candidate_blockers(status, metrics)
        }

        self.assertEqual(status, "fluent_parity_blocked_reference_incomplete")
        self.assertIn("fluent_reference_incomplete", blockers)
        self.assertIn("no_fluent_parity_claim", blockers)

    def test_near_zero_reference_uses_deterministic_denominator_floor(self):
        comparison = RUNNER._relative_comparison(
            source_value=1.0e-9,
            reference_value=0.0,
            tolerance=0.1,
        )

        self.assertEqual(comparison["gate_status"], "failed")
        self.assertAlmostEqual(comparison["relative_error"], 1000.0, places=9)

    def test_active_manifest_rejects_absolute_contract_path(self):
        manifest = _real_active_manifest()
        manifest["active_contract"] = "C:/tmp/fluent_reference_contract.json"

        with self.assertRaisesRegex(ValueError, "not repo-relative"):
            RUNNER._validate_active_contract_manifest(manifest)

    def test_active_manifest_rejects_path_traversal(self):
        manifest = _real_active_manifest()
        manifest["active_contract"] = "../fluent_reference_contract_2026-06-27.json"

        with self.assertRaisesRegex(ValueError, "not repo-relative"):
            RUNNER._validate_active_contract_manifest(manifest)

    def test_active_manifest_rejects_stale_contract_sha(self):
        manifest = _real_active_manifest()
        manifest["active_contract_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            RUNNER._validate_active_contract_manifest(manifest)

    def test_active_manifest_rejects_complete_status_with_incomplete_schema(self):
        manifest = _real_active_manifest()
        manifest["active_contract_status"] = "fluent_reference_complete"

        with self.assertRaisesRegex(ValueError, "status/schema mismatch"):
            RUNNER._validate_active_contract_manifest(manifest)

    def test_active_manifest_rejects_candidate_schema_mismatch(self):
        manifest = _real_active_manifest()
        manifest["candidate_contract_status"] = "fluent_reference_complete"

        with self.assertRaisesRegex(ValueError, "candidate.*status/schema mismatch"):
            RUNNER._validate_active_contract_manifest(manifest)


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
        "case": "ansys_vertical_flap_fsi",
        "contract_id": "synthetic-complete-contract",
        "contract_status": "fluent_reference_complete",
        "schema_version": "ansys_vertical_flap_fluent_reference_contract_v1",
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
            "tip_displacement_m": {
                "status": "available",
                "value": 0.0105,
                "unit": "m",
                "source": "synthetic-test-only",
                "extraction_method": "synthetic_report_csv",
                "time_s": 0.025,
            },
            "max_displacement_m": {
                "status": "available",
                "value": 0.0195,
                "unit": "m",
                "source": "synthetic-test-only",
                "extraction_method": "synthetic_report_csv",
                "time_s": 0.025,
            },
            "force_z_N": {
                "status": "available",
                "value": 2.1,
                "unit": "N",
                "source": "synthetic-test-only",
                "extraction_method": "synthetic_report_csv",
                "time_s": 0.025,
            },
            "flow_rate_m3s": {
                "status": "available",
                "value": 0.00105,
                "unit": "m3/s",
                "source": "synthetic-test-only",
                "extraction_method": "synthetic_report_csv",
                "time_s": 0.025,
            },
            "pressure_range_pa": {
                "status": "available",
                "value": 58.0,
                "unit": "Pa",
                "source": "synthetic-test-only",
                "extraction_method": "synthetic_report_csv",
                "time_s": 0.025,
            },
        },
        "tolerances": {
            "tip_displacement_relative": {
                "status": "available",
                "value": 0.1,
                "comparator": "relative_error",
                "source": "synthetic-test-only",
                "rationale": "synthetic-test-only",
            },
            "max_displacement_relative": {
                "status": "available",
                "value": 0.1,
                "comparator": "relative_error",
                "source": "synthetic-test-only",
                "rationale": "synthetic-test-only",
            },
            "force_z_relative": {
                "status": "available",
                "value": 0.2,
                "comparator": "relative_error",
                "source": "synthetic-test-only",
                "rationale": "synthetic-test-only",
            },
            "flow_rate_relative": {
                "status": "available",
                "value": 0.1,
                "comparator": "relative_error",
                "source": "synthetic-test-only",
                "rationale": "synthetic-test-only",
            },
            "pressure_sanity_absolute": {
                "status": "available",
                "value": 5.0,
                "comparator": "absolute_error",
                "source": "synthetic-test-only",
                "rationale": "synthetic-test-only",
            },
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
        "sampling_definitions": {
            "tip_displacement": {
                "definition": "synthetic tip total displacement",
                "unit": "m",
                "status": "complete",
            },
            "max_displacement": {
                "definition": "synthetic max solid displacement",
                "unit": "m",
                "status": "complete",
            },
            "force_z": {
                "definition": "synthetic z force",
                "unit": "N",
                "status": "complete",
            },
            "flow_rate": {
                "definition": "synthetic outlet flow rate",
                "unit": "m3/s",
                "status": "complete",
            },
            "pressure_range": {
                "definition": "synthetic pressure range",
                "unit": "Pa",
                "status": "complete",
            },
        },
        "comparison_policy": {
            "status": "complete",
            "reference_complete_required": True,
            "parity_claim_requires_all_gates": True,
        },
        "geometry": {
            "duct_length_m": 0.1,
            "duct_height_m": 0.04,
            "modeled_domain": "lower-symmetry-half",
            "flap_height_m": 0.01,
            "flap_thickness_m": 0.003,
            "flap_streamwise_min_m": 0.05,
            "flap_streamwise_max_m": 0.053,
        },
        "material": {
            "air_density_kgm3": 1.225,
            "air_viscosity_pa_s": 1.8e-5,
            "solid_density_kgm3": 1600.0,
            "youngs_modulus_pa": 1.0e6,
            "poisson_ratio": 0.47,
        },
    }


def _schema_validation(contract):
    result = RUNNER.validate_fluent_reference_contract(contract)
    assert result["contract_status"] == "fluent_reference_complete", result
    return result


def _real_active_manifest():
    contract = json.loads(
        RUNNER.FLUENT_REFERENCE_CONTRACT_JSON.read_text(encoding="utf-8")
    )
    validation = RUNNER.validate_fluent_reference_contract(contract)
    return {
        "manifest_schema_version": RUNNER.ACTIVE_MANIFEST_SCHEMA_VERSION,
        "active_contract": RUNNER.FLUENT_REFERENCE_CONTRACT_JSON.as_posix(),
        "active_contract_sha256": RUNNER._sha256_file(
            RUNNER.FLUENT_REFERENCE_CONTRACT_JSON
        ),
        "active_contract_status": validation["contract_status"],
        "active_contract_schema_validation": validation,
        "candidate_contract_status": validation["contract_status"],
        "candidate_contract_schema_validation": validation,
    }


if __name__ == "__main__":
    unittest.main()
