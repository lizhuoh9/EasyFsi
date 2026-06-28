from __future__ import annotations

import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path


SCHEMA_PATH = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "fluent_reference_contract_schema.py"
)
REAL_CONTRACT = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "fluent_reference"
    / "fluent_reference_contract_2026-06-27.json"
)


def _load_schema():
    spec = importlib.util.spec_from_file_location("fluent_reference_contract_schema", SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SCHEMA = _load_schema()


class AnsysVerticalFlapFluentReferenceContractSchemaTests(unittest.TestCase):
    def test_current_real_contract_remains_fail_closed(self):
        import json

        contract = json.loads(REAL_CONTRACT.read_text(encoding="utf-8"))
        result = SCHEMA.validate_fluent_reference_contract(contract)
        blockers = {item["blocker"] for item in result["blockers"]}

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertEqual(result["validated_metric_count"], 0)
        self.assertEqual(result["required_metric_count"], 5)
        self.assertEqual(
            set(result["missing_required_metrics"]),
            {
                "tip_displacement_m",
                "max_displacement_m",
                "force_z_N",
                "flow_rate_m3s",
                "pressure_range_pa",
            },
        )
        self.assertIn("fluent_reference_provenance_incomplete", blockers)
        self.assertIn("fluent_reference_tolerances_incomplete", blockers)

    def test_missing_provenance_fails_closed(self):
        contract = _complete_contract()
        contract["source_provenance"]["run_id"] = ""

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_provenance_incomplete",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_missing_sign_convention_fails_closed(self):
        contract = _complete_contract()
        contract["sign_conventions"]["flow_rate_positive"] = "MISSING"

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_sign_conventions_incomplete",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_missing_tolerance_source_fails_closed(self):
        contract = _complete_contract()
        del contract["tolerances"]["force_z_relative"]["source"]

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_tolerances_incomplete",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_missing_schema_version_fails_closed(self):
        contract = _complete_contract()
        del contract["schema_version"]

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_schema_version_missing",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_metric_name_or_unit_mismatch_fails_closed(self):
        contract = _complete_contract()
        contract["reference_metrics"]["force_z_kN"] = contract["reference_metrics"].pop(
            "force_z_N"
        )

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn("force_z_N", result["missing_required_metrics"])

    def test_wrong_metric_unit_fails_closed(self):
        contract = _complete_contract()
        contract["reference_metrics"]["force_z_N"]["unit"] = "kN"

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn("force_z_N", result["missing_required_metrics"])

    def test_missing_metric_source_or_extraction_method_fails_closed(self):
        contract = _complete_contract()
        del contract["reference_metrics"]["flow_rate_m3s"]["extraction_method"]

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn("flow_rate_m3s", result["missing_required_metrics"])

    def test_metric_final_time_mismatch_fails_closed(self):
        contract = _complete_contract()
        contract["reference_metrics"]["pressure_range_pa"]["time_s"] = 0.02

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn("pressure_range_pa", result["missing_required_metrics"])

    def test_missing_sampling_definition_fails_closed(self):
        contract = _complete_contract()
        del contract["sampling_definitions"]["force_z"]

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_sampling_definitions_incomplete",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_missing_comparison_policy_fails_closed(self):
        contract = _complete_contract()
        contract["comparison_policy"]["reference_complete_required"] = False

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_comparison_policy_incomplete",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_unsupported_tolerance_comparator_fails_closed(self):
        contract = _complete_contract()
        contract["tolerances"]["pressure_sanity_absolute"]["comparator"] = (
            "relative_error"
        )

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_tolerances_incomplete",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_step_count_mismatch_fails_closed(self):
        contract = _complete_contract()
        contract["simulation"]["step_count"] = 49

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_step_count_mismatch",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_time_step_mismatch_fails_closed(self):
        contract = _complete_contract()
        contract["simulation"]["time_step_s"] = 0.001

        result = SCHEMA.validate_fluent_reference_contract(contract)

        self.assertEqual(result["contract_status"], "fluent_reference_incomplete")
        self.assertIn(
            "fluent_reference_time_step_mismatch",
            {item["blocker"] for item in result["blockers"]},
        )

    def test_complete_synthetic_contract_validates(self):
        result = SCHEMA.validate_fluent_reference_contract(_complete_contract())

        self.assertEqual(result["contract_status"], "fluent_reference_complete")
        self.assertEqual(result["validated_metric_count"], 5)
        self.assertEqual(result["required_metric_count"], 5)
        self.assertEqual(result["missing_required_metrics"], [])
        self.assertEqual(result["blockers"], [])


def _complete_contract():
    return deepcopy(
        {
            "case": "ansys_vertical_flap_fsi",
            "contract_id": "synthetic-complete-contract",
            "contract_status": "fluent_reference_complete",
            "schema_version": "ansys_vertical_flap_fluent_reference_contract_v1",
            "source_provenance": {
                "document": "synthetic schema test",
                "run_id": "synthetic-complete-contract",
                "author": "schema-test",
                "date": "2026-06-29",
                "status": "complete",
            },
            "simulation": {
                "step_count": 50,
                "time_step_s": 0.0005,
                "total_time_s": 0.025,
            },
            "step_count": 50,
            "time_step_s": 0.0005,
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
            "displacement_definition": {
                "metric": "tip_displacement_norm_m",
                "source_step50_metric": "tip_mean_displacement_m",
                "point": "synthetic flap tip",
                "status": "complete",
            },
            "sign_conventions": {
                "force_z_positive": "positive z is downstream",
                "flow_rate_positive": "positive zmin outlet flow leaves the domain",
                "pressure_reference": "gauge pressure relative to pressure outlet",
                "status": "complete",
            },
            "sampling_definitions": {
                "tip_displacement": {
                    "definition": "synthetic tip total displacement",
                    "unit": "m",
                    "status": "complete",
                },
                "max_displacement": {
                    "definition": "synthetic max solid total displacement",
                    "unit": "m",
                    "status": "complete",
                },
                "force_z": {
                    "definition": "synthetic total z force",
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
            "reference_metrics": {
                "tip_displacement_m": {
                    "status": "available",
                    "value": 0.0105,
                    "unit": "m",
                    "source": "synthetic source",
                    "extraction_method": "synthetic_report_csv",
                    "time_s": 0.025,
                },
                "max_displacement_m": {
                    "status": "available",
                    "value": 0.0195,
                    "unit": "m",
                    "source": "synthetic source",
                    "extraction_method": "synthetic_report_csv",
                    "time_s": 0.025,
                },
                "force_z_N": {
                    "status": "available",
                    "value": 2.1,
                    "unit": "N",
                    "source": "synthetic source",
                    "extraction_method": "synthetic_report_csv",
                    "time_s": 0.025,
                },
                "flow_rate_m3s": {
                    "status": "available",
                    "value": 0.00105,
                    "unit": "m3/s",
                    "source": "synthetic source",
                    "extraction_method": "synthetic_report_csv",
                    "time_s": 0.025,
                },
                "pressure_range_pa": {
                    "status": "available",
                    "value": 58.0,
                    "unit": "Pa",
                    "source": "synthetic source",
                    "extraction_method": "synthetic_report_csv",
                    "time_s": 0.025,
                },
            },
            "tolerances": {
                "tip_displacement_relative": {
                    "status": "available",
                    "value": 0.1,
                    "comparator": "relative_error",
                    "source": "synthetic tolerance",
                    "rationale": "synthetic tolerance",
                },
                "max_displacement_relative": {
                    "status": "available",
                    "value": 0.1,
                    "comparator": "relative_error",
                    "source": "synthetic tolerance",
                    "rationale": "synthetic tolerance",
                },
                "force_z_relative": {
                    "status": "available",
                    "value": 0.2,
                    "comparator": "relative_error",
                    "source": "synthetic tolerance",
                    "rationale": "synthetic tolerance",
                },
                "flow_rate_relative": {
                    "status": "available",
                    "value": 0.1,
                    "comparator": "relative_error",
                    "source": "synthetic tolerance",
                    "rationale": "synthetic tolerance",
                },
                "pressure_sanity_absolute": {
                    "status": "available",
                    "value": 5.0,
                    "comparator": "absolute_error",
                    "source": "synthetic tolerance",
                    "rationale": "synthetic tolerance",
                },
            },
        }
    )


if __name__ == "__main__":
    unittest.main()
