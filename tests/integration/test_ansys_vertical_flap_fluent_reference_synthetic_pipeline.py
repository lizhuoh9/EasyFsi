from __future__ import annotations

import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi" / "scripts"
REAL_CONTRACT = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "fluent_reference"
    / "fluent_reference_contract_2026-06-27.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module(
    "build_synthetic_fluent_reference_fixture",
    SCRIPT_ROOT / "build_synthetic_fluent_reference_fixture.py",
)
COLLECTION = _load_module(
    "run_fluent_reference_collection_validation",
    SCRIPT_ROOT / "run_fluent_reference_collection_validation.py",
)
PARITY = _load_module(
    "run_traction_selected_formulation_fluent_parity",
    SCRIPT_ROOT / "run_traction_selected_formulation_fluent_parity.py",
)


class AnsysVerticalFlapFluentReferenceSyntheticPipelineTests(unittest.TestCase):
    def test_easyfsi_hibm_source_exports_do_not_complete_collection_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_exports = root / "source_exports"
            output_dir = root / "diagnostics"
            active_manifest = root / "active_fluent_reference_contract.json"
            current_contract = root / "fluent_reference_contract.json"
            _write_complete_current_contract(current_contract)
            BUILDER.build_synthetic_fluent_reference_fixture(source_exports)
            _rewrite_csv_sources(
                source_exports,
                "EasyFsi/HIBM-MPM validation_runs placeholder not Fluent truth",
            )

            payload = COLLECTION.run_with_paths(
                source_exports_root=source_exports,
                current_contract_json=current_contract,
                output_dir=output_dir,
                active_manifest_json=active_manifest,
            )
            candidate = _read_json(
                output_dir / "fluent_reference_collection_candidate_contract.json"
            )

        self.assertEqual(
            payload["candidate_status"],
            "fluent_reference_collection_pending",
        )
        self.assertEqual(candidate["contract_status"], "fluent_reference_incomplete")
        self.assertLess(candidate["schema_validation"]["validated_metric_count"], 5)
        self.assertEqual(payload["promotion_status"], "blocked_reference_incomplete")
        for check in payload["source_checks"]:
            self.assertEqual(check["metric_status"], "missing")
            self.assertIn("disallowed_source_provenance", check["schema_blockers"])
            self.assertEqual(check["reference_values"], {})

    def test_synthetic_source_exports_complete_collection_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_exports = root / "source_exports"
            output_dir = root / "diagnostics"
            active_manifest = root / "active_fluent_reference_contract.json"
            current_contract = root / "fluent_reference_contract.json"
            _write_complete_current_contract(current_contract)
            BUILDER.build_synthetic_fluent_reference_fixture(source_exports)

            payload = COLLECTION.run_with_paths(
                source_exports_root=source_exports,
                current_contract_json=current_contract,
                output_dir=output_dir,
                active_manifest_json=active_manifest,
            )

            candidate = _read_json(
                output_dir / "fluent_reference_collection_candidate_contract.json"
            )
            manifest = _read_json(active_manifest)
            candidate_sha = _sha256_file(
                output_dir / "fluent_reference_collection_candidate_contract.json"
            )
            current_sha = _sha256_file(current_contract)

        self.assertEqual(
            payload["candidate_status"],
            "fluent_reference_collection_complete",
        )
        self.assertEqual(candidate["contract_status"], "fluent_reference_complete")
        self.assertEqual(candidate["schema_validation"]["validated_metric_count"], 5)
        self.assertEqual(payload["promotion_status"], "ready_for_versioned_contract_promotion")
        self.assertEqual(
            manifest["manifest_schema_version"],
            "active_fluent_reference_contract_manifest_v1",
        )
        self.assertEqual(
            manifest["candidate_contract_sha256"],
            candidate_sha,
        )
        self.assertEqual(manifest["active_contract_sha256"], current_sha)

    def test_synthetic_parity_pass_and_fail_are_temp_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_exports = root / "source_exports"
            output_dir = root / "diagnostics"
            active_manifest = root / "active_fluent_reference_contract.json"
            current_contract = root / "fluent_reference_contract.json"
            _write_complete_current_contract(current_contract)
            BUILDER.build_synthetic_fluent_reference_fixture(source_exports)
            COLLECTION.run_with_paths(
                source_exports_root=source_exports,
                current_contract_json=current_contract,
                output_dir=output_dir,
                active_manifest_json=active_manifest,
            )
            contract = _read_json(
                output_dir / "fluent_reference_collection_candidate_contract.json"
            )

            pass_metrics = PARITY._parity_metrics(
                source_matrix=_source_matrix(),
                source_row=_source_row(),
                source_history_rows=[_final_step()],
                reference_contract=contract,
                contract_schema_validation=contract["schema_validation"],
            )
            pass_status = PARITY._candidate_status(contract, pass_metrics)

            fail_contract = json.loads(json.dumps(contract))
            fail_contract["reference_metrics"]["force_z_N"]["value"] = 10.0
            fail_metrics = PARITY._parity_metrics(
                source_matrix=_source_matrix(),
                source_row=_source_row(),
                source_history_rows=[_final_step()],
                reference_contract=fail_contract,
                contract_schema_validation=fail_contract["schema_validation"],
            )
            fail_status = PARITY._candidate_status(fail_contract, fail_metrics)

        self.assertEqual(pass_status, "fluent_parity_validated")
        self.assertEqual(fail_status, "fluent_parity_failed")
        self.assertEqual(fail_metrics["force"]["gate_status"], "failed")
        self.assertFalse(fail_status == "fluent_parity_validated")


def _write_complete_current_contract(path: Path) -> None:
    contract = _read_json(REAL_CONTRACT)
    contract["tolerances"] = {
        "tip_displacement_relative": _tolerance(0.1, "relative_error"),
        "max_displacement_relative": _tolerance(0.1, "relative_error"),
        "force_z_relative": _tolerance(0.1, "relative_error"),
        "flow_rate_relative": _tolerance(0.1, "relative_error"),
        "pressure_sanity_absolute": _tolerance(5.0, "absolute_error"),
    }
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_csv_sources(source_exports: Path, source: str) -> None:
    for path in source_exports.glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                updated = dict(row)
                updated["source"] = source
                writer.writerow(updated)


def _tolerance(value: float, comparator: str) -> dict[str, object]:
    return {
        "status": "available",
        "value": value,
        "comparator": comparator,
        "source": "synthetic-test-only-not-fluent-truth",
        "rationale": "synthetic dry-run tolerance",
    }


def _source_matrix() -> dict[str, str]:
    return {
        "candidate_status": "selected_formulation_coupled_step50_passed",
        "reference_formulation_candidate": (
            "anchored_dual_face_pressure_pair_with_per_face_one_sided"
        ),
    }


def _source_row() -> dict[str, object]:
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
        "selected_anchor_markers_source": "temp-synthetic-source-row",
        "selected_anchor_markers_source_sha256": "temp-synthetic-source-row",
        "pressure_pair_anchor_map_sha256": "temp-synthetic-source-row",
    }


def _final_step() -> dict[str, float]:
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


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
