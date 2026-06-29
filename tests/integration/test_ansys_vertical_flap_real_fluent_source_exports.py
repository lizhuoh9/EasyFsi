from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi" / "scripts"
REFERENCE_ROOT = (
    Path("validation_runs") / "ansys_vertical_flap_fsi" / "fluent_reference"
)
SOURCE_EXPORTS_ROOT = REFERENCE_ROOT / "source_exports"
REAL_CONTRACT = REFERENCE_ROOT / "fluent_reference_contract_2026-06-27.json"
FLUENT_SOURCE = (
    "ANSYS Fluent 2025 R1 two-way FSI vertical flap run run-2026-07-01 "
    "report export"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module(
    "build_synthetic_fluent_reference_fixture_for_real_gate",
    SCRIPT_ROOT / "build_synthetic_fluent_reference_fixture.py",
)
COLLECTION = _load_module(
    "run_fluent_reference_collection_validation_for_real_gate",
    SCRIPT_ROOT / "run_fluent_reference_collection_validation.py",
)


class AnsysVerticalFlapRealFluentSourceExportGateTests(unittest.TestCase):
    def test_current_schema_only_exports_block_real_fluent_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = COLLECTION.run_with_paths(
                source_exports_root=SOURCE_EXPORTS_ROOT,
                current_contract_json=REAL_CONTRACT,
                output_dir=root / "diagnostics",
                active_manifest_json=root / "active_fluent_reference_contract.json",
            )

        gate = payload["real_fluent_import_gate"]
        self.assertEqual(gate["status"], "blocked_real_fluent_import_incomplete")
        self.assertFalse(gate["can_import_real_fluent_reference"])
        self.assertFalse(gate["can_run_solver_evaluation"])
        self.assertFalse(gate["fluent_parity_claimed"])
        self.assertEqual(gate["candidate_contract_status"], "fluent_reference_incomplete")
        self.assertEqual(gate["promotion_status"], "blocked_reference_incomplete")
        self.assertIn("source_export_not_ready", _blocker_rules(gate))
        self.assertIn("metadata_not_ready", _blocker_rules(gate))
        self.assertIn("candidate_contract_incomplete", _blocker_rules(gate))
        self.assertIn("active_manifest_promotion_blocked", _blocker_rules(gate))
        self.assertFalse(gate["metadata"]["ready"])
        self.assertEqual(len(gate["source_exports"]), 4)
        for export in gate["source_exports"]:
            self.assertFalse(export["ready"])
            self.assertEqual(export["metric_status"], "missing")

    def test_test_source_allowance_never_counts_as_real_fluent_import_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_exports = root / "source_exports"
            current_contract = root / "fluent_reference_contract.json"
            _write_complete_current_contract(current_contract)
            BUILDER.build_synthetic_fluent_reference_fixture(source_exports)

            payload = COLLECTION.run_with_paths(
                source_exports_root=source_exports,
                current_contract_json=current_contract,
                output_dir=root / "diagnostics",
                active_manifest_json=root / "active_fluent_reference_contract.json",
                allow_test_sources=True,
            )

        self.assertEqual(payload["candidate_status"], "fluent_reference_collection_complete")
        gate = payload["real_fluent_import_gate"]
        self.assertEqual(gate["status"], "blocked_real_fluent_import_incomplete")
        self.assertFalse(gate["can_import_real_fluent_reference"])
        self.assertIn("test_source_allowance_enabled", _blocker_rules(gate))

    def test_provenance_complete_fluent_style_bundle_can_satisfy_import_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_exports = root / "source_exports"
            current_contract = root / "fluent_reference_contract.json"
            _write_complete_current_contract(current_contract)
            BUILDER.build_synthetic_fluent_reference_fixture(source_exports)
            _rewrite_csv_sources(source_exports, FLUENT_SOURCE)
            _write_fluent_metadata(source_exports / "fluent_metadata_2026-06-28.md")

            payload = COLLECTION.run_with_paths(
                source_exports_root=source_exports,
                current_contract_json=current_contract,
                output_dir=root / "diagnostics",
                active_manifest_json=root / "active_fluent_reference_contract.json",
            )

        gate = payload["real_fluent_import_gate"]
        self.assertEqual(payload["candidate_status"], "fluent_reference_collection_complete")
        self.assertEqual(gate["status"], "ready_for_real_fluent_import")
        self.assertTrue(gate["can_import_real_fluent_reference"])
        self.assertTrue(gate["can_run_solver_evaluation"])
        self.assertFalse(gate["fluent_parity_claimed"])
        self.assertEqual(gate["blockers"], [])
        self.assertTrue(gate["metadata"]["ready"])
        for export in gate["source_exports"]:
            self.assertTrue(export["ready"])
            self.assertEqual(export["schema_blockers"], [])


def _blocker_rules(gate: dict[str, object]) -> set[str]:
    return {str(item["blocker"]) for item in gate["blockers"]}


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


def _tolerance(value: float, comparator: str) -> dict[str, object]:
    return {
        "status": "available",
        "value": value,
        "comparator": comparator,
        "source": FLUENT_SOURCE,
        "rationale": "temporary complete contract for import-gate testing",
    }


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


def _write_fluent_metadata(path: Path) -> None:
    lines = [
        "# ANSYS Fluent Reference Metadata",
        "",
        "- Source document: ANSYS Fluent vertical flap FSI run record",
        "- Fluent run id: run-2026-07-01",
        "- Export author: local validation operator",
        "- Export date: 2026-07-01",
        "- Fluent version: ANSYS Fluent 2025 R1",
        "- mesh/domain source: ANSYS vertical-flap lower-symmetry-half domain",
        "- geometry units: m",
        "- material model: air and linear elastic vertical flap",
        "- boundary conditions: velocity inlet, pressure outlet, symmetry, moving flap",
        "- time step: 0.0005",
        "- number of steps: 50",
        "- coupling settings if applicable: two-way FSI report export",
        "- export procedure: Fluent report definitions exported at step 50",
        "- who/when/how generated: local validation operator exported CSV reports",
        "- force_z_positive: positive z is reported by Fluent force monitor",
        "- flow_rate_positive: positive outlet flux leaves the lower-half domain",
        "- pressure_reference: Fluent gauge pressure relative to pressure outlet",
        "- displacement_definition: total displacement norm at the flap tip point",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
