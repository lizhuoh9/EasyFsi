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
IMPORTER_PATH = SCRIPT_ROOT / "import_real_fluent_source_exports.py"

FLUENT_SOURCE = (
    "ANSYS Fluent 2025 R1 two-way FSI vertical flap run "
    "fluent-run-2026-07-01 report export"
)

CSV_SPECS = {
    "fluent_tip_displacement_history.csv": (
        [
            "step",
            "time_s",
            "tip_displacement_x_m",
            "tip_displacement_y_m",
            "tip_displacement_z_m",
            "tip_displacement_norm_m",
            "max_displacement_m",
            "source",
        ],
        {
            "step": 50,
            "time_s": 0.025,
            "tip_displacement_x_m": 0.0,
            "tip_displacement_y_m": 0.0,
            "tip_displacement_z_m": 5.1e-5,
            "tip_displacement_norm_m": 5.1e-5,
            "max_displacement_m": 6.2e-5,
            "source": FLUENT_SOURCE,
        },
    ),
    "fluent_force_history.csv": (
        [
            "step",
            "time_s",
            "force_x_N",
            "force_y_N",
            "force_z_N",
            "primary_force_z_N",
            "secondary_force_z_N",
            "source",
        ],
        {
            "step": 50,
            "time_s": 0.025,
            "force_x_N": 0.0,
            "force_y_N": 0.0,
            "force_z_N": -1.25e-3,
            "primary_force_z_N": -6.25e-4,
            "secondary_force_z_N": -6.25e-4,
            "source": FLUENT_SOURCE,
        },
    ),
    "fluent_flow_balance_history.csv": (
        [
            "step",
            "time_s",
            "inlet_flow_rate_m3s",
            "outlet_flow_rate_m3s",
            "pressure_outlet_flux_m3s",
            "velocity_outlet_flux_m3s",
            "source",
        ],
        {
            "step": 50,
            "time_s": 0.025,
            "inlet_flow_rate_m3s": 2.5e-5,
            "outlet_flow_rate_m3s": 2.5e-5,
            "pressure_outlet_flux_m3s": 2.5e-5,
            "velocity_outlet_flux_m3s": 2.5e-5,
            "source": FLUENT_SOURCE,
        },
    ),
    "fluent_pressure_summary_history.csv": (
        [
            "step",
            "time_s",
            "pressure_min_pa",
            "pressure_max_pa",
            "pressure_range_pa",
            "source",
        ],
        {
            "step": 50,
            "time_s": 0.025,
            "pressure_min_pa": -120.0,
            "pressure_max_pa": 35.0,
            "pressure_range_pa": 155.0,
            "source": FLUENT_SOURCE,
        },
    ),
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


IMPORTER = _load_module(
    "import_real_fluent_source_exports_for_tests",
    IMPORTER_PATH,
)


class AnsysVerticalFlapRealFluentSourceExportImportTests(unittest.TestCase):
    def test_schema_only_committed_exports_fail_import_preflight(self):
        with self.assertRaises(IMPORTER.ImportPreflightError) as raised:
            IMPORTER.validate_import_bundle(SOURCE_EXPORTS_ROOT)

        self.assertIn("schema_only", raised.exception.summary["blockers"])
        self.assertFalse(raised.exception.summary["ready"])

    def test_missing_required_file_fails_before_destination_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            _write_complete_bundle(source)
            (source / "fluent_pressure_summary_history.csv").unlink()
            _write_sentinel_destination(destination)

            with self.assertRaises(IMPORTER.ImportPreflightError) as raised:
                IMPORTER.import_real_fluent_source_exports(
                    input_dir=source,
                    destination_dir=destination,
                )

            self.assertIn("missing_file", raised.exception.summary["blockers"])
            self.assertEqual(
                (destination / "fluent_tip_displacement_history.csv").read_text(
                    encoding="utf-8"
                ),
                "sentinel\n",
            )

    def test_disallowed_source_provenance_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            _write_complete_bundle(
                source,
                source_text="ANSYS Fluent 2025 R1 official web baseline export",
            )
            _write_sentinel_destination(destination)

            with self.assertRaises(IMPORTER.ImportPreflightError) as raised:
                IMPORTER.import_real_fluent_source_exports(
                    input_dir=source,
                    destination_dir=destination,
                )

            self.assertIn("disallowed_source_provenance", raised.exception.summary["blockers"])
            self.assertEqual(
                (destination / "fluent_tip_displacement_history.csv").read_text(
                    encoding="utf-8"
                ),
                "sentinel\n",
            )

    def test_public_tutorial_metadata_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            _write_complete_bundle(source, metadata_source="public tutorial page")
            _write_sentinel_destination(destination)

            with self.assertRaises(IMPORTER.ImportPreflightError) as raised:
                IMPORTER.import_real_fluent_source_exports(
                    input_dir=source,
                    destination_dir=destination,
                )

            self.assertIn(
                "fluent_reference_metadata_disallowed_provenance",
                raised.exception.summary["blockers"],
            )
            self.assertEqual(
                (destination / "fluent_tip_displacement_history.csv").read_text(
                    encoding="utf-8"
                ),
                "sentinel\n",
            )

    def test_complete_bundle_imports_to_temp_destination_and_readies_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            output_dir = root / "diagnostics"
            active_manifest = root / "active_fluent_reference_contract.json"
            current_contract = root / "fluent_reference_contract.json"
            _write_complete_bundle(source)
            _write_complete_current_contract(current_contract)

            summary = IMPORTER.import_real_fluent_source_exports(
                input_dir=source,
                destination_dir=destination,
                current_contract_json=current_contract,
                output_dir=output_dir,
                active_manifest_json=active_manifest,
                run_collection_validator=True,
            )

        collection = summary["collection"]
        gate = collection["real_fluent_import_gate"]
        self.assertTrue(summary["ready"])
        self.assertEqual(summary["copied_file_count"], 5)
        self.assertEqual(
            collection["candidate_status"],
            "fluent_reference_collection_complete",
        )
        self.assertEqual(
            collection["candidate_contract_status"],
            "fluent_reference_complete",
        )
        self.assertEqual(collection["schema_validation"]["validated_metric_count"], 5)
        self.assertEqual(collection["schema_validation"]["required_metric_count"], 5)
        self.assertEqual(collection["schema_validation"]["missing_required_metrics"], [])
        self.assertEqual(
            collection["promotion_status"],
            "ready_for_versioned_contract_promotion",
        )
        self.assertEqual(gate["status"], "ready_for_real_fluent_import")
        self.assertTrue(gate["can_import_real_fluent_reference"])
        self.assertTrue(gate["can_run_solver_evaluation"])
        self.assertFalse(gate["fluent_parity_claimed"])
        self.assertEqual(gate["blockers"], [])


def _write_complete_bundle(
    root: Path,
    *,
    source_text: str = FLUENT_SOURCE,
    metadata_source: str = "ANSYS Fluent project/run archive fluent-run-2026-07-01",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename, (header, row) in CSV_SPECS.items():
        updated = dict(row)
        updated["source"] = source_text
        with (root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerow(updated)
    _write_metadata(root / "fluent_metadata_2026-06-28.md", metadata_source)


def _write_metadata(path: Path, source_document: str) -> None:
    lines = [
        "# ANSYS Fluent Reference Metadata",
        "",
        f"- Source document: {source_document}",
        "- Fluent run id: fluent-run-2026-07-01",
        "- Export author: local Fluent validation operator",
        "- Export date: 2026-07-01",
        "- Fluent version: ANSYS Fluent 2025 R1",
        "- mesh/domain source: ANSYS vertical-flap lower-symmetry-half Fluent model",
        "- geometry units: m",
        "- material model: air and silicone rubber linear elastic flap",
        "- boundary conditions: velocity inlet, pressure outlet, symmetry, moving flap",
        "- time step: 0.0005",
        "- number of steps: 50",
        "- coupling settings if applicable: two-way Fluent FSI report export",
        "- export procedure: Fluent report definitions exported at step 50",
        "- who/when/how generated: local operator exported Fluent report CSVs",
        "- force_z_positive: positive z is Fluent report force monitor positive z",
        "- flow_rate_positive: positive outlet flux leaves the lower-half domain",
        "- pressure_reference: Fluent gauge pressure relative to pressure outlet",
        "- displacement_definition: total displacement norm at the flap tip point",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sentinel_destination(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "fluent_tip_displacement_history.csv").write_text(
        "sentinel\n",
        encoding="utf-8",
    )


def _write_complete_current_contract(path: Path) -> None:
    contract = json.loads(REAL_CONTRACT.read_text(encoding="utf-8"))
    contract["tolerances"] = {
        "tip_displacement_relative": _tolerance(0.1),
        "max_displacement_relative": _tolerance(0.1),
        "force_z_relative": _tolerance(0.1),
        "flow_rate_relative": _tolerance(0.1),
        "pressure_sanity_absolute": _tolerance(5.0, comparator="absolute_error"),
    }
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tolerance(value: float, *, comparator: str = "relative_error") -> dict[str, object]:
    return {
        "status": "available",
        "value": value,
        "comparator": comparator,
        "source": FLUENT_SOURCE,
        "rationale": "temporary complete contract for import execution testing",
    }


if __name__ == "__main__":
    unittest.main()
