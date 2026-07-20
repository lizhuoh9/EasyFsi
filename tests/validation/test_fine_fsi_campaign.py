from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refactored.validation.ansys_vertical_flap_fsi.fine_fsi_campaign import (
    CampaignValidationError,
    discover_step_pairs,
    field_statistics,
    prepare_new_output_dir,
    read_structure_snapshot,
    validate_phase_manifest,
)


CAMPAIGN_ROOT = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "official_fluent_fine_fsi_valid_2026-07-10"
)
LAUNCHER_PATH = CAMPAIGN_ROOT / "scripts" / "run_fine_fsi_campaign.py"
LEGACY_FINE_ROOT = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "official_fluent_fine_mesh_steady_2026-07-01"
    / "fsi_50step_serial_from_adapt_cycle3_mesh"
)
LEGACY_FINE_CASE = LEGACY_FINE_ROOT / "fine_fsi_50step_final.cas.h5"
LEGACY_FINE_DATA = LEGACY_FINE_ROOT / "fine_fsi_50step_final.dat.h5"
LEGACY_FINE_TRANSCRIPT = LEGACY_FINE_ROOT / "fine_fsi_50step_run.trn"
VALID_COARSE_ROOT = Path(
    os.environ.get(
        "EASYFSI_VALID_COARSE_FLUENT_ROOT",
        str(CAMPAIGN_ROOT / "external_reference"),
    )
).expanduser()
VALID_COARSE_CASE = VALID_COARSE_ROOT / "official_fsi_1step.cas.h5"
VALID_COARSE_DATA = VALID_COARSE_ROOT / "official_fsi_1step.dat.h5"


def _write_structure_pair(
    case_path: Path,
    data_path: Path,
    displacement_rows: np.ndarray,
) -> None:
    rows = np.asarray(displacement_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError("displacement_rows must have shape (node_count, 2)")
    node_count = rows.shape[0]
    coordinates = np.column_stack(
        (
            np.linspace(0.0500, 0.0510, node_count),
            np.linspace(0.0090, 0.0100, node_count),
            np.zeros(node_count),
        )
    )
    with h5py.File(case_path, "w") as case_file:
        case_file.create_dataset("meshes/1/nodes/coords/8", data=coordinates)

    structure_rows = np.zeros((node_count, 14), dtype=np.float64)
    structure_rows[:, 0] = rows[:, 0]
    structure_rows[:, 6] = rows[:, 1]
    with h5py.File(data_path, "w") as data_file:
        data_file.create_dataset(
            "special/structure-direct-data/data",
            data=np.arange(11431, 11445, dtype=np.int64),
        )
        node_group = data_file.create_group("special/structure-node-data/nodes/8")
        node_group.create_dataset(
            "elemids", data=np.arange(1, node_count + 1, dtype=np.int64)
        )
        node_group.create_dataset(
            "ndata", data=np.full(node_count, 14, dtype=np.int64)
        )
        node_group.create_dataset("data", data=structure_rows.reshape(-1))


def _write_phase_manifest(run_dir: Path, step_count: int, dt_s: float) -> None:
    steps = [
        {
            "step": step,
            "time_s": step * dt_s,
            "case_path": str((run_dir / "steps" / f"step_{step:04d}.cas.h5").resolve()),
            "data_path": str((run_dir / "steps" / f"step_{step:04d}.dat.h5").resolve()),
        }
        for step in range(1, step_count + 1)
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "requested_steps": step_count,
                "completed_steps": step_count,
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )


def _load_launcher_module():
    if not LAUNCHER_PATH.is_file():
        raise AssertionError(f"fine-grid campaign launcher is missing: {LAUNCHER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "fine_fsi_campaign_launcher_for_integration_tests",
        LAUNCHER_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import campaign launcher: {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FineFsiCampaignContractTests(unittest.TestCase):
    def test_discovers_strictly_consecutive_case_data_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            steps.mkdir()
            for step in (1, 2):
                (steps / f"step_{step:04d}.cas.h5").touch()
                (steps / f"step_{step:04d}.dat.h5").touch()

            pairs = discover_step_pairs(run_dir)

            self.assertEqual([pair.step for pair in pairs], [1, 2])
            self.assertEqual(pairs[0].case_path.name, "step_0001.cas.h5")
            self.assertEqual(pairs[1].data_path.name, "step_0002.dat.h5")

    def test_orphan_data_file_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            steps.mkdir()
            (steps / "step_0001.dat.h5").touch()

            with self.assertRaisesRegex(
                CampaignValidationError,
                "case/data step mismatch",
            ):
                discover_step_pairs(run_dir)

    def test_missing_data_in_middle_of_campaign_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            steps.mkdir()
            for step in (1, 3):
                (steps / f"step_{step:04d}.cas.h5").touch()
                (steps / f"step_{step:04d}.dat.h5").touch()
            (steps / "step_0002.cas.h5").touch()

            with self.assertRaisesRegex(
                CampaignValidationError,
                r"case/data step mismatch:.*missing_data=\[2\]",
            ):
                discover_step_pairs(run_dir)

    def test_paired_but_nonconsecutive_campaign_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            steps.mkdir()
            for step in (1, 3):
                (steps / f"step_{step:04d}.cas.h5").touch()
                (steps / f"step_{step:04d}.dat.h5").touch()

            with self.assertRaisesRegex(
                CampaignValidationError,
                "must start at 1 and be consecutive",
            ):
                discover_step_pairs(run_dir)

    def test_nonfinite_field_fails_loudly(self) -> None:
        with self.assertRaisesRegex(CampaignValidationError, "not finite"):
            field_statistics("velocity", np.array([0.0, np.nan]))

    def test_structure_snapshot_rejects_zero_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "step_0001.cas.h5"
            data_path = Path(tmp) / "step_0001.dat.h5"
            _write_structure_pair(case_path, data_path, np.zeros((2, 2)))

            with self.assertRaisesRegex(
                CampaignValidationError,
                "structure displacement is zero",
            ):
                read_structure_snapshot(case_path, data_path)

    def test_structure_snapshot_rejects_nan_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "step_0001.cas.h5"
            data_path = Path(tmp) / "step_0001.dat.h5"
            _write_structure_pair(
                case_path,
                data_path,
                np.array([[0.0, 0.0], [np.nan, 1.0e-6]]),
            )

            with self.assertRaisesRegex(
                CampaignValidationError,
                "structure node data is not finite",
            ):
                read_structure_snapshot(case_path, data_path)

    def test_structure_snapshot_accepts_finite_nonzero_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "step_0001.cas.h5"
            data_path = Path(tmp) / "step_0001.dat.h5"
            _write_structure_pair(
                case_path,
                data_path,
                np.array([[0.0, 0.0], [3.0e-6, 4.0e-6]]),
            )

            report = read_structure_snapshot(case_path, data_path)

        self.assertAlmostEqual(report["max_displacement_m"], 5.0e-6)
        self.assertEqual(report["nonzero_displacement_node_count"], 1)

    def test_structure_snapshot_matches_coordinate_section_to_node_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "step_0001.cas.h5"
            data_path = Path(tmp) / "step_0001.dat.h5"
            _write_structure_pair(
                case_path,
                data_path,
                np.array([[0.0, 0.0], [3.0e-6, 4.0e-6]]),
            )
            with h5py.File(case_path, "a") as case_file:
                case_file.create_dataset(
                    "meshes/1/nodes/coords/7",
                    data=np.array([[10.0, 10.0, 0.0], [11.0, 11.0, 0.0]]),
                )

            report = read_structure_snapshot(case_path, data_path)

        self.assertLess(report["selected_max_initial_distance_m"], 0.01)

    def test_phase_manifest_must_prove_expected_step_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            steps.mkdir()
            for step in (1, 2):
                (steps / f"step_{step:04d}.cas.h5").touch()
                (steps / f"step_{step:04d}.dat.h5").touch()
            _write_phase_manifest(run_dir, 2, 5.0e-4)
            pairs = discover_step_pairs(run_dir)

            with self.assertRaisesRegex(
                CampaignValidationError,
                "expected 50 paired steps",
            ):
                validate_phase_manifest(
                    run_dir,
                    pairs,
                    expected_steps=50,
                    dt_s=5.0e-4,
                )

    def test_phase_manifest_rejects_time_step_misalignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            steps.mkdir()
            (steps / "step_0001.cas.h5").touch()
            (steps / "step_0001.dat.h5").touch()
            _write_phase_manifest(run_dir, 1, 5.0e-4)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["steps"][0]["time_s"] = 0.25
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(CampaignValidationError, "time mismatch"):
                validate_phase_manifest(
                    run_dir,
                    discover_step_pairs(run_dir),
                    expected_steps=1,
                    dt_s=5.0e-4,
                )

    @unittest.skipUnless(
        VALID_COARSE_CASE.is_file() and VALID_COARSE_DATA.is_file(),
        "local official coarse-grid Fluent 1-step reference is unavailable",
    )
    def test_real_valid_coarse_one_step_passes_structure_and_topology_gates(
        self,
    ) -> None:
        launcher = _load_launcher_module()

        topology = launcher.require_supported_solid_topology(VALID_COARSE_CASE)
        structure = read_structure_snapshot(VALID_COARSE_CASE, VALID_COARSE_DATA)

        self.assertEqual(topology["cell_type_counts"], {3: 30})
        self.assertEqual(topology["cell_count"], 30)
        self.assertGreater(structure["max_displacement_m"], 1.0e-6)
        self.assertLess(structure["max_displacement_m"], 1.0e-4)

    @unittest.skipUnless(
        LEGACY_FINE_CASE.is_file() and LEGACY_FINE_DATA.is_file(),
        "legacy fine-grid Fluent artifact is unavailable",
    )
    def test_real_legacy_fine_case_is_rejected_as_invalid_fsi(self) -> None:
        launcher = _load_launcher_module()

        with self.assertRaisesRegex(
            RuntimeError,
            r"solid\.5 changed from 30 to 198 cells|type 7",
        ):
            launcher.require_supported_solid_topology(LEGACY_FINE_CASE)
        with self.assertRaisesRegex(
            CampaignValidationError,
            "structure displacement is zero",
        ):
            read_structure_snapshot(LEGACY_FINE_CASE, LEGACY_FINE_DATA)

    @unittest.skipUnless(
        LEGACY_FINE_TRANSCRIPT.is_file(),
        "legacy fine-grid Fluent transcript is unavailable",
    )
    def test_real_legacy_fine_transcript_is_rejected(self) -> None:
        launcher = _load_launcher_module()
        transcript = LEGACY_FINE_TRANSCRIPT.read_text(
            encoding="utf-8",
            errors="replace",
        )

        errors = launcher.find_fatal_transcript_errors(transcript)

        self.assertIn("compute processes interrupted", errors)
        self.assertIn("element type not implemented", errors)

    def test_output_directory_must_be_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "existing"
            output_dir.mkdir()

            with self.assertRaisesRegex(
                CampaignValidationError,
                "already exists",
            ):
                prepare_new_output_dir(output_dir)


if __name__ == "__main__":
    unittest.main()
