from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
FIELD_NPZ = DIAG_ROOT / "step020_fields.npz"
MANIFEST_JSON = DIAG_ROOT / "snapshot_manifest.json"
SUMMARY_MD = DIAG_ROOT / "snapshot_summary.md"
VERIFICATION_MD = DIAG_ROOT / "verification_shared_snapshot_2026-06-26.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"

REQUIRED_ARRAYS = {
    "pressure",
    "velocity",
    "obstacle",
    "cell_face_x_m",
    "cell_face_y_m",
    "cell_face_z_m",
    "cell_center_x_m",
    "cell_center_y_m",
    "cell_center_z_m",
    "cell_width_x_m",
    "cell_width_y_m",
    "cell_width_z_m",
    "grid_nodes",
    "preflow_step",
    "dt_s",
    "inlet_velocity_mps",
    "source_strength",
    "source_ramp_steps",
}


class AnsysVerticalFlapTractionSharedSnapshotArtifactTests(unittest.TestCase):
    def test_shared_snapshot_npz_and_manifest_are_complete(self):
        self.assertTrue(DIAG_ROOT.exists(), DIAG_ROOT)
        for path in (
            FIELD_NPZ,
            MANIFEST_JSON,
            SUMMARY_MD,
            VERIFICATION_MD,
            CHECKSUMS,
        ):
            self.assertTrue(path.exists(), path)

        manifest = _read_json(MANIFEST_JSON)
        with np.load(FIELD_NPZ) as fields:
            self.assertTrue(REQUIRED_ARRAYS.issubset(set(fields.files)))
            grid_nodes = tuple(int(value) for value in fields["grid_nodes"])
            pressure = fields["pressure"]
            velocity = fields["velocity"]
            obstacle = fields["obstacle"]

            self.assertEqual(pressure.shape, grid_nodes)
            self.assertEqual(obstacle.shape, pressure.shape)
            self.assertEqual(velocity.shape, pressure.shape + (3,))
            self.assertTrue(np.all(np.isfinite(pressure)))
            self.assertTrue(np.all(np.isfinite(velocity)))
            self.assertEqual(int(fields["preflow_step"][0]), 20)
            self.assertAlmostEqual(float(fields["source_strength"][0]), 0.80)
            self.assertEqual(int(fields["source_ramp_steps"][0]), 2)
            if "sampling_obstacle" in fields.files:
                self.assertEqual(fields["sampling_obstacle"].shape, pressure.shape)

        self.assertEqual(manifest["case"], "ansys-vertical-flap-fsi")
        self.assertEqual(manifest["preflow_steps"], 20)
        self.assertEqual(manifest["preflow_steps_completed"], 20)
        self.assertEqual(manifest["step_count"], 0)
        self.assertEqual(manifest["flow_driver_mode"], "sustained_volume_source_inlet")
        self.assertAlmostEqual(float(manifest["source_strength"]), 0.80)
        self.assertEqual(manifest["source_profile"], "linear_ramp")
        self.assertEqual(manifest["source_ramp_steps"], 2)
        self.assertEqual(manifest["source_schedule_scope"], "global")
        self.assertEqual(manifest["field_sha256"], _sha256_file(FIELD_NPZ))
        self.assertEqual(manifest["field_path"], FIELD_NPZ.as_posix())
        self.assertEqual(manifest["reference_formulation_candidate"], "none")
        self.assertEqual(
            manifest["candidate_status"],
            "snapshot_only_no_reference_selection",
        )
        self.assertIn("no coupled 50-step", manifest["scope_limit"])
        self.assertIn("Fluent parity", manifest["scope_limit"])
        self.assertNotEqual(manifest["marker_geometry_sha256"], "")
        self.assertEqual(
            manifest["marker_geometry_sha256"],
            _sha256_stable_json(manifest["marker_geometry"]),
        )
        self.assertTrue(
            {
                "pressure",
                "velocity",
                "obstacle",
                "cell_face_x_m",
                "cell_center_x_m",
                "cell_width_x_m",
            }.issubset(manifest["field_arrays"])
        )

    def test_summary_and_checksums_preserve_snapshot_scope(self):
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("fixed-solid shared flow snapshot only", summary)
        self.assertIn("exact same pressure/velocity field", summary)
        self.assertIn("no reference formulation is selected", summary)
        self.assertIn("does not prove Fluent parity", summary)
        self.assertIn("next_intended_step = snapshot resampling matrix", summary)

        checksum_rows = _read_checksums(CHECKSUMS)
        for artifact in (
            FIELD_NPZ.name,
            MANIFEST_JSON.name,
            SUMMARY_MD.name,
            VERIFICATION_MD.name,
        ):
            self.assertIn(artifact, checksum_rows)
            self.assertEqual(checksum_rows[artifact], _sha256_file(DIAG_ROOT / artifact))


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


def _sha256_stable_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
