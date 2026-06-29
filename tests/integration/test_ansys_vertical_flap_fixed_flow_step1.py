from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CASE_DIR = ROOT / "validation_cases" / "ansys_vertical_flap_fixed_flow"
CONFIG_PATH = CASE_DIR / "config.yaml"
RUNNER_PATH = CASE_DIR / "run_fixed_flap_flow.py"
DEFAULT_OUTPUT_ROOT = ROOT / "validation_runs" / "ansys_vertical_flap_fixed_flow"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AnsysVerticalFlapFixedFlowStep1Tests(unittest.TestCase):
    def test_fixed_flow_preprocessor_generates_case_contract(self):
        self.assertTrue(CONFIG_PATH.exists(), msg=str(CONFIG_PATH))
        self.assertTrue(RUNNER_PATH.exists(), msg=str(RUNNER_PATH))

        from refactored.validation.ansys_vertical_flap_fixed.preprocess_fixed_flap import (
            load_config,
            run_preprocess,
        )

        tmp_parent = ROOT / "tmp"
        tmp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="ansys_fixed_flow_step1_", dir=tmp_parent
        ) as tmp_dir:
            config = load_config(CONFIG_PATH)
            config = {
                **config,
                "output": {
                    **config["output"],
                    "root": str(Path(tmp_dir) / "fixed_flow_contract"),
                },
            }
            result = run_preprocess(config)

            output_root = Path(result["output_root"])
            geometry_npz = output_root / "preprocess" / "geometry_mask.npz"
            bc_npz = output_root / "preprocess" / "bc_map.npz"
            fields_npz = output_root / "fields" / "initial_fields.npz"
            preview_png = output_root / "rendered_results" / "geometry_preview.png"
            manifest_json = output_root / "case_manifest.json"

            self.assertTrue(geometry_npz.exists(), msg=str(geometry_npz))
            self.assertTrue(bc_npz.exists(), msg=str(bc_npz))
            self.assertTrue(fields_npz.exists(), msg=str(fields_npz))
            self.assertTrue(preview_png.exists(), msg=str(preview_png))
            self.assertTrue(manifest_json.exists(), msg=str(manifest_json))
            self.assertEqual(preview_png.read_bytes()[:8], PNG_MAGIC)

            with np.load(geometry_npz) as geometry, np.load(bc_npz) as bc, np.load(
                fields_npz
            ) as fields:
                self.assertEqual(geometry["solid_mask"].shape, (128, 360))
                self.assertEqual(geometry["fluid_mask"].shape, (128, 360))
                self.assertEqual(geometry["S"].shape, (128, 360))
                self.assertEqual(geometry["Y"].shape, (128, 360))
                np.testing.assert_allclose(geometry["Z"], -geometry["S"])

                solid = geometry["solid_mask"].astype(bool)
                fluid = geometry["fluid_mask"].astype(bool)
                upper_flap = geometry["upper_flap_mask"].astype(bool)
                lower_flap = geometry["lower_flap_mask"].astype(bool)
                gap = geometry["gap_mask"].astype(bool)

                self.assertTrue(solid[0, :].all())
                self.assertTrue(solid[-1, :].all())
                self.assertTrue(upper_flap.any())
                self.assertTrue(lower_flap.any())
                self.assertTrue(gap.any())
                self.assertTrue(solid[upper_flap].all())
                self.assertTrue(solid[lower_flap].all())
                self.assertTrue(fluid[gap].all())

                flap_column = int(np.argmin(np.abs(geometry["s"] - 0.048)))
                self.assertGreater(int(upper_flap[:, flap_column].sum()), 0)
                self.assertGreater(int(lower_flap[:, flap_column].sum()), 0)
                self.assertGreater(int(gap[:, flap_column].sum()), 0)

                inlet = bc["inlet_mask"].astype(bool)
                outlet = bc["outlet_mask"].astype(bool)
                inlet_Uz = bc["inlet_Uz"]
                inlet_Uy = bc["inlet_Uy"]

                self.assertEqual(inlet.shape, (128, 360))
                self.assertEqual(outlet.shape, (128, 360))
                self.assertTrue(np.logical_or(~inlet, fluid).all())
                self.assertTrue(np.logical_or(~outlet, fluid).all())
                self.assertEqual(int(inlet.sum()), int(fluid[:, 0].sum()))
                self.assertEqual(int(outlet.sum()), int(fluid[:, -1].sum()))
                self.assertTrue(inlet[:, 0].any())
                self.assertFalse(inlet[:, 1:].any())
                self.assertTrue(outlet[:, -1].any())
                self.assertFalse(outlet[:, :-1].any())
                self.assertTrue((inlet_Uz[inlet] < 0.0).all())
                np.testing.assert_allclose(inlet_Uy[inlet], 0.0)
                np.testing.assert_allclose(bc["outlet_pressure"][outlet], 0.0)

                for key in ("Uz", "Uy", "p", "streamwise_minus_Uz"):
                    self.assertIn(key, fields.files)
                    self.assertEqual(fields[key].shape, (128, 360))
                np.testing.assert_allclose(fields["streamwise_minus_Uz"], -fields["Uz"])
                self.assertTrue((fields["Uz"][inlet] < 0.0).all())
                np.testing.assert_allclose(fields["Uy"][fluid], 0.0)

            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["case"], "ansys_vertical_flap_fixed_flow")
            self.assertEqual(
                manifest["scope"],
                "fixed-flap flow preprocessing only; no solver step executed",
            )
            self.assertEqual(
                manifest["sign_convention"],
                "left_to_right_display_flow_has_Uz_negative",
            )
            self.assertEqual(manifest["streamwise_display_velocity"], "-Uz")
            self.assertEqual(manifest["claims"]["fluent_parity"], "not_claimed")
            self.assertEqual(manifest["claims"]["fsi"], "not_claimed")
            self.assertEqual(manifest["claims"]["solver_result"], "not_claimed")

    def test_default_runner_artifact_bundle_matches_manifest(self):
        required = {
            "geometry_mask": DEFAULT_OUTPUT_ROOT
            / "preprocess"
            / "geometry_mask.npz",
            "bc_map": DEFAULT_OUTPUT_ROOT / "preprocess" / "bc_map.npz",
            "initial_fields": DEFAULT_OUTPUT_ROOT / "fields" / "initial_fields.npz",
            "geometry_preview": DEFAULT_OUTPUT_ROOT
            / "rendered_results"
            / "geometry_preview.png",
            "case_manifest": DEFAULT_OUTPUT_ROOT / "case_manifest.json",
        }

        for path in required.values():
            self.assertTrue(path.exists(), msg=str(path))

        self.assertEqual(required["geometry_preview"].read_bytes()[:8], PNG_MAGIC)

        manifest = json.loads(
            required["case_manifest"].read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["source"],
            "validation_cases/ansys_vertical_flap_fixed_flow/config.yaml",
        )
        self.assertEqual(
            manifest["scope"],
            "fixed-flap flow preprocessing only; no solver step executed",
        )
        self.assertEqual(manifest["claims"]["fluent_parity"], "not_claimed")
        self.assertEqual(manifest["claims"]["fsi"], "not_claimed")
        self.assertIn("generated_files", manifest)

        for key, path in required.items():
            manifest_path = _resolve_manifest_path(manifest["generated_files"][key])
            self.assertEqual(manifest_path, path)


def _resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    unittest.main()
