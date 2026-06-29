from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
GEOMETRY_PATH = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fixed_flow"
    / "preprocess"
    / "geometry_mask.npz"
)
BC_PATH = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fixed_flow"
    / "preprocess"
    / "bc_map.npz"
)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AnsysVerticalFlapFixedFlowStep2SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from refactored.validation.ansys_vertical_flap_fixed.operators import (
            infer_spacing,
        )
        from refactored.validation.ansys_vertical_flap_fixed.projection_solver import (
            run_projection_solver,
        )

        cls.geometry = dict(np.load(GEOMETRY_PATH))
        cls.bc_map = dict(np.load(BC_PATH))
        cls.ds, cls.dy = infer_spacing(cls.geometry["s"], cls.geometry["y"])

        cls.tmp_parent = ROOT / "tmp"
        cls.tmp_parent.mkdir(exist_ok=True)
        cls.tmpdir = tempfile.TemporaryDirectory(
            prefix="ansys_fixed_flow_step2_", dir=cls.tmp_parent
        )
        cls.output_root = Path(cls.tmpdir.name) / "fixed_flow_solver"
        cls.result = run_projection_solver(
            GEOMETRY_PATH,
            BC_PATH,
            cls.output_root,
            config={
                "solver": {
                    "max_steps": 80,
                    "cfl": 0.35,
                    "steady_tolerance": 0.0,
                    "divergence_tolerance": 1.0e-3,
                    "poisson_max_iters": 60,
                    "poisson_tolerance": 1.0e-5,
                    "poisson_omega": 1.0,
                    "history_interval": 10,
                    "write_checkpoints": False,
                }
            },
        )
        cls.final_fields_path = cls.output_root / "fields" / "final_fields.npz"
        cls.history_path = cls.output_root / "logs" / "solver_history.csv"
        cls.mass_balance_path = cls.output_root / "logs" / "mass_balance.csv"
        cls.manifest_path = cls.output_root / "case_manifest_step2.json"

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_solver_consumes_step1_geometry_and_bc_contract(self):
        self.assertTrue(GEOMETRY_PATH.exists(), msg=str(GEOMETRY_PATH))
        self.assertTrue(BC_PATH.exists(), msg=str(BC_PATH))
        self.assertEqual(self.geometry["solid_mask"].shape, (128, 360))
        self.assertEqual(self.geometry["fluid_mask"].shape, (128, 360))
        self.assertEqual(self.bc_map["inlet_mask"].shape, (128, 360))
        self.assertEqual(self.bc_map["outlet_mask"].shape, (128, 360))
        self.assertGreater(self.ds, 0.0)
        self.assertGreater(self.dy, 0.0)

        inlet = self.bc_map["inlet_mask"].astype(bool)
        inlet_u = -self.bc_map["inlet_Uz"][inlet]
        self.assertTrue((inlet_u > 0.0).all())
        np.testing.assert_allclose(inlet_u, 7.0)

    def test_short_projection_output_preserves_masks_and_boundary_conditions(self):
        self.assertTrue(self.final_fields_path.exists(), msg=str(self.final_fields_path))

        with np.load(self.final_fields_path) as fields:
            solid = fields["solid_mask"].astype(bool)
            inlet = fields["inlet_mask"].astype(bool)
            for key in (
                "u",
                "v",
                "Uz",
                "Uy",
                "p",
                "speed",
                "streamwise_minus_Uz",
            ):
                self.assertIn(key, fields.files)
                self.assertEqual(fields[key].shape, (128, 360))
                self.assertFalse(np.isnan(fields[key]).any(), msg=key)
                self.assertFalse(np.isinf(fields[key]).any(), msg=key)

            np.testing.assert_allclose(fields["u"][solid], 0.0)
            np.testing.assert_allclose(fields["v"][solid], 0.0)
            np.testing.assert_allclose(fields["speed"][solid], 0.0)
            np.testing.assert_allclose(fields["u"][inlet], 7.0)
            np.testing.assert_allclose(fields["v"][inlet], 0.0)
            np.testing.assert_allclose(fields["Uz"], -fields["u"])
            np.testing.assert_allclose(fields["Uy"], fields["v"])
            np.testing.assert_allclose(fields["streamwise_minus_Uz"], fields["u"])
            self.assertTrue((fields["speed"] >= 0.0).all())

    def test_projection_solver_produces_gap_acceleration_and_downstream_jet(self):
        with np.load(self.final_fields_path) as fields:
            u = fields["u"]
            v = fields["v"]
            fluid = fields["fluid_mask"].astype(bool)
            gap = self.geometry["gap_mask"].astype(bool)
            s = fields["s"]
            y = fields["y"]
            center_row = int(np.argmin(np.abs(y)))
            flap_center_s = 0.048
            downstream = s > flap_center_s + 0.004
            centerline_fluid = fluid[center_row, :] & downstream

            self.assertGreater(float(u[gap].mean()), 7.0)
            self.assertGreater(float(u[fluid].max()), 10.5)
            self.assertGreater(float(u[center_row, centerline_fluid].max()), 7.5)
            self.assertGreater(float(np.max(np.abs(v[fluid]))), 1.0e-6)

    def test_diagnostics_and_manifest_are_written_and_non_parity_claims(self):
        self.assertTrue(self.history_path.exists(), msg=str(self.history_path))
        self.assertTrue(
            self.mass_balance_path.exists(), msg=str(self.mass_balance_path)
        )
        self.assertTrue(self.manifest_path.exists(), msg=str(self.manifest_path))

        history_rows = _read_csv(self.history_path)
        mass_rows = _read_csv(self.mass_balance_path)
        self.assertGreaterEqual(len(history_rows), 2)
        self.assertGreaterEqual(len(mass_rows), 2)
        for key in (
            "max_speed",
            "divergence_linf",
            "mass_imbalance_rel",
            "poisson_iters",
            "poisson_residual_linf",
            "interior_max_speed_excluding_near_solid",
        ):
            self.assertIn(key, history_rows[-1])

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["case"], "ansys_vertical_flap_fixed_flow")
        self.assertEqual(
            manifest["step"], "step2_fixed_flap_projection_solver"
        )
        self.assertEqual(manifest["claims"]["fluent_parity"], "not_claimed")
        self.assertEqual(manifest["claims"]["fsi"], "not_claimed")
        self.assertEqual(
            manifest["sources"]["geometry_mask"],
            "validation_runs/ansys_vertical_flap_fixed_flow/preprocess/geometry_mask.npz",
        )
        self.assertEqual(
            manifest["sources"]["bc_map"],
            "validation_runs/ansys_vertical_flap_fixed_flow/preprocess/bc_map.npz",
        )
        self.assertEqual(
            manifest["forbidden_sources"]["traction_shared_snapshot_diagnostics"],
            "not_used",
        )
        self.assertNotIn(
            "traction_shared_snapshot_diagnostics",
            manifest["sources"]["geometry_mask"],
        )
        self.assertNotIn(
            "traction_shared_snapshot_diagnostics",
            manifest["sources"]["bc_map"],
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
