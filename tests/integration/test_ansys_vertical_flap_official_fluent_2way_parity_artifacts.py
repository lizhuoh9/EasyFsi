from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_RUNNER_PATH = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_official_fluent_2way_fsi50_snapshot.py"
)


def _load_snapshot_runner():
    spec = importlib.util.spec_from_file_location(
        "official_fluent_fsi50_snapshot",
        SNAPSHOT_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OfficialFluent2WaySnapshotArtifactTests(unittest.TestCase):
    def test_snapshot_solver_history_exports_face_forces_and_mpm_motion(self):
        runner = _load_snapshot_runner()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "solver_history.csv"
            summary = runner._write_solver_history_csv(
                path,
                [
                    {
                        "step": 1,
                        "flow_step_index_global": 20,
                        "flow_driver_mode": "sustained_inlet_predictor",
                        "flow_predictor_applied": True,
                        "local_velocity_peak_mps": 28.0,
                        "fluid_speed_p99_mps": 20.0,
                        "fluid_speed_p999_mps": 27.0,
                        "pressure_min_pa": -10.0,
                        "pressure_max_pa": 400.0,
                        "total_marker_force_n": (0.0, 0.0, -0.01),
                        "primary_face_force_n": (0.0, 0.0, -0.012),
                        "secondary_face_force_n": (0.0, 0.0, 0.002),
                        "primary_face_force_z_N": -0.012,
                        "secondary_face_force_z_N": 0.002,
                        "primary_plus_secondary_force_z_N": -0.01,
                        "force_decomposition_residual_N": 0.0,
                        "marker_force_z_N": -0.01,
                        "mpm_external_force_n": (0.0, 0.0, -0.009),
                        "mpm_primary_mean_velocity_mps": (0.0, 0.0, -0.1),
                        "mpm_secondary_mean_velocity_mps": (0.0, 0.0, -0.2),
                        "mpm_primary_mean_displacement_m": (0.0, 0.0, -1e-4),
                        "mpm_secondary_mean_displacement_m": (0.0, 0.0, -2e-4),
                        "mpm_active_grid_nodes": 10,
                        "mpm_grid_out_of_bounds_particle_count": 0,
                        "mpm_max_speed_mps": 0.2,
                        "mpm_deformation_clamp_count": 0,
                        "source_volume_flux_m3s": 1e-5,
                        "positive_source_volume_flux_m3s": 1e-5,
                        "abs_source_volume_flux_m3s": 1e-5,
                        "zmin_pressure_outlet_flux_m3s": -9e-6,
                        "zmin_velocity_outlet_flux_m3s": -8e-6,
                        "pressure_outlet_flux_ratio": -0.9,
                        "velocity_outlet_flux_ratio": -0.8,
                        "tip_mean_displacement_m": (0.0, 0.0, -1e-4),
                        "max_displacement_m": 1.1e-4,
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")

        self.assertEqual(summary["row_count"], 1)
        self.assertIn("primary_face_force_z_N", text)
        self.assertIn("mpm_primary_mean_velocity_z_mps", text)
        self.assertIn("pressure_outlet_flux_ratio", text)
        self.assertIn("-0.012", text)
        self.assertIn("-0.1", text)

    def test_snapshot_runner_accepts_positive_tuple_overrides(self):
        runner = _load_snapshot_runner()

        self.assertEqual(runner._parse_int_tuple3("4,64,128"), (4, 64, 128))
        self.assertEqual(runner._parse_int_tuple3(" 1, 2, 3 "), (1, 2, 3))

        with self.assertRaisesRegex(Exception, "three comma-separated integers"):
            runner._parse_int_tuple3("4,64")
        with self.assertRaisesRegex(Exception, "positive"):
            runner._parse_int_tuple3("4,0,128")

    def test_snapshot_runner_derives_grid_from_official_reference_cells(self):
        runner = _load_snapshot_runner()
        with tempfile.TemporaryDirectory() as tmp:
            reference_root = Path(tmp)
            x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=np.float64)
            y = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.25], dtype=np.float64)
            zeros = np.zeros_like(x)
            np.savez(
                reference_root / "steady_fluent_fields.npz",
                x=x,
                y=y,
                u=zeros,
                v=zeros,
                p=zeros,
                speed=zeros,
                cell_ids=np.arange(x.size, dtype=np.int64),
            )

            self.assertEqual(
                runner._official_reference_grid_nodes(
                    reference_root,
                    span_nodes=3,
                ),
                (3, 2, 3),
            )

    def test_preflow_only_solver_history_can_be_exported(self):
        runner = _load_snapshot_runner()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "solver_history.csv"
            summary = runner._write_solver_history_csv(
                path,
                [
                    {
                        "flow_step_index_global": 1,
                        "flow_driver_mode": "sustained_boundary_predictor",
                        "flow_predictor_applied": True,
                        "local_velocity_peak_mps": 22.0,
                        "pressure_min_pa": -12.0,
                        "pressure_max_pa": 120.0,
                        "source_volume_flux_m3s": 1.0e-5,
                        "zmin_pressure_outlet_flux_m3s": -9.0e-6,
                    },
                    {
                        "flow_step_index_global": 2,
                        "flow_driver_mode": "sustained_boundary_predictor",
                        "flow_predictor_applied": True,
                        "local_velocity_peak_mps": 24.0,
                        "pressure_min_pa": -15.0,
                        "pressure_max_pa": 150.0,
                        "source_volume_flux_m3s": 1.0e-5,
                        "zmin_pressure_outlet_flux_m3s": -9.5e-6,
                    },
                ],
            )

            text = path.read_text(encoding="utf-8")

        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["local_velocity_peak_mps"], 24.0)
        self.assertIn("sustained_boundary_predictor", text)
        self.assertIn("zmin_pressure_outlet_flux_m3s", text)


if __name__ == "__main__":
    unittest.main()
