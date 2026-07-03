from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_official_fluent_2way_parity.py"
)
SNAPSHOT_RUNNER_PATH = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_official_fluent_2way_fsi50_snapshot.py"
)
REFERENCE_ROOT = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "official_fluent_2way_reference"
)
OFFICIAL_FIXED_FIELDS = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fixed_flow"
    / "official_fluent_preflow"
    / "stabilized_solver"
    / "fields"
    / "final_fields_stabilized.npz"
)
STEP50_MATRIX = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "traction_selected_formulation_coupled_step50_diagnostics"
    / "traction_selected_formulation_coupled_step50_matrix.json"
)
FSI50_SOLVER_FIELDS = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "official_fluent_2way_solver_snapshot"
    / "step50_solver_u_v_p_fields.npz"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("official_fluent_parity", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_snapshot_runner():
    spec = importlib.util.spec_from_file_location(
        "official_fluent_fsi50_snapshot",
        SNAPSHOT_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OfficialFluent2WayParityArtifactTests(unittest.TestCase):
    def test_error_localization_keeps_fixed_flow_blocker_on_fluid_solver(self):
        runner = _load_runner()
        fixed_result = {
            "status": "failed",
            "metrics": {
                "centerline_u_nrmse": 0.44,
                "downstream_u_nrmse": 0.39,
                "throat_u_nrmse": 0.16,
                "speed_max_rel_error": 0.04,
                "pressure_drop_rel_error": 15.0,
            },
            "gates": {
                "centerline_u_nrmse": {"status": "failed"},
                "downstream_u_nrmse": {"status": "failed"},
                "throat_u_nrmse": {"status": "failed"},
                "speed_max_rel_error": {"status": "passed"},
                "pressure_drop_rel_error": {"status": "failed"},
            },
            "diagnostics": {
                "pressure_extrema": {
                    "reference_pressure_min_point": {
                        "x_m": 0.05048,
                        "y_m": 0.0103,
                        "reference_pressure_pa": -277.49,
                        "solver_pressure_pa": 244.06,
                        "reference_speed_mps": 24.58,
                        "solver_speed_mps": 0.16,
                    }
                },
                "downstream_near_wall_backflow": {
                    "sample_count": 12,
                    "reference_negative_u_fraction": 0.75,
                    "solver_negative_u_fraction": 0.25,
                    "min_u_abs_error_mps": 3.5,
                },
            },
        }

        findings = runner._fixed_flow_localization_findings(fixed_result)
        text = "\n".join(findings)

        self.assertIn("fixed-flow preflow blocker", text)
        self.assertIn("downstream centerline jet", text)
        self.assertIn("pressure-drop scale", text)
        self.assertIn("Fluent reference pressure minimum", text)
        self.assertIn("solver samples p=244.06 Pa", text)
        self.assertIn("Downstream near-wall backflow diagnostic", text)
        self.assertIn("Fluent negative-U fraction `0.75`", text)
        self.assertIn("solver negative-U fraction `0.25`", text)
        self.assertIn("Do not replace this with a dedicated solid/modal model", text)

    def test_diagnostic_pressure_range_is_not_reported_as_failed_gate(self):
        runner = _load_runner()
        fixed_result = {
            "status": "failed",
            "metrics": {
                "downstream_u_nrmse": 0.16,
                "pressure_range_rel_error": 0.32,
                "pressure_drop_rel_error": 0.01,
            },
            "gates": {
                "downstream_u_nrmse": {"status": "failed"},
                "pressure_range_rel_error": {"status": "diagnostic"},
                "pressure_drop_rel_error": {"status": "passed"},
            },
        }

        findings = runner._fixed_flow_localization_findings(fixed_result)
        failed_line = findings[0]

        self.assertEqual(runner._failed_gate_names(fixed_result), ["downstream_u_nrmse"])
        self.assertIn("downstream_u_nrmse", failed_line)
        self.assertNotIn("pressure_range_rel_error", failed_line)
        self.assertNotIn("pressure-drop scale", "\n".join(findings))

    def test_fixed_flow_only_error_report_marks_fsi_and_structure_not_run(self):
        runner = _load_runner()
        fixed_result = {
            "status": "failed",
            "metrics": {"downstream_u_nrmse": 0.16},
            "gates": {"downstream_u_nrmse": {"status": "failed"}},
        }
        not_run = {"status": "not_run", "blockers": []}

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            runner._write_error_localization(output_root, fixed_result, not_run, not_run)
            text = (output_root / "error_localization_report.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("FSI50 full-field parity was not run", text)
        self.assertIn("Structure response parity was not run", text)
        self.assertNotIn("FSI50 final-field mismatch is present", text)
        self.assertNotIn("Structure response mismatch is present", text)

    def test_sampling_points_csv_includes_physical_y_and_pressure_columns(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aligned_sampling_points.csv"
            runner._write_sampling_points(
                path,
                {
                    "cell_ids": np.array([31, 32], dtype=np.int64),
                    "x": np.array([0.01, 0.02], dtype=np.float64),
                    "y": np.array([0.003, 0.004], dtype=np.float64),
                    "u": np.array([10.0, 11.0], dtype=np.float64),
                    "p": np.array([101.0, 102.0], dtype=np.float64),
                    "speed": np.array([10.5, 11.5], dtype=np.float64),
                },
                {
                    "u": np.array([9.5, 10.5], dtype=np.float64),
                    "p": np.array([99.0, 100.0], dtype=np.float64),
                    "speed": np.array([9.8, 10.8], dtype=np.float64),
                    "valid": np.array([True, False], dtype=bool),
                },
                np.array([0.0, 0.01], dtype=np.float64),
            )

            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertIn("fluent_p", reader.fieldnames)
        self.assertIn("solver_p", reader.fieldnames)
        self.assertEqual(rows[0]["y"], "0.003")
        self.assertEqual(rows[0]["fluent_p"], "101.0")
        self.assertEqual(rows[0]["solver_p"], "99.0")

    def test_default_fixed_flow_artifact_prefers_core_solver_preflow(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            legacy = tmp_path / "legacy_fixed.npz"
            core = tmp_path / "core_fixed.npz"
            legacy.write_bytes(b"legacy")

            previous_legacy = runner.LEGACY_FIXED_SOLVER_FIELDS
            previous_core = runner.CORE_FIXED_PREFLOW_SOLVER_FIELDS
            try:
                runner.LEGACY_FIXED_SOLVER_FIELDS = legacy
                runner.CORE_FIXED_PREFLOW_SOLVER_FIELDS = core
                self.assertEqual(runner._default_fixed_solver_fields(), legacy)

                core.write_bytes(b"core")
                self.assertEqual(runner._default_fixed_solver_fields(), core)
            finally:
                runner.LEGACY_FIXED_SOLVER_FIELDS = previous_legacy
                runner.CORE_FIXED_PREFLOW_SOLVER_FIELDS = previous_core

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

    def test_structure_comparison_blocks_on_empty_solver_history(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_root = root / "reference"
            output_root = root / "out"
            reference_root.mkdir()
            output_root.mkdir()
            (reference_root / "fsi50_structure_monitor.csv").write_text(
                "step,time_s,monitor_avg_total_col0_col6_m,solid_max_total_col0_col6_m\n"
                "1,0.0005,1.0e-4,1.1e-4\n",
                encoding="utf-8",
            )
            solver_history = root / "empty_solver_structure_history.csv"
            solver_history.write_text(
                "step,time_s,tip_displacement_norm_m,max_displacement_m\n",
                encoding="utf-8",
            )

            result = runner._run_structure_comparison(
                reference_root=reference_root,
                fsi50_solver_structure_history=solver_history,
                step50_matrix=root / "missing_step50_matrix.json",
                output_root=output_root,
            )

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(
                result["blockers"][0]["blocker"],
                "empty_solver_structure_history",
            )
            self.assertTrue(
                (output_root / "fsi50_structure_response_metrics.json").exists()
            )

    def test_runner_supports_fixed_flow_only_scope(self):
        if not (REFERENCE_ROOT / "official_reference_manifest.json").exists():
            self.skipTest("official Fluent reference bundle has not been generated")
        if not OFFICIAL_FIXED_FIELDS.exists():
            self.skipTest("official fixed-flow solver artifact has not been generated")

        runner = _load_runner()
        tmp_parent = ROOT / "tmp"
        tmp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="official_fluent_fixed_flow_only_", dir=tmp_parent
        ) as tmp:
            payload = runner.run(
                reference_root=REFERENCE_ROOT,
                output_root=Path(tmp),
                fixed_solver_fields=OFFICIAL_FIXED_FIELDS,
                fsi50_solver_fields=Path(tmp) / "missing_fsi_fields.npz",
                fsi50_solver_structure_history=Path(tmp) / "missing_structure.csv",
                step50_matrix=Path(tmp) / "missing_step50_matrix.json",
                fixed_flow_only=True,
            )

            self.assertEqual(
                payload["parity_status"]["comparison_scope"],
                "fixed_flow_only",
            )
            self.assertFalse(payload["parity_status"]["fluent_parity_claimed"])
            self.assertEqual(
                payload["parity_status"]["fsi50_final_field_status"],
                "not_run",
            )
            self.assertEqual(
                payload["parity_status"]["fsi50_structure_status"],
                "not_run",
            )
            self.assertTrue(
                (Path(tmp) / "fixed_flow_parity_metrics.json").exists()
            )

    def test_runner_writes_fail_closed_parity_bundle(self):
        if not (REFERENCE_ROOT / "official_reference_manifest.json").exists():
            self.skipTest("official Fluent reference bundle has not been generated")
        if not OFFICIAL_FIXED_FIELDS.exists():
            self.skipTest("official fixed-flow solver artifact has not been generated")
        if not STEP50_MATRIX.exists():
            self.skipTest("selected-formulation Step50 matrix is absent")

        runner = _load_runner()
        tmp_parent = ROOT / "tmp"
        tmp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="official_fluent_2way_parity_", dir=tmp_parent
        ) as tmp:
            payload = runner.run(
                reference_root=REFERENCE_ROOT,
                output_root=Path(tmp),
                fixed_solver_fields=OFFICIAL_FIXED_FIELDS,
                fsi50_solver_fields=FSI50_SOLVER_FIELDS,
                step50_matrix=STEP50_MATRIX,
            )

            self.assertEqual(payload["parity_status"]["fluent_parity_claimed"], False)
            self.assertIn(payload["parity_status"]["status"], {"blocked", "failed"})
            if FSI50_SOLVER_FIELDS.exists():
                self.assertIn(
                    "fsi50_speed_max_rel_error",
                    payload["fsi50_final_field"]["metrics"],
                )
                self.assertNotIn("blockers", payload["fsi50_final_field"])
            else:
                self.assertEqual(payload["parity_status"]["status"], "blocked")
                self.assertEqual(
                    payload["fsi50_final_field"]["blockers"][0]["blocker"],
                    "missing_solver_fsi50_full_field",
                )
            self.assertIn("speed_max_rel_error", payload["fixed_flow"]["metrics"])
            self.assertIn(
                "monitor_displacement_peak_rel_error",
                payload["fsi50_structure"]["metrics"],
            )

            for relative in (
                "official_reference_manifest.json",
                "solver_run_manifest.json",
                "alignment_manifest.json",
                "fixed_flow_parity_metrics.json",
                "fsi50_final_field_parity_metrics.json",
                "fsi50_structure_response_metrics.json",
                "error_localization_report.md",
                "correction_summary.md",
                "parity_status.json",
                "aligned_sampling_points.csv",
                "steady_profile_centerline.csv",
                "steady_profile_throat.csv",
                "solver_structure_response_proxy.csv",
            ):
                self.assertTrue((Path(tmp) / relative).exists(), msg=relative)

            status = json.loads((Path(tmp) / "parity_status.json").read_text())
            self.assertFalse(status["fluent_parity_claimed"])
            alignment = json.loads((Path(tmp) / "alignment_manifest.json").read_text())
            face_bounds = alignment["official_face_zone_node_bounds"]
            self.assertEqual(face_bounds["wall"]["y_min"], 0.0)
            self.assertEqual(face_bounds["wall"]["y_max"], 0.0)
            self.assertEqual(face_bounds["symmetry.2"]["y_min"], 0.02)
            self.assertEqual(face_bounds["symmetry.2"]["y_max"], 0.02)
            self.assertEqual(face_bounds["flap_wall"]["x_min"], 0.05)
            self.assertEqual(face_bounds["flap_wall"]["x_max"], 0.053)
            self.assertEqual(face_bounds["flap_wall"]["y_min"], 0.0)
            self.assertEqual(face_bounds["flap_wall"]["y_max"], 0.01)


if __name__ == "__main__":
    unittest.main()
