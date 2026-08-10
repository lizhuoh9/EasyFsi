from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import csv
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (
    compare_solver_to_fluent_field,
    load_solver_npz,
    physical_solid_mask_from_bounds,
    sample_structured_solver_at_fluent_points,
    save_solver_npz_from_flow_snapshot,
)
from refactored.validation.ansys_vertical_flap_fsi.official_fluent_reference import (
    DEFAULT_OFFICIAL_SOURCE_ROOT,
    import_official_fluent_reference,
    read_fluent_cell_fields,
)
from refactored.validation.ansys_vertical_flap_fixed.solver_diagnostics import (
    metadata_json,
)
from benchmarks.official import solid_mpm_fsi_runner as fsi_runner


def _load_snapshot_runner_module():
    path = (
        ROOT
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "scripts"
        / "run_official_fluent_2way_fsi50_snapshot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_official_fluent_2way_fsi50_snapshot",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load snapshot runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialFluent2WayReferenceImportTests(unittest.TestCase):
    def test_snapshot_runner_loads_official_solid_center_bounds(self):
        runner = _load_snapshot_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            reference_root = Path(tmp)
            mesh_summary = {
                "cell_zone_center_bounds": {
                    "solid.5": {
                        "x_min": 0.050333333333333334,
                        "x_max": 0.0525,
                    }
                }
            }
            np.savez(
                reference_root / "steady_fluent_fields.npz",
                mesh_summary_json=json.dumps(mesh_summary),
            )

            bounds = runner._official_reference_flap_streamwise_bounds(reference_root)

        self.assertEqual(bounds, (0.050333333333333334, 0.0525))

    def test_snapshot_runner_infers_cartesian_grid_from_fluent_mesh_topology(self):
        runner = _load_snapshot_runner_module()
        mesh_summary = {
            "cell_zone_center_bounds": {
                "fluid.4": {"count": 4450},
            },
            "face_zone_node_bounds": {
                "symmetry.2": {
                    "count": 100,
                    "x_min": 0.0,
                    "x_max": 0.1,
                    "y_min": 0.02,
                    "y_max": 0.02,
                },
                "velocity_inlet.1": {
                    "count": 20,
                    "x_min": 0.0,
                    "x_max": 0.0,
                    "y_min": 0.0,
                    "y_max": 0.02,
                },
            },
        }

        self.assertEqual(
            runner._grid_nodes_from_fluent_mesh_summary(mesh_summary),
            (4, 50, 89),
        )

    def test_snapshot_history_exports_hibm_sharp_boundary_diagnostics(self):
        runner = _load_snapshot_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            runner._write_solver_history_csv(
                path,
                [
                    {
                        "flow_solid_boundary_mode": "hibm_sharp_marker_rows",
                        "hibm_sharp_marker_boundary_enabled": True,
                        "hibm_sharp_marker_boundary_search_reused": True,
                        "hibm_sharp_marker_boundary_near_node_count": 31,
                        "hibm_sharp_marker_boundary_external_node_count": 12,
                        "hibm_sharp_marker_boundary_internal_node_count": 19,
                        "hibm_sharp_marker_boundary_internal_obstacle_cell_count": 5,
                        "hibm_sharp_marker_boundary_no_slip_rows": 12,
                        "hibm_sharp_marker_boundary_pressure_neumann_rows": 11,
                        "hibm_sharp_marker_boundary_pressure_gradient_updated": True,
                        "hibm_pressure_neumann_skipped_velocity_dirichlet_count": 7,
                        "hibm_pressure_neumann_invalid_reconstruction_count": 3,
                    }
                ],
            )

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["flow_solid_boundary_mode"], "hibm_sharp_marker_rows")
        self.assertEqual(rows[0]["hibm_sharp_marker_boundary_enabled"], "True")
        self.assertEqual(rows[0]["hibm_sharp_marker_boundary_search_reused"], "True")
        self.assertEqual(rows[0]["hibm_sharp_marker_boundary_near_node_count"], "31")
        self.assertEqual(rows[0]["hibm_sharp_marker_boundary_no_slip_rows"], "12")
        self.assertEqual(
            rows[0]["hibm_sharp_marker_boundary_pressure_neumann_rows"],
            "11",
        )
        self.assertEqual(
            rows[0]["hibm_pressure_neumann_skipped_velocity_dirichlet_count"],
            "7",
        )
        self.assertEqual(
            rows[0]["hibm_pressure_neumann_invalid_reconstruction_count"],
            "3",
        )

    def test_real_official_fluent_reference_import_matches_known_summary(self):
        _require_h5py_or_skip(self)
        if not DEFAULT_OFFICIAL_SOURCE_ROOT.exists():
            self.skipTest(f"missing local Fluent reference: {DEFAULT_OFFICIAL_SOURCE_ROOT}")

        tmp_parent = ROOT / "tmp"
        tmp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="official_fluent_2way_reference_", dir=tmp_parent
        ) as tmp:
            payload = import_official_fluent_reference(
                source_root=DEFAULT_OFFICIAL_SOURCE_ROOT,
                output_root=Path(tmp),
            )

            summary = payload["summary"]
            self.assertAlmostEqual(
                summary["steady"]["speed_max"],
                28.031949251257803,
                places=10,
            )
            self.assertAlmostEqual(
                summary["fsi50_final"]["speed_max"],
                28.241359808842144,
                places=10,
            )
            monitor = summary["structure_monitor"]
            self.assertEqual(monitor["step_count"], 50)
            self.assertEqual(monitor["monitor_displacement_peak_step"], 9)
            self.assertAlmostEqual(
                monitor["monitor_displacement_peak_time_s"],
                0.0045,
                places=12,
            )
            self.assertAlmostEqual(
                monitor["monitor_displacement_peak_m"],
                0.0004009369923131437,
                places=15,
            )
            self.assertAlmostEqual(
                monitor["monitor_final_displacement_m"],
                5.5272183929070876e-05,
                places=16,
            )

            for relative in (
                "official_reference_manifest.json",
                "steady_fluent_fields.npz",
                "fsi50_final_fluent_fields.npz",
                "fsi50_structure_monitor.csv",
                "fluent_hdf5_field_map.json",
                "fluent_reference_summary.json",
            ):
                self.assertTrue((Path(tmp) / relative).exists(), msg=relative)

            with np.load(Path(tmp) / "steady_fluent_fields.npz") as fields:
                mesh_summary = json.loads(str(fields["mesh_summary_json"]))
            solid_bounds = mesh_summary["cell_zone_center_bounds"]["solid.5"]
            self.assertEqual(solid_bounds["count"], 30)
            self.assertAlmostEqual(solid_bounds["x_min"], 0.050333333333333334)
            self.assertAlmostEqual(solid_bounds["x_max"], 0.0525)
            self.assertAlmostEqual(solid_bounds["y_min"], 0.0003333333333333333)
            self.assertAlmostEqual(solid_bounds["y_max"], 0.0095)

    def test_missing_fluent_hdf5_field_fails_closed(self):
        h5py = _require_h5py_or_skip(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "case.h5"
            data_path = root / "data.h5"
            _write_minimal_case_hdf5(h5py, case_path)
            with h5py.File(data_path, "w") as data_file:
                data_file.create_dataset(
                    "results/1/phase-1/cells/SV_V/1",
                    data=np.array([0.0], dtype=np.float64),
                )
                data_file.create_dataset(
                    "results/1/phase-1/cells/SV_P/1",
                    data=np.array([0.0], dtype=np.float64),
                )

            with self.assertRaisesRegex(KeyError, "SV_U"):
                read_fluent_cell_fields(case_path, data_path)

    def test_import_records_face_zone_node_bounds(self):
        h5py = _require_h5py_or_skip(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "case.h5"
            data_path = root / "data.h5"
            _write_minimal_case_hdf5(h5py, case_path)
            with h5py.File(data_path, "w") as data_file:
                data_file.create_dataset(
                    "results/1/phase-1/cells/SV_U/1",
                    data=np.array([1.0], dtype=np.float64),
                )
                data_file.create_dataset(
                    "results/1/phase-1/cells/SV_V/1",
                    data=np.array([0.0], dtype=np.float64),
                )
                data_file.create_dataset(
                    "results/1/phase-1/cells/SV_P/1",
                    data=np.array([2.0], dtype=np.float64),
                )

            bundle = read_fluent_cell_fields(case_path, data_path)

            face_bounds = bundle.mesh_summary["face_zone_node_bounds"]
            self.assertIn("default-interior", face_bounds)
            self.assertEqual(face_bounds["default-interior"]["count"], 4)
            self.assertAlmostEqual(face_bounds["default-interior"]["x_min"], 0.0)
            self.assertAlmostEqual(face_bounds["default-interior"]["x_max"], 1.0)
            self.assertAlmostEqual(face_bounds["default-interior"]["y_min"], 0.0)
            self.assertAlmostEqual(face_bounds["default-interior"]["y_max"], 1.0)

    def test_five_percent_field_gate_passes_and_fails_deterministically(self):
        solver = _structured_solver(scale=1.0)
        fluent = _fluent_points()
        passed = compare_solver_to_fluent_field(
            solver,
            fluent,
            throat_x=0.0,
            downstream_x=(0.0, 1.0),
        )
        self.assertEqual(passed["status"], "passed", msg=passed)
        self.assertLessEqual(passed["metrics"]["u_max_rel_error"], 0.05)

        failed = compare_solver_to_fluent_field(
            _structured_solver(scale=1.10),
            fluent,
            throat_x=0.0,
            downstream_x=(0.0, 1.0),
        )
        self.assertEqual(failed["status"], "failed", msg=failed)
        self.assertGreater(failed["metrics"]["u_max_rel_error"], 0.05)

    def test_downstream_near_wall_backflow_diagnostics_are_diagnostic_only(self):
        solver = _structured_backflow_solver(near_wall_sign=1.0)
        fluent = _structured_backflow_fluent()

        result = compare_solver_to_fluent_field(
            solver,
            fluent,
            throat_x=0.06,
            downstream_x=(0.06, 0.09),
        )

        self.assertEqual(result["status"], "passed", msg=result)
        gate = result["gates"]["downstream_near_wall_backflow_fraction_abs_error"]
        self.assertEqual(gate["status"], "diagnostic")
        self.assertGreater(gate["value"], 0.5)
        backflow = result["diagnostics"]["downstream_near_wall_backflow"]
        self.assertGreater(backflow["sample_count"], 0)
        self.assertGreater(backflow["reference_negative_u_fraction"], 0.5)
        self.assertEqual(backflow["solver_negative_u_fraction"], 0.0)

    def test_field_comparison_reports_pressure_extrema_sample_pairs(self):
        solver = _structured_solver(scale=1.0)
        fluent = _fluent_points()
        solver["p"] = np.array([[0.0, 3.0], [0.0, 3.0]], dtype=np.float64)
        fluent["p"] = np.array([2.0, -5.0, 1.0, 4.0], dtype=np.float64)

        result = compare_solver_to_fluent_field(
            solver,
            fluent,
            throat_x=0.0,
            downstream_x=(0.0, 1.0),
        )

        extrema = result["diagnostics"]["pressure_extrema"]
        reference_min = extrema["reference_pressure_min_point"]
        self.assertEqual(reference_min["sample_index"], 1)
        self.assertEqual(reference_min["x_m"], 1.0)
        self.assertEqual(reference_min["y_m"], 0.0)
        self.assertEqual(reference_min["reference_pressure_pa"], -5.0)
        self.assertEqual(reference_min["solver_pressure_pa"], 3.0)
        self.assertEqual(reference_min["reference_u_mps"], 2.0)
        self.assertEqual(reference_min["solver_u_mps"], 2.0)

    def test_sampling_keeps_fluent_fluid_points_adjacent_to_solver_obstacles(self):
        solver = {
            "s": np.array([0.0, 1.0], dtype=np.float64),
            "y": np.array([0.0, 1.0], dtype=np.float64),
            "u": np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float64),
            "v": np.zeros((2, 2), dtype=np.float64),
            "p": np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float64),
            "speed": np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float64),
            "fluid_mask": np.array([[False, True], [True, True]], dtype=bool),
        }
        fluent = {
            "x": np.array([0.25], dtype=np.float64),
            "y": np.array([0.25], dtype=np.float64),
            "u": np.array([1.0], dtype=np.float64),
            "v": np.array([0.0], dtype=np.float64),
            "p": np.array([1.0], dtype=np.float64),
            "speed": np.array([1.0], dtype=np.float64),
            "cell_ids": np.array([1], dtype=np.int64),
        }

        samples = sample_structured_solver_at_fluent_points(solver, fluent)

        self.assertTrue(bool(samples["valid"][0]))
        self.assertGreater(float(samples["fluid_mask"][0]), 0.0)
        self.assertLess(float(samples["fluid_mask"][0]), 1.0)
        expected = (0.25 * 0.75 * 10.0 + 0.75 * 0.25 * 20.0 + 0.25 * 0.25 * 30.0) / (
            0.25 * 0.75 + 0.75 * 0.25 + 0.25 * 0.25
        )
        self.assertAlmostEqual(float(samples["u"][0]), expected)

    def test_metadata_json_is_numpy_2_compatible_string_scalar(self):
        value = metadata_json({"source": "official-fluent-test"})

        self.assertEqual(value.shape, ())
        self.assertIn("official-fluent-test", str(value))

    def test_flow_snapshot_export_uses_official_axis_and_sign_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot_fields.npz"
            summary = save_solver_npz_from_flow_snapshot(path, _flow_snapshot_fixture())
            fields = load_solver_npz(path)

            self.assertEqual(summary["shape"], [2, 3])
            np.testing.assert_allclose(fields["s"], [0.25, 0.75, 1.25])
            np.testing.assert_allclose(fields["y"], [0.0, 0.02])
            np.testing.assert_allclose(fields["u"][0], [30.0, 25.0, 15.0])
            np.testing.assert_allclose(fields["v"][fields["fluid_mask"]], 2.0)
            self.assertFalse(fields["fluid_mask"][1, 0])
            self.assertTrue(fields["fluid_mask"][0, 0])

    def test_official_flap_physical_solid_mask_uses_geometry_bounds(self):
        s = np.arange(64, dtype=np.float64) * (0.10 / 64.0)
        s[32] = np.float32(0.050)
        y = (np.arange(32, dtype=np.float64) + 0.5) * (0.02 / 32.0)

        solid_mask = physical_solid_mask_from_bounds(
            s,
            y,
            {
                "streamwise_min_m": 0.050,
                "streamwise_max_m": 0.053,
                "y_min_m": 0.0,
                "y_max_m": 0.010,
            },
        )

        rows, cols = np.where(solid_mask)
        self.assertEqual(sorted(set(cols.tolist())), [32, 33])
        self.assertEqual(min(rows), 0)
        self.assertEqual(max(rows), 15)
        self.assertAlmostEqual(float(s[cols.min()]), 0.050)
        self.assertLess(float(s[cols.max()]), 0.053)

    def test_flow_snapshot_export_separates_physical_solid_from_numeric_obstacle_mask(self):
        snapshot = _flow_snapshot_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot_fields.npz"
            summary = save_solver_npz_from_flow_snapshot(
                path,
                snapshot,
                physical_solid_bounds={
                    "streamwise_min_m": 0.25,
                    "streamwise_max_m": 0.76,
                    "y_min_m": 0.0,
                    "y_max_m": 0.021,
                },
            )
            fields = load_solver_npz(path)

        self.assertIn("physical_solid_bounds", summary)
        self.assertEqual(summary["solid_cell_count"], 4)
        np.testing.assert_array_equal(
            fields["solid_mask"],
            np.array([[True, True, False], [True, True, False]], dtype=bool),
        )
        self.assertFalse(fields["fluid_mask"][1, 0])

    def test_flow_snapshot_export_excludes_velocity_dirichlet_surrogate_rows(self):
        snapshot = _flow_snapshot_fixture()
        active = np.zeros((2, 2, 3), dtype=np.int32)
        weights = np.zeros((2, 2, 3), dtype=np.float32)
        active[:, 0, 1] = 1
        weights[:, 0, 1] = 1.0
        snapshot["velocity_dirichlet_boundary_active"] = active
        snapshot["velocity_dirichlet_boundary_projection_weight"] = weights

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot_fields.npz"
            summary = save_solver_npz_from_flow_snapshot(path, snapshot)
            fields = dict(np.load(path))

            self.assertEqual(summary["boundary_surrogate_cell_count"], 1)
            self.assertTrue(summary["exclude_velocity_dirichlet_rows"])
            self.assertFalse(bool(fields["fluid_mask"][0, 1]))
            self.assertTrue(bool(fields["boundary_surrogate_mask"][0, 1]))
            # Rendering must show the solved boundary-row value instead of a
            # white comparison-mask halo, while the strict parity mask above
            # continues to exclude that row from numerical metrics.
            self.assertTrue(bool(fields["display_fluid_mask"][0, 1]))
            self.assertFalse(bool(fields["display_obstacle_mask"][0, 1]))
            # The true obstacle in the fixture remains hidden/solid after the
            # streamwise reversal (original k=2 -> display column 0).
            self.assertFalse(bool(fields["display_fluid_mask"][1, 0]))
            self.assertTrue(bool(fields["display_obstacle_mask"][1, 0]))

    def test_pressure_range_metric_is_diagnostic_not_acceptance_gate(self):
        solver = _structured_solver(scale=1.0)
        fluent = _fluent_points()
        solver["p"] = np.array([[0.0, 100.0], [0.0, 100.0]], dtype=np.float64)
        fluent["p"] = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)

        result = compare_solver_to_fluent_field(solver, fluent)

        self.assertEqual(
            result["gates"]["pressure_range_rel_error"]["status"],
            "diagnostic",
        )
        self.assertEqual(result["status"], "passed")

    def test_display_mask_does_not_change_parity_metrics_or_samples(self):
        solver = _structured_solver(scale=1.0)
        fluent = _fluent_points()
        baseline_samples = sample_structured_solver_at_fluent_points(
            solver,
            fluent,
        )
        baseline_metrics = compare_solver_to_fluent_field(solver, fluent)

        solver_with_display_override = dict(solver)
        solver_with_display_override["display_fluid_mask"] = np.zeros_like(
            solver["fluid_mask"],
            dtype=bool,
        )
        overridden_samples = sample_structured_solver_at_fluent_points(
            solver_with_display_override,
            fluent,
        )
        overridden_metrics = compare_solver_to_fluent_field(
            solver_with_display_override,
            fluent,
        )

        for key in ("u", "v", "p"):
            np.testing.assert_array_equal(
                overridden_samples[key],
                baseline_samples[key],
            )
        np.testing.assert_array_equal(
            overridden_samples["valid"],
            baseline_samples["valid"],
        )
        self.assertEqual(overridden_metrics, baseline_metrics)

    def test_flow_snapshot_export_rejects_nonfinite_candidate_fields(self):
        snapshot = _flow_snapshot_fixture()
        velocity = np.array(snapshot["velocity"], copy=True)
        velocity[0, 0, 0, 2] = np.nan
        snapshot["velocity"] = velocity

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_snapshot_fields.npz"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                save_solver_npz_from_flow_snapshot(path, snapshot)
            self.assertFalse(path.exists())

    def test_sustained_inlet_predictor_projects_each_physical_substep(self):
        fluid = _FakePredictorFluid()
        projection_calls: list[tuple[float, bool]] = []

        def record_projection(
            fluid: _FakePredictorFluid,
            config: object,
            *,
            reset_pressure: bool,
            **projection_options: object,
        ) -> dict[str, object]:
            projection_calls.append((float(config.dt_s), bool(reset_pressure)))
            return _fake_project_current_flow(
                fluid,
                config,
                reset_pressure=reset_pressure,
                **projection_options,
            )

        config = SimpleNamespace(
            dt_s=5.0e-4,
            grid_nodes=(2, 2, 3),
            inlet_velocity_mps=10.0,
            flow_driver_mode="sustained_inlet_predictor",
            flow_inlet_source_strength=0.8,
            flow_inlet_source_profile="constant",
            flow_inlet_source_ramp_steps=0,
            flow_inlet_source_schedule_scope="global",
            flow_advection_scheme="rk2",
            flow_predictor_substeps=2,
        )
        original_project = fsi_runner._project_current_flow
        try:
            fsi_runner._project_current_flow = record_projection
            report = fsi_runner._flow_advance_current_step(
                fluid,
                config,
                flow_phase="fsi",
                step_index_local=3,
                step_index_global=3,
                preflow_history=[],
                reset_pressure=True,
            )
        finally:
            fsi_runner._project_current_flow = original_project

        self.assertEqual(
            fluid.calls,
            [
                "clear_volume_source",
                "add_zmax_velocity_inlet_volume_source:-8.0",
                "apply_velocity_dirichlet_boundary_rows",
                "predict:0.00025:rk2",
                "apply_velocity_dirichlet_boundary_rows",
                "project",
                "clear_volume_source",
                "add_zmax_velocity_inlet_volume_source:-8.0",
                "apply_velocity_dirichlet_boundary_rows",
                "predict:0.00025:rk2",
                "apply_velocity_dirichlet_boundary_rows",
                "project",
            ],
        )
        self.assertTrue(report["flow_predictor_applied"])
        self.assertEqual(
            projection_calls,
            [(2.5e-4, True), (2.5e-4, False)],
        )
        self.assertIn("advection_scheme=rk2", report["flow_predictor_note"])
        self.assertEqual(report["flow_predictor_projection_segment_count"], 2)
        self.assertAlmostEqual(
            report["flow_predictor_projection_segment_dt_s"],
            2.5e-4,
        )

    def test_sustained_boundary_predictor_does_not_inject_volume_source(self):
        fluid = _FakePredictorFluid()
        config = SimpleNamespace(
            dt_s=5.0e-4,
            grid_nodes=(2, 2, 3),
            inlet_velocity_mps=10.0,
            flow_driver_mode="sustained_boundary_predictor",
            flow_inlet_source_strength=0.8,
            flow_inlet_source_profile="constant",
            flow_inlet_source_ramp_steps=0,
            flow_inlet_source_schedule_scope="global",
            flow_advection_scheme="rk2",
            flow_predictor_substeps=2,
        )
        original_project = fsi_runner._project_current_flow
        try:
            fsi_runner._project_current_flow = _fake_project_current_flow
            report = fsi_runner._flow_advance_current_step(
                fluid,
                config,
                flow_phase="fsi",
                step_index_local=3,
                step_index_global=3,
                preflow_history=[],
                reset_pressure=False,
            )
        finally:
            fsi_runner._project_current_flow = original_project

        self.assertEqual(
            fluid.calls,
            [
                "clear_volume_source",
                "apply_velocity_dirichlet_boundary_rows",
                "predict:0.00025:rk2",
                "apply_velocity_dirichlet_boundary_rows",
                "project",
                "clear_volume_source",
                "apply_velocity_dirichlet_boundary_rows",
                "predict:0.00025:rk2",
                "apply_velocity_dirichlet_boundary_rows",
                "project",
            ],
        )
        self.assertTrue(report["flow_predictor_applied"])
        self.assertFalse(report["flow_volume_source_applied"])
        self.assertEqual(report["flow_inlet_source_factor"], 0.0)
        self.assertEqual(report["flow_inlet_source_normal_velocity_mps"], 0.0)
        self.assertEqual(report["flow_predictor_projection_segment_count"], 2)

    def test_interleaved_segment_report_preserves_main_projection_and_topology_truth(self):
        combined = fsi_runner._combine_interleaved_flow_predictor_segment_reports(
            [
                {
                    "projection_report": {
                        "pre_projection_l2": 1.0,
                        "pre_projection_max_abs": 2.0,
                        "projection_l2": 0.1,
                        "projection_max_abs": 0.2,
                        "cg_iterations_total": 3,
                    },
                    "flow_main_projection_pre_projection_l2": 11.0,
                    "flow_main_projection_pre_projection_max_abs": 12.0,
                    "flow_main_projection_l2": 0.11,
                    "flow_main_projection_max_abs": 0.12,
                    "hibm_preassembly_topology_mutated": True,
                    "hibm_post_dirichlet_consistency_projection_count": 2,
                },
                {
                    "projection_report": {
                        "pre_projection_l2": 4.0,
                        "pre_projection_max_abs": 5.0,
                        "projection_l2": 0.4,
                        "projection_max_abs": 0.5,
                        "cg_iterations_total": 7,
                    },
                    "flow_main_projection_pre_projection_l2": 21.0,
                    "flow_main_projection_pre_projection_max_abs": 22.0,
                    "flow_main_projection_l2": 0.21,
                    "flow_main_projection_max_abs": 0.22,
                    "hibm_preassembly_topology_mutated": False,
                    "hibm_post_dirichlet_consistency_projection_count": 1,
                },
            ],
            configured_substeps=2,
            segment_dt_s=2.5e-4,
            reset_pressure=True,
        )

        self.assertTrue(combined["hibm_preassembly_topology_mutated"])
        self.assertEqual(
            combined["hibm_post_dirichlet_consistency_projection_count"],
            3,
        )
        self.assertEqual(
            combined["projection_report"]["cg_iterations_total"],
            10,
        )
        self.assertEqual(
            combined["flow_predictor_projection_segment_pre_projection_l2_max"],
            21.0,
        )
        self.assertEqual(
            combined["flow_predictor_projection_segment_trace"],
            [
                {
                    "segment_index": 1,
                    "pre_projection_l2": 11.0,
                    "pre_projection_max_abs": 12.0,
                    "projection_l2": 0.11,
                    "projection_max_abs": 0.12,
                    "cg_converged_all": True,
                    "cg_iterations_total": 3,
                },
                {
                    "segment_index": 2,
                    "pre_projection_l2": 21.0,
                    "pre_projection_max_abs": 22.0,
                    "projection_l2": 0.21,
                    "projection_max_abs": 0.22,
                    "cg_converged_all": True,
                    "cg_iterations_total": 7,
                },
            ],
        )

    def test_nested_projection_report_preserves_existing_all_flags(self):
        combined = fsi_runner._combine_flow_projection_reports(
            [
                {
                    "pre_projection_velocity_projector_prepared": True,
                    "pre_projection_velocity_projector_prepared_all": False,
                    "pre_projection_velocity_projector_converged": True,
                    "pre_projection_velocity_projector_converged_all": False,
                    "pre_projection_velocity_projector_committed": True,
                    "pre_projection_velocity_projector_committed_all": False,
                },
                {
                    "pre_projection_velocity_projector_prepared": True,
                    "pre_projection_velocity_projector_prepared_all": True,
                    "pre_projection_velocity_projector_converged": True,
                    "pre_projection_velocity_projector_converged_all": True,
                    "pre_projection_velocity_projector_committed": True,
                    "pre_projection_velocity_projector_committed_all": True,
                },
            ]
        )

        self.assertFalse(
            combined["pre_projection_velocity_projector_prepared_all"]
        )
        self.assertFalse(
            combined["pre_projection_velocity_projector_converged_all"]
        )
        self.assertFalse(
            combined["pre_projection_velocity_projector_committed_all"]
        )

    def test_marker_feedback_populates_velocity_constraint_fields(self):
        fluid = _FakePredictorFluid(shape=(2, 2, 2))
        markers = _FakeMarkers(
            positions=np.array(
                [
                    [0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25],
                    [0.75, 0.25, 0.75],
                ],
                dtype=np.float64,
            ),
            velocities=np.array(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0],
                ],
                dtype=np.float64,
            ),
            region_ids=np.array(
                [
                    fsi_runner.PRIMARY_REGION_ID,
                    fsi_runner.PRIMARY_REGION_ID,
                    fsi_runner.SECONDARY_REGION_ID,
                ],
                dtype=np.int32,
            ),
        )
        config = SimpleNamespace(
            grid_nodes=(2, 2, 2),
            span_m=1.0,
            duct_height_m=2.0,
            duct_length_m=1.0,
            preserve_marker_velocity_constraints=True,
        )

        report = fsi_runner._apply_marker_feedback_to_fluid(
            markers,
            fluid,
            config,
            feedback_available=True,
            previous_feedback_constraint_cells=set(),
        )

        self.assertTrue(report["fluid_marker_velocity_constraints_enabled"])
        self.assertEqual(report["fluid_marker_velocity_constraint_active_cell_count"], 2)
        self.assertEqual(fluid.calls, ["clear_velocity_constraints"])
        np.testing.assert_allclose(
            fluid.velocity_constraint_sum.value[0, 0, 0],
            [0.0, 4.0, 6.0],
        )
        self.assertEqual(float(fluid.velocity_constraint_weight.value[0, 0, 0]), 2.0)
        np.testing.assert_allclose(
            fluid.velocity_constraint_primary_sum.value[0, 0, 0],
            [0.0, 4.0, 6.0],
        )
        self.assertEqual(
            float(fluid.velocity_constraint_primary_weight.value[0, 0, 0]),
            2.0,
        )
        np.testing.assert_allclose(
            fluid.velocity_constraint_secondary_sum.value[1, 0, 1],
            [5.0, 6.0, 7.0],
        )
        self.assertEqual(
            float(fluid.velocity_constraint_secondary_weight.value[1, 0, 1]),
            1.0,
        )

    def test_project_current_flow_preserves_marker_velocity_constraints(self):
        fluid = _FakePredictorFluid(shape=(2, 2, 2))
        config = SimpleNamespace(
            dt_s=5.0e-4,
            flow_projection_iterations=7,
            flow_pressure_outlet_enabled=True,
            flow_pressure_outlet_backflow_policy="allow",
            flow_obstacle_normal_velocity_policy="cell_zero_only",
            preserve_marker_velocity_constraints=True,
            marker_velocity_constraint_blend=0.25,
            marker_velocity_constraint_solid_mobility_ratio=2.0,
            flow_pressure_solver="fv_cg",
            flow_cg_tolerance=2.5e-7,
            flow_divergence_cleanup_iterations=3,
            flow_hibm_tiny_unreached_cleanup_component_cells=128,
        )

        report = fsi_runner._project_current_flow(
            fluid,
            config,
            reset_pressure=True,
        )

        self.assertEqual(fluid.project_kwargs["iterations"], 7)
        self.assertEqual(fluid.project_kwargs["pressure_solver"], "fv_cg")
        self.assertEqual(fluid.project_kwargs["cg_tolerance"], 2.5e-7)
        self.assertEqual(fluid.project_kwargs["divergence_cleanup_iterations"], 3)
        self.assertEqual(
            fluid.project_kwargs["hibm_tiny_unreached_cleanup_component_cells"],
            128,
        )
        self.assertEqual(fluid.project_kwargs["pressure_outlet_backflow_policy"], "allow")
        self.assertEqual(
            fluid.project_kwargs["obstacle_normal_velocity_policy"],
            "cell_zero_only",
        )
        self.assertTrue(fluid.project_kwargs["preserve_velocity_constraints"])
        self.assertEqual(fluid.project_kwargs["velocity_constraint_blend"], 0.25)
        self.assertEqual(
            fluid.project_kwargs["velocity_constraint_solid_mobility_ratio"],
            2.0,
        )
        self.assertTrue(fluid.project_kwargs["reset_pressure"])
        self.assertTrue(report["projection_report"]["preserve_velocity_constraints"])


def _require_h5py_or_skip(case: unittest.TestCase):
    try:
        import h5py
    except ImportError:
        case.skipTest("h5py is not installed in this interpreter")
    return h5py


def _write_minimal_case_hdf5(h5py, path: Path) -> None:
    with h5py.File(path, "w") as case_file:
        case_file.create_dataset(
            "meshes/1/nodes/coords/8",
            data=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        case_file.create_dataset(
            "meshes/1/cells/zoneTopology/name",
            data=np.array([b"fluid.4"]),
        )
        case_file.create_dataset("meshes/1/cells/zoneTopology/id", data=np.array([2]))
        case_file.create_dataset(
            "meshes/1/cells/zoneTopology/minId", data=np.array([1])
        )
        case_file.create_dataset(
            "meshes/1/cells/zoneTopology/maxId", data=np.array([1])
        )
        case_file.create_dataset(
            "meshes/1/cells/zoneTopology/zoneType", data=np.array([1])
        )
        case_file.create_dataset(
            "meshes/1/faces/zoneTopology/name",
            data=np.array([b"default-interior"]),
        )
        case_file.create_dataset("meshes/1/faces/zoneTopology/id", data=np.array([11]))
        case_file.create_dataset(
            "meshes/1/faces/zoneTopology/minId", data=np.array([1])
        )
        case_file.create_dataset(
            "meshes/1/faces/zoneTopology/maxId", data=np.array([4])
        )
        case_file.create_dataset(
            "meshes/1/faces/zoneTopology/zoneType", data=np.array([2])
        )
        case_file.create_dataset(
            "meshes/1/faces/zoneTopology/faceType", data=np.array([2])
        )
        case_file.create_dataset(
            "meshes/1/faces/nodes/1/nodes",
            data=np.array([1, 2, 2, 3, 3, 4, 4, 1], dtype=np.uint32),
        )
        case_file.create_dataset(
            "meshes/1/faces/c0/1",
            data=np.array([1, 1, 1, 1], dtype=np.uint32),
        )
        case_file.create_dataset(
            "meshes/1/faces/c1/1",
            data=np.array([], dtype=np.uint32),
        )


def _structured_solver(scale: float) -> dict[str, np.ndarray]:
    u = scale * np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    v = np.zeros_like(u)
    return {
        "s": np.array([0.0, 1.0], dtype=np.float64),
        "y": np.array([0.0, 1.0], dtype=np.float64),
        "u": u,
        "v": v,
        "p": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        "speed": np.hypot(u, v),
        "fluid_mask": np.ones_like(u, dtype=bool),
    }


def _fluent_points() -> dict[str, np.ndarray]:
    u = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64)
    v = np.zeros_like(u)
    return {
        "x": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        "y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        "u": u,
        "v": v,
        "p": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        "speed": np.hypot(u, v),
        "cell_ids": np.array([1, 2, 3, 4], dtype=np.int64),
    }


def _structured_backflow_solver(*, near_wall_sign: float) -> dict[str, np.ndarray]:
    u = np.array(
        [
            [1.0, near_wall_sign * 0.01, near_wall_sign * 0.01],
            [1.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    v = np.zeros_like(u)
    return {
        "s": np.array([0.0, 0.06, 0.09], dtype=np.float64),
        "y": np.array([0.0, 1.0], dtype=np.float64),
        "u": u,
        "v": v,
        "p": np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64),
        "speed": np.hypot(u, v),
        "fluid_mask": np.ones_like(u, dtype=bool),
    }


def _structured_backflow_fluent() -> dict[str, np.ndarray]:
    u = np.array([1.0, -0.01, -0.01, 1.0, 2.0, 2.0], dtype=np.float64)
    v = np.zeros_like(u)
    return {
        "x": np.array([0.0, 0.06, 0.09, 0.0, 0.06, 0.09], dtype=np.float64),
        "y": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64),
        "u": u,
        "v": v,
        "p": np.array([2.0, 1.0, 0.0, 2.0, 1.0, 0.0], dtype=np.float64),
        "speed": np.hypot(u, v),
        "cell_ids": np.arange(1, 7, dtype=np.int64),
    }


def _flow_snapshot_fixture() -> dict[str, np.ndarray]:
    velocity = np.zeros((2, 2, 3, 3), dtype=np.float64)
    velocity[:, :, :, 1] = 2.0
    velocity[:, :, 0, 2] = -10.0
    velocity[:, :, 1, 2] = -20.0
    velocity[:, :, 2, 2] = -30.0
    obstacle = np.zeros((2, 2, 3), dtype=np.int32)
    obstacle[:, 1, 2] = 1
    center_y = np.zeros((2, 2, 3), dtype=np.float64)
    center_z = np.zeros((2, 2, 3), dtype=np.float64)
    center_y[:, 0, :] = 0.0
    center_y[:, 1, :] = 0.02
    center_z[:, :, 0] = 0.0
    center_z[:, :, 1] = 0.5
    center_z[:, :, 2] = 1.0
    return {
        "pressure": np.ones((2, 2, 3), dtype=np.float64),
        "velocity": velocity,
        "obstacle": obstacle,
        "cell_center_y_m": center_y,
        "cell_center_z_m": center_z,
    }


class _FakeField:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.array(value, copy=True)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.value, copy=True)

    def from_numpy(self, value: np.ndarray) -> None:
        self.value = np.array(value, copy=True)


class _FakeMarkers:
    def __init__(
        self,
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        region_ids: np.ndarray,
    ) -> None:
        self.marker_count = int(len(positions))
        self.x_gamma_m = _FakeField(positions)
        self.v_gamma_mps = _FakeField(velocities)
        self.region_id = _FakeField(region_ids)


class _FakePredictorFluid:
    def __init__(self, shape: tuple[int, int, int] = (2, 2, 3)) -> None:
        self.calls: list[str] = []
        self.obstacle = _FakeField(np.zeros(shape, dtype=np.int32))
        self.velocity = _FakeField(np.zeros((*shape, 3), dtype=np.float32))
        self.pressure = _FakeField(np.zeros(shape, dtype=np.float32))
        self.velocity_dirichlet_boundary_active = _FakeField(
            np.zeros(shape, dtype=np.int32)
        )
        self.velocity_dirichlet_boundary_value_mps = _FakeField(
            np.zeros((*shape, 3), dtype=np.float32)
        )
        self.velocity_dirichlet_boundary_projection_weight = _FakeField(
            np.zeros(shape, dtype=np.float32)
        )
        self.velocity_dirichlet_boundary_enforcement_weight = _FakeField(
            np.zeros(shape, dtype=np.float32)
        )
        self.velocity_dirichlet_boundary_marker_region_id = _FakeField(
            -np.ones(shape, dtype=np.int32)
        )
        self.velocity_dirichlet_boundary_hard_fixed_component_mask = _FakeField(
            np.zeros(shape, dtype=np.int32)
        )
        self.velocity_dirichlet_boundary_external_exact_component_mask = _FakeField(
            np.zeros(shape, dtype=np.int32)
        )
        self.velocity_dirichlet_boundary_owned_row = _FakeField(
            np.zeros(shape, dtype=np.int32)
        )
        self.velocity_constraint_sum = _FakeField(
            np.zeros((*shape, 3), dtype=np.float32)
        )
        self.velocity_constraint_weight = _FakeField(np.zeros(shape, dtype=np.float32))
        self.velocity_constraint_primary_sum = _FakeField(
            np.zeros((*shape, 3), dtype=np.float32)
        )
        self.velocity_constraint_primary_weight = _FakeField(
            np.zeros(shape, dtype=np.float32)
        )
        self.velocity_constraint_secondary_sum = _FakeField(
            np.zeros((*shape, 3), dtype=np.float32)
        )
        self.velocity_constraint_secondary_weight = _FakeField(
            np.zeros(shape, dtype=np.float32)
        )
        self.project_kwargs: dict[str, object] = {}

    def clear_velocity_constraints(self) -> None:
        self.calls.append("clear_velocity_constraints")
        for field in (
            self.velocity_constraint_sum,
            self.velocity_constraint_weight,
            self.velocity_constraint_primary_sum,
            self.velocity_constraint_primary_weight,
            self.velocity_constraint_secondary_sum,
            self.velocity_constraint_secondary_weight,
        ):
            field.from_numpy(np.zeros_like(field.value))

    def clear_volume_source(self) -> None:
        self.calls.append("clear_volume_source")

    def add_zmax_velocity_inlet_volume_source(
        self,
        *,
        normal_velocity_mps: float,
    ) -> None:
        self.calls.append(
            f"add_zmax_velocity_inlet_volume_source:{normal_velocity_mps}"
        )

    def apply_velocity_dirichlet_boundary_rows(
        self,
        *,
        read_report: bool = True,
    ) -> None:
        self.calls.append("apply_velocity_dirichlet_boundary_rows")
        self._last_boundary_read_report = bool(read_report)

    def predict(
        self,
        dt_s: float,
        *,
        advection_scheme: str,
        kinematic_viscosity_m2_s: float | None = None,
        no_slip_domain_walls: tuple[bool, bool, bool, bool, bool, bool] | None = None,
    ) -> None:
        self.calls.append(f"predict:{dt_s}:{advection_scheme}")

    def project(self, **kwargs: object) -> dict[str, object]:
        self.calls.append("project")
        self.project_kwargs = dict(kwargs)
        return {
            "preserve_velocity_constraints": bool(
                kwargs.get("preserve_velocity_constraints", False)
            ),
            "velocity_constraint_active_cells": 0,
            "velocity_constraint_max_delta_mps": 0.0,
            "velocity_constraint_mean_delta_mps": 0.0,
        }

    def pressure_outlet_fv_flux_report(self, *, dt_s: float) -> dict[str, object]:
        self.last_flux_dt_s = float(dt_s)
        return {}

    def snapshot_pressure(self, *, preserve_if_current_is_zero: bool) -> bool:
        self.last_snapshot_pressure_preserve = bool(preserve_if_current_is_zero)
        return True


def _fake_project_current_flow(
    fluid: _FakePredictorFluid,
    config: object,
    *,
    reset_pressure: bool,
    pressure_solve_context: object | None = None,
    **projection_options: object,
) -> dict[str, object]:
    del config, reset_pressure, pressure_solve_context, projection_options
    fluid.calls.append("project")
    return {
        "local_velocity_peak_mps": 0.0,
        "fluid_speed_p99_mps": 0.0,
        "fluid_speed_p999_mps": 0.0,
        "pressure_min_pa": 0.0,
        "pressure_max_pa": 0.0,
        "projection_report": {},
    }
