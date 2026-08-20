from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.refactored.validation.ansys_vertical_flap_fsi.native_fine_comparison import (
    NativeFineComparisonError,
    compare_interface_force_histories,
    evaluate_five_percent_diagnostic_gate,
    postprocess_native_fine_comparison,
)
from src.refactored.validation.ansys_vertical_flap_fsi.native_fine_rendering import (
    _display_fluid_mask,
    _overlay_deformed_solid,
)


FLUENT_CHECKSUM_RELATIVE_PATHS = [
    "summary.json",
    "input_manifest.json",
    "fields/final_fields.npz",
    "histories/pressure_history.csv",
    "histories/residual_history.csv",
    "histories/residual_snapshot_summary.csv",
    "histories/structure_displacement_history.csv",
    "histories/velocity_history.csv",
]


def _five_percent_metric_fixture(*, field_error: float) -> tuple[dict, dict, dict]:
    field_comparison = {
        "direct_errors": {
            key: {"nrmse_by_reference_max_abs": field_error}
            for key in ("u", "v", "speed", "p")
        },
        "pressure_reference_diagnostic": {
            "zero_mean_pressure_difference_rmse_pa": 4.0,
            "raw_pressure_difference_rmse_pa": 4.0,
            "sampled_mean_offset_our_minus_fluent_pa": 0.0,
            "native_fluent_sampled_pressure_range_pa": 100.0,
        },
    }
    series = {
        "nrmse_by_reference_peak": 0.04,
        "our_peak_m": 1.04,
        "fluent_peak_m": 1.0,
        "our_peak_step": 21,
        "fluent_peak_step": 20,
        "our_final_m": 0.52,
        "fluent_final_m": 0.50,
    }
    displacement_comparison = {
        "tip_streamwise": dict(series),
        "tip_transverse": dict(series),
        "tip_mean_vector": dict(series),
        "solid_max": dict(series),
    }
    force_series = {
        "nrmse_by_reference_peak": 0.04,
        "our_peak_n_per_m": 1.04,
        "fluent_peak_n_per_m": 1.0,
        "our_peak_step": 21,
        "fluent_peak_step": 20,
        "our_final_n_per_m": 0.52,
        "fluent_final_n_per_m": 0.50,
    }
    force_comparison = {
        "streamwise": dict(force_series),
        "transverse": dict(force_series),
        "out_of_plane_leakage_by_reference_streamwise_peak": 0.01,
    }
    return field_comparison, displacement_comparison, force_comparison


def test_five_percent_diagnostic_gate_is_explicit_and_fail_closed() -> None:
    fields, displacement, force = _five_percent_metric_fixture(field_error=0.049)
    passed = evaluate_five_percent_diagnostic_gate(
        fields,
        displacement,
        force,
        expected_steps=50,
    )
    assert passed["status"] == "passed"
    assert passed["all_metrics_within_tolerance"] is True
    assert passed["parity_claimed"] is False

    failing_fields, failing_displacement, failing_force = _five_percent_metric_fixture(
        field_error=0.051
    )
    failed = evaluate_five_percent_diagnostic_gate(
        failing_fields,
        failing_displacement,
        failing_force,
        expected_steps=50,
    )
    assert failed["status"] == "failed"
    assert failed["all_metrics_within_tolerance"] is False
    assert failed["metrics"]["field_u_nrmse"]["passed"] is False


def test_five_percent_gate_includes_signed_tip_components() -> None:
    fields, displacement, force = _five_percent_metric_fixture(field_error=0.01)
    displacement["tip_transverse"]["nrmse_by_reference_peak"] = 0.051

    result = evaluate_five_percent_diagnostic_gate(
        fields,
        displacement,
        force,
        expected_steps=50,
    )

    assert result["status"] == "failed"
    assert result["metrics"]["tip_transverse_waveform_nrmse"]["passed"] is False


def test_five_percent_gate_does_not_hide_constant_pressure_offset() -> None:
    fields, displacement, force = _five_percent_metric_fixture(field_error=0.01)
    pressure = fields["pressure_reference_diagnostic"]
    pressure["zero_mean_pressure_difference_rmse_pa"] = 0.0
    pressure["raw_pressure_difference_rmse_pa"] = 6.0
    pressure["sampled_mean_offset_our_minus_fluent_pa"] = 6.0

    result = evaluate_five_percent_diagnostic_gate(
        fields,
        displacement,
        force,
        expected_steps=50,
    )

    assert result["status"] == "failed"
    assert result["metrics"]["pressure_raw_nrmse"]["passed"] is False
    assert result["metrics"]["pressure_mean_offset_fraction"]["passed"] is False


def test_five_percent_gate_includes_interface_force_components_and_leakage() -> None:
    fields, displacement, force = _five_percent_metric_fixture(field_error=0.01)
    force["transverse"]["nrmse_by_reference_peak"] = 0.051
    force["out_of_plane_leakage_by_reference_streamwise_peak"] = 0.052

    result = evaluate_five_percent_diagnostic_gate(
        fields,
        displacement,
        force,
        expected_steps=50,
    )

    assert result["status"] == "failed"
    assert result["metrics"]["force_transverse_waveform_nrmse"]["passed"] is False
    assert result["metrics"]["force_out_of_plane_leakage"]["passed"] is False


def test_interface_force_history_maps_axes_and_normalizes_by_span() -> None:
    solver_rows = [
        {
            "step": step,
            "time_s": 5.0e-4 * step,
            "total_marker_force_n": json.dumps(
                [3.0e-4 * step, 6.0e-4 * step, -1.5e-3 * step]
            ),
        }
        for step in range(1, 4)
    ]
    fluent_rows = [
        {
            "step": step,
            "time_s": 5.0e-4 * step,
            "flap_fluid_force_x_n": 0.5 * step,
            "flap_fluid_force_y_n": 0.2 * step,
            "flap_fluid_force_z_n": 0.0,
        }
        for step in range(1, 4)
    ]

    rows, comparison = compare_interface_force_histories(
        solver_rows,
        fluent_rows,
        span_m=0.003,
        expected_steps=3,
        dt_s=5.0e-4,
    )

    assert comparison["schema"] == "native_fine_interface_force_comparison_v1"
    assert comparison["units"] == "N/m"
    assert comparison["streamwise"]["nrmse_by_reference_peak"] == pytest.approx(0.0)
    assert comparison["transverse"]["nrmse_by_reference_peak"] == pytest.approx(0.0)
    assert comparison["out_of_plane_leakage_by_reference_streamwise_peak"] == pytest.approx(0.2)
    assert rows[0]["our_streamwise_force_n_per_m"] == pytest.approx(0.5)
    assert rows[0]["our_transverse_force_n_per_m"] == pytest.approx(0.2)
    assert rows[0]["our_out_of_plane_force_n_per_m"] == pytest.approx(0.1)


def test_interface_force_history_rejects_nonfinite_solver_force() -> None:
    solver_rows = [
        {
            "step": 1,
            "time_s": 5.0e-4,
            "total_marker_force_n": [0.0, float("nan"), -1.0],
        }
    ]
    fluent_rows = [
        {
            "step": 1,
            "time_s": 5.0e-4,
            "flap_fluid_force_x_n": 1.0,
            "flap_fluid_force_y_n": 0.0,
            "flap_fluid_force_z_n": 0.0,
        }
    ]

    with pytest.raises(NativeFineComparisonError, match="finite interface force"):
        compare_interface_force_histories(
            solver_rows,
            fluent_rows,
            span_m=0.003,
            expected_steps=1,
            dt_s=5.0e-4,
        )


def test_interface_force_history_rejects_fluent_time_mismatch() -> None:
    solver_rows = [
        {"step": 1, "time_s": 5.0e-4, "total_marker_force_n": [0.0, 0.3, -1.0]}
    ]
    fluent_rows = [
        {
            "step": 1,
            "time_s": 9.0e-4,
            "flap_fluid_force_x_n": 1.0,
            "flap_fluid_force_y_n": 0.3,
            "flap_fluid_force_z_n": 0.0,
        }
    ]

    with pytest.raises(NativeFineComparisonError, match="force history time mismatch"):
        compare_interface_force_histories(
            solver_rows,
            fluent_rows,
            span_m=1.0,
            expected_steps=1,
            dt_s=5.0e-4,
        )


def test_velocity_overlay_renders_only_deformed_solid_by_default() -> None:
    class RecordingAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[dict[str, object]] = []

        def scatter(self, *args, **kwargs) -> None:
            self.scatter_calls.append(dict(kwargs))

        def legend(self, *args, **kwargs) -> None:
            return None

    axis = RecordingAxis()
    _overlay_deformed_solid(
        axis,
        {
            "solid_rest_x_m": np.asarray([0.04, 0.05]),
            "solid_rest_y_m": np.asarray([0.00, 0.01]),
            "solid_x_m": np.asarray([0.041, 0.051]),
            "solid_y_m": np.asarray([0.00, 0.01]),
            "marker_x_m": np.asarray([0.041, 0.051]),
            "marker_y_m": np.asarray([0.00, 0.01]),
        },
    )

    assert [call["label"] for call in axis.scatter_calls] == ["deformed solid"]


def test_display_mask_restores_boundary_surrogate_without_opening_obstacle() -> None:
    strict = np.asarray([[True, False, False]], dtype=bool)
    surrogate = np.asarray([[False, True, False]], dtype=bool)

    actual = _display_fluid_mask(
        {
            "fluid_mask": strict,
            "boundary_surrogate_mask": surrogate,
        }
    )

    np.testing.assert_array_equal(actual, [[True, True, False]])


def _fine_solver_config(steps: int) -> dict[str, object]:
    return {
        "dt_s": 5.0e-4,
        "span_m": 0.003,
        "duct_length_m": 0.1,
        "step_count": steps,
        "grid_nodes": [4, 256, 320],
        "solid_particle_counts": [1, 256, 20],
        "marker_count": 64,
        "flow_projection_iterations": 1080,
        "solid_substeps": 1600,
        "solid_density_kgm3": 1600.0,
        "young_modulus_pa": 1.0e6,
        "poisson_ratio": 0.47,
        "velocity_damping": 0.995,
        "solid_constitutive_model": "plane_stress_linear_elastic",
        "flow_advection_scheme": "muscl_tvd",
        "flow_turbulence_model": "sst_2003",
        "flow_sst_near_wall_treatment": "resolved",
        "flow_symmetry_domain_walls": ["ymax"],
        "flow_predictor_substeps": 1,
        "flow_hibm_sharp_search_radius_m": 1.7e-3,
        "flow_hibm_sharp_search_radius_xyz_m": [
            1.2e-3,
            0.390625e-3,
            0.46875e-3,
        ],
        "flow_hibm_sharp_interior_probe_distance_m": 1.125e-3,
        "flow_hibm_sharp_interior_probe_distance_xyz_m": None,
        "flow_hibm_sharp_interpolate_velocity_rows": False,
        "flow_hibm_marker_mac_constraint_iterations": 64,
        "flow_hibm_dynamic_solid_volume_enabled": True,
        "update_fluid_obstacle_from_solid": True,
        "flow_hibm_tiny_unreached_cleanup_component_cells": 0,
        "preflow_steps": 200,
        "preflow_convergence_mode": "windowed_stationary",
        "preflow_stationary_min_steps": 20,
        "preflow_stationary_window_steps": 10,
        "preflow_stationary_consecutive_windows": 3,
        "preflow_stationary_tolerance": 0.01,
        "preflow_stationary_divergence_tolerance": 0.05,
        "preflow_stationary_no_slip_tolerance_fraction": 0.05,
        "flow_cg_preconditioner": "fv_multigrid",
        "flow_cg_tolerance": 1.0e-6,
        "flow_pressure_solve_failure_policy": "raise",
        "traction_tip_cap_pressure_enabled": False,
        "traction_pressure_pair_runtime_provider_mode": (
            "runtime_anchored_cell_pair"
        ),
    }


def _fine_solver_summary_identity() -> dict[str, object]:
    return {
        "solver_npz_summary": {
            "span_reduction": "mean",
            "streamwise_velocity_sign": -1.0,
            "reverse_streamwise_axis": True,
        }
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_checksums(root: Path, relative_paths: list[str]) -> None:
    rows = []
    for relative_path in relative_paths:
        payload = (root / relative_path).read_bytes()
        rows.append(f"{hashlib.sha256(payload).hexdigest()}  {relative_path}")
    (root / "CHECKSUMS.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _write_native_source_pair(root: Path, step: int) -> dict[str, object]:
    case_path = root / f"native_step_{step:04d}.cas.h5"
    data_path = root / f"native_step_{step:04d}.dat.h5"
    case_path.write_bytes(f"synthetic-case-{step}".encode("ascii"))
    data_path.write_bytes(f"synthetic-data-{step}".encode("ascii"))
    return {
        "step": step,
        "case_path": str(case_path),
        "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "case_size_bytes": case_path.stat().st_size,
        "data_path": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_size_bytes": data_path.stat().st_size,
    }


def _solver_field(
    step: int,
    *,
    include_deformed_geometry: bool = True,
) -> dict[str, np.ndarray]:
    s = np.asarray([0.02, 0.04, 0.06, 0.08], dtype=np.float64)
    y = np.asarray([0.003, 0.009, 0.015], dtype=np.float64)
    ss, yy = np.meshgrid(s, y)
    u = 10.0 + ss + 2.0 * yy + 0.1 * step
    v = 0.5 * yy
    p = 100.0 * ss - 20.0 * yy + step
    fields = {
        "s": s,
        "y": y,
        "u": u,
        "v": v,
        "p": p,
        "speed": np.hypot(u, v),
        "fluid_mask": np.ones_like(u, dtype=bool),
        "solid_mask": np.zeros_like(u, dtype=bool),
        "boundary_surrogate_mask": np.zeros_like(u, dtype=bool),
        "flow_solution_stage": np.asarray("pre_solid_projection"),
        "boundary_topology_stage": np.asarray("pre_solid_projection"),
        "flow_boundary_state_synchronized": np.asarray(True),
        "structure_geometry_stage": np.asarray("post_solid_observer"),
    }
    if not include_deformed_geometry:
        return fields

    solid_rest_position_m = np.asarray(
        [
            [0.0, 0.0030, 0.0500],
            [0.0, 0.0030, 0.0495],
            [0.0, 0.0090, 0.0500],
            [0.0, 0.0090, 0.0495],
            [0.0, 0.0150, 0.0500],
            [0.0, 0.0150, 0.0495],
        ],
        dtype=np.float64,
    )
    solid_position_m = solid_rest_position_m.copy()
    solid_position_m[:, 2] -= 2.0e-5 * step * np.linspace(0.0, 1.0, 6)
    solid_position_m[:, 1] += 1.0e-5 * step * np.linspace(0.0, 1.0, 6)
    solid_velocity_mps = np.zeros_like(solid_position_m)
    solid_velocity_mps[:, 2] = -0.04 * np.linspace(0.0, 1.0, 6)
    marker_position_m = solid_position_m[[0, 2, 4]]
    marker_velocity_mps = solid_velocity_mps[[0, 2, 4]]
    fields.update(
        {
            "solid_x_m": 0.1 - solid_position_m[:, 2],
            "solid_y_m": solid_position_m[:, 1],
            "solid_rest_x_m": 0.1 - solid_rest_position_m[:, 2],
            "solid_rest_y_m": solid_rest_position_m[:, 1],
            "solid_vx_mps": -solid_velocity_mps[:, 2],
            "solid_vy_mps": solid_velocity_mps[:, 1],
            "solid_position_m": solid_position_m,
            "solid_velocity_mps": solid_velocity_mps,
            "solid_rest_position_m": solid_rest_position_m,
            "solid_fixed_mask": np.asarray([True, True, False, False, False, False]),
            "solid_tip_mask": np.asarray([False, False, False, False, True, True]),
            "marker_x_m": 0.1 - marker_position_m[:, 2],
            "marker_y_m": marker_position_m[:, 1],
            "marker_position_m": marker_position_m,
            "marker_velocity_mps": marker_velocity_mps,
            "marker_normal": np.tile([1.0, 0.0, 0.0], (3, 1)),
            "marker_area_m2": np.full(3, 1.0e-6),
            "marker_region_id": np.arange(3, dtype=np.int64),
        }
    )
    return fields


def _write_solver_frame(
    path: Path,
    step: int,
    *,
    include_deformed_geometry: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **_solver_field(step, include_deformed_geometry=include_deformed_geometry),
    )


def _write_fluent_final_fields(path: Path, step: int) -> None:
    solver = _solver_field(step)
    ss, yy = np.meshgrid(solver["s"], solver["y"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=ss.ravel(),
        y=yy.ravel(),
        u=solver["u"].ravel(),
        v=solver["v"].ravel(),
        p=solver["p"].ravel(),
        speed=solver["speed"].ravel(),
        cell_ids=np.arange(1, ss.size + 1, dtype=np.int64),
    )


def _synthetic_inputs(tmp_path: Path, *, steps: int = 3) -> tuple[Path, Path]:
    our_dir = tmp_path / "our_native_campaign"
    fluent_dir = tmp_path / "native_fluent_postprocess"
    dt_s = 5.0e-4

    _write_json(
        our_dir / "run_manifest.json",
        {
            "run_label": "synthetic_native_comparison",
            "save_step_fields": True,
            "config": _fine_solver_config(steps),
            "source_sha256": {"cases/ansys_vertical_flap_fsi.py": "abc"},
        },
    )
    _write_json(
        our_dir / "our_solver_summary.json",
        {
            "status": "completed",
            "step_count_requested": steps,
            "step_count_completed": steps,
            "dt_s": dt_s,
            "step_artifact_validation": {
                "status": "passed",
                "frame_count": steps,
                "history_count": steps,
            },
            "step_field_frame_count": steps,
            **_fine_solver_summary_identity(),
        },
    )
    _write_json(
        our_dir / "progress.json",
        {"status": "completed", "step_completed": steps, "time_s": steps * dt_s},
    )
    our_history: list[dict[str, object]] = []
    fluent_structure: list[dict[str, object]] = []
    fluent_velocity: list[dict[str, object]] = []
    fluent_pressure: list[dict[str, object]] = []
    fluent_residual: list[dict[str, object]] = []
    fluent_residual_summary: list[dict[str, object]] = []
    fluent_force: list[dict[str, object]] = []
    for step in range(1, steps + 1):
        _write_solver_frame(our_dir / "step_fields" / f"step_{step:04d}.npz", step)
        # Solver streamwise is -z while Fluent streamwise displacement is +x.
        vector = [0.0, 1.0e-5 * step, -2.0e-5 * step]
        vector_norm = float(np.linalg.norm(vector))
        solid_max = 3.0e-5 * step
        our_history.append(
            {
                "step": step,
                "time_s": step * dt_s,
                "tip_mean_displacement_m": json.dumps(vector),
                "tip_displacement_norm_m": vector_norm,
                "max_displacement_m": solid_max,
                "local_velocity_peak_mps": float(np.max(_solver_field(step)["speed"])),
                "total_marker_force_n": json.dumps(
                    [0.0, 6.0e-4 * step, -1.5e-3 * step]
                ),
            }
        )
        _write_json(
            our_dir / "step_history" / f"step_{step:04d}.json",
            {
                "step_index": step,
                "time_s": step * dt_s,
                "history": {
                    "step": step,
                    "tip_mean_displacement_m": vector,
                    "max_displacement_m": solid_max,
                    "local_velocity_peak_mps": float(
                        np.max(_solver_field(step)["speed"])
                    ),
                    "flow_projection_pressure_solve_failed": False,
                    "flow_projection_cg_converged_all": True,
                    "flow_projection_cg_relative_residual_max": 1.0e-8 * step,
                    "flow_projection_cg_project_calls": 1,
                    "flow_projection_pre_projection_l2": 100.0 + step,
                    "flow_projection_post_boundary_l2": 40.0 + step,
                    "flow_projection_projection_l2": 40.0 + step,
                    "flow_projection_pressure_solver_requested": "fv_jacobi",
                    "flow_projection_pressure_solver": "fv_jacobi",
                    "flow_projection_report": {
                        "pressure_solve_failed": False,
                        "cg_converged_all": True,
                        "cg_exact_relative_residual_max": 9.0e-7,
                        "cg_multigrid_to_jacobi_fallback_count": 0,
                        "cg_preconditioner_effective": "fv_multigrid",
                        "cg_preconditioner_requested": "fv_multigrid",
                        "cg_relative_residual_max": 1.0e-8 * step,
                        "pre_projection_l2": 100.0 + step,
                        "post_boundary_l2": 40.0 + step,
                        "projection_l2": 40.0 + step,
                        "pressure_solver_requested": "fv_jacobi",
                        "pressure_solver": "fv_jacobi",
                        "pressure_interface_matrix_row_active_count": 12,
                        "pressure_interface_matrix_row_count": 12,
                        "pressure_interface_matrix_row_invalid_count": 0,
                    },
                    "no_slip_projected_residual_after_projection_mps": 1.0e-4 * step,
                    "total_marker_count": 64,
                    "marker_action_reaction_residual_N": 1.0e-12 * step,
                    "max_abs_traction_pa": 100.0 + step,
                    "mpm_max_speed_mps": 0.01 * step,
                },
            },
        )
        fluent_structure.append(
            {
                "step": step,
                "time_s": step * dt_s,
                "target_x_m": 0.0505,
                "target_y_m": 0.0095,
                "selected_node_count": 4,
                "tip_displacement_x_m": 2.0e-5 * step,
                "tip_displacement_y_m": 1.0e-5 * step,
                "tip_displacement_norm_m": vector_norm,
                "tip_mean_vector_norm_m": vector_norm,
                "max_displacement_m": solid_max,
            }
        )
        fluent_velocity.append(
            {
                "step": step,
                "time_s": step * dt_s,
                "speed_max": float(np.max(_solver_field(step)["speed"])),
            }
        )
        fluent_pressure.append(
            {
                "step": step,
                "time_s": step * dt_s,
                "pressure_min": float(np.min(_solver_field(step)["p"])),
                "pressure_max": float(np.max(_solver_field(step)["p"])),
            }
        )
        fluent_force.append(
            {
                "step": step,
                "time_s": step * dt_s,
                "flap_fluid_force_x_n": 0.5 * step,
                "flap_fluid_force_y_n": 0.2 * step,
                "flap_fluid_force_z_n": 0.0,
            }
        )
        residual_values = (1.0 / (step + 1.0), 1.0 / (step + 2.0))
        data_path = str(tmp_path / f"native_step_{step:04d}.dat.h5")
        for equation, equation_values in (
            ("continuity", residual_values),
            ("x-velocity", tuple(2.0 * value for value in residual_values)),
        ):
            for sample_index, value in enumerate(equation_values):
                fluent_residual.append(
                    {
                        "snapshot_step": step,
                        "snapshot_time_s": step * dt_s,
                        "equation": equation,
                        "sample_index": sample_index,
                        "iteration": sample_index + 1,
                        "value_col0": value,
                        "value_col1": value,
                        "value_col2": 0.0,
                        "value_col3": 0.0,
                        "data_path": data_path,
                    }
                )
            fluent_residual_summary.append(
                {
                    "step": step,
                    "time_s": step * dt_s,
                    "equation": equation,
                    "sample_count": len(equation_values),
                    "first_iteration": 1,
                    "last_iteration": len(equation_values),
                    "primary_initial": equation_values[0],
                    "primary_final": equation_values[-1],
                    "primary_min": min(equation_values),
                    "primary_max": max(equation_values),
                    "stored_value_column_count": 4,
                    "data_path": data_path,
                }
            )

    _write_csv(our_dir / "our_solver_history.csv", our_history)
    _write_csv(
        fluent_dir / "histories" / "structure_displacement_history.csv",
        fluent_structure,
    )
    _write_csv(fluent_dir / "histories" / "velocity_history.csv", fluent_velocity)
    _write_csv(fluent_dir / "histories" / "pressure_history.csv", fluent_pressure)
    _write_csv(fluent_dir / "histories" / "residual_history.csv", fluent_residual)
    _write_csv(
        fluent_dir / "histories" / "residual_snapshot_summary.csv",
        fluent_residual_summary,
    )
    fluent_run_dir = tmp_path / "native_fluent_fsi_production"
    _write_csv(fluent_run_dir / "history.csv", fluent_force)
    _write_fluent_final_fields(fluent_dir / "fields" / "final_fields.npz", steps)
    _write_json(
        fluent_dir / "summary.json",
        {
            "schema": "fluent_fine_fsi_offline_postprocess_v1",
            "status": "complete",
            "offline_only": True,
            "fluent_launched": False,
            "source_artifacts_modified": False,
            "phase_manifest_status": "passed",
            "all_structure_steps_nonzero": True,
            "expected_step_count": steps,
            "step_count": steps,
            "dt_s": dt_s,
            "velocity_display_range_mps": [0.0, 31.0],
            "outputs": {
                "final_fields_npz": "fields/final_fields.npz",
                "pressure_history_csv": "histories/pressure_history.csv",
                "residual_history_csv": "histories/residual_history.csv",
                "residual_snapshot_summary_csv": (
                    "histories/residual_snapshot_summary.csv"
                ),
                "structure_displacement_history_csv": (
                    "histories/structure_displacement_history.csv"
                ),
                "velocity_history_csv": "histories/velocity_history.csv",
            },
        },
    )
    _write_json(
        fluent_dir / "input_manifest.json",
        {
            "schema": "fluent_fine_fsi_input_pairs_v1",
            "step_count": steps,
            "run_dir": str(fluent_run_dir),
            "pairs": [
                _write_native_source_pair(tmp_path, step)
                for step in range(1, steps + 1)
            ],
        },
    )
    _write_checksums(
        fluent_dir,
        FLUENT_CHECKSUM_RELATIVE_PATHS,
    )
    return our_dir, fluent_dir


def test_postprocess_generates_native_only_diagnostic_bundle(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    output_dir = tmp_path / "comparison_output"

    report = postprocess_native_fine_comparison(
        our_dir,
        fluent_dir,
        output_dir,
        expected_steps=3,
        velocity_vmax_mps=31.0,
        gif_duration_ms=20,
    )

    assert report["status"] == "diagnostic_complete"
    assert report["parity_claimed"] is False
    assert report["legacy_puma_reference_used"] is False
    assert report["step_count"] == 3
    assert report["displacement_comparison"]["sample_count"] == 3
    assert report["displacement_comparison"]["tip_streamwise"]["rmse_m"] == pytest.approx(0.0)
    assert report["displacement_comparison"]["tip_transverse"]["rmse_m"] == pytest.approx(0.0)
    assert report["displacement_comparison"]["tip_mean_vector"]["rmse_m"] == pytest.approx(0.0)
    assert report["displacement_comparison"]["solid_max"]["rmse_m"] == pytest.approx(0.0)
    assert report["interface_force_comparison"]["streamwise"][
        "nrmse_by_reference_peak"
    ] == pytest.approx(0.0)
    assert report["interface_force_comparison"]["transverse"][
        "nrmse_by_reference_peak"
    ] == pytest.approx(0.0)
    assert report["final_field_comparison"]["sample_count"] == 12
    assert report["final_field_comparison"]["diagnostic_only"] is True
    assert report["deformed_geometry_contract"]["frame_count"] == 3
    assert report["deformed_geometry_contract"]["true_deformed_geometry_overlay"] is True
    assert report["deformed_geometry_contract"]["solid_point_count"] == 6
    assert report["deformed_geometry_contract"]["marker_point_count"] == 3
    assert report["deformed_geometry_contract"]["observed_nonzero_deformation"] is True
    assert report["deformed_geometry_contract"]["scalar_alias_cross_check"] == "passed"
    assert report["deformed_geometry_contract"]["overlay_layers"] == ["solid_deformed"]
    assert report["deformed_geometry_contract"]["hibm_markers_rendered"] is False
    assert report["deformed_geometry_contract"]["rest_positions_rendered"] is False
    assert report["deformed_geometry_contract"]["streamwise_mapping"] == {
        "reverse_streamwise_axis": True,
        "streamwise_length_m": pytest.approx(0.1),
        "streamwise_velocity_sign": -1.0,
    }
    assert report["fluent_residual_history_contract"]["status"] == "passed"
    assert report["fluent_residual_history_contract"]["covered_steps"] == [1, 2, 3]
    history_contract = report["step_history_contract"]
    assert history_contract["schema"] == "our_solver_flattened_step_history_v1"
    assert history_contract["status"] == "passed"
    assert history_contract["step_count"] == 3
    assert history_contract["history_layout"] == "flattened_diagnostics"
    assert history_contract["aggregate_csv_cross_check"] == "passed"
    assert history_contract["flattened_diagnostic_payload_count"] == 3
    assert history_contract["projection_report_cross_check"] == "passed"
    assert "flow_projection_report" in history_contract["required_flattened_keys"]
    assert len(report["diagnostic_model_blockers"]) >= 4

    displacement_csv = output_dir / "histories" / "displacement_comparison_50point.csv"
    force_csv = output_dir / "histories" / "interface_force_comparison_50point.csv"
    with displacement_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [1, 2, 3]
    assert all(row["definition_alignment"] == "diagnostic_analog_not_identical" for row in rows)
    assert force_csv.is_file()

    gif_path = output_dir / "our_velocity_magnitude_fixed_0_31.gif"
    with Image.open(gif_path) as image:
        assert image.n_frames == 3
    assert (output_dir / "figures" / "final_velocity_comparison.png").is_file()
    assert (output_dir / "figures" / "final_pressure_comparison.png").is_file()
    assert (output_dir / "figures" / "displacement_comparison_50point.png").is_file()
    markdown = (output_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "not a parity claim" in markdown.lower()
    assert "native Fluent" in markdown
    assert "PUMA" not in markdown
    assert "GIF overlay: deformed solid only" in markdown
    assert "deformed solid + deformed HIBM markers" not in markdown
    input_manifest = json.loads((output_dir / "input_manifest.json").read_text(encoding="utf-8"))
    assert input_manifest["solver_step_field_count"] == 3
    assert input_manifest["solver_step_history_count"] == 3
    input_paths = [entry["path"].replace("\\", "/") for entry in input_manifest["inputs"]]
    assert sum("/step_history/step_" in path for path in input_paths) == 3
    assert any(path.endswith("/histories/residual_history.csv") for path in input_paths)
    assert any(path.endswith("/history.csv") for path in input_paths)
    assert any(
        path.endswith("/histories/residual_snapshot_summary.csv")
        for path in input_paths
    )
    checksum_rows = (output_dir / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    assert checksum_rows
    assert all("CHECKSUMS.sha256" not in row for row in checksum_rows)


def test_postprocess_requires_an_exact_contiguous_frame_sequence(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    (our_dir / "step_fields" / "step_0002.npz").unlink()

    with pytest.raises(NativeFineComparisonError, match="exact solver frame sequence"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_frame_without_true_deformed_geometry(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    _write_solver_frame(
        our_dir / "step_fields" / "step_0002.npz",
        2,
        include_deformed_geometry=False,
    )

    with pytest.raises(
        NativeFineComparisonError,
        match="deformed solid geometry.*solid_x_m",
    ):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_mixed_flow_and_structure_time_layer(
    tmp_path: Path,
) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    frame_path = our_dir / "step_fields" / "step_0002.npz"
    with np.load(frame_path, allow_pickle=False) as archive:
        frame = {key: np.asarray(archive[key]) for key in archive.files}
    frame["flow_solution_stage"] = np.asarray(
        "post_solid_kinematic_projection"
    )
    np.savez_compressed(frame_path, **frame)

    with pytest.raises(
        NativeFineComparisonError,
        match="synchronized time layer",
    ):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
            gif_duration_ms=10,
        )
    assert not (tmp_path / "comparison_output").exists()


def test_postprocess_rejects_non_native_fluent_reference(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    summary = json.loads((fluent_dir / "summary.json").read_text(encoding="utf-8"))
    summary["schema"] = "legacy_adapted_fluent_reference"
    _write_json(fluent_dir / "summary.json", summary)

    with pytest.raises(NativeFineComparisonError, match="native Fluent postprocess schema"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_gapped_displacement_history(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    history_path = our_dir / "our_solver_history.csv"
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _write_csv(history_path, [rows[0], rows[2]])

    with pytest.raises(NativeFineComparisonError, match="our-solver history steps"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_missing_full_step_history(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    (our_dir / "step_history" / "step_0002.json").unlink()

    with pytest.raises(NativeFineComparisonError, match="step-history sequence"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_truncated_step_diagnostics(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    history_path = our_dir / "step_history" / "step_0002.json"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    payload["history"].pop("flow_projection_report")
    _write_json(history_path, payload)

    with pytest.raises(NativeFineComparisonError, match="flattened diagnostic keys"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_accepts_twelve_step_flattened_history_generic_validation(
    tmp_path: Path,
) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path, steps=12)

    report = postprocess_native_fine_comparison(
        our_dir,
        fluent_dir,
        tmp_path / "comparison_output",
        expected_steps=12,
        gif_duration_ms=10,
    )

    assert report["step_count"] == 12
    assert report["step_history_contract"]["step_count"] == 12
    assert report["step_history_contract"]["history_layout"] == "flattened_diagnostics"


def test_postprocess_rejects_inconsistent_flattened_projection_report(
    tmp_path: Path,
) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    history_path = our_dir / "step_history" / "step_0002.json"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    payload["history"]["flow_projection_report"]["cg_converged_all"] = False
    _write_json(history_path, payload)

    with pytest.raises(NativeFineComparisonError, match="projection report mismatch"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_wrong_flattened_diagnostic_type(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    history_path = our_dir / "step_history" / "step_0002.json"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    payload["history"]["flow_projection_pressure_solve_failed"] = "false"
    _write_json(history_path, payload)

    with pytest.raises(NativeFineComparisonError, match="must be boolean"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_puma_reference_path(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path / "PUMA_cycle3")

    with pytest.raises(NativeFineComparisonError, match="legacy or adapted Fluent"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_tampered_native_fluent_bundle(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    pressure_path = fluent_dir / "histories" / "pressure_history.csv"
    pressure_path.write_text(
        pressure_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(NativeFineComparisonError, match="checksum mismatch"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_rigid_geometry_frames(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    for step in range(1, 4):
        _write_solver_frame(
            our_dir / "step_fields" / f"step_{step:04d}.npz",
            0,
        )

    with pytest.raises(NativeFineComparisonError, match="zero solid deformation"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_refuses_nonlocked_gif_scale(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)

    with pytest.raises(ValueError, match="fixed 0..31 m/s"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
            velocity_vmax_mps=31.000001,
        )
