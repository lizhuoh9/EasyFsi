from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.tools.test_turek_hron_component_runtime_provenance import (
    RUNTIME_IDENTITY,
)


def _module():
    return importlib.import_module("tools.validation.run_turek_hron_component_gates")


class _RuntimeBase:
    effective_arch = "cuda"
    taichi_runtime_identity = RUNTIME_IDENTITY

    def __init__(self, effective_config):
        self.effective_config = effective_config


def _expected_effective_config(config: dict[str, object]) -> dict[str, object]:
    from cases import turek_hron_fsi as turek

    fields = turek.TurekHronFsiConfig.__dataclass_fields__
    case_config = turek.TurekHronFsiConfig(
        **{key: value for key, value in config.items() if key in fields}
    )
    if config["mode"] in {"fixed-fluid", "coupled-preflight"}:
        case_config = turek.with_beam_surface_force_support(case_config)
    return {
        **asdict(case_config),
        "mode": str(config["mode"]),
        "effective_arch": "cuda",
        "acceleration_mps2": tuple(float(value) for value in config["acceleration_mps2"]),
    }


def _solid_row() -> dict[str, object]:
    return {
        "history_schema_version": 4,
        "solid_macro_requested_time_s": 0.005,
        "solid_macro_accepted_time_s": 0.005,
        "solid_macro_remaining_unadvanced_time_s": 0.0,
        "solid_substeps_requested": 100,
        "solid_substeps_observed": 100,
        "point_a_displacement_solver_xyz_m": [0.0, 0.01, -0.02],
        "_transient_solid_displacement_field_m": [
            [0.0, 0.01, -0.02],
            [0.0, 0.02, -0.04],
        ],
        "_transient_solid_velocity_field_mps": [
            [0.0, 0.1, -0.2],
            [0.0, 0.2, -0.4],
        ],
        "fixed_root_max_displacement_m": 0.0,
        "grid_out_of_bounds_particle_count": 0,
        "deformation_clamp_count": 0,
        "total_mass_kg": 2.0,
        "applied_force_sum_solver_xyz_n": [0.0, 0.02, 0.0],
    }


def _fluid_row(step: int) -> dict[str, object]:
    return {
        "history_schema_version": 4,
        "step": step,
        "time_s": step * 0.005,
        "fluid_macro_requested_time_s": 0.005,
        "fluid_macro_accepted_time_s": 0.005,
        "fluid_macro_remaining_unadvanced_time_s": 0.0,
        "fluid_predictor_substeps": 1,
        "_transient_fluid_velocity_active_field_mps": [
            [0.0, 0.1, -0.2],
            [0.0, 0.2, -0.3],
        ],
        "reported_force_solver_xyz_n": [0.0, 2.0, -3.0],
        "beam_force_solver_xyz_n": [0.0, 1.0, -1.0],
        "cylinder_pressure_force_solver_xyz_n": [0.0, 0.5, -1.0],
        "cylinder_viscous_force_solver_xyz_n": [0.0, 0.5, -1.0],
        "inlet_flux_m3ps": 0.0041,
        "outlet_flux_m3ps": 0.0041,
        "marker_layout_sha256": "frozen-marker-layout",
        "external_wall_face_max_residual_mps": 0.0,
        "base_cylinder_velocity_max_abs_mps": 0.0,
        "base_obstacle_crossing_normal_max_abs_mps": 0.0,
        "beam_marker_no_slip_rms_mps": 0.0,
        "beam_marker_no_slip_max_mps": 0.0,
        "beam_marker_valid_count": 112,
        "beam_marker_invalid_count": 0,
        "projection_pressure_finite": True,
        "projection_cg_converged_all": True,
        "projection_cg_breakdown_count": 0,
        "outlet_pressure_reference_rows_valid": True,
        "hibm_topology_valid": True,
        "pressure_nullspace_component_labels_converged": True,
        "pressure_nullspace_component_overflow": False,
        "hibm_pressure_reachability_converged": True,
        "hibm_pressure_reachability_valid": True,
        "hibm_pressure_component_labels_converged": True,
        "cg_unreached_component_overflow": False,
        "projection_pressure_solve_failed": False,
        "projection_physical_failure": False,
        "projection_rhs_nonzero": True,
        "cg_nonzero_rhs_project_calls": 1,
        "cg_nonzero_rhs_preconditioner_requested": "fv_multigrid",
        "cg_nonzero_rhs_preconditioner_effective": "fv_multigrid",
        "cg_nonzero_rhs_multigrid_to_jacobi_fallback_count": 0,
        "expected_marker_count": 112,
        "total_drag_per_span_n_per_m": 60.0,
        "total_lift_per_span_n_per_m": 40.0,
    }


def test_frozen_component_config_uses_real_turek_fields_and_values():
    config = _module().frozen_component_config("solid-only", nx=4, solid_substeps=100)
    assert config["grid_nodes"] == (4, 48, 288)
    assert config["step_count"] == 40
    assert config["markers_per_side"] == config["markers_per_tip"] == "auto"
    assert config["flow_predictor_substeps"] == 1
    assert config["fluid_advection_scheme"] == "rk2"
    assert config["flow_projection_iterations"] == 4000
    assert config["flow_cg_tolerance"] == pytest.approx(1.0e-6)
    assert config["ib_anisotropic_envelope"] is True
    assert config["classify_far_internal_nodes"] is True
    assert config["flow_cg_preconditioner"] == "fv_multigrid"
    assert config["flow_reprojection_iterations"] == 1200
    assert config["flow_reprojection_cg_tolerance"] == pytest.approx(1.0e-4)
    assert config["marker_reseed_interval_steps"] is None
    assert config["velocity_damping"] == pytest.approx(1.0)


def test_pure_formula_contracts_reject_zero_or_nonfinite_denominators():
    module = _module()
    assert module.relative_vector_delta([3.0, 4.0], [0.0, 5.0]) == pytest.approx(
        0.632455532
    )
    assert np.allclose(module.point_a_vector([0.0, 2.0, 3.0]), [-3.0, 2.0])
    assert np.allclose(module.force_vector([8.0, 4.0, -6.0]), [6.0, 4.0])
    assert module.force_closure([0.0, 0.0], [0.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="zero"):
        module.relative_vector_delta([1.0], [0.0])
    with pytest.raises(ValueError, match="ZERO_DENOMINATOR"):
        module.concatenated_axis_l2_leakage(
            [{"axis_l2": [0.0, 0.0, 0.0]}],
            "axis_l2",
        )


def test_solid_gate_records_full_axis_norms_and_rejects_false_mean_leakage():
    module = _module()
    config = module.frozen_component_config("solid-only")
    evaluated = module.evaluate_solid_row(_solid_row(), config)
    assert evaluated["point_a_turek_hron_m"] == pytest.approx([0.02, 0.01])
    assert evaluated["solid_displacement_axis_l2_m"] == pytest.approx(
        [0.0, 0.0223606798, 0.0447213595]
    )
    leaking = _solid_row()
    leaking["_transient_solid_velocity_field_mps"] = [
        [1.0, 0.1, -0.2],
        [-1.0, 0.2, -0.4],
    ]
    evaluated = module.evaluate_solid_row(leaking, config)
    assert evaluated["solid_velocity_axis_l2_mps"] == pytest.approx(
        [np.sqrt(2.0), np.sqrt(0.05), np.sqrt(0.2)]
    )
    legacy = _solid_row()
    legacy["solid_displacement_field_m"] = legacy.pop("_transient_solid_displacement_field_m")
    legacy["solid_velocity_field_mps"] = legacy.pop("_transient_solid_velocity_field_mps")
    with pytest.raises(ValueError, match="FAIL_SCHEMA"):
        module.evaluate_solid_row(legacy, config)


def test_solid_gate_rejects_missing_time_substeps_clamp_and_force_reapplication():
    module = _module()
    config = module.frozen_component_config("solid-only")
    bad = _solid_row()
    bad["solid_substeps_observed"] = 99
    with pytest.raises(ValueError, match="FAIL_SOLID_SUBSTEPS"):
        module.evaluate_solid_row(bad, config)
    bad = _solid_row()
    bad["deformation_clamp_count"] = 1
    with pytest.raises(ValueError, match="FAIL_SOLID_DEFORMATION_CLAMP"):
        module.evaluate_solid_row(bad, config)
    bad = _solid_row()
    bad["applied_force_sum_solver_xyz_n"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="FAIL_SOLID_MASS_PROPORTIONAL_FORCE"):
        module.evaluate_solid_row(bad, config)
    rounded = _solid_row()
    rounded["applied_force_sum_solver_xyz_n"] = [0.0, 0.02000002, 0.0]
    module.evaluate_solid_row(rounded, config)


def test_fixed_fluid_gate_rejects_force_closure_and_solver_health_false_positives():
    module = _module()
    config = module.frozen_component_config("fixed-fluid")
    assert (
        module.evaluate_fixed_fluid_row(_fluid_row(401), config)["raw_force_closure"]
        == 0.0
    )
    bad = _fluid_row(401)
    bad["reported_force_solver_xyz_n"] = [0.0, 2.0, -2.9]
    with pytest.raises(ValueError, match="FAIL_FORCE_CLOSURE"):
        module.evaluate_fixed_fluid_row(bad, config)
    bad = _fluid_row(401)
    bad["cg_nonzero_rhs_preconditioner_effective"] = "jacobi"
    with pytest.raises(ValueError, match="FAIL_FLUID_HEALTH"):
        module.evaluate_fixed_fluid_row(bad, config)
    inconsistent = _fluid_row(401)
    inconsistent["projection_rhs_nonzero"] = False
    with pytest.raises(ValueError, match="FAIL_FLUID_HEALTH"):
        module.evaluate_fixed_fluid_row(inconsistent, config)
    zero_rhs = _fluid_row(401)
    zero_rhs.update(
        {
            "projection_rhs_nonzero": False,
            "cg_nonzero_rhs_project_calls": 0,
            "cg_nonzero_rhs_preconditioner_requested": "not_applicable",
            "cg_nonzero_rhs_preconditioner_effective": "not_applicable",
            "cg_nonzero_rhs_multigrid_to_jacobi_fallback_count": 0,
        }
    )
    module.evaluate_fixed_fluid_row(zero_rhs, config)
    field = _fluid_row(401).pop("_transient_fluid_velocity_active_field_mps")
    replacements = ("fluid_velocity_active_field_mps", field), (
        "fluid_velocity_axis_l2_mps", np.linalg.norm(field, axis=0).tolist()
    )
    for key, value in replacements:
        legacy = _fluid_row(401)
        legacy.pop("_transient_fluid_velocity_active_field_mps")
        legacy[key] = value
        with pytest.raises(ValueError, match="FAIL_SCHEMA"):
            module.evaluate_fixed_fluid_row(legacy, config)


def test_window_leakage_concatenates_axis_norms_before_dividing():
    module = _module()
    rows = [
        {"axis_l2": [0.002, 0.0, 1.0]},
        {"axis_l2": [0.0, 0.0, 100.0]},
    ]

    assert module.concatenated_axis_l2_leakage(rows, "axis_l2") == pytest.approx(
        0.002 / np.sqrt(1.0 + 10000.0)
    )


def test_initialization_audit_rejects_obstacles_outside_physical_volume():
    module = _module()
    config = module.frozen_component_config("fixed-fluid")
    audit = {
        "cylinder_connected": True,
        "beam_connected": True,
        "no_sealed_fluid_pocket": True,
        "zero_load_fields_finite": True,
        "hibm_base_obstacle_established": True,
        "hibm_topology_valid": True,
        "base_cylinder_mask_exact": True,
        "beam_interior_mask_complete": True,
        "obstacle_union_single_component": True,
        "unexpected_obstacle_outside_beam_or_cylinder_cell_count": 1,
        "marker_counts": (54, 4),
    }

    with pytest.raises(ValueError, match="FAIL_INIT_AUDIT"):
        module.evaluate_initialization_audit(audit, config)


def test_fixed_fluid_runner_uses_step_dt_and_exact_post_ramp_window(tmp_path: Path):
    module = _module()
    events: list[tuple[str, float | None]] = []

    class Runtime(_RuntimeBase):
        marker_layout_hash = "frozen-marker-layout"

        def initialize_time_zero(self):
            self.write_boundary(0.0)
            events.append(("initialize_time_zero", None))
            return self.initialization_audit()

        def initialization_audit(self):
            return {
                "cylinder_connected": True,
                "beam_connected": True,
                "beam_interior_obstacle_complete": True,
                "no_sealed_fluid_pocket": True,
                "zero_load_fields_finite": True,
                "hibm_base_obstacle_established": True,
                "hibm_topology_valid": True,
                "base_cylinder_mask_exact": True,
                "beam_interior_mask_complete": True,
                "obstacle_union_single_component": True,
                "unexpected_obstacle_outside_beam_or_cylinder_cell_count": 0,
                "marker_counts": (54, 4),
            }

        def write_boundary(self, time_s):
            events.append(("boundary", time_s))

        def assemble(self):
            physical_steps = sum(name == "assemble" for name, _ in events) + 1
            events.append(("assemble", None))
            row = _fluid_row(physical_steps)
            if physical_steps <= 400:
                row["_transient_fluid_velocity_active_field_mps"] = [
                    [100.0, 0.0, 0.0]
                ]
            elif physical_steps == 401:
                row["_transient_fluid_velocity_active_field_mps"] = [
                    [0.002, 0.0, 1.0]
                ]
            elif physical_steps > 401:
                row["_transient_fluid_velocity_active_field_mps"] = [
                    [0.0, 0.0, 100.0]
                ]
            row.pop("history_schema_version")
            return row

        def assert_fixed_markers(self):
            return None

        def final_arrays(self):
            return {"fluid": np.zeros((2, 3))}

    result = module.run_fixed_fluid(
        module.frozen_component_config("fixed-fluid"),
        Runtime,
        tmp_path,
        label="fluid",
    )
    assert result["status"] == "PASS_COMPONENT_ONLY"
    assert events[:2] == [("boundary", 0.0), ("initialize_time_zero", None)]
    boundary_times = [value for name, value in events if name == "boundary"]
    assert boundary_times[1] == pytest.approx(0.005)
    assert boundary_times[401] == pytest.approx(2.005)
    assert result["metrics"]["inlet_mean_relative_error_max"] == pytest.approx(0.0)
    assert result["metrics"]["initialization_audit"]["base_cylinder_mask_exact"]
    assert result["manifest"]["parent_identities"] == {}
    history_header = (tmp_path / "fluid" / "history.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "_transient_fluid_velocity_active_field_mps" not in history_header
    assert "history_schema_version" in history_header


def test_run_claims_output_before_runtime_work_and_json_is_strict(tmp_path: Path):
    module = _module()
    config = module.frozen_component_config("solid-only")

    class Runtime(_RuntimeBase):
        def __init__(self, effective_config):
            super().__init__(effective_config)
            assert (tmp_path / "solid").is_dir()

        def apply_acceleration(self, value):
            assert (tmp_path / "solid").is_dir()

        def advance(self):
            row = _solid_row()
            row.pop("history_schema_version")
            return row

        def tip_row(self):
            return {
                "tip_ux_turek_hron_m": 0.02,
                "tip_uy_turek_hron_m": 0.01,
                "fixed_root_max_displacement_m": 0.0,
            }

        def final_arrays(self):
            return {"solid": np.array([1.0])}

    result = module.run_solid_only(config, Runtime, tmp_path, label="solid")
    json.loads((tmp_path / "solid" / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS_COMPONENT_ONLY"
    history_header = (tmp_path / "solid" / "history.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "history_schema_version" in history_header
    assert "_transient_solid_displacement_field_m" not in history_header
    assert "_transient_solid_velocity_field_mps" not in history_header
    with pytest.raises(TypeError):
        module._canonical({"not_json": object()})
    with pytest.raises(ValueError, match="FAIL_RUNTIME_TYPE"):
        module.run_solid_only(config, lambda _: object(), tmp_path, label="missing-arch")


def _completed_run(
    module, root: Path, label: str, *, mode: str, nx: int, substeps: int = 100
) -> Path:
    config = module.frozen_component_config(mode, nx=nx, solid_substeps=substeps)
    step_count = int(config["step_count"])
    rows = []
    for step in range(1, step_count + 1):
        row = _solid_row() if mode == "solid-only" else _fluid_row(step)
        row["step"] = step
        row["time_s"] = step * float(config["dt_s"])
        if mode == "solid-only":
            row.update({"tip_ux_turek_hron_m": 0.02, "tip_uy_turek_hron_m": 0.01})
        else:
            row.update(
                {
                    "total_drag_per_span_n_per_m": 60.0,
                    "total_lift_per_span_n_per_m": 40.0,
                }
            )
        rows.append(row)
    run_dir = module._claim_output_dir(root, label)
    return Path(
        module._write_artifacts(
            run_dir,
            label,
            config,
            rows,
            {"value": np.asarray([nx])},
            {"status": "PASS_COMPONENT_ONLY", "completed": True, "metrics": {}},
            "same-layout",
            taichi_runtime_identity=RUNTIME_IDENTITY,
        )["run_dir"]
    )


@pytest.mark.parametrize(
    ("mode", "expected_steps"),
    (("solid-only", 40), ("fixed-fluid", 500)),
)
def test_completed_reader_requires_exact_component_history_ledgers(
    tmp_path: Path,
    mode: str,
    expected_steps: int,
):
    module = _module()
    config = module.frozen_component_config(mode)
    rows = []
    for step in range(1, expected_steps):
        row = _solid_row() if mode == "solid-only" else _fluid_row(step)
        row.update({"step": step, "time_s": step * float(config["dt_s"])})
        if mode == "solid-only":
            row.update({"tip_ux_turek_hron_m": 0.02, "tip_uy_turek_hron_m": 0.01})
        rows.append(row)
    run_dir = module._claim_output_dir(tmp_path, f"short-{mode}")
    module._write_artifacts(
        run_dir,
        f"short-{mode}",
        config,
        rows,
        {"value": np.asarray([expected_steps])},
        {"status": "PASS_COMPONENT_ONLY", "completed": True, "metrics": {}},
    )

    with pytest.raises(ValueError, match="FAIL_ARTIFACT_HISTORY_LENGTH"):
        module._read_completed(run_dir)


def test_completed_reader_rejects_nonsequential_component_history(tmp_path: Path):
    module = _module()
    config = module.frozen_component_config("solid-only")
    rows = []
    for step in range(1, 41):
        row = _solid_row()
        recorded_step = 39 if step == 40 else step
        row.update(
            {
                "step": recorded_step,
                "time_s": recorded_step * float(config["dt_s"]),
                "tip_ux_turek_hron_m": 0.02,
                "tip_uy_turek_hron_m": 0.01,
            }
        )
        rows.append(row)
    run_dir = module._claim_output_dir(tmp_path, "duplicate-step")
    module._write_artifacts(
        run_dir,
        "duplicate-step",
        config,
        rows,
        {"value": np.asarray([40])},
        {"status": "PASS_COMPONENT_ONLY", "completed": True, "metrics": {}},
    )

    with pytest.raises(ValueError, match="FAIL_ARTIFACT_HISTORY_LEDGER"):
        module._read_completed(run_dir)


def test_completed_reader_rejects_row_history_schema_mismatch(tmp_path: Path):
    module = _module()
    config = module.frozen_component_config("solid-only")
    rows = []
    for step in range(1, 41):
        row = _solid_row()
        row.update(
            {
                "history_schema_version": 3 if step == 20 else 4,
                "step": step,
                "time_s": step * float(config["dt_s"]),
                "tip_ux_turek_hron_m": 0.02,
                "tip_uy_turek_hron_m": 0.01,
            }
        )
        rows.append(row)
    run_dir = module._claim_output_dir(tmp_path, "bad-row-schema")
    module._write_artifacts(
        run_dir,
        "bad-row-schema",
        config,
        rows,
        {"value": np.asarray([40])},
        {"status": "PASS_COMPONENT_ONLY", "completed": True, "metrics": {}},
    )

    with pytest.raises(ValueError, match="FAIL_ARTIFACT_HISTORY_SCHEMA"):
        module._read_completed(run_dir)


def test_mode_specific_compare_verifies_hashes_and_persists_parent_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()
    left = _completed_run(
        module, tmp_path, "solid100", mode="solid-only", nx=4, substeps=100
    )
    right = _completed_run(
        module, tmp_path, "solid200", mode="solid-only", nx=4, substeps=200
    )
    result = module.persist_comparison(left, right, tmp_path, label="solid-compare")
    assert result["status"] == "PASS_COMPONENT_ONLY"
    assert result["manifest"]["parent_identities"]["left_run_id"] == "solid100"
    module._read_completed(Path(result["run_dir"]))
    payload = json.loads((right / "run_manifest.json").read_text(encoding="utf-8"))
    payload["config_sha256"] = "tampered"
    (right / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="FAIL_ARTIFACT_CONFIG_HASH"):
        module.compare_completed_runs(left, right)
    monkeypatch.setattr(module, "persist_comparison", lambda *args, **kwargs: {"status": "FAIL_COMPONENT_COMPARISON"})
    assert module.main(["compare", "--left", "l", "--right", "r", "--label", "failed"]) == 1


def test_compare_requires_one_oriented_axis_and_uses_fixed_fluid_window_mean(
    tmp_path: Path,
):
    module = _module()
    left = _completed_run(
        module, tmp_path, "fluid4", mode="fixed-fluid", nx=4
    )
    right_config = module.frozen_component_config("fixed-fluid", nx=8)
    right_rows = []
    for step in range(1, 501):
        row = _fluid_row(step)
        offset = 10.0 / 99.0 if 401 <= step < 500 else -10.0 if step == 500 else 0.0
        row.update(
            {
                "total_drag_per_span_n_per_m": 60.0 + offset,
                "total_lift_per_span_n_per_m": 40.0,
            }
        )
        right_rows.append(row)
    right_dir = module._claim_output_dir(tmp_path, "fluid8")
    module._write_artifacts(
        right_dir,
        "fluid8",
        right_config,
        right_rows,
        {"value": np.asarray([8])},
        {"status": "PASS_COMPONENT_ONLY", "completed": True, "metrics": {}},
        "same-layout",
        taichi_runtime_identity=RUNTIME_IDENTITY,
    )

    assert module.compare_completed_runs(left, right_dir)["status"] == (
        "PASS_COMPONENT_ONLY"
    )
    with pytest.raises(ValueError, match="coarse-to-fine"):
        module.compare_completed_runs(right_dir, left)
    mixed = _completed_run(
        module,
        tmp_path,
        "solid8x200",
        mode="solid-only",
        nx=8,
        substeps=200,
    )
    solid4x100 = _completed_run(
        module,
        tmp_path,
        "solid4x100",
        mode="solid-only",
        nx=4,
        substeps=100,
    )
    with pytest.raises(ValueError, match="exactly one comparison axis"):
        module.compare_completed_runs(solid4x100, mixed)

    solid4x200 = _completed_run(
        module,
        tmp_path,
        "solid4x200",
        mode="solid-only",
        nx=4,
        substeps=200,
    )
    solid8x100 = _completed_run(module, tmp_path, "solid8x100", mode="solid-only", nx=8, substeps=100)
    for invalid_pair in ((solid4x100, solid8x100), (solid8x100, mixed)):
        with pytest.raises(ValueError, match="FAIL_SOLID_COMPARISON_ANCHOR"):
            module.compare_completed_runs(*invalid_pair)
    with pytest.raises(ValueError, match="coarse-to-fine"):
        module.compare_completed_runs(solid4x200, solid4x100)


def test_completed_reader_checks_npz_array_and_config_source_hashes(tmp_path: Path):
    module = _module()
    run = _completed_run(
        module, tmp_path, "solid", mode="solid-only", nx=4
    )
    np.savez(run / "final.npz", value=np.asarray([999]))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["artifact_sha256"]["final.npz"] = module._sha_file(run / "final.npz")
    (run / "summary.json").write_bytes(module._canonical(summary) + b"\n")
    manifest["artifacts"]["final.npz"] = module._sha_file(run / "final.npz")
    manifest["artifacts"]["summary.json"] = module._sha_file(run / "summary.json")
    (run / "run_manifest.json").write_bytes(module._canonical(manifest) + b"\n")
    with pytest.raises(ValueError, match="FAIL_ARTIFACT_ARRAY_HASH"):
        module._read_completed(run)

    clean = _completed_run(
        module, tmp_path, "solid-clean", mode="solid-only", nx=4
    )
    clean_manifest = json.loads(
        (clean / "run_manifest.json").read_text(encoding="utf-8")
    )
    clean_manifest["config_source_sha256"] = "tampered"
    (clean / "run_manifest.json").write_bytes(
        module._canonical(clean_manifest) + b"\n"
    )
    with pytest.raises(ValueError, match="FAIL_ARTIFACT_CONFIG_SOURCE_HASH"):
        module._read_completed(clean)


def test_completed_reader_rejects_schema_source_and_artifact_tampering(
    tmp_path: Path,
):
    module = _module()

    bad_schema = _completed_run(
        module, tmp_path, "bad-schema", mode="solid-only", nx=4
    )
    manifest = json.loads(
        (bad_schema / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest["schema"] = int(manifest["schema"]) + 1
    (bad_schema / "run_manifest.json").write_bytes(
        module._canonical(manifest) + b"\n"
    )
    with pytest.raises(ValueError, match="completed PASS evidence"):
        module._read_completed(bad_schema)

    bad_source = _completed_run(
        module, tmp_path, "bad-source", mode="solid-only", nx=4
    )
    manifest = json.loads(
        (bad_source / "run_manifest.json").read_text(encoding="utf-8")
    )
    source_name = next(iter(manifest["source_hashes"]))
    manifest["source_hashes"][source_name] = "0" * 64
    manifest["source_sha256"] = module.hashlib.sha256(
        module._canonical(manifest["source_hashes"])
    ).hexdigest()
    manifest["config_source_sha256"] = module.hashlib.sha256(
        module._canonical(
            {"config": manifest["config"], "sources": manifest["source_hashes"]}
        )
    ).hexdigest()
    (bad_source / "run_manifest.json").write_bytes(
        module._canonical(manifest) + b"\n"
    )
    with pytest.raises(ValueError, match="FAIL_ARTIFACT_SOURCE_DRIFT"):
        module._read_completed(bad_source)

    bad_artifact = _completed_run(
        module, tmp_path, "bad-artifact", mode="solid-only", nx=4
    )
    with (bad_artifact / "history.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="FAIL_ARTIFACT_HASH"):
        module._read_completed(bad_artifact)

    bad_summary_hashes = _completed_run(
        module, tmp_path, "bad-summary-hashes", mode="solid-only", nx=4
    )
    manifest = json.loads(
        (bad_summary_hashes / "run_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (bad_summary_hashes / "summary.json").read_text(encoding="utf-8")
    )
    summary["artifact_sha256"]["history.csv"] = "0" * 64
    (bad_summary_hashes / "summary.json").write_bytes(
        module._canonical(summary) + b"\n"
    )
    manifest["artifacts"]["summary.json"] = module._sha_file(
        bad_summary_hashes / "summary.json"
    )
    (bad_summary_hashes / "run_manifest.json").write_bytes(
        module._canonical(manifest) + b"\n"
    )
    with pytest.raises(ValueError, match="FAIL_ARTIFACT_SUMMARY_HASHES"):
        module._read_completed(bad_summary_hashes)


def test_coupled_preflight_requires_measured_generic_fsi_history(tmp_path: Path):
    module = _module()
    config = module.frozen_component_config("coupled-preflight")
    config["step_count"] = 2
    effective_config = _expected_effective_config(config)
    runtime_module = importlib.import_module("tools.validation.turek_hron_component_gate_runtimes")
    assert runtime_module.normalize_component_effective_config(config) == effective_config

    class Runtime(_RuntimeBase):
        marker_verified = True
        config_override = None

        def __init__(self, config):
            super().__init__(config)
            if self.config_override is not None:
                self.effective_config = {**config, **self.config_override}

        def run(self):
            return {
                "generic_runtime_completed_steps": 2,
                "completed_steps": 2,
                "history": [
                    _preflight_row(1, config),
                    _preflight_row(2, config),
                ],
                "solver_path": "simulation_core.drivers.generic_fsi_solver.solve_fsi_runtime",
                "accepted_time_s": 0.01,
                "effective_config": self.effective_config,
                "marker_layout_sha256": "a" * 64,
                "marker_layout_identity_verified": self.marker_verified,
                "taichi_runtime_identity": self.taichi_runtime_identity,
            }

    result = module.run_coupled_preflight(config, Runtime, tmp_path, label="preflight")
    assert result["manifest"]["marker_layout_sha256"] == "a" * 64
    assert result["manifest"]["config"] == effective_config
    Runtime.config_override = {"dt_s": 0.004}
    with pytest.raises(ValueError, match="FAIL_EFFECTIVE_CONFIG"):
        module.run_coupled_preflight(config, Runtime, tmp_path, label="bad-config")
    Runtime.config_override, Runtime.marker_verified = None, False
    with pytest.raises(ValueError, match="FAIL_PREFLIGHT_MARKER_LAYOUT"):
        module.run_coupled_preflight(config, Runtime, tmp_path, label="bad-marker")
    config["step_count"] = 3
    with pytest.raises(ValueError, match="only permits"):
        module.run_coupled_preflight(config, Runtime, tmp_path, label="bad")


def _preflight_row(step: int, config: dict[str, object]) -> dict[str, object]:
    dt = float(config["dt_s"])
    return {
        "step": step,
        "time_s": step * dt,
        "history_schema_version": 4,
        "fluid_macro_requested_time_s": dt,
        "fluid_macro_accepted_time_s": dt,
        "fluid_macro_remaining_unadvanced_time_s": 0.0,
        "fluid_predictor_substeps": int(config["flow_predictor_substeps"]),
        "solid_macro_requested_time_s": dt,
        "solid_macro_accepted_time_s": dt,
        "solid_macro_remaining_unadvanced_time_s": 0.0,
        "solid_substeps": int(config["solid_substeps"]),
        "mpm_grid_out_of_bounds_particle_count": 0,
        "mpm_deformation_clamp_count": 0,
        "fsi_coupling_converged": True,
    }


def test_coupled_preflight_rejects_missing_row_ledgers(tmp_path: Path):
    module = _module()
    config = module.frozen_component_config("coupled-preflight")

    class Runtime(_RuntimeBase):
        def run(self):
            return {
                "generic_runtime_completed_steps": 1,
                "completed_steps": 1,
                "history": [{"step": 1, "time_s": 0.005}],
                "solver_path": "simulation_core.drivers.generic_fsi_solver.solve_fsi_runtime",
                "accepted_time_s": 0.005,
                "effective_config": self.effective_config,
                "marker_layout_sha256": "a" * 64,
                "marker_layout_identity_verified": True,
                "taichi_runtime_identity": self.taichi_runtime_identity,
            }

    with pytest.raises(ValueError, match="FAIL_PREFLIGHT_ROW_SCHEMA"):
        module.run_coupled_preflight(
            config,
            Runtime,
            tmp_path,
            label="missing-ledger",
        )


def test_cli_exposes_frozen_matrix_axes_and_rejects_cpu():
    module = _module()
    parser = module._parser()
    args = parser.parse_args(
        ["solid-only", "--nx", "8", "--solid-substeps", "200"]
    )
    assert args.nx == 8
    assert args.solid_substeps == 200
    with pytest.raises(SystemExit):
        parser.parse_args(["solid-only", "--arch", "cpu"])
    with pytest.raises(ValueError, match="FAIL_PREFLIGHT_GRID"):
        module.frozen_component_config("coupled-preflight", nx=8)
