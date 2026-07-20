from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.official import solid_mpm_fsi_runner
from cases.ansys_vertical_flap_fsi import selected_formulation_solver_config
from simulation_core.fluids.preflow_snapshot import (
    PREFLOW_SNAPSHOT_FIELD_NAMES,
    PreflowSnapshot,
    PreflowSnapshotIdentity,
    PreflowSnapshotValidationError,
    load_preflow_snapshot,
    save_preflow_snapshot,
)


class _FakeField:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values).copy()
        self.from_numpy_calls = 0

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()

    def from_numpy(self, values: np.ndarray) -> None:
        self.from_numpy_calls += 1
        self.values = np.asarray(values).copy()


class _FailOnceField(_FakeField):
    def __init__(self, values: np.ndarray):
        super().__init__(values)
        self._failures_remaining = 1

    def from_numpy(self, values: np.ndarray) -> None:
        self.from_numpy_calls += 1
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise RuntimeError("injected from_numpy failure")
        self.values = np.asarray(values).copy()


def _fake_fluid() -> SimpleNamespace:
    scalar_f32 = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    scalar_f64 = scalar_f32.astype(np.float64)
    scalar_i32 = scalar_f32.astype(np.int32)
    vector_f32 = np.stack((scalar_f32, scalar_f32 + 1, scalar_f32 + 2), axis=-1)
    boundary_active = scalar_i32 % 2
    boundary_inactive = boundary_active == 0
    boundary_weight = scalar_f32 / 7.0
    boundary_weight[boundary_inactive] = 0.0
    # These are legacy direct rows, so their projection and enforcement
    # weights must match exactly under the snapshot contract.
    boundary_enforcement_weight = boundary_weight.copy()
    boundary_region = scalar_i32 + 6
    boundary_region[boundary_inactive] = -1
    boundary_hard_mask = scalar_i32 % 8
    boundary_hard_mask[boundary_inactive] = 0
    boundary_external_exact_mask = boundary_hard_mask & np.int32(0b100)
    canonical_active_mask = np.zeros_like(boundary_active)
    canonical_vector_shape = boundary_active.shape + (3,)
    external_x_plane_shape = (2, boundary_active.shape[1], boundary_active.shape[2])
    external_y_plane_shape = (2, boundary_active.shape[0], boundary_active.shape[2])
    external_z_plane_shape = (2, boundary_active.shape[0], boundary_active.shape[1])
    fluid = SimpleNamespace(
        cell_face_x_m=_FakeField(np.linspace(0.0, 1.0, 3, dtype=np.float64)),
        cell_face_y_m=_FakeField(np.linspace(0.0, 1.0, 3, dtype=np.float64)),
        cell_face_z_m=_FakeField(np.linspace(0.0, 1.0, 3, dtype=np.float64)),
        cell_center_x_m=_FakeField(np.asarray((0.25, 0.75), dtype=np.float64)),
        cell_center_y_m=_FakeField(np.asarray((0.25, 0.75), dtype=np.float64)),
        cell_center_z_m=_FakeField(np.asarray((0.25, 0.75), dtype=np.float64)),
        cell_width_x_m=_FakeField(np.asarray((0.5, 0.5), dtype=np.float64)),
        cell_width_y_m=_FakeField(np.asarray((0.5, 0.5), dtype=np.float64)),
        cell_width_z_m=_FakeField(np.asarray((0.5, 0.5), dtype=np.float64)),
        velocity=_FakeField(vector_f32),
        velocity_prev=_FakeField(vector_f32 + 3),
        pressure=_FakeField(scalar_f64),
        fsi_pressure=_FakeField(scalar_f64 + 1),
        sst_turbulent_kinetic_energy=_FakeField(
            np.full_like(scalar_f32, 0.375)
        ),
        sst_specific_dissipation_rate=_FakeField(
            np.full_like(scalar_f32, 125.0)
        ),
        sst_eddy_viscosity_pa_s=_FakeField(
            np.full_like(scalar_f32, 1.8e-4)
        ),
        sst_wall_distance_m=_FakeField(
            np.full_like(scalar_f32, 2.5e-3)
        ),
        obstacle=_FakeField(np.zeros_like(scalar_i32)),
        hibm_base_obstacle=_FakeField((scalar_i32 + 1) % 2),
        hibm_dynamic_solid_volume_obstacle=_FakeField((scalar_i32 + 2) % 2),
        hibm_dynamic_solid_volume_external_carve=_FakeField((scalar_i32 + 3) % 2),
        velocity_dirichlet_boundary_active=_FakeField(boundary_active),
        velocity_dirichlet_boundary_value_mps=_FakeField(vector_f32 + 4),
        velocity_dirichlet_boundary_projection_weight=_FakeField(boundary_weight),
        velocity_dirichlet_boundary_enforcement_weight=_FakeField(
            boundary_enforcement_weight
        ),
        velocity_dirichlet_boundary_marker_region_id=_FakeField(boundary_region),
        velocity_dirichlet_boundary_hard_fixed_component_mask=_FakeField(
            boundary_hard_mask
        ),
        velocity_dirichlet_boundary_external_exact_component_mask=_FakeField(
            boundary_external_exact_mask
        ),
        velocity_dirichlet_boundary_owned_row=_FakeField(
            np.zeros_like(boundary_active)
        ),
        velocity_dirichlet_boundary_active_component_mask=_FakeField(
            canonical_active_mask
        ),
        velocity_dirichlet_boundary_pressure_mobility=_FakeField(
            np.ones(canonical_vector_shape, dtype=np.float32)
        ),
        velocity_dirichlet_boundary_component_enforcement_weight=_FakeField(
            np.zeros(canonical_vector_shape, dtype=np.float32)
        ),
        velocity_dirichlet_boundary_component_region_id=_FakeField(
            np.full(canonical_vector_shape, -1, dtype=np.int32)
        ),
        velocity_dirichlet_boundary_owned_component_mask=_FakeField(
            np.zeros_like(boundary_active)
        ),
        external_velocity_boundary_x_face_active_component_mask=_FakeField(
            np.full(external_x_plane_shape, 0b001, dtype=np.int32)
        ),
        external_velocity_boundary_x_face_value_mps=_FakeField(
            np.full(external_x_plane_shape + (3,), 1.0, dtype=np.float32)
        ),
        external_velocity_boundary_y_face_active_component_mask=_FakeField(
            np.full(external_y_plane_shape, 0b010, dtype=np.int32)
        ),
        external_velocity_boundary_y_face_value_mps=_FakeField(
            np.full(external_y_plane_shape + (3,), 2.0, dtype=np.float32)
        ),
        external_velocity_boundary_z_face_active_component_mask=_FakeField(
            np.full(external_z_plane_shape, 0b100, dtype=np.int32)
        ),
        external_velocity_boundary_z_face_value_mps=_FakeField(
            np.full(external_z_plane_shape + (3,), 3.0, dtype=np.float32)
        ),
        velocity_dirichlet_boundary_authority="legacy",
        velocity_dirichlet_component_ledger_generation=0,
        velocity_dirichlet_component_ledger_sealed=False,
        _velocity_dirichlet_component_ledger_consumer_generations={},
        _velocity_dirichlet_component_ledger_consumer_capabilities={},
    )
    fluid._require_velocity_dirichlet_component_ledger_sealed = lambda: None
    fluid._invalidate_hibm_pressure_reachability = lambda: None
    fluid.build_hibm_no_slip_sampling_obstacle = lambda: None
    fluid.build_hibm_no_slip_component_face_valid_mask = lambda: None
    return fluid


def _zero_snapshot_fields(fluid: SimpleNamespace) -> None:
    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        getattr(fluid, name).values.fill(0)


def _fake_snapshot_identity_geometry() -> tuple[SimpleNamespace, SimpleNamespace]:
    markers = SimpleNamespace(
        marker_count=2,
        projection_vertex_count=2,
        projection_triangle_count=0,
        projection_segment_count=0,
        x_gamma_m=_FakeField(
            np.asarray(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)), dtype=np.float32)
        ),
        v_gamma_mps=_FakeField(np.zeros((2, 3), dtype=np.float32)),
        n_gamma=_FakeField(
            np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32)
        ),
        A_gamma_m2=_FakeField(np.asarray((0.01, 0.02), dtype=np.float32)),
        region_id=_FakeField(np.asarray((1, 2), dtype=np.int32)),
        projection_vertex_pressure_owner_index=_FakeField(
            np.asarray((0, 1), dtype=np.int32)
        ),
        projection_triangle_indices=_FakeField(
            np.full((2, 3), -1, dtype=np.int32)
        ),
        pressure_probe_origin_m=_FakeField(
            np.asarray(((0.1, 0.2, 0.31), (0.4, 0.5, 0.61)), dtype=np.float32)
        ),
        pressure_probe_origin_explicit=_FakeField(
            np.asarray((1, 1), dtype=np.int32)
        ),
    )
    solid = SimpleNamespace(
        particle_count=2,
        rest_x=_FakeField(
            np.asarray(((0.2, 0.3, 0.4), (0.5, 0.6, 0.7)), dtype=np.float32)
        ),
        fixed_particle=_FakeField(np.asarray((1, 0), dtype=np.int32)),
    )
    return markers, solid


def _snapshot_health_config() -> SimpleNamespace:
    return SimpleNamespace(
        flow_solid_boundary_mode="hibm_sharp_marker_rows",
        marker_count=1,
        traction_marker_layout="single_mid_surface",
        inlet_velocity_mps=10.0,
        preflow_convergence_mode="single_step_legacy",
        preflow_stationary_no_slip_tolerance_fraction=0.05,
        preflow_traction_readiness_mode="flow_only",
    )


def _healthy_hibm_velocity_report(
    *,
    active_rows: int = 1,
    boundary_velocity_only_rows: int = 1,
) -> dict[str, object]:
    projection_weight = 0.5 if active_rows > 0 else 0.0
    return {
        "hibm_sharp_marker_boundary_enabled": True,
        "hibm_sharp_marker_boundary_topology_reused": True,
        "hibm_preassembly_cleanup_reused": False,
        "hibm_preassembly_topology_mutated": False,
        "hibm_velocity_dirichlet_active_rows": active_rows,
        "hibm_velocity_dirichlet_primary_region_active_rows": active_rows,
        "hibm_velocity_dirichlet_secondary_region_active_rows": 0,
        "hibm_velocity_dirichlet_other_region_active_rows": 0,
        "hibm_velocity_dirichlet_unassigned_region_active_rows": 0,
        "hibm_velocity_dirichlet_max_abs_velocity_mps": 0.25,
        "hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": 0.5,
        "hibm_velocity_dirichlet_boundary_velocity_only_rows": (
            boundary_velocity_only_rows
        ),
        "hibm_velocity_dirichlet_invalid_reconstruction_count": 0,
        "hibm_velocity_dirichlet_invalid_no_fluid_sample_count": 0,
        "hibm_velocity_dirichlet_invalid_nonpositive_gap_count": 0,
        "hibm_velocity_dirichlet_invalid_node_behind_boundary_count": 0,
        "hibm_velocity_dirichlet_invalid_node_beyond_interior_count": 0,
        "hibm_velocity_dirichlet_narrow_gap_count": 0,
        "hibm_velocity_dirichlet_relocated_rows": 0,
        "hibm_velocity_dirichlet_relocation_merged_rows": 0,
        "hibm_velocity_dirichlet_relocation_blocked_rows": 0,
        "hibm_velocity_dirichlet_min_projection_weight": projection_weight,
        "hibm_velocity_dirichlet_max_projection_weight": projection_weight,
        "hibm_velocity_dirichlet_row_ledger_snapshot_generation": 1,
        "hibm_velocity_dirichlet_row_ledger_matches_reference": True,
        "hibm_velocity_dirichlet_row_ledger_mismatch_rows": 0,
    }


def _exact_velocity_dirichlet_ledger_fake_methods() -> dict[str, object]:
    state = {"generation": 0}

    def capture() -> int:
        state["generation"] += 1
        return int(state["generation"])

    def mismatch_rows(*, expected_generation: int) -> int:
        if int(expected_generation) != int(state["generation"]):
            raise ValueError("unexpected ledger generation")
        return 0

    def comparison(*, expected_generation: int) -> dict[str, object]:
        mismatch_rows(expected_generation=expected_generation)
        return {
            "schema_version": 1,
            "reference_generation": int(expected_generation),
            "device_content_mismatch_rows": 0,
            "identity_mismatch_rows": 0,
            "content_equivalence_mismatch_rows": 0,
            "authority_changed": False,
            "component_generation_changed": False,
            "face_symmetric_changed": False,
            "reference_authority": "canonical",
            "current_authority": "canonical",
            "reference_component_generation": 1,
            "current_component_generation": 1,
            "reference_face_symmetric": 0,
            "current_face_symmetric": 0,
            "first_identity_mismatch_field": None,
            "first_content_mismatch_field": None,
        }

    return {
        "capture_velocity_dirichlet_boundary_ledger_reference": capture,
        "velocity_dirichlet_boundary_ledger_comparison": comparison,
        "velocity_dirichlet_boundary_ledger_mismatch_rows": mismatch_rows,
    }


def _healthy_preflow_report() -> dict[str, object]:
    return {
        "preflow_history": [
            {
                "preflow_step": 1,
                "flow_projection_report": {
                    "projection_l2": 1.0,
                    "pressure_nullspace_policy": (
                        "pressure_outlet_dirichlet_operator_anchored"
                    ),
                    "pressure_nullspace_compatibility_measured": True,
                    "pressure_outlet_operator_graph_prepared": True,
                    "pressure_nullspace_component_labels_converged": True,
                    "pressure_nullspace_component_overflow": False,
                    "pressure_nullspace_component_count": 0,
                    "pressure_nullspace_incompatible_component_count": 0,
                    "pressure_nullspace_componentwise_projection_applied": False,
                    "pressure_nullspace_zero_mean_projection_applied": False,
                    "pressure_interface_matrix_active": True,
                    "pressure_interface_matrix_row_invalid_count": 0,
                    "pressure_interface_matrix_row_overflow_count": 0,
                    "unreached_cells_with_interface_diagonal": 0,
                    "unreached_cells_with_interface_coupling": 0,
                    "cg_unreached_component_count": 0,
                    "cg_unreached_component_raw_count": 0,
                    "unreached_components_with_interface_hits": 0,
                },
                "flow_projection_cg_converged_all": True,
                "flow_projection_cg_breakdown_count": 0,
                "flow_projection_pressure_solve_failed": False,
                "flow_projection_pressure_projection_physical_failure": False,
                "flow_projection_pre_projection_velocity_projector_prepared_all": (
                    True
                ),
                "flow_projection_pre_projection_velocity_projector_converged_all": (
                    True
                ),
                "flow_projection_pre_projection_velocity_projector_committed_all": (
                    True
                ),
                "hibm_no_slip_valid_marker_count": 1,
                "hibm_no_slip_invalid_marker_count": 0,
                "hibm_no_slip_max_residual_mps": 0.1,
                "stress_valid_marker_count": 0,
                "stress_invalid_marker_count": 1,
                "hibm_preassembly_topology_mutated": False,
                "hibm_preassembly_remaining_unreached_cell_count": 0,
                **_healthy_hibm_velocity_report(),
            }
        ],
        "preflow_steps_completed": 1,
        "preflow_convergence_mode": "single_step_legacy",
        "preflow_converged": False,
        "preflow_status": "max_steps",
        "preflow_stop_reason": "max_steps",
        "final_flow_field_snapshot": {},
    }


def _windowed_snapshot_config() -> SimpleNamespace:
    return SimpleNamespace(
        **{
            **vars(_snapshot_health_config()),
            "preflow_convergence_mode": "windowed_stationary",
            "preflow_stationary_min_steps": 2,
            "preflow_stationary_window_steps": 2,
            "preflow_stationary_consecutive_windows": 2,
            "preflow_stationary_tolerance": 0.05,
            "preflow_stationary_divergence_tolerance": 0.05,
            "air_density_kgm3": 1.2,
            "flap_height_m": 0.01,
            "span_m": 0.003,
            "duct_length_m": 0.1,
            "duct_height_m": 0.04,
            "grid_nodes": (2, 2, 2),
        }
    )


def _windowed_stationary_report() -> dict[str, object]:
    history = []
    for step in range(1, 6):
        history.append(
            {
                "preflow_step": step,
                "local_velocity_peak_mps": 30.0,
                "pressure_min_pa": -10.0,
                "pressure_max_pa": 300.0,
                "total_marker_force_n": [0.0, 0.0, -0.009],
                "flow_projection_l2": 1.0,
                "flow_projection_report": {
                    "projection_l2": 1.0,
                    "pressure_nullspace_policy": (
                        "pressure_outlet_dirichlet_operator_anchored"
                    ),
                    "pressure_nullspace_compatibility_measured": True,
                    "pressure_outlet_operator_graph_prepared": True,
                    "pressure_nullspace_component_labels_converged": True,
                    "pressure_nullspace_component_overflow": False,
                    "pressure_nullspace_component_count": 0,
                    "pressure_nullspace_incompatible_component_count": 0,
                    "pressure_nullspace_componentwise_projection_applied": False,
                    "pressure_nullspace_zero_mean_projection_applied": False,
                    "pressure_interface_matrix_active": True,
                    "pressure_interface_matrix_row_invalid_count": 0,
                    "pressure_interface_matrix_row_overflow_count": 0,
                    "unreached_cells_with_interface_diagonal": 0,
                    "unreached_cells_with_interface_coupling": 0,
                    "cg_unreached_component_count": 0,
                    "cg_unreached_component_raw_count": 0,
                    "unreached_components_with_interface_hits": 0,
                },
                "flow_projection_cg_converged_all": True,
                "flow_projection_cg_breakdown_count": 0,
                "flow_projection_pressure_solve_failed": False,
                "flow_projection_pressure_projection_physical_failure": False,
                "flow_projection_pre_projection_velocity_projector_prepared_all": (
                    True
                ),
                "flow_projection_pre_projection_velocity_projector_converged_all": (
                    True
                ),
                "flow_projection_pre_projection_velocity_projector_committed_all": (
                    True
                ),
                "hibm_no_slip_valid_marker_count": 1,
                "hibm_no_slip_invalid_marker_count": 0,
                "hibm_no_slip_max_residual_mps": 0.1,
                "stress_valid_marker_count": 0,
                "stress_invalid_marker_count": 1,
                "hibm_preassembly_topology_mutated": False,
                "hibm_preassembly_remaining_unreached_cell_count": 0,
                **_healthy_hibm_velocity_report(),
                "flow_volume_source_applied": False,
                "flow_inlet_boundary_reapplied": True,
                "flow_inlet_source_factor": 1.0,
            }
        )
    return {
        "preflow_history": history,
        "preflow_steps_completed": len(history),
        "preflow_convergence_mode": "windowed_stationary",
        "preflow_converged": True,
        "preflow_status": "windowed_stationary",
        "preflow_stop_reason": "windowed_stationary",
        "final_flow_field_snapshot": {},
    }


def test_preflow_snapshot_state_fields_round_trip_without_loss():
    source = _fake_fluid()
    captured = solid_mpm_fsi_runner._capture_preflow_snapshot_fields(source)

    target = _fake_fluid()
    _zero_snapshot_fields(target)
    solid_mpm_fsi_runner._restore_preflow_snapshot_fields(target, captured)

    assert set(captured) == set(PREFLOW_SNAPSHOT_FIELD_NAMES)
    for name, expected in captured.items():
        np.testing.assert_array_equal(getattr(target, name).to_numpy(), expected)


def test_preflow_snapshot_restore_validates_all_fields_before_first_mutation():
    source = _fake_fluid()
    captured = solid_mpm_fsi_runner._capture_preflow_snapshot_fields(source)
    invalid = {**captured, "velocity": captured["velocity"].astype(np.float64)}
    target = _fake_fluid()
    before = {
        name: getattr(target, name).to_numpy()
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES
    }

    with pytest.raises(PreflowSnapshotValidationError, match="dtype"):
        solid_mpm_fsi_runner._restore_preflow_snapshot_fields(target, invalid)

    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        field = getattr(target, name)
        assert field.from_numpy_calls == 0
        np.testing.assert_array_equal(field.to_numpy(), before[name])


def test_preflow_snapshot_restore_rolls_back_partial_runtime_commit():
    source = _fake_fluid()
    captured = solid_mpm_fsi_runner._capture_preflow_snapshot_fields(source)
    target = _fake_fluid()
    _zero_snapshot_fields(target)
    target.pressure = _FailOnceField(target.pressure.to_numpy())
    before = {
        name: getattr(target, name).to_numpy()
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES
    }
    generation_before = target.velocity_dirichlet_component_ledger_generation

    with pytest.raises(RuntimeError, match="injected from_numpy failure"):
        solid_mpm_fsi_runner._restore_preflow_snapshot_fields(target, captured)

    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        np.testing.assert_array_equal(getattr(target, name).to_numpy(), before[name])
    assert target.velocity_dirichlet_component_ledger_generation == generation_before


def test_preflow_identity_changes_for_any_directed_external_face_profile():
    """A portable snapshot cannot silently replace a different live BC map."""

    config = {"grid_nodes": (2, 2, 2), "boundary_profile_mode": "directed"}
    markers, solid = _fake_snapshot_identity_geometry()
    baseline_fluid = _fake_fluid()
    with patch.object(
        solid_mpm_fsi_runner,
        "_preflow_snapshot_source_payload",
        return_value={"solver.py": b"stable-source"},
    ):
        baseline_identity = solid_mpm_fsi_runner._preflow_snapshot_identity(
            markers=markers,
            fluid=baseline_fluid,
            solid=solid,
            config=config,
        )
        plane_fields = (
            (
                "external_velocity_boundary_x_face_active_component_mask",
                "external_velocity_boundary_x_face_value_mps",
            ),
            (
                "external_velocity_boundary_y_face_active_component_mask",
                "external_velocity_boundary_y_face_value_mps",
            ),
            (
                "external_velocity_boundary_z_face_active_component_mask",
                "external_velocity_boundary_z_face_value_mps",
            ),
        )
        for axis_index, (mask_name, value_name) in enumerate(plane_fields):
            for side_index in range(2):
                changed_fluid = _fake_fluid()
                changed_mask = getattr(changed_fluid, mask_name).to_numpy()
                changed_value = getattr(changed_fluid, value_name).to_numpy()
                changed_mask[side_index, 0, 0] ^= np.int32(0b111)
                changed_value[side_index, 0, 0] += np.asarray(
                    (0.125, -0.25, 0.5),
                    dtype=np.float32,
                )
                getattr(changed_fluid, mask_name).from_numpy(changed_mask)
                getattr(changed_fluid, value_name).from_numpy(changed_value)

                changed_identity = solid_mpm_fsi_runner._preflow_snapshot_identity(
                    markers=markers,
                    fluid=changed_fluid,
                    solid=solid,
                    config=config,
                )
                assert (
                    changed_identity.geometry_sha256
                    != baseline_identity.geometry_sha256
                ), (
                    "all six directed external faces are model boundary "
                    "geometry for snapshot compatibility; a legal but "
                    "different profile must not reuse the same portable "
                    f"snapshot identity (axis={axis_index}, side={side_index})"
                )


def test_preflow_identity_changes_when_only_tip_cap_projection_geometry_moves():
    config = {"grid_nodes": (2, 2, 2), "boundary_profile_mode": "directed"}
    markers, solid = _fake_snapshot_identity_geometry()
    markers.projection_vertex_count = 4
    markers.projection_segment_count = 1
    markers.x_gamma_m = _FakeField(
        np.asarray(
            (
                (0.1, 0.2, 0.3),
                (0.4, 0.5, 0.6),
                (0.1, 0.7, 0.3),
                (0.4, 0.7, 0.6),
            ),
            dtype=np.float32,
        )
    )
    markers.v_gamma_mps = _FakeField(np.zeros((4, 3), dtype=np.float32))
    markers.n_gamma = _FakeField(
        np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=np.float32,
        )
    )
    markers.A_gamma_m2 = _FakeField(
        np.asarray((0.01, 0.02, 0.005, 0.005), dtype=np.float32)
    )
    markers.region_id = _FakeField(np.asarray((1, 2, 303, 303), dtype=np.int32))
    markers.projection_vertex_pressure_owner_index = _FakeField(
        np.asarray((0, 1, 2, 3), dtype=np.int32)
    )
    markers.projection_triangle_indices = _FakeField(
        np.asarray(((2, 3, -1), (-1, -1, -1)), dtype=np.int32)
    )
    fluid = _fake_fluid()

    with patch.object(
        solid_mpm_fsi_runner,
        "_preflow_snapshot_source_payload",
        return_value={"solver.py": b"stable-source"},
    ):
        baseline_identity = solid_mpm_fsi_runner._preflow_snapshot_identity(
            markers=markers,
            fluid=fluid,
            solid=solid,
            config=config,
        )
        changed_positions = markers.x_gamma_m.to_numpy()
        changed_positions[2, 1] += np.float32(0.01)
        markers.x_gamma_m.from_numpy(changed_positions)
        changed_identity = solid_mpm_fsi_runner._preflow_snapshot_identity(
            markers=markers,
            fluid=fluid,
            solid=solid,
            config=config,
        )

    assert changed_identity.geometry_sha256 != baseline_identity.geometry_sha256


def _install_complete_canonical_restore_prepare_fixture(
    target: SimpleNamespace,
    events: list[str],
    *,
    fail_consumer: str | None = None,
    expected_fields: dict[str, np.ndarray] | None = None,
) -> tuple[str, ...]:
    prepare_methods = dict(
        solid_mpm_fsi_runner._CANONICAL_SNAPSHOT_RESTORE_PREPARE_METHODS
    )
    target._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS = frozenset(
        prepare_methods
    )
    target._velocity_dirichlet_component_ledger_generation_errors = (
        lambda: ([], [], [], [])
    )
    for consumer, method_name in prepare_methods.items():
        def prepare(
            consumer_name: str = consumer,
        ) -> None:
            if expected_fields is not None:
                for field_name, expected in expected_fields.items():
                    np.testing.assert_array_equal(
                        getattr(target, field_name).to_numpy(),
                        expected,
                    )
            events.append(f"prepare:{consumer_name}")
            if consumer_name == fail_consumer:
                raise RuntimeError(
                    f"injected canonical prepare failure: {consumer_name}"
                )

        setattr(target, method_name, prepare)
    return tuple(prepare_methods)


def _canonical_snapshot_source(*, generation: int) -> SimpleNamespace:
    source = _fake_fluid()
    for name in (
        "velocity_dirichlet_boundary_active",
        "velocity_dirichlet_boundary_value_mps",
        "velocity_dirichlet_boundary_projection_weight",
        "velocity_dirichlet_boundary_enforcement_weight",
        "velocity_dirichlet_boundary_hard_fixed_component_mask",
        "velocity_dirichlet_boundary_external_exact_component_mask",
        "velocity_dirichlet_boundary_owned_row",
        "velocity_dirichlet_boundary_active_component_mask",
        "velocity_dirichlet_boundary_component_enforcement_weight",
        "velocity_dirichlet_boundary_owned_component_mask",
    ):
        getattr(source, name).values.fill(0)
    source.velocity_dirichlet_boundary_marker_region_id.values.fill(-1)
    source.velocity_dirichlet_boundary_pressure_mobility.values.fill(1.0)
    source.velocity_dirichlet_boundary_component_region_id.values.fill(-1)
    source.velocity_dirichlet_boundary_authority = "canonical"
    source.velocity_dirichlet_component_ledger_generation = generation
    return source


def test_runner_capture_round_trips_canonical_component_without_legacy_active_row(
    tmp_path,
):
    source = _canonical_snapshot_source(generation=7)
    row = (0, 0, 0)
    source.velocity_dirichlet_boundary_active_component_mask.values[row] = 0b100
    source.velocity_dirichlet_boundary_hard_fixed_component_mask.values[row] = 0b100
    source.velocity_dirichlet_boundary_owned_component_mask.values[row] = 0b100
    source.velocity_dirichlet_boundary_pressure_mobility.values[row + (2,)] = 0.0
    source.velocity_dirichlet_boundary_component_enforcement_weight.values[
        row + (2,)
    ] = 1.0
    source.velocity_dirichlet_boundary_component_region_id.values[row + (2,)] = 0

    identity = PreflowSnapshotIdentity(
        config_sha256="0" * 64,
        source_sha256="1" * 64,
        geometry_sha256="2" * 64,
    )
    snapshot = PreflowSnapshot(
        fields=solid_mpm_fsi_runner._capture_preflow_snapshot_fields(source),
        identity=identity,
        velocity_dirichlet_boundary_authority="canonical",
        velocity_dirichlet_component_ledger_generation=7,
    )
    prefix = tmp_path / "canonical_component_without_legacy_row"
    save_preflow_snapshot(prefix, snapshot)
    loaded = load_preflow_snapshot(
        prefix,
        expected_identity=identity,
        expected_velocity_dirichlet_boundary_authority="canonical",
    )

    target = _fake_fluid()
    _zero_snapshot_fields(target)
    target.velocity_dirichlet_boundary_authority = "canonical"
    events: list[str] = []
    _install_complete_canonical_restore_prepare_fixture(
        target,
        events,
        expected_fields=dict(loaded.fields),
    )
    target._invalidate_hibm_pressure_reachability = lambda: events.append(
        "invalidate"
    )

    def seal() -> None:
        target.velocity_dirichlet_component_ledger_sealed = True

    target.seal_velocity_dirichlet_component_ledger = seal
    target._require_velocity_dirichlet_component_ledger_sealed = lambda: None
    solid_mpm_fsi_runner._restore_preflow_snapshot_fields(
        target,
        loaded.fields,
        velocity_dirichlet_boundary_authority="canonical",
        velocity_dirichlet_component_ledger_generation=7,
    )

    assert loaded.fields["velocity_dirichlet_boundary_active"][row] == 0
    assert (
        loaded.fields["velocity_dirichlet_boundary_active_component_mask"][row]
        == 0b100
    )
    assert target.velocity_dirichlet_boundary_active.to_numpy()[row] == 0
    assert (
        target.velocity_dirichlet_boundary_owned_component_mask.to_numpy()[row]
        == 0b100
    )


def test_canonical_snapshot_restore_missing_consumer_fails_before_first_write():
    source = _canonical_snapshot_source(generation=7)
    captured = solid_mpm_fsi_runner._capture_preflow_snapshot_fields(source)

    target = _fake_fluid()
    target.velocity_dirichlet_boundary_authority = "canonical"
    target._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS = frozenset(
        solid_mpm_fsi_runner._CANONICAL_SNAPSHOT_RESTORE_PREPARE_METHODS
    )
    before = {
        name: getattr(target, name).to_numpy()
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES
    }
    events: list[str] = []
    target.prepare_velocity_dirichlet_component_ledger_apply = lambda: events.append(
        "prepare:apply"
    )
    target.seal_velocity_dirichlet_component_ledger = lambda: events.append("seal")

    with pytest.raises(RuntimeError, match="canonical snapshot restore.*missing"):
        solid_mpm_fsi_runner._restore_preflow_snapshot_fields(
            target,
            captured,
            velocity_dirichlet_boundary_authority="canonical",
            velocity_dirichlet_component_ledger_generation=7,
        )

    assert events == []
    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        field = getattr(target, name)
        assert field.from_numpy_calls == 0
        np.testing.assert_array_equal(field.to_numpy(), before[name])


def test_canonical_snapshot_restore_prepares_all_consumers_before_seal():
    source = _canonical_snapshot_source(generation=7)
    captured = solid_mpm_fsi_runner._capture_preflow_snapshot_fields(source)

    target = _fake_fluid()
    target.velocity_dirichlet_boundary_authority = "canonical"
    events: list[str] = []
    consumers = _install_complete_canonical_restore_prepare_fixture(
        target,
        events,
        expected_fields=captured,
    )
    target._invalidate_hibm_pressure_reachability = lambda: events.append(
        "invalidate"
    )
    target.build_hibm_no_slip_sampling_obstacle = lambda: pytest.fail(
        "canonical restore must not enter the sealed-only no-slip builder"
    )
    target.build_hibm_no_slip_component_face_valid_mask = lambda: pytest.fail(
        "canonical restore must use the no-slip prepare API"
    )

    def seal() -> None:
        events.append("seal")
        target.velocity_dirichlet_component_ledger_sealed = True

    def require_sealed() -> None:
        events.append("require")
        if not target.velocity_dirichlet_component_ledger_sealed:
            raise RuntimeError("canonical ledger is not sealed")

    target.seal_velocity_dirichlet_component_ledger = seal
    target._require_velocity_dirichlet_component_ledger_sealed = require_sealed

    solid_mpm_fsi_runner._restore_preflow_snapshot_fields(
        target,
        captured,
        velocity_dirichlet_boundary_authority="canonical",
        velocity_dirichlet_component_ledger_generation=7,
    )

    assert events == [
        "invalidate",
        *(f"prepare:{consumer}" for consumer in consumers),
        "seal",
        "require",
    ]


def test_canonical_snapshot_restore_prepare_failure_rolls_back_without_retry():
    source = _canonical_snapshot_source(generation=8)
    captured = solid_mpm_fsi_runner._capture_preflow_snapshot_fields(source)

    target = _fake_fluid()
    _zero_snapshot_fields(target)
    target.velocity_dirichlet_boundary_authority = "canonical"
    before = {
        name: getattr(target, name).to_numpy()
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES
    }
    generation_before = target.velocity_dirichlet_component_ledger_generation
    events: list[str] = []
    _install_complete_canonical_restore_prepare_fixture(
        target,
        events,
        fail_consumer="gradient",
    )
    target._invalidate_hibm_pressure_reachability = lambda: events.append(
        "invalidate"
    )
    target.seal_velocity_dirichlet_component_ledger = lambda: events.append("seal")
    target.build_hibm_no_slip_sampling_obstacle = lambda: pytest.fail(
        "rollback must not recursively rebuild canonical derived state"
    )
    target.build_hibm_no_slip_component_face_valid_mask = lambda: pytest.fail(
        "rollback must not recursively rebuild canonical derived state"
    )

    with pytest.raises(
        RuntimeError,
        match="injected canonical prepare failure: gradient",
    ):
        solid_mpm_fsi_runner._restore_preflow_snapshot_fields(
            target,
            captured,
            velocity_dirichlet_boundary_authority="canonical",
            velocity_dirichlet_component_ledger_generation=8,
        )

    assert events.count("prepare:gradient") == 1
    assert "seal" not in events
    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        np.testing.assert_array_equal(getattr(target, name).to_numpy(), before[name])
    assert target.velocity_dirichlet_component_ledger_generation == generation_before


def test_legacy_snapshot_restore_preserves_derived_rebuild_order():
    source = _fake_fluid()
    captured = solid_mpm_fsi_runner._capture_preflow_snapshot_fields(source)
    target = _fake_fluid()
    events: list[str] = []
    target._invalidate_hibm_pressure_reachability = lambda: events.append(
        "invalidate"
    )
    target.build_hibm_no_slip_sampling_obstacle = lambda: events.append("obstacle")
    target.build_hibm_no_slip_component_face_valid_mask = lambda: events.append(
        "component_mask"
    )
    target._require_velocity_dirichlet_component_ledger_sealed = (
        lambda: events.append("require")
    )

    solid_mpm_fsi_runner._restore_preflow_snapshot_fields(target, captured)

    assert events == ["invalidate", "obstacle", "component_mask", "require"]


def test_runner_snapshot_helpers_round_trip_frozen_history(tmp_path):
    identity = PreflowSnapshotIdentity(
        config_sha256="0" * 64,
        source_sha256="1" * 64,
        geometry_sha256="2" * 64,
    )
    config = _snapshot_health_config()
    report = _healthy_preflow_report()
    source = _fake_fluid()
    prefix = tmp_path / "preflow"

    solid_mpm_fsi_runner._write_fixed_solid_preflow_snapshot(
        path=prefix,
        report=report,
        markers=object(),
        fluid=source,
        solid=object(),
        config=config,
        identity=identity,
    )
    target = _fake_fluid()
    _zero_snapshot_fields(target)
    restored = solid_mpm_fsi_runner._restore_fixed_solid_preflow_snapshot(
        path=prefix,
        markers=object(),
        fluid=target,
        solid=object(),
        config=config,
        expected_identity=identity,
    )

    assert isinstance(restored["preflow_history"], list)
    assert isinstance(restored["preflow_history"][0], dict)
    assert isinstance(restored["preflow_history"][0]["flow_projection_report"], dict)
    assert restored["preflow_snapshot_loaded"] is True
    json.dumps(restored)
    for name, expected in solid_mpm_fsi_runner._capture_preflow_snapshot_fields(
        source
    ).items():
        np.testing.assert_array_equal(getattr(target, name).to_numpy(), expected)


def test_flow_only_snapshot_records_traction_as_not_evaluated():
    payload = solid_mpm_fsi_runner._preflow_report_snapshot_payload(
        _healthy_preflow_report(),
        _snapshot_health_config(),
    )

    assert payload["preflow_traction_readiness_mode"] == "flow_only"
    assert payload["preflow_traction_readiness"] == "not_evaluated"


def test_snapshot_accepts_reused_saturated_cleanup_provenance():
    report = _healthy_preflow_report()
    report["preflow_history"][-1].update(
        {
            "hibm_preassembly_cleanup_reused": True,
            "hibm_preassembly_topology_mutated": False,
            "hibm_preassembly_overflow_singleton_cleanup_cell_count": 1,
            "hibm_preassembly_overflow_singleton_cleanup_component_count": 1,
        }
    )

    payload = solid_mpm_fsi_runner._preflow_report_snapshot_payload(
        report,
        _snapshot_health_config(),
    )

    terminal = payload["preflow_history"][-1]
    assert terminal["hibm_preassembly_cleanup_reused"] is True
    assert terminal["hibm_preassembly_topology_mutated"] is False
    assert terminal["hibm_preassembly_overflow_singleton_cleanup_cell_count"] == 1


def test_preflow_snapshot_accepts_legacy_canonical_schema_two_history():
    report = _healthy_preflow_report()
    terminal_row = report["preflow_history"][-1]
    extension_keys = {
        "direct_geometry_reconstructed_component_count",
        "direct_geometry_one_sided_component_count",
        "max_compatible_direct_target_spread_mps",
        "marker_target_closure",
    }
    legacy_device_report = {
        key: 0
        for key in (
            solid_mpm_fsi_runner
            .CANONICAL_HIBM_VELOCITY_DIRICHLET_LEGACY_SCHEMA_TWO_DEVICE_REPORT_KEYS
        )
    }
    legacy_device_report.update(
        {
            "schema_version": 2,
            "authority": "canonical_component_face",
            "new_owned_claim_component_count": 3,
            "final_active_component_count": 5,
            "final_owned_component_count": 3,
            "final_external_exact_component_count": 2,
            "final_hard_component_count": 2,
            "final_soft_component_count": 3,
            "final_active_storage_row_count": 4,
            "final_active_x_component_count": 2,
            "final_active_y_component_count": 2,
            "final_active_z_component_count": 1,
            "primary_region_active_component_count": 1,
            "secondary_region_active_component_count": 1,
            "other_region_active_component_count": 1,
            "unassigned_region_active_component_count": 2,
            "max_abs_claim_target_mps": 4.0,
            "max_abs_committed_target_mps": 4.0,
            "min_active_pressure_mobility": 0.0,
            "max_active_pressure_mobility": 1.0,
            "min_active_enforcement_weight": 0.25,
            "max_active_enforcement_weight": 1.0,
            "actual_geometry_claim_count": 3,
        }
    )
    terminal_row.update(
        {
            "hibm_velocity_dirichlet_authority": "canonical",
            "hibm_velocity_dirichlet_ledger_generation": 133,
            "hibm_velocity_dirichlet_authority_registered": True,
            "hibm_velocity_dirichlet_authority_sealed": True,
            "canonical_velocity_dirichlet_report": legacy_device_report,
        }
    )

    payload = solid_mpm_fsi_runner._preflow_report_snapshot_payload(
        report,
        _snapshot_health_config(),
    )

    restored_device_report = payload["preflow_history"][-1][
        "canonical_velocity_dirichlet_report"
    ]
    assert restored_device_report["schema_version"] == 2
    assert extension_keys.isdisjoint(restored_device_report)


def _with_interface_covered_cartesian_pockets(
    row: dict[str, object],
    *,
    remaining_cell_count: int = 524,
    raw_component_count: int = 272,
    compact_component_count: int | None = None,
) -> dict[str, object]:
    compact_count = (
        raw_component_count
        if compact_component_count is None
        else compact_component_count
    )
    return {
        **row,
        "hibm_preassembly_remaining_unreached_cell_count": remaining_cell_count,
        "flow_projection_report": {
            **row["flow_projection_report"],
            "unreached_cells_with_interface_diagonal": remaining_cell_count,
            "unreached_cells_with_interface_coupling": remaining_cell_count,
            "cg_unreached_component_count": compact_count,
            "cg_unreached_component_raw_count": raw_component_count,
            "unreached_components_with_interface_hits": compact_count,
        },
    }


def test_snapshot_accepts_interface_coupled_cartesian_unreached_cells_when_exact_graph_is_healthy():
    report = _healthy_preflow_report()
    report["preflow_history"][-1] = _with_interface_covered_cartesian_pockets(
        report["preflow_history"][-1]
    )

    payload = solid_mpm_fsi_runner._preflow_report_snapshot_payload(
        report,
        _snapshot_health_config(),
    )

    assert payload["preflow_steps_completed"] == 1
    assert payload["preflow_history"][-1][
        "hibm_preassembly_remaining_unreached_cell_count"
    ] == 524


@pytest.mark.parametrize(
    "override",
    (
        {"pressure_nullspace_compatibility_measured": False},
        {"pressure_nullspace_component_labels_converged": False},
        {"pressure_nullspace_component_overflow": True},
        {"pressure_nullspace_incompatible_component_count": 1},
        {"pressure_nullspace_component_count": 1},
    ),
)
def test_snapshot_rejects_unhealthy_final_exact_pressure_graph(override):
    report = _healthy_preflow_report()
    projection_report = report["preflow_history"][-1]["flow_projection_report"]
    projection_report.update(override)

    with pytest.raises(ValueError, match="numerical health gate"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            report,
            _snapshot_health_config(),
        )


@pytest.mark.parametrize(
    "override",
    (
        {"pressure_outlet_operator_graph_prepared": False},
        {"pressure_interface_matrix_active": False},
        {"pressure_interface_matrix_row_invalid_count": 1},
        {"pressure_interface_matrix_row_overflow_count": 1},
    ),
)
def test_snapshot_nonzero_cartesian_pockets_require_valid_exact_interface_graph(
    override,
):
    report = _healthy_preflow_report()
    final_row = _with_interface_covered_cartesian_pockets(
        report["preflow_history"][-1]
    )
    report["preflow_history"][-1] = final_row
    final_row["flow_projection_report"].update(override)

    with pytest.raises(ValueError, match="numerical health gate"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            report,
            _snapshot_health_config(),
        )


@pytest.mark.parametrize(
    "override",
    (
        {"unreached_cells_with_interface_diagonal": 523},
        {"unreached_cells_with_interface_coupling": 523},
        {"cg_unreached_component_count": 0},
        {"cg_unreached_component_raw_count": 0},
        {"unreached_components_with_interface_hits": 271},
    ),
)
def test_snapshot_nonzero_cartesian_pockets_require_complete_interface_coverage(
    override,
):
    report = _healthy_preflow_report()
    final_row = _with_interface_covered_cartesian_pockets(
        report["preflow_history"][-1]
    )
    report["preflow_history"][-1] = {
        **final_row,
        "flow_projection_report": {
            **final_row["flow_projection_report"],
            **override,
        },
    }

    with pytest.raises(ValueError, match="numerical health gate"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            report,
            _snapshot_health_config(),
        )


@pytest.mark.parametrize("bad_value", (False, 0.5, "0"))
@pytest.mark.parametrize(
    "scope, key",
    (
        ("row", "hibm_preassembly_remaining_unreached_cell_count"),
        ("projection", "pressure_nullspace_component_count"),
        ("projection", "pressure_nullspace_incompatible_component_count"),
        ("projection", "pressure_interface_matrix_row_invalid_count"),
        ("projection", "pressure_interface_matrix_row_overflow_count"),
        ("projection", "unreached_cells_with_interface_diagonal"),
        ("projection", "unreached_cells_with_interface_coupling"),
        ("projection", "cg_unreached_component_count"),
        ("projection", "cg_unreached_component_raw_count"),
        ("projection", "unreached_components_with_interface_hits"),
    ),
)
def test_snapshot_pressure_health_counts_require_strict_integers(
    scope,
    key,
    bad_value,
):
    report = _healthy_preflow_report()
    final_row = _with_interface_covered_cartesian_pockets(
        report["preflow_history"][-1]
    )
    if scope == "row":
        final_row = {**final_row, key: bad_value}
    else:
        final_row = {
            **final_row,
            "flow_projection_report": {
                **final_row["flow_projection_report"],
                key: bad_value,
            },
        }
    report["preflow_history"][-1] = final_row

    with pytest.raises(ValueError, match="numerical health gate"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            report,
            _snapshot_health_config(),
        )


def test_snapshot_accepts_capacity_compacted_fully_interface_covered_pockets():
    report = _healthy_preflow_report()
    report["preflow_history"][-1] = _with_interface_covered_cartesian_pockets(
        report["preflow_history"][-1],
        remaining_cell_count=1296,
        raw_component_count=1296,
        compact_component_count=768,
    )

    payload = solid_mpm_fsi_runner._preflow_report_snapshot_payload(
        report,
        _snapshot_health_config(),
    )

    assert payload["preflow_steps_completed"] == 1


def test_snapshot_accepts_compatible_projected_exact_nullspace_components():
    report = _healthy_preflow_report()
    projection_report = report["preflow_history"][-1]["flow_projection_report"]
    projection_report.update(
        {
            "pressure_nullspace_policy": (
                "outlet_disconnected_fv_cg_operator_componentwise_zero_mean"
            ),
            "pressure_nullspace_component_count": 2,
            "pressure_nullspace_componentwise_projection_applied": True,
        }
    )

    payload = solid_mpm_fsi_runner._preflow_report_snapshot_payload(
        report,
        _snapshot_health_config(),
    )

    assert payload["preflow_steps_completed"] == 1


def test_snapshot_outlet_components_require_exact_componentwise_projection():
    report = _healthy_preflow_report()
    final_row = _with_interface_covered_cartesian_pockets(
        report["preflow_history"][-1]
    )
    report["preflow_history"][-1] = {
        **final_row,
        "flow_projection_report": {
            **final_row["flow_projection_report"],
            "pressure_nullspace_policy": (
                "outlet_disconnected_fv_cg_operator_componentwise_zero_mean"
            ),
            "pressure_nullspace_component_count": 1,
            "pressure_nullspace_componentwise_projection_applied": False,
            "pressure_nullspace_zero_mean_projection_applied": True,
        },
    }

    with pytest.raises(ValueError, match="numerical health gate"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            report,
            _snapshot_health_config(),
        )


def test_velocity_dirichlet_report_fields_preserve_reconstruction_diagnostics():
    report = SimpleNamespace(
        active_velocity_dirichlet_rows=17,
        primary_region_active_rows=8,
        secondary_region_active_rows=7,
        other_region_active_rows=1,
        unassigned_region_active_rows=1,
        max_abs_velocity_mps=3.0,
        raw_reconstructed_max_abs_velocity_mps=4.0,
        boundary_velocity_only_row_count=5,
        invalid_reconstruction_row_count=2,
        invalid_no_fluid_sample_row_count=1,
        invalid_nonpositive_gap_row_count=1,
        invalid_node_behind_boundary_row_count=0,
        invalid_node_beyond_interior_row_count=0,
        narrow_gap_boundary_velocity_row_count=3,
        relocated_row_count=4,
        relocation_merged_row_count=2,
        relocation_blocked_row_count=1,
        min_projection_weight=0.125,
        max_projection_weight=1.0,
    )

    fields = solid_mpm_fsi_runner._hibm_velocity_dirichlet_report_fields(report)

    assert fields == {
        "hibm_velocity_dirichlet_active_rows": 17,
        "hibm_velocity_dirichlet_primary_region_active_rows": 8,
        "hibm_velocity_dirichlet_secondary_region_active_rows": 7,
        "hibm_velocity_dirichlet_other_region_active_rows": 1,
        "hibm_velocity_dirichlet_unassigned_region_active_rows": 1,
        "hibm_velocity_dirichlet_max_abs_velocity_mps": 3.0,
        "hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": 4.0,
        "hibm_velocity_dirichlet_boundary_velocity_only_rows": 5,
        "hibm_velocity_dirichlet_invalid_reconstruction_count": 2,
        "hibm_velocity_dirichlet_invalid_no_fluid_sample_count": 1,
        "hibm_velocity_dirichlet_invalid_nonpositive_gap_count": 1,
        "hibm_velocity_dirichlet_invalid_node_behind_boundary_count": 0,
        "hibm_velocity_dirichlet_invalid_node_beyond_interior_count": 0,
        "hibm_velocity_dirichlet_narrow_gap_count": 3,
        "hibm_velocity_dirichlet_relocated_rows": 4,
        "hibm_velocity_dirichlet_relocation_merged_rows": 2,
        "hibm_velocity_dirichlet_relocation_blocked_rows": 1,
        "hibm_velocity_dirichlet_min_projection_weight": 0.125,
        "hibm_velocity_dirichlet_max_projection_weight": 1.0,
    }


def test_flow_advance_publishes_terminal_consistency_velocity_diagnostics():
    project_soft_row_flags: list[bool] = []
    pre_predictor_report = _healthy_hibm_velocity_report(active_rows=1)
    main_projection_report = {
        **_healthy_hibm_velocity_report(active_rows=3),
        "hibm_velocity_dirichlet_relocated_rows": 1,
    }
    terminal_consistency_report = {
        **_healthy_hibm_velocity_report(active_rows=3),
        "hibm_velocity_dirichlet_max_abs_velocity_mps": 0.75,
        "hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": 1.0,
        "hibm_velocity_dirichlet_relocated_rows": 1,
    }
    boundary_reports = iter(
        (
            pre_predictor_report,
            main_projection_report,
            terminal_consistency_report,
        )
    )

    def assemble_boundary(*_args, **_kwargs):
        return next(boundary_reports)

    def project_flow(*_args, **_kwargs):
        project_soft_row_flags.append(
            bool(
                _kwargs.get(
                    "velocity_dirichlet_soft_rows_already_applied",
                    False,
                )
            )
        )
        return {
            "local_velocity_peak_mps": 0.0,
            "fluid_speed_p99_mps": 0.0,
            "fluid_speed_p999_mps": 0.0,
            "pressure_min_pa": 0.0,
            "pressure_max_pa": 0.0,
            "projection_report": {},
        }

    config = SimpleNamespace(
        flow_solid_boundary_mode="hibm_sharp_marker_rows",
        flow_driver_mode="projection_only",
        flow_post_dirichlet_consistency_projection_iterations=1,
        flow_hibm_sharp_interpolate_velocity_rows=False,
        flow_projection_iterations=1,
        flow_cg_tolerance=1.0e-8,
    )
    fluid = SimpleNamespace(
        clear_volume_source=lambda: None,
        **_exact_velocity_dirichlet_ledger_fake_methods(),
    )
    no_slip_report = {
        "hibm_no_slip_report": {},
        "hibm_no_slip_valid_marker_count": 1,
        "hibm_no_slip_invalid_marker_count": 0,
        "hibm_no_slip_max_residual_mps": 0.0,
        "hibm_no_slip_l2_residual_mps": 0.0,
    }

    with (
        patch.object(
            solid_mpm_fsi_runner,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
            side_effect=assemble_boundary,
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_hibm_pre_projection_velocity_projector_from_cache",
            return_value=object(),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_project_current_flow",
            side_effect=project_flow,
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_sample_hibm_no_slip_report",
            return_value=no_slip_report,
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_require_hibm_joint_qp_convergence",
            return_value=None,
        ),
    ):
        report = solid_mpm_fsi_runner._flow_advance_current_step(
            fluid,
            config,
            markers=object(),
            flow_phase="preflow",
            step_index_local=1,
            step_index_global=1,
            preflow_history=[],
            reset_pressure=False,
        )

    assert report["hibm_velocity_dirichlet_active_rows"] == 3
    assert report["hibm_velocity_dirichlet_max_abs_velocity_mps"] == 0.75
    assert (
        report["hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps"]
        == 1.0
    )
    assert report["hibm_velocity_dirichlet_relocated_rows"] == 1
    assert project_soft_row_flags == [False, True]


def test_sustained_boundary_predictor_applies_velocity_only_soft_rows_once():
    project_soft_row_flags: list[bool] = []
    boundary_reports = iter(
        (
            _healthy_hibm_velocity_report(active_rows=3),
            _healthy_hibm_velocity_report(active_rows=3),
            _healthy_hibm_velocity_report(active_rows=3),
        )
    )

    def project_flow(*_args, **kwargs):
        project_soft_row_flags.append(
            bool(kwargs.get("velocity_dirichlet_soft_rows_already_applied", False))
        )
        return {
            "local_velocity_peak_mps": 0.0,
            "fluid_speed_p99_mps": 0.0,
            "fluid_speed_p999_mps": 0.0,
            "pressure_min_pa": 0.0,
            "pressure_max_pa": 0.0,
            "projection_report": {},
        }

    config = SimpleNamespace(
        flow_solid_boundary_mode="hibm_sharp_marker_rows",
        flow_driver_mode="sustained_boundary_predictor",
        flow_post_dirichlet_consistency_projection_iterations=1,
        flow_hibm_sharp_interpolate_velocity_rows=False,
        flow_projection_iterations=1,
        flow_cg_tolerance=1.0e-8,
        flow_advection_scheme="euler",
        flow_predictor_substeps=1,
        flow_predictor_no_slip_domain_walls=(),
        dt_s=5.0e-4,
        air_viscosity_pa_s=1.8e-5,
        air_density_kgm3=1.225,
    )
    fluid = SimpleNamespace(
        clear_volume_source=lambda: None,
        apply_velocity_dirichlet_boundary_rows=lambda **_kwargs: None,
        predict=lambda **_kwargs: None,
        **_exact_velocity_dirichlet_ledger_fake_methods(),
    )
    no_slip_report = {
        "hibm_no_slip_report": {},
        "hibm_no_slip_valid_marker_count": 1,
        "hibm_no_slip_invalid_marker_count": 0,
        "hibm_no_slip_max_residual_mps": 0.0,
        "hibm_no_slip_l2_residual_mps": 0.0,
    }

    with (
        patch.object(
            solid_mpm_fsi_runner,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
            side_effect=lambda *_args, **_kwargs: next(boundary_reports),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_hibm_pre_projection_velocity_projector_from_cache",
            return_value=object(),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_refresh_zmax_inlet_boundary",
            return_value={},
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_project_current_flow",
            side_effect=project_flow,
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_sample_hibm_no_slip_report",
            return_value=no_slip_report,
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_require_hibm_joint_qp_convergence",
            return_value=None,
        ),
    ):
        solid_mpm_fsi_runner._flow_advance_current_step(
            fluid,
            config,
            markers=object(),
            flow_phase="fsi",
            step_index_local=1,
            step_index_global=1,
            preflow_history=[],
            reset_pressure=False,
        )

    assert project_soft_row_flags == [True, True]


@pytest.mark.parametrize(
    ("missing_key", "override"),
    (
        ("hibm_sharp_marker_boundary_topology_reused", {}),
        (None, {"hibm_sharp_marker_boundary_topology_reused": False}),
        ("hibm_preassembly_topology_mutated", {}),
        (None, {"hibm_preassembly_topology_mutated": True}),
    ),
)
def test_sustained_boundary_predictor_requires_explicit_topology_reuse_before_main_skip(
    missing_key,
    override,
):
    pre_predictor_report = _healthy_hibm_velocity_report(active_rows=3)
    projection_report = {
        **_healthy_hibm_velocity_report(active_rows=3),
        **override,
    }
    if missing_key is not None:
        del projection_report[missing_key]
    boundary_reports = iter((pre_predictor_report, projection_report))

    config = SimpleNamespace(
        flow_solid_boundary_mode="hibm_sharp_marker_rows",
        flow_driver_mode="sustained_boundary_predictor",
        flow_post_dirichlet_consistency_projection_iterations=0,
        flow_hibm_sharp_interpolate_velocity_rows=False,
        flow_projection_iterations=1,
        flow_cg_tolerance=1.0e-8,
        flow_advection_scheme="euler",
        flow_predictor_substeps=1,
        flow_predictor_no_slip_domain_walls=(),
        dt_s=5.0e-4,
        air_viscosity_pa_s=1.8e-5,
        air_density_kgm3=1.225,
    )
    fluid = SimpleNamespace(
        clear_volume_source=lambda: None,
        apply_velocity_dirichlet_boundary_rows=lambda **_kwargs: None,
        predict=lambda **_kwargs: None,
        **_exact_velocity_dirichlet_ledger_fake_methods(),
    )

    with (
        patch.object(
            solid_mpm_fsi_runner,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
            side_effect=lambda *_args, **_kwargs: next(boundary_reports),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_hibm_pre_projection_velocity_projector_from_cache",
            return_value=object(),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_refresh_zmax_inlet_boundary",
            return_value={},
        ),
        patch.object(solid_mpm_fsi_runner, "_project_current_flow") as project,
    ):
        with pytest.raises(RuntimeError, match="topology"):
            solid_mpm_fsi_runner._flow_advance_current_step(
                fluid,
                config,
                markers=object(),
                flow_phase="fsi",
                step_index_local=1,
                step_index_global=1,
                preflow_history=[],
                reset_pressure=False,
            )

    project.assert_not_called()


@pytest.mark.parametrize(
    "missing_field",
    (
        "velocity_dirichlet_boundary_enforcement_weight",
        "velocity_dirichlet_boundary_marker_region_id",
        "velocity_dirichlet_boundary_hard_fixed_component_mask",
        "velocity_dirichlet_boundary_external_exact_component_mask",
        "velocity_dirichlet_boundary_owned_row",
    ),
)
def test_host_feedback_fallback_requires_complete_boundary_ledger_before_clearing(
    missing_field,
):
    fluid = _fake_fluid()
    clear_calls: list[bool] = []
    fluid.clear_velocity_constraints = lambda: clear_calls.append(True)
    delattr(fluid, missing_field)

    with pytest.raises(RuntimeError, match=missing_field):
        solid_mpm_fsi_runner._apply_marker_feedback_to_fluid_host_fallback(
            SimpleNamespace(marker_count=0),
            fluid,
            SimpleNamespace(preserve_marker_velocity_constraints=True),
            feedback_available=False,
            previous_feedback_constraint_cells=set(),
        )

    assert clear_calls == []


def test_host_feedback_fallback_writes_back_every_boundary_ledger_field():
    fluid = _fake_fluid()
    fluid.clear_velocity_constraints = lambda: None
    ledger_field_names = (
        "velocity_dirichlet_boundary_active",
        "velocity_dirichlet_boundary_value_mps",
        "velocity_dirichlet_boundary_projection_weight",
        "velocity_dirichlet_boundary_enforcement_weight",
        "velocity_dirichlet_boundary_marker_region_id",
        "velocity_dirichlet_boundary_hard_fixed_component_mask",
        "velocity_dirichlet_boundary_external_exact_component_mask",
        "velocity_dirichlet_boundary_owned_row",
    )

    solid_mpm_fsi_runner._apply_marker_feedback_to_fluid_host_fallback(
        SimpleNamespace(marker_count=0),
        fluid,
        SimpleNamespace(preserve_marker_velocity_constraints=True),
        feedback_available=False,
        previous_feedback_constraint_cells={(0, 0, 0)},
    )

    assert {
        name: getattr(fluid, name).from_numpy_calls for name in ledger_field_names
    } == {name: 1 for name in ledger_field_names}


@pytest.mark.parametrize(
    ("override", "reason_fragment"),
    (
        ({"hibm_velocity_dirichlet_invalid_reconstruction_count": 1}, "invalid"),
        ({"hibm_velocity_dirichlet_narrow_gap_count": 1}, "narrow"),
        ({"hibm_velocity_dirichlet_min_projection_weight": np.nan}, "weight"),
        ({"hibm_velocity_dirichlet_max_projection_weight": 1.01}, "weight"),
        (
            {
                "hibm_velocity_dirichlet_min_projection_weight": 0.75,
                "hibm_velocity_dirichlet_max_projection_weight": 0.5,
            },
            "weight",
        ),
        ({"hibm_velocity_dirichlet_boundary_velocity_only_rows": 18}, "row count"),
        ({"hibm_velocity_dirichlet_relocation_merged_rows": 1}, "relocation"),
        ({"hibm_velocity_dirichlet_relocation_blocked_rows": 1}, "relocation"),
    ),
)
def test_velocity_dirichlet_health_failure_rejects_unphysical_reports(
    override,
    reason_fragment,
):
    report = {
        **_healthy_hibm_velocity_report(
            active_rows=17,
            boundary_velocity_only_rows=5,
        ),
        "hibm_velocity_dirichlet_min_projection_weight": 0.125,
        "hibm_velocity_dirichlet_max_projection_weight": 1.0,
        **override,
    }

    failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(report)

    assert failure is not None
    assert reason_fragment in failure


def test_velocity_only_consistency_reuse_rejects_changed_row_topology():
    baseline = _healthy_hibm_velocity_report(active_rows=3)
    cases = (
        (
            {"hibm_sharp_marker_boundary_topology_reused": False},
            "not reused",
        ),
        ({"hibm_preassembly_topology_mutated": True}, "mutated"),
        ({"hibm_velocity_dirichlet_active_rows": 4}, "active_rows"),
        (
            {"hibm_velocity_dirichlet_min_projection_weight": 0.25},
            "min_projection_weight",
        ),
    )

    for override, reason in cases:
        with pytest.raises(RuntimeError, match=reason):
            solid_mpm_fsi_runner._require_velocity_only_consistency_row_reuse(
                baseline,
                {**baseline, **override},
                context="test consistency",
            )


@pytest.mark.parametrize(
    "mismatch_field",
    (
        "active_indices",
        "projection_weight",
        "enforcement_weight",
        "target_velocity_mps",
        "owned_row",
        "hard_fixed_component_mask",
        "external_exact_component_mask",
    ),
)
def test_velocity_only_consistency_reuse_rejects_exact_row_ledger_mismatch(
    mismatch_field,
):
    baseline = _healthy_hibm_velocity_report(active_rows=3)
    consistency = {
        **baseline,
        "hibm_velocity_dirichlet_row_ledger_matches_reference": False,
        "hibm_velocity_dirichlet_row_ledger_mismatch_rows": 1,
        "hibm_velocity_dirichlet_row_ledger_first_mismatch_field": mismatch_field,
    }

    with pytest.raises(RuntimeError, match="row ledger"):
        solid_mpm_fsi_runner._require_velocity_only_consistency_row_reuse(
            baseline,
            consistency,
            context="test exact row ledger reuse",
        )


@pytest.mark.parametrize(
    ("reference_generation", "consistency_generation"),
    ((0, 0), (1, 2)),
)
def test_velocity_only_consistency_reuse_requires_same_nonzero_snapshot_generation(
    reference_generation,
    consistency_generation,
):
    baseline = {
        **_healthy_hibm_velocity_report(active_rows=3),
        "hibm_velocity_dirichlet_row_ledger_snapshot_generation": (
            reference_generation
        ),
    }
    consistency = {
        **baseline,
        "hibm_velocity_dirichlet_row_ledger_snapshot_generation": (
            consistency_generation
        ),
    }

    with pytest.raises(RuntimeError, match="snapshot generation"):
        solid_mpm_fsi_runner._require_velocity_only_consistency_row_reuse(
            baseline,
            consistency,
            context="test exact row ledger reuse",
        )


@pytest.mark.parametrize(
    "missing_key",
    (
        "hibm_velocity_dirichlet_row_ledger_snapshot_generation",
        "hibm_velocity_dirichlet_row_ledger_matches_reference",
        "hibm_velocity_dirichlet_row_ledger_mismatch_rows",
    ),
)
def test_velocity_only_consistency_reuse_requires_complete_row_ledger_diagnostics(
    missing_key,
):
    baseline = _healthy_hibm_velocity_report(active_rows=3)
    consistency = dict(baseline)
    del consistency[missing_key]

    with pytest.raises(RuntimeError, match="row ledger"):
        solid_mpm_fsi_runner._require_velocity_only_consistency_row_reuse(
            baseline,
            consistency,
            context="test exact row ledger reuse",
        )


def test_velocity_dirichlet_health_allows_disabled_or_boundary_only_soft_rows():
    assert (
        solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            {"hibm_sharp_marker_boundary_enabled": False}
        )
        is None
    )
    assert (
        solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            _healthy_hibm_velocity_report(
                active_rows=17,
                boundary_velocity_only_rows=17,
            )
        )
        is None
    )


@pytest.mark.parametrize(
    "missing_key",
    solid_mpm_fsi_runner.HIBM_VELOCITY_DIRICHLET_REPORT_KEYS,
)
def test_strict_velocity_health_requires_complete_report_keys(missing_key):
    report = _healthy_hibm_velocity_report()
    del report[missing_key]

    failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(report)

    assert failure is not None
    assert missing_key in failure


@pytest.mark.parametrize(
    ("override", "reason_fragment"),
    (
        ({"hibm_velocity_dirichlet_primary_region_active_rows": 0}, "region"),
        ({"hibm_velocity_dirichlet_relocated_rows": -1}, "relocat"),
        ({"hibm_velocity_dirichlet_relocation_merged_rows": -1}, "relocat"),
        ({"hibm_velocity_dirichlet_relocation_blocked_rows": -1}, "relocat"),
        ({"hibm_velocity_dirichlet_max_abs_velocity_mps": np.nan}, "velocity"),
        ({"hibm_velocity_dirichlet_max_abs_velocity_mps": -1.0}, "velocity"),
        (
            {
                "hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": (
                    np.inf
                )
            },
            "velocity",
        ),
        (
            {
                "hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": (
                    -1.0
                )
            },
            "velocity",
        ),
    ),
)
def test_strict_velocity_health_validates_region_relocation_and_velocity_extrema(
    override,
    reason_fragment,
):
    report = {**_healthy_hibm_velocity_report(), **override}

    failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(report)

    assert failure is not None
    assert reason_fragment in failure


def test_snapshot_and_stationary_certificate_require_complete_velocity_diagnostics():
    snapshot_report = _healthy_preflow_report()
    del snapshot_report["preflow_history"][-1][
        "hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps"
    ]
    with pytest.raises(ValueError, match="numerical health gate"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            snapshot_report,
            _snapshot_health_config(),
        )

    stationary_report = _windowed_stationary_report()
    del stationary_report["preflow_history"][-1][
        "hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps"
    ]
    certificate = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        stationary_report["preflow_history"],
        _windowed_snapshot_config(),
    )
    assert certificate["stationary"] is False
    assert certificate["reason"] == "physical_guard_failed"
    assert certificate["velocity_dirichlet_health_failure"]


@pytest.mark.parametrize(
    "field_name",
    (
        "hibm_velocity_dirichlet_invalid_reconstruction_count",
        "hibm_velocity_dirichlet_narrow_gap_count",
    ),
)
def test_preflow_snapshot_rejects_unhealthy_velocity_reconstruction(field_name):
    report = _healthy_preflow_report()
    report["preflow_history"][-1][field_name] = 1

    with pytest.raises(ValueError, match="numerical health gate"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            report,
            _snapshot_health_config(),
        )


def test_preflow_snapshot_final_health_rejection_carries_complete_history():
    report = _healthy_preflow_report()
    template = report["preflow_history"][0]
    report["preflow_history"] = [
        {
            **deepcopy(template),
            "preflow_step": step,
            "hibm_no_slip_max_residual_mps": (
                4.575837135314941 if step == 40 else 0.1
            ),
        }
        for step in range(1, 41)
    ]
    report["preflow_steps_completed"] = 40

    with pytest.raises(
        ValueError,
        match="final no-slip residual exceeds the configured limit",
    ) as exc_info:
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            report,
            _snapshot_health_config(),
        )

    diagnostics = getattr(exc_info.value, "diagnostics", {})
    assert "preflow_snapshot_rejection" in diagnostics
    rejection = diagnostics["preflow_snapshot_rejection"]
    assert rejection["status"] == "rejected"
    assert rejection["gate"] == "final_no_slip_residual"
    assert rejection["preflow_steps_completed"] == 40
    assert len(rejection["preflow_history"]) == 40
    assert rejection["preflow_history"][0]["preflow_step"] == 1
    assert rejection["preflow_history"][-1]["preflow_step"] == 40
    assert rejection["terminal_preflow_diagnostics"] == rejection[
        "preflow_history"
    ][-1]
    assert rejection["terminal_preflow_diagnostics"][
        "hibm_no_slip_max_residual_mps"
    ] == pytest.approx(4.575837135314941)


def test_coupling_ready_snapshot_rejects_unmeasured_traction():
    config = SimpleNamespace(
        **{
            **vars(_snapshot_health_config()),
            "preflow_traction_readiness_mode": "coupling_ready",
        }
    )

    with pytest.raises(ValueError, match="traction readiness"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            _healthy_preflow_report(),
            config,
        )


def test_flow_only_snapshot_rejects_partially_evaluated_traction():
    report = _healthy_preflow_report()
    report["preflow_history"][-1]["stress_valid_marker_count"] = 1
    report["preflow_history"][-1]["stress_invalid_marker_count"] = 1

    with pytest.raises(ValueError, match="traction readiness"):
        solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            report,
            _snapshot_health_config(),
        )


def test_runner_snapshot_converts_optional_nonfinite_diagnostics_to_null(tmp_path):
    identity = PreflowSnapshotIdentity(
        config_sha256="0" * 64,
        source_sha256="1" * 64,
        geometry_sha256="2" * 64,
    )
    config = _snapshot_health_config()
    report = _healthy_preflow_report()
    projection_report = report["preflow_history"][0]["flow_projection_report"]
    optional_nan_fields = (
        "zmin_unreached_source_centroid_x_m",
        "zmin_unreached_source_centroid_y_m",
        "zmin_unreached_source_centroid_z_m",
        "zmin_unreached_source_min_x_m",
        "zmin_unreached_source_min_y_m",
        "zmin_unreached_source_min_z_m",
        "zmin_unreached_source_max_x_m",
        "zmin_unreached_source_max_y_m",
        "zmin_unreached_source_max_z_m",
    )
    projection_report.update({field_name: np.nan for field_name in optional_nan_fields})
    prefix = tmp_path / "nonfinite_optional_diagnostic"
    source = _fake_fluid()

    solid_mpm_fsi_runner._write_fixed_solid_preflow_snapshot(
        path=prefix,
        report=report,
        markers=object(),
        fluid=source,
        solid=object(),
        config=config,
        identity=identity,
    )
    target = _fake_fluid()
    _zero_snapshot_fields(target)
    restored = solid_mpm_fsi_runner._restore_fixed_solid_preflow_snapshot(
        path=prefix,
        markers=object(),
        fluid=target,
        solid=object(),
        config=config,
        expected_identity=identity,
    )

    restored_projection_report = restored["preflow_history"][0][
        "flow_projection_report"
    ]
    for field_name in optional_nan_fields:
        assert np.isnan(projection_report[field_name])
        assert restored_projection_report[field_name] is None
    json.dumps(restored, allow_nan=False)
    for name, expected in solid_mpm_fsi_runner._capture_preflow_snapshot_fields(
        source
    ).items():
        np.testing.assert_array_equal(getattr(target, name).to_numpy(), expected)


def test_runner_snapshot_still_rejects_infinite_diagnostics(tmp_path):
    report = _healthy_preflow_report()
    report["preflow_history"][0]["flow_projection_report"][
        "zmin_unreached_source_centroid_x_m"
    ] = np.inf

    with pytest.raises(PreflowSnapshotValidationError, match="non-finite"):
        solid_mpm_fsi_runner._write_fixed_solid_preflow_snapshot(
            path=tmp_path / "infinite_diagnostic",
            report=report,
            markers=object(),
            fluid=_fake_fluid(),
            solid=object(),
            config=_snapshot_health_config(),
            identity=PreflowSnapshotIdentity(
                config_sha256="0" * 64,
                source_sha256="1" * 64,
                geometry_sha256="2" * 64,
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "flow_projection_pressure_solve_failed",
        "flow_projection_pressure_projection_physical_failure",
        "hibm_preassembly_topology_mutated",
    ),
)
def test_runner_snapshot_nan_health_failure_flags_fail_closed(tmp_path, field_name):
    report = _healthy_preflow_report()
    report["preflow_history"][0][field_name] = np.nan

    with pytest.raises(ValueError, match="numerical health gate"):
        solid_mpm_fsi_runner._write_fixed_solid_preflow_snapshot(
            path=tmp_path / field_name,
            report=report,
            markers=object(),
            fluid=_fake_fluid(),
            solid=object(),
            config=_snapshot_health_config(),
            identity=PreflowSnapshotIdentity(
                config_sha256="0" * 64,
                source_sha256="1" * 64,
                geometry_sha256="2" * 64,
            ),
        )


def test_windowed_snapshot_round_trip_recomputes_stationary_certificate(tmp_path):
    identity = PreflowSnapshotIdentity(
        config_sha256="0" * 64,
        source_sha256="1" * 64,
        geometry_sha256="2" * 64,
    )
    config = _windowed_snapshot_config()
    report = _windowed_stationary_report()
    prefix = tmp_path / "windowed"

    solid_mpm_fsi_runner._write_fixed_solid_preflow_snapshot(
        path=prefix,
        report=report,
        markers=object(),
        fluid=_fake_fluid(),
        solid=object(),
        config=config,
        identity=identity,
    )
    restored = solid_mpm_fsi_runner._restore_fixed_solid_preflow_snapshot(
        path=prefix,
        markers=object(),
        fluid=_fake_fluid(),
        solid=object(),
        config=config,
        expected_identity=identity,
    )

    certificate = restored["preflow_stationary_certificate"]
    assert certificate["stationary"] is True
    assert certificate["reason"] == "stationary"
    assert certificate["consecutive_windows_passed"] == 2
    assert certificate["traction_readiness"] == "not_evaluated"
    assert certificate["marker_force_metric_evaluated"] is False
    assert certificate["excluded_window_metrics"] == ["marker_force_relative_span"]


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_match"),
    (
        ("preflow_converged", False, "preflow_converged"),
        ("preflow_stop_reason", "max_steps", "windowed_stationary"),
    ),
)
def test_windowed_snapshot_save_requires_converged_stop_certificate(
    tmp_path,
    field_name,
    invalid_value,
    error_match,
):
    report = _windowed_stationary_report()
    report[field_name] = invalid_value

    with pytest.raises(ValueError, match=error_match):
        solid_mpm_fsi_runner._write_fixed_solid_preflow_snapshot(
            path=tmp_path / field_name,
            report=report,
            markers=object(),
            fluid=_fake_fluid(),
            solid=object(),
            config=_windowed_snapshot_config(),
            identity=PreflowSnapshotIdentity(
                config_sha256="0" * 64,
                source_sha256="1" * 64,
                geometry_sha256="2" * 64,
            ),
        )


def test_windowed_snapshot_save_revalidates_terminal_stationary_windows(tmp_path):
    report = _windowed_stationary_report()
    report["preflow_history"][-1]["local_velocity_peak_mps"] = 100.0

    with pytest.raises(ValueError, match="stationary window"):
        solid_mpm_fsi_runner._write_fixed_solid_preflow_snapshot(
            path=tmp_path / "drifted",
            report=report,
            markers=object(),
            fluid=_fake_fluid(),
            solid=object(),
            config=_windowed_snapshot_config(),
            identity=PreflowSnapshotIdentity(
                config_sha256="0" * 64,
                source_sha256="1" * 64,
                geometry_sha256="2" * 64,
            ),
        )


def test_windowed_snapshot_load_revalidates_stored_terminal_windows(tmp_path):
    identity = PreflowSnapshotIdentity(
        config_sha256="0" * 64,
        source_sha256="1" * 64,
        geometry_sha256="2" * 64,
    )
    invalid_report = _windowed_stationary_report()
    invalid_report["preflow_history"][-1]["local_velocity_peak_mps"] = 100.0
    prefix = tmp_path / "untrusted_history"
    save_preflow_snapshot(
        prefix,
        PreflowSnapshot(
            fields=solid_mpm_fsi_runner._capture_preflow_snapshot_fields(
                _fake_fluid()
            ),
            identity=identity,
            history=invalid_report,
        ),
    )

    with pytest.raises(ValueError, match="stationary window"):
        solid_mpm_fsi_runner._restore_fixed_solid_preflow_snapshot(
            path=prefix,
            markers=object(),
            fluid=_fake_fluid(),
            solid=object(),
            config=_windowed_snapshot_config(),
            expected_identity=identity,
        )


@pytest.mark.parametrize(
    ("history_override", "error_match"),
    (
        ({"hibm_no_slip_valid_marker_count": 0}, "valid marker count"),
        ({"hibm_no_slip_max_residual_mps": 0.6}, "no-slip residual"),
    ),
)
def test_snapshot_health_gate_rejects_incomplete_or_slipping_interface(
    tmp_path,
    history_override,
    error_match,
):
    identity = PreflowSnapshotIdentity(
        config_sha256="0" * 64,
        source_sha256="1" * 64,
        geometry_sha256="2" * 64,
    )
    report = _healthy_preflow_report()
    report["preflow_history"][-1].update(history_override)

    with pytest.raises(ValueError, match=error_match):
        solid_mpm_fsi_runner._write_fixed_solid_preflow_snapshot(
            path=tmp_path / "preflow",
            report=report,
            markers=object(),
            fluid=_fake_fluid(),
            solid=object(),
            config=_snapshot_health_config(),
            identity=identity,
        )

    assert not (tmp_path / "preflow.json").exists()


def test_snapshot_path_conflict_is_rejected_before_cuda_runtime_construction():
    config = replace(
        selected_formulation_solver_config(step_count=1),
        preflow_snapshot_input_path="cache/input",
        preflow_snapshot_output_path="cache/output",
    )

    with (
        patch.object(solid_mpm_fsi_runner, "TaichiRuntimeConfig") as runtime,
        pytest.raises(ValueError, match="cannot both be set"),
    ):
        solid_mpm_fsi_runner.run_rectangular_solid_marker_mpm_fsi_smoke(
            case_id="snapshot-path-conflict",
            case_metadata={},
            boundary_conditions={},
            reference_results={},
            config=config,
        )

    runtime.assert_not_called()


def test_preflow_snapshot_config_hash_excludes_fsi_only_and_path_fields():
    base = selected_formulation_solver_config(step_count=5)
    material_and_output_change = replace(
        base,
        step_count=50,
        young_modulus_pa=2.0 * base.young_modulus_pa,
        solid_density_kgm3=1.5 * base.solid_density_kgm3,
        preflow_snapshot_input_path="cache/input.npz",
        preflow_snapshot_output_path="cache/output.npz",
    )

    assert solid_mpm_fsi_runner._preflow_snapshot_config_payload(
        base
    ) == solid_mpm_fsi_runner._preflow_snapshot_config_payload(
        material_and_output_change
    )
    assert solid_mpm_fsi_runner._preflow_snapshot_config_payload(
        base
    ) != solid_mpm_fsi_runner._preflow_snapshot_config_payload(
        replace(base, inlet_velocity_mps=11.0)
    )
    assert solid_mpm_fsi_runner._preflow_snapshot_config_payload(
        base
    ) != solid_mpm_fsi_runner._preflow_snapshot_config_payload(
        replace(base, traction_marker_face_offset_cells=0.51)
    )


def test_explicit_preflow_snapshot_input_skips_fresh_preflow():
    config = replace(
        selected_formulation_solver_config(step_count=5),
        preflow_snapshot_input_path="cache/preflow.npz",
    )
    restored = {
        "preflow_history": [{"preflow_step": step} for step in range(1, 41)],
        "preflow_steps_completed": 40,
        "preflow_status": "snapshot_loaded",
    }

    with (
        patch.object(
            solid_mpm_fsi_runner,
            "_preflow_snapshot_identity",
            return_value=object(),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_restore_fixed_solid_preflow_snapshot",
            return_value=restored,
        ) as restore_snapshot,
        patch.object(solid_mpm_fsi_runner, "_run_fixed_solid_preflow") as run_fresh,
    ):
        result = solid_mpm_fsi_runner._run_or_restore_fixed_solid_preflow(
            markers=object(),
            fluid=object(),
            solid=object(),
            config=config,
        )

    assert result is restored
    restore_snapshot.assert_called_once()
    run_fresh.assert_not_called()


def test_preflow_snapshot_output_is_written_only_after_successful_fresh_run():
    config = replace(
        selected_formulation_solver_config(step_count=5),
        preflow_snapshot_output_path="cache/preflow.npz",
    )
    fresh = {
        "preflow_history": [{"preflow_step": step} for step in range(1, 41)],
        "preflow_steps_completed": 40,
        "preflow_status": "max_steps",
    }

    with (
        patch.object(
            solid_mpm_fsi_runner,
            "_preflow_snapshot_identity",
            return_value=object(),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_run_fixed_solid_preflow",
            return_value=fresh,
        ) as run_fresh,
        patch.object(
            solid_mpm_fsi_runner,
            "_write_fixed_solid_preflow_snapshot",
        ) as write_snapshot,
    ):
        result = solid_mpm_fsi_runner._run_or_restore_fixed_solid_preflow(
            markers=object(),
            fluid=object(),
            solid=object(),
            config=config,
        )

    assert result is fresh
    run_fresh.assert_called_once()
    write_snapshot.assert_called_once()
