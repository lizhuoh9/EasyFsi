from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import benchmarks.official.solid_mpm_fsi_runner as solid_runner


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_fine_vs_fluent_2026-07-02"
    / "scripts"
    / "run_our_solver_vertical_flap.py"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_our_solver_vertical_flap_output_contract_test",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_config_accepts_preflow_only_and_explicit_full_multigrid() -> None:
    module = _load_runner_module()
    config = module._build_config(
        SimpleNamespace(
            steps=0,
            preflow_steps=1,
            pressure_pair_provider_mode="runtime_anchored_cell_pair",
            selected_anchor_markers_json=None,
            grid_nodes=(4, 16, 20),
            solid_particle_counts=(1, 16, 4),
            marker_count=12,
            flow_projection_iterations=32,
            flow_cg_preconditioner="fv_multigrid",
            flow_pressure_solve_failure_policy="raise",
            solid_substeps=8,
            flow_predictor_substeps=2,
            young_modulus_pa=None,
            hibm_search_radius_m=None,
        )
    )

    assert config.step_count == 0
    assert config.preflow_steps == 1
    assert config.flow_cg_preconditioner == "fv_multigrid"


def test_structured_pressure_failure_diagnostics_are_json_safe() -> None:
    module = _load_runner_module()
    failure = RuntimeError("pressure failed")
    failure.diagnostics = {
        "preconditioner_requested": "fv_multigrid",
        "exact_relative_residual": np.float64(1.25e-5),
    }

    assert module._exception_diagnostics(failure) == {
        "preconditioner_requested": "fv_multigrid",
        "exact_relative_residual": 1.25e-5,
    }


def _snapshot() -> dict[str, np.ndarray]:
    shape = (1, 2, 3)
    return {
        "velocity": np.zeros(shape + (3,), dtype=np.float32),
        "pressure": np.zeros(shape, dtype=np.float32),
        "obstacle": np.zeros(shape, dtype=np.int32),
        "velocity_dirichlet_boundary_active": np.zeros(shape, dtype=np.int32),
        "velocity_dirichlet_boundary_projection_weight": np.zeros(
            shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_enforcement_weight": np.zeros(
            shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_hard_fixed_component_mask": np.zeros(
            shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_owned_row": np.zeros(shape, dtype=np.int32),
        "velocity_dirichlet_boundary_marker_region_id": np.full(
            shape, -1, dtype=np.int32
        ),
        "flow_solution_stage": np.asarray("post_solid_kinematic_projection"),
        "boundary_topology_stage": np.asarray(
            "post_solid_kinematic_projection"
        ),
        "flow_boundary_state_synchronized": np.asarray(True),
        "structure_geometry_stage": np.asarray(
            "post_solid_kinematic_projection"
        ),
        "cell_center_y_m": np.asarray([0.005, 0.015], dtype=np.float32),
        "cell_center_z_m": np.asarray([0.02, 0.05, 0.08], dtype=np.float32),
        "solid_position_m": np.asarray(
            [[0.001, 0.002, 0.047], [0.001, 0.009, 0.050]], dtype=np.float32
        ),
        "solid_velocity_mps": np.asarray(
            [[0.0, 0.1, -2.0], [0.0, 0.2, 3.0]], dtype=np.float32
        ),
        "solid_rest_position_m": np.asarray(
            [[0.001, 0.001, 0.048], [0.001, 0.009, 0.050]], dtype=np.float32
        ),
        "solid_fixed_mask": np.asarray([True, False]),
        "solid_tip_mask": np.asarray([False, True]),
        "marker_position_m": np.asarray(
            [[0.001, 0.003, 0.046]], dtype=np.float32
        ),
        "marker_velocity_mps": np.asarray(
            [[0.0, 0.4, -5.0]], dtype=np.float32
        ),
        "marker_normal": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        "marker_area_m2": np.asarray([2.5e-6], dtype=np.float32),
        "marker_region_id": np.asarray([1], dtype=np.int32),
    }


class _ArrayField:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def to_numpy(self) -> np.ndarray:
        return self._value.copy()


def test_core_step_snapshot_exports_active_solid_and_marker_geometry() -> None:
    solid = SimpleNamespace(
        particle_count=2,
        v=_ArrayField(np.ones((3, 3), dtype=np.float32)),
    )
    markers = SimpleNamespace(
        marker_count=1,
        x_gamma_m=_ArrayField(np.ones((4, 3), dtype=np.float32)),
        v_gamma_mps=_ArrayField(np.full((4, 3), 2.0, dtype=np.float32)),
        n_gamma=_ArrayField(np.full((4, 3), 3.0, dtype=np.float32)),
        A_gamma_m2=_ArrayField(np.asarray([4.0, 0.0, 0.0, 0.0])),
        region_id=_ArrayField(np.asarray([5, 0, 0, 0], dtype=np.int32)),
    )
    positions = np.arange(6, dtype=np.float32).reshape(2, 3)
    rest = positions - 0.25
    flow_stage_snapshot = {
        "pressure": np.zeros((1, 1, 1), dtype=np.float32),
        "obstacle": np.zeros((1, 1, 1), dtype=np.int32),
        "flow_solution_stage": np.asarray(
            "post_solid_kinematic_projection"
        ),
        "boundary_topology_stage": np.asarray(
            "post_solid_kinematic_projection"
        ),
        "flow_boundary_state_synchronized": np.asarray(True),
        "velocity_dirichlet_boundary_active": np.ones(
            (1, 1, 1), dtype=np.int32
        ),
    }

    snapshot = solid_runner._step_observer_snapshot(
        flow_stage_snapshot,
        solid,
        markers,
        solid_positions_m=positions,
        solid_rest_positions_m=rest,
        fixed_mask=np.asarray([True, False]),
        tip_mask=np.asarray([False, True]),
    )

    assert snapshot["solid_position_m"] == pytest.approx(positions)
    assert snapshot["solid_velocity_mps"].shape == (2, 3)
    assert snapshot["marker_position_m"].shape == (1, 3)
    assert snapshot["marker_area_m2"] == pytest.approx([4.0])
    assert snapshot["marker_region_id"].tolist() == [5]
    assert snapshot["flow_solution_stage"].item() == (
        "post_solid_kinematic_projection"
    )
    assert snapshot["boundary_topology_stage"].item() == (
        "post_solid_kinematic_projection"
    )
    assert bool(snapshot["flow_boundary_state_synchronized"].item())
    assert snapshot["structure_geometry_stage"].item() == (
        "post_solid_kinematic_projection"
    )
    assert snapshot["obstacle"].item() == 0
    assert snapshot["velocity_dirichlet_boundary_active"].item() == 1


def test_synchronized_flow_snapshot_tags_stage_without_mutating_source() -> None:
    source = {
        "pressure": np.zeros((1, 1, 1), dtype=np.float32),
        "obstacle": np.ones((1, 1, 1), dtype=np.int32),
    }

    tagged = solid_runner._synchronized_flow_boundary_snapshot(
        source,
        stage="fixed_solid_preflow_terminal_projection",
    )

    assert set(source) == {"pressure", "obstacle"}
    assert tagged["flow_solution_stage"].item() == (
        "fixed_solid_preflow_terminal_projection"
    )
    assert tagged["boundary_topology_stage"].item() == (
        "fixed_solid_preflow_terminal_projection"
    )
    assert bool(tagged["flow_boundary_state_synchronized"].item())


def test_step_observer_persists_true_deformation_and_complete_history(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    observer = runner._make_step_observer(
        output_dir=tmp_path,
        span_reduction="mean",
        streamwise_velocity_sign=-1.0,
        reverse_streamwise_axis=True,
        streamwise_length_m=0.1,
    )
    history = {
        "max_displacement_m": 1.0e-3,
        "tip_mean_displacement_m": [0.0, 2.0e-4, -1.0e-4],
        "all_diagnostics": {"traction_n": [1.0, 2.0, 3.0]},
    }

    observer(1, 5.0e-4, history, _snapshot())

    frame_path = tmp_path / "step_fields" / "step_0001.npz"
    history_path = tmp_path / "step_history" / "step_0001.json"
    with np.load(frame_path, allow_pickle=False) as frame:
        required = {
            "solid_x_m",
            "solid_y_m",
            "solid_rest_x_m",
            "solid_rest_y_m",
            "solid_vx_mps",
            "solid_vy_mps",
            "solid_position_m",
            "solid_velocity_mps",
            "solid_rest_position_m",
            "solid_fixed_mask",
            "solid_tip_mask",
            "marker_x_m",
            "marker_y_m",
            "marker_position_m",
            "marker_velocity_mps",
            "marker_normal",
            "marker_area_m2",
            "marker_region_id",
            "velocity_dirichlet_boundary_active",
            "velocity_dirichlet_boundary_projection_weight",
            "velocity_dirichlet_boundary_enforcement_weight",
            "velocity_dirichlet_boundary_hard_fixed_component_mask",
            "velocity_dirichlet_boundary_owned_row",
            "velocity_dirichlet_boundary_marker_region_id",
            "flow_solution_stage",
            "boundary_topology_stage",
            "flow_boundary_state_synchronized",
            "structure_geometry_stage",
        }
        assert required <= set(frame.files)
        assert frame["solid_x_m"] == pytest.approx([0.053, 0.050])
        assert frame["solid_y_m"] == pytest.approx([0.002, 0.009])
        assert frame["solid_vx_mps"] == pytest.approx([2.0, -3.0])
        assert frame["marker_x_m"] == pytest.approx([0.054])

    saved_history = json.loads(history_path.read_text(encoding="utf-8"))
    assert saved_history == {
        "history": history,
        "step_index": 1,
        "time_s": 5.0e-4,
    }
    assert not list(tmp_path.rglob("*.tmp"))


def test_prepare_output_dir_rejects_nonempty_directory(tmp_path: Path) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "stale.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-empty"):
        runner._prepare_output_dir(output_dir)


def test_step_artifact_gate_requires_exact_contiguous_readable_pairs(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    observer = runner._make_step_observer(
        output_dir=tmp_path,
        span_reduction="mean",
        streamwise_velocity_sign=-1.0,
        reverse_streamwise_axis=True,
        streamwise_length_m=0.1,
    )
    observer(1, 5.0e-4, {"max_displacement_m": 0.0}, _snapshot())
    observer(2, 1.0e-3, {"max_displacement_m": 1.0e-4}, _snapshot())

    report = runner._validate_step_artifacts(tmp_path, expected_steps=2)
    assert report["status"] == "passed"
    assert report["frame_count"] == 2
    assert report["history_count"] == 2

    (tmp_path / "step_history" / "step_0002.json").unlink()
    with pytest.raises(RuntimeError, match="step artifact sequence mismatch"):
        runner._validate_step_artifacts(tmp_path, expected_steps=2)


def test_step_artifact_gate_rejects_unreadable_npz(tmp_path: Path) -> None:
    runner = _load_runner_module()
    fields = tmp_path / "step_fields"
    histories = tmp_path / "step_history"
    fields.mkdir()
    histories.mkdir()
    (fields / "step_0001.npz").write_bytes(b"not-an-npz")
    (histories / "step_0001.json").write_text(
        json.dumps({"step_index": 1, "time_s": 5.0e-4, "history": {}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unreadable step frame"):
        runner._validate_step_artifacts(tmp_path, expected_steps=1)
