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
        "flow_solution_stage": np.asarray("pre_solid_projection"),
        "boundary_topology_stage": np.asarray("pre_solid_projection"),
        "flow_boundary_state_synchronized": np.asarray(True),
        "structure_geometry_stage": np.asarray("post_solid_observer"),
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


class _CountingField(_ArrayField):
    def __init__(self, value: np.ndarray) -> None:
        super().__init__(value)
        self.to_numpy_calls = 0

    def to_numpy(self) -> np.ndarray:
        self.to_numpy_calls += 1
        return super().to_numpy()


def _flow_snapshot_fluid() -> tuple[SimpleNamespace, dict[str, _CountingField]]:
    grid_shape = (1, 2, 3)
    field_values = {
        "fsi_pressure": np.arange(6, dtype=np.float32).reshape(grid_shape),
        "pressure": np.zeros(grid_shape, dtype=np.float32),
        "velocity": np.arange(18, dtype=np.float32).reshape(grid_shape + (3,)),
        "obstacle": np.zeros(grid_shape, dtype=np.int32),
        "hibm_base_obstacle": np.zeros(grid_shape, dtype=np.int32),
        "hibm_dynamic_solid_volume_obstacle": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "hibm_dynamic_solid_volume_external_carve": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_active": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_projection_weight": np.zeros(
            grid_shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_enforcement_weight": np.zeros(
            grid_shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_hard_fixed_component_mask": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_external_exact_component_mask": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_owned_row": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_marker_region_id": np.full(
            grid_shape, -1, dtype=np.int32
        ),
        "cell_face_x_m": np.asarray([0.0, 0.01], dtype=np.float32),
        "cell_face_y_m": np.asarray([0.0, 0.01, 0.02], dtype=np.float32),
        "cell_face_z_m": np.asarray([0.0, 0.01, 0.02, 0.03], dtype=np.float32),
        "cell_center_x_m": np.asarray([0.005], dtype=np.float32),
        "cell_center_y_m": np.asarray([0.005, 0.015], dtype=np.float32),
        "cell_center_z_m": np.asarray([0.005, 0.015, 0.025], dtype=np.float32),
        "cell_width_x_m": np.asarray([0.01], dtype=np.float32),
        "cell_width_y_m": np.asarray([0.01, 0.01], dtype=np.float32),
        "cell_width_z_m": np.asarray([0.01, 0.01, 0.01], dtype=np.float32),
    }
    fields = {
        name: _CountingField(value) for name, value in field_values.items()
    }
    return SimpleNamespace(**fields, sampling_obstacle=None), fields


@pytest.mark.parametrize(
    ("step_count", "has_step_observer", "export_final", "expected"),
    (
        (0, False, False, False),
        (0, True, False, False),
        (0, False, True, False),
        (1, False, False, False),
        (1, True, False, True),
        (1, False, True, True),
        (1, True, True, True),
    ),
)
def test_flow_geometry_cache_is_only_built_for_fsi_snapshot_exports(
    step_count: int,
    has_step_observer: bool,
    export_final: bool,
    expected: bool,
) -> None:
    assert (
        solid_runner._flow_geometry_snapshot_cache_required(
            step_count=step_count,
            has_step_observer=has_step_observer,
            export_final_flow_snapshot=export_final,
        )
        is expected
    )


def test_flow_snapshots_reuse_read_only_immutable_geometry_without_redownload() -> None:
    fluid, fields = _flow_snapshot_fluid()
    immutable_geometry = solid_runner._immutable_flow_geometry_snapshot(
        fluid,
        include_full_geometry=True,
    )

    parity_a = solid_runner._flow_parity_snapshot(
        fluid,
        immutable_geometry=immutable_geometry,
    )
    parity_b = solid_runner._flow_parity_snapshot(
        fluid,
        immutable_geometry=immutable_geometry,
    )
    full = solid_runner._flow_field_snapshot(
        fluid,
        immutable_geometry=immutable_geometry,
    )

    for name, value in immutable_geometry.items():
        assert fields[name].to_numpy_calls == 1
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = value.flat[0]
        assert full[name] is value
    for parity in (parity_a, parity_b):
        assert parity["cell_center_y_m"] is immutable_geometry["cell_center_y_m"]
        assert parity["cell_center_z_m"] is immutable_geometry["cell_center_z_m"]
    immutable_names = set(solid_runner._IMMUTABLE_FLOW_GEOMETRY_FIELD_NAMES)
    for name, field in fields.items():
        if name in immutable_names:
            expected_calls = 1
        elif name == "pressure":
            expected_calls = 0
        else:
            expected_calls = 3
        assert field.to_numpy_calls == expected_calls, name

    uncached_fluid, _ = _flow_snapshot_fluid()
    uncached_parity = solid_runner._flow_parity_snapshot(uncached_fluid)
    uncached_full = solid_runner._flow_field_snapshot(uncached_fluid)
    for cached, uncached in ((parity_a, uncached_parity), (full, uncached_full)):
        assert tuple(cached) == tuple(uncached)
        for name in cached:
            assert cached[name].dtype == uncached[name].dtype
            np.testing.assert_array_equal(cached[name], uncached[name])

    partial_fluid, partial_fields = _flow_snapshot_fluid()
    partial_geometry = solid_runner._immutable_flow_geometry_snapshot(
        partial_fluid,
        include_full_geometry=False,
    )
    partial_parity = solid_runner._flow_parity_snapshot(
        partial_fluid,
        immutable_geometry=partial_geometry,
    )
    assert tuple(partial_geometry) == solid_runner._PARITY_FLOW_GEOMETRY_FIELD_NAMES
    for name in solid_runner._IMMUTABLE_FLOW_GEOMETRY_FIELD_NAMES:
        expected_calls = 1 if name in partial_geometry else 0
        assert partial_fields[name].to_numpy_calls == expected_calls
    for name, field in partial_fields.items():
        if name not in immutable_names and name != "pressure":
            assert field.to_numpy_calls == 1, name
    assert partial_fields["pressure"].to_numpy_calls == 0
    assert partial_parity["cell_center_y_m"] is partial_geometry["cell_center_y_m"]
    assert partial_parity["cell_center_z_m"] is partial_geometry["cell_center_z_m"]


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
    assert snapshot["flow_solution_stage"].item() == "pre_solid_projection"
    assert snapshot["boundary_topology_stage"].item() == "pre_solid_projection"
    assert bool(snapshot["flow_boundary_state_synchronized"].item())
    assert snapshot["structure_geometry_stage"].item() == "post_solid_observer"
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
