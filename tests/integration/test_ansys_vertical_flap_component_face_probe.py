from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_preflow_snapshot_component_face_probe.py"
)


class _Field:
    def __init__(self, values: Any) -> None:
        self._values = values

    def __getitem__(self, index: Any) -> Any:
        return self._values[index]

    def to_numpy(self) -> Any:
        return self._values


class _Boundary:
    face = (1, 0, 5)
    axis = 0
    pair_index = (*face, axis)
    first_row = (0, 0, 5)
    second_row = (1, 0, 5)
    first_key = 5
    second_key = 53

    def __init__(self) -> None:
        self.grid_nodes = (4, 4, 12)
        self.report_velocity_dirichlet_component_face_target_conflict_count = _Field(
            {None: 1}
        )
        self.report_velocity_dirichlet_component_face_actual_sample_evaluation_count = (
            _Field({None: 9})
        )
        self.report_velocity_dirichlet_component_face_missing_actual_sample_count = (
            _Field({None: 3})
        )
        self.velocity_dirichlet_component_face_segment_projection_only_seam = _Field(
            {self.pair_index: 20}
        )
        self.velocity_dirichlet_component_face_segment_pair_first_author_linear_key = (
            _Field({self.pair_index: self.first_key})
        )
        self.velocity_dirichlet_component_face_segment_pair_second_author_linear_key = (
            _Field({self.pair_index: self.second_key})
        )
        self.velocity_dirichlet_component_face_segment_pair_first_author_kind = _Field(
            {self.pair_index: 0}
        )
        self.velocity_dirichlet_component_face_segment_pair_second_author_kind = _Field(
            {self.pair_index: 0}
        )
        self.velocity_dirichlet_component_face_segment_pair_admission_valid = _Field(
            {self.pair_index: 1}
        )
        self.velocity_dirichlet_component_face_segment_pair_full_valid = _Field(
            {self.pair_index: 1}
        )
        self.velocity_dirichlet_component_face_segment_pair_boundary_point_m = _Field(
            {self.pair_index: (1.5, 0.5, 5.5)}
        )
        self.velocity_dirichlet_component_face_segment_pair_normal = _Field(
            {self.pair_index: (0.0, 0.0, -1.0)}
        )
        self.velocity_dirichlet_component_face_segment_pair_nominal_probe_m = _Field(
            {self.pair_index: (1.5, 0.5, 3.5)}
        )
        self.velocity_dirichlet_component_face_segment_pair_boundary_target_mps = (
            _Field({self.pair_index: 0.25})
        )
        self.velocity_dirichlet_component_face_segment_pair_endpoint_clamped = _Field(
            {self.pair_index: 1}
        )
        self.velocity_dirichlet_component_face_segment_pair_clamp_support_ratio = (
            _Field({self.pair_index: 1.0})
        )
        self.velocity_dirichlet_component_face_segment_pair_geometry_tolerance = (
            _Field({self.pair_index: 0.01})
        )
        self.velocity_dirichlet_component_face_actual_sample_valid = _Field(
            {self.first_row: 1, self.second_row: 0}
        )
        self.velocity_dirichlet_component_face_actual_sample_point_m = _Field(
            {
                self.first_row: (1.5, 0.5, 3.5),
                self.second_row: (1.5, 0.5, 3.0),
            }
        )
        self.velocity_dirichlet_component_face_actual_sample_velocity_mps = _Field(
            {
                self.first_row: (1.0, 2.0, 3.0),
                self.second_row: (4.0, 5.0, 6.0),
            }
        )
        self.pressure_neumann_normal_field = _Field(
            {
                self.first_row: (0.0, 0.0, -1.0),
                self.second_row: (0.0, 0.0, -1.0),
            }
        )

    def _canonical_velocity_dirichlet_first_target_conflict_diagnostic(
        self,
    ) -> dict[str, Any]:
        return {
            "component_face": self.face,
            "component_axis": self.axis,
            "conflict_source": "segment_reconstruction_invalid",
            "conflict_path_code": 2,
            "claim_count": 2,
            "author_linear_keys": (self.first_key, self.second_key),
            "authors": (
                {"source_row": self.first_row, "nearest_marker": 64},
                {"source_row": self.second_row, "nearest_marker": 64},
            ),
        }

    def assemble_velocity_dirichlet_component_face_ledger(
        self, **_kwargs: Any
    ) -> None:
        self._validate_canonical_velocity_dirichlet_target_conflict_precommit()

    def _validate_canonical_velocity_dirichlet_target_conflict_precommit(
        self,
    ) -> None:
        raise RuntimeError("original target conflict")


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_component_face_probe",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_fields() -> dict[str, Any]:
    nodes = (4, 4, 12)
    obstacle = {
        (i, j, k): 0
        for i in range(nodes[0])
        for j in range(nodes[1])
        for k in range(nodes[2])
    }
    velocity = {index: (1.0, 2.0, 3.0) for index in obstacle}
    search = SimpleNamespace(
        node_boundary_point_m=_Field(
            {
                _Boundary.first_row: (1.5, 0.5, 5.5),
                _Boundary.second_row: (1.5, 0.5, 5.5),
            }
        ),
        node_interior_fluid_point_m=_Field(
            {
                _Boundary.first_row: (1.5, 0.5, 3.5),
                _Boundary.second_row: (1.5, 0.5, 3.5),
            }
        ),
    )
    return {
        "obstacle_field": _Field(obstacle),
        "velocity_field": _Field(velocity),
        "search": search,
        "cell_face_x_m": _Field([0.0, 1.0, 2.0, 3.0, 4.0]),
        "cell_face_y_m": _Field([0.0, 1.0, 2.0, 3.0, 4.0]),
        "cell_face_z_m": _Field([float(value) for value in range(13)]),
        "cell_center_x_m": _Field([0.5, 1.5, 2.5, 3.5]),
        "cell_center_y_m": _Field([0.5, 1.5, 2.5, 3.5]),
        "cell_center_z_m": _Field([value + 0.5 for value in range(12)]),
        "grid_nodes": nodes,
    }


def test_probe_writes_conflict_state_before_original_validator_and_restores_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    original_assemble = _Boundary.assemble_velocity_dirichlet_component_face_ledger
    original_validator = (
        _Boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
    )
    output_dir = tmp_path / "new-diagnostic-output"
    fields = _runtime_fields()

    def replay(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["output_dir"] == output_dir
        output_dir.mkdir(parents=True, exist_ok=False)
        _Boundary().assemble_velocity_dirichlet_component_face_ledger(**fields)
        raise AssertionError("unreachable")

    monkeypatch.setattr(module, "_load_boundary_type", lambda: _Boundary)
    monkeypatch.setattr(module, "run_diagnostic_replay", replay)

    with pytest.raises(RuntimeError, match="original target conflict"):
        module.run_component_face_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=output_dir,
            allowed_source_diffs=("simulation_core/coupling/hibm_mpm/core.py",),
        )

    assert _Boundary.assemble_velocity_dirichlet_component_face_ledger is original_assemble
    assert (
        _Boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        is original_validator
    )
    payload = json.loads(
        (output_dir / "component_face_probe.json").read_text(encoding="utf-8")
    )
    assert payload["component_face_probe"] is True
    assert payload["evidence_class"] == "diagnostic_only"
    assert payload["formal_validation_eligible"] is False
    assert payload["parity_claimed"] is False
    assert payload["fluent_parity_claimed"] is False
    assert payload["fresh_preflow"] is False
    assert payload["first_conflict"]["component_face"] == [1, 0, 5]
    assert payload["first_conflict"]["conflict_path_code"] == 2
    assert payload["global_counters"] == {
        "scope": "global_not_face_local",
        "actual_sample_evaluation_count": 9,
        "missing_actual_sample_count": 3,
    }
    pair = payload["pair_reconstruction_state"]
    assert pair["mode"] == 20
    assert pair["author_linear_keys"] == [5, 53]
    assert pair["author_kinds"] == [0, 0]
    assert pair["admission_valid"] is True
    assert pair["full_valid"] is True
    assert pair["boundary_point_m"] == [1.5, 0.5, 5.5]
    assert pair["normal"] == [0.0, 0.0, -1.0]
    assert pair["nominal_probe_m"] == [1.5, 0.5, 3.5]
    assert pair["endpoint_clamped"] is True
    assert pair["clamp_support_ratio"] == 1.0
    assert pair["geometry_tolerance"] == 0.01

    authors = payload["authors"]
    assert [row["author_linear_key"] for row in authors] == [5, 53]
    assert authors[0]["raw_node_interior_fluid_point_m"] == [1.5, 0.5, 3.5]
    assert authors[0]["node_boundary_normal"] == [0.0, 0.0, -1.0]
    assert authors[0]["nominal_sample"]["valid"] is True
    assert authors[0]["nominal_sample"]["velocity_mps"] == [1.0, 2.0, 3.0]
    assert authors[0]["actual_sample"]["valid"] is True
    assert authors[1]["actual_sample"]["valid"] is False
    assert authors[0]["live_storage_velocity_finite"] is True
    assert authors[0]["live_storage_velocity_mps"] == [1.0, 2.0, 3.0]

    stencil = payload["runtime_obstacle_stencil"]
    assert stencil["target"] == [1, 0, 5]
    assert stencil["offsets"] == {"i": [-1, 1], "j": [-1, 1], "k": [-8, 8]}
    assert len(stencil["cells"]) == 72
    assert stencil["axis_coordinates"]["y"][0] == {
        "index": 0,
        "lower_face_m": 0.0,
        "center_m": 0.5,
        "upper_face_m": 1.0,
    }
    candidates = payload["canonical_walk_candidates"]
    assert len(candidates) == 5
    assert candidates[0]["point_m"] == [1.5, 0.5, 3.5]
    assert candidates[0]["component_weights"] == [1.0, 0.5, 1.0]
    assert candidates[0]["minimum_component_weight"] == 0.5
    assert candidates[0]["velocity_mps"] == [1.0, 2.0, 3.0]
    assert candidates[0]["first_accepted"] is True
    assert candidates[0]["would_execute_before_core_short_circuit"] is True
    assert candidates[1]["first_accepted"] is False
    assert candidates[1]["would_execute_before_core_short_circuit"] is False


def test_probe_restores_both_class_methods_when_replay_fails_before_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    original_assemble = _Boundary.assemble_velocity_dirichlet_component_face_ledger
    original_validator = (
        _Boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
    )
    monkeypatch.setattr(module, "_load_boundary_type", lambda: _Boundary)

    def fail_before_assembly(**_kwargs: Any) -> dict[str, Any]:
        raise ValueError("preflight failed")

    monkeypatch.setattr(module, "run_diagnostic_replay", fail_before_assembly)

    with pytest.raises(ValueError, match="preflight failed"):
        module.run_component_face_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=tmp_path / "diagnostic-output",
        )

    assert _Boundary.assemble_velocity_dirichlet_component_face_ledger is original_assemble
    assert (
        _Boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        is original_validator
    )


def test_probe_write_failure_preserves_original_conflict_and_restores_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    original_assemble = _Boundary.assemble_velocity_dirichlet_component_face_ledger
    original_validator = (
        _Boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
    )
    output_dir = tmp_path / "new-diagnostic-output"
    fields = _runtime_fields()

    def replay(**_kwargs: Any) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=False)
        _Boundary().assemble_velocity_dirichlet_component_face_ledger(**fields)
        raise AssertionError("unreachable")

    def fail_probe_write(_path: Path, _payload: dict[str, Any]) -> None:
        raise OSError("probe disk unavailable")

    monkeypatch.setattr(module, "_load_boundary_type", lambda: _Boundary)
    monkeypatch.setattr(module, "run_diagnostic_replay", replay)
    monkeypatch.setattr(module, "write_json_exclusive", fail_probe_write)

    with pytest.raises(RuntimeError, match="original target conflict") as exc_info:
        module.run_component_face_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=output_dir,
        )

    assert _Boundary.assemble_velocity_dirichlet_component_face_ledger is original_assemble
    assert (
        _Boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        is original_validator
    )
    assert any(
        "component-face probe capture failed: OSError: probe disk unavailable" in note
        for note in getattr(exc_info.value, "__notes__", ())
    )
    assert not (output_dir / "component_face_probe.json").exists()


def test_probe_cli_uses_the_existing_replay_argument_contract() -> None:
    module = _load_script()
    args = module._build_parser().parse_args(
        [
            "--snapshot",
            "snapshot",
            "--config-json",
            "config.json",
            "--source-manifest-json",
            "manifest.json",
            "--output-dir",
            "new-output",
            "--allow-source-diff",
            "simulation_core/coupling/hibm_mpm/core.py",
        ]
    )

    assert args.snapshot == Path("snapshot")
    assert args.config_json == Path("config.json")
    assert args.source_manifest_json == Path("manifest.json")
    assert args.output_dir == Path("new-output")
    assert args.allow_source_diff == ["simulation_core/coupling/hibm_mpm/core.py"]


def test_probe_cli_reports_failed_diagnostic_without_claiming_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    monkeypatch.setattr(
        module,
        "run_component_face_probe",
        lambda **_kwargs: {"status": "failed"},
    )

    exit_code = module.main(
        [
            "--snapshot",
            "snapshot",
            "--config-json",
            "config.json",
            "--source-manifest-json",
            "manifest.json",
            "--output-dir",
            "new-output",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "status 'failed'" in output
    assert "no target conflict interrupted the step" not in output
