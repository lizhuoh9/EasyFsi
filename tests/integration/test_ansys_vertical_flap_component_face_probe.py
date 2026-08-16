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


class _ReadRaisesField:
    def __getitem__(self, index: Any) -> Any:
        raise AssertionError(f"guarded field was read at {index!r}")


class _Boundary:
    face = (1, 0, 5)
    axis = 0
    pair_index = (*face, axis)
    first_row = (0, 0, 5)
    second_row = (1, 0, 5)
    first_key = 5
    second_key = 53
    conflict_path_code = 0
    node_count = 4 * 4 * 12

    @classmethod
    def _conflict_linear_key(cls, conflict_path_code: int) -> int:
        component_face_linear_key = (
            ((cls.face[0] * 4 + cls.face[1]) * 12 + cls.face[2]) * 3 + cls.axis
        )
        return component_face_linear_key * 4 + conflict_path_code

    @classmethod
    def _author_witness(cls, author_linear_key: int, conflict_path_code: int) -> int:
        reverse_author_key = cls.node_count - 1 - author_linear_key
        return -2 - (
            (3 - conflict_path_code) * cls.node_count + reverse_author_key
        )

    def __init__(self) -> None:
        self.grid_nodes = (4, 4, 12)
        self.marker_capacity = 128
        self.report_velocity_dirichlet_component_face_target_conflict_count = _Field(
            {None: 1}
        )
        self.report_velocity_dirichlet_component_face_actual_sample_evaluation_count = (
            _Field({None: 9})
        )
        self.report_velocity_dirichlet_component_face_missing_actual_sample_count = (
            _Field({None: 3})
        )
        self.report_velocity_dirichlet_component_face_first_target_conflict_linear_key = (
            _Field({None: self._conflict_linear_key(self.conflict_path_code)})
        )
        self.velocity_dirichlet_component_face_segment_first_author_linear_key = (
            _Field(
                {
                    self.pair_index: self._author_witness(
                        self.first_key, self.conflict_path_code
                    )
                }
            )
        )
        self.velocity_dirichlet_component_face_segment_second_author_linear_key = (
            _Field(
                {
                    self.pair_index: self._author_witness(
                        self.second_key, self.conflict_path_code
                    )
                }
            )
        )
        self.velocity_dirichlet_component_face_claim_count = _Field(
            {self.face: (2, 0, 0)}
        )
        self.velocity_dirichlet_component_face_claim_target_mps = _Field(
            {self.face: (0.25, 0.0, 0.0)}
        )
        self.velocity_dirichlet_component_face_claim_region_id = _Field(
            {self.face: (303, -1, -1)}
        )
        self.velocity_dirichlet_mps_field = _Field(
            {
                self.first_row: (0.25, 0.0, 0.0),
                self.second_row: (0.25, 0.0, 0.0),
            }
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
        self.velocity_dirichlet_component_face_segment_pair_strict_owner_cause = (
            _Field({self.pair_index: 0})
        )
        self.velocity_dirichlet_component_face_segment_pair_derived_terminal_cause = (
            _Field({self.pair_index: 0})
        )
        self.velocity_dirichlet_component_face_adjacent_direct_pair_target_valid = (
            _Field({self.pair_index: 0})
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
            {self.first_row: 1, self.second_row: 1}
        )
        self.velocity_dirichlet_component_face_actual_sample_point_m = _Field(
            {
                self.first_row: (1.5, 0.5, 3.5),
                self.second_row: (1.5, 0.6, 3.0),
            }
        )
        self.velocity_dirichlet_component_face_actual_sample_velocity_mps = _Field(
            {
                self.first_row: (1.0, 2.0, 3.0),
                self.second_row: (4.0, 5.0, 6.0),
            }
        )
        self.velocity_dirichlet_component_face_direct_selected_storage_offset = (
            _Field(
                {
                    self.first_row: (1, -1, -1),
                    self.second_row: (0, -1, -1),
                }
            )
        )
        self.pressure_neumann_normal_field = _Field(
            {
                self.first_row: (9.0, 0.0, -2.0),
                self.second_row: (9.0, 0.0, -2.0),
            }
        )
        self.marker_region_id = _Field({64: 303, 65: 303, 66: 303})
        self.marker_pressure_owner_index = _Field({64: 64, 65: 65, 66: 66})
        self.marker_position_m = _Field(
            {
                64: (1.5, 0.4, 5.5),
                65: (1.5, 0.5, 5.5),
                66: (1.5, 0.6, 5.5),
            }
        )
        self.marker_velocity_mps = _Field(
            {
                64: (0.25, 0.0, 0.0),
                65: (0.25, 0.0, 0.0),
                66: (0.25, 0.0, 0.0),
            }
        )

    def _canonical_velocity_dirichlet_first_target_conflict_diagnostic(
        self,
    ) -> dict[str, Any]:
        from simulation_core.coupling.hibm_mpm.core import (
            HibmMpmIbBoundaryConditions,
        )

        diagnostic = HibmMpmIbBoundaryConditions._canonical_velocity_dirichlet_first_target_conflict_diagnostic(
            self
        )
        assert diagnostic is not None
        return diagnostic

    def assemble_velocity_dirichlet_component_face_ledger(
        self, **kwargs: Any
    ) -> None:
        self.__dict__["_canonical_velocity_dirichlet_precommit_diagnostic_context"] = {
            "search": kwargs["search"],
            "marker_position_m": self.marker_position_m,
            "marker_velocity_mps": self.marker_velocity_mps,
            "marker_region_id": self.marker_region_id,
            "marker_pressure_owner_index": self.marker_pressure_owner_index,
            "marker_pressure_owner_available": True,
            "marker_geometry_available": True,
            "physical_marker_count": 64,
            "projection_vertex_count": 67,
            "inactive_axis": 0,
            "cell_center_x_m": kwargs["cell_center_x_m"],
            "cell_center_y_m": kwargs["cell_center_y_m"],
            "cell_center_z_m": kwargs["cell_center_z_m"],
            "cell_face_x_m": kwargs["cell_face_x_m"],
            "cell_face_y_m": kwargs["cell_face_y_m"],
            "cell_face_z_m": kwargs["cell_face_z_m"],
            "projection_segment_indices": _Field(((64, 65), (65, 66))),
            "projection_segment_count": 2,
            "projection_segment_topology_available": True,
            "source_search_support_available": True,
            "source_search_support_anisotropic": False,
            "source_search_support_radius_xyz_m": (2.0, 2.0, 2.0),
        }
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
        node_projection_marker_indices=_Field(
            {
                _Boundary.first_row: (64, 65),
                _Boundary.second_row: (65, 66),
            }
        ),
        node_projection_marker_weights=_Field(
            {
                _Boundary.first_row: (0.75, 0.25),
                _Boundary.second_row: (0.25, 0.75),
            }
        ),
        nearest_marker=_Field(
            {_Boundary.first_row: 64, _Boundary.second_row: 65}
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
    first_conflict = payload["first_conflict"]
    assert first_conflict["component_face"] == [1, 0, 5]
    assert first_conflict["conflict_source"] == "prepare_pair_arbitration"
    assert first_conflict["conflict_path_code"] == 0
    pair_cache = first_conflict["pair_cache"]
    assert pair_cache["mode"] == 20
    assert pair_cache["author_linear_keys"] == [5, 53]
    assert pair_cache["author_kinds"] == [0, 0]
    assert pair_cache["admission_valid"] is True
    assert pair_cache["full_valid"] is True
    assert pair_cache["strict_owner_cause"] == 0
    assert pair_cache["strict_owner_cause_name"] == "none"
    assert pair_cache["derived_terminal_cause"] == 0
    assert type(pair_cache["adjacent_direct_pair_target_valid"]) is int
    assert pair_cache["adjacent_direct_pair_target_valid"] == 0
    assert type(pair_cache["first_direct_selected_storage_offset"]) is int
    assert pair_cache["first_direct_selected_storage_offset"] == 1
    assert type(pair_cache["second_direct_selected_storage_offset"]) is int
    assert pair_cache["second_direct_selected_storage_offset"] == 0
    assert pair_cache["endpoint_clamped"] is True
    assert pair_cache["boundary_point_m"] == [1.5, 0.5, 5.5]
    assert pair_cache["normal"] == [0.0, 0.0, -1.0]
    assert pair_cache["nominal_probe_m"] == [1.5, 0.5, 3.5]
    assert pair_cache["boundary_target_mps"] == 0.25
    assert pair_cache["clamp_support_ratio"] == 1.0
    assert pair_cache["geometry_tolerance_m"] == 0.01

    diagnostic_authors = first_conflict["authors"]
    assert [author["source_row"] for author in diagnostic_authors] == [
        [0, 0, 5],
        [1, 0, 5],
    ]
    first_author, second_author = diagnostic_authors
    assert first_author["normal"] == [0.0, 0.0, -1.0]
    assert first_author["nominal_interior_point_m"] == [1.5, 0.5, 3.5]
    assert first_author["actual_sample_valid"] is True
    assert first_author["actual_sample_point_m"] == [1.5, 0.5, 3.5]
    assert first_author["actual_sample_velocity_mps"] == [1.0, 2.0, 3.0]
    assert first_author["source_center_m"] == [0.5, 0.5, 5.5]
    assert second_author["source_center_m"] == [1.5, 0.5, 5.5]
    first_post = first_author["post_admission"]
    assert first_post["normal_norm"] == 1.0
    assert first_post["nominal_progress_m"] == 2.0
    assert first_post["actual_progress_m"] == 2.0
    assert first_post["nominal_lateral_squared_m2"] == 0.0
    assert first_post["nominal_lateral_limit_squared_m2"] == pytest.approx(0.000108)
    assert first_post["nominal_lateral_valid"] is True
    assert first_post["actual_lateral_squared_m2"] == 0.0
    assert first_post["actual_lateral_limit_squared_m2"] == pytest.approx(0.000108)
    assert first_post["actual_lateral_valid"] is True
    assert first_post["probe_margin_m"] == 2.0
    second_post = second_author["post_admission"]
    assert second_post["normal_norm"] == 1.0
    assert second_post["nominal_progress_m"] == 2.0
    assert second_post["actual_progress_m"] == 2.5
    assert second_post["nominal_lateral_squared_m2"] == 0.0
    assert second_post["nominal_lateral_limit_squared_m2"] == pytest.approx(0.000108)
    assert second_post["nominal_lateral_valid"] is True
    assert second_post["actual_lateral_squared_m2"] == pytest.approx(0.01)
    assert second_post["actual_lateral_limit_squared_m2"] == pytest.approx(0.00011252)
    assert second_post["actual_lateral_valid"] is False
    assert second_post["probe_margin_m"] == 2.0
    post_admission_pair = first_conflict["post_admission_pair"]
    assert post_admission_pair["available"] is True
    assert post_admission_pair["alignment_residual_ratio_squared"] == 2.0e-6
    assert post_admission_pair["geometry_tolerance_m"] == 0.01
    assert post_admission_pair["probe_margin_min_m"] == 2.0
    assert post_admission_pair["probe_margin_delta_m"] == 0.0
    assert post_admission_pair["probe_margin_positive_vs_tolerance"] is True
    assert post_admission_pair["probe_margin_pair_match"] is True
    assert post_admission_pair["all_predicates_valid"] is False
    assert post_admission_pair["failed_predicates"] == [
        "second_actual_lateral_valid"
    ]
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
    assert authors[0]["node_boundary_normal"] == [9.0, 0.0, -2.0]
    assert authors[0]["nominal_sample"]["valid"] is True
    assert authors[0]["nominal_sample"]["velocity_mps"] == [1.0, 2.0, 3.0]
    assert authors[0]["actual_sample"]["valid"] is True
    assert authors[1]["actual_sample"]["valid"] is True
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


@pytest.mark.parametrize(
    ("conflict_path_code", "cached_first_key"),
    ((2, _Boundary.first_key), (0, _Boundary.first_key + 1)),
    ids=("segment-reconstruction-path", "pair-cache-key-mismatch"),
)
def test_production_conflict_diagnostic_guards_post_admission_actual_fields(
    conflict_path_code: int,
    cached_first_key: int,
) -> None:
    boundary = _Boundary()
    with pytest.raises(RuntimeError, match="original target conflict"):
        boundary.assemble_velocity_dirichlet_component_face_ledger(
            **_runtime_fields()
        )

    boundary.report_velocity_dirichlet_component_face_first_target_conflict_linear_key = (
        _Field({None: boundary._conflict_linear_key(conflict_path_code)})
    )
    boundary.velocity_dirichlet_component_face_segment_first_author_linear_key = _Field(
        {
            boundary.pair_index: boundary._author_witness(
                boundary.first_key, conflict_path_code
            )
        }
    )
    boundary.velocity_dirichlet_component_face_segment_second_author_linear_key = _Field(
        {
            boundary.pair_index: boundary._author_witness(
                boundary.second_key, conflict_path_code
            )
        }
    )
    boundary.velocity_dirichlet_component_face_segment_pair_first_author_linear_key = (
        _Field({boundary.pair_index: cached_first_key})
    )
    guarded_actual_field = _ReadRaisesField()
    boundary.velocity_dirichlet_component_face_actual_sample_valid = (
        guarded_actual_field
    )
    boundary.velocity_dirichlet_component_face_actual_sample_point_m = (
        guarded_actual_field
    )
    boundary.velocity_dirichlet_component_face_actual_sample_velocity_mps = (
        guarded_actual_field
    )

    diagnostic = (
        boundary._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
    )

    assert diagnostic["post_admission_pair"]["available"] is False


def test_production_conflict_diagnostic_uses_raw_rows_when_pair_cache_is_invalid(
) -> None:
    boundary = _Boundary()
    with pytest.raises(RuntimeError, match="original target conflict"):
        boundary.assemble_velocity_dirichlet_component_face_ledger(
            **_runtime_fields()
        )
    boundary.velocity_dirichlet_component_face_segment_pair_full_valid = _Field(
        {boundary.pair_index: 0}
    )
    boundary.velocity_dirichlet_component_face_segment_pair_boundary_point_m = _Field(
        {boundary.pair_index: (0.0, 0.0, 0.0)}
    )
    boundary.velocity_dirichlet_component_face_segment_pair_normal = _Field(
        {boundary.pair_index: (0.0, 0.0, 0.0)}
    )
    boundary.velocity_dirichlet_component_face_segment_pair_nominal_probe_m = _Field(
        {boundary.pair_index: (0.0, 0.0, 0.0)}
    )

    diagnostic = (
        boundary._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
    )

    assert diagnostic["pair_cache"]["full_valid"] is False
    assert diagnostic["pair_cache"]["boundary_point_m"] == (0.0, 0.0, 0.0)
    assert diagnostic["pair_cache"]["normal"] == (0.0, 0.0, 0.0)
    assert diagnostic["pair_cache"]["nominal_probe_m"] == (0.0, 0.0, 0.0)
    first_author, second_author = diagnostic["authors"]
    assert first_author["normal"] == (0.0, 0.0, -1.0)
    assert first_author["post_admission"]["normal_norm"] == 1.0
    assert first_author["post_admission"]["nominal_progress_m"] == 2.0
    assert first_author["post_admission"]["nominal_lateral_squared_m2"] == 0.0
    assert first_author["post_admission"]["probe_margin_m"] == 2.0
    assert second_author["normal"] == (0.0, 0.0, -1.0)
    assert second_author["post_admission"]["actual_progress_m"] == 2.5
    assert second_author["post_admission"]["actual_lateral_squared_m2"] == pytest.approx(
        0.01
    )
    assert diagnostic["post_admission_pair"]["available"] is True
    assert diagnostic["post_admission_pair"]["failed_predicates"] == (
        "second_actual_lateral_valid",
    )
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize(
    "pair_cache_field_name",
    (
        "velocity_dirichlet_component_face_segment_projection_only_seam",
        "velocity_dirichlet_component_face_segment_pair_first_author_linear_key",
        "velocity_dirichlet_component_face_segment_pair_second_author_linear_key",
        "velocity_dirichlet_component_face_segment_pair_first_author_kind",
        "velocity_dirichlet_component_face_segment_pair_second_author_kind",
        "velocity_dirichlet_component_face_segment_pair_admission_valid",
        "velocity_dirichlet_component_face_segment_pair_full_valid",
        "velocity_dirichlet_component_face_segment_pair_strict_owner_cause",
        "velocity_dirichlet_component_face_segment_pair_derived_terminal_cause",
        "velocity_dirichlet_component_face_adjacent_direct_pair_target_valid",
        "velocity_dirichlet_component_face_direct_selected_storage_offset",
        "velocity_dirichlet_component_face_segment_pair_endpoint_clamped",
        "velocity_dirichlet_component_face_segment_pair_boundary_point_m",
        "velocity_dirichlet_component_face_segment_pair_normal",
        "velocity_dirichlet_component_face_segment_pair_nominal_probe_m",
        "velocity_dirichlet_component_face_segment_pair_boundary_target_mps",
        "velocity_dirichlet_component_face_segment_pair_clamp_support_ratio",
        "velocity_dirichlet_component_face_segment_pair_geometry_tolerance",
    ),
)
def test_production_conflict_diagnostic_preserves_base_when_pair_cache_read_fails(
    pair_cache_field_name: str,
) -> None:
    boundary = _Boundary()
    with pytest.raises(RuntimeError, match="original target conflict"):
        boundary.assemble_velocity_dirichlet_component_face_ledger(
            **_runtime_fields()
        )
    setattr(boundary, pair_cache_field_name, _ReadRaisesField())

    diagnostic = (
        boundary._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
    )

    assert diagnostic["component_face"] == boundary.face
    assert diagnostic["author_linear_keys"] == (
        boundary.first_key,
        boundary.second_key,
    )
    assert [author["source_row"] for author in diagnostic["authors"]] == [
        boundary.first_row,
        boundary.second_row,
    ]
    assert [marker["marker_index"] for marker in diagnostic["markers"]] == [
        64,
        65,
        66,
    ]
    assert diagnostic["pair_cache_capture_error"]["type"] == "AssertionError"
    assert diagnostic["pair_cache_capture_error"]["message"]
    assert diagnostic["post_admission_pair"]["available"] is False
    assert (
        diagnostic["post_admission_pair"]["reason"]
        == "pair_cache_capture_error"
    )
    json.dumps(diagnostic, allow_nan=False)


def test_production_conflict_validator_preserves_original_error_when_cache_read_fails(
) -> None:
    boundary = _Boundary()
    with pytest.raises(RuntimeError, match="original target conflict"):
        boundary.assemble_velocity_dirichlet_component_face_ledger(
            **_runtime_fields()
        )
    boundary.velocity_dirichlet_component_face_adjacent_direct_pair_target_valid = (
        _ReadRaisesField()
    )

    from simulation_core.coupling.hibm_mpm.core import (
        HibmMpmIbBoundaryConditions,
    )
    production_validator = getattr(
        HibmMpmIbBoundaryConditions,
        "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
    )

    with pytest.raises(
        RuntimeError,
        match="conflicting canonical component-face claims",
    ) as exc_info:
        production_validator(boundary)

    assert getattr(exc_info.value, "reason_code") == "target_conflict"
    diagnostics = getattr(exc_info.value, "diagnostics")
    diagnostic = diagnostics["context"]["first_conflict"]
    assert [author["source_row"] for author in diagnostic["authors"]] == [
        list(boundary.first_row),
        list(boundary.second_row),
    ]
    assert [marker["marker_index"] for marker in diagnostic["markers"]] == [
        64,
        65,
        66,
    ]
    assert diagnostic["pair_cache_capture_error"]["type"] == "AssertionError"
    assert diagnostic["post_admission_pair"]["reason"] == "pair_cache_capture_error"
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize(
    "first_raw_normal",
    (
        (0.0, 0.0, 0.0),
        (9.0, float("nan"), -2.0),
        (9.0, 0.0, float("inf")),
    ),
    ids=("zero", "active-y-nan", "active-z-infinite"),
)
def test_production_conflict_diagnostic_sanitizes_invalid_author_normal(
    first_raw_normal: tuple[float, float, float],
) -> None:
    boundary = _Boundary()
    with pytest.raises(RuntimeError, match="original target conflict"):
        boundary.assemble_velocity_dirichlet_component_face_ledger(
            **_runtime_fields()
        )
    boundary.pressure_neumann_normal_field = _Field(
        {
            boundary.first_row: first_raw_normal,
            boundary.second_row: (9.0, 0.0, -2.0),
        }
    )

    diagnostic = (
        boundary._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
    )

    first_author, second_author = diagnostic["authors"]
    assert diagnostic["markers"] == (
        {
            "marker_index": 64,
            "region_id": 303,
            "pressure_owner_index": 64,
            "position_m": (1.5, 0.4, 5.5),
            "velocity_mps": (0.25, 0.0, 0.0),
        },
        {
            "marker_index": 65,
            "region_id": 303,
            "pressure_owner_index": 65,
            "position_m": (1.5, 0.5, 5.5),
            "velocity_mps": (0.25, 0.0, 0.0),
        },
        {
            "marker_index": 66,
            "region_id": 303,
            "pressure_owner_index": 66,
            "position_m": (1.5, 0.6, 5.5),
            "velocity_mps": (0.25, 0.0, 0.0),
        },
    )
    assert first_author["normal"] is None
    first_post = first_author["post_admission"]
    assert first_post["available"] is False
    assert first_post["reason"] == "invalid_author_normal"
    for field_name in (
        "normal_norm",
        "normal_norm_valid",
        "nominal_progress_m",
        "nominal_progress_positive",
        "actual_progress_m",
        "actual_progress_positive",
        "nominal_lateral_squared_m2",
        "nominal_lateral_limit_squared_m2",
        "nominal_lateral_valid",
        "actual_lateral_squared_m2",
        "actual_lateral_limit_squared_m2",
        "actual_lateral_valid",
        "probe_margin_m",
        "probe_margin_finite",
    ):
        assert first_post[field_name] is None
    assert second_author["normal"] == (0.0, 0.0, -1.0)
    assert second_author["post_admission"]["available"] is True
    assert second_author["post_admission"]["normal_norm"] == 1.0
    assert diagnostic["post_admission_pair"]["available"] is False
    assert diagnostic["post_admission_pair"]["reason"] == "invalid_author_normal"
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize(
    "inactive_normal_component",
    (float("nan"), float("inf")),
    ids=("inactive-x-nan", "inactive-x-infinite"),
)
def test_production_conflict_diagnostic_ignores_nonfinite_inactive_normal_component(
    inactive_normal_component: float,
) -> None:
    boundary = _Boundary()
    with pytest.raises(RuntimeError, match="original target conflict"):
        boundary.assemble_velocity_dirichlet_component_face_ledger(
            **_runtime_fields()
        )
    boundary.pressure_neumann_normal_field = _Field(
        {
            boundary.first_row: (inactive_normal_component, 0.0, -2.0),
            boundary.second_row: (9.0, 0.0, -2.0),
        }
    )

    diagnostic = (
        boundary._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
    )

    first_author, second_author = diagnostic["authors"]
    assert first_author["normal"] == (0.0, 0.0, -1.0)
    first_post = first_author["post_admission"]
    assert first_post["available"] is True
    assert first_post["normal_norm"] == 1.0
    assert first_post["normal_norm_valid"] is True
    assert first_post["nominal_progress_m"] == 2.0
    assert first_post["actual_progress_m"] == 2.0
    assert first_post["nominal_lateral_squared_m2"] == 0.0
    assert first_post["actual_lateral_squared_m2"] == 0.0
    assert first_post["probe_margin_m"] == 2.0
    assert second_author["normal"] == (0.0, 0.0, -1.0)
    assert second_author["post_admission"]["available"] is True
    post_admission_pair = diagnostic["post_admission_pair"]
    assert post_admission_pair["available"] is True
    assert post_admission_pair["first_raw_inputs_finite"] is True
    assert post_admission_pair["failed_predicates"] == (
        "second_actual_lateral_valid",
    )
    json.dumps(diagnostic, allow_nan=False)


def test_production_conflict_diagnostic_reports_pre_admission_pair_host_metrics(
) -> None:
    boundary = _Boundary()
    fields = _runtime_fields()
    fields["search"].node_projection_marker_weights = _Field(
        {
            boundary.first_row: (0.25, 0.75, 0.0),
            boundary.second_row: (0.75, 0.25, 0.0),
        }
    )
    fields["search"].nearest_marker = _Field(
        {boundary.first_row: 65, boundary.second_row: 65}
    )
    with pytest.raises(RuntimeError, match="original target conflict"):
        boundary.assemble_velocity_dirichlet_component_face_ledger(**fields)
    boundary.velocity_dirichlet_component_face_segment_pair_admission_valid = _Field(
        {boundary.pair_index: 0}
    )
    boundary.velocity_dirichlet_component_face_adjacent_direct_pair_target_valid = (
        _Field({boundary.pair_index: 1})
    )

    diagnostic = (
        boundary._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
    )

    pre_admission_pair = diagnostic["pre_admission_pair"]
    assert pre_admission_pair["available"] is True
    assert pre_admission_pair["parity_claimed"] is False
    assert pre_admission_pair["kernel_predicate_parity_guaranteed"] is False
    assert (
        pre_admission_pair["source_precision"]
        == "f32_field_read_promoted_to_python_float_recomputed_f64"
    )
    assert pre_admission_pair["source"] == "raw_transaction_fields_and_context"
    assert pre_admission_pair["guard"] == {
        "prepare_pair_path": True,
        "forward_direct_direct": True,
        "selected_storage_offsets": [1, 0],
        "adjacent_direct_pair_target_valid": True,
        "pair_cache_admission_valid": False,
    }
    assert pre_admission_pair["raw_authors"][0]["source_row"] == [0, 0, 5]
    assert pre_admission_pair["raw_authors"][0]["raw_normal"] == [9.0, 0.0, -2.0]
    assert pre_admission_pair["raw_authors"][1]["actual_sample_valid"] is True
    assert pre_admission_pair["topology"] == {
        "projection_segment_count": 2,
        "topology_available": True,
        "same_segment": False,
        "adjacent_segments": True,
        "shared_vertex_index": 65,
        "first_segment_seen_count": 1,
        "second_segment_seen_count": 1,
        "shared_incident_segment_count": 2,
    }
    assert pre_admission_pair["tie"]["adjacent_distance_tie"] is True
    assert pre_admission_pair["shared_vertex"]["coownership_valid"] is True
    assert pre_admission_pair["predicates"][0]["name"] == "projection_weights_valid"
    assert pre_admission_pair["failed_predicates"] == []
    assert pre_admission_pair["first_failed_predicate"] is None
    json.dumps(diagnostic, allow_nan=False)


def test_production_conflict_diagnostic_preserves_base_when_pre_admission_read_fails(
) -> None:
    boundary = _Boundary()
    fields = _runtime_fields()
    with pytest.raises(RuntimeError, match="original target conflict"):
        boundary.assemble_velocity_dirichlet_component_face_ledger(**fields)
    boundary.velocity_dirichlet_component_face_segment_pair_admission_valid = _Field(
        {boundary.pair_index: 0}
    )
    boundary.velocity_dirichlet_component_face_adjacent_direct_pair_target_valid = (
        _Field({boundary.pair_index: 1})
    )
    boundary.__dict__["_canonical_velocity_dirichlet_precommit_diagnostic_context"][
        "projection_segment_indices"
    ] = _ReadRaisesField()

    diagnostic = (
        boundary._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
    )

    assert diagnostic["component_face"] == boundary.face
    assert diagnostic["author_linear_keys"] == (boundary.first_key, boundary.second_key)
    assert [author["source_row"] for author in diagnostic["authors"]] == [
        boundary.first_row,
        boundary.second_row,
    ]
    assert [marker["marker_index"] for marker in diagnostic["markers"]] == [64, 65, 66]
    assert diagnostic["pre_admission_pair"] == {
        "available": False,
        "reason": "raw_field_read_failed",
    }
    assert diagnostic["pre_admission_pair_capture_error"]["type"] == "AssertionError"
    assert diagnostic["pre_admission_pair_capture_error"]["message"]
    json.dumps(diagnostic, allow_nan=False)


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
