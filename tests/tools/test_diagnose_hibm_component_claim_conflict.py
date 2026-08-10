from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools import diagnose_hibm_component_claim_conflict as diagnostic


class _HostSnapshotOnlyField:
    """Test double that makes every accidental Taichi scalar read visible."""

    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)
        self.readback_count = 0

    def to_numpy(self) -> np.ndarray:
        self.readback_count += 1
        return self._values.copy()

    def __getitem__(self, _index: object) -> object:
        raise AssertionError("diagnostic must not perform scalar field reads")


class HibmComponentClaimConflictDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        diagnostic._ASSEMBLY_CONTEXT = None
        if hasattr(diagnostic, "_CAPTURED"):
            diagnostic._CAPTURED = False

    def _run_main(
        self,
        *,
        output_path: Path,
        boundary: object,
        validate: object,
        assemble: object,
        capture: mock.Mock,
    ) -> None:
        runner_script = output_path.parent / "synthetic_runner.py"

        def run_synthetic_runner(*_args: object, **_kwargs: object) -> None:
            installed_assemble = (
                diagnostic.HibmMpmIbBoundaryConditions
                .assemble_velocity_dirichlet_component_face_ledger
            )
            installed_assemble(
                boundary,
                interpolate_interior_velocity=True,
                synthetic_context_token="context-is-live",
            )

        argv = [
            "diagnose_hibm_component_claim_conflict.py",
            "--diagnostic-output",
            str(output_path),
            "--runner-script",
            str(runner_script),
        ]
        with (
            mock.patch.object(
                diagnostic.HibmMpmIbBoundaryConditions,
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
                validate,
            ),
            mock.patch.object(
                diagnostic.HibmMpmIbBoundaryConditions,
                "assemble_velocity_dirichlet_component_face_ledger",
                assemble,
            ),
            mock.patch.object(diagnostic, "_capture", capture),
            mock.patch.object(diagnostic.runpy, "run_path", run_synthetic_runner),
            mock.patch.object(sys, "argv", argv),
        ):
            diagnostic.main()
            self.assertEqual(sys.argv, argv)

    def test_successful_validations_never_capture_or_write(self) -> None:
        events: list[str] = []
        boundary = object()

        def validate(_boundary: object) -> None:
            events.append("validate")

        def assemble(_boundary: object, **_kwargs: object) -> dict[str, object]:
            installed_validate = (
                diagnostic.HibmMpmIbBoundaryConditions
                ._validate_canonical_velocity_dirichlet_target_conflict_precommit
            )
            installed_validate(_boundary)
            installed_validate(_boundary)
            return {}

        capture = mock.Mock(side_effect=AssertionError("healthy path captured"))
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "claims.json"

            self._run_main(
                output_path=output_path,
                boundary=boundary,
                validate=validate,
                assemble=assemble,
                capture=capture,
            )

            self.assertEqual(events, ["validate", "validate"])
            capture.assert_not_called()
            self.assertFalse(output_path.exists())

    def test_atomic_writer_emits_strict_json_for_nonfinite_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "claims.json"
            diagnostic._write_json_atomic(
                output_path,
                {"finite": 1.0, "not_finite": math.nan},
            )

            raw = output_path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertIsNone(json.loads(raw)["not_finite"])

    def test_target_conflict_captures_once_before_scratch_cleanup_and_reraises(self) -> None:
        events: list[str] = []
        boundary = object()
        solver_failure = RuntimeError("sentinel target conflict")

        def validate(_boundary: object) -> None:
            events.append("validate")
            raise solver_failure

        def assemble(_boundary: object, **_kwargs: object) -> dict[str, object]:
            installed_validate = (
                diagnostic.HibmMpmIbBoundaryConditions
                ._validate_canonical_velocity_dirichlet_target_conflict_precommit
            )
            try:
                installed_validate(_boundary)
            finally:
                events.append("scratch_cleanup")
            return {}

        def capture_payload(_boundary: object) -> dict[str, object]:
            self.assertIsNotNone(diagnostic._ASSEMBLY_CONTEXT)
            self.assertEqual(
                diagnostic._ASSEMBLY_CONTEXT["synthetic_context_token"],
                "context-is-live",
            )
            events.append("capture")
            return {"schema_version": 1, "target_conflict_count": 20}

        capture = mock.Mock(side_effect=capture_payload)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "claims.json"

            with self.assertRaises(RuntimeError) as raised:
                self._run_main(
                    output_path=output_path,
                    boundary=boundary,
                    validate=validate,
                    assemble=assemble,
                    capture=capture,
                )

            self.assertIs(raised.exception, solver_failure)
            self.assertEqual(events, ["validate", "capture", "scratch_cleanup"])
            capture.assert_called_once_with(boundary)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["target_conflict_count"], 20)
            self.assertTrue(payload["interpolate_interior_velocity"])
            self.assertFalse(output_path.with_suffix(".json.tmp").exists())

    def test_capture_failure_never_masks_original_solver_failure(self) -> None:
        events: list[str] = []
        boundary = object()
        solver_failure = RuntimeError("sentinel target conflict")

        def validate(_boundary: object) -> None:
            events.append("validate")
            raise solver_failure

        def assemble(_boundary: object, **_kwargs: object) -> dict[str, object]:
            installed_validate = (
                diagnostic.HibmMpmIbBoundaryConditions
                ._validate_canonical_velocity_dirichlet_target_conflict_precommit
            )
            try:
                installed_validate(_boundary)
            finally:
                events.append("scratch_cleanup")
            return {}

        capture = mock.Mock(side_effect=OSError("diagnostic readback failed"))
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "claims.json"

            with self.assertRaises(RuntimeError) as raised:
                self._run_main(
                    output_path=output_path,
                    boundary=boundary,
                    validate=validate,
                    assemble=assemble,
                    capture=capture,
                )

            self.assertIs(raised.exception, solver_failure)
            self.assertEqual(events, ["validate", "scratch_cleanup"])
            capture.assert_called_once_with(boundary)
            diagnostic_error = str(
                getattr(raised.exception, "diagnostic_capture_error", "")
            )
            notes = "\n".join(getattr(raised.exception, "__notes__", ()))
            self.assertIn(
                "diagnostic readback failed",
                diagnostic_error + notes,
            )
            self.assertFalse(output_path.exists())

    def test_capture_guard_is_scoped_to_one_runner_invocation(self) -> None:
        boundary = object()
        solver_failure = RuntimeError("sentinel target conflict")

        def validate(_boundary: object) -> None:
            raise solver_failure

        def assemble(_boundary: object, **_kwargs: object) -> dict[str, object]:
            installed_validate = (
                diagnostic.HibmMpmIbBoundaryConditions
                ._validate_canonical_velocity_dirichlet_target_conflict_precommit
            )
            installed_validate(_boundary)
            return {}

        capture = mock.Mock(
            return_value={"schema_version": 2, "target_conflict_count": 20}
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for run_index in range(2):
                with self.assertRaises(RuntimeError) as raised:
                    self._run_main(
                        output_path=base / f"claims_{run_index}.json",
                        boundary=boundary,
                        validate=validate,
                        assemble=assemble,
                        capture=capture,
                    )
                self.assertIs(raised.exception, solver_failure)

        self.assertEqual(capture.call_count, 2)

    def test_author_witness_decoder_recovers_exact_path_and_source_rows(self) -> None:
        grid_nodes = (4, 256, 320)
        node_count = grid_nodes[0] * grid_nodes[1] * grid_nodes[2]
        author_keys = (41115, 41435)

        def encode(author_key: int, path_code: int) -> int:
            reverse_author_key = node_count - 1 - author_key
            return -2 - ((3 - path_code) * node_count + reverse_author_key)

        decoded = diagnostic._decode_conflict_author_witnesses(
            (
                encode(author_keys[0], 0),
                encode(author_keys[1], 0),
            ),
            grid_nodes=grid_nodes,
        )

        self.assertEqual(decoded["conflict_path_code"], 0)
        self.assertEqual(decoded["author_linear_keys"], list(author_keys))
        self.assertEqual(
            decoded["author_source_rows"],
            [[0, 128, 155], [0, 129, 155]],
        )
        self.assertEqual(decoded["decode_errors"], [])

    def test_author_witness_decoder_rejects_invalid_or_mixed_paths(self) -> None:
        grid_nodes = (4, 256, 320)
        node_count = grid_nodes[0] * grid_nodes[1] * grid_nodes[2]

        def encode(author_key: int, path_code: int) -> int:
            reverse_author_key = node_count - 1 - author_key
            return -2 - ((3 - path_code) * node_count + reverse_author_key)

        mixed = diagnostic._decode_conflict_author_witnesses(
            (encode(41115, 0), encode(41435, 2)),
            grid_nodes=grid_nodes,
        )
        invalid = diagnostic._decode_conflict_author_witnesses(
            (-1, encode(41435, 0)),
            grid_nodes=grid_nodes,
        )

        self.assertTrue(mixed["decode_errors"])
        self.assertTrue(invalid["decode_errors"])

    def _production_shaped_lane(self) -> dict[str, object]:
        marker_positions = [
            [0.001500000013038516, 0.010010981932282448, 0.04992637783288956],
            [0.001500000013038516, 0.009988013654947281, 0.04692742973566055],
        ]
        marker_velocities = [
            [-1.3704311296880434e-10, 0.02048536017537117, -0.10759842395782471],
            [1.3237330502569034e-11, -0.025348100811243057, -0.10963963717222214],
        ]
        author_payloads = [
            {
                "source_row": [0, 128, 155],
                "active_ib_node": True,
                "obstacle": False,
                "actual_sample_valid": True,
                "boundary_point_m": [
                    0.000375000003259629,
                    0.010000777430832386,
                    0.04859404265880585,
                ],
                "nominal_interior_point_m": [
                    0.000375000003259629,
                    0.010200777430832386,
                    0.04859404265880585,
                ],
                "actual_sample_point_m": [
                    0.000375000003259629,
                    0.010200777430832386,
                    0.04859404265880585,
                ],
                "actual_sample_velocity_mps": [0.0, 0.4, 0.0],
                "normal": [0.0, 1.0, 0.0],
                "nearest_marker": 130,
                "nearest_marker_region_id": 303,
                "projection_marker_indices": [130, 131, -1],
                "projection_marker_weights": [
                    0.5557321310043335,
                    0.4442678987979889,
                    0.0,
                ],
                "projection_marker_positions_m": marker_positions,
                "projection_marker_velocities_mps": marker_velocities,
                "projection_marker_regions": [303, 303],
                "serialized_target_mps": [
                    -7.02783387040995e-11,
                    0.00012302510731387883,
                    -0.10850527137517929,
                ],
            },
            {
                "source_row": [0, 129, 155],
                "active_ib_node": True,
                "obstacle": False,
                "actual_sample_valid": True,
                "boundary_point_m": [
                    0.000375000003259629,
                    0.010000782087445259,
                    0.0485946387052536,
                ],
                "nominal_interior_point_m": [
                    0.000375000003259629,
                    0.010200782087445259,
                    0.0485946387052536,
                ],
                "actual_sample_point_m": [
                    0.000375000003259629,
                    0.010200782087445259,
                    0.0485946387052536,
                ],
                "actual_sample_velocity_mps": [0.0, 0.4, 0.0],
                "normal": [0.0, 1.0, 0.0],
                "nearest_marker": 130,
                "nearest_marker_region_id": 303,
                "projection_marker_indices": [130, 131, -1],
                "projection_marker_weights": [
                    0.555931568145752,
                    0.44406840205192566,
                    0.0,
                ],
                "projection_marker_positions_m": marker_positions,
                "projection_marker_velocities_mps": marker_velocities,
                "projection_marker_regions": [303, 303],
                "serialized_target_mps": [
                    -7.030831472576438e-11,
                    0.00013216768275015056,
                    -0.10850485414266586,
                ],
            },
        ]
        return {
            "component_face": [0, 129, 155],
            "component_axis": 1,
            "conflict_path_code": 0,
            "claim_count": 2,
            "claim_region_id": 303,
            "interpolate_interior_velocity": True,
            "inactive_axis": 0,
            "face_center_m": [0.000375, 0.010078125, 0.04859375],
            "target_cell_width_m": [0.00075, 0.000078125, 0.0003125],
            "authors": author_payloads,
        }

    def test_face_projected_pair_classifies_production_shaped_distinct_anchors(
        self,
    ) -> None:
        derived = diagnostic._classify_face_projected_segment_pair(
            self._production_shaped_lane()
        )

        self.assertEqual(
            derived["classification"],
            "same_segment_distinct_anchor_face_projected_bracket",
        )
        self.assertTrue(derived["segment_parameter_bracketed"])
        self.assertFalse(derived["component_axis_coordinate_bracketed"])
        self.assertAlmostEqual(
            derived["face_segment_parameter"],
            0.4441675623650562,
            places=10,
        )
        self.assertAlmostEqual(
            derived["author_segment_parameters"][0],
            0.4442678987979889,
            places=9,
        )
        self.assertAlmostEqual(
            derived["author_segment_parameters"][1],
            0.44406840205192566,
            places=9,
        )
        self.assertGreater(derived["segment_parameter_bracket_margin"], 0.0)

    def test_face_projected_pair_rejects_unbracketed_face(self) -> None:
        lane = copy.deepcopy(self._production_shaped_lane())
        lane["face_center_m"][2] = 0.0480

        derived = diagnostic._classify_face_projected_segment_pair(lane)

        self.assertFalse(derived["segment_parameter_bracketed"])
        self.assertNotEqual(
            derived["classification"],
            "same_segment_distinct_anchor_face_projected_bracket",
        )

    def test_face_projected_pair_rejects_degenerate_or_misaligned_probe_ray(
        self,
    ) -> None:
        degenerate = self._production_shaped_lane()
        degenerate["authors"][0]["actual_sample_point_m"] = list(
            degenerate["authors"][0]["boundary_point_m"]
        )
        misaligned = self._production_shaped_lane()
        for author in misaligned["authors"]:
            author["normal"] = [0.0, 0.0, 1.0]

        for lane in (degenerate, misaligned):
            derived = diagnostic._classify_face_projected_segment_pair(lane)
            self.assertNotEqual(
                derived["classification"],
                "same_segment_distinct_anchor_face_projected_bracket",
            )

    def test_face_projected_pair_requires_face_axis_bracketing_to_fail(self) -> None:
        lane = self._production_shaped_lane()
        author_points = [
            author["boundary_point_m"] for author in lane["authors"]
        ]
        lane["face_center_m"][1] = sum(point[1] for point in author_points) / 2.0

        derived = diagnostic._classify_face_projected_segment_pair(lane)

        self.assertTrue(derived["component_axis_coordinate_bracketed"])
        self.assertNotEqual(
            derived["classification"],
            "same_segment_distinct_anchor_face_projected_bracket",
        )

    def test_face_projected_pair_rejects_endpoint_weight_and_negative_progress(
        self,
    ) -> None:
        endpoint = self._production_shaped_lane()
        endpoint_author = endpoint["authors"][0]
        endpoint_author["projection_marker_weights"] = [1.0, 0.0, 0.0]
        endpoint_author["serialized_target_mps"] = list(
            endpoint_author["projection_marker_velocities_mps"][0]
        )
        endpoint_derived = diagnostic._classify_face_projected_segment_pair(
            endpoint
        )
        self.assertIn(
            "author 0 uses an endpoint-clamped segment weight",
            endpoint_derived["classification_failures"],
        )

        negative_progress = self._production_shaped_lane()
        negative_progress["face_center_m"][1] = 0.0099
        progress_derived = diagnostic._classify_face_projected_segment_pair(
            negative_progress
        )
        self.assertFalse(progress_derived["probe_rays_are_valid"])
        self.assertNotEqual(
            progress_derived["classification"],
            "same_segment_distinct_anchor_face_projected_bracket",
        )

    def test_aggregate_gate_requires_all_20_unique_complete_matching_lanes(
        self,
    ) -> None:
        matching_lane = {
            "decode_errors": [],
            "derived": {
                "classification": (
                    "same_segment_distinct_anchor_face_projected_bracket"
                )
            },
        }
        indices = [(0, index, 0, 1) for index in range(20)]
        matching_lanes = [copy.deepcopy(matching_lane) for _ in range(20)]
        complete = diagnostic._summarize_witness_capture(
            target_conflict_event_count=20,
            witness_indices=indices,
            selected_witness_count=20,
            lanes=matching_lanes,
            lane_capture_errors=[],
        )
        self.assertTrue(complete["capture_complete"])
        self.assertTrue(
            complete[
                "all_target_conflicts_match_face_projected_segment_invariant"
            ]
        )

        cases = {
            "nineteen_of_twenty": {
                "witness_indices": indices,
                "selected_witness_count": 20,
                "lanes": matching_lanes[:-1]
                + [
                    {
                        "decode_errors": [],
                        "derived": {
                            "classification": "different_geometry"
                        },
                    }
                ],
            },
            "duplicate_witness_lane": {
                "witness_indices": indices[:-1] + [indices[-2]],
                "selected_witness_count": 20,
                "lanes": matching_lanes,
            },
            "truncated": {
                "witness_indices": indices,
                "selected_witness_count": 19,
                "lanes": matching_lanes[:-1],
            },
            "decode_error": {
                "witness_indices": indices,
                "selected_witness_count": 20,
                "lanes": [
                    {**copy.deepcopy(matching_lane), "decode_errors": ["bad"]},
                    *[copy.deepcopy(matching_lane) for _ in range(19)],
                ],
            },
            "empty": {
                "witness_indices": [],
                "selected_witness_count": 0,
                "lanes": [],
            },
            "capture_overflow": {
                "witness_indices": [
                    (0, index, 0, 1) for index in range(65)
                ],
                "selected_witness_count": 64,
                "lanes": [
                    copy.deepcopy(matching_lane) for _ in range(64)
                ],
            },
        }
        for name, case in cases.items():
            with self.subTest(name=name):
                summary = diagnostic._summarize_witness_capture(
                    target_conflict_event_count=(
                        65 if name == "capture_overflow" else 20
                    ),
                    lane_capture_errors=[],
                    **case,
                )
                self.assertFalse(
                    summary[
                        "all_target_conflicts_match_face_projected_segment_invariant"
                    ]
                )

    def test_capture_reads_all_lane_fields_from_one_host_snapshot(self) -> None:
        grid_nodes = (2, 2, 2)
        scalar_shape = grid_nodes
        vector_shape = (*grid_nodes, 3)
        first_author = np.zeros(vector_shape, dtype=np.int32)
        second_author = np.zeros(vector_shape, dtype=np.int32)
        # Path 2, authors (0, 1, 0) and (0, 1, 1): the exact adjacent
        # pair for the component-z face (0, 1, 1).
        first_author[0, 1, 1, 2] = -15
        second_author[0, 1, 1, 2] = -14
        claim_count = np.zeros(vector_shape, dtype=np.int32)
        claim_count[0, 1, 1, 2] = 2

        fields = {
            "first_author": _HostSnapshotOnlyField(first_author),
            "second_author": _HostSnapshotOnlyField(second_author),
            "claim_count": _HostSnapshotOnlyField(claim_count),
            "claim_target": _HostSnapshotOnlyField(np.zeros(vector_shape)),
            "claim_region": _HostSnapshotOnlyField(np.zeros(vector_shape, dtype=np.int32)),
            "claim_alpha": _HostSnapshotOnlyField(np.zeros(vector_shape)),
            "active_ib_node": _HostSnapshotOnlyField(np.ones(scalar_shape, dtype=np.int32)),
            "actual_sample_valid": _HostSnapshotOnlyField(np.ones(scalar_shape, dtype=np.int32)),
            "actual_sample_point": _HostSnapshotOnlyField(np.zeros(vector_shape)),
            "actual_sample_velocity": _HostSnapshotOnlyField(np.zeros(vector_shape)),
            "normal": _HostSnapshotOnlyField(np.zeros(vector_shape)),
            "dirichlet_velocity": _HostSnapshotOnlyField(np.zeros(vector_shape)),
            "obstacle": _HostSnapshotOnlyField(np.zeros(scalar_shape, dtype=np.int32)),
            "projection_indices": _HostSnapshotOnlyField(
                np.broadcast_to(np.array([0, 0, -1], dtype=np.int32), vector_shape)
            ),
            "projection_weights": _HostSnapshotOnlyField(
                np.broadcast_to(np.array([0.5, 0.5, 0.0]), vector_shape)
            ),
            "nearest_marker": _HostSnapshotOnlyField(np.zeros(scalar_shape, dtype=np.int32)),
            "boundary_point": _HostSnapshotOnlyField(np.zeros(vector_shape)),
            "interior_point": _HostSnapshotOnlyField(np.zeros(vector_shape)),
            "marker_region": _HostSnapshotOnlyField(np.zeros(1, dtype=np.int32)),
            "marker_position": _HostSnapshotOnlyField(np.zeros((1, 3))),
            "marker_velocity": _HostSnapshotOnlyField(np.zeros((1, 3))),
            "face_x": _HostSnapshotOnlyField(np.array([0.0, 1.0, 2.0])),
            "face_y": _HostSnapshotOnlyField(np.array([0.0, 1.0, 2.0])),
            "face_z": _HostSnapshotOnlyField(np.array([0.0, 1.0, 2.0])),
            "center_x": _HostSnapshotOnlyField(np.array([0.5, 1.5])),
            "center_y": _HostSnapshotOnlyField(np.array([0.5, 1.5])),
            "center_z": _HostSnapshotOnlyField(np.array([0.5, 1.5])),
            "target_conflict_report": _HostSnapshotOnlyField(np.asarray(1, dtype=np.int32)),
            "region_conflict_report": _HostSnapshotOnlyField(np.asarray(0, dtype=np.int32)),
            "alpha_conflict_report": _HostSnapshotOnlyField(np.asarray(0, dtype=np.int32)),
            "claim_conflict_report": _HostSnapshotOnlyField(np.asarray(0, dtype=np.int32)),
            "duplicate_claim_report": _HostSnapshotOnlyField(np.asarray(0, dtype=np.int32)),
        }
        search = SimpleNamespace(
            node_projection_marker_indices=fields["projection_indices"],
            node_projection_marker_weights=fields["projection_weights"],
            nearest_marker=fields["nearest_marker"],
            node_boundary_point_m=fields["boundary_point"],
            node_interior_fluid_point_m=fields["interior_point"],
        )
        boundary = SimpleNamespace(
            grid_nodes=grid_nodes,
            marker_capacity=1,
            velocity_dirichlet_component_face_segment_first_author_linear_key=fields["first_author"],
            velocity_dirichlet_component_face_segment_second_author_linear_key=fields["second_author"],
            velocity_dirichlet_component_face_claim_count=fields["claim_count"],
            velocity_dirichlet_component_face_claim_target_mps=fields["claim_target"],
            velocity_dirichlet_component_face_claim_region_id=fields["claim_region"],
            velocity_dirichlet_component_face_claim_alpha=fields["claim_alpha"],
            active_ib_node=fields["active_ib_node"],
            velocity_dirichlet_component_face_actual_sample_valid=fields["actual_sample_valid"],
            velocity_dirichlet_component_face_actual_sample_point_m=fields["actual_sample_point"],
            velocity_dirichlet_component_face_actual_sample_velocity_mps=fields["actual_sample_velocity"],
            pressure_neumann_normal_field=fields["normal"],
            velocity_dirichlet_mps_field=fields["dirichlet_velocity"],
            report_velocity_dirichlet_component_face_target_conflict_count=fields["target_conflict_report"],
            report_velocity_dirichlet_component_face_region_conflict_count=fields["region_conflict_report"],
            report_velocity_dirichlet_component_face_alpha_conflict_count=fields["alpha_conflict_report"],
            report_velocity_dirichlet_component_face_conflict_count=fields["claim_conflict_report"],
            report_velocity_dirichlet_component_face_duplicate_claim_count=fields["duplicate_claim_report"],
        )
        diagnostic._ASSEMBLY_CONTEXT = {
            "obstacle_field": fields["obstacle"],
            "cell_face_x_m": fields["face_x"],
            "cell_face_y_m": fields["face_y"],
            "cell_face_z_m": fields["face_z"],
            "cell_center_x_m": fields["center_x"],
            "cell_center_y_m": fields["center_y"],
            "cell_center_z_m": fields["center_z"],
            "interpolate_interior_velocity": True,
        }
        boundary._canonical_velocity_dirichlet_precommit_diagnostic_context = {
            "search": search,
            "marker_region_id": fields["marker_region"],
            "marker_position_m": fields["marker_position"],
            "marker_velocity_mps": fields["marker_velocity"],
            "marker_geometry_available": True,
            "inactive_axis": 0,
        }

        payload = diagnostic._capture(boundary)

        self.assertEqual(payload["target_conflict_event_count"], 1)
        self.assertEqual(
            payload["capture_policy"],
            "failure_only_bounded_bulk_host_witness_capture",
        )
        self.assertEqual(payload["lane_capture_errors"], [])
        self.assertEqual(len(payload["witness_lanes"]), 1)
        self.assertTrue(
            all(field.readback_count == 1 for field in fields.values())
        )

if __name__ == "__main__":
    unittest.main()
