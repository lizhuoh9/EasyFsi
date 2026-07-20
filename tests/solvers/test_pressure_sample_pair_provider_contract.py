from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from simulation_core.coupling.pressure_sample_pairs import (
    PressureSamplePair,
    PressureSamplePairMap,
    RuntimeAnchoredCellPairProvider,
    compute_runtime_anchored_cell_pair_map,
    pressure_sample_pair_map_sha256,
    pressure_sample_pair_map_from_pairs,
)


class PressureSamplePairProviderContractTests(unittest.TestCase):
    def test_runtime_anchored_pair_map_is_deterministic(self) -> None:
        first = _runtime_pair_map()
        second = _runtime_pair_map()

        self.assertEqual(first.pair_map_sha256, second.pair_map_sha256)
        self.assertEqual(first.provider_mode, "runtime_anchored_cell_pair")
        self.assertEqual(first.fallback_count, 0)
        self.assertEqual(first.selected_count, 2)
        self.assertEqual(first.inside_cells, ((2, 0, 32), (2, 0, 32)))
        self.assertEqual(first.outside_cells, ((2, 0, 35), (2, 0, 30)))

    def test_pair_schema_and_counts_are_explicit(self) -> None:
        pair_map = pressure_sample_pair_map_from_pairs(
            (
                PressureSamplePair(
                    marker_index=0,
                    region_id="primary",
                    inside_cell=(1, 2, 3),
                    outside_cell=(1, 2, 4),
                    sample_status="runtime_generated",
                    fallback_status="no_fallback",
                    diagnostic_reason="unit_test",
                ),
                PressureSamplePair(
                    marker_index=1,
                    region_id="secondary",
                    inside_cell=(1, 2, 3),
                    outside_cell=(1, 2, 2),
                    sample_status="missing",
                    fallback_status="fallback_used",
                    diagnostic_reason="unit_test",
                ),
            ),
            provider_mode="runtime_anchored_cell_pair",
        )

        diagnostics = pair_map.as_diagnostics()
        self.assertEqual(pair_map.selected_count, 1)
        self.assertEqual(pair_map.fallback_count, 1)
        self.assertEqual(
            set(diagnostics["pairs"][0]),
            {
                "marker_index",
                "region_id",
                "inside_cell",
                "outside_cell",
                "sample_status",
                "fallback_status",
                "diagnostic_reason",
            },
        )

    def test_runtime_provider_accepts_generic_marker_geometry_mapping(self) -> None:
        provider = RuntimeAnchoredCellPairProvider(
            domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            grid_nodes=(4, 4, 4),
            anchor_axis=2,
            inside_axis_position_m=0.5,
            outside_axis_offset_cells=1,
        )

        pair_map = provider.compute_pairs(
            {
                "marker_positions_m": ((0.5, 0.5, 0.75),),
                "marker_normals": ((0.0, 0.0, 1.0),),
                "marker_region_ids": ("primary",),
            }
        )

        self.assertEqual(pair_map.provider_mode, "runtime_anchored_cell_pair")
        self.assertEqual(pair_map.selected_count, 1)
        self.assertEqual(pair_map.fallback_count, 0)
        self.assertTrue(pair_map.marker_geometry_sha256)
        self.assertEqual(pair_map.inside_cells, ((2, 2, 2),))
        self.assertEqual(pair_map.outside_cells, ((2, 2, 3),))

    def test_pair_map_rejects_stale_geometry_revision_and_hash(self) -> None:
        provider = _moving_interface_provider()
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        original = _moving_marker_geometry(z_m=0.5, revision=7)
        pair_map = provider.compute_pairs(
            original,
            fluid_state={"obstacle": obstacle},
        )

        pair_map.require_current_marker_geometry(original)
        self.assertEqual(pair_map.marker_geometry_revision, 7)

        with self.assertRaisesRegex(ValueError, "revision"):
            pair_map.require_current_marker_geometry(
                _moving_marker_geometry(z_m=0.625, revision=8)
            )
        with self.assertRaisesRegex(ValueError, "hash"):
            pair_map.require_current_marker_geometry(
                _moving_marker_geometry(z_m=0.625, revision=7)
            )

    def test_dynamic_pairs_are_on_declared_fluid_sides_and_not_obstacles(self) -> None:
        provider = _moving_interface_provider()
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[4, 4, 3:7] = 1
        geometry = _moving_marker_geometry(z_m=0.5, revision=11)

        pair_map = provider.compute_pairs(
            geometry,
            fluid_state={"obstacle": obstacle},
        )

        inside = pair_map.inside_cells[0]
        outside = pair_map.outside_cells[0]
        marker_cell_z = 4
        self.assertLess(inside[2], marker_cell_z)
        self.assertGreater(outside[2], marker_cell_z)
        self.assertEqual(int(obstacle[inside]), 0)
        self.assertEqual(int(obstacle[outside]), 0)
        self.assertEqual(pair_map.pairs[0].diagnostic_reason, "runtime_dynamic_fluid_side_cell_pair")

    def test_linear_pressure_field_load_is_translation_invariant_after_refresh(self) -> None:
        provider = _moving_interface_provider()
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        first = provider.compute_pairs(
            _moving_marker_geometry(z_m=0.4375, revision=3),
            fluid_state={"obstacle": obstacle},
        )
        translated = provider.compute_pairs(
            _moving_marker_geometry(z_m=0.5625, revision=4),
            fluid_state={"obstacle": obstacle},
        )

        def pressure_pa(cell: tuple[int, int, int]) -> float:
            z_center_m = (float(cell[2]) + 0.5) / 8.0
            return 13.0 + 24.0 * z_center_m

        def marker_load_z_n(pair_map: PressureSamplePairMap) -> float:
            pair = pair_map.pairs[0]
            pressure_jump_pa = pressure_pa(pair.inside_cell) - pressure_pa(
                pair.outside_cell
            )
            return pressure_jump_pa * 1.0 * 0.25

        self.assertEqual(first.inside_cells, ((4, 4, 2),))
        self.assertEqual(first.outside_cells, ((4, 4, 4),))
        self.assertEqual(translated.inside_cells, ((4, 4, 3),))
        self.assertEqual(translated.outside_cells, ((4, 4, 5),))
        self.assertAlmostEqual(marker_load_z_n(first), -1.5)
        self.assertAlmostEqual(marker_load_z_n(translated), -1.5)

    def test_runner_refresh_retires_old_map_before_transactional_install(self) -> None:
        from benchmarks.official import solid_mpm_fsi_runner

        markers = _RecordingAnchorMarkers()
        pair_map = _RecordingPairMap(markers.events)
        with (
            patch.object(
                solid_mpm_fsi_runner,
                "_is_selected_traction_formulation_coupled_smoke",
                return_value=True,
            ),
            patch.object(
                solid_mpm_fsi_runner,
                "_traction_pressure_pair_runtime_provider_mode",
                return_value="runtime_anchored_cell_pair",
            ),
            patch.object(
                solid_mpm_fsi_runner,
                "_runtime_pressure_pair_anchor_map",
                return_value=pair_map,
            ),
        ):
            refreshed = solid_mpm_fsi_runner._refresh_runtime_pressure_pair_anchor_markers(
                markers,
                object(),
                object(),
                refresh_count=6,
            )

        self.assertIsNotNone(refreshed)
        report, installed_map = refreshed
        self.assertIs(installed_map, pair_map)
        self.assertEqual(markers.events, ["reset", "require-current", "set"])
        self.assertEqual(report["pressure_pair_anchor_current_marker_geometry_sha256"], "current-geometry")
        self.assertEqual(report["pressure_pair_anchor_current_marker_geometry_revision"], 12)
        self.assertEqual(report["pressure_pair_anchor_runtime_refresh_count"], 6)

    def test_runner_refresh_failure_leaves_old_anchor_retired(self) -> None:
        from benchmarks.official import solid_mpm_fsi_runner

        markers = _RecordingAnchorMarkers()
        with (
            patch.object(
                solid_mpm_fsi_runner,
                "_is_selected_traction_formulation_coupled_smoke",
                return_value=True,
            ),
            patch.object(
                solid_mpm_fsi_runner,
                "_traction_pressure_pair_runtime_provider_mode",
                return_value="runtime_anchored_cell_pair",
            ),
            patch.object(
                solid_mpm_fsi_runner,
                "_runtime_pressure_pair_anchor_map",
                side_effect=ValueError("synthetic stale geometry"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "stale geometry"):
                solid_mpm_fsi_runner._refresh_runtime_pressure_pair_anchor_markers(
                    markers,
                    object(),
                    object(),
                    refresh_count=1,
                )

        self.assertEqual(markers.events, ["reset"])

    def test_anchor_axes_are_explicitly_supported(self) -> None:
        for axis in (0, 1, 2):
            with self.subTest(axis=axis):
                position = [0.5, 0.5, 0.5]
                normal = [0.0, 0.0, 0.0]
                position[axis] = 0.75
                normal[axis] = 1.0

                pair_map = compute_runtime_anchored_cell_pair_map(
                    marker_positions_m=(tuple(position),),
                    marker_normals=(tuple(normal),),
                    marker_region_ids=("region",),
                    domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    grid_nodes=(4, 4, 4),
                    anchor_axis=axis,
                    inside_axis_position_m=0.5,
                    outside_axis_offset_cells=1,
                )

                inside_cell = pair_map.inside_cells[0]
                outside_cell = pair_map.outside_cells[0]
                self.assertEqual(inside_cell[axis], 2)
                self.assertEqual(outside_cell[axis], 3)
                self.assertNotEqual(inside_cell, outside_cell)

    def test_missing_pairs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one marker"):
            compute_runtime_anchored_cell_pair_map(
                marker_positions_m=(),
                marker_normals=(),
                marker_region_ids=(),
                domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                grid_nodes=(4, 4, 4),
                anchor_axis=2,
                inside_axis_position_m=0.5,
            )
        with self.assertRaisesRegex(ValueError, "must match"):
            compute_runtime_anchored_cell_pair_map(
                marker_positions_m=((0.5, 0.5, 0.5),),
                marker_normals=(),
                marker_region_ids=(101,),
                domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                grid_nodes=(4, 4, 4),
                anchor_axis=2,
                inside_axis_position_m=0.5,
            )
        with self.assertRaisesRegex(ValueError, "nonzero anchor-axis"):
            compute_runtime_anchored_cell_pair_map(
                marker_positions_m=((0.5, 0.5, 0.5),),
                marker_normals=((1.0, 0.0, 0.0),),
                marker_region_ids=(101,),
                domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                grid_nodes=(4, 4, 4),
                anchor_axis=2,
                inside_axis_position_m=0.5,
            )

    def test_invalid_provider_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside_axis_offset_cells"):
            compute_runtime_anchored_cell_pair_map(
                marker_positions_m=((0.5, 0.5, 0.5),),
                marker_normals=((0.0, 0.0, 1.0),),
                marker_region_ids=(101,),
                domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                grid_nodes=(4, 4, 4),
                anchor_axis=2,
                inside_axis_position_m=0.5,
                outside_axis_offset_cells=0,
            )
        with self.assertRaisesRegex(ValueError, "grid_nodes"):
            compute_runtime_anchored_cell_pair_map(
                marker_positions_m=((0.5, 0.5, 0.5),),
                marker_normals=((0.0, 0.0, 1.0),),
                marker_region_ids=(101,),
                domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                grid_nodes=(4, 0, 4),
                anchor_axis=2,
                inside_axis_position_m=0.5,
            )
        with self.assertRaisesRegex(ValueError, "positive finite cell spacing"):
            compute_runtime_anchored_cell_pair_map(
                marker_positions_m=((0.5, 0.5, 0.5),),
                marker_normals=((0.0, 0.0, 1.0),),
                marker_region_ids=(101,),
                domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
                grid_nodes=(4, 4, 4),
                anchor_axis=2,
                inside_axis_position_m=0.5,
            )

    def test_outside_direction_and_clamp_policy_are_deterministic(self) -> None:
        pair_map = compute_runtime_anchored_cell_pair_map(
            marker_positions_m=(
                (0.5, 0.5, 2.0),
                (0.5, 0.5, -1.0),
            ),
            marker_normals=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            marker_region_ids=("positive", "negative"),
            domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            grid_nodes=(4, 4, 4),
            anchor_axis=2,
            inside_axis_position_m=0.5,
            outside_axis_offset_cells=1,
        )

        self.assertEqual(pair_map.inside_cells, ((2, 2, 2), (2, 2, 2)))
        self.assertEqual(pair_map.outside_cells, ((2, 2, 3), (2, 2, 0)))
        for pair in pair_map.pairs:
            self.assertNotEqual(pair.inside_cell, pair.outside_cell)
            for cell in (pair.inside_cell, pair.outside_cell):
                self.assertTrue(all(0 <= component < 4 for component in cell))

    def test_pair_map_sha_is_order_sensitive(self) -> None:
        first = _runtime_pair_map()
        reversed_sha = pressure_sample_pair_map_sha256(tuple(reversed(first.pairs)))

        self.assertNotEqual(first.pair_map_sha256, reversed_sha)

    def test_contract_is_case_agnostic(self) -> None:
        import simulation_core.coupling.pressure_sample_pairs as pressure_sample_pairs

        source = inspect.getsource(pressure_sample_pairs)
        for term in ("ansys", "fluent", "vertical_flap", "vertical flap"):
            self.assertNotIn(term, source.lower())


def _runtime_pair_map() -> PressureSamplePairMap:
    return compute_runtime_anchored_cell_pair_map(
        marker_positions_m=(
            (0.0015, 0.0004166667, 0.0537968762),
            (0.0015, 0.0004166667, 0.0492031239),
        ),
        marker_normals=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        marker_region_ids=(101, 202),
        domain_bounds_m=((0.0, 0.0, 0.0), (0.003, 0.02, 0.1)),
        grid_nodes=(4, 32, 64),
        anchor_axis=2,
        inside_axis_position_m=0.0515,
        outside_axis_offset_cells=1,
    )


def _moving_interface_provider() -> RuntimeAnchoredCellPairProvider:
    return RuntimeAnchoredCellPairProvider(
        domain_bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        grid_nodes=(8, 8, 8),
        anchor_axis=2,
        inside_axis_position_m=0.5,
        outside_axis_offset_cells=1,
    )


def _moving_marker_geometry(*, z_m: float, revision: int) -> dict[str, object]:
    return {
        "marker_positions_m": ((0.5, 0.5, float(z_m)),),
        "marker_normals": ((0.0, 0.0, 1.0),),
        "marker_region_ids": (101,),
        "marker_geometry_revision": int(revision),
    }


class _RecordingAnchorMarkers:
    marker_count = 1

    def __init__(self) -> None:
        self.events: list[str] = []

    def reset_pressure_pair_anchor_cells(self) -> None:
        self.events.append("reset")

    def set_pressure_pair_anchor_cells(self, **kwargs) -> None:
        self.events.append("set")
        assert kwargs["inside_cells"] == ((1, 1, 1),)
        assert kwargs["outside_cells"] == ((1, 1, 3),)
        assert kwargs["source_marker_geometry_revision"] == 12
        assert kwargs["source_marker_geometry_sha256"] == "current-geometry"


class _RecordingPairMap:
    inside_cells = ((1, 1, 1),)
    outside_cells = ((1, 1, 3),)
    selected_count = 1
    pair_map_sha256 = "pair-map"
    marker_geometry_sha256 = "current-geometry"
    marker_geometry_revision = 12

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def require_current_marker_geometry(self, _markers) -> None:
        self._events.append("require-current")

    def as_diagnostics(self) -> dict[str, object]:
        return {
            "pair_map_sha256": self.pair_map_sha256,
            "marker_geometry_sha256": self.marker_geometry_sha256,
            "marker_geometry_revision": self.marker_geometry_revision,
        }


if __name__ == "__main__":
    unittest.main()
