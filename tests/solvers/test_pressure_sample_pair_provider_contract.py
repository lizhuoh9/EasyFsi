from __future__ import annotations

import inspect
import unittest

from simulation_core.pressure_sample_pairs import (
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
        import simulation_core.pressure_sample_pairs as pressure_sample_pairs

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


if __name__ == "__main__":
    unittest.main()
