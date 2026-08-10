import math
import unittest
from unittest import mock

from simulation_core.fluids import grid as grid_module
from simulation_core.fluids.grid import GradedGridSpec, RefinementRegion


class GradedGridPerformanceContractTests(unittest.TestCase):
    def _assert_width_contract(
        self,
        widths: tuple[float, ...],
        *,
        distance: float,
        growth_ratio: float,
        left_width: float,
        right_width: float | None = None,
    ) -> None:
        values = (left_width, *widths)
        if right_width is not None:
            values = (*values, right_width)
        self.assertTrue(
            math.isclose(sum(widths), distance, rel_tol=1.0e-11, abs_tol=1.0e-12)
        )
        self.assertTrue(all(width > 0.0 for width in widths))
        self.assertTrue(
            all(
                max(left / right, right / left)
                <= growth_ratio * (1.0 + 1.0e-10)
                for left, right in zip(values, values[1:])
            )
        )

    def test_bridge_search_evaluates_only_logarithmically_many_cell_counts(self) -> None:
        original = grid_module._bridge_width_bounds
        with mock.patch.object(
            grid_module,
            "_bridge_width_bounds",
            wraps=original,
        ) as width_bounds:
            widths = grid_module._graded_bridge_widths(
                distance=24.0,
                left_width=0.01,
                right_width=0.01,
                farfield_spacing=0.01,
                max_growth_ratio=1.2,
            )

        self.assertLessEqual(width_bounds.call_count, 32)
        self.assertGreater(len(widths), 2_000)
        self._assert_width_contract(
            widths,
            distance=24.0,
            growth_ratio=1.2,
            left_width=0.01,
            right_width=0.01,
        )

    def test_bridge_width_bounds_match_cellwise_definition(self) -> None:
        cases = (
            (1, 0.01, 0.04, 0.05, 1.2),
            (7, 0.04, 0.01, 0.05, 1.2),
            (18, 0.003, 0.02, 0.01, 1.5),
        )
        for cells, left, right, farfield, growth in cases:
            lower = tuple(
                max(left / growth**step, right / growth ** (cells - step + 1))
                for step in range(1, cells + 1)
            )
            upper = tuple(
                min(farfield, left * growth**step, right * growth ** (cells - step + 1))
                for step in range(1, cells + 1)
            )

            with self.subTest(cells=cells, left=left, right=right):
                actual_lower, actual_upper = grid_module._bridge_width_bounds(
                    cells,
                    left_width=left,
                    right_width=right,
                    farfield_spacing=farfield,
                    max_growth_ratio=growth,
                )
                self.assertTrue(
                    all(
                        math.isclose(actual, expected, rel_tol=1.0e-14)
                        for actual, expected in zip(actual_lower, lower, strict=True)
                    )
                )
                self.assertTrue(
                    all(
                        math.isclose(actual, expected, rel_tol=1.0e-14)
                        for actual, expected in zip(actual_upper, upper, strict=True)
                    )
                )

    def test_side_search_evaluates_only_logarithmically_many_cell_counts(self) -> None:
        original = grid_module._side_width_bounds
        with mock.patch.object(
            grid_module,
            "_side_width_bounds",
            wraps=original,
        ) as width_bounds:
            widths = grid_module._graded_side_widths(
                distance=24.0,
                inner_width=0.01,
                farfield_spacing=0.01,
                max_growth_ratio=1.2,
            )

        self.assertLessEqual(width_bounds.call_count, 32)
        self.assertGreater(len(widths), 2_000)
        self._assert_width_contract(
            widths,
            distance=24.0,
            growth_ratio=1.2,
            left_width=0.01,
        )


class GradedGridFiniteValidationTests(unittest.TestCase):
    def test_spec_rejects_nonfinite_bounds_spacing_and_growth(self) -> None:
        valid = dict(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            farfield_spacing_m=0.1,
            max_growth_ratio=1.2,
        )
        invalid_overrides = (
            {"bounds_min_m": (math.nan, 0.0, 0.0)},
            {"bounds_max_m": (math.inf, 1.0, 1.0)},
            {"farfield_spacing_m": math.nan},
            {"farfield_spacing_m": math.inf},
            {"max_growth_ratio": math.nan},
            {"max_growth_ratio": math.inf},
            {"max_cells": math.nan},
            {"max_cells": math.inf},
        )

        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "finite"):
                    GradedGridSpec(**(valid | override))

    def test_refinement_region_rejects_nonfinite_values(self) -> None:
        invalid_arguments = (
            dict(
                bounds_min_m=(math.nan, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                target_spacing_m=0.1,
            ),
            dict(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, math.inf),
                target_spacing_m=0.1,
            ),
            dict(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                target_spacing_m=math.nan,
            ),
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "finite"):
                    RefinementRegion(**arguments)


if __name__ == "__main__":
    unittest.main()
