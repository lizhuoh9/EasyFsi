from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refactored.validation.ansys_vertical_flap_fixed.projection_solver import (
    DEFAULT_SOLVER_CONFIG,
    DEFAULT_STABILIZED_SOLVER_CONFIG,
    _finite_or_raise,
)


class FiniteOrRaiseTests(unittest.TestCase):
    def test_finite_array_is_returned_as_a_copy(self) -> None:
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = _finite_or_raise("field", array)

        np.testing.assert_allclose(result, array)
        self.assertIsNot(result, array)

    def test_nan_and_inf_raise_by_default_with_count_and_breakdown(self) -> None:
        # Audit probe: `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`
        # used to silently zero a blown-up field every step, hiding CFL
        # violations / divergent pressure solves behind output that looked
        # perfectly finite. Strict mode must surface the blow-up instead.
        array = np.zeros((3, 4))
        array[1, 2] = float("nan")
        array[0, 0] = float("inf")
        array[2, 3] = float("-inf")

        with self.assertRaises(FloatingPointError) as ctx:
            _finite_or_raise("velocity_u", array)

        message = str(ctx.exception)
        self.assertIn("velocity_u", message)
        self.assertIn("3", message)  # 3 non-finite entries total, of 12
        self.assertIn("nan=1", message)
        self.assertIn("+inf=1", message)
        self.assertIn("-inf=1", message)

    def test_single_nan_entry_raises(self) -> None:
        array = np.array([1.0, float("nan"), 3.0])

        with self.assertRaises(FloatingPointError):
            _finite_or_raise("pressure", array)

    def test_single_inf_entry_raises(self) -> None:
        array = np.array([1.0, float("inf"), 3.0])

        with self.assertRaises(FloatingPointError):
            _finite_or_raise("pressure", array)

    def test_allow_non_finite_opt_in_preserves_old_zeroing_behavior(self) -> None:
        # The opt-in tolerance mode must reproduce the exact previous
        # np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0) behavior for
        # callers that have explicitly decided to accept it.
        array = np.array([1.0, float("nan"), float("inf"), float("-inf")])

        result = _finite_or_raise("field", array, allow_non_finite=True)

        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0])

    def test_all_finite_with_allow_non_finite_true_is_unaffected(self) -> None:
        array = np.array([1.0, 2.0, 3.0])

        result = _finite_or_raise("field", array, allow_non_finite=True)

        np.testing.assert_allclose(result, array)

    def test_default_solver_configs_are_strict_by_default(self) -> None:
        self.assertFalse(DEFAULT_SOLVER_CONFIG["allow_non_finite_fields"])
        self.assertFalse(DEFAULT_STABILIZED_SOLVER_CONFIG["allow_non_finite_fields"])


if __name__ == "__main__":
    unittest.main()
