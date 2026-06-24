from __future__ import annotations

import unittest

from simulation_core.pressure_interface import (
    far_pressure_side_normal_sign_from_direction,
)


class PressureInterfaceBoundaryMappingTests(unittest.TestCase):
    def test_maps_requested_pressure_direction_to_far_side_of_interface(self) -> None:
        self.assertEqual(
            far_pressure_side_normal_sign_from_direction(
                pressure_direction=(0.0, 0.0, -1.0),
                interface_normal=(0.0, 0.0, -1.0),
            ),
            -1.0,
        )
        self.assertEqual(
            far_pressure_side_normal_sign_from_direction(
                pressure_direction=(0.0, 0.0, -1.0),
                interface_normal=(0.0, 0.0, 1.0),
            ),
            1.0,
        )

    def test_rejects_tangential_pressure_direction_for_scalar_closure(self) -> None:
        with self.assertRaisesRegex(ValueError, "normal component"):
            far_pressure_side_normal_sign_from_direction(
                pressure_direction=(1.0, 0.0, 0.0),
                interface_normal=(0.0, 0.0, 1.0),
            )


if __name__ == "__main__":
    unittest.main()
