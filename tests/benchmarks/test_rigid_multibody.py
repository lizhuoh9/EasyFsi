from __future__ import annotations

import math
import unittest


class RigidMultibodyTests(unittest.TestCase):
    def test_hinge_rotation_schedule_interpolates_then_holds(self) -> None:
        from benchmarks.official.rigid_multibody import HingeRotationSchedule

        schedule = HingeRotationSchedule(
            start_time_s=0.0,
            end_time_s=0.25,
            start_angle_rad=0.0,
            end_angle_rad=1.0,
        )

        self.assertEqual(schedule.angle_at(-0.1), 0.0)
        self.assertAlmostEqual(schedule.angle_at(0.125), 0.5)
        self.assertEqual(schedule.angle_at(0.5), 1.0)

    def test_hinge_rotation_schedule_supports_quarter_sine_profile(self) -> None:
        from benchmarks.official.rigid_multibody import HingeRotationSchedule

        schedule = HingeRotationSchedule(
            start_time_s=0.0,
            end_time_s=0.25,
            start_angle_rad=0.0,
            end_angle_rad=1.0,
            profile="quarter-sine",
        )

        self.assertAlmostEqual(schedule.angle_at(0.05), math.sin(0.1 * math.pi))
        self.assertEqual(schedule.angle_at(0.5), 1.0)

    def test_equal_opposite_fin_pair_motion_is_case_agnostic(self) -> None:
        from benchmarks.official.rigid_multibody import EqualOppositeHingePair

        pair = EqualOppositeHingePair(
            positive_name="left-fin",
            negative_name="right-fin",
            schedule_end_angle_rad=0.8,
            active_until_s=0.25,
        )

        self.assertEqual(pair.angles_at(0.125), {"left-fin": 0.4, "right-fin": -0.4})


if __name__ == "__main__":
    unittest.main()
