from __future__ import annotations

import unittest


class MovingBoundaryTests(unittest.TestCase):
    def test_full_mesh_displacement_uses_all_mapped_components(self) -> None:
        from simulation_core.interface_pair import InterfacePairMap, PairMapEntry
        from simulation_core.moving_boundary import MovingBoundaryCondition

        boundary = MovingBoundaryCondition(
            name="all-components",
            pair_map=InterfacePairMap(
                target_count=1,
                entries=(PairMapEntry(target_index=0, source_index=0, weight=1.0),),
            ),
            transfer_mode="full",
        )

        self.assertEqual(
            boundary.mesh_displacements(((1.0, 2.0, 3.0),)),
            ((1.0, 2.0, 3.0),),
        )

    def test_normal_mesh_displacement_keeps_tangent_free(self) -> None:
        from simulation_core.interface_pair import InterfacePairMap, PairMapEntry
        from simulation_core.moving_boundary import MovingBoundaryCondition

        boundary = MovingBoundaryCondition(
            name="normal-only",
            pair_map=InterfacePairMap(
                target_count=1,
                entries=(PairMapEntry(target_index=0, source_index=0, weight=1.0),),
            ),
            transfer_mode="normal",
        )

        self.assertEqual(
            boundary.mesh_displacements(
                ((1.0, 2.0, 0.0),),
                target_normals=((0.0, 1.0, 0.0),),
            ),
            ((0.0, 2.0, 0.0),),
        )


if __name__ == "__main__":
    unittest.main()
