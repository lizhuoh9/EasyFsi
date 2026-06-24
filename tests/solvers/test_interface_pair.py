from __future__ import annotations

import unittest


class InterfacePairMapTests(unittest.TestCase):
    def test_pair_map_transfers_full_vector_displacements(self) -> None:
        from simulation_core.interface_pair import InterfacePairMap, PairMapEntry

        pair_map = InterfacePairMap(
            target_count=2,
            entries=(
                PairMapEntry(target_index=0, source_index=0, weight=0.25),
                PairMapEntry(target_index=0, source_index=1, weight=0.75),
                PairMapEntry(target_index=1, source_index=1, weight=1.0),
            ),
        )

        mapped = pair_map.map_vectors(((2.0, 0.0, 0.0), (0.0, 4.0, 0.0)))

        self.assertEqual(mapped, ((0.5, 3.0, 0.0), (0.0, 4.0, 0.0)))

    def test_pair_map_transfers_normal_component_only(self) -> None:
        from simulation_core.interface_pair import InterfacePairMap, PairMapEntry

        pair_map = InterfacePairMap(
            target_count=1,
            entries=(PairMapEntry(target_index=0, source_index=0, weight=1.0),),
        )

        mapped = pair_map.map_normal_vectors(
            source_vectors=((3.0, 4.0, 0.0),),
            target_normals=((1.0, 0.0, 0.0),),
        )

        self.assertEqual(mapped, ((3.0, 0.0, 0.0),))

    def test_pair_map_transposes_fluid_forces_as_equal_opposite_solid_reactions(self) -> None:
        from simulation_core.fsi_coupling import action_reaction_balance
        from simulation_core.interface_pair import InterfacePairMap, PairMapEntry

        pair_map = InterfacePairMap(
            target_count=2,
            entries=(
                PairMapEntry(target_index=0, source_index=0, weight=0.5),
                PairMapEntry(target_index=0, source_index=1, weight=0.5),
                PairMapEntry(target_index=1, source_index=1, weight=1.0),
            ),
        )

        fluid_forces = ((2.0, 0.0, 0.0), (0.0, 6.0, 0.0))
        solid_reactions = pair_map.transpose_forces(
            target_forces=fluid_forces,
            source_count=2,
            action_reaction_sign=-1.0,
        )

        total_fluid_force = tuple(sum(force[axis] for force in fluid_forces) for axis in range(3))
        total_solid_force = tuple(sum(force[axis] for force in solid_reactions) for axis in range(3))
        self.assertLessEqual(
            action_reaction_balance(total_fluid_force, total_solid_force).relative_error,
            1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
