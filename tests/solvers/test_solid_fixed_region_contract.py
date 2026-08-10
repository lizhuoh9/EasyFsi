from __future__ import annotations

import inspect
import unittest

from simulation_core.solids import neo_hookean_mpm
from simulation_core.solids.mooney_shell import core as mooney_shell_core


class SolidFixedRegionContractTests(unittest.TestCase):
    def _assert_shared_contract(self, validator) -> None:
        with self.assertRaisesRegex(ValueError, "distinct.*primary_region_id"):
            validator(
                fixed_region_id=7,
                primary_region_id=7,
                secondary_region_id=8,
                active_region_ids=(7, 8),
            )
        with self.assertRaisesRegex(ValueError, "distinct.*secondary_region_id"):
            validator(
                fixed_region_id=8,
                primary_region_id=7,
                secondary_region_id=8,
                active_region_ids=(7, 8),
            )
        with self.assertRaisesRegex(ValueError, "matched no faces"):
            validator(
                fixed_region_id=5,
                primary_region_id=7,
                secondary_region_id=8,
                active_region_ids=(7, 8),
            )
        self.assertEqual(
            validator(
                fixed_region_id=5,
                primary_region_id=7,
                secondary_region_id=8,
                active_region_ids=(5, 7, 8),
            ),
            5,
        )

    def test_neo_fixed_region_contract_rejects_aliases_and_missing_faces(self) -> None:
        validator = getattr(neo_hookean_mpm, "_validate_fixed_region_contract")
        self._assert_shared_contract(validator)

    def test_mooney_fixed_region_contract_matches_neo(self) -> None:
        validator = getattr(mooney_shell_core, "_validate_fixed_region_contract")
        self._assert_shared_contract(validator)

    def test_neo_initializer_applies_contract_before_particle_initialization(self) -> None:
        source = inspect.getsource(
            neo_hookean_mpm.NeoHookeanMpmState.initialize_layered_tri_surface
        )
        self.assertLess(
            source.index("_validate_fixed_region_contract("),
            source.index("particle_count ="),
        )

    def test_mooney_constructor_applies_contract_before_taichi_initialization(self) -> None:
        source = inspect.getsource(mooney_shell_core.TriMooneyShellMpmState.__init__)
        self.assertLess(
            source.index("_validate_fixed_region_contract("),
            source.index("init_taichi(runtime)"),
        )


if __name__ == "__main__":
    unittest.main()
