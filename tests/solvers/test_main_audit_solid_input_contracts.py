from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.geometry_tools import SurfaceMesh
from simulation_core.solids.mooney_shell import TriMooneyShellMpmState
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmState


class MainAuditSolidInputContractTests(unittest.TestCase):
    @staticmethod
    def _neo_state() -> NeoHookeanMpmState:
        return NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-1.0, -1.0, -1.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

    def test_neo_constructor_rejects_nonfinite_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "bounds_min_m"):
            NeoHookeanMpmState(
                particle_capacity=1,
                bounds_min_m=(math.nan, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(4, 4, 4),
                runtime=TaichiRuntimeConfig(arch="cuda"),
            )

    def test_neo_initializers_reject_invalid_density_geometry_and_thickness(self) -> None:
        state = self._neo_state()
        for name, overrides in (
            ("density_kgm3", {"density_kgm3": math.nan}),
            ("box bounds", {"box_min_m": (0.5, 0.0, 0.0)}),
        ):
            arguments = {
                "particle_counts": (1, 1, 1),
                "box_min_m": (-0.5, -0.5, -0.5),
                "box_max_m": (0.5, 0.5, 0.5),
                "density_kgm3": 1.0,
                **overrides,
            }
            with self.subTest(name=name), self.assertRaises(ValueError):
                state.initialize_box(**arguments)

        with self.assertRaisesRegex(ValueError, "primary_thickness_m"):
            state.initialize_layered_tri_surface(
                SimpleNamespace(face_count=1),
                layer_count=1,
                primary_region_id=1,
                secondary_region_id=2,
                density_kgm3=1.0,
                primary_thickness_m=-0.1,
                secondary_thickness_m=0.1,
            )

    def test_neo_step_rejects_negative_modulus_before_device_work(self) -> None:
        state = self._neo_state()
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.1, -0.1, -0.1),
            box_max_m=(0.1, 0.1, 0.1),
            density_kgm3=1.0,
        )

        with self.assertRaisesRegex(ValueError, "mu_pa"):
            state.step(
                dt_s=1.0e-3,
                mu_pa=-1.0,
                lambda_pa=0.0,
                primary_region_id=0,
                secondary_region_id=1,
                read_report=False,
            )

    def test_mooney_rejects_nonfinite_density_and_negative_thickness_override(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        common = {
            "mesh": mesh,
            "thickness_m": 0.1,
            "density_kgm3": 1.0,
            "c1_pa": 1.0,
            "c2_pa": 0.0,
            "primary_region_id": 1,
            "secondary_region_id": 2,
        }

        with self.assertRaisesRegex(ValueError, "density_kgm3"):
            TriMooneyShellMpmState(**{**common, "density_kgm3": math.nan})
        with self.assertRaisesRegex(ValueError, "primary_thickness_m"):
            TriMooneyShellMpmState(**{**common, "primary_thickness_m": -0.1})


if __name__ == "__main__":
    unittest.main()
