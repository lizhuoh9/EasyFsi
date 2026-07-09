from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

# CRITICAL: a CUDA production run may be using the GPU concurrently in this
# environment. Never ti.init(arch=ti.cuda) here -- CPU backend only.
#
# NeoHookeanMpmState/TriMooneyShellMpmState.__init__ unconditionally call
# simulation_core.diagnostics.runtime.init_taichi(runtime), which defaults to
# arch="cuda" and explicitly rejects arch="cpu" ("simulation_core is
# GPU-only"). init_taichi() has "first call wins" semantics gated by its own
# private module-level _INITIALIZED flag, so we take real Taichi CPU
# ownership ourselves first and then mark that flag pre-satisfied so the
# constructors' own init_taichi(None) call becomes a no-op instead of trying
# (and failing, or worse, re-initializing onto CUDA) to set up the GPU.
ti.init(arch=ti.cpu, default_fp=ti.f32)

from simulation_core.diagnostics import runtime as sim_runtime

sim_runtime._INITIALIZED = True
sim_runtime._INITIALIZED_ARCH = "cpu"
sim_runtime._INITIALIZED_FP = "f32"

from simulation_core.geometry_tools import SurfaceMesh
from simulation_core.solids.mooney_shell import TriMooneyShellMpmState
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmState


# ---------------------------------------------------------------------
# FINDING 2 (audit S2): the out-of-bounds guard used to run only when
# read_report=True, because it lived inside report()'s host readback of the
# full snapshot. Callers that pass read_report=False to avoid that large
# host readback (e.g. Turek-Hron, for 99 of every 100 substeps) silently
# carried a partially-escaped solid forward for up to 100 substeps before
# the guard ever got a chance to fire. The fix reads back ONLY the single
# already-computed i32 out-of-bounds counter (a negligible 4-byte device
# readback) and enforces the guard unconditionally, every substep.
# ---------------------------------------------------------------------
class NeoHookeanOutOfBoundsGuardFiresWithoutReportTests(unittest.TestCase):
    def test_step_with_read_report_false_still_raises_on_out_of_bounds_particle(
        self,
    ) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.004, -0.004, -0.004),
            box_max_m=(0.004, 0.004, 0.004),
            density_kgm3=1000.0,
        )
        positions = state.x.to_numpy()
        positions[0, 0] = float(state.bounds_max[0] + 2.0 * state.dx[0])
        state.x.from_numpy(positions.astype(np.float32))

        with self.assertRaisesRegex(RuntimeError, "outside the background grid"):
            state.step(
                dt_s=1.0e-5,
                mu_pa=0.0,
                lambda_pa=0.0,
                primary_region_id=0,
                secondary_region_id=-1,
                read_report=False,
            )

    def test_step_with_read_report_false_does_not_raise_when_all_particles_in_bounds(
        self,
    ) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.004, -0.004, -0.004),
            box_max_m=(0.004, 0.004, 0.004),
            density_kgm3=1000.0,
        )

        result = state.step(
            dt_s=1.0e-5,
            mu_pa=0.0,
            lambda_pa=0.0,
            primary_region_id=0,
            secondary_region_id=-1,
            read_report=False,
        )

        self.assertIsNone(result)
        self.assertEqual(state.last_report_host_reads, 0)


class TriMooneyShellOutOfBoundsGuardFiresWithoutReportTests(unittest.TestCase):
    """Mirrors the NeoHookeanMpmState fix for TriMooneyShellMpmState's three
    read_report-gated stepping entry points (step, advance_region_loads,
    advance_with_external_forces)."""

    @staticmethod
    def _single_triangle_state(out_of_bounds_particle_tolerance: int = 0) -> TriMooneyShellMpmState:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        return TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=0.25,
            primary_region_id=1,
            secondary_region_id=2,
            out_of_bounds_particle_tolerance=out_of_bounds_particle_tolerance,
        )

    def test_step_with_read_report_false_still_raises_on_out_of_bounds_particle(
        self,
    ) -> None:
        state = self._single_triangle_state()
        positions = state.x.to_numpy()
        positions[0, 0] = float(state.bounds_max[0] + state.dx[0])
        state.x.from_numpy(positions.astype(np.float32))

        with self.assertRaisesRegex(RuntimeError, "outside the background grid"):
            state.step(
                dt_s=0.0,
                pressure_pa=0.0,
                velocity_damping=1.0,
                read_report=False,
            )

    def test_step_with_read_report_false_does_not_raise_when_all_particles_in_bounds(
        self,
    ) -> None:
        state = self._single_triangle_state()

        result = state.step(
            dt_s=0.0,
            pressure_pa=0.0,
            velocity_damping=1.0,
            read_report=False,
        )

        self.assertIsNone(result)
        self.assertEqual(state.last_report_host_reads, 0)


if __name__ == "__main__":
    unittest.main()
