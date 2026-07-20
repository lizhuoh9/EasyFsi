from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

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
from simulation_core.solids.neo_hookean_mpm import (
    NeoHookeanMpmState,
    _raise_if_out_of_bounds_exceeds_tolerance,
)


# ---------------------------------------------------------------------
# FINDING 2 (audit S2): ordinary read_report=False calls must remain
# immediately fail-closed. Multi-substep production callers may avoid a host
# synchronization per substep only through the explicit guard-batch API, which
# retains the maximum OOB count on device and performs one fail-closed packed
# report read before coupling is allowed to continue.
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


class NeoHookeanOutOfBoundsGuardBatchTests(unittest.TestCase):
    @staticmethod
    def _state() -> NeoHookeanMpmState:
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
        return state

    @staticmethod
    def _step_without_report(state: NeoHookeanMpmState) -> None:
        result = state.step(
            dt_s=1.0e-5,
            mu_pa=0.0,
            lambda_pa=0.0,
            primary_region_id=0,
            secondary_region_id=-1,
            read_report=False,
        )
        assert result is None

    def test_batch_defers_guard_host_reads_until_one_final_report(self) -> None:
        state = self._state()

        state.begin_out_of_bounds_guard_batch()
        for _ in range(3):
            self._step_without_report(state)
            self.assertEqual(state.last_out_of_bounds_guard_host_reads, 0)

        report = state.end_out_of_bounds_guard_batch()

        self.assertEqual(report.grid_out_of_bounds_particle_count, 0)
        self.assertEqual(state.last_report_host_reads, 1)
        self.assertEqual(state.last_out_of_bounds_guard_host_reads, 1)

    def test_batch_sticky_guard_catches_transient_out_of_bounds_particle(self) -> None:
        state = self._state()
        in_bounds_positions = state.x.to_numpy()
        out_of_bounds_positions = in_bounds_positions.copy()
        out_of_bounds_positions[0, 0] = float(
            state.bounds_max[0] + 2.0 * state.dx[0]
        )

        state.begin_out_of_bounds_guard_batch()
        state.x.from_numpy(out_of_bounds_positions.astype(np.float32))
        self._step_without_report(state)
        self.assertEqual(state.last_out_of_bounds_guard_host_reads, 0)

        # Re-enter before the final substep. A current-state-only check would
        # now miss the escape; the device-side batch maximum must retain it.
        state.x.from_numpy(in_bounds_positions.astype(np.float32))
        self._step_without_report(state)

        with self.assertRaisesRegex(RuntimeError, "outside the background grid"):
            state.end_out_of_bounds_guard_batch()
        self.assertEqual(state.last_out_of_bounds_guard_host_reads, 1)

    def test_batch_catches_particle_leaving_on_the_final_substep(self) -> None:
        state = self._state()
        positions = state.x.to_numpy()
        positions[:, 0] = 0.011
        state.x.from_numpy(positions.astype(np.float32))
        velocities = np.zeros((state.particle_capacity, 3), dtype=np.float32)
        velocities[:, 0] = 100.0
        state.v.from_numpy(velocities)

        state.begin_out_of_bounds_guard_batch()
        result = state.step(
            dt_s=5.0e-5,
            mu_pa=0.0,
            lambda_pa=0.0,
            primary_region_id=0,
            secondary_region_id=-1,
            read_report=False,
        )
        self.assertIsNone(result)

        # The particles start inside the valid quadratic-B-spline stencil but
        # cross its upper x limit during this final integration. The batch-end
        # guard must inspect the post-integration positions on device; waiting
        # for a hypothetical next substep would let coupling consume bad state.
        with self.assertRaisesRegex(RuntimeError, "outside the background grid"):
            state.end_out_of_bounds_guard_batch()

    def test_same_substep_reentry_cannot_erase_entry_oob_count(self) -> None:
        source = inspect.getsource(NeoHookeanMpmState._step_kernel)
        post_recount_reset = source.index(
            "self.report_grid_out_of_bounds_particle_count[None] = 0"
        )
        sticky_update = (
            "self.out_of_bounds_guard_batch_max_particle_count[None] = ti.max("
        )
        sticky_update_indices: list[int] = []
        search_start = 0
        while True:
            index = source.find(sticky_update, search_start)
            if index < 0:
                break
            sticky_update_indices.append(index)
            search_start = index + len(sticky_update)

        # The entry-position count must be retained before it is cleared for
        # the post-integration recount; the second max then retains particles
        # that leave during integration. This keeps both sides fail-closed if
        # an integration path can re-enter within the same substep.
        self.assertEqual(len(sticky_update_indices), 2)
        self.assertLess(sticky_update_indices[0], post_recount_reset)
        self.assertLess(post_recount_reset, sticky_update_indices[1])

    def test_post_oob_recount_is_fused_into_existing_g2p_loop(self) -> None:
        source = inspect.getsource(NeoHookeanMpmState._step_kernel)
        post_recount_reset = source.index(
            "self.report_grid_out_of_bounds_particle_count[None] = 0"
        )
        report_coord = source.index(
            "report_coord = self._particle_grid_coordinate(p)"
        )
        radial_report = source.index("radial_center_count = ti.max(")
        report_coord_block = source[report_coord:radial_report]

        self.assertLess(post_recount_reset, report_coord)
        self.assertIn(
            "ti.atomic_add(\n"
            "                        self.report_grid_out_of_bounds_particle_count[None]",
            report_coord_block,
        )
        # P2G + G2P/report + radial diagnostics. A fourth particle loop would
        # reintroduce the avoidable O(Np) post-recount pass.
        self.assertEqual(source.count("for p in range(self.particle_capacity):"), 3)


class NeoHookeanOutOfBoundsGuardHostContractTests(unittest.TestCase):
    @staticmethod
    def _host_only_state() -> SimpleNamespace:
        return SimpleNamespace(
            particle_count=1,
            _out_of_bounds_guard_batch_active=False,
            _out_of_bounds_guard_batch_step_count=0,
            last_report_host_reads=7,
            last_out_of_bounds_guard_host_reads=9,
            _reset_out_of_bounds_guard_batch_kernel=lambda: None,
        )

    def test_full_out_of_bounds_is_unconditional_even_with_high_tolerance(self) -> None:
        for tolerance in (8, 80):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "8 of 8 MPM particles are outside the background grid",
                ):
                    _raise_if_out_of_bounds_exceeds_tolerance(8, 8, tolerance)

    def test_abort_is_idempotent_and_allows_a_fresh_begin(self) -> None:
        state = self._host_only_state()
        NeoHookeanMpmState.begin_out_of_bounds_guard_batch(state)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            NeoHookeanMpmState.begin_out_of_bounds_guard_batch(state)

        NeoHookeanMpmState.abort_out_of_bounds_guard_batch(state)
        NeoHookeanMpmState.abort_out_of_bounds_guard_batch(state)
        self.assertFalse(state._out_of_bounds_guard_batch_active)
        self.assertEqual(state._out_of_bounds_guard_batch_step_count, 0)

        NeoHookeanMpmState.begin_out_of_bounds_guard_batch(state)
        self.assertTrue(state._out_of_bounds_guard_batch_active)
        self.assertIn(
            "restore_state()",
            inspect.getdoc(NeoHookeanMpmState.abort_out_of_bounds_guard_batch) or "",
        )

    def test_end_rejects_inactive_or_empty_batch(self) -> None:
        state = self._host_only_state()
        with self.assertRaisesRegex(RuntimeError, "no out-of-bounds guard batch"):
            NeoHookeanMpmState.end_out_of_bounds_guard_batch(state)
        NeoHookeanMpmState.begin_out_of_bounds_guard_batch(state)
        with self.assertRaisesRegex(RuntimeError, "contains no completed steps"):
            NeoHookeanMpmState.end_out_of_bounds_guard_batch(state)


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
