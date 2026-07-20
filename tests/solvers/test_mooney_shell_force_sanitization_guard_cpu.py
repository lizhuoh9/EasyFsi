from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

# CRITICAL: a CUDA production run may be using the GPU concurrently in this
# environment. Never ti.init(arch=ti.cuda) here -- CPU backend only.
#
# TriMooneyShellMpmState.__init__ unconditionally calls
# simulation_core.diagnostics.runtime.init_taichi(runtime), which defaults to
# arch="cuda" and explicitly rejects arch="cpu" ("simulation_core is
# GPU-only"). init_taichi() has "first call wins" semantics gated by its own
# private module-level _INITIALIZED flag, so we take real Taichi CPU
# ownership ourselves first and then mark that flag pre-satisfied so the
# constructor's own init_taichi(None) call becomes a no-op instead of trying
# (and failing, or worse, re-initializing onto CUDA) to set up the GPU.
# Mirrors tests/solvers/test_neo_hookean_mpm_oob_guard_cpu.py.
ti.init(arch=ti.cpu, default_fp=ti.f32)

from simulation_core.diagnostics import runtime as sim_runtime

sim_runtime._INITIALIZED = True
sim_runtime._INITIALIZED_ARCH = "cpu"
sim_runtime._INITIALIZED_FP = "f32"

from simulation_core.geometry_tools import SurfaceMesh, UvSphereResolution
from simulation_core.solids.mooney_shell import (
    TriMooneyShellMpmState,
    UvMooneyShellMpmState,
)


# ---------------------------------------------------------------------
# S2-audit FINDING 3: TriMooneyShellMpmState used to silently drop any
# non-finite/oversized force contribution (_atomic_add_particle_force /
# _atomic_add_particle_external_force) and silently hard-clamp the
# Cauchy-Green constitutive invariants and per-face force gradients
# (_accumulate_mooney_face's c_cap clamp and _limit_vector_norm), with no
# visibility -- masking genuine coupling instabilities. The fix (see
# _raise_if_force_sanitization_detected / _check_force_sanitization in
# mooney_shell/core.py) makes both anomaly categories STRICT by default:
# step()/advance_region_loads()/advance_with_external_forces() now raise a
# RuntimeError naming the counts and first few affected indices, instead of
# silently returning a report as if nothing happened. The clamp VALUES are
# unchanged either way; allow_force_sanitization=True restores the exact
# pre-fix silent behavior (still with the counts populated on the report).
# ---------------------------------------------------------------------
def _single_triangle_state(
    *,
    bounds_padding_fraction: float = 1.0,
    face_region_id: np.ndarray | None = None,
    require_nonempty_region_counts: bool | None = None,
) -> TriMooneyShellMpmState:
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
        face_region_id=face_region_id,
        grid_nodes=(12, 12, 12),
        bounds_padding_fraction=bounds_padding_fraction,
        primary_region_id=1,
        secondary_region_id=2,
        require_nonempty_region_counts=require_nonempty_region_counts,
    )


def _stretch_triangle_past_constitutive_cap(
    state: TriMooneyShellMpmState, *, stretch: float = 1500.0
) -> None:
    """Deform the rest unit triangle enough to exceed the c_cap=1e6 clamp.

    rest edges have unit length and inv00=inv11=1, inv01=0 (axis-aligned
    right triangle), so f0=(stretch,0,0), f1=(0,stretch,0) and
    c00=c11=stretch**2. stretch=1500 gives c00=c11=2.25e6 > 1e6: finite
    (nowhere near float32 overflow), so this isolates the constitutive/
    force-cap clamp path from the non-finite/sanitized-force path.
    """
    positions = state.x.to_numpy()
    positions[1] = [stretch, 0.0, 0.0]
    positions[2] = [0.0, stretch, 0.0]
    state.x.from_numpy(positions.astype(np.float32))


def _small_uv_state() -> UvMooneyShellMpmState:
    return UvMooneyShellMpmState(
        UvSphereResolution(latitude_bands=4, longitude_segments=8),
        radius_m=1.0,
        thickness_m=0.05,
        density_kgm3=1.0,
        c1_pa=20.0,
        c2_pa=10.0,
        grid_nodes=(12, 12, 12),
        bounds_scale=2.0,
    )


def _collapse_uv_shell(state: UvMooneyShellMpmState) -> None:
    state.x.from_numpy(np.zeros((state.particle_count, 3), dtype=np.float32))


class TriMooneyShellSanitizedForceGuardTests(unittest.TestCase):
    """sanitized_force_count: non-finite/oversized forces dropped silently."""

    def test_step_strict_default_raises_on_infinite_pressure_force(self) -> None:
        # pressure_pa=inf makes _accumulate_mooney_face's pressure_force
        # infinite; _atomic_add_particle_external_force's own safety gate
        # (not _limit_vector_norm, which only wraps the internal/stress
        # force) is what catches this, in isolation from any constitutive
        # clamp.
        state = _single_triangle_state()

        with self.assertRaisesRegex(
            RuntimeError, r"3 non-finite/oversized force sanitization event"
        ) as raised:
            state.step(dt_s=0.0, pressure_pa=float("inf"), velocity_damping=1.0)
        self.assertIn("0 constitutive/force-cap clamp activation", str(raised.exception))
        self.assertIn("allow_force_sanitization=True", str(raised.exception))

    def test_step_legacy_flag_preserves_silent_zero_force_and_reports_count(
        self,
    ) -> None:
        state = _single_triangle_state()

        report = state.step(
            dt_s=0.0,
            pressure_pa=float("inf"),
            velocity_damping=1.0,
            allow_force_sanitization=True,
        )

        # Pre-fix behavior: the infinite pressure contribution is dropped
        # (external_force_n stays exactly zero), not committed or crashed.
        self.assertEqual(report.sanitized_force_count, 3)
        self.assertEqual(report.constitutive_clamp_count, 0)
        external_force = state.external_force_n.to_numpy()
        np.testing.assert_array_equal(external_force, np.zeros_like(external_force))

    def test_default_report_has_zero_sanitization_counts_for_ordinary_step(
        self,
    ) -> None:
        # Regression guard: a normal, physically reasonable step must not
        # false-positive the new guard (which would otherwise turn every
        # ordinary simulation into an immediate RuntimeError).
        state = _single_triangle_state()

        report = state.step(dt_s=1.0e-3, pressure_pa=2.0, velocity_damping=1.0)

        self.assertEqual(report.sanitized_force_count, 0)
        self.assertEqual(report.constitutive_clamp_count, 0)

    def test_advance_region_loads_strict_default_raises_on_infinite_area_load(
        self,
    ) -> None:
        state = _single_triangle_state(
            face_region_id=np.array([1], dtype=np.int32),
            require_nonempty_region_counts=False,
        )

        with self.assertRaisesRegex(
            RuntimeError, r"3 non-finite/oversized force sanitization event"
        ):
            state.advance_region_loads(
                dt_s=0.0,
                primary_region_id=1,
                secondary_region_id=2,
                primary_area_load_npm2=(float("inf"), 0.0, 0.0),
                primary_interface_reaction_n=(0.0, 0.0, 0.0),
                secondary_interface_reaction_n=(0.0, 0.0, 0.0),
            )

    def test_advance_region_loads_legacy_flag_preserves_silent_behavior(self) -> None:
        state = _single_triangle_state(
            face_region_id=np.array([1], dtype=np.int32),
            require_nonempty_region_counts=False,
        )

        report = state.advance_region_loads(
            dt_s=0.0,
            primary_region_id=1,
            secondary_region_id=2,
            primary_area_load_npm2=(float("inf"), 0.0, 0.0),
            primary_interface_reaction_n=(0.0, 0.0, 0.0),
            secondary_interface_reaction_n=(0.0, 0.0, 0.0),
            allow_force_sanitization=True,
        )

        self.assertEqual(report.sanitized_force_count, 3)


class TriMooneyShellConstitutiveClampGuardTests(unittest.TestCase):
    """constitutive_clamp_count: Cauchy-Green invariant / force-cap clamps."""

    def test_step_strict_default_raises_on_extreme_but_finite_stretch(self) -> None:
        # Padding must be large enough that the stretched (but still
        # finite/in-bounds) vertices don't also trip the unrelated
        # out-of-bounds guard, which runs -- and would raise -- first.
        state = _single_triangle_state(bounds_padding_fraction=3000.0)
        _stretch_triangle_past_constitutive_cap(state)

        with self.assertRaisesRegex(
            RuntimeError, r"4 constitutive/force-cap clamp activation"
        ) as raised:
            state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)
        self.assertIn("0 non-finite/oversized force sanitization", str(raised.exception))

    def test_step_legacy_flag_preserves_force_cap_clamp_value_and_reports_count(
        self,
    ) -> None:
        thickness_m = 0.05
        c1_pa = 20.0
        c2_pa = 10.0
        state = _single_triangle_state(bounds_padding_fraction=3000.0)
        _stretch_triangle_past_constitutive_cap(state)

        report = state.step(
            dt_s=0.0,
            pressure_pa=0.0,
            velocity_damping=1.0,
            allow_force_sanitization=True,
        )

        self.assertEqual(report.sanitized_force_count, 0)
        self.assertEqual(report.constitutive_clamp_count, 4)
        # The clamp VALUE itself must be unchanged by this fix: vertex 0's
        # internal force is still exactly force_cap_n in magnitude (the
        # clamp -- not the fix -- decides the number; the fix only makes
        # its activation visible/rejectable).
        rest_area_m2 = 0.5
        force_cap_n = (c1_pa + c2_pa) * thickness_m * (rest_area_m2**0.5) * 100.0
        internal_force = state.internal_force_n.to_numpy()
        self.assertAlmostEqual(
            float(np.linalg.norm(internal_force[0])), force_cap_n, places=2
        )

    def test_advance_with_external_forces_strict_default_raises_on_extreme_stretch(
        self,
    ) -> None:
        state = _single_triangle_state(
            bounds_padding_fraction=3000.0,
            face_region_id=np.array([1], dtype=np.int32),
            require_nonempty_region_counts=False,
        )
        _stretch_triangle_past_constitutive_cap(state)

        with self.assertRaisesRegex(
            RuntimeError, r"4 constitutive/force-cap clamp activation"
        ):
            state.advance_with_external_forces(
                dt_s=0.0, primary_region_id=1, secondary_region_id=2
            )

    def test_advance_with_external_forces_legacy_flag_reports_clamp_count(
        self,
    ) -> None:
        state = _single_triangle_state(
            bounds_padding_fraction=3000.0,
            face_region_id=np.array([1], dtype=np.int32),
            require_nonempty_region_counts=False,
        )
        _stretch_triangle_past_constitutive_cap(state)

        report = state.advance_with_external_forces(
            dt_s=0.0,
            primary_region_id=1,
            secondary_region_id=2,
            allow_force_sanitization=True,
        )

        self.assertEqual(report.constitutive_clamp_count, 4)


class UvMooneyShellSanitizationGuardTests(unittest.TestCase):
    def test_step_strict_default_raises_on_infinite_pressure_force(self) -> None:
        state = _small_uv_state()

        with self.assertRaisesRegex(
            RuntimeError, r"non-finite/oversized force sanitization"
        ):
            state.step(dt_s=0.0, pressure_pa=float("inf"), velocity_damping=1.0)

    def test_step_legacy_flag_reports_sanitized_force_count(self) -> None:
        state = _small_uv_state()

        report = state.step(
            dt_s=0.0,
            pressure_pa=float("inf"),
            velocity_damping=1.0,
            allow_force_sanitization=True,
        )

        self.assertEqual(report.sanitized_force_count, 144)
        self.assertEqual(report.constitutive_clamp_count, 0)
        recorded = state.diag_sanitized_force_first_indices.to_numpy()
        self.assertTrue(
            np.all((recorded >= 0) & (recorded < state.particle_count))
        )

    def test_step_strict_default_raises_on_degenerate_constitutive_state(self) -> None:
        state = _small_uv_state()
        _collapse_uv_shell(state)

        with self.assertRaisesRegex(
            RuntimeError, r"constitutive/force-cap clamp activation"
        ):
            state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

    def test_step_legacy_flag_reports_constitutive_clamp_count(self) -> None:
        state = _small_uv_state()
        _collapse_uv_shell(state)

        report = state.step(
            dt_s=0.0,
            pressure_pa=0.0,
            velocity_damping=1.0,
            allow_force_sanitization=True,
        )

        self.assertEqual(report.sanitized_force_count, 0)
        self.assertEqual(report.constitutive_clamp_count, 48)
        recorded = state.diag_constitutive_clamp_first_indices.to_numpy()
        self.assertTrue(
            np.all((recorded >= 0) & (recorded < state.particle_count))
        )

    def test_anomaly_counters_are_cleared_between_steps(self) -> None:
        state = _small_uv_state()

        pressure_report = state.step(
            dt_s=0.0,
            pressure_pa=float("inf"),
            velocity_damping=1.0,
            allow_force_sanitization=True,
        )
        self.assertEqual(pressure_report.sanitized_force_count, 144)
        self.assertEqual(pressure_report.constitutive_clamp_count, 0)

        _collapse_uv_shell(state)
        collapsed_report = state.step(
            dt_s=0.0,
            pressure_pa=0.0,
            velocity_damping=1.0,
            allow_force_sanitization=True,
        )
        self.assertEqual(collapsed_report.sanitized_force_count, 0)
        self.assertEqual(collapsed_report.constitutive_clamp_count, 48)

        state.x.from_numpy(state.rest_x.to_numpy())
        recovered_report = state.step(
            dt_s=0.0,
            pressure_pa=0.0,
            velocity_damping=1.0,
            allow_force_sanitization=True,
        )
        self.assertEqual(recovered_report.sanitized_force_count, 0)
        self.assertEqual(recovered_report.constitutive_clamp_count, 0)


if __name__ == "__main__":
    unittest.main()
