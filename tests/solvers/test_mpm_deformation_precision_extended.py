"""Focused f64-deformation regression contracts for the Neo MPM candidate.

This module reuses the CPU/CUDA fixture without initializing Taichi at import.
Every case uses the existing eight-particle box so the kernel compilation
footprint stays bounded.
"""

from __future__ import annotations

import math

import numpy as np

from simulation_core.solids import neo_hookean_mpm
from tests.solvers.test_mpm_deformation_precision import precision_runtime, state


_PARTICLE_COUNT = 8
_I64 = np.eye(3, dtype=np.float64)


def _identity_f64() -> np.ndarray:
    return np.broadcast_to(_I64, (_PARTICLE_COUNT, 3, 3)).copy()


def _step(
    solid,
    *,
    dt_s: float,
    constitutive_model: str,
    mu_pa: float = 0.0,
    lambda_pa: float = 0.0,
):
    return solid.step(
        dt_s=dt_s,
        mu_pa=mu_pa,
        lambda_pa=lambda_pa,
        primary_region_id=0,
        secondary_region_id=-1,
        velocity_damping=1.0,
        constitutive_model=constitutive_model,
        read_report=True,
    )


def _set_f64_deformation(solid, deformation: np.ndarray) -> None:
    assert deformation.dtype == np.dtype(np.float64)
    assert deformation.shape == (_PARTICLE_COUNT, 3, 3)
    solid.F.from_numpy(deformation)


def _set_affine_velocity_and_c(solid, affine: np.ndarray) -> None:
    positions = solid.x.to_numpy()
    velocity = np.einsum("pij,pj->pi", affine, positions).astype(np.float32)
    solid.v.from_numpy(velocity)
    solid.C.from_numpy(affine)


def test_legal_f64_raw_f_low_bits_survive_one_stress_free_step(
    precision_runtime, state
):
    deformation = _identity_f64()
    deformation[:, 0, 0] = 1.0 + math.ldexp(1.0, -35)
    deformation[:, 0, 1] = math.ldexp(1.0, -35)
    _set_f64_deformation(state, deformation)
    state.v.from_numpy(np.zeros_like(state.v.to_numpy()))
    state.C.from_numpy(np.zeros_like(state.C.to_numpy()))

    report = _step(
        state,
        dt_s=1.0e-5,
        constitutive_model="plane_stress_linear_elastic",
    )

    assert state.F.to_numpy().dtype == np.dtype(np.float64)
    assert report.deformation_clamp_count == 0
    np.testing.assert_array_equal(state.F.to_numpy(), deformation)


def test_c64_recurrence_matches_post_g2p_affine_for_normal_and_shear_modes(
    precision_runtime, state
):
    dt_s = float(np.float32(5.0e-5))
    rate = np.float32(1.0e-2)
    modes: list[tuple[int, int, float]] = []
    for axis in range(3):
        modes.extend((axis, axis, sign * float(rate)) for sign in (-1.0, 1.0))
    for row in range(3):
        for column in range(3):
            if row != column:
                modes.extend((row, column, sign * float(rate)) for sign in (-1.0, 1.0))

    for row, column, signed_rate in modes:
        deformation_before = _identity_f64()
        affine = np.zeros((_PARTICLE_COUNT, 3, 3), dtype=np.float32)
        affine[:, row, column] = np.float32(signed_rate)
        _set_f64_deformation(state, deformation_before)
        _set_affine_velocity_and_c(state, affine)

        report = _step(
            state,
            dt_s=dt_s,
            constitutive_model="plane_stress_linear_elastic",
        )
        affine_after = state.C.to_numpy().astype(np.float64)
        expected = np.einsum(
            "pij,pjk->pik",
            _I64[None, :, :] + dt_s * affine_after,
            deformation_before,
        )

        assert report.deformation_clamp_count == 0
        assert state.F.to_numpy().dtype == np.dtype(np.float64)
        np.testing.assert_allclose(state.F.to_numpy(), expected, rtol=0.0, atol=2.0e-12)


def test_finite_rotation_has_no_neo_or_svk_virtual_stress(
    precision_runtime, state
):
    angle = 0.31
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    for model in ("3d_neo_hookean", "saint_venant_kirchhoff"):
        _set_f64_deformation(
            state, np.broadcast_to(rotation, (_PARTICLE_COUNT, 3, 3)).copy()
        )
        state.v.from_numpy(np.zeros_like(state.v.to_numpy()))
        state.C.from_numpy(np.zeros_like(state.C.to_numpy()))

        report = _step(
            state,
            dt_s=1.0e-5,
            constitutive_model=model,
            mu_pa=1.0e6,
            lambda_pa=2.0e6,
        )

        assert report.deformation_clamp_count == 0
        np.testing.assert_allclose(
            state.grid_velocity_mps.to_numpy(), 0.0, rtol=0.0, atol=1.0e-9
        )


def test_linear_i_plus_skew_has_no_linearized_strain_stress(
    precision_runtime, state
):
    skew = np.array(
        [[0.0, 0.013, 0.0], [-0.013, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    _set_f64_deformation(
        state, np.broadcast_to(_I64 + skew, (_PARTICLE_COUNT, 3, 3)).copy()
    )
    state.v.from_numpy(np.zeros_like(state.v.to_numpy()))
    state.C.from_numpy(np.zeros_like(state.C.to_numpy()))

    report = _step(
        state,
        dt_s=1.0e-5,
        constitutive_model="linear_elastic",
        mu_pa=1.0e6,
        lambda_pa=2.0e6,
    )

    assert report.deformation_clamp_count == 0
    np.testing.assert_allclose(
        state.grid_velocity_mps.to_numpy(), 0.0, rtol=0.0, atol=1.0e-9
    )


def test_minimum_and_maximum_deformation_overflow_is_never_tolerance_waived(
    precision_runtime, state
):
    deformation = _identity_f64()
    lower = neo_hookean_mpm.MIN_DEFORMATION_SINGULAR_VALUE * (1.0 - 5.0e-7)
    upper = neo_hookean_mpm.MAX_DEFORMATION_SINGULAR_VALUE * (1.0 + 5.0e-7)
    deformation[:4, 0, 0] = lower
    deformation[4:, 0, 0] = upper
    _set_f64_deformation(state, deformation)

    report = _step(
        state,
        dt_s=0.0,
        constitutive_model="plane_stress_linear_elastic",
    )
    singular_values = np.linalg.svd(state.F.to_numpy(), compute_uv=False)

    assert report.deformation_clamp_count == _PARTICLE_COUNT
    assert np.all(
        singular_values
        >= np.nextafter(neo_hookean_mpm.MIN_DEFORMATION_SINGULAR_VALUE, 0.0)
    )
    assert np.all(
        singular_values
        <= np.nextafter(neo_hookean_mpm.MAX_DEFORMATION_SINGULAR_VALUE, np.inf)
    )


def test_f64_nominal_minimum_is_not_lowered_by_default_f32_constant_cast(
    precision_runtime, state
):
    deformation = _identity_f64()
    deformation[:, 0, 0] = (
        neo_hookean_mpm.MIN_DEFORMATION_SINGULAR_VALUE - 1.0e-10
    )
    _set_f64_deformation(state, deformation)

    report = _step(
        state,
        dt_s=0.0,
        constitutive_model="plane_stress_linear_elastic",
    )

    assert report.deformation_clamp_count == _PARTICLE_COUNT
    # Projection includes two f64 matrix products.  This permits only their
    # rounding, not the old f32 lower-bound error (over 128 million f64 ULPs).
    np.testing.assert_array_max_ulp(
        state.F.to_numpy()[:, 0, 0],
        np.full(
            _PARTICLE_COUNT,
            neo_hookean_mpm.MIN_DEFORMATION_SINGULAR_VALUE,
            dtype=np.float64,
        ),
        maxulp=8,
    )


def test_exact_nominal_deformation_limits_are_legal_and_not_reconstructed(
    precision_runtime, state
):
    deformation = _identity_f64()
    deformation[:4, 0, 0] = neo_hookean_mpm.MIN_DEFORMATION_SINGULAR_VALUE
    deformation[4:, 0, 0] = neo_hookean_mpm.MAX_DEFORMATION_SINGULAR_VALUE
    _set_f64_deformation(state, deformation)

    report = _step(
        state,
        dt_s=0.0,
        constitutive_model="plane_stress_linear_elastic",
    )

    assert report.deformation_clamp_count == 0
    np.testing.assert_array_equal(state.F.to_numpy(), deformation)


def test_f64_recurrence_left_multiplies_a_noncommuting_deformation(
    precision_runtime, state
):
    dt_s = float(np.float32(5.0e-5))
    deformation_before = np.broadcast_to(
        np.array(
            [[1.10, 0.07, 0.00], [0.00, 0.90, 0.02], [0.00, 0.00, 1.05]],
            dtype=np.float64,
        ),
        (_PARTICLE_COUNT, 3, 3),
    ).copy()
    affine = np.zeros((_PARTICLE_COUNT, 3, 3), dtype=np.float32)
    affine[:, 0, 1] = np.float32(1.0e-2)
    affine[:, 1, 2] = np.float32(-2.0e-2)
    _set_f64_deformation(state, deformation_before)
    _set_affine_velocity_and_c(state, affine)

    report = _step(
        state,
        dt_s=dt_s,
        constitutive_model="plane_stress_linear_elastic",
    )
    affine_after = state.C.to_numpy().astype(np.float64)
    expected = np.einsum(
        "pij,pjk->pik",
        _I64[None, :, :] + dt_s * affine_after,
        deformation_before,
    )

    assert report.deformation_clamp_count == 0
    np.testing.assert_allclose(state.F.to_numpy(), expected, rtol=0.0, atol=2.0e-12)


def test_negative_j_is_projected_and_reported(precision_runtime, state):
    deformation = _identity_f64()
    deformation[:, 2, 2] = -1.0
    _set_f64_deformation(state, deformation)

    report = _step(
        state,
        dt_s=0.0,
        constitutive_model="plane_stress_linear_elastic",
    )

    assert report.deformation_clamp_count == _PARTICLE_COUNT
    assert np.all(np.linalg.det(state.F.to_numpy()) > 0.0)
