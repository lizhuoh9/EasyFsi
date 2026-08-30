"""Production-MPM strain preservation and zero-small-strain load contracts."""

import os

import numpy as np
import pytest
import taichi as ti

from simulation_core.solids import neo_hookean_mpm


@pytest.fixture(scope="module", autouse=True)
def precision_runtime():
    requested = os.environ.get("MPM_PRECISION_TEST_ARCH", "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("MPM_PRECISION_TEST_ARCH must be cpu or cuda")
    arch = ti.cuda if requested == "cuda" else ti.cpu
    ti.reset()
    ti.init(arch=arch, default_fp=ti.f32, offline_cache=False,
            cfg_optimization=False, opt_level=1, advanced_optimization=True,
            fast_math=True, cpu_max_num_threads=1)
    if ti.cfg.arch != arch:
        raise RuntimeError(f"requested {requested} but Taichi used {ti.cfg.arch}")
    yield
    ti.reset()


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(neo_hookean_mpm, "init_taichi", lambda runtime: None)
    solid = neo_hookean_mpm.NeoHookeanMpmState(
        particle_capacity=8, bounds_min_m=(-0.02, -0.02, -0.02),
        bounds_max_m=(0.02, 0.02, 0.02), grid_nodes=(10, 10, 10),
    )
    solid.initialize_box(
        particle_counts=(2, 2, 2), box_min_m=(-0.004, -0.004, -0.004),
        box_max_m=(0.004, 0.004, 0.004), density_kgm3=1600.0,
    )
    return solid


def step(solid, dt, *, mu=0.0, lame=0.0, read_report=False):
    return solid.step(
        dt_s=dt, mu_pa=mu, lambda_pa=lame,
        primary_region_id=0, secondary_region_id=-1,
        velocity_damping=1.0, constitutive_model="plane_stress_linear_elastic",
        read_report=read_report,
    )


@pytest.mark.parametrize("rate", [0.01, -0.01])
def test_affine_patch_preserves_sub_ulp_normal_strain_over_full_macro_step(state, rate):
    macro_dt, substeps = 0.0005, 1284
    dt = float(np.float32(macro_dt / substeps))
    coordinates = state.x.to_numpy()
    velocity = np.zeros_like(coordinates)
    velocity[:, 1] = rate * coordinates[:, 1]
    gradient = np.zeros((state.particle_count, 3, 3), dtype=np.float32)
    gradient[:, 1, 1] = rate
    state.v.from_numpy(velocity)
    state.C.from_numpy(gradient)
    state.begin_out_of_bounds_guard_batch()
    try:
        for _ in range(substeps):
            step(state, dt)
        report = state.end_out_of_bounds_guard_batch()
    except BaseException:
        state.abort_out_of_bounds_guard_batch()
        raise
    assert report.grid_out_of_bounds_particle_count == 0
    assert report.deformation_clamp_count == 0
    # Stress-free APIC preserves the affine velocity field to transfer roundoff.
    # Its convective change over this short interval is only O(rate*macro_dt).
    np.testing.assert_allclose(state.C.to_numpy()[:, 1, 1], rate, rtol=0.01, atol=0)
    strain = state.F.to_numpy().astype(np.float64)[:, 1, 1] - 1.0
    expected = rate * dt * substeps
    np.testing.assert_allclose(strain, expected, rtol=0.01, atol=1.0e-9)


def test_zero_infinitesimal_strain_does_not_create_svd_reconstruction_load(state):
    # I + skew is the zero-strain infinitesimal-rotation patch for this LINEAR
    # material law.  It is deliberately not a finite rigid-rotation test.
    deformation = np.tile(np.eye(3, dtype=np.float32), (state.particle_count, 1, 1))
    deformation[:, 1, 2] = np.float32(0.0123)
    deformation[:, 2, 1] = -np.float32(0.0123)
    state.F.from_numpy(deformation)
    young, poisson = 1.0e6, 0.47
    report = step(
        state, 1.0e-5, mu=young / (2.0 * (1.0 + poisson)),
        lame=young * poisson / (1.0 - poisson * poisson), read_report=True,
    )
    assert report.grid_out_of_bounds_particle_count == 0
    assert report.deformation_clamp_count == 0
    np.testing.assert_array_equal(state.v.to_numpy(), np.zeros((state.particle_count, 3)))
    np.testing.assert_array_equal(state.C.to_numpy(), np.zeros((state.particle_count, 3, 3)))
