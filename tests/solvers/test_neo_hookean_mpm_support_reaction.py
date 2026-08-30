"""CPU contracts for Neo-Hookean MPM support and damping impulses."""

import numpy as np
import pytest
import taichi as ti

from simulation_core.solids import neo_hookean_mpm



@pytest.fixture(scope="module", autouse=True)
def _cpu_taichi_owner():
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f32,
        offline_cache=False,
        cpu_max_num_threads=1,
        opt_level=1,
        advanced_optimization=False,
    )
    yield
    ti.reset()


@pytest.fixture
def state_factory(monkeypatch):
    monkeypatch.setattr(neo_hookean_mpm, "init_taichi", lambda runtime: None)
    return _state



def _state(particle_count=2):
    state = neo_hookean_mpm.NeoHookeanMpmState(
        particle_capacity=particle_count,
        bounds_min_m=(-0.02, -0.02, -0.02),
        bounds_max_m=(0.02, 0.02, 0.02),
        grid_nodes=(10, 10, 10),
    )
    state.initialize_box(
        particle_counts=(particle_count, 1, 1),
        box_min_m=(-0.004, -0.001, -0.001),
        box_max_m=(0.004, 0.001, 0.001),
        density_kgm3=1000.0,
    )
    return state


def _step(
    state,
    *,
    dt_s,
    damping=1.0,
    fixed_node_lock_policy="any_fixed_particle",
    read_report=True,
):
    return state.step(
        dt_s=dt_s,
        mu_pa=0.0,
        lambda_pa=0.0,
        primary_region_id=0,
        secondary_region_id=-1,
        velocity_damping=damping,
        fixed_node_lock_policy=fixed_node_lock_policy,
        read_report=read_report,
    )


def _set_fixed(state, indices):
    fixed = np.zeros(state.particle_capacity, dtype=np.int32)
    fixed[np.asarray(indices, dtype=np.int32)] = 1
    state.fixed_particle.from_numpy(fixed)


def _set_particle_force(state, particle_index, force_n):
    forces = np.zeros((state.particle_capacity, 3), dtype=np.float32)
    forces[particle_index] = np.asarray(force_n, dtype=np.float32)
    state.external_force_n.from_numpy(forces)


def _fixed_g2p_grid_impulses(state, particle_index):
    position = state.x.to_numpy()[particle_index]
    coordinate = (position - np.asarray(state.bounds_min)) / np.asarray(state.dx)
    base = np.floor(coordinate - 0.5).astype(np.int32)
    fractional = coordinate - base
    axis_weights = [
        np.array(
            [
                0.5 * (1.5 - value) ** 2,
                0.75 - (value - 1.0) ** 2,
                0.5 * (value - 0.5) ** 2,
            ],
            dtype=np.float64,
        )
        for value in fractional
    ]
    mass = float(state.mass_kg.to_numpy()[particle_index])
    grid_velocity = state.grid_velocity_mps.to_numpy()
    impulse = np.zeros(3, dtype=np.float64)
    angular_impulse = np.zeros(3, dtype=np.float64)
    for ox in range(3):
        for oy in range(3):
            for oz in range(3):
                node = base + np.array([ox, oy, oz], dtype=np.int32)
                weight = axis_weights[0][ox] * axis_weights[1][oy] * axis_weights[2][oz]
                node_position = np.asarray(state.bounds_min) + node * np.asarray(state.dx)
                transferred_momentum = mass * weight * grid_velocity[tuple(node)]
                impulse -= transferred_momentum
                angular_impulse -= np.cross(node_position, transferred_momentum)
    return impulse, angular_impulse


def test_fixed_particle_omitted_force_becomes_negative_support_force_impulse_and_origin_torque(state_factory):
    state = state_factory()
    positions = state.x.to_numpy()
    positions[0] = np.array([0.006, 0.0, 0.0], dtype=np.float32)
    state.x.from_numpy(positions)
    _set_fixed(state, [0])
    force_n = np.array([0.0, 3.0, 0.0])
    dt_s = 0.02
    _set_particle_force(state, 0, force_n)

    report = _step(state, dt_s=dt_s)

    expected_impulse = -dt_s * force_n
    expected_angular_impulse = np.cross(positions[0], expected_impulse)
    np.testing.assert_allclose(report.direct_fixed_external_force_n, force_n, rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(report.support_reaction_impulse_n_s, expected_impulse, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(
        report.support_reaction_angular_impulse_n_m_s,
        expected_angular_impulse,
        rtol=0.0,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(report.damping_impulse_n_s, (0.0, 0.0, 0.0), atol=2.0e-8)


def test_grid_clamp_and_damping_impulses_are_separate_and_include_origin_torque(state_factory):
    state = state_factory()
    positions = state.x.to_numpy()
    positions[:2] = np.array([0.006, 0.0, 0.0], dtype=np.float32)
    state.x.from_numpy(positions)
    velocities = np.zeros((state.particle_capacity, 3), dtype=np.float32)
    velocities[1, 1] = 1.0
    state.v.from_numpy(velocities)
    _set_fixed(state, [0])
    particle_mass = float(state.mass_kg.to_numpy()[0])

    report = _step(state, dt_s=1.0e-4, damping=0.5)

    expected_damping = np.array([0.0, -0.5 * particle_mass, 0.0])
    expected_clamp = np.array([0.0, -0.5 * particle_mass, 0.0])
    expected_torque = np.cross(positions[0], expected_clamp)
    np.testing.assert_allclose(report.direct_fixed_external_force_n, (0.0, 0.0, 0.0), atol=2.0e-8)
    np.testing.assert_allclose(report.damping_impulse_n_s, expected_damping, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(report.support_reaction_impulse_n_s, expected_clamp, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(report.damping_angular_impulse_n_m_s, expected_torque, rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(
        report.support_reaction_angular_impulse_n_m_s, expected_torque, rtol=0.0, atol=2.0e-9
    )


@pytest.mark.parametrize(
    ("fixed_node_lock_policy", "expected_free_velocity", "expected_support_scale"),
    [
        ("any_fixed_particle", 0.0, 1.0),
        ("pure_fixed_mass", 0.5, 0.5),
    ],
)
def test_fixed_particle_g2p_discard_is_reported_as_support_reaction_for_both_lock_policies(
    state_factory,
    fixed_node_lock_policy,
    expected_free_velocity,
    expected_support_scale,
):
    state = state_factory()
    positions = state.x.to_numpy()
    positions[:2] = np.array([0.006, 0.0, 0.0], dtype=np.float32)
    state.x.from_numpy(positions)
    velocities = np.zeros((state.particle_capacity, 3), dtype=np.float32)
    velocities[1, 1] = 1.0
    state.v.from_numpy(velocities)
    _set_fixed(state, [0])
    particle_mass = float(state.mass_kg.to_numpy()[0])

    report = _step(
        state,
        dt_s=1.0e-4,
        damping=1.0,
        fixed_node_lock_policy=fixed_node_lock_policy,
    )

    expected_support = np.array([0.0, -expected_support_scale * particle_mass, 0.0])
    expected_torque = np.cross(positions[0], expected_support)
    np.testing.assert_allclose(state.v.to_numpy()[0], (0.0, 0.0, 0.0), atol=2.0e-8)
    np.testing.assert_allclose(
        state.v.to_numpy()[1],
        (0.0, expected_free_velocity, 0.0),
        atol=2.0e-8,
    )
    np.testing.assert_allclose(report.direct_fixed_external_force_n, (0.0, 0.0, 0.0), atol=2.0e-8)
    np.testing.assert_allclose(report.damping_impulse_n_s, (0.0, 0.0, 0.0), atol=2.0e-8)
    np.testing.assert_allclose(report.support_reaction_impulse_n_s, expected_support, atol=2.0e-8)
    np.testing.assert_allclose(
        report.support_reaction_angular_impulse_n_m_s,
        expected_torque,
        atol=2.0e-9,
    )


def test_fixed_particle_g2p_affine_angular_discard_uses_grid_moment_not_particle_center(
    state_factory,
):
    state = state_factory()
    positions = state.x.to_numpy()
    positions[:2] = np.array([0.006, 0.0, 0.0], dtype=np.float32)
    state.x.from_numpy(positions)
    affine = np.zeros((state.particle_capacity, 3, 3), dtype=np.float32)
    angular_rate = 250.0
    affine[1, 0, 1] = -angular_rate
    affine[1, 1, 0] = angular_rate
    state.C.from_numpy(affine)
    _set_fixed(state, [0])

    report = _step(
        state,
        dt_s=1.0e-4,
        damping=1.0,
        fixed_node_lock_policy="pure_fixed_mass",
    )

    expected_impulse, expected_angular_impulse = _fixed_g2p_grid_impulses(state, 0)
    particle_center_angular_impulse = np.cross(positions[0], expected_impulse)
    np.testing.assert_allclose(report.direct_fixed_external_force_n, (0.0, 0.0, 0.0), atol=2.0e-8)
    np.testing.assert_allclose(report.support_reaction_impulse_n_s, expected_impulse, atol=2.0e-8)
    np.testing.assert_allclose(
        report.support_reaction_angular_impulse_n_m_s,
        expected_angular_impulse,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(state.C.to_numpy()[0], np.zeros((3, 3)), atol=2.0e-8)
    assert np.linalg.norm(expected_angular_impulse) > 2.0e-9
    assert np.linalg.norm(expected_angular_impulse - particle_center_angular_impulse) > 2.0e-9


def test_fixed_particle_g2p_discard_batch_restore_retries_without_duplicate_impulse(
    state_factory,
):
    state = state_factory()
    positions = state.x.to_numpy()
    positions[:2] = np.array([0.006, 0.0, 0.0], dtype=np.float32)
    state.x.from_numpy(positions)
    velocities = np.zeros((state.particle_capacity, 3), dtype=np.float32)
    velocities[1, 1] = 1.0
    state.v.from_numpy(velocities)
    _set_fixed(state, [0])
    particle_mass = float(state.mass_kg.to_numpy()[0])

    state.begin_out_of_bounds_guard_batch()
    state.save_state()
    assert _step(
        state,
        dt_s=1.0e-4,
        damping=1.0,
        fixed_node_lock_policy="pure_fixed_mass",
        read_report=False,
    ) is None
    state.restore_state()
    assert _step(
        state,
        dt_s=1.0e-4,
        damping=1.0,
        fixed_node_lock_policy="pure_fixed_mass",
        read_report=False,
    ) is None
    report = state.end_out_of_bounds_guard_batch()

    expected_support = np.array([0.0, -0.5 * particle_mass, 0.0])
    np.testing.assert_allclose(report.direct_fixed_external_force_n, (0.0, 0.0, 0.0), atol=2.0e-8)
    np.testing.assert_allclose(report.support_reaction_impulse_n_s, expected_support, atol=2.0e-8)
    np.testing.assert_allclose(
        report.support_reaction_angular_impulse_n_m_s,
        np.cross(positions[0], expected_support),
        atol=2.0e-9,
    )


def test_guard_batch_accumulates_every_substep_and_independent_step_is_not_stale(state_factory):
    state = state_factory()
    _set_fixed(state, [0, 1])
    force_n = np.array([2.0, 0.0, 0.0])
    dt_s = 0.01
    state.set_uniform_external_force(tuple(force_n))

    state.begin_out_of_bounds_guard_batch()
    for _ in range(3):
        assert _step(state, dt_s=dt_s, read_report=False) is None
    batch_report = state.end_out_of_bounds_guard_batch()
    np.testing.assert_allclose(
        batch_report.support_reaction_impulse_n_s,
        -3.0 * dt_s * 2.0 * force_n,
        atol=2.0e-7,
    )

    independent = state_factory()
    _set_fixed(independent, [0, 1])
    independent.set_uniform_external_force(tuple(force_n))
    one_step_report = _step(independent, dt_s=dt_s)
    np.testing.assert_allclose(
        one_step_report.support_reaction_impulse_n_s,
        -dt_s * 2.0 * force_n,
        atol=2.0e-7,
    )


def test_restore_retries_from_saved_batch_accounting_without_duplicate_or_missing_impulse(state_factory):
    state = state_factory()
    _set_fixed(state, [0, 1])
    force_n = np.array([0.0, 1.5, 0.0])
    dt_s = 0.01
    state.set_uniform_external_force(tuple(force_n))
    state.begin_out_of_bounds_guard_batch()
    assert _step(state, dt_s=dt_s, read_report=False) is None
    state.save_state()

    assert _step(state, dt_s=dt_s, read_report=False) is None
    state.restore_state()
    state.set_uniform_external_force(tuple(force_n))
    assert _step(state, dt_s=dt_s, read_report=False) is None
    report = state.end_out_of_bounds_guard_batch()

    np.testing.assert_allclose(
        report.support_reaction_impulse_n_s,
        -2.0 * dt_s * 2.0 * force_n,
        atol=2.0e-7,
    )
    assert report.direct_fixed_external_force_n == pytest.approx(2.0 * force_n, abs=2.0e-6)
