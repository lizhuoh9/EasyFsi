"""Actual material kernels; MATERIAL_SURFACE_TEST_ARCH=cuda selects strict CUDA."""

import itertools
import os

import numpy as np
import pytest
import taichi as ti

from simulation_core.coupling.hibm_mpm import core
from simulation_core.coupling.hibm_mpm.interface_state import (
    capture_marker_interface_state,
    marker_layout_identity,
    marker_trial_state,
    restore_marker_interface_state,
)


@pytest.fixture(scope="module", params=range(3))
def material_surface(request):
    requested_arch = os.environ.get("MATERIAL_SURFACE_TEST_ARCH", "cpu")
    if requested_arch not in {"cpu", "cuda"}:
        raise ValueError("MATERIAL_SURFACE_TEST_ARCH must be cpu or cuda")
    architecture = ti.cuda if requested_arch == "cuda" else ti.cpu
    ti.init(arch=architecture, default_fp=ti.f32, cpu_max_num_threads=1,
            offline_cache=False, enable_fallback=False)
    actual_arch = ti.lang.impl.current_cfg().arch
    assert actual_arch == architecture
    print(f"MATERIAL_TEST_RUNTIME requested={requested_arch} actual={actual_arch} fallback=False")
    initializer = core.init_taichi
    core.init_taichi = lambda runtime=None: None
    try:
        markers = core.HibmMpmSurfaceMarkers(marker_capacity=10, projection_triangle_capacity=8)
    finally:
        core.init_taichi = initializer
    axis = request.param
    permutation = np.roll(np.arange(3), axis)
    particle_reference = np.asarray(list(itertools.product(
        [0.5], [0.5, 0.625, 0.75], [0.375, 0.625]
    )), dtype=np.float32)[:, permutation].copy()
    reference = np.asarray([
        [0.5, y, z] for z in (0.75, 0.25) for y in (0.5, 0.625, 0.75)
    ], dtype=np.float32)[:, permutation].copy()
    normals = np.asarray([[0, 0, 1]] * 3 + [[0, 0, -1]] * 3)[:, permutation]
    markers.load_markers(
        positions_m=reference, velocities_mps=np.zeros_like(reference),
        normals=normals, areas_m2=[0.125] * 6, region_ids=[101] * 3 + [202] * 3,
    )
    markers.configure_open_ribbon_tip_cap(
        primary_previous_marker_index=1, primary_tip_marker_index=2,
        secondary_previous_marker_index=4, secondary_tip_marker_index=5,
        cap_region_id=303, cap_area_m2=0.25, inactive_axis=axis,
    )
    markers.set_projection_segments(((0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (5, 7), (8, 9)))
    position = ti.Vector.field(3, ti.f32, shape=6)
    velocity = ti.Vector.field(3, ti.f32, shape=6)
    force = ti.Vector.field(3, ti.f32, shape=6)
    position.from_numpy(particle_reference)
    velocity.fill(0)
    markers.configure_material_surface_binding(
        particle_reference_positions_m=particle_reference,
        particle_mass_kg=np.ones(6), inactive_axis=axis,
    )
    yield markers, position, velocity, force, particle_reference, reference, axis
    ti.reset()


def _reset(surface):
    markers, position, velocity, force, particle_reference, reference, axis = surface
    position.from_numpy(particle_reference)
    velocity.fill(0)
    force.fill(0)
    markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)
    return markers, position, velocity, force, particle_reference, reference, axis


@pytest.mark.parametrize("component,sign", tuple(itertools.product(range(3), (-1.0, 1.0))))
def test_material_motion_uses_current_particle_state_not_endpoint_euler(material_surface, component, sign):
    markers, position, velocity, _, particle_reference, reference, _ = _reset(material_surface)
    # Two exact accepted substeps move forward then back. The endpoint velocity
    # remains nonzero, but accepted material displacement is exactly zero.
    delta = np.eye(3, dtype=np.float32)[component] * np.float32(sign / 64)
    position.from_numpy(particle_reference + delta)
    velocity.from_numpy(np.broadcast_to(delta * 64, particle_reference.shape).copy())
    markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)
    np.testing.assert_array_equal(markers.x_gamma_m.to_numpy()[:6], reference + delta)
    position.from_numpy(particle_reference)
    velocity.from_numpy(np.broadcast_to(-delta * 64, particle_reference.shape).copy())
    report = markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)
    np.testing.assert_array_equal(markers.x_gamma_m.to_numpy()[:6], reference)
    assert report.invalid_marker_count == report.geometry_invalid_marker_count == 0


def test_material_state_is_independent_of_macro_partition(material_surface):
    markers, position, velocity, _, particle_reference, reference, axis = _reset(material_surface)
    delta = np.eye(3, dtype=np.float32)[(axis + 1) % 3] / 64
    position.from_numpy(particle_reference + delta)
    velocity.from_numpy(np.broadcast_to(-delta * 64, particle_reference.shape).copy())
    markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)
    one_macro = markers.x_gamma_m.to_numpy()[:6].copy()
    _reset(material_surface)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        position.from_numpy(particle_reference + fraction * delta)
        markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)
    np.testing.assert_array_equal(markers.x_gamma_m.to_numpy()[:6], one_macro)
    np.testing.assert_array_equal(one_macro, reference + delta)


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_geometry_normals_follow_oriented_material_edges(material_surface, sign):
    markers, position, velocity, _, particle_reference, _, axis = _reset(material_surface)
    first, second = [component for component in range(3) if component != axis]
    deformation = np.eye(3)
    deformation[second, first] = sign * 0.5
    position.from_numpy((particle_reference @ deformation.T).astype(np.float32))
    markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)
    points = markers.x_gamma_m.to_numpy()[:6].astype(float)
    normals = markers.n_gamma.to_numpy()[:6].astype(float)
    for start in (0, 3):
        tangent = points[start + 2] - points[start]
        assert abs(np.dot(tangent, normals[start + 1])) < 2e-7
        for offset in (0, 1):
            edge = points[start + offset + 1] - points[start + offset]
            assert abs(np.dot(edge, normals[start + offset])) < 2e-7
    np.testing.assert_allclose(normals[:3], -normals[3:6], atol=2e-7, rtol=0)


@pytest.mark.parametrize("cap_enabled", (False, True))
def test_actual_rounded_adjoint_load_conserves_force_torque_and_material_power(material_surface, cap_enabled):
    markers, position, velocity, force, particle_reference, _, axis = _reset(material_surface)
    velocity.from_numpy((particle_reference * 0.25).astype(np.float32))
    markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)
    traction = np.arange(18, dtype=np.float64).reshape(6, 3) / 16 - 0.5
    markers.set_marker_tractions_pa(traction)
    if cap_enabled:
        markers.set_tip_cap_gauge_pressure_tractions_pa((0.5, -0.25))
    markers.compute_marker_forces()
    report = markers.scatter_marker_forces_to_mpm_particles(
        force, position, particle_count=6, support_radius_m=0.25,
        particle_velocity_mps=velocity,
    )
    indices = list(range(6)) + ([8, 9] if cap_enabled else [])
    marker_force = markers.F_gamma_n.to_numpy()[indices].astype(float)
    marker_position = markers.x_gamma_m.to_numpy()[indices].astype(float)
    marker_velocity = markers.v_gamma_mps.to_numpy()[indices].astype(float)
    applied = force.to_numpy().astype(float)
    np.testing.assert_allclose(applied.sum(0), marker_force.sum(0), atol=1e-7, rtol=0)
    np.testing.assert_allclose(np.cross(particle_reference, applied).sum(0),
                               np.cross(marker_position, marker_force).sum(0), atol=1e-7, rtol=0)
    assert abs(np.sum(velocity.to_numpy() * applied) - np.sum(marker_velocity * marker_force)) < 1e-7
    assert report.material_transfer_verified is True
    assert report.material_binding_identity == markers.material_surface_binding_identity
    assert report.torque_residual_n_m <= report.torque_roundoff_bound_n_m
    assert report.material_power_residual_w <= report.material_power_roundoff_bound_w
    assert report.active_marker_count == len(indices)


def test_iqn_trial_guess_survives_restore_and_cap_uses_same_derivative(material_surface):
    markers, _, _, _, _, _, _ = _reset(material_surface)
    base = capture_marker_interface_state(markers)
    guess = np.arange(18, dtype=float).reshape(6, 3) / 64
    restore_marker_interface_state(markers, marker_trial_state(base, guess))
    observed = markers.v_gamma_mps.to_numpy()
    np.testing.assert_array_equal(observed[:6], guess)
    np.testing.assert_array_equal(observed[8], 1.5 * guess[2] - 0.5 * guess[1])
    np.testing.assert_array_equal(observed[9], 1.5 * guess[5] - 0.5 * guess[4])


def test_malformed_material_state_is_rejected_before_geometry_publication(material_surface):
    markers, position, velocity, _, particle_reference, _, _ = _reset(material_surface)
    invalid = particle_reference.copy()
    invalid[0, 0] = np.nan
    position.from_numpy(invalid)
    with pytest.raises(ValueError, match="finite"):
        markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)


@pytest.mark.parametrize("corruption", ("position", "force", "overflow", "transpose"))
def test_invalid_material_load_does_not_write_particle_force(material_surface, corruption):
    markers, position, velocity, force, _, _, _ = _reset(material_surface)
    markers.set_marker_tractions_pa(np.ones((6, 3)))
    markers.compute_marker_forces()
    force.fill(0.25)
    before = force.to_numpy().copy()
    if corruption == "position":
        bad = markers.x_gamma_m.to_numpy()
        bad[0, 0] += 0.01
        markers.x_gamma_m.from_numpy(bad)
    elif corruption in ("force", "overflow"):
        bad = markers.F_gamma_n.to_numpy()
        bad[0, 0] = np.nan if corruption == "force" else 1e100
        markers.F_gamma_n.from_numpy(bad)
    original_weights = markers._material_surface_transfer.column_weight.to_numpy().copy()
    if corruption == "transpose":
        markers._material_surface_transfer.column_weight[0] *= 2
    try:
        with pytest.raises((ValueError, RuntimeError), match="material"):
            markers.scatter_marker_forces_to_mpm_particles(
                force, position, particle_count=6, support_radius_m=0.25,
                particle_velocity_mps=velocity,
            )
    finally:
        markers._material_surface_transfer.column_weight.from_numpy(original_weights)
    np.testing.assert_array_equal(force.to_numpy(), before)


@pytest.mark.parametrize("field", ("x_gamma_m", "v_gamma_mps", "n_gamma", "A_gamma_m2", "pressure_probe_origin_m"))
def test_accepted_material_restore_rejects_inconsistent_surface_without_writes(material_surface, field):
    markers, position, velocity, _, _, _, _ = _reset(material_surface)
    state = capture_marker_interface_state(markers)
    markers.validate_accepted_material_surface_state(
        state, particle_positions_m=position.to_numpy(), particle_velocities_mps=velocity.to_numpy(),
    )
    before = {name: getattr(markers, name).to_numpy().copy() for name in state if not name.startswith("_")}
    state[field].flat[0] += 0.01
    with pytest.raises(ValueError, match="material"):
        markers.validate_accepted_material_surface_state(
            state, particle_positions_m=position.to_numpy(), particle_velocities_mps=velocity.to_numpy(),
        )
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(markers, name).to_numpy(), expected)


def test_material_identity_is_part_of_iqn_layout_and_restore_contract(material_surface):
    markers, _, _, _, _, reference, _ = _reset(material_surface)
    bound = marker_layout_identity(markers, reference_positions_m=reference)
    transfer = markers._material_surface_transfer
    state = capture_marker_interface_state(markers)
    assert state["_marker_geometry"]["material_surface_binding_identity"] == transfer.identity_sha256
    try:
        markers._material_surface_transfer = None
        unbound = marker_layout_identity(markers, reference_positions_m=reference)
    finally:
        markers._material_surface_transfer = transfer
    assert bound != unbound
    state["_marker_geometry"]["material_surface_binding_identity"] = "0" * 64
    before = markers.x_gamma_m.to_numpy().copy()
    with pytest.raises(ValueError, match="material.*identity"):
        restore_marker_interface_state(markers, state)
    np.testing.assert_array_equal(markers.x_gamma_m.to_numpy(), before)


def test_bound_topology_cannot_be_silently_replaced(material_surface):
    markers, _, _, _, _, _, _ = _reset(material_surface)
    before = markers.projection_triangle_indices.to_numpy().copy()
    with pytest.raises(ValueError, match="material.*bound"):
        markers.set_projection_segments(())
    np.testing.assert_array_equal(markers.projection_triangle_indices.to_numpy(), before)


def test_failed_geometry_can_restore_original_accepted_material_layout(material_surface):
    markers, position, velocity, _, reference_particles, _, _ = _reset(material_surface)
    state = capture_marker_interface_state(markers)
    position.from_numpy(np.full_like(reference_particles, 0.5))
    with pytest.raises(RuntimeError, match="material.*geometry"):
        markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=6)
    assert markers.marker_count == 0
    restore_marker_interface_state(markers, state)
    position.from_numpy(reference_particles)
    markers.validate_accepted_material_surface_state(
        state, particle_positions_m=position.to_numpy(), particle_velocities_mps=velocity.to_numpy(),
    )
    np.testing.assert_array_equal(markers.x_gamma_m.to_numpy()[:6], state["x_gamma_m"])


def test_accepted_material_restore_rejects_collapsed_derived_cap_before_writes(material_surface):
    from simulation_core.coupling.hibm_mpm.material_surface_binding import interpolate_material_surface

    markers, position, velocity, _, reference_particles, _, axis = _reset(material_surface)
    state = capture_marker_interface_state(markers)
    collapsed_particles = reference_particles.copy()
    collapsed_particles[:, (axis + 2) % 3] = 0.5
    collapsed_markers = interpolate_material_surface(
        markers._material_surface_transfer.binding, collapsed_particles,
    ).astype(np.float32)
    proposed = {**state, "x_gamma_m": collapsed_markers,
                "pressure_probe_origin_m": collapsed_markers.copy()}
    before = {name: getattr(markers, name).to_numpy().copy()
              for name in state if not name.startswith("_")}
    # Each side chain still has nonzero length and the state satisfies W*x,
    # W*v, normals and area. Only the derived cap has collapsed thickness.
    with pytest.raises(ValueError, match="cap"):
        markers.validate_accepted_material_surface_state(
            proposed, particle_positions_m=collapsed_particles,
            particle_velocities_mps=velocity.to_numpy(),
        )
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(markers, name).to_numpy(), expected)
    np.testing.assert_array_equal(position.to_numpy(), reference_particles)


def test_runtime_restore_rejects_collapsed_cap_with_real_state_validators(material_surface, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from benchmarks.official import solid_mpm_fsi_runner as runner
    from simulation_core.coupling.hibm_mpm.material_surface_binding import interpolate_material_surface
    from tests.solvers.test_material_surface_checkpoint import _bound_checkpoint_state

    state = _bound_checkpoint_state(material_surface)
    markers, _, _, _, _, _, axis = material_surface
    particles = state.macro_state.solid_fields["x"].copy()
    particles[:, (axis + 2) % 3] = 0.5
    points = interpolate_material_surface(markers._material_surface_transfer.binding, particles).astype(np.float32)
    macro = replace(state.macro_state,
                    solid_fields={**state.macro_state.solid_fields, "x": particles},
                    marker_state={**state.macro_state.marker_state, "x_gamma_m": points,
                                  "pressure_probe_origin_m": points.copy()})
    proposed = replace(state, macro_state=macro)
    writes = []

    class Field:
        def __init__(self, values):
            self.values = values.copy()

        def to_numpy(self):
            return self.values.copy()

        def from_numpy(self, values):
            writes.append("field")
            raise AssertionError("invalid cap reached a runtime write")

    fluid = SimpleNamespace(**{name: Field(value) for name, value in {
        **macro.fluid_fields, **proposed.fluid_boundary_fields,
    }.items()}, velocity_dirichlet_boundary_authority="canonical")
    solid = SimpleNamespace(**{name: Field(value) for name, value in macro.solid_fields.items()},
                            particle_count=macro.solid_particle_count)

    def write_barrier(*args, **kwargs):
        writes.append("macro")
        raise AssertionError("invalid cap reached macro restore")

    # Only unrelated backend preparation/writes are isolated. Both the full
    # checkpoint validator and the bound material validator execute unchanged.
    monkeypatch.setattr(runner, "_snapshot_auxiliary_restore_inputs", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_canonical_snapshot_restore_prepare_plan", lambda *a, **k: None)
    monkeypatch.setattr(runner, "restore_host_macro_step_state", write_barrier)
    before = markers.x_gamma_m.to_numpy().copy()
    with pytest.raises(ValueError, match="cap"):
        runner._restore_accepted_fsi_runtime_state(
            proposed, fluid=fluid, solid=solid, markers=markers, gradient_field=None,
        )
    assert not writes
    np.testing.assert_array_equal(markers.x_gamma_m.to_numpy(), before)
