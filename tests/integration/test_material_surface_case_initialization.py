"""Real case initializers must satisfy the material binding before fluid work."""

from dataclasses import replace

import numpy as np
import pytest
import taichi as ti

from benchmarks.official.solid_mpm_fsi_runner import (
    _build_markers,
    _build_solid,
    _configure_material_surface_transfer,
)
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig, selected_formulation_solver_config
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, init_taichi, taichi_runtime_identity
from simulation_core.coupling.hibm_mpm.core import HibmMpmSurfaceMarkers
from simulation_core.coupling.hibm_mpm.interface_state import capture_marker_interface_state


@pytest.fixture(scope="module")
def strict_runtime():
    runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32", strict_arch=True)
    init_taichi(runtime)
    identity = taichi_runtime_identity()
    assert identity["actual_arch"] == "cuda" and identity["strict_arch_verified"]
    return runtime


@pytest.mark.parametrize("profile,counts,marker_count,grid", (
    ("selected", (1, 12, 4), 12, (4, 32, 64)),
    ("selected", (1, 256, 20), 64, (4, 256, 320)),
    ("default", (1, 12, 4), 12, (4, 32, 64)),
))
def test_real_case_initialization_has_a_material_surface_map(
    strict_runtime, profile, counts, marker_count, grid,
):
    base = selected_formulation_solver_config(step_count=1) if profile == "selected" else VerticalFlapFsiConfig(step_count=1)
    config = replace(base, solid_particle_counts=counts, marker_count=marker_count, grid_nodes=grid)
    solid = _build_solid(config, strict_runtime)
    markers = _build_markers(config, strict_runtime)
    physical_positions = markers.x_gamma_m.to_numpy()[:markers.marker_count].copy()
    report = _configure_material_surface_transfer(markers, solid, config)
    assert report["method"] == "cartesian_reference_adjoint_v1"
    assert markers.material_surface_binding_identity is not None
    markers.update_material_surface_from_mpm_particles(solid.x, solid.v, particle_count=solid.particle_count)
    np.testing.assert_allclose(
        markers.x_gamma_m.to_numpy()[:markers.marker_count], physical_positions,
        rtol=0.0, atol=2 * np.finfo(np.float32).eps * np.max(np.abs(physical_positions)),
    )


@pytest.mark.parametrize("first_speed,second_speed", ((0.31, 0.29), (1.3, -0.731), (271.7, 0.0123)))
def test_nonbinary_cuda_material_velocity_capture_is_an_admissible_accepted_state(
    strict_runtime, first_speed, second_speed,
):
    markers = HibmMpmSurfaceMarkers(marker_capacity=4, runtime=strict_runtime)
    reference_particles = np.asarray([
        [0.5, y, z] for y in (0.25, 0.75) for z in (0.35, 0.65)
    ], dtype=np.float32)
    marker_positions = np.asarray([
        [0.5, y, z] for z in (0.7, 0.3) for y in (0.5, 0.625)
    ], dtype=np.float32)
    markers.load_markers(
        positions_m=marker_positions, velocities_mps=np.zeros((4, 3)),
        normals=((0, 0, 1), (0, 0, 1), (0, 0, -1), (0, 0, -1)),
        areas_m2=(0.1,) * 4, region_ids=(101, 101, 202, 202),
    )
    markers.set_projection_segments(((0, 1), (2, 3)))
    markers.configure_material_surface_binding(
        particle_reference_positions_m=reference_particles,
        particle_mass_kg=np.ones(4), inactive_axis=0,
    )
    position = ti.Vector.field(3, ti.f32, shape=4)
    velocity = ti.Vector.field(3, ti.f32, shape=4)
    position.from_numpy(reference_particles)
    speeds = np.asarray([first_speed, second_speed, -first_speed, -second_speed], dtype=np.float32)
    particle_velocity = speeds[:, None] * np.asarray([[0, 0, 1]], dtype=np.float32)
    velocity.from_numpy(particle_velocity)
    markers.update_material_surface_from_mpm_particles(position, velocity, particle_count=4)
    # At y=.5 the material interpolation cancels opposing particle velocities.
    # Device fused operations and host summation may differ by f64 roundoff,
    # which cannot be budgeted only from the near-zero final sum.
    state = capture_marker_interface_state(markers)
    markers.validate_accepted_material_surface_state(
        state, particle_positions_m=position.to_numpy(), particle_velocities_mps=velocity.to_numpy(),
    )
    damaged = state["v_gamma_mps"].copy()
    damaged[0, 2] += np.float32(1e-6)
    with pytest.raises(ValueError, match="v_gamma_mps"):
        markers.validate_accepted_material_surface_state(
            {**state, "v_gamma_mps": damaged},
            particle_positions_m=position.to_numpy(), particle_velocities_mps=velocity.to_numpy(),
        )
