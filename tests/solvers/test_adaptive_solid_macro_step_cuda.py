from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig
from simulation_core.diagnostics.runtime import (
    TaichiRuntimeConfig,
    taichi_runtime_identity,
)
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmState


def test_real_cuda_controller_closes_one_macro_step_in_adaptive_and_fixed_modes(
) -> None:
    """Exercise the production controller on one dedicated strict-CUDA runtime."""

    damping_macro = 0.81
    initial_velocity_mps = np.asarray([0.0, 0.02, 0.0], dtype=np.float32)
    base_config = replace(
        VerticalFlapFsiConfig(),
        grid_nodes=(8, 8, 8),
        solid_particle_counts=(1, 1, 1),
        dt_s=1.0e-5,
        velocity_damping=damping_macro,
        solid_substeps=None,
        solid_max_substep_retries=0,
        solid_max_automatic_substeps=16,
        solid_max_deformation_clamp_count_per_macro_step=0,
        enforce_plane_strain_x=False,
    )
    bounds_min_m, bounds_max_m = runner._solid_mpm_bounds(base_config)
    box_min_m, box_max_m = runner._solid_box(base_config)
    solid = NeoHookeanMpmState(
        particle_capacity=1,
        bounds_min_m=bounds_min_m,
        bounds_max_m=bounds_max_m,
        grid_nodes=base_config.grid_nodes,
        runtime=TaichiRuntimeConfig(arch="cuda", strict_arch=True),
    )
    runtime_identity = taichi_runtime_identity()
    assert runtime_identity["requested_arch"] == "cuda"
    assert runtime_identity["actual_arch"] == "cuda"
    assert runtime_identity["default_fp"] == "f32"
    assert runtime_identity["random_seed"] == 0
    assert runtime_identity["strict_arch_verified"] is True
    assert runtime_identity["offline_cache_identity"]["enabled"] is True
    mu_pa, lambda_pa = runner._lame_parameters(base_config)

    cases = (
        (None, "adaptive", 2),
        (4, "fixed_override", 4),
    )
    for requested_substeps, expected_mode, expected_substeps in cases:
        config = replace(base_config, solid_substeps=requested_substeps)
        solid.initialize_box(
            particle_counts=config.solid_particle_counts,
            box_min_m=box_min_m,
            box_max_m=box_max_m,
            density_kgm3=config.solid_density_kgm3,
        )
        solid.set_uniform_velocity(tuple(float(value) for value in initial_velocity_mps))
        particle_position_write_count = 0

        def observe_particle_position_write() -> None:
            nonlocal particle_position_write_count
            particle_position_write_count += 1

        def unexpected_retry_prepare() -> None:
            raise AssertionError("a healthy one-macro-step CUDA trial must not retry")

        result = runner._select_and_advance_solid_macro_step(
            solid,
            config,
            mu_pa=mu_pa,
            lambda_pa=lambda_pa,
            retry_prepare=unexpected_retry_prepare,
            particle_position_write_observer=observe_particle_position_write,
            profile_wall_time=True,
        )

        assert result["solid_substeps_requested"] == requested_substeps
        assert result["solid_substeps_mode"] == expected_mode
        assert result["solid_substeps_selected"] == expected_substeps
        assert result["solid_selector_evaluation_count"] == 1
        assert result["solid_accepted_substep_count"] == expected_substeps
        assert result["solid_substeps_executed_total"] == expected_substeps
        assert result["solid_step_kernel_launch_count"] == expected_substeps
        assert result["solid_selector_device_to_host_scalar_read_count"] == 1
        assert result["solid_packed_report_device_to_host_transfer_count"] == 1
        assert result["solid_guard_batch_count"] == 1
        assert particle_position_write_count == expected_substeps
        assert result["solid_retry_count"] == 0
        assert result["solid_rejected_trial_count"] == 0
        assert result["solid_wall_time_synchronized"] is True
        assert float(result["solid_wall_time_s"]) >= 0.0
        assert result["requested_macro_dt_s"] == pytest.approx(config.dt_s)
        assert result["solid_accepted_time_s"] == pytest.approx(config.dt_s)
        assert result["solid_remaining_unadvanced_time_s"] == 0.0
        assert (
            expected_substeps * float(result["solid_substep_dt_s"])
            == pytest.approx(config.dt_s)
        )

        damping_substep = runner._solid_substep_velocity_damping(
            config,
            solid_substeps=expected_substeps,
        )
        assert damping_substep**expected_substeps == pytest.approx(
            damping_macro,
            rel=1.0e-12,
            abs=1.0e-15,
        )

        report = result["solid_report"]
        assert report.particle_count == 1
        assert report.grid_out_of_bounds_particle_count == 0
        assert report.deformation_clamp_count == 0
        assert report.max_abs_j > 0.0
        assert math.isfinite(report.max_speed_mps)
        assert math.isfinite(float(result["solid_estimated_cfl"]))
        assert float(result["solid_estimated_cfl"]) <= config.solid_cfl_target

        position = solid.x.to_numpy()[: solid.particle_count]
        velocity = solid.v.to_numpy()[: solid.particle_count]
        deformation = solid.F.to_numpy()[: solid.particle_count]
        assert np.isfinite(position).all()
        assert np.isfinite(velocity).all()
        assert np.isfinite(deformation).all()
        np.testing.assert_allclose(
            velocity[0],
            initial_velocity_mps * damping_macro,
            rtol=2.0e-4,
            atol=2.0e-7,
        )
