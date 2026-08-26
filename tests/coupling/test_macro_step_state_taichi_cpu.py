from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

# This is a bounded transaction test.  Own a real CPU runtime before importing
# simulation_core so the test cannot contend with an unrelated CUDA run.
ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)

from simulation_core.diagnostics import runtime as sim_runtime

sim_runtime._INITIALIZED = True
sim_runtime._INITIALIZED_ARCH = "cpu"
sim_runtime._INITIALIZED_FP = "f32"

from cases.squid_soft_robot.coupling_sharp import (
    build_hibm_mpm_sharp_coupling_state,
)
from simulation_core.coupling.hibm_mpm.interface_state import (
    MARKER_INTERFACE_STATE_FIELDS,
    capture_marker_interface_state,
    marker_layout_identity,
    marker_trial_state,
    restore_marker_interface_state,
)
from simulation_core.coupling.hibm_mpm.macro_step_state import (
    FLUID_MACRO_STATE_FIELDS,
    SOLID_MACRO_STATE_FIELDS,
    capture_host_macro_step_state,
    restore_host_macro_step_state,
)
from simulation_core.fluids.solver import CartesianFluidSolver, FluidDomainSpec
from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingConfig,
    FsiSolverConfig,
    solve_fsi_runtime,
)
from simulation_core.drivers.hibm_mpm_marker_velocity_runtime import (
    HibmMpmMarkerVelocityRuntime,
)
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmState


def _real_owners():
    fluid = CartesianFluidSolver(
        FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3)
    )
    solid = NeoHookeanMpmState(
        particle_capacity=1,
        bounds_min_m=(0.0, 0.0, 0.0),
        bounds_max_m=(1.0, 1.0, 1.0),
        grid_nodes=(4, 4, 4),
    )
    solid.initialize_box(
        particle_counts=(1, 1, 1),
        box_min_m=(0.25, 0.25, 0.25),
        box_max_m=(0.5, 0.5, 0.5),
        density_kgm3=1.0,
    )
    solid.surface_normal[0] = (0.0, 0.0, 1.0)
    solid.area_weight_m2[0] = 0.04
    solid.region_id[0] = 8
    solid.v[0] = (0.0, 0.0, -0.125)
    coupling = build_hibm_mpm_sharp_coupling_state(
        fluid=fluid,
        solid_mpm=solid,
        runtime=None,
    )
    gradient = ti.field(dtype=ti.f32, shape=1)
    gradient[0] = 1.0
    fluid._sst_wall_distance_valid = True
    fluid._sst_wall_distance_cache_key = ("accepted", 3)
    fluid._sst_no_slip_domain_walls = (
        True,
        False,
        False,
        False,
        False,
        True,
    )
    fluid.sst_no_slip_domain_wall_mask[None] = 33
    fluid.hibm_dynamic_solid_volume_enabled = True
    return fluid, solid, coupling.markers, gradient


class RealTaichiMacroStepRollbackTests(unittest.TestCase):
    def test_rejected_trial_restores_all_device_owners_after_nested_save(self) -> None:
        fluid, solid, markers, gradient = _real_owners()

        accepted = capture_host_macro_step_state(
            fluid=fluid,
            solid=solid,
            markers=markers,
            accepted_step_index=3,
            accepted_time_s=0.0015,
            feedback_available_for_projection=True,
            marker_pressure_neumann_gradient_field=gradient,
        )

        for name in FLUID_MACRO_STATE_FIELDS:
            field = getattr(fluid, name)
            field.from_numpy(np.full_like(field.to_numpy(), 7))
        for name in SOLID_MACRO_STATE_FIELDS:
            field = getattr(solid, name)
            field.from_numpy(np.full_like(field.to_numpy(), 8))
        for name in MARKER_INTERFACE_STATE_FIELDS:
            field = getattr(markers, name)
            field.from_numpy(np.full_like(field.to_numpy(), 9))
        gradient.from_numpy(np.full_like(gradient.to_numpy(), 10))
        fluid._sst_wall_distance_valid = False
        fluid._sst_wall_distance_cache_key = ("rejected", 99)
        fluid._sst_no_slip_domain_walls = (False,) * 6
        fluid.sst_no_slip_domain_wall_mask[None] = 0
        fluid.hibm_dynamic_solid_volume_enabled = False

        # Reproduce the real nested-integrator hazard: each owner overwrites its
        # ordinary single save slot after the coupling transaction was captured.
        fluid.save_state()
        solid.save_state()
        position_writes: list[str] = []
        restore_host_macro_step_state(
            accepted,
            fluid=fluid,
            solid=solid,
            markers=markers,
            marker_pressure_neumann_gradient_field=gradient,
            record_particle_position_write=lambda: position_writes.append("x"),
        )

        for name, expected in accepted.fluid_fields.items():
            np.testing.assert_array_equal(getattr(fluid, name).to_numpy(), expected)
        for name, expected in accepted.solid_fields.items():
            np.testing.assert_array_equal(getattr(solid, name).to_numpy(), expected)
        for name in MARKER_INTERFACE_STATE_FIELDS:
            np.testing.assert_array_equal(
                getattr(markers, name).to_numpy()[: markers.marker_count],
                accepted.marker_state[name],
            )
        np.testing.assert_array_equal(
            gradient.to_numpy()[: markers.marker_count],
            accepted.marker_pressure_neumann_gradient,
        )
        self.assertEqual(position_writes, ["x"])
        self.assertEqual(
            fluid._sst_wall_distance_valid,
            accepted.fluid_host_metadata["sst_wall_distance_valid"],
        )
        self.assertEqual(
            fluid._sst_wall_distance_cache_key,
            accepted.fluid_host_metadata["sst_wall_distance_cache_key"],
        )
        self.assertEqual(
            tuple(fluid._sst_no_slip_domain_walls),
            accepted.fluid_host_metadata["sst_no_slip_domain_walls"],
        )
        self.assertEqual(
            int(fluid.sst_no_slip_domain_wall_mask[None]),
            accepted.fluid_host_metadata["sst_no_slip_domain_wall_mask"],
        )
        self.assertEqual(
            fluid.hibm_dynamic_solid_volume_enabled,
            accepted.fluid_host_metadata[
                "hibm_dynamic_solid_volume_enabled"
            ],
        )

    def test_generic_runtime_repeats_trials_from_identical_real_fields(self) -> None:
        fluid, solid, markers, gradient = _real_owners()
        reference_positions = markers.x_gamma_m.to_numpy()[:1].copy()
        base_fluid_velocity = fluid.velocity.to_numpy()
        base_solid_x = solid.x.to_numpy()
        base_gradient = gradient.to_numpy()
        trial_starts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        position_write_count = 0

        def record_position_write() -> None:
            nonlocal position_write_count
            position_write_count += 1

        def capture_state():
            return capture_host_macro_step_state(
                fluid=fluid,
                solid=solid,
                markers=markers,
                accepted_step_index=0,
                accepted_time_s=0.0,
                feedback_available_for_projection=False,
                marker_pressure_neumann_gradient_field=gradient,
            )

        def restore_state(state, _context) -> None:
            restore_host_macro_step_state(
                state,
                fluid=fluid,
                solid=solid,
                markers=markers,
                marker_pressure_neumann_gradient_field=gradient,
                record_particle_position_write=record_position_write,
            )

        def apply_guess(marker_base, guess) -> None:
            restore_marker_interface_state(
                markers,
                marker_trial_state(marker_base, guess),
            )

        def advance_trial(_context, trial_index):
            trial_starts.append(
                (
                    fluid.velocity.to_numpy(),
                    solid.x.to_numpy(),
                    gradient.to_numpy(),
                )
            )
            fluid_velocity = fluid.velocity.to_numpy()
            fluid_velocity[...] = 10.0 + trial_index
            fluid.velocity.from_numpy(fluid_velocity)
            solid_x = solid.x.to_numpy()
            solid_x[...] += 20.0 + trial_index
            solid.x.from_numpy(solid_x)
            gradient.from_numpy(
                np.full_like(gradient.to_numpy(), 30.0 + trial_index)
            )
            fluid.save_state()
            solid.save_state()
            marker_state = capture_marker_interface_state(markers)
            guess = np.asarray(marker_state["v_gamma_mps"], dtype=np.float64)
            candidate = 0.5 * guess + 1.0
            restore_marker_interface_state(
                markers,
                marker_trial_state(marker_state, candidate),
            )
            return {"trial_index": trial_index}

        runtime = HibmMpmMarkerVelocityRuntime(
            capture_step_state=capture_state,
            restore_step_state=restore_state,
            prepare_step=lambda _context: None,
            capture_marker_state=lambda: capture_marker_interface_state(markers),
            apply_marker_velocity_guess=apply_guess,
            advance_trial=advance_trial,
            commit_case_step=lambda _context, _trial, coupling: {
                "case_iterations": coupling.iterations
            },
            finalize_case_run=lambda: {},
            layout_identity=lambda: marker_layout_identity(
                markers,
                reference_positions_m=reference_positions,
                namespace="real-taichi-cpu-transaction",
            ),
        )

        result = solve_fsi_runtime(
            runtime,
            FsiSolverConfig(
                step_count=1,
                time_step_s=1.0e-3,
                coupling=FsiCouplingConfig(
                    max_iterations=8,
                    relative_tolerance=1.0e-12,
                    absolute_tolerance_mps=1.0e-12,
                    iqn_max_update_ratio=None,
                ),
            ),
        )

        self.assertGreater(len(trial_starts), 1)
        for fluid_start, solid_start, gradient_start in trial_starts:
            np.testing.assert_array_equal(fluid_start, base_fluid_velocity)
            np.testing.assert_array_equal(solid_start, base_solid_x)
            np.testing.assert_array_equal(gradient_start, base_gradient)
        self.assertEqual(result.history[0]["case_iterations"], len(trial_starts))
        self.assertGreaterEqual(position_write_count, len(trial_starts))


if __name__ == "__main__":
    unittest.main()
