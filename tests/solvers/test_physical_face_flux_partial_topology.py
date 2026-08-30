"""Independent spatial-mask and preflight atomicity boundary oracles."""

import numpy as np
import pytest

from tests.solvers.test_physical_face_flux_contract import _build_ledger, _cuda_solver


def test_spatial_partial_normal_masks_never_open_unregistered_or_solid_faces():
    solver = _cuda_solver()
    velocity = np.full((4, 4, 4, 3), 13.0, dtype=np.float32)
    solver.velocity.from_numpy(velocity)
    obstacle = np.zeros((4, 4, 4), dtype=np.int32)
    expected_faces = {}
    for axis, name in enumerate("xyz"):
        masks = np.zeros((2, 4, 4), dtype=np.int32)
        values = np.zeros((2, 4, 4, 3), dtype=np.float32)
        for side in (0, 1):
            normal = (1.0 if side == 0 else -1.0) * (axis + 1) * 0.25
            masks[side, 0, 0] = 1 << axis
            values[side, 0, 0, axis] = normal
            masks[side, 1, 1] = 1 << axis  # Explicit zero is still a normal owner.
            masks[side, 2, 2] = 1 << ((axis + 1) % 3)
            values[side, 2, 2, axis] = 99.0  # Inactive normal data must not leak.
            masks[side, 3, 3] = 1 << axis
            values[side, 3, 3, axis] = 0.5
            adjacent = [3, 3, 3]
            adjacent[axis] = 0 if side == 0 else 3
            obstacle[tuple(adjacent)] = 1
            expected = np.zeros((4, 4), dtype=np.float32)
            expected[0, 0] = normal
            expected_faces[axis, side] = expected
        getattr(
            solver, f"external_velocity_boundary_{name}_face_active_component_mask"
        ).from_numpy(masks)
        getattr(solver, f"external_velocity_boundary_{name}_face_value_mps").from_numpy(values)
    solver.obstacle.from_numpy(obstacle)

    _build_ledger(solver, solver.velocity)

    faces = [getattr(solver, f"muscl_normal_velocity_{name}").to_numpy() for name in "xyz"]
    for axis in range(3):
        for side in (0, 1):
            actual = np.take(faces[axis], 0 if side == 0 else -1, axis=axis)
            np.testing.assert_array_equal(actual, expected_faces[axis, side])
    np.testing.assert_array_equal(solver.velocity.to_numpy(), velocity)
    solver.compute_divergence()
    expected_divergence = sum(np.diff(faces[axis], axis=axis) * 4.0 for axis in range(3))
    expected_divergence[obstacle != 0] = 0.0
    np.testing.assert_array_equal(solver.divergence.to_numpy(), expected_divergence)


def _physical_state(solver):
    names = (
        "velocity", "velocity_prev", "velocity_transport_base", "pressure",
        "muscl_normal_velocity_x", "muscl_normal_velocity_y", "muscl_normal_velocity_z",
        "sst_turbulent_kinetic_energy", "sst_specific_dissipation_rate",
        "sst_eddy_viscosity_pa_s", "sst_wall_distance_m",
        "sst_turbulent_kinetic_energy_prev", "sst_specific_dissipation_rate_prev",
        "sst_turbulent_kinetic_energy_next", "sst_specific_dissipation_rate_next",
        "sst_turbulent_kinetic_energy_transport_base",
        "sst_specific_dissipation_rate_transport_base", "sst_no_slip_domain_wall_mask",
    )
    return {name: getattr(solver, name).to_numpy() for name in names}


def test_public_topology_rejection_preserves_sst_and_all_velocity_state():
    solver = _cuda_solver()
    solver.configure_sst_2003(
        inlet_velocity_mps=1.0,
        turbulence_intensity=0.05,
        turbulent_viscosity_ratio=10.0,
        no_slip_domain_walls=(False,) * 6,
    )
    # A zero exact zmax normal must contradict explicit False just as a nonzero
    # target does.  It must not be classified by its numerical velocity value.
    solver.refresh_external_velocity_boundary_face_uniform(
        axis_index=2, side_index=1,
        target_velocity_mps=(0.0, 0.0, 0.0), active_component_mask=4,
    )
    cases = (
        ({"velocity_inlet_zmax": False}, "velocity_inlet_zmax=False conflicts"),
        (
            {"pressure_outlet_zmin": True,
             "no_slip_domain_walls": (False, False, False, False, True, False)},
            "no_slip_zmin.*pressure_outlet_zmin",
        ),
        (
            {"velocity_inlet_zmax": True,
             "no_slip_domain_walls": (False, False, False, False, False, True)},
            "no_slip_zmax.*velocity_inlet_zmax",
        ),
    )
    for method in (solver.predict, solver.advance_sst_transport):
        for arguments, pattern in cases:
            before = _physical_state(solver)
            walls_before = solver._sst_no_slip_domain_walls
            validity_before = solver._sst_wall_distance_valid
            with pytest.raises(ValueError, match=pattern):
                method(advection_scheme="muscl_tvd", **arguments)
            after = _physical_state(solver)
            assert after.keys() == before.keys()
            for name, expected in before.items():
                np.testing.assert_array_equal(after[name], expected, err_msg=name)
            assert solver._sst_no_slip_domain_walls == walls_before
            assert solver._sst_wall_distance_valid == validity_before
