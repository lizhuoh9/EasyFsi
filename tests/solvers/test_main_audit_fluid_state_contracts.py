from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import taichi as ti

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.fluids import solver as fluid_solver_module
from simulation_core.fluids.solver import CartesianFluidSolver
from simulation_core.fluids.spec import FluidDomainSpec


class _SyntheticDeviceFailure(RuntimeError):
    pass


def _device_work_must_not_start(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("device work must not start")


class _ScalarField:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __getitem__(self, _index: object) -> int:
        return self.value

    def __setitem__(self, _index: object, value: int) -> None:
        self.value = int(value)


class _VectorField:
    n = 3
    m = 1

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.shape = (len(self.values),)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.values, copy=True)


class FluidDomainFiniteContracts(unittest.TestCase):
    def test_domain_rejects_nonfinite_bounds_and_physical_parameters(self) -> None:
        valid = {
            "bounds_min_m": (0.0, 0.0, 0.0),
            "bounds_max_m": (1.0, 1.0, 1.0),
            "grid_nodes": (4, 4, 4),
            "density_kgm3": 1.0,
            "viscosity_pa_s": 1.0e-5,
            "dt_s": 1.0e-3,
        }
        invalid_overrides = (
            {"bounds_min_m": (float("nan"), 0.0, 0.0)},
            {"bounds_max_m": (1.0, float("inf"), 1.0)},
            {"density_kgm3": float("nan")},
            {"density_kgm3": float("inf")},
            {"density_kgm3": 0.0},
            {"viscosity_pa_s": float("nan")},
            {"viscosity_pa_s": float("inf")},
            {"viscosity_pa_s": -1.0},
            {"dt_s": float("nan")},
            {"dt_s": float("inf")},
            {"dt_s": 0.0},
        )
        for override in invalid_overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                FluidDomainSpec(**{**valid, **override})


class FluidPublicBoundaryContracts(unittest.TestCase):
    @staticmethod
    def _bare_solver() -> CartesianFluidSolver:
        solver = object.__new__(CartesianFluidSolver)
        solver.dt = 1.0e-3
        solver.rho = 1.0
        solver.mu = 1.0e-5
        solver._require_velocity_dirichlet_component_ledger_sealed = lambda: None
        return solver

    def test_step_apis_reject_nonfinite_dt_before_device_work(self) -> None:
        for method_name in ("predict", "apply_body_force", "project"):
            for bad_dt in (float("nan"), float("inf"), -float("inf"), 1.0e-100):
                with self.subTest(method=method_name, dt=bad_dt):
                    solver = self._bare_solver()
                    solver._max_fluid_speed_kernel = _device_work_must_not_start
                    solver._apply_body_force_kernel = _device_work_must_not_start
                    solver._resolve_velocity_inlet_zmax_topology_mode = (
                        _device_work_must_not_start
                    )
                    with self.assertRaises(ValueError):
                        getattr(solver, method_name)(dt_s=bad_dt)

    def test_pressure_coefficients_reject_derived_f32_overflow(self) -> None:
        with self.assertRaises(ValueError):
            fluid_solver_module._pressure_poisson_coefficients(
                density_kgm3=3.0e38,
                dt_s=1.0e-38,
                spacing_m=(1.0, 1.0, 1.0),
            )

        solver = self._bare_solver()
        solver.rho = 3.0e38
        solver.dt = 1.0e-38
        solver.dx = solver.dy = solver.dz = 1.0
        solver._require_velocity_dirichlet_component_ledger_sealed = (
            _device_work_must_not_start
        )
        with self.assertRaises(ValueError):
            solver.project()

    def test_step_apis_reject_invalid_stored_density_before_device_work(self) -> None:
        for method_name in ("predict", "apply_body_force", "project"):
            for density in (float("nan"), float("inf"), 0.0, -1.0):
                with self.subTest(method=method_name, density=density):
                    solver = self._bare_solver()
                    solver.rho = density
                    solver._max_fluid_speed_kernel = _device_work_must_not_start
                    solver._apply_body_force_kernel = _device_work_must_not_start
                    solver._resolve_velocity_inlet_zmax_topology_mode = (
                        _device_work_must_not_start
                    )
                    with self.assertRaises(ValueError):
                        getattr(solver, method_name)()

    def test_predict_rejects_invalid_stored_viscosity_before_device_work(self) -> None:
        for viscosity in (float("nan"), float("inf"), -1.0):
            with self.subTest(viscosity=viscosity):
                solver = self._bare_solver()
                solver.mu = viscosity
                solver._max_fluid_speed_kernel = _device_work_must_not_start
                with self.assertRaises(ValueError):
                    solver.predict()


class FluidTopologyInvalidationContracts(unittest.TestCase):
    @staticmethod
    def _bare_solver() -> CartesianFluidSolver:
        solver = object.__new__(CartesianFluidSolver)
        solver._hibm_base_obstacle_initialized = True
        solver._sst_wall_distance_valid = True
        solver._sst_wall_distance_cache_key = ("stale",)
        solver._sst_wall_distance_base_cache_key = ("stale-base",)
        solver._invalidator_calls = []

        def invalidate_reachability() -> None:
            solver._invalidator_calls.append("reachability")

        def invalidate_ledger() -> None:
            solver._invalidator_calls.append("ledger")

        solver._invalidate_hibm_pressure_reachability = invalidate_reachability
        solver._invalidate_velocity_dirichlet_component_ledger = invalidate_ledger
        return solver

    def test_unified_invalidator_retires_every_topology_cache(self) -> None:
        solver = self._bare_solver()

        solver._invalidate_external_obstacle_topology_derived_state()

        self.assertEqual(solver._invalidator_calls, ["reachability", "ledger"])
        self.assertFalse(solver._sst_wall_distance_valid)
        self.assertIsNone(solver._sst_wall_distance_cache_key)
        self.assertIsNone(solver._sst_wall_distance_base_cache_key)

    def test_dynamic_obstacle_noop_preserves_sst_base_cache(self) -> None:
        solver = self._bare_solver()
        base_key = ("obstacle-domain-base",)
        solver._sst_wall_distance_base_cache_key = base_key
        solver.hibm_external_obstacle_topology_revision = 7
        solver.hibm_dynamic_solid_volume_enabled = False
        solver.report_hibm_fresh_fluid_cells = _ScalarField()
        solver.report_dynamic_obstacle_cell_count = _ScalarField(4)
        solver.report_dynamic_obstacle_added_cell_count = _ScalarField()
        solver.report_dynamic_obstacle_removed_cell_count = _ScalarField()
        solver.reduction_count = _ScalarField()
        solver._store_hibm_dynamic_solid_volume_candidate_kernel = lambda: None
        solver._reconstruct_fresh_fluid_cells = lambda: None

        solver._commit_hibm_dynamic_solid_volume_candidate()

        self.assertEqual(solver.hibm_external_obstacle_topology_revision, 7)
        self.assertEqual(solver._sst_wall_distance_base_cache_key, base_key)
        self.assertFalse(solver._sst_wall_distance_valid)

    def test_generic_dynamic_obstacle_noop_preserves_sst_base_cache(self) -> None:
        solver = self._bare_solver()
        base_key = ("obstacle-domain-base",)
        solver._sst_wall_distance_base_cache_key = base_key
        solver.hibm_external_obstacle_topology_revision = 7
        solver.report_hibm_fresh_fluid_cells = _ScalarField()
        solver.report_dynamic_obstacle_cell_count = _ScalarField(4)
        solver.report_dynamic_obstacle_added_cell_count = _ScalarField()
        solver.report_dynamic_obstacle_removed_cell_count = _ScalarField()
        solver.reduction_count = _ScalarField()
        solver._count_nonfinite_dynamic_solid_particle_positions_kernel = (
            lambda *_args: None
        )
        solver._reset_dynamic_solid_obstacle_candidate_kernel = lambda: None
        solver._ensure_hibm_base_obstacle = lambda: None
        solver._invalidate_external_obstacle_topology_derived_state = lambda: setattr(
            solver,
            "_sst_wall_distance_base_cache_key",
            None,
        )
        solver._apply_dynamic_solid_obstacle_candidate_kernel = lambda: None
        solver._reconstruct_fresh_fluid_cells = lambda: None

        solver.update_dynamic_solid_obstacle_from_particles(
            _VectorField(np.empty((0, 3), dtype=np.float32)),
            particle_count=0,
            particle_support_size_m=(1.0, 1.0, 1.0),
            store_as_hibm_dynamic_solid_volume=False,
        )

        self.assertEqual(solver.hibm_external_obstacle_topology_revision, 7)
        self.assertEqual(solver._sst_wall_distance_base_cache_key, base_key)

    def test_hibm_obstacle_writers_use_unified_invalidation_before_device_write(
        self,
    ) -> None:
        invocations = (
            lambda solver: solver.apply_hibm_internal_obstacles(
                object(), internal_node_code=2
            ),
            lambda solver: solver.mark_hibm_solid_band_nonprojectable_cells(),
            lambda solver: solver.convert_hibm_air_backed_cells(),
        )
        for invoke in invocations:
            with self.subTest(invoke=invoke):
                solver = self._bare_solver()
                unified_calls: list[str] = []

                def invalidate_all() -> None:
                    unified_calls.append("all")

                def fail_device_write(*_args: object, **_kwargs: object) -> None:
                    raise _SyntheticDeviceFailure

                solver._invalidate_external_obstacle_topology_derived_state = (
                    invalidate_all
                )
                solver._apply_hibm_internal_obstacles_kernel = fail_device_write
                solver._mark_hibm_solid_band_nonprojectable_cells_kernel = (
                    fail_device_write
                )
                solver._mark_hibm_air_backed_cells_kernel = fail_device_write
                solver._reset_row_cloud_orphan_cleanup_report_kernel = (
                    fail_device_write
                )
                solver.last_hibm_pressure_unreached_component_overflow = False
                solver.last_hibm_pressure_component_labels_converged = True
                with mock.patch.dict(
                    os.environ,
                    {"HIBM_BAND_COUNT_ONLY": "0", "HIBM_BAND_INTERIOR_ONLY": "0"},
                    clear=False,
                ), self.assertRaises(_SyntheticDeviceFailure):
                    invoke(solver)
                self.assertEqual(unified_calls, ["all"])

        # Row-cloud cleanup deliberately performs a selection-only device pass
        # before invalidation so a no-op preserves the current topology-derived
        # contracts.  Its dedicated regression verifies that a non-empty
        # selection invalidates immediately before the commit pass.


class FluidSnapshotContracts(unittest.TestCase):
    def test_rollback_restores_sst_wall_generation_sources(self) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver._save_state_kernel = lambda: None
        solver._restore_state_kernel = lambda: None
        solver._clear_hibm_pressure_outlet_classification_kernel = lambda: None
        solver._invalidate_external_obstacle_topology_derived_state = lambda: None
        solver._reset_hibm_pressure_unreached_component_distribution_stats = lambda: None
        solver.hibm_external_obstacle_topology_revision = 3
        solver.hibm_dynamic_solid_volume_enabled = False
        solver.sst_no_slip_domain_wall_mask = _ScalarField(0b001011)
        solver._sst_no_slip_domain_walls = (True, True, False, True, False, False)
        solver._sst_wall_distance_valid = True
        solver._sst_wall_distance_cache_key = ("saved-geometry",)

        solver.save_state()
        solver._sst_no_slip_domain_walls = (False,) * 6
        solver.sst_no_slip_domain_wall_mask[None] = 0
        solver._sst_wall_distance_valid = False
        solver._sst_wall_distance_cache_key = None
        solver._sst_wall_distance_base_cache_key = None
        solver.restore_state()

        self.assertEqual(
            solver._sst_no_slip_domain_walls,
            (True, True, False, True, False, False),
        )
        self.assertEqual(solver.sst_no_slip_domain_wall_mask[None], 0b001011)
        self.assertTrue(solver._sst_wall_distance_valid)
        self.assertEqual(solver._sst_wall_distance_cache_key, ("saved-geometry",))

    def test_device_snapshot_round_trips_fsi_pressure_and_sst_wall_source(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4)),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        expected_pressure = np.full((4, 4, 4), 17.25, dtype=np.float64)
        expected_distance = np.full((4, 4, 4), 0.125, dtype=np.float32)
        wall_flags = (True, False, True, False, False, True)
        wall_mask = sum(
            (1 << face) for face, active in enumerate(wall_flags) if active
        )
        solver.fsi_pressure.from_numpy(expected_pressure)
        solver.sst_wall_distance_m.from_numpy(expected_distance)
        solver._sst_no_slip_domain_walls = wall_flags
        solver.sst_no_slip_domain_wall_mask[None] = wall_mask
        solver._sst_wall_distance_valid = True
        solver._sst_wall_distance_cache_key = (
            solver._sst_wall_distance_input_identity(
                wall_flags=wall_flags,
                marker_position_m=None,
                marker_count=0,
                projection_segment_indices=None,
                projection_segment_count=0,
                inactive_axis=-1,
            )
        )
        solver.save_state()

        solver.fsi_pressure.fill(-3.0)
        solver.sst_wall_distance_m.fill(9.0)
        solver._sst_no_slip_domain_walls = (False,) * 6
        solver.sst_no_slip_domain_wall_mask[None] = 0
        solver.restore_state()

        np.testing.assert_allclose(solver.fsi_pressure.to_numpy(), expected_pressure)
        np.testing.assert_allclose(
            solver.sst_wall_distance_m.to_numpy(),
            expected_distance,
        )
        self.assertEqual(solver._sst_no_slip_domain_walls, wall_flags)
        self.assertEqual(int(solver.sst_no_slip_domain_wall_mask[None]), wall_mask)
        self.assertTrue(solver._sst_wall_distance_valid)
        self.assertEqual(
            solver._sst_wall_distance_cache_key.obstacle_topology_revision,
            solver.hibm_external_obstacle_topology_revision,
        )


class SSTCountAndCacheContracts(unittest.TestCase):
    @staticmethod
    def _cache_solver() -> CartesianFluidSolver:
        solver = object.__new__(CartesianFluidSolver)
        solver.nx = solver.ny = solver.nz = 4
        solver.hibm_external_obstacle_topology_revision = 7
        solver._sst_no_slip_domain_walls = (False,) * 6
        solver._sst_wall_distance_valid = False
        solver._sst_wall_distance_cache_key = None
        solver._sst_wall_distance_base_cache_key = None
        solver.sst_no_slip_domain_wall_mask = _ScalarField()
        solver.reduction_count = _ScalarField()
        solver._wall_distance_kernel_calls = []

        def record(name: str):
            def call(*_args: object, **_kwargs: object) -> None:
                solver._wall_distance_kernel_calls.append(name)

            return call

        solver._initialize_sst_wall_distance_kernel = record("initialize")
        solver._restore_sst_base_wall_distance_kernel = record("restore_base")
        solver._include_sst_marker_wall_distance_kernel = record("marker")
        solver._include_sst_projection_segment_wall_distance_kernel = record(
            "segment"
        )
        return solver

    def test_counted_fields_reject_counts_beyond_capacity(self) -> None:
        marker = _VectorField(np.zeros((2, 3), dtype=np.float32))
        segment = _VectorField(np.zeros((1, 3), dtype=np.int32))
        solver = self._cache_solver()
        with self.assertRaises(ValueError):
            solver.prepare_sst_wall_distance(
                marker_position_m=marker,
                marker_count=3,
            )
        with self.assertRaises(ValueError):
            solver.prepare_sst_wall_distance(
                marker_position_m=marker,
                projection_segment_indices=segment,
                projection_segment_count=2,
            )
        with self.assertRaises(ValueError):
            solver.spread_surface_forces(
                marker,
                marker,
                vertex_count=3,
                center_m=(0.0, 0.0, 0.0),
            )

    def test_surface_force_parameters_must_be_representable_as_f32(self) -> None:
        marker = _VectorField(np.zeros((1, 3), dtype=np.float32))
        solver = object.__new__(CartesianFluidSolver)
        solver.reduction_count = _ScalarField()
        solver._count_invalid_surface_force_inputs_kernel = lambda *_args: None
        solver._spread_surface_forces_kernel = _device_work_must_not_start

        for center, sign in (
            ((1.0e100, 0.0, 0.0), 1.0),
            ((0.0, 0.0, 0.0), 1.0e100),
            ((1.0e-100, 0.0, 0.0), 1.0),
            ((0.0, 0.0, 0.0), 1.0e-100),
        ):
            with self.subTest(center=center, sign=sign), self.assertRaises(ValueError):
                solver.spread_surface_forces(
                    marker,
                    marker,
                    vertex_count=1,
                    center_m=center,
                    force_sign=sign,
                )

    def test_surface_force_fields_reject_f64_values_outside_f32_range(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4)),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        position = ti.Vector.field(3, dtype=ti.f64, shape=2)
        force = ti.Vector.field(3, dtype=ti.f64, shape=2)
        position.from_numpy(
            np.asarray(((1.0e100, 0.0, 0.0), (0.5, 0.5, 0.5)), dtype=np.float64)
        )
        force.from_numpy(
            np.asarray(((1.0, 0.0, 0.0), (1.0e100, 0.0, 0.0)), dtype=np.float64)
        )

        with self.assertRaisesRegex(ValueError, "f32-representable"):
            solver.spread_surface_forces(
                position,
                force,
                vertex_count=2,
                center_m=(0.0, 0.0, 0.0),
            )
        np.testing.assert_array_equal(solver.force.to_numpy(), 0.0)

    def test_sst_segments_reject_out_of_range_endpoints(self) -> None:
        marker = _VectorField(np.zeros((2, 3), dtype=np.float32))
        segment = _VectorField(np.array(((0, 2, -1),), dtype=np.int32))
        solver = self._cache_solver()
        with self.assertRaisesRegex(ValueError, "endpoint"):
            solver.prepare_sst_wall_distance(
                marker_position_m=marker,
                marker_count=2,
                projection_segment_indices=segment,
                projection_segment_count=1,
            )

    def test_identical_fixed_geometry_reuses_wall_distance_but_changes_rebuild(
        self,
    ) -> None:
        marker = _VectorField(
            np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32)
        )
        segment = _VectorField(np.array(((0, 1, -1),), dtype=np.int32))
        solver = self._cache_solver()
        arguments = {
            "marker_position_m": marker,
            "marker_count": 2,
            "projection_segment_indices": segment,
            "projection_segment_count": 1,
            "inactive_axis": 2,
        }

        solver.prepare_sst_wall_distance(**arguments)
        solver.prepare_sst_wall_distance(**arguments)

        self.assertEqual(solver._wall_distance_kernel_calls.count("segment"), 1)

        marker.values[1, 0] = 0.75
        solver.prepare_sst_wall_distance(**arguments)

        self.assertEqual(solver._wall_distance_kernel_calls.count("segment"), 2)

        solver.hibm_external_obstacle_topology_revision += 1
        solver.prepare_sst_wall_distance(**arguments)
        arguments["inactive_axis"] = 1
        solver.prepare_sst_wall_distance(**arguments)
        arguments["no_slip_domain_walls"] = (True, False, False, False, False, False)
        solver.prepare_sst_wall_distance(**arguments)

        self.assertEqual(solver._wall_distance_kernel_calls.count("segment"), 5)

    def test_moving_marker_reuses_obstacle_base_and_matches_full_rebuild(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4)),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        marker = ti.Vector.field(3, dtype=ti.f32, shape=2)
        segment = ti.Vector.field(3, dtype=ti.i32, shape=1)
        marker.from_numpy(
            np.asarray(((0.25, 0.4, 0.5), (0.75, 0.4, 0.5)), dtype=np.float32)
        )
        segment.from_numpy(np.asarray(((0, 1, -1),), dtype=np.int32))
        calls = {"initialize": 0, "overlay": 0}
        initialize = solver._initialize_sst_wall_distance_kernel
        overlay = solver._include_sst_projection_segment_wall_distance_kernel

        def counted_initialize(*args: object) -> None:
            calls["initialize"] += 1
            initialize(*args)

        def counted_overlay(*args: object) -> None:
            calls["overlay"] += 1
            overlay(*args)

        solver._initialize_sst_wall_distance_kernel = counted_initialize
        solver._include_sst_projection_segment_wall_distance_kernel = counted_overlay
        arguments = {
            "marker_position_m": marker,
            "marker_count": 2,
            "projection_segment_indices": segment,
            "projection_segment_count": 1,
            "inactive_axis": 2,
        }

        solver.prepare_sst_wall_distance(**arguments)
        moved = marker.to_numpy()
        moved[:, 1] += 0.1
        marker.from_numpy(moved)
        solver.prepare_sst_wall_distance(**arguments)
        cached_result = solver.sst_wall_distance_m.to_numpy()

        self.assertEqual(calls, {"initialize": 1, "overlay": 2})

        solver._sst_wall_distance_valid = False
        solver._sst_wall_distance_cache_key = None
        solver._sst_wall_distance_base_cache_key = None
        solver.prepare_sst_wall_distance(**arguments)

        self.assertEqual(calls, {"initialize": 2, "overlay": 3})
        np.testing.assert_allclose(
            cached_result,
            solver.sst_wall_distance_m.to_numpy(),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
