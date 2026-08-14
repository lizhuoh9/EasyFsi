from __future__ import annotations

import inspect
import os
import unittest
from unittest import mock

from simulation_core.fluids.solver import CartesianFluidSolver


class _SyntheticDeviceMutation(RuntimeError):
    """Sentinel raised at the first topology-mutating device call."""


class _HostScalar:
    def __init__(self, value: int | float = 0) -> None:
        self.value = value

    def __getitem__(self, _index: object) -> int | float:
        return self.value

    def __setitem__(self, _index: object, value: int | float) -> None:
        self.value = value


class _FailingHostScalar(_HostScalar):
    def __init__(self, observe) -> None:
        super().__init__(0)
        self._observe = observe

    def __setitem__(self, _index: object, value: int | float) -> None:
        self._observe()
        raise _SyntheticDeviceMutation("synthetic scalar device-write failure")


class _ParticleVectorField:
    shape = (1,)
    n = 3
    m = 1


class TopologyCacheInvalidationContracts(unittest.TestCase):
    @staticmethod
    def _solver_with_published_caches() -> CartesianFluidSolver:
        solver = object.__new__(CartesianFluidSolver)
        solver._pressure_outlet_nullspace_graph_valid = True
        solver._pressure_outlet_nullspace_source_component_count = 7
        solver._pressure_outlet_nullspace_component_count = 5
        solver._pressure_outlet_nullspace_graph_context = "stale-graph"
        solver.last_hibm_reachability_valid = True
        solver.hibm_reachability_revision = 11
        solver.last_hibm_reachability_revision = 11
        solver._hibm_reachability_checksum = ("stale",)
        solver._hibm_base_obstacle_initialized = True
        solver.hibm_external_obstacle_topology_revision = 23
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.velocity_dirichlet_component_ledger_generation = 31
        solver.velocity_dirichlet_component_ledger_sealed = True
        solver._velocity_dirichlet_component_ledger_consumer_generations = {
            "apply": 31,
        }
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = {
            "apply": object(),
        }
        solver._sst_wall_distance_valid = True
        solver._sst_wall_distance_cache_key = "stale-overlay"
        solver._sst_wall_distance_base_cache_key = "stale-base"
        return solver

    def _assert_both_caches_invalid(self, solver: CartesianFluidSolver) -> None:
        self.assertFalse(
            bool(solver._pressure_outlet_nullspace_graph_valid),
            "a topology writer may not enter a device mutation with a cached graph",
        )
        self.assertEqual(solver._pressure_outlet_nullspace_source_component_count, 0)
        self.assertEqual(solver._pressure_outlet_nullspace_component_count, 0)
        self.assertEqual(solver._pressure_outlet_nullspace_graph_context, "")
        self.assertFalse(
            bool(solver.last_hibm_reachability_valid),
            "a topology writer may not enter a device mutation with cached reachability",
        )
        self.assertIsNone(solver._hibm_reachability_checksum)
        self.assertFalse(solver._sst_wall_distance_valid)
        self.assertIsNone(solver._sst_wall_distance_cache_key)
        self.assertIsNone(solver._sst_wall_distance_base_cache_key)

    def _raising_mutation(
        self,
        solver: CartesianFluidSolver,
        observed: list[tuple[bool, bool, int, bool, int, int]],
    ):
        def mutate(*_args: object, **_kwargs: object) -> None:
            observed.append(
                (
                    bool(solver._pressure_outlet_nullspace_graph_valid),
                    bool(solver.last_hibm_reachability_valid),
                    int(solver.velocity_dirichlet_component_ledger_generation),
                    bool(solver.velocity_dirichlet_component_ledger_sealed),
                    len(
                        solver._velocity_dirichlet_component_ledger_consumer_generations
                    ),
                    len(
                        solver._velocity_dirichlet_component_ledger_consumer_capabilities
                    ),
                )
            )
            raise _SyntheticDeviceMutation("synthetic topology-kernel failure")

        return mutate

    def _assert_writer_invalidates_before_failed_device_mutation(
        self,
        *,
        solver: CartesianFluidSolver,
        invoke,
    ) -> list[tuple[bool, bool, int, bool, int, int]]:
        observed: list[tuple[bool, bool, int, bool, int, int]] = []
        invoke(observed)
        self.assertEqual(
            [(state[0], state[1]) for state in observed],
            [(False, False)],
            "both derived caches must be retired before the device call begins",
        )
        self._assert_both_caches_invalid(solver)
        return observed

    def test_dynamic_solid_commit_invalidates_both_caches_before_device_write(
        self,
    ) -> None:
        solver = self._solver_with_published_caches()
        solver.reduction_count = _HostScalar(0)
        solver.report_hibm_fresh_fluid_cells = _HostScalar(0)
        solver._count_nonfinite_dynamic_solid_particle_positions_kernel = lambda *_: None
        solver._reset_dynamic_solid_obstacle_candidate_kernel = lambda: None
        solver._rasterize_dynamic_solid_obstacle_candidate_kernel = lambda *_: None

        def invoke(observed: list[tuple[bool, bool]]) -> None:
            solver._store_hibm_dynamic_solid_volume_candidate_kernel = (
                self._raising_mutation(solver, observed)
            )
            with self.assertRaisesRegex(
                _SyntheticDeviceMutation,
                "synthetic topology-kernel failure",
            ):
                solver.update_dynamic_solid_obstacle_from_particles(
                    _ParticleVectorField(),
                    particle_count=1,
                    particle_support_size_m=(1.0, 1.0, 1.0),
                    store_as_hibm_dynamic_solid_volume=True,
                )

        observed = self._assert_writer_invalidates_before_failed_device_mutation(
            solver=solver,
            invoke=invoke,
        )
        self.assertEqual(observed, [(False, False, 32, False, 0, 0)])
        self.assertEqual(solver.hibm_external_obstacle_topology_revision, 24)

    def test_clear_invalidates_all_caches_before_first_device_write(self) -> None:
        solver = self._solver_with_published_caches()
        observed: list[tuple[bool, bool, int, bool, int, int]] = []

        def observe() -> None:
            observed.append(
                (
                    bool(solver._pressure_outlet_nullspace_graph_valid),
                    bool(solver.last_hibm_reachability_valid),
                    int(solver.velocity_dirichlet_component_ledger_generation),
                    bool(solver.velocity_dirichlet_component_ledger_sealed),
                    len(
                        solver._velocity_dirichlet_component_ledger_consumer_generations
                    ),
                    len(
                        solver._velocity_dirichlet_component_ledger_consumer_capabilities
                    ),
                )
            )

        solver.velocity_dirichlet_boundary_authority_code_device = _FailingHostScalar(
            observe
        )

        with self.assertRaisesRegex(
            _SyntheticDeviceMutation,
            "synthetic scalar device-write failure",
        ):
            solver.clear()

        self.assertEqual(observed, [(False, False, 32, False, 0, 0)])
        self._assert_both_caches_invalid(solver)

    def test_dynamic_composite_apply_invalidates_all_caches_before_device_write(
        self,
    ) -> None:
        solver = self._solver_with_published_caches()
        solver.reduction_count = _HostScalar(0)
        solver.report_hibm_fresh_fluid_cells = _HostScalar(0)
        solver._count_nonfinite_dynamic_solid_particle_positions_kernel = lambda *_: None
        solver._reset_dynamic_solid_obstacle_candidate_kernel = lambda: None
        solver._rasterize_dynamic_solid_obstacle_candidate_kernel = lambda *_: None

        def invoke(observed: list[tuple[bool, bool, int, bool, int, int]]) -> None:
            solver._apply_dynamic_solid_obstacle_candidate_kernel = (
                self._raising_mutation(solver, observed)
            )
            with self.assertRaisesRegex(
                _SyntheticDeviceMutation,
                "synthetic topology-kernel failure",
            ):
                solver.update_dynamic_solid_obstacle_from_particles(
                    _ParticleVectorField(),
                    particle_count=1,
                    particle_support_size_m=(1.0, 1.0, 1.0),
                    store_as_hibm_dynamic_solid_volume=False,
                )

        observed = self._assert_writer_invalidates_before_failed_device_mutation(
            solver=solver,
            invoke=invoke,
        )
        self.assertEqual(observed, [(False, False, 32, False, 0, 0)])
        self.assertEqual(solver.hibm_external_obstacle_topology_revision, 24)

    def test_apply_hibm_internal_obstacles_invalidates_both_caches_before_device_write(
        self,
    ) -> None:
        solver = self._solver_with_published_caches()

        def invoke(observed: list[tuple[bool, bool]]) -> None:
            solver._apply_hibm_internal_obstacles_kernel = self._raising_mutation(
                solver,
                observed,
            )
            with self.assertRaises(_SyntheticDeviceMutation):
                solver.apply_hibm_internal_obstacles(
                    object(),
                    internal_node_code=2,
                )

        self._assert_writer_invalidates_before_failed_device_mutation(
            solver=solver,
            invoke=invoke,
        )

    def test_mark_hibm_solid_band_invalidates_both_caches_before_device_write(
        self,
    ) -> None:
        solver = self._solver_with_published_caches()

        def invoke(observed: list[tuple[bool, bool]]) -> None:
            solver._mark_hibm_solid_band_nonprojectable_cells_kernel = (
                self._raising_mutation(solver, observed)
            )
            with mock.patch.dict(
                os.environ,
                {"HIBM_BAND_COUNT_ONLY": "0", "HIBM_BAND_INTERIOR_ONLY": "0"},
                clear=False,
            ):
                with self.assertRaises(_SyntheticDeviceMutation):
                    solver.mark_hibm_solid_band_nonprojectable_cells()

        self._assert_writer_invalidates_before_failed_device_mutation(
            solver=solver,
            invoke=invoke,
        )

    def test_convert_hibm_air_backed_invalidates_both_caches_before_device_write(
        self,
    ) -> None:
        solver = self._solver_with_published_caches()

        def invoke(observed: list[tuple[bool, bool]]) -> None:
            solver._mark_hibm_air_backed_cells_kernel = self._raising_mutation(
                solver,
                observed,
            )
            with self.assertRaises(_SyntheticDeviceMutation):
                solver.convert_hibm_air_backed_cells()

        self._assert_writer_invalidates_before_failed_device_mutation(
            solver=solver,
            invoke=invoke,
        )

    def test_noop_row_cloud_conversion_preserves_published_topology_contracts(
        self,
    ) -> None:
        solver = self._solver_with_published_caches()
        solver.last_hibm_pressure_unreached_component_overflow = False
        solver.last_hibm_pressure_component_labels_converged = True
        solver._hibm_pressure_unreached_component_count = 0
        solver.report_hibm_row_cloud_orphan_cells = _HostScalar(0)
        solver.report_hibm_row_cloud_orphan_components = _HostScalar(0)
        calls: list[tuple[object, ...]] = []

        def reset_report() -> None:
            solver.report_hibm_row_cloud_orphan_cells[None] = 0
            solver.report_hibm_row_cloud_orphan_components[None] = 0

        def no_candidates(*args: object) -> int:
            calls.append(args)
            return 0

        solver._reset_row_cloud_orphan_cleanup_report_kernel = reset_report
        solver._convert_row_cloud_raw_singletons_kernel = no_candidates

        converted = solver.convert_hibm_row_cloud_orphan_components(
            max_component_cells=1,
        )

        self.assertEqual(converted, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-1], 0, "the first pass must be selection-only")
        self.assertTrue(solver._pressure_outlet_nullspace_graph_valid)
        self.assertTrue(solver.last_hibm_reachability_valid)
        self.assertEqual(solver.velocity_dirichlet_component_ledger_generation, 31)
        self.assertTrue(solver.velocity_dirichlet_component_ledger_sealed)
        self.assertTrue(solver._sst_wall_distance_valid)
        self.assertEqual(solver._sst_wall_distance_cache_key, "stale-overlay")
        self.assertEqual(solver._sst_wall_distance_base_cache_key, "stale-base")

    def test_convert_row_cloud_orphans_invalidates_before_commit_device_write(
        self,
    ) -> None:
        solver = self._solver_with_published_caches()
        solver.last_hibm_pressure_unreached_component_overflow = False
        solver.last_hibm_pressure_component_labels_converged = True
        solver._hibm_pressure_unreached_component_count = 0
        solver.report_hibm_row_cloud_orphan_cells = _HostScalar(0)
        solver.report_hibm_row_cloud_orphan_components = _HostScalar(0)

        def reset_report() -> None:
            solver.report_hibm_row_cloud_orphan_cells[None] = 0
            solver.report_hibm_row_cloud_orphan_components[None] = 0

        solver._reset_row_cloud_orphan_cleanup_report_kernel = reset_report

        observed: list[tuple[bool, bool, int, bool, int, int]] = []

        def select_then_commit(*args: object) -> int:
            observed.append(
                (
                    bool(solver._pressure_outlet_nullspace_graph_valid),
                    bool(solver.last_hibm_reachability_valid),
                    int(solver.velocity_dirichlet_component_ledger_generation),
                    bool(solver.velocity_dirichlet_component_ledger_sealed),
                    len(
                        solver._velocity_dirichlet_component_ledger_consumer_generations
                    ),
                    len(
                        solver._velocity_dirichlet_component_ledger_consumer_capabilities
                    ),
                )
            )
            if len(args) < 7:
                raise _SyntheticDeviceMutation(
                    "row-cloud conversion is not split into select and commit"
                )
            commit_obstacle = int(args[-1])
            solver.report_hibm_row_cloud_orphan_cells[None] = 1
            solver.report_hibm_row_cloud_orphan_components[None] = 1
            if commit_obstacle != 0:
                raise _SyntheticDeviceMutation(
                    "synthetic topology-kernel failure"
                )
            return 1

        solver._convert_row_cloud_raw_singletons_kernel = select_then_commit
        with self.assertRaises(_SyntheticDeviceMutation):
            solver.convert_hibm_row_cloud_orphan_components(
                max_component_cells=1,
            )

        self.assertEqual(
            observed,
            [
                (
                    True,
                    True,
                    31,
                    True,
                    1,
                    1,
                ),
                (False, False, 32, False, 0, 0),
            ],
        )
        self._assert_both_caches_invalid(solver)

    def test_canonical_tiny_cleanup_rebuilds_before_every_reflood(self) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.velocity_dirichlet_component_ledger_sealed = True
        solver.last_hibm_pressure_unreached_cell_count = 9
        solver.last_hibm_row_cloud_orphan_component_count = 0
        conversions = iter((2, 1, 0))
        events: list[str] = []

        def convert(**_kwargs: object) -> int:
            events.append("convert")
            converted = next(conversions)
            solver.last_hibm_row_cloud_orphan_component_count = (
                1 if converted > 0 else 0
            )
            if converted > 0:
                solver.velocity_dirichlet_component_ledger_sealed = False
            return converted

        def rebuild() -> None:
            events.append("rebuild")
            solver.velocity_dirichlet_component_ledger_sealed = True

        def reflood(**_kwargs: object) -> int:
            self.assertTrue(solver.velocity_dirichlet_component_ledger_sealed)
            events.append("reflood")
            solver.last_hibm_pressure_unreached_cell_count -= 1
            return solver.last_hibm_pressure_unreached_cell_count

        solver.convert_hibm_row_cloud_orphan_components = convert
        solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells = reflood

        report = solver.cleanup_hibm_pressure_outlet_tiny_unreached_components(
            max_component_cells=4,
            reachability_is_current=True,
            after_topology_mutation=rebuild,
        )

        self.assertEqual(
            events,
            [
                "convert",
                "rebuild",
                "reflood",
                "convert",
                "rebuild",
                "reflood",
                "convert",
            ],
        )
        self.assertEqual(
            report["hibm_preassembly_tiny_unreached_cleanup_cell_count"],
            3,
        )
        self.assertEqual(
            report["hibm_preassembly_tiny_unreached_cleanup_component_count"],
            2,
        )
        self.assertEqual(
            report["hibm_preassembly_tiny_unreached_cleanup_pass_count"],
            2,
        )

    def test_canonical_tiny_cleanup_requires_a_post_mutation_rebuild(self) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.last_hibm_pressure_unreached_cell_count = 0
        solver.convert_hibm_row_cloud_orphan_components = lambda **_kwargs: 0

        with self.assertRaisesRegex(RuntimeError, "rebuild|canonical"):
            solver.cleanup_hibm_pressure_outlet_tiny_unreached_components(
                max_component_cells=1,
                reachability_is_current=True,
            )

    def test_project_rejects_canonical_cleanup_before_post_mutation_reflood(
        self,
    ) -> None:
        source = inspect.getsource(CartesianFluidSolver.project)
        conversion = "self.convert_hibm_row_cloud_orphan_components("
        guard = "reject_canonical_projection_topology_mutation()"
        reflood = (
            "self.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells("
        )
        cursor = 0
        for _ in range(2):
            conversion_index = source.index(conversion, cursor)
            guard_index = source.index(guard, conversion_index)
            reflood_index = source.index(reflood, conversion_index)
            self.assertLess(conversion_index, guard_index)
            self.assertLess(guard_index, reflood_index)
            cursor = guard_index + len(guard)

    def test_mark_sphere_invalidates_both_caches_before_device_write(self) -> None:
        solver = self._solver_with_published_caches()

        def invoke(observed: list[tuple[bool, bool]]) -> None:
            solver._mark_sphere_kernel = self._raising_mutation(solver, observed)
            with self.assertRaises(_SyntheticDeviceMutation):
                solver.mark_sphere_obstacle((0.5, 0.5, 0.5), 0.1)

        observed = self._assert_writer_invalidates_before_failed_device_mutation(
            solver=solver,
            invoke=invoke,
        )
        self.assertEqual(observed, [(False, False, 32, False, 0, 0)])
        self.assertEqual(solver.hibm_external_obstacle_topology_revision, 24)

    def test_mark_sphere_rejects_invalid_geometry_before_any_mutation(self) -> None:
        invalid_geometry = (
            (None, 0.1),
            ((0.5, 0.5), 0.1),
            ((0.5, 0.5, 0.5, 0.5), 0.1),
            ((0.5, float("nan"), 0.5), 0.1),
            ((0.5, float("inf"), 0.5), 0.1),
            ((0.5, 0.5, 0.5), float("nan")),
            ((0.5, 0.5, 0.5), float("inf")),
            ((0.5, 0.5, 0.5), 0.0),
            ((0.5, 0.5, 0.5), -0.1),
        )
        for center_m, radius_m in invalid_geometry:
            with self.subTest(center_m=center_m, radius_m=radius_m):
                solver = self._solver_with_published_caches()
                solver.reduction_count = _HostScalar(0)
                kernel_calls: list[object] = []
                solver._mark_sphere_kernel = lambda *_args: kernel_calls.append(
                    _args
                )

                with self.assertRaises(ValueError):
                    solver.mark_sphere_obstacle(center_m, radius_m)

                self.assertEqual(kernel_calls, [])
                self.assertEqual(
                    solver.hibm_external_obstacle_topology_revision,
                    23,
                )
                self.assertTrue(solver._pressure_outlet_nullspace_graph_valid)
                self.assertTrue(solver.last_hibm_reachability_valid)
                self.assertEqual(
                    solver.velocity_dirichlet_component_ledger_generation,
                    31,
                )
                self.assertTrue(solver.velocity_dirichlet_component_ledger_sealed)


if __name__ == "__main__":
    unittest.main()
