"""Host contracts for deferred HIBM closure; no Taichi runtime is initialized."""
from __future__ import annotations

import unittest
from collections.abc import Callable
from types import SimpleNamespace

from simulation_core.coupling.hibm_mpm import core as hibm_core
from simulation_core.fluids.solver import CartesianFluidSolver
from tests.solvers.test_canonical_velocity_boundary_authority import _host_only_solver


def _pending_solver() -> CartesianFluidSolver:
    solver = _host_only_solver(authority="canonical", generation=7, sealed=True)
    solver._hibm_marker_compatibility_closure_pending = True
    solver._velocity_dirichlet_component_ledger_consumer_generations = {
        name: 7 for name in solver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS
    }
    solver._velocity_dirichlet_component_ledger_consumer_capabilities = dict(
        solver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
    )
    return solver


def _forbidden_dispatch(
    solver: CartesianFluidSolver,
) -> tuple[Callable[..., None], list[tuple[bool, bool]]]:
    calls: list[tuple[bool, bool]] = []

    def dispatch(*_args: object, **_kwargs: object) -> None:
        calls.append((bool(solver._hibm_marker_compatibility_closure_pending),
                      bool(solver.velocity_dirichlet_component_ledger_sealed)))
        raise AssertionError(f"pending ledger reached a physical dispatch: calls={calls!r}")

    return dispatch, calls


class HibmPendingLedgerGuardTests(unittest.TestCase):
    """These calls must reject pending closure even with otherwise valid capabilities."""

    def test_pending_blocks_prepare_before_any_dispatch(self) -> None:
        solver = _pending_solver()
        dispatch, calls = _forbidden_dispatch(solver)
        solver._invalidate_hibm_pressure_reachability = dispatch
        with self.assertRaisesRegex(RuntimeError, "pending|compatibility|closure"):
            solver.prepare_and_seal_velocity_dirichlet_component_ledger()
        self.assertEqual(calls, [])
        self.assertTrue(solver._hibm_marker_compatibility_closure_pending)

    def test_pending_blocks_seal_before_any_dispatch(self) -> None:
        solver = _pending_solver()
        dispatch, calls = _forbidden_dispatch(solver)
        solver._refresh_canonical_exact_hard_face_component_mask = dispatch
        with self.assertRaisesRegex(RuntimeError, "pending|compatibility|closure"):
            solver.seal_velocity_dirichlet_component_ledger()
        self.assertEqual(calls, [])

    def test_pending_blocks_current_sealed_consumer_guard(self) -> None:
        solver = _pending_solver()
        with self.assertRaisesRegex(RuntimeError, "pending|compatibility|closure"):
            solver._require_velocity_dirichlet_component_ledger_sealed()

    def test_pending_blocks_public_apply_before_any_dispatch(self) -> None:
        solver = _pending_solver()
        dispatch, calls = _forbidden_dispatch(solver)
        solver._apply_velocity_dirichlet_boundary_rows_dispatch = dispatch
        with self.assertRaisesRegex(RuntimeError, "pending|compatibility|closure"):
            solver.apply_velocity_dirichlet_boundary_rows(read_report=False)
        self.assertEqual(calls, [])

    def test_pending_blocks_public_reference_before_any_dispatch(self) -> None:
        solver = _pending_solver()
        dispatch, calls = _forbidden_dispatch(solver)
        solver._capture_canonical_velocity_dirichlet_boundary_ledger_reference_kernel = dispatch
        with self.assertRaisesRegex(RuntimeError, "pending|compatibility|closure"):
            solver.capture_velocity_dirichlet_boundary_ledger_reference()
        self.assertEqual(calls, [])

    def test_pending_blocks_trial_save_before_any_dispatch(self) -> None:
        solver = _pending_solver()
        dispatch, calls = _forbidden_dispatch(solver)
        solver._save_state_kernel = dispatch
        with self.assertRaisesRegex(RuntimeError, "pending|compatibility|closure"):
            solver.save_state()
        self.assertEqual(calls, [])

    def test_invalidation_preserves_pending(self) -> None:
        solver = _pending_solver()
        generation = solver._invalidate_velocity_dirichlet_component_ledger()
        self.assertEqual(generation, 8)
        self.assertTrue(solver._hibm_marker_compatibility_closure_pending)
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)
        self.assertEqual(solver._velocity_dirichlet_component_ledger_consumer_generations, {})
        self.assertEqual(solver._velocity_dirichlet_component_ledger_consumer_capabilities, {})


    def test_authority_switch_does_not_bypass_pending_consumer_guard(self) -> None:
        solver = _pending_solver()
        solver._invalidate_hibm_pressure_reachability = lambda: None
        solver._invalidate_pressure_nullspace_component_graph = lambda: None
        solver.velocity_dirichlet_boundary_authority_code_device = {None: 1}
        solver.set_velocity_dirichlet_boundary_authority("legacy")
        self.assertTrue(solver._hibm_marker_compatibility_closure_pending)
        with self.assertRaisesRegex(RuntimeError, "pending|compatibility|closure"):
            solver._require_velocity_dirichlet_component_ledger_sealed()


def _restorable_solver() -> tuple[CartesianFluidSolver, list[tuple[str, bool]]]:
    solver = _pending_solver()
    events: list[tuple[str, bool]] = []

    def record(name: str) -> Callable[..., None]:
        def dispatch(*_args: object, **_kwargs: object) -> None:
            events.append((name, bool(solver._hibm_marker_compatibility_closure_pending)))
        return dispatch

    for name in (
        "_invalidate_hibm_pressure_reachability", "_invalidate_pressure_nullspace_component_graph",
        "_save_state_kernel", "_restore_state_kernel",
        "_clear_hibm_pressure_outlet_classification_kernel",
        "_clear_velocity_dirichlet_boundary_rows_kernel",
        "_reset_hibm_pressure_unreached_component_distribution_stats",
    ):
        setattr(solver, name, record(name))
    solver.hibm_external_obstacle_topology_revision = 12
    solver._saved_hibm_dynamic_solid_volume_enabled = True
    solver.hibm_dynamic_solid_volume_enabled = True
    solver._sst_saved_no_slip_domain_walls = ()
    solver._sst_saved_no_slip_domain_wall_mask = 0
    solver._sst_saved_wall_distance_valid = False
    solver._sst_saved_wall_distance_cache_key = None
    solver._sst_no_slip_domain_walls = ()
    solver._sst_wall_distance_valid = False
    solver._sst_wall_distance_cache_key = None
    solver.sst_no_slip_domain_wall_mask = {None: 0}
    return solver, events


class HibmPendingLedgerRollbackTests(unittest.TestCase):
    def test_trial_save_allows_unsealed_nonpending_state(self) -> None:
        solver, events = _restorable_solver()
        solver._hibm_marker_compatibility_closure_pending = False
        solver.velocity_dirichlet_component_ledger_sealed = False
        solver.save_state()
        self.assertEqual(events, [("_save_state_kernel", False)])
        self.assertTrue(solver._saved_hibm_dynamic_solid_volume_enabled)

    def test_full_clear_releases_pending_only_after_success(self) -> None:
        solver, events = _restorable_solver()
        solver.clear_velocity_dirichlet_boundary_rows()
        self.assertIn(("_clear_velocity_dirichlet_boundary_rows_kernel", True), events)
        self.assertTrue(all(pending for _, pending in events))
        self.assertFalse(solver._hibm_marker_compatibility_closure_pending)
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)

    def test_failed_full_clear_keeps_pending(self) -> None:
        solver, _ = _restorable_solver()
        error = RuntimeError("injected full row clear failure")

        def fail_clear() -> None:
            self.assertTrue(solver._hibm_marker_compatibility_closure_pending)
            raise error

        solver._clear_velocity_dirichlet_boundary_rows_kernel = fail_clear
        with self.assertRaises(RuntimeError) as caught:
            solver.clear_velocity_dirichlet_boundary_rows()
        self.assertIs(caught.exception, error)
        self.assertTrue(solver._hibm_marker_compatibility_closure_pending)
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)

    def test_successful_trial_restore_releases_pending_at_end(self) -> None:
        solver, events = _restorable_solver()
        solver.restore_state()
        self.assertIn(("_restore_state_kernel", True), events)
        self.assertIn(("_reset_hibm_pressure_unreached_component_distribution_stats", True), events)
        self.assertTrue(all(pending for _, pending in events))
        self.assertFalse(solver._hibm_marker_compatibility_closure_pending)
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)
        self.assertEqual(solver.hibm_external_obstacle_topology_revision, 13)

    def test_early_and_late_restore_failure_keep_pending(self) -> None:
        for entry in ("_restore_state_kernel", "_reset_hibm_pressure_unreached_component_distribution_stats"):
            with self.subTest(entry=entry):
                solver, _ = _restorable_solver()
                error = RuntimeError(f"injected restore failure: {entry}")

                def fail_restore() -> None:
                    self.assertTrue(solver._hibm_marker_compatibility_closure_pending)
                    raise error

                setattr(solver, entry, fail_restore)
                with self.assertRaises(RuntimeError) as caught:
                    solver.restore_state()
                self.assertIs(caught.exception, error)
                self.assertTrue(solver._hibm_marker_compatibility_closure_pending)
                self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)


def _band_fixture(increments: list[int]):
    remaining = iter(increments)
    events: list[tuple[str, object]] = []
    band_calls: list[dict[str, object]] = []
    state = {"pending": False, "last_increment": None}
    final_report = {"closed": True}
    search = SimpleNamespace(node_kind_code=object())

    def assemble(*, enable_marker_compatibility_closure: bool = True) -> dict[str, object]:
        events.append(("assemble", enable_marker_compatibility_closure))
        if enable_marker_compatibility_closure:
            if state["last_increment"] != 0:
                raise AssertionError("marker closure ran before band saturation")
            state["pending"] = False
            return final_report
        state["pending"] = True
        return {"geometry_only": True}

    def mark_band(**kwargs: object) -> int:
        if not state["pending"]:
            raise AssertionError("band sweep lacked current geometry-only claims")
        increment = next(remaining)
        state["last_increment"] = increment
        events.append(("band", increment))
        band_calls.append(kwargs)
        return increment

    fluid = SimpleNamespace(mark_hibm_solid_band_nonprojectable_cells=mark_band)
    arguments = dict(fluid=fluid, ib_search=search, assemble_velocity_component_face_ledger=assemble,
                     pressure_outlet_zmin=True)
    return arguments, events, band_calls, final_report


class HibmSolidBandOrderingTests(unittest.TestCase):
    def test_stable_band_closes_after_zero_sweep(self) -> None:
        arguments, events, calls, final_report = _band_fixture([0])
        count, report = hibm_core._stabilize_hibm_solid_band(**arguments, max_passes=8)
        self.assertEqual(count, 0)
        self.assertIs(report, final_report)
        self.assertEqual(events, [("assemble", False), ("band", 0), ("assemble", True)])
        self.assertEqual(len(calls), 1)

    def test_positive_sweeps_rebuild_geometry_without_closing(self) -> None:
        arguments, events, calls, final_report = _band_fixture([2, 1, 0])
        count, report = hibm_core._stabilize_hibm_solid_band(**arguments, max_passes=8)
        self.assertEqual(count, 3)
        self.assertIs(report, final_report)
        self.assertEqual(events, [
            ("assemble", False), ("band", 2), ("assemble", False),
            ("band", 1), ("assemble", False), ("band", 0), ("assemble", True),
        ])
        self.assertEqual(len(calls), 3)

    def test_saturation_on_last_allowed_pass_preserves_cap8_and_cap9(self) -> None:
        for cap in (8, 9):
            with self.subTest(cap=cap):
                arguments, events, calls, final_report = _band_fixture([1] * (cap - 1) + [0])
                count, report = hibm_core._stabilize_hibm_solid_band(**arguments, max_passes=cap)
                self.assertEqual(count, cap - 1)
                self.assertIs(report, final_report)
                self.assertEqual(len(calls), cap)
                self.assertEqual(events[-2:], [("band", 0), ("assemble", True)])
                self.assertEqual(events.count(("assemble", True)), 1)

    def test_cap_exhaustion_rejects_before_any_target_closure(self) -> None:
        for cap in (8, 9):
            with self.subTest(cap=cap):
                arguments, events, calls, _ = _band_fixture([1] * cap)
                with self.assertRaisesRegex(RuntimeError, f"max_passes={cap}.*last_increment=1"):
                    hibm_core._stabilize_hibm_solid_band(**arguments, max_passes=cap)
                self.assertEqual(len(calls), cap)
                self.assertNotIn(("assemble", True), events)

    def test_band_policy_arguments_remain_identical(self) -> None:
        for outlet in (False, True):
            with self.subTest(outlet=outlet):
                arguments, _, calls, _ = _band_fixture([1, 0])
                arguments["pressure_outlet_zmin"] = outlet
                hibm_core._stabilize_hibm_solid_band(**arguments, max_passes=8)
                expected = dict(
                    pressure_outlet_zmin=outlet,
                    node_kind_code=arguments["ib_search"].node_kind_code,
                    unclassified_node_code=hibm_core.HibmMpmIbNodeSearch._NODE_NONE,
                    protect_velocity_dirichlet_radius_cells=0,
                    protect_unstamped_velocity_dirichlet_components=True,
                    protect_solid_band_mask=True,
                )
                self.assertEqual(calls, [expected, expected])

    def test_nonpositive_cap_rejects_before_assembly(self) -> None:
        for cap in (0, -1):
            with self.subTest(cap=cap):
                arguments, events, calls, _ = _band_fixture([])
                with self.assertRaises(ValueError):
                    hibm_core._stabilize_hibm_solid_band(**arguments, max_passes=cap)
                self.assertEqual(events, [])
                self.assertEqual(calls, [])


def _assembly_fixture():
    solver = _pending_solver()
    solver.grid = SimpleNamespace(grid_nodes=(4, 4, 4))
    solver.rho = 1000.0
    solver.obstacle = object()
    solver.velocity = object()
    for name in (
        "active_component_mask", "value_mps", "pressure_mobility", "component_enforcement_weight",
        "component_region_id", "hard_fixed_component_mask", "external_exact_component_mask",
        "owned_component_mask",
    ):
        setattr(solver, f"velocity_dirichlet_boundary_{name}", object())
    for kind in ("face", "center", "width"):
        for axis in "xyz":
            setattr(solver, f"cell_{kind}_{axis}_m", object())
    events: list[str] = []
    calls: list[dict[str, object]] = []

    def assemble(**kwargs: object) -> dict[str, object]:
        if not solver._hibm_marker_compatibility_closure_pending or solver.velocity_dirichlet_component_ledger_sealed:
            raise AssertionError("assembly started with a reusable ledger")
        events.append("assemble")
        calls.append(kwargs)
        return {"assembly_result": "current"}

    def refresh_pressure_hard_mask() -> tuple[int, int]:
        if not solver._hibm_marker_compatibility_closure_pending or solver.velocity_dirichlet_component_ledger_sealed:
            raise AssertionError("geometry refresh escaped its pending interval")
        events.append("raw_pressure_hard_refresh")
        return 0, 0

    def prepare_and_seal() -> None:
        if solver._hibm_marker_compatibility_closure_pending:
            raise AssertionError("full preparation observed pending closure")
        events.append("prepare_and_seal")
        generation = solver.velocity_dirichlet_component_ledger_generation
        solver._velocity_dirichlet_component_ledger_consumer_generations = {
            name: generation for name in solver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS
        }
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = dict(
            solver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )
        solver.velocity_dirichlet_component_ledger_sealed = True

    solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask = refresh_pressure_hard_mask
    solver.prepare_and_seal_velocity_dirichlet_component_ledger = prepare_and_seal
    arguments = dict(
        fluid=solver, markers=SimpleNamespace(region_id=object()), ib_search=object(),
        ib_boundary=SimpleNamespace(assemble_velocity_dirichlet_component_face_ledger=assemble),
        surface_projection_inactive_axis=0, primary_region_id=101, secondary_region_id=202,
        interpolate_interior_velocity=False, marker_mac_constraint_operator=object(),
        marker_mac_constraint_absolute_tolerance_mps=1e-4,
    )
    return solver, arguments, events, calls


class HibmDeferredAssemblyTests(unittest.TestCase):
    def test_closure_flag_and_physical_arguments_propagate_without_threshold_change(self) -> None:
        for close in (False, True):
            with self.subTest(close=close):
                solver, arguments, events, calls = _assembly_fixture()
                report = hibm_core._assemble_hibm_velocity_component_face_ledger(
                    **arguments, enable_marker_compatibility_closure=close)
                self.assertEqual(events, ["assemble", "prepare_and_seal" if close else "raw_pressure_hard_refresh"])
                self.assertEqual(solver._hibm_marker_compatibility_closure_pending, not close)
                self.assertEqual(report["authority_sealed"], close)
                self.assertEqual(report["authority_registered"], close)
                self.assertEqual(report["ledger_generation"], 8)
                self.assertEqual(len(calls), 1)
                call = calls[0]
                self.assertIs(call["enable_marker_compatibility_closure"], close)
                self.assertEqual(call["marker_compatibility_absolute_tolerance_mps"], 1e-4)
                self.assertEqual(call["marker_compatibility_density_kgm3"], 1000.0)
                self.assertEqual((call["primary_region_id"], call["secondary_region_id"]), (101, 202))
                self.assertIs(call["markers"], arguments["markers"])
                self.assertIs(call["marker_mac_constraint_operator"], arguments["marker_mac_constraint_operator"])
                self.assertIs(call["obstacle_field"], solver.obstacle)
                self.assertIs(call["velocity_field"], solver.velocity)
                for axis in "xyz":
                    self.assertIs(call[f"cell_width_{axis}_m"], getattr(solver, f"cell_width_{axis}_m"))

    def test_assembly_failure_keeps_pending_and_original_exception(self) -> None:
        for close in (False, True):
            with self.subTest(close=close):
                solver, arguments, events, _ = _assembly_fixture()
                error = RuntimeError("injected geometry or closure rejection")

                def fail_assembly(**_kwargs: object) -> dict[str, object]:
                    self.assertTrue(solver._hibm_marker_compatibility_closure_pending)
                    self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)
                    raise error

                arguments["ib_boundary"].assemble_velocity_dirichlet_component_face_ledger = fail_assembly
                with self.assertRaises(RuntimeError) as caught:
                    hibm_core._assemble_hibm_velocity_component_face_ledger(
                        **arguments, enable_marker_compatibility_closure=close)
                self.assertIs(caught.exception, error)
                self.assertTrue(solver._hibm_marker_compatibility_closure_pending)
                self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)
                self.assertEqual(events, [])

    def test_missing_operator_tolerance_rejects_before_invalidating_valid_rows(self) -> None:
        solver, arguments, events, _ = _assembly_fixture()
        solver._hibm_marker_compatibility_closure_pending = False
        arguments["marker_mac_constraint_absolute_tolerance_mps"] = None
        with self.assertRaisesRegex(ValueError, "tolerance.*required"):
            hibm_core._assemble_hibm_velocity_component_face_ledger(**arguments)
        self.assertEqual(solver.velocity_dirichlet_component_ledger_generation, 7)
        self.assertTrue(solver.velocity_dirichlet_component_ledger_sealed)
        self.assertFalse(solver._hibm_marker_compatibility_closure_pending)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
