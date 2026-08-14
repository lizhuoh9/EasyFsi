from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.official import solid_mpm_fsi_runner
from benchmarks.official.solid_mpm_fsi_runner import (
    _combine_flow_projection_reports,
)
from simulation_core.fluids.solver import (
    CartesianFluidSolver,
    _run_pre_projection_velocity_transaction,
)


ROOT = Path(__file__).resolve().parents[2]
SOLVER_PATH = ROOT / "simulation_core" / "fluids" / "solver.py"
RUNNER_PATH = ROOT / "benchmarks" / "official" / "solid_mpm_fsi_runner.py"

PROJECTOR_ARGUMENT = "pre_projection_velocity_projector"
TRANSACTION_HELPER = "_run_pre_projection_velocity_transaction"
PREPARE_METHOD = "prepare_projection_transaction"
SOLVE_METHOD = "solve_projection_transaction"
COMMIT_METHOD = "commit_projection_transaction"
PROJECTOR_CACHE_KEY = "pre_projection_velocity_projector"

Q_REPORT_KEYS = (
    "pre_projection_velocity_projector_prepared",
    "pre_projection_velocity_projector_converged",
    "pre_projection_velocity_projector_committed",
)
Q_AGGREGATE_KEYS = tuple(f"{key}_all" for key in Q_REPORT_KEYS)
JOINT_QP_REPORT_KEYS = (
    "hibm_joint_qp_measured",
    "hibm_joint_qp_converged",
    "hibm_joint_qp_cycle_budget",
    "hibm_joint_qp_cycles_used",
    "hibm_joint_qp_terminal_no_slip_vector_max_residual_mps",
    "hibm_joint_qp_terminal_divergence_l2_s_inv",
    "hibm_joint_qp_terminal_divergence_max_abs_s_inv",
    "hibm_joint_qp_pressure_exact_relative_residual",
    "hibm_joint_qp_pressure_reintroduced_no_slip_mps",
    "hibm_joint_qp_final_operation",
    "hibm_joint_qp_cycle_trace",
)


class _RecordingPreProjectionVelocityProjector:
    def __init__(self, diagnostics: object) -> None:
        self.diagnostics = diagnostics
        self.calls: tuple[object, ...] = ()

    def prepare_projection_transaction(
        self,
        *,
        fluid,
        pressure_solve_context,
    ) -> None:
        self.calls = (
            *self.calls,
            ("prepare", fluid, pressure_solve_context),
        )

    def solve_projection_transaction(self) -> None:
        self.calls = (*self.calls, "solve")

    def commit_projection_transaction(self):
        self.calls = (*self.calls, "commit")
        return self.diagnostics


def _runtime_projection_stage(
    pressure_solve_context: object,
) -> str:
    context = dict(pressure_solve_context)
    if context.get("projection_stage") == "post_dirichlet_consistency":
        return f"consistency_{int(context['consistency_projection_index'])}"
    return "main"


def _healthy_pressure_projection_report(
    *,
    exact_relative_residual: float = 5.0e-7,
    divergence_l2: float = 2.0e-8,
    divergence_max_abs: float = 4.0e-8,
) -> dict[str, object]:
    return {
        "cg_converged_all": True,
        "pressure_solve_failed": False,
        "pressure_projection_physical_failure": False,
        "l2": float(divergence_l2),
        "max_abs": float(divergence_max_abs),
        "cg_exact_relative_residual_max": float(exact_relative_residual),
    }


def _terminal_no_slip_report(vector_max_residual_mps: float) -> dict[str, object]:
    residual = float(vector_max_residual_mps)
    return {
        "hibm_no_slip_report": {},
        "hibm_no_slip_valid_marker_count": 1,
        "hibm_no_slip_invalid_marker_count": 0,
        "hibm_no_slip_max_residual_mps": residual,
        "hibm_no_slip_l2_residual_mps": residual,
        "hibm_no_slip_residual_argmax_residual_vector_mps": (
            residual,
            0.0,
            0.0,
        ),
        "hibm_no_slip_residual_argmax_marker_index": 7,
        "hibm_no_slip_residual_argmax_marker_region_id": 3,
        "hibm_no_slip_residual_argmax_sample_source": "normal_walk",
        "hibm_no_slip_residual_argmax_sample_position_m": (0.1, 0.2, 0.3),
        "hibm_no_slip_residual_argmax_fluid_velocity_mps": (1.0, 2.0, 3.0),
        "hibm_no_slip_direct_sample_marker_count": 0,
        "hibm_no_slip_normal_walk_sample_marker_count": 1,
        "hibm_no_slip_nearest_fluid_sample_marker_count": 0,
        "hibm_no_slip_no_fluid_sample_marker_count": 0,
        "hibm_no_slip_sampling_identity_generation": 23,
        "hibm_no_slip_topology_generation": 31,
        "hibm_no_slip_component_face_valid_mask_generation": 47,
    }


class _FlowAdvanceQpTrace:
    def __init__(
        self,
        *,
        failing_q_index: int | None = None,
        failure_mode: str | None = None,
    ) -> None:
        self.failing_q_index = failing_q_index
        self.failure_mode = failure_mode
        self.events: tuple[tuple[str, str], ...] = ()
        self.pressure_solve_contexts: tuple[dict[str, object], ...] = ()
        self.current_q_index = -1
        self.current_stage = ""

    def begin_q(self, stage: str) -> None:
        self.current_q_index += 1
        self.current_stage = str(stage)
        self.record("Q.prepare", stage)

    def record(self, phase: str, stage: str) -> None:
        self.events = (*self.events, (str(phase), str(stage)))

    def fails_current_q(self, mode: str) -> bool:
        return (
            self.current_q_index == self.failing_q_index
            and self.failure_mode == mode
        )


class _FlowAdvanceRuntimeProjector:
    def __init__(self, trace: _FlowAdvanceQpTrace) -> None:
        self.trace = trace

    def prepare_projection_transaction(self, *, fluid, pressure_solve_context) -> None:
        del fluid
        self.trace.begin_q(_runtime_projection_stage(pressure_solve_context))

    def solve_projection_transaction(self) -> None:
        self.trace.record("Q.solve", self.trace.current_stage)
        if self.trace.fails_current_q("raise"):
            raise RuntimeError("injected Q failure before pressure")

    def commit_projection_transaction(self) -> dict[str, object]:
        self.trace.record("Q.commit", self.trace.current_stage)
        return {
            "prepared": True,
            "converged": not self.trace.fails_current_q("false"),
            "committed": True,
            "iterations": 6,
            "max_residual_mps": 2.5e-5,
            "sample_identity_generation": 23,
            "topology_generation": 31,
            "component_face_valid_mask_generation": 47,
        }


class _FlowAdvanceRuntimeFluid:
    def __init__(
        self,
        trace: _FlowAdvanceQpTrace,
        pressure_reports: tuple[dict[str, object], ...],
    ) -> None:
        self.trace = trace
        self.pressure_reports = tuple(dict(report) for report in pressure_reports)

    def clear_volume_source(self) -> None:
        return None

    def project(
        self,
        *,
        pressure_solve_context,
        pre_projection_velocity_projector,
        **_kwargs,
    ) -> dict[str, object]:
        self.trace.pressure_solve_contexts = (
            *self.trace.pressure_solve_contexts,
            dict(pressure_solve_context),
        )
        diagnostics = _run_pre_projection_velocity_transaction(
            pre_projection_velocity_projector=pre_projection_velocity_projector,
            fluid=self,
            pressure_solve_context=pressure_solve_context,
        )
        self.trace.record(
            "P",
            _runtime_projection_stage(pressure_solve_context),
        )
        pressure_report = self.pressure_reports[self.trace.current_q_index]
        return {**pressure_report, **diagnostics}

    def pressure_outlet_fv_flux_report(
        self,
        *,
        dt_s: float,
    ) -> dict[str, object]:
        del dt_s
        return {}

    def snapshot_pressure(self, *, preserve_if_current_is_zero: bool) -> bool:
        del preserve_if_current_is_zero
        return True


class _PressureProjectionReached(RuntimeError):
    """Test sentinel raised at the first real project() pressure operation."""


def _run_real_cartesian_project_to_first_pressure(
    trace: _FlowAdvanceQpTrace,
) -> None:
    """Execute production project() through Q and stop at its first P operation.

    The instance intentionally skips the Taichi-heavy constructor.  Only the
    setup kernels that precede the Q transaction are replaced; the ordering and
    fail-closed decision remain the real CartesianFluidSolver.project method.
    """

    fluid = object.__new__(CartesianFluidSolver)
    fluid.grid = SimpleNamespace(is_uniform=True)
    fluid.rho = 1.0
    fluid.dx = 1.0
    fluid.dy = 1.0
    fluid.dz = 1.0
    fluid._pressure_warmstart_slots = ()
    # project() now performs an unconditional interface-operator preflight
    # before Q.  Model the valid empty-row state explicitly so this lifecycle
    # test keeps exercising the production host-side preflight instead of
    # bypassing it with a mocked method.
    fluid.pressure_interface_row_count = {None: 0}
    fluid.pressure_interface_row_capacity = 1
    fluid.pressure_interface_preflight_invalid_counts = {
        None: np.zeros(8, dtype=np.int32)
    }
    projector = _FlowAdvanceRuntimeProjector(trace)

    def no_op(_fluid, *_args, **_kwargs) -> None:
        return None

    def zero(_fluid, *_args, **_kwargs) -> int:
        return 0

    def empty_hard_fixed_counts(_fluid, *_args, **_kwargs) -> tuple[int, int]:
        return (0, 0)

    def first_pressure_operation(_fluid, *_args, **_kwargs) -> None:
        trace.record("P", "main")
        raise _PressureProjectionReached("first pressure operation reached")

    pressure_interface_report = {
        "active_cells": 0,
        "row_count": 0,
        "row_active_count": 0,
        "max_abs_diagonal": 0.0,
    }

    def pressure_interface_policy(_fluid, *_args, **_kwargs) -> dict[str, object]:
        return pressure_interface_report

    with (
        patch.object(
            CartesianFluidSolver,
            "_require_velocity_dirichlet_component_ledger_sealed",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_velocity_dirichlet_boundary_authority_code",
            new=zero,
        ),
        patch.object(
            CartesianFluidSolver,
            "_resolve_velocity_inlet_zmax_topology_mode",
            new=zero,
        ),
        patch.object(
            CartesianFluidSolver,
            "_require_face_symmetric_owned_soft_compatibility",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_refresh_velocity_dirichlet_pressure_hard_fixed_component_mask",
            new=empty_hard_fixed_counts,
        ),
        patch.object(
            CartesianFluidSolver,
            "_pressure_interface_matrix_policy_report",
            new=pressure_interface_policy,
        ),
        patch.object(
            CartesianFluidSolver,
            "_validate_pressure_interface_operator_storage_kernel",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_clear_pressure_interface_preflight_incident_diagonal_kernel",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_accumulate_pressure_interface_preflight_incident_diagonal_kernel",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_validate_pressure_interface_local_diagonal_provenance_kernel",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_reset_zmin_projection_flux_report_kernel",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_clear_pressure_interface_projection_divergence_kernel",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_apply_velocity_dirichlet_boundary_rows_dispatch",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "_apply_obstacle_no_normal_flow_kernel",
            new=no_op,
        ),
        patch.object(
            CartesianFluidSolver,
            "obstacle_cell_count",
            new=zero,
        ),
        patch.object(
            CartesianFluidSolver,
            "_compute_divergence_with_topology_mode",
            new=first_pressure_operation,
        ),
    ):
        CartesianFluidSolver.project(
            fluid,
            iterations=1,
            pressure_outlet_zmin=False,
            dt_s=5.0e-4,
            pressure_solver="fv_cg",
            pressure_solve_context={"projection_stage": "main"},
            pre_projection_velocity_projector=projector,
            read_report=False,
        )


def _run_runtime_flow_advance(
    trace: _FlowAdvanceQpTrace,
    *,
    consistency_projection_count: int,
    terminal_no_slip_residuals_mps: tuple[float, ...] | None = None,
    pressure_reports: tuple[dict[str, object], ...] | None = None,
    preflow_history: list[dict[str, object]] | None = None,
    sharp_boundary: bool = True,
    reprojection_cg_tolerance: float | None = None,
    sharp_boundary_stage_times: tuple[dict[str, float], ...] | None = None,
) -> dict[str, object]:
    cycle_budget = 1 + int(consistency_projection_count)
    if terminal_no_slip_residuals_mps is None:
        terminal_no_slip_residuals_mps = (
            *(2.0e-4 for _ in range(cycle_budget - 1)),
            5.0e-5,
        )
    if pressure_reports is None:
        pressure_reports = tuple(
            _healthy_pressure_projection_report() for _ in range(cycle_budget)
        )
    if len(terminal_no_slip_residuals_mps) != cycle_budget:
        raise ValueError("terminal no-slip fixture count must equal the cycle budget")
    if len(pressure_reports) != cycle_budget:
        raise ValueError("pressure report fixture count must equal the cycle budget")
    if sharp_boundary_stage_times is None:
        sharp_boundary_stage_times = tuple({} for _ in range(cycle_budget))
    if len(sharp_boundary_stage_times) != cycle_budget:
        raise ValueError("sharp-boundary timing fixture count must equal the budget")
    projector = _FlowAdvanceRuntimeProjector(trace)
    fluid = _FlowAdvanceRuntimeFluid(trace, pressure_reports)
    config = SimpleNamespace(
        dt_s=5.0e-4,
        flow_driver_mode=solid_mpm_fsi_runner.FLOW_DRIVER_PROJECTION_ONLY,
        flow_solid_boundary_mode=(
            solid_mpm_fsi_runner.FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS
            if sharp_boundary
            else solid_mpm_fsi_runner.FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS
        ),
        flow_hibm_sharp_interpolate_velocity_rows=True,
        flow_hibm_marker_mac_constraint_absolute_tolerance_mps=1.0e-4,
        flow_post_dirichlet_consistency_projection_iterations=(
            int(consistency_projection_count)
        ),
        flow_projection_iterations=1,
        flow_pressure_solver="fv_cg",
        flow_cg_tolerance=1.0e-6,
        flow_reprojection_cg_tolerance=reprojection_cg_tolerance,
        flow_divergence_cleanup_iterations=0,
        flow_pressure_outlet_enabled=True,
    )
    boundary_reports = iter(
        (
            {"hibm_sharp_marker_boundary_enabled": bool(sharp_boundary)},
            *(
                {
                    "hibm_sharp_marker_boundary_enabled": bool(sharp_boundary),
                    "hibm_sharp_marker_boundary_stage_wall_time_s": dict(
                        stage_times
                    ),
                }
                for stage_times in sharp_boundary_stage_times
            ),
        )
    )
    terminal_reports = iter(
        _terminal_no_slip_report(value)
        for value in terminal_no_slip_residuals_mps
    )
    history_owner = [] if preflow_history is None else preflow_history

    with (
        patch.object(
            solid_mpm_fsi_runner,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
            side_effect=lambda *_args, **_kwargs: next(boundary_reports),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_require_hibm_velocity_dirichlet_health",
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_hibm_pre_projection_velocity_projector_from_cache",
            return_value=projector,
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_sample_hibm_no_slip_report",
            side_effect=lambda *_args, **_kwargs: next(terminal_reports),
        ),
        patch.object(
            solid_mpm_fsi_runner,
            "_flow_state_report",
            side_effect=lambda _fluid, projection_report, **_kwargs: {
                "projection_report": dict(projection_report),
            },
        ),
    ):
        return solid_mpm_fsi_runner._flow_advance_current_step(
            fluid,
            config,
            markers=object(),
            sharp_boundary_cache={},
            flow_phase="preflow",
            step_index_local=0,
            step_index_global=0,
            preflow_history=history_owner,
            reset_pressure=True,
        )


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_function(path: Path, name: str) -> ast.FunctionDef:
    for statement in _module(path).body:
        if isinstance(statement, ast.FunctionDef) and statement.name == name:
            return statement
    raise AssertionError(f"missing function {name!r} in {path}")


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for statement in _module(path).body:
        if not isinstance(statement, ast.ClassDef) or statement.name != class_name:
            continue
        for member in statement.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                return member
    raise AssertionError(f"missing method {class_name}.{method_name} in {path}")


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _call_name(candidate) == name
    ]


def _keyword_names(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


def _literal_strings(node: ast.AST) -> set[str]:
    return {
        candidate.value
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    }


def _argument_default(function: ast.FunctionDef, argument_name: str) -> ast.AST:
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_default_offset = len(positional) - len(function.args.defaults)
    for index, argument in enumerate(positional):
        if argument.arg == argument_name:
            if index < positional_default_offset:
                raise AssertionError(f"argument {argument_name!r} has no default")
            return function.args.defaults[index - positional_default_offset]
    for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        if argument.arg == argument_name:
            if default is None:
                raise AssertionError(f"argument {argument_name!r} has no default")
            return default
    raise AssertionError(f"missing argument {argument_name!r}")


def _single_call(function: ast.FunctionDef, name: str) -> ast.Call:
    calls = _calls(function, name)
    if len(calls) != 1:
        raise AssertionError(
            f"expected one {name} call in {function.name}, found {len(calls)}"
        )
    return calls[0]


def _ancestor_statement(function: ast.FunctionDef, target: ast.AST) -> ast.stmt:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(function):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    current: ast.AST = target
    while current in parents and not isinstance(current, ast.stmt):
        current = parents[current]
    if not isinstance(current, ast.stmt):
        raise AssertionError("call has no containing statement")
    return current


class GenericFluidProjectionHookContracts(unittest.TestCase):
    def test_transaction_helper_accepts_only_true_boolean_health_statuses(self) -> None:
        healthy_statuses = (
            {"prepared": True, "converged": True, "committed": True},
            {
                "prepared": np.bool_(True),
                "converged": np.bool_(True),
                "committed": np.bool_(True),
            },
        )
        for statuses in healthy_statuses:
            with self.subTest(statuses=statuses):
                projector = _RecordingPreProjectionVelocityProjector(statuses)
                diagnostics = _run_pre_projection_velocity_transaction(
                    pre_projection_velocity_projector=projector,
                    fluid="fluid-owner",
                    pressure_solve_context={"stage": "main"},
                )
                self.assertTrue(
                    diagnostics["pre_projection_velocity_projector_prepared"]
                )
                self.assertTrue(
                    diagnostics["pre_projection_velocity_projector_converged"]
                )
                self.assertTrue(
                    diagnostics["pre_projection_velocity_projector_committed"]
                )
                self.assertEqual(
                    projector.calls,
                    (
                        ("prepare", "fluid-owner", {"stage": "main"}),
                        "solve",
                        "commit",
                    ),
                )

        unhealthy_statuses = (
            {"prepared": False, "converged": True, "committed": True},
            {"prepared": True, "converged": False, "committed": True},
            {"prepared": True, "converged": True, "committed": False},
            {"prepared": "false", "converged": True, "committed": True},
            {"prepared": 1, "converged": True, "committed": True},
        )
        for statuses in unhealthy_statuses:
            with self.subTest(statuses=statuses):
                projector = _RecordingPreProjectionVelocityProjector(statuses)
                with self.assertRaisesRegex(
                    (TypeError, RuntimeError),
                    "pre-projection velocity transaction",
                ):
                    _run_pre_projection_velocity_transaction(
                        pre_projection_velocity_projector=projector,
                        fluid="fluid-owner",
                        pressure_solve_context={"stage": "main"},
                    )
                self.assertEqual(projector.calls[-2:], ("solve", "commit"))

    def test_project_exposes_optional_generic_projector_without_hibm_import(self) -> None:
        project = _class_method(
            SOLVER_PATH,
            "CartesianFluidSolver",
            "project",
        )
        default = _argument_default(project, PROJECTOR_ARGUMENT)

        self.assertIsInstance(default, ast.Constant)
        self.assertIsNone(default.value)
        solver_source = SOLVER_PATH.read_text(encoding="utf-8").lower()
        self.assertNotIn(
            "from simulation_core.coupling.hibm",
            solver_source,
            "the fluid solver hook must remain case- and HIBM-agnostic",
        )

    def test_transaction_helper_is_three_phase_fail_closed_and_noop_for_none(self) -> None:
        helper = _top_level_function(SOLVER_PATH, TRANSACTION_HELPER)
        calls = {
            name: _single_call(helper, name)
            for name in (PREPARE_METHOD, SOLVE_METHOD, COMMIT_METHOD)
        }

        self.assertLess(calls[PREPARE_METHOD].lineno, calls[SOLVE_METHOD].lineno)
        self.assertLess(calls[SOLVE_METHOD].lineno, calls[COMMIT_METHOD].lineno)
        self.assertFalse(
            any(
                isinstance(
                    node,
                    (ast.Try, getattr(ast, "TryStar", ast.Try)),
                )
                for node in ast.walk(helper)
            ),
            "transaction exceptions must propagate; pressure projection must not start",
        )
        none_guards = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.If)
            and PROJECTOR_ARGUMENT in {
                candidate.id
                for candidate in ast.walk(node.test)
                if isinstance(candidate, ast.Name)
            }
            and any(
                isinstance(candidate, ast.Constant) and candidate.value is None
                for candidate in ast.walk(node.test)
            )
        ]
        self.assertTrue(none_guards, "None must be an explicit no-op path")
        helper_strings = _literal_strings(helper)
        for key in Q_REPORT_KEYS:
            with self.subTest(report_key=key):
                self.assertIn(key, helper_strings)

    def test_project_runs_fresh_q_after_topology_and_before_first_p_divergence(self) -> None:
        project = _class_method(SOLVER_PATH, "CartesianFluidSolver", "project")
        transaction = _single_call(project, TRANSACTION_HELPER)
        transaction_statement = _ancestor_statement(project, transaction)
        first_boundary_clamp = min(
            call.lineno for call in _calls(project, "apply_velocity_boundary_conditions")
        )
        topology_cleanup = max(
            call.lineno
            for name in (
                "convert_hibm_row_cloud_orphan_components",
                "cleanup_hibm_pressure_outlet_tiny_unreached_components",
            )
            for call in _calls(project, name)
        )
        hard_mask_refresh = max(
            call.lineno
            for call in _calls(
                project,
                "_refresh_velocity_dirichlet_pressure_hard_fixed_component_mask",
            )
        )
        pressure_component_prep = _single_call(
            project,
            "_prepare_closed_neumann_pressure_components",
        ).lineno
        first_divergence = min(
            call.lineno for call in _calls(project, "compute_current_divergence")
        )

        self.assertGreater(transaction_statement.lineno, first_boundary_clamp)
        self.assertGreater(transaction_statement.lineno, topology_cleanup)
        self.assertGreater(transaction_statement.lineno, hard_mask_refresh)
        self.assertGreater(transaction_statement.lineno, pressure_component_prep)
        self.assertLess(transaction_statement.lineno, first_divergence)
        self.assertIn(PROJECTOR_ARGUMENT, _keyword_names(transaction))
        self.assertIn("fluid", _keyword_names(transaction))
        self.assertIn("pressure_solve_context", _keyword_names(transaction))

    def test_project_merges_q_diagnostics_into_the_pressure_report(self) -> None:
        project = _class_method(SOLVER_PATH, "CartesianFluidSolver", "project")
        project_strings = _literal_strings(project)
        for key in Q_REPORT_KEYS:
            with self.subTest(report_key=key):
                self.assertIn(key, project_strings)


class GenericRunnerQpLifecycleContracts(unittest.TestCase):
    def test_real_cartesian_project_runtime_orders_q_before_p_and_fails_closed(
        self,
    ) -> None:
        healthy_trace = _FlowAdvanceQpTrace()
        with self.assertRaisesRegex(
            _PressureProjectionReached,
            "first pressure operation reached",
        ):
            _run_real_cartesian_project_to_first_pressure(healthy_trace)
        self.assertEqual(
            healthy_trace.events,
            (
                ("Q.prepare", "main"),
                ("Q.solve", "main"),
                ("Q.commit", "main"),
                ("P", "main"),
            ),
        )

        expected_failure_events = {
            "false": (
                ("Q.prepare", "main"),
                ("Q.solve", "main"),
                ("Q.commit", "main"),
            ),
            "raise": (
                ("Q.prepare", "main"),
                ("Q.solve", "main"),
            ),
        }
        for failure_mode in ("false", "raise"):
            with self.subTest(failure_mode=failure_mode):
                failing_trace = _FlowAdvanceQpTrace(
                    failing_q_index=0,
                    failure_mode=failure_mode,
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "pre-projection velocity transaction|injected Q failure",
                ):
                    _run_real_cartesian_project_to_first_pressure(failing_trace)
                self.assertEqual(
                    failing_trace.events,
                    expected_failure_events[failure_mode],
                )
                self.assertNotIn(("P", "main"), failing_trace.events)

    def test_runtime_flow_advance_orders_every_q_before_p_and_fails_closed(self) -> None:
        consistency_projection_count = 3
        stages = (
            "main",
            *(
                f"consistency_{index}"
                for index in range(1, consistency_projection_count + 1)
            ),
        )
        healthy_trace = _FlowAdvanceQpTrace()
        report = _run_runtime_flow_advance(
            healthy_trace,
            consistency_projection_count=consistency_projection_count,
        )

        expected_events = tuple(
            (phase, stage)
            for stage in stages
            for phase in ("Q.prepare", "Q.solve", "Q.commit", "P")
        )
        self.assertEqual(healthy_trace.events, expected_events)
        self.assertEqual(healthy_trace.events[-1], ("P", stages[-1]))
        for aggregate_key in Q_AGGREGATE_KEYS:
            with self.subTest(aggregate_key=aggregate_key):
                self.assertTrue(report["projection_report"][aggregate_key])

        for failure_mode in ("false", "raise"):
            for failing_q_index, failing_stage in enumerate(stages):
                with self.subTest(
                    failure_mode=failure_mode,
                    failing_stage=failing_stage,
                ):
                    failing_trace = _FlowAdvanceQpTrace(
                        failing_q_index=failing_q_index,
                        failure_mode=failure_mode,
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "pre-projection velocity transaction|injected Q failure",
                    ):
                        _run_runtime_flow_advance(
                            failing_trace,
                            consistency_projection_count=(
                                consistency_projection_count
                            ),
                        )

                    pressure_stages = tuple(
                        stage
                        for phase, stage in failing_trace.events
                        if phase == "P"
                    )
                    self.assertEqual(
                        pressure_stages,
                        stages[:failing_q_index],
                    )
                    self.assertNotIn(("P", failing_stage), failing_trace.events)
                    self.assertEqual(
                        {
                            stage
                            for _phase, stage in failing_trace.events
                        },
                        set(stages[: failing_q_index + 1]),
                    )

    def test_joint_qp_stops_only_after_terminal_q_and_pressure_contracts_pass(
        self,
    ) -> None:
        trace = _FlowAdvanceQpTrace()
        pressure_not_converged = _healthy_pressure_projection_report(
            exact_relative_residual=2.0e-6,
        )
        report = _run_runtime_flow_advance(
            trace,
            consistency_projection_count=3,
            terminal_no_slip_residuals_mps=(5.0e-5, 2.0e-4, 5.0e-5, 5.0e-5),
            pressure_reports=(
                pressure_not_converged,
                _healthy_pressure_projection_report(),
                _healthy_pressure_projection_report(),
                _healthy_pressure_projection_report(),
            ),
        )

        self.assertEqual(
            trace.events,
            tuple(
                (phase, stage)
                for stage in ("main", "consistency_1", "consistency_2")
                for phase in ("Q.prepare", "Q.solve", "Q.commit", "P")
            ),
        )
        joint_report = report["projection_report"]
        self.assertTrue(joint_report["hibm_joint_qp_converged"])
        self.assertEqual(joint_report["hibm_joint_qp_cycle_budget"], 4)
        self.assertEqual(joint_report["hibm_joint_qp_cycles_used"], 3)
        cycle_trace = joint_report["hibm_joint_qp_cycle_trace"]
        self.assertTrue(cycle_trace[0]["no_slip_converged"])
        self.assertFalse(cycle_trace[0]["pressure_converged"])
        self.assertFalse(cycle_trace[1]["no_slip_converged"])
        self.assertTrue(cycle_trace[1]["pressure_converged"])
        self.assertTrue(cycle_trace[2]["converged"])

    def test_joint_qp_uses_consistency_iterations_as_a_maximum_budget(self) -> None:
        trace = _FlowAdvanceQpTrace()
        report = _run_runtime_flow_advance(
            trace,
            consistency_projection_count=3,
            terminal_no_slip_residuals_mps=(5.0e-5, 5.0e-5, 5.0e-5, 5.0e-5),
        )

        self.assertEqual(
            trace.events,
            (
                ("Q.prepare", "main"),
                ("Q.solve", "main"),
                ("Q.commit", "main"),
                ("P", "main"),
            ),
        )
        joint_report = report["projection_report"]
        self.assertEqual(joint_report["hibm_joint_qp_cycle_budget"], 4)
        self.assertEqual(joint_report["hibm_joint_qp_cycles_used"], 1)
        self.assertEqual(
            joint_report["hibm_post_dirichlet_consistency_projection_count"],
            0,
        )
        self.assertEqual(
            report["hibm_post_dirichlet_consistency_projection_count"],
            0,
        )

    def test_joint_qp_budget_exhaustion_fails_closed_before_history_append(
        self,
    ) -> None:
        trace = _FlowAdvanceQpTrace()
        history = [{"sentinel": "must remain unchanged"}]

        with self.assertRaisesRegex(RuntimeError, "joint Q/P.*budget exhausted"):
            _run_runtime_flow_advance(
                trace,
                consistency_projection_count=1,
                terminal_no_slip_residuals_mps=(2.0e-4, 2.0e-4),
                preflow_history=history,
            )

        self.assertEqual(history, [{"sentinel": "must remain unchanged"}])
        self.assertEqual(trace.events[-1], ("P", "consistency_1"))

    def test_joint_qp_budget_error_carries_structured_cycle_diagnostics(
        self,
    ) -> None:
        trace = _FlowAdvanceQpTrace()

        with self.assertRaisesRegex(
            RuntimeError,
            "joint Q/P.*budget exhausted",
        ) as caught:
            _run_runtime_flow_advance(
                trace,
                consistency_projection_count=1,
                terminal_no_slip_residuals_mps=(2.0e-4, 2.0e-4),
            )

        diagnostics = caught.exception.diagnostics
        cycle_trace = diagnostics["hibm_joint_qp_cycle_trace"]
        self.assertEqual(len(cycle_trace), 2)
        terminal = cycle_trace[-1]
        expected = {
            "no_slip_direct_sample_marker_count": 0,
            "no_slip_normal_walk_sample_marker_count": 1,
            "no_slip_nearest_fluid_sample_marker_count": 0,
            "no_slip_sampling_identity_generation": 23,
            "no_slip_topology_generation": 31,
            "no_slip_component_face_valid_mask_generation": 47,
            "pre_projection_velocity_iterations": 6,
            "pre_projection_velocity_max_residual_mps": 2.5e-5,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(terminal[key], value)
        self.assertEqual(terminal["no_slip_argmax_marker_index"], 7)
        self.assertEqual(terminal["no_slip_argmax_marker_region_id"], 3)
        self.assertEqual(
            terminal["no_slip_argmax_residual_vector_mps"],
            (2.0e-4, 0.0, 0.0),
        )
        self.assertEqual(
            terminal["no_slip_argmax_sample_position_m"],
            (0.1, 0.2, 0.3),
        )
        self.assertEqual(
            terminal["no_slip_argmax_fluid_velocity_mps"],
            (1.0, 2.0, 3.0),
        )
        json.dumps(diagnostics, allow_nan=False)

    def test_joint_qp_malformed_optional_diagnostics_fail_as_json_safe_joint_error(
        self,
    ) -> None:
        no_slip_report = _terminal_no_slip_report(float("nan"))
        no_slip_report.update(
            {
                "hibm_no_slip_direct_sample_marker_count": None,
                "hibm_no_slip_normal_walk_sample_marker_count": float("inf"),
                "hibm_no_slip_nearest_fluid_sample_marker_count": "malformed",
                "hibm_no_slip_no_fluid_sample_marker_count": {},
                "hibm_no_slip_sampling_identity_generation": [],
                "hibm_no_slip_topology_generation": object(),
                "hibm_no_slip_component_face_valid_mask_generation": float("nan"),
                "hibm_no_slip_residual_argmax_marker_index": None,
                "hibm_no_slip_residual_argmax_marker_region_id": float("inf"),
                "hibm_no_slip_residual_argmax_residual_vector_mps": (
                    np.float32(float("nan")),
                    np.float64(float("inf")),
                    object(),
                ),
                "hibm_no_slip_residual_argmax_sample_position_m": None,
                "hibm_no_slip_residual_argmax_fluid_velocity_mps": "malformed",
            }
        )
        pressure_report = {
            **_healthy_pressure_projection_report(),
            "l2": float("nan"),
            "iterations": object(),
        }

        cycle = solid_mpm_fsi_runner._hibm_joint_qp_cycle_diagnostics(
            cycle_index=1,
            projection_stage="main",
            no_slip_report=no_slip_report,
            pressure_report=pressure_report,
            no_slip_absolute_tolerance_mps=1.0e-4,
            pressure_cg_tolerance=1.0e-6,
        )
        cycle["nested_optional_diagnostics"] = {
            "nonfinite_vector": (float("nan"), np.float32(float("inf"))),
            "opaque_value": object(),
        }
        diagnostics = solid_mpm_fsi_runner._hibm_joint_qp_terminal_diagnostics(
            cycle_budget=1,
            cycle_trace=[cycle],
        )

        json.dumps(diagnostics, allow_nan=False)
        with self.assertRaises(
            solid_mpm_fsi_runner.HibmJointQpConvergenceError
        ) as caught:
            solid_mpm_fsi_runner._require_hibm_joint_qp_convergence(
                diagnostics,
                context="malformed optional diagnostics",
            )
        json.dumps(caught.exception.diagnostics, allow_nan=False)

    def test_joint_qp_success_preserves_cycle_terminal_and_total_stage_timings(
        self,
    ) -> None:
        stage_names = tuple(
            solid_mpm_fsi_runner._HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES
        )
        main_times = {
            stage_name: float(index + 1)
            for index, stage_name in enumerate(stage_names)
        }
        consistency_times = {
            stage_name: 0.25 * float(index + 1)
            for index, stage_name in enumerate(stage_names)
        }
        expected_total = {
            stage_name: main_times[stage_name] + consistency_times[stage_name]
            for stage_name in stage_names
        }

        report = _run_runtime_flow_advance(
            _FlowAdvanceQpTrace(),
            consistency_projection_count=1,
            terminal_no_slip_residuals_mps=(2.0e-4, 5.0e-5),
            sharp_boundary_stage_times=(main_times, consistency_times),
        )

        projection_report = report["projection_report"]
        cycle_trace = projection_report["hibm_joint_qp_cycle_trace"]
        self.assertEqual(
            cycle_trace[0]["hibm_sharp_marker_boundary_stage_wall_time_s"],
            main_times,
        )
        self.assertEqual(
            cycle_trace[1]["hibm_sharp_marker_boundary_stage_wall_time_s"],
            consistency_times,
        )
        for owner in (report, projection_report):
            self.assertEqual(
                owner[
                    "hibm_sharp_marker_boundary_terminal_stage_wall_time_s"
                ],
                consistency_times,
            )
            self.assertEqual(
                owner["hibm_sharp_marker_boundary_total_stage_wall_time_s"],
                expected_total,
            )
        json.dumps(report, allow_nan=False)

    def test_pressure_solve_context_carries_current_cycle_stage_timings(
        self,
    ) -> None:
        stage_names = tuple(
            solid_mpm_fsi_runner._HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES
        )
        main_times = {
            stage_name: float(index + 1)
            for index, stage_name in enumerate(stage_names)
        }
        consistency_times = {
            stage_name: 0.5 * float(index + 1)
            for index, stage_name in enumerate(stage_names)
        }
        trace = _FlowAdvanceQpTrace()

        _run_runtime_flow_advance(
            trace,
            consistency_projection_count=1,
            terminal_no_slip_residuals_mps=(2.0e-4, 5.0e-5),
            sharp_boundary_stage_times=(main_times, consistency_times),
        )

        self.assertEqual(len(trace.pressure_solve_contexts), 2)
        main_context, consistency_context = trace.pressure_solve_contexts
        for context, expected_times in (
            (main_context, main_times),
            (consistency_context, consistency_times),
        ):
            self.assertEqual(context["phase"], "preflow")
            self.assertEqual(context["step_index_local"], 0)
            self.assertEqual(context["step_index_global"], 0)
            self.assertEqual(
                context["hibm_sharp_marker_boundary_stage_wall_time_s"],
                expected_times,
            )
            json.dumps(context, allow_nan=False)
        self.assertNotIn("projection_stage", main_context)
        self.assertEqual(
            consistency_context["projection_stage"],
            "post_dirichlet_consistency",
        )

    def test_joint_qp_failure_carries_cycle_and_total_stage_timings(self) -> None:
        stage_names = tuple(
            solid_mpm_fsi_runner._HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES
        )
        main_times = {stage_name: 1.0 for stage_name in stage_names}
        consistency_times = {stage_name: 2.0 for stage_name in stage_names}
        expected_total = {stage_name: 3.0 for stage_name in stage_names}

        with self.assertRaises(
            solid_mpm_fsi_runner.HibmJointQpConvergenceError
        ) as caught:
            _run_runtime_flow_advance(
                _FlowAdvanceQpTrace(),
                consistency_projection_count=1,
                terminal_no_slip_residuals_mps=(2.0e-4, 2.0e-4),
                sharp_boundary_stage_times=(main_times, consistency_times),
            )

        diagnostics = caught.exception.diagnostics
        self.assertEqual(
            diagnostics[
                "hibm_sharp_marker_boundary_terminal_stage_wall_time_s"
            ],
            consistency_times,
        )
        self.assertEqual(
            diagnostics["hibm_sharp_marker_boundary_total_stage_wall_time_s"],
            expected_total,
        )
        cycle_trace = diagnostics["hibm_joint_qp_cycle_trace"]
        self.assertEqual(
            cycle_trace[0]["hibm_sharp_marker_boundary_stage_wall_time_s"],
            main_times,
        )
        self.assertEqual(
            cycle_trace[1]["hibm_sharp_marker_boundary_stage_wall_time_s"],
            consistency_times,
        )
        json.dumps(diagnostics, allow_nan=False)

    def test_joint_qp_pressure_health_is_strict_and_uses_effective_tolerance(
        self,
    ) -> None:
        no_slip = _terminal_no_slip_report(5.0e-5)
        healthy = solid_mpm_fsi_runner._hibm_joint_qp_cycle_diagnostics(
            cycle_index=1,
            projection_stage="main",
            no_slip_report=no_slip,
            pressure_report=_healthy_pressure_projection_report(),
            no_slip_absolute_tolerance_mps=1.0e-4,
            pressure_cg_tolerance=1.0e-6,
        )
        self.assertTrue(healthy["pressure_converged"])
        self.assertTrue(healthy["converged"])

        unhealthy_reports = (
            {**_healthy_pressure_projection_report(), "cg_converged_all": False},
            {**_healthy_pressure_projection_report(), "pressure_solve_failed": True},
            {
                **_healthy_pressure_projection_report(),
                "pressure_projection_physical_failure": True,
            },
            {**_healthy_pressure_projection_report(), "l2": float("nan")},
            {**_healthy_pressure_projection_report(), "max_abs": float("inf")},
            _healthy_pressure_projection_report(exact_relative_residual=1.1e-6),
            {
                key: value
                for key, value in _healthy_pressure_projection_report().items()
                if key != "cg_exact_relative_residual_max"
            },
        )
        for pressure_report in unhealthy_reports:
            with self.subTest(pressure_report=pressure_report):
                cycle = solid_mpm_fsi_runner._hibm_joint_qp_cycle_diagnostics(
                    cycle_index=1,
                    projection_stage="main",
                    no_slip_report=no_slip,
                    pressure_report=pressure_report,
                    no_slip_absolute_tolerance_mps=1.0e-4,
                    pressure_cg_tolerance=1.0e-6,
                )
                self.assertFalse(cycle["pressure_converged"])
                self.assertFalse(cycle["converged"])
                json.dumps(cycle, allow_nan=False)

    def test_joint_qp_consistency_uses_the_configured_reprojection_tolerance(
        self,
    ) -> None:
        trace = _FlowAdvanceQpTrace()
        with self.assertRaisesRegex(RuntimeError, "joint Q/P.*budget exhausted"):
            _run_runtime_flow_advance(
                trace,
                consistency_projection_count=1,
                terminal_no_slip_residuals_mps=(2.0e-4, 5.0e-5),
                pressure_reports=(
                    _healthy_pressure_projection_report(),
                    _healthy_pressure_projection_report(
                        exact_relative_residual=5.0e-7,
                    ),
                ),
                reprojection_cg_tolerance=1.0e-7,
            )
        self.assertEqual(trace.events[-1], ("P", "consistency_1"))

    def test_joint_qp_diagnostics_are_json_safe_and_end_in_pressure(self) -> None:
        trace = _FlowAdvanceQpTrace()
        report = _run_runtime_flow_advance(
            trace,
            consistency_projection_count=2,
            terminal_no_slip_residuals_mps=(2.0e-4, 5.0e-5, 5.0e-5),
        )
        joint_report = report["projection_report"]

        for key in JOINT_QP_REPORT_KEYS:
            with self.subTest(report_key=key):
                self.assertIn(key, joint_report)
        self.assertTrue(joint_report["hibm_joint_qp_measured"])
        self.assertEqual(
            joint_report["hibm_joint_qp_terminal_no_slip_vector_max_residual_mps"],
            5.0e-5,
        )
        self.assertAlmostEqual(
            joint_report["hibm_joint_qp_pressure_reintroduced_no_slip_mps"],
            5.0e-5 - (3.0**0.5) * 2.5e-5,
        )
        self.assertEqual(
            joint_report["hibm_joint_qp_final_operation"],
            "pressure_projection",
        )
        self.assertEqual(trace.events[-1][0], "P")
        json.dumps(joint_report, allow_nan=False)

    def test_non_hibm_flow_keeps_one_pressure_projection_without_joint_gate(
        self,
    ) -> None:
        trace = _FlowAdvanceQpTrace()
        report = _run_runtime_flow_advance(
            trace,
            consistency_projection_count=3,
            sharp_boundary=False,
        )

        self.assertEqual(trace.events, (("P", "main"),))
        self.assertNotIn("hibm_joint_qp_converged", report["projection_report"])

    def test_projection_report_q_health_uses_runtime_all_truth_table(self) -> None:
        healthy_report = {key: True for key in Q_REPORT_KEYS}
        projection_reports = [dict(healthy_report) for _ in range(4)]
        combined = _combine_flow_projection_reports(projection_reports)
        for aggregate_key in Q_AGGREGATE_KEYS:
            with self.subTest(aggregate_key=aggregate_key):
                self.assertTrue(combined[aggregate_key])

        for stage in range(len(projection_reports)):
            for report_key, aggregate_key in zip(Q_REPORT_KEYS, Q_AGGREGATE_KEYS):
                with self.subTest(stage=stage, report_key=report_key):
                    reports = [dict(healthy_report) for _ in range(4)]
                    reports[stage][report_key] = False
                    combined = _combine_flow_projection_reports(reports)
                    self.assertFalse(combined[aggregate_key])

        for report_key, aggregate_key in zip(Q_REPORT_KEYS, Q_AGGREGATE_KEYS):
            with self.subTest(missing_report_key=report_key):
                reports = [dict(healthy_report) for _ in range(4)]
                reports[2].pop(report_key)
                combined = _combine_flow_projection_reports(reports)
                self.assertFalse(combined[aggregate_key])

    def test_project_current_flow_forwards_the_optional_projector(self) -> None:
        function = _top_level_function(RUNNER_PATH, "_project_current_flow")
        default = _argument_default(function, PROJECTOR_ARGUMENT)
        project_call = _single_call(function, "project")

        self.assertIsInstance(default, ast.Constant)
        self.assertIsNone(default.value)
        self.assertIn(PROJECTOR_ARGUMENT, _keyword_names(project_call))

    def test_projector_shares_the_sharp_boundary_cache_key_and_is_case_agnostic(self) -> None:
        assembly = _top_level_function(
            RUNNER_PATH,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
        )
        strings = _literal_strings(assembly)

        self.assertIn(PROJECTOR_CACHE_KEY, strings)
        assigned_cache_keys = {
            candidate.id
            for node in ast.walk(assembly)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for candidate in ast.walk(node)
            if isinstance(candidate, ast.Name)
            and candidate.id
            in {"resource_cache_key", "classified_topology_key"}
        }
        self.assertEqual(
            assigned_cache_keys,
            {"resource_cache_key", "classified_topology_key"},
        )
        assembly_source = ast.get_source_segment(
            RUNNER_PATH.read_text(encoding="utf-8"),
            assembly,
        ).lower()
        self.assertNotIn("vertical_flap", assembly_source)
        self.assertNotIn("ansys", assembly_source)

    def test_topology_only_path_never_runs_a_q_transaction(self) -> None:
        assembly = _top_level_function(
            RUNNER_PATH,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
        )
        topology_only_branches = [
            node
            for node in ast.walk(assembly)
            if isinstance(node, ast.If)
            and "topology_only" in {
                candidate.id
                for candidate in ast.walk(node.test)
                if isinstance(candidate, ast.Name)
            }
            and any(isinstance(statement, ast.Return) for statement in node.body)
        ]
        self.assertEqual(len(topology_only_branches), 1)
        branch = topology_only_branches[0]
        for phase in (PREPARE_METHOD, SOLVE_METHOD, COMMIT_METHOD):
            with self.subTest(transaction_phase=phase):
                self.assertFalse(_calls(branch, phase))

    def test_main_and_every_configured_consistency_projection_receive_q(self) -> None:
        advance = _top_level_function(RUNNER_PATH, "_flow_advance_current_step")
        qp_calls = _calls(advance, "_project_current_flow")
        self.assertEqual(
            len(qp_calls),
            2,
            "one source call is the main Q/P and one is the looped consistency Q/P",
        )
        for call in qp_calls:
            with self.subTest(call_lineno=call.lineno):
                self.assertIn(PROJECTOR_ARGUMENT, _keyword_names(call))

        consistency_loops = [
            node
            for node in ast.walk(advance)
            if isinstance(node, ast.For)
            and any(
                isinstance(candidate, ast.Name)
                and candidate.id == "consistency_projection_count"
                for candidate in ast.walk(node.iter)
            )
        ]
        self.assertEqual(len(consistency_loops), 1)
        loop = consistency_loops[0]
        loop_qp_call = _single_call(loop, "_project_current_flow")
        self.assertIn(PROJECTOR_ARGUMENT, _keyword_names(loop_qp_call))
        self.assertFalse(
            any(
                call.lineno > loop_qp_call.lineno
                for phase in (PREPARE_METHOD, SOLVE_METHOD, COMMIT_METHOD)
                for call in _calls(loop, phase)
            ),
            "each consistency iteration must terminate with P, not a trailing Q",
        )

    def test_projection_report_combines_q_health_across_main_and_consistency(self) -> None:
        combine = _top_level_function(RUNNER_PATH, "_combine_flow_projection_reports")
        strings = _literal_strings(combine)
        for key in Q_AGGREGATE_KEYS:
            with self.subTest(aggregate_key=key):
                self.assertIn(key, strings)

    def test_terminal_preflow_gate_requires_q_no_slip_and_p_health_together(self) -> None:
        snapshot_gate = _top_level_function(
            RUNNER_PATH,
            "_preflow_report_snapshot_payload",
        )
        window_gate = _top_level_function(
            RUNNER_PATH,
            "_preflow_windowed_stationary_report",
        )
        mandatory_keys = {
            *{f"flow_projection_{key}" for key in Q_AGGREGATE_KEYS},
            "hibm_no_slip_invalid_marker_count",
            "hibm_no_slip_max_residual_mps",
            "flow_projection_cg_converged_all",
            "flow_projection_cg_breakdown_count",
            "flow_projection_pressure_solve_failed",
            "flow_projection_pressure_projection_physical_failure",
        }
        for gate in (snapshot_gate, window_gate):
            strings = _literal_strings(gate)
            for key in mandatory_keys:
                with self.subTest(gate=gate.name, health_key=key):
                    self.assertIn(key, strings)

    def test_q_diagnostics_are_exposed_by_the_public_projection_report_ledger(self) -> None:
        runner = _module(RUNNER_PATH)
        assignments = [
            statement
            for statement in runner.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "FLOW_PROJECTION_REPORT_KEYS"
                for target in statement.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        report_keys = set(ast.literal_eval(assignments[0].value))
        for key in Q_AGGREGATE_KEYS:
            with self.subTest(report_key=key):
                self.assertIn(key, report_keys)

    def test_joint_qp_diagnostics_are_exposed_by_the_public_projection_report_ledger(
        self,
    ) -> None:
        runner = _module(RUNNER_PATH)
        assignments = [
            statement
            for statement in runner.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "FLOW_PROJECTION_REPORT_KEYS"
                for target in statement.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        report_keys = set(ast.literal_eval(assignments[0].value))
        for key in JOINT_QP_REPORT_KEYS:
            with self.subTest(report_key=key):
                self.assertIn(key, report_keys)


if __name__ == "__main__":
    unittest.main()
