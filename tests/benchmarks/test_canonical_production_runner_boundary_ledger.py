from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from benchmarks.official import solid_mpm_fsi_runner
from simulation_core.coupling.hibm_mpm.core import HibmMpmIbBoundaryConditions
from simulation_core.fluids.solver import CartesianFluidSolver
from tests.solvers.test_canonical_velocity_boundary_component_report import (
    CANONICAL_REPORT_KEYS,
)


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "official"
    / "solid_mpm_fsi_runner.py"
)
CANONICAL_LEDGER_FIELDS = (
    "velocity_dirichlet_boundary_active_component_mask",
    "velocity_dirichlet_boundary_value_mps",
    "velocity_dirichlet_boundary_pressure_mobility",
    "velocity_dirichlet_boundary_component_enforcement_weight",
    "velocity_dirichlet_boundary_component_region_id",
    "velocity_dirichlet_boundary_hard_fixed_component_mask",
    "velocity_dirichlet_boundary_external_exact_component_mask",
    "velocity_dirichlet_boundary_owned_component_mask",
)
EXPECTED_PREPARE_METHODS = (
    "prepare_velocity_dirichlet_component_ledger_apply",
    "prepare_velocity_dirichlet_component_ledger_divergence",
    "prepare_velocity_dirichlet_component_ledger_reachability",
    "prepare_velocity_dirichlet_component_ledger_fv_operator",
    "prepare_velocity_dirichlet_component_ledger_gradient",
    "prepare_velocity_dirichlet_component_ledger_multigrid",
    "prepare_velocity_dirichlet_component_ledger_projection",
    "prepare_hibm_no_slip_component_face_valid_mask",
    "prepare_velocity_dirichlet_component_ledger_reference",
    "prepare_velocity_dirichlet_component_ledger_snapshot",
)
def _module_ast() -> ast.Module:
    return ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))


def _function_node(name: str) -> ast.FunctionDef:
    for statement in _module_ast().body:
        if isinstance(statement, ast.FunctionDef) and statement.name == name:
            return statement
    raise AssertionError(f"missing function {name!r}")


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
    """Return every concrete statement list nested below ``node``."""

    result: list[list[ast.stmt]] = []
    for _, value in ast.iter_fields(node):
        if isinstance(value, list):
            statements = [item for item in value if isinstance(item, ast.stmt)]
            if statements:
                result.append(statements)
            for item in value:
                if isinstance(item, ast.AST):
                    result.extend(_statement_lists(item))
        elif isinstance(value, ast.AST):
            result.extend(_statement_lists(value))
    return result


def _assignment_call_name(statement: ast.stmt) -> tuple[str, str] | None:
    if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
        return None
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        return None
    return statement.targets[0].id, _call_name(statement.value)


def _healthy_canonical_device_report() -> dict[str, object]:
    report: dict[str, object] = {key: 0 for key in CANONICAL_REPORT_KEYS}
    report.update(
        {
            "schema_version": 5,
            "authority": "canonical_component_face",
            "new_owned_claim_component_count": 3,
            "final_active_component_count": 5,
            "final_owned_component_count": 3,
            "final_external_exact_component_count": 2,
            "final_hard_component_count": 2,
            "final_soft_component_count": 3,
            "final_active_storage_row_count": 4,
            "final_active_x_component_count": 2,
            "final_active_y_component_count": 2,
            "final_active_z_component_count": 1,
            "primary_region_active_component_count": 1,
            "secondary_region_active_component_count": 1,
            "other_region_active_component_count": 1,
            "unassigned_region_active_component_count": 2,
            "max_abs_claim_target_mps": 4.0,
            "max_abs_committed_target_mps": 4.0,
            "min_active_pressure_mobility": 0.0,
            "max_active_pressure_mobility": 1.0,
            "min_active_enforcement_weight": 0.25,
            "max_active_enforcement_weight": 1.0,
            "actual_geometry_claim_count": 3,
            "marker_target_closure": {
                "enabled": True,
                "constraint_count": 3,
                "adjustable_constraint_count": 2,
                "immutable_constraint_count": 1,
                "solver": "serialized_kaczmarz",
                "solve_count": 1,
                "initial_max_residual_mps": 2.0e-4,
                "final_max_residual_mps": 5.0e-5,
                "final_max_adjustable_residual_mps": 5.0e-7,
                "final_max_immutable_residual_mps": 5.0e-5,
                "absolute_tolerance_mps": 1.0e-4,
                "closure_tolerance_mps": 1.0e-6,
                "density_kgm3": 1.225,
                "projection_only_marker_count": 0,
                "projection_only_evaluated_axis_count": 0,
                "projection_only_invalid_axis_count": 0,
                "projection_only_constraint_count": 0,
                "projection_only_max_residual_mps": 0.0,
            },
        }
    )
    return report


def _healthy_canonical_runner_report() -> dict[str, object]:
    return {
        "hibm_sharp_marker_boundary_enabled": True,
        "hibm_velocity_dirichlet_authority": "canonical",
        "hibm_velocity_dirichlet_ledger_generation": 7,
        "hibm_velocity_dirichlet_authority_registered": True,
        "hibm_velocity_dirichlet_authority_sealed": True,
        "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count": 0,
        "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count": 0,
        "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio": 0.0,
        "canonical_velocity_dirichlet_report": (
            _healthy_canonical_device_report()
        ),
    }


class _LifecycleFluid:
    velocity_dirichlet_boundary_authority = "canonical"
    _VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS = frozenset(
        {
            "apply",
            "projection",
            "reference",
            "snapshot",
            "divergence",
            "reachability",
            "fv_operator",
            "gradient",
            "multigrid",
            "no_slip",
        }
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _invalidate_hibm_pressure_reachability(self) -> None:
        self.calls.append("invalidate_reachability")

    def _velocity_dirichlet_component_ledger_generation_errors(self):
        self.calls.append("generation_errors")
        return [], [], [], []

    def seal_velocity_dirichlet_component_ledger(self) -> None:
        self.calls.append("seal")

    def _require_velocity_dirichlet_component_ledger_sealed(self) -> None:
        self.calls.append("require_sealed")


for _method_name in EXPECTED_PREPARE_METHODS:
    setattr(
        _LifecycleFluid,
        _method_name,
        lambda self, method_name=_method_name: self.calls.append(method_name),
    )


class CanonicalProductionRunnerBoundaryLedgerContracts(unittest.TestCase):
    def test_windowed_preflow_must_converge_before_fsi(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not reach stationary"):
            solid_mpm_fsi_runner._require_preflow_ready_for_fsi(
                {
                    "preflow_convergence_mode": "windowed_stationary",
                    "preflow_converged": False,
                    "preflow_status": "max_steps",
                    "preflow_steps_completed": 200,
                },
                expected_mode="windowed_stationary",
            )

    def test_preflow_report_mode_must_match_validated_config(self) -> None:
        for reported_mode in (None, "windowed_stationry", "single_step_legacy"):
            with self.subTest(reported_mode=reported_mode):
                with self.assertRaisesRegex(RuntimeError, "convergence mode"):
                    solid_mpm_fsi_runner._require_preflow_ready_for_fsi(
                        {
                            "preflow_convergence_mode": reported_mode,
                            "preflow_converged": True,
                        },
                        expected_mode="windowed_stationary",
                    )

    def test_official_runner_requires_a_fresh_complete_load_before_solid(self) -> None:
        with mock.patch.object(
            solid_mpm_fsi_runner,
            "hibm_mpm_external_force_parts_fresh_for_solid_step",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "external force transaction"):
                solid_mpm_fsi_runner._require_fresh_external_force_for_solid_step(
                    clear=object(),
                    scatter=object(),
                    marker_forces=object(),
                    stress=object(),
                    no_slip={},
                    projection={},
                )

        run_source = inspect.getsource(
            solid_mpm_fsi_runner.run_hibm_mpm_fsi
        )
        self.assertIn(
            "_require_preflow_ready_for_fsi(",
            run_source,
        )
        self.assertIn("expected_mode=str(config.preflow_convergence_mode)", run_source)
        self.assertIn("_select_and_advance_solid_macro_step(", run_source)
        self.assertNotIn("_advance_solid_substeps_batched(", run_source)
        self.assertLess(
            run_source.index("_require_fresh_external_force_for_solid_step("),
            run_source.index("_select_and_advance_solid_macro_step("),
        )

    def test_preflow_detailed_stage_progress_is_opt_in_and_host_only(self) -> None:
        """Default progress stays step-level; opt-in stages remain host-only."""

        preflow = _function_node("_run_fixed_solid_preflow")
        advance = _function_node("_flow_advance_current_step_trial")
        boundary = _function_node("_apply_hibm_sharp_marker_boundary_to_fluid")
        self.assertIn("preflow_stage_observer", ast.unparse(advance.args))
        self.assertIn("stage_observer", ast.unparse(boundary.args))
        canonical_ledger_calls = [
            call
            for call in ast.walk(boundary)
            if isinstance(call, ast.Call)
            and _call_name(call)
            == "assemble_velocity_dirichlet_component_face_ledger"
        ]
        self.assertEqual(len(canonical_ledger_calls), 1)
        canonical_ledger_observer_keywords = [
            keyword
            for keyword in canonical_ledger_calls[0].keywords
            if keyword.arg == "stage_observer"
        ]
        self.assertEqual(len(canonical_ledger_observer_keywords), 1)
        self.assertEqual(
            ast.unparse(canonical_ledger_observer_keywords[0].value),
            "canonical_ledger_stage_observer",
        )
        canonical_ledger_measure_calls = [
            call
            for call in ast.walk(boundary)
            if isinstance(call, ast.Call)
            and _call_name(call) == "_measure_hibm_sharp_boundary_stage"
            and len(call.args) >= 2
            and ast.unparse(call.args[1]) == "'canonical_ledger_build'"
        ]
        self.assertEqual(len(canonical_ledger_measure_calls), 1)
        excluded_wall_time_keywords = [
            keyword
            for keyword in canonical_ledger_measure_calls[0].keywords
            if keyword.arg == "excluded_wall_time"
        ]
        self.assertEqual(len(excluded_wall_time_keywords), 1)
        self.assertIn(
            "canonical_ledger_observer_wall_time_s",
            ast.unparse(excluded_wall_time_keywords[0].value),
        )
        preflow_source = ast.unparse(preflow)
        self.assertIn("phase='preflow_stage'", preflow_source)
        detailed_stage_guard = (
            "progress_observer is not None and bool(getattr(config, "
            "'detailed_preflow_stage_progress', False))"
        )
        self.assertIn(detailed_stage_guard, preflow_source)
        self.assertIn(
            "preflow_stage_observer=preflow_stage_observer",
            preflow_source,
        )
        self.assertIn("PreflowStageObserverError", preflow_source)
        self.assertGreaterEqual(preflow_source.count("observer_wall_time_s"), 4)
        observer_handler = next(
            handler
            for handler in ast.walk(preflow)
            if isinstance(handler, ast.ExceptHandler)
            and isinstance(handler.type, ast.Name)
            and handler.type.id == "Exception"
        )
        raised = next(
            node for node in ast.walk(observer_handler) if isinstance(node, ast.Raise)
        )
        self.assertEqual(ast.unparse(raised.exc.func), "PreflowStageObserverError")
        self.assertEqual(ast.unparse(raised.cause), "exc")
        self.assertNotIn("except BaseException as exc", preflow_source)
        for field_name in (
            "preflow_step",
            "preflow_steps_requested",
            "preflow_steps_completed",
            "preflow_stage",
        ):
            self.assertIn(field_name, preflow_source)
        required_stages = {
            "hibm_resource_allocate",
            "hibm_search_classify",
            "hibm_internal_obstacle_publish",
            "hibm_boundary_build",
            "hibm_velocity_row_assembly",
            "pre_predictor_hibm",
            "sst_wall_distance",
            "sst_transport",
            "momentum_predictor",
            "projection_hibm",
            "main_pressure_projection",
        }
        runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        for stage in required_stages:
            with self.subTest(stage=stage):
                self.assertIn(f'"{stage}_before"', runner_source)
                self.assertIn(f'"{stage}_after"', runner_source)
        self.assertIn("consistency_hibm_before[", advance_source := ast.unparse(advance))
        self.assertIn("consistency_pressure_projection_before[", advance_source)
        indexed_consistency_events = (
            "consistency_hibm_before",
            "consistency_hibm_after",
            "consistency_pressure_projection_before",
            "consistency_pressure_projection_after",
        )
        for event in indexed_consistency_events:
            with self.subTest(indexed_event=event):
                event_calls = [
                    call
                    for call in ast.walk(advance)
                    if isinstance(call, ast.Call)
                    and _call_name(call) == "preflow_stage_observer"
                    and call.args
                    and event in ast.unparse(call.args[0])
                ]
                self.assertEqual(len(event_calls), 1)
                self.assertIn(
                    "consistency_projection_index + 1",
                    ast.unparse(event_calls[0].args[0]),
                )
        apply_calls = [
            call
            for call in ast.walk(advance)
            if isinstance(call, ast.Call)
            and _call_name(call) == "_apply_hibm_sharp_marker_boundary_to_fluid"
        ]
        self.assertGreaterEqual(len(apply_calls), 3)
        for call in apply_calls:
            stage_observer_keyword = next(
                (
                    keyword
                    for keyword in call.keywords
                    if keyword.arg == "stage_observer"
                ),
                None,
            )
            self.assertIsNotNone(stage_observer_keyword)
            self.assertEqual(
                ast.unparse(stage_observer_keyword.value),
                "preflow_stage_observer",
            )
        recursive_calls = [
            call
            for call in ast.walk(advance)
            if isinstance(call, ast.Call)
            and _call_name(call) == "_flow_advance_current_step_trial"
        ]
        self.assertEqual(len(recursive_calls), 1)
        recursive_observer_keywords = [
            keyword
            for keyword in recursive_calls[0].keywords
            if keyword.arg == "preflow_stage_observer"
        ]
        self.assertEqual(len(recursive_observer_keywords), 1)
        self.assertEqual(
            ast.unparse(recursive_observer_keywords[0].value),
            "preflow_stage_observer",
        )
        for before, assignment_target, call_name, after in (
            (
                "projection_hibm_before",
                "sharp_boundary_report",
                "_apply_hibm_sharp_marker_boundary_to_fluid",
                "projection_hibm_after",
            ),
            (
                "main_pressure_projection_before",
                "main_flow_report",
                "_project_current_flow",
                "main_pressure_projection_after",
            ),
            (
                "consistency_hibm_before",
                "consistency_boundary_report",
                "_apply_hibm_sharp_marker_boundary_to_fluid",
                "consistency_hibm_after",
            ),
            (
                "consistency_pressure_projection_before",
                "consistency_flow_report",
                "_project_current_flow",
                "consistency_pressure_projection_after",
            ),
        ):
            with self.subTest(before=before):
                matching_statement_lists: list[list[ast.stmt]] = []
                for statements in _statement_lists(advance):
                    before_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if before in ast.unparse(statement)
                    ]
                    call_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if _assignment_call_name(statement)
                        == (assignment_target, call_name)
                    ]
                    after_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if after in ast.unparse(statement)
                    ]
                    if any(
                        before_index < call_index < after_index
                        for before_index in before_indices
                        for call_index in call_indices
                        for after_index in after_indices
                    ):
                        matching_statement_lists.append(statements)
                self.assertEqual(
                    len(matching_statement_lists),
                    1,
                    "before/call/after must be ordered in one enclosing statement list",
                )
        initial_assignment = next(
            node
            for node in ast.walk(preflow)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "preflow_stage_observer"
        )
        self.assertEqual(ast.unparse(initial_assignment.value), "None")
        observer_guard = next(
            node
            for node in ast.walk(preflow)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == detailed_stage_guard
        )
        self.assertIn("preflow_stage_observer = emit_preflow_stage", ast.unparse(observer_guard))
        preflow_loop = next(
            node
            for node in preflow.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "preflow_index"
        )
        step_started_call = preflow_loop.body[0]
        self.assertIsInstance(step_started_call, ast.Expr)
        self.assertEqual(_call_name(step_started_call.value), "_emit_run_progress")
        self.assertIn("phase='preflow_step'", ast.unparse(step_started_call))
        emit_completed_step = next(
            node
            for node in preflow.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "emit_completed_step"
        )
        self.assertIn("phase='preflow_step'", ast.unparse(emit_completed_step))
        timer_assignments = [
            node
            for node in ast.walk(preflow)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"preflow_flow_advance_wall_time_s", "preflow_step_wall_time_s"}
                for target in node.targets
            )
        ]
        self.assertEqual(len(timer_assignments), 2)
        for assignment in timer_assignments:
            self.assertIsInstance(assignment.value, ast.Call)
            timer_call = assignment.value
            self.assertEqual(_call_name(timer_call), "max")
            self.assertEqual(len(timer_call.args), 2)
            self.assertEqual(ast.unparse(timer_call.args[0]), "0.0")
            elapsed_minus_observer = timer_call.args[1]
            self.assertIsInstance(elapsed_minus_observer, ast.BinOp)
            self.assertIsInstance(elapsed_minus_observer.op, ast.Sub)
            self.assertEqual(
                ast.unparse(elapsed_minus_observer.right),
                "observer_wall_time_s",
            )
        numeric_handlers = [
            handler
            for handler in ast.walk(preflow)
            if isinstance(handler, ast.ExceptHandler)
            and "FloatingPointError" in ast.unparse(handler.type)
        ]
        self.assertEqual(len(numeric_handlers), 1)
        self.assertNotIn("PreflowStageObserverError", ast.unparse(numeric_handlers[0]))

    def test_runner_device_report_key_set_matches_canonical_report_contract(
        self,
    ) -> None:
        self.assertEqual(
            set(
                solid_mpm_fsi_runner.CANONICAL_HIBM_VELOCITY_DIRICHLET_DEVICE_REPORT_KEYS
            ),
            set(CANONICAL_REPORT_KEYS),
        )
        self.assertEqual(
            set(
                solid_mpm_fsi_runner.CANONICAL_HIBM_VELOCITY_DIRICHLET_NUMERIC_DEVICE_REPORT_KEYS
            ),
            set(CANONICAL_REPORT_KEYS) - {"marker_target_closure"},
        )

    def test_build_fluid_selects_canonical_only_for_sharp_component_face_hibm(
        self,
    ) -> None:
        function = _function_node("_build_fluid")
        source = ast.unparse(function)
        authority_call = next(
            call
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and _call_name(call) == "set_velocity_dirichlet_boundary_authority"
        )
        self.assertEqual(len(authority_call.args), 1)
        self.assertIsInstance(authority_call.args[0], ast.Constant)
        self.assertEqual(authority_call.args[0].value, "canonical")
        conditional = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and "_use_hibm_sharp_marker_boundary" in ast.unparse(node.test)
        )
        self.assertIn(authority_call, tuple(ast.walk(conditional)))
        self.assertLess(
            source.index(ast.unparse(authority_call)),
            source.index("return fluid"),
        )

    def test_non_sharp_runner_uses_legacy_authority_device_feedback(
        self,
    ) -> None:
        function = _function_node("_build_fluid")
        source = ast.unparse(function)
        self.assertNotIn("else:\n        fluid.set_velocity_dirichlet_boundary_authority", source)

        feedback_source = inspect.getsource(
            solid_mpm_fsi_runner._apply_marker_feedback_to_fluid
        )
        self.assertIn('authority != "legacy"', feedback_source)
        self.assertIn("fluid.apply_marker_feedback_constraints", feedback_source)
        self.assertNotIn("_host_fallback", feedback_source)

    def test_initialize_inlet_flow_writes_all_eight_canonical_fields(self) -> None:
        function = _function_node("_initialize_inlet_flow")
        source = ast.unparse(function)
        for field_name in CANONICAL_LEDGER_FIELDS:
            with self.subTest(field_name=field_name):
                self.assertIn(f"fluid.{field_name}.from_numpy", source)
        invalidate = "_invalidate_velocity_dirichlet_component_ledger"
        directed_zmax_writer = "refresh_zmax_inlet_boundary_canonical"
        prepare = "_prepare_and_seal_canonical_velocity_dirichlet_component_ledger"
        self.assertIn(invalidate, source)
        self.assertIn(directed_zmax_writer, source)
        self.assertIn(prepare, source)
        first_ledger_write = (
            "fluid.velocity_dirichlet_boundary_active_component_mask.from_numpy"
        )
        last_ledger_write = (
            "fluid.velocity_dirichlet_boundary_owned_component_mask.from_numpy"
        )
        self.assertLess(source.index(invalidate), source.index(first_ledger_write))
        self.assertGreater(source.index(prepare), source.index(last_ledger_write))
        self.assertLess(source.index(directed_zmax_writer), source.index(prepare))

    def test_hibm_assembly_requires_canonical_authority_and_seals_write(self) -> None:
        function = _function_node("_apply_hibm_sharp_marker_boundary_to_fluid")
        nested = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.FunctionDef) and node.name == "assemble_velocity_rows"
        )
        source = ast.unparse(nested)
        canonical_builder = "assemble_velocity_dirichlet_component_face_ledger"
        invalidate = "_invalidate_velocity_dirichlet_component_ledger"
        prepare = "_prepare_and_seal_canonical_velocity_dirichlet_component_ledger"
        for required in (
            "velocity_dirichlet_boundary_authority",
            canonical_builder,
            invalidate,
            prepare,
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn(
            "assemble_velocity_dirichlet_reconstructed_boundary_rows",
            source,
        )
        self.assertLess(source.index(invalidate), source.index(canonical_builder))
        self.assertLess(source.index(canonical_builder), source.index(prepare))
        canonical_call = next(
            call
            for call in ast.walk(nested)
            if isinstance(call, ast.Call) and _call_name(call) == canonical_builder
        )
        keyword_names = {keyword.arg for keyword in canonical_call.keywords}
        for field_name in CANONICAL_LEDGER_FIELDS:
            argument_name = field_name.replace("boundary_", "", 1)
            with self.subTest(canonical_argument=argument_name):
                self.assertIn(argument_name, keyword_names)
        compatibility_keywords = {
            "markers": "markers",
            "surface_projection_inactive_axis": "OUT_OF_PLANE_AXIS_INDEX",
            "marker_compatibility_absolute_tolerance_mps": (
                "marker_mac_constraint_absolute_tolerance_mps"
            ),
            "marker_compatibility_density_kgm3": "float(fluid.rho)",
        }
        keyword_values = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in canonical_call.keywords
            if keyword.arg is not None
        }
        for keyword, expected_value in compatibility_keywords.items():
            with self.subTest(marker_compatibility_argument=keyword):
                self.assertEqual(keyword_values.get(keyword), expected_value)
        closure_expression = keyword_values.get(
            "marker_compatibility_closure_tolerance_mps",
            "",
        )
        self.assertEqual(
            closure_expression,
            "marker_compatibility_closure_tolerance_mps",
        )

    def test_marker_compatibility_closure_tolerance_is_explicit_and_bounded(
        self,
    ) -> None:
        default_config = SimpleNamespace(
            flow_hibm_marker_mac_constraint_absolute_tolerance_mps=1.0e-4,
        )
        self.assertEqual(
            solid_mpm_fsi_runner._hibm_marker_compatibility_closure_tolerance_mps(
                default_config
            ),
            1.0e-6,
        )

        research_override = SimpleNamespace(
            flow_hibm_marker_mac_constraint_absolute_tolerance_mps=1.0e-4,
            flow_hibm_marker_compatibility_closure_tolerance_mps=2.0e-6,
        )
        self.assertEqual(
            solid_mpm_fsi_runner._hibm_marker_compatibility_closure_tolerance_mps(
                research_override
            ),
            2.0e-6,
        )

        for invalid_value in (0.0, -1.0e-6, float("nan"), float("inf")):
            with self.subTest(invalid_value=invalid_value):
                invalid_config = SimpleNamespace(
                    flow_hibm_marker_mac_constraint_absolute_tolerance_mps=1.0e-4,
                    flow_hibm_marker_compatibility_closure_tolerance_mps=(
                        invalid_value
                    ),
                )
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    solid_mpm_fsi_runner._hibm_marker_compatibility_closure_tolerance_mps(
                        invalid_config
                    )

        excessive_config = SimpleNamespace(
            flow_hibm_marker_mac_constraint_absolute_tolerance_mps=1.0e-4,
            flow_hibm_marker_compatibility_closure_tolerance_mps=1.1e-4,
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            solid_mpm_fsi_runner._hibm_marker_compatibility_closure_tolerance_mps(
                excessive_config
            )

    def test_direct_runner_refreshes_pressure_gradient_at_ib_nodes(self) -> None:
        function = _function_node("_apply_hibm_sharp_marker_boundary_to_fluid")
        relevant_names = {
            "build_from_search_device_fields",
            "update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes",
            "assemble_pressure_neumann_matrix_rows",
        }
        call_lines = {
            _call_name(call): int(call.lineno)
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and _call_name(call) in relevant_names
        }
        self.assertEqual(relevant_names, relevant_names & call_lines.keys())
        self.assertLess(
            call_lines["build_from_search_device_fields"],
            call_lines[
                "update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes"
            ],
        )
        self.assertLess(
            call_lines[
                "update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes"
            ],
            call_lines["assemble_pressure_neumann_matrix_rows"],
        )

    def test_prepare_and_seal_executes_the_full_dependency_order(self) -> None:
        fluid = _LifecycleFluid()
        solid_mpm_fsi_runner._prepare_and_seal_canonical_velocity_dirichlet_component_ledger(
            fluid
        )
        self.assertEqual(
            fluid.calls,
            [
                "invalidate_reachability",
                *EXPECTED_PREPARE_METHODS,
                "generation_errors",
                "seal",
                "require_sealed",
            ],
        )

    def test_canonical_builder_returns_a_device_measured_health_report(self) -> None:
        source = inspect.getsource(
            HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger
        )
        self.assertIn("_report_velocity_dirichlet_component_face_ledger_kernel", source)
        self.assertIn('"canonical_velocity_dirichlet_report"', source)
        self.assertNotIn("HibmMpmVelocityDirichletBoundaryReport(", source)

    def test_direct_runner_uses_the_only_serialized_kaczmarz_closure(
        self,
    ) -> None:
        function = _function_node("_apply_hibm_sharp_marker_boundary_to_fluid")
        builder_call = next(
            call
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and _call_name(call)
            == "assemble_velocity_dirichlet_component_face_ledger"
        )
        keywords = {keyword.arg: keyword.value for keyword in builder_call.keywords}
        self.assertNotIn("marker_compatibility_solver", keywords)
        iterations = keywords["marker_compatibility_iterations_per_batch"]
        self.assertIsInstance(iterations, ast.Constant)
        self.assertEqual(iterations.value, 64)

    def test_canonical_health_accepts_a_sealed_device_measured_report(self) -> None:
        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                _healthy_canonical_runner_report()
            )
        )

    def test_canonical_health_rejects_removed_weighted_closure(self) -> None:
        report = _healthy_canonical_runner_report()
        closure = report["canonical_velocity_dirichlet_report"][
            "marker_target_closure"
        ]
        closure["solver"] = "weighted_minimum_norm_lstsq"
        failure = (
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(report)
        )
        self.assertIsNotNone(failure)
        self.assertIn("solver is invalid", failure)

    def test_canonical_health_rejects_removed_weighted_diagnostics(
        self,
    ) -> None:
        for key in (
            "matrix_rank",
            "adjustable_dof_count",
            "least_squares_max_residual_mps",
            "materialized_max_residual_mps",
            "max_abs_correction_mps",
        ):
            with self.subTest(key=key):
                report = _healthy_canonical_runner_report()
                closure = report["canonical_velocity_dirichlet_report"][
                    "marker_target_closure"
                ]
                closure[key] = 0
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("unexpected key", failure)

    def test_canonical_health_rejects_removed_schema_versions(self) -> None:
        for schema_version in (2, 3, 4):
            with self.subTest(schema_version=schema_version):
                report = _healthy_canonical_runner_report()
                report["canonical_velocity_dirichlet_report"][
                    "schema_version"
                ] = schema_version
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("schema version", failure.lower())

    def test_canonical_health_requires_exact_schema_five_keys(self) -> None:
        missing = _healthy_canonical_runner_report()
        missing["canonical_velocity_dirichlet_report"].pop(
            "marker_target_closure"
        )
        unexpected = _healthy_canonical_runner_report()
        unexpected["canonical_velocity_dirichlet_report"]["legacy_row_count"] = 0

        for label, report in (("missing", missing), ("unexpected", unexpected)):
            with self.subTest(label=label):
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("canonical", failure.lower())

    def test_canonical_health_validates_marker_target_closure_report(self) -> None:
        healthy = _healthy_canonical_runner_report()
        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(healthy)
        )
        for solve_count in range(5):
            with self.subTest(solve_count=solve_count):
                report = copy.deepcopy(healthy)
                report["canonical_velocity_dirichlet_report"][
                    "marker_target_closure"
                ]["solve_count"] = solve_count
                self.assertIsNone(
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
        topology_mutation_report = copy.deepcopy(healthy)
        topology_mutation_closure = topology_mutation_report[
            "canonical_velocity_dirichlet_report"
        ]["marker_target_closure"]
        topology_mutation_closure.update(
            {
                "constraint_count": 83,
                "adjustable_constraint_count": 83,
                "immutable_constraint_count": 0,
                "solve_count": 15,
            }
        )
        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                topology_mutation_report
            )
        )
        excessive_solve_count = copy.deepcopy(healthy)
        excessive_solve_count["canonical_velocity_dirichlet_report"][
            "marker_target_closure"
        ]["solve_count"] = 5
        failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            excessive_solve_count
        )
        self.assertIsNotNone(failure)
        self.assertIn("solve count", failure.lower())
        mutations = (
            (
                "disabled",
                lambda closure: closure.__setitem__("enabled", False),
            ),
            (
                "bad partition",
                lambda closure: closure.__setitem__(
                    "adjustable_constraint_count",
                    3,
                ),
            ),
            (
                "nonfinite",
                lambda closure: closure.__setitem__(
                    "final_max_residual_mps",
                    float("nan"),
                ),
            ),
            (
                "loose closure tolerance",
                lambda closure: closure.__setitem__(
                    "closure_tolerance_mps",
                    1.0e-4,
                ),
            ),
            (
                "adjustable residual",
                lambda closure: closure.__setitem__(
                    "final_max_adjustable_residual_mps",
                    2.0e-6,
                ),
            ),
            (
                "immutable residual",
                lambda closure: closure.__setitem__(
                    "final_max_immutable_residual_mps",
                    2.0e-4,
                ),
            ),
            (
                "projection-only incomplete evaluation",
                lambda closure: closure.update(
                    {
                        "projection_only_marker_count": 1,
                        "projection_only_evaluated_axis_count": 2,
                    }
                ),
            ),
            (
                "projection-only invalid axis",
                lambda closure: closure.__setitem__(
                    "projection_only_invalid_axis_count",
                    1,
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                report = copy.deepcopy(healthy)
                closure = report["canonical_velocity_dirichlet_report"][
                    "marker_target_closure"
                ]
                mutate(closure)
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("closure", failure.lower())

    def test_canonical_health_accepts_pressure_normal_external_subset(self) -> None:
        report = _healthy_canonical_runner_report()
        device_report = report["canonical_velocity_dirichlet_report"]
        device_report.update(
            {
                # A zmax velocity boundary fixes all three velocity
                # components, while external-exact provenance marks only the
                # pressure-normal z face.  External and owned are independent
                # subsets of active; they do not partition active components.
                "new_owned_claim_component_count": 0,
                "final_active_component_count": 3,
                "final_owned_component_count": 0,
                "final_external_exact_component_count": 1,
                "final_hard_component_count": 3,
                "final_soft_component_count": 0,
                "final_active_storage_row_count": 1,
                "final_active_x_component_count": 1,
                "final_active_y_component_count": 1,
                "final_active_z_component_count": 1,
                "primary_region_active_component_count": 0,
                "secondary_region_active_component_count": 0,
                "other_region_active_component_count": 0,
                "unassigned_region_active_component_count": 3,
                "actual_geometry_claim_count": 0,
            }
        )

        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(report)
        )

        for label, key in (
            ("owned", "final_owned_component_count"),
            ("external", "final_external_exact_component_count"),
        ):
            with self.subTest(label=label):
                invalid_report = copy.deepcopy(report)
                invalid_report["canonical_velocity_dirichlet_report"][key] = 4
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        invalid_report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("subset", failure.lower())

        overlap_report = copy.deepcopy(report)
        overlap_report["canonical_velocity_dirichlet_report"][
            "external_owned_overlap_count"
        ] = 1
        overlap_failure = (
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                overlap_report
            )
        )
        self.assertIsNotNone(overlap_failure)
        self.assertIn("canonical", overlap_failure.lower())

    def test_canonical_health_fails_closed_on_schema_and_lifecycle_drift(self) -> None:
        cases = (
            ("missing authority", lambda report: report.pop("hibm_velocity_dirichlet_authority")),
            ("wrong authority", lambda report: report.__setitem__("hibm_velocity_dirichlet_authority", "legacy")),
            ("missing generation", lambda report: report.pop("hibm_velocity_dirichlet_ledger_generation")),
            ("zero generation", lambda report: report.__setitem__("hibm_velocity_dirichlet_ledger_generation", 0)),
            ("missing registered", lambda report: report.pop("hibm_velocity_dirichlet_authority_registered")),
            ("not registered", lambda report: report.__setitem__("hibm_velocity_dirichlet_authority_registered", False)),
            ("missing sealed", lambda report: report.pop("hibm_velocity_dirichlet_authority_sealed")),
            ("not sealed", lambda report: report.__setitem__("hibm_velocity_dirichlet_authority_sealed", False)),
            ("missing nested report", lambda report: report.pop("canonical_velocity_dirichlet_report")),
            ("wrong schema", lambda report: report["canonical_velocity_dirichlet_report"].__setitem__("schema_version", 1)),
            ("wrong nested authority", lambda report: report["canonical_velocity_dirichlet_report"].__setitem__("authority", "legacy")),
            ("missing nested key", lambda report: report["canonical_velocity_dirichlet_report"].pop("final_active_component_count")),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                report = _healthy_canonical_runner_report()
                mutate(report)
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("canonical", failure.lower())

    def test_canonical_health_rejects_every_nonzero_invariant_counter(self) -> None:
        invariant_counters = (
            "illegal_active_on_obstacle_storage_component_count",
            "invalid_mask_bits_count",
            "mask_subset_violation_count",
            "external_owned_overlap_count",
            "external_not_hard_count",
            "active_provenance_missing_count",
            "inactive_neutral_violation_count",
            "nonfinite_active_value_count",
            "nonfinite_active_mobility_count",
            "nonfinite_active_enforcement_count",
            "active_mobility_range_violation_count",
            "active_enforcement_range_violation_count",
            "hard_mobility_contract_violation_count",
            "hard_enforcement_contract_violation_count",
            "claim_conflict_count",
            "target_conflict_count",
            "region_conflict_count",
            "alpha_conflict_count",
            "nonfinite_claim_target_count",
            "nonfinite_geometry_count",
            "degenerate_geometry_count",
            "external_claim_collision_count",
            "missing_actual_sample_count",
            "relocation_blocked_count",
            "relocation_unavailable_count",
            "direct_geometry_one_sided_component_count",
        )
        for counter in invariant_counters:
            with self.subTest(counter=counter):
                report = _healthy_canonical_runner_report()
                report["canonical_velocity_dirichlet_report"][counter] = 1
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("canonical", failure.lower())

    def test_canonical_health_allows_nonzero_compatible_relocation_merge_count(
        self,
    ) -> None:
        report = _healthy_canonical_runner_report()
        report["canonical_velocity_dirichlet_report"][
            "relocation_merged_count"
        ] = 3

        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(report)
        )

    def test_canonical_health_accepts_segment_supported_pair_route_counter(
        self,
    ) -> None:
        key = "segment_supported_pair_route_fallback_count"
        report = _healthy_canonical_runner_report()
        report["canonical_velocity_dirichlet_report"][key] = 1
        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(report)
        )

        for invalid_value in (True, -1, 0.5, "1"):
            with self.subTest(invalid_value=invalid_value):
                invalid_report = _healthy_canonical_runner_report()
                invalid_report["canonical_velocity_dirichlet_report"][key] = (
                    invalid_value
                )
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        invalid_report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("count", failure.lower())

    def test_canonical_health_cross_checks_direct_geometry_metrics(self) -> None:
        healthy = _healthy_canonical_runner_report()
        healthy_device = healthy["canonical_velocity_dirichlet_report"]
        healthy_device.update(
            {
                "duplicate_claim_component_count": 1,
                "direct_geometry_reconstructed_component_count": 1,
                "max_compatible_direct_target_spread_mps": 5.0e-4,
                "nominal_direct_claim_count": 2,
            }
        )
        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(healthy)
        )

        identical_only = _healthy_canonical_runner_report()
        identical_only[
            "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count"
        ] = 1
        identical_only["canonical_velocity_dirichlet_report"].update(
            {
                "duplicate_claim_component_count": 1,
                "nominal_direct_claim_count": 2,
            }
        )
        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                identical_only
            )
        )

        insufficient_identical_claims = copy.deepcopy(identical_only)
        insufficient_identical_claims["canonical_velocity_dirichlet_report"][
            "nominal_direct_claim_count"
        ] = 1
        failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            insufficient_identical_claims
        )
        self.assertIsNotNone(failure)
        self.assertIn("two nominal direct claims", failure.lower())

        zero_spread_reconstruction = copy.deepcopy(healthy)
        zero_spread_reconstruction["canonical_velocity_dirichlet_report"].update(
            {
                "direct_geometry_reconstructed_component_count": 1,
                "max_compatible_direct_target_spread_mps": 0.0,
            }
        )
        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                zero_spread_reconstruction
            ),
            msg=(
                "segment provenance must reconstruct even when the two author "
                "targets happen to be equal"
            ),
        )

        for label, updates in (
            (
                "spread without reconstruction",
                {
                    "direct_geometry_reconstructed_component_count": 0,
                    "max_compatible_direct_target_spread_mps": 5.0e-4,
                },
            ),
            (
                "reconstruction exceeds nominal direct claims",
                {
                    "direct_geometry_reconstructed_component_count": 2,
                    "duplicate_claim_component_count": 2,
                    "nominal_direct_claim_count": 3,
                },
            ),
        ):
            with self.subTest(label=label):
                report = copy.deepcopy(healthy)
                report["canonical_velocity_dirichlet_report"].update(updates)
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("segment provenance", failure.lower())

    def test_canonical_health_allows_legal_obstacle_storage_but_rejects_illegal(
        self,
    ) -> None:
        legal_report = _healthy_canonical_runner_report()
        legal_device_report = legal_report["canonical_velocity_dirichlet_report"]
        legal_device_report.update(
            {
                # Backward-MAC orientation B stores the physical interface
                # face on the obstacle cell.  The raw counter remains useful
                # evidence, while the legal/illegal partition decides health.
                "active_on_obstacle_storage_component_count": 1,
                "legal_obstacle_interface_storage_component_count": 1,
                "illegal_active_on_obstacle_storage_component_count": 0,
            }
        )
        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                legal_report
            )
        )

        illegal_report = copy.deepcopy(legal_report)
        illegal_report["canonical_velocity_dirichlet_report"].update(
            {
                "legal_obstacle_interface_storage_component_count": 0,
                "illegal_active_on_obstacle_storage_component_count": 1,
            }
        )
        failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            illegal_report
        )
        self.assertIsNotNone(failure)
        self.assertIn("canonical", failure.lower())

    def test_canonical_health_rejects_component_partition_drift(self) -> None:
        cases = (
            ("hard-soft", "final_soft_component_count", 2),
            ("axis", "final_active_z_component_count", 0),
            ("region", "unassigned_region_active_component_count", 1),
        )
        for label, key, value in cases:
            with self.subTest(label=label):
                report = _healthy_canonical_runner_report()
                report["canonical_velocity_dirichlet_report"][key] = value
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("canonical", failure.lower())

    def test_canonical_health_rejects_segment_endpoint_clamp_diagnostic_drift(
        self,
    ) -> None:
        cases = (
            (
                "bool count",
                "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count",
                True,
            ),
            (
                "nonfinite ratio",
                "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio",
                float("nan"),
            ),
            (
                "outside support",
                "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio",
                1.01,
            ),
            (
                "ratio without count",
                "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio",
                0.5,
            ),
        )
        for label, key, value in cases:
            with self.subTest(label=label):
                report = _healthy_canonical_runner_report()
                report[key] = value
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("endpoint-clamp", failure.lower())

        exceeds_duplicate_components = _healthy_canonical_runner_report()
        exceeds_duplicate_components[
            "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count"
        ] = 1
        exceeds_duplicate_components[
            "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio"
        ] = 0.5
        failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            exceeds_duplicate_components
        )
        self.assertIsNotNone(failure)
        self.assertIn("exceed", failure.lower())

        endpoint_plus_identical_exceeds_duplicates = (
            _healthy_canonical_runner_report()
        )
        endpoint_plus_identical_exceeds_duplicates.update(
            {
                "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count": 1,
                "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count": 10,
                "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio": 0.4,
            }
        )
        endpoint_plus_identical_exceeds_duplicates[
            "canonical_velocity_dirichlet_report"
        ].update(
            {
                "duplicate_claim_component_count": 10,
                "direct_geometry_reconstructed_component_count": 0,
            }
        )
        failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            endpoint_plus_identical_exceeds_duplicates
        )
        self.assertIsNotNone(failure)
        self.assertIn("duplicate", failure.lower())

    def test_canonical_health_accepts_face_first_endpoint_clamp_accounting(
        self,
    ) -> None:
        report = _healthy_canonical_runner_report()
        report.update(
            {
                "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count": 0,
                "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count": 10,
                "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio": 0.4,
            }
        )
        report["canonical_velocity_dirichlet_report"].update(
            {
                "duplicate_claim_component_count": 10,
                "direct_geometry_reconstructed_component_count": 0,
            }
        )

        self.assertIsNone(
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(report)
        )

    def test_canonical_health_rejects_identical_segment_provenance_drift(
        self,
    ) -> None:
        for label, value in (
            ("bool", True),
            ("fractional", 1.5),
            ("negative", -1),
        ):
            with self.subTest(label=label):
                invalid = _healthy_canonical_runner_report()
                invalid[
                    "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count"
                ] = value
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        invalid
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("segment-provenance", failure.lower())

        exceeds_duplicates = _healthy_canonical_runner_report()
        exceeds_duplicates[
            "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count"
        ] = 1
        failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            exceeds_duplicates
        )
        self.assertIsNotNone(failure)
        self.assertIn("exceed duplicate", failure.lower())

    def test_schema_five_requires_segment_diagnostics(self) -> None:
        for key in (
            solid_mpm_fsi_runner.CANONICAL_HIBM_VELOCITY_DIRICHLET_SEGMENT_RUNNER_REPORT_KEYS
        ):
            with self.subTest(missing_segment_diagnostic=key):
                report = _healthy_canonical_runner_report()
                report.pop(key)
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn(key, failure)
                with self.assertRaises(KeyError):
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_mapping_fields(
                        report
                    )

    def test_canonical_mapper_preserves_endpoint_diagnostic_types_for_health_check(
        self,
    ) -> None:
        class FakeFluid:
            velocity_dirichlet_boundary_authority = "canonical"
            velocity_dirichlet_component_ledger_generation = 7
            velocity_dirichlet_component_ledger_sealed = True

            @staticmethod
            def _velocity_dirichlet_component_ledger_generation_errors():
                return (), (), (), ()

        mapped = solid_mpm_fsi_runner._canonical_hibm_velocity_dirichlet_report_fields(
            {
                "canonical_velocity_dirichlet_report": (
                    _healthy_canonical_device_report()
                ),
                "segment_identical_provenance_merged_component_count": 0,
                "segment_endpoint_clamped_component_count": True,
                "max_segment_endpoint_clamp_overrun_support_ratio": "0.5",
            },
            fluid=FakeFluid(),
        )

        self.assertIs(
            mapped[
                "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count"
            ],
            True,
        )
        self.assertEqual(
            mapped[
                "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio"
            ],
            "0.5",
        )
        failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            mapped
        )
        self.assertIsNotNone(failure)
        self.assertIn("endpoint-clamped count", failure.lower())

    def test_current_canonical_builder_must_publish_endpoint_diagnostics(
        self,
    ) -> None:
        class FakeFluid:
            velocity_dirichlet_boundary_authority = "canonical"
            velocity_dirichlet_component_ledger_generation = 7
            velocity_dirichlet_component_ledger_sealed = True

            @staticmethod
            def _velocity_dirichlet_component_ledger_generation_errors():
                return (), (), (), ()

        with self.assertRaisesRegex(KeyError, "segment_identical"):
            solid_mpm_fsi_runner._canonical_hibm_velocity_dirichlet_report_fields(
                {
                    "canonical_velocity_dirichlet_report": (
                        _healthy_canonical_device_report()
                    ),
                    "segment_endpoint_clamped_component_count": 0,
                    "max_segment_endpoint_clamp_overrun_support_ratio": 0.0,
                },
                fluid=FakeFluid(),
            )

    def test_canonical_mapping_does_not_invent_legacy_row_diagnostics(self) -> None:
        report = _healthy_canonical_runner_report()
        fields = solid_mpm_fsi_runner._hibm_velocity_dirichlet_mapping_fields(report)
        self.assertEqual(
            fields,
            {
                key: report[key]
                for key in solid_mpm_fsi_runner.CANONICAL_HIBM_VELOCITY_DIRICHLET_RUNNER_REPORT_KEYS
            },
        )
        staged_fields = (
            solid_mpm_fsi_runner._hibm_velocity_dirichlet_mapping_fields(
                report,
                stage="observer",
            )
        )
        self.assertIn(
            "hibm_observer_canonical_velocity_dirichlet_report",
            staged_fields,
        )
        self.assertNotIn("canonical_velocity_dirichlet_report", staged_fields)

    def test_canonical_consistency_reuse_compares_nested_component_report(self) -> None:
        reference = _healthy_canonical_runner_report()
        consistency = _healthy_canonical_runner_report()
        for report in (reference, consistency):
            report.update(
                {
                    "hibm_sharp_marker_boundary_topology_reused": True,
                    "hibm_preassembly_topology_mutated": False,
                    "hibm_velocity_dirichlet_row_ledger_snapshot_generation": 11,
                    "hibm_velocity_dirichlet_row_ledger_matches_reference": True,
                    "hibm_velocity_dirichlet_row_ledger_mismatch_rows": 0,
                }
            )
        solid_mpm_fsi_runner._require_velocity_only_consistency_row_reuse(
            reference,
            consistency,
            context="canonical-host-contract",
        )
        consistency["canonical_velocity_dirichlet_report"] = dict(
            consistency["canonical_velocity_dirichlet_report"]
        )
        consistency["canonical_velocity_dirichlet_report"][
            "duplicate_claim_component_count"
        ] = 1
        with self.assertRaisesRegex(RuntimeError, "canonical.*changed"):
            solid_mpm_fsi_runner._require_velocity_only_consistency_row_reuse(
                reference,
                consistency,
                context="canonical-host-contract",
            )

    def test_canonical_consistency_reuse_compares_endpoint_clamp_diagnostics(
        self,
    ) -> None:
        reference = _healthy_canonical_runner_report()
        consistency = _healthy_canonical_runner_report()
        for report in (reference, consistency):
            report.update(
                {
                    "hibm_sharp_marker_boundary_topology_reused": True,
                    "hibm_preassembly_topology_mutated": False,
                    "hibm_velocity_dirichlet_row_ledger_snapshot_generation": 11,
                    "hibm_velocity_dirichlet_row_ledger_matches_reference": True,
                    "hibm_velocity_dirichlet_row_ledger_mismatch_rows": 0,
                    "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count": 1,
                    "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio": 0.4,
                }
            )
            report["canonical_velocity_dirichlet_report"].update(
                {
                    "duplicate_claim_component_count": 1,
                    "direct_geometry_reconstructed_component_count": 1,
                    "nominal_direct_claim_count": 2,
                }
            )
        consistency[
            "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio"
        ] = 0.5

        with self.assertRaisesRegex(RuntimeError, "segment diagnostic.*changed"):
            solid_mpm_fsi_runner._require_velocity_only_consistency_row_reuse(
                reference,
                consistency,
                context="canonical-endpoint-clamp-contract",
            )

        identity_reference = _healthy_canonical_runner_report()
        identity_consistency = _healthy_canonical_runner_report()
        for report in (identity_reference, identity_consistency):
            report.update(
                {
                    "hibm_sharp_marker_boundary_topology_reused": True,
                    "hibm_preassembly_topology_mutated": False,
                    "hibm_velocity_dirichlet_row_ledger_snapshot_generation": 11,
                    "hibm_velocity_dirichlet_row_ledger_matches_reference": True,
                    "hibm_velocity_dirichlet_row_ledger_mismatch_rows": 0,
                    "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count": 1,
                }
            )
            report["canonical_velocity_dirichlet_report"].update(
                {
                    "duplicate_claim_component_count": 1,
                    "nominal_direct_claim_count": 2,
                }
            )
        identity_consistency[
            "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count"
        ] = 0
        with self.assertRaisesRegex(RuntimeError, "segment diagnostic.*changed"):
            solid_mpm_fsi_runner._require_velocity_only_consistency_row_reuse(
                identity_reference,
                identity_consistency,
                context="canonical-identical-provenance-contract",
            )

    def test_runner_content_equivalence_ignores_only_component_generation(self) -> None:
        class FakeFluid:
            content_mismatch_rows = 0

            def velocity_dirichlet_boundary_ledger_comparison(
                self,
                *,
                expected_generation: int,
            ) -> dict[str, object]:
                self.expected_generation = expected_generation
                content_mismatch_rows = int(self.content_mismatch_rows)
                return {
                    "schema_version": 1,
                    "reference_generation": expected_generation,
                    "device_content_mismatch_rows": content_mismatch_rows,
                    "identity_mismatch_rows": 1 + content_mismatch_rows,
                    "content_equivalence_mismatch_rows": content_mismatch_rows,
                    "authority_changed": False,
                    "component_generation_changed": True,
                    "face_symmetric_changed": False,
                    "reference_authority": "canonical",
                    "current_authority": "canonical",
                    "reference_component_generation": 124,
                    "current_component_generation": 126,
                    "reference_face_symmetric": 0,
                    "current_face_symmetric": 0,
                    "first_identity_mismatch_field": (
                        "device_content"
                        if content_mismatch_rows
                        else "component_ledger_generation"
                    ),
                    "first_content_mismatch_field": (
                        "device_content" if content_mismatch_rows else None
                    ),
                }

        fluid = FakeFluid()
        report = solid_mpm_fsi_runner._velocity_dirichlet_row_ledger_comparison(
            fluid,
            reference_generation=7,
            comparison_mode="content_equivalence",
            context="canonical controlled rebuild",
        )

        self.assertEqual(fluid.expected_generation, 7)
        self.assertTrue(
            report["hibm_velocity_dirichlet_row_ledger_matches_reference"]
        )
        self.assertEqual(
            report["hibm_velocity_dirichlet_row_ledger_mismatch_rows"],
            0,
        )
        self.assertTrue(
            report[
                "hibm_velocity_dirichlet_row_ledger_component_generation_changed"
            ]
        )
        self.assertEqual(
            report[
                "hibm_velocity_dirichlet_row_ledger_identity_mismatch_rows"
            ],
            1,
        )
        self.assertIsNone(
            report[
                "hibm_velocity_dirichlet_row_ledger_first_mismatch_field"
            ]
        )

        fluid.content_mismatch_rows = 1
        changed_report = (
            solid_mpm_fsi_runner._velocity_dirichlet_row_ledger_comparison(
                fluid,
                reference_generation=7,
                comparison_mode="content_equivalence",
                context="canonical changed content",
            )
        )
        self.assertFalse(
            changed_report[
                "hibm_velocity_dirichlet_row_ledger_matches_reference"
            ]
        )
        self.assertEqual(
            changed_report["hibm_velocity_dirichlet_row_ledger_mismatch_rows"],
            1,
        )
        self.assertEqual(
            changed_report[
                "hibm_velocity_dirichlet_row_ledger_first_mismatch_field"
            ],
            "device_content",
        )

    def test_runner_detailed_comparison_fails_closed_on_malformed_mapping(self) -> None:
        valid = {
            "schema_version": 1,
            "reference_generation": 7,
            "device_content_mismatch_rows": 0,
            "identity_mismatch_rows": 1,
            "content_equivalence_mismatch_rows": 0,
            "authority_changed": False,
            "component_generation_changed": True,
            "face_symmetric_changed": False,
            "reference_authority": "canonical",
            "current_authority": "canonical",
            "reference_component_generation": 124,
            "current_component_generation": 126,
            "reference_face_symmetric": 0,
            "current_face_symmetric": 0,
            "first_identity_mismatch_field": "component_ledger_generation",
            "first_content_mismatch_field": None,
        }
        cases = (
            {"content_equivalence_mismatch_rows": True},
            {
                "device_content_mismatch_rows": 1,
                "content_equivalence_mismatch_rows": 0,
            },
            {"component_generation_changed": "true"},
            {"authority_changed": True},
            {"first_content_mismatch_field": "device_content"},
        )

        for override in cases:
            class FakeFluid:
                def velocity_dirichlet_boundary_ledger_comparison(
                    self,
                    *,
                    expected_generation: int,
                ) -> dict[str, object]:
                    return {**valid, **override}

            with self.subTest(override=override):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "detailed comparison failed",
                ):
                    solid_mpm_fsi_runner._velocity_dirichlet_row_ledger_comparison(
                        FakeFluid(),
                        reference_generation=7,
                        comparison_mode="content_equivalence",
                        context="malformed detailed comparison",
                    )

    def test_runner_strict_fallback_rejects_coercible_tokens(self) -> None:
        class FakeFluid:
            mismatch_value: object = 0

            def velocity_dirichlet_boundary_ledger_mismatch_rows(
                self,
                *,
                expected_generation: int,
            ) -> object:
                return self.mismatch_value

        fluid = FakeFluid()
        for invalid_generation in (True, 1.0):
            with self.subTest(generation=invalid_generation):
                with self.assertRaisesRegex(TypeError, "exact integer"):
                    solid_mpm_fsi_runner._velocity_dirichlet_row_ledger_comparison(
                        fluid,
                        reference_generation=invalid_generation,
                        context="strict fallback generation",
                    )

        for invalid_mismatch in (True, 0.0):
            fluid.mismatch_value = invalid_mismatch
            with self.subTest(mismatch=invalid_mismatch):
                with self.assertRaisesRegex(RuntimeError, "comparison failed"):
                    solid_mpm_fsi_runner._velocity_dirichlet_row_ledger_comparison(
                        fluid,
                        reference_generation=1,
                        context="strict fallback mismatch",
                    )

    def test_canonical_health_rejects_bool_and_fractional_numeric_types(self) -> None:
        cases = (
            (
                "bool schema",
                lambda report: report["canonical_velocity_dirichlet_report"].__setitem__(
                    "schema_version", True
                ),
            ),
            (
                "bool generation",
                lambda report: report.__setitem__(
                    "hibm_velocity_dirichlet_ledger_generation", True
                ),
            ),
            (
                "fractional generation",
                lambda report: report.__setitem__(
                    "hibm_velocity_dirichlet_ledger_generation", 7.5
                ),
            ),
            (
                "bool count",
                lambda report: report["canonical_velocity_dirichlet_report"].__setitem__(
                    "duplicate_claim_component_count", True
                ),
            ),
            (
                "fractional count",
                lambda report: report["canonical_velocity_dirichlet_report"].__setitem__(
                    "duplicate_claim_component_count", 0.5
                ),
            ),
            (
                "bool extrema",
                lambda report: report["canonical_velocity_dirichlet_report"].__setitem__(
                    "max_abs_claim_target_mps", True
                ),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                report = _healthy_canonical_runner_report()
                mutate(report)
                failure = (
                    solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
                        report
                    )
                )
                self.assertIsNotNone(failure)
                self.assertIn("canonical", failure.lower())

    def test_removed_legacy_health_contract_is_rejected(self) -> None:
        report = {
            "hibm_sharp_marker_boundary_enabled": True,
            "hibm_velocity_dirichlet_active_rows": 1,
        }
        failure = solid_mpm_fsi_runner._hibm_velocity_dirichlet_health_failure(
            report
        )
        self.assertIsNotNone(failure)
        self.assertIn("canonical", failure.lower())

    def test_solver_has_a_canonical_zmax_directed_face_writer_without_compact_alias(
        self,
    ) -> None:
        canonical_refresh = getattr(
            CartesianFluidSolver,
            "refresh_zmax_inlet_boundary_canonical",
            None,
        )
        self.assertIsNotNone(canonical_refresh)
        source = inspect.getsource(canonical_refresh)
        self.assertIn("_require_canonical_velocity_dirichlet_boundary_authority", source)
        self.assertIn("_refresh_zmax_inlet_boundary_canonical_kernel", source)
        self.assertIn("_invalidate_velocity_dirichlet_component_ledger", source)
        kernel_source = inspect.getsource(
            CartesianFluidSolver._refresh_zmax_inlet_boundary_canonical_kernel
        )
        self.assertIn(
            "self.external_velocity_boundary_z_face_active_component_mask",
            kernel_source,
        )
        self.assertIn(
            "self.external_velocity_boundary_z_face_value_mps",
            kernel_source,
        )
        for field_name in CANONICAL_LEDGER_FIELDS:
            with self.subTest(canonical_zmax_field=field_name):
                self.assertNotIn(
                    f"self.{field_name}",
                    kernel_source,
                    msg=(
                        "a plus-side physical boundary face must not alias the "
                        "final backward internal MAC row"
                    ),
                )

    def test_runner_canonical_zmax_reprepares_and_reseals_after_device_write(self) -> None:
        source = inspect.getsource(solid_mpm_fsi_runner._refresh_zmax_inlet_boundary)
        writer = "refresh_zmax_inlet_boundary_canonical"
        prepare = "_prepare_and_seal_canonical_velocity_dirichlet_component_ledger"
        self.assertIn("velocity_dirichlet_boundary_authority", source)
        self.assertIn(writer, source)
        self.assertIn(prepare, source)
        self.assertLess(source.index(writer), source.index(prepare))

    def test_sharp_canonical_feedback_bypasses_the_legacy_collocated_writer(self) -> None:
        class _CanonicalFluid:
            velocity_dirichlet_boundary_authority = "canonical"

            def apply_marker_feedback_constraints(self, *_args, **_kwargs):
                raise AssertionError("legacy collocated writer must not be called")

        markers = SimpleNamespace(
            marker_count=3,
            x_gamma_m=object(),
            v_gamma_mps=object(),
            region_id=object(),
        )
        config = SimpleNamespace(
            flow_solid_boundary_mode="hibm_sharp_marker_rows",
            preserve_marker_velocity_constraints=True,
        )
        report = solid_mpm_fsi_runner._apply_marker_feedback_to_fluid(
            markers,
            _CanonicalFluid(),
            config,
            feedback_available=True,
        )
        self.assertFalse(report["fluid_marker_feedback_collocated_writer_used"])
        self.assertFalse(report["fluid_marker_velocity_constraints_enabled"])
        self.assertEqual(report["legacy_constraint_active_cell_count"], 0)
        self.assertEqual(
            report["fluid_marker_feedback_enforcement_mode"],
            "hibm_sharp_reconstructed_rows",
        )


if __name__ == "__main__":
    unittest.main()
