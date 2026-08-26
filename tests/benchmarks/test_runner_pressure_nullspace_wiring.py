from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import textwrap
import unittest

from benchmarks.official.solid_mpm_fsi_runner import (
    FLOW_PROJECTION_REPORT_KEYS,
    _HibmPreProjectionVelocityProjector,
    _combine_flow_projection_reports,
    _flow_advance_current_step_trial,
    _project_current_flow,
)


_PRESSURE_MARKER_REPORT_KEYS = (
    "pressure_marker_nullspace_enabled",
    "pressure_marker_nullspace_prepared",
    "pressure_marker_nullspace_active_constraint_count",
    "pressure_marker_nullspace_active_constraint_count_min",
    "pressure_marker_nullspace_active_constraint_count_max",
    "pressure_marker_nullspace_independent_constraint_count",
    "pressure_marker_nullspace_dependent_constraint_count",
    "pressure_marker_nullspace_unactuated_constraint_count",
    "pressure_marker_nullspace_apply_count",
    "pressure_marker_nullspace_pressure_actuation_generation",
    "pressure_marker_nullspace_min_factor_pivot",
    "pressure_marker_nullspace_max_dependent_normalized_pivot",
    "pressure_marker_nullspace_max_input_constraint_mps",
    "pressure_marker_nullspace_max_unactuated_input_constraint_mps",
    "pressure_marker_nullspace_max_constraint_residual_mps",
    "pressure_marker_nullspace_solver_scratch_resource_bytes",
    "pressure_marker_nullspace_marker_operator_resource_bytes",
    "pressure_marker_nullspace_resource_bytes",
    "pressure_marker_nullspace_actuation_invalid_count",
    "pressure_marker_nullspace_correction_invalid_count",
    "pressure_marker_nullspace_operator_apply_count",
    "pressure_marker_nullspace_velocity_correction_apply_count",
    "pressure_marker_nullspace_all_velocity_paths_projected",
)


@dataclass(frozen=True)
class _PressureNullspaceReport:
    prepared: bool
    active_constraint_count: int
    apply_count: int
    pressure_actuation_generation: int
    min_factor_pivot: float
    last_max_input_constraint: float
    last_max_constraint_residual: float
    resource_bytes: int
    independent_constraint_count: int = 0
    dependent_constraint_count: int = 0
    unactuated_constraint_count: int = 0
    max_dependent_normalized_pivot: float = 0.0
    max_unactuated_input_constraint: float = 0.0


class _RecordingMarkerOperator:
    def __init__(self) -> None:
        self._phase = "committed"
        self.prepare_calls: list[dict[str, object]] = []
        self.apply_calls: list[dict[str, object]] = []
        self.finalize_calls: list[dict[str, object]] = []

    def prepare_pressure_nullspace_transaction(self, **kwargs: object) -> None:
        self.prepare_calls.append(dict(kwargs))

    def apply_pressure_nullspace_transaction_device_only(
        self,
        **kwargs: object,
    ) -> None:
        self.apply_calls.append(dict(kwargs))
        return None

    def finalize_pressure_nullspace_transaction(
        self,
        **kwargs: object,
    ) -> _PressureNullspaceReport:
        self.finalize_calls.append(dict(kwargs))
        return _PressureNullspaceReport(
            prepared=True,
            active_constraint_count=6,
            apply_count=len(self.apply_calls),
            pressure_actuation_generation=int(
                kwargs["pressure_actuation_generation"]
            ),
            min_factor_pivot=0.5,
            last_max_input_constraint=0.25,
            last_max_constraint_residual=1.0e-13,
            resource_bytes=4096,
            independent_constraint_count=4,
            dependent_constraint_count=1,
            unactuated_constraint_count=1,
            max_dependent_normalized_pivot=1.0e-14,
            max_unactuated_input_constraint=0.0,
        )


class _FluidOwner:
    def __init__(self) -> None:
        self.hibm_reachability_revision = 7
        self.velocity_dirichlet_component_ledger_generation = 11
        self.pressure_velocity_actuation_generation = 13
        self.pressure_velocity_actuation_weight = object()


def _prepared_wrapper() -> tuple[
    _HibmPreProjectionVelocityProjector,
    _RecordingMarkerOperator,
    _FluidOwner,
    object,
]:
    operator = _RecordingMarkerOperator()
    wrapper = _HibmPreProjectionVelocityProjector(
        markers=object(),
        operator=operator,
        max_iterations=32,
        absolute_tolerance_mps=1.0e-4,
    )
    fluid = _FluidOwner()
    component_face_valid_mask = object()
    wrapper._prepared_fluid = fluid
    wrapper._prepared_sampling_identity = object()
    wrapper._prepared_component_face_valid_mask = component_face_valid_mask
    wrapper._prepared_topology_generation = 7
    wrapper._prepared_component_face_valid_mask_generation = 11
    return wrapper, operator, fluid, component_face_valid_mask


def _prepare_pressure_nullspace(
    wrapper: _HibmPreProjectionVelocityProjector,
    fluid: _FluidOwner,
    component_face_valid_mask: object,
) -> None:
    wrapper.prepare_pressure_nullspace_transaction(
        fluid=fluid,
        pressure_actuated_component_mobility=(
            fluid.pressure_velocity_actuation_weight
        ),
        component_face_valid_mask=component_face_valid_mask,
        pressure_actuation_generation=13,
        topology_generation=7,
        component_face_valid_mask_generation=11,
    )


class RunnerPressureNullspaceWiringTests(unittest.TestCase):
    def test_project_current_flow_forwards_optional_pressure_projector(self) -> None:
        parameter = inspect.signature(_project_current_flow).parameters[
            "pressure_velocity_nullspace_projector"
        ]
        self.assertIsNone(parameter.default)

        function = ast.parse(
            textwrap.dedent(inspect.getsource(_project_current_flow))
        ).body[0]
        project_call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "fluid"
            and node.func.attr == "project"
        )
        keywords = {keyword.arg: keyword.value for keyword in project_call.keywords}
        forwarded = keywords["pressure_velocity_nullspace_projector"]
        self.assertIsInstance(forwarded, ast.Name)
        self.assertEqual(forwarded.id, "pressure_velocity_nullspace_projector")

    def test_sharp_main_and_consistency_share_q_and_pressure_projector(self) -> None:
        function = ast.parse(
            textwrap.dedent(inspect.getsource(_flow_advance_current_step_trial))
        ).body[0]
        projection_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_project_current_flow"
        ]
        self.assertEqual(len(projection_calls), 2)
        for call in projection_calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            for argument_name in (
                "pre_projection_velocity_projector",
                "pressure_velocity_nullspace_projector",
            ):
                value = keywords[argument_name]
                self.assertIsInstance(value, ast.Name)
                self.assertEqual(value.id, "pre_projection_velocity_projector")

    def test_wrapper_defers_report_until_generation_explicit_finalize(self) -> None:
        wrapper, operator, fluid, valid_mask = _prepared_wrapper()
        _prepare_pressure_nullspace(wrapper, fluid, valid_mask)

        input_correction = object()
        output_correction = object()
        apply_result = wrapper.project_pressure_actuated_grid_vector_to_marker_nullspace(
            input_velocity_mps=input_correction,
            output_velocity_mps=output_correction,
            max_iterations=32,
            absolute_tolerance_mps=1.0e-4,
            component_face_valid_mask=valid_mask,
        )

        self.assertEqual(len(operator.prepare_calls), 1)
        self.assertIs(
            operator.prepare_calls[0]["pressure_actuated_component_mobility"],
            fluid.pressure_velocity_actuation_weight,
        )
        self.assertEqual(len(operator.apply_calls), 1)
        self.assertIs(
            operator.apply_calls[0]["input_face_correction"], input_correction
        )
        self.assertIs(
            operator.apply_calls[0]["output_face_correction"], output_correction
        )
        self.assertIsNone(apply_result)
        self.assertEqual(operator.finalize_calls, [])

        report = wrapper.finalize_pressure_nullspace_transaction()
        self.assertEqual(len(operator.finalize_calls), 1)
        self.assertEqual(report["pressure_actuation_generation"], 13)
        self.assertEqual(report["topology_generation"], 7)
        self.assertEqual(report["component_face_valid_mask_generation"], 11)

    def test_pressure_prepare_requires_committed_q(self) -> None:
        wrapper, operator, fluid, valid_mask = _prepared_wrapper()
        operator._phase = "solved"

        with self.assertRaisesRegex(RuntimeError, "committed marker Q"):
            _prepare_pressure_nullspace(wrapper, fluid, valid_mask)

        self.assertEqual(operator.prepare_calls, [])
        self.assertIsNone(wrapper._pressure_nullspace_fluid)

    def test_generation_or_identity_drift_fails_before_explicit_apply(self) -> None:
        mutations = (
            (
                "pressure actuation generation",
                lambda fluid: setattr(
                    fluid, "pressure_velocity_actuation_generation", 14
                ),
                "pressure actuation generation",
            ),
            (
                "pressure actuation identity",
                lambda fluid: setattr(
                    fluid, "pressure_velocity_actuation_weight", object()
                ),
                "pressure actuation weight owner",
            ),
            (
                "topology generation",
                lambda fluid: setattr(fluid, "hibm_reachability_revision", 8),
                "topology generation",
            ),
            (
                "valid-mask generation",
                lambda fluid: setattr(
                    fluid,
                    "velocity_dirichlet_component_ledger_generation",
                    12,
                ),
                "valid-mask generation",
            ),
        )
        for label, mutate, expected_error in mutations:
            with self.subTest(label=label):
                wrapper, operator, fluid, valid_mask = _prepared_wrapper()
                _prepare_pressure_nullspace(wrapper, fluid, valid_mask)
                mutate(fluid)

                with self.assertRaisesRegex(RuntimeError, expected_error):
                    wrapper.project_pressure_actuated_grid_vector_to_marker_nullspace(
                        input_velocity_mps=object(),
                        output_velocity_mps=object(),
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-4,
                        component_face_valid_mask=valid_mask,
                    )

                self.assertEqual(operator.apply_calls, [])

    def test_projection_csv_whitelist_contains_pressure_marker_evidence(self) -> None:
        missing = sorted(
            set(_PRESSURE_MARKER_REPORT_KEYS) - set(FLOW_PROJECTION_REPORT_KEYS)
        )
        self.assertEqual(missing, [])

    def test_joint_qp_report_combines_pressure_marker_evidence_honestly(self) -> None:
        reports = [
            {
                "pressure_marker_nullspace_enabled": True,
                "pressure_marker_nullspace_prepared": True,
                "pressure_marker_nullspace_active_constraint_count": 6,
                "pressure_marker_nullspace_independent_constraint_count": 4,
                "pressure_marker_nullspace_dependent_constraint_count": 1,
                "pressure_marker_nullspace_unactuated_constraint_count": 1,
                "pressure_marker_nullspace_apply_count": 11,
                "pressure_marker_nullspace_pressure_actuation_generation": 3,
                "pressure_marker_nullspace_min_factor_pivot": 0.7,
                "pressure_marker_nullspace_max_dependent_normalized_pivot": 1.0e-14,
                "pressure_marker_nullspace_max_input_constraint_mps": 0.2,
                "pressure_marker_nullspace_max_unactuated_input_constraint_mps": 0.0,
                "pressure_marker_nullspace_max_constraint_residual_mps": 2.0e-13,
                "pressure_marker_nullspace_solver_scratch_resource_bytes": 1024,
                "pressure_marker_nullspace_marker_operator_resource_bytes": 4096,
                "pressure_marker_nullspace_resource_bytes": 5120,
                "pressure_marker_nullspace_actuation_invalid_count": 0,
                "pressure_marker_nullspace_correction_invalid_count": 0,
                "pressure_marker_nullspace_operator_apply_count": 10,
                "pressure_marker_nullspace_velocity_correction_apply_count": 1,
                "pressure_marker_nullspace_all_velocity_paths_projected": True,
            },
            {
                "pressure_marker_nullspace_enabled": True,
                "pressure_marker_nullspace_prepared": True,
                "pressure_marker_nullspace_active_constraint_count": 6,
                "pressure_marker_nullspace_independent_constraint_count": 4,
                "pressure_marker_nullspace_dependent_constraint_count": 1,
                "pressure_marker_nullspace_unactuated_constraint_count": 1,
                "pressure_marker_nullspace_apply_count": 13,
                "pressure_marker_nullspace_pressure_actuation_generation": 4,
                "pressure_marker_nullspace_min_factor_pivot": 0.5,
                "pressure_marker_nullspace_max_dependent_normalized_pivot": 2.0e-14,
                "pressure_marker_nullspace_max_input_constraint_mps": 0.3,
                "pressure_marker_nullspace_max_unactuated_input_constraint_mps": 0.0,
                "pressure_marker_nullspace_max_constraint_residual_mps": 3.0e-13,
                "pressure_marker_nullspace_solver_scratch_resource_bytes": 1024,
                "pressure_marker_nullspace_marker_operator_resource_bytes": 4096,
                "pressure_marker_nullspace_resource_bytes": 5120,
                "pressure_marker_nullspace_actuation_invalid_count": 0,
                "pressure_marker_nullspace_correction_invalid_count": 0,
                "pressure_marker_nullspace_operator_apply_count": 12,
                "pressure_marker_nullspace_velocity_correction_apply_count": 1,
                "pressure_marker_nullspace_all_velocity_paths_projected": True,
            },
        ]

        combined = _combine_flow_projection_reports(reports)

        self.assertTrue(combined["pressure_marker_nullspace_enabled_all"])
        self.assertTrue(combined["pressure_marker_nullspace_prepared_all"])
        self.assertTrue(
            combined["pressure_marker_nullspace_all_velocity_paths_projected_all"]
        )
        self.assertEqual(combined["pressure_marker_nullspace_apply_count"], 24)
        self.assertEqual(
            combined["pressure_marker_nullspace_operator_apply_count"], 22
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_velocity_correction_apply_count"],
            2,
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_pressure_actuation_generation"],
            4,
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_active_constraint_count_min"], 6
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_active_constraint_count_max"], 6
        )
        self.assertEqual(combined["pressure_marker_nullspace_min_factor_pivot"], 0.5)
        self.assertEqual(
            combined["pressure_marker_nullspace_independent_constraint_count"],
            4,
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_dependent_constraint_count"],
            1,
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_unactuated_constraint_count"],
            1,
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_max_dependent_normalized_pivot"],
            2.0e-14,
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_max_constraint_residual_mps"],
            3.0e-13,
        )
        self.assertEqual(
            combined["pressure_marker_nullspace_resource_bytes"], 5120
        )

    def test_joint_qp_report_rejects_invalid_active_factor_pivot(self) -> None:
        for invalid_pivot in (0.0, -1.0, float("nan")):
            with self.subTest(pivot=invalid_pivot):
                report = {
                    "pressure_marker_nullspace_enabled": True,
                    "pressure_marker_nullspace_prepared": True,
                    "pressure_marker_nullspace_active_constraint_count": 6,
                    "pressure_marker_nullspace_min_factor_pivot": invalid_pivot,
                }
                with self.assertRaisesRegex(
                    RuntimeError,
                    "invalid active pressure marker-nullspace factor pivot",
                ):
                    _combine_flow_projection_reports([report])

    def test_joint_qp_report_rejects_zero_rank_for_actuated_rows(self) -> None:
        report = {
            "pressure_marker_nullspace_active_constraint_count": 1,
            "pressure_marker_nullspace_independent_constraint_count": 0,
            "pressure_marker_nullspace_dependent_constraint_count": 1,
            "pressure_marker_nullspace_unactuated_constraint_count": 0,
            "pressure_marker_nullspace_min_factor_pivot": 0.0,
        }
        with self.assertRaisesRegex(RuntimeError, "zero rank|rank.*zero"):
            _combine_flow_projection_reports([report])

    def test_joint_qp_report_rejects_rank_partition_drift(self) -> None:
        reports = [
            {
                "pressure_marker_nullspace_active_constraint_count": 6,
                "pressure_marker_nullspace_independent_constraint_count": 4,
                "pressure_marker_nullspace_dependent_constraint_count": 2,
                "pressure_marker_nullspace_unactuated_constraint_count": 0,
                "pressure_marker_nullspace_min_factor_pivot": 0.5,
            },
            {
                "pressure_marker_nullspace_active_constraint_count": 6,
                "pressure_marker_nullspace_independent_constraint_count": 5,
                "pressure_marker_nullspace_dependent_constraint_count": 0,
                "pressure_marker_nullspace_unactuated_constraint_count": 1,
                "pressure_marker_nullspace_min_factor_pivot": 0.4,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "rank partition.*changed"):
            _combine_flow_projection_reports(reports)

    def test_joint_qp_report_rejects_nonfinite_rank_diagnostic(self) -> None:
        report = {
            "pressure_marker_nullspace_active_constraint_count": 1,
            "pressure_marker_nullspace_independent_constraint_count": 1,
            "pressure_marker_nullspace_dependent_constraint_count": 0,
            "pressure_marker_nullspace_unactuated_constraint_count": 0,
            "pressure_marker_nullspace_min_factor_pivot": 1.0,
            "pressure_marker_nullspace_max_dependent_normalized_pivot": float(
                "nan"
            ),
        }
        with self.assertRaisesRegex(RuntimeError, "finite|non-negative"):
            _combine_flow_projection_reports([report])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
