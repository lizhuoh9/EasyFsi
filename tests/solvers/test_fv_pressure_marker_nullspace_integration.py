"""RED contracts for marker-preserving FV pressure projection.

The marker Q solve and the pressure P solve cannot be made compatible by
projecting the velocity only after pressure has converged.  The fine-grid
pressure operator itself must use the same homogeneous marker-nullspace
projector as the committed pressure velocity correction::

    A_N p = -D N(delta_u_p) + A_interface p

These tests intentionally describe the missing solver-side protocol.  Until
that protocol exists, one API test fails with a compact list of missing
symbols and the integration tests are skipped.  The small NumPy test remains
active as the graded-volume/obstacle reference oracle; it does not launch a
Taichi runtime or a real CG solve.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import unittest
from unittest.mock import MagicMock

import numpy as np
import taichi as ti

from simulation_core import FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids.solver import CartesianFluidSolver


_PROJECTOR_PARAMETER = "pressure_velocity_nullspace_projector"
_FINE_OPERATOR_WRAPPER = "_apply_fv_pressure_operator"
_FINAL_CORRECTION_WRAPPER = "_apply_fv_pressure_velocity_correction"
_PROJECTOR_APPLY_METHOD = (
    "project_pressure_actuated_grid_vector_to_marker_nullspace"
)
_PROJECTOR_FINALIZE_METHOD = "finalize_pressure_nullspace_transaction"

_PRESSURE_MARKER_REPORT_KEYS = {
    "pressure_marker_nullspace_enabled",
    "pressure_marker_nullspace_prepared",
    "pressure_marker_nullspace_active_constraint_count",
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
}


def _signature_parameter_names(callable_object: object) -> set[str]:
    return set(inspect.signature(callable_object).parameters)


def _solver_api_contract_violations() -> list[str]:
    violations: list[str] = []
    project_parameters = inspect.signature(
        CartesianFluidSolver.project
    ).parameters
    projector_parameter = project_parameters.get(_PROJECTOR_PARAMETER)
    if projector_parameter is None:
        violations.append(
            f"CartesianFluidSolver.project missing {_PROJECTOR_PARAMETER}=None"
        )
    elif projector_parameter.default is not None:
        violations.append(
            f"CartesianFluidSolver.project {_PROJECTOR_PARAMETER} must default "
            "to None"
        )

    for method_name in (
        _FINE_OPERATOR_WRAPPER,
        _FINAL_CORRECTION_WRAPPER,
    ):
        method = getattr(CartesianFluidSolver, method_name, None)
        if method is None:
            violations.append(f"CartesianFluidSolver missing {method_name}")
            continue
        if _PROJECTOR_PARAMETER not in _signature_parameter_names(method):
            violations.append(
                f"CartesianFluidSolver.{method_name} missing "
                f"{_PROJECTOR_PARAMETER}"
            )
    return violations


_SOLVER_API_CONTRACT_VIOLATIONS = _solver_api_contract_violations()
_SOLVER_API_READY = not _SOLVER_API_CONTRACT_VIOLATIONS


def _method_source(method_name: str) -> str:
    method = getattr(CartesianFluidSolver, method_name)
    return textwrap.dedent(inspect.getsource(method))


def _wrapper_call_arguments(
    method: object,
    *,
    projector: object | None,
) -> dict[str, object]:
    """Supply inert arguments to a Python orchestration wrapper.

    The integration wrappers are expected to orchestrate kernels and the
    projector; they must not require a full pressure solve merely to exercise
    protocol routing.  MagicMock supplies opaque field-like objects while
    named scalar arguments receive physically harmless values.
    """

    scalar_values: dict[str, object] = {
        "pressure_outlet_zmin": False,
        "outlet": 0,
        "canonical_authority": 0,
        "dt_over_rho": 1.0,
        "pressure_scale": 1.0,
        "max_iterations": 8,
        "absolute_tolerance_mps": 1.0e-12,
    }
    arguments: dict[str, object] = {}
    for name, parameter in inspect.signature(method).parameters.items():
        if name == "self":
            continue
        if name == _PROJECTOR_PARAMETER:
            arguments[name] = projector
        elif name in scalar_values:
            arguments[name] = scalar_values[name]
        elif parameter.default is inspect.Parameter.empty:
            arguments[name] = MagicMock(name=name)
    return arguments


class _CountingPressureVelocityNullspaceProjector:
    """Protocol fake with a fail-closed topology-generation guard."""

    def __init__(
        self,
        *,
        topology_generation: int = 17,
        projector_generation: int = 23,
    ) -> None:
        self.expected_topology_generation = int(topology_generation)
        self.topology_generation = int(topology_generation)
        self.expected_projector_generation = int(projector_generation)
        self.projector_generation = int(projector_generation)
        self.apply_attempt_count = 0
        self.apply_count = 0
        self.output_write_count = 0
        self.finalize_count = 0
        self.finalize_report_overrides: dict[str, object] = {}

    def project_pressure_actuated_grid_vector_to_marker_nullspace(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.apply_attempt_count += 1
        if self.topology_generation != self.expected_topology_generation:
            raise RuntimeError(
                "pressure marker-nullspace topology generation changed"
            )
        if self.projector_generation != self.expected_projector_generation:
            raise RuntimeError(
                "pressure marker-nullspace projector generation changed"
            )
        self.apply_count += 1
        output = kwargs.get("output_velocity_mps")
        source = kwargs.get("input_velocity_mps")
        if isinstance(output, np.ndarray) and isinstance(source, np.ndarray):
            output[...] = source
            self.output_write_count += 1
        return {
            "prepared": True,
            "converged": True,
            "topology_generation": self.topology_generation,
            "projector_generation": self.projector_generation,
        }

    def finalize_pressure_nullspace_transaction(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        del kwargs
        self.finalize_count += 1
        report = {
            "prepared": True,
            "active_constraint_count": 6,
            "independent_constraint_count": 4,
            "dependent_constraint_count": 1,
            "unactuated_constraint_count": 1,
            "apply_count": self.apply_count,
            "pressure_actuation_generation": self.projector_generation,
            "min_factor_pivot": 0.5,
            "max_dependent_normalized_pivot": 1.0e-14,
            "last_max_input_constraint": 0.25,
            "max_unactuated_input_constraint": 0.0,
            "last_max_constraint_residual": 1.0e-13,
            "resource_bytes": 4096,
        }
        return {**report, **self.finalize_report_overrides}


class _DevicePressureVelocityProjector:
    """Small device-only identity or zero projector for solver integration."""

    def __init__(self, *, zero_output: bool) -> None:
        self.max_iterations = 1
        self.absolute_tolerance_mps = 1.0e-12
        self.last_component_face_valid_mask = None
        self.zero_output = bool(zero_output)
        self.apply_count = 0
        self.finalize_count = 0

    def project_pressure_actuated_grid_vector_to_marker_nullspace(
        self,
        *,
        input_velocity_mps,
        output_velocity_mps,
        max_iterations: int,
        absolute_tolerance_mps: float,
        component_face_valid_mask,
    ) -> dict[str, object]:
        del max_iterations, absolute_tolerance_mps, component_face_valid_mask
        if self.zero_output:
            output_velocity_mps.fill(0.0)
        else:
            output_velocity_mps.copy_from(input_velocity_mps)
        self.apply_count += 1
        return {"prepared": True, "converged": True}

    def finalize_pressure_nullspace_transaction(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        del kwargs
        self.finalize_count += 1
        return {
            "prepared": True,
            "active_constraint_count": 1,
            "independent_constraint_count": 1,
            "dependent_constraint_count": 0,
            "unactuated_constraint_count": 0,
            "apply_count": self.apply_count,
            "pressure_actuation_generation": 1,
            "min_factor_pivot": 1.0,
            "max_dependent_normalized_pivot": 0.0,
            "last_max_input_constraint": 0.0,
            "max_unactuated_input_constraint": 0.0,
            "last_max_constraint_residual": 0.0,
            "resource_bytes": 4096,
        }


class FvPressureMarkerNullspaceApiRedTests(unittest.TestCase):
    def test_solver_exposes_optional_marker_nullspace_pressure_protocol(
        self,
    ) -> None:
        self.assertEqual(
            _SOLVER_API_CONTRACT_VIOLATIONS,
            [],
            msg=(
                "FV pressure/marker-nullspace integration API is not "
                "implemented: "
                + "; ".join(_SOLVER_API_CONTRACT_VIOLATIONS)
            ),
        )


@unittest.skipUnless(
    _SOLVER_API_READY,
    "FV pressure/marker-nullspace integration API is not implemented",
)
class FvPressureMarkerNullspaceStaticIntegrationTests(unittest.TestCase):
    def test_every_fine_grid_cg_matvec_uses_the_same_operator_wrapper(
        self,
    ) -> None:
        # Jacobi/MG may remain a base-A0 preconditioner.  This contract is
        # intentionally limited to the exact fine-grid operator used by the
        # initial residual, PCG iteration, both restart paths, BiCGSTAB, and
        # final exact-residual confirmation.
        methods = (
            "_solve_pressure_poisson_fv_cg",
            "_continue_pressure_poisson_fv_bicgstab",
            "_confirm_pressure_poisson_fv_cg_exact_residual",
        )
        for method_name in methods:
            with self.subTest(method=method_name):
                source = _method_source(method_name)
                self.assertIn(
                    f"self.{_FINE_OPERATOR_WRAPPER}(",
                    source,
                    msg=f"{method_name} bypasses the constrained fine operator",
                )
                self.assertNotIn(
                    "self._fv_laplacian_apply_kernel(",
                    source,
                    msg=(
                        f"{method_name} still has a raw fine-grid matvec; "
                        "restart/fallback paths must not bypass N"
                    ),
                )

    def test_pcg_initial_iteration_and_restart_sites_are_not_collapsed_away(
        self,
    ) -> None:
        source = _method_source("_solve_pressure_poisson_fv_cg")
        wrapper_call = f"self.{_FINE_OPERATOR_WRAPPER}("
        self.assertGreaterEqual(
            source.count(wrapper_call),
            4,
            msg=(
                "FV-CG must route its initial residual, iteration matvec, "
                "device-breakdown restart, and periodic restart through A_N"
            ),
        )

    def test_final_velocity_correction_uses_the_projector_not_only_matvec(
        self,
    ) -> None:
        project_source = _method_source("project")
        correction_source = _method_source(_FINAL_CORRECTION_WRAPPER)
        self.assertIn(
            f"self.{_FINAL_CORRECTION_WRAPPER}(",
            project_source,
        )
        self.assertNotIn(
            "self._subtract_pressure_gradient_kernel(",
            project_source,
            msg=(
                "project() still commits an unprojected pressure correction; "
                "all primary and cleanup corrections must use the wrapper"
            ),
        )
        self.assertIn(_PROJECTOR_APPLY_METHOD, correction_source)
        self.assertIn(_PROJECTOR_PARAMETER, correction_source)

    def test_project_threads_one_projector_to_solve_and_commit(self) -> None:
        source = _method_source("project")
        self.assertGreaterEqual(
            source.count(_PROJECTOR_PARAMETER),
            3,
            msg=(
                "project() must accept, route to the fine operator, and route "
                "to the final correction the same projector object"
            ),
        )

    def test_matvec_is_device_only_and_final_correction_audits_before_commit(
        self,
    ) -> None:
        operator_source = _method_source(_FINE_OPERATOR_WRAPPER)
        correction_source = _method_source(_FINAL_CORRECTION_WRAPPER)
        self.assertNotIn(
            "_require_finite_fv_pressure_velocity_correction",
            operator_source,
            msg="every FV-CG matvec still performs a host invalid-count read",
        )
        self.assertNotIn(
            _PROJECTOR_FINALIZE_METHOD,
            operator_source,
            msg="every FV-CG matvec still performs a host finalization/report",
        )
        self.assertIn(_PROJECTOR_FINALIZE_METHOD, correction_source)
        self.assertLess(
            correction_source.index(_PROJECTOR_FINALIZE_METHOD),
            correction_source.index(
                "_subtract_projected_pressure_velocity_correction_kernel"
            ),
            msg="pressure nullspace must fail closed before fluid.velocity write",
        )

    def test_invalid_correction_audit_is_reset_once_per_prepared_transaction(
        self,
    ) -> None:
        build_source = _method_source(
            "_build_fv_pressure_velocity_correction_kernel"
        )
        prepare_source = _method_source(
            "_prepare_pressure_velocity_actuation_weight_kernel"
        )
        reset = "pressure_velocity_correction_invalid_count[None] = 0"
        self.assertNotIn(
            reset,
            build_source,
            msg="every fine-grid matvec still resets a solve-wide device audit",
        )
        self.assertIn(
            reset,
            prepare_source,
            msg="a new prepared pressure transaction must reset its audit",
        )

    def test_project_report_exposes_pressure_marker_nullspace_evidence(self) -> None:
        source = _method_source("project")
        string_literals = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        missing = sorted(
            key for key in _PRESSURE_MARKER_REPORT_KEYS if key not in string_literals
        )
        self.assertEqual(
            missing,
            [],
            msg="project report is missing pressure-marker evidence: "
            + ", ".join(missing),
        )


@unittest.skipUnless(
    _SOLVER_API_READY,
    "FV pressure/marker-nullspace integration API is not implemented",
)
class FvPressureMarkerNullspaceProtocolRoutingTests(unittest.TestCase):
    def _invoke_wrapper(
        self,
        method_name: str,
        *,
        solver: MagicMock,
        projector: object | None,
    ) -> None:
        method = getattr(CartesianFluidSolver, method_name)
        arguments = _wrapper_call_arguments(method, projector=projector)
        method(solver, **arguments)

    def test_none_path_uses_legacy_operator_and_correction_without_nullspace_work(
        self,
    ) -> None:
        operator_solver = MagicMock(name="operator_solver")
        self._invoke_wrapper(
            _FINE_OPERATOR_WRAPPER,
            solver=operator_solver,
            projector=None,
        )
        operator_solver._fv_laplacian_apply_kernel.assert_called_once()
        self.assertFalse(
            any(
                "nullspace" in str(call).lower()
                for call in operator_solver.method_calls
            ),
            msg="projector=None must add no marker-nullspace calls",
        )

        correction_solver = MagicMock(name="correction_solver")
        self._invoke_wrapper(
            _FINAL_CORRECTION_WRAPPER,
            solver=correction_solver,
            projector=None,
        )
        correction_solver._subtract_pressure_gradient_kernel.assert_called_once()
        self.assertFalse(
            any(
                "nullspace" in str(call).lower()
                for call in correction_solver.method_calls
            ),
            msg="projector=None must preserve the non-HIBM fast path",
        )

    def test_matvec_and_final_correction_call_the_same_projector(self) -> None:
        projector = _CountingPressureVelocityNullspaceProjector()
        self._invoke_wrapper(
            _FINE_OPERATOR_WRAPPER,
            solver=MagicMock(name="operator_solver"),
            projector=projector,
        )
        self._invoke_wrapper(
            _FINAL_CORRECTION_WRAPPER,
            solver=MagicMock(name="correction_solver"),
            projector=projector,
        )

        self.assertEqual(projector.apply_count, 2)
        self.assertEqual(projector.apply_attempt_count, 2)
        self.assertEqual(projector.finalize_count, 1)

    def test_matvec_defers_host_finalize_until_velocity_commit(self) -> None:
        projector = _CountingPressureVelocityNullspaceProjector()
        self._invoke_wrapper(
            _FINE_OPERATOR_WRAPPER,
            solver=MagicMock(name="operator_solver"),
            projector=projector,
        )
        self.assertEqual(projector.apply_count, 1)
        self.assertEqual(projector.finalize_count, 0)

        correction_solver = MagicMock(name="correction_solver")
        report = CartesianFluidSolver._apply_fv_pressure_velocity_correction(
            correction_solver,
            **_wrapper_call_arguments(
                CartesianFluidSolver._apply_fv_pressure_velocity_correction,
                projector=projector,
            ),
        )
        self.assertEqual(projector.apply_count, 2)
        self.assertEqual(projector.finalize_count, 1)
        self.assertTrue(report["pressure_marker_nullspace_prepared"])
        self.assertEqual(report["pressure_marker_nullspace_apply_count"], 2)

    def test_finalize_rejects_zero_rank_when_actuated_rows_exist(self) -> None:
        projector = _CountingPressureVelocityNullspaceProjector()
        projector.finalize_report_overrides = {
            "active_constraint_count": 1,
            "independent_constraint_count": 0,
            "dependent_constraint_count": 1,
            "unactuated_constraint_count": 0,
            "min_factor_pivot": 0.0,
        }
        solver = MagicMock(name="solver")
        solver.pressure_velocity_actuation_invalid_count = {None: 0}
        solver.pressure_velocity_correction_invalid_count = {None: 0}
        solver._pressure_marker_nullspace_operator_apply_count = 0
        solver._pressure_marker_nullspace_velocity_correction_apply_count = 0
        solver._pressure_velocity_nullspace_transaction_active = False
        solver._pressure_velocity_nullspace_resource_bytes = 0

        with self.assertRaisesRegex(RuntimeError, "zero rank|rank.*zero"):
            CartesianFluidSolver._finalize_pressure_marker_nullspace_projection(
                solver,
                projector,
            )

    def test_topology_generation_mutation_propagates_before_projector_output_write(
        self,
    ) -> None:
        projector = _CountingPressureVelocityNullspaceProjector()
        solver = MagicMock(name="solver")
        self._invoke_wrapper(
            _FINE_OPERATOR_WRAPPER,
            solver=solver,
            projector=projector,
        )
        successful_writes = projector.output_write_count
        projector.topology_generation += 1

        with self.assertRaisesRegex(RuntimeError, "topology generation changed"):
            self._invoke_wrapper(
                _FINE_OPERATOR_WRAPPER,
                solver=solver,
                projector=projector,
            )

        self.assertEqual(projector.apply_count, 1)
        self.assertEqual(projector.apply_attempt_count, 2)
        self.assertEqual(projector.output_write_count, successful_writes)

    def test_generation_mutation_fails_before_final_velocity_commit(self) -> None:
        projector = _CountingPressureVelocityNullspaceProjector()
        projector.projector_generation += 1
        solver = MagicMock(name="solver")

        with self.assertRaisesRegex(RuntimeError, "projector generation changed"):
            self._invoke_wrapper(
                _FINAL_CORRECTION_WRAPPER,
                solver=solver,
                projector=projector,
            )

        self.assertEqual(projector.apply_count, 0)
        self.assertFalse(
            any(
                "commit" in str(call).lower()
                or "subtract_pressure_gradient" in str(call).lower()
                for call in solver.method_calls
            ),
            msg="stale projector input must fail before physical velocity write",
        )


@unittest.skipUnless(
    _SOLVER_API_READY,
    "FV pressure/marker-nullspace integration API is not implemented",
)
class FvPressureMarkerNullspaceTaichiKernelIntegrationTests(unittest.TestCase):
    """Exercise the real fine-grid kernels on one tiny GPU-resident grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid_nodes = (4, 4, 4)
        cls.solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=cls.grid_nodes,
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        i, j, k = np.indices(cls.grid_nodes, dtype=np.float64)
        cls.pressure = (
            0.37 * i * i
            - 0.23 * j
            + 0.19 * k * k
            + 0.11 * i * j
            - 0.07 * j * k
        ).astype(np.float64)
        cls.initial_velocity = np.empty(cls.grid_nodes + (3,), dtype=np.float32)
        cls.initial_velocity[..., 0] = (0.03 * i - 0.02 * j + 0.01 * k)
        cls.initial_velocity[..., 1] = (-0.01 * i + 0.04 * j - 0.02 * k)
        cls.initial_velocity[..., 2] = (0.02 * i + 0.01 * j - 0.03 * k)
        cls.obstacle = np.zeros(cls.grid_nodes, dtype=np.int32)
        cls.canonical_authority = (
            cls.solver._velocity_dirichlet_boundary_authority_code()
        )
        cls.dt_over_rho = 0.125

    def _reset_physical_fields(self) -> None:
        self.solver.obstacle.from_numpy(self.obstacle)
        self.solver.pressure.from_numpy(self.pressure)
        self.solver.velocity.from_numpy(self.initial_velocity)

    def test_00_none_path_does_not_allocate_pressure_nullspace_resources(
        self,
    ) -> None:
        self._reset_physical_fields()
        self.assertFalse(
            self.solver._pressure_velocity_nullspace_resources_allocated
        )
        self.assertEqual(
            self.solver._pressure_velocity_nullspace_resource_bytes,
            0,
        )
        self.assertIsNone(self.solver.pressure_velocity_actuation_weight)
        self.assertIsNone(self.solver.pressure_velocity_raw_correction)
        self.assertIsNone(self.solver.pressure_velocity_projected_correction)

        self.solver._apply_fv_pressure_operator(
            self.solver.pressure,
            self.solver.cg_Ad,
            0,
            self.canonical_authority,
            None,
        )
        self.solver._apply_fv_pressure_velocity_correction(
            self.dt_over_rho,
            0,
            self.canonical_authority,
            None,
        )

        self.assertFalse(
            self.solver._pressure_velocity_nullspace_resources_allocated
        )
        self.assertEqual(
            self.solver._pressure_velocity_nullspace_resource_bytes,
            0,
        )
        self.assertEqual(self.solver.pressure_velocity_actuation_generation, 0)
        self.assertIsNone(self.solver.pressure_velocity_actuation_weight)
        self.assertIsNone(self.solver.pressure_velocity_raw_correction)
        self.assertIsNone(self.solver.pressure_velocity_projected_correction)

    def test_10_identity_projector_matches_legacy_operator_and_commit(
        self,
    ) -> None:
        self._reset_physical_fields()
        self.solver._fv_laplacian_apply_kernel(
            self.solver.pressure,
            self.solver.cg_Ad,
            0,
            self.canonical_authority,
        )
        legacy_operator = self.solver.cg_Ad.to_numpy()
        self.solver._subtract_pressure_gradient_kernel(
            self.dt_over_rho,
            0,
            self.canonical_authority,
        )
        legacy_velocity = self.solver.velocity.to_numpy()

        self.solver.velocity.from_numpy(self.initial_velocity)
        identity_projector = _DevicePressureVelocityProjector(
            zero_output=False
        )
        self.solver._apply_fv_pressure_operator(
            self.solver.pressure,
            self.solver.cg_r,
            0,
            self.canonical_authority,
            identity_projector,
        )
        projected_operator = self.solver.cg_r.to_numpy()
        pressure_marker_report = self.solver._apply_fv_pressure_velocity_correction(
            self.dt_over_rho,
            0,
            self.canonical_authority,
            identity_projector,
        )
        projected_velocity = self.solver.velocity.to_numpy()

        np.testing.assert_allclose(
            projected_operator,
            legacy_operator,
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            projected_velocity,
            legacy_velocity,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(identity_projector.apply_count, 2)
        self.assertEqual(identity_projector.finalize_count, 1)
        self.assertTrue(
            self.solver._pressure_velocity_nullspace_resources_allocated
        )
        for field in (
            self.solver.pressure_velocity_actuation_weight,
            self.solver.pressure_velocity_raw_correction,
            self.solver.pressure_velocity_projected_correction,
        ):
            self.assertIsNotNone(field)
            self.assertEqual(field.dtype, ti.f64)
            self.assertEqual(tuple(field.shape), self.grid_nodes)
        expected_bytes = int(np.prod(self.grid_nodes)) * 3 * 3 * 8 + 2 * 4
        self.assertEqual(
            self.solver._pressure_velocity_nullspace_resource_bytes,
            expected_bytes,
        )
        self.assertEqual(
            pressure_marker_report[
                "pressure_marker_nullspace_solver_scratch_resource_bytes"
            ],
            expected_bytes,
        )
        self.assertEqual(
            pressure_marker_report[
                "pressure_marker_nullspace_marker_operator_resource_bytes"
            ],
            4096,
        )
        self.assertEqual(
            pressure_marker_report["pressure_marker_nullspace_resource_bytes"],
            expected_bytes + 4096,
        )

    def test_20_zero_projector_cancels_standard_operator_and_commit(
        self,
    ) -> None:
        self._reset_physical_fields()
        self.assertEqual(
            int(self.solver.pressure_interface_row_count[None]),
            0,
        )
        np.testing.assert_array_equal(
            self.solver.pressure_interface_matrix_diagonal.to_numpy(),
            np.zeros(self.grid_nodes, dtype=np.float64),
        )
        zero_projector = _DevicePressureVelocityProjector(zero_output=True)

        self.solver._apply_fv_pressure_operator(
            self.solver.pressure,
            self.solver.cg_r,
            0,
            self.canonical_authority,
            zero_projector,
        )
        constrained_operator = self.solver.cg_r.to_numpy()
        self.solver._apply_fv_pressure_velocity_correction(
            self.dt_over_rho,
            0,
            self.canonical_authority,
            zero_projector,
        )
        constrained_velocity = self.solver.velocity.to_numpy()

        np.testing.assert_allclose(
            constrained_operator,
            np.zeros(self.grid_nodes, dtype=np.float64),
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_array_equal(
            constrained_velocity,
            self.initial_velocity,
        )
        self.assertEqual(zero_projector.apply_count, 2)
        self.assertEqual(zero_projector.finalize_count, 1)


class FvPressureMarkerNullspaceGradedObstacleReferenceTests(unittest.TestCase):
    @staticmethod
    def _reference_operators() -> dict[str, np.ndarray]:
        # Four cells with unequal volumes; cell 2 is an obstacle and therefore
        # has no incidence or interface row.  B stores oriented face areas.
        volume = np.diag(np.asarray((1.0, 1.7, 0.8, 2.3), dtype=np.float64))
        obstacle = np.asarray((False, False, True, False))
        incidence = np.asarray(
            (
                (-1.0, 0.0, -0.6),
                (1.0, -0.75, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.75, 0.6),
            ),
            dtype=np.float64,
        )
        pressure_actuation = np.diag(
            np.asarray((0.7, 1.3, 0.9), dtype=np.float64)
        )
        marker_interpolation = np.asarray(
            ((0.4, -0.2, 0.7),),
            dtype=np.float64,
        )
        schur = marker_interpolation @ pressure_actuation @ marker_interpolation.T
        nullspace = (
            np.eye(3, dtype=np.float64)
            - pressure_actuation
            @ marker_interpolation.T
            @ np.linalg.inv(schur)
            @ marker_interpolation
        )

        # A symmetric integrated interface row between active cells 0 and 3.
        interface_integrated = np.zeros((4, 4), dtype=np.float64)
        transmissibility = 0.31
        interface_integrated[0, 0] += transmissibility
        interface_integrated[3, 3] += transmissibility
        interface_integrated[0, 3] -= transmissibility
        interface_integrated[3, 0] -= transmissibility

        inverse_volume = np.linalg.inv(volume)
        divergence = inverse_volume @ incidence
        interface_operator = inverse_volume @ interface_integrated
        operator = inverse_volume @ (
            incidence @ nullspace @ pressure_actuation @ incidence.T
            + interface_integrated
        )
        return {
            "volume": volume,
            "obstacle": obstacle,
            "incidence": incidence,
            "pressure_actuation": pressure_actuation,
            "marker_interpolation": marker_interpolation,
            "nullspace": nullspace,
            "divergence": divergence,
            "interface_operator": interface_operator,
            "operator": operator,
        }

    def test_graded_obstacle_operator_is_volume_symmetric_and_matches_commit(
        self,
    ) -> None:
        operators = self._reference_operators()
        volume = operators["volume"]
        obstacle = operators["obstacle"]
        incidence = operators["incidence"]
        pressure_actuation = operators["pressure_actuation"]
        marker_interpolation = operators["marker_interpolation"]
        nullspace = operators["nullspace"]
        divergence = operators["divergence"]
        interface_operator = operators["interface_operator"]
        operator = operators["operator"]

        p = np.asarray((0.3, -0.7, 11.0, 1.2), dtype=np.float64)
        q = np.asarray((-0.4, 0.9, -13.0, 0.2), dtype=np.float64)
        p_aq = float(p @ volume @ operator @ q)
        q_ap = float(q @ volume @ operator @ p)
        self.assertAlmostEqual(p_aq, q_ap, delta=2.0e-14)

        raw_pressure_velocity_correction = (
            -pressure_actuation @ incidence.T @ q
        )
        projected_pressure_velocity_correction = (
            nullspace @ raw_pressure_velocity_correction
        )
        committed_matvec = (
            -divergence @ projected_pressure_velocity_correction
            + interface_operator @ q
        )
        np.testing.assert_allclose(
            operator @ q,
            committed_matvec,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            marker_interpolation @ projected_pressure_velocity_correction,
            0.0,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(operator[obstacle], 0.0, atol=0.0)
        np.testing.assert_allclose(operator[:, obstacle], 0.0, atol=0.0)
        self.assertGreater(np.ptp(np.diag(volume)), 0.0)


if __name__ == "__main__":
    unittest.main()
