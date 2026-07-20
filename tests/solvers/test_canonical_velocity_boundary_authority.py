from __future__ import annotations

import ast
import inspect
import re
import unittest
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Callable

from simulation_core.fluids import preflow_snapshot as preflow_snapshot_module
from simulation_core.fluids.solver import CartesianFluidSolver


_CANONICAL_LEDGER_FIELDS = frozenset(
    {
        "velocity_dirichlet_boundary_active_component_mask",
        "velocity_dirichlet_boundary_value_mps",
        "velocity_dirichlet_boundary_pressure_mobility",
        "velocity_dirichlet_boundary_component_enforcement_weight",
        "velocity_dirichlet_boundary_component_region_id",
        "velocity_dirichlet_boundary_hard_fixed_component_mask",
        "velocity_dirichlet_boundary_external_exact_component_mask",
        "velocity_dirichlet_boundary_owned_component_mask",
    }
)
_SNAPSHOT_LEDGER_METADATA = frozenset(
    {
        "velocity_dirichlet_boundary_authority",
        "velocity_dirichlet_component_ledger_generation",
    }
)
_PUBLIC_CONSUMERS = frozenset({"apply", "projection", "reference", "snapshot"})
_PHYSICAL_CONSUMERS = frozenset(
    {
        *_PUBLIC_CONSUMERS,
        "divergence",
        "reachability",
        "fv_operator",
        "gradient",
        "multigrid",
        "no_slip",
    }
)
_SEALED_GUARD = "_require_velocity_dirichlet_component_ledger_sealed"


def _host_only_solver(
    *,
    authority: str = "legacy",
    generation: int = 0,
    sealed: bool = False,
) -> CartesianFluidSolver:
    """Make a host-only fixture without initializing Taichi or allocating fields."""

    solver = object.__new__(CartesianFluidSolver)
    solver.velocity_dirichlet_boundary_authority = authority
    solver.velocity_dirichlet_component_ledger_generation = generation
    solver.velocity_dirichlet_component_ledger_sealed = sealed
    solver.velocity_dirichlet_face_symmetric = 0
    solver._velocity_dirichlet_component_ledger_consumer_generations = {}
    solver._velocity_dirichlet_component_ledger_consumer_capabilities = {}
    return solver


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == function_name
        ):
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise AssertionError(f"could not extract {function_name!r} from {path}")
            return segment
    raise AssertionError(f"function {function_name!r} is missing from {path}")


def _taichi_vector_pattern(values: tuple[str, ...]) -> str:
    return (
        r"ti\.Vector\(\s*\[\s*"
        + r"\s*,\s*".join(re.escape(value) for value in values)
        + r"\s*\]\s*\)"
    )


class CanonicalVelocityBoundaryAuthorityContracts(unittest.TestCase):
    """RED contracts for switching the component ledger to production authority."""

    def test_authority_accepts_only_legacy_or_canonical(self) -> None:
        solver = _host_only_solver()

        for authority in ("legacy", "canonical"):
            with self.subTest(authority=authority):
                solver.set_velocity_dirichlet_boundary_authority(authority)
                self.assertEqual(
                    solver.velocity_dirichlet_boundary_authority,
                    authority,
                )

        for invalid in (None, "", "CANONICAL", "hybrid", 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    solver.set_velocity_dirichlet_boundary_authority(invalid)

    def test_authority_switch_advances_generation_and_invalidates_seal(self) -> None:
        solver = _host_only_solver(generation=9, sealed=True)
        solver._velocity_dirichlet_component_ledger_consumer_generations = {
            "apply": 9
        }
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = {
            "apply": object()
        }

        solver.set_velocity_dirichlet_boundary_authority("canonical")

        self.assertEqual(solver.velocity_dirichlet_boundary_authority, "canonical")
        self.assertGreater(solver.velocity_dirichlet_component_ledger_generation, 9)
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)
        self.assertEqual(
            solver._velocity_dirichlet_component_ledger_consumer_generations,
            {},
        )
        self.assertEqual(
            solver._velocity_dirichlet_component_ledger_consumer_capabilities,
            {},
        )

    def test_canonical_unsealed_apply_and_reference_fail_before_any_kernel(self) -> None:
        entries: tuple[tuple[str, Callable[[CartesianFluidSolver], object]], ...] = (
            (
                "apply",
                lambda solver: solver.apply_velocity_dirichlet_boundary_rows(
                    read_report=False
                ),
            ),
            (
                "reference",
                lambda solver: solver.capture_velocity_dirichlet_boundary_ledger_reference(),
            ),
        )

        for entry_name, invoke in entries:
            with self.subTest(entry=entry_name):
                solver = _host_only_solver(authority="canonical", generation=7)
                kernel_calls: list[str] = []
                solver._apply_velocity_dirichlet_boundary_rows_kernel = (
                    lambda *_args: kernel_calls.append("apply")
                )
                solver._capture_velocity_dirichlet_boundary_ledger_reference_kernel = (
                    lambda: kernel_calls.append("reference")
                )
                solver._velocity_dirichlet_ledger_reference_generation = 0
                solver._velocity_dirichlet_ledger_reference_valid = False

                with self.assertRaisesRegex(RuntimeError, "canonical|sealed"):
                    invoke(solver)
                self.assertEqual(kernel_calls, [])

    def test_canonical_unsealed_projection_and_snapshot_capture_guard_first(self) -> None:
        project_source = inspect.getsource(CartesianFluidSolver.project)
        self.assertIn(_SEALED_GUARD, project_source)
        self.assertLess(
            project_source.index(_SEALED_GUARD),
            project_source.index("_compute_divergence_with_topology_mode"),
        )

        repository_root = Path(inspect.getfile(CartesianFluidSolver)).resolve().parents[2]
        runner_path = (
            repository_root / "benchmarks" / "official" / "solid_mpm_fsi_runner.py"
        )
        capture_source = _function_source(
            runner_path,
            "_capture_preflow_snapshot_fields",
        )
        self.assertIn(_SEALED_GUARD, capture_source)
        self.assertLess(
            capture_source.index(_SEALED_GUARD),
            capture_source.index("PREFLOW_SNAPSHOT_FIELD_NAMES"),
        )

    def test_canonical_snapshot_restore_preflights_then_prepares_before_seal(
        self,
    ) -> None:
        repository_root = (
            Path(inspect.getfile(CartesianFluidSolver)).resolve().parents[2]
        )
        runner_path = (
            repository_root / "benchmarks" / "official" / "solid_mpm_fsi_runner.py"
        )
        restore_source = _function_source(
            runner_path,
            "_restore_preflow_snapshot_fields",
        )
        prepare_plan = "_canonical_snapshot_restore_prepare_plan"
        first_commit = "runtime_fields[name].from_numpy"
        self.assertIn(prepare_plan, restore_source)
        self.assertLess(
            restore_source.index(prepare_plan),
            restore_source.index(first_commit),
        )

        prepare_source = _function_source(
            runner_path,
            "_prepare_canonical_preflow_snapshot_restore",
        )
        no_slip_prepare = "prepare_hibm_no_slip_component_face_valid_mask"
        seal = "seal_velocity_dirichlet_component_ledger"
        require = "_require_velocity_dirichlet_component_ledger_sealed"
        self.assertIn(no_slip_prepare, prepare_source)
        self.assertNotIn("build_hibm_no_slip_component_face_valid_mask", prepare_source)
        self.assertLess(
            prepare_source.index(no_slip_prepare),
            prepare_source.index(seal),
        )
        self.assertLess(prepare_source.index(seal), prepare_source.index(require))

        # The except path restores direct backups only.  It must not call the
        # prepare/seal orchestration a second time and obscure the first error.
        self.assertEqual(
            restore_source.count("_rollback_preflow_snapshot_restore"),
            1,
        )

    def test_clear_resets_every_canonical_field_and_invalidates_generation(self) -> None:
        clear_kernel_source = inspect.getsource(
            CartesianFluidSolver._clear_velocity_dirichlet_boundary_rows_kernel
        )
        for field_name in _CANONICAL_LEDGER_FIELDS:
            with self.subTest(field=field_name):
                self.assertIn(field_name, clear_kernel_source)

        index = r"\[\s*i\s*,\s*j\s*,\s*k\s*\]"
        neutral_assignments = (
            rf"velocity_dirichlet_boundary_active_component_mask{index}\s*=\s*0",
            rf"velocity_dirichlet_boundary_value_mps{index}\s*=\s*"
            rf"{_taichi_vector_pattern(('0.0',) * 3)}",
            rf"velocity_dirichlet_boundary_pressure_mobility{index}\s*=\s*"
            rf"{_taichi_vector_pattern(('1.0',) * 3)}",
            rf"velocity_dirichlet_boundary_component_enforcement_weight{index}\s*=\s*"
            rf"{_taichi_vector_pattern(('0.0',) * 3)}",
            rf"velocity_dirichlet_boundary_component_region_id{index}\s*=\s*"
            rf"{_taichi_vector_pattern(('-1',) * 3)}",
            rf"velocity_dirichlet_boundary_hard_fixed_component_mask{index}\s*=\s*0",
            rf"velocity_dirichlet_boundary_external_exact_component_mask{index}\s*=\s*0",
            rf"velocity_dirichlet_boundary_owned_component_mask{index}\s*=\s*0",
        )
        for assignment_pattern in neutral_assignments:
            with self.subTest(assignment=assignment_pattern):
                self.assertRegex(clear_kernel_source, assignment_pattern)

        solver = _host_only_solver(authority="canonical", generation=11, sealed=True)
        solver._clear_velocity_dirichlet_boundary_rows_kernel = lambda: None
        solver._invalidate_hibm_pressure_reachability = lambda: None
        solver.clear_velocity_dirichlet_boundary_rows()

        self.assertGreater(solver.velocity_dirichlet_component_ledger_generation, 11)
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)

    def test_seal_requires_real_migrated_capabilities_not_forged_dictionaries(self) -> None:
        seal_method = CartesianFluidSolver.seal_velocity_dirichlet_component_ledger
        self.assertEqual(tuple(inspect.signature(seal_method).parameters), ("self",))
        required_consumers = frozenset(
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS
        )
        self.assertEqual(required_consumers, _PHYSICAL_CONSUMERS)

        register_method = (
            CartesianFluidSolver._register_velocity_dirichlet_component_ledger_consumer_generation
        )
        self.assertEqual(
            tuple(inspect.signature(register_method).parameters),
            ("self", "consumer", "capability"),
        )

        generation = 13
        solver = _host_only_solver(authority="canonical", generation=generation)
        with self.assertRaises(TypeError):
            register_method(
                solver,
                "apply",
                generation=generation,
            )
        with self.assertRaisesRegex(RuntimeError, "capability|migrated"):
            register_method(solver, "apply", capability=object())

        # Directly filling the generation map cannot impersonate a migrated
        # consumer: sealing also verifies opaque class-owned capabilities.
        solver._velocity_dirichlet_component_ledger_consumer_generations = {
            consumer: generation for consumer in required_consumers
        }
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = {
            consumer: object() for consumer in required_consumers
        }
        with self.assertRaisesRegex(RuntimeError, "capabilit|consumer"):
            solver.seal_velocity_dirichlet_component_ledger()
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)

    def test_marker_feedback_legacy_writer_guards_then_invalidates_generation(self) -> None:
        canonical = _host_only_solver(authority="canonical", generation=3)
        kernel_calls: list[str] = []
        canonical._clear_marker_feedback_constraints_kernel = (
            lambda: kernel_calls.append("clear")
        )
        with self.assertRaisesRegex(RuntimeError, "canonical|legacy"):
            canonical.apply_marker_feedback_constraints(
                object(),
                object(),
                object(),
                0,
                feedback_available=False,
                preserve_velocity_constraints=False,
                primary_region_id=1,
                secondary_region_id=2,
            )
        self.assertEqual(kernel_calls, [])

        legacy = _host_only_solver(authority="legacy", generation=4, sealed=True)
        legacy._clear_marker_feedback_constraints_kernel = lambda: None
        legacy._invalidate_hibm_pressure_reachability = lambda: None
        legacy.marker_feedback_constraint_report = lambda **_kwargs: {}
        legacy.apply_marker_feedback_constraints(
            object(),
            object(),
            object(),
            0,
            feedback_available=False,
            preserve_velocity_constraints=False,
            primary_region_id=1,
            secondary_region_id=2,
        )
        self.assertGreater(legacy.velocity_dirichlet_component_ledger_generation, 4)
        self.assertFalse(legacy.velocity_dirichlet_component_ledger_sealed)

    def test_canonical_forbids_legacy_writer_and_face_symmetric_path(self) -> None:
        writer_source = inspect.getsource(CartesianFluidSolver.refresh_zmax_inlet_boundary)
        legacy_guard = "_require_legacy_velocity_dirichlet_boundary_authority"
        self.assertIn(legacy_guard, writer_source)
        self.assertLess(
            writer_source.index(legacy_guard),
            writer_source.index("_refresh_zmax_inlet_boundary_kernel"),
        )

        solver = _host_only_solver(authority="legacy")
        solver.velocity_dirichlet_face_symmetric = 1
        with self.assertRaisesRegex((RuntimeError, ValueError), "canonical|face_symmetric"):
            solver.set_velocity_dirichlet_boundary_authority("canonical")

        solver = _host_only_solver(authority="canonical", generation=5, sealed=True)
        solver.velocity_dirichlet_face_symmetric = 2
        solver._apply_velocity_dirichlet_boundary_rows_kernel = lambda *_args: None
        with self.assertRaisesRegex((RuntimeError, ValueError), "canonical|face_symmetric"):
            solver.apply_velocity_dirichlet_boundary_rows(read_report=False)

    def test_snapshot_schema_persists_authority_generation_and_eight_fields(self) -> None:
        self.assertGreaterEqual(
            preflow_snapshot_module.PREFLOW_SNAPSHOT_SCHEMA_VERSION,
            6,
        )
        self.assertTrue(
            _CANONICAL_LEDGER_FIELDS.issubset(
                preflow_snapshot_module.PREFLOW_SNAPSHOT_FIELD_NAMES
            )
        )
        self.assertTrue(
            _SNAPSHOT_LEDGER_METADATA.issubset(
                preflow_snapshot_module._MANIFEST_FIELD_NAMES
            )
        )
        snapshot_attributes = {
            field.name
            for field in dataclass_fields(preflow_snapshot_module.PreflowSnapshot)
        }
        self.assertTrue(_SNAPSHOT_LEDGER_METADATA.issubset(snapshot_attributes))


if __name__ == "__main__":
    unittest.main()
