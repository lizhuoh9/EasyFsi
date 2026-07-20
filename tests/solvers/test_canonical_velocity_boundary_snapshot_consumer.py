from __future__ import annotations

import ast
import inspect
import unittest
from dataclasses import fields as dataclass_fields
from pathlib import Path

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


def _host_only_solver(
    *,
    authority: str,
    generation: int,
) -> CartesianFluidSolver:
    solver = object.__new__(CartesianFluidSolver)
    solver.velocity_dirichlet_boundary_authority = authority
    solver.velocity_dirichlet_component_ledger_generation = generation
    solver.velocity_dirichlet_component_ledger_sealed = False
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
                raise AssertionError(f"could not extract {function_name!r}")
            return segment
    raise AssertionError(f"function {function_name!r} is missing from {path}")


class CanonicalVelocityBoundarySnapshotConsumerContracts(unittest.TestCase):
    def test_snapshot_prepare_uses_class_owned_capability_and_current_generation(
        self,
    ) -> None:
        capability = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_SNAPSHOT_CAPABILITY
        )
        self.assertIs(type(capability), object)
        self.assertIs(
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES[
                "snapshot"
            ],
            capability,
        )

        legacy = _host_only_solver(authority="legacy", generation=4)
        legacy.prepare_velocity_dirichlet_component_ledger_snapshot()
        self.assertEqual(
            legacy._velocity_dirichlet_component_ledger_consumer_generations,
            {},
        )
        self.assertEqual(
            legacy._velocity_dirichlet_component_ledger_consumer_capabilities,
            {},
        )

        canonical = _host_only_solver(authority="canonical", generation=17)
        canonical.prepare_velocity_dirichlet_component_ledger_snapshot()
        self.assertEqual(
            canonical._velocity_dirichlet_component_ledger_consumer_generations,
            {"snapshot": 17},
        )
        self.assertIs(
            canonical._velocity_dirichlet_component_ledger_consumer_capabilities[
                "snapshot"
            ],
            capability,
        )

        next_generation = canonical._invalidate_velocity_dirichlet_component_ledger()
        canonical.prepare_velocity_dirichlet_component_ledger_snapshot()
        self.assertEqual(
            canonical._velocity_dirichlet_component_ledger_consumer_generations,
            {"snapshot": next_generation},
        )

    def test_current_schema_capture_is_sealed_first_and_persists_exact_ledger(
        self,
    ) -> None:
        self.assertEqual(preflow_snapshot_module.PREFLOW_SNAPSHOT_SCHEMA_VERSION, 8)
        ledger_field_occurrences = tuple(
            name
            for name in preflow_snapshot_module.PREFLOW_SNAPSHOT_FIELD_NAMES
            if name in _CANONICAL_LEDGER_FIELDS
        )
        self.assertEqual(len(ledger_field_occurrences), 8)
        self.assertEqual(set(ledger_field_occurrences), _CANONICAL_LEDGER_FIELDS)
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

        repository_root = Path(inspect.getfile(CartesianFluidSolver)).resolve().parents[2]
        runner_path = (
            repository_root / "benchmarks" / "official" / "solid_mpm_fsi_runner.py"
        )
        capture_source = _function_source(
            runner_path,
            "_capture_preflow_snapshot_fields",
        )
        sealed_guard = "_require_velocity_dirichlet_component_ledger_sealed"
        self.assertIn(sealed_guard, capture_source)
        self.assertLess(
            capture_source.index(sealed_guard),
            capture_source.index("PREFLOW_SNAPSHOT_FIELD_NAMES"),
        )

        write_source = _function_source(
            runner_path,
            "_write_fixed_solid_preflow_snapshot",
        )
        self.assertIn("velocity_dirichlet_boundary_authority", write_source)
        self.assertIn("velocity_dirichlet_component_ledger_generation", write_source)


if __name__ == "__main__":
    unittest.main()
