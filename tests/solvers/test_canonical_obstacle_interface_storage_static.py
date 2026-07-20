from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOLVER_PATH = (
    Path(__file__).resolve().parents[2]
    / "simulation_core"
    / "fluids"
    / "solver.py"
)
SOLVER_CLASS_NAME = "CartesianFluidSolver"
CLEANUP_KERNELS = (
    "_apply_obstacle_no_normal_flow_kernel",
    "_zero_obstacle_cell_velocity_kernel",
)


def _solver_class() -> ast.ClassDef:
    module = ast.parse(
        SOLVER_PATH.read_text(encoding="utf-8"),
        filename=str(SOLVER_PATH),
    )
    for statement in module.body:
        if isinstance(statement, ast.ClassDef) and statement.name == SOLVER_CLASS_NAME:
            return statement
    raise AssertionError(f"missing class {SOLVER_CLASS_NAME!r}")


def _method(name: str) -> ast.FunctionDef:
    for statement in _solver_class().body:
        if isinstance(statement, ast.FunctionDef) and statement.name == name:
            return statement
    raise AssertionError(f"missing method {SOLVER_CLASS_NAME}.{name}")


def _self_call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        function = candidate.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "self"
        ):
            names.add(function.attr)
    return names


class CanonicalObstacleInterfaceStorageStaticContracts(unittest.TestCase):
    def test_obstacle_cleanup_kernels_share_a_canonical_interface_guard(
        self,
    ) -> None:
        common_calls = set.intersection(
            *(_self_call_names(_method(name)) for name in CLEANUP_KERNELS)
        )
        guard_names = {
            name
            for name in common_calls
            if "canonical" in name
            and "obstacle" in name
            and "interface" in name
        }
        self.assertEqual(
            len(guard_names),
            1,
            "both obstacle cleanup policies must use one shared canonical "
            "obstacle-interface component guard",
        )

        guard_source = ast.unparse(_method(next(iter(guard_names))))
        for required_field in (
            "velocity_dirichlet_boundary_active_component_mask",
            "velocity_dirichlet_boundary_hard_fixed_component_mask",
            "velocity_dirichlet_boundary_external_exact_component_mask",
            "velocity_dirichlet_boundary_owned_component_mask",
            "velocity_dirichlet_boundary_pressure_mobility",
            "velocity_dirichlet_boundary_component_enforcement_weight",
            "obstacle",
        ):
            with self.subTest(required_guard_input=required_field):
                self.assertIn(required_field, guard_source)


if __name__ == "__main__":
    unittest.main()
