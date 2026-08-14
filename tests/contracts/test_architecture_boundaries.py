from __future__ import annotations

import ast
import unittest
from pathlib import Path

from tests._paths import REPO_ROOT


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".", 1)[0])

    return roots


PACKAGE_IMPLEMENTATIONS = {
    ("fluids", "solver.py"): "class CartesianFluidSolver",
    ("coupling", "interface_forces.py"): "class ForceBalanceReport",
    ("coupling", "hibm.py"): "def classify_hibm_near_boundary_nodes",
    ("coupling", "hibm_mpm", "core.py"): "class HibmMpmSharpCouplingState",
    ("coupling", "interface_pair.py"): "class InterfacePairMap",
    ("coupling", "moving_boundary.py"): "class MovingBoundaryCondition",
    ("coupling", "pressure_interface.py"): "def far_pressure_side_normal_sign_from_direction",
    ("coupling", "pressure_sample_pairs.py"): "class PressureSamplePair",
    ("coupling", "projected_ibm.py"): "class ProjectedIbmRegionPairStepConfig",
    ("coupling", "tri_surface.py"): "class TriSurfaceRegionDiagnostics",
    ("diagnostics", "runtime.py"): "class TaichiRuntimeConfig",
    ("drivers", "case_spec.py"): "class FsiCaseSpec",
    ("drivers", "generic_fsi_solver.py"): "class FsiProblem",
    ("solids", "neo_hookean_mpm.py"): "class NeoHookeanMpmState",
    ("solids", "mooney_shell", "core.py"): "class TriMooneyShellMpmState",
    ("geometry_tools", "surface_mesh.py"): "class SurfaceMesh",
    ("geometry_tools", "cad_tessellation.py"): "class StepTessellationSettings",
    ("materials", "hyperelastic.py"): "class NeoHookeanMaterial",
    ("diagnostics", "validation.py"): "class ReferenceCurve",
    ("diagnostics", "time_stepping.py"): "class CflSubstepController",
}

PACKAGE_IMPLEMENTATION_ROOTS = (
    REPO_ROOT / "simulation_core" / "fluids",
    REPO_ROOT / "simulation_core" / "coupling",
    REPO_ROOT / "simulation_core" / "solids",
    REPO_ROOT / "simulation_core" / "geometry_tools",
    REPO_ROOT / "simulation_core" / "materials",
    REPO_ROOT / "simulation_core" / "diagnostics",
    REPO_ROOT / "simulation_core" / "drivers",
)

LEGACY_IMPORT_TOKENS = (
    "from simulation_core.fluid import",
    "from simulation_core.fsi_coupling import",
    "from simulation_core.fsi_driver import",
    "from simulation_core.generic_fsi_solver import",
    "from simulation_core.hibm import",
    "from simulation_core.hibm_mpm import",
    "from simulation_core.interface_pair import",
    "from simulation_core.moving_boundary import",
    "from simulation_core.pressure_interface import",
    "from simulation_core.pressure_sample_pairs import",
    "from simulation_core.projected_ibm import",
    "from simulation_core.runtime import",
    "from simulation_core.tri_surface import",
    "from simulation_core.neo_hookean_mpm import",
    "from simulation_core.mooney_shell_mpm import",
    "from simulation_core.geometry import",
    "from simulation_core.coordinate_models import",
    "from simulation_core.fluid_domain import",
    "from simulation_core.cad_import import",
    "from simulation_core.cad_tessellation import",
    "from simulation_core.hyperelastic import",
    "from simulation_core.validation import",
    "from simulation_core.time_stepping import",
)


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_simulation_core_does_not_import_cases_tools_or_benchmarks(self) -> None:
        forbidden = {"cases", "tools", "benchmarks"}

        for path in _python_files(REPO_ROOT / "simulation_core"):
            roots = _import_roots(path)
            leaked = roots & forbidden
            self.assertFalse(
                leaked,
                msg=f"{path} imports forbidden top-level packages: {sorted(leaked)}",
            )

    def test_cases_do_not_import_tools(self) -> None:
        forbidden = {"tools"}

        for path in _python_files(REPO_ROOT / "cases"):
            roots = _import_roots(path)
            leaked = roots & forbidden
            self.assertFalse(
                leaked,
                msg=f"{path} imports forbidden top-level packages: {sorted(leaked)}",
            )

    def test_benchmarks_do_not_import_cases_or_tools(self) -> None:
        forbidden = {"cases", "tools"}

        for path in _python_files(REPO_ROOT / "benchmarks"):
            roots = _import_roots(path)
            leaked = roots & forbidden
            self.assertFalse(
                leaked,
                msg=f"{path} imports forbidden top-level packages: {sorted(leaked)}",
            )

    def test_simulation_core_has_no_benchmarking_package(self) -> None:
        self.assertFalse((REPO_ROOT / "simulation_core" / "benchmarking").exists())

    def test_simulation_core_layered_facade_packages_exist(self) -> None:
        for name in (
            "fluids",
            "solids",
            "coupling",
            "drivers",
            "geometry_tools",
            "materials",
            "diagnostics",
        ):
            self.assertTrue(
                (REPO_ROOT / "simulation_core" / name / "__init__.py").exists(),
                msg=f"missing simulation_core facade package: {name}",
            )

    def test_root_simulation_core_contains_only_package_facade(self) -> None:
        expected = {"__init__.py"}
        actual = {
            path.name
            for path in (REPO_ROOT / "simulation_core").glob("*.py")
        }

        self.assertEqual(actual, expected)

    def test_package_implementations_live_under_layered_packages(self) -> None:
        for parts, token in PACKAGE_IMPLEMENTATIONS.items():
            source = (REPO_ROOT / "simulation_core" / Path(*parts)).read_text(
                encoding="utf-8"
            )

            self.assertIn(token, source, msg="/".join(parts))

    def test_package_implementation_modules_do_not_import_legacy_shims(self) -> None:
        for root in PACKAGE_IMPLEMENTATION_ROOTS:
            for path in _python_files(root):
                source = path.read_text(encoding="utf-8")
                for token in LEGACY_IMPORT_TOKENS:
                    self.assertNotIn(token, source, msg=f"{path}: {token}")

    def test_root_public_api_imports_use_layered_facades(self) -> None:
        source = (REPO_ROOT / "simulation_core" / "__init__.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("from .fluids import", source)
        self.assertIn("from .coupling import", source)
        self.assertIn("from .solids import", source)
        self.assertIn("from .geometry_tools import", source)
        self.assertIn("from .materials import", source)
        self.assertIn("from .diagnostics import", source)
        self.assertNotIn("from .fluid import", source)
        self.assertNotIn("from .hibm_mpm import", source)
        self.assertNotIn("_COMPAT_MODULE_ALIASES", source)
        self.assertNotIn("_install_compat_module_aliases", source)
        self.assertNotIn("sys.modules[f", source)

    def test_hibm_mpm_support_modules_do_not_import_core(self) -> None:
        for name in (
            "constants.py",
            "paper_requirements.py",
            "reports.py",
        ):
            source = (
                REPO_ROOT / "simulation_core" / "coupling" / "hibm_mpm" / name
            ).read_text(encoding="utf-8")

            self.assertNotIn("import simulation_core.coupling.hibm_mpm.core", source)
            self.assertNotIn(
                "from simulation_core.coupling.hibm_mpm.core",
                source,
            )
            self.assertNotIn("from .core", source)

    def test_fluid_support_modules_do_not_import_solver(self) -> None:
        for name in (
            "constants.py",
            "grid.py",
            "spec.py",
            "reports.py",
            "pressure_outlet.py",
        ):
            source = (
                REPO_ROOT / "simulation_core" / "fluids" / name
            ).read_text(encoding="utf-8")

            self.assertNotIn("import simulation_core.fluids.solver", source)
            self.assertNotIn("from simulation_core.fluids.solver", source)
            self.assertNotIn("from .solver", source)

    def test_legacy_top_level_tool_scripts_are_not_present(self) -> None:
        for name in (
            "inspect_latest_progress.py",
            "inspect_visit_stats.py",
            "summarize_preflight_log.py",
            "_finalize_copy.py",
        ):
            self.assertFalse((REPO_ROOT / name).exists())

    def test_cases_contains_no_rendering_helpers(self) -> None:
        for name in (
            "squid_jet_render.py",
            "squid_current_visit_render.py",
        ):
            self.assertFalse((REPO_ROOT / "cases" / name).exists())

    def test_squid_case_package_has_no_sys_modules_alias(self) -> None:
        source = (
            REPO_ROOT / "cases" / "squid_soft_robot" / "__init__.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("sys.modules", source)

    def test_squid_runner_does_not_define_argparse_bulk_after_cli_split(self) -> None:
        source = (
            REPO_ROOT / "cases" / "squid_soft_robot" / "runner.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("argparse.ArgumentParser(", source)

    def test_squid_runner_does_not_define_runtime_state_class_after_split(self) -> None:
        source = (
            REPO_ROOT / "cases" / "squid_soft_robot" / "runner.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("class ReducedSquidFSI", source)

    def test_squid_runner_does_not_hold_final_summary_bulk_after_split(self) -> None:
        source = (
            REPO_ROOT / "cases" / "squid_soft_robot" / "runner.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "final_pressure_outlet_velocity_to_source_ratio =",
            source,
        )
        self.assertNotIn(
            "max_velocity_constraint_equivalent_force_norm_n =",
            source,
        )

    def test_squid_runner_does_not_hold_main_step_loop_after_step5_split(self) -> None:
        source = (
            REPO_ROOT / "cases" / "squid_soft_robot" / "runner.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("for step in range(first_step, step_count + 1):", source)

    def test_squid_step_loop_delegates_main_step_loop_to_generic_solver(self) -> None:
        source = (
            REPO_ROOT / "cases" / "squid_soft_robot" / "step_loop.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("for step in range(", source)
        self.assertIn("solve_fsi_runtime(runtime, solver_config)", source)

    def test_squid_runner_does_not_define_sharp_trial_closure_after_step5_split(self) -> None:
        source = (
            REPO_ROOT / "cases" / "squid_soft_robot" / "runner.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("def advance_sharp_trial_once", source)

    def test_squid_case_modules_do_not_import_runner_except_init(self) -> None:
        root = REPO_ROOT / "cases" / "squid_soft_robot"

        for path in root.glob("*.py"):
            if path.name == "__init__.py":
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("from . import runner", source)
            self.assertNotIn("from cases.squid_soft_robot import runner", source)


if __name__ == "__main__":
    unittest.main()
