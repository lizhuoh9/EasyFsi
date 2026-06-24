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
            "geometry_tools",
            "materials",
            "diagnostics",
        ):
            self.assertTrue(
                (REPO_ROOT / "simulation_core" / name / "__init__.py").exists(),
                msg=f"missing simulation_core facade package: {name}",
            )

    def test_simulation_core_legacy_modules_still_exist_during_facade_migration(self) -> None:
        for name in (
            "fluid.py",
            "hibm_mpm.py",
            "cad_import.py",
            "cad_tessellation.py",
            "coordinate_models.py",
            "fluid_domain.py",
            "geometry.py",
            "hyperelastic.py",
            "mooney_shell_mpm.py",
            "neo_hookean_mpm.py",
            "time_stepping.py",
            "validation.py",
        ):
            self.assertTrue(
                (REPO_ROOT / "simulation_core" / name).exists(),
                msg=f"missing legacy simulation_core module: {name}",
            )

    def test_fluid_legacy_module_is_shim_after_step7(self) -> None:
        source = (REPO_ROOT / "simulation_core" / "fluid.py").read_text(encoding="utf-8")

        self.assertIn("from simulation_core.fluids import", source)
        self.assertNotIn("@ti.kernel", source)
        self.assertNotIn("class CartesianFluidSolver", source)

    def test_fluid_solver_implementation_lives_under_fluids_package(self) -> None:
        source = (
            REPO_ROOT / "simulation_core" / "fluids" / "solver.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class CartesianFluidSolver", source)

    def test_hibm_mpm_legacy_module_is_shim_after_step8(self) -> None:
        source = (REPO_ROOT / "simulation_core" / "hibm_mpm.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("from simulation_core.coupling.hibm_mpm import", source)
        self.assertNotIn("@ti.kernel", source)
        self.assertNotIn("class HibmMpmSurfaceMarkers", source)
        self.assertNotIn("class HibmMpmSharpCouplingState", source)

    def test_hibm_mpm_implementation_lives_under_coupling_package(self) -> None:
        source = (
            REPO_ROOT / "simulation_core" / "coupling" / "hibm_mpm" / "core.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class HibmMpmSurfaceMarkers", source)
        self.assertIn("class HibmMpmSharpCouplingState", source)

    def test_solver_support_legacy_modules_are_shims_after_step9(self) -> None:
        expected_imports = {
            "cad_import.py": "from simulation_core.geometry_tools.cad_import import",
            "cad_tessellation.py": "from simulation_core.geometry_tools.cad_tessellation import",
            "coordinate_models.py": "from simulation_core.geometry_tools.coordinate_models import",
            "fluid_domain.py": "from simulation_core.geometry_tools.fluid_domain import",
            "geometry.py": "from simulation_core.geometry_tools.surface_mesh import",
            "hyperelastic.py": "from simulation_core.materials.hyperelastic import",
            "mooney_shell_mpm.py": "from simulation_core.solids.mooney_shell import",
            "neo_hookean_mpm.py": "from simulation_core.solids.neo_hookean_mpm import",
            "time_stepping.py": "from simulation_core.diagnostics.time_stepping import",
            "validation.py": "from simulation_core.diagnostics.validation import",
        }
        forbidden_tokens = (
            "@ti.kernel",
            "class NeoHookeanMpmState",
            "class TriMooneyShellMpmState",
            "class UvMooneyShellMpmState",
            "class SurfaceMesh",
            "class NeoHookeanMaterial",
            "class ReferenceCurve",
            "class CflSubstepController",
        )

        for name, import_line in expected_imports.items():
            source = (REPO_ROOT / "simulation_core" / name).read_text(encoding="utf-8")

            self.assertIn(import_line, source)
            for token in forbidden_tokens:
                self.assertNotIn(token, source, msg=f"{name}: {token}")

    def test_solver_support_implementations_live_under_layered_packages(self) -> None:
        expected_tokens = {
            ("solids", "neo_hookean_mpm.py"): "class NeoHookeanMpmState",
            ("solids", "mooney_shell", "core.py"): "class TriMooneyShellMpmState",
            ("solids", "mooney_shell", "reports.py"): "class TriMooneyShellMpmReport",
            ("geometry_tools", "surface_mesh.py"): "class SurfaceMesh",
            ("geometry_tools", "coordinate_models.py"): "class Cartesian3DCoordinateModel",
            ("geometry_tools", "fluid_domain.py"): "class FluidDomain",
            ("geometry_tools", "cad_import.py"): "class StepCadSummary",
            ("geometry_tools", "cad_tessellation.py"): "class StepTessellationSettings",
            ("materials", "hyperelastic.py"): "class NeoHookeanMaterial",
            ("diagnostics", "validation.py"): "class ReferenceCurve",
            ("diagnostics", "time_stepping.py"): "class CflSubstepController",
        }

        for parts, token in expected_tokens.items():
            source = (REPO_ROOT / "simulation_core" / Path(*parts)).read_text(
                encoding="utf-8"
            )

            self.assertIn(token, source, msg=f"{parts}: {token}")

    def test_hibm_mpm_support_modules_do_not_import_core(self) -> None:
        for name in (
            "constants.py",
            "modes.py",
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

    def test_squid_step_loop_module_owns_main_step_loop_after_step5_split(self) -> None:
        source = (
            REPO_ROOT / "cases" / "squid_soft_robot" / "step_loop.py"
        ).read_text(encoding="utf-8")

        self.assertIn("for step in range(", source)

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
