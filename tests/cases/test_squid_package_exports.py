from __future__ import annotations

import importlib
import unittest


class SquidPackageExportTests(unittest.TestCase):
    def test_case_package_exposes_main_for_generic_entrypoint(self) -> None:
        module = importlib.import_module("cases.squid_soft_robot")
        self.assertTrue(callable(module.main))

    def test_case_package_allows_runner_submodule_import(self) -> None:
        runner = importlib.import_module("cases.squid_soft_robot.runner")
        self.assertTrue(callable(runner.main))

    def test_new_code_can_import_explicit_submodules(self) -> None:
        for module_name in (
            "cases.squid_soft_robot.cli",
            "cases.squid_soft_robot.spec",
            "cases.squid_soft_robot.source_config",
            "cases.squid_soft_robot.schedules",
            "cases.squid_soft_robot.history",
            "cases.squid_soft_robot.checkpointing",
            "cases.squid_soft_robot.diagnostics",
            "cases.squid_soft_robot.snapshots",
            "cases.squid_soft_robot.runtime_state",
            "cases.squid_soft_robot.summary",
            "cases.squid_soft_robot.rows",
            "cases.squid_soft_robot.setup",
            "cases.squid_soft_robot.step_context",
            "cases.squid_soft_robot.step_loop",
            "cases.squid_soft_robot.trial_replay",
            "cases.squid_soft_robot.solid_step",
            "cases.squid_soft_robot.fluid_step",
            "cases.squid_soft_robot.coupling_common",
            "cases.squid_soft_robot.coupling_legacy",
            "cases.squid_soft_robot.coupling_sharp",
        ):
            module = importlib.import_module(module_name)
            self.assertIsNotNone(module)

    def test_runtime_state_is_reexported_for_legacy_imports(self) -> None:
        module = importlib.import_module("cases.squid_soft_robot")
        self.assertTrue(hasattr(module, "ReducedSquidFSI"))

    def test_legacy_private_helpers_remain_importable_during_split(self) -> None:
        module = importlib.import_module("cases.squid_soft_robot")
        self.assertTrue(callable(module._cell_indices_for_points))


if __name__ == "__main__":
    unittest.main()
