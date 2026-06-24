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

    def test_legacy_private_helpers_remain_importable_during_split(self) -> None:
        module = importlib.import_module("cases.squid_soft_robot")
        self.assertTrue(callable(module._cell_indices_for_points))


if __name__ == "__main__":
    unittest.main()
