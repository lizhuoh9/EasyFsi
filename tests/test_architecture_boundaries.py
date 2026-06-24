from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
