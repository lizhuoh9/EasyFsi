from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from benchmarks.official import solid_mpm_fsi_runner
from simulation_core.fluids.preflow_snapshot import canonical_source_sha256


class PreflowSnapshotSourceScopeTests(unittest.TestCase):
    @staticmethod
    def _source_identity_for(runner_path: Path) -> tuple[str, frozenset[str]]:
        with mock.patch.object(
            solid_mpm_fsi_runner,
            "__file__",
            str(runner_path),
        ):
            sources = solid_mpm_fsi_runner._preflow_snapshot_source_payload()
        return canonical_source_sha256(sources), frozenset(sources)

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_unrelated_case_edit_does_not_invalidate_active_solver_source_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            runner_path = (
                repo_root
                / "benchmarks"
                / "official"
                / "solid_mpm_fsi_runner.py"
            )
            core_path = repo_root / "simulation_core" / "fluids" / "solver.py"
            unrelated_case_path = repo_root / "cases" / "unrelated_case.py"
            self._write(runner_path, "def run():\n    return 'runner-v1'\n")
            self._write(core_path, "class FluidSolver:\n    pass\n")
            self._write(unrelated_case_path, "CASE_LABEL = 'unrelated-v1'\n")

            baseline_identity, baseline_sources = self._source_identity_for(
                runner_path
            )
            self._write(unrelated_case_path, "CASE_LABEL = 'unrelated-v2'\n")
            edited_identity, edited_sources = self._source_identity_for(runner_path)

            self.assertEqual(edited_identity, baseline_identity)
            self.assertEqual(edited_sources, baseline_sources)
            self.assertNotIn("cases/unrelated_case.py", baseline_sources)

    def test_runner_and_core_edits_invalidate_active_solver_source_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            runner_path = (
                repo_root
                / "benchmarks"
                / "official"
                / "solid_mpm_fsi_runner.py"
            )
            core_path = repo_root / "simulation_core" / "fluids" / "solver.py"
            self._write(runner_path, "def run():\n    return 'runner-v1'\n")
            self._write(core_path, "class FluidSolver:\n    pass\n")

            baseline_identity, baseline_sources = self._source_identity_for(
                runner_path
            )
            self._write(runner_path, "def run():\n    return 'runner-v2'\n")
            runner_identity, _ = self._source_identity_for(runner_path)

            self._write(runner_path, "def run():\n    return 'runner-v1'\n")
            self._write(core_path, "class FluidSolver:\n    version = 2\n")
            core_identity, _ = self._source_identity_for(runner_path)

            self.assertIn(
                "benchmarks/official/solid_mpm_fsi_runner.py",
                baseline_sources,
            )
            self.assertIn("simulation_core/fluids/solver.py", baseline_sources)
            self.assertNotEqual(runner_identity, baseline_identity)
            self.assertNotEqual(core_identity, baseline_identity)


if __name__ == "__main__":
    unittest.main()
