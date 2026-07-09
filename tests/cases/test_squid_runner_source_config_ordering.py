from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cases.squid_soft_robot import cli, runner


class SquidRunnerSourceConfigOrderingTests(unittest.TestCase):
    """Missing --source-config must fail before ANY filesystem side effects.

    Regression guard for the 2026-07 ordering bug: run() used to mkdir the
    output directory and write run_process.json (status=running) BEFORE
    checking that the source config exists, and the failure guard then
    rewrote run_process.json with status=failed -- so a simple typo in
    --source-config littered the tree with a fake run directory.
    """

    def test_missing_source_config_fails_with_no_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            missing_config = root / "not_checked_in" / "simulation_config.json"
            output_dir = root / "run_output"
            args = cli.parse_args(
                [
                    "--source-config",
                    str(missing_config),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            previous_cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaises(
                    runner.SourceConfigNotFoundError
                ) as raised:
                    runner.run(args)
            finally:
                os.chdir(previous_cwd)

            message = str(raised.exception)
            self.assertIn("source config not found", message)
            self.assertIn("--source-config", message)
            self.assertIn(str(missing_config), message)
            # Actionable AND side-effect free: no output dir, no
            # run_process.json, nothing at all created under the tmp cwd.
            self.assertFalse(output_dir.exists())
            self.assertFalse((output_dir / "run_process.json").exists())
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                [],
                "missing source config must not create any files or dirs",
            )

    def test_source_config_error_is_a_file_not_found_error(self) -> None:
        # Callers that already handle FileNotFoundError keep working.
        self.assertTrue(
            issubclass(runner.SourceConfigNotFoundError, FileNotFoundError)
        )

    def test_main_propagates_missing_source_config_without_side_effects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output_dir = root / "run_output"

            previous_cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaises(runner.SourceConfigNotFoundError):
                    runner.main(
                        [
                            "--source-config",
                            str(root / "missing.json"),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertFalse(output_dir.exists())
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
