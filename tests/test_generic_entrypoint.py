from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import run_simulation


class GenericEntrypointTests(unittest.TestCase):
    def test_dispatches_squid_case_without_running_solver_in_entrypoint(self) -> None:
        received_args: list[str] | None = None

        def fake_case_main(argv: list[str] | None = None) -> dict[str, object]:
            nonlocal received_args
            received_args = argv
            return {"case": "squid-soft-robot"}

        with patch("run_simulation._load_case_main", return_value=fake_case_main):
            result = run_simulation.dispatch(["squid-soft-robot", "--steps", "1"])

        self.assertEqual(result, {"case": "squid-soft-robot"})
        self.assertEqual(received_args, ["--steps", "1"])

    def test_squid_case_module_exposes_dispatch_main(self) -> None:
        case_main = run_simulation._load_case_main("squid-soft-robot")

        self.assertTrue(callable(case_main))

    def test_generic_entrypoint_does_not_hardcode_squid_case(self) -> None:
        source = Path("run_simulation.py").read_text(encoding="utf-8")

        self.assertNotIn("squid-soft-robot", source)
        self.assertNotIn("cases.squid_soft_robot", source)
        self.assertIn("CASE_MODULES", source)

    def test_unknown_case_fails_before_any_solver_import(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            run_simulation.dispatch(["unknown-case"])

        self.assertIn("Unknown case", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
