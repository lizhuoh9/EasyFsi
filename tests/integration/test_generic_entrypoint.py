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

    def test_registered_module_without_main_raises_clear_error(self) -> None:
        with patch.dict(
            run_simulation.CASE_MODULES, {"mainless-case": "math"}, clear=False
        ):
            with self.assertRaises(SystemExit) as raised:
                run_simulation.dispatch(["mainless-case"])

        message = str(raised.exception)
        self.assertIn("mainless-case", message)
        self.assertIn("has no main() entry point", message)
        self.assertIn("Available cases:", message)

    def test_comsol_spec_only_cases_report_missing_main_not_attribute_error(
        self,
    ) -> None:
        for case_name in (
            "comsol-water-balloon-fsi",
            "comsol-multibody-mechanism-fsi",
        ):
            with self.subTest(case=case_name):
                with self.assertRaises(SystemExit) as raised:
                    run_simulation.dispatch([case_name])

                message = str(raised.exception)
                self.assertIn(case_name, message)
                self.assertIn("has no main() entry point", message)

    def test_public_registry_only_lists_cases_with_cli_main(self) -> None:
        from cases import AVAILABLE_CASES, CASE_MODULES

        self.assertNotIn("comsol-water-balloon-fsi", AVAILABLE_CASES)
        self.assertNotIn("comsol-multibody-mechanism-fsi", AVAILABLE_CASES)
        # The module registry keeps the spec-only benchmark cases registered.
        self.assertIn("comsol-water-balloon-fsi", CASE_MODULES)
        self.assertIn("comsol-multibody-mechanism-fsi", CASE_MODULES)
        self.assertEqual(
            tuple(sorted(AVAILABLE_CASES)),
            ("ansys-vertical-flap-fsi", "squid-soft-robot", "turek-hron-fsi"),
        )
        usage = run_simulation._usage()
        self.assertNotIn("comsol", usage)
        for case_name in AVAILABLE_CASES:
            self.assertIn(case_name, usage)


if __name__ == "__main__":
    unittest.main()
