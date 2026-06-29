from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path


SCRIPT_ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi" / "scripts"
PACKAGE_PREFIX = "tools.validation.ansys_vertical_flap"
PACKAGE_ROOT = Path("tools") / "validation" / "ansys_vertical_flap"

PACKAGE_MODULES = (
    PACKAGE_ROOT / "fluent_reference_contract_schema.py",
    PACKAGE_ROOT / "fluent_source_export_schema.py",
    PACKAGE_ROOT / "fluent_reference_collection.py",
    PACKAGE_ROOT / "fluent_parity.py",
    PACKAGE_ROOT / "fluent_artifact_policy.py",
    PACKAGE_ROOT / "validation_artifact_hygiene.py",
    PACKAGE_ROOT / "policy_report_writer.py",
)

CLI_WRAPPERS = (
    SCRIPT_ROOT / "check_fluent_artifact_policy.py",
    SCRIPT_ROOT / "check_validation_artifact_hygiene.py",
    SCRIPT_ROOT / "run_fluent_reference_collection_validation.py",
    SCRIPT_ROOT / "run_traction_selected_formulation_fluent_parity.py",
    Path("scripts") / "check_validation_artifact_hygiene.py",
)

ARGPARSE_WRAPPERS = (
    SCRIPT_ROOT / "check_fluent_artifact_policy.py",
    SCRIPT_ROOT / "check_validation_artifact_hygiene.py",
    Path("scripts") / "check_validation_artifact_hygiene.py",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnsysVerticalFlapValidationPackageTests(unittest.TestCase):
    def test_target_package_modules_are_importable(self):
        for module_name in (
            "fluent_reference_contract_schema",
            "fluent_source_export_schema",
            "fluent_reference_collection",
            "fluent_parity",
            "fluent_artifact_policy",
            "validation_artifact_hygiene",
            "policy_report_writer",
        ):
            module = importlib.import_module(f"{PACKAGE_PREFIX}.{module_name}")
            self.assertEqual(module.__name__, f"{PACKAGE_PREFIX}.{module_name}")

    def test_validation_script_wrappers_delegate_to_package_functions(self):
        checks = (
            (
                SCRIPT_ROOT / "fluent_reference_contract_schema.py",
                "validate_fluent_reference_contract",
            ),
            (
                SCRIPT_ROOT / "fluent_source_export_schema.py",
                "validate_source_export_csv",
            ),
            (
                SCRIPT_ROOT / "check_fluent_artifact_policy.py",
                "check_fluent_artifact_policy",
            ),
            (
                SCRIPT_ROOT / "run_fluent_reference_collection_validation.py",
                "run_with_paths",
            ),
            (
                SCRIPT_ROOT / "run_traction_selected_formulation_fluent_parity.py",
                "_parity_metrics",
            ),
            (
                Path("scripts") / "check_validation_artifact_hygiene.py",
                "check_validation_artifact_hygiene",
            ),
            (
                SCRIPT_ROOT / "check_validation_artifact_hygiene.py",
                "check_validation_artifact_hygiene",
            ),
        )

        for path, function_name in checks:
            with self.subTest(path=path.as_posix(), function=function_name):
                module = _load_module(path.stem, path)
                function = getattr(module, function_name)
                self.assertTrue(
                    function.__module__.startswith(PACKAGE_PREFIX),
                    function.__module__,
                )

    def test_script_wrappers_do_not_define_validation_business_functions(self):
        forbidden_defs = {
            SCRIPT_ROOT / "fluent_reference_contract_schema.py": (
                "def validate_fluent_reference_contract(",
            ),
            SCRIPT_ROOT / "fluent_source_export_schema.py": (
                "def validate_source_export_csv(",
            ),
            SCRIPT_ROOT / "check_fluent_artifact_policy.py": (
                "def check_fluent_artifact_policy(",
                "def _check_payload(",
            ),
            SCRIPT_ROOT / "run_fluent_reference_collection_validation.py": (
                "def _candidate_contract(",
                "def _source_check(",
            ),
            SCRIPT_ROOT / "run_traction_selected_formulation_fluent_parity.py": (
                "def _parity_metrics(",
                "def _candidate_status(",
            ),
            Path("scripts") / "check_validation_artifact_hygiene.py": (
                "def check_validation_artifact_hygiene(",
                "def _check_text(",
            ),
            SCRIPT_ROOT / "check_validation_artifact_hygiene.py": (
                "def check_validation_artifact_hygiene(",
                "def _check_text(",
            ),
        }

        for path, definitions in forbidden_defs.items():
            with self.subTest(path=path.as_posix()):
                text = path.read_text(encoding="utf-8")
                for definition in definitions:
                    self.assertNotIn(definition, text)

    def test_package_modules_do_not_own_cli_entrypoints(self):
        for path in PACKAGE_MODULES:
            with self.subTest(path=path.as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("def main(", text)
                self.assertNotIn('if __name__ == "__main__"', text)

    def test_package_modules_do_not_own_argparse_boundaries(self):
        for path in (
            PACKAGE_ROOT / "fluent_artifact_policy.py",
            PACKAGE_ROOT / "validation_artifact_hygiene.py",
        ):
            with self.subTest(path=path.as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("argparse.ArgumentParser", text)

    def test_cli_wrappers_own_cli_entrypoints(self):
        for path in CLI_WRAPPERS:
            with self.subTest(path=path.as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertIn("def main(", text)

    def test_argparse_wrappers_own_argparse_boundaries(self):
        for path in ARGPARSE_WRAPPERS:
            with self.subTest(path=path.as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertIn("argparse.ArgumentParser", text)

    def test_policy_and_hygiene_wrapper_mains_preserve_success_exit(self):
        for path in ARGPARSE_WRAPPERS:
            with self.subTest(path=path.as_posix()):
                module = _load_module(path.stem, path)
                previous_argv = sys.argv[:]
                try:
                    sys.argv = [path.as_posix()]
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(module.main(), 0)
                finally:
                    sys.argv = previous_argv


if __name__ == "__main__":
    unittest.main()
