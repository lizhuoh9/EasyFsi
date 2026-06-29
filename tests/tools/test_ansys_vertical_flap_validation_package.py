from __future__ import annotations

import importlib
import importlib.util
import unittest
from pathlib import Path


SCRIPT_ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi" / "scripts"
PACKAGE_PREFIX = "tools.validation.ansys_vertical_flap"


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


if __name__ == "__main__":
    unittest.main()
