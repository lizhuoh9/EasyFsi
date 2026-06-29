from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path("scripts") / "check_validation_artifact_hygiene.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_validation_artifact_hygiene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


class ValidationArtifactHygieneTests(unittest.TestCase):
    def test_current_validation_artifacts_pass_hygiene(self):
        result = CHECKER.check_validation_artifact_hygiene()

        self.assertEqual(result["status"], "passed", result["violations"])
        self.assertGreater(result["checked_file_count"], 0)

    def test_local_absolute_path_in_artifact_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.json"
            artifact.write_text('{"path": "D:\\\\working\\\\bad"}\n', encoding="utf-8")
            _write_checksums(root, [artifact])

            result = CHECKER.check_validation_artifact_hygiene(
                [root],
                active_contract_manifest=Path(tmp) / "missing_manifest.json",
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "local_absolute_path",
            {item["rule"] for item in result["violations"]},
        )


def _write_checksums(root: Path, files: list[Path]) -> None:
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "CHECKSUMS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
