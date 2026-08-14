from __future__ import annotations

import hashlib
import importlib.util
import subprocess
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

    def test_default_discovers_every_tracked_checksum_manifest(self):
        expected = {
            Path(line).as_posix()
            for line in subprocess.check_output(
                ["git", "ls-files", "--", "**/CHECKSUMS.sha256"],
                text=True,
            ).splitlines()
        }

        result = CHECKER.check_validation_artifact_hygiene()

        self.assertEqual(set(result["checked_manifests"]), expected)

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

    def test_unlisted_artifact_fails_checksum_manifest_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            listed = root / "listed.json"
            listed.write_text('{"status": "tracked"}\n', encoding="utf-8")
            (root / "omitted.json").write_text(
                '{"status": "must also be tracked"}\n',
                encoding="utf-8",
            )
            _write_checksums(root, [listed])

            result = CHECKER.check_validation_artifact_hygiene(
                [root],
                active_contract_manifest=root / "missing_manifest.json",
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "checksum_entry_missing",
            {item["rule"] for item in result["violations"]},
        )

    def test_binary_artifact_is_checksum_checked_without_utf8_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "fields.npz"
            artifact.write_bytes(b"\x89NPZ\x00\xff")
            _write_checksums(root, [artifact])

            result = CHECKER.check_validation_artifact_hygiene(
                [root],
                active_contract_manifest=root / "missing_manifest.json",
            )

        self.assertEqual(result["status"], "passed", result["violations"])

    def test_tracked_manifest_rejects_ignored_untracked_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository_root = Path(tmp)
            artifact_root = repository_root / "artifact"
            artifact_root.mkdir()
            target = artifact_root / "worker.log"
            target.write_text("ephemeral\n", encoding="utf-8")
            _write_checksums(artifact_root, [target])
            (repository_root / ".gitignore").write_text("*.log\n", encoding="utf-8")
            _git(repository_root, "init", "--quiet")
            _git(repository_root, "add", ".gitignore", "artifact/CHECKSUMS.sha256")

            result = CHECKER.check_validation_artifact_hygiene(
                repository_root=repository_root,
                active_contract_manifest=repository_root / "missing.json",
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "checksum_target_untracked",
            {item["rule"] for item in result["violations"]},
        )

    def test_default_discovery_does_not_scan_untracked_run_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository_root = Path(tmp)
            artifact_root = repository_root / "artifact"
            artifact_root.mkdir()
            tracked = artifact_root / "result.json"
            tracked.write_text('{"status": "passed"}\n', encoding="utf-8")
            _write_checksums(artifact_root, [tracked])
            _git(repository_root, "init", "--quiet")
            _git(
                repository_root,
                "add",
                "artifact/CHECKSUMS.sha256",
                "artifact/result.json",
            )
            (artifact_root / "large_local_run.json").write_text(
                '{"path": "D:\\\\working\\\\local-only"}\n',
                encoding="utf-8",
            )

            result = CHECKER.check_validation_artifact_hygiene(
                repository_root=repository_root,
                active_contract_manifest=repository_root / "missing.json",
            )

        self.assertEqual(result["status"], "passed", result["violations"])

    def test_default_discovery_fails_when_repository_has_no_tracked_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository_root = Path(tmp)
            (repository_root / "README.md").write_text("fixture\n", encoding="utf-8")
            _git(repository_root, "init", "--quiet")
            _git(repository_root, "add", "README.md")

            result = CHECKER.check_validation_artifact_hygiene(
                repository_root=repository_root,
                active_contract_manifest=repository_root / "missing.json",
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "tracked_manifests_missing",
            {item["rule"] for item in result["violations"]},
        )


def _write_checksums(root: Path, files: list[Path]) -> None:
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "CHECKSUMS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    unittest.main()
