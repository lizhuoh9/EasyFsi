from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
ACTIVE_CONTRACT_MANIFEST = ROOT / "fluent_reference" / "active_fluent_reference_contract.json"
CONTENT_HYGIENE_ROOTS = (
    ROOT / "fluent_reference" / "validation_diagnostics",
    ROOT / "traction_selected_formulation_fluent_parity_diagnostics",
)
# Checksum/tracking coverage is repository-wide.  Content scanning remains on
# the promoted artifact surfaces so historical raw failure traces retain their
# original provenance text.
TEXT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sha256",
    ".txt",
}

WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:\\")
SECRET_RE = re.compile(r"(?i)(api[_-]?key|password|secret|token)")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def check_validation_artifact_hygiene(
    roots: Iterable[Path] | None = None,
    *,
    active_contract_manifest: Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    violations: list[dict[str, str]] = []
    checked_files: list[str] = []
    tracked_files: set[Path] | None = None
    if roots is None:
        tracked_files = _git_tracked_files(repository_root)
        manifests = sorted(
            path
            for path in tracked_files
            if path.name == "CHECKSUMS.sha256"
        )
        if not manifests:
            _add_violation(
                violations,
                repository_root / "CHECKSUMS.sha256",
                "tracked_manifests_missing",
                "repository has no tracked CHECKSUMS.sha256 manifest",
            )
    else:
        manifests = [Path(root).resolve() / "CHECKSUMS.sha256" for root in roots]
    if active_contract_manifest is None:
        active_contract_manifest = repository_root / ACTIVE_CONTRACT_MANIFEST
    active_incomplete = _active_contract_incomplete(active_contract_manifest)

    for checksum_path in manifests:
        root = checksum_path.parent
        check_content = tracked_files is None or any(
            root == (repository_root / content_root).resolve()
            for content_root in CONTENT_HYGIENE_ROOTS
        )
        if tracked_files is None:
            files = sorted(item for item in root.rglob("*") if item.is_file())
        else:
            files = sorted(path for path in tracked_files if path.is_relative_to(root))
        for path in files:
            checked_files.append(_display_path(path, repository_root))
            if not check_content or path.suffix.lower() not in TEXT_ARTIFACT_SUFFIXES:
                continue
            text = _read_utf8(path, violations)
            if text is None:
                continue
            _check_text(path, text, active_incomplete, violations)
        _check_checksums(root, violations, tracked_files=tracked_files)

    return {
        "checker": "check_validation_artifact_hygiene",
        "policy": "validation_artifact_hygiene_v1",
        "policy_id": "validation_artifact_hygiene_v1",
        "checked_file_count": len(checked_files),
        "checked_files": checked_files,
        "checked_manifests": [
            _display_path(path, repository_root) for path in manifests
        ],
        "status": "passed" if not violations else "failed",
        "violations": violations,
    }


def _read_utf8(path: Path, violations: list[dict[str, str]]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        _add_violation(violations, path, "utf8_decode_failed", str(exc))
        return None


def _check_text(
    path: Path,
    text: str,
    active_incomplete: bool,
    violations: list[dict[str, str]],
) -> None:
    if WINDOWS_ABSOLUTE_RE.search(text) or "D:\\working" in text:
        _add_violation(
            violations,
            path,
            "local_absolute_path",
            "generated artifact contains a local Windows path",
        )
    if active_incomplete and "fluent_parity_validated" in text:
        _add_violation(
            violations,
            path,
            "overclaimed_fluent_parity",
            "active contract is incomplete but artifact mentions fluent_parity_validated",
        )
    for match in SECRET_RE.finditer(text):
        token = match.group(0)
        if _allowed_secret_like_text(token, text):
            continue
        _add_violation(
            violations,
            path,
            "secret_like_text",
            f"matched {token}",
        )


def _allowed_secret_like_text(token: str, text: str) -> bool:
    if token.lower() == "token" and "synthetic-test-only" in text:
        return True
    return bool(SHA256_RE.fullmatch(token))


def _check_checksums(
    root: Path,
    violations: list[dict[str, str]],
    *,
    tracked_files: set[Path] | None,
) -> None:
    checksum_path = root / "CHECKSUMS.sha256"
    if not checksum_path.exists():
        _add_violation(
            violations,
            checksum_path,
            "checksums_missing",
            "CHECKSUMS.sha256 is missing",
        )
        return
    listed_paths: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, name = line.split("  ", 1)
        except ValueError:
            _add_violation(violations, checksum_path, "checksum_row_invalid", line)
            continue
        if not SHA256_RE.fullmatch(expected):
            _add_violation(
                violations,
                checksum_path,
                "checksum_digest_invalid",
                expected,
            )
            continue
        normalized_name = Path(name).as_posix()
        if normalized_name in listed_paths:
            _add_violation(
                violations,
                checksum_path,
                "checksum_target_duplicate",
                normalized_name,
            )
            continue
        listed_paths.add(normalized_name)
        root_resolved = root.resolve()
        target = (root / name).resolve()
        if not target.is_relative_to(root_resolved):
            _add_violation(
                violations,
                checksum_path,
                "checksum_target_outside_root",
                name,
            )
            continue
        if tracked_files is not None and target not in tracked_files:
            _add_violation(
                violations,
                checksum_path,
                "checksum_target_untracked",
                name,
            )
            continue
        if not target.is_file():
            _add_violation(
                violations,
                checksum_path,
                "checksum_target_missing",
                name,
            )
            continue
        actual = _sha256_file(target)
        if actual != expected:
            _add_violation(
                violations,
                checksum_path,
                "checksum_mismatch",
                f"{name}: expected {expected}, got {actual}",
            )

    if tracked_files is None:
        managed_files = (
            path for path in root.rglob("*") if path.is_file()
        )
    else:
        managed_files = (
            path for path in tracked_files if path.is_relative_to(root)
        )
    managed_paths = {
        path.relative_to(root).as_posix()
        for path in managed_files
        if path != checksum_path
    }
    for name in sorted(managed_paths - listed_paths):
        _add_violation(
            violations,
            checksum_path,
            "checksum_entry_missing",
            name,
        )


def _active_contract_incomplete(path: Path) -> bool:
    if not path.exists():
        return True
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("active_contract_status") != "fluent_reference_complete"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_tracked_files(repository_root: Path) -> set[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return {
        (repository_root / name.decode("utf-8")).resolve()
        for name in completed.stdout.split(b"\0")
        if name
    }


def _display_path(path: Path, repository_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.as_posix()


def _add_violation(
    violations: list[dict[str, str]],
    path: Path,
    rule: str,
    detail: str,
) -> None:
    violations.append(
        {
            "path": path.as_posix(),
            "rule": rule,
            "detail": detail,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-report",
        type=Path,
        help="Optional path to write the checker JSON report.",
    )
    args = parser.parse_args()

    result = check_validation_artifact_hygiene()
    if args.write_report is not None:
        _write_json_report(args.write_report, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


def _write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
