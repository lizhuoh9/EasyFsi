from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DEFAULT_ROOTS = (
    ROOT / "fluent_reference" / "validation_diagnostics",
    ROOT / "traction_selected_formulation_fluent_parity_diagnostics",
)
ACTIVE_CONTRACT_MANIFEST = ROOT / "fluent_reference" / "active_fluent_reference_contract.json"

WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:\\")
SECRET_RE = re.compile(r"(?i)(api[_-]?key|password|secret|token)")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def check_validation_artifact_hygiene(
    roots: Iterable[Path] = DEFAULT_ROOTS,
    *,
    active_contract_manifest: Path = ACTIVE_CONTRACT_MANIFEST,
) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    checked_files: list[str] = []
    active_incomplete = _active_contract_incomplete(active_contract_manifest)

    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            checked_files.append(path.as_posix())
            text = _read_utf8(path, violations)
            if text is None:
                continue
            _check_text(path, text, active_incomplete, violations)
        _check_checksums(root, violations)

    return {
        "checker": "check_validation_artifact_hygiene",
        "policy": "validation_artifact_hygiene_v1",
        "policy_id": "validation_artifact_hygiene_v1",
        "checked_file_count": len(checked_files),
        "checked_files": checked_files,
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


def _check_checksums(root: Path, violations: list[dict[str, str]]) -> None:
    checksum_path = root / "CHECKSUMS.sha256"
    if not checksum_path.exists():
        _add_violation(
            violations,
            checksum_path,
            "checksums_missing",
            "CHECKSUMS.sha256 is missing",
        )
        return
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, name = line.split("  ", 1)
        except ValueError:
            _add_violation(violations, checksum_path, "checksum_row_invalid", line)
            continue
        target = root / name
        if not target.exists():
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


def _active_contract_incomplete(path: Path) -> bool:
    if not path.exists():
        return True
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("active_contract_status") != "fluent_reference_complete"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
