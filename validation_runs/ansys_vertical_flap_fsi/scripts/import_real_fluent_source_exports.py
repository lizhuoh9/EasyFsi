from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_fluent_reference_collection_validation as collection


SOURCE_EXPORTS_ROOT = collection.SOURCE_EXPORTS_ROOT
CURRENT_CONTRACT_JSON = collection.CURRENT_CONTRACT_JSON
OUTPUT_DIR = collection.OUTPUT_DIR
ACTIVE_CONTRACT_MANIFEST_JSON = collection.ACTIVE_CONTRACT_MANIFEST_JSON
REQUIRED_METADATA = "fluent_metadata_2026-06-28.md"
PUBLIC_EVIDENCE = "public_tutorial_evidence_map.json"
REQUIRED_ARTIFACTS = tuple(str(spec["artifact"]) for spec in collection.METRIC_SPECS)
REQUIRED_FILES = REQUIRED_ARTIFACTS + (REQUIRED_METADATA,)


class ImportPreflightError(RuntimeError):
    def __init__(self, summary: Mapping[str, Any]):
        super().__init__("real Fluent source export import preflight failed")
        self.summary = dict(summary)


def validate_import_bundle(
    input_dir: Path,
    *,
    current_contract_json: Path = CURRENT_CONTRACT_JSON,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        staged_source_exports = tmp_root / "source_exports"
        _prepare_validation_source_exports(
            input_dir=input_dir,
            staging_dir=staged_source_exports,
        )
        payload = collection.run_with_paths(
            source_exports_root=staged_source_exports,
            current_contract_json=Path(current_contract_json),
            output_dir=tmp_root / "diagnostics",
            active_manifest_json=tmp_root / "active_fluent_reference_contract.json",
        )

    summary = _preflight_summary(input_dir=input_dir, payload=payload)
    if not summary["ready"]:
        raise ImportPreflightError(summary)
    return summary


def import_real_fluent_source_exports(
    *,
    input_dir: Path,
    destination_dir: Path = SOURCE_EXPORTS_ROOT,
    current_contract_json: Path = CURRENT_CONTRACT_JSON,
    output_dir: Path = OUTPUT_DIR,
    active_manifest_json: Path = ACTIVE_CONTRACT_MANIFEST_JSON,
    run_collection_validator: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    destination_dir = Path(destination_dir)
    current_contract_json = Path(current_contract_json)
    output_dir = Path(output_dir)
    active_manifest_json = Path(active_manifest_json)

    preflight = validate_import_bundle(
        input_dir,
        current_contract_json=current_contract_json,
    )
    if run_collection_validator:
        _validate_ready_after_staged_copy(
            input_dir=input_dir,
            current_contract_json=current_contract_json,
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_required_files(input_dir=input_dir, destination_dir=destination_dir)
    _ensure_public_evidence_map(destination_dir)

    collection_payload: dict[str, Any] | None = None
    if run_collection_validator:
        collection_payload = collection.run_with_paths(
            source_exports_root=destination_dir,
            current_contract_json=current_contract_json,
            output_dir=output_dir,
            active_manifest_json=active_manifest_json,
        )
        if not _collection_ready(collection_payload):
            summary = {
                "ready": False,
                "input_dir": input_dir.as_posix(),
                "destination_dir": destination_dir.as_posix(),
                "copied_files": copied,
                "copied_file_count": len(copied),
                "blockers": _collection_blockers(collection_payload),
                "collection": collection_payload,
            }
            raise ImportPreflightError(summary)

    return {
        "ready": True,
        "input_dir": input_dir.as_posix(),
        "destination_dir": destination_dir.as_posix(),
        "copied_files": copied,
        "copied_file_count": len(copied),
        "preflight": preflight,
        "collection": collection_payload,
    }


def _validate_ready_after_staged_copy(
    *,
    input_dir: Path,
    current_contract_json: Path,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        staged_source_exports = tmp_root / "source_exports"
        staged_output = tmp_root / "diagnostics"
        staged_active_manifest = tmp_root / "active_fluent_reference_contract.json"
        _copy_required_files(input_dir=input_dir, destination_dir=staged_source_exports)
        _ensure_public_evidence_map(staged_source_exports)
        payload = collection.run_with_paths(
            source_exports_root=staged_source_exports,
            current_contract_json=current_contract_json,
            output_dir=staged_output,
            active_manifest_json=staged_active_manifest,
        )
        if _collection_ready(payload):
            return
        raise ImportPreflightError(
            {
                "ready": False,
                "input_dir": input_dir.as_posix(),
                "destination_dir": staged_source_exports.as_posix(),
                "blockers": _collection_blockers(payload),
                "collection": payload,
            }
        )


def _preflight_summary(
    *,
    input_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = _preflight_blockers(payload)
    return {
        "ready": not blockers,
        "input_dir": input_dir.as_posix(),
        "required_files": list(REQUIRED_FILES),
        "blockers": blockers,
        "source_checks": list(payload["source_checks"]),
        "metadata_check": dict(payload["metadata_check"]),
        "real_fluent_import_gate": dict(payload["real_fluent_import_gate"]),
    }


def _preflight_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: set[str] = set()
    for check in payload["source_checks"]:
        if check["file_status"] != "present_complete":
            blockers.add(str(check["file_status"]))
        for blocker in check.get("schema_blockers", []):
            blockers.add(str(blocker))

    metadata = payload["metadata_check"]
    metadata_blocker = metadata.get("blocker")
    if metadata["provenance_status"] != "complete":
        blockers.add(str(metadata_blocker))
    if metadata.get("disallowed_provenance"):
        blockers.add(str(collection.BLOCKER_METADATA_DISALLOWED_PROVENANCE))
    for missing in metadata.get("missing_fields", []):
        blockers.add(f"metadata_missing:{missing}")
    return sorted(blocker for blocker in blockers if blocker and blocker != "None")


def _collection_ready(payload: Mapping[str, Any]) -> bool:
    gate = payload["real_fluent_import_gate"]
    return (
        payload["candidate_status"] == collection.STATUS_COMPLETE
        and payload["candidate_contract_status"] == collection.CONTRACT_COMPLETE
        and payload["promotion_status"] == "ready_for_versioned_contract_promotion"
        and gate["status"] == "ready_for_real_fluent_import"
        and gate["can_import_real_fluent_reference"]
        and gate["can_run_solver_evaluation"]
        and not gate["fluent_parity_claimed"]
        and not gate["blockers"]
    )


def _collection_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers = {str(item["blocker"]) for item in payload.get("candidate_blockers", [])}
    blockers.update(
        str(item["blocker"]) for item in payload.get("promotion_blockers", [])
    )
    blockers.update(
        str(item["blocker"])
        for item in payload["real_fluent_import_gate"].get("blockers", [])
    )
    return sorted(blocker for blocker in blockers if blocker and blocker != "None")


def _copy_required_files(
    *,
    input_dir: Path,
    destination_dir: Path,
) -> list[str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in REQUIRED_FILES:
        source = input_dir / name
        destination = destination_dir / name
        shutil.copy2(source, destination)
        copied.append(destination.as_posix())
    return copied


def _prepare_validation_source_exports(
    *,
    input_dir: Path,
    staging_dir: Path,
) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        source = input_dir / name
        if source.exists():
            shutil.copy2(source, staging_dir / name)
    _ensure_public_evidence_map(staging_dir)


def _ensure_public_evidence_map(destination_dir: Path) -> None:
    destination = destination_dir / PUBLIC_EVIDENCE
    if destination.exists():
        return
    source = SOURCE_EXPORTS_ROOT / PUBLIC_EVIDENCE
    if not source.exists():
        raise FileNotFoundError(source)
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _cli_preflight_summary(
    preflight: Mapping[str, Any],
    *,
    destination_dir: Path,
) -> dict[str, Any]:
    return {
        **dict(preflight),
        "mode": "preflight",
        "destination_dir": Path(destination_dir).as_posix(),
        "copied_files": [],
        "copied_file_count": 0,
        "collection": None,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or import validated real ANSYS Fluent vertical-flap "
            "source exports."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=SOURCE_EXPORTS_ROOT,
    )
    parser.add_argument(
        "--current-contract-json",
        type=Path,
        default=CURRENT_CONTRACT_JSON,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--active-manifest-json",
        type=Path,
        default=ACTIVE_CONTRACT_MANIFEST_JSON,
    )
    parser.add_argument("--run-collection-validator", action="store_true")
    parser.add_argument(
        "--commit-import",
        action="store_true",
        help="Copy validated files into the destination after preflight passes.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.commit_import:
            imported = import_real_fluent_source_exports(
                input_dir=args.input_dir,
                destination_dir=args.destination_dir,
                current_contract_json=args.current_contract_json,
                output_dir=args.output_dir,
                active_manifest_json=args.active_manifest_json,
                run_collection_validator=args.run_collection_validator,
            )
            summary = {**imported, "mode": "commit_import"}
        else:
            preflight = validate_import_bundle(
                args.input_dir,
                current_contract_json=args.current_contract_json,
            )
            summary = _cli_preflight_summary(
                preflight,
                destination_dir=args.destination_dir,
            )
    except ImportPreflightError as exc:
        failure_summary = dict(exc.summary)
        if args.commit_import:
            failure_summary = {**failure_summary, "mode": "commit_import"}
        else:
            failure_summary = _cli_preflight_summary(
                failure_summary,
                destination_dir=args.destination_dir,
            )
        print(json.dumps(failure_summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
