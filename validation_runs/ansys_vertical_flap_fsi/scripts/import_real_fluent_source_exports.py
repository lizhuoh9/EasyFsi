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
BLOCKER_COLLECTION_VALIDATOR_REQUIRES_COMMIT_IMPORT = (
    "collection_validator_requires_commit_import"
)
BLOCKER_SOURCE_EXPORTS_COMMIT_REQUIRES_COLLECTION_VALIDATOR = (
    "source_exports_commit_requires_collection_validator"
)
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
    copied, collection_payload = _commit_required_files_atomically(
        input_dir=input_dir,
        destination_dir=destination_dir,
        current_contract_json=current_contract_json,
        output_dir=output_dir,
        active_manifest_json=active_manifest_json,
        run_collection_validator=run_collection_validator,
    )

    return {
        "ready": True,
        "input_dir": input_dir.as_posix(),
        "destination_dir": destination_dir.as_posix(),
        "copied_files": copied,
        "copied_file_count": len(copied),
        "preflight": preflight,
        "collection": collection_payload,
    }


def _commit_required_files_atomically(
    *,
    input_dir: Path,
    destination_dir: Path,
    current_contract_json: Path,
    output_dir: Path,
    active_manifest_json: Path,
    run_collection_validator: bool,
) -> tuple[list[str], dict[str, Any] | None]:
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_dir.name}.staging-",
            dir=destination_dir.parent,
        )
    )
    backup_dir: Path | None = None
    installed_staging = False
    copied = _destination_required_file_paths(destination_dir)
    collection_payload: dict[str, Any] | None = None

    try:
        _copy_required_files(input_dir=input_dir, destination_dir=staging_dir)
        _ensure_public_evidence_map(staging_dir)
        if run_collection_validator:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_root = Path(tmp)
                _validate_collection_ready(
                    source_exports_root=staging_dir,
                    input_dir=input_dir,
                    destination_dir=staging_dir,
                    current_contract_json=current_contract_json,
                    output_dir=tmp_root / "diagnostics",
                    active_manifest_json=(
                        tmp_root / "active_fluent_reference_contract.json"
                    ),
                    copied_files=copied,
                )

        if destination_dir.exists():
            backup_dir = _unique_sibling_path(destination_dir, "backup")
            destination_dir.replace(backup_dir)
        staging_dir.replace(destination_dir)
        installed_staging = True

        if run_collection_validator:
            collection_payload = _validate_collection_ready(
                source_exports_root=destination_dir,
                input_dir=input_dir,
                destination_dir=destination_dir,
                current_contract_json=current_contract_json,
                output_dir=output_dir,
                active_manifest_json=active_manifest_json,
                copied_files=copied,
            )

        if backup_dir is not None and backup_dir.exists():
            _remove_path_tree(backup_dir)
        return copied, collection_payload
    except Exception:
        if installed_staging:
            _remove_path_tree(destination_dir)
            if backup_dir is not None and backup_dir.exists():
                backup_dir.replace(destination_dir)
        elif (
            backup_dir is not None
            and backup_dir.exists()
            and not destination_dir.exists()
        ):
            backup_dir.replace(destination_dir)
        raise
    finally:
        if staging_dir.exists():
            _remove_path_tree(staging_dir)


def _validate_collection_ready(
    *,
    source_exports_root: Path,
    input_dir: Path,
    destination_dir: Path,
    current_contract_json: Path,
    output_dir: Path,
    active_manifest_json: Path,
    copied_files: list[str],
) -> dict[str, Any]:
    payload = collection.run_with_paths(
        source_exports_root=source_exports_root,
        current_contract_json=current_contract_json,
        output_dir=output_dir,
        active_manifest_json=active_manifest_json,
    )
    if _collection_ready(payload):
        return payload
    raise ImportPreflightError(
        {
            "ready": False,
            "input_dir": input_dir.as_posix(),
            "destination_dir": destination_dir.as_posix(),
            "copied_files": copied_files,
            "copied_file_count": len(copied_files),
            "blockers": _collection_blockers(payload),
            "collection": payload,
        }
    )


def _destination_required_file_paths(destination_dir: Path) -> list[str]:
    return [(destination_dir / name).as_posix() for name in REQUIRED_FILES]


def _unique_sibling_path(path: Path, suffix: str) -> Path:
    for index in range(1000):
        candidate = path.parent / f".{path.name}.{suffix}-{index}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate sibling {suffix} path for {path}")


def _remove_path_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


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


def _cli_invalid_preflight_option_summary(
    *,
    input_dir: Path,
    destination_dir: Path,
    blockers: Iterable[str],
) -> dict[str, Any]:
    return {
        "mode": "preflight",
        "ready": False,
        "input_dir": Path(input_dir).as_posix(),
        "destination_dir": Path(destination_dir).as_posix(),
        "blockers": list(blockers),
        "copied_files": [],
        "copied_file_count": 0,
        "collection": None,
    }


def _cli_invalid_commit_import_option_summary(
    *,
    input_dir: Path,
    destination_dir: Path,
    blockers: Iterable[str],
) -> dict[str, Any]:
    return {
        "mode": "commit_import",
        "ready": False,
        "input_dir": Path(input_dir).as_posix(),
        "destination_dir": Path(destination_dir).as_posix(),
        "blockers": list(blockers),
        "copied_files": [],
        "copied_file_count": 0,
        "collection": None,
    }


def _same_resolved_path(left: Path, right: Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


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

    if args.run_collection_validator and not args.commit_import:
        summary = _cli_invalid_preflight_option_summary(
            input_dir=args.input_dir,
            destination_dir=args.destination_dir,
            blockers=[BLOCKER_COLLECTION_VALIDATOR_REQUIRES_COMMIT_IMPORT],
        )
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    if (
        args.commit_import
        and not args.run_collection_validator
        and _same_resolved_path(args.destination_dir, SOURCE_EXPORTS_ROOT)
    ):
        summary = _cli_invalid_commit_import_option_summary(
            input_dir=args.input_dir,
            destination_dir=args.destination_dir,
            blockers=[BLOCKER_SOURCE_EXPORTS_COMMIT_REQUIRES_COLLECTION_VALIDATOR],
        )
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1

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
