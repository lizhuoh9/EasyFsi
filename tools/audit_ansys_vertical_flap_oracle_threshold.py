#!/usr/bin/env python3
"""Taichi-free CLI for the R24C displacement and threshold audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit R24C displacement and carry-to-oracle thresholds."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    displacement = subparsers.add_parser(
        "displacement",
        help="Recompute accepted displacement metrics from a source-matched Q0/Q3 pair.",
    )
    displacement.add_argument("--q0-root", type=Path, required=True)
    displacement.add_argument("--q3-root", type=Path, required=True)
    displacement.add_argument("--source-commit", required=True)
    displacement.add_argument("--r24b-bundle-root", type=Path, required=True)
    displacement.add_argument("--output", type=Path, required=True)

    verify_displacement = subparsers.add_parser(
        "verify-displacement",
        help="Recompute a self-hashed displacement artifact from sealed roots.",
    )
    verify_displacement.add_argument("--output", type=Path, required=True)

    threshold = subparsers.add_parser(
        "threshold",
        help="Validate nine terminal probes and write hash-bound local evidence.",
    )
    _add_threshold_roots(threshold)
    threshold.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify-threshold",
        help="Recompute threshold evidence from the roots bound in its manifest.",
    )
    verify.add_argument("--output-dir", type=Path, required=True)

    reuse = subparsers.add_parser(
        "reuse",
        help=(
            "Write the conditional IQN-reuse terminal record or verified "
            "carry/oracle by reuse off/on exact8 matrix."
        ),
    )
    reuse.add_argument("--threshold-evidence-dir", type=Path, required=True)
    for arm in (
        "carry-reuse-off",
        "carry-reuse-on",
        "oracle-reuse-off",
        "oracle-reuse-on",
    ):
        reuse.add_argument(f"--{arm}", type=Path)
    reuse.add_argument("--output", type=Path, required=True)

    verify_reuse = subparsers.add_parser(
        "verify-reuse",
        help="Recompute a conditional IQN-reuse artifact from its bound roots.",
    )
    verify_reuse.add_argument("--output", type=Path, required=True)

    publication = subparsers.add_parser(
        "publication",
        help="Write a path-free summary from verified local R24C evidence.",
    )
    publication.add_argument("--displacement-evidence", type=Path, required=True)
    publication.add_argument("--threshold-evidence-dir", type=Path, required=True)
    publication.add_argument("--output", type=Path, required=True)
    return parser


def _add_threshold_roots(parser: argparse.ArgumentParser) -> None:
    for tag in ("050", "075", "100"):
        parser.add_argument(f"--q0-omega{tag}", type=Path, required=True)
        for step in ("02", "05", "08"):
            parser.add_argument(
                f"--probe-omega{tag}-step{step}",
                type=Path,
                required=True,
            )


def _threshold_roots(
    args: argparse.Namespace,
) -> tuple[dict[float, Path], dict[tuple[float, int], Path]]:
    omega_tags = ((0.5, "050"), (0.75, "075"), (1.0, "100"))
    q0 = {
        omega: getattr(args, f"q0_omega{tag}")
        for omega, tag in omega_tags
    }
    probes = {
        (omega, step): getattr(args, f"probe_omega{tag}_step{step:02d}")
        for omega, tag in omega_tags
        for step in (2, 5, 8)
    }
    return q0, probes


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_common import (
        OracleThresholdContractError,
    )
    from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_evidence import (
        verify_threshold_evidence,
        write_threshold_evidence,
    )
    from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_publication import (
        write_publication_projection,
    )
    from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_reuse_evidence import (
        verify_reuse_evidence,
        write_reuse_evidence,
    )
    from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_displacement_evidence import (
        verify_displacement_evidence,
        write_displacement_evidence,
    )

    try:
        if args.command == "displacement":
            artifact_sha256 = write_displacement_evidence(
                args.q0_root,
                args.q3_root,
                source_commit=args.source_commit,
                sealed_r24b_bundle_root=args.r24b_bundle_root,
                output_path=args.output,
            )
            _emit(
                {
                    "artifact_sha256": artifact_sha256,
                    "verified": verify_displacement_evidence(args.output),
                }
            )
        elif args.command == "verify-displacement":
            _emit(verify_displacement_evidence(args.output))
        elif args.command == "threshold":
            q0_roots, probe_roots = _threshold_roots(args)
            artifact_sha256 = write_threshold_evidence(
                q0_roots,
                probe_roots,
                args.output_dir,
            )
            _emit(
                {
                    "artifact_sha256": artifact_sha256,
                    "verified": verify_threshold_evidence(args.output_dir),
                }
            )
        elif args.command == "verify-threshold":
            _emit(verify_threshold_evidence(args.output_dir))
        elif args.command == "reuse":
            raw_roots = {
                name: getattr(args, name)
                for name in (
                    "carry_reuse_off",
                    "carry_reuse_on",
                    "oracle_reuse_off",
                    "oracle_reuse_on",
                )
                if getattr(args, name) is not None
            }
            artifact_sha256 = write_reuse_evidence(
                args.threshold_evidence_dir,
                raw_roots or None,
                args.output,
            )
            _emit(
                {
                    "artifact_sha256": artifact_sha256,
                    "verified": verify_reuse_evidence(args.output),
                }
            )
        elif args.command == "verify-reuse":
            _emit(verify_reuse_evidence(args.output))
        else:
            _emit(
                {
                    "artifact_sha256": write_publication_projection(
                        args.displacement_evidence,
                        args.threshold_evidence_dir,
                        args.output,
                    ),
                    "bottom_up_reverification": False,
                    "output": str(args.output.resolve()),
                }
            )
    except OracleThresholdContractError as exc:
        _emit({"error": str(exc), "status": "CONTRACT_ERROR"})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
