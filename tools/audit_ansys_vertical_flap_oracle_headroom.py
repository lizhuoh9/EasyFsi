#!/usr/bin/env python3
"""CLI for the Taichi-free R24B oracle-headroom evidence audit."""

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
        description="Audit source-matched exact8 Q0/Q3 oracle headroom evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze Q0/Q3 and write the four deterministic decision artifacts.",
    )
    analyze.add_argument("--q0-root", type=Path, required=True)
    analyze.add_argument("--q3-root", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)

    blend = subparsers.add_parser(
        "prepare-blend",
        help="Prepare a sealed non-deployable Q0-to-oracle alpha trajectory.",
    )
    blend.add_argument("--q0-root", type=Path, required=True)
    blend.add_argument("--output-dir", type=Path, required=True)
    blend.add_argument("--alpha", type=float, required=True)

    curve = subparsers.add_parser(
        "complete-blends",
        help="Audit the conditional alpha 0.25/0.50/0.75 response curve.",
    )
    curve.add_argument("--q0-root", type=Path, required=True)
    curve.add_argument("--q3-root", type=Path, required=True)
    curve.add_argument("--alpha025-producer", type=Path, required=True)
    curve.add_argument("--alpha025-run", type=Path, required=True)
    curve.add_argument("--alpha050-producer", type=Path, required=True)
    curve.add_argument("--alpha050-run", type=Path, required=True)
    curve.add_argument("--alpha075-producer", type=Path, required=True)
    curve.add_argument("--alpha075-run", type=Path, required=True)
    curve.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify",
        help="Recompute the bundle from its live Q0/Q3 and alpha run identities.",
    )
    verify.add_argument("--output-dir", type=Path, required=True)
    return parser


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from src.refactored.validation.ansys_vertical_flap_fsi.kalman_oracle_headroom import (
        OracleHeadroomContractError,
        complete_oracle_blend_response,
        prepare_oracle_blend,
        run_oracle_headroom_campaign,
        verify_oracle_artifacts,
    )

    try:
        if args.command == "analyze":
            artifact_sha256 = run_oracle_headroom_campaign(
                q0_root=args.q0_root,
                q3_root=args.q3_root,
                output_dir=args.output_dir,
            )
            verified = verify_oracle_artifacts(args.output_dir)
            _emit(
                {
                    "artifact_sha256": artifact_sha256,
                    "verified": verified,
                }
            )
        elif args.command == "prepare-blend":
            _emit(
                prepare_oracle_blend(
                    args.q0_root,
                    args.output_dir,
                    alpha=args.alpha,
                )
            )
        elif args.command == "complete-blends":
            _emit(
                complete_oracle_blend_response(
                    q0_root=args.q0_root,
                    q3_root=args.q3_root,
                    blend_producers={
                        0.25: args.alpha025_producer,
                        0.5: args.alpha050_producer,
                        0.75: args.alpha075_producer,
                    },
                    blend_runs={
                        0.25: args.alpha025_run,
                        0.5: args.alpha050_run,
                        0.75: args.alpha075_run,
                    },
                    output_dir=args.output_dir,
                )
            )
        else:
            _emit(verify_oracle_artifacts(args.output_dir))
    except OracleHeadroomContractError as exc:
        _emit({"error": str(exc), "status": "CONTRACT_ERROR"})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
