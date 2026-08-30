#!/usr/bin/env python3
"""Run the CPU-only R24 Kalman statistical calibration audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0-root", required=True, type=Path)
    parser.add_argument("--d0-attempt", required=True, type=Path)
    parser.add_argument("--d1-root", required=True, type=Path)
    parser.add_argument("--d1-attempt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--predictor-source",
        type=Path,
        default=_REPO_ROOT
        / "simulation_core"
        / "coupling"
        / "interface_kalman_predictor.py",
    )
    parser.add_argument("--fit-stop", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from src.refactored.validation.ansys_vertical_flap_fsi.kalman_statistical_reporting import (
        run_r24_campaign,
    )

    try:
        result = run_r24_campaign(
            d0_root=args.d0_root,
            d0_attempt=args.d0_attempt,
            d1_root=args.d1_root,
            d1_attempt=args.d1_attempt,
            output_dir=args.output_dir,
            predictor_source=args.predictor_source,
            fit_stop=args.fit_stop,
        )
    except ValueError as exc:
        print(f"R24 Kalman audit failed: {exc}", file=sys.stderr)
        return 2
    print(result["exit_classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
