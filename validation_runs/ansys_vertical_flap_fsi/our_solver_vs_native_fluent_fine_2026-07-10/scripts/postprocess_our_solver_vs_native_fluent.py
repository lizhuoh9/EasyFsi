"""Build an offline our-solver vs native-Fluent fine-grid diagnostic bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cases" / "ansys_vertical_flap_fsi.py").is_file() and (
            parent / "src" / "refactored"
        ).is_dir():
            return parent
    raise RuntimeError("could not locate the refactored repository root")


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from refactored.validation.ansys_vertical_flap_fsi.native_fine_contracts import (  # noqa: E402
    CANONICAL_NATIVE_FLUENT_POSTPROCESS_RELATIVE_DIR,
)

LOCKED_NATIVE_FLUENT_POSTPROCESS_DIR = (
    REPO_ROOT / CANONICAL_NATIVE_FLUENT_POSTPROCESS_RELATIVE_DIR
)

from refactored.validation.ansys_vertical_flap_fsi.native_fine_comparison import (  # noqa: E402
    NativeFineComparisonError,
    postprocess_native_fine_comparison,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an exactly completed 50-step our-solver fine run to the locked "
            "native Fluent fine-grid postprocess bundle without launching either solver"
        )
    )
    parser.add_argument(
        "--our-run-dir",
        required=True,
        help="Completed our-solver run containing step_fields and history",
    )
    parser.add_argument(
        "--fluent-postprocess-dir",
        default=str(LOCKED_NATIVE_FLUENT_POSTPROCESS_DIR),
        help=(
            "Locked native Fluent postprocess directory containing fields/history; "
            "defaults to the latest validated native fresh50 bundle"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory; existing paths are refused",
    )
    parser.add_argument(
        "--fluent-force-history",
        help=(
            "Explicit native Fluent fsi_production/history.csv; otherwise resolved "
            "strictly from the locked postprocess input manifest run_dir"
        ),
    )
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    parser.add_argument("--gif-max-width", type=int, default=1600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = postprocess_native_fine_comparison(
            args.our_run_dir,
            args.fluent_postprocess_dir,
            args.output_dir,
            expected_steps=50,
            velocity_vmax_mps=31.0,
            gif_duration_ms=args.gif_duration_ms,
            gif_max_width_px=args.gif_max_width,
            pressure_semantics_mode="strict",
            fluent_force_history_path=args.fluent_force_history,
        )
    except NativeFineComparisonError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"POSTPROCESS FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    five_percent_gate = report.get("five_percent_diagnostic_gate")
    if not isinstance(five_percent_gate, dict) or (
        five_percent_gate.get("all_metrics_within_tolerance") is not True
    ):
        print(
            "VALIDATION FAILED: one or more required diagnostics exceed 5%",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
