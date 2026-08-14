"""Offline post-process a paired fine-grid Fluent FSI campaign.

This command never starts Fluent. The input contract is::

    <run-dir>/steps/step_0001.cas.h5
    <run-dir>/steps/step_0001.dat.h5
    ...

Outputs are always written to a new directory; an existing destination is a
hard error so earlier validation artifacts cannot be overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "refactored").is_dir() and (
            parent / "cases" / "ansys_vertical_flap_fsi.py"
        ).is_file():
            return parent
    raise RuntimeError("could not locate refactored repository root")


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from refactored.validation.ansys_vertical_flap_fsi.fine_fsi_campaign import (  # noqa: E402
    CampaignValidationError,
    default_output_dir,
    postprocess_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and post-process paired Fluent fine-grid FSI HDF5 steps "
            "without launching Fluent"
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Campaign directory containing steps/step_XXXX.cas.h5 and .dat.h5",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "New output directory. Defaults to a timestamped postprocess_* "
            "directory below run-dir. Existing paths are refused."
        ),
    )
    parser.add_argument("--dt", type=float, default=5.0e-4, help="Time step in seconds")
    parser.add_argument(
        "--velocity-vmax",
        type=float,
        default=28.1,
        help="Fixed velocity color-scale maximum in m/s",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=120,
        help="Per-frame GIF duration in milliseconds",
    )
    parser.add_argument(
        "--gif-max-width",
        type=int,
        default=1600,
        help="Maximum encoded GIF width; source PNGs keep full resolution",
    )
    parser.add_argument("--target-x", type=float, default=0.0505)
    parser.add_argument("--target-y", type=float, default=0.0095)
    parser.add_argument("--nearest-node-count", type=int, default=4)
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=50,
        help="Required passed phase length; use 1 explicitly for the gate phase",
    )
    parser.add_argument(
        "--nonzero-displacement-tolerance",
        type=float,
        default=1.0e-18,
        help="Every step must exceed this maximum structural displacement in m",
    )
    parser.add_argument(
        "--fallback-scatter-renderer",
        action="store_true",
        help="Skip the existing polygon renderer and use the offline cell-center renderer",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_output_dir(run_dir)
    )
    try:
        summary = postprocess_campaign(
            run_dir,
            output_dir,
            repo_root=REPO_ROOT,
            dt_s=args.dt,
            velocity_vmax_mps=args.velocity_vmax,
            gif_duration_ms=args.gif_duration_ms,
            gif_max_width_px=args.gif_max_width,
            target_x_m=args.target_x,
            target_y_m=args.target_y,
            nearest_node_count=args.nearest_node_count,
            nonzero_tolerance_m=args.nonzero_displacement_tolerance,
            expected_steps=args.expected_steps,
            prefer_existing_polygon_renderer=not args.fallback_scatter_renderer,
        )
    except CampaignValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"POSTPROCESS FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
