from __future__ import annotations

import argparse
from pathlib import Path

from cases.squid_jet_render import render


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render the squid FSI output using the fixed nozzle-jet centerline style. "
            "This intentionally replaces the older 3D surface GIF view."
        )
    )
    parser.add_argument("--visit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--duration-ms", type=int, default=70)
    parser.add_argument("--min-age-s", type=float, default=5.0)
    parser.add_argument("--scalar", choices=("downward", "speed"), default="downward")
    parser.add_argument("--vmin-mps", type=float, default=0.005)
    parser.add_argument("--vmax-mps", type=float, default=0.12)
    parser.add_argument("--interp-frames", type=int, default=3)
    args = parser.parse_args()

    summary = render(
        args.visit_dir.resolve(),
        args.output_dir.resolve(),
        min_age_s=args.min_age_s,
        duration_ms=args.duration_ms,
        scalar=args.scalar,
        vmin_mps=args.vmin_mps,
        vmax_mps=args.vmax_mps,
        interp_frames=args.interp_frames,
    )
    print(f"gif {summary['gif']}")
    print(
        "frames "
        f"{summary['gif_frame_count']} "
        f"raw {summary['raw_frame_count']} "
        f"first {summary['first_step']:06d} "
        f"latest {summary['last_step']:06d}"
    )
    print(
        "jet_scalar "
        f"{summary['scalar_description']} "
        f"observed_p99={summary['observed_p99_mps']:.6g}m/s "
        f"observed_max={summary['observed_max_mps']:.6g}m/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
