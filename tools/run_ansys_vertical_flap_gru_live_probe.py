"""Generate the two frozen R25B source-matched candidate bundles on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path = [_REPO_ROOT, *sys.path]

from tools.validation.gru_kalman_live.candidate_generation import (
    generate_candidate_bundles,
    load_frozen_r25a_selection,
    load_generation_inputs,
    load_source_matched_marker_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the fixed G0-M/GDelta-M controls and generate the "
            "13-arm step-7/8 no-commit candidate bundles."
        )
    )
    parser.add_argument("--d0-canonical", required=True)
    parser.add_argument("--d0-attempt", required=True)
    parser.add_argument("--exact8-canonical", required=True)
    parser.add_argument("--exact8-attempt", required=True)
    parser.add_argument(
        "--exact8-step-fields",
        default=None,
        help=(
            "Exact source-matched step_fields directory; defaults to "
            "<exact8-canonical>/step_fields."
        ),
    )
    parser.add_argument("--r25a-selection-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    exact8_canonical = Path(args.exact8_canonical).expanduser().resolve()
    step_fields = (
        exact8_canonical / "step_fields"
        if args.exact8_step_fields is None
        else Path(args.exact8_step_fields).expanduser().resolve()
    )
    d0_trace, exact8_trace = load_generation_inputs(
        d0_canonical=Path(args.d0_canonical).expanduser().resolve(),
        d0_attempt=Path(args.d0_attempt).expanduser().resolve(),
        exact8_canonical=exact8_canonical,
        exact8_attempt=Path(args.exact8_attempt).expanduser().resolve(),
    )
    marker_identity = load_source_matched_marker_identity(
        step_fields,
        expected_steps=8,
    )
    selection = load_frozen_r25a_selection(
        Path(args.r25a_selection_root).expanduser().resolve()
    )
    manifests = generate_candidate_bundles(
        d0_trace=d0_trace,
        exact8_trace=exact8_trace,
        marker_identity=marker_identity,
        selection=selection,
        output_root=Path(args.output_root).expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "offline_candidates": True,
                "deployable": False,
                "candidate_manifests": [str(path) for path in manifests],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
