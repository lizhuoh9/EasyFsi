"""Stop a diagnostic replay after projection-assembly HIBM health succeeds.

The target is fixed in this entry point.  It reuses the process-local sentinel,
base replay classification, snapshot hashing, and method-restoration contract
from the pre-predictor health probe without exposing an arbitrary context CLI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation_runs.ansys_vertical_flap_fsi.scripts import (
    run_preflow_snapshot_pre_predictor_health_probe as _health_probe_support,
)


DiagnosticReplayError = _health_probe_support.DiagnosticReplayError
DEFAULT_ALLOWED_SOURCE_DIFFS = _health_probe_support.DEFAULT_ALLOWED_SOURCE_DIFFS
PROBE_FILENAME = _health_probe_support.PROJECTION_ASSEMBLY_PROBE_FILENAME
PRE_PREDICTOR_HEALTH_CONTEXT = _health_probe_support.TARGET_HEALTH_CONTEXT
TARGET_HEALTH_CONTEXT = _health_probe_support.PROJECTION_ASSEMBLY_HEALTH_CONTEXT
_ProjectionAssemblyHealthGatePassed = (
    _health_probe_support._ProjectionAssemblyHealthGatePassed
)
_PROJECTION_ASSEMBLY_CONTRACT = (
    _health_probe_support._PROJECTION_ASSEMBLY_CONTRACT
)


def run_projection_assembly_health_probe(
    *,
    snapshot_path: str | Path,
    config_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    allowed_source_diffs: Sequence[str] = DEFAULT_ALLOWED_SOURCE_DIFFS,
) -> dict[str, Any]:
    """Run until the fixed projection-assembly health validator succeeds."""

    return _health_probe_support.run_hibm_health_gate_probe(
        snapshot_path=snapshot_path,
        config_path=config_path,
        source_manifest_path=source_manifest_path,
        output_dir=output_dir,
        contract=_PROJECTION_ASSEMBLY_CONTRACT,
        allowed_source_diffs=allowed_source_diffs,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostic-only ANSYS vertical-flap replay until the fixed "
            "production projection-assembly HIBM health gate succeeds; stop "
            "before the main pressure projection and complete no FSI step."
        )
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--source-manifest-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-source-diff", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    allowed = tuple(args.allow_source_diff or DEFAULT_ALLOWED_SOURCE_DIFFS)
    try:
        payload = run_projection_assembly_health_probe(
            snapshot_path=args.snapshot,
            config_path=args.config_json,
            source_manifest_path=args.source_manifest_json,
            output_dir=args.output_dir,
            allowed_source_diffs=allowed,
        )
    except Exception as exc:  # pragma: no cover - command-line failure path.
        print(
            f"[preflow_snapshot_projection_assembly_health_probe] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "[preflow_snapshot_projection_assembly_health_probe] wrote "
        "diagnostic-only gate evidence with status "
        f"{payload['status']!r}, "
        f"completed_fsi_steps={payload['completed_fsi_steps']}, "
        "projection_assembly_health_check_passed="
        f"{payload['projection_assembly_health_check_passed']}, "
        "stopped_before_main_pressure_projection="
        f"{payload['stopped_before_main_pressure_projection']}, "
        "full_fsi_step_completed="
        f"{payload['full_fsi_step_completed']} to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
