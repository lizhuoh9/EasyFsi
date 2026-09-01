"""CLI for the frozen R25A CPU-only POD-GRU feasibility study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path = [_REPO_ROOT, *sys.path]

from tools.validation.gru_kalman.campaign import (
    CampaignConfig,
    CampaignContractError,
    DEFAULT_D0_ATTEMPT,
    DEFAULT_D0_CANONICAL,
    DEFAULT_D1_ATTEMPT,
    DEFAULT_D1_CANONICAL,
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    SEALED_MANIFEST_PATH,
    run_campaign,
)
from tools.validation.gru_kalman.dataset import DatasetContractError
from tools.validation.gru_kalman.models import parse_architectures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CPU-only R25A POD-GRU/Kalman-residual-GRU feasibility study."
    )
    parser.add_argument("--d0-canonical", default=DEFAULT_D0_CANONICAL)
    parser.add_argument("--d0-attempt", default=DEFAULT_D0_ATTEMPT)
    parser.add_argument("--d1-canonical", default=DEFAULT_D1_CANONICAL)
    parser.add_argument("--d1-attempt", default=DEFAULT_D1_ATTEMPT)
    parser.add_argument("--fit-stop", type=int, default=100)
    parser.add_argument("--pod-configs", default="4:4:8,8:4:16,8:8:16,16:8:16")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--models", default="pod_ar,gru,kalman0_gru,kalman1_gru")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--manifest", default=SEALED_MANIFEST_PATH)
    return parser


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise CampaignContractError("--seeds must be a comma-separated integer list") from exc


def _parse_models(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split(",") if token.strip())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = CampaignConfig(
            d0_canonical=Path(args.d0_canonical),
            d0_attempt=Path(args.d0_attempt),
            d1_canonical=Path(args.d1_canonical),
            d1_attempt=Path(args.d1_attempt),
            output_root=Path(args.output),
            report_path=Path(args.report),
            manifest_path=Path(args.manifest),
            fit_stop=args.fit_stop,
            pod_configs=parse_architectures(args.pod_configs),
            seeds=_parse_seeds(args.seeds),
            models=_parse_models(args.models),
        )
        result = run_campaign(config)
    except DatasetContractError as exc:
        print(f"R25A blocked: {exc}", file=sys.stderr)
        return 2
    print(f"R25A completed: {result.classifications['overall']}")
    print(f"output={result.output_root}")
    print(f"report={result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
