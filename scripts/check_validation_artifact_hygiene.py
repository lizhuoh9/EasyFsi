from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.validation.ansys_vertical_flap import validation_artifact_hygiene as _impl  # noqa: E402
from tools.validation.ansys_vertical_flap.validation_artifact_hygiene import *  # noqa: F401,F403,E402
from tools.validation.ansys_vertical_flap.policy_report_writer import write_json_report  # noqa: E402


def __getattr__(name: str):
    return getattr(_impl, name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-report",
        type=Path,
        help="Optional path to write the checker JSON report.",
    )
    args = parser.parse_args()

    result = _impl.check_validation_artifact_hygiene()
    if args.write_report is not None:
        write_json_report(args.write_report, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
