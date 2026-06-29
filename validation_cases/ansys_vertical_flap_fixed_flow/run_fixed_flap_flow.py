from __future__ import annotations

import json
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").is_dir() and (parent / "validation_cases").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refactored.validation.ansys_vertical_flap_fixed.preprocess_fixed_flap import (  # noqa: E402
    load_config,
    run_preprocess,
)


def main() -> int:
    config_path = Path(__file__).with_name("config.yaml")
    result = run_preprocess(load_config(config_path))
    print(
        json.dumps(
            {
                "case": "ansys_vertical_flap_fixed_flow",
                "output_root": result["output_root"],
                "manifest": result["manifest_path"],
                "scope": "fixed-flap flow preprocessing only; no solver step executed",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
