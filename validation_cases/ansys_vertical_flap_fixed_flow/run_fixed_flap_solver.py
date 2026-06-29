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
from refactored.validation.ansys_vertical_flap_fixed.projection_solver import (  # noqa: E402
    run_projection_solver,
)


RUNNER_SOLVER_CONFIG = {
    "solver": {
        "max_steps": 200,
        "cfl": 0.35,
        "steady_tolerance": 0.0,
        "divergence_tolerance": 1.0e-3,
        "poisson_max_iters": 80,
        "poisson_tolerance": 1.0e-5,
        "poisson_omega": 1.0,
        "history_interval": 10,
        "write_checkpoints": False,
    }
}


def main() -> int:
    config_path = Path(__file__).with_name("config.yaml")
    config = load_config(config_path)
    output_root = PROJECT_ROOT / config["output"]["root"]
    geometry_path = output_root / "preprocess" / "geometry_mask.npz"
    bc_path = output_root / "preprocess" / "bc_map.npz"
    if not geometry_path.exists() or not bc_path.exists():
        run_preprocess(config)

    result = run_projection_solver(
        geometry_path, bc_path, output_root, config=RUNNER_SOLVER_CONFIG
    )
    print(
        json.dumps(
            {
                "case": "ansys_vertical_flap_fixed_flow",
                "step": "step2_fixed_flap_projection_solver",
                "output_root": "validation_runs/ansys_vertical_flap_fixed_flow",
                "final_fields": "validation_runs/ansys_vertical_flap_fixed_flow/fields/final_fields.npz",
                "solver_history": "validation_runs/ansys_vertical_flap_fixed_flow/logs/solver_history.csv",
                "mass_balance": "validation_runs/ansys_vertical_flap_fixed_flow/logs/mass_balance.csv",
                "manifest": "validation_runs/ansys_vertical_flap_fixed_flow/case_manifest_step2.json",
                "claims": result["claims"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
