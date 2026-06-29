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

from refactored.validation.ansys_vertical_flap_fixed.postprocess_fluent_style import (  # noqa: E402
    run_fluent_style_postprocess,
)
from refactored.validation.ansys_vertical_flap_fixed.preprocess_fixed_flap import (  # noqa: E402
    load_config,
    run_preprocess,
)
from refactored.validation.ansys_vertical_flap_fixed.projection_solver import (  # noqa: E402
    run_projection_solver,
)


STEP2_SOLVER_CONFIG = {
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
    final_fields_path = output_root / "fields" / "final_fields.npz"
    solver_history_path = output_root / "logs" / "solver_history.csv"
    mass_balance_path = output_root / "logs" / "mass_balance.csv"
    step2_manifest_path = output_root / "case_manifest_step2.json"

    if not geometry_path.exists() or not bc_path.exists():
        run_preprocess(config)
    if (
        not final_fields_path.exists()
        or not solver_history_path.exists()
        or not mass_balance_path.exists()
        or not step2_manifest_path.exists()
    ):
        run_projection_solver(
            geometry_path, bc_path, output_root, config=STEP2_SOLVER_CONFIG
        )

    step3_root = output_root / "rendered_results" / "step3_fluent_style"
    result = run_fluent_style_postprocess(
        final_fields_path,
        solver_history_path,
        mass_balance_path,
        step2_manifest_path,
        step3_root,
    )
    print(
        json.dumps(
            {
                "case": "ansys_vertical_flap_fixed_flow",
                "step": "step3_fluent_style_postprocess",
                "output_root": "validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style",
                "report": "validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/validation_report.md",
                "quality": result["quality"],
                "claims": result["claims"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
