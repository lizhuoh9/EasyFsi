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
    run_stabilized_projection_solver,
)


OFFICIAL_SOLVER_CONFIG = {
    "solver": {
        "max_steps": 48,
        "cfl": 0.20,
        "steady_tolerance": 0.0,
        "poisson_method": "sor",
        "poisson_max_iters": 180,
        "poisson_tolerance_abs": 1.0e-4,
        "poisson_tolerance_rel": 1.0e-3,
        "poisson_omega": 1.65,
        "poisson_check_interval": 20,
        "poisson_compatibility_correction": True,
        "initialization_mode": "uniform",
        "area_flux_projection": False,
        "outlet_flux_correction": True,
        "history_interval": 10,
        "write_checkpoints": False,
    },
    "sensitivity": {
        "max_steps": 12,
        "poisson_max_iters": 80,
    },
}


def main() -> int:
    config_path = Path(__file__).with_name("config_official_fluent.yaml")
    config = load_config(config_path)
    solver_config = {
        **OFFICIAL_SOLVER_CONFIG,
        "fluid": dict(config["fluid"]),
    }
    case_root = PROJECT_ROOT / config["output"]["root"]
    preprocess = run_preprocess(config)
    solver_root = case_root / "stabilized_solver"
    postprocess_root = case_root / "rendered_results" / "official_fluent_preflow"
    result = run_stabilized_projection_solver(
        Path(preprocess["geometry_path"]),
        Path(preprocess["bc_path"]),
        solver_root,
        baseline_root=case_root,
        postprocess_root=postprocess_root,
        config=solver_config,
    )
    print(
        json.dumps(
            {
                "case": "ansys_vertical_flap_fixed_flow_official_fluent_preflow",
                "fields": (
                    "validation_runs/ansys_vertical_flap_fixed_flow/"
                    "official_fluent_preflow/stabilized_solver/fields/"
                    "final_fields_stabilized.npz"
                ),
                "overall_status": result["quality"]["overall_status"],
                "claims": result["claims"],
                "final_summary": result["final_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
