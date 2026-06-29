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


STABILIZED_SOLVER_CONFIG = {
    "solver": {
        "max_steps": 120,
        "cfl": 0.20,
        "steady_tolerance": 0.0,
        "poisson_method": "sor",
        "poisson_max_iters": 260,
        "poisson_tolerance_abs": 1.0e-4,
        "poisson_tolerance_rel": 1.0e-3,
        "poisson_omega": 1.65,
        "poisson_check_interval": 20,
        "poisson_compatibility_correction": True,
        "initialization_mode": "uniform",
        "outlet_flux_correction": True,
        "history_interval": 10,
        "write_checkpoints": False,
    },
    "sensitivity": {
        "max_steps": 32,
        "poisson_max_iters": 160,
    },
}


def main() -> int:
    config_path = Path(__file__).with_name("config.yaml")
    config = load_config(config_path)
    case_root = PROJECT_ROOT / config["output"]["root"]
    geometry_path = case_root / "preprocess" / "geometry_mask.npz"
    bc_path = case_root / "preprocess" / "bc_map.npz"
    if not geometry_path.exists() or not bc_path.exists():
        run_preprocess(config)

    solver_root = case_root / "stabilized_solver"
    postprocess_root = case_root / "rendered_results" / "step4_stabilized_fluent_style"
    result = run_stabilized_projection_solver(
        geometry_path,
        bc_path,
        solver_root,
        baseline_root=case_root,
        postprocess_root=postprocess_root,
        config=STABILIZED_SOLVER_CONFIG,
    )
    quality = result["quality"]
    print(
        json.dumps(
            {
                "case": "ansys_vertical_flap_fixed_flow",
                "step": "step4_solver_stabilization",
                "initialization_mode": "uniform",
                "stabilized_fields": "validation_runs/ansys_vertical_flap_fixed_flow/stabilized_solver/fields/final_fields_stabilized.npz",
                "stabilized_history": "validation_runs/ansys_vertical_flap_fixed_flow/stabilized_solver/logs/solver_history_stabilized.csv",
                "stabilized_report": "validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/validation_report.md",
                "quality": {
                    "mass_quality": quality["mass_quality"]["status"],
                    "incompressibility_quality": quality["incompressibility_quality"][
                        "status"
                    ],
                    "overall_status": quality["overall_status"],
                },
                "claims": result["claims"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
