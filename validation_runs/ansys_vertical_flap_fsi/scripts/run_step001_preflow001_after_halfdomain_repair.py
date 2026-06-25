from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases.ansys_vertical_flap_fsi import (
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
REPORT = ROOT / "easyfsi" / "easyfsi_step001_preflow001_after_halfdomain_repair.json"
PROCESS = ROOT / "easyfsi" / "easyfsi_step001_preflow001_after_halfdomain_repair_process.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    started = time.perf_counter()
    _write_json(
        PROCESS,
        {
            "case": "ansys-vertical-flap-fsi",
            "status": "running",
            "step_count": 1,
            "preflow_steps": 1,
            "report_json": str(REPORT),
            "started_at_epoch_s": time.time(),
        },
    )
    try:
        report = run_vertical_flap_fsi_smoke(
            VerticalFlapFsiConfig(step_count=1, preflow_steps=1)
        )
    except Exception as exc:  # pragma: no cover - process evidence path.
        _write_json(
            PROCESS,
            {
                "case": "ansys-vertical-flap-fsi",
                "status": "failed",
                "step_count": 1,
                "preflow_steps": 1,
                "report_json": str(REPORT),
                "elapsed_s": time.perf_counter() - started,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1

    _write_json(REPORT, report)
    _write_json(
        PROCESS,
        {
            "case": "ansys-vertical-flap-fsi",
            "status": "completed",
            "step_count": 1,
            "preflow_steps": 1,
            "report_json": str(REPORT),
            "elapsed_s": time.perf_counter() - started,
            "history_rows": len(report.get("history", [])),
            "preflow_steps_completed": report.get("preflow_steps_completed"),
            "preflow_converged": report.get("preflow_converged"),
            "local_velocity_peak_mps": report.get("local_velocity_peak_mps"),
            "fluid_speed_p999_mps": report.get("fluid_speed_p999_mps"),
            "solid_substeps_selected": report.get("solid_substeps_selected"),
            "solid_estimated_cfl": report.get("solid_estimated_cfl"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
