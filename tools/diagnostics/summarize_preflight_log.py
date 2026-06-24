from __future__ import annotations

import json
import sys
from pathlib import Path


def _section_summary(section: dict) -> tuple[str | None, float | None, float | None]:
    return (
        section.get("id"),
        section.get("volumetric_flow_rate_m3ps"),
        section.get("mean_normal_velocity_mps"),
    )


def _projected_residual_mps(sim: dict) -> float | None:
    return sim.get("projected_ibm_residual_mps")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m tools.diagnostics.summarize_preflight_log <stderr-log>",
            file=sys.stderr,
        )
        return 2
    path = Path(argv[1])
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[GUI_PROGRESS]"):
            continue
        payload = json.loads(line.split(" ", 1)[1])
        if isinstance(payload.get("simulation"), dict):
            rows.append(payload)
    print(f"sim_rows {len(rows)}")
    for row in rows[-5:]:
        sim = row["simulation"]
        sections = sim.get("custom_section_monitor_latest") or []
        projected_residual = _projected_residual_mps(sim)
        compact = {
            "step": row.get("step"),
            "percent": row.get("percent"),
            "elapsed_wall_time_s": row.get("elapsed_wall_time_s"),
            "time_s": sim.get("time_s"),
            "residual_mps": projected_residual,
            "converged": sim.get("fsi_projected_ibm_converged"),
            "solver_failed": sim.get("fsi_projected_ibm_solver_failed"),
            "pressure_probe_max_abs_Pa": sim.get(
                "fsi_fluid_pressure_probe_max_abs_pressure_Pa"
            ),
            "pressure_traction_max_N": sim.get("fsi_fluid_pressure_traction_max_N"),
            "grid_clamp_count": sim.get("solid_debug_grid_clamp_count"),
            "deformation_reset_counts": [
                sim.get("solid_debug_input_deformation_reset_count"),
                sim.get("solid_debug_midpoint_deformation_reset_count"),
                sim.get("solid_debug_updated_deformation_reset_count"),
            ],
            "mobility_update_count": sim.get(
                "fsi_projected_ibm_solid_mobility_update_count"
            ),
            "section_flows": [_section_summary(section) for section in sections],
        }
        print(json.dumps(compact, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
