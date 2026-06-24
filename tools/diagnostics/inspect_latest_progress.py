from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GUI_PREFIX = b"[GUI_PROGRESS] "


def _find_last_gui_progress(path: Path, *, chunk_size: int = 1 << 20) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        pos = handle.tell()
        data = b""
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            handle.seek(pos)
            data = handle.read(read_size) + data
            lines = [line for line in data.splitlines() if line.startswith(GUI_PREFIX)]
            if lines:
                return json.loads(lines[-1][len(GUI_PREFIX) :].decode("utf-8", errors="replace"))
    raise RuntimeError(f"No [GUI_PROGRESS] line found in {path}")


def _finite_or_none(value: Any, field_name: str, warnings: list[str], bad_fields: list[str]) -> Any:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(numeric):
        bad_fields.append(field_name)
        return None
    return value


def _compact_payload(progress: dict[str, Any]) -> dict[str, Any]:
    sim = progress.get("simulation")
    if not isinstance(sim, dict):
        final = progress.get("final")
        sim = final if isinstance(final, dict) else progress
    physical = sim.get("physical_monitor_summary") or {}
    jet = physical.get("jet_sections", {}).get("sections", {})
    membrane = physical.get("membrane_displacement_mm") or {}
    warnings: list[str] = []
    if not physical:
        warnings.append("physical_monitor_summary not present; using flat runner fields when available")
    if sim is progress.get("final"):
        warnings.append("summary final record detected; using final runner fields")

    flat_sections = {
        "lip": {
            "mean_normal_velocity_mps": None,
            "volumetric_flow_rate_m3ps": sim.get("lip_flow_negative_z_m3s"),
            "mean_pressure_Pa": None,
            "sample_count": sim.get("lip_sample_count"),
        },
        "outlet": {
            "mean_normal_velocity_mps": None,
            "volumetric_flow_rate_m3ps": sim.get("outlet_flow_negative_z_m3s"),
            "mean_pressure_Pa": None,
            "sample_count": sim.get("outlet_sample_count"),
        },
        "downstream": {
            "mean_normal_velocity_mps": None,
            "volumetric_flow_rate_m3ps": sim.get("downstream_flow_negative_z_m3s"),
            "mean_pressure_Pa": None,
            "sample_count": sim.get("downstream_sample_count"),
        },
    }
    if not jet:
        jet = flat_sections
        missing_section_keys = [
            key
            for key in (
                "lip_flow_negative_z_m3s",
                "outlet_flow_negative_z_m3s",
                "downstream_flow_negative_z_m3s",
            )
            if key not in sim
        ]
        if missing_section_keys:
            warnings.append("missing flat section keys: " + ", ".join(missing_section_keys))
    core_key_groups = {
        "time_s": ("time_s",),
        "residual_mps": ("projected_ibm_residual_mps",),
        "max_fluid_speed_mps": ("max_fluid_speed_mps", "max_speed_mps"),
        "divergence_l2": ("divergence_l2",),
    }
    missing_core_keys = [
        name
        for name, keys in core_key_groups.items()
        if not any(key in sim for key in keys)
    ]
    if missing_core_keys:
        warnings.append("missing core progress keys: " + ", ".join(missing_core_keys))
    residual_mps = sim.get("projected_ibm_residual_mps")
    max_fluid_speed_mps = sim.get("max_fluid_speed_mps", sim.get("max_speed_mps"))
    bad_fields: list[str] = []
    compact_jet_sections = {}
    for name, section in jet.items():
        compact_jet_sections[name] = {
            "mean_normal_velocity_mps": _finite_or_none(
                section.get("mean_normal_velocity_mps"),
                f"jet_sections.{name}.mean_normal_velocity_mps",
                warnings,
                bad_fields,
            ),
            "flow_rate_m3ps": _finite_or_none(
                section.get("volumetric_flow_rate_m3ps"),
                f"jet_sections.{name}.flow_rate_m3ps",
                warnings,
                bad_fields,
            ),
            "mean_pressure_Pa": _finite_or_none(
                section.get("mean_pressure_Pa"),
                f"jet_sections.{name}.mean_pressure_Pa",
                warnings,
                bad_fields,
            ),
            "sample_count": section.get("sample_count"),
        }
    time_s = _finite_or_none(sim.get("time_s"), "time_s", warnings, bad_fields)
    residual_mps = _finite_or_none(residual_mps, "residual_mps", warnings, bad_fields)
    max_fluid_speed_mps = _finite_or_none(
        max_fluid_speed_mps,
        "max_fluid_speed_mps",
        warnings,
        bad_fields,
    )
    force_residual_relative = _finite_or_none(
        sim.get("fsi_force_residual_relative"),
        "force_residual_relative",
        warnings,
        bad_fields,
    )
    main_displacement = _finite_or_none(
        membrane.get("main_membrane", sim.get("main_displacement_z_m")),
        "membrane.main",
        warnings,
        bad_fields,
    )
    tail_displacement = _finite_or_none(
        membrane.get("tail_membrane", sim.get("tail_displacement_z_m")),
        "membrane.tail",
        warnings,
        bad_fields,
    )
    if bad_fields:
        warnings.append("non-finite progress fields: " + ", ".join(sorted(set(bad_fields))))
    return {
        "_warnings": warnings,
        "step": progress.get("step", sim.get("step")),
        "total_steps": progress.get("total_steps"),
        "percent": progress.get("percent"),
        "time_s": time_s,
        "residual_mps": residual_mps,
        "force_residual_relative": force_residual_relative,
        "converged": sim.get("fsi_converged"),
        "solver_failed": sim.get("solver_failed"),
        "max_fluid_speed_mps": max_fluid_speed_mps,
        "grid_clamp_count": sim.get("solid_debug_grid_clamp_count"),
        "deformation_resets": {
            "input": sim.get("solid_debug_input_deformation_reset_count"),
            "midpoint": sim.get("solid_debug_midpoint_deformation_reset_count"),
            "updated": sim.get("solid_debug_updated_deformation_reset_count"),
        },
        "projected_ibm": {
            "active_marker_count": sim.get("fsi_projected_ibm_active_marker_count"),
            "contact_marker_count": sim.get("fsi_projected_ibm_contact_marker_count"),
            "active_marker_fraction": sim.get("fsi_projected_ibm_active_marker_fraction"),
            "residual_mps": sim.get("projected_ibm_residual_mps"),
            "residual_l2_mps": sim.get("projected_ibm_residual_l2_mps"),
            "sample_count": sim.get("projected_ibm_sample_count"),
            "probe_valid_fraction": sim.get("fsi_probe_valid_fraction"),
            "active_marker_policy": sim.get("fsi_projected_ibm_active_marker_policy"),
            "exclude_region_ids": sim.get("fsi_projected_ibm_exclude_region_ids"),
            "probe_plus_contact_regions": sim.get(
                "fsi_projected_ibm_velocity_probe_plus_contact_region_counts_text"
            ),
            "probe_minus_contact_regions": sim.get(
                "fsi_projected_ibm_velocity_probe_minus_contact_region_counts_text"
            ),
            "probe_invalid_contact_regions": sim.get(
                "fsi_projected_ibm_velocity_probe_invalid_contact_region_counts_text"
            ),
        },
        "membrane": {
            "main": main_displacement,
            "tail": tail_displacement,
        },
        "jet_sections": compact_jet_sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the latest compact GUI_PROGRESS record.")
    parser.add_argument("stderr_log", type=Path)
    args = parser.parse_args()
    progress = _find_last_gui_progress(args.stderr_log)
    print(json.dumps(_compact_payload(progress), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
