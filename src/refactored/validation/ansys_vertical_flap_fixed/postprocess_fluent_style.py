from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .plot_fields import plot_geometry_overlay, plot_history, plot_scalar_field
from .profile_extractors import (
    extract_centerline_profile,
    extract_downstream_profiles,
    extract_throat_profile,
    summarize_profiles,
    write_profile_csv,
)
from .quality_gates import (
    evaluate_quality_gates,
    load_mass_balance,
    load_solver_history,
)
from .report_builder import build_validation_report


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def run_fluent_style_postprocess(
    final_fields_path: str | Path,
    solver_history_path: str | Path,
    mass_balance_path: str | Path,
    step2_manifest_path: str | Path,
    output_root: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_fields_path = Path(final_fields_path)
    solver_history_path = Path(solver_history_path)
    mass_balance_path = Path(mass_balance_path)
    step2_manifest_path = Path(step2_manifest_path)
    output_root = _resolve_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fields = dict(np.load(final_fields_path))
    history_rows = load_solver_history(solver_history_path)
    mass_rows = load_mass_balance(mass_balance_path)
    step2_manifest = json.loads(step2_manifest_path.read_text(encoding="utf-8"))
    final_summary = dict(step2_manifest.get("final_summary", {}))
    quality = evaluate_quality_gates(
        history_rows, mass_rows, final_summary, (config or {}).get("quality_gates")
    )

    figures = _write_figures(output_root, fields, history_rows, mass_rows)
    profiles = _write_profiles(output_root, fields)
    profile_summary = summarize_profiles(
        extract_centerline_profile(fields),
        extract_throat_profile(fields),
        extract_downstream_profiles(fields),
    )
    report_path = output_root / "validation_report.md"
    build_validation_report(
        report_path,
        field_summary=final_summary,
        quality=quality,
        figures=figures,
        profiles=profiles,
        profile_summary=profile_summary,
    )
    manifest_path = output_root / "case_manifest_step3.json"
    manifest = _build_manifest(
        output_root=output_root,
        final_fields_path=final_fields_path,
        solver_history_path=solver_history_path,
        mass_balance_path=mass_balance_path,
        step2_manifest_path=step2_manifest_path,
        figures=figures,
        profiles=profiles,
        report_path=report_path,
        manifest_path=manifest_path,
        quality=quality,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_root": str(output_root),
        "figures": figures,
        "profiles": profiles,
        "report": str(report_path),
        "manifest": str(manifest_path),
        "quality": quality,
        "claims": manifest["claims"],
    }


def _write_figures(
    output_root: Path,
    fields: dict[str, np.ndarray],
    history_rows: list[dict[str, float]],
    mass_rows: list[dict[str, float]],
) -> dict[str, str]:
    solid = fields["solid_mask"].astype(bool)
    max_speed = float(np.max(fields["speed"][~solid]))
    max_streamwise = float(np.max(fields["streamwise_minus_Uz"][~solid]))
    max_uy = float(np.max(np.abs(fields["Uy"][~solid])))
    figures = {
        "speed_full_fluent_scale_0_28p1": output_root
        / "speed_full_fluent_scale_0_28p1.png",
        "speed_full_autoscale": output_root / "speed_full_autoscale.png",
        "streamwise_minus_Uz_fluent_scale_0_28p1": output_root
        / "streamwise_minus_Uz_fluent_scale_0_28p1.png",
        "streamwise_minus_Uz_autoscale": output_root
        / "streamwise_minus_Uz_autoscale.png",
        "Uy_full": output_root / "Uy_full.png",
        "pressure_full": output_root / "pressure_full.png",
        "geometry_overlay": output_root / "geometry_overlay.png",
        "solver_history_plot": output_root / "solver_history_plot.png",
        "mass_balance_plot": output_root / "mass_balance_plot.png",
    }
    plot_scalar_field(
        figures["speed_full_fluent_scale_0_28p1"],
        fields["S"],
        fields["Y"],
        fields["speed"],
        solid,
        "Step 2 solver output; no Fluent parity claim",
        "|U| (m/s)",
        vmin=0.0,
        vmax=28.1,
    )
    plot_scalar_field(
        figures["speed_full_autoscale"],
        fields["S"],
        fields["Y"],
        fields["speed"],
        solid,
        "Step 2 solver output; no Fluent parity claim",
        "|U| (m/s)",
        vmin=0.0,
        vmax=max_speed,
    )
    plot_scalar_field(
        figures["streamwise_minus_Uz_fluent_scale_0_28p1"],
        fields["S"],
        fields["Y"],
        fields["streamwise_minus_Uz"],
        solid,
        "Step 2 solver output; no Fluent parity claim",
        "streamwise -Uz (m/s)",
        vmin=0.0,
        vmax=28.1,
    )
    plot_scalar_field(
        figures["streamwise_minus_Uz_autoscale"],
        fields["S"],
        fields["Y"],
        fields["streamwise_minus_Uz"],
        solid,
        "Step 2 solver output; no Fluent parity claim",
        "streamwise -Uz (m/s)",
        vmin=0.0,
        vmax=max_streamwise,
    )
    plot_scalar_field(
        figures["Uy_full"],
        fields["S"],
        fields["Y"],
        fields["Uy"],
        solid,
        "Step 2 solver output; no Fluent parity claim",
        "Uy (m/s)",
        vmin=-max_uy,
        vmax=max_uy,
        cmap="diverging",
    )
    plot_scalar_field(
        figures["pressure_full"],
        fields["S"],
        fields["Y"],
        fields["p"],
        solid,
        "Step 2 solver output; no Fluent parity claim",
        "p",
    )
    plot_geometry_overlay(
        figures["geometry_overlay"],
        fields["fluid_mask"],
        fields["solid_mask"],
        fields["near_solid_mask"],
    )
    plot_history(
        figures["solver_history_plot"],
        history_rows,
        ("max_speed", "divergence_l2", "mass_imbalance_rel"),
    )
    plot_history(
        figures["mass_balance_plot"],
        mass_rows,
        ("inlet_flux", "outlet_flux", "mass_imbalance_rel"),
    )
    return {key: _manifest_path(path) for key, path in figures.items()}


def _write_profiles(output_root: Path, fields: dict[str, np.ndarray]) -> dict[str, str]:
    centerline = extract_centerline_profile(fields)
    throat = extract_throat_profile(fields)
    downstream = extract_downstream_profiles(fields)
    paths = {
        "centerline_streamwise_minus_Uz": output_root
        / "centerline_streamwise_minus_Uz.csv",
        "throat_profile_streamwise_minus_Uz": output_root
        / "throat_profile_streamwise_minus_Uz.csv",
        "downstream_profiles_streamwise_minus_Uz": output_root
        / "downstream_profiles_streamwise_minus_Uz.csv",
    }
    write_profile_csv(paths["centerline_streamwise_minus_Uz"], centerline)
    write_profile_csv(paths["throat_profile_streamwise_minus_Uz"], throat)
    write_profile_csv(paths["downstream_profiles_streamwise_minus_Uz"], downstream)
    return {key: _manifest_path(path) for key, path in paths.items()}


def _build_manifest(
    *,
    output_root: Path,
    final_fields_path: Path,
    solver_history_path: Path,
    mass_balance_path: Path,
    step2_manifest_path: Path,
    figures: dict[str, str],
    profiles: dict[str, str],
    report_path: Path,
    manifest_path: Path,
    quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case": "ansys_vertical_flap_fixed_flow",
        "step": "step3_fluent_style_postprocess",
        "scope": "Fluent-style visualization and numerical quality report for Step 2 solver output; no Fluent parity claim",
        "output_root": _manifest_path(output_root),
        "sources": {
            "final_fields": _manifest_path(final_fields_path),
            "solver_history": _manifest_path(solver_history_path),
            "mass_balance": _manifest_path(mass_balance_path),
            "step2_manifest": _manifest_path(step2_manifest_path),
        },
        "forbidden_sources": {
            "traction_shared_snapshot_diagnostics": "not_used",
        },
        "generated_files": {
            **figures,
            **profiles,
            "validation_report": _manifest_path(report_path),
            "case_manifest_step3": _manifest_path(manifest_path),
        },
        "quality": quality,
        "claims": {
            "fluent_parity": "not_claimed",
            "fsi": "not_claimed",
            "solver_result": "step2_fixed_flap_projection_solver",
        },
    }


def _resolve_output_root(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _manifest_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)
