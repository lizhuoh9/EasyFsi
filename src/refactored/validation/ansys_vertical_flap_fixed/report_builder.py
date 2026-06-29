from __future__ import annotations

from pathlib import Path
from typing import Any


def build_validation_report(
    path: str | Path,
    *,
    field_summary: dict[str, Any],
    quality: dict[str, Any],
    figures: dict[str, str],
    profiles: dict[str, str],
    profile_summary: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = quality["metrics"]
    lines = [
        "# ANSYS Vertical Flap Fixed-Flow Step 3 Validation Report",
        "",
        "## Scope",
        "",
        "- Source: Step 2 solver output.",
        "- No Fluent parity claim.",
        "- No FSI claim.",
        "- traction_shared_snapshot_diagnostics not used.",
        "- This is a Fluent-style visualization of the Step 2 solver output, not a Fluent parity validation.",
        "",
        "## Field Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| max_u | {field_summary.get('max_u', metrics['max_u']):.9g} |",
        f"| max_speed | {field_summary.get('max_speed', metrics['max_speed']):.9g} |",
        f"| centerline_max_u | {profile_summary['centerline_max_u']:.9g} |",
        f"| mass_imbalance_rel | {metrics['mass_imbalance_rel']:.9g} |",
        f"| divergence_linf | {metrics['divergence_linf']:.9g} |",
        f"| divergence_l2 | {metrics['divergence_l2']:.9g} |",
        f"| poisson_residual_linf | {metrics['poisson_residual_linf']:.9g} |",
        f"| throat_max_u | {profile_summary['throat_max_u']:.9g} |",
        f"| throat_mean_u | {profile_summary['throat_mean_u']:.9g} |",
        "",
        "## Visual Outputs",
        "",
    ]
    for label, value in figures.items():
        lines.append(f"- {label}: `{value}`")
    lines.extend(["", "## Profile Outputs", ""])
    for label, value in profiles.items():
        lines.append(f"- {label}: `{value}`")
    lines.extend(
        [
            "",
            "## Quality Gates",
            "",
            "| gate | status | reason |",
            "|---|---|---|",
            _gate_row("visual_candidate", quality["visual_candidate"]),
            _gate_row("mass_quality", quality["mass_quality"]),
            _gate_row("incompressibility_quality", quality["incompressibility_quality"]),
            f"| overall_status | {quality['overall_status']} | diagnostic_only_not_parity until parity data and solver convergence justify stronger claims |",
            "",
            "## Interpretation",
            "",
            "Current Step 2 produces a jet-like fixed-flap field, but the report keeps visual similarity separate from numerical convergence and official Fluent parity.",
            "diagnostic_only_not_parity is the controlling status whenever divergence or pressure Poisson convergence remains outside the warning thresholds.",
            "",
            "## Required Next Solver Improvement",
            "",
            "- Improve pressure Poisson convergence.",
            "- Add a divergence-reduction regression test.",
            "- Compare uniform-initialized runs against the current jet-structured initialization.",
            "- Introduce official Fluent numeric exports before any Fluent parity claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _gate_row(name: str, gate: dict[str, str]) -> str:
    return f"| {name} | {gate['status']} | {gate['reason']} |"
