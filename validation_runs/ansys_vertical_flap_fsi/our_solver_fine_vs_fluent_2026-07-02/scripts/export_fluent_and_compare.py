from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "refactored").exists() and (
            parent / "cases" / "ansys_vertical_flap_fsi.py"
        ).exists():
            return parent
    raise RuntimeError("could not locate repo root from validation script path")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (  # noqa: E402
    compare_solver_to_fluent_field,
    load_fluent_npz,
    load_solver_npz,
    sample_structured_solver_at_fluent_points,
)
from src.refactored.validation.ansys_vertical_flap_fsi.official_fluent_reference import (  # noqa: E402
    _write_field_npz,
    read_fluent_cell_fields,
)


VELOCITY_DISPLAY_MAX_MPS = 28.1
FLAP_BOUNDS = {
    "x_min": 0.050,
    "x_max": 0.053,
    "y_min": 0.0,
    "y_max": 0.010,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _solver_display_y(solver: dict[str, np.ndarray], fluent: dict[str, np.ndarray]) -> np.ndarray:
    y = solver["y"]
    if float(np.min(y)) < 0.0 < float(np.max(y)) and float(np.min(fluent["y"])) >= -1.0e-12:
        return y + float(np.max(fluent["y"]))
    return y


def _masked_2d(field: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(field, mask=~mask.astype(bool))


def _plot_fluent_velocity(fluent: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    sc = ax.scatter(
        fluent["x"],
        fluent["y"],
        c=fluent["speed"],
        s=2.0,
        cmap="turbo",
        vmin=0.0,
        vmax=VELOCITY_DISPLAY_MAX_MPS,
        linewidths=0,
    )
    _draw_flap(ax)
    ax.set_title("Official Fluent fsi_2way velocity magnitude, t = 0.025 s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0.0, 0.10)
    ax.set_ylim(0.0, 0.022)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Velocity magnitude (m/s)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_solver_velocity(
    solver: dict[str, np.ndarray], fluent: dict[str, np.ndarray], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    y_display = _solver_display_y(solver, fluent)
    image = ax.imshow(
        _masked_2d(solver["speed"], solver["fluid_mask"]),
        origin="lower",
        extent=[
            float(np.min(solver["s"])),
            float(np.max(solver["s"])),
            float(np.min(y_display)),
            float(np.max(y_display)),
        ],
        aspect="auto",
        cmap="turbo",
        vmin=0.0,
        vmax=VELOCITY_DISPLAY_MAX_MPS,
        interpolation="nearest",
    )
    _draw_flap(ax)
    ax.set_title("Our solver velocity magnitude, t = 0.025 s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0.0, 0.10)
    ax.set_ylim(0.0, 0.022)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Velocity magnitude (m/s)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_velocity_difference(
    fluent: dict[str, np.ndarray], samples: dict[str, np.ndarray], path: Path
) -> dict[str, float]:
    valid = samples["valid"].astype(bool)
    abs_error = np.abs(samples["speed"] - fluent["speed"])
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    sc = ax.scatter(
        fluent["x"][valid],
        fluent["y"][valid],
        c=abs_error[valid],
        s=2.0,
        cmap="magma",
        vmin=0.0,
        vmax=float(np.nanpercentile(abs_error[valid], 99.0)) if np.any(valid) else 1.0,
        linewidths=0,
    )
    _draw_flap(ax)
    ax.set_title("Absolute velocity-magnitude difference sampled at Fluent cells")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0.0, 0.10)
    ax.set_ylim(0.0, 0.022)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("|Delta speed| (m/s)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {
        "velocity_abs_error_l1_mps": float(np.nanmean(abs_error[valid])),
        "velocity_abs_error_l2_mps": float(
            np.sqrt(np.nanmean(abs_error[valid] ** 2))
        ),
        "velocity_abs_error_linf_mps": float(np.nanmax(abs_error[valid])),
    }


def _plot_pressure_comparison(
    fluent: dict[str, np.ndarray],
    solver: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
    path: Path,
) -> dict[str, float]:
    valid = samples["valid"].astype(bool)
    y_display = _solver_display_y(solver, fluent)
    pressure_error = samples["p"] - fluent["p"]
    finite_solver = solver["fluid_mask"].astype(bool) & np.isfinite(solver["p"])
    p_min = float(
        min(np.nanpercentile(fluent["p"], 1.0), np.nanpercentile(solver["p"][finite_solver], 1.0))
    )
    p_max = float(
        max(np.nanpercentile(fluent["p"], 99.0), np.nanpercentile(solver["p"][finite_solver], 99.0))
    )
    err_lim = float(np.nanpercentile(np.abs(pressure_error[valid]), 99.0)) if np.any(valid) else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    fluent_plot = axes[0].scatter(
        fluent["x"],
        fluent["y"],
        c=fluent["p"],
        s=2.0,
        cmap="coolwarm",
        vmin=p_min,
        vmax=p_max,
        linewidths=0,
    )
    axes[0].set_title("Fluent pressure")
    solver_plot = axes[1].imshow(
        _masked_2d(solver["p"], solver["fluid_mask"]),
        origin="lower",
        extent=[
            float(np.min(solver["s"])),
            float(np.max(solver["s"])),
            float(np.min(y_display)),
            float(np.max(y_display)),
        ],
        aspect="auto",
        cmap="coolwarm",
        vmin=p_min,
        vmax=p_max,
        interpolation="nearest",
    )
    axes[1].set_title("Our solver pressure")
    diff_plot = axes[2].scatter(
        fluent["x"][valid],
        fluent["y"][valid],
        c=pressure_error[valid],
        s=2.0,
        cmap="coolwarm",
        vmin=-err_lim,
        vmax=err_lim,
        linewidths=0,
    )
    axes[2].set_title("Solver - Fluent pressure")
    for ax in axes:
        _draw_flap(ax)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(0.0, 0.10)
        ax.set_ylim(0.0, 0.022)
    fig.colorbar(fluent_plot, ax=axes[:2], label="Pressure (Pa)")
    fig.colorbar(diff_plot, ax=axes[2], label="Delta pressure (Pa)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {
        "pressure_signed_error_mean_pa": float(np.nanmean(pressure_error[valid])),
        "pressure_abs_error_l1_pa": float(np.nanmean(np.abs(pressure_error[valid]))),
        "pressure_abs_error_l2_pa": float(
            np.sqrt(np.nanmean(pressure_error[valid] ** 2))
        ),
        "pressure_abs_error_linf_pa": float(np.nanmax(np.abs(pressure_error[valid]))),
    }


def _plot_flap_overlay(
    solver: dict[str, np.ndarray], fluent: dict[str, np.ndarray], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    y_display = _solver_display_y(solver, fluent)
    ax.imshow(
        solver["solid_mask"].astype(float),
        origin="lower",
        extent=[
            float(np.min(solver["s"])),
            float(np.max(solver["s"])),
            float(np.min(y_display)),
            float(np.max(y_display)),
        ],
        aspect="auto",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        alpha=0.7,
    )
    _draw_flap(ax, edgecolor="tab:red", linewidth=2.0)
    ax.set_title("Flap/interface overlay")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0.044, 0.060)
    ax.set_ylim(0.0, 0.014)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _draw_flap(ax: plt.Axes, edgecolor: str = "black", linewidth: float = 1.1) -> None:
    rect = plt.Rectangle(
        (FLAP_BOUNDS["x_min"], FLAP_BOUNDS["y_min"]),
        FLAP_BOUNDS["x_max"] - FLAP_BOUNDS["x_min"],
        FLAP_BOUNDS["y_max"] - FLAP_BOUNDS["y_min"],
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=10,
    )
    ax.add_patch(rect)


def _field_stats(prefix: str, fields: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, float]:
    output: dict[str, float] = {}
    if mask is None:
        mask = np.ones_like(fields["speed"], dtype=bool)
    for key in ("speed", "p"):
        values = np.asarray(fields[key], dtype=np.float64)
        active = mask.astype(bool) & np.isfinite(values)
        output[f"{prefix}_{key}_min"] = float(np.min(values[active]))
        output[f"{prefix}_{key}_max"] = float(np.max(values[active]))
        output[f"{prefix}_{key}_mean"] = float(np.mean(values[active]))
    return output


def _write_profile_csvs(
    profiles_dir: Path,
    fluent: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
) -> list[str]:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    valid = samples["valid"].astype(bool)
    targets = {
        "throat_x_0p050": ("x", 0.050),
        "downstream_x_0p054": ("x", 0.054),
        "downstream_x_0p060": ("x", 0.060),
        "downstream_x_0p070": ("x", 0.070),
        "downstream_x_0p090": ("x", 0.090),
        "top_centerline": ("y", float(np.max(fluent["y"]))),
    }
    for name, (axis, target) in targets.items():
        values = fluent[axis]
        tolerance = max(float(np.max(values) - np.min(values)) / 200.0, 1.0e-4)
        mask = valid & (np.abs(values - target) <= tolerance)
        path = profiles_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "x_m",
                    "y_m",
                    "fluent_u_mps",
                    "solver_u_mps",
                    "fluent_speed_mps",
                    "solver_speed_mps",
                    "fluent_pressure_pa",
                    "solver_pressure_pa",
                ],
            )
            writer.writeheader()
            order = np.argsort(fluent["y"][mask] if axis == "x" else fluent["x"][mask])
            indices = np.flatnonzero(mask)[order]
            for idx in indices:
                writer.writerow(
                    {
                        "x_m": float(fluent["x"][idx]),
                        "y_m": float(fluent["y"][idx]),
                        "fluent_u_mps": float(fluent["u"][idx]),
                        "solver_u_mps": float(samples["u"][idx]),
                        "fluent_speed_mps": float(fluent["speed"][idx]),
                        "solver_speed_mps": float(samples["speed"][idx]),
                        "fluent_pressure_pa": float(fluent["p"][idx]),
                        "solver_pressure_pa": float(samples["p"][idx]),
                    }
                )
        outputs.append(str(path))
    return outputs


def _write_report(path: Path, metrics: dict[str, Any], figures: dict[str, str], profile_paths: list[str]) -> None:
    lines = [
        "# Our Solver vs Official Fluent fsi_2way",
        "",
        "This report compares the current in-repo solver output against the locked Fluent transient FSI reference at t = 0.025 s.",
        "",
        "## Key Metrics",
        "",
    ]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (float, int)):
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Figures", ""])
    for key, value in figures.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Profiles", ""])
    for value in profile_paths:
        lines.append(f"- `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fluent-case", required=True)
    parser.add_argument("--fluent-data", required=True)
    parser.add_argument("--solver-npz", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fluent-npz")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    figures_dir = output_root / "figures"
    profiles_dir = output_root / "profiles"
    comparison_dir = output_root / "comparison"
    fluent_npz = (
        Path(args.fluent_npz).resolve()
        if args.fluent_npz
        else output_root / "fluent_reference" / "fluent_fsi50_final_fields.npz"
    )

    if not fluent_npz.exists():
        bundle = read_fluent_cell_fields(args.fluent_case, args.fluent_data)
        _write_field_npz(fluent_npz, bundle)

    fluent = load_fluent_npz(fluent_npz)
    solver = load_solver_npz(args.solver_npz)
    samples = sample_structured_solver_at_fluent_points(solver, fluent)
    parity = compare_solver_to_fluent_field(solver, fluent, prefix="our_vs_fluent")

    figures = {
        "fluent_velocity": str(figures_dir / "fluent_velocity_t0p025.png"),
        "our_solver_velocity": str(figures_dir / "our_solver_velocity_t0p025.png"),
        "velocity_abs_difference": str(figures_dir / "velocity_abs_difference_t0p025.png"),
        "pressure_comparison": str(figures_dir / "pressure_comparison_t0p025.png"),
        "flap_overlay": str(figures_dir / "flap_overlay_t0p025.png"),
    }
    _plot_fluent_velocity(fluent, Path(figures["fluent_velocity"]))
    _plot_solver_velocity(solver, fluent, Path(figures["our_solver_velocity"]))
    velocity_metrics = _plot_velocity_difference(
        fluent, samples, Path(figures["velocity_abs_difference"])
    )
    pressure_metrics = _plot_pressure_comparison(
        fluent, solver, samples, Path(figures["pressure_comparison"])
    )
    _plot_flap_overlay(solver, fluent, Path(figures["flap_overlay"]))
    profile_paths = _write_profile_csvs(profiles_dir, fluent, samples)

    valid = samples["valid"].astype(bool)
    metrics: dict[str, Any] = {
        "status": parity["status"],
        "sample_count": int(np.count_nonzero(valid)),
        "fluent_npz": str(fluent_npz),
        "solver_npz": str(Path(args.solver_npz).resolve()),
        **_field_stats("fluent", fluent),
        **_field_stats("solver", solver, solver["fluid_mask"]),
        **velocity_metrics,
        **pressure_metrics,
        "parity": parity,
    }
    _write_json(comparison_dir / "comparison_metrics.json", metrics)
    _write_json(comparison_dir / "aligned_sampling_summary.json", {
        "valid_sample_count": int(np.count_nonzero(valid)),
        "invalid_sample_count": int(np.count_nonzero(~valid)),
        "sample_count": int(valid.size),
        "velocity_abs_error_l1_mps": velocity_metrics["velocity_abs_error_l1_mps"],
        "pressure_abs_error_l1_pa": pressure_metrics["pressure_abs_error_l1_pa"],
    })
    _write_report(output_root / "comparison_report.md", metrics, figures, profile_paths)
    print(json.dumps(_json_safe(metrics), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
