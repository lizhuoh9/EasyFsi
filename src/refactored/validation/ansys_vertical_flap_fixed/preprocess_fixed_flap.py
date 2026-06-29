from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .bc import build_bc_map, save_bc_map
from .geometry import build_geometry_mask, plot_geometry_preview, save_geometry_mask


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def run_preprocess(config: dict[str, Any]) -> dict[str, Any]:
    output_root = _resolve_output_root(str(config["output"]["root"]))
    preprocess_dir = output_root / "preprocess"
    fields_dir = output_root / "fields"
    rendered_dir = output_root / "rendered_results"

    geometry = build_geometry_mask(config)
    bc_map = build_bc_map(config, geometry)
    fields = _build_initial_fields(config, geometry)

    geometry_path = preprocess_dir / "geometry_mask.npz"
    bc_path = preprocess_dir / "bc_map.npz"
    fields_path = fields_dir / "initial_fields.npz"
    preview_path = rendered_dir / "geometry_preview.png"
    manifest_path = output_root / "case_manifest.json"

    save_geometry_mask(geometry_path, geometry)
    save_bc_map(bc_path, bc_map)
    _save_fields(fields_path, fields)
    plot_geometry_preview(preview_path, geometry)

    manifest = _build_manifest(
        config=config,
        output_root=output_root,
        geometry_path=geometry_path,
        bc_path=bc_path,
        fields_path=fields_path,
        preview_path=preview_path,
        manifest_path=manifest_path,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "output_root": str(output_root),
        "geometry_path": str(geometry_path),
        "bc_path": str(bc_path),
        "fields_path": str(fields_path),
        "preview_path": str(preview_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _build_initial_fields(
    config: dict[str, Any], geometry: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    fluid = geometry["fluid_mask"].astype(bool)
    inlet_Uz = float(config["boundary_conditions"]["inlet_Uz"])
    inlet_Uy = float(config["boundary_conditions"]["inlet_Uy"])
    Uz = np.zeros(fluid.shape, dtype=np.float64)
    Uy = np.zeros(fluid.shape, dtype=np.float64)
    p = np.zeros(fluid.shape, dtype=np.float64)
    Uz[fluid] = inlet_Uz
    Uy[fluid] = inlet_Uy
    return {
        "s": geometry["s"],
        "y": geometry["y"],
        "S": geometry["S"],
        "Y": geometry["Y"],
        "Z": geometry["Z"],
        "Uz": Uz,
        "Uy": Uy,
        "p": p,
        "streamwise_minus_Uz": -Uz,
    }


def _save_fields(path: Path, fields: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)


def _build_manifest(
    *,
    config: dict[str, Any],
    output_root: Path,
    geometry_path: Path,
    bc_path: Path,
    fields_path: Path,
    preview_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "case": "ansys_vertical_flap_fixed_flow",
        "source": "validation_cases/ansys_vertical_flap_fixed_flow/config.yaml",
        "output_root": _manifest_path(output_root),
        "scope": "fixed-flap flow preprocessing only; no solver step executed",
        "sign_convention": "left_to_right_display_flow_has_Uz_negative",
        "streamwise_display_velocity": "-Uz",
        "axis_order": {
            "array_shape": "(ny, ns)",
            "axis0": "y",
            "axis1": "display streamwise coordinate s",
            "physical_z": "-s",
        },
        "grid": {
            "ns": int(config["grid"]["ns"]),
            "ny": int(config["grid"]["ny"]),
        },
        "geometry": dict(config["geometry"]),
        "fluid": dict(config["fluid"]),
        "boundary_conditions": dict(config["boundary_conditions"]),
        "claims": {
            "fluent_parity": "not_claimed",
            "fsi": "not_claimed",
            "solver_result": "not_claimed",
        },
        "generated_files": {
            "geometry_mask": _manifest_path(geometry_path),
            "bc_map": _manifest_path(bc_path),
            "initial_fields": _manifest_path(fields_path),
            "geometry_preview": _manifest_path(preview_path),
            "case_manifest": _manifest_path(manifest_path),
        },
    }


def _resolve_output_root(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = line.strip().partition(":")
        if separator != ":":
            raise ValueError(f"Unsupported config line: {raw_line!r}")
        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        if any(token in value for token in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value
