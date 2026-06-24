from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from simulation_core import cad_provenance_report

DEFAULT_SOURCE_CONFIG = str(
    Path("_diagnostic_runs")
    / "supportdiag_region4_008step_outflowguard_finalsamples_debugdl2_20260602"
    / "simulation_config.json"
)

def load_source_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"source config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))

def source_config_cad_provenance_report(
    config: Mapping[str, object],
    *,
    source_config_path: Path | None,
    cad_step_path: Path | None,
) -> dict[str, object]:
    return cad_provenance_report(
        cad_step_path,
        source_config=config,
        source_config_path=source_config_path,
    )

def _selection_ids_contain_region(selection_ids: object, region_id: int) -> bool:
    return int(region_id) in _selection_ids_as_int_tuple(selection_ids)

def _selection_ids_as_int_tuple(selection_ids: object) -> tuple[int, ...]:
    if isinstance(selection_ids, str):
        candidates: Sequence[object] = tuple(
            item
            for item in selection_ids.replace(",", " ").split()
            if item
        )
    elif isinstance(selection_ids, Mapping):
        candidates = tuple(selection_ids.values())
    elif isinstance(selection_ids, Sequence):
        candidates = selection_ids
    elif selection_ids is None:
        candidates = ()
    else:
        candidates = (selection_ids,)
    region_ids: list[int] = []
    for candidate in candidates:
        try:
            region_ids.append(int(candidate))
        except (TypeError, ValueError):
            continue
    return tuple(region_ids)

def source_config_requests_region14_aperture_carve(
    config: Mapping[str, object],
) -> bool:
    analysis = config.get("analysis_settings", {})
    if not isinstance(analysis, Mapping):
        return False
    return bool(analysis.get("solid_obstacle_opening_carve_enabled", False)) and (
        _selection_ids_contain_region(
            analysis.get("solid_obstacle_opening_carve_selection_ids", ()),
            14,
        )
    )

def source_config_requests_fluid_active_mask(
    config: Mapping[str, object],
) -> bool:
    analysis = config.get("analysis_settings", {})
    if not isinstance(analysis, Mapping):
        return False
    return bool(analysis.get("fluid_active_mask_enabled", False))

def source_config_requests_reduced_water_intersection(
    config: Mapping[str, object],
) -> bool:
    analysis = config.get("analysis_settings", {})
    if not isinstance(analysis, Mapping):
        return False
    return bool(
        analysis.get("fluid_active_mask_intersect_reduced_water_domain", False)
    )

def source_config_volume_particle_cache_path(source_config_path: Path) -> Path:
    pattern = f"{source_config_path.stem}.*.volume_particles.npz"
    candidates = sorted(source_config_path.parent.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            "source config requests a CAD-derived active mask, but no adjacent "
            f"volume particle cache matched {pattern!r}"
        )
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda path: path.stat().st_mtime)

def source_config_solid_obstacle_particle_region_ids(
    config: Mapping[str, object],
    available_region_ids: Sequence[int],
) -> tuple[int, ...]:
    analysis = config.get("analysis_settings", {})
    if not isinstance(analysis, Mapping):
        return tuple(sorted({int(value) for value in available_region_ids}))
    available = {int(value) for value in available_region_ids}
    surface_only_region_ids = set(
        _selection_ids_as_int_tuple(
            analysis.get("solid_obstacle_surface_only_region_ids", ()),
        )
    )
    if surface_only_region_ids:
        return tuple(sorted(available & surface_only_region_ids))
    selected = set(available)
    if bool(analysis.get("solid_obstacle_exclude_fsi_contact_regions", False)):
        selected -= set(
            _selection_ids_as_int_tuple(
                analysis.get("solid_obstacle_moving_fsi_contact_region_ids", ()),
            )
        )
    return tuple(sorted(selected))

def _mapping_config_float(
    mapping: Mapping[str, object],
    keys: Sequence[str],
    default: float,
    *,
    field: str,
) -> float:
    for key in keys:
        if key in mapping:
            value = float(mapping[key])
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            return value
    return float(default)

@dataclass(frozen=True)
class PressureBoundaryShellMapping:
    source_region_id: int
    target_shell_region_id: int
    primary_shell_region_id: int
    secondary_shell_region_id: int
    mapping_source: str
    source_selection_name: str
    target_selection_name: str
    boundary_condition_input_only: bool = True

def _face_ids_for_region(config: dict[str, object], region_id: int) -> list[int]:
    selections = config.get("named_selections", [])
    if not isinstance(selections, list):
        return []
    for selection in selections:
        if isinstance(selection, dict) and int(selection.get("id", -1)) == int(region_id):
            values = selection.get("face_ids", [])
            if isinstance(values, list):
                return [int(value) for value in values]
    return []

def _source_config_named_selection(
    config: Mapping[str, object],
    region_id: int,
) -> Mapping[str, object] | None:
    selections = config.get("named_selections", [])
    if not isinstance(selections, list):
        return None
    for selection in selections:
        if not isinstance(selection, Mapping):
            continue
        try:
            selection_id = int(selection.get("id", -1))
        except (TypeError, ValueError):
            continue
        if selection_id == int(region_id):
            return selection
    return None

def _source_config_selection_name(
    config: Mapping[str, object],
    region_id: int,
) -> str:
    selection = _source_config_named_selection(config, region_id)
    if selection is None:
        return ""
    return str(selection.get("name", ""))

def source_config_pressure_load_region_id(config: Mapping[str, object]) -> int:
    """Return the CAD selection carrying the prescribed pressure boundary."""
    selections = config.get("named_selections", [])
    if not isinstance(selections, list):
        return 7
    pressure_region_ids: list[int] = []
    for selection in selections:
        if not isinstance(selection, Mapping):
            continue
        boundary_condition = selection.get("boundary_condition", {})
        if not isinstance(boundary_condition, Mapping):
            continue
        if str(boundary_condition.get("type", "")).lower() != "pressure":
            continue
        region_id = int(selection.get("id", -1))
        if region_id >= 0:
            pressure_region_ids.append(region_id)
    if len(pressure_region_ids) == 1:
        return int(pressure_region_ids[0])
    if len(pressure_region_ids) > 1:
        raise ValueError(
            "source-config contains multiple pressure boundary selections; "
            "the squid case must name exactly one primary actuation pressure surface"
        )
    return 7

def source_config_shell_region_pair(config: Mapping[str, object]) -> tuple[int, int]:
    analysis = config.get("analysis_settings", {})
    if isinstance(analysis, Mapping):
        fsi_surface_ids = _selection_ids_as_int_tuple(
            analysis.get("solid_obstacle_moving_fsi_contact_surface_region_ids", ()),
        )
        if len(fsi_surface_ids) >= 2:
            return int(fsi_surface_ids[0]), int(fsi_surface_ids[1])
    return 7, 8

def _explicit_pressure_target_region_id(
    analysis: Mapping[str, object],
    source_region_id: int,
) -> int | None:
    map_keys = (
        "pressure_boundary_to_fsi_shell_region_ids",
        "pressure_load_region_map",
        "pressure_boundary_target_region_map",
    )
    for key in map_keys:
        raw_mapping = analysis.get(key)
        if not isinstance(raw_mapping, Mapping):
            continue
        for raw_source, raw_target in raw_mapping.items():
            try:
                mapped_source = int(raw_source)
                mapped_target = int(raw_target)
            except (TypeError, ValueError):
                continue
            if mapped_source == int(source_region_id):
                return mapped_target
    scalar_keys = (
        "pressure_load_target_region_id",
        "pressure_boundary_target_fsi_region_id",
        "actuation_pressure_fsi_region_id",
        "primary_pressure_fsi_shell_region_id",
    )
    for key in scalar_keys:
        if key not in analysis:
            continue
        try:
            return int(analysis[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer region id") from exc
    return None

def _selection_name_supports_pressure_side_mapping(
    *,
    source_selection_name: str,
    target_selection_name: str,
) -> bool:
    source = source_selection_name.lower()
    target = target_selection_name.lower()
    source_pressure_side = "pressure" in source and (
        "air" in source or "load" in source or "actuat" in source
    )
    target_fsi_side = "fsi" in target or "water" in target
    return source_pressure_side and target_fsi_side

def source_config_pressure_boundary_shell_mapping(
    config: Mapping[str, object],
) -> PressureBoundaryShellMapping:
    source_region_id = source_config_pressure_load_region_id(config)
    primary_shell_region_id, secondary_shell_region_id = source_config_shell_region_pair(
        config,
    )
    source_name = _source_config_selection_name(config, source_region_id)
    primary_name = _source_config_selection_name(config, primary_shell_region_id)
    analysis = config.get("analysis_settings", {})
    if not isinstance(analysis, Mapping):
        analysis = {}
    explicit_target = _explicit_pressure_target_region_id(
        analysis,
        source_region_id,
    )
    if explicit_target is not None:
        if explicit_target != primary_shell_region_id:
            raise ValueError(
                "pressure boundary target region must match the primary FSI "
                f"shell region {primary_shell_region_id}; got {explicit_target}"
            )
        mapping_source = "explicit_source_config_pressure_boundary_target"
        target_region_id = int(explicit_target)
    elif source_region_id == primary_shell_region_id:
        mapping_source = "pressure_boundary_selection_is_primary_fsi_shell"
        target_region_id = int(primary_shell_region_id)
    else:
        fsi_surface_ids = _selection_ids_as_int_tuple(
            analysis.get("solid_obstacle_moving_fsi_contact_surface_region_ids", ()),
        )
        if not fsi_surface_ids or int(primary_shell_region_id) != int(fsi_surface_ids[0]):
            raise ValueError(
                "pressure boundary selection differs from the primary FSI shell, "
                "but source_config does not declare ordered moving FSI contact "
                "surface regions"
            )
        if int(source_region_id) in {int(region_id) for region_id in fsi_surface_ids}:
            raise ValueError(
                "pressure boundary selection must not also be an FSI contact "
                "surface when it is mapped as a separate dry-side pressure face"
            )
        if not _selection_name_supports_pressure_side_mapping(
            source_selection_name=source_name,
            target_selection_name=primary_name,
        ):
            raise ValueError(
                "pressure boundary selection differs from the primary FSI shell, "
                "but named_selection names do not identify a pressure-side face "
                "mapped to a water/FSI shell face"
            )
        mapping_source = (
            "inferred_dry_pressure_side_to_primary_fsi_shell_from_source_config"
        )
        target_region_id = int(primary_shell_region_id)
    return PressureBoundaryShellMapping(
        source_region_id=int(source_region_id),
        target_shell_region_id=int(target_region_id),
        primary_shell_region_id=int(primary_shell_region_id),
        secondary_shell_region_id=int(secondary_shell_region_id),
        mapping_source=mapping_source,
        source_selection_name=source_name,
        target_selection_name=_source_config_selection_name(config, target_region_id),
    )

def _source_config_pressure_load_direction(
    config: Mapping[str, object],
) -> tuple[float, float, float]:
    selections = config.get("named_selections", [])
    if not isinstance(selections, list):
        return (0.0, 0.0, -1.0)
    for selection in selections:
        if not isinstance(selection, Mapping):
            continue
        boundary_condition = selection.get("boundary_condition", {})
        if not isinstance(boundary_condition, Mapping):
            continue
        if str(boundary_condition.get("type", "")).lower() != "pressure":
            continue
        params = boundary_condition.get("params", {})
        if not isinstance(params, Mapping):
            params = selection.get("params", {})
        direction = str(params.get("Direction", "-z")).strip().lower()
        if direction in {"-x", "negative_x", "x-"}:
            return (-1.0, 0.0, 0.0)
        if direction in {"+x", "x", "positive_x", "x+"}:
            return (1.0, 0.0, 0.0)
        if direction in {"-y", "negative_y", "y-"}:
            return (0.0, -1.0, 0.0)
        if direction in {"+y", "y", "positive_y", "y+"}:
            return (0.0, 1.0, 0.0)
        if direction in {"-z", "negative_z", "z-"}:
            return (0.0, 0.0, -1.0)
        if direction in {"+z", "z", "positive_z", "z+"}:
            return (0.0, 0.0, 1.0)
        raise ValueError(f"unsupported pressure load Direction: {direction!r}")
    return (0.0, 0.0, -1.0)

def _vector3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (vector[0], vector[1], vector[2])
