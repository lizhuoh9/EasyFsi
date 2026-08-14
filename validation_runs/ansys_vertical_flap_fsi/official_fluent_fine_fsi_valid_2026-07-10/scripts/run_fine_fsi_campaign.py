"""Run a guarded fine-mesh Fluent intrinsic-FSI campaign.

The historical fine-mesh run adapted both ``fluid.4`` and ``solid.5``.  Fluent's
2-D structural solver does not accept the resulting type-7 (polygonal) solid
cells, but a serial Fluent call can return after the compute process is
interrupted.  This launcher therefore treats mesh topology, the transcript,
and a non-zero structural displacement as hard gates.

Every invocation writes to a new run directory.  It never edits the historical
fine-mesh scripts or artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import time
import traceback
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = SCRIPT_DIR.parent
# Private Fluent inputs are deliberately outside version control.  Keep the
# fallback portable; production runs can override either file via the
# FLUENT_FINE_SOURCE_CASE and FLUENT_FINE_SOURCE_DATA environment variables.
DEFAULT_SOURCE_DIR = CAMPAIGN_DIR / "local_inputs" / "fsi_2way"
SOLID_ZONE = "solid.5"
FLUID_ZONE = "fluid.4"
FLUID_ZONE_ID = 2
FLAP_FLUID_FORCE_ZONE = "flap_wall-shadow"
INLET_ZONE = "velocity_inlet.1"
OUTLET_ZONE = "po.3"
DEFORMING_DYNAMIC_ZONES = ("po.3", "symmetry.2", "velocity_inlet.1", "wall")
EXPECTED_SOLID_CELL_COUNT = 30
SUPPORTED_2D_SOLID_TYPES = frozenset({1, 3})
POLYHEDRAL_CELL_TYPE = 7
TARGET_X_M = 0.0505
TARGET_Y_M = 0.0095
STEP_NONZERO_DISPLACEMENT_TOLERANCE_M = 1.0e-18

FATAL_TRANSCRIPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "element type not implemented",
        re.compile(r"element\s+type\s+not\s+implemented", re.IGNORECASE),
    ),
    (
        "compute processes interrupted",
        re.compile(r"compute\s+process(?:es)?\s+interrupted", re.IGNORECASE),
    ),
    ("error at node", re.compile(r"error\s+at\s+node", re.IGNORECASE)),
    ("operation stopped", re.compile(r"operation\s+stopped", re.IGNORECASE)),
    ("fatal error", re.compile(r"fatal\s+error", re.IGNORECASE)),
    ("stream removed", re.compile(r"stream\s+removed", re.IGNORECASE)),
)


@dataclass(frozen=True)
class CampaignConfig:
    run_dir: Path
    source_case: Path
    source_data: Path
    processor_count: int = 1
    adapt_cycles: int = 3
    post_adapt_iterations: int = 80
    maximum_cell_count: int = 1_000_000
    maximum_refinement_level: int = 4
    minimum_edge_length_m: float = 1.0e-5
    dt_s: float = 5.0e-4
    max_iter_per_step: int = 40
    production_steps: int = 50
    displacement_tolerance_m: float = 1.0e-12
    expected_solid_cell_count: int = EXPECTED_SOLID_CELL_COUNT
    skip_adaptation: bool = False
    gate_only: bool = False

    @classmethod
    def from_environment(cls) -> "CampaignConfig":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_run_dir = CAMPAIGN_DIR / "runs" / f"run_{timestamp}"
        source_case = Path(
            os.environ.get(
                "FLUENT_FINE_SOURCE_CASE",
                str(DEFAULT_SOURCE_DIR / "steady_fluid_flow_export.cas.h5"),
            )
        ).resolve()
        source_data = Path(
            os.environ.get(
                "FLUENT_FINE_SOURCE_DATA",
                str(DEFAULT_SOURCE_DIR / "steady_fluid_flow_export.dat.h5"),
            )
        ).resolve()
        return cls(
            run_dir=Path(
                os.environ.get("FLUENT_FINE_CAMPAIGN_RUN_DIR", str(default_run_dir))
            ).resolve(),
            source_case=source_case,
            source_data=source_data,
            processor_count=int(os.environ.get("FLUENT_FSI_PROCESSOR_COUNT", "1")),
            adapt_cycles=int(os.environ.get("FLUENT_ADAPT_CYCLES", "3")),
            post_adapt_iterations=int(
                os.environ.get("FLUENT_POST_ADAPT_ITERATIONS", "80")
            ),
            maximum_cell_count=int(
                os.environ.get("FLUENT_ADAPT_MAX_CELL_COUNT", "1000000")
            ),
            maximum_refinement_level=int(
                os.environ.get("FLUENT_ADAPT_MAX_LEVEL", "4")
            ),
            minimum_edge_length_m=float(
                os.environ.get("FLUENT_ADAPT_MIN_EDGE_M", "1e-5")
            ),
            dt_s=float(os.environ.get("FLUENT_FSI_DT_S", "5e-4")),
            max_iter_per_step=int(
                os.environ.get("FLUENT_FSI_MAX_ITER_PER_STEP", "40")
            ),
            production_steps=int(os.environ.get("FLUENT_FSI_N_STEPS", "50")),
            displacement_tolerance_m=float(
                os.environ.get("FLUENT_FSI_DISPLACEMENT_GATE_M", "1e-12")
            ),
            expected_solid_cell_count=int(
                os.environ.get(
                    "FLUENT_EXPECTED_SOLID_CELL_COUNT",
                    str(EXPECTED_SOLID_CELL_COUNT),
                )
            ),
            skip_adaptation=os.environ.get(
                "FLUENT_SKIP_ADAPTATION", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            gate_only=os.environ.get("FLUENT_FINE_GATE_ONLY", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )


def config_from_cli(argv: list[str] | None = None) -> CampaignConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run fluid-only PUMA refinement, a mandatory one-step Fluent FSI gate, "
            "and optionally the full production campaign."
        )
    )
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--skip-adaptation", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-case", type=Path)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--processor-count", type=int)
    parser.add_argument("--adapt-cycles", type=int)
    parser.add_argument("--production-steps", type=int)
    parser.add_argument("--expected-solid-cell-count", type=int)
    args = parser.parse_args(argv)
    config = CampaignConfig.from_environment()
    overrides: dict[str, Any] = {}
    for argument, field_name in (
        (args.run_dir, "run_dir"),
        (args.source_case, "source_case"),
        (args.source_data, "source_data"),
        (args.processor_count, "processor_count"),
        (args.adapt_cycles, "adapt_cycles"),
        (args.production_steps, "production_steps"),
        (args.expected_solid_cell_count, "expected_solid_cell_count"),
    ):
        if argument is not None:
            overrides[field_name] = argument.resolve() if isinstance(argument, Path) else argument
    if args.gate_only:
        overrides["gate_only"] = True
    if args.skip_adaptation:
        overrides["skip_adaptation"] = True
    return replace(config, **overrides)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, default=json_default)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def append_event(run_dir: Path, label: str, **payload: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": label,
        **payload,
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                event,
                sort_keys=True,
                default=json_default,
                allow_nan=False,
            )
            + "\n"
        )


def split_hdf_names(raw: Any) -> list[str]:
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return [name for name in str(raw).replace("\x00", "").split(";") if name]


def read_cell_type_counts(case_path: Path, zone_name: str) -> dict[str, Any]:
    with h5py.File(case_path, "r") as case:
        topology = case["meshes/1/cells/zoneTopology"]
        names = split_hdf_names(topology["name"][()])
        minimum_ids = topology["minId"][()].astype(int).tolist()
        maximum_ids = topology["maxId"][()].astype(int).tolist()
        if zone_name not in names:
            raise RuntimeError(f"cell zone {zone_name!r} is absent from {case_path}")
        zone_index = names.index(zone_name)
        zone_min = int(minimum_ids[zone_index])
        zone_max = int(maximum_ids[zone_index])

        counts: Counter[int] = Counter()
        for section in case["meshes/1/cells/ctype"].values():
            section_min = int(section.attrs["minId"][0])
            section_max = int(section.attrs["maxId"][0])
            overlap_min = max(zone_min, section_min)
            overlap_max = min(zone_max, section_max)
            if overlap_min > overlap_max:
                continue
            element_type = int(section.attrs["elementType"][0])
            if element_type != 0:
                counts[element_type] += overlap_max - overlap_min + 1
                continue
            cell_types = section["cell-types"][()]
            start = overlap_min - section_min
            stop = overlap_max - section_min + 1
            unique, unique_counts = np.unique(cell_types[start:stop], return_counts=True)
            counts.update(
                {int(kind): int(count) for kind, count in zip(unique, unique_counts)}
            )

    return {
        "zone": zone_name,
        "cell_count": zone_max - zone_min + 1,
        "min_cell_id": zone_min,
        "max_cell_id": zone_max,
        "cell_type_counts": dict(sorted(counts.items())),
    }


def require_supported_solid_topology(
    case_path: Path,
    *,
    expected_cell_count: int = EXPECTED_SOLID_CELL_COUNT,
) -> dict[str, Any]:
    report = read_cell_type_counts(case_path, SOLID_ZONE)
    type_counts = set(report["cell_type_counts"])
    if report["cell_count"] != expected_cell_count:
        raise RuntimeError(
            f"{SOLID_ZONE} changed from {expected_cell_count} to "
            f"{report['cell_count']} cells in {case_path}"
        )
    if POLYHEDRAL_CELL_TYPE in type_counts:
        raise RuntimeError(
            f"{SOLID_ZONE} contains unsupported Fluent cell type 7 in {case_path}: "
            f"{report['cell_type_counts']}"
        )
    unsupported = type_counts - SUPPORTED_2D_SOLID_TYPES
    if unsupported:
        raise RuntimeError(
            f"{SOLID_ZONE} contains unsupported 2-D structural cell types "
            f"{sorted(unsupported)} in {case_path}"
        )
    return report


def normalize_zone_selection(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def configure_fluid_only_pressure_hessian(
    session: Any,
    *,
    maximum_cell_count: int,
    maximum_refinement_level: int,
    minimum_edge_length_m: float,
) -> dict[str, Any]:
    settings = session.mesh.adapt.set
    settings.adaption_method.set_state("puma")
    method = str(settings.adaption_method.get_state()).strip().lower()
    if "puma" not in method:
        raise RuntimeError(f"Fluent did not retain PUMA adaption: {method!r}")

    settings.cell_zones.set_state([FLUID_ZONE])
    session.tui.mesh.adapt.set.cell_zones(FLUID_ZONE_ID, "()")
    selected_state = settings.cell_zones.get_state()
    selected = (
        None
        if selected_state is None
        else normalize_zone_selection(selected_state)
    )
    if selected is not None and selected != [FLUID_ZONE]:
        raise RuntimeError(f"adaption cell-zone restriction failed: {selected!r}")
    settings.maximum_cell_count.set_state(maximum_cell_count)
    settings.maximum_refinement_level.set_state(maximum_refinement_level)
    settings.minimum_edge_length.set_state(minimum_edge_length_m)

    session.tui.mesh.adapt.predefined_criteria.aerodynamics.error_based.pressure_hessian_indicator()

    # Predefined criteria may update adaption settings. Re-assert the scope after it.
    settings.cell_zones.set_state([FLUID_ZONE])
    session.tui.mesh.adapt.set.cell_zones(FLUID_ZONE_ID, "()")
    selected_state = settings.cell_zones.get_state()
    selected = (
        None
        if selected_state is None
        else normalize_zone_selection(selected_state)
    )
    if selected is not None and selected != [FLUID_ZONE]:
        raise RuntimeError(
            f"pressure-Hessian criterion widened adaption scope: {selected!r}"
        )
    return {
        "adaption_method": method,
        "cell_zones": [FLUID_ZONE],
        "cell_zones_get_state": selected,
        "cell_zones_get_state_available": selected is not None,
        "cell_zone_scope_verification": (
            "settings_get_state_and_post_adapt_hdf"
            if selected is not None
            else "post_adapt_hdf_solid_topology"
        ),
        "cell_zone_tui_command": [FLUID_ZONE_ID, "()"],
        "maximum_cell_count": maximum_cell_count,
        "maximum_refinement_level": maximum_refinement_level,
        "minimum_edge_length_m": minimum_edge_length_m,
    }


def find_fatal_transcript_errors(text: str) -> list[str]:
    return [label for label, pattern in FATAL_TRANSCRIPT_PATTERNS if pattern.search(text)]


def latest_transcript(run_dir: Path) -> Path | None:
    transcripts = sorted(
        run_dir.glob("fluent-*.trn"), key=lambda path: path.stat().st_mtime
    )
    return transcripts[-1] if transcripts else None


def transcript_cursor(run_dir: Path) -> tuple[Path | None, int]:
    transcript = latest_transcript(run_dir)
    return transcript, transcript.stat().st_size if transcript else 0


def transcript_delta(
    run_dir: Path, cursor: tuple[Path | None, int]
) -> tuple[str, tuple[Path, int]]:
    previous_path, previous_size = cursor
    transcript = latest_transcript(run_dir)
    if transcript is None:
        raise RuntimeError(f"Fluent transcript was not created in {run_dir}")
    offset = previous_size if previous_path == transcript else 0
    with transcript.open("rb") as stream:
        stream.seek(min(offset, transcript.stat().st_size))
        text = stream.read().decode("utf-8", errors="replace")
    return text, (transcript, transcript.stat().st_size)


def require_clean_transcript(text: str, context: str) -> None:
    errors = find_fatal_transcript_errors(text)
    if errors:
        tail = "\n".join(text.splitlines()[-40:])
        raise RuntimeError(
            f"fatal Fluent transcript evidence during {context}: {errors}\n{tail}"
        )


def first_dataset(group: h5py.Group | h5py.Dataset) -> np.ndarray:
    if isinstance(group, h5py.Dataset):
        return group[()]
    for value in group.values():
        try:
            return first_dataset(value)
        except KeyError:
            continue
    raise KeyError(f"no dataset below {group.name}")


def read_flow_summary(data_path: Path) -> dict[str, float]:
    with h5py.File(data_path, "r") as data:
        cells = data["results/1/phase-1/cells"]
        u = np.asarray(first_dataset(cells["SV_U"]), dtype=float).ravel()
        v = np.asarray(first_dataset(cells["SV_V"]), dtype=float).ravel()
        pressure = np.asarray(first_dataset(cells["SV_P"]), dtype=float).ravel()
    speed = np.hypot(u, v)
    return {
        "velocity_min_mps": float(np.min(speed)),
        "velocity_mean_mps": float(np.mean(speed)),
        "velocity_max_mps": float(np.max(speed)),
        "pressure_min_pa": float(np.min(pressure)),
        "pressure_mean_pa": float(np.mean(pressure)),
        "pressure_max_pa": float(np.max(pressure)),
    }


def structure_node_group(data: h5py.File) -> tuple[str, h5py.Group]:
    nodes = data["special/structure-node-data/nodes"]
    for name, value in nodes.items():
        if isinstance(value, h5py.Group) and {"elemids", "ndata", "data"}.issubset(
            value.keys()
        ):
            return name, value
    raise RuntimeError("structure node data group is absent")


def read_structure_monitor(case_path: Path, data_path: Path) -> dict[str, Any]:
    with h5py.File(case_path, "r") as case, h5py.File(data_path, "r") as data:
        node_group_name, node_group = structure_node_group(data)
        coordinates_root = case["meshes/1/nodes/coords"]
        coordinates = np.asarray(
            first_dataset(
                coordinates_root[node_group_name]
                if node_group_name in coordinates_root
                else coordinates_root
            ),
            dtype=float,
        )
        node_ids = node_group["elemids"][()].astype(int)
        widths = node_group["ndata"][()].astype(int)
        if not len(widths) or len(set(widths.tolist())) != 1:
            raise RuntimeError("unexpected variable-width structure node data")
        column_count = int(widths[0])
        if column_count <= 6:
            raise RuntimeError(f"expected structural displacement columns 0 and 6, got {column_count}")
        rows = np.asarray(node_group["data"][()], dtype=float).reshape(
            len(node_ids), column_count
        )
        direct_svars = (
            data["special/structure-direct-data/data"][()].astype(int).tolist()
        )

        ranked: list[tuple[float, int, int]] = []
        for row_index, node_id in enumerate(node_ids):
            if node_id < 1 or node_id > len(coordinates):
                continue
            x, y = coordinates[node_id - 1][:2]
            distance = math.hypot(float(x) - TARGET_X_M, float(y) - TARGET_Y_M)
            ranked.append((distance, row_index, int(node_id)))
        if not ranked:
            raise RuntimeError("no structural nodes could be mapped to mesh coordinates")
        ranked.sort(key=lambda item: item[0])
        selected = ranked[: min(4, len(ranked))]
        selected_rows = np.asarray([rows[index] for _, index, _ in selected])
        displacement_x = selected_rows[:, 0]
        displacement_y = selected_rows[:, 6]
        selected_total = np.hypot(displacement_x, displacement_y)
        all_total = np.hypot(rows[:, 0], rows[:, 6])

    return {
        "target_x_displacement_m": float(np.mean(displacement_x)),
        "target_y_displacement_m": float(np.mean(displacement_y)),
        "target_total_displacement_m": float(np.mean(selected_total)),
        "solid_max_total_displacement_m": float(np.max(all_total)),
        "selected_node_ids": [node_id for _, _, node_id in selected],
        "direct_svars": direct_svars,
    }


def require_nonzero_structure_displacement(
    monitor: dict[str, Any], *, tolerance_m: float
) -> None:
    value = float(monitor["target_total_displacement_m"])
    if not math.isfinite(value) or value <= tolerance_m:
        raise RuntimeError(
            "the one-step FSI gate did not produce a finite non-zero structural "
            f"displacement: {value!r} m (gate {tolerance_m} m)"
        )


def require_valid_step_measurements(
    monitor: dict[str, Any],
    flow: dict[str, Any],
    *,
    nonzero_tolerance_m: float = STEP_NONZERO_DISPLACEMENT_TOLERANCE_M,
) -> None:
    """Reject non-finite or structurally inactive saved step measurements."""

    required_monitor = (
        "target_x_displacement_m",
        "target_y_displacement_m",
        "target_total_displacement_m",
        "solid_max_total_displacement_m",
    )
    required_flow = (
        "velocity_min_mps",
        "velocity_mean_mps",
        "velocity_max_mps",
        "pressure_min_pa",
        "pressure_mean_pa",
        "pressure_max_pa",
    )
    for label, values, keys in (
        ("structure", monitor, required_monitor),
        ("flow", flow, required_flow),
    ):
        for key in keys:
            try:
                value = float(values[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"{label} measurement {key} is invalid") from exc
            if not math.isfinite(value):
                raise RuntimeError(
                    f"{label} measurement {key} is not finite: {value!r}"
                )
    solid_max = float(monitor["solid_max_total_displacement_m"])
    if solid_max <= nonzero_tolerance_m:
        raise RuntimeError(
            "saved FSI step has zero structural displacement: "
            f"max={solid_max:.17g} m, tolerance={nonzero_tolerance_m:.17g} m"
        )


def read_surface_integrals(
    session: Any,
    *,
    reduction_api: Any | None = None,
) -> dict[str, Any]:
    """Read force and signed mass-flow integrals from the live Fluent state."""

    if reduction_api is None:
        from ansys.fluent.core.solver.function import reduction as reduction_api

    boundaries = session.setup.boundary_conditions
    flap = boundaries.wall[FLAP_FLUID_FORCE_ZONE]
    inlet = boundaries.velocity_inlet[INLET_ZONE]
    outlet = boundaries.pressure_outlet[OUTLET_ZONE]

    force = reduction_api.force(locations=[flap])
    inlet_mass_flow = float(reduction_api.mass_flow(locations=[inlet]))
    outlet_mass_flow = float(reduction_api.mass_flow(locations=[outlet]))
    force_x = float(force.x)
    force_y = float(force.y)
    force_z = float(force.z)
    net_mass_flow = inlet_mass_flow + outlet_mass_flow
    mass_flow_reference = max(abs(inlet_mass_flow), abs(outlet_mass_flow))
    relative_mass_imbalance = (
        abs(net_mass_flow) / mass_flow_reference
        if mass_flow_reference > 0.0
        else (0.0 if net_mass_flow == 0.0 else math.inf)
    )
    numeric = {
        "flap_fluid_force_x_n": force_x,
        "flap_fluid_force_y_n": force_y,
        "flap_fluid_force_z_n": force_z,
        "inlet_mass_flow_kg_s": inlet_mass_flow,
        "outlet_mass_flow_kg_s": outlet_mass_flow,
        "net_mass_flow_kg_s": net_mass_flow,
        "relative_mass_imbalance": relative_mass_imbalance,
    }
    for name, value in numeric.items():
        if not math.isfinite(value):
            raise RuntimeError(f"surface integral {name} is not finite: {value!r}")
    return {
        "flap_fluid_force_zone": FLAP_FLUID_FORCE_ZONE,
        "inlet_zone": INLET_ZONE,
        "outlet_zone": OUTLET_ZONE,
        **numeric,
    }


def launch_fluent(run_dir: Path, processor_count: int) -> Any:
    import ansys.fluent.core as pyfluent

    return pyfluent.launch_fluent(
        mode="solver",
        precision="double",
        dimension=2,
        processor_count=processor_count,
        start_timeout=240,
        cwd=str(run_dir),
        ui_mode="no_gui",
    )


def copy_latest_transcript(run_dir: Path) -> str | None:
    transcript = latest_transcript(run_dir)
    if transcript is None:
        return None
    target = run_dir / "transcript.trn"
    shutil.copy2(transcript, target)
    return str(target)


def run_adaptation(config: CampaignConfig) -> tuple[Path, Path, dict[str, Any]]:
    run_dir = config.run_dir / "adaptation"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    history_path = run_dir / "cycles.csv"
    manifest: dict[str, Any] = {
        "status": "running",
        "source_case": str(config.source_case),
        "source_data": str(config.source_data),
        "cycles": [],
    }
    atomic_write_json(manifest_path, manifest)
    session = None
    try:
        session = launch_fluent(run_dir, config.processor_count)
        session.file.read_case_data(file_name=str(config.source_case))
        for cycle in range(1, config.adapt_cycles + 1):
            cursor = transcript_cursor(run_dir)
            controls = configure_fluid_only_pressure_hessian(
                session,
                maximum_cell_count=config.maximum_cell_count,
                maximum_refinement_level=config.maximum_refinement_level,
                minimum_edge_length_m=config.minimum_edge_length_m,
            )
            session.tui.mesh.adapt.adapt_mesh()
            if config.post_adapt_iterations > 0:
                session.solution.run_calculation.iterate(
                    iter_count=config.post_adapt_iterations
                )
            transcript_text, _ = transcript_delta(run_dir, cursor)
            require_clean_transcript(transcript_text, f"adapt cycle {cycle}")

            case_path = run_dir / f"adapt_cycle_{cycle:02d}.cas.h5"
            data_path = run_dir / f"adapt_cycle_{cycle:02d}.dat.h5"
            session.file.write_case_data(file_name=str(case_path))
            topology = require_supported_solid_topology(
                case_path,
                expected_cell_count=config.expected_solid_cell_count,
            )
            flow = read_flow_summary(data_path)
            row = {"cycle": cycle, **controls, **topology, **flow}
            manifest = {**manifest, "cycles": [*manifest["cycles"], row]}
            atomic_write_json(manifest_path, manifest)
            atomic_write_csv(history_path, manifest["cycles"])
            append_event(run_dir, "adapt_cycle_passed", **row)

        final_case = run_dir / f"adapt_cycle_{config.adapt_cycles:02d}.cas.h5"
        final_data = run_dir / f"adapt_cycle_{config.adapt_cycles:02d}.dat.h5"
        manifest = {
            **manifest,
            "status": "passed",
            "final_case": str(final_case),
            "final_data": str(final_data),
        }
        atomic_write_json(manifest_path, manifest)
        return final_case, final_data, manifest
    except Exception as exc:
        manifest = {
            **manifest,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(manifest_path, manifest)
        append_event(run_dir, "adaptation_failed", error=str(exc))
        raise
    finally:
        if session is not None:
            session.exit()
        transcript = copy_latest_transcript(run_dir)
        if transcript:
            manifest = {**manifest, "transcript": transcript}
            atomic_write_json(manifest_path, manifest)


def create_required_dynamic_mesh_zones(
    session: Any, run_dir: Path
) -> dict[str, Any]:
    """Create every required dynamic zone and return Fluent's own zone listing.

    Dynamic-zone creation is part of the physical FSI contract.  A failed
    create call must therefore abort the gate; logging and continuing can
    otherwise leave a numerically running but uncoupled model.
    """

    created: list[dict[str, Any]] = []
    zone_specs = [
        *((zone_name,) for zone_name in DEFORMING_DYNAMIC_ZONES),
        (FLAP_FLUID_FORCE_ZONE, "intrinsic-fsi"),
    ]
    for arguments in zone_specs:
        zone_name = str(arguments[0])
        append_event(
            run_dir,
            "dynamic_mesh_zone_create_started",
            zone=zone_name,
            arguments=list(arguments),
        )
        try:
            session.tui.define.dynamic_mesh.zones.create(*arguments)
        except Exception as exc:
            append_event(
                run_dir,
                "dynamic_mesh_zone_create_failed",
                zone=zone_name,
                arguments=list(arguments),
                error=str(exc),
            )
            raise
        record = {"zone": zone_name, "arguments": list(arguments)}
        created.append(record)
        append_event(run_dir, "dynamic_mesh_zone_created", **record)

    try:
        raw_list_output = session.tui.define.dynamic_mesh.zones.list()
    except Exception as exc:
        append_event(run_dir, "dynamic_mesh_zone_list_failed", error=str(exc))
        raise
    list_output = "" if raw_list_output is None else str(raw_list_output)
    append_event(
        run_dir,
        "dynamic_mesh_zone_list_captured",
        output=list_output,
        returned_output=raw_list_output is not None,
    )
    return {
        "created": created,
        "list_output": list_output,
        "list_returned_output": raw_list_output is not None,
    }


def require_dynamic_zone_setup_evidence(
    *, list_output: str, transcript_text: str
) -> dict[str, Any]:
    """Require Fluent evidence for the coupled flap dynamic-zone setup.

    Some PyFluent releases return the TUI listing as a string while others
    only print it into the transcript.  The gate therefore checks the union
    of both first-party evidence channels and records which one supplied it.
    """

    combined = f"{list_output}\n{transcript_text}"
    flap_match = re.search(r"flap[_-]wall[_-]shadow", combined, re.IGNORECASE)
    motion_match = re.search(
        r"intrinsic(?:\s*[-_]\s*|\s+)fsi|deforming", combined, re.IGNORECASE
    )
    if flap_match is None or motion_match is None:
        missing = []
        if flap_match is None:
            missing.append("flap_wall-shadow")
        if motion_match is None:
            missing.append("intrinsic-fsi/deforming")
        raise RuntimeError(
            "Fluent dynamic-zone setup evidence is incomplete; missing "
            f"{missing}. TUI list={list_output!r}"
        )
    return {
        "flap_wall_shadow_evidence": flap_match.group(0),
        "motion_evidence": motion_match.group(0),
        "tui_list_contains_flap": bool(
            re.search(r"flap[_-]wall[_-]shadow", list_output, re.IGNORECASE)
        ),
        "tui_list_contains_motion": bool(
            re.search(
                r"intrinsic(?:\s*[-_]\s*|\s+)fsi|deforming",
                list_output,
                re.IGNORECASE,
            )
        ),
        "transcript_contains_flap": bool(
            re.search(r"flap[_-]wall[_-]shadow", transcript_text, re.IGNORECASE)
        ),
        "transcript_contains_motion": bool(
            re.search(
                r"intrinsic(?:\s*[-_]\s*|\s+)fsi|deforming",
                transcript_text,
                re.IGNORECASE,
            )
        ),
    }


def apply_official_fsi_setup(
    session: Any, config: CampaignConfig, run_dir: Path
) -> dict[str, Any]:
    session.setup.general.solver.time.set_state("unsteady-1st-order")
    session.setup.models.structure.model.set_state("linear-elasticity")
    try:
        session.setup.materials.solid.make_a_copy(
            from_="aluminum", to="silicone-rubber"
        )
    except Exception as exc:
        append_event(run_dir, "material_copy_skipped", error=str(exc))
    rubber = session.setup.materials.solid["silicone-rubber"]
    rubber.density.value.set_state(1600.0)
    rubber.struct_youngs_modulus.value.set_state(1.0e6)
    rubber.struct_poisson_ratio.value.set_state(0.47)
    session.setup.cell_zone_conditions.solid[SOLID_ZONE].general.material.set_state(
        "silicone-rubber"
    )

    attach = session.setup.boundary_conditions.wall["flap_attach"].structure
    attach.x_disp_boundary_condition.set_state("Node X-Displacement")
    attach.x_disp_boundary_value.set_state(0.0)
    attach.y_disp_boundary_condition.set_state("Node Y-Displacement")
    attach.y_disp_boundary_value.set_state(0.0)
    flap_wall = session.setup.boundary_conditions.wall["flap_wall"].structure
    flap_wall.x_disp_boundary_condition.set_state("Intrinsic FSI")
    flap_wall.y_disp_boundary_condition.set_state("Intrinsic FSI")

    session.setup.dynamic_mesh.enabled.set_state(True)
    dynamic_zones = create_required_dynamic_mesh_zones(session, run_dir)
    controls = session.solution.run_calculation.transient_controls
    controls.time_step_size.set_state(config.dt_s)
    controls.max_iter_per_time_step.set_state(config.max_iter_per_step)
    return {
        "dynamic_zones": dynamic_zones,
        "time_step_size_s": config.dt_s,
        "max_iter_per_time_step": config.max_iter_per_step,
    }


def flatten_step_row(step: dict[str, Any]) -> dict[str, Any]:
    topology = step["solid_topology"]
    monitor = step["structure"]
    flow = step["flow"]
    surface_integrals = step["surface_integrals"]
    return {
        "step": step["step"],
        "time_s": step["time_s"],
        "seconds": step["seconds"],
        "case_path": step["case_path"],
        "data_path": step["data_path"],
        "solid_cell_count": topology["cell_count"],
        "solid_cell_type_counts": topology["cell_type_counts"],
        "target_x_displacement_m": monitor["target_x_displacement_m"],
        "target_y_displacement_m": monitor["target_y_displacement_m"],
        "target_total_displacement_m": monitor["target_total_displacement_m"],
        "solid_max_total_displacement_m": monitor[
            "solid_max_total_displacement_m"
        ],
        "selected_node_ids": monitor["selected_node_ids"],
        **flow,
        **surface_integrals,
    }


def run_fsi_phase(
    config: CampaignConfig,
    *,
    phase_name: str,
    steady_case: Path,
    steady_data: Path,
    step_count: int,
    require_first_step_displacement: bool,
) -> dict[str, Any]:
    run_dir = config.run_dir / phase_name
    run_dir.mkdir(parents=True, exist_ok=False)
    steps_dir = run_dir / "steps"
    steps_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    history_path = run_dir / "history.csv"
    manifest: dict[str, Any] = {
        "status": "running",
        "steady_case": str(steady_case),
        "steady_data": str(steady_data),
        "requested_steps": step_count,
        "steps": [],
    }
    atomic_write_json(manifest_path, manifest)
    session = None
    try:
        require_supported_solid_topology(
            steady_case,
            expected_cell_count=config.expected_solid_cell_count,
        )
        session = launch_fluent(run_dir, config.processor_count)
        session.file.read_case_data(file_name=str(steady_case))
        setup_cursor = transcript_cursor(run_dir)
        setup_evidence = apply_official_fsi_setup(session, config, run_dir)
        setup_transcript_text, _ = transcript_delta(run_dir, setup_cursor)
        require_clean_transcript(setup_transcript_text, f"{phase_name} setup")
        dynamic_zone_evidence = require_dynamic_zone_setup_evidence(
            list_output=setup_evidence["dynamic_zones"]["list_output"],
            transcript_text=setup_transcript_text,
        )
        setup_evidence = {
            **setup_evidence,
            "dynamic_zone_evidence": dynamic_zone_evidence,
            "transcript_delta_character_count": len(setup_transcript_text),
        }
        manifest = {**manifest, "setup_evidence": setup_evidence}
        atomic_write_json(manifest_path, manifest)
        append_event(run_dir, "fsi_setup_evidence_passed", **setup_evidence)
        setup_case = run_dir / "fsi_setup.cas.h5"
        session.file.write_case_data(file_name=str(setup_case))
        require_supported_solid_topology(
            setup_case,
            expected_cell_count=config.expected_solid_cell_count,
        )

        for step_index in range(1, step_count + 1):
            started = time.time()
            cursor = transcript_cursor(run_dir)
            session.solution.run_calculation.dual_time_iterate(
                time_step_count=1,
                max_iter_per_step=config.max_iter_per_step,
            )
            transcript_text, _ = transcript_delta(run_dir, cursor)
            require_clean_transcript(
                transcript_text, f"{phase_name} step {step_index}"
            )

            case_path = steps_dir / f"step_{step_index:04d}.cas.h5"
            data_path = steps_dir / f"step_{step_index:04d}.dat.h5"
            session.file.write_case_data(file_name=str(case_path))
            topology = require_supported_solid_topology(
                case_path,
                expected_cell_count=config.expected_solid_cell_count,
            )
            monitor = read_structure_monitor(case_path, data_path)
            flow = read_flow_summary(data_path)
            require_valid_step_measurements(monitor, flow)
            surface_integrals = read_surface_integrals(session)
            if step_index == 1 and require_first_step_displacement:
                require_nonzero_structure_displacement(
                    monitor, tolerance_m=config.displacement_tolerance_m
                )
            step = {
                "step": step_index,
                "time_s": step_index * config.dt_s,
                "seconds": time.time() - started,
                "case_path": str(case_path),
                "data_path": str(data_path),
                "solid_topology": topology,
                "structure": monitor,
                "flow": flow,
                "surface_integrals": surface_integrals,
            }
            manifest = {**manifest, "steps": [*manifest["steps"], step]}
            atomic_write_json(manifest_path, manifest)
            atomic_write_csv(
                history_path, [flatten_step_row(item) for item in manifest["steps"]]
            )
            append_event(run_dir, "fsi_step_passed", **flatten_step_row(step))

        final_step = manifest["steps"][-1]
        manifest = {
            **manifest,
            "status": "passed",
            "completed_steps": len(manifest["steps"]),
            "final_case": final_step["case_path"],
            "final_data": final_step["data_path"],
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest = {
            **manifest,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(manifest_path, manifest)
        append_event(run_dir, "fsi_phase_failed", error=str(exc))
        raise
    finally:
        if session is not None:
            session.exit()
        transcript = copy_latest_transcript(run_dir)
        if transcript:
            manifest = {**manifest, "transcript": transcript}
            atomic_write_json(manifest_path, manifest)


def config_manifest(config: CampaignConfig) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(config).items()
    }


def validate_config(config: CampaignConfig) -> dict[str, Any]:
    if config.run_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite campaign directory: {config.run_dir}"
        )
    if not config.source_case.is_file() or not config.source_data.is_file():
        raise FileNotFoundError(
            f"valid steady case/data are required: {config.source_case}, "
            f"{config.source_data}"
        )
    if config.processor_count != 1:
        raise ValueError(
            "this guarded campaign is serial-only; processor_count must equal 1"
        )
    if config.adapt_cycles < 1:
        raise ValueError("adapt_cycles must be positive")
    if config.production_steps < 1:
        raise ValueError("production_steps must be positive")
    if config.expected_solid_cell_count < 1:
        raise ValueError("expected_solid_cell_count must be positive")
    return require_supported_solid_topology(
        config.source_case,
        expected_cell_count=config.expected_solid_cell_count,
    )


def main(argv: list[str] | None = None) -> int:
    config = config_from_cli(argv)
    source_topology = validate_config(config)
    config.run_dir.mkdir(parents=True)
    manifest_path = config.run_dir / "campaign_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "config": config_manifest(config),
        "source_solid_topology": source_topology,
        "phases": {},
    }
    atomic_write_json(manifest_path, manifest)
    append_event(config.run_dir, "campaign_started", config=config_manifest(config))
    try:
        if config.skip_adaptation:
            fine_case = config.source_case
            fine_data = config.source_data
            adaptation = {
                "status": "skipped_prebuilt_fine_mesh",
                "reason": "externally generated mesh; Fluent intrinsic structure is incompatible with solver mesh adaption",
                "final_case": str(fine_case),
                "final_data": str(fine_data),
                "solid_topology": source_topology,
            }
            append_event(
                config.run_dir,
                "adaptation_skipped_prebuilt_fine_mesh",
                case_path=fine_case,
                data_path=fine_data,
            )
        else:
            fine_case, fine_data, adaptation = run_adaptation(config)
        manifest = {
            **manifest,
            "phases": {**manifest["phases"], "adaptation": adaptation},
        }
        atomic_write_json(manifest_path, manifest)

        gate = run_fsi_phase(
            config,
            phase_name="fsi_gate_1step",
            steady_case=fine_case,
            steady_data=fine_data,
            step_count=1,
            require_first_step_displacement=True,
        )
        manifest = {
            **manifest,
            "phases": {**manifest["phases"], "fsi_gate_1step": gate},
        }
        atomic_write_json(manifest_path, manifest)

        if not config.gate_only:
            production = run_fsi_phase(
                config,
                phase_name="fsi_production",
                steady_case=fine_case,
                steady_data=fine_data,
                step_count=config.production_steps,
                require_first_step_displacement=True,
            )
            manifest = {
                **manifest,
                "phases": {**manifest["phases"], "fsi_production": production},
            }

        manifest = {**manifest, "status": "passed"}
        atomic_write_json(manifest_path, manifest)
        append_event(config.run_dir, "campaign_passed")
        return 0
    except Exception as exc:
        manifest = {
            **manifest,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(manifest_path, manifest)
        append_event(config.run_dir, "campaign_failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
