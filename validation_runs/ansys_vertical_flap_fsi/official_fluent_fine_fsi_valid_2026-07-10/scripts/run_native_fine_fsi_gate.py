"""Import the native fine mesh and run an evidence-gated Fluent FSI smoke test.

This is intentionally a fresh run.  It reuses the public tutorial's physical
setup, but never reads the historical PUMA-adapted case/data because those
artifacts contain unsupported type-7 cells in ``solid.5``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_fine_fsi_campaign as guarded
import refine_native_fluent_mesh as native_mesh


DEFAULT_MESH = CAMPAIGN_DIR / "mesh" / "flap_halfdomain_h0p25mm_native_ascii.msh"
DEFAULT_MESH_MANIFEST = (
    CAMPAIGN_DIR / "mesh" / "flap_halfdomain_h0p25mm_native_ascii_manifest.json"
)
EXPECTED_SOLID_CELL_COUNT = 480
EXPECTED_FLUID_CELL_COUNT = 63_040
EXPECTED_SOLID_CELL_TYPES = frozenset({1, 3})
REQUIRED_CELL_ZONES = frozenset({"solid.5", "fluid.4"})
REQUIRED_FACE_ZONES = frozenset(
    {
        "flap_attach",
        "flap_wall",
        "flap_wall-shadow",
        "velocity_inlet.1",
        "po.3",
        "symmetry.2",
        "wall",
    }
)


@dataclass(frozen=True)
class NativeGateConfig:
    run_dir: Path
    mesh_path: Path = DEFAULT_MESH
    mesh_manifest_path: Path = DEFAULT_MESH_MANIFEST
    processor_count: int = 1
    steady_iterations: int = 100
    dt_s: float = 5.0e-4
    max_iter_per_step: int = 40
    displacement_tolerance_m: float = 1.0e-12


def config_from_cli(argv: list[str] | None = None) -> NativeGateConfig:
    parser = argparse.ArgumentParser(
        description="Run a native-fine-mesh Fluent steady preflow and 1-step FSI gate."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--mesh-manifest", type=Path, default=DEFAULT_MESH_MANIFEST)
    parser.add_argument("--processor-count", type=int, default=1)
    parser.add_argument("--steady-iterations", type=int, default=100)
    parser.add_argument("--dt-s", type=float, default=5.0e-4)
    parser.add_argument("--max-iter-per-step", type=int, default=40)
    parser.add_argument("--displacement-tolerance-m", type=float, default=1.0e-12)
    args = parser.parse_args(argv)
    return NativeGateConfig(
        run_dir=args.run_dir.resolve(),
        mesh_path=args.mesh.resolve(),
        mesh_manifest_path=args.mesh_manifest.resolve(),
        processor_count=args.processor_count,
        steady_iterations=args.steady_iterations,
        dt_s=args.dt_s,
        max_iter_per_step=args.max_iter_per_step,
        displacement_tolerance_m=args.displacement_tolerance_m,
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_event(run_dir: Path, label: str, **payload: Any) -> None:
    guarded.append_event(run_dir, label, **payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_matching_value(
    *, label: str, manifest_value: Any, parsed_value: Any
) -> None:
    if manifest_value != parsed_value:
        raise RuntimeError(
            f"native mesh manifest mismatch for {label}: "
            f"manifest={manifest_value!r}, parsed={parsed_value!r}"
        )


def validate_offline_mesh(mesh_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not mesh_path.is_file():
        raise FileNotFoundError(f"native fine mesh is absent: {mesh_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"native fine mesh manifest is absent: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = payload.get("validation", {})
    cell_counts = validation.get("cell_counts_by_zone_name", {})
    raw_types = validation.get("cell_types_by_zone_name", {}).get("solid.5", {})
    solid_types = {int(kind): int(count) for kind, count in raw_types.items()}
    solid_count = int(cell_counts.get("solid.5", 0))
    cross_faces = int(validation.get("cross_cell_zone_face_count", 0))
    if solid_count != EXPECTED_SOLID_CELL_COUNT:
        raise RuntimeError(
            f"offline solid.5 count is {solid_count}, expected {EXPECTED_SOLID_CELL_COUNT}"
        )
    unsupported = set(solid_types) - EXPECTED_SOLID_CELL_TYPES
    if unsupported:
        raise RuntimeError(
            f"offline solid.5 contains unsupported cell types {sorted(unsupported)}: "
            f"{solid_types}"
        )
    if sum(solid_types.values()) != solid_count:
        raise RuntimeError(
            f"offline solid.5 type counts do not total {solid_count}: {solid_types}"
        )
    if cross_faces <= 0:
        raise RuntimeError("offline mesh has no fluid-solid cross-zone faces")

    declared_mesh = payload.get("output_mesh")
    if declared_mesh is None or Path(declared_mesh).resolve() != mesh_path.resolve():
        raise RuntimeError(
            "native mesh manifest output_mesh does not identify the selected mesh: "
            f"manifest={declared_mesh!r}, selected={str(mesh_path)!r}"
        )
    expected_sha256 = str(payload.get("output_mesh_sha256", "")).lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise RuntimeError("native mesh manifest has no valid output_mesh_sha256")
    actual_sha256 = sha256_file(mesh_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "native mesh SHA-256 does not match its manifest: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )

    parsed_mesh = native_mesh.load_fluent_ascii_mesh(mesh_path)
    parsed_validation = native_mesh.validate_mesh(parsed_mesh)
    parsed_counts = parsed_validation["cell_counts_by_zone_name"]
    parsed_raw_types = parsed_validation["cell_types_by_zone_name"]["solid.5"]
    parsed_solid_types = {
        int(kind): int(count) for kind, count in parsed_raw_types.items()
    }
    parsed_solid_count = int(parsed_counts.get("solid.5", 0))
    parsed_fluid_count = int(parsed_counts.get("fluid.4", 0))
    parsed_cross_faces = int(parsed_validation["cross_cell_zone_face_count"])
    if parsed_solid_count != EXPECTED_SOLID_CELL_COUNT:
        raise RuntimeError(
            f"parsed solid.5 count is {parsed_solid_count}, "
            f"expected {EXPECTED_SOLID_CELL_COUNT}"
        )
    if parsed_fluid_count != EXPECTED_FLUID_CELL_COUNT:
        raise RuntimeError(
            f"parsed fluid.4 count is {parsed_fluid_count}, "
            f"expected {EXPECTED_FLUID_CELL_COUNT}"
        )
    parsed_unsupported = set(parsed_solid_types) - EXPECTED_SOLID_CELL_TYPES
    if parsed_unsupported or sum(parsed_solid_types.values()) != parsed_solid_count:
        raise RuntimeError(
            "parsed solid.5 topology is not supported by intrinsic structure: "
            f"{parsed_solid_types}"
        )
    if parsed_cross_faces <= 0:
        raise RuntimeError("parsed mesh has no fluid-solid cross-zone faces")

    normalized_manifest_types = {
        zone_name: {int(kind): int(count) for kind, count in type_counts.items()}
        for zone_name, type_counts in validation.get(
            "cell_types_by_zone_name", {}
        ).items()
    }
    normalized_parsed_types = {
        zone_name: {int(kind): int(count) for kind, count in type_counts.items()}
        for zone_name, type_counts in parsed_validation[
            "cell_types_by_zone_name"
        ].items()
    }
    require_matching_value(
        label="cell_counts_by_zone_name",
        manifest_value={key: int(value) for key, value in cell_counts.items()},
        parsed_value={key: int(value) for key, value in parsed_counts.items()},
    )
    require_matching_value(
        label="cell_types_by_zone_name",
        manifest_value=normalized_manifest_types,
        parsed_value=normalized_parsed_types,
    )
    require_matching_value(
        label="cross_cell_zone_face_count",
        manifest_value=cross_faces,
        parsed_value=parsed_cross_faces,
    )

    parsed_bounds = {
        key: float(value) for key, value in parsed_validation["bounds_m"].items()
    }
    expected_domain_bounds = {
        "x_min": 0.0,
        "x_max": native_mesh.DOMAIN_X_M,
        "y_min": 0.0,
        "y_max": native_mesh.DOMAIN_Y_M,
    }
    require_matching_value(
        label="domain bounds",
        manifest_value=expected_domain_bounds,
        parsed_value=parsed_bounds,
    )
    manifest_bounds = {
        key: float(value)
        for key, value in validation.get("bounds_m", {}).items()
    }
    require_matching_value(
        label="validation.bounds_m",
        manifest_value=manifest_bounds,
        parsed_value=parsed_bounds,
    )

    solid_node_ids = {
        node_id
        for cell in parsed_mesh.cells
        if cell.zone_id == native_mesh.SOLID_ZONE_ID
        for node_id in cell.nodes
    }
    solid_x = [parsed_mesh.nodes[node_id][0] for node_id in solid_node_ids]
    solid_y = [parsed_mesh.nodes[node_id][1] for node_id in solid_node_ids]
    parsed_solid_bounds = {
        "x_min": min(solid_x),
        "x_max": max(solid_x),
        "y_min": min(solid_y),
        "y_max": max(solid_y),
    }
    expected_solid_bounds = {
        "x_min": native_mesh.FLAP_X0_M,
        "x_max": native_mesh.FLAP_X1_M,
        "y_min": 0.0,
        "y_max": native_mesh.FLAP_Y1_M,
    }
    require_matching_value(
        label="solid flap bounds",
        manifest_value=expected_solid_bounds,
        parsed_value=parsed_solid_bounds,
    )
    return {
        "mesh_path": str(mesh_path),
        "mesh_manifest_path": str(manifest_path),
        "actual_mesh_sha256": actual_sha256,
        "solid_cell_count": parsed_solid_count,
        "solid_cell_type_counts": parsed_solid_types,
        "fluid_cell_count": parsed_fluid_count,
        "cross_cell_zone_face_count": parsed_cross_faces,
        "parsed_bounds_m": parsed_bounds,
        "parsed_solid_bounds_m": parsed_solid_bounds,
        "parsed_node_count": int(parsed_validation["node_count"]),
        "parsed_face_count": int(parsed_validation["face_count"]),
        "parsed_cell_count": int(parsed_validation["cell_count"]),
    }


def validate_config(config: NativeGateConfig) -> dict[str, Any]:
    if config.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite native gate run: {config.run_dir}")
    if config.processor_count != 1:
        raise ValueError("the intrinsic-structure gate is serial-only")
    if config.steady_iterations < 1:
        raise ValueError("steady_iterations must be positive")
    if config.dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if config.max_iter_per_step < 1:
        raise ValueError("max_iter_per_step must be positive")
    return validate_offline_mesh(config.mesh_path, config.mesh_manifest_path)


def official_steady_commands() -> list[str]:
    """Return the historical tutorial physics without any mesh adaption command."""

    return [
        "/define/models/viscous/kw-sst? yes",
        "/define/boundary-conditions/zone-name wall:008 flap_attach",
        "/define/boundary-conditions/zone-name default-interior:010 flap_wall",
        "/define/boundary-conditions/zone-type solid.5 solid",
        "/define/boundary-conditions/set/velocity-inlet velocity_inlet.1 () vmag no 10 ()",
        "/define/operating-conditions/operating-pressure 1013250",
        "/define/materials/change-create air air yes constant 1.2 no no yes constant 1.8e-05 no no no",
        "/solve/set/p-v-coupling 24",
        "/solve/initialize/hyb-initialization",
        "/solve/set/pseudo-transient yes yes 1 5 0",
    ]


def execute_tui(session: Any, run_dir: Path, command: str) -> None:
    started = time.time()
    append_event(run_dir, "tui_started", command=command)
    session.execute_tui(command)
    append_event(
        run_dir,
        "tui_completed",
        command=command,
        seconds=time.time() - started,
    )


def read_zone_names(case_path: Path) -> dict[str, list[str]]:
    import h5py

    with h5py.File(case_path, "r") as case:
        cell_names = guarded.split_hdf_names(
            case["meshes/1/cells/zoneTopology/name"][()]
        )
        face_names = guarded.split_hdf_names(
            case["meshes/1/faces/zoneTopology/name"][()]
        )
    return {"cell_zones": cell_names, "face_zones": face_names}


def require_expected_zone_names(case_path: Path) -> dict[str, list[str]]:
    report = read_zone_names(case_path)
    missing_cells = REQUIRED_CELL_ZONES - set(report["cell_zones"])
    missing_faces = REQUIRED_FACE_ZONES - set(report["face_zones"])
    if missing_cells or missing_faces:
        raise RuntimeError(
            "Fluent zone-name gate failed: "
            f"missing cell zones={sorted(missing_cells)}, "
            f"missing face zones={sorted(missing_faces)}, actual={report}"
        )
    return report


def copy_latest_transcript(run_dir: Path) -> str | None:
    transcript = guarded.latest_transcript(run_dir)
    if transcript is None:
        return None
    target = run_dir / "transcript.trn"
    shutil.copy2(transcript, target)
    return str(target)


def run_steady_preflow(
    config: NativeGateConfig,
) -> tuple[Path, Path, dict[str, Any]]:
    run_dir = config.run_dir / "steady_preflow"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    case_path = run_dir / "native_fine_steady.cas.h5"
    data_path = run_dir / "native_fine_steady.dat.h5"
    manifest: dict[str, Any] = {
        "status": "running",
        "mesh_path": str(config.mesh_path),
        "processor_count": config.processor_count,
        "steady_iterations": config.steady_iterations,
        "adaptation": "forbidden_not_run",
        "commands": official_steady_commands(),
    }
    atomic_write_json(manifest_path, manifest)
    session = None
    try:
        session = guarded.launch_fluent(run_dir, config.processor_count)
        manifest = {**manifest, "fluent_version": str(session.get_fluent_version())}
        atomic_write_json(manifest_path, manifest)
        append_event(run_dir, "fluent_started", version=manifest["fluent_version"])

        cursor = guarded.transcript_cursor(run_dir)
        session.file.read_case(file_name=str(config.mesh_path))
        append_event(run_dir, "mesh_imported", mesh_path=config.mesh_path)
        for command in official_steady_commands():
            execute_tui(session, run_dir, command)
        append_event(run_dir, "steady_setup_completed")
        session.solution.run_calculation.iterate(iter_count=config.steady_iterations)
        append_event(run_dir, "steady_iterations_completed", count=config.steady_iterations)
        session.file.write_case_data(file_name=str(case_path))
        append_event(run_dir, "steady_case_data_written", case_path=case_path, data_path=data_path)

        transcript_text, _ = guarded.transcript_delta(run_dir, cursor)
        guarded.require_clean_transcript(transcript_text, "native fine mesh steady preflow")
        topology = guarded.require_supported_solid_topology(
            case_path, expected_cell_count=EXPECTED_SOLID_CELL_COUNT
        )
        zones = require_expected_zone_names(case_path)
        flow = guarded.read_flow_summary(data_path)
        manifest = {
            **manifest,
            "status": "passed",
            "case_path": str(case_path),
            "data_path": str(data_path),
            "solid_topology": topology,
            "zones": zones,
            "flow": flow,
        }
        atomic_write_json(manifest_path, manifest)
        append_event(run_dir, "steady_preflow_passed", solid_topology=topology)
        return case_path, data_path, manifest
    except Exception as exc:
        manifest = {
            **manifest,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(manifest_path, manifest)
        append_event(run_dir, "steady_preflow_failed", error=str(exc))
        raise
    finally:
        if session is not None:
            session.exit()
        transcript = copy_latest_transcript(run_dir)
        if transcript:
            manifest = {**manifest, "transcript": transcript}
            atomic_write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    config = config_from_cli(argv)
    offline_mesh = validate_config(config)
    config.run_dir.mkdir(parents=True)
    manifest_path = config.run_dir / "native_gate_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()},
        "offline_mesh": offline_mesh,
        "adaptation": "forbidden_not_run",
        "phases": {},
    }
    atomic_write_json(manifest_path, manifest)
    append_event(config.run_dir, "native_gate_started", offline_mesh=offline_mesh)
    try:
        steady_case, steady_data, steady = run_steady_preflow(config)
        manifest = {**manifest, "phases": {"steady_preflow": steady}}
        atomic_write_json(manifest_path, manifest)

        fsi_config = guarded.CampaignConfig(
            run_dir=config.run_dir,
            source_case=steady_case,
            source_data=steady_data,
            processor_count=1,
            adapt_cycles=1,
            post_adapt_iterations=0,
            dt_s=config.dt_s,
            max_iter_per_step=config.max_iter_per_step,
            production_steps=1,
            displacement_tolerance_m=config.displacement_tolerance_m,
            expected_solid_cell_count=EXPECTED_SOLID_CELL_COUNT,
            skip_adaptation=True,
            gate_only=True,
        )
        gate = guarded.run_fsi_phase(
            fsi_config,
            phase_name="fsi_gate_1step",
            steady_case=steady_case,
            steady_data=steady_data,
            step_count=1,
            require_first_step_displacement=True,
        )
        manifest = {
            **manifest,
            "status": "passed",
            "phases": {**manifest["phases"], "fsi_gate_1step": gate},
        }
        atomic_write_json(manifest_path, manifest)
        append_event(config.run_dir, "native_gate_passed")
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
        append_event(config.run_dir, "native_gate_failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
