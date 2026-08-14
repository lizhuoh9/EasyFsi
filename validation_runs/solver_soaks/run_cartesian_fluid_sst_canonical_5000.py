"""Run one uninterrupted 5000-step Cartesian fluid/SST durability soak.

The primary trajectory owns exactly one ``CartesianFluidSolver`` from step 1
through step 5000.  Step 2500 writes a schema-v8 checkpoint, but the primary
solver is neither restored nor replaced.  Only after the primary trajectory
has finished is a fresh solver restored from that checkpoint and replayed from
step 2501 through step 5000.  Non-pressure state remains bitwise identical;
the f64 pressure fields use a machine-precision, scale-aware comparison because
CUDA reductions do not promise a fixed addition order across fresh allocations.
Full state digests remain in the report as diagnostics rather than a false
physical pass/fail gate.

This is an opt-in CUDA validation runner, not a default unit test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids.preflow_snapshot import (
    PREFLOW_SNAPSHOT_FIELD_NAMES,
    PREFLOW_SNAPSHOT_SCHEMA_VERSION,
    PreflowSnapshot,
    PreflowSnapshotIdentity,
    load_preflow_snapshot,
    save_preflow_snapshot,
    validate_preflow_snapshot_fields,
)


TOTAL_STEPS = 5000
CHECKPOINT_STEP = 2500
DIAGNOSTIC_INTERVAL = 100
DT_S = 2.0e-3
GRID_NODES = (4, 5, 6)
MOVING_WALL_SPEED_MPS = 0.05
MOVING_WALL_SEGMENT_SCALES = (1.0, 0.5, -1.0, -0.5, 1.0)
MAX_TRANSPORT_SUBSTEPS = 64
MAX_MOMENTUM_SUBSTEP_CFL = 0.450001
MAX_SST_SUBSTEP_CFL = 0.900001
PRESSURE_REPLAY_ROUNDOFF_FACTOR = 64.0
PRESSURE_STATE_FIELDS = frozenset(("pressure", "fsi_pressure"))
PRESSURE_DIAGNOSTIC_FIELDS = frozenset(
    ("pressure_min_pa", "pressure_max_pa", "pressure_range_pa")
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "validation_runs"
    / "solver_soaks"
    / "cartesian_fluid_sst_canonical_5000"
)


@dataclass(frozen=True)
class SoakConfig:
    output_dir: Path
    arch: str = "cuda"


def config_from_cli(argv: list[str] | None = None) -> SoakConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run an uninterrupted 5000-step CartesianFluidSolver SST/canonical "
            "soak, then replay steps 2501..5000 from its schema-v8 checkpoint."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the live report and schema-v8 checkpoint.",
    )
    parser.add_argument(
        "--arch",
        choices=("cuda", "gpu"),
        default="cuda",
        help="GPU Taichi backend (default: cuda).",
    )
    args = parser.parse_args(argv)
    return SoakConfig(output_dir=args.output_dir.resolve(), arch=args.arch)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _bit_count(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=np.uint32).reshape(-1)
    return int(sum(int(value).bit_count() for value in flat))


def _wall_scale_for_step(step: int) -> float:
    if not 1 <= step <= TOTAL_STEPS:
        raise ValueError(f"step must lie in [1, {TOTAL_STEPS}], got {step}")
    segment = min(
        (step - 1) // (TOTAL_STEPS // len(MOVING_WALL_SEGMENT_SCALES)),
        len(MOVING_WALL_SEGMENT_SCALES) - 1,
    )
    return float(MOVING_WALL_SEGMENT_SCALES[segment])


def _canonical_ledger_arrays(
    *, wall_scale: float
) -> dict[str, np.ndarray]:
    """Build an internal A/B wall ledger with counter-moving tangential patches.

    The lower y interface stores its normal component on the fluid owner row
    (canonical A orientation, j=1).  The upper interface stores its normal
    component on the obstacle owner row (canonical B orientation, j=4).
    Tangential x targets live on the adjacent fluid rows, as required by the
    component-face storage contract.
    """

    shape = GRID_NODES
    vector_shape = shape + (3,)
    active = np.zeros(shape, dtype=np.int32)
    hard = np.zeros(shape, dtype=np.int32)
    owned = np.zeros(shape, dtype=np.int32)
    target = np.zeros(vector_shape, dtype=np.float32)
    mobility = np.ones(vector_shape, dtype=np.float32)
    enforcement = np.zeros(vector_shape, dtype=np.float32)
    region = np.full(vector_shape, -1, dtype=np.int32)

    # Complete zero-normal-flux coverage on both internal obstacle interfaces.
    active[:, 1, :] |= 0b010
    hard[:, 1, :] |= 0b010
    owned[:, 1, :] |= 0b010
    active[:, 4, :] |= 0b010
    hard[:, 4, :] |= 0b010
    owned[:, 4, :] |= 0b010

    # Counter-moving tangential patches.  Leave one-cell x/z margins so the
    # soak exercises interior canonical rows rather than external corners.
    patch = (slice(1, 3), slice(1, 5))
    active[patch[0], 1, patch[1]] |= 0b001
    hard[patch[0], 1, patch[1]] |= 0b001
    owned[patch[0], 1, patch[1]] |= 0b001
    active[patch[0], 3, patch[1]] |= 0b001
    hard[patch[0], 3, patch[1]] |= 0b001
    owned[patch[0], 3, patch[1]] |= 0b001
    speed = np.float32(MOVING_WALL_SPEED_MPS * wall_scale)
    target[patch[0], 1, patch[1], 0] = speed
    target[patch[0], 3, patch[1], 0] = -speed

    for axis in range(3):
        component_active = (active & (1 << axis)) != 0
        mobility[..., axis][component_active] = 0.0
        enforcement[..., axis][component_active] = 1.0
        region[..., axis][component_active] = np.int32(101 + 101 * (axis == 0))
    # Preserve distinct lower/upper provenance for diagnostics.
    region[:, 1, :, 1] = 101
    region[:, 4, :, 1] = 202
    region[patch[0], 1, patch[1], 0] = 101
    region[patch[0], 3, patch[1], 0] = 202

    return {
        "velocity_dirichlet_boundary_active_component_mask": active,
        "velocity_dirichlet_boundary_hard_fixed_component_mask": hard,
        "velocity_dirichlet_boundary_owned_component_mask": owned,
        "velocity_dirichlet_boundary_value_mps": target,
        "velocity_dirichlet_boundary_pressure_mobility": mobility,
        "velocity_dirichlet_boundary_component_enforcement_weight": enforcement,
        "velocity_dirichlet_boundary_component_region_id": region,
    }


def _prepare_and_seal_canonical_ledger(solver: CartesianFluidSolver) -> None:
    """Publish one immutable generation to every physical consumer."""

    solver.prepare_velocity_dirichlet_component_ledger_apply()
    solver.prepare_velocity_dirichlet_component_ledger_divergence()
    solver.prepare_velocity_dirichlet_component_ledger_reachability()
    solver.prepare_velocity_dirichlet_component_ledger_fv_operator()
    solver.prepare_velocity_dirichlet_component_ledger_gradient()
    solver.prepare_velocity_dirichlet_component_ledger_multigrid()
    solver.prepare_velocity_dirichlet_component_ledger_projection()
    solver.prepare_hibm_no_slip_component_face_valid_mask()
    solver.prepare_velocity_dirichlet_component_ledger_reference()
    solver.prepare_velocity_dirichlet_component_ledger_snapshot()
    solver.seal_velocity_dirichlet_component_ledger()
    solver._require_velocity_dirichlet_component_ledger_sealed()


def _publish_wall_scale(
    solver: CartesianFluidSolver,
    *,
    wall_scale: float,
    replace_complete_ledger: bool,
) -> None:
    arrays = _canonical_ledger_arrays(wall_scale=wall_scale)
    if replace_complete_ledger:
        for name, values in arrays.items():
            getattr(solver, name).from_numpy(values)
    else:
        solver.velocity_dirichlet_boundary_value_mps.from_numpy(
            arrays["velocity_dirichlet_boundary_value_mps"]
        )
    solver._invalidate_velocity_dirichlet_component_ledger()
    _prepare_and_seal_canonical_ledger(solver)
    solver.apply_velocity_dirichlet_boundary_rows(read_report=False)


def _build_solver(*, arch: str, initial_step: int) -> CartesianFluidSolver:
    solver = CartesianFluidSolver(
        FluidDomainSpec.unit_box(
            grid_nodes=GRID_NODES,
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-3,
            dt_s=DT_S,
        ),
        runtime=TaichiRuntimeConfig(arch=arch),
    )

    obstacle = np.zeros(GRID_NODES, dtype=np.int32)
    obstacle[:, 0, :] = 1
    obstacle[:, 4, :] = 1
    solver.obstacle.from_numpy(obstacle)
    solver.hibm_base_obstacle.from_numpy(obstacle)
    solver._hibm_base_obstacle_initialized = True
    solver.configure_sst_2003(
        inlet_velocity_mps=0.05,
        turbulence_intensity=0.05,
        turbulent_viscosity_ratio=10.0,
        inlet_face="zmax",
        outlet_face="zmin",
        no_slip_domain_walls=(False, False, False, False, False, False),
        max_automatic_substeps=MAX_TRANSPORT_SUBSTEPS,
    )
    solver.set_uniform_velocity((0.0, 0.0, 0.0))
    solver.set_velocity_dirichlet_boundary_authority("canonical")
    _publish_wall_scale(
        solver,
        wall_scale=_wall_scale_for_step(initial_step),
        replace_complete_ledger=True,
    )
    return solver


def _snapshot_identity() -> PreflowSnapshotIdentity:
    runner_path = Path(__file__).resolve()
    solver_path = REPO_ROOT / "simulation_core" / "fluids" / "solver.py"
    obstacle = np.zeros(GRID_NODES, dtype=np.int32)
    obstacle[:, 0, :] = 1
    obstacle[:, 4, :] = 1
    ledger = _canonical_ledger_arrays(wall_scale=1.0)
    return PreflowSnapshotIdentity.from_inputs(
        config={
            "grid_nodes": GRID_NODES,
            "dt_s": DT_S,
            "density_kgm3": 1.0,
            "viscosity_pa_s": 1.0e-3,
            "total_steps": TOTAL_STEPS,
            "checkpoint_step": CHECKPOINT_STEP,
            "wall_speed_mps": MOVING_WALL_SPEED_MPS,
            "wall_segment_scales": MOVING_WALL_SEGMENT_SCALES,
            "pressure_solver": "fv_cg",
            "pressure_iterations": 96,
            "pressure_tolerance": 1.0e-6,
            "pressure_preconditioner": "jacobi",
        },
        sources={
            runner_path.relative_to(REPO_ROOT).as_posix(): runner_path.read_bytes(),
            solver_path.relative_to(REPO_ROOT).as_posix(): solver_path.read_bytes(),
        },
        geometry={
            "obstacle": obstacle,
            "active_component_mask": ledger[
                "velocity_dirichlet_boundary_active_component_mask"
            ],
            "owned_component_mask": ledger[
                "velocity_dirichlet_boundary_owned_component_mask"
            ],
        },
    )


def _capture_snapshot_fields(
    solver: CartesianFluidSolver,
) -> dict[str, np.ndarray]:
    solver._require_velocity_dirichlet_component_ledger_sealed()
    fields = {
        name: np.asarray(getattr(solver, name).to_numpy()).copy()
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES
    }
    return validate_preflow_snapshot_fields(
        fields,
        velocity_dirichlet_boundary_authority="canonical",
    )


def _restore_snapshot_fields(
    solver: CartesianFluidSolver,
    snapshot: PreflowSnapshot,
) -> None:
    """Commit a validated schema-v8 payload, then rebuild derived consumers."""

    fields = validate_preflow_snapshot_fields(
        snapshot.fields,
        velocity_dirichlet_boundary_authority="canonical",
    )
    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        getattr(solver, name).from_numpy(fields[name])
    for mirror_name in ("pressure_accum", "pressure_tmp"):
        mirror = getattr(solver, mirror_name, None)
        if mirror is not None:
            mirror.from_numpy(fields["pressure"])

    solver.velocity_dirichlet_component_ledger_generation = int(
        snapshot.velocity_dirichlet_component_ledger_generation
    )
    solver.velocity_dirichlet_component_ledger_sealed = False
    solver._velocity_dirichlet_component_ledger_consumer_generations = {}
    solver._velocity_dirichlet_component_ledger_consumer_capabilities = {}
    solver._hibm_base_obstacle_initialized = True
    solver.hibm_dynamic_solid_volume_enabled = bool(
        np.any(fields["hibm_dynamic_solid_volume_obstacle"])
    )
    solver._sst_wall_distance_valid = True
    solver._invalidate_hibm_pressure_reachability()
    _prepare_and_seal_canonical_ledger(solver)


def _state_sha256(fields: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _nonpressure_state_sha256(fields: Mapping[str, np.ndarray]) -> str:
    return _state_sha256(
        {
            name: values
            for name, values in fields.items()
            if name not in PRESSURE_STATE_FIELDS
        }
    )


def _pressure_replay_tolerance_pa(
    expected: np.ndarray | float,
    actual: np.ndarray | float,
) -> tuple[float, float]:
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    scale_pa = max(
        float(np.max(np.abs(expected_array), initial=0.0)),
        float(np.max(np.abs(actual_array), initial=0.0)),
        float(np.finfo(np.float64).tiny),
    )
    tolerance_pa = (
        PRESSURE_REPLAY_ROUNDOFF_FACTOR
        * float(np.finfo(np.float64).eps)
        * scale_pa
    )
    return scale_pa, tolerance_pa


def _pressure_replay_comparison(
    *, expected: np.ndarray, actual: np.ndarray
) -> dict[str, float]:
    expected_f64 = np.asarray(expected, dtype=np.float64)
    actual_f64 = np.asarray(actual, dtype=np.float64)
    delta = actual_f64 - expected_f64
    scale_pa, tolerance_pa = _pressure_replay_tolerance_pa(
        expected_f64, actual_f64
    )
    centered_expected = expected_f64 - float(np.mean(expected_f64))
    centered_actual = actual_f64 - float(np.mean(actual_f64))
    return {
        "scale_pa": scale_pa,
        "tolerance_pa": tolerance_pa,
        "linf_difference_pa": float(np.max(np.abs(delta), initial=0.0)),
        "rms_difference_pa": float(np.sqrt(np.mean(delta * delta))),
        "mean_offset_pa": float(np.mean(delta)),
        "centered_linf_difference_pa": float(
            np.max(np.abs(centered_actual - centered_expected), initial=0.0)
        ),
    }


def _array_bitwise_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    expected_contiguous = np.ascontiguousarray(expected)
    actual_contiguous = np.ascontiguousarray(actual)
    return (
        expected_contiguous.dtype == actual_contiguous.dtype
        and expected_contiguous.shape == actual_contiguous.shape
        and expected_contiguous.tobytes(order="C")
        == actual_contiguous.tobytes(order="C")
    )


def _strict_finite(values: np.ndarray, *, label: str, step: int) -> None:
    if not bool(np.all(np.isfinite(values))):
        raise FloatingPointError(f"step {step}: {label} contains non-finite values")


def _advance_one_step(
    solver: CartesianFluidSolver,
    *,
    step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sst = solver.advance_sst_transport(dt_s=DT_S, advection_scheme="muscl_tvd")
    solver.predict(dt_s=DT_S, advection_scheme="muscl_tvd")
    projection = solver.project(
        iterations=96,
        pressure_outlet_zmin=False,
        velocity_inlet_zmax=False,
        dt_s=DT_S,
        preserve_velocity_constraints=True,
        reset_pressure=(step == 1),
        pressure_solver="fv_cg",
        cg_tolerance=1.0e-6,
        cg_preconditioner="jacobi",
        pressure_solve_failure_policy="raise",
        warm_start_slot=-1,
    )

    velocity = solver.velocity.to_numpy()
    pressure = solver.pressure.to_numpy()
    _strict_finite(velocity, label="velocity", step=step)
    _strict_finite(pressure, label="pressure", step=step)
    if int(sst["nonfinite_or_nonpositive_cell_count"]) != 0:
        raise FloatingPointError(f"step {step}: invalid SST state reported")
    if float(sst["turbulent_kinetic_energy_min_m2_s2"]) <= 0.0:
        raise FloatingPointError(f"step {step}: SST k lost positivity")
    if float(sst["specific_dissipation_rate_min_s"]) <= 0.0:
        raise FloatingPointError(f"step {step}: SST omega lost positivity")
    sst_substeps = int(sst["explicit_transport_substeps"])
    if not 1 <= sst_substeps <= MAX_TRANSPORT_SUBSTEPS:
        raise RuntimeError(f"step {step}: invalid SST substep count {sst_substeps}")
    if float(sst["maximum_substep_transport_cfl"]) > MAX_SST_SUBSTEP_CFL:
        raise RuntimeError(f"step {step}: SST substep CFL gate failed")
    momentum_substeps = int(solver._last_momentum_advection_substeps)
    if not 1 <= momentum_substeps <= MAX_TRANSPORT_SUBSTEPS:
        raise RuntimeError(
            f"step {step}: invalid momentum substep count {momentum_substeps}"
        )
    momentum_cfl = float(solver._last_momentum_advection_max_substep_cfl)
    if not math.isfinite(momentum_cfl) or momentum_cfl > MAX_MOMENTUM_SUBSTEP_CFL:
        raise RuntimeError(f"step {step}: momentum substep CFL gate failed")
    if projection.get("cg_converged_all") is not True:
        raise RuntimeError(f"step {step}: FV-CG did not converge")
    if int(projection.get("cg_breakdown_count", -1)) != 0:
        raise RuntimeError(f"step {step}: FV-CG breakdown was reported")
    if projection.get("pressure_solve_failed") is not False:
        raise RuntimeError(f"step {step}: pressure solve failure was reported")
    if projection.get("pressure_projection_physical_failure") is not False:
        raise RuntimeError(f"step {step}: pressure projection physical gate failed")
    return dict(sst), dict(projection)


def _diagnostic_row(
    solver: CartesianFluidSolver,
    *,
    step: int,
    wall_scale: float,
    sst: Mapping[str, Any],
    projection: Mapping[str, Any],
    previous_velocity: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fields = _capture_snapshot_fields(solver)
    velocity = fields["velocity"]
    pressure = fields["pressure"]
    active = fields["velocity_dirichlet_boundary_active_component_mask"]
    hard = fields["velocity_dirichlet_boundary_hard_fixed_component_mask"]
    owned = fields["velocity_dirichlet_boundary_owned_component_mask"]
    stationarity_delta = (
        None
        if previous_velocity is None
        else float(np.sqrt(np.mean((velocity - previous_velocity) ** 2)))
    )
    row = {
        "step": int(step),
        "time_s": float(step * DT_S),
        "wall_scale": float(wall_scale),
        "wall_speed_mps": float(MOVING_WALL_SPEED_MPS * wall_scale),
        "schema_version": int(PREFLOW_SNAPSHOT_SCHEMA_VERSION),
        "snapshot_field_count": len(PREFLOW_SNAPSHOT_FIELD_NAMES),
        "state_sha256": _state_sha256(fields),
        "velocity_max_abs_mps": float(np.max(np.abs(velocity))),
        "pressure_min_pa": float(np.min(pressure)),
        "pressure_max_pa": float(np.max(pressure)),
        "pressure_range_pa": float(np.max(pressure) - np.min(pressure)),
        "stationarity_velocity_rms_delta_mps": stationarity_delta,
        "canonical_generation": int(
            solver.velocity_dirichlet_component_ledger_generation
        ),
        "canonical_sealed": bool(solver.velocity_dirichlet_component_ledger_sealed),
        "canonical_active_component_count": _bit_count(active),
        "canonical_hard_component_count": _bit_count(hard),
        "canonical_owned_component_count": _bit_count(owned),
        "sst_transport_substeps": int(sst["explicit_transport_substeps"]),
        "sst_maximum_substep_cfl": float(
            sst["maximum_substep_transport_cfl"]
        ),
        "sst_k_min_m2_s2": float(sst["turbulent_kinetic_energy_min_m2_s2"]),
        "sst_omega_min_s": float(sst["specific_dissipation_rate_min_s"]),
        "momentum_transport_substeps": int(
            solver._last_momentum_advection_substeps
        ),
        "momentum_maximum_substep_cfl": float(
            solver._last_momentum_advection_max_substep_cfl
        ),
        "cg_converged_all": bool(projection["cg_converged_all"]),
        "cg_breakdown_count": int(projection["cg_breakdown_count"]),
        "pressure_solve_failed": bool(projection["pressure_solve_failed"]),
        "pressure_projection_physical_failure": bool(
            projection["pressure_projection_physical_failure"]
        ),
    }
    return row, fields


def _save_schema_v8_checkpoint(
    *,
    output_dir: Path,
    fields: Mapping[str, np.ndarray],
    identity: PreflowSnapshotIdentity,
    solver: CartesianFluidSolver,
    diagnostic: Mapping[str, Any],
):
    snapshot = PreflowSnapshot(
        fields=fields,
        identity=identity,
        history={
            "purpose": "cartesian_fluid_sst_canonical_5000_replay_checkpoint",
            "completed_step": CHECKPOINT_STEP,
            "diagnostic": dict(diagnostic),
        },
        velocity_dirichlet_boundary_authority="canonical",
        velocity_dirichlet_component_ledger_generation=int(
            solver.velocity_dirichlet_component_ledger_generation
        ),
    )
    files = save_preflow_snapshot(
        output_dir / f"checkpoint_step_{CHECKPOINT_STEP:06d}",
        snapshot,
    )
    manifest = json.loads(files.metadata_path.read_text(encoding="utf-8"))
    if int(manifest["schema_version"]) != PREFLOW_SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError("checkpoint writer did not emit schema v8")
    return files


def _assert_replay_fields_physically_equivalent(
    *,
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    step: int,
) -> dict[str, Any]:
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"checkpoint replay field set mismatch at step {step}: "
            f"missing={missing}, extra={extra}"
        )

    non_bitwise_fields: list[str] = []
    pressure_comparisons: dict[str, dict[str, float]] = {}
    for name in sorted(expected):
        expected_values = np.asarray(expected[name])
        actual_values = np.asarray(actual[name])
        if expected_values.shape != actual_values.shape:
            raise RuntimeError(
                f"checkpoint replay field shape mismatch at step {step}: "
                f"field={name}, expected={expected_values.shape}, "
                f"actual={actual_values.shape}"
            )
        if expected_values.dtype != actual_values.dtype:
            raise RuntimeError(
                f"checkpoint replay field dtype mismatch at step {step}: "
                f"field={name}, expected={expected_values.dtype}, "
                f"actual={actual_values.dtype}"
            )
        if np.issubdtype(expected_values.dtype, np.floating):
            if not bool(np.all(np.isfinite(expected_values))) or not bool(
                np.all(np.isfinite(actual_values))
            ):
                raise RuntimeError(
                    f"checkpoint replay field is non-finite at step {step}: "
                    f"field={name}"
                )

        bitwise_equal = _array_bitwise_equal(expected_values, actual_values)
        if name in PRESSURE_STATE_FIELDS:
            comparison = _pressure_replay_comparison(
                expected=expected_values,
                actual=actual_values,
            )
            pressure_comparisons[name] = comparison
            if bitwise_equal:
                continue
            non_bitwise_fields.append(name)
            if comparison["linf_difference_pa"] > comparison["tolerance_pa"]:
                raise RuntimeError(
                    f"checkpoint replay pressure mismatch at step {step}: "
                    f"field={name}, linf_pa={comparison['linf_difference_pa']}, "
                    f"tolerance_pa={comparison['tolerance_pa']}, "
                    f"scale_pa={comparison['scale_pa']}"
                )
            continue

        if not bitwise_equal:
            non_bitwise_fields.append(name)
            delta = (
                actual_values.astype(np.float64)
                - expected_values.astype(np.float64)
                if np.issubdtype(expected_values.dtype, np.number)
                else None
            )
            linf = (
                float(np.max(np.abs(delta), initial=0.0))
                if delta is not None
                else None
            )
            raise RuntimeError(
                f"checkpoint replay non-pressure field mismatch at step {step}: "
                f"field={name}, linf={linf}"
            )

    return {
        "step": int(step),
        "bitwise_identical": not non_bitwise_fields,
        "non_bitwise_fields": non_bitwise_fields,
        "nonpressure_state_sha256_identical": (
            _nonpressure_state_sha256(expected)
            == _nonpressure_state_sha256(actual)
        ),
        **pressure_comparisons,
    }


def _assert_replay_diagnostic_equivalent(
    *, expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "checkpoint replay diagnostic field set mismatch at "
            f"step {actual.get('step')}: missing={missing}, extra={extra}"
        )

    tolerated_pressure_diagnostics: list[str] = []
    pressure_comparisons: dict[str, dict[str, float]] = {}
    differing: list[str] = []
    diagnostic_pressure_scale_pa, diagnostic_pressure_tolerance_pa = (
        _pressure_replay_tolerance_pa(
            np.asarray(
                [expected[key] for key in PRESSURE_DIAGNOSTIC_FIELDS],
                dtype=np.float64,
            ),
            np.asarray(
                [actual[key] for key in PRESSURE_DIAGNOSTIC_FIELDS],
                dtype=np.float64,
            ),
        )
        if PRESSURE_DIAGNOSTIC_FIELDS.issubset(expected)
        else (0.0, 0.0)
    )
    for key in sorted(expected):
        expected_value = expected[key]
        actual_value = actual[key]
        if expected_value == actual_value:
            continue
        if key == "state_sha256":
            continue
        if key in PRESSURE_DIAGNOSTIC_FIELDS:
            difference_pa = abs(float(actual_value) - float(expected_value))
            pressure_comparisons[key] = {
                "expected_pa": float(expected_value),
                "actual_pa": float(actual_value),
                "absolute_difference_pa": difference_pa,
                "scale_pa": diagnostic_pressure_scale_pa,
                "tolerance_pa": diagnostic_pressure_tolerance_pa,
            }
            if difference_pa <= diagnostic_pressure_tolerance_pa:
                tolerated_pressure_diagnostics.append(key)
                continue
        differing.append(key)

    if differing:
        raise RuntimeError(
            "checkpoint replay diagnostic mismatch at "
            f"step {actual.get('step')}: fields={differing}"
        )
    return {
        "step": int(actual["step"]),
        "state_sha256_identical": (
            actual.get("state_sha256") == expected.get("state_sha256")
        ),
        "tolerated_pressure_diagnostics": tolerated_pressure_diagnostics,
        "pressure_diagnostics": pressure_comparisons,
    }


def _emit_progress(*, phase: str, diagnostic: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {
                "phase": phase,
                "step": diagnostic["step"],
                "state_sha256": diagnostic["state_sha256"],
                "velocity_max_abs_mps": diagnostic["velocity_max_abs_mps"],
                "pressure_range_pa": diagnostic["pressure_range_pa"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_impl(config: SoakConfig) -> dict[str, Any]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "soak_report.json"
    identity = _snapshot_identity()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running_primary",
        "contract": "single_solver_1_through_5000_then_fresh_replay_2501_through_5000",
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "requested_primary_steps": TOTAL_STEPS,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_schema_version": PREFLOW_SNAPSHOT_SCHEMA_VERSION,
        "replay_field_check_interval_steps": DIAGNOSTIC_INTERVAL,
        "pressure_replay_roundoff_factor_times_float64_eps": (
            PRESSURE_REPLAY_ROUNDOFF_FACTOR
        ),
        "primary_completed_steps": 0,
        "replay_completed_steps": 0,
        "primary_diagnostics": [],
        "replay_diagnostics": [],
        "replay_equivalence_diagnostics": [],
    }
    _atomic_write_json(report_path, report)

    # Primary durability contract: one solver, one uninterrupted 1..5000 loop.
    primary_solver = _build_solver(arch=config.arch, initial_step=1)
    primary_previous_velocity: np.ndarray | None = None
    primary_diagnostics_by_step: dict[int, dict[str, Any]] = {}
    primary_fields_by_step: dict[int, dict[str, np.ndarray]] = {}
    checkpoint_files = None
    checkpoint_fields: dict[str, np.ndarray] | None = None
    current_scale = _wall_scale_for_step(1)
    last_sst: dict[str, Any] = {}
    last_projection: dict[str, Any] = {}
    for step in range(1, TOTAL_STEPS + 1):
        requested_scale = _wall_scale_for_step(step)
        if requested_scale != current_scale:
            _publish_wall_scale(
                primary_solver,
                wall_scale=requested_scale,
                replace_complete_ledger=False,
            )
            current_scale = requested_scale
        last_sst, last_projection = _advance_one_step(primary_solver, step=step)
        report["primary_completed_steps"] = step

        if step % DIAGNOSTIC_INTERVAL == 0:
            diagnostic, fields = _diagnostic_row(
                primary_solver,
                step=step,
                wall_scale=current_scale,
                sst=last_sst,
                projection=last_projection,
                previous_velocity=primary_previous_velocity,
            )
            primary_previous_velocity = fields["velocity"].copy()
            primary_diagnostics_by_step[step] = diagnostic
            primary_fields_by_step[step] = fields
            report["primary_diagnostics"].append(diagnostic)
            if step == CHECKPOINT_STEP:
                checkpoint_fields = fields
                checkpoint_files = _save_schema_v8_checkpoint(
                    output_dir=output_dir,
                    fields=fields,
                    identity=identity,
                    solver=primary_solver,
                    diagnostic=diagnostic,
                )
                report["checkpoint_metadata_path"] = str(
                    checkpoint_files.metadata_path
                )
                report["checkpoint_npz_path"] = str(checkpoint_files.npz_path)
            _atomic_write_json(report_path, report)
            _emit_progress(phase="primary", diagnostic=diagnostic)

    if checkpoint_files is None or checkpoint_fields is None:
        raise RuntimeError("primary run finished without its step-2500 checkpoint")
    primary_final_fields = _capture_snapshot_fields(primary_solver)
    primary_final_sha256 = _state_sha256(primary_final_fields)
    report["primary_final_state_sha256"] = primary_final_sha256
    report["primary_final_nonpressure_state_sha256"] = (
        _nonpressure_state_sha256(primary_final_fields)
    )
    report["status"] = "running_fresh_replay"
    _atomic_write_json(report_path, report)

    # Replay is deliberately created only after the uninterrupted primary run.
    fresh_solver = _build_solver(arch=config.arch, initial_step=CHECKPOINT_STEP)
    loaded = load_preflow_snapshot(
        checkpoint_files,
        expected_identity=identity,
        expected_velocity_dirichlet_boundary_authority="canonical",
    )
    _restore_snapshot_fields(fresh_solver, loaded)
    restored_fields = _capture_snapshot_fields(fresh_solver)
    if _state_sha256(restored_fields) != _state_sha256(checkpoint_fields):
        raise RuntimeError("fresh solver does not exactly match the step-2500 checkpoint")
    report["checkpoint_restore_bitwise_identical"] = True
    _atomic_write_json(report_path, report)

    replay_previous_velocity = restored_fields["velocity"].copy()
    replay_scale = _wall_scale_for_step(CHECKPOINT_STEP)
    for step in range(CHECKPOINT_STEP + 1, TOTAL_STEPS + 1):
        requested_scale = _wall_scale_for_step(step)
        if requested_scale != replay_scale:
            _publish_wall_scale(
                fresh_solver,
                wall_scale=requested_scale,
                replace_complete_ledger=False,
            )
            replay_scale = requested_scale
        replay_sst, replay_projection = _advance_one_step(fresh_solver, step=step)
        report["replay_completed_steps"] = step - CHECKPOINT_STEP

        if step % DIAGNOSTIC_INTERVAL == 0:
            diagnostic, fields = _diagnostic_row(
                fresh_solver,
                step=step,
                wall_scale=replay_scale,
                sst=replay_sst,
                projection=replay_projection,
                previous_velocity=replay_previous_velocity,
            )
            replay_previous_velocity = fields["velocity"].copy()
            # Persist completed progress and the actual diagnostic before a
            # strict equivalence gate can raise.  A failed replay must never
            # leave the live report claiming that zero replay steps ran.
            report["replay_diagnostics"].append(diagnostic)
            _atomic_write_json(report_path, report)
            field_equivalence = _assert_replay_fields_physically_equivalent(
                expected=primary_fields_by_step[step],
                actual=fields,
                step=step,
            )
            diagnostic_equivalence = _assert_replay_diagnostic_equivalent(
                expected=primary_diagnostics_by_step[step],
                actual=diagnostic,
            )
            report["replay_equivalence_diagnostics"].append(
                {
                    "step": step,
                    "field_equivalence": field_equivalence,
                    "diagnostic_equivalence": diagnostic_equivalence,
                }
            )
            _atomic_write_json(report_path, report)
            _emit_progress(phase="fresh_replay", diagnostic=diagnostic)

    replay_final_fields = _capture_snapshot_fields(fresh_solver)
    replay_final_sha256 = _state_sha256(replay_final_fields)
    final_equivalence = _assert_replay_fields_physically_equivalent(
        expected=primary_final_fields,
        actual=replay_final_fields,
        step=TOTAL_STEPS,
    )
    replay_final_nonpressure_sha256 = _nonpressure_state_sha256(
        replay_final_fields
    )
    checked_steps = [
        int(entry["step"]) for entry in report["replay_equivalence_diagnostics"]
    ]
    pressure_tolerance_ratios: list[float] = []
    for entry in report["replay_equivalence_diagnostics"]:
        field_equivalence = entry["field_equivalence"]
        for name in PRESSURE_STATE_FIELDS:
            comparison = field_equivalence[name]
            pressure_tolerance_ratios.append(
                float(comparison["linf_difference_pa"])
                / float(comparison["tolerance_pa"])
            )
    for name in PRESSURE_STATE_FIELDS:
        comparison = final_equivalence[name]
        pressure_tolerance_ratios.append(
            float(comparison["linf_difference_pa"])
            / float(comparison["tolerance_pa"])
        )
    all_nonpressure_sample_hashes_identical = all(
        bool(entry["field_equivalence"]["nonpressure_state_sha256_identical"])
        for entry in report["replay_equivalence_diagnostics"]
    )
    final_nonpressure_bitwise_identical = (
        replay_final_nonpressure_sha256
        == report["primary_final_nonpressure_state_sha256"]
    )
    report.update(
        {
            "status": "completed",
            "primary_completed_steps": TOTAL_STEPS,
            "replay_completed_steps": TOTAL_STEPS - CHECKPOINT_STEP,
            "replay_final_state_sha256": replay_final_sha256,
            "replay_final_nonpressure_state_sha256": (
                replay_final_nonpressure_sha256
            ),
            "checkpoint_replay_bitwise_identical": (
                replay_final_sha256 == primary_final_sha256
            ),
            "checkpoint_replay_final_state_bitwise_identical": (
                replay_final_sha256 == primary_final_sha256
            ),
            "replay_field_checked_steps": checked_steps,
            "checkpoint_replay_nonpressure_bitwise_identical_at_sampled_and_final_states": (
                all_nonpressure_sample_hashes_identical
                and final_nonpressure_bitwise_identical
            ),
            "checkpoint_replay_pressure_machine_precision_equivalent_at_sampled_and_final_states": True,
            "checkpoint_replay_sampled_and_final_physically_equivalent": True,
            "maximum_pressure_linf_to_tolerance_ratio": float(
                max(pressure_tolerance_ratios, default=0.0)
            ),
            "final_replay_equivalence": final_equivalence,
            "elapsed_s": float(time.perf_counter() - started),
        }
    )
    _atomic_write_json(report_path, report)
    return report


def run(config: SoakConfig) -> dict[str, Any]:
    """Run the soak while keeping live and failure artifacts transactional."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "soak_report.json"
    failure_path = output_dir / "soak_failure.json"
    # A reused output directory must not expose the previous run's failure
    # beside the current run's live state.  Publish this run's ownership before
    # identity construction/JIT so even an initialization failure cannot
    # mutate a stale report from another invocation.
    if failure_path.exists():
        failure_path.unlink()
    _atomic_write_json(
        report_path,
        {
            "status": "initializing",
            "config": {**asdict(config), "output_dir": str(output_dir)},
            "primary_completed_steps": 0,
            "replay_completed_steps": 0,
        },
    )
    try:
        report = _run_impl(config)
    except BaseException as exc:
        if report_path.exists():
            try:
                live_report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                live_report = {}
        else:
            live_report = {}
        failure_phase = str(live_report.get("status", "initializing"))
        failure = {
            "status": "failed",
            "failure_phase": failure_phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "primary_completed_steps": int(
                live_report.get("primary_completed_steps", 0)
            ),
            "replay_completed_steps": int(
                live_report.get("replay_completed_steps", 0)
            ),
        }
        live_report.update(failure)
        _atomic_write_json(report_path, live_report)
        _atomic_write_json(failure_path, failure)
        raise
    if failure_path.exists():
        failure_path.unlink()
    return report


def main(argv: list[str] | None = None) -> int:
    config = config_from_cli(argv)
    report = run(config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "primary_completed_steps": report["primary_completed_steps"],
                "replay_completed_steps": report["replay_completed_steps"],
                "checkpoint_replay_bitwise_identical": report[
                    "checkpoint_replay_bitwise_identical"
                ],
                "checkpoint_replay_sampled_and_final_physically_equivalent": report[
                    "checkpoint_replay_sampled_and_final_physically_equivalent"
                ],
                "output_dir": str(config.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
