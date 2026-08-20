"""Strict input contracts for the native-fine Fluent comparison campaign."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .native_fine_final_contracts import (
    FINAL_PROJECTION_REQUIRED_KEYS,
    NativeFineFinalContractError,
    validate_final_projection_success as _validate_final_projection_success_detail,
    validate_final_run_identity as _validate_final_run_identity_detail,
    validate_native_source_pair_identity as _validate_native_source_pair_identity_detail,
)
from .native_fine_residual_contracts import (
    NativeFineResidualContractError,
    validate_fluent_residual_histories as _validate_fluent_residual_histories_detail,
)

NATIVE_FLUENT_SCHEMA = "fluent_fine_fsi_offline_postprocess_v1"
DEFAULT_EXPECTED_STEPS = 50
DEFAULT_DT_S = 5.0e-4
VELOCITY_VMIN_MPS = 0.0
VELOCITY_VMAX_MPS = 31.0
FRAME_RE = re.compile(r"^step_(?P<step>\d{4})\.npz$")
STEP_HISTORY_RE = re.compile(r"^step_(?P<step>\d{4})\.json$")
FORBIDDEN_REFERENCE_TOKENS = (
    "official_fluent_fine_mesh_steady_2026-07-01",
    "coarse_before_adapt",
    "fine_mesh_after_adapt",
    "fsi_50step_serial_from_adapt_cycle3_mesh",
    "official_fluent_2way_reference",
    "our_solver_fine_vs_fluent_2026-07-02",
    "puma",
)
CANONICAL_NATIVE_FLUENT_POSTPROCESS_RELATIVE_DIR = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "official_fluent_fine_fsi_valid_2026-07-10"
    / "runs"
    / "fresh50_20260713_104843"
    / "postprocess_compare31_strict_pressure_20260719_142808_r2"
)
CANONICAL_NATIVE_FLUENT_PATH_MARKERS = (
    "official_fluent_fine_fsi_valid_2026-07-10",
    "fresh50_20260713_104843",
    "postprocess_compare31_strict_pressure_20260719_142808_r2",
)

FLATTENED_STEP_HISTORY_REQUIRED_KEYS = (
    "step",
    "tip_mean_displacement_m",
    "max_displacement_m",
    "local_velocity_peak_mps",
    "flow_projection_pressure_solve_failed",
    "flow_projection_cg_converged_all",
    "flow_projection_cg_relative_residual_max",
    "flow_projection_cg_project_calls",
    "flow_projection_pre_projection_l2",
    "flow_projection_post_boundary_l2",
    "flow_projection_projection_l2",
    "flow_projection_pressure_solver_requested",
    "flow_projection_pressure_solver",
    "flow_projection_report",
    "no_slip_projected_residual_after_projection_mps",
    "total_marker_count",
    "marker_action_reaction_residual_N",
    "max_abs_traction_pa",
    "mpm_max_speed_mps",
)
FLATTENED_PROJECTION_REPORT_REQUIRED_KEYS = (
    "pressure_solve_failed",
    "cg_converged_all",
    "cg_relative_residual_max",
    "pre_projection_l2",
    "post_boundary_l2",
    "projection_l2",
    "pressure_solver_requested",
    "pressure_solver",
)
_FLATTENED_BOOLEAN_KEYS = (
    "flow_projection_pressure_solve_failed",
    "flow_projection_cg_converged_all",
)
_FLATTENED_STRING_KEYS = (
    "flow_projection_pressure_solver_requested",
    "flow_projection_pressure_solver",
)
_FLATTENED_NONNEGATIVE_NUMERIC_KEYS = (
    "max_displacement_m",
    "local_velocity_peak_mps",
    "flow_projection_cg_relative_residual_max",
    "flow_projection_cg_project_calls",
    "flow_projection_pre_projection_l2",
    "flow_projection_post_boundary_l2",
    "flow_projection_projection_l2",
    "no_slip_projected_residual_after_projection_mps",
    "total_marker_count",
    "marker_action_reaction_residual_N",
    "max_abs_traction_pa",
    "mpm_max_speed_mps",
)
_PROJECTION_REPORT_MIRRORS = {
    "flow_projection_pressure_solve_failed": "pressure_solve_failed",
    "flow_projection_cg_converged_all": "cg_converged_all",
    "flow_projection_cg_relative_residual_max": "cg_relative_residual_max",
    "flow_projection_pre_projection_l2": "pre_projection_l2",
    "flow_projection_post_boundary_l2": "post_boundary_l2",
    "flow_projection_projection_l2": "projection_l2",
    "flow_projection_pressure_solver_requested": "pressure_solver_requested",
    "flow_projection_pressure_solver": "pressure_solver",
}


class NativeFineComparisonError(RuntimeError):
    """Raised when a comparison input is incomplete or not the native run."""

def discover_solver_frames(
    our_run_dir: str | Path,
    *,
    expected_steps: int = DEFAULT_EXPECTED_STEPS,
) -> list[Path]:
    fields_dir = Path(our_run_dir) / "step_fields"
    if not fields_dir.is_dir():
        raise NativeFineComparisonError(f"solver step_fields directory is missing: {fields_dir}")
    npz_paths = sorted(fields_dir.glob("*.npz"))
    parsed: list[tuple[int, Path]] = []
    for path in npz_paths:
        match = FRAME_RE.fullmatch(path.name)
        if match is None:
            raise NativeFineComparisonError(f"unexpected solver frame name: {path.name}")
        parsed.append((int(match.group("step")), path))
    expected = list(range(1, expected_steps + 1))
    actual = [step for step, _ in parsed]
    if actual != expected:
        raise NativeFineComparisonError(
            f"exact solver frame sequence required: expected={expected}, actual={actual}"
        )
    return [path for _, path in parsed]

def discover_solver_step_histories(
    our_run_dir: str | Path,
    *,
    expected_steps: int = DEFAULT_EXPECTED_STEPS,
) -> list[Path]:
    history_dir = Path(our_run_dir) / "step_history"
    if not history_dir.is_dir():
        raise NativeFineComparisonError(
            f"solver step_history directory is missing: {history_dir}"
        )
    json_paths = sorted(history_dir.glob("*.json"))
    parsed: list[tuple[int, Path]] = []
    for path in json_paths:
        match = STEP_HISTORY_RE.fullmatch(path.name)
        if match is None:
            raise NativeFineComparisonError(
                f"unexpected solver step history name: {path.name}"
            )
        parsed.append((int(match.group("step")), path))
    expected = list(range(1, expected_steps + 1))
    actual = [step for step, _ in parsed]
    if actual != expected:
        raise NativeFineComparisonError(
            "exact solver step-history sequence required: "
            f"expected={expected}, actual={actual}"
        )
    return [path for _, path in parsed]


def validate_solver_step_histories(
    step_history_paths: Sequence[Path],
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    expected_steps: int,
    dt_s: float,
    require_numerical_success: bool = False,
) -> dict[str, Any]:
    """Cross-check every full per-step JSON payload against the aggregate CSV."""

    if len(step_history_paths) != expected_steps:
        raise NativeFineComparisonError(
            "per-step history count does not match the required step count"
        )
    aggregate_time_cross_check = _validate_exact_history(
        aggregate_rows,
        "our-solver aggregate history",
        expected_steps,
        dt_s,
        allow_missing_time=True,
    )
    flattened_diagnostic_payload_count = 0
    pressure_solve_failure_count = 0
    cg_nonconverged_step_count = 0
    for step, (path, aggregate_row) in enumerate(
        zip(step_history_paths, aggregate_rows, strict=True),
        start=1,
    ):
        payload = _read_json(path)
        if int(payload.get("step_index", -1)) != step:
            raise NativeFineComparisonError(
                f"per-step history index mismatch in {path.name}"
            )
        time_s = _finite_float(
            payload.get("time_s"),
            f"per-step history time in {path.name}",
        )
        expected_time_s = step * dt_s
        if not math.isclose(time_s, expected_time_s, rel_tol=0.0, abs_tol=1.0e-12):
            raise NativeFineComparisonError(
                f"per-step history time mismatch in {path.name}: "
                f"{time_s} vs {expected_time_s}"
            )
        history = payload.get("history")
        if not isinstance(history, Mapping):
            raise NativeFineComparisonError(
                f"per-step history payload is missing in {path.name}"
            )
        _validate_flattened_step_history(history, step=step, path=path)
        pressure_solve_failed = history["flow_projection_pressure_solve_failed"]
        cg_converged_all = history["flow_projection_cg_converged_all"]
        pressure_solve_failure_count += int(pressure_solve_failed)
        cg_nonconverged_step_count += int(not cg_converged_all)
        if require_numerical_success and (
            pressure_solve_failed or not cg_converged_all
        ):
            raise NativeFineComparisonError(
                "final 50-step numerical success requires "
                "flow_projection_pressure_solve_failed=False and "
                "flow_projection_cg_converged_all=True in every frame; "
                f"failed in {path.name}"
            )
        if require_numerical_success:
            _validate_final_projection_success(history, path=path)
        json_max = _finite_float(
            history.get("max_displacement_m"),
            f"per-step maximum displacement in {path.name}",
        )
        csv_max = _finite_float(
            aggregate_row.get("max_displacement_m"),
            f"aggregate maximum displacement at step {step}",
        )
        if not math.isclose(json_max, csv_max, rel_tol=0.0, abs_tol=1.0e-15):
            raise NativeFineComparisonError(
                f"per-step and aggregate maximum displacement disagree at step {step}"
            )
        json_tip = _vector(history.get("tip_mean_displacement_m"))
        csv_tip = _vector(aggregate_row.get("tip_mean_displacement_m"))
        if len(json_tip) < 3 or len(csv_tip) < 3 or not np.allclose(
            np.asarray(json_tip[:3], dtype=np.float64),
            np.asarray(csv_tip[:3], dtype=np.float64),
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise NativeFineComparisonError(
                f"per-step and aggregate tip displacement disagree at step {step}"
            )
        flattened_diagnostic_payload_count += 1
    return {
        "schema": "our_solver_flattened_step_history_v1",
        "status": "passed",
        "step_count": expected_steps,
        "history_layout": "flattened_diagnostics",
        "aggregate_csv_cross_check": "passed",
        "flattened_diagnostic_payload_count": flattened_diagnostic_payload_count,
        "projection_report_cross_check": "passed",
        "time_source": "step_history_wrapper",
        "aggregate_time_cross_check": aggregate_time_cross_check,
        "numerical_success_required": require_numerical_success,
        "pressure_solve_failure_count": pressure_solve_failure_count,
        "cg_nonconverged_step_count": cg_nonconverged_step_count,
        "required_final_projection_keys": (
            list(FINAL_PROJECTION_REQUIRED_KEYS)
            if require_numerical_success
            else []
        ),
        "required_flattened_keys": list(FLATTENED_STEP_HISTORY_REQUIRED_KEYS),
        "required_projection_report_keys": list(
            FLATTENED_PROJECTION_REPORT_REQUIRED_KEYS
        ),
        "json_reader": "python_json_case_sensitive",
        "case_distinct_keys_preserved_if_present": True,
    }


def validate_final_solver_step_histories(
    step_history_paths: Sequence[Path],
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    expected_steps: int,
    dt_s: float,
) -> dict[str, Any]:
    """Validate the exact 50-step publishable history, including solver success."""

    if expected_steps != DEFAULT_EXPECTED_STEPS:
        raise NativeFineComparisonError(
            "final solver history contract requires exactly 50 steps"
        )
    contract = validate_solver_step_histories(
        step_history_paths, aggregate_rows, expected_steps=expected_steps,
        dt_s=dt_s, require_numerical_success=True,
    )
    return {**contract, "contract_role": "final_exact_50_step"}

def validate_partial_diagnostic_step_histories(
    step_history_paths: Sequence[Path],
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    expected_steps: int,
    dt_s: float,
) -> dict[str, Any]:
    """Validate incomplete/failed baselines without claiming numerical success."""

    contract = validate_solver_step_histories(
        step_history_paths, aggregate_rows, expected_steps=expected_steps,
        dt_s=dt_s, require_numerical_success=False,
    )
    return {**contract, "contract_role": "partial_diagnostic"}


def _validate_flattened_step_history(
    history: Mapping[str, Any],
    *,
    step: int,
    path: Path,
) -> None:
    missing = sorted(set(FLATTENED_STEP_HISTORY_REQUIRED_KEYS) - set(history))
    if missing:
        raise NativeFineComparisonError(
            f"flattened diagnostic keys are missing in {path.name}: {missing}"
        )
    if isinstance(history["step"], bool) or not isinstance(history["step"], int):
        raise NativeFineComparisonError(
            f"flattened history step is invalid in {path.name}"
        )
    history_step = history["step"]
    if history_step != step:
        raise NativeFineComparisonError(
            f"flattened history step mismatch in {path.name}: {history_step} vs {step}"
        )

    for key in _FLATTENED_BOOLEAN_KEYS:
        if not isinstance(history[key], bool):
            raise NativeFineComparisonError(
                f"flattened diagnostic {key} must be boolean in {path.name}"
            )
    for key in _FLATTENED_STRING_KEYS:
        if not isinstance(history[key], str) or not history[key].strip():
            raise NativeFineComparisonError(
                f"flattened diagnostic {key} must be a non-empty string in {path.name}"
            )
    for key in _FLATTENED_NONNEGATIVE_NUMERIC_KEYS:
        if isinstance(history[key], bool) or not isinstance(history[key], (int, float)):
            raise NativeFineComparisonError(
                f"flattened diagnostic {key} must be numeric in {path.name}"
            )
        value = _finite_float(history[key], f"flattened diagnostic {key} in {path.name}")
        if value < 0.0:
            raise NativeFineComparisonError(
                f"flattened diagnostic {key} must be nonnegative in {path.name}"
            )
    if not float(history["flow_projection_cg_project_calls"]).is_integer() or int(
        history["flow_projection_cg_project_calls"]
    ) <= 0:
        raise NativeFineComparisonError(
            f"flow_projection_cg_project_calls must be positive in {path.name}"
        )
    if not float(history["total_marker_count"]).is_integer() or int(
        history["total_marker_count"]
    ) <= 0:
        raise NativeFineComparisonError(
            f"total_marker_count must be positive in {path.name}"
        )

    projection_report = history["flow_projection_report"]
    if not isinstance(projection_report, Mapping):
        raise NativeFineComparisonError(
            f"flow_projection_report must be an object in {path.name}"
        )
    missing_report = sorted(
        set(FLATTENED_PROJECTION_REPORT_REQUIRED_KEYS) - set(projection_report)
    )
    if missing_report:
        raise NativeFineComparisonError(
            f"projection report keys are missing in {path.name}: {missing_report}"
        )
    for flattened_key, report_key in _PROJECTION_REPORT_MIRRORS.items():
        if not _diagnostic_values_equal(
            history[flattened_key],
            projection_report[report_key],
        ):
            raise NativeFineComparisonError(
                "projection report mismatch in "
                f"{path.name}: {flattened_key} != flow_projection_report.{report_key}"
            )


def _validate_final_projection_success(
    history: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    try:
        _validate_final_projection_success_detail(history, path=path)
    except NativeFineFinalContractError as exc:
        raise NativeFineComparisonError(str(exc)) from exc


def _diagnostic_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return left == right
    return math.isfinite(left_float) and math.isfinite(right_float) and math.isclose(
        left_float,
        right_float,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )


def _validate_final_run_identity(
    our_manifest: Mapping[str, Any],
    our_summary: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return _validate_final_run_identity_detail(our_manifest, our_summary)
    except NativeFineFinalContractError as exc:
        raise NativeFineComparisonError(str(exc)) from exc


def _validate_run_contracts(
    our_manifest: Mapping[str, Any],
    our_summary: Mapping[str, Any],
    our_progress: Mapping[str, Any],
    fluent_summary: Mapping[str, Any],
    *,
    expected_steps: int,
) -> float:
    config = our_manifest.get("config")
    if not isinstance(config, Mapping):
        raise NativeFineComparisonError("our run manifest has no config mapping")
    if config.get("flow_hibm_sharp_interpolate_velocity_rows") is not False:
        raise NativeFineComparisonError(
            "native-fine comparison requires "
            "flow_hibm_sharp_interpolate_velocity_rows=false for the validated "
            "direct sharp pipeline"
        )
    probe_distance = config.get("flow_hibm_sharp_interior_probe_distance_m")
    if isinstance(probe_distance, bool) or not isinstance(
        probe_distance, (int, float)
    ):
        raise NativeFineComparisonError(
            "native-fine comparison requires a finite positive scalar "
            "flow_hibm_sharp_interior_probe_distance_m"
        )
    if not math.isfinite(float(probe_distance)) or float(probe_distance) <= 0.0:
        raise NativeFineComparisonError(
            "native-fine comparison requires a finite positive scalar "
            "flow_hibm_sharp_interior_probe_distance_m"
        )
    probe_distance_xyz = config.get(
        "flow_hibm_sharp_interior_probe_distance_xyz_m"
    )
    if probe_distance_xyz is not None:
        raise NativeFineComparisonError(
            "native-fine comparison requires the validated scalar probe and "
            "flow_hibm_sharp_interior_probe_distance_xyz_m=None"
        )
    if expected_steps == DEFAULT_EXPECTED_STEPS:
        _validate_final_run_identity(our_manifest, our_summary)
    dt_s = _finite_float(config.get("dt_s", DEFAULT_DT_S), "our run dt")
    if int(config.get("step_count", -1)) != expected_steps:
        raise NativeFineComparisonError("our run manifest step count does not match expected steps")
    if our_manifest.get("save_step_fields") is not True:
        raise NativeFineComparisonError("our run did not enable save_step_fields")
    if our_summary.get("status") != "completed":
        raise NativeFineComparisonError("our solver summary is not completed")
    if int(our_summary.get("step_count_completed", -1)) != expected_steps:
        raise NativeFineComparisonError("our solver summary is not exactly complete")
    if our_progress.get("status") != "completed" or int(
        our_progress.get("step_completed", -1)
    ) != expected_steps:
        raise NativeFineComparisonError("our atomic progress record is not exactly complete")
    step_validation = our_summary.get("step_artifact_validation")
    if not isinstance(step_validation, Mapping):
        raise NativeFineComparisonError(
            "our solver summary has no step_artifact_validation contract"
        )
    if step_validation.get("status") != "passed" or int(
        step_validation.get("frame_count", -1)
    ) != expected_steps or int(step_validation.get("history_count", -1)) != expected_steps:
        raise NativeFineComparisonError(
            "our solver step field/history artifact gate is not exactly complete"
        )
    if int(our_summary.get("step_field_frame_count", -1)) != expected_steps:
        raise NativeFineComparisonError(
            "our solver summary step-field count is not exactly complete"
        )

    if fluent_summary.get("schema") != NATIVE_FLUENT_SCHEMA:
        raise NativeFineComparisonError(
            "input does not use the required native Fluent postprocess schema"
        )
    required_values = {
        "status": "complete",
        "offline_only": True,
        "fluent_launched": False,
        "source_artifacts_modified": False,
        "phase_manifest_status": "passed",
        "all_structure_steps_nonzero": True,
    }
    for key, expected in required_values.items():
        if fluent_summary.get(key) != expected:
            raise NativeFineComparisonError(
                f"native Fluent summary field {key!r} must equal {expected!r}"
            )
    if int(fluent_summary.get("expected_step_count", -1)) != expected_steps or int(
        fluent_summary.get("step_count", -1)
    ) != expected_steps:
        raise NativeFineComparisonError("native Fluent postprocess is not exactly complete")
    fluent_dt = _finite_float(fluent_summary.get("dt_s"), "native Fluent dt")
    if not math.isclose(dt_s, fluent_dt, rel_tol=0.0, abs_tol=1.0e-15):
        raise NativeFineComparisonError(f"time-step mismatch: our={dt_s}, Fluent={fluent_dt}")
    return dt_s


def _validate_native_fluent_bundle(
    fluent_postprocess_dir: Path,
    fluent_summary: Mapping[str, Any],
    fluent_input_manifest: Mapping[str, Any],
    *,
    expected_steps: int,
) -> None:
    """Lock the comparison to native Fluent data and reject adapted/PUMA inputs."""

    _reject_legacy_reference_path(fluent_postprocess_dir)
    _reject_legacy_reference_payload(fluent_summary, label="native Fluent summary")
    _reject_legacy_reference_payload(
        fluent_input_manifest,
        label="native Fluent input manifest",
    )
    if expected_steps == DEFAULT_EXPECTED_STEPS:
        canonical_dir = (
            _repository_root() / CANONICAL_NATIVE_FLUENT_POSTPROCESS_RELATIVE_DIR
        ).resolve()
        actual_dir = fluent_postprocess_dir.resolve()
        if actual_dir != canonical_dir:
            raise NativeFineComparisonError(
                "50-step comparison must use the locked native Fluent bundle at "
                f"{canonical_dir}; got {actual_dir}"
            )

    if fluent_input_manifest.get("schema") != "fluent_fine_fsi_input_pairs_v1":
        raise NativeFineComparisonError(
            "native Fluent input manifest has the wrong schema"
        )
    if int(fluent_input_manifest.get("step_count", -1)) != expected_steps:
        raise NativeFineComparisonError(
            "native Fluent input manifest has the wrong step count"
        )
    pairs = fluent_input_manifest.get("pairs")
    if not isinstance(pairs, list):
        raise NativeFineComparisonError("native Fluent input manifest has no pair list")
    try:
        actual_steps = [int(pair["step"]) for pair in pairs]
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeFineComparisonError(
            "native Fluent input manifest contains invalid step pairs"
        ) from exc
    if actual_steps != list(range(1, expected_steps + 1)):
        raise NativeFineComparisonError(
            "native Fluent input pairs are not exact and contiguous"
        )
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise NativeFineComparisonError(
                "native Fluent input manifest contains a non-object pair"
            )
        for key in ("case_path", "data_path"):
            path_value = pair.get(key)
            if not isinstance(path_value, str) or not path_value:
                raise NativeFineComparisonError(
                    f"native Fluent input pair is missing {key}"
                )
            _reject_legacy_reference_path(Path(path_value))
        try:
            _validate_native_source_pair_identity_detail(pair)
        except NativeFineFinalContractError as exc:
            raise NativeFineComparisonError(str(exc)) from exc

    outputs = fluent_summary.get("outputs")
    expected_outputs = {
        "final_fields_npz": "fields/final_fields.npz",
        "pressure_history_csv": "histories/pressure_history.csv",
        "residual_history_csv": "histories/residual_history.csv",
        "residual_snapshot_summary_csv": (
            "histories/residual_snapshot_summary.csv"
        ),
        "structure_displacement_history_csv": (
            "histories/structure_displacement_history.csv"
        ),
        "velocity_history_csv": "histories/velocity_history.csv",
    }
    if not isinstance(outputs, Mapping):
        raise NativeFineComparisonError("native Fluent summary has no output mapping")
    for key, expected in expected_outputs.items():
        if outputs.get(key) != expected:
            raise NativeFineComparisonError(
                f"native Fluent summary output {key!r} must equal {expected!r}"
            )
    display_range = fluent_summary.get("velocity_display_range_mps")
    if not isinstance(display_range, list) or len(display_range) != 2:
        raise NativeFineComparisonError(
            "native Fluent bundle does not use the locked 0..31 m/s display range"
        )
    display_min = _finite_float(display_range[0], "native Fluent display minimum")
    display_max = _finite_float(display_range[1], "native Fluent display maximum")
    if display_min != VELOCITY_VMIN_MPS or display_max != VELOCITY_VMAX_MPS:
        raise NativeFineComparisonError(
            "native Fluent bundle does not use the locked 0..31 m/s display range"
        )

    _verify_required_fluent_checksums(
        fluent_postprocess_dir / "CHECKSUMS.sha256",
        fluent_postprocess_dir,
        required_relative_paths=(
            "summary.json",
            "input_manifest.json",
            "fields/final_fields.npz",
            "histories/pressure_history.csv",
            "histories/residual_history.csv",
            "histories/residual_snapshot_summary.csv",
            "histories/structure_displacement_history.csv",
            "histories/velocity_history.csv",
        ),
    )


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cases" / "ansys_vertical_flap_fsi.py").is_file() and (
            parent / "src" / "refactored"
        ).is_dir():
            return parent
    raise NativeFineComparisonError(
        "could not locate repository root for locked native Fluent bundle"
    )


def _verify_required_fluent_checksums(
    checksum_path: Path,
    root: Path,
    *,
    required_relative_paths: Sequence[str],
) -> None:
    if not checksum_path.is_file():
        raise NativeFineComparisonError(
            f"native Fluent checksum manifest is missing: {checksum_path}"
        )
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative_path = line.split(None, 1)
        except ValueError as exc:
            raise NativeFineComparisonError(
                f"invalid native Fluent checksum row: {line!r}"
            ) from exc
        entries[relative_path.strip().replace("\\", "/")] = digest.lower()
    for relative_path in required_relative_paths:
        expected = entries.get(relative_path)
        if expected is None:
            raise NativeFineComparisonError(
                f"native Fluent checksum is missing {relative_path}"
            )
        path = root / Path(relative_path)
        if not path.is_file():
            raise NativeFineComparisonError(
                f"checksummed native Fluent input is missing: {path}"
            )
        if sha256_file(path).lower() != expected:
            raise NativeFineComparisonError(
                f"native Fluent checksum mismatch for {relative_path}"
            )


def validate_fluent_residual_histories(
    residual_history_path: str | Path,
    residual_snapshot_summary_path: str | Path,
    *,
    expected_steps: int,
    dt_s: float,
) -> dict[str, Any]:
    try:
        return _validate_fluent_residual_histories_detail(
            residual_history_path,
            residual_snapshot_summary_path,
            expected_steps=expected_steps,
            dt_s=dt_s,
        )
    except NativeFineResidualContractError as exc:
        raise NativeFineComparisonError(str(exc)) from exc


def _validate_exact_history(
    rows: Sequence[Mapping[str, Any]],
    label: str,
    expected_steps: int,
    dt_s: float,
    *,
    allow_missing_time: bool = False,
) -> str:
    expected = list(range(1, expected_steps + 1))
    try:
        actual = [int(row["step"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeFineComparisonError(f"{label} has invalid step values") from exc
    if actual != expected:
        raise NativeFineComparisonError(
            f"{label} steps must be exact and contiguous: expected={expected}, actual={actual}"
        )
    time_presence = [row.get("time_s") not in (None, "") for row in rows]
    if allow_missing_time and any(time_presence) and not all(time_presence):
        raise NativeFineComparisonError(f"{label} has partially missing time values")
    if allow_missing_time and not any(time_presence):
        return "not_present"
    for step, row in zip(expected, rows, strict=True):
        time_s = _finite_float(row.get("time_s"), f"{label} time at step {step}")
        if not math.isclose(time_s, step * dt_s, rel_tol=0.0, abs_tol=1.0e-12):
            raise NativeFineComparisonError(
                f"{label} time mismatch at step {step}: {time_s} vs {step * dt_s}"
            )
    return "passed"


def _reject_legacy_reference_path(path: Path) -> None:
    lowered = path.as_posix().lower()
    for token in FORBIDDEN_REFERENCE_TOKENS:
        if token in lowered:
            raise NativeFineComparisonError(
                f"legacy or adapted Fluent reference is forbidden for this campaign: {path}"
            )


def _reject_legacy_reference_payload(value: Any, *, label: str) -> None:
    for text in _iter_strings(value):
        lowered = text.lower().replace("\\", "/")
        for token in FORBIDDEN_REFERENCE_TOKENS:
            if token in lowered:
                raise NativeFineComparisonError(
                    f"{label} contains forbidden legacy/PUMA reference token {token!r}"
                )


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise NativeFineComparisonError(f"required JSON artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NativeFineComparisonError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NativeFineComparisonError(f"JSON artifact must contain an object: {path}")
    return payload


def _parse_csv_value(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return ""
    if text[0] in "[{" and text[-1] in "]}":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ()
    if not isinstance(value, (list, tuple, np.ndarray)):
        return ()
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return ()
    return vector if all(math.isfinite(component) for component in vector) else ()


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeFineComparisonError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise NativeFineComparisonError(f"{label} is not finite: {result!r}")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
