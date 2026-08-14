"""Offline comparison of our fine-grid flap run to native Fluent output.

The comparison is intentionally diagnostic.  It validates exact 50-step input
contracts, renders our solver's velocity history, and compares fields after
sampling the structured solver grid at Fluent cell centers.  It does not turn
different numerical models or different displacement monitors into a parity
claim.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .official_fluent_parity import (
    PRESSURE_QUANTITY,
    PRESSURE_REFERENCE,
    compare_solver_to_fluent_field,
    load_fluent_npz,
    load_pressure_semantics,
    sample_structured_solver_at_fluent_points,
)
from .native_fine_contracts import (
    CANONICAL_NATIVE_FLUENT_PATH_MARKERS,
    DEFAULT_EXPECTED_STEPS,
    NATIVE_FLUENT_SCHEMA,
    VELOCITY_VMAX_MPS,
    VELOCITY_VMIN_MPS,
    NativeFineComparisonError,
    _finite_float,
    _parse_csv_value,
    _read_json,
    _reject_legacy_reference_path,
    _validate_exact_history,
    _validate_final_run_identity,
    _validate_native_fluent_bundle,
    _validate_run_contracts,
    _vector,
    discover_solver_frames,
    discover_solver_step_histories,
    sha256_file,
    validate_final_solver_step_histories,
    validate_fluent_residual_histories,
    validate_partial_diagnostic_step_histories,
)
from .native_fine_rendering import (
    DeformedGeometryContractError,
    build_gif,
    load_solver_frame_with_geometry,
    render_displacement_comparison,
    render_final_field_comparison,
    render_solver_velocity_frames,
    validate_deformed_geometry_frames,
)


REPORT_SCHEMA = "our_solver_vs_native_fluent_fine_diagnostic_v2"
PRESSURE_SEMANTICS_MODES = frozenset(("legacy_compatible", "strict"))
SYNCHRONIZED_STEP_END_STAGE = "post_solid_kinematic_projection"


DIAGNOSTIC_MODEL_BLOCKERS: tuple[dict[str, str], ...] = (
    {
        "id": "different_fluid_discretization",
        "detail": (
            "Our solver is a structured Cartesian projection/HIBM flow model; "
            "the reference is native unstructured Fluent. Field errors require "
            "interpolation and are not same-cell residuals."
        ),
    },
    {
        "id": "different_structure_discretization",
        "detail": (
            "Our flap is an MPM particle solid in a 3-D-equivalent slab; the "
            "reference uses Fluent intrinsic structural finite elements in 2-D."
        ),
    },
    {
        "id": "different_flow_closure",
        "detail": (
            "Our run uses a sustained-boundary predictor and pressure projection; "
            "the Fluent setup uses its native pressure-velocity/turbulence models."
        ),
    },
    {
        "id": "different_tip_monitor_support",
        "detail": (
            "Our tip signal is the norm of the mean displacement of all top-row "
            "MPM particles. Fluent reports the norm of the mean vector of four "
            "nodes nearest (0.0505 m, 0.0095 m). The 50-point curves are useful "
            "diagnostic analogs, not identical observables."
        ),
    },
    {
        "id": "pressure_reference_convention",
        "detail": (
            "Pressure comparison assumes the exported gauge conventions are "
            "compatible; a constant pressure offset is reported but not hidden."
        ),
    },
)


def validate_synchronized_solver_time_layer(
    frame_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Require every comparison frame to use one synchronized step-end state."""

    stage_keys = (
        "flow_solution_stage",
        "boundary_topology_stage",
        "structure_geometry_stage",
    )
    for raw_path in frame_paths:
        path = Path(raw_path)
        try:
            with np.load(path, allow_pickle=False) as frame:
                missing = sorted(
                    ({*stage_keys, "flow_boundary_state_synchronized"})
                    - set(frame.files)
                )
                if missing:
                    raise NativeFineComparisonError(
                        "synchronized time layer metadata is missing from "
                        f"{path.name}: {missing}"
                    )
                for key in stage_keys:
                    value = np.asarray(frame[key])
                    actual = value.item() if value.shape == () else None
                    if (
                        not isinstance(actual, str)
                        or actual != SYNCHRONIZED_STEP_END_STAGE
                    ):
                        raise NativeFineComparisonError(
                            "synchronized time layer mismatch in "
                            f"{path.name}: {key}={actual!r}, expected "
                            f"{SYNCHRONIZED_STEP_END_STAGE!r}"
                        )
                synchronized = np.asarray(
                    frame["flow_boundary_state_synchronized"]
                )
                synchronized_value = (
                    synchronized.item() if synchronized.shape == () else None
                )
                if (
                    not isinstance(synchronized_value, (bool, np.bool_))
                    or not bool(synchronized_value)
                ):
                    raise NativeFineComparisonError(
                        "synchronized time layer flag is not true in "
                        f"{path.name}"
                    )
        except NativeFineComparisonError:
            raise
        except (KeyError, OSError, ValueError) as exc:
            raise NativeFineComparisonError(
                f"could not validate synchronized time layer in {path}: {exc}"
            ) from exc
    return {
        "status": "passed",
        "stage": SYNCHRONIZED_STEP_END_STAGE,
        "frame_count": len(frame_paths),
    }


def _validate_pressure_semantics_contract(
    *,
    our_solver_fields_path: Path,
    native_fluent_fields_path: Path,
    mode: str,
) -> dict[str, Any]:
    expected = {
        "pressure_quantity": PRESSURE_QUANTITY,
        "pressure_reference": PRESSURE_REFERENCE,
    }
    sources = (
        ("our_solver", our_solver_fields_path),
        ("native_fluent", native_fluent_fields_path),
    )
    evidence: dict[str, dict[str, Any]] = {}
    missing_sources: list[str] = []
    for label, path in sources:
        try:
            semantics = load_pressure_semantics(path)
        except (OSError, ValueError) as exc:
            raise NativeFineComparisonError(
                f"invalid pressure semantics metadata for {label}: {exc}"
            ) from exc
        present = {
            key: semantics.get(key) is not None
            for key in expected
        }
        if any(present.values()) and not all(present.values()):
            raise NativeFineComparisonError(
                f"partial pressure semantics metadata for {label}: {semantics}"
            )
        if all(present.values()):
            for key, expected_value in expected.items():
                actual = semantics[key]
                if actual != expected_value:
                    raise NativeFineComparisonError(
                        "incompatible pressure semantics metadata for "
                        f"{label}: {key}={actual!r}, expected {expected_value!r}"
                    )
            source_status = "verified"
        else:
            source_status = "legacy_missing"
            missing_sources.append(label)
        evidence[label] = {
            "status": source_status,
            **semantics,
        }

    if mode == "strict" and missing_sources:
        raise NativeFineComparisonError(
            "strict pressure semantics require exact metadata for both field "
            f"artifacts; missing={missing_sources}"
        )
    return {
        "mode": mode,
        "status": "passed" if not missing_sources else "legacy_unverified",
        "expected": expected,
        **evidence,
    }


def _resolve_fluent_force_history_path(
    fluent_input_manifest: Mapping[str, Any],
    *,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is None:
        run_dir = fluent_input_manifest.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir.strip():
            raise NativeFineComparisonError(
                "native Fluent input manifest is missing run_dir for force history"
            )
        path = Path(run_dir) / "history.csv"
    else:
        path = Path(explicit_path)
    path = path.resolve()
    _reject_legacy_reference_path(path)
    if path.name != "history.csv" or not path.is_file():
        raise NativeFineComparisonError(
            f"native Fluent force history must be an existing history.csv: {path}"
        )
    return path


def postprocess_native_fine_comparison(
    our_run_dir: str | Path,
    fluent_postprocess_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_steps: int = DEFAULT_EXPECTED_STEPS,
    velocity_vmax_mps: float = VELOCITY_VMAX_MPS,
    gif_duration_ms: int = 120,
    gif_max_width_px: int = 1600,
    pressure_semantics_mode: str = "legacy_compatible",
    fluent_force_history_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a native-only, offline diagnostic comparison bundle."""

    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    if float(velocity_vmax_mps) != VELOCITY_VMAX_MPS:
        raise ValueError("our velocity GIF must use the fixed 0..31 m/s display range")
    if gif_duration_ms <= 0 or gif_max_width_px <= 0:
        raise ValueError("GIF duration and maximum width must be positive")
    if pressure_semantics_mode not in PRESSURE_SEMANTICS_MODES:
        raise ValueError(
            "pressure_semantics_mode must be 'legacy_compatible' or 'strict'"
        )

    our_run_dir = Path(our_run_dir).resolve()
    fluent_postprocess_dir = Path(fluent_postprocess_dir).resolve()
    output_dir = Path(output_dir).resolve()
    _reject_legacy_reference_path(fluent_postprocess_dir)

    our_manifest = _read_json(our_run_dir / "run_manifest.json")
    our_summary = _read_json(our_run_dir / "our_solver_summary.json")
    our_progress = _read_json(our_run_dir / "progress.json")
    fluent_summary = _read_json(fluent_postprocess_dir / "summary.json")
    fluent_input_manifest_path = fluent_postprocess_dir / "input_manifest.json"
    fluent_input_manifest = _read_json(fluent_input_manifest_path)
    fluent_force_history_path = _resolve_fluent_force_history_path(
        fluent_input_manifest,
        explicit_path=fluent_force_history_path,
    )
    dt_s = _validate_run_contracts(
        our_manifest,
        our_summary,
        our_progress,
        fluent_summary,
        expected_steps=expected_steps,
    )
    _validate_native_fluent_bundle(
        fluent_postprocess_dir,
        fluent_summary,
        fluent_input_manifest,
        expected_steps=expected_steps,
    )

    frame_paths = discover_solver_frames(our_run_dir, expected_steps=expected_steps)
    synchronized_time_layer_contract = validate_synchronized_solver_time_layer(
        frame_paths
    )
    fluent_fields_path = fluent_postprocess_dir / "fields" / "final_fields.npz"
    pressure_semantics_contract = _validate_pressure_semantics_contract(
        our_solver_fields_path=frame_paths[-1],
        native_fluent_fields_path=fluent_fields_path,
        mode=pressure_semantics_mode,
    )
    step_history_paths = discover_solver_step_histories(
        our_run_dir,
        expected_steps=expected_steps,
    )
    our_history_path = our_run_dir / "our_solver_history.csv"
    fluent_structure_path = (
        fluent_postprocess_dir / "histories" / "structure_displacement_history.csv"
    )
    fluent_velocity_path = fluent_postprocess_dir / "histories" / "velocity_history.csv"
    fluent_pressure_path = fluent_postprocess_dir / "histories" / "pressure_history.csv"
    fluent_residual_path = fluent_postprocess_dir / "histories" / "residual_history.csv"
    fluent_residual_summary_path = (
        fluent_postprocess_dir / "histories" / "residual_snapshot_summary.csv"
    )
    our_rows = read_typed_csv(our_history_path)
    fluent_structure_rows = read_typed_csv(fluent_structure_path)
    fluent_velocity_rows = read_typed_csv(fluent_velocity_path)
    fluent_pressure_rows = read_typed_csv(fluent_pressure_path)
    fluent_force_rows = read_typed_csv(fluent_force_history_path)
    _validate_exact_history(
        our_rows, "our-solver history", expected_steps, dt_s,
        allow_missing_time=True,
    )
    _validate_exact_history(
        fluent_structure_rows,
        "native Fluent structure history",
        expected_steps,
        dt_s,
    )
    _validate_exact_history(
        fluent_velocity_rows,
        "native Fluent velocity history",
        expected_steps,
        dt_s,
    )
    _validate_exact_history(
        fluent_pressure_rows,
        "native Fluent pressure history",
        expected_steps,
        dt_s,
    )
    fluent_residual_history_contract = validate_fluent_residual_histories(
        fluent_residual_path,
        fluent_residual_summary_path,
        expected_steps=expected_steps,
        dt_s=dt_s,
    )
    if expected_steps == DEFAULT_EXPECTED_STEPS:
        step_history_contract = validate_final_solver_step_histories(
            step_history_paths,
            our_rows,
            expected_steps=expected_steps,
            dt_s=dt_s,
        )
        final_run_identity_contract = _validate_final_run_identity(
            our_manifest,
            our_summary,
        )
    else:
        step_history_contract = validate_partial_diagnostic_step_histories(
            step_history_paths,
            our_rows,
            expected_steps=expected_steps,
            dt_s=dt_s,
        )
        final_run_identity_contract = {
            "schema": "our_solver_final_native_fine_identity_v1",
            "status": "not_required_for_partial_diagnostic",
        }
    our_config = our_manifest.get("config")
    export_mapping = our_summary.get("solver_npz_summary")
    assert isinstance(our_config, Mapping)
    if not isinstance(export_mapping, Mapping):
        export_mapping = {}
    try:
        deformed_geometry_contract = validate_deformed_geometry_frames(
            frame_paths,
            expected_reverse_streamwise_axis=export_mapping.get(
                "reverse_streamwise_axis"
            ),
            expected_streamwise_length_m=our_config.get("duct_length_m"),
            expected_streamwise_velocity_sign=export_mapping.get(
                "streamwise_velocity_sign"
            ),
        )
    except DeformedGeometryContractError as exc:
        raise NativeFineComparisonError(str(exc)) from exc

    try:
        solver_fields = load_solver_frame_with_geometry(frame_paths[-1])
        fluent_fields = load_fluent_npz(fluent_fields_path)
    except (KeyError, OSError, ValueError) as exc:
        raise NativeFineComparisonError(f"could not load final comparison fields: {exc}") from exc
    _validate_solver_fields(solver_fields)
    _validate_fluent_fields(fluent_fields)
    field_comparison, sampled = _compare_final_fields(solver_fields, fluent_fields)
    displacement_rows, displacement_comparison = compare_displacement_histories(
        our_rows,
        fluent_structure_rows,
        expected_steps=expected_steps,
        dt_s=dt_s,
    )
    span_m = _finite_float(our_config.get("span_m"), "solver span_m")
    force_rows, interface_force_comparison = compare_interface_force_histories(
        our_rows,
        fluent_force_rows,
        span_m=span_m,
        expected_steps=expected_steps,
        dt_s=dt_s,
    )
    five_percent_diagnostic_gate = evaluate_five_percent_diagnostic_gate(
        field_comparison,
        displacement_comparison,
        interface_force_comparison,
        expected_steps=expected_steps,
    )

    prepare_new_output_dir(output_dir)
    frames_dir = output_dir / "figures" / "our_velocity_frames"
    rendered_frames = render_solver_velocity_frames(
        frame_paths,
        frames_dir,
        dt_s=dt_s,
        velocity_vmax_mps=velocity_vmax_mps,
    )
    velocity_gif = output_dir / "our_velocity_magnitude_fixed_0_31.gif"
    build_gif(
        rendered_frames,
        velocity_gif,
        duration_ms=gif_duration_ms,
        max_width_px=gif_max_width_px,
    )
    figures_dir = output_dir / "figures"
    velocity_figure = figures_dir / "final_velocity_comparison.png"
    pressure_figure = figures_dir / "final_pressure_comparison.png"
    displacement_figure = figures_dir / "displacement_comparison_50point.png"
    render_final_field_comparison(
        solver_fields,
        fluent_fields,
        sampled,
        velocity_figure,
        pressure_figure,
        velocity_vmax_mps=velocity_vmax_mps,
    )
    render_displacement_comparison(displacement_rows, displacement_figure)

    displacement_csv = output_dir / "histories" / "displacement_comparison_50point.csv"
    write_csv(displacement_csv, displacement_rows)
    force_csv = output_dir / "histories" / "interface_force_comparison_50point.csv"
    write_csv(force_csv, force_rows)
    input_manifest = _input_manifest(
        our_run_dir=our_run_dir,
        fluent_postprocess_dir=fluent_postprocess_dir,
        frame_paths=frame_paths,
        step_history_paths=step_history_paths,
        additional_paths=(
            our_history_path,
            fluent_input_manifest_path,
            fluent_postprocess_dir / "CHECKSUMS.sha256",
            fluent_structure_path,
            fluent_velocity_path,
            fluent_pressure_path,
            fluent_residual_path,
            fluent_residual_summary_path,
            fluent_fields_path,
            fluent_force_history_path,
        ),
        expected_steps=expected_steps,
        dt_s=dt_s,
    )
    input_manifest_path = output_dir / "input_manifest.json"
    write_json(input_manifest_path, input_manifest)

    outputs = {
        "our_velocity_gif": _relative(velocity_gif, output_dir),
        "our_velocity_frames_dir": _relative(frames_dir, output_dir),
        "final_velocity_comparison": _relative(velocity_figure, output_dir),
        "final_pressure_comparison": _relative(pressure_figure, output_dir),
        "displacement_comparison_figure": _relative(displacement_figure, output_dir),
        "displacement_comparison_csv": _relative(displacement_csv, output_dir),
        "interface_force_comparison_csv": _relative(force_csv, output_dir),
        "input_manifest": _relative(input_manifest_path, output_dir),
        "comparison_report_json": "comparison_report.json",
        "comparison_report_markdown": "comparison_report.md",
        "checksums": "CHECKSUMS.sha256",
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic_complete",
        "parity_claimed": False,
        "legacy_puma_reference_used": False,
        "comparison_role": "native_fine_cross_model_diagnostic",
        "step_count": expected_steps,
        "dt_s": dt_s,
        "final_time_s": expected_steps * dt_s,
        "velocity_display_range_mps": [VELOCITY_VMIN_MPS, velocity_vmax_mps],
        "our_run_dir": str(our_run_dir),
        "native_fluent_postprocess_dir": str(fluent_postprocess_dir),
        "output_dir": str(output_dir),
        "native_fluent_reference_contract": {
            "schema": NATIVE_FLUENT_SCHEMA,
            "input_manifest_schema": "fluent_fine_fsi_input_pairs_v1",
            "canonical_path_markers": list(CANONICAL_NATIVE_FLUENT_PATH_MARKERS),
            "old_puma_or_adapted_reference_rejected": True,
        },
        "step_history_contract": step_history_contract,
        "final_run_identity_contract": final_run_identity_contract,
        "fluent_residual_history_contract": fluent_residual_history_contract,
        "deformed_geometry_contract": deformed_geometry_contract,
        "synchronized_time_layer_contract": synchronized_time_layer_contract,
        "pressure_semantics_contract": pressure_semantics_contract,
        "final_field_comparison": field_comparison,
        "displacement_comparison": displacement_comparison,
        "interface_force_comparison": interface_force_comparison,
        "five_percent_diagnostic_gate": five_percent_diagnostic_gate,
        "diagnostic_model_blockers": [dict(item) for item in DIAGNOSTIC_MODEL_BLOCKERS],
        "outputs": outputs,
    }
    report_json = output_dir / "comparison_report.json"
    report_md = output_dir / "comparison_report.md"
    write_json(report_json, report)
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    checksum_path = write_checksums(output_dir)
    verify_checksums(checksum_path, output_dir)
    return report


def evaluate_five_percent_diagnostic_gate(
    field_comparison: Mapping[str, Any],
    displacement_comparison: Mapping[str, Any],
    interface_force_comparison: Mapping[str, Any],
    *,
    expected_steps: int,
    relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Evaluate the requested 5% numerical target without claiming model parity."""

    if isinstance(expected_steps, bool) or not isinstance(expected_steps, int):
        raise TypeError("expected_steps must be a non-bool integer")
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("relative_tolerance must be finite and in (0, 1)")

    try:
        direct_errors = field_comparison["direct_errors"]
        pressure = field_comparison["pressure_reference_diagnostic"]
        pressure_range_pa = float(
            pressure["native_fluent_sampled_pressure_range_pa"]
        )
        pressure_rmse_pa = float(
            pressure["zero_mean_pressure_difference_rmse_pa"]
        )
        pressure_raw_rmse_pa = float(
            pressure["raw_pressure_difference_rmse_pa"]
        )
        pressure_mean_offset_pa = float(
            pressure["sampled_mean_offset_our_minus_fluent_pa"]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise NativeFineComparisonError(
            "five-percent field gate requires complete finite comparison metrics"
        ) from exc
    if not np.isfinite(pressure_range_pa) or pressure_range_pa <= 0.0:
        raise NativeFineComparisonError(
            "native Fluent sampled pressure range must be positive and finite"
        )

    values: dict[str, float] = {}
    try:
        for field_name in ("u", "v", "speed"):
            values[f"field_{field_name}_nrmse"] = float(
                direct_errors[field_name]["nrmse_by_reference_max_abs"]
            )
        values["pressure_gauge_nrmse"] = pressure_rmse_pa / pressure_range_pa
        values["pressure_raw_nrmse"] = pressure_raw_rmse_pa / pressure_range_pa
        values["pressure_mean_offset_fraction"] = (
            abs(pressure_mean_offset_pa) / pressure_range_pa
        )
        for series_name, prefix in (
            ("tip_streamwise", "tip_streamwise"),
            ("tip_transverse", "tip_transverse"),
            ("tip_mean_vector", "tip_norm"),
            ("solid_max", "solid_max"),
        ):
            series = displacement_comparison[series_name]
            reference_peak = abs(float(series["fluent_peak_m"]))
            if not np.isfinite(reference_peak) or reference_peak <= 0.0:
                raise NativeFineComparisonError(
                    f"{series_name} Fluent peak must be positive and finite"
                )
            values[f"{prefix}_waveform_nrmse"] = float(
                series["nrmse_by_reference_peak"]
            )
            values[f"{prefix}_peak_amplitude_relative_error"] = (
                abs(
                    abs(float(series["our_peak_m"]))
                    - abs(float(series["fluent_peak_m"]))
                )
                / reference_peak
            )
            values[f"{prefix}_peak_phase_fraction"] = (
                abs(int(series["our_peak_step"]) - int(series["fluent_peak_step"]))
                / float(expected_steps)
            )
            values[f"{prefix}_final_error_by_reference_peak"] = (
                abs(float(series["our_final_m"]) - float(series["fluent_final_m"]))
                / reference_peak
            )
        for series_name, prefix in (
            ("streamwise", "force_streamwise"),
            ("transverse", "force_transverse"),
        ):
            series = interface_force_comparison[series_name]
            reference_peak = abs(float(series["fluent_peak_n_per_m"]))
            if not np.isfinite(reference_peak) or reference_peak <= 0.0:
                raise NativeFineComparisonError(
                    f"{series_name} Fluent force peak must be positive and finite"
                )
            values[f"{prefix}_waveform_nrmse"] = float(
                series["nrmse_by_reference_peak"]
            )
            values[f"{prefix}_peak_amplitude_relative_error"] = (
                abs(
                    abs(float(series["our_peak_n_per_m"]))
                    - abs(float(series["fluent_peak_n_per_m"]))
                )
                / reference_peak
            )
            values[f"{prefix}_peak_phase_fraction"] = (
                abs(int(series["our_peak_step"]) - int(series["fluent_peak_step"]))
                / float(expected_steps)
            )
            values[f"{prefix}_final_error_by_reference_peak"] = (
                abs(
                    float(series["our_final_n_per_m"])
                    - float(series["fluent_final_n_per_m"])
                )
                / reference_peak
            )
        values["force_out_of_plane_leakage"] = float(
            interface_force_comparison[
                "out_of_plane_leakage_by_reference_streamwise_peak"
            ]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, NativeFineComparisonError):
            raise
        raise NativeFineComparisonError(
            "five-percent displacement and force gate requires complete finite metrics"
        ) from exc

    metrics = {
        name: {
            "value": value,
            "tolerance": tolerance,
            "passed": bool(np.isfinite(value) and 0.0 <= value <= tolerance),
        }
        for name, value in values.items()
    }
    passed = all(metric["passed"] for metric in metrics.values())
    return {
        "schema": "native_fine_five_percent_diagnostic_gate_v2",
        "status": "passed" if passed else "failed",
        "all_metrics_within_tolerance": passed,
        "relative_tolerance": tolerance,
        "parity_claimed": False,
        "metrics": metrics,
    }


def compare_interface_force_histories(
    our_rows: Sequence[Mapping[str, Any]],
    fluent_rows: Sequence[Mapping[str, Any]],
    *,
    span_m: float,
    expected_steps: int,
    dt_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare the 3-D slab load with Fluent's 2-D unit-depth wall force."""

    _validate_exact_history(
        our_rows,
        "our-solver interface force history",
        expected_steps,
        dt_s,
        allow_missing_time=True,
    )
    _validate_exact_history(
        fluent_rows,
        "native Fluent interface force history",
        expected_steps,
        dt_s,
    )
    span = _finite_float(span_m, "solver span_m")
    if span <= 0.0:
        raise NativeFineComparisonError("solver span_m must be positive")

    comparison_rows: list[dict[str, Any]] = []
    our_streamwise: list[float] = []
    fluent_streamwise: list[float] = []
    our_transverse: list[float] = []
    fluent_transverse: list[float] = []
    our_out_of_plane: list[float] = []
    for our_row, fluent_row in zip(our_rows, fluent_rows, strict=True):
        step = int(our_row["step"])
        vector = _vector(our_row.get("total_marker_force_n"))
        if len(vector) != 3:
            raise NativeFineComparisonError(
                f"our-solver step {step} lacks finite interface force: "
                "total_marker_force_n must have 3 components"
            )
        try:
            solver_force = tuple(
                _finite_float(value, f"solver interface force step {step}")
                for value in vector
            )
            fluent_x = _finite_float(
                fluent_row.get("flap_fluid_force_x_n"),
                f"Fluent streamwise interface force step {step}",
            )
            fluent_y = _finite_float(
                fluent_row.get("flap_fluid_force_y_n"),
                f"Fluent transverse interface force step {step}",
            )
            _finite_float(
                fluent_row.get("flap_fluid_force_z_n"),
                f"Fluent out-of-plane interface force step {step}",
            )
        except (NativeFineComparisonError, TypeError, ValueError, OverflowError) as exc:
            raise NativeFineComparisonError(
                f"interface force history must contain finite interface force values: {exc}"
            ) from exc
        mapped_streamwise = -solver_force[2] / span
        mapped_transverse = solver_force[1] / span
        mapped_out_of_plane = solver_force[0] / span
        our_streamwise.append(mapped_streamwise)
        fluent_streamwise.append(fluent_x)
        our_transverse.append(mapped_transverse)
        fluent_transverse.append(fluent_y)
        our_out_of_plane.append(mapped_out_of_plane)
        comparison_rows.append(
            {
                "step": step,
                "time_s": float(step * dt_s),
                "our_streamwise_force_n_per_m": mapped_streamwise,
                "fluent_streamwise_force_n_per_m": fluent_x,
                "streamwise_signed_error_n_per_m": mapped_streamwise - fluent_x,
                "our_transverse_force_n_per_m": mapped_transverse,
                "fluent_transverse_force_n_per_m": fluent_y,
                "transverse_signed_error_n_per_m": mapped_transverse - fluent_y,
                "our_out_of_plane_force_n_per_m": mapped_out_of_plane,
            }
        )

    streamwise_metrics = _force_series_error_metrics(
        our_streamwise, fluent_streamwise
    )
    transverse_metrics = _force_series_error_metrics(
        our_transverse, fluent_transverse
    )
    streamwise_peak = abs(float(streamwise_metrics["fluent_peak_n_per_m"]))
    if streamwise_peak <= 0.0:
        raise NativeFineComparisonError(
            "native Fluent streamwise interface force peak must be positive"
        )
    leakage = max(abs(value) for value in our_out_of_plane) / streamwise_peak
    return comparison_rows, {
        "schema": "native_fine_interface_force_comparison_v1",
        "sample_count": expected_steps,
        "units": "N/m",
        "solver_span_m": span,
        "axis_mapping": {
            "streamwise": "-solver_z/span_m",
            "transverse": "+solver_y/span_m",
            "out_of_plane": "+solver_x/span_m",
        },
        "streamwise": streamwise_metrics,
        "transverse": transverse_metrics,
        "out_of_plane_leakage_by_reference_streamwise_peak": leakage,
    }


def compare_displacement_histories(
    our_rows: Sequence[Mapping[str, Any]],
    fluent_rows: Sequence[Mapping[str, Any]],
    *,
    expected_steps: int,
    dt_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_exact_history(
        our_rows, "our-solver history", expected_steps, dt_s,
        allow_missing_time=True,
    )
    _validate_exact_history(
        fluent_rows,
        "native Fluent structure history",
        expected_steps,
        dt_s,
    )
    comparison_rows: list[dict[str, Any]] = []
    solver_tip_streamwise: list[float] = []
    fluent_tip_streamwise: list[float] = []
    solver_tip_transverse: list[float] = []
    fluent_tip_transverse: list[float] = []
    solver_tip: list[float] = []
    fluent_tip: list[float] = []
    solver_solid: list[float] = []
    fluent_solid: list[float] = []
    for our_row, fluent_row in zip(our_rows, fluent_rows, strict=True):
        step = int(our_row["step"])
        our_vector = _vector(our_row.get("tip_mean_displacement_m"))
        if len(our_vector) < 3:
            raise NativeFineComparisonError(
                f"our-solver step {step} lacks a 3-component tip_mean_displacement_m"
            )
        # The canonical solver embeds the 2-D Fluent x/y plane as -z/+y.
        our_tip_streamwise = _finite_float(
            -float(our_vector[2]),
            f"our-solver streamwise tip displacement at step {step}",
        )
        our_tip_transverse = _finite_float(
            our_vector[1],
            f"our-solver transverse tip displacement at step {step}",
        )
        fluent_tip_streamwise_value = _finite_float(
            fluent_row.get("tip_displacement_x_m"),
            f"Fluent streamwise tip displacement at step {step}",
        )
        fluent_tip_transverse_value = _finite_float(
            fluent_row.get("tip_displacement_y_m"),
            f"Fluent transverse tip displacement at step {step}",
        )
        our_tip_norm = float(np.linalg.norm(np.asarray(our_vector[:3], dtype=np.float64)))
        fluent_tip_norm = _finite_float(
            fluent_row.get("tip_mean_vector_norm_m"),
            f"Fluent tip mean vector norm at step {step}",
        )
        our_solid_max = _finite_float(
            our_row.get("max_displacement_m"),
            f"our-solver maximum displacement at step {step}",
        )
        fluent_solid_max = _finite_float(
            fluent_row.get("max_displacement_m"),
            f"Fluent maximum displacement at step {step}",
        )
        solver_tip_streamwise.append(our_tip_streamwise)
        fluent_tip_streamwise.append(fluent_tip_streamwise_value)
        solver_tip_transverse.append(our_tip_transverse)
        fluent_tip_transverse.append(fluent_tip_transverse_value)
        solver_tip.append(our_tip_norm)
        fluent_tip.append(fluent_tip_norm)
        solver_solid.append(our_solid_max)
        fluent_solid.append(fluent_solid_max)
        comparison_rows.append(
            {
                "step": step,
                "time_s": float(step * dt_s),
                "our_tip_streamwise_displacement_m": our_tip_streamwise,
                "fluent_tip_streamwise_displacement_m": (
                    fluent_tip_streamwise_value
                ),
                "tip_streamwise_signed_error_m": (
                    our_tip_streamwise - fluent_tip_streamwise_value
                ),
                "our_tip_transverse_displacement_m": our_tip_transverse,
                "fluent_tip_transverse_displacement_m": (
                    fluent_tip_transverse_value
                ),
                "tip_transverse_signed_error_m": (
                    our_tip_transverse - fluent_tip_transverse_value
                ),
                "our_tip_mean_vector_norm_m": our_tip_norm,
                "fluent_tip_mean_vector_norm_m": fluent_tip_norm,
                "tip_mean_signed_error_m": our_tip_norm - fluent_tip_norm,
                "tip_mean_abs_error_m": abs(our_tip_norm - fluent_tip_norm),
                "our_solid_max_displacement_m": our_solid_max,
                "fluent_solid_max_displacement_m": fluent_solid_max,
                "solid_max_signed_error_m": our_solid_max - fluent_solid_max,
                "solid_max_abs_error_m": abs(our_solid_max - fluent_solid_max),
                "fluent_selected_node_count": int(fluent_row.get("selected_node_count", 0)),
                "definition_alignment": "diagnostic_analog_not_identical",
            }
        )
    return comparison_rows, {
        "diagnostic_only": True,
        "definition_alignment": "diagnostic_analog_not_identical",
        "sample_count": expected_steps,
        "tip_streamwise": _series_error_metrics(
            solver_tip_streamwise,
            fluent_tip_streamwise,
        ),
        "tip_transverse": _series_error_metrics(
            solver_tip_transverse,
            fluent_tip_transverse,
        ),
        "tip_mean_vector": _series_error_metrics(solver_tip, fluent_tip),
        "solid_max": _series_error_metrics(solver_solid, fluent_solid),
        "our_tip_definition": (
            "norm(mean(displacement vector)) across all MPM particles in the "
            "uppermost rest-position particle row"
        ),
        "fluent_tip_definition": (
            "norm(mean(x/y displacement vector)) across four intrinsic-structure "
            "nodes nearest the configured target point"
        ),
        "solid_max_definition": "maximum displacement-vector norm over each full solid discretization",
    }


def prepare_new_output_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise NativeFineComparisonError(f"comparison output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def read_typed_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise NativeFineComparisonError(f"required history is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise NativeFineComparisonError(f"history has no header: {path}")
        return [
            {str(key): _parse_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


def render_markdown_report(report: Mapping[str, Any]) -> str:
    field = report["final_field_comparison"]
    displacement = report["displacement_comparison"]
    interface_force = report["interface_force_comparison"]
    geometry = report["deformed_geometry_contract"]
    step_history = report["step_history_contract"]
    extrema = field["final_extrema"]
    lines = [
        "# Our Solver vs Native Fluent Fine-Grid Diagnostic",
        "",
        "This bundle uses only the locked native Fluent fine-grid postprocess output.",
        "It is **not a parity claim** because the numerical models and monitor supports differ.",
        "",
        "## Scope",
        "",
        f"- Status: `{report['status']}`",
        f"- Steps: `{report['step_count']}`",
        f"- dt: `{report['dt_s']}` s",
        f"- Final time: `{report['final_time_s']}` s",
        "- Velocity GIF display range: `0..31 m/s`",
        f"- Final overlapping Fluent-cell samples: `{field['sample_count']}`",
        f"- Full per-step JSON histories verified: `{step_history['step_count']}`",
        "- Adapted/legacy Fluent references: `rejected by contract`",
        "",
        "## True moving-geometry overlay",
        "",
        f"- Solver frames validated: `{geometry['frame_count']}`",
        f"- Solid particles per frame: `{geometry['solid_point_count']}`",
        f"- HIBM markers per frame: `{geometry['marker_point_count']}`",
        f"- Peak particle displacement from saved geometry: "
        f"`{geometry['peak_particle_displacement_m']}` m at step "
        f"`{geometry['peak_particle_displacement_step']}`",
        "- GIF overlay: deformed solid only; HIBM markers and rest positions hidden",
        "",
        "## Final field diagnostics",
        "",
        f"- Our global final speed maximum: "
        f"`{extrema['our_solver_global_speed_max_mps']}` m/s",
        f"- Native Fluent global final speed maximum: "
        f"`{extrema['native_fluent_global_speed_max_mps']}` m/s",
        f"- Sampled speed RMSE: `{field['direct_errors']['speed']['rmse']}` m/s",
        f"- Raw sampled pressure RMSE: `{field['direct_errors']['p']['rmse']}` Pa",
        "- Pressure offset is reported explicitly; it is not removed from the primary metrics.",
        "",
        "## Displacement comparison",
        "",
        f"- Exact aligned samples: `{displacement['sample_count']}`",
        f"- Tip diagnostic RMSE: `{displacement['tip_mean_vector']['rmse_m']}` m",
        f"- Whole-solid maximum RMSE: `{displacement['solid_max']['rmse_m']}` m",
        f"- Our tip definition: {displacement['our_tip_definition']}",
        f"- Fluent tip definition: {displacement['fluent_tip_definition']}",
        "",
        "## Interface-force comparison",
        "",
        f"- Exact aligned samples: `{interface_force['sample_count']}`",
        "- Solver-to-Fluent mapping: `Fx=-Fz/span`, `Fy=+Fy/span`",
        f"- Streamwise waveform NRMSE: "
        f"`{interface_force['streamwise']['nrmse_by_reference_peak']}`",
        f"- Transverse waveform NRMSE: "
        f"`{interface_force['transverse']['nrmse_by_reference_peak']}`",
        f"- Out-of-plane leakage fraction: "
        f"`{interface_force['out_of_plane_leakage_by_reference_streamwise_peak']}`",
        "",
        "## Diagnostic model blockers",
        "",
    ]
    for blocker in report["diagnostic_model_blockers"]:
        lines.append(f"- `{blocker['id']}`: {blocker['detail']}")
    lines.extend(["", "## Outputs", ""])
    for name, path in report["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise NativeFineComparisonError(f"cannot write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_checksums(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    checksum_path = output_dir / "CHECKSUMS.sha256"
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checksum_path


def verify_checksums(checksum_path: str | Path, output_dir: str | Path) -> int:
    checksum_path = Path(checksum_path)
    output_dir = Path(output_dir)
    verified = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative_path = line.split(None, 1)
        except ValueError as exc:
            raise NativeFineComparisonError(
                f"invalid generated checksum row: {line!r}"
            ) from exc
        target = output_dir / Path(relative_path.strip())
        if not target.is_file():
            raise NativeFineComparisonError(
                f"generated checksummed output is missing: {target}"
            )
        actual = sha256_file(target)
        if actual.lower() != expected.lower():
            raise NativeFineComparisonError(
                f"generated checksum mismatch for {relative_path.strip()}"
            )
        verified += 1
    if verified <= 0:
        raise NativeFineComparisonError("generated checksum manifest is empty")
    return verified


def _compare_final_fields(
    solver_fields: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    helper_result = compare_solver_to_fluent_field(solver_fields, fluent_fields)
    sampled = sample_structured_solver_at_fluent_points(solver_fields, fluent_fields)
    valid = np.asarray(sampled["valid"], dtype=bool)
    if not np.any(valid):
        raise NativeFineComparisonError("no valid overlap for final field comparison")
    direct_errors = {}
    for key in ("u", "v", "speed", "p"):
        direct_errors[key] = _field_error_metrics(
            np.asarray(sampled[key])[valid],
            np.asarray(fluent_fields[key])[valid],
        )
    solver_fluid_mask = np.asarray(solver_fields["fluid_mask"], dtype=bool)
    solver_speed = np.asarray(solver_fields["speed"], dtype=np.float64)[
        solver_fluid_mask
    ]
    solver_pressure = np.asarray(solver_fields["p"], dtype=np.float64)[
        solver_fluid_mask
    ]
    fluent_speed = np.asarray(fluent_fields["speed"], dtype=np.float64)
    fluent_pressure = np.asarray(fluent_fields["p"], dtype=np.float64)
    sampled_pressure = np.asarray(sampled["p"], dtype=np.float64)[valid]
    reference_pressure = fluent_pressure[valid]
    pressure_difference = sampled_pressure - reference_pressure
    pressure_mean_offset = float(np.mean(pressure_difference))
    pressure_zero_mean_rmse = float(
        np.sqrt(np.mean((pressure_difference - pressure_mean_offset) ** 2))
    )
    return {
        "diagnostic_only": True,
        "interpolation": "fluid-weighted bilinear sampling at native Fluent cell centers",
        "sample_count": int(np.count_nonzero(valid)),
        "direct_errors": direct_errors,
        "final_extrema": {
            "our_solver_global_speed_max_mps": float(np.max(solver_speed)),
            "native_fluent_global_speed_max_mps": float(np.max(fluent_speed)),
            "our_solver_sampled_speed_max_mps": float(
                np.max(np.asarray(sampled["speed"], dtype=np.float64)[valid])
            ),
            "native_fluent_sampled_speed_max_mps": float(
                np.max(fluent_speed[valid])
            ),
            "our_solver_global_pressure_min_pa": float(np.min(solver_pressure)),
            "our_solver_global_pressure_max_pa": float(np.max(solver_pressure)),
            "native_fluent_global_pressure_min_pa": float(np.min(fluent_pressure)),
            "native_fluent_global_pressure_max_pa": float(np.max(fluent_pressure)),
        },
        "pressure_reference_diagnostic": {
            "sampled_mean_offset_our_minus_fluent_pa": pressure_mean_offset,
            "zero_mean_pressure_difference_rmse_pa": pressure_zero_mean_rmse,
            "raw_pressure_difference_rmse_pa": direct_errors["p"]["rmse"],
            "native_fluent_sampled_pressure_range_pa": float(
                np.max(reference_pressure) - np.min(reference_pressure)
            ),
            "constant_offset_removed_from_primary_metrics": False,
        },
        "existing_parity_helper_diagnostics": helper_result,
    }, sampled


def _field_error_metrics(values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_scale = max(float(np.max(np.abs(reference))), np.finfo(np.float64).eps)
    return {
        "mean_signed_error": float(np.mean(difference)),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "max_absolute_error": float(np.max(np.abs(difference))),
        "nrmse_by_reference_max_abs": rmse / reference_scale,
    }


def _series_error_metrics(values: Iterable[float], reference: Iterable[float]) -> dict[str, Any]:
    values_array = np.asarray(list(values), dtype=np.float64)
    reference_array = np.asarray(list(reference), dtype=np.float64)
    difference = values_array - reference_array
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_scale = max(float(np.max(np.abs(reference_array))), np.finfo(np.float64).eps)
    our_peak_index = int(np.argmax(np.abs(values_array)))
    fluent_peak_index = int(np.argmax(np.abs(reference_array)))
    return {
        "rmse_m": rmse,
        "nrmse_by_reference_peak": rmse / reference_scale,
        "mean_signed_error_m": float(np.mean(difference)),
        "mean_absolute_error_m": float(np.mean(np.abs(difference))),
        "max_absolute_error_m": float(np.max(np.abs(difference))),
        "our_peak_m": float(values_array[our_peak_index]),
        "fluent_peak_m": float(reference_array[fluent_peak_index]),
        "our_peak_step": our_peak_index + 1,
        "fluent_peak_step": fluent_peak_index + 1,
        "our_final_m": float(values_array[-1]),
        "fluent_final_m": float(reference_array[-1]),
    }


def _force_series_error_metrics(
    values: Iterable[float],
    reference: Iterable[float],
) -> dict[str, Any]:
    values_array = np.asarray(list(values), dtype=np.float64)
    reference_array = np.asarray(list(reference), dtype=np.float64)
    if (
        values_array.size == 0
        or values_array.shape != reference_array.shape
        or not np.all(np.isfinite(values_array))
        or not np.all(np.isfinite(reference_array))
    ):
        raise NativeFineComparisonError(
            "interface force series must be non-empty, aligned, and finite"
        )
    difference = values_array - reference_array
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_scale = float(np.max(np.abs(reference_array)))
    if reference_scale <= 0.0:
        raise NativeFineComparisonError(
            "native Fluent interface force component peak must be positive"
        )
    our_peak_index = int(np.argmax(np.abs(values_array)))
    fluent_peak_index = int(np.argmax(np.abs(reference_array)))
    return {
        "rmse_n_per_m": rmse,
        "nrmse_by_reference_peak": rmse / reference_scale,
        "mean_signed_error_n_per_m": float(np.mean(difference)),
        "mean_absolute_error_n_per_m": float(np.mean(np.abs(difference))),
        "max_absolute_error_n_per_m": float(np.max(np.abs(difference))),
        "our_peak_n_per_m": float(values_array[our_peak_index]),
        "fluent_peak_n_per_m": float(reference_array[fluent_peak_index]),
        "our_peak_step": our_peak_index + 1,
        "fluent_peak_step": fluent_peak_index + 1,
        "our_final_n_per_m": float(values_array[-1]),
        "fluent_final_n_per_m": float(reference_array[-1]),
    }


def _validate_solver_fields(fields: Mapping[str, np.ndarray]) -> None:
    shape = np.asarray(fields["speed"]).shape
    if len(shape) != 2 or np.asarray(fields["fluid_mask"]).shape != shape:
        raise NativeFineComparisonError("solver field arrays must share a 2-D grid")
    for key in ("u", "v", "p", "speed"):
        values = np.asarray(fields[key])
        if values.shape != shape or not np.all(np.isfinite(values[fields["fluid_mask"]])):
            raise NativeFineComparisonError(f"invalid solver field array: {key}")


def _validate_fluent_fields(fields: Mapping[str, np.ndarray]) -> None:
    count = np.asarray(fields["x"]).size
    if count <= 0:
        raise NativeFineComparisonError("native Fluent final field is empty")
    for key in ("x", "y", "u", "v", "p", "speed", "cell_ids"):
        values = np.asarray(fields[key])
        if values.size != count:
            raise NativeFineComparisonError(f"native Fluent field size mismatch: {key}")
        if key != "cell_ids" and not np.all(np.isfinite(values)):
            raise NativeFineComparisonError(f"non-finite native Fluent field: {key}")


def _input_manifest(
    *,
    our_run_dir: Path,
    fluent_postprocess_dir: Path,
    frame_paths: Sequence[Path],
    step_history_paths: Sequence[Path],
    additional_paths: Sequence[Path],
    expected_steps: int,
    dt_s: float,
) -> dict[str, Any]:
    paths = [
        our_run_dir / "run_manifest.json",
        our_run_dir / "our_solver_summary.json",
        our_run_dir / "progress.json",
        fluent_postprocess_dir / "summary.json",
        *frame_paths,
        *step_history_paths,
        *additional_paths,
    ]
    return {
        "schema": "our_solver_vs_native_fluent_fine_inputs_v2",
        "expected_steps": expected_steps,
        "dt_s": dt_s,
        "native_fluent_schema": NATIVE_FLUENT_SCHEMA,
        "legacy_reference_used": False,
        "solver_step_field_count": len(frame_paths),
        "solver_step_history_count": len(step_history_paths),
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in paths
        ],
    }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
