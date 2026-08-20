from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import runpy

import numpy as np
import pytest

from src.refactored.validation.ansys_vertical_flap_fsi.native_fine_comparison import (
    NativeFineComparisonError,
    postprocess_native_fine_comparison,
    read_typed_csv,
)
from src.refactored.validation.ansys_vertical_flap_fsi.native_fine_contracts import (
    CANONICAL_NATIVE_FLUENT_PATH_MARKERS,
    _validate_native_fluent_bundle,
    _validate_run_contracts,
    _validate_final_projection_success,
    _validate_final_run_identity,
    discover_solver_step_histories,
    validate_final_solver_step_histories,
    validate_partial_diagnostic_step_histories,
)

from .test_native_fine_comparison import (
    FLUENT_CHECKSUM_RELATIVE_PATHS,
    _fine_solver_config,
    _fine_solver_summary_identity,
    _solver_field,
    _synthetic_inputs,
    _write_checksums,
    _write_csv,
    _write_json,
)


def _set_projection_status(
    our_dir: Path,
    step: int,
    *,
    pressure_solve_failed: bool,
    cg_converged_all: bool,
) -> None:
    history_path = our_dir / "step_history" / f"step_{step:04d}.json"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload["history"]
    history["flow_projection_pressure_solve_failed"] = pressure_solve_failed
    history["flow_projection_cg_converged_all"] = cg_converged_all
    history["flow_projection_report"]["pressure_solve_failed"] = pressure_solve_failed
    history["flow_projection_report"]["cg_converged_all"] = cg_converged_all
    _write_json(history_path, payload)


def _drop_aggregate_history_time(our_dir: Path) -> None:
    aggregate_path = our_dir / "our_solver_history.csv"
    with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("time_s")
    _write_csv(aggregate_path, rows)


def test_final_50_step_history_contract_rejects_any_numerical_failure(
    tmp_path: Path,
) -> None:
    our_dir, _ = _synthetic_inputs(tmp_path, steps=50)
    _set_projection_status(
        our_dir,
        17,
        pressure_solve_failed=True,
        cg_converged_all=False,
    )

    with pytest.raises(
        NativeFineComparisonError,
        match="final 50-step numerical success.*step_0017",
    ):
        validate_final_solver_step_histories(
            discover_solver_step_histories(our_dir, expected_steps=50),
            read_typed_csv(our_dir / "our_solver_history.csv"),
            expected_steps=50,
            dt_s=5.0e-4,
        )


def test_final_50_step_history_uses_wrapper_time_when_aggregate_omits_it(
    tmp_path: Path,
) -> None:
    our_dir, _ = _synthetic_inputs(tmp_path, steps=50)
    _drop_aggregate_history_time(our_dir)

    contract = validate_final_solver_step_histories(
        discover_solver_step_histories(our_dir, expected_steps=50),
        read_typed_csv(our_dir / "our_solver_history.csv"),
        expected_steps=50,
        dt_s=5.0e-4,
    )

    assert contract["time_source"] == "step_history_wrapper"
    assert contract["aggregate_time_cross_check"] == "not_present"


def test_step_history_rejects_wrong_wrapper_time_without_aggregate_time(
    tmp_path: Path,
) -> None:
    our_dir, _ = _synthetic_inputs(tmp_path, steps=3)
    _drop_aggregate_history_time(our_dir)
    history_path = our_dir / "step_history" / "step_0002.json"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    payload["time_s"] = 9.0e-3
    _write_json(history_path, payload)

    with pytest.raises(NativeFineComparisonError, match="per-step history time mismatch"):
        validate_partial_diagnostic_step_histories(
            discover_solver_step_histories(our_dir, expected_steps=3),
            read_typed_csv(our_dir / "our_solver_history.csv"),
            expected_steps=3,
            dt_s=5.0e-4,
        )


def test_final_identity_accepts_json_float_round_trip_for_locked_vector() -> None:
    manifest: dict[str, object] = {"config": _fine_solver_config(50)}
    config = manifest["config"]
    assert isinstance(config, dict)
    config["flow_hibm_sharp_search_radius_xyz_m"] = [
        0.0012000000000000001,
        0.000390625,
        0.00046875,
    ]
    config["flow_hibm_sharp_interpolate_velocity_rows"] = False

    contract = _validate_final_run_identity(
        manifest,
        _fine_solver_summary_identity(),
    )

    assert contract["status"] == "passed"


def test_final_identity_accepts_locked_windowed_stationary_preflow() -> None:
    manifest: dict[str, object] = {"config": _fine_solver_config(50)}
    config = manifest["config"]
    assert isinstance(config, dict)
    config.update(
        {
            "preflow_steps": 200,
            "preflow_convergence_mode": "windowed_stationary",
            "preflow_stationary_min_steps": 20,
            "preflow_stationary_window_steps": 10,
            "preflow_stationary_consecutive_windows": 3,
            "preflow_stationary_tolerance": 0.01,
            "preflow_stationary_divergence_tolerance": 0.05,
            "preflow_stationary_no_slip_tolerance_fraction": 0.05,
        }
    )

    contract = _validate_final_run_identity(
        manifest,
        _fine_solver_summary_identity(),
    )

    assert contract["status"] == "passed"
    assert contract["config"]["preflow_steps"] == 200
    assert contract["config"]["preflow_convergence_mode"] == "windowed_stationary"


def test_final_identity_rejects_interpolated_hibm_velocity_rows() -> None:
    manifest: dict[str, object] = {"config": _fine_solver_config(50)}
    config = manifest["config"]
    assert isinstance(config, dict)
    config["flow_hibm_sharp_interpolate_velocity_rows"] = True

    with pytest.raises(
        NativeFineComparisonError,
        match="flow_hibm_sharp_interpolate_velocity_rows",
    ):
        _validate_final_run_identity(
            manifest,
            _fine_solver_summary_identity(),
        )


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("flow_turbulence_model", "laminar"),
        ("flow_sst_near_wall_treatment", "fluent_correlation"),
        ("flow_symmetry_domain_walls", ["xmin", "xmax", "ymax"]),
        ("solid_constitutive_model", "linear_elastic"),
        ("young_modulus_pa", 9.0e5),
        ("poisson_ratio", 0.45),
        ("solid_density_kgm3", 1500.0),
        ("velocity_damping", 1.0),
    ],
)
def test_final_identity_locks_sst_and_structural_material_contracts(
    key: str,
    bad_value: object,
) -> None:
    manifest: dict[str, object] = {"config": _fine_solver_config(50)}
    config = manifest["config"]
    assert isinstance(config, dict)
    config[key] = bad_value

    with pytest.raises(NativeFineComparisonError, match=key):
        _validate_final_run_identity(
            manifest,
            _fine_solver_summary_identity(),
        )


def test_final_identity_rejects_anisotropic_pressure_probe_distance() -> None:
    manifest: dict[str, object] = {"config": _fine_solver_config(50)}
    config = manifest["config"]
    assert isinstance(config, dict)
    config["flow_hibm_sharp_interior_probe_distance_xyz_m"] = [
        1.125e-3,
        0.1171875e-3,
        0.46875e-3,
    ]

    with pytest.raises(
        NativeFineComparisonError,
        match="flow_hibm_sharp_interior_probe_distance_xyz_m",
    ):
        _validate_final_run_identity(
            manifest,
            _fine_solver_summary_identity(),
        )


def test_partial_native_comparison_rejects_interpolated_hibm_rows() -> None:
    with pytest.raises(
        NativeFineComparisonError,
        match="flow_hibm_sharp_interpolate_velocity_rows",
    ):
        _validate_run_contracts(
            {
                "config": {
                    "flow_hibm_sharp_interpolate_velocity_rows": True,
                    "flow_hibm_sharp_interior_probe_distance_m": 1.0e-3,
                    "flow_hibm_sharp_interior_probe_distance_xyz_m": None,
                }
            },
            {},
            {},
            {},
            expected_steps=1,
        )


def test_locked_native_fluent_identity_tracks_latest_fresh50_bundle() -> None:
    assert CANONICAL_NATIVE_FLUENT_PATH_MARKERS == (
        "official_fluent_fine_fsi_valid_2026-07-10",
        "fresh50_20260713_104843",
        "postprocess_compare31_strict_pressure_20260719_142808_r2",
    )


def test_latest_fresh50_path_passes_locked_bundle_path_gate() -> None:
    fresh_path = Path(
        "validation_runs/ansys_vertical_flap_fsi/"
        "official_fluent_fine_fsi_valid_2026-07-10/runs/"
        "fresh50_20260713_104843/"
        "postprocess_compare31_strict_pressure_20260719_142808_r2"
    )

    with pytest.raises(NativeFineComparisonError, match="wrong schema"):
        _validate_native_fluent_bundle(
            fresh_path,
            {},
            {},
            expected_steps=50,
        )


def test_superseded_50_step_bundle_fails_locked_bundle_path_gate() -> None:
    superseded_path = Path(
        "validation_runs/ansys_vertical_flap_fsi/"
        "official_fluent_fine_fsi_valid_2026-07-10/runs/"
        "native_production_hardened_50step_20260710_1613/"
        "postprocess_50step_20260710_1625"
    )

    with pytest.raises(NativeFineComparisonError, match="locked native Fluent bundle"):
        _validate_native_fluent_bundle(
            superseded_path,
            {},
            {},
            expected_steps=50,
        )


def test_path_marker_suffix_spoof_fails_locked_bundle_path_gate(
    tmp_path: Path,
) -> None:
    spoof_path = (
        tmp_path
        / "official_fluent_fine_fsi_valid_2026-07-10"
        / "fresh50_20260713_104843-copy"
        / "postprocess_compare31_20260713-spoof"
    )
    spoof_path.mkdir(parents=True)

    with pytest.raises(NativeFineComparisonError, match="locked native Fluent bundle"):
        _validate_native_fluent_bundle(
            spoof_path,
            {},
            {},
            expected_steps=50,
        )


def test_postprocess_cli_default_tracks_latest_fresh50_bundle() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "our_solver_vs_native_fluent_fine_2026-07-10"
        / "scripts"
        / "postprocess_our_solver_vs_native_fluent.py"
    )

    namespace = runpy.run_path(str(script_path))
    locked_path = namespace["LOCKED_NATIVE_FLUENT_POSTPROCESS_DIR"]

    assert isinstance(locked_path, Path)
    assert locked_path.relative_to(repo_root).as_posix() == (
        "validation_runs/ansys_vertical_flap_fsi/"
        "official_fluent_fine_fsi_valid_2026-07-10/runs/"
        "fresh50_20260713_104843/"
        "postprocess_compare31_strict_pressure_20260719_142808_r2"
    )


def test_postprocess_cli_returns_nonzero_when_five_percent_gate_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "our_solver_vs_native_fluent_fine_2026-07-10"
        / "scripts"
        / "postprocess_our_solver_vs_native_fluent.py"
    )
    namespace = runpy.run_path(str(script_path))
    main = namespace["main"]
    main.__globals__["postprocess_native_fine_comparison"] = lambda *_args, **_kwargs: {
        "five_percent_diagnostic_gate": {
            "status": "failed",
            "all_metrics_within_tolerance": False,
        }
    }

    exit_code = main(
        [
            "--our-run-dir",
            str(tmp_path / "solver"),
            "--fluent-postprocess-dir",
            str(tmp_path / "fluent"),
            "--output-dir",
            str(tmp_path / "comparison"),
        ]
    )

    assert exit_code != 0


def test_postprocess_cli_forwards_explicit_fluent_force_history(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "our_solver_vs_native_fluent_fine_2026-07-10"
        / "scripts"
        / "postprocess_our_solver_vs_native_fluent.py"
    )
    namespace = runpy.run_path(str(script_path))
    main = namespace["main"]
    captured: dict[str, object] = {}

    def _postprocess(*_args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "five_percent_diagnostic_gate": {
                "status": "passed",
                "all_metrics_within_tolerance": True,
            }
        }

    main.__globals__["postprocess_native_fine_comparison"] = _postprocess
    force_history = tmp_path / "history.csv"
    exit_code = main(
        [
            "--our-run-dir",
            str(tmp_path / "solver"),
            "--fluent-postprocess-dir",
            str(tmp_path / "fluent"),
            "--fluent-force-history",
            str(force_history),
            "--output-dir",
            str(tmp_path / "comparison"),
        ]
    )

    assert exit_code == 0
    assert captured["fluent_force_history_path"] == str(force_history)


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("config", "grid_nodes", [4, 64, 320]),
        ("config", "solid_particle_counts", [1, 64, 12]),
        ("config", "marker_count", 32),
        ("config", "flow_projection_iterations", 1079),
        ("config", "solid_substeps", 1599),
        ("config", "solid_density_kgm3", 1500.0),
        ("config", "young_modulus_pa", 9.0e5),
        ("config", "poisson_ratio", 0.45),
        ("config", "velocity_damping", 1.0),
        ("config", "solid_constitutive_model", "linear_elastic"),
        ("config", "flow_advection_scheme", "euler"),
        ("config", "flow_turbulence_model", "laminar"),
        ("config", "flow_sst_near_wall_treatment", "fluent_correlation"),
        ("config", "flow_predictor_substeps", 64),
        ("config", "flow_hibm_sharp_search_radius_m", 1.6e-3),
        (
            "config",
            "flow_hibm_sharp_search_radius_xyz_m",
            [1.2e-3, 0.3125e-3, 0.46875e-3],
        ),
        (
            "config",
            "flow_hibm_sharp_search_radius_xyz_m",
            [1.2e-3 + 9.0e-16, 0.390625e-3, 0.46875e-3],
        ),
        (
            "config",
            "flow_hibm_sharp_interior_probe_distance_xyz_m",
            [1.125e-3, 1.125e-3, 1.125e-3],
        ),
        ("config", "flow_hibm_sharp_interior_probe_distance_m", 1.0e-3),
        ("config", "solid_particle_counts", [True, 256, 20]),
        ("config", "flow_hibm_sharp_interpolate_velocity_rows", True),
        ("config", "flow_hibm_marker_mac_constraint_iterations", 63),
        ("config", "traction_tip_cap_pressure_enabled", True),
        ("config", "flow_hibm_dynamic_solid_volume_enabled", False),
        ("config", "update_fluid_obstacle_from_solid", False),
        ("config", "flow_hibm_tiny_unreached_cleanup_component_cells", 128),
        ("config", "preflow_steps", 40),
        ("config", "preflow_convergence_mode", "single_step_legacy"),
        ("config", "preflow_stationary_min_steps", 19),
        ("config", "preflow_stationary_window_steps", 9),
        ("config", "preflow_stationary_consecutive_windows", 2),
        ("config", "preflow_stationary_tolerance", 0.051),
        ("config", "preflow_stationary_divergence_tolerance", 0.051),
        (
            "config",
            "preflow_stationary_no_slip_tolerance_fraction",
            0.051,
        ),
        ("config", "flow_cg_preconditioner", "fv_multigrid_light"),
        ("config", "flow_cg_tolerance", 1.0e-5),
        ("config", "flow_cg_tolerance", 1.0e-6 + 9.0e-16),
        ("config", "flow_pressure_solve_failure_policy", "report"),
        (
            "config",
            "traction_pressure_pair_runtime_provider_mode",
            "replay_from_diagnostics",
        ),
        ("solver_npz_summary", "span_reduction", "center"),
        ("solver_npz_summary", "streamwise_velocity_sign", 1.0),
        ("solver_npz_summary", "reverse_streamwise_axis", False),
    ],
)
def test_final_50_step_native_fine_identity_rejects_drift(
    section: str,
    key: str,
    bad_value: object,
) -> None:
    manifest: dict[str, object] = {"config": _fine_solver_config(50)}
    summary: dict[str, object] = _fine_solver_summary_identity()
    target = manifest["config"] if section == "config" else summary[section]
    assert isinstance(target, dict)
    target[key] = bad_value

    with pytest.raises(
        NativeFineComparisonError,
        match=rf"final 50-step native-fine identity.*{key}",
    ):
        _validate_final_run_identity(manifest, summary)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_final_identity_rejects_nonfinite_float_inside_locked_vector(
    bad_value: float,
) -> None:
    manifest: dict[str, object] = {"config": _fine_solver_config(50)}
    config = manifest["config"]
    assert isinstance(config, dict)
    config["flow_hibm_sharp_search_radius_xyz_m"] = [
        bad_value,
        0.390625e-3,
        0.46875e-3,
    ]

    with pytest.raises(
        NativeFineComparisonError,
        match="flow_hibm_sharp_search_radius_xyz_m",
    ):
        _validate_final_run_identity(manifest, _fine_solver_summary_identity())


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("cg_exact_relative_residual_max", 1.0001e-6),
        ("cg_multigrid_to_jacobi_fallback_count", 1),
        ("cg_preconditioner_effective", "jacobi"),
        ("cg_preconditioner_requested", "fv_multigrid_light"),
        ("pressure_interface_matrix_row_invalid_count", 1),
        ("pressure_interface_matrix_row_active_count", 11),
        ("pressure_interface_matrix_row_count", 0),
    ],
)
def test_final_projection_success_rejects_nonexact_cg_or_invalid_row_list(
    key: str,
    bad_value: object,
) -> None:
    report: dict[str, object] = {
        "cg_exact_relative_residual_max": 9.0e-7,
        "cg_multigrid_to_jacobi_fallback_count": 0,
        "cg_preconditioner_effective": "fv_multigrid",
        "cg_preconditioner_requested": "fv_multigrid",
        "pressure_interface_matrix_row_active_count": 12,
        "pressure_interface_matrix_row_count": 12,
        "pressure_interface_matrix_row_invalid_count": 0,
    }
    report[key] = bad_value

    with pytest.raises(NativeFineComparisonError, match="final 50-step numerical success"):
        _validate_final_projection_success(
            {"flow_projection_report": report},
            path=Path("step_0001.json"),
        )


def test_partial_diagnostic_history_accepts_known_failed_baseline(
    tmp_path: Path,
) -> None:
    our_dir, _ = _synthetic_inputs(tmp_path, steps=12)
    _set_projection_status(
        our_dir,
        6,
        pressure_solve_failed=True,
        cg_converged_all=False,
    )
    _drop_aggregate_history_time(our_dir)

    contract = validate_partial_diagnostic_step_histories(
        discover_solver_step_histories(our_dir, expected_steps=12),
        read_typed_csv(our_dir / "our_solver_history.csv"),
        expected_steps=12,
        dt_s=5.0e-4,
    )

    assert contract["status"] == "passed"
    assert contract["numerical_success_required"] is False
    assert contract["pressure_solve_failure_count"] == 1
    assert contract["cg_nonconverged_step_count"] == 1


def test_postprocess_rejects_stale_scalar_geometry_aliases(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    stale = _solver_field(2)
    stale["solid_x_m"] = stale["solid_rest_x_m"].copy()
    stale["marker_x_m"] = stale["solid_rest_x_m"][[0, 2, 4]].copy()
    np.savez_compressed(our_dir / "step_fields" / "step_0002.npz", **stale)

    with pytest.raises(
        NativeFineComparisonError,
        match="scalar alias solid_x_m.*step_0002",
    ):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_gapped_fluent_residual_history(tmp_path: Path) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    residual_path = fluent_dir / "histories" / "residual_history.csv"
    with residual_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _write_csv(
        residual_path,
        [row for row in rows if int(row["snapshot_step"]) != 2],
    )
    _write_checksums(fluent_dir, FLUENT_CHECKSUM_RELATIVE_PATHS)

    with pytest.raises(NativeFineComparisonError, match="residual history steps"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_inconsistent_fluent_residual_summary(
    tmp_path: Path,
) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    summary_path = fluent_dir / "histories" / "residual_snapshot_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["sample_count"] = "3"
    _write_csv(summary_path, rows)
    _write_checksums(fluent_dir, FLUENT_CHECKSUM_RELATIVE_PATHS)

    with pytest.raises(NativeFineComparisonError, match="residual sample_count"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


def test_postprocess_rejects_missing_residual_equation_group_for_one_step(
    tmp_path: Path,
) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    residual_path = fluent_dir / "histories" / "residual_history.csv"
    summary_path = fluent_dir / "histories" / "residual_snapshot_summary.csv"
    with residual_path.open("r", encoding="utf-8", newline="") as handle:
        residual_rows = list(csv.DictReader(handle))
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    _write_csv(
        residual_path,
        [
            row
            for row in residual_rows
            if not (
                int(row["snapshot_step"]) == 2
                and row["equation"] == "x-velocity"
            )
        ],
    )
    _write_csv(
        summary_path,
        [
            row
            for row in summary_rows
            if not (int(row["step"]) == 2 and row["equation"] == "x-velocity")
        ],
    )
    _write_checksums(fluent_dir, FLUENT_CHECKSUM_RELATIVE_PATHS)

    with pytest.raises(NativeFineComparisonError, match="residual equation groups"):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("data_sha256", "0" * 64, "data_sha256 mismatch"),
        ("case_size_bytes", -1, "case_size_bytes mismatch"),
    ],
)
def test_postprocess_rejects_tampered_native_source_pair_identity(
    tmp_path: Path,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    our_dir, fluent_dir = _synthetic_inputs(tmp_path)
    manifest_path = fluent_dir / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pairs"][1][field] = bad_value
    _write_json(manifest_path, manifest)
    _write_checksums(fluent_dir, FLUENT_CHECKSUM_RELATIVE_PATHS)

    with pytest.raises(NativeFineComparisonError, match=message):
        postprocess_native_fine_comparison(
            our_dir,
            fluent_dir,
            tmp_path / "comparison_output",
            expected_steps=3,
        )
