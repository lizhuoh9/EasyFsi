from __future__ import annotations

from dataclasses import dataclass
import json
import importlib
import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.validation import run_turek_hron_fsi_campaign as campaign
from tools.validation import turek_hron_component_gate_contracts as contracts


RUNTIME_IDENTITY = {
    "requested_arch": "cuda",
    "actual_arch": "cuda",
    "default_fp": "f32",
    "random_seed": 0,
    "strict_arch_verified": True,
    "compiler_configuration": {
        "advanced_optimization": True,
        "cfg_optimization": True,
        "debug": False,
        "default_ip": "i32",
        "fast_math": True,
        "opt_level": 1,
        "taichi_version": "1.7.4",
    },
    "offline_cache_identity": {"enabled": True, "file_path": None},
}


def _provenance() -> dict[str, object]:
    return {
        "git": {"commit": "b" * 40, "dirty": False, "dirty_paths": []},
        "config": {"stage": "fsi1-s0"},
        "config_sha256": "c" * 64,
        "source_hashes": {"cases/turek_hron_fsi.py": "d" * 64},
        "source_sha256": "e" * 64,
        "config_source_sha256": "f" * 64,
        "host_numerics_identity": contracts.host_numerics_identity(),
        "host_numerics_identity_sha256": (
            contracts.host_numerics_identity_sha256(
                contracts.host_numerics_identity()
            )
        ),
    }


def _step(step: int, marker_count: int = 2) -> dict[str, np.ndarray]:
    reference = np.arange(marker_count * 3, dtype=np.float64).reshape(marker_count, 3)
    current = reference + float(step) * 1.0e-3
    return {
        "accepted_step": np.asarray(step, dtype=np.int64),
        "accepted_time_s": np.asarray(step * 0.005, dtype=np.float64),
        "marker_reference_position_m": reference,
        "marker_current_position_m": current,
        "marker_material_displacement_m": current - reference,
        "marker_velocity_mps": np.full((marker_count, 3), step, dtype=np.float64),
        "marker_normal": np.tile(np.asarray([[0.0, 1.0, 0.0]]), (marker_count, 1)),
        "marker_fixed_area_m2": np.full(marker_count, 0.01, dtype=np.float64),
        "marker_region_id": np.ones(marker_count, dtype=np.int32),
        "marker_order": np.arange(marker_count, dtype=np.int64),
        "marker_force_pre_solid_n": np.tile(
            np.asarray([[1.0, 2.0, 3.0]]) / marker_count,
            (marker_count, 1),
        ),
        "point_a_displacement_turek_xy_m": np.asarray([step, -step], dtype=np.float64),
        "point_a_velocity_turek_xy_mps": np.asarray([0.1, -0.1], dtype=np.float64),
        "beam_force_solver_xyz_n": np.asarray([1.0, 2.0, 3.0]),
        "cylinder_pressure_force_solver_xyz_n": np.asarray([4.0, 5.0, 6.0]),
        "cylinder_viscous_force_solver_xyz_n": np.asarray([7.0, 8.0, 9.0]),
        "total_force_solver_xyz_n": np.asarray([12.0, 15.0, 18.0]),
        "coupling_trial_count": np.asarray(3, dtype=np.int64),
        "coupling_rejected_trial_count": np.asarray(2, dtype=np.int64),
        "pressure_cg_iterations_total": np.asarray(30, dtype=np.int64),
        "pressure_matvec_count_total": np.asarray(36, dtype=np.int64),
        "fluid_solve_count": np.asarray(3, dtype=np.int64),
        "solid_macro_solve_count": np.asarray(3, dtype=np.int64),
        "mpm_substeps_executed_total": np.asarray(300, dtype=np.int64),
        "iqn_fallback_count": np.asarray(1, dtype=np.int64),
        "coupling_relative_residual_history": np.asarray([0.5, 0.1, 0.001]),
        "coupling_absolute_residual_history_mps": np.asarray([0.05, 0.01, 0.0001]),
        "coupling_update_mode_history": np.asarray(
            ["picard", "iqn_ils"], dtype="<U64"
        ),
        "iqn_rank_history": np.asarray([0, 1], dtype=np.int64),
        "iqn_condition_number_history": np.asarray([1.0, 2.0]),
        "iqn_fallback_reason_history": np.asarray(
            ["rank_deficient_history", "none"], dtype="<U64"
        ),
        "iqn_update_limited_history": np.asarray([False, True]),
    }


def test_fsi1_s0_spec_is_the_review_frozen_matrix() -> None:
    assert campaign.FSI1_S0_SPEC == {
        "stage": "fsi1-s0",
        "preset": "fsi1",
        "grid_nodes": (4, 48, 288),
        "dt_s": 0.005,
        "step_count": 1600,
        "markers_per_side": "auto",
        "markers_per_tip": "auto",
        "ib_anisotropic_envelope": True,
        "classify_far_internal_nodes": True,
        "flow_cg_preconditioner": "fv_multigrid",
        "flow_predictor_substeps": 1,
        "fluid_advection_scheme": "rk2",
        "flow_projection_iterations": 4000,
        "flow_cg_tolerance": 1.0e-6,
        "flow_reprojection_iterations": 1200,
        "flow_reprojection_cg_tolerance": 1.0e-4,
        "fsi_coupling_absolute_tolerance_mps": 1.0e-4,
        "solid_substeps": 100,
        "velocity_damping": 1.0,
        "marker_reseed_interval_steps": None,
    }
    assert campaign.expected_fsi1_s0_marker_count() == 112
    assert campaign._FORMAL_ACCEPTED_INTERFACE_CHUNK_SIZE == 1000


def test_formal_source_hashes_cover_direct_and_transitive_production_code() -> None:
    paths = set(campaign._source_paths())

    assert {
        "cases/turek_hron_fsi.py",
        "simulation_core/coupling/marker_seeding.py",
        "simulation_core/coupling/iqn_ils.py",
        "simulation_core/fluids/solver.py",
        "src/refactored/validation/turek_hron_fsi/references.py",
        "tools/validation/run_turek_hron_fsi_campaign.py",
        "requirements.txt",
    } <= paths


def test_formal_campaign_chunk_size_cannot_be_overridden() -> None:
    assert "chunk_size" not in inspect.signature(
        campaign.run_fsi1_s0_campaign
    ).parameters
    with pytest.raises(SystemExit):
        campaign._build_parser().parse_args(
            [
                "--output-root",
                "runs",
                "--label",
                "formal",
                "--chunk-size",
                "999",
            ]
        )


def test_output_claim_is_create_only_and_rejects_unsafe_labels(tmp_path: Path) -> None:
    claimed = campaign.claim_output_dir(tmp_path, "turek_fsi1_s0__r01")
    assert claimed.is_dir()
    with pytest.raises(FileExistsError):
        campaign.claim_output_dir(tmp_path, "turek_fsi1_s0__r01")
    with pytest.raises(ValueError, match="directory name"):
        campaign.claim_output_dir(tmp_path, "../escape")


def test_effective_config_binding_preserves_captured_source_identity() -> None:
    source = _provenance()
    effective = {
        "stage": "fsi1-s0",
        "effective_case_config": {"grid_nodes": [4, 48, 288]},
    }

    bound = campaign.bind_effective_config(source, effective)

    assert bound["git"] == source["git"]
    assert bound["source_hashes"] == source["source_hashes"]
    assert bound["host_numerics_identity"] == source["host_numerics_identity"]
    assert bound["config"] == effective
    assert bound["config_sha256"] != source["config_sha256"]


def test_formal_config_is_normalized_before_effective_binding() -> None:
    requested = object()
    normalized = object()

    class FakeCase:
        received: dict[str, object] | None = None

        @classmethod
        def fsi1_config(cls, **kwargs: object) -> object:
            cls.received = kwargs
            return requested

        @staticmethod
        def with_beam_surface_force_support(config: object) -> object:
            assert config is requested
            return normalized

    observed = campaign._build_effective_fsi1_s0_config(FakeCase)

    assert observed is normalized
    assert FakeCase.received == {
        key: value
        for key, value in campaign.FSI1_S0_SPEC.items()
        if key not in {"stage", "preset"}
    }


def test_formal_config_binds_probe_runtime_and_chunk_protocol() -> None:
    @dataclass(frozen=True)
    class Probe:
        min_step: int = 180

    @dataclass(frozen=True)
    class RuntimeRequest:
        arch: str = "cuda"
        strict_arch: bool = True

    payload = campaign._formal_config_payload(
        {"grid_nodes": [4, 48, 288]},
        mechanism_probe=Probe(),
        runtime_request=RuntimeRequest(),
    )

    assert payload["accepted_interface_chunk_size"] == 1000
    assert payload["mechanism_probe"] == {"min_step": 180}
    assert payload["taichi_runtime_request"] == {
        "arch": "cuda",
        "strict_arch": True,
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        (
            "point_a_velocity_turek_xy_mps",
            np.asarray(["1", "2"]),
        ),
        (
            "marker_velocity_mps",
            np.asarray([["1", "2", "3"], ["4", "5", "6"]]),
        ),
        ("marker_order", np.asarray([0.0, 1.0])),
        ("iqn_update_limited_history", np.asarray([0, 1])),
        (
            "coupling_update_mode_history",
            np.asarray([b"picard", b"iqn_ils"]),
        ),
    ),
)
def test_accepted_writer_rejects_wrong_field_dtypes(
    tmp_path: Path,
    field: str,
    bad_value: np.ndarray,
) -> None:
    writer = campaign.AcceptedInterfaceChunkWriter(
        tmp_path,
        chunk_size=1,
        expected_steps=1,
        provenance=_provenance(),
        marker_layout_sha256="a" * 64,
        taichi_runtime_identity=RUNTIME_IDENTITY,
        parent_checkpoint_lineage={"kind": "from_start", "parent": None},
    )
    record = _step(1)
    record[field] = bad_value

    with pytest.raises(ValueError, match="dtype"):
        writer.record(record)


def test_accepted_writer_publishes_exact_immutable_chunks_and_manifests(
    tmp_path: Path,
) -> None:
    writer = campaign.AcceptedInterfaceChunkWriter(
        tmp_path,
        chunk_size=3,
        expected_steps=5,
        provenance=_provenance(),
        marker_layout_sha256="a" * 64,
        taichi_runtime_identity=RUNTIME_IDENTITY,
        parent_checkpoint_lineage={"kind": "from_start", "parent": None},
    )

    for step in range(1, 6):
        writer.record(_step(step))
    manifests = writer.finalize()

    assert [path.name for path in manifests] == [
        "accepted_interface_000000_000002.manifest.json",
        "accepted_interface_000003_000004.manifest.json",
    ]
    first_npz = tmp_path / "accepted_interface_000000_000002.npz"
    second_npz = tmp_path / "accepted_interface_000003_000004.npz"
    with np.load(first_npz, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["accepted_step"], [1, 2, 3])
        assert archive["marker_current_position_m"].shape == (3, 2, 3)
        assert archive["marker_reference_position_m"].shape == (2, 3)
        first_names = set(archive.files)
    with np.load(second_npz, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["accepted_step"], [4, 5])
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["first_accepted_step"] == 1
    assert manifest["last_accepted_step"] == 3
    assert manifest["accepted_step_count"] == 3
    assert manifest["npz_sha256"] == campaign.sha256_file(first_npz)
    assert set(manifest["array_sha256"]) == first_names
    assert manifest["git"]["dirty"] is False
    assert manifest["host_numerics_identity"] == contracts.host_numerics_identity()
    assert manifest["host_numerics_identity_sha256"] == (
        contracts.host_numerics_identity_sha256(
            contracts.host_numerics_identity()
        )
    )
    assert manifest["parent_checkpoint_lineage"] == {
        "kind": "from_start",
        "parent": None,
    }


def test_accepted_writer_rejects_gaps_static_drift_and_double_finalize(
    tmp_path: Path,
) -> None:
    writer = campaign.AcceptedInterfaceChunkWriter(
        tmp_path,
        chunk_size=2,
        expected_steps=2,
        provenance=_provenance(),
        marker_layout_sha256="a" * 64,
        taichi_runtime_identity=RUNTIME_IDENTITY,
        parent_checkpoint_lineage={"kind": "from_start", "parent": None},
    )
    writer.record(_step(1))
    with pytest.raises(ValueError, match="continuous"):
        writer.record(_step(3))

    drifted = _step(2)
    drifted["marker_reference_position_m"] = (
        drifted["marker_reference_position_m"] + 1.0
    )
    drifted["marker_material_displacement_m"] = (
        drifted["marker_current_position_m"]
        - drifted["marker_reference_position_m"]
    )
    with pytest.raises(ValueError, match="static"):
        writer.record(drifted)

    writer.record(_step(2))
    writer.finalize()
    with pytest.raises(RuntimeError, match="already finalized"):
        writer.finalize()


def test_chunk_npz_is_published_before_manifest_and_manifest_failure_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = campaign.AcceptedInterfaceChunkWriter(
        tmp_path,
        chunk_size=1,
        expected_steps=1,
        provenance=_provenance(),
        marker_layout_sha256="a" * 64,
        taichi_runtime_identity=RUNTIME_IDENTITY,
        parent_checkpoint_lineage={"kind": "from_start", "parent": None},
    )
    atomic_file = importlib.import_module(
        "simulation_core.diagnostics.atomic_file"
    )
    original_publish = atomic_file.publish_file_create_only
    destinations: list[str] = []

    def fail_manifest(source: Path, destination: Path) -> None:
        destinations.append(Path(destination).name)
        if str(destination).endswith(".manifest.json"):
            raise OSError("injected manifest publication failure")
        original_publish(source, destination)

    monkeypatch.setattr(atomic_file, "publish_file_create_only", fail_manifest)

    with pytest.raises(OSError, match="injected"):
        writer.record(_step(1))

    assert destinations == [
        "accepted_interface_000000_000000.npz",
        "accepted_interface_000000_000000.manifest.json",
    ]
    assert (tmp_path / destinations[0]).is_file()
    assert not (tmp_path / destinations[1]).exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_existing_chunk_destination_is_never_overwritten(tmp_path: Path) -> None:
    writer = campaign.AcceptedInterfaceChunkWriter(
        tmp_path,
        chunk_size=1,
        expected_steps=1,
        provenance=_provenance(),
        marker_layout_sha256="a" * 64,
        taichi_runtime_identity=RUNTIME_IDENTITY,
        parent_checkpoint_lineage={"kind": "from_start", "parent": None},
    )
    destination = tmp_path / "accepted_interface_000000_000000.npz"
    destination.write_bytes(b"existing evidence")

    with pytest.raises(FileExistsError):
        writer.record(_step(1))

    assert destination.read_bytes() == b"existing evidence"
    assert not (tmp_path / "accepted_interface_000000_000000.manifest.json").exists()


def test_force_closure_uses_the_frozen_32_epsilon64_gate(tmp_path: Path) -> None:
    writer = campaign.AcceptedInterfaceChunkWriter(
        tmp_path,
        chunk_size=1,
        expected_steps=1,
        provenance=_provenance(),
        marker_layout_sha256="a" * 64,
        taichi_runtime_identity=RUNTIME_IDENTITY,
        parent_checkpoint_lineage={"kind": "from_start", "parent": None},
    )
    record = _step(1)
    record["total_force_solver_xyz_n"] = (
        record["total_force_solver_xyz_n"]
        + np.asarray([1.0e-12, 0.0, 0.0])
    )

    with pytest.raises(ValueError, match="total force"):
        writer.record(record)


def test_campaign_module_import_does_not_import_taichi() -> None:
    script = (
        "import sys; "
        "import tools.validation.run_turek_hron_fsi_campaign; "
        "raise SystemExit(1 if 'taichi' in sys.modules else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )

    assert completed.returncode == 0


def test_s0_gate_does_not_require_single_run_canonical_accuracy() -> None:
    report = {
        "status": "canonical_reference_failed",
        "numerical_contract_passed": True,
        "steady_state_passed": True,
        "acceptance_passed": False,
    }

    assert campaign.fsi1_s0_gate_passed(report) is True
    assert campaign.fsi1_s0_gate_passed(
        {**report, "steady_state_passed": False}
    ) is False


@pytest.mark.parametrize(
    ("phase_status", "error", "expected"),
    (
        (
            "BLOCKED_SOURCE_MISMATCH",
            RuntimeError("FAIL_PROVENANCE_DRIFT"),
            "BLOCKED_SOURCE_MISMATCH",
        ),
        (
            "BLOCKED_SOURCE_MISMATCH",
            ValueError("FAIL_HOST_NUMERICS_IDENTITY"),
            "BLOCKED_ENVIRONMENT",
        ),
        (
            "BLOCKED_ENVIRONMENT",
            RuntimeError("strict CUDA initialization failed"),
            "BLOCKED_ENVIRONMENT",
        ),
        (
            "FAIL_NUMERICAL_HEALTH",
            FloatingPointError("nonfinite pressure"),
            "FAIL_NUMERICAL_HEALTH",
        ),
        (
            "FAIL_NUMERICAL_HEALTH",
            OSError("artifact publication failed"),
            "BLOCKED_ENVIRONMENT",
        ),
        (
            "FAIL_NUMERICAL_HEALTH",
            ModuleNotFoundError("taichi"),
            "BLOCKED_ENVIRONMENT",
        ),
    ),
)
def test_campaign_failures_use_registered_boundary_classifications(
    phase_status: str,
    error: BaseException,
    expected: str,
) -> None:
    assert campaign._campaign_failure_status(
        error,
        phase_status=phase_status,
    ) == expected


def test_campaign_failure_artifact_preserves_original_diagnostics(
    tmp_path: Path,
) -> None:
    class Writer:
        accepted_count = 17

    error = FloatingPointError("nonfinite pressure")
    campaign._write_failure(
        tmp_path,
        error,
        _provenance(),
        Writer(),
        status="FAIL_NUMERICAL_HEALTH",
    )

    payload = json.loads(
        (tmp_path / "campaign_failure.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "FAIL_NUMERICAL_HEALTH"
    assert payload["error_type"] == "FloatingPointError"
    assert payload["error"] == "nonfinite pressure"
    assert payload["accepted_step_count"] == 17
    assert payload["provenance"] == _provenance()
    assert payload["host_numerics_identity"] == contracts.host_numerics_identity()
    assert payload["host_numerics_identity_sha256"] == (
        contracts.host_numerics_identity_sha256(
            contracts.host_numerics_identity()
        )
    )


def test_host_numerics_identity_is_captured_before_solver_imports() -> None:
    source = inspect.getsource(campaign.run_fsi1_s0_campaign)

    assert source.index("provenance = capture_provenance(FSI1_S0_SPEC)") < source.index(
        'case = importlib.import_module("cases.turek_hron_fsi")'
    )


def test_strict_runtime_is_initialized_before_the_numerical_run() -> None:
    source = inspect.getsource(campaign.run_fsi1_s0_campaign)
    initialize_index = source.index("runtime.init_taichi(runtime_request)")
    numerical_status_index = source.index(
        'failure_status = "FAIL_NUMERICAL_HEALTH"',
        initialize_index,
    )
    run_index = source.index("summary = case.run_turek_hron_fsi(")

    assert initialize_index < numerical_status_index < run_index


def test_accepted_observer_classifies_identity_before_record_content() -> None:
    source = inspect.getsource(campaign.run_fsi1_s0_campaign)
    observer_index = source.index("def accepted_step_observer(")
    source_status_index = source.index(
        'failure_status = "BLOCKED_SOURCE_MISMATCH"',
        observer_index,
    )
    identity_index = source.index(
        'record, "marker_layout_sha256"',
        source_status_index,
    )
    numerical_status_index = source.index(
        'failure_status = "FAIL_NUMERICAL_HEALTH"',
        identity_index,
    )
    record_index = source.index("writer.record(arrays)", numerical_status_index)

    assert (
        source_status_index
        < identity_index
        < numerical_status_index
        < record_index
    )


@pytest.mark.parametrize(
    ("s0_gate_passed", "expected_exit"),
    ((True, 0), (False, 2)),
)
def test_main_exit_code_tracks_only_the_s0_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    s0_gate_passed: bool,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        campaign,
        "run_fsi1_s0_campaign",
        lambda *_args, **_kwargs: {
            "status": "PASS_FSI1_S0_GATE_ONLY"
            if s0_gate_passed
            else "FAIL_FSI1_S0_GATE",
            "s0_gate_passed": s0_gate_passed,
        },
    )

    assert campaign.main(
        ["--output-root", str(tmp_path), "--label", "formal"]
    ) == expected_exit
