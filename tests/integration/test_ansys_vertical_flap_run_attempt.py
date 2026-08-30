from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_CLI = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_fine_vs_fluent_2026-07-02"
    / "scripts"
    / "run_our_solver_vertical_flap.py"
)
GENERATION = "a" * 32
IDENTITY = {
    "config_sha256": "b" * 64,
    "source_sha256": "c" * 64,
    "geometry_sha256": "d" * 64,
}
SOURCE_HASHES = {"simulation_core/example.py": "e" * 64}


def _load_validation_cli():
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_run_attempt_under_test", VALIDATION_CLI
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resume_args(*, canonical_root: Path, attempt_root: Path) -> list[str]:
    return [
        str(VALIDATION_CLI),
        "--output-dir",
        str(attempt_root),
        "--resume-run-dir",
        str(canonical_root),
        "--steps",
        "2",
    ]


def _seed_canonical_evidence(canonical_root: Path) -> dict[str, bytes]:
    canonical_root.mkdir()
    evidence = {
        "our_solver_config.json": b'{"prior":"accepted"}\n',
        "checkpoint.json": b'{"head":"accepted-k2"}\n',
        "checkpoint/state.bin": b"accepted checkpoint state",
        "step_fields/step_0002.npz": b"accepted field frame",
        "step_history/step_0002.json": b'{"step":2}\n',
    }
    for relative_path, payload in evidence.items():
        target = canonical_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return evidence


def _tree_bytes(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _k2_head():
    return SimpleNamespace(
        generation=GENERATION,
        metadata={"identity": IDENTITY},
        accepted_step=2,
    )


def _successful_report() -> dict[str, object]:
    return {
        "history": [{"step": 1}, {"step": 2}],
        "profile_wall_time_enabled": False,
        "taichi_runtime_identity": {"arch": "cpu", "default_fp": "f32"},
        "final_flow_field_snapshot": {},
    }


def test_main_resumed_k2_to_k2_keeps_canonical_bytes_and_uses_derived_checkpoint(
    tmp_path: Path,
) -> None:
    module = _load_validation_cli()
    canonical_root = tmp_path / "canonical"
    attempt_root = tmp_path / "attempt"
    expected_canonical = _seed_canonical_evidence(canonical_root)
    captured_configs: list[object] = []

    def run_benchmark(config, **_kwargs):
        captured_configs.append(config)
        return _successful_report()

    with (
        patch.object(module, "_preflight_checkpoint_resume", return_value=_k2_head()),
        patch.object(
            module,
            "read_checkpoint_head",
            side_effect=AssertionError("main must reuse the preflight head"),
        ),
        patch.object(module, "_source_hashes", return_value=SOURCE_HASHES),
        patch.object(
            module,
            "_configure_taichi_offline_cache",
            return_value={"offline_cache_enabled": False},
        ),
        patch.object(module, "run_ansys_vertical_flap_benchmark", side_effect=run_benchmark),
        patch.object(
            module.sys,
            "argv",
            _resume_args(canonical_root=canonical_root, attempt_root=attempt_root),
        ),
    ):
        assert module.main() == 0

    assert _tree_bytes(canonical_root) == expected_canonical
    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert Path(config.fsi_checkpoint_input_path).resolve() == (
        canonical_root / "checkpoint"
    ).resolve()
    assert Path(config.fsi_checkpoint_output_path).resolve() == (
        canonical_root / "checkpoint"
    ).resolve()
    assert config.fsi_checkpoint_expected_generation == GENERATION

    metadata = json.loads((attempt_root / "metadata.json").read_text("utf-8"))
    assert metadata["format"] == "validation-run-attempt-v2"
    assert metadata["canonical"] == {"resolved_path": str(canonical_root.resolve())}
    assert metadata["checkpoint"] == {
        "generation": GENERATION,
        "identity": IDENTITY,
        "accepted_step": 2,
    }
    assert metadata["source_sha256"] == SOURCE_HASHES
    assert metadata["target_step"] == 2
    assert metadata["attempt"]["role"] == "resume"
    assert metadata["attempt"]["id"]
    assert json.loads((attempt_root / "progress.json").read_text("utf-8"))["status"] == "completed"
    expected_provenance = {
        "canonical_root": str(canonical_root.resolve()),
        "checkpoint_generation": GENERATION,
        "checkpoint_identity": IDENTITY,
        "accepted_step": 2,
        "artifact_root": str(canonical_root.resolve()),
    }
    manifest = json.loads((attempt_root / "run_manifest.json").read_text("utf-8"))
    assert manifest["artifact_root"] == str(canonical_root.resolve())
    assert manifest["resume_provenance"] == expected_provenance
    assert manifest["config"]["fsi_checkpoint_expected_generation"] == GENERATION
    summary = json.loads(
        (attempt_root / "our_solver_summary.json").read_text("utf-8")
    )
    assert summary["status"] == "completed"
    assert summary["output_dir"] == str(attempt_root.resolve())
    assert summary["artifact_root"] == str(canonical_root.resolve())
    assert summary["resume_provenance"] == expected_provenance
    assert summary["step_field_frame_count"] == 1
    assert not (attempt_root / "failure.json").exists()


def test_main_resumed_research_probe_summary_records_storage_roles(
    tmp_path: Path,
) -> None:
    module = _load_validation_cli()
    canonical_root = tmp_path / "canonical"
    attempt_root = tmp_path / "attempt"
    expected_canonical = _seed_canonical_evidence(canonical_root)
    report = {
        "status": "research_probe_terminal",
        "history": [{"step": 1}, {"step": 2}],
        "accepted_step_count": 2,
        "accepted_time_s": 0.001,
        "research_probe_wall_time_s": 0.25,
    }

    with (
        patch.object(module, "_preflight_checkpoint_resume", return_value=_k2_head()),
        patch.object(module, "_source_hashes", return_value=SOURCE_HASHES),
        patch.object(
            module,
            "_configure_taichi_offline_cache",
            return_value={"offline_cache_enabled": False},
        ),
        patch.object(module, "run_ansys_vertical_flap_benchmark", return_value=report),
        patch.object(
            module.sys,
            "argv",
            _resume_args(canonical_root=canonical_root, attempt_root=attempt_root),
        ),
    ):
        assert module.main() == 0

    assert _tree_bytes(canonical_root) == expected_canonical
    expected_provenance = {
        "canonical_root": str(canonical_root.resolve()),
        "checkpoint_generation": GENERATION,
        "checkpoint_identity": IDENTITY,
        "accepted_step": 2,
        "artifact_root": str(canonical_root.resolve()),
    }
    summary = json.loads(
        (attempt_root / "our_solver_summary.json").read_text("utf-8")
    )
    assert summary["status"] == "research_probe_terminal"
    assert summary["output_dir"] == str(attempt_root.resolve())
    assert summary["artifact_root"] == str(canonical_root.resolve())
    assert summary["resume_provenance"] == expected_provenance
    assert json.loads((attempt_root / "progress.json").read_text("utf-8"))[
        "status"
    ] == "research_probe_terminal"


def test_main_resumed_solver_failure_keeps_canonical_bytes_and_writes_failure_only_to_attempt(
    tmp_path: Path,
) -> None:
    module = _load_validation_cli()
    canonical_root = tmp_path / "canonical"
    attempt_root = tmp_path / "attempt"
    expected_canonical = _seed_canonical_evidence(canonical_root)

    with (
        patch.object(module, "_preflight_checkpoint_resume", return_value=_k2_head()),
        patch.object(module, "_source_hashes", return_value=SOURCE_HASHES),
        patch.object(
            module,
            "_configure_taichi_offline_cache",
            return_value={"offline_cache_enabled": False},
        ),
        patch.object(
            module,
            "run_ansys_vertical_flap_benchmark",
            side_effect=RuntimeError("new solver failure"),
        ),
        patch.object(
            module.sys,
            "argv",
            _resume_args(canonical_root=canonical_root, attempt_root=attempt_root),
        ),
    ):
        with pytest.raises(RuntimeError, match="new solver failure"):
            module.main()

    assert _tree_bytes(canonical_root) == expected_canonical
    metadata = json.loads((attempt_root / "metadata.json").read_text("utf-8"))
    assert metadata["format"] == "validation-run-attempt-v2"
    assert metadata["checkpoint"]["generation"] == GENERATION
    failure = json.loads((attempt_root / "failure.json").read_text("utf-8"))
    assert failure["error"] == "new solver failure"
    assert not (canonical_root / "failure.json").exists()
    assert not (canonical_root / "progress.json").exists()
    assert not (canonical_root / "our_solver_summary.json").exists()


def test_main_resume_preflight_rejection_leaves_canonical_untouched_without_attempt(
    tmp_path: Path,
) -> None:
    module = _load_validation_cli()
    canonical_root = tmp_path / "canonical"
    attempt_root = tmp_path / "attempt"
    expected_canonical = _seed_canonical_evidence(canonical_root)

    with (
        patch.object(
            module,
            "_preflight_checkpoint_resume",
            side_effect=ValueError("checkpoint source identity does not match current source"),
        ),
        patch.object(
            module.sys,
            "argv",
            _resume_args(canonical_root=canonical_root, attempt_root=attempt_root),
        ),
    ):
        with pytest.raises(ValueError, match="source identity does not match current source"):
            module.main()

    assert _tree_bytes(canonical_root) == expected_canonical
    assert not attempt_root.exists()
