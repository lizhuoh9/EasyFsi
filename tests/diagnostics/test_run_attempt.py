from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from types import SimpleNamespace

import pytest

import simulation_core.diagnostics.run_attempt as run_attempt
from simulation_core.diagnostics.run_attempt import (
    prepare_resume_attempt,
    require_completed_output,
)


GENERATION = "a" * 32
IDENTITY = {
    "config_sha256": "b" * 64,
    "source_sha256": "c" * 64,
    "geometry_sha256": "d" * 64,
}
SOURCE_HASHES = {"simulation_core/example.py": "e" * 64}


def _completed_dual_root_attempt(tmp_path: Path, *, steps: int = 3) -> tuple[Path, Path]:
    canonical = tmp_path / "canonical"
    attempt = tmp_path / "attempt"
    canonical.mkdir()
    attempt.mkdir()
    provenance = {
        "canonical_root": str(canonical.resolve()),
        "checkpoint_generation": GENERATION,
        "checkpoint_identity": IDENTITY,
        "accepted_step": 1,
        "artifact_root": str(canonical.resolve()),
    }
    (attempt / "metadata.json").write_text(
        json.dumps(
            {
                "format": "validation-run-attempt-v2",
                "canonical": {"resolved_path": str(canonical.resolve())},
                "checkpoint": {
                    "generation": GENERATION,
                    "identity": IDENTITY,
                    "accepted_step": 1,
                },
                "source_sha256": SOURCE_HASHES,
                "target_step": steps,
                "attempt": {"id": "resume-1-to-3", "role": "resume"},
            }
        ),
        "utf-8",
    )
    (attempt / "run_manifest.json").write_text(
        json.dumps(
            {
                "artifact_root": str(canonical.resolve()),
                "source_sha256": SOURCE_HASHES,
                "resume_provenance": provenance,
            }
        ),
        "utf-8",
    )
    (attempt / "progress.json").write_text(
        json.dumps({"status": "completed", "step_completed": steps}), "utf-8"
    )
    (attempt / "our_solver_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_count_completed": steps,
                "output_dir": str(attempt.resolve()),
                "artifact_root": str(canonical.resolve()),
                "resume_provenance": provenance,
            }
        ),
        "utf-8",
    )
    return attempt, canonical


def test_completed_output_requires_no_active_terminal_artifact(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text('{"status":"completed"}', "utf-8")
    (tmp_path / "our_solver_summary.json").write_text(
        '{"status":"completed"}', "utf-8"
    )
    require_completed_output(tmp_path)

    for name in ("failure.json", "interruption.json"):
        (tmp_path / name).write_text("{}", "utf-8")
        with pytest.raises(ValueError, match="terminal-complete"):
            require_completed_output(tmp_path)
        (tmp_path / name).unlink()


def test_prepare_attempt_writes_v2_metadata_without_mutating_canonical(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    evidence = {
        "failure.json": b'{"status":"failed"}\n',
        "interruption.json": b'{"status":"interrupted"}\n',
        "our_solver_summary.json": b'{"status":"completed"}\n',
        "accepted_checkpoint.bin": b"accepted checkpoint bytes",
    }
    for name, payload in evidence.items():
        (canonical / name).write_bytes(payload)
    attempt = tmp_path / "resume-attempt"

    prepared = prepare_resume_attempt(
        canonical_root=canonical,
        attempt_root=attempt,
        checkpoint_generation=GENERATION,
        checkpoint_identity=IDENTITY,
        accepted_step=2,
        source_hashes=SOURCE_HASHES,
        target_step=2,
        attempt_id="resume-k2-to-k2",
        attempt_role="resume",
    )

    assert prepared == attempt
    assert {name: (canonical / name).read_bytes() for name in evidence} == evidence
    assert json.loads((attempt / "metadata.json").read_text("utf-8")) == {
        "format": "validation-run-attempt-v2",
        "canonical": {"resolved_path": str(canonical.resolve())},
        "checkpoint": {
            "generation": GENERATION,
            "identity": IDENTITY,
            "accepted_step": 2,
        },
        "source_sha256": SOURCE_HASHES,
        "target_step": 2,
        "attempt": {"id": "resume-k2-to-k2", "role": "resume"},
    }


def test_prepare_attempt_metadata_failure_preserves_canonical_and_empty_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    evidence = canonical / "failure.json"
    original = b"previous failure"
    evidence.write_bytes(original)
    attempt = tmp_path / "existing-empty-attempt"
    attempt.mkdir()

    monkeypatch.setattr(
        run_attempt,
        "publish_file_create_only",
        lambda *_args: (_ for _ in ()).throw(OSError("metadata write denied")),
    )

    with pytest.raises(OSError, match="metadata write denied"):
        prepare_resume_attempt(
            canonical_root=canonical,
            attempt_root=attempt,
            checkpoint_generation=GENERATION,
            checkpoint_identity=IDENTITY,
            accepted_step=2,
            source_hashes=SOURCE_HASHES,
            target_step=3,
            attempt_id="resume-k2-to-k3",
            attempt_role="resume",
        )

    assert evidence.read_bytes() == original
    assert list(attempt.iterdir()) == []


def test_prepare_attempt_removes_new_empty_root_when_metadata_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    attempt = tmp_path / "new-attempt"

    monkeypatch.setattr(
        run_attempt,
        "publish_file_create_only",
        lambda *_args: (_ for _ in ()).throw(OSError("metadata write denied")),
    )

    with pytest.raises(OSError, match="metadata write denied"):
        prepare_resume_attempt(
            canonical_root=canonical,
            attempt_root=attempt,
            checkpoint_generation=GENERATION,
            checkpoint_identity=IDENTITY,
            accepted_step=2,
            source_hashes=SOURCE_HASHES,
            target_step=3,
            attempt_id="resume-k2-to-k3",
            attempt_role="resume",
        )

    assert not attempt.exists()


def test_prepare_attempt_concurrent_claim_allows_one_metadata_owner(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    evidence = canonical / "accepted_checkpoint.bin"
    original = b"accepted checkpoint bytes"
    evidence.write_bytes(original)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    barrier = threading.Barrier(2)
    original_iterdir = Path.iterdir

    def synchronized_iterdir(path: Path):
        entries = tuple(original_iterdir(path))
        if path == attempt:
            barrier.wait(timeout=5)
        return iter(entries)

    monkeypatch.setattr(Path, "iterdir", synchronized_iterdir)
    successes: list[Path] = []
    failures: list[BaseException] = []

    def claim(attempt_id: str) -> None:
        try:
            successes.append(
                prepare_resume_attempt(
                    canonical_root=canonical,
                    attempt_root=attempt,
                    checkpoint_generation=GENERATION,
                    checkpoint_identity=IDENTITY,
                    accepted_step=2,
                    source_hashes=SOURCE_HASHES,
                    target_step=3,
                    attempt_id=attempt_id,
                    attempt_role="resume",
                )
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=claim, args=(attempt_id,))
        for attempt_id in ("resume-a", "resume-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    monkeypatch.setattr(Path, "iterdir", original_iterdir)
    assert not any(thread.is_alive() for thread in threads)
    assert successes == [attempt]
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    metadata = json.loads((attempt / "metadata.json").read_text("utf-8"))
    assert metadata["attempt"]["id"] in {"resume-a", "resume-b"}
    assert list(attempt.iterdir()) == [attempt / "metadata.json"]
    assert evidence.read_bytes() == original


@pytest.mark.skipif(os.name != "nt", reason="requires Windows UNC rename semantics")
def test_windows_unc_create_only_metadata_claim_never_overwrites() -> None:
    wsl_tmp = Path(r"\\wsl.localhost\Ubuntu-22.04\tmp")
    if not wsl_tmp.is_dir():
        pytest.skip("Ubuntu WSL temporary directory is unavailable")
    with tempfile.TemporaryDirectory(dir=wsl_tmp) as directory:
        metadata_path = Path(directory) / "metadata.json"
        first = {"attempt": "first"}
        run_attempt._write_json_create_only(metadata_path, first)
        with pytest.raises(FileExistsError):
            run_attempt._write_json_create_only(metadata_path, {"attempt": "second"})
        assert json.loads(metadata_path.read_text("utf-8")) == first


def test_create_only_writer_removes_hard_link_source_temp(
    tmp_path: Path, monkeypatch
) -> None:
    metadata_path = tmp_path / "metadata.json"
    payload = {"attempt": "first"}

    monkeypatch.setattr(
        run_attempt,
        "publish_file_create_only",
        lambda source, destination: os.link(source, destination),
    )
    run_attempt._write_json_create_only(metadata_path, payload)

    assert json.loads(metadata_path.read_text("utf-8")) == payload
    assert not list(tmp_path.glob(".metadata.json.*.tmp"))


def test_prepare_attempt_rejects_target_before_accepted_step_before_publication(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    evidence = canonical / "accepted_checkpoint.bin"
    evidence.write_bytes(b"accepted checkpoint bytes")
    attempt = tmp_path / "attempt"

    with pytest.raises(ValueError, match="target step must not precede accepted step"):
        prepare_resume_attempt(
            canonical_root=canonical,
            attempt_root=attempt,
            checkpoint_generation=GENERATION,
            checkpoint_identity=IDENTITY,
            accepted_step=3,
            source_hashes=SOURCE_HASHES,
            target_step=2,
            attempt_id="resume-k3-to-k2",
            attempt_role="resume",
        )

    assert evidence.read_bytes() == b"accepted checkpoint bytes"
    assert not attempt.exists()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("canonical_root", True, TypeError),
        ("checkpoint_generation", None, ValueError),
        ("checkpoint_identity", {}, ValueError),
        ("checkpoint_identity", {"source_sha256": "c" * 64, "geometry_sha256": "d" * 64}, ValueError),
        ("checkpoint_identity", {"config_sha256": "b" * 64, "source_sha256": "c" * 64}, ValueError),
        ("checkpoint_identity", {**IDENTITY, "unexpected_sha256": "f" * 64}, ValueError),
        ("accepted_step", True, TypeError),
        ("source_hashes", {"source.py": "bad"}, ValueError),
        ("target_step", True, TypeError),
        ("attempt_id", "../escape", ValueError),
        ("attempt_role", "../escape", ValueError),
    ],
)
def test_prepare_attempt_rejects_untrusted_metadata_before_publication(
    tmp_path: Path, field: str, value: object, expected: type[Exception]
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    evidence = canonical / "failure.json"
    evidence.write_bytes(b"previous failure")
    attempt = tmp_path / "attempt"
    kwargs = {
        "canonical_root": canonical,
        "attempt_root": attempt,
        "checkpoint_generation": GENERATION,
        "checkpoint_identity": IDENTITY,
        "accepted_step": 2,
        "source_hashes": SOURCE_HASHES,
        "target_step": 3,
        "attempt_id": "resume-k2-to-k3",
        "attempt_role": "resume",
    }
    kwargs[field] = value

    with pytest.raises(expected):
        prepare_resume_attempt(**kwargs)
    assert evidence.read_bytes() == b"previous failure"
    assert not attempt.exists()


def test_prepare_attempt_rejects_nonempty_or_canonical_attempt_root(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    nonempty_attempt = tmp_path / "attempt"
    nonempty_attempt.mkdir()
    (nonempty_attempt / "old.json").write_text("{}", "utf-8")

    kwargs = {
        "canonical_root": canonical,
        "checkpoint_generation": GENERATION,
        "checkpoint_identity": IDENTITY,
        "accepted_step": 2,
        "source_hashes": SOURCE_HASHES,
        "target_step": 3,
        "attempt_id": "resume-k2-to-k3",
        "attempt_role": "resume",
    }
    with pytest.raises(ValueError, match="empty"):
        prepare_resume_attempt(attempt_root=nonempty_attempt, **kwargs)
    with pytest.raises(ValueError, match="differ"):
        prepare_resume_attempt(attempt_root=canonical, **kwargs)
    assert not (nonempty_attempt / "metadata.json").exists()


def test_prepare_attempt_rejects_symlink_canonical_root(tmp_path: Path) -> None:
    real_canonical = tmp_path / "canonical"
    real_canonical.mkdir()
    canonical = tmp_path / "canonical-link"
    try:
        canonical.symlink_to(real_canonical, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"host does not permit symlink test: {exc}")

    with pytest.raises(ValueError, match="real directory"):
        prepare_resume_attempt(
            canonical_root=canonical,
            attempt_root=tmp_path / "attempt",
            checkpoint_generation=GENERATION,
            checkpoint_identity=IDENTITY,
            accepted_step=2,
            source_hashes=SOURCE_HASHES,
            target_step=3,
            attempt_id="resume-k2-to-k3",
            attempt_role="resume",
        )


def test_dual_root_attempt_requires_v2_metadata_and_distinct_real_roots(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    attempt = tmp_path / "attempt"
    canonical.mkdir()
    attempt.mkdir()
    (attempt / "progress.json").write_text(
        '{"status":"completed","step_completed":3}', "utf-8"
    )
    (attempt / "our_solver_summary.json").write_text(
        '{"status":"completed","step_count_completed":3}', "utf-8"
    )

    with pytest.raises(ValueError, match="validation-run-attempt-v2"):
        run_attempt.validate_dual_root_attempt_provenance(
            attempt_root=attempt,
            canonical_artifact_root=canonical,
            expected_steps=3,
        )


def test_dual_root_attempt_requires_matching_control_plane_provenance(
    tmp_path: Path,
) -> None:
    attempt, canonical = _completed_dual_root_attempt(tmp_path)
    manifest_path = attempt / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["source_sha256"] = {"simulation_core/example.py": "f" * 64}
    manifest_path.write_text(json.dumps(manifest), "utf-8")

    with pytest.raises(ValueError, match="source SHA256"):
        run_attempt.validate_dual_root_attempt_provenance(
            attempt_root=attempt,
            canonical_artifact_root=canonical,
            expected_steps=3,
        )


def _dual_root_history_rows() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    nan_fields = {
        f"zmin_unreached_source_{stat}_{axis}_m": float("nan")
        for stat in ("centroid", "min", "max")
        for axis in ("x", "y", "z")
    }
    journal = {
        "step": 7,
        "time_s": 3.5e-3,
        "tip_mean_displacement_m": (0.0, 1.0e-5, -2.0e-5),
        "max_displacement_m": 3.0e-5,
        "flow_projection_cg_converged_all": True,
        "diagnostic_empty_text": "",
        "total_marker_force_n": (0.0, 0.0, -1.0e-3),
        "flow_projection_report": {
            "zmin_unreached_source_cell_count": 0,
            "zmin_unreached_source_volume_flux_m3s": 0.0,
            "zmin_unreached_source_abs_flux_m3s": 0.0,
            **nan_fields,
        },
        "marker_action_reaction_residual_n": 1.0e-12,
        "scatter_action_reaction_residual_n": 2.0e-12,
    }
    step_json = {
        **journal,
        "tip_mean_displacement_m": [0.0, 1.0e-5, -2.0e-5],
        "total_marker_force_n": [0.0, 0.0, -1.0e-3],
    }
    step_json.pop("marker_action_reaction_residual_n")
    step_json.pop("scatter_action_reaction_residual_n")
    csv_row = {
        **step_json,
        "diagnostic_empty_text": None,
        "flow_projection_cg_converged_all": "True",
    }
    return journal, step_json, csv_row


def test_dual_root_history_binding_rejects_modified_noncore_field(
    tmp_path: Path,
) -> None:
    del tmp_path
    journal, step_json, csv_row = _dual_root_history_rows()
    csv_row["flow_projection_cg_converged_all"] = False

    with pytest.raises(ValueError, match="flow_projection_cg_converged_all.*step 7"):
        run_attempt.validate_dual_root_history_row_semantics(
            journal_history_row=journal,
            step_history_row=step_json,
            aggregate_csv_row=csv_row,
            step=7,
        )


def test_dual_root_history_binding_accepts_only_known_aliases_and_sentinels() -> None:
    journal, step_json, csv_row = _dual_root_history_rows()

    contract = run_attempt.validate_dual_root_history_row_semantics(
        journal_history_row=journal,
        step_history_row=step_json,
        aggregate_csv_row=csv_row,
        step=7,
    )

    assert contract["status"] == "passed"
    assert contract["public_field_count"] == len(step_json)
    assert contract["journal_only_aliases"] == [
        "marker_action_reaction_residual_n",
        "scatter_action_reaction_residual_n",
    ]


def test_dual_root_history_binding_rejects_csv_boolean_spelling_in_step_json() -> None:
    journal, step_json, csv_row = _dual_root_history_rows()
    step_json["flow_projection_cg_converged_all"] = "True"

    with pytest.raises(ValueError, match="flow_projection_cg_converged_all.*step 7"):
        run_attempt.validate_dual_root_history_row_semantics(
            journal_history_row=journal,
            step_history_row=step_json,
            aggregate_csv_row=csv_row,
            step=7,
        )


def test_prepare_attempt_rejects_non_real_attempt_root(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    attempt = tmp_path / "attempt-link"
    try:
        attempt.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"host does not permit symlink test: {exc}")

    with pytest.raises(ValueError, match="real directory"):
        prepare_resume_attempt(
            canonical_root=canonical,
            attempt_root=attempt,
            checkpoint_generation=GENERATION,
            checkpoint_identity=IDENTITY,
            accepted_step=2,
            source_hashes=SOURCE_HASHES,
            target_step=3,
            attempt_id="resume-k2-to-k3",
            attempt_role="resume",
        )


def test_windows_reparse_terminal_entry_is_not_treated_as_absent(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "progress.json").write_text('{"status":"completed"}', "utf-8")
    (tmp_path / "our_solver_summary.json").write_text(
        '{"status":"completed"}', "utf-8"
    )
    original_lstat = Path.lstat

    def lstat_with_reparse(path: Path):
        if path.name == "failure.json":
            return SimpleNamespace(
                st_mode=stat.S_IFREG,
                st_file_attributes=getattr(
                    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400
                ),
            )
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse)
    with pytest.raises(ValueError, match="terminal-complete"):
        require_completed_output(tmp_path)
