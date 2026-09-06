"""CPU-only case failure persistence across the real generic transaction guard."""
from __future__ import annotations

import ast
import csv
import functools
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cases import turek_hron_fsi as turek
from simulation_core.drivers.generic_fsi_solver import FsiCouplingConfig, FsiSolverConfig


class _State:
    def __init__(self, value: float):
        self.value = value
        self.saved = None
        self.restore_error = None

    def save_state(self):
        self.saved = self.value

    def restore_state(self):
        if self.restore_error is not None:
            raise self.restore_error
        self.value = self.saved


def _runtime(monkeypatch, *, phase, error, rollback_error=None, cause=None):
    fluid, solid = _State(1.0), _State(2.0)
    history, published = [], []
    marker = {"v_gamma_mps": np.zeros((1, 3))}
    monkeypatch.setattr(turek, "capture_marker_interface_state", lambda markers: marker)
    monkeypatch.setattr(turek, "restore_marker_interface_state", lambda *args: None)
    monkeypatch.setattr(turek, "marker_trial_state", lambda *args: marker)
    monkeypatch.setattr(turek, "_marker_pressure_neumann_gradient_state", lambda *args: np.zeros((1, 3)))
    monkeypatch.setattr(turek, "_restore_marker_pressure_neumann_gradient_state", lambda *args: None)
    monkeypatch.setattr(turek, "_fsi_coupling_marker_candidate_from_step_base", lambda **kwargs: marker)

    def fail():
        fluid.restore_error = rollback_error
        if cause is not None:
            raise error from cause
        raise error

    def prepare(context):
        if phase == "begin_step" and context.step == 8:
            fluid.value, solid.value = -100.0, -200.0
            fail()

    def advance(context, trial_index):
        fluid.value += 10.0
        solid.value += 20.0
        if phase == "evaluate_trial" and context.step == 8:
            fail()
        return object(), {}

    def commit(context, trial, coupling):
        if phase == "commit_step" and context.step == 8:
            fail()
        row = dict.fromkeys(turek.HISTORY_FIELDS, 0)
        row.update(step=context.step, time_s=context.time_s)
        history.append(row)
        return row

    def publish(context, row):
        if phase == "publish_step" and context.step == 8:
            fail()
        published.append(context.step)

    def finalize():
        if phase == "finalize_run":
            fail()
        return {}

    runtime = turek._TurekHronFsiRuntime(
        fluid=fluid, solid=solid, markers=SimpleNamespace(), boundary=SimpleNamespace(),
        advance_trial=advance, prepare_step=prepare, restore_case_boundaries=lambda context: None,
        commit_case_step=commit, finalize_case_run=finalize, publish_case_step=publish,
    )
    return runtime, history, published, fluid, solid


def _boundary_namespace(runtime, history, output_dir, **overrides):
    namespace = dict(vars(turek))
    namespace.update(
        case_runtime=runtime,
        solver_config=FsiSolverConfig(step_count=8, time_step_s=0.005,
                                      coupling=FsiCouplingConfig(max_iterations=2)),
        config=turek.fsi1_config(step_count=8), preset="fsi1", history=history,
        output_dir=output_dir,
        incremental_history_path=output_dir / "turek_hron_fsi_history.csv",
        last_flushed_index=0, incremental_header_written=False,
        pending_mechanism_failure_payload=None,
    )
    namespace.update(overrides)
    return namespace


def _execute_actual_case_exception_guard(namespace):
    # Execute the actual production try/except block with fake fields, so this
    # regression fails if the generic fallback handler is absent or miswired.
    # This avoids initializing the unrelated fluid/solid/Taichi case setup.
    tree = ast.parse(inspect.getsource(turek.run_turek_hron_fsi))
    guards = [
        node for node in ast.walk(tree) if isinstance(node, ast.Try)
        and any(isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "solve_fsi_runtime"
                for statement in node.body)
    ]
    assert len(guards) == 1
    module = ast.Module(body=[guards[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "actual_turek_exception_guard", "exec"), namespace)


def _execute_actual_final_output_guard(namespace):
    """Execute the production final-output guard without solver setup."""

    tree = ast.parse(inspect.getsource(turek.run_turek_hron_fsi))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "write_text"
            for call in ast.walk(node)
        )
    ]
    assert len(guards) == 1
    wrapper = ast.FunctionDef(
        name="_execute_final_output_guard",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[guards[0]],
        decorator_list=[],
    )
    module = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "actual_turek_final_output_guard", "exec"), namespace)
    namespace["_execute_final_output_guard"]()


def _evidence(path: Path):
    return json.loads((path / "turek_hron_fsi_coupling_failure.json").read_text())


def _csv_rows(path: Path):
    with (path / "turek_hron_fsi_history.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("phase", ["begin_step", "evaluate_trial", "commit_step"])
def test_numerical_exception_flushes_only_accepted_prefix_after_real_rollback(tmp_path, monkeypatch, phase):
    original = RuntimeError("hard-target closure failed")
    runtime, history, published, fluid, solid = _runtime(monkeypatch, phase=phase, error=original)
    namespace = _boundary_namespace(runtime, history, tmp_path)
    with pytest.raises(RuntimeError, match="hard-target closure") as caught:
        _execute_actual_case_exception_guard(namespace)
    assert caught.value is original
    assert (fluid.value, solid.value) == (71.0, 142.0)
    assert published == list(range(1, 8))
    assert [int(row["step"]) for row in _csv_rows(tmp_path)] == list(range(1, 8))
    evidence = _evidence(tmp_path)
    assert evidence["failure_phase"] == phase
    assert evidence["failed_step"] == 8 and evidence["failed_time_s"] == 0.04
    assert evidence["completed_steps"] == evidence["completed_history_rows_flushed"] == 7
    assert evidence["last_accepted_step"] == 7 and evidence["last_accepted_time_s"] == 0.035
    assert evidence["physical_state_restored"] is True
    assert evidence["rollback_outcome"] == "restored" and evidence["rollback_failure"] is None
    assert evidence["fsi_coupling_certificate_available"] is False
    assert "fsi_coupling_certificate" not in evidence and "candidate_row" not in evidence
    assert "iteration_budget_exhausted" not in json.dumps(evidence)
    assert "hard-target closure failed" in "".join(evidence["error_traceback"])
    assert not list(tmp_path.glob("*.tmp"))


def test_original_exception_cause_is_not_misidentified_as_rollback_failure(tmp_path, monkeypatch):
    original, cause = RuntimeError("closure"), ValueError("factorization input")
    runtime, history, _, _, _ = _runtime(monkeypatch, phase="evaluate_trial", error=original, cause=cause)
    with pytest.raises(RuntimeError) as caught:
        _execute_actual_case_exception_guard(_boundary_namespace(runtime, history, tmp_path))
    assert caught.value is original and caught.value.__cause__ is cause
    evidence = _evidence(tmp_path)
    assert evidence["physical_state_restored"] is True and evidence["rollback_failure"] is None
    assert evidence["exception_cause"] == "ValueError:factorization input"


def test_rollback_failure_preserves_original_error_and_records_failed_restore(tmp_path, monkeypatch):
    original, rollback = RuntimeError("closure"), OSError("restore failed")
    runtime, history, _, _, _ = _runtime(monkeypatch, phase="evaluate_trial", error=original, rollback_error=rollback)
    with pytest.raises(RuntimeError) as caught:
        _execute_actual_case_exception_guard(_boundary_namespace(runtime, history, tmp_path))
    assert caught.value is original and caught.value.__cause__ is rollback
    evidence = _evidence(tmp_path)
    assert evidence["physical_state_restored"] is False
    assert evidence["rollback_outcome"] == "failed" and evidence["rollback_failure"] == "OSError:restore failed"
    assert evidence["restored_state_scope"] is None
    assert len(_csv_rows(tmp_path)) == 7


@pytest.mark.parametrize("phase", ["publish_step", "finalize_run"])
def test_post_commit_failure_keeps_committed_step_and_does_not_claim_rollback(tmp_path, monkeypatch, phase):
    original = OSError("output failed")
    runtime, history, _, fluid, solid = _runtime(monkeypatch, phase=phase, error=original)
    with pytest.raises(OSError) as caught:
        _execute_actual_case_exception_guard(_boundary_namespace(runtime, history, tmp_path))
    assert caught.value is original
    assert (fluid.value, solid.value) == (81.0, 162.0)
    evidence = _evidence(tmp_path)
    assert evidence["completed_steps"] == evidence["last_accepted_step"] == 8
    assert evidence["physical_state_restored"] is None
    assert evidence["rollback_outcome"] == "not_attempted" and evidence["rollback_failure"] is None
    assert evidence["failed_step"] == (8 if phase == "publish_step" else None)
    assert evidence["failed_time_s"] == (0.04 if phase == "publish_step" else None)
    assert len(_csv_rows(tmp_path)) == 8


def test_no_complete_snapshot_reports_unknown_restore(tmp_path, monkeypatch):
    original = OSError("snapshot failed")
    runtime, history, _, fluid, _ = _runtime(monkeypatch, phase="evaluate_trial", error=original)
    def fail_save():
        raise original
    fluid.save_state = fail_save
    with pytest.raises(OSError) as caught:
        _execute_actual_case_exception_guard(_boundary_namespace(runtime, history, tmp_path))
    assert caught.value is original
    evidence = _evidence(tmp_path)
    assert evidence["failed_step"] == 1 and evidence["completed_steps"] == 0
    assert evidence["physical_state_restored"] is None
    assert evidence["rollback_outcome"] == "no_complete_snapshot"


@pytest.mark.parametrize("history_fails,artifact_fails", [(True, False), (False, True), (True, True)])
def test_persistence_io_errors_cannot_replace_original_exception(tmp_path, monkeypatch, history_fails, artifact_fails):
    original = RuntimeError("closure")
    runtime, history, _, _, _ = _runtime(monkeypatch, phase="evaluate_trial", error=original)
    def history_writer(*args, **kwargs):
        if history_fails:
            raise OSError("history disk full")
        return turek._flush_history_csv(*args, **kwargs)
    def artifact_writer(*args, **kwargs):
        if artifact_fails:
            raise OSError("artifact disk full")
        return turek._write_fsi_coupling_failure_artifact(*args, **kwargs)
    namespace = _boundary_namespace(runtime, history, tmp_path,
        _persist_runtime_failure_evidence=functools.partial(
            turek._persist_runtime_failure_evidence, history_writer=history_writer, artifact_writer=artifact_writer))
    with pytest.raises(RuntimeError) as caught:
        _execute_actual_case_exception_guard(namespace)
    assert caught.value is original and caught.value.__cause__ is None
    names = {item.split(":")[0] for item in namespace["_persistence_errors"]}
    assert names == ({"history_flush"} if history_fails else set()) | ({"failure_artifact"} if artifact_fails else set())
    if not artifact_fails:
        evidence = _evidence(tmp_path)
        assert evidence["completed_history_rows_flushed"] == 0
        assert evidence["persistence_errors"] == ["history_flush:OSError:history disk full"]
    if not history_fails:
        assert len(_csv_rows(tmp_path)) == 7


def test_failure_flushes_history_even_when_periodic_flush_is_disabled(tmp_path, monkeypatch):
    original = RuntimeError("closure")
    runtime, history, _, _, _ = _runtime(monkeypatch, phase="evaluate_trial", error=original)
    namespace = _boundary_namespace(runtime, history, tmp_path, incremental_history_path=None)
    with pytest.raises(RuntimeError):
        _execute_actual_case_exception_guard(namespace)
    assert len(_csv_rows(tmp_path)) == 7


def test_failure_appends_only_unflushed_accepted_rows(tmp_path, monkeypatch):
    original = RuntimeError("closure")
    runtime, history, _, _, _ = _runtime(monkeypatch, phase="evaluate_trial", error=original)
    def persist(error, **kwargs):
        path = kwargs["incremental_history_path"]
        turek._flush_history_csv(path, history[:3], header_written=False)
        kwargs.update(last_flushed_index=3, incremental_header_written=True)
        return turek._persist_runtime_failure_evidence(error, **kwargs)
    namespace = _boundary_namespace(runtime, history, tmp_path, _persist_runtime_failure_evidence=persist)
    with pytest.raises(RuntimeError):
        _execute_actual_case_exception_guard(namespace)
    assert [int(row["step"]) for row in _csv_rows(tmp_path)] == list(range(1, 8))


def test_snapshot_failure_after_successful_periodic_flush_rewrites_full_history(
    tmp_path,
    monkeypatch,
):
    """A stale in-memory cursor cannot duplicate accepted rows on recovery."""

    original = OSError("snapshot failed after periodic history flush")
    runtime, history, _, _, _ = _runtime(
        monkeypatch,
        phase="evaluate_trial",
        error=original,
    )
    history_path = tmp_path / "turek_hron_fsi_history.csv"
    def persist_after_stale_publish(error, **kwargs):
        # The generic handler runs after steps 1--7 have committed.  Mirror a
        # completed publish callback: rows 1--3 were durable, then its actual
        # append wrote rows 4--7 before snapshot publication failed and before
        # its in-memory cursor advanced from 3 to 7.
        turek._flush_history_csv(history_path, history[:3], header_written=False)
        turek._flush_history_csv(history_path, history[3:], header_written=True)
        kwargs.update(last_flushed_index=3, incremental_header_written=True)
        return turek._persist_runtime_failure_evidence(error, **kwargs)

    namespace = _boundary_namespace(
        runtime,
        history,
        tmp_path,
        incremental_history_path=history_path,
        _persist_runtime_failure_evidence=persist_after_stale_publish,
    )
    with pytest.raises(OSError) as caught:
        _execute_actual_case_exception_guard(namespace)
    assert caught.value is original
    assert [int(row["step"]) for row in _csv_rows(tmp_path)] == list(range(1, 8))
    evidence = _evidence(tmp_path)
    assert evidence["completed_history_rows_flushed"] == 7


def test_atomic_failure_writer_leaves_existing_history_bytes_unchanged(
    tmp_path,
    monkeypatch,
):
    original = RuntimeError("closure")
    runtime, history, _, _, _ = _runtime(
        monkeypatch,
        phase="evaluate_trial",
        error=original,
    )
    history_path = tmp_path / "turek_hron_fsi_history.csv"
    original_bytes = b"old-header\nold-row\n"
    history_path.write_bytes(original_bytes)

    def partial_writer(path, rows, *, header_written):
        with Path(path).open("wb") as handle:
            handle.write(b"partial-row")
        raise OSError("writer failed after bytes")

    namespace = _boundary_namespace(
        runtime,
        history,
        tmp_path,
        incremental_history_path=history_path,
        _persist_runtime_failure_evidence=functools.partial(
            turek._persist_runtime_failure_evidence,
            history_writer=partial_writer,
        ),
    )
    with pytest.raises(RuntimeError) as caught:
        _execute_actual_case_exception_guard(namespace)
    assert caught.value is original
    assert history_path.read_bytes() == original_bytes
    evidence = _evidence(tmp_path)
    assert evidence["completed_history_rows_flushed"] == 0
    assert evidence["persistence_errors"] == [
        "history_flush:OSError:writer failed after bytes"
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_replace_failure_leaves_existing_history_bytes_unchanged(
    tmp_path,
    monkeypatch,
):
    original = RuntimeError("closure")
    runtime, history, _, _, _ = _runtime(
        monkeypatch,
        phase="evaluate_trial",
        error=original,
    )
    history_path = tmp_path / "turek_hron_fsi_history.csv"
    original_bytes = b"old-header\nold-row\n"
    history_path.write_bytes(original_bytes)
    replace = Path.replace

    def fail_temporary_replace(source, destination):
        if Path(destination) == history_path:
            raise OSError("replace failed")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_temporary_replace)
    namespace = _boundary_namespace(
        runtime,
        history,
        tmp_path,
        incremental_history_path=history_path,
    )
    with pytest.raises(RuntimeError) as caught:
        _execute_actual_case_exception_guard(namespace)
    assert caught.value is original
    assert history_path.read_bytes() == original_bytes
    evidence = _evidence(tmp_path)
    assert evidence["completed_history_rows_flushed"] == 0
    assert evidence["persistence_errors"] == [
        "history_flush:OSError:replace failed"
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_actual_final_summary_write_failure_persists_committed_history(
    tmp_path,
    monkeypatch,
):
    """The production final-output guard must not lose accepted rows."""

    original = OSError("summary write failed")
    runtime, _, _, fluid, solid = _runtime(
        monkeypatch,
        phase="evaluate_trial",
        error=RuntimeError("unused"),
    )
    runtime._failure_phase = "final_output"
    runtime._rollback_outcome = "not_attempted"
    runtime._rollback_failure = None
    history = [dict.fromkeys(turek.HISTORY_FIELDS, 0)]
    history[0].update(step=8, time_s=0.04)

    original_write_text = Path.write_text

    def fail_summary_write(path, *args, **kwargs):
        if Path(path).name == "turek_hron_fsi_summary.json":
            raise original
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_summary_write)
    namespace = dict(vars(turek))
    namespace.update(
        latest_report_box={"value": object()},
        marker_layout_identity=lambda *args, **kwargs: "layout",
        markers=SimpleNamespace(),
        marker_reference_positions_m=np.zeros((1, 3)),
        marker_layout_sha256="layout",
        measured_taichi_runtime_identity={},
        generic_run=SimpleNamespace(history=history),
        config=turek.fsi1_config(step_count=8),
        preset="fsi1",
        history=history,
        output_dir=tmp_path,
        export_final_flow_snapshot=False,
        fluid=fluid,
        solid=solid,
        case_runtime=runtime,
        incremental_history_path=None,
        last_flushed_index=0,
        incremental_header_written=False,
    )
    with pytest.raises(OSError) as caught:
        _execute_actual_final_output_guard(namespace)
    assert caught.value is original
    assert (fluid.value, solid.value) == (1.0, 2.0)
    assert [int(row["step"]) for row in _csv_rows(tmp_path)] == [8]
    evidence = _evidence(tmp_path)
    assert evidence["failure_phase"] == "final_output"
    assert evidence["failed_step"] is None and evidence["failed_time_s"] is None
    assert evidence["completed_steps"] == 1
    assert evidence["completed_history_rows_flushed"] == 1
    assert evidence["physical_state_restored"] is None
    assert evidence["rollback_outcome"] == "not_attempted"
    assert evidence["rollback_failure"] is None
    assert evidence["fsi_coupling_certificate_available"] is False


def test_final_history_write_and_failed_recovery_preserve_existing_bytes(
    tmp_path,
    monkeypatch,
):
    """A failed final rewrite cannot truncate the prior accepted CSV."""

    original = OSError("final history writer failed")
    runtime, _, _, fluid, solid = _runtime(
        monkeypatch,
        phase="evaluate_trial",
        error=RuntimeError("unused"),
    )
    runtime._failure_phase = "final_output"
    runtime._rollback_outcome = "not_attempted"
    runtime._rollback_failure = None
    history = [dict.fromkeys(turek.HISTORY_FIELDS, 0)]
    history[0].update(step=8, time_s=0.04)
    history_path = tmp_path / "turek_hron_fsi_history.csv"
    original_bytes = b"step,time_s\n1,0.005\n"
    history_path.write_bytes(original_bytes)

    original_writerow = csv.DictWriter.writerow

    def write_row_then_fail(writer, row):
        result = original_writerow(writer, row)
        if row.get("step") == 8:
            raise original
        return result

    # Both the historical direct final writer and the atomic writer use this
    # boundary.  It writes a data row before failing, exposing truncation if
    # final publication targets the destination instead of a sibling temporary.
    monkeypatch.setattr(csv.DictWriter, "writerow", write_row_then_fail)
    namespace = dict(vars(turek))
    namespace.update(
        latest_report_box={"value": object()},
        marker_layout_identity=lambda *args, **kwargs: "layout",
        markers=SimpleNamespace(),
        marker_reference_positions_m=np.zeros((1, 3)),
        marker_layout_sha256="layout",
        measured_taichi_runtime_identity={},
        generic_run=SimpleNamespace(history=history),
        config=turek.fsi1_config(step_count=8),
        preset="fsi1",
        history=history,
        output_dir=tmp_path,
        export_final_flow_snapshot=False,
        fluid=fluid,
        solid=solid,
        case_runtime=runtime,
        incremental_history_path=None,
        last_flushed_index=0,
        incremental_header_written=False,
    )
    with pytest.raises(OSError) as caught:
        _execute_actual_final_output_guard(namespace)
    assert caught.value is original
    assert history_path.read_bytes() == original_bytes
    evidence = _evidence(tmp_path)
    assert evidence["failure_phase"] == "final_output"
    assert evidence["completed_history_rows_flushed"] == 0
    assert evidence["completed_history_rows_flushed_scope"] == (
        "complete accepted history published by this failure-recovery rewrite"
    )
    assert evidence["persistence_errors"] == [
        "history_flush:OSError:final history writer failed"
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_error_reporting_cannot_replace_original_exception(tmp_path, monkeypatch):
    original = RuntimeError("closure")
    runtime, history, _, _, _ = _runtime(monkeypatch, phase="evaluate_trial", error=original)
    def unavailable_writer(*args, **kwargs):
        raise OSError("storage unavailable")
    def unavailable_print(*args, **kwargs):
        raise OSError("stderr unavailable")
    monkeypatch.setattr("builtins.print", unavailable_print)
    namespace = _boundary_namespace(runtime, history, tmp_path,
        _persist_runtime_failure_evidence=functools.partial(
            turek._persist_runtime_failure_evidence,
            history_writer=unavailable_writer, artifact_writer=unavailable_writer))
    with pytest.raises(RuntimeError) as caught:
        _execute_actual_case_exception_guard(namespace)
    assert caught.value is original
