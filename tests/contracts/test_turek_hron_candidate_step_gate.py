"""Candidate gates must precede both accepted state publication surfaces."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from cases import turek_hron_fsi as turek
from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingConfig,
    FsiSolverConfig,
    solve_fsi_runtime,
)


class _State:
    def __init__(self, value):
        self.value = value
        self.saved = None

    def save_state(self):
        self.saved = self.value

    def restore_state(self):
        self.value = self.saved


@pytest.mark.parametrize("reject", [False, True])
def test_candidate_gate_is_before_history_and_publication_and_rolls_back(reject, tmp_path):
    fluid, solid = _State(1.0), _State(2.0)
    history, published = [], []
    row = {"step": 1, "time_s": 0.005, "mpm_deformation_clamp_count": int(reject)}
    context_box = {}

    def validator(candidate):
        if candidate["mpm_deformation_clamp_count"]:
            raise ValueError("MPM integrity gate failed")

    def advance(context, trial_index):
        fluid.value, solid.value = 10.0, 20.0
        return object(), {}

    def commit(context, trial, coupling):
        context_box["value"] = context
        turek._append_validated_fsi_row(
            history, row, context=context, candidate_step_validator=validator
        )
        return row

    runtime = turek._TurekHronFsiRuntime(
        fluid=fluid,
        solid=solid,
        markers=SimpleNamespace(),
        boundary=SimpleNamespace(),
        advance_trial=advance,
        prepare_step=lambda context: None,
        restore_case_boundaries=lambda context: None,
        commit_case_step=commit,
        finalize_case_run=lambda: {},
        publish_case_step=lambda context, candidate: published.append(candidate),
    )
    marker = {"v_gamma_mps": np.zeros((1, 3))}
    config = FsiSolverConfig(
        step_count=1, time_step_s=0.005,
        coupling=FsiCouplingConfig(max_iterations=2),
    )
    with (
        patch.object(turek, "capture_marker_interface_state", return_value=marker),
        patch.object(turek, "restore_marker_interface_state"),
        patch.object(turek, "marker_trial_state", return_value=marker),
        patch.object(turek, "_marker_pressure_neumann_gradient_state", return_value=np.zeros((1, 3))),
        patch.object(turek, "_restore_marker_pressure_neumann_gradient_state"),
        patch.object(turek, "_fsi_coupling_marker_candidate_from_step_base", return_value=marker),
    ):
        if reject:
            with pytest.raises(turek.TurekHronStepAcceptanceError, match="MPM integrity") as caught:
                solve_fsi_runtime(runtime, config)
            assert history == []
            assert published == []
            assert (fluid.value, solid.value) == (1.0, 2.0)
            payload = turek._step_acceptance_failure_payload(
                caught.value, preset="fsi1", completed_steps=len(history)
            )
            _, _, errors = turek._persist_fsi_coupling_failure_evidence(
                incremental_history_path=None, history=history,
                last_flushed_index=0, incremental_header_written=False,
                output_dir=tmp_path, failure_payload=payload,
            )
            assert errors == ()
            evidence = json.loads((tmp_path / "turek_hron_fsi_coupling_failure.json").read_text())
            assert evidence["failed_step"] == 1
            assert evidence["failed_time_s"] == 0.005
            assert evidence["completed_steps"] == 0
            assert evidence["physical_state_restored"] is True
            assert evidence["candidate_row"] == row
        else:
            run = solve_fsi_runtime(runtime, config)
            assert len(run.history) == len(history) == len(published) == 1
            assert (fluid.value, solid.value) == (10.0, 20.0)


def test_rejected_nonfinite_candidate_is_preserved_in_strict_json(tmp_path):
    from simulation_core.drivers.generic_fsi_solver import FsiStepContext
    row = {"step": 1, "time_s": 0.005, "residual": float("nan")}
    def reject(candidate):
        raise ValueError("nonfinite residual")
    with pytest.raises(turek.TurekHronStepAcceptanceError) as caught:
        turek._append_validated_fsi_row(
            [], row,
            context=FsiStepContext(step=1, step_index=0, time_s=0.005, dt_s=0.005),
            candidate_step_validator=reject,
        )
    payload = turek._step_acceptance_failure_payload(
        caught.value, preset="fsi1", completed_steps=0
    )
    path = turek._write_fsi_coupling_failure_artifact(tmp_path, payload)
    assert json.loads(path.read_text())["candidate_row"]["residual"] == {
        "nonfinite_float": "nan"
    }
