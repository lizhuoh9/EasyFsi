from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

from tools.validation import ansys_vertical_flap_temporal_gates as gates


MODULE_PATH = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_source_candidate_step20_matrix.py"
)
PREFLOW_RELEASE_MODULE_PATH = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_preflow_release_step20_matrix.py"
)


def _load_matrix_module():
    spec = importlib.util.spec_from_file_location("source_candidate_step20", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_preflow_release_module():
    spec = importlib.util.spec_from_file_location(
        "preflow_release_step20",
        PREFLOW_RELEASE_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PREFLOW_RELEASE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load_matrix_module()
preflow_release_matrix = _load_preflow_release_module()


class AnsysVerticalFlapTemporalGateTests(unittest.TestCase):
    def test_flow_temporal_strict_and_coupling_settled_promotes_when_combined_passes(self):
        row = _row()
        history = _history()

        row.update(gates.classify_flow_temporal(row, history))
        row.update(gates.classify_coupling_settling(row, history))
        row.update(gates.classify_combined_temporal(row, history))
        row["promotion_candidate_status"] = gates.promotion_status(row)

        self.assertEqual(row["flow_temporal_status"], "flow_temporal_strict")
        self.assertEqual(row["flow_post_warmup_failed_step_count"], 0)
        self.assertEqual(row["coupling_settling_status"], "coupling_settled")
        self.assertEqual(row["temporal_candidate_status"], "temporal_strict")
        self.assertEqual(row["promotion_candidate_status"], "promotion_ready")

    def test_flow_temporal_soft_allows_exactly_two_post_warmup_failures(self):
        history = _history(step_count=12, p999_overrides={6: 19.0, 7: 19.5})
        row = _row()

        report = gates.classify_flow_temporal(row, history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_soft")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 2)
        self.assertEqual(report["flow_last_window_failed_step_count"], 0)

    def test_flow_temporal_failed_after_three_post_warmup_failures(self):
        history = _history(
            step_count=12,
            p999_overrides={6: 19.0, 7: 19.5, 8: 19.8},
        )

        report = gates.classify_flow_temporal(_row(), history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_failed")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 3)

    def test_flow_temporal_failed_when_last_window_fails(self):
        history = _history(step_count=12, p999_overrides={12: 19.0})

        report = gates.classify_flow_temporal(_row(), history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_failed")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 1)
        self.assertEqual(report["flow_last_window_failed_step_count"], 1)

    def test_flow_temporal_failed_when_only_stricter_last_window_fails(self):
        history = _history(step_count=12, outlet_overrides={12: 1.22})

        report = gates.classify_flow_temporal(_row(), history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_failed")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 0)
        self.assertEqual(report["flow_last_window_failed_step_count"], 1)
        self.assertIn(
            "velocity_outlet_ratio_outside_0.80_1.20",
            report["flow_temporal_fail_reasons"],
        )

    def test_missing_history_is_not_applicable(self):
        flow_report = gates.classify_flow_temporal(_row(), [])
        coupling_report = gates.classify_coupling_settling(_row(), [])
        combined_report = gates.classify_combined_temporal(_row(), [])

        self.assertEqual(
            flow_report["flow_temporal_status"],
            "flow_temporal_not_applicable",
        )
        self.assertEqual(
            flow_report["flow_temporal_fail_reasons"],
            ["missing_history"],
        )
        self.assertEqual(
            coupling_report["coupling_settling_status"],
            "coupling_not_applicable",
        )
        self.assertEqual(
            combined_report["temporal_candidate_status"],
            "temporal_not_applicable",
        )

    def test_ramp_warmup_excludes_early_ramp_steps(self):
        row = _row(source_ramp_steps=5)
        history = _history(step_count=12, p999_overrides={6: 18.0, 7: 19.0})

        report = gates.classify_flow_temporal(row, history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_strict")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 0)

    def test_flow_strict_but_coupling_unsettled_is_not_promotable(self):
        row = _row()
        history = _history(force_overrides={10: 0.1}, tip_overrides={10: 0.1})

        row.update(gates.classify_flow_temporal(row, history))
        row.update(gates.classify_coupling_settling(row, history))
        row.update(gates.classify_combined_temporal(row, history))
        row["promotion_candidate_status"] = gates.promotion_status(row)

        self.assertEqual(row["flow_temporal_status"], "flow_temporal_strict")
        self.assertEqual(row["coupling_settling_status"], "coupling_unsettled")
        self.assertEqual(row["temporal_candidate_status"], "temporal_failed")
        self.assertEqual(row["promotion_candidate_status"], "not_promotion_candidate")

    def test_payload_separates_flow_candidate_from_promotion_candidate(self):
        flow_only = _row(scenario="flow_only")
        flow_only_history = _history(
            force_overrides={6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1}
        )
        fallback = _row(scenario="fallback", final_velocity_p999_mps=24.5)
        fallback_history = _history(p999_overrides={10: 19.0})
        rows = [flow_only, fallback]
        histories = {
            "flow_only": flow_only_history,
            "fallback": fallback_history,
        }

        matrix._apply_temporal_classification(rows, histories)
        payload = matrix._payload(rows=rows)

        self.assertEqual(payload["best_flow_temporal_candidate"], "flow_only")
        self.assertEqual(payload["best_combined_temporal_candidate"], "none")
        self.assertEqual(payload["promotion_candidate"], "none")
        self.assertEqual(payload["diagnostic_fallback_candidate"], "fallback")

    def test_reclassify_logic_is_idempotent(self):
        rows = [
            _row(scenario="flow_only"),
            _row(scenario="fallback", final_velocity_p999_mps=24.5),
        ]
        histories = {
            "flow_only": _history(force_overrides={6: 0.1}),
            "fallback": _history(p999_overrides={10: 19.0}),
        }

        first_rows = copy.deepcopy(rows)
        matrix._apply_temporal_classification(first_rows, histories)
        first_payload = matrix._payload(rows=first_rows)

        second_rows = copy.deepcopy(first_rows)
        matrix._apply_temporal_classification(second_rows, histories)
        second_payload = matrix._payload(rows=second_rows)

        self.assertEqual(
            json.dumps(first_payload, sort_keys=True),
            json.dumps(second_payload, sort_keys=True),
        )

    def test_preflow_release_flow_candidate_ignores_coupling_penalty(self):
        flow_best = _preflow_release_row(
            scenario="flow_best",
            release_coupling_settling_status="coupling_unsettled",
            release_first_permanently_valid_step="",
            release_longest_consecutive_pass_steps=0,
        )
        coupling_best = _preflow_release_row(
            scenario="coupling_best",
            final_velocity_p999_mps=27.5,
            release_flow_last_window_min_p999_mps=21.0,
            release_flow_last_window_mean_outlet_ratio=1.18,
            release_coupling_settling_status="coupling_settled",
            release_first_permanently_valid_step=1,
            release_longest_consecutive_pass_steps=20,
        )

        payload = preflow_release_matrix._payload([flow_best, coupling_best])

        self.assertEqual(payload["best_release_flow_candidate"], "flow_best")
        self.assertEqual(payload["best_release_coupling_candidate"], "coupling_best")
        self.assertEqual(payload["best_release_promotion_candidate"], "none")

    def test_preflow_release_promotion_gate_requires_root_and_residual_bounds(self):
        ready = _preflow_release_promotion_row()
        missing_root = {
            **ready,
            "release_final_root_max_displacement_m": "",
        }
        high_root = {
            **ready,
            "release_final_root_max_displacement_m": 1.0e-6,
        }
        high_marker_residual = {
            **ready,
            "release_final_marker_action_reaction_residual_N": 1.0e-5,
        }
        high_scatter_residual = {
            **ready,
            "release_final_scatter_action_reaction_residual_N": 1.0e-5,
        }

        self.assertEqual(
            preflow_release_matrix._promotion_candidate_status(ready),
            "promotion_ready",
        )
        for row in (
            missing_root,
            high_root,
            high_marker_residual,
            high_scatter_residual,
        ):
            self.assertEqual(
                preflow_release_matrix._promotion_candidate_status(row),
                "not_promotion_candidate",
            )


def _row(
    *,
    scenario: str = "case",
    source_ramp_steps: int = 0,
    final_velocity_p999_mps: float = 24.0,
) -> dict:
    return {
        "scenario": scenario,
        "run_status": "completed",
        "flow_driver_uses_full_velocity_reset": False,
        "source_ramp_steps": source_ramp_steps,
        "candidate_status": "candidate",
        "final_velocity_p999_mps": final_velocity_p999_mps,
        "final_velocity_peak_mps": 30.0,
        "max_velocity_peak_mps": 32.0,
        "velocity_outlet_flux_ratio": 1.0,
        "marker_force_z_N": -0.1,
        "tip_dz_final_m": -0.001,
        "stress_invalid_marker_count": 0,
        "scatter_invalid_marker_count": 0,
        "feedback_invalid_marker_count": 0,
    }


def _history(
    *,
    step_count: int = 10,
    p999_overrides: dict[int, float] | None = None,
    outlet_overrides: dict[int, float] | None = None,
    force_overrides: dict[int, float] | None = None,
    tip_overrides: dict[int, float] | None = None,
) -> list[dict]:
    p999_overrides = p999_overrides or {}
    outlet_overrides = outlet_overrides or {}
    force_overrides = force_overrides or {}
    tip_overrides = tip_overrides or {}
    rows: list[dict] = []
    for step in range(1, step_count + 1):
        rows.append(
            {
                "step": step,
                "velocity_peak_mps": 30.0,
                "velocity_p999_mps": p999_overrides.get(step, 24.0),
                "velocity_outlet_flux_ratio": outlet_overrides.get(step, 1.0),
                "marker_force_z_N": force_overrides.get(step, -0.1),
                "tip_dz_m": tip_overrides.get(step, -0.001),
                "stress_invalid_marker_count": 0,
                "scatter_invalid_marker_count": 0,
                "feedback_invalid_marker_count": 0,
            }
        )
    return rows


def _preflow_release_row(**overrides) -> dict:
    row = {
        "scenario": "case",
        "release_flow_temporal_status": "flow_temporal_strict",
        "release_coupling_settling_status": "coupling_unsettled",
        "promotion_candidate_status": "not_promotion_candidate",
        "final_velocity_p999_mps": 24.5,
        "final_velocity_peak_mps": 30.0,
        "max_velocity_peak_mps": 32.0,
        "velocity_outlet_flux_ratio": 1.0,
        "release_flow_last_window_min_p999_mps": 24.0,
        "release_flow_last_window_mean_outlet_ratio": 1.0,
        "release_first_permanently_valid_step": "",
        "release_longest_consecutive_pass_steps": 0,
    }
    row.update(overrides)
    return row


def _preflow_release_promotion_row(**overrides) -> dict:
    row = {
        "run_status": "completed",
        "apply_marker_feedback_to_fluid": True,
        "flow_source_schedule_scope": "global",
        "preflow_flow_temporal_status": "flow_temporal_strict",
        "release_flow_temporal_status": "flow_temporal_strict",
        "release_temporal_candidate_status": "temporal_strict",
        "candidate_status": "candidate",
        "release_ramp_restarted_after_preflow": False,
        "release_final_root_max_displacement_m": 0.0,
        "release_final_marker_action_reaction_residual_N": 0.0,
        "release_final_scatter_action_reaction_residual_N": 0.0,
    }
    row.update(overrides)
    return row


if __name__ == "__main__":
    unittest.main()
