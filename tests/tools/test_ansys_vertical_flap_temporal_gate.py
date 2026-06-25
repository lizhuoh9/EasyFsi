from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_source_candidate_step20_matrix.py"
)


def _load_matrix_module():
    spec = importlib.util.spec_from_file_location("source_candidate_step20", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load_matrix_module()


class AnsysVerticalFlapTemporalGateTests(unittest.TestCase):
    def test_flow_temporal_strict_and_coupling_settled_promotes_when_combined_passes(self):
        row = _row()
        history = _history()

        matrix._apply_temporal_classification([row], {"case": history})

        self.assertEqual(row["flow_temporal_status"], "flow_temporal_strict")
        self.assertEqual(row["flow_post_warmup_failed_step_count"], 0)
        self.assertEqual(row["coupling_settling_status"], "coupling_settled")
        self.assertEqual(row["temporal_candidate_status"], "temporal_strict")
        self.assertEqual(row["promotion_candidate_status"], "promotion_ready")

    def test_flow_temporal_soft_allows_exactly_two_post_warmup_failures(self):
        history = _history(step_count=12, p999_overrides={6: 19.0, 7: 19.5})
        row = _row()

        report = matrix._flow_temporal_report(row, history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_soft")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 2)
        self.assertEqual(report["flow_last_window_failed_step_count"], 0)

    def test_flow_temporal_failed_after_three_post_warmup_failures(self):
        history = _history(
            step_count=12,
            p999_overrides={6: 19.0, 7: 19.5, 8: 19.8},
        )

        report = matrix._flow_temporal_report(_row(), history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_failed")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 3)

    def test_flow_temporal_failed_when_last_window_fails(self):
        history = _history(step_count=12, p999_overrides={12: 19.0})

        report = matrix._flow_temporal_report(_row(), history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_failed")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 1)
        self.assertEqual(report["flow_last_window_failed_step_count"], 1)

    def test_missing_history_is_not_applicable(self):
        flow_report = matrix._flow_temporal_report(_row(), [])
        coupling_report = matrix._coupling_settling_report(_row(), [])
        combined_report = matrix._temporal_report(_row(), [])

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

        report = matrix._flow_temporal_report(row, history)

        self.assertEqual(report["flow_temporal_status"], "flow_temporal_strict")
        self.assertEqual(report["flow_post_warmup_failed_step_count"], 0)

    def test_flow_strict_but_coupling_unsettled_is_not_promotable(self):
        row = _row()
        history = _history(force_overrides={10: 0.1}, tip_overrides={10: 0.1})

        matrix._apply_temporal_classification([row], {"case": history})

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
    force_overrides: dict[int, float] | None = None,
    tip_overrides: dict[int, float] | None = None,
) -> list[dict]:
    p999_overrides = p999_overrides or {}
    force_overrides = force_overrides or {}
    tip_overrides = tip_overrides or {}
    rows: list[dict] = []
    for step in range(1, step_count + 1):
        rows.append(
            {
                "step": step,
                "velocity_peak_mps": 30.0,
                "velocity_p999_mps": p999_overrides.get(step, 24.0),
                "velocity_outlet_flux_ratio": 1.0,
                "marker_force_z_N": force_overrides.get(step, -0.1),
                "tip_dz_m": tip_overrides.get(step, -0.001),
                "stress_invalid_marker_count": 0,
                "scatter_invalid_marker_count": 0,
                "feedback_invalid_marker_count": 0,
            }
        )
    return rows


if __name__ == "__main__":
    unittest.main()
