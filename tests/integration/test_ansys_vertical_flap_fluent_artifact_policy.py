from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "check_fluent_artifact_policy.py"
)


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_fluent_artifact_policy", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


class AnsysVerticalFlapFluentArtifactPolicyTests(unittest.TestCase):
    def test_current_real_artifacts_pass_policy(self):
        result = CHECKER.check_fluent_artifact_policy()

        self.assertEqual(result["status"], "passed", result["violations"])
        self.assertGreater(result["checked_file_count"], 0)

    def test_incomplete_reference_with_claimed_parity_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "bad.json",
                {
                    "candidate_status": "fluent_parity_validated",
                    "reference_contract_status": "fluent_reference_incomplete",
                    "fluent_parity_claimed": True,
                    "parity_metrics": _passed_metrics(),
                },
            )

            result = CHECKER.check_fluent_artifact_policy([root])

        rules = {item["rule"] for item in result["violations"]}
        self.assertEqual(result["status"], "failed")
        self.assertIn("claimed_parity_without_complete_reference", rules)
        self.assertIn("validated_candidate_with_incomplete_reference", rules)

    def test_complete_reference_with_failed_metric_and_claimed_parity_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = _passed_metrics()
            metrics["force"]["gate_status"] = "failed"
            _write_json(
                root / "bad.json",
                {
                    "candidate_status": "fluent_parity_validated",
                    "reference_contract_status": "fluent_reference_complete",
                    "fluent_parity_claimed": True,
                    "parity_metrics": metrics,
                },
            )

            result = CHECKER.check_fluent_artifact_policy([root])

        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "claimed_parity_with_failed_metric_gate",
            {item["rule"] for item in result["violations"]},
        )

    def test_complete_reference_with_all_passed_metric_gates_can_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "good.json",
                {
                    "candidate_status": "fluent_parity_validated",
                    "reference_contract_status": "fluent_reference_complete",
                    "fluent_parity_claimed": True,
                    "parity_metrics": _passed_metrics(),
                },
            )

            result = CHECKER.check_fluent_artifact_policy([root])

        self.assertEqual(result["status"], "passed", result["violations"])

    def test_synthetic_marker_in_real_artifact_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "bad.json",
                {
                    "candidate_status": "fluent_parity_blocked_reference_incomplete",
                    "reference_contract_status": "fluent_reference_incomplete",
                    "fluent_parity_claimed": False,
                    "source": "synthetic-test-only-not-fluent-truth",
                },
            )

            result = CHECKER.check_fluent_artifact_policy([root])

        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "synthetic_marker_in_real_artifact",
            {item["rule"] for item in result["violations"]},
        )


def _passed_metrics() -> dict[str, dict[str, str]]:
    return {
        "displacement": {"gate_status": "passed"},
        "force": {"gate_status": "passed"},
        "flow_outlet": {"gate_status": "passed"},
        "pressure": {"gate_status": "passed"},
        "metadata": {"gate_status": "passed"},
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
