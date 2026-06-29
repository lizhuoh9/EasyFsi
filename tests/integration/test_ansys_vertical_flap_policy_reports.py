from __future__ import annotations

import json
import unittest
from pathlib import Path


REPORT_ROOT = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "policy_reports"
)
FLUENT_POLICY_REPORT = REPORT_ROOT / "fluent_artifact_policy_report.json"
HYGIENE_REPORT = REPORT_ROOT / "validation_artifact_hygiene_report.json"


class AnsysVerticalFlapPolicyReportTests(unittest.TestCase):
    def test_fluent_policy_report_is_committed_and_passed(self):
        report = _read_report(FLUENT_POLICY_REPORT)

        self.assertEqual(report["checker"], "check_fluent_artifact_policy")
        self.assertEqual(report["policy"], "fluent_artifact_policy_v1")
        self.assertEqual(report["policy_id"], "fluent_artifact_policy_v1")
        self.assertEqual(report["status"], "passed", report["violations"])
        self.assertEqual(report["violations"], [])
        self.assertGreater(report["checked_file_count"], 0)

    def test_hygiene_report_is_committed_and_passed(self):
        report = _read_report(HYGIENE_REPORT)

        self.assertEqual(report["checker"], "check_validation_artifact_hygiene")
        self.assertEqual(report["policy"], "validation_artifact_hygiene_v1")
        self.assertEqual(report["policy_id"], "validation_artifact_hygiene_v1")
        self.assertEqual(report["status"], "passed", report["violations"])
        self.assertEqual(report["violations"], [])
        self.assertGreater(report["checked_file_count"], 0)


def _read_report(path: Path) -> dict[str, object]:
    assert path.exists(), path
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
