from __future__ import annotations

import unittest
from pathlib import Path


REVIEW_MAP = (
    Path("docs") / "refactoring" / "BRANCH_REVIEW_MAP_2026-06-29.md"
)
CHECKLIST = (
    Path("docs")
    / "refactoring"
    / "ANSYS_VERTICAL_FLAP_BRANCH_MERGE_CHECKLIST_2026-06-29.md"
)
BASELINE_COMMIT = "c94332888fe09d792a119086a4969f78b03bb134"


class AnsysVerticalFlapBranchReviewDocsTests(unittest.TestCase):
    def test_review_map_contains_merge_readiness_sections(self):
        text = REVIEW_MAP.read_text(encoding="utf-8")

        for phrase in (
            "Merge Risk Summary",
            "What Changes Runtime Physics",
            "What Is Pure Artifact/Validation",
            "What Is Generated",
            "What Remains Fail-Closed",
            "Required CI Evidence",
            "Suggested PR Split If Review Blocks",
            "Review Navigation",
            "ANSYS_VERTICAL_FLAP_PR_SPLIT_STRATEGY_2026-06-29.md",
            "ARTIFACT_MANIFEST.json",
            "fluent_parity_claimed=false",
            "fluent_reference_incomplete",
            "no_fluent_parity_claim",
            "No EasyFsi",
            "No HIBM-MPM",
        ):
            self.assertIn(phrase, text)

    def test_merge_checklist_records_required_evidence_slots(self):
        text = CHECKLIST.read_text(encoding="utf-8")

        for phrase in (
            "Commit SHA",
            "Branch:",
            "GitHub Actions run URL / run id",
            "Local interpreter path",
            "py_compile",
            "Focused Unit Tests",
            "Artifact Regeneration",
            "git diff --check",
            "Secret scan result",
            "Artifact Checksums",
            "fluent_parity_claimed=false",
            "fluent_reference_incomplete",
            "no_fluent_parity_claim",
            "No EasyFsi",
            "No HIBM-MPM",
            "CI run URL",
            BASELINE_COMMIT,
            "53 tests OK",
            "Fluent artifact policy checker: `PASSED_LOCAL`",
            "Validation artifact hygiene checker: `PASSED_LOCAL`",
            "Remote CI evidence: `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`",
            "Remote CI source: `NOT_AVAILABLE_CONNECTOR_EMPTY`",
            "GitHub Actions run URL / run id: `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`",
        ):
            self.assertIn(phrase, text)

        self.assertNotIn("PENDING_FINAL_COMMIT", text)
        self.assertNotIn("PENDING_REMOTE_CI_RUN", text)


if __name__ == "__main__":
    unittest.main()
