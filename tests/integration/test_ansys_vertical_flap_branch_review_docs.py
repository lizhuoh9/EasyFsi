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
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
