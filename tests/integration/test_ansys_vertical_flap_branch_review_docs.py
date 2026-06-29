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
PR_HANDOFF = (
    Path("docs")
    / "refactoring"
    / "ANSYS_VERTICAL_FLAP_PR_HANDOFF_2026-06-29.md"
)
ARTIFACT_GENERATION_SOURCE_COMMIT = "c94332888fe09d792a119086a4969f78b03bb134"
REVIEWED_HEAD_COMMIT = "25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9"
CURRENT_PR_HEAD = "97de386279dcaa9e00693b8344d082a21a0114f9"


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
            "ANSYS_VERTICAL_FLAP_PR_HANDOFF_2026-06-29.md",
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
            "Branch:",
            f"Reviewed HEAD commit at checklist update: `{REVIEWED_HEAD_COMMIT}`",
            (
                "Artifact generation source commit: "
                f"`{ARTIFACT_GENERATION_SOURCE_COMMIT}`"
            ),
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
            ARTIFACT_GENERATION_SOURCE_COMMIT,
            REVIEWED_HEAD_COMMIT,
            "53 tests OK",
            "80 tests OK",
            "Fluent artifact policy checker: `PASSED_LOCAL`",
            "Validation artifact hygiene checker: `PASSED_LOCAL`",
            "artifact_generation_source_commit",
            "artifact_committed_in_review_head",
            "Remote CI evidence: `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`",
            "Remote CI source: `NOT_AVAILABLE_CONNECTOR_EMPTY`",
            "GitHub Actions run URL / run id: `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`",
        ):
            self.assertIn(phrase, text)

        self.assertNotIn("Commit SHA", text)
        self.assertNotIn("Result: `PENDING`", text)
        self.assertNotIn("PENDING_FINAL_COMMIT", text)
        self.assertNotIn("PENDING_REMOTE_CI_RUN", text)

    def test_pr_handoff_doc_records_freeze_and_ci_handoff(self):
        text = PR_HANDOFF.read_text(encoding="utf-8")

        for phrase in (
            f"Current PR head: `{CURRENT_PR_HEAD}`",
            "Suggested PR title",
            "PR Body Draft",
            "fluent_parity_claimed=false",
            "fluent_reference_incomplete",
            "no_fluent_parity_claim",
            "metadata_only_not_parity_truth",
            "EasyFsi and HIBM-MPM outputs are not promoted as real Fluent truth",
            "BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK",
            "connector returned no workflow run and no combined status",
            "Do not keep updating repository files just to chase the latest final commit hash",
            "GitHub Actions evidence should be added to the PR description or a PR comment",
            "CI Failure Response",
            "py_compile/import failure",
            "artifact checksum failure",
            "policy checker failure",
            "hygiene failure",
            "CRLF/whitespace failure",
            "synthetic fixture leakage",
            "ANSYS_VERTICAL_FLAP_VALIDATION_TOOLS_PACKAGE_GOAL_2026-06-30",
        ):
            self.assertIn(phrase, text)

        pr_body = text.split("## Review Guide", 1)[0]
        self.assertNotIn("status: passed", pr_body.split("Remote CI:", 1)[1])


if __name__ == "__main__":
    unittest.main()
