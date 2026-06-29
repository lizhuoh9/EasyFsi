# ANSYS Vertical-Flap PR Handoff Goal - 2026-06-29

## Source Review

- Target branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Current remote/local reviewed head: `97de386279dcaa9e00693b8344d082a21a0114f9`
- Current head commit message: `validation: clarify artifact provenance wording`
- Remote CI state from review attachment: GitHub Actions connector still returns no workflow run and no combined status.

## Objective

Freeze the branch for PR review and add one small PR handoff document that records how to create/update the PR, how to handle the still-missing remote CI evidence, and how to avoid future self-referential commit-provenance churn.

This goal deliberately does not add solver functionality, validation functionality, artifact semantics, new Fluent truth, or new parity claims. It is a documentation/test closure that turns the current branch into a review handoff package.

## Non-Goals

- Do not update solver code.
- Do not update Fluent parity logic.
- Do not regenerate validation artifacts.
- Do not edit `ARTIFACT_MANIFEST.json` solely to chase the newest commit SHA.
- Do not replace `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK` unless a real GitHub Actions run URL/status is available.
- Do not claim remote CI green from local tests.
- Do not open real Fluent parity.

## Required Claim Boundaries

The handoff must keep these boundaries explicit:

- `fluent_parity_claimed=false`
- `fluent_reference_incomplete`
- `no_fluent_parity_claim`
- public ANSYS tutorial metadata remains `metadata_only_not_parity_truth`
- synthetic fixtures are temporary test fixtures only
- EasyFsi/HIBM-MPM outputs are not real Fluent truth

## Required Documentation

Add:

- `docs/refactoring/ANSYS_VERTICAL_FLAP_PR_HANDOFF_2026-06-29.md`

The handoff document must include:

- Current PR head: `97de386279dcaa9e00693b8344d082a21a0114f9`
- Branch name.
- Suggested PR title.
- PR body draft with:
  - scope,
  - claim boundary,
  - key artifacts,
  - local evidence,
  - remote CI status,
  - review guide links.
- A clear instruction not to keep updating repository files just to chase the latest final commit hash.
- The rule that GitHub Actions evidence should be added to the PR description/comment unless a reviewer explicitly asks for a committed checklist update.
- CI handling steps:
  - find existing branch/PR run,
  - run workflow_dispatch if needed,
  - create PR if pull_request workflow is needed,
  - record run URL/status in PR comment,
  - if CI fails, fix only the failing surface.
- CI failure response matrix covering:
  - py_compile/import failure,
  - checksum failure,
  - policy checker failure,
  - hygiene failure,
  - CRLF/whitespace failure,
  - synthetic fixture leakage.
- PR review order and split strategy references.
- Post-merge next technical line:
  - `ANSYS_VERTICAL_FLAP_VALIDATION_TOOLS_PACKAGE_GOAL_2026-06-30`

Update:

- `docs/refactoring/BRANCH_REVIEW_MAP_2026-06-29.md`

The review map must link the handoff document in `Review Navigation`.

## Required Tests

Extend existing test:

- `tests/integration/test_ansys_vertical_flap_branch_review_docs.py`

Add assertions that the new handoff doc:

- exists,
- includes current PR head `97de386279dcaa9e00693b8344d082a21a0114f9`,
- includes `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`,
- includes claim boundaries,
- includes the no-commit-chasing rule,
- includes PR title/body guidance,
- includes CI failure handling,
- includes the validation tools package next line.

Also assert that the review map links the handoff doc.

## Verification

Run:

```powershell
python -m py_compile tests\integration\test_ansys_vertical_flap_branch_review_docs.py
```

```powershell
python -m unittest -v tests.integration.test_ansys_vertical_flap_branch_review_docs
```

Also run:

- `git diff --check`
- after staging, `git diff --cached --check`
- changed-file secret keyword scan, expecting docs/test-rule text only and no credential material

## Completion Criteria

- Handoff doc exists and is linked from the review map.
- Branch review docs test passes.
- No validation artifact, solver, or parity logic changes are made.
- No remote CI green claim is made.
- Worktree is clean after commit.
- Branch is pushed to `origin/solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
- Final response reports final commit hash, pushed branch, tests/checks, and remote CI remains pending/manual.
