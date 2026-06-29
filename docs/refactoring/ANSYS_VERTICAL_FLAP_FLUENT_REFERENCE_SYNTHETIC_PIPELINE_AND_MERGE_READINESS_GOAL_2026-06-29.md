# ANSYS Vertical-Flap Fluent Reference Synthetic Pipeline And Merge Readiness Goal 2026-06-29

## Branch And Starting Point

- Branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Starting remote/head commit reviewed by the user: `babd3d96b36bb294c1d15f69632ee0dcc8b0f79d`
- Starting commit message: `validation: harden fluent reference export gates`
- Current known remote CI status: GitHub connector did not return a workflow run for that commit, so this goal must rely on local focused verification plus workflow-file hardening until a real Actions run is visible.

## One-Sentence Objective

In the ANSYS vertical-flap Fluent reference workflow, add merge-readiness documentation, a synthetic end-to-end source-export dry-run, JSON-aware artifact policy checking, artifact manifests, real-data templates, workflow coverage, and focused tests without introducing real Fluent data, without promoting synthetic data as Fluent truth, and without claiming Fluent parity while the real reference contract remains incomplete.

## Non-Negotiable Claim Boundaries

- Do not introduce real Fluent numeric reference data in this task.
- Do not claim Fluent parity for committed real artifacts.
- Keep the real active contract at `fluent_reference_incomplete`.
- Keep real parity artifacts at `fluent_parity_claimed=false`.
- Keep `no_fluent_parity_claim` active for real artifacts.
- Do not use EasyFsi/HIBM-MPM output as Fluent reference truth.
- Do not use public ANSYS tutorial data as numeric parity truth.
- Synthetic data may be used only inside tests or temporary directories and must be explicitly labeled `synthetic-test-only-not-fluent-truth`.
- Generated real artifact roots must not contain synthetic truth sources.
- Preserve existing CLI defaults for collection and parity runners.

## Phase 1: PR-Ready Merge Package

### 1.1 Expand Branch Review Map

Update:

`docs/refactoring/BRANCH_REVIEW_MAP_2026-06-29.md`

Add these sections:

- `Merge Risk Summary`
- `What Changes Runtime Physics`
- `What Is Pure Artifact/Validation`
- `What Is Generated`
- `What Remains Fail-Closed`
- `Required CI Evidence`
- `Suggested PR Split If Review Blocks`

The review map must explicitly state:

- No Fluent parity claim exists for real artifacts.
- The real Fluent contract remains incomplete.
- Runtime pressure-pair work is audited separately from Fluent reference truth.
- Fluent source exports are currently schema-only.
- No EasyFsi/HIBM-MPM output is promoted as Fluent truth.

### 1.2 Add Branch Merge Checklist

Add:

`docs/refactoring/ANSYS_VERTICAL_FLAP_BRANCH_MERGE_CHECKLIST_2026-06-29.md`

The checklist must include:

- Commit SHA.
- Branch name.
- GitHub Actions run URL / run id placeholder.
- Local interpreter path.
- Py-compile command.
- Focused unittest command.
- Artifact regeneration commands.
- Overclaim scan / policy command.
- `git diff --check` result slot.
- Secret scan result slot.
- Artifact checksum verification slot.
- Reviewer sign-off boxes.

### 1.3 Add Review Docs Test

Add:

`tests/integration/test_ansys_vertical_flap_branch_review_docs.py`

The test must assert the review map and checklist exist and include key phrases:

- `fluent_parity_claimed=false`
- `fluent_reference_incomplete`
- `no_fluent_parity_claim`
- `No EasyFsi`
- `No HIBM-MPM`
- `CI run URL`

## Phase 2: Synthetic Fluent Source-Exports End-To-End Dry-Run

### 2.1 Add Synthetic Fixture Builder

Add:

`validation_runs/ansys_vertical_flap_fsi/scripts/build_synthetic_fluent_reference_fixture.py`

The builder must write a complete synthetic source-export bundle into a caller-provided directory, preferably a temporary directory in tests. It must generate:

- `fluent_tip_displacement_history.csv`
- `fluent_force_history.csv`
- `fluent_flow_balance_history.csv`
- `fluent_pressure_summary_history.csv`
- `fluent_metadata_2026-06-28.md`

Each CSV must contain the expected header plus at least one `step=50`, `time_s=0.025` row. Each row's `source` must be `synthetic-test-only-not-fluent-truth`.

The metadata file must fill every field required by the collection validator and include:

- `Source document: synthetic test fixture, not Fluent truth`
- `Fluent run id: synthetic-test-only`

### 2.2 Add Injectable Collection Runner Entry Point

Extend:

`validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py`

Add a testable entry point equivalent to:

```python
run_with_paths(
    source_exports_root: Path,
    current_contract_json: Path,
    output_dir: Path,
    active_manifest_json: Path,
) -> dict[str, Any]
```

This entry point must preserve CLI default behavior, must not mutate module-level path constants permanently, and must be usable from temporary test directories.

### 2.3 Add Synthetic Collection Pipeline Test

Add:

`tests/integration/test_ansys_vertical_flap_fluent_reference_synthetic_pipeline.py`

The test must:

- Build temp synthetic source exports.
- Run collection validation against temp paths.
- Assert `candidate_contract_status == fluent_reference_complete`.
- Assert `schema_validation.validated_metric_count == 5`.
- Assert `promotion_status == ready_for_versioned_contract_promotion`.
- Assert active manifest schema version is valid.
- Assert active/candidate manifest hashes match generated files.

### 2.4 Add Synthetic Parity Pass/Fail Tests

In the same synthetic pipeline test or a separate test, verify:

- Matching synthetic reference metrics can produce `candidate_status == fluent_parity_validated` in temp-only test output.
- A deliberately mismatched synthetic force reference produces `candidate_status == fluent_parity_failed` and `fluent_parity_claimed=false`.
- These synthetic parity outputs must not be written into committed real diagnostics roots.

## Phase 3: JSON-Aware Fluent Artifact Policy Checker

### 3.1 Add Policy Checker Script

Add:

`validation_runs/ansys_vertical_flap_fsi/scripts/check_fluent_artifact_policy.py`

Default scanned roots:

- `validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics`

Rules:

- If any JSON has `fluent_parity_claimed=true`, then:
  - `candidate_status` must be `fluent_parity_validated`.
  - `reference_contract_status` must be `fluent_reference_complete`.
  - All parity metric `gate_status` values must be `passed` or explicitly allowed `report_only`.
- If `reference_contract_status != fluent_reference_complete`, then:
  - `fluent_parity_claimed` must be false.
  - `candidate_status` must not be `fluent_parity_validated`.
- Real generated artifacts must not contain `synthetic-test-only`.
- Real generated artifacts must not claim public tutorial numeric metric truth; only `metadata_only_not_parity_truth` is allowed.

The checker should return a structured payload from a callable function and exit nonzero from CLI if violations are present.

### 3.2 Update Workflow

Update:

`.github/workflows/ansys-vertical-flap-validation.yml`

Add:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py
```

Keep the existing simple string overclaim scan unless replacing it is clearly safer.

### 3.3 Add Policy Checker Tests

Add:

`tests/integration/test_ansys_vertical_flap_fluent_artifact_policy.py`

Cover:

- Incomplete reference plus claimed parity fails.
- Complete reference plus failed metric and claimed parity fails.
- Complete reference plus all-passed metrics and claimed parity passes.
- Current real artifacts pass.
- `synthetic-test-only` in a real artifact fails.

## Phase 4: Validation Artifact Hygiene

### 4.1 Add Hygiene Script

Add:

`scripts/check_validation_artifact_hygiene.py`

Checks:

- Generated JSON/CSV/MD artifacts are UTF-8 readable.
- No absolute Windows local paths are committed in generated validation artifacts.
- No `D:\working` paths exist in generated validation artifacts.
- No secret-like tokens appear except allowlisted SHA-256 hashes and explicit synthetic test strings.
- No `fluent_parity_validated` appears in real generated artifact roots while the active contract is incomplete.
- All `CHECKSUMS.sha256` entries match the current files.

### 4.2 Add Hygiene Tests

Add:

`tests/tools/test_validation_artifact_hygiene.py`

Cover current real artifacts passing and at least one failing temp artifact case.

## Phase 5: Artifact Manifest Format

### 5.1 Add Fluent Reference Collection Manifest

Update collection runner to generate:

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/ARTIFACT_MANIFEST.json`

Schema:

```json
{
  "manifest_schema_version": "validation_artifact_manifest_v1",
  "artifact_group": "fluent_reference_collection",
  "source_script": "...",
  "generated_from_commit": "unknown-or-current",
  "inputs": {
    "current_contract": "...",
    "source_exports_root": "...",
    "public_tutorial_evidence_map": "..."
  },
  "outputs": {
    "matrix_json": {"path": "...", "sha256": "..."},
    "matrix_csv": {"path": "...", "sha256": "..."},
    "candidate_contract": {"path": "...", "sha256": "..."},
    "summary_md": {"path": "...", "sha256": "..."}
  },
  "claim_policy": {
    "fluent_parity_claimed": false,
    "reason": "reference incomplete"
  }
}
```

### 5.2 Add Fluent Parity Diagnostics Manifest

Update parity runner to generate:

`validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/ARTIFACT_MANIFEST.json`

It must use the same `validation_artifact_manifest_v1` schema and reflect the parity matrix claim policy.

### 5.3 Add Manifest Tests

Update or add tests to verify:

- Manifest outputs are listed in `CHECKSUMS.sha256`.
- Manifest SHA values match current files.
- `source_script` is repo-relative.
- No manifest paths are absolute.
- Manifest `claim_policy` matches the matrix artifact.

## Phase 6: Real Fluent Data Intake Templates

### 6.1 Add Metadata Template

Add:

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/fluent_metadata_TEMPLATE.md`

It must include every required field:

- Source document
- Fluent run id
- Export author
- Export date
- Fluent version
- mesh/domain source
- geometry units
- material model
- boundary conditions
- time step
- number of steps
- coupling settings if applicable
- export procedure
- who/when/how generated
- force_z_positive
- flow_rate_positive
- pressure_reference
- displacement_definition

The active `fluent_metadata_2026-06-28.md` must remain `MISSING`; the template must not be treated as active metadata.

### 6.2 Add CSV Export Templates

Add template CSVs under:

`docs/validation/fluent_reference_export_templates/`

Do not put template CSVs in active `source_exports/`.

## Phase 7: Long-Term Tools Package Plan

Add a planning-only file:

`docs/refactoring/ANSYS_VERTICAL_FLAP_VALIDATION_TOOLS_PACKAGE_GOAL_2026-06-30.md`

It must propose moving reusable Fluent reference validation logic into:

`tools/validation/ansys_vertical_flap/`

The plan must keep current `validation_runs/.../scripts/*.py` as stable thin wrappers, avoid artifact behavior changes during the future move, and keep tests unchanged during the planning step.

Do not perform the large migration in this task.

## Phase 8: Final Verification Matrix

Before commit and push, run:

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\fluent_reference_contract_schema.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\fluent_source_export_schema.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\build_synthetic_fluent_reference_fixture.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py `
  scripts\check_validation_artifact_hygiene.py

python validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py
python validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py
python validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py
python scripts\check_validation_artifact_hygiene.py

python -m unittest -v `
  tests.integration.test_ansys_vertical_flap_branch_review_docs `
  tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema `
  tests.integration.test_ansys_vertical_flap_fluent_source_export_schema `
  tests.integration.test_ansys_vertical_flap_fluent_reference_export_protocol `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_fluent_reference_synthetic_pipeline `
  tests.integration.test_ansys_vertical_flap_fluent_artifact_policy `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic `
  tests.tools.test_validation_artifact_hygiene

git diff --check
```

Expected real artifact state:

- Collection matrix:
  - `candidate_status = fluent_reference_collection_pending`
  - `candidate_contract_status = fluent_reference_incomplete`
  - `schema_validation.validated_metric_count = 0`
  - `schema_validation.required_metric_count = 5`
  - `public_reference_use_policy = metadata_only_not_parity_truth`
- Parity matrix:
  - `candidate_status = fluent_parity_blocked_reference_incomplete`
  - `fluent_parity_claimed = false`
  - `candidate_blockers` include `fluent_reference_incomplete`
  - `candidate_blockers` include `no_fluent_parity_claim`

## Commit And Push Requirement

After implementation and verification:

- Commit with a conventional message.
- Push to `origin solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
- Report the final commit hash, pushed branch, verification commands, and whether the working tree is clean.
