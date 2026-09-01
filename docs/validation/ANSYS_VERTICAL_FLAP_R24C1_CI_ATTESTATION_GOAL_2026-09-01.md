# ANSYS Vertical-Flap R24C.1 CI Attestation Goal

Date: 2026-09-01

Status: ACTIVE — fail-closed at a production traction-default regression;
remote green acceptance and the actual attestation remain pending.

## Baseline, authority, and authorization

This goal was reviewed against baseline commit aef06288d098ac5e674cdec381a5a41df13eff8b.
The authoritative checkout is the WSL Ubuntu-22.04 worktree:

    /home/zhuohengli/worktrees/HIBM-MPM-r21-validation

The working branch is codex/r24c1-ci-attestation. The Windows checkout is only
the project entry point and is not an implementation source. All source reads,
edits, tests, and validation commands use the WSL checkout explicitly.

The current authorization is limited to the R24C.1 core/contracts/focused tests,
seal CLI, CI workflow and the documentation files named by this goal. On
2026-09-01 the user explicitly authorized commit and push for the R24C.1
branch. This does not authorize a pull request, publication attestation, CUDA
run, Fluent run, numerical rerun, or changes outside that set. A dirty
checkout, stale source map, artifact drift, verifier mismatch, or failed CI
gate is a stop condition.

## Objective and scope

R24C.1 closes the CPU-only CI and publication-attestation contract around the
already completed R24C evidence. It does not rerun numerical production.

The implementation order is:

1. Validate strict JSON, SHA/commit identities, the complete source map, and
   the current checkout bytes.
2. Reverify displacement, threshold, reuse, and legacy artifacts bottom-up,
   including before/after byte and hash snapshots.
3. Verify preflow state.json and NPZ bytes and publish only their hashes.
4. Derive numerical runtime consensus from the three formal Q0 compact reports,
   while keeping producer runtime separate from attestation-host identity.
5. Require a clean final HEAD and matching successful GitHub workflow/jobs.
6. Construct the schema-1 attestation core and schema-3 projection, write the
   projection first and terminal attestation second, then verify the pair.
7. Keep all old evidence and the schema-2 legacy projection immutable.

The full source map is exactly 139 safe relative paths with canonical SHA256
84afaa15c7c4cc07ebceadbc141cb087c53a00efb71f6ca2e81b787649f350d8. Every
current-checkout byte must match it. The execution_source mode is
source_map_bound_working_tree; its producer commit, count, and map SHA are
bound separately from the final clean HEAD. Any source change invalidates the
map and stops publication.

## CI root causes closed by R24C.1

Historical GitHub run 33465020431 concluded failure. The four local semantic
root-cause categories were:

1. Particle-bin generation tests still described the removed surface-feedback
   consumer instead of the current material-surface consumer and its
   generation/support-radius contract.
2. Generation timing was asserted only from the old static statement order;
   the preflow runtime path needed a test that captures the current generation
   and support radius at the scatter call.
3. The solid-substep test expected the retired batched helper rather than the
   current selected macro-step runtime report and persisted accepted counts.
4. The removed-cell-backend test passed an incomplete namespace, so validation
   failed on missing configuration before reaching the intended full-config
   backend rejection.

The fixes are test-first: the initial local RED result was 3 failures and
1 error, followed by runtime/AST contract corrections. The five required
semantic nodes are now CPU-only and must remain in the Windows contracts job:
the two original generation tests, the preflow runtime-capture test, the solid
substeps runtime test, and the removed-backend full-config test.

## Remote run 33492835609 and fail-closed production finding

Clean commit ea3884b4d78cfa380ab2954467cbb6d4961910ab was pushed to the
R24C.1 branch and triggered GitHub Actions run 33492835609. The Ubuntu
quality-and-fast-contracts job passed, the scheduled CUDA job was correctly
skipped, and every new R24C.1 compile, publication, and five-node semantic
step in the Windows contracts job passed. Windows failed later in the existing
source-level runner contract group.

The six failing nodes reproduced identically in WSL, so this was additional
test coverage rather than a Windows or CRLF defect. Five failures were stale
test contracts after the deef2f3 continuous-execution refactor: material
markers versus pressure-probe origins, checkpoint-aware step-loop discovery,
history append discovery, and the material-surface update API name. The
test-only corrections reduce the complete source-level group from two failures
and four errors to 22 passes and one unresolved valid failure.

The remaining failure is a production validation regression. The current
VerticalFlapFsiConfig defaults place material markers on the physical face and
offset only pressure probes, but _is_default_traction_formulation still
recognizes the retired marker-offset formulation. Consequently
_validate_rectangular_solid_config rejects an otherwise default coupled config
as a non-default fixed-solid diagnostic. Changing the assertion would hide a
real error and is forbidden.

Correcting that predicate would modify the frozen executable runner, invalidate
the 139-file source map, and make every retained R24C numerical root old-source
evidence. Because this goal excludes CUDA and numerical reruns, the runner is
unchanged and the workflow remains intentionally fail-closed until a separately
authorized source-matched regeneration path is chosen.

## Publication schema and transaction invariants

The core has schema_version 1 and bottom_up_reverification=true. Its
clean_checkout_reconstruction, producer execution_source, complete source
map, preflow hashes, six-artifact map, successful GitHub identity, numerical
runtime, and strict four-part host identity are all required. The core flags
deployable, release, release_recommendation, and
numerical_artifacts_fully_public are false.

The projection has schema_version 3, preserves the schema-2 raw four-hash
legacy map, embeds the reuse hash, legacy-projection hash, GitHub identity, and
core SHA, and has bottom_up_reverification=false. All release/deployable flags
remain false. The old schema-2 artifact is never overwritten.

The core SHA excludes the projection file SHA, so there is no core/projection
cycle. Publication writes a unique temporary file, flushes and fsyncs it,
replaces the destination, and fsyncs the directory where supported. The
projection is written first; the terminal attestation records its SHA and is
written second; the pair is not described as atomically committed. Output
destinations must be distinct and absent before clean-head validation.

The attestation host is an independent mapping with exactly python, taichi,
cuda, and gpu entries. A seal requires all four to be recorded with non-empty
identity: Python and Taichi versions, CUDA support/driver identity, and GPU
name/UUID/device identity. The numerical producer fields remain explicitly
recorded=false when absent and must never be filled from host data.

## CLI and remote acceptance

Seal mode requires absolute, resolvable displacement, threshold directory,
reuse, legacy projection, source manifest, GitHub run ID, projection, and
attestation paths. It performs no numerical run. Verify mode needs only an
existing projection and attestation and performs no GitHub or GPU probe.

The Linux R24 evidence job compiles the core, contracts helper, CLI, and
focused test; it executes the focused pytest and CLI help. The Windows
contracts job compiles the same four files, executes the focused pytest, and
executes the five semantic nodes above. Existing fast gates remain, and the
scheduled CUDA job retains its existing conditional behavior; no CUDA is used
by these R24C.1 checks.

Remote acceptance is green only when both Ubuntu and Windows jobs pass on a
clean final commit, with a verified schema-3 projection and all source,
artifact, runtime, host, and GitHub bindings. Run 33492835609 is red because
the existing Windows source-level group correctly exposes the production
traction-default regression described above. The schema-3 attestation pair is
therefore pending and must not be generated or claimed from this red branch.

## Rollback and explicit exclusions

Any verifier failure, source-map staleness, preflow/artifact drift, output
collision, host identity gap, or failed pair re-verification stops the flow
before terminal publication. Verify-only mode is read-only. Existing evidence
is never rewritten. No seal is generated from the current dirty/red checkout.

R24C.1 explicitly excludes R24D, R24E, adaptive Kalman, GRU or other learned
predictors, production predictor writeback, CUDA/numerical reruns, Fluent
parity, exact50 or long-run claims. It preserves deployable=false and the
projection bottom_up_reverification=false boundary.
