# ANSYS Vertical-Flap R24C.1 CI Attestation Goal

Date: 2026-09-01

Status: EVIDENCE COMPLETE — the traction-default repair and source-matched
CUDA regeneration are complete. Terminal closure is accepted only when the
generated pair verifies against a matching successful clean-final-HEAD CI run;
the pair, rather than this prose status, is authoritative.

## Baseline, authority, and authorization

This goal was reviewed against baseline commit aef06288d098ac5e674cdec381a5a41df13eff8b.
The authoritative checkout is the WSL Ubuntu-22.04 worktree:

    /home/zhuohengli/worktrees/HIBM-MPM-r21-validation

The working branch is codex/r24c1-ci-attestation. The Windows checkout is only
the project entry point and is not an implementation source. All source reads,
edits, tests, and validation commands use the WSL checkout explicitly.

On 2026-09-01 the user expanded the R24C.1 authorization to correct the frozen
runner's default traction predicate and rebuild the complete source-matched
preflow, Q0, probe, reuse, and strict-CUDA numerical evidence. Commit and
non-force push of the R24C.1 branch are explicitly authorized. This does not
authorize a pull request, Fluent run, R24D/R24E work, a deployable predictor,
or changes outside the named R24C.1 implementation, evidence, seal, tests, CI,
and documentation surface. A dirty checkout at seal time, stale source map,
artifact drift, verifier mismatch, or failed CI gate is a stop condition.

## Objective and scope

R24C.1 now closes both the production traction-default regression and the
publication-attestation contract. The earlier CPU-only boundary is retained as
historical context, but is superseded by the user's explicit authorization for
one direct runner fix and a complete source-matched numerical regeneration.

The implementation order is:

1. Correct only the default traction predicate and add a runtime integration
   contract for both preallocation validators.
2. Freeze the resulting 139-file source map and rebuild a complete stationary
   preflow snapshot plus a separate load-check consumer.
3. Run the three exact8 Q0 arms and all nine terminal no-commit probes.
4. Build and verify the threshold matrix, run the four carry/oracle by
   reuse-off/on exact8 arms, and verify the reuse matrix bottom-up.
5. Keep the accepted-displacement artifact immutable and write a new path-free
   legacy projection bound to the regenerated threshold evidence.
6. Validate strict JSON, SHA/commit identities, preflow bytes, the complete
   source map, and current checkout bytes.
7. Require a clean final HEAD and matching successful GitHub workflow/jobs.
8. Construct the schema-1 attestation core and schema-3 projection, write the
   projection first and terminal attestation second, then verify the pair.

The full source map is exactly 139 safe relative paths with canonical SHA256
a14a313568d86f6773c8fcbb2d5b1611e833389eb7455272554ae2e78d566b00. Every
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

## Historical remote failure and production repair

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

That finding was not weakened. Commit
4475da464ad098e26991f0c8bcf5eba836900186 directly aligned the executable
predicate with the production defaults: material marker offset 0.0,
physical_face_offset pressure-probe origin, and probe offset 0.51 cells. It
also added a bare-default integration assertion that reaches both preallocation
validators. The runner SHA256 is
56f005af2a3d8b03a50dadb61ea630dc0eb00673df84276d4112e5aa4a922ce8.

The TDD gate moved from 4 failures and 15 passes to 20 passes. The exact
Windows runner group passed 23 tests, and the broader relevant suite passed
352 tests with one pre-existing NumPy warning. A fresh read-only Sol Ultra
review returned ship with no findings. GitHub Actions run 33500763817 for that
source-fix commit concluded success. Because the runner changed, none of the
old-source R24C numerical roots are used as current source-matched evidence;
the regeneration record below replaces them for R24C.1.

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

The three Q0 compact producer reports contain legacy non-standard JSON
constants only in unrelated undefined zmin diagnostics. Seal-time runtime
extraction therefore reads each raw report once, hashes those exact bytes,
rejects duplicate keys, recursively rejects NaN or infinity anywhere in the
runtime/producer subtree, and copies only the finite runtime whitelist. It
accepts only the known legacy NaN token outside that subtree;
Infinity/-Infinity are rejected before field selection in every location. It
never publishes the absolute offline-cache path. The required path-free
raw-byte bindings are:

- omega 0.50: `a1e8cc0dcd2dee73b33ded7d9e808ce09f0eb4b8ee51d769166e9da65b93c69e`;
- omega 0.75: `feee24643817a0c0d3ee5e6fc9283534a5b31f404f565adb7c4c5693a952fd81`;
- omega 1.00: `7b3db40d75d4f8e077e96e5570194ea5a10a07dd85d1e830c61ed016c1d77270`.

The attestation runtime schema requires exactly these three distinct hashes.
Projection and attestation loading and writing remain strict JSON and still
reject every non-finite value.
The credential-key exception is limited to the exact validated
`source_sha256` mapping and entries whose values are SHA-256 digests; the
same credential-like filename token anywhere else remains rejected.

The initial boundary test run recorded 11 failures and 18 passes. After the
implementation and the added missing/non-object and duplicate-evidence cases,
the focused WSL suite passed all 34 tests. The same suite now covers the
source-map filename regression and its negative boundary; both WSL and Windows
pass all 34 tests. A production consensus read of the three regenerated roots
returned strict CUDA, f32, seed 0, Taichi 1.7.4 and the exact three hashes
above.

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
by these R24C.1 CI checks. They do not substitute for the separately completed
local strict-CUDA regeneration.

Remote acceptance is green only when both Ubuntu and Windows jobs pass on the
clean final documentation/seal-contract commit, with a verified schema-3 pair
and all source, artifact, runtime, host, and GitHub bindings. Run 33500763817
proves the source-fix commit green, and run 33518531919 proves the first
seal-constant/documentation closure green. Neither historical run substitutes
for the pair's exact-final-HEAD binding after any later seal-contract edit.
The exact terminal run is intentionally recorded in the generated pair rather
than frozen into this document. The schema-3 pair must not be generated
against any earlier or dirty HEAD.

## Rollback and explicit exclusions

Any verifier failure, source-map staleness, preflow/artifact drift, output
collision, host identity gap, or failed pair re-verification stops the flow
before terminal publication. Verify-only mode is read-only. Existing evidence
is never rewritten. No seal is generated until the final checkout is clean and
its matching GitHub run is successful.

R24C.1 explicitly excludes R24D, R24E, adaptive Kalman, GRU or other learned
predictors, production predictor writeback, Fluent parity, exact50, and
long-run claims. It preserves deployable=false and the projection
bottom_up_reverification=false boundary.

## Source-matched regeneration record

All runs below use the new 139-file source map, strict CUDA, f32, random seed
0, and Taichi 1.7.4. The preflow root is
`validation_runs/solver_soaks/ansys_vf__preflow__material_fine__20260901__r24c1_tractionfix`;
it reached stationary convergence naturally at preflow step 79 with zero FSI
steps and zero physical FSI time. The independent consumer is
`ansys_vf__preflow_loadcheck__material_fine__20260901__r24c1_tractionfix`.
The snapshot identities are:

- config: `16298a97d45eb03639e7f3d1c7f4048386d66bd573af55c7e485ceedabc4783f`;
- source: `ce0bd5489268cb6669ba3c258c35c6e53484edb27fc3d11dfcacde74aa85fad8`;
- geometry: `eb28d5616a3ac1f270c2ceae286534a387c5176873086cbd5105d06dae568bbb`;
- state JSON: `875acc70bb5c925e7f062e473574b71f9c984f2a04648868918032ca60891b14`;
- internal manifest: `09523cd9ad7d4d9de04cd692686ea7a3ee07b9e301544c8cb6b2291b4c18f85e`;
- NPZ: `d2e562cfb63fd7a81f1b471b028a636adc568bd693ae625d0e137fb7b9673616`.

The exact8 Q0 roots are the `r24c1_tractionfix` omega 0.50, 0.75, and 1.00
arms. Their `(coupling trials, rejected trials, CG iterations)` totals are
respectively `(24, 16, 5728)`, `(24, 16, 5728)`, and `(19, 11, 4528)`.
Each has exactly 8 frames and 8 histories through physical time 0.004 s.
Every accepted fluid and solid macro step consumes the full `dt_s=0.0005`
within floating-point roundoff; convergence never advances physical time.

All nine `r24c1_tractionfix` probes for omega 0.50/0.75/1.00 and target step
2/5/8 ended with `research_probe_terminal`. Each evaluated 10 alphas, retained
only its accepted prefix of 1/4/7 frames and histories, and reported converged
rows, rollback equality, sweep-state equality, and no mismatch fields. The
bottom-up threshold verifier returned `PASS_ORACLE_THRESHOLD_MATRIX`, selected
safe `omega=0.75`, authorized only the conditional reuse branch, and retained
`predictor_decision=academic_offline_feasibility_only` with `deployable=false`.

The omega-0.75 reuse matrix binds carry/off to the regenerated Q0 root. Its
carry/off and carry/on arms are `(24, 16, 5728)` and `(19, 11, 4528)`; the
offline-oracle off/on arms are both `(8, 0, 1920)`. Both oracle arms remain
non-causal and non-deployable. The bottom-up reuse verifier returned
`PASS_IQN_REUSE_FACTOR_MATRIX` with `status=reuse_matrix_authorized`.

The six seal inputs have SHA256:

- immutable accepted displacement: `84ee846dc09ec30607eedd21e2f2ecbd0206e594cf17d79d28451b78d219f98f`;
- threshold response: `7a0bd9dee1c6b16e966c981e518b1c8044eba0816c0803034c4ef559b158b2d6`;
- threshold source manifest: `bded530116914dd795d74c20628fd6690760dfcf87c9264fc9beffab58f75e9f`;
- threshold summary: `92a77a62cc4d9e9b48756ae58779cc31a8fffe80c6d484ee83e4fdca4b2487ba`;
- IQN reuse: `11bb1bb309e64cfa01f1ad3569c082ca839db55ccfc06942a643402e62373b59`;
- new legacy projection: `3e010664011acd5fd5695d76c6671aa2746bb5a05c8f06e2abdd47791650e742`.

The ignored numerical roots remain local and are not committed. The terminal
schema-3 projection and attestation may be written only after the
non-source-mapped goal/tool/test/documentation closure is committed, pushed,
and matched by a successful exact-final-HEAD GitHub run. Their embedded HEAD,
run ID, hashes, and successful verify result are the authoritative closure
record.
