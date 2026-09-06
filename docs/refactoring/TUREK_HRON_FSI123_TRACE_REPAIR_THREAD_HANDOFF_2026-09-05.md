# Turek-Hron FSI trace-repair continuation

## Explicit continuation and WIP publication, 2026-09-06

The user resumed work after the archived quota stop, explicitly withdrew the
10% stop threshold, and authorized committing and pushing all current project
work to GitHub before quota exhaustion. The earlier pause below is historical.
No quota-reset credit has been used. The current core is still the reviewed
`26f37eb1` candidate; its 25-test native verification and continuous physical
validation remain incomplete at this source checkpoint.

The next focused run will use a fresh output label and a manifest that records
this commit's actual execution identity while preserving source and input
hashes. The old A50 baseline and all original captured identities remain intact.
The full-replay geometry-binding review correction is being prepared externally.

Publication destination: [R15 WIP source and complete diagnostic evidence](https://github.com/lizhuoh9/EasyFsi/releases/tag/r15-wip-20260906-1330).
The 141,016,737-byte archived snapshot predates this explicit continuation;
its SHA256 is `671728300c42b7e0a0d50a9169d39ab50854b66ef53d2f9d808ac2c2fa15b5c7`.
It preserves the stopped WIP and original evidence without claiming a native,
continuous, component or formal benchmark pass.

## Earlier R15 quota-stop snapshot, 2026-09-06

The campaign is `PAUSED_AT_USER_QUOTA_BOUNDARY`. The root observed 10% main
quota remaining and stopped all owned test/numerical work under the user's
preserved stop/archive rule. No reset credit was redeemed; both existing credits
remain untouched. Resume requires an explicit user continuation.

The reviewed candidate is applied at core `26f37eb1`, with focused test mixin
`2423bdc6` and new R15 fixture `c4141569`. Its native verification is incomplete.
The frozen 25-test strict-CUDA/f32 regression was interrupted during first
compilation, before any case file or terminal test report was completed.
Owner PID/PGID400 received SIGINT and then SIGTERM; exit status was -15.
Exec32892 and the stop helper are closed. This interruption is an administrative
quota stop, not a numerical failure or a regression pass. No owned GPU job remains.

The original A50 R15 failure was reproduced at exactly four faces with unchanged
inputs and source: (x=0..3,y=50,z=342), axis2. Full readback verified18x153 native
snapshot arrays,207 original input arrays,193 source files, all eight atomic
publication fields and39 cleanup fields. The frozen replay advances no time.
Its readback report is `039f3270a52e408e56be740ab05e795dbcd7cc803fd493c37abac1062c9dc5c0`.

The isolated R04 native candidate query identified the cause of that rejection:
the first ordinary pair lacks a valid physical geometry proof, while five
alternative pairs among the four materialized candidates agree on one owner
and physical B/Q. The repair searches the candidate pairs only when the old
admission-and-full-valid cache is unavailable. It requires a unique consistent
trace, a seed containing a direct member and proof of every actually consumed
member. Ambiguous candidates retain rejection; original valid caches and the
zero/one-consumer behavior are preserved by contract. Two new temporary i32
fields participate in native cleanup, bringing the cleanup inventory to41.

The unchanged A50 consumer baseline completed2/2 strict-CUDA tests, with
1189.862763533 s native time and1230.101830959 s owned-process time.
Its16 exported canonical arrays were decoded and hash-checked, and source193,
host and runtime identities remained unchanged. Report:
`239766dd9e669da86573ac5749c8511dc9b07117e582e9e40354dd5258405839`.
This is the old-source comparison baseline; it does not validate the candidate.
Independent Astra/max reviews accepted the candidate core and focused tests
for controlled native validation; final launch guards9/9 and Ruff F/E9 passed.

The R12/R13/R14/R15 full-replay package completed host preparation only:
828 original arrays,1377 old-success snapshot arrays and983 immutable files
were verified. Its overall independent review is incomplete. A confirmed gap
remains in `success_adapter.py:115`: bind the replay's cached B/Q and related
geometry/target contracts to the existing frozen R04 query. Stable array hashes
alone do not prove correct geometry. The four native full replays were not started.

The latest physical result remains A50 R15: five accepted steps to0.025 s,
then a step6 rejection at trial time0.030 s. Accepted fluid and solid time were
each audited as0.025 s. Continuous success after this candidate, independent
complete rollback equality, the source-current coarse S0 operator check,
component qualification and formal FSI1/FSI2/FSI3 acceptance remain unverified.

After explicit resume, verify source/host/process/quota state; use a fresh
`native_green_r02` output and owner label for the unchanged25 focused contracts.
Do not reuse occupied `native_green_r01` or overwrite its manifest/evidence.
Then close the query-binding review gap, refresh package pins, run the four
frozen controls serially, and execute a fresh complete physical diagnostic.
Rebuild the original coarse operator from the R04 pose with4x48x288, auto112,
solid100,dt0.005,nine post-solid passes and external time0.040 s; keep original
targets and the1e-6 gate. Recompute source manifests before component/formal
qualification. Reduced field dumps and failed prefixes are not restart states.
No new repair commit or push was made.

## Authoritative state

- WSL Ubuntu-22.04, user `zhuohengli`.
- Root: `/home/zhuohengli/worktrees/HIBM-MPM-r25b-live`.
- Branch: `codex/turek-hron-fsi123-validation-r26a`.
- HEAD: `5dcba9601d254f899488a44865f353c1a67497b3`; preserve the dirty tree.
- Interpreter: `/home/zhuohengli/.venvs/hibm-mpm-r26a-py310/bin/python`.
- CPython3.10.12 / NumPy2.1.2 / SciPy1.15.3 / Taichi1.7.4, strict CUDA/f32.
- Unset PYTHONPATH/PYTHONHOME; set LD_LIBRARY_PATH=/usr/lib/wsl/lib,
  SIMULATION_TAICHI_OFFLINE_CACHE=1 and PYTHONUNBUFFERED=1.
- Use an explicit WSL root/interpreter and one owned expensive CUDA process.
  Never reuse an occupied label or restart from reduced diagnostic dumps.

| Current file | SHA-256 |
| --- | --- |
| `cases/turek_hron_fsi.py` | `d311c5b6befdba00e397632670f8e264dd3ea1e09f71992ccddbf259d4861523` |
| `simulation_core/coupling/hibm_mpm/core.py` | `26f37eb14179b73c0c4c4a7e29be53954dcade4a377ed4c91310026b88e51751` |
| `simulation_core/coupling/hibm_mpm/marker_mac_constraint.py` | `cc5d2601ac4d016ee6c6f8da141740fe15d8f215fe7f092804592a724a746947` |
| `simulation_core/fluids/solver.py` | `a1a3324bad2ef98e83c10e144bb368625381fbf3d9e00058dc492525d3fe9234` |

The unchanged endpoint ledger-test dependency SHA is
`f5c9c66f4d998a4c233058b9da60b79a52ba588aaede931daad3b28157a999e6`.
Common-cohort module SHA:
`2423bdc677060b9aaa4dc89fd0c77edfc0e1a7d9f13bf15e6ece6b3af1fa7cdc`;
fixture SHA `529df0f530f5fb9fe40dca9bbae58ed32cb8360f5c32a3d4ad39f2094d252db4`;
geometry-test SHA `1d48bbeb4c0d0699c31848ca4c33c0fc640a25aa5855849f4878814e71eaaeef`.
New tilted R14 fixture `tests/solvers/fixtures/turek_hron_common_trace_r14_tilted.json`
has SHA `4e0888122217433d64b571b1e3caab9c06ec99355cefbabb166e4a51a9737a37`.
All current source changes need their actual formal qualification; test or
review success does not replace numerical evidence.

## Earlier A50 results and physical failure

The common-trace face-route repair is applied at core `a50b67f0`.
All 14 focused contracts pass on strict CUDA/f32 in 1231.187055591 s
(`PASS_FOURTEEN_FOCUSED_ROUTE_CONTRACTS`), with no skips, errors or audit
failures and unchanged source/runtime identity. This includes the new tilted
R14 D/S affine regression and the previous 13 contracts.

The preceding from-zero coupled run was R14 on the earlier `d8a14f64` core:
five accepted steps to t = 0.025 s, then step 6 at trial t = 0.030 s rejects
during assembly 83 with 56 common-trace reconstruction conflicts. The frozen
failure replay records 18 native stages of 153 arrays; 76 common cohorts are
prepared and all 56 newly rejected lanes have D/S cached seeds. An isolated
native query at face (0, 50, 305), axis 2, finds scalar primary face 306 but
certified pair face 305. The repair retains the existing pair route for every
admitted common trace while preserving cached B/N/Q, membership checks,
numerical gates and atomic publication.

Full-domain R12/R13/R14 controls on `a50b67f0` now pass their required
native results and complete artifact readback. Each records 21 snapshots of
153 arrays (3,213 readbacks), verifies 193 source files and 747 pinned files,
preserves 32 non-output inputs, audits all eight outputs and clears 39 temporary
fields. R12 retains its endpoint route with no common mode and eight byte-exact
outputs. R13 preserves all 12 common cohorts and 15 cache payloads. R14 preserves
all 76 common cohorts, all 15 full-domain cache arrays, prepared certificates and
positive keys, with zero conflicts in the 56 formerly rejected lanes.

Fresh R15 on `a50b67f0` exited 1 after five accepted steps to t = 0.025 s.
It requested eight steps from zero with fixed 112 markers, 4x96x400, solid200
and dt = 0.005 s. Step 6 at trial t = 0.030 s failed during assembly 99 with
four `prepare_pair_arbitration` conflicts; the first is face (0, 50, 342),
axis 2, path 0, claim_count 2. The 153-array precleanup snapshot and 207-field
assembly-input manifest are complete with no capture errors. Source193 and
the 25 execution dependencies match before and after the run.

The host-only `accepted_time_audit_r02` passes its audit with outcome
`FAILED_WITH_VALIDATED_ACCEPTED_PREFIX`: accepted fluid and solid time each
equals 0.025 s, and `fresh8_diagnostic_passed=false`. Five candidate/native
records and their CSV fields validate. The failure reporter records restored
physical state; complete independent post-rollback equality remains outstanding.
At that A50 checkpoint the R15 prepare-pair root cause was unresolved. The
later candidate diagnosis and stopped verification are recorded above. These
historical zero-time controls do not establish current coupled or formal acceptance.

The original coarse S0 obstruction has no recorded source-current recheck in
the inspected R04/frozen-coarse and later band/common evidence. A zero-time
current-operator check still needs the R04 pose, 4x48x288, automatic 112 markers,
solid100 and dt = 0.005 s, with the current post-solid budget of nine band passes
and external-boundary time 0.040 s. Rebuild geometry and topology from current
source; the old coefficients and obstruction certificate remain historical.
That reconstructed-pose check is not a complete physical restart or an S0 pass.
Source audit `source_identity_boundaries_r03.json` records component15
`f4a5891b` and formal206 `3693bf85`; all 10 historical component manifests
still need current-source qualification.

### Prior common-cohort repair at core `d8a14f64`

The earlier reviewed common-cohort repair was applied at core `d8a14f64`.
Three unchanged compact fixtures covering r13 z305/z306/z347 now pass strict
CUDA/f32 assembly and the existing affine known-solution check. The original
core rejected all three in558.449204s; the candidate passes all three in
1133.775381s, including cold compilation. The highest affine error is
1.49011612e-8m/s. This comparison measures correctness, not acceleration.
All193 source hashes and full runtime identity stay fixed within each run;
only core.py differs between runs. Fixture bytes, cached geometry/routes and
original/shadow sample payload observations are identical.

The fix separates cached geometry seeds from actual consumed authors. A new
path can replace an existing rejection only after every actual member and
the cached seeds pass current source/storage, registered-owner and geometric
support checks. It preserves the cached B/N/Q, canonical sample formula and
single atomic publication. Failed proof retains the original rejection
events; no global health counter is reset to admit a cohort. The new temporary
owner/mask fields are included in native commit and error cleanup.

The integrated strict-CUDA suite passes all 10 new and 3 existing contracts
in 792.493602486 s. Seven negative cases preserve all eight canonical fields
byte-exact before fixture reset and inspect native error cleanup. The targeted
bad additional source reaches actual arbitration, separately from the six
global-invalid-input controls.

Both full-domain frozen replays pass. R13 completes all 12 original conflict
lanes with common mode256 and zero global conflicts. All14 old cache payloads
match the immutable original precompute; all15 including specified owners
remain fixed through both precommits. R12 retains its endpoint route with no
common mode and all eight final arrays byte-exact to the old native success.
Both restore40 real inputs, preserve32 non-output inputs, satisfy complete C8
publication and clear39 temporary fields. Six native snapshots of153 arrays
per run and final source/runtime identities pass artifact checks.
Owned times1198.438568592/39.114366293s include different compilation/cache
conditions; they do not establish acceleration.

At that earlier checkpoint, original ad7b570e r13 had five accepted steps
to t=.025s, then step6 rejected at t=.030s. Accepted fluid and solid time each
equaled .025s; complete independent physical post-rollback equality was still
outstanding. That evidence led to the fresh fixed112,4x96x400,solid200,eight-step
R14 diagnostic, whose later failure is recorded above.
These zero-time controls are not a coupled, component or formal FSI1/2/3 pass.
The112 physical marker obligations, thresholds and formal order are unchanged.

## Earlier systemic diagnosis (superseded next actions)

Keep the established failure families distinct:

1. Original coarse formal S0 and fine frozen matrices have certified trace
   residual obstructions. Iteration count cannot solve those same fixed systems.
2. r07's zero-free collective contract and r09's unsaturated solid-band support
   are independently diagnosed and repaired. The final r09 ordering control
   changes legal support and passes its frozen-pose residual audit; this is not
   a solution to the old rank-deficient fixed system.
3. r12 fails a same-storage inactive-axis terminal geometry rule. Its reviewed
   repair and known-solution/negative checks pass.
4. r13 exposes precomputed geometry seeds and actual member lists that do
   not agree: same-segment D/D/S, adjacent-segment D/D/S, and geometry-DD
   with actual-DS. Its common member certificate is implemented and the three
   compact affine regressions and both full-domain native controls passed on
   `d8a14f64`; R14 then exposed the separate reconstruction route defect below.
5. R14 prepares certified common cohorts but the D/S reconstruction branch
   selects a scalar primary face that differs from its cached pair face. The
   common-only route correction is applied at `a50b67f0` and passes all 14
   focused contracts and all three repaired full-domain native controls.
6. R15 on the repaired source accepts five steps, then assembly 99 reports four
   prepare-pair conflicts at step 6. The first face is (0, 50, 342), axis 2,
   path 0, claim_count 2. Its complete precleanup capture is available; the
   diagnostic signature does not yet establish this failure's root cause.

The interpolated path currently uses separate pair selection, raw-author
arbitration and reconstruction contracts. The full-source registry assembler
is restricted to non-interpolated2D segments. Static code identifies a coverage
risk at those contracts, not proof that every failure has one root cause.

Current work is diagnosis of the fully captured R15 prepare-pair failure and
the outstanding source-current coarse operator check. The source-matched
R12/R13/R14 native controls in `systemic_pair_route_replay_r01` remain complete
and audited. The reviewed `accepted_time_r15_preparation` auditor has now run:
`accepted_time_audit_r02` validates five original candidate/native records and
0.025 s of accepted time for each subsystem, while explicitly failing fresh8.
Its first invocation, `accepted_time_audit_r01`, failed the authoritative-cwd
guard before auditing; that failed attempt is preserved. Full independent
physical post-rollback equality remains unverified. The zero-time controls
and the passed audit of a failed prefix do not qualify formal coupled acceptance.
The original observer and its old-source evidence remain in
`systemic_frozen_cohort_r01`; the candidate success replay is prepared separately.
The reviewed native source/storage helper observations are in
`systemic_geometry_binding_probe_r03`. No physical target is smoothed or discarded.

## Source and formal boundaries

The current source identity audit is
`resume_live_20260905_01/source_identity_boundaries_r03.json`.
Component15 SHA is `f4a5891bb7d036610bc706351202707fab968fac9c98cd8587761354eaad604b`;
formal206 SHA is `3693bf856569ccc068f1dcec2a7a48f6e718a7444724f19ec7d194805b017037`.
All 10 historical component manifests use 15 files and old digest `9172c37f`;
the current source changes case/core/MAC/fluid. All 10 need current-source
qualification under their shared manifest contract. Diagnostic captures cover
193 files. These three manifest scopes are not interchangeable; this host-only
identity audit grants no qualification. The original component/formal evidence
and the earlier `r02` identity audit remain historical.

The fixed112 R14/R15 diagnostic configuration is4x96x400, solid200, dt.005,8steps/.04s.
Registered S0 is4x48x288, automatic112 markers, solid100,1600steps/8s.
The diagnostic is not S0 or M0; registered finer automatic marker counts also
differ. No stage definitions or numerical thresholds have changed. Current dirty
source independently fails formal clean-source entry. Current commit/publication
permission remains required; no generic workflow provides it.

After systemic geometry/support proof, freeze and qualify the required component
order, then FSI1-S0, M0/M1, conditional F0, FSI2 and FSI3. Oracle/learning remains
behind benchmark quality gates. Any source/config change needs a new matching
run label; failed prefixes and field/history dumps are not restart checkpoints.

## Evidence locations

External Windows root:
`C:/Users/lizhu/.codex/visualizations/2026/09/05/01a06f3c-1b43-7841-ac96-ca57e7365e03/robustness_work`.
WSL reads it under `/mnt/c/Users/lizhu/...`.

- `hard_closure_diagnostic_5dcba96__r13`: actual five-step prefix,207-array
  failure input capture and accepted_time_audit_r01. All207 arrays/27,226,148
  bytes match captured hashes/shapes/dtypes. The old observer's fixed window
  misses the first conflict; cleanup cleared later cache evidence.
- `resume_live_20260905_01/r13_capture_host_audit_r01`: host-only inventory,
  all108 nonzero visible cohorts and explicit missing third-source/live routes.
- `resume_live_20260905_01/repaired_assembly_r12_01`: successful exact-input
  geometry control with8 committed output files.
- `resume_live_20260905_01/same_storage_endpoint_positive_r02`,
  `same_storage_endpoint_remaining_r02`, `shared_x_cleanup_green_r01`: owned
  JSON/log prefixes. Positive1 passes; remaining17 has15 passes plus2 failures
  from a shared X-coordinate test leak; ordered roundoff/cap0/cap1 then passes
  after the two-line fixture cleanup. This is18 distinct tests across batches.
- `resume_live_20260905_01/systemic_assembly_review_snapshot_r01`:17 exact
  current implementation/test/design exports and manifest for independent review.

- `resume_live_20260905_01/systemic_frozen_cohort_r01`: frozen native
  observer/runner, r13 rejection and r12 success, full-domain149-array snapshots,
  complete host analysis in `analysis_native_pair_r01`.
- `resume_live_20260905_01/systemic_registered_owner_host_r01`: independent
  F64 finite-distance scan of all109 registered segments; this is host geometry,
  not native member/support certification.
- `resume_live_20260905_01/systemic_geometry_binding_probe_r01`: reviewed
  source/seed query matrix; no production repair is implied by its proposal.

See [the chronological trace audit](../validation/TUREK_HRON_TRACE_SPACE_AUDIT_2026-09-05.md),
[validation contract](../TUREK_HRON_VALIDATION.md) and
[campaign goal](../validation/TUREK_HRON_FSI123_NUMERICAL_VALIDATION_GOAL_2026-09-02.md).

New candidate evidence:
- `resume_live_20260905_01/systemic_cohort_regression_draft_r03/native_red_r01`:
  all three unchanged known-solution cases rejected on the original core.
- `resume_live_20260905_01/systemic_cohort_postfix_probe_r01/native_green_r01`:
  the same three cases pass on d8a; no physical time advances.
- `resume_live_20260905_01/systemic_cohort_postfix_audit_r01.json`:
  independent source/runtime/input and known-solution comparison.

## R14 reconstruction route evidence (2026-09-06)

The frozen rejection audit reads all 2,754 arrays across 18 native stages.
Prepare admits 76 exact mode256 cohorts; reconstruction newly rejects 56,
all with D/S cached seeds and path2 / claim_count3. The first is
(0,50,305), axis2: actual positive keys20304/20305, cached keys20305/19905,
owner[54,55,-1]. No canonical output is published on the rejected call.

The isolated native selector query finds primary face(0,50,306) and pair
face(0,50,305). It forces entry validity only for this subpath, does not
independently certify every common-entry condition, and publishes nothing.
It localizes the face-selection disagreement; the separate compact regression
exercises actual native prepare, reconstruction, failure cleanup and publication.

The correction changes only `_reconstruct_canonical_component_face_common_trace`
and its call in `_reconstruct_velocity_dirichlet_component_face_segment_claims_kernel`.
Every admitted common trace now requests the existing pair route at its
certified physical face. The D/D-versus-D/S selector split and its unused
projection-marker argument are removed. Cached B/N/Q, actual-member checks,
owner/mask state, route/alpha/geometry gates, target formula and atomic
publication remain unchanged.

The tilted affine case rejects on `d8a14f64` in 1179.790855181 s with
`CONFIRMED_R14_COMMON_ROUTE_RED`, exact path2 / claim_count3, all eight
canonical arrays unchanged, four common scratch fields and nine coordinates
restored. On `a50b67f0`, all 14 focused contracts pass in 1231.187055591 s.
Both are zero-time correctness evidence with compilation/cache cost included.

Evidence under `resume_live_20260905_01`:

- `systemic_r14_frozen_rejection_r01/native_r14_r01/root_audit_r01.json`:
  complete frozen failure classification and array readback.
- `r14_canonical_selector_probe_r01/native_r01/query_report.json`:
  isolated native scalar/pair selector result and its stated evidence limits.
- `r14_common_route_contracts_r01/native_red_r01/process.json` and
  `r14_common_route_contracts_r01/native_green_r01/process.json`:
  source/runtime identities, exact outcomes, durations and per-case hashes.
- `r14_pair_route_core_r01/core.diff`: common-route production change.
- `systemic_pair_route_replay_r01`: completed repaired R12/R13/R14 controls
  and their hash-linked `root_artifact_readback.json` reports.
- `accepted_time_r15_preparation`: reviewed auditor used for the completed
  R15 terminal check. Its result is under
  `robustness_work/hard_closure_diagnostic_5dcba96__r15/accepted_time_audit_r02`;
  the wrong-cwd `accepted_time_audit_r01` attempt is preserved alongside it.

## Completed repaired frozen controls and terminal R15 (2026-09-06)

All three controls on `a50b67f0` exit 0 and pass
`PASS_COMPLETE_NATIVE_ARTIFACT_READBACK`. Each contains 21 snapshots of
153 arrays (3,213 checked arrays), with 193 source and 747 pinned files verified.
All 32 non-output inputs are unchanged, all eight final outputs satisfy native
publication, and all 39 temporary fields are neutral after native cleanup.

| Replay | Required native result | Owned wall time (s) |
| --- | --- | --- |
| R12 | `PASS_R12_FROZEN_SUCCESS_UNCHANGED_OUTPUTS` | 483.8980868 |
| R13 | `PASS_R13_FROZEN_COMMON_TRACE_SUCCESS` | 56.6939531 |
| R14 | `PASS_R14_FROZEN_COMMON_ROUTE_SUCCESS` | 57.8818058 |

R12 keeps its original endpoint control at face(1,50,304), axis0, never enters
common mode, and all eight arrays are byte-exact to its immutable old success.
R13 keeps 12 common cohorts and all 15 cache payloads through reconstruction and
both precommits. R14 keeps all 76 common cohorts, all 15 full-domain cache arrays,
prepared certificates and positive author keys unchanged; reconstructed alpha
and target pass their checks, and the 56 former conflicts are absent.
Source193, host numerics and strict CUDA/f32 runtime match before and after.

The wall times come from the completed owner records; they include compilation,
cache and process overhead and establish no acceleration. All three controls
advance zero physical time and reconstruct only the native assembly inputs,
not a complete physical restart state. They do not provide component or formal
FSI1/2/3 qualification, or independent full physical rollback equality.

Frozen replay execution manifest SHA:
`522616b78d1eac279e629d656d71a9c780df0dd4b64b18b8f2bdc9d9e86106cb`.
Each result is paired with its `root_artifact_readback.json` in
`systemic_pair_route_replay_r01/native_r12_r01`, `native_r13_r01` or
`native_r14_r01`; the chronological trace audit lists their hashes.

R15 terminated under owner record `fresh96_common_route_r15.json`:
PID/PGID 391, start Unix 1788687465.7436998, end Unix 1788689126.4022677,
exit 1; the owning tool session was 34446. The owner record SHA is
`5ff2a34635897e4f4b1ca42446bac3492f457498d16dbfdab59e7baab4bab66d`.
Its output is `hard_closure_diagnostic_5dcba96__r15` under the external
`robustness_work` root. Execution manifest
`fresh96_r15_preparation/execution_manifest.json` SHA:
`0b4d21ddd21facf317882d1ed5d116643c41bc3f00ed21128e18d866b564c6fa`.

The run used fixed112, 4x96x400, solid200, dt0.005 and eight requested steps
from zero. Five were accepted at t = 0.005, 0.010, 0.015, 0.020 and 0.025 s.
Step 6 failed at trial t = 0.030 s, during assembly 99. Native precommit
reported four target conflicts from `prepare_pair_arbitration`; the first
face is (0, 50, 342), axis 2, path 0, claim_count 2. No root cause is inferred
from this signature alone. The full capture contains 144 owner arrays plus
nine argument arrays, taken before native cleanup, with no capture errors.
The separate input manifest contains 207 fields and is complete.
`r15_execution_audit.json` reports `WRAPPER_RAISED_ORIGINAL_EXCEPTION`,
zero audit errors, and unchanged Source193 and 25 dependency maps.

The host-only terminal auditor passes with
`run_outcome=FAILED_WITH_VALIDATED_ACCEPTED_PREFIX`:
all five original candidate and native-record validators pass, with 89
CSV fields matched per record. Accepted fluid and solid time each totals
0.025 s. `fresh8_diagnostic_passed=false` and `formal_benchmark_pass=false`.
The reported rollback outcome is `restored`, with derived search/boundary
state requiring rebuild; `rollback_full_state_independently_verified=false`.
The audit executes zero physical steps and imports no Taichi or solver.
Accepted-prefix work counters exclude the final unaccepted trial.
The capture is diagnostic evidence, not a restart checkpoint.

The initial host audit `accepted_time_audit_r01` failed the authoritative-cwd
guard before auditing and remains preserved. The corrected-cwd
`accepted_time_audit_r02/audit_report.json` is the passing failed-prefix audit.
Its pass does not imply that R15 completed or that continuous failures are fixed.

The terminal auditor is
`accepted_time_r15_preparation/audit_accepted_time.py`, SHA
`934c7fb8d98eeb66b45f64fa04967f7f415eaef084f656de8b93fb44c2d9ebcb`.
Preparation SHA:
`2514c2dbe096e0298d5a2279c7cad08b58e01d167b6821ba08d02ed2c8748f60`;
root/Astra acceptance SHA:
`e69b869b50cc377622a638e07bd893417a5b9c04ec536b19cbaef569bbfe9b63`.

| R15 artifact | SHA-256 |
| --- | --- |
| `r15_execution_audit.json` | `6885814b00492a80695c4c874a22c5d06ed8240c5fec5f17ed58fcb5dd737244` |
| `precommit_failure_capture.json` | `ed04e16ff4cb6601e45a9b686cc32ffc49437d0e7ab05114580924ed614c91ac` |
| `failed_assembly_input_manifest.json` | `07f7a6c538276d91704d6e0b0a613b85a131121e2c5898dffa9383d3e38a1a31` |
| `accepted_time_audit_r01/audit_report.json` (wrong cwd) | `acae76109f36acb5265151420d13e8f6de257c0f5b6c2d5a18197503dc2ac8a6` |
| `accepted_time_audit_r02/audit_report.json` (passed audit) | `f9c64204171025fb4cfbea0c669c5bad4088c67577346ce30d4b0db95377a433` |

## Source-current coarse S0 check remains unverified (2026-09-06)

The bounded read-only inspection of original R04, `frozen_trace_48x288__r01`,
and the later band/common control evidence found no recorded source-current
recheck of the original 4x48x288 step-8/call-95 obstruction. Later production
band and R12/R13/R14 controls use 4x96x400 and different poses or assembly
inputs. Their passes do not qualify this original coarse operator.

R04 preserves the failing pose after seven accepted steps, raw 112-marker
targets and solid x/v/F/rest/fixed data. It does not preserve the complete
search, canonical, pressure and physical restart state. The old coarse helper
also raises solid100 to 200, uses boundary time 0.035 s, restores the old
obstacle mask and calls a removed private assembler. It cannot be reused
unchanged for a source-current check. Its saved coefficients and dual/minimax
obstruction remain valid evidence for those historical fixed systems only.

The next bounded check requires a fresh, source-pinned zero-time adapter:
preserve the R04 pose and raw physical targets with grid 4x48x288, automatic
112 markers, solid100 and dt = 0.005 s. Rebuild current case geometry,
dynamic volume, search, protection and band topology; do not overwrite the
resulting masks with historical masks. Use the current post-solid
`_stabilize_hibm_solid_band` route with nine passes and external faces at
the failing step-8 context time, 0.040 s. The R09 post-band helper's eight-pass
budget applies to a different starting phase.

Capture the current native operator and actual committed f32 H/global residuals
with a separate audit operator, without changing the live prospective candidate.
Retain all 112 physical markers and the existing 1e-6/1e-4 m/s thresholds,
and preserve original exceptions separately from capture errors. Recompute
current rows, coefficients and topology; the old 320 hard-row count and old
dual certificate do not establish the new operator's feasibility. No fluid,
solid or coupled physical time advances in this check.

This preparation has not been implemented or executed. Even a later passing
reconstructed-pose check would not reproduce R04's complete original physical
state or qualify registered S0; coupled current-source evidence and the formal
entry gates would still be required.

## Exact stopped R15 artifacts

Artifact root: `/mnt/c/Users/lizhu/.codex/visualizations/2026/09/05/01a06f3c-1b43-7841-ac96-ca57e7365e03/robustness_work/resume_live_20260905_01`.

- `r15_fallback_contracts_r01/root_review_acceptance.json`: SHA7b64b9d8ef437e87b1282f65ae3bfcdd11a3fa66683862262da719f0f358d4df.
- `r15_fallback_contracts_r01/applied.json`: SHA8e6fdd9c6dd82b517829b89577a3007e7acaa08754e14a17f38667dc69ae0e93.
- `r15_fallback_contracts_r01/execution_green.json`: SHA9138cdd96ef0cb6531953b67fe87862600677dccbfbf771d07cbfd6d5e2fa752;61 dependencies. Preserve this occupied r01 manifest.
- `r15_fallback_contracts_green_r01.json/.log` and `r15_green_quota_stop_r01.json`: actual interrupted process ownership and stop signals. No terminal native `process.json` or completed case exists.
- `r15_consumer_baseline_r01/native_baseline_r01/report.json`: completed original-source2/2 comparison baseline.
- `r15_frozen_red_readback_r01/readback_r01/report.json`: complete original R15 frozen rejection readback.
- `r15_candidate_union_probe_r01/native_query_r04/report.json`: SHA3dca4de1635b05a77a488f939a7d638174ea2b8678f5d8e12524abaebe33651a; source/input-matched geometric query, no publication or physical advancement.
- `r15_fallback_full_replay_r01/READINESS.md`, `host_preparation.json` and `r15_full_replay_review_wip_at_quota_stop.md`: host-only package and outstanding independent review gap. No executable frozen full-replay manifest exists.
- Current R15 fixture SHA: `c41415690040c7a10533d29aeacde9ce352403203345ad0c679b95d3cdecb727`.

The resumption must preserve the completed A50 baseline. Do not revert source or rerun the old baseline merely to regenerate evidence. Create a fresh focused-run manifest retaining all source/host/test pins and changing only the explicitly reviewed output reservation.
