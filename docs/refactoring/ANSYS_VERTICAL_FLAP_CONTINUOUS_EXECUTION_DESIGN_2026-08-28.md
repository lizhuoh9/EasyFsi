# Continuous FSI execution: design and acceptance contract

Date: 2026-08-28; updated 2026-08-30. Status: **r46 source-matched
fresh50 and K50-to-K52 continuation passed; compiler-identity hardening is
host-tested, while a fresh post-hardening CUDA campaign and long physical soak
remain incomplete**.
Authoritative tree: `/home/zhuohengli/worktrees/HIBM-MPM-r21-validation`.

The user authorized backup followed by refactoring. The material reference map,
adjoint load transfer, geometry-owned MAC aggregation, accepted-state checkpoint
and run-attempt lifecycle are implemented. The current evidence and remaining
boundary are in the final dated section. r46 completed an exact fine50 run, a
source-matched no-advance restore, and a source-matched K50-to-K52 continuation.
That evidence predates the compiler-identity source patch and therefore cannot be
reused as a post-patch source-matched campaign. The locked Fluent comparison is a
diagnostic comparison, not a declaration that Fluent is ground truth. Historical
dated sections below retain their original evidence and failure boundaries.

## Historical r37 status before the authorized material refactor

This supersedes the proposed step-46 mode132 permutation patch. The r37
**50/50 accepted and durable steps** passed the strict 108-source, physical-time,
pressure/FSI/closure, field/history and complete-checkpoint gates. The r37 107 new
tests, 71 existing regressions and independent diff review also passed.
The later B9-to-K50 recovery completed normally and passed independent complete-state,
field/history and physical-time audits. Its trajectory is close to, but not
byte-identical to, the uninterrupted reference. The K50-to-K200 extension then
failed while attempting K51; no new step was accepted and the durable K50 state
remained exact. The earlier unexplained K21 interruption remains unexplained.
The small-grid run has a large, non-qualifying Fluent discrepancy; fine-grid
accuracy and physical-200/5000 stability are not established. Current evidence
and the proposed architectural boundary are in the final section below.
Dated sections below preserve earlier failed attempts; the current geometry
contract is the r37 bounded disk, not the superseded hull.

## Why the implementation route changes

The old component-face preparation enumerates four storage slots, retains two
representative authors and admits selected direct/relocation arrangements.
Relocation first compresses multiple valid source routes into one winner.
Consequently a change in storage provenance can reject valid geometry despite
unchanged physical constraints. Adding another arrangement does not remove that
structural dependency. Existing fail-closed behavior must remain until a complete
replacement certificate is validated; the target tolerance is not a tuning knob.

The replacement has two independent responsibilities:

1. Derive a face constraint from registered physical geometry, and verify every
   source route without using source multiplicity to define the target.
2. Durably commit the complete accepted FSI state so interruption does not erase
   prior accepted work. A saved field export is not a solver restart state.

A checkpoint cannot repair an invalid numerical method. Nor can finite tests
guarantee convergence for arbitrary meshes, parameters, contact or deformation.
The contract is precise admissibility, atomic rejection, bounded recovery, and
reproducible continuation of an explicitly validated problem class.

## Geometry-owned face assembly

Initial implementation domain: two-dimensional/extruded, non-interpolated,
registered finite segments with explicit topology. The 3D triangle, interpolated
and field-only paths retain their current semantics. Dispatch is by declared
problem representation, never a fallback after the new certificate fails.

- Preserve complete source-keyed direct and relocation routes. Record capacity is
  derived from grid-source/axis routing, not a constant number of authors per face.
- Validate each raw source, including duplicates; atomically accumulate face
  activation and counts. Do not select the physical target during this pass.
- At each active MAC face, scan the complete registered finite geometry using
  F64 projection arithmetic. Select the closest physical point before applying
  region, side or endpoint-support permissions. A nearer invalid/backside owner
  must not disappear so a farther convenient segment can win.
- A unique finite projection defines the target through its geometric weights and
  current vertex velocities, once, with alpha zero. Equal-distance distinct points
  are ambiguous. A common registered C0 endpoint needs a geometric/topological
  certificate; raw half-edge weights and target averaging are not certificates.
- Revisit every source against the final owner and a unique local connected patch.
  Local support, orientation and a common open normal half-plane exclude remote
  connections and folded-back branches; global connectivity or region equality
  alone is insufficient. Each source normal is checked against its own primitive.
  Raw source/anchor classification retains its three strict source-support checks.
  The owner projection and every actual/alias connector use one strict
  face-global Euclidean disk circumscribing that active-plane source box
  (scalar radius unchanged). A distant unused endpoint is not a connector.
  This intentionally broadens geometric connector support, while retaining
  nearest-owner-before-permission ordering and every physical target tolerance.
- At a true convex vertex, use the outward normal cone, not positive dot products
  with every incident normal. This permits valid obtuse convex corners.
- Projection-only tip/cap aliases require the explicit binding pairs [4]/[6] and
  [5]/[7], with live coordinate, velocity, role and registered-edge checks. Neither
  equal pressure-owner indices nor coincident coordinates create alias authority.
- Reuse marker compatibility closure, pressure semantics and the existing sole
  eight-field ledger publication. Any failed source leaves public fields unchanged.

This is a numerical-method migration, including the single-source case; it is not
a change of the absolute target tolerance. Existing arrangement-specific tests
must be classified individually. A test that mutates a source into invalid
provenance remains negative even if its A=C label is no longer intrinsically bad.

The body-intercept discussion in Mittal et al. motivates geometry-first ownership
and explicit handling of nonunique closest points. The particular MAC assembly
above is our design inference, not a claim that the paper validates this code:
[Mittal et al., JCP 227 (2008), section 2.2.2](https://engineering.jhu.edu/fsag/wp-content/uploads/2014/06/JCP_sharp_interface.pdf).

## Accepted-state continuation

The durable boundary is after physical acceptance, retained IQN history, initial
guess acceptance, active-Kalman commit and creation of the accepted history row,
but before external artifact observers. No trial state is a checkpoint.

Required state includes full fluid velocity/pressure/SST/obstacle/boundary fields,
solid x/v/C/F/position-increment residual, current marker and projection-only
geometry, accepted canonical ledger and authority generation, physical time and
feedback flag, IQN secants, both controller states and fixed-size runner counters.
Derived caches are invalidated/rebuilt explicitly; they are not silently mistaken
for independent physical state. Constructor/static geometry and oracle content
belong to strict identity. The resume target is a total step count N; state K
continues K+1 through N, never repeats K or advances time by a shortened macro dt.

Persistence uses immutable numeric NPZ generations plus checksummed incremental
accepted-step journal records. A small latest manifest is replaced only after all
referenced payloads are complete. Validate schema, identities, hashes, names,
shapes, dtypes, finite physical values and contiguous history before restoring any
runtime field. No pickle, arbitrary type imports or path traversal is accepted.
Metadata diagnostics may have explicitly tagged missing/nonfinite values; that
does not authorize nonfinite physical arrays. A failed save preserves the previous
durable generation and reports durable and in-memory accepted progress separately.

Old-generation retention must be bounded for long runs, but only exact files owned
by the checkpoint store may be retired after successful publication. Process
interruption atomicity is the initial guarantee; power-loss durability is not
claimed without filesystem-specific fsync verification. Single writer only.

## Verification order and release boundaries

1. RED geometry contracts: single/multiple/>4 sources, all lawful arrangements,
   permutations, mirrors, axis rotations, unmentioned nearest primitive, local
   disconnection/foldback, explicit seam and convex corner, every corrupt source.
2. RED persistence/controller contracts: complete round trips, active-trial guards,
   copy isolation, invalid-state atomicity, write/replace failures, hashes/paths,
   strict identity and interrupted history. Demonstrate nontrivial predictor state.
3. Focused GREEN and regression, then independent inspected-diff review. Report
   actual counts/coverage; do not infer an 80% coverage gate from test counts.
4. Fresh source-matched preflow and strict CUDA short gates. Compare continuous
   execution with a separate-process interrupted/resumed execution, including
   physical fields, ledger, histories, controllers and consumed physical time.
5. Complete real 50 steps, interrupted/resumed 50, then a longer soak if earlier
   gates pass. Each accepted fluid and solid advancement consumes the full macro
   dt; retries never count discarded time. No tolerance relaxation, skipped failed
   steps or result reuse across changed source identities.
6. Publish measured results and a run/resume procedure. A 5000-step capability is
   not certified by extrapolating a 50-step prefix or timing estimate.

## 2026-08-28 implementation snapshot and validation boundary

This section records source and focused-gate status only. It is not a numerical
success claim, a completed formal run, or evidence of a CUDA checkpoint resume.

The abandoned `mode132` / three-author arrangement-specific patch lane remains
documented in the r29 worklog as historical diagnostic evidence; it is not the
active production route. The active implementation dispatch is instead the
geometry-owned full-source A/B/C assembly. Its production dispatch is present,
but its source and numerical validation are still WIP.

The new `checkpoint_store` schema 2 implements an O(1) checkpoint head, an
incremental checksummed accepted-history journal, and bounded retention of two
numeric NPZ generations. It persists observer outbox entries with an accepted
state, requires strict source/configuration/geometry identities, refuses field
export NPZ files as restart input, and has a single-writer contract. Publication
uses same-process temporary-file/replace atomicity for process interruption;
this is not a power-loss durability guarantee without filesystem-specific
verification.

If backend restoration I/O fails, the newly built runtime is still unexposed and
is aborted rather than published. The previous durable generation remains
available. This is intentionally narrower than a fully transactional rollback of
every allocation or I/O action used to construct that fresh runtime.

Measured focused gates currently available are:

- Combined checkpoint host suite: 109 passed, 2 skipped in 21.62 s (runner 26,
  CLI 10, wrapper 29, store 30 with 2 skips, codec 14).
- Controller host suite: 85 passed in 6.94 s (earlier run).
- CUDA capture suite: 4 passed in 73.83 s. This did not execute an actual
  separate-process CUDA checkpoint/resume.
- Segment audit: 5 RED and 8 passed in 43.25 s; corner/cap audit: 2 RED and 4
  passed in 16.80 s. The later full registered-segment audit passed 19 strict-CUDA
  checks in 31.75 s (session 4431, exit 0). These gates do not replace a formal
  numerical run.

At this snapshot, current GREEN runs that were still in progress have unknown
results. The latest real numerical result remains r30's accepted 45/50-step
prefix followed by a fail-closed step 46; exact FSI50 has not passed. No actual
CUDA checkpoint-resume execution has yet been recorded.

## 2026-08-28 r33 source-freeze update

The source is frozen for the running r33 preflow at
`69f5d760278d258d58a4c06ece4340fe0518d254f964b13db8d9f1115bb72b3d`
(63 files). No source changes are permitted while that run is active. This
update supersedes only the dated “unknown” gate status above; it does not alter
the historical r30 or r32 failure records.

The r32 fresh preflow succeeded under the strict source gate (107 files): runner
`398.9819721 s`, wrapper `414.0400877 s`. Its first FSI step physically accepted,
but checkpoint publication failed because the production history row lacked
`time_s`: `memory_accepted=1`, `durable_accepted=0`, wrapper `366.6365083 s`,
progress `357.1897965 s`. There was no step export, checkpoint, or full-success
claim from that r32 FSI attempt.

The narrow repair adds the canonical physical timestamp to the real history-row
producer. Host evidence evaluates the selected actual `step`/`time_s` producer
expressions from that row and round-trips the resulting row through the accepted
checkpoint wrapper with a complete state fixture. It is not evidence that the
entire row schema or complete runner commit path has been executed. The optional
raw IQN trial-vector observer payload is now explicitly unsupported with an
accepted checkpoint and fails before runtime build; ordinary checkpoint observers
and standalone raw-IQN export remain supported. The numeric observer outbox and
numeric-only checkpoint codec are unchanged.

Review P2 for lazy geometry allocation is closed: legacy allocation would add
612 bytes/cell (`4.78125 MiB` at the current grid; `1.1953125 GiB` at `128^3`),
which is now avoided. Evidence: 6 genuine RED and 1 pass in `7.14 s`, 12 host
GREEN in `7.16 s`, and 71 geometry/lazy-regression tests in `188.67 s` under the strict-CUDA profile, including host-only tests.
Metadata tests added 12 passes in `7.05 s` for cap SST-gradient and 11 invalid-
history atomic cases. The stdlib-trace combined host suite is `271 passed, 2
skipped in 51.04 s`; local host line coverage is wrapper 86%, codec 91%, store
88%, initial controller 86%, and active controller 88%. These are neither
global nor GPU coverage. A test-only actual-entry `iqn_ils` correction passed
1 test (27 deselected) in `6.58 s`.

All reviewed host P2 items are closed, but true CUDA checkpoint commit/resume and
a full FSI50 remain pending for r33; this is not numerical success.

## 2026-08-28 r33 execution evidence and r34 boundary

This dated update preserves the r32 record above, including its missing-`time_s` checkpoint-publication failure.

- r33 fresh preflow passed its strict source gate: runner `377.7516787 s`, outer wrapper `391.7570719 s`. The following r33 FSI01 reached physical accepted step 1 but did not durably publish it (`memory_accepted=1`, `durable_accepted=0`): ordinary zero-dimensional Unicode snapshot-stage tags were rejected by the codec; progress was `370.7574285 s` and wrapper `380.3095066 s`. This is not a successful checkpoint, restart, or numerical continuation.

- The codec change makes Unicode scalars explicit opt-in JSON metadata for such stage tags. Physical arrays remain finite-only and numeric. The obsolete IQN compatibility guard was removed after the real full-runner commit, disk, and observer-replay checks passed in the 31-test subset reported below.

- A host-only synthetic checkpoint-store soak completed 5,000 accepted records in `417.0396941 s`; ten 500-record blocks took `39.7719`--`41.2474 s`. The final store had two NPZ generations plus 5,000 journal entries totaling `3,307,049 bytes`. The detached launcher's exit code was not captured. Its metrics were written only after all final assertions; an independent strict-load verification exited 0, and the Taichi runtime's `prog` remained `None`. This is not a 5,000-step physics, CUDA, or FSI guarantee.

- For F accepted steps, journal records contain current-step report deltas. The complete per-step state and fixed preflow payload P are still serialized per commit, O(F*P); P is 1 in the current case, not an F-dependent prefix. Restart intentionally performs O(F) journal/report materialization. These are finite resource boundaries, not a real 5,000-step claim.

- The unchanged geometry/dispatch/lazy suite passed 71 tests in `188.67 s` under the strict-CUDA profile; it includes host tests and is not “71 CUDA-only tests”. The completed ten-file host gate passed 301 tests with 2 skips in `54.18 s`; local line coverage is wrapper 86%, codec 91%, store 88%, initial controller 86%, active controller 88%, predictor 81%. Unicode codec coverage includes 27 cases including two encoder-negative cases. The 31-passed `7.55 s` real-producer/full-runner-commit/disk/replay/IQN/physical-negative worker gate is a subset of the 301, not an additional total.

r34 source is frozen at `bf0262db073f7a426b5aa9fd9bea9009dd63e4d4f528cea12365afcf82cc0b20` (63 files); its then-active strict-CUDA preflow subsequently completed, and its short-gate result is recorded below. No FSI50 claim is implied.

## 2026-08-28 r34 short continuation evidence and unresolved parity

This section records completed short gates after the preceding “running” status; it preserves all r31--r33 failures and does not waive any comparison result.

- Fresh preflow passed its 107-source gate: solver `235.8071233 s`, wrapper `245.5991915 s`. FSI01 passed 1/1: solver `210.8294641 s`, wrapper `221.1777761 s`. Continuous FSI02 passed 2/2: solver `237.5784015 s`, wrapper `247.2869525 s`; warm step 2 was `8.0053065 s`.
- A separate-process resume from the same FSI01 output to step 2 passed numerical and full-state audit, wrapper `227.8522954 s`. All 116 arrays and five U0D Unicode scalars were read-only; journals were complete and normal two-NPZ retention held.

- Independent continuous-FSI02 versus FSI01-to-resume-step2 comparison is **not bitwise parity**. Step 1 already differed before resume, at about `1e-19` in time. `continuous_r34_resume02_comparison.json` records `differences_measured_not_waived`, with 38/116 differing arrays and 2/61 differing control scalars. This evidence neither infers a CUDA cause nor waives parity.

The r34 continuous FSI50 subsequently failed; the dated persistence boundary follows.

## 2026-08-28 r34 continuous-FSI50 persistence failure and r35 next gate

The 50-step run reached seven durable frames and physically accepted step 8, then failed publishing the manifest with `os.replace` `WinError 5`; progress elapsed was `348.5026734 s`. `failure.json` is retained. This is neither FSI50 success nor a source-matched eight-step prefix: no such prefix was captured. The outer driver then hit a GBK `UnicodeEncodeError` while printing decoded stderr, so its exit was 1, it produced no `result.json`, and no wrapper elapsed is reliable. The staging driver now writes UTF-8/log output before printing.

- The cause is confirmed on the WSL UNC path: an ordinary Windows native Python read handle blocks destination replacement with `WinError 5`. The production shared `atomic_file.py` helper retries only Windows codes 5/32/33, at most eight attempts for the same replace operation and at most `0.95 s` total backoff, across all three checkpoint-store replacements and the CLI JSON/NPZ publishers. The old head remains published until manifest swap succeeds; no retry advances physical work. A persistent lock still fails and may leave an unreferenced orphan. This matches [Microsoft `FILE_SHARE_DELETE` semantics](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew), where delete access permits rename.

- Evidence: 6 manifest-contention REDs, 3 helper-boundary REDs, and 2 CLI REDs, with the old JSON positive retained; focused result 60 passed, 2 skipped in `18.17 s`. Actual WSL-UNC contention result: 20 passed in `17.56 s`. The completed 11-file host trace is 321 passed, 2 skipped in `75.62 s`; its missing-coverage mode printed 100%, so this is not a coverage claim. Reviewer accepted the helper and readonly restore harness.

r35 source is frozen at `e02c5e1150995ed2471c4b880d2765758d138739c52d90102aac3fa332725ab2` (64 source files including `atomic_file`); new strict fresh preflow is running (PID 37272, session 81914). Kernel/source geometry and physical tolerances are unchanged. r35 requires fresh source-matched preflow before FSI50; no result is claimed.

## 2026-08-28 r35 actual preflow and 31-step failure boundary

This dated entry supersedes only the preceding r35 “running” status. It does
not alter any historical r30--r34 failure, host-only, or parity boundary.

- r35 fresh preflow passed the strict 108-source gate: solver
  `235.48416759999236 s`, outer wrapper `245.7792247000034 s`. This is a
  source-matched preflow result, not an FSI50 result.
- The subsequent continuous FSI50 failed after 31 accepted/durable steps
  (`31/50`), at attempted physical step 32: runner last-step elapsed
  `564.8447479 s`, wrapper `581.146270000012 s`. The strict full-source
  geometry audit reported physical generation 331, `owner_reason=2`
  (strict-support), owner markers 14/15, source markers 15/16, face
  `(0,4,28)`, component axis 1, and count 8 (faces plus sources). This is a
  fail-closed geometry rejection, not a 50-step success.
- The durable head is K31, generation
  `7c7d4403d6ad4862b5a9709efa5fad7d`. Before any replay branch, the whole
  failed output was copied recoverably to staging as
  `continuous_r35_failed31_before_replay`: 100 files, 10,345,146 bytes, with
  every SHA-256 verified. The original failure artifacts remain evidence.
- A staging K31-to-K32 diagnostic/recovery run is in progress. Its post-restore
  exact-state audit and geometry dump have no result yet; this entry makes no
  restored-state, resumed-step, or numerical-parity claim.
- The 5,000-record checkpoint result remains a synthetic host persistence soak
  only. It is not evidence of 5,000 physical FSI steps or of a 50-step run.

Host evidence recorded in `checkpoint_host_r35_line_coverage.json` is 321
passed, 2 skipped in `75.62 s`. Its selected-host executable-line measurements
are accepted-state 86.5574%, active-Kalman 88.5417%, initial-guess controller
86.6388%, atomic-file 100%, checkpoint codec 91.8919%, and checkpoint store
88.4837%. The method is stdlib trace hits intersected with compiled executable
lines; it is explicitly not branch, GPU, or repository-wide coverage.

## 2026-08-28 captured support mismatch and projection-closed locality

The source-frozen `reproduce32c` restored K31 with zero saved-state mismatches,
then reproduced the same source/face/owner rejection and saved 53 directly read,
dtype/shape/SHA-256-verified arrays before rollback. Wrapper elapsed was
`282.50864770000044 s`, exit 1; no step 32 was accepted. The fresh transaction
generation was 3, versus 331 in the uninterrupted run; these are scratch
transaction counters, not physical time or accepted-step counts.

At face `(0,4,28,1)`, the global nearest point on segment 14/15 is
`0.00237753208341789 m` from the MAC center. The valid source anchor is vertex
15, `0.002382114887093532 m` away. It is inside the original source and face
boxes with z margin `2.2076070308688772e-6 m`; the nearest point is outside
both boxes by `1.5881562795012558e-5 m`. Thus changing the support center alone
does not fix the conflict. The one actual connector is the valid local vertex
15. This is a mismatch between Euclidean closest-point ownership and a box
point-membership requirement, not a target-tolerance or roundoff issue.

The reviewed next contract, pending RED/GREEN and fresh numerical validation,
separates classification from aggregation locality. Keep all three original
strict source/route support predicates, all raw provenance/ray checks, and B's
global nearest-owner ranking. For the selected point P, use the face-global
minimal convex patch `C = conv(open D_face union {P})` for owner/corner/alias
and every connected-path connector. Require P strictly inside the Euclidean
bounding disk with squared radius `sum(r_i**2)` over the active axes (scalar
support retains its original squared radius). No new radius parameter is used.

For an anisotropic box, Q belongs to C when Q=P in the active plane, or the ray
`P + s*(Q-P)`, `s >= 1`, intersects the original OPEN box. Open slab intervals
must overlap strictly; tangent hull facets, parallel boundary rays, nonfinite
points and points outside C reject. When P is inside D, C is exactly D. This
preserves former accepted local paths but intentionally changes connector
locality: a visible former box-boundary connector can become interior to C.
Raw anchors and source centers on the original boundary still reject. This is
a geometric certificate redesign, not a claim of unchanged connector support.

This patch is spatially bounded; it does not promise geodesic-length, curvature,
arbitrary-mesh or 5,000-physical-step guarantees. Marker/pressure/FSI tolerances,
physical macro time, sole-ledger publication and strict restart identity remain
unchanged. New source requires fresh source-matched preflow before formal FSI50.

## 2026-08-28 r36 implementation and completed regression gates

The projection hull is now implemented in `component_face_segment_audit.py`.
The real three-vertex F32 fixture first reproduced `owner_failure=2` on strict
CUDA (`projection_support_r36_capture_red_b.xml`, 22.92 s). An earlier test
collection/import error is not counted as a RED. After implementation, all
66 new strict-CUDA cases passed in 23.41 s, including axis/storage permutations,
an inserted connector outside the old box but inside C, exact exposed facets,
strict owner bounds, nonfinite inactive coordinates, and the three unchanged
raw-box boundary negatives. New plus existing geometry regressions passed
137 tests in 172.95 s; that mixed host/device suite is not a GPU coverage claim.

The CLI clears stale failure diagnostics only when merging a new running event,
without changing the existing input object or dropping diagnostics in the new
event. Both progress observers use it; the full CLI module passed 14 tests in
13.79 s. Independent inspected-diff review found no blocker in the implementation
or the corrected staging process/source guards. The r36 64-file source lock is
`5b7134212538026988aed62a788b153d6ea20a966f1f83e4a1ba91792a383779`.
Fresh source-matched preflow is running; this entry does not claim FSI50 success.

## Run and resume procedure (single writer, unchanged source/configuration)

On the validated Windows host, use native Taichi Python against the authoritative
WSL source path. The Windows mirror is not the working tree. The example below
retains the r36 physical and numerical settings; choose new, unused output names.
It documents the interface, not a claim that a 5,000-step physical run was tested.

```powershell
Set-Location '\\wsl.localhost\Ubuntu-22.04\home\zhuohengli\worktrees\HIBM-MPM-r21-validation'
$solverPython = 'D:\working\taichi\env\python.exe'
$solverCli = 'validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/scripts/run_our_solver_vertical_flap.py'
$env:TI_CFG_OPTIMIZATION = '0'
$env:TI_OPT_LEVEL = '1'
$env:TI_ADVANCED_OPTIMIZATION = '1'
$env:PYTHONUTF8 = '1'
$common = @(
    '--dt-s', '0.0005', '--preflow-steps', '1',
    '--preflow-convergence-mode', 'single_step_legacy',
    '--grid-nodes', '4', '32', '64', '--solid-particle-counts', '1', '12', '4',
    '--marker-count', '12', '--flow-projection-iterations', '1080',
    '--flow-hibm-marker-compatibility-closure-tolerance-mps', '1.1e-6',
    '--flow-post-dirichlet-consistency-projections', '1',
    '--flow-cg-preconditioner', 'fv_multigrid_light',
    '--flow-pressure-solve-failure-policy', 'raise',
    '--coupling-mode', 'iqn_ils', '--initial-guess-mode', 'carry_forward',
    '--fsi-max-iterations', '16', '--fsi-absolute-tolerance-mps', '0',
    '--fsi-relative-tolerance', '0.001', '--iqn-history-limit', '8',
    '--iqn-initial-picard-relaxation', '0.5', '--iqn-svd-relative-cutoff', '1e-10',
    '--kalman-mode', 'off', '--flow-predictor-substeps', '1',
    '--young-modulus-pa', '1000000', '--span-reduction', 'mean',
    '--taichi-offline-cache-dir', 'validation_runs/.taichi_cache/ansys_vf__segment_agg__20260828__r30',
    '--profile-wall-time'
)
$preflowRun = 'validation_runs/solver_soaks/my_source_matched_preflow'
$canonicalRun = 'validation_runs/solver_soaks/my_accepted_fsi'
$resumeAttempt = 'validation_runs/solver_soaks/my_fsi_attempt_002'
$targetSteps = 50  # Total target; 5000 is a separate, unvalidated longer campaign.

# Execute once for this frozen source and configuration; require exit code 0.
& $solverPython $solverCli @common --steps 0 --output-dir $preflowRun --preflow-snapshot-out "$preflowRun/state"
# Fresh execution: canonical is the completed initial-run root and receives its
# checkpoint, accepted step artifacts, and initial-run records.
& $solverPython $solverCli @common --steps $targetSteps --output-dir $canonicalRun --save-step-fields --preflow-snapshot-in "$preflowRun/state" --fsi-checkpoint-out "$canonicalRun/checkpoint"
# After an interruption, use a new, previously unused attempt directory. Do not pass
# either explicit checkpoint flag: --resume-run-dir owns canonical checkpoint selection.
& $solverPython $solverCli @common --steps $targetSteps --output-dir $resumeAttempt --resume-run-dir $canonicalRun --save-step-fields
```

Run each stage separately and inspect its exit/progress before the next one.
`--steps N` is the final total, not N additional steps: a durable K resumes at
K+1 and finishes at N. A K-to-K invocation restores and audits the accepted
state but performs no physical advance. A K-to-N invocation begins at K+1;
every later accepted fluid and solid advancement consumes the full macro dt.
The target is never interpreted as an increment.

The canonical run directory is the completed initial-run root and accepted-state
authority. It owns that initial run's checkpoint store, accumulated
`step_fields/` and `step_history/` evidence, plus its metadata/manifest,
progress, final summary and run-specific exports. Later resume attempts never
mutate those initial-run terminal files. Each resumed `--output-dir` is one
isolated process attempt and must be a different, fresh and empty directory.
It owns that resume attempt's metadata/manifest, progress, final summary,
failure/interruption diagnostics and any run-specific exports. A failed resume
attempt therefore never moves, archives, overwrites or otherwise mutates prior
canonical evidence. Direct `--fsi-checkpoint-in` use is rejected for this
continuation route; `--resume-run-dir CANONICAL` derives the canonical
checkpoint input and output and rejects ambiguous explicit checkpoint flags.

If an observer replays an already accepted K artifact, exact semantic equality
(keys, dtypes, shapes and values) preserves its existing bytes. A semantic
mismatch fails closed. If only one member of the accepted fields/history pair
exists and is semantically equal, the missing peer is repaired without replacing
the existing file. Keep the checkpoint manifest, its referenced immutable
generations and every referenced journal record together. `step_fields/*.npz`
is a reduced observer export and is deliberately rejected as restart input.
Do not change the canonical source or physics configuration, bypass identity
checks, reuse an attempt directory, or run two writers.

Normal checkpoint retention is two NPZ generations plus one journal delta per
accepted step; requested observer exports also accumulate with step count.
Restart materializes the accepted history, so neither disk nor host memory is
claimed constant for unlimited N. A process interruption may leave an unreferenced
generation; it is not a committed step and should not be used to advance time.
This workflow protects accepted work against process interruption, not untested
filesystem power loss. A genuine geometry/physical rejection still stops with
a durable accepted boundary; it is never converted into an accepted step.

## 2026-08-28 r37: one bounded Euclidean aggregation domain

This section supersedes the r36 projection-hull contract, whose history above
is retained. r36 source-matched preflow passed the 108-source check (wrapper
215.04502249999496 s), but continuous FSI50 failed at step 49 after 48 accepted
and durable steps (wrapper 673.7183566000022 s). All 53 exception-time arrays
were independently verified against their hashes, shapes and dtypes.

The unique registered subarc from source edge 14 to owner edge 12 crosses
vertex 13. Its normal fan spans 0.5640962694098663 rad, with valid edges and
no alias or branch. That connector is 2.548860933 mm from the face, strictly
inside the original box's 3.90625 mm diagonal disk, but outside the r36 hull.
Consequently, admitting only the nearest projection did not give a sufficient
geometric domain for a normal curved boundary.

The current geometry domain is the strict face-global disk
`sum(active_delta_i**2) < sum(active_radius_i**2)`. It is the smallest
Euclidean disk containing the original open anisotropic source box and uses
the same distance metric as independent global nearest-owner selection.
Scalar support retains `radius.x**2` exactly. All owner points, owner-corner
vertices, aliases and traversed connectors use this one disk. Raw anchors
and source centers still pass all three original strict-support predicates.
The certificate additionally requires the existing unique qualified registered
path, region/normal/alias/side checks and full raw count. The disk is not a
permission to attach arbitrary geometry that happens to lie inside it.

The old hull helper and its duplicate raw-route projection are removed.
This intentionally broadens connector geometry, not source classification or
target/pressure/FSI tolerances. It does not guarantee bounded geodesic length,
arbitrary topology, self-contact resolution, or 5,000 physical steps.
Real-capture RED: six failures, 16.58 s. Symmetric curved-bridge RED: 12 failed,
19 passed, 27.24 s. New strict-CUDA-focused gate: 107 passed, 70.57 s.
The unchanged existing regression gate passed 71 tests in 184.80 s; independent
inspected-diff review found no blocker. The 64-source r37 freeze is
`65af075de01d90e593bf03ed2e5e6f57a9aac10e54e5e998b5191e294a0c1e3a`.
Fresh source-matched preflow passed the 108-source gate: solver
245.96080210000218 s, wrapper 256.79527399998915 s. Continuous50 then completed
all 50 accepted/durable steps with exit 0: solver 769.536465600002 s,
CLI 771.245453999989 s, wrapper 782.3919382000022 s (preflow separate).
Cold-start-to-first-step was 261.5633334000013 s. The 49 warm increments had
mean 10.362905155102027 s, median 10.311846800002968 s, range
8.218361999999615--15.661034800010384 s; these are measured, not extrapolated.

Independent validation checked all 50 field/history pairs and pressure, FSI,
closure and complete fluid/solid macro time. Final accepted time is 0.025 s.
Disk reload verified 50 linked records, 116 arrays including five opt-in Unicode
metadata scalars, exact source/config/geometry identity and two retained NPZ
generations with no orphans. The state arrays occupy 3,052,090 uncompressed
bytes; the two compressed generations occupy 1,878,415 bytes. Raw IQN trial
vectors were not exported, so the field audit covers scalar histories, not
a raw-trial-vector comparison.

The complete reference is preserved at
`validation_runs/solver_soaks/ansys_vf__continuous50_reference__continuous__20260828__r37`.
The paired fault-injection run began by installing an exact copy of the captured
K8 prefix in the original output path. The K8 generation is
`3ac89cb7bc5e4923bcae8632bb5a2458`; K8 is a verified journal ancestor of the
completed K50 reference. The post-restore K8 audit passed with no mismatches.
The later observations and the current stopped K21 state are recorded below.
No resumed-trajectory parity, physical-200/5000 result or Fluent comparison is
claimed by the continuous50 result.

## 2026-08-28 r37: read-only restart discrepancy analysis

No production source, numerical tolerance or physics configuration changed
during this analysis. The 64-file source lock above still matches. Analysis
artifacts are under the host staging directory
`C:/Users/lizhu/.codex/visualizations/2026/08/27/01a0434a-786c-7bc2-a8c6-1a381dda007a`.

The first paired run deliberately terminated its own child only after durable
K9 publication. Its actual child exit was 1 because termination was requested,
not a normal numerical exit. The wrapper completed its fault-injection checks
in 271.0909398000076 s. K9 generation
`5d978254ae0944b3812fc53fade03f4d` and its complete prefix are preserved in
`continuous_r37_interrupted9_prefix`. The original continuous K9 is preserved
separately in `continuous_r37_same_trajectory_prefixes/step_0009`.

The comparison is `differences_measured_not_waived`: 33 of 90 array entries
(including duplicated state/outbox mirrors) and 2 of 61 control scalars differ.
All physical-time ledger values and the source/config/geometry identities match.
Measured maxima include fluid velocity 2.855062484741211e-5 m/s, pressure
3.670687681136542e-4 Pa, solid velocity 6.677582859992981e-6 m/s and solid
position 3.725290298461914e-9 m. These inter-run differences are not the
within-run marker-closure residual and are not assigned an invented tolerance.
The two initial-guess diagnostic differences are computed from the accepted
result in `InterfaceInitialGuessController.accept_step`; they are not evidence
of different initial-guess configuration.

Three distinct findings must not be conflated:

1. **Export packaging:** the replayed `step_fields/step_0008.npz` has a different
   whole-file hash, but all 41 decompressed NPY members, array shapes/dtypes and
   data bytes are exact. Only 28 ZIP-member DOS timestamps changed. The original
   full-file integrity gate correctly detected a byte change; no gate was relaxed.
   See `continuous_r37_k8_step8_npz_comparison.json`.
2. **Advanced-state numerical differences:** journal scalar records K1--K8 are
   exact. At K9 the first fixed-point evaluation already differs: residual RMS
   0.033168228009225635 versus 0.03316800686826735 m/s; candidate RMS
   0.018113972084486195 versus 0.018113629923403658 m/s. Both paths use three
   trials and identical CG/advection/SST/solid work counts. Saved trial work
   reports differ only in wall times; they do not expose the first diverging
   kernel. See `continuous_r37_k9_trial_diagnostics_comparison.json`.
3. **Unfinished continuation:** the subsequent K9-to-K50 attempt passed its K9
   saved-state restore audit, then left durable K21. At reconnect there was no
   Python process and no result/failure/geometry-error artifact. Its cause and
   exit code are unknown, not a diagnosed solver rejection. The immutable
   `continuous_r37_unfinished21_prefix` contains 68 hash-verified files. Independent
   strict reload using identity derived from the continuous50 reference, preflow
   and current source lock verifies 21 contiguous journal rows, exact accepted
   time 0.0105 s, 90 read-only arrays with finite numeric values, and two live
   NPZ generations with no orphan. See
   `continuous_r37_unfinished21_prefix_audit_v2.json`; v1 is retained but its
   `taichi_imported` label was incorrect. v2 records import=true and runtime
   initialized=false. This is persistence evidence, not completed paired FSI50.

The actual saved-state restore audit excludes byte equality for rebuilt
projection-only cap rows and runtime scratch. It must not be represented as proof
that every input to the next kernel is identical. Both normal and resumed
`evaluate_trial` paths restore their captured base before trial 0. Marker restore
then rebuilds cap position, velocity, normal, area and owner rows from the same
physical rows and static binding and clears cap traction/force; unpersisted cap
rows are therefore not a demonstrated state omission. A direct hash of their
rebuilt values is still absent. Both paths invalidate pressure warmstart; solid
restore rebuilds surface geometry and clears external force; the marker-PCG
preparation resets its multipliers and scratch. SST seeds its work arrays from
live k/omega before transport. No concrete missing-state or stale-first-read
cause has yet been proved.
MPM P2G uses parallel f32 atomic accumulation, whereas marker-force scatter is
explicitly serialized. Taichi documents possible parallel nondeterminism, but
that general mechanism does not establish the cause of this observed difference:
[Taichi serial-execution debugging](https://docs.taichi-lang.org/docs/debugging#serial-execution).

The next causal control is another exact immutable K8-to-K9 replay, not an
independent fresh0-to-K9 trajectory. Capture read-only stage fingerprints after
restore plus initial guess, boundary assembly, SST/predictor/projection, and
before/after MPM to find the first divergence. A repeat that differs only proves
variation within the same restore mode; it does not by itself identify atomics
or rule out an omitted field. Do not change tolerances or silently regenerate
reference states. The physical-200 extension remains paused.

Long-run acceptance has three separate obligations: physical/numerical health
under the original gates; complete accepted-state persistence and recovery;
and measured trajectory reproducibility. None substitutes for the others, and
none is established for arbitrary geometry or 5,000 physical steps by this run.

## 2026-08-28 r37: recovery complete, K51 rejection, material-surface boundary

This section supersedes earlier instantaneous run status, not the retained
historical failures. Detailed timings, hashes and measured errors are recorded
in the [r29-onward worklog](../validation/ANSYS_VERTICAL_FLAP_SEGMENT_AGGREGATION_R29_WORKLOG_2026-08-28.md).
There is currently no numerical process. Production solver source remains r37.

### What has and has not been proved

- A complete recovered50 reference is preserved separately from both the original
  uninterrupted50 and the failed extension. After the K51 failure, all 116
  accepted-state arrays, 61 control scalars and physical-time ledger values are
  byte/value exact against that recovered K50. Durable recovery is not the cause
  of this diagnosed geometry rejection.
- The source-matched K8 A/B stage replay found exact inputs through the first
  boundary-assembly entry. Differences then include queue/append ordering and
  compact labels; canonical pressure rows and component membership agree. The
  first directly observed MPM-state differences follow the first MPM substep.
  This supports a floating-point/order mechanism but is not proof that every
  hidden state is identical. Cross-run equality is measured, not waived.
- The K51 dump contains two independent rejection classes: 20 faces at edge 4--5
  whose geometric normal was locally flipped against material-feedback normals;
  and four faces at edge 7--8 that are genuinely inward by about 8.46 micrometres.
  The latter cannot be fixed by removing the first flip. Original strict support
  is sufficient at the first face; target tolerances and side gates remain intact.
- The dump contains no obstacle mask or synchronous solid F/substep trajectory.
  Therefore it does not establish the two adjacent cell masks or uniquely assign
  causality to marker drift, material strain, or coarse spatial resolution.

### Root mechanism reproduced independently of step number

The actual ANSYS feedback call advances marker position by
`old_marker_position + macro_dt * interpolated_final_particle_velocity` after
the solid has already accumulated all accepted physical substeps. Its normal
is independently interpolated from the particles' F-derived normals. These
are not a single material surface representation.

The unchanged production kernel, executed on single-thread CPU with only the
runtime initializer isolated, passes six signed/three-axis constant translations
but fails six exact-binary two-substep round trips and three macro-partition
invariance tests. This is a material-motion consistency failure, not a CUDA
accuracy measurement and not proof that it alone caused K51.

Cap positions are `1.5*x_tip - 0.5*x_previous`, while cap velocity copies only
`v_tip`. Twelve axis/direction uniform-translation controls pass; twelve affine
velocity derivative cases fail. In r37 cap pressure is disabled, so this cannot
be called an active cap-load error; the cap does participate in projection geometry
and wall velocity. An enabled cap-load path must eventually use the same adjoint.

### Proposed architecture: one material mapping and its adjoint

This is a proposal awaiting scope confirmation, not a landed solver fix.

1. Build a fixed, low-order reference material stencil W. Require partition of
   unity and first-moment reproduction: `W*1=1`, `W*X=X_gamma`. Reconstruct
   `x_gamma=W*x`, the post-solid candidate `v_gamma=W*v`, and transfer loads
   with `f_particle=W.T*f_gamma`. Check force, moment and power on the actual
   rounded particle load fields, including fixed-end reactions and damping.
   This interface-transfer identity does not prove energy conservation of the
   full fluid/solid time-discrete method.
2. Physical surface points are outside the convex hull of particle centres.
   Nonnegative weights, affine reproduction and the original boundary location
   cannot all hold. Do not truncate negative weights or move the wall onto
   particle centres. A bounded reference stencil must have justified conditioning
   and mass-weighted gain. If that cannot be established, use real material
   boundary degrees of freedom and account for their state explicitly.
3. Derive positions, topology-oriented normals, cap geometry and pressure probes
   from the same surface. Preserve the current area/cap policy until its traction
   convention is explicitly migrated; Cauchy traction times current area is not
   interchangeable with nominal traction times reference area.
4. Keep IQN trial-state semantics: a fluid trial uses its proposed wall velocity
   with base geometry. Generic restore must not overwrite the IQN guess with Wv.
   Material consistency is checked for post-solid candidates and accepted states;
   cap trial velocities use the derivative of the same cap mapping.
5. Rebuild static binding before checkpoint identity is formed; include particle
   reference identity, W, cap composition and mapping version in geometry/layout
   hashes. Reject an inconsistent accepted checkpoint, do not repair it silently.
   New source requires fresh preflow and cannot continue an r37 checkpoint.

The minimal material-mapping boundary is seven production files: a new
`material_surface_binding.py`, plus `hibm_mpm/core.py`, the official solid runner,
the case config, `hibm_mpm/interface_state.py`, `solids/neo_hookean_mpm.py`, and
`hibm_mpm/reports.py`. This excludes tests/docs and the separate MAC construction
work. Adding independent dynamic boundary nodes would also change macro-state
and accepted-checkpoint contracts. Do not implement only a position replacement
while retaining a different load scatter.

### MAC admissibility is a separate obligation

The current selector tests progress along the original source anchor-to-sample
line. It does not prove that the chosen staggered face is outside its final
global-nearest surface owner. The registered route then publishes hard wall
velocity with alpha zero and pressure mobility zero; it is not an existing
signed-ghost reconstruction. Obstacle-cell storage permission is not an
exemption from the physical side certificate.

Before implementation, add slanted-plane subcell translations in both storage
orientations, curved source-outside/owner-inside cases and true inward ghost
negatives. Observe obstacle masks, signed owner distance, wall interpolation,
discrete divergence and pressure mobility together. Do not select a farther
convenient owner, discard a raw route, or relax side/target tolerances.

### Accuracy and long-run acceptance order

The fixed native Fluent fresh50 bundle passed source-pair hashes, all 50 steps,
seven residual equation groups, 350 snapshots and strict outlet-zero static
gauge-pressure semantics. The coarse r37 result is not fine50 eligible: its
final solid displacement is about 6.83 mm versus Fluent's 0.217 mm. The material
particles themselves show the large displacement, so geometry-output repair
alone cannot close the accuracy question.

Run independent solid loading and time/grid refinement, mapping/adjoint and
rollback tests, then new source-matched preflow and same-condition fine50
comparison. Hard numerical gates remain unchanged. The quasi-2D/span leakage
gate remains at <=5%; for the main 2D-versus-3D waveform quantities, NRMSE
<=10% is the target. A 5% waveform result is only a strict diagnostic, never
the default parity gate. Only then extend actual physical runs through 200 and
5000 steps with checkpoint recovery and bounded work/memory monitoring.
Synthetic 5000-record persistence or repeated geometry calls are not physical
5000-step simulation evidence.

## 2026-08-28 authorized material-reference implementation and current boundary

Backup preceded every refactor edit. The immutable archive and Git bundle are
at `/home/zhuohengli/backups/HIBM-MPM-r21-validation/pre_material_refactor_20260828T112045Z`.
An actual extraction and bare-repository restore verified 3,407 regular files,
3,718 paths, SHA-256 contents, metadata and Git HEAD. Dirty, untracked and ignored
worktree evidence was included; the root `.git` pointer was saved separately and
the bundle preserves repository refs. This is a verified same-machine backup,
not an off-machine disaster-recovery claim.

The current implementation is a method migration, not a failure-step exception:

- `material_surface_binding.py` builds an immutable Cartesian reference W with
  partition of unity and affine coordinate reproduction. Signed boundary
  extrapolation remains limited to half a reference cell plus input/arithmetic
  roundoff; no clipping or radius-search fallback is used. Physical wall f32
  quantization and particle-coordinate quantization are both accounted for.
- `material_surface_transfer.py` reconstructs x and accepted v from the same W,
  derives oriented edge normals/probes, and uses deterministic particle-CSR W.T
  for loads. It stages the actual rounded f32 particle load and checks force,
  torque and material virtual power before committing those exact bits. These
  checks do not establish energy conservation of the complete time-discrete FSI.
- The cap velocity is the derivative of its existing 1.5-tip-minus-0.5-previous
  position map; cap load folding uses the same adjoint. The existing fixed
  reference-area convention is preserved, not promoted to a current-area model.
- IQN trial wall velocity remains an independent algebraic guess at base geometry.
  Post-solid candidate and accepted state use Wv; generic restoration does not
  replace an IQN guess. A binding/layout hash and pre-write accepted-state
  geometry checks protect complete checkpoint capture and restore. Invalid cap
  geometry cannot partially write a restored macro state.
- ANSYS physical markers now default to zero face offset, with the pressure probe
  offset represented separately (`physical_face_offset`, 0.51). Invalid material
  configuration and incompatible interface writeback fail before fluid allocation.
- `component_face_candidate_geometry.py` checks the same global-nearest owner
  before MAC progress ranking. It does not choose a farther convenient owner,
  discard raw sources, or grant final authority. Full raw capture, final source
  audit and the sole canonical-ledger commit remain required.
- Solid Cartesian initialization evaluates coordinates in f64 before one f32
  store. Support/damping diagnostics distinguish final-substep fixed force from
  batch impulse and angular impulse. For `pure_fixed_mass`, the discarded fixed
  PIC/APIC share is audited at unclamped grid nodes, including affine angular
  momentum; already-clamped nodes are not counted twice. This does not establish
  global momentum closure of the separate free-particle PIC/FLIP difference.

The independent `current_iqn_adaptive_material_reference` Fluent profile retains
the current IQN physical-time/raw-trial and strict-pressure/fine50 requirements.
It adds immutable material identity, 50 measured adjoint audits, five finite
three-component support/damping vectors and exact final-summary/last-history
consistency. The real CLI forwards these fields and exports the established
canonical `scatter_action_reaction_residual_N` JSON key. Historical profiles and
the 5% diagnostic gate are unchanged. A passed fixture is not Fluent parity.

Completed focused evidence (do not add overlapping counts):

- Material binding: 68 host tests; the translated f32-wall RED had 6 failures and
  12 strict-outside controls, followed by GREEN with both input errors budgeted.
- Integrated material/cap/binding/runner/checkpoint CPU gate: 236 tests passed.
- Explicit strict-CUDA material/checkpoint gate: 88 tests passed, with 87 actual
  CUDA runtime observations and one host-only metadata test; eight scoped
  source/test SHA values match before/after. CPU fallback was disabled.
- New material Fluent profile including actual CLI atomic summary/history
  serialization: 63 tests passed. The three existing native comparison files:
  130 tests passed. All data in these comparator tests is synthetic.
- Existing MAC assembly/owner/raw-capture/path-audit gate: 71 tests passed under
  strict CUDA (including host-only tests); 14 scoped source/test SHA values match.
- Final support/runner CPU gate: 13 tests passed. Selected solid dynamics,
  cantilever clamp, complete restore and actual material case initialization:
  16 strict-CUDA tests passed in 674.35 s; 10 scoped source/test SHA values match.
  The legacy cantilever's 20,000 solid substeps are not coupled FSI evidence.
- Pure-host executable-line coverage: binding 227/239 (94.979%), new material
  profile 138/138 (100%). This is not branch, GPU-kernel or whole-repository coverage.

Machine evidence and exact commands are under
`C:/Users/lizhu/.codex/visualizations/2026/08/27/01a0434a-786c-7bc2-a8c6-1a381dda007a`;
see the worklog for file names. Repository-wide 80% coverage is not established
by these test counts. Solid/MAC regression is complete, and independent review
found no remaining blocker within the scoped refactor. The verdict authorizes
numerical validation only. The historical r38 67-file source set was frozen as
`a5e73ba968a04a207bd0d330ea5e6e26b263162ad07da152cd8b5adb9a70d78a`.
Validation proceeds through fresh strict preflow, physical short
and recovery gates, then exact fine50/Fluent and only afterwards 200/5000 steps.
No target, pressure or coupling tolerance has been relaxed; both components still
owe the full accepted macro dt. The current source is not production-certified.

The r38 short-run gates have now completed on that frozen source: continuous
4/4 steps and a separate K2-to-K4 resume, each ending at 0.002 s. The live K2
post-restore audit found no mismatch within its declared saved-state scope.
The two K4 states differ in 41 of 124 arrays; their prefixes already differed
before restart. Full state/journal differences are retained rather than waived.
For example, maximum absolute differences are 3.7253e-9 m in solid position,
4.1869e-5 m/s in fluid velocity and 9.2428e-4 Pa in pressure. These are not
Fluent errors, a restart-only causal comparison, or a long-duration guarantee.
The final complete comparator has 27 focused tests and a real K4 self-comparison
with zero differences. See the worklog for exact artifacts and timing.

The user's 2026-08-29 ordering is explicit: finish this refactor, attempt-state
lifecycle, regression/review and documentation first; only then perform full
50-step, Fluent and progressively longer numerical validation. Short-run
completion must not be described as long-duration robustness.

### F64 deformation-state precision contract and renewed validation boundary

The solid precision repair is deliberately narrow: persistent Neo MPM `F` and
`saved_F` are f64, while `C`, particle/grid velocity and the existing P2G/APIC
storage remain f32. The deformation recurrence, identity, constitutive intermediates,
stress map and P use explicit f64 arithmetic. A legal raw `F` is consumed without
an SVD round trip; SVD projection still protects only genuinely out-of-bound singular
values or a reversed determinant. This is not a change to pressure, target, IQN,
or deformation-limit tolerances.

Accepted-checkpoint restore now rejects a non-f64 solid `F` before any owner write;
it does not silently promote legacy f32 deformation. Save/restore and macro rollback
therefore preserve legal f64 low bits, while the declared f32 fields remain strict.
Every pre-repair source identity, including r39/K25, is historical evidence only and
cannot be resumed into the new source as an accepted state.

The repair has focused CPU/CUDA, checkpoint and isolated solid A/B evidence recorded
in the worklog. It has **not** yet produced a new complete source freeze, fresh
preflow, 50 accepted FSI steps, Fluent comparison, or 200/5000-step result. The r39
25/50 failure remains a historical failed campaign and is not replaced by those
focused tests. The required order remains: freeze the repaired full source, create
a fresh source-matched preflow, run the exact fine50 gate, then the locked Fluent
diagnostic comparison before any long-run claim.

### Final run-attempt lifecycle and refactor phase boundary

The v2 lifecycle separates accepted state from process attempts. Before an
attempt begins, the CLI validates current production source identity, the
canonical checkpoint head/generation and the requested total target. It then
creates attempt-local v2 metadata/manifest that binds the canonical path,
checkpoint generation and identity, accepted step, source hashes, target and
attempt identifier. The canonical directory is never used as an output directory
for a resumed attempt and is never archived or moved by the lifecycle helper.

Canonical storage is the completed initial-run root: it retains the initial
metadata/manifest, progress, summary and run-specific exports as well as the
checkpoint generations/journal and accepted field/history evidence. Each later
resume attempt contains only its own metadata, manifest, progress, summary,
failure/interruption records and run-specific exports. A terminal resume failure
remains local to that attempt; it cannot alter, conceal or invalidate a
previously accepted canonical prefix or the initial-run terminal files.
Completion/oracle consumption rejects active terminal records and unsafe paths,
including Linux symlinks presented as Windows/WSL reparse points.

This is a single-writer, ordinary-process-error preservation contract, not a
power-loss durability guarantee. Invalid source, generation, output prefix or
terminal entry is not a reason to resume through a weakened numerical check.
The preflight checkpoint-head generation is passed into checkpoint loading and
checked before state decode; the same generation is then carried as an operational
runner pin and recorded in attempt config/manifest/provenance. If another process
advances the head between preflight and runner restore, the invocation fails closed
instead of consuming the newer generation. The pin is deliberately excluded from
the physical/configuration identity.

Accepted `step_fields` and `step_history` directories and files are inspected with
`lstat`, must be real directory/regular-file entries, must resolve beneath the
canonical root, and must use the unique canonical `step_{K:04d}` spelling. This
guard applies both to resume-prefix validation and the terminal exact-sequence
gate. Observer replay publishes complete fsynced candidates create-only: Windows
uses rename-without-overwrite, while POSIX uses a hard-link claim followed by
unconditional staged-temp cleanup. Existing content is preserved unless exact
NPZ keys/dtypes/shapes/values (with NaNs in the same positions) or the exact JSON
object matches; one missing member of a frame/history pair can be repaired.

This is not a multi-writer checkpoint CAS. The checkpoint store still reads the
expected generation and later replaces the manifest under the documented
single-writer rule; two concurrent store writers are not serialized. The bounded
path guard also does not claim defense against a hostile post-`lstat` path swap.
The supported deployment boundary is a trusted local filesystem with one writer.

The same CLI now writes a separate `fsi_coupling_diagnostics` field only for the
concrete `FsiCouplingConvergenceError`: complete step context, dataclass report,
and the raw 16-trial vector histories are retained without changing the legacy
`pressure_solve_diagnostics` meaning. The current accepted progress index/time is
not advanced by failure export; non-FSI failures cannot inherit stale FSI data, and
a subsequent `running` event removes it. Reporting failure is subordinate to the
primary exception and never turns a diagnostic write error into numerical success.

The renewed RED gate failed exactly four cases: two K1 noncanonical/duplicate
aliases, simulated Windows reparse containment, and missing pre-decode generation
pin propagation (`4 failed, 7 passed, 2 skipped`). After the implementation, that
same hardening file passed **11 passed, 2 skipped**. The two skips are Windows
symlink-creation permission limits; WSL-native directory and artifact-file symlinks
were then created under `/tmp`, and the production preflight loaded through trusted
Windows Python rejected both without following them.

Current focused host evidence is: attempt diagnostics **21 passed, 2 skipped**;
the combined metadata/checkpoint/CLI/main gate **87 passed, 2 skipped**; and the
affected checkpoint-runner/output-contract gate **110 passed**. Syntax compilation
and focused Ruff checks passed; the runner Ruff invocation ignored only two F841
locals verified present in the pre-refactor backup. A broader selected run ended
at **176 passed, 1 xfailed, 10 failed**; every failure was in the pre-existing
`test_ansys_vertical_flap_fsi.py` case-contract debt, so it is not reported as a
green broad suite. Backup comparison shows this refactor added only the 19-line
runner generation-pin path and one operational config field in those files.

These are lifecycle contract tests, not an actual K50-to-K50 or K50-to-K52
execution, fresh formal50, Fluent comparison, power-loss durability or 5000-step
physical soak. Numerical equations and tolerances were not relaxed.
Independent final read-only review returned SHIP with no blocker after its own
`125 passed, 4 skipped`, runner/output `79 passed`, incremental `15 passed, 2
skipped`, and Ruff gates. It explicitly retained the single-writer/no-hostile-swap
boundary and made no CUDA, formal50, long-soak or Fluent claim.

For the later FSI failure-export addition, the real pre-change CLI test-only RED
was 4 failed / 2 passed in 5.50 s (`fsi_failure_export_actual_red.xml`). After the
candidate CLI was applied, the real focused host integration set was 105 passed in
27.79 s (`fsi_failure_export_actual_green.xml`), including six new failure-artifact tests plus the
run-attempt, checkpoint-CLI, output-contract and runner suites. Its CLI SHA-256 is
`f4adbfe692187c2e64876c51c2a3c54ad9cc27a95aca3f326da1f015d605e0b7`.
This is host integration evidence for diagnostic persistence, not a new accepted
FSI step, preflow, fine50 or Fluent result.

The new CLI diff has 33/33 executable added lines exercised by nine related host
tests (7.52 s), measured with stdlib line events intersected with executable line
numbers. This is added-line coverage only, not branch or whole-repository coverage.
The real integration and its source diff received independent read-only review.

The historical r39 68-file source freeze was
`fdb5919fbb897c061ef507a03a923d9a6dd5ddd3a63e5b7f3975b18c9425e4dc`;
its production checkpoint source identity was
`cda30f7b85eec55276e5942830491fa43df698e93c8d0a5187fe95dad89f5b1d`.
That campaign accepted 25/50 steps and failed at K26. These identities predate the
F64 repair and failure-export addition: neither its preflow nor its K25 checkpoint
may be reused as a valid new-source restart. Earlier r38 evidence is historical too.

### Physical-face flux/state refactor and remaining regression gate

A read-only audit of the old r39 preflow identified an external-face contract gap:
projection treats unregistered x faces as closed, but the MUSCL/SST primal-Q builder
uses the last internal compact MAC face as the maximum-side boundary value. In
that snapshot the spurious xmax absolute-Q ledger is 3.44339% of inlet volume flow.
This is not a measured net momentum loss or an explanation of the 45% pressure-force
gap: the advective update contains a continuity correction that cancels the uniform
external state contribution in exact arithmetic.

Before a new source freeze, require symmetric RED/GREEN tests and one explicit
external-normal rule shared by projection and transport. Transport must receive
the same pressure-outlet and velocity-inlet topology as projection on each call;
do not infer it from SST defaults or add unsaved active-mode state. Preserve exact
external targets, the pressure-correctable zmin outlet, existing explicit zmax
topology, and the separate immersed wall-relative-Q semantics. Standalone throughflow
tests must declare their physical boundaries without weakening their accuracy or
conservation assertions. The shared-rule implementation is now applied, including
the generic sharp HIBM entry point, after independent read-only review. Existing
throughflow fixtures declare inlet/outlet explicitly; their numerical assertions
are unchanged. The four new contract files passed 20 tests plus 8 subtests in
168.66 s on the actual new source. Related host integration passed 34 tests plus
4 subtests in 7.99 s; one historical Fluent-import test skipped because its old
reference directory is absent, not because canonical fresh50 data is absent.
Fresh read-only review verified these artifacts and all active official/generic,
SSP-source and retry routes. Legacy numerical regressions are still in progress;
none of this is a new coupled50 or long-run result.

The first legacy batch finished with 25 passes, 9 passing subtests and one
generic sharp-HIBM failure: a zero-increment band sweep invalidated the ledger,
then the caller exited before resealing. Band/air zero exits and positive
overflow/tiny cleanup now reseal before readers and publish current-generation
reports. The original real CUDA node passes (359.21 s); new host control-flow
tests pass 4 parent tests and 22 subtests. The sealed guard remains unchanged.

A separate numerical RED showed that correcting Q alone was insufficient:
stale compact normals still affected momentum states, SST reconstruction and
normal matrix classification. The shared rule now returns prescribed/value,
distinguishing a free extrapolated zero from a fixed exact/default zero. It is
routed through minimum synchronization, maximum ghost states, all SST stage
reconstruction/strain/transpose paths, and the unsplit normal boundary matrix.
Tangential slip/correlation behavior and the last internal maximum-side MAC face
are preserved. Independent static review found no production blocker.

The applied implementation passed 7 new CUDA tests plus 51 subtests in 364.72 s.
That gate covers state, cached gradients and nonzero-viscosity row/face helpers;
the initial zero-dt/zero-viscosity stage checks alone do not certify the final
transpose divergence or host-to-matrix routing. The stronger direct stage gate
subsequently passed while two old SST manufactured fixtures failed: 2 failed,
1 passed plus 6 passing subtests in 353.39 s, XML SHA
`84fbe9ea5f9e4f53edabac6b4aa566f106eee1b3d472d610455fd90e79296659`.
The two failures were invalid legacy physical-boundary declarations, not failures
of the strengthened real transpose/row/diagonal checks.

The broad legacy batch finished with 46 passes, 31 passing subtests and five
failures in 1983.16 s, XML SHA
`cabc4db95b6ede0b9091eaa629bd0d3ba821599968b285990021d2f97675470f`.
Four fixtures implicitly prescribed nonzero x-normal flow while leaving those
physical faces undeclared/closed; one coefficient mock retained the old signature.
The two SST fixture declarations and the moving-obstacle mock were migrated in
test SHA `d12a6b42c3dfa06e20b5b49d56798e65a89ccfc48bad1bd37e55d79ae8b7b8b7`;
the correlation-transpose probe received the mechanical signature migration in
test SHA `f8ac1ab362304275121fdb8161da24acbfcb8a44e2613f02a3d204e4aff916a6`.
The repaired SST, viscosity, transpose and real generic-HIBM batch completed with
32 passes, 26 passing subtests and 30 deselections in 3334.81 s. Its JUnit XML
contains 58 tests with zero failures/errors/skips and has SHA
`4f013e41fa550d1c98dbe59739e1f721af19ceeb18b75cd52661ed75fc9f8620`.

The first MUSCL migration rerun stopped at 1 pass plus 2 passing subtests and
1 failure in 210.75 s, XML SHA
`b6dfed5d6ef36e9fd9f9defc5764a7a420ad0f5a831aa0801f391c1568aa4a3c`.
Its sole failure was a test-harness assumption that one macro step must contain
exactly one adaptive slice. At the z-min half control volume the nominal CFL is
at least 0.5, above the initial target 0.45, so the production solver correctly
used two slices. The replacement observer supports the actual accepted slice
sequence, rejects any retry, preserves the public predictor and full physical-
time accounting, and retains the original accuracy and momentum tolerances.
Candidate SHA `f5864edf...` passed independent static review and was applied
exactly to the authoritative WSL test file; its hash, Python syntax and diff
whitespace check passed. The complete MUSCL CUDA rerun then passed 13 tests and
6 subtests in 1235.08 s. Its JUnit contains 19 tests with zero
failures/errors/skips and has SHA
`0e44f629303888ec56399d173463cd407bf90d4c56c303345e2a71f5e68d9e57`.
No solver tolerance or physics threshold was relaxed.

A fresh final read-only review inspected the complete 582-line fluid-solver diff
and 17-line sharp-core diff plus the changed call surfaces and tests. It found no
production blocker and issued `ship` only for creating the r40 source freeze:
physical-face state, current/previous routing, implicit/transpose pairing,
retry rollback and ledger resealing were mutually consistent. This review is
not a fresh preflow, accepted FSI step, fine50, Fluent or long-duration result.

Old-source expanded RED reported 13 failures, 5 passes and 6 passing subtests
(163.21 s): five numerical face/divergence failures, seven missing-API failure
items and one missing runner-routing item. A separate real generic sharp-entry
host probe failed for all four outlet/inlet boolean combinations (4.94 s), then
the two missing predictor arguments were added. The old-source throughflow
baseline retained all three selected accuracy/amplitude/inlet contracts
(429.24 s). None of these measurements is candidate GREEN.

The staging `material_precision_validation_campaign.py` prepares exclusive r40
dryrun/preflow/formal50 stages without automatic chaining or resume. It locks the
same 114-file inventory as the production manifest plus its launcher, constant
source, inventory function, admission gates and diagnostic wrapper dependencies.
The existing admission gates accept an explicit campaign policy without changing
their numerical conditions. Root's independent host rerun passed 45 tests in
20.21 s; this tests the harness, not any new fluid or FSI simulation. No real r40
source lock, fresh preflow, formal50 output, Fluent comparison or long-duration
run has been created at this point.

An independent raw K0 audit reproduced historical r39 outlet streamwise momentum
of 4.216296317359675 N/m against canonical Fluent's 6.641229869957881 N/m
(ratio 0.6348667942412924). This uses actual boundary mass/velocity data, not a
force-unit rescaling. The outlet profiles differ before FSI starts; it does not
identify the numerical cause. The known Fluent pressure/shear minus advected
momentum remainder is incomplete and must not be labeled a discrete conservation
error. Its diagnostic script, real-input result hashes and 18 host helper tests
are recorded in the validation worklog.

The next campaign must use a new output directory and newly frozen full sources:
fresh preflow, exact fine50, then the locked Fluent diagnostic comparison. Only
after those gates may 200/5000-step numerical validation begin.

## 2026-08-30 r46 fine50, continuation, and compiler-identity boundary

The pre-refactor backup is
`/home/zhuohengli/backups/HIBM-MPM-r21-validation/pre_canonical_resume_refactor_20260829T235515KST`.
The r46 numerical campaign ran on branch `codex/closure-diagnostic-r23` at
commit `f61758e0ef09045a0b995067df0d263a118bab61`, before the
compiler-identity patch described below.

The fresh fine-grid run accepted exactly 50 steps and reached physical time
`0.025 s`. The strict post-audit found one field and one history artifact
for every step. Maximum exact CG relative residual was
`8.51341690505797e-07`, maximum main marker closure was
`9.899138149194187e-07`, maximum observer closure was
`9.885826557365363e-07`, maximum no-slip residual was
`1.6376294297515415e-05`, and the maximum IQN iteration count was 3. At
the historical r28 failure location, step 42 accepted 428 compatible duplicate
claims with zero alpha, claim, region or target conflict and zero segment-route
fallback. Every accepted fluid and solid step consumed the full macro
`dt_s=0.0005`; no numerical tolerance was changed.

The locked exact50 Fluent diagnostic passed its input, pressure-semantics,
residual and step-history contracts. The all-metric 5% diagnostic gate did not
pass, so parity is not claimed. Primary normalized differences were:

- tip-displacement waveform: `20.9220%`;
- maximum solid displacement: `23.2269%`;
- streamwise force: `37.5234%`;
- transverse force: `57.2514%`.

Measured out-of-plane force leakage was `0.0`, below the 5% quasi-2D
leakage limit. Consequently, the force discrepancy cannot be waived solely
because one solver is 3D/extruded and the Fluent reference is 2D. A 10%
difference is retained as a high-consistency target only after geometry,
boundary conditions, temporal window, observable definitions and mesh/time
convergence are aligned. It is not a universal truth gate. Before that
alignment, displacement may be triaged in a 20--25% engineering diagnostic
band; force requires separate amplitude, phase, sign and absolute-error
analysis. These comparison bands do not relax pressure, closure, conservation,
physical-time or no-slip solver tolerances.

The first K50 restore attempt failed closed because the checkpoint recorded
`cfg_optimization=false` while the new process obtained
`cfg_optimization=true`; source and geometry identities matched. The
preflight had only checked checkpoint self-consistency, while the real runner
correctly recomputed runtime identity. Repeating under the original compiler
identity proved both continuation invariants:

- K50-to-K50 performed no physical advance and left the complete canonical tree
  SHA-256 unchanged at
  `ce593911f2e33cb54fc982175423e2abd46672e8f999ae03bb2a97ca7d0b9e1a`.
- K50-to-K52 advanced the checkpoint from generation
  `6b650ca5eab74f1883f1673393595b7f` to
  `256b05cb7934498285ac24f2a214c2f2`. The combined digest of the first
  50 field and history pairs remained exactly
  `b3e354c63e6220854aed9db5214942a34be26e049c537d37ff0b33d13d3a3a28`.
  The durable store contains 118 arrays and 52 contiguous journals; accepted
  time is `0.026 s`.

For steps 51/52, exact CG residuals were `5.281951695825032e-07` and
`5.28770078455093e-07`; main closures were
`9.81815446721157e-07` and `9.785641168491566e-07`;
no-slip residuals were `6.130675956228515e-06` and
`6.645414487138623e-06`. Both steps used three IQN iterations, zero
IQN fallback and full fluid/solid macro time. The restored first new step took
about 772 seconds because the separate process rebuilt the complete CUDA JIT
path; the following warm step took about 58 seconds. Long execution should
therefore continue in one process and use checkpoints for interruption recovery,
not restart a process for every small block.

The canonical root deliberately retains the initial run's terminal
`our_solver_summary.json`, `progress.json` and
`our_solver_history.csv`; after the resume they still describe K50.
Current accepted state must be read from the checkpoint head and contiguous step
artifacts, while the K52 attempt's aggregate files describe that attempt. A
consumer must not treat the canonical historical summary as a live accepted-state
pointer.

The production hardening now makes all six compiler fields that participate in
checkpoint identity optional runtime pins: `default_ip`,
`cfg_optimization`, `opt_level`,
`advanced_optimization`, `fast_math` and `debug`.
Legacy callers retain `None` defaults. The official runner explicitly
requests `i32/false/1/true/true/false`, passes only explicit pins to
`ti.init`, and verifies actual `ti.cfg` values both after
first initialization and on the already-initialized fast path. Any mismatch fails
before wrapper state is published or continuation is attempted. Checkpoint
identity errors now name each differing config, source or geometry digest without
accepting the mismatch.

Test-first evidence for this hardening is 25 expected runtime RED failures plus
one official-runner profile RED. A later independent review found and repaired
one public dataclass positional-argument compatibility RED. After implementation,
the complete runtime file passed 78 tests; the profile contract passed 1 test; the three
checkpoint-identity diagnostics passed; and the bounded checkpoint/lifecycle
regression passed `175 passed, 6 skipped` in 56.34 seconds. Diff
whitespace validation passed. These are host contract tests, not a fresh
post-patch CUDA fine50. Because the runtime and runner source hashes changed, all
r46 snapshots/checkpoints must correctly fail source identity under the hardened
source. The next numerical gate is a new source-matched preflow and short
fresh/resume smoke before a longer 200-step soak; 5000 physical steps remain
unproven.

## 2026-08-30 r47 source-matched K200 and dual-root comparison boundary

The hardened source completed its source-matched preflow and short fresh/resume
gates before one continuous K50-to-K200 process was admitted. The resume attempt
finished with exit code 0, accepted `200/200` macro steps and reached physical
time `0.1 s`. The 150 newly accepted steps took `8933.812750 s`. Every accepted
step used three IQN iterations and two rejected same-time algebraic trials; those
trials did not advance physical time. The accepted-step journal remained
contiguous through K200.

The marker operator, pressure-nullspace operator and solver-scratch allocations
remained fixed at `18,309,056`, `41,902,024` and `23,592,968` bytes. The largest
exact pressure relative residual over the campaign was
`9.919589959059803e-7`: it remained below the unchanged `1e-6` gate, but the
margin is small and must not be described as a relaxed or comfortably separated
pressure result. This K200 run establishes the measured 200-step boundary and
restart path only. It is not a proof or guarantee that K5000 will complete.

The velocity visualization is a local ignored artifact under
`validation_runs/solver_soaks/ansys_vf__k200_velocity_viz__material_fine__20260830__r47`.
It uses a shared `0..45 m/s` scale for K50/K100/K150/K200 and renders the K200
field separately. SHA-256 values are:

- `velocity_magnitude_step_0200.png`:
  `a67836c10238dcc310c6247278dbe0f6b61a5f0f5cfe84db43ed60feb2b50582`;
- `velocity_magnitude_steps_0050_0100_0150_0200.png`:
  `a4758b320d27afd31c2d1558e145edfabace1066333d92b2d033bb6ac73f9c31`;
- `velocity_render_manifest.json`:
  `e97ebf6cb4c46f98611b52a05fbd3f5b9efd9a9bf5ddcb5a6f01a24dcff95de0`.

The locked exact50 diagnostic comparison now uses two roots. The completed K50
attempt supplies its aggregate history and terminal control-plane records, while
the canonical artifact root may already have a K200 checkpoint head. The
dual-root contract validates the attempt-v2 provenance, source/configuration/
geometry identity, the canonical journal and the exact K1--K50 field/history
prefix. A final source-matched audit passed all 50 prefix rows with 460 public
history fields per row, comparing the checkpoint journal, per-step JSON and
aggregate CSV. Only the two declared journal-only residual aliases and the
strict serialization rules for CSV booleans, empty values and the nine legal
empty-set NaN coordinates are accepted; a changed non-core field fails closed.
It does not substitute the stale canonical K50 summary for the current K200
checkpoint head and does not silently mix roots.

The comparison completed as a diagnostic, not as parity. The principal
normalized differences were speed `13.33%`, gauge pressure `19.07%`, tip
displacement waveform `20.92%`, maximum-solid-displacement waveform `23.23%`,
streamwise-force waveform `37.52%` and transverse-force waveform `57.25%`.
The transverse velocity metric was `4.27%`, and out-of-plane force leakage was
`0.0`; those two individual diagnostics passed the 5% check. The major metrics
did not satisfy either the existing 5% diagnostic gate or a separately
interpreted 10% high-consistency target.

Fluent is a locked comparison reference, not ground truth. Ten percent is only
a high-consistency target after geometry, boundary conditions, time window,
observable definitions and mesh/time convergence are aligned. The 3D/extruded
versus 2D model difference can explain some deviation, but it does not
automatically waive the larger displacement, pressure or force discrepancies.
None of these cross-solver bands changes the internal pressure, closure,
conservation, no-slip or accepted-physical-time gates.

After the full-field binding and create-only Windows publication retry were
connected to the production paths, the locked comparison was regenerated as
`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/material_reference_r47_fresh50_20260830_r3`.
It completed with `dual_root_artifact_contract.status=passed` against canonical
head K200 and retained the expected failed five-percent diagnostic gate. All
reported principal metric deltas from r2 were exactly zero, and WSL
`sha256sum -c` passed for all 59 packaged files. The final bounded regressions
were `169 passed, 3 skipped, 4 subtests passed` for dual-root/Fluent and
`530 passed, 4 skipped, 24 subtests passed` for checkpoint/runtime/resume/
lifecycle. A fresh read-only Sol/Ultra review found no P0/P1/P2 and returned
`ship`; it did not expand the K200, Fluent or checkpoint-to-step-NPZ evidence
boundaries stated above.
