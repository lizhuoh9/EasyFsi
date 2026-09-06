# Turek–Hron FSI Validation Report

## Latest result: focused gates passed; R16 stopped for publication

The strict-CUDA/f32 GREEN25 r03 passed all 25 tests in 81.096244 s on clean
source commit `efdb2527900ff455e2150fdef701cd15298be8fe`. The complete R12,
R13, R14 and R15 frozen replays also passed. Each replay's 21 snapshots of
155 fields and 41 cleanup fields were independently read back and recomputed.
These checks advance zero physical time.

The from-zero eight-step R16 diagnostic was stopped with SIGINT during Taichi
compilation at 1% quota remaining to complete the user's authorized GitHub
publication before exhaustion. Owner PID/PGID 410 exited -2 after 686.565025 s,
with zero accepted steps and `KeyboardInterrupt`. Source193, all 77 dependencies,
and host identities remained unchanged; wrapper audit errors are empty.
This is an administrative interruption, not a measured numerical failure.
The initial native `process.json` still says running; the final owner record
and execution audit establish the actual exit.

Continuous eight-step success, independent full rollback validation, and the
registered component/formal benchmark gates remain unproved for this source.
No reset credit was used. The old 10% threshold remains withdrawn.
The following older checkpoints are historical; this block is the current state.

Full evidence and exact continuation requirements: [R15 fallback report](validation/TUREK_HRON_R15_FALLBACK_VALIDATION_REPORT_2026-09-06.md).

Publication: [source and complete evidence](https://github.com/lizhuoh9/EasyFsi/releases/tag/r15-wip-20260906-1330).

## Historical explicit continuation checkpoint, 2026-09-06

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

**Earlier A50 repair state (historical evidence):**

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

**Prior common-cohort repair at core `d8a14f64` (historical evidence):**

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

**Prior band-ordering evidence:** the reviewed production band-ordering correction
is applied at core `c989f6ba` / fluid `a1a3324b`. All 45 distinct host checks pass.
The strict-CUDA production-helper frozen-pose check r03 completed in 58.056119 s
with `PASS_PRODUCTION_ORDERING_FROZEN_POSE`; it advanced zero physical time.
Coupled and benchmark acceptance remain pending.

The preceding r02 reconstructed ordering control passed on unchanged old source
`cda0eca0` / `fe841f4c`, preserving its strict shared-field RED comparison and old
hard-system lower bound `2.184912935710554e-6 m/s`. Both frozen-pose checks added
92 excluded cells and then zero, reaching 2536 from 2444. The final 336 rows
include 290 hard-only rows and 46 with free support; the former witness rows
254/257 gain legal free support, so this does not solve the old H-only system.
Native final H/global residuals are `8.992282687358966e-7` /
`4.3388867197791114e-5 m/s`, below the unchanged `1e-6` / `1e-4` gates.
All 112 raw positions and targets remain unchanged. The checks reconstruct a
private owner from an incomplete capture; intermediate marker health is not
qualified and no physical step, component gate or formal benchmark passed.

The from-zero diagnostic r11 accepted four steps to `t=0.020 s`, then
rejected step 5 with two geometry `prepare_pair_arbitration` conflicts. Its first
reported face is `(1,50,304)`, axis 0. Revalidation with the original candidate
and native-record validators passes for all four accepted records: all 356 CSV
values match exactly, and all 4 x 34 native fields match in dtype, shape and byte
hash. Fluid and solid each consume `.005 s` per step and `.020 s` in total, with
zero remaining time; each accepted step has one trial and no rejection. All 193
recorded solver source hashes are unchanged before and after the run.

Runtime reports restored fluid, solid, marker interface and marker pressure
gradient state, while derived search/boundary state requires rebuilding.
Capture errors are empty; a full post-rollback state comparison remains pending.
The earlier r09 result remains one accepted step to `t=0.005 s` before its
certified step-2 failure; its candidate/native-record and fluid/solid time audits
passed.
Earlier geometry/provenance and zero-free certificate corrections passed 16 and
19 focused strict-CUDA tests respectively. [The trace-space audit](validation/TUREK_HRON_TRACE_SPACE_AUDIT_2026-09-05.md)
records the exact source, control and implementation boundaries. Source-matched
component requalification and formal FSI1/2/3 acceptance remain pending.

Base results as of 2026-07-07; R26A status updated through 2026-09-05. Solver:
HIBM-MPM (sharp immersed boundary + Material Point Method), Python + Taichi,
CUDA. For R26A artifacts at and after `293dc69`, the active interpreter must
measure as CPython `3.10.12`, NumPy `2.1.2`, and SciPy `1.15.3`; any
mismatch is `BLOCKED_ENVIRONMENT`. Case: `cases/turek_hron_fsi.py`.

<!-- BEGIN R26A ROBUSTNESS 2026-09-05 -->
## R26A robustness preparation, 2026-09-05

All ten component stages passed at frozen HEAD5dcba96. Fresh formal S0 then
failed after seven accepted steps: the original hard-target trace cannot meet
the strict1e-6 gate, as independently certified. Fixed112 refinement remains
under investigation. See [the trace-space audit](validation/TUREK_HRON_TRACE_SPACE_AUDIT_2026-09-05.md) for complete
current evidence. Subsequent component-identity changes require requalification;
no formal FSI1/2/3 benchmark pass exists. The focused checks below describe the
reviewed5dcba96 implementation scope.

- The case validates finite physical inputs and the declared material CFL
  before allocating simulation state. L1 FSI1 at dt=0.005 uses 200 solid
  substeps; L1 at dt=0.0025 uses 100, and L2 at dt=0.0025 uses 200. Both
  accepted fluid time and accepted solid time still equal the full macro dt.
- A candidate step is checked against its stage's numerical health contract
  before acceptance/history publication. Failure rolls back physical state and
  records the candidate and rollback scope; derived search/boundary caches are
  explicitly marked for rebuild. No rejected prefix is a restart checkpoint.
- The four former 512-row barriers (rank-direct Q, pressure P, collective F,
  and certificate-authorized F/H) now select a bounded sparse backend for
  larger capacities. Small-system paths keep their original algorithms.
  Structural rank, all-row f32 residual audits, H authorization, and resource
  limits remain enforced. The final Q audit includes the actual rounded
  velocity addition and the same hard/external masks as commit.
- Large P prepares a QR basis once and reuses device ndarray buffers. Its
  device triangular solve uses the full Jx, including input on zero-mobility
  faces; it preserves the fixed linear projector and leaves those faces
  unchanged. Per-apply convergence loops or host scalar reads are not added.
- The formal CLI now selects all 13 frozen stages with --stage and explicit
  --prerequisite campaign manifests. It recomputes prerequisite assessments
  from hashed CSV/chunks and checks full config, source, host and runtime
  identity before initialization. Progress is printed for accepted step 1 and
  each 25th accepted step; a silent first trial can still be computing.
- FSI2/3 have their own relative-convergence policy. FSI3 M0/tight retain zero
  absolute tolerance. Dynamic final quality rechecks every M0/M1/F0 run against
  the strict benchmark limits, even though M0 may enter refinement at the
  exploratory limits. Periodic telemetry uses accepted physical fields and the
  existing last-three-complete-cycle analyzer; no adaptive-relaxation history
  is invented where only initial Picard relaxation and the IQN limiter exist.

Focused validation completed before source freeze:

| Gate | Result |
| --- | --- |
| New nonzero Q / weighted P strict-CUDA regressions | 6 tests + 6 subtests passed (400.00 s) |
| Existing pressure-projector contracts, including resource/lifecycle/zero-mobility behavior | 29 passed (1171.03 s) |
| Isolated CPU direct-Q regressions | 7 passed (358.93 s) |
| Large F/FH CUDA closure plus sparse-structure CPU equivalence/budget | 3 tests + 2 subtests passed (6.92 s) |
| Formal runner, FSI1/periodic acceptance, stage contracts | 138 CPU tests passed (25.60 s) |
| All 21 changed/new Python files | compilation, full Ruff and diff checks passed |

The physical/CFL guard and candidate-rollback tests also passed during their
implementation (9 and 3 tests respectively). Independent Astra/max review
returned ACCEPTED with no remaining P0-P3 findings in Q/P/f32, F/FH, stage
contracts, formal runner and acceptance. This accepts the reviewed code and
focused tests only. Component recertification subsequently passed for that
frozen source, while formal S0 exposed the unresolved trace-space defect. That
trajectory result governs the current numerical status; a review verdict does
not establish a formal FSI1/2/3 pass.

<!-- END R26A ROBUSTNESS 2026-09-05 -->

Architecture update (2026-08-13): Turek-Hron FSI1/2/3 no longer owns a
case-local Picard/Aitken/IQN state machine or an explicit single-pass mode. All
presets now use the shared marker-velocity IQN-ILS loop in
`simulation_core/drivers/generic_fsi_solver.py`; the case retains only its
physics adapter and component-local fluid/solid substeps. Numerical results in
this report predate that migration and must be rerun before they are treated as
evidence for the unified core.

R26A pre-campaign correctness update (2026-09-03): the beam's lower face,
upper face, and physical free tip are registered as three disconnected open
segment chains for local HIBM projection. Open chains are never used as a
global inside/outside certificate. When `classify_far_internal_nodes=True`,
the far-interior authority is the live deformed MPM particle volume, installed
before the first search and refreshed after each solid macro step; the search
fails closed if that volume mask is absent. Before internal classifications are
converted to obstacles, a device-side invariant also requires every `INTERNAL`
node to belong to the live mask. The fixed-fluid time-zero audit requires zero
obstacle cells outside the analytic beam-cell-intersection or the canonical
cylinder. This update invalidates older component artifacts by source identity
and does not itself establish new FSI1/2/3 numerical evidence.

R26A fixed-fluid same-point-arbitration update (2026-09-03): the first
source-matched nx4 fixed-fluid attempt after that correction passed its
time-zero audit, then stopped during the first physical step with six canonical
component-face target conflicts. It produced no accepted physical row. The
conflicting authors were a direct fluid row and its relocation shadow at the
same physical free-tip endpoint: their serialized targets were exactly equal,
but their independently reconstructed effective targets differed only by
floating-point roundoff. The first repair covered exact one-hot endpoints.

After regenerating the source-matched solid-only chain, nx4 fixed-fluid attempt
`r03` again passed its time-zero audit and stopped during the first physical
step, before any accepted row, with two target conflicts. Its first witness was
not an endpoint: the direct row and its relocation shadow projected to the same
exact interior point of registered segment `(109,110)`, with the same nearest
marker, projection weights, boundary point, region, and serialized zero target.
Their effective targets differed by about `2.07e-19 m/s`.

The generalized repair still discards the shadow only after the existing full
redundant-shadow proof succeeds. It additionally requires the inactive axis,
an exactly equal serialized component target, the exact same nearest marker,
all three projection weights, and all three boundary-point coordinates, finite
effective targets, and an effective-target difference no greater than
`1e-6 m/s`. It preserves the direct target bits without averaging or rewriting
them. Three focused tests and twelve neighboring strict-CUDA contract tests
pass; Python compilation, diff checks, and Ruff also pass. The complete
component-face geometry module was not rerun; its preceding 1200 s attempt has
no module-level verdict. All component artifacts predating this source change
are source-stale and must be regenerated. This update authorizes another
component-gate attempt; it is not fixed-fluid or FSI1 numerical evidence.

**R26A source-matched component execution update (2026-09-03).** At clean
commit `f16737e`, the complete frozen solid-only chain passed as
`PASS_COMPONENT_ONLY` under strict CUDA. Runs
`turek_hron__component__solid_s100_nx4__20260903__r07`,
`turek_hron__component__solid_s200_nx4__20260903__r05`, and
`turek_hron__component__solid_s200_nx8__20260903__r05` each completed all 40
rows with zero root displacement and zero displacement/velocity spanwise
leakage. The source-matched S100/S200 Point-A relative-vector delta was
`0.0003264915974131584` (0.032649%), and the nx4/nx8 delta was `0.0`.

The next canonical nx4 fixed-fluid attempt,
`turek_hron__component__fixed_fluid_nx4__20260903__r04`, reached physical step
401 (`t=2.005 s`) and failed closed with `FAIL_BEAM_MARKER_NO_SLIP`: RMS
residual `0.0012601176039343787 m/s` and maximum residual
`0.005936640314757824 m/s`, against limits `1e-4` and `0.002 m/s`. Its 112
markers were all valid. Noncanonical in-memory traces first crossed the RMS
limit at step 33: overall RMS/max
`0.00010018624307816692/0.0004438578907866031 m/s`; the 108 direct samples
contributed `8.716031220311704e-05/0.0002774639579001814`, while the four
free-tip `normal_walk` samples contributed
`0.0002755487092652955/0.0004438578907866031`. At step 100 the corresponding
overall, direct, and `normal_walk` pairs were
`0.000522078997451709/0.002698277123272419`,
`0.0004219407204557312/0.0013645613798871636`, and
`0.0016807570266407306/0.002698277123272419 m/s`.

The defect was localized to projection wiring, not an alternate tip sampler:
time-zero closure directly constrained 308 of 336 marker-axis equations, while
28 q-free directions required marker-Q. The generic HIBM-MPM core did not pass
the existing marker-Q adapter into either the pre-projection or pressure
velocity-nullspace hook, and every pressure marker-nullspace diagnostic was
therefore false or zero. The terminal `normal_walk` and closure positions were
bitwise identical. The repair exposes the proven runner adapter publicly and
uses one persistent Q/P owner for Turek main, consistency, and post-solid
projections. Its default `None` path retains the prior ANSYS behavior; a review
found and blocked a transient residual/viscous obstacle mix-up before CUDA, and
a focused behavior regression now preserves their distinct legacy fields.

After `142 passed, 15 subtests passed`, compilation, Ruff, diff checks, and a
fresh read-only review, one **noncanonical, in-memory** strict-CUDA nx4 step
passed: requested/accepted time was exactly `0.005/0.005 s` with zero
unadvanced time; Q was prepared, converged, and committed; pressure-nullspace
projection covered all velocity paths with zero invalid actuation/correction
entries; 112/112 markers were valid; and terminal no-slip RMS/max was
`2.98977615920801e-07/9.697889709059382e-07 m/s`. This probe created no
component artifact and is not a canonical gate result. The failed canonical
`r04` directory and diagnostic `r01`/`r02` directories are empty. Because the
source has changed, all `f16737e` solid artifacts are source-stale and the
entire frozen component chain must be regenerated. No nx8 fixed-fluid, coupled
preflight, or FSI1 run is yet authorized.

**R26A component prerequisite completion (2026-09-03).** The stale boundary
above was superseded at clean commit `7b80a6b`. The marker-Q failure was a
feasible rank-deficient affine system whose f32 device PCG lost its required
self-adjoint/positive-semidefinite behavior. Turek now opts into a bounded f64
rank-revealing direct fallback; the public adapter still defaults to PCG, so
the ANSYS route is unchanged. Every projection cycle persists its actual
backend, rank, iteration/constraint counts, and residuals. Focused verification
passed `86` tests plus `120` subtests, compilation, Ruff, diff checks, and a
fresh read-only review with no remaining P0--P3 finding.

A non-artifact 20-step strict-CUDA reproduction crossed the old failure point.
At step 19 the main Q transaction split 336 equations into 16 independent, 12
dependent, and 308 unactuated rows and completed with a maximum structural
residual of `6.686404049105477e-06 m/s`, below the frozen `1e-4 m/s` limit.
The complete source-matched component chain then passed:

- three 40-row solid runs plus both comparisons; S100/S200 Point-A relative
  delta was `0.0003264915974131584`, and nx4/nx8 was `0.0`;
- fixed-fluid nx4 `r06` and nx8 `r01`, each with 500 accepted rows and two
  recorded Q cycles per row; their velocity/force span-leakage pairs were
  `0.0005504236666806834/0.0006353449740090614` and
  `0.0004303681250445346/0.00035212848977038484`;
- fixed-fluid nx4/nx8 force-per-span relative delta
  `0.0063723531512411 < 0.02`; and
- independent one-step and two-step coupled preflights. Each accepted macro
  step advanced both fluid and solid by exactly `0.005 s`, with zero remaining
  time; final accepted times were `0.005 s` and `0.010 s`.

This completes only the preregistered component prerequisite and authorizes
FSI1-S0 next. It is not FSI1 numerical evidence. The current dense fallback is
limited to 512 constraints: L0 has 336, while L1/L2 have 669/1002. A scalable
rank-deficient backend is therefore still required before M0/M1; no later case,
Oracle arm, or learned predictor is authorized.

**R26A formal S0 bridge-repair update (2026-09-03).** At clean commit
`0df48f4`, formal run
`turek_hron__fsi1_s0__0df48f4__r01` accepted exactly three macro steps, then
failed closed while preparing candidate step 4 with
`FAIL_NUMERICAL_HEALTH`. The accepted-only rows for steps 0--2 remain
preserved. The first canonical component-face witness had two direct author
segments, `(1,2)` and `(3,4)`, separated by the registered local connector
`(2,3)`; the same conflict appeared in four spanwise copies. This is a
failed S0 attempt, not FSI1 evidence, and it did not authorize any later stage.

The root cause was a missing arbitration case for that unique local connector.
Commit `7862472` admits a registered bridge only when it is the sole connector,
joins each author's nearest marker, makes both joins degree two, is unclamped,
and is strictly closer to the face than both authors. Disconnected and
ambiguous topologies, a long outer chord, endpoint clamping, and exact distance
ties remain fail-closed. Strict-CUDA verification passed the four integration
contracts for successful reconstruction plus disconnected, ambiguous, and
outer-chord rejection; the direct geometry contracts cover bridge win, clamping,
tie rejection, and author-order parity. Python compilation and
`git diff --check` also passed, and an independent final review found no
P0--P2 issue.

Every component artifact from before `7862472` is now source-stale. The entire
frozen component chain must be regenerated at the final clean source identity
before a fresh S0 run starts from zero. No FSI1-S0 pass, FSI2 authorization,
Oracle result, or learned-model evidence exists yet.

**R26A collective F-space repair and component requalification
(2026-09-04).** A later source-matched component attempt at clean commit
`ef1b8fe` passed the first five solid-only stages, then failed closed in nx4
fixed-fluid with no certified prospective hard-target repair. The zero-
correction collective residual in the captured state was
`3.944443960790522e-6 m/s`, already below the frozen public absolute tolerance
`1e-4 m/s`; the old cyclic Kaczmarz path nevertheless increased it to
`2.238149609e-4 m/s` after 21,504 sweeps. A compact f32-audited least-squares
witness reached about `3.92419578e-6 m/s`, and the active matrix had rank 10
with a clean singular-value gap. This was a solver-path defect, not evidence
for widening a tolerance or adding another geometry allowlist.

Clean commit `8fb44de` measures the identity correction first and, only when
needed, builds a bounded per-axis F-space least-squares sufficient witness.
The candidate is materialized as f32 and accepted only after the existing
device audit checks every active row. Any nonfinite value, inconsistent
repeated mobility, out-of-bounds support, failed solve, or residual above the
frozen tolerance remains fail-closed. The private witness does not reuse or
alter terminal-Q or pressure-nullspace transaction state. Fourteen focused
strict-CUDA contracts passed in one process (`39.749 s`), together with Python
compilation, structure validation, `git diff --check`, and a fresh independent
read-only review with no blocking finding.

The complete frozen component protocol then passed at clean commit `8fb44de`,
source SHA256 `4c88aa85bb2db42d75907ae794da7d9d4028b6c900d124c5191fc838fb878877`:

- solid S100-nx4, S200-nx4, and S200-nx8 each completed 40 rows; the
  S100/S200 Point-A relative-vector delta was
  `0.0003264915974131584`, and the nx4/nx8 delta was `0.0`;
- fixed-fluid nx4 and nx8 each completed 500 rows. Their velocity/force
  span-leakage pairs were
  `0.0005504236896309334/0.0006353428425121972` and
  `0.00043036809698871847/0.0003521289640626043`; the source-matched
  force-per-span relative-vector delta was `0.006372355169562249 < 0.02`;
- independent one-step and two-step coupled preflights both passed as
  `PASS_SMOKE_ONLY`. Every accepted row advanced fluid and solid by exactly
  `0.005 s` with zero unadvanced time; their final accepted times were
  `0.005 s` and `0.010 s`.

The artifact labels all end in `__8fb44de__r01` and live under
`validation_runs/turek_hron_component_gates/`. This completes only the
source-matched component prerequisite and authorizes a new formal FSI1-S0 run
from zero. It is not `PASS_FSI1_S0_GATE_ONLY`, does not establish FSI1
benchmark quality, and does not authorize FSI2, FSI3, Oracle, or learning.

**R26A certificate-connected weighted F/H repair and requalification
(2026-09-05).** The next formal run,
`turek_hron__fsi1_s0__0025e12__r01`, started from zero at clean commit
`0025e12`. It accepted steps 1--7 through `t=0.035 s`, then failed closed while
preparing candidate step 8 because marker compatibility closure did not
converge with three hard-target certificates. The accepted prefix is preserved,
but it is not an S0 pass and cannot be resumed as formal evidence because no
complete transition checkpoint was published.

The captured candidate-step algebra contained 336 active marker-axis rows. An
F-only solve remained just above the frozen absolute limit, while an F/H solve
was feasible. The defect was therefore not a reason to widen the `1e-4 m/s`
tolerance or add a geometry allowlist: the then-current hard-target solve scope
and the global acceptance audit were not aligned. Clean commit `5afba27`
authorizes H columns only within certificate-connected row components, includes
all active F rows in the joint solve, determines structural rank without
mobility scaling, and then computes the inverse-mass minimum-energy correction.
The f64 candidate is cast to f32 and must pass separate device audits for both
authorized repair rows and every active row. All nonfinite, rank, support, and
residual failures remain fail-closed.

The canonical device report schema is now version 6 and records whether the
collective repair was applied, its exact backend, certificate count, repair and
global residuals, hard-target DOF count, and maximum hard-target delta. The
strict runner cross-validates those fields. A captured-step diagnostic replay
closed with repair/global maxima
`2.27050833246e-7/8.83974644239e-5 m/s`, 70 hard-target DOFs, and maximum H
delta `1.05458639155e-4 m/s`. This replay is diagnostic only, not formal S0
evidence.

Verification included a RED/GREEN inverse-mass weighting and mobility-rank
contract, four focused F/H strict-CUDA tests (`93.276 s`), one real hybrid
integration (`177.698 s`), all 18 collective strict-CUDA tests (`862.117 s`),
65 host/static report tests (`2.275 s`), Python compilation, structure
validation, and `git diff --check`. A final independent read-only review gave a
`ship` verdict with no P0/P1 finding.

The complete frozen component protocol was then regenerated at clean commit
`5afba27`, with every artifact bound to source SHA256
`d1deddc51e16b3a862f25318d4bef75a773bd202d9d2e3e596615df5fe67a492`:

- the three 40-row solid runs passed; S100/S200 nx4 and S200 nx4/nx8 Point-A
  relative-vector deltas were `0.0003372753055382118` and
  `1.0950965526574071e-5`;
- fixed-fluid nx4 and nx8 each completed 500 rows. Their velocity/force
  span-leakage pairs were
  `0.0005504237140905238/0.0006353417123744162` and
  `0.0004303681097731811/0.00035212962421930457`; their drag means were
  `12.84566414514538` and `12.927746012667047 N/m`;
- the fixed-fluid nx4/nx8 force-per-span relative-vector delta was
  `0.006372354632268758 < 0.02`; and
- independent one- and two-step coupled preflights passed as
  `PASS_SMOKE_ONLY`, with final accepted times exactly `0.005 s` and
  `0.010 s`.

All ten labels end in `__5afba27__r01` under
`validation_runs/turek_hron_component_gates/`. This new source-matched chain
supersedes `8fb44de` and authorizes only a fresh formal FSI1-S0 run from zero.
It is not `PASS_FSI1_S0_GATE_ONLY` and does not authorize M0/M1, FSI2, FSI3,
Oracle, or learning.

**R26A minimax F-only repair, host pinning, and requalification
(2026-09-05).** A fresh formal run,
`turek_hron__fsi1_s0__27111d3__r01`, started from zero at clean commit
`27111d3`. It accepted steps 1--7 through `t=0.035 s`, then failed closed
while preparing candidate step 8 with `certificate_count=0`. The
zero-correction residual was `1.0104837565449998e-4 m/s`; the existing
least-squares F-only candidate
reduced its device-audited maximum residual to
`1.0004986688727513e-4 m/s`, still just above the frozen `1e-4 m/s`
limit. The accepted prefix is failure evidence only, not an S0 pass, and the
formal run cannot resume because no complete transition checkpoint was
published.

An independent minimax solve of the same captured 336-row system reached
`9.82453917360385e-5 m/s` in f64 and
`9.824539301916957e-5 m/s` under the production f32 device audit. The
`certificate_count=0` result was correct: this system needed no H
authorization. The false negative belonged only to the old F-feasibility
decision, which used the minimum-L2 witness as its sole sufficient witness for
an L-infinity acceptance criterion. It was not authorization to widen the
tolerance, admit a new topology, or expose H columns.

Clean commit `d3044d9` retains the bounded least-squares fast path and, only
when its f32 device audit fails, solves a column-normalized minimax LP. The LP
candidate remains provisional until the unchanged device audit accepts every
active row. Backend failure, nonfinite data, inconsistent constraints, or a
remaining excessive residual stays atomic and fail-closed; scratch is cleared
and no physical state is committed. Fourteen targeted collective strict-CUDA
contracts passed in one process (`758.463 s`), including signed minimax,
fast-path, backend-failure, true-inconsistency, F/H, rank, and scratch-
retirement cases. Python compilation, Ruff, `git diff --check`, 66 focused
host/static tests, and two independent read-only reviews also passed. The
earlier broad component-face module attempt was interrupted and has no
module-level verdict.

Clean commit `293dc69` additionally binds formal and component evidence to a
strict JSON-safe host numerics identity: CPython `3.10.12`, NumPy `2.1.2`,
and SciPy `1.15.3`. Component `run_manifest.json` files use schema 2,
future formal accepted-chunk manifests use `schema_version: 2`, and the
host-identity payload remains schema 1. `requirements.txt` participates in
source provenance, and a wrong host environment fails before solver or Taichi
initialization. The host identity
SHA256 for this campaign is
`db88ab4094ab1be43ad58b7c18cc59c756a45e1ac4f44978bc6238a4738c6a47`.

The complete ten-stage component protocol then passed at clean `293dc69`.
Every artifact is bound to source SHA256
`f186fa55278ef55a82f8ae2f740defad0766f3350450355bf6c1710fa279d7f3`
and the host identity above:

- the three solid runs each completed 40 rows. S100/S200 nx4 and S200
  nx4/nx8 Point-A relative-vector deltas were
  `0.0003264915974131584` and `0.0`;
- fixed-fluid nx4 and nx8 each completed 500 rows. Their velocity/force
  span-leakage pairs were
  `0.0005504236652169761/0.0006353445200368427` and
  `0.00043036809704860384/0.0003521293000030293`; their drag means were
  `12.84566396988839` and `12.927745876346897 N/m`;
- the fixed-fluid nx4/nx8 force-per-span relative-vector delta was
  `0.006372353579412674 < 0.02`; and
- independent one- and two-step coupled preflights passed as
  `PASS_SMOKE_ONLY`, with final accepted times exactly `0.005 s` and
  `0.010 s` and zero unadvanced fluid/solid time.

All ten labels end in `__293dc69__r01` under
`validation_runs/turek_hron_component_gates/`. This chain supersedes
`5afba27` and authorizes only one fresh formal 1600-step FSI1-S0 strict-CUDA
run from zero after this documentation-only record is committed. The new label
must contain that clean documentation commit's short SHA; it must not resume or
reuse the `27111d3` prefix. This is not `PASS_FSI1_S0_GATE_ONLY` and does
not authorize M0/M1, FSI2, FSI3, Oracle, or learning.

**R26A rank-direct max-residual repair and component requalification
(2026-09-05).** Formal run `turek_hron__fsi1_s0__f513f9b__r01` started from
zero at clean `f513f9b`, accepted steps 1--7 through `t=0.035 s`, and failed
closed while preparing candidate step 8 as `FAIL_NUMERICAL_HEALTH`. The
rank-revealing-direct minimum-L2 candidate's device audit returned
`0.00010005016520153731 m/s`, just above the unchanged `1e-4 m/s` limit.
The accepted prefix is failure evidence only, is not an S0 pass, and has no
complete transition checkpoint for formal resume.

A diagnostic-only replay from zero at the same source isolated physical step
8, coupling trial 7, Q solve 5. Its 336 active rows contained 30 positive rows
and a 17-column selected basis. The existing L2 correction had an f32
all-active-row maximum residual of `0.00010005030344473198 m/s`; the same
selected basis admitted an f32 device-audited minimax witness of
`9.82458223006688e-5 m/s`. The replay rethrew the original error and did not
commit physical velocity. This established an objective mismatch, not a
reason to alter geometry admission or the public tolerance.

Clean `f2320f5` preserves least squares as the fast path. Only when that
candidate fails the existing f32 all-active-row audit does it solve the
column-normalized selected-basis Chebyshev problem, materialize the correction
to f32, and repeat the same device audit. Backend failure, nonfinite output,
or a truly infeasible system remains atomic and fail-closed; no topology gate,
tolerance, or soft fallback changed. The complete seven-test CPU module passed
in `16.169 s`, and five focused strict-CUDA contracts passed in one process in
`184.138 s`. Python compilation, Ruff check, `git diff --check`, and a fresh
final read-only review (`SHIP`, no P0--P3 finding) also passed. This does not
claim a verdict for the broad component-face module or the full test suite.

The complete ten-stage source- and host-matched component chain then passed at
clean `f2320f5`. A post-run integrity pass revalidated every file and array
hash, parent identity, row ledger, and accepted-time ledger. Every manifest is
bound to source SHA256
`478eb7707fd20761654443b334782a252ffee90dc29ba2c9b9707ba747b67c89`
and host identity
`db88ab4094ab1be43ad58b7c18cc59c756a45e1ac4f44978bc6238a4738c6a47`:

- the three solid constituents each completed 40 rows; the S100/S200 nx4 and
  S200 nx4/nx8 Point-A relative-vector deltas were
  `0.0003264915974131584` and `0.0`;
- fixed-fluid nx4 and nx8 each completed 500 rows. Their velocity/force
  span-leakage pairs were
  `0.0005504236378906395/0.0006353412795713001` and
  `0.00043036809436382663/0.0003521303648396529`; their drag means were
  `12.845664033171927` and `12.927745742812588 N/m`;
- the fixed-fluid nx4/nx8 force-per-span relative-vector delta was
  `0.006372346408124445 < 0.02`; and
- independent one- and two-step coupled preflights passed as
  `PASS_SMOKE_ONLY`, with coupling-iteration counts `[10]` and `[10, 10]`,
  exact final accepted times `0.005 s` and `0.010 s`, and zero unadvanced
  fluid/solid time.

All ten labels end in `__f2320f5__r01` under
`validation_runs/turek_hron_component_gates/`. This chain supersedes
`293dc69` and authorizes only a new documentation commit followed by one fresh
1600-step FSI1-S0 strict-CUDA run from zero whose label contains that
documentation commit's short SHA. This is not `PASS_FSI1_S0_GATE_ONLY` and
does not authorize M0/M1, FSI2, FSI3, Oracle, or learning.

**R26A formal absolute-coupling gate binding (2026-09-05).** Formal run
`turek_hron__fsi1_s0__75c240a__r01` started from zero at clean `75c240a`,
accepted steps 1--7 through `t=0.035 s`, and then failed closed at step 8 as
`FAIL_NUMERICAL_HEALTH`. It exhausted all 16 coupling trials with final
relative residual `0.2993190969189178` and final absolute RMS residual
`7.839528620993855e-6 m/s`. The best observed pair was relative residual
`0.022642592094207343` with absolute RMS residual
`4.6934343661611526e-7 m/s`. The accepted prefix is failure evidence only;
there is no transition checkpoint, accepted-interface output is not restart
state, and this run cannot be resumed or reclassified as an S0 pass.

The generic solver behaved exactly as configured: it accepts relative or
enabled absolute convergence, but the formal spec omitted
`fsi_coupling_absolute_tolerance_mps` and therefore inherited the case default
`0.0`, disabling the absolute branch. Near the start of the two-second inlet
ramp, the final candidate RMS speed was only about `2.6191e-5 m/s`, so a small
absolute mismatch was divided by a near-zero physical scale. The offline FSI1
acceptance contract already requires every accepted step's absolute coupling
RMS to be at most `1e-4 m/s`, so this was a formal-runner configuration
omission rather than permission to relax the registered numerical gate.

Clean `33b3db0` adds that existing `1e-4 m/s` limit explicitly to
`FSI1_S0_SPEC` and its exact frozen-matrix test. It does not change the generic
solver, the case default, offline acceptance, topology, physics, or any of the
15 files in the component-gate source identity. The targeted test first failed
on the missing key and then passed; the complete formal-campaign focused module
passed 31 tests in `1.16 s`. Python compilation, Ruff, `git diff --check`, and
a final read-only review (`SHIP`, no P0--P3 finding) passed. The component
source SHA256 remains exactly
`478eb7707fd20761654443b334782a252ffee90dc29ba2c9b9707ba747b67c89`,
so the `f2320f5` component chain remains source-matched and nx4/nx8 must not be
rerun. This authorizes only a new documentation commit followed by one fresh
1600-step strict-CUDA S0 from zero under a new label containing that commit's
short SHA; it does not authorize any later gate.

This report records what has been **verified by runnable experiment**, what has
been **diagnosed but not fixed**, and what is a **method-limited frontier**. It
deliberately separates confirmed results from confounded comparisons. Every
physical number below EMERGES from the unchanged force/displacement integrals —
nothing was scaled or assigned to match a reference.

## 1. Fixes landed this campaign

| Commit | Change | Verification |
|---|---|---|
| `e5c80cb` | **Momentum-consistent step pressure** — `project()` increment mode (`accumulate_pressure_into_previous`) so the per-step re-projections stop overwriting the physical pressure | FSI1 total drag 0.32× → **0.92×** reference; form drag ~0 → 9.05 N/m |
| `e5c80cb` | **Obstacle-mask surface force integration** (pressure/form + viscous) | Cylinder-only reproduces Schäfer–Turek 2D-1: **Cd 5.79 vs 5.58 (within 4%)** |
| `e5c80cb` | **y-symmetry face-symmetric Dirichlet (mode 2)** — static wall rows only | Mirror unit test passes; structurally correct (dynamical impact small — see §4) |
| `e5c80cb` | **Re-projection budget** (1200 iters / 1e-4 tol) | Measured A/B: **1.72× speedup**, accuracy deltas < 0.2% |
| `7f7da08` | **Strong coupling (Picard + Aitken)** at the case level, gated `fsi_coupling_iterations` (default 1 = legacy) | Defeats FSI3 added-mass divergence: 30 ms blow-up → 1200 steps stable |

The pressure-overwrite root cause (each step ran up to 3 `fluid.project()` calls;
the re-projections solved a tiny residual that overwrote the ~200 Pa physical
field, collapsing cylinder stagnation) is documented in
`docs/` and the project memory.

## 2. Results vs reference (settled windows)

| Case | Quantity | Ours | Reference | Verdict |
|---|---|---|---|---|
| **Cylinder-only Re=20** | Cd | 5.79 | 5.58 (Schäfer–Turek 2D-1) | ✅ **4%** — solver + force integration validated |
| **FSI1** (steady) | total drag /span | 13.10 N/m | 14.295 | 0.92× (was 0.32×) |
| | tip uy | +0.25 mm | +0.82 | sign correct (was −0.58); magnitude ~30% |
| | form/friction drag | 9.05 / 4.53 | — | form recovered from ~0 |
| **FSI2** (ratio 10) | stability | 2000 steps, no blow-up | — | ✅ stable at real loads |
| | tip uy (transient) | rings ±6 mm @ ~1 Hz | +1.23 ± 80.6 @ 2.0 Hz | limit cycle not reached in 2 s (see §4) |
| **FSI3** (ratio 1) | stability | 1200 steps stable (strong coupling) | — | ✅ no more added-mass blow-up |
| | tip uy | +18 ± 6.5 mm @ 2.5–2.9 Hz | +1.5 ± 33.5 @ 5.3 Hz | ❌ self-excited flutter NOT reproduced (§4) |
| | total drag /span | 637 N/m | 452 | high; comparison partly confounded (§4) |

## 3. FSI3 diagnosis (complete, evidence-based)

The FSI3 self-excited flutter is **not reproduced**; the beam damps to a steady
+17 mm deflection instead of the ±33.5 mm limit cycle. The failure was localized
by three isolating experiments:

1. **Wake sheds correctly** — cylinder-only at Re=200 sheds von Kármán vortices
   (Cl ± 0.15, ~5 Hz, close to the reference 5.3 Hz flutter frequency).
2. **The shedding force reaches the beam** — the coupled lift oscillates at
   3.75 Hz.
3. **The beam does not lock in** — tip_uy responds at ~2.5 Hz, near its own
   natural frequency (measured 2.67 Hz in vacuum, blend-independent), far below
   the 5 Hz shedding, so it never synchronizes into the lock-in limit cycle.

Cheap levers **ruled out by direct measurement** (not argued away):

- **Coupling convergence** is not the limiter — the existing run already
  converges (post-ramp residual median 6e-4, 82% of steps below 1e-3 tolerance);
  more iterations cannot remove damping that is not there.
- **PIC dissipation** is not the limiter — a vacuum beam free-vibration test
  shows the numerical damping ratio ζ ≈ 0.008 and is essentially independent of
  the PIC/FLIP blend (0.0087 → 0.0081 across blend 0.0 → 0.9).
- **2× grid does not fix the drag** — cylinder-only Cd is 3.25 (base) vs 3.19
  (2×): not resolution-sensitive at this refinement.

## 4. Diagnosed limitations (honest caveats)

- **FSI3 flutter is method-limited.** Reproducing lock-in flutter at density
  ratio 1 would require, together: substantially finer grid (> 2×), a
  less-dissipative advection scheme (the current predictor is semi-Lagrangian,
  which smears the fine wake structures that pump energy into the beam), and a
  clean reference decomposition. Investment is large with uncertain payoff.
- **Drag "41% high" is partly a confounded comparison.** The cylinder-only Cd
  was compared against a value inferred from the cylinder+beam (oscillating)
  reference; a static-beam solver state and an oscillating-beam reference are not
  the same configuration. The clean, unconfounded anchor is the Re=20
  cylinder-only match (4%).
- **MPM beam is ~35% stiffer than Euler–Bernoulli** (vacuum f₁ 2.67 Hz vs 1.97
  Hz theory) — a separate solid-accuracy item, not chased here.
- **y-symmetry mode 2 is structurally correct but dynamically small** — it fixes
  the wall face-constraint asymmetry (unit-tested) but does not by itself move
  the FSI1 tip_uy magnitude; the residual is resolution.
- **FSI2 limit cycle needs longer physical time** — 2 s of simulation shows
  transient ringing but not the developed ±80 mm cycle (which establishes over
  many periods); this is a time-budget × step-cost limit, not a stability limit.
- **CUDA backend is not run-to-run bit-deterministic** (float-atomic scatter
  ordering); regression checks use a variance-envelope test, not bit-identity.

## 5. Reproduction

```bash
cd /path/to/EasyFsi
PY=python

# FSI1 steady (700 steps, ~2.5 h): total drag → ~13.1 vs ref 14.295
$PY -c "from dataclasses import replace; import cases.turek_hron_fsi as t; \
  t.run_turek_hron_fsi(replace(t.fsi1_config(step_count=700), inlet_ramp_time_s=1.0), preset='fsi1', output_dir='out_fsi1')"

# FSI3 strong-coupled (density ratio 1; needs fsi_coupling_iterations>1 or it diverges)
$PY -c "from dataclasses import replace; import cases.turek_hron_fsi as t; \
  t.run_turek_hron_fsi(replace(t.fsi3_config(step_count=500), inlet_ramp_time_s=0.3, \
    flow_predictor_substeps=4, fsi_coupling_iterations=4, fsi_coupling_tolerance=1e-3), \
    preset='fsi3', output_dir='out_fsi3')"
```

Key config knobs:
`fsi_coupling_iterations` (minimum 2), `fsi_coupling_tolerance`,
`fsi_coupling_absolute_tolerance_mps`, `fsi_coupling_initial_relaxation`,
`accumulate_reprojection_pressure`, `flow_reprojection_iterations`,
and the flow/solid substep controls. Turek-Hron now uses the canonical
component-face ledger: y-min/y-max no-slip walls and the z-max parabolic inlet are
directed external component faces, and the ledger is prepared and sealed before
each solve. Beam surface rows use the three physical finite-segment chains;
deep beam cells use the current MPM particle-volume mask rather than a global
signed-distance extrapolation from those open chains.

## 6. Open frontiers (if FSI3 flutter is pursued later)

1. **Grid convergence, full FSI** — 2× grid + dt/2 for the whole cylinder+beam
   (tens of GPU-hours) to see whether flutter onsets with sharper wake forcing.
2. **Conservative flux advection** — replace the semi-Lagrangian predictor with
   a `div(u⊗u)` finite-volume convection (core-solver change, stability-gated per
   case); reduces the numerical dissipation that damps the wake–beam energy
   transfer.
3. **IQN-ILS coupling** — quasi-Newton interface solver (parts already exist in
   `simulation_core/coupling/fsi_coupling.py`) for tighter convergence with less
   artificial relaxation damping than Aitken.
4. **MPM beam stiffness calibration** — reconcile the ~35% over-stiffness vs
   Euler–Bernoulli.
