# Turek–Hron FSI1/FSI2/FSI3 Numerical Validation R26A Goal

## R15 test-correction checkpoint, 2026-09-06

Full GREEN25 r02 completed with three failing methods; two corrected fixtures
subsequently passed 2/2. The reviewed extrusion cache assertion correction
and complete GREEN25 r03 remain pending native verification at this commit.
The solver core is unchanged; continuous and formal success remain unproved.
See [TUREK_HRON_R15_FALLBACK_VALIDATION_REPORT_2026-09-06.md](TUREK_HRON_R15_FALLBACK_VALIDATION_REPORT_2026-09-06.md) for hashes,
exact test changes, evidence boundaries and required next gates.

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

Historical A50 result: FAILED_WITH_VALIDATED_ACCEPTED_PREFIX;14 focused contracts and three frozen controls passed at that source.

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

The following numerical prefix was measured before the terminal repair.
The reviewed production band-ordering correction is applied at core `c989f6ba`
/ fluid `a1a3324b`, with 45 distinct host checks passing. Production frozen-pose
r03 passed on strict CUDA in 58.056119 s, adding 92 excluded cells then zero.
Its native final H/global residuals are `8.992282687358966e-7` /
`4.3388867197791114e-5 m/s`, with all 112 raw positions/targets unchanged and
zero physical-time advancement. The preceding old-source r02 shared-field RED
control and lower bound `2.184912935710554e-6 m/s` remain valid. The finalized
topology gives the old witness rows legal free support; it does not solve their
old H-only system. These private reconstructed-owner checks retain missing
capture-state and unqualified-intermediate-health limits. Full coupled
numerical validation remains pending.

The from-zero diagnostic r11 accepted four steps to t=.020 s, then
rejected step 5 with two geometry `prepare_pair_arbitration` conflicts; the first
reported face is `(1,50,304)`, axis 0. Original candidate/native-record validators
pass for all four accepted records: all 356 CSV values match exactly, and all
4 x 34 native fields match in dtype, shape and byte hash. Fluid and solid each
consume .005 s per step, .020 s in total, with zero remaining time; each accepted
step has one trial and no rejection. All 193 recorded solver source hashes match
before and after the run. Runtime reports restored fluid, solid, marker interface
and marker pressure gradient state and no capture errors; derived search/boundary
state requires rebuilding. An independent full post-rollback state comparison
remains pending.
This partial diagnostic does not qualify component or formal FSI1/2/3 acceptance.

The coincidence/provenance correction passed 16 focused strict-CUDA tests;
the exact zero-free certificate correction passed 19 and closes the actual
r07 public collective input below the unchanged 1e-6 m/s hard gate.
The earlier from-zero r09 diagnostic accepted one step to t=0.005 s, then
rejected step 2 at closure call 19. CPU revalidation of the accepted candidate,
native arrays and all 89 CSV fields passes: fluid and solid each advance .005 s
with zero remaining time. This accepted step has one trial and no rejection.
The runtime reports physical rollback restored after the step-2 failure.
The r09 actual strict-CUDA matrix certifies an unavoidable 2.184912935710554e-6
m/s hard residual: physical markers 84/85 have identical z support rows and
different original targets. Increasing iterations cannot meet the 1e-6 gate.
The applied correction delays target closure until band saturation; full
coupled and benchmark proof remain outstanding. Fixed112 z refinement
beyond 405 currently violates the existing marker-spacing guard; no guard or
threshold has been relaxed. Code remains dirty and uncommitted. Source-matched
component requalification and formal FSI1/2/3 acceptance remain outstanding.
The separate optional118-vertex projection-endpoint design remains external;
it is distinct from the applied same-storage terminal repair.
See [the current handoff](../refactoring/TUREK_HRON_FSI123_TRACE_REPAIR_THREAD_HANDOFF_2026-09-05.md) and
[the trace-space audit](TUREK_HRON_TRACE_SPACE_AUDIT_2026-09-05.md).

Branch: **codex/turek-hron-fsi123-validation-r26a**

Start commit: **b7f32c0b66b8bc86277cdac86c88724301746cbf**

The start commit is the reviewed R25B import-boundary repair. R25B remains
complete and frozen. It is not rerun on CUDA, reinterpreted, or extended by this
goal.

## Current execution stop condition

The user requested a quota checkpoint on 2026-09-05: when the main Codex
quota is observed at 10% remaining or less, stop this task's owned test and
numerical processes, preserve completed artifacts and an exact handoff, then
stop work. This is not a numerical failure or permission to consume a reset.
Use append-only live checkpoints outside the repository so monitoring does
not dirty an active formal run. User authorization is required to resume after
that stop boundary.

## 1. Purpose and evidence boundary

R26A answers one question:

> Does the current shared HIBM–MPM solver reproduce the numerical behavior of
> the Turek–Hron FSI1, FSI2, and FSI3 benchmark family with controlled spatial,
> temporal, force, geometry, and coupling error?

The required order is:

\[
\mathrm{FSI1\ correctness}
\rightarrow
\mathrm{FSI2\ large\ deformation}
\rightarrow
\mathrm{FSI3\ strong\ added\ mass}
\rightarrow
\mathrm{Oracle\ headroom}
\rightarrow
\mathrm{Kalman/GRU}
\rightarrow
\mathrm{squid}.
\]

R26A contains only the first three stages. During R26A:

- do not train or tune GRU, Kalman, Kalman+GRU, POD-AR, or another predictor;
- do not run a predictor in the accepted solver path;
- do not run an Oracle probe;
- do not optimize coupling trials, pressure work, or wall time;
- do not tune geometry, marker, force, damping, or solver parameters against the
  final reference error;
- do not call an old run, a contract test, or a smoke run a numerical validation;
- do not use a Turek–Hron numerical benchmark as experimental validation of a
  biological squid.

Old Turek–Hron runs predate the current shared-driver state and remain historical
diagnostics. Only fresh, source-matched R26A outputs can pass this goal.

## 2. Hard gates and stop rules

The gates are sequential and fail closed:

1. Lock canonical sources and implement offline reference contracts.
2. Validate the Featflow importer and deterministic limit-cycle analysis.
3. Validate geometry, quasi-2D equivalence, material parameters, force scope,
   Point A, and isolated fluid/solid constituents.
4. Run and assess FSI1.
5. Enter FSI2 only after FSI1 reaches benchmark-quality pass.
6. Enter FSI3 only after FSI2 reaches benchmark-quality pass.
7. Declare R26A complete only after all three cases reach benchmark-quality pass.
8. Oracle remains forbidden until R26A is complete.
9. Learning remains forbidden until a later Oracle gate demonstrates solver-work
   headroom.

A failed gate stops later numerical stages. The response to failure is diagnosis
at that gate, not looser tolerances or a model that learns around it.

## 3. Source-first reference contract

### 3.1 Source identities

Reference data must be keyed first by source, then by case. At minimum:

- **featflow_tu_dortmund_web**: primary canonical benchmark definitions and
  published convergence tables;
- **featflow_raw_fsi2_dt_0p0005**: primary raw FSI2 series;
- **featflow_raw_fsi3_dt_0p00025**: primary raw FSI3 series;
- **lsdyna_2013_crosscheck**: diagnostic cross-check only.

No lookup may silently merge sources or fall back from Featflow to LS-DYNA.
LS-DYNA values must never overwrite Featflow values.

Primary sources:

- https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_fsi_tests.html
- https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_reference.html
- https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_definitions.html
- https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_quantities.html

LS-DYNA cross-check:

- https://lsdyna.ansys.com/aerofsi1/

### 3.2 Canonical geometry and case parameters

The contract records:

- channel length 2.5 m and height 0.41 m;
- cylinder center \((0.2,0.2)\) m and radius 0.05 m;
- beam length 0.35 m and thickness 0.02 m;
- material Point A initially at \((0.6,0.2)\) m;
- fluid density 1000 kg/m3 and kinematic viscosity \(10^{-3}\) m2/s;
- Poisson ratio 0.4;
- FSI1: mean inlet 0.2 m/s, solid density 1000 kg/m3, \(E=1.4\) MPa;
- FSI2: mean inlet 1.0 m/s, solid density 10000 kg/m3, \(E=1.4\) MPa;
- FSI3: mean inlet 2.0 m/s, solid density 1000 kg/m3, \(E=5.6\) MPa;
- parabolic inlet and the documented 2 s cosine startup ramp;
- no slip on walls, cylinder, and interface;
- a zero-mean outlet pressure reference;
- St. Venant–Kirchhoff elasticity with the local plane-strain convention.

### 3.3 FSI1 primary anchor

Use the finest published Featflow level 7+0 row as the explicit primary anchor:

| quantity | value |
| --- | ---: |
| Point A \(u_x\) | \(2.270493\times10^{-5}\) m |
| Point A \(u_y\) | \(8.208773\times10^{-4}\) m |
| total drag | 14.29426 N |
| total lift | 0.7637460 N |

This is a selected published numerical anchor, not an exact analytical solution.

### 3.4 FSI2 and FSI3 primary anchors and envelopes

The primary periodic comparison uses the frozen raw Featflow series and the
deterministic analyzer in Section 4.1. The last complete rising-crossing-bounded
cycle produces these source-matched anchors:

| case | gated quantity | primary raw anchor |
| --- | --- | ---: |
| FSI2 | Point A \(u_y\) amplitude | 0.08165565385 m |
| FSI2 | Point A \(u_y\) frequency | 1.93061437683808 Hz |
| FSI2 | total-drag midrange | 215.088610865 N |
| FSI2 | total-lift amplitude | 237.661293 N |
| FSI3 | Point A \(u_y\) amplitude | 0.03491637475 m |
| FSI3 | Point A \(u_y\) frequency | 5.47355995969522 Hz |
| FSI3 | total-drag midrange | 460.31160355 N |
| FSI3 | total-lift amplitude | 153.527757 N |

For each gated quantity \(q\) with raw anchor \(r_q\), define:

\[
e_q=\frac{|q-r_q|}{|r_q|},\qquad
\delta_q(a,b)=\frac{|q_a-q_b|}{|r_q|}.
\]

Only these four nonzero quantities enter the periodic percentage gates. Point A
\(u_x\), displacement midranges, drag amplitude, lift midrange, and per-signal
harmonics are still reported, but are diagnostic because a near-zero midrange
does not define a stable relative-error denominator.

The broader Featflow convergence family remains a consistency envelope, not an
alternative denominator or a pass-on-overlap rule.

FSI2 expected community envelope:

- Point A \(u_y\) amplitude: 79–82 mm;
- Point A \(u_y\) frequency: about 1.93 Hz;
- total drag midrange: 210–215 N;
- total lift amplitude: 228–238 N.

FSI3 expected community envelope:

- Point A \(u_y\) amplitude: 34–36 mm;
- Point A \(u_y\) frequency: 5.3–5.5 Hz;
- total drag midrange: 458–461 N;
- total lift amplitude: 146–155 N.

The exact Featflow table rows and their level/time-step identities must also be
stored. Rounded envelope values are for assessment reporting; they do not replace
the selected raw anchors or permit choosing a favorable reference after a run.

### 3.5 Raw-series identity

Store immutable upstream bytes under:

~~~text
docs/validation/turek_hron_featflow/
  ref_fsi2.point
  ref_fsi2.manifest.json
  ref_fsi3.point
  ref_fsi3.manifest.json
~~~

Pre-audited upstream identities:

| case | URL suffix | bytes | rows | time range | dt | SHA256 |
| --- | --- | ---: | ---: | --- | ---: | --- |
| FSI2 | fsi2/0p0005/ref_fsi2.point | 1,783,320 | 9,240 | 10.0–14.6195 s | 0.0005 s | d4e192f5ae6aa493d36c472b68d8d734d54194aca6e9f0a70014816b6835dd75 |
| FSI3 | fsi3/0p00025/ref_fsi3.point | 1,113,417 | 5,769 | 5.0–6.4420 s | 0.00025 s | c428bc3ff48c1698cd10aacf0941cba6aabdfcb9d61c8b7e42c343f381434e63 |

Each raw row must have exactly 12 finite numeric columns and strictly increasing,
uniform time.

Documented one-based columns are:

| column | meaning |
| ---: | --- |
| 1 | time |
| 5 | beam drag |
| 6 | beam lift |
| 7 | cylinder drag |
| 8 | cylinder lift |
| 11 | Point A \(u_x\) |
| 12 | Point A \(u_y\) |

Columns 2–4 and 9–10 remain opaque. The importer must not invent meanings for
them. Whole-body forces are exactly:

\[
F_D = c_5 + c_7,\qquad F_L = c_6 + c_8.
\]

Every manifest binds schema version, source ID, exact URL, filename, byte count,
row count, column count, time range, dt, SHA256, documented column meanings,
opaque columns, units, force scope, coordinate/sign convention, and extraction
policy. Unknown, missing, extra, contradictory, malformed, or hash-mismatched
fields fail closed.

## 4. Offline validation implementation

Extend the existing solver-free package
**src/refactored/validation/turek_hron_fsi** rather than adding a third
case-local reference table.

Planned ownership:

- **references.py**: immutable source-first FSI1/2/3 contracts;
- **featflow.py**: manifest validation and raw 12-column importer;
- **limit_cycle.py**: deterministic periodic analysis;
- **acceptance.py**: retain the current FSI1 public API while consuming the
  central canonical contract;
- **cases/turek_hron_fsi.py**: compatibility projection only, not independent
  data ownership.

Returned reference and series objects must be immutable. Arrays, if used, are
read-only. Importing this package must not initialize Taichi or a solver.

### 4.1 Deterministic limit-cycle rule

The periodic gate uses Point A \(u_y=0\) rising crossings.

A rising event is detected from the last nonzero negative sign to the next
nonzero positive sign. With no exact-zero samples, event time is linearly
interpolated. A run of exact-zero samples creates one event at the first zero
sample and never duplicate events.

At least four rising events are required. The three intervals bounded by the
last four events are the last three complete cycles. Samples after the final
event form a partial tail and are excluded.

For every signal and complete cycle:

\[
\mathrm{midrange}=(\max+\min)/2,\qquad
\mathrm{amplitude}=(\max-\min)/2.
\]

The primary period is the interpolated Point A \(u_y\) crossing interval and the
primary \(u_y\) frequency is \(1/T\).

For per-signal frequency reporting, use the last three complete \(u_y\) cycles,
uniformly sample at the validated dt, subtract the arithmetic mean for spectral
analysis only, use a rectangular real FFT, exclude DC, and select the largest
non-DC magnitude; ties choose the lower frequency. This spectral estimate is
not used to redefine extrema-based midrange or amplitude.

Limit-cycle stability requires:

- period spread divided by mean period below 1%;
- Point A \(u_y\) amplitude spread divided by mean amplitude below 2%;
- Point A \(u_y\) midrange range below 1% of mean amplitude;
- final total-drag and total-lift amplitude no more than 2% above the first of
  the three cycles, and neither force amplitude strictly increasing over all
  three cycles.

These are R26A preregistered local gates. Featflow does not prescribe this
zero-crossing implementation.

## 5. Component gates before coupled runs

### 5.1 Geometry and quasi-2D equivalence

Focused tests and one bounded comparison must verify:

- exact cylinder center/radius, beam length/thickness, Point A, root clamp, two
  long marker faces, and tip cap;
- connected cylinder/beam geometry with no fluid pocket inside the solid;
- correct inlet/outlet axis and no-slip walls;
- span 0.05 m with \(n_x=4\) versus \(n_x=8\);
- Point A displacement and force per span differ by at most 2% between the two
  span resolutions;
- spanwise velocity/displacement norm is at most \(10^{-3}\) of the in-plane
  norm;
- spanwise force magnitude is at most \(10^{-3}\) of in-plane force magnitude.

### 5.2 Solid-only gate

Verify before coupled FSI:

- \(E=1.4\) MPa, \(\nu=0.4\) gives \(\mu=0.5\) MPa and the expected
  plane-strain Lamé convention;
- FSI3 \(E=5.6\) MPa gives \(\mu=2.0\) MPa;
- the fixed root does not drift beyond the solver’s declared position tolerance;
- artificial damping is disabled for the canonical baseline;
- 100 versus 200 MPM substeps changes the selected displacement response by
  less than 2%;
- every accepted solid physical step consumes the full macro \(\Delta t_s\).

### 5.3 Fluid-only gate

With the cylinder and beam fixed, verify:

- discrete inlet mean agrees with the preset mean within 0.5%;
- walls, cylinder, and beam obey no slip;
- outlet pressure reference and flow direction are correct;
- time-windowed mass imbalance is below 1%;
- pressure and viscous force parts have documented signs and finite values;
- total force equals beam plus cylinder pressure plus cylinder viscous force to
  numerical roundoff;
- every accepted fluid physical step consumes the full macro \(\Delta t_s\).

Point A extrapolation, case presets, marker-area accounting, and plane-strain
wiring already have focused tests. R26A adds only missing arithmetic, runtime,
and constituent-construction checks.

### 5.4 Frozen executable component protocol

The component campaign is fixed before any component result is observed. Its
common FSI1 controls are:

- L0 grid `(4,48,288)`, `dt_s = 0.005`, automatic side/tip marker counts,
  `flow_predictor_substeps = 1`, `fluid_advection_scheme = "rk2"`,
  `flow_projection_iterations = 4000`, and `flow_cg_tolerance = 1e-6`;
- `ib_anisotropic_envelope = True`,
  `classify_far_internal_nodes = True`, and
  `flow_cg_preconditioner = "fv_multigrid"`;
- `flow_reprojection_iterations = 1200`,
  `flow_reprojection_cg_tolerance = 1e-4`,
  `marker_reseed_interval_steps = None`, and
  `velocity_damping = 1.0` (no artificial damping).

Automatic markers resolve to side/tip counts `(54,4)` on L0, `(108,7)` on
L1, and `(162,10)` on L2; changing only nx from 4 to 8 must not change them.

These choices follow the pre-reference geometry/topology contract, not a fit to
Featflow output. A time-zero initialization audit must still prove connected
cylinder/beam geometry, complete beam-interior obstacle coverage, no sealed
fluid pocket, finite zero-load fields, and the expected automatic marker layout.
Failure stops the campaign; it does not authorize switching a flag after seeing
reference error.

#### 5.4.1 Pre-campaign segment and far-interior correction (2026-09-03)

The first fixed-fluid time-zero attempt failed before any physical step with 12
conflicting canonical alpha claims at the physical free tip. A bounded device
diagnostic traced every conflict to adjacent point-marker authors. Registering
the three existing physical marker groups as three disconnected open polylines
(lower face, upper face, and free tip; 109 segments on L0) reduced alpha,
target, region, and aggregate claim conflicts to zero.

Those open polylines are valid local projection geometry, but they are not a
closed-volume certificate. An intentionally bypassed diagnostic produced 348
globally signed far-internal nodes and exposed an upstream fluid witness that
the free-tip normal would misclassify. Therefore the frozen
`classify_far_internal_nodes = True` setting has the following stricter
meaning before the first canonical campaign:

- finite segments own only local boundary classification and interpolation;
- far-interior membership comes only from the current deformed MPM particle
  volume (`solid.x` plus `solid.F`), installed at time zero and refreshed after
  every solid macro step before the post-solid search;
- segment-plus-far search without that scalar `i32` grid mask fails closed;
- each initial and post-solid search fails closed unless every `INTERNAL` node
  is a member of that current live particle-volume mask;
- the dynamic volume and carve fields remain part of fluid save/restore, so a
  rejected coupling trial cannot leak geometry into an accepted state; and
- the time-zero component audit requires both complete beam-interior coverage
  and exactly zero obstacle cells outside the analytic beam-cell-intersection
  or canonical cylinder volume.

This is a pre-reference topology correction, not parameter fitting. It changes
the source identity, so all earlier component artifacts are source-stale and
must be regenerated before any fixed-fluid or coupled result is canonical.

A subsequent source-matched nx4 fixed-fluid attempt (`r02`) passed the time-zero
audit but failed during the first physical step, before any accepted row, with
six canonical component-face target conflicts. The first witness was the
inactive-axis face at the physical free tip. A direct fluid row and its
relocation shadow had the same source/storage identity, unique registered
segment, nearest marker, one-hot endpoint weights, boundary point, region, and
exact serialized zero target; their distinct interior rays nevertheless
produced non-bit-identical effective targets at roundoff scale.

That endpoint repair remained fail-closed. It consumed the relocation shadow
only when the existing redundant-shadow proof succeeded and all endpoint
identity fields matched exactly, both effective targets were finite, and their
difference was at most `1e-6 m/s`. The direct claim remained authoritative;
the implementation neither averaged claims nor relaxed the global conflict
rule. Its focused strict-CUDA evidence was two endpoint tests plus six
neighboring component-face contract tests passing. The full component-face
geometry module timed out at 1200 s and therefore had no verdict.

After the solid-only chain was regenerated against that source, nx4 fixed-fluid
attempt `r03` passed the time-zero audit and then failed during its first
physical step, before any accepted row or usable artifact, with two canonical
target conflicts. The first direct/shadow pair projected to the same exact
interior point of segment `(109,110)`: nearest marker `110`, weights
`(0.3541681767,0.6458318233,0)`, full boundary point, region, normal, and
serialized zero target all agreed. Both outer redundant-shadow proofs were
valid, while independently reconstructed effective targets differed by only
about `2.07e-19 m/s`. The old one-hot-only predicate was therefore the sole
rejected condition.

The narrow repair now applies to a same exact projected wall point, whether an
endpoint or segment interior. It still requires the complete outer
redundant-shadow proof, the inactive axis, exact serialized component target,
exact nearest marker, exact three-component projection weights, exact
three-coordinate boundary point, finite effective targets, and an effective
difference no greater than `1e-6 m/s`. Only the shadow is consumed; the direct
f32 target bits remain authoritative and no averaging or global tolerance
change is allowed. Focused strict-CUDA evidence is three tests covering the
interior and both endpoints plus serialized/effective/identity-drift negative
cases, followed by twelve neighboring component-face contract tests passing.
Python compilation, `git diff --check`, and Ruff pass. The full module was not
rerun, so there is still no module-level verdict. This source change makes all
preceding solid-only artifacts source-stale again: the entire frozen component
order must be regenerated before nx4 fixed-fluid is retried, and none of this
is yet FSI1 numerical evidence.

#### 5.4.2 Source-matched solid pass and fixed-fluid stop (2026-09-03)

At clean commit `f16737e`, the complete frozen solid-only chain passed as
`PASS_COMPONENT_ONLY` under strict CUDA:

- `turek_hron__component__solid_s100_nx4__20260903__r07`;
- `turek_hron__component__solid_s200_nx4__20260903__r05`;
- `turek_hron__component__solid_s100_s200_nx4__20260903__r05`, with Point-A
  relative-vector delta `0.0003264915974131584` (0.032649%);
- `turek_hron__component__solid_s200_nx8__20260903__r05`;
- `turek_hron__component__solid_s200_nx4_nx8__20260903__r05`, with Point-A
  relative-vector delta `0.0`.

Every solid run completed all 40 rows with zero root displacement and zero
displacement/velocity spanwise leakage. The next canonical nx4 fixed-fluid
attempt, `turek_hron__component__fixed_fluid_nx4__20260903__r04`, reached the
first frozen evaluation row at physical step 401 (`t=2.005 s`) and failed
closed with `FAIL_BEAM_MARKER_NO_SLIP`: RMS residual
`0.0012601176039343787 m/s` and maximum residual
`0.005936640314757824 m/s`, against `1e-4` and `0.002 m/s`. Sampler coverage
was complete: 112 valid and zero invalid markers.

Noncanonical in-memory diagnostics first crossed the RMS limit at step 33.
Overall RMS/max was `0.00010018624307816692/0.0004438578907866031 m/s`; the
108 direct samples contributed
`8.716031220311704e-05/0.0002774639579001814`, while four free-tip
`normal_walk` samples contributed
`0.0002755487092652955/0.0004438578907866031`. At step 100 the corresponding
overall, direct, and `normal_walk` RMS/max pairs were
`0.000522078997451709/0.002698277123272419`,
`0.0004219407204557312/0.0013645613798871636`, and
`0.0016807570266407306/0.002698277123272419 m/s`. The failed canonical `r04`
directory and explicitly diagnostic `r01`/`r02` directories are empty; these
diagnostic values exist only in console/in-memory traces.

The execution-path defect was localized: time-zero marker closure directly
constrained 308 of 336 marker-axis equations, while 28 q-free directions
depended on marker-Q. The generic HIBM-MPM core called `fluid.project(...)`
without the existing marker-Q/pressure-nullspace adapter. All pressure
marker-nullspace enable/prepare/apply diagnostics were false or zero, while
terminal `normal_walk` positions were bitwise identical to the closure
positions. The repair now shares one persistent Q/P adapter across Turek main,
consistency, and post-solid projections and preserves the legacy/default ANSYS
path. A first review caught a default-path residual/viscous obstacle regression;
that issue was fixed under a focused behavior test before any CUDA rerun.

The repaired dirty source passed `142` focused tests plus `15` subtests,
compilation, Ruff, diff checks, and a fresh read-only final review. One
**noncanonical, in-memory** strict-CUDA nx4 step then passed with exact
requested/accepted time `0.005/0.005 s`, zero unadvanced time, Q
prepared/converged/committed, pressure-nullspace projection on all velocity
paths, zero invalid actuation/correction entries, 112/112 valid markers, and
terminal no-slip RMS/max
`2.98977615920801e-07/9.697889709059382e-07 m/s`. This probe wrote no artifact
and is not a component PASS. The source change invalidates every `f16737e`
solid artifact by source identity, so the complete frozen component chain must
be regenerated before nx4/nx8 fixed-fluid and coupled-preflight gates. No
fixed-fluid PASS or FSI1 evidence exists.

#### 5.4.3 Rank-deficient Q repair and complete component pass (2026-09-03)

At clean commit `7b80a6b`, the generic marker-Q defect was closed for the L0
component path without changing the default ANSYS behavior. The failing 336-row
affine Q system was physically feasible but rank deficient. Its intended
static rank was 16: 12 positive-mobility rows were dependent and 308 rows were
unactuated. The prior f32 device matvec was not sufficiently self-adjoint for
PCG at the failure point. Turek now explicitly opts into a bounded f64
rank-revealing direct fallback after the unchanged PCG path fails; the public
projector default remains PCG and the official ANSYS allocator does not opt in.
The fallback audits the actual rounded f32 correction against all active rows
before commit and retains the existing atomic transaction boundary.

The same commit makes marker-Q evidence source-complete. Both Q implementation
files are explicit source-hash members, and each generic projection cycle is
retained in order with a strict stage, backend, rank flag, marker/constraint and
iteration counts, actual maximum residual, and rank-partition residuals. The
fixed-fluid history stores this ledger as strict deterministic JSON. Focused
verification passed `86` tests plus `120` subtests; compilation, Ruff E9/F,
`git diff --check`, and a fresh read-only review also passed with no remaining
P0--P3 finding.

One non-artifact 20-step strict-CUDA nx4 reproduction completed the old step-19
failure. Its main Q transaction used `rank_revealing_direct`, with 112 markers,
336 constraints, rank partition `(16,12,308)`, and actual/structural maximum
residual `6.686404049105477e-06 m/s`. The consistency Q stayed on PCG. Step 20
also completed. This diagnostic authorized canonical component reruns but is
not itself a component artifact.

The complete frozen solid chain was regenerated against `7b80a6b` and passed
as `PASS_COMPONENT_ONLY`:

- `turek_hron__component__solid_s100_nx4__20260903__r10`;
- `turek_hron__component__solid_s200_nx4__20260903__r07`;
- `turek_hron__component__solid_s100_s200_nx4__20260903__r07`, Point-A
  relative-vector delta `0.0003264915974131584`;
- `turek_hron__component__solid_s200_nx8__20260903__r07`; and
- `turek_hron__component__solid_s200_nx4_nx8__20260903__r07`, Point-A
  relative-vector delta `0.0`.

All three constituent runs contain 40 rows, zero root displacement, and zero
displacement/velocity span leakage. Their manifests are clean, source-matched,
strict-CUDA artifacts at the same commit.

Both frozen fixed-fluid runs then passed all 500 rows:

| run | velocity span leakage | force span leakage | max inlet error | mass imbalance |
| --- | ---: | ---: | ---: | ---: |
| `turek_hron__component__fixed_fluid_nx4__20260903__r06` | `5.504236666806834e-4` | `6.353449740090614e-4` | `0.0023600231749394684` | `0.0006576385367210474` |
| `turek_hron__component__fixed_fluid_nx8__20260903__r01` | `4.303681250445346e-4` | `3.5212848977038484e-4` | `0.0023600154078545765` | `0.0011460183582938626` |

Every row contains exactly two ordered Q cycles (`main` and
`post_dirichlet_reconstruction_consistency`). Each run used PCG for 523 cycles
and rank-revealing direct Q for 477 cycles; the first and last direct
occurrences were at main steps 19 and 500. The rank signature was always
`(16,12,308)`.
Maximum Q residuals were `9.701458475319669e-05 m/s` for nx4 and
`9.830097405938432e-05 m/s` for nx8; maximum direct structural residuals were
`7.689253834541887e-06` and `7.699662091908976e-06 m/s`. The source-matched
comparison
`turek_hron__component__fixed_fluid_nx4_nx8__20260903__r01` passed with
force-per-span relative-vector delta `0.0063723531512411`.

Finally, fresh parentless production-path preflights passed as
`PASS_SMOKE_ONLY`:

- `turek_hron__component__coupled_preflight_step1_nx4__20260903__r01`
  accepted one step to `0.005 s` in 10 coupling iterations; and
- `turek_hron__component__coupled_preflight_step2_nx4__20260903__r01`
  independently started at zero and accepted two steps to `0.010 s`, using 10
  coupling iterations per step.

For every accepted preflight row, both fluid and solid consumed the complete
`0.005 s` macro interval and reported zero unadvanced time. All artifact hashes,
source hashes, clean-commit identity, row schemas, and final accepted-time
arrays were revalidated after completion.

The component prerequisite was complete at `7b80a6b` and authorized
**FSI1-S0 only** for that exact source. This is not an FSI1 numerical pass. The
current dense fallback has a
hard 512-constraint limit: L0 has 336 constraints, but L1 and L2 have 669 and
1002. A scalable rank-deficient Q backend must be implemented and validated
before M0/M1; this known boundary cannot be bypassed by changing markers,
tolerance, or the preregistered matrix.

The solid-only matrix contains three fresh runs:

| run | grid | solid substeps | macro steps | prescribed load |
| --- | --- | ---: | ---: | --- |
| S100-nx4 | (4,48,288) | 100 | 40 | \(a_y=0.01\ {\rm m/s^2}\) |
| S200-nx4 | (4,48,288) | 200 | 40 | \(a_y=0.01\ {\rm m/s^2}\) |
| S200-nx8 | (8,48,288) | 200 | 40 | \(a_y=0.01\ {\rm m/s^2}\) |

At the beginning of each run, every particle receives

\[
\mathbf F_p=m_p(0,0.01,0),
\]

not a particle-count-dependent uniform force. The force is then held fixed for
all \(40\times0.005=0.2\) s. Every macro step must complete its declared solid
substeps and full physical time. Each accepted row records Point A, the full
particle displacement and velocity norms by axis, fixed-root drift, out-of-bounds
count, deformation-clamp count, and applied-force sums. The root tolerance is
\(10^{-8}\) m. S100-nx4 versus S200-nx4 and S200-nx4 versus S200-nx8 must each
satisfy the relative Point-A vector criterion below at 0.2 s.

The fixed-fluid matrix contains fresh nx4 and nx8 runs. Each uses a fixed
cylinder and a fixed, zero-velocity beam for 500 macro steps (2.5 s); no solid
advance is allowed. The frozen marker position, velocity, normal, area, region,
and ordering must remain identical after every step. The boundary for physical
step \(n\) is imposed at \(t_n=n\Delta t\). Only completed steps 401--500
(2.005--2.5 s), after the 2 s inlet ramp, enter inlet, flux, no-slip, force,
and quasi-2D aggregate checks.

After both constituent matrices pass, launch one fresh one-step coupled preflight
and one separate fresh two-step coupled preflight through the production generic
FSI path. The two-step run starts at \(t=0\); it must not resume or copy the
one-step state. These runs check launch, transaction, accepted-time, and artifact
contracts only.

#### 5.4.4 Formal S0 evidence path and source invalidation (2026-09-03)

Clean local commit `db22bb1` adds the formal FSI1-S0 campaign path. It binds the
complete effective case configuration, fail-fast mechanism probe, requested and
measured strict-CUDA runtime identity, immutable marker-layout identity, clean
Git commit, and recursive hashes of every production Python file under
`benchmarks/`, `cases/`, `simulation_core/`, `src/`, and `tools/`.

The accepted observer publishes only after the physical step commits. Rejected
coupling trials never enter accepted arrays, while their fluid solves, solid
macro solves, MPM substeps, pressure-CG iterations, and exact pressure-operator
applications remain in the accepted step's work ledger. Beam-marker force,
cylinder pressure force, and cylinder viscous force are all copied at the same
final accepted trial's pre-solid-load stage. IQN rank, condition number, update
mode, fallback reason, update limiting, and fallback count are retained with
explicit padding semantics.

Accepted evidence is create-only and written in fixed chunks of 1000 steps,
plus one shorter final chunk. NPZ publication precedes its manifest. Every
manifest binds the source/config identity, array dtype/shape/hash, units,
accepted step/time range, marker identity, runtime identity, and from-start
lineage. Failure artifacts use the registered
`BLOCKED_SOURCE_MISMATCH`, `BLOCKED_ENVIRONMENT`, or
`FAIL_NUMERICAL_HEALTH` classification. Nonfinite mechanism-probe candidates
are encoded as explicit `nan`, `positive_infinity`, or `negative_infinity`
diagnostics and remain strict JSON; rollback success is reported only after the
generic solver has actually restored the physical state.

The reviewed implementation passed 199 focused non-CUDA tests plus 30 subtests.
Three focused CUDA pressure-work checks also passed: legacy Jacobi clears stale
CG counters, a positive-iteration FV-CG solve satisfies exact
`matvec = iterations + 2`, and zero-RHS FV-CG reports zero iterations with one
exact-confirmation matvec. Compilation and `git diff --check` passed, and two
independent read-only reviews reported no remaining P0--P3 finding. These are
implementation and focused-regression results, not a full-suite result and not
an FSI1 numerical result.

Because `db22bb1` changes production source after `7b80a6b`, every component
artifact listed in Section 5.4.3 is now historical and source-stale. The entire
solid, fixed-fluid, nx4/nx8 comparison, and independent one-/two-step coupled
preflight chain must be regenerated from the final clean HEAD before S0 may
start. The formal S0 command is:

~~~text
python3 -m tools.validation.run_turek_hron_fsi_campaign \
  --output-root validation_runs/turek_hron_fsi123_r26a \
  --label <new-unique-label>
~~~

The runner has no chunk-size override and always starts from zero with marker
re-seeding disabled. The first formal attempt later failed closed at candidate
step 4 and is retained only as accepted-prefix failure evidence; it is not an
S0 pass. See Section 5.4.5 for the final component requalification boundary.

#### 5.4.5 Collective F-space repair and final component pass (2026-09-04)

At clean commit `ef1b8fe`, the regenerated solid-only stages passed, but nx4
fixed-fluid failed closed because collective closure reported no certified
hard-target repair. The captured zero-correction residual was
`3.944443960790522e-6 m/s`, below the frozen public absolute tolerance
`1e-4 m/s`. The old cyclic Kaczmarz path worsened it to
`2.238149609e-4 m/s` after 21,504 sweeps, while a compact f32-audited
least-squares witness reached about `3.92419578e-6 m/s`. The active matrix had
rank 10 with a clean singular-value gap.

Commit `8fb44de` fixes the class rather than the captured face: it accepts an
already-compatible identity correction before solving, otherwise constructs a
bounded, isolated, per-axis F-space least-squares sufficient witness. The f64
candidate is cast to f32 and accepted only by the device all-active-row audit.
No public tolerance changed, no target is averaged, and no soft fallback or
geometry allowlist was added. Nonfinite data and witness invariant failures
remain fail-closed, and all collective scratch is retired without touching
terminal-Q or pressure-nullspace committed state.

Fourteen directly affected strict-CUDA contracts passed in one process in
`39.749 s`. Python compilation, structure validation, `git diff --check`, and
a fresh independent read-only review also passed. The complete source-matched
component protocol then passed at clean `8fb44de`, with every artifact bound to
source SHA256
`4c88aa85bb2db42d75907ae794da7d9d4028b6c900d124c5191fc838fb878877`:

- `turek_hron__component__solid_s100_nx4__8fb44de__r01`;
- `turek_hron__component__solid_s200_nx4__8fb44de__r01`;
- `turek_hron__component__solid_s100_s200_nx4__8fb44de__r01`, Point-A
  relative-vector delta `0.0003264915974131584`;
- `turek_hron__component__solid_s200_nx8__8fb44de__r01`;
- `turek_hron__component__solid_s200_nx4_nx8__8fb44de__r01`, Point-A
  relative-vector delta `0.0`;
- `turek_hron__component__fixed_fluid_nx4__8fb44de__r01` and
  `turek_hron__component__fixed_fluid_nx8__8fb44de__r01`, both 500 rows;
- `turek_hron__component__fixed_fluid_nx4_nx8__8fb44de__r01`, force-per-span
  relative-vector delta `0.006372355169562249`; and
- independent `turek_hron__component__coupled_preflight_step1_nx4__8fb44de__r01`
  and `turek_hron__component__coupled_preflight_step2_nx4__8fb44de__r01` runs,
  both `PASS_SMOKE_ONLY`, with exact fluid/solid macro-time accounting.

This is `PASS_COMPONENT_ONLY` plus the preregistered smoke prerequisite. It
authorizes only a fresh 1600-step FSI1-S0 strict-CUDA run from zero. There is
still no `PASS_FSI1_S0_GATE_ONLY`, no FSI1 pass, and no authorization for M0/M1
or later cases.

#### 5.4.6 Certificate-connected weighted F/H repair and requalification (2026-09-05)

Formal run `turek_hron__fsi1_s0__0025e12__r01` started from zero at clean
commit `0025e12`, accepted steps 1--7 through `t=0.035 s`, and failed closed
while preparing candidate step 8 with three hard-target certificates because
marker compatibility closure did not converge. The accepted prefix is retained
as failure evidence only. It is not an S0 pass, and no complete transition
checkpoint exists from which a formal run may resume.

The captured candidate contained 336 active marker-axis rows. F-only closure
remained just above the frozen `1e-4 m/s` absolute limit, while a joint F/H
solution was feasible. The repair therefore fixes the solver class rather than
widening a tolerance or admitting another topology. Clean commit `5afba27`
forms certificate-connected row components, exposes H columns only inside
those components, includes every active F row in the joint system, determines
structural rank before mobility weighting, and solves the inverse-mass
minimum-energy problem. The f64 result is materialized as f32 and accepted only
after device audits pass both the authorized repair-row tolerance and the
global all-active-row tolerance. Failure remains atomic and fail-closed.

Device report schema 6 identifies the applied backend and records certificate
count, repair/global residuals, hard-target DOF count, and maximum H delta. The
strict runner checks field types, finiteness, bounds, and cross-consistency. An
exact captured-step diagnostic replay closed with repair/global residual maxima
`2.27050833246e-7/8.83974644239e-5 m/s`, 70 hard-target DOFs, and maximum H
delta `1.05458639155e-4 m/s`; it remains diagnostic-only evidence.

Focused verification passed the RED/GREEN inverse-mass weighting and
mobility-rank contract, four F/H strict-CUDA tests (`93.276 s`), one real
hybrid integration (`177.698 s`), all 18 collective strict-CUDA tests
(`862.117 s`), 65 host/static report tests (`2.275 s`), compilation, structure
validation, and `git diff --check`. A fresh independent review returned
`ship` with no P0/P1 finding.

The complete frozen component chain was regenerated at clean `5afba27`, and
all ten artifacts are bound to source SHA256
`d1deddc51e16b3a862f25318d4bef75a773bd202d9d2e3e596615df5fe67a492`:

- `turek_hron__component__solid_s100_nx4__5afba27__r01`;
- `turek_hron__component__solid_s200_nx4__5afba27__r01`;
- `turek_hron__component__solid_s100_s200_nx4__5afba27__r01`, Point-A
  relative-vector delta `0.0003372753055382118`;
- `turek_hron__component__solid_s200_nx8__5afba27__r01`;
- `turek_hron__component__solid_s200_nx4_nx8__5afba27__r01`, Point-A
  relative-vector delta `1.0950965526574071e-5`;
- `turek_hron__component__fixed_fluid_nx4__5afba27__r01` and
  `turek_hron__component__fixed_fluid_nx8__5afba27__r01`, both 500 rows;
- `turek_hron__component__fixed_fluid_nx4_nx8__5afba27__r01`, force-per-span
  relative-vector delta `0.006372354632268758`; and
- independent `turek_hron__component__coupled_preflight_step1_nx4__5afba27__r01`
  and `turek_hron__component__coupled_preflight_step2_nx4__5afba27__r01` runs,
  both `PASS_SMOKE_ONLY`, with final accepted times `0.005 s` and `0.010 s`.

This supersedes the `8fb44de` prerequisite. It authorizes only a fresh
1600-step FSI1-S0 strict-CUDA campaign from zero. No S0 pass or authorization
for M0/M1 and later cases exists yet.

#### 5.4.7 Minimax F-only repair, host identity, and requalification (2026-09-05)

Formal run `turek_hron__fsi1_s0__27111d3__r01` started from zero at clean
`27111d3`, accepted steps 1--7 through `t=0.035 s`, and failed closed
while preparing candidate step 8 with `certificate_count=0`. The
zero-correction residual was `1.0104837565449998e-4 m/s`. The existing
minimum-L2 F-only candidate reached a device maximum residual of
`1.0004986688727513e-4 m/s`, just above the frozen `1e-4 m/s` limit. The
accepted prefix remains failure evidence only; it is not an S0 pass and lacks
a complete transition checkpoint for formal resume.

The same captured 336-row system has a feasible minimax F-only witness:
`9.82453917360385e-5 m/s` in f64 and
`9.824539301916957e-5 m/s` in the production f32 device audit.
`certificate_count=0` was therefore correct; the false negative belonged
only to the old L2-only F-feasibility decision. This was not a reason to
change the public tolerance,
admit geometry, or authorize H. Clean `d3044d9` retains least squares as the
fast path and invokes a column-normalized minimax LP only after that
candidate's f32 device audit fails. The LP output cannot commit state directly;
the unchanged all-row device audit is the sole acceptance boundary, and every
failure path remains atomic and fail-closed.

Verification passed 14 targeted collective strict-CUDA contracts in one
process (`758.463 s`), 66 focused host/static tests, Python compilation,
Ruff, `git diff --check`, and two fresh independent read-only reviews. This
does not claim a verdict for the interrupted broad component-face module run.

Clean `293dc69` also makes the numerical host part of formal provenance:
CPython `3.10.12`, NumPy `2.1.2`, SciPy `1.15.3`, identity SHA256
`db88ab4094ab1be43ad58b7c18cc59c756a45e1ac4f44978bc6238a4738c6a47`.
Component `run_manifest.json` files use schema 2, future formal
accepted-chunk manifests use `schema_version: 2`, and the host-identity
payload remains schema 1. `requirements.txt` is source-tracked, and host
mismatch fails before solver or Taichi initialization.

The complete ten-stage source-matched chain passed at clean `293dc69`, with
every manifest bound to source SHA256
`f186fa55278ef55a82f8ae2f740defad0766f3350450355bf6c1710fa279d7f3`
and the host identity above:

- `solid_s100_nx4`, `solid_s200_nx4`, and `solid_s200_nx8` each
  completed 40 rows;
- solid S100/S200 nx4 and S200 nx4/nx8 Point-A relative-vector deltas were
  `0.0003264915974131584` and `0.0`;
- `fixed_fluid_nx4` and `fixed_fluid_nx8` each completed 500 rows, with
  velocity/force span-leakage pairs
  `0.0005504236652169761/0.0006353445200368427` and
  `0.00043036809704860384/0.0003521293000030293`;
- their force-per-span relative-vector delta was
  `0.006372353579412674 < 0.02`; and
- independent one- and two-step coupled preflights passed as
  `PASS_SMOKE_ONLY`, with exact final accepted times `0.005 s` and
  `0.010 s` and zero unadvanced fluid/solid time.

All ten artifact labels end in `__293dc69__r01`. This supersedes
`5afba27` and authorizes only a fresh 1600-step FSI1-S0 strict-CUDA campaign
from zero after this documentation-only record is committed. Its label must
contain that clean commit's short SHA and must not reuse the `27111d3`
prefix. No S0 pass or authorization for M0/M1 and later cases exists yet.

#### 5.4.8 Rank-direct max-residual repair and requalification (2026-09-05)

Formal run `turek_hron__fsi1_s0__f513f9b__r01` started from zero at clean
`f513f9b`, accepted steps 1--7 through `t=0.035 s`, and failed closed while
preparing candidate step 8 as `FAIL_NUMERICAL_HEALTH`. The
rank-revealing-direct minimum-L2 candidate's device audit returned
`0.00010005016520153731 m/s` against the unchanged `1e-4 m/s` limit. The
accepted prefix is failure evidence only, is not an S0 pass, and has no
complete transition checkpoint for formal resume.

A diagnostic-only replay from zero at that source isolated physical step 8,
coupling trial 7, Q solve 5. Its 336 active rows contained 30 positive rows and
a 17-column selected basis. The L2 correction had an f32 all-active-row
maximum residual of `0.00010005030344473198 m/s`; the same selected basis had
an f32 device-audited minimax witness of `9.82458223006688e-5 m/s`. The replay
re-threw the original error and committed no physical velocity. This was an
objective mismatch, not evidence for a topology or tolerance change.

Clean `f2320f5` retains least squares as the fast path and invokes the
column-normalized selected-basis Chebyshev solve only after that candidate
fails the existing f32 all-active-row device audit. The correction is rounded
to f32 and must pass that unchanged audit before commit. Backend failure,
nonfinite output, and true minimax infeasibility remain atomic and fail-closed;
no public tolerance, topology admission, or soft fallback changed.

The complete seven-test CPU module passed in `16.169 s`; five focused
strict-CUDA contracts passed in one process in `184.138 s`, covering the L2
fast path, minimax success, backend-exception atomicity, true infeasibility,
and full collective/scratch behavior. Python compilation, Ruff check,
`git diff --check`, and a fresh final read-only review (`SHIP`, no P0--P3
finding) also passed. This is not a broad module or full-suite verdict.

The complete ten-stage source- and host-matched component chain then passed at
clean `f2320f5`. A post-run verifier revalidated file and array hashes, source,
host and Taichi identities, parent identities, row counts, and time ledgers.
Every manifest is bound to source SHA256
`478eb7707fd20761654443b334782a252ffee90dc29ba2c9b9707ba747b67c89`
and host identity
`db88ab4094ab1be43ad58b7c18cc59c756a45e1ac4f44978bc6238a4738c6a47`:

- `solid_s100_nx4`, `solid_s200_nx4`, and `solid_s200_nx8` each completed
  40 rows;
- solid S100/S200 nx4 and S200 nx4/nx8 Point-A relative-vector deltas were
  `0.0003264915974131584` and `0.0`;
- fixed-fluid nx4 and nx8 each completed 500 rows, with velocity/force
  span-leakage pairs `0.0005504236378906395/0.0006353412795713001` and
  `0.00043036809436382663/0.0003521303648396529`, and drag means
  `12.845664033171927` and `12.927745742812588 N/m`;
- their force-per-span relative-vector delta was
  `0.006372346408124445 < 0.02`; and
- independent one- and two-step coupled preflights passed as
  `PASS_SMOKE_ONLY`, used `[10]` and `[10, 10]` coupling iterations, ended at
  exactly `0.005 s` and `0.010 s`, and left zero unadvanced fluid/solid time.

All ten labels end in `__f2320f5__r01` under
`validation_runs/turek_hron_component_gates/`. This supersedes `293dc69` and
authorizes only a new documentation commit followed by one fresh 1600-step
FSI1-S0 strict-CUDA campaign from zero whose label contains that documentation
commit's short SHA. It is not `PASS_FSI1_S0_GATE_ONLY` and does not authorize
M0/M1, FSI2, FSI3, Oracle, or learning.

#### 5.4.9 Formal absolute-coupling gate binding (2026-09-05)

Formal run `turek_hron__fsi1_s0__75c240a__r01` started from zero at clean
`75c240a`, accepted steps 1--7 through `t=0.035 s`, and failed closed at step
8 as `FAIL_NUMERICAL_HEALTH`. It exhausted all 16 coupling trials with final
relative residual `0.2993190969189178` and final absolute RMS residual
`7.839528620993855e-6 m/s`. The best observed relative/absolute pair was
`0.022642592094207343 / 4.6934343661611526e-7 m/s`. This accepted prefix is
failure evidence only. It has no transition checkpoint; accepted-interface
arrays and CSV history are not restart state, so it cannot be formally resumed
or reclassified as an S0 pass.

The generic solver correctly evaluates relative convergence or an enabled
absolute gate. The formal S0 spec, however, omitted
`fsi_coupling_absolute_tolerance_mps` and inherited the case default `0.0`,
which disabled that gate. At the final trial the candidate RMS speed was only
about `2.6191e-5 m/s`, making the pure-relative ratio ill-conditioned during
the early two-second inlet ramp. The offline FSI1 acceptance contract already
requires every accepted step's absolute coupling RMS to be no greater than
`1e-4 m/s`; the failure therefore exposed a formal-runner configuration
omission, not a reason to relax the preregistered acceptance contract.

Clean `33b3db0` adds only `fsi_coupling_absolute_tolerance_mps = 1e-4 m/s` to
the frozen `FSI1_S0_SPEC` and its exact-matrix test. It leaves the generic
solver, case default, offline acceptance, topology, and physical model
unchanged. The targeted test demonstrated RED on the missing key and GREEN
after the fix; all 31 formal-campaign focused tests passed in `1.16 s`.
Compilation, Ruff, `git diff --check`, and a final read-only review (`SHIP`, no
P0--P3 finding) also passed. This is a focused configuration-contract result,
not a CUDA or full-suite verdict.

Neither changed file belongs to the component gate's explicit 15-file source
identity. Its SHA256 remains
`478eb7707fd20761654443b334782a252ffee90dc29ba2c9b9707ba747b67c89`,
so the `f2320f5` ten-stage chain remains current and nx4/nx8 must not be rerun.
The formal all-Python source identity does change. This authorizes only a clean
documentation commit and one fresh 1600-step strict-CUDA S0 from zero under a
new label containing that documentation commit's short SHA. No resume, S0
pass, M0/M1, FSI2, FSI3, Oracle, or learning authorization exists yet.

### 5.5 Frozen formulas, tolerances, and evidence labels

For nonzero finer/reference vector \(\mathbf b\), define

\[
\delta(\mathbf a,\mathbf b)=
\frac{\lVert\mathbf a-\mathbf b\rVert_2}{\lVert\mathbf b\rVert_2}.
\]

A zero or nonfinite denominator fails closed. The two compared observable vectors
are Point A \((-\Delta z,\Delta y)\) and force per span \((D,L)\). Both the
100/200-solid-substep Point-A comparison and each nx4/nx8 constituent comparison
must satisfy \(\delta<0.02\). The solid comparison owns the Point-A check; the
fixed-fluid comparison owns the force-per-span check.

For a vector field or time series \(\mathbf a=(a_x,a_y,a_z)\), define spanwise
leakage

\[
R_x(\mathbf a)=
\frac{\lVert a_x\rVert_2}
{\sqrt{\lVert a_y\rVert_2^2+\lVert a_z\rVert_2^2}}.
\]

The denominator must be nonzero and finite. Concatenate the declared completed
window before taking each norm. Apply this formula separately to all solid
particles' displacement and velocity, all active fluid cells' velocity, and raw
reported force; each must satisfy \(R_x\le10^{-3}\).

For each fully ramped fixed-fluid row, let

\[
U_{\rm in}=\frac{|Q_{\rm in}|}{H s},\qquad
e_U=\frac{|U_{\rm in}-\bar U|}{|\bar U|}.
\]

Require \(\max e_U<0.005\), positive-magnitude inlet and outlet flux in the
documented -z flow direction, and the time-integrated mass imbalance

\[
E_Q=
\frac{\sum_n |Q_{{\rm out},n}-Q_{{\rm in},n}|\Delta t_n}
{\sum_n \max(|Q_{{\rm in},n}|,|Q_{{\rm out},n}|)\Delta t_n}<0.01.
\]

Let \(\tau_{32}=32\epsilon_{32}\max(1,|\bar U|)\) m/s. Both external y-wall
device-ledger planes must have full-component mask 7. Against the independently
frozen zero-wall value, their boundary-ledger residual
\(\max|\mathbf u_{\rm face,ledger}-\mathbf u_{\rm face,expected}|\), the
base-cylinder obstacle-cell velocity, and normal velocity on every base-cylinder
obstacle/fluid crossing face must each be at most \(\tau_{32}\). This first
quantity proves exact boundary registration, not a persistent RK2 face-state;
focused RK2 backtrace tests separately prove that all three registered
components are consumed. Fixed-beam marker no-slip uses the production marker
sampler, requires exactly the expected valid markers and zero invalid markers,
and must have finite RMS at most \(10^{-4}\) m/s and finite maximum residual at
most \(0.01|\bar U|=0.002\) m/s for FSI1.

The outlet check proves that the pressure operator used
`pressure_outlet_zmin = True`; finite pressure, valid outlet graph/topology,
converged component labels, zero CG breakdown, and converged pressure solves are
required. On nonzero-RHS evaluation rows, requested and effective preconditioner
must both be `fv_multigrid` with zero fallback. The check must not assert that a
cell-centred `pressure[:,:,0]` slice is identically zero.

Raw solver-axis beam, cylinder-pressure, and cylinder-viscous force vectors are
persisted before conversion. Drag maps from solver \(-z\), lift from solver
\(+y\), and span normalization occurs exactly once. Force closure requires

\[
\mathbf F_{\rm sum}=\mathbf F_{\rm beam}+\mathbf F_{\rm cyl,p}
+\mathbf F_{\rm cyl,v},\qquad
\frac{\lVert\mathbf F_{\rm total}-\mathbf F_{\rm sum}\rVert_2}
{\lVert\mathbf F_{\rm sum}\rVert_2}\le32\epsilon_{64}.
\]

If the total-force norm is zero, exact zero closure is required instead. Every
component must be finite. A physical-time comparison uses
`max(1e-15, 1e-12*dt_s)` as its absolute tolerance and also requires exact
declared/observed substep counts.

Every command claims a new run directory with create-if-absent semantics before
constructing its runtime. The runner accepts a runtime class, not an instance or
callable factory, and the manifest records commit and dirty state, the complete
normalized effective case configuration, executable source paths and per-source
hashes, a combined source hash, configuration hash, marker-layout hash when
applicable, artifact byte hashes, and per-array hashes. Constituent and coupled
artifacts also duplicate one measured Taichi identity in the summary and
manifest: requested and actual architecture, default floating-point and integer
types, random seed, strict-architecture verification, compiler controls, Taichi
version, and offline-cache identity. Offline-cache location/state remains
auditable but is excluded from numerical-equivalence comparison; all numerical
runtime fields must match exactly. A comparison command does not initialize
Taichi, so its two identity fields are strictly `not-applicable` and still
hash-checked. The canonical array hash is SHA256 over the dtype string,
canonical shape, and contiguous C-order bytes. Histories contain only completed
component steps or accepted coupled steps; a failed or rejected state is never
appended as accepted.

Successful constituent and comparison commands are
**PASS_COMPONENT_ONLY**. Successful coupled preflights are
**PASS_SMOKE_ONLY**. Neither label is a numerical FSI1 pass, and constituent
nx4/nx8 agreement cannot substitute for a coupled discretization study.

## 6. FSI1 steady validation

### 6.1 Frozen grids

| level | grid | approximate cells through beam thickness | role |
| --- | --- | ---: | --- |
| L0 | (4,48,288) | 2.3 | smoke/baseline |
| L1 | (4,96,576) | 4.7 | main validation |
| L2 | (4,144,864) | 7.0 | convergence confirmation |

Refinement preserves \(dy\approx dz\). Marker counts are automatic and
geometry-derived. R26A freezes **ib_anisotropic_envelope = True**,
**classify_far_internal_nodes = True**, and
**flow_cg_preconditioner = "fv_multigrid"** before any reference comparison.
The Section 5.4 initialization audit must pass without changing those controls.

### 6.2 Run matrix

| run | grid | dt | planned end | role |
| --- | --- | ---: | ---: | --- |
| FSI1-S0 | L0 | 0.005 s | 8 s / 1600 steps | low-resolution steady gate |
| FSI1-M0 | L1 | 0.005 s | 8 s / 1600 steps | main spatial result |
| FSI1-M1 | L1 | 0.0025 s | 8 s / 3200 steps | time-step confirmation |
| FSI1-F0 | L2 | 0.0025 s | 8 s / 3200 steps | conditional fine result |

One- or two-step preflights may check launch and artifact contracts but have no
numerical-pass status. FSI1-F0 is forbidden until S0, M0, and M1 are complete
with zero numerical-contract violations and the exploratory formula below passes
for all four primary metrics.

Before any L1 or L2 coupled run, the case configuration must also validate the
configured solid substep against
`NeoHookeanMaterial.stable_explicit_dt_s`, using the minimum active solid-grid
spacing. A 2026-09-03 static audit estimated the current FSI1 L0 limit as about
`5.46e-5 s` versus the configured `5e-5 s`, so S0 remains inside the
declared bound. At L1 the estimated limit is about `2.73e-5 s`; reusing 100
solid substeps with `dt=0.005 s` would exceed it by about 1.83 times and must
fail configuration validation. This safety gate is fixed before M0 and may not
be relaxed in response to the numerical result.

FSI1 formal runs are uninterrupted from step 1. The 2 s ramp is followed by 2 s
settling. Adjacent windows \(t=4\)–6 s and \(t=6\)–8 s are assessed.

For each FSI1 primary metric \(q\), let \(r_q\) be its Section 3.3 anchor. In each
window compute the arithmetic mean, p05, p95, and least-squares slope. Steady state
requires all of the following:

\[
\frac{|\bar q_{6:8}-\bar q_{4:6}|}{|r_q|}<0.005,
\]

\[
\frac{q^{95}_{6:8}-q^{05}_{6:8}}{|r_q|}<0.005,
\qquad
\frac{|s_{6:8}|(2\ {\rm s})}{|r_q|}<0.005.
\]

The last expression is the preregistered no-persistent-drift rule. In addition,
the run must contain the exact expected rows and schema with no coupling,
pressure, flux, fixed-root, scatter/action-reaction, fluid-time, solid-time, MPM,
finite-value, or accepted-step violation.

### 6.3 FSI1 pass levels

For each metric use the late-window mean and define
\(e_q=|\bar q-r_q|/|r_q|\) and
\(\delta_q(a,b)=|\bar q_a-\bar q_b|/|r_q|\).

Exploratory pass requires:

- S0, M0, and M1 are complete, numerically healthy, and steady;
- both M0 and M1 Point A displacement errors are below 10%;
- both M0 and M1 total-drag and total-lift errors are below 15%;
- \(e_q(\mathrm{M0}) < e_q(\mathrm{S0})\) for every primary metric;
- \(\delta_q(\mathrm{M1},\mathrm{M0})<0.05\) for every primary metric.

Benchmark-quality pass additionally requires:

- F0 is complete, numerically healthy, and steady;
- F0 Point A displacement errors are below 5%;
- F0 total-drag and total-lift errors are below 10%;
- \(\delta_q(\mathrm{F0},\mathrm{M1})<0.03\) for Point A displacement;
- \(\delta_q(\mathrm{F0},\mathrm{M1})<0.05\) for total drag and lift.

These are local research gates, not official benchmark rules.

FSI1 failure blocks FSI2 and all model work.

## 7. FSI2 large-deformation validation

FSI2 starts only after FSI1 benchmark-quality pass.

1. L0, dt 0.001 s, \(t=0\)–3 s: health-only short run.
2. L1, dt 0.001 s: continue to the limit-cycle gate, with a hard end at
   \(t=35\) s.
3. L1, dt 0.0005 s: temporal confirmation to the same limit-cycle gate.
4. L2, dt 0.001 s: spatial confirmation to the same limit-cycle gate.

Items 3 and 4 are both fixed before observing the L1 error and are both required
for benchmark-quality status. They begin only after item 2 is complete, healthy,
stable, and within the exploratory reference thresholds.

The canonical baseline has
**marker_reseed_interval_steps = None**. Record marker-spacing ratio and
coverage. Reseeding is a later isolated A/B only if the baseline shows geometric
degradation; it cannot be mixed into the first physical validation.

Report the last three stable cycles:

- Point A \(u_x,u_y\) midrange, amplitude, and frequency;
- total drag/lift midrange, amplitude, and frequency;
- \(u_y\)-velocity phase portrait;
- coupling trials and rejected trials;
- first/second residual and full convergence state;
- pressure CG/matvec work;
- marker-spacing max/min ratio and coverage;
- fixed-root drift and maximum deformation.

Exploratory pass requires the L1/dt 0.001 run to be complete and numerically
healthy, to pass the limit-cycle stability gate, to have \(e_q<0.10\) for Point A
\(u_y\) amplitude/frequency, and to have \(e_q<0.15\) for total-drag midrange
and total-lift amplitude.

Benchmark-quality pass requires all three L1/L1-time/L2 runs to be complete,
healthy, and limit-cycle stable; every run must have \(e_q<0.05\) for the two
Point A quantities and \(e_q<0.10\) for the two force quantities. Both the
temporal pair and spatial pair must satisfy \(\delta_q<0.03\) for Point A
amplitude/frequency and \(\delta_q<0.05\) for force midrange/amplitude.

FSI2 failure blocks FSI3, Oracle, and model work.

## 8. FSI3 strong-coupling validation

FSI3 starts only after FSI2 benchmark-quality pass and follows a frozen staged
pattern:

1. L0, dt 0.001 s, \(t=0\)–3 s health run;
2. L1, dt 0.001 s, to a stable limit cycle no later than \(t=35\) s;
3. L1, dt 0.0005 s, temporal confirmation;
4. L2, dt 0.001 s, spatial confirmation.

The canonical FSI3 baseline and both discretization confirmations explicitly use
**marker_reseed_interval_steps = None**. Any later reseed experiment is a separate
diagnostic and cannot replace these runs.

In addition to the FSI2 outputs, record and compare:

- coupling-trial distribution versus FSI2;
- first-residual growth by cycle phase;
- relaxation limiting;
- IQN rank, condition number, fallback, and update limiting;
- pressure CG and interface residual by cycle phase;
- results with dimensionless **fsi_coupling_tolerance** changed from \(10^{-3}\)
  to \(10^{-4}\).

The tolerance comparison keeps
**fsi_coupling_absolute_tolerance_mps = 0.0 m/s** and holds grid, dt, source,
initial state, marker layout, IQN settings, and all physical parameters fixed.
It must satisfy \(\delta_q<0.03\) for Point A amplitude/frequency and
\(\delta_q<0.05\) for force midrange/amplitude.

FSI3 applies the same exact four-metric formulas and exploratory/benchmark-quality
thresholds as FSI2, using the FSI3 anchors in Section 3.4. Both temporal and
spatial confirmations are required. Failure blocks Oracle and model work.

## 9. Accepted-only interface evidence

Every valid R26A run records accepted state while it is already paying the FSI
cost:

- marker reference/current position and material displacement;
- marker velocity, normal, fixed area, region ID, and ordering;
- Point A displacement and velocity;
- total, beam, cylinder-pressure, and cylinder-viscous forces;
- coupling trials, rejected trials, residuals, and IQN diagnostics;
- pressure CG/matvec, fluid solves, solid solves, and MPM substeps.

Write immutable chunks of 1000 accepted steps, with a shorter final chunk:

~~~text
accepted_interface_000000_000999.npz
accepted_interface_001000_001999.npz
~~~

Each chunk manifest binds source commit/dirtiness, executable-source hashes,
complete configuration, case/grid/dt, marker-layout hash, first/last accepted
step and time, units, dtype, shapes, NPZ hash, per-array hashes, and parent
checkpoint lineage.

Rejected trials never enter accepted arrays or advance accepted time. Predictor
or learning state is absent in R26A. Large chunks remain local and are not
committed.

## 10. Time, transaction, restart, and rollback invariants

For every accepted macro step:

\[
\sum \Delta t_{\mathrm{fluid,accepted}}=\Delta t_s,\qquad
\sum \Delta t_{\mathrm{solid,accepted}}=\Delta t_s.
\]

Residual convergence may stop same-time algebraic iterations. It may not truncate
fluid or solid physical-time advancement.

Every rejected coupling trial restores the complete pre-trial state. Failed
steps publish fail-closed diagnostics and leave the last accepted state intact.
No rejected state may contaminate IQN history, force history, checkpoints, or
accepted-interface chunks.

FSI1 formal evidence is uninterrupted. FSI2/FSI3 may use checkpoints only after
a focused resume-equivalence contract proves that the checkpoint contains the
complete physical, coupling, geometry, marker, and history state. Reduced
**step_fields** files are never restart states.

Every run uses a new output directory. Failed output is never overwritten or
continued as canonical evidence. Only one expensive CUDA process may run at a
time.

Source, configuration, geometry, material, force convention, marker layout, and
checkpoint lineage must match within a comparison. A source change invalidates
the old executable identity; old artifacts are never re-signed.

## 11. Status vocabulary and reporting

Allowed classifications include:

- **PASS_CONTRACT_ONLY**
- **PASS_COMPONENT_ONLY**
- **PASS_SMOKE_ONLY**
- **PASS_FSI1_S0_GATE_ONLY**
- **PASS_EXPLORATORY**
- **PASS_BENCHMARK_QUALITY**
- **FAIL_FSI1_S0_GATE**
- **FAIL_REFERENCE_CONTRACT**
- **FAIL_NUMERICAL_HEALTH**
- **FAIL_STEADY_STATE**
- **FAIL_LIMIT_CYCLE**
- **FAIL_REFERENCE_ERROR**
- **FAIL_DISCRETIZATION**
- **BLOCKED_ENVIRONMENT**
- **BLOCKED_SOURCE_MISMATCH**

A contract or unit test never upgrades a numerical run. A short run never
upgrades to exploratory pass. A historical result never upgrades to
source-matched R26A evidence. Wall time is diagnostic and is never called
acceleration.

## 12. Required RED-to-GREEN checks

Before numerical execution, add focused tests for:

- complete immutable Featflow FSI1/2/3 contracts;
- LS-DYNA source separation and no fallback;
- exact raw manifest/hash/bytes/rows/12-column validation;
- malformed, nonfinite, wrong-time, wrong-unit, wrong-force-scope, and unknown
  manifest rejection;
- immutable imported series;
- whole-body raw force columns 5+7 and 6+8;
- rising-crossing interpolation, exact-zero plateau handling, partial-tail
  rejection, and insufficient-cycle failure;
- extrema-based midrange/amplitude rather than arithmetic mean/RMS;
- last-three-cycle stability and deterministic FFT frequency;
- compatibility of current FSI1 acceptance API;
- exact beam+cylinder force arithmetic;
- quasi-2D leakage and span normalization;
- fluid/solid constituent forwarding and full-\(\Delta t_s\) accounting.

Use the smallest focused test first. No CUDA run is authorized merely because
these tests pass.

## 13. Downstream authorization boundary

R26B begins only after FSI1, FSI2, and FSI3 all reach
**PASS_BENCHMARK_QUALITY**.

Its preregistered no-commit arms are:

| arm | position initial guess | velocity initial guess |
| --- | --- | --- |
| C0 | carry | carry |
| O-v | carry | same-step Oracle |
| O-x | same-step Oracle | carry |
| O-xv | same-step Oracle | same-step Oracle |

FSI1 uses one ramp and one steady target. FSI2/FSI3 use four stable-cycle phases:
positive displacement peak, negative displacement peak, positive velocity peak,
and negative velocity peak.

Learning is authorized only if one fixed Oracle arm meets one of these paired,
source-matched aggregate gates across the complete preregistered target set:

- it reduces at least one coupling trial at two or more distinct phases, never
  increases coupling trials at another phase, and does not increase aggregate
  pressure-CG work; or
- it reduces aggregate pressure-CG work by at least 15%, never increases
  coupling trials at any phase, and does not increase aggregate coupling trials.

Every target must converge, match the requested initial state, roll back exactly,
leave accepted time unchanged, and preserve the full fluid/solid physical-time
contract. An improvement from one arm cannot be combined with another arm, and
an isolated gain cannot offset a regression elsewhere.

Only then may R26C compare the minimal C0/AR/K/G/GK/GDelta/Q matrix. Periodic
data splits are by complete cycles, never random time steps. Offline error is
diagnostic; the final claim remains no-commit live coupling/CG/matvec work.

## 14. Immediate execution order

1. R25B import-boundary repair — complete at `b7f32c0`; no CUDA rerun.
2. Freeze the vertical-flap predictor route — complete.
3. Create the R26A branch and this goal — complete.
4. Implement and test canonical multi-source references — complete.
5. Import and manifest the two official raw series — complete.
6. Implement and test the deterministic limit-cycle analyzer — complete.
7. Close the generic-core marker-Q/pressure-nullspace wiring defect — complete
   at `7b80a6b`; the 512-constraint dense limit remains explicit.
8. Implement and review formal S0 provenance, accepted-only chunks, work
   accounting, rollback evidence, and offline acceptance — complete at
   `db22bb1`.
9. Record the first formal S0 attempt at `0df48f4`: it accepted three macro
   steps, then failed closed at candidate step 4 on a registered local-connector
   component-face conflict. Diagnose and repair that exact gate — complete at
   `7862472`; this is not an S0 pass.
10. Regenerate the entire frozen component chain at `8fb44de` — complete, then
    superseded by the next source change.
11. Record the formal `0025e12` S0 attempt: it accepted seven macro steps and
    failed closed at candidate step 8. Diagnose and repair the certificate-
    connected weighted F/H class — complete at `5afba27`; this is not an S0
    pass.
12. Regenerate the frozen component chain at `5afba27` — complete, then
    consumed by the `27111d3` formal attempt and superseded by later source
    changes.
13. Record formal attempt `27111d3`: seven accepted steps, then a candidate-
    step 8 F-only false negative with a correct zero-certificate result —
    complete; this is not an S0 pass.
14. Replace the L2-only sufficient decision with the device-audited max-norm
    fallback — complete at `d3044d9`.
15. Bind and fail closed on the pinned host-numerics identity — complete at
    `293dc69`.
16. Regenerate the entire ten-stage component chain at `293dc69` — complete;
    every stage is source- and host-matched and passed.
17. Record formal attempt `f513f9b`: seven accepted steps, then a candidate-
    step 8 rank-direct L2/max-residual mismatch — complete; this is
    `FAIL_NUMERICAL_HEALTH`, not an S0 pass.
18. Diagnose that exact rank-direct failure and implement the unchanged-audit
    minimax fallback — complete at `f2320f5`; no tolerance or topology change.
19. Regenerate the entire ten-stage component chain at `f2320f5` — complete;
    every stage is source-, host-, and strict-CUDA-matched and passed.
20. Record formal attempt `75c240a`: seven accepted steps, then step 8 exhausted
    all 16 coupling trials because the formal spec inherited the disabled
    absolute gate — complete; this is `FAIL_NUMERICAL_HEALTH`, not an S0 pass,
    and has no resumable checkpoint.
21. Bind the already-registered `1e-4 m/s` absolute coupling gate explicitly in
    the formal S0 spec — complete at `33b3db0`; focused tests and review passed,
    and the unchanged component source identity forbids an nx4/nx8 rerun.
22. The former next action at `33b3db0` was a new documentation commit and
    fresh 1600-step FSI1-S0 run. The current trace-repair stage supersedes that
    launch instruction: resolve the R15 rejection and perform the source-current
    coarse check before component/formal entry. A future formal run still needs
    matching qualified clean source and a new unused label; any commit or
    publication requires its own authorization.
23. Implement and validate a scalable rank-deficient Q backend and enforce the
    solid explicit-stability substep gate, then run M0/M1 and conditional F0.
    L1/L2 remain blocked until both prerequisites pass.
24. If and only if FSI1 passes, run FSI2.
25. If and only if FSI2 passes, run FSI3.
26. If and only if all three reach benchmark quality, close R26A and open the
    separately preregistered Oracle goal.

No later item may be started to avoid, dilute, or reinterpret an earlier failed
gate.
