# Turek-Hron R15 fallback validation, 2026-09-06

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

## Verified evidence and continuation

The source core SHA256 remains
`26f37eb14179b73c0c4c4a7e29be53954dcade4a377ed4c91310026b88e51751`.
GREEN25 r03 has no failures, errors, skips or expected failures. Zero and
single consumer controls retain all 15 comparison keys and all eight canonical
output arrays; both original nonempty C0 orders remain valid.
Process SHA256:
`5c706e671fd1320a3b405dc4a78f10d758e9eacf6644485a56cf79ff1c830874`.

| Frozen input | Native result | Owner elapsed seconds | Preservation |
| --- | --- | ---: | --- |
| R12 | PASS | 1429.131 | All eight A50 outputs byte-exact |
| R13 | PASS | 61.976 | All eight A50 outputs byte-exact |
| R14 | PASS | 65.676 | All eight A50 outputs byte-exact |
| R15 | PASS | 62.281 | Four former failure faces select owner 90/91 with query-matched geometry |

Each replay rechecked 207 original input arrays, the 987-file immutable input
inventory, source193, host/runtime and dependencies. R12/R13/R14 preserve the
original valid-cache subsets. R15's source inputs remain byte-exact; its
successful output is a corrected assembly, not a continuous FSI result.
Full replay manifest SHA256:
`c23236506c97ab4e503c51f596abf7c591f5a949defd10477fda9bce1c1d26bd`.

The preserved R16 manifest SHA256 is
`c2fe21f5084c9055d9427d30ff22bd3544dff7faba590e5d6de9034847cd1268`.
Its accepted-record stream is empty; both accepted fluid and solid time are
zero. Its 155-field precommit-failure observer was prepared but no precommit
failure occurred before the compilation interruption. The four replay cleanup
checks do not establish a fresh R16 after-cleanup capture or full rollback.
The normal terminal auditor was reviewed and prepared, but was not run against
this administrative interruption because its required normal terminal
artifacts do not exist. Its SHA256 is
`e29af2dced25cdcc4ffd1a864a3e2a3023de7ca8b523ab53dd9aaf286cefc003`.

All logs, captures, reviewed adapters, manifests, interrupted output, earlier
failed evidence and unapplied endpoint drafts are retained in the complete
archive attached to the WIP release. They must not be treated as restart states.
The release tag retains its historical source target; this documentation commit
and the source identity in the new archive identify the latest checkpoint.

1. On explicit continuation, verify WSL Ubuntu-22.04 user zhuohengli, worktree
   `/home/zhuohengli/worktrees/HIBM-MPM-r25b-live`, branch
   `codex/turek-hron-fsi123-validation-r26a`, current HEAD/status and interpreter
   `/home/zhuohengli/.venvs/hibm-mpm-r26a-py310/bin/python`.
2. Preserve the occupied R16 label. Prepare a new from-zero eight-step wrapper
   and manifest for a new label, keeping 4x96x400, 112 physical markers,
   dt_s=0.005, solid_substeps=200 and all original numerical thresholds.
   Bridge this documentation-only HEAD to the tested source by all 193 hashes;
   do not silently edit the archived manifests or reuse an occupied label.
3. Clear PYTHONPATH/PYTHONHOME; set LD_LIBRARY_PATH=/usr/lib/wsl/lib,
   SIMULATION_TAICHI_OFFLINE_CACHE=1, PYTHONUNBUFFERED=1 and
   PYTHONDONTWRITEBYTECODE=1. Run one CUDA job, monitor its actual owner, then
   bind the final owner SHA and run the corresponding reviewed terminal audit.
   Every accepted macro step must consume the full 0.005 s for each subsystem.
4. If the new run fails, diagnose the actual captured contract before another
   launch. After continuous success, recompute the component source manifest and
   complete the campaign's required requalification and registered ordering:
   FSI1-S0, M0/M1, conditional F0, FSI2, then FSI3. Focused passes do not open the
   Oracle/learning gate.

Publication: [source and complete evidence](https://github.com/lizhuoh9/EasyFsi/releases/tag/r15-wip-20260906-1330).

## Historical checkpoint: reviewed test corrections; full GREEN25 pending

The user explicitly resumed work past the earlier 10% quota boundary and
authorized committing and pushing all work before exhaustion. No quota reset
was redeemed. This checkpoint retains the core candidate SHA256
`26f37eb14179b73c0c4c4a7e29be53954dcade4a377ed4c91310026b88e51751`.
These changes correct test construction and cache expectations; they do not
add a new solver change or establish continuous physical success.

The completed strict-CUDA/f32 GREEN25 r02 ran 25 methods in 2410.098444 s.
It failed in three methods with five assertions; 22 methods had no failure.
Its process report SHA256 is
`c14ea84102356afa4d6867fa6f30324679577a8d0468bb3745cb128a4d16e12c`.
The root independently verified all case hashes/payloads, source193, host and
runtime identities. Zero/single consumer controls each matched all 15 baseline
keys and all eight output byte arrays; the original nonempty C0 control passed
both author orders. No identity errors, skips or expected failures were hidden.

## Three reviewed corrections

1. The mirrored fixture landed on a strict f32 route tie. Moving only its
   derived shadow sample Qz by one f32 ULP restores the intended S0/D1 route;
   no selector or tolerance changes. Native consumption is mask6, keys1/6.
2. The supposed shadow-only fault changed a cached selector which prepare
   recomputes, so it still admitted direct consumers. The corrected fault clears
   exactly the two direct activity flags while preserving shadow authors and
   cache/proof bytes. Native prepare now rejects two shadow-only consumers,
   preserving all eight output arrays and clearing all 41 transient fields.
3. The extrusion test assumed both transverse z caches were empty. The new
   fallback stores proved D/S geometry even for zero/one-consumer faces. The
   test now requires exact admission/full/fallback bits, owner0/1 and independent
   D/S seed keys. The original pair offset, counts, canonical values, one-ray
   equations, negative controls and cleanup remain unchanged. Of 58 original
   assertion calls, exactly two are replaced; the other 56 are AST-identical,
   and one exact fallback assertion is added.

The first two corrections passed 2/2 strict-CUDA tests in 32.782850 s,
process SHA256
`a2aef9d197251d172e97419e363ba0aa5c27be292ffe50226d63e9bbf7eaf2ab`.
The root verified both case payloads and all frozen dependencies. Astra/max
accepted the exact test diffs for native validation. The extrusion correction
has passed AST and Ruff F,E9 checks; its complete native method and the full
25-test suite are pending at this commit.

Current common mixin SHA256:
`0d2eadcd383137e91ff6d5999a70d577600566b2a7d4245f427f86f39da7064e`.
Current legacy ledger mixin SHA256:
`a4720ea3c257e95f44267ae5cbcc4c23a7f0f3090794ff22e94d850fc20e982f`.
Unit-test files are pinned separately from the 193-file solver identity.

## Evidence and next gates

Authoritative checkout: `/home/zhuohengli/worktrees/HIBM-MPM-r25b-live`,
branch `codex/turek-hron-fsi123-validation-r26a`, Ubuntu-22.04 user zhuohengli.
Interpreter: `/home/zhuohengli/.venvs/hibm-mpm-r26a-py310/bin/python`.
All original evidence is retained under the external
`robustness_work/resume_live_20260905_01` archive subtree:
`r15_fallback_contracts_r01/native_green_r02`,
`r15_fixture_corrections_r01/native_two_r01`, and
`r15_extrusion_cache_contract_r01`.

Run fresh `native_green_r03` with a manifest pinned to this committed source.
After full GREEN25 succeeds, update the reviewed full replay package to the
actual commit/test identities and run the original R12/R13/R14/R15 frozen
controls. Preserve captured HEAD5dcba96 and every original input byte; execution
HEAD is recorded separately. The old r02 freeze script is stale and must not
be executed. Then run the fresh eight-step physical R16 check and the original
coarse operator check before the registered component/formal entry gates.

All new checks so far advanced zero physical time. The last physical R15 run
still accepted five steps to 0.025 s and failed on the sixth at 0.030 s.
No full replay, R16 continuous pass, source-current coarse pass, component
requalification or formal FSI1/FSI2/FSI3 pass is claimed.

Published original snapshot and subsequent evidence updates:
[R15 WIP release](https://github.com/lizhuoh9/EasyFsi/releases/tag/r15-wip-20260906-1330).
The original release asset remains immutable; later receipts and archives
record their own hashes and source commit.
