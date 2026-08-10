# Validation Run Index

`validation_runs/` stores reproducibility evidence and large generated results.
It is not a source-code directory. Select one campaign from this index before
reading files; do not recursively scan the whole tree.

Git tracks this index and lightweight maintenance files, not the large artifact
bodies; those paths resolve in the complete local evidence workspace.

Machine-readable canonical entries are listed in `ARTIFACT_INDEX.json`.

## Current ANSYS Vertical-Flap Evidence

| Purpose | Canonical path | Notes |
| --- | --- | --- |
| Fluent steady reference | `ansys_vertical_flap_fsi/official_fluent_fine_mesh_steady_2026-07-01/` | Native Fluent mesh, case/data, transcript, and steady reference evidence. |
| Fluent two-way FSI reference | `ansys_vertical_flap_fsi/official_fluent_fine_fsi_valid_2026-07-10/` | Native transient FSI reference. |
| Solver-vs-Fluent workspace | `ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/` | Comparison scripts, runs, post-processing, and diagnostics. |
| Latest cold inner-stage trace | `solver_soaks/vf48w_sst_inner_stage_trace_20260724_c/` | Current cold-start timing evidence referenced by the handoff. |
| Latest warm trace | `solver_soaks/vf48x_sst_warm_trace_20260724_a/` | Warm-cache comparison for initialization analysis. |
| Reusable warm Taichi cache | `.taichi_cache/vf48w_sst_inner_stage_trace_cuda_f32/` | Preserve while investigating cold/warm initialization. |

The authoritative task boundary is
`docs/refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-23.md`.
Older `probe*`, `diag*`, `dryrun*`, `vf47*`, and earlier `vf48*` directories are
historical evidence. They are not the default starting point and must not be
deleted solely because their names look temporary.

## Other Campaigns

- `ansys_vertical_flap_fixed_flow/`: fixed-flow visualization and reports.
- `turek_hron_fsi/`: Turek-Hron reference and solver validation.
- `solver_vs_fluent_comparison_20260713/`: cross-solver comparison bundle.
- `solver_soaks/`: bounded solver soak, cold-start, and timing probes.

## Naming New Runs

Use:

```text
<case>__<stage>__<purpose>__YYYYMMDD__rNN
```

Example:

```text
ansys_vf__preflow__sst_cold_profile__20260810__r01
```

Rules:

- Use a readable case and purpose; do not use an opaque version code alone.
- Keep every path segment under 80 characters and the intended full path under
  200 characters.
- Put `status`, `command`, `source_sha256`, `parent_run_id`, tolerances, and
  environment details in a run manifest instead of adding them all to the
  directory name.
- Use stable suffixes: `.stdout.log`, `.stderr.log`, `.summary.json`,
  `.manifest.json`.
- Never write pytest temporary trees inside a published campaign directory.

Existing campaign names remain unchanged so that handoffs, manifests, and
reports keep their evidence links.
