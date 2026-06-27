# ANSYS vertical-flap pressure pair anchor map

## Scope

This artifact reuses one archived shared preflow snapshot and re-runs only marker traction sampling. It does not advance the flow, the structure, or a coupled FSI loop, and it does not claim Fluent parity.

## Candidate decision

- reference_formulation_candidate: none
- candidate_status: `pressure_pair_anchor_map_stable_candidate_found`
- stable_pressure_pair_policy: `baseline_anchored_cell_pair`
- candidate_blockers:
  - reference_selection_deferred
  - dual_face_one_sided_unsupported
  - sampling_only_no_coupled_fsi
  - no_fluent_parity_claim

## Shared snapshot

- Manifest: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/snapshot_manifest.json`
- Fields: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/step020_fields.npz`
- Source commit: `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`
- Field SHA-256: `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`

## Anchor Map Gate

| gate | value |
|---|---:|
| accepted | True |
| force span | 0 |
| max traction residual | 0 |
| anchor selected all markers | True |
| anchor fallback zero | True |

## Rows

| scenario | policy | probe-origin offset | ratio | anchor selected |
|---|---|---:|---:|---:|
| baseline_independent_probe0p51 | independent_ladder | 0.51 |  | 0 |
| anchored_from_baseline_probe0p00 | baseline_anchored_cell_pair | 0.0 | 0.996273 | 24 |
| anchored_from_baseline_probe0p25 | baseline_anchored_cell_pair | 0.25 | 0.996273 | 24 |
| anchored_from_baseline_probe0p375 | baseline_anchored_cell_pair | 0.375 | 0.996273 | 24 |
| anchored_from_baseline_probe0p51 | baseline_anchored_cell_pair | 0.51 | 0.996273 | 24 |
| anchored_from_baseline_probe0p625 | baseline_anchored_cell_pair | 0.625 | 0.996273 | 24 |
| anchored_from_baseline_probe0p75 | baseline_anchored_cell_pair | 0.75 | 0.996273 | 24 |
| anchored_from_baseline_probe1p00 | baseline_anchored_cell_pair | 1.0 | 0.996273 | 24 |
| anchored_from_baseline_probe1p50 | baseline_anchored_cell_pair | 1.5 | 0.996273 | 24 |

## Non-claims

- Does not claim Fluent parity.
- Does not run coupled 50-step FSI.
- Does not select a reference formulation.
- Does not implement per-face one-sided pressure.
