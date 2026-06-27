# ANSYS vertical-flap symmetric pressure pair

## Scope

This artifact reuses one archived shared preflow snapshot and re-runs only marker traction sampling. It does not advance the flow, the structure, or a coupled FSI loop, and it does not claim Fluent parity.

## Candidate decision

- reference_formulation_candidate: none
- candidate_status: `symmetric_pressure_pair_no_stable_candidate`
- stable_symmetric_pressure_pair_candidate: `None`
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

## Symmetric Pair Gate

| gate | value |
|---|---:|
| accepted | False |
| force span | 43.5428 |
| max traction residual | 0 |
| max pair residual | 3.8147e-06 |
| pair selected all markers | True |
| pair fallback zero | True |

## Rows

| scenario | policy | probe-origin offset | ratio | pair selected |
|---|---|---:|---:|---:|
| independent_ladder_baseline_probe0p51 | independent_ladder | 0.51 | 1 | 0 |
| independent_ladder_baseline_probe0p625 | independent_ladder | 0.625 | 0.0451745 | 0 |
| independent_ladder_baseline_probe1p00 | independent_ladder | 1.0 | 0.0675151 | 0 |
| symmetric_pair_probe0p51 | symmetric_cell_pair | 0.51 | 1 | 24 |
| symmetric_pair_probe0p625 | symmetric_cell_pair | 0.625 | 0.0441251 | 24 |
| symmetric_pair_probe1p00 | symmetric_cell_pair | 1.0 | 0.0659468 | 24 |
| symmetric_pair_probe0p00 | symmetric_cell_pair | 0.0 | 1.94363 | 24 |
| symmetric_pair_probe0p25 | symmetric_cell_pair | 0.25 | 1.95818 | 24 |
| symmetric_pair_probe0p375 | symmetric_cell_pair | 0.375 | 1.96545 | 24 |
| symmetric_pair_probe0p75 | symmetric_cell_pair | 0.75 | 0.051399 | 24 |
| symmetric_pair_probe1p50 | symmetric_cell_pair | 1.5 | 0.0950423 | 24 |

## Non-claims

- Does not claim Fluent parity.
- Does not run coupled 50-step FSI.
- Does not select a reference formulation.
- Does not implement per-face one-sided pressure.
