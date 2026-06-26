# ANSYS vertical-flap traction probe offset decoupling

## Scope

This artifact reuses one archived shared preflow snapshot and re-runs only marker traction sampling. It does not advance the flow, the structure, or a coupled FSI loop, and it does not claim Fluent parity.

## Candidate decision

- reference_formulation_candidate: none
- candidate_status: `probe_offset_decoupling_diagnostic_only`
- candidate_blockers:
  - reference_selection_deferred
  - dual_face_one_sided_unsupported
  - probe_offset_decoupling_diagnostic_only
  - sampling_only_no_coupled_fsi

## Shared snapshot

- Manifest: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/snapshot_manifest.json`
- Fields: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/step020_fields.npz`
- Source commit: `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`
- Field SHA-256: `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`

## Ratio spans

- Fixed marker, swept pressure-probe origin relative span: 27.8932
- Fixed pressure-probe origin, swept marker relative span: 0

## Scenarios

| scenario | group | marker offset | probe-origin offset | ratio |
|---|---|---|---|---|
| fixed_marker0p51_probe0p00 | fixed_marker | 0.51 | 0.0 | 1.94081 |
| fixed_marker0p51_probe0p25 | fixed_marker | 0.51 | 0.25 | 1.95073 |
| fixed_marker0p51_probe0p51 | fixed_marker | 0.51 | 0.51 | 1 |
| fixed_marker0p51_probe1p00 | fixed_marker | 0.51 | 1.0 | 0.0675151 |
| fixed_probe0p51_marker0p00 | fixed_probe | 0.0 | 0.51 | 1 |
| fixed_probe0p51_marker0p25 | fixed_probe | 0.25 | 0.51 | 1 |
| fixed_probe0p51_marker0p51 | fixed_probe | 0.51 | 0.51 | 1 |
| fixed_probe0p51_marker1p00 | fixed_probe | 1.0 | 0.51 | 1 |

## Non-claims

- Does not claim Fluent parity.
- Does not run coupled 50-step FSI.
- Does not select a reference formulation.

## Next step

Use these fixed-marker and fixed-probe sweeps to decide whether the offset pathology is dominated by pressure-probe ladder origin, force marker geometry, or both before implementing per-face one-sided pressure support.
