# ANSYS vertical-flap pressure-probe ladder stability

## Scope

This artifact reuses one archived shared preflow snapshot and re-runs only marker traction sampling. It does not advance the flow, the structure, or a coupled FSI loop, and it does not claim Fluent parity.

## Candidate decision

- reference_formulation_candidate: none
- candidate_status: `probe_ladder_stability_diagnostic_only`
- candidate_blockers:
  - reference_selection_deferred
  - dual_face_one_sided_unsupported
  - probe_ladder_stability_diagnostic_only
  - sampling_only_no_coupled_fsi
  - no_fluent_parity_claim

## Shared snapshot

- Manifest: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/snapshot_manifest.json`
- Fields: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/step020_fields.npz`
- Source commit: `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`
- Field SHA-256: `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`

## Probe Ladder Transition Summary

- Force-ratio span across probe-origin offsets: 42.2919
- First offset with force amplification > 1.5: 0.0
- First offset with force collapse < 0.1: 0.625
- First primary nearest-cell transition offset: 0.0
- First secondary nearest-cell transition offset: 0.0
- First high-side force collapse offset: 0.625
- First high-side primary nearest-cell transition offset: 0.625
- First high-side secondary nearest-cell transition offset: 1.0
- Nearest below-baseline amplification offset: 0.375
- Nearest below-baseline amplification ratio: 1.95569
- Nearest above-baseline force ratio: 0.0451745
- 0.51 to 1.00 collapse has nearest-cell/rung transition: True

## Scenarios

| scenario | probe-origin offset | force ratio | primary jump | secondary jump |
|---|---:|---:|---:|---:|
| probe_offset0p00 | 0.0 | 1.94081 | -5.79676 | 5.82162 |
| probe_offset0p125 | 0.125 | 1.94577 | -5.80951 | 5.83855 |
| probe_offset0p25 | 0.25 | 1.95073 | -5.82227 | 5.85548 |
| probe_offset0p375 | 0.375 | 1.95569 | -5.83502 | 5.87241 |
| probe_offset0p51 | 0.51 | 1 | -5.8488 | 0.137553 |
| probe_offset0p625 | 0.625 | 0.0451745 | -0.108585 | 0.161845 |
| probe_offset0p75 | 0.75 | 0.0526213 | -0.12676 | 0.188249 |
| probe_offset0p875 | 0.875 | 0.0600683 | -0.144935 | 0.214654 |
| probe_offset1p00 | 1.0 | 0.0675151 | -0.16311 | 0.241059 |
| probe_offset1p25 | 1.25 | 0.0824088 | -0.19946 | 0.293868 |
| probe_offset1p50 | 1.5 | 0.0973026 | -0.23581 | 0.346677 |

## Non-claims

- Does not claim Fluent parity.
- Does not run coupled 50-step FSI.
- Does not select a reference formulation.
- Does not change the core pressure formula or force aggregation.

## Next step

Use this nearest-cell/rung transition map to decide whether the next diagnostic should split probe origin from ladder start and spacing controls. Do not move to one-sided pressure or reference selection until a stable ladder candidate is proven on a shared snapshot.
