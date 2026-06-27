# ANSYS vertical-flap pressure-probe ladder control

## Scope

This artifact reuses one archived shared preflow snapshot and re-runs only marker traction sampling. It does not advance the flow, the structure, or a coupled FSI loop, and it does not claim Fluent parity.

## Candidate decision

- reference_formulation_candidate: none
- candidate_status: `probe_ladder_control_no_stable_candidate`
- stable_ladder_candidate: `None`
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

## Strategy Acceptance

| strategy | accepted | force span | max residual |
|---|---:|---:|---:|
| current_control_baseline | False | 21.1364 | 0 |
| origin0p51_start1p00_spacing0p50 | False | 21.1364 | 0 |
| origin0p51_start0p75_spacing0p50 | False | 0.828017 | 0 |
| origin0p51_start0p625_spacing0p25 | False | 0.895919 | 0 |
| origin0p51_start0p51_spacing0p25 | False | 1.01559 | 0 |

## Rows

| scenario | probe-origin offset | start | spacing | ratio |
|---|---:|---:|---:|---:|
| current_control_baseline_probe0p51 | 0.51 |  | 0.5 | 1 |
| current_control_baseline_probe0p625 | 0.625 |  | 0.5 | 0.0451745 |
| current_control_baseline_probe1p00 | 1.0 |  | 0.5 | 0.0675151 |
| origin0p51_start1p00_spacing0p50_probe0p51 | 0.51 | 1.0 | 0.5 | 1 |
| origin0p51_start1p00_spacing0p50_probe0p625 | 0.625 | 1.0 | 0.5 | 0.0451745 |
| origin0p51_start1p00_spacing0p50_probe1p00 | 1.0 | 1.0 | 0.5 | 0.0675151 |
| origin0p51_start0p75_spacing0p50_probe0p51 | 0.51 | 0.75 | 0.5 | 1 |
| origin0p51_start0p75_spacing0p50_probe0p625 | 0.625 | 0.75 | 0.5 | 1.15848 |
| origin0p51_start0p75_spacing0p50_probe1p00 | 1.0 | 0.75 | 0.5 | 1.82802 |
| origin0p51_start0p625_spacing0p25_probe0p51 | 0.51 | 0.625 | 0.25 | 1 |
| origin0p51_start0p625_spacing0p25_probe0p625 | 0.625 | 0.625 | 0.25 | 1.19147 |
| origin0p51_start0p625_spacing0p25_probe1p00 | 1.0 | 0.625 | 0.25 | 1.89592 |
| origin0p51_start0p51_spacing0p25_probe0p51 | 0.51 | 0.51 | 0.25 | 1 |
| origin0p51_start0p51_spacing0p25_probe0p625 | 0.625 | 0.51 | 0.25 | 1.23681 |
| origin0p51_start0p51_spacing0p25_probe1p00 | 1.0 | 0.51 | 0.25 | 2.01559 |

## Deferred strategies

- origin0p51_symmetric_cell_pair_policy: report_only_deferred

## Non-claims

- Does not claim Fluent parity.
- Does not run coupled 50-step FSI.
- Does not select a reference formulation.
- Does not implement per-face one-sided pressure.
