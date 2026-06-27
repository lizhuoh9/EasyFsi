# ANSYS vertical-flap pressure-pair reference preselection

## Scope

This artifact reuses one archived shared preflow snapshot and re-runs only marker traction sampling. It does not advance the flow, the structure, or a coupled FSI loop, and it does not claim Fluent parity.

## Candidate decision

- candidate_status: `pressure_pair_policy_preselection_candidate_found`
- pressure_pair_policy_candidate: `baseline_anchored_cell_pair`
- reference_formulation_candidate: none
- dual_one_sided_offset0p51_pressure_only_unsupported_confirmed: True

## Gates

| gate | value |
|---|---:|
| accepted | True |
| force span | 0 |
| absolute baseline bias | 0.00372681 |
| max traction residual | 0 |
| anchor selected all markers | True |
| anchor fallback zero | True |

## Rows

| scenario | status | policy | ratio | anchor selected |
|---|---|---|---:|---:|
| baseline_independent_ladder_probe0p51 | completed | independent_ladder |  | 0 |
| anchored_pair_dual_faces_probe0p00 | completed | baseline_anchored_cell_pair | 0.996273 | 24 |
| anchored_pair_dual_faces_probe0p25 | completed | baseline_anchored_cell_pair | 0.996273 | 24 |
| anchored_pair_dual_faces_probe0p375 | completed | baseline_anchored_cell_pair | 0.996273 | 24 |
| anchored_pair_dual_faces_probe0p51 | completed | baseline_anchored_cell_pair | 0.996273 | 24 |
| anchored_pair_dual_faces_probe0p625 | completed | baseline_anchored_cell_pair | 0.996273 | 24 |
| anchored_pair_dual_faces_probe0p75 | completed | baseline_anchored_cell_pair | 0.996273 | 24 |
| anchored_pair_dual_faces_probe1p00 | completed | baseline_anchored_cell_pair | 0.996273 | 24 |
| anchored_pair_dual_faces_probe1p50 | completed | baseline_anchored_cell_pair | 0.996273 | 24 |
| dual_one_sided_offset0p51_pressure_only_unsupported_confirmed | unsupported | per_face_one_sided_pressure |  | 0 |

## Candidate blockers

- dual_face_one_sided_unsupported: dual-face one-sided pressure needs per-face one-sided region support
- sampling_only_no_coupled_fsi: Rows reuse one flow snapshot and do not advance coupled FSI.
- no_fluent_parity_claim: No coupled or Fluent comparison run is part of this artifact.
- reference_selection_deferred: This artifact preselects only a pressure-pair component.

## Non-claims

- Does not claim Fluent parity.
- Does not run coupled 50-step FSI.
- Does not select a complete reference formulation.
- Does not implement per-face one-sided pressure.
