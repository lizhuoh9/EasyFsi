# ANSYS vertical-flap per-face one-sided pressure

## Scope

This artifact reuses one archived shared preflow snapshot and re-runs only marker traction sampling. It does not advance the flow, the structure, or a coupled FSI loop, and it does not claim Fluent parity.

## Candidate decision

- candidate_status: `per_face_one_sided_pressure_completed`
- pressure_pair_policy_candidate: `baseline_anchored_cell_pair`
- one_sided_pressure_policy_candidate: `per_face_mirrored`
- reference_formulation_candidate: none

## Gates

| gate | value |
|---|---:|
| accepted | True |
| per-face rows | 3 |
| one-sided complete | True |
| pressure complete | True |
| invalid counts zero | True |
| anchor selected all markers | True |
| anchor fallback zero | True |
| max traction residual | 0 |

## Rows

| scenario | status | one-sided policy | total force z | one-sided markers |
|---|---|---|---:|---:|
| baseline_anchored_two_sided_probe0p51 | completed | disabled | -0.000178921 | 0 |
| dual_one_sided_per_face_probe0p51 | completed | per_face_mirrored | -0.000178921 | 24 |
| dual_one_sided_per_face_probe0p625 | completed | per_face_mirrored | -0.000178921 | 24 |
| dual_one_sided_per_face_probe1p00 | completed | per_face_mirrored | -0.000178921 | 24 |

## Candidate blockers

- reference_selection_deferred: This artifact completes one-sided pressure only.
- sampling_only_no_coupled_fsi: Rows reuse one flow snapshot and do not advance coupled FSI.
- no_fluent_parity_claim: No coupled or Fluent comparison run is part of this artifact.

## Non-claims

- Does not claim Fluent parity.
- Does not run coupled 50-step FSI.
- Does not select a complete reference formulation.
