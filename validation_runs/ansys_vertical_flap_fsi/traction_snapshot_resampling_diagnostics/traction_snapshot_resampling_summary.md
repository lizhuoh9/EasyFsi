# ANSYS vertical-flap shared-snapshot traction resampling

## Scope

This artifact reuses one archived shared preflow velocity/pressure/obstacle snapshot and re-runs only the marker traction sampling path. It does not advance the flow, the structure, or a coupled FSI loop.

## Shared snapshot

- Manifest: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/snapshot_manifest.json`
- Fields: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/step020_fields.npz`
- Source commit: `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`
- Field SHA-256: `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- Preflow steps: `20`
- Grid nodes: `[4, 32, 64]`

## Resampling result

- Completed formulations: 5
- Unsupported formulations: 1
- Candidate status: `snapshot_resampling_no_reference_selection`
- Snapshot identity: `shared_snapshot_sha256_identical_completed_rows`
- Baseline total marker force: -0.000179591 N
- Offset 0.25 force ratio vs baseline: 1.95073
- Offset 1.00 force ratio vs baseline: 0.0675151
- One-sided dual-face row: `unsupported` / `dual-face one-sided pressure needs per-face one-sided region support; current core exposes one one_sided_pressure_region_id`

## Files

- Matrix JSON: `validation_runs/ansys_vertical_flap_fsi/traction_snapshot_resampling_diagnostics/traction_snapshot_resampling_matrix.json`
- Matrix CSV: `validation_runs/ansys_vertical_flap_fsi/traction_snapshot_resampling_diagnostics/traction_snapshot_resampling_matrix.csv`
- History JSON: `validation_runs/ansys_vertical_flap_fsi/traction_snapshot_resampling_diagnostics/traction_snapshot_resampling_history.json`
- Marker diagnostics: `validation_runs/ansys_vertical_flap_fsi/traction_snapshot_resampling_diagnostics/marker_diagnostics`
- Checksums: `validation_runs/ansys_vertical_flap_fsi/traction_snapshot_resampling_diagnostics/CHECKSUMS.sha256`

## Findings

- The completed rows all use the same archived flow snapshot SHA-256, so force differences come from sampling formulation/geometry choices rather than from independently evolved flow fields.
- The 0.25-cell and 1.00-cell offsets remain strongly offset-sensitive, so the matrix is diagnostic evidence, not a reference-formulation selection.
- The dual-face one-sided pressure scenario remains fail-closed because the current core cannot assign separate one-sided pressure regions to the two opposing flap faces.
