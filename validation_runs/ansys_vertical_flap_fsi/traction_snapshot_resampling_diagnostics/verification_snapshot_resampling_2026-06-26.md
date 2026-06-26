# Verification: shared-snapshot traction resampling

- Date: 2026-06-26
- Command: `python validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py`
- Shared snapshot source commit: `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`
- Shared snapshot SHA-256: `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- Completed formulations: 5
- Unsupported formulations: 1
- Candidate status: `snapshot_resampling_no_reference_selection`

The script checks the archived NPZ checksum against the shared snapshot manifest before any formulation is sampled. Completed formulation rows therefore share the same velocity/pressure/obstacle fields.
