# ANSYS Traction Shared Snapshot - 2026-06-26

scope_limit = fixed-solid shared flow snapshot only; no coupled 50-step or Fluent parity claim

field_path = validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/step020_fields.npz

field_sha256 = 3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968

grid_shape = [4, 32, 64]

pressure_range_pa = [-4.472715807665312, 25.999032120465625]

velocity_peak_mps = 31.640314138895192

velocity_p999_mps = 25.1116388870788

This snapshot exists so future formulation rows can be sampled from the exact same pressure/velocity field.

It does not prove Fluent parity, does not run coupled 50-step FSI, and no reference formulation is selected.

next_intended_step = snapshot resampling matrix
