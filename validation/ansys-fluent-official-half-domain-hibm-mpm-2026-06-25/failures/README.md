# Failed and superseded attempts

These files are retained to document what went wrong before the official
half-domain HIBM-MPM run was produced.

- `superseded_full_domain_4x640x640_p1080_pressure_failure.json`: hand-built
  full-domain two-flap attempt, not the official Fluent half-domain convention.
  It reached the pressure solve but failed before stress sampling because the
  pressure correction did not converge at `1080` iterations.
- `superseded_full_domain_4x640x640_p4096_s200_mpm_oob_failure.json`: same
  superseded full-domain geometry with `4096` projection iterations. The pressure
  part progressed farther, but the solid step failed because `108/3840` MPM
  particles left the background grid when `solid_substeps=200`.

The successful archived result is the official half-domain run in `../data/`.
