# Archive

This directory contains historical maintenance tools and snapshots. Nothing
under `archive/` is a production import target or a default agent scan target.

- `maintenance/`: retired repository-maintenance code.
- `tools/`: one-off historical diagnostics and launch helpers.
- `snapshots/`: explicitly dated legacy source snapshots.

New snapshots must use `<component>_YYYY-MM-DD_<purpose>.<ext>` and include a
short note describing why Git history is not sufficient.
