# Vertical-flap fine-grid solver campaign

This directory retains historical comparison artifacts. The only supported
solver entrypoint is `scripts/run_our_solver_vertical_flap.py`.

The current comparison campaign writes new runs below
`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/`.
Every launch must use a new timestamped directory; never reuse an existing
output directory or any historical `our_solver` directory in this tree.

Run the fine 50-step configuration from the repository root with PowerShell:

```powershell
$repo = (Resolve-Path '.').Path
$python = if ($env:EASYFSI_PYTHON) { $env:EASYFSI_PYTHON } else { 'python' }
$runName = "fine50_current_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
$out = Join-Path $repo "validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs\$runName"
if (Test-Path -LiteralPath $out) { throw "Refusing to reuse output directory: $out" }

& $python `
  "$repo\validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\scripts\run_our_solver_vertical_flap.py" `
  --output-dir $out `
  --run-label $runName `
  --steps 50 `
  --grid-nodes 4 256 320 `
  --solid-particle-counts 1 256 20 `
  --marker-count 64 `
  --flow-projection-iterations 1080 `
  --preflow-steps 40 `
  --flow-cg-preconditioner fv_multigrid `
  --flow-pressure-solve-failure-policy raise `
  --solid-substeps 1600 `
  --flow-predictor-substeps 64 `
  --hibm-search-radius-m 0.0017 `
  --span-reduction mean `
  --streamwise-velocity-sign -1.0 `
  --save-step-fields
```

The runner writes `run_manifest.json`, `our_solver_config.json`, and
`progress.json` inside that unique output directory. Historical launch records
below `our_solver/` are provenance only and are not executable instructions.
