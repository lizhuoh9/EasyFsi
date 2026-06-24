# Post-refactor Baseline Goal

Objective:
Record the post-refactor integration and solver-validation baseline after the
ten-step structure refactor is complete.

Base branch:
`refactor/final-layout-closure-step10`

Base commit:
`56d5167331a1f2d07dd32c8943c27d9bc1bf430f`

Working branch:
`docs/post-refactor-baseline`

Hard boundaries:
- Do not change solver physics.
- Do not change Taichi kernel math.
- Do not change fluid projection formulas.
- Do not change HIBM/MPM coupling behavior.
- Do not change solid MPM formulas.
- Do not change material formulas.
- Do not change geometry or CAD behavior.
- Do not change benchmark formulas.
- Do not change case defaults.
- Do not change CLI defaults.
- Do not change tests.
- Do not modify `simulation_core/`.
- Do not modify `cases/`.
- Do not modify `benchmarks/`.
- Do not modify `tools/`.
- Do not modify `tests/`.
- Do not fix known solver red lights in this docs-only branch.

Allowed edit surface:
- `docs/POST_REFACTOR_BASELINE.md`
- `docs/refactoring/POST_REFACTOR_BASELINE_GOAL_2026-06-24.md`

Required documentation:
1. Record the final structure-refactor commit:
   `56d5167331a1f2d07dd32c8943c27d9bc1bf430f`.
2. Record the planned tag name:
   `structure-refactor-2026-06-24`.
3. Record the passing structure gates from `docs/VALIDATION.md`.
4. Record known non-gating failures:
   - ANSYS vertical-flap displacement tolerance.
   - Mooney/Neo-Hookean secondary region behavior.
   - Long-running or historical `test_core_fluid` failures/timeouts.
   - Long-running or historical `test_hibm` failures/timeouts.
5. Record policy that these failures must not be fixed in structure-only PRs.
   Each future fix needs a solver-specific branch with physical validation
   evidence.

Required validation:
```powershell
& 'D:\working\taichi\env\python.exe' scripts\validate_structure.py
git diff --check
git diff --name-only -- simulation_core cases benchmarks tools tests run_simulation.py
```

Acceptance:
- `docs/POST_REFACTOR_BASELINE.md` exists.
- The baseline document records the final commit, tag name, structure gates,
  known non-gating failures, and future-branch policy.
- The only changed files are the baseline document and this goal document.
- No solver logic, FSI physics, case behavior, benchmark behavior, tools, or
  tests are changed.
