# Step 8 Goal: Move HIBM-MPM implementation into simulation_core.coupling.hibm_mpm

Objective:
Move the `simulation_core/hibm_mpm.py` implementation into
`simulation_core/coupling/hibm_mpm` while preserving legacy imports.

Base branch:
`refactor/fluid-package-split-step7`

Working branch:
`refactor/hibm-mpm-package-split-step8`

Hard boundaries:
- Do not change HIBM/MPM solver physics.
- Do not change Taichi kernel math.
- Do not change IB node search behavior.
- Do not change inside/outside classification.
- Do not change pressure Neumann matrix or pressure Neumann gradient behavior.
- Do not change stress sampling behavior.
- Do not change surface marker force scatter behavior.
- Do not change sharp HIBM-MPM step order.
- Do not change FSI coupling formulas.
- Do not change default constants.
- Do not change report field names.
- Do not change public class names or function names.
- Do not fix known solver red lights in this step.
- Keep `simulation_core.hibm_mpm` as a compatibility shim.

Allowed code surface:
- `simulation_core/hibm_mpm.py`
- `simulation_core/coupling/__init__.py`
- `simulation_core/coupling/hibm_mpm/*`
- `tests/solvers/test_simulation_core_facades.py`
- `tests/contracts/test_architecture_boundaries.py`
- `tests/contracts/test_source_static_contracts.py`
- `tests/solvers/test_hibm.py`
- `tests/cases/test_squid_latest_core_config.py`
- `README.md`
- `SIMULATION_CORE_USAGE.md`
- `REFACTORING_NOTES.md`

Forbidden code surface:
- `simulation_core/fluids/*`
- `simulation_core/fluid.py`
- `simulation_core/mooney_shell_mpm.py`
- `simulation_core/neo_hookean_mpm.py`
- `simulation_core/tri_surface.py`
- `simulation_core/projected_ibm.py`
- `simulation_core/fsi_coupling.py`
- `benchmarks/`
- `cases/`

Target package layout:
```text
simulation_core/coupling/hibm_mpm/
|-- __init__.py
|-- constants.py
|-- modes.py
|-- reports.py
|-- paper_requirements.py
`-- core.py
```

Done criteria:
- `simulation_core/coupling/hibm_mpm/core.py` owns the HIBM-MPM implementation.
- `simulation_core/hibm_mpm.py` is a compatibility shim.
- HIBM constants live in `simulation_core/coupling/hibm_mpm/constants.py`.
- HIBM mode constants/helpers live in `simulation_core/coupling/hibm_mpm/modes.py`.
- HIBM report dataclasses live in `simulation_core/coupling/hibm_mpm/reports.py`.
- Paper requirement dataclass/helper live in
  `simulation_core/coupling/hibm_mpm/paper_requirements.py`.
- `simulation_core.coupling.hibm_mpm` re-exports public HIBM-MPM APIs.
- `simulation_core.coupling` re-exports HIBM-MPM APIs from the new package, not
  from the legacy shim.
- Old legacy import, new package import, and coupling facade import are
  identity-compatible.
- HIBM-MPM support modules do not import `core.py`.

Required validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  simulation_core\hibm_mpm.py `
  simulation_core\coupling\__init__.py `
  simulation_core\coupling\hibm_mpm\__init__.py `
  simulation_core\coupling\hibm_mpm\constants.py `
  simulation_core\coupling\hibm_mpm\modes.py `
  simulation_core\coupling\hibm_mpm\reports.py `
  simulation_core\coupling\hibm_mpm\paper_requirements.py `
  simulation_core\coupling\hibm_mpm\core.py

& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_simulation_core_facades -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\contracts -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\integration -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\tools -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\cases -p "test_squid*.py" -v
```

Optional validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_hibm -v
```

Residual checks:
- `simulation_core/hibm_mpm.py` contains no `class HibmMpmSurfaceMarkers`.
- `simulation_core/hibm_mpm.py` contains no `class HibmMpmSharpCouplingState`.
- `simulation_core/hibm_mpm.py` contains no `@ti.kernel`.
- `simulation_core/coupling/__init__.py` does not import from
  `simulation_core.hibm_mpm`.
- HIBM-MPM support modules do not import `simulation_core.coupling.hibm_mpm.core`.
- Forbidden solver/case/benchmark files remain unchanged.
- `git diff --check` passes apart from repository line-ending normalization warnings.

Completion notes:
- Implemented `simulation_core.coupling.hibm_mpm` with `core`, `constants`,
  `modes`, `reports`, and `paper_requirements`.
- Kept `simulation_core.hibm_mpm` as a compatibility shim.
- Updated `simulation_core.coupling` to import HIBM-MPM exports from the new
  package.
- Updated facade, architecture, HIBM source-path, and squid source-path tests
  to read the package-backed implementation.
- Verified old legacy import, new package import, and coupling facade import are
  identity-compatible.
- Verified the split is behavior-preserving at text level: Step 7 base
  `constants`, `reports`, `paper`, `modes`, and `core_body` segments all match
  the Step 8 split files by SHA-256.

Validation results:
- `py_compile` for the shim, facade, and all new HIBM-MPM package modules:
  passed.
- HIBM facade identity smoke: passed.
- `tests.solvers.test_simulation_core_facades` plus
  `tests.contracts.test_architecture_boundaries`: 30 tests passed.
- `tests/contracts` discovery: 41 tests passed.
- `tests/integration` discovery: 4 tests passed.
- `tests/tools` discovery: 9 tests passed.
- Squid case tests
  `tests.cases.test_squid_latest_core_config tests.cases.test_squid_package_exports`:
  244 tests passed.
- `tests.solvers.test_simulation_core_package`: 26 tests passed.
- Full `tests/cases` discovery was also attempted. After HIBM source-path test
  fixes, the remaining failure was the out-of-scope ANSYS vertical-flap smoke
  (`max_displacement_relative_error=0.45353834739174037` vs tolerance `0.05`),
  not a Step 8 directory-boundary regression.
- Optional `tests.solvers.test_hibm` was attempted and timed out after 900 s.
  The two early failing tests were rerun in this branch and in a temporary
  Step 7 base worktree; both branches fail identically with `unreached 1 != 2`,
  so they are pre-existing relative to Step 8.
