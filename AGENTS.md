# Project Navigation for Agents

## Authoritative WSL worktree

For this project, the Windows checkout is only the Codex project entrypoint. The authoritative source tree is the WSL checkout:

```text
/home/zhuohengli/work/squid-robot/HIBM-MPM-refactored
```

At the start of every task, inspect the WSL repository branch and status. Run all Git operations, source edits, tests, and numerical simulations against that WSL path through `wsl.exe -d Ubuntu-22.04 -- ...`. Do not silently use the Windows mirror as the source of truth or copy changes back from Windows into WSL. If a command cannot be executed against WSL, report the limitation instead of falling back to the Windows checkout.

This file is the fast path into this repository. Do not start by recursively
reading every file.

## Read First

1. Read the exact file named by the user, especially a handoff under
   `docs/refactoring/`.
2. Read `docs/README.md` for the documentation index.
3. Read `docs/MODULE_MAP.md` before changing solver code.
4. Inspect only the relevant source package and its focused tests.

For the current ANSYS vertical-flap work, start with:

- `docs/refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-23.md`
- `cases/ansys_vertical_flap_fsi.py`
- `benchmarks/official/solid_mpm_fsi_runner.py`
- `simulation_core/fluids/solver.py`

## Default Scan Exclusions

Unless the user names a specific artifact, do not recursively scan:

- `.git/`
- `.claude/worktrees/`
- `validation_runs/`
- `tmp/`
- `archive/`
- `__pycache__/`, `.pytest_cache/`, and Taichi caches

`validation_runs/` contains historical numerical evidence, not implementation
source. Use `validation_runs/README.md` to select a specific campaign first.

## Live Code Map

- `simulation_core/`: reusable solver implementation.
- `cases/`: case definitions and user-facing case configuration.
- `benchmarks/official/`: official/reference benchmark runners.
- `tools/`: diagnostics, validation, rendering, and post-processing.
- `tests/`: focused tests grouped by responsibility.
- `scripts/`: repository maintenance and structure checks.
- `docs/`: architecture, validation instructions, and handoffs.

Do not import code from `archive/` or `validation_runs/` into production paths.

## Working-Tree Discipline

- The worktree is intentionally dirty. Preserve unrelated modified and
  untracked files.
- Never use `git clean`, `git reset --hard`, or bulk-delete untracked files.
- Untracked tests, handoffs, CLIs, Fluent files, and run artifacts may be
  required evidence.
- Treat source changes, validation execution, and evidence publication as
  separate steps.

## Validation

Use the trusted interpreter on this machine:

```powershell
$python = 'D:\working\taichi\env\python.exe'
```

Start with the smallest relevant test. The structure-only validation matrix is
in `docs/VALIDATION.md`. Do not launch a long Taichi or Fluent run unless the
task explicitly requires it and the expected output directory is known.

## Naming New Files

- Python modules and tests: descriptive `snake_case`; avoid version-only names.
- Documentation: `<TOPIC>_<PURPOSE>_YYYY-MM-DD.md`.
- Handoffs: `<TOPIC>_THREAD_HANDOFF_YYYY-MM-DD.md`.
- Validation runs: follow `validation_runs/README.md`; keep each path segment
  under 80 characters to avoid Windows long-path failures.
