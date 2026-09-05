# HIBM-MPM project navigation

This subtree is the refactored HIBM-MPM solver. Parent instructions for the
separate GUI/sim_core project do not select this solver's modules or interpreter.
Keep this entrypoint short; read the relevant handoff and validation contract.

## Select the authoritative worktree

The Windows checkout is a Codex entrypoint. Run solver Git operations, edits,
tests, and numerical work in WSL Ubuntu-22.04. Windows user-level configuration
and this entrypoint document can be maintained from PowerShell when requested.

Use the WSL worktree explicitly selected by the user's current handoff. Verify
its branch, HEAD, and status before acting. Do not substitute a historical root
or copy solver changes from the Windows mirror into WSL.

The R26A handoff verified on 2026-09-05 uses:

```text
/home/zhuohengli/worktrees/HIBM-MPM-r25b-live
codex/turek-hron-fsi123-validation-r26a
```

`/home/zhuohengli/work/squid-robot/HIBM-MPM-refactored` is a different, older
checkout. Inspect it only when the current task selects it; its previous dirty
state must be preserved. When no handoff selects a root, identify the intended
worktree from repository metadata/documentation before making changes.

If a sandboxed WSL command returns E_ACCESSDENIED, use the available scoped
approval mechanism for the authorized operation. A transport or permission
error does not authorize switching source trees or relaxing the sandbox.

## Read first

1. The exact file or current handoff named by the user.
2. `docs/README.md`, then the selected campaign's current goal/report.
3. `docs/MODULE_MAP.md` before editing solver implementation.
4. The relevant source package and focused tests, using symbol searches.

For Turek-Hron, start with `docs/TUREK_HRON_VALIDATION.md` and
`docs/validation/TUREK_HRON_FSI123_NUMERICAL_VALIDATION_GOAL_2026-09-02.md`.
For ANSYS vertical flap, use the handoff selected by the current request rather
than an old default dated July. Existing numerical requirements remain in those
documents; this instruction update does not change acceptance thresholds.

Personal workflow skills used by this Windows entrypoint:

- `C:/Users/lizhu/.codex/skills/windows-wsl-execution/SKILL.md` for cross-shell execution.
- `C:/Users/lizhu/.codex/skills/hibm-fsi-validation/SKILL.md` for this project's numerical work.
- The catalog's `taichi-docs` skill for Taichi semantics and runtime questions.

## Runtime and process ownership

Use the selected campaign's verified Linux interpreter, not Windows Python or
an unqualified `python3`. The restored R26A environment is:

```text
/home/zhuohengli/.venvs/hibm-mpm-r26a-py310/bin/python
```

Its CPython 3.10.12 / NumPy 2.1.2 / SciPy 1.15.3 host identity matched the frozen
campaign on 2026-09-05. Recheck before a run; the path alone proves nothing.
The earlier `/tmp/hibm-mpm-r26a-py310` environment disappeared. Do not recreate
long-lived validation environments under `/tmp` or substitute the R25A CPU venv.
See `docs/refactoring/AGENT_WORKFLOW_AUDIT_2026-09-05.md` for recovery evidence.

R26A launch context clears PYTHONPATH and PYTHONHOME and sets
LD_LIBRARY_PATH=/usr/lib/wsl/lib, SIMULATION_TAICHI_OFFLINE_CACHE=1, and
PYTHONUNBUFFERED=1. Use wsl.exe -d Ubuntu-22.04 --cd <selected-root> --exec
with direct native arguments, or execute an inspected LF script. Ordinary --
can reparse metacharacters through Bash; an absolute script path does not set
its working directory. Do not build nested PowerShell/Bash programs.

Run one expensive CUDA job at a time. Track its session, PID/process group,
command, and output directory. Poll that same process. Quiet stdout or absent
chunk files is not proof of a pause, failure, or hang; inspect actual activity
and the documented flush cadence. Do not restart an uncertain job blindly.

## Numerical and evidence boundaries

- Fix the actual failing contract; check neighboring paths and configuration
  propagation before another expensive run. Use meaningful bounded regressions.
- Fluid and solid accepted physical substeps each consume the full macro dt_s.
  Rejected trials consume no accepted time; algebraic convergence does not end
  physical advancement. Preserve accepted-state and rollback boundaries.
- Keep focused CPU/CUDA tests, component gates, and formal benchmark evidence
  distinct. A reviewer SHIP is a code-review verdict, not a numerical pass.
- Recompute the component manifest before invalidating its evidence. Do not
  rerun nx4/nx8 for changes outside its source identity. Formal source/config/host
  identities and the clean-source gate still apply independently.
- Never reuse an occupied output label, even when its directory is empty.
  Failed prefixes and reduced field/history dumps are not restart checkpoints.
- Follow the selected campaign's registered order. For the current Turek-Hron
  R26A campaign, this is FSI1-S0, M0/M1, conditional F0, FSI2, then FSI3;
  Oracle/learning requires its benchmark quality gates first. Other campaigns
  retain their own acceptance and research-entry contracts.
- A user pause or quota stop ends the owned long work and requires a handoff.
  An automatic continuation cannot resume it; a later user instruction can.

## Structure and working-tree discipline

- `simulation_core/`: reusable solver implementation; `cases/`: case definitions.
- `benchmarks/official/`: benchmark runners; `tools/`: diagnostics/validation.
- `tests/`: focused tests; `scripts/`: maintenance; `docs/`: contracts/handoffs.
- Preserve unrelated modified/untracked files; never reset/clean or delete
  evidence for tidiness. Commit and publication require their actual authorization.
- Exclude `.git`, `.claude/worktrees`, `validation_runs`, `tmp`, `archive`, and
  caches from broad scans. Use `validation_runs/README.md` to choose artifacts.
- Use `rg` when available; WSL grep/find or Python and PowerShell Select-String
  are valid fallbacks. Do not repeatedly invoke a missing or denied rg binary.
- Do not import historical code from `archive/` or `validation_runs/`.
- Python files use descriptive snake_case. New reports/handoffs follow the
  existing topic/purpose/date convention; keep run path segments under 80 characters.
