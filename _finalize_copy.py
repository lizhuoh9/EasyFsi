"""One-shot helper: copy the unchanged big files from the parent tree into refactored/.

The refactored/ tree already contains every MODIFIED module (see
REFACTORING_NOTES.md). This script fills in the unchanged files so the copy is
complete and the test suite can run self-contained. Running it twice is safe:
existing files in refactored/ are never overwritten.

Usage:
    "D:/TOOL/Anaconda/python.exe" refactored/_finalize_copy.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

REFACTORED_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REFACTORED_ROOT.parent

UNCHANGED_FILES = (
    "simulation_core/fluid.py",
    "simulation_core/hibm_mpm.py",
    "simulation_core/tri_surface.py",
    "simulation_core/mooney_shell_mpm.py",
    "cases/squid_soft_robot.py",
    "cases/squid_jet_render.py",
    "cases/squid_current_visit_render.py",
    "inspect_latest_progress.py",
    "inspect_visit_stats.py",
    "summarize_preflight_log.py",
    # Documentation contracts pinned by tests/test_simulation_core_package.py.
    "SIMULATION_CORE_USAGE.md",
    "HIBM_MPM_PAPER_VS_CODE.md",
)

ARCHIVE_GLOBS = ("tools_*.py", "run_phase0_raw_map_scaling.py")


def main() -> None:
    copied: list[str] = []
    skipped: list[str] = []
    for relative in UNCHANGED_FILES:
        source = SOURCE_ROOT / relative
        target = REFACTORED_ROOT / relative
        if target.exists():
            skipped.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)

    tests_target = REFACTORED_ROOT / "tests"
    tests_target.mkdir(exist_ok=True)
    for test_file in sorted((SOURCE_ROOT / "tests").glob("*.py")):
        target = tests_target / test_file.name
        if target.exists():
            skipped.append(f"tests/{test_file.name}")
            continue
        shutil.copy2(test_file, target)
        copied.append(f"tests/{test_file.name}")

    archive_target = REFACTORED_ROOT / "archive" / "tools"
    archive_target.mkdir(parents=True, exist_ok=True)
    for pattern in ARCHIVE_GLOBS:
        for script in sorted(SOURCE_ROOT.glob(pattern)):
            target = archive_target / script.name
            if target.exists():
                skipped.append(f"archive/tools/{script.name}")
                continue
            shutil.copy2(script, target)
            copied.append(f"archive/tools/{script.name}")

    print(f"copied {len(copied)} files, skipped {len(skipped)} already-present files")
    for name in copied:
        print(f"  + {name}")


if __name__ == "__main__":
    main()
