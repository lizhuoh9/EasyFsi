from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cases" / "ansys_vertical_flap_fsi.py").exists():
            return parent
    raise RuntimeError("could not locate repo root")


def main() -> int:
    repo = _repo_root()
    run_root = (
        repo
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "our_solver_fine_vs_fluent_2026-07-02"
    )
    output_dir = run_root / "our_solver" / "production"
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "production_stdout.log"
    stderr_path = output_dir / "production_stderr.log"
    run_script = run_root / "scripts" / "run_our_solver_vertical_flap.py"
    python = Path(r"D:\working\taichi\env\python.exe")
    args = [
        str(python),
        str(run_script),
        "--output-dir",
        str(output_dir),
        "--run-label",
        "production_fine_grid4x256x320",
        "--steps",
        "50",
        "--grid-nodes",
        "4",
        "256",
        "320",
        # 2026-07-03 seeding audit: (1, 64, 12) leaves ~2 background cells
        # between wall-normal particle layers on the 4x256x320 grid; the MPM
        # solid fractures at the root and ejects particles. (1, 256, 20)
        # keeps particle spacing at ~0.5 cells and rings stably about the
        # Euler-Bernoulli static deflection.
        "--solid-particle-counts",
        "1",
        "256",
        "20",
        "--marker-count",
        "64",
        "--flow-projection-iterations",
        "1080",
        "--solid-substeps",
        "1600",
    ]
    creationflags = 0
    stdout = stdout_path.open("wb")
    stderr = stderr_path.open("wb")
    env = os.environ.copy()
    env.setdefault("TI_OFFLINE_CACHE", "0")
    process = subprocess.Popen(
        args,
        cwd=str(repo),
        env=env,
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    stdout.close()
    stderr.close()
    time.sleep(10.0)
    initial_returncode = process.poll()
    launch = {
        "pid": process.pid,
        "initial_poll_after_s": 10.0,
        "initial_returncode": initial_returncode,
        "repo_root": str(repo),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": args,
        "note": "Child process writes run_manifest.json after imports/config construction and our_solver_summary.json on completion.",
    }
    (output_dir / "production_launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(launch, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
