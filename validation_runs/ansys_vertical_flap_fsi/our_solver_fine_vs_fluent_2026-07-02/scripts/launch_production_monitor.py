from __future__ import annotations

import json
import subprocess
import sys
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
    script = run_root / "scripts" / "monitor_and_postprocess_production.py"
    output_dir = run_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "production_monitor_stdout.log"
    stderr_path = output_dir / "production_monitor_stderr.log"
    args = [
        sys.executable,
        str(script),
        "--poll-interval-s",
        "60",
        "--max-wait-s",
        str(7 * 24 * 3600),
    ]
    stdout = stdout_path.open("wb")
    stderr = stderr_path.open("wb")
    process = subprocess.Popen(
        args,
        cwd=str(repo),
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
    )
    stdout.close()
    stderr.close()
    launch = {
        "pid": process.pid,
        "command": args,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "status": str(output_dir / "production_monitor_status.json"),
    }
    (output_dir / "production_monitor_launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(launch, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
