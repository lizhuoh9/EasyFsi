from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-name", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--grid-nodes", type=int, nargs=3, required=True)
    parser.add_argument("--solid-particle-counts", type=int, nargs=3, required=True)
    parser.add_argument("--marker-count", type=int, required=True)
    parser.add_argument("--flow-projection-iterations", type=int, default=1080)
    parser.add_argument("--solid-substeps", type=int, default=1600)
    parser.add_argument("--initial-wait-s", type=float, default=30.0)
    args = parser.parse_args()

    repo = _repo_root()
    run_root = (
        repo
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "our_solver_fine_vs_fluent_2026-07-02"
    )
    output_dir = run_root / "our_solver" / args.run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "solver_stdout.log"
    stderr_path = output_dir / "solver_stderr.log"
    run_script = run_root / "scripts" / "run_our_solver_vertical_flap.py"
    python = Path(r"D:\working\taichi\env\python.exe")
    command = [
        str(python),
        str(run_script),
        "--output-dir",
        str(output_dir),
        "--run-label",
        str(args.run_label),
        "--steps",
        str(args.steps),
        "--grid-nodes",
        *(str(v) for v in args.grid_nodes),
        "--solid-particle-counts",
        *(str(v) for v in args.solid_particle_counts),
        "--marker-count",
        str(args.marker_count),
        "--flow-projection-iterations",
        str(args.flow_projection_iterations),
        "--solid-substeps",
        str(args.solid_substeps),
    ]
    env = os.environ.copy()
    env.setdefault("TI_OFFLINE_CACHE", "0")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(repo),
            env=env,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
        )
    time.sleep(float(args.initial_wait_s))
    launch = {
        "pid": process.pid,
        "initial_wait_s": float(args.initial_wait_s),
        "initial_returncode": process.poll(),
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "output_dir": str(output_dir),
        "expected_summary": str(output_dir / "our_solver_summary.json"),
        "expected_final_fields": str(output_dir / "our_solver_final_fields.npz"),
    }
    (output_dir / "launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(launch, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
