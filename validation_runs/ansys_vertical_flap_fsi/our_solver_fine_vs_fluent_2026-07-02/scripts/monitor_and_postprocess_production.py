from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ANSYS_PYTHON = Path(
    r"C:\Program Files\ANSYS Inc\v251\commonfiles\CPython\3_10\winx64\Release\python\python.exe"
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cases" / "ansys_vertical_flap_fsi.py").exists():
            return parent
    raise RuntimeError("could not locate repo root")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _status_path(run_root: Path) -> Path:
    return run_root / "comparison" / "production_monitor_status.json"


def _postprocess(run_root: Path, repo: Path) -> int:
    script = run_root / "scripts" / "export_fluent_and_compare.py"
    production = run_root / "our_solver" / "production"
    fluent_root = (
        repo
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "official_fluent_fine_mesh_steady_2026-07-01"
        / "fsi_50step_serial_from_adapt_cycle3_mesh"
    )
    stdout_path = run_root / "comparison" / "production_postprocess_stdout.log"
    stderr_path = run_root / "comparison" / "production_postprocess_stderr.log"
    args = [
        str(ANSYS_PYTHON),
        str(script),
        "--fluent-case",
        str(fluent_root / "fine_fsi_50step_final.cas.h5"),
        "--fluent-data",
        str(fluent_root / "fine_fsi_50step_final.dat.h5"),
        "--solver-npz",
        str(production / "our_solver_final_fields.npz"),
        "--output-root",
        str(run_root),
    ]
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(args, cwd=str(repo), stdout=stdout, stderr=stderr)
    return int(completed.returncode)


def monitor(args: argparse.Namespace) -> int:
    repo = _repo_root()
    run_root = (
        repo
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "our_solver_fine_vs_fluent_2026-07-02"
    )
    production = run_root / "our_solver" / "production"
    summary = production / "our_solver_summary.json"
    fields = production / "our_solver_final_fields.npz"
    failure = production / "failure.json"
    start = time.time()
    _write_json(
        _status_path(run_root),
        {
            "status": "watching",
            "started_at_epoch_s": start,
            "poll_interval_s": float(args.poll_interval_s),
            "max_wait_s": float(args.max_wait_s),
            "summary": str(summary),
            "fields": str(fields),
            "failure": str(failure),
        },
    )
    while True:
        elapsed = time.time() - start
        if failure.exists():
            _write_json(
                _status_path(run_root),
                {
                    "status": "production_failed",
                    "elapsed_s": elapsed,
                    "failure": str(failure),
                },
            )
            return 1
        if summary.exists() and fields.exists():
            rc = _postprocess(run_root, repo)
            _write_json(
                _status_path(run_root),
                {
                    "status": "postprocess_completed" if rc == 0 else "postprocess_failed",
                    "elapsed_s": elapsed,
                    "postprocess_exit_code": rc,
                    "summary": str(summary),
                    "fields": str(fields),
                    "metrics": str(run_root / "comparison" / "comparison_metrics.json"),
                },
            )
            return rc
        if elapsed > float(args.max_wait_s):
            _write_json(
                _status_path(run_root),
                {
                    "status": "timeout_waiting_for_production",
                    "elapsed_s": elapsed,
                    "summary_exists": summary.exists(),
                    "fields_exists": fields.exists(),
                    "failure_exists": failure.exists(),
                },
            )
            return 2
        time.sleep(float(args.poll_interval_s))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-interval-s", type=float, default=60.0)
    parser.add_argument("--max-wait-s", type=float, default=7.0 * 24.0 * 3600.0)
    args = parser.parse_args()
    return monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
