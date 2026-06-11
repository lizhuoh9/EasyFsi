"""Launch the 2-second (4000-step) sharp squid run as a detached process."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path(r"D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM")
TAICHI_PYTHON = r"D:\working\taichi\env\python.exe"
OUTPUT_DIR = r"_codex_validation/hibm_sharp_run_20260611_2s_full_waveform"

command = [
    TAICHI_PYTHON,
    "cases/squid_soft_robot.py",
    "--source-config",
    r"_codex_validation/squid_case_run_render_20260610_005/simulation_config.json",
    "--output-dir",
    OUTPUT_DIR,
    "--fsi-coupling-mode",
    "hibm_mpm_sharp",
    "--solid-model",
    "neo_hookean_mpm",
    "--steps",
    "4000",
    "--arch",
    "cuda",
    "--pressure-solver",
    "fv_cg",
    "--projection-iterations",
    "512",
    "--fluid-substeps",
    "3",
    "--solid-mpm-substeps",
    "64",
    "--divergence-cleanup-iterations",
    "32",
    "--fluid-snapshot-interval",
    "25",
    "--checkpoint-every-step",
    "--max-wall-time-s",
    "43200",
    "--progress",
    "--progress-interval",
    "50",
]

log_path = PROJECT / OUTPUT_DIR / "run_console.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
with open(log_path, "w", encoding="utf-8") as log_file:
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )
print(f"launched pid={process.pid} log={log_path}")
sys.exit(0)
