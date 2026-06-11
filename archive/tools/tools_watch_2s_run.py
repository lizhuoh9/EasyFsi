"""Watch the resumed 2-second run: emit progress every 200 steps, exit on terminal."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

RUN = Path(r"_codex_validation/hibm_sharp_run_20260611_2s_full_waveform")
LOG = RUN / "run_console_resume.log"
PROCESS = RUN / "run_process.json"

last_emit = -1
while True:
    status = ""
    try:
        status = str(json.loads(PROCESS.read_text()).get("status", ""))
    except Exception:
        pass
    last_line = ""
    try:
        for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("step="):
                last_line = line
    except Exception:
        pass
    if status in {"failed", "finished"}:
        print(f"TERMINAL: {status} | last: {last_line}", flush=True)
        try:
            tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
            for line in tail:
                if re.search(r"RuntimeError|Error|guard|breakdown", line):
                    print(line, flush=True)
        except Exception:
            pass
        break
    match = re.match(r"step=(\d+)", last_line)
    if match:
        step = int(match.group(1))
        if step % 200 == 0 and step != last_emit:
            print(f"progress: {last_line}", flush=True)
            last_emit = step
    time.sleep(120)
