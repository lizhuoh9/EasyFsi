from __future__ import annotations

from .history import _final_row_int, _row_bool

def run_process_completion_status(
    *,
    validation_scope_complete: bool,
    validation_passed: bool | None,
    partial_run_stopped: bool,
    requested_steps: int,
    completed_steps: int,
) -> str:
    if bool(partial_run_stopped) or int(completed_steps) < int(requested_steps):
        return "partial"
    if bool(validation_scope_complete):
        return "finished" if bool(validation_passed) else "validation_failed"
    return "finished"
