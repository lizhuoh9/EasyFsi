"""Extend an existing Fluent steady solution until physical monitors are stationary.

The runner is deliberately serial and checkpointed.  It opens one Fluent
session, advances a fixed number of steady iterations per block, writes a
case/data checkpoint, and evaluates windowed changes of pressure, velocity,
and SST fields from Fluent's HDF5 data.  A residual-only or iteration-count
claim is never reported as stationary.

An interrupted run can be continued with ``--resume``.  A terminal
``not_stationary`` run is immutable: use its final case/data as the input to a
new output directory if a larger iteration budget is required.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_fine_fsi_campaign as guarded


SCHEMA_VERSION = 2
GLOBAL_FLUENT_LOCK = CAMPAIGN_DIR / ".fluent_steady_extension.lock"
HDF5_MONITOR_FIELDS = (
    "pressure_min_pa",
    "pressure_max_pa",
    "pressure_mean_pa",
    "pressure_range_pa",
    "speed_max_mps",
    "speed_mean_mps",
    "k_mean_m2_s2",
    "k_p90_m2_s2",
    "k_max_m2_s2",
    "omega_mean_s_inv",
    "omega_p90_s_inv",
    "omega_max_s_inv",
    "mu_t_mean_pa_s",
    "mu_t_p90_pa_s",
    "mu_t_max_pa_s",
)
SURFACE_MONITOR_FIELDS = (
    "flap_fluid_force_x_n",
    "flap_fluid_force_y_n",
    "flap_fluid_force_z_n",
    "inlet_mass_flow_kg_s",
    "outlet_mass_flow_kg_s",
    "net_mass_flow_kg_s",
    "relative_mass_imbalance",
)
MONITOR_FIELDS = (*HDF5_MONITOR_FIELDS, *SURFACE_MONITOR_FIELDS)
TERMINAL_STATUSES = {"stationary", "not_stationary", "fixed_budget_complete"}


@dataclass(frozen=True)
class SteadyExtensionConfig:
    run_dir: Path
    source_case: Path
    source_data: Path
    processor_count: int = 1
    block_iterations: int = 100
    max_additional_iterations: int = 2_000
    window_blocks: int = 3
    minimum_windows: int = 3
    consecutive_windows: int = 3
    relative_tolerance: float = 0.01
    resume: bool = False
    dry_run: bool = False
    recover_stale_lock: bool = False
    force_full_budget: bool = False


def config_from_cli(argv: list[str] | None = None) -> SteadyExtensionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Extend an existing Fluent steady case/data in checkpointed blocks "
            "until physical HDF5 monitors become stationary."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--processor-count", type=int, default=1)
    parser.add_argument("--block-iterations", type=int, default=100)
    parser.add_argument("--max-additional-iterations", type=int, default=2_000)
    parser.add_argument("--window-blocks", type=int, default=3)
    parser.add_argument("--minimum-windows", type=int, default=3)
    parser.add_argument("--consecutive-windows", type=int, default=3)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help=(
            "Explicitly remove the global lock only after its recorded PID is "
            "confirmed dead; malformed or live-owner locks remain fail-closed."
        ),
    )
    parser.add_argument(
        "--force-full-budget",
        action="store_true",
        help=(
            "Run the complete additional-iteration budget. Stationarity is "
            "recorded as a diagnostic only and never used as a stop/success claim."
        ),
    )
    args = parser.parse_args(argv)
    return SteadyExtensionConfig(
        run_dir=args.run_dir.resolve(),
        source_case=args.source_case.resolve(),
        source_data=args.source_data.resolve(),
        processor_count=args.processor_count,
        block_iterations=args.block_iterations,
        max_additional_iterations=args.max_additional_iterations,
        window_blocks=args.window_blocks,
        minimum_windows=args.minimum_windows,
        consecutive_windows=args.consecutive_windows,
        relative_tolerance=args.relative_tolerance,
        resume=args.resume,
        dry_run=args.dry_run,
        recover_stale_lock=args.recover_stale_lock,
        force_full_budget=args.force_full_budget,
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def paired_data_path(case_path: Path) -> Path:
    if not case_path.name.endswith(".cas.h5"):
        raise ValueError(f"case checkpoint must end in .cas.h5: {case_path}")
    return case_path.with_name(case_path.name[: -len(".cas.h5")] + ".dat.h5")


def _datasets_below(node: h5py.Group | h5py.Dataset) -> Iterator[np.ndarray]:
    if isinstance(node, h5py.Dataset):
        yield np.asarray(node[()], dtype=float).ravel()
        return
    for child in node.values():
        yield from _datasets_below(child)


def _cell_field(cells: h5py.Group, name: str) -> np.ndarray:
    if name not in cells:
        raise KeyError(f"Fluent data has no required cell field {name}")
    partitions = [values for values in _datasets_below(cells[name]) if values.size]
    if not partitions:
        raise ValueError(f"Fluent cell field {name} is empty")
    values = np.concatenate(partitions)
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"Fluent cell field {name} contains non-finite values")
    return values


def _summary(values: np.ndarray, *, prefix: str, unit_suffix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_{unit_suffix}": float(np.mean(values)),
        f"{prefix}_p90_{unit_suffix}": float(np.percentile(values, 90.0)),
        f"{prefix}_max_{unit_suffix}": float(np.max(values)),
    }


def read_hdf5_monitor(data_path: Path) -> dict[str, float]:
    """Read physical steady-state monitors from a serial Fluent data file."""

    with h5py.File(data_path, "r") as handle:
        try:
            cells = handle["results/1/phase-1/cells"]
        except KeyError as exc:
            raise KeyError(
                f"Fluent data lacks results/1/phase-1/cells: {data_path}"
            ) from exc
        u = _cell_field(cells, "SV_U")
        v = _cell_field(cells, "SV_V")
        pressure = _cell_field(cells, "SV_P")
        k = _cell_field(cells, "SV_K")
        omega = _cell_field(cells, "SV_O")
        mu_t = _cell_field(cells, "SV_MU_T")

    if u.size != v.size:
        raise ValueError(f"Fluent SV_U/SV_V sizes differ: {u.size} != {v.size}")
    for name, values, strictly_positive in (
        ("SV_K", k, False),
        ("SV_O", omega, True),
        ("SV_MU_T", mu_t, False),
    ):
        valid = bool(np.all(values > 0.0)) if strictly_positive else bool(np.all(values >= 0.0))
        if not valid:
            constraint = "positive" if strictly_positive else "nonnegative"
            raise ValueError(f"Fluent {name} must be finite and {constraint}")

    speed = np.hypot(u, v)
    result = {
        "pressure_min_pa": float(np.min(pressure)),
        "pressure_max_pa": float(np.max(pressure)),
        "pressure_mean_pa": float(np.mean(pressure)),
        "pressure_range_pa": float(np.max(pressure) - np.min(pressure)),
        "speed_max_mps": float(np.max(speed)),
        "speed_mean_mps": float(np.mean(speed)),
        **_summary(k, prefix="k", unit_suffix="m2_s2"),
        **_summary(omega, prefix="omega", unit_suffix="s_inv"),
        **_summary(mu_t, prefix="mu_t", unit_suffix="pa_s"),
    }
    if set(result) != set(HDF5_MONITOR_FIELDS):
        raise AssertionError(f"internal monitor schema mismatch: {sorted(result)}")
    return result


def read_live_surface_monitor(session: Any) -> dict[str, float]:
    """Read the guarded Fluent force/mass integrals, failing closed on drift."""

    raw = guarded.read_surface_integrals(session)
    missing = [field for field in SURFACE_MONITOR_FIELDS if field not in raw]
    if missing:
        raise RuntimeError(
            "guarded Fluent surface-integral result lacks required fields: "
            + ", ".join(missing)
        )
    result = {field: float(raw[field]) for field in SURFACE_MONITOR_FIELDS}
    invalid = [field for field, value in result.items() if not math.isfinite(value)]
    if invalid:
        raise RuntimeError(
            "guarded Fluent surface-integral result has non-finite fields: "
            + ", ".join(invalid)
        )
    if result["relative_mass_imbalance"] < 0.0:
        raise RuntimeError("relative_mass_imbalance must be nonnegative")
    return result


def _relative_span(values: Iterable[float]) -> float:
    numeric = [float(value) for value in values]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        return math.inf
    scale = max(max(abs(value) for value in numeric), 1.0e-30)
    return (max(numeric) - min(numeric)) / scale


def _relative_trend(values: Iterable[float]) -> float:
    """Return the fitted end-to-end drift relative to the observed scale."""

    numeric = np.asarray([float(value) for value in values], dtype=float)
    if numeric.size == 0 or not bool(np.all(np.isfinite(numeric))):
        return math.inf
    if numeric.size == 1:
        return 0.0
    x = np.arange(numeric.size, dtype=float)
    centered_x = x - float(np.mean(x))
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        return 0.0
    slope = float(np.dot(centered_x, numeric - float(np.mean(numeric))) / denominator)
    projected_change = abs(slope) * float(numeric.size - 1)
    scale = max(float(np.max(np.abs(numeric))), 1.0e-30)
    return projected_change / scale


def evaluate_stationarity(
    monitors: list[Mapping[str, float]],
    *,
    window_blocks: int,
    minimum_windows: int,
    consecutive_windows: int,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Evaluate overlapping monitor windows; every field must pass every gate."""

    windows: list[dict[str, Any]] = []
    for start in range(max(0, len(monitors) - window_blocks + 1)):
        rows = monitors[start : start + window_blocks]
        relative_spans = {
            field: _relative_span(row[field] for row in rows)
            for field in MONITOR_FIELDS
        }
        failed = [
            field
            for field, value in relative_spans.items()
            if value > relative_tolerance
        ]
        windows = [
            *windows,
            {
                "window_index": start + 1,
                "start_block": start + 1,
                "end_block": start + window_blocks,
                "passed": not failed,
                "relative_spans": relative_spans,
                "failed_metrics": failed,
            },
        ]

    consecutive = 0
    for window in reversed(windows):
        if not window["passed"]:
            break
        consecutive += 1
    enough_windows = len(windows) >= minimum_windows
    enough_consecutive = consecutive >= consecutive_windows
    if enough_consecutive:
        accepted_windows = windows[-consecutive_windows:]
        first = accepted_windows[0]
        last = accepted_windows[-1]
        accepted_rows = monitors[
            int(first["start_block"]) - 1 : int(last["end_block"])
        ]
        cumulative_spans = {
            field: _relative_span(row[field] for row in accepted_rows)
            for field in MONITOR_FIELDS
        }
        cumulative_trends = {
            field: _relative_trend(row[field] for row in accepted_rows)
            for field in MONITOR_FIELDS
        }
        failed_span_metrics = [
            field
            for field, value in cumulative_spans.items()
            if value > relative_tolerance
        ]
        failed_trend_metrics = [
            field
            for field, value in cumulative_trends.items()
            if value > relative_tolerance
        ]
        accepted_interval = {
            "available": True,
            "start_block": first["start_block"],
            "end_block": last["end_block"],
            "row_count": len(accepted_rows),
            "passed": not failed_span_metrics and not failed_trend_metrics,
            "relative_spans": cumulative_spans,
            "relative_trends": cumulative_trends,
            "failed_span_metrics": failed_span_metrics,
            "failed_trend_metrics": failed_trend_metrics,
        }
    else:
        accepted_interval = {
            "available": False,
            "start_block": None,
            "end_block": None,
            "row_count": 0,
            "passed": False,
            "relative_spans": {},
            "relative_trends": {},
            "failed_span_metrics": list(MONITOR_FIELDS),
            "failed_trend_metrics": list(MONITOR_FIELDS),
        }
    stationary = enough_windows and enough_consecutive and accepted_interval["passed"]
    if not enough_windows:
        reason = "insufficient_windows"
    elif stationary:
        reason = "consecutive_and_cumulative_physical_gates_passed"
    elif enough_consecutive and not accepted_interval["passed"]:
        reason = "accepted_interval_cumulative_drift"
    else:
        reason = "physical_monitors_still_changing"
    latest = windows[-1] if windows else None
    return {
        "stationary": stationary,
        "reason": reason,
        "window_blocks": window_blocks,
        "minimum_windows": minimum_windows,
        "consecutive_windows_required": consecutive_windows,
        "relative_tolerance": relative_tolerance,
        "evaluated_window_count": len(windows),
        "consecutive_windows_passed": consecutive,
        "accepted_interval": accepted_interval,
        "latest_relative_spans": {} if latest is None else latest["relative_spans"],
        "latest_failed_metrics": list(MONITOR_FIELDS) if latest is None else latest["failed_metrics"],
        "windows": windows,
    }


def _contract(config: SteadyExtensionConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_case": str(config.source_case),
        "source_data": str(config.source_data),
        "processor_count": config.processor_count,
        "block_iterations": config.block_iterations,
        "max_additional_iterations": config.max_additional_iterations,
        "window_blocks": config.window_blocks,
        "minimum_windows": config.minimum_windows,
        "consecutive_windows": config.consecutive_windows,
        "relative_tolerance": config.relative_tolerance,
        "force_full_budget": config.force_full_budget,
        "monitor_fields": list(MONITOR_FIELDS),
        "hdf5_monitor_fields": list(HDF5_MONITOR_FIELDS),
        "surface_monitor_fields": list(SURFACE_MONITOR_FIELDS),
        "surface_integrals_policy": "guarded_read_surface_integrals_fail_closed",
    }


def validate_config(config: SteadyExtensionConfig) -> dict[str, Any]:
    if config.processor_count != 1:
        raise ValueError(
            "steady extension permits one Fluent process in serial; processor_count must be 1"
        )
    if config.block_iterations <= 0 or config.max_additional_iterations <= 0:
        raise ValueError("block and maximum additional iterations must be positive")
    if config.window_blocks < 2:
        raise ValueError("window_blocks must be at least 2")
    if config.minimum_windows <= 0 or config.consecutive_windows <= 0:
        raise ValueError("minimum and consecutive windows must be positive")
    if config.minimum_windows < config.consecutive_windows:
        raise ValueError("minimum_windows must be >= consecutive_windows")
    if not (0.0 < config.relative_tolerance < 1.0):
        raise ValueError("relative_tolerance must be in (0, 1)")
    if not config.source_case.is_file() or not config.source_data.is_file():
        raise FileNotFoundError(
            f"source case/data must exist: {config.source_case}, {config.source_data}"
        )
    if not config.source_case.name.endswith(".cas.h5"):
        raise ValueError("source case must end in .cas.h5")
    if not config.source_data.name.endswith(".dat.h5"):
        raise ValueError("source data must end in .dat.h5")
    if config.resume:
        if not config.run_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {config.run_dir}")
    elif config.run_dir.exists():
        raise FileExistsError(
            f"output directory already exists; use --resume only for its exact contract: {config.run_dir}"
        )
    return _contract(config)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stationarity_for_history(
    history: list[Mapping[str, Any]], config: SteadyExtensionConfig
) -> dict[str, Any]:
    result = evaluate_stationarity(
        [row["monitor"] for row in history],
        window_blocks=config.window_blocks,
        minimum_windows=config.minimum_windows,
        consecutive_windows=config.consecutive_windows,
        relative_tolerance=config.relative_tolerance,
    )
    if not config.force_full_budget:
        return result
    return {
        **result,
        "diagnostic_stationary": bool(result["stationary"]),
        "diagnostic_reason": result["reason"],
        "stationary": False,
        "reason": "force_full_budget_diagnostic_only",
    }


def _report_from_progress(
    progress: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    history = list(progress.get("history", []))
    latest = dict(progress["latest_checkpoint"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": progress["status"],
        "contract": dict(contract),
        "source_monitor": progress["source_monitor"],
        "completed_blocks": progress["completed_blocks"],
        "completed_iterations": progress["completed_iterations"],
        "stationarity": progress["stationarity"],
        "history": history,
        "final_monitor": progress["source_monitor"] if not history else history[-1]["monitor"],
        "final_case": latest["case_path"],
        "final_data": latest["data_path"],
        "updated_unix_s": progress["updated_unix_s"],
    }


def _persist_state(
    run_dir: Path, progress: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    report = _report_from_progress(progress, contract)
    atomic_write_json(run_dir / "progress.json", progress)
    atomic_write_json(run_dir / "report.json", report)
    return report


def _new_run_state(
    config: SteadyExtensionConfig, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config.run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = config.run_dir / "checkpoints"
    baseline_case = checkpoint_dir / "block_0000.cas.h5"
    baseline_data = checkpoint_dir / "block_0000.dat.h5"
    _copy_atomic(config.source_case, baseline_case)
    _copy_atomic(config.source_data, baseline_data)
    source_monitor = read_hdf5_monitor(baseline_data)
    stationarity = _stationarity_for_history([], config)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "completed_blocks": 0,
        "completed_iterations": 0,
        "latest_checkpoint": {
            "case_path": str(baseline_case),
            "data_path": str(baseline_data),
        },
        "source_monitor": source_monitor,
        "stationarity": stationarity,
        "history": [],
        "updated_unix_s": time.time(),
    }
    atomic_write_json(config.run_dir / "run_contract.json", contract)
    _persist_state(config.run_dir, progress, contract)
    return progress, dict(contract)


def _resume_state(
    config: SteadyExtensionConfig, expected_contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = config.run_dir / "run_contract.json"
    progress_path = config.run_dir / "progress.json"
    if not contract_path.is_file() or not progress_path.is_file():
        raise RuntimeError("resume directory lacks run_contract.json or progress.json")
    contract = _load_json(contract_path)
    if contract != dict(expected_contract):
        raise RuntimeError("resume contract differs from the original immutable contract")
    progress = _load_json(progress_path)
    if progress.get("status") in TERMINAL_STATUSES:
        return progress, contract
    latest = progress.get("latest_checkpoint", {})
    for key in ("case_path", "data_path"):
        if not Path(latest.get(key, "")).is_file():
            raise RuntimeError(f"resume checkpoint is missing: {key}={latest.get(key)!r}")
    return progress, contract


def _process_is_alive(pid: int) -> bool:
    """Conservatively test a recorded lock PID without terminating it."""

    if isinstance(pid, bool) or pid <= 0:
        raise ValueError(f"lock PID must be a positive integer, got {pid!r}")
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        # Unknown platform/permission failures cannot prove the owner is dead.
        return True
    return True


def _windows_process_is_alive(pid: int) -> bool:
    """Probe a Windows PID via OpenProcess/GetExitCodeProcess, never os.kill."""

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_invalid_parameter = 87
    error_not_found = 1168
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        error = ctypes.get_last_error()
        if error in {error_invalid_parameter, error_not_found}:
            return False
        # Access denied and unknown failures do not prove that the PID is dead.
        return True
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _recover_confirmed_stale_lock() -> None:
    """Remove the lock only when its unchanged recorded owner is proven dead."""

    try:
        original = GLOBAL_FLUENT_LOCK.read_bytes()
    except FileNotFoundError:
        return
    try:
        owner = json.loads(original.decode("utf-8"))
        pid = owner["pid"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"cannot recover malformed fail-closed Fluent lock: {GLOBAL_FLUENT_LOCK}"
        ) from exc
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError(
            f"cannot verify owner PID in fail-closed Fluent lock: {GLOBAL_FLUENT_LOCK}"
        )
    if _process_is_alive(pid):
        raise RuntimeError(
            "another Fluent extension owns the fail-closed lock: "
            f"{GLOBAL_FLUENT_LOCK} (live pid={pid})"
        )
    try:
        current = GLOBAL_FLUENT_LOCK.read_bytes()
    except FileNotFoundError:
        return
    if current != original:
        raise RuntimeError(
            f"Fluent lock owner changed during stale recovery: {GLOBAL_FLUENT_LOCK}"
        )
    GLOBAL_FLUENT_LOCK.unlink()


@contextmanager
def _single_fluent_process(run_dir: Path, *, recover_stale_lock: bool = False):
    GLOBAL_FLUENT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "run_dir": str(run_dir),
            "created_unix_s": time.time(),
            "lock_id": uuid.uuid4().hex,
        },
        sort_keys=True,
    ).encode("utf-8")
    descriptor = -1
    recovered = False
    while descriptor < 0:
        try:
            descriptor = os.open(
                GLOBAL_FLUENT_LOCK,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            if not recover_stale_lock or recovered:
                raise RuntimeError(
                    "another Fluent extension owns the fail-closed lock: "
                    f"{GLOBAL_FLUENT_LOCK}"
                ) from exc
            _recover_confirmed_stale_lock()
            recovered = True
    try:
        os.write(descriptor, payload)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            owned_payload = GLOBAL_FLUENT_LOCK.read_bytes()
        except FileNotFoundError:
            owned_payload = None
        if owned_payload == payload:
            GLOBAL_FLUENT_LOCK.unlink(missing_ok=True)


def _checkpoint_paths(run_dir: Path, block: int) -> tuple[Path, Path, Path, Path]:
    checkpoint_dir = run_dir / "checkpoints"
    final_case = checkpoint_dir / f"block_{block:04d}.cas.h5"
    final_data = paired_data_path(final_case)
    temporary_case = checkpoint_dir / f"block_{block:04d}.tmp.cas.h5"
    temporary_data = paired_data_path(temporary_case)
    return temporary_case, temporary_data, final_case, final_data


def _write_checkpoint(session: Any, run_dir: Path, block: int) -> tuple[Path, Path, dict[str, float]]:
    temporary_case, temporary_data, final_case, final_data = _checkpoint_paths(
        run_dir, block
    )
    temporary_case.unlink(missing_ok=True)
    temporary_data.unlink(missing_ok=True)
    session.file.write_case_data(file_name=str(temporary_case))
    if not temporary_case.is_file() or not temporary_data.is_file():
        raise RuntimeError(
            "Fluent did not create the complete temporary case/data checkpoint pair"
        )
    hdf5_monitor = read_hdf5_monitor(temporary_data)
    surface_monitor = read_live_surface_monitor(session)
    monitor = {**hdf5_monitor, **surface_monitor}
    if set(monitor) != set(MONITOR_FIELDS):
        raise RuntimeError(
            "combined Fluent physical monitor schema differs from the locked contract"
        )
    temporary_case.replace(final_case)
    temporary_data.replace(final_data)
    return final_case, final_data, monitor


def _failure_payload(
    exc: BaseException, progress: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "completed_blocks": 0 if progress is None else progress.get("completed_blocks", 0),
        "completed_iterations": 0 if progress is None else progress.get("completed_iterations", 0),
        "failed_unix_s": time.time(),
    }


def _cleanup_failure_artifact(run_dir: Path) -> None:
    (run_dir / "failure.json").unlink(missing_ok=True)


def run_extension(config: SteadyExtensionConfig) -> dict[str, Any]:
    """Acquire the global lock before any run-directory validation or state I/O."""

    with _single_fluent_process(
        config.run_dir,
        recover_stale_lock=config.recover_stale_lock,
    ):
        return _run_extension_locked(config)


def _run_extension_locked(config: SteadyExtensionConfig) -> dict[str, Any]:
    expected_contract = validate_config(config)
    if config.dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run",
            "contract": expected_contract,
            "source_monitor": read_hdf5_monitor(config.source_data),
            "would_create": str(config.run_dir),
            "fluent_launched": False,
        }

    progress: dict[str, Any] | None = None
    contract: dict[str, Any] = dict(expected_contract)
    session = None
    try:
        if config.resume:
            progress, contract = _resume_state(config, expected_contract)
            if progress["status"] in TERMINAL_STATUSES:
                report = _report_from_progress(progress, contract)
                _cleanup_failure_artifact(config.run_dir)
                return report
            progress = {
                key: value
                for key, value in progress.items()
                if key != "failure"
            }
            progress = {
                **progress,
                "status": "running",
                "updated_unix_s": time.time(),
            }
            _persist_state(config.run_dir, progress, contract)
        else:
            progress, contract = _new_run_state(config, expected_contract)

        session = guarded.launch_fluent(
            config.run_dir, processor_count=config.processor_count
        )
        load_cursor = guarded.transcript_cursor(config.run_dir)
        session.file.read_case_data(
            file_name=progress["latest_checkpoint"]["case_path"]
        )
        load_text, _ = guarded.transcript_delta(config.run_dir, load_cursor)
        guarded.require_clean_transcript(load_text, "steady extension checkpoint load")

        while progress["completed_iterations"] < config.max_additional_iterations:
            block = int(progress["completed_blocks"]) + 1
            remaining = (
                config.max_additional_iterations
                - int(progress["completed_iterations"])
            )
            iteration_count = min(config.block_iterations, remaining)
            started = time.time()
            cursor = guarded.transcript_cursor(config.run_dir)
            session.solution.run_calculation.iterate(iter_count=iteration_count)
            transcript_text, _ = guarded.transcript_delta(config.run_dir, cursor)
            guarded.require_clean_transcript(
                transcript_text, f"steady extension block {block}"
            )
            case_path, data_path, monitor = _write_checkpoint(
                session, config.run_dir, block
            )
            history = [
                *progress["history"],
                {
                    "block": block,
                    "iterations_this_block": iteration_count,
                    "completed_iterations": int(progress["completed_iterations"])
                    + iteration_count,
                    "seconds": time.time() - started,
                    "case_path": str(case_path),
                    "data_path": str(data_path),
                    "monitor": monitor,
                },
            ]
            stationarity = _stationarity_for_history(history, config)
            completed_iterations = int(progress["completed_iterations"]) + iteration_count
            if config.force_full_budget:
                status = (
                    "fixed_budget_complete"
                    if completed_iterations >= config.max_additional_iterations
                    else "running"
                )
            elif stationarity["stationary"]:
                status = "stationary"
            elif completed_iterations >= config.max_additional_iterations:
                status = "not_stationary"
            else:
                status = "running"
            progress = {
                **progress,
                "status": status,
                "completed_blocks": block,
                "completed_iterations": completed_iterations,
                "latest_checkpoint": {
                    "case_path": str(case_path),
                    "data_path": str(data_path),
                },
                "stationarity": stationarity,
                "history": history,
                "updated_unix_s": time.time(),
            }
            report = _persist_state(config.run_dir, progress, contract)
            if status in TERMINAL_STATUSES:
                if config.resume:
                    _cleanup_failure_artifact(config.run_dir)
                return report

        raise AssertionError("steady extension loop ended without a terminal report")
    except Exception as exc:
        failure = _failure_payload(exc, progress)
        if config.run_dir.is_dir():
            atomic_write_json(config.run_dir / "failure.json", failure)
            if progress is not None:
                progress = {
                    **progress,
                    "status": "failed",
                    "failure": failure,
                    "updated_unix_s": time.time(),
                }
                _persist_state(config.run_dir, progress, contract)
            else:
                atomic_write_json(config.run_dir / "report.json", failure)
        raise
    finally:
        if session is not None:
            session.exit()
        if config.run_dir.is_dir():
            transcript = guarded.copy_latest_transcript(config.run_dir)
            if transcript and (config.run_dir / "report.json").is_file():
                report = _load_json(config.run_dir / "report.json")
                atomic_write_json(
                    config.run_dir / "report.json",
                    {**report, "transcript": transcript},
                )


def main(argv: list[str] | None = None) -> int:
    config = config_from_cli(argv)
    report = run_extension(config)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
