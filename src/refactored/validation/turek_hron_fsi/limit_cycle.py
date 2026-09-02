"""Deterministic, solver-free Featflow limit-cycle analysis."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


class LimitCycleValidationError(ValueError):
    """Raised when a series cannot provide deterministic cycle evidence."""


@dataclass(frozen=True)
class SignalExtrema:
    midrange: float
    amplitude: float


@dataclass(frozen=True)
class CompleteCycle:
    start_s: float
    end_s: float
    period_s: float
    signals: Mapping[str, SignalExtrema]


@dataclass(frozen=True)
class LimitCycleStability:
    period_spread_rel: float
    point_a_uy_amplitude_spread_rel: float
    point_a_uy_midrange_range_rel: float
    total_drag_amplitude_growth_rel: float
    total_lift_amplitude_growth_rel: float
    checks: Mapping[str, bool]
    passed: bool


@dataclass(frozen=True)
class LimitCycleReport:
    crossing_times_s: tuple[float, ...]
    cycles: tuple[CompleteCycle, ...]
    primary_period_s: float
    primary_frequency_hz: float
    spectral_frequency_hz: Mapping[str, float]
    stability: LimitCycleStability


def _array(
    values: Sequence[float] | np.ndarray, field: str, length: int | None = None
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LimitCycleValidationError(f"{field} must be numeric") from exc
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise LimitCycleValidationError(
            f"{field} must be a one-dimensional series of matching length"
        )
    if not len(result) or not np.all(np.isfinite(result)):
        raise LimitCycleValidationError(f"{field} must contain finite values")
    return result


def _time(time_s: Sequence[float] | np.ndarray) -> tuple[np.ndarray, float]:
    time = _array(time_s, "time_s")
    if len(time) < 2:
        raise LimitCycleValidationError("time_s needs at least two samples")
    differences = np.diff(time)
    if np.any(differences <= 0.0):
        raise LimitCycleValidationError("time_s must be strictly increasing")
    dt_s = float(differences[0])
    if not np.allclose(
        differences, dt_s, rtol=1.0e-9, atol=max(1.0e-12, dt_s * 1.0e-9)
    ):
        raise LimitCycleValidationError("time_s must be uniform")
    return time, dt_s


def _rising_crossings(time_s: np.ndarray, uy: np.ndarray) -> tuple[float, ...]:
    crossings: list[float] = []
    last_negative: int | None = None
    index = 0
    while index < len(uy):
        value = uy[index]
        if value < 0.0:
            last_negative = index
        elif value == 0.0:
            first_zero = index
            while index + 1 < len(uy) and uy[index + 1] == 0.0:
                index += 1
            next_index = index + 1
            if last_negative is not None and next_index < len(uy) and uy[next_index] > 0.0:
                crossings.append(float(time_s[first_zero]))
                last_negative = None
        elif last_negative is not None:
            previous = last_negative
            fraction = -uy[previous] / (uy[index] - uy[previous])
            crossings.append(
                float(time_s[previous] + fraction * (time_s[index] - time_s[previous]))
            )
            last_negative = None
        index += 1
    return tuple(crossings)


def _cycle(
    time_s: np.ndarray, signals: Mapping[str, np.ndarray], start_s: float, end_s: float
) -> CompleteCycle:
    selected = (time_s >= start_s) & (time_s <= end_s)
    if not np.any(selected):
        raise LimitCycleValidationError("crossing-bounded cycle contains no samples")
    metrics = {
        name: SignalExtrema(
            midrange=float((np.max(values[selected]) + np.min(values[selected])) / 2.0),
            amplitude=float((np.max(values[selected]) - np.min(values[selected])) / 2.0),
        )
        for name, values in signals.items()
    }
    return CompleteCycle(start_s, end_s, end_s - start_s, MappingProxyType(metrics))


def _fft_frequency(
    time_s: np.ndarray, values: np.ndarray, start_s: float, end_s: float, dt_s: float
) -> float:
    count = int(np.floor((end_s - start_s) / dt_s + 1.0e-9))
    sample_times = start_s + np.arange(count + 1, dtype=np.float64) * dt_s
    samples = np.interp(sample_times, time_s, values)
    magnitudes = np.abs(np.fft.rfft(samples - np.mean(samples)))
    frequencies = np.fft.rfftfreq(len(samples), d=dt_s)
    return float(frequencies[1 + int(np.argmax(magnitudes[1:]))]) if len(magnitudes) > 1 else 0.0


def _stability(cycles: tuple[CompleteCycle, ...]) -> LimitCycleStability:
    periods = np.asarray([cycle.period_s for cycle in cycles])
    uy_amplitudes = np.asarray([cycle.signals["point_a_uy_m"].amplitude for cycle in cycles])
    uy_midranges = np.asarray([cycle.signals["point_a_uy_m"].midrange for cycle in cycles])
    drag_amplitudes = np.asarray([cycle.signals["total_drag_n"].amplitude for cycle in cycles])
    lift_amplitudes = np.asarray([cycle.signals["total_lift_n"].amplitude for cycle in cycles])
    mean_period, mean_uy = float(np.mean(periods)), float(np.mean(uy_amplitudes))
    if mean_period <= 0.0 or mean_uy <= 0.0:
        raise LimitCycleValidationError("cycle period and Point A uy amplitude must be positive")

    def growth(values: np.ndarray) -> float:
        return float((values[-1] - values[0]) / values[0]) if values[0] else float("inf")

    period_spread = float((np.max(periods) - np.min(periods)) / mean_period)
    amplitude_spread = float((np.max(uy_amplitudes) - np.min(uy_amplitudes)) / mean_uy)
    midrange_range = float((np.max(uy_midranges) - np.min(uy_midranges)) / mean_uy)
    drag_growth, lift_growth = growth(drag_amplitudes), growth(lift_amplitudes)
    checks = {
        "period_spread": period_spread < 0.01,
        "point_a_uy_amplitude_spread": amplitude_spread < 0.02,
        "point_a_uy_midrange_range": midrange_range < 0.01,
        "total_drag_amplitude_growth": drag_growth <= 0.02,
        "total_lift_amplitude_growth": lift_growth <= 0.02,
        "total_drag_not_strictly_increasing": not bool(np.all(np.diff(drag_amplitudes) > 0.0)),
        "total_lift_not_strictly_increasing": not bool(np.all(np.diff(lift_amplitudes) > 0.0)),
    }
    return LimitCycleStability(
        period_spread, amplitude_spread, midrange_range, drag_growth, lift_growth,
        MappingProxyType(checks), all(checks.values()),
    )


def analyze_limit_cycle(
    time_s: Sequence[float] | np.ndarray,
    point_a_ux_m: Sequence[float] | np.ndarray,
    point_a_uy_m: Sequence[float] | np.ndarray,
    total_drag_n: Sequence[float] | np.ndarray,
    total_lift_n: Sequence[float] | np.ndarray,
) -> LimitCycleReport:
    """Use the last three complete Point-A-uy rising-crossing bounded cycles."""

    time, dt_s = _time(time_s)
    signals = {
        "point_a_ux_m": _array(point_a_ux_m, "point_a_ux_m", len(time)),
        "point_a_uy_m": _array(point_a_uy_m, "point_a_uy_m", len(time)),
        "total_drag_n": _array(total_drag_n, "total_drag_n", len(time)),
        "total_lift_n": _array(total_lift_n, "total_lift_n", len(time)),
    }
    crossings = _rising_crossings(time, signals["point_a_uy_m"])
    if len(crossings) < 4:
        raise LimitCycleValidationError("at least four rising crossings are required")
    bounds = crossings[-4:]
    cycles = tuple(_cycle(time, signals, start, end) for start, end in zip(bounds[:-1], bounds[1:]))
    spectra = MappingProxyType(
        {name: _fft_frequency(time, values, bounds[0], bounds[-1], dt_s) for name, values in signals.items()}
    )
    period = cycles[-1].period_s
    return LimitCycleReport(crossings, cycles, period, 1.0 / period, spectra, _stability(cycles))


def analyze_featflow_limit_cycle(series: object) -> LimitCycleReport:
    """Analyze a validated Featflow-series-like object without solver imports."""

    return analyze_limit_cycle(
        series.time_s, series.point_a_ux_m, series.point_a_uy_m,
        series.total_drag_n, series.total_lift_n,
    )
