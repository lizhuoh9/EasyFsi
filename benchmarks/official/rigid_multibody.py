from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HingeRotationSchedule:
    start_time_s: float
    end_time_s: float
    start_angle_rad: float
    end_angle_rad: float
    profile: str = "linear"

    def __post_init__(self) -> None:
        if self.end_time_s <= self.start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")
        if self.profile not in {"linear", "quarter-sine"}:
            raise ValueError("unsupported hinge rotation profile")

    def angle_at(self, time_s: float) -> float:
        time = float(time_s)
        if time <= self.start_time_s:
            return float(self.start_angle_rad)
        if time >= self.end_time_s:
            return float(self.end_angle_rad)
        phase = (time - self.start_time_s) / (self.end_time_s - self.start_time_s)
        if self.profile == "quarter-sine":
            phase = math.sin(0.5 * math.pi * phase)
        return float(
            self.start_angle_rad
            + phase * (self.end_angle_rad - self.start_angle_rad)
        )


@dataclass(frozen=True)
class EqualOppositeHingePair:
    positive_name: str
    negative_name: str
    schedule_end_angle_rad: float
    active_until_s: float
    profile: str = "linear"

    def __post_init__(self) -> None:
        if not self.positive_name or not self.negative_name:
            raise ValueError("hinge names must be non-empty")
        if self.positive_name == self.negative_name:
            raise ValueError("hinge names must be different")
        if self.active_until_s <= 0.0:
            raise ValueError("active_until_s must be positive")

    def angles_at(self, time_s: float) -> dict[str, float]:
        schedule = HingeRotationSchedule(
            start_time_s=0.0,
            end_time_s=self.active_until_s,
            start_angle_rad=0.0,
            end_angle_rad=self.schedule_end_angle_rad,
            profile=self.profile,
        )
        angle = schedule.angle_at(time_s)
        return {
            self.positive_name: angle,
            self.negative_name: -angle,
        }
