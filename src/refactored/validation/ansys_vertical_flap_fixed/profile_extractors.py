from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_COLUMNS = [
    "profile",
    "offset",
    "s",
    "y",
    "u",
    "Uz",
    "Uy",
    "speed",
    "fluid_mask",
    "near_solid_mask",
]


def extract_centerline_profile(fields: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    y = fields["y"]
    row = int(np.argmin(np.abs(y)))
    return [_row(fields, row, column, "centerline", 0.0) for column in range(len(fields["s"]))]


def extract_throat_profile(
    fields: dict[str, np.ndarray], flap_center_s: float = 0.048
) -> list[dict[str, Any]]:
    s = fields["s"]
    column = int(np.argmin(np.abs(s - flap_center_s)))
    return [_row(fields, row, column, "throat", 0.0) for row in range(len(fields["y"]))]


def extract_downstream_profiles(
    fields: dict[str, np.ndarray],
    offsets: tuple[float, ...] = (0.004, 0.010, 0.020, 0.040),
    flap_center_s: float = 0.048,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    s = fields["s"]
    for offset in offsets:
        column = int(np.argmin(np.abs(s - (flap_center_s + offset))))
        for row in range(len(fields["y"])):
            rows.append(_row(fields, row, column, f"downstream_{offset:.3f}", offset))
    return rows


def summarize_profiles(
    centerline_rows: list[dict[str, Any]],
    throat_rows: list[dict[str, Any]],
    downstream_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    centerline_fluid = [row for row in centerline_rows if row["fluid_mask"]]
    throat_fluid = [row for row in throat_rows if row["fluid_mask"]]
    downstream_by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in downstream_rows:
        if row["fluid_mask"]:
            downstream_by_profile.setdefault(str(row["profile"]), []).append(row)

    centerline_max = max(centerline_fluid, key=lambda row: row["u"])
    throat_max = max(throat_fluid, key=lambda row: row["u"])
    downstream_peaks = {
        profile: max(rows, key=lambda row: row["u"])["u"]
        for profile, rows in downstream_by_profile.items()
        if rows
    }
    return {
        "centerline_max_u": float(centerline_max["u"]),
        "centerline_max_s": float(centerline_max["s"]),
        "throat_max_u": float(throat_max["u"]),
        "throat_mean_u": float(np.mean([row["u"] for row in throat_fluid])),
        "downstream_profile_peaks": downstream_peaks,
    }


def write_profile_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    fields: dict[str, np.ndarray],
    row: int,
    column: int,
    profile: str,
    offset: float,
) -> dict[str, Any]:
    return {
        "profile": profile,
        "offset": float(offset),
        "s": float(fields["S"][row, column]),
        "y": float(fields["Y"][row, column]),
        "u": float(fields["u"][row, column]),
        "Uz": float(fields["Uz"][row, column]),
        "Uy": float(fields["Uy"][row, column]),
        "speed": float(fields["speed"][row, column]),
        "fluid_mask": bool(fields["fluid_mask"][row, column]),
        "near_solid_mask": bool(fields["near_solid_mask"][row, column]),
    }
