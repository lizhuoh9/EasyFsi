from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


# A single fluid owner cell may receive many sharp-interface pressure rows on
# locally refined CAD surfaces. Keep one power-of-two slot budget above the
# largest short-probe refined-CAD row count observed so far.
PRESSURE_INTERFACE_COUPLING_SLOT_COUNT = 16
PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS = (
    PRESSURE_INTERFACE_COUPLING_SLOT_COUNT - 1
)


class CanonicalVelocityBoundaryTopologyIncompatibilityError(RuntimeError):
    """A provisional obstacle mutation cannot retain a canonical ledger."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        count: int,
        diagnostics: Mapping[object, object] | None = None,
    ) -> None:
        super().__init__(str(message))
        self.reason_code = str(reason_code)
        self.count = int(count)
        context = _json_safe_plain_dict(diagnostics)
        self.diagnostics = {
            "schema_version": 1,
            "reason_code": self.reason_code,
            "count": self.count,
            "context": context,
        }


def _json_safe_plain_dict(
    diagnostics: Mapping[object, object] | None,
) -> dict[str, object]:
    """Return exception details suitable for a JSON report without aliases."""

    if diagnostics is None:
        return {}
    return {
        str(key): _json_safe_value(value)
        for key, value in dict(diagnostics).items()
    }


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def far_pressure_side_normal_sign_from_direction(
    *,
    pressure_direction: Sequence[float],
    interface_normal: Sequence[float],
    tangential_tolerance: float = 1.0e-8,
) -> float:
    """Map a directional pressure boundary to the HIBM far-pressure side.

    Return +1.0 when the prescribed pressure is on the +n side of the
    interface and -1.0 when it is on the -n side. A positive pressure pushes
    the solid away from the pressured side, so a requested force direction
    aligned with the interface normal means the pressure is on the -n side.
    """

    direction = _unit_vector3(pressure_direction, name="pressure_direction")
    normal = _unit_vector3(interface_normal, name="interface_normal")
    normal_component = sum(d * n for d, n in zip(direction, normal, strict=True))
    tolerance = float(tangential_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tangential_tolerance must be a finite non-negative number")
    if abs(normal_component) <= tolerance:
        raise ValueError(
            "pressure_direction must have a non-zero normal component relative "
            "to interface_normal"
        )
    return -1.0 if normal_component > 0.0 else 1.0


def _unit_vector3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} components must be finite")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1.0e-30:
        raise ValueError(f"{name} must be non-zero")
    return tuple(value / norm for value in vector)
