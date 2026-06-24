from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class OgdenMembraneMaterial:
    alpha: tuple[float, ...]
    shear_modulus_pa: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.alpha) == 0:
            raise ValueError("alpha must contain at least one term")
        if len(self.alpha) != len(self.shear_modulus_pa):
            raise ValueError("alpha and shear_modulus_pa must have the same length")
        for name, values in (
            ("alpha", self.alpha),
            ("shear_modulus_pa", self.shear_modulus_pa),
        ):
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} terms must be finite")
        if any(abs(value) <= 1.0e-12 for value in self.alpha):
            raise ValueError("alpha terms must be non-zero")

    @classmethod
    def from_sequences(
        cls,
        *,
        alpha: Sequence[float],
        shear_modulus_pa: Sequence[float],
    ) -> "OgdenMembraneMaterial":
        return cls(
            alpha=tuple(float(value) for value in alpha),
            shear_modulus_pa=tuple(float(value) for value in shear_modulus_pa),
        )

    def equibiaxial_cauchy_stress_pa(self, stretch: float) -> float:
        if not math.isfinite(stretch) or stretch <= 0.0:
            raise ValueError("stretch must be positive and finite")
        stress_pa = 0.0
        thickness_stretch = stretch**-2
        for alpha_i, mu_i in zip(self.alpha, self.shear_modulus_pa):
            stress_pa += (2.0 * mu_i / alpha_i) * (
                stretch**alpha_i - thickness_stretch**alpha_i
            )
        return stress_pa


def stretch_from_volume_ratio(
    *,
    current_volume_m3: float,
    rest_volume_m3: float,
) -> float:
    if not math.isfinite(current_volume_m3) or not math.isfinite(rest_volume_m3):
        raise ValueError("volumes must be finite")
    if current_volume_m3 <= 0.0 or rest_volume_m3 <= 0.0:
        raise ValueError("volumes must be positive")
    return (current_volume_m3 / rest_volume_m3) ** (1.0 / 3.0)
