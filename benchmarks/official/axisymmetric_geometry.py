import math
from dataclasses import dataclass

import taichi as ti

from simulation_core.runtime import TaichiRuntimeConfig, init_taichi


@dataclass(frozen=True)
class AxisymmetricNeckedEllipseProfile:
    """Axisymmetric union of a straight neck and an elliptical bulb profile."""

    neck_radius_m: float
    neck_z_min_m: float
    neck_z_max_m: float
    bulb_radius_m: float
    bulb_center_z_m: float
    bulb_half_height_m: float

    def __post_init__(self) -> None:
        _require_positive(self.neck_radius_m, "neck_radius_m")
        _require_positive(self.bulb_radius_m, "bulb_radius_m")
        _require_positive(self.bulb_half_height_m, "bulb_half_height_m")
        if not self.neck_z_min_m < self.neck_z_max_m:
            raise ValueError("neck_z_min_m must be less than neck_z_max_m")
        for name, value in (
            ("neck_z_min_m", self.neck_z_min_m),
            ("neck_z_max_m", self.neck_z_max_m),
            ("bulb_center_z_m", self.bulb_center_z_m),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    @property
    def z_min_m(self) -> float:
        return min(self.neck_z_min_m, self.bulb_center_z_m - self.bulb_half_height_m)

    @property
    def z_max_m(self) -> float:
        return max(self.neck_z_max_m, self.bulb_center_z_m + self.bulb_half_height_m)

    def volume_m3(
        self,
        *,
        sample_count: int = 16384,
        runtime_arch: str = "cuda",
    ) -> float:
        integrator = _AxisymmetricNeckedEllipseVolumeIntegrator(
            sample_count=sample_count,
            runtime=TaichiRuntimeConfig(arch=runtime_arch),
        )
        return integrator.integrate(self)


@ti.data_oriented
class _AxisymmetricNeckedEllipseVolumeIntegrator:
    def __init__(
        self,
        *,
        sample_count: int,
        runtime: TaichiRuntimeConfig,
    ):
        if sample_count < 16:
            raise ValueError("sample_count must be at least 16")
        init_taichi(runtime)
        self.sample_count = int(sample_count)
        self.volume_m3 = ti.field(dtype=ti.f32, shape=())

    def integrate(self, profile: AxisymmetricNeckedEllipseProfile) -> float:
        self._integrate_kernel(
            float(profile.neck_radius_m),
            float(profile.neck_z_min_m),
            float(profile.neck_z_max_m),
            float(profile.bulb_radius_m),
            float(profile.bulb_center_z_m),
            float(profile.bulb_half_height_m),
            float(profile.z_min_m),
            float(profile.z_max_m),
        )
        return float(self.volume_m3[None])

    @ti.kernel
    def _integrate_kernel(
        self,
        neck_radius_m: ti.f32,
        neck_z_min_m: ti.f32,
        neck_z_max_m: ti.f32,
        bulb_radius_m: ti.f32,
        bulb_center_z_m: ti.f32,
        bulb_half_height_m: ti.f32,
        z_min_m: ti.f32,
        z_max_m: ti.f32,
    ):
        self.volume_m3[None] = 0.0
        dz = (z_max_m - z_min_m) / ti.cast(self.sample_count, ti.f32)
        for index in range(self.sample_count):
            z = z_min_m + (ti.cast(index, ti.f32) + 0.5) * dz
            radius_m = 0.0
            if neck_z_min_m <= z <= neck_z_max_m:
                radius_m = neck_radius_m
            eta = (z - bulb_center_z_m) / bulb_half_height_m
            ellipse_term = 1.0 - eta * eta
            if ellipse_term > 0.0:
                ellipse_radius_m = bulb_radius_m * ti.sqrt(ellipse_term)
                radius_m = ti.max(radius_m, ellipse_radius_m)
            ti.atomic_add(self.volume_m3[None], ti.math.pi * radius_m * radius_m * dz)


def _require_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
