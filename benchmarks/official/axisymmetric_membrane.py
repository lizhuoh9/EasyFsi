import math
from collections.abc import Sequence
from dataclasses import dataclass

import taichi as ti

from simulation_core.runtime import TaichiRuntimeConfig, init_taichi


@dataclass(frozen=True)
class SmoothAxisymmetricMembraneStressConfig:
    neck_radius_m: float
    initial_height_m: float
    final_height_m: float
    initial_bulb_radius_m: float
    target_volume_m3: float
    ogden_alpha: tuple[float, ...]
    ogden_shear_modulus_pa: tuple[float, ...]
    blend_exponent: float = 2.0
    sample_count: int = 4096
    runtime_arch: str = "cuda"

    def __post_init__(self) -> None:
        for name, value in (
            ("neck_radius_m", self.neck_radius_m),
            ("initial_height_m", self.initial_height_m),
            ("final_height_m", self.final_height_m),
            ("initial_bulb_radius_m", self.initial_bulb_radius_m),
            ("target_volume_m3", self.target_volume_m3),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if len(self.ogden_alpha) == 0:
            raise ValueError("ogden_alpha must contain at least one term")
        if len(self.ogden_alpha) != len(self.ogden_shear_modulus_pa):
            raise ValueError(
                "ogden_alpha and ogden_shear_modulus_pa must have the same length"
            )
        if self.sample_count < 32:
            raise ValueError("sample_count must be at least 32")
        if not math.isfinite(self.blend_exponent) or self.blend_exponent <= 1.0:
            raise ValueError("blend_exponent must be greater than 1")


@dataclass(frozen=True)
class SmoothAxisymmetricMembraneStressReport:
    initial_volume_m3: float
    final_volume_m3: float
    final_bulb_radius_m: float
    max_mises_stress_pa: float
    max_circumferential_stretch: float
    max_meridional_stretch: float
    max_stress_z_m: float


def smooth_axisymmetric_membrane_stress_report(
    config: SmoothAxisymmetricMembraneStressConfig,
) -> SmoothAxisymmetricMembraneStressReport:
    evaluator = _SmoothAxisymmetricMembraneStressEvaluator(
        term_count=len(config.ogden_alpha),
        sample_count=config.sample_count,
        runtime=TaichiRuntimeConfig(arch=config.runtime_arch),
    )
    evaluator.load_material_terms(
        alpha=config.ogden_alpha,
        shear_modulus_pa=config.ogden_shear_modulus_pa,
    )
    return evaluator.evaluate(config)


@ti.data_oriented
class _SmoothAxisymmetricMembraneStressEvaluator:
    def __init__(
        self,
        *,
        term_count: int,
        sample_count: int,
        runtime: TaichiRuntimeConfig,
    ):
        if term_count <= 0:
            raise ValueError("term_count must be positive")
        if sample_count < 32:
            raise ValueError("sample_count must be at least 32")
        init_taichi(runtime)
        self.term_count = int(term_count)
        self.sample_count = int(sample_count)
        self.alpha = ti.field(dtype=ti.f32, shape=self.term_count)
        self.mu_pa = ti.field(dtype=ti.f32, shape=self.term_count)
        self.report = ti.Vector.field(7, dtype=ti.f32, shape=())
        self.volume_m3 = ti.field(dtype=ti.f32, shape=())

    def load_material_terms(
        self,
        *,
        alpha: Sequence[float],
        shear_modulus_pa: Sequence[float],
    ) -> None:
        if len(alpha) != self.term_count or len(shear_modulus_pa) != self.term_count:
            raise ValueError("material term lengths must match term_count")
        for index, value in enumerate(alpha):
            if not math.isfinite(value) or abs(value) <= 1.0e-12:
                raise ValueError("alpha terms must be finite and non-zero")
            self.alpha[index] = float(value)
        for index, value in enumerate(shear_modulus_pa):
            if not math.isfinite(value):
                raise ValueError("shear modulus terms must be finite")
            self.mu_pa[index] = float(value)

    def evaluate(
        self,
        config: SmoothAxisymmetricMembraneStressConfig,
    ) -> SmoothAxisymmetricMembraneStressReport:
        initial_volume_m3 = self._volume_for_radius(
            neck_radius_m=float(config.neck_radius_m),
            height_m=float(config.initial_height_m),
            bulb_radius_m=float(config.initial_bulb_radius_m),
            blend_exponent=float(config.blend_exponent),
        )
        lower_radius_m = float(config.neck_radius_m)
        upper_radius_m = max(
            float(config.initial_bulb_radius_m) * 8.0,
            float(config.neck_radius_m) * 2.0,
        )
        for _ in range(48):
            mid_radius_m = 0.5 * (lower_radius_m + upper_radius_m)
            mid_volume_m3 = self._volume_for_radius(
                neck_radius_m=float(config.neck_radius_m),
                height_m=float(config.final_height_m),
                bulb_radius_m=mid_radius_m,
                blend_exponent=float(config.blend_exponent),
            )
            if mid_volume_m3 < float(config.target_volume_m3):
                lower_radius_m = mid_radius_m
            else:
                upper_radius_m = mid_radius_m
        final_bulb_radius_m = 0.5 * (lower_radius_m + upper_radius_m)
        final_volume_m3 = self._volume_for_radius(
            neck_radius_m=float(config.neck_radius_m),
            height_m=float(config.final_height_m),
            bulb_radius_m=final_bulb_radius_m,
            blend_exponent=float(config.blend_exponent),
        )
        self._stress_kernel(
            float(config.neck_radius_m),
            float(config.initial_height_m),
            float(config.final_height_m),
            float(config.initial_bulb_radius_m),
            float(initial_volume_m3),
            float(final_volume_m3),
            float(final_bulb_radius_m),
            float(config.blend_exponent),
            int(config.sample_count),
        )
        values = tuple(float(value) for value in self.report[None])
        return SmoothAxisymmetricMembraneStressReport(
            initial_volume_m3=values[0],
            final_volume_m3=values[1],
            final_bulb_radius_m=values[2],
            max_mises_stress_pa=values[3],
            max_circumferential_stretch=values[4],
            max_meridional_stretch=values[5],
            max_stress_z_m=values[6],
        )

    def _volume_for_radius(
        self,
        *,
        neck_radius_m: float,
        height_m: float,
        bulb_radius_m: float,
        blend_exponent: float,
    ) -> float:
        self._volume_kernel(
            float(neck_radius_m),
            float(height_m),
            float(bulb_radius_m),
            float(blend_exponent),
            int(self.sample_count),
        )
        return float(self.volume_m3[None])

    @ti.func
    def _smooth_radius(
        self,
        z_m,
        neck_radius_m,
        height_m,
        bulb_radius_m,
        blend_exponent,
    ):
        half_height_m = height_m / 3.0
        center_z_m = -2.0 * height_m / 3.0
        neck_radius = 0.0
        if -0.5 * height_m <= z_m <= 0.0:
            neck_radius = neck_radius_m
        eta = (z_m - center_z_m) / half_height_m
        ellipse_q = 1.0 - eta * eta
        ellipse_radius = 0.0
        ellipse_slope = 0.0
        if ellipse_q > 0.0:
            ellipse_radius = bulb_radius_m * ti.sqrt(ellipse_q)
        neck_power = 0.0
        if neck_radius > 0.0:
            neck_power = ti.pow(neck_radius, blend_exponent)
        ellipse_power = 0.0
        if ellipse_radius > 0.0:
            ellipse_power = ti.pow(ellipse_radius, blend_exponent)
        blend_sum = neck_power + ellipse_power
        radius_m = ti.pow(ti.max(blend_sum, 1.0e-20), 1.0 / blend_exponent)
        return radius_m

    @ti.func
    def _smooth_radius_and_slope(
        self,
        z_m,
        neck_radius_m,
        height_m,
        bulb_radius_m,
        blend_exponent,
    ):
        half_height_m = height_m / 3.0
        center_z_m = -2.0 * height_m / 3.0
        ellipse_q = 1.0 - ((z_m - center_z_m) / half_height_m) ** 2
        ellipse_radius = 0.0
        ellipse_slope = 0.0
        if ellipse_q > 0.0:
            ellipse_radius = bulb_radius_m * ti.sqrt(ellipse_q)
            ellipse_slope = (
                -bulb_radius_m
                * (z_m - center_z_m)
                / (half_height_m * half_height_m * ti.sqrt(ellipse_q))
            )
        radius_m = self._smooth_radius(
            z_m,
            neck_radius_m,
            height_m,
            bulb_radius_m,
            blend_exponent,
        )
        neck_radius = 0.0
        if -0.5 * height_m <= z_m <= 0.0:
            neck_radius = neck_radius_m
        neck_power = 0.0
        if neck_radius > 0.0:
            neck_power = ti.pow(neck_radius, blend_exponent)
        ellipse_power = 0.0
        if ellipse_radius > 0.0:
            ellipse_power = ti.pow(ellipse_radius, blend_exponent)
        blend_sum = neck_power + ellipse_power
        slope = 0.0
        if ellipse_radius > 1.0e-12:
            slope = (
                ti.pow(blend_sum, 1.0 / blend_exponent - 1.0)
                * ti.pow(ti.max(ellipse_radius, 1.0e-20), blend_exponent - 1.0)
                * ellipse_slope
            )
        return radius_m, slope

    @ti.func
    def _smooth_volume(
        self,
        neck_radius_m,
        height_m,
        bulb_radius_m,
        blend_exponent,
        sample_count,
    ):
        volume_m3 = 0.0
        dz_m = height_m / ti.cast(sample_count, ti.f32)
        for sample in range(sample_count):
            material_s = (
                ti.cast(sample, ti.f32) + 0.5
            ) / ti.cast(sample_count, ti.f32)
            z_m = -height_m + material_s * height_m
            radius_m = self._smooth_radius(
                z_m,
                neck_radius_m,
                height_m,
                bulb_radius_m,
                blend_exponent,
            )
            volume_m3 += ti.math.pi * radius_m * radius_m * dz_m
        return volume_m3

    @ti.func
    def _ogden_biaxial_mises(self, stretch_theta, stretch_meridional):
        stretch_thickness = 1.0 / (stretch_theta * stretch_meridional)
        sigma_theta = 0.0
        sigma_meridional = 0.0
        for term in range(self.term_count):
            alpha_i = self.alpha[term]
            mu_i = self.mu_pa[term]
            thickness_term = ti.pow(stretch_thickness, alpha_i)
            sigma_theta += (2.0 * mu_i / alpha_i) * (
                ti.pow(stretch_theta, alpha_i) - thickness_term
            )
            sigma_meridional += (2.0 * mu_i / alpha_i) * (
                ti.pow(stretch_meridional, alpha_i) - thickness_term
            )
        mises_sq = (
            sigma_theta * sigma_theta
            + sigma_meridional * sigma_meridional
            - sigma_theta * sigma_meridional
        )
        return ti.sqrt(ti.max(mises_sq, 0.0))

    @ti.func
    def _segment_stress(
        self,
        material_s0,
        material_s1,
        neck_radius_m,
        initial_height_m,
        final_height_m,
        initial_bulb_radius_m,
        final_bulb_radius_m,
        blend_exponent,
    ):
        initial_z0_m = -initial_height_m + material_s0 * initial_height_m
        initial_z1_m = -initial_height_m + material_s1 * initial_height_m
        final_z0_m = -final_height_m + material_s0 * final_height_m
        final_z1_m = -final_height_m + material_s1 * final_height_m
        initial_radius0_m = self._smooth_radius(
            initial_z0_m,
            neck_radius_m,
            initial_height_m,
            initial_bulb_radius_m,
            blend_exponent,
        )
        initial_radius1_m = self._smooth_radius(
            initial_z1_m,
            neck_radius_m,
            initial_height_m,
            initial_bulb_radius_m,
            blend_exponent,
        )
        final_radius0_m = self._smooth_radius(
            final_z0_m,
            neck_radius_m,
            final_height_m,
            final_bulb_radius_m,
            blend_exponent,
        )
        final_radius1_m = self._smooth_radius(
            final_z1_m,
            neck_radius_m,
            final_height_m,
            final_bulb_radius_m,
            blend_exponent,
        )
        initial_radius_avg_m = 0.5 * (initial_radius0_m + initial_radius1_m)
        final_radius_avg_m = 0.5 * (final_radius0_m + final_radius1_m)
        mises_stress_pa = 0.0
        stretch_theta = 1.0
        stretch_meridional = 1.0
        if initial_radius_avg_m > 1.0e-8 and final_radius_avg_m > 1.0e-8:
            stretch_theta = final_radius_avg_m / initial_radius_avg_m
            initial_dr_m = initial_radius1_m - initial_radius0_m
            final_dr_m = final_radius1_m - final_radius0_m
            initial_dz_m = initial_z1_m - initial_z0_m
            final_dz_m = final_z1_m - final_z0_m
            initial_ds = ti.sqrt(
                initial_dr_m * initial_dr_m + initial_dz_m * initial_dz_m
            )
            final_ds = ti.sqrt(final_dr_m * final_dr_m + final_dz_m * final_dz_m)
            stretch_meridional = final_ds / initial_ds
            mises_stress_pa = self._ogden_biaxial_mises(
                stretch_theta,
                stretch_meridional,
            )
        return ti.Vector(
            [
                mises_stress_pa,
                stretch_theta,
                stretch_meridional,
                0.5 * (initial_z0_m + initial_z1_m),
            ]
        )

    @ti.kernel
    def _volume_kernel(
        self,
        neck_radius_m: ti.f32,
        height_m: ti.f32,
        bulb_radius_m: ti.f32,
        blend_exponent: ti.f32,
        sample_count: ti.i32,
    ):
        self.volume_m3[None] = self._smooth_volume(
            neck_radius_m,
            height_m,
            bulb_radius_m,
            blend_exponent,
            sample_count,
        )

    @ti.kernel
    def _stress_kernel(
        self,
        neck_radius_m: ti.f32,
        initial_height_m: ti.f32,
        final_height_m: ti.f32,
        initial_bulb_radius_m: ti.f32,
        initial_volume_m3: ti.f32,
        final_volume_m3: ti.f32,
        final_bulb_radius_m: ti.f32,
        blend_exponent: ti.f32,
        sample_count: ti.i32,
    ):
        max_mises_stress_pa = 0.0
        max_theta = 1.0
        max_meridional = 1.0
        max_z_m = 0.0
        for sample in range(sample_count - 1):
            material_s0 = ti.cast(sample, ti.f32) / ti.cast(sample_count - 1, ti.f32)
            material_s1 = ti.cast(sample + 1, ti.f32) / ti.cast(
                sample_count - 1,
                ti.f32,
            )
            segment = self._segment_stress(
                material_s0,
                material_s1,
                neck_radius_m,
                initial_height_m,
                final_height_m,
                initial_bulb_radius_m,
                final_bulb_radius_m,
                blend_exponent,
            )
            if segment[0] > max_mises_stress_pa:
                max_mises_stress_pa = segment[0]
                max_theta = segment[1]
                max_meridional = segment[2]
                max_z_m = segment[3]

        transition_s = 2.0 / 3.0
        band_s = ti.min(0.1, ti.max(0.01, 2.0 * neck_radius_m / initial_height_m))
        for sample in range(sample_count - 1):
            local_s0 = ti.cast(sample, ti.f32) / ti.cast(sample_count - 1, ti.f32)
            local_s1 = ti.cast(sample + 1, ti.f32) / ti.cast(sample_count - 1, ti.f32)
            material_s0 = ti.max(0.0, transition_s - band_s + 2.0 * band_s * local_s0)
            material_s1 = ti.min(1.0, transition_s - band_s + 2.0 * band_s * local_s1)
            segment = self._segment_stress(
                material_s0,
                material_s1,
                neck_radius_m,
                initial_height_m,
                final_height_m,
                initial_bulb_radius_m,
                final_bulb_radius_m,
                blend_exponent,
            )
            if segment[0] > max_mises_stress_pa:
                max_mises_stress_pa = segment[0]
                max_theta = segment[1]
                max_meridional = segment[2]
                max_z_m = segment[3]

        self.report[None] = ti.Vector(
            [
                initial_volume_m3,
                final_volume_m3,
                final_bulb_radius_m,
                max_mises_stress_pa,
                max_theta,
                max_meridional,
                max_z_m,
            ]
        )
