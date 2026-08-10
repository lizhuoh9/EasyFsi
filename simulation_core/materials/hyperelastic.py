from dataclasses import dataclass
import math

import taichi as ti

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, init_taichi


PSI_TO_PA = 6894.757293168
ECOFLEX_SERIES_TECH_BULLETIN_URL = "https://www.smooth-on.com/tb/files/ECOFLEX_SERIES_TB.pdf"


def psi_to_pa(value_psi: float) -> float:
    return _finite_float(value_psi, "value_psi") * PSI_TO_PA


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def incompressible_uniaxial_nominal_stress_pa(stretch: float, shear_modulus_pa: float) -> float:
    """Nominal stress for incompressible Neo-Hookean uniaxial tension."""

    stretch = _positive_float(stretch, "stretch")
    shear_modulus = _positive_float(shear_modulus_pa, "shear_modulus_pa")
    return shear_modulus * (stretch - stretch ** -2)


@dataclass(frozen=True)
class NeoHookeanMaterial:
    """SI-unit compressible Neo-Hookean material card."""

    name: str
    density_kgm3: float
    shear_modulus_pa: float
    bulk_modulus_pa: float
    youngs_modulus_pa: float
    poissons_ratio: float
    source: str = ""
    calibration_note: str = ""
    shore_hardness: str | None = None
    tensile_strength_pa: float | None = None
    modulus_100_pa: float | None = None
    elongation_at_break_percent: float | None = None

    @property
    def lame_lambda_pa(self) -> float:
        return self.bulk_modulus_pa - (2.0 / 3.0) * self.shear_modulus_pa

    @property
    def sound_speed_mps(self) -> float:
        self.validate()
        return math.sqrt(
            (self.bulk_modulus_pa + (4.0 / 3.0) * self.shear_modulus_pa)
            / self.density_kgm3
        )

    def stable_explicit_dt_s(self, spacing_m: float, cfl: float = 0.35) -> float:
        spacing_m = _positive_float(spacing_m, "spacing_m")
        cfl = _positive_float(cfl, "cfl")
        return cfl * spacing_m / self.sound_speed_mps

    def validate(self) -> None:
        _positive_float(self.density_kgm3, "density_kgm3")
        _positive_float(self.shear_modulus_pa, "shear_modulus_pa")
        _positive_float(self.bulk_modulus_pa, "bulk_modulus_pa")
        _positive_float(self.youngs_modulus_pa, "youngs_modulus_pa")
        poissons_ratio = _finite_float(self.poissons_ratio, "poissons_ratio")
        if not 0.0 <= poissons_ratio < 0.5:
            raise ValueError("poissons_ratio must be in [0, 0.5)")
        for name in (
            "tensile_strength_pa",
            "modulus_100_pa",
            "elongation_at_break_percent",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive_float(value, name)

    @classmethod
    def from_youngs_modulus(
        cls,
        *,
        name: str,
        density_kgm3: float,
        youngs_modulus_pa: float,
        poissons_ratio: float,
        source: str = "",
        calibration_note: str = "",
        shore_hardness: str | None = None,
        tensile_strength_pa: float | None = None,
        modulus_100_pa: float | None = None,
        elongation_at_break_percent: float | None = None,
    ) -> "NeoHookeanMaterial":
        youngs_modulus_pa = _positive_float(youngs_modulus_pa, "youngs_modulus_pa")
        poissons_ratio = _finite_float(poissons_ratio, "poissons_ratio")
        if not 0.0 <= poissons_ratio < 0.5:
            raise ValueError("poissons_ratio must be in [0, 0.5)")
        shear_modulus_pa = youngs_modulus_pa / (2.0 * (1.0 + poissons_ratio))
        bulk_modulus_pa = youngs_modulus_pa / (3.0 * (1.0 - 2.0 * poissons_ratio))
        material = cls(
            name=name,
            density_kgm3=float(density_kgm3),
            shear_modulus_pa=shear_modulus_pa,
            bulk_modulus_pa=bulk_modulus_pa,
            youngs_modulus_pa=youngs_modulus_pa,
            poissons_ratio=poissons_ratio,
            source=source,
            calibration_note=calibration_note,
            shore_hardness=shore_hardness,
            tensile_strength_pa=tensile_strength_pa,
            modulus_100_pa=modulus_100_pa,
            elongation_at_break_percent=elongation_at_break_percent,
        )
        material.validate()
        return material

    @classmethod
    def from_100_percent_modulus(
        cls,
        *,
        name: str,
        density_kgm3: float,
        modulus_100_pa: float,
        poissons_ratio: float = 0.49,
        source: str = "",
        calibration_note: str = "",
        shore_hardness: str | None = None,
        tensile_strength_pa: float | None = None,
        elongation_at_break_percent: float | None = None,
    ) -> "NeoHookeanMaterial":
        modulus_100_pa = _positive_float(modulus_100_pa, "modulus_100_pa")
        shear_modulus_pa = modulus_100_pa / (2.0 - 2.0 ** -2)
        youngs_modulus_pa = 2.0 * shear_modulus_pa * (1.0 + float(poissons_ratio))
        return cls.from_youngs_modulus(
            name=name,
            density_kgm3=density_kgm3,
            youngs_modulus_pa=youngs_modulus_pa,
            poissons_ratio=poissons_ratio,
            source=source,
            calibration_note=calibration_note,
            shore_hardness=shore_hardness,
            tensile_strength_pa=tensile_strength_pa,
            modulus_100_pa=modulus_100_pa,
            elongation_at_break_percent=elongation_at_break_percent,
        )


def ecoflex_0010_material(poissons_ratio: float = 0.49) -> NeoHookeanMaterial:
    """Return a datasheet-derived Ecoflex 00-10 starter material card.

    Smooth-On publishes Ecoflex 00-10 density, tensile strength, 100% modulus,
    and break elongation, but not a full stress-strain curve. This card
    calibrates a Neo-Hookean shear modulus from the 100% nominal modulus.
    """

    return NeoHookeanMaterial.from_100_percent_modulus(
        name="Ecoflex 00-10",
        density_kgm3=1040.0,
        modulus_100_pa=psi_to_pa(8.0),
        poissons_ratio=poissons_ratio,
        source=ECOFLEX_SERIES_TECH_BULLETIN_URL,
        calibration_note=(
            "Starter Neo-Hookean card calibrated from Smooth-On Ecoflex 00-10 "
            "100% modulus. Fit Mooney-Rivlin/Ogden/Yeoh coefficients to measured "
            "stress-strain data for production accuracy."
        ),
        shore_hardness="Shore 00-10",
        tensile_strength_pa=psi_to_pa(120.0),
        elongation_at_break_percent=800.0,
    )


@dataclass(frozen=True)
class NeoHookeanStressReport:
    stretch: tuple[float, float, float]
    jacobian: float
    strain_energy_density_jm3: float
    first_piola_pa: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    cauchy_stress_pa: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

    @property
    def first_piola_x_pa(self) -> float:
        return self.first_piola_pa[0][0]

    @property
    def cauchy_x_pa(self) -> float:
        return self.cauchy_stress_pa[0][0]


@ti.data_oriented
class NeoHookeanStressProbe:
    """Device-side Neo-Hookean stress evaluator for material validation."""

    def __init__(
        self,
        material: NeoHookeanMaterial | None = None,
        runtime: TaichiRuntimeConfig | None = None,
    ):
        init_taichi(runtime)
        self.material = material or ecoflex_0010_material()
        self.material.validate()
        self.first_piola_pa = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.cauchy_stress_pa = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.strain_energy_density_jm3 = ti.field(dtype=ti.f32, shape=())
        self.jacobian = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def _evaluate_diagonal_kernel(
        self,
        stretch_x: ti.f32,
        stretch_y: ti.f32,
        stretch_z: ti.f32,
        shear_modulus_pa: ti.f32,
        bulk_modulus_pa: ti.f32,
    ):
        deformation_gradient = ti.Matrix(
            [
                [stretch_x, 0.0, 0.0],
                [0.0, stretch_y, 0.0],
                [0.0, 0.0, stretch_z],
            ]
        )
        jacobian = stretch_x * stretch_y * stretch_z
        safe_jacobian = ti.max(jacobian, 1.0e-8)
        log_j = ti.log(safe_jacobian)
        lame_lambda_pa = bulk_modulus_pa - (2.0 / 3.0) * shear_modulus_pa
        inverse_transpose = deformation_gradient.inverse().transpose()
        first_piola = (
            shear_modulus_pa * (deformation_gradient - inverse_transpose)
            + lame_lambda_pa * log_j * inverse_transpose
        )
        cauchy = (first_piola @ deformation_gradient.transpose()) / safe_jacobian
        invariant_1 = (
            stretch_x * stretch_x
            + stretch_y * stretch_y
            + stretch_z * stretch_z
        )
        energy = (
            0.5 * shear_modulus_pa * (invariant_1 - 3.0)
            - shear_modulus_pa * log_j
            + 0.5 * lame_lambda_pa * log_j * log_j
        )
        self.first_piola_pa[None] = first_piola
        self.cauchy_stress_pa[None] = cauchy
        self.strain_energy_density_jm3[None] = energy
        self.jacobian[None] = jacobian

    def evaluate_diagonal_stretch(self, stretch: tuple[float, float, float]) -> NeoHookeanStressReport:
        if len(stretch) != 3:
            raise ValueError("stretch must contain exactly three values")
        stretch = tuple(float(value) for value in stretch)
        if not all(math.isfinite(value) and value > 0.0 for value in stretch):
            raise ValueError("all stretch values must be positive")
        self._evaluate_diagonal_kernel(
            stretch[0],
            stretch[1],
            stretch[2],
            float(self.material.shear_modulus_pa),
            float(self.material.bulk_modulus_pa),
        )
        return NeoHookeanStressReport(
            stretch=stretch,
            jacobian=float(self.jacobian[None]),
            strain_energy_density_jm3=float(self.strain_energy_density_jm3[None]),
            first_piola_pa=self._read_matrix(self.first_piola_pa),
            cauchy_stress_pa=self._read_matrix(self.cauchy_stress_pa),
        )

    @staticmethod
    def _read_matrix(field: ti.template()) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        value = field[None]
        return (
            (float(value[0, 0]), float(value[0, 1]), float(value[0, 2])),
            (float(value[1, 0]), float(value[1, 1]), float(value[1, 2])),
            (float(value[2, 0]), float(value[2, 1]), float(value[2, 2])),
        )
