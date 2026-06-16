from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from simulation_core.benchmarking.inlet_flow import TimeWindowedInletFlow
from simulation_core.geometry import UvSphereResolution
from simulation_core.mooney_shell_mpm import UvMooneyShellMpmState
from simulation_core.runtime import TaichiRuntimeConfig


@dataclass(frozen=True)
class MembraneInflationConfig:
    radius_m: float
    thickness_m: float
    density_kgm3: float
    c1_pa: float
    c2_pa: float
    inlet_flow_m3s: float
    fill_duration_s: float
    dt_s: float
    step_count: int
    pressure_bulk_modulus_pa: float
    latitude_bands: int = 8
    longitude_segments: int = 16
    grid_nodes: tuple[int, int, int] = (20, 20, 20)
    velocity_damping: float = 0.995
    gravity_mps2: float = 0.0
    runtime_arch: str = "cuda"
    fill_start_s: float = 0.0
    initial_volume_m3: float | None = None
    observation_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        if self.thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive")
        if self.density_kgm3 <= 0.0:
            raise ValueError("density_kgm3 must be positive")
        if self.c1_pa <= 0.0 or self.c2_pa < 0.0:
            raise ValueError("Mooney constants must be non-negative with c1 > 0")
        if self.inlet_flow_m3s < 0.0:
            raise ValueError("inlet_flow_m3s must be non-negative")
        if self.fill_duration_s < 0.0:
            raise ValueError("fill_duration_s must be non-negative")
        if self.fill_start_s < 0.0 or not math.isfinite(self.fill_start_s):
            raise ValueError("fill_start_s must be non-negative and finite")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.step_count <= 0:
            raise ValueError("step_count must be positive")
        if self.pressure_bulk_modulus_pa <= 0.0:
            raise ValueError("pressure_bulk_modulus_pa must be positive")
        if self.initial_volume_m3 is not None and (
            not math.isfinite(self.initial_volume_m3) or self.initial_volume_m3 <= 0.0
        ):
            raise ValueError("initial_volume_m3 must be positive and finite when provided")
        if self.observation_time_s is not None and (
            not math.isfinite(self.observation_time_s) or self.observation_time_s < 0.0
        ):
            raise ValueError(
                "observation_time_s must be non-negative and finite when provided"
            )
        if self.latitude_bands < 4 or self.longitude_segments < 8:
            raise ValueError("sphere resolution is too low")
        if min(self.grid_nodes) < 4:
            raise ValueError("grid_nodes must be at least 4 in each direction")


def run_uv_membrane_inflation_smoke(
    config: MembraneInflationConfig,
) -> dict[str, object]:
    rest_volume_m3 = _initial_volume_m3(config)
    target_volume_m3 = rest_volume_m3
    current_volume_m3 = rest_volume_m3
    max_mean_radial_stretch = 1.0
    pressure_pa = 0.0
    history: list[dict[str, float | int]] = []
    inlet_flow = _inlet_flow(config)

    state = UvMooneyShellMpmState(
        UvSphereResolution(
            latitude_bands=config.latitude_bands,
            longitude_segments=config.longitude_segments,
        ),
        radius_m=config.radius_m,
        thickness_m=config.thickness_m,
        density_kgm3=config.density_kgm3,
        c1_pa=config.c1_pa,
        c2_pa=config.c2_pa,
        grid_nodes=config.grid_nodes,
        runtime=TaichiRuntimeConfig(arch=config.runtime_arch),
    )

    for step_index in range(config.step_count):
        time_s = float(step_index) * config.dt_s
        target_volume_m3 += inlet_flow.volume_between_exact_m3(
            time_s,
            time_s + config.dt_s,
        )
        volume_error_m3 = max(target_volume_m3 - current_volume_m3, 0.0)
        pressure_pa = (
            config.pressure_bulk_modulus_pa * volume_error_m3 / rest_volume_m3
        )
        step_report = state.step(
            dt_s=config.dt_s,
            pressure_pa=pressure_pa,
            velocity_damping=config.velocity_damping,
            body_acceleration_mps2=(0.0, 0.0, -config.gravity_mps2),
        )
        mean_stretch = float(step_report.mean_radial_stretch)
        max_mean_radial_stretch = max(max_mean_radial_stretch, mean_stretch)
        current_volume_m3 = rest_volume_m3 * mean_stretch**3
        history.append(
            {
                "step": step_index + 1,
                "time_s": time_s + config.dt_s,
                "target_volume_m3": target_volume_m3,
                "current_volume_m3": current_volume_m3,
                "pressure_pa": pressure_pa,
                "mean_radial_stretch": mean_stretch,
                "max_edge_strain": float(step_report.max_edge_strain),
                "active_grid_nodes": int(step_report.active_grid_nodes),
                "transfer_relative_error": float(step_report.transfer_relative_error),
            }
        )

    observation = membrane_inflation_volume_observables(config)
    return {
        "case": "generic-uv-membrane-inflation-fsi",
        "config": asdict(config),
        "computed_result_sources": {
            "target_volume_m3": "integral(inlet_flow_m3s, dt)",
            "current_volume_m3": "rest_volume * mean_radial_stretch**3",
            "pressure_pa": "bulk_modulus * (target_volume-current_volume) / rest_volume",
            "max_mean_radial_stretch": "UvMooneyShellMpmState.step report",
            "observation_target_volume_m3": (
                "initial_volume_m3 + integral(inlet_flow_m3s, time)"
            ),
        },
        "rest_volume_m3": rest_volume_m3,
        "final_target_volume_m3": target_volume_m3,
        "observation_time_s": observation["observation_time_s"],
        "observation_inlet_volume_m3": observation["inlet_volume_m3"],
        "observation_target_volume_m3": observation["target_volume_m3"],
        "final_volume_m3": current_volume_m3,
        "final_pressure_pa": pressure_pa,
        "max_mean_radial_stretch": max_mean_radial_stretch,
        "history": history,
    }


def membrane_inflation_volume_observables(
    config: MembraneInflationConfig,
) -> dict[str, float]:
    observation_time_s = (
        float(config.observation_time_s)
        if config.observation_time_s is not None
        else float(config.step_count) * config.dt_s
    )
    rest_volume_m3 = _initial_volume_m3(config)
    inlet_volume_m3 = _inlet_flow(config).volume_between_m3(
        0.0,
        observation_time_s,
        runtime_arch=config.runtime_arch,
    )
    return {
        "rest_volume_m3": rest_volume_m3,
        "observation_time_s": observation_time_s,
        "inlet_volume_m3": inlet_volume_m3,
        "target_volume_m3": rest_volume_m3 + inlet_volume_m3,
    }


def _sphere_volume(radius_m: float) -> float:
    return 4.0 * math.pi * radius_m**3 / 3.0


def _initial_volume_m3(config: MembraneInflationConfig) -> float:
    if config.initial_volume_m3 is not None:
        return float(config.initial_volume_m3)
    return _sphere_volume(config.radius_m)


def _inlet_flow(config: MembraneInflationConfig) -> TimeWindowedInletFlow:
    return TimeWindowedInletFlow(
        volumetric_flow_m3s=config.inlet_flow_m3s,
        start_s=config.fill_start_s,
        end_s=config.fill_start_s + config.fill_duration_s,
    )
