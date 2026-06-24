from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UvMooneyShellMpmReport:
    particle_count: int
    edge_count: int
    active_grid_nodes: int
    grid_out_of_bounds_particle_count: int
    mean_radial_stretch: float
    max_radial_stretch_error: float
    max_edge_strain: float
    max_speed_mps: float
    total_mass_kg: float
    internal_force_rms_n: float
    net_internal_force_relative_error: float
    transfer_relative_error: float


@dataclass(frozen=True)
class TriMooneyShellMpmReport:
    particle_count: int
    face_count: int
    edge_count: int
    active_grid_nodes: int
    grid_out_of_bounds_particle_count: int
    particle_spacing_m: float
    grid_spacing_m: tuple[float, float, float]
    mean_radial_stretch: float
    max_radial_stretch_error: float
    max_edge_strain: float
    max_speed_mps: float
    total_mass_kg: float
    total_area_m2: float
    primary_mean_displacement_m: tuple[float, float, float]
    primary_mean_velocity_mps: tuple[float, float, float]
    secondary_mean_displacement_m: tuple[float, float, float]
    secondary_mean_velocity_mps: tuple[float, float, float]
    particle_momentum_kg_mps: tuple[float, float, float]
    grid_momentum_kg_mps: tuple[float, float, float]
    total_force_n: tuple[float, float, float]
    internal_force_rms_n: float
    net_internal_force_relative_error: float
    transfer_relative_error: float
    primary_particle_count: int = 0
    secondary_particle_count: int = 0


__all__ = [
    "TriMooneyShellMpmReport",
    "UvMooneyShellMpmReport",
]
