from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.official.solid_mpm_fsi_runner import (
    PRIMARY_REGION_ID,
    SECONDARY_UNUSED_REGION_ID,
    _lame_parameters,
)
# Strong-coupling (Picard + Aitken) marker-state helpers. These are generic
# host-side numpy snapshot/relax/restore operations on HibmMpmSurfaceMarkers
# fields, proven by the squid case's sharp marker fixed-point loop
# (cases/squid_soft_robot/step_loop.py, advance_sharp_marker_fixed_point_step);
# imported rather than duplicated. The pressure-Neumann-gradient snapshot/
# restore pair is NOT imported because the squid versions are bound to that
# case's coupling-state object — thin local equivalents live below.
from cases.squid_soft_robot.checkpointing import (
    relaxed_sharp_marker_state_arrays,
    relaxed_sharp_pressure_neumann_gradient_state_array,
    restore_sharp_marker_state_arrays,
    sharp_marker_state_arrays,
)
from simulation_core.coupling.hibm_mpm import (
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    advance_hibm_mpm_sharp_mpm_step,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmState
from cases.turek_hron_kernels import (
    outlet_zflux_sum_kernel,
    th_channel_boundary_rows_kernel,
)


TUREK_HRON_CASE_ID = "turek-hron-fsi"

# Every-N-steps incremental history flush (progress inspectable mid-run).
TUREK_HRON_HISTORY_FLUSH_INTERVAL_STEPS = 25

# Channel walls (solver y=0 and y=channel_height_m) are physical no-slip walls.
# They are imposed as zero-velocity Dirichlet projection rows rewritten each
# step alongside the parabolic zmax inlet plane, then stamped into the velocity
# field via apply_velocity_dirichlet_boundary_rows before every advance call
# (the HIBM step clears and reassembles marker Dirichlet rows internally, so
# the prescribed zmax inlet flux is carried by the velocity field under
# velocity_inlet_zmax=True). zmin is the p=0 pressure outlet.
TUREK_HRON_WALL_BOUNDARY_MODEL = "zero_velocity_dirichlet_rows_rewritten_each_step"

TUREK_HRON_PRESET_PARAMETERS: dict[str, dict[str, float]] = {
    "fsi1": {
        "mean_inlet_velocity_mps": 0.2,
        "solid_density_kgm3": 1000.0,
        "young_modulus_pa": 1.4e6,
        "poisson_ratio": 0.4,
        "dt_s": 5.0e-3,
    },
    "fsi2": {
        "mean_inlet_velocity_mps": 1.0,
        "solid_density_kgm3": 10000.0,
        "young_modulus_pa": 1.4e6,
        "poisson_ratio": 0.4,
        "dt_s": 1.0e-3,
    },
    "fsi3": {
        "mean_inlet_velocity_mps": 2.0,
        "solid_density_kgm3": 1000.0,
        "young_modulus_pa": 5.6e6,
        "poisson_ratio": 0.4,
        "dt_s": 1.0e-3,
    },
}

# Reference values are per unit span (2D benchmark); compare force / span_m.
TUREK_HRON_REFERENCE_RESULTS: dict[str, dict[str, Any]] = {
    "fsi1": {
        "source": "Turek & Hron (2006) canonical steady benchmark",
        "ux_a_m": 2.27e-5,
        "uy_a_m": 8.209e-4,
        "drag_n_per_m": 14.295,
        "lift_n_per_m": 0.7638,
        "regime": "steady",
    },
    "fsi2": {
        "source": "Turek & Hron (2006) canonical benchmark",
        "regime": "periodic-large-amplitude",
    },
    "fsi3": {
        "source": "LS-DYNA ICFD aerofsi1 report (mean +/- amplitude)",
        "ux_a_mean_m": -2.35e-3,
        "ux_a_amplitude_m": 2.45e-3,
        "uy_a_mean_m": 1.5e-3,
        "uy_a_amplitude_m": 33.5e-3,
        "drag_mean_n_per_m": 452.0,
        "drag_amplitude_n_per_m": 31.0,
        "lift_mean_n_per_m": 3.3,
        "lift_amplitude_n_per_m": 83.1,
        "regime": "periodic",
    },
}

TUREK_HRON_CASE_METADATA: dict[str, Any] = {
    "source": {
        "name": "Turek & Hron FSI benchmark (2006) with LS-DYNA ICFD cross-reference",
        "cross_reference": "ICFD_example_aerofsi1",
    },
    "geometry": {
        "channel_length_m": 2.5,
        "channel_height_m": 0.41,
        "cylinder_center_x_m": 0.2,
        "cylinder_center_y_m": 0.2,
        "cylinder_radius_m": 0.05,
        "beam_length_m": 0.35,
        "beam_thickness_m": 0.02,
        "beam_tip_x_m": 0.6,
    },
    "fluid": {
        "density_kgm3": 1000.0,
        "viscosity_pa_s": 1.0,
        "inlet_profile": "parabolic 1.5*U*4*y*(H-y)/H^2 with cosine ramp over 2 s",
        "gravity": "none",
    },
    "solver_axis_convention": {
        "streamwise": "solver z, inlet at zmax flowing toward -z",
        "wall_normal": "solver y",
        "span": "solver x (thin slab)",
        "mapping": "turek_hron_x -> solver_z = channel_length_m - x",
    },
    "wall_boundary_model": TUREK_HRON_WALL_BOUNDARY_MODEL,
    "presets": TUREK_HRON_PRESET_PARAMETERS,
    "reference_results": TUREK_HRON_REFERENCE_RESULTS,
}


@dataclass(frozen=True)
class TurekHronFsiConfig:
    channel_length_m: float = 2.5
    channel_height_m: float = 0.41
    span_m: float = 0.05
    cylinder_center_x_m: float = 0.2
    cylinder_center_y_m: float = 0.2
    cylinder_radius_m: float = 0.05
    beam_length_m: float = 0.35
    beam_thickness_m: float = 0.02
    beam_tip_x_m: float = 0.6
    mean_inlet_velocity_mps: float = 0.2
    inlet_ramp_time_s: float = 2.0
    fluid_density_kgm3: float = 1000.0
    fluid_viscosity_pa_s: float = 1.0
    solid_density_kgm3: float = 1000.0
    young_modulus_pa: float = 1.4e6
    poisson_ratio: float = 0.4
    dt_s: float = 5.0e-3
    step_count: int = 200
    grid_nodes: tuple[int, int, int] = (4, 48, 288)
    solid_particle_counts: tuple[int, int, int] = (1, 8, 140)
    markers_per_side: int = 48
    markers_per_tip: int = 4
    solid_substeps: int = 100
    flow_predictor_substeps: int = 1
    # The 2.5 m channel spans 288 streamwise cells at the default grid; the
    # FV-CG iteration budget must scale with the domain length (the 64-cell
    # vertical-flap case needs 1080).
    flow_projection_iterations: int = 4000
    flow_pressure_solver: str = "fv_cg"
    flow_cg_tolerance: float = 1.0e-6
    # CG preconditioner. Default "auto" -> Jacobi on a uniform grid. Set
    # "fv_multigrid" to force the multigrid preconditioner (converges the
    # near-obstacle deficient-stencil modes the plain CG leaves under-resolved).
    flow_cg_preconditioner: str = "auto"
    # Re-projection solve budget (perf, 2026-07): the dirichlet-consistency
    # and post-solid re-projections correct a tiny (~1% of total) pressure
    # increment left after the main solve, yet CG's relative tolerance is
    # scale-invariant, so with inherited settings they burn the SAME
    # iteration budget as the main solve for a physically negligible
    # correction. Production values below are the measured A/B winner
    # (th_perf_ab.py V1, 2026-07-05): 27.94 -> 16.20 s/step (1.72x) with
    # accuracy guards passing (total drag delta 0.08%, tip_uy delta 0.2%,
    # re-projection iters 4009->1229 / 2922->868). Tolerances are numerics,
    # not physics: a 1e-4 relative solve of a ~1% increment perturbs the
    # accumulated pressure by ~1e-6 relative. Set either to None to inherit
    # flow_projection_iterations/flow_cg_tolerance (legacy behavior).
    flow_reprojection_iterations: int | None = 1200
    flow_reprojection_cg_tolerance: float | None = 1.0e-4
    fluid_advection_scheme: str = "rk2"
    velocity_damping: float = 1.0
    enforce_plane_strain_x: bool = True
    mpm_support_radius_m: float | None = None
    # Strong coupling (Picard + Aitken) over the marker interface state
    # (2026-07): the default single pass is explicit loose coupling, which is
    # added-mass unstable at solid/fluid density ratio 1 (FSI3 diverges
    # monotonically in ~30 ms at ~93% of the reference load). With
    # fsi_coupling_iterations > 1, each time step re-runs the full advance
    # from a saved fluid+solid+marker base state, Aitken-relaxing the marker
    # surface state (positions/velocities/normals/areas + pressure-Neumann
    # gradients) between trials until the relative interface-velocity
    # residual drops below fsi_coupling_tolerance. 1 (default) preserves the
    # legacy single pass byte-for-byte — the gated branch is never entered.
    # Cost scales linearly with iterations used (each trial re-runs the full
    # fluid solve); that is the price of density ratio 1, not a knob to tune
    # away via looser solver tolerances.
    fsi_coupling_iterations: int = 1
    fsi_coupling_tolerance: float = 1.0e-3
    fsi_aitken_initial_relaxation: float = 0.5


def fsi1_config(**overrides: Any) -> TurekHronFsiConfig:
    return TurekHronFsiConfig(**{**TUREK_HRON_PRESET_PARAMETERS["fsi1"], **overrides})


def fsi2_config(**overrides: Any) -> TurekHronFsiConfig:
    return TurekHronFsiConfig(**{**TUREK_HRON_PRESET_PARAMETERS["fsi2"], **overrides})


def fsi3_config(**overrides: Any) -> TurekHronFsiConfig:
    return TurekHronFsiConfig(**{**TUREK_HRON_PRESET_PARAMETERS["fsi3"], **overrides})


PRESET_BUILDERS = {
    "fsi1": fsi1_config,
    "fsi2": fsi2_config,
    "fsi3": fsi3_config,
}


def solver_z_from_turek_hron_x_m(x_m: float, config: TurekHronFsiConfig) -> float:
    return float(config.channel_length_m) - float(x_m)


def cylinder_center_solver_m(config: TurekHronFsiConfig) -> tuple[float, float]:
    return (
        float(config.cylinder_center_y_m),
        solver_z_from_turek_hron_x_m(config.cylinder_center_x_m, config),
    )


def beam_root_x_m(config: TurekHronFsiConfig) -> float:
    return float(config.beam_tip_x_m) - float(config.beam_length_m)


def beam_box_solver_m(
    config: TurekHronFsiConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    half_thickness = 0.5 * float(config.beam_thickness_m)
    y_min = float(config.cylinder_center_y_m) - half_thickness
    y_max = float(config.cylinder_center_y_m) + half_thickness
    z_tip = solver_z_from_turek_hron_x_m(config.beam_tip_x_m, config)
    z_root = solver_z_from_turek_hron_x_m(beam_root_x_m(config), config)
    return (0.0, y_min, z_tip), (float(config.span_m), y_max, z_root)


def fluid_cell_spacing_m(config: TurekHronFsiConfig) -> tuple[float, float, float]:
    nx, ny, nz = (int(value) for value in config.grid_nodes)
    return (
        float(config.span_m) / float(nx),
        float(config.channel_height_m) / float(ny),
        float(config.channel_length_m) / float(nz),
    )


def inlet_ramp_factor(t_s: float, config: TurekHronFsiConfig) -> float:
    ramp_time_s = float(config.inlet_ramp_time_s)
    if ramp_time_s <= 0.0 or float(t_s) >= ramp_time_s:
        return 1.0
    if float(t_s) <= 0.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(math.pi * float(t_s) / ramp_time_s))


def inlet_profile_mps(y_m: float, t_s: float, config: TurekHronFsiConfig) -> float:
    height_m = float(config.channel_height_m)
    parabola = 4.0 * float(y_m) * (height_m - float(y_m)) / (height_m * height_m)
    return (
        1.5
        * float(config.mean_inlet_velocity_mps)
        * parabola
        * inlet_ramp_factor(t_s, config)
    )


def build_cylinder_obstacle_mask(config: TurekHronFsiConfig) -> np.ndarray:
    nx, ny, nz = (int(value) for value in config.grid_nodes)
    _, dy, dz = fluid_cell_spacing_m(config)
    center_y, center_z = cylinder_center_solver_m(config)
    y_centers = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    z_centers = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    radius_sq = float(config.cylinder_radius_m) ** 2
    distance_sq = (
        (y_centers[:, None] - center_y) ** 2 + (z_centers[None, :] - center_z) ** 2
    )
    mask_yz = (distance_sq <= radius_sq).astype(np.int32)
    return np.broadcast_to(mask_yz[None, :, :], (nx, ny, nz)).copy()


def beam_fixed_particle_mask(
    rest_positions_m: np.ndarray, config: TurekHronFsiConfig
) -> np.ndarray:
    rest = np.asarray(rest_positions_m, dtype=np.float64)
    z_root = solver_z_from_turek_hron_x_m(beam_root_x_m(config), config)
    particle_dz = float(config.beam_length_m) / float(config.solid_particle_counts[2])
    root_fixed = rest[:, 2] >= z_root - 1.01 * particle_dz
    _, dy, dz = fluid_cell_spacing_m(config)
    center_y, center_z = cylinder_center_solver_m(config)
    clamp_radius = float(config.cylinder_radius_m) + max(dy, dz)
    inside_cylinder = (
        (rest[:, 1] - center_y) ** 2 + (rest[:, 2] - center_z) ** 2
    ) <= clamp_radius * clamp_radius
    return root_fixed | inside_cylinder


def build_marker_layout(
    config: TurekHronFsiConfig,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], list[float]]:
    _, dy, dz = fluid_cell_spacing_m(config)
    box_min, box_max = beam_box_solver_m(config)
    y_low = box_min[1]
    y_high = box_max[1]
    z_tip = box_min[2]
    x_center = 0.5 * float(config.span_m)
    markers_per_side = int(config.markers_per_side)
    markers_per_tip = int(config.markers_per_tip)
    long_segment_m = float(config.beam_length_m) / float(markers_per_side)
    long_area_m2 = float(config.span_m) * long_segment_m
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    areas: list[float] = []
    for normal_sign, y_face in ((-1.0, y_low), (1.0, y_high)):
        y = y_face + normal_sign * 0.51 * dy
        for marker in range(markers_per_side):
            z = z_tip + (float(marker) + 0.5) * long_segment_m
            positions.append((x_center, y, z))
            normals.append((0.0, normal_sign, 0.0))
            areas.append(long_area_m2)
    tip_segment_m = float(config.beam_thickness_m) / float(markers_per_tip)
    tip_area_m2 = float(config.span_m) * tip_segment_m
    z = z_tip - 0.51 * dz
    # D4 note (2026-07): these markers_per_tip (default 4) tip markers face
    # -z with their "inside" probe walking into the 0.35 m beam body along
    # +z. That interior is structurally unsampleable fluid (it is solid
    # beam, not water), so under two-sided pressure sampling (traction
    # needs BOTH inside_found and outside_found) these markers were
    # permanently invalid: stress_invalid_marker_count == 4 at the default
    # config under the old two-sided sampling mode.
    #
    # D5 update (2026-07): run_turek_hron_fsi now samples one-sided per
    # face (traction = (p_ref - p_outside) * normal), which only needs the
    # "outside" probe along each marker's own +normal - here (0,0,-1), i.e.
    # away from the beam body, which IS real wetted fluid. The unsampleable
    # +z interior is never probed under one-sided sampling, so these tip
    # markers are now expected to be VALID (stress_invalid_marker_count
    # drops from 4 to 0 at the default config).
    for marker in range(markers_per_tip):
        y = y_low + (float(marker) + 0.5) * tip_segment_m
        positions.append((x_center, y, z))
        normals.append((0.0, 0.0, -1.0))
        areas.append(tip_area_m2)
    return positions, normals, areas


def thin_beam_pressure_probe_max_multiplier(config: TurekHronFsiConfig) -> float:
    # D3 fix (2026-07): base_multiplier was hardcoded to 12.0 as a
    # compensation for the oversized D1 envelope (search_radius derived
    # from max(dx, dy, dz), which included the physically irrelevant span
    # spacing dx and exceeded the beam thickness). With D1 now deriving the
    # envelope from the wall-normal/streamwise plane spacings only
    # (search_radius = 1.5 * max(dy, dz)), fall back to the code default of
    # 3.0 and recompute the thickness-driven multiplier from that same,
    # now-correct envelope. For the default grid (dy=0.41/48, dz=2.5/288)
    # this recomputes to ~6.37, so max(3.0, recomputed) == ~6.37 (not the
    # old hardcoded 12.0 floor).
    base_multiplier = 3.0
    _, dy, dz = fluid_cell_spacing_m(config)
    plane_spacing_m = max(dy, dz)
    hibm_search_envelope_m = 1.5 * plane_spacing_m
    thickness_multiplier = (
        float(config.beam_thickness_m) + hibm_search_envelope_m
    ) / dy + 2.5
    return max(base_multiplier, thickness_multiplier)


def beam_surface_force_support_radius_m(config: TurekHronFsiConfig) -> float:
    particle_dy = float(config.beam_thickness_m) / float(config.solid_particle_counts[1])
    particle_dz = float(config.beam_length_m) / float(config.solid_particle_counts[2])
    _, grid_dy, grid_dz = fluid_cell_spacing_m(config)
    local_radius = max(
        2.5 * particle_dy, 2.5 * particle_dz, 2.0 * grid_dy, 2.0 * grid_dz
    )
    thickness_limited = min(local_radius, 0.5 * float(config.beam_thickness_m))
    return max(thickness_limited, 1.25 * max(particle_dy, particle_dz))


def with_beam_surface_force_support(config: TurekHronFsiConfig) -> TurekHronFsiConfig:
    if config.mpm_support_radius_m is not None:
        return config
    return replace(
        config, mpm_support_radius_m=beam_surface_force_support_radius_m(config)
    )


def _full_bounds(
    config: TurekHronFsiConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (0.0, 0.0, 0.0), (
        float(config.span_m),
        float(config.channel_height_m),
        float(config.channel_length_m),
    )


def _build_fluid(
    config: TurekHronFsiConfig, runtime: TaichiRuntimeConfig
) -> CartesianFluidSolver:
    bounds_min, bounds_max = _full_bounds(config)
    fluid = CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            grid_nodes=config.grid_nodes,
            density_kgm3=config.fluid_density_kgm3,
            viscosity_pa_s=config.fluid_viscosity_pa_s,
            dt_s=config.dt_s,
        ),
        runtime=runtime,
    )
    grid_nodes = tuple(int(value) for value in config.grid_nodes)
    fluid.velocity.from_numpy(np.zeros((*grid_nodes, 3), dtype=np.float32))
    fluid.velocity_prev.from_numpy(np.zeros((*grid_nodes, 3), dtype=np.float32))
    fluid.pressure.from_numpy(np.zeros(grid_nodes, dtype=np.float32))
    fluid.obstacle.from_numpy(build_cylinder_obstacle_mask(config))
    fluid.clear_volume_source()
    # NOTE (2026-07-05): mode 2 = static-rows-only face-symmetric stamping
    # (fluid.velocity_dirichlet_face_symmetric, solver.py
    # _apply_velocity_dirichlet_boundary_rows_kernel). Scoped to STATIC Dirichlet
    # rows only (region_id == -1: channel walls + inlet), it fixes the top/bottom
    # wall face-constraint asymmetry: the bottom wall's fluid-adjacent y-face was
    # left unpinned while the top wall's was pinned, giving a net downward
    # transport bias (wrong tip_uy sign). Mode 1 (all Dirichlet rows, including
    # the thin beam marker rows) is known to over-seal the diffuse marker-IB band
    # and blow the beam particles out of the grid within ~25 steps — do not use
    # mode 1 here.
    fluid.velocity_dirichlet_face_symmetric = 2
    return fluid


def _build_solid(
    config: TurekHronFsiConfig, runtime: TaichiRuntimeConfig
) -> tuple[NeoHookeanMpmState, dict[str, np.ndarray]]:
    bounds_min, bounds_max = _full_bounds(config)
    count = math.prod(int(value) for value in config.solid_particle_counts)
    solid = NeoHookeanMpmState(
        particle_capacity=count,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        grid_nodes=config.grid_nodes,
        runtime=runtime,
    )
    box_min, box_max = beam_box_solver_m(config)
    solid.initialize_box(
        particle_counts=config.solid_particle_counts,
        box_min_m=box_min,
        box_max_m=box_max,
        density_kgm3=config.solid_density_kgm3,
    )
    rest = solid.rest_x.to_numpy()[:count]
    fixed = beam_fixed_particle_mask(rest, config)
    solid.fixed_particle.from_numpy(fixed.astype(np.int32))
    solid.region_id.from_numpy(np.full(count, PRIMARY_REGION_ID, dtype=np.int32))
    normals = np.zeros((count, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    solid.surface_normal.from_numpy(normals)
    solid.rest_surface_normal.from_numpy(normals)
    areas = np.full(
        count,
        float(config.beam_length_m * config.span_m) / float(count),
        dtype=np.float32,
    )
    solid.area_weight_m2.from_numpy(areas)
    solid.rest_area_weight_m2.from_numpy(areas)
    solid.external_force_n.from_numpy(np.zeros((count, 3), dtype=np.float32))
    particle_dz = float(config.beam_length_m) / float(config.solid_particle_counts[2])
    z_tip = box_min[2]
    tip = np.flatnonzero(rest[:, 2] <= z_tip + 1.01 * particle_dz)
    masks = {
        "fixed": fixed,
        "tip": tip,
        # rest positions never change; cache this one to_numpy so the per-step
        # tip/displacement row does not re-fetch the whole rest array each step
        "rest": rest,
    }
    return solid, masks


def _build_markers(
    config: TurekHronFsiConfig, runtime: TaichiRuntimeConfig
) -> HibmMpmSurfaceMarkers:
    positions, normals, areas = build_marker_layout(config)
    markers = HibmMpmSurfaceMarkers(marker_capacity=len(positions), runtime=runtime)
    markers.load_markers(
        positions_m=positions,
        velocities_mps=[(0.0, 0.0, 0.0)] * len(positions),
        normals=normals,
        areas_m2=areas,
        region_ids=[PRIMARY_REGION_ID] * len(positions),
    )
    return markers


# Aitken relaxation clamp for the marker-state Picard loop. The spec'd
# working range for a Dirichlet-Neumann velocity-residual fixed point at
# density ratio ~1: never freeze the update (lower > 0), never extrapolate
# (upper = 1.0 keeps every relaxed marker state a convex combination of
# guess and candidate, which relaxed_sharp_marker_state_arrays' normal
# renormalization and area clamping assume).
FSI_AITKEN_RELAXATION_LOWER = 0.05
FSI_AITKEN_RELAXATION_UPPER = 1.0


def _aitken_relaxation_factor(
    *,
    previous_relaxation: float,
    previous_residual: np.ndarray,
    current_residual: np.ndarray,
    lower: float = FSI_AITKEN_RELAXATION_LOWER,
    upper: float = FSI_AITKEN_RELAXATION_UPPER,
) -> float:
    """Scalar Aitken Delta^2 update for an interface-velocity residual vector.

    omega_k = -omega_{k-1} * <r_{k-1}, r_k - r_{k-1}> / ||r_k - r_{k-1}||^2,
    clamped to [lower, upper]. Local, credited copy of the squid case's
    private helper (cases/squid_soft_robot/checkpointing.py,
    _sharp_marker_aitken_relaxation) — copied instead of imported so this
    case does not depend on another case's underscore-private symbol; the
    public fsi_coupling.aitken_relaxation_factor variant is equivalent but
    converts per-component through Python floats, which is pointless overhead
    on numpy residual vectors.
    """
    previous = np.asarray(previous_residual, dtype=np.float64).reshape(-1)
    current = np.asarray(current_residual, dtype=np.float64).reshape(-1)
    if previous.shape != current.shape:
        raise ValueError("Aitken residual vectors must have the same shape")
    delta = current - previous
    denominator = float(np.dot(delta, delta))
    if denominator <= 1.0e-30:
        # Stalled residual: keep the previous relaxation instead of dividing
        # by (numerically) zero — matches both precedent implementations.
        return float(previous_relaxation)
    raw = -float(previous_relaxation) * float(np.dot(previous, delta)) / denominator
    if not math.isfinite(raw):
        return float(previous_relaxation)
    return max(float(lower), min(float(upper), raw))


def _marker_pressure_neumann_gradient_state(
    boundary: HibmMpmIbBoundaryConditions,
    markers: HibmMpmSurfaceMarkers,
) -> np.ndarray:
    """Host snapshot of the marker pressure-Neumann gradients (Picard state).

    The advance's fluid predictor rewrites these gradients every trial and
    the NEXT trial's pressure-Neumann matrix rows are assembled from them, so
    they are part of the interface fixed-point state alongside the marker
    arrays. Mirrors the squid case's sharp_pressure_neumann_gradient_state_array,
    which is bound to that case's coupling-state object and therefore not
    directly importable here.
    """
    count = int(markers.marker_count)
    field = boundary.marker_pressure_neumann_gradient_field
    return np.asarray(field.to_numpy())[:count].copy()


def _restore_marker_pressure_neumann_gradient_state(
    boundary: HibmMpmIbBoundaryConditions,
    markers: HibmMpmSurfaceMarkers,
    state: np.ndarray,
) -> None:
    """Restore marker pressure-Neumann gradients exported above."""
    count = int(markers.marker_count)
    field = boundary.marker_pressure_neumann_gradient_field
    full = field.to_numpy()
    array = np.asarray(state, dtype=full.dtype)
    expected_shape = tuple(full[:count].shape)
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            "marker pressure-Neumann gradient state shape mismatch: "
            f"{tuple(array.shape)} != {expected_shape}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("marker pressure-Neumann gradient state must be finite")
    full[:count] = array
    field.from_numpy(full)


def _write_channel_boundary_rows(
    fluid: CartesianFluidSolver, config: TurekHronFsiConfig, t_s: float
) -> None:
    nx, ny, nz = (int(value) for value in config.grid_nodes)
    _, dy, _ = fluid_cell_spacing_m(config)
    peak_scale = (
        1.5
        * float(config.mean_inlet_velocity_mps)
        * inlet_ramp_factor(t_s, config)
    )
    th_channel_boundary_rows_kernel(
        fluid.velocity_dirichlet_boundary_active,
        fluid.velocity_dirichlet_boundary_value_mps,
        fluid.velocity_dirichlet_boundary_projection_weight,
        fluid.velocity_dirichlet_boundary_marker_region_id,
        int(ny),
        int(nz - 1),
        float(dy),
        float(config.channel_height_m),
        float(peak_scale),
    )


def _outlet_flux_m3ps(fluid: CartesianFluidSolver, config: TurekHronFsiConfig) -> float:
    dx, dy, _ = fluid_cell_spacing_m(config)
    nx, ny, _ = (int(value) for value in config.grid_nodes)
    z_sum = float(outlet_zflux_sum_kernel(fluid.velocity, int(nx), int(ny)))
    return -z_sum * dx * dy


def _tip_displacement_row(
    solid: NeoHookeanMpmState, masks: dict[str, np.ndarray]
) -> dict[str, float]:
    count = int(solid.particle_count)
    current = solid.x.to_numpy()[:count]
    rest = masks["rest"]
    displacement = current - rest
    norm = np.linalg.norm(displacement, axis=1)
    tip_mean = displacement[masks["tip"]].mean(axis=0)
    return {
        "tip_mean_displacement_solver_y_m": float(tip_mean[1]),
        "tip_mean_displacement_solver_z_m": float(tip_mean[2]),
        "tip_ux_turek_hron_m": float(-tip_mean[2]),
        "tip_uy_turek_hron_m": float(tip_mean[1]),
        "max_displacement_m": float(norm.max(initial=0.0)),
        "fixed_root_max_displacement_m": float(norm[masks["fixed"]].max(initial=0.0)),
    }


HISTORY_FIELDS = (
    "step",
    "time_s",
    "ramp_factor",
    "tip_mean_displacement_solver_y_m",
    "tip_mean_displacement_solver_z_m",
    "tip_ux_turek_hron_m",
    "tip_uy_turek_hron_m",
    "max_displacement_m",
    "fixed_root_max_displacement_m",
    "marker_force_x_n",
    "marker_force_y_n",
    "marker_force_z_n",
    "beam_drag_per_span_n_per_m",
    "beam_lift_per_span_n_per_m",
    "cylinder_form_drag_per_span_n_per_m",
    "cylinder_friction_drag_per_span_n_per_m",
    "cylinder_drag_per_span_n_per_m",
    "cylinder_lift_per_span_n_per_m",
    "total_drag_per_span_n_per_m",
    "total_lift_per_span_n_per_m",
    "fluid_speed_max_mps",
    "outlet_flux_m3ps",
    "inlet_flux_target_m3ps",
    "projection_l2",
    "projection_max_abs",
    "stress_valid_marker_count",
    "stress_invalid_marker_count",
    # Strong-coupling diagnostics (legacy single pass reports the sentinels
    # 1 / 0.0 / 1.0: one trial, no measured residual, no relaxation applied).
    "fsi_coupling_iterations_used",
    "fsi_coupling_residual",
    "fsi_aitken_relaxation",
)


def _mid_span_index(nx: int) -> int:
    return int(nx) // 2


def build_turek_hron_final_fields_snapshot(
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    config: TurekHronFsiConfig,
) -> dict[str, np.ndarray]:
    """Assemble a mid-span y-z slice of the fluid state plus the deflected beam.

    Run-END operation (a to_numpy on the whole velocity/pressure/obstacle
    field is fine here). Returns a flat dict of arrays for np.savez; keys are
    self-describing so the .npz is directly plottable without this module.
    The mid-span plane is selected because the beam/cylinder obstacle is
    span-uniform (build_cylinder_obstacle_mask broadcasts along x), so a
    single y-z slice captures the full 2D benchmark wake.
    """

    nx, ny, nz = (int(value) for value in config.grid_nodes)
    span_index = _mid_span_index(nx)
    _, dy, dz = fluid_cell_spacing_m(config)
    y_centers = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    z_centers = (np.arange(nz, dtype=np.float64) + 0.5) * dz

    velocity = np.asarray(fluid.velocity.to_numpy(), dtype=np.float64)
    pressure = np.asarray(fluid.pressure.to_numpy(), dtype=np.float64)
    obstacle = np.asarray(fluid.obstacle.to_numpy())

    # Mid-span y-z slice (shape (ny, nz)).
    vel_slice = velocity[span_index]  # (ny, nz, 3)
    velocity_magnitude_yz = np.linalg.norm(vel_slice, axis=-1)
    velocity_y_yz = vel_slice[:, :, 1]
    velocity_z_yz = vel_slice[:, :, 2]
    pressure_yz = pressure[span_index]
    obstacle_mask_yz = (obstacle[span_index] != 0).astype(np.int32)

    count = int(solid.particle_count)
    current = np.asarray(solid.x.to_numpy()[:count], dtype=np.float64)
    rest = np.asarray(solid.rest_x.to_numpy()[:count], dtype=np.float64)

    return {
        "y_centers_m": y_centers,
        "z_centers_m": z_centers,
        "velocity_magnitude_yz_mps": velocity_magnitude_yz,
        "velocity_y_yz_mps": velocity_y_yz,
        "velocity_z_yz_mps": velocity_z_yz,
        "pressure_yz_pa": pressure_yz,
        "obstacle_mask_yz": obstacle_mask_yz,
        "span_index": np.asarray(span_index, dtype=np.int64),
        "grid_nodes": np.asarray([nx, ny, nz], dtype=np.int64),
        "cell_spacing_m": np.asarray([dy, dz], dtype=np.float64),
        "beam_marker_current_xyz_m": current,
        "beam_marker_rest_xyz_m": rest,
        "beam_marker_displacement_xyz_m": current - rest,
    }


def _write_final_fields_contour_png(
    snapshot: dict[str, np.ndarray], png_path: Path
) -> bool:
    """Contour of velocity magnitude with obstacle + deflected beam overlaid.

    Mirrors the flap comparison plots' style (filled contour, obstacle in a
    muted overlay, deflected marker cloud scattered on top). Returns False if
    matplotlib is unavailable, so the .npz export is never blocked by a
    missing plotting dependency.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    z_centers = np.asarray(snapshot["z_centers_m"], dtype=np.float64)
    y_centers = np.asarray(snapshot["y_centers_m"], dtype=np.float64)
    speed = np.asarray(snapshot["velocity_magnitude_yz_mps"], dtype=np.float64)
    obstacle = np.asarray(snapshot["obstacle_mask_yz"]) != 0
    displacement = np.asarray(snapshot["beam_marker_current_xyz_m"], dtype=np.float64)

    # Grids are (ny, nz); plot streamwise solver-z on x-axis, wall-normal
    # solver-y on y-axis so the channel reads left-to-right.
    zz, yy = np.meshgrid(z_centers, y_centers)

    fig, ax = plt.subplots(figsize=(12.0, 2.6))
    contour = ax.contourf(zz, yy, speed, levels=32, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="|u| (m/s)")
    obstacle_overlay = np.ma.masked_where(~obstacle, np.ones_like(speed))
    ax.contourf(zz, yy, obstacle_overlay, colors="0.35", alpha=0.65)
    if displacement.size:
        ax.scatter(
            displacement[:, 2],
            displacement[:, 1],
            s=2.0,
            c="red",
            alpha=0.6,
            label="deflected beam markers",
        )
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("solver z (streamwise) (m)")
    ax.set_ylabel("solver y (wall-normal) (m)")
    ax.set_title("Turek-Hron final velocity magnitude (mid-span y-z plane)")
    ax.set_aspect("equal")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return True


def _flush_history_csv(
    history_path: Path, rows: list[dict[str, Any]], *, header_written: bool
) -> bool:
    """Append completed history rows to the CSV on disk (robust incremental flush).

    Opens in append mode and writes the header exactly once. Returns the new
    header_written state so the caller can chain flushes without re-emitting
    the header. The final full write in run_turek_hron_fsi still rewrites the
    complete CSV, so a truncated mid-run file is always superseded by a clean
    final one.
    """

    if not rows:
        return header_written
    history_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if header_written else "w"
    with history_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HISTORY_FIELDS))
        if not header_written:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in HISTORY_FIELDS})
    return True


def run_turek_hron_fsi(
    config: TurekHronFsiConfig,
    *,
    preset: str = "fsi1",
    output_dir: Path | str | None = None,
    export_final_flow_snapshot: bool = True,
    history_flush_interval_steps: int = TUREK_HRON_HISTORY_FLUSH_INTERVAL_STEPS,
) -> dict[str, Any]:
    config = with_beam_surface_force_support(config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = _build_fluid(config, runtime)
    solid, masks = _build_solid(config, runtime)
    markers = _build_markers(config, runtime)
    bounds_min, bounds_max = _full_bounds(config)
    search = HibmMpmIbNodeSearch(
        grid_nodes=config.grid_nodes,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        marker_capacity=markers.marker_count,
        runtime=runtime,
    )
    boundary = HibmMpmIbBoundaryConditions(
        grid_nodes=config.grid_nodes,
        marker_capacity=markers.marker_count,
        runtime=runtime,
    )
    mu_pa, lambda_pa = _lame_parameters(config)
    solid_substep_dt_s = float(config.dt_s) / float(config.solid_substeps)
    solid_damping = float(config.velocity_damping) ** (
        1.0 / float(config.solid_substeps)
    )
    # D1 fix (2026-07): the beam is thin in solver y (thickness 0.02 m) and
    # spans only 4 cells in solver x (dx=0.0125 m); dx is physically
    # irrelevant to this thin-beam problem and, if included via
    # max(dx, dy, dz), inflates search_radius/interior_probe past the beam
    # thickness and saturates the classification band around the beam with
    # ambiguous nodes. Derive the envelope from the wall-normal/streamwise
    # plane spacings only (dy, dz), and size it so the envelope stays
    # strictly below the beam's physical thickness/half-thickness:
    #   search_radius  = 1.5 * plane_spacing < beam_thickness_m (0.02 m)
    #   interior_probe = 1.0 * plane_spacing < beam_thickness_m / 2 (0.01 m)
    # while both remain >= 1-1.5 plane cells for robust row formation.
    _, plane_dy_m, plane_dz_m = fluid_cell_spacing_m(config)
    plane_spacing_m = max(plane_dy_m, plane_dz_m)
    search_radius_m = 1.5 * plane_spacing_m
    interior_probe_distance_m = 1.0 * plane_spacing_m

    solid_substep_count = int(config.solid_substeps)

    # Strong-coupling gate: with the default fsi_coupling_iterations=1 the
    # gated Picard branch below is never entered, so the legacy explicit
    # loose single pass is preserved byte-for-byte (no extra save_state, no
    # marker snapshots, no residual math on the legacy path).
    fsi_iterations = max(1, int(config.fsi_coupling_iterations))
    strong_coupling_enabled = fsi_iterations > 1
    fsi_tolerance = float(config.fsi_coupling_tolerance)

    def solid_step() -> Any:
        latest = None
        for substep_index in range(solid_substep_count):
            # The per-substep report does a device->host snapshot (the
            # out-of-bounds particle safety check) that the caller discards for
            # every substep but the last; read it only on the final substep,
            # turning 100 host round-trips per step into 1. A blow-up is still
            # caught (at most one solid-step later), and correctness is
            # unaffected because only the last report is consumed.
            latest = solid.step(
                dt_s=solid_substep_dt_s,
                mu_pa=mu_pa,
                lambda_pa=lambda_pa,
                primary_region_id=PRIMARY_REGION_ID,
                secondary_region_id=SECONDARY_UNUSED_REGION_ID,
                velocity_damping=solid_damping,
                read_report=(substep_index == solid_substep_count - 1),
            )
            if config.enforce_plane_strain_x:
                solid.enforce_rest_x_plane()
        return latest

    history: list[dict[str, Any]] = []
    latest_report = None
    # Incremental history flush: append completed rows to disk every N steps so
    # a multi-hour run is inspectable mid-flight instead of a black box. The
    # final full write below still rewrites the complete CSV.
    incremental_history_path: Path | None = None
    incremental_header_written = False
    last_flushed_index = 0
    flush_interval = int(history_flush_interval_steps)
    if output_dir is not None and flush_interval > 0:
        incremental_history_path = Path(output_dir) / "turek_hron_fsi_history.csv"
        incremental_history_path.parent.mkdir(parents=True, exist_ok=True)
    for step_index in range(int(config.step_count)):
        t_s = (step_index + 1) * float(config.dt_s)
        _write_channel_boundary_rows(fluid, config, t_s)
        fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)

        def _advance_trial_once() -> Any:
            # Single sharp HIBM-MPM advance with the case's canonical argument
            # set. Both the legacy single pass and every strong-coupling trial
            # run through this one closure so their advance arguments cannot
            # drift apart — iteration 0 of the gated loop is EXACTLY the
            # legacy pass. mpm_particle_position_m/velocity_mps stay solid.x /
            # solid.v on purpose: the fluid never reads them for its Dirichlet
            # or pressure rows (those come from the markers object); they feed
            # the marker->particle force scatter and, after solid_step, the
            # particle->marker surface feedback, both of which must see the
            # TRUE re-integrated solid state each trial.
            return advance_hibm_mpm_sharp_mpm_step(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_velocity_mps=solid.v,
            mpm_particle_normal=solid.surface_normal,
            mpm_particle_area_m2=solid.area_weight_m2,
            mpm_particle_count=solid.particle_count,
            solid_step=solid_step,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=search_radius_m,
            interior_probe_distance_m=interior_probe_distance_m,
            mpm_support_radius_m=float(config.mpm_support_radius_m),
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_UNUSED_REGION_ID,
            # D2 fix (2026-07): arm the beam-aware sampling_obstacle_field
            # guard (core.py's `if int(far_pressure_region_id) != -1:`
            # branch) so the extended two-sided probe walk cannot tunnel
            # through the thin beam. SECONDARY_UNUSED_REGION_ID (202) is used
            # as a barrier-only sentinel, not PRIMARY_REGION_ID: the closure
            # branch keys off `self.region_id[marker] == far_pressure_region_id`
            # (core.py ~1861), and every real beam marker (including the
            # tip markers) carries PRIMARY_REGION_ID. Passing PRIMARY_REGION_ID
            # here would flip every beam marker into "closure" sampling mode
            # and silently substitute far_pressure_pa (0.0 by default) on
            # any marker whose two-sided walk misses one side, corrupting
            # ordinary wetted two-sided pressure sampling. No marker is ever
            # assigned SECONDARY_UNUSED_REGION_ID, so this only builds the
            # barrier and never triggers closure substitution.
            far_pressure_region_id=SECONDARY_UNUSED_REGION_ID,
            # D5 fix (2026-07): switch from two-sided pressure-jump sampling
            # to one-sided per-face sampling, mirroring the ANSYS
            # vertical-flap precedent (cases/ansys_vertical_flap_fsi.py's
            # selected_formulation_solver_config,
            # traction_pressure_sampling_mode="one_sided_surface_pressure").
            # Root cause: every beam marker (lower face, upper face, tip)
            # carries the SAME PRIMARY_REGION_ID and was sampled two-sided,
            # i.e. traction = (p_inside - p_outside) * normal, where the
            # "inside" probe deliberately crosses the 0.02 m-thin beam. Both
            # faces then get the FULL cross-beam pressure jump, so summing
            # both faces' forces double-counts the net beam force to
            # ~2 * dp * A. One-sided per-face sampling instead sets
            # traction = (p_ref - p_outside) * normal per marker, using only
            # the fluid-side probe along that marker's own outward normal
            # (gauge p_ref=0), giving each face its true net contribution
            # (net == dp * A across the whole beam, not 2x).
            #
            # All markers built by build_marker_layout already carry an
            # OUTWARD normal into the fluid on their own side (lower face
            # (0,-1,0), upper face (0,1,0), tip face (0,0,-1)) - see
            # core.py's per-face branch: fluid_side_normal_sign is relative
            # to each marker's own stored normal, not absolute space, so
            # sign=+1.0 ("sample along +normal") is correct for every face
            # simultaneously. That means all beam markers can share a
            # single per-face region slot: no marker-layout/region-id split
            # is needed, unlike the flap (which uses two distinct physical
            # faces with two distinct regions purely for anchor-pair
            # bookkeeping, not because the sign differs). Per the task
            # note: use PRIMARY_REGION_ID here (this is the one-sided
            # machinery's region key, semantically distinct from the
            # far_pressure_region_id barrier sentinel above - the barrier
            # keys off SECONDARY_UNUSED_REGION_ID, owned by no marker).
            #
            # Per-face one-sided sampling only exists on
            # sample_fluid_stress_to_marker_tractions's pressure-only fast
            # path (viscosity_pa_s == 0.0, far_pressure_region_id < 0, no
            # anchor fallback/node_anchor_cell/sampling_obstacle_field -
            # core.py's per_face_one_sided_configured guard). core.py's
            # assemble_hibm_mpm_sharp_fluid_to_mpm_loads now detects
            # per-face one-sided kwargs and automatically skips the
            # far_pressure closure/barrier machinery and clears
            # node_anchor_cell/sampling_obstacle_field for THIS call only
            # (the barrier only ever fed the viscous stress kernels, never
            # the pressure-only kernel, so per-face one-sided callers gain
            # nothing from it regardless). stress_viscosity_pa_s_override=0.0
            # decouples the traction-sampling viscosity from the real fluid
            # viscosity (1.0 Pa*s) so this call reaches that fast path
            # without changing the fluid's own physical viscosity used by
            # the predictor/projection elsewhere.
            one_sided_pressure_primary_region_id=PRIMARY_REGION_ID,
            one_sided_primary_fluid_side_normal_sign=1.0,
            stress_viscosity_pa_s_override=0.0,
            fluid_dt_s=float(config.dt_s),
            fluid_substeps=int(config.flow_predictor_substeps),
            projection_iterations=int(config.flow_projection_iterations),
            run_fluid_predictor=True,
            pressure_neumann_density_kgm3=float(config.fluid_density_kgm3),
            pressure_neumann_dt_s=float(config.dt_s),
            pressure_outlet_zmin=True,
            velocity_inlet_zmax=True,
            two_sided_probe_max_multiplier=thin_beam_pressure_probe_max_multiplier(
                config
            ),
            reset_pressure=False,
            pressure_solver=str(config.flow_pressure_solver),
            cg_tolerance=float(config.flow_cg_tolerance),
            cg_preconditioner=str(config.flow_cg_preconditioner),
            # Momentum-consistent step pressure (2026-07): the advance runs up
            # to three projections per step (main + dirichlet-consistency +
            # post-solid). The re-projections act on an already projected
            # velocity, so without increment accumulation their tiny residual
            # solve OVERWRITES the physical pressure (measured: 212 Pa ->
            # 5.6 Pa every step), flattening the cylinder stagnation field and
            # killing form drag (~0 instead of ~9 N/m; cylinder-only
            # benchmark reproduces Schaefer-Turek Cd=5.79 vs ref 5.58). With
            # accumulation the stored pressure is the sum of the step's
            # increments -- the field that actually balances momentum.
            accumulate_reprojection_pressure=True,
            # Re-projection budget/tolerance passthrough (perf, 2026-07): both
            # None by default, so every existing run of this case is
            # byte-for-byte identical until an A/B (th_perf_ab.py) picks a
            # production value for these two fields.
            reprojection_projection_iterations=(
                int(config.flow_reprojection_iterations)
                if config.flow_reprojection_iterations is not None
                else None
            ),
            reprojection_cg_tolerance=(
                float(config.flow_reprojection_cg_tolerance)
                if config.flow_reprojection_cg_tolerance is not None
                else None
            ),
            fluid_advection_scheme=str(config.fluid_advection_scheme),
            post_dirichlet_consistency_projection_iterations=1,
            update_surface_geometry_from_mpm=False,
            interpolate_velocity_dirichlet_with_interior=False,
            )

        if not strong_coupling_enabled:
            # Legacy explicit loose coupling: one pass, no save/restore, no
            # residual measurement. Sentinels 1 / 0.0 / 1.0 mirror the squid
            # precedent's "unmeasured_single_pass" reporting convention.
            latest_report = _advance_trial_once()
            fsi_coupling_iterations_used = 1
            fsi_coupling_residual = 0.0
            fsi_aitken_relaxation = 1.0
            fsi_residual_history = []
            fsi_relaxation_history = []
        else:
            # --- Strong coupling: marker-state Picard + Aitken ------------
            # WHY the marker state is the Picard variable (and NOT the solid
            # kinematics args): the fluid's velocity-Dirichlet and pressure-
            # Neumann rows are assembled from the HibmMpmSurfaceMarkers
            # fields (x_gamma_m / v_gamma_mps / n_gamma / A_gamma_m2), which
            # the advance mutates in place via the post-solid surface
            # feedback. Iterating that state (plus the marker pressure-
            # Neumann gradients written by the fluid predictor) closes the
            # per-step loop fluid tractions -> solid end-of-step velocity ->
            # marker Dirichlet rows -> fluid pressure. Precedent:
            # cases/squid_soft_robot/step_loop.py,
            # advance_sharp_marker_fixed_point_step.
            #
            # State handling per trial:
            # - fluid.restore_state() restores velocity+pressure to the step
            #   base and zeroes all marker-row scratch fields; the obstacle
            #   mask needs NO explicit restore because every advance rebuilds
            #   fluid.obstacle from the frozen hibm_base_obstacle plus a
            #   fresh IB classification (solver.py,
            #   _apply_hibm_internal_obstacles_kernel resets to base before
            #   re-marking).
            # - solid.restore_state() restores x/v/C/F, refreshes surface
            #   geometry, and zeroes external_force_n, so solid_step()
            #   re-integrates from the true step base.
            # - accumulate_reprojection_pressure accumulates from the
            #   restored base pressure each trial — no cross-trial leakage.
            fluid.save_state()
            solid.save_state()
            marker_guess = sharp_marker_state_arrays(markers)
            gradient_guess = _marker_pressure_neumann_gradient_state(
                boundary, markers
            )
            relaxation = float(config.fsi_aitken_initial_relaxation)
            previous_velocity_residual: np.ndarray | None = None
            fsi_coupling_iterations_used = 0
            fsi_coupling_residual = math.inf
            fsi_residual_history: list[float] = []
            fsi_relaxation_history: list[float] = []
            latest_report = None
            for coupling_iteration in range(fsi_iterations):
                # Restore at the start of EVERY trial, iteration 0 included —
                # exactly like the squid precedent (step_loop.py,
                # advance_sharp_marker_fixed_point_step restores before each
                # advance). WHY iteration 0 too: fluid.restore_state()
                # collapses velocity_prev := velocity and re-zeroes all row
                # scratch fields, so restoring every trial makes the Picard
                # map IDENTICAL across iterations. Skipping it at iteration 0
                # would let trial 0 sample a different map (true
                # velocity_prev history) than trials 1+ (collapsed history),
                # so the fixed point being iterated would not be the map the
                # committed trial evaluates. At iteration 0 the marker and
                # gradient restores are exact no-ops (the guess was
                # snapshotted from this same state above). The legacy
                # single-pass path never enters this branch, so its byte
                # behavior is unaffected.
                fluid.restore_state()
                solid.restore_state()
                restore_sharp_marker_state_arrays(markers, marker_guess)
                _restore_marker_pressure_neumann_gradient_state(
                    boundary, markers, gradient_guess
                )
                # restore_state cleared the case-written static wall / inlet
                # Dirichlet rows; re-write and re-stamp them exactly as at
                # the top of the step so every trial starts from the
                # identical pre-advance fluid state (the stamped velocity
                # itself was captured by save_state, so re-stamping the same
                # t_s values is idempotent on the velocity field — this
                # re-fills the row fields).
                _write_channel_boundary_rows(fluid, config, t_s)
                fluid.apply_velocity_dirichlet_boundary_rows(
                    read_report=False
                )
                latest_report = _advance_trial_once()
                fsi_coupling_iterations_used = coupling_iteration + 1
                marker_candidate = sharp_marker_state_arrays(markers)
                gradient_candidate = _marker_pressure_neumann_gradient_state(
                    boundary, markers
                )
                # Relative interface-velocity residual: candidate marker
                # velocities (what the solid returned) vs the guess the fluid
                # rows were built from. float64 up-cast before the norms so
                # the convergence test is not polluted by f32 storage noise.
                new_velocity = np.asarray(
                    marker_candidate["v_gamma_mps"], dtype=np.float64
                )
                guess_velocity = np.asarray(
                    marker_guess["v_gamma_mps"], dtype=np.float64
                )
                velocity_residual = (new_velocity - guess_velocity).reshape(-1)
                fsi_coupling_residual = float(
                    np.linalg.norm(velocity_residual)
                ) / max(1.0e-30, float(np.linalg.norm(new_velocity)))
                fsi_residual_history.append(float(fsi_coupling_residual))
                fsi_relaxation_history.append(float(relaxation))
                if fsi_coupling_residual < fsi_tolerance:
                    # Converged: keep the last advance's result (fluid and
                    # solid are already stepped consistently with the marker
                    # state the fluid saw).
                    break
                if coupling_iteration == fsi_iterations - 1:
                    # Budget exhausted: keep the last (unrelaxed) trial. Not
                    # an error — the residual column records how far the
                    # fixed point got; blind tuning belongs to the caller.
                    break
                if previous_velocity_residual is not None:
                    relaxation = _aitken_relaxation_factor(
                        previous_relaxation=relaxation,
                        previous_residual=previous_velocity_residual,
                        current_residual=velocity_residual,
                    )
                previous_velocity_residual = velocity_residual.copy()
                marker_guess = relaxed_sharp_marker_state_arrays(
                    marker_guess, marker_candidate, relaxation=relaxation
                )
                gradient_guess = (
                    relaxed_sharp_pressure_neumann_gradient_state_array(
                        gradient_guess,
                        gradient_candidate,
                        relaxation=relaxation,
                    )
                )
            fsi_aitken_relaxation = float(relaxation)
        load = latest_report.fluid_to_mpm_loads
        force_n = tuple(float(v) for v in load.marker_forces.total_marker_force_n)
        projection = load.fluid_projection
        # device reduction (returns the max speed magnitude) instead of pulling
        # the whole nx*ny*nz*3 velocity field to host every step
        speed_max_mps = float(fluid._max_fluid_speed_kernel())
        # Cylinder (obstacle mask) surface force. The beam markers sample only the
        # beam; the reference full-body drag/lift also needs the cylinder, which
        # is a velocity mask with no marker force. Full traction = pressure/form
        # + viscous/friction, integrated over the mask surface (viscous computed
        # after projection so the field is ~divergence-free).
        cyl_pressure_force_n = fluid.compute_obstacle_surface_pressure_force_n()
        cyl_viscous_force_n = fluid.compute_obstacle_surface_viscous_force_n()
        cyl_force_n = (
            cyl_pressure_force_n[0] + cyl_viscous_force_n[0],
            cyl_pressure_force_n[1] + cyl_viscous_force_n[1],
            cyl_pressure_force_n[2] + cyl_viscous_force_n[2],
        )
        span_m = float(config.span_m)
        ramp = inlet_ramp_factor(t_s, config)
        row: dict[str, Any] = {
            "step": step_index + 1,
            "time_s": t_s,
            "ramp_factor": ramp,
            **_tip_displacement_row(solid, masks),
            "marker_force_x_n": force_n[0],
            "marker_force_y_n": force_n[1],
            "marker_force_z_n": force_n[2],
            "beam_drag_per_span_n_per_m": -force_n[2] / span_m,
            "beam_lift_per_span_n_per_m": force_n[1] / span_m,
            "cylinder_form_drag_per_span_n_per_m": -cyl_pressure_force_n[2] / span_m,
            "cylinder_friction_drag_per_span_n_per_m": -cyl_viscous_force_n[2] / span_m,
            "cylinder_drag_per_span_n_per_m": -cyl_force_n[2] / span_m,
            "cylinder_lift_per_span_n_per_m": cyl_force_n[1] / span_m,
            "total_drag_per_span_n_per_m": -(force_n[2] + cyl_force_n[2]) / span_m,
            "total_lift_per_span_n_per_m": (force_n[1] + cyl_force_n[1]) / span_m,
            "fluid_speed_max_mps": speed_max_mps,
            "outlet_flux_m3ps": _outlet_flux_m3ps(fluid, config),
            "inlet_flux_target_m3ps": (
                ramp
                * float(config.mean_inlet_velocity_mps)
                * float(config.channel_height_m)
                * float(config.span_m)
            ),
            "projection_l2": float(projection.get("l2", 0.0)),
            "projection_max_abs": float(projection.get("max_abs", 0.0)),
            "stress_valid_marker_count": int(load.fluid_stress.valid_marker_count),
            "stress_invalid_marker_count": int(load.fluid_stress.invalid_marker_count),
            "fsi_coupling_iterations_used": int(fsi_coupling_iterations_used),
            "fsi_coupling_residual": float(fsi_coupling_residual),
            "fsi_aitken_relaxation": float(fsi_aitken_relaxation),
            # Per-iteration diagnostics (JSON summary only — NOT in
            # HISTORY_FIELDS, so the CSV schema stays at the three spec'd
            # strong-coupling columns). Empty lists on the legacy path.
            "fsi_coupling_residual_history": list(fsi_residual_history),
            "fsi_aitken_relaxation_history": list(fsi_relaxation_history),
            # Discrete-state observability counters (2026-07): the T-H thin-beam
            # case suffered a discrete jump caused by EXTERNAL_IB row-set
            # membership flips (velocity-Dirichlet rows added/removed as
            # advected markers cross the search-radius distance gate). These
            # counters are read from latest_report.next_* so they reflect the
            # row-set state that governs the *next* step (where a flip becomes
            # active), making future flips visible in the per-step history
            # instead of only in the internal sharp-step report.
            "hibm_next_external_ib_node_count": int(
                latest_report.next_ib_node_search.external_ib_node_count
            ),
            "hibm_next_internal_node_count": int(
                latest_report.next_ib_node_search.internal_node_count
            ),
            "hibm_next_internal_obstacle_cell_count": int(
                latest_report.next_internal_obstacle_cell_count
            ),
            "hibm_next_velocity_dirichlet_active_rows": int(
                latest_report.next_velocity_dirichlet.active_velocity_dirichlet_rows
            ),
            "hibm_next_pressure_neumann_active_rows": int(
                latest_report.next_pressure_neumann.active_pressure_neumann_rows
            ),
            "hibm_next_solid_band_nonprojectable_cell_count": int(
                latest_report.next_solid_band_nonprojectable_cell_count
            ),
            "hibm_stress_two_sided_extended_marker_count": int(
                load.fluid_stress.two_sided_extended_marker_count
            ),
        }
        history.append(row)
        if (
            incremental_history_path is not None
            and (step_index + 1) % flush_interval == 0
        ):
            incremental_header_written = _flush_history_csv(
                incremental_history_path,
                history[last_flushed_index:],
                header_written=incremental_header_written,
            )
            last_flushed_index = len(history)
    # Flush any trailing rows that did not land on a flush boundary.
    if incremental_history_path is not None and last_flushed_index < len(history):
        incremental_header_written = _flush_history_csv(
            incremental_history_path,
            history[last_flushed_index:],
            header_written=incremental_header_written,
        )
        last_flushed_index = len(history)
    if latest_report is None:
        raise RuntimeError("turek-hron FSI run did not advance")
    summary: dict[str, Any] = {
        "case": TUREK_HRON_CASE_ID,
        "preset": str(preset),
        "config": asdict(config),
        "solver_path": "advance_hibm_mpm_sharp_mpm_step",
        "wall_boundary_model": TUREK_HRON_WALL_BOUNDARY_MODEL,
        "reference_results": TUREK_HRON_REFERENCE_RESULTS.get(str(preset), {}),
        "completed_steps": len(history),
        "history": history,
        "final": history[-1],
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "turek_hron_fsi_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        history_path = out / "turek_hron_fsi_history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(HISTORY_FIELDS))
            writer.writeheader()
            for row in history:
                writer.writerow({key: row[key] for key in HISTORY_FIELDS})
        summary["summary_json"] = str(summary_path)
        summary["history_csv"] = str(history_path)
        if export_final_flow_snapshot:
            snapshot = build_turek_hron_final_fields_snapshot(fluid, solid, config)
            npz_path = out / "turek_hron_final_fields.npz"
            np.savez(npz_path, **snapshot)
            summary["final_fields_npz"] = str(npz_path)
            png_path = out / "turek_hron_final_fields.png"
            if _write_final_fields_contour_png(snapshot, png_path):
                summary["final_fields_png"] = str(png_path)
    return summary


def _parse_grid_nodes(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in str(value).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--grid-nodes expects 'nx,ny,nz', got {value!r}"
        )
    return parts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Turek-Hron FSI benchmark (sharp HIBM-MPM)."
    )
    parser.add_argument("--preset", choices=tuple(PRESET_BUILDERS), default="fsi1")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--grid-nodes", type=_parse_grid_nodes, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--projection-iterations", type=int, default=None)
    parser.add_argument(
        "--fsi-coupling-iterations",
        type=int,
        default=None,
        help=(
            "Strong-coupling Picard iterations per time step "
            "(1 = legacy explicit loose single pass, the default)."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-tolerance",
        type=float,
        default=None,
        help="Relative interface-velocity residual convergence tolerance.",
    )
    parser.add_argument(
        "--fsi-aitken-initial-relaxation",
        type=float,
        default=None,
        help="Initial Aitken relaxation factor for the strong-coupling loop.",
    )
    parser.add_argument(
        "--no-final-flow-snapshot",
        action="store_true",
        help="Disable the run-end turek_hron_final_fields.npz / .png export.",
    )
    parser.add_argument(
        "--history-flush-interval-steps",
        type=int,
        default=TUREK_HRON_HISTORY_FLUSH_INTERVAL_STEPS,
        help="Append history rows to the CSV every N steps (0 disables incremental flush).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.steps is not None:
        overrides["step_count"] = int(args.steps)
    if args.grid_nodes is not None:
        overrides["grid_nodes"] = args.grid_nodes
    if args.projection_iterations is not None:
        overrides["flow_projection_iterations"] = int(args.projection_iterations)
    if args.fsi_coupling_iterations is not None:
        overrides["fsi_coupling_iterations"] = int(args.fsi_coupling_iterations)
    if args.fsi_coupling_tolerance is not None:
        overrides["fsi_coupling_tolerance"] = float(args.fsi_coupling_tolerance)
    if args.fsi_aitken_initial_relaxation is not None:
        overrides["fsi_aitken_initial_relaxation"] = float(
            args.fsi_aitken_initial_relaxation
        )
    config = PRESET_BUILDERS[str(args.preset)](**overrides)
    summary = run_turek_hron_fsi(
        config,
        preset=str(args.preset),
        output_dir=args.output_dir,
        export_final_flow_snapshot=not bool(args.no_final_flow_snapshot),
        history_flush_interval_steps=int(args.history_flush_interval_steps),
    )
    final = summary["final"]
    print(
        f"turek-hron {args.preset}: steps={summary['completed_steps']} "
        f"tip_ux={final['tip_ux_turek_hron_m']:.6e} m "
        f"tip_uy={final['tip_uy_turek_hron_m']:.6e} m "
        f"drag/span={final['beam_drag_per_span_n_per_m']:.6e} N/m "
        f"lift/span={final['beam_lift_per_span_n_per_m']:.6e} N/m "
        f"umax={final['fluid_speed_max_mps']:.6e} m/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
