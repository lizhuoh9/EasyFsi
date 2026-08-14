from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.official.solid_mpm_fsi_runner import (
    PRIMARY_REGION_ID,
    SECONDARY_UNUSED_REGION_ID,
    _lame_parameters,
)
from simulation_core.coupling.hibm_mpm import (
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    MARKER_INTERFACE_STATE_FIELDS,
    advance_hibm_mpm_sharp_mpm_step,
    capture_marker_interface_state,
    marker_trial_state,
    marker_velocity_state,
    restore_marker_interface_state,
)
# Tier-2 marker re-seeding (2026-07-09): host-only, numpy-only arc-length
# polyline resampler; see the module docstring for why it has no Taichi
# dependency despite living in a Taichi-heavy module.
from simulation_core.coupling.marker_seeding import (
    resample_polyline_markers_by_arc_length,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingConfig,
    FsiCouplingConvergenceError,
    FsiCouplingReport,
    FsiSolverConfig,
    FsiStepContext,
    FsiTrialResult,
    solve_fsi_runtime,
)
from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec
from simulation_core.solids.neo_hookean_mpm import (
    CONSTITUTIVE_MODELS,
    NeoHookeanMpmState,
)
from cases.turek_hron_kernels import (
    boundary_zflux_sums_kernel,
    outlet_zflux_sum_kernel,
    th_channel_external_velocity_faces_kernel,
)


TUREK_HRON_CASE_ID = "turek-hron-fsi"

# Every-N-steps incremental history flush (progress inspectable mid-run).
TUREK_HRON_HISTORY_FLUSH_INTERVAL_STEPS = 25

# Diagnostic-only committed-state checkpoint format.  This is deliberately
# case-local: it captures the exact FSI transition state needed to replay a
# narrow failure window without pretending to be a general production restart
# format.
TUREK_HRON_TRANSITION_CHECKPOINT_VERSION = 1

# Channel walls (solver y=0 and y=channel_height_m) are physical no-slip walls.
# They are exact external component-face constraints. The parabolic inlet lives
# on zmax; zmin remains the p=0 pressure outlet.
TUREK_HRON_WALL_BOUNDARY_MODEL = "canonical_external_component_faces"


def _advance_particle_position_generation(current_generation: int) -> int:
    """Return the next owner generation after a write to ``solid.x``."""

    return int(current_generation) + 1

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
    solid_constitutive_model: str = "saint_venant_kirchhoff"
    dt_s: float = 5.0e-3
    step_count: int = 200
    grid_nodes: tuple[int, int, int] = (4, 48, 288)
    solid_particle_counts: tuple[int, int, int] = (1, 8, 140)
    # int = explicit count (default 48/4, legacy byte-for-byte). "auto" =
    # grid-adaptive: ceil(face_length / (0.75 * face-tangential cell size)),
    # guaranteeing >= 1 marker per surface cell with 25% margin at ANY grid,
    # so refining the grid can never reopen the porous-interface gap that the
    # 2026-07-09 diagnosis traced to this grid-independent default.
    markers_per_side: int | str = 48
    markers_per_tip: int | str = 4
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
    # A sharp boundary lies between Cartesian cell centres.  The active IB row
    # therefore needs the linearly reconstructed value between the prescribed
    # wall velocity and an interior-fluid sample; stamping the wall velocity
    # directly at the cell centre moves the effective wall by O(h) and leaves a
    # large marker-sampled slip.  Keep the core's physically consistent default
    # explicit in this benchmark so the formal certificate cannot silently use
    # the old boundary-velocity-only shortcut.
    interpolate_velocity_dirichlet_with_interior: bool = True
    velocity_damping: float = 1.0
    enforce_plane_strain_x: bool = True
    mpm_support_radius_m: float | None = None
    # Marker-velocity coupling controls consumed by the sole generic FSI
    # runtime. Every physical step and every interface trial follows that path.
    fsi_coupling_iterations: int = 16
    fsi_coupling_tolerance: float = 1.0e-3
    fsi_coupling_initial_relaxation: float = 0.5
    # Absolute interface-velocity convergence floor (2026-07-10). The Picard
    # residual is RELATIVE (||dv|| / ||v_new||); during the inlet ramp the
    # interface velocities are ~0, so the relative residual is noise/noise and
    # never reaches the tolerance -- every early step burns the full iteration
    # budget on dynamically irrelevant mismatches (measured: 300 s/step at
    # 10 iterations while ||dv|| ~ 2e-5 m/s). With this floor, a step also
    # counts as converged when the per-marker RMS absolute mismatch (m/s) drops
    # below the floor: interface velocity errors far below the physical velocity
    # scale (mean inlet 0.2 m/s) cannot feed the added-mass loop.
    # 0.0 disables the absolute gate.
    fsi_coupling_absolute_tolerance_mps: float = 0.0
    # Periodic flow-contour snapshot export (2026-07-09, observability parity
    # with the run-end export_final_flow_snapshot output): None (default)
    # preserves current behavior byte-for-byte -- nothing is written mid-run.
    # Set to an integer N to additionally write
    # output_dir/flow_snapshots/step_{step:06d}.npz every N physical steps,
    # reusing the SAME build_turek_hron_final_fields_snapshot builder as the
    # final export, so a velocity-contour animation can be rendered across the
    # whole run instead of only its last frame.
    flow_snapshot_interval_steps: int | None = None
    # Anisotropic IB classification envelope (gated, 2026-07-09; see the
    # _validate_marker_grid_consistency docstring for the full diagnosis).
    # False (default) preserves the scalar search_radius_m = 1.5*max(dy,dz)
    # envelope byte-for-byte. True switches run_turek_hron_fsi to a per-axis
    # box acceptance |d_i| < 1.5*h_i, so the boundary band scales with each
    # face's own normal cell size instead of the coarser of dy/dz. This is
    # what lets a y-only-refined grid (dy shrunk, dz unchanged) keep a thin
    # wall-normal band on the beam's long faces without the band reaching
    # through the beam interior -- the failure mode that produced 880x
    # spurious lift at rest on grid_nodes=(4, 96, 288) with the isotropic
    # envelope.
    ib_anisotropic_envelope: bool = False
    # Global signed-distance interior classification (2026-07-09). The band
    # classification only reaches search_radius from each face; on refined
    # grids the beam interior extends beyond both faces' bands, leaving
    # beam-center cells UNCLASSIFIED: sealed "fluid" pockets between the
    # Dirichlet bands whose pressure blocks are near-singular (measured on
    # (4,96,288)+aniso: only 36% of beam-interior cells obstacle-flagged and
    # one CG solve per advance burning its full budget; with this flag the
    # interior is 100% covered). False preserves the legacy base-grid
    # behavior byte-for-byte (there the 2.3-cell interior is fully inside the
    # bands, so far classification never fires). Geometrically sound for this
    # case: the marker surface's only opening (beam root) is embedded in the
    # cylinder obstacle mask.
    classify_far_internal_nodes: bool = False
    # Tier-2 marker re-seeding (2026-07-09). build_marker_layout() builds the
    # beam surface markers ONCE at rest and every step thereafter they just
    # advect with the solid (position/velocity/normal/area written back from
    # the MPM surface feedback) -- there is no re-parametrization. Under
    # large deformation (FSI2 +/-80 mm tip displacement) that advection lets
    # the markers' along-curve spacing drift far from uniform, clustering on
    # the compressed side of a bend and thinning on the stretched side.
    # None (default) preserves legacy advect-only marker tracking
    # byte-for-byte -- the gated branch in run_turek_hron_fsi is never
    # entered. Set to a positive integer N to additionally resample each of
    # the three marker face-curves (lower face, upper face, tip cap) by arc
    # length on their CURRENT deformed positions every N physical steps
    # (see _reseed_turek_hron_markers), restoring near-uniform spacing
    # without changing per-group or total marker count. Precedent: the squid
    # case's marker_remap_interval_steps gate (this repo's top-level
    # CLAUDE.md, "B2 marker 重建").
    marker_reseed_interval_steps: int | None = None


@dataclass(frozen=True)
class TurekHronMechanismProbe:
    """Opt-in host-only guard for known finite-but-runaway FSI1 signatures."""

    min_step: int = 180
    consecutive_steps: int = 10
    max_displacement_m: float = 1.0e-2
    fixed_root_max_displacement_m: float = 1.0e-8
    projection_l2: float = 1.0e-1
    projection_max_abs: float = 10.0
    fluid_speed_max_mps: float = 5.0e-1
    expected_valid_marker_count: int = 100


@dataclass(frozen=True)
class TurekHronMechanismProbeDecision:
    triggered: bool
    reason: str
    streaks: dict[str, int]


class TurekHronMechanismProbeTriggered(RuntimeError):
    """Raised after the triggering completed row has been persisted."""


def _evaluate_turek_hron_mechanism_probe(
    probe: TurekHronMechanismProbe | None,
    row: dict[str, object],
    *,
    streaks: dict[str, int] | None = None,
) -> TurekHronMechanismProbeDecision:
    next_streaks = dict(streaks or {})
    if probe is None:
        return TurekHronMechanismProbeDecision(False, "", next_streaks)

    monitored = (
        "max_displacement_m",
        "fixed_root_max_displacement_m",
        "projection_l2",
        "projection_max_abs",
        "fluid_speed_max_mps",
    )
    for name in monitored:
        value = float(row[name])
        if not math.isfinite(value):
            return TurekHronMechanismProbeDecision(
                True, f"nonfinite:{name}", next_streaks
            )

    expected_valid_marker_count = int(
        row.get(
            "stress_expected_marker_count",
            probe.expected_valid_marker_count,
        )
    )
    if (
        int(row["stress_valid_marker_count"])
        != expected_valid_marker_count
        or int(row["stress_invalid_marker_count"]) != 0
    ):
        return TurekHronMechanismProbeDecision(
            True, "marker_integrity", next_streaks
        )
    if float(row["fixed_root_max_displacement_m"]) > float(
        probe.fixed_root_max_displacement_m
    ):
        return TurekHronMechanismProbeDecision(
            True, "fixed_root_displacement", next_streaks
        )

    step = int(row["step"])
    if step < int(probe.min_step):
        next_streaks["projection_runaway"] = 0
        next_streaks["fluid_speed_runaway"] = 0
        return TurekHronMechanismProbeDecision(False, "", next_streaks)
    if float(row["max_displacement_m"]) > float(probe.max_displacement_m):
        return TurekHronMechanismProbeDecision(
            True, "max_displacement", next_streaks
        )

    projection_bad = (
        float(row["projection_l2"]) > float(probe.projection_l2)
        and float(row["projection_max_abs"]) > float(probe.projection_max_abs)
    )
    next_streaks["projection_runaway"] = (
        int(next_streaks.get("projection_runaway", 0)) + 1
        if projection_bad
        else 0
    )
    speed_bad = float(row["fluid_speed_max_mps"]) > float(
        probe.fluid_speed_max_mps
    )
    next_streaks["fluid_speed_runaway"] = (
        int(next_streaks.get("fluid_speed_runaway", 0)) + 1
        if speed_bad
        else 0
    )
    if next_streaks["projection_runaway"] >= int(probe.consecutive_steps):
        return TurekHronMechanismProbeDecision(
            True, "projection_runaway", next_streaks
        )
    if next_streaks["fluid_speed_runaway"] >= int(probe.consecutive_steps):
        return TurekHronMechanismProbeDecision(
            True, "fluid_speed_runaway", next_streaks
        )
    return TurekHronMechanismProbeDecision(False, "", next_streaks)


def _history_flush_required(
    *, completed_step: int, flush_interval: int, probe_triggered: bool
) -> bool:
    if bool(probe_triggered):
        return True
    interval = int(flush_interval)
    return interval > 0 and int(completed_step) % interval == 0


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
    center_y, center_z = cylinder_center_solver_m(config)
    clamp_radius = float(config.cylinder_radius_m)
    inside_cylinder = (
        (rest[:, 1] - center_y) ** 2 + (rest[:, 2] - center_z) ** 2
    ) <= clamp_radius * clamp_radius
    return root_fixed | inside_cylinder


def resolved_marker_counts(config: TurekHronFsiConfig) -> tuple[int, int]:
    """Resolve markers_per_side / markers_per_tip, honoring "auto".

    "auto" derives the counts from the CURRENT grid so marker long-spacing can
    never exceed the face-tangential cell size (the porous-interface failure):
    side markers span beam_length_m along z -> spacing target 0.75*dz; tip
    markers span beam_thickness_m along y -> target 0.75*dy. Floored at the
    legacy 48/4 so "auto" on the base grid is never sparser than the validated
    default. Explicit ints pass through unchanged (legacy byte-for-byte).
    """
    _, dy, dz = fluid_cell_spacing_m(config)
    side = config.markers_per_side
    tip = config.markers_per_tip
    if isinstance(side, str):
        if side.strip().lower() != "auto":
            raise ValueError("markers_per_side must be an int or 'auto'")
        side = max(48, math.ceil(float(config.beam_length_m) / (0.75 * dz)))
    if isinstance(tip, str):
        if tip.strip().lower() != "auto":
            raise ValueError("markers_per_tip must be an int or 'auto'")
        tip = max(4, math.ceil(float(config.beam_thickness_m) / (0.75 * dy)))
    return int(side), int(tip)


def build_marker_layout(
    config: TurekHronFsiConfig,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], list[float]]:
    box_min, box_max = beam_box_solver_m(config)
    y_low = box_min[1]
    y_high = box_max[1]
    z_tip = box_min[2]
    x_center = 0.5 * float(config.span_m)
    markers_per_side, markers_per_tip = resolved_marker_counts(config)
    long_segment_m = float(config.beam_length_m) / float(markers_per_side)
    long_area_m2 = float(config.span_m) * long_segment_m
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    areas: list[float] = []
    for normal_sign, y_face in ((-1.0, y_low), (1.0, y_high)):
        # Marker geometry is the physical wetted surface. Pressure/viscous
        # probes walk outward from this location; displacing the marker itself
        # by half a fluid cell made the numerical beam 43.5% too thick on the
        # default grid and polluted classification, loads, and rendered masks.
        y = y_face
        for marker in range(markers_per_side):
            z = z_tip + (float(marker) + 0.5) * long_segment_m
            positions.append((x_center, y, z))
            normals.append((0.0, normal_sign, 0.0))
            areas.append(long_area_m2)
    tip_segment_m = float(config.beam_thickness_m) / float(markers_per_tip)
    tip_area_m2 = float(config.span_m) * tip_segment_m
    z = z_tip
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


def _validate_marker_grid_consistency(config: TurekHronFsiConfig) -> None:
    """Guard against the 2026-07-09 porous-beam failure mode.

    Diagnosis: markers_per_side is grid-independent (default 48) and the HIBM
    classification envelope is search_radius_m = 1.5 * max(dy, dz) (see the D1
    comment in run_turek_hron_fsi and beam_box_solver_m). Refining ONLY ny
    (wall-normal) while leaving markers_per_side and nz unscaled shrinks dy but
    leaves max(dy, dz) pinned to the now-unchanged dz, so the search radius in
    PHYSICAL units does not shrink even though the beam half-thickness measured
    in dy-cells does. The classification bands grown from the beam's two
    opposing faces then reach through the interior and overlap, so the sharp
    fluid-solid interface leaks (measured: 880x spurious lift at rest on
    grid_nodes=(4, 96, 288) with markers_per_side=48 -- a y-only refinement of
    the (4, 48, 288)/48 base that left markers_per_side and nz untouched). The
    isotropic fine grid (4, 96, 576)/96 -- which refines nz and
    markers_per_side alongside ny -- does not exhibit this failure.

    Two independent checks:
      1. Marker long-spacing (beam_length_m / markers_per_side, the spacing of
         the surface-marker rows along the beam's length) must not exceed the
         streamwise cell size dz, else the marker rows develop gaps.
      2. The classification search radius (1.5 * max(dy, dz)) must not reach
         through the beam half-thickness as measured in wall-normal (dy)
         cells: require search_radius_m <= 0.5 * beam_thickness_m + 0.5 * dy.
         The 0.5 * dy slack lets the search band legitimately claim the
         nearest interior cell layer without letting it meet the opposing
         face's band. This must be a dy-weighted (cell-based) check, not a
         plain physical bound like "1.5 * max(dy, dz) <= 0.75 * beam_thickness_m":
         that naive form is blind to y-only refinement, because max(dy, dz) is
         set by the UNCHANGED dz in both the working base grid and the broken
         y-only-refined grid, so it cannot tell them apart. Folding in dy
         explicitly (which DOES shrink under y-only refinement) is what makes
         the check catch the regression.

    Worked verification (2026-07-09, exact numbers via fluid_cell_spacing_m):
      base    (4, 48, 288) grid, 48 markers/side:
        dy=8.541667 mm, dz=8.680556 mm
        long_spacing=7.291667 mm <= 1.05*dz=9.114583 mm -> OK
        search_radius=1.5*max(dy,dz)=13.020833 mm
        band_limit=0.5*20+0.5*dy=14.270833 mm -> 13.020833 <= 14.270833 -> PASS
      fine-iso (4, 96, 576) grid, 96 markers/side:
        dy=4.270833 mm, dz=4.340278 mm
        long_spacing=3.645833 mm <= 1.05*dz=4.557292 mm -> OK
        search_radius=1.5*max(dy,dz)=6.510417 mm
        band_limit=0.5*20+0.5*dy=12.135417 mm -> 6.510417 <= 12.135417 -> PASS
      broken  (4, 96, 288) grid, 48 markers/side (y-only refined, unscaled markers):
        dy=4.270833 mm, dz=8.680556 mm (dz UNCHANGED from base)
        long_spacing=7.291667 mm <= 1.05*dz=9.114583 mm -> OK (long-spacing
          check alone does NOT catch this failure -- markers_per_side and dz
          are both unchanged from the passing base grid)
        search_radius=1.5*max(dy,dz)=13.020833 mm (SAME as base: dz still
          dominates the max and dz did not change)
        band_limit=0.5*20+0.5*dy=12.135417 mm -> 13.020833 > 12.135417 -> FAILS

    Anisotropic escape valve (config.ib_anisotropic_envelope=True, 2026-07-09):
    when the per-axis classification box is armed (see core.py's
    _search_and_classify_kernel), the beam's long faces are y-normal, so their
    band depth is governed by search_radius_y_m = 1.5*dy alone -- dz never
    enters the wall-normal budget. Check 2 above becomes purely dy-based:
    1.5*dy <= 0.5*beam_thickness_m + 0.5*dy (equivalently dy <=
    0.5*beam_thickness_m), dropping the max(dy, dz) coupling that made the
    isotropic check blind to y-only refinement. On the "broken" grid above
    this recomputes to 1.5*4.270833=6.510417 mm <= 12.135417 mm -> PASSES.
    """
    _, dy, dz = fluid_cell_spacing_m(config)
    beam_length_m = float(config.beam_length_m)
    beam_thickness_m = float(config.beam_thickness_m)
    # Resolve "auto" first: the guard must validate the counts actually used.
    markers_per_side, _ = resolved_marker_counts(config)

    long_spacing_m = beam_length_m / float(markers_per_side)
    long_spacing_limit_m = 1.05 * dz
    if long_spacing_m > long_spacing_limit_m:
        required_markers_per_side = math.ceil(beam_length_m / dz)
        raise ValueError(
            "Turek-Hron marker grid mismatch: beam-length marker spacing "
            f"{long_spacing_m:.6e} m (beam_length_m={beam_length_m:.4f} / "
            f"markers_per_side={markers_per_side}) exceeds 1.05x the "
            f"streamwise cell size dz={dz:.6e} m (limit {long_spacing_limit_m:.6e} m), "
            "which leaves gaps in the beam surface marker rows. Increase "
            f"markers_per_side to at least {required_markers_per_side} "
            "(ceil(beam_length_m / dz))."
        )

    band_limit_m = 0.5 * beam_thickness_m + 0.5 * dy
    if bool(config.ib_anisotropic_envelope):
        # Anisotropic path: the per-axis box uses search_radius_y_m = 1.5*dy
        # for the beam's y-normal long faces, so dz cannot mask a y-only
        # refinement the way the isotropic max(dy, dz) does above.
        search_radius_y_m = 1.5 * dy
        if search_radius_y_m > band_limit_m:
            raise ValueError(
                "Turek-Hron marker grid mismatch: the anisotropic HIBM "
                f"classification y-normal search radius {search_radius_y_m:.6e} m "
                f"(= 1.5 * dy={dy:.6e} m) exceeds the wall-normal band budget "
                f"{band_limit_m:.6e} m (= 0.5*beam_thickness_m="
                f"{beam_thickness_m:.4f} + 0.5*dy={dy:.6e} m). The classification "
                "bands grown from the beam's opposing y-faces would overlap "
                "inside the beam interior, leaking the sharp fluid-solid "
                "interface (a porous beam). Refine ny further (or increase "
                "beam_thickness_m) so that dy <= beam_thickness_m / 2."
            )
    else:
        search_radius_m = 1.5 * max(dy, dz)
        if search_radius_m > band_limit_m:
            raise ValueError(
                "Turek-Hron marker grid mismatch: the HIBM classification search "
                f"radius {search_radius_m:.6e} m (= 1.5 * max(dy={dy:.6e} m, "
                f"dz={dz:.6e} m)) exceeds the wall-normal band budget "
                f"{band_limit_m:.6e} m (= 0.5*beam_thickness_m="
                f"{beam_thickness_m:.4f} + 0.5*dy={dy:.6e} m). The classification "
                "bands grown from the beam's opposing faces would overlap inside "
                "the beam interior, leaking the sharp fluid-solid interface (a "
                "porous beam -- see the 2026-07-09 diagnosis: y-only refinement "
                "with unscaled markers measured 880x spurious lift at rest). "
                "Refine y and z together (e.g. scale nz and markers_per_side "
                "alongside ny) so that max(dy, dz) <= beam_thickness_m / 2. "
                "Alternatively, set ib_anisotropic_envelope=True so the "
                "classification band depth is governed by dy alone."
            )


def _validate_fsi_coupling_controls(config: TurekHronFsiConfig) -> None:
    """Reject malformed coupling/reseed controls before any solver state exists.

    Sibling of _validate_marker_grid_consistency, called from
    run_turek_hron_fsi before Taichi state is built. These fields gate
    branches deep inside the step loop, where malformed values either
    silently change the physics (fsi_coupling_iterations=0 used to be
    clamped up to 1 by max(1, ...), masking a config typo) or crash mid-run
    (marker_reseed_interval_steps=0 reaches ``step_index % 0`` at the first
    gated reseed check -- a ZeroDivisionError at step 2 instead of a config
    error at step 0).
    """
    solid_model = str(config.solid_constitutive_model)
    if solid_model not in CONSTITUTIVE_MODELS:
        raise ValueError(
            "solid_constitutive_model must be one of "
            f"{sorted(CONSTITUTIVE_MODELS)!r}; got {solid_model!r}"
        )
    try:
        axial_particle_count = config.solid_particle_counts[2]
    except (TypeError, IndexError) as exc:
        raise ValueError(
            "solid_particle_counts[2] must be an int >= 2 for Point A "
            "extrapolation"
        ) from exc
    if (
        isinstance(axial_particle_count, bool)
        or not isinstance(axial_particle_count, int)
        or axial_particle_count < 2
    ):
        raise ValueError(
            "solid_particle_counts[2] must be an int >= 2 for Point A "
            f"extrapolation; got {axial_particle_count!r}"
        )

    iterations = config.fsi_coupling_iterations
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError(
            "fsi_coupling_iterations must be an int >= 2; got "
            f"{iterations!r} ({type(iterations).__name__})"
        )
    if iterations < 2:
        raise ValueError(
            f"fsi_coupling_iterations must be an int >= 2; got {iterations}"
        )

    tolerance = config.fsi_coupling_tolerance
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fsi_coupling_tolerance must be a finite float > 0; got "
            f"{tolerance!r}"
        ) from exc
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError(
            "fsi_coupling_tolerance must be a finite float > 0; got "
            f"{tolerance!r}. It is the relative interface-velocity residual "
            "threshold for the strong-coupling loop."
        )

    absolute_tolerance = config.fsi_coupling_absolute_tolerance_mps
    try:
        absolute_tolerance_value = float(absolute_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fsi_coupling_absolute_tolerance_mps must be a finite float >= 0; "
            f"got {absolute_tolerance!r}"
        ) from exc
    if (
        not math.isfinite(absolute_tolerance_value)
        or absolute_tolerance_value < 0.0
    ):
        raise ValueError(
            "fsi_coupling_absolute_tolerance_mps must be a finite float >= 0; "
            f"got {absolute_tolerance!r}. 0 disables the absolute convergence "
            "gate."
        )

    relaxation = config.fsi_coupling_initial_relaxation
    try:
        relaxation_value = float(relaxation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fsi_coupling_initial_relaxation must be a finite float in "
            f"(0, 1]; got {relaxation!r}"
        ) from exc
    if not math.isfinite(relaxation_value) or not 0.0 < relaxation_value <= 1.0:
        raise ValueError(
            "fsi_coupling_initial_relaxation must be a finite float in "
            f"(0, 1]; got {relaxation!r}"
        )

    reseed_interval = config.marker_reseed_interval_steps
    if reseed_interval is not None:
        if isinstance(reseed_interval, bool) or not isinstance(
            reseed_interval, int
        ):
            raise ValueError(
                "marker_reseed_interval_steps must be None or an int >= 1; "
                f"got {reseed_interval!r} ({type(reseed_interval).__name__})"
            )
        if reseed_interval < 1:
            raise ValueError(
                "marker_reseed_interval_steps must be None or an int >= 1; "
                f"got {reseed_interval}. 0 would evaluate "
                "``step_index % 0`` at the first gated reseed check "
                "(ZeroDivisionError at step 2); use None to disable "
                "re-seeding."
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
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    return fluid


def _build_solid(
    config: TurekHronFsiConfig, runtime: TaichiRuntimeConfig
) -> tuple[NeoHookeanMpmState, dict[str, np.ndarray | float]]:
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
        "physical_tip_solver_z_m": float(z_tip),
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





def _marker_pressure_neumann_gradient_state(
    boundary: HibmMpmIbBoundaryConditions,
    markers: HibmMpmSurfaceMarkers,
) -> np.ndarray:
    """Host snapshot of the marker pressure-Neumann gradients (Picard state).

    The advance's fluid predictor rewrites these gradients before that trial's
    pressure rows consume them.  They are therefore paired diagnostic/scratch
    evidence rather than an independent physical fixed-point input for this
    Turek path. Mirrors the squid case's snapshot helper, which is bound to that
    case's coupling-state object and therefore not directly importable here.
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


class _TurekHronFsiRuntime:
    """Case adapter for the generic physical-step and marker-velocity loops."""

    def __init__(
        self,
        *,
        fluid: Any,
        solid: Any,
        markers: Any,
        boundary: Any,
        advance_trial: Callable[[FsiStepContext, int], Any],
        prepare_step: Callable[[FsiStepContext], None],
        restore_case_boundaries: Callable[[FsiStepContext], None],
        commit_case_step: Callable[
            [FsiStepContext, FsiTrialResult, FsiCouplingReport],
            dict[str, Any],
        ],
        finalize_case_run: Callable[[], dict[str, Any]],
        publish_case_step: Callable[
            [FsiStepContext, dict[str, Any]],
            None,
        ]
        | None = None,
        record_particle_position_write: Callable[[], None] | None = None,
        before_trial: Callable[[FsiStepContext, int], Any] | None = None,
        after_trial: Callable[
            [FsiStepContext, int, Any, np.ndarray, dict[str, Any], Any],
            None,
        ]
        | None = None,
        clear_trial: Callable[[], None] | None = None,
    ) -> None:
        self.fluid = fluid
        self.solid = solid
        self.markers = markers
        self.boundary = boundary
        self.advance_trial = advance_trial
        self.prepare_step = prepare_step
        self.restore_case_boundaries = restore_case_boundaries
        self.commit_case_step = commit_case_step
        self.finalize_case_run = finalize_case_run
        self.publish_case_step = publish_case_step
        self._record_particle_position_write = record_particle_position_write
        self.before_trial = before_trial
        self.after_trial = after_trial
        self.clear_trial = clear_trial
        self._marker_step_base: dict[str, Any] | None = None
        self._gradient_step_base: np.ndarray | None = None
        self._marker_rollback_base: dict[str, Any] | None = None
        self._gradient_rollback_base: np.ndarray | None = None
        self._step_transaction_ready = False
        self._trial_index = 0

    def begin_step(self, context: FsiStepContext) -> np.ndarray:
        self._clear_step_bases()
        self.fluid.save_state()
        self.solid.save_state()
        marker_rollback_base = capture_marker_interface_state(self.markers)
        gradient_rollback_base = _marker_pressure_neumann_gradient_state(
            self.boundary,
            self.markers,
        )
        self._marker_rollback_base = marker_rollback_base
        self._gradient_rollback_base = gradient_rollback_base
        self._step_transaction_ready = True
        self._trial_index = 0
        self.prepare_step(context)
        self._marker_step_base = capture_marker_interface_state(self.markers)
        self._gradient_step_base = _marker_pressure_neumann_gradient_state(
            self.boundary,
            self.markers,
        )
        return marker_velocity_state(self._marker_step_base)

    def _restore_step_base(
        self,
        physical_context: FsiStepContext,
        marker_velocity_guess_mps: np.ndarray | None,
    ) -> None:
        if (
            not self._step_transaction_ready
            or self._marker_step_base is None
            or self._gradient_step_base is None
        ):
            raise RuntimeError("Turek FSI step base has not been saved")
        self.fluid.restore_state()
        self.solid.restore_state()
        if self._record_particle_position_write is not None:
            self._record_particle_position_write()
        marker_state = (
            self._marker_step_base
            if marker_velocity_guess_mps is None
            else marker_trial_state(
                self._marker_step_base,
                marker_velocity_guess_mps,
            )
        )
        restore_marker_interface_state(self.markers, marker_state)
        _restore_marker_pressure_neumann_gradient_state(
            self.boundary,
            self.markers,
            self._gradient_step_base,
        )
        self.restore_case_boundaries(physical_context)

    def evaluate_trial(
        self,
        context: FsiStepContext,
        marker_velocity_guess_mps: np.ndarray,
    ) -> FsiTrialResult:
        self._restore_step_base(context, marker_velocity_guess_mps)
        trial_index = self._trial_index
        trial_token = (
            None
            if self.before_trial is None
            else self.before_trial(context, trial_index)
        )
        try:
            latest_report = self.advance_trial(context, trial_index)
        finally:
            if self.clear_trial is not None:
                self.clear_trial()
        if self._marker_step_base is None:
            raise RuntimeError("Turek FSI marker base disappeared during a trial")
        marker_candidate = _fsi_coupling_marker_candidate_from_step_base(
            step_base_state=self._marker_step_base,
            candidate_state=capture_marker_interface_state(self.markers),
            dt_s=context.dt_s,
        )
        restore_marker_interface_state(self.markers, marker_candidate)
        candidate_velocity = np.asarray(
            marker_candidate["v_gamma_mps"],
            dtype=np.float64,
        ).copy()
        if self.after_trial is not None:
            self.after_trial(
                context,
                trial_index,
                trial_token,
                np.asarray(marker_velocity_guess_mps, dtype=np.float64).copy(),
                marker_candidate,
                latest_report,
            )
        self._trial_index += 1
        return FsiTrialResult(
            marker_velocity_mps=candidate_velocity,
            payload={
                "latest_report": latest_report,
                "marker_state": marker_candidate,
                "physical_context": context,
            },
        )

    def commit_step(
        self,
        context: FsiStepContext,
        trial: FsiTrialResult,
        coupling: FsiCouplingReport,
    ) -> dict[str, Any]:
        row = dict(
            self.commit_case_step(
                context,
                trial,
                coupling,
            )
        )
        self._clear_step_bases()
        return row

    def publish_step(
        self,
        context: FsiStepContext,
        committed_row: dict[str, Any],
    ) -> None:
        if self.publish_case_step is not None:
            self.publish_case_step(context, committed_row)

    def rollback_step(self, context: FsiStepContext) -> None:
        del context
        if (
            not self._step_transaction_ready
            or self._marker_rollback_base is None
            or self._gradient_rollback_base is None
        ):
            self._clear_step_bases()
            return
        try:
            self.fluid.restore_state()
            self.solid.restore_state()
            if self._record_particle_position_write is not None:
                self._record_particle_position_write()
            restore_marker_interface_state(self.markers, self._marker_rollback_base)
            _restore_marker_pressure_neumann_gradient_state(
                self.boundary,
                self.markers,
                self._gradient_rollback_base,
            )
        finally:
            self._clear_step_bases()

    def _clear_step_bases(self) -> None:
        self._marker_step_base = None
        self._gradient_step_base = None
        self._marker_rollback_base = None
        self._gradient_rollback_base = None
        self._step_transaction_ready = False

    def finalize_run(self) -> dict[str, Any]:
        return dict(self.finalize_case_run())


def _turek_hron_coupling_report_fields(
    report: FsiCouplingReport,
    *,
    relative_tolerance: float,
    absolute_tolerance_mps: float,
    initial_relaxation: float,
) -> dict[str, Any]:
    """Map the generic report onto the established Turek result schema."""

    if not bool(report.converged):
        reason = "iteration_budget_exhausted"
    elif float(report.relative_residual) <= float(relative_tolerance):
        reason = "relative_tolerance"
    elif (
        float(absolute_tolerance_mps) > 0.0
        and float(report.absolute_residual_mps) <= float(absolute_tolerance_mps)
    ):
        reason = "absolute_tolerance"
    else:
        reason = "generic_solver_report"
    return {
        "fsi_coupling_iterations_used": int(report.iterations),
        "fsi_coupling_residual": float(report.relative_residual),
        "fsi_coupling_initial_relaxation": float(initial_relaxation),
        "fsi_coupling_residual_measured": True,
        "fsi_coupling_converged": bool(report.converged),
        "fsi_coupling_convergence_reason": reason,
        "fsi_coupling_absolute_residual_mps": float(
            report.absolute_residual_mps
        ),
        "fsi_coupling_max_marker_residual_mps": float(
            report.max_marker_residual_mps
        ),
        "fsi_coupling_residual_history": list(
            report.relative_residual_history
        ),
        "fsi_coupling_absolute_residual_history_mps": list(
            report.absolute_residual_history_mps
        ),
        "fsi_coupling_update_diagnostics": [
            {"iteration": index, "update_mode": str(mode)}
            for index, mode in enumerate(report.update_modes, start=1)
        ],
    }


def _resample_marker_group_arrays(
    x: np.ndarray,
    n: np.ndarray,
    a: np.ndarray,
    v: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Arc-length resample ONE ordered marker group's state (numpy only).

    x/n/a (positions/normals/areas) are resampled via
    resample_polyline_markers_by_arc_length on the group's CURRENT
    (deformed) positions. v (velocity) has no analog in that helper, so it
    is carried along by plain per-component numpy.interp over the SAME
    cumulative arc-length stations the resampler placed x/n/a at -- see the
    target_lengths computation below, which reproduces
    resample_polyline_markers_by_arc_length's own open-polyline station
    placement (np.linspace(0, total_length, count) evaluated against the
    identical cumulative-arc-length array) so every returned array lines up
    on the same m output markers.

    target_spacing is chosen so the resampler's own open-polyline
    station-count formula (station_count == ceil(total_length /
    target_spacing) + 1) evaluates to exactly `count`. The analytically
    exact spacing is total_length / (count - 1), but IEEE-754 rounding can
    push the ratio total_length/target_spacing a hair past the (count - 1)
    integer boundary and add a spurious extra station. Empirically
    (verified against resample_polyline_markers_by_arc_length for count in
    {3, 4, 5, 8, 48, 96, 140} on both straight and bent curves), SHRINKING
    target_spacing pushes that ratio ABOVE (count - 1) and returns
    count + 1 stations on every input tried; GROWING target_spacing by the
    same tiny margin instead pulls the ratio back to (or below) count - 1
    and reliably reproduces exactly `count` stations, so growing (not
    shrinking) is the direction used below. The trailing assert is the
    actionable backstop if some future input still disagrees.

    The output count must exactly equal the input count: markers are a
    FIXED-capacity Taichi field, and every downstream index
    (resolved_marker_counts, HibmMpmIbNodeSearch's marker_capacity, the
    lower/upper/tip slice boundaries in _reseed_turek_hron_markers) assumes
    the per-group and total marker counts never change step to step.
    """
    x = np.asarray(x, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    target_count = int(count)
    if target_count < 2:
        raise ValueError(
            "_resample_marker_group_arrays requires count >= 2, got "
            f"{target_count}"
        )
    if x.shape != (target_count, 3):
        raise ValueError(
            f"_resample_marker_group_arrays: x has shape {x.shape}, "
            f"expected ({target_count}, 3)"
        )

    segment_lengths = np.linalg.norm(np.diff(x, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError(
            "_resample_marker_group_arrays: marker group has zero arc length"
        )
    target_spacing = (total_length / float(target_count - 1)) * (1.0 + 1.0e-9)

    x_out, n_out, a_out = resample_polyline_markers_by_arc_length(
        x, n, a, target_spacing_m=target_spacing, closed=False
    )
    if x_out.shape[0] != target_count:
        raise ValueError(
            "_resample_marker_group_arrays: "
            "resample_polyline_markers_by_arc_length returned "
            f"{x_out.shape[0]} stations, expected count={target_count} "
            f"(total_length={total_length:.9e} m, "
            f"target_spacing={target_spacing:.9e} m); the station-count "
            "safety margin needs widening for this input."
        )

    target_lengths = np.linspace(0.0, total_length, target_count)
    v_out = np.empty((target_count, 3), dtype=np.float64)
    for axis in range(3):
        v_out[:, axis] = np.interp(target_lengths, cumulative, v[:, axis])

    return (
        x_out.astype(np.float64, copy=False),
        n_out.astype(np.float64, copy=False),
        a_out.astype(np.float64, copy=False),
        v_out,
    )


def _reseed_turek_hron_markers(
    markers: HibmMpmSurfaceMarkers, config: TurekHronFsiConfig
) -> None:
    """Tier-2 arc-length marker re-seeding (gated by marker_reseed_interval_steps).

    Marker order verified directly from build_marker_layout's construction
    loop (the two normal_sign passes followed by the tip pass):
      1. LOWER face: the first markers_per_side markers, normal
         (0, -1, 0), ordered by INCREASING solver z (beam tip -> beam root).
      2. UPPER face: the next markers_per_side markers, normal (0, 1, 0),
         same z-ordering as the lower face.
      3. TIP cap: the last markers_per_tip markers, normal (0, 0, -1),
         ordered by INCREASING solver y (lower face -> upper face).
    Each group is already a naturally-ordered OPEN polyline (confirmed by
    TurekHronMarkerGroupOrderTests), so resampling every group independently
    with closed=False keeps group membership, per-group count, and total
    marker count fixed while restoring near-uniform arc-length spacing on
    the CURRENT deformed positions.
    """
    side_count, tip_count = resolved_marker_counts(config)
    state = capture_marker_interface_state(markers)
    x = np.asarray(state["x_gamma_m"], dtype=np.float64)
    v = np.asarray(state["v_gamma_mps"], dtype=np.float64)
    n = np.asarray(state["n_gamma"], dtype=np.float64)
    a = np.asarray(state["A_gamma_m2"], dtype=np.float64)

    groups = (
        (slice(0, side_count), side_count),
        (slice(side_count, 2 * side_count), side_count),
        (slice(2 * side_count, 2 * side_count + tip_count), tip_count),
    )

    new_x = np.array(x, dtype=np.float64, copy=True)
    new_v = np.array(v, dtype=np.float64, copy=True)
    new_n = np.array(n, dtype=np.float64, copy=True)
    new_a = np.array(a, dtype=np.float64, copy=True)
    for group_slice, group_count in groups:
        gx, gn, ga, gv = _resample_marker_group_arrays(
            x[group_slice],
            n[group_slice],
            a[group_slice],
            v[group_slice],
            group_count,
        )
        new_x[group_slice] = gx
        new_n[group_slice] = gn
        new_a[group_slice] = ga
        new_v[group_slice] = gv

    new_state = dict(state)
    new_state["x_gamma_m"] = new_x.astype(
        np.asarray(state["x_gamma_m"]).dtype, copy=False
    )
    new_state["v_gamma_mps"] = new_v.astype(
        np.asarray(state["v_gamma_mps"]).dtype, copy=False
    )
    new_state["pressure_probe_origin_m"] = new_x.astype(
        np.asarray(state["pressure_probe_origin_m"]).dtype,
        copy=False,
    )
    new_state["n_gamma"] = new_n.astype(
        np.asarray(state["n_gamma"]).dtype, copy=False
    )
    new_state["A_gamma_m2"] = new_a.astype(
        np.asarray(state["A_gamma_m2"]).dtype, copy=False
    )
    restore_marker_interface_state(markers, new_state)


def _write_channel_external_velocity_faces(
    fluid: CartesianFluidSolver, config: TurekHronFsiConfig, t_s: float
) -> None:
    nx, ny, _ = (int(value) for value in config.grid_nodes)
    _, dy, _ = fluid_cell_spacing_m(config)
    peak_scale = (
        1.5
        * float(config.mean_inlet_velocity_mps)
        * inlet_ramp_factor(t_s, config)
    )
    fluid._invalidate_pressure_nullspace_component_graph()
    fluid._invalidate_velocity_dirichlet_component_ledger()
    fluid._invalidate_hibm_pressure_reachability()
    th_channel_external_velocity_faces_kernel(
        fluid.external_velocity_boundary_y_face_active_component_mask,
        fluid.external_velocity_boundary_y_face_value_mps,
        fluid.external_velocity_boundary_z_face_active_component_mask,
        fluid.external_velocity_boundary_z_face_value_mps,
        int(nx),
        int(ny),
        float(dy),
        float(config.channel_height_m),
        float(peak_scale),
    )
    fluid.prepare_and_seal_velocity_dirichlet_component_ledger()


def _outlet_flux_m3ps(fluid: CartesianFluidSolver, config: TurekHronFsiConfig) -> float:
    dx, dy, _ = fluid_cell_spacing_m(config)
    nx, ny, _ = (int(value) for value in config.grid_nodes)
    z_sum = float(outlet_zflux_sum_kernel(fluid.velocity, int(nx), int(ny)))
    return -z_sum * dx * dy


def _boundary_fluxes_m3ps(
    fluid: CartesianFluidSolver,
    config: TurekHronFsiConfig,
) -> tuple[float, float]:
    """Return actual positive inlet/outlet fluxes from the committed field."""

    dx, dy, _ = fluid_cell_spacing_m(config)
    nx, ny, nz = (int(value) for value in config.grid_nodes)
    z_sums = boundary_zflux_sums_kernel(
        fluid.velocity,
        int(nx),
        int(ny),
        int(nz),
    )
    inlet_z_sum = float(z_sums[0])
    outlet_z_sum = float(z_sums[1])
    face_area_m2 = dx * dy
    return -inlet_z_sum * face_area_m2, -outlet_z_sum * face_area_m2


def _point_a_displacement_from_tip_sections(
    rest_positions_m: np.ndarray,
    displacement_m: np.ndarray,
    *,
    physical_tip_solver_z_m: float,
) -> np.ndarray:
    """Extrapolate the two nearest particle-section means to physical Point A."""

    rest = np.asarray(rest_positions_m, dtype=np.float64)
    displacement = np.asarray(displacement_m, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1:] != (3,) or rest.shape[0] == 0:
        raise ValueError("rest_positions_m must have non-empty shape (particle, 3)")
    if displacement.shape != rest.shape:
        raise ValueError(
            "displacement_m must have the same (particle, 3) shape as "
            "rest_positions_m"
        )
    if not np.all(np.isfinite(rest)):
        raise ValueError("rest_positions_m must contain only finite values")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("displacement_m must contain only finite values")
    try:
        physical_tip_z = float(physical_tip_solver_z_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("physical_tip_solver_z_m must be a finite scalar") from exc
    if not math.isfinite(physical_tip_z):
        raise ValueError("physical_tip_solver_z_m must be a finite scalar")

    section_z = np.unique(rest[:, 2])
    if section_z.size < 2:
        raise ValueError("Point A extrapolation requires at least two z sections")
    tip_on_min_side = physical_tip_z < float(section_z[0])
    tip_on_max_side = physical_tip_z > float(section_z[-1])
    if not (tip_on_min_side or tip_on_max_side):
        raise ValueError(
            "physical_tip_solver_z_m must lie outside the particle z-section "
            f"range [{float(section_z[0]):.16g}, {float(section_z[-1]):.16g}]"
        )
    nearest_order = np.lexsort((section_z, np.abs(section_z - physical_tip_z)))
    nearest_z = np.sort(section_z[nearest_order[:2]])
    nearest_sections_are_inside = (
        bool(np.all(nearest_z > physical_tip_z))
        if tip_on_min_side
        else bool(np.all(nearest_z < physical_tip_z))
    )
    if not nearest_sections_are_inside:
        raise ValueError(
            "the two nearest Point A particle sections must lie on the same "
            "interior side of the physical tip"
        )
    section_spacing = float(nearest_z[1] - nearest_z[0])
    spacing_scale = max(
        1.0,
        abs(float(nearest_z[0])),
        abs(float(nearest_z[1])),
        abs(physical_tip_z),
    )
    if section_spacing <= 16.0 * np.finfo(np.float64).eps * spacing_scale:
        raise ValueError("nearest Point A z sections are numerically degenerate")

    first_mean = displacement[rest[:, 2] == nearest_z[0]].mean(axis=0)
    second_mean = displacement[rest[:, 2] == nearest_z[1]].mean(axis=0)
    point_a = first_mean + (physical_tip_z - float(nearest_z[0])) * (
        second_mean - first_mean
    ) / section_spacing
    if not np.all(np.isfinite(point_a)):
        raise ValueError("Point A extrapolation produced a non-finite displacement")
    return np.asarray(point_a, dtype=np.float64)


def _tip_displacement_row(
    solid: NeoHookeanMpmState, masks: dict[str, np.ndarray | float]
) -> dict[str, float]:
    count = int(solid.particle_count)
    current = solid.x.to_numpy()[:count]
    rest = np.asarray(masks["rest"], dtype=np.float64)
    displacement = current - rest
    norm = np.linalg.norm(displacement, axis=1)
    tip_mean = displacement[np.asarray(masks["tip"])].mean(axis=0)
    point_a = _point_a_displacement_from_tip_sections(
        rest,
        displacement,
        physical_tip_solver_z_m=float(masks["physical_tip_solver_z_m"]),
    )
    return {
        "tip_mean_displacement_solver_y_m": float(tip_mean[1]),
        "tip_mean_displacement_solver_z_m": float(tip_mean[2]),
        "tip_ux_turek_hron_m": float(-point_a[2]),
        "tip_uy_turek_hron_m": float(point_a[1]),
        "max_displacement_m": float(norm.max(initial=0.0)),
        "fixed_root_max_displacement_m": float(
            norm[np.asarray(masks["fixed"])].max(initial=0.0)
        ),
    }


def _fsi_coupling_marker_candidate_from_step_base(
    *,
    step_base_state: dict[str, Any],
    candidate_state: dict[str, Any],
    dt_s: float,
) -> dict[str, Any]:
    """Anchor an end-step marker candidate to one physical time increment."""

    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if "x_gamma_m" not in step_base_state or "v_gamma_mps" not in candidate_state:
        raise ValueError("marker states require x_gamma_m and v_gamma_mps")
    base_position = np.asarray(step_base_state["x_gamma_m"])
    candidate_velocity = np.asarray(candidate_state["v_gamma_mps"])
    if base_position.shape != candidate_velocity.shape:
        raise ValueError("base marker positions and candidate velocities must align")
    anchored = dict(candidate_state)
    for name in MARKER_INTERFACE_STATE_FIELDS:
        anchored[name] = np.asarray(candidate_state[name]).copy()
    anchored["x_gamma_m"] = (
        base_position.astype(np.float64, copy=False)
        + dt * candidate_velocity.astype(np.float64, copy=False)
    ).astype(base_position.dtype, copy=False)
    return anchored


def _optional_report_value(
    report: dict[str, Any] | None,
    key: str,
    converter: Any,
) -> Any:
    if report is None or key not in report or report[key] is None:
        return None
    return converter(report[key])


def _committed_step_observability_row(
    *,
    latest_report: Any,
    load: Any,
    fsi_coupling_max_marker_residual_mps: float | None,
    expected_marker_count: int,
) -> dict[str, Any]:
    """Certificate fields for the fluid/solid state that is actually committed."""

    main_projection = load.fluid_projection
    post_projection = latest_report.post_solid_fluid_projection
    post_no_slip = latest_report.post_solid_no_slip_residual
    scatter = load.mpm_force_scatter
    marker_forces = load.marker_forces
    return {
        "history_schema_version": 3,
        "stress_expected_marker_count": int(expected_marker_count),
        "projection_cg_converged_all": _optional_report_value(
            main_projection, "cg_converged_all", bool
        ),
        "projection_cg_breakdown_count": _optional_report_value(
            main_projection, "cg_breakdown_count", int
        ),
        "projection_cg_relative_residual_max": _optional_report_value(
            main_projection, "cg_relative_residual_max", float
        ),
        "post_solid_projection_applied": bool(
            latest_report.post_solid_kinematic_projection_applied
        ),
        "post_solid_projection_report_available": post_projection is not None,
        "post_solid_projection_pressure_solver": _optional_report_value(
            post_projection, "pressure_solver", str
        ),
        "post_solid_projection_l2": _optional_report_value(
            post_projection, "l2", float
        ),
        "post_solid_projection_max_abs": _optional_report_value(
            post_projection, "max_abs", float
        ),
        "post_solid_projection_cg_project_calls": _optional_report_value(
            post_projection, "cg_project_calls", int
        ),
        "post_solid_projection_cg_converged_all": _optional_report_value(
            post_projection, "cg_converged_all", bool
        ),
        "post_solid_projection_cg_breakdown_count": _optional_report_value(
            post_projection, "cg_breakdown_count", int
        ),
        "post_solid_projection_cg_relative_residual_max": _optional_report_value(
            post_projection, "cg_relative_residual_max", float
        ),
        "post_solid_projection_pressure_solve_failed": _optional_report_value(
            post_projection, "pressure_solve_failed", bool
        ),
        "post_solid_projection_pressure_solve_failure_action": _optional_report_value(
            post_projection, "pressure_solve_failure_action", str
        ),
        "post_solid_projection_physical_failure": _optional_report_value(
            post_projection, "pressure_projection_physical_failure", bool
        ),
        "post_solid_projection_physical_failure_reason": _optional_report_value(
            post_projection, "pressure_projection_physical_failure_reason", str
        ),
        "post_solid_no_slip_report_available": post_no_slip is not None,
        "post_solid_no_slip_valid_marker_count": (
            None if post_no_slip is None else int(post_no_slip.valid_marker_count)
        ),
        "post_solid_no_slip_invalid_marker_count": (
            None if post_no_slip is None else int(post_no_slip.invalid_marker_count)
        ),
        "post_solid_no_slip_max_mps": (
            None
            if post_no_slip is None
            else float(post_no_slip.max_no_slip_residual_mps)
        ),
        "post_solid_no_slip_l2_mps": (
            None
            if post_no_slip is None
            else float(post_no_slip.l2_no_slip_residual_mps)
        ),
        "marker_total_count": int(marker_forces.total_marker_count),
        "mpm_scatter_active_marker_count": int(scatter.active_marker_count),
        "mpm_scatter_invalid_marker_count": int(scatter.invalid_marker_count),
        "mpm_scatter_active_pair_count": int(scatter.active_pair_count),
        "mpm_scatter_action_reaction_residual_n": float(
            scatter.action_reaction_residual_n
        ),
        "fsi_coupling_max_marker_residual_mps": (
            None
            if fsi_coupling_max_marker_residual_mps is None
            else float(fsi_coupling_max_marker_residual_mps)
        ),
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
    # Generic marker-velocity coupling diagnostics.
    "fsi_coupling_iterations_used",
    "fsi_coupling_residual",
    "fsi_coupling_initial_relaxation",
    "fsi_coupling_residual_measured",
    "fsi_coupling_converged",
    "fsi_coupling_convergence_reason",
    "fsi_coupling_absolute_residual_mps",
    # Crash-surviving incremental diagnostics. These values already exist in
    # the per-step report; adding them to CSV introduces no extra device read.
    "history_schema_version",
    "stress_viscous_gradient_invalid_marker_count",
    "stress_one_sided_pressure_marker_count",
    "hibm_next_external_ib_node_count",
    "hibm_next_internal_node_count",
    "hibm_next_internal_obstacle_cell_count",
    "hibm_next_velocity_dirichlet_active_components",
    "hibm_next_pressure_neumann_active_rows",
    "hibm_next_solid_band_nonprojectable_cell_count",
    "hibm_stress_two_sided_extended_marker_count",
    "mechanism_probe_enabled",
    "mechanism_probe_triggered",
    "mechanism_probe_reason",
    "mechanism_probe_streak",
    # Schema v3: committed-state projection/no-slip/scatter certificate. These
    # are appended so every legacy field keeps its stable CSV position.
    "stress_expected_marker_count",
    "inlet_flux_actual_m3ps",
    "flux_imbalance_m3ps",
    "flux_imbalance_rel",
    "projection_cg_converged_all",
    "projection_cg_breakdown_count",
    "projection_cg_relative_residual_max",
    "post_solid_projection_applied",
    "post_solid_projection_report_available",
    "post_solid_projection_pressure_solver",
    "post_solid_projection_l2",
    "post_solid_projection_max_abs",
    "post_solid_projection_cg_project_calls",
    "post_solid_projection_cg_converged_all",
    "post_solid_projection_cg_breakdown_count",
    "post_solid_projection_cg_relative_residual_max",
    "post_solid_projection_pressure_solve_failed",
    "post_solid_projection_pressure_solve_failure_action",
    "post_solid_projection_physical_failure",
    "post_solid_projection_physical_failure_reason",
    "post_solid_no_slip_report_available",
    "post_solid_no_slip_valid_marker_count",
    "post_solid_no_slip_invalid_marker_count",
    "post_solid_no_slip_max_mps",
    "post_solid_no_slip_l2_mps",
    "marker_total_count",
    "mpm_scatter_active_marker_count",
    "mpm_scatter_invalid_marker_count",
    "mpm_scatter_active_pair_count",
    "mpm_scatter_action_reaction_residual_n",
    "fsi_coupling_max_marker_residual_mps",
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


def _write_fsi_coupling_failure_artifact(
    output_dir: Path | str, payload: dict[str, Any]
) -> Path:
    """Atomically persist the rejected strong-coupling trial diagnostics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "turek_hron_fsi_coupling_failure.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path,
            prefix=artifact_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
        temporary_path.replace(artifact_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return artifact_path


def _turek_hron_checkpoint_config_fingerprint(
    config: TurekHronFsiConfig,
) -> str:
    """Return a stable fingerprint of transition-relevant case controls."""

    payload = asdict(config)
    # These controls change only how far/how often a run writes output.  A
    # committed state captured at step N must be reusable for a longer replay.
    payload.pop("step_count", None)
    payload.pop("flow_snapshot_interval_steps", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _turek_hron_transition_checkpoint_metadata(
    *,
    config: TurekHronFsiConfig,
    preset: str,
    completed_step: int,
    particle_count: int,
    marker_count: int,
) -> dict[str, Any]:
    """Build fail-closed metadata for a committed transition checkpoint."""

    if int(completed_step) <= 0:
        raise ValueError("completed_step must be positive")
    if int(particle_count) <= 0:
        raise ValueError("particle_count must be positive")
    if int(marker_count) <= 0:
        raise ValueError("marker_count must be positive")
    return {
        "version": TUREK_HRON_TRANSITION_CHECKPOINT_VERSION,
        "case_id": TUREK_HRON_CASE_ID,
        "preset": str(preset),
        "completed_step": int(completed_step),
        "grid_nodes": [int(value) for value in config.grid_nodes],
        "particle_count": int(particle_count),
        "marker_count": int(marker_count),
        "config_fingerprint": _turek_hron_checkpoint_config_fingerprint(config),
    }


def _validate_turek_hron_transition_checkpoint_metadata(
    *,
    metadata: dict[str, Any],
    config: TurekHronFsiConfig,
    preset: str,
    particle_count: int,
    marker_count: int,
) -> int:
    """Validate that a checkpoint can exactly continue the requested case."""

    expected_version = TUREK_HRON_TRANSITION_CHECKPOINT_VERSION
    if int(metadata.get("version", -1)) != expected_version:
        raise ValueError("transition checkpoint version mismatch")
    if str(metadata.get("case_id", "")) != TUREK_HRON_CASE_ID:
        raise ValueError("transition checkpoint case_id mismatch")
    if str(metadata.get("preset", "")) != str(preset):
        raise ValueError("transition checkpoint preset mismatch")
    completed_step = int(metadata.get("completed_step", 0))
    if completed_step <= 0 or completed_step >= int(config.step_count):
        raise ValueError(
            "transition checkpoint completed_step must be positive and less "
            "than requested_steps"
        )
    expected_grid = [int(value) for value in config.grid_nodes]
    if list(metadata.get("grid_nodes", ())) != expected_grid:
        raise ValueError("transition checkpoint grid_nodes mismatch")
    if int(metadata.get("particle_count", -1)) != int(particle_count):
        raise ValueError("transition checkpoint particle_count mismatch")
    if int(metadata.get("marker_count", -1)) != int(marker_count):
        raise ValueError("transition checkpoint marker_count mismatch")
    expected_fingerprint = _turek_hron_checkpoint_config_fingerprint(config)
    if str(metadata.get("config_fingerprint", "")) != expected_fingerprint:
        raise ValueError("transition checkpoint configuration fingerprint mismatch")
    return completed_step


def _validated_checkpoint_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"checkpoint array {name!r} cannot use object dtype")
    if np.issubdtype(array.dtype, np.inexact) and not bool(
        np.all(np.isfinite(array))
    ):
        raise ValueError(f"checkpoint array {name!r} must be finite")
    return np.array(array, copy=True)


def _write_turek_hron_transition_checkpoint(
    path: Path | str,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    """Atomically write a compressed diagnostic transition checkpoint."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not arrays:
        raise ValueError("transition checkpoint arrays cannot be empty")
    validated = {
        str(name): _validated_checkpoint_array(str(name), value)
        for name, value in arrays.items()
    }
    if "__metadata__" in validated:
        raise ValueError("__metadata__ is a reserved checkpoint array name")
    metadata_json = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp.npz")
    try:
        np.savez_compressed(
            temporary_path,
            __metadata__=np.asarray(metadata_json),
            **validated,
        )
        temporary_path.replace(checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return checkpoint_path


def _load_turek_hron_transition_checkpoint(
    path: Path | str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load a transition checkpoint without permitting pickled payloads."""

    checkpoint_path = Path(path)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        if "__metadata__" not in archive.files:
            raise ValueError("transition checkpoint is missing metadata")
        raw_metadata = np.asarray(archive["__metadata__"])
        if raw_metadata.shape != ():
            raise ValueError("transition checkpoint metadata must be scalar")
        try:
            metadata = json.loads(str(raw_metadata.item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("transition checkpoint metadata is invalid") from error
        if not isinstance(metadata, dict):
            raise ValueError("transition checkpoint metadata must be an object")
        arrays = {
            name: _validated_checkpoint_array(name, archive[name])
            for name in archive.files
            if name != "__metadata__"
        }
    if not arrays:
        raise ValueError("transition checkpoint contains no state arrays")
    return metadata, arrays


def _numpy_field_checkpoint_payload(
    owner: Any,
    *,
    names: tuple[str, ...],
    prefix: str,
) -> dict[str, np.ndarray]:
    """Copy named Taichi-like fields into a plain NumPy checkpoint payload."""

    payload: dict[str, np.ndarray] = {}
    for name in names:
        field = getattr(owner, name, None)
        if field is None or not callable(getattr(field, "to_numpy", None)):
            raise ValueError(f"checkpoint field {prefix}_{name} is unavailable")
        key = f"{prefix}_{name}"
        payload[key] = _validated_checkpoint_array(key, field.to_numpy())
    return payload


def _restore_numpy_field_checkpoint_payload(
    owner: Any,
    payload: dict[str, np.ndarray],
    *,
    names: tuple[str, ...],
    prefix: str,
) -> None:
    """Restore named Taichi-like fields after strict shape/dtype validation."""

    restores = _validated_numpy_field_checkpoint_restores(
        owner,
        payload,
        names=names,
        prefix=prefix,
    )
    for field, restored in restores:
        field.from_numpy(restored)


def _validated_numpy_field_checkpoint_restores(
    owner: Any,
    payload: dict[str, np.ndarray],
    *,
    names: tuple[str, ...],
    prefix: str,
) -> list[tuple[Any, np.ndarray]]:
    """Validate a field group completely before returning write operations."""

    restores: list[tuple[Any, np.ndarray]] = []
    for name in names:
        key = f"{prefix}_{name}"
        if key not in payload:
            raise ValueError(f"checkpoint payload is missing {key}")
        field = getattr(owner, name, None)
        if field is None or not callable(getattr(field, "to_numpy", None)):
            raise ValueError(f"checkpoint field {key} is unavailable")
        if not callable(getattr(field, "from_numpy", None)):
            raise ValueError(f"checkpoint field {key} cannot be restored")
        expected = np.asarray(field.to_numpy())
        restored = _validated_checkpoint_array(key, payload[key])
        if restored.shape != expected.shape:
            raise ValueError(
                f"checkpoint field {key} shape mismatch: "
                f"{restored.shape} != {expected.shape}"
            )
        if restored.dtype != expected.dtype:
            raise ValueError(
                f"checkpoint field {key} dtype mismatch: "
                f"{restored.dtype} != {expected.dtype}"
            )
        restores.append((field, restored))
    return restores


_TUREK_HRON_CHECKPOINT_FLUID_FIELDS = (
    "velocity",
    "velocity_prev",
    "pressure",
    "obstacle",
    "hibm_base_obstacle",
    "hibm_dynamic_solid_volume_obstacle",
    "hibm_dynamic_solid_volume_external_carve",
    "hibm_fresh_fluid_cell",
)
_TUREK_HRON_CHECKPOINT_SOLID_FIELDS = (
    "x",
    "position_increment_residual_m",
    "v",
    "C",
    "F",
)
_TUREK_HRON_CHECKPOINT_MARKER_FIELDS = (
    "x_gamma_m",
    "v_gamma_mps",
    "n_gamma",
    "A_gamma_m2",
)
_TUREK_HRON_CHECKPOINT_BOUNDARY_FIELDS = (
    "marker_pressure_neumann_gradient_field",
)


def _turek_hron_transition_checkpoint_arrays(
    *,
    fluid: Any,
    solid: Any,
    markers: Any,
    boundary: Any,
) -> dict[str, np.ndarray]:
    """Capture every dynamic field needed for an exact cross-process replay."""

    if not bool(getattr(fluid, "_hibm_base_obstacle_initialized", False)):
        raise ValueError("fluid HIBM base obstacle must be initialized")
    payload = {
        **_numpy_field_checkpoint_payload(
            fluid,
            names=_TUREK_HRON_CHECKPOINT_FLUID_FIELDS,
            prefix="fluid",
        ),
        **_numpy_field_checkpoint_payload(
            solid,
            names=_TUREK_HRON_CHECKPOINT_SOLID_FIELDS,
            prefix="solid",
        ),
        **_numpy_field_checkpoint_payload(
            markers,
            names=_TUREK_HRON_CHECKPOINT_MARKER_FIELDS,
            prefix="marker",
        ),
        **_numpy_field_checkpoint_payload(
            boundary,
            names=_TUREK_HRON_CHECKPOINT_BOUNDARY_FIELDS,
            prefix="boundary",
        ),
    }
    payload["fluid_hibm_dynamic_solid_volume_enabled"] = np.asarray(
        int(bool(fluid.hibm_dynamic_solid_volume_enabled)), dtype=np.int8
    )
    payload["fluid_hibm_base_obstacle_initialized"] = np.asarray(
        1, dtype=np.int8
    )
    return payload


def _restore_turek_hron_transition_checkpoint_arrays(
    *,
    fluid: Any,
    solid: Any,
    markers: Any,
    boundary: Any,
    payload: dict[str, np.ndarray],
    particle_position_write_observer: Callable[[], None] | None = None,
) -> None:
    """Restore a full transition state only after global prevalidation."""

    enabled_key = "fluid_hibm_dynamic_solid_volume_enabled"
    initialized_key = "fluid_hibm_base_obstacle_initialized"
    for key in (enabled_key, initialized_key):
        if key not in payload:
            raise ValueError(f"checkpoint payload is missing {key}")
        value = _validated_checkpoint_array(key, payload[key])
        if value.shape != () or int(value.item()) not in (0, 1):
            raise ValueError(f"checkpoint scalar {key} must be zero or one")
    if int(np.asarray(payload[initialized_key]).item()) != 1:
        raise ValueError("checkpoint HIBM base obstacle must be initialized")

    # Build every write operation first.  A bad field in any owner therefore
    # cannot leave a partially restored CUDA state.
    restores = [
        *_validated_numpy_field_checkpoint_restores(
            fluid,
            payload,
            names=_TUREK_HRON_CHECKPOINT_FLUID_FIELDS,
            prefix="fluid",
        ),
        *_validated_numpy_field_checkpoint_restores(
            solid,
            payload,
            names=_TUREK_HRON_CHECKPOINT_SOLID_FIELDS,
            prefix="solid",
        ),
        *_validated_numpy_field_checkpoint_restores(
            markers,
            payload,
            names=_TUREK_HRON_CHECKPOINT_MARKER_FIELDS,
            prefix="marker",
        ),
        *_validated_numpy_field_checkpoint_restores(
            boundary,
            payload,
            names=_TUREK_HRON_CHECKPOINT_BOUNDARY_FIELDS,
            prefix="boundary",
        ),
    ]
    pressure = _validated_checkpoint_array(
        "fluid_pressure", payload["fluid_pressure"]
    )
    for scratch_name in ("pressure_tmp", "pressure_accum"):
        scratch = getattr(fluid, scratch_name, None)
        if scratch is None:
            continue
        expected = np.asarray(scratch.to_numpy())
        if pressure.shape != expected.shape or pressure.dtype != expected.dtype:
            raise ValueError(
                f"checkpoint pressure is incompatible with {scratch_name}"
            )
        restores.append((scratch, pressure.copy()))

    for field, restored in restores:
        field.from_numpy(restored)
        if field is solid.x and particle_position_write_observer is not None:
            particle_position_write_observer()
    fluid.hibm_dynamic_solid_volume_enabled = bool(
        int(np.asarray(payload[enabled_key]).item())
    )
    fluid._hibm_base_obstacle_initialized = True
    # Rebuild deformation-derived surface normals/areas while preserving the
    # exact restored particle fields as the solid's next save/restore base.
    if callable(getattr(solid, "save_state", None)) and callable(
        getattr(solid, "restore_state", None)
    ):
        solid.save_state()
        solid.restore_state()
        if particle_position_write_observer is not None:
            particle_position_write_observer()


def _committed_transition_checkpoint_requested(
    *, configured_step: int | None, completed_step: int
) -> bool:
    return configured_step is not None and int(configured_step) == int(
        completed_step
    )


def _transition_diagnostic_requested(
    *,
    configured_step: int | None,
    physical_step: int,
    coupling_iteration: int,
) -> bool:
    return (
        configured_step is not None
        and int(configured_step) == int(physical_step)
        and int(coupling_iteration) == 0
    )


def _checkpoint_array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _turek_hron_row_membership_change_report(
    *,
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Expose equal-count HIBM row/stencil identity switches."""

    names = sorted(set(before) | set(after))
    arrays: dict[str, dict[str, Any]] = {}
    any_identity_changed = False
    for name in names:
        if name not in before or name not in after:
            arrays[name] = {
                "present_before": name in before,
                "present_after": name in after,
                "fingerprint_changed": True,
            }
            any_identity_changed = True
            continue
        before_array = _validated_checkpoint_array(name, before[name])
        after_array = _validated_checkpoint_array(name, after[name])
        same_shape = before_array.shape == after_array.shape
        changed_count = (
            int(np.count_nonzero(before_array != after_array))
            if same_shape
            else int(before_array.size + after_array.size)
        )
        before_fingerprint = _checkpoint_array_fingerprint(before_array)
        after_fingerprint = _checkpoint_array_fingerprint(after_array)
        fingerprint_changed = before_fingerprint != after_fingerprint
        any_identity_changed = any_identity_changed or fingerprint_changed
        arrays[name] = {
            "shape_before": list(before_array.shape),
            "shape_after": list(after_array.shape),
            "before_nonzero_count": int(np.count_nonzero(before_array)),
            "after_nonzero_count": int(np.count_nonzero(after_array)),
            "changed_element_count": changed_count,
            "before_fingerprint": before_fingerprint,
            "after_fingerprint": after_fingerprint,
            "fingerprint_changed": fingerprint_changed,
        }
    return {
        "arrays": arrays,
        "any_identity_changed": bool(any_identity_changed),
    }


_TUREK_HRON_TRANSITION_DIAGNOSTIC_FLUID_FIELDS = (
    "velocity",
    "pressure",
    "obstacle",
    "hibm_base_obstacle",
    "hibm_dynamic_solid_volume_obstacle",
    "hibm_dynamic_solid_volume_external_carve",
    "hibm_fresh_fluid_cell",
    "velocity_dirichlet_boundary_active_component_mask",
    "velocity_dirichlet_boundary_value_mps",
    "velocity_dirichlet_boundary_pressure_mobility",
    "velocity_dirichlet_boundary_component_enforcement_weight",
    "velocity_dirichlet_boundary_component_region_id",
    "velocity_dirichlet_boundary_hard_fixed_component_mask",
    "velocity_dirichlet_boundary_external_exact_component_mask",
    "velocity_dirichlet_boundary_owned_component_mask",
    "external_velocity_boundary_y_face_active_component_mask",
    "external_velocity_boundary_y_face_value_mps",
    "external_velocity_boundary_z_face_active_component_mask",
    "external_velocity_boundary_z_face_value_mps",
    "pressure_interface_matrix_diagonal",
    "pressure_interface_matrix_rhs",
    "pressure_interface_coupling_active",
    "pressure_interface_coupling_neighbor",
    "pressure_interface_coupling_coefficient",
)
_TUREK_HRON_TRANSITION_DIAGNOSTIC_SOLID_FIELDS = (
    "x",
    "position_increment_residual_m",
    "v",
    "C",
    "F",
    "external_force_n",
    "surface_normal",
    "area_weight_m2",
)
_TUREK_HRON_TRANSITION_DIAGNOSTIC_MARKER_FIELDS = (
    "x_gamma_m",
    "v_gamma_mps",
    "n_gamma",
    "A_gamma_m2",
    "region_id",
    "t_gamma_pa",
    "t_pressure_gamma_pa",
    "t_viscous_gamma_pa",
    "F_gamma_n",
    "marker_pressure_anchor_cell",
    "_stress_pressure_valid",
    "_stress_viscous_mode",
    "_stress_invalid_reason_code",
    "_stress_fluid_side_pressure_pa",
    "_stress_reference_pressure_pa",
    "_stress_inside_probe_rung",
    "_stress_outside_probe_rung",
    "_stress_inside_probe_cell",
    "_stress_outside_probe_cell",
    "_stress_one_sided_anchor_selected",
    "_stress_one_sided_anchor_fallback_used",
)
_TUREK_HRON_TRANSITION_DIAGNOSTIC_SEARCH_FIELDS = (
    "node_kind_code",
    "nearest_marker",
    "node_signed_distance_m",
    "node_projection_marker_indices",
    "node_projection_marker_weights",
    "node_anchor_cell",
)
_TUREK_HRON_TRANSITION_DIAGNOSTIC_BOUNDARY_FIELDS = (
    "active_ib_node",
    "velocity_dirichlet_owned_row",
    "velocity_dirichlet_region_min",
    "velocity_dirichlet_region_max",
    "velocity_dirichlet_mps_field",
    "pressure_neumann_normal_field",
    "pressure_neumann_gradient_field",
    "marker_pressure_neumann_gradient_field",
    "marker_pressure_neumann_row_count",
    "marker_pressure_neumann_candidate_node_count",
)


def _turek_hron_transition_diagnostic_stage_arrays(
    *,
    stage: str,
    fluid: Any,
    solid: Any,
    markers: Any,
    search: Any,
    boundary: Any,
) -> dict[str, np.ndarray]:
    """Capture raw arrays at one precisely labelled transition stage."""

    normalized_stage = str(stage).strip().lower()
    if not normalized_stage or not normalized_stage.replace("_", "").isalnum():
        raise ValueError("diagnostic stage must be a non-empty identifier")
    return {
        **_numpy_field_checkpoint_payload(
            fluid,
            names=_TUREK_HRON_TRANSITION_DIAGNOSTIC_FLUID_FIELDS,
            prefix=f"{normalized_stage}_fluid",
        ),
        **_numpy_field_checkpoint_payload(
            solid,
            names=_TUREK_HRON_TRANSITION_DIAGNOSTIC_SOLID_FIELDS,
            prefix=f"{normalized_stage}_solid",
        ),
        **_numpy_field_checkpoint_payload(
            markers,
            names=_TUREK_HRON_TRANSITION_DIAGNOSTIC_MARKER_FIELDS,
            prefix=f"{normalized_stage}_marker",
        ),
        **_numpy_field_checkpoint_payload(
            search,
            names=_TUREK_HRON_TRANSITION_DIAGNOSTIC_SEARCH_FIELDS,
            prefix=f"{normalized_stage}_search",
        ),
        **_numpy_field_checkpoint_payload(
            boundary,
            names=_TUREK_HRON_TRANSITION_DIAGNOSTIC_BOUNDARY_FIELDS,
            prefix=f"{normalized_stage}_boundary",
        ),
    }


def _transition_diagnostic_array_inventory(
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "nonzero_count": int(np.count_nonzero(value)),
            "fingerprint": _checkpoint_array_fingerprint(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }


def _transition_diagnostic_component_arrays(
    arrays: dict[str, np.ndarray],
    *,
    stage: str,
    component: str,
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in names:
        key = f"{stage}_{component}_{name}"
        if key in arrays:
            result[name] = arrays[key]
    return result


def _turek_hron_transition_diagnostic_summary(
    arrays: dict[str, np.ndarray],
    *,
    dt_s: float,
) -> dict[str, Any]:
    """Summarize row identity, load, and MPM threshold changes."""

    row_names = (
        "node_kind_code",
        "nearest_marker",
        "node_projection_marker_indices",
        "node_projection_marker_weights",
        "node_anchor_cell",
    )
    committed_rows = _transition_diagnostic_component_arrays(
        arrays,
        stage="committed_step",
        component="search",
        names=row_names,
    )
    current_rows = _transition_diagnostic_component_arrays(
        arrays,
        stage="pre_solid_load",
        component="search",
        names=row_names,
    )
    next_rows = _transition_diagnostic_component_arrays(
        arrays,
        stage="post_surface_feedback",
        component="search",
        names=row_names,
    )
    row_reports: dict[str, Any] = {}
    if committed_rows and current_rows:
        row_reports["committed_to_current_load"] = (
            _turek_hron_row_membership_change_report(
                before=committed_rows,
                after=current_rows,
            )
        )
    if current_rows and next_rows:
        row_reports["current_load_to_post_surface_next"] = (
            _turek_hron_row_membership_change_report(
                before=current_rows,
                after=next_rows,
            )
        )

    solid_report: dict[str, Any] = {}
    entry_x = arrays.get("pre_solid_load_solid_x")
    exit_x = arrays.get("post_solid_solid_x")
    entry_residual = arrays.get(
        "pre_solid_load_solid_position_increment_residual_m"
    )
    exit_residual = arrays.get("post_solid_solid_position_increment_residual_m")
    if entry_x is not None and exit_x is not None:
        dx = np.asarray(exit_x, dtype=np.float64) - np.asarray(
            entry_x, dtype=np.float64
        )
        solid_report.update(
            {
                "changed_position_component_count": int(np.count_nonzero(dx)),
                "position_delta_l2_m": float(np.linalg.norm(dx)),
                "position_delta_max_abs_m": float(np.max(np.abs(dx))),
            }
        )
    if entry_x is not None and entry_residual is not None:
        threshold = np.maximum(
            1.0e-9,
            np.abs(np.asarray(entry_x, dtype=np.float64)) * 5.0e-7,
        )
        margin = threshold - np.abs(
            np.asarray(entry_residual, dtype=np.float64)
        )
        solid_report.update(
            {
                "entry_position_residual_threshold_margin_min_m": float(
                    np.min(margin)
                ),
                "entry_position_residual_near_threshold_component_count": int(
                    np.count_nonzero(margin <= max(1.0e-12, float(dt_s) * 1.0e-8))
                ),
            }
        )
    if entry_residual is not None and exit_residual is not None:
        entry_nonzero = np.asarray(entry_residual) != 0.0
        exit_zero = np.asarray(exit_residual) == 0.0
        solid_report["position_residual_nonzero_to_zero_component_count"] = int(
            np.count_nonzero(entry_nonzero & exit_zero)
        )

    load_report: dict[str, Any] = {}
    committed_force = arrays.get("committed_step_marker_F_gamma_n")
    current_force = arrays.get("pre_solid_load_marker_F_gamma_n")
    if committed_force is not None and current_force is not None:
        force_delta = np.asarray(current_force, dtype=np.float64) - np.asarray(
            committed_force, dtype=np.float64
        )
        load_report = {
            "marker_force_delta_l2_n": float(np.linalg.norm(force_delta)),
            "marker_force_delta_max_abs_n": float(np.max(np.abs(force_delta))),
            "marker_force_changed_component_count": int(
                np.count_nonzero(force_delta)
            ),
        }
    return {
        "array_inventory": _transition_diagnostic_array_inventory(arrays),
        "row_identity_reports": row_reports,
        "solid_position_update": solid_report,
        "committed_to_current_marker_load": load_report,
    }


def _persist_fsi_coupling_failure_evidence(
    *,
    incremental_history_path: Path | None,
    history: list[dict[str, Any]],
    last_flushed_index: int,
    incremental_header_written: bool,
    output_dir: Path | str | None,
    failure_payload: dict[str, Any],
    history_writer: Any = _flush_history_csv,
    artifact_writer: Any = _write_fsi_coupling_failure_artifact,
) -> tuple[bool, int, tuple[str, ...]]:
    """Best-effort persistence that never replaces the physical failure."""

    persistence_errors: list[str] = []
    updated_header_written = bool(incremental_header_written)
    updated_flushed_index = int(last_flushed_index)
    if (
        incremental_history_path is not None
        and updated_flushed_index < len(history)
    ):
        try:
            updated_header_written = bool(
                history_writer(
                    incremental_history_path,
                    history[updated_flushed_index:],
                    header_written=updated_header_written,
                )
            )
            updated_flushed_index = len(history)
        except Exception as error:
            persistence_errors.append(
                f"history_flush:{type(error).__name__}:{error}"
            )
    artifact_payload = {
        **failure_payload,
        "completed_history_rows_flushed": int(updated_flushed_index),
    }
    if output_dir is not None:
        try:
            artifact_writer(Path(output_dir), artifact_payload)
        except Exception as error:
            persistence_errors.append(
                f"failure_artifact:{type(error).__name__}:{error}"
            )
    return (
        updated_header_written,
        updated_flushed_index,
        tuple(persistence_errors),
    )


def run_turek_hron_fsi(
    config: TurekHronFsiConfig,
    *,
    preset: str = "fsi1",
    output_dir: Path | str | None = None,
    export_final_flow_snapshot: bool = True,
    history_flush_interval_steps: int = TUREK_HRON_HISTORY_FLUSH_INTERVAL_STEPS,
    fail_fast_probe: TurekHronMechanismProbe | None = None,
    transition_checkpoint_step: int | None = None,
    transition_diagnostic_step: int | None = None,
    resume_transition_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    """Run every Turek-Hron preset through the sole generic FSI runtime."""

    _validate_marker_grid_consistency(config)
    _validate_fsi_coupling_controls(config)
    config = with_beam_surface_force_support(config)
    for control_name, configured_step in (
        ("transition_checkpoint_step", transition_checkpoint_step),
        ("transition_diagnostic_step", transition_diagnostic_step),
    ):
        if configured_step is None:
            continue
        if int(configured_step) <= 0 or int(configured_step) > int(
            config.step_count
        ):
            raise ValueError(
                f"{control_name} must be between 1 and config.step_count"
            )
        if output_dir is None:
            raise ValueError(f"{control_name} requires output_dir")

    taichi_runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = _build_fluid(config, taichi_runtime)
    solid, masks = _build_solid(config, taichi_runtime)
    particle_position_generation = 0

    def record_particle_position_write() -> None:
        nonlocal particle_position_generation
        particle_position_generation = _advance_particle_position_generation(
            particle_position_generation
        )

    record_particle_position_write()
    markers = _build_markers(config, taichi_runtime)
    expected_marker_count = int(markers.marker_count)
    bounds_min, bounds_max = _full_bounds(config)
    search = HibmMpmIbNodeSearch(
        grid_nodes=config.grid_nodes,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        marker_capacity=markers.marker_count,
        runtime=taichi_runtime,
    )
    boundary = HibmMpmIbBoundaryConditions(
        grid_nodes=config.grid_nodes,
        marker_capacity=markers.marker_count,
        runtime=taichi_runtime,
    )
    mu_pa, lambda_pa = _lame_parameters(config)
    solid_substep_dt_s = float(config.dt_s) / float(config.solid_substeps)
    solid_damping = float(config.velocity_damping) ** (
        1.0 / float(config.solid_substeps)
    )
    plane_dx_m, plane_dy_m, plane_dz_m = fluid_cell_spacing_m(config)
    plane_spacing_m = max(plane_dy_m, plane_dz_m)
    search_radius_m = 1.5 * plane_spacing_m
    interior_probe_distance_m = plane_spacing_m
    search_radius_xyz_m = (
        (1.5 * plane_dx_m, 1.5 * plane_dy_m, 1.5 * plane_dz_m)
        if config.ib_anisotropic_envelope
        else None
    )
    interior_probe_distance_xyz_m = (
        (plane_dx_m, plane_dy_m, plane_dz_m)
        if config.ib_anisotropic_envelope
        else None
    )
    solid_substep_count = int(config.solid_substeps)
    active_transition_diagnostic_arrays: dict[str, np.ndarray] | None = None

    def solid_step() -> Any:
        if active_transition_diagnostic_arrays is not None:
            active_transition_diagnostic_arrays.update(
                _turek_hron_transition_diagnostic_stage_arrays(
                    stage="pre_solid_load",
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                    search=search,
                    boundary=boundary,
                )
            )
        solid.begin_out_of_bounds_guard_batch()
        try:
            for _solid_substep_index in range(solid_substep_count):
                solid.step(
                    dt_s=solid_substep_dt_s,
                    mu_pa=mu_pa,
                    lambda_pa=lambda_pa,
                    primary_region_id=PRIMARY_REGION_ID,
                    secondary_region_id=SECONDARY_UNUSED_REGION_ID,
                    velocity_damping=solid_damping,
                    constitutive_model=str(config.solid_constitutive_model),
                    read_report=False,
                )
                record_particle_position_write()
                if config.enforce_plane_strain_x:
                    solid.enforce_rest_x_plane()
                    record_particle_position_write()
            report = solid.end_out_of_bounds_guard_batch()
            if active_transition_diagnostic_arrays is not None:
                active_transition_diagnostic_arrays.update(
                    _turek_hron_transition_diagnostic_stage_arrays(
                        stage="post_solid",
                        fluid=fluid,
                        solid=solid,
                        markers=markers,
                        search=search,
                        boundary=boundary,
                    )
                )
            return report
        except BaseException:
            solid.abort_out_of_bounds_guard_batch()
            raise

    history: list[dict[str, Any]] = []
    completed_step_offset = 0
    committed_transition_reference_arrays: dict[str, np.ndarray] = {}
    if resume_transition_checkpoint is not None:
        resume_metadata, resume_arrays = _load_turek_hron_transition_checkpoint(
            Path(resume_transition_checkpoint)
        )
        completed_step_offset = _validate_turek_hron_transition_checkpoint_metadata(
            metadata=resume_metadata,
            config=config,
            preset=str(preset),
            particle_count=int(solid.particle_count),
            marker_count=int(markers.marker_count),
        )
        if completed_step_offset >= int(config.step_count):
            raise ValueError(
                "transition checkpoint must precede the requested final step"
            )
        checkpoint_history = resume_metadata.get("history")
        if not isinstance(checkpoint_history, list) or len(
            checkpoint_history
        ) != int(completed_step_offset):
            raise ValueError(
                "transition checkpoint history must contain every committed row"
            )
        for expected_step, checkpoint_row in enumerate(
            checkpoint_history, start=1
        ):
            if not isinstance(checkpoint_row, dict) or int(
                checkpoint_row.get("step", -1)
            ) != expected_step:
                raise ValueError(
                    "transition checkpoint history step sequence is invalid"
                )
        _restore_turek_hron_transition_checkpoint_arrays(
            fluid=fluid,
            solid=solid,
            markers=markers,
            boundary=boundary,
            payload=resume_arrays,
            particle_position_write_observer=record_particle_position_write,
        )
        history = [dict(row) for row in checkpoint_history]
        committed_transition_reference_arrays = {
            name: np.asarray(value).copy()
            for name, value in resume_arrays.items()
            if name.startswith("committed_step_")
        }

    incremental_history_path: Path | None = None
    incremental_header_written = False
    last_flushed_index = 0
    flush_interval = int(history_flush_interval_steps)
    mechanism_probe_streaks: dict[str, int] = {}
    if output_dir is not None and (
        flush_interval > 0 or fail_fast_probe is not None
    ):
        incremental_history_path = Path(output_dir) / "turek_hron_fsi_history.csv"
        incremental_history_path.parent.mkdir(parents=True, exist_ok=True)

    flow_snapshots_dir: Path | None = None
    flow_snapshot_interval = (
        int(config.flow_snapshot_interval_steps)
        if config.flow_snapshot_interval_steps is not None
        else None
    )
    if (
        output_dir is not None
        and flow_snapshot_interval is not None
        and flow_snapshot_interval > 0
    ):
        flow_snapshots_dir = Path(output_dir) / "flow_snapshots"
        flow_snapshots_dir.mkdir(parents=True, exist_ok=True)

    latest_report_box: dict[str, Any] = {"value": None}
    pending_transition_diagnostic: dict[str, Any] | None = None

    def prepare_step(context: FsiStepContext) -> None:
        if (
            config.marker_reseed_interval_steps is not None
            and context.step_index > 0
            and context.step_index
            % int(config.marker_reseed_interval_steps)
            == 0
        ):
            _reseed_turek_hron_markers(markers, config)
        _write_channel_external_velocity_faces(fluid, config, context.time_s)
        fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)

    def restore_case_boundaries(context: FsiStepContext) -> None:
        _write_channel_external_velocity_faces(fluid, config, context.time_s)
        fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)

    def advance_trial(
        _context: FsiStepContext,
        _trial_index: int,
    ) -> Any:
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
            mpm_particle_position_generation=(
                lambda: particle_position_generation
            ),
            solid_step=solid_step,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=search_radius_m,
            interior_probe_distance_m=interior_probe_distance_m,
            mpm_support_radius_m=float(config.mpm_support_radius_m),
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_UNUSED_REGION_ID,
            far_pressure_region_id=SECONDARY_UNUSED_REGION_ID,
            one_sided_pressure_primary_region_id=PRIMARY_REGION_ID,
            one_sided_primary_fluid_side_normal_sign=1.0,
            search_inactive_axis=(
                0 if config.enforce_plane_strain_x else None
            ),
            viscous_inactive_axis=(
                0 if config.enforce_plane_strain_x else None
            ),
            fluid_dt_s=float(config.dt_s),
            fluid_substeps=int(config.flow_predictor_substeps),
            projection_iterations=int(config.flow_projection_iterations),
            run_fluid_predictor=True,
            pressure_neumann_density_kgm3=float(config.fluid_density_kgm3),
            # The shared HIBM core divides both fluid_dt_s and this physical
            # step duration by fluid_substeps exactly once.
            pressure_neumann_dt_s=float(config.dt_s),
            pressure_outlet_zmin=True,
            velocity_inlet_zmax=True,
            two_sided_probe_max_multiplier=(
                thin_beam_pressure_probe_max_multiplier(config)
            ),
            reset_pressure=False,
            pressure_solver=str(config.flow_pressure_solver),
            cg_tolerance=float(config.flow_cg_tolerance),
            cg_preconditioner=str(config.flow_cg_preconditioner),
            accumulate_reprojection_pressure=True,
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
            interpolate_velocity_dirichlet_with_interior=(
                bool(config.interpolate_velocity_dirichlet_with_interior)
            ),
            classify_far_internal_nodes=bool(config.classify_far_internal_nodes),
            search_radius_xyz_m=search_radius_xyz_m,
            interior_probe_distance_xyz_m=interior_probe_distance_xyz_m,
        )

    def before_trial(
        context: FsiStepContext,
        trial_index: int,
    ) -> dict[str, np.ndarray] | None:
        nonlocal active_transition_diagnostic_arrays
        if not _transition_diagnostic_requested(
            configured_step=transition_diagnostic_step,
            physical_step=context.step,
            coupling_iteration=trial_index,
        ):
            return None
        arrays = {
            name: np.asarray(value).copy()
            for name, value in committed_transition_reference_arrays.items()
        }
        arrays.update(
            _turek_hron_transition_diagnostic_stage_arrays(
                stage="pre_trial",
                fluid=fluid,
                solid=solid,
                markers=markers,
                search=search,
                boundary=boundary,
            )
        )
        active_transition_diagnostic_arrays = arrays
        return arrays

    def clear_trial() -> None:
        nonlocal active_transition_diagnostic_arrays
        active_transition_diagnostic_arrays = None

    def after_trial(
        context: FsiStepContext,
        trial_index: int,
        trial_token: Any,
        marker_velocity_guess_mps: np.ndarray,
        marker_candidate: dict[str, Any],
        _latest_report: Any,
    ) -> None:
        nonlocal pending_transition_diagnostic
        if trial_token is None:
            return
        arrays = dict(trial_token)
        arrays.update(
            _turek_hron_transition_diagnostic_stage_arrays(
                stage="post_surface_feedback",
                fluid=fluid,
                solid=solid,
                markers=markers,
                search=search,
                boundary=boundary,
            )
        )
        candidate_velocity = np.asarray(
            marker_candidate["v_gamma_mps"],
            dtype=np.float64,
        )
        velocity_delta = candidate_velocity - np.asarray(
            marker_velocity_guess_mps,
            dtype=np.float64,
        )
        pending_transition_diagnostic = {
            "arrays": arrays,
            "context": context,
            "trial_index": int(trial_index),
            "max_marker_interface_residual_mps": float(
                np.max(np.linalg.norm(velocity_delta, axis=1))
            ),
        }

    def write_pending_transition_diagnostic(
        report: FsiCouplingReport,
    ) -> None:
        nonlocal pending_transition_diagnostic
        if pending_transition_diagnostic is None:
            return
        diagnostic = pending_transition_diagnostic
        arrays = dict(diagnostic["arrays"])
        context = diagnostic["context"]
        diagnostic_metadata = {
            "schema_version": 1,
            "case_id": TUREK_HRON_CASE_ID,
            "preset": str(preset),
            "diagnostic_kind": "first_generic_marker_velocity_trial_transition",
            "physical_step": int(context.step),
            "time_s": float(context.time_s),
            "coupling_iteration_zero_based": int(diagnostic["trial_index"]),
            "absolute_interface_residual_mps": float(
                report.absolute_residual_history_mps[0]
            ),
            "relative_interface_residual": float(
                report.relative_residual_history[0]
            ),
            "max_marker_interface_residual_mps": float(
                diagnostic["max_marker_interface_residual_mps"]
            ),
            "config_fingerprint": _turek_hron_checkpoint_config_fingerprint(
                config
            ),
            "stage_semantics": {
                "pre_trial": "restored step base before core row assembly",
                "pre_solid_load": (
                    "current trial rows, marker stress, and scattered solid "
                    "load at solid_step entry"
                ),
                "post_solid": (
                    "particle state after all solid substeps, before marker "
                    "feedback"
                ),
                "post_surface_feedback": (
                    "anchored marker candidate and next-step rows"
                ),
            },
            "summary": _turek_hron_transition_diagnostic_summary(
                arrays,
                dt_s=float(config.dt_s),
            ),
        }
        diagnostic_path = (
            Path(output_dir)
            / (
                "turek_hron_transition_diagnostic_step_"
                f"{context.step:06d}_trial_001.npz"
            )
        )
        _write_turek_hron_transition_checkpoint(
            diagnostic_path,
            metadata=diagnostic_metadata,
            arrays=arrays,
        )
        pending_transition_diagnostic = None

    def commit_case_step(
        context: FsiStepContext,
        trial: FsiTrialResult,
        coupling: FsiCouplingReport,
    ) -> dict[str, Any]:
        nonlocal mechanism_probe_streaks

        latest_report = trial.payload.get("latest_report")
        if latest_report is None:
            raise RuntimeError("generic Turek FSI trial omitted its sharp-step report")
        write_pending_transition_diagnostic(coupling)
        coupling_fields = _turek_hron_coupling_report_fields(
            coupling,
            relative_tolerance=float(config.fsi_coupling_tolerance),
            absolute_tolerance_mps=float(
                config.fsi_coupling_absolute_tolerance_mps
            ),
            initial_relaxation=float(config.fsi_coupling_initial_relaxation),
        )
        load = latest_report.fluid_to_mpm_loads
        force_n = tuple(
            float(value) for value in load.marker_forces.total_marker_force_n
        )
        projection = load.fluid_projection
        speed_max_mps = float(fluid._max_fluid_speed_kernel())
        cylinder_pressure_force_n = (
            fluid.compute_obstacle_surface_pressure_force_n()
        )
        cylinder_viscous_force_n = (
            fluid.compute_obstacle_surface_viscous_force_n()
        )
        cylinder_force_n = tuple(
            pressure + viscous
            for pressure, viscous in zip(
                cylinder_pressure_force_n,
                cylinder_viscous_force_n,
                strict=True,
            )
        )
        span_m = float(config.span_m)
        ramp = inlet_ramp_factor(context.time_s, config)
        inlet_flux_actual_m3ps, outlet_flux_m3ps = _boundary_fluxes_m3ps(
            fluid,
            config,
        )
        flux_imbalance_m3ps = outlet_flux_m3ps - inlet_flux_actual_m3ps
        flux_scale_m3ps = max(
            abs(inlet_flux_actual_m3ps),
            abs(outlet_flux_m3ps),
        )
        flux_imbalance_rel = (
            abs(flux_imbalance_m3ps) / flux_scale_m3ps
            if flux_scale_m3ps > 0.0
            else None
        )
        committed_observability = _committed_step_observability_row(
            latest_report=latest_report,
            load=load,
            fsi_coupling_max_marker_residual_mps=(
                coupling.max_marker_residual_mps
            ),
            expected_marker_count=expected_marker_count,
        )
        row: dict[str, Any] = {
            "step": int(context.step),
            "time_s": float(context.time_s),
            "ramp_factor": ramp,
            **_tip_displacement_row(solid, masks),
            "marker_force_x_n": force_n[0],
            "marker_force_y_n": force_n[1],
            "marker_force_z_n": force_n[2],
            "beam_drag_per_span_n_per_m": -force_n[2] / span_m,
            "beam_lift_per_span_n_per_m": force_n[1] / span_m,
            "cylinder_form_drag_per_span_n_per_m": (
                -cylinder_pressure_force_n[2] / span_m
            ),
            "cylinder_friction_drag_per_span_n_per_m": (
                -cylinder_viscous_force_n[2] / span_m
            ),
            "cylinder_drag_per_span_n_per_m": (
                -cylinder_force_n[2] / span_m
            ),
            "cylinder_lift_per_span_n_per_m": (
                cylinder_force_n[1] / span_m
            ),
            "total_drag_per_span_n_per_m": (
                -(force_n[2] + cylinder_force_n[2]) / span_m
            ),
            "total_lift_per_span_n_per_m": (
                (force_n[1] + cylinder_force_n[1]) / span_m
            ),
            "fluid_speed_max_mps": speed_max_mps,
            "outlet_flux_m3ps": outlet_flux_m3ps,
            "inlet_flux_target_m3ps": (
                ramp
                * float(config.mean_inlet_velocity_mps)
                * float(config.channel_height_m)
                * span_m
            ),
            "projection_l2": float(projection.get("l2", 0.0)),
            "projection_max_abs": float(projection.get("max_abs", 0.0)),
            "stress_valid_marker_count": int(
                load.fluid_stress.valid_marker_count
            ),
            "stress_invalid_marker_count": int(
                load.fluid_stress.invalid_marker_count
            ),
            "history_schema_version": int(
                committed_observability["history_schema_version"]
            ),
            "stress_viscous_gradient_invalid_marker_count": int(
                load.fluid_stress.viscous_gradient_invalid_marker_count
            ),
            "stress_one_sided_pressure_marker_count": int(
                load.fluid_stress.one_sided_pressure_marker_count
            ),
            **coupling_fields,
            "inlet_flux_actual_m3ps": inlet_flux_actual_m3ps,
            "flux_imbalance_m3ps": flux_imbalance_m3ps,
            "flux_imbalance_rel": flux_imbalance_rel,
            **committed_observability,
            "hibm_next_external_ib_node_count": int(
                latest_report.next_ib_node_search.external_ib_node_count
            ),
            "hibm_next_internal_node_count": int(
                latest_report.next_ib_node_search.internal_node_count
            ),
            "hibm_next_internal_obstacle_cell_count": int(
                latest_report.next_internal_obstacle_cell_count
            ),
            "hibm_next_velocity_dirichlet_active_components": int(
                latest_report.next_velocity_dirichlet[
                    "canonical_velocity_dirichlet_report"
                ]["final_active_component_count"]
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
        mechanism_probe_decision = _evaluate_turek_hron_mechanism_probe(
            fail_fast_probe,
            row,
            streaks=mechanism_probe_streaks,
        )
        mechanism_probe_streaks = mechanism_probe_decision.streaks
        row.update(
            {
                "mechanism_probe_enabled": fail_fast_probe is not None,
                "mechanism_probe_triggered": mechanism_probe_decision.triggered,
                "mechanism_probe_reason": mechanism_probe_decision.reason,
                "mechanism_probe_streak": max(
                    mechanism_probe_decision.streaks.values(),
                    default=0,
                ),
            }
        )
        if mechanism_probe_decision.triggered:
            _persist_fsi_coupling_failure_evidence(
                incremental_history_path=incremental_history_path,
                history=history,
                last_flushed_index=last_flushed_index,
                incremental_header_written=incremental_header_written,
                output_dir=output_dir,
                failure_payload={
                    "schema_version": 1,
                    "case": TUREK_HRON_CASE_ID,
                    "preset": str(preset),
                    "failed_step": int(context.step),
                    "failed_time_s": float(context.time_s),
                    "completed_steps": len(history),
                    "failure_kind": "mechanism_probe",
                    "reason": mechanism_probe_decision.reason,
                    "candidate_row": row,
                    "physical_state_restored": True,
                },
            )
            raise TurekHronMechanismProbeTriggered(
                "Turek-Hron mechanism probe triggered at candidate step "
                f"{context.step}: {mechanism_probe_decision.reason}"
            )

        history.append(row)
        latest_report_box["value"] = latest_report
        return row

    def publish_case_step(
        context: FsiStepContext,
        committed_row: dict[str, Any],
    ) -> None:
        nonlocal committed_transition_reference_arrays
        nonlocal incremental_header_written
        nonlocal last_flushed_index

        if not history or int(history[-1]["step"]) != int(context.step):
            raise RuntimeError("Turek committed history is out of sequence")
        history[-1] = dict(committed_row)
        prospective_history = history
        if _committed_transition_checkpoint_requested(
            configured_step=transition_checkpoint_step,
            completed_step=context.step,
        ):
            committed_transition_reference_arrays = (
                _turek_hron_transition_diagnostic_stage_arrays(
                    stage="committed_step",
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                    search=search,
                    boundary=boundary,
                )
            )
            checkpoint_arrays = {
                **_turek_hron_transition_checkpoint_arrays(
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                    boundary=boundary,
                ),
                **committed_transition_reference_arrays,
            }
            checkpoint_metadata = _turek_hron_transition_checkpoint_metadata(
                config=config,
                preset=str(preset),
                completed_step=int(context.step),
                particle_count=int(solid.particle_count),
                marker_count=int(markers.marker_count),
            )
            checkpoint_metadata.update(
                {
                    "checkpoint_kind": "committed_fsi_transition_state",
                    "history": json.loads(
                        json.dumps(
                            prospective_history,
                            sort_keys=True,
                            allow_nan=False,
                        )
                    ),
                    "committed_stage_semantics": {
                        "marker_stress": "accepted current-step load",
                        "search_and_boundary_rows": (
                            "post-surface-feedback rows for the next step"
                        ),
                    },
                }
            )
            checkpoint_path = (
                Path(output_dir)
                / (
                    "turek_hron_transition_checkpoint_step_"
                    f"{context.step:06d}.npz"
                )
            )
            _write_turek_hron_transition_checkpoint(
                checkpoint_path,
                metadata=checkpoint_metadata,
                arrays=checkpoint_arrays,
            )

        flush_required = _history_flush_required(
            completed_step=context.step,
            flush_interval=flush_interval,
            probe_triggered=False,
        )
        next_header_written = incremental_header_written
        next_flushed_index = last_flushed_index
        if incremental_history_path is not None and flush_required:
            next_header_written = _flush_history_csv(
                incremental_history_path,
                prospective_history[last_flushed_index:],
                header_written=incremental_header_written,
            )
            next_flushed_index = len(prospective_history)

        if (
            flow_snapshots_dir is not None
            and context.step % int(flow_snapshot_interval) == 0
        ):
            periodic_snapshot = build_turek_hron_final_fields_snapshot(
                fluid,
                solid,
                config,
            )
            periodic_snapshot["time_s"] = np.asarray(
                context.time_s,
                dtype=np.float64,
            )
            np.savez(
                flow_snapshots_dir / f"step_{context.step:06d}.npz",
                **periodic_snapshot,
            )

        incremental_header_written = next_header_written
        last_flushed_index = next_flushed_index

    def finalize_case_run() -> dict[str, Any]:
        nonlocal incremental_header_written
        nonlocal last_flushed_index
        if (
            incremental_history_path is not None
            and last_flushed_index < len(history)
        ):
            incremental_header_written = _flush_history_csv(
                incremental_history_path,
                history[last_flushed_index:],
                header_written=incremental_header_written,
            )
            last_flushed_index = len(history)
        return {
            "diagnostics": {
                "interface_unknown": "marker_velocity_mps",
                "coupling_accelerator": "iqn_ils",
            }
        }

    coupling_config = FsiCouplingConfig(
        max_iterations=int(config.fsi_coupling_iterations),
        relative_tolerance=float(config.fsi_coupling_tolerance),
        absolute_tolerance_mps=float(
            config.fsi_coupling_absolute_tolerance_mps
        ),
        initial_relaxation=float(config.fsi_coupling_initial_relaxation),
    )
    solver_config = FsiSolverConfig(
        step_count=int(config.step_count) - int(completed_step_offset),
        time_step_s=float(config.dt_s),
        coupling=coupling_config,
        completed_step_offset=int(completed_step_offset),
    )
    case_runtime = _TurekHronFsiRuntime(
        fluid=fluid,
        solid=solid,
        markers=markers,
        boundary=boundary,
        advance_trial=advance_trial,
        prepare_step=prepare_step,
        restore_case_boundaries=restore_case_boundaries,
        commit_case_step=commit_case_step,
        finalize_case_run=finalize_case_run,
        publish_case_step=publish_case_step,
        record_particle_position_write=record_particle_position_write,
        before_trial=before_trial,
        after_trial=after_trial,
        clear_trial=clear_trial,
    )

    try:
        generic_run = solve_fsi_runtime(case_runtime, solver_config)
    except FsiCouplingConvergenceError as error:
        transition_persistence_error: str | None = None
        try:
            write_pending_transition_diagnostic(error.report)
        except Exception as persistence_error:
            transition_persistence_error = (
                f"{type(persistence_error).__name__}:{persistence_error}"
            )
        physical_context = error.context
        physical_step = int(physical_context.step)
        report_fields = _turek_hron_coupling_report_fields(
            error.report,
            relative_tolerance=float(config.fsi_coupling_tolerance),
            absolute_tolerance_mps=float(
                config.fsi_coupling_absolute_tolerance_mps
            ),
            initial_relaxation=float(config.fsi_coupling_initial_relaxation),
        )
        failure_payload = {
            "schema_version": 1,
            "case": TUREK_HRON_CASE_ID,
            "preset": str(preset),
            "failed_step": int(physical_step),
            "failed_time_s": float(physical_context.time_s),
            "completed_steps": int(len(history)),
            "fsi_coupling_accelerator": "iqn_ils",
            "fsi_coupling_iteration_budget": int(
                config.fsi_coupling_iterations
            ),
            "fsi_coupling_maximum_trial_limit": int(
                config.fsi_coupling_iterations
            ),
            "fsi_coupling_near_tolerance_extra_trial_limit": 0,
            "fsi_coupling_iterations_used": int(error.report.iterations),
            "fsi_coupling_relative_tolerance": float(
                config.fsi_coupling_tolerance
            ),
            "fsi_coupling_absolute_tolerance_mps": float(
                config.fsi_coupling_absolute_tolerance_mps
            ),
            "fsi_coupling_residual_history": list(
                error.report.relative_residual_history
            ),
            "fsi_coupling_absolute_residual_history_mps": list(
                error.report.absolute_residual_history_mps
            ),
            "fsi_coupling_update_diagnostics": report_fields[
                "fsi_coupling_update_diagnostics"
            ],
            "fsi_coupling_certificate": {
                "fsi_coupling_residual_measured": True,
                "fsi_coupling_converged": False,
                "fsi_coupling_convergence_reason": (
                    "iteration_budget_exhausted"
                ),
                "fsi_coupling_absolute_residual_mps": float(
                    error.report.absolute_residual_mps
                ),
            },
            "fsi_coupling_max_marker_residual_mps": float(
                error.report.max_marker_residual_mps
            ),
            "physical_state_restored": True,
            "transition_diagnostic_persistence_error": (
                transition_persistence_error
            ),
        }
        (
            incremental_header_written,
            last_flushed_index,
            _persistence_errors,
        ) = _persist_fsi_coupling_failure_evidence(
            incremental_history_path=incremental_history_path,
            history=history,
            last_flushed_index=last_flushed_index,
            incremental_header_written=incremental_header_written,
            output_dir=output_dir,
            failure_payload=failure_payload,
        )
        raise

    latest_report = latest_report_box["value"]
    if latest_report is None:
        raise RuntimeError("turek-hron FSI run did not advance")
    summary: dict[str, Any] = {
        "case": TUREK_HRON_CASE_ID,
        "preset": str(preset),
        "config": asdict(config),
        "solver_path": (
            "simulation_core.drivers.generic_fsi_solver.solve_fsi_runtime"
        ),
        "interface_unknown": "marker_velocity_mps",
        "coupling_accelerator": "iqn_ils",
        "generic_runtime_completed_steps": len(generic_run.history),
        "wall_boundary_model": TUREK_HRON_WALL_BOUNDARY_MODEL,
        "reference_results": TUREK_HRON_REFERENCE_RESULTS.get(
            str(preset),
            {},
        ),
        "completed_steps": len(history),
        "history": history,
        "final": history[-1],
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "turek_hron_fsi_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
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
            snapshot = build_turek_hron_final_fields_snapshot(
                fluid,
                solid,
                config,
            )
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
    parser.add_argument(
        "--dt-s",
        type=float,
        default=None,
        help="Physical time-step size in seconds (for temporal-convergence campaigns).",
    )
    parser.add_argument("--grid-nodes", type=_parse_grid_nodes, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--projection-iterations", type=int, default=None)
    parser.add_argument(
        "--fsi-coupling-iterations",
        type=int,
        default=None,
        help=(
            "Maximum generic marker-velocity iterations per physical step "
            "(must be at least 2)."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-tolerance",
        type=float,
        default=None,
        help="Relative interface-velocity residual convergence tolerance.",
    )
    parser.add_argument(
        "--fsi-coupling-absolute-tolerance-mps",
        type=float,
        default=None,
        help=(
            "Optional absolute interface-velocity residual convergence floor "
            "in m/s (0 keeps the pure-relative gate)."
        ),
    )
    parser.add_argument(
        "--fail-fast-mechanism-probe",
        action="store_true",
        help=(
            "Enable the host-only FSI1 runaway guard used by staged repair "
            "runs; it never changes the solver state or equations."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-initial-relaxation",
        type=float,
        default=None,
        help="Initial relaxed-Picard scale passed to the generic FSI core.",
    )
    parser.add_argument(
        "--flow-predictor-substeps",
        type=int,
        default=None,
        help="Flow predictor substeps per coupling trial.",
    )
    parser.add_argument(
        "--flow-cg-preconditioner",
        choices=("auto", "jacobi", "fv_multigrid", "fv_multigrid_light"),
        default=None,
        help="FV-CG pressure preconditioner used by every coupling trial.",
    )
    parser.add_argument(
        "--flow-snapshot-interval-steps",
        type=int,
        default=None,
        help=(
            "Write flow_snapshots/step_XXXXXX.npz every N physical steps "
            "for GIF rendering."
        ),
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
    parser.add_argument(
        "--transition-checkpoint-step",
        type=int,
        default=None,
        help=(
            "Write a diagnostic restart checkpoint immediately after this "
            "physical step commits."
        ),
    )
    parser.add_argument(
        "--transition-diagnostic-step",
        type=int,
        default=None,
        help=(
            "Capture first-trial transition diagnostics at this physical step."
        ),
    )
    parser.add_argument(
        "--resume-transition-checkpoint",
        type=str,
        default=None,
        help="Resume from a case-local committed transition checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.steps is not None:
        overrides["step_count"] = int(args.steps)
    if args.dt_s is not None:
        dt_s = float(args.dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("--dt-s must be finite and positive")
        overrides["dt_s"] = dt_s
    if args.grid_nodes is not None:
        overrides["grid_nodes"] = args.grid_nodes
    if args.projection_iterations is not None:
        overrides["flow_projection_iterations"] = int(args.projection_iterations)
    if args.fsi_coupling_iterations is not None:
        overrides["fsi_coupling_iterations"] = int(args.fsi_coupling_iterations)
    if args.fsi_coupling_tolerance is not None:
        overrides["fsi_coupling_tolerance"] = float(args.fsi_coupling_tolerance)
    if args.fsi_coupling_absolute_tolerance_mps is not None:
        overrides["fsi_coupling_absolute_tolerance_mps"] = float(
            args.fsi_coupling_absolute_tolerance_mps
        )
    if args.fsi_coupling_initial_relaxation is not None:
        overrides["fsi_coupling_initial_relaxation"] = float(
            args.fsi_coupling_initial_relaxation
        )
    if args.flow_predictor_substeps is not None:
        overrides["flow_predictor_substeps"] = int(args.flow_predictor_substeps)
    if args.flow_cg_preconditioner is not None:
        overrides["flow_cg_preconditioner"] = str(args.flow_cg_preconditioner)
    if args.flow_snapshot_interval_steps is not None:
        overrides["flow_snapshot_interval_steps"] = int(
            args.flow_snapshot_interval_steps
        )
    config = PRESET_BUILDERS[str(args.preset)](**overrides)
    summary = run_turek_hron_fsi(
        config,
        preset=str(args.preset),
        output_dir=args.output_dir,
        export_final_flow_snapshot=not bool(args.no_final_flow_snapshot),
        history_flush_interval_steps=int(args.history_flush_interval_steps),
        fail_fast_probe=(
            TurekHronMechanismProbe()
            if bool(args.fail_fast_mechanism_probe)
            else None
        ),
        transition_checkpoint_step=args.transition_checkpoint_step,
        transition_diagnostic_step=args.transition_diagnostic_step,
        resume_transition_checkpoint=args.resume_transition_checkpoint,
    )
    final = summary["final"]
    print(
        f"turek-hron {args.preset}: steps={summary['completed_steps']} "
        f"tip_ux={final['tip_ux_turek_hron_m']:.6e} m "
        f"tip_uy={final['tip_uy_turek_hron_m']:.6e} m "
        f"total_drag/span={final['total_drag_per_span_n_per_m']:.6e} N/m "
        f"total_lift/span={final['total_lift_per_span_n_per_m']:.6e} N/m "
        f"beam_drag/span={final['beam_drag_per_span_n_per_m']:.6e} N/m "
        f"beam_lift/span={final['beam_lift_per_span_n_per_m']:.6e} N/m "
        f"umax={final['fluid_speed_max_mps']:.6e} m/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
