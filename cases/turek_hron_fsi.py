from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
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
# Tier-2 marker re-seeding (2026-07-09): host-only, numpy-only arc-length
# polyline resampler; see the module docstring for why it has no Taichi
# dependency despite living in a Taichi-heavy module.
from simulation_core.coupling.marker_seeding import (
    resample_polyline_markers_by_arc_length,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec
from simulation_core.solids.neo_hookean_mpm import (
    CONSTITUTIVE_MODELS,
    NeoHookeanMpmState,
)
from cases.turek_hron_kernels import (
    boundary_zflux_sums_kernel,
    outlet_zflux_sum_kernel,
    th_channel_boundary_rows_kernel,
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
    # Interface-velocity fixed-point accelerator (2026-07). "aitken" (default)
    # is the scalar Aitken Delta^2 relaxation preserved byte-for-byte. On a
    # refined grid the density-ratio-1 added-mass residual becomes multi-mode
    # and oscillatory: scalar Aitken thrashes (measured relaxation swinging
    # 0.05<->1.0, residual plateauing ~0.1, 0/30 steps converged, tip runaway).
    # "iqn_ils" replaces the scalar update with the interface quasi-Newton
    # inverse-least-squares multi-secant update (Degroote 2009), which converges
    # the multi-mode residual a single scalar cannot. Only the guess-update rule
    # changes; the relative interface-velocity residual and the convergence
    # tolerance test are identical for both. Requires fsi_coupling_iterations>1.
    fsi_coupling_accelerator: str = "aitken"
    # Absolute interface-velocity convergence floor (2026-07-10). The Picard
    # residual is RELATIVE (||dv|| / ||v_new||); during the inlet ramp the
    # interface velocities are ~0, so the relative residual is noise/noise and
    # never reaches the tolerance -- every early step burns the full iteration
    # budget on dynamically irrelevant mismatches (measured: 300 s/step at
    # 10 iterations while ||dv|| ~ 2e-5 m/s). With this floor, a step also
    # counts as converged when the per-marker RMS absolute mismatch (m/s) drops
    # below the floor: interface velocity errors far below the physical velocity
    # scale (mean inlet 0.2 m/s) cannot feed the added-mass loop.
    # 0.0 (default) preserves the pure-relative legacy test byte-for-byte.
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
            "fsi_coupling_iterations must be an int >= 1; got "
            f"{iterations!r} ({type(iterations).__name__}). 1 = legacy loose "
            "coupling; >1 enables the Picard strong-coupling loop."
        )
    if iterations < 1:
        raise ValueError(
            f"fsi_coupling_iterations must be an int >= 1; got {iterations}. "
            "1 = legacy loose coupling; >1 enables the Picard "
            "strong-coupling loop."
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

    relaxation = config.fsi_aitken_initial_relaxation
    try:
        relaxation_value = float(relaxation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fsi_aitken_initial_relaxation must be a finite float in "
            f"[0, 1.5]; got {relaxation!r}"
        ) from exc
    if not math.isfinite(relaxation_value) or not 0.0 <= relaxation_value <= 1.5:
        raise ValueError(
            "fsi_aitken_initial_relaxation must be a finite float in "
            f"[0, 1.5]; got {relaxation!r}. The Aitken update itself is "
            "bounded to this range, so an initial value outside it is "
            "always a config error."
        )

    accelerator = str(config.fsi_coupling_accelerator).strip().lower()
    if accelerator not in {"aitken", "iqn_ils"}:
        raise ValueError(
            "fsi_coupling_accelerator must be one of {'aitken', 'iqn_ils'}; "
            f"got {config.fsi_coupling_accelerator!r}. Refusing to fall back "
            "silently: a typo here would run a different coupling algorithm "
            "than the one requested."
        )
    if (
        accelerator == "iqn_ils"
        and relaxation_value < float(FSI_AITKEN_RELAXATION_LOWER)
    ):
        raise ValueError(
            "IQN-ILS initial relaxation must be at least "
            f"{FSI_AITKEN_RELAXATION_LOWER:.2f}; got {relaxation_value!r}. "
            "The configured reference and cold-recovery scale use this "
            "positive lower bound; evaluated Picard memory may subsequently "
            "learn a smaller registered line-search scale."
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


# Retained secant columns for the IQN-ILS interface-velocity update; matches
# simulation_core.coupling.fsi_coupling._IQN_ILS_HISTORY_LIMIT so both
# accelerators cap the least-squares history identically.
FSI_IQN_ILS_VELOCITY_HISTORY_LIMIT = 8
# Marker interface histories originate in f32 Taichi fields even though the
# host residual algebra is promoted to float64.  Use the conventional
# max(matrix.shape)*eps(f32) numerical-rank cutoff instead of float64's much
# smaller default.  The old default inverted a cond~2e5 mode into a finite
# 1e5-scale interface step.  The separate trust-region guard remains a second
# fail-closed barrier for poorly predicted output-space steps.
FSI_IQN_ILS_SECANT_RCOND = float(np.finfo(np.float32).eps)
# Trust radius normalized by the *unrelaxed residual* step.  Using the relaxed
# fallback as denominator makes a mathematically exact IQN correction look
# arbitrarily large when recovery backtracking reaches omega=0.125/0.05.
# Five residual lengths preserves the old default omega=0.5 radius
# (10 * 0.5||r||) while remaining independent of recovery damping.
FSI_IQN_ILS_MAX_STEP_OVER_RESIDUAL_STEP = 5.0
# Globalization guard: IQN is allowed to be non-monotone, but a trial whose
# residual is more than 2x the best point already evaluated
# in this physical step is not a useful new iterate.  Recover from that best
# point with the trusted relaxed Picard direction and restart the local secant
# model.  The 2026-07-12 FSI1 formal failure exercised exactly this path:
# 1.023e-4 -> 2.170e-4 m/s after an otherwise finite IQN proposal.
FSI_IQN_ILS_RESIDUAL_REGRESSION_LIMIT = 2.0
FSI_IQN_ILS_NEAR_TOLERANCE_FACTOR = 1.25
FSI_IQN_ILS_NEAR_TOLERANCE_EXTRA_TRIALS = 8
# One cold-recovery direction is allowed after a normal IQN/Picard direction
# exhausts inside the registered near-tolerance band.  This scale is separate
# from both normal Picard memory and the older regression-recovery damping.
FSI_IQN_ILS_NEAR_TOLERANCE_RECOVERY_RELAXATION = 0.05
# Stagnation guard for a collapsed IQN proposal.  The two conditions are
# deliberately conjunctive: a small Newton step that actually eliminates the
# residual remains valid, and an ordinary-size nonmonotone IQN step remains
# governed by the separate catastrophic-regression guard.  Step 185 repeatedly
# produced <5% of the trusted Picard step while retaining >97% of the residual.
FSI_IQN_ILS_STALLED_STEP_OVER_FALLBACK_LIMIT = 0.05
FSI_IQN_ILS_STALLED_OBSERVED_RESIDUAL_RATIO_LIMIT = 0.90
FSI_IQN_ILS_LINE_SEARCH_MIN_BETA = 0.125
# IQN Picard memory must retain the relaxation that the real nonlinear map
# actually accepted, even when evaluated backtracking goes below the scalar
# Aitken floor.  Its independent safety floor is the smallest effective
# Picard step already registered by the existing bounded search ladder.
FSI_IQN_ILS_PICARD_MEMORY_MIN_RELAXATION = (
    FSI_AITKEN_RELAXATION_LOWER * FSI_IQN_ILS_LINE_SEARCH_MIN_BETA
)


@dataclass(frozen=True)
class _IqnPendingLineSearch:
    """One evaluated velocity proposal and its immutable search direction."""

    diagnostic_index: int
    phase: str
    source_velocity_flat: np.ndarray
    source_candidate_velocity_flat: np.ndarray
    full_proposal_velocity_flat: np.ndarray
    source_gradient_guess: np.ndarray
    source_gradient_candidate: np.ndarray
    full_proposal_gradient: np.ndarray
    source_absolute_residual_mps: float
    beta: float
    full_proposed_over_fallback_step: float | None
    full_picard_relaxation: float | None
    configured_picard_reference_relaxation: float | None
    had_prior_rejection: bool = False


def _iqn_ils_secant_rcond(matrix: np.ndarray) -> float:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("IQN secant matrix must be two-dimensional")
    return min(
        1.0,
        float(max(array.shape, default=1)) * float(FSI_IQN_ILS_SECANT_RCOND),
    )


def _iqn_ils_independent_secant_indices(
    residual_matrix: np.ndarray,
) -> list[int]:
    """Keep a newest-first, numerically independent IQN secant subset.

    Marker histories are effectively f32 even after their host-side promotion
    to float64.  A low-dimensional interface response therefore often produces
    more retained columns than resolved modes.  Rejecting the *whole* history
    when that happens throws away good independent modes and reduces IQN to
    fixed relaxation.  Solving a truncated least-squares problem in the
    original redundant column basis is also unsafe: a near-null combination in
    ``V`` can have a large partner in ``W``.  Modified Gram-Schmidt removes the
    redundant ``V`` column and its matching ``W`` column together.

    Newest columns are preferred because they linearize the fixed-point map
    closest to the current trial.  Each tentative column is accepted only when
    the exact SVD rank test used by the subsequent least-squares solve still
    reports full column rank; the filter and solve therefore cannot disagree at
    their tolerance boundary.
    """

    matrix = np.asarray(residual_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return []
    if not bool(np.all(np.isfinite(matrix))):
        return []
    selected_newest_first: list[int] = []
    for index in range(matrix.shape[1] - 1, -1, -1):
        tentative_indices = sorted([*selected_newest_first, index])
        tentative = matrix[:, tentative_indices]
        try:
            singular_values = np.linalg.svd(tentative, compute_uv=False)
        except np.linalg.LinAlgError:
            return []
        if singular_values.size == 0 or not bool(
            np.all(np.isfinite(singular_values))
        ):
            return []
        largest_singular_value = float(singular_values[0])
        if largest_singular_value <= 0.0:
            continue
        cutoff = largest_singular_value * _iqn_ils_secant_rcond(tentative)
        numerical_rank = int(np.count_nonzero(singular_values > cutoff))
        if numerical_rank == tentative.shape[1]:
            selected_newest_first.append(index)
    return sorted(selected_newest_first)


def _iqn_ils_velocity_guess(
    *,
    velocity_guess_history: list[np.ndarray],
    velocity_candidate_history: list[np.ndarray],
    exclude_newest_secant: bool = False,
    fallback_relaxation: float,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Standard IQN-ILS next guess for the flat interface-velocity fixed point.

    The per-step fixed-point map sends the interface-velocity guess v_guess to
    the velocity v_candidate the solid returns when the fluid Dirichlet rows are
    built from v_guess; the residual is r = v_candidate - v_guess. At solid/fluid
    density ratio 1 on a refined grid this residual is multi-mode and
    oscillatory, so scalar Aitken thrashes (measured: relaxation swinging
    0.05<->1.0, residual plateauing ~0.1, tip runaway). The interface
    quasi-Newton inverse-least-squares update below models the current residual
    as a combination of past residual differences and applies the matching
    combination of past map-output differences (Degroote 2009). It is the same
    algebra as
    simulation_core.coupling.fsi_coupling._iqn_ils_interface_reaction_guess but
    kept local so this case iterates the marker interface velocity rather than
    the reaction-force formulation, and cross-checked against that helper in the
    unit test.

    With fewer than two history entries there is no secant pair yet, so fall
    back to a relaxed step guess + relax*(candidate - guess). The marker data
    originate in f32 device fields, so singular modes below that effective
    precision, non-finite proposals, and quasi-Newton steps grossly larger than
    the trusted relaxed step all fail closed to that same relaxed update. A
    well-resolved, bounded proposal uses IQN-ILS on the *relaxed* fixed-point
    map.  Fully modelled secant modes retain their quasi-Newton correction,
    while the unresolved complement keeps the same damping as the trusted
    Picard fallback.  This avoids an implicit omega=1 update in unresolved
    added-mass modes when the configured fallback relaxation is smaller.
    """
    guesses = velocity_guess_history
    candidates = velocity_candidate_history
    count = len(candidates)
    relaxation = float(fallback_relaxation)
    if not math.isfinite(relaxation) or relaxation < 0.0:
        raise ValueError("fallback_relaxation must be finite and non-negative")
    residual_last = candidates[-1] - guesses[-1]
    fallback = guesses[-1] + relaxation * residual_last
    diagnostics: dict[str, Any] = {
        "update_mode": "relaxed_fallback",
        "history_count": int(count),
        "raw_secant_column_count_before_exclusion": 0,
        "newest_secant_excluded_count": 0,
        "newest_secant_exclusion_reason": "",
        "raw_secant_column_count": 0,
        "retained_secant_column_count": 0,
        "numerical_rank": 0,
        "singular_value_ratio": None,
        "unmodeled_residual_ratio": None,
        "unmodeled_complement_relaxation": relaxation,
        "proposed_over_fallback_step": None,
        "proposed_over_residual_step": None,
        "fallback_reason": "",
    }

    def finish(
        value: np.ndarray,
        *,
        update_mode: str,
        fallback_reason: str = "",
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        result = np.asarray(value).copy()
        report = {
            **diagnostics,
            "update_mode": str(update_mode),
            "fallback_reason": str(fallback_reason),
        }
        if bool(return_diagnostics):
            return result, report
        return result

    if count < 2:
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="insufficient_history",
        )
    history_limit = min(FSI_IQN_ILS_VELOCITY_HISTORY_LIMIT, count - 1)
    first_index = count - history_limit
    residual_delta_columns: list[np.ndarray] = []
    relaxed_output_delta_columns: list[np.ndarray] = []
    for index in range(first_index, count):
        guess_delta = guesses[index] - guesses[index - 1]
        residual_delta = (candidates[index] - guesses[index]) - (
            candidates[index - 1] - guesses[index - 1]
        )
        if float(np.linalg.norm(residual_delta)) <= 1.0e-30:
            continue
        residual_delta_columns.append(residual_delta)
        # IQN on G_omega(x)=x+omega*(G(x)-x).  In resolved secant modes this
        # has the same fixed point/Newton correction as the undamped map; in
        # the unresolved complement it preserves omega instead of silently
        # applying a raw Picard update with omega=1.
        relaxed_output_delta_columns.append(
            guess_delta + relaxation * residual_delta
        )
    diagnostics["raw_secant_column_count_before_exclusion"] = int(
        len(residual_delta_columns)
    )
    if bool(exclude_newest_secant) and residual_delta_columns:
        residual_delta_columns = residual_delta_columns[:-1]
        relaxed_output_delta_columns = relaxed_output_delta_columns[:-1]
        diagnostics["newest_secant_excluded_count"] = 1
        diagnostics["newest_secant_exclusion_reason"] = (
            "backtracked_iqn_acceptance"
        )
    if not residual_delta_columns:
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="no_resolved_residual_secants",
        )
    residual_matrix_all = np.column_stack(residual_delta_columns)
    relaxed_output_matrix_all = np.column_stack(
        relaxed_output_delta_columns
    )
    diagnostics["raw_secant_column_count"] = int(residual_matrix_all.shape[1])
    independent_indices = _iqn_ils_independent_secant_indices(
        residual_matrix_all
    )
    if not independent_indices:
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="rank_filter_rejected_all_columns",
        )
    residual_matrix = residual_matrix_all[:, independent_indices]
    relaxed_output_matrix = relaxed_output_matrix_all[:, independent_indices]
    diagnostics["retained_secant_column_count"] = int(
        residual_matrix.shape[1]
    )
    try:
        singular_values = np.linalg.svd(residual_matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="svd_failure",
        )
    if singular_values.size == 0 or not bool(np.all(np.isfinite(singular_values))):
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="invalid_singular_values",
        )
    largest_singular_value = float(singular_values[0])
    if largest_singular_value <= 0.0:
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="zero_singular_spectrum",
        )
    diagnostics["singular_value_ratio"] = float(
        singular_values[-1] / largest_singular_value
    )
    numerical_rank = int(
        np.count_nonzero(
            singular_values
            > largest_singular_value * _iqn_ils_secant_rcond(residual_matrix)
        )
    )
    diagnostics["numerical_rank"] = int(numerical_rank)
    if numerical_rank < residual_matrix.shape[1]:
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="retained_matrix_rank_mismatch",
        )
    # V c ~= r_current (least squares), then apply the matching secants of the
    # relaxed map G_omega.  At omega=1 this is the standard IQN-ILS formula.
    try:
        coefficients, *_ = np.linalg.lstsq(
            residual_matrix,
            residual_last,
            rcond=_iqn_ils_secant_rcond(residual_matrix),
        )
    except np.linalg.LinAlgError:
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="least_squares_failure",
        )
    if not bool(np.all(np.isfinite(coefficients))):
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="invalid_least_squares_coefficients",
        )
    unmodeled_residual = residual_last - residual_matrix @ coefficients
    residual_last_norm = float(np.linalg.norm(residual_last))
    diagnostics["unmodeled_residual_ratio"] = (
        float(np.linalg.norm(unmodeled_residual)) / residual_last_norm
        if residual_last_norm > 0.0
        else 0.0
    )
    if residual_matrix.shape[1] == 1:
        # A single secant spans only a 1-D subspace; if the current residual has
        # a significant component outside it, the single-secant Newton step is
        # unreliable, so fall back to a relaxed step. Matches the repo IQN-ILS
        # safeguard (fsi_coupling._iqn_ils_interface_reaction_guess).
        if float(np.linalg.norm(unmodeled_residual)) > max(
            1.0e-12, 1.0e-8 * float(np.linalg.norm(residual_last))
        ):
            return finish(
                fallback,
                update_mode="relaxed_fallback",
                fallback_reason="single_secant_unmodeled_residual",
            )
    proposed = fallback - relaxed_output_matrix @ coefficients
    if not bool(np.all(np.isfinite(proposed))):
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="nonfinite_proposal",
        )
    proposed_step_norm = float(np.linalg.norm(proposed - guesses[-1]))
    fallback_step_norm = float(np.linalg.norm(fallback - guesses[-1]))
    diagnostics["proposed_over_fallback_step"] = (
        proposed_step_norm / fallback_step_norm
        if fallback_step_norm > 0.0
        else (0.0 if proposed_step_norm == 0.0 else None)
    )
    diagnostics["proposed_over_residual_step"] = (
        proposed_step_norm / residual_last_norm
        if residual_last_norm > 0.0
        else (0.0 if proposed_step_norm == 0.0 else None)
    )
    if proposed_step_norm > float(FSI_IQN_ILS_MAX_STEP_OVER_RESIDUAL_STEP) * max(
        residual_last_norm, 1.0e-30
    ):
        return finish(
            fallback,
            update_mode="relaxed_fallback",
            fallback_reason="trust_region",
        )
    return finish(proposed, update_mode="iqn_ils")


def _globalized_iqn_velocity_guess(
    *,
    velocity_guess_history: list[np.ndarray],
    velocity_candidate_history: list[np.ndarray],
    exclude_newest_secant: bool = False,
    current_absolute_residual_mps: float,
    best_absolute_residual_mps: float,
    best_velocity_guess: np.ndarray,
    best_velocity_candidate: np.ndarray,
    fallback_relaxation: float,
    recovery_relaxation: float | None = None,
    forced_recovery_reason: str | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Safeguard IQN with a best-residual relaxed recovery step.

    Multi-secant IQN is intentionally non-monotone, so ordinary residual
    increases are retained.  A catastrophic regression or an independently
    evaluated near-no-op/stagnation signal means the local inverse is no longer
    useful.  In either case use the relaxed Picard step from the best point
    already *evaluated* in this physical step and request a secant-history
    restart.  This changes only nonlinear iteration steering; the fluid/solid
    map, convergence residual, and physical tolerance are untouched.
    """

    current = float(current_absolute_residual_mps)
    best = float(best_absolute_residual_mps)
    if not math.isfinite(current) or current < 0.0:
        raise ValueError("current_absolute_residual_mps must be finite and non-negative")
    if not math.isfinite(best) or best < 0.0:
        raise ValueError("best_absolute_residual_mps must be finite and non-negative")
    best_guess = np.asarray(best_velocity_guess)
    best_candidate = np.asarray(best_velocity_candidate)
    if best_guess.shape != best_candidate.shape:
        raise ValueError("best IQN guess and candidate must have matching shapes")
    if not bool(np.all(np.isfinite(best_guess))) or not bool(
        np.all(np.isfinite(best_candidate))
    ):
        raise ValueError("best IQN guess and candidate must be finite")
    relaxation = float(fallback_relaxation)
    if not math.isfinite(relaxation) or relaxation < 0.0:
        raise ValueError("fallback_relaxation must be finite and non-negative")
    recovery = (
        relaxation
        if recovery_relaxation is None
        else float(recovery_relaxation)
    )
    if not math.isfinite(recovery) or recovery < 0.0:
        raise ValueError("recovery_relaxation must be finite and non-negative")
    recovery_reason = (
        str(forced_recovery_reason).strip()
        if forced_recovery_reason is not None
        else ""
    )
    if forced_recovery_reason is not None and not recovery_reason:
        raise ValueError("forced_recovery_reason must be non-empty when provided")
    regression_ratio: float | None = (
        current / best if best > 0.0 else (0.0 if current == 0.0 else None)
    )
    large_regression = (
        current > 0.0
        if best == 0.0
        else current > float(FSI_IQN_ILS_RESIDUAL_REGRESSION_LIMIT) * best
    )
    if large_regression or recovery_reason:
        if large_regression:
            recovery_reason = "residual_regression"
        proposal = best_guess + recovery * (best_candidate - best_guess)
        diagnostics: dict[str, Any] = {
            "update_mode": "best_residual_relaxed_recovery",
            "history_count": int(len(velocity_candidate_history)),
            "raw_secant_column_count": 0,
            "retained_secant_column_count": 0,
            "numerical_rank": 0,
            "singular_value_ratio": None,
            "unmodeled_residual_ratio": None,
            "proposed_over_fallback_step": 1.0,
            "fallback_reason": recovery_reason,
            "residual_regression_ratio": regression_ratio,
            "history_reset_required": True,
            "recovery_relaxation": recovery,
        }
        result = np.asarray(proposal).copy()
        if return_diagnostics:
            return result, diagnostics
        return result

    proposal, base_diagnostics = _iqn_ils_velocity_guess(
        velocity_guess_history=velocity_guess_history,
        velocity_candidate_history=velocity_candidate_history,
        exclude_newest_secant=bool(exclude_newest_secant),
        fallback_relaxation=relaxation,
        return_diagnostics=True,
    )
    diagnostics = {
        **base_diagnostics,
        "residual_regression_ratio": regression_ratio,
        "history_reset_required": False,
        "recovery_relaxation": None,
    }
    if return_diagnostics:
        return np.asarray(proposal).copy(), diagnostics
    return np.asarray(proposal).copy()


def _iqn_ils_newest_secant_rollback_report(
    *,
    phase: str,
    accepted: bool,
    accepted_beta: float,
    had_prior_rejection: bool,
    rollback_already_attempted: bool,
) -> dict[str, Any]:
    """Arm one newest-column rollback after a backtracked IQN acceptance."""

    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"iqn", "picard", "recovery"}:
        raise ValueError("IQN line-search phase is invalid")
    beta = float(accepted_beta)
    if not math.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError("accepted IQN beta must be finite and in (0, 1]")
    arm = bool(
        bool(accepted)
        and normalized_phase == "iqn"
        and beta < 1.0
        and bool(had_prior_rejection)
        and not bool(rollback_already_attempted)
    )
    return {
        "arm_exclusion_once": arm,
        "reason": "backtracked_iqn_acceptance" if arm else "not_armed",
        "phase": normalized_phase,
        "accepted": bool(accepted),
        "accepted_beta": beta,
        "had_prior_rejection": bool(had_prior_rejection),
        "rollback_already_attempted": bool(rollback_already_attempted),
    }


def _iqn_ils_registered_iqn_exhaustion_picard_acceptance_report(
    *,
    evaluated_line_search_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recognize one fully measured IQN-to-Picard transition.

    This is deliberately stricter than checking the final ``phase``.  The
    leave-newest-out experiment is eligible only after the complete registered
    IQN beta ladder was rejected by the real nonlinear map and the ensuing
    Picard direction needed a real backtrack before acceptance.
    """

    registered_betas = [1.0]
    minimum_beta = float(FSI_IQN_ILS_LINE_SEARCH_MIN_BETA)
    while registered_betas[-1] > minimum_beta:
        registered_betas.append(max(0.5 * registered_betas[-1], minimum_beta))

    trials = [dict(trial) for trial in evaluated_line_search_trials]
    iqn_trials: list[dict[str, Any]] = []
    picard_trials: list[dict[str, Any]] = []
    phase_changed = False
    valid_order = True
    for trial in trials:
        phase = str(trial.get("phase", "")).strip().lower()
        if phase == "iqn" and not phase_changed:
            iqn_trials.append(trial)
        elif phase == "picard":
            phase_changed = True
            picard_trials.append(trial)
        else:
            valid_order = False
            break

    iqn_betas: list[float] = []
    picard_betas: list[float] = []
    try:
        iqn_betas = [float(trial["current_beta"]) for trial in iqn_trials]
        picard_betas = [
            float(trial["current_beta"]) for trial in picard_trials
        ]
        final_picard_beta = (
            float(picard_betas[-1])
            if picard_betas
            else math.nan
        )
    except (KeyError, TypeError, ValueError):
        valid_order = False
        final_picard_beta = math.nan

    complete_registered_ladder = bool(
        valid_order
        and len(iqn_betas) == len(registered_betas)
        and all(
            math.isclose(
                observed,
                registered,
                rel_tol=0.0,
                abs_tol=16.0 * np.finfo(float).eps,
            )
            for observed, registered in zip(iqn_betas, registered_betas)
        )
        and all(not bool(trial.get("accepted", False)) for trial in iqn_trials)
        and bool(iqn_trials)
        and bool(iqn_trials[-1].get("backtracking_exhausted", False))
    )
    valid_picard_halving_ladder = bool(
        len(picard_betas) >= 2
        and all(math.isfinite(beta) for beta in picard_betas)
        and math.isclose(
            picard_betas[0],
            1.0,
            rel_tol=0.0,
            abs_tol=16.0 * np.finfo(float).eps,
        )
        and all(
            math.isclose(
                observed,
                0.5 * previous,
                rel_tol=0.0,
                abs_tol=16.0 * np.finfo(float).eps,
            )
            for previous, observed in zip(
                picard_betas[:-1], picard_betas[1:]
            )
        )
    )
    picard_rejection_before_acceptance = bool(
        valid_picard_halving_ladder
        and all(
            not bool(trial.get("accepted", False))
            for trial in picard_trials[:-1]
        )
        and bool(picard_trials[-1].get("accepted", False))
        and math.isfinite(final_picard_beta)
        and 0.0 < final_picard_beta < 1.0
    )
    latch = bool(
        complete_registered_ladder and picard_rejection_before_acceptance
    )
    return {
        "latch_candidate": latch,
        "reason": (
            "registered_iqn_exhaustion_then_backtracked_picard_acceptance"
            if latch
            else "not_latched"
        ),
        "registered_iqn_beta_count": len(registered_betas),
        "observed_iqn_beta_count": len(iqn_betas),
        "complete_registered_iqn_ladder_rejected": bool(
            complete_registered_ladder
        ),
        "picard_rejection_before_acceptance": bool(
            picard_rejection_before_acceptance
        ),
        "accepted_picard_beta": (
            float(final_picard_beta)
            if math.isfinite(final_picard_beta)
            else None
        ),
    }


def _iqn_ils_late_budget_leave_newest_out_report(
    *,
    prior_transition_eligible: bool,
    current_update_mode: str,
    full_history_initial_beta: float,
    retained_secant_column_count: int,
    completed_trials: int,
    base_iteration_budget: int,
    best_absolute_residual_mps: float,
    absolute_tolerance_mps: float,
    already_attempted: bool,
) -> dict[str, Any]:
    """Gate one pre-evaluation leave-newest-out IQN direction experiment."""

    mode = str(current_update_mode).strip().lower()
    initial_beta = float(full_history_initial_beta)
    retained = int(retained_secant_column_count)
    completed = int(completed_trials)
    budget = int(base_iteration_budget)
    best = float(best_absolute_residual_mps)
    tolerance = float(absolute_tolerance_mps)
    if not math.isfinite(initial_beta) or initial_beta <= 0.0 or initial_beta > 1.0:
        raise ValueError("full_history_initial_beta must be finite in (0, 1]")
    if retained < 0 or completed < 0 or budget <= 0:
        raise ValueError("IQN column and trial counts are invalid")
    if not math.isfinite(best) or best < 0.0:
        raise ValueError("best_absolute_residual_mps must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance_mps must be finite and non-negative")

    registered_beta_count = 1
    beta = 1.0
    minimum_beta = float(FSI_IQN_ILS_LINE_SEARCH_MIN_BETA)
    while beta > minimum_beta:
        beta = max(0.5 * beta, minimum_beta)
        registered_beta_count += 1
    remaining_base_slots = budget - completed
    late_budget_slot_limit = registered_beta_count + 1
    near_limit = float(FSI_IQN_ILS_NEAR_TOLERANCE_FACTOR) * tolerance
    use_leave_newest_out = bool(
        bool(prior_transition_eligible)
        and not bool(already_attempted)
        and mode == "iqn_ils"
        and initial_beta < minimum_beta
        and retained >= 3
        and tolerance > 0.0
        and best > near_limit
        and 1 <= remaining_base_slots <= late_budget_slot_limit
    )
    return {
        "action": (
            "recompute_without_newest_secant"
            if use_leave_newest_out
            else "keep_full_history_proposal"
        ),
        "reason": (
            "late_budget_subminimum_iqn_after_backtracked_picard"
            if use_leave_newest_out
            else "not_triggered"
        ),
        "prior_transition_eligible": bool(prior_transition_eligible),
        "current_update_mode": mode,
        "full_history_initial_beta": initial_beta,
        "registered_minimum_beta": minimum_beta,
        "retained_secant_column_count": retained,
        "completed_trials": completed,
        "base_iteration_budget": budget,
        "remaining_base_trial_slots": remaining_base_slots,
        "late_budget_slot_limit": late_budget_slot_limit,
        "maximum_trial_limit": budget
        + int(FSI_IQN_ILS_NEAR_TOLERANCE_EXTRA_TRIALS),
        "best_absolute_residual_mps": best,
        "absolute_tolerance_mps": tolerance,
        "near_tolerance_limit_mps": near_limit,
        "already_attempted": bool(already_attempted),
    }


def _iqn_ils_leave_newest_out_selection_report(
    *,
    selection_requested: bool,
    normal_velocity_flat: np.ndarray,
    normal_diagnostic: dict[str, Any],
    alternate_velocity_flat: np.ndarray,
    alternate_diagnostic: dict[str, Any],
    counterfactual_full_history_iqn: dict[str, Any],
) -> dict[str, Any]:
    """Select a usable LNO IQN proposal or fail closed to full history."""

    normal_velocity = np.asarray(normal_velocity_flat)
    alternate_velocity = np.asarray(alternate_velocity_flat)
    if normal_velocity.shape != alternate_velocity.shape:
        raise ValueError("normal and LNO velocity proposals must have equal shape")
    normal = dict(normal_diagnostic)
    alternate = dict(alternate_diagnostic)
    try:
        excluded_count = int(alternate.get("newest_secant_excluded_count", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "newest_secant_excluded_count must be zero or one"
        ) from error
    if excluded_count not in (0, 1):
        raise ValueError("newest_secant_excluded_count must be zero or one")
    requested = bool(selection_requested)
    applied = bool(
        requested
        and excluded_count == 1
        and str(alternate.get("update_mode", "")) == "iqn_ils"
    )
    superseded = bool(requested and not applied)
    if applied:
        selected_velocity = alternate_velocity.copy()
        selected_diagnostic = dict(alternate)
        selected_diagnostic["counterfactual_full_history_iqn"] = dict(
            counterfactual_full_history_iqn
        )
    else:
        selected_velocity = normal_velocity.copy()
        selected_diagnostic = dict(normal)
        if superseded:
            selected_diagnostic[
                "leave_newest_out_alternate_diagnostic"
            ] = dict(alternate)
    return {
        "applied": applied,
        "superseded": superseded,
        "selected_velocity_flat": selected_velocity,
        "selected_diagnostic": selected_diagnostic,
    }


def _iqn_ils_newest_secant_rollback_consumption_report(
    *,
    exclusion_requested: bool,
    iqn_update_diagnostic: dict[str, Any],
) -> dict[str, bool]:
    """Distinguish a column rollback from a globalizer that superseded it."""

    diagnostic = dict(iqn_update_diagnostic)
    raw_excluded_count = diagnostic.get("newest_secant_excluded_count", 0)
    try:
        excluded_count = int(raw_excluded_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "newest_secant_excluded_count must be zero or one"
        ) from error
    if excluded_count not in (0, 1):
        raise ValueError("newest_secant_excluded_count must be zero or one")
    requested = bool(exclusion_requested)
    applied = bool(requested and excluded_count == 1)
    superseded = bool(requested and not applied)
    return {
        "applied": applied,
        "superseded": superseded,
        "mark_rollback_attempted": applied,
        "clear_exclusion_request": requested,
    }


def _iqn_ils_stagnation_report(
    *,
    source_absolute_residual_mps: float,
    observed_absolute_residual_mps: float,
    proposed_over_fallback_step: float | None,
) -> dict[str, float | bool | None]:
    """Detect an evaluated near-no-op IQN step without rejecting true Newton steps."""

    source = float(source_absolute_residual_mps)
    observed = float(observed_absolute_residual_mps)
    if not math.isfinite(source) or source < 0.0:
        raise ValueError(
            "source_absolute_residual_mps must be finite and non-negative"
        )
    if not math.isfinite(observed) or observed < 0.0:
        raise ValueError(
            "observed_absolute_residual_mps must be finite and non-negative"
        )
    step_ratio = (
        float(proposed_over_fallback_step)
        if proposed_over_fallback_step is not None
        else None
    )
    if step_ratio is not None and (
        not math.isfinite(step_ratio) or step_ratio < 0.0
    ):
        raise ValueError(
            "proposed_over_fallback_step must be finite and non-negative"
        )
    observed_ratio: float | None = (
        observed / source
        if source > 0.0
        else (0.0 if observed == 0.0 else None)
    )
    available = step_ratio is not None and observed_ratio is not None
    rejected = bool(
        available
        and step_ratio <= float(
            FSI_IQN_ILS_STALLED_STEP_OVER_FALLBACK_LIMIT
        )
        and observed_ratio >= float(
            FSI_IQN_ILS_STALLED_OBSERVED_RESIDUAL_RATIO_LIMIT
        )
    )
    return {
        "available": bool(available),
        "source_absolute_residual_mps": source,
        "observed_absolute_residual_mps": observed,
        "proposed_over_fallback_step": step_ratio,
        "observed_residual_ratio": observed_ratio,
        "maximum_stalled_step_ratio": float(
            FSI_IQN_ILS_STALLED_STEP_OVER_FALLBACK_LIMIT
        ),
        "minimum_required_residual_reduction": float(
            1.0 - FSI_IQN_ILS_STALLED_OBSERVED_RESIDUAL_RATIO_LIMIT
        ),
        "rejected": rejected,
    }


def _iqn_ils_scale_aware_initial_beta_report(
    *,
    full_proposed_over_current_picard_step: float,
) -> dict[str, float | str]:
    """Choose the first IQN beta that can reach the trusted Picard scale.

    The ordinary evaluated line search deliberately starts from the full IQN
    proposal and registers the bounded halving ladder down to
    ``FSI_IQN_ILS_LINE_SEARCH_MIN_BETA``.  If even that last registered point
    is larger than the current relaxed-Picard step, evaluating the entire
    ordinary ladder only spends physical-step trials on known-oversized
    states.  In that exceptional case, continue the same halving ladder before
    the first evaluation until the IQN displacement is no larger than the
    current Picard displacement.
    """

    full_ratio = float(full_proposed_over_current_picard_step)
    if not math.isfinite(full_ratio) or full_ratio < 0.0:
        raise ValueError(
            "full_proposed_over_current_picard_step must be finite and "
            "non-negative"
        )
    registered_minimum = float(FSI_IQN_ILS_LINE_SEARCH_MIN_BETA)
    if (
        not math.isfinite(registered_minimum)
        or registered_minimum <= 0.0
        or registered_minimum > 1.0
    ):
        raise ValueError("registered minimum IQN beta must be finite in (0, 1]")

    initial_beta = 1.0
    action = "keep_full_proposal"
    if registered_minimum * full_ratio > 1.0:
        action = "scale_before_evaluation"
        initial_beta = registered_minimum
        while initial_beta * full_ratio > 1.0:
            next_beta = 0.5 * initial_beta
            if not math.isfinite(next_beta) or next_beta <= 0.0:
                raise OverflowError(
                    "IQN scale-aware initial beta underflowed before reaching "
                    "the current Picard scale"
                )
            initial_beta = next_beta

    return {
        "action": action,
        "full_proposed_over_current_picard_step": full_ratio,
        "registered_minimum_beta": registered_minimum,
        "initial_beta": float(initial_beta),
        "effective_proposed_over_current_picard_step": float(
            initial_beta * full_ratio
        ),
    }


def _iqn_ils_stagnation_rejection_policy_report(
    *,
    phase: str,
    best_absolute_residual_mps: float,
    absolute_tolerance_mps: float,
) -> dict[str, Any]:
    """Allow measured near-band decreases to become accepted IQN secants.

    Outside the registered refinement band, tiny IQN steps must still make a
    material reduction before they are admitted to the local inverse model.
    Once a real evaluated best has entered the bounded near band, however, the
    extra trials exist specifically to refine that state: a strict real-map
    decrease is useful information even when its displacement is small.
    """

    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"iqn", "picard", "recovery"}:
        raise ValueError("IQN line-search phase is invalid")
    best = float(best_absolute_residual_mps)
    tolerance = float(absolute_tolerance_mps)
    if not math.isfinite(best) or best < 0.0:
        raise ValueError("best_absolute_residual_mps must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance_mps must be finite and non-negative")
    near_tolerance_refinement = bool(
        normalized_phase == "iqn"
        and tolerance > 0.0
        and best > tolerance
        and best
        <= float(FSI_IQN_ILS_NEAR_TOLERANCE_FACTOR) * tolerance
    )
    enforce = bool(
        normalized_phase == "iqn" and not near_tolerance_refinement
    )
    return {
        "phase": normalized_phase,
        "best_absolute_residual_mps": best,
        "absolute_tolerance_mps": tolerance,
        "near_tolerance_limit_mps": float(
            FSI_IQN_ILS_NEAR_TOLERANCE_FACTOR
        )
        * tolerance,
        "near_tolerance_refinement": near_tolerance_refinement,
        "enforce_stagnation_rejection": enforce,
        "reason": (
            "near_band_strict_decrease_refinement"
            if near_tolerance_refinement
            else "ordinary_phase_policy"
        ),
    }


def _iqn_ils_evaluated_line_search_report(
    *,
    source_absolute_residual_mps: float,
    observed_absolute_residual_mps: float,
    current_beta: float,
    full_proposed_over_fallback_step: float | None,
    enforce_stagnation_rejection: bool = True,
) -> dict[str, float | bool | str | None]:
    """Accept a velocity proposal only after evaluating the real FSI map.

    Pre-evaluation trust bounds cannot detect a locally inaccurate multi-secant
    model.  This report compares the next trial with the exact source trial and
    either accepts the actual decrease or requests a bounded half-step along the
    *same* proposal direction.  The optional tiny-step stagnation rule is for a
    collapsed IQN model direction; trusted Picard and explicit recovery steps
    still require real strict decrease, but must not discard that measured
    improvement merely because their damping is small.
    """

    if not isinstance(enforce_stagnation_rejection, (bool, np.bool_)):
        raise TypeError("enforce_stagnation_rejection must be a boolean")
    beta = float(current_beta)
    if not math.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError("current_beta must be finite in (0, 1]")
    full_ratio = (
        None
        if full_proposed_over_fallback_step is None
        else float(full_proposed_over_fallback_step)
    )
    if full_ratio is not None and (
        not math.isfinite(full_ratio) or full_ratio < 0.0
    ):
        raise ValueError(
            "full_proposed_over_fallback_step must be finite and non-negative"
        )
    effective_ratio = None if full_ratio is None else beta * full_ratio
    stagnation = _iqn_ils_stagnation_report(
        source_absolute_residual_mps=source_absolute_residual_mps,
        observed_absolute_residual_mps=observed_absolute_residual_mps,
        proposed_over_fallback_step=effective_ratio,
    )
    source = float(source_absolute_residual_mps)
    observed = float(observed_absolute_residual_mps)
    strict_decrease = observed < source
    stalled_detected = bool(stagnation["rejected"])
    stalled = bool(enforce_stagnation_rejection and stalled_detected)
    accepted = bool(strict_decrease and not stalled)
    rejection_reason = ""
    if not strict_decrease:
        rejection_reason = "residual_regression"
    elif stalled:
        rejection_reason = "insufficient_residual_reduction"
    next_beta: float | None = None
    if not accepted and not stalled:
        candidate_beta = 0.5 * beta
        if candidate_beta >= float(FSI_IQN_ILS_LINE_SEARCH_MIN_BETA):
            next_beta = float(candidate_beta)
    return {
        "source_absolute_residual_mps": source,
        "observed_absolute_residual_mps": observed,
        "observed_residual_ratio": stagnation["observed_residual_ratio"],
        "current_beta": beta,
        "effective_proposed_over_fallback_step": effective_ratio,
        "strict_residual_decrease": bool(strict_decrease),
        "stalled_model_detected": stalled_detected,
        "stalled_model_rejected": stalled,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        # A stalled tiny IQN step can still be a genuinely better evaluated
        # physical state.  Reject the local-model direction, not that measured
        # improvement: the Picard fallback must restart from this point so its
        # next residual is compared with the current algorithmic best rather
        # than the older line-search origin.
        "stalled_improvement_available": bool(strict_decrease and stalled),
        "next_beta": next_beta,
        "backtracking_exhausted": bool(not accepted and next_beta is None),
    }


def _iqn_ils_observed_trial_improves_global_best(
    *,
    observed_absolute_residual_mps: float,
    best_absolute_residual_mps: float,
) -> bool:
    """Return whether one evaluated trial is strictly better than global best."""

    observed = float(observed_absolute_residual_mps)
    best = float(best_absolute_residual_mps)
    if not math.isfinite(observed) or observed < 0.0:
        raise ValueError(
            "observed_absolute_residual_mps must be finite and non-negative"
        )
    if not math.isfinite(best) or best < 0.0:
        raise ValueError(
            "best_absolute_residual_mps must be finite and non-negative"
        )
    return bool(observed < best)


def _iqn_ils_normal_reference_step_report(
    *,
    proposed_over_current_picard_step: float | None,
    current_picard_relaxation: float,
    beta: float,
    configured_picard_reference_relaxation: float,
) -> dict[str, float | None]:
    """Rescale a normal Picard/IQN step to one immutable physical-step scale."""

    current_omega = float(current_picard_relaxation)
    accepted_beta = float(beta)
    configured_omega = float(configured_picard_reference_relaxation)
    if not math.isfinite(current_omega) or current_omega <= 0.0:
        raise ValueError("current_picard_relaxation must be finite and positive")
    if (
        not math.isfinite(accepted_beta)
        or accepted_beta <= 0.0
        or accepted_beta > 1.0
    ):
        raise ValueError("beta must be finite in (0, 1]")
    if not math.isfinite(configured_omega) or configured_omega <= 0.0:
        raise ValueError(
            "configured_picard_reference_relaxation must be finite and positive"
        )
    current_ratio = (
        None
        if proposed_over_current_picard_step is None
        else float(proposed_over_current_picard_step)
    )
    if current_ratio is not None and (
        not math.isfinite(current_ratio) or current_ratio < 0.0
    ):
        raise ValueError(
            "proposed_over_current_picard_step must be finite and non-negative"
        )
    full_configured_ratio = (
        None
        if current_ratio is None
        else current_ratio * current_omega / configured_omega
    )
    effective_configured_ratio = (
        None
        if full_configured_ratio is None
        else accepted_beta * full_configured_ratio
    )
    return {
        "configured_picard_reference_relaxation": configured_omega,
        "current_picard_relaxation": current_omega,
        "full_proposed_over_current_picard_step": current_ratio,
        "full_proposed_over_configured_picard_step": full_configured_ratio,
        "effective_proposed_over_configured_picard_step": (
            effective_configured_ratio
        ),
    }


def _iqn_ils_picard_reference_step_report(
    *,
    full_picard_relaxation: float,
    beta: float,
    configured_picard_reference_relaxation: float,
) -> dict[str, float | None]:
    """Report one Picard trial against the physical step's fixed reference.

    The learned Picard memory may shrink within one physical step.  Measuring
    a later line-search trial against that mutable value would renormalize every
    learned full step to one and hide cumulative near-no-op updates.  Keep the
    configured relaxation as an immutable reference until the next physical
    step resets both values.
    """

    full_omega = float(full_picard_relaxation)
    if not math.isfinite(full_omega) or full_omega <= 0.0:
        raise ValueError("full_picard_relaxation must be finite and positive")
    normal_reference = _iqn_ils_normal_reference_step_report(
        proposed_over_current_picard_step=1.0,
        current_picard_relaxation=full_omega,
        beta=beta,
        configured_picard_reference_relaxation=(
            configured_picard_reference_relaxation
        ),
    )
    raw_effective = full_omega * float(beta)
    return {
        **normal_reference,
        "full_picard_relaxation": full_omega,
        "raw_effective_picard_relaxation": raw_effective,
    }


def _iqn_ils_picard_memory_update_report(
    *,
    current_picard_relaxation: float,
    measured_accepted_picard_relaxation: float | None,
    minimum_picard_relaxation: float = (
        FSI_IQN_ILS_PICARD_MEMORY_MIN_RELAXATION
    ),
) -> dict[str, float | bool | None]:
    """Bound stored Picard memory without rewriting the measured trial scale."""

    current = float(current_picard_relaxation)
    floor = float(minimum_picard_relaxation)
    if not math.isfinite(current) or current <= 0.0:
        raise ValueError("current_picard_relaxation must be finite and positive")
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum_picard_relaxation must be finite and positive")
    if current < floor:
        raise ValueError(
            "current_picard_relaxation must not be below the configured "
            f"memory floor {floor!r}; got {current!r}"
        )
    measured = (
        None
        if measured_accepted_picard_relaxation is None
        else float(measured_accepted_picard_relaxation)
    )
    if measured is not None and (not math.isfinite(measured) or measured <= 0.0):
        raise ValueError(
            "measured_accepted_picard_relaxation must be finite and positive"
        )
    stored = current if measured is None else max(floor, measured)
    return {
        "measured_accepted_picard_effective_relaxation": measured,
        "picard_memory_relaxation_before": current,
        "picard_memory_relaxation_after_acceptance": float(stored),
        "picard_relaxation_floor": floor,
        "picard_memory_updated": bool(measured is not None),
        "picard_relaxation_floor_applied": bool(
            measured is not None and measured < floor
        ),
    }


def _iqn_ils_accepted_picard_effective_relaxation(
    *,
    phase: str,
    full_picard_relaxation: float | None,
    full_recovery_relaxation: float | None = None,
    accepted_beta: float,
    accepted: bool,
) -> float | None:
    """Return the measured Picard damping learned by an accepted map step.

    An ordinary Picard proposal and the explicit global-best recovery both move
    along a measured fixed-point residual; after either one strictly decreases
    the real FSI residual, its effective damping is the safest next Picard
    memory.  A quasi-Newton direction has different semantics and must not
    overwrite that memory.  Rejected trials likewise teach no accepted scale.
    """

    phase_name = str(phase).strip().lower()
    if phase_name not in {"picard", "iqn", "recovery"}:
        raise ValueError(f"unsupported line-search phase: {phase!r}")
    beta = float(accepted_beta)
    if not math.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError("accepted_beta must be finite in (0, 1]")
    if not bool(accepted) or phase_name == "iqn":
        return None
    if phase_name == "picard":
        raw_full_relaxation = full_picard_relaxation
        relaxation_name = "full_picard_relaxation"
    else:
        raw_full_relaxation = full_recovery_relaxation
        relaxation_name = "full_recovery_relaxation"
    if raw_full_relaxation is None:
        raise ValueError(
            f"accepted {phase_name} line search requires {relaxation_name}"
        )
    full_omega = float(raw_full_relaxation)
    if not math.isfinite(full_omega) or full_omega <= 0.0:
        raise ValueError(
            f"{relaxation_name} must be finite and positive"
        )
    effective = full_omega * beta
    if not math.isfinite(effective) or effective <= 0.0:
        raise ValueError("effective Picard relaxation must be finite and positive")
    return float(effective)


def _iqn_ils_interpolated_line_search_state(
    *,
    source_state: np.ndarray,
    full_proposal: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Return a copied point on one immutable IQN/Picard search direction."""

    source = np.asarray(source_state)
    proposal = np.asarray(full_proposal)
    if source.shape != proposal.shape:
        raise ValueError("line-search source and full proposal shapes must match")
    if not np.issubdtype(source.dtype, np.floating) or not np.issubdtype(
        proposal.dtype, np.floating
    ):
        raise ValueError("line-search states must use floating-point dtypes")
    if not bool(np.all(np.isfinite(source))) or not bool(
        np.all(np.isfinite(proposal))
    ):
        raise ValueError("line-search states must be finite")
    weight = float(beta)
    if not math.isfinite(weight) or weight <= 0.0 or weight > 1.0:
        raise ValueError("line-search beta must be finite in (0, 1]")
    value64 = np.asarray(source, dtype=np.float64) + weight * (
        np.asarray(proposal, dtype=np.float64)
        - np.asarray(source, dtype=np.float64)
    )
    if not bool(np.all(np.isfinite(value64))):
        raise OverflowError("line-search interpolation overflow")
    with np.errstate(over="ignore", invalid="ignore"):
        value = value64.astype(source.dtype, copy=True)
    if not bool(np.all(np.isfinite(value))):
        raise OverflowError("line-search dtype conversion overflow")
    return value


def _iqn_ils_history_after_evaluation(
    *,
    velocity_guess_history: list[np.ndarray],
    velocity_candidate_history: list[np.ndarray],
    evaluated_velocity_guess: np.ndarray,
    evaluated_velocity_candidate: np.ndarray,
    accepted: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Copy history and append only a proposal accepted by the real map."""

    if len(velocity_guess_history) != len(velocity_candidate_history):
        raise ValueError("IQN guess and candidate histories must have equal length")
    guesses = [np.asarray(value).copy() for value in velocity_guess_history]
    candidates = [
        np.asarray(value).copy() for value in velocity_candidate_history
    ]
    guess = np.asarray(evaluated_velocity_guess)
    candidate = np.asarray(evaluated_velocity_candidate)
    if guess.shape != candidate.shape:
        raise ValueError("evaluated IQN guess and candidate shapes must match")
    if not bool(np.all(np.isfinite(guess))) or not bool(
        np.all(np.isfinite(candidate))
    ):
        raise ValueError("evaluated IQN guess and candidate must be finite")
    if bool(accepted):
        guesses.append(guess.copy())
        candidates.append(candidate.copy())
    return guesses, candidates


def _restore_and_persist_fsi_coupling_state_machine_failure(
    *,
    restore_fluid: Any,
    restore_solid: Any,
    restore_markers: Any,
    restore_gradient: Any,
    persist_failure: Any,
) -> Any:
    """Restore every coupled state before persisting one failure artifact."""

    operations = (
        ("restore_fluid", restore_fluid),
        ("restore_solid", restore_solid),
        ("restore_markers", restore_markers),
        ("restore_gradient", restore_gradient),
        ("persist_failure", persist_failure),
    )
    for name, operation in operations:
        if not callable(operation):
            raise TypeError(f"{name} must be callable")
    restore_fluid()
    restore_solid()
    restore_markers()
    restore_gradient()
    return persist_failure()


def _run_fsi_coupling_state_machine_operation(
    *,
    operation: Any,
    on_error: Any,
) -> Any:
    """Run one host-side coupling operation with fail-closed state handling."""

    if not callable(operation):
        raise TypeError("operation must be callable")
    if not callable(on_error):
        raise TypeError("on_error must be callable")
    try:
        return operation()
    except Exception as error:
        try:
            on_error(error)
        except Exception as handling_error:
            raise RuntimeError(
                "FSI coupling state-machine error handling failed: "
                f"{type(handling_error).__name__}: {handling_error}"
            ) from error
        raise


def _iqn_ils_near_tolerance_continuation_allowed(
    *,
    completed_trials: int,
    base_iteration_budget: int,
    best_absolute_residual_mps: float,
    absolute_tolerance_mps: float,
) -> bool:
    """Allow a small, fail-closed extension only for an already-near solution."""

    completed = int(completed_trials)
    budget = int(base_iteration_budget)
    best = float(best_absolute_residual_mps)
    tolerance = float(absolute_tolerance_mps)
    if completed < 0 or budget <= 0:
        raise ValueError("IQN trial counts must be non-negative with a positive budget")
    if not math.isfinite(best) or best < 0.0:
        raise ValueError("best_absolute_residual_mps must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance_mps must be finite and non-negative")
    if tolerance <= 0.0 or completed < budget:
        return False
    if completed >= budget + int(FSI_IQN_ILS_NEAR_TOLERANCE_EXTRA_TRIALS):
        return False
    return best <= float(FSI_IQN_ILS_NEAR_TOLERANCE_FACTOR) * tolerance


def _iqn_ils_budget_aware_next_beta_report(
    *,
    current_beta: float,
    ordinary_next_beta: float,
    completed_trials: int,
    base_iteration_budget: int,
    best_absolute_residual_mps: float,
    absolute_tolerance_mps: float,
) -> dict[str, Any]:
    """Fit the remaining registered IQN beta ladder inside its legal budget.

    A rejected full IQN proposal normally visits ``1, 1/2, 1/4, 1/8``.
    Starting that ladder late in a physical step can otherwise consume the
    remaining base trials before the registered minimum beta is evaluated.
    This helper selects the largest suffix entry whose complete tail still
    fits.  It never creates trials or borrows the near-tolerance extension
    unless the best *evaluated* residual already authorizes that extension.
    """

    beta = float(current_beta)
    ordinary = float(ordinary_next_beta)
    completed = int(completed_trials)
    budget = int(base_iteration_budget)
    best = float(best_absolute_residual_mps)
    tolerance = float(absolute_tolerance_mps)
    minimum_beta = float(FSI_IQN_ILS_LINE_SEARCH_MIN_BETA)
    if not math.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError("current_beta must be finite in (0, 1]")
    if not math.isfinite(ordinary) or ordinary <= 0.0 or ordinary >= beta:
        raise ValueError(
            "ordinary_next_beta must be finite, positive, and below current_beta"
        )
    if completed < 0 or budget <= 0:
        raise ValueError("IQN trial counts require completed >= 0 and budget > 0")
    if not math.isfinite(best) or best < 0.0:
        raise ValueError("best_absolute_residual_mps must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance_mps must be finite and non-negative")
    if (
        not math.isfinite(minimum_beta)
        or minimum_beta <= 0.0
        or minimum_beta > 1.0
    ):
        raise ValueError("registered minimum IQN beta must be finite in (0, 1]")

    registered_betas = [1.0]
    while registered_betas[-1] > minimum_beta:
        candidate = 0.5 * registered_betas[-1]
        if candidate < minimum_beta and not math.isclose(
            candidate,
            minimum_beta,
            rel_tol=0.0,
            abs_tol=16.0 * np.finfo(float).eps,
        ):
            raise ValueError(
                "registered minimum IQN beta must lie on the halving ladder"
            )
        registered_betas.append(max(candidate, minimum_beta))

    def _registered_index(value: float, *, name: str) -> int:
        for index, registered in enumerate(registered_betas):
            if math.isclose(
                value,
                registered,
                rel_tol=0.0,
                abs_tol=16.0 * np.finfo(float).eps,
            ):
                return index
        raise ValueError(f"{name} must lie on the registered IQN beta ladder")

    current_index = _registered_index(beta, name="current_beta")
    ordinary_index = _registered_index(
        ordinary,
        name="ordinary_next_beta",
    )
    if ordinary_index != current_index + 1:
        raise ValueError(
            "ordinary_next_beta must be the adjacent successor of current_beta"
        )

    near_tolerance_eligible = bool(
        tolerance > 0.0
        and best > tolerance
        and best <= float(FSI_IQN_ILS_NEAR_TOLERANCE_FACTOR) * tolerance
    )
    legal_trial_limit = budget + (
        int(FSI_IQN_ILS_NEAR_TOLERANCE_EXTRA_TRIALS)
        if near_tolerance_eligible
        else 0
    )
    remaining_trial_slots = max(0, legal_trial_limit - completed)
    selected_index: int | None = ordinary_index
    while (
        selected_index is not None
        and len(registered_betas) - selected_index > remaining_trial_slots
    ):
        selected_index += 1
        if selected_index >= len(registered_betas):
            selected_index = None

    selected_beta = (
        None if selected_index is None else float(registered_betas[selected_index])
    )
    return {
        "action": "stop" if selected_beta is None else "schedule",
        "next_beta": selected_beta,
        "current_beta": beta,
        "ordinary_next_beta": ordinary,
        "minimum_beta": minimum_beta,
        "completed_trials": completed,
        "base_iteration_budget": budget,
        "legal_trial_limit": legal_trial_limit,
        "remaining_trial_slots": remaining_trial_slots,
        "near_tolerance_eligible": near_tolerance_eligible,
        "extra_trials_authorized": near_tolerance_eligible,
        "ordinary_tail_evaluations": len(registered_betas) - ordinary_index,
        "selected_tail_evaluations": (
            0 if selected_index is None else len(registered_betas) - selected_index
        ),
        "skipped_beta_count": (
            0 if selected_index is None else selected_index - ordinary_index
        ),
        "failure_reason": (
            "iteration_budget_exhausted" if selected_beta is None else None
        ),
    }


def _iqn_ils_line_search_exhaustion_transition_report(
    *,
    line_search_exhausted: bool,
    completed_trials: int,
    base_iteration_budget: int,
    best_absolute_residual_mps: float,
    absolute_tolerance_mps: float,
    cold_recovery_attempted: bool,
) -> dict[str, Any]:
    """Choose the fail-closed transition after one search direction ends."""

    completed = int(completed_trials)
    budget = int(base_iteration_budget)
    best = float(best_absolute_residual_mps)
    tolerance = float(absolute_tolerance_mps)
    if completed < 0 or budget <= 0:
        raise ValueError("IQN trial counts require completed >= 0 and budget > 0")
    if not math.isfinite(best) or best < 0.0:
        raise ValueError("best_absolute_residual_mps must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance_mps must be finite and non-negative")
    maximum_trial_limit = budget + int(
        FSI_IQN_ILS_NEAR_TOLERANCE_EXTRA_TRIALS
    )
    near_tolerance_eligible = bool(
        tolerance > 0.0
        and best > tolerance
        and best
        <= float(FSI_IQN_ILS_NEAR_TOLERANCE_FACTOR) * tolerance
        and completed < maximum_trial_limit
    )

    if completed >= maximum_trial_limit:
        action = "stop"
        failure_reason: str | None = "iteration_budget_exhausted"
        next_trial_index: int | None = None
    elif not bool(line_search_exhausted):
        action = "continue_current_search"
        failure_reason = None
        next_trial_index = completed + 1
    elif near_tolerance_eligible and not bool(cold_recovery_attempted):
        action = "schedule_best_recovery"
        failure_reason = None
        next_trial_index = completed + 1
    else:
        action = "stop"
        failure_reason = "line_search_exhausted"
        next_trial_index = None

    return {
        "action": action,
        "failure_reason": failure_reason,
        "converged": False,
        "completed_trials": completed,
        "next_trial_index": next_trial_index,
        "base_iteration_budget": budget,
        "maximum_trial_limit": maximum_trial_limit,
        "near_tolerance_eligible": near_tolerance_eligible,
        "cold_recovery_attempted": bool(cold_recovery_attempted),
        "best_absolute_residual_mps": best,
        "absolute_tolerance_mps": tolerance,
    }


def _iqn_ils_restarted_velocity_history(
    best_velocity_guess: np.ndarray,
    best_velocity_candidate: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return a one-pair copied history for a globalized IQN restart."""

    guess = np.asarray(best_velocity_guess)
    candidate = np.asarray(best_velocity_candidate)
    if guess.shape != candidate.shape:
        raise ValueError("best IQN guess and candidate must have matching shapes")
    if not bool(np.all(np.isfinite(guess))) or not bool(np.all(np.isfinite(candidate))):
        raise ValueError("best IQN guess and candidate must be finite")
    return [guess.copy()], [candidate.copy()]


def _iqn_ils_picard_fallback_history(
    *,
    velocity_guess_history: list[np.ndarray],
    velocity_candidate_history: list[np.ndarray],
    source_velocity_guess: np.ndarray,
    source_velocity_candidate: np.ndarray,
    restart_from_unregistered_source: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Retain accepted secants unless fallback starts from an unseen point.

    Evaluated line-search rejections are never appended to the IQN history, so
    an ordinary fallback from the existing global best has no contaminated
    columns to remove.  A stalled trial that becomes a better source is
    intentionally not registered as an accepted IQN point; only that path must
    restart from its newly measured map pair.
    """

    if bool(restart_from_unregistered_source):
        return _iqn_ils_restarted_velocity_history(
            source_velocity_guess,
            source_velocity_candidate,
        )
    return _iqn_ils_history_after_evaluation(
        velocity_guess_history=velocity_guess_history,
        velocity_candidate_history=velocity_candidate_history,
        evaluated_velocity_guess=source_velocity_guess,
        evaluated_velocity_candidate=source_velocity_candidate,
        accepted=False,
    )


def _iqn_ils_velocity_proposal_is_novel(
    *,
    proposed_velocity_flat: np.ndarray,
    evaluated_velocity_guesses: list[np.ndarray],
    application_dtype: Any,
) -> bool:
    """Compare a proposal after conversion to the actual marker-field dtype."""

    dtype = np.dtype(application_dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("application_dtype must be a floating-point dtype")
    proposal = np.asarray(proposed_velocity_flat, dtype=dtype)
    if not bool(np.all(np.isfinite(proposal))):
        raise ValueError("proposed velocity must be finite after dtype conversion")
    for value in evaluated_velocity_guesses:
        evaluated = np.asarray(value, dtype=dtype)
        if evaluated.shape != proposal.shape:
            raise ValueError(
                "evaluated and proposed recovery velocities must have equal shapes"
            )
        if not bool(np.all(np.isfinite(evaluated))):
            raise ValueError("evaluated recovery velocities must be finite")
        if np.array_equal(evaluated, proposal):
            return False
    return True


def _iqn_ils_first_novel_recovery_state(
    *,
    source_velocity_flat: np.ndarray,
    full_proposal_velocity_flat: np.ndarray,
    source_gradient_guess: np.ndarray,
    full_proposal_gradient: np.ndarray,
    first_beta: float,
    evaluated_velocity_guesses: list[np.ndarray],
    application_dtype: Any,
) -> dict[str, Any]:
    """Return the first application-state-novel recovery backtrack."""

    beta = float(first_beta)
    if not math.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError("first_beta must be finite in (0, 1]")
    while beta >= float(FSI_IQN_ILS_LINE_SEARCH_MIN_BETA):
        velocity = _iqn_ils_interpolated_line_search_state(
            source_state=source_velocity_flat,
            full_proposal=full_proposal_velocity_flat,
            beta=beta,
        )
        applied_velocity = np.asarray(velocity, dtype=np.dtype(application_dtype)).copy()
        if _iqn_ils_velocity_proposal_is_novel(
            proposed_velocity_flat=applied_velocity,
            evaluated_velocity_guesses=evaluated_velocity_guesses,
            application_dtype=application_dtype,
        ):
            gradient = _iqn_ils_interpolated_line_search_state(
                source_state=source_gradient_guess,
                full_proposal=full_proposal_gradient,
                beta=beta,
            )
            return {
                "action": "schedule",
                "beta": beta,
                "forced_next_velocity_flat": applied_velocity.copy(),
                "forced_next_gradient": np.asarray(gradient).copy(),
            }
        beta *= 0.5
    return {
        "action": "stop",
        "beta": None,
        "forced_next_velocity_flat": None,
        "forced_next_gradient": None,
    }


def _iqn_ils_global_best_cold_recovery_plan(
    *,
    diagnostic_index: int,
    best_global_trial_index: int,
    best_velocity_guess: np.ndarray,
    best_velocity_candidate: np.ndarray,
    best_gradient_guess: np.ndarray,
    best_gradient_candidate: np.ndarray,
    best_absolute_residual_mps: float,
    evaluated_velocity_guesses: list[np.ndarray],
    application_dtype: Any,
    recovery_relaxation: float = (
        FSI_IQN_ILS_NEAR_TOLERANCE_RECOVERY_RELAXATION
    ),
) -> dict[str, Any]:
    """Build one paired, novel cold recovery from the immutable global best."""

    diagnostic_slot = int(diagnostic_index)
    source_trial = int(best_global_trial_index)
    if diagnostic_slot < 0 or source_trial <= 0:
        raise ValueError("recovery diagnostic and source-trial indices are invalid")
    best_residual = float(best_absolute_residual_mps)
    if not math.isfinite(best_residual) or best_residual < 0.0:
        raise ValueError("best recovery residual must be finite and non-negative")
    omega = float(recovery_relaxation)
    if not math.isfinite(omega) or omega <= 0.0 or omega > 1.0:
        raise ValueError("recovery_relaxation must be finite in (0, 1]")

    velocity_guess = np.asarray(best_velocity_guess)
    velocity_candidate = np.asarray(best_velocity_candidate)
    gradient_guess = np.asarray(best_gradient_guess)
    gradient_candidate = np.asarray(best_gradient_candidate)
    if velocity_guess.shape != velocity_candidate.shape:
        raise ValueError("best recovery velocity pair must have matching shapes")
    if gradient_guess.shape != gradient_candidate.shape:
        raise ValueError("best recovery gradient pair must have matching shapes")
    for name, value in (
        ("best_velocity_guess", velocity_guess),
        ("best_velocity_candidate", velocity_candidate),
        ("best_gradient_guess", gradient_guess),
        ("best_gradient_candidate", gradient_candidate),
    ):
        if not np.issubdtype(value.dtype, np.floating) or not bool(
            np.all(np.isfinite(value))
        ):
            raise ValueError(f"{name} must be a finite floating-point array")

    full_velocity = _iqn_ils_interpolated_line_search_state(
        source_state=velocity_guess,
        full_proposal=velocity_candidate,
        beta=omega,
    )
    full_velocity = np.asarray(
        full_velocity, dtype=np.dtype(application_dtype)
    ).copy()
    full_gradient = _iqn_ils_interpolated_line_search_state(
        source_state=gradient_guess,
        full_proposal=gradient_candidate,
        beta=omega,
    )
    novel_state = _iqn_ils_first_novel_recovery_state(
        source_velocity_flat=velocity_guess,
        full_proposal_velocity_flat=full_velocity,
        source_gradient_guess=gradient_guess,
        full_proposal_gradient=full_gradient,
        first_beta=1.0,
        evaluated_velocity_guesses=evaluated_velocity_guesses,
        application_dtype=application_dtype,
    )
    base_diagnostic = {
        "update_mode": "near_band_global_best_cold_recovery",
        "phase": "recovery",
        "fallback_reason": "line_search_exhausted_near_tolerance",
        "history_reset_required": True,
        "source_global_trial_index": source_trial,
        "best_absolute_residual_mps_at_proposal": best_residual,
        "cold_recovery_full_relaxation": omega,
        "cold_recovery_attempted": True,
    }
    if novel_state["action"] != "schedule":
        return {
            "action": "stop",
            "failure_reason": "line_search_exhausted",
            "pending_line_search": None,
            "forced_next_velocity_flat": None,
            "forced_next_gradient": None,
            "velocity_guess_history": None,
            "velocity_candidate_history": None,
            "diagnostic": {
                **base_diagnostic,
                "proposal_novel": False,
                "cold_recovery_effective_relaxation": None,
                "termination_reason": "line_search_exhausted",
            },
        }

    selected_beta = float(novel_state["beta"])
    forced_velocity = np.asarray(
        novel_state["forced_next_velocity_flat"]
    ).copy()
    forced_gradient = np.asarray(novel_state["forced_next_gradient"]).copy()
    pending = _IqnPendingLineSearch(
        diagnostic_index=diagnostic_slot,
        phase="recovery",
        source_velocity_flat=velocity_guess.copy(),
        source_candidate_velocity_flat=velocity_candidate.copy(),
        full_proposal_velocity_flat=full_velocity.copy(),
        source_gradient_guess=gradient_guess.copy(),
        source_gradient_candidate=gradient_candidate.copy(),
        full_proposal_gradient=np.asarray(full_gradient).copy(),
        source_absolute_residual_mps=best_residual,
        beta=selected_beta,
        full_proposed_over_fallback_step=None,
        full_picard_relaxation=None,
        configured_picard_reference_relaxation=None,
    )
    velocity_guesses, velocity_candidates = _iqn_ils_restarted_velocity_history(
        velocity_guess,
        velocity_candidate,
    )
    return {
        "action": "schedule_best_recovery",
        "failure_reason": None,
        "pending_line_search": pending,
        "forced_next_velocity_flat": forced_velocity.copy(),
        "forced_next_gradient": forced_gradient.copy(),
        "velocity_guess_history": velocity_guesses,
        "velocity_candidate_history": velocity_candidates,
        "diagnostic": {
            **base_diagnostic,
            "proposal_novel": True,
            "cold_recovery_selected_beta": selected_beta,
            "cold_recovery_effective_relaxation": omega * selected_beta,
            "termination_reason": None,
        },
    }


def _iqn_ils_shrunk_recovery_relaxation(current_relaxation: float) -> float:
    """Backtrack repeated best-point recoveries without repeating one guess."""

    relaxation = float(current_relaxation)
    if not math.isfinite(relaxation) or relaxation <= 0.0:
        raise ValueError("current_recovery_relaxation must be finite and positive")
    return max(
        float(FSI_AITKEN_RELAXATION_LOWER),
        0.5 * relaxation,
    )


def _iqn_ils_pressure_neumann_gradient_guess(
    *,
    current_gradient_guess: np.ndarray,
    current_gradient_candidate: np.ndarray,
    best_gradient_guess: np.ndarray,
    best_gradient_candidate: np.ndarray,
    iqn_update_diagnostic: dict[str, Any],
) -> np.ndarray:
    """Keep the diagnostic gradient pair consistent with an IQN velocity step.

    Core recomputes the physical pressure-Neumann gradient from the current
    velocity guess before assembling pressure rows, so this array is not an
    independent input to the physical fixed-point map.  It is nevertheless
    restored between trials and recorded in failure evidence.  A best-point
    recovery must therefore keep its input/output pair from the same evaluated
    trial, while an ordinary velocity update uses the registered relaxed-map
    damping instead of silently jumping this diagnostic state with omega=1.
    """

    current_guess = np.asarray(current_gradient_guess)
    current = np.asarray(current_gradient_candidate)
    best_guess = np.asarray(best_gradient_guess)
    best_candidate = np.asarray(best_gradient_candidate)
    if best_guess.shape != best_candidate.shape:
        raise ValueError(
            "best IQN pressure-Neumann gradient guess and candidate must match"
        )
    if current_guess.shape != current.shape or current.shape != best_guess.shape:
        raise ValueError(
            "current and best IQN pressure-Neumann gradients must match"
        )
    if not bool(np.all(np.isfinite(current_guess))) or not bool(
        np.all(np.isfinite(current))
    ) or not bool(
        np.all(np.isfinite(best_guess))
    ) or not bool(np.all(np.isfinite(best_candidate))):
        raise ValueError("IQN pressure-Neumann gradient states must be finite")
    if not bool(iqn_update_diagnostic.get("history_reset_required", False)):
        raw_complement_relaxation = iqn_update_diagnostic.get(
            "unmodeled_complement_relaxation"
        )
        if raw_complement_relaxation is None:
            raise ValueError(
                "normal IQN gradient update requires "
                "unmodeled_complement_relaxation"
            )
        return relaxed_sharp_pressure_neumann_gradient_state_array(
            current_guess,
            current,
            relaxation=float(raw_complement_relaxation),
        ).copy()
    raw_recovery = iqn_update_diagnostic.get("recovery_relaxation")
    if raw_recovery is None:
        raise ValueError(
            "IQN gradient recovery requires the applied recovery_relaxation"
        )
    return relaxed_sharp_pressure_neumann_gradient_state_array(
        best_guess,
        best_candidate,
        relaxation=float(raw_recovery),
    ).copy()


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
    state = sharp_marker_state_arrays(markers)
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

    # Preserve every OTHER state key untouched (currently none beyond the
    # four resampled below, but this keeps the helper correct if
    # CHECKPOINT_MARKER_STATE_FIELD_NAMES ever grows).
    new_state = dict(state)
    new_state["x_gamma_m"] = new_x.astype(
        np.asarray(state["x_gamma_m"]).dtype, copy=False
    )
    new_state["v_gamma_mps"] = new_v.astype(
        np.asarray(state["v_gamma_mps"]).dtype, copy=False
    )
    new_state["n_gamma"] = new_n.astype(
        np.asarray(state["n_gamma"]).dtype, copy=False
    )
    new_state["A_gamma_m2"] = new_a.astype(
        np.asarray(state["A_gamma_m2"]).dtype, copy=False
    )
    restore_sharp_marker_state_arrays(markers, new_state)


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


def _fsi_coupling_tolerance_gate_report(
    *,
    relative_residual: float,
    relative_tolerance: float,
    absolute_residual_mps: float,
    absolute_tolerance_mps: float,
    require_absolute_tolerance: bool = False,
) -> dict[str, bool | str | float]:
    """Apply one authoritative coupling gate without relative-only bypass."""

    relative = float(relative_residual)
    relative_limit = float(relative_tolerance)
    absolute = float(absolute_residual_mps)
    absolute_limit = float(absolute_tolerance_mps)
    if not math.isfinite(relative) or relative < 0.0:
        raise ValueError("relative coupling residual must be finite and non-negative")
    if not math.isfinite(relative_limit) or relative_limit <= 0.0:
        raise ValueError("relative coupling tolerance must be finite and positive")
    if not math.isfinite(absolute) or absolute < 0.0:
        raise ValueError("absolute coupling residual must be finite and non-negative")
    if not math.isfinite(absolute_limit) or absolute_limit < 0.0:
        raise ValueError("absolute coupling tolerance must be finite and non-negative")

    relative_hit = relative < relative_limit
    absolute_gate_enabled = absolute_limit > 0.0
    absolute_hit = bool(absolute_gate_enabled and absolute <= absolute_limit)
    absolute_gate_required = bool(require_absolute_tolerance)
    if absolute_gate_required and not absolute_gate_enabled:
        raise ValueError(
            "required absolute coupling gate needs a positive absolute tolerance"
        )
    converged = (
        absolute_hit
        if absolute_gate_required
        else bool(relative_hit or absolute_hit)
    )
    reason = ""
    if converged:
        reason = (
            "absolute_tolerance"
            if absolute_gate_required
            else ("relative_tolerance" if relative_hit else "absolute_tolerance")
        )
    return {
        "converged": bool(converged),
        "reason": reason,
        "relative_tolerance_hit": bool(relative_hit),
        "absolute_tolerance_hit": bool(absolute_hit),
        "absolute_gate_enabled": bool(absolute_gate_enabled),
        "absolute_gate_required": absolute_gate_required,
        "relative_residual": relative,
        "relative_tolerance": relative_limit,
        "absolute_residual_mps": absolute,
        "absolute_tolerance_mps": absolute_limit,
    }


def _fsi_coupling_convergence_certificate(
    *,
    residual_measured: bool,
    relative_residual: float,
    relative_tolerance: float,
    absolute_residual_mps: float | None,
    absolute_tolerance_mps: float,
    require_absolute_tolerance: bool = False,
    nonconvergence_reason: str | None = None,
) -> dict[str, bool | str | float | None]:
    """Classify an already-completed coupling loop without changing its path.

    The legacy single-pass path keeps its historical relative-residual sentinel
    (0.0), but receives an explicit unmeasured certificate and a JSON-safe null
    absolute residual. On the measured path, reason precedence mirrors the
    existing convergence condition: relative tolerance first, then the gated
    absolute tolerance, otherwise the iteration budget was exhausted.
    """

    absolute_tolerance = float(absolute_tolerance_mps)
    explicit_nonconvergence_reason = (
        None
        if nonconvergence_reason is None
        else str(nonconvergence_reason).strip()
    )
    if nonconvergence_reason is not None and not explicit_nonconvergence_reason:
        raise ValueError("nonconvergence_reason must be non-empty when provided")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError(
            "absolute coupling tolerance must be finite and >= 0"
        )
    if not residual_measured:
        return {
            "fsi_coupling_residual_measured": False,
            "fsi_coupling_converged": False,
            "fsi_coupling_convergence_reason": "unmeasured_single_pass",
            "fsi_coupling_absolute_residual_mps": None,
        }
    if absolute_residual_mps is None:
        raise ValueError(
            "measured FSI coupling residual requires an absolute residual"
        )

    absolute_residual = float(absolute_residual_mps)
    gate_report = _fsi_coupling_tolerance_gate_report(
        relative_residual=relative_residual,
        relative_tolerance=relative_tolerance,
        absolute_residual_mps=absolute_residual,
        absolute_tolerance_mps=absolute_tolerance,
        require_absolute_tolerance=bool(require_absolute_tolerance),
    )
    converged = bool(gate_report["converged"])
    if converged:
        reason = str(gate_report["reason"])
    else:
        reason = (
            explicit_nonconvergence_reason
            if explicit_nonconvergence_reason is not None
            else "iteration_budget_exhausted"
        )
    return {
        "fsi_coupling_residual_measured": True,
        "fsi_coupling_converged": converged,
        "fsi_coupling_convergence_reason": reason,
        "fsi_coupling_absolute_residual_mps": absolute_residual,
    }


def _fsi_coupling_velocity_residual_metrics(
    *,
    new_velocity_mps: np.ndarray,
    guess_velocity_mps: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Return marker-count-invariant interface velocity residual metrics."""

    candidate = np.asarray(new_velocity_mps, dtype=np.float64)
    guess = np.asarray(guess_velocity_mps, dtype=np.float64)
    if candidate.shape != guess.shape or candidate.ndim != 2 or candidate.shape[1] != 3:
        raise ValueError(
            "new and guess marker velocities must have the same (marker, 3) shape"
        )
    if candidate.shape[0] == 0:
        raise ValueError("at least one marker velocity is required")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(guess)):
        raise ValueError("marker velocities must be finite")
    delta = candidate - guess
    absolute_residual_mps = float(
        np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
    )
    velocity_scale_mps = float(
        np.sqrt(np.mean(np.sum(candidate * candidate, axis=1)))
    )
    max_marker_residual_mps = float(np.max(np.linalg.norm(delta, axis=1)))
    return {
        "velocity_residual_mps": delta.reshape(-1),
        "absolute_residual_mps": absolute_residual_mps,
        "max_marker_residual_mps": max_marker_residual_mps,
        "relative_residual": absolute_residual_mps
        / max(1.0e-30, velocity_scale_mps),
    }


def _fsi_coupling_marker_candidate_from_step_base(
    *,
    step_base_state: dict[str, np.ndarray],
    candidate_state: dict[str, np.ndarray],
    dt_s: float,
) -> dict[str, np.ndarray]:
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
    anchored = {
        name: np.asarray(value).copy()
        for name, value in candidate_state.items()
    }
    anchored["x_gamma_m"] = (
        base_position.astype(np.float64, copy=False)
        + dt * candidate_velocity.astype(np.float64, copy=False)
    ).astype(base_position.dtype, copy=False)
    return anchored


def _require_formal_fsi_coupling_convergence(
    certificate: dict[str, Any],
) -> None:
    """Reject a physical step that lacks a measured convergence certificate."""

    measured = bool(certificate.get("fsi_coupling_residual_measured", False))
    converged = bool(certificate.get("fsi_coupling_converged", False))
    reason = str(
        certificate.get("fsi_coupling_convergence_reason", "missing_certificate")
    )
    if not measured:
        raise RuntimeError(
            f"formal FSI coupling residual was not measured ({reason})"
        )
    if not converged:
        raise RuntimeError(f"formal FSI coupling did not converge ({reason})")


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
    # Strong-coupling diagnostics (legacy single pass reports the sentinels
    # 1 / 0.0 / 1.0: one trial, no measured residual, no relaxation applied).
    "fsi_coupling_iterations_used",
    "fsi_coupling_residual",
    "fsi_aitken_relaxation",
    # Additive convergence certificate. The three legacy fields above keep
    # their original order and sentinels for existing CSV consumers.
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
    "hibm_next_velocity_dirichlet_active_rows",
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


def _turek_hron_replay_step_indices(
    *,
    completed_step: int,
    requested_steps: int,
) -> tuple[int, ...]:
    """Return zero-based loop indices strictly after a committed checkpoint."""

    completed = int(completed_step)
    requested = int(requested_steps)
    if completed <= 0 or requested <= completed:
        raise ValueError(
            "completed_step must be positive and less than requested_steps"
        )
    return tuple(range(completed, requested))


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
    "velocity_dirichlet_boundary_active",
    "velocity_dirichlet_boundary_value_mps",
    "velocity_dirichlet_boundary_projection_weight",
    "velocity_dirichlet_boundary_marker_region_id",
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
    require_coupling_convergence: bool = False,
    fail_fast_probe: TurekHronMechanismProbe | None = None,
    transition_checkpoint_step: int | None = None,
    transition_diagnostic_step: int | None = None,
    resume_transition_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
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
    if (
        transition_diagnostic_step is not None
        and int(config.fsi_coupling_iterations) <= 1
    ):
        raise ValueError(
            "transition_diagnostic_step requires strong coupling"
        )
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = _build_fluid(config, runtime)
    solid, masks = _build_solid(config, runtime)
    markers = _build_markers(config, runtime)
    expected_marker_count = int(markers.marker_count)
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
    plane_dx_m, plane_dy_m, plane_dz_m = fluid_cell_spacing_m(config)
    plane_spacing_m = max(plane_dy_m, plane_dz_m)
    search_radius_m = 1.5 * plane_spacing_m
    interior_probe_distance_m = 1.0 * plane_spacing_m
    # Anisotropic envelope (gated, default off; see TurekHronFsiConfig's
    # ib_anisotropic_envelope comment and _validate_marker_grid_consistency's
    # docstring for the motivating porous-beam bug). None on the False path
    # keeps advance_hibm_mpm_sharp_mpm_step's scalar search_radius_m /
    # interior_probe_distance_m arguments byte-for-byte the sole envelope.
    search_radius_xyz_m = (
        (1.5 * plane_dx_m, 1.5 * plane_dy_m, 1.5 * plane_dz_m)
        if config.ib_anisotropic_envelope
        else None
    )
    interior_probe_distance_xyz_m = (
        (1.0 * plane_dx_m, 1.0 * plane_dy_m, 1.0 * plane_dz_m)
        if config.ib_anisotropic_envelope
        else None
    )

    solid_substep_count = int(config.solid_substeps)

    # Strong-coupling gate: with the default fsi_coupling_iterations=1 the
    # gated Picard branch below is never entered, so the legacy explicit
    # loose single pass is preserved byte-for-byte (no extra save_state, no
    # marker snapshots, no residual math on the legacy path).
    fsi_iterations = max(1, int(config.fsi_coupling_iterations))
    strong_coupling_enabled = fsi_iterations > 1
    if bool(require_coupling_convergence) and not strong_coupling_enabled:
        raise ValueError(
            "formal coupling convergence requires fsi_coupling_iterations > 1"
        )
    fsi_tolerance = float(config.fsi_coupling_tolerance)
    fsi_absolute_tolerance_mps = float(config.fsi_coupling_absolute_tolerance_mps)
    # Accelerator select: "aitken" (default) keeps the scalar Aitken update
    # byte-for-byte; "iqn_ils" swaps only the guess-update rule for the
    # interface quasi-Newton multi-secant update. Unknown values fall back to
    # Aitken rather than silently mis-coupling.
    fsi_accelerator = str(config.fsi_coupling_accelerator).strip().lower()
    fsi_use_iqn_ils = strong_coupling_enabled and fsi_accelerator == "iqn_ils"

    active_transition_diagnostic_arrays: dict[str, np.ndarray] | None = None

    def solid_step() -> Any:
        # The OOB count is accumulated as a sticky device-side maximum across
        # the full solid trial. This keeps the audit's fail-closed guarantee
        # (even a transient escape is retained), while replacing 100 scalar
        # GPU->CPU synchronizations with the final packed report's single host
        # read. end_out_of_bounds_guard_batch() raises before marker feedback
        # or any post-solid fluid/coupling work can continue.
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
            for substep_index in range(solid_substep_count):
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
                if config.enforce_plane_strain_x:
                    solid.enforce_rest_x_plane()
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
            # Clear only the host lifecycle before propagating. Device/particle
            # state is intentionally not rolled back here; a caller that elects
            # to recover must restore the previously saved strong-coupling base.
            solid.abort_out_of_bounds_guard_batch()
            raise

    history: list[dict[str, Any]] = []
    step_indices = tuple(range(int(config.step_count)))
    transition_checkpoint_path: Path | None = None
    transition_diagnostic_path: Path | None = None
    committed_transition_reference_arrays: dict[str, np.ndarray] = {}
    if resume_transition_checkpoint is not None:
        resume_metadata, resume_arrays = (
            _load_turek_hron_transition_checkpoint(
                Path(resume_transition_checkpoint)
            )
        )
        completed_step = _validate_turek_hron_transition_checkpoint_metadata(
            metadata=resume_metadata,
            config=config,
            preset=str(preset),
            particle_count=int(solid.particle_count),
            marker_count=int(markers.marker_count),
        )
        checkpoint_history = resume_metadata.get("history")
        if not isinstance(checkpoint_history, list) or len(
            checkpoint_history
        ) != int(completed_step):
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
        )
        history = [dict(row) for row in checkpoint_history]
        step_indices = _turek_hron_replay_step_indices(
            completed_step=int(completed_step),
            requested_steps=int(config.step_count),
        )
        committed_transition_reference_arrays = {
            name: np.asarray(value).copy()
            for name, value in resume_arrays.items()
            if name.startswith("committed_step_")
        }
    latest_report = None
    # Incremental history flush: append completed rows to disk every N steps so
    # a multi-hour run is inspectable mid-flight instead of a black box. The
    # final full write below still rewrites the complete CSV.
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
    # Periodic flow-contour snapshot export (2026-07-09): off by default
    # (config.flow_snapshot_interval_steps is None). When set, reuses the SAME
    # snapshot builder as the run-end export_final_flow_snapshot output below,
    # so the mid-run and final snapshots never drift apart -- only the
    # destination path and the added time_s scalar differ.
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
    for step_index in step_indices:
        # Tier-2 marker re-seeding (2026-07-09, gated by
        # marker_reseed_interval_steps -- see TurekHronFsiConfig's field
        # comment; precedent: the squid case's marker_remap_interval_steps).
        # Must run BEFORE _write_channel_boundary_rows and BEFORE the
        # strong-coupling save_state/marker snapshot below: the reseeded
        # marker state has to become THIS step's fixed-point base, not
        # something computed after that base snapshot -- reseeding later
        # would just be overwritten every trial by the Picard loop's
        # restore_sharp_marker_state_arrays(marker_guess) restore-to-base
        # calls, silently discarding the reseed. None (default) never
        # enters this branch, preserving legacy advect-only marker tracking
        # byte-for-byte.
        if (
            config.marker_reseed_interval_steps is not None
            and step_index > 0
            and step_index % int(config.marker_reseed_interval_steps) == 0
        ):
            _reseed_turek_hron_markers(markers, config)
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
            # The core split sampler combines this per-face pressure with the
            # physical viscous traction from fluid.mu. That shear term is part
            # of the low-Re FSI1 load and supplies physical viscous damping;
            # forcing stress viscosity to zero here produced pressure-only
            # coupling and destabilized the full inlet ramp.
            one_sided_pressure_primary_region_id=PRIMARY_REGION_ID,
            one_sided_primary_fluid_side_normal_sign=1.0,
            # Exact plane strain: solver x is the invariant span direction,
            # so both IB classification and viscous sampling extrude the
            # mid-plane marker surface across x rather than treating the four
            # spanwise layers as geometrically different slices.
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
            interpolate_velocity_dirichlet_with_interior=(
                bool(config.interpolate_velocity_dirichlet_with_interior)
            ),
            classify_far_internal_nodes=bool(config.classify_far_internal_nodes),
            search_radius_xyz_m=search_radius_xyz_m,
            interior_probe_distance_xyz_m=interior_probe_distance_xyz_m,
            )

        fsi_coupling_termination_reason: str | None = None
        if not strong_coupling_enabled:
            # Legacy explicit loose coupling: one pass, no save/restore, no
            # residual measurement. Sentinels 1 / 0.0 / 1.0 mirror the squid
            # precedent's "unmeasured_single_pass" reporting convention.
            latest_report = _advance_trial_once()
            fsi_coupling_iterations_used = 1
            fsi_coupling_residual = 0.0
            fsi_aitken_relaxation = 1.0
            fsi_residual_history = []
            fsi_absolute_residual_history: list[float] = []
            fsi_relaxation_history = []
            fsi_iqn_update_diagnostics: list[dict[str, Any]] = []
            fsi_coupling_absolute_residual_mps: float | None = None
            fsi_coupling_max_marker_residual_mps: float | None = None
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
            marker_step_base = {
                name: np.asarray(value).copy()
                for name, value in marker_guess.items()
            }
            gradient_guess = _marker_pressure_neumann_gradient_state(
                boundary, markers
            )
            gradient_step_base = np.asarray(gradient_guess).copy()
            relaxation = float(config.fsi_aitken_initial_relaxation)
            iqn_recovery_relaxation = float(relaxation)
            # Learned only within this physical step.  A new step always
            # starts from the configured value, while accepted Picard line
            # searches may reduce it for subsequent fallback/complement work.
            iqn_picard_reference_relaxation = float(relaxation)
            iqn_picard_relaxation = float(iqn_picard_reference_relaxation)
            previous_velocity_residual: np.ndarray | None = None
            fsi_coupling_iterations_used = 0
            fsi_coupling_residual = math.inf
            fsi_residual_history: list[float] = []
            fsi_absolute_residual_history: list[float] = []
            fsi_relaxation_history: list[float] = []
            fsi_iqn_update_diagnostics: list[dict[str, Any]] = []
            fsi_coupling_absolute_residual_mps: float | None = math.inf
            fsi_coupling_max_marker_residual_mps: float | None = math.inf
            latest_report = None
            # IQN-ILS secant history, reset every physical step (the quasi-Newton
            # columns are per-timestep; carrying them across steps would model a
            # stale map). Only appended to on the iqn_ils path.
            iqn_velocity_guess_history: list[np.ndarray] = []
            iqn_velocity_candidate_history: list[np.ndarray] = []
            # A proposal is accepted only after the next real nonlinear-map
            # evaluation.  The immutable source/full direction permits bounded
            # backtracking without letting rejected trial points pollute IQN
            # secants or drift the line-search origin.
            iqn_pending_line_search: _IqnPendingLineSearch | None = None
            iqn_best_absolute_residual_mps = math.inf
            iqn_best_velocity_guess: np.ndarray | None = None
            iqn_best_velocity_candidate: np.ndarray | None = None
            iqn_best_gradient_guess: np.ndarray | None = None
            iqn_best_gradient_candidate: np.ndarray | None = None
            iqn_best_global_trial_index: int | None = None
            iqn_evaluated_velocity_guesses: list[np.ndarray] = []
            iqn_near_band_cold_recovery_attempted = False
            iqn_newest_secant_rollback_attempted = False
            iqn_post_exhaustion_picard_lno_latched = False
            fsi_trial_limit = int(fsi_iterations) + (
                int(FSI_IQN_ILS_NEAR_TOLERANCE_EXTRA_TRIALS)
                if fsi_use_iqn_ils
                else 0
            )

            def _handle_coupling_state_machine_error(error: Exception) -> None:
                """Restore the rejected trial and persist its internal failure."""

                nonlocal incremental_header_written, last_flushed_index
                failure_payload = {
                    "schema_version": 1,
                    "case": TUREK_HRON_CASE_ID,
                    "preset": str(preset),
                    "failure_kind": "fsi_coupling_state_machine_exception",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "failed_step": int(step_index + 1),
                    "failed_time_s": float(t_s),
                    "completed_steps": int(len(history)),
                    "fsi_coupling_accelerator": str(fsi_accelerator),
                    "fsi_coupling_iteration_budget": int(fsi_iterations),
                    "fsi_coupling_maximum_trial_limit": int(fsi_trial_limit),
                    "fsi_coupling_iterations_used": int(
                        fsi_coupling_iterations_used
                    ),
                    "fsi_coupling_residual_history": [
                        float(value) for value in fsi_residual_history
                    ],
                    "fsi_coupling_absolute_residual_history_mps": [
                        float(value) for value in fsi_absolute_residual_history
                    ],
                    "fsi_aitken_relaxation_history": [
                        float(value) for value in fsi_relaxation_history
                    ],
                    "fsi_iqn_update_diagnostics": [
                        dict(value) for value in fsi_iqn_update_diagnostics
                    ],
                    "physical_state_restored": True,
                    "completed_history_rows_flushed": int(last_flushed_index),
                }
                (
                    incremental_header_written,
                    last_flushed_index,
                    _persistence_errors,
                ) = _restore_and_persist_fsi_coupling_state_machine_failure(
                    restore_fluid=fluid.restore_state,
                    restore_solid=solid.restore_state,
                    restore_markers=lambda: restore_sharp_marker_state_arrays(
                        markers, marker_step_base
                    ),
                    restore_gradient=lambda: (
                        _restore_marker_pressure_neumann_gradient_state(
                            boundary, markers, gradient_step_base
                        )
                    ),
                    persist_failure=lambda: (
                        _persist_fsi_coupling_failure_evidence(
                            incremental_history_path=incremental_history_path,
                            history=history,
                            last_flushed_index=last_flushed_index,
                            incremental_header_written=(
                                incremental_header_written
                            ),
                            output_dir=output_dir,
                            failure_payload=failure_payload,
                        )
                    ),
                )

            def _state_machine_call(operation: Any) -> Any:
                return _run_fsi_coupling_state_machine_operation(
                    operation=operation,
                    on_error=_handle_coupling_state_machine_error,
                )

            for coupling_iteration in range(fsi_trial_limit):
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
                # Iterate interface velocity/Neumann data while keeping the
                # geometric integration anchored to the start of this physical
                # step. Feeding a relaxed end-step x back into core's
                # x_new=x_old+dt*v would advance the marker more than one dt
                # across Picard trials.
                marker_trial_input = {
                    name: np.asarray(value).copy()
                    for name, value in marker_guess.items()
                }
                for geometric_name in ("x_gamma_m", "n_gamma", "A_gamma_m2"):
                    marker_trial_input[geometric_name] = np.asarray(
                        marker_step_base[geometric_name]
                    ).copy()
                restore_sharp_marker_state_arrays(markers, marker_trial_input)
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
                transition_trial_diagnostic_arrays: (
                    dict[str, np.ndarray] | None
                ) = None
                capture_transition_trial = _transition_diagnostic_requested(
                    configured_step=transition_diagnostic_step,
                    physical_step=step_index + 1,
                    coupling_iteration=coupling_iteration,
                )
                if capture_transition_trial:
                    transition_trial_diagnostic_arrays = {
                        name: np.asarray(value).copy()
                        for name, value in (
                            committed_transition_reference_arrays.items()
                        )
                    }
                    transition_trial_diagnostic_arrays.update(
                        _turek_hron_transition_diagnostic_stage_arrays(
                            stage="pre_trial",
                            fluid=fluid,
                            solid=solid,
                            markers=markers,
                            search=search,
                            boundary=boundary,
                        )
                    )
                    active_transition_diagnostic_arrays = (
                        transition_trial_diagnostic_arrays
                    )
                try:
                    # Route advance/diagnostic callback failures through the
                    # same atomic rollback path as accelerator state-machine
                    # failures.  The successful path is numerically identical.
                    latest_report = _state_machine_call(_advance_trial_once)
                finally:
                    active_transition_diagnostic_arrays = None
                fsi_coupling_iterations_used = coupling_iteration + 1
                marker_candidate = _fsi_coupling_marker_candidate_from_step_base(
                    step_base_state=marker_step_base,
                    candidate_state=sharp_marker_state_arrays(markers),
                    dt_s=float(config.dt_s),
                )
                # Keep the committed/current device marker state identical to
                # the one-step-anchored candidate used by the residual test.
                restore_sharp_marker_state_arrays(markers, marker_candidate)
                if transition_trial_diagnostic_arrays is not None:
                    transition_trial_diagnostic_arrays.update(
                        _turek_hron_transition_diagnostic_stage_arrays(
                            stage="post_surface_feedback",
                            fluid=fluid,
                            solid=solid,
                            markers=markers,
                            search=search,
                            boundary=boundary,
                        )
                    )
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
                if fsi_use_iqn_ils:
                    iqn_evaluated_velocity_guesses = [
                        *(
                            np.asarray(value).copy()
                            for value in iqn_evaluated_velocity_guesses
                        ),
                        guess_velocity.reshape(-1).copy(),
                    ]
                residual_metrics = _fsi_coupling_velocity_residual_metrics(
                    new_velocity_mps=new_velocity,
                    guess_velocity_mps=guess_velocity,
                )
                velocity_residual = np.asarray(
                    residual_metrics["velocity_residual_mps"], dtype=np.float64
                )
                absolute_residual_mps = float(
                    residual_metrics["absolute_residual_mps"]
                )
                fsi_coupling_absolute_residual_mps = absolute_residual_mps
                fsi_coupling_max_marker_residual_mps = float(
                    residual_metrics["max_marker_residual_mps"]
                )
                fsi_coupling_residual = float(
                    residual_metrics["relative_residual"]
                )
                fsi_residual_history.append(float(fsi_coupling_residual))
                fsi_absolute_residual_history.append(float(absolute_residual_mps))
                # This historical field is the normal Aitken/relaxed-map
                # factor.  A backtracked best-point recovery has its separate
                # exact value in fsi_iqn_update_diagnostics; mixing the two
                # made the failure artifact report the wrong normal IQN scale.
                fsi_relaxation_history.append(
                    float(
                        iqn_picard_relaxation
                        if fsi_use_iqn_ils
                        else relaxation
                    )
                )
                if transition_trial_diagnostic_arrays is not None:
                    transition_diagnostic_metadata = {
                        "schema_version": 1,
                        "case_id": TUREK_HRON_CASE_ID,
                        "preset": str(preset),
                        "diagnostic_kind": (
                            "first_strong_coupling_trial_transition"
                        ),
                        "physical_step": int(step_index + 1),
                        "time_s": float(t_s),
                        "coupling_iteration_zero_based": int(
                            coupling_iteration
                        ),
                        "absolute_interface_residual_mps": float(
                            absolute_residual_mps
                        ),
                        "relative_interface_residual": float(
                            fsi_coupling_residual
                        ),
                        "max_marker_interface_residual_mps": float(
                            fsi_coupling_max_marker_residual_mps
                        ),
                        "config_fingerprint": (
                            _turek_hron_checkpoint_config_fingerprint(config)
                        ),
                        "stage_semantics": {
                            "pre_trial": (
                                "restored step base before core row assembly"
                            ),
                            "pre_solid_load": (
                                "current trial rows, marker stress, and "
                                "scattered solid load at solid_step entry"
                            ),
                            "post_solid": (
                                "particle state after all solid substeps, "
                                "before marker feedback"
                            ),
                            "post_surface_feedback": (
                                "anchored marker candidate and next-step rows"
                            ),
                        },
                        "summary": _turek_hron_transition_diagnostic_summary(
                            transition_trial_diagnostic_arrays,
                            dt_s=float(config.dt_s),
                        ),
                    }
                    transition_diagnostic_path = (
                        Path(output_dir)
                        / (
                            "turek_hron_transition_diagnostic_step_"
                            f"{step_index + 1:06d}_trial_001.npz"
                        )
                    )
                    _state_machine_call(
                        lambda: _write_turek_hron_transition_checkpoint(
                            transition_diagnostic_path,
                            metadata=transition_diagnostic_metadata,
                            arrays=transition_trial_diagnostic_arrays,
                        )
                    )
                iqn_line_search_rejected = False
                iqn_line_search_exhausted = False
                iqn_forced_next_velocity_flat: np.ndarray | None = None
                iqn_forced_next_gradient: np.ndarray | None = None
                if fsi_use_iqn_ils and iqn_pending_line_search is not None:
                    pending = iqn_pending_line_search
                    stagnation_rejection_policy = _state_machine_call(
                        lambda: _iqn_ils_stagnation_rejection_policy_report(
                            phase=pending.phase,
                            best_absolute_residual_mps=(
                                iqn_best_absolute_residual_mps
                            ),
                            absolute_tolerance_mps=(
                                fsi_absolute_tolerance_mps
                            ),
                        )
                    )
                    line_search_report = _state_machine_call(
                        lambda: _iqn_ils_evaluated_line_search_report(
                            source_absolute_residual_mps=(
                                pending.source_absolute_residual_mps
                            ),
                            observed_absolute_residual_mps=(
                                absolute_residual_mps
                            ),
                            current_beta=pending.beta,
                            full_proposed_over_fallback_step=(
                                pending.full_proposed_over_fallback_step
                            ),
                            enforce_stagnation_rejection=(
                                stagnation_rejection_policy[
                                    "enforce_stagnation_rejection"
                                ]
                            ),
                        )
                    )
                    if not (
                        0
                        <= int(pending.diagnostic_index)
                        < len(fsi_iqn_update_diagnostics)
                    ):
                        diagnostic_error = RuntimeError(
                            "IQN line search has an invalid source diagnostic"
                        )
                        _handle_coupling_state_machine_error(diagnostic_error)
                        raise diagnostic_error
                    diagnostic = fsi_iqn_update_diagnostics[
                        int(pending.diagnostic_index)
                    ]
                    effective_picard_relaxation = (
                        None
                        if pending.full_picard_relaxation is None
                        else float(pending.full_picard_relaxation)
                        * float(pending.beta)
                    )
                    cold_recovery_full_relaxation = (
                        diagnostic.get("cold_recovery_full_relaxation")
                        if str(pending.phase) == "recovery"
                        else None
                    )
                    cold_recovery_effective_relaxation = (
                        None
                        if cold_recovery_full_relaxation is None
                        else float(cold_recovery_full_relaxation)
                        * float(pending.beta)
                    )
                    normal_reference_report: dict[str, float | None] = {}
                    if str(pending.phase) == "picard":
                        normal_reference_report = _state_machine_call(
                            lambda: _iqn_ils_picard_reference_step_report(
                                full_picard_relaxation=float(
                                    pending.full_picard_relaxation
                                ),
                                beta=float(pending.beta),
                                configured_picard_reference_relaxation=float(
                                    pending.configured_picard_reference_relaxation
                                ),
                            )
                        )
                    elif str(pending.phase) == "iqn":
                        configured_reference = float(
                            pending.configured_picard_reference_relaxation
                        )
                        full_configured_ratio = (
                            None
                            if pending.full_proposed_over_fallback_step is None
                            else float(pending.full_proposed_over_fallback_step)
                        )
                        normal_reference_report = {
                            "configured_picard_reference_relaxation": (
                                configured_reference
                            ),
                            "full_proposed_over_configured_picard_step": (
                                full_configured_ratio
                            ),
                            "effective_proposed_over_configured_picard_step": (
                                None
                                if full_configured_ratio is None
                                else float(pending.beta) * full_configured_ratio
                            ),
                        }
                    best_after_evaluation = min(
                        float(iqn_best_absolute_residual_mps),
                        float(absolute_residual_mps),
                    )
                    ordinary_next_beta = line_search_report["next_beta"]
                    budget_aware_next_beta: dict[str, Any] | None = None
                    if (
                        str(pending.phase) == "iqn"
                        and not bool(line_search_report["accepted"])
                        and ordinary_next_beta is not None
                    ):
                        budget_aware_next_beta = _state_machine_call(
                            lambda: _iqn_ils_budget_aware_next_beta_report(
                                current_beta=float(pending.beta),
                                ordinary_next_beta=float(ordinary_next_beta),
                                completed_trials=int(coupling_iteration + 1),
                                base_iteration_budget=int(fsi_iterations),
                                best_absolute_residual_mps=float(
                                    best_after_evaluation
                                ),
                                absolute_tolerance_mps=float(
                                    fsi_absolute_tolerance_mps
                                ),
                            )
                        )
                    iqn_ordinary_next_beta = (
                        None
                        if str(pending.phase) != "iqn"
                        or ordinary_next_beta is None
                        else float(ordinary_next_beta)
                    )
                    iqn_scheduled_next_beta = (
                        iqn_ordinary_next_beta
                        if budget_aware_next_beta is None
                        else budget_aware_next_beta["next_beta"]
                    )
                    evaluated_trial = {
                        **line_search_report,
                        "stagnation_rejection_policy": dict(
                            stagnation_rejection_policy
                        ),
                        "phase": str(pending.phase),
                        "global_trial_index": int(coupling_iteration + 1),
                        "full_picard_relaxation": (
                            None
                            if pending.full_picard_relaxation is None
                            else float(pending.full_picard_relaxation)
                        ),
                        "effective_picard_relaxation": (
                            effective_picard_relaxation
                        ),
                        "full_recovery_relaxation": (
                            cold_recovery_full_relaxation
                        ),
                        "effective_recovery_relaxation": (
                            cold_recovery_effective_relaxation
                        ),
                        "best_absolute_residual_mps_after_evaluation": (
                            best_after_evaluation
                        ),
                        "ordinary_next_beta": iqn_ordinary_next_beta,
                        "scheduled_next_beta": iqn_scheduled_next_beta,
                        "budget_aware_next_beta_report": (
                            None
                            if budget_aware_next_beta is None
                            else dict(budget_aware_next_beta)
                        ),
                        **normal_reference_report,
                    }
                    diagnostic.setdefault(
                        "evaluated_line_search_trials", []
                    ).append(evaluated_trial)
                    if budget_aware_next_beta is not None:
                        budget_decision = dict(budget_aware_next_beta)
                        diagnostic[
                            "budget_aware_next_beta_decisions"
                        ] = [
                            *diagnostic.get(
                                "budget_aware_next_beta_decisions", []
                            ),
                            budget_decision,
                        ]
                        diagnostic["budget_aware_next_beta"] = dict(
                            budget_decision
                        )
                    diagnostic.update(
                        {
                            "observed_next_absolute_residual_mps": float(
                                absolute_residual_mps
                            ),
                            "observed_next_residual_ratio": (
                                line_search_report["observed_residual_ratio"]
                            ),
                            "stalled_model_rejected": bool(
                                line_search_report["stalled_model_rejected"]
                            ),
                            "line_search_last_accepted": bool(
                                line_search_report["accepted"]
                            ),
                            "line_search_last_rejection_reason": str(
                                line_search_report["rejection_reason"]
                            ),
                        }
                    )
                    if bool(line_search_report["accepted"]):
                        accepted_relaxed_map_relaxation = _state_machine_call(
                            lambda: _iqn_ils_accepted_picard_effective_relaxation(
                                phase=pending.phase,
                                full_picard_relaxation=(
                                    pending.full_picard_relaxation
                                ),
                                full_recovery_relaxation=(
                                    cold_recovery_full_relaxation
                                ),
                                accepted_beta=pending.beta,
                                accepted=True,
                            )
                        )
                        picard_memory_report = _state_machine_call(
                            lambda: _iqn_ils_picard_memory_update_report(
                                current_picard_relaxation=(
                                    iqn_picard_relaxation
                                ),
                                measured_accepted_picard_relaxation=(
                                    accepted_relaxed_map_relaxation
                                ),
                            )
                        )
                        diagnostic.update(picard_memory_report)
                        if accepted_relaxed_map_relaxation is not None:
                            iqn_picard_relaxation = float(
                                picard_memory_report[
                                    "picard_memory_relaxation_after_acceptance"
                                ]
                            )
                            diagnostic[
                                "accepted_relaxed_map_effective_relaxation"
                            ] = float(accepted_relaxed_map_relaxation)
                            diagnostic[
                                (
                                    "accepted_recovery_effective_relaxation"
                                    if str(pending.phase) == "recovery"
                                    else "accepted_picard_effective_relaxation"
                                )
                            ] = float(accepted_relaxed_map_relaxation)
                        post_exhaustion_picard_report = _state_machine_call(
                            lambda: _iqn_ils_registered_iqn_exhaustion_picard_acceptance_report(
                                evaluated_line_search_trials=list(
                                    diagnostic.get(
                                        "evaluated_line_search_trials", []
                                    )
                                )
                            )
                        )
                        diagnostic[
                            "post_exhaustion_picard_lno_latch_report"
                        ] = dict(post_exhaustion_picard_report)
                        if bool(
                            post_exhaustion_picard_report["latch_candidate"]
                            and not iqn_newest_secant_rollback_attempted
                        ):
                            iqn_post_exhaustion_picard_lno_latched = True
                        iqn_pending_line_search = None
                    else:
                        iqn_line_search_rejected = True
                        next_beta = ordinary_next_beta
                        if next_beta is not None:
                            if str(pending.phase) == "recovery" and diagnostic.get(
                                "cold_recovery_full_relaxation"
                            ) is not None:
                                recovery_backtrack = _state_machine_call(
                                    lambda: _iqn_ils_first_novel_recovery_state(
                                        source_velocity_flat=(
                                            pending.source_velocity_flat
                                        ),
                                        full_proposal_velocity_flat=(
                                            pending.full_proposal_velocity_flat
                                        ),
                                        source_gradient_guess=(
                                            pending.source_gradient_guess
                                        ),
                                        full_proposal_gradient=(
                                            pending.full_proposal_gradient
                                        ),
                                        first_beta=float(next_beta),
                                        evaluated_velocity_guesses=(
                                            iqn_evaluated_velocity_guesses
                                        ),
                                        application_dtype=np.asarray(
                                            marker_candidate["v_gamma_mps"]
                                        ).dtype,
                                    )
                                )
                                if recovery_backtrack["action"] == "schedule":
                                    beta = float(recovery_backtrack["beta"])
                                    iqn_pending_line_search = replace(
                                        pending,
                                        beta=beta,
                                        had_prior_rejection=True,
                                    )
                                    iqn_forced_next_velocity_flat = np.asarray(
                                        recovery_backtrack[
                                            "forced_next_velocity_flat"
                                        ]
                                    ).copy()
                                    iqn_forced_next_gradient = np.asarray(
                                        recovery_backtrack[
                                            "forced_next_gradient"
                                        ]
                                    ).copy()
                                    diagnostic[
                                        "cold_recovery_next_effective_relaxation"
                                    ] = float(
                                        diagnostic[
                                            "cold_recovery_full_relaxation"
                                        ]
                                    ) * float(beta)
                                else:
                                    iqn_pending_line_search = None
                                    iqn_line_search_exhausted = True
                                    diagnostic["line_search_exhausted"] = True
                                    diagnostic[
                                        "line_search_last_rejection_reason"
                                    ] = "duplicate_recovery_proposal"
                            else:
                                beta = float(next_beta)
                                schedule_backtrack = True
                                if str(pending.phase) == "iqn":
                                    if budget_aware_next_beta is None:
                                        missing_budget_report = RuntimeError(
                                            "IQN budget-aware beta report was not initialized"
                                        )
                                        _handle_coupling_state_machine_error(
                                            missing_budget_report
                                        )
                                        raise missing_budget_report
                                    schedule_backtrack = bool(
                                        budget_aware_next_beta["action"]
                                        == "schedule"
                                    )
                                    if schedule_backtrack:
                                        beta = float(
                                            budget_aware_next_beta["next_beta"]
                                        )
                                    else:
                                        iqn_pending_line_search = None
                                        iqn_line_search_exhausted = True
                                        diagnostic["line_search_exhausted"] = True
                                        diagnostic[
                                            "line_search_last_rejection_reason"
                                        ] = str(
                                            budget_aware_next_beta[
                                                "failure_reason"
                                            ]
                                        )
                                if schedule_backtrack:
                                    iqn_pending_line_search = replace(
                                        pending,
                                        beta=beta,
                                        had_prior_rejection=True,
                                    )
                                    iqn_forced_next_velocity_flat = (
                                        _state_machine_call(
                                            lambda: _iqn_ils_interpolated_line_search_state(
                                                source_state=(
                                                    pending.source_velocity_flat
                                                ),
                                                full_proposal=(
                                                    pending.full_proposal_velocity_flat
                                                ),
                                                beta=beta,
                                            )
                                        )
                                    )
                                    iqn_forced_next_gradient = (
                                        _state_machine_call(
                                            lambda: _iqn_ils_interpolated_line_search_state(
                                                source_state=(
                                                    pending.source_gradient_guess
                                                ),
                                                full_proposal=(
                                                    pending.full_proposal_gradient
                                                ),
                                                beta=beta,
                                            )
                                        )
                                    )
                        elif str(pending.phase) == "iqn":
                            # The local IQN direction failed every registered
                            # beta. A tiny stalled step that nevertheless made
                            # a real decrease becomes the new source; otherwise
                            # retain the immutable pre-IQN source. This keeps the
                            # fallback monotone with the algorithmic best.
                            observed_is_new_global_best = bool(
                                line_search_report[
                                    "stalled_improvement_available"
                                ]
                                and _state_machine_call(
                                    lambda: _iqn_ils_observed_trial_improves_global_best(
                                        observed_absolute_residual_mps=(
                                            absolute_residual_mps
                                        ),
                                        best_absolute_residual_mps=(
                                            iqn_best_absolute_residual_mps
                                        ),
                                    )
                                )
                            )
                            diagnostic[
                                "observed_trial_improves_global_best"
                            ] = observed_is_new_global_best
                            if observed_is_new_global_best:
                                picard_source_velocity = (
                                    guess_velocity.reshape(-1).copy()
                                )
                                picard_source_candidate = (
                                    new_velocity.reshape(-1).copy()
                                )
                                picard_source_gradient = np.asarray(
                                    gradient_guess
                                ).copy()
                                picard_source_gradient_candidate = np.asarray(
                                    gradient_candidate
                                ).copy()
                                picard_source_residual = float(
                                    absolute_residual_mps
                                )
                                diagnostic["line_search_restart_source"] = (
                                    "observed_new_global_best"
                                )
                            else:
                                if (
                                    iqn_best_velocity_guess is None
                                    or iqn_best_velocity_candidate is None
                                    or iqn_best_gradient_guess is None
                                    or iqn_best_gradient_candidate is None
                                ):
                                    best_state_error = RuntimeError(
                                        "IQN global-best state was not initialized"
                                    )
                                    _handle_coupling_state_machine_error(
                                        best_state_error
                                    )
                                    raise best_state_error
                                picard_source_velocity = (
                                    iqn_best_velocity_guess.copy()
                                )
                                picard_source_candidate = (
                                    iqn_best_velocity_candidate.copy()
                                )
                                picard_source_gradient = (
                                    iqn_best_gradient_guess.copy()
                                )
                                picard_source_gradient_candidate = (
                                    iqn_best_gradient_candidate.copy()
                                )
                                picard_source_residual = float(
                                    iqn_best_absolute_residual_mps
                                )
                                diagnostic["line_search_restart_source"] = (
                                    "existing_global_best"
                                )
                            # Try the trusted relaxed-Picard direction from the
                            # selected source under the same real-residual gate.
                            picard_reference_step_report = _state_machine_call(
                                lambda: _iqn_ils_picard_reference_step_report(
                                    full_picard_relaxation=(
                                        iqn_picard_relaxation
                                    ),
                                    beta=1.0,
                                    configured_picard_reference_relaxation=(
                                        iqn_picard_reference_relaxation
                                    ),
                                )
                            )
                            picard_velocity = (
                                _state_machine_call(
                                    lambda: _iqn_ils_interpolated_line_search_state(
                                        source_state=picard_source_velocity,
                                        full_proposal=(
                                            picard_source_candidate
                                        ),
                                        beta=iqn_picard_relaxation,
                                    )
                                )
                            )
                            picard_gradient = (
                                _state_machine_call(
                                    lambda: _iqn_ils_interpolated_line_search_state(
                                        source_state=picard_source_gradient,
                                        full_proposal=(
                                            picard_source_gradient_candidate
                                        ),
                                        beta=iqn_picard_relaxation,
                                    )
                                )
                            )
                            iqn_pending_line_search = replace(
                                pending,
                                phase="picard",
                                source_velocity_flat=(
                                    picard_source_velocity.copy()
                                ),
                                source_candidate_velocity_flat=(
                                    picard_source_candidate.copy()
                                ),
                                full_proposal_velocity_flat=(
                                    picard_velocity.copy()
                                ),
                                source_gradient_guess=(
                                    picard_source_gradient.copy()
                                ),
                                source_gradient_candidate=(
                                    picard_source_gradient_candidate.copy()
                                ),
                                full_proposal_gradient=picard_gradient.copy(),
                                source_absolute_residual_mps=(
                                    picard_source_residual
                                ),
                                beta=1.0,
                                had_prior_rejection=False,
                                full_proposed_over_fallback_step=float(
                                    picard_reference_step_report[
                                        "full_proposed_over_configured_picard_step"
                                    ]
                                ),
                                full_picard_relaxation=float(
                                    iqn_picard_relaxation
                                ),
                                configured_picard_reference_relaxation=float(
                                    iqn_picard_reference_relaxation
                                ),
                            )
                            iqn_forced_next_velocity_flat = picard_velocity
                            iqn_forced_next_gradient = picard_gradient
                            (
                                iqn_velocity_guess_history,
                                iqn_velocity_candidate_history,
                            ) = _state_machine_call(
                                lambda: _iqn_ils_picard_fallback_history(
                                    velocity_guess_history=(
                                        iqn_velocity_guess_history
                                    ),
                                    velocity_candidate_history=(
                                        iqn_velocity_candidate_history
                                    ),
                                    source_velocity_guess=(
                                        picard_source_velocity
                                    ),
                                    source_velocity_candidate=(
                                        picard_source_candidate
                                    ),
                                    restart_from_unregistered_source=bool(
                                        observed_is_new_global_best
                                    ),
                                )
                            )
                            diagnostic[
                                "line_search_history_reset_required"
                            ] = bool(observed_is_new_global_best)
                            diagnostic["line_search_history_policy"] = (
                                "restart_from_unregistered_observed_best"
                                if observed_is_new_global_best
                                else "retain_accepted_secants"
                            )
                            diagnostic["picard_restart_full_relaxation"] = float(
                                iqn_picard_relaxation
                            )
                        else:
                            iqn_pending_line_search = None
                            iqn_line_search_exhausted = True
                            diagnostic["line_search_exhausted"] = True
                if (
                    fsi_use_iqn_ils
                    and absolute_residual_mps < iqn_best_absolute_residual_mps
                ):
                    iqn_best_absolute_residual_mps = float(absolute_residual_mps)
                    iqn_best_global_trial_index = int(coupling_iteration + 1)
                    iqn_best_velocity_guess = guess_velocity.reshape(-1).copy()
                    iqn_best_velocity_candidate = new_velocity.reshape(-1).copy()
                    iqn_best_gradient_guess = np.asarray(gradient_guess).copy()
                    iqn_best_gradient_candidate = np.asarray(
                        gradient_candidate
                    ).copy()
                # Converged when the RELATIVE residual is below tolerance, OR
                # (gated, default off) when the ABSOLUTE mismatch is below the
                # physical floor -- see fsi_coupling_absolute_tolerance_mps.
                if fsi_coupling_residual < fsi_tolerance or (
                    fsi_absolute_tolerance_mps > 0.0
                    and absolute_residual_mps <= fsi_absolute_tolerance_mps
                ):
                    coupling_gate_report = _state_machine_call(
                        lambda: _fsi_coupling_tolerance_gate_report(
                            relative_residual=fsi_coupling_residual,
                            relative_tolerance=fsi_tolerance,
                            absolute_residual_mps=absolute_residual_mps,
                            absolute_tolerance_mps=(
                                fsi_absolute_tolerance_mps
                            ),
                            require_absolute_tolerance=bool(
                                require_coupling_convergence
                            ),
                        )
                    )
                    if bool(coupling_gate_report["converged"]):
                        # Keep the last advance only after the authoritative
                        # absolute gate approves it. A relative hit remains a
                        # diagnostic when an absolute tolerance is configured.
                        break
                completed_trials = coupling_iteration + 1
                if completed_trials >= fsi_iterations:
                    continue_near_tolerance = (
                        fsi_use_iqn_ils
                        and _iqn_ils_near_tolerance_continuation_allowed(
                            completed_trials=completed_trials,
                            base_iteration_budget=fsi_iterations,
                            best_absolute_residual_mps=(
                                iqn_best_absolute_residual_mps
                            ),
                            absolute_tolerance_mps=(
                                fsi_absolute_tolerance_mps
                            ),
                        )
                    )
                    if not continue_near_tolerance:
                        # The configured budget stays strict unless IQN has
                        # already entered the registered near-tolerance band;
                        # that continuation is itself capped at eight trials.
                        fsi_coupling_termination_reason = (
                            "iteration_budget_exhausted"
                        )
                        break
                if fsi_use_iqn_ils and iqn_line_search_rejected:
                    # The current trial was a real-map rejection.  It may still
                    # become the best diagnostic point above, but it must not
                    # enter the secant history.  Evaluate only the prescribed
                    # backtracked point on the next trial.
                    velocity_field = np.asarray(marker_candidate["v_gamma_mps"])
                    if iqn_line_search_exhausted:
                        exhaustion_transition = _state_machine_call(
                            lambda: _iqn_ils_line_search_exhaustion_transition_report(
                                line_search_exhausted=True,
                                completed_trials=completed_trials,
                                base_iteration_budget=fsi_iterations,
                                best_absolute_residual_mps=(
                                    iqn_best_absolute_residual_mps
                                ),
                                absolute_tolerance_mps=(
                                    fsi_absolute_tolerance_mps
                                ),
                                cold_recovery_attempted=(
                                    iqn_near_band_cold_recovery_attempted
                                ),
                            )
                        )
                        diagnostic["line_search_exhaustion_transition"] = dict(
                            exhaustion_transition
                        )
                        if (
                            exhaustion_transition["action"]
                            == "schedule_best_recovery"
                        ):
                            recovery_diagnostic_index = len(
                                fsi_iqn_update_diagnostics
                            )
                            recovery_plan = _state_machine_call(
                                lambda: _iqn_ils_global_best_cold_recovery_plan(
                                    diagnostic_index=(
                                        recovery_diagnostic_index
                                    ),
                                    best_global_trial_index=(
                                        iqn_best_global_trial_index
                                    ),
                                    best_velocity_guess=(
                                        iqn_best_velocity_guess
                                    ),
                                    best_velocity_candidate=(
                                        iqn_best_velocity_candidate
                                    ),
                                    best_gradient_guess=(
                                        iqn_best_gradient_guess
                                    ),
                                    best_gradient_candidate=(
                                        iqn_best_gradient_candidate
                                    ),
                                    best_absolute_residual_mps=(
                                        iqn_best_absolute_residual_mps
                                    ),
                                    evaluated_velocity_guesses=(
                                        iqn_evaluated_velocity_guesses
                                    ),
                                    application_dtype=velocity_field.dtype,
                                )
                            )
                            iqn_near_band_cold_recovery_attempted = True
                            fsi_iqn_update_diagnostics = [
                                *fsi_iqn_update_diagnostics,
                                dict(recovery_plan["diagnostic"]),
                            ]
                            diagnostic["near_band_cold_recovery_scheduled"] = bool(
                                recovery_plan["action"]
                                == "schedule_best_recovery"
                            )
                            diagnostic[
                                "near_band_cold_recovery_diagnostic_index"
                            ] = int(recovery_diagnostic_index)
                            if recovery_plan["action"] != "schedule_best_recovery":
                                fsi_coupling_termination_reason = str(
                                    recovery_plan["failure_reason"]
                                )
                                break
                            iqn_pending_line_search = recovery_plan[
                                "pending_line_search"
                            ]
                            iqn_forced_next_velocity_flat = np.asarray(
                                recovery_plan["forced_next_velocity_flat"]
                            ).copy()
                            iqn_forced_next_gradient = np.asarray(
                                recovery_plan["forced_next_gradient"]
                            ).copy()
                            iqn_velocity_guess_history = [
                                np.asarray(value).copy()
                                for value in recovery_plan[
                                    "velocity_guess_history"
                                ]
                            ]
                            iqn_velocity_candidate_history = [
                                np.asarray(value).copy()
                                for value in recovery_plan[
                                    "velocity_candidate_history"
                                ]
                            ]
                            iqn_line_search_exhausted = False
                        else:
                            fsi_coupling_termination_reason = str(
                                exhaustion_transition["failure_reason"]
                            )
                            break
                    if (
                        iqn_forced_next_velocity_flat is None
                        or iqn_forced_next_gradient is None
                    ):
                        fsi_coupling_termination_reason = (
                            "line_search_exhausted"
                        )
                        break
                    marker_guess = {
                        key: (value.copy() if hasattr(value, "copy") else value)
                        for key, value in marker_candidate.items()
                    }
                    marker_guess["v_gamma_mps"] = (
                        iqn_forced_next_velocity_flat.reshape(
                            velocity_field.shape
                        ).astype(velocity_field.dtype, copy=False)
                    )
                    gradient_guess = np.asarray(
                        iqn_forced_next_gradient,
                        dtype=np.asarray(gradient_candidate).dtype,
                    ).copy()
                    continue
                if fsi_use_iqn_ils:
                    # IQN-ILS accelerates the interface velocity, which is the
                    # actual changing input consumed by the physical trial map.
                    # The marker pressure-Neumann gradient is recomputed from
                    # this velocity inside core before pressure-row assembly;
                    # it is therefore paired for recovery consistency but is
                    # not an independent IQN input block.
                    (
                        iqn_velocity_guess_history,
                        iqn_velocity_candidate_history,
                    ) = _state_machine_call(
                        lambda: _iqn_ils_history_after_evaluation(
                            velocity_guess_history=(
                                iqn_velocity_guess_history
                            ),
                            velocity_candidate_history=(
                                iqn_velocity_candidate_history
                            ),
                            evaluated_velocity_guess=(
                                guess_velocity.reshape(-1)
                            ),
                            evaluated_velocity_candidate=(
                                new_velocity.reshape(-1)
                            ),
                            accepted=True,
                        )
                    )
                    if (
                        iqn_best_velocity_guess is None
                        or iqn_best_velocity_candidate is None
                        or iqn_best_gradient_guess is None
                        or iqn_best_gradient_candidate is None
                    ):
                        raise RuntimeError(
                            "IQN best-residual state was not initialized"
                        )
                    exclude_newest_secant_for_proposal = False
                    next_velocity_flat, iqn_update_diagnostic = _state_machine_call(
                        lambda: _globalized_iqn_velocity_guess(
                                velocity_guess_history=(
                                    iqn_velocity_guess_history
                                ),
                                velocity_candidate_history=(
                                    iqn_velocity_candidate_history
                                ),
                                exclude_newest_secant=(
                                    exclude_newest_secant_for_proposal
                                ),
                                current_absolute_residual_mps=(
                                    absolute_residual_mps
                                ),
                                best_absolute_residual_mps=(
                                    iqn_best_absolute_residual_mps
                                ),
                                best_velocity_guess=iqn_best_velocity_guess,
                                best_velocity_candidate=(
                                    iqn_best_velocity_candidate
                                ),
                                # Normal IQN/fallback always uses the registered
                                # relaxed-map damping. Geometric backtracking is
                                # reserved for explicit best-point recovery.
                                fallback_relaxation=iqn_picard_relaxation,
                                recovery_relaxation=(
                                    iqn_recovery_relaxation
                                ),
                                return_diagnostics=True,
                        )
                    )
                    if (
                        iqn_post_exhaustion_picard_lno_latched
                        and not exclude_newest_secant_for_proposal
                    ):
                        full_history_update_mode = str(
                            iqn_update_diagnostic.get("update_mode", "")
                        )
                        full_history_step_ratio = iqn_update_diagnostic.get(
                            "proposed_over_fallback_step"
                        )
                        full_history_initial_beta = 1.0
                        full_history_scale_report: dict[str, Any] | None = None
                        if (
                            full_history_update_mode == "iqn_ils"
                            and full_history_step_ratio is not None
                        ):
                            full_history_scale_report = _state_machine_call(
                                lambda: _iqn_ils_scale_aware_initial_beta_report(
                                    full_proposed_over_current_picard_step=float(
                                        full_history_step_ratio
                                    )
                                )
                            )
                            full_history_initial_beta = float(
                                full_history_scale_report["initial_beta"]
                            )
                        late_budget_lno_report = _state_machine_call(
                            lambda: _iqn_ils_late_budget_leave_newest_out_report(
                                prior_transition_eligible=True,
                                current_update_mode=full_history_update_mode,
                                full_history_initial_beta=(
                                    full_history_initial_beta
                                ),
                                retained_secant_column_count=int(
                                    iqn_update_diagnostic.get(
                                        "retained_secant_column_count", 0
                                    )
                                ),
                                completed_trials=int(coupling_iteration + 1),
                                base_iteration_budget=int(fsi_iterations),
                                best_absolute_residual_mps=float(
                                    iqn_best_absolute_residual_mps
                                ),
                                absolute_tolerance_mps=float(
                                    fsi_absolute_tolerance_mps
                                ),
                                already_attempted=(
                                    iqn_newest_secant_rollback_attempted
                                ),
                            )
                        )
                        iqn_update_diagnostic[
                            "late_budget_leave_newest_out_report"
                        ] = dict(late_budget_lno_report)
                        if (
                            late_budget_lno_report["action"]
                            == "recompute_without_newest_secant"
                        ):
                            counterfactual_full_history_iqn = {
                                "update_mode": full_history_update_mode,
                                "raw_secant_column_count": (
                                    iqn_update_diagnostic.get(
                                        "raw_secant_column_count"
                                    )
                                ),
                                "retained_secant_column_count": (
                                    iqn_update_diagnostic.get(
                                        "retained_secant_column_count"
                                    )
                                ),
                                "numerical_rank": iqn_update_diagnostic.get(
                                    "numerical_rank"
                                ),
                                "proposed_over_fallback_step": (
                                    full_history_step_ratio
                                ),
                                "scale_aware_initial_beta_report": (
                                    None
                                    if full_history_scale_report is None
                                    else dict(full_history_scale_report)
                                ),
                            }
                            (
                                lno_velocity_flat,
                                lno_update_diagnostic,
                            ) = _state_machine_call(
                                lambda: _globalized_iqn_velocity_guess(
                                    velocity_guess_history=(
                                        iqn_velocity_guess_history
                                    ),
                                    velocity_candidate_history=(
                                        iqn_velocity_candidate_history
                                    ),
                                    exclude_newest_secant=True,
                                    current_absolute_residual_mps=(
                                        absolute_residual_mps
                                    ),
                                    best_absolute_residual_mps=(
                                        iqn_best_absolute_residual_mps
                                    ),
                                    best_velocity_guess=iqn_best_velocity_guess,
                                    best_velocity_candidate=(
                                        iqn_best_velocity_candidate
                                    ),
                                    fallback_relaxation=(
                                        iqn_picard_relaxation
                                    ),
                                    recovery_relaxation=(
                                        iqn_recovery_relaxation
                                    ),
                                    return_diagnostics=True,
                                )
                            )
                            lno_selection_report = _state_machine_call(
                                lambda: _iqn_ils_leave_newest_out_selection_report(
                                    selection_requested=True,
                                    normal_velocity_flat=next_velocity_flat,
                                    normal_diagnostic=iqn_update_diagnostic,
                                    alternate_velocity_flat=lno_velocity_flat,
                                    alternate_diagnostic=(
                                        lno_update_diagnostic
                                    ),
                                    counterfactual_full_history_iqn=(
                                        counterfactual_full_history_iqn
                                    ),
                                )
                            )
                            next_velocity_flat = np.asarray(
                                lno_selection_report[
                                    "selected_velocity_flat"
                                ]
                            ).copy()
                            iqn_update_diagnostic = dict(
                                lno_selection_report[
                                    "selected_diagnostic"
                                ]
                            )
                            lno_applied = bool(
                                lno_selection_report["applied"]
                            )
                            if lno_applied:
                                iqn_update_diagnostic[
                                    "newest_secant_exclusion_reason"
                                ] = (
                                    "late_budget_post_exhaustion_picard_lno"
                                )
                                iqn_update_diagnostic[
                                    "late_budget_leave_newest_out_report"
                                ] = dict(late_budget_lno_report)
                                exclude_newest_secant_for_proposal = True
                            else:
                                iqn_update_diagnostic[
                                    "late_budget_leave_newest_out_superseded"
                                ] = bool(
                                    lno_selection_report["superseded"]
                                )
                        iqn_post_exhaustion_picard_lno_latched = False
                    elif iqn_post_exhaustion_picard_lno_latched:
                        # A different one-shot rollback already owns this next
                        # proposal; do not carry stale transition provenance.
                        iqn_post_exhaustion_picard_lno_latched = False
                    rollback_consumption_report = _state_machine_call(
                        lambda: _iqn_ils_newest_secant_rollback_consumption_report(
                            exclusion_requested=(
                                exclude_newest_secant_for_proposal
                            ),
                            iqn_update_diagnostic=iqn_update_diagnostic,
                        )
                    )
                    if bool(
                        rollback_consumption_report[
                            "mark_rollback_attempted"
                        ]
                    ):
                        iqn_newest_secant_rollback_attempted = True
                    iqn_update_diagnostic[
                        "newest_secant_rollback_consumed"
                    ] = bool(rollback_consumption_report["applied"])
                    iqn_update_diagnostic[
                        "newest_secant_rollback_superseded"
                    ] = bool(rollback_consumption_report["superseded"])
                    iqn_update_diagnostic[
                        "newest_secant_rollback_consumption_report"
                    ] = dict(rollback_consumption_report)
                    iqn_update_diagnostic[
                        "newest_secant_rollback_already_attempted"
                    ] = bool(iqn_newest_secant_rollback_attempted)
                    next_gradient_guess = _state_machine_call(
                        lambda: _iqn_ils_pressure_neumann_gradient_guess(
                            current_gradient_guess=gradient_guess,
                            current_gradient_candidate=gradient_candidate,
                            best_gradient_guess=iqn_best_gradient_guess,
                            best_gradient_candidate=(
                                iqn_best_gradient_candidate
                            ),
                            iqn_update_diagnostic=iqn_update_diagnostic,
                        )
                    )
                    paired_gradient_recovery = bool(
                        iqn_update_diagnostic["history_reset_required"]
                    )
                    iqn_update_diagnostic[
                        "paired_neumann_gradient_recovery"
                    ] = paired_gradient_recovery
                    iqn_update_diagnostic["neumann_gradient_update_mode"] = (
                        "paired_best_recovery"
                        if paired_gradient_recovery
                        else "relaxed_unmodeled_complement"
                    )
                    iqn_update_diagnostic["neumann_gradient_relaxation"] = (
                        float(iqn_update_diagnostic["recovery_relaxation"])
                        if paired_gradient_recovery
                        else float(
                            iqn_update_diagnostic[
                                "unmodeled_complement_relaxation"
                            ]
                        )
                    )
                    diagnostic_index = len(fsi_iqn_update_diagnostics)
                    fsi_iqn_update_diagnostics.append(iqn_update_diagnostic)
                    proposed_step_ratio = iqn_update_diagnostic.get(
                        "proposed_over_fallback_step"
                    )
                    if paired_gradient_recovery:
                        pending_phase = "recovery"
                    elif iqn_update_diagnostic["update_mode"] == "iqn_ils":
                        pending_phase = "iqn"
                    else:
                        pending_phase = "picard"
                    iqn_update_diagnostic[
                        "proposal_source_global_trial_index"
                    ] = int(coupling_iteration + 1)
                    iqn_update_diagnostic[
                        "normal_picard_relaxation"
                    ] = float(iqn_picard_relaxation)
                    iqn_update_diagnostic[
                        "best_absolute_residual_mps_at_proposal"
                    ] = float(iqn_best_absolute_residual_mps)
                    if paired_gradient_recovery:
                        line_search_source_velocity = iqn_best_velocity_guess
                        line_search_source_candidate = (
                            iqn_best_velocity_candidate
                        )
                        line_search_source_gradient = iqn_best_gradient_guess
                        line_search_source_gradient_candidate = (
                            iqn_best_gradient_candidate
                        )
                        line_search_source_residual = (
                            iqn_best_absolute_residual_mps
                        )
                    else:
                        line_search_source_velocity = guess_velocity.reshape(-1)
                        line_search_source_candidate = new_velocity.reshape(-1)
                        line_search_source_gradient = np.asarray(gradient_guess)
                        line_search_source_gradient_candidate = np.asarray(
                            gradient_candidate
                        )
                        line_search_source_residual = absolute_residual_mps
                    full_next_velocity_flat = np.asarray(
                        next_velocity_flat
                    ).copy()
                    full_next_gradient_guess = np.asarray(
                        next_gradient_guess
                    ).copy()
                    initial_line_search_beta = 1.0
                    if pending_phase == "iqn":
                        if proposed_step_ratio is None:
                            missing_ratio_error = RuntimeError(
                                "IQN proposal is missing its current-Picard "
                                "step ratio"
                            )
                            _handle_coupling_state_machine_error(
                                missing_ratio_error
                            )
                            raise missing_ratio_error
                        scale_aware_initial_beta_report = _state_machine_call(
                            lambda: _iqn_ils_scale_aware_initial_beta_report(
                                full_proposed_over_current_picard_step=float(
                                    proposed_step_ratio
                                )
                            )
                        )
                        initial_line_search_beta = float(
                            scale_aware_initial_beta_report["initial_beta"]
                        )
                        iqn_update_diagnostic[
                            "scale_aware_initial_beta_report"
                        ] = dict(scale_aware_initial_beta_report)
                        if initial_line_search_beta < 1.0:
                            next_velocity_flat = _state_machine_call(
                                lambda: _iqn_ils_interpolated_line_search_state(
                                    source_state=line_search_source_velocity,
                                    full_proposal=full_next_velocity_flat,
                                    beta=initial_line_search_beta,
                                )
                            )
                            next_gradient_guess = _state_machine_call(
                                lambda: _iqn_ils_interpolated_line_search_state(
                                    source_state=line_search_source_gradient,
                                    full_proposal=full_next_gradient_guess,
                                    beta=initial_line_search_beta,
                                )
                            )
                    iqn_update_diagnostic["initial_line_search_beta"] = float(
                        initial_line_search_beta
                    )
                    pending_normal_reference_report: dict[
                        str, float | None
                    ] = {}
                    if pending_phase == "picard":
                        pending_normal_reference_report = _state_machine_call(
                            lambda: _iqn_ils_picard_reference_step_report(
                                full_picard_relaxation=(
                                    iqn_picard_relaxation
                                ),
                                beta=1.0,
                                configured_picard_reference_relaxation=(
                                    iqn_picard_reference_relaxation
                                ),
                            )
                        )
                    elif pending_phase == "iqn":
                        pending_normal_reference_report = _state_machine_call(
                            lambda: _iqn_ils_normal_reference_step_report(
                                proposed_over_current_picard_step=(
                                    proposed_step_ratio
                                ),
                                current_picard_relaxation=(
                                    iqn_picard_relaxation
                                ),
                                beta=initial_line_search_beta,
                                configured_picard_reference_relaxation=(
                                    iqn_picard_reference_relaxation
                                ),
                            )
                        )
                    iqn_update_diagnostic.update(
                        pending_normal_reference_report
                    )
                    configured_reference_ratio = (
                        pending_normal_reference_report.get(
                            "full_proposed_over_configured_picard_step"
                        )
                    )
                    if pending_phase in {"picard", "iqn"}:
                        pending_full_step_ratio = (
                            None
                            if configured_reference_ratio is None
                            else float(configured_reference_ratio)
                        )
                    else:
                        pending_full_step_ratio = (
                            1.0
                            if proposed_step_ratio is None
                            else float(proposed_step_ratio)
                        )
                    iqn_pending_line_search = _IqnPendingLineSearch(
                        diagnostic_index=int(diagnostic_index),
                        phase=pending_phase,
                        source_velocity_flat=np.asarray(
                            line_search_source_velocity, dtype=np.float64
                        ).copy(),
                        source_candidate_velocity_flat=np.asarray(
                            line_search_source_candidate, dtype=np.float64
                        ).copy(),
                        full_proposal_velocity_flat=np.asarray(
                            full_next_velocity_flat, dtype=np.float64
                        ).copy(),
                        source_gradient_guess=np.asarray(
                            line_search_source_gradient
                        ).copy(),
                        source_gradient_candidate=np.asarray(
                            line_search_source_gradient_candidate
                        ).copy(),
                        full_proposal_gradient=np.asarray(
                            full_next_gradient_guess
                        ).copy(),
                        source_absolute_residual_mps=float(
                            line_search_source_residual
                        ),
                        beta=float(initial_line_search_beta),
                        full_proposed_over_fallback_step=pending_full_step_ratio,
                        full_picard_relaxation=(
                            float(iqn_picard_relaxation)
                            if pending_phase == "picard"
                            else None
                        ),
                        configured_picard_reference_relaxation=(
                            float(iqn_picard_reference_relaxation)
                            if pending_phase in {"picard", "iqn"}
                            else None
                        ),
                    )
                    if bool(iqn_update_diagnostic["history_reset_required"]):
                        # The rejected extrapolation is still present in the
                        # failure diagnostics, but its secants must not steer
                        # the restarted local model away from the trusted best
                        # point.  Seed the new history with that evaluated map
                        # pair; the next trial supplies the next secant.
                        (
                            iqn_velocity_guess_history,
                            iqn_velocity_candidate_history,
                        ) = _state_machine_call(
                            lambda: _iqn_ils_restarted_velocity_history(
                                iqn_best_velocity_guess,
                                iqn_best_velocity_candidate,
                            )
                        )
                        iqn_recovery_relaxation = _state_machine_call(
                            lambda: _iqn_ils_shrunk_recovery_relaxation(
                                iqn_recovery_relaxation
                            )
                        )
                    velocity_field = np.asarray(marker_candidate["v_gamma_mps"])
                    marker_guess = {
                        key: (value.copy() if hasattr(value, "copy") else value)
                        for key, value in marker_candidate.items()
                    }
                    marker_guess["v_gamma_mps"] = next_velocity_flat.reshape(
                        velocity_field.shape
                    ).astype(velocity_field.dtype, copy=False)
                    gradient_guess = next_gradient_guess
                else:
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
        # Observability only: classify the result after the pre-existing loop
        # has selected the committed trial. This does not participate in the
        # convergence condition, break path, relaxation, or any physics state.
        fsi_coupling_certificate = _fsi_coupling_convergence_certificate(
            residual_measured=strong_coupling_enabled,
            relative_residual=fsi_coupling_residual,
            relative_tolerance=fsi_tolerance,
            absolute_residual_mps=fsi_coupling_absolute_residual_mps,
            absolute_tolerance_mps=fsi_absolute_tolerance_mps,
            require_absolute_tolerance=bool(require_coupling_convergence),
            nonconvergence_reason=fsi_coupling_termination_reason,
        )
        if bool(require_coupling_convergence):
            try:
                _require_formal_fsi_coupling_convergence(
                    fsi_coupling_certificate
                )
            except RuntimeError as convergence_error:
                # A non-converged fixed-point trial is not a physical time
                # step. Restore physical state before attempting any I/O so a
                # disk/permission failure cannot strand the rejected trial.
                fluid.restore_state()
                solid.restore_state()
                restore_sharp_marker_state_arrays(markers, marker_step_base)
                _restore_marker_pressure_neumann_gradient_state(
                    boundary, markers, gradient_step_base
                )
                failure_payload = {
                    "schema_version": 1,
                    "case": TUREK_HRON_CASE_ID,
                    "preset": str(preset),
                    "failed_step": int(step_index + 1),
                    "failed_time_s": float(t_s),
                    "completed_steps": int(len(history)),
                    "fsi_coupling_accelerator": str(fsi_accelerator),
                    "fsi_coupling_iteration_budget": int(fsi_iterations),
                    "fsi_coupling_maximum_trial_limit": int(fsi_trial_limit),
                    "fsi_coupling_near_tolerance_extra_trial_limit": int(
                        fsi_trial_limit - fsi_iterations
                    ),
                    "fsi_coupling_iterations_used": int(
                        fsi_coupling_iterations_used
                    ),
                    "fsi_coupling_relative_tolerance": float(fsi_tolerance),
                    "fsi_coupling_absolute_tolerance_mps": float(
                        fsi_absolute_tolerance_mps
                    ),
                    "fsi_coupling_residual_history": [
                        float(value) for value in fsi_residual_history
                    ],
                    "fsi_coupling_absolute_residual_history_mps": [
                        float(value) for value in fsi_absolute_residual_history
                    ],
                    "fsi_aitken_relaxation_history": [
                        float(value) for value in fsi_relaxation_history
                    ],
                    "fsi_iqn_update_diagnostics": [
                        dict(value) for value in fsi_iqn_update_diagnostics
                    ],
                    "fsi_coupling_certificate": {
                        key: value for key, value in fsi_coupling_certificate.items()
                    },
                    "physical_state_restored": True,
                    "completed_history_rows_flushed": int(last_flushed_index),
                    "last_guess_velocity_rms_mps": float(
                        np.sqrt(np.mean(np.sum(guess_velocity * guess_velocity, axis=1)))
                    ),
                    "last_candidate_velocity_rms_mps": float(
                        np.sqrt(np.mean(np.sum(new_velocity * new_velocity, axis=1)))
                    ),
                    "last_velocity_residual_max_mps": float(
                        residual_metrics["max_marker_residual_mps"]
                    ),
                }
                # Persist every earlier completed row; no history row is
                # appended for the rejected trial. Each I/O channel is
                # best-effort and independent so an I/O failure cannot mask
                # the already-restored physical convergence failure.
                (
                    incremental_header_written,
                    last_flushed_index,
                    persistence_errors,
                ) = _persist_fsi_coupling_failure_evidence(
                    incremental_history_path=incremental_history_path,
                    history=history,
                    last_flushed_index=last_flushed_index,
                    incremental_header_written=incremental_header_written,
                    output_dir=output_dir,
                    failure_payload=failure_payload,
                )
                # Search/boundary row scratch is not rebuilt here; this
                # exception is fatal-only and the run must not retry the step
                # from these objects.
                raise RuntimeError(
                    f"{convergence_error}; failed_step={step_index + 1}; "
                    f"absolute_residual_history_mps="
                    f"{fsi_absolute_residual_history}; "
                    f"relative_residual_history={fsi_residual_history}; "
                    f"evidence_persistence_errors={list(persistence_errors)}"
                ) from convergence_error
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
        inlet_flux_actual_m3ps, outlet_flux_m3ps = _boundary_fluxes_m3ps(
            fluid, config
        )
        flux_imbalance_m3ps = outlet_flux_m3ps - inlet_flux_actual_m3ps
        flux_scale_m3ps = max(
            abs(inlet_flux_actual_m3ps), abs(outlet_flux_m3ps)
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
                fsi_coupling_max_marker_residual_mps
            ),
            expected_marker_count=expected_marker_count,
        )
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
            "outlet_flux_m3ps": outlet_flux_m3ps,
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
            "history_schema_version": int(
                committed_observability["history_schema_version"]
            ),
            "stress_viscous_gradient_invalid_marker_count": int(
                load.fluid_stress.viscous_gradient_invalid_marker_count
            ),
            "stress_one_sided_pressure_marker_count": int(
                load.fluid_stress.one_sided_pressure_marker_count
            ),
            "fsi_coupling_iterations_used": int(fsi_coupling_iterations_used),
            "fsi_coupling_residual": float(fsi_coupling_residual),
            "fsi_aitken_relaxation": float(fsi_aitken_relaxation),
            **fsi_coupling_certificate,
            # Per-iteration diagnostics remain JSON-only; scalar convergence-
            # certificate fields are included in the CSV.
            "fsi_coupling_residual_history": list(fsi_residual_history),
            "fsi_coupling_absolute_residual_history_mps": list(
                fsi_absolute_residual_history
            ),
            "fsi_aitken_relaxation_history": list(fsi_relaxation_history),
            "fsi_iqn_update_diagnostics": [
                dict(value) for value in fsi_iqn_update_diagnostics
            ],
            "inlet_flux_actual_m3ps": inlet_flux_actual_m3ps,
            "flux_imbalance_m3ps": flux_imbalance_m3ps,
            "flux_imbalance_rel": flux_imbalance_rel,
            **committed_observability,
            # Discrete-state observability counters (2026-07): the T-H thin-beam
            # case can undergo signed-side membership flips when the moving
            # surface crosses a grid center (velocity-Dirichlet rows are then
            # added/removed in groups replicated along the inactive axis). These
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
                    mechanism_probe_decision.streaks.values(), default=0
                ),
            }
        )
        history.append(row)
        if _committed_transition_checkpoint_requested(
            configured_step=transition_checkpoint_step,
            completed_step=step_index + 1,
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
            checkpoint_metadata = (
                _turek_hron_transition_checkpoint_metadata(
                    config=config,
                    preset=str(preset),
                    completed_step=int(step_index + 1),
                    particle_count=int(solid.particle_count),
                    marker_count=int(markers.marker_count),
                )
            )
            checkpoint_metadata.update(
                {
                    "checkpoint_kind": "committed_fsi_transition_state",
                    "history": json.loads(
                        json.dumps(
                            history,
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
            transition_checkpoint_path = (
                Path(output_dir)
                / (
                    "turek_hron_transition_checkpoint_step_"
                    f"{step_index + 1:06d}.npz"
                )
            )
            _write_turek_hron_transition_checkpoint(
                transition_checkpoint_path,
                metadata=checkpoint_metadata,
                arrays=checkpoint_arrays,
            )
        flush_required = _history_flush_required(
            completed_step=step_index + 1,
            flush_interval=flush_interval,
            probe_triggered=mechanism_probe_decision.triggered,
        )
        if incremental_history_path is not None and flush_required:
            incremental_header_written = _flush_history_csv(
                incremental_history_path,
                history[last_flushed_index:],
                header_written=incremental_header_written,
            )
            last_flushed_index = len(history)
        if mechanism_probe_decision.triggered:
            raise TurekHronMechanismProbeTriggered(
                "Turek-Hron mechanism probe triggered at completed step "
                f"{step_index + 1}: {mechanism_probe_decision.reason}"
            )
        if (
            flow_snapshots_dir is not None
            and (step_index + 1) % flow_snapshot_interval == 0
        ):
            periodic_snapshot = build_turek_hron_final_fields_snapshot(
                fluid, solid, config
            )
            periodic_snapshot["time_s"] = np.asarray(t_s, dtype=np.float64)
            np.savez(
                flow_snapshots_dir / f"step_{step_index + 1:06d}.npz",
                **periodic_snapshot,
            )
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
        "--fsi-coupling-absolute-tolerance-mps",
        type=float,
        default=None,
        help=(
            "Optional absolute interface-velocity residual convergence floor "
            "in m/s (0 keeps the pure-relative gate)."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-accelerator",
        choices=("aitken", "iqn_ils"),
        default=None,
        help="Interface fixed-point accelerator for measured strong coupling.",
    )
    parser.add_argument(
        "--require-coupling-convergence",
        action="store_true",
        help=(
            "Fail closed and roll back the physical step when measured strong "
            "coupling does not meet its convergence tolerance."
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
        "--fsi-aitken-initial-relaxation",
        type=float,
        default=None,
        help="Initial Aitken relaxation factor for the strong-coupling loop.",
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
    if args.fsi_coupling_accelerator is not None:
        overrides["fsi_coupling_accelerator"] = str(
            args.fsi_coupling_accelerator
        )
    if args.fsi_aitken_initial_relaxation is not None:
        overrides["fsi_aitken_initial_relaxation"] = float(
            args.fsi_aitken_initial_relaxation
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
        require_coupling_convergence=bool(args.require_coupling_convergence),
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
