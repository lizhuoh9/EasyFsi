import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import taichi as ti

from simulation_core.pressure_interface import (
    PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS,
    PRESSURE_INTERFACE_COUPLING_SLOT_COUNT,
)
from simulation_core.runtime import TaichiRuntimeConfig, init_taichi

from .constants import (
    HIBM_NO_SLIP_NEAREST_FLUID_FALLBACK_RADIUS_CELLS,
    HIBM_OVERFLOW_SINGLETON_NO_SLIP_PROTECTION_RADIUS_CELLS,
    HIBM_OWNER_RELOCATION_WALK_STEPS,
    HIBM_PRESSURE_DISCONNECTED_SMALL_COMPONENT_THRESHOLD_CELLS,
    HIBM_PRESSURE_NEUMANN_NEAREST_FLUID_FALLBACK_RADIUS_CELLS,
    HIBM_PRESSURE_NEUMANN_ZERO_GRADIENT_TOLERANCE_PA_PER_M,
    HIBM_TINY_UNREACHED_COMPONENT_CLEANUP_THRESHOLD_CELLS,
    PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
    PRESSURE_NEUMANN_INVALID_REASON_BAD_MARKER,
    PRESSURE_NEUMANN_INVALID_REASON_NAMES,
    PRESSURE_NEUMANN_INVALID_REASON_NONPOSITIVE_VOLUME,
    PRESSURE_NEUMANN_INVALID_REASON_UNRECONSTRUCTABLE,
    STRESS_INVALID_REASON_BASE_PRESSURE_MISSING,
    STRESS_INVALID_REASON_NAMES,
    STRESS_INVALID_REASON_NONE,
    STRESS_INVALID_REASON_TWO_SIDED_PRESSURE_MISSING,
    STRESS_INVALID_REASON_VISCOUS_GRADIENT_MISSING,
)
from .modes import (
    FSI_COUPLING_MODE_CHOICES,
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
)
from .reports import (
    HibmMpmExternalForceClearReport,
    HibmMpmFluidStressSampleReport,
    HibmMpmIbBoundaryConditionReport,
    HibmMpmIbNodeSearchReport,
    HibmMpmMpmForceScatterReport,
    HibmMpmNoSlipResidualReport,
    HibmMpmPressureDisconnectedRegionReport,
    HibmMpmPressureNeumannGradientReport,
    HibmMpmPressureNeumannMatrixReport,
    HibmMpmSharpFluidToMpmLoadReport,
    HibmMpmSharpMpmStepReport,
    HibmMpmSharpNeoHookeanStepReport,
    HibmMpmSurfaceMarkerForceReport,
    HibmMpmSurfaceUpdateReport,
    HibmMpmVelocityDirichletBoundaryReport,
)

STRESS_PROBE_MODE_NAMES = {
    0: "none",
    1: "base_pressure",
    2: "two_sided_pressure_jump",
    3: "far_pressure_closure_inside_water",
    4: "far_pressure_closure_outside_water",
    5: "one_sided_inside_water",
    6: "one_sided_outside_water",
    7: "pressure_anchor_fallback",
}

STRESS_PROBE_LADDER_MODE_NAMES = {
    0: "none",
    1: "general_half_cell_ladder",
    2: "closure_extension_ladder",
    3: "two_sided_extension_ladder",
    4: "pressure_only_integer_ladder",
    5: "pressure_only_configured_ladder",
}
STRESS_PRESSURE_PROBE_LADDER_MODE_CURRENT_NORMAL_CELL = (
    "current_normal_cell_ladder"
)
STRESS_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER = "independent_ladder"
STRESS_PRESSURE_PAIR_POLICY_SYMMETRIC_CELL_PAIR = "symmetric_cell_pair"
STRESS_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR = (
    "baseline_anchored_cell_pair"
)
STRESS_PRESSURE_PAIR_POLICY_CODES = {
    STRESS_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER: 0,
    STRESS_PRESSURE_PAIR_POLICY_SYMMETRIC_CELL_PAIR: 1,
    STRESS_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR: 2,
}
STRESS_PRESSURE_PAIR_POLICY_NAMES = {
    0: STRESS_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER,
    1: STRESS_PRESSURE_PAIR_POLICY_SYMMETRIC_CELL_PAIR,
    2: STRESS_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR,
}
STRESS_ONE_SIDED_POLICY_NAMES = {
    0: "disabled",
    1: "legacy_single_region",
    2: "per_face_region",
}


def _debug_stage_progress(message: str) -> None:
    if os.environ.get("HIBM_DEBUG_STAGE_PROGRESS") == "1":
        print(f"[hibm-stage {time.perf_counter():.6f}] {message}", flush=True)


def _vector3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    try:
        vector = tuple(float(component) for component in value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly 3 components") from exc
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    if any(not math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain only finite values")
    return (vector[0], vector[1], vector[2])


def _normalize_vector3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = _vector3(value, name=name)
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must contain non-zero vectors")
    return tuple(component / norm for component in vector)


def _cell_vector3(value: Sequence[int], *, name: str) -> tuple[int, int, int]:
    try:
        vector = tuple(int(component) for component in value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly 3 components") from exc
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    if any(component < 0 for component in vector):
        raise ValueError(f"{name} must contain non-negative cell indices")
    return (vector[0], vector[1], vector[2])


def _positive_divergence_stencil_touch_mask(row_mask: np.ndarray) -> np.ndarray:
    touch = np.array(row_mask, dtype=bool, copy=True)
    touch[:-1, :, :] |= row_mask[1:, :, :]
    touch[:, :-1, :] |= row_mask[:, 1:, :]
    touch[:, :, :-1] |= row_mask[:, :, 1:]
    return touch


def _pressure_disconnected_component_distribution_kwargs(
    fluid: Any,
) -> dict[str, int]:
    return {
        "component_raw_count": int(
            getattr(fluid, "last_hibm_pressure_unreached_component_raw_count", 0)
        ),
        "largest_component_cell_count": int(
            getattr(
                fluid,
                "last_hibm_pressure_unreached_component_largest_cell_count",
                0,
            )
        ),
        "singleton_component_count": int(
            getattr(
                fluid,
                "last_hibm_pressure_unreached_component_singleton_count",
                0,
            )
        ),
        "small_component_threshold_cells": int(
            getattr(
                fluid,
                "last_hibm_pressure_unreached_component_small_threshold_cells",
                HIBM_PRESSURE_DISCONNECTED_SMALL_COMPONENT_THRESHOLD_CELLS,
            )
        ),
        "small_component_count": int(
            getattr(
                fluid,
                "last_hibm_pressure_unreached_component_small_count",
                0,
            )
        ),
        "small_component_cell_count": int(
            getattr(
                fluid,
                "last_hibm_pressure_unreached_component_small_cell_count",
                0,
            )
        ),
    }


def hibm_mpm_pressure_disconnected_region_report(
    fluid: Any,
    *,
    primary_region_id: int | None,
    secondary_region_id: int | None,
) -> HibmMpmPressureDisconnectedRegionReport:
    obstacle = fluid.obstacle.to_numpy()
    reachable = fluid.hibm_pressure_outlet_reachable.to_numpy()
    barrier = fluid.hibm_pressure_reachability_barrier.to_numpy()
    unreached = (obstacle == 0) & (barrier == 0) & (reachable == 0)
    cell_count = int(np.count_nonzero(unreached))
    if cell_count <= 0:
        return HibmMpmPressureDisconnectedRegionReport(
            component_count=int(
                getattr(fluid, "last_hibm_pressure_unreached_component_count", 0)
            ),
            **_pressure_disconnected_component_distribution_kwargs(fluid),
            component_overflow=bool(
                getattr(fluid, "last_hibm_pressure_unreached_component_overflow", False)
            ),
            component_labels_converged=bool(
                getattr(fluid, "last_hibm_pressure_component_labels_converged", True)
            ),
        )

    coords = np.argwhere(unreached)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    active_rows = fluid.velocity_dirichlet_boundary_active.to_numpy() != 0
    row_region = fluid.velocity_dirichlet_boundary_marker_region_id.to_numpy()
    primary_mask = np.zeros_like(active_rows, dtype=bool)
    if primary_region_id is not None and int(primary_region_id) >= 0:
        primary_mask = active_rows & (row_region == int(primary_region_id))
    secondary_mask = np.zeros_like(active_rows, dtype=bool)
    if secondary_region_id is not None and int(secondary_region_id) >= 0:
        secondary_mask = (
            active_rows
            & (row_region == int(secondary_region_id))
            & ~primary_mask
        )
    unassigned_mask = active_rows & (row_region < 0)
    other_mask = active_rows & ~(primary_mask | secondary_mask | unassigned_mask)

    primary_touch = _positive_divergence_stencil_touch_mask(primary_mask)
    secondary_touch = _positive_divergence_stencil_touch_mask(secondary_mask)
    other_touch = _positive_divergence_stencil_touch_mask(other_mask)
    unassigned_touch = _positive_divergence_stencil_touch_mask(unassigned_mask)
    return HibmMpmPressureDisconnectedRegionReport(
        cell_count=cell_count,
        component_count=int(
            getattr(fluid, "last_hibm_pressure_unreached_component_count", 0)
        ),
        **_pressure_disconnected_component_distribution_kwargs(fluid),
        component_overflow=bool(
            getattr(fluid, "last_hibm_pressure_unreached_component_overflow", False)
        ),
        component_labels_converged=bool(
            getattr(fluid, "last_hibm_pressure_component_labels_converged", True)
        ),
        min_i=int(mins[0]),
        min_j=int(mins[1]),
        min_k=int(mins[2]),
        max_i=int(maxs[0]),
        max_j=int(maxs[1]),
        max_k=int(maxs[2]),
        primary_region_stencil_cell_count=int(
            np.count_nonzero(unreached & primary_touch)
        ),
        secondary_region_stencil_cell_count=int(
            np.count_nonzero(unreached & secondary_touch)
        ),
        other_region_stencil_cell_count=int(np.count_nonzero(unreached & other_touch)),
        unassigned_region_stencil_cell_count=int(
            np.count_nonzero(unreached & unassigned_touch)
        ),
    )


@ti.data_oriented
class HibmMpmSurfaceMarkers:
    def __init__(
        self,
        marker_capacity: int,
        projection_triangle_capacity: int | None = None,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        init_taichi(runtime)
        if int(marker_capacity) <= 0:
            raise ValueError("marker_capacity must be positive")
        self.marker_capacity = int(marker_capacity)
        if projection_triangle_capacity is None:
            projection_triangle_capacity = self.marker_capacity
        if int(projection_triangle_capacity) <= 0:
            raise ValueError("projection_triangle_capacity must be positive")
        self.projection_triangle_capacity = int(projection_triangle_capacity)
        self.marker_count = 0
        self.projection_triangle_count = 0

        self.x_gamma_m = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_capacity)
        self.pressure_probe_origin_m = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=self.marker_capacity,
        )
        self.pressure_probe_origin_explicit = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self.v_gamma_mps = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_capacity)
        self.n_gamma = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_capacity)
        self.A_gamma_m2 = ti.field(dtype=ti.f32, shape=self.marker_capacity)
        self.region_id = ti.field(dtype=ti.i32, shape=self.marker_capacity)
        self.t_gamma_pa = ti.Vector.field(3, dtype=ti.f64, shape=self.marker_capacity)
        self.t_pressure_gamma_pa = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self.t_viscous_gamma_pa = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_pressure_valid = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_viscous_mode = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_base_pressure_found = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_inside_pressure_found = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_outside_pressure_found = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_marker_anchor_available = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_invalid_reason_code = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_base_pressure_pa = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_inside_pressure_pa = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_outside_pressure_pa = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_pressure_jump_pa = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_fluid_side_pressure_pa = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_reference_pressure_pa = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_inside_probe_rung = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_outside_probe_rung = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_inside_probe_distance_m = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_outside_probe_distance_m = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_inside_probe_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_outside_probe_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_inside_probe_grid_coordinate = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_outside_probe_grid_coordinate = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_inside_probe_fluid_weight = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_outside_probe_fluid_weight = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_inside_probe_multiplier = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_outside_probe_multiplier = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_inside_probe_ladder_mode = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_outside_probe_ladder_mode = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_probe_mode = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_pressure_pair_policy_code = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_pressure_pair_selected = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_pressure_pair_fallback_used = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_pressure_pair_inside_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_pressure_pair_outside_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_pressure_pair_cell_delta = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_pressure_pair_symmetry_residual_cells = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self.pressure_pair_anchor_active = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self.pressure_pair_anchor_inside_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self.pressure_pair_anchor_outside_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_pressure_pair_anchor_fallback_used = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_one_sided_policy_code = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_one_sided_region_id = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_one_sided_side_normal_sign = ti.field(
            dtype=ti.f64,
            shape=self.marker_capacity,
        )
        self._stress_one_sided_anchor_selected = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._stress_one_sided_anchor_fallback_used = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self.F_gamma_n = ti.Vector.field(3, dtype=ti.f64, shape=self.marker_capacity)
        self.projection_triangle_indices = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.projection_triangle_capacity,
        )
        # S2-A6: per-marker pressure-Neumann anchor cell. The pressure
        # matrix row assembly records, for every marker that received at
        # least one row, the (i, j, k) of the row-owning fluid cell (a
        # solve-participating, non-obstacle cell by construction);
        # (-1, -1, -1) means "no row / unset". The stress sampler may use
        # it as a last-resort water-side pressure probe for closure-region
        # markers whose normal walks are fully sealed by band obstacles.
        self.marker_pressure_anchor_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        # S2-A7: 1x1x1 stand-in bound to the sampler kernel's
        # node_anchor_cell template slot when the caller supplies no
        # node-level anchor field. Guarded by node_anchor_available == 0
        # in the kernel, it is never indexed at runtime; it only keeps
        # the template parameter bindable.
        self._node_anchor_cell_unset = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=(1, 1, 1),
        )
        # S2-A8'': 1x1x1 stand-in bound to the sampler kernel's
        # sampling_obstacle_field template slot when the caller supplies
        # no dedicated sampling view. Guarded by use_sampling_obstacle ==
        # 0 at every sample site (same anti-instantiation pattern as
        # _node_anchor_cell_unset), it is never indexed at runtime; it
        # only keeps the template parameter bindable, so omitting the
        # kwarg and passing None share one stable kernel instantiation.
        self._sampling_obstacle_unset = ti.field(
            dtype=ti.i32,
            shape=(1, 1, 1),
        )

        self.report_primary_force_n = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_secondary_force_n = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_total_force_n = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_primary_force_norm_sum_n = ti.field(dtype=ti.f64, shape=())
        self.report_secondary_force_norm_sum_n = ti.field(dtype=ti.f64, shape=())
        self.report_total_force_norm_sum_n = ti.field(dtype=ti.f64, shape=())
        self.report_primary_force_norm_max_n = ti.field(dtype=ti.f64, shape=())
        self.report_secondary_force_norm_max_n = ti.field(dtype=ti.f64, shape=())
        self.report_total_force_norm_max_n = ti.field(dtype=ti.f64, shape=())
        self.report_primary_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_secondary_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_total_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_primary_stress_valid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_primary_stress_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_secondary_stress_valid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_secondary_stress_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_valid_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_stress_invalid_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_stress_max_abs_traction_pa = ti.field(dtype=ti.f64, shape=())
        self.report_stress_two_sided_pressure_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_viscous_gradient_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_far_pressure_closed_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_far_pressure_closed_extended_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_far_pressure_anchor_closed_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_far_pressure_node_anchor_closed_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_closure_gradient_missing_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_far_pressure_outside_suppressed_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_two_sided_extended_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_one_sided_pressure_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_one_sided_extended_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_one_sided_gradient_missing_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_air_backed_seed_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_air_backed_seed_missed_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_air_backed_seed_fallback_cell_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_valid_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_no_slip_invalid_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_no_slip_max_residual_mps = ti.field(dtype=ti.f32, shape=())
        self.report_no_slip_sum_residual2_mps2 = ti.field(dtype=ti.f64, shape=())
        self.report_no_slip_direct_sample_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_normal_walk_sample_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_nearest_fluid_sample_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_zero_normal_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_no_fluid_sample_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_primary_region_valid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_primary_region_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_secondary_region_valid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_secondary_region_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_other_region_valid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_other_region_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_scatter_marker_force_n = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=(),
        )
        self.report_mpm_scatter_external_force_n = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=(),
        )
        self.report_mpm_scatter_active_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_scatter_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_scatter_active_particle_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_external_force_clear_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_external_force_clear_max_abs_n = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_feedback_updated_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_feedback_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_feedback_max_displacement_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_feedback_max_speed_mps = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_feedback_geometry_updated_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_feedback_geometry_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_feedback_max_normal_change = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_feedback_max_area_change_m2 = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_field_load_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_field_load_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_max_abs = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_projection_triangle_invalid_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        # Taichi zero-initializes fields, and (0, 0, 0) is a real cell
        # index: establish the unset sentinel before anyone can sample.
        self._reset_stress_diagnostics_kernel(int(self.marker_capacity))
        self.reset_pressure_anchor_cells()
        self.reset_pressure_pair_anchor_cells()
        self._reset_node_anchor_cell_unset_kernel()

    @ti.kernel
    def _reset_stress_diagnostics_kernel(self, marker_count: ti.i32):
        for marker in range(marker_count):
            self.t_pressure_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
            self.t_viscous_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
            self._stress_pressure_valid[marker] = 0
            self._stress_viscous_mode[marker] = 0
            self._stress_base_pressure_found[marker] = 0
            self._stress_inside_pressure_found[marker] = 0
            self._stress_outside_pressure_found[marker] = 0
            self._stress_marker_anchor_available[marker] = 0
            self._stress_invalid_reason_code[marker] = STRESS_INVALID_REASON_NONE
            self._stress_base_pressure_pa[marker] = 0.0
            self._stress_inside_pressure_pa[marker] = 0.0
            self._stress_outside_pressure_pa[marker] = 0.0
            self._stress_pressure_jump_pa[marker] = 0.0
            self._stress_fluid_side_pressure_pa[marker] = 0.0
            self._stress_reference_pressure_pa[marker] = 0.0
            self._stress_inside_probe_rung[marker] = -1
            self._stress_outside_probe_rung[marker] = -1
            self._stress_inside_probe_distance_m[marker] = -1.0
            self._stress_outside_probe_distance_m[marker] = -1.0
            self._stress_inside_probe_cell[marker] = ti.Vector([-1, -1, -1])
            self._stress_outside_probe_cell[marker] = ti.Vector([-1, -1, -1])
            self._stress_inside_probe_grid_coordinate[marker] = ti.Vector(
                [-1.0, -1.0, -1.0],
            )
            self._stress_outside_probe_grid_coordinate[marker] = ti.Vector(
                [-1.0, -1.0, -1.0],
            )
            self._stress_inside_probe_fluid_weight[marker] = 0.0
            self._stress_outside_probe_fluid_weight[marker] = 0.0
            self._stress_inside_probe_multiplier[marker] = 0.0
            self._stress_outside_probe_multiplier[marker] = 0.0
            self._stress_inside_probe_ladder_mode[marker] = 0
            self._stress_outside_probe_ladder_mode[marker] = 0
            self._stress_probe_mode[marker] = 0
            self._stress_pressure_pair_policy_code[marker] = 0
            self._stress_pressure_pair_selected[marker] = 0
            self._stress_pressure_pair_fallback_used[marker] = 0
            self._stress_pressure_pair_inside_cell[marker] = ti.Vector([-1, -1, -1])
            self._stress_pressure_pair_outside_cell[marker] = ti.Vector([-1, -1, -1])
            self._stress_pressure_pair_cell_delta[marker] = -1
            self._stress_pressure_pair_symmetry_residual_cells[marker] = -1.0
            self._stress_pressure_pair_anchor_fallback_used[marker] = 0
            self._stress_one_sided_policy_code[marker] = 0
            self._stress_one_sided_region_id[marker] = -1
            self._stress_one_sided_side_normal_sign[marker] = 0.0
            self._stress_one_sided_anchor_selected[marker] = 0
            self._stress_one_sided_anchor_fallback_used[marker] = 0

    @ti.kernel
    def _reset_pressure_anchor_cells_kernel(self):
        for marker in self.marker_pressure_anchor_cell:
            self.marker_pressure_anchor_cell[marker] = ti.Vector([-1, -1, -1])

    @ti.kernel
    def _reset_pressure_pair_anchor_cells_kernel(self):
        for marker in self.pressure_pair_anchor_active:
            self.pressure_pair_anchor_active[marker] = 0
            self.pressure_pair_anchor_inside_cell[marker] = ti.Vector([-1, -1, -1])
            self.pressure_pair_anchor_outside_cell[marker] = ti.Vector([-1, -1, -1])

    @ti.kernel
    def _reset_node_anchor_cell_unset_kernel(self):
        # S2-A7: keep the never-read stand-in on the unset sentinel too,
        # purely for self-consistency (it is shielded by
        # node_anchor_available == 0 at every read site).
        for node in ti.grouped(self._node_anchor_cell_unset):
            self._node_anchor_cell_unset[node] = ti.Vector([-1, -1, -1])

    def reset_pressure_anchor_cells(self) -> None:
        """Reset every marker's pressure-Neumann anchor cell to unset.

        Runs over the full capacity so markers that never receive a
        pressure matrix row (including slots beyond marker_count) keep the
        (-1, -1, -1) sentinel and stay on the invalid path in the sampler.
        """
        self._reset_pressure_anchor_cells_kernel()

    def reset_pressure_pair_anchor_cells(self) -> None:
        """Reset every marker's pressure-pair anchor cells to unset."""
        self._reset_pressure_pair_anchor_cells_kernel()

    def reset_stress_diagnostics(self, marker_count: int | None = None) -> None:
        count = int(self.marker_count if marker_count is None else marker_count)
        if count < 0 or count > self.marker_capacity:
            raise ValueError("marker_count out of range")
        self._reset_stress_diagnostics_kernel(count)

    def load_markers(
        self,
        *,
        positions_m: Sequence[Sequence[float]],
        velocities_mps: Sequence[Sequence[float]],
        normals: Sequence[Sequence[float]],
        areas_m2: Sequence[float],
        region_ids: Sequence[int],
        pressure_probe_origins_m: Sequence[Sequence[float]] | None = None,
    ) -> None:
        count = len(positions_m)
        if count > self.marker_capacity:
            raise ValueError("marker count exceeds marker_capacity")
        if not (
            len(velocities_mps)
            == len(normals)
            == len(areas_m2)
            == len(region_ids)
            == count
        ):
            raise ValueError("marker inputs must have matching lengths")
        if (
            pressure_probe_origins_m is not None
            and len(pressure_probe_origins_m) != count
        ):
            raise ValueError("pressure probe origin marker count must match markers")
        self.marker_count = int(count)
        self.projection_triangle_count = 0
        for marker in range(count):
            position = _vector3(positions_m[marker], name="positions_m")
            if pressure_probe_origins_m is None:
                pressure_probe_origin = position
                pressure_probe_origin_explicit = 0
            else:
                pressure_probe_origin = _vector3(
                    pressure_probe_origins_m[marker],
                    name="pressure_probe_origins_m",
                )
                pressure_probe_origin_explicit = 1
            velocity = _vector3(velocities_mps[marker], name="velocities_mps")
            normal = _normalize_vector3(normals[marker], name="normals")
            area = float(areas_m2[marker])
            if not math.isfinite(area) or area < 0.0:
                raise ValueError("areas_m2 must contain finite non-negative values")
            self.x_gamma_m[marker] = position
            self.pressure_probe_origin_m[marker] = pressure_probe_origin
            self.pressure_probe_origin_explicit[marker] = (
                pressure_probe_origin_explicit
            )
            self.v_gamma_mps[marker] = velocity
            self.n_gamma[marker] = normal
            self.A_gamma_m2[marker] = area
            self.region_id[marker] = int(region_ids[marker])
            self.t_gamma_pa[marker] = (0.0, 0.0, 0.0)
            self.F_gamma_n[marker] = (0.0, 0.0, 0.0)
        self.reset_pressure_pair_anchor_cells()
        self.reset_stress_diagnostics(count)

    def set_pressure_probe_origins_m(
        self,
        origins_m: Sequence[Sequence[float]],
    ) -> None:
        count = int(self.marker_count)
        if len(origins_m) != count:
            raise ValueError("pressure probe origin marker count must match markers")
        for marker in range(count):
            origin = _vector3(origins_m[marker], name="pressure_probe_origins_m")
            self.pressure_probe_origin_m[marker] = origin
            self.pressure_probe_origin_explicit[marker] = 1

    def set_pressure_pair_anchor_cells(
        self,
        *,
        inside_cells: Sequence[Sequence[int]],
        outside_cells: Sequence[Sequence[int]],
    ) -> None:
        count = int(self.marker_count)
        if len(inside_cells) != count or len(outside_cells) != count:
            raise ValueError("pressure pair anchor marker count must match markers")
        active = np.zeros((self.marker_capacity,), dtype=np.int32)
        inside = np.full((self.marker_capacity, 3), -1, dtype=np.int32)
        outside = np.full((self.marker_capacity, 3), -1, dtype=np.int32)
        for marker in range(count):
            inside[marker, :] = _cell_vector3(
                inside_cells[marker],
                name="inside_cells",
            )
            outside[marker, :] = _cell_vector3(
                outside_cells[marker],
                name="outside_cells",
            )
            active[marker] = 1
        self.pressure_pair_anchor_active.from_numpy(active)
        self.pressure_pair_anchor_inside_cell.from_numpy(inside)
        self.pressure_pair_anchor_outside_cell.from_numpy(outside)

    def set_projection_triangles(
        self,
        triangle_indices: Sequence[Sequence[int]],
    ) -> int:
        count = len(triangle_indices)
        if count > self.projection_triangle_capacity:
            raise ValueError(
                "projection triangle count exceeds projection_triangle_capacity"
            )
        marker_count = int(self.marker_count)
        for triangle_index, triangle in enumerate(triangle_indices):
            if len(triangle) != 3:
                raise ValueError("projection triangles must contain three indices")
            ia, ib, ic = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
            if (
                ia < 0
                or ib < 0
                or ic < 0
                or ia >= marker_count
                or ib >= marker_count
                or ic >= marker_count
            ):
                raise ValueError("projection triangle index out of marker range")
            self.projection_triangle_indices[triangle_index] = (ia, ib, ic)
        self.projection_triangle_count = int(count)
        return self.projection_triangle_count

    @ti.kernel
    def _load_projection_triangles_from_field_kernel(
        self,
        triangle_indices: ti.template(),
        triangle_count: ti.i32,
        marker_count: ti.i32,
    ):
        self.report_projection_triangle_invalid_count[None] = 0
        for triangle_index in range(triangle_count):
            triangle = triangle_indices[triangle_index]
            invalid = (
                triangle.x < 0
                or triangle.y < 0
                or triangle.z < 0
                or triangle.x >= marker_count
                or triangle.y >= marker_count
                or triangle.z >= marker_count
            )
            if invalid:
                self.report_projection_triangle_invalid_count[None] += 1
            self.projection_triangle_indices[triangle_index] = triangle

    def load_projection_triangles_from_field(
        self,
        triangle_indices,
        *,
        triangle_count: int,
    ) -> int:
        count = int(triangle_count)
        if count < 0:
            raise ValueError("triangle_count must be non-negative")
        if count > self.projection_triangle_capacity:
            raise ValueError(
                "projection triangle count exceeds projection_triangle_capacity"
            )
        self._load_projection_triangles_from_field_kernel(
            triangle_indices,
            count,
            int(self.marker_count),
        )
        invalid_count = int(self.report_projection_triangle_invalid_count[None])
        if invalid_count > 0:
            raise ValueError("projection triangle index out of marker range")
        self.projection_triangle_count = count
        return self.projection_triangle_count

    @ti.kernel
    def _load_markers_from_surface_fields_kernel(
        self,
        surface_position_m: ti.template(),
        surface_normal: ti.template(),
        surface_area_m2: ti.template(),
        surface_region_id: ti.template(),
        marker_count: ti.i32,
        initial_velocity_x_mps: ti.f32,
        initial_velocity_y_mps: ti.f32,
        initial_velocity_z_mps: ti.f32,
    ):
        self.report_surface_field_load_marker_count[None] = marker_count
        self.report_surface_field_load_invalid_marker_count[None] = 0
        initial_velocity = ti.Vector(
            [
                initial_velocity_x_mps,
                initial_velocity_y_mps,
                initial_velocity_z_mps,
            ]
        )
        for marker in range(marker_count):
            normal = surface_normal[marker]
            normal_norm = normal.norm()
            area = surface_area_m2[marker]
            if normal_norm <= 1.0e-12 or area < 0.0:
                self.report_surface_field_load_invalid_marker_count[None] += 1
            self.x_gamma_m[marker] = surface_position_m[marker]
            self.pressure_probe_origin_m[marker] = surface_position_m[marker]
            self.pressure_probe_origin_explicit[marker] = 0
            self.v_gamma_mps[marker] = initial_velocity
            self.n_gamma[marker] = normal / ti.max(normal_norm, 1.0e-12)
            self.A_gamma_m2[marker] = ti.max(area, 0.0)
            self.region_id[marker] = surface_region_id[marker]
            self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
            self.F_gamma_n[marker] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _load_markers_from_surface_velocity_fields_kernel(
        self,
        surface_position_m: ti.template(),
        surface_velocity_mps: ti.template(),
        surface_normal: ti.template(),
        surface_area_m2: ti.template(),
        surface_region_id: ti.template(),
        marker_count: ti.i32,
    ):
        self.report_surface_field_load_marker_count[None] = marker_count
        self.report_surface_field_load_invalid_marker_count[None] = 0
        for marker in range(marker_count):
            normal = surface_normal[marker]
            normal_norm = normal.norm()
            area = surface_area_m2[marker]
            if normal_norm <= 1.0e-12 or area < 0.0:
                self.report_surface_field_load_invalid_marker_count[None] += 1
            velocity = surface_velocity_mps[marker]
            self.x_gamma_m[marker] = surface_position_m[marker]
            self.pressure_probe_origin_m[marker] = surface_position_m[marker]
            self.pressure_probe_origin_explicit[marker] = 0
            self.v_gamma_mps[marker] = velocity
            self.n_gamma[marker] = normal / ti.max(normal_norm, 1.0e-12)
            self.A_gamma_m2[marker] = ti.max(area, 0.0)
            self.region_id[marker] = surface_region_id[marker]
            self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
            self.F_gamma_n[marker] = ti.Vector([0.0, 0.0, 0.0])

    def load_markers_from_surface_fields(
        self,
        surface_position_m,
        surface_normal,
        surface_area_m2,
        surface_region_id,
        *,
        marker_count: int,
        initial_velocity_mps: Sequence[float] = (0.0, 0.0, 0.0),
        surface_velocity_mps=None,
    ) -> int:
        count = int(marker_count)
        if count < 0:
            raise ValueError("marker_count must be non-negative")
        if count > self.marker_capacity:
            raise ValueError("marker_count exceeds marker_capacity")
        self.marker_count = count
        self.projection_triangle_count = 0
        if surface_velocity_mps is None:
            initial_velocity = _vector3(
                initial_velocity_mps,
                name="initial_velocity_mps",
            )
            self._load_markers_from_surface_fields_kernel(
                surface_position_m,
                surface_normal,
                surface_area_m2,
                surface_region_id,
                count,
                float(initial_velocity[0]),
                float(initial_velocity[1]),
                float(initial_velocity[2]),
            )
        else:
            self._load_markers_from_surface_velocity_fields_kernel(
                surface_position_m,
                surface_velocity_mps,
                surface_normal,
                surface_area_m2,
                surface_region_id,
                count,
            )
        self.reset_stress_diagnostics(count)
        invalid_count = int(self.report_surface_field_load_invalid_marker_count[None])
        if invalid_count > 0:
            raise ValueError(
                "surface fields must contain non-negative areas and non-zero normals"
            )
        return count

    @ti.kernel
    def _update_pressure_neumann_gradient_from_fluid_predictor_kernel(
        self,
        marker_pressure_neumann_gradient_pa_per_m_field: ti.template(),
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        marker_count: ti.i32,
        density_kgm3: ti.f32,
        dt_s: ti.f32,
        probe_distance_m: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_pressure_neumann_gradient_marker_count[None] = 0
        self.report_pressure_neumann_gradient_max_abs[None] = 0.0
        for marker in range(marker_count):
            sample_position = (
                self.x_gamma_m[marker] + self.n_gamma[marker] * probe_distance_m
            )
            grid_coordinate = self._grid_coordinate_from_fields(
                sample_position,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                nx,
                ny,
                nz,
            )
            predictor_velocity, fluid_weight = self._sample_fluid_velocity_trilinear(
                velocity_field,
                obstacle_field,
                grid_coordinate.x,
                grid_coordinate.y,
                grid_coordinate.z,
                nx,
                ny,
                nz,
            )
            normal_gradient = 0.0
            if fluid_weight > 1.0e-12:
                normal_gradient = (
                    density_kgm3
                    * (predictor_velocity - self.v_gamma_mps[marker]).dot(
                        self.n_gamma[marker]
                    )
                    / dt_s
                )
                self.report_pressure_neumann_gradient_marker_count[None] += 1
                ti.atomic_max(
                    self.report_pressure_neumann_gradient_max_abs[None],
                    ti.abs(normal_gradient),
                )
            marker_pressure_neumann_gradient_pa_per_m_field[marker] = normal_gradient

    def update_pressure_neumann_gradient_from_fluid_predictor(
        self,
        marker_pressure_neumann_gradient_pa_per_m_field,
        *,
        velocity_field,
        obstacle_field,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
        density_kgm3: float,
        dt_s: float,
        probe_distance_m: float,
    ) -> HibmMpmPressureNeumannGradientReport:
        density = float(density_kgm3)
        dt = float(dt_s)
        probe_distance = float(probe_distance_m)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("density_kgm3 must be finite and positive")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if not math.isfinite(probe_distance) or probe_distance <= 0.0:
            raise ValueError("probe_distance_m must be finite and positive")
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._update_pressure_neumann_gradient_from_fluid_predictor_kernel(
            marker_pressure_neumann_gradient_pa_per_m_field,
            velocity_field,
            obstacle_field,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(self.marker_count),
            density,
            dt,
            probe_distance,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        return HibmMpmPressureNeumannGradientReport(
            active_marker_count=int(
                self.report_pressure_neumann_gradient_marker_count[None]
            ),
            max_abs_gradient_pa_per_m=float(
                self.report_pressure_neumann_gradient_max_abs[None]
            ),
        )

    def set_marker_tractions_pa(
        self,
        tractions_pa: Sequence[Sequence[float]],
    ) -> None:
        if len(tractions_pa) != self.marker_count:
            raise ValueError("tractions_pa must match marker_count")
        self.reset_stress_diagnostics()
        for marker in range(self.marker_count):
            traction = _vector3(tractions_pa[marker], name="tractions_pa")
            self.t_gamma_pa[marker] = traction
            self.t_pressure_gamma_pa[marker] = traction
            self.t_viscous_gamma_pa[marker] = (0.0, 0.0, 0.0)
            self._stress_pressure_valid[marker] = 1
            self._stress_invalid_reason_code[marker] = STRESS_INVALID_REASON_NONE

    @ti.kernel
    def _compute_marker_forces_kernel(self, marker_count: ti.i32):
        for marker in range(marker_count):
            self.F_gamma_n[marker] = self.t_gamma_pa[marker] * self.A_gamma_m2[marker]

    def compute_marker_forces(self) -> None:
        self._compute_marker_forces_kernel(int(self.marker_count))

    @ti.func
    def _axis_grid_coordinate_device(
        self,
        value: ti.f32,
        faces: ti.template(),
        centers: ti.template(),
        count: ti.i32,
    ):
        coordinate = 0.0
        if value <= centers[0]:
            half_width = ti.max(centers[0] - faces[0], 1.0e-18)
            coordinate = -0.5 * (centers[0] - value) / half_width
        elif value >= centers[count - 1]:
            half_width = ti.max(faces[count] - centers[count - 1], 1.0e-18)
            coordinate = ti.cast(count - 1, ti.f32) + 0.5 * (
                value - centers[count - 1]
            ) / half_width
        else:
            lower = 0
            upper = count - 1
            while upper - lower > 1:
                middle = (lower + upper) // 2
                if value >= centers[middle]:
                    lower = middle
                else:
                    upper = middle
            upper = ti.min(lower + 1, count - 1)
            distance = ti.max(centers[upper] - centers[lower], 1.0e-18)
            coordinate = ti.cast(lower, ti.f32) + (value - centers[lower]) / distance
        return coordinate

    @ti.func
    def _grid_coordinate_from_fields(
        self,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        return ti.Vector(
            [
                self._axis_grid_coordinate_device(
                    position.x,
                    cell_face_x_m,
                    cell_center_x_m,
                    nx,
                ),
                self._axis_grid_coordinate_device(
                    position.y,
                    cell_face_y_m,
                    cell_center_y_m,
                    ny,
                ),
                self._axis_grid_coordinate_device(
                    position.z,
                    cell_face_z_m,
                    cell_center_z_m,
                    nz,
                ),
            ]
        )

    @ti.func
    def _sample_pressure_trilinear(
        self,
        pressure_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = ti.cast(0.0, ti.f64)
        fluid_weight = ti.cast(0.0, ti.f64)
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            weight = wx * wy * wz
            if obstacle_field[i0 + oi, j0 + oj, k0 + ok] == 0:
                value += weight * pressure_field[i0 + oi, j0 + oj, k0 + ok]
                fluid_weight += weight
        if fluid_weight > 1.0e-12:
            value /= fluid_weight
        return value, fluid_weight

    @ti.func
    def _sample_fluid_velocity_trilinear(
        self,
        velocity_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = ti.Vector([0.0, 0.0, 0.0])
        fluid_weight = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            weight = wx * wy * wz
            if obstacle_field[i0 + oi, j0 + oj, k0 + ok] == 0:
                value += weight * velocity_field[i0 + oi, j0 + oj, k0 + ok]
                fluid_weight += weight
        if fluid_weight > 1.0e-12:
            value /= fluid_weight
        return value, fluid_weight

    @ti.func
    def _sample_velocity_gradient(
        self,
        velocity_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
    ):
        ix0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        iy0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        iz0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1
        dx = ti.max(cell_center_x_m[ix1] - cell_center_x_m[ix0], 1.0e-18)
        dy = ti.max(cell_center_y_m[iy1] - cell_center_y_m[iy0], 1.0e-18)
        dz = ti.max(cell_center_z_m[iz1] - cell_center_z_m[iz0], 1.0e-18)
        vx0, wx0 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            ti.cast(ix0, ti.f32),
            gy,
            gz,
            nx,
            ny,
            nz,
        )
        vx1, wx1 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            ti.cast(ix1, ti.f32),
            gy,
            gz,
            nx,
            ny,
            nz,
        )
        vy0, wy0 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            gx,
            ti.cast(iy0, ti.f32),
            gz,
            nx,
            ny,
            nz,
        )
        vy1, wy1 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            gx,
            ti.cast(iy1, ti.f32),
            gz,
            nx,
            ny,
            nz,
        )
        vz0, wz0 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            gx,
            gy,
            ti.cast(iz0, ti.f32),
            nx,
            ny,
            nz,
        )
        vz1, wz1 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            gx,
            gy,
            ti.cast(iz1, ti.f32),
            nx,
            ny,
            nz,
        )
        dvdx = ti.Vector([0.0, 0.0, 0.0])
        dvdy = ti.Vector([0.0, 0.0, 0.0])
        dvdz = ti.Vector([0.0, 0.0, 0.0])
        gradient_valid = 1
        if wx0 > 1.0e-12 and wx1 > 1.0e-12:
            dvdx = (vx1 - vx0) / dx
        else:
            gradient_valid = 0
        if wy0 > 1.0e-12 and wy1 > 1.0e-12:
            dvdy = (vy1 - vy0) / dy
        else:
            gradient_valid = 0
        if wz0 > 1.0e-12 and wz1 > 1.0e-12:
            dvdz = (vz1 - vz0) / dz
        else:
            gradient_valid = 0
        return (
            ti.Matrix(
                [
                    [dvdx.x, dvdy.x, dvdz.x],
                    [dvdx.y, dvdy.y, dvdz.y],
                    [dvdx.z, dvdy.z, dvdz.z],
                ]
            ),
            gradient_valid,
        )

    @ti.func
    def _sample_pressure_trilinear_sampling_view(
        self,
        pressure_field: ti.template(),
        obstacle_field: ti.template(),
        sampling_obstacle_field: ti.template(),
        use_sampling_obstacle: ti.i32,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        # S2-A8'' view switch: with the runtime gate off the else branch
        # executes the exact original expression tree against the
        # projection obstacle view (bit for bit the status quo); with the
        # gate on, the same masked trilinear body reads the dedicated
        # sampling view instead, so the classified row-cloud envelope
        # stays dry while back-filled converted sealed water is
        # samplable. The gate is a uniform runtime i32, so the kernel is
        # not re-instantiated per call - only per bound field pair (the
        # A6/A7 stand-in pattern keeps the no-view binding stable).
        pressure = ti.cast(0.0, ti.f64)
        pressure_weight = ti.cast(0.0, ti.f64)
        if use_sampling_obstacle != 0:
            pressure, pressure_weight = self._sample_pressure_trilinear(
                pressure_field,
                sampling_obstacle_field,
                gx,
                gy,
                gz,
                nx,
                ny,
                nz,
            )
        else:
            pressure, pressure_weight = self._sample_pressure_trilinear(
                pressure_field,
                obstacle_field,
                gx,
                gy,
                gz,
                nx,
                ny,
                nz,
            )
        return pressure, pressure_weight

    @ti.func
    def _sample_velocity_gradient_sampling_view(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        sampling_obstacle_field: ti.template(),
        use_sampling_obstacle: ti.i32,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
    ):
        # S2-A8'' view switch for the one-sided viscous gradient stencil;
        # same gate semantics as _sample_pressure_trilinear_sampling_view.
        gradient = ti.Matrix.zero(ti.f32, 3, 3)
        gradient_valid = 1
        if use_sampling_obstacle != 0:
            gradient, gradient_valid = self._sample_velocity_gradient(
                velocity_field,
                sampling_obstacle_field,
                gx,
                gy,
                gz,
                nx,
                ny,
                nz,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
            )
        else:
            gradient, gradient_valid = self._sample_velocity_gradient(
                velocity_field,
                obstacle_field,
                gx,
                gy,
                gz,
                nx,
                ny,
                nz,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
            )
        return gradient, gradient_valid

    @ti.kernel
    def _sample_fluid_stress_to_marker_tractions_kernel(
        self,
        velocity_field: ti.template(),
        pressure_field: ti.template(),
        obstacle_field: ti.template(),
        sampling_obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        marker_pressure_anchor_cell: ti.template(),
        node_anchor_cell: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        viscosity_pa_s: ti.template(),
        two_sided_pressure: ti.i32,
        far_pressure_region_id: ti.i32,
        far_pressure_pa: ti.f32,
        far_pressure_side_normal_sign: ti.f32,
        far_pressure_inside_probe_max_multiplier: ti.f32,
        two_sided_probe_max_multiplier: ti.f32,
        one_sided_pressure_region_id: ti.i32,
        one_sided_reference_pressure_pa: ti.f32,
        one_sided_probe_max_multiplier: ti.f32,
        use_pressure_anchor_fallback: ti.i32,
        node_anchor_available: ti.i32,
        use_sampling_obstacle: ti.i32,
    ):
        self.report_stress_valid_marker_count[None] = 0
        self.report_stress_invalid_marker_count[None] = 0
        self.report_stress_max_abs_traction_pa[None] = 0.0
        self.report_stress_two_sided_pressure_marker_count[None] = 0
        self.report_stress_viscous_gradient_invalid_marker_count[None] = 0
        self.report_stress_far_pressure_closed_marker_count[None] = 0
        self.report_stress_far_pressure_closed_extended_marker_count[None] = 0
        self.report_stress_far_pressure_anchor_closed_marker_count[None] = 0
        self.report_stress_far_pressure_node_anchor_closed_marker_count[None] = 0
        self.report_stress_closure_gradient_missing_marker_count[None] = 0
        self.report_stress_far_pressure_outside_suppressed_marker_count[None] = 0
        self.report_stress_two_sided_extended_marker_count[None] = 0
        self.report_stress_one_sided_pressure_marker_count[None] = 0
        self.report_stress_one_sided_extended_marker_count[None] = 0
        self.report_stress_one_sided_gradient_missing_marker_count[None] = 0
        for marker in range(marker_count):
            position = self.x_gamma_m[marker]
            normal = self.n_gamma[marker]
            probe_origin = position
            if self.pressure_probe_origin_explicit[marker] != 0:
                probe_origin = self.pressure_probe_origin_m[marker]
            grid_coordinate = self._grid_coordinate_from_fields(
                position,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                nx,
                ny,
                nz,
            )
            i_near = ti.min(ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0), nx - 1)
            j_near = ti.min(ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0), ny - 1)
            k_near = ti.min(ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0), nz - 1)
            pressure, pressure_weight = self._sample_pressure_trilinear_sampling_view(
                pressure_field,
                obstacle_field,
                sampling_obstacle_field,
                use_sampling_obstacle,
                grid_coordinate.x,
                grid_coordinate.y,
                grid_coordinate.z,
                nx,
                ny,
                nz,
            )
            pressure_traction = -pressure * normal
            pressure_sample_valid = pressure_weight > 1.0e-12
            diagnostic_base_pressure_found = 0
            if pressure_sample_valid:
                diagnostic_base_pressure_found = 1
            diagnostic_base_pressure_pa = ti.cast(pressure, ti.f64)
            diagnostic_inside_pressure_pa = ti.cast(0.0, ti.f64)
            diagnostic_outside_pressure_pa = ti.cast(0.0, ti.f64)
            diagnostic_pressure_jump_pa = ti.cast(0.0, ti.f64)
            diagnostic_fluid_side_pressure_pa = ti.cast(0.0, ti.f64)
            diagnostic_reference_pressure_pa = ti.cast(0.0, ti.f64)
            diagnostic_inside_probe_rung = -1
            diagnostic_outside_probe_rung = -1
            diagnostic_inside_probe_distance_m = ti.cast(-1.0, ti.f64)
            diagnostic_outside_probe_distance_m = ti.cast(-1.0, ti.f64)
            diagnostic_inside_probe_cell = ti.Vector([-1, -1, -1])
            diagnostic_outside_probe_cell = ti.Vector([-1, -1, -1])
            diagnostic_inside_probe_grid_coordinate = ti.Vector(
                [
                    ti.cast(-1.0, ti.f64),
                    ti.cast(-1.0, ti.f64),
                    ti.cast(-1.0, ti.f64),
                ]
            )
            diagnostic_outside_probe_grid_coordinate = ti.Vector(
                [
                    ti.cast(-1.0, ti.f64),
                    ti.cast(-1.0, ti.f64),
                    ti.cast(-1.0, ti.f64),
                ]
            )
            diagnostic_inside_probe_fluid_weight = ti.cast(0.0, ti.f64)
            diagnostic_outside_probe_fluid_weight = ti.cast(0.0, ti.f64)
            diagnostic_inside_probe_multiplier = ti.cast(0.0, ti.f64)
            diagnostic_outside_probe_multiplier = ti.cast(0.0, ti.f64)
            diagnostic_inside_probe_ladder_mode = 0
            diagnostic_outside_probe_ladder_mode = 0
            diagnostic_probe_mode = 0
            if pressure_sample_valid:
                diagnostic_fluid_side_pressure_pa = diagnostic_base_pressure_pa
                diagnostic_probe_mode = 1
            diagnostic_inside_pressure_found = 0
            diagnostic_outside_pressure_found = 0
            diagnostic_marker_anchor_available = 0
            if marker_pressure_anchor_cell[marker].x >= 0:
                diagnostic_marker_anchor_available = 1
            gradient = ti.Matrix.zero(ti.f32, 3, 3)
            gradient_valid = 1
            viscous_mode = 1
            if two_sided_pressure != 0:
                viscous_mode = 0
                normal_spacing_inv = (
                    ti.abs(normal.x) / cell_width_x_m[i_near]
                    + ti.abs(normal.y) / cell_width_y_m[j_near]
                    + ti.abs(normal.z) / cell_width_z_m[k_near]
                )
                probe_distance_m = 1.0 / ti.max(normal_spacing_inv, 1.0e-12)
                outside_pressure = ti.cast(0.0, ti.f64)
                inside_pressure = ti.cast(0.0, ti.f64)
                outside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                inside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                # S2-A4: per-side "found" is split into pressure-found and
                # gradient-found. In far-pressure closure regions narrow
                # obstacle slabs can leave only 1-3 cell wide water gaps
                # behind membranes at production grids: trilinear
                # pressure is samplable inside such a gap, but the one-sided
                # viscous gradient stencil (complete fluid neighbor pairs on
                # all three axes) almost never is, and the merged acceptance
                # gate rejected the whole candidate - silently dropping the
                # O(1e3 Pa) pressure drive over an O(0.1 Pa) viscous term.
                # Branch decisions (two-sided / closure / mirrored closure)
                # key on the pressure flags; an unfound gradient side simply
                # keeps its zero matrix. Outside closure regions both flags
                # are only ever set together, preserving the original merged
                # gate bit for bit.
                outside_pressure_found = 0
                outside_gradient_found = 0
                inside_pressure_found = 0
                inside_gradient_found = 0
                inside_found_extended = 0
                two_sided_found_extended = 0
                closure_region_marker = 0
                if (
                    far_pressure_region_id != -1
                    and self.region_id[marker] == far_pressure_region_id
                ):
                    closure_region_marker = 1
                one_sided_region_marker = 0
                if (
                    one_sided_pressure_region_id != -1
                    and self.region_id[marker] == one_sided_pressure_region_id
                ):
                    one_sided_region_marker = 1
                # S2-A5: runtime range(5) instead of ti.static. The body only
                # consumes probe_index through ti.cast(..., ti.f32) (exact for
                # 0..4), the per-thread serial loop runs the iterations in the
                # same order with the same carried flag state, so every float
                # is produced by the identical expression tree in the
                # identical order - bitwise-preserving - while the JIT stops
                # cloning the inlined trilinear/gradient sampling bodies five
                # times per side.
                for probe_index in range(5):
                    probe_distance = probe_distance_m * (
                        1.0 + 0.5 * ti.cast(probe_index, ti.f32)
                    )
                    # Walk on while the side pressure is missing; in closure
                    # regions additionally walk on to fill a still-missing
                    # gradient from a farther candidate (the gradient update
                    # below never overwrites an already-found pressure).
                    if outside_pressure_found == 0 or (
                        closure_region_marker == 1 and outside_gradient_found == 0
                    ):
                        outside_position = probe_origin + normal * probe_distance
                        outside_coordinate = self._grid_coordinate_from_fields(
                            outside_position,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            nx,
                            ny,
                            nz,
                        )
                        sample_pressure, sample_weight = self._sample_pressure_trilinear_sampling_view(
                            pressure_field,
                            obstacle_field,
                            sampling_obstacle_field,
                            use_sampling_obstacle,
                            outside_coordinate.x,
                            outside_coordinate.y,
                            outside_coordinate.z,
                            nx,
                            ny,
                            nz,
                        )
                        sample_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                        sample_gradient_valid = 1
                        if ti.static(viscosity_pa_s > 0.0):
                            sample_gradient, sample_gradient_valid = (
                                self._sample_velocity_gradient_sampling_view(
                                    velocity_field,
                                    obstacle_field,
                                    sampling_obstacle_field,
                                    use_sampling_obstacle,
                                    outside_coordinate.x,
                                    outside_coordinate.y,
                                    outside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                )
                            )
                        # S2-A5 single-copy acceptance: the accept decision is
                        # split from the stores so each store exists once in
                        # the IR. Closure regions keep the S2-A4 decoupled
                        # guards (pressure needs trilinear fluid weight; the
                        # gradient additionally needs its complete one-sided
                        # stencil, so a farther candidate can still supply
                        # it). Outside closure regions the original merged
                        # gate applies atomically: pressure and gradient are
                        # accepted together from the same candidate or not at
                        # all, so a nearer pressure can never pair with a
                        # farther gradient - bit for bit the original gate.
                        candidate_merged_ok = 0
                        if sample_weight > 1.0e-12 and sample_gradient_valid == 1:
                            candidate_merged_ok = 1
                        accept_pressure = 0
                        accept_gradient = 0
                        if closure_region_marker == 1:
                            if sample_weight > 1.0e-12 and outside_pressure_found == 0:
                                accept_pressure = 1
                            if candidate_merged_ok == 1 and outside_gradient_found == 0:
                                accept_gradient = 1
                        else:
                            if candidate_merged_ok == 1:
                                accept_pressure = 1
                                accept_gradient = 1
                        if accept_pressure == 1:
                            outside_pressure = sample_pressure
                            outside_pressure_found = 1
                            diagnostic_outside_pressure_pa = ti.cast(
                                sample_pressure,
                                ti.f64,
                            )
                            diagnostic_outside_probe_rung = probe_index
                            diagnostic_outside_probe_distance_m = ti.cast(
                                probe_distance,
                                ti.f64,
                            )
                            diagnostic_outside_probe_cell = ti.Vector(
                                [
                                    ti.min(
                                        ti.max(
                                            ti.floor(outside_coordinate.x + 0.5, ti.i32),
                                            0,
                                        ),
                                        nx - 1,
                                    ),
                                    ti.min(
                                        ti.max(
                                            ti.floor(outside_coordinate.y + 0.5, ti.i32),
                                            0,
                                        ),
                                        ny - 1,
                                    ),
                                    ti.min(
                                        ti.max(
                                            ti.floor(outside_coordinate.z + 0.5, ti.i32),
                                            0,
                                        ),
                                        nz - 1,
                                    ),
                                ]
                            )
                            diagnostic_outside_probe_grid_coordinate = ti.Vector(
                                [
                                    ti.cast(outside_coordinate.x, ti.f64),
                                    ti.cast(outside_coordinate.y, ti.f64),
                                    ti.cast(outside_coordinate.z, ti.f64),
                                ]
                            )
                            diagnostic_outside_probe_fluid_weight = ti.cast(
                                sample_weight,
                                ti.f64,
                            )
                            diagnostic_outside_probe_multiplier = ti.cast(
                                1.0 + 0.5 * ti.cast(probe_index, ti.f32),
                                ti.f64,
                            )
                            diagnostic_outside_probe_ladder_mode = 1
                        if accept_gradient == 1:
                            outside_gradient = sample_gradient
                            outside_gradient_found = 1
                    if inside_pressure_found == 0 or (
                        closure_region_marker == 1 and inside_gradient_found == 0
                    ):
                        inside_position = probe_origin - normal * probe_distance
                        inside_coordinate = self._grid_coordinate_from_fields(
                            inside_position,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            nx,
                            ny,
                            nz,
                        )
                        sample_pressure, sample_weight = self._sample_pressure_trilinear_sampling_view(
                            pressure_field,
                            obstacle_field,
                            sampling_obstacle_field,
                            use_sampling_obstacle,
                            inside_coordinate.x,
                            inside_coordinate.y,
                            inside_coordinate.z,
                            nx,
                            ny,
                            nz,
                        )
                        sample_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                        sample_gradient_valid = 1
                        if ti.static(viscosity_pa_s > 0.0):
                            sample_gradient, sample_gradient_valid = (
                                self._sample_velocity_gradient_sampling_view(
                                    velocity_field,
                                    obstacle_field,
                                    sampling_obstacle_field,
                                    use_sampling_obstacle,
                                    inside_coordinate.x,
                                    inside_coordinate.y,
                                    inside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                )
                            )
                        # S2-A5 single-copy acceptance, symmetric to the
                        # outside (+n) walk above.
                        candidate_merged_ok = 0
                        if sample_weight > 1.0e-12 and sample_gradient_valid == 1:
                            candidate_merged_ok = 1
                        accept_pressure = 0
                        accept_gradient = 0
                        if closure_region_marker == 1:
                            if sample_weight > 1.0e-12 and inside_pressure_found == 0:
                                accept_pressure = 1
                            if candidate_merged_ok == 1 and inside_gradient_found == 0:
                                accept_gradient = 1
                        else:
                            if candidate_merged_ok == 1:
                                accept_pressure = 1
                                accept_gradient = 1
                        if accept_pressure == 1:
                            inside_pressure = sample_pressure
                            inside_pressure_found = 1
                            diagnostic_inside_pressure_pa = ti.cast(
                                sample_pressure,
                                ti.f64,
                            )
                            diagnostic_inside_probe_rung = probe_index
                            diagnostic_inside_probe_distance_m = ti.cast(
                                probe_distance,
                                ti.f64,
                            )
                            diagnostic_inside_probe_cell = ti.Vector(
                                [
                                    ti.min(
                                        ti.max(
                                            ti.floor(inside_coordinate.x + 0.5, ti.i32),
                                            0,
                                        ),
                                        nx - 1,
                                    ),
                                    ti.min(
                                        ti.max(
                                            ti.floor(inside_coordinate.y + 0.5, ti.i32),
                                            0,
                                        ),
                                        ny - 1,
                                    ),
                                    ti.min(
                                        ti.max(
                                            ti.floor(inside_coordinate.z + 0.5, ti.i32),
                                            0,
                                        ),
                                        nz - 1,
                                    ),
                                ]
                            )
                            diagnostic_inside_probe_grid_coordinate = ti.Vector(
                                [
                                    ti.cast(inside_coordinate.x, ti.f64),
                                    ti.cast(inside_coordinate.y, ti.f64),
                                    ti.cast(inside_coordinate.z, ti.f64),
                                ]
                            )
                            diagnostic_inside_probe_fluid_weight = ti.cast(
                                sample_weight,
                                ti.f64,
                            )
                            diagnostic_inside_probe_multiplier = ti.cast(
                                1.0 + 0.5 * ti.cast(probe_index, ti.f32),
                                ti.f64,
                            )
                            diagnostic_inside_probe_ladder_mode = 1
                        if accept_gradient == 1:
                            inside_gradient = sample_gradient
                            inside_gradient_found = 1
                # S2-A3 extended inside (-n) walk: opt-in, far-pressure
                # closure regions only. It runs strictly after the standard
                # ladder above failed to find inside water within 3x, and
                # extends the reach uniformly to the requested multiplier.
                # The outside (+n) walk is intentionally never extended.
                # The outside_pressure_found == 0 gate keeps
                # mirrored-orientation markers (water on +n, structurally dry
                # -n) on the mirrored closure branch: without it the extended
                # walk could tunnel through a thin dry band to unrelated deep
                # water and silently replace the known far pressure with a
                # spurious two-sided sample, dropping the drive on exactly
                # the markers the closure exists for. S2-A4: the gate keys on
                # the pressure flag (not the gradient flag) so a mirrored
                # marker whose outside water has a broken gradient stencil
                # stays mirror-protected as well.
                if (
                    far_pressure_region_id != -1
                    and self.region_id[marker] == far_pressure_region_id
                    and inside_pressure_found == 0
                    and outside_pressure_found == 0
                    and far_pressure_inside_probe_max_multiplier > 3.0
                ):
                    # S2-A5: runtime loop, same bitwise-preservation argument
                    # as the standard walk above.
                    for probe_index in range(5):
                        probe_distance = probe_distance_m * (
                            3.0
                            + (far_pressure_inside_probe_max_multiplier - 3.0)
                            * (ti.cast(probe_index, ti.f32) + 1.0)
                            / 5.0
                        )
                        # The extension only runs for closure-region markers
                        # (see the entry gate above), so the S2-A4 decoupled
                        # acceptance applies unconditionally here.
                        if inside_pressure_found == 0 or inside_gradient_found == 0:
                            inside_position = probe_origin - normal * probe_distance
                            inside_coordinate = self._grid_coordinate_from_fields(
                                inside_position,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            sample_pressure, sample_weight = self._sample_pressure_trilinear_sampling_view(
                                pressure_field,
                                obstacle_field,
                                sampling_obstacle_field,
                                use_sampling_obstacle,
                                inside_coordinate.x,
                                inside_coordinate.y,
                                inside_coordinate.z,
                                nx,
                                ny,
                                nz,
                            )
                            sample_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                            sample_gradient_valid = 1
                            if ti.static(viscosity_pa_s > 0.0):
                                sample_gradient, sample_gradient_valid = (
                                    self._sample_velocity_gradient_sampling_view(
                                        velocity_field,
                                        obstacle_field,
                                        sampling_obstacle_field,
                                        use_sampling_obstacle,
                                        inside_coordinate.x,
                                        inside_coordinate.y,
                                        inside_coordinate.z,
                                        nx,
                                        ny,
                                        nz,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                    )
                                )
                            if sample_weight > 1.0e-12 and inside_pressure_found == 0:
                                inside_pressure = sample_pressure
                                inside_pressure_found = 1
                                inside_found_extended = 1
                                diagnostic_inside_pressure_pa = ti.cast(
                                    sample_pressure,
                                    ti.f64,
                                )
                                diagnostic_inside_probe_rung = 5 + probe_index
                                diagnostic_inside_probe_distance_m = ti.cast(
                                    probe_distance,
                                    ti.f64,
                                )
                                diagnostic_inside_probe_cell = ti.Vector(
                                    [
                                        ti.min(
                                            ti.max(
                                                ti.floor(
                                                    inside_coordinate.x + 0.5,
                                                    ti.i32,
                                                ),
                                                0,
                                            ),
                                            nx - 1,
                                        ),
                                        ti.min(
                                            ti.max(
                                                ti.floor(
                                                    inside_coordinate.y + 0.5,
                                                    ti.i32,
                                                ),
                                                0,
                                            ),
                                            ny - 1,
                                        ),
                                        ti.min(
                                            ti.max(
                                                ti.floor(
                                                    inside_coordinate.z + 0.5,
                                                    ti.i32,
                                                ),
                                                0,
                                            ),
                                            nz - 1,
                                        ),
                                    ]
                                )
                                diagnostic_inside_probe_grid_coordinate = ti.Vector(
                                    [
                                        ti.cast(inside_coordinate.x, ti.f64),
                                        ti.cast(inside_coordinate.y, ti.f64),
                                        ti.cast(inside_coordinate.z, ti.f64),
                                    ]
                                )
                                diagnostic_inside_probe_fluid_weight = ti.cast(
                                    sample_weight,
                                    ti.f64,
                                )
                                diagnostic_inside_probe_multiplier = ti.cast(
                                    3.0
                                    + (
                                        far_pressure_inside_probe_max_multiplier
                                        - 3.0
                                    )
                                    * (ti.cast(probe_index, ti.f32) + 1.0)
                                    / 5.0,
                                    ti.f64,
                                )
                                diagnostic_inside_probe_ladder_mode = 2
                            if (
                                sample_weight > 1.0e-12
                                and sample_gradient_valid == 1
                                and inside_gradient_found == 0
                            ):
                                inside_gradient = sample_gradient
                                inside_gradient_found = 1
                # S2-A10 two-sided extended walk: opt-in, NON-closure
                # markers only - closure regions keep their dedicated
                # extension above and their branch priority below. The
                # S2-A8'' dedicated sampling view can starve genuinely thin
                # features that sit entirely inside their own row-cloud
                # envelope, so one or both standard walks (max 3.0x) can run
                # dry and leave a physically two-sided marker invalid. When
                # armed (two_sided_probe_max_multiplier > 3.0), re-walk each
                # still-missing side out to the requested multiplier with the
                # same
                # 5-candidate ladder spacing, the same sampling view and
                # the S2-A4 decoupled acceptance the closure extension
                # uses. H1-type crossing guard, applied per side: an
                # extension rung whose nearest cell is a projection-view
                # obstacle has crossed foreign solid - water beyond it
                # belongs to the opposite side / another compartment and
                # is never accepted (the closure extension's
                # outside_found gate carries the same do-not-tunnel
                # semantics; here both sides walk, so the guard is per
                # side). The marker's own sub-envelope feature (<= 3.0x,
                # e.g. the fin's own thickness) can never set the flag:
                # only extension rungs are tested. Both sides share one
                # fused runtime rung loop (S2-A5 single-copy style: the
                # trilinear / gradient sampling bodies are inlined once,
                # not once per side; even rung indices walk +n, odd walk
                # -n, preserving the standard walk's outside-then-inside
                # serial order and its carried-flag determinism).
                extended_probe_max_multiplier = two_sided_probe_max_multiplier
                if (
                    one_sided_region_marker == 1
                    and one_sided_probe_max_multiplier
                    > extended_probe_max_multiplier
                ):
                    extended_probe_max_multiplier = one_sided_probe_max_multiplier
                if (
                    closure_region_marker == 0
                    and (inside_pressure_found == 0 or outside_pressure_found == 0)
                    and extended_probe_max_multiplier > 3.0
                ):
                    marker_near_is_obstacle = 0
                    if obstacle_field[i_near, j_near, k_near] != 0:
                        marker_near_is_obstacle = 1
                    outside_crossed_solid = 0
                    inside_crossed_solid = 0
                    for extension_index in range(10):
                        rung_distance = probe_distance_m * (
                            3.0
                            + (extended_probe_max_multiplier - 3.0)
                            * (ti.cast(extension_index // 2, ti.f32) + 1.0)
                            / 5.0
                        )
                        side_is_inside = extension_index % 2
                        side_sign = 1.0
                        side_crossed = outside_crossed_solid
                        side_pressure_found = outside_pressure_found
                        side_gradient_found = outside_gradient_found
                        if side_is_inside == 1:
                            side_sign = -1.0
                            side_crossed = inside_crossed_solid
                            side_pressure_found = inside_pressure_found
                            side_gradient_found = inside_gradient_found
                        if side_crossed == 0 and (
                            side_pressure_found == 0 or side_gradient_found == 0
                        ):
                            extension_position = probe_origin + normal * (
                                side_sign * rung_distance
                            )
                            extension_coordinate = self._grid_coordinate_from_fields(
                                extension_position,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            near_extension_i = ti.min(
                                ti.max(
                                    ti.floor(extension_coordinate.x + 0.5, ti.i32),
                                    0,
                                ),
                                nx - 1,
                            )
                            near_extension_j = ti.min(
                                ti.max(
                                    ti.floor(extension_coordinate.y + 0.5, ti.i32),
                                    0,
                                ),
                                ny - 1,
                            )
                            near_extension_k = ti.min(
                                ti.max(
                                    ti.floor(extension_coordinate.z + 0.5, ti.i32),
                                    0,
                                ),
                                nz - 1,
                            )
                            if (
                                obstacle_field[
                                    near_extension_i,
                                    near_extension_j,
                                    near_extension_k,
                                ]
                                != 0
                            ):
                                if (
                                    marker_near_is_obstacle == 0
                                    and use_sampling_obstacle == 0
                                ):
                                    if side_is_inside == 1:
                                        inside_crossed_solid = 1
                                    else:
                                        outside_crossed_solid = 1
                            else:
                                sample_pressure, sample_weight = self._sample_pressure_trilinear_sampling_view(
                                    pressure_field,
                                    obstacle_field,
                                    sampling_obstacle_field,
                                    use_sampling_obstacle,
                                    extension_coordinate.x,
                                    extension_coordinate.y,
                                    extension_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                )
                                sample_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                                sample_gradient_valid = 1
                                if ti.static(viscosity_pa_s > 0.0):
                                    sample_gradient, sample_gradient_valid = (
                                        self._sample_velocity_gradient_sampling_view(
                                            velocity_field,
                                            obstacle_field,
                                            sampling_obstacle_field,
                                            use_sampling_obstacle,
                                            extension_coordinate.x,
                                            extension_coordinate.y,
                                            extension_coordinate.z,
                                            nx,
                                            ny,
                                            nz,
                                            cell_center_x_m,
                                            cell_center_y_m,
                                            cell_center_z_m,
                                        )
                                    )
                                if sample_weight > 1.0e-12 and side_pressure_found == 0:
                                    if side_is_inside == 1:
                                        inside_pressure = sample_pressure
                                        inside_pressure_found = 1
                                        diagnostic_inside_pressure_pa = ti.cast(
                                            sample_pressure,
                                            ti.f64,
                                        )
                                        diagnostic_inside_probe_rung = (
                                            10 + extension_index // 2
                                        )
                                        diagnostic_inside_probe_distance_m = ti.cast(
                                            rung_distance,
                                            ti.f64,
                                        )
                                        diagnostic_inside_probe_cell = ti.Vector(
                                            [
                                                near_extension_i,
                                                near_extension_j,
                                                near_extension_k,
                                            ]
                                        )
                                        diagnostic_inside_probe_grid_coordinate = (
                                            ti.Vector(
                                                [
                                                    ti.cast(
                                                        extension_coordinate.x,
                                                        ti.f64,
                                                    ),
                                                    ti.cast(
                                                        extension_coordinate.y,
                                                        ti.f64,
                                                    ),
                                                    ti.cast(
                                                        extension_coordinate.z,
                                                        ti.f64,
                                                    ),
                                                ]
                                            )
                                        )
                                        diagnostic_inside_probe_fluid_weight = (
                                            ti.cast(sample_weight, ti.f64)
                                        )
                                        diagnostic_inside_probe_multiplier = ti.cast(
                                            rung_distance / probe_distance_m,
                                            ti.f64,
                                        )
                                        diagnostic_inside_probe_ladder_mode = 3
                                    else:
                                        outside_pressure = sample_pressure
                                        outside_pressure_found = 1
                                        diagnostic_outside_pressure_pa = ti.cast(
                                            sample_pressure,
                                            ti.f64,
                                        )
                                        diagnostic_outside_probe_rung = (
                                            10 + extension_index // 2
                                        )
                                        diagnostic_outside_probe_distance_m = ti.cast(
                                            rung_distance,
                                            ti.f64,
                                        )
                                        diagnostic_outside_probe_cell = ti.Vector(
                                            [
                                                near_extension_i,
                                                near_extension_j,
                                                near_extension_k,
                                            ]
                                        )
                                        diagnostic_outside_probe_grid_coordinate = (
                                            ti.Vector(
                                                [
                                                    ti.cast(
                                                        extension_coordinate.x,
                                                        ti.f64,
                                                    ),
                                                    ti.cast(
                                                        extension_coordinate.y,
                                                        ti.f64,
                                                    ),
                                                    ti.cast(
                                                        extension_coordinate.z,
                                                        ti.f64,
                                                    ),
                                                ]
                                            )
                                        )
                                        diagnostic_outside_probe_fluid_weight = (
                                            ti.cast(sample_weight, ti.f64)
                                        )
                                        diagnostic_outside_probe_multiplier = ti.cast(
                                            rung_distance / probe_distance_m,
                                            ti.f64,
                                        )
                                        diagnostic_outside_probe_ladder_mode = 3
                                    two_sided_found_extended = 1
                                if (
                                    sample_weight > 1.0e-12
                                    and sample_gradient_valid == 1
                                    and side_gradient_found == 0
                                ):
                                    if side_is_inside == 1:
                                        inside_gradient = sample_gradient
                                        inside_gradient_found = 1
                                    else:
                                        outside_gradient = sample_gradient
                                        outside_gradient_found = 1
                # S2-A9 declared air-backed interface: a far-pressure
                # closure region declares its dry side to be the configured
                # far pressure. That is the region's physical meaning, not a
                # fallback for when the opposite-side walk finds nothing. The
                # decision order is therefore keyed on
                # closure_region_marker (S2-A5 single-copy style: only the
                # if-chain conditions are reordered, no sampling body is
                # duplicated): closure markers never enter the two-sided
                # branch; the closure branch fires whenever the inside
                # found water, regardless of the outside flag, and the
                # suppressed spurious outside water is counted for
                # observability. Non-closure markers keep the original
                # chain bit for bit.
                if (
                    closure_region_marker == 0
                    and one_sided_region_marker == 1
                    and inside_pressure_found == 1
                ):
                    outside_pressure = one_sided_reference_pressure_pa
                    outside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    pressure_traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    gradient = outside_gradient - inside_gradient
                    viscous_mode = 5
                    ti.atomic_add(
                        self.report_stress_one_sided_pressure_marker_count[None],
                        1,
                    )
                    if two_sided_found_extended == 1:
                        ti.atomic_add(
                            self.report_stress_one_sided_extended_marker_count[None],
                            1,
                        )
                    if inside_gradient_found == 0:
                        ti.atomic_add(
                            self.report_stress_one_sided_gradient_missing_marker_count[
                                None
                            ],
                            1,
                        )
                elif (
                    closure_region_marker == 0
                    and one_sided_region_marker == 1
                    and outside_pressure_found == 1
                ):
                    inside_pressure = one_sided_reference_pressure_pa
                    inside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    pressure_traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    gradient = outside_gradient - inside_gradient
                    viscous_mode = 6
                    ti.atomic_add(
                        self.report_stress_one_sided_pressure_marker_count[None],
                        1,
                    )
                    if two_sided_found_extended == 1:
                        ti.atomic_add(
                            self.report_stress_one_sided_extended_marker_count[None],
                            1,
                        )
                    if outside_gradient_found == 0:
                        ti.atomic_add(
                            self.report_stress_one_sided_gradient_missing_marker_count[
                                None
                            ],
                            1,
                        )
                elif (
                    closure_region_marker == 0
                    and outside_pressure_found == 1
                    and inside_pressure_found == 1
                ):
                    pressure_traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    gradient = outside_gradient - inside_gradient
                    viscous_mode = 2
                    ti.atomic_add(
                        self.report_stress_two_sided_pressure_marker_count[None],
                        1,
                    )
                    if two_sided_found_extended == 1:
                        ti.atomic_add(
                            self.report_stress_two_sided_extended_marker_count[None],
                            1,
                        )
                elif (
                    closure_region_marker == 1
                    and inside_pressure_found == 1
                    and far_pressure_side_normal_sign >= -0.5
                ):
                    outside_pressure = far_pressure_pa
                    # The declared air side carries no fluid: any gradient
                    # the outside walk may have stored came from the same
                    # spurious water as the suppressed pressure and is
                    # discarded with it. On the legacy outside-not-found
                    # population this rewrites the still-zero matrix with
                    # the same zero (a gradient can only be found at a
                    # candidate whose fluid weight also sets the pressure
                    # flag), keeping that population bit for bit.
                    outside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    pressure_traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    gradient = outside_gradient - inside_gradient
                    viscous_mode = 3
                    ti.atomic_add(
                        self.report_stress_far_pressure_closed_marker_count[None],
                        1,
                    )
                    if inside_found_extended == 1:
                        ti.atomic_add(
                            self.report_stress_far_pressure_closed_extended_marker_count[None],
                            1,
                        )
                    if outside_pressure_found == 1:
                        ti.atomic_add(
                            self.report_stress_far_pressure_outside_suppressed_marker_count[None],
                            1,
                        )
                elif (
                    closure_region_marker == 1
                    and outside_pressure_found == 1
                    and far_pressure_side_normal_sign <= 0.5
                ):
                    # Mirrored closure: the structurally dry side sits on the
                    # inside (-n) walk because of the CAD winding (the chain
                    # position guarantees inside_pressure_found == 0 here).
                    # The same covariant formula applies with the known far
                    # pressure substituted on the dry side; inside_gradient
                    # is provably zero (it is only written when
                    # inside_gradient_found is set, and a gradient can only
                    # be found at a candidate whose fluid weight also sets
                    # the pressure flag, which is 0 on this branch). The
                    # outside water is the genuine water side of the
                    # interface here, so nothing is suppressed.
                    inside_pressure = far_pressure_pa
                    pressure_traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    gradient = outside_gradient - inside_gradient
                    viscous_mode = 4
                    ti.atomic_add(
                        self.report_stress_far_pressure_closed_marker_count[None],
                        1,
                    )
                else:
                    pressure_sample_valid = False
                    # S2-A6 anchor fallback: both closure branches missed
                    # because the normal walk never sampled even a pressure
                    # weight on either side (the band obstacle slabs seal
                    # the whole 12x reach in some cavity columns). If the
                    # pressure-Neumann row assembly anchored this marker to
                    # a row-owning fluid cell, read that cell-center
                    # pressure directly (no interpolation - the anchor
                    # participates in the pressure solve by construction,
                    # never a band/stale cell) as the water side and the
                    # known far pressure as the dry side. The water side
                    # follows from the anchor center's normal projection:
                    # anchor on -n means water inside, so the covariant
                    # two-sided formula gives (p_anchor - p_far) * n;
                    # otherwise the mirrored orientation gives
                    # (p_far - p_anchor) * n. The viscous gradient stays
                    # the zero matrix: no walk candidate had fluid weight,
                    # so no one-sided gradient was ever found either.
                    if (
                        use_pressure_anchor_fallback != 0
                        and (
                            closure_region_marker == 1
                            or (
                                closure_region_marker == 0
                                and one_sided_region_marker == 1
                            )
                        )
                        and marker_pressure_anchor_cell[marker].x >= 0
                    ):
                        anchor_cell = marker_pressure_anchor_cell[marker]
                        anchor_pressure = pressure_field[
                            anchor_cell.x,
                            anchor_cell.y,
                            anchor_cell.z,
                        ]
                        anchor_center = ti.Vector(
                            [
                                cell_center_x_m[anchor_cell.x],
                                cell_center_y_m[anchor_cell.y],
                                cell_center_z_m[anchor_cell.z],
                            ]
                        )
                        reference_pressure = ti.cast(far_pressure_pa, ti.f64)
                        if closure_region_marker == 0 and one_sided_region_marker == 1:
                            reference_pressure = ti.cast(
                                one_sided_reference_pressure_pa,
                                ti.f64,
                            )
                        inside_pressure = reference_pressure
                        outside_pressure = anchor_pressure
                        if (
                            closure_region_marker == 1
                            and far_pressure_side_normal_sign > 0.5
                        ):
                            inside_pressure = anchor_pressure
                            outside_pressure = reference_pressure
                        elif (
                            closure_region_marker == 1
                            and far_pressure_side_normal_sign < -0.5
                        ):
                            inside_pressure = reference_pressure
                            outside_pressure = anchor_pressure
                        elif (anchor_center - position).dot(normal) < 0.0:
                            inside_pressure = anchor_pressure
                            outside_pressure = reference_pressure
                        pressure_traction = (
                            inside_pressure - outside_pressure
                        ) * normal
                        pressure_sample_valid = True
                        gradient = ti.Matrix.zero(ti.f32, 3, 3)
                        viscous_mode = 7
                        if closure_region_marker == 1:
                            ti.atomic_add(
                                self.report_stress_far_pressure_closed_marker_count[
                                    None
                                ],
                                1,
                            )
                            ti.atomic_add(
                                self.report_stress_far_pressure_anchor_closed_marker_count[
                                    None
                                ],
                                1,
                            )
                        else:
                            ti.atomic_add(
                                self.report_stress_one_sided_pressure_marker_count[None],
                                1,
                            )
                            ti.atomic_add(
                                self.report_stress_one_sided_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    # S2-A7 second-stage fallback: the marker-level anchor
                    # above is sourced from the pressure-Neumann row
                    # assembly, but in geometries where every near-boundary
                    # fluid cell carries a velocity-Dirichlet row the
                    # Neumann assembly produces zero rows and that source
                    # is structurally empty. The node-level anchor field
                    # (owned by the IB node search, populated by the
                    # velocity-Dirichlet row assembly's interior-fluid
                    # sample / relocated claim and by the interior-point
                    # prefill) decouples the closure from marker-row
                    # existence: take the 8 corner nodes of the marker's
                    # cell base floor(grid_coord) in fixed z-fastest order
                    # (indices clamped to the node grid) and use the first
                    # node whose anchor is set. The anchor cell is a
                    # non-obstacle, solve-participating cell by
                    # construction, so the same direct cell-center read
                    # and orientation-covariant formula apply; only the
                    # dedicated node-anchor counter (plus the shared
                    # closed counter) advances. The marker-level anchor
                    # keeps priority: where Neumann rows exist it stays
                    # the more precise source.
                    elif (
                        use_pressure_anchor_fallback != 0
                        and node_anchor_available != 0
                        and (
                            closure_region_marker == 1
                            or (
                                closure_region_marker == 0
                                and one_sided_region_marker == 1
                            )
                        )
                    ):
                        corner_base_i = ti.floor(grid_coordinate.x, ti.i32)
                        corner_base_j = ti.floor(grid_coordinate.y, ti.i32)
                        corner_base_k = ti.floor(grid_coordinate.z, ti.i32)
                        node_anchor_found = 0
                        node_anchor_i = 0
                        node_anchor_j = 0
                        node_anchor_k = 0
                        for corner_index in range(8):
                            if node_anchor_found == 0:
                                corner_i = ti.min(
                                    ti.max(
                                        corner_base_i + corner_index // 4,
                                        0,
                                    ),
                                    nx - 1,
                                )
                                corner_j = ti.min(
                                    ti.max(
                                        corner_base_j + (corner_index // 2) % 2,
                                        0,
                                    ),
                                    ny - 1,
                                )
                                corner_k = ti.min(
                                    ti.max(
                                        corner_base_k + corner_index % 2,
                                        0,
                                    ),
                                    nz - 1,
                                )
                                corner_anchor = node_anchor_cell[
                                    corner_i,
                                    corner_j,
                                    corner_k,
                                ]
                                if corner_anchor.x >= 0:
                                    node_anchor_found = 1
                                    node_anchor_i = corner_anchor.x
                                    node_anchor_j = corner_anchor.y
                                    node_anchor_k = corner_anchor.z
                        if node_anchor_found == 1:
                            anchor_pressure = pressure_field[
                                node_anchor_i,
                                node_anchor_j,
                                node_anchor_k,
                            ]
                            anchor_center = ti.Vector(
                                [
                                    cell_center_x_m[node_anchor_i],
                                    cell_center_y_m[node_anchor_j],
                                    cell_center_z_m[node_anchor_k],
                                ]
                            )
                            reference_pressure = ti.cast(far_pressure_pa, ti.f64)
                            if (
                                closure_region_marker == 0
                                and one_sided_region_marker == 1
                            ):
                                reference_pressure = ti.cast(
                                    one_sided_reference_pressure_pa,
                                    ti.f64,
                                )
                            inside_pressure = reference_pressure
                            outside_pressure = anchor_pressure
                            if (
                                closure_region_marker == 1
                                and far_pressure_side_normal_sign > 0.5
                            ):
                                inside_pressure = anchor_pressure
                                outside_pressure = reference_pressure
                            elif (
                                closure_region_marker == 1
                                and far_pressure_side_normal_sign < -0.5
                            ):
                                inside_pressure = reference_pressure
                                outside_pressure = anchor_pressure
                            elif (anchor_center - position).dot(normal) < 0.0:
                                inside_pressure = anchor_pressure
                                outside_pressure = reference_pressure
                            pressure_traction = (
                                inside_pressure - outside_pressure
                            ) * normal
                            pressure_sample_valid = True
                            gradient = ti.Matrix.zero(ti.f32, 3, 3)
                            viscous_mode = 7
                            if closure_region_marker == 1:
                                ti.atomic_add(
                                    self.report_stress_far_pressure_closed_marker_count[
                                        None
                                    ],
                                    1,
                                )
                                ti.atomic_add(
                                    self.report_stress_far_pressure_node_anchor_closed_marker_count[
                                        None
                                    ],
                                    1,
                                )
                            else:
                                ti.atomic_add(
                                    self.report_stress_one_sided_pressure_marker_count[
                                        None
                                    ],
                                    1,
                                )
                                ti.atomic_add(
                                    self.report_stress_one_sided_gradient_missing_marker_count[
                                        None
                                    ],
                                    1,
                                )
                # S2-A4 diagnostic: the marker closes on pressure while at
                # least one pressure-found side never completed a viscous
                # gradient stencil (that side's gradient contribution is
                # exactly zero). In two-sided mode the outer gradient_valid
                # stays 1, so final validity equals pressure_sample_valid and
                # counting here counts exactly the finally-valid markers.
                if closure_region_marker == 1 and pressure_sample_valid:
                    if (
                        outside_pressure_found == 1
                        and outside_gradient_found == 0
                    ) or (
                        inside_pressure_found == 1
                        and inside_gradient_found == 0
                    ):
                        ti.atomic_add(
                            self.report_stress_closure_gradient_missing_marker_count[
                                None
                            ],
                            1,
                        )
                diagnostic_inside_pressure_found = inside_pressure_found
                diagnostic_outside_pressure_found = outside_pressure_found
                if pressure_sample_valid:
                    diagnostic_inside_pressure_pa = ti.cast(inside_pressure, ti.f64)
                    diagnostic_outside_pressure_pa = ti.cast(outside_pressure, ti.f64)
                    diagnostic_pressure_jump_pa = ti.cast(
                        inside_pressure - outside_pressure,
                        ti.f64,
                    )
                    diagnostic_fluid_side_pressure_pa = ti.cast(
                        inside_pressure,
                        ti.f64,
                    )
                    diagnostic_reference_pressure_pa = ti.cast(
                        outside_pressure,
                        ti.f64,
                    )
                    if viscous_mode == 4 or viscous_mode == 6:
                        diagnostic_fluid_side_pressure_pa = ti.cast(
                            outside_pressure,
                            ti.f64,
                        )
                        diagnostic_reference_pressure_pa = ti.cast(
                            inside_pressure,
                            ti.f64,
                        )
                    diagnostic_probe_mode = viscous_mode
            else:
                if ti.static(viscosity_pa_s <= 0.0):
                    gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    gradient_valid = 1
                else:
                    gradient, gradient_valid = self._sample_velocity_gradient_sampling_view(
                        velocity_field,
                        obstacle_field,
                        sampling_obstacle_field,
                        use_sampling_obstacle,
                        grid_coordinate.x,
                        grid_coordinate.y,
                        grid_coordinate.z,
                        nx,
                        ny,
                        nz,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                    )
            stress_sample_valid = pressure_sample_valid
            if ti.static(viscosity_pa_s > 0.0):
                if gradient_valid == 0:
                    stress_sample_valid = False
                    ti.atomic_add(
                        self.report_stress_viscous_gradient_invalid_marker_count[None],
                        1,
                    )
            invalid_reason = STRESS_INVALID_REASON_NONE
            if not stress_sample_valid:
                if diagnostic_base_pressure_found == 0:
                    invalid_reason = STRESS_INVALID_REASON_BASE_PRESSURE_MISSING
                if two_sided_pressure != 0 and (
                    diagnostic_inside_pressure_found == 0
                    or diagnostic_outside_pressure_found == 0
                ):
                    invalid_reason = STRESS_INVALID_REASON_TWO_SIDED_PRESSURE_MISSING
                if (
                    pressure_sample_valid
                    and ti.static(viscosity_pa_s > 0.0)
                    and gradient_valid == 0
                ):
                    invalid_reason = STRESS_INVALID_REASON_VISCOUS_GRADIENT_MISSING
                diagnostic_probe_mode = 0
            self._stress_base_pressure_found[marker] = diagnostic_base_pressure_found
            self._stress_inside_pressure_found[marker] = diagnostic_inside_pressure_found
            self._stress_outside_pressure_found[marker] = diagnostic_outside_pressure_found
            self._stress_marker_anchor_available[marker] = (
                diagnostic_marker_anchor_available
            )
            self._stress_invalid_reason_code[marker] = invalid_reason
            self._stress_base_pressure_pa[marker] = diagnostic_base_pressure_pa
            self._stress_inside_pressure_pa[marker] = diagnostic_inside_pressure_pa
            self._stress_outside_pressure_pa[marker] = diagnostic_outside_pressure_pa
            self._stress_pressure_jump_pa[marker] = diagnostic_pressure_jump_pa
            self._stress_fluid_side_pressure_pa[marker] = (
                diagnostic_fluid_side_pressure_pa
            )
            self._stress_reference_pressure_pa[marker] = diagnostic_reference_pressure_pa
            self._stress_inside_probe_rung[marker] = diagnostic_inside_probe_rung
            self._stress_outside_probe_rung[marker] = diagnostic_outside_probe_rung
            self._stress_inside_probe_distance_m[marker] = (
                diagnostic_inside_probe_distance_m
            )
            self._stress_outside_probe_distance_m[marker] = (
                diagnostic_outside_probe_distance_m
            )
            self._stress_inside_probe_cell[marker] = diagnostic_inside_probe_cell
            self._stress_outside_probe_cell[marker] = diagnostic_outside_probe_cell
            self._stress_inside_probe_grid_coordinate[marker] = (
                diagnostic_inside_probe_grid_coordinate
            )
            self._stress_outside_probe_grid_coordinate[marker] = (
                diagnostic_outside_probe_grid_coordinate
            )
            self._stress_inside_probe_fluid_weight[marker] = (
                diagnostic_inside_probe_fluid_weight
            )
            self._stress_outside_probe_fluid_weight[marker] = (
                diagnostic_outside_probe_fluid_weight
            )
            self._stress_inside_probe_multiplier[marker] = (
                diagnostic_inside_probe_multiplier
            )
            self._stress_outside_probe_multiplier[marker] = (
                diagnostic_outside_probe_multiplier
            )
            self._stress_inside_probe_ladder_mode[marker] = (
                diagnostic_inside_probe_ladder_mode
            )
            self._stress_outside_probe_ladder_mode[marker] = (
                diagnostic_outside_probe_ladder_mode
            )
            self._stress_probe_mode[marker] = diagnostic_probe_mode
            if stress_sample_valid:
                traction = pressure_traction
                viscous_traction = ti.Vector([0.0, 0.0, 0.0])
                if ti.static(viscosity_pa_s > 0.0):
                    viscous_stress = viscosity_pa_s * (gradient + gradient.transpose())
                    viscous_traction = viscous_stress @ normal
                    traction = pressure_traction + viscous_traction
                self.t_gamma_pa[marker] = traction
                self.t_pressure_gamma_pa[marker] = pressure_traction
                self.t_viscous_gamma_pa[marker] = viscous_traction
                self._stress_pressure_valid[marker] = 1
                self._stress_viscous_mode[marker] = viscous_mode
                self.report_stress_valid_marker_count[None] += 1
                ti.atomic_max(
                    self.report_stress_max_abs_traction_pa[None],
                    traction.norm(),
                )
            else:
                self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                self.t_pressure_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                self.t_viscous_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                self._stress_pressure_valid[marker] = 0
                self._stress_viscous_mode[marker] = 0
                self.report_stress_invalid_marker_count[None] += 1

    @ti.kernel
    def _sample_pressure_only_marker_tractions_kernel(
        self,
        pressure_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        two_sided_pressure: ti.i32,
        one_sided_pressure_region_id: ti.i32,
        one_sided_reference_pressure_pa: ti.f32,
        one_sided_pressure_primary_region_id: ti.i32,
        one_sided_pressure_secondary_region_id: ti.i32,
        one_sided_primary_reference_pressure_pa: ti.f32,
        one_sided_secondary_reference_pressure_pa: ti.f32,
        one_sided_primary_fluid_side_normal_sign: ti.f32,
        one_sided_secondary_fluid_side_normal_sign: ti.f32,
        pressure_probe_ladder_start_multiplier: ti.f32,
        pressure_probe_ladder_spacing_multiplier: ti.f32,
        pressure_probe_ladder_rung_count: ti.i32,
        pressure_probe_ladder_configured: ti.i32,
        pressure_pair_policy_code: ti.i32,
        pressure_pair_max_cell_delta: ti.i32,
        pressure_pair_require_opposite_sides: ti.i32,
    ):
        self.report_stress_valid_marker_count[None] = 0
        self.report_stress_invalid_marker_count[None] = 0
        self.report_stress_max_abs_traction_pa[None] = 0.0
        self.report_stress_two_sided_pressure_marker_count[None] = 0
        self.report_stress_viscous_gradient_invalid_marker_count[None] = 0
        self.report_stress_far_pressure_closed_marker_count[None] = 0
        self.report_stress_far_pressure_closed_extended_marker_count[None] = 0
        self.report_stress_far_pressure_anchor_closed_marker_count[None] = 0
        self.report_stress_far_pressure_node_anchor_closed_marker_count[None] = 0
        self.report_stress_closure_gradient_missing_marker_count[None] = 0
        self.report_stress_far_pressure_outside_suppressed_marker_count[None] = 0
        self.report_stress_two_sided_extended_marker_count[None] = 0
        self.report_stress_one_sided_pressure_marker_count[None] = 0
        self.report_stress_one_sided_extended_marker_count[None] = 0
        self.report_stress_one_sided_gradient_missing_marker_count[None] = 0
        for marker in range(marker_count):
            position = self.x_gamma_m[marker]
            normal = self.n_gamma[marker]
            probe_origin = position
            if self.pressure_probe_origin_explicit[marker] != 0:
                probe_origin = self.pressure_probe_origin_m[marker]
            grid_coordinate = self._grid_coordinate_from_fields(
                position,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                nx,
                ny,
                nz,
            )
            pressure, pressure_weight = self._sample_pressure_trilinear(
                pressure_field,
                obstacle_field,
                grid_coordinate.x,
                grid_coordinate.y,
                grid_coordinate.z,
                nx,
                ny,
                nz,
            )
            traction = -pressure * normal
            pressure_sample_valid = pressure_weight > 1.0e-12
            diagnostic_base_pressure_found = 0
            if pressure_sample_valid:
                diagnostic_base_pressure_found = 1
            diagnostic_inside_pressure_found = 0
            diagnostic_outside_pressure_found = 0
            diagnostic_base_pressure_pa = ti.cast(pressure, ti.f64)
            diagnostic_inside_pressure_pa = ti.cast(0.0, ti.f64)
            diagnostic_outside_pressure_pa = ti.cast(0.0, ti.f64)
            diagnostic_pressure_jump_pa = ti.cast(0.0, ti.f64)
            diagnostic_fluid_side_pressure_pa = ti.cast(0.0, ti.f64)
            diagnostic_reference_pressure_pa = ti.cast(0.0, ti.f64)
            diagnostic_inside_probe_rung = -1
            diagnostic_outside_probe_rung = -1
            diagnostic_inside_probe_distance_m = ti.cast(-1.0, ti.f64)
            diagnostic_outside_probe_distance_m = ti.cast(-1.0, ti.f64)
            diagnostic_inside_probe_cell = ti.Vector([-1, -1, -1])
            diagnostic_outside_probe_cell = ti.Vector([-1, -1, -1])
            diagnostic_inside_probe_grid_coordinate = ti.Vector(
                [
                    ti.cast(-1.0, ti.f64),
                    ti.cast(-1.0, ti.f64),
                    ti.cast(-1.0, ti.f64),
                ]
            )
            diagnostic_outside_probe_grid_coordinate = ti.Vector(
                [
                    ti.cast(-1.0, ti.f64),
                    ti.cast(-1.0, ti.f64),
                    ti.cast(-1.0, ti.f64),
                ]
            )
            diagnostic_inside_probe_fluid_weight = ti.cast(0.0, ti.f64)
            diagnostic_outside_probe_fluid_weight = ti.cast(0.0, ti.f64)
            diagnostic_inside_probe_multiplier = ti.cast(0.0, ti.f64)
            diagnostic_outside_probe_multiplier = ti.cast(0.0, ti.f64)
            diagnostic_inside_probe_ladder_mode = 0
            diagnostic_outside_probe_ladder_mode = 0
            diagnostic_probe_mode = 0
            diagnostic_pressure_pair_selected = 0
            diagnostic_pressure_pair_fallback_used = 0
            diagnostic_pressure_pair_inside_cell = ti.Vector([-1, -1, -1])
            diagnostic_pressure_pair_outside_cell = ti.Vector([-1, -1, -1])
            diagnostic_pressure_pair_cell_delta = -1
            diagnostic_pressure_pair_symmetry_residual_cells = ti.cast(-1.0, ti.f64)
            diagnostic_pressure_pair_anchor_fallback_used = 0
            diagnostic_one_sided_policy_code = 0
            diagnostic_one_sided_region_id = -1
            diagnostic_one_sided_side_normal_sign = ti.cast(0.0, ti.f64)
            diagnostic_one_sided_anchor_selected = 0
            diagnostic_one_sided_anchor_fallback_used = 0
            if pressure_sample_valid:
                diagnostic_fluid_side_pressure_pa = diagnostic_base_pressure_pa
                diagnostic_probe_mode = 1
            if two_sided_pressure != 0:
                i_near = ti.min(ti.max(ti.floor(grid_coordinate.x, ti.i32), 0), nx - 1)
                j_near = ti.min(ti.max(ti.floor(grid_coordinate.y, ti.i32), 0), ny - 1)
                k_near = ti.min(ti.max(ti.floor(grid_coordinate.z, ti.i32), 0), nz - 1)
                normal_spacing_inv = (
                    ti.abs(normal.x) / cell_width_x_m[i_near]
                    + ti.abs(normal.y) / cell_width_y_m[j_near]
                    + ti.abs(normal.z) / cell_width_z_m[k_near]
                )
                probe_distance_m = 1.0 / ti.max(normal_spacing_inv, 1.0e-12)
                outside_pressure = ti.cast(0.0, ti.f64)
                inside_pressure = ti.cast(0.0, ti.f64)
                outside_found = 0
                inside_found = 0
                probe_start = ti.cast(1.0, ti.f32)
                probe_spacing = ti.cast(1.0, ti.f32)
                probe_count = 3
                probe_ladder_mode = 4
                if pressure_probe_ladder_configured != 0:
                    probe_start = pressure_probe_ladder_start_multiplier
                    probe_spacing = pressure_probe_ladder_spacing_multiplier
                    probe_count = pressure_probe_ladder_rung_count
                    probe_ladder_mode = 5
                probe_origin_grid = self._grid_coordinate_from_fields(
                    probe_origin,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                if pressure_pair_policy_code == 2:
                    anchor_inside_cell = self.pressure_pair_anchor_inside_cell[marker]
                    anchor_outside_cell = self.pressure_pair_anchor_outside_cell[marker]
                    anchor_active = self.pressure_pair_anchor_active[marker]
                    anchor_inside_in_bounds = 0
                    anchor_outside_in_bounds = 0
                    if (
                        anchor_inside_cell.x >= 0
                        and anchor_inside_cell.x < nx
                        and anchor_inside_cell.y >= 0
                        and anchor_inside_cell.y < ny
                        and anchor_inside_cell.z >= 0
                        and anchor_inside_cell.z < nz
                    ):
                        anchor_inside_in_bounds = 1
                    if (
                        anchor_outside_cell.x >= 0
                        and anchor_outside_cell.x < nx
                        and anchor_outside_cell.y >= 0
                        and anchor_outside_cell.y < ny
                        and anchor_outside_cell.z >= 0
                        and anchor_outside_cell.z < nz
                    ):
                        anchor_outside_in_bounds = 1
                    if (
                        anchor_active != 0
                        and anchor_inside_in_bounds != 0
                        and anchor_outside_in_bounds != 0
                    ):
                        sampled_inside = ti.cast(
                            pressure_field[
                                anchor_inside_cell.x,
                                anchor_inside_cell.y,
                                anchor_inside_cell.z,
                            ],
                            ti.f64,
                        )
                        sampled_outside = ti.cast(
                            pressure_field[
                                anchor_outside_cell.x,
                                anchor_outside_cell.y,
                                anchor_outside_cell.z,
                            ],
                            ti.f64,
                        )
                        inside_pressure = sampled_inside
                        outside_pressure = sampled_outside
                        inside_found = 1
                        outside_found = 1
                        diagnostic_inside_pressure_pa = sampled_inside
                        diagnostic_outside_pressure_pa = sampled_outside
                        diagnostic_inside_pressure_found = 1
                        diagnostic_outside_pressure_found = 1
                        diagnostic_inside_probe_rung = -1
                        diagnostic_outside_probe_rung = -1
                        diagnostic_inside_probe_distance_m = ti.cast(0.0, ti.f64)
                        diagnostic_outside_probe_distance_m = ti.cast(0.0, ti.f64)
                        diagnostic_inside_probe_cell = anchor_inside_cell
                        diagnostic_outside_probe_cell = anchor_outside_cell
                        diagnostic_inside_probe_grid_coordinate = ti.Vector(
                            [
                                ti.cast(anchor_inside_cell.x, ti.f64),
                                ti.cast(anchor_inside_cell.y, ti.f64),
                                ti.cast(anchor_inside_cell.z, ti.f64),
                            ]
                        )
                        diagnostic_outside_probe_grid_coordinate = ti.Vector(
                            [
                                ti.cast(anchor_outside_cell.x, ti.f64),
                                ti.cast(anchor_outside_cell.y, ti.f64),
                                ti.cast(anchor_outside_cell.z, ti.f64),
                            ]
                        )
                        diagnostic_inside_probe_fluid_weight = ti.cast(1.0, ti.f64)
                        diagnostic_outside_probe_fluid_weight = ti.cast(1.0, ti.f64)
                        diagnostic_inside_probe_multiplier = ti.cast(0.0, ti.f64)
                        diagnostic_outside_probe_multiplier = ti.cast(0.0, ti.f64)
                        diagnostic_inside_probe_ladder_mode = 0
                        diagnostic_outside_probe_ladder_mode = 0
                        diagnostic_pressure_pair_selected = 1
                        diagnostic_pressure_pair_inside_cell = anchor_inside_cell
                        diagnostic_pressure_pair_outside_cell = anchor_outside_cell
                        anchor_cell_delta_x = ti.abs(
                            anchor_inside_cell.x - anchor_outside_cell.x
                        )
                        anchor_cell_delta_y = ti.abs(
                            anchor_inside_cell.y - anchor_outside_cell.y
                        )
                        anchor_cell_delta_z = ti.abs(
                            anchor_inside_cell.z - anchor_outside_cell.z
                        )
                        diagnostic_pressure_pair_cell_delta = ti.max(
                            ti.max(anchor_cell_delta_x, anchor_cell_delta_y),
                            anchor_cell_delta_z,
                        )
                        diagnostic_pressure_pair_symmetry_residual_cells = ti.cast(
                            0.0,
                            ti.f64,
                        )
                for probe_index in range(probe_count):
                    multiplier = (
                        probe_start + probe_spacing * ti.cast(probe_index, ti.f32)
                    )
                    outside_position = (
                        probe_origin + multiplier * probe_distance_m * normal
                    )
                    outside_grid = self._grid_coordinate_from_fields(
                        outside_position,
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    inside_position = (
                        probe_origin - multiplier * probe_distance_m * normal
                    )
                    inside_grid = self._grid_coordinate_from_fields(
                        inside_position,
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    sampled_outside, outside_weight = self._sample_pressure_trilinear(
                        pressure_field,
                        obstacle_field,
                        outside_grid.x,
                        outside_grid.y,
                        outside_grid.z,
                        nx,
                        ny,
                        nz,
                    )
                    sampled_inside, inside_weight = self._sample_pressure_trilinear(
                        pressure_field,
                        obstacle_field,
                        inside_grid.x,
                        inside_grid.y,
                        inside_grid.z,
                        nx,
                        ny,
                        nz,
                    )
                    outside_cell = ti.Vector(
                        [
                            ti.min(
                                ti.max(ti.floor(outside_grid.x + 0.5, ti.i32), 0),
                                nx - 1,
                            ),
                            ti.min(
                                ti.max(ti.floor(outside_grid.y + 0.5, ti.i32), 0),
                                ny - 1,
                            ),
                            ti.min(
                                ti.max(ti.floor(outside_grid.z + 0.5, ti.i32), 0),
                                nz - 1,
                            ),
                        ]
                    )
                    inside_cell = ti.Vector(
                        [
                            ti.min(
                                ti.max(ti.floor(inside_grid.x + 0.5, ti.i32), 0),
                                nx - 1,
                            ),
                            ti.min(
                                ti.max(ti.floor(inside_grid.y + 0.5, ti.i32), 0),
                                ny - 1,
                            ),
                            ti.min(
                                ti.max(ti.floor(inside_grid.z + 0.5, ti.i32), 0),
                                nz - 1,
                            ),
                        ]
                    )
                    if pressure_pair_policy_code == 1:
                        pair_residual_cells = ti.max(
                            ti.max(
                                ti.abs(
                                    outside_grid.x
                                    - probe_origin_grid.x
                                    + inside_grid.x
                                    - probe_origin_grid.x
                                ),
                                ti.abs(
                                    outside_grid.y
                                    - probe_origin_grid.y
                                    + inside_grid.y
                                    - probe_origin_grid.y
                                ),
                            ),
                            ti.abs(
                                outside_grid.z
                                - probe_origin_grid.z
                                + inside_grid.z
                                - probe_origin_grid.z
                            ),
                        )
                        opposite_sides_ok = 1
                        if pressure_pair_require_opposite_sides != 0:
                            outside_side = (outside_position - probe_origin).dot(normal)
                            inside_side = (inside_position - probe_origin).dot(normal)
                            if outside_side <= 0.0 or inside_side >= 0.0:
                                opposite_sides_ok = 0
                        if (
                            diagnostic_pressure_pair_selected == 0
                            and outside_weight > 1.0e-12
                            and inside_weight > 1.0e-12
                            and pair_residual_cells
                            <= ti.cast(pressure_pair_max_cell_delta, ti.f32) + 1.0e-8
                            and opposite_sides_ok != 0
                        ):
                            outside_pressure = sampled_outside
                            inside_pressure = sampled_inside
                            outside_found = 1
                            inside_found = 1
                            diagnostic_outside_pressure_pa = ti.cast(
                                sampled_outside,
                                ti.f64,
                            )
                            diagnostic_inside_pressure_pa = ti.cast(
                                sampled_inside,
                                ti.f64,
                            )
                            diagnostic_outside_pressure_found = 1
                            diagnostic_inside_pressure_found = 1
                            diagnostic_outside_probe_rung = probe_index
                            diagnostic_inside_probe_rung = probe_index
                            diagnostic_outside_probe_distance_m = ti.cast(
                                multiplier * probe_distance_m,
                                ti.f64,
                            )
                            diagnostic_inside_probe_distance_m = ti.cast(
                                multiplier * probe_distance_m,
                                ti.f64,
                            )
                            diagnostic_outside_probe_cell = outside_cell
                            diagnostic_inside_probe_cell = inside_cell
                            diagnostic_outside_probe_grid_coordinate = ti.Vector(
                                [
                                    ti.cast(outside_grid.x, ti.f64),
                                    ti.cast(outside_grid.y, ti.f64),
                                    ti.cast(outside_grid.z, ti.f64),
                                ]
                            )
                            diagnostic_inside_probe_grid_coordinate = ti.Vector(
                                [
                                    ti.cast(inside_grid.x, ti.f64),
                                    ti.cast(inside_grid.y, ti.f64),
                                    ti.cast(inside_grid.z, ti.f64),
                                ]
                            )
                            diagnostic_outside_probe_fluid_weight = ti.cast(
                                outside_weight,
                                ti.f64,
                            )
                            diagnostic_inside_probe_fluid_weight = ti.cast(
                                inside_weight,
                                ti.f64,
                            )
                            diagnostic_outside_probe_multiplier = ti.cast(
                                multiplier,
                                ti.f64,
                            )
                            diagnostic_inside_probe_multiplier = ti.cast(
                                multiplier,
                                ti.f64,
                            )
                            diagnostic_outside_probe_ladder_mode = probe_ladder_mode
                            diagnostic_inside_probe_ladder_mode = probe_ladder_mode
                            diagnostic_pressure_pair_selected = 1
                            diagnostic_pressure_pair_inside_cell = inside_cell
                            diagnostic_pressure_pair_outside_cell = outside_cell
                            cell_delta_x = ti.abs(inside_cell.x - outside_cell.x)
                            cell_delta_y = ti.abs(inside_cell.y - outside_cell.y)
                            cell_delta_z = ti.abs(inside_cell.z - outside_cell.z)
                            diagnostic_pressure_pair_cell_delta = ti.max(
                                ti.max(cell_delta_x, cell_delta_y),
                                cell_delta_z,
                            )
                            diagnostic_pressure_pair_symmetry_residual_cells = ti.cast(
                                pair_residual_cells,
                                ti.f64,
                            )
                    elif pressure_pair_policy_code == 0:
                        if outside_found == 0 and outside_weight > 1.0e-12:
                            outside_pressure = sampled_outside
                            outside_found = 1
                            diagnostic_outside_pressure_pa = ti.cast(
                                sampled_outside,
                                ti.f64,
                            )
                            diagnostic_outside_pressure_found = 1
                            diagnostic_outside_probe_rung = probe_index
                            diagnostic_outside_probe_distance_m = ti.cast(
                                multiplier * probe_distance_m,
                                ti.f64,
                            )
                            diagnostic_outside_probe_cell = outside_cell
                            diagnostic_outside_probe_grid_coordinate = ti.Vector(
                                [
                                    ti.cast(outside_grid.x, ti.f64),
                                    ti.cast(outside_grid.y, ti.f64),
                                    ti.cast(outside_grid.z, ti.f64),
                                ]
                            )
                            diagnostic_outside_probe_fluid_weight = ti.cast(
                                outside_weight,
                                ti.f64,
                            )
                            diagnostic_outside_probe_multiplier = ti.cast(
                                multiplier,
                                ti.f64,
                            )
                            diagnostic_outside_probe_ladder_mode = probe_ladder_mode
                        if inside_found == 0 and inside_weight > 1.0e-12:
                            inside_pressure = sampled_inside
                            inside_found = 1
                            diagnostic_inside_pressure_pa = ti.cast(
                                sampled_inside,
                                ti.f64,
                            )
                            diagnostic_inside_pressure_found = 1
                            diagnostic_inside_probe_rung = probe_index
                            diagnostic_inside_probe_distance_m = ti.cast(
                                multiplier * probe_distance_m,
                                ti.f64,
                            )
                            diagnostic_inside_probe_cell = inside_cell
                            diagnostic_inside_probe_grid_coordinate = ti.Vector(
                                [
                                    ti.cast(inside_grid.x, ti.f64),
                                    ti.cast(inside_grid.y, ti.f64),
                                    ti.cast(inside_grid.z, ti.f64),
                                ]
                            )
                            diagnostic_inside_probe_fluid_weight = ti.cast(
                                inside_weight,
                                ti.f64,
                            )
                            diagnostic_inside_probe_multiplier = ti.cast(
                                multiplier,
                                ti.f64,
                            )
                            diagnostic_inside_probe_ladder_mode = probe_ladder_mode
                per_face_region_marker = 0
                per_face_reference_pressure = ti.cast(0.0, ti.f64)
                per_face_side_normal_sign = ti.cast(0.0, ti.f64)
                if (
                    one_sided_pressure_primary_region_id >= 0
                    and self.region_id[marker] == one_sided_pressure_primary_region_id
                ):
                    per_face_region_marker = 1
                    per_face_reference_pressure = ti.cast(
                        one_sided_primary_reference_pressure_pa,
                        ti.f64,
                    )
                    per_face_side_normal_sign = ti.cast(
                        one_sided_primary_fluid_side_normal_sign,
                        ti.f64,
                    )
                elif (
                    one_sided_pressure_secondary_region_id >= 0
                    and self.region_id[marker] == one_sided_pressure_secondary_region_id
                ):
                    per_face_region_marker = 1
                    per_face_reference_pressure = ti.cast(
                        one_sided_secondary_reference_pressure_pa,
                        ti.f64,
                    )
                    per_face_side_normal_sign = ti.cast(
                        one_sided_secondary_fluid_side_normal_sign,
                        ti.f64,
                    )
                if per_face_region_marker != 0 and per_face_side_normal_sign < 0.0:
                    if inside_found != 0:
                        traction = (inside_pressure - per_face_reference_pressure) * normal
                        pressure_sample_valid = True
                        diagnostic_inside_pressure_pa = ti.cast(inside_pressure, ti.f64)
                        diagnostic_outside_pressure_pa = per_face_reference_pressure
                        diagnostic_pressure_jump_pa = ti.cast(
                            inside_pressure - per_face_reference_pressure,
                            ti.f64,
                        )
                        diagnostic_fluid_side_pressure_pa = ti.cast(
                            inside_pressure,
                            ti.f64,
                        )
                        diagnostic_reference_pressure_pa = per_face_reference_pressure
                        diagnostic_probe_mode = 5
                        diagnostic_one_sided_policy_code = 2
                        diagnostic_one_sided_region_id = self.region_id[marker]
                        diagnostic_one_sided_side_normal_sign = per_face_side_normal_sign
                        diagnostic_one_sided_anchor_selected = (
                            diagnostic_pressure_pair_selected
                        )
                        diagnostic_one_sided_anchor_fallback_used = (
                            diagnostic_pressure_pair_anchor_fallback_used
                        )
                        ti.atomic_add(
                            self.report_stress_one_sided_pressure_marker_count[None],
                            1,
                        )
                    else:
                        pressure_sample_valid = False
                elif per_face_region_marker != 0 and per_face_side_normal_sign > 0.0:
                    if outside_found != 0:
                        traction = (per_face_reference_pressure - outside_pressure) * normal
                        pressure_sample_valid = True
                        diagnostic_inside_pressure_pa = per_face_reference_pressure
                        diagnostic_outside_pressure_pa = ti.cast(outside_pressure, ti.f64)
                        diagnostic_pressure_jump_pa = ti.cast(
                            per_face_reference_pressure - outside_pressure,
                            ti.f64,
                        )
                        diagnostic_fluid_side_pressure_pa = ti.cast(
                            outside_pressure,
                            ti.f64,
                        )
                        diagnostic_reference_pressure_pa = per_face_reference_pressure
                        diagnostic_probe_mode = 6
                        diagnostic_one_sided_policy_code = 2
                        diagnostic_one_sided_region_id = self.region_id[marker]
                        diagnostic_one_sided_side_normal_sign = per_face_side_normal_sign
                        diagnostic_one_sided_anchor_selected = (
                            diagnostic_pressure_pair_selected
                        )
                        diagnostic_one_sided_anchor_fallback_used = (
                            diagnostic_pressure_pair_anchor_fallback_used
                        )
                        ti.atomic_add(
                            self.report_stress_one_sided_pressure_marker_count[None],
                            1,
                        )
                    else:
                        pressure_sample_valid = False
                elif (
                    one_sided_pressure_region_id != -1
                    and self.region_id[marker] == one_sided_pressure_region_id
                    and inside_found != 0
                ):
                    traction = (
                        inside_pressure - one_sided_reference_pressure_pa
                    ) * normal
                    pressure_sample_valid = True
                    diagnostic_inside_pressure_pa = ti.cast(inside_pressure, ti.f64)
                    diagnostic_outside_pressure_pa = ti.cast(
                        one_sided_reference_pressure_pa,
                        ti.f64,
                    )
                    diagnostic_pressure_jump_pa = ti.cast(
                        inside_pressure - one_sided_reference_pressure_pa,
                        ti.f64,
                    )
                    diagnostic_fluid_side_pressure_pa = ti.cast(
                        inside_pressure,
                        ti.f64,
                    )
                    diagnostic_reference_pressure_pa = ti.cast(
                        one_sided_reference_pressure_pa,
                        ti.f64,
                    )
                    diagnostic_probe_mode = 5
                    diagnostic_one_sided_policy_code = 1
                    diagnostic_one_sided_region_id = self.region_id[marker]
                    diagnostic_one_sided_side_normal_sign = ti.cast(-1.0, ti.f64)
                    diagnostic_one_sided_anchor_selected = (
                        diagnostic_pressure_pair_selected
                    )
                    diagnostic_one_sided_anchor_fallback_used = (
                        diagnostic_pressure_pair_anchor_fallback_used
                    )
                    ti.atomic_add(
                        self.report_stress_one_sided_pressure_marker_count[None],
                        1,
                    )
                elif (
                    one_sided_pressure_region_id != -1
                    and self.region_id[marker] == one_sided_pressure_region_id
                    and outside_found != 0
                ):
                    traction = (
                        one_sided_reference_pressure_pa - outside_pressure
                    ) * normal
                    pressure_sample_valid = True
                    diagnostic_inside_pressure_pa = ti.cast(
                        one_sided_reference_pressure_pa,
                        ti.f64,
                    )
                    diagnostic_outside_pressure_pa = ti.cast(outside_pressure, ti.f64)
                    diagnostic_pressure_jump_pa = ti.cast(
                        one_sided_reference_pressure_pa - outside_pressure,
                        ti.f64,
                    )
                    diagnostic_fluid_side_pressure_pa = ti.cast(
                        outside_pressure,
                        ti.f64,
                    )
                    diagnostic_reference_pressure_pa = ti.cast(
                        one_sided_reference_pressure_pa,
                        ti.f64,
                    )
                    diagnostic_probe_mode = 6
                    diagnostic_one_sided_policy_code = 1
                    diagnostic_one_sided_region_id = self.region_id[marker]
                    diagnostic_one_sided_side_normal_sign = ti.cast(1.0, ti.f64)
                    diagnostic_one_sided_anchor_selected = (
                        diagnostic_pressure_pair_selected
                    )
                    diagnostic_one_sided_anchor_fallback_used = (
                        diagnostic_pressure_pair_anchor_fallback_used
                    )
                    ti.atomic_add(
                        self.report_stress_one_sided_pressure_marker_count[None],
                        1,
                    )
                elif outside_found != 0 and inside_found != 0:
                    traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    diagnostic_inside_pressure_pa = ti.cast(inside_pressure, ti.f64)
                    diagnostic_outside_pressure_pa = ti.cast(outside_pressure, ti.f64)
                    diagnostic_pressure_jump_pa = ti.cast(
                        inside_pressure - outside_pressure,
                        ti.f64,
                    )
                    diagnostic_fluid_side_pressure_pa = ti.cast(
                        inside_pressure,
                        ti.f64,
                    )
                    diagnostic_reference_pressure_pa = ti.cast(
                        outside_pressure,
                        ti.f64,
                    )
                    diagnostic_probe_mode = 2
                    ti.atomic_add(
                        self.report_stress_two_sided_pressure_marker_count[None],
                        1,
                    )
                else:
                    pressure_sample_valid = False
            invalid_reason = STRESS_INVALID_REASON_NONE
            if not pressure_sample_valid:
                if diagnostic_base_pressure_found == 0:
                    invalid_reason = STRESS_INVALID_REASON_BASE_PRESSURE_MISSING
                if two_sided_pressure != 0 and (
                    diagnostic_inside_pressure_found == 0
                    or diagnostic_outside_pressure_found == 0
                ):
                    invalid_reason = STRESS_INVALID_REASON_TWO_SIDED_PRESSURE_MISSING
                diagnostic_probe_mode = 0
            self._stress_base_pressure_found[marker] = diagnostic_base_pressure_found
            self._stress_inside_pressure_found[marker] = diagnostic_inside_pressure_found
            self._stress_outside_pressure_found[marker] = diagnostic_outside_pressure_found
            self._stress_marker_anchor_available[marker] = 0
            self._stress_invalid_reason_code[marker] = invalid_reason
            self._stress_base_pressure_pa[marker] = diagnostic_base_pressure_pa
            self._stress_inside_pressure_pa[marker] = diagnostic_inside_pressure_pa
            self._stress_outside_pressure_pa[marker] = diagnostic_outside_pressure_pa
            self._stress_pressure_jump_pa[marker] = diagnostic_pressure_jump_pa
            self._stress_fluid_side_pressure_pa[marker] = (
                diagnostic_fluid_side_pressure_pa
            )
            self._stress_reference_pressure_pa[marker] = diagnostic_reference_pressure_pa
            self._stress_inside_probe_rung[marker] = diagnostic_inside_probe_rung
            self._stress_outside_probe_rung[marker] = diagnostic_outside_probe_rung
            self._stress_inside_probe_distance_m[marker] = (
                diagnostic_inside_probe_distance_m
            )
            self._stress_outside_probe_distance_m[marker] = (
                diagnostic_outside_probe_distance_m
            )
            self._stress_inside_probe_cell[marker] = diagnostic_inside_probe_cell
            self._stress_outside_probe_cell[marker] = diagnostic_outside_probe_cell
            self._stress_inside_probe_grid_coordinate[marker] = (
                diagnostic_inside_probe_grid_coordinate
            )
            self._stress_outside_probe_grid_coordinate[marker] = (
                diagnostic_outside_probe_grid_coordinate
            )
            self._stress_inside_probe_fluid_weight[marker] = (
                diagnostic_inside_probe_fluid_weight
            )
            self._stress_outside_probe_fluid_weight[marker] = (
                diagnostic_outside_probe_fluid_weight
            )
            self._stress_inside_probe_multiplier[marker] = (
                diagnostic_inside_probe_multiplier
            )
            self._stress_outside_probe_multiplier[marker] = (
                diagnostic_outside_probe_multiplier
            )
            self._stress_inside_probe_ladder_mode[marker] = (
                diagnostic_inside_probe_ladder_mode
            )
            self._stress_outside_probe_ladder_mode[marker] = (
                diagnostic_outside_probe_ladder_mode
            )
            self._stress_probe_mode[marker] = diagnostic_probe_mode
            self._stress_pressure_pair_policy_code[marker] = pressure_pair_policy_code
            self._stress_pressure_pair_selected[marker] = diagnostic_pressure_pair_selected
            self._stress_pressure_pair_fallback_used[marker] = (
                diagnostic_pressure_pair_fallback_used
            )
            self._stress_pressure_pair_inside_cell[marker] = (
                diagnostic_pressure_pair_inside_cell
            )
            self._stress_pressure_pair_outside_cell[marker] = (
                diagnostic_pressure_pair_outside_cell
            )
            self._stress_pressure_pair_cell_delta[marker] = (
                diagnostic_pressure_pair_cell_delta
            )
            self._stress_pressure_pair_symmetry_residual_cells[marker] = (
                diagnostic_pressure_pair_symmetry_residual_cells
            )
            self._stress_pressure_pair_anchor_fallback_used[marker] = (
                diagnostic_pressure_pair_anchor_fallback_used
            )
            self._stress_one_sided_policy_code[marker] = diagnostic_one_sided_policy_code
            self._stress_one_sided_region_id[marker] = diagnostic_one_sided_region_id
            self._stress_one_sided_side_normal_sign[marker] = (
                diagnostic_one_sided_side_normal_sign
            )
            self._stress_one_sided_anchor_selected[marker] = (
                diagnostic_one_sided_anchor_selected
            )
            self._stress_one_sided_anchor_fallback_used[marker] = (
                diagnostic_one_sided_anchor_fallback_used
            )
            if pressure_sample_valid:
                self.t_gamma_pa[marker] = traction
                self.t_pressure_gamma_pa[marker] = traction
                self.t_viscous_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                self._stress_pressure_valid[marker] = 1
                self._stress_viscous_mode[marker] = diagnostic_probe_mode
                self.report_stress_valid_marker_count[None] += 1
                ti.atomic_max(
                    self.report_stress_max_abs_traction_pa[None],
                    traction.norm(),
                )
            else:
                self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                self.t_pressure_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                self.t_viscous_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                self._stress_pressure_valid[marker] = 0
                self._stress_viscous_mode[marker] = 0
                self.report_stress_invalid_marker_count[None] += 1

    @ti.kernel
    def _add_viscous_marker_tractions_kernel(
        self,
        velocity_field: ti.template(),
        pressure_field: ti.template(),
        obstacle_field: ti.template(),
        sampling_obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        viscosity_pa_s: ti.f32,
        two_sided_pressure: ti.i32,
        far_pressure_region_id: ti.i32,
        far_pressure_inside_probe_max_multiplier: ti.f32,
        two_sided_probe_max_multiplier: ti.f32,
        one_sided_pressure_region_id: ti.i32,
        one_sided_probe_max_multiplier: ti.f32,
        use_sampling_obstacle: ti.i32,
    ):
        self.report_stress_max_abs_traction_pa[None] = 0.0
        self.report_stress_viscous_gradient_invalid_marker_count[None] = 0
        self.report_stress_closure_gradient_missing_marker_count[None] = 0
        self.report_stress_one_sided_gradient_missing_marker_count[None] = 0
        for marker in range(marker_count):
            if self._stress_pressure_valid[marker] != 0:
                position = self.x_gamma_m[marker]
                normal = self.n_gamma[marker]
                grid_coordinate = self._grid_coordinate_from_fields(
                    position,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                gradient = ti.Matrix.zero(ti.f32, 3, 3)
                gradient_valid = 1
                if two_sided_pressure != 0:
                    i_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0),
                        nx - 1,
                    )
                    j_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0),
                        ny - 1,
                    )
                    k_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0),
                        nz - 1,
                    )
                    normal_spacing_inv = (
                        ti.abs(normal.x) / cell_width_x_m[i_near]
                        + ti.abs(normal.y) / cell_width_y_m[j_near]
                        + ti.abs(normal.z) / cell_width_z_m[k_near]
                    )
                    probe_distance_m = 1.0 / ti.max(normal_spacing_inv, 1.0e-12)
                    outside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    inside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    outside_pressure_found = 0
                    outside_gradient_found = 0
                    inside_pressure_found = 0
                    inside_gradient_found = 0
                    closure_region_marker = 0
                    if (
                        far_pressure_region_id != -1
                        and self.region_id[marker] == far_pressure_region_id
                    ):
                        closure_region_marker = 1
                    one_sided_region_marker = 0
                    if (
                        one_sided_pressure_region_id != -1
                        and self.region_id[marker] == one_sided_pressure_region_id
                    ):
                        one_sided_region_marker = 1

                    for probe_index in range(5):
                        probe_distance = probe_distance_m * (
                            1.0 + 0.5 * ti.cast(probe_index, ti.f32)
                        )
                        if outside_pressure_found == 0 or outside_gradient_found == 0:
                            outside_position = position + normal * probe_distance
                            outside_coordinate = self._grid_coordinate_from_fields(
                                outside_position,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            _, sample_weight = self._sample_pressure_trilinear_sampling_view(
                                pressure_field,
                                obstacle_field,
                                sampling_obstacle_field,
                                use_sampling_obstacle,
                                outside_coordinate.x,
                                outside_coordinate.y,
                                outside_coordinate.z,
                                nx,
                                ny,
                                nz,
                            )
                            sample_gradient, sample_gradient_valid = (
                                self._sample_velocity_gradient_sampling_view(
                                    velocity_field,
                                    obstacle_field,
                                    sampling_obstacle_field,
                                    use_sampling_obstacle,
                                    outside_coordinate.x,
                                    outside_coordinate.y,
                                    outside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                )
                            )
                            if sample_weight > 1.0e-12:
                                if outside_pressure_found == 0:
                                    outside_pressure_found = 1
                                if (
                                    sample_gradient_valid == 1
                                    and outside_gradient_found == 0
                                ):
                                    outside_gradient = sample_gradient
                                    outside_gradient_found = 1
                        if inside_pressure_found == 0 or inside_gradient_found == 0:
                            inside_position = position - normal * probe_distance
                            inside_coordinate = self._grid_coordinate_from_fields(
                                inside_position,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            _, sample_weight = self._sample_pressure_trilinear_sampling_view(
                                pressure_field,
                                obstacle_field,
                                sampling_obstacle_field,
                                use_sampling_obstacle,
                                inside_coordinate.x,
                                inside_coordinate.y,
                                inside_coordinate.z,
                                nx,
                                ny,
                                nz,
                            )
                            sample_gradient, sample_gradient_valid = (
                                self._sample_velocity_gradient_sampling_view(
                                    velocity_field,
                                    obstacle_field,
                                    sampling_obstacle_field,
                                    use_sampling_obstacle,
                                    inside_coordinate.x,
                                    inside_coordinate.y,
                                    inside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                )
                            )
                            if sample_weight > 1.0e-12:
                                if inside_pressure_found == 0:
                                    inside_pressure_found = 1
                                if (
                                    sample_gradient_valid == 1
                                    and inside_gradient_found == 0
                                ):
                                    inside_gradient = sample_gradient
                                    inside_gradient_found = 1

                    if (
                        closure_region_marker == 1
                        and inside_pressure_found == 0
                        and outside_pressure_found == 0
                        and far_pressure_inside_probe_max_multiplier > 3.0
                    ):
                        for probe_index in range(5):
                            probe_distance = probe_distance_m * (
                                3.0
                                + (far_pressure_inside_probe_max_multiplier - 3.0)
                                * (ti.cast(probe_index, ti.f32) + 1.0)
                                / 5.0
                            )
                            if inside_pressure_found == 0 or inside_gradient_found == 0:
                                inside_position = position - normal * probe_distance
                                inside_coordinate = self._grid_coordinate_from_fields(
                                    inside_position,
                                    cell_face_x_m,
                                    cell_face_y_m,
                                    cell_face_z_m,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                    nx,
                                    ny,
                                    nz,
                                )
                                _, sample_weight = self._sample_pressure_trilinear_sampling_view(
                                    pressure_field,
                                    obstacle_field,
                                    sampling_obstacle_field,
                                    use_sampling_obstacle,
                                    inside_coordinate.x,
                                    inside_coordinate.y,
                                    inside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                )
                                sample_gradient, sample_gradient_valid = (
                                    self._sample_velocity_gradient_sampling_view(
                                        velocity_field,
                                        obstacle_field,
                                        sampling_obstacle_field,
                                        use_sampling_obstacle,
                                        inside_coordinate.x,
                                        inside_coordinate.y,
                                        inside_coordinate.z,
                                        nx,
                                        ny,
                                        nz,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                    )
                                )
                                if sample_weight > 1.0e-12:
                                    if inside_pressure_found == 0:
                                        inside_pressure_found = 1
                                    if (
                                        sample_gradient_valid == 1
                                        and inside_gradient_found == 0
                                    ):
                                        inside_gradient = sample_gradient
                                        inside_gradient_found = 1

                    extended_probe_max_multiplier = two_sided_probe_max_multiplier
                    if (
                        one_sided_region_marker == 1
                        and one_sided_probe_max_multiplier
                        > extended_probe_max_multiplier
                    ):
                        extended_probe_max_multiplier = one_sided_probe_max_multiplier
                    if (
                        closure_region_marker == 0
                        and inside_pressure_found == 0
                        and outside_pressure_found == 0
                        and extended_probe_max_multiplier > 3.0
                    ):
                        outside_crossed_solid = 0
                        inside_crossed_solid = 0
                        for extension_index in range(10):
                            rung_distance = probe_distance_m * (
                                3.0
                                + (extended_probe_max_multiplier - 3.0)
                                * (ti.cast(extension_index // 2, ti.f32) + 1.0)
                                / 5.0
                            )
                            side_is_inside = extension_index % 2
                            side_sign = 1.0
                            side_crossed = outside_crossed_solid
                            side_pressure_found = outside_pressure_found
                            side_gradient_found = outside_gradient_found
                            if side_is_inside == 1:
                                side_sign = -1.0
                                side_crossed = inside_crossed_solid
                                side_pressure_found = inside_pressure_found
                                side_gradient_found = inside_gradient_found
                            if side_crossed == 0 and (
                                side_pressure_found == 0 or side_gradient_found == 0
                            ):
                                extension_position = position + normal * (
                                    side_sign * rung_distance
                                )
                                extension_coordinate = self._grid_coordinate_from_fields(
                                    extension_position,
                                    cell_face_x_m,
                                    cell_face_y_m,
                                    cell_face_z_m,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                    nx,
                                    ny,
                                    nz,
                                )
                                near_extension_i = ti.min(
                                    ti.max(
                                        ti.floor(extension_coordinate.x + 0.5, ti.i32),
                                        0,
                                    ),
                                    nx - 1,
                                )
                                near_extension_j = ti.min(
                                    ti.max(
                                        ti.floor(extension_coordinate.y + 0.5, ti.i32),
                                        0,
                                    ),
                                    ny - 1,
                                )
                                near_extension_k = ti.min(
                                    ti.max(
                                        ti.floor(extension_coordinate.z + 0.5, ti.i32),
                                        0,
                                    ),
                                    nz - 1,
                                )
                                if (
                                    obstacle_field[
                                        near_extension_i,
                                        near_extension_j,
                                        near_extension_k,
                                    ]
                                    != 0
                                ):
                                    if use_sampling_obstacle == 0:
                                        if side_is_inside == 1:
                                            inside_crossed_solid = 1
                                        else:
                                            outside_crossed_solid = 1
                                else:
                                    _, sample_weight = self._sample_pressure_trilinear_sampling_view(
                                        pressure_field,
                                        obstacle_field,
                                        sampling_obstacle_field,
                                        use_sampling_obstacle,
                                        extension_coordinate.x,
                                        extension_coordinate.y,
                                        extension_coordinate.z,
                                        nx,
                                        ny,
                                        nz,
                                    )
                                    sample_gradient, sample_gradient_valid = (
                                        self._sample_velocity_gradient_sampling_view(
                                            velocity_field,
                                            obstacle_field,
                                            sampling_obstacle_field,
                                            use_sampling_obstacle,
                                            extension_coordinate.x,
                                            extension_coordinate.y,
                                            extension_coordinate.z,
                                            nx,
                                            ny,
                                            nz,
                                            cell_center_x_m,
                                            cell_center_y_m,
                                            cell_center_z_m,
                                        )
                                    )
                                    if sample_weight > 1.0e-12:
                                        if side_is_inside == 1:
                                            if inside_pressure_found == 0:
                                                inside_pressure_found = 1
                                            if (
                                                sample_gradient_valid == 1
                                                and inside_gradient_found == 0
                                            ):
                                                inside_gradient = sample_gradient
                                                inside_gradient_found = 1
                                        else:
                                            if outside_pressure_found == 0:
                                                outside_pressure_found = 1
                                            if (
                                                sample_gradient_valid == 1
                                                and outside_gradient_found == 0
                                            ):
                                                outside_gradient = sample_gradient
                                                outside_gradient_found = 1

                    if (
                        closure_region_marker == 0
                        and one_sided_region_marker == 1
                        and inside_pressure_found == 1
                    ):
                        gradient = -inside_gradient
                        if inside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_one_sided_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif (
                        closure_region_marker == 0
                        and one_sided_region_marker == 1
                        and outside_pressure_found == 1
                    ):
                        gradient = outside_gradient
                        if outside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_one_sided_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif (
                        closure_region_marker == 0
                        and outside_pressure_found == 1
                        and inside_pressure_found == 1
                    ):
                        gradient = outside_gradient - inside_gradient
                    elif closure_region_marker == 1 and inside_pressure_found == 1:
                        gradient = -inside_gradient
                        if (
                            (
                                inside_pressure_found == 1
                                and inside_gradient_found == 0
                            )
                            or (
                                outside_pressure_found == 1
                                and outside_gradient_found == 0
                            )
                        ):
                            ti.atomic_add(
                                self.report_stress_closure_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif closure_region_marker == 1 and outside_pressure_found == 1:
                        gradient = outside_gradient
                        if (
                            (
                                inside_pressure_found == 1
                                and inside_gradient_found == 0
                            )
                            or (
                                outside_pressure_found == 1
                                and outside_gradient_found == 0
                            )
                        ):
                            ti.atomic_add(
                                self.report_stress_closure_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    else:
                        gradient = ti.Matrix.zero(ti.f32, 3, 3)
                else:
                    gradient, gradient_valid = self._sample_velocity_gradient_sampling_view(
                        velocity_field,
                        obstacle_field,
                        sampling_obstacle_field,
                        use_sampling_obstacle,
                        grid_coordinate.x,
                        grid_coordinate.y,
                        grid_coordinate.z,
                        nx,
                        ny,
                        nz,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                    )
                    if gradient_valid == 0:
                        self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                        self.t_pressure_gamma_pa[marker] = ti.Vector(
                            [0.0, 0.0, 0.0]
                        )
                        self.t_viscous_gamma_pa[marker] = ti.Vector(
                            [0.0, 0.0, 0.0]
                        )
                        self._stress_pressure_valid[marker] = 0
                        self._stress_invalid_reason_code[marker] = (
                            STRESS_INVALID_REASON_VISCOUS_GRADIENT_MISSING
                        )
                        self._stress_probe_mode[marker] = 0
                        ti.atomic_add(
                            self.report_stress_valid_marker_count[None],
                            -1,
                        )
                        ti.atomic_add(
                            self.report_stress_invalid_marker_count[None],
                            1,
                        )
                        ti.atomic_add(
                            self.report_stress_viscous_gradient_invalid_marker_count[
                                None
                            ],
                            1,
                        )

                if self._stress_pressure_valid[marker] != 0:
                    viscous_stress = viscosity_pa_s * (
                        gradient + gradient.transpose()
                    )
                    viscous_traction = viscous_stress @ normal
                    self.t_viscous_gamma_pa[marker] = viscous_traction
                    traction = self.t_pressure_gamma_pa[marker] + viscous_traction
                    self.t_gamma_pa[marker] = traction
                    ti.atomic_max(
                        self.report_stress_max_abs_traction_pa[None],
                        traction.norm(),
                    )

    @ti.kernel
    def _add_base_viscous_marker_tractions_kernel(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        viscosity_pa_s: ti.f32,
    ):
        self.report_stress_max_abs_traction_pa[None] = 0.0
        self.report_stress_viscous_gradient_invalid_marker_count[None] = 0
        self.report_stress_closure_gradient_missing_marker_count[None] = 0
        self.report_stress_one_sided_gradient_missing_marker_count[None] = 0
        for marker in range(marker_count):
            if self._stress_pressure_valid[marker] != 0:
                position = self.x_gamma_m[marker]
                normal = self.n_gamma[marker]
                grid_coordinate = self._grid_coordinate_from_fields(
                    position,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                gradient, gradient_valid = self._sample_velocity_gradient(
                    velocity_field,
                    obstacle_field,
                    grid_coordinate.x,
                    grid_coordinate.y,
                    grid_coordinate.z,
                    nx,
                    ny,
                    nz,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                )
                if gradient_valid == 0:
                    self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                    self.t_pressure_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                    self.t_viscous_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                    self._stress_pressure_valid[marker] = 0
                    self._stress_invalid_reason_code[marker] = (
                        STRESS_INVALID_REASON_VISCOUS_GRADIENT_MISSING
                    )
                    self._stress_probe_mode[marker] = 0
                    ti.atomic_add(self.report_stress_valid_marker_count[None], -1)
                    ti.atomic_add(self.report_stress_invalid_marker_count[None], 1)
                    ti.atomic_add(
                        self.report_stress_viscous_gradient_invalid_marker_count[
                            None
                        ],
                        1,
                    )
                else:
                    viscous_stress = viscosity_pa_s * (
                        gradient + gradient.transpose()
                    )
                    viscous_traction = viscous_stress @ normal
                    self.t_viscous_gamma_pa[marker] = viscous_traction
                    traction = self.t_pressure_gamma_pa[marker] + viscous_traction
                    self.t_gamma_pa[marker] = traction
                    ti.atomic_max(
                        self.report_stress_max_abs_traction_pa[None],
                        traction.norm(),
                    )

    @ti.kernel
    def _add_split_viscous_marker_tractions_kernel(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        sampling_obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        viscosity_pa_s: ti.f32,
        far_pressure_inside_probe_max_multiplier: ti.f32,
        two_sided_probe_max_multiplier: ti.f32,
        one_sided_probe_max_multiplier: ti.f32,
        use_sampling_obstacle: ti.i32,
    ):
        self.report_stress_max_abs_traction_pa[None] = 0.0
        self.report_stress_viscous_gradient_invalid_marker_count[None] = 0
        self.report_stress_closure_gradient_missing_marker_count[None] = 0
        self.report_stress_one_sided_gradient_missing_marker_count[None] = 0
        for marker in range(marker_count):
            if self._stress_pressure_valid[marker] != 0:
                position = self.x_gamma_m[marker]
                normal = self.n_gamma[marker]
                grid_coordinate = self._grid_coordinate_from_fields(
                    position,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                mode = self._stress_viscous_mode[marker]
                gradient = ti.Matrix.zero(ti.f32, 3, 3)
                gradient_valid = 1
                if mode == 1:
                    gradient, gradient_valid = self._sample_velocity_gradient_sampling_view(
                        velocity_field,
                        obstacle_field,
                        sampling_obstacle_field,
                        use_sampling_obstacle,
                        grid_coordinate.x,
                        grid_coordinate.y,
                        grid_coordinate.z,
                        nx,
                        ny,
                        nz,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                    )
                    if gradient_valid == 0:
                        self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                        self.t_pressure_gamma_pa[marker] = ti.Vector(
                            [0.0, 0.0, 0.0]
                        )
                        self.t_viscous_gamma_pa[marker] = ti.Vector(
                            [0.0, 0.0, 0.0]
                        )
                        self._stress_pressure_valid[marker] = 0
                        self._stress_invalid_reason_code[marker] = (
                            STRESS_INVALID_REASON_VISCOUS_GRADIENT_MISSING
                        )
                        self._stress_probe_mode[marker] = 0
                        ti.atomic_add(self.report_stress_valid_marker_count[None], -1)
                        ti.atomic_add(self.report_stress_invalid_marker_count[None], 1)
                        ti.atomic_add(
                            self.report_stress_viscous_gradient_invalid_marker_count[
                                None
                            ],
                            1,
                        )
                elif mode >= 2 and mode <= 6:
                    i_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0),
                        nx - 1,
                    )
                    j_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0),
                        ny - 1,
                    )
                    k_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0),
                        nz - 1,
                    )
                    normal_spacing_inv = (
                        ti.abs(normal.x) / cell_width_x_m[i_near]
                        + ti.abs(normal.y) / cell_width_y_m[j_near]
                        + ti.abs(normal.z) / cell_width_z_m[k_near]
                    )
                    probe_distance_m = 1.0 / ti.max(normal_spacing_inv, 1.0e-12)
                    outside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    inside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    outside_gradient_found = 0
                    inside_gradient_found = 0
                    need_outside = mode == 2 or mode == 4 or mode == 6
                    need_inside = mode == 2 or mode == 3 or mode == 5

                    for probe_index in range(5):
                        probe_distance = probe_distance_m * (
                            1.0 + 0.5 * ti.cast(probe_index, ti.f32)
                        )
                        if need_outside and outside_gradient_found == 0:
                            outside_coordinate = self._grid_coordinate_from_fields(
                                position + normal * probe_distance,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            sample_gradient, sample_gradient_valid = (
                                self._sample_velocity_gradient_sampling_view(
                                    velocity_field,
                                    obstacle_field,
                                    sampling_obstacle_field,
                                    use_sampling_obstacle,
                                    outside_coordinate.x,
                                    outside_coordinate.y,
                                    outside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                )
                            )
                            if sample_gradient_valid == 1:
                                outside_gradient = sample_gradient
                                outside_gradient_found = 1
                        if need_inside and inside_gradient_found == 0:
                            inside_coordinate = self._grid_coordinate_from_fields(
                                position - normal * probe_distance,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            sample_gradient, sample_gradient_valid = (
                                self._sample_velocity_gradient_sampling_view(
                                    velocity_field,
                                    obstacle_field,
                                    sampling_obstacle_field,
                                    use_sampling_obstacle,
                                    inside_coordinate.x,
                                    inside_coordinate.y,
                                    inside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                )
                            )
                            if sample_gradient_valid == 1:
                                inside_gradient = sample_gradient
                                inside_gradient_found = 1

                    if (
                        mode == 3
                        and inside_gradient_found == 0
                        and far_pressure_inside_probe_max_multiplier > 3.0
                    ):
                        for probe_index in range(5):
                            if inside_gradient_found == 0:
                                probe_distance = probe_distance_m * (
                                    3.0
                                    + (
                                        far_pressure_inside_probe_max_multiplier
                                        - 3.0
                                    )
                                    * (ti.cast(probe_index, ti.f32) + 1.0)
                                    / 5.0
                                )
                                inside_coordinate = self._grid_coordinate_from_fields(
                                    position - normal * probe_distance,
                                    cell_face_x_m,
                                    cell_face_y_m,
                                    cell_face_z_m,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                    nx,
                                    ny,
                                    nz,
                                )
                                sample_gradient, sample_gradient_valid = (
                                    self._sample_velocity_gradient_sampling_view(
                                        velocity_field,
                                        obstacle_field,
                                        sampling_obstacle_field,
                                        use_sampling_obstacle,
                                        inside_coordinate.x,
                                        inside_coordinate.y,
                                        inside_coordinate.z,
                                        nx,
                                        ny,
                                        nz,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                    )
                                )
                                if sample_gradient_valid == 1:
                                    inside_gradient = sample_gradient
                                    inside_gradient_found = 1

                    extended_probe_max_multiplier = two_sided_probe_max_multiplier
                    if (
                        (mode == 5 or mode == 6)
                        and one_sided_probe_max_multiplier
                        > extended_probe_max_multiplier
                    ):
                        extended_probe_max_multiplier = one_sided_probe_max_multiplier
                    if (
                        (mode == 2 or mode == 5 or mode == 6)
                        and extended_probe_max_multiplier > 3.0
                    ):
                        outside_crossed_solid = 0
                        inside_crossed_solid = 0
                        for extension_index in range(10):
                            rung_distance = probe_distance_m * (
                                3.0
                                + (extended_probe_max_multiplier - 3.0)
                                * (ti.cast(extension_index // 2, ti.f32) + 1.0)
                                / 5.0
                            )
                            side_is_inside = extension_index % 2
                            side_sign = 1.0
                            side_crossed = outside_crossed_solid
                            side_needed = need_outside
                            side_found = outside_gradient_found
                            if side_is_inside == 1:
                                side_sign = -1.0
                                side_crossed = inside_crossed_solid
                                side_needed = need_inside
                                side_found = inside_gradient_found
                            if side_needed and side_found == 0 and side_crossed == 0:
                                extension_position = position + normal * (
                                    side_sign * rung_distance
                                )
                                extension_coordinate = self._grid_coordinate_from_fields(
                                    extension_position,
                                    cell_face_x_m,
                                    cell_face_y_m,
                                    cell_face_z_m,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                    nx,
                                    ny,
                                    nz,
                                )
                                near_i = ti.min(
                                    ti.max(
                                        ti.floor(extension_coordinate.x + 0.5, ti.i32),
                                        0,
                                    ),
                                    nx - 1,
                                )
                                near_j = ti.min(
                                    ti.max(
                                        ti.floor(extension_coordinate.y + 0.5, ti.i32),
                                        0,
                                    ),
                                    ny - 1,
                                )
                                near_k = ti.min(
                                    ti.max(
                                        ti.floor(extension_coordinate.z + 0.5, ti.i32),
                                        0,
                                    ),
                                    nz - 1,
                                )
                                if obstacle_field[near_i, near_j, near_k] != 0:
                                    if use_sampling_obstacle == 0:
                                        if side_is_inside == 1:
                                            inside_crossed_solid = 1
                                        else:
                                            outside_crossed_solid = 1
                                else:
                                    sample_gradient, sample_gradient_valid = (
                                        self._sample_velocity_gradient_sampling_view(
                                            velocity_field,
                                            obstacle_field,
                                            sampling_obstacle_field,
                                            use_sampling_obstacle,
                                            extension_coordinate.x,
                                            extension_coordinate.y,
                                            extension_coordinate.z,
                                            nx,
                                            ny,
                                            nz,
                                            cell_center_x_m,
                                            cell_center_y_m,
                                            cell_center_z_m,
                                        )
                                    )
                                    if sample_gradient_valid == 1:
                                        if side_is_inside == 1:
                                            inside_gradient = sample_gradient
                                            inside_gradient_found = 1
                                        else:
                                            outside_gradient = sample_gradient
                                            outside_gradient_found = 1

                    if mode == 2:
                        gradient = outside_gradient - inside_gradient
                    elif mode == 3:
                        gradient = -inside_gradient
                        if inside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_closure_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif mode == 4:
                        gradient = outside_gradient
                        if outside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_closure_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif mode == 5:
                        gradient = -inside_gradient
                        if inside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_one_sided_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif mode == 6:
                        gradient = outside_gradient
                        if outside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_one_sided_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )

                if self._stress_pressure_valid[marker] != 0:
                    viscous_stress = viscosity_pa_s * (
                        gradient + gradient.transpose()
                    )
                    viscous_traction = viscous_stress @ normal
                    self.t_viscous_gamma_pa[marker] = viscous_traction
                    traction = self.t_pressure_gamma_pa[marker] + viscous_traction
                    self.t_gamma_pa[marker] = traction
                    ti.atomic_max(
                        self.report_stress_max_abs_traction_pa[None],
                        traction.norm(),
                    )

    @ti.kernel
    def _reset_split_viscous_marker_traction_reports_kernel(self):
        self.report_stress_max_abs_traction_pa[None] = 0.0
        self.report_stress_viscous_gradient_invalid_marker_count[None] = 0
        self.report_stress_closure_gradient_missing_marker_count[None] = 0
        self.report_stress_one_sided_gradient_missing_marker_count[None] = 0

    @ti.kernel
    def _add_split_viscous_mode_marker_tractions_kernel(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        sampling_obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        viscosity_pa_s: ti.f32,
        far_pressure_inside_probe_max_multiplier: ti.f32,
        two_sided_probe_max_multiplier: ti.f32,
        one_sided_probe_max_multiplier: ti.f32,
        use_sampling_obstacle: ti.i32,
        mode_filter: ti.template(),
    ):
        for marker in range(marker_count):
            if (
                self._stress_pressure_valid[marker] != 0
                and self._stress_viscous_mode[marker] == mode_filter
            ):
                position = self.x_gamma_m[marker]
                normal = self.n_gamma[marker]
                grid_coordinate = self._grid_coordinate_from_fields(
                    position,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                gradient = ti.Matrix.zero(ti.f32, 3, 3)
                gradient_valid = 1
                if ti.static(mode_filter == 1):
                    gradient, gradient_valid = self._sample_velocity_gradient_sampling_view(
                        velocity_field,
                        obstacle_field,
                        sampling_obstacle_field,
                        use_sampling_obstacle,
                        grid_coordinate.x,
                        grid_coordinate.y,
                        grid_coordinate.z,
                        nx,
                        ny,
                        nz,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                    )
                    if gradient_valid == 0:
                        self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                        self.t_pressure_gamma_pa[marker] = ti.Vector(
                            [0.0, 0.0, 0.0]
                        )
                        self.t_viscous_gamma_pa[marker] = ti.Vector(
                            [0.0, 0.0, 0.0]
                        )
                        self._stress_pressure_valid[marker] = 0
                        self._stress_invalid_reason_code[marker] = (
                            STRESS_INVALID_REASON_VISCOUS_GRADIENT_MISSING
                        )
                        self._stress_probe_mode[marker] = 0
                        ti.atomic_add(self.report_stress_valid_marker_count[None], -1)
                        ti.atomic_add(self.report_stress_invalid_marker_count[None], 1)
                        ti.atomic_add(
                            self.report_stress_viscous_gradient_invalid_marker_count[
                                None
                            ],
                            1,
                        )
                elif ti.static(mode_filter >= 2 and mode_filter <= 6):
                    i_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0),
                        nx - 1,
                    )
                    j_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0),
                        ny - 1,
                    )
                    k_near = ti.min(
                        ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0),
                        nz - 1,
                    )
                    normal_spacing_inv = (
                        ti.abs(normal.x) / cell_width_x_m[i_near]
                        + ti.abs(normal.y) / cell_width_y_m[j_near]
                        + ti.abs(normal.z) / cell_width_z_m[k_near]
                    )
                    probe_distance_m = 1.0 / ti.max(normal_spacing_inv, 1.0e-12)
                    outside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    inside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                    outside_gradient_found = 0
                    inside_gradient_found = 0
                    for probe_index in range(5):
                        probe_distance = probe_distance_m * (
                            1.0 + 0.5 * ti.cast(probe_index, ti.f32)
                        )
                        if ti.static(
                            mode_filter == 2
                            or mode_filter == 4
                            or mode_filter == 6
                        ):
                            if outside_gradient_found == 0:
                                outside_coordinate = self._grid_coordinate_from_fields(
                                    position + normal * probe_distance,
                                    cell_face_x_m,
                                    cell_face_y_m,
                                    cell_face_z_m,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                    nx,
                                    ny,
                                    nz,
                                )
                                sample_gradient, sample_gradient_valid = (
                                    self._sample_velocity_gradient_sampling_view(
                                        velocity_field,
                                        obstacle_field,
                                        sampling_obstacle_field,
                                        use_sampling_obstacle,
                                        outside_coordinate.x,
                                        outside_coordinate.y,
                                        outside_coordinate.z,
                                        nx,
                                        ny,
                                        nz,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                    )
                                )
                                if sample_gradient_valid == 1:
                                    outside_gradient = sample_gradient
                                    outside_gradient_found = 1
                        if ti.static(
                            mode_filter == 2
                            or mode_filter == 3
                            or mode_filter == 5
                        ):
                            if inside_gradient_found == 0:
                                inside_coordinate = self._grid_coordinate_from_fields(
                                    position - normal * probe_distance,
                                    cell_face_x_m,
                                    cell_face_y_m,
                                    cell_face_z_m,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                    nx,
                                    ny,
                                    nz,
                                )
                                sample_gradient, sample_gradient_valid = (
                                    self._sample_velocity_gradient_sampling_view(
                                        velocity_field,
                                        obstacle_field,
                                        sampling_obstacle_field,
                                        use_sampling_obstacle,
                                        inside_coordinate.x,
                                        inside_coordinate.y,
                                        inside_coordinate.z,
                                        nx,
                                        ny,
                                        nz,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                    )
                                )
                                if sample_gradient_valid == 1:
                                    inside_gradient = sample_gradient
                                    inside_gradient_found = 1

                    if ti.static(mode_filter == 3):
                        if (
                            inside_gradient_found == 0
                            and far_pressure_inside_probe_max_multiplier > 3.0
                        ):
                            for probe_index in range(5):
                                if inside_gradient_found == 0:
                                    probe_distance = probe_distance_m * (
                                        3.0
                                        + (
                                            far_pressure_inside_probe_max_multiplier
                                            - 3.0
                                        )
                                        * (ti.cast(probe_index, ti.f32) + 1.0)
                                        / 5.0
                                    )
                                    inside_coordinate = self._grid_coordinate_from_fields(
                                        position - normal * probe_distance,
                                        cell_face_x_m,
                                        cell_face_y_m,
                                        cell_face_z_m,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                        nx,
                                        ny,
                                        nz,
                                    )
                                    sample_gradient, sample_gradient_valid = (
                                        self._sample_velocity_gradient_sampling_view(
                                            velocity_field,
                                            obstacle_field,
                                            sampling_obstacle_field,
                                            use_sampling_obstacle,
                                            inside_coordinate.x,
                                            inside_coordinate.y,
                                            inside_coordinate.z,
                                            nx,
                                            ny,
                                            nz,
                                            cell_center_x_m,
                                            cell_center_y_m,
                                            cell_center_z_m,
                                        )
                                    )
                                    if sample_gradient_valid == 1:
                                        inside_gradient = sample_gradient
                                        inside_gradient_found = 1

                    if ti.static(
                        mode_filter == 2
                        or mode_filter == 5
                        or mode_filter == 6
                    ):
                        extended_probe_max_multiplier = two_sided_probe_max_multiplier
                        if (
                            ti.static(mode_filter == 5 or mode_filter == 6)
                            and one_sided_probe_max_multiplier
                            > extended_probe_max_multiplier
                        ):
                            extended_probe_max_multiplier = (
                                one_sided_probe_max_multiplier
                            )
                        if extended_probe_max_multiplier > 3.0:
                            outside_crossed_solid = 0
                            inside_crossed_solid = 0
                            for extension_index in range(10):
                                rung_distance = probe_distance_m * (
                                    3.0
                                    + (extended_probe_max_multiplier - 3.0)
                                    * (ti.cast(extension_index // 2, ti.f32) + 1.0)
                                    / 5.0
                                )
                                side_is_inside = extension_index % 2
                                side_sign = 1.0
                                side_crossed = outside_crossed_solid
                                side_found = outside_gradient_found
                                if side_is_inside == 1:
                                    side_sign = -1.0
                                    side_crossed = inside_crossed_solid
                                    side_found = inside_gradient_found
                                if side_crossed == 0 and side_found == 0:
                                    extension_position = position + normal * (
                                        side_sign * rung_distance
                                    )
                                    extension_coordinate = (
                                        self._grid_coordinate_from_fields(
                                            extension_position,
                                            cell_face_x_m,
                                            cell_face_y_m,
                                            cell_face_z_m,
                                            cell_center_x_m,
                                            cell_center_y_m,
                                            cell_center_z_m,
                                            nx,
                                            ny,
                                            nz,
                                        )
                                    )
                                    near_i = ti.min(
                                        ti.max(
                                            ti.floor(
                                                extension_coordinate.x + 0.5,
                                                ti.i32,
                                            ),
                                            0,
                                        ),
                                        nx - 1,
                                    )
                                    near_j = ti.min(
                                        ti.max(
                                            ti.floor(
                                                extension_coordinate.y + 0.5,
                                                ti.i32,
                                            ),
                                            0,
                                        ),
                                        ny - 1,
                                    )
                                    near_k = ti.min(
                                        ti.max(
                                            ti.floor(
                                                extension_coordinate.z + 0.5,
                                                ti.i32,
                                            ),
                                            0,
                                        ),
                                        nz - 1,
                                    )
                                    if obstacle_field[near_i, near_j, near_k] != 0:
                                        if use_sampling_obstacle == 0:
                                            if side_is_inside == 1:
                                                inside_crossed_solid = 1
                                            else:
                                                outside_crossed_solid = 1
                                    else:
                                        sample_gradient, sample_gradient_valid = (
                                            self._sample_velocity_gradient_sampling_view(
                                                velocity_field,
                                                obstacle_field,
                                                sampling_obstacle_field,
                                                use_sampling_obstacle,
                                                extension_coordinate.x,
                                                extension_coordinate.y,
                                                extension_coordinate.z,
                                                nx,
                                                ny,
                                                nz,
                                                cell_center_x_m,
                                                cell_center_y_m,
                                                cell_center_z_m,
                                            )
                                        )
                                        if sample_gradient_valid == 1:
                                            if side_is_inside == 1:
                                                inside_gradient = sample_gradient
                                                inside_gradient_found = 1
                                            else:
                                                outside_gradient = sample_gradient
                                                outside_gradient_found = 1

                    if ti.static(mode_filter == 2):
                        gradient = outside_gradient - inside_gradient
                    elif ti.static(mode_filter == 3):
                        gradient = -inside_gradient
                        if inside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_closure_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif ti.static(mode_filter == 4):
                        gradient = outside_gradient
                        if outside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_closure_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif ti.static(mode_filter == 5):
                        gradient = -inside_gradient
                        if inside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_one_sided_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )
                    elif ti.static(mode_filter == 6):
                        gradient = outside_gradient
                        if outside_gradient_found == 0:
                            ti.atomic_add(
                                self.report_stress_one_sided_gradient_missing_marker_count[
                                    None
                                ],
                                1,
                            )

                if self._stress_pressure_valid[marker] != 0:
                    viscous_stress = viscosity_pa_s * (
                        gradient + gradient.transpose()
                    )
                    viscous_traction = viscous_stress @ normal
                    self.t_viscous_gamma_pa[marker] = viscous_traction
                    traction = self.t_pressure_gamma_pa[marker] + viscous_traction
                    self.t_gamma_pa[marker] = traction
                    ti.atomic_max(
                        self.report_stress_max_abs_traction_pa[None],
                        traction.norm(),
                    )

    def sample_fluid_stress_to_marker_tractions(
        self,
        velocity_field,
        pressure_field,
        obstacle_field,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        cell_width_x_m,
        cell_width_y_m,
        cell_width_z_m,
        grid_nodes: tuple[int, int, int],
        *,
        viscosity_pa_s: float,
        two_sided_pressure: bool = False,
        far_pressure_region_id: int = -1,
        far_pressure_pa: float = 0.0,
        far_pressure_side_normal_sign: float = 0.0,
        far_pressure_inside_probe_max_multiplier: float = 3.0,
        two_sided_probe_max_multiplier: float = 3.0,
        one_sided_pressure_region_id: int = -1,
        one_sided_reference_pressure_pa: float = 0.0,
        one_sided_pressure_primary_region_id: int = -1,
        one_sided_pressure_secondary_region_id: int = -1,
        one_sided_primary_reference_pressure_pa: float = 0.0,
        one_sided_secondary_reference_pressure_pa: float = 0.0,
        one_sided_primary_fluid_side_normal_sign: float = 0.0,
        one_sided_secondary_fluid_side_normal_sign: float = 0.0,
        one_sided_probe_max_multiplier: float = 3.0,
        use_pressure_anchor_fallback: bool = False,
        node_anchor_cell=None,
        sampling_obstacle_field=None,
        pressure_probe_ladder_start_offset_cells: float | None = None,
        pressure_probe_ladder_spacing_cells: float = 0.5,
        pressure_probe_ladder_rung_count: int = 5,
        pressure_probe_ladder_mode: str = STRESS_PRESSURE_PROBE_LADDER_MODE_CURRENT_NORMAL_CELL,
        pressure_pair_policy: str = STRESS_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER,
        pressure_pair_max_cell_delta: int = 1,
        pressure_pair_require_opposite_sides: bool = True,
    ) -> HibmMpmFluidStressSampleReport:
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        node_anchor_was_none = node_anchor_cell is None
        sampling_obstacle_was_none = sampling_obstacle_field is None
        # S2-A7: the node-level anchor field is optional. Callers without
        # one (or callers that never arm use_pressure_anchor_fallback)
        # get the never-indexed 1x1x1 stand-in plus a hard availability
        # gate, so the second-stage fallback branch short-circuits and
        # the existing paths are reproduced bit for bit.
        node_anchor_field = node_anchor_cell
        node_anchor_available = 1
        if node_anchor_field is None:
            node_anchor_field = self._node_anchor_cell_unset
            node_anchor_available = 0
        # S2-A8'': the dedicated sampling view is optional. None (the
        # default) binds the never-indexed 1x1x1 stand-in with the
        # runtime gate off, so every pressure / velocity-gradient sample
        # reads the projection obstacle view exactly as before (bit for
        # bit). A provided view replaces the obstacle argument of every
        # _sample_pressure_trilinear / _sample_velocity_gradient site in
        # the stress kernel - and ONLY there; the no-slip residual, the
        # Neumann gradient sampling and all other consumers keep the
        # projection obstacle view.
        sampling_obstacle = sampling_obstacle_field
        use_sampling_obstacle = 1
        if sampling_obstacle is None:
            sampling_obstacle = self._sampling_obstacle_unset
            use_sampling_obstacle = 0
        elif tuple(sampling_obstacle.shape) != nodes:
            raise ValueError(
                "sampling_obstacle_field shape "
                f"{tuple(sampling_obstacle.shape)} does not match grid_nodes "
                f"{nodes}"
            )
        viscosity = float(viscosity_pa_s)
        if not math.isfinite(viscosity) or viscosity < 0.0:
            raise ValueError("viscosity_pa_s must be a finite non-negative number")
        far_region_id = int(far_pressure_region_id)
        far_pressure = float(far_pressure_pa)
        if not math.isfinite(far_pressure):
            raise ValueError("far_pressure_pa must be a finite number")
        far_pressure_side = float(far_pressure_side_normal_sign)
        if far_pressure_side not in (-1.0, 0.0, 1.0):
            raise ValueError(
                "far_pressure_side_normal_sign must be -1.0, 0.0, or 1.0"
            )
        far_inside_probe_max = float(far_pressure_inside_probe_max_multiplier)
        if not math.isfinite(far_inside_probe_max) or far_inside_probe_max < 3.0:
            raise ValueError(
                "far_pressure_inside_probe_max_multiplier must be finite and >= 3.0"
            )
        two_sided_probe_max = float(two_sided_probe_max_multiplier)
        if not math.isfinite(two_sided_probe_max) or two_sided_probe_max < 3.0:
            raise ValueError(
                "two_sided_probe_max_multiplier must be finite and >= 3.0"
            )
        one_sided_region_id = int(one_sided_pressure_region_id)
        one_sided_reference_pressure = float(one_sided_reference_pressure_pa)
        if not math.isfinite(one_sided_reference_pressure):
            raise ValueError("one_sided_reference_pressure_pa must be a finite number")
        one_sided_primary_region_id = int(one_sided_pressure_primary_region_id)
        one_sided_secondary_region_id = int(one_sided_pressure_secondary_region_id)
        one_sided_primary_reference_pressure = float(
            one_sided_primary_reference_pressure_pa
        )
        one_sided_secondary_reference_pressure = float(
            one_sided_secondary_reference_pressure_pa
        )
        if not math.isfinite(one_sided_primary_reference_pressure):
            raise ValueError(
                "one_sided_primary_reference_pressure_pa must be a finite number"
            )
        if not math.isfinite(one_sided_secondary_reference_pressure):
            raise ValueError(
                "one_sided_secondary_reference_pressure_pa must be a finite number"
            )
        one_sided_primary_side = float(one_sided_primary_fluid_side_normal_sign)
        one_sided_secondary_side = float(one_sided_secondary_fluid_side_normal_sign)
        if one_sided_primary_side not in (-1.0, 0.0, 1.0):
            raise ValueError(
                "one_sided_primary_fluid_side_normal_sign must be -1.0, 0.0, or 1.0"
            )
        if one_sided_secondary_side not in (-1.0, 0.0, 1.0):
            raise ValueError(
                "one_sided_secondary_fluid_side_normal_sign must be -1.0, 0.0, or 1.0"
            )
        per_face_one_sided_configured = (
            one_sided_primary_region_id >= 0 or one_sided_secondary_region_id >= 0
        )
        if per_face_one_sided_configured:
            if one_sided_region_id >= 0:
                raise ValueError(
                    "legacy one_sided_pressure_region_id cannot be combined with "
                    "per-face one-sided pressure regions"
                )
            if one_sided_primary_region_id >= 0 and one_sided_primary_side == 0.0:
                raise ValueError(
                    "one_sided_primary_fluid_side_normal_sign must be -1.0 or 1.0 "
                    "when one_sided_pressure_primary_region_id is set"
                )
            if one_sided_secondary_region_id >= 0 and one_sided_secondary_side == 0.0:
                raise ValueError(
                    "one_sided_secondary_fluid_side_normal_sign must be -1.0 or 1.0 "
                    "when one_sided_pressure_secondary_region_id is set"
                )
        one_sided_probe_max = float(one_sided_probe_max_multiplier)
        if not math.isfinite(one_sided_probe_max) or one_sided_probe_max < 3.0:
            raise ValueError(
                "one_sided_probe_max_multiplier must be finite and >= 3.0"
            )
        if (
            str(pressure_probe_ladder_mode)
            != STRESS_PRESSURE_PROBE_LADDER_MODE_CURRENT_NORMAL_CELL
        ):
            raise ValueError(
                "pressure_probe_ladder_mode must be "
                f"{STRESS_PRESSURE_PROBE_LADDER_MODE_CURRENT_NORMAL_CELL!r}"
            )
        pressure_probe_ladder_configured = (
            pressure_probe_ladder_start_offset_cells is not None
        )
        if pressure_probe_ladder_configured:
            pressure_probe_ladder_start = float(
                pressure_probe_ladder_start_offset_cells
            )
            if (
                not math.isfinite(pressure_probe_ladder_start)
                or pressure_probe_ladder_start < 0.0
            ):
                raise ValueError(
                    "pressure_probe_ladder_start_offset_cells must be finite "
                    "and non-negative"
                )
        else:
            pressure_probe_ladder_start = 1.0
        pressure_probe_ladder_spacing = float(pressure_probe_ladder_spacing_cells)
        if (
            not math.isfinite(pressure_probe_ladder_spacing)
            or pressure_probe_ladder_spacing <= 0.0
        ):
            raise ValueError(
                "pressure_probe_ladder_spacing_cells must be finite and positive"
            )
        pressure_probe_ladder_count = int(pressure_probe_ladder_rung_count)
        if pressure_probe_ladder_count <= 0:
            raise ValueError("pressure_probe_ladder_rung_count must be positive")
        if not pressure_probe_ladder_configured:
            pressure_probe_ladder_spacing = 1.0
            pressure_probe_ladder_count = 3
        pressure_pair_policy_name = str(pressure_pair_policy)
        if pressure_pair_policy_name not in STRESS_PRESSURE_PAIR_POLICY_CODES:
            raise ValueError(
                f"unsupported pressure_pair_policy: {pressure_pair_policy_name!r}"
            )
        pressure_pair_policy_code = STRESS_PRESSURE_PAIR_POLICY_CODES[
            pressure_pair_policy_name
        ]
        pressure_pair_max_delta = int(pressure_pair_max_cell_delta)
        if pressure_pair_max_delta < 0:
            raise ValueError("pressure_pair_max_cell_delta must be non-negative")
        pressure_pair_require_opposite_sides_flag = (
            1 if bool(pressure_pair_require_opposite_sides) else 0
        )
        pressure_only_fast_path = (
            viscosity == 0.0
            and
            far_region_id < 0
            and not bool(use_pressure_anchor_fallback)
            and node_anchor_was_none
            and sampling_obstacle_was_none
        )
        base_viscous_split_path = (
            viscosity > 0.0
            and not bool(two_sided_pressure)
            and far_region_id < 0
            and one_sided_region_id < 0
            and not bool(use_pressure_anchor_fallback)
            and node_anchor_was_none
            and sampling_obstacle_was_none
        )
        split_viscous_path = viscosity > 0.0 and bool(two_sided_pressure)
        if pressure_probe_ladder_configured and not (
            pressure_only_fast_path or base_viscous_split_path
        ):
            raise ValueError(
                "pressure_probe_ladder controls are pressure-only diagnostics"
            )
        if per_face_one_sided_configured and not pressure_only_fast_path:
            raise ValueError(
                "per-face one-sided pressure controls are pressure-only diagnostics"
            )
        if pressure_pair_policy_code != 0 and not (
            pressure_only_fast_path
            and bool(two_sided_pressure)
            and one_sided_region_id < 0
        ):
            raise ValueError(
                "pressure_pair_policy controls are pressure-only two-sided diagnostics"
            )
        if pressure_only_fast_path:
            _debug_stage_progress("stress_sampling:pressure_only:start")
            self._sample_pressure_only_marker_tractions_kernel(
                pressure_field,
                obstacle_field,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                cell_width_x_m,
                cell_width_y_m,
                cell_width_z_m,
                int(self.marker_count),
                int(nodes[0]),
                int(nodes[1]),
                int(nodes[2]),
                1 if bool(two_sided_pressure) else 0,
                one_sided_region_id,
                one_sided_reference_pressure,
                one_sided_primary_region_id,
                one_sided_secondary_region_id,
                one_sided_primary_reference_pressure,
                one_sided_secondary_reference_pressure,
                one_sided_primary_side,
                one_sided_secondary_side,
                pressure_probe_ladder_start,
                pressure_probe_ladder_spacing,
                pressure_probe_ladder_count,
                1 if pressure_probe_ladder_configured else 0,
                pressure_pair_policy_code,
                pressure_pair_max_delta,
                pressure_pair_require_opposite_sides_flag,
            )
            _debug_stage_progress("stress_sampling:pressure_only:done")
        elif base_viscous_split_path:
            _debug_stage_progress("stress_sampling:base_pressure:start")
            self._sample_pressure_only_marker_tractions_kernel(
                pressure_field,
                obstacle_field,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                cell_width_x_m,
                cell_width_y_m,
                cell_width_z_m,
                int(self.marker_count),
                int(nodes[0]),
                int(nodes[1]),
                int(nodes[2]),
                0,
                -1,
                0.0,
                -1,
                -1,
                0.0,
                0.0,
                0.0,
                0.0,
                pressure_probe_ladder_start,
                pressure_probe_ladder_spacing,
                pressure_probe_ladder_count,
                1 if pressure_probe_ladder_configured else 0,
                0,
                1,
                1,
            )
            _debug_stage_progress("stress_sampling:base_pressure:done")
            _debug_stage_progress("stress_sampling:base_viscous:start")
            self._add_base_viscous_marker_tractions_kernel(
                velocity_field,
                obstacle_field,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                int(self.marker_count),
                int(nodes[0]),
                int(nodes[1]),
                int(nodes[2]),
                viscosity,
            )
            _debug_stage_progress("stress_sampling:base_viscous:done")
        elif split_viscous_path:
            _debug_stage_progress("stress_sampling:split_pressure:start")
            self._sample_fluid_stress_to_marker_tractions_kernel(
                velocity_field,
                pressure_field,
                obstacle_field,
                sampling_obstacle,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                cell_width_x_m,
                cell_width_y_m,
                cell_width_z_m,
                self.marker_pressure_anchor_cell,
                node_anchor_field,
                int(self.marker_count),
                int(nodes[0]),
                int(nodes[1]),
                int(nodes[2]),
                0.0,
                1,
                far_region_id,
                far_pressure,
                far_pressure_side,
                far_inside_probe_max,
                two_sided_probe_max,
                one_sided_region_id,
                one_sided_reference_pressure,
                one_sided_probe_max,
                1 if bool(use_pressure_anchor_fallback) else 0,
                node_anchor_available,
                use_sampling_obstacle,
            )
            _debug_stage_progress("stress_sampling:split_pressure:done")
            _debug_stage_progress("stress_sampling:split_reset_reports:start")
            self._reset_split_viscous_marker_traction_reports_kernel()
            _debug_stage_progress("stress_sampling:split_reset_reports:done")
            _debug_stage_progress("stress_sampling:split_viscous_mode_2:start")
            self._add_split_viscous_mode_marker_tractions_kernel(
                velocity_field,
                obstacle_field,
                sampling_obstacle,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                cell_width_x_m,
                cell_width_y_m,
                cell_width_z_m,
                int(self.marker_count),
                int(nodes[0]),
                int(nodes[1]),
                int(nodes[2]),
                viscosity,
                far_inside_probe_max,
                two_sided_probe_max,
                one_sided_probe_max,
                use_sampling_obstacle,
                2,
            )
            _debug_stage_progress("stress_sampling:split_viscous_mode_2:done")
            if far_region_id != -1:
                _debug_stage_progress("stress_sampling:split_viscous_mode_3:start")
                self._add_split_viscous_mode_marker_tractions_kernel(
                    velocity_field,
                    obstacle_field,
                    sampling_obstacle,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    cell_width_x_m,
                    cell_width_y_m,
                    cell_width_z_m,
                    int(self.marker_count),
                    int(nodes[0]),
                    int(nodes[1]),
                    int(nodes[2]),
                    viscosity,
                    far_inside_probe_max,
                    two_sided_probe_max,
                    one_sided_probe_max,
                    use_sampling_obstacle,
                    3,
                )
                _debug_stage_progress("stress_sampling:split_viscous_mode_3:done")
                _debug_stage_progress("stress_sampling:split_viscous_mode_4:start")
                self._add_split_viscous_mode_marker_tractions_kernel(
                    velocity_field,
                    obstacle_field,
                    sampling_obstacle,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    cell_width_x_m,
                    cell_width_y_m,
                    cell_width_z_m,
                    int(self.marker_count),
                    int(nodes[0]),
                    int(nodes[1]),
                    int(nodes[2]),
                    viscosity,
                    far_inside_probe_max,
                    two_sided_probe_max,
                    one_sided_probe_max,
                    use_sampling_obstacle,
                    4,
                )
                _debug_stage_progress("stress_sampling:split_viscous_mode_4:done")
            if one_sided_region_id != -1:
                _debug_stage_progress("stress_sampling:split_viscous_mode_5:start")
                self._add_split_viscous_mode_marker_tractions_kernel(
                    velocity_field,
                    obstacle_field,
                    sampling_obstacle,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    cell_width_x_m,
                    cell_width_y_m,
                    cell_width_z_m,
                    int(self.marker_count),
                    int(nodes[0]),
                    int(nodes[1]),
                    int(nodes[2]),
                    viscosity,
                    far_inside_probe_max,
                    two_sided_probe_max,
                    one_sided_probe_max,
                    use_sampling_obstacle,
                    5,
                )
                _debug_stage_progress("stress_sampling:split_viscous_mode_5:done")
                _debug_stage_progress("stress_sampling:split_viscous_mode_6:start")
                self._add_split_viscous_mode_marker_tractions_kernel(
                    velocity_field,
                    obstacle_field,
                    sampling_obstacle,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    cell_width_x_m,
                    cell_width_y_m,
                    cell_width_z_m,
                    int(self.marker_count),
                    int(nodes[0]),
                    int(nodes[1]),
                    int(nodes[2]),
                    viscosity,
                    far_inside_probe_max,
                    two_sided_probe_max,
                    one_sided_probe_max,
                    use_sampling_obstacle,
                    6,
                )
                _debug_stage_progress("stress_sampling:split_viscous_mode_6:done")
            if bool(use_pressure_anchor_fallback):
                _debug_stage_progress("stress_sampling:split_viscous_mode_7:start")
                self._add_split_viscous_mode_marker_tractions_kernel(
                    velocity_field,
                    obstacle_field,
                    sampling_obstacle,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    cell_width_x_m,
                    cell_width_y_m,
                    cell_width_z_m,
                    int(self.marker_count),
                    int(nodes[0]),
                    int(nodes[1]),
                    int(nodes[2]),
                    viscosity,
                    far_inside_probe_max,
                    two_sided_probe_max,
                    one_sided_probe_max,
                    use_sampling_obstacle,
                    7,
                )
                _debug_stage_progress("stress_sampling:split_viscous_mode_7:done")
        else:
            _debug_stage_progress("stress_sampling:full:start")
            self._sample_fluid_stress_to_marker_tractions_kernel(
                velocity_field,
                pressure_field,
                obstacle_field,
                sampling_obstacle,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                cell_width_x_m,
                cell_width_y_m,
                cell_width_z_m,
                self.marker_pressure_anchor_cell,
                node_anchor_field,
                int(self.marker_count),
                int(nodes[0]),
                int(nodes[1]),
                int(nodes[2]),
                viscosity,
                1 if bool(two_sided_pressure) else 0,
                far_region_id,
                far_pressure,
                far_pressure_side,
                far_inside_probe_max,
                two_sided_probe_max,
                one_sided_region_id,
                one_sided_reference_pressure,
                one_sided_probe_max,
                1 if bool(use_pressure_anchor_fallback) else 0,
                node_anchor_available,
                use_sampling_obstacle,
            )
            _debug_stage_progress("stress_sampling:full:done")
        return HibmMpmFluidStressSampleReport(
            valid_marker_count=int(self.report_stress_valid_marker_count[None]),
            invalid_marker_count=int(self.report_stress_invalid_marker_count[None]),
            max_abs_traction_pa=float(
                self.report_stress_max_abs_traction_pa[None]
            ),
            two_sided_pressure_marker_count=int(
                self.report_stress_two_sided_pressure_marker_count[None]
            ),
            viscous_gradient_invalid_marker_count=int(
                self.report_stress_viscous_gradient_invalid_marker_count[None]
            ),
            far_pressure_closed_marker_count=int(
                self.report_stress_far_pressure_closed_marker_count[None]
            ),
            far_pressure_closed_extended_marker_count=int(
                self.report_stress_far_pressure_closed_extended_marker_count[None]
            ),
            far_pressure_anchor_closed_marker_count=int(
                self.report_stress_far_pressure_anchor_closed_marker_count[None]
            ),
            far_pressure_node_anchor_closed_marker_count=int(
                self.report_stress_far_pressure_node_anchor_closed_marker_count[
                    None
                ]
            ),
            closure_gradient_missing_marker_count=int(
                self.report_stress_closure_gradient_missing_marker_count[None]
            ),
            far_pressure_outside_suppressed_marker_count=int(
                self.report_stress_far_pressure_outside_suppressed_marker_count[
                    None
                ]
            ),
            two_sided_extended_marker_count=int(
                self.report_stress_two_sided_extended_marker_count[None]
            ),
            one_sided_pressure_marker_count=int(
                self.report_stress_one_sided_pressure_marker_count[None]
            ),
            one_sided_extended_marker_count=int(
                self.report_stress_one_sided_extended_marker_count[None]
            ),
            one_sided_gradient_missing_marker_count=int(
                self.report_stress_one_sided_gradient_missing_marker_count[None]
            ),
            marker_diagnostics=self.stress_marker_diagnostics(),
        )

    def stress_marker_diagnostics(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        marker_count = int(self.marker_count)
        if limit is not None:
            marker_count = min(marker_count, max(0, int(limit)))
        valid = self._stress_pressure_valid.to_numpy()[:marker_count]
        base_pressure_found = self._stress_base_pressure_found.to_numpy()[:marker_count]
        inside_pressure_found = self._stress_inside_pressure_found.to_numpy()[
            :marker_count
        ]
        outside_pressure_found = self._stress_outside_pressure_found.to_numpy()[
            :marker_count
        ]
        marker_anchor_available = self._stress_marker_anchor_available.to_numpy()[
            :marker_count
        ]
        invalid_reason_code = self._stress_invalid_reason_code.to_numpy()[
            :marker_count
        ]
        base_pressure_pa = self._stress_base_pressure_pa.to_numpy()[:marker_count]
        inside_pressure_pa = self._stress_inside_pressure_pa.to_numpy()[:marker_count]
        outside_pressure_pa = self._stress_outside_pressure_pa.to_numpy()[:marker_count]
        pressure_jump_pa = self._stress_pressure_jump_pa.to_numpy()[:marker_count]
        fluid_side_pressure_pa = self._stress_fluid_side_pressure_pa.to_numpy()[
            :marker_count
        ]
        reference_pressure_pa = self._stress_reference_pressure_pa.to_numpy()[
            :marker_count
        ]
        inside_probe_rung = self._stress_inside_probe_rung.to_numpy()[:marker_count]
        outside_probe_rung = self._stress_outside_probe_rung.to_numpy()[:marker_count]
        inside_probe_distance_m = self._stress_inside_probe_distance_m.to_numpy()[
            :marker_count
        ]
        outside_probe_distance_m = self._stress_outside_probe_distance_m.to_numpy()[
            :marker_count
        ]
        inside_probe_cell = self._stress_inside_probe_cell.to_numpy()[:marker_count]
        outside_probe_cell = self._stress_outside_probe_cell.to_numpy()[:marker_count]
        inside_probe_grid_coordinate = (
            self._stress_inside_probe_grid_coordinate.to_numpy()[:marker_count]
        )
        outside_probe_grid_coordinate = (
            self._stress_outside_probe_grid_coordinate.to_numpy()[:marker_count]
        )
        inside_probe_fluid_weight = self._stress_inside_probe_fluid_weight.to_numpy()[
            :marker_count
        ]
        outside_probe_fluid_weight = self._stress_outside_probe_fluid_weight.to_numpy()[
            :marker_count
        ]
        inside_probe_multiplier = self._stress_inside_probe_multiplier.to_numpy()[
            :marker_count
        ]
        outside_probe_multiplier = self._stress_outside_probe_multiplier.to_numpy()[
            :marker_count
        ]
        inside_probe_ladder_mode = (
            self._stress_inside_probe_ladder_mode.to_numpy()[:marker_count]
        )
        outside_probe_ladder_mode = (
            self._stress_outside_probe_ladder_mode.to_numpy()[:marker_count]
        )
        probe_mode = self._stress_probe_mode.to_numpy()[:marker_count]
        pressure_pair_policy_code = (
            self._stress_pressure_pair_policy_code.to_numpy()[:marker_count]
        )
        pressure_pair_selected = (
            self._stress_pressure_pair_selected.to_numpy()[:marker_count]
        )
        pressure_pair_fallback_used = (
            self._stress_pressure_pair_fallback_used.to_numpy()[:marker_count]
        )
        pressure_pair_inside_cell = (
            self._stress_pressure_pair_inside_cell.to_numpy()[:marker_count]
        )
        pressure_pair_outside_cell = (
            self._stress_pressure_pair_outside_cell.to_numpy()[:marker_count]
        )
        pressure_pair_cell_delta = (
            self._stress_pressure_pair_cell_delta.to_numpy()[:marker_count]
        )
        pressure_pair_symmetry_residual_cells = (
            self._stress_pressure_pair_symmetry_residual_cells.to_numpy()[:marker_count]
        )
        pressure_pair_anchor_active = self.pressure_pair_anchor_active.to_numpy()[
            :marker_count
        ]
        pressure_pair_anchor_inside_cell = (
            self.pressure_pair_anchor_inside_cell.to_numpy()[:marker_count]
        )
        pressure_pair_anchor_outside_cell = (
            self.pressure_pair_anchor_outside_cell.to_numpy()[:marker_count]
        )
        pressure_pair_anchor_fallback_used = (
            self._stress_pressure_pair_anchor_fallback_used.to_numpy()[:marker_count]
        )
        one_sided_policy_code = self._stress_one_sided_policy_code.to_numpy()[
            :marker_count
        ]
        one_sided_region_id = self._stress_one_sided_region_id.to_numpy()[
            :marker_count
        ]
        one_sided_side_normal_sign = (
            self._stress_one_sided_side_normal_sign.to_numpy()[:marker_count]
        )
        one_sided_anchor_selected = (
            self._stress_one_sided_anchor_selected.to_numpy()[:marker_count]
        )
        one_sided_anchor_fallback_used = (
            self._stress_one_sided_anchor_fallback_used.to_numpy()[:marker_count]
        )
        pressure_traction = self.t_pressure_gamma_pa.to_numpy()[:marker_count]
        viscous_traction = self.t_viscous_gamma_pa.to_numpy()[:marker_count]
        total_traction = self.t_gamma_pa.to_numpy()[:marker_count]
        positions = self.x_gamma_m.to_numpy()[:marker_count]
        probe_origins = self.pressure_probe_origin_m.to_numpy()[:marker_count]
        probe_origin_explicit = self.pressure_probe_origin_explicit.to_numpy()[
            :marker_count
        ]
        normals = self.n_gamma.to_numpy()[:marker_count]
        regions = self.region_id.to_numpy()[:marker_count]
        diagnostics: list[dict[str, Any]] = []
        for marker in range(marker_count):
            reason_code = int(invalid_reason_code[marker])
            explicit_probe_origin = bool(int(probe_origin_explicit[marker]))
            probe_origin = (
                probe_origins[marker] if explicit_probe_origin else positions[marker]
            )
            pressure_traction_vector = [
                float(value) for value in pressure_traction[marker]
            ]
            viscous_traction_vector = [
                float(value) for value in viscous_traction[marker]
            ]
            total_traction_vector = [float(value) for value in total_traction[marker]]
            traction_decomposition_residual = float(
                np.linalg.norm(
                    total_traction[marker]
                    - pressure_traction[marker]
                    - viscous_traction[marker]
                )
            )
            mode_code = int(probe_mode[marker])
            fluid_side_pressure_defined = mode_code not in (0, 2)
            inside_ladder_code = int(inside_probe_ladder_mode[marker])
            outside_ladder_code = int(outside_probe_ladder_mode[marker])
            pair_policy_code = int(pressure_pair_policy_code[marker])
            one_sided_code = int(one_sided_policy_code[marker])
            one_sided_sign = float(one_sided_side_normal_sign[marker])
            one_sided_side_selected = ""
            if one_sided_sign < 0.0:
                one_sided_side_selected = "inside"
            elif one_sided_sign > 0.0:
                one_sided_side_selected = "outside"
            diagnostics.append(
                {
                    "marker_index": int(marker),
                    "valid": bool(int(valid[marker])),
                    "invalid_reason_code": reason_code,
                    "invalid_reason": STRESS_INVALID_REASON_NAMES.get(
                        reason_code,
                        f"unknown_{reason_code}",
                    ),
                    "base_pressure_found": bool(int(base_pressure_found[marker])),
                    "inside_pressure_found": bool(int(inside_pressure_found[marker])),
                    "outside_pressure_found": bool(
                        int(outside_pressure_found[marker])
                    ),
                    "pressure_anchor_available": bool(
                        int(marker_anchor_available[marker])
                    ),
                    "probe_mode": STRESS_PROBE_MODE_NAMES.get(
                        mode_code,
                        f"unknown_{mode_code}",
                    ),
                    "probe_mode_code": mode_code,
                    "base_pressure_pa": float(base_pressure_pa[marker]),
                    "inside_pressure_pa": float(inside_pressure_pa[marker]),
                    "outside_pressure_pa": float(outside_pressure_pa[marker]),
                    "pressure_jump_pa": float(pressure_jump_pa[marker]),
                    "fluid_side_pressure_pa": float(fluid_side_pressure_pa[marker]),
                    "fluid_side_pressure_defined": fluid_side_pressure_defined,
                    "selected_water_pressure_pa": float(fluid_side_pressure_pa[marker]),
                    "reference_pressure_pa": float(reference_pressure_pa[marker]),
                    "inside_probe_rung": int(inside_probe_rung[marker]),
                    "outside_probe_rung": int(outside_probe_rung[marker]),
                    "inside_probe_ladder_mode": STRESS_PROBE_LADDER_MODE_NAMES.get(
                        inside_ladder_code,
                        f"unknown_{inside_ladder_code}",
                    ),
                    "inside_probe_ladder_mode_code": inside_ladder_code,
                    "outside_probe_ladder_mode": STRESS_PROBE_LADDER_MODE_NAMES.get(
                        outside_ladder_code,
                        f"unknown_{outside_ladder_code}",
                    ),
                    "outside_probe_ladder_mode_code": outside_ladder_code,
                    "pressure_pair_policy": STRESS_PRESSURE_PAIR_POLICY_NAMES.get(
                        pair_policy_code,
                        f"unknown_{pair_policy_code}",
                    ),
                    "pressure_pair_selected": bool(
                        int(pressure_pair_selected[marker])
                    ),
                    "pressure_pair_fallback_used": bool(
                        int(pressure_pair_fallback_used[marker])
                    ),
                    "pressure_pair_inside_cell": [
                        int(value) for value in pressure_pair_inside_cell[marker]
                    ],
                    "pressure_pair_outside_cell": [
                        int(value) for value in pressure_pair_outside_cell[marker]
                    ],
                    "pressure_pair_cell_delta": int(
                        pressure_pair_cell_delta[marker]
                    ),
                    "pressure_pair_symmetry_residual_cells": float(
                        pressure_pair_symmetry_residual_cells[marker]
                    ),
                    "pressure_pair_anchor_active": bool(
                        int(pressure_pair_anchor_active[marker])
                    ),
                    "pressure_pair_anchor_inside_cell": [
                        int(value) for value in pressure_pair_anchor_inside_cell[marker]
                    ],
                    "pressure_pair_anchor_outside_cell": [
                        int(value) for value in pressure_pair_anchor_outside_cell[marker]
                    ],
                    "pressure_pair_anchor_source": (
                        "api"
                        if bool(int(pressure_pair_anchor_active[marker]))
                        else "unset"
                    ),
                    "pressure_pair_anchor_fallback_used": bool(
                        int(pressure_pair_anchor_fallback_used[marker])
                    ),
                    "one_sided_policy": STRESS_ONE_SIDED_POLICY_NAMES.get(
                        one_sided_code,
                        f"unknown_{one_sided_code}",
                    ),
                    "one_sided_policy_code": one_sided_code,
                    "one_sided_region_id": int(one_sided_region_id[marker]),
                    "one_sided_side_normal_sign": one_sided_sign,
                    "one_sided_side_selected": one_sided_side_selected,
                    "one_sided_fluid_side_pressure_pa": float(
                        fluid_side_pressure_pa[marker]
                    ),
                    "one_sided_reference_pressure_pa": float(
                        reference_pressure_pa[marker]
                    ),
                    "one_sided_pressure_pair_policy": (
                        STRESS_PRESSURE_PAIR_POLICY_NAMES.get(
                            pair_policy_code,
                            f"unknown_{pair_policy_code}",
                        )
                    ),
                    "one_sided_anchor_selected": bool(
                        int(one_sided_anchor_selected[marker])
                    ),
                    "one_sided_anchor_fallback_used": bool(
                        int(one_sided_anchor_fallback_used[marker])
                    ),
                    "inside_probe_multiplier": float(
                        inside_probe_multiplier[marker]
                    ),
                    "outside_probe_multiplier": float(
                        outside_probe_multiplier[marker]
                    ),
                    "inside_probe_distance_m": float(
                        inside_probe_distance_m[marker]
                    ),
                    "outside_probe_distance_m": float(
                        outside_probe_distance_m[marker]
                    ),
                    "inside_probe_cell": [
                        int(value) for value in inside_probe_cell[marker]
                    ],
                    "outside_probe_cell": [
                        int(value) for value in outside_probe_cell[marker]
                    ],
                    "inside_probe_nearest_cell": [
                        int(value) for value in inside_probe_cell[marker]
                    ],
                    "outside_probe_nearest_cell": [
                        int(value) for value in outside_probe_cell[marker]
                    ],
                    "inside_probe_grid_coordinate": [
                        float(value) for value in inside_probe_grid_coordinate[marker]
                    ],
                    "outside_probe_grid_coordinate": [
                        float(value) for value in outside_probe_grid_coordinate[marker]
                    ],
                    "inside_probe_fluid_weight": float(
                        inside_probe_fluid_weight[marker]
                    ),
                    "outside_probe_fluid_weight": float(
                        outside_probe_fluid_weight[marker]
                    ),
                    "pressure_traction_pa": pressure_traction_vector,
                    "viscous_traction_pa": viscous_traction_vector,
                    "total_traction_pa": total_traction_vector,
                    "traction_decomposition_residual_pa": (
                        traction_decomposition_residual
                    ),
                    "position_m": [float(value) for value in positions[marker]],
                    "pressure_probe_origin_m": [
                        float(value) for value in probe_origin
                    ],
                    "pressure_probe_origin_source": (
                        "explicit" if explicit_probe_origin else "marker_position"
                    ),
                    "pressure_probe_origin_explicit": explicit_probe_origin,
                    "normal": [float(value) for value in normals[marker]],
                    "region_id": int(regions[marker]),
                }
            )
        return tuple(diagnostics)

    def stress_face_diagnostics(
        self,
        *,
        primary_region_id: int,
        secondary_region_id: int | None = None,
        streamwise_axis_index: int = 2,
    ) -> dict[str, object]:
        diagnostics = self.stress_marker_diagnostics()

        def _mean(markers: list[dict[str, Any]], field: str) -> float | str:
            if not markers:
                return ""
            return float(np.mean([float(marker[field]) for marker in markers]))

        def _mean_vector_component(
            markers: list[dict[str, Any]],
            field: str,
            component: int,
        ) -> float | str:
            if not markers:
                return ""
            return float(
                np.mean(
                    [
                        float(marker[field][component])
                        for marker in markers
                    ]
                )
            )

        def _mean_found(
            markers: list[dict[str, Any]],
            found_field: str,
            value_field: str,
        ) -> float | str:
            found_markers = [
                marker for marker in markers if bool(marker[found_field])
            ]
            return _mean(found_markers, value_field)

        def _distance_stats(
            markers: list[dict[str, Any]],
            distance_field: str,
        ) -> tuple[float | str, float | str, float | str]:
            values = [
                float(marker[distance_field])
                for marker in markers
                if float(marker[distance_field]) >= 0.0
            ]
            if not values:
                return "", "", ""
            return float(min(values)), float(np.mean(values)), float(max(values))

        def _rung_histogram(
            markers: list[dict[str, Any]],
            rung_field: str,
        ) -> dict[str, int]:
            histogram: dict[str, int] = {}
            for marker in markers:
                rung = int(marker[rung_field])
                if rung < 0:
                    continue
                key = str(rung)
                histogram[key] = histogram.get(key, 0) + 1
            return histogram

        def _is_set_cell(cell: object) -> bool:
            values = [int(value) for value in cell]
            return len(values) == 3 and all(value >= 0 for value in values)

        def _unique_cell_count(
            markers: list[dict[str, Any]],
            cell_field: str,
        ) -> int:
            cells = {
                tuple(int(value) for value in marker[cell_field])
                for marker in markers
                if _is_set_cell(marker[cell_field])
            }
            return len(cells)

        def _max_decomposition_residual(
            markers: list[dict[str, Any]],
        ) -> float | str:
            values = [
                float(marker["traction_decomposition_residual_pa"])
                for marker in markers
                if math.isfinite(float(marker["traction_decomposition_residual_pa"]))
            ]
            if not values:
                return ""
            return float(max(abs(value) for value in values))

        def _invalid_decomposition_count(markers: list[dict[str, Any]]) -> int:
            return sum(
                1
                for marker in markers
                if not math.isfinite(
                    float(marker["traction_decomposition_residual_pa"])
                )
            )

        def _summarize(prefix: str, region_id: int) -> dict[str, object]:
            face_markers = [
                marker
                for marker in diagnostics
                if int(marker["region_id"]) == int(region_id)
            ]
            valid_markers = [
                marker for marker in face_markers if bool(marker["valid"])
            ]
            invalid_markers = [
                marker for marker in face_markers if not bool(marker["valid"])
            ]
            pressure_complete_markers = [
                marker
                for marker in valid_markers
                if bool(marker["inside_pressure_found"])
                and bool(marker["outside_pressure_found"])
            ]
            pressure_missing_markers = [
                marker
                for marker in valid_markers
                if not (
                    bool(marker["inside_pressure_found"])
                    and bool(marker["outside_pressure_found"])
                )
            ]
            inside_distance_min, inside_distance_mean, inside_distance_max = (
                _distance_stats(valid_markers, "inside_probe_distance_m")
            )
            outside_distance_min, outside_distance_mean, outside_distance_max = (
                _distance_stats(valid_markers, "outside_probe_distance_m")
            )
            return {
                f"{prefix}_face_marker_count": len(face_markers),
                f"{prefix}_face_valid_marker_count": len(valid_markers),
                f"{prefix}_face_invalid_marker_count": len(invalid_markers),
                f"{prefix}_face_pressure_complete_marker_count": len(
                    pressure_complete_markers
                ),
                f"{prefix}_face_pressure_missing_marker_count": len(
                    pressure_missing_markers
                ),
                f"{prefix}_face_base_pressure_found_marker_count": sum(
                    1 for marker in valid_markers if bool(marker["base_pressure_found"])
                ),
                f"{prefix}_face_inside_pressure_found_marker_count": sum(
                    1
                    for marker in valid_markers
                    if bool(marker["inside_pressure_found"])
                ),
                f"{prefix}_face_outside_pressure_found_marker_count": sum(
                    1
                    for marker in valid_markers
                    if bool(marker["outside_pressure_found"])
                ),
                f"{prefix}_face_mean_pressure_pa": _mean(
                    valid_markers,
                    "pressure_jump_pa",
                ),
                f"{prefix}_face_mean_pressure_jump_pa": _mean(
                    valid_markers,
                    "pressure_jump_pa",
                ),
                f"{prefix}_face_mean_base_pressure_pa": _mean_found(
                    valid_markers,
                    "base_pressure_found",
                    "base_pressure_pa",
                ),
                f"{prefix}_face_mean_inside_pressure_pa": _mean(
                    [
                        marker
                        for marker in valid_markers
                        if bool(marker["inside_pressure_found"])
                    ],
                    "inside_pressure_pa",
                ),
                f"{prefix}_face_mean_outside_pressure_pa": _mean(
                    [
                        marker
                        for marker in valid_markers
                        if bool(marker["outside_pressure_found"])
                    ],
                    "outside_pressure_pa",
                ),
                f"{prefix}_face_mean_fluid_side_pressure_pa": _mean(
                    [
                        marker
                        for marker in valid_markers
                        if bool(marker["fluid_side_pressure_defined"])
                    ],
                    "fluid_side_pressure_pa",
                ),
                f"{prefix}_face_mean_reference_pressure_pa": _mean(
                    valid_markers,
                    "reference_pressure_pa",
                ),
                f"{prefix}_face_inside_probe_rung_histogram": _rung_histogram(
                    valid_markers,
                    "inside_probe_rung",
                ),
                f"{prefix}_face_outside_probe_rung_histogram": _rung_histogram(
                    valid_markers,
                    "outside_probe_rung",
                ),
                f"{prefix}_face_inside_probe_distance_min_m": inside_distance_min,
                f"{prefix}_face_inside_probe_distance_mean_m": inside_distance_mean,
                f"{prefix}_face_inside_probe_distance_max_m": inside_distance_max,
                f"{prefix}_face_outside_probe_distance_min_m": outside_distance_min,
                f"{prefix}_face_outside_probe_distance_mean_m": outside_distance_mean,
                f"{prefix}_face_outside_probe_distance_max_m": outside_distance_max,
                f"{prefix}_face_inside_unique_nearest_cell_count": _unique_cell_count(
                    valid_markers,
                    "inside_probe_nearest_cell",
                ),
                f"{prefix}_face_outside_unique_nearest_cell_count": _unique_cell_count(
                    valid_markers,
                    "outside_probe_nearest_cell",
                ),
                f"{prefix}_face_mean_traction_z_pa": _mean_vector_component(
                    valid_markers,
                    "total_traction_pa",
                    int(streamwise_axis_index),
                ),
                f"{prefix}_face_mean_total_traction_z_pa": _mean_vector_component(
                    valid_markers,
                    "total_traction_pa",
                    int(streamwise_axis_index),
                ),
                f"{prefix}_face_mean_pressure_traction_z_pa": (
                    _mean_vector_component(
                        valid_markers,
                        "pressure_traction_pa",
                        int(streamwise_axis_index),
                    )
                ),
                f"{prefix}_face_mean_viscous_traction_z_pa": (
                    _mean_vector_component(
                        valid_markers,
                        "viscous_traction_pa",
                        int(streamwise_axis_index),
                    )
                ),
                f"{prefix}_face_traction_decomposition_max_abs_residual_pa": (
                    _max_decomposition_residual(valid_markers)
                ),
                f"{prefix}_face_traction_decomposition_invalid_marker_count": (
                    _invalid_decomposition_count(valid_markers)
                ),
            }

        result = _summarize("primary", int(primary_region_id))
        if secondary_region_id is not None:
            result.update(_summarize("secondary", int(secondary_region_id)))
        return result

    @ti.kernel
    def _mark_far_pressure_air_backed_seed_components_kernel(
        self,
        obstacle_field: ti.template(),
        base_obstacle_field: ti.template(),
        outlet_reachable_field: ti.template(),
        unreached_component_label_field: ti.template(),
        air_component_selected_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        far_pressure_region_id: ti.i32,
        far_probe_max_multiplier: ti.f32,
        air_probe_normal_sign: ti.f32,
    ):
        self.report_air_backed_seed_marker_count[None] = 0
        self.report_air_backed_seed_missed_marker_count[None] = 0
        self.report_air_backed_seed_fallback_cell_count[None] = 0
        component_capacity = air_component_selected_field.shape[0]
        for slot in air_component_selected_field:
            air_component_selected_field[slot] = 0
        for marker in range(marker_count):
            if self.region_id[marker] == far_pressure_region_id:
                position = self.x_gamma_m[marker]
                normal = self.n_gamma[marker]
                grid_coordinate = self._grid_coordinate_from_fields(
                    position,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                i_near = ti.min(
                    ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0),
                    nx - 1,
                )
                j_near = ti.min(
                    ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0),
                    ny - 1,
                )
                k_near = ti.min(
                    ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0),
                    nz - 1,
                )
                normal_spacing_inv = (
                    ti.abs(normal.x) / cell_width_x_m[i_near]
                    + ti.abs(normal.y) / cell_width_y_m[j_near]
                    + ti.abs(normal.z) / cell_width_z_m[k_near]
                )
                probe_distance_m = 1.0 / ti.max(normal_spacing_inv, 1.0e-12)
                seed_found = 0
                # Mirrored CAD windings are valid. By default the seed scans
                # both normal directions and accepts only outlet-unreachable
                # components; cases with known interface orientation may pin a
                # single normal side to avoid selecting isolated water pockets.
                for side in range(2):
                    crossed_base = 0
                    if base_obstacle_field[i_near, j_near, k_near] != 0:
                        crossed_base = 1
                    direction = 1.0
                    scan_side = 1
                    if ti.abs(air_probe_normal_sign) > 0.5:
                        if side == 0:
                            direction = air_probe_normal_sign
                        else:
                            scan_side = 0
                    elif side == 1:
                        direction = -1.0
                    # 10-rung ladder: the standard sampler rungs
                    # (1.0 + 0.5*k, k = 0..4) then the closure-extension rungs
                    # (3 + (mult - 3) * (k+1)/5) - runtime range per the
                    # S2-A5 no-unroll rule.
                    for probe_index in range(10):
                        if scan_side != 0 and seed_found == 0 and crossed_base == 0:
                            multiplier = 1.0 + 0.5 * ti.cast(probe_index, ti.f32)
                            if probe_index >= 5:
                                multiplier = 3.0 + (
                                    far_probe_max_multiplier - 3.0
                                ) * (ti.cast(probe_index - 4, ti.f32) / 5.0)
                            probe_position = position + normal * (
                                direction * probe_distance_m * multiplier
                            )
                            probe_coordinate = self._grid_coordinate_from_fields(
                                probe_position,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            pi = ti.min(
                                ti.max(
                                    ti.floor(probe_coordinate.x + 0.5, ti.i32),
                                    0,
                                ),
                                nx - 1,
                            )
                            pj = ti.min(
                                ti.max(
                                    ti.floor(probe_coordinate.y + 0.5, ti.i32),
                                    0,
                                ),
                                ny - 1,
                            )
                            pk = ti.min(
                                ti.max(
                                    ti.floor(probe_coordinate.z + 0.5, ti.i32),
                                    0,
                                ),
                                nz - 1,
                            )
                            if base_obstacle_field[pi, pj, pk] != 0:
                                # H1-style crossing guard: base geometry ends
                                # the physical air column - never seed another
                                # compartment through the chamber wall.
                                crossed_base = 1
                            elif (
                                obstacle_field[pi, pj, pk] == 0
                                and outlet_reachable_field[pi, pj, pk] == 0
                            ):
                                label = unreached_component_label_field[
                                    pi,
                                    pj,
                                    pk,
                                ]
                                if label >= -component_capacity and label <= -1:
                                    air_component_selected_field[-label - 1] = 1
                                    seed_found = 1
                if seed_found == 1:
                    ti.atomic_add(
                        self.report_air_backed_seed_marker_count[None],
                        1,
                    )
                else:
                    ti.atomic_add(
                        self.report_air_backed_seed_missed_marker_count[None],
                        1,
                    )

    @ti.func
    def _select_far_pressure_air_component_if_unreached(
        self,
        obstacle_field: ti.template(),
        outlet_reachable_field: ti.template(),
        unreached_component_label_field: ti.template(),
        air_component_selected_field: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ):
        if obstacle_field[i, j, k] == 0 and outlet_reachable_field[i, j, k] == 0:
            component_capacity = air_component_selected_field.shape[0]
            label = unreached_component_label_field[i, j, k]
            if label >= -component_capacity and label <= -1:
                air_component_selected_field[-label - 1] = 1
                ti.atomic_add(
                    self.report_air_backed_seed_fallback_cell_count[None],
                    1,
                )

    @ti.kernel
    def _mark_far_pressure_air_backed_region_adjacent_components_kernel(
        self,
        obstacle_field: ti.template(),
        outlet_reachable_field: ti.template(),
        unreached_component_label_field: ti.template(),
        air_component_selected_field: ti.template(),
        node_kind_code: ti.template(),
        nearest_marker: ti.template(),
        marker_count: ti.i32,
        far_pressure_region_id: ti.i32,
    ):
        self.report_air_backed_seed_fallback_cell_count[None] = 0
        component_capacity = air_component_selected_field.shape[0]
        for i, j, k in obstacle_field:
            if obstacle_field[i, j, k] != 0 or outlet_reachable_field[i, j, k] != 0:
                continue
            label = unreached_component_label_field[i, j, k]
            if label < -component_capacity or label > -1:
                continue
            marker = nearest_marker[i, j, k]
            if marker < 0 or marker >= marker_count:
                continue
            if self.region_id[marker] == far_pressure_region_id:
                air_component_selected_field[-label - 1] = 1
                ti.atomic_add(
                    self.report_air_backed_seed_fallback_cell_count[None],
                    1,
                )
        for i, j, k in obstacle_field:
            if node_kind_code[i, j, k] == HibmMpmIbNodeSearch._NODE_NONE:
                continue
            marker = nearest_marker[i, j, k]
            if marker < 0 or marker >= marker_count:
                continue
            if self.region_id[marker] != far_pressure_region_id:
                continue
            if i > 0:
                self._select_far_pressure_air_component_if_unreached(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    i - 1,
                    j,
                    k,
                )
            if i < obstacle_field.shape[0] - 1:
                self._select_far_pressure_air_component_if_unreached(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    i + 1,
                    j,
                    k,
                )
            if j > 0:
                self._select_far_pressure_air_component_if_unreached(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    i,
                    j - 1,
                    k,
                )
            if j < obstacle_field.shape[1] - 1:
                self._select_far_pressure_air_component_if_unreached(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    i,
                    j + 1,
                    k,
                )
            if k > 0:
                self._select_far_pressure_air_component_if_unreached(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    i,
                    j,
                    k - 1,
                )
            if k < obstacle_field.shape[2] - 1:
                self._select_far_pressure_air_component_if_unreached(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    i,
                    j,
                    k + 1,
                )

    @ti.kernel
    def _mark_far_pressure_air_backed_anchor_cell_components_kernel(
        self,
        obstacle_field: ti.template(),
        outlet_reachable_field: ti.template(),
        unreached_component_label_field: ti.template(),
        air_component_selected_field: ti.template(),
        node_kind_code: ti.template(),
        nearest_marker: ti.template(),
        node_anchor_cell: ti.template(),
        marker_count: ti.i32,
        far_pressure_region_id: ti.i32,
    ):
        for i, j, k in obstacle_field:
            if node_kind_code[i, j, k] == HibmMpmIbNodeSearch._NODE_NONE:
                continue
            marker = nearest_marker[i, j, k]
            if marker < 0 or marker >= marker_count:
                continue
            if self.region_id[marker] != far_pressure_region_id:
                continue
            anchor = node_anchor_cell[i, j, k]
            anchor_i = anchor.x
            anchor_j = anchor.y
            anchor_k = anchor.z
            if (
                anchor_i >= 0
                and anchor_i < obstacle_field.shape[0]
                and anchor_j >= 0
                and anchor_j < obstacle_field.shape[1]
                and anchor_k >= 0
                and anchor_k < obstacle_field.shape[2]
            ):
                self._select_far_pressure_air_component_if_unreached(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    anchor_i,
                    anchor_j,
                    anchor_k,
                )

    @ti.kernel
    def _mark_far_pressure_air_backed_dirichlet_region_components_kernel(
        self,
        obstacle_field: ti.template(),
        outlet_reachable_field: ti.template(),
        unreached_component_label_field: ti.template(),
        air_component_selected_field: ti.template(),
        velocity_dirichlet_marker_region_id: ti.template(),
        far_pressure_region_id: ti.i32,
    ):
        for i, j, k in obstacle_field:
            if velocity_dirichlet_marker_region_id[i, j, k] != far_pressure_region_id:
                continue
            self._select_far_pressure_air_component_if_unreached(
                obstacle_field,
                outlet_reachable_field,
                unreached_component_label_field,
                air_component_selected_field,
                i,
                j,
                k,
            )

    def mark_far_pressure_air_backed_seed_components(
        self,
        obstacle_field,
        base_obstacle_field,
        outlet_reachable_field,
        unreached_component_label_field,
        air_component_selected_field,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        cell_width_x_m,
        cell_width_y_m,
        cell_width_z_m,
        grid_nodes,
        *,
        far_pressure_region_id: int,
        far_pressure_inside_probe_max_multiplier: float = 3.0,
        far_pressure_air_backed_probe_normal_sign: float = 0.0,
        fallback_to_bidirectional_if_all_missed: bool = False,
        fallback_to_region_adjacency_if_all_missed: bool = False,
        node_kind_code=None,
        nearest_marker=None,
        node_anchor_cell=None,
        velocity_dirichlet_marker_region_id=None,
    ) -> tuple[int, int]:
        """Select unreached components on closure markers' air side (S2-A12).

        For every marker of the declared closure region the configured normal
        side is walked with the 10-rung ladder (standard sampler rungs then the
        closure-extension rungs up to
        ``far_pressure_inside_probe_max_multiplier``); the first rung whose
        nearest cell is active, flood-unreached and component-labeled
        selects that component in the fluid-owned fixed-capacity mask. Walks
        stop at base geometry (H1 crossing-guard semantics). Returns
        ``(seeded_marker_count, missed_marker_count)``; a fully missed scan
        with a large unreached set is the partial-enclosure signature
        (mechanism inert, debt returns) and must be visible in history.
        A zero normal-side sign scans both sides for backward compatibility;
        +1 or -1 scans only that marker-normal side.
        Outlet-reachable cells never seed: legitimate water is structurally
        unselectable, and non-closure regions are
        untouched because only closure markers walk.
        When ``fallback_to_bidirectional_if_all_missed`` is true, a pinned
        single-side scan that finds no component retries the bidirectional
        scan. The retry is still outlet-unreachable-only, so it cannot mark
        normal outlet-connected water as air.
        When ``fallback_to_region_adjacency_if_all_missed`` is true and any
        closure marker still misses after the normal/bidirectional scan, the
        selector marks unreached components containing cells whose nearest
        HIBM marker is in the declared closure region. This is a conservative
        CAD fallback for row clouds where the normal ladder only partially
        reaches the labeled component set. If ``node_anchor_cell`` is
        provided, the fallback also checks classified closure nodes'
        velocity-Dirichlet owner/anchor cells. If
        ``velocity_dirichlet_marker_region_id`` is provided, it also selects
        components containing row-owner cells stamped as coming from the
        closure region.
        """
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        far_region_id = int(far_pressure_region_id)
        if far_region_id == -1:
            raise ValueError(
                "far_pressure_region_id must name a closure region "
                "(air-backed classification is closure-gated)"
            )
        far_probe_max = float(far_pressure_inside_probe_max_multiplier)
        if not math.isfinite(far_probe_max) or far_probe_max < 3.0:
            raise ValueError(
                "far_pressure_inside_probe_max_multiplier must be finite "
                "and >= 3.0"
            )
        probe_normal_sign = float(far_pressure_air_backed_probe_normal_sign)
        if probe_normal_sign not in (-1.0, 0.0, 1.0):
            raise ValueError(
                "far_pressure_air_backed_probe_normal_sign must be -1.0, 0.0, or 1.0"
            )
        air_component_shape = tuple(
            int(value) for value in air_component_selected_field.shape
        )
        if len(air_component_shape) != 1 or air_component_shape[0] <= 0:
            raise ValueError(
                "air_component_selected_field must be a positive one-dimensional field"
            )
        if bool(fallback_to_region_adjacency_if_all_missed):
            if nearest_marker is None:
                raise ValueError(
                    "fallback_to_region_adjacency_if_all_missed requires "
                    "nearest_marker"
                )
            if tuple(nearest_marker.shape) != tuple(obstacle_field.shape):
                raise ValueError(
                    "nearest_marker shape must match the fluid cell grid"
                )
            if node_kind_code is not None and tuple(node_kind_code.shape) != tuple(
                obstacle_field.shape
            ):
                raise ValueError(
                    "node_kind_code shape must match the fluid cell grid"
                )
            if node_anchor_cell is not None and tuple(node_anchor_cell.shape) != tuple(
                obstacle_field.shape
            ):
                raise ValueError(
                    "node_anchor_cell shape must match the fluid cell grid"
                )
            if (
                velocity_dirichlet_marker_region_id is not None
                and tuple(velocity_dirichlet_marker_region_id.shape)
                != tuple(obstacle_field.shape)
            ):
                raise ValueError(
                    "velocity_dirichlet_marker_region_id shape must match "
                    "the fluid cell grid"
                )
        self._mark_far_pressure_air_backed_seed_components_kernel(
            obstacle_field,
            base_obstacle_field,
            outlet_reachable_field,
            unreached_component_label_field,
            air_component_selected_field,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            cell_width_x_m,
            cell_width_y_m,
            cell_width_z_m,
            int(self.marker_count),
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
            far_region_id,
            far_probe_max,
            probe_normal_sign,
        )
        seeded = int(self.report_air_backed_seed_marker_count[None])
        missed = int(self.report_air_backed_seed_missed_marker_count[None])
        if (
            bool(fallback_to_bidirectional_if_all_missed)
            and probe_normal_sign != 0.0
            and seeded == 0
            and missed > 0
        ):
            self._mark_far_pressure_air_backed_seed_components_kernel(
                obstacle_field,
                base_obstacle_field,
                outlet_reachable_field,
                unreached_component_label_field,
                air_component_selected_field,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                cell_width_x_m,
                cell_width_y_m,
                cell_width_z_m,
                int(self.marker_count),
                int(nodes[0]),
                int(nodes[1]),
                int(nodes[2]),
                far_region_id,
                far_probe_max,
                0.0,
            )
            seeded = int(self.report_air_backed_seed_marker_count[None])
            missed = int(self.report_air_backed_seed_missed_marker_count[None])
        if (
            bool(fallback_to_region_adjacency_if_all_missed)
            and missed > 0
        ):
            self._mark_far_pressure_air_backed_region_adjacent_components_kernel(
                obstacle_field,
                outlet_reachable_field,
                unreached_component_label_field,
                air_component_selected_field,
                node_kind_code,
                nearest_marker,
                int(self.marker_count),
                far_region_id,
            )
            if node_anchor_cell is not None:
                self._mark_far_pressure_air_backed_anchor_cell_components_kernel(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    node_kind_code,
                    nearest_marker,
                    node_anchor_cell,
                    int(self.marker_count),
                    far_region_id,
                )
            if velocity_dirichlet_marker_region_id is not None:
                self._mark_far_pressure_air_backed_dirichlet_region_components_kernel(
                    obstacle_field,
                    outlet_reachable_field,
                    unreached_component_label_field,
                    air_component_selected_field,
                    velocity_dirichlet_marker_region_id,
                    far_region_id,
                )
        return (
            seeded,
            missed,
        )

    @ti.kernel
    def _write_region_pressure_reachability_barrier_kernel(
        self,
        barrier_field: ti.template(),
        node_kind_code: ti.template(),
        nearest_marker: ti.template(),
        marker_count: ti.i32,
        barrier_node_code: ti.i32,
        barrier_region_id: ti.i32,
        secondary_barrier_region_id: ti.i32,
        tertiary_barrier_region_id: ti.i32,
        include_all_classified_region_nodes: ti.i32,
    ) -> ti.i32:
        count = 0
        for node in ti.grouped(barrier_field):
            value = 0
            marker = nearest_marker[node]
            node_matches = node_kind_code[node] == barrier_node_code
            if include_all_classified_region_nodes != 0:
                node_matches = node_kind_code[node] != HibmMpmIbNodeSearch._NODE_NONE
            region_matches = False
            if 0 <= marker and marker < marker_count:
                region = self.region_id[marker]
                if region == barrier_region_id or (
                    secondary_barrier_region_id != -1
                    and region == secondary_barrier_region_id
                ) or (
                    tertiary_barrier_region_id != -1
                    and region == tertiary_barrier_region_id
                ):
                    region_matches = True
            if (
                node_matches
                and region_matches
            ):
                value = 1
                count += 1
            barrier_field[node] = value
        return count

    def write_region_pressure_reachability_barrier(
        self,
        barrier_field,
        node_kind_code,
        nearest_marker,
        *,
        barrier_node_code: int,
        barrier_region_id: int,
        secondary_barrier_region_id: int = -1,
        tertiary_barrier_region_id: int = -1,
        include_all_classified_region_nodes: bool = False,
    ) -> int:
        if tuple(barrier_field.shape) != tuple(node_kind_code.shape):
            raise ValueError(
                "barrier_field shape "
                f"{tuple(barrier_field.shape)} does not match node_kind_code "
                f"shape {tuple(node_kind_code.shape)}"
            )
        if tuple(barrier_field.shape) != tuple(nearest_marker.shape):
            raise ValueError(
                "barrier_field shape "
                f"{tuple(barrier_field.shape)} does not match nearest_marker "
                f"shape {tuple(nearest_marker.shape)}"
            )
        return int(
            self._write_region_pressure_reachability_barrier_kernel(
                barrier_field,
                node_kind_code,
                nearest_marker,
                int(self.marker_count),
                int(barrier_node_code),
                int(barrier_region_id),
                int(secondary_barrier_region_id),
                int(tertiary_barrier_region_id),
                1 if bool(include_all_classified_region_nodes) else 0,
            )
        )

    @ti.kernel
    def _sample_no_slip_residual_kernel(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
    ):
        self.report_no_slip_valid_marker_count[None] = 0
        self.report_no_slip_invalid_marker_count[None] = 0
        self.report_no_slip_max_residual_mps[None] = 0.0
        self.report_no_slip_sum_residual2_mps2[None] = ti.cast(0.0, ti.f64)
        self.report_no_slip_direct_sample_marker_count[None] = 0
        self.report_no_slip_normal_walk_sample_marker_count[None] = 0
        self.report_no_slip_nearest_fluid_sample_marker_count[None] = 0
        self.report_no_slip_zero_normal_marker_count[None] = 0
        self.report_no_slip_no_fluid_sample_marker_count[None] = 0
        self.report_no_slip_primary_region_valid_marker_count[None] = 0
        self.report_no_slip_primary_region_invalid_marker_count[None] = 0
        self.report_no_slip_secondary_region_valid_marker_count[None] = 0
        self.report_no_slip_secondary_region_invalid_marker_count[None] = 0
        self.report_no_slip_other_region_valid_marker_count[None] = 0
        self.report_no_slip_other_region_invalid_marker_count[None] = 0
        for marker in range(marker_count):
            grid_coordinate = self._grid_coordinate_from_fields(
                self.x_gamma_m[marker],
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                cell_center_x_m,
                cell_center_y_m,
                cell_center_z_m,
                nx,
                ny,
                nz,
            )
            sample_source = 0
            fluid_velocity, fluid_weight = self._sample_fluid_velocity_trilinear(
                velocity_field,
                obstacle_field,
                grid_coordinate.x,
                grid_coordinate.y,
                grid_coordinate.z,
                nx,
                ny,
                nz,
            )
            if fluid_weight > 1.0e-12:
                sample_source = 1
            if fluid_weight <= 1.0e-12:
                normal = self.n_gamma[marker]
                normal_norm = normal.norm()
                if normal_norm > 1.0e-12:
                    walk_normal = normal / normal_norm
                    base_i = ti.min(
                        ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0),
                        nx - 1,
                    )
                    base_j = ti.min(
                        ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0),
                        ny - 1,
                    )
                    base_k = ti.min(
                        ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0),
                        nz - 1,
                    )
                    cell_width_x = cell_face_x_m[base_i + 1] - cell_face_x_m[base_i]
                    cell_width_y = cell_face_y_m[base_j + 1] - cell_face_y_m[base_j]
                    cell_width_z = cell_face_z_m[base_k + 1] - cell_face_z_m[base_k]
                    walk_step_m = 0.5 / ti.max(
                        ti.abs(walk_normal.x) / ti.max(cell_width_x, 1.0e-12)
                        + ti.abs(walk_normal.y) / ti.max(cell_width_y, 1.0e-12)
                        + ti.abs(walk_normal.z) / ti.max(cell_width_z, 1.0e-12),
                        1.0e-12,
                    )
                    fallback_found = 0
                    for side_index in ti.static(range(2)):
                        side_sign = 1.0
                        if side_index == 1:
                            side_sign = -1.0
                        step_index = 0
                        while (
                            step_index < HIBM_OWNER_RELOCATION_WALK_STEPS
                            and fallback_found == 0
                        ):
                            candidate_position = (
                                self.x_gamma_m[marker]
                                + walk_normal
                                * (
                                    side_sign
                                    * walk_step_m
                                    * ti.cast(step_index + 1, ti.f32)
                                )
                            )
                            candidate_coordinate = self._grid_coordinate_from_fields(
                                candidate_position,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            candidate_velocity, candidate_weight = (
                                self._sample_fluid_velocity_trilinear(
                                    velocity_field,
                                    obstacle_field,
                                    candidate_coordinate.x,
                                    candidate_coordinate.y,
                                    candidate_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                )
                            )
                            if candidate_weight > 1.0e-12:
                                fluid_velocity = candidate_velocity
                                fluid_weight = candidate_weight
                                sample_source = 2
                                fallback_found = 1
                            step_index += 1
                else:
                    self.report_no_slip_zero_normal_marker_count[None] += 1
                if fluid_weight <= 1.0e-12:
                    base_i = ti.min(
                        ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0),
                        nx - 1,
                    )
                    base_j = ti.min(
                        ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0),
                        ny - 1,
                    )
                    base_k = ti.min(
                        ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0),
                        nz - 1,
                    )
                    nearest_found = 0
                    nearest_distance2 = 1.0e30
                    nearest_velocity = ti.Vector([0.0, 0.0, 0.0])
                    di = -HIBM_NO_SLIP_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                    while di <= HIBM_NO_SLIP_NEAREST_FLUID_FALLBACK_RADIUS_CELLS:
                        dj = -HIBM_NO_SLIP_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                        while dj <= HIBM_NO_SLIP_NEAREST_FLUID_FALLBACK_RADIUS_CELLS:
                            dk = -HIBM_NO_SLIP_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                            while (
                                dk
                                <= HIBM_NO_SLIP_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                            ):
                                candidate_i = base_i + di
                                candidate_j = base_j + dj
                                candidate_k = base_k + dk
                                if (
                                    0 <= candidate_i
                                    and candidate_i < nx
                                    and 0 <= candidate_j
                                    and candidate_j < ny
                                    and 0 <= candidate_k
                                    and candidate_k < nz
                                ):
                                    if (
                                        obstacle_field[
                                            candidate_i,
                                            candidate_j,
                                            candidate_k,
                                        ]
                                        == 0
                                    ):
                                        candidate_center = ti.Vector(
                                            [
                                                cell_center_x_m[candidate_i],
                                                cell_center_y_m[candidate_j],
                                                cell_center_z_m[candidate_k],
                                            ]
                                        )
                                        delta = (
                                            candidate_center - self.x_gamma_m[marker]
                                        )
                                        distance2 = delta.dot(delta)
                                        if distance2 < nearest_distance2:
                                            nearest_distance2 = distance2
                                            nearest_velocity = velocity_field[
                                                candidate_i,
                                                candidate_j,
                                                candidate_k,
                                            ]
                                            nearest_found = 1
                                dk += 1
                            dj += 1
                        di += 1
                    if nearest_found != 0:
                        fluid_velocity = nearest_velocity
                        fluid_weight = 1.0
                        sample_source = 3
            if fluid_weight > 1.0e-12:
                residual = fluid_velocity - self.v_gamma_mps[marker]
                residual_norm = residual.norm()
                self.report_no_slip_valid_marker_count[None] += 1
                marker_region = self.region_id[marker]
                if marker_region == primary_region_id:
                    self.report_no_slip_primary_region_valid_marker_count[None] += 1
                elif marker_region == secondary_region_id:
                    self.report_no_slip_secondary_region_valid_marker_count[None] += 1
                else:
                    self.report_no_slip_other_region_valid_marker_count[None] += 1
                if sample_source == 1:
                    self.report_no_slip_direct_sample_marker_count[None] += 1
                elif sample_source == 2:
                    self.report_no_slip_normal_walk_sample_marker_count[None] += 1
                elif sample_source == 3:
                    self.report_no_slip_nearest_fluid_sample_marker_count[None] += 1
                ti.atomic_max(
                    self.report_no_slip_max_residual_mps[None],
                    residual_norm,
                )
                self.report_no_slip_sum_residual2_mps2[None] += ti.cast(
                    residual_norm * residual_norm,
                    ti.f64,
                )
            else:
                self.report_no_slip_invalid_marker_count[None] += 1
                self.report_no_slip_no_fluid_sample_marker_count[None] += 1
                marker_region = self.region_id[marker]
                if marker_region == primary_region_id:
                    self.report_no_slip_primary_region_invalid_marker_count[None] += 1
                elif marker_region == secondary_region_id:
                    self.report_no_slip_secondary_region_invalid_marker_count[None] += 1
                else:
                    self.report_no_slip_other_region_invalid_marker_count[None] += 1

    def sample_no_slip_residual(
        self,
        velocity_field,
        obstacle_field,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
        primary_region_id: int = -1,
        secondary_region_id: int = -1,
    ) -> HibmMpmNoSlipResidualReport:
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._sample_no_slip_residual_kernel(
            velocity_field,
            obstacle_field,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(self.marker_count),
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
            int(primary_region_id),
            int(secondary_region_id),
        )
        valid_count = int(self.report_no_slip_valid_marker_count[None])
        sum_residual2 = float(self.report_no_slip_sum_residual2_mps2[None])
        l2_residual = (
            math.sqrt(sum_residual2 / float(valid_count)) if valid_count else 0.0
        )
        return HibmMpmNoSlipResidualReport(
            valid_marker_count=valid_count,
            invalid_marker_count=int(self.report_no_slip_invalid_marker_count[None]),
            max_no_slip_residual_mps=float(
                self.report_no_slip_max_residual_mps[None]
            ),
            l2_no_slip_residual_mps=l2_residual,
            direct_sample_marker_count=int(
                self.report_no_slip_direct_sample_marker_count[None]
            ),
            normal_walk_sample_marker_count=int(
                self.report_no_slip_normal_walk_sample_marker_count[None]
            ),
            nearest_fluid_sample_marker_count=int(
                self.report_no_slip_nearest_fluid_sample_marker_count[None]
            ),
            zero_normal_marker_count=int(
                self.report_no_slip_zero_normal_marker_count[None]
            ),
            no_fluid_sample_marker_count=int(
                self.report_no_slip_no_fluid_sample_marker_count[None]
            ),
            primary_region_valid_marker_count=int(
                self.report_no_slip_primary_region_valid_marker_count[None]
            ),
            primary_region_invalid_marker_count=int(
                self.report_no_slip_primary_region_invalid_marker_count[None]
            ),
            secondary_region_valid_marker_count=int(
                self.report_no_slip_secondary_region_valid_marker_count[None]
            ),
            secondary_region_invalid_marker_count=int(
                self.report_no_slip_secondary_region_invalid_marker_count[None]
            ),
            other_region_valid_marker_count=int(
                self.report_no_slip_other_region_valid_marker_count[None]
            ),
            other_region_invalid_marker_count=int(
                self.report_no_slip_other_region_invalid_marker_count[None]
            ),
        )

    @ti.func
    def _marker_particle_shape_weight(
        self,
        marker_position_m,
        particle_position_m,
        support_radius_m: ti.f32,
    ):
        relative = marker_position_m - particle_position_m
        wx = ti.max(1.0 - ti.abs(relative.x) / support_radius_m, 0.0)
        wy = ti.max(1.0 - ti.abs(relative.y) / support_radius_m, 0.0)
        wz = ti.max(1.0 - ti.abs(relative.z) / support_radius_m, 0.0)
        return wx * wy * wz

    @ti.func
    def _vector3_is_finite(self, value):
        finite = 1
        if ti.math.isnan(value.x) or ti.math.isinf(value.x):
            finite = 0
        if ti.math.isnan(value.y) or ti.math.isinf(value.y):
            finite = 0
        if ti.math.isnan(value.z) or ti.math.isinf(value.z):
            finite = 0
        return finite

    @ti.kernel
    def _clear_mpm_external_forces_kernel(
        self,
        external_force_n: ti.template(),
        particle_count: ti.i32,
    ):
        self.report_mpm_external_force_clear_count[None] = particle_count
        self.report_mpm_external_force_clear_max_abs_n[None] = 0.0
        for particle in range(particle_count):
            force = external_force_n[particle]
            force_norm = force.norm()
            if force_norm > 0.0:
                ti.atomic_max(
                    self.report_mpm_external_force_clear_max_abs_n[None],
                    force_norm,
                )
            external_force_n[particle] = ti.Vector([0.0, 0.0, 0.0])

    def clear_mpm_external_forces(
        self,
        external_force_n,
        *,
        particle_count: int,
    ) -> HibmMpmExternalForceClearReport:
        particles = int(particle_count)
        if particles <= 0:
            raise ValueError("particle_count must be positive")
        self._clear_mpm_external_forces_kernel(external_force_n, particles)
        return HibmMpmExternalForceClearReport(
            cleared_particle_count=int(
                self.report_mpm_external_force_clear_count[None]
            ),
            max_abs_external_force_before_n=float(
                self.report_mpm_external_force_clear_max_abs_n[None]
            ),
        )

    @ti.kernel
    def _scatter_marker_forces_to_mpm_particles_kernel(
        self,
        external_force_n: ti.template(),
        particle_position_m: ti.template(),
        marker_count: ti.i32,
        particle_count: ti.i32,
        support_radius_m: ti.f32,
    ):
        self.report_mpm_scatter_marker_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_mpm_scatter_external_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_mpm_scatter_active_marker_count[None] = 0
        self.report_mpm_scatter_invalid_marker_count[None] = 0
        self.report_mpm_scatter_active_particle_count[None] = 0
        for marker in range(marker_count):
            marker_position = self.x_gamma_m[marker]
            marker_force = self.F_gamma_n[marker]
            weight_sum = 0.0
            marker_valid = (
                self._vector3_is_finite(marker_position)
                * self._vector3_is_finite(marker_force)
            )
            if marker_valid != 0:
                for particle in range(particle_count):
                    particle_position = particle_position_m[particle]
                    if self._vector3_is_finite(particle_position) != 0:
                        weight_sum += self._marker_particle_shape_weight(
                            marker_position,
                            particle_position,
                            support_radius_m,
                        )
            if marker_valid != 0 and weight_sum > 1.0e-12:
                self.report_mpm_scatter_active_marker_count[None] += 1
                self.report_mpm_scatter_marker_force_n[None] += marker_force
                for particle in range(particle_count):
                    particle_position = particle_position_m[particle]
                    weight = 0.0
                    if self._vector3_is_finite(particle_position) != 0:
                        weight = self._marker_particle_shape_weight(
                            marker_position,
                            particle_position,
                            support_radius_m,
                        )
                    if weight > 0.0:
                        force_contribution = marker_force * (weight / weight_sum)
                        # The current MPM particle/grid dynamics are f32; keep
                        # this adapter cast explicit and report the applied load.
                        force_contribution_for_external = ti.Vector(
                            [
                                ti.cast(force_contribution.x, ti.f32),
                                ti.cast(force_contribution.y, ti.f32),
                                ti.cast(force_contribution.z, ti.f32),
                            ]
                        )
                        external_force_n[particle] += force_contribution_for_external
                        self.report_mpm_scatter_external_force_n[None] += (
                            force_contribution_for_external
                        )
                        self.report_mpm_scatter_active_particle_count[None] += 1
            else:
                self.report_mpm_scatter_invalid_marker_count[None] += 1

    def scatter_marker_forces_to_mpm_particles(
        self,
        external_force_n,
        particle_position_m,
        *,
        particle_count: int,
        support_radius_m: float,
    ) -> HibmMpmMpmForceScatterReport:
        particles = int(particle_count)
        if particles <= 0:
            raise ValueError("particle_count must be positive")
        support_radius = float(support_radius_m)
        if not math.isfinite(support_radius) or support_radius <= 0.0:
            raise ValueError("support_radius_m must be a finite positive number")
        self._scatter_marker_forces_to_mpm_particles_kernel(
            external_force_n,
            particle_position_m,
            int(self.marker_count),
            particles,
            support_radius,
        )
        total_marker_force = self._vector_field_tuple(
            self.report_mpm_scatter_marker_force_n
        )
        total_external_force = self._vector_field_tuple(
            self.report_mpm_scatter_external_force_n
        )
        residual = math.sqrt(
            sum(
                (marker_component - external_component)
                * (marker_component - external_component)
                for marker_component, external_component in zip(
                    total_marker_force,
                    total_external_force,
                    strict=True,
                )
            )
        )
        return HibmMpmMpmForceScatterReport(
            active_marker_count=int(self.report_mpm_scatter_active_marker_count[None]),
            invalid_marker_count=int(
                self.report_mpm_scatter_invalid_marker_count[None]
            ),
            active_particle_count=int(
                self.report_mpm_scatter_active_particle_count[None]
            ),
            total_marker_force_n=total_marker_force,
            total_mpm_external_force_n=total_external_force,
            action_reaction_residual_n=residual,
        )

    @ti.kernel
    def _update_surface_feedback_from_mpm_particles_kernel(
        self,
        particle_position_m: ti.template(),
        particle_velocity_mps: ti.template(),
        marker_count: ti.i32,
        particle_count: ti.i32,
        support_radius_m: ti.f32,
        dt_s: ti.f32,
    ):
        self.report_surface_feedback_updated_marker_count[None] = 0
        self.report_surface_feedback_invalid_marker_count[None] = 0
        self.report_surface_feedback_max_displacement_m[None] = 0.0
        self.report_surface_feedback_max_speed_mps[None] = 0.0
        for marker in range(marker_count):
            old_position = self.x_gamma_m[marker]
            velocity_sum = ti.Vector([0.0, 0.0, 0.0])
            weight_sum = 0.0
            for particle in range(particle_count):
                weight = self._marker_particle_shape_weight(
                    old_position,
                    particle_position_m[particle],
                    support_radius_m,
                )
                if weight > 0.0:
                    velocity_sum += weight * particle_velocity_mps[particle]
                    weight_sum += weight
            if weight_sum > 1.0e-12:
                new_velocity = velocity_sum / weight_sum
                new_position = old_position + dt_s * new_velocity
                self.x_gamma_m[marker] = new_position
                self.v_gamma_mps[marker] = new_velocity
                self.report_surface_feedback_updated_marker_count[None] += 1
                ti.atomic_max(
                    self.report_surface_feedback_max_displacement_m[None],
                    (new_position - old_position).norm(),
                )
                ti.atomic_max(
                    self.report_surface_feedback_max_speed_mps[None],
                    new_velocity.norm(),
                )
            else:
                self.report_surface_feedback_invalid_marker_count[None] += 1

    def update_surface_feedback_from_mpm_particles(
        self,
        particle_position_m,
        particle_velocity_mps,
        *,
        particle_count: int,
        support_radius_m: float,
        dt_s: float,
    ) -> HibmMpmSurfaceUpdateReport:
        particles = int(particle_count)
        if particles <= 0:
            raise ValueError("particle_count must be positive")
        support_radius = float(support_radius_m)
        if not math.isfinite(support_radius) or support_radius <= 0.0:
            raise ValueError("support_radius_m must be a finite positive number")
        feedback_dt = float(dt_s)
        if not math.isfinite(feedback_dt) or feedback_dt <= 0.0:
            raise ValueError("dt_s must be a finite positive number")
        self._update_surface_feedback_from_mpm_particles_kernel(
            particle_position_m,
            particle_velocity_mps,
            int(self.marker_count),
            particles,
            support_radius,
            feedback_dt,
        )
        return HibmMpmSurfaceUpdateReport(
            updated_marker_count=int(
                self.report_surface_feedback_updated_marker_count[None]
            ),
            invalid_marker_count=int(
                self.report_surface_feedback_invalid_marker_count[None]
            ),
            max_marker_displacement_m=float(
                self.report_surface_feedback_max_displacement_m[None]
            ),
            max_marker_speed_mps=float(
                self.report_surface_feedback_max_speed_mps[None]
            ),
        )

    @ti.kernel
    def _update_surface_feedback_from_mpm_surface_particles_kernel(
        self,
        particle_position_m: ti.template(),
        particle_velocity_mps: ti.template(),
        particle_normal: ti.template(),
        particle_area_m2: ti.template(),
        marker_count: ti.i32,
        particle_count: ti.i32,
        support_radius_m: ti.f32,
        dt_s: ti.f32,
    ):
        self.report_surface_feedback_updated_marker_count[None] = 0
        self.report_surface_feedback_invalid_marker_count[None] = 0
        self.report_surface_feedback_max_displacement_m[None] = 0.0
        self.report_surface_feedback_max_speed_mps[None] = 0.0
        self.report_surface_feedback_geometry_updated_marker_count[None] = 0
        self.report_surface_feedback_geometry_invalid_marker_count[None] = 0
        self.report_surface_feedback_max_normal_change[None] = 0.0
        self.report_surface_feedback_max_area_change_m2[None] = 0.0
        for marker in range(marker_count):
            old_position = self.x_gamma_m[marker]
            old_normal = self.n_gamma[marker]
            old_area = self.A_gamma_m2[marker]
            velocity_sum = ti.Vector([0.0, 0.0, 0.0])
            normal_sum = ti.Vector([0.0, 0.0, 0.0])
            area_sum = 0.0
            weight_sum = 0.0
            geometry_weight_sum = 0.0
            for particle in range(particle_count):
                weight = self._marker_particle_shape_weight(
                    old_position,
                    particle_position_m[particle],
                    support_radius_m,
                )
                if weight > 0.0:
                    velocity_sum += weight * particle_velocity_mps[particle]
                    weight_sum += weight
                    particle_surface_normal = particle_normal[particle]
                    particle_surface_area = particle_area_m2[particle]
                    if (
                        particle_surface_normal.norm() > 1.0e-12
                        and particle_surface_area > 0.0
                    ):
                        normal_sum += weight * particle_surface_normal
                        area_sum += weight * particle_surface_area
                        geometry_weight_sum += weight
            if weight_sum > 1.0e-12:
                new_velocity = velocity_sum / weight_sum
                new_position = old_position + dt_s * new_velocity
                self.x_gamma_m[marker] = new_position
                self.v_gamma_mps[marker] = new_velocity
                self.report_surface_feedback_updated_marker_count[None] += 1
                ti.atomic_max(
                    self.report_surface_feedback_max_displacement_m[None],
                    (new_position - old_position).norm(),
                )
                ti.atomic_max(
                    self.report_surface_feedback_max_speed_mps[None],
                    new_velocity.norm(),
                )
                if geometry_weight_sum > 1.0e-12 and normal_sum.norm() > 1.0e-12:
                    new_normal = normal_sum.normalized()
                    new_area = area_sum / geometry_weight_sum
                    self.n_gamma[marker] = new_normal
                    self.A_gamma_m2[marker] = new_area
                    self.report_surface_feedback_geometry_updated_marker_count[
                        None
                    ] += 1
                    ti.atomic_max(
                        self.report_surface_feedback_max_normal_change[None],
                        (new_normal - old_normal).norm(),
                    )
                    ti.atomic_max(
                        self.report_surface_feedback_max_area_change_m2[None],
                        ti.abs(new_area - old_area),
                    )
                else:
                    self.report_surface_feedback_geometry_invalid_marker_count[
                        None
                    ] += 1
            else:
                self.report_surface_feedback_invalid_marker_count[None] += 1
                self.report_surface_feedback_geometry_invalid_marker_count[None] += 1

    def update_surface_feedback_from_mpm_surface_particles(
        self,
        particle_position_m,
        particle_velocity_mps,
        particle_normal,
        particle_area_m2,
        *,
        particle_count: int,
        support_radius_m: float,
        dt_s: float,
    ) -> HibmMpmSurfaceUpdateReport:
        particles = int(particle_count)
        if particles <= 0:
            raise ValueError("particle_count must be positive")
        support_radius = float(support_radius_m)
        if not math.isfinite(support_radius) or support_radius <= 0.0:
            raise ValueError("support_radius_m must be a finite positive number")
        feedback_dt = float(dt_s)
        if not math.isfinite(feedback_dt) or feedback_dt <= 0.0:
            raise ValueError("dt_s must be a finite positive number")
        self._update_surface_feedback_from_mpm_surface_particles_kernel(
            particle_position_m,
            particle_velocity_mps,
            particle_normal,
            particle_area_m2,
            int(self.marker_count),
            particles,
            support_radius,
            feedback_dt,
        )
        return HibmMpmSurfaceUpdateReport(
            updated_marker_count=int(
                self.report_surface_feedback_updated_marker_count[None]
            ),
            invalid_marker_count=int(
                self.report_surface_feedback_invalid_marker_count[None]
            ),
            max_marker_displacement_m=float(
                self.report_surface_feedback_max_displacement_m[None]
            ),
            max_marker_speed_mps=float(
                self.report_surface_feedback_max_speed_mps[None]
            ),
            geometry_updated_marker_count=int(
                self.report_surface_feedback_geometry_updated_marker_count[None]
            ),
            geometry_invalid_marker_count=int(
                self.report_surface_feedback_geometry_invalid_marker_count[None]
            ),
            max_marker_normal_change=float(
                self.report_surface_feedback_max_normal_change[None]
            ),
            max_marker_area_change_m2=float(
                self.report_surface_feedback_max_area_change_m2[None]
            ),
        )

    @ti.kernel
    def _aggregate_region_forces_kernel(
        self,
        marker_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
    ):
        self.report_primary_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_total_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_force_norm_sum_n[None] = 0.0
        self.report_secondary_force_norm_sum_n[None] = 0.0
        self.report_total_force_norm_sum_n[None] = 0.0
        self.report_primary_force_norm_max_n[None] = 0.0
        self.report_secondary_force_norm_max_n[None] = 0.0
        self.report_total_force_norm_max_n[None] = 0.0
        self.report_primary_marker_count[None] = 0
        self.report_secondary_marker_count[None] = 0
        self.report_total_marker_count[None] = 0
        self.report_primary_stress_valid_marker_count[None] = 0
        self.report_primary_stress_invalid_marker_count[None] = 0
        self.report_secondary_stress_valid_marker_count[None] = 0
        self.report_secondary_stress_invalid_marker_count[None] = 0
        for marker in range(marker_count):
            force = self.F_gamma_n[marker]
            force_norm = force.norm()
            self.report_total_force_n[None] += force
            self.report_total_force_norm_sum_n[None] += force_norm
            ti.atomic_max(self.report_total_force_norm_max_n[None], force_norm)
            self.report_total_marker_count[None] += 1
            if self.region_id[marker] == primary_region_id:
                self.report_primary_force_n[None] += force
                self.report_primary_force_norm_sum_n[None] += force_norm
                ti.atomic_max(self.report_primary_force_norm_max_n[None], force_norm)
                self.report_primary_marker_count[None] += 1
                if self._stress_pressure_valid[marker] != 0:
                    self.report_primary_stress_valid_marker_count[None] += 1
                else:
                    self.report_primary_stress_invalid_marker_count[None] += 1
            if self.region_id[marker] == secondary_region_id:
                self.report_secondary_force_n[None] += force
                self.report_secondary_force_norm_sum_n[None] += force_norm
                ti.atomic_max(
                    self.report_secondary_force_norm_max_n[None],
                    force_norm,
                )
                self.report_secondary_marker_count[None] += 1
                if self._stress_pressure_valid[marker] != 0:
                    self.report_secondary_stress_valid_marker_count[None] += 1
                else:
                    self.report_secondary_stress_invalid_marker_count[None] += 1

    def aggregate_region_forces(
        self,
        *,
        primary_region_id: int,
        secondary_region_id: int,
    ) -> HibmMpmSurfaceMarkerForceReport:
        self._aggregate_region_forces_kernel(
            int(self.marker_count),
            int(primary_region_id),
            int(secondary_region_id),
        )
        primary = self._vector_field_tuple(self.report_primary_force_n)
        secondary = self._vector_field_tuple(self.report_secondary_force_n)
        total = self._vector_field_tuple(self.report_total_force_n)
        primary_count = int(self.report_primary_marker_count[None])
        secondary_count = int(self.report_secondary_marker_count[None])
        total_count = int(self.report_total_marker_count[None])
        primary_norm_sum = float(self.report_primary_force_norm_sum_n[None])
        secondary_norm_sum = float(self.report_secondary_force_norm_sum_n[None])
        total_norm_sum = float(self.report_total_force_norm_sum_n[None])
        primary_norm_max = float(self.report_primary_force_norm_max_n[None])
        secondary_norm_max = float(self.report_secondary_force_norm_max_n[None])
        total_norm_max = float(self.report_total_force_norm_max_n[None])
        primary_stress_valid_count = int(
            self.report_primary_stress_valid_marker_count[None]
        )
        primary_stress_invalid_count = int(
            self.report_primary_stress_invalid_marker_count[None]
        )
        secondary_stress_valid_count = int(
            self.report_secondary_stress_valid_marker_count[None]
        )
        secondary_stress_invalid_count = int(
            self.report_secondary_stress_invalid_marker_count[None]
        )
        fluid_reaction = tuple(-component for component in total)
        residual = math.sqrt(
            sum(
                (total_component + reaction_component)
                * (total_component + reaction_component)
                for total_component, reaction_component in zip(
                    total,
                    fluid_reaction,
                    strict=True,
                )
            )
        )
        return HibmMpmSurfaceMarkerForceReport(
            primary_marker_force_n=primary,
            secondary_marker_force_n=secondary,
            total_marker_force_n=total,
            primary_marker_count=primary_count,
            secondary_marker_count=secondary_count,
            total_marker_count=total_count,
            fluid_reaction_force_n=fluid_reaction,
            action_reaction_residual_n=residual,
            primary_stress_valid_marker_count=primary_stress_valid_count,
            primary_stress_invalid_marker_count=primary_stress_invalid_count,
            secondary_stress_valid_marker_count=secondary_stress_valid_count,
            secondary_stress_invalid_marker_count=secondary_stress_invalid_count,
            primary_marker_force_norm_sum_n=primary_norm_sum,
            secondary_marker_force_norm_sum_n=secondary_norm_sum,
            total_marker_force_norm_sum_n=total_norm_sum,
            primary_marker_force_norm_max_n=primary_norm_max,
            secondary_marker_force_norm_max_n=secondary_norm_max,
            total_marker_force_norm_max_n=total_norm_max,
        )

    def marker_force_n(self, marker_index: int) -> tuple[float, float, float]:
        return self._vector_value_tuple(self.F_gamma_n, marker_index)

    def marker_traction_pa(self, marker_index: int) -> tuple[float, float, float]:
        return self._vector_value_tuple(self.t_gamma_pa, marker_index)

    def marker_normal(self, marker_index: int) -> tuple[float, float, float]:
        return self._vector_value_tuple(self.n_gamma, marker_index)

    def marker_velocity_mps(self, marker_index: int) -> tuple[float, float, float]:
        return self._vector_value_tuple(self.v_gamma_mps, marker_index)

    def marker_region_id(self, marker_index: int) -> int:
        return int(self.region_id[int(marker_index)])

    @staticmethod
    def _vector_value_tuple(field, index: int) -> tuple[float, float, float]:
        value = field[int(index)]
        return (float(value[0]), float(value[1]), float(value[2]))

    @staticmethod
    def _vector_field_tuple(field) -> tuple[float, float, float]:
        value = field[None]
        return (float(value[0]), float(value[1]), float(value[2]))


@ti.data_oriented
class HibmMpmIbNodeSearch:
    _NODE_NONE = 0
    _NODE_EXTERNAL_IB = 1
    _NODE_INTERNAL = 2
    _F32_EPSILON = 1.1920928955078125e-7

    def __init__(
        self,
        *,
        grid_nodes: tuple[int, int, int],
        bounds_min_m: Sequence[float],
        bounds_max_m: Sequence[float],
        marker_capacity: int,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        init_taichi(runtime)
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value <= 0 for value in nodes):
            raise ValueError("grid_nodes must contain three positive integers")
        bounds_min = _vector3(bounds_min_m, name="bounds_min_m")
        bounds_max = _vector3(bounds_max_m, name="bounds_max_m")
        if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
            raise ValueError("bounds_max_m must be greater than bounds_min_m")
        if int(marker_capacity) <= 0:
            raise ValueError("marker_capacity must be positive")
        self.grid_nodes = nodes
        self.bounds_min_m = bounds_min
        self.bounds_max_m = bounds_max
        self.spacing_m = tuple(
            (hi - lo) / float(count)
            for lo, hi, count in zip(bounds_min, bounds_max, nodes, strict=True)
        )
        self.marker_capacity = int(marker_capacity)

        self.node_kind_code = ti.field(dtype=ti.i32, shape=nodes)
        self.nearest_marker = ti.field(dtype=ti.i32, shape=nodes)
        self.node_signed_distance_m = ti.field(dtype=ti.f32, shape=nodes)
        self.node_boundary_point_m = ti.Vector.field(3, dtype=ti.f32, shape=nodes)
        self.node_interior_fluid_point_m = ti.Vector.field(3, dtype=ti.f32, shape=nodes)
        self.node_projection_marker_indices = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=nodes,
        )
        self.node_projection_marker_weights = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=nodes,
        )
        # S2-A7: per-node interior-fluid anchor cell. The velocity-Dirichlet
        # row assembly publishes, for every active IB node, the (i, j, k)
        # of a non-obstacle, solve-participating fluid cell: prefilled from
        # the containing cell of node_interior_fluid_point_m, then refined
        # by the row's accepted interior velocity sample (row write-out
        # success path) or the relocated row's claimed fluid cell
        # (relocation success path). (-1, -1, -1) means "no fluid anchor".
        # The stress sampler reads it as a second-stage closure fallback
        # for markers whose marker-level pressure-Neumann anchor source is
        # empty (geometries where the Neumann assembly produced zero rows
        # because every near-boundary fluid cell carries a
        # velocity-Dirichlet row).
        self.node_anchor_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=nodes,
        )

        self.report_near_boundary_node_count = ti.field(dtype=ti.i32, shape=())
        self.report_external_ib_node_count = ti.field(dtype=ti.i32, shape=())
        self.report_internal_node_count = ti.field(dtype=ti.i32, shape=())
        self.report_invalid_projection_count = ti.field(dtype=ti.i32, shape=())
        # Taichi zero-initializes fields, and (0, 0, 0) is a real cell
        # index: establish the unset sentinel before anyone can read the
        # anchors (the velocity-Dirichlet assembly re-resets the full
        # field before every capture pass).
        self.reset_node_anchor_cells()

    @ti.kernel
    def _reset_node_anchor_cells_kernel(self):
        for node in ti.grouped(self.node_anchor_cell):
            self.node_anchor_cell[node] = ti.Vector([-1, -1, -1])

    def reset_node_anchor_cells(self) -> None:
        """Reset every node's interior-fluid anchor cell to unset.

        Runs over the full node grid so nodes that never receive a
        velocity-Dirichlet row (and never prefill) keep the (-1, -1, -1)
        sentinel and stay invisible to the sampler's corner-node scan.
        """
        self._reset_node_anchor_cells_kernel()

    def _default_sign_tolerance_m(self) -> float:
        min_spacing_m = max(min(self.spacing_m), 1.0e-12)
        max_extent_m = max(
            hi - lo for lo, hi in zip(self.bounds_min_m, self.bounds_max_m, strict=True)
        )
        max_coordinate_m = max(
            max(abs(value) for value in self.bounds_min_m),
            max(abs(value) for value in self.bounds_max_m),
        )
        coordinate_scale_m = max(max_extent_m, max_coordinate_m, min_spacing_m, 1.0)
        return max(
            1.0e-12,
            1.0e-6 * min_spacing_m,
            2.0 * self._F32_EPSILON * coordinate_scale_m,
        )

    @ti.func
    def _closest_point_on_triangle(self, position, a, b, c):
        ab = b - a
        ac = c - a
        face_normal = ab.cross(ac)
        normal_norm = face_normal.norm()
        closest = a
        valid = 0
        if normal_norm > 1.0e-12:
            valid = 1
            face_normal = face_normal / normal_norm
            ap = position - a
            d1 = ab.dot(ap)
            d2 = ac.dot(ap)
            if d1 <= 0.0 and d2 <= 0.0:
                closest = a
            else:
                bp = position - b
                d3 = ab.dot(bp)
                d4 = ac.dot(bp)
                if d3 >= 0.0 and d4 <= d3:
                    closest = b
                else:
                    vc = d1 * d4 - d3 * d2
                    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                        edge_fraction = d1 / ti.max(d1 - d3, 1.0e-30)
                        closest = a + edge_fraction * ab
                    else:
                        cp = position - c
                        d5 = ab.dot(cp)
                        d6 = ac.dot(cp)
                        if d6 >= 0.0 and d5 <= d6:
                            closest = c
                        else:
                            vb = d5 * d2 - d1 * d6
                            if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                                edge_fraction = d2 / ti.max(d2 - d6, 1.0e-30)
                                closest = a + edge_fraction * ac
                            else:
                                va = d3 * d6 - d5 * d4
                                if (
                                    va <= 0.0
                                    and d4 - d3 >= 0.0
                                    and d5 - d6 >= 0.0
                                ):
                                    edge_fraction = (d4 - d3) / ti.max(
                                        d4 - d3 + d5 - d6,
                                        1.0e-30,
                                    )
                                    closest = b + edge_fraction * (c - b)
                                else:
                                    denom = ti.max(va + vb + vc, 1.0e-30)
                                    v = vb / denom
                                    w = vc / denom
                                    closest = a + ab * v + ac * w
        return closest, face_normal, valid

    @ti.func
    def _oriented_triangle_normal(self, face_normal, marker_normals, ia, ib, ic):
        normal = face_normal
        average_normal = marker_normals[ia] + marker_normals[ib] + marker_normals[ic]
        average_norm = average_normal.norm()
        if average_norm > 1.0e-12:
            average_normal = average_normal / average_norm
            if normal.dot(average_normal) < 0.0:
                normal = -normal
        return normal

    @ti.func
    def _barycentric_weights_on_triangle(self, point, a, b, c):
        ab = b - a
        ac = c - a
        ap = point - a
        d00 = ab.dot(ab)
        d01 = ab.dot(ac)
        d11 = ac.dot(ac)
        d20 = ap.dot(ab)
        d21 = ap.dot(ac)
        denom = d00 * d11 - d01 * d01
        weights = ti.Vector([1.0, 0.0, 0.0])
        if ti.abs(denom) > 1.0e-30:
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            u = ti.min(ti.max(u, 0.0), 1.0)
            v = ti.min(ti.max(v, 0.0), 1.0)
            w = ti.min(ti.max(w, 0.0), 1.0)
            total = ti.max(u + v + w, 1.0e-30)
            weights = ti.Vector([u / total, v / total, w / total])
        return weights

    @ti.func
    def _nearest_triangle_marker(self, position, marker_positions_m, ia, ib, ic):
        nearest = ia
        offset_a = position - marker_positions_m[ia]
        nearest_distance2 = offset_a.dot(offset_a)
        offset_b = position - marker_positions_m[ib]
        distance_b2 = offset_b.dot(offset_b)
        if distance_b2 < nearest_distance2:
            nearest_distance2 = distance_b2
            nearest = ib
        offset_c = position - marker_positions_m[ic]
        distance_c2 = offset_c.dot(offset_c)
        if distance_c2 < nearest_distance2:
            nearest = ic
        return nearest

    @ti.kernel
    def _search_and_classify_kernel(
        self,
        marker_positions_m: ti.template(),
        marker_normals: ti.template(),
        projection_triangle_indices: ti.template(),
        marker_count: ti.i32,
        projection_triangle_count: ti.i32,
        search_radius_m: ti.f32,
        interior_probe_distance_m: ti.f32,
        sign_tolerance_m: ti.f32,
        classify_far_internal_nodes: ti.i32,
        bounds_min_x_m: ti.f32,
        bounds_min_y_m: ti.f32,
        bounds_min_z_m: ti.f32,
        spacing_x_m: ti.f32,
        spacing_y_m: ti.f32,
        spacing_z_m: ti.f32,
    ):
        self.report_near_boundary_node_count[None] = 0
        self.report_external_ib_node_count[None] = 0
        self.report_internal_node_count[None] = 0
        self.report_invalid_projection_count[None] = 0
        for node in ti.grouped(self.node_kind_code):
            position = ti.Vector(
                [
                    bounds_min_x_m + (ti.cast(node[0], ti.f32) + 0.5) * spacing_x_m,
                    bounds_min_y_m + (ti.cast(node[1], ti.f32) + 0.5) * spacing_y_m,
                    bounds_min_z_m + (ti.cast(node[2], ti.f32) + 0.5) * spacing_z_m,
                ]
            )
            self.node_kind_code[node] = self._NODE_NONE
            self.nearest_marker[node] = -1
            self.node_signed_distance_m[node] = 0.0
            self.node_boundary_point_m[node] = ti.Vector([0.0, 0.0, 0.0])
            self.node_interior_fluid_point_m[node] = ti.Vector([0.0, 0.0, 0.0])
            self.node_projection_marker_indices[node] = ti.Vector([-1, -1, -1])
            self.node_projection_marker_weights[node] = ti.Vector([0.0, 0.0, 0.0])

            nearest = -1
            nearest_distance = 1.0e30
            nearest_signed_distance = 0.0
            nearest_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_projection_indices = ti.Vector([-1, -1, -1])
            nearest_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            nearest_external = -1
            nearest_external_distance = 1.0e30
            nearest_external_signed_distance = 0.0
            nearest_external_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_external_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_external_projection_indices = ti.Vector([-1, -1, -1])
            nearest_external_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            nearest_global = -1
            nearest_global_distance = 1.0e30
            nearest_global_signed_distance = 0.0
            nearest_global_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_global_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_global_projection_indices = ti.Vector([-1, -1, -1])
            nearest_global_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            external_seen = 0
            global_external_seen = 0
            if projection_triangle_count > 0:
                for triangle_index in range(projection_triangle_count):
                    triangle = projection_triangle_indices[triangle_index]
                    ia = triangle.x
                    ib = triangle.y
                    ic = triangle.z
                    closest, face_normal, valid = self._closest_point_on_triangle(
                        position,
                        marker_positions_m[ia],
                        marker_positions_m[ib],
                        marker_positions_m[ic],
                    )
                    if valid != 0:
                        normal = self._oriented_triangle_normal(
                            face_normal,
                            marker_normals,
                            ia,
                            ib,
                            ic,
                        )
                        offset = position - closest
                        distance = offset.norm()
                        signed_distance = offset.dot(normal)
                        projection_weights = self._barycentric_weights_on_triangle(
                            closest,
                            marker_positions_m[ia],
                            marker_positions_m[ib],
                            marker_positions_m[ic],
                        )
                        projection_indices = ti.Vector([ia, ib, ic])
                        marker = self._nearest_triangle_marker(
                            position,
                            marker_positions_m,
                            ia,
                            ib,
                            ic,
                        )
                        if classify_far_internal_nodes != 0:
                            if signed_distance > sign_tolerance_m:
                                global_external_seen = 1
                            if distance < nearest_global_distance:
                                nearest_global_distance = distance
                                nearest_global = marker
                                nearest_global_signed_distance = signed_distance
                                nearest_global_boundary_point = closest
                                nearest_global_normal = normal
                                nearest_global_projection_indices = projection_indices
                                nearest_global_projection_weights = projection_weights
                        if distance < search_radius_m:
                            if signed_distance > sign_tolerance_m:
                                external_seen = 1
                                if distance < nearest_external_distance:
                                    nearest_external_distance = distance
                                    nearest_external = marker
                                    nearest_external_signed_distance = signed_distance
                                    nearest_external_boundary_point = closest
                                    nearest_external_normal = normal
                                    nearest_external_projection_indices = (
                                        projection_indices
                                    )
                                    nearest_external_projection_weights = (
                                        projection_weights
                                    )
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest = marker
                                nearest_signed_distance = signed_distance
                                nearest_boundary_point = closest
                                nearest_normal = normal
                                nearest_projection_indices = projection_indices
                                nearest_projection_weights = projection_weights
            else:
                for marker in range(marker_count):
                    offset = position - marker_positions_m[marker]
                    distance = offset.norm()
                    signed_distance = offset.dot(marker_normals[marker])
                    if classify_far_internal_nodes != 0:
                        if signed_distance > sign_tolerance_m:
                            global_external_seen = 1
                        if distance < nearest_global_distance:
                            nearest_global_distance = distance
                            nearest_global = marker
                            nearest_global_signed_distance = signed_distance
                            nearest_global_boundary_point = marker_positions_m[marker]
                            nearest_global_normal = marker_normals[marker]
                            nearest_global_projection_indices = ti.Vector(
                                [marker, -1, -1]
                            )
                            nearest_global_projection_weights = ti.Vector(
                                [1.0, 0.0, 0.0]
                            )
                    if distance < search_radius_m:
                        if signed_distance > sign_tolerance_m:
                            external_seen = 1
                            if distance < nearest_external_distance:
                                nearest_external_distance = distance
                                nearest_external = marker
                                nearest_external_signed_distance = signed_distance
                                nearest_external_boundary_point = marker_positions_m[
                                    marker
                                ]
                                nearest_external_normal = marker_normals[marker]
                                nearest_external_projection_indices = ti.Vector(
                                    [marker, -1, -1]
                                )
                                nearest_external_projection_weights = ti.Vector(
                                    [1.0, 0.0, 0.0]
                                )
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest = marker
                            nearest_signed_distance = signed_distance
                            nearest_boundary_point = marker_positions_m[marker]
                            nearest_normal = marker_normals[marker]
                            nearest_projection_indices = ti.Vector([marker, -1, -1])
                            nearest_projection_weights = ti.Vector([1.0, 0.0, 0.0])

            if nearest >= 0:
                self.report_near_boundary_node_count[None] += 1
                selected = nearest
                selected_signed_distance = nearest_signed_distance
                boundary_point = nearest_boundary_point
                normal = nearest_normal
                selected_projection_indices = nearest_projection_indices
                selected_projection_weights = nearest_projection_weights
                if external_seen == 1 and nearest_external >= 0:
                    selected = nearest_external
                    selected_signed_distance = nearest_external_signed_distance
                    boundary_point = nearest_external_boundary_point
                    normal = nearest_external_normal
                    selected_projection_indices = nearest_external_projection_indices
                    selected_projection_weights = nearest_external_projection_weights
                self.nearest_marker[node] = selected
                self.node_signed_distance_m[node] = selected_signed_distance
                normal_distance = (position - boundary_point).dot(normal)
                interior_distance = interior_probe_distance_m
                if normal_distance > sign_tolerance_m:
                    interior_distance = normal_distance + interior_probe_distance_m
                self.node_boundary_point_m[node] = boundary_point
                self.node_projection_marker_indices[node] = selected_projection_indices
                self.node_projection_marker_weights[node] = selected_projection_weights
                self.node_interior_fluid_point_m[node] = (
                    boundary_point + normal * interior_distance
                )
                if ti.abs(selected_signed_distance) <= sign_tolerance_m:
                    self.report_invalid_projection_count[None] += 1
                if external_seen == 1:
                    self.node_kind_code[node] = self._NODE_EXTERNAL_IB
                    self.report_external_ib_node_count[None] += 1
                else:
                    self.node_kind_code[node] = self._NODE_INTERNAL
                    self.report_internal_node_count[None] += 1
            elif (
                classify_far_internal_nodes != 0
                and nearest_global >= 0
                and global_external_seen == 0
            ):
                self.node_kind_code[node] = self._NODE_INTERNAL
                self.nearest_marker[node] = nearest_global
                self.node_signed_distance_m[node] = nearest_global_signed_distance
                boundary_point = nearest_global_boundary_point
                normal = nearest_global_normal
                self.node_boundary_point_m[node] = boundary_point
                self.node_projection_marker_indices[node] = (
                    nearest_global_projection_indices
                )
                self.node_projection_marker_weights[node] = (
                    nearest_global_projection_weights
                )
                self.node_interior_fluid_point_m[node] = (
                    boundary_point + normal * interior_probe_distance_m
                )
                self.report_internal_node_count[None] += 1

    @ti.kernel
    def _search_and_classify_grid_fields_kernel(
        self,
        marker_positions_m: ti.template(),
        marker_normals: ti.template(),
        projection_triangle_indices: ti.template(),
        marker_count: ti.i32,
        projection_triangle_count: ti.i32,
        search_radius_m: ti.f32,
        interior_probe_distance_m: ti.f32,
        sign_tolerance_m: ti.f32,
        classify_far_internal_nodes: ti.i32,
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
    ):
        self.report_near_boundary_node_count[None] = 0
        self.report_external_ib_node_count[None] = 0
        self.report_internal_node_count[None] = 0
        self.report_invalid_projection_count[None] = 0
        for node in ti.grouped(self.node_kind_code):
            position = ti.Vector(
                [
                    cell_center_x_m[node[0]],
                    cell_center_y_m[node[1]],
                    cell_center_z_m[node[2]],
                ]
            )
            self.node_kind_code[node] = self._NODE_NONE
            self.nearest_marker[node] = -1
            self.node_signed_distance_m[node] = 0.0
            self.node_boundary_point_m[node] = ti.Vector([0.0, 0.0, 0.0])
            self.node_interior_fluid_point_m[node] = ti.Vector([0.0, 0.0, 0.0])
            self.node_projection_marker_indices[node] = ti.Vector([-1, -1, -1])
            self.node_projection_marker_weights[node] = ti.Vector([0.0, 0.0, 0.0])

            nearest = -1
            nearest_distance = 1.0e30
            nearest_signed_distance = 0.0
            nearest_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_projection_indices = ti.Vector([-1, -1, -1])
            nearest_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            nearest_external = -1
            nearest_external_distance = 1.0e30
            nearest_external_signed_distance = 0.0
            nearest_external_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_external_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_external_projection_indices = ti.Vector([-1, -1, -1])
            nearest_external_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            nearest_global = -1
            nearest_global_distance = 1.0e30
            nearest_global_signed_distance = 0.0
            nearest_global_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_global_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_global_projection_indices = ti.Vector([-1, -1, -1])
            nearest_global_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            external_seen = 0
            global_external_seen = 0
            if projection_triangle_count > 0:
                for triangle_index in range(projection_triangle_count):
                    triangle = projection_triangle_indices[triangle_index]
                    ia = triangle.x
                    ib = triangle.y
                    ic = triangle.z
                    closest, face_normal, valid = self._closest_point_on_triangle(
                        position,
                        marker_positions_m[ia],
                        marker_positions_m[ib],
                        marker_positions_m[ic],
                    )
                    if valid != 0:
                        normal = self._oriented_triangle_normal(
                            face_normal,
                            marker_normals,
                            ia,
                            ib,
                            ic,
                        )
                        offset = position - closest
                        distance = offset.norm()
                        signed_distance = offset.dot(normal)
                        projection_weights = self._barycentric_weights_on_triangle(
                            closest,
                            marker_positions_m[ia],
                            marker_positions_m[ib],
                            marker_positions_m[ic],
                        )
                        projection_indices = ti.Vector([ia, ib, ic])
                        marker = self._nearest_triangle_marker(
                            position,
                            marker_positions_m,
                            ia,
                            ib,
                            ic,
                        )
                        if classify_far_internal_nodes != 0:
                            if signed_distance > sign_tolerance_m:
                                global_external_seen = 1
                            if distance < nearest_global_distance:
                                nearest_global_distance = distance
                                nearest_global = marker
                                nearest_global_signed_distance = signed_distance
                                nearest_global_boundary_point = closest
                                nearest_global_normal = normal
                                nearest_global_projection_indices = projection_indices
                                nearest_global_projection_weights = projection_weights
                        if distance < search_radius_m:
                            if signed_distance > sign_tolerance_m:
                                external_seen = 1
                                if distance < nearest_external_distance:
                                    nearest_external_distance = distance
                                    nearest_external = marker
                                    nearest_external_signed_distance = signed_distance
                                    nearest_external_boundary_point = closest
                                    nearest_external_normal = normal
                                    nearest_external_projection_indices = (
                                        projection_indices
                                    )
                                    nearest_external_projection_weights = (
                                        projection_weights
                                    )
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest = marker
                                nearest_signed_distance = signed_distance
                                nearest_boundary_point = closest
                                nearest_normal = normal
                                nearest_projection_indices = projection_indices
                                nearest_projection_weights = projection_weights
            else:
                for marker in range(marker_count):
                    offset = position - marker_positions_m[marker]
                    distance = offset.norm()
                    signed_distance = offset.dot(marker_normals[marker])
                    if classify_far_internal_nodes != 0:
                        if signed_distance > sign_tolerance_m:
                            global_external_seen = 1
                        if distance < nearest_global_distance:
                            nearest_global_distance = distance
                            nearest_global = marker
                            nearest_global_signed_distance = signed_distance
                            nearest_global_boundary_point = marker_positions_m[marker]
                            nearest_global_normal = marker_normals[marker]
                            nearest_global_projection_indices = ti.Vector(
                                [marker, -1, -1]
                            )
                            nearest_global_projection_weights = ti.Vector(
                                [1.0, 0.0, 0.0]
                            )
                    if distance < search_radius_m:
                        if signed_distance > sign_tolerance_m:
                            external_seen = 1
                            if distance < nearest_external_distance:
                                nearest_external_distance = distance
                                nearest_external = marker
                                nearest_external_signed_distance = signed_distance
                                nearest_external_boundary_point = marker_positions_m[
                                    marker
                                ]
                                nearest_external_normal = marker_normals[marker]
                                nearest_external_projection_indices = ti.Vector(
                                    [marker, -1, -1]
                                )
                                nearest_external_projection_weights = ti.Vector(
                                    [1.0, 0.0, 0.0]
                                )
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest = marker
                            nearest_signed_distance = signed_distance
                            nearest_boundary_point = marker_positions_m[marker]
                            nearest_normal = marker_normals[marker]
                            nearest_projection_indices = ti.Vector([marker, -1, -1])
                            nearest_projection_weights = ti.Vector([1.0, 0.0, 0.0])

            if nearest >= 0:
                self.report_near_boundary_node_count[None] += 1
                selected = nearest
                selected_signed_distance = nearest_signed_distance
                boundary_point = nearest_boundary_point
                normal = nearest_normal
                selected_projection_indices = nearest_projection_indices
                selected_projection_weights = nearest_projection_weights
                if external_seen == 1 and nearest_external >= 0:
                    selected = nearest_external
                    selected_signed_distance = nearest_external_signed_distance
                    boundary_point = nearest_external_boundary_point
                    normal = nearest_external_normal
                    selected_projection_indices = nearest_external_projection_indices
                    selected_projection_weights = nearest_external_projection_weights
                self.nearest_marker[node] = selected
                self.node_signed_distance_m[node] = selected_signed_distance
                normal_distance = (position - boundary_point).dot(normal)
                interior_distance = interior_probe_distance_m
                if normal_distance > sign_tolerance_m:
                    interior_distance = normal_distance + interior_probe_distance_m
                self.node_boundary_point_m[node] = boundary_point
                self.node_projection_marker_indices[node] = selected_projection_indices
                self.node_projection_marker_weights[node] = selected_projection_weights
                self.node_interior_fluid_point_m[node] = (
                    boundary_point + normal * interior_distance
                )
                if ti.abs(selected_signed_distance) <= sign_tolerance_m:
                    self.report_invalid_projection_count[None] += 1
                if external_seen == 1:
                    self.node_kind_code[node] = self._NODE_EXTERNAL_IB
                    self.report_external_ib_node_count[None] += 1
                else:
                    self.node_kind_code[node] = self._NODE_INTERNAL
                    self.report_internal_node_count[None] += 1
            elif (
                classify_far_internal_nodes != 0
                and nearest_global >= 0
                and global_external_seen == 0
            ):
                self.node_kind_code[node] = self._NODE_INTERNAL
                self.nearest_marker[node] = nearest_global
                self.node_signed_distance_m[node] = nearest_global_signed_distance
                boundary_point = nearest_global_boundary_point
                normal = nearest_global_normal
                self.node_boundary_point_m[node] = boundary_point
                self.node_projection_marker_indices[node] = (
                    nearest_global_projection_indices
                )
                self.node_projection_marker_weights[node] = (
                    nearest_global_projection_weights
                )
                self.node_interior_fluid_point_m[node] = (
                    boundary_point + normal * interior_probe_distance_m
                )
                self.report_internal_node_count[None] += 1

    def _validate_search_inputs(
        self,
        markers: HibmMpmSurfaceMarkers,
        *,
        search_radius_m: float,
        interior_probe_distance_m: float,
        sign_tolerance_m: float | None,
    ) -> tuple[float, float, float]:
        if int(markers.marker_count) > self.marker_capacity:
            raise ValueError("markers.marker_count exceeds marker_capacity")
        search_radius = float(search_radius_m)
        if not math.isfinite(search_radius) or search_radius <= 0.0:
            raise ValueError("search_radius_m must be a finite positive number")
        probe_distance = float(interior_probe_distance_m)
        if not math.isfinite(probe_distance) or probe_distance <= 0.0:
            raise ValueError(
                "interior_probe_distance_m must be a finite positive number"
            )
        sign_tolerance = (
            self._default_sign_tolerance_m()
            if sign_tolerance_m is None
            else float(sign_tolerance_m)
        )
        if not math.isfinite(sign_tolerance) or sign_tolerance < 0.0:
            raise ValueError("sign_tolerance_m must be a finite non-negative number")
        return search_radius, probe_distance, sign_tolerance

    def search_and_classify(
        self,
        markers: HibmMpmSurfaceMarkers,
        *,
        search_radius_m: float,
        interior_probe_distance_m: float,
        sign_tolerance_m: float | None = None,
        classify_far_internal_nodes: bool = False,
    ) -> HibmMpmIbNodeSearchReport:
        search_radius, probe_distance, sign_tolerance = self._validate_search_inputs(
            markers,
            search_radius_m=search_radius_m,
            interior_probe_distance_m=interior_probe_distance_m,
            sign_tolerance_m=sign_tolerance_m,
        )
        self._search_and_classify_kernel(
            markers.x_gamma_m,
            markers.n_gamma,
            markers.projection_triangle_indices,
            int(markers.marker_count),
            int(markers.projection_triangle_count),
            search_radius,
            probe_distance,
            sign_tolerance,
            1 if bool(classify_far_internal_nodes) else 0,
            float(self.bounds_min_m[0]),
            float(self.bounds_min_m[1]),
            float(self.bounds_min_m[2]),
            float(self.spacing_m[0]),
            float(self.spacing_m[1]),
            float(self.spacing_m[2]),
        )
        return HibmMpmIbNodeSearchReport(
            near_boundary_node_count=int(self.report_near_boundary_node_count[None]),
            external_ib_node_count=int(self.report_external_ib_node_count[None]),
            internal_node_count=int(self.report_internal_node_count[None]),
            invalid_projection_count=int(self.report_invalid_projection_count[None]),
        )

    def search_and_classify_grid_fields(
        self,
        markers: HibmMpmSurfaceMarkers,
        *,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        search_radius_m: float,
        interior_probe_distance_m: float,
        sign_tolerance_m: float | None = None,
        classify_far_internal_nodes: bool = False,
    ) -> HibmMpmIbNodeSearchReport:
        search_radius, probe_distance, sign_tolerance = self._validate_search_inputs(
            markers,
            search_radius_m=search_radius_m,
            interior_probe_distance_m=interior_probe_distance_m,
            sign_tolerance_m=sign_tolerance_m,
        )
        self._search_and_classify_grid_fields_kernel(
            markers.x_gamma_m,
            markers.n_gamma,
            markers.projection_triangle_indices,
            int(markers.marker_count),
            int(markers.projection_triangle_count),
            search_radius,
            probe_distance,
            sign_tolerance,
            1 if bool(classify_far_internal_nodes) else 0,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
        )
        return HibmMpmIbNodeSearchReport(
            near_boundary_node_count=int(self.report_near_boundary_node_count[None]),
            external_ib_node_count=int(self.report_external_ib_node_count[None]),
            internal_node_count=int(self.report_internal_node_count[None]),
            invalid_projection_count=int(self.report_invalid_projection_count[None]),
        )

    def node_kind(self, node_index: tuple[int, int, int]) -> str:
        code = int(self.node_kind_code[self._node_index(node_index)])
        if code == self._NODE_EXTERNAL_IB:
            return "external_ib"
        if code == self._NODE_INTERNAL:
            return "internal"
        return "none"

    def nearest_marker_index(self, node_index: tuple[int, int, int]) -> int:
        return int(self.nearest_marker[self._node_index(node_index)])

    def signed_distance_m(self, node_index: tuple[int, int, int]) -> float:
        return float(self.node_signed_distance_m[self._node_index(node_index)])

    def boundary_point_m(
        self,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return self._vector_field_value_tuple(self.node_boundary_point_m, node_index)

    def interior_fluid_point_m(
        self,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return self._vector_field_value_tuple(
            self.node_interior_fluid_point_m,
            node_index,
        )

    def _node_index(self, node_index: tuple[int, int, int]) -> tuple[int, int, int]:
        index = tuple(int(value) for value in node_index)
        if len(index) != 3 or any(
            value < 0 or value >= limit
            for value, limit in zip(index, self.grid_nodes, strict=True)
        ):
            raise IndexError("node_index out of range")
        return index

    def _vector_field_value_tuple(
        self,
        field,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        value = field[self._node_index(node_index)]
        return (float(value[0]), float(value[1]), float(value[2]))


@ti.data_oriented
class HibmMpmIbBoundaryConditions:
    _NODE_EXTERNAL_IB = HibmMpmIbNodeSearch._NODE_EXTERNAL_IB
    _NODE_INTERNAL = HibmMpmIbNodeSearch._NODE_INTERNAL

    def __init__(
        self,
        *,
        grid_nodes: tuple[int, int, int],
        marker_capacity: int,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        init_taichi(runtime)
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value <= 0 for value in nodes):
            raise ValueError("grid_nodes must contain three positive integers")
        if int(marker_capacity) <= 0:
            raise ValueError("marker_capacity must be positive")
        self.grid_nodes = nodes
        self.marker_capacity = int(marker_capacity)

        self.active_ib_node = ti.field(dtype=ti.i32, shape=nodes)
        self.velocity_dirichlet_mps_field = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=nodes,
        )
        self.pressure_neumann_normal_field = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=nodes,
        )
        self.pressure_neumann_gradient_field = ti.field(dtype=ti.f32, shape=nodes)
        self.marker_pressure_neumann_gradient_field = ti.field(
            dtype=ti.f32,
            shape=self.marker_capacity,
        )
        self.marker_pressure_neumann_row_count = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self.marker_pressure_neumann_candidate_node_count = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        extra_coupling_shape = nodes + (PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS,)
        self._fallback_pressure_coupling_extra_neighbor = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=extra_coupling_shape,
        )
        self._fallback_pressure_coupling_extra_coefficient = ti.field(
            dtype=ti.f32,
            shape=extra_coupling_shape,
        )
        self._fallback_pressure_interface_row_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self._fallback_pressure_interface_row_owner = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=1,
        )
        self._fallback_pressure_interface_row_neighbor = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=1,
        )
        self._fallback_pressure_interface_row_transmissibility = ti.field(
            dtype=ti.f32,
            shape=1,
        )

        self.report_no_slip_dirichlet_count = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_neumann_count = ti.field(dtype=ti.i32, shape=())
        self.report_inactive_internal_node_count = ti.field(dtype=ti.i32, shape=())
        self.report_boundary_condition_max_abs_velocity = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_matrix_rows = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_neumann_rhs_integral = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_neumann_max_abs_rhs = ti.field(dtype=ti.f32, shape=())
        self.report_pressure_neumann_invalid_reconstruction_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_invalid_unreconstructable_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_invalid_bad_marker_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_invalid_nonpositive_volume_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_min_reconstruction_gap_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_max_reconstruction_gap_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_max_transmissibility_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_max_raw_transmissibility_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_max_transmissibility_limit_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_transmissibility_capped_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_max_diagonal_per_m2 = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_skipped_velocity_dirichlet_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_skipped_pressure_boundary_adjacent_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_active_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_max_rows_per_marker = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_node_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_max_abs = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_raw_max_abs = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_limited_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_boundary_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_obstacle_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_max_abs_velocity = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_raw_max_abs_velocity = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_boundary_velocity_only_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_reconstruction_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_no_fluid_sample_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_nonpositive_gap_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_node_behind_boundary_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_narrow_gap_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_relocated_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_relocation_merged_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_skipped_obstacle_owner_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_relocated_obstacle_owner_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_duplicate_owner_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_overflow_owner_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_max_owner_slot_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_neumann_invalid_diag_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_neumann_invalid_diag_reason = ti.field(
            dtype=ti.i32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.pressure_neumann_invalid_diag_marker = ti.field(
            dtype=ti.i32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.pressure_neumann_invalid_diag_node = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.pressure_neumann_invalid_diag_owner = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.pressure_neumann_invalid_diag_neighbor = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.pressure_neumann_invalid_diag_anchor = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.pressure_neumann_invalid_diag_node_distance_m = ti.field(
            dtype=ti.f32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.pressure_neumann_invalid_diag_normal_denominator_m = ti.field(
            dtype=ti.f32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.pressure_neumann_invalid_diag_reconstruction_gap_m = ti.field(
            dtype=ti.f32,
            shape=PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        self.report_velocity_dirichlet_invalid_node_beyond_interior_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_min_projection_weight = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_max_projection_weight = ti.field(
            dtype=ti.f32,
            shape=(),
        )

    @ti.kernel
    def _build_from_search_kernel(
        self,
        node_kind_code: ti.template(),
        nearest_marker: ti.template(),
        projection_marker_indices: ti.template(),
        projection_marker_weights: ti.template(),
        marker_velocities_mps: ti.template(),
        marker_normals: ti.template(),
        marker_pressure_neumann_gradient_pa_per_m: ti.template(),
        marker_count: ti.i32,
    ):
        self.report_no_slip_dirichlet_count[None] = 0
        self.report_pressure_neumann_count[None] = 0
        self.report_inactive_internal_node_count[None] = 0
        self.report_boundary_condition_max_abs_velocity[None] = 0.0
        for node in ti.grouped(self.active_ib_node):
            self.active_ib_node[node] = 0
            self.velocity_dirichlet_mps_field[node] = ti.Vector([0.0, 0.0, 0.0])
            self.pressure_neumann_normal_field[node] = ti.Vector([0.0, 0.0, 0.0])
            self.pressure_neumann_gradient_field[node] = 0.0

            if node_kind_code[node] == self._NODE_EXTERNAL_IB:
                marker = nearest_marker[node]
                if 0 <= marker < marker_count:
                    target_velocity = ti.Vector([0.0, 0.0, 0.0])
                    target_normal = ti.Vector([0.0, 0.0, 0.0])
                    target_gradient = 0.0
                    total_weight = 0.0
                    indices = projection_marker_indices[node]
                    weights = projection_marker_weights[node]
                    for slot in ti.static(range(3)):
                        projection_marker = indices[slot]
                        weight = weights[slot]
                        if (
                            0 <= projection_marker < marker_count
                            and weight > 0.0
                        ):
                            target_velocity += (
                                weight * marker_velocities_mps[projection_marker]
                            )
                            target_normal += weight * marker_normals[projection_marker]
                            target_gradient += (
                                weight
                                * marker_pressure_neumann_gradient_pa_per_m[
                                    projection_marker
                                ]
                            )
                            total_weight += weight
                    if total_weight > 1.0e-12:
                        inv_weight = 1.0 / total_weight
                        target_velocity *= inv_weight
                        target_normal *= inv_weight
                        target_gradient *= inv_weight
                        normal_norm = target_normal.norm()
                        if normal_norm > 1.0e-12:
                            target_normal = target_normal / normal_norm
                        else:
                            target_normal = marker_normals[marker]
                    else:
                        target_velocity = marker_velocities_mps[marker]
                        target_normal = marker_normals[marker]
                        target_gradient = marker_pressure_neumann_gradient_pa_per_m[
                            marker
                        ]
                    self.active_ib_node[node] = 1
                    self.velocity_dirichlet_mps_field[node] = target_velocity
                    self.pressure_neumann_normal_field[node] = target_normal
                    self.pressure_neumann_gradient_field[node] = target_gradient
                    self.report_no_slip_dirichlet_count[None] += 1
                    self.report_pressure_neumann_count[None] += 1
                    ti.atomic_max(
                        self.report_boundary_condition_max_abs_velocity[None],
                        ti.max(
                            ti.max(ti.abs(target_velocity.x), ti.abs(target_velocity.y)),
                            ti.abs(target_velocity.z),
                        ),
                    )
            elif node_kind_code[node] == self._NODE_INTERNAL:
                self.report_inactive_internal_node_count[None] += 1

    def build_from_search(
        self,
        search: HibmMpmIbNodeSearch,
        markers: HibmMpmSurfaceMarkers,
        *,
        marker_pressure_neumann_gradient_pa_per_m: Sequence[float],
    ) -> HibmMpmIbBoundaryConditionReport:
        self._validate_search_and_markers(search, markers)
        if len(marker_pressure_neumann_gradient_pa_per_m) != markers.marker_count:
            raise ValueError(
                "marker_pressure_neumann_gradient_pa_per_m must match marker_count"
            )
        for marker, gradient in enumerate(marker_pressure_neumann_gradient_pa_per_m):
            value = float(gradient)
            if not math.isfinite(value):
                raise ValueError(
                    "marker_pressure_neumann_gradient_pa_per_m must be finite"
                )
            self.marker_pressure_neumann_gradient_field[marker] = value

        return self.build_from_search_device_fields(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                self.marker_pressure_neumann_gradient_field
            ),
        )

    def build_from_search_device_fields(
        self,
        search: HibmMpmIbNodeSearch,
        markers: HibmMpmSurfaceMarkers,
        *,
        marker_pressure_neumann_gradient_pa_per_m_field,
    ) -> HibmMpmIbBoundaryConditionReport:
        self._validate_search_and_markers(search, markers)
        self._build_from_search_kernel(
            search.node_kind_code,
            search.nearest_marker,
            search.node_projection_marker_indices,
            search.node_projection_marker_weights,
            markers.v_gamma_mps,
            markers.n_gamma,
            marker_pressure_neumann_gradient_pa_per_m_field,
            int(markers.marker_count),
        )
        return HibmMpmIbBoundaryConditionReport(
            no_slip_dirichlet_count=int(self.report_no_slip_dirichlet_count[None]),
            pressure_neumann_count=int(self.report_pressure_neumann_count[None]),
            inactive_internal_node_count=int(
                self.report_inactive_internal_node_count[None]
            ),
            max_abs_velocity_mps=float(
                self.report_boundary_condition_max_abs_velocity[None]
            ),
        )

    def _validate_search_and_markers(
        self,
        search: HibmMpmIbNodeSearch,
        markers: HibmMpmSurfaceMarkers,
    ) -> None:
        if tuple(search.grid_nodes) != self.grid_nodes:
            raise ValueError("search.grid_nodes must match boundary grid_nodes")
        if int(markers.marker_count) > self.marker_capacity:
            raise ValueError("markers.marker_count exceeds marker_capacity")
        if int(search.marker_capacity) > self.marker_capacity:
            raise ValueError("search.marker_capacity exceeds marker_capacity")

    @ti.kernel
    def _assemble_velocity_dirichlet_boundary_rows_kernel(
        self,
        velocity_dirichlet_active: ti.template(),
        velocity_dirichlet_value_mps: ti.template(),
        obstacle_field: ti.template(),
    ):
        self.report_velocity_dirichlet_boundary_rows[None] = 0
        self.report_velocity_dirichlet_obstacle_rows[None] = 0
        self.report_velocity_dirichlet_max_abs_velocity[None] = 0.0
        self.report_velocity_dirichlet_raw_max_abs_velocity[None] = 0.0
        self.report_velocity_dirichlet_boundary_velocity_only_rows[None] = 0
        self.report_velocity_dirichlet_invalid_reconstruction_rows[None] = 0
        self.report_velocity_dirichlet_invalid_no_fluid_sample_rows[None] = 0
        self.report_velocity_dirichlet_invalid_nonpositive_gap_rows[None] = 0
        self.report_velocity_dirichlet_invalid_node_behind_boundary_rows[None] = 0
        self.report_velocity_dirichlet_invalid_node_beyond_interior_rows[None] = 0
        self.report_velocity_dirichlet_min_projection_weight[None] = 1.0e30
        self.report_velocity_dirichlet_max_projection_weight[None] = 0.0
        for node in ti.grouped(velocity_dirichlet_active):
            velocity_dirichlet_active[node] = 0
            velocity_dirichlet_value_mps[node] = ti.Vector([0.0, 0.0, 0.0])
            if self.active_ib_node[node] == 1:
                if obstacle_field[node] == 0:
                    target_velocity = self.velocity_dirichlet_mps_field[node]
                    velocity_dirichlet_active[node] = 1
                    velocity_dirichlet_value_mps[node] = target_velocity
                    ti.atomic_add(
                        self.report_velocity_dirichlet_boundary_rows[None],
                        1,
                    )
                    ti.atomic_max(
                        self.report_velocity_dirichlet_max_abs_velocity[None],
                        ti.max(
                            ti.max(ti.abs(target_velocity.x), ti.abs(target_velocity.y)),
                            ti.abs(target_velocity.z),
                        ),
                    )
                else:
                    ti.atomic_add(
                        self.report_velocity_dirichlet_obstacle_rows[None],
                        1,
                    )

    def assemble_velocity_dirichlet_boundary_rows(
        self,
        velocity_dirichlet_active,
        velocity_dirichlet_value_mps,
        obstacle_field,
    ) -> HibmMpmVelocityDirichletBoundaryReport:
        self._assemble_velocity_dirichlet_boundary_rows_kernel(
            velocity_dirichlet_active,
            velocity_dirichlet_value_mps,
            obstacle_field,
        )
        return HibmMpmVelocityDirichletBoundaryReport(
            active_velocity_dirichlet_rows=int(
                self.report_velocity_dirichlet_boundary_rows[None]
            ),
            inactive_obstacle_rows=int(
                self.report_velocity_dirichlet_obstacle_rows[None]
            ),
            max_abs_velocity_mps=float(
                self.report_velocity_dirichlet_max_abs_velocity[None]
            ),
            raw_reconstructed_max_abs_velocity_mps=float(
                self.report_velocity_dirichlet_raw_max_abs_velocity[None]
            ),
            boundary_velocity_only_row_count=int(
                self.report_velocity_dirichlet_boundary_velocity_only_rows[None]
            ),
            unassigned_region_active_rows=int(
                self.report_velocity_dirichlet_boundary_rows[None]
            ),
        )

    @staticmethod
    def _velocity_dirichlet_region_row_counts(
        velocity_dirichlet_active,
        velocity_dirichlet_marker_region_id,
        *,
        primary_region_id: int | None,
        secondary_region_id: int | None,
    ) -> tuple[int, int, int, int]:
        active_mask = velocity_dirichlet_active.to_numpy() != 0
        active_rows = int(active_mask.sum())
        if active_rows <= 0 or velocity_dirichlet_marker_region_id is None:
            return (0, 0, 0, active_rows)
        region_id = velocity_dirichlet_marker_region_id.to_numpy()
        primary_mask = np.zeros_like(active_mask, dtype=bool)
        if primary_region_id is not None and int(primary_region_id) >= 0:
            primary_mask = active_mask & (region_id == int(primary_region_id))
        secondary_mask = np.zeros_like(active_mask, dtype=bool)
        if secondary_region_id is not None and int(secondary_region_id) >= 0:
            secondary_mask = (
                active_mask
                & (region_id == int(secondary_region_id))
                & ~primary_mask
            )
        unassigned_mask = active_mask & (region_id < 0)
        assigned_known_mask = primary_mask | secondary_mask | unassigned_mask
        other_mask = active_mask & ~assigned_known_mask
        return (
            int(primary_mask.sum()),
            int(secondary_mask.sum()),
            int(other_mask.sum()),
            int(unassigned_mask.sum()),
        )

    @ti.func
    def _axis_grid_coordinate_device(
        self,
        value: ti.f32,
        faces: ti.template(),
        centers: ti.template(),
        count: ti.i32,
    ):
        coordinate = 0.0
        if value <= centers[0]:
            half_width = ti.max(centers[0] - faces[0], 1.0e-18)
            coordinate = -0.5 * (centers[0] - value) / half_width
        elif value >= centers[count - 1]:
            half_width = ti.max(faces[count] - centers[count - 1], 1.0e-18)
            coordinate = ti.cast(count - 1, ti.f32) + 0.5 * (
                value - centers[count - 1]
            ) / half_width
        else:
            lower = 0
            upper = count - 1
            while upper - lower > 1:
                middle = (lower + upper) // 2
                if value >= centers[middle]:
                    lower = middle
                else:
                    upper = middle
            upper = ti.min(lower + 1, count - 1)
            distance = ti.max(centers[upper] - centers[lower], 1.0e-18)
            coordinate = ti.cast(lower, ti.f32) + (value - centers[lower]) / distance
        return coordinate

    @ti.func
    def _grid_coordinate_from_fields(
        self,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        return ti.Vector(
            [
                self._axis_grid_coordinate_device(
                    position.x,
                    cell_face_x_m,
                    cell_center_x_m,
                    nx,
                ),
                self._axis_grid_coordinate_device(
                    position.y,
                    cell_face_y_m,
                    cell_center_y_m,
                    ny,
                ),
                self._axis_grid_coordinate_device(
                    position.z,
                    cell_face_z_m,
                    cell_center_z_m,
                    nz,
                ),
            ]
        )

    @ti.func
    def _sample_fluid_velocity_trilinear(
        self,
        velocity_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = ti.Vector([0.0, 0.0, 0.0])
        fluid_weight = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            weight = wx * wy * wz
            if obstacle_field[i0 + oi, j0 + oj, k0 + ok] == 0:
                value += weight * velocity_field[i0 + oi, j0 + oj, k0 + ok]
                fluid_weight += weight
        if fluid_weight > 1.0e-12:
            value /= fluid_weight
        return value, fluid_weight

    @ti.func
    def _sample_pressure_trilinear(
        self,
        pressure_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = ti.cast(0.0, ti.f64)
        fluid_weight = ti.cast(0.0, ti.f64)
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            weight = wx * wy * wz
            if obstacle_field[i0 + oi, j0 + oj, k0 + ok] == 0:
                value += weight * pressure_field[i0 + oi, j0 + oj, k0 + ok]
                fluid_weight += weight
        if fluid_weight > 1.0e-12:
            value /= fluid_weight
        return value, fluid_weight

    @ti.func
    def _walk_interior_velocity_sample(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        boundary_point,
        walk_normal,
        base_distance,
        step_m,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        sample_velocity = ti.Vector([0.0, 0.0, 0.0])
        sample_weight = 0.0
        sample_distance = base_distance
        found = 0
        for step_index in ti.static(range(5)):
            if found == 0:
                candidate_distance = base_distance + step_m * ti.cast(
                    step_index, ti.f32
                )
                candidate_point = boundary_point + walk_normal * candidate_distance
                candidate_coordinate = self._grid_coordinate_from_fields(
                    candidate_point,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                velocity_value, weight = self._sample_fluid_velocity_trilinear(
                    velocity_field,
                    obstacle_field,
                    candidate_coordinate.x,
                    candidate_coordinate.y,
                    candidate_coordinate.z,
                    nx,
                    ny,
                    nz,
                )
                if weight > 1.0e-12:
                    sample_velocity = velocity_value
                    sample_weight = weight
                    sample_distance = candidate_distance
                    found = 1
        return sample_velocity, sample_weight, sample_distance

    @ti.kernel
    def _assemble_velocity_dirichlet_reconstructed_boundary_rows_kernel(
        self,
        velocity_dirichlet_active: ti.template(),
        velocity_dirichlet_value_mps: ti.template(),
        velocity_dirichlet_projection_weight: ti.template(),
        obstacle_field: ti.template(),
        velocity_field: ti.template(),
        node_boundary_point_m: ti.template(),
        node_interior_fluid_point_m: ti.template(),
        node_anchor_cell: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        preserve_existing_rows: ti.i32,
        interpolate_interior_velocity: ti.i32,
    ):
        self.report_velocity_dirichlet_boundary_rows[None] = 0
        self.report_velocity_dirichlet_obstacle_rows[None] = 0
        self.report_velocity_dirichlet_max_abs_velocity[None] = 0.0
        self.report_velocity_dirichlet_raw_max_abs_velocity[None] = 0.0
        self.report_velocity_dirichlet_boundary_velocity_only_rows[None] = 0
        self.report_velocity_dirichlet_invalid_reconstruction_rows[None] = 0
        self.report_velocity_dirichlet_invalid_no_fluid_sample_rows[None] = 0
        self.report_velocity_dirichlet_invalid_nonpositive_gap_rows[None] = 0
        self.report_velocity_dirichlet_invalid_node_behind_boundary_rows[None] = 0
        self.report_velocity_dirichlet_invalid_node_beyond_interior_rows[None] = 0
        self.report_velocity_dirichlet_narrow_gap_rows[None] = 0
        self.report_velocity_dirichlet_min_projection_weight[None] = 1.0e30
        self.report_velocity_dirichlet_max_projection_weight[None] = 0.0
        for node in ti.grouped(velocity_dirichlet_active):
            if preserve_existing_rows == 0:
                velocity_dirichlet_active[node] = 0
                velocity_dirichlet_value_mps[node] = ti.Vector([0.0, 0.0, 0.0])
                velocity_dirichlet_projection_weight[node] = 0.0
            if self.active_ib_node[node] == 1:
                if obstacle_field[node] == 0:
                    boundary_velocity = self.velocity_dirichlet_mps_field[node]
                    normal = self.pressure_neumann_normal_field[node]
                    boundary_point = node_boundary_point_m[node]
                    interior_point = node_interior_fluid_point_m[node]
                    node_position = ti.Vector(
                        [
                            cell_center_x_m[node[0]],
                            cell_center_y_m[node[1]],
                            cell_center_z_m[node[2]],
                        ]
                    )
                    normal_denominator = (interior_point - boundary_point).dot(normal)
                    normal_distance = (node_position - boundary_point).dot(normal)
                    node_width_x = cell_face_x_m[node[0] + 1] - cell_face_x_m[node[0]]
                    node_width_y = cell_face_y_m[node[1] + 1] - cell_face_y_m[node[1]]
                    node_width_z = cell_face_z_m[node[2] + 1] - cell_face_z_m[node[2]]
                    walk_step_m = 0.5 / ti.max(
                        ti.abs(normal.x) / ti.max(node_width_x, 1.0e-12)
                        + ti.abs(normal.y) / ti.max(node_width_y, 1.0e-12)
                        + ti.abs(normal.z) / ti.max(node_width_z, 1.0e-12),
                        1.0e-12,
                    )
                    interior_velocity, fluid_weight, sample_denominator = (
                        self._walk_interior_velocity_sample(
                            velocity_field,
                            obstacle_field,
                            boundary_point,
                            normal,
                            normal_denominator,
                            walk_step_m,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            nx,
                            ny,
                            nz,
                        )
                    )
                    fallback_velocity = interior_velocity
                    fallback_denominator = sample_denominator
                    fallback_alpha = 0.0
                    fallback_anchor_i = -1
                    fallback_anchor_j = -1
                    fallback_anchor_k = -1
                    normal_segment_reconstructable = 0
                    if (
                        fluid_weight > 1.0e-12
                        and sample_denominator > 1.0e-12
                        and normal_distance >= 0.0
                        and normal_distance <= sample_denominator
                    ):
                        normal_segment_reconstructable = 1
                    fallback_reconstructable = 0
                    if (
                        normal_segment_reconstructable == 0
                        and fluid_weight > 1.0e-12
                        and sample_denominator > 1.0e-12
                        and normal_denominator > 1.0e-12
                    ):
                        node_offset = node_position - boundary_point
                        node_line_distance = ti.sqrt(
                            ti.max(node_offset.dot(node_offset), 0.0)
                        )
                        if node_line_distance > 1.0e-12:
                            fallback_normal = node_offset / node_line_distance
                            fallback_walk_step_m = 0.5 / ti.max(
                                ti.abs(fallback_normal.x) / ti.max(node_width_x, 1.0e-12)
                                + ti.abs(fallback_normal.y) / ti.max(node_width_y, 1.0e-12)
                                + ti.abs(fallback_normal.z) / ti.max(node_width_z, 1.0e-12),
                                1.0e-12,
                            )
                            (
                                fallback_velocity,
                                fallback_weight,
                                fallback_denominator,
                            ) = self._walk_interior_velocity_sample(
                                velocity_field,
                                obstacle_field,
                                boundary_point,
                                fallback_normal,
                                node_line_distance + fallback_walk_step_m,
                                fallback_walk_step_m,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                            if (
                                fallback_weight > 1.0e-12
                                and fallback_denominator
                                > node_line_distance + 1.0e-12
                            ):
                                fallback_alpha = ti.min(
                                    ti.max(
                                        node_line_distance
                                        / ti.max(fallback_denominator, 1.0e-12),
                                        0.0,
                                    ),
                                    1.0,
                                )
                                fallback_sample_point = (
                                    boundary_point
                                    + fallback_normal * fallback_denominator
                                )
                                fallback_sample_coordinate = (
                                    self._grid_coordinate_from_fields(
                                        fallback_sample_point,
                                        cell_face_x_m,
                                        cell_face_y_m,
                                        cell_face_z_m,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                        nx,
                                        ny,
                                        nz,
                                    )
                                )
                                fallback_anchor_i = ti.min(
                                    ti.max(
                                        ti.floor(
                                            fallback_sample_coordinate.x + 0.5,
                                            ti.i32,
                                        ),
                                        0,
                                    ),
                                    nx - 1,
                                )
                                fallback_anchor_j = ti.min(
                                    ti.max(
                                        ti.floor(
                                            fallback_sample_coordinate.y + 0.5,
                                            ti.i32,
                                        ),
                                        0,
                                    ),
                                    ny - 1,
                                )
                                fallback_anchor_k = ti.min(
                                    ti.max(
                                        ti.floor(
                                            fallback_sample_coordinate.z + 0.5,
                                            ti.i32,
                                        ),
                                        0,
                                    ),
                                    nz - 1,
                                )
                                if (
                                    obstacle_field[
                                        fallback_anchor_i,
                                        fallback_anchor_j,
                                        fallback_anchor_k,
                                    ]
                                    == 0
                                ):
                                    fallback_reconstructable = 1
                    target_velocity = boundary_velocity
                    reconstruction_alpha = 0.0
                    if normal_segment_reconstructable != 0:
                        alpha = ti.min(
                            ti.max(normal_distance / sample_denominator, 0.0),
                            1.0,
                        )
                        reconstruction_alpha = alpha
                        raw_target_velocity = (
                            boundary_velocity
                            + (interior_velocity - boundary_velocity) * alpha
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_raw_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(raw_target_velocity.x),
                                    ti.abs(raw_target_velocity.y),
                                ),
                                ti.abs(raw_target_velocity.z),
                            ),
                        )
                        if interpolate_interior_velocity != 0:
                            target_velocity = raw_target_velocity
                        else:
                            target_velocity = boundary_velocity
                            ti.atomic_add(
                                self.report_velocity_dirichlet_boundary_velocity_only_rows[
                                    None
                                ],
                                1,
                            )
                        velocity_dirichlet_active[node] = 1
                        velocity_dirichlet_value_mps[node] = target_velocity
                        velocity_dirichlet_projection_weight[node] = (
                            reconstruction_alpha
                        )
                        # S2-A7 row write-out success path: refine this
                        # node's interior-fluid anchor with the containing
                        # cell of the accepted walk sample (the point the
                        # row's reconstruction actually consumed). Only a
                        # non-obstacle containing cell overwrites the
                        # prefill, preserving the field invariant that a
                        # set anchor is always a fluid cell.
                        sample_point = (
                            boundary_point + normal * sample_denominator
                        )
                        sample_coordinate = self._grid_coordinate_from_fields(
                            sample_point,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            nx,
                            ny,
                            nz,
                        )
                        sample_anchor_i = ti.min(
                            ti.max(
                                ti.floor(sample_coordinate.x + 0.5, ti.i32),
                                0,
                            ),
                            nx - 1,
                        )
                        sample_anchor_j = ti.min(
                            ti.max(
                                ti.floor(sample_coordinate.y + 0.5, ti.i32),
                                0,
                            ),
                            ny - 1,
                        )
                        sample_anchor_k = ti.min(
                            ti.max(
                                ti.floor(sample_coordinate.z + 0.5, ti.i32),
                                0,
                            ),
                            nz - 1,
                        )
                        if (
                            obstacle_field[
                                sample_anchor_i,
                                sample_anchor_j,
                                sample_anchor_k,
                            ]
                            == 0
                        ):
                            node_anchor_cell[node] = ti.Vector(
                                [
                                    sample_anchor_i,
                                    sample_anchor_j,
                                    sample_anchor_k,
                                ]
                            )
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(target_velocity.x),
                                    ti.abs(target_velocity.y),
                                ),
                                ti.abs(target_velocity.z),
                            ),
                        )
                    elif fallback_reconstructable != 0:
                        reconstruction_alpha = fallback_alpha
                        raw_target_velocity = (
                            boundary_velocity
                            + (fallback_velocity - boundary_velocity)
                            * reconstruction_alpha
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_raw_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(raw_target_velocity.x),
                                    ti.abs(raw_target_velocity.y),
                                ),
                                ti.abs(raw_target_velocity.z),
                            ),
                        )
                        if interpolate_interior_velocity != 0:
                            target_velocity = raw_target_velocity
                        else:
                            target_velocity = boundary_velocity
                            ti.atomic_add(
                                self.report_velocity_dirichlet_boundary_velocity_only_rows[
                                    None
                                ],
                                1,
                            )
                        velocity_dirichlet_active[node] = 1
                        velocity_dirichlet_value_mps[node] = target_velocity
                        velocity_dirichlet_projection_weight[node] = (
                            reconstruction_alpha
                        )
                        node_anchor_cell[node] = ti.Vector(
                            [
                                fallback_anchor_i,
                                fallback_anchor_j,
                                fallback_anchor_k,
                            ]
                        )
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(target_velocity.x),
                                    ti.abs(target_velocity.y),
                                ),
                                ti.abs(target_velocity.z),
                            ),
                        )
                    elif (
                        fluid_weight <= 1.0e-12
                        and normal_denominator > 1.0e-12
                        and normal_distance >= 0.0
                    ):
                        ti.atomic_add(
                            self.report_velocity_dirichlet_narrow_gap_rows[None],
                            1,
                        )
                        velocity_dirichlet_active[node] = 1
                        velocity_dirichlet_value_mps[node] = boundary_velocity
                        velocity_dirichlet_projection_weight[node] = 0.0
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            0.0,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(boundary_velocity.x),
                                    ti.abs(boundary_velocity.y),
                                ),
                                ti.abs(boundary_velocity.z),
                            ),
                        )
                    else:
                        ti.atomic_add(
                            self.report_velocity_dirichlet_invalid_reconstruction_rows[
                                None
                            ],
                            1,
                        )
                        if fluid_weight <= 1.0e-12:
                            ti.atomic_add(
                                self.report_velocity_dirichlet_invalid_no_fluid_sample_rows[
                                    None
                                ],
                                1,
                            )
                        elif normal_denominator <= 1.0e-12:
                            ti.atomic_add(
                                self.report_velocity_dirichlet_invalid_nonpositive_gap_rows[
                                    None
                                ],
                                1,
                            )
                        elif normal_distance < 0.0:
                            ti.atomic_add(
                                self.report_velocity_dirichlet_invalid_node_behind_boundary_rows[
                                    None
                                ],
                                1,
                            )
                        elif normal_distance > normal_denominator:
                            ti.atomic_add(
                                self.report_velocity_dirichlet_invalid_node_beyond_interior_rows[
                                    None
                                ],
                                1,
                            )
                        velocity_dirichlet_active[node] = 1
                        velocity_dirichlet_value_mps[node] = boundary_velocity
                        velocity_dirichlet_projection_weight[node] = 0.0
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            0.0,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(boundary_velocity.x),
                                    ti.abs(boundary_velocity.y),
                                ),
                                ti.abs(boundary_velocity.z),
                            ),
                        )
                else:
                    ti.atomic_add(
                        self.report_velocity_dirichlet_obstacle_rows[None],
                        1,
                    )

    @ti.kernel
    def _relocate_masked_velocity_dirichlet_rows_kernel(
        self,
        velocity_dirichlet_active: ti.template(),
        velocity_dirichlet_value_mps: ti.template(),
        velocity_dirichlet_projection_weight: ti.template(),
        obstacle_field: ti.template(),
        velocity_field: ti.template(),
        node_boundary_point_m: ti.template(),
        node_anchor_cell: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_velocity_dirichlet_relocated_rows[None] = 0
        self.report_velocity_dirichlet_relocation_merged_rows[None] = 0
        self.report_velocity_dirichlet_obstacle_rows[None] = 0
        for node in ti.grouped(self.active_ib_node):
            if self.active_ib_node[node] == 1 and obstacle_field[node] != 0:
                normal = self.pressure_neumann_normal_field[node]
                boundary_point = node_boundary_point_m[node]
                boundary_velocity = self.velocity_dirichlet_mps_field[node]
                node_position = ti.Vector(
                    [
                        cell_center_x_m[node[0]],
                        cell_center_y_m[node[1]],
                        cell_center_z_m[node[2]],
                    ]
                )
                side = (node_position - boundary_point).dot(normal)
                primary_walk_normal = normal
                if side < 0.0:
                    primary_walk_normal = -normal
                node_distance = ti.abs(side)
                target_i = -1
                target_j = -1
                target_k = -1
                target_distance = 0.0
                target_walk_normal = primary_walk_normal
                target_walk_step_m = 0.0
                node_width_x = cell_face_x_m[node[0] + 1] - cell_face_x_m[node[0]]
                node_width_y = cell_face_y_m[node[1] + 1] - cell_face_y_m[node[1]]
                node_width_z = cell_face_z_m[node[2] + 1] - cell_face_z_m[node[2]]
                for side_index in ti.static(range(2)):
                    walk_normal = primary_walk_normal
                    start_distance = node_distance
                    if side_index == 1:
                        walk_normal = -primary_walk_normal
                        start_distance = 0.0
                    walk_step_m = 0.5 / ti.max(
                        ti.abs(walk_normal.x) / ti.max(node_width_x, 1.0e-12)
                        + ti.abs(walk_normal.y) / ti.max(node_width_y, 1.0e-12)
                        + ti.abs(walk_normal.z) / ti.max(node_width_z, 1.0e-12),
                        1.0e-12,
                    )
                    step_index = 0
                    while (
                        step_index < HIBM_OWNER_RELOCATION_WALK_STEPS
                        and target_i < 0
                    ):
                        candidate_distance = start_distance + walk_step_m * ti.cast(
                            step_index + 1, ti.f32
                        )
                        candidate_point = boundary_point + walk_normal * candidate_distance
                        candidate_coordinate = self._grid_coordinate_from_fields(
                            candidate_point,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            nx,
                            ny,
                            nz,
                        )
                        candidate_i = ti.min(
                            ti.max(
                                ti.floor(candidate_coordinate.x + 0.5, ti.i32),
                                0,
                            ),
                            nx - 1,
                        )
                        candidate_j = ti.min(
                            ti.max(
                                ti.floor(candidate_coordinate.y + 0.5, ti.i32),
                                0,
                            ),
                            ny - 1,
                        )
                        candidate_k = ti.min(
                            ti.max(
                                ti.floor(candidate_coordinate.z + 0.5, ti.i32),
                                0,
                            ),
                            nz - 1,
                        )
                        if obstacle_field[candidate_i, candidate_j, candidate_k] == 0:
                            candidate_center = ti.Vector(
                                [
                                    cell_center_x_m[candidate_i],
                                    cell_center_y_m[candidate_j],
                                    cell_center_z_m[candidate_k],
                                ]
                            )
                            candidate_center_distance = (
                                candidate_center - boundary_point
                            ).dot(walk_normal)
                            if candidate_center_distance > 1.0e-12:
                                target_i = candidate_i
                                target_j = candidate_j
                                target_k = candidate_k
                                target_distance = candidate_center_distance
                                target_walk_normal = walk_normal
                                target_walk_step_m = walk_step_m
                        step_index += 1
                if target_i >= 0:
                    previous = ti.atomic_or(
                        velocity_dirichlet_active[target_i, target_j, target_k],
                        1,
                    )
                    if previous == 0:
                        sample_velocity, sample_weight, sample_distance = (
                            self._walk_interior_velocity_sample(
                                velocity_field,
                                obstacle_field,
                                boundary_point,
                                target_walk_normal,
                                target_distance + 2.0 * target_walk_step_m,
                                target_walk_step_m,
                                cell_face_x_m,
                                cell_face_y_m,
                                cell_face_z_m,
                                cell_center_x_m,
                                cell_center_y_m,
                                cell_center_z_m,
                                nx,
                                ny,
                                nz,
                            )
                        )
                        target_velocity = boundary_velocity
                        reconstruction_alpha = 0.0
                        if (
                            sample_weight > 1.0e-12
                            and sample_distance > target_distance
                        ):
                            reconstruction_alpha = ti.min(
                                ti.max(target_distance / sample_distance, 0.0),
                                1.0,
                            )
                            target_velocity = (
                                boundary_velocity
                                + (sample_velocity - boundary_velocity)
                                * reconstruction_alpha
                            )
                        velocity_dirichlet_value_mps[
                            target_i, target_j, target_k
                        ] = target_velocity
                        velocity_dirichlet_projection_weight[
                            target_i, target_j, target_k
                        ] = reconstruction_alpha
                        # S2-A7 relocation success path: the claimed cell
                        # was obstacle-checked at claim time, so it is a
                        # fluid cell by construction; publish it as the
                        # masked owner node's interior-fluid anchor.
                        node_anchor_cell[node] = ti.Vector(
                            [target_i, target_j, target_k]
                        )
                        ti.atomic_add(
                            self.report_velocity_dirichlet_relocated_rows[None],
                            1,
                        )
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(target_velocity.x),
                                    ti.abs(target_velocity.y),
                                ),
                                ti.abs(target_velocity.z),
                            ),
                        )
                    else:
                        ti.atomic_add(
                            self.report_velocity_dirichlet_relocation_merged_rows[
                                None
                            ],
                            1,
                        )
                else:
                    ti.atomic_add(
                        self.report_velocity_dirichlet_obstacle_rows[None],
                        1,
                    )

    @ti.kernel
    def _clear_owned_velocity_dirichlet_rows_kernel(
        self,
        velocity_dirichlet_active: ti.template(),
        velocity_dirichlet_value_mps: ti.template(),
        velocity_dirichlet_projection_weight: ti.template(),
        velocity_dirichlet_marker_region_id: ti.template(),
    ):
        for node in ti.grouped(velocity_dirichlet_active):
            if velocity_dirichlet_marker_region_id[node] >= 0:
                velocity_dirichlet_active[node] = 0
                velocity_dirichlet_value_mps[node] = ti.Vector([0.0, 0.0, 0.0])
                velocity_dirichlet_projection_weight[node] = 0.0
                velocity_dirichlet_marker_region_id[node] = -1

    @ti.kernel
    def _reset_and_prefill_node_anchor_cells_kernel(
        self,
        node_anchor_cell: ti.template(),
        node_interior_fluid_point_m: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        # S2-A7 prefill layer: before each velocity-Dirichlet assembly the
        # whole anchor field is reset to the unset sentinel, then every
        # active IB node whose interior-fluid probe point lands in a
        # non-obstacle containing cell is pre-anchored to that cell. This
        # extends anchor coverage to ALL active IB nodes with a wet
        # interior probe - including narrow-gap and invalid-reconstruction
        # rows - independent of whether the row write-out below succeeds;
        # the assembly success paths then overwrite with the refined
        # sample/claim cell.
        for node in ti.grouped(self.active_ib_node):
            node_anchor_cell[node] = ti.Vector([-1, -1, -1])
            if self.active_ib_node[node] == 1:
                interior_point = node_interior_fluid_point_m[node]
                interior_coordinate = self._grid_coordinate_from_fields(
                    interior_point,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                prefill_i = ti.min(
                    ti.max(
                        ti.floor(interior_coordinate.x + 0.5, ti.i32),
                        0,
                    ),
                    nx - 1,
                )
                prefill_j = ti.min(
                    ti.max(
                        ti.floor(interior_coordinate.y + 0.5, ti.i32),
                        0,
                    ),
                    ny - 1,
                )
                prefill_k = ti.min(
                    ti.max(
                        ti.floor(interior_coordinate.z + 0.5, ti.i32),
                        0,
                    ),
                    nz - 1,
                )
                if obstacle_field[prefill_i, prefill_j, prefill_k] == 0:
                    node_anchor_cell[node] = ti.Vector(
                        [prefill_i, prefill_j, prefill_k]
                    )

    @ti.kernel
    def _stamp_velocity_dirichlet_marker_regions_kernel(
        self,
        velocity_dirichlet_active: ti.template(),
        velocity_dirichlet_marker_region_id: ti.template(),
        nearest_marker: ti.template(),
        node_anchor_cell: ti.template(),
        marker_region_id: ti.template(),
        marker_count: ti.i32,
    ):
        for node in ti.grouped(self.active_ib_node):
            if self.active_ib_node[node] != 1:
                continue
            marker = nearest_marker[node]
            if marker < 0 or marker >= marker_count:
                continue
            region = marker_region_id[marker]
            if velocity_dirichlet_active[node] != 0:
                velocity_dirichlet_marker_region_id[node] = region
            anchor = node_anchor_cell[node]
            anchor_i = anchor.x
            anchor_j = anchor.y
            anchor_k = anchor.z
            if (
                anchor_i >= 0
                and anchor_i < velocity_dirichlet_active.shape[0]
                and anchor_j >= 0
                and anchor_j < velocity_dirichlet_active.shape[1]
                and anchor_k >= 0
                and anchor_k < velocity_dirichlet_active.shape[2]
                and velocity_dirichlet_active[anchor_i, anchor_j, anchor_k] != 0
            ):
                velocity_dirichlet_marker_region_id[
                    anchor_i,
                    anchor_j,
                    anchor_k,
                ] = region

    def assemble_velocity_dirichlet_reconstructed_boundary_rows(
        self,
        velocity_dirichlet_active,
        velocity_dirichlet_value_mps,
        velocity_dirichlet_projection_weight,
        obstacle_field,
        velocity_field,
        search: HibmMpmIbNodeSearch,
        *,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
        velocity_dirichlet_marker_region_id=None,
        marker_region_id=None,
        primary_region_id: int | None = None,
        secondary_region_id: int | None = None,
        interpolate_interior_velocity: bool = True,
    ) -> HibmMpmVelocityDirichletBoundaryReport:
        if tuple(search.grid_nodes) != self.grid_nodes:
            raise ValueError("search.grid_nodes must match boundary grid_nodes")
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        preserve_existing_rows = velocity_dirichlet_marker_region_id is not None
        if preserve_existing_rows:
            self._clear_owned_velocity_dirichlet_rows_kernel(
                velocity_dirichlet_active,
                velocity_dirichlet_value_mps,
                velocity_dirichlet_projection_weight,
                velocity_dirichlet_marker_region_id,
            )
        _debug_stage_progress("velocity_rows:prefill_anchor:start")
        # S2-A7: full-field sentinel reset + interior-point prefill,
        # strictly before the row assembly so the success paths overwrite
        # the prefill with the row's own fluid sample/claim cell.
        self._reset_and_prefill_node_anchor_cells_kernel(
            search.node_anchor_cell,
            search.node_interior_fluid_point_m,
            obstacle_field,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        _debug_stage_progress("velocity_rows:prefill_anchor:done")
        _debug_stage_progress("velocity_rows:assemble_reconstructed:start")
        self._assemble_velocity_dirichlet_reconstructed_boundary_rows_kernel(
            velocity_dirichlet_active,
            velocity_dirichlet_value_mps,
            velocity_dirichlet_projection_weight,
            obstacle_field,
            velocity_field,
            search.node_boundary_point_m,
            search.node_interior_fluid_point_m,
            search.node_anchor_cell,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
            1 if preserve_existing_rows else 0,
            1 if bool(interpolate_interior_velocity) else 0,
        )
        _debug_stage_progress("velocity_rows:assemble_reconstructed:done")
        _debug_stage_progress("velocity_rows:relocate_masked:start")
        self._relocate_masked_velocity_dirichlet_rows_kernel(
            velocity_dirichlet_active,
            velocity_dirichlet_value_mps,
            velocity_dirichlet_projection_weight,
            obstacle_field,
            velocity_field,
            search.node_boundary_point_m,
            search.node_anchor_cell,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        _debug_stage_progress("velocity_rows:relocate_masked:done")
        if velocity_dirichlet_marker_region_id is not None or marker_region_id is not None:
            if velocity_dirichlet_marker_region_id is None or marker_region_id is None:
                raise ValueError(
                    "velocity_dirichlet_marker_region_id and marker_region_id "
                    "must be provided together"
                )
            if tuple(velocity_dirichlet_marker_region_id.shape) != tuple(
                velocity_dirichlet_active.shape
            ):
                raise ValueError(
                    "velocity_dirichlet_marker_region_id shape must match "
                    "velocity_dirichlet_active"
                )
            _debug_stage_progress("velocity_rows:stamp_regions:start")
            self._stamp_velocity_dirichlet_marker_regions_kernel(
                velocity_dirichlet_active,
                velocity_dirichlet_marker_region_id,
                search.nearest_marker,
                search.node_anchor_cell,
                marker_region_id,
                int(marker_region_id.shape[0]),
            )
            _debug_stage_progress("velocity_rows:stamp_regions:done")
        active_rows = int(self.report_velocity_dirichlet_boundary_rows[None])
        min_projection_weight = 0.0
        if active_rows > 0:
            min_projection_weight = float(
                self.report_velocity_dirichlet_min_projection_weight[None]
            )
        (
            primary_rows,
            secondary_rows,
            other_rows,
            unassigned_rows,
        ) = self._velocity_dirichlet_region_row_counts(
            velocity_dirichlet_active,
            velocity_dirichlet_marker_region_id,
            primary_region_id=primary_region_id,
            secondary_region_id=secondary_region_id,
        )
        return HibmMpmVelocityDirichletBoundaryReport(
            active_velocity_dirichlet_rows=active_rows,
            inactive_obstacle_rows=int(
                self.report_velocity_dirichlet_obstacle_rows[None]
            ),
            max_abs_velocity_mps=float(
                self.report_velocity_dirichlet_max_abs_velocity[None]
            ),
            raw_reconstructed_max_abs_velocity_mps=float(
                self.report_velocity_dirichlet_raw_max_abs_velocity[None]
            ),
            boundary_velocity_only_row_count=int(
                self.report_velocity_dirichlet_boundary_velocity_only_rows[None]
            ),
            primary_region_active_rows=primary_rows,
            secondary_region_active_rows=secondary_rows,
            other_region_active_rows=other_rows,
            unassigned_region_active_rows=unassigned_rows,
            invalid_reconstruction_row_count=int(
                self.report_velocity_dirichlet_invalid_reconstruction_rows[None]
            ),
            invalid_no_fluid_sample_row_count=int(
                self.report_velocity_dirichlet_invalid_no_fluid_sample_rows[None]
            ),
            invalid_nonpositive_gap_row_count=int(
                self.report_velocity_dirichlet_invalid_nonpositive_gap_rows[None]
            ),
            invalid_node_behind_boundary_row_count=int(
                self.report_velocity_dirichlet_invalid_node_behind_boundary_rows[None]
            ),
            invalid_node_beyond_interior_row_count=int(
                self.report_velocity_dirichlet_invalid_node_beyond_interior_rows[None]
            ),
            narrow_gap_boundary_velocity_row_count=int(
                self.report_velocity_dirichlet_narrow_gap_rows[None]
            ),
            relocated_row_count=int(
                self.report_velocity_dirichlet_relocated_rows[None]
            ),
            relocation_merged_row_count=int(
                self.report_velocity_dirichlet_relocation_merged_rows[None]
            ),
            relocation_blocked_row_count=int(
                self.report_velocity_dirichlet_obstacle_rows[None]
            ),
            min_projection_weight=min_projection_weight,
            max_projection_weight=float(
                self.report_velocity_dirichlet_max_projection_weight[None]
            ),
        )

    @ti.kernel
    def _clear_pressure_neumann_rows_by_marker_kernel(
        self,
        marker_count: ti.i32,
    ):
        for marker in range(marker_count):
            self.marker_pressure_neumann_row_count[marker] = 0
            self.marker_pressure_neumann_candidate_node_count[marker] = 0

    @ti.kernel
    def _count_pressure_neumann_candidate_nodes_by_marker_kernel(
        self,
        nearest_marker: ti.template(),
        marker_count: ti.i32,
    ):
        for node in ti.grouped(self.active_ib_node):
            if self.active_ib_node[node] == 1:
                marker = nearest_marker[node]
                if 0 <= marker < marker_count:
                    ti.atomic_add(
                        self.marker_pressure_neumann_candidate_node_count[marker],
                        1,
                    )

    @ti.kernel
    def _summarize_pressure_neumann_rows_by_marker_kernel(
        self,
        marker_count: ti.i32,
    ):
        self.report_pressure_neumann_active_marker_count[None] = 0
        self.report_pressure_neumann_max_rows_per_marker[None] = 0
        for marker in range(marker_count):
            row_count = self.marker_pressure_neumann_row_count[marker]
            if row_count > 0:
                ti.atomic_add(
                    self.report_pressure_neumann_active_marker_count[None],
                    1,
                )
                ti.atomic_max(
                    self.report_pressure_neumann_max_rows_per_marker[None],
                    row_count,
                )

    @ti.func
    def _pressure_neumann_cell_touches_velocity_dirichlet_projection_face(
        self,
        velocity_dirichlet_active: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        touches = velocity_dirichlet_active[i, j, k] != 0
        if i < nx - 1 and velocity_dirichlet_active[i + 1, j, k] != 0:
            touches = True
        if j < ny - 1 and velocity_dirichlet_active[i, j + 1, k] != 0:
            touches = True
        if k < nz - 1 and velocity_dirichlet_active[i, j, k + 1] != 0:
            touches = True
        return touches

    @ti.func
    def _record_pressure_neumann_invalid_diagnostic_row(
        self,
        node: ti.template(),
        owner_i: ti.i32,
        owner_j: ti.i32,
        owner_k: ti.i32,
        neighbor_i: ti.i32,
        neighbor_j: ti.i32,
        neighbor_k: ti.i32,
        anchor_i: ti.i32,
        anchor_j: ti.i32,
        anchor_k: ti.i32,
        marker: ti.i32,
        reason_code: ti.i32,
        node_distance_m: ti.f32,
        normal_denominator_m: ti.f32,
        reconstruction_gap_m: ti.f32,
    ):
        row_index = ti.atomic_add(self.pressure_neumann_invalid_diag_count[None], 1)
        if row_index < PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY:
            self.pressure_neumann_invalid_diag_node[row_index] = node
            self.pressure_neumann_invalid_diag_owner[row_index] = ti.Vector(
                [owner_i, owner_j, owner_k]
            )
            self.pressure_neumann_invalid_diag_neighbor[row_index] = ti.Vector(
                [neighbor_i, neighbor_j, neighbor_k]
            )
            self.pressure_neumann_invalid_diag_anchor[row_index] = ti.Vector(
                [anchor_i, anchor_j, anchor_k]
            )
            self.pressure_neumann_invalid_diag_marker[row_index] = marker
            self.pressure_neumann_invalid_diag_reason[row_index] = reason_code
            self.pressure_neumann_invalid_diag_node_distance_m[row_index] = (
                node_distance_m
            )
            self.pressure_neumann_invalid_diag_normal_denominator_m[row_index] = (
                normal_denominator_m
            )
            self.pressure_neumann_invalid_diag_reconstruction_gap_m[row_index] = (
                reconstruction_gap_m
            )

    @ti.kernel
    def _assemble_pressure_neumann_matrix_rows_kernel(
        self,
        pressure_matrix_diagonal: ti.template(),
        pressure_matrix_rhs: ti.template(),
        pressure_coupling_active: ti.template(),
        pressure_coupling_neighbor: ti.template(),
        pressure_coupling_coefficient: ti.template(),
        pressure_coupling_extra_neighbor: ti.template(),
        pressure_coupling_extra_coefficient: ti.template(),
        pressure_interface_row_count: ti.template(),
        pressure_interface_row_owner: ti.template(),
        pressure_interface_row_neighbor: ti.template(),
        pressure_interface_row_transmissibility: ti.template(),
        obstacle_field: ti.template(),
        velocity_dirichlet_active: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        node_boundary_point_m: ti.template(),
        node_interior_fluid_point_m: ti.template(),
        node_anchor_cell: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nearest_marker: ti.template(),
        marker_surface_area_m2: ti.template(),
        marker_pressure_neumann_candidate_node_count: ti.template(),
        marker_pressure_anchor_cell: ti.template(),
        marker_count: ti.i32,
        max_pressure_coupling_slots: ti.i32,
        pressure_interface_row_list_enabled: ti.i32,
        pressure_interface_row_capacity: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_pressure_neumann_matrix_rows[None] = 0
        self.report_pressure_neumann_rhs_integral[None] = ti.cast(0.0, ti.f64)
        self.report_pressure_neumann_max_abs_rhs[None] = 0.0
        self.report_pressure_neumann_invalid_reconstruction_rows[None] = 0
        self.report_pressure_neumann_invalid_unreconstructable_rows[None] = 0
        self.report_pressure_neumann_invalid_bad_marker_rows[None] = 0
        self.report_pressure_neumann_invalid_nonpositive_volume_rows[None] = 0
        self.report_pressure_neumann_min_reconstruction_gap_m[None] = 1.0e30
        self.report_pressure_neumann_max_reconstruction_gap_m[None] = 0.0
        self.report_pressure_neumann_max_transmissibility_m[None] = 0.0
        self.report_pressure_neumann_max_raw_transmissibility_m[None] = 0.0
        self.report_pressure_neumann_max_transmissibility_limit_m[None] = 0.0
        self.report_pressure_neumann_transmissibility_capped_rows[None] = 0
        self.report_pressure_neumann_max_diagonal_per_m2[None] = 0.0
        self.report_pressure_neumann_skipped_velocity_dirichlet_rows[None] = 0
        self.report_pressure_neumann_skipped_pressure_boundary_adjacent_rows[None] = 0
        self.report_pressure_neumann_skipped_obstacle_owner_rows[None] = 0
        self.report_pressure_neumann_relocated_obstacle_owner_rows[None] = 0
        self.report_pressure_neumann_duplicate_owner_rows[None] = 0
        self.report_pressure_neumann_overflow_owner_rows[None] = 0
        self.report_pressure_neumann_max_owner_slot_count[None] = 0
        self.pressure_neumann_invalid_diag_count[None] = 0
        for node in ti.grouped(self.active_ib_node):
            owner_i = node[0]
            owner_j = node[1]
            owner_k = node[2]
            row_owner_is_fluid = 0
            relocated_obstacle_owner = 0
            normal = self.pressure_neumann_normal_field[node]
            boundary_point = node_boundary_point_m[node]
            if self.active_ib_node[node] == 1:
                if obstacle_field[owner_i, owner_j, owner_k] == 0:
                    row_owner_is_fluid = 1
                else:
                    anchor_cell = node_anchor_cell[node]
                    anchor_i = anchor_cell.x
                    anchor_j = anchor_cell.y
                    anchor_k = anchor_cell.z
                    if (
                        anchor_i >= 0
                        and anchor_i < nx
                        and anchor_j >= 0
                        and anchor_j < ny
                        and anchor_k >= 0
                        and anchor_k < nz
                        and obstacle_field[anchor_i, anchor_j, anchor_k] == 0
                    ):
                        owner_i = anchor_i
                        owner_j = anchor_j
                        owner_k = anchor_k
                        row_owner_is_fluid = 1
                        relocated_obstacle_owner = 1
                    if row_owner_is_fluid == 0:
                        original_position = ti.Vector(
                            [
                                cell_center_x_m[node[0]],
                                cell_center_y_m[node[1]],
                                cell_center_z_m[node[2]],
                            ]
                        )
                        side = (original_position - boundary_point).dot(normal)
                        primary_walk_normal = normal
                        if side < 0.0:
                            primary_walk_normal = -normal
                        node_width_x = (
                            cell_face_x_m[node[0] + 1] - cell_face_x_m[node[0]]
                        )
                        node_width_y = (
                            cell_face_y_m[node[1] + 1] - cell_face_y_m[node[1]]
                        )
                        node_width_z = (
                            cell_face_z_m[node[2] + 1] - cell_face_z_m[node[2]]
                        )
                        node_distance = ti.abs(side)
                        target_i = -1
                        target_j = -1
                        target_k = -1
                        for side_index in ti.static(range(2)):
                            walk_normal = primary_walk_normal
                            start_distance = node_distance
                            if side_index == 1:
                                walk_normal = -primary_walk_normal
                                start_distance = 0.0
                            walk_step_m = 0.5 / ti.max(
                                ti.abs(walk_normal.x)
                                / ti.max(node_width_x, 1.0e-12)
                                + ti.abs(walk_normal.y)
                                / ti.max(node_width_y, 1.0e-12)
                                + ti.abs(walk_normal.z)
                                / ti.max(node_width_z, 1.0e-12),
                                1.0e-12,
                            )
                            step_index = 0
                            while (
                                step_index < HIBM_OWNER_RELOCATION_WALK_STEPS
                                and target_i < 0
                            ):
                                candidate_distance = (
                                    start_distance
                                    + walk_step_m
                                    * ti.cast(step_index + 1, ti.f32)
                                )
                                candidate_point = (
                                    boundary_point
                                    + walk_normal * candidate_distance
                                )
                                candidate_coordinate = (
                                    self._grid_coordinate_from_fields(
                                        candidate_point,
                                        cell_face_x_m,
                                        cell_face_y_m,
                                        cell_face_z_m,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                        nx,
                                        ny,
                                        nz,
                                    )
                                )
                                candidate_i = ti.min(
                                    ti.max(
                                        ti.floor(
                                            candidate_coordinate.x + 0.5,
                                            ti.i32,
                                        ),
                                        0,
                                    ),
                                    nx - 1,
                                )
                                candidate_j = ti.min(
                                    ti.max(
                                        ti.floor(
                                            candidate_coordinate.y + 0.5,
                                            ti.i32,
                                        ),
                                        0,
                                    ),
                                    ny - 1,
                                )
                                candidate_k = ti.min(
                                    ti.max(
                                        ti.floor(
                                            candidate_coordinate.z + 0.5,
                                            ti.i32,
                                        ),
                                        0,
                                    ),
                                    nz - 1,
                                )
                                if (
                                    obstacle_field[
                                        candidate_i,
                                        candidate_j,
                                        candidate_k,
                                    ]
                                    == 0
                                ):
                                    candidate_center = ti.Vector(
                                        [
                                            cell_center_x_m[candidate_i],
                                            cell_center_y_m[candidate_j],
                                            cell_center_z_m[candidate_k],
                                        ]
                                    )
                                    candidate_center_distance = (
                                        candidate_center - boundary_point
                                    ).dot(walk_normal)
                                    if candidate_center_distance > 1.0e-12:
                                        target_i = candidate_i
                                        target_j = candidate_j
                                        target_k = candidate_k
                                step_index += 1
                        if target_i >= 0:
                            owner_i = target_i
                            owner_j = target_j
                            owner_k = target_k
                            row_owner_is_fluid = 1
                            relocated_obstacle_owner = 1
                if row_owner_is_fluid == 0:
                    ti.atomic_add(
                        self.report_pressure_neumann_skipped_obstacle_owner_rows[None],
                        1,
                    )
                if (
                    row_owner_is_fluid != 0
                    and velocity_dirichlet_active[owner_i, owner_j, owner_k] != 0
                ):
                    target_i = -1
                    target_j = -1
                    target_k = -1
                    anchor_cell = node_anchor_cell[node]
                    anchor_i = anchor_cell.x
                    anchor_j = anchor_cell.y
                    anchor_k = anchor_cell.z
                    if (
                        anchor_i >= 0
                        and anchor_i < nx
                        and anchor_j >= 0
                        and anchor_j < ny
                        and anchor_k >= 0
                        and anchor_k < nz
                        and obstacle_field[anchor_i, anchor_j, anchor_k] == 0
                        and velocity_dirichlet_active[anchor_i, anchor_j, anchor_k]
                        == 0
                    ):
                        target_i = anchor_i
                        target_j = anchor_j
                        target_k = anchor_k
                    if target_i < 0:
                        for offset in ti.static(
                            (
                                (-1, 0, 0),
                                (1, 0, 0),
                                (0, -1, 0),
                                (0, 1, 0),
                                (0, 0, -1),
                                (0, 0, 1),
                            )
                        ):
                            if target_i < 0:
                                candidate_i = owner_i + offset[0]
                                candidate_j = owner_j + offset[1]
                                candidate_k = owner_k + offset[2]
                                if (
                                    0 <= candidate_i
                                    and candidate_i < nx
                                    and 0 <= candidate_j
                                    and candidate_j < ny
                                    and 0 <= candidate_k
                                    and candidate_k < nz
                                ):
                                    if (
                                        obstacle_field[
                                            candidate_i,
                                            candidate_j,
                                            candidate_k,
                                        ]
                                        == 0
                                        and velocity_dirichlet_active[
                                            candidate_i,
                                            candidate_j,
                                            candidate_k,
                                        ]
                                        == 0
                                    ):
                                        target_i = candidate_i
                                        target_j = candidate_j
                                        target_k = candidate_k
                    if target_i >= 0:
                        owner_i = target_i
                        owner_j = target_j
                        owner_k = target_k
                    else:
                        row_owner_is_fluid = 0
                        ti.atomic_add(
                            self.report_pressure_neumann_skipped_velocity_dirichlet_rows[
                                None
                            ],
                            1,
                        )
            if (
                self.active_ib_node[node] == 1
                and row_owner_is_fluid != 0
            ):
                marker = nearest_marker[node]
                diagnostic_anchor_cell = node_anchor_cell[node]
                diagnostic_anchor_i = diagnostic_anchor_cell.x
                diagnostic_anchor_j = diagnostic_anchor_cell.y
                diagnostic_anchor_k = diagnostic_anchor_cell.z
                if 0 <= marker < marker_count:
                    volume_m3 = (
                        cell_width_x_m[owner_i]
                        * cell_width_y_m[owner_j]
                        * cell_width_z_m[owner_k]
                    )
                    if volume_m3 > 0.0:
                        interior_point = node_interior_fluid_point_m[node]
                        node_position = ti.Vector(
                            [
                                cell_center_x_m[owner_i],
                                cell_center_y_m[owner_j],
                                cell_center_z_m[owner_k],
                            ]
                        )
                        grid_coordinate = self._grid_coordinate_from_fields(
                            interior_point,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            nx,
                            ny,
                            nz,
                        )
                        neighbor_i = ti.min(
                            ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0),
                            nx - 1,
                        )
                        neighbor_j = ti.min(
                            ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0),
                            ny - 1,
                        )
                        neighbor_k = ti.min(
                            ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0),
                            nz - 1,
                        )
                        neighbor_position = ti.Vector(
                            [
                                cell_center_x_m[neighbor_i],
                                cell_center_y_m[neighbor_j],
                                cell_center_z_m[neighbor_k],
                            ]
                        )
                        node_distance = (node_position - boundary_point).dot(normal)
                        neighbor_distance = (
                            neighbor_position - boundary_point
                        ).dot(normal)
                        normal_denominator = (interior_point - boundary_point).dot(
                            normal
                        )
                        reconstruction_gap = ti.abs(neighbor_distance - node_distance)
                        node_spacing_inv = (
                            ti.abs(normal.x) / cell_width_x_m[owner_i]
                            + ti.abs(normal.y) / cell_width_y_m[owner_j]
                            + ti.abs(normal.z) / cell_width_z_m[owner_k]
                        )
                        neighbor_spacing_inv = (
                            ti.abs(normal.x) / cell_width_x_m[neighbor_i]
                            + ti.abs(normal.y) / cell_width_y_m[neighbor_j]
                            + ti.abs(normal.z) / cell_width_z_m[neighbor_k]
                        )
                        node_normal_width = 1.0 / ti.max(
                            node_spacing_inv,
                            1.0e-12,
                        )
                        neighbor_normal_width = 1.0 / ti.max(
                            neighbor_spacing_inv,
                            1.0e-12,
                        )
                        min_normal_width = ti.min(
                            node_normal_width,
                            neighbor_normal_width,
                        )
                        reconstruction_gap_floor = ti.max(
                            1.0e-12,
                            1.0e-3 * min_normal_width,
                        )
                        row_reconstructable = 0
                        if (
                            reconstruction_gap > reconstruction_gap_floor
                            and normal_denominator > 1.0e-12
                            and node_distance >= 0.0
                            and node_distance <= normal_denominator
                            and obstacle_field[neighbor_i, neighbor_j, neighbor_k] == 0
                            and velocity_dirichlet_active[
                                neighbor_i,
                                neighbor_j,
                                neighbor_k,
                            ]
                            == 0
                        ):
                            row_reconstructable = 1
                        else:
                            abs_x = ti.abs(normal.x)
                            abs_y = ti.abs(normal.y)
                            abs_z = ti.abs(normal.z)
                            fallback_i = owner_i
                            fallback_j = owner_j
                            fallback_k = owner_k
                            if abs_x >= abs_y and abs_x >= abs_z:
                                step = 1
                                if normal.x < 0.0:
                                    step = -1
                                fallback_i = owner_i + step
                                if fallback_i < 0 or fallback_i >= nx:
                                    fallback_i = owner_i - step
                                fallback_i = ti.min(ti.max(fallback_i, 0), nx - 1)
                                if (
                                    obstacle_field[fallback_i, fallback_j, fallback_k] != 0
                                    or velocity_dirichlet_active[
                                        fallback_i,
                                        fallback_j,
                                        fallback_k,
                                    ]
                                    != 0
                                ):
                                    alternate_i = ti.min(ti.max(owner_i - step, 0), nx - 1)
                                    if (
                                        obstacle_field[alternate_i, fallback_j, fallback_k]
                                        == 0
                                        and velocity_dirichlet_active[
                                            alternate_i,
                                            fallback_j,
                                            fallback_k,
                                        ]
                                        == 0
                                    ):
                                        fallback_i = alternate_i
                            elif abs_y >= abs_x and abs_y >= abs_z:
                                step = 1
                                if normal.y < 0.0:
                                    step = -1
                                fallback_j = owner_j + step
                                if fallback_j < 0 or fallback_j >= ny:
                                    fallback_j = owner_j - step
                                fallback_j = ti.min(ti.max(fallback_j, 0), ny - 1)
                                if (
                                    obstacle_field[fallback_i, fallback_j, fallback_k] != 0
                                    or velocity_dirichlet_active[
                                        fallback_i,
                                        fallback_j,
                                        fallback_k,
                                    ]
                                    != 0
                                ):
                                    alternate_j = ti.min(ti.max(owner_j - step, 0), ny - 1)
                                    if (
                                        obstacle_field[fallback_i, alternate_j, fallback_k]
                                        == 0
                                        and velocity_dirichlet_active[
                                            fallback_i,
                                            alternate_j,
                                            fallback_k,
                                        ]
                                        == 0
                                    ):
                                        fallback_j = alternate_j
                            else:
                                step = 1
                                if normal.z < 0.0:
                                    step = -1
                                fallback_k = owner_k + step
                                if fallback_k < 0 or fallback_k >= nz:
                                    fallback_k = owner_k - step
                                fallback_k = ti.min(ti.max(fallback_k, 0), nz - 1)
                                if (
                                    obstacle_field[fallback_i, fallback_j, fallback_k] != 0
                                    or velocity_dirichlet_active[
                                        fallback_i,
                                        fallback_j,
                                        fallback_k,
                                    ]
                                    != 0
                                ):
                                    alternate_k = ti.min(ti.max(owner_k - step, 0), nz - 1)
                                    if (
                                        obstacle_field[fallback_i, fallback_j, alternate_k]
                                        == 0
                                        and velocity_dirichlet_active[
                                            fallback_i,
                                            fallback_j,
                                            alternate_k,
                                        ]
                                        == 0
                                    ):
                                        fallback_k = alternate_k
                            neighbor_i = fallback_i
                            neighbor_j = fallback_j
                            neighbor_k = fallback_k
                            neighbor_position = ti.Vector(
                                [
                                    cell_center_x_m[neighbor_i],
                                    cell_center_y_m[neighbor_j],
                                    cell_center_z_m[neighbor_k],
                                ]
                            )
                            neighbor_distance = (
                                neighbor_position - boundary_point
                            ).dot(normal)
                            reconstruction_gap = ti.abs(
                                neighbor_distance - node_distance
                            )
                            neighbor_spacing_inv = (
                                ti.abs(normal.x) / cell_width_x_m[neighbor_i]
                                + ti.abs(normal.y) / cell_width_y_m[neighbor_j]
                                + ti.abs(normal.z) / cell_width_z_m[neighbor_k]
                            )
                            neighbor_normal_width = 1.0 / ti.max(
                                neighbor_spacing_inv,
                                1.0e-12,
                            )
                            min_normal_width = ti.min(
                                node_normal_width,
                                neighbor_normal_width,
                            )
                            reconstruction_gap_floor = ti.max(
                                1.0e-12,
                                1.0e-3 * min_normal_width,
                            )
                            if (
                                reconstruction_gap > reconstruction_gap_floor
                                and obstacle_field[neighbor_i, neighbor_j, neighbor_k] == 0
                                and velocity_dirichlet_active[
                                    neighbor_i,
                                    neighbor_j,
                                    neighbor_k,
                                ]
                                == 0
                                and (
                                    neighbor_i != owner_i
                                    or neighbor_j != owner_j
                                    or neighbor_k != owner_k
                                )
                            ):
                                row_reconstructable = 1
                            if row_reconstructable == 0:
                                best_i = -1
                                best_j = -1
                                best_k = -1
                                best_gap = 0.0
                                best_neighbor_distance = 0.0
                                best_neighbor_normal_width = 0.0
                                best_min_normal_width = 0.0
                                best_gap_floor = 0.0
                                for offset in ti.static(
                                    (
                                        (-1, 0, 0),
                                        (1, 0, 0),
                                        (0, -1, 0),
                                        (0, 1, 0),
                                        (0, 0, -1),
                                        (0, 0, 1),
                                    )
                                ):
                                    candidate_i = owner_i + offset[0]
                                    candidate_j = owner_j + offset[1]
                                    candidate_k = owner_k + offset[2]
                                    if (
                                        0 <= candidate_i
                                        and candidate_i < nx
                                        and 0 <= candidate_j
                                        and candidate_j < ny
                                        and 0 <= candidate_k
                                        and candidate_k < nz
                                        and obstacle_field[
                                            candidate_i,
                                            candidate_j,
                                            candidate_k,
                                        ]
                                        == 0
                                        and velocity_dirichlet_active[
                                            candidate_i,
                                            candidate_j,
                                            candidate_k,
                                        ]
                                        == 0
                                    ):
                                        candidate_position = ti.Vector(
                                            [
                                                cell_center_x_m[candidate_i],
                                                cell_center_y_m[candidate_j],
                                                cell_center_z_m[candidate_k],
                                            ]
                                        )
                                        candidate_distance = (
                                            candidate_position - boundary_point
                                        ).dot(normal)
                                        candidate_gap = ti.abs(
                                            candidate_distance - node_distance
                                        )
                                        candidate_spacing_inv = (
                                            ti.abs(normal.x)
                                            / cell_width_x_m[candidate_i]
                                            + ti.abs(normal.y)
                                            / cell_width_y_m[candidate_j]
                                            + ti.abs(normal.z)
                                            / cell_width_z_m[candidate_k]
                                        )
                                        candidate_normal_width = 1.0 / ti.max(
                                            candidate_spacing_inv,
                                            1.0e-12,
                                        )
                                        candidate_min_width = ti.min(
                                            node_normal_width,
                                            candidate_normal_width,
                                        )
                                        candidate_gap_floor = ti.max(
                                            1.0e-12,
                                            1.0e-3 * candidate_min_width,
                                        )
                                        if (
                                            candidate_gap > candidate_gap_floor
                                            and candidate_gap > best_gap
                                        ):
                                            best_i = candidate_i
                                            best_j = candidate_j
                                            best_k = candidate_k
                                            best_gap = candidate_gap
                                            best_neighbor_distance = candidate_distance
                                            best_neighbor_normal_width = (
                                                candidate_normal_width
                                            )
                                            best_min_normal_width = candidate_min_width
                                            best_gap_floor = candidate_gap_floor
                                if best_i >= 0:
                                    neighbor_i = best_i
                                    neighbor_j = best_j
                                    neighbor_k = best_k
                                    neighbor_distance = best_neighbor_distance
                                    reconstruction_gap = best_gap
                                    neighbor_normal_width = best_neighbor_normal_width
                                    min_normal_width = best_min_normal_width
                                    reconstruction_gap_floor = best_gap_floor
                                    row_reconstructable = 1
                            if row_reconstructable == 0:
                                anchor_cell = node_anchor_cell[node]
                                anchor_i = anchor_cell.x
                                anchor_j = anchor_cell.y
                                anchor_k = anchor_cell.z
                                if (
                                    anchor_i >= 0
                                    and anchor_i < nx
                                    and anchor_j >= 0
                                    and anchor_j < ny
                                    and anchor_k >= 0
                                    and anchor_k < nz
                                    and (
                                        anchor_i != owner_i
                                        or anchor_j != owner_j
                                        or anchor_k != owner_k
                                    )
                                    and obstacle_field[anchor_i, anchor_j, anchor_k] == 0
                                    and velocity_dirichlet_active[
                                        anchor_i,
                                        anchor_j,
                                        anchor_k,
                                    ]
                                    == 0
                                ):
                                    anchor_position = ti.Vector(
                                        [
                                            cell_center_x_m[anchor_i],
                                            cell_center_y_m[anchor_j],
                                            cell_center_z_m[anchor_k],
                                        ]
                                    )
                                    anchor_distance = (
                                        anchor_position - boundary_point
                                    ).dot(normal)
                                    anchor_gap = ti.abs(
                                        anchor_distance - node_distance
                                    )
                                    anchor_spacing_inv = (
                                        ti.abs(normal.x) / cell_width_x_m[anchor_i]
                                        + ti.abs(normal.y) / cell_width_y_m[anchor_j]
                                        + ti.abs(normal.z) / cell_width_z_m[anchor_k]
                                    )
                                    anchor_normal_width = 1.0 / ti.max(
                                        anchor_spacing_inv,
                                        1.0e-12,
                                    )
                                    anchor_min_width = ti.min(
                                        node_normal_width,
                                        anchor_normal_width,
                                    )
                                    anchor_gap_floor = ti.max(
                                        1.0e-12,
                                        1.0e-3 * anchor_min_width,
                                    )
                                    if (
                                        anchor_gap > anchor_gap_floor
                                        and anchor_distance >= 0.0
                                    ):
                                        neighbor_i = anchor_i
                                        neighbor_j = anchor_j
                                        neighbor_k = anchor_k
                                        neighbor_distance = anchor_distance
                                        reconstruction_gap = anchor_gap
                                        neighbor_normal_width = anchor_normal_width
                                        min_normal_width = anchor_min_width
                                        reconstruction_gap_floor = anchor_gap_floor
                                        row_reconstructable = 1
                            if row_reconstructable == 0:
                                normal_line_i = -1
                                normal_line_j = -1
                                normal_line_k = -1
                                normal_line_distance = 0.0
                                normal_line_gap = 0.0
                                normal_line_normal_width = 0.0
                                normal_line_min_width = 0.0
                                normal_line_gap_floor = 0.0
                                primary_walk_normal = normal
                                if normal_denominator < 0.0:
                                    primary_walk_normal = -normal
                                for side_index in ti.static(range(2)):
                                    walk_normal = primary_walk_normal
                                    if side_index == 1:
                                        walk_normal = -primary_walk_normal
                                    walk_step_m = 0.5 / ti.max(
                                        ti.abs(walk_normal.x)
                                        / ti.max(cell_width_x_m[owner_i], 1.0e-12)
                                        + ti.abs(walk_normal.y)
                                        / ti.max(cell_width_y_m[owner_j], 1.0e-12)
                                        + ti.abs(walk_normal.z)
                                        / ti.max(cell_width_z_m[owner_k], 1.0e-12),
                                        1.0e-12,
                                    )
                                    step_index = 0
                                    while (
                                        step_index
                                        < HIBM_OWNER_RELOCATION_WALK_STEPS
                                        and normal_line_i < 0
                                    ):
                                        candidate_distance_from_boundary = (
                                            walk_step_m
                                            * ti.cast(step_index + 1, ti.f32)
                                        )
                                        candidate_point = (
                                            boundary_point
                                            + walk_normal
                                            * candidate_distance_from_boundary
                                        )
                                        candidate_coordinate = (
                                            self._grid_coordinate_from_fields(
                                                candidate_point,
                                                cell_face_x_m,
                                                cell_face_y_m,
                                                cell_face_z_m,
                                                cell_center_x_m,
                                                cell_center_y_m,
                                                cell_center_z_m,
                                                nx,
                                                ny,
                                                nz,
                                            )
                                        )
                                        candidate_i = ti.min(
                                            ti.max(
                                                ti.floor(
                                                    candidate_coordinate.x + 0.5,
                                                    ti.i32,
                                                ),
                                                0,
                                            ),
                                            nx - 1,
                                        )
                                        candidate_j = ti.min(
                                            ti.max(
                                                ti.floor(
                                                    candidate_coordinate.y + 0.5,
                                                    ti.i32,
                                                ),
                                                0,
                                            ),
                                            ny - 1,
                                        )
                                        candidate_k = ti.min(
                                            ti.max(
                                                ti.floor(
                                                    candidate_coordinate.z + 0.5,
                                                    ti.i32,
                                                ),
                                                0,
                                            ),
                                            nz - 1,
                                        )
                                        if (
                                            obstacle_field[
                                                candidate_i,
                                                candidate_j,
                                                candidate_k,
                                            ]
                                            == 0
                                            and velocity_dirichlet_active[
                                                candidate_i,
                                                candidate_j,
                                                candidate_k,
                                            ]
                                            == 0
                                            and (
                                                candidate_i != owner_i
                                                or candidate_j != owner_j
                                                or candidate_k != owner_k
                                            )
                                        ):
                                            candidate_center = ti.Vector(
                                                [
                                                    cell_center_x_m[candidate_i],
                                                    cell_center_y_m[candidate_j],
                                                    cell_center_z_m[candidate_k],
                                                ]
                                            )
                                            candidate_normal_distance = (
                                                candidate_center - boundary_point
                                            ).dot(normal)
                                            candidate_gap = ti.abs(
                                                candidate_normal_distance
                                                - node_distance
                                            )
                                            candidate_spacing_inv = (
                                                ti.abs(normal.x)
                                                / cell_width_x_m[candidate_i]
                                                + ti.abs(normal.y)
                                                / cell_width_y_m[candidate_j]
                                                + ti.abs(normal.z)
                                                / cell_width_z_m[candidate_k]
                                            )
                                            candidate_normal_width = 1.0 / ti.max(
                                                candidate_spacing_inv,
                                                1.0e-12,
                                            )
                                            candidate_min_width = ti.min(
                                                node_normal_width,
                                                candidate_normal_width,
                                            )
                                            candidate_gap_floor = ti.max(
                                                1.0e-12,
                                                1.0e-3 * candidate_min_width,
                                            )
                                            if candidate_gap > candidate_gap_floor:
                                                normal_line_i = candidate_i
                                                normal_line_j = candidate_j
                                                normal_line_k = candidate_k
                                                normal_line_distance = (
                                                    candidate_normal_distance
                                                )
                                                normal_line_gap = candidate_gap
                                                normal_line_normal_width = (
                                                    candidate_normal_width
                                                )
                                                normal_line_min_width = (
                                                    candidate_min_width
                                                )
                                                normal_line_gap_floor = (
                                                    candidate_gap_floor
                                                )
                                        step_index += 1
                                if normal_line_i >= 0:
                                    neighbor_i = normal_line_i
                                    neighbor_j = normal_line_j
                                    neighbor_k = normal_line_k
                                    neighbor_distance = normal_line_distance
                                    reconstruction_gap = normal_line_gap
                                    neighbor_normal_width = normal_line_normal_width
                                    min_normal_width = normal_line_min_width
                                    reconstruction_gap_floor = normal_line_gap_floor
                                    row_reconstructable = 1
                            if row_reconstructable == 0:
                                nearest_i = -1
                                nearest_j = -1
                                nearest_k = -1
                                nearest_distance = 0.0
                                nearest_gap = 0.0
                                nearest_normal_width = 0.0
                                nearest_min_width = 0.0
                                nearest_gap_floor = 0.0
                                nearest_metric = 1.0e30
                                di = (
                                    -HIBM_PRESSURE_NEUMANN_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                                )
                                while (
                                    di
                                    <= HIBM_PRESSURE_NEUMANN_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                                ):
                                    dj = (
                                        -HIBM_PRESSURE_NEUMANN_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                                    )
                                    while (
                                        dj
                                        <= HIBM_PRESSURE_NEUMANN_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                                    ):
                                        dk = (
                                            -HIBM_PRESSURE_NEUMANN_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                                        )
                                        while (
                                            dk
                                            <= HIBM_PRESSURE_NEUMANN_NEAREST_FLUID_FALLBACK_RADIUS_CELLS
                                        ):
                                            candidate_i = owner_i + di
                                            candidate_j = owner_j + dj
                                            candidate_k = owner_k + dk
                                            if (
                                                0 <= candidate_i
                                                and candidate_i < nx
                                                and 0 <= candidate_j
                                                and candidate_j < ny
                                                and 0 <= candidate_k
                                                and candidate_k < nz
                                                and (
                                                    candidate_i != owner_i
                                                    or candidate_j != owner_j
                                                    or candidate_k != owner_k
                                                )
                                                and obstacle_field[
                                                    candidate_i,
                                                    candidate_j,
                                                    candidate_k,
                                                ]
                                                == 0
                                                and velocity_dirichlet_active[
                                                    candidate_i,
                                                    candidate_j,
                                                    candidate_k,
                                                ]
                                                == 0
                                            ):
                                                candidate_position = ti.Vector(
                                                    [
                                                        cell_center_x_m[candidate_i],
                                                        cell_center_y_m[candidate_j],
                                                        cell_center_z_m[candidate_k],
                                                    ]
                                                )
                                                candidate_distance = (
                                                    candidate_position - boundary_point
                                                ).dot(normal)
                                                candidate_gap = ti.abs(
                                                    candidate_distance - node_distance
                                                )
                                                candidate_spacing_inv = (
                                                    ti.abs(normal.x)
                                                    / cell_width_x_m[candidate_i]
                                                    + ti.abs(normal.y)
                                                    / cell_width_y_m[candidate_j]
                                                    + ti.abs(normal.z)
                                                    / cell_width_z_m[candidate_k]
                                                )
                                                candidate_normal_width = 1.0 / ti.max(
                                                    candidate_spacing_inv,
                                                    1.0e-12,
                                                )
                                                candidate_min_width = ti.min(
                                                    node_normal_width,
                                                    candidate_normal_width,
                                                )
                                                candidate_gap_floor = ti.max(
                                                    1.0e-12,
                                                    1.0e-3 * candidate_min_width,
                                                )
                                                index_metric = ti.cast(
                                                    di * di + dj * dj + dk * dk,
                                                    ti.f32,
                                                )
                                                if (
                                                    candidate_gap > candidate_gap_floor
                                                    and (
                                                        index_metric < nearest_metric
                                                        or (
                                                            index_metric
                                                            == nearest_metric
                                                            and candidate_gap
                                                            > nearest_gap
                                                        )
                                                    )
                                                ):
                                                    nearest_i = candidate_i
                                                    nearest_j = candidate_j
                                                    nearest_k = candidate_k
                                                    nearest_distance = candidate_distance
                                                    nearest_gap = candidate_gap
                                                    nearest_normal_width = (
                                                        candidate_normal_width
                                                    )
                                                    nearest_min_width = candidate_min_width
                                                    nearest_gap_floor = (
                                                        candidate_gap_floor
                                                    )
                                                    nearest_metric = index_metric
                                            dk += 1
                                        dj += 1
                                    di += 1
                                if nearest_i >= 0:
                                    neighbor_i = nearest_i
                                    neighbor_j = nearest_j
                                    neighbor_k = nearest_k
                                    neighbor_distance = nearest_distance
                                    reconstruction_gap = nearest_gap
                                    neighbor_normal_width = nearest_normal_width
                                    min_normal_width = nearest_min_width
                                    reconstruction_gap_floor = nearest_gap_floor
                                    row_reconstructable = 1
                        row_rejected_by_pressure_boundary = 0
                        if (
                            row_reconstructable != 0
                            and (
                                self._pressure_neumann_cell_touches_velocity_dirichlet_projection_face(
                                    velocity_dirichlet_active,
                                    owner_i,
                                    owner_j,
                                    owner_k,
                                    nx,
                                    ny,
                                    nz,
                                )
                                or self._pressure_neumann_cell_touches_velocity_dirichlet_projection_face(
                                    velocity_dirichlet_active,
                                    neighbor_i,
                                    neighbor_j,
                                    neighbor_k,
                                    nx,
                                    ny,
                                    nz,
                                )
                            )
                        ):
                            row_reconstructable = 0
                            row_rejected_by_pressure_boundary = 1
                            ti.atomic_add(
                                self.report_pressure_neumann_skipped_pressure_boundary_adjacent_rows[
                                    None
                                ],
                                1,
                            )
                        if row_reconstructable != 0:
                            neighbor_volume_m3 = (
                                cell_width_x_m[neighbor_i]
                                * cell_width_y_m[neighbor_j]
                                * cell_width_z_m[neighbor_k]
                            )
                            cell_interface_area_m2 = 0.5 * (
                                volume_m3 / ti.max(node_normal_width, 1.0e-12)
                                + neighbor_volume_m3
                                / ti.max(neighbor_normal_width, 1.0e-12)
                            )
                            candidate_count = ti.max(
                                marker_pressure_neumann_candidate_node_count[marker],
                                1,
                            )
                            marker_area_per_candidate_m2 = (
                                ti.max(marker_surface_area_m2[marker], 0.0)
                                / ti.cast(candidate_count, ti.f32)
                            )
                            interface_area_m2 = ti.min(
                                cell_interface_area_m2,
                                marker_area_per_candidate_m2,
                            )
                            raw_transmissibility = (
                                interface_area_m2 / reconstruction_gap
                            )
                            transmissibility_limit = (
                                20.0
                                * interface_area_m2
                                / ti.max(min_normal_width, 1.0e-12)
                            )
                            transmissibility = ti.min(
                                raw_transmissibility,
                                transmissibility_limit,
                            )
                            if raw_transmissibility > transmissibility_limit:
                                ti.atomic_add(
                                    self.report_pressure_neumann_transmissibility_capped_rows[
                                        None
                                    ],
                                    1,
                                )
                            node_coefficient = transmissibility / volume_m3
                            neighbor_coefficient = (
                                transmissibility / neighbor_volume_m3
                            )
                            pressure_jump = (
                                self.pressure_neumann_gradient_field[node]
                                * (node_distance - neighbor_distance)
                            )
                            node_rhs_density = node_coefficient * pressure_jump
                            neighbor_rhs_density = (
                                -neighbor_coefficient * pressure_jump
                            )
                            slot_index = 0
                            if max_pressure_coupling_slots > 1:
                                slot_index = ti.atomic_add(
                                    pressure_coupling_active[
                                        owner_i,
                                        owner_j,
                                        owner_k,
                                    ],
                                    1,
                                )
                            else:
                                slot_index = ti.atomic_or(
                                    pressure_coupling_active[
                                        owner_i,
                                        owner_j,
                                        owner_k,
                                    ],
                                    1,
                                )
                            ti.atomic_max(
                                self.report_pressure_neumann_max_owner_slot_count[
                                    None
                                ],
                                slot_index + 1,
                            )
                            if slot_index > 0:
                                ti.atomic_add(
                                    self.report_pressure_neumann_duplicate_owner_rows[
                                        None
                                    ],
                                    1,
                                )
                            row_enters_pressure_matrix = 0
                            if slot_index < max_pressure_coupling_slots:
                                if slot_index == 0:
                                    pressure_coupling_neighbor[
                                        owner_i,
                                        owner_j,
                                        owner_k,
                                    ] = ti.Vector([neighbor_i, neighbor_j, neighbor_k])
                                    pressure_coupling_coefficient[
                                        owner_i,
                                        owner_j,
                                        owner_k,
                                    ] = transmissibility
                                else:
                                    extra_slot = slot_index - 1
                                    pressure_coupling_extra_neighbor[
                                        owner_i,
                                        owner_j,
                                        owner_k,
                                        extra_slot,
                                    ] = ti.Vector([neighbor_i, neighbor_j, neighbor_k])
                                    pressure_coupling_extra_coefficient[
                                        owner_i,
                                        owner_j,
                                        owner_k,
                                        extra_slot,
                                    ] = transmissibility
                                row_enters_pressure_matrix = 1
                            else:
                                merge_slot = -1
                                primary_neighbor = pressure_coupling_neighbor[
                                    owner_i,
                                    owner_j,
                                    owner_k,
                                ]
                                if (
                                    primary_neighbor[0] == neighbor_i
                                    and primary_neighbor[1] == neighbor_j
                                    and primary_neighbor[2] == neighbor_k
                                ):
                                    merge_slot = 0
                                for candidate_extra_slot in ti.static(
                                    range(PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS)
                                ):
                                    if (
                                        merge_slot < 0
                                        and candidate_extra_slot + 1
                                        < max_pressure_coupling_slots
                                    ):
                                        extra_neighbor = (
                                            pressure_coupling_extra_neighbor[
                                                owner_i,
                                                owner_j,
                                                owner_k,
                                                candidate_extra_slot,
                                            ]
                                        )
                                        if (
                                            extra_neighbor[0] == neighbor_i
                                            and extra_neighbor[1] == neighbor_j
                                            and extra_neighbor[2] == neighbor_k
                                        ):
                                            merge_slot = candidate_extra_slot + 1
                                if merge_slot == 0:
                                    ti.atomic_add(
                                        pressure_coupling_coefficient[
                                            owner_i,
                                            owner_j,
                                            owner_k,
                                        ],
                                        transmissibility,
                                    )
                                    row_enters_pressure_matrix = 1
                                elif merge_slot > 0:
                                    ti.atomic_add(
                                        pressure_coupling_extra_coefficient[
                                            owner_i,
                                            owner_j,
                                            owner_k,
                                            merge_slot - 1,
                                        ],
                                        transmissibility,
                                    )
                                    row_enters_pressure_matrix = 1
                                else:
                                    if pressure_interface_row_list_enabled == 0:
                                        ti.atomic_add(
                                            self.report_pressure_neumann_overflow_owner_rows[
                                                None
                                            ],
                                            1,
                                        )
                            if pressure_interface_row_list_enabled != 0:
                                row_enters_pressure_matrix = 0
                                row_index = ti.atomic_add(
                                    pressure_interface_row_count[None],
                                    1,
                                )
                                if row_index < pressure_interface_row_capacity:
                                    pressure_interface_row_owner[row_index] = ti.Vector(
                                        [owner_i, owner_j, owner_k]
                                    )
                                    pressure_interface_row_neighbor[row_index] = (
                                        ti.Vector([neighbor_i, neighbor_j, neighbor_k])
                                    )
                                    pressure_interface_row_transmissibility[
                                        row_index
                                    ] = transmissibility
                                    row_enters_pressure_matrix = 1
                                else:
                                    ti.atomic_add(
                                        self.report_pressure_neumann_overflow_owner_rows[
                                            None
                                        ],
                                        1,
                                    )
                            if row_enters_pressure_matrix != 0:
                                if relocated_obstacle_owner != 0:
                                    ti.atomic_add(
                                        self.report_pressure_neumann_relocated_obstacle_owner_rows[
                                            None
                                        ],
                                        1,
                                    )
                                ti.atomic_add(
                                    pressure_matrix_diagonal[
                                        owner_i,
                                        owner_j,
                                        owner_k,
                                    ],
                                    node_coefficient,
                                )
                                ti.atomic_add(
                                    pressure_matrix_diagonal[
                                        neighbor_i,
                                        neighbor_j,
                                        neighbor_k,
                                    ],
                                    neighbor_coefficient,
                                )
                                ti.atomic_add(
                                    pressure_matrix_rhs[owner_i, owner_j, owner_k],
                                    node_rhs_density,
                                )
                                ti.atomic_add(
                                    pressure_matrix_rhs[
                                        neighbor_i,
                                        neighbor_j,
                                        neighbor_k,
                                    ],
                                    neighbor_rhs_density,
                                )
                                ti.atomic_add(
                                    self.report_pressure_neumann_matrix_rows[None],
                                    1,
                                )
                                previous_marker_row_count = ti.atomic_add(
                                    self.marker_pressure_neumann_row_count[marker],
                                    1,
                                )
                                if previous_marker_row_count == 0:
                                    # S2-A6: anchor the marker to its first
                                    # row-owning fluid cell. All acceptance
                                    # paths (direct interior-point walk, the
                                    # bounds-relocated axial fallback and the
                                    # obstacle-alternate axial fallback)
                                    # converge on this row-write block, so
                                    # every row writer is covered; the 0 -> 1
                                    # transition of the per-marker row counter
                                    # elects exactly one writer per marker,
                                    # keeping the 3-component store tear-free
                                    # without a second guard field. The node is
                                    # obstacle-free and receives diagonal/rhs
                                    # terms above, i.e. it participates in the
                                    # pressure solve by construction.
                                    marker_pressure_anchor_cell[marker] = ti.Vector(
                                        [owner_i, owner_j, owner_k]
                                    )
                                self.report_pressure_neumann_rhs_integral[None] += (
                                    ti.cast(node_rhs_density * volume_m3, ti.f64)
                                    + ti.cast(
                                        neighbor_rhs_density * neighbor_volume_m3,
                                        ti.f64,
                                    )
                                )
                                ti.atomic_max(
                                    self.report_pressure_neumann_max_abs_rhs[None],
                                    ti.max(
                                        ti.abs(node_rhs_density),
                                        ti.abs(neighbor_rhs_density),
                                    ),
                                )
                                ti.atomic_min(
                                    self.report_pressure_neumann_min_reconstruction_gap_m[
                                        None
                                    ],
                                    reconstruction_gap,
                                )
                                ti.atomic_max(
                                    self.report_pressure_neumann_max_reconstruction_gap_m[
                                        None
                                    ],
                                    reconstruction_gap,
                                )
                                ti.atomic_max(
                                    self.report_pressure_neumann_max_transmissibility_m[
                                        None
                                    ],
                                    transmissibility,
                                )
                                ti.atomic_max(
                                    self.report_pressure_neumann_max_raw_transmissibility_m[
                                        None
                                    ],
                                    raw_transmissibility,
                                )
                                ti.atomic_max(
                                    self.report_pressure_neumann_max_transmissibility_limit_m[
                                        None
                                    ],
                                    transmissibility_limit,
                                )
                                ti.atomic_max(
                                    self.report_pressure_neumann_max_diagonal_per_m2[
                                        None
                                    ],
                                    ti.max(node_coefficient, neighbor_coefficient),
                                )
                        elif (
                            row_rejected_by_pressure_boundary == 0
                            and ti.abs(self.pressure_neumann_gradient_field[node])
                            > HIBM_PRESSURE_NEUMANN_ZERO_GRADIENT_TOLERANCE_PA_PER_M
                        ):
                            ti.atomic_add(
                                self.report_pressure_neumann_invalid_reconstruction_rows[
                                    None
                                ],
                                1,
                            )
                            ti.atomic_add(
                                self.report_pressure_neumann_invalid_unreconstructable_rows[
                                    None
                                ],
                                1,
                            )
                            self._record_pressure_neumann_invalid_diagnostic_row(
                                node,
                                owner_i,
                                owner_j,
                                owner_k,
                                neighbor_i,
                                neighbor_j,
                                neighbor_k,
                                diagnostic_anchor_i,
                                diagnostic_anchor_j,
                                diagnostic_anchor_k,
                                marker,
                                PRESSURE_NEUMANN_INVALID_REASON_UNRECONSTRUCTABLE,
                                node_distance,
                                normal_denominator,
                                reconstruction_gap,
                            )
                    else:
                        ti.atomic_add(
                            self.report_pressure_neumann_invalid_reconstruction_rows[
                                None
                            ],
                            1,
                        )
                        ti.atomic_add(
                            self.report_pressure_neumann_invalid_nonpositive_volume_rows[
                                None
                            ],
                            1,
                        )
                        self._record_pressure_neumann_invalid_diagnostic_row(
                            node,
                            owner_i,
                            owner_j,
                            owner_k,
                            -1,
                            -1,
                            -1,
                            diagnostic_anchor_i,
                            diagnostic_anchor_j,
                            diagnostic_anchor_k,
                            marker,
                            PRESSURE_NEUMANN_INVALID_REASON_NONPOSITIVE_VOLUME,
                            0.0,
                            0.0,
                            0.0,
                        )
                else:
                    ti.atomic_add(
                        self.report_pressure_neumann_invalid_reconstruction_rows[None],
                        1,
                    )
                    ti.atomic_add(
                        self.report_pressure_neumann_invalid_bad_marker_rows[None],
                        1,
                    )
                    self._record_pressure_neumann_invalid_diagnostic_row(
                        node,
                        owner_i,
                        owner_j,
                        owner_k,
                        -1,
                        -1,
                        -1,
                        diagnostic_anchor_i,
                        diagnostic_anchor_j,
                        diagnostic_anchor_k,
                        marker,
                        PRESSURE_NEUMANN_INVALID_REASON_BAD_MARKER,
                        0.0,
                        0.0,
                        0.0,
                    )

    def assemble_pressure_neumann_matrix_rows(
        self,
        pressure_matrix_diagonal,
        pressure_matrix_rhs,
        pressure_coupling_active,
        pressure_coupling_neighbor,
        pressure_coupling_coefficient,
        obstacle_field,
        velocity_dirichlet_active,
        cell_width_x_m,
        cell_width_y_m,
        cell_width_z_m,
        search: HibmMpmIbNodeSearch,
        markers: HibmMpmSurfaceMarkers,
        *,
        pressure_coupling_extra_neighbor=None,
        pressure_coupling_extra_coefficient=None,
        pressure_interface_row_count=None,
        pressure_interface_row_owner=None,
        pressure_interface_row_neighbor=None,
        pressure_interface_row_transmissibility=None,
        pressure_interface_row_capacity: int = 0,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
        velocity_dirichlet_marker_region_id=None,
        marker_region_id=None,
    ) -> HibmMpmPressureNeumannMatrixReport:
        self._validate_search_and_markers(search, markers)
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._clear_pressure_neumann_rows_by_marker_kernel(
            int(markers.marker_count),
        )
        self._count_pressure_neumann_candidate_nodes_by_marker_kernel(
            search.nearest_marker,
            int(markers.marker_count),
        )
        extra_slots_available = (
            pressure_coupling_extra_neighbor is not None
            and pressure_coupling_extra_coefficient is not None
        )
        if (
            pressure_coupling_extra_neighbor is None
            and pressure_coupling_extra_coefficient is not None
        ) or (
            pressure_coupling_extra_neighbor is not None
            and pressure_coupling_extra_coefficient is None
        ):
            raise ValueError(
                "pressure_coupling_extra_neighbor and "
                "pressure_coupling_extra_coefficient must be provided together"
            )
        max_pressure_coupling_slots = 1
        if extra_slots_available:
            max_pressure_coupling_slots = PRESSURE_INTERFACE_COUPLING_SLOT_COUNT
        if pressure_coupling_extra_neighbor is None:
            pressure_coupling_extra_neighbor = (
                self._fallback_pressure_coupling_extra_neighbor
            )
        if pressure_coupling_extra_coefficient is None:
            pressure_coupling_extra_coefficient = (
                self._fallback_pressure_coupling_extra_coefficient
            )
        row_list_enabled = (
            pressure_interface_row_count is not None
            and pressure_interface_row_owner is not None
            and pressure_interface_row_neighbor is not None
            and pressure_interface_row_transmissibility is not None
            and int(pressure_interface_row_capacity) > 0
        )
        if not row_list_enabled:
            pressure_interface_row_count = self._fallback_pressure_interface_row_count
            pressure_interface_row_owner = self._fallback_pressure_interface_row_owner
            pressure_interface_row_neighbor = (
                self._fallback_pressure_interface_row_neighbor
            )
            pressure_interface_row_transmissibility = (
                self._fallback_pressure_interface_row_transmissibility
            )
            pressure_interface_row_capacity = 0
        # S2-A6: full-capacity sentinel reset before assembly so markers
        # that fail to produce a row (invalid reconstruction paths) keep
        # (-1, -1, -1) and stay on the sampler's invalid path.
        markers.reset_pressure_anchor_cells()
        self._assemble_pressure_neumann_matrix_rows_kernel(
            pressure_matrix_diagonal,
            pressure_matrix_rhs,
            pressure_coupling_active,
            pressure_coupling_neighbor,
            pressure_coupling_coefficient,
            pressure_coupling_extra_neighbor,
            pressure_coupling_extra_coefficient,
            pressure_interface_row_count,
            pressure_interface_row_owner,
            pressure_interface_row_neighbor,
            pressure_interface_row_transmissibility,
            obstacle_field,
            velocity_dirichlet_active,
            cell_width_x_m,
            cell_width_y_m,
            cell_width_z_m,
            search.node_boundary_point_m,
            search.node_interior_fluid_point_m,
            search.node_anchor_cell,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            search.nearest_marker,
            markers.A_gamma_m2,
            self.marker_pressure_neumann_candidate_node_count,
            markers.marker_pressure_anchor_cell,
            int(markers.marker_count),
            int(max_pressure_coupling_slots),
            1 if bool(row_list_enabled) else 0,
            int(pressure_interface_row_capacity),
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        self._summarize_pressure_neumann_rows_by_marker_kernel(
            int(markers.marker_count),
        )
        if bool(row_list_enabled):
            compacted_row_count = self._compact_pressure_interface_row_list(
                pressure_interface_row_count,
                pressure_interface_row_owner,
                pressure_interface_row_neighbor,
                pressure_interface_row_transmissibility,
                pressure_interface_row_capacity=int(pressure_interface_row_capacity),
            )
            self.report_pressure_neumann_matrix_rows[None] = int(compacted_row_count)
        active_rows = int(self.report_pressure_neumann_matrix_rows[None])
        min_reconstruction_gap_m = 0.0
        if active_rows > 0:
            min_reconstruction_gap_m = float(
                self.report_pressure_neumann_min_reconstruction_gap_m[None]
            )
        return HibmMpmPressureNeumannMatrixReport(
            active_pressure_neumann_rows=active_rows,
            rhs_integral=float(self.report_pressure_neumann_rhs_integral[None]),
            max_abs_rhs=float(self.report_pressure_neumann_max_abs_rhs[None]),
            skipped_velocity_dirichlet_row_count=int(
                self.report_pressure_neumann_skipped_velocity_dirichlet_rows[None]
            ),
            skipped_pressure_boundary_adjacent_row_count=int(
                self.report_pressure_neumann_skipped_pressure_boundary_adjacent_rows[
                    None
                ]
            ),
            skipped_obstacle_owner_row_count=int(
                self.report_pressure_neumann_skipped_obstacle_owner_rows[None]
            ),
            relocated_obstacle_owner_row_count=int(
                self.report_pressure_neumann_relocated_obstacle_owner_rows[None]
            ),
            duplicate_owner_row_count=int(
                self.report_pressure_neumann_duplicate_owner_rows[None]
            ),
            overflow_owner_row_count=int(
                self.report_pressure_neumann_overflow_owner_rows[None]
            ),
            max_owner_slot_count=int(
                self.report_pressure_neumann_max_owner_slot_count[None]
            ),
            pressure_interface_row_list_enabled=bool(row_list_enabled),
            pressure_interface_row_list_count=(
                int(pressure_interface_row_count[None])
                if bool(row_list_enabled)
                else 0
            ),
            active_pressure_neumann_marker_count=int(
                self.report_pressure_neumann_active_marker_count[None]
            ),
            max_pressure_neumann_rows_per_marker=int(
                self.report_pressure_neumann_max_rows_per_marker[None]
            ),
            invalid_reconstruction_row_count=int(
                self.report_pressure_neumann_invalid_reconstruction_rows[None]
            ),
            invalid_unreconstructable_row_count=int(
                self.report_pressure_neumann_invalid_unreconstructable_rows[None]
            ),
            invalid_bad_marker_row_count=int(
                self.report_pressure_neumann_invalid_bad_marker_rows[None]
            ),
            invalid_nonpositive_volume_row_count=int(
                self.report_pressure_neumann_invalid_nonpositive_volume_rows[None]
            ),
            min_reconstruction_gap_m=min_reconstruction_gap_m,
            max_reconstruction_gap_m=float(
                self.report_pressure_neumann_max_reconstruction_gap_m[None]
            ),
            max_transmissibility_m=float(
                self.report_pressure_neumann_max_transmissibility_m[None]
            ),
            max_raw_transmissibility_m=float(
                self.report_pressure_neumann_max_raw_transmissibility_m[None]
            ),
            max_transmissibility_limit_m=float(
                self.report_pressure_neumann_max_transmissibility_limit_m[None]
            ),
            transmissibility_capped_row_count=int(
                self.report_pressure_neumann_transmissibility_capped_rows[None]
            ),
            max_diagonal_per_m2=float(
                self.report_pressure_neumann_max_diagonal_per_m2[None]
            ),
        )

    def _compact_pressure_interface_row_list(
        self,
        pressure_interface_row_count,
        pressure_interface_row_owner,
        pressure_interface_row_neighbor,
        pressure_interface_row_transmissibility,
        *,
        pressure_interface_row_capacity: int,
    ) -> int:
        raw_row_count = int(pressure_interface_row_count[None])
        row_capacity = max(0, int(pressure_interface_row_capacity))
        if raw_row_count <= 1 or row_capacity <= 1:
            return raw_row_count
        if raw_row_count > row_capacity:
            return raw_row_count

        owners = pressure_interface_row_owner.to_numpy()
        neighbors = pressure_interface_row_neighbor.to_numpy()
        transmissibility = pressure_interface_row_transmissibility.to_numpy()
        pair_to_compact_index: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
        compact_count = 0

        for row_index in range(raw_row_count):
            owner = tuple(int(value) for value in owners[row_index])
            neighbor = tuple(int(value) for value in neighbors[row_index])
            coefficient = float(transmissibility[row_index])
            if coefficient <= 0.0:
                owners[compact_count] = owners[row_index]
                neighbors[compact_count] = neighbors[row_index]
                transmissibility[compact_count] = transmissibility[row_index]
                compact_count += 1
                continue
            pair_key = (owner, neighbor)
            compact_index = pair_to_compact_index.get(pair_key)
            if compact_index is None:
                pair_to_compact_index[pair_key] = compact_count
                owners[compact_count] = owners[row_index]
                neighbors[compact_count] = neighbors[row_index]
                transmissibility[compact_count] = transmissibility[row_index]
                compact_count += 1
            else:
                transmissibility[compact_index] += transmissibility[row_index]

        if compact_count == raw_row_count:
            return raw_row_count

        owners[compact_count:raw_row_count] = -1
        neighbors[compact_count:raw_row_count] = -1
        transmissibility[compact_count:raw_row_count] = 0.0
        pressure_interface_row_owner.from_numpy(owners)
        pressure_interface_row_neighbor.from_numpy(neighbors)
        pressure_interface_row_transmissibility.from_numpy(transmissibility)
        pressure_interface_row_count[None] = int(compact_count)
        return int(compact_count)

    def is_active(self, node_index: tuple[int, int, int]) -> bool:
        return bool(int(self.active_ib_node[self._node_index(node_index)]))

    def velocity_dirichlet_mps(
        self,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return self._vector_field_value_tuple(
            self.velocity_dirichlet_mps_field,
            node_index,
        )

    def pressure_neumann_normal(
        self,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return self._vector_field_value_tuple(
            self.pressure_neumann_normal_field,
            node_index,
        )

    def pressure_neumann_gradient_pa_per_m(
        self,
        node_index: tuple[int, int, int],
    ) -> float:
        return float(
            self.pressure_neumann_gradient_field[self._node_index(node_index)]
        )

    def pressure_neumann_invalid_diagnostic_rows(
        self,
        *,
        search=None,
        markers=None,
        fluid=None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return captured pressure-Neumann invalid-row diagnostics.

        The Taichi assembly kernel records only the first fixed-capacity slice;
        the scalar report count remains the authoritative total invalid count.
        """

        total_count = int(self.pressure_neumann_invalid_diag_count[None])
        captured_count = min(
            total_count,
            PRESSURE_NEUMANN_INVALID_DIAGNOSTIC_CAPACITY,
        )
        if limit is not None:
            captured_count = min(captured_count, max(0, int(limit)))
        if captured_count <= 0:
            return []

        reasons = self.pressure_neumann_invalid_diag_reason.to_numpy()
        markers_idx = self.pressure_neumann_invalid_diag_marker.to_numpy()
        nodes = self.pressure_neumann_invalid_diag_node.to_numpy()
        owners = self.pressure_neumann_invalid_diag_owner.to_numpy()
        neighbors = self.pressure_neumann_invalid_diag_neighbor.to_numpy()
        anchors = self.pressure_neumann_invalid_diag_anchor.to_numpy()
        node_distances = (
            self.pressure_neumann_invalid_diag_node_distance_m.to_numpy()
        )
        normal_denominators = (
            self.pressure_neumann_invalid_diag_normal_denominator_m.to_numpy()
        )
        reconstruction_gaps = (
            self.pressure_neumann_invalid_diag_reconstruction_gap_m.to_numpy()
        )
        normals = self.pressure_neumann_normal_field.to_numpy()
        gradients = self.pressure_neumann_gradient_field.to_numpy()

        boundary_points = None
        interior_points = None
        nearest_marker = None
        if search is not None:
            boundary_points = search.node_boundary_point_m.to_numpy()
            interior_points = search.node_interior_fluid_point_m.to_numpy()
            nearest_marker = search.nearest_marker.to_numpy()

        marker_count = 0
        marker_regions = None
        if markers is not None:
            marker_count = int(markers.marker_count)
            marker_regions = markers.region_id.to_numpy()

        cell_center_x = None
        cell_center_y = None
        cell_center_z = None
        obstacle = None
        velocity_dirichlet_active = None
        if fluid is not None:
            cell_center_x = fluid.cell_center_x_m.to_numpy()
            cell_center_y = fluid.cell_center_y_m.to_numpy()
            cell_center_z = fluid.cell_center_z_m.to_numpy()
            obstacle = fluid.obstacle.to_numpy()
            velocity_dirichlet_active = (
                fluid.velocity_dirichlet_boundary_active.to_numpy()
            )

        def _triple(values) -> tuple[int, int, int]:
            return (int(values[0]), int(values[1]), int(values[2]))

        def _valid(index: tuple[int, int, int]) -> bool:
            return all(
                0 <= value < limit
                for value, limit in zip(index, self.grid_nodes, strict=True)
            )

        def _point_from_grid(index: tuple[int, int, int]) -> tuple[float, float, float]:
            if (
                cell_center_x is None
                or cell_center_y is None
                or cell_center_z is None
                or not _valid(index)
            ):
                return (math.nan, math.nan, math.nan)
            return (
                float(cell_center_x[index[0]]),
                float(cell_center_y[index[1]]),
                float(cell_center_z[index[2]]),
            )

        def _point_from_field(field, index: tuple[int, int, int]) -> tuple[float, float, float]:
            if field is None or not _valid(index):
                return (math.nan, math.nan, math.nan)
            value = field[index]
            return (float(value[0]), float(value[1]), float(value[2]))

        def _cell_flag(field, index: tuple[int, int, int]) -> int:
            if field is None or not _valid(index):
                return -1
            return int(field[index])

        rows: list[dict[str, Any]] = []
        for row_index in range(captured_count):
            node = _triple(nodes[row_index])
            owner = _triple(owners[row_index])
            neighbor = _triple(neighbors[row_index])
            anchor = _triple(anchors[row_index])
            marker = int(markers_idx[row_index])
            reason_code = int(reasons[row_index])
            node_center = _point_from_grid(node)
            owner_center = _point_from_grid(owner)
            neighbor_center = _point_from_grid(neighbor)
            anchor_center = _point_from_grid(anchor)
            boundary_point = _point_from_field(boundary_points, node)
            interior_point = _point_from_field(interior_points, node)
            row = {
                "row_index": row_index,
                "reason_code": reason_code,
                "reason": PRESSURE_NEUMANN_INVALID_REASON_NAMES.get(
                    reason_code,
                    "unknown",
                ),
                "node_i": node[0],
                "node_j": node[1],
                "node_k": node[2],
                "owner_i": owner[0],
                "owner_j": owner[1],
                "owner_k": owner[2],
                "neighbor_i": neighbor[0],
                "neighbor_j": neighbor[1],
                "neighbor_k": neighbor[2],
                "anchor_i": anchor[0],
                "anchor_j": anchor[1],
                "anchor_k": anchor[2],
                "marker_index": marker,
                "marker_region_id": (
                    int(marker_regions[marker])
                    if marker_regions is not None and 0 <= marker < marker_count
                    else -1
                ),
                "nearest_marker_index": (
                    int(nearest_marker[node])
                    if nearest_marker is not None and _valid(node)
                    else -1
                ),
                "node_distance_m": float(node_distances[row_index]),
                "normal_denominator_m": float(normal_denominators[row_index]),
                "reconstruction_gap_m": float(reconstruction_gaps[row_index]),
                "pressure_neumann_gradient_pa_per_m": (
                    float(gradients[node]) if _valid(node) else math.nan
                ),
                "normal_x": float(normals[node][0]) if _valid(node) else math.nan,
                "normal_y": float(normals[node][1]) if _valid(node) else math.nan,
                "normal_z": float(normals[node][2]) if _valid(node) else math.nan,
                "node_obstacle": _cell_flag(obstacle, node),
                "owner_obstacle": _cell_flag(obstacle, owner),
                "neighbor_obstacle": _cell_flag(obstacle, neighbor),
                "anchor_obstacle": _cell_flag(obstacle, anchor),
                "node_velocity_dirichlet_active": _cell_flag(
                    velocity_dirichlet_active,
                    node,
                ),
                "owner_velocity_dirichlet_active": _cell_flag(
                    velocity_dirichlet_active,
                    owner,
                ),
                "neighbor_velocity_dirichlet_active": _cell_flag(
                    velocity_dirichlet_active,
                    neighbor,
                ),
                "anchor_velocity_dirichlet_active": _cell_flag(
                    velocity_dirichlet_active,
                    anchor,
                ),
                "node_x_m": node_center[0],
                "node_y_m": node_center[1],
                "node_z_m": node_center[2],
                "owner_x_m": owner_center[0],
                "owner_y_m": owner_center[1],
                "owner_z_m": owner_center[2],
                "neighbor_x_m": neighbor_center[0],
                "neighbor_y_m": neighbor_center[1],
                "neighbor_z_m": neighbor_center[2],
                "anchor_x_m": anchor_center[0],
                "anchor_y_m": anchor_center[1],
                "anchor_z_m": anchor_center[2],
                "boundary_x_m": boundary_point[0],
                "boundary_y_m": boundary_point[1],
                "boundary_z_m": boundary_point[2],
                "interior_x_m": interior_point[0],
                "interior_y_m": interior_point[1],
                "interior_z_m": interior_point[2],
            }
            rows.append(row)
        return rows

    @ti.kernel
    def _update_pressure_neumann_gradient_from_ib_nodes_kernel(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        node_interior_fluid_point_m: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        density_kgm3: ti.f32,
        dt_s: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_pressure_neumann_gradient_node_count[None] = 0
        self.report_pressure_neumann_gradient_max_abs[None] = 0.0
        self.report_pressure_neumann_gradient_raw_max_abs[None] = 0.0
        self.report_pressure_neumann_gradient_limited_count[None] = 0
        for node in ti.grouped(self.active_ib_node):
            if self.active_ib_node[node] != 1:
                self.pressure_neumann_gradient_field[node] = 0.0
            if self.active_ib_node[node] == 1:
                self.pressure_neumann_gradient_field[node] = 0.0
                interior_point = node_interior_fluid_point_m[node]
                grid_coordinate = self._grid_coordinate_from_fields(
                    interior_point,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                    nx,
                    ny,
                    nz,
                )
                predictor_velocity, fluid_weight = self._sample_fluid_velocity_trilinear(
                    velocity_field,
                    obstacle_field,
                    grid_coordinate.x,
                    grid_coordinate.y,
                    grid_coordinate.z,
                    nx,
                    ny,
                    nz,
                )
                if fluid_weight > 1.0e-12:
                    normal = self.pressure_neumann_normal_field[node]
                    boundary_velocity = self.velocity_dirichlet_mps_field[node]
                    normal_gradient = (
                        density_kgm3
                        * (predictor_velocity - boundary_velocity).dot(normal)
                        / dt_s
                    )
                    self.pressure_neumann_gradient_field[node] = normal_gradient
                    ti.atomic_add(
                        self.report_pressure_neumann_gradient_node_count[None],
                        1,
                    )
                    ti.atomic_max(
                        self.report_pressure_neumann_gradient_raw_max_abs[None],
                        ti.abs(normal_gradient),
                    )
                    ti.atomic_max(
                        self.report_pressure_neumann_gradient_max_abs[None],
                        ti.abs(normal_gradient),
                    )

    def update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
        self,
        *,
        velocity_field,
        obstacle_field,
        search: HibmMpmIbNodeSearch,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
        density_kgm3: float,
        dt_s: float,
    ) -> HibmMpmPressureNeumannGradientReport:
        if tuple(search.grid_nodes) != self.grid_nodes:
            raise ValueError("search.grid_nodes must match boundary grid_nodes")
        density = float(density_kgm3)
        dt = float(dt_s)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("density_kgm3 must be finite and positive")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._update_pressure_neumann_gradient_from_ib_nodes_kernel(
            velocity_field,
            obstacle_field,
            search.node_interior_fluid_point_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            density,
            dt,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        return HibmMpmPressureNeumannGradientReport(
            active_marker_count=int(
                self.report_pressure_neumann_gradient_node_count[None]
            ),
            max_abs_gradient_pa_per_m=float(
                self.report_pressure_neumann_gradient_max_abs[None]
            ),
            max_raw_abs_gradient_pa_per_m=float(
                self.report_pressure_neumann_gradient_raw_max_abs[None]
            ),
            limited_gradient_count=int(
                self.report_pressure_neumann_gradient_limited_count[None]
            ),
        )

    def _node_index(self, node_index: tuple[int, int, int]) -> tuple[int, int, int]:
        index = tuple(int(value) for value in node_index)
        if len(index) != 3 or any(
            value < 0 or value >= limit
            for value, limit in zip(index, self.grid_nodes, strict=True)
        ):
            raise IndexError("node_index out of range")
        return index

    def _vector_field_value_tuple(
        self,
        field,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        value = field[self._node_index(node_index)]
        return (float(value[0]), float(value[1]), float(value[2]))


class HibmMpmSharpCouplingState:
    """Generic sharp HIBM-MPM coupling state owned by simulation_core.

    The state bundles marker fields, IB node search fields, boundary-condition
    fields, and the per-marker pressure-Neumann gradient field so case runners
    can select the solver path without owning HIBM internals.
    """

    def __init__(
        self,
        *,
        grid_nodes: tuple[int, int, int],
        bounds_min_m: Sequence[float],
        bounds_max_m: Sequence[float],
        marker_capacity: int,
        projection_triangle_capacity: int | None = None,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        self.markers = HibmMpmSurfaceMarkers(
            marker_capacity=marker_capacity,
            projection_triangle_capacity=projection_triangle_capacity,
            runtime=runtime,
        )
        self.ib_search = HibmMpmIbNodeSearch(
            grid_nodes=grid_nodes,
            bounds_min_m=bounds_min_m,
            bounds_max_m=bounds_max_m,
            marker_capacity=marker_capacity,
            runtime=runtime,
        )
        self.ib_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=grid_nodes,
            marker_capacity=marker_capacity,
            runtime=runtime,
        )
        self.marker_pressure_neumann_gradient_pa_per_m = (
            self.ib_boundary.marker_pressure_neumann_gradient_field
        )

    def load_markers_from_surface_fields(
        self,
        surface_position_m,
        surface_normal,
        surface_area_m2,
        surface_region_id,
        *,
        marker_count: int,
        initial_velocity_mps: Sequence[float] = (0.0, 0.0, 0.0),
        surface_velocity_mps=None,
        projection_triangle_indices=None,
        projection_triangle_count: int | None = None,
    ) -> int:
        loaded_marker_count = self.markers.load_markers_from_surface_fields(
            surface_position_m,
            surface_normal,
            surface_area_m2,
            surface_region_id,
            marker_count=int(marker_count),
            initial_velocity_mps=initial_velocity_mps,
            surface_velocity_mps=surface_velocity_mps,
        )
        if projection_triangle_indices is not None:
            if projection_triangle_count is None:
                raise ValueError(
                    "projection_triangle_count is required with projection_triangle_indices"
                )
            self.markers.load_projection_triangles_from_field(
                projection_triangle_indices,
                triangle_count=int(projection_triangle_count),
            )
        return loaded_marker_count

    def advance_mpm_step(
        self,
        *,
        fluid: Any,
        mpm_external_force_n,
        mpm_particle_position_m,
        mpm_particle_velocity_mps,
        mpm_particle_normal,
        mpm_particle_area_m2,
        mpm_particle_count: int,
        solid_step: Callable[[], Any],
        search_radius_m: float,
        interior_probe_distance_m: float,
        mpm_support_radius_m: float,
        primary_region_id: int = 0,
        secondary_region_id: int = 0,
        far_pressure_region_id: int = -1,
        far_pressure_barrier_region_id: int = -1,
        far_pressure_pa: float = 0.0,
        far_pressure_side_normal_sign: float = 0.0,
        far_pressure_inside_probe_max_multiplier: float = 3.0,
        two_sided_probe_max_multiplier: float = 3.0,
        one_sided_pressure_region_id: int = -1,
        one_sided_reference_pressure_pa: float = 0.0,
        one_sided_probe_max_multiplier: float = 3.0,
        far_pressure_air_backed: bool = False,
        far_pressure_air_backed_probe_normal_sign: float = 0.0,
        fluid_dt_s: float | None = None,
        fluid_substeps: int = 1,
        projection_iterations: int = 40,
        run_fluid_predictor: bool = True,
        fluid_advection_scheme: str = "euler",
        pressure_neumann_density_kgm3: float | None = None,
        pressure_neumann_dt_s: float | None = None,
        pressure_outlet_zmin: bool = False,
        reset_pressure: bool = False,
        pressure_solver: str = "fv_cg",
        pressure_solve_failure_policy: str = "raise",
        multigrid_cycles: int | None = None,
        cg_tolerance: float = 1.0e-6,
        cg_preconditioner: str = "auto",
        surface_feedback_dt_s: float | None = None,
        divergence_cleanup_iterations: int = 0,
        divergence_cleanup_relaxation: float = 0.7,
        classify_far_internal_nodes: bool = False,
        convert_internal_nodes_to_obstacles: bool = True,
        post_dirichlet_consistency_projection_iterations: int = 3,
        diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
    ) -> HibmMpmSharpMpmStepReport:
        return advance_hibm_mpm_sharp_mpm_step(
            fluid=fluid,
            markers=self.markers,
            ib_search=self.ib_search,
            ib_boundary=self.ib_boundary,
            mpm_external_force_n=mpm_external_force_n,
            mpm_particle_position_m=mpm_particle_position_m,
            mpm_particle_velocity_mps=mpm_particle_velocity_mps,
            mpm_particle_normal=mpm_particle_normal,
            mpm_particle_area_m2=mpm_particle_area_m2,
            mpm_particle_count=int(mpm_particle_count),
            solid_step=solid_step,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                self.marker_pressure_neumann_gradient_pa_per_m
            ),
            search_radius_m=float(search_radius_m),
            interior_probe_distance_m=float(interior_probe_distance_m),
            mpm_support_radius_m=float(mpm_support_radius_m),
            primary_region_id=int(primary_region_id),
            secondary_region_id=int(secondary_region_id),
            far_pressure_region_id=int(far_pressure_region_id),
            far_pressure_barrier_region_id=int(far_pressure_barrier_region_id),
            far_pressure_pa=float(far_pressure_pa),
            far_pressure_side_normal_sign=float(far_pressure_side_normal_sign),
            far_pressure_inside_probe_max_multiplier=float(
                far_pressure_inside_probe_max_multiplier
            ),
            two_sided_probe_max_multiplier=float(
                two_sided_probe_max_multiplier
            ),
            one_sided_pressure_region_id=int(one_sided_pressure_region_id),
            one_sided_reference_pressure_pa=float(one_sided_reference_pressure_pa),
            one_sided_probe_max_multiplier=float(one_sided_probe_max_multiplier),
            far_pressure_air_backed=bool(far_pressure_air_backed),
            far_pressure_air_backed_probe_normal_sign=float(
                far_pressure_air_backed_probe_normal_sign
            ),
            fluid_dt_s=fluid_dt_s,
            fluid_substeps=int(fluid_substeps),
            projection_iterations=int(projection_iterations),
            run_fluid_predictor=bool(run_fluid_predictor),
            fluid_advection_scheme=str(fluid_advection_scheme),
            pressure_neumann_density_kgm3=pressure_neumann_density_kgm3,
            pressure_neumann_dt_s=pressure_neumann_dt_s,
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            reset_pressure=bool(reset_pressure),
            pressure_solver=str(pressure_solver),
            pressure_solve_failure_policy=str(pressure_solve_failure_policy),
            multigrid_cycles=multigrid_cycles,
            cg_tolerance=float(cg_tolerance),
            cg_preconditioner=str(cg_preconditioner),
            surface_feedback_dt_s=surface_feedback_dt_s,
            divergence_cleanup_iterations=int(divergence_cleanup_iterations),
            divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
            classify_far_internal_nodes=bool(classify_far_internal_nodes),
            convert_internal_nodes_to_obstacles=bool(
                convert_internal_nodes_to_obstacles
            ),
            post_dirichlet_consistency_projection_iterations=int(
                post_dirichlet_consistency_projection_iterations
            ),
            diagnostic_disable_pressure_neumann_matrix_rows=bool(
                diagnostic_disable_pressure_neumann_matrix_rows
            ),
        )

    def advance_neo_hookean_step(
        self,
        *,
        fluid: Any,
        solid: Any,
        search_radius_m: float,
        interior_probe_distance_m: float,
        mpm_support_radius_m: float,
        solid_dt_s: float,
        mu_pa: float,
        lambda_pa: float,
        primary_region_id: int,
        secondary_region_id: int,
        far_pressure_region_id: int = -1,
        far_pressure_barrier_region_id: int = -1,
        far_pressure_pa: float = 0.0,
        far_pressure_side_normal_sign: float = 0.0,
        far_pressure_inside_probe_max_multiplier: float = 3.0,
        two_sided_probe_max_multiplier: float = 3.0,
        one_sided_pressure_region_id: int = -1,
        one_sided_reference_pressure_pa: float = 0.0,
        one_sided_probe_max_multiplier: float = 3.0,
        far_pressure_air_backed: bool = False,
        far_pressure_air_backed_probe_normal_sign: float = 0.0,
        fluid_dt_s: float | None = None,
        fluid_substeps: int = 1,
        projection_iterations: int = 40,
        run_fluid_predictor: bool = True,
        fluid_advection_scheme: str = "euler",
        pressure_neumann_density_kgm3: float | None = None,
        pressure_neumann_dt_s: float | None = None,
        pressure_outlet_zmin: bool = False,
        reset_pressure: bool = False,
        pressure_solver: str = "fv_cg",
        pressure_solve_failure_policy: str = "raise",
        multigrid_cycles: int | None = None,
        cg_tolerance: float = 1.0e-6,
        cg_preconditioner: str = "auto",
        divergence_cleanup_iterations: int = 0,
        divergence_cleanup_relaxation: float = 0.7,
        read_mpm_report: bool = True,
        solid_external_loads: Callable[[], None] | None = None,
        classify_far_internal_nodes: bool = False,
        convert_internal_nodes_to_obstacles: bool = True,
        diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
    ) -> HibmMpmSharpNeoHookeanStepReport:
        return advance_hibm_mpm_sharp_neo_hookean_step(
            fluid=fluid,
            markers=self.markers,
            ib_search=self.ib_search,
            ib_boundary=self.ib_boundary,
            solid=solid,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                self.marker_pressure_neumann_gradient_pa_per_m
            ),
            search_radius_m=float(search_radius_m),
            interior_probe_distance_m=float(interior_probe_distance_m),
            mpm_support_radius_m=float(mpm_support_radius_m),
            solid_dt_s=float(solid_dt_s),
            mu_pa=float(mu_pa),
            lambda_pa=float(lambda_pa),
            primary_region_id=int(primary_region_id),
            secondary_region_id=int(secondary_region_id),
            far_pressure_region_id=int(far_pressure_region_id),
            far_pressure_barrier_region_id=int(far_pressure_barrier_region_id),
            far_pressure_pa=float(far_pressure_pa),
            far_pressure_side_normal_sign=float(far_pressure_side_normal_sign),
            far_pressure_inside_probe_max_multiplier=float(
                far_pressure_inside_probe_max_multiplier
            ),
            two_sided_probe_max_multiplier=float(
                two_sided_probe_max_multiplier
            ),
            one_sided_pressure_region_id=int(one_sided_pressure_region_id),
            one_sided_reference_pressure_pa=float(one_sided_reference_pressure_pa),
            one_sided_probe_max_multiplier=float(one_sided_probe_max_multiplier),
            far_pressure_air_backed=bool(far_pressure_air_backed),
            far_pressure_air_backed_probe_normal_sign=float(
                far_pressure_air_backed_probe_normal_sign
            ),
            fluid_dt_s=fluid_dt_s,
            fluid_substeps=int(fluid_substeps),
            projection_iterations=int(projection_iterations),
            run_fluid_predictor=bool(run_fluid_predictor),
            fluid_advection_scheme=str(fluid_advection_scheme),
            pressure_neumann_density_kgm3=pressure_neumann_density_kgm3,
            pressure_neumann_dt_s=pressure_neumann_dt_s,
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            reset_pressure=bool(reset_pressure),
            pressure_solver=str(pressure_solver),
            pressure_solve_failure_policy=str(pressure_solve_failure_policy),
            multigrid_cycles=multigrid_cycles,
            cg_tolerance=float(cg_tolerance),
            cg_preconditioner=str(cg_preconditioner),
            divergence_cleanup_iterations=int(divergence_cleanup_iterations),
            divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
            read_mpm_report=bool(read_mpm_report),
            solid_external_loads=solid_external_loads,
            classify_far_internal_nodes=bool(classify_far_internal_nodes),
            convert_internal_nodes_to_obstacles=bool(
                convert_internal_nodes_to_obstacles
            ),
            diagnostic_disable_pressure_neumann_matrix_rows=bool(
                diagnostic_disable_pressure_neumann_matrix_rows
            ),
        )


def assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
    *,
    fluid: Any,
    markers: HibmMpmSurfaceMarkers,
    ib_search: HibmMpmIbNodeSearch,
    ib_boundary: HibmMpmIbBoundaryConditions,
    mpm_external_force_n,
    mpm_particle_position_m,
    mpm_particle_count: int,
    marker_pressure_neumann_gradient_pa_per_m_field,
    search_radius_m: float,
    interior_probe_distance_m: float,
    mpm_support_radius_m: float,
    primary_region_id: int = 0,
    secondary_region_id: int = 0,
    far_pressure_region_id: int = -1,
    far_pressure_barrier_region_id: int = -1,
    far_pressure_pa: float = 0.0,
    far_pressure_side_normal_sign: float = 0.0,
    far_pressure_inside_probe_max_multiplier: float = 3.0,
    two_sided_probe_max_multiplier: float = 3.0,
    one_sided_pressure_region_id: int = -1,
    one_sided_reference_pressure_pa: float = 0.0,
    one_sided_probe_max_multiplier: float = 3.0,
    far_pressure_air_backed: bool = False,
    far_pressure_air_backed_probe_normal_sign: float = 0.0,
    dt_s: float | None = None,
    fluid_substeps: int = 1,
    projection_iterations: int = 40,
    run_fluid_predictor: bool = True,
    fluid_advection_scheme: str = "euler",
    pressure_neumann_density_kgm3: float | None = None,
    pressure_neumann_dt_s: float | None = None,
    pressure_outlet_zmin: bool = False,
    reset_pressure: bool = False,
    pressure_solver: str = "fv_cg",
    pressure_solve_failure_policy: str = "raise",
    multigrid_cycles: int | None = None,
    cg_tolerance: float = 1.0e-6,
    cg_preconditioner: str = "auto",
    divergence_cleanup_iterations: int = 0,
    divergence_cleanup_relaxation: float = 0.7,
    classify_far_internal_nodes: bool = False,
    convert_internal_nodes_to_obstacles: bool = True,
    post_dirichlet_consistency_projection_iterations: int = 3,
    diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
    interpolate_velocity_dirichlet_with_interior: bool = True,
) -> HibmMpmSharpFluidToMpmLoadReport:
    """Run the sharp-interface fluid solve up to marker traction MPM loading.

    This is the generic HIBM-MPM coupling field path: it assembles no-slip and
    pressure Neumann boundary rows from surface markers, projects the fluid with
    those rows, samples full fluid stress to marker tractions, clears any stale
    MPM external force, and scatters only marker forces into MPM particles.
    """
    particles = int(mpm_particle_count)
    if particles <= 0:
        raise ValueError("mpm_particle_count must be positive")
    iterations = int(projection_iterations)
    if iterations <= 0:
        raise ValueError("projection_iterations must be positive")
    substeps = int(fluid_substeps)
    if substeps <= 0:
        raise ValueError("fluid_substeps must be positive")
    advection_scheme = str(fluid_advection_scheme)
    pressure_solve_failure_policy_name = str(pressure_solve_failure_policy)
    if pressure_solve_failure_policy_name not in {"raise", "report"}:
        raise ValueError("pressure_solve_failure_policy must be 'raise' or 'report'")
    fluid_substep_dt = None
    if dt_s is not None:
        fluid_substep_dt = float(dt_s) / float(substeps)
    else:
        fluid_dt = getattr(fluid, "dt", None)
        if fluid_dt is not None:
            fluid_substep_dt = float(fluid_dt) / float(substeps)
    pressure_neumann_density = None
    pressure_neumann_dt = None
    if pressure_neumann_density_kgm3 is not None:
        pressure_neumann_density = float(pressure_neumann_density_kgm3)
        if (
            not math.isfinite(pressure_neumann_density)
            or pressure_neumann_density <= 0.0
        ):
            raise ValueError("pressure_neumann_density_kgm3 must be finite and positive")
        if pressure_neumann_dt_s is None:
            raise ValueError(
                "pressure_neumann_dt_s must be provided when updating pressure "
                "Neumann from the fluid predictor"
            )
        pressure_neumann_dt = float(pressure_neumann_dt_s)
        if not math.isfinite(pressure_neumann_dt) or pressure_neumann_dt <= 0.0:
            raise ValueError("pressure_neumann_dt_s must be finite and positive")
        pressure_neumann_dt = pressure_neumann_dt / float(substeps)

    def assemble_velocity_dirichlet_rows() -> HibmMpmVelocityDirichletBoundaryReport:
        fluid.clear_velocity_dirichlet_boundary_rows()
        return ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            ib_search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            marker_region_id=markers.region_id,
            primary_region_id=primary_region_id,
            secondary_region_id=secondary_region_id,
            interpolate_interior_velocity=bool(
                interpolate_velocity_dirichlet_with_interior
            ),
        )

    def assemble_pressure_neumann_rows() -> HibmMpmPressureNeumannMatrixReport:
        fluid.clear_pressure_interface_matrix_terms()
        if diagnostic_disable_pressure_neumann_matrix_rows:
            return HibmMpmPressureNeumannMatrixReport(
                active_pressure_neumann_rows=0,
                rhs_integral=0.0,
                max_abs_rhs=0.0,
            )
        return ib_boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            ib_search,
            markers,
            pressure_coupling_extra_neighbor=(
                fluid.pressure_interface_coupling_extra_neighbor
            ),
            pressure_coupling_extra_coefficient=(
                fluid.pressure_interface_coupling_extra_coefficient
            ),
            pressure_interface_row_count=fluid.pressure_interface_row_count,
            pressure_interface_row_owner=fluid.pressure_interface_row_owner,
            pressure_interface_row_neighbor=fluid.pressure_interface_row_neighbor,
            pressure_interface_row_transmissibility=(
                fluid.pressure_interface_row_transmissibility
            ),
            pressure_interface_row_capacity=fluid.pressure_interface_row_capacity,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )

    def combine_projection_reports(
        projection_reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not projection_reports:
            return {
                "fluid_substeps": substeps,
                "fluid_advection_scheme": advection_scheme,
            }
        combined = dict(projection_reports[-1])
        combined["fluid_substeps"] = substeps
        combined["fluid_advection_scheme"] = advection_scheme
        sum_keys = (
            "cg_project_calls",
            "cg_iterations_total",
            "cg_host_residual_checks",
            "cg_mean_host_reads",
            "cg_mean_projection_count",
            "cg_breakdown_count",
            "hibm_post_dirichlet_consistency_projection_count",
            "hibm_projection_overflow_singleton_cleanup_cell_count",
            "hibm_projection_overflow_singleton_cleanup_component_count",
            "hibm_projection_tiny_unreached_cleanup_cell_count",
            "hibm_projection_tiny_unreached_cleanup_component_count",
        )
        for key in sum_keys:
            if any(key in report for report in projection_reports):
                combined[key] = sum(int(report.get(key, 0)) for report in projection_reports)
        max_keys = (
            "cg_iterations_max",
            "cg_initial_relative_residual_max",
            "cg_relative_residual_max",
            "hibm_unreached_component_rhs_mean_max_abs",
            "hibm_unreached_component_rhs_integral_max_abs",
        )
        for key in max_keys:
            if any(key in report for report in projection_reports):
                combined[key] = max(
                    float(report.get(key, 0.0)) for report in projection_reports
                )
        max_int_keys = (
            "hibm_unreached_incompatible_component_count",
            "cg_unreached_component_count",
            "cg_unreached_component_raw_count",
            "cg_unreached_component_largest_cell_count",
            "cg_unreached_component_singleton_count",
            "cg_unreached_component_small_count",
            "cg_unreached_component_small_cell_count",
        )
        for key in max_int_keys:
            if any(key in report for report in projection_reports):
                combined[key] = max(
                    int(report.get(key, 0)) for report in projection_reports
                )
        if any("cg_unreached_component_overflow" in report for report in projection_reports):
            combined["cg_unreached_component_overflow"] = any(
                bool(report.get("cg_unreached_component_overflow", False))
                for report in projection_reports
            )
        if any("cg_converged_all" in report for report in projection_reports):
            combined["cg_converged_all"] = all(
                bool(report.get("cg_converged_all", True))
                for report in projection_reports
            )
        if any("pressure_solve_failed" in report for report in projection_reports):
            failed = any(
                bool(report.get("pressure_solve_failed", False))
                for report in projection_reports
            )
            combined["pressure_solve_failed"] = failed
            if failed:
                actions = [
                    str(report.get("pressure_solve_failure_action", "reported"))
                    for report in projection_reports
                    if bool(report.get("pressure_solve_failed", False))
                ]
                combined["pressure_solve_failure_action"] = (
                    ",".join(dict.fromkeys(actions)) if actions else "reported"
                )
            else:
                combined["pressure_solve_failure_action"] = "none"
        if any(
            "pressure_projection_physical_failure" in report
            for report in projection_reports
        ):
            physical_failed = any(
                bool(report.get("pressure_projection_physical_failure", False))
                for report in projection_reports
            )
            combined["pressure_projection_physical_failure"] = physical_failed
            if physical_failed:
                combined["pressure_projection_physical_failure_reason"] = next(
                    str(
                        report.get(
                            "pressure_projection_physical_failure_reason",
                            "",
                        )
                    )
                    for report in projection_reports
                    if bool(report.get("pressure_projection_physical_failure", False))
                )
                combined["pressure_projection_physical_failure_action"] = "reported"
            else:
                combined["pressure_projection_physical_failure_reason"] = ""
                combined["pressure_projection_physical_failure_action"] = "none"
        if any(
            "hibm_post_dirichlet_consistency_projection_applied" in report
            for report in projection_reports
        ):
            combined["hibm_post_dirichlet_consistency_projection_applied"] = any(
                bool(
                    report.get(
                        "hibm_post_dirichlet_consistency_projection_applied",
                        False,
                    )
                )
                for report in projection_reports
            )
        breakdown = ""
        for report in projection_reports:
            if report.get("cg_breakdown"):
                breakdown = str(report["cg_breakdown"])
        if breakdown:
            combined["cg_breakdown"] = breakdown
        return combined

    _debug_stage_progress("search_and_classify_grid_fields:start")
    ib_report = ib_search.search_and_classify_grid_fields(
        markers,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        search_radius_m=float(search_radius_m),
        interior_probe_distance_m=float(interior_probe_distance_m),
        classify_far_internal_nodes=bool(classify_far_internal_nodes),
    )
    _debug_stage_progress("search_and_classify_grid_fields:done")
    _debug_stage_progress("apply_hibm_internal_obstacles:start")
    internal_obstacle_cell_count = fluid.apply_hibm_internal_obstacles(
        ib_search.node_kind_code,
        internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
        convert_internal_nodes=bool(convert_internal_nodes_to_obstacles),
    )
    _debug_stage_progress("apply_hibm_internal_obstacles:done")
    _debug_stage_progress("build_boundary_conditions:start")
    boundary_report = ib_boundary.build_from_search_device_fields(
        ib_search,
        markers,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            marker_pressure_neumann_gradient_pa_per_m_field
        ),
    )
    _debug_stage_progress("build_boundary_conditions:done")
    _debug_stage_progress("assemble_velocity_dirichlet_rows:start")
    velocity_report = assemble_velocity_dirichlet_rows()
    _debug_stage_progress("assemble_velocity_dirichlet_rows:done")
    solid_band_nonprojectable_cell_count = 0
    _debug_stage_progress("solid_band_fixed_point:start")
    for _band_pass in range(8):
        band_increment = fluid.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            node_kind_code=ib_search.node_kind_code,
            unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE,
            protect_velocity_dirichlet_radius_cells=0,
            protect_unstamped_velocity_dirichlet_rows=True,
            protect_solid_band_mask=True,
        )
        if int(band_increment) <= 0:
            break
        solid_band_nonprojectable_cell_count += int(band_increment)
        velocity_report = assemble_velocity_dirichlet_rows()
    _debug_stage_progress("solid_band_fixed_point:done")
    # Final-sweep band populations (S2-A8'): in interior-only mode the
    # sliver count saturates to zero while the enclosed-water count is the
    # surviving sealed real-water population; -1 means the sweep ran
    # without a split (default mode).
    solid_band_interior_cell_count = int(
        getattr(fluid, "last_hibm_solid_band_interior_cells", -1)
    )
    solid_band_enclosed_water_cell_count = int(
        getattr(fluid, "last_hibm_solid_band_enclosed_water_cells", -1)
    )
    solid_band_velocity_dirichlet_protected_cell_count = int(
        getattr(
            fluid,
            "last_hibm_solid_band_velocity_dirichlet_protected_cells",
            -1,
        )
    )
    solid_band_mask_protected_cell_count = int(
        getattr(fluid, "last_hibm_solid_band_mask_protected_cells", -1)
    )
    hibm_air_backed_reachability_barrier_cell_count = -1
    use_air_backed_reachability_barrier = (
        bool(far_pressure_air_backed) and int(far_pressure_region_id) != -1
    )
    if use_air_backed_reachability_barrier:
        _debug_stage_progress("write_region_pressure_reachability_barrier:start")
        hibm_air_backed_reachability_barrier_cell_count = (
            markers.write_region_pressure_reachability_barrier(
                fluid.hibm_pressure_reachability_barrier,
                ib_search.node_kind_code,
                ib_search.nearest_marker,
                barrier_node_code=HibmMpmIbNodeSearch._NODE_EXTERNAL_IB,
                barrier_region_id=int(far_pressure_region_id),
                secondary_barrier_region_id=int(far_pressure_barrier_region_id),
                tertiary_barrier_region_id=int(one_sided_pressure_region_id),
                include_all_classified_region_nodes=True,
            )
        )
        _debug_stage_progress("write_region_pressure_reachability_barrier:done")
    _debug_stage_progress("mark_pressure_outlet_disconnected:start")
    pressure_disconnected_nonprojectable_cell_count = (
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            use_existing_reachability_barrier=use_air_backed_reachability_barrier,
        )
    )
    _debug_stage_progress("mark_pressure_outlet_disconnected:done")
    hibm_air_backed_cell_count = -1
    hibm_air_backed_component_count = -1
    hibm_air_backed_cell_volume_m3 = -1.0
    hibm_air_backed_seed_marker_count = -1
    hibm_air_backed_seed_missed_marker_count = -1
    hibm_air_backed_seed_fallback_cell_count = -1
    if bool(far_pressure_air_backed) and int(far_pressure_region_id) != -1:
        # S2-A12: the declared air-backed closure region gets a fluid-side
        # air zone. The flood + per-component labels above are this step's
        # classification input: closure markers walk the configured normal
        # side and select the unreached component(s) they land in. Selected
        # components convert to obstacle-like air cells, so dry-side volume
        # does not participate in the incompressible water solve. Stateless
        # per step; outlet-reachable water is structurally unselectable (air
        # is a subset of the unreached set).
        (
            hibm_air_backed_seed_marker_count,
            hibm_air_backed_seed_missed_marker_count,
        ) = markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            far_pressure_region_id=int(far_pressure_region_id),
            far_pressure_inside_probe_max_multiplier=float(
                far_pressure_inside_probe_max_multiplier
            ),
            far_pressure_air_backed_probe_normal_sign=float(
                far_pressure_air_backed_probe_normal_sign
            ),
            fallback_to_bidirectional_if_all_missed=True,
            fallback_to_region_adjacency_if_all_missed=True,
            node_kind_code=ib_search.node_kind_code,
            nearest_marker=ib_search.nearest_marker,
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
        )
        hibm_air_backed_seed_fallback_cell_count = int(
            markers.report_air_backed_seed_fallback_cell_count[None]
        )
        _debug_stage_progress("convert_hibm_air_backed_cells:start")
        hibm_air_backed_cell_count = fluid.convert_hibm_air_backed_cells()
        _debug_stage_progress("convert_hibm_air_backed_cells:done")
        hibm_air_backed_component_count = int(
            fluid.last_hibm_air_backed_component_count
        )
        hibm_air_backed_cell_volume_m3 = float(
            fluid.last_hibm_air_backed_cell_volume_m3
        )
        if int(hibm_air_backed_cell_count) > 0:
            _debug_stage_progress("air_backed_post_convert_band:start")
            # Conversion can orphan relocated rows that owned ex-pocket
            # cells and, through row-owned faces, leave new
            # zero-correctable candidates: re-assemble rows and rerun the
            # band fixed point so no all-blocked active row reaches the CG
            # (the S2-A8' zero-row lesson). Monotone like the original
            # loop - conversion only adds obstacle.
            velocity_report = assemble_velocity_dirichlet_rows()
            for _air_band_pass in range(8):
                band_increment = fluid.mark_hibm_solid_band_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                    node_kind_code=ib_search.node_kind_code,
                    unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE,
                    protect_velocity_dirichlet_radius_cells=0,
                    protect_unstamped_velocity_dirichlet_rows=True,
                    protect_solid_band_mask=True,
                )
                if int(band_increment) <= 0:
                    break
                solid_band_nonprojectable_cell_count += int(band_increment)
                velocity_report = assemble_velocity_dirichlet_rows()
            _debug_stage_progress("air_backed_post_convert_band:done")
            solid_band_interior_cell_count = int(
                getattr(fluid, "last_hibm_solid_band_interior_cells", -1)
            )
            solid_band_enclosed_water_cell_count = int(
                getattr(fluid, "last_hibm_solid_band_enclosed_water_cells", -1)
            )
            solid_band_velocity_dirichlet_protected_cell_count = int(
                getattr(
                    fluid,
                    "last_hibm_solid_band_velocity_dirichlet_protected_cells",
                    -1,
                )
            )
            solid_band_mask_protected_cell_count = int(
                getattr(fluid, "last_hibm_solid_band_mask_protected_cells", -1)
            )
    pressure_gradient_report = None
    pressure_report = HibmMpmPressureNeumannMatrixReport(
        active_pressure_neumann_rows=0,
        rhs_integral=0.0,
        max_abs_rhs=0.0,
    )
    projection_reports: list[dict[str, Any]] = []
    row_cloud_orphan_cell_count = 0
    row_cloud_orphan_component_count = 0
    overflow_singleton_cleanup_cell_count = 0
    overflow_singleton_cleanup_component_count = 0
    projection_tiny_unreached_cleanup_cell_count = 0
    projection_tiny_unreached_cleanup_component_count = 0

    def convert_row_cloud_orphans_until_saturated() -> None:
        nonlocal pressure_disconnected_nonprojectable_cell_count
        nonlocal row_cloud_orphan_cell_count
        nonlocal row_cloud_orphan_component_count
        nonlocal velocity_report
        for _row_cloud_orphan_pass in range(8):
            converted_count = fluid.convert_hibm_row_cloud_orphan_components(
                max_component_cells=(
                    HIBM_PRESSURE_DISCONNECTED_SMALL_COMPONENT_THRESHOLD_CELLS
                ),
            )
            if int(converted_count) <= 0:
                break
            row_cloud_orphan_cell_count += int(converted_count)
            row_cloud_orphan_component_count += int(
                getattr(fluid, "last_hibm_row_cloud_orphan_component_count", 0)
            )
            velocity_report = assemble_velocity_dirichlet_rows()
            pressure_disconnected_nonprojectable_cell_count = (
                fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                )
            )

    def convert_overflow_singletons_without_row_reload() -> bool:
        nonlocal pressure_disconnected_nonprojectable_cell_count
        nonlocal overflow_singleton_cleanup_cell_count
        nonlocal overflow_singleton_cleanup_component_count
        converted_count = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=1,
            overflow_singletons_only=True,
            protect_velocity_dirichlet_radius_cells=(
                HIBM_OVERFLOW_SINGLETON_NO_SLIP_PROTECTION_RADIUS_CELLS
            ),
        )
        if int(converted_count) <= 0:
            return False
        overflow_singleton_cleanup_cell_count += int(converted_count)
        overflow_singleton_cleanup_component_count += int(
            getattr(fluid, "last_hibm_row_cloud_orphan_component_count", 0)
        )
        pressure_disconnected_nonprojectable_cell_count = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
        )
        return True

    def convert_projection_topology_cleanup_until_saturated() -> None:
        nonlocal pressure_disconnected_nonprojectable_cell_count
        nonlocal velocity_report
        nonlocal projection_tiny_unreached_cleanup_cell_count
        nonlocal projection_tiny_unreached_cleanup_component_count
        convert_row_cloud_orphans_until_saturated()
        tiny_unreached_cleanup_threshold = max(
            0,
            HIBM_TINY_UNREACHED_COMPONENT_CLEANUP_THRESHOLD_CELLS,
        )
        for _projection_cleanup_pass in range(8):
            mutated = convert_overflow_singletons_without_row_reload()
            if tiny_unreached_cleanup_threshold > 0:
                for _tiny_unreached_cleanup_pass in range(8):
                    converted_tiny_unreached = (
                        fluid.convert_hibm_row_cloud_orphan_components(
                            max_component_cells=tiny_unreached_cleanup_threshold,
                            convert_unstamped_small_components=True,
                            protect_velocity_dirichlet_radius_cells=0,
                            protect_solid_band_mask=False,
                        )
                    )
                    if int(converted_tiny_unreached) <= 0:
                        break
                    mutated = True
                    projection_tiny_unreached_cleanup_cell_count += int(
                        converted_tiny_unreached
                    )
                    projection_tiny_unreached_cleanup_component_count += int(
                        getattr(fluid, "last_hibm_row_cloud_orphan_component_count", 0)
                    )
                    pressure_disconnected_nonprojectable_cell_count = (
                        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                            pressure_outlet_zmin=bool(pressure_outlet_zmin),
                        )
                    )
            if not mutated:
                break
            velocity_report = assemble_velocity_dirichlet_rows()
            pressure_disconnected_nonprojectable_cell_count = (
                fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                )
            )
            convert_row_cloud_orphans_until_saturated()

    _debug_stage_progress("fluid_substeps:start")
    for _ in range(substeps):
        velocity_report = assemble_velocity_dirichlet_rows()
        pressure_disconnected_nonprojectable_cell_count = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
        )
        convert_row_cloud_orphans_until_saturated()
        if bool(run_fluid_predictor):
            fluid.predict(
                dt_s=fluid_substep_dt,
                advection_scheme=advection_scheme,
            )
        if pressure_neumann_density is not None and pressure_neumann_dt is not None:
            markers.update_pressure_neumann_gradient_from_fluid_predictor(
                marker_pressure_neumann_gradient_pa_per_m_field,
                velocity_field=fluid.velocity,
                obstacle_field=fluid.obstacle,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
                density_kgm3=pressure_neumann_density,
                dt_s=pressure_neumann_dt,
                probe_distance_m=float(interior_probe_distance_m),
            )
            boundary_report = ib_boundary.build_from_search_device_fields(
                ib_search,
                markers,
                marker_pressure_neumann_gradient_pa_per_m_field=(
                    marker_pressure_neumann_gradient_pa_per_m_field
                ),
            )
            velocity_report = assemble_velocity_dirichlet_rows()
            pressure_disconnected_nonprojectable_cell_count = (
                fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                )
            )
            pressure_gradient_report = (
                ib_boundary.update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
                    velocity_field=fluid.velocity,
                    obstacle_field=fluid.obstacle,
                    search=ib_search,
                    cell_face_x_m=fluid.cell_face_x_m,
                    cell_face_y_m=fluid.cell_face_y_m,
                    cell_face_z_m=fluid.cell_face_z_m,
                    cell_center_x_m=fluid.cell_center_x_m,
                    cell_center_y_m=fluid.cell_center_y_m,
                    cell_center_z_m=fluid.cell_center_z_m,
                    grid_nodes=fluid.grid.grid_nodes,
                    density_kgm3=pressure_neumann_density,
                    dt_s=pressure_neumann_dt,
                )
            )
            convert_row_cloud_orphans_until_saturated()

        projection_overflow_cell_count_before = overflow_singleton_cleanup_cell_count
        projection_overflow_component_count_before = (
            overflow_singleton_cleanup_component_count
        )
        projection_tiny_cell_count_before = (
            projection_tiny_unreached_cleanup_cell_count
        )
        projection_tiny_component_count_before = (
            projection_tiny_unreached_cleanup_component_count
        )
        convert_projection_topology_cleanup_until_saturated()
        pressure_report = assemble_pressure_neumann_rows()
        requested_pressure_solver = str(pressure_solver)
        effective_pressure_solver = requested_pressure_solver
        pressure_solver_forced_to_fv_cg = False
        pressure_solver_force_reason = ""
        if (
            int(pressure_report.active_pressure_neumann_rows) > 0
            and effective_pressure_solver in {"jacobi", "compact_jacobi", "fv_multigrid"}
        ):
            effective_pressure_solver = "fv_cg"
            pressure_solver_forced_to_fv_cg = True
            pressure_solver_force_reason = "hibm_pressure_neumann_requires_fv_solver"
        project_report = dict(
            fluid.project(
                iterations=iterations,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                dt_s=fluid_substep_dt,
                preserve_velocity_constraints=False,
                reset_pressure=bool(reset_pressure),
                pressure_solver=effective_pressure_solver,
                multigrid_cycles=multigrid_cycles,
                cg_tolerance=float(cg_tolerance),
                cg_preconditioner=str(cg_preconditioner),
                pressure_solve_failure_policy=pressure_solve_failure_policy_name,
                hibm_tiny_unreached_cleanup_component_cells=0,
                divergence_cleanup_iterations=int(divergence_cleanup_iterations),
                divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
                read_report=True,
            )
        )
        project_report["hibm_projection_overflow_singleton_cleanup_cell_count"] = (
            int(project_report.get("hibm_projection_overflow_singleton_cleanup_cell_count", 0))
            + int(overflow_singleton_cleanup_cell_count)
            - int(projection_overflow_cell_count_before)
        )
        project_report["hibm_projection_overflow_singleton_cleanup_component_count"] = (
            int(
                project_report.get(
                    "hibm_projection_overflow_singleton_cleanup_component_count",
                    0,
                )
            )
            + int(overflow_singleton_cleanup_component_count)
            - int(projection_overflow_component_count_before)
        )
        project_report["hibm_projection_tiny_unreached_cleanup_cell_count"] = (
            int(project_report.get("hibm_projection_tiny_unreached_cleanup_cell_count", 0))
            + int(projection_tiny_unreached_cleanup_cell_count)
            - int(projection_tiny_cell_count_before)
        )
        project_report["hibm_projection_tiny_unreached_cleanup_component_count"] = (
            int(
                project_report.get(
                    "hibm_projection_tiny_unreached_cleanup_component_count",
                    0,
                )
            )
            + int(projection_tiny_unreached_cleanup_component_count)
            - int(projection_tiny_component_count_before)
        )
        project_report.update(
            {
                "pressure_solver_requested": requested_pressure_solver,
                "pressure_solver": effective_pressure_solver,
                "pressure_solver_forced_to_fv_cg": pressure_solver_forced_to_fv_cg,
                "pressure_solver_force_reason": pressure_solver_force_reason,
                "pressure_interface_neumann_active_rows": int(
                    pressure_report.active_pressure_neumann_rows
                ),
            }
        )
        projection_reports.append(project_report)
    _debug_stage_progress("fluid_substeps:done")
    projection_report = combine_projection_reports(projection_reports)
    _debug_stage_progress("post_substep_velocity_rows_and_reachability:start")
    consistency_projection_iterations = max(
        0,
        int(post_dirichlet_consistency_projection_iterations),
    )
    for consistency_projection_index in range(consistency_projection_iterations):
        velocity_report = assemble_velocity_dirichlet_rows()
        pressure_disconnected_nonprojectable_cell_count = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
        )
        projection_overflow_cell_count_before = overflow_singleton_cleanup_cell_count
        projection_overflow_component_count_before = (
            overflow_singleton_cleanup_component_count
        )
        projection_tiny_cell_count_before = (
            projection_tiny_unreached_cleanup_cell_count
        )
        projection_tiny_component_count_before = (
            projection_tiny_unreached_cleanup_component_count
        )
        convert_projection_topology_cleanup_until_saturated()
        if int(velocity_report.active_velocity_dirichlet_rows) <= 0:
            break
        pressure_report = assemble_pressure_neumann_rows()
        requested_pressure_solver = str(pressure_solver)
        effective_pressure_solver = requested_pressure_solver
        pressure_solver_forced_to_fv_cg = False
        pressure_solver_force_reason = ""
        if (
            int(pressure_report.active_pressure_neumann_rows) > 0
            and effective_pressure_solver in {"jacobi", "compact_jacobi", "fv_multigrid"}
        ):
            effective_pressure_solver = "fv_cg"
            pressure_solver_forced_to_fv_cg = True
            pressure_solver_force_reason = "hibm_pressure_neumann_requires_fv_solver"
        consistency_project_report = dict(
            fluid.project(
                iterations=iterations,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                dt_s=fluid_substep_dt,
                preserve_velocity_constraints=False,
                reset_pressure=bool(reset_pressure),
                pressure_solver=effective_pressure_solver,
                multigrid_cycles=multigrid_cycles,
                cg_tolerance=float(cg_tolerance),
                cg_preconditioner=str(cg_preconditioner),
                pressure_solve_failure_policy=pressure_solve_failure_policy_name,
                hibm_tiny_unreached_cleanup_component_cells=0,
                divergence_cleanup_iterations=int(divergence_cleanup_iterations),
                divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
                read_report=True,
            )
        )
        consistency_project_report[
            "hibm_projection_overflow_singleton_cleanup_cell_count"
        ] = (
            int(
                consistency_project_report.get(
                    "hibm_projection_overflow_singleton_cleanup_cell_count",
                    0,
                )
            )
            + int(overflow_singleton_cleanup_cell_count)
            - int(projection_overflow_cell_count_before)
        )
        consistency_project_report[
            "hibm_projection_overflow_singleton_cleanup_component_count"
        ] = (
            int(
                consistency_project_report.get(
                    "hibm_projection_overflow_singleton_cleanup_component_count",
                    0,
                )
            )
            + int(overflow_singleton_cleanup_component_count)
            - int(projection_overflow_component_count_before)
        )
        consistency_project_report[
            "hibm_projection_tiny_unreached_cleanup_cell_count"
        ] = (
            int(
                consistency_project_report.get(
                    "hibm_projection_tiny_unreached_cleanup_cell_count",
                    0,
                )
            )
            + int(projection_tiny_unreached_cleanup_cell_count)
            - int(projection_tiny_cell_count_before)
        )
        consistency_project_report[
            "hibm_projection_tiny_unreached_cleanup_component_count"
        ] = (
            int(
                consistency_project_report.get(
                    "hibm_projection_tiny_unreached_cleanup_component_count",
                    0,
                )
            )
            + int(projection_tiny_unreached_cleanup_component_count)
            - int(projection_tiny_component_count_before)
        )
        consistency_project_report.update(
            {
                "pressure_solver_requested": requested_pressure_solver,
                "pressure_solver": effective_pressure_solver,
                "pressure_solver_forced_to_fv_cg": pressure_solver_forced_to_fv_cg,
                "pressure_solver_force_reason": pressure_solver_force_reason,
                "pressure_interface_neumann_active_rows": int(
                    pressure_report.active_pressure_neumann_rows
                ),
                "hibm_projection_stage": "post_dirichlet_reconstruction_consistency",
                "hibm_post_dirichlet_consistency_projection_index": (
                    int(consistency_projection_index) + 1
                ),
                "hibm_post_dirichlet_consistency_projection_applied": True,
                "hibm_post_dirichlet_consistency_projection_count": 1,
            }
        )
        projection_reports.append(consistency_project_report)
        projection_report = combine_projection_reports(projection_reports)
    _debug_stage_progress("post_substep_velocity_rows_and_reachability:done")
    _debug_stage_progress("final_divergence_and_partition_stats:start")
    fluid.compute_divergence(pressure_outlet_zmin=bool(pressure_outlet_zmin))
    (
        final_raw_stats,
        final_stats,
        final_interior_raw_stats,
        final_interior_stats,
        velocity_dirichlet_near_raw_stats,
        velocity_dirichlet_near_stats,
        velocity_dirichlet_far_raw_stats,
        velocity_dirichlet_far_stats,
        pressure_correctable_raw_stats,
        pressure_correctable_stats,
        pressure_fixed_raw_stats,
        pressure_fixed_stats,
        interior_pressure_correctable_raw_stats,
        interior_pressure_correctable_stats,
        interior_pressure_fixed_raw_stats,
        interior_pressure_fixed_stats,
    ) = fluid.final_and_dirichlet_partition_report_stats(
        pressure_outlet_zmin=bool(pressure_outlet_zmin),
    )
    projection_report = {
        **projection_report,
        "l2": final_stats["l2"],
        "max_abs": final_stats["max_abs"],
        "raw_l2": final_raw_stats["l2"],
        "raw_max_abs": final_raw_stats["max_abs"],
        "interior_l2": final_interior_stats["l2"],
        "interior_max_abs": final_interior_stats["max_abs"],
        "interior_raw_l2": final_interior_raw_stats["l2"],
        "interior_raw_max_abs": final_interior_raw_stats["max_abs"],
        "unreached_l2": float(fluid.last_unreached_divergence_stats["l2"]),
        "unreached_max_abs": float(
            fluid.last_unreached_divergence_stats["max_abs"]
        ),
        "unreached_cell_count": int(
            fluid.last_unreached_divergence_stats["count"]
        ),
        "post_constraint_l2": final_stats["l2"],
        "post_constraint_max_abs": final_stats["max_abs"],
        "post_constraint_raw_l2": final_raw_stats["l2"],
        "post_constraint_raw_max_abs": final_raw_stats["max_abs"],
        "velocity_dirichlet_near_l2": velocity_dirichlet_near_stats["l2"],
        "velocity_dirichlet_near_max_abs": velocity_dirichlet_near_stats["max_abs"],
        "velocity_dirichlet_near_raw_l2": velocity_dirichlet_near_raw_stats["l2"],
        "velocity_dirichlet_near_raw_max_abs": (
            velocity_dirichlet_near_raw_stats["max_abs"]
        ),
        "velocity_dirichlet_far_l2": velocity_dirichlet_far_stats["l2"],
        "velocity_dirichlet_far_max_abs": velocity_dirichlet_far_stats["max_abs"],
        "velocity_dirichlet_far_raw_l2": velocity_dirichlet_far_raw_stats["l2"],
        "velocity_dirichlet_far_raw_max_abs": (
            velocity_dirichlet_far_raw_stats["max_abs"]
        ),
        "pressure_correctable_l2": pressure_correctable_stats["l2"],
        "pressure_correctable_max_abs": pressure_correctable_stats["max_abs"],
        "pressure_correctable_cell_count": pressure_correctable_stats["count"],
        "pressure_correctable_raw_l2": pressure_correctable_raw_stats["l2"],
        "pressure_correctable_raw_max_abs": (
            pressure_correctable_raw_stats["max_abs"]
        ),
        "pressure_fixed_l2": pressure_fixed_stats["l2"],
        "pressure_fixed_max_abs": pressure_fixed_stats["max_abs"],
        "pressure_fixed_cell_count": pressure_fixed_stats["count"],
        "pressure_fixed_raw_l2": pressure_fixed_raw_stats["l2"],
        "pressure_fixed_raw_max_abs": pressure_fixed_raw_stats["max_abs"],
        "interior_pressure_correctable_l2": (
            interior_pressure_correctable_stats["l2"]
        ),
        "interior_pressure_correctable_max_abs": (
            interior_pressure_correctable_stats["max_abs"]
        ),
        "interior_pressure_correctable_cell_count": (
            interior_pressure_correctable_stats["count"]
        ),
        "interior_pressure_correctable_raw_l2": (
            interior_pressure_correctable_raw_stats["l2"]
        ),
        "interior_pressure_correctable_raw_max_abs": (
            interior_pressure_correctable_raw_stats["max_abs"]
        ),
        "interior_pressure_fixed_l2": interior_pressure_fixed_stats["l2"],
        "interior_pressure_fixed_max_abs": (
            interior_pressure_fixed_stats["max_abs"]
        ),
        "interior_pressure_fixed_cell_count": interior_pressure_fixed_stats["count"],
        "interior_pressure_fixed_raw_l2": interior_pressure_fixed_raw_stats["l2"],
        "interior_pressure_fixed_raw_max_abs": (
            interior_pressure_fixed_raw_stats["max_abs"]
        ),
    }
    _debug_stage_progress("final_divergence_and_partition_stats:done")
    if bool(projection_report.get("pressure_solve_failed", False)):
        failure_action = str(
            projection_report.get("pressure_solve_failure_action", "reported")
        )
        residual = float(projection_report.get("cg_relative_residual_max", math.nan))
        raise RuntimeError(
            "HIBM-MPM pressure solve failed before stress sampling; refusing to "
            "sample marker tractions or scatter MPM forces from an invalid pressure "
            f"field (action={failure_action}, "
            f"cg_relative_residual_max={residual:.6g})"
        )
    _debug_stage_progress("sample_no_slip_residual:start")
    no_slip_sampling_obstacle = fluid.build_hibm_no_slip_sampling_obstacle()
    no_slip_report = markers.sample_no_slip_residual(
        fluid.velocity,
        no_slip_sampling_obstacle,
        fluid.cell_face_x_m,
        fluid.cell_face_y_m,
        fluid.cell_face_z_m,
        fluid.cell_center_x_m,
        fluid.cell_center_y_m,
        fluid.cell_center_z_m,
        fluid.grid.grid_nodes,
        primary_region_id=int(primary_region_id),
        secondary_region_id=int(secondary_region_id),
    )
    _debug_stage_progress("sample_no_slip_residual:done")
    # S2-A8'' sampling preparation, strictly after the LAST fluid.project(...)
    # (including the post-Dirichlet consistency projection) and strictly before
    # the stress sampling. Pressure fill is a closure-only operation: it gives
    # declared far-pressure pockets a readable p_far value. Ordinary two-sided
    # thin-wall sampling must walk through projected solid-band obstacles to the
    # next real solved fluid cell; otherwise it can stop on back-filled interior
    # cells and artificially erase the pressure jump across the flap.
    stress_sampling_obstacle_field = None
    if int(far_pressure_region_id) != -1:
        _debug_stage_progress("sampling_obstacle_and_pressure_fill:start")
        fluid.fill_hibm_converted_cell_pressures()
        if bool(far_pressure_air_backed) and int(hibm_air_backed_cell_count) > 0:
            # S2-A12: stamp the declared chamber pressure into air cells
            # strictly AFTER the converted-cell fill and strictly BEFORE
            # the stress sampling, so the dedicated sampling view reads
            # exactly p_far in the pocket.
            fluid.write_hibm_air_backed_cell_pressures(float(far_pressure_pa))
        fluid.build_hibm_sampling_obstacle(
            ib_search.node_kind_code,
            unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE,
        )
        stress_sampling_obstacle_field = fluid.sampling_obstacle
        _debug_stage_progress("sampling_obstacle_and_pressure_fill:done")
    elif bool(convert_internal_nodes_to_obstacles):
        _debug_stage_progress("plain_two_sided_sampling_uses_projected_obstacle")
    _debug_stage_progress("sample_fluid_stress_to_marker_tractions:start")
    stress_report = markers.sample_fluid_stress_to_marker_tractions(
        fluid.velocity,
        fluid.pressure,
        fluid.obstacle,
        fluid.cell_face_x_m,
        fluid.cell_face_y_m,
        fluid.cell_face_z_m,
        fluid.cell_center_x_m,
        fluid.cell_center_y_m,
        fluid.cell_center_z_m,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        fluid.grid.grid_nodes,
        viscosity_pa_s=fluid.mu,
        two_sided_pressure=True,
        # S2-A8'': None only when no HIBM internal projection conversion is
        # active; converted thin-wall cases use the dedicated sampling view.
        sampling_obstacle_field=stress_sampling_obstacle_field,
        far_pressure_region_id=int(far_pressure_region_id),
        far_pressure_pa=float(far_pressure_pa),
        far_pressure_side_normal_sign=float(far_pressure_side_normal_sign),
        far_pressure_inside_probe_max_multiplier=float(
            far_pressure_inside_probe_max_multiplier
        ),
        two_sided_probe_max_multiplier=float(
            two_sided_probe_max_multiplier
        ),
        one_sided_pressure_region_id=int(one_sided_pressure_region_id),
        one_sided_reference_pressure_pa=float(one_sided_reference_pressure_pa),
        one_sided_probe_max_multiplier=float(one_sided_probe_max_multiplier),
        # S2-A6: the anchor fallback rides the closure opt-in. It is only
        # armed when this very call also assembled the pressure-Neumann
        # rows (anchors are captured there, strictly before this sampling
        # and before the projection that solved fluid.pressure); with the
        # diagnostic row disable the anchors would be stale, so keep the
        # fallback off.
        use_pressure_anchor_fallback=(
            int(far_pressure_region_id) != -1
            and not bool(diagnostic_disable_pressure_neumann_matrix_rows)
        ),
        # S2-A7: node-level anchors ride the same switch. They were
        # captured by the LAST velocity-Dirichlet assembly above (reset +
        # prefill + row capture), strictly before the projection(s) that
        # solved fluid.pressure and before this sampling, so every set
        # anchor points at a solved, non-obstacle cell.
        node_anchor_cell=ib_search.node_anchor_cell,
    )
    _debug_stage_progress("sample_fluid_stress_to_marker_tractions:done")
    _debug_stage_progress("marker_force_scatter:start")
    markers.compute_marker_forces()
    marker_force_report = markers.aggregate_region_forces(
        primary_region_id=int(primary_region_id),
        secondary_region_id=int(secondary_region_id),
    )
    clear_report = markers.clear_mpm_external_forces(
        mpm_external_force_n,
        particle_count=particles,
    )
    scatter_report = markers.scatter_marker_forces_to_mpm_particles(
        mpm_external_force_n,
        mpm_particle_position_m,
        particle_count=particles,
        support_radius_m=float(mpm_support_radius_m),
    )
    _debug_stage_progress("marker_force_scatter:done")
    pressure_neumann_invalid_diagnostic_rows = tuple(
        ib_boundary.pressure_neumann_invalid_diagnostic_rows(
            search=ib_search,
            markers=markers,
            fluid=fluid,
        )
    )
    pressure_disconnected_region = hibm_mpm_pressure_disconnected_region_report(
        fluid,
        primary_region_id=primary_region_id,
        secondary_region_id=secondary_region_id,
    )
    return HibmMpmSharpFluidToMpmLoadReport(
        ib_node_search=ib_report,
        internal_obstacle_cell_count=internal_obstacle_cell_count,
        solid_band_nonprojectable_cell_count=solid_band_nonprojectable_cell_count,
        solid_band_interior_cell_count=solid_band_interior_cell_count,
        solid_band_enclosed_water_cell_count=solid_band_enclosed_water_cell_count,
        solid_band_velocity_dirichlet_protected_cell_count=(
            solid_band_velocity_dirichlet_protected_cell_count
        ),
        solid_band_mask_protected_cell_count=solid_band_mask_protected_cell_count,
        row_cloud_orphan_cell_count=int(row_cloud_orphan_cell_count),
        row_cloud_orphan_component_count=int(row_cloud_orphan_component_count),
        overflow_singleton_cleanup_cell_count=int(
            overflow_singleton_cleanup_cell_count
        ),
        overflow_singleton_cleanup_component_count=int(
            overflow_singleton_cleanup_component_count
        ),
        pressure_disconnected_nonprojectable_cell_count=(
            pressure_disconnected_nonprojectable_cell_count
        ),
        pressure_disconnected_region=pressure_disconnected_region,
        air_backed_cell_count=int(hibm_air_backed_cell_count),
        air_backed_component_count=int(hibm_air_backed_component_count),
        air_backed_cell_volume_m3=float(hibm_air_backed_cell_volume_m3),
        air_backed_seed_marker_count=int(hibm_air_backed_seed_marker_count),
        air_backed_seed_missed_marker_count=int(
            hibm_air_backed_seed_missed_marker_count
        ),
        air_backed_seed_fallback_cell_count=int(
            hibm_air_backed_seed_fallback_cell_count
        ),
        air_backed_reachability_barrier_cell_count=int(
            hibm_air_backed_reachability_barrier_cell_count
        ),
        boundary_conditions=boundary_report,
        pressure_neumann_gradient=pressure_gradient_report,
        velocity_dirichlet=velocity_report,
        pressure_neumann=pressure_report,
        fluid_predictor_applied=bool(run_fluid_predictor),
        fluid_projection=projection_report,
        no_slip_residual=no_slip_report,
        fluid_stress=stress_report,
        marker_forces=marker_force_report,
        mpm_external_force_clear=clear_report,
        mpm_force_scatter=scatter_report,
        pressure_neumann_invalid_diagnostic_rows=(
            pressure_neumann_invalid_diagnostic_rows
        ),
    )


def hibm_mpm_external_force_fresh_for_solid_step(
    load_report: HibmMpmSharpFluidToMpmLoadReport,
) -> bool:
    clear = load_report.mpm_external_force_clear
    scatter = load_report.mpm_force_scatter
    force_values = (
        *scatter.total_marker_force_n,
        *scatter.total_mpm_external_force_n,
        scatter.action_reaction_residual_n,
        clear.max_abs_external_force_before_n,
    )
    return (
        clear.cleared_particle_count > 0
        and scatter.active_marker_count > 0
        and scatter.active_particle_count > 0
        and all(math.isfinite(float(value)) for value in force_values)
    )


def advance_hibm_mpm_sharp_mpm_step(
    *,
    fluid: Any,
    markers: HibmMpmSurfaceMarkers,
    ib_search: HibmMpmIbNodeSearch,
    ib_boundary: HibmMpmIbBoundaryConditions,
    mpm_external_force_n,
    mpm_particle_position_m,
    mpm_particle_velocity_mps,
    mpm_particle_normal,
    mpm_particle_area_m2,
    mpm_particle_count: int,
    solid_step: Callable[[], Any],
    marker_pressure_neumann_gradient_pa_per_m_field,
    search_radius_m: float,
    interior_probe_distance_m: float,
    mpm_support_radius_m: float,
    primary_region_id: int = 0,
    secondary_region_id: int = 0,
    far_pressure_region_id: int = -1,
    far_pressure_barrier_region_id: int = -1,
    far_pressure_pa: float = 0.0,
    far_pressure_side_normal_sign: float = 0.0,
    far_pressure_inside_probe_max_multiplier: float = 3.0,
    two_sided_probe_max_multiplier: float = 3.0,
    one_sided_pressure_region_id: int = -1,
    one_sided_reference_pressure_pa: float = 0.0,
    one_sided_probe_max_multiplier: float = 3.0,
    far_pressure_air_backed: bool = False,
    far_pressure_air_backed_probe_normal_sign: float = 0.0,
    fluid_dt_s: float | None = None,
    fluid_substeps: int = 1,
    projection_iterations: int = 40,
    run_fluid_predictor: bool = True,
    fluid_advection_scheme: str = "euler",
    pressure_neumann_density_kgm3: float | None = None,
    pressure_neumann_dt_s: float | None = None,
    pressure_outlet_zmin: bool = False,
    reset_pressure: bool = False,
    pressure_solver: str = "fv_cg",
    pressure_solve_failure_policy: str = "raise",
    multigrid_cycles: int | None = None,
    cg_tolerance: float = 1.0e-6,
    cg_preconditioner: str = "auto",
    surface_feedback_dt_s: float | None = None,
    divergence_cleanup_iterations: int = 0,
    divergence_cleanup_relaxation: float = 0.7,
    classify_far_internal_nodes: bool = False,
    convert_internal_nodes_to_obstacles: bool = True,
    post_dirichlet_consistency_projection_iterations: int = 3,
    diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
    update_surface_geometry_from_mpm: bool = True,
    interpolate_velocity_dirichlet_with_interior: bool = True,
) -> HibmMpmSharpMpmStepReport:
    if not callable(solid_step):
        raise ValueError("solid_step must be callable")
    particles = int(mpm_particle_count)
    if surface_feedback_dt_s is None:
        feedback_dt = float(fluid_dt_s) if fluid_dt_s is not None else float(fluid.dt)
    else:
        feedback_dt = float(surface_feedback_dt_s)
    if not math.isfinite(feedback_dt) or feedback_dt <= 0.0:
        raise ValueError("surface_feedback_dt_s must be finite and positive")
    post_solid_fluid_substeps = max(1, int(fluid_substeps))
    if fluid_dt_s is not None:
        post_solid_project_dt = float(fluid_dt_s) / float(post_solid_fluid_substeps)
    else:
        post_solid_project_dt = float(fluid.dt) / float(post_solid_fluid_substeps)
    pressure_neumann_density = None
    pressure_neumann_dt = None
    if pressure_neumann_density_kgm3 is not None:
        pressure_neumann_density = float(pressure_neumann_density_kgm3)
        if (
            not math.isfinite(pressure_neumann_density)
            or pressure_neumann_density <= 0.0
        ):
            raise ValueError("pressure_neumann_density_kgm3 must be finite and positive")
        if pressure_neumann_dt_s is None:
            raise ValueError(
                "pressure_neumann_dt_s must be provided when updating pressure "
                "Neumann from the fluid predictor"
            )
        pressure_neumann_dt = float(pressure_neumann_dt_s)
        if not math.isfinite(pressure_neumann_dt) or pressure_neumann_dt <= 0.0:
            raise ValueError("pressure_neumann_dt_s must be finite and positive")
    load_report = assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
        fluid=fluid,
        markers=markers,
        ib_search=ib_search,
        ib_boundary=ib_boundary,
        mpm_external_force_n=mpm_external_force_n,
        mpm_particle_position_m=mpm_particle_position_m,
        mpm_particle_count=particles,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            marker_pressure_neumann_gradient_pa_per_m_field
        ),
        search_radius_m=float(search_radius_m),
        interior_probe_distance_m=float(interior_probe_distance_m),
        mpm_support_radius_m=float(mpm_support_radius_m),
        primary_region_id=int(primary_region_id),
        secondary_region_id=int(secondary_region_id),
        far_pressure_region_id=int(far_pressure_region_id),
        far_pressure_barrier_region_id=int(far_pressure_barrier_region_id),
        far_pressure_pa=float(far_pressure_pa),
        far_pressure_side_normal_sign=float(far_pressure_side_normal_sign),
        far_pressure_inside_probe_max_multiplier=float(
            far_pressure_inside_probe_max_multiplier
        ),
        two_sided_probe_max_multiplier=float(
            two_sided_probe_max_multiplier
        ),
        one_sided_pressure_region_id=int(one_sided_pressure_region_id),
        one_sided_reference_pressure_pa=float(one_sided_reference_pressure_pa),
        one_sided_probe_max_multiplier=float(one_sided_probe_max_multiplier),
        far_pressure_air_backed=bool(far_pressure_air_backed),
        far_pressure_air_backed_probe_normal_sign=float(
            far_pressure_air_backed_probe_normal_sign
        ),
        dt_s=fluid_dt_s,
        fluid_substeps=int(fluid_substeps),
        projection_iterations=int(projection_iterations),
        run_fluid_predictor=bool(run_fluid_predictor),
        fluid_advection_scheme=str(fluid_advection_scheme),
        pressure_neumann_density_kgm3=pressure_neumann_density,
        pressure_neumann_dt_s=pressure_neumann_dt,
        pressure_outlet_zmin=bool(pressure_outlet_zmin),
        reset_pressure=bool(reset_pressure),
        pressure_solver=str(pressure_solver),
        pressure_solve_failure_policy=str(pressure_solve_failure_policy),
        multigrid_cycles=multigrid_cycles,
        cg_tolerance=float(cg_tolerance),
        cg_preconditioner=str(cg_preconditioner),
        divergence_cleanup_iterations=int(divergence_cleanup_iterations),
        divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
        classify_far_internal_nodes=bool(classify_far_internal_nodes),
        convert_internal_nodes_to_obstacles=bool(convert_internal_nodes_to_obstacles),
        post_dirichlet_consistency_projection_iterations=int(
            post_dirichlet_consistency_projection_iterations
        ),
        diagnostic_disable_pressure_neumann_matrix_rows=bool(
            diagnostic_disable_pressure_neumann_matrix_rows
        ),
        interpolate_velocity_dirichlet_with_interior=bool(
            interpolate_velocity_dirichlet_with_interior
        ),
    )
    if not hibm_mpm_external_force_fresh_for_solid_step(load_report):
        scatter = load_report.mpm_force_scatter
        clear = load_report.mpm_external_force_clear
        raise RuntimeError(
            "solid_step requires a fresh HIBM-MPM external force scatter: "
            f"cleared_particles={clear.cleared_particle_count}, "
            f"active_markers={scatter.active_marker_count}, "
            f"active_particles={scatter.active_particle_count}, "
            f"action_reaction_residual_n={scatter.action_reaction_residual_n}"
        )
    mpm_report = solid_step()
    if bool(update_surface_geometry_from_mpm):
        feedback_report = markers.update_surface_feedback_from_mpm_surface_particles(
            mpm_particle_position_m,
            mpm_particle_velocity_mps,
            mpm_particle_normal,
            mpm_particle_area_m2,
            particle_count=particles,
            support_radius_m=float(mpm_support_radius_m),
            dt_s=feedback_dt,
        )
    else:
        feedback_report = markers.update_surface_feedback_from_mpm_particles(
            mpm_particle_position_m,
            mpm_particle_velocity_mps,
            particle_count=particles,
            support_radius_m=float(mpm_support_radius_m),
            dt_s=feedback_dt,
        )
    next_ib_report = ib_search.search_and_classify_grid_fields(
        markers,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        search_radius_m=float(search_radius_m),
        interior_probe_distance_m=float(interior_probe_distance_m),
        classify_far_internal_nodes=bool(classify_far_internal_nodes),
    )
    next_pressure_neumann_gradient_report = None
    if pressure_neumann_density is not None and pressure_neumann_dt is not None:
        next_pressure_neumann_gradient_report = (
            ib_boundary.update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
                velocity_field=fluid.velocity,
                obstacle_field=fluid.obstacle,
                search=ib_search,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
                density_kgm3=pressure_neumann_density,
                dt_s=pressure_neumann_dt,
            )
        )
    next_internal_obstacle_cell_count = fluid.apply_hibm_internal_obstacles(
        ib_search.node_kind_code,
        internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
        convert_internal_nodes=bool(convert_internal_nodes_to_obstacles),
    )
    next_boundary_report = ib_boundary.build_from_search_device_fields(
        ib_search,
        markers,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            marker_pressure_neumann_gradient_pa_per_m_field
        ),
    )
    fluid.clear_velocity_dirichlet_boundary_rows()
    fluid.clear_pressure_interface_matrix_terms()
    next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
        fluid.velocity_dirichlet_boundary_active,
        fluid.velocity_dirichlet_boundary_value_mps,
        fluid.velocity_dirichlet_boundary_projection_weight,
        fluid.obstacle,
        fluid.velocity,
        ib_search,
        cell_face_x_m=fluid.cell_face_x_m,
        cell_face_y_m=fluid.cell_face_y_m,
        cell_face_z_m=fluid.cell_face_z_m,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        grid_nodes=fluid.grid.grid_nodes,
        velocity_dirichlet_marker_region_id=(
            fluid.velocity_dirichlet_boundary_marker_region_id
        ),
        marker_region_id=markers.region_id,
        primary_region_id=primary_region_id,
        secondary_region_id=secondary_region_id,
        interpolate_interior_velocity=bool(
            interpolate_velocity_dirichlet_with_interior
        ),
    )
    next_solid_band_nonprojectable_cell_count = (
        fluid.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            node_kind_code=ib_search.node_kind_code,
            unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE,
            protect_velocity_dirichlet_radius_cells=0,
            protect_unstamped_velocity_dirichlet_rows=True,
            protect_solid_band_mask=True,
        )
    )
    next_pressure_disconnected_nonprojectable_cell_count = (
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
        )
    )
    next_row_cloud_orphan_cell_count = 0
    next_row_cloud_orphan_component_count = 0
    next_overflow_singleton_cleanup_cell_count = 0
    next_overflow_singleton_cleanup_component_count = 0
    next_projection_tiny_unreached_cleanup_cell_count = 0
    next_projection_tiny_unreached_cleanup_component_count = 0

    def convert_next_row_cloud_orphans_until_saturated() -> None:
        nonlocal next_pressure_disconnected_nonprojectable_cell_count
        nonlocal next_row_cloud_orphan_cell_count
        nonlocal next_row_cloud_orphan_component_count
        nonlocal next_velocity_report
        for _next_row_cloud_orphan_pass in range(8):
            converted_count = fluid.convert_hibm_row_cloud_orphan_components(
                max_component_cells=(
                    HIBM_PRESSURE_DISCONNECTED_SMALL_COMPONENT_THRESHOLD_CELLS
                ),
            )
            if int(converted_count) <= 0:
                break
            next_row_cloud_orphan_cell_count += int(converted_count)
            next_row_cloud_orphan_component_count += int(
                getattr(fluid, "last_hibm_row_cloud_orphan_component_count", 0)
            )
            next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
                fluid.velocity_dirichlet_boundary_active,
                fluid.velocity_dirichlet_boundary_value_mps,
                fluid.velocity_dirichlet_boundary_projection_weight,
                fluid.obstacle,
                fluid.velocity,
                ib_search,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
                velocity_dirichlet_marker_region_id=(
                    fluid.velocity_dirichlet_boundary_marker_region_id
                ),
                marker_region_id=markers.region_id,
                primary_region_id=primary_region_id,
                secondary_region_id=secondary_region_id,
                interpolate_interior_velocity=bool(
                    interpolate_velocity_dirichlet_with_interior
                ),
            )
            next_pressure_disconnected_nonprojectable_cell_count = (
                fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                )
            )

    def convert_next_overflow_singletons_without_row_reload() -> bool:
        nonlocal next_pressure_disconnected_nonprojectable_cell_count
        nonlocal next_overflow_singleton_cleanup_cell_count
        nonlocal next_overflow_singleton_cleanup_component_count
        converted_count = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=1,
            overflow_singletons_only=True,
            protect_velocity_dirichlet_radius_cells=(
                HIBM_OVERFLOW_SINGLETON_NO_SLIP_PROTECTION_RADIUS_CELLS
            ),
        )
        if int(converted_count) <= 0:
            return False
        next_overflow_singleton_cleanup_cell_count += int(converted_count)
        next_overflow_singleton_cleanup_component_count += int(
            getattr(fluid, "last_hibm_row_cloud_orphan_component_count", 0)
        )
        next_pressure_disconnected_nonprojectable_cell_count = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
        )
        return True

    def rebuild_next_velocity_rows() -> None:
        nonlocal next_velocity_report
        next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            ib_search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            marker_region_id=markers.region_id,
            primary_region_id=primary_region_id,
            secondary_region_id=secondary_region_id,
            interpolate_interior_velocity=bool(
                interpolate_velocity_dirichlet_with_interior
            ),
        )

    def convert_next_projection_topology_cleanup_until_saturated() -> None:
        nonlocal next_pressure_disconnected_nonprojectable_cell_count
        nonlocal next_projection_tiny_unreached_cleanup_cell_count
        nonlocal next_projection_tiny_unreached_cleanup_component_count
        convert_next_row_cloud_orphans_until_saturated()
        tiny_unreached_cleanup_threshold = max(
            0,
            HIBM_TINY_UNREACHED_COMPONENT_CLEANUP_THRESHOLD_CELLS,
        )
        for _projection_cleanup_pass in range(8):
            mutated = convert_next_overflow_singletons_without_row_reload()
            if tiny_unreached_cleanup_threshold > 0:
                for _tiny_unreached_cleanup_pass in range(8):
                    converted_tiny_unreached = (
                        fluid.convert_hibm_row_cloud_orphan_components(
                            max_component_cells=tiny_unreached_cleanup_threshold,
                            convert_unstamped_small_components=True,
                            protect_velocity_dirichlet_radius_cells=0,
                            protect_solid_band_mask=False,
                        )
                    )
                    if int(converted_tiny_unreached) <= 0:
                        break
                    mutated = True
                    next_projection_tiny_unreached_cleanup_cell_count += int(
                        converted_tiny_unreached
                    )
                    next_projection_tiny_unreached_cleanup_component_count += int(
                        getattr(fluid, "last_hibm_row_cloud_orphan_component_count", 0)
                    )
                    next_pressure_disconnected_nonprojectable_cell_count = (
                        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                            pressure_outlet_zmin=bool(pressure_outlet_zmin),
                        )
                    )
            if not mutated:
                break
            rebuild_next_velocity_rows()
            next_pressure_disconnected_nonprojectable_cell_count = (
                fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                )
            )
            convert_next_row_cloud_orphans_until_saturated()

    if (
        int(next_solid_band_nonprojectable_cell_count) > 0
        or int(next_pressure_disconnected_nonprojectable_cell_count) > 0
    ):
        next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            ib_search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            marker_region_id=markers.region_id,
            primary_region_id=primary_region_id,
            secondary_region_id=secondary_region_id,
            interpolate_interior_velocity=bool(
                interpolate_velocity_dirichlet_with_interior
            ),
        )
        for _next_band_pass in range(8):
            next_band_increment = (
                fluid.mark_hibm_solid_band_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                    node_kind_code=ib_search.node_kind_code,
                    unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE,
                    protect_velocity_dirichlet_radius_cells=0,
                    protect_unstamped_velocity_dirichlet_rows=True,
                    protect_solid_band_mask=True,
                )
            )
            if int(next_band_increment) <= 0:
                break
            next_solid_band_nonprojectable_cell_count = int(
                next_solid_band_nonprojectable_cell_count
            ) + int(next_band_increment)
            next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
                fluid.velocity_dirichlet_boundary_active,
                fluid.velocity_dirichlet_boundary_value_mps,
                fluid.velocity_dirichlet_boundary_projection_weight,
                fluid.obstacle,
                fluid.velocity,
                ib_search,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
                velocity_dirichlet_marker_region_id=(
                    fluid.velocity_dirichlet_boundary_marker_region_id
                ),
                marker_region_id=markers.region_id,
                primary_region_id=primary_region_id,
                secondary_region_id=secondary_region_id,
                interpolate_interior_velocity=bool(
                    interpolate_velocity_dirichlet_with_interior
                ),
            )
        next_pressure_disconnected_nonprojectable_cell_count = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
        )
    use_next_air_backed_reachability_barrier = (
        bool(far_pressure_air_backed) and int(far_pressure_region_id) != -1
    )
    if use_next_air_backed_reachability_barrier:
        markers.write_region_pressure_reachability_barrier(
            fluid.hibm_pressure_reachability_barrier,
            ib_search.node_kind_code,
            ib_search.nearest_marker,
            barrier_node_code=HibmMpmIbNodeSearch._NODE_EXTERNAL_IB,
            barrier_region_id=int(far_pressure_region_id),
            secondary_barrier_region_id=int(far_pressure_barrier_region_id),
            tertiary_barrier_region_id=int(one_sided_pressure_region_id),
            include_all_classified_region_nodes=True,
        )
    next_pressure_disconnected_nonprojectable_cell_count = (
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            use_existing_reachability_barrier=use_next_air_backed_reachability_barrier,
        )
    )
    convert_next_row_cloud_orphans_until_saturated()
    next_air_backed_cell_count = 0
    if bool(far_pressure_air_backed) and int(far_pressure_region_id) != -1:
        markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            far_pressure_region_id=int(far_pressure_region_id),
            far_pressure_inside_probe_max_multiplier=float(
                far_pressure_inside_probe_max_multiplier
            ),
            far_pressure_air_backed_probe_normal_sign=float(
                far_pressure_air_backed_probe_normal_sign
            ),
            fallback_to_bidirectional_if_all_missed=True,
            fallback_to_region_adjacency_if_all_missed=True,
            node_kind_code=ib_search.node_kind_code,
            nearest_marker=ib_search.nearest_marker,
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
        )
        next_air_backed_cell_count = fluid.convert_hibm_air_backed_cells()
        if int(next_air_backed_cell_count) > 0:
            for _next_air_backed_band_pass in range(8):
                next_band_increment = (
                    fluid.mark_hibm_solid_band_nonprojectable_cells(
                        pressure_outlet_zmin=bool(pressure_outlet_zmin),
                        node_kind_code=ib_search.node_kind_code,
                        unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE,
                        protect_velocity_dirichlet_radius_cells=0,
                        protect_unstamped_velocity_dirichlet_rows=True,
                        protect_solid_band_mask=True,
                    )
                )
                if int(next_band_increment) <= 0:
                    break
                next_solid_band_nonprojectable_cell_count = int(
                    next_solid_band_nonprojectable_cell_count
                ) + int(next_band_increment)
                next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
                    fluid.velocity_dirichlet_boundary_active,
                    fluid.velocity_dirichlet_boundary_value_mps,
                    fluid.velocity_dirichlet_boundary_projection_weight,
                    fluid.obstacle,
                    fluid.velocity,
                    ib_search,
                    cell_face_x_m=fluid.cell_face_x_m,
                    cell_face_y_m=fluid.cell_face_y_m,
                    cell_face_z_m=fluid.cell_face_z_m,
                    cell_center_x_m=fluid.cell_center_x_m,
                    cell_center_y_m=fluid.cell_center_y_m,
                    cell_center_z_m=fluid.cell_center_z_m,
                    grid_nodes=fluid.grid.grid_nodes,
                    velocity_dirichlet_marker_region_id=(
                        fluid.velocity_dirichlet_boundary_marker_region_id
                    ),
                    marker_region_id=markers.region_id,
                    primary_region_id=primary_region_id,
                    secondary_region_id=secondary_region_id,
                    interpolate_interior_velocity=bool(
                        interpolate_velocity_dirichlet_with_interior
                    ),
                )
    # Air conversion changes pressure reachability; keep current-step velocity
    # rows intact so the post-solid projection does not consume diagnostic rows.
    next_pressure_disconnected_nonprojectable_cell_count = (
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
        )
    )
    next_projection_overflow_cell_count_before = (
        next_overflow_singleton_cleanup_cell_count
    )
    next_projection_overflow_component_count_before = (
        next_overflow_singleton_cleanup_component_count
    )
    next_projection_tiny_cell_count_before = (
        next_projection_tiny_unreached_cleanup_cell_count
    )
    next_projection_tiny_component_count_before = (
        next_projection_tiny_unreached_cleanup_component_count
    )
    convert_next_projection_topology_cleanup_until_saturated()
    # Final-sweep band populations for the post-step rebuild (S2-A8'):
    # -1 when the band ran without a split (default mode).
    next_solid_band_interior_cell_count = int(
        getattr(fluid, "last_hibm_solid_band_interior_cells", -1)
    )
    next_solid_band_enclosed_water_cell_count = int(
        getattr(fluid, "last_hibm_solid_band_enclosed_water_cells", -1)
    )
    next_solid_band_velocity_dirichlet_protected_cell_count = int(
        getattr(
            fluid,
            "last_hibm_solid_band_velocity_dirichlet_protected_cells",
            -1,
        )
    )
    next_solid_band_mask_protected_cell_count = int(
        getattr(fluid, "last_hibm_solid_band_mask_protected_cells", -1)
    )
    next_pressure_disconnected_region = hibm_mpm_pressure_disconnected_region_report(
        fluid,
        primary_region_id=primary_region_id,
        secondary_region_id=secondary_region_id,
    )
    next_pressure_report = ib_boundary.assemble_pressure_neumann_matrix_rows(
        fluid.pressure_interface_matrix_diagonal,
        fluid.pressure_interface_matrix_rhs,
        fluid.pressure_interface_coupling_active,
        fluid.pressure_interface_coupling_neighbor,
        fluid.pressure_interface_coupling_coefficient,
        fluid.obstacle,
        fluid.velocity_dirichlet_boundary_active,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        ib_search,
        markers,
        pressure_coupling_extra_neighbor=(
            fluid.pressure_interface_coupling_extra_neighbor
        ),
        pressure_coupling_extra_coefficient=(
            fluid.pressure_interface_coupling_extra_coefficient
        ),
        pressure_interface_row_count=fluid.pressure_interface_row_count,
        pressure_interface_row_owner=fluid.pressure_interface_row_owner,
        pressure_interface_row_neighbor=fluid.pressure_interface_row_neighbor,
        pressure_interface_row_transmissibility=(
            fluid.pressure_interface_row_transmissibility
        ),
        pressure_interface_row_capacity=fluid.pressure_interface_row_capacity,
        cell_face_x_m=fluid.cell_face_x_m,
        cell_face_y_m=fluid.cell_face_y_m,
        cell_face_z_m=fluid.cell_face_z_m,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        grid_nodes=fluid.grid.grid_nodes,
    )
    next_pressure_neumann_invalid_diagnostic_rows = tuple(
        ib_boundary.pressure_neumann_invalid_diagnostic_rows(
            search=ib_search,
            markers=markers,
            fluid=fluid,
        )
    )
    post_solid_projection_applied = False
    post_solid_project_report: dict[str, Any] | None = None
    post_solid_no_slip_report: HibmMpmNoSlipResidualReport | None = None
    if int(next_velocity_report.active_velocity_dirichlet_rows) > 0:
        requested_pressure_solver = str(pressure_solver)
        effective_pressure_solver = requested_pressure_solver
        pressure_solver_forced_to_fv_cg = False
        pressure_solver_force_reason = ""
        if (
            int(next_pressure_report.active_pressure_neumann_rows) > 0
            and effective_pressure_solver
            in {"jacobi", "compact_jacobi", "fv_multigrid"}
        ):
            effective_pressure_solver = "fv_cg"
            pressure_solver_forced_to_fv_cg = True
            pressure_solver_force_reason = "hibm_pressure_neumann_requires_fv_solver"
        post_solid_project_report = dict(
            fluid.project(
                iterations=int(projection_iterations),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                dt_s=float(post_solid_project_dt),
                preserve_velocity_constraints=False,
                reset_pressure=False,
                pressure_solver=effective_pressure_solver,
                multigrid_cycles=multigrid_cycles,
                cg_tolerance=float(cg_tolerance),
                cg_preconditioner=str(cg_preconditioner),
                pressure_solve_failure_policy=str(pressure_solve_failure_policy),
                hibm_tiny_unreached_cleanup_component_cells=0,
                divergence_cleanup_iterations=int(divergence_cleanup_iterations),
                divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
                read_report=True,
            )
        )
        post_solid_project_report[
            "hibm_projection_overflow_singleton_cleanup_cell_count"
        ] = (
            int(
                post_solid_project_report.get(
                    "hibm_projection_overflow_singleton_cleanup_cell_count",
                    0,
                )
            )
            + int(next_overflow_singleton_cleanup_cell_count)
            - int(next_projection_overflow_cell_count_before)
        )
        post_solid_project_report[
            "hibm_projection_overflow_singleton_cleanup_component_count"
        ] = (
            int(
                post_solid_project_report.get(
                    "hibm_projection_overflow_singleton_cleanup_component_count",
                    0,
                )
            )
            + int(next_overflow_singleton_cleanup_component_count)
            - int(next_projection_overflow_component_count_before)
        )
        post_solid_project_report[
            "hibm_projection_tiny_unreached_cleanup_cell_count"
        ] = (
            int(
                post_solid_project_report.get(
                    "hibm_projection_tiny_unreached_cleanup_cell_count",
                    0,
                )
            )
            + int(next_projection_tiny_unreached_cleanup_cell_count)
            - int(next_projection_tiny_cell_count_before)
        )
        post_solid_project_report[
            "hibm_projection_tiny_unreached_cleanup_component_count"
        ] = (
            int(
                post_solid_project_report.get(
                    "hibm_projection_tiny_unreached_cleanup_component_count",
                    0,
                )
            )
            + int(next_projection_tiny_unreached_cleanup_component_count)
            - int(next_projection_tiny_component_count_before)
        )
        post_solid_project_report.update(
            {
                "pressure_solver_requested": requested_pressure_solver,
                "pressure_solver": effective_pressure_solver,
                "pressure_solver_forced_to_fv_cg": pressure_solver_forced_to_fv_cg,
                "pressure_solver_force_reason": pressure_solver_force_reason,
                "pressure_interface_neumann_active_rows": int(
                    next_pressure_report.active_pressure_neumann_rows
                ),
                "hibm_projection_stage": "post_solid_kinematic_consistency",
                "hibm_post_solid_kinematic_projection_applied": True,
                "hibm_post_solid_kinematic_projection_count": 1,
            }
        )
        post_solid_projection_applied = True
        post_solid_no_slip_sampling_obstacle = (
            fluid.build_hibm_no_slip_sampling_obstacle()
        )
        post_solid_no_slip_report = markers.sample_no_slip_residual(
            fluid.velocity,
            post_solid_no_slip_sampling_obstacle,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.grid.grid_nodes,
            primary_region_id=int(primary_region_id),
            secondary_region_id=int(secondary_region_id),
        )
    return HibmMpmSharpMpmStepReport(
        fluid_to_mpm_loads=load_report,
        mpm=mpm_report,
        surface_feedback=feedback_report,
        next_ib_node_search=next_ib_report,
        next_internal_obstacle_cell_count=next_internal_obstacle_cell_count,
        next_solid_band_nonprojectable_cell_count=next_solid_band_nonprojectable_cell_count,
        next_solid_band_interior_cell_count=next_solid_band_interior_cell_count,
        next_solid_band_enclosed_water_cell_count=(
            next_solid_band_enclosed_water_cell_count
        ),
        next_solid_band_velocity_dirichlet_protected_cell_count=(
            next_solid_band_velocity_dirichlet_protected_cell_count
        ),
        next_solid_band_mask_protected_cell_count=(
            next_solid_band_mask_protected_cell_count
        ),
        next_row_cloud_orphan_cell_count=int(next_row_cloud_orphan_cell_count),
        next_row_cloud_orphan_component_count=int(
            next_row_cloud_orphan_component_count
        ),
        next_overflow_singleton_cleanup_cell_count=int(
            next_overflow_singleton_cleanup_cell_count
        ),
        next_overflow_singleton_cleanup_component_count=int(
            next_overflow_singleton_cleanup_component_count
        ),
        next_pressure_disconnected_nonprojectable_cell_count=(
            next_pressure_disconnected_nonprojectable_cell_count
        ),
        next_pressure_disconnected_region=next_pressure_disconnected_region,
        next_boundary_conditions=next_boundary_report,
        next_velocity_dirichlet=next_velocity_report,
        next_pressure_neumann=next_pressure_report,
        next_pressure_neumann_gradient=next_pressure_neumann_gradient_report,
        next_pressure_neumann_invalid_diagnostic_rows=(
            next_pressure_neumann_invalid_diagnostic_rows
        ),
        post_solid_kinematic_projection_applied=post_solid_projection_applied,
        post_solid_fluid_projection=post_solid_project_report,
        post_solid_no_slip_residual=post_solid_no_slip_report,
    )


def hibm_mpm_sharp_step_summary(
    report: HibmMpmSharpMpmStepReport,
) -> dict[str, Any]:
    load = report.fluid_to_mpm_loads
    marker_forces = load.marker_forces
    clear = load.mpm_external_force_clear
    scatter = load.mpm_force_scatter
    feedback = report.surface_feedback
    pressure_gradient = load.pressure_neumann_gradient
    next_gradient = report.next_pressure_neumann_gradient
    post_projection = report.post_solid_fluid_projection or {}
    post_no_slip = report.post_solid_no_slip_residual
    return {
        "hibm_coupling_scheme": "explicit_loose",
        "hibm_added_mass_stability_status": "unmeasured_single_pass",
        "hibm_added_mass_stability_measured": False,
        "hibm_added_mass_stabilization": "none",
        "hibm_semi_implicit_coupling_enabled": False,
        "hibm_semi_implicit_coupling_matrix_active": False,
        "hibm_fsi_coupling_iterations_used": 1,
        "hibm_fsi_coupling_converged": False,
        "hibm_fsi_coupling_explicit_single_pass": True,
        "hibm_fsi_coupling_residual_source": "unmeasured_single_pass",
        "hibm_ib_node_count": load.ib_node_search.near_boundary_node_count,
        "hibm_ib_external_node_count": load.ib_node_search.external_ib_node_count,
        "hibm_ib_internal_node_count": load.ib_node_search.internal_node_count,
        "hibm_internal_obstacle_cell_count": load.internal_obstacle_cell_count,
        "hibm_solid_band_nonprojectable_cell_count": (
            load.solid_band_nonprojectable_cell_count
        ),
        "hibm_solid_band_interior_cell_count": (
            load.solid_band_interior_cell_count
        ),
        "hibm_solid_band_enclosed_water_cell_count": (
            load.solid_band_enclosed_water_cell_count
        ),
        "hibm_solid_band_velocity_dirichlet_protected_cell_count": (
            load.solid_band_velocity_dirichlet_protected_cell_count
        ),
        "hibm_solid_band_mask_protected_cell_count": (
            load.solid_band_mask_protected_cell_count
        ),
        "hibm_row_cloud_orphan_cell_count": load.row_cloud_orphan_cell_count,
        "hibm_row_cloud_orphan_component_count": (
            load.row_cloud_orphan_component_count
        ),
        "hibm_overflow_singleton_cleanup_cell_count": (
            load.overflow_singleton_cleanup_cell_count
        ),
        "hibm_overflow_singleton_cleanup_component_count": (
            load.overflow_singleton_cleanup_component_count
        ),
        "hibm_pressure_disconnected_nonprojectable_cell_count": (
            load.pressure_disconnected_nonprojectable_cell_count
        ),
        "hibm_pressure_disconnected_component_count": (
            load.pressure_disconnected_region.component_count
        ),
        "hibm_pressure_disconnected_component_raw_count": (
            load.pressure_disconnected_region.component_raw_count
        ),
        "hibm_pressure_disconnected_largest_component_cell_count": (
            load.pressure_disconnected_region.largest_component_cell_count
        ),
        "hibm_pressure_disconnected_singleton_component_count": (
            load.pressure_disconnected_region.singleton_component_count
        ),
        "hibm_pressure_disconnected_small_component_threshold_cells": (
            load.pressure_disconnected_region.small_component_threshold_cells
        ),
        "hibm_pressure_disconnected_small_component_count": (
            load.pressure_disconnected_region.small_component_count
        ),
        "hibm_pressure_disconnected_small_component_cell_count": (
            load.pressure_disconnected_region.small_component_cell_count
        ),
        "hibm_pressure_disconnected_component_overflow": (
            load.pressure_disconnected_region.component_overflow
        ),
        "hibm_pressure_disconnected_component_labels_converged": (
            load.pressure_disconnected_region.component_labels_converged
        ),
        "hibm_pressure_disconnected_min_i": load.pressure_disconnected_region.min_i,
        "hibm_pressure_disconnected_min_j": load.pressure_disconnected_region.min_j,
        "hibm_pressure_disconnected_min_k": load.pressure_disconnected_region.min_k,
        "hibm_pressure_disconnected_max_i": load.pressure_disconnected_region.max_i,
        "hibm_pressure_disconnected_max_j": load.pressure_disconnected_region.max_j,
        "hibm_pressure_disconnected_max_k": load.pressure_disconnected_region.max_k,
        "hibm_pressure_disconnected_primary_region_stencil_cell_count": (
            load.pressure_disconnected_region.primary_region_stencil_cell_count
        ),
        "hibm_pressure_disconnected_secondary_region_stencil_cell_count": (
            load.pressure_disconnected_region.secondary_region_stencil_cell_count
        ),
        "hibm_pressure_disconnected_other_region_stencil_cell_count": (
            load.pressure_disconnected_region.other_region_stencil_cell_count
        ),
        "hibm_pressure_disconnected_unassigned_region_stencil_cell_count": (
            load.pressure_disconnected_region.unassigned_region_stencil_cell_count
        ),
        "hibm_air_backed_cell_count": load.air_backed_cell_count,
        "hibm_air_backed_component_count": load.air_backed_component_count,
        "hibm_air_backed_cell_volume_m3": load.air_backed_cell_volume_m3,
        "hibm_air_backed_seed_marker_count": (
            load.air_backed_seed_marker_count
        ),
        "hibm_air_backed_seed_missed_marker_count": (
            load.air_backed_seed_missed_marker_count
        ),
        "hibm_air_backed_seed_fallback_cell_count": (
            load.air_backed_seed_fallback_cell_count
        ),
        "hibm_air_backed_reachability_barrier_cell_count": (
            load.air_backed_reachability_barrier_cell_count
        ),
        "hibm_ib_invalid_projection_count": (
            load.ib_node_search.invalid_projection_count
        ),
        "hibm_boundary_no_slip_count": (
            load.boundary_conditions.no_slip_dirichlet_count
        ),
        "hibm_boundary_pressure_neumann_count": (
            load.boundary_conditions.pressure_neumann_count
        ),
        "hibm_boundary_max_abs_velocity_mps": (
            load.boundary_conditions.max_abs_velocity_mps
        ),
        "hibm_velocity_dirichlet_active_rows": (
            load.velocity_dirichlet.active_velocity_dirichlet_rows
        ),
        "hibm_velocity_dirichlet_primary_region_active_rows": (
            load.velocity_dirichlet.primary_region_active_rows
        ),
        "hibm_velocity_dirichlet_secondary_region_active_rows": (
            load.velocity_dirichlet.secondary_region_active_rows
        ),
        "hibm_velocity_dirichlet_other_region_active_rows": (
            load.velocity_dirichlet.other_region_active_rows
        ),
        "hibm_velocity_dirichlet_unassigned_region_active_rows": (
            load.velocity_dirichlet.unassigned_region_active_rows
        ),
        "hibm_velocity_dirichlet_max_abs_velocity_mps": (
            load.velocity_dirichlet.max_abs_velocity_mps
        ),
        "hibm_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": (
            load.velocity_dirichlet.raw_reconstructed_max_abs_velocity_mps
        ),
        "hibm_velocity_dirichlet_boundary_velocity_only_rows": (
            load.velocity_dirichlet.boundary_velocity_only_row_count
        ),
        "hibm_velocity_dirichlet_invalid_reconstruction_count": (
            load.velocity_dirichlet.invalid_reconstruction_row_count
        ),
        "hibm_velocity_dirichlet_invalid_no_fluid_sample_count": (
            load.velocity_dirichlet.invalid_no_fluid_sample_row_count
        ),
        "hibm_velocity_dirichlet_invalid_nonpositive_gap_count": (
            load.velocity_dirichlet.invalid_nonpositive_gap_row_count
        ),
        "hibm_velocity_dirichlet_invalid_node_behind_boundary_count": (
            load.velocity_dirichlet.invalid_node_behind_boundary_row_count
        ),
        "hibm_velocity_dirichlet_invalid_node_beyond_interior_count": (
            load.velocity_dirichlet.invalid_node_beyond_interior_row_count
        ),
        "hibm_velocity_dirichlet_narrow_gap_count": (
            load.velocity_dirichlet.narrow_gap_boundary_velocity_row_count
        ),
        "hibm_velocity_dirichlet_relocated_rows": (
            load.velocity_dirichlet.relocated_row_count
        ),
        "hibm_velocity_dirichlet_relocation_merged_rows": (
            load.velocity_dirichlet.relocation_merged_row_count
        ),
        "hibm_velocity_dirichlet_relocation_blocked_rows": (
            load.velocity_dirichlet.relocation_blocked_row_count
        ),
        "hibm_velocity_dirichlet_min_projection_weight": (
            load.velocity_dirichlet.min_projection_weight
        ),
        "hibm_velocity_dirichlet_max_projection_weight": (
            load.velocity_dirichlet.max_projection_weight
        ),
        "hibm_pressure_neumann_active_rows": (
            load.pressure_neumann.active_pressure_neumann_rows
        ),
        "hibm_pressure_neumann_skipped_velocity_dirichlet_count": (
            load.pressure_neumann.skipped_velocity_dirichlet_row_count
        ),
        "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
            load.pressure_neumann.skipped_pressure_boundary_adjacent_row_count
        ),
        "hibm_pressure_neumann_skipped_obstacle_owner_count": (
            load.pressure_neumann.skipped_obstacle_owner_row_count
        ),
        "hibm_pressure_neumann_relocated_obstacle_owner_count": (
            load.pressure_neumann.relocated_obstacle_owner_row_count
        ),
        "hibm_pressure_neumann_duplicate_owner_count": (
            load.pressure_neumann.duplicate_owner_row_count
        ),
        "hibm_pressure_neumann_overflow_owner_count": (
            load.pressure_neumann.overflow_owner_row_count
        ),
        "hibm_pressure_neumann_max_owner_slot_count": (
            load.pressure_neumann.max_owner_slot_count
        ),
        "hibm_pressure_interface_row_list_enabled": (
            load.pressure_neumann.pressure_interface_row_list_enabled
        ),
        "hibm_pressure_interface_row_list_count": (
            load.pressure_neumann.pressure_interface_row_list_count
        ),
        "hibm_pressure_neumann_rhs_integral": load.pressure_neumann.rhs_integral,
        "hibm_pressure_neumann_max_abs_rhs": load.pressure_neumann.max_abs_rhs,
        "hibm_pressure_neumann_invalid_reconstruction_count": (
            load.pressure_neumann.invalid_reconstruction_row_count
        ),
        "hibm_pressure_neumann_invalid_unreconstructable_count": (
            load.pressure_neumann.invalid_unreconstructable_row_count
        ),
        "hibm_pressure_neumann_invalid_bad_marker_count": (
            load.pressure_neumann.invalid_bad_marker_row_count
        ),
        "hibm_pressure_neumann_invalid_nonpositive_volume_count": (
            load.pressure_neumann.invalid_nonpositive_volume_row_count
        ),
        "hibm_pressure_neumann_min_reconstruction_gap_m": (
            load.pressure_neumann.min_reconstruction_gap_m
        ),
        "hibm_pressure_neumann_max_reconstruction_gap_m": (
            load.pressure_neumann.max_reconstruction_gap_m
        ),
        "hibm_pressure_neumann_max_transmissibility_m": (
            load.pressure_neumann.max_transmissibility_m
        ),
        "hibm_pressure_neumann_max_raw_transmissibility_m": (
            load.pressure_neumann.max_raw_transmissibility_m
        ),
        "hibm_pressure_neumann_max_transmissibility_limit_m": (
            load.pressure_neumann.max_transmissibility_limit_m
        ),
        "hibm_pressure_neumann_transmissibility_capped_row_count": (
            load.pressure_neumann.transmissibility_capped_row_count
        ),
        "hibm_pressure_neumann_max_diagonal_per_m2": (
            load.pressure_neumann.max_diagonal_per_m2
        ),
        "hibm_pressure_neumann_active_marker_count": (
            load.pressure_neumann.active_pressure_neumann_marker_count
        ),
        "hibm_pressure_neumann_max_rows_per_marker": (
            load.pressure_neumann.max_pressure_neumann_rows_per_marker
        ),
        "hibm_pressure_neumann_gradient_available": pressure_gradient is not None,
        "hibm_pressure_neumann_gradient_active_marker_count": (
            0 if pressure_gradient is None else pressure_gradient.active_marker_count
        ),
        "hibm_pressure_neumann_gradient_max_abs_pa_per_m": (
            0.0
            if pressure_gradient is None
            else pressure_gradient.max_abs_gradient_pa_per_m
        ),
        "hibm_pressure_neumann_gradient_raw_max_abs_pa_per_m": (
            0.0
            if pressure_gradient is None
            else pressure_gradient.max_raw_abs_gradient_pa_per_m
        ),
        "hibm_pressure_neumann_gradient_limited_count": (
            0 if pressure_gradient is None else pressure_gradient.limited_gradient_count
        ),
        "hibm_velocity_dirichlet_near_divergence_l2": load.fluid_projection.get(
            "velocity_dirichlet_near_l2",
            0.0,
        ),
        "hibm_velocity_dirichlet_near_divergence_max_abs": load.fluid_projection.get(
            "velocity_dirichlet_near_max_abs",
            0.0,
        ),
        "hibm_velocity_dirichlet_far_divergence_l2": load.fluid_projection.get(
            "velocity_dirichlet_far_l2",
            0.0,
        ),
        "hibm_velocity_dirichlet_far_divergence_max_abs": load.fluid_projection.get(
            "velocity_dirichlet_far_max_abs",
            0.0,
        ),
        "hibm_velocity_dirichlet_apply_calls": load.fluid_projection.get(
            "velocity_dirichlet_boundary_apply_calls",
            0,
        ),
        "hibm_velocity_dirichlet_applied_active_cells_total": (
            load.fluid_projection.get(
                "velocity_dirichlet_boundary_active_cells_total",
                0,
            )
        ),
        "hibm_velocity_dirichlet_applied_active_cells_max": (
            load.fluid_projection.get(
                "velocity_dirichlet_boundary_active_cells_max",
                0,
            )
        ),
        "hibm_velocity_dirichlet_applied_max_delta_mps": (
            load.fluid_projection.get(
                "velocity_dirichlet_boundary_max_delta_mps",
                0.0,
            )
        ),
        "hibm_velocity_dirichlet_applied_mean_delta_mps": (
            load.fluid_projection.get(
                "velocity_dirichlet_boundary_mean_delta_mps",
                0.0,
            )
        ),
        "hibm_velocity_dirichlet_applied_momentum_delta_n_s": (
            load.fluid_projection.get(
                "velocity_dirichlet_boundary_momentum_delta_n_s",
                (0.0, 0.0, 0.0),
            )
        ),
        "hibm_pressure_correctable_divergence_l2": load.fluid_projection.get(
            "pressure_correctable_l2",
            0.0,
        ),
        "hibm_pressure_correctable_divergence_max_abs": load.fluid_projection.get(
            "pressure_correctable_max_abs",
            0.0,
        ),
        "hibm_pressure_correctable_divergence_cell_count": load.fluid_projection.get(
            "pressure_correctable_cell_count",
            0,
        ),
        "hibm_pressure_fixed_divergence_l2": load.fluid_projection.get(
            "pressure_fixed_l2",
            0.0,
        ),
        "hibm_pressure_fixed_divergence_max_abs": load.fluid_projection.get(
            "pressure_fixed_max_abs",
            0.0,
        ),
        "hibm_pressure_fixed_divergence_cell_count": load.fluid_projection.get(
            "pressure_fixed_cell_count",
            0,
        ),
        "hibm_interior_pressure_correctable_divergence_l2": (
            load.fluid_projection.get("interior_pressure_correctable_l2", 0.0)
        ),
        "hibm_interior_pressure_correctable_divergence_max_abs": (
            load.fluid_projection.get("interior_pressure_correctable_max_abs", 0.0)
        ),
        "hibm_interior_pressure_correctable_divergence_cell_count": (
            load.fluid_projection.get("interior_pressure_correctable_cell_count", 0)
        ),
        "hibm_interior_pressure_fixed_divergence_l2": load.fluid_projection.get(
            "interior_pressure_fixed_l2",
            0.0,
        ),
        "hibm_interior_pressure_fixed_divergence_max_abs": load.fluid_projection.get(
            "interior_pressure_fixed_max_abs",
            0.0,
        ),
        "hibm_interior_pressure_fixed_divergence_cell_count": load.fluid_projection.get(
            "interior_pressure_fixed_cell_count",
            0,
        ),
        "hibm_fluid_predictor_applied": load.fluid_predictor_applied,
        "hibm_no_slip_residual_valid_marker_count": (
            load.no_slip_residual.valid_marker_count
        ),
        "hibm_no_slip_residual_invalid_marker_count": (
            load.no_slip_residual.invalid_marker_count
        ),
        "hibm_no_slip_residual_max_mps": (
            load.no_slip_residual.max_no_slip_residual_mps
        ),
        "hibm_no_slip_residual_l2_mps": (
            load.no_slip_residual.l2_no_slip_residual_mps
        ),
        "hibm_no_slip_residual_direct_sample_marker_count": (
            load.no_slip_residual.direct_sample_marker_count
        ),
        "hibm_no_slip_residual_normal_walk_sample_marker_count": (
            load.no_slip_residual.normal_walk_sample_marker_count
        ),
        "hibm_no_slip_residual_nearest_fluid_sample_marker_count": (
            load.no_slip_residual.nearest_fluid_sample_marker_count
        ),
        "hibm_no_slip_residual_zero_normal_marker_count": (
            load.no_slip_residual.zero_normal_marker_count
        ),
        "hibm_no_slip_residual_no_fluid_sample_marker_count": (
            load.no_slip_residual.no_fluid_sample_marker_count
        ),
        "hibm_no_slip_residual_primary_region_valid_marker_count": (
            load.no_slip_residual.primary_region_valid_marker_count
        ),
        "hibm_no_slip_residual_primary_region_invalid_marker_count": (
            load.no_slip_residual.primary_region_invalid_marker_count
        ),
        "hibm_no_slip_residual_secondary_region_valid_marker_count": (
            load.no_slip_residual.secondary_region_valid_marker_count
        ),
        "hibm_no_slip_residual_secondary_region_invalid_marker_count": (
            load.no_slip_residual.secondary_region_invalid_marker_count
        ),
        "hibm_no_slip_residual_other_region_valid_marker_count": (
            load.no_slip_residual.other_region_valid_marker_count
        ),
        "hibm_no_slip_residual_other_region_invalid_marker_count": (
            load.no_slip_residual.other_region_invalid_marker_count
        ),
        "hibm_post_solid_kinematic_projection_applied": (
            report.post_solid_kinematic_projection_applied
        ),
        "hibm_post_solid_kinematic_projection_count": post_projection.get(
            "hibm_post_solid_kinematic_projection_count",
            0,
        ),
        "hibm_post_solid_pressure_projection_cg_converged_all": (
            post_projection.get("cg_converged_all", True)
        ),
        "hibm_post_solid_pressure_projection_cg_breakdown_count": (
            post_projection.get("cg_breakdown_count", 0)
        ),
        "hibm_post_solid_pressure_projection_cg_relative_residual_max": (
            post_projection.get("cg_relative_residual_max", 0.0)
        ),
        "hibm_post_solid_divergence_l2": post_projection.get("l2", 0.0),
        "hibm_post_solid_divergence_max_abs": post_projection.get("max_abs", 0.0),
        "hibm_post_solid_interior_divergence_l2": post_projection.get(
            "interior_l2",
            0.0,
        ),
        "hibm_post_solid_interior_divergence_max_abs": post_projection.get(
            "interior_max_abs",
            0.0,
        ),
        "hibm_post_solid_projection_divergence_l2": post_projection.get(
            "projection_l2",
            0.0,
        ),
        "hibm_post_solid_projection_divergence_max_abs": post_projection.get(
            "projection_max_abs",
            0.0,
        ),
        "hibm_post_solid_post_boundary_divergence_l2": post_projection.get(
            "post_boundary_l2",
            0.0,
        ),
        "hibm_post_solid_post_boundary_divergence_max_abs": post_projection.get(
            "post_boundary_max_abs",
            0.0,
        ),
        "hibm_post_solid_post_constraint_divergence_l2": post_projection.get(
            "post_constraint_l2",
            0.0,
        ),
        "hibm_post_solid_post_constraint_divergence_max_abs": post_projection.get(
            "post_constraint_max_abs",
            0.0,
        ),
        "hibm_post_solid_velocity_dirichlet_apply_calls": (
            post_projection.get("velocity_dirichlet_boundary_apply_calls", 0)
        ),
        "hibm_post_solid_velocity_dirichlet_applied_active_cells_total": (
            post_projection.get(
                "velocity_dirichlet_boundary_active_cells_total",
                0,
            )
        ),
        "hibm_post_solid_velocity_dirichlet_applied_max_delta_mps": (
            post_projection.get("velocity_dirichlet_boundary_max_delta_mps", 0.0)
        ),
        "hibm_post_solid_no_slip_residual_valid_marker_count": (
            0 if post_no_slip is None else post_no_slip.valid_marker_count
        ),
        "hibm_post_solid_no_slip_residual_invalid_marker_count": (
            0 if post_no_slip is None else post_no_slip.invalid_marker_count
        ),
        "hibm_post_solid_no_slip_residual_max_mps": (
            0.0 if post_no_slip is None else post_no_slip.max_no_slip_residual_mps
        ),
        "hibm_post_solid_no_slip_residual_l2_mps": (
            0.0 if post_no_slip is None else post_no_slip.l2_no_slip_residual_mps
        ),
        "hibm_post_solid_no_slip_residual_direct_sample_marker_count": (
            0 if post_no_slip is None else post_no_slip.direct_sample_marker_count
        ),
        "hibm_post_solid_no_slip_residual_normal_walk_sample_marker_count": (
            0 if post_no_slip is None else post_no_slip.normal_walk_sample_marker_count
        ),
        "hibm_post_solid_no_slip_residual_nearest_fluid_sample_marker_count": (
            0
            if post_no_slip is None
            else post_no_slip.nearest_fluid_sample_marker_count
        ),
        "hibm_post_solid_no_slip_residual_zero_normal_marker_count": (
            0 if post_no_slip is None else post_no_slip.zero_normal_marker_count
        ),
        "hibm_post_solid_no_slip_residual_no_fluid_sample_marker_count": (
            0 if post_no_slip is None else post_no_slip.no_fluid_sample_marker_count
        ),
        "hibm_post_solid_no_slip_residual_primary_region_valid_marker_count": (
            0
            if post_no_slip is None
            else post_no_slip.primary_region_valid_marker_count
        ),
        "hibm_post_solid_no_slip_residual_primary_region_invalid_marker_count": (
            0
            if post_no_slip is None
            else post_no_slip.primary_region_invalid_marker_count
        ),
        "hibm_post_solid_no_slip_residual_secondary_region_valid_marker_count": (
            0
            if post_no_slip is None
            else post_no_slip.secondary_region_valid_marker_count
        ),
        "hibm_post_solid_no_slip_residual_secondary_region_invalid_marker_count": (
            0
            if post_no_slip is None
            else post_no_slip.secondary_region_invalid_marker_count
        ),
        "hibm_post_solid_no_slip_residual_other_region_valid_marker_count": (
            0 if post_no_slip is None else post_no_slip.other_region_valid_marker_count
        ),
        "hibm_post_solid_no_slip_residual_other_region_invalid_marker_count": (
            0
            if post_no_slip is None
            else post_no_slip.other_region_invalid_marker_count
        ),
        "hibm_full_stress_valid_marker_count": load.fluid_stress.valid_marker_count,
        "hibm_full_stress_invalid_marker_count": (
            load.fluid_stress.invalid_marker_count
        ),
        "hibm_full_stress_viscous_gradient_invalid_marker_count": (
            load.fluid_stress.viscous_gradient_invalid_marker_count
        ),
        "hibm_full_stress_max_abs_traction_pa": (
            load.fluid_stress.max_abs_traction_pa
        ),
        "hibm_full_stress_far_pressure_closed_marker_count": (
            load.fluid_stress.far_pressure_closed_marker_count
        ),
        "hibm_full_stress_far_pressure_closed_extended_marker_count": (
            load.fluid_stress.far_pressure_closed_extended_marker_count
        ),
        "hibm_full_stress_far_pressure_anchor_closed_marker_count": (
            load.fluid_stress.far_pressure_anchor_closed_marker_count
        ),
        "hibm_full_stress_far_pressure_node_anchor_closed_marker_count": (
            load.fluid_stress.far_pressure_node_anchor_closed_marker_count
        ),
        "hibm_full_stress_closure_gradient_missing_marker_count": (
            load.fluid_stress.closure_gradient_missing_marker_count
        ),
        "hibm_full_stress_far_pressure_outside_suppressed_marker_count": (
            load.fluid_stress.far_pressure_outside_suppressed_marker_count
        ),
        "hibm_full_stress_two_sided_pressure_marker_count": (
            load.fluid_stress.two_sided_pressure_marker_count
        ),
        "hibm_full_stress_two_sided_extended_marker_count": (
            load.fluid_stress.two_sided_extended_marker_count
        ),
        "hibm_full_stress_one_sided_pressure_marker_count": (
            load.fluid_stress.one_sided_pressure_marker_count
        ),
        "hibm_full_stress_one_sided_extended_marker_count": (
            load.fluid_stress.one_sided_extended_marker_count
        ),
        "hibm_full_stress_one_sided_gradient_missing_marker_count": (
            load.fluid_stress.one_sided_gradient_missing_marker_count
        ),
        "hibm_marker_primary_count": marker_forces.primary_marker_count,
        "hibm_marker_secondary_count": marker_forces.secondary_marker_count,
        "hibm_marker_total_count": marker_forces.total_marker_count,
        "hibm_marker_primary_stress_valid_count": (
            marker_forces.primary_stress_valid_marker_count
        ),
        "hibm_marker_primary_stress_invalid_count": (
            marker_forces.primary_stress_invalid_marker_count
        ),
        "hibm_marker_secondary_stress_valid_count": (
            marker_forces.secondary_stress_valid_marker_count
        ),
        "hibm_marker_secondary_stress_invalid_count": (
            marker_forces.secondary_stress_invalid_marker_count
        ),
        "hibm_marker_primary_force_n": marker_forces.primary_marker_force_n,
        "hibm_marker_secondary_force_n": marker_forces.secondary_marker_force_n,
        "hibm_marker_total_force_n": marker_forces.total_marker_force_n,
        "hibm_marker_primary_force_norm_sum_n": (
            marker_forces.primary_marker_force_norm_sum_n
        ),
        "hibm_marker_secondary_force_norm_sum_n": (
            marker_forces.secondary_marker_force_norm_sum_n
        ),
        "hibm_marker_total_force_norm_sum_n": (
            marker_forces.total_marker_force_norm_sum_n
        ),
        "hibm_marker_primary_force_norm_max_n": (
            marker_forces.primary_marker_force_norm_max_n
        ),
        "hibm_marker_secondary_force_norm_max_n": (
            marker_forces.secondary_marker_force_norm_max_n
        ),
        "hibm_marker_total_force_norm_max_n": (
            marker_forces.total_marker_force_norm_max_n
        ),
        "hibm_marker_fluid_reaction_force_n": marker_forces.fluid_reaction_force_n,
        "hibm_marker_action_reaction_residual_n": (
            marker_forces.action_reaction_residual_n
        ),
        "hibm_mpm_external_force_clear_particle_count": (
            clear.cleared_particle_count
        ),
        "hibm_mpm_external_force_clear_max_abs_before_n": (
            clear.max_abs_external_force_before_n
        ),
        "hibm_mpm_external_force_fresh_for_solid_step": (
            hibm_mpm_external_force_fresh_for_solid_step(load)
        ),
        "hibm_mpm_scatter_active_marker_count": scatter.active_marker_count,
        "hibm_mpm_scatter_invalid_marker_count": scatter.invalid_marker_count,
        "hibm_mpm_scatter_active_particle_count": scatter.active_particle_count,
        "hibm_mpm_scatter_total_external_force_n": (
            scatter.total_mpm_external_force_n
        ),
        "hibm_mpm_scatter_action_reaction_residual_n": (
            scatter.action_reaction_residual_n
        ),
        "hibm_surface_updated_marker_count": feedback.updated_marker_count,
        "hibm_surface_invalid_marker_count": feedback.invalid_marker_count,
        "hibm_surface_max_displacement_m": feedback.max_marker_displacement_m,
        "hibm_surface_max_speed_mps": feedback.max_marker_speed_mps,
        "hibm_surface_geometry_updated_marker_count": (
            feedback.geometry_updated_marker_count
        ),
        "hibm_surface_geometry_invalid_marker_count": (
            feedback.geometry_invalid_marker_count
        ),
        "hibm_next_ib_node_count": report.next_ib_node_search.near_boundary_node_count,
        "hibm_next_ib_external_node_count": (
            report.next_ib_node_search.external_ib_node_count
        ),
        "hibm_next_ib_internal_node_count": (
            report.next_ib_node_search.internal_node_count
        ),
        "hibm_next_internal_obstacle_cell_count": (
            report.next_internal_obstacle_cell_count
        ),
        "hibm_next_solid_band_nonprojectable_cell_count": (
            report.next_solid_band_nonprojectable_cell_count
        ),
        "hibm_next_solid_band_interior_cell_count": (
            report.next_solid_band_interior_cell_count
        ),
        "hibm_next_solid_band_enclosed_water_cell_count": (
            report.next_solid_band_enclosed_water_cell_count
        ),
        "hibm_next_solid_band_velocity_dirichlet_protected_cell_count": (
            report.next_solid_band_velocity_dirichlet_protected_cell_count
        ),
        "hibm_next_solid_band_mask_protected_cell_count": (
            report.next_solid_band_mask_protected_cell_count
        ),
        "hibm_next_row_cloud_orphan_cell_count": (
            report.next_row_cloud_orphan_cell_count
        ),
        "hibm_next_row_cloud_orphan_component_count": (
            report.next_row_cloud_orphan_component_count
        ),
        "hibm_next_overflow_singleton_cleanup_cell_count": (
            report.next_overflow_singleton_cleanup_cell_count
        ),
        "hibm_next_overflow_singleton_cleanup_component_count": (
            report.next_overflow_singleton_cleanup_component_count
        ),
        "hibm_next_pressure_disconnected_nonprojectable_cell_count": (
            report.next_pressure_disconnected_nonprojectable_cell_count
        ),
        "hibm_next_pressure_disconnected_component_count": (
            report.next_pressure_disconnected_region.component_count
        ),
        "hibm_next_pressure_disconnected_component_raw_count": (
            report.next_pressure_disconnected_region.component_raw_count
        ),
        "hibm_next_pressure_disconnected_largest_component_cell_count": (
            report.next_pressure_disconnected_region.largest_component_cell_count
        ),
        "hibm_next_pressure_disconnected_singleton_component_count": (
            report.next_pressure_disconnected_region.singleton_component_count
        ),
        "hibm_next_pressure_disconnected_small_component_threshold_cells": (
            report.next_pressure_disconnected_region.small_component_threshold_cells
        ),
        "hibm_next_pressure_disconnected_small_component_count": (
            report.next_pressure_disconnected_region.small_component_count
        ),
        "hibm_next_pressure_disconnected_small_component_cell_count": (
            report.next_pressure_disconnected_region.small_component_cell_count
        ),
        "hibm_next_pressure_disconnected_component_overflow": (
            report.next_pressure_disconnected_region.component_overflow
        ),
        "hibm_next_pressure_disconnected_component_labels_converged": (
            report.next_pressure_disconnected_region.component_labels_converged
        ),
        "hibm_next_pressure_disconnected_min_i": (
            report.next_pressure_disconnected_region.min_i
        ),
        "hibm_next_pressure_disconnected_min_j": (
            report.next_pressure_disconnected_region.min_j
        ),
        "hibm_next_pressure_disconnected_min_k": (
            report.next_pressure_disconnected_region.min_k
        ),
        "hibm_next_pressure_disconnected_max_i": (
            report.next_pressure_disconnected_region.max_i
        ),
        "hibm_next_pressure_disconnected_max_j": (
            report.next_pressure_disconnected_region.max_j
        ),
        "hibm_next_pressure_disconnected_max_k": (
            report.next_pressure_disconnected_region.max_k
        ),
        "hibm_next_pressure_disconnected_primary_region_stencil_cell_count": (
            report.next_pressure_disconnected_region.primary_region_stencil_cell_count
        ),
        "hibm_next_pressure_disconnected_secondary_region_stencil_cell_count": (
            report.next_pressure_disconnected_region.secondary_region_stencil_cell_count
        ),
        "hibm_next_pressure_disconnected_other_region_stencil_cell_count": (
            report.next_pressure_disconnected_region.other_region_stencil_cell_count
        ),
        "hibm_next_pressure_disconnected_unassigned_region_stencil_cell_count": (
            report.next_pressure_disconnected_region.unassigned_region_stencil_cell_count
        ),
        "hibm_next_ib_invalid_projection_count": (
            report.next_ib_node_search.invalid_projection_count
        ),
        "hibm_next_boundary_no_slip_count": (
            report.next_boundary_conditions.no_slip_dirichlet_count
        ),
        "hibm_next_boundary_pressure_neumann_count": (
            report.next_boundary_conditions.pressure_neumann_count
        ),
        "hibm_next_boundary_max_abs_velocity_mps": (
            report.next_boundary_conditions.max_abs_velocity_mps
        ),
        "hibm_next_velocity_dirichlet_active_rows": (
            report.next_velocity_dirichlet.active_velocity_dirichlet_rows
        ),
        "hibm_next_velocity_dirichlet_primary_region_active_rows": (
            report.next_velocity_dirichlet.primary_region_active_rows
        ),
        "hibm_next_velocity_dirichlet_secondary_region_active_rows": (
            report.next_velocity_dirichlet.secondary_region_active_rows
        ),
        "hibm_next_velocity_dirichlet_other_region_active_rows": (
            report.next_velocity_dirichlet.other_region_active_rows
        ),
        "hibm_next_velocity_dirichlet_unassigned_region_active_rows": (
            report.next_velocity_dirichlet.unassigned_region_active_rows
        ),
        "hibm_next_velocity_dirichlet_max_abs_velocity_mps": (
            report.next_velocity_dirichlet.max_abs_velocity_mps
        ),
        "hibm_next_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": (
            report.next_velocity_dirichlet.raw_reconstructed_max_abs_velocity_mps
        ),
        "hibm_next_velocity_dirichlet_boundary_velocity_only_rows": (
            report.next_velocity_dirichlet.boundary_velocity_only_row_count
        ),
        "hibm_next_velocity_dirichlet_invalid_reconstruction_count": (
            report.next_velocity_dirichlet.invalid_reconstruction_row_count
        ),
        "hibm_next_velocity_dirichlet_min_projection_weight": (
            report.next_velocity_dirichlet.min_projection_weight
        ),
        "hibm_next_velocity_dirichlet_max_projection_weight": (
            report.next_velocity_dirichlet.max_projection_weight
        ),
        "hibm_next_pressure_neumann_active_rows": (
            report.next_pressure_neumann.active_pressure_neumann_rows
        ),
        "hibm_next_pressure_neumann_skipped_velocity_dirichlet_count": (
            report.next_pressure_neumann.skipped_velocity_dirichlet_row_count
        ),
        "hibm_next_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
            report.next_pressure_neumann.skipped_pressure_boundary_adjacent_row_count
        ),
        "hibm_next_pressure_neumann_skipped_obstacle_owner_count": (
            report.next_pressure_neumann.skipped_obstacle_owner_row_count
        ),
        "hibm_next_pressure_neumann_relocated_obstacle_owner_count": (
            report.next_pressure_neumann.relocated_obstacle_owner_row_count
        ),
        "hibm_next_pressure_neumann_duplicate_owner_count": (
            report.next_pressure_neumann.duplicate_owner_row_count
        ),
        "hibm_next_pressure_neumann_overflow_owner_count": (
            report.next_pressure_neumann.overflow_owner_row_count
        ),
        "hibm_next_pressure_neumann_max_owner_slot_count": (
            report.next_pressure_neumann.max_owner_slot_count
        ),
        "hibm_next_pressure_interface_row_list_enabled": (
            report.next_pressure_neumann.pressure_interface_row_list_enabled
        ),
        "hibm_next_pressure_interface_row_list_count": (
            report.next_pressure_neumann.pressure_interface_row_list_count
        ),
        "hibm_next_pressure_neumann_invalid_reconstruction_count": (
            report.next_pressure_neumann.invalid_reconstruction_row_count
        ),
        "hibm_next_pressure_neumann_invalid_unreconstructable_count": (
            report.next_pressure_neumann.invalid_unreconstructable_row_count
        ),
        "hibm_next_pressure_neumann_invalid_bad_marker_count": (
            report.next_pressure_neumann.invalid_bad_marker_row_count
        ),
        "hibm_next_pressure_neumann_invalid_nonpositive_volume_count": (
            report.next_pressure_neumann.invalid_nonpositive_volume_row_count
        ),
        "hibm_next_pressure_neumann_min_reconstruction_gap_m": (
            report.next_pressure_neumann.min_reconstruction_gap_m
        ),
        "hibm_next_pressure_neumann_max_reconstruction_gap_m": (
            report.next_pressure_neumann.max_reconstruction_gap_m
        ),
        "hibm_next_pressure_neumann_max_transmissibility_m": (
            report.next_pressure_neumann.max_transmissibility_m
        ),
        "hibm_next_pressure_neumann_max_raw_transmissibility_m": (
            report.next_pressure_neumann.max_raw_transmissibility_m
        ),
        "hibm_next_pressure_neumann_max_transmissibility_limit_m": (
            report.next_pressure_neumann.max_transmissibility_limit_m
        ),
        "hibm_next_pressure_neumann_transmissibility_capped_row_count": (
            report.next_pressure_neumann.transmissibility_capped_row_count
        ),
        "hibm_next_pressure_neumann_max_diagonal_per_m2": (
            report.next_pressure_neumann.max_diagonal_per_m2
        ),
        "hibm_next_pressure_neumann_gradient_available": next_gradient is not None,
        "hibm_next_pressure_neumann_gradient_active_marker_count": (
            0 if next_gradient is None else next_gradient.active_marker_count
        ),
        "hibm_next_pressure_neumann_gradient_max_abs_pa_per_m": (
            0.0
            if next_gradient is None
            else next_gradient.max_abs_gradient_pa_per_m
        ),
        "hibm_next_pressure_neumann_gradient_raw_max_abs_pa_per_m": (
            0.0
            if next_gradient is None
            else next_gradient.max_raw_abs_gradient_pa_per_m
        ),
        "hibm_next_pressure_neumann_gradient_limited_count": (
            0 if next_gradient is None else next_gradient.limited_gradient_count
        ),
    }


def advance_hibm_mpm_sharp_neo_hookean_step(
    *,
    fluid: Any,
    markers: HibmMpmSurfaceMarkers,
    ib_search: HibmMpmIbNodeSearch,
    ib_boundary: HibmMpmIbBoundaryConditions,
    solid: Any,
    marker_pressure_neumann_gradient_pa_per_m_field,
    search_radius_m: float,
    interior_probe_distance_m: float,
    mpm_support_radius_m: float,
    solid_dt_s: float,
    mu_pa: float,
    lambda_pa: float,
        primary_region_id: int,
        secondary_region_id: int,
        far_pressure_region_id: int = -1,
        far_pressure_barrier_region_id: int = -1,
        far_pressure_pa: float = 0.0,
    far_pressure_side_normal_sign: float = 0.0,
    far_pressure_inside_probe_max_multiplier: float = 3.0,
    two_sided_probe_max_multiplier: float = 3.0,
    one_sided_pressure_region_id: int = -1,
    one_sided_reference_pressure_pa: float = 0.0,
    one_sided_probe_max_multiplier: float = 3.0,
    far_pressure_air_backed: bool = False,
    far_pressure_air_backed_probe_normal_sign: float = 0.0,
    fluid_dt_s: float | None = None,
    fluid_substeps: int = 1,
    projection_iterations: int = 40,
    run_fluid_predictor: bool = True,
    fluid_advection_scheme: str = "euler",
    pressure_neumann_density_kgm3: float | None = None,
    pressure_neumann_dt_s: float | None = None,
    pressure_outlet_zmin: bool = False,
    reset_pressure: bool = False,
    pressure_solver: str = "fv_cg",
    pressure_solve_failure_policy: str = "raise",
    multigrid_cycles: int | None = None,
    cg_tolerance: float = 1.0e-6,
    cg_preconditioner: str = "auto",
    divergence_cleanup_iterations: int = 0,
    divergence_cleanup_relaxation: float = 0.7,
    read_mpm_report: bool = True,
    solid_external_loads: Callable[[], None] | None = None,
    classify_far_internal_nodes: bool = False,
    convert_internal_nodes_to_obstacles: bool = True,
    diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
) -> HibmMpmSharpNeoHookeanStepReport:
    def run_solid_step_with_external_loads() -> Any:
        if solid_external_loads is not None:
            solid_external_loads()
        return solid.step(
            dt_s=float(solid_dt_s),
            mu_pa=float(mu_pa),
            lambda_pa=float(lambda_pa),
            primary_region_id=int(primary_region_id),
            secondary_region_id=int(secondary_region_id),
            velocity_damping=1.0,
            read_report=bool(read_mpm_report),
        )

    report = advance_hibm_mpm_sharp_mpm_step(
        fluid=fluid,
        markers=markers,
        ib_search=ib_search,
        ib_boundary=ib_boundary,
        mpm_external_force_n=solid.external_force_n,
        mpm_particle_position_m=solid.x,
        mpm_particle_velocity_mps=solid.v,
        mpm_particle_normal=solid.surface_normal,
        mpm_particle_area_m2=solid.area_weight_m2,
        mpm_particle_count=solid.particle_count,
        solid_step=run_solid_step_with_external_loads,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            marker_pressure_neumann_gradient_pa_per_m_field
        ),
        search_radius_m=float(search_radius_m),
        interior_probe_distance_m=float(interior_probe_distance_m),
        mpm_support_radius_m=float(mpm_support_radius_m),
            primary_region_id=int(primary_region_id),
            secondary_region_id=int(secondary_region_id),
            far_pressure_region_id=int(far_pressure_region_id),
            far_pressure_barrier_region_id=int(far_pressure_barrier_region_id),
            far_pressure_pa=float(far_pressure_pa),
        far_pressure_side_normal_sign=float(far_pressure_side_normal_sign),
        far_pressure_inside_probe_max_multiplier=float(
            far_pressure_inside_probe_max_multiplier
        ),
        two_sided_probe_max_multiplier=float(
            two_sided_probe_max_multiplier
        ),
        one_sided_pressure_region_id=int(one_sided_pressure_region_id),
        one_sided_reference_pressure_pa=float(one_sided_reference_pressure_pa),
        one_sided_probe_max_multiplier=float(one_sided_probe_max_multiplier),
        far_pressure_air_backed=bool(far_pressure_air_backed),
        far_pressure_air_backed_probe_normal_sign=float(
            far_pressure_air_backed_probe_normal_sign
        ),
        fluid_dt_s=fluid_dt_s,
        fluid_substeps=int(fluid_substeps),
        projection_iterations=int(projection_iterations),
        run_fluid_predictor=bool(run_fluid_predictor),
        fluid_advection_scheme=str(fluid_advection_scheme),
        pressure_neumann_density_kgm3=pressure_neumann_density_kgm3,
        pressure_neumann_dt_s=(
            float(solid_dt_s) if pressure_neumann_dt_s is None else pressure_neumann_dt_s
        ),
        pressure_outlet_zmin=bool(pressure_outlet_zmin),
        reset_pressure=bool(reset_pressure),
        pressure_solver=str(pressure_solver),
        pressure_solve_failure_policy=str(pressure_solve_failure_policy),
        multigrid_cycles=multigrid_cycles,
        cg_tolerance=float(cg_tolerance),
        cg_preconditioner=str(cg_preconditioner),
        surface_feedback_dt_s=float(solid_dt_s),
        divergence_cleanup_iterations=int(divergence_cleanup_iterations),
        divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
        classify_far_internal_nodes=bool(classify_far_internal_nodes),
        convert_internal_nodes_to_obstacles=bool(convert_internal_nodes_to_obstacles),
        diagnostic_disable_pressure_neumann_matrix_rows=bool(
            diagnostic_disable_pressure_neumann_matrix_rows
        ),
    )
    return HibmMpmSharpNeoHookeanStepReport(
        fluid_to_mpm_loads=report.fluid_to_mpm_loads,
        mpm=report.mpm,
        surface_feedback=report.surface_feedback,
        next_ib_node_search=report.next_ib_node_search,
        next_internal_obstacle_cell_count=report.next_internal_obstacle_cell_count,
        next_solid_band_nonprojectable_cell_count=(
            report.next_solid_band_nonprojectable_cell_count
        ),
        next_solid_band_interior_cell_count=(
            report.next_solid_band_interior_cell_count
        ),
        next_solid_band_enclosed_water_cell_count=(
            report.next_solid_band_enclosed_water_cell_count
        ),
        next_solid_band_velocity_dirichlet_protected_cell_count=(
            report.next_solid_band_velocity_dirichlet_protected_cell_count
        ),
        next_solid_band_mask_protected_cell_count=(
            report.next_solid_band_mask_protected_cell_count
        ),
        next_overflow_singleton_cleanup_cell_count=(
            report.next_overflow_singleton_cleanup_cell_count
        ),
        next_overflow_singleton_cleanup_component_count=(
            report.next_overflow_singleton_cleanup_component_count
        ),
        next_pressure_disconnected_nonprojectable_cell_count=(
            report.next_pressure_disconnected_nonprojectable_cell_count
        ),
        next_boundary_conditions=report.next_boundary_conditions,
        next_velocity_dirichlet=report.next_velocity_dirichlet,
        next_pressure_neumann=report.next_pressure_neumann,
        next_pressure_neumann_gradient=report.next_pressure_neumann_gradient,
    )
