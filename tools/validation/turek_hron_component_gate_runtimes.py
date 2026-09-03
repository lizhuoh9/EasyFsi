"""Strict-CUDA production adapters for Turek-Hron component gates.

This module intentionally owns runtime construction only.  The gate evaluator
and persistence policy remain in :mod:`run_turek_hron_component_gates` so these
adapters can be inspected without importing that CLI back into production code.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _strict_marker_mac_q_cycle_trace_json(
    fluid_projection: Mapping[str, object],
) -> str:
    """Encode the complete generic affine-Q ledger as one history scalar."""

    trace = fluid_projection.get("hibm_marker_mac_q_cycle_trace")
    if not isinstance(trace, list) or not trace or not all(
        isinstance(cycle, Mapping) for cycle in trace
    ):
        raise RuntimeError("FAIL_MARKER_MAC_Q_TRACE:missing-or-invalid")
    try:
        return json.dumps(trace, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("FAIL_MARKER_MAC_Q_TRACE:not-strict-json") from error


def _strict_cuda_runtime() -> Any:
    from simulation_core.diagnostics.runtime import TaichiRuntimeConfig

    return TaichiRuntimeConfig(arch="cuda", strict_arch=True)


def _measured_taichi_runtime_identity() -> dict[str, Any]:
    from simulation_core.diagnostics.runtime import taichi_runtime_identity

    return taichi_runtime_identity()


def normalize_component_effective_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the complete case configuration before evidence is claimed."""

    from cases.turek_hron_fsi import (
        TurekHronFsiConfig,
        with_beam_surface_force_support,
    )

    mode = str(config.get("mode", ""))
    if mode not in {"solid-only", "fixed-fluid", "coupled-preflight"}:
        raise ValueError(f"unsupported component mode: {mode}")
    if config.get("effective_arch", "cuda") != "cuda":
        raise ValueError("FAIL_EFFECTIVE_ARCH")
    fields = TurekHronFsiConfig.__dataclass_fields__
    extras = {"mode", "effective_arch", "acceleration_mps2"}
    unknown = sorted(set(config) - set(fields) - extras)
    if unknown:
        raise ValueError(f"unknown component configuration fields: {unknown}")
    acceleration = np.asarray(config.get("acceleration_mps2"), dtype=np.float64)
    if acceleration.shape != (3,) or not bool(np.all(np.isfinite(acceleration))):
        raise ValueError("acceleration_mps2 must be one finite solver-axis vector")
    case_config = TurekHronFsiConfig(
        **{key: value for key, value in config.items() if key in fields}
    )
    if mode in {"fixed-fluid", "coupled-preflight"}:
        case_config = with_beam_surface_force_support(case_config)
    return {
        **asdict(case_config),
        "mode": mode,
        "effective_arch": "cuda",
        "acceleration_mps2": tuple(float(value) for value in acceleration),
    }


def validate_component_effective_config(
    observed: Any,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(observed, Mapping):
        raise ValueError("FAIL_EFFECTIVE_CONFIG:missing")
    actual, required = dict(observed), dict(expected)
    differing = sorted(
        key
        for key in set(actual) | set(required)
        if actual.get(key) != required.get(key)
    )
    if differing:
        raise ValueError(f"FAIL_EFFECTIVE_CONFIG:{','.join(differing)}")


def construct_component_runtime(
    runtime_type: Any,
    effective_config: Mapping[str, Any],
) -> Any:
    if not isinstance(runtime_type, type):
        raise ValueError("FAIL_RUNTIME_TYPE")
    runtime = runtime_type(MappingProxyType(dict(effective_config)))
    if getattr(runtime, "effective_arch", None) != "cuda":
        raise ValueError("FAIL_EFFECTIVE_ARCH")
    validate_component_effective_config(
        getattr(runtime, "effective_config", None),
        effective_config,
    )
    return runtime


class _SolidOnlyRuntime:
    """Production MPM solid adapter with an explicit accepted-time ledger."""

    effective_arch = "cuda"

    def __init__(self, config: Mapping[str, Any]) -> None:
        from benchmarks.official.solid_mpm_fsi_runner import _lame_parameters
        from cases.turek_hron_fsi import (
            TurekHronFsiConfig,
            _advance_turek_hron_solid_macro_step,
            _build_solid,
            _tip_displacement_row,
        )

        effective = normalize_component_effective_config(config)
        if effective["mode"] != "solid-only":
            raise ValueError("solid runtime requires solid-only mode")
        self.effective_config = MappingProxyType(effective)
        fields = TurekHronFsiConfig.__dataclass_fields__
        self._config = TurekHronFsiConfig(
            **{key: value for key, value in effective.items() if key in fields}
        )
        self.solid, self.masks = _build_solid(self._config, _strict_cuda_runtime())
        self.taichi_runtime_identity = _measured_taichi_runtime_identity()
        self._advance = _advance_turek_hron_solid_macro_step
        self._tip = _tip_displacement_row
        self.mu, self.lam = _lame_parameters(self._config)
        self._acceleration = np.zeros(3, dtype=np.float64)
        self._latest_tip: dict[str, float] = {}

    def apply_acceleration(self, acceleration: np.ndarray) -> None:
        value = np.asarray(acceleration, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("acceleration must be one finite solver-axis vector")
        self._acceleration = value.copy()
        self._apply_mass_proportional_force()

    def _apply_mass_proportional_force(self) -> None:
        count = int(self.solid.particle_count)
        masses = self.solid.mass_kg.to_numpy()[:count]
        force = np.asarray(
            masses[:, None] * self._acceleration.astype(np.float32)[None, :],
            dtype=np.float32,
        )
        # This intentionally includes fixed particles: it proves the applied
        # external load is mass-proportional rather than a free-particle proxy.
        self.solid.external_force_n.from_numpy(force)

    def advance(self) -> dict[str, Any]:
        self._apply_mass_proportional_force()
        self._advance(
            solid=self.solid,
            dt_s=float(self._config.dt_s),
            solid_substeps=int(self._config.solid_substeps),
            mu_pa=float(self.mu),
            lambda_pa=float(self.lam),
            velocity_damping=float(self._config.velocity_damping),
            constitutive_model=str(self._config.solid_constitutive_model),
            enforce_plane_strain_x=bool(self._config.enforce_plane_strain_x),
            particle_position_write_observer=lambda: None,
        )
        report = self.solid.report()
        count = int(self.solid.particle_count)
        rest = self.solid.rest_x.to_numpy()[:count]
        displacement = self.solid.x.to_numpy()[:count] - rest
        velocity = self.solid.v.to_numpy()[:count]
        applied = self.solid.external_force_n.to_numpy()[:count].sum(axis=0)
        fixed = np.asarray(self.masks["fixed"], dtype=bool)
        self._latest_tip = self._tip(self.solid, self.masks)
        point_a = np.asarray(
            (0.0, self._latest_tip["tip_uy_turek_hron_m"],
             -self._latest_tip["tip_ux_turek_hron_m"]),
            dtype=np.float64,
        )
        return {
            "solid_macro_requested_time_s": float(self._config.dt_s),
            "solid_macro_accepted_time_s": float(self._config.dt_s),
            "solid_macro_remaining_unadvanced_time_s": 0.0,
            "solid_substeps_requested": int(self._config.solid_substeps),
            "solid_substeps_observed": int(self._config.solid_substeps),
            "point_a_displacement_solver_xyz_m": point_a.tolist(),
            "_transient_solid_displacement_field_m": displacement,
            "_transient_solid_velocity_field_mps": velocity,
            "fixed_root_max_displacement_m": float(
                np.max(np.linalg.norm(displacement[fixed], axis=1), initial=0.0)
            ),
            "grid_out_of_bounds_particle_count": int(
                report.grid_out_of_bounds_particle_count
            ),
            "deformation_clamp_count": int(report.deformation_clamp_count),
            "total_mass_kg": float(report.total_mass_kg),
            "applied_force_sum_solver_xyz_n": applied.astype(np.float64).tolist(),
        }

    def tip_row(self) -> dict[str, float]:
        return dict(self._latest_tip)

    def final_arrays(self) -> dict[str, Any]:
        return {
            "solid_x_m": self.solid.x.to_numpy(),
            "solid_rest_x_m": self.solid.rest_x.to_numpy(),
            "external_force_n": self.solid.external_force_n.to_numpy(),
        }


class _FixedFluidRuntime:
    """Fixed-marker sharp-fluid adapter using the public production assembly."""

    effective_arch = "cuda"

    def __init__(self, config: Mapping[str, Any]) -> None:
        from benchmarks.official.solid_mpm_fsi_runner import (
            PRIMARY_REGION_ID,
            SECONDARY_UNUSED_REGION_ID,
        )
        from cases.turek_hron_fsi import (
            TurekHronFsiConfig,
            _boundary_fluxes_m3ps,
            _build_fluid,
            _build_markers,
            _build_solid,
            _update_turek_hron_dynamic_solid_volume,
            _capture_fluid_predictor_time_observations,
            _full_bounds,
            _force_reporting_per_span_fields,
            _verified_fluid_macro_step_time_fields,
            _write_channel_external_velocity_faces,
            beam_box_solver_m,
            build_cylinder_obstacle_mask,
            fluid_cell_spacing_m,
            resolved_marker_counts,
            thin_beam_pressure_probe_max_multiplier,
            _validate_marker_grid_consistency,
        )
        from simulation_core.coupling.hibm_mpm import (
            HibmMpmIbBoundaryConditions,
            HibmMpmIbNodeSearch,
            HibmMpmMarkerMacConstraintOperator,
            HibmMpmMarkerMacConstraintProjector,
            assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
            capture_marker_interface_state,
            marker_layout_identity,
        )
        from simulation_core.coupling.hibm_mpm.core import (
            _hibm_velocity_component_invariant_violation_count,
        )

        effective = normalize_component_effective_config(config)
        if effective["mode"] != "fixed-fluid":
            raise ValueError("fixed-fluid runtime requires fixed-fluid mode")
        self.effective_config = MappingProxyType(effective)
        fields = TurekHronFsiConfig.__dataclass_fields__
        self._config = TurekHronFsiConfig(
            **{key: value for key, value in effective.items() if key in fields}
        )
        _validate_marker_grid_consistency(self._config)
        runtime = _strict_cuda_runtime()
        self.fluid = _build_fluid(self._config, runtime)
        self.solid, self._masks = _build_solid(self._config, runtime)
        if bool(self._config.classify_far_internal_nodes):
            _update_turek_hron_dynamic_solid_volume(
                self.fluid,
                self.solid,
                self._config,
            )
        self.markers = _build_markers(self._config, runtime)
        self.taichi_runtime_identity = _measured_taichi_runtime_identity()
        bounds_min, bounds_max = _full_bounds(self._config)
        self.search = HibmMpmIbNodeSearch(
            grid_nodes=self._config.grid_nodes,
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            marker_capacity=self.markers.marker_count,
            runtime=runtime,
        )
        self.boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=self._config.grid_nodes,
            marker_capacity=self.markers.marker_count,
            runtime=runtime,
        )
        self._marker_mac_constraint_projector = (
            HibmMpmMarkerMacConstraintProjector(
                markers=self.markers,
                operator=HibmMpmMarkerMacConstraintOperator(
                    grid_nodes=self._config.grid_nodes,
                    marker_capacity=self.markers.marker_count,
                ),
                max_iterations=int(
                    self._config.flow_hibm_marker_mac_constraint_iterations
                ),
                absolute_tolerance_mps=float(
                    self._config.flow_hibm_marker_mac_constraint_absolute_tolerance_mps
                ),
                primary_region_id=PRIMARY_REGION_ID,
                secondary_region_id=SECONDARY_UNUSED_REGION_ID,
                rank_revealing_direct=True,
            )
        )
        self._assemble_loads = assemble_hibm_mpm_sharp_fluid_to_mpm_loads
        self._capture_predictor_time = _capture_fluid_predictor_time_observations
        self._verified_fluid_time = _verified_fluid_macro_step_time_fields
        self._write_faces = _write_channel_external_velocity_faces
        self._boundary_fluxes = _boundary_fluxes_m3ps
        self._force_report = _force_reporting_per_span_fields
        self._velocity_invariant_violations = (
            _hibm_velocity_component_invariant_violation_count
        )
        self._marker_layout_identity = marker_layout_identity
        self._thin_beam_probe_multiplier = thin_beam_pressure_probe_max_multiplier
        self._canonical_cylinder_mask = build_cylinder_obstacle_mask(self._config)
        self._marker_counts = resolved_marker_counts(self._config)
        self._primary = PRIMARY_REGION_ID
        self._secondary = SECONDARY_UNUSED_REGION_ID
        dx, dy, dz = fluid_cell_spacing_m(self._config)
        self._spacing_xyz_m = (dx, dy, dz)
        self._beam_box_min, self._beam_box_max = beam_box_solver_m(self._config)
        plane_spacing = max(dy, dz)
        self._search_radius_m = 1.5 * plane_spacing
        self._interior_probe_distance_m = plane_spacing
        self._search_radius_xyz_m = (
            (1.5 * dx, 1.5 * dy, 1.5 * dz)
            if self._config.ib_anisotropic_envelope
            else None
        )
        self._interior_probe_distance_xyz_m = (
            (dx, dy, dz) if self._config.ib_anisotropic_envelope else None
        )
        initial_state = capture_marker_interface_state(self.markers)
        self._marker_reference_positions = np.asarray(
            initial_state["x_gamma_m"], dtype=np.float64
        ).copy()
        self._fixed_marker_state = {
            key: np.asarray(value).copy()
            for key, value in initial_state.items()
            if key != "_marker_geometry"
        }
        self._fixed_marker_regions = self.markers.region_id.to_numpy()[
            : int(self.markers.marker_count)
        ].copy()
        self._fixed_marker_geometry = self._geometry_metadata(
            initial_state["_marker_geometry"]
        )
        self.marker_layout_hash = self._marker_layout_identity(
            self.markers,
            reference_positions_m=self._marker_reference_positions,
        )
        self._last_report: Any | None = None
        self._initialization_report: Any | None = None

    @staticmethod
    def _geometry_metadata(value: Any) -> Any:
        """Normalize small marker metadata without object-array hashing."""

        if isinstance(value, Mapping):
            return tuple(
                (str(key), _FixedFluidRuntime._geometry_metadata(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        if isinstance(value, (tuple, list)):
            return tuple(_FixedFluidRuntime._geometry_metadata(item) for item in value)
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"unsupported marker geometry metadata: {type(value).__name__}")

    @staticmethod
    def _external_y_face_ledger_residual(
        mask: np.ndarray, ledger: np.ndarray, expected: np.ndarray
    ) -> float:
        """Compare the production device ledger with the independent case target."""

        face_mask = np.asarray(mask, dtype=np.int32)
        actual = np.asarray(ledger, dtype=np.float64)
        required = np.asarray(expected, dtype=np.float64)
        expected_mask = (1 << 3) - 1
        if (
            actual.ndim != 4
            or actual.shape[0] != 2
            or actual.shape[-1] != 3
            or required.shape != actual.shape
            or face_mask.shape != actual.shape[:-1]
        ):
            return math.inf
        if (
            not np.all(face_mask == expected_mask)
            or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(required))
        ):
            return math.inf
        return float(np.max(np.abs(actual - required), initial=0.0))

    @staticmethod
    def _base_crossing_normal_speed(
        velocity: np.ndarray, base_obstacle: np.ndarray, current_obstacle: np.ndarray
    ) -> float:
        """Inspect only base-cylinder/current-fluid backward-MAC crossings."""

        maxima: list[float] = []
        for axis in range(3):
            lower = np.take(base_obstacle, range(base_obstacle.shape[axis] - 1), axis=axis)
            upper = np.take(base_obstacle, range(1, base_obstacle.shape[axis]), axis=axis)
            current_lower = np.take(current_obstacle, range(current_obstacle.shape[axis] - 1), axis=axis)
            current_upper = np.take(current_obstacle, range(1, current_obstacle.shape[axis]), axis=axis)
            crossing = ((lower != 0) & (current_upper == 0)) | (
                (upper != 0) & (current_lower == 0)
            )
            normal = np.take(velocity[..., axis], range(1, velocity.shape[axis]), axis=axis)
            if np.any(crossing):
                maxima.append(float(np.max(np.abs(normal[crossing]))))
        return max(maxima, default=0.0)

    @staticmethod
    def _connected_yz(mask: np.ndarray) -> bool:
        active = np.asarray(mask, dtype=bool)
        locations = np.argwhere(active)
        if locations.size == 0:
            return False
        start = tuple(int(value) for value in locations[0])
        reached = {start}
        frontier = [start]
        while frontier:
            y_index, z_index = frontier.pop()
            for neighbor in (
                (y_index - 1, z_index),
                (y_index + 1, z_index),
                (y_index, z_index - 1),
                (y_index, z_index + 1),
            ):
                if (
                    0 <= neighbor[0] < active.shape[0]
                    and 0 <= neighbor[1] < active.shape[1]
                    and active[neighbor]
                    and neighbor not in reached
                ):
                    reached.add(neighbor)
                    frontier.append(neighbor)
        return len(reached) == int(np.count_nonzero(active))

    def write_boundary(self, time_s: float) -> None:
        self._write_faces(self.fluid, self._config, float(time_s))
        self.fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)

    def _topology_valid(self, report: Any) -> bool:
        projection = report.fluid_projection
        disconnected = report.pressure_disconnected_region
        velocity = report.velocity_dirichlet
        return bool(
            int(report.ib_node_search.invalid_projection_count) == 0
            and int(report.pressure_disconnected_nonprojectable_cell_count) == 0
            and bool(disconnected.component_labels_converged)
            and not bool(disconnected.component_overflow)
            and bool(projection.get("pressure_outlet_operator_graph_prepared", False))
            and bool(
                projection.get(
                    "pressure_nullspace_component_labels_converged",
                    False,
                )
            )
            and not bool(
                projection.get("pressure_nullspace_component_overflow", True)
            )
            and bool(projection.get("hibm_pressure_reachability_converged", False))
            and bool(projection.get("hibm_pressure_reachability_valid", False))
            and bool(
                projection.get("hibm_pressure_component_labels_converged", False)
            )
            and int(projection.get("cg_unreached_component_overflow", -1)) == 0
            and bool(projection.get("cg_converged_all", False))
            and int(projection.get("cg_breakdown_count", -1)) == 0
            and not bool(projection.get("pressure_solve_failed", True))
            and not bool(
                projection.get("pressure_projection_physical_failure", True)
            )
            and bool(velocity.get("authority_registered", False))
            and bool(velocity.get("authority_sealed", False))
            and int(self._velocity_invariant_violations(velocity)) == 0
        )

    def _assemble(self, *, run_fluid_predictor: bool) -> tuple[Any, tuple[dict[str, float], ...]]:
        return self._capture_predictor_time(
            lambda: self._assemble_loads(
                fluid=self.fluid,
                markers=self.markers,
                ib_search=self.search,
                ib_boundary=self.boundary,
                mpm_external_force_n=self.solid.external_force_n,
                mpm_particle_position_m=self.solid.x,
                mpm_particle_count=int(self.solid.particle_count),
                mpm_particle_position_generation=0,
                marker_pressure_neumann_gradient_pa_per_m_field=(
                    self.boundary.marker_pressure_neumann_gradient_field
                ),
                search_radius_m=self._search_radius_m,
                interior_probe_distance_m=self._interior_probe_distance_m,
                mpm_support_radius_m=float(self._config.mpm_support_radius_m),
                search_radius_xyz_m=self._search_radius_xyz_m,
                interior_probe_distance_xyz_m=self._interior_probe_distance_xyz_m,
                search_inactive_axis=(
                    0 if self._config.enforce_plane_strain_x else None
                ),
                primary_region_id=self._primary,
                secondary_region_id=self._secondary,
                far_pressure_region_id=self._secondary,
                one_sided_pressure_primary_region_id=self._primary,
                one_sided_primary_fluid_side_normal_sign=1.0,
                viscous_inactive_axis=(
                    0 if self._config.enforce_plane_strain_x else None
                ),
                dt_s=float(self._config.dt_s),
                fluid_substeps=int(self._config.flow_predictor_substeps),
                projection_iterations=int(self._config.flow_projection_iterations),
                run_fluid_predictor=run_fluid_predictor,
                fluid_advection_scheme=str(self._config.fluid_advection_scheme),
                pressure_neumann_density_kgm3=float(self._config.fluid_density_kgm3),
                pressure_neumann_dt_s=float(self._config.dt_s),
                pressure_outlet_zmin=True,
                velocity_inlet_zmax=True,
                two_sided_probe_max_multiplier=self._thin_beam_probe_multiplier(
                    self._config
                ),
                pressure_solver=str(self._config.flow_pressure_solver),
                cg_tolerance=float(self._config.flow_cg_tolerance),
                cg_preconditioner=str(self._config.flow_cg_preconditioner),
                accumulate_reprojection_pressure=True,
                reprojection_projection_iterations=int(self._config.flow_reprojection_iterations),
                reprojection_cg_tolerance=float(self._config.flow_reprojection_cg_tolerance),
                post_dirichlet_consistency_projection_iterations=1,
                marker_mac_constraint_projector=(
                    self._marker_mac_constraint_projector
                ),
                interpolate_velocity_dirichlet_with_interior=bool(
                    self._config.interpolate_velocity_dirichlet_with_interior
                ),
                classify_far_internal_nodes=bool(self._config.classify_far_internal_nodes),
            ),
            fluid_to_observe=self.fluid,
        )

    def initialize_time_zero(self) -> dict[str, Any]:
        """Establish the immutable base-cylinder snapshot before physical steps."""

        self.write_boundary(0.0)
        report, observations = self._assemble(run_fluid_predictor=False)
        if observations:
            raise RuntimeError("time-zero fixed-fluid initialization advanced predictor time")
        self._initialization_report = report
        return self.initialization_audit()

    def assemble(self) -> dict[str, Any]:
        report, observations = self._assemble(run_fluid_predictor=True)
        self._last_report = report
        projection = report.fluid_projection
        marker_mac_q_cycle_trace_json = _strict_marker_mac_q_cycle_trace_json(
            projection
        )
        time_fields = self._verified_fluid_time(
            predictor_time_observations=observations,
            fluid_projection=projection,
            fluid_predictor_applied=report.fluid_predictor_applied,
            dt_s=float(self._config.dt_s),
            fluid_substeps=int(self._config.flow_predictor_substeps),
        )
        beam = np.asarray(report.marker_forces.total_marker_force_n, dtype=np.float64)
        cylinder_pressure = np.asarray(self.fluid.compute_obstacle_surface_pressure_force_n(), dtype=np.float64)
        cylinder_viscous = np.asarray(self.fluid.compute_obstacle_surface_viscous_force_n(), dtype=np.float64)
        per_span = self._force_report(
            beam_force_n=tuple(float(value) for value in beam),
            cylinder_pressure_force_n=tuple(float(value) for value in cylinder_pressure),
            cylinder_viscous_force_n=tuple(float(value) for value in cylinder_viscous),
            span_m=float(self._config.span_m),
        )
        reported_total = np.asarray((
            beam[0] + cylinder_pressure[0] + cylinder_viscous[0],
            per_span["total_lift_per_span_n_per_m"] * float(self._config.span_m),
            -per_span["total_drag_per_span_n_per_m"] * float(self._config.span_m),
        ), dtype=np.float64)
        velocity = np.asarray(self.fluid.velocity.to_numpy(), dtype=np.float64)
        obstacle = np.asarray(self.fluid.obstacle.to_numpy(), dtype=np.int32)
        base = np.asarray(self.fluid.hibm_base_obstacle.to_numpy(), dtype=np.int32)
        active_velocity = velocity[obstacle == 0]
        if active_velocity.size == 0:
            raise RuntimeError("FAIL_FLUID_ACTIVE_FIELD_EMPTY")
        no_slip = report.no_slip_residual
        mask = self.fluid.external_velocity_boundary_y_face_active_component_mask.to_numpy()
        ledger = self.fluid.external_velocity_boundary_y_face_value_mps.to_numpy()
        expected_wall_faces = np.zeros(
            (2, int(self.fluid.nx), int(self.fluid.nz), 3),
            dtype=np.float64,
        )
        inlet, outlet = self._boundary_fluxes(self.fluid, self._config)
        topology = self._topology_valid(report)
        return {
            **time_fields,
            "_transient_fluid_velocity_active_field_mps": active_velocity,
            "reported_force_solver_xyz_n": reported_total.tolist(),
            "beam_force_solver_xyz_n": beam.tolist(),
            "cylinder_pressure_force_solver_xyz_n": cylinder_pressure.tolist(),
            "cylinder_viscous_force_solver_xyz_n": cylinder_viscous.tolist(),
            "total_drag_per_span_n_per_m": float(per_span["total_drag_per_span_n_per_m"]),
            "total_lift_per_span_n_per_m": float(per_span["total_lift_per_span_n_per_m"]),
            "inlet_flux_m3ps": float(inlet),
            "outlet_flux_m3ps": float(outlet),
            "marker_layout_sha256": self.marker_layout_hash,
            "external_wall_face_max_residual_mps": self._external_y_face_ledger_residual(
                mask, ledger, expected_wall_faces
            ),
            "external_wall_face_full_component_mask_valid": bool(
                np.all(np.asarray(mask, dtype=np.int32) == 7)
            ),
            "base_cylinder_velocity_max_abs_mps": float(np.max(np.abs(velocity[base != 0]), initial=0.0)),
            "base_obstacle_crossing_normal_max_abs_mps": self._base_crossing_normal_speed(
                velocity, base, obstacle
            ),
            "beam_marker_no_slip_rms_mps": float(no_slip.l2_no_slip_residual_mps),
            "beam_marker_no_slip_max_mps": float(no_slip.max_no_slip_residual_mps),
            "beam_marker_valid_count": int(no_slip.valid_marker_count),
            "beam_marker_invalid_count": int(no_slip.invalid_marker_count),
            "projection_pressure_finite": bool(np.all(np.isfinite(self.fluid.pressure.to_numpy()))),
            "projection_cg_converged_all": bool(projection.get("cg_converged_all", False)),
            "projection_cg_breakdown_count": int(projection.get("cg_breakdown_count", -1)),
            "outlet_pressure_reference_rows_valid": bool(projection.get("pressure_outlet_operator_graph_prepared", False)),
            "hibm_topology_valid": topology,
            "pressure_nullspace_component_labels_converged": bool(
                projection.get(
                    "pressure_nullspace_component_labels_converged",
                    False,
                )
            ),
            "pressure_nullspace_component_overflow": bool(
                projection.get("pressure_nullspace_component_overflow", True)
            ),
            "hibm_pressure_reachability_converged": bool(
                projection.get("hibm_pressure_reachability_converged", False)
            ),
            "hibm_pressure_reachability_valid": bool(
                projection.get("hibm_pressure_reachability_valid", False)
            ),
            "hibm_pressure_component_labels_converged": bool(
                projection.get("hibm_pressure_component_labels_converged", False)
            ),
            "cg_unreached_component_overflow": bool(
                projection.get("cg_unreached_component_overflow", True)
            ),
            "projection_pressure_solve_failed": bool(
                projection.get("pressure_solve_failed", True)
            ),
            "projection_physical_failure": bool(
                projection.get("pressure_projection_physical_failure", True)
            ),
            "hibm_marker_mac_q_cycle_trace_json": marker_mac_q_cycle_trace_json,
            "expected_marker_count": int(
                2 * int(self._marker_counts[0]) + int(self._marker_counts[1])
            ),
            "velocity_dirichlet_authority_registered": bool(
                report.velocity_dirichlet.get("authority_registered", False)
            ),
            "velocity_dirichlet_authority_sealed": bool(
                report.velocity_dirichlet.get("authority_sealed", False)
            ),
            "velocity_dirichlet_invariant_violation_count": int(
                self._velocity_invariant_violations(report.velocity_dirichlet)
            ),
            "projection_cg_preconditioner_requested": str(projection.get("cg_preconditioner_requested", "unavailable")),
            "projection_cg_preconditioner_effective": str(projection.get("cg_preconditioner_effective", "unavailable")),
            "projection_cg_fallback_count": int(projection.get("cg_multigrid_to_jacobi_fallback_count", -1)),
            "projection_rhs_nonzero": bool(projection.get("projection_rhs_nonzero", False)),
            "cg_nonzero_rhs_project_calls": int(projection.get("cg_nonzero_rhs_project_calls", -1)),
            "cg_nonzero_rhs_preconditioner_requested": str(
                projection.get("cg_nonzero_rhs_preconditioner_requested", "unavailable")
            ),
            "cg_nonzero_rhs_preconditioner_effective": str(
                projection.get("cg_nonzero_rhs_preconditioner_effective", "unavailable")
            ),
            "cg_nonzero_rhs_multigrid_to_jacobi_fallback_count": int(
                projection.get("cg_nonzero_rhs_multigrid_to_jacobi_fallback_count", -1)
            ),
        }

    def initialization_audit(self) -> dict[str, Any]:
        report = self._initialization_report or self._last_report
        if report is None:
            raise RuntimeError("FAIL_INIT_AUDIT: initialization assembly was not observed")
        base = np.asarray(self.fluid.hibm_base_obstacle.to_numpy(), dtype=np.int32)
        dynamic = np.asarray(self.fluid.obstacle.to_numpy(), dtype=np.int32)
        base_exact = bool(np.array_equal(base, self._canonical_cylinder_mask))
        x_min, y_min, z_min = self._beam_box_min
        x_max, y_max, z_max = self._beam_box_max
        dx, dy, dz = self._spacing_xyz_m
        nx, ny, nz = (int(value) for value in self._config.grid_nodes)
        x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
        y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
        z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
        expected_beam = (
            (x[:, None, None] >= x_min)
            & (x[:, None, None] <= x_max)
            & (y[None, :, None] >= y_min)
            & (y[None, :, None] <= y_max)
            & (z[None, None, :] >= z_min)
            & (z[None, None, :] <= z_max)
        )
        beam_cell_intersection = (
            ((x + 0.5 * dx)[:, None, None] >= x_min)
            & ((x - 0.5 * dx)[:, None, None] <= x_max)
            & ((y + 0.5 * dy)[None, :, None] >= y_min)
            & ((y - 0.5 * dy)[None, :, None] <= y_max)
            & ((z + 0.5 * dz)[None, None, :] >= z_min)
            & ((z - 0.5 * dz)[None, None, :] <= z_max)
        )
        expected_beam_only = expected_beam & (base == 0)
        dynamic_only = (dynamic != 0) & (base == 0)
        unexpected_obstacle = dynamic_only & ~beam_cell_intersection
        expected_beam_complete = bool(np.all(dynamic[expected_beam_only] != 0))
        union_connected = self._connected_yz(np.any(dynamic != 0, axis=0))
        topology = self._topology_valid(report)
        beam_complete = bool(
            expected_beam_complete
            and np.count_nonzero(dynamic_only) > 0
            and report.internal_obstacle_cell_count > 0
            and np.count_nonzero(unexpected_obstacle) == 0
        )
        return {
            "cylinder_connected": base_exact and bool(np.any(base != 0)),
            "beam_connected": bool(np.any(dynamic_only)) and union_connected,
            "beam_interior_mask_complete": beam_complete,
            "obstacle_union_single_component": union_connected,
            "no_sealed_fluid_pocket": topology,
            "zero_load_fields_finite": bool(
                np.all(np.isfinite(self.fluid.velocity.to_numpy()))
                and np.all(np.isfinite(self.fluid.pressure.to_numpy()))
                and np.all(np.isfinite(self.solid.external_force_n.to_numpy()))
            ),
            "hibm_base_obstacle_established": bool(
                getattr(self.fluid, "_hibm_base_obstacle_initialized", False) and base_exact
            ),
            "hibm_topology_valid": topology,
            "marker_counts": tuple(int(value) for value in self._marker_counts),
            "base_cylinder_mask_exact": base_exact,
            "dynamic_beam_only_cell_count": int(np.count_nonzero(dynamic_only)),
            "internal_obstacle_cell_count": int(report.internal_obstacle_cell_count),
            "expected_beam_only_cell_count": int(np.count_nonzero(expected_beam_only)),
            "expected_beam_only_complete": expected_beam_complete,
            "unexpected_obstacle_outside_beam_or_cylinder_cell_count": int(
                np.count_nonzero(unexpected_obstacle)
            ),
        }

    def assert_fixed_markers(self) -> None:
        from simulation_core.coupling.hibm_mpm import (
            capture_marker_interface_state,
            marker_layout_identity,
        )

        if marker_layout_identity(
            self.markers, reference_positions_m=self._marker_reference_positions
        ) != self.marker_layout_hash:
            raise RuntimeError("fixed-fluid marker identity changed")
        state = capture_marker_interface_state(self.markers)
        for key, expected in self._fixed_marker_state.items():
            if not np.array_equal(np.asarray(state[key]), expected):
                raise RuntimeError(f"fixed-fluid marker state changed: {key}")
        regions = self.markers.region_id.to_numpy()[: int(self.markers.marker_count)]
        if not np.array_equal(regions, self._fixed_marker_regions):
            raise RuntimeError("fixed-fluid marker region layout changed")
        if self._geometry_metadata(state["_marker_geometry"]) != self._fixed_marker_geometry:
            raise RuntimeError("fixed-fluid marker geometry metadata changed")

    def final_arrays(self) -> dict[str, Any]:
        return {
            "fluid_velocity_mps": self.fluid.velocity.to_numpy(),
            "fluid_pressure_pa": self.fluid.pressure.to_numpy(),
            "solid_x_m": self.solid.x.to_numpy(),
        }


class _CoupledPreflightRuntime:
    """Thin strict-CUDA adapter around the sole generic-FSI production entry."""

    effective_arch = "cuda"

    def __init__(self, config: Mapping[str, Any]) -> None:
        effective = normalize_component_effective_config(config)
        if effective["mode"] != "coupled-preflight":
            raise ValueError("coupled runtime requires coupled-preflight mode")
        self.effective_config = MappingProxyType(effective)
        self._config = effective

    def run(self) -> dict[str, Any]:
        from cases.turek_hron_fsi import TurekHronFsiConfig, run_turek_hron_fsi
        from simulation_core.diagnostics.runtime import init_taichi

        init_taichi(_strict_cuda_runtime())
        fields = TurekHronFsiConfig.__dataclass_fields__
        summary = run_turek_hron_fsi(
            TurekHronFsiConfig(
                **{key: value for key, value in self._config.items() if key in fields}
            ),
            output_dir=None,
        )
        self.taichi_runtime_identity = _measured_taichi_runtime_identity()
        history = list(summary["history"])
        if not history:
            raise RuntimeError("coupled preflight returned no accepted history")
        try:
            fluid_accepted_time_s = sum(
                float(row["fluid_macro_accepted_time_s"]) for row in history
            )
            solid_accepted_time_s = sum(
                float(row["solid_macro_accepted_time_s"]) for row in history
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "coupled preflight omitted accepted-time ledger fields"
            ) from error
        expected_time_s = int(summary["completed_steps"]) * float(self._config["dt_s"])
        if not (
            math.isclose(
                fluid_accepted_time_s,
                expected_time_s,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                solid_accepted_time_s,
                expected_time_s,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise RuntimeError("coupled preflight history does not close accepted time")
        expected_case_config = {
            key: self._config[key] for key in TurekHronFsiConfig.__dataclass_fields__
        }
        if summary.get("config") != expected_case_config:
            raise RuntimeError("coupled preflight effective configuration changed")
        marker_hash = str(summary.get("marker_layout_sha256", ""))
        if (
            summary.get("marker_layout_identity_verified") is not True
            or len(marker_hash) != 64
            or any(character not in "0123456789abcdef" for character in marker_hash)
        ):
            raise RuntimeError("coupled preflight marker layout was not verified")
        return {
            "generic_runtime_completed_steps": int(summary["generic_runtime_completed_steps"]),
            "completed_steps": int(summary["completed_steps"]),
            "history": history,
            "solver_path": str(summary["solver_path"]),
            "accepted_time_s": fluid_accepted_time_s,
            "effective_arch": self.effective_arch,
            "effective_config": dict(self.effective_config),
            "marker_layout_sha256": marker_hash,
            "marker_layout_identity_verified": True,
            "taichi_runtime_identity": dict(self.taichi_runtime_identity),
        }


SolidOnlyRuntime = _SolidOnlyRuntime
FixedFluidRuntime = _FixedFluidRuntime
CoupledPreflightRuntime = _CoupledPreflightRuntime
