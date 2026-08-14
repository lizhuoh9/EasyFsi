from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulation_core import HibmMpmSharpCouplingState, TaichiRuntimeConfig
from simulation_core.coupling.hibm_mpm.interface_state import (
    capture_marker_interface_state,
    marker_trial_state,
    marker_velocity_state,
    restore_marker_interface_state,
)
from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingReport,
    FsiStepContext,
    FsiTrialResult,
)


@dataclass
class SquidSharpFsiRuntime:
    """Adapt Squid sharp callbacks to the canonical FSI runtime contract."""

    simulator: Any
    solid_mpm: Any
    sharp_coupling_state: Any
    prepare_step: Callable[[FsiStepContext], None]
    evaluate_trial_once: Callable[[FsiStepContext], Any]
    commit_trial: Callable[
        [FsiStepContext, Any, FsiCouplingReport],
        Mapping[str, Any],
    ]
    publish_trial: Callable[[FsiStepContext, Mapping[str, Any]], None]
    finalize: Callable[[], Mapping[str, Any]]
    _step_base_marker_state: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _step_base_pressure_gradient: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _step_transaction_ready: bool = field(default=False, init=False, repr=False)

    def begin_step(self, context: FsiStepContext) -> np.ndarray:
        self._clear_step_transaction()
        self.simulator.save_reduced_state()
        self.simulator.fluid.save_state()
        self.solid_mpm.save_state()
        marker_state = capture_marker_interface_state(
            self.sharp_coupling_state.markers
        )
        pressure_gradient = _capture_pressure_gradient_state(
            self.sharp_coupling_state
        )
        self._step_base_marker_state = marker_state
        self._step_base_pressure_gradient = pressure_gradient
        self._step_transaction_ready = True
        self.prepare_step(context)
        return marker_velocity_state(marker_state)

    def evaluate_trial(
        self,
        context: FsiStepContext,
        marker_velocity_guess_mps: np.ndarray,
    ) -> FsiTrialResult:
        marker_base, pressure_base = self._require_step_base_state()
        self._restore_step_base_state(marker_base, pressure_base)
        restore_marker_interface_state(
            self.sharp_coupling_state.markers,
            marker_trial_state(marker_base, marker_velocity_guess_mps),
        )
        sharp_report = self.evaluate_trial_once(context)
        candidate_state = capture_marker_interface_state(
            self.sharp_coupling_state.markers
        )
        return FsiTrialResult(
            marker_velocity_mps=marker_velocity_state(candidate_state),
            payload={"sharp_report": sharp_report},
        )

    def commit_step(
        self,
        context: FsiStepContext,
        trial: FsiTrialResult,
        coupling: FsiCouplingReport,
    ) -> Mapping[str, Any]:
        sharp_report = trial.payload.get("sharp_report")
        if sharp_report is None:
            raise RuntimeError(
                "canonical Squid sharp trial did not return a sharp report"
            )
        row = self.commit_trial(context, sharp_report, coupling)
        self._clear_step_transaction()
        return row

    def rollback_step(self, context: FsiStepContext) -> None:
        del context
        if not self._step_transaction_ready:
            self._clear_step_transaction()
            return
        try:
            marker_base, pressure_base = self._require_step_base_state()
            self._restore_step_base_state(marker_base, pressure_base)
        finally:
            self._clear_step_transaction()

    def publish_step(
        self,
        context: FsiStepContext,
        committed_row: Mapping[str, Any],
    ) -> None:
        self.publish_trial(context, committed_row)

    def finalize_run(self) -> Mapping[str, Any]:
        return self.finalize()

    def _require_step_base_state(
        self,
    ) -> tuple[dict[str, Any], np.ndarray]:
        if (
            not self._step_transaction_ready
            or self._step_base_marker_state is None
            or self._step_base_pressure_gradient is None
        ):
            raise RuntimeError("begin_step must be called before a Squid sharp trial")
        return self._step_base_marker_state, self._step_base_pressure_gradient

    def _clear_step_transaction(self) -> None:
        self._step_base_marker_state = None
        self._step_base_pressure_gradient = None
        self._step_transaction_ready = False

    def _restore_step_base_state(
        self,
        marker_base: dict[str, Any],
        pressure_base: np.ndarray,
    ) -> None:
        self.simulator.restore_reduced_state()
        self.simulator.fluid.restore_state()
        self.solid_mpm.restore_state()
        restore_marker_interface_state(
            self.sharp_coupling_state.markers,
            marker_base,
        )
        _restore_pressure_gradient_state(
            self.sharp_coupling_state,
            pressure_base,
        )


def squid_sharp_coupling_summary(
    report: FsiCouplingReport,
) -> dict[str, object]:
    """Map the canonical coupling report to Squid's sharp-row diagnostics."""

    return {
        "hibm_coupling_scheme": "marker_velocity_iqn_ils",
        "hibm_fsi_interface_unknown": "marker_velocity_mps",
        "hibm_fsi_coupling_accelerator": "iqn_ils",
        "hibm_fsi_coupling_iterations_used": int(report.iterations),
        "hibm_fsi_coupling_converged": bool(report.converged),
        "hibm_fsi_coupling_residual_source": (
            "canonical_marker_velocity_absolute_rms_mps"
        ),
        "hibm_fsi_coupling_relative_residual": float(report.relative_residual),
        "hibm_fsi_coupling_residual_l2_mps": float(
            report.absolute_residual_mps
        ),
        "hibm_fsi_coupling_residual_max_mps": float(
            report.max_marker_residual_mps
        ),
        "hibm_fsi_coupling_relative_residual_history": list(
            report.relative_residual_history
        ),
        "hibm_fsi_coupling_residual_history_mps": list(
            report.absolute_residual_history_mps
        ),
        "hibm_fsi_coupling_update_modes": list(report.update_modes),
    }


def _capture_pressure_gradient_state(sharp_coupling_state: Any) -> np.ndarray:
    count = int(sharp_coupling_state.markers.marker_count)
    field = sharp_coupling_state.marker_pressure_neumann_gradient_pa_per_m
    return np.asarray(field.to_numpy())[:count].copy()


def _restore_pressure_gradient_state(
    sharp_coupling_state: Any,
    state: np.ndarray,
) -> None:
    count = int(sharp_coupling_state.markers.marker_count)
    field = sharp_coupling_state.marker_pressure_neumann_gradient_pa_per_m
    full = field.to_numpy()
    array = np.asarray(state, dtype=full.dtype)
    expected_shape = tuple(full[:count].shape)
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            "sharp pressure-Neumann gradient state shape mismatch: "
            f"{tuple(array.shape)} != {expected_shape}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("sharp pressure-Neumann gradient state must be finite")
    full[:count] = array
    field.from_numpy(full)


def build_hibm_mpm_sharp_coupling_state(
    *,
    fluid,
    solid_mpm,
    runtime: TaichiRuntimeConfig | None,
) -> HibmMpmSharpCouplingState:
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    marker_count = int(getattr(solid_mpm, "particle_count"))
    if marker_count <= 0:
        raise ValueError("initialize solid_mpm particles before HIBM-MPM coupling")
    surface_region_id = getattr(solid_mpm, "region_id", None)
    if surface_region_id is None:
        surface_region_id = getattr(solid_mpm, "vertex_region_id", None)
    if surface_region_id is None:
        raise ValueError("solid_mpm must expose a Taichi surface region field")
    projection_triangle_indices = getattr(solid_mpm, "face_indices", None)
    projection_triangle_count = int(getattr(solid_mpm, "face_count", 0) or 0)
    projection_triangle_capacity = (
        projection_triangle_count
        if projection_triangle_indices is not None and projection_triangle_count > 0
        else None
    )
    coupling = HibmMpmSharpCouplingState(
        grid_nodes=fluid.grid.grid_nodes,
        bounds_min_m=fluid.grid.bounds_min_m,
        bounds_max_m=fluid.grid.bounds_max_m,
        marker_capacity=marker_count,
        projection_triangle_capacity=projection_triangle_capacity,
        runtime=runtime,
    )
    projection_kwargs = {}
    if projection_triangle_indices is not None and projection_triangle_count > 0:
        projection_kwargs = {
            "projection_triangle_indices": projection_triangle_indices,
            "projection_triangle_count": projection_triangle_count,
        }
    coupling.load_markers_from_surface_fields(
        solid_mpm.x,
        solid_mpm.surface_normal,
        solid_mpm.area_weight_m2,
        surface_region_id,
        marker_count=marker_count,
        surface_velocity_mps=solid_mpm.v,
        **projection_kwargs,
    )
    return coupling
