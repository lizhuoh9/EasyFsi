"""Public marker-MAC Q/P adapter for sharp HIBM-MPM fluid projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .marker_mac_constraint import HibmMpmMarkerMacConstraintOperator

if TYPE_CHECKING:
    from simulation_core.fluids.solver import CartesianFluidSolver
    from .core import HibmMpmSurfaceMarkers


class HibmMpmMarkerMacConstraintProjector:
    """Bind generic fluid Q/P hooks to one marker-space HIBM constraint owner."""

    def __init__(
        self,
        *,
        markers: HibmMpmSurfaceMarkers,
        operator: HibmMpmMarkerMacConstraintOperator,
        max_iterations: int,
        absolute_tolerance_mps: float,
        primary_region_id: int,
        secondary_region_id: int,
    ) -> None:
        self.markers_owner = markers
        self.operator = operator
        self.max_iterations = int(max_iterations)
        self.absolute_tolerance_mps = float(absolute_tolerance_mps)
        self.primary_region_id = int(primary_region_id)
        self.secondary_region_id = int(secondary_region_id)
        self._prepared_fluid: CartesianFluidSolver | None = None
        self._prepared_sampling_identity: Any | None = None
        self._prepared_component_face_valid_mask: Any | None = None
        self._prepared_sampling_obstacle: Any | None = None
        self._prepared_topology_generation = -1
        self._prepared_component_face_valid_mask_generation = -1
        self._pressure_solve_context: dict[str, object] = {}
        self._pressure_nullspace_fluid: CartesianFluidSolver | None = None
        self._pressure_actuated_component_mobility: Any | None = None
        self._pressure_nullspace_component_face_valid_mask: Any | None = None
        self._pressure_actuation_generation = -1
        self._pressure_nullspace_topology_generation = -1
        self._pressure_nullspace_component_face_valid_mask_generation = -1

    @property
    def last_prepared_sampling_identity(self) -> Any | None:
        return self._prepared_sampling_identity

    @property
    def last_component_face_valid_mask(self) -> Any | None:
        return self._prepared_component_face_valid_mask

    @property
    def last_sampling_obstacle(self) -> Any | None:
        return self._prepared_sampling_obstacle

    def _current_generations(
        self,
        fluid: CartesianFluidSolver,
    ) -> tuple[int, int]:
        return (
            int(fluid.hibm_reachability_revision),
            int(fluid.velocity_dirichlet_component_ledger_generation),
        )

    def _require_prepared_fluid(self) -> CartesianFluidSolver:
        if self._prepared_fluid is None or self._prepared_sampling_identity is None:
            raise RuntimeError(
                "pre-projection velocity transaction has not been prepared"
            )
        return self._prepared_fluid

    def _clear_pressure_nullspace_transaction_state(self) -> None:
        self._pressure_nullspace_fluid = None
        self._pressure_actuated_component_mobility = None
        self._pressure_nullspace_component_face_valid_mask = None
        self._pressure_actuation_generation = -1
        self._pressure_nullspace_topology_generation = -1
        self._pressure_nullspace_component_face_valid_mask_generation = -1

    def prepare_projection_transaction(
        self,
        *,
        fluid: CartesianFluidSolver,
        pressure_solve_context: Mapping[str, object],
    ) -> None:
        self._clear_pressure_nullspace_transaction_state()
        if not isinstance(pressure_solve_context, Mapping):
            raise TypeError("pressure_solve_context must be a mapping")
        component_face_valid_mask = (
            fluid.prepare_hibm_no_slip_component_face_valid_mask()
        )
        sampling_obstacle = fluid.hibm_no_slip_sampling_obstacle
        topology_generation, valid_mask_generation = self._current_generations(fluid)
        sampling_identity = self.markers_owner.prepare_no_slip_sampling_identity(
            obstacle_field=sampling_obstacle,
            component_face_valid_mask=component_face_valid_mask,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=valid_mask_generation,
        )
        self.operator.prepare(
            markers=self.markers_owner,
            fluid=fluid,
            component_face_valid_mask=component_face_valid_mask,
            primary_region_id=self.primary_region_id,
            secondary_region_id=self.secondary_region_id,
            prepared_sampling_identity=sampling_identity,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=valid_mask_generation,
        )
        self._prepared_fluid = fluid
        self._prepared_sampling_identity = sampling_identity
        self._prepared_component_face_valid_mask = component_face_valid_mask
        self._prepared_sampling_obstacle = sampling_obstacle
        self._prepared_topology_generation = topology_generation
        self._prepared_component_face_valid_mask_generation = valid_mask_generation
        self._pressure_solve_context = {
            str(key): value for key, value in pressure_solve_context.items()
        }

    def solve_projection_transaction(self) -> None:
        fluid = self._require_prepared_fluid()
        topology_generation, valid_mask_generation = self._current_generations(fluid)
        self.operator.solve_device(
            max_iterations=self.max_iterations,
            absolute_tolerance_mps=self.absolute_tolerance_mps,
            component_face_valid_mask=fluid.hibm_no_slip_component_face_valid_mask,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=valid_mask_generation,
            obstacle_field=fluid.hibm_no_slip_sampling_obstacle,
        )

    def commit_projection_transaction(self) -> Mapping[str, object]:
        fluid = self._require_prepared_fluid()
        topology_generation, valid_mask_generation = self._current_generations(fluid)
        self.operator.commit_if_converged(
            fluid,
            component_face_valid_mask=fluid.hibm_no_slip_component_face_valid_mask,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=valid_mask_generation,
            obstacle_field=fluid.hibm_no_slip_sampling_obstacle,
        )
        report = asdict(self.operator.report())
        return {
            **report,
            "prepared": bool(report["prepared"]),
            "converged": bool(report["converged"]),
            "committed": bool(report["committed"]),
            "pressure_solve_context": dict(self._pressure_solve_context),
            "topology_generation": int(self._prepared_topology_generation),
            "component_face_valid_mask_generation": int(
                self._prepared_component_face_valid_mask_generation
            ),
        }

    def prepare_pressure_nullspace_transaction(
        self,
        *,
        fluid: CartesianFluidSolver,
        pressure_actuated_component_mobility: Any,
        component_face_valid_mask: Any,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
    ) -> None:
        self._clear_pressure_nullspace_transaction_state()
        prepared_fluid = self._require_prepared_fluid()
        if self.operator._phase != "committed":
            raise RuntimeError(
                "pressure nullspace prepare requires a committed marker Q transaction"
            )
        if fluid is not prepared_fluid:
            raise RuntimeError("pressure nullspace fluid owner changed")
        if component_face_valid_mask is not self._prepared_component_face_valid_mask:
            raise RuntimeError(
                "pressure nullspace component-face valid-mask owner changed"
            )
        if (
            pressure_actuated_component_mobility
            is not fluid.pressure_velocity_actuation_weight
        ):
            raise RuntimeError("pressure actuation weight owner changed")

        current_topology_generation, current_valid_mask_generation = (
            self._current_generations(fluid)
        )
        supplied_generations = (
            pressure_actuation_generation,
            topology_generation,
            component_face_valid_mask_generation,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or int(value) < 0
            for value in supplied_generations
        ):
            raise ValueError(
                "pressure nullspace generations must be non-negative integers"
            )
        if int(pressure_actuation_generation) != int(
            fluid.pressure_velocity_actuation_generation
        ):
            raise RuntimeError("pressure actuation generation changed")
        if int(topology_generation) != current_topology_generation or int(
            topology_generation
        ) != int(self._prepared_topology_generation):
            raise RuntimeError("pressure nullspace topology generation changed")
        if int(component_face_valid_mask_generation) != (
            current_valid_mask_generation
        ) or int(component_face_valid_mask_generation) != int(
            self._prepared_component_face_valid_mask_generation
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask generation changed"
            )

        self.operator.prepare_pressure_nullspace_transaction(
            fluid=fluid,
            pressure_actuated_component_mobility=(
                pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            pressure_actuation_generation=int(pressure_actuation_generation),
            topology_generation=int(topology_generation),
            component_face_valid_mask_generation=int(
                component_face_valid_mask_generation
            ),
        )
        self._pressure_nullspace_fluid = fluid
        self._pressure_actuated_component_mobility = (
            pressure_actuated_component_mobility
        )
        self._pressure_nullspace_component_face_valid_mask = (
            component_face_valid_mask
        )
        self._pressure_actuation_generation = int(pressure_actuation_generation)
        self._pressure_nullspace_topology_generation = int(topology_generation)
        self._pressure_nullspace_component_face_valid_mask_generation = int(
            component_face_valid_mask_generation
        )

    def _require_current_pressure_nullspace_transaction(
        self,
        *,
        component_face_valid_mask: Any | None = None,
    ) -> tuple[CartesianFluidSolver, Any, Any]:
        fluid = self._require_prepared_fluid()
        pressure_fluid = self._pressure_nullspace_fluid
        pressure_actuation_weight = self._pressure_actuated_component_mobility
        prepared_valid_mask = self._pressure_nullspace_component_face_valid_mask
        if (
            pressure_fluid is None
            or pressure_actuation_weight is None
            or prepared_valid_mask is None
        ):
            raise RuntimeError(
                "pressure constraint nullspace transaction is not prepared"
            )
        if self.operator._phase != "committed":
            raise RuntimeError(
                "pressure nullspace apply requires a committed marker Q transaction"
            )
        if fluid is not pressure_fluid:
            raise RuntimeError("pressure nullspace fluid owner changed")
        if prepared_valid_mask is not self._prepared_component_face_valid_mask:
            raise RuntimeError(
                "pressure nullspace component-face valid-mask owner changed"
            )
        if (
            component_face_valid_mask is not None
            and component_face_valid_mask is not prepared_valid_mask
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask owner changed"
            )
        if pressure_actuation_weight is not fluid.pressure_velocity_actuation_weight:
            raise RuntimeError("pressure actuation weight owner changed")

        topology_generation, valid_mask_generation = self._current_generations(fluid)
        if int(fluid.pressure_velocity_actuation_generation) != int(
            self._pressure_actuation_generation
        ):
            raise RuntimeError("pressure actuation generation changed")
        if topology_generation != int(self._pressure_nullspace_topology_generation):
            raise RuntimeError("pressure nullspace topology generation changed")
        if valid_mask_generation != int(
            self._pressure_nullspace_component_face_valid_mask_generation
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask generation changed"
            )
        return fluid, pressure_actuation_weight, prepared_valid_mask

    def project_pressure_actuated_grid_vector_to_marker_nullspace(
        self,
        *,
        input_velocity_mps: Any,
        output_velocity_mps: Any,
        max_iterations: int,
        absolute_tolerance_mps: float,
        component_face_valid_mask: Any,
    ) -> None:
        if isinstance(max_iterations, (bool, np.bool_)) or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if int(max_iterations) != max_iterations:
            raise ValueError("max_iterations must be a positive integer")
        if isinstance(absolute_tolerance_mps, (bool, np.bool_)):
            raise ValueError("absolute_tolerance_mps must be finite and positive")
        tolerance = float(absolute_tolerance_mps)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("absolute_tolerance_mps must be finite and positive")
        fluid, pressure_actuation_weight, prepared_valid_mask = (
            self._require_current_pressure_nullspace_transaction(
                component_face_valid_mask=component_face_valid_mask
            )
        )
        self.operator.apply_pressure_nullspace_transaction_device_only(
            input_face_correction=input_velocity_mps,
            output_face_correction=output_velocity_mps,
            fluid=fluid,
            pressure_actuated_component_mobility=pressure_actuation_weight,
            component_face_valid_mask=prepared_valid_mask,
            pressure_actuation_generation=int(self._pressure_actuation_generation),
            topology_generation=int(self._pressure_nullspace_topology_generation),
            component_face_valid_mask_generation=int(
                self._pressure_nullspace_component_face_valid_mask_generation
            ),
        )

    def finalize_pressure_nullspace_transaction(self) -> Mapping[str, object]:
        fluid, pressure_actuation_weight, prepared_valid_mask = (
            self._require_current_pressure_nullspace_transaction()
        )
        report = self.operator.finalize_pressure_nullspace_transaction(
            fluid=fluid,
            pressure_actuated_component_mobility=pressure_actuation_weight,
            component_face_valid_mask=prepared_valid_mask,
            pressure_actuation_generation=int(self._pressure_actuation_generation),
            topology_generation=int(self._pressure_nullspace_topology_generation),
            component_face_valid_mask_generation=int(
                self._pressure_nullspace_component_face_valid_mask_generation
            ),
            absolute_tolerance_mps=float(self.absolute_tolerance_mps),
        )
        scalar_report = asdict(report)
        if int(scalar_report["pressure_actuation_generation"]) != int(
            self._pressure_actuation_generation
        ):
            raise RuntimeError("pressure nullspace report actuation generation changed")
        return {
            **scalar_report,
            "topology_generation": int(
                self._pressure_nullspace_topology_generation
            ),
            "component_face_valid_mask_generation": int(
                self._pressure_nullspace_component_face_valid_mask_generation
            ),
        }

    def terminal_no_slip_sampling_inputs(
        self,
        *,
        fluid: CartesianFluidSolver,
    ) -> tuple[Any, Any, Any, int, int]:
        """Return the current prepared sampling identity or fail closed."""

        if self.markers_owner is None or fluid is not self._require_prepared_fluid():
            raise RuntimeError("terminal no-slip sampler marker/fluid owner changed")
        if self.operator._phase != "committed":
            raise RuntimeError("terminal no-slip sampling requires committed marker Q")
        identity = self._prepared_sampling_identity
        valid_mask = self._prepared_component_face_valid_mask
        obstacle = self._prepared_sampling_obstacle
        topology_generation, valid_mask_generation = self._current_generations(fluid)
        if (
            identity is None
            or valid_mask is None
            or obstacle is None
            or topology_generation != self._prepared_topology_generation
            or valid_mask_generation
            != self._prepared_component_face_valid_mask_generation
        ):
            raise RuntimeError("terminal no-slip sampling identity is stale")
        return (
            identity,
            valid_mask,
            obstacle,
            topology_generation,
            valid_mask_generation,
        )
