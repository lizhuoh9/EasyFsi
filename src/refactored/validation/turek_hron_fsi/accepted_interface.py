"""Pure NumPy validation for accepted Turek--Hron interface records."""

from __future__ import annotations

from typing import Any, Collection, Mapping, Sequence

import numpy as np


MARKER_FORCE_SUM_RELATIVE_TOLERANCE = 32.0 * np.finfo(np.float32).eps
TOTAL_FORCE_CLOSURE_RELATIVE_TOLERANCE = 32.0 * np.finfo(np.float64).eps
STATIC_ARRAYS = (
    "marker_reference_position_m",
    "marker_fixed_area_m2",
    "marker_region_id",
    "marker_order",
)
RAGGED_ARRAYS = (
    "coupling_relative_residual_history",
    "coupling_absolute_residual_history_mps",
    "coupling_update_mode_history",
    "iqn_rank_history",
    "iqn_condition_number_history",
    "iqn_fallback_reason_history",
    "iqn_update_limited_history",
)
NAN_PADDED_ARRAYS = frozenset(
    {
        "coupling_relative_residual_history",
        "coupling_absolute_residual_history_mps",
        "iqn_condition_number_history",
    }
)
REQUIRED_ARRAYS = frozenset(
    {
        "accepted_step",
        "accepted_time_s",
        "marker_reference_position_m",
        "marker_current_position_m",
        "marker_material_displacement_m",
        "marker_velocity_mps",
        "marker_normal",
        "marker_fixed_area_m2",
        "marker_region_id",
        "marker_order",
        "marker_force_pre_solid_n",
        "point_a_displacement_turek_xy_m",
        "point_a_velocity_turek_xy_mps",
        "beam_force_solver_xyz_n",
        "cylinder_pressure_force_solver_xyz_n",
        "cylinder_viscous_force_solver_xyz_n",
        "total_force_solver_xyz_n",
        "coupling_trial_count",
        "coupling_rejected_trial_count",
        "pressure_cg_iterations_total",
        "pressure_matvec_count_total",
        "fluid_solve_count",
        "solid_macro_solve_count",
        "mpm_substeps_executed_total",
        "iqn_fallback_count",
        *RAGGED_ARRAYS,
    }
)
UNITS = {
    "accepted_time_s": "s",
    "marker_reference_position_m": "m",
    "marker_current_position_m": "m",
    "marker_material_displacement_m": "m",
    "marker_velocity_mps": "m/s",
    "marker_fixed_area_m2": "m^2",
    "marker_force_pre_solid_n": "N",
    "point_a_displacement_turek_xy_m": "m",
    "point_a_velocity_turek_xy_mps": "m/s",
    "beam_force_solver_xyz_n": "N",
    "cylinder_pressure_force_solver_xyz_n": "N",
    "cylinder_viscous_force_solver_xyz_n": "N",
    "total_force_solver_xyz_n": "N",
    "coupling_absolute_residual_history_mps": "m/s",
}

_FLOAT_ARRAYS = frozenset(
    {
        "accepted_time_s",
        "marker_reference_position_m",
        "marker_current_position_m",
        "marker_material_displacement_m",
        "marker_velocity_mps",
        "marker_normal",
        "marker_fixed_area_m2",
        "marker_force_pre_solid_n",
        "point_a_displacement_turek_xy_m",
        "point_a_velocity_turek_xy_mps",
        "beam_force_solver_xyz_n",
        "cylinder_pressure_force_solver_xyz_n",
        "cylinder_viscous_force_solver_xyz_n",
        "total_force_solver_xyz_n",
        "coupling_relative_residual_history",
        "coupling_absolute_residual_history_mps",
        "iqn_condition_number_history",
    }
)
_INTEGER_ARRAYS = frozenset(
    {
        "accepted_step",
        "marker_region_id",
        "marker_order",
        "coupling_trial_count",
        "coupling_rejected_trial_count",
        "pressure_cg_iterations_total",
        "pressure_matvec_count_total",
        "fluid_solve_count",
        "solid_macro_solve_count",
        "mpm_substeps_executed_total",
        "iqn_fallback_count",
        "iqn_rank_history",
    }
)
_BOOLEAN_ARRAYS = frozenset({"iqn_update_limited_history"})
_UNICODE_ARRAYS = frozenset(
    {"coupling_update_mode_history", "iqn_fallback_reason_history"}
)


def _force_vector_closes(
    observed: Any,
    expected: Any,
    *,
    relative_tolerance: float,
) -> bool:
    observed_array = np.asarray(observed, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    error = float(np.linalg.norm(observed_array - expected_array))
    scale = float(np.linalg.norm(expected_array))
    return error == 0.0 if scale == 0.0 else error / scale <= relative_tolerance


class AcceptedRecordValidator:
    """Fail closed on record schema, physics identities, and static drift."""

    def __init__(
        self,
        *,
        required_arrays: Collection[str],
        nan_padded_arrays: Collection[str],
        static_arrays: Collection[str],
        expected_dt_s: float | None,
        expected_solid_substeps: int | None,
    ) -> None:
        self.required_arrays = frozenset(required_arrays)
        self.nan_padded_arrays = frozenset(nan_padded_arrays)
        self.static_names = tuple(static_arrays)
        self.expected_dt_s = expected_dt_s
        self.expected_solid_substeps = expected_solid_substeps
        self.inferred_dt_s: float | None = None
        self.static_arrays: dict[str, np.ndarray] | None = None

    def validate(
        self,
        record: Mapping[str, Any],
        *,
        accepted_count: int,
        finalized: bool,
    ) -> dict[str, np.ndarray]:
        if finalized:
            raise RuntimeError("writer is already finalized")
        if set(record) != self.required_arrays:
            missing = sorted(self.required_arrays - set(record))
            extra = sorted(set(record) - self.required_arrays)
            raise ValueError(
                f"accepted record schema mismatch; missing={missing}, extra={extra}"
            )
        arrays = {
            name: np.asarray(value).copy() for name, value in record.items()
        }
        expected_kinds = (
            (_FLOAT_ARRAYS, "f", "floating"),
            (_INTEGER_ARRAYS, "iu", "integral"),
            (_BOOLEAN_ARRAYS, "b", "boolean"),
            (_UNICODE_ARRAYS, "U", "Unicode"),
        )
        for names, kinds, description in expected_kinds:
            for name in names:
                if arrays[name].dtype.kind not in kinds:
                    raise ValueError(f"{name} must use a {description} dtype")
        for name, array in arrays.items():
            if array.dtype.hasobject:
                raise ValueError(f"{name} must not use object dtype")
            if np.issubdtype(array.dtype, np.number):
                if name in self.nan_padded_arrays:
                    if np.any(np.isinf(array)):
                        raise ValueError(f"{name} must not contain infinity")
                elif not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must be finite")

        self._validate_step_and_time(arrays, accepted_count=accepted_count)
        self._validate_marker_fields(arrays)
        self._validate_force_fields(arrays)
        self._validate_work_and_histories(arrays)
        self._validate_static_arrays(arrays)
        return arrays

    def _validate_step_and_time(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        accepted_count: int,
    ) -> int:
        raw_step = arrays["accepted_step"]
        if raw_step.shape != () or not np.issubdtype(raw_step.dtype, np.integer):
            raise ValueError("accepted_step must be an integer scalar")
        step = int(raw_step)
        if step != int(accepted_count) + 1:
            raise ValueError("accepted steps must be continuous and one-based")
        raw_time = arrays["accepted_time_s"]
        if raw_time.shape != () or not np.issubdtype(raw_time.dtype, np.floating):
            raise ValueError("accepted_time_s must be a floating scalar")
        time_s = float(raw_time)
        if not np.isfinite(time_s) or time_s <= 0.0:
            raise ValueError("accepted_time_s must be finite and positive")
        if self.inferred_dt_s is None:
            self.inferred_dt_s = time_s
        if self.expected_dt_s is not None and not np.isclose(
            time_s,
            step * self.expected_dt_s,
            rtol=0.0,
            atol=max(1.0e-15, 1.0e-12 * self.expected_dt_s),
        ):
            raise ValueError("accepted physical time does not match frozen dt_s")
        if not np.isclose(
            time_s,
            step * self.inferred_dt_s,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("accepted physical time must be continuous")
        return step

    @staticmethod
    def _validate_marker_fields(arrays: Mapping[str, np.ndarray]) -> int:
        reference = arrays["marker_reference_position_m"]
        if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] != 3:
            raise ValueError("marker reference positions must have shape (n, 3)")
        marker_count = int(reference.shape[0])
        for name in (
            "marker_current_position_m",
            "marker_material_displacement_m",
            "marker_velocity_mps",
            "marker_normal",
            "marker_force_pre_solid_n",
        ):
            if arrays[name].shape != (marker_count, 3):
                raise ValueError(f"{name} must have shape (marker_count, 3)")
        for name in ("marker_fixed_area_m2", "marker_region_id", "marker_order"):
            if arrays[name].shape != (marker_count,):
                raise ValueError(f"{name} must have shape (marker_count,)")
        if not np.array_equal(
            arrays["marker_material_displacement_m"],
            arrays["marker_current_position_m"] - reference,
        ):
            raise ValueError("marker material displacement does not close")
        if np.any(arrays["marker_fixed_area_m2"] <= 0.0):
            raise ValueError("marker fixed areas must be positive")
        if not np.allclose(
            np.linalg.norm(arrays["marker_normal"], axis=1),
            1.0,
            rtol=1.0e-6,
            atol=1.0e-8,
        ):
            raise ValueError("marker normals must be unit length")
        if not np.issubdtype(arrays["marker_region_id"].dtype, np.integer):
            raise ValueError("marker_region_id must be integral")
        if not np.array_equal(
            arrays["marker_order"],
            np.arange(marker_count, dtype=arrays["marker_order"].dtype),
        ):
            raise ValueError("marker_order must be canonical and continuous")
        return marker_count

    @staticmethod
    def _validate_force_fields(
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        for name, shape in (
            ("point_a_displacement_turek_xy_m", (2,)),
            ("point_a_velocity_turek_xy_mps", (2,)),
            ("beam_force_solver_xyz_n", (3,)),
            ("cylinder_pressure_force_solver_xyz_n", (3,)),
            ("cylinder_viscous_force_solver_xyz_n", (3,)),
            ("total_force_solver_xyz_n", (3,)),
        ):
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not _force_vector_closes(
            arrays["marker_force_pre_solid_n"].sum(axis=0),
            arrays["beam_force_solver_xyz_n"],
            relative_tolerance=MARKER_FORCE_SUM_RELATIVE_TOLERANCE,
        ):
            raise ValueError("marker force sum does not match beam force")
        if not _force_vector_closes(
            arrays["total_force_solver_xyz_n"],
            arrays["beam_force_solver_xyz_n"]
            + arrays["cylinder_pressure_force_solver_xyz_n"]
            + arrays["cylinder_viscous_force_solver_xyz_n"],
            relative_tolerance=TOTAL_FORCE_CLOSURE_RELATIVE_TOLERANCE,
        ):
            raise ValueError("total force does not close")

    def _validate_work_and_histories(
        self,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        integer_scalars: dict[str, int] = {}
        for name in (
            "coupling_trial_count",
            "coupling_rejected_trial_count",
            "pressure_cg_iterations_total",
            "pressure_matvec_count_total",
            "fluid_solve_count",
            "solid_macro_solve_count",
            "mpm_substeps_executed_total",
            "iqn_fallback_count",
        ):
            value = arrays[name]
            if value.shape != () or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"{name} must be an integer scalar")
            integer_scalars[name] = int(value)
        trials = integer_scalars["coupling_trial_count"]
        if trials <= 0 or any(value < 0 for value in integer_scalars.values()):
            raise ValueError("work-ledger counts must be nonnegative with trials > 0")
        if integer_scalars["coupling_rejected_trial_count"] != trials - 1:
            raise ValueError("rejected coupling trials do not close")
        if (
            integer_scalars["fluid_solve_count"] != trials
            or integer_scalars["solid_macro_solve_count"] != trials
        ):
            raise ValueError("fluid/solid solve counts do not match coupling trials")
        if (
            self.expected_solid_substeps is not None
            and integer_scalars["mpm_substeps_executed_total"]
            != trials * self.expected_solid_substeps
        ):
            raise ValueError("MPM work does not match frozen solid substeps")
        if (
            integer_scalars["pressure_matvec_count_total"]
            < integer_scalars["pressure_cg_iterations_total"]
        ):
            raise ValueError("pressure matvec count is smaller than CG iterations")

        relative = arrays["coupling_relative_residual_history"]
        absolute = arrays["coupling_absolute_residual_history_mps"]
        if relative.ndim != 1 or absolute.ndim != 1:
            raise ValueError("coupling residual histories must be one-dimensional")
        for name, value in (("relative", relative), ("absolute", absolute)):
            if value.size < trials or not np.all(np.isfinite(value[:trials])):
                raise ValueError(f"{name} residual history is incomplete")
            if value.size > trials and not np.all(np.isnan(value[trials:])):
                raise ValueError(f"{name} residual padding must be NaN")
        update_count = trials - 1
        ranks = arrays["iqn_rank_history"]
        conditions = arrays["iqn_condition_number_history"]
        modes = arrays["coupling_update_mode_history"]
        fallbacks = arrays["iqn_fallback_reason_history"]
        limited = arrays["iqn_update_limited_history"]
        if not (
            ranks.ndim
            == conditions.ndim
            == modes.ndim
            == fallbacks.ndim
            == limited.ndim
            == 1
            and ranks.size
            == conditions.size
            == modes.size
            == fallbacks.size
            == limited.size
            and ranks.size >= update_count
        ):
            raise ValueError("IQN histories must be aligned one-dimensional arrays")
        if np.any(ranks[:update_count] < 0) or np.any(ranks[update_count:] != -1):
            raise ValueError("IQN rank padding must use -1 after valid updates")
        if np.any(np.isinf(conditions[:update_count])) or (
            conditions.size > update_count
            and not np.all(np.isnan(conditions[update_count:]))
        ):
            raise ValueError("IQN condition padding must use NaN")
        if limited.size > update_count and np.any(limited[update_count:]):
            raise ValueError("IQN update-limited padding must be false")
        if np.any(modes[:update_count] == "") or np.any(
            modes[update_count:] != ""
        ):
            raise ValueError("coupling update-mode padding must use empty strings")
        if np.any(fallbacks[:update_count] == "") or np.any(
            fallbacks[update_count:] != ""
        ):
            raise ValueError("IQN fallback padding must use empty strings")
        observed_fallback_count = int(
            np.count_nonzero(fallbacks[:update_count] != "none")
        )
        if integer_scalars["iqn_fallback_count"] != observed_fallback_count:
            raise ValueError("IQN fallback count does not match fallback history")

    def _validate_static_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        static = {name: arrays[name] for name in self.static_names}
        if self.static_arrays is None:
            self.static_arrays = {
                name: value.copy() for name, value in static.items()
            }
        elif any(
            not np.array_equal(value, self.static_arrays[name])
            for name, value in static.items()
        ):
            raise ValueError("static marker arrays changed")


def stack_accepted_arrays(
    name: str,
    values: Sequence[np.ndarray],
    *,
    ragged_arrays: Collection[str],
    nan_padded_arrays: Collection[str],
) -> dict[str, np.ndarray]:
    """Stack equal shapes or pad an explicitly ragged one-dimensional field."""

    if len({value.shape for value in values}) == 1:
        return {name: np.stack(values, axis=0)}
    if name not in ragged_arrays or any(value.ndim != 1 for value in values):
        raise ValueError(f"dynamic array shape changed for {name}")
    width = max(value.size for value in values)
    if name in nan_padded_arrays:
        fill = np.nan
    elif name == "iqn_rank_history":
        fill = -1
    elif np.issubdtype(values[0].dtype, np.str_):
        fill = ""
    else:
        fill = False if np.issubdtype(values[0].dtype, np.bool_) else 0
    padded = np.full((len(values), width), fill, dtype=np.result_type(*values))
    lengths = np.asarray([value.size for value in values], dtype=np.int64)
    for index, value in enumerate(values):
        padded[index, : value.size] = value
    return {name: padded, f"{name}_length": lengths}


__all__ = [
    "AcceptedRecordValidator",
    "MARKER_FORCE_SUM_RELATIVE_TOLERANCE",
    "NAN_PADDED_ARRAYS",
    "RAGGED_ARRAYS",
    "REQUIRED_ARRAYS",
    "STATIC_ARRAYS",
    "TOTAL_FORCE_CLOSURE_RELATIVE_TOLERANCE",
    "UNITS",
    "stack_accepted_arrays",
]
