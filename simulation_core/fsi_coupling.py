from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


ForceVector = tuple[float, ...]
INTERFACE_REACTION_SOLVER_CHOICES = ("aitken", "iqn_ils")
_IQN_ILS_HISTORY_LIMIT = 8


@dataclass(frozen=True)
class InterfaceReactionUpdate:
    force_n: ForceVector
    residual_n: ForceVector
    power_w: ForceVector
    passivity_limited: tuple[bool, ...]

    @property
    def residual_norm_n(self) -> float:
        return math.sqrt(sum(component * component for component in self.residual_n))


@dataclass(frozen=True)
class InterfaceReactionRelaxationState:
    previous_residual_n: ForceVector | None = None
    previous_velocity_mps: ForceVector | None = None
    relaxation: float = 1.0


@dataclass(frozen=True)
class InterfaceReactionStepUpdate:
    update: InterfaceReactionUpdate
    relaxation: float
    next_state: InterfaceReactionRelaxationState
    robin_impedance_force_n: ForceVector = ()


@dataclass(frozen=True)
class InterfaceReactionTargetEvaluation:
    target_force_n: ForceVector
    velocity_mps: ForceVector
    diagnostic_target_force_n: ForceVector | None = None
    payload: object | None = None


@dataclass(frozen=True)
class RegionPairInterfaceReactionTarget:
    primary_force_n: ForceVector
    secondary_force_n: ForceVector

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "primary_force_n",
            _force_vector3(self.primary_force_n, name="primary_force_n"),
        )
        object.__setattr__(
            self,
            "secondary_force_n",
            _force_vector3(self.secondary_force_n, name="secondary_force_n"),
        )


@dataclass(frozen=True)
class InterfaceReactionFixedPointResult:
    force_n: ForceVector
    iterations_used: int
    converged: bool
    residual_norm_n: float
    relaxation: float
    solver: str = "aitken"
    trial_force_history_n: tuple[ForceVector, ...] = ()
    target_force_history_n: tuple[ForceVector, ...] = ()
    residual_history_n: tuple[ForceVector, ...] = ()
    physical_target_force_history_n: tuple[ForceVector, ...] = ()
    physical_residual_history_n: tuple[ForceVector, ...] = ()
    diagnostic_target_force_history_n: tuple[ForceVector, ...] = ()
    diagnostic_residual_history_n: tuple[ForceVector, ...] = ()
    accepted_trial_index: int | None = None
    accepted_state_reusable: bool = False
    accepted_payload: object | None = None
    all_trials_rejected: bool = False
    zero_force_commit_blocked: bool = False
    fallback_force_source: str = ""
    rejected_trial_count: int = 0
    rejected_trial_backtrack_count: int = 0
    residual_growth_rejected_trial_count: int = 0
    max_residual_rejected_trial_count: int = 0
    trust_region_limited_update_count: int = 0
    trust_region_shrink_count: int = 0
    trust_region_growth_count: int = 0
    trust_region_rebound_backtrack_count: int = 0
    trust_region_rebound_stop_count: int = 0
    trust_region_rebound_stop_suppressed_count: int = 0
    residual_continuation_iteration_count: int = 0
    residual_continuation_rebound_secant_count: int = 0
    residual_continuation_rebound_secant_evaluation_extension_count: int = 0
    trust_region_effective_force_increment_n: float = math.inf
    iqn_ils_least_squares_update_count: int = 0
    interface_map_amplification_max: float = 0.0
    residual_jacobian_amplification_max: float = 0.0
    physical_interface_map_amplification_max: float = 0.0
    physical_residual_jacobian_amplification_max: float = 0.0
    diagnostic_interface_map_amplification_max: float = 0.0
    diagnostic_residual_jacobian_amplification_max: float = 0.0
    interface_map_amplification_sample_count: int = 0
    residual_jacobian_amplification_sample_count: int = 0
    physical_interface_map_amplification_sample_count: int = 0
    physical_residual_jacobian_amplification_sample_count: int = 0
    diagnostic_interface_map_amplification_sample_count: int = 0
    diagnostic_residual_jacobian_amplification_sample_count: int = 0
    target_map_relaxation: float = 1.0


@dataclass(frozen=True)
class ForceBalanceReport:
    residual_components_n: tuple[float, ...]
    residual_norm_n: float
    relative_error: float
    scale_n: float


def _force_vector(values: Sequence[float], *, name: str) -> ForceVector:
    try:
        vector = tuple(float(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain at least one force component") from exc
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric force components") from exc
    if len(vector) == 0:
        raise ValueError(f"{name} must contain at least one force component")
    if any(not math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain only finite force components")
    return vector


def _finite_float(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _force_vector3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = _force_vector(values, name=name)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 force components")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _same_length(lhs: ForceVector, rhs: ForceVector, *, lhs_name: str, rhs_name: str) -> None:
    if len(lhs) != len(rhs):
        raise ValueError(f"{lhs_name} and {rhs_name} must have the same length")


def _force_residual(*, target_force_n: ForceVector, committed_force_n: ForceVector) -> ForceVector:
    _same_length(
        committed_force_n,
        target_force_n,
        lhs_name="committed_force_n",
        rhs_name="target_force_n",
    )
    return tuple(
        target_value - committed_value
        for committed_value, target_value in zip(committed_force_n, target_force_n)
    )


def _force_norm(force_n: ForceVector) -> float:
    return math.sqrt(sum(component * component for component in force_n))


def _force_dot(lhs: ForceVector, rhs: ForceVector) -> float:
    _same_length(lhs, rhs, lhs_name="lhs", rhs_name="rhs")
    return float(sum(lhs_value * rhs_value for lhs_value, rhs_value in zip(lhs, rhs)))


def _force_difference(lhs: ForceVector, rhs: ForceVector) -> ForceVector:
    _same_length(lhs, rhs, lhs_name="lhs", rhs_name="rhs")
    return tuple(lhs_value - rhs_value for lhs_value, rhs_value in zip(lhs, rhs))


def _relaxed_target_map_force(
    *,
    trial_force_n: ForceVector,
    physical_target_force_n: ForceVector,
    target_map_relaxation: float,
) -> ForceVector:
    _same_length(
        trial_force_n,
        physical_target_force_n,
        lhs_name="trial_force_n",
        rhs_name="physical_target_force_n",
    )
    relaxation = float(target_map_relaxation)
    if not math.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
        raise ValueError("target_map_relaxation must be a finite number in (0, 1]")
    return tuple(
        trial_value + relaxation * (target_value - trial_value)
        for trial_value, target_value in zip(trial_force_n, physical_target_force_n)
    )


def _secant_amplification_stats(
    input_history: Sequence[ForceVector],
    output_history: Sequence[ForceVector],
) -> tuple[float, int]:
    history_count = min(len(input_history), len(output_history))
    max_amplification = 0.0
    sample_count = 0
    for index in range(1, history_count):
        input_delta_norm = _force_norm(
            _force_difference(input_history[index], input_history[index - 1])
        )
        if input_delta_norm <= 1.0e-30:
            continue
        output_delta_norm = _force_norm(
            _force_difference(output_history[index], output_history[index - 1])
        )
        amplification = output_delta_norm / input_delta_norm
        if math.isfinite(amplification):
            sample_count += 1
            max_amplification = max(max_amplification, amplification)
    return max_amplification, sample_count


def _force_linear_combination(
    columns: Sequence[ForceVector],
    coefficients: Sequence[float],
) -> ForceVector:
    if len(columns) == 0:
        raise ValueError("columns must contain at least one vector")
    if len(columns) != len(coefficients):
        raise ValueError("columns and coefficients must have the same length")
    dimension = len(columns[0])
    result = [0.0 for _ in range(dimension)]
    for column, coefficient in zip(columns, coefficients):
        if len(column) != dimension:
            raise ValueError("all columns must have the same length")
        for index, value in enumerate(column):
            result[index] += float(coefficient) * value
    return tuple(result)


def _force_is_finite(force_n: ForceVector) -> bool:
    return all(math.isfinite(component) for component in force_n)


def _validate_interface_reaction_solver(solver: str) -> str:
    solver_name = str(solver)
    if solver_name not in INTERFACE_REACTION_SOLVER_CHOICES:
        choices = ", ".join(INTERFACE_REACTION_SOLVER_CHOICES)
        raise ValueError(f"solver must be one of: {choices}")
    return solver_name


def robin_neumann_impedance_force(
    *,
    velocity_mps: Sequence[float],
    previous_velocity_mps: Sequence[float] | None,
    impedance_ns_per_m: float,
) -> ForceVector:
    """Return the Robin-Neumann impedance force opposing interface acceleration."""
    velocity = _force_vector(velocity_mps, name="velocity_mps")
    impedance = float(impedance_ns_per_m)
    if not math.isfinite(impedance) or impedance < 0.0:
        raise ValueError("impedance_ns_per_m must be a finite non-negative number")
    if impedance == 0.0 or previous_velocity_mps is None:
        return tuple(0.0 for _ in velocity)
    previous_velocity = _force_vector(
        previous_velocity_mps,
        name="previous_velocity_mps",
    )
    _same_length(
        previous_velocity,
        velocity,
        lhs_name="previous_velocity_mps",
        rhs_name="velocity_mps",
    )
    return tuple(
        -impedance * (velocity_value - previous_velocity_value)
        for velocity_value, previous_velocity_value in zip(velocity, previous_velocity)
    )


def _least_squares_coefficients(
    columns: Sequence[ForceVector],
    rhs: ForceVector,
) -> tuple[float, ...] | None:
    if len(columns) == 0:
        return None
    for column in columns:
        _same_length(column, rhs, lhs_name="column", rhs_name="rhs")
    dimension = len(rhs)
    max_column_norm = max((_force_norm(column) for column in columns), default=0.0)
    tolerance = 1.0e-12 * max(max_column_norm, _force_norm(rhs), 1.0)
    q_vectors: list[ForceVector] = []
    r_matrix: list[list[float]] = []
    retained_indices: list[int] = []

    for original_index, column in enumerate(columns):
        candidate = [float(value) for value in column]
        projections: list[float] = []
        for q_vector in q_vectors:
            projection = sum(q_value * value for q_value, value in zip(q_vector, candidate))
            projections.append(projection)
            for axis in range(dimension):
                candidate[axis] -= projection * q_vector[axis]
        diagonal = math.sqrt(sum(value * value for value in candidate))
        if diagonal <= tolerance:
            continue
        for row_index, projection in enumerate(projections):
            r_matrix[row_index].append(projection)
        r_matrix.append([0.0 for _ in retained_indices] + [diagonal])
        q_vectors.append(tuple(value / diagonal for value in candidate))
        retained_indices.append(original_index)

    if len(retained_indices) == 0:
        return None

    projected_rhs = [_force_dot(q_vector, rhs) for q_vector in q_vectors]
    retained_solution = [0.0 for _ in retained_indices]
    for row_index in range(len(retained_indices) - 1, -1, -1):
        diagonal = r_matrix[row_index][row_index]
        if abs(diagonal) <= tolerance:
            return None
        tail = sum(
            r_matrix[row_index][column_index] * retained_solution[column_index]
            for column_index in range(row_index + 1, len(retained_indices))
        )
        retained_solution[row_index] = (projected_rhs[row_index] - tail) / diagonal

    coefficients = [0.0 for _ in columns]
    for retained_index, coefficient in zip(retained_indices, retained_solution):
        coefficients[retained_index] = coefficient
    if not all(math.isfinite(value) for value in coefficients):
        return None
    return tuple(coefficients)


def _relaxed_interface_reaction_guess(
    *,
    reaction_guess: ForceVector,
    target_force_n: ForceVector,
    velocity_mps: ForceVector,
    relaxation: float,
) -> ForceVector:
    return relax_interface_reaction_forces(
        previous_force_n=reaction_guess,
        target_force_n=target_force_n,
        velocity_mps=velocity_mps,
        relaxation=relaxation,
        passivity_limit=False,
    ).force_n


def _limit_force_update(
    *,
    current_force_n: ForceVector,
    proposed_force_n: ForceVector,
    max_increment_n: float,
) -> tuple[ForceVector, bool]:
    if math.isinf(max_increment_n):
        return proposed_force_n, False
    _same_length(
        current_force_n,
        proposed_force_n,
        lhs_name="current_force_n",
        rhs_name="proposed_force_n",
    )
    delta = tuple(
        proposed_value - current_value
        for current_value, proposed_value in zip(current_force_n, proposed_force_n)
    )
    delta_norm = _force_norm(delta)
    if delta_norm <= max_increment_n:
        return proposed_force_n, False
    scale = max_increment_n / delta_norm
    limited = tuple(
        current_value + scale * delta_value
        for current_value, delta_value in zip(current_force_n, delta)
    )
    return limited, True


def _iqn_ils_interface_reaction_guess(
    *,
    trial_force_history: Sequence[ForceVector],
    residual_history: Sequence[ForceVector],
    current_residual_n: ForceVector,
    current_target_force_n: ForceVector,
    current_velocity_mps: ForceVector,
    relaxation: float,
) -> tuple[ForceVector, bool]:
    current_force_n = trial_force_history[-1]
    fallback = _relaxed_interface_reaction_guess(
        reaction_guess=current_force_n,
        target_force_n=current_target_force_n,
        velocity_mps=current_velocity_mps,
        relaxation=relaxation,
    )
    # IQN-ILS needs at least two trials to form one secant pair; the force
    # vector is validated non-empty, so the historical
    # min(2, len(current_force_n) + 1) always evaluated to 2.
    required_history = 2
    if len(trial_force_history) < required_history or len(residual_history) < required_history:
        return fallback, False

    dimension = len(current_force_n)
    history_limit = min(_IQN_ILS_HISTORY_LIMIT, dimension, len(trial_force_history) - 1)
    first_delta_index = len(trial_force_history) - history_limit
    residual_delta_columns: list[ForceVector] = []
    force_delta_columns: list[ForceVector] = []
    output_delta_columns: list[ForceVector] = []
    for index in range(first_delta_index, len(trial_force_history)):
        force_delta = _force_difference(trial_force_history[index], trial_force_history[index - 1])
        residual_delta = _force_difference(residual_history[index], residual_history[index - 1])
        if _force_norm(force_delta) <= 1.0e-30 or _force_norm(residual_delta) <= 1.0e-30:
            continue
        force_delta_columns.append(force_delta)
        residual_delta_columns.append(residual_delta)
        output_delta_columns.append(
            tuple(
                force_delta_value + residual_delta_value
                for force_delta_value, residual_delta_value in zip(force_delta, residual_delta)
            )
        )
    coefficients = _least_squares_coefficients(residual_delta_columns, current_residual_n)
    if coefficients is None and len(residual_delta_columns) > 1:
        residual_delta_columns = residual_delta_columns[-1:]
        force_delta_columns = force_delta_columns[-1:]
        output_delta_columns = output_delta_columns[-1:]
        coefficients = _least_squares_coefficients(residual_delta_columns, current_residual_n)
    if coefficients is None:
        return fallback, False

    if len(residual_delta_columns) == 1:
        modeled_residual = _force_linear_combination(residual_delta_columns, coefficients)
        unmodeled_residual = tuple(
            residual_value - modeled_value
            for residual_value, modeled_value in zip(current_residual_n, modeled_residual)
        )
        residual_norm = _force_norm(current_residual_n)
        if _force_norm(unmodeled_residual) > max(1.0e-12, 1.0e-8 * residual_norm):
            return fallback, False

    output_delta_residual = _force_linear_combination(output_delta_columns, coefficients)
    proposed = tuple(
        force_value + residual_value - correction
        for force_value, residual_value, correction in zip(
            current_force_n,
            current_residual_n,
            output_delta_residual,
        )
    )
    if not _force_is_finite(proposed):
        return fallback, False
    return proposed, True


def _diagonal_secant_interface_reaction_guess_from_anchor(
    *,
    trial_force_history: Sequence[ForceVector],
    residual_history: Sequence[ForceVector],
    anchor_force_n: ForceVector,
    anchor_residual_n: ForceVector,
    anchor_target_force_n: ForceVector,
    anchor_velocity_mps: ForceVector,
    relaxation: float,
) -> tuple[ForceVector, bool]:
    fallback = _relaxed_interface_reaction_guess(
        reaction_guess=anchor_force_n,
        target_force_n=anchor_target_force_n,
        velocity_mps=anchor_velocity_mps,
        relaxation=relaxation,
    )
    if len(trial_force_history) < 2 or len(residual_history) < 2:
        return fallback, False

    proposed_components: list[float] = []
    used_secant = False
    for component_index, (anchor_value, anchor_residual_value) in enumerate(
        zip(anchor_force_n, anchor_residual_n)
    ):
        proposed_value: float | None = None
        for history_index in range(len(trial_force_history) - 1, 0, -1):
            force_delta = (
                trial_force_history[history_index][component_index]
                - trial_force_history[history_index - 1][component_index]
            )
            residual_delta = (
                residual_history[history_index][component_index]
                - residual_history[history_index - 1][component_index]
            )
            if abs(force_delta) <= 1.0e-30 or abs(residual_delta) <= 1.0e-30:
                continue
            slope = residual_delta / force_delta
            if abs(slope) <= 1.0e-30:
                continue
            candidate = anchor_value - anchor_residual_value / slope
            if math.isfinite(candidate):
                proposed_value = candidate
                used_secant = True
                break
        if proposed_value is None:
            proposed_value = fallback[component_index]
        proposed_components.append(proposed_value)

    proposed = tuple(proposed_components)
    if not used_secant or not _force_is_finite(proposed):
        return fallback, False
    return proposed, True


def action_reaction_balance(
    action_force_n: Sequence[float],
    reaction_force_n: Sequence[float],
) -> ForceBalanceReport:
    """Return force-balance residual for an action/reaction interface pair."""
    action = _force_vector(action_force_n, name="action_force_n")
    reaction = _force_vector(reaction_force_n, name="reaction_force_n")
    _same_length(action, reaction, lhs_name="action_force_n", rhs_name="reaction_force_n")
    residual = tuple(a + r for a, r in zip(action, reaction))
    residual_norm = math.sqrt(sum(component * component for component in residual))
    scale = max(
        math.sqrt(sum(component * component for component in action))
        + math.sqrt(sum(component * component for component in reaction)),
        1.0e-30,
    )
    return ForceBalanceReport(
        residual_components_n=residual,
        residual_norm_n=residual_norm,
        relative_error=residual_norm / scale,
        scale_n=scale,
    )


def aitken_relaxation_factor(
    previous_relaxation: float,
    previous_residual: Sequence[float],
    current_residual: Sequence[float],
    *,
    lower: float = 0.01,
    upper: float = 1.5,
) -> float:
    """Return Aitken Delta^2 relaxation for an interface-reaction residual vector."""
    previous_relaxation_value = _finite_float(
        previous_relaxation,
        name="previous_relaxation",
    )
    lower_value = _finite_float(lower, name="lower")
    upper_value = _finite_float(upper, name="upper")
    if lower_value > upper_value:
        raise ValueError("lower must be less than or equal to upper")
    previous = _force_vector(previous_residual, name="previous_residual")
    current = _force_vector(current_residual, name="current_residual")
    if len(previous) != len(current):
        raise ValueError("previous_residual and current_residual must have the same length")
    delta = [current_value - previous_value for previous_value, current_value in zip(previous, current)]
    denominator = sum(component * component for component in delta)
    if denominator <= 1.0e-30:
        return float(min(max(previous_relaxation_value, lower_value), upper_value))
    numerator = sum(previous_value * change for previous_value, change in zip(previous, delta))
    proposed = -previous_relaxation_value * numerator / denominator
    return float(min(max(proposed, lower_value), upper_value))


def interface_reaction_force(interface_fluid_force_n: Sequence[float]) -> ForceVector:
    """Return the equal-and-opposite force applied to the solid interface."""
    fluid_force = _force_vector(interface_fluid_force_n, name="interface_fluid_force_n")
    return tuple(-component for component in fluid_force)


def region_pair_interface_reaction_forces(
    *,
    primary_fluid_force_n: Sequence[float],
    secondary_fluid_force_n: Sequence[float],
) -> RegionPairInterfaceReactionTarget:
    """Return equal-and-opposite solid reactions for a two-region FSI interface."""
    primary_fluid_force = _force_vector3(primary_fluid_force_n, name="primary_fluid_force_n")
    secondary_fluid_force = _force_vector3(secondary_fluid_force_n, name="secondary_fluid_force_n")
    return RegionPairInterfaceReactionTarget(
        primary_force_n=tuple(-component for component in primary_fluid_force),
        secondary_force_n=tuple(-component for component in secondary_fluid_force),
    )


def relax_interface_reaction_forces(
    *,
    previous_force_n: Sequence[float],
    target_force_n: Sequence[float],
    velocity_mps: Sequence[float],
    relaxation: float,
    passivity_limit: bool,
) -> InterfaceReactionUpdate:
    previous = _force_vector(previous_force_n, name="previous_force_n")
    target = _force_vector(target_force_n, name="target_force_n")
    velocity = _force_vector(velocity_mps, name="velocity_mps")
    relaxation_value = _finite_float(relaxation, name="relaxation")
    _same_length(previous, target, lhs_name="previous_force_n", rhs_name="target_force_n")
    _same_length(previous, velocity, lhs_name="previous_force_n", rhs_name="velocity_mps")
    residual = tuple(target_value - previous_value for previous_value, target_value in zip(previous, target))
    relaxed = tuple(
        previous_value + relaxation_value * residual_value
        for previous_value, residual_value in zip(previous, residual)
    )
    limited = [False for _ in relaxed]
    relaxed_list = list(relaxed)
    if passivity_limit:
        total_power = sum(force * velocity_value for force, velocity_value in zip(relaxed, velocity))
        velocity_norm_sq = sum(velocity_value * velocity_value for velocity_value in velocity)
        if total_power > 0.0 and velocity_norm_sq > 1.0e-30:
            scale = total_power / velocity_norm_sq
            for index, velocity_value in enumerate(velocity):
                correction = scale * velocity_value
                relaxed_list[index] -= correction
                limited[index] = abs(correction) > 0.0
    committed_force = tuple(relaxed_list)
    committed_residual = _force_residual(
        target_force_n=target,
        committed_force_n=committed_force,
    )
    limited_power = tuple(force * velocity_value for force, velocity_value in zip(committed_force, velocity))
    return InterfaceReactionUpdate(
        force_n=committed_force,
        residual_n=committed_residual,
        power_w=limited_power,
        passivity_limited=tuple(limited),
    )


def update_interface_reaction_for_next_step(
    *,
    previous_force_n: Sequence[float],
    target_force_n: Sequence[float],
    velocity_mps: Sequence[float],
    state: InterfaceReactionRelaxationState,
    initial_relaxation: float,
    use_aitken: bool,
    passivity_limit: bool,
    robin_impedance_ns_per_m: float = 0.0,
    aitken_lower_bound: float = 0.01,
    aitken_upper_bound: float = 1.5,
) -> InterfaceReactionStepUpdate:
    """Return the accepted interface-reaction update and next relaxation state."""
    previous_force = _force_vector(previous_force_n, name="previous_force_n")
    target_force = _force_vector(target_force_n, name="target_force_n")
    velocity = _force_vector(velocity_mps, name="velocity_mps")
    _same_length(previous_force, target_force, lhs_name="previous_force_n", rhs_name="target_force_n")
    _same_length(previous_force, velocity, lhs_name="previous_force_n", rhs_name="velocity_mps")
    robin_impedance_force = robin_neumann_impedance_force(
        velocity_mps=velocity,
        previous_velocity_mps=state.previous_velocity_mps,
        impedance_ns_per_m=robin_impedance_ns_per_m,
    )
    stabilized_target_force = tuple(
        target_value + impedance_value
        for target_value, impedance_value in zip(target_force, robin_impedance_force)
    )
    residual = tuple(
        target_value - previous_value
        for previous_value, target_value in zip(previous_force, stabilized_target_force)
    )
    aitken_lower_bound_value = _finite_float(
        aitken_lower_bound,
        name="aitken_lower_bound",
    )
    aitken_upper_bound_value = _finite_float(
        aitken_upper_bound,
        name="aitken_upper_bound",
    )
    if not 0.0 <= aitken_lower_bound_value <= aitken_upper_bound_value <= 1.5:
        raise ValueError(
            "aitken_lower_bound and aitken_upper_bound must be finite and satisfy "
            "0 <= lower <= upper <= 1.5"
        )
    relaxation = float(initial_relaxation)
    if use_aitken and state.previous_residual_n is not None:
        relaxation = aitken_relaxation_factor(
            state.relaxation,
            state.previous_residual_n,
            residual,
            lower=aitken_lower_bound_value,
            upper=aitken_upper_bound_value,
        )
    update = relax_interface_reaction_forces(
        previous_force_n=previous_force,
        target_force_n=stabilized_target_force,
        velocity_mps=velocity,
        relaxation=relaxation,
        passivity_limit=passivity_limit,
    )
    return InterfaceReactionStepUpdate(
        update=update,
        relaxation=relaxation,
        next_state=InterfaceReactionRelaxationState(
            previous_residual_n=update.residual_n,
            previous_velocity_mps=velocity,
            relaxation=relaxation,
        ),
        robin_impedance_force_n=robin_impedance_force,
    )


def solve_interface_reaction_fixed_point(
    *,
    initial_force_n: Sequence[float],
    evaluate_target: Callable[[ForceVector], InterfaceReactionTargetEvaluation],
    restore_state: Callable[[], None],
    max_iterations: int,
    tolerance_n: float,
    initial_relaxation: float,
    use_aitken: bool,
    passivity_limit: bool,
    solver: str = "aitken",
    target_map_relaxation: float = 1.0,
    accept_evaluation: Callable[[InterfaceReactionTargetEvaluation], bool] | None = None,
    aitken_lower_bound: float = 0.01,
    aitken_upper_bound: float = 1.5,
    rejected_trial_backtrack: float = 1.0,
    residual_growth_rejection_factor: float = math.inf,
    max_accepted_residual_n: float = math.inf,
    trust_region_force_increment_n: float = math.inf,
    trust_region_adaptive: bool = False,
    trust_region_shrink_factor: float = 0.5,
    trust_region_growth_factor: float = 1.25,
    trust_region_rebound_factor: float = math.inf,
    trust_region_rebound_backtrack: float = 0.5,
    trust_region_rebound_stop_factor: float = math.inf,
    trust_region_rebound_stop_max_residual_n: float = math.inf,
    residual_continuation_iterations_max: int = 0,
    residual_continuation_threshold_n: float = math.inf,
    residual_continuation_rebound_secant_from_best: bool = False,
    residual_continuation_rebound_secant_factor: float = math.inf,
    residual_continuation_rebound_secant_evaluation_extensions_max: int = 0,
    all_rejected_trial_policy: str = "raise",
) -> InterfaceReactionFixedPointResult:
    """Run an interface-reaction fixed-point solve around caller-managed snapshots."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    tolerance_value = _finite_float(tolerance_n, name="tolerance_n")
    if tolerance_value < 0.0:
        raise ValueError("tolerance_n must be non-negative")
    solver_name = _validate_interface_reaction_solver(solver)
    solver_target_map_relaxation = float(target_map_relaxation)
    if (
        not math.isfinite(solver_target_map_relaxation)
        or not 0.0 < solver_target_map_relaxation <= 1.0
    ):
        raise ValueError("target_map_relaxation must be a finite number in (0, 1]")
    aitken_lower_bound_value = _finite_float(
        aitken_lower_bound,
        name="aitken_lower_bound",
    )
    aitken_upper_bound_value = _finite_float(
        aitken_upper_bound,
        name="aitken_upper_bound",
    )
    if not 0.0 <= aitken_lower_bound_value <= aitken_upper_bound_value <= 1.5:
        raise ValueError(
            "aitken_lower_bound and aitken_upper_bound must be finite and satisfy "
            "0 <= lower <= upper <= 1.5"
        )
    rejected_trial_backtrack_value = _finite_float(
        rejected_trial_backtrack,
        name="rejected_trial_backtrack",
    )
    if not 0.0 < rejected_trial_backtrack_value <= 1.0:
        raise ValueError("rejected_trial_backtrack must be finite and in (0, 1]")
    if all_rejected_trial_policy not in {"raise", "initial_force"}:
        raise ValueError(
            "all_rejected_trial_policy must be 'raise' or 'initial_force'"
        )
    residual_growth_rejection_factor_value = float(residual_growth_rejection_factor)
    if not (
        math.isinf(residual_growth_rejection_factor_value)
        or (
            math.isfinite(residual_growth_rejection_factor_value)
            and residual_growth_rejection_factor_value >= 1.0
        )
    ):
        raise ValueError(
            "residual_growth_rejection_factor must be >= 1 or infinity"
        )
    max_accepted_residual_value = float(max_accepted_residual_n)
    if not (
        math.isinf(max_accepted_residual_value)
        or (math.isfinite(max_accepted_residual_value) and max_accepted_residual_value >= 0.0)
    ):
        raise ValueError("max_accepted_residual_n must be non-negative or infinity")
    trust_region_force_increment_value = float(trust_region_force_increment_n)
    if not (
        math.isinf(trust_region_force_increment_value)
        or (
            math.isfinite(trust_region_force_increment_value)
            and trust_region_force_increment_value > 0.0
        )
    ):
        raise ValueError("trust_region_force_increment_n must be positive or infinity")
    trust_region_adaptive_enabled = bool(trust_region_adaptive)
    trust_region_shrink_factor_value = _finite_float(
        trust_region_shrink_factor,
        name="trust_region_shrink_factor",
    )
    if not 0.0 < trust_region_shrink_factor_value <= 1.0:
        raise ValueError("trust_region_shrink_factor must be finite and in (0, 1]")
    trust_region_growth_factor_value = _finite_float(
        trust_region_growth_factor,
        name="trust_region_growth_factor",
    )
    if trust_region_growth_factor_value < 1.0:
        raise ValueError("trust_region_growth_factor must be finite and >= 1")
    trust_region_rebound_factor_value = float(trust_region_rebound_factor)
    if not (
        math.isinf(trust_region_rebound_factor_value)
        or (
            math.isfinite(trust_region_rebound_factor_value)
            and trust_region_rebound_factor_value >= 1.0
        )
    ):
        raise ValueError("trust_region_rebound_factor must be >= 1 or infinity")
    trust_region_rebound_backtrack_value = _finite_float(
        trust_region_rebound_backtrack,
        name="trust_region_rebound_backtrack",
    )
    if not 0.0 < trust_region_rebound_backtrack_value < 1.0:
        raise ValueError("trust_region_rebound_backtrack must be finite and in (0, 1)")
    trust_region_rebound_stop_factor_value = float(trust_region_rebound_stop_factor)
    if not (
        math.isinf(trust_region_rebound_stop_factor_value)
        or (
            math.isfinite(trust_region_rebound_stop_factor_value)
            and trust_region_rebound_stop_factor_value >= 1.0
        )
    ):
        raise ValueError("trust_region_rebound_stop_factor must be >= 1 or infinity")
    trust_region_rebound_stop_max_residual_value = float(
        trust_region_rebound_stop_max_residual_n
    )
    if not (
        math.isinf(trust_region_rebound_stop_max_residual_value)
        or (
            math.isfinite(trust_region_rebound_stop_max_residual_value)
            and trust_region_rebound_stop_max_residual_value >= 0.0
        )
    ):
        raise ValueError(
            "trust_region_rebound_stop_max_residual_n must be non-negative or infinity"
        )
    if trust_region_adaptive_enabled and math.isinf(trust_region_force_increment_value):
        raise ValueError(
            "trust_region_adaptive requires a finite trust_region_force_increment_n"
        )
    residual_continuation_iterations_max_value = int(
        residual_continuation_iterations_max
    )
    if residual_continuation_iterations_max_value < 0:
        raise ValueError("residual_continuation_iterations_max must be non-negative")
    residual_continuation_threshold_value = float(
        residual_continuation_threshold_n
    )
    if not (
        math.isinf(residual_continuation_threshold_value)
        or (
            math.isfinite(residual_continuation_threshold_value)
            and residual_continuation_threshold_value >= 0.0
        )
    ):
        raise ValueError(
            "residual_continuation_threshold_n must be non-negative or infinity"
        )
    residual_continuation_rebound_secant_evaluation_extensions_max_value = int(
        residual_continuation_rebound_secant_evaluation_extensions_max
    )
    if residual_continuation_rebound_secant_evaluation_extensions_max_value < 0:
        raise ValueError(
            "residual_continuation_rebound_secant_evaluation_extensions_max "
            "must be non-negative"
        )
    residual_continuation_rebound_secant_factor_value = float(
        residual_continuation_rebound_secant_factor
    )
    if not (
        math.isinf(residual_continuation_rebound_secant_factor_value)
        or (
            math.isfinite(residual_continuation_rebound_secant_factor_value)
            and residual_continuation_rebound_secant_factor_value >= 1.0
        )
    ):
        raise ValueError(
            "residual_continuation_rebound_secant_factor must be >= 1 or infinity"
        )
    effective_residual_continuation_rebound_secant_factor = (
        trust_region_rebound_stop_factor_value
        if math.isinf(residual_continuation_rebound_secant_factor_value)
        else residual_continuation_rebound_secant_factor_value
    )

    reaction_guess = _force_vector(initial_force_n, name="initial_force_n")
    initial_reaction_force_n = reaction_guess
    previous_residual: ForceVector | None = None
    previous_accepted_residual_norm_n: float | None = None
    relaxation = _finite_float(initial_relaxation, name="initial_relaxation")
    residual_norm_n = 0.0
    converged = False
    iterations_used = 0
    accepted_force_n = reaction_guess
    accepted_velocity_mps: ForceVector = tuple(0.0 for _ in reaction_guess)
    accepted_target_force_n = reaction_guess
    accepted_residual_norm_n = math.inf
    accepted_trial_index: int | None = None
    accepted_payload: object | None = None
    rejected_trial_count = 0
    rejected_trial_backtrack_count = 0
    residual_growth_rejected_trial_count = 0
    max_residual_rejected_trial_count = 0
    trust_region_limited_update_count = 0
    trust_region_shrink_count = 0
    trust_region_growth_count = 0
    trust_region_rebound_backtrack_count = 0
    trust_region_rebound_stop_count = 0
    trust_region_rebound_stop_suppressed_count = 0
    residual_continuation_iteration_count = 0
    residual_continuation_rebound_secant_count = 0
    residual_continuation_rebound_secant_evaluation_extension_count = 0
    trust_region_effective_force_increment_n = trust_region_force_increment_value
    trial_force_history: list[ForceVector] = []
    target_force_history: list[ForceVector] = []
    residual_history: list[ForceVector] = []
    physical_target_force_history: list[ForceVector] = []
    physical_residual_history: list[ForceVector] = []
    diagnostic_target_force_history: list[ForceVector] = []
    diagnostic_residual_history: list[ForceVector] = []
    iqn_ils_least_squares_update_count = 0

    base_continuation_iteration_budget = (
        int(max_iterations) + residual_continuation_iterations_max_value
    )
    total_iteration_budget = base_continuation_iteration_budget
    iteration = 0
    while iteration < total_iteration_budget:
        if iteration >= max_iterations:
            if converged or accepted_residual_norm_n <= residual_continuation_threshold_value:
                break
            if iteration < base_continuation_iteration_budget:
                residual_continuation_iteration_count += 1
        restore_state()
        trial_force_history.append(reaction_guess)
        evaluation = evaluate_target(reaction_guess)
        physical_target = _force_vector(evaluation.target_force_n, name="target_force_n")
        diagnostic_target = (
            physical_target
            if evaluation.diagnostic_target_force_n is None
            else _force_vector(
                evaluation.diagnostic_target_force_n,
                name="diagnostic_target_force_n",
            )
        )
        target = _relaxed_target_map_force(
            trial_force_n=reaction_guess,
            physical_target_force_n=physical_target,
            target_map_relaxation=solver_target_map_relaxation,
        )
        velocity = _force_vector(evaluation.velocity_mps, name="velocity_mps")
        _same_length(reaction_guess, physical_target, lhs_name="reaction_guess", rhs_name="target_force_n")
        _same_length(
            reaction_guess,
            diagnostic_target,
            lhs_name="reaction_guess",
            rhs_name="diagnostic_target_force_n",
        )
        _same_length(reaction_guess, velocity, lhs_name="reaction_guess", rhs_name="velocity_mps")
        physical_residual = tuple(
            target_value - guess_value
            for guess_value, target_value in zip(reaction_guess, physical_target)
        )
        diagnostic_residual = tuple(
            target_value - guess_value
            for guess_value, target_value in zip(reaction_guess, diagnostic_target)
        )
        residual = tuple(target_value - guess_value for guess_value, target_value in zip(reaction_guess, target))
        target_force_history.append(target)
        residual_history.append(residual)
        physical_target_force_history.append(physical_target)
        physical_residual_history.append(physical_residual)
        diagnostic_target_force_history.append(diagnostic_target)
        diagnostic_residual_history.append(diagnostic_residual)
        residual_norm_n = math.sqrt(sum(component * component for component in physical_residual))
        iterations_used = iteration + 1
        trial_accepted_by_predicate = (
            True if accept_evaluation is None else bool(accept_evaluation(evaluation))
        )
        trial_rejected_by_residual_growth = False
        if (
            trial_accepted_by_predicate
            and math.isfinite(residual_growth_rejection_factor_value)
            and accepted_trial_index is not None
            and accepted_residual_norm_n < math.inf
            and residual_norm_n > tolerance_value
            and residual_norm_n
            > max(
                accepted_residual_norm_n * residual_growth_rejection_factor_value,
                tolerance_value,
            )
        ):
            trial_rejected_by_residual_growth = True
            residual_growth_rejected_trial_count += 1
        trial_rejected_by_max_residual = False
        if (
            trial_accepted_by_predicate
            and not trial_rejected_by_residual_growth
            and math.isfinite(max_accepted_residual_value)
            and residual_norm_n > max_accepted_residual_value
        ):
            trial_rejected_by_max_residual = True
            max_residual_rejected_trial_count += 1
        trial_accepted = (
            trial_accepted_by_predicate
            and not trial_rejected_by_residual_growth
            and not trial_rejected_by_max_residual
        )
        if not trial_accepted:
            rejected_trial_count += 1
        if (
            trust_region_adaptive_enabled
            and previous_accepted_residual_norm_n is not None
        ):
            if residual_norm_n > previous_accepted_residual_norm_n:
                updated_increment = (
                    trust_region_effective_force_increment_n
                    * trust_region_shrink_factor_value
                )
                if updated_increment < trust_region_effective_force_increment_n:
                    trust_region_shrink_count += 1
                trust_region_effective_force_increment_n = updated_increment
            elif trial_accepted and residual_norm_n < previous_accepted_residual_norm_n:
                updated_increment = min(
                    trust_region_force_increment_value,
                    trust_region_effective_force_increment_n
                    * trust_region_growth_factor_value,
                )
                if updated_increment > trust_region_effective_force_increment_n:
                    trust_region_growth_count += 1
                trust_region_effective_force_increment_n = updated_increment
        if trial_accepted:
            previous_accepted_residual_norm_n = residual_norm_n
        if trial_accepted and residual_norm_n <= accepted_residual_norm_n:
            accepted_force_n = reaction_guess
            accepted_velocity_mps = velocity
            accepted_target_force_n = physical_target
            accepted_residual_norm_n = residual_norm_n
            accepted_trial_index = iteration
            accepted_payload = evaluation.payload
        if residual_norm_n <= tolerance_value:
            converged = trial_accepted
            if (
                trial_accepted
                or rejected_trial_backtrack_value >= 1.0
            ):
                break
        trial_rebounded_stop_from_best = (
            trial_accepted
            and math.isfinite(trust_region_rebound_stop_factor_value)
            and accepted_trial_index is not None
            and accepted_residual_norm_n < math.inf
            and residual_norm_n > tolerance_value
            and residual_norm_n
            > max(
                accepted_residual_norm_n * trust_region_rebound_stop_factor_value,
                tolerance_value,
            )
            and reaction_guess != accepted_force_n
        )
        trial_rebounded_secant_from_best = (
            residual_continuation_rebound_secant_from_best
            and math.isfinite(effective_residual_continuation_rebound_secant_factor)
            and iteration >= max_iterations
            and accepted_trial_index is not None
            and accepted_residual_norm_n < math.inf
            and accepted_residual_norm_n > residual_continuation_threshold_value
            and residual_norm_n > tolerance_value
            and residual_norm_n
            > max(
                accepted_residual_norm_n
                * effective_residual_continuation_rebound_secant_factor,
                tolerance_value,
            )
            and reaction_guess != accepted_force_n
        )
        if trial_rebounded_secant_from_best:
            accepted_residual = residual_history[accepted_trial_index]
            accepted_solver_target = target_force_history[accepted_trial_index]
            proposed_reaction_guess, used_secant = (
                _diagonal_secant_interface_reaction_guess_from_anchor(
                    trial_force_history=trial_force_history,
                    residual_history=residual_history,
                    anchor_force_n=accepted_force_n,
                    anchor_residual_n=accepted_residual,
                    anchor_target_force_n=accepted_solver_target,
                    anchor_velocity_mps=accepted_velocity_mps,
                    relaxation=relaxation,
                )
            )
            reaction_guess, trust_region_limited = _limit_force_update(
                current_force_n=accepted_force_n,
                proposed_force_n=proposed_reaction_guess,
                max_increment_n=trust_region_effective_force_increment_n,
            )
            if trust_region_limited:
                trust_region_limited_update_count += 1
            if used_secant:
                residual_continuation_rebound_secant_count += 1
                if (
                    iteration + 1 >= total_iteration_budget
                    and residual_continuation_rebound_secant_evaluation_extension_count
                    < residual_continuation_rebound_secant_evaluation_extensions_max_value
                ):
                    total_iteration_budget += 1
                    residual_continuation_rebound_secant_evaluation_extension_count += 1
            previous_residual = None
            iteration += 1
            continue
        if trial_rebounded_stop_from_best:
            if (
                accepted_residual_norm_n
                <= trust_region_rebound_stop_max_residual_value
            ):
                trust_region_rebound_stop_count += 1
                break
            trust_region_rebound_stop_suppressed_count += 1
        trial_rebounded_from_best = (
            trial_accepted_by_predicate
            and math.isfinite(trust_region_rebound_factor_value)
            and accepted_trial_index is not None
            and accepted_residual_norm_n < math.inf
            and residual_norm_n > tolerance_value
            and residual_norm_n
            > max(
                accepted_residual_norm_n * trust_region_rebound_factor_value,
                tolerance_value,
            )
            and reaction_guess != accepted_force_n
        )
        if trial_rebounded_from_best:
            reaction_guess = tuple(
                accepted_value
                + trust_region_rebound_backtrack_value
                * (current_value - accepted_value)
                for accepted_value, current_value in zip(
                    accepted_force_n,
                    reaction_guess,
                )
            )
            trust_region_rebound_backtrack_count += 1
            previous_residual = None
            relaxation = min(
                max(
                    relaxation * trust_region_rebound_backtrack_value,
                    aitken_lower_bound_value,
                ),
                aitken_upper_bound_value,
            )
            iteration += 1
            continue
        if not trial_accepted and rejected_trial_backtrack_value < 1.0:
            backtrack_anchor = (
                accepted_force_n
                if accepted_trial_index is not None
                else tuple(0.0 for _ in reaction_guess)
            )
            reaction_guess = tuple(
                accepted_value
                + rejected_trial_backtrack_value * (rejected_value - accepted_value)
                for accepted_value, rejected_value in zip(backtrack_anchor, reaction_guess)
            )
            rejected_trial_backtrack_count += 1
            previous_residual = None
            relaxation = min(
                max(relaxation * rejected_trial_backtrack_value, aitken_lower_bound_value),
                aitken_upper_bound_value,
            )
            iteration += 1
            continue
        if use_aitken and previous_residual is not None:
            relaxation = aitken_relaxation_factor(
                relaxation,
                previous_residual,
                residual,
                lower=aitken_lower_bound_value,
                upper=aitken_upper_bound_value,
            )
        previous_residual = residual
        if solver_name == "iqn_ils":
            proposed_reaction_guess, used_least_squares = _iqn_ils_interface_reaction_guess(
                trial_force_history=trial_force_history,
                residual_history=residual_history,
                current_residual_n=residual,
                current_target_force_n=target,
                current_velocity_mps=velocity,
                relaxation=relaxation,
            )
            if used_least_squares:
                iqn_ils_least_squares_update_count += 1
        else:
            proposed_reaction_guess = _relaxed_interface_reaction_guess(
                reaction_guess=reaction_guess,
                target_force_n=target,
                velocity_mps=velocity,
                relaxation=relaxation,
            )
        reaction_guess, trust_region_limited = _limit_force_update(
            current_force_n=trial_force_history[-1],
            proposed_force_n=proposed_reaction_guess,
            max_increment_n=trust_region_effective_force_increment_n,
        )
        if trust_region_limited:
            trust_region_limited_update_count += 1
        iteration += 1

    all_trials_rejected = accepted_trial_index is None and rejected_trial_count > 0
    zero_force_commit_blocked = False
    fallback_force_source = ""
    if all_trials_rejected:
        zero_force_commit_blocked = True
        converged = False
        if all_rejected_trial_policy == "raise":
            raise RuntimeError(
                "all FSI trials rejected; refusing to commit zero interface force"
            )
        accepted_force_n = initial_reaction_force_n
        accepted_velocity_mps = tuple(0.0 for _ in initial_reaction_force_n)
        accepted_target_force_n = initial_reaction_force_n
        accepted_residual_norm_n = math.inf
        fallback_force_source = "initial_force"

    if passivity_limit and accepted_trial_index is not None:
        force_before_passivity = accepted_force_n
        accepted_force_n = relax_interface_reaction_forces(
            previous_force_n=accepted_force_n,
            target_force_n=accepted_force_n,
            velocity_mps=accepted_velocity_mps,
            relaxation=1.0,
            passivity_limit=True,
        ).force_n
        passivity_changed_force = accepted_force_n != force_before_passivity
        accepted_residual_norm_n = _force_norm(
            _force_residual(
                target_force_n=accepted_target_force_n,
                committed_force_n=accepted_force_n,
            )
        )
        converged = accepted_residual_norm_n <= tolerance_value
    else:
        passivity_changed_force = False
    accepted_state_reusable = (
        accepted_trial_index is not None
        and accepted_trial_index == iterations_used - 1
        and accepted_trial_index < len(trial_force_history)
        and accepted_force_n == trial_force_history[accepted_trial_index]
        and not passivity_changed_force
    )

    interface_map_amplification = _secant_amplification_stats(
        trial_force_history,
        target_force_history,
    )
    residual_jacobian_amplification = _secant_amplification_stats(
        trial_force_history,
        residual_history,
    )
    physical_interface_map_amplification = _secant_amplification_stats(
        trial_force_history,
        physical_target_force_history,
    )
    physical_residual_jacobian_amplification = _secant_amplification_stats(
        trial_force_history,
        physical_residual_history,
    )
    diagnostic_interface_map_amplification = _secant_amplification_stats(
        trial_force_history,
        diagnostic_target_force_history,
    )
    diagnostic_residual_jacobian_amplification = _secant_amplification_stats(
        trial_force_history,
        diagnostic_residual_history,
    )

    return InterfaceReactionFixedPointResult(
        force_n=accepted_force_n,
        iterations_used=iterations_used,
        converged=converged,
        residual_norm_n=accepted_residual_norm_n,
        relaxation=relaxation,
        solver=solver_name,
        trial_force_history_n=tuple(trial_force_history),
        target_force_history_n=tuple(target_force_history),
        residual_history_n=tuple(residual_history),
        physical_target_force_history_n=tuple(physical_target_force_history),
        physical_residual_history_n=tuple(physical_residual_history),
        diagnostic_target_force_history_n=tuple(diagnostic_target_force_history),
        diagnostic_residual_history_n=tuple(diagnostic_residual_history),
        accepted_trial_index=accepted_trial_index,
        accepted_state_reusable=accepted_state_reusable,
        accepted_payload=accepted_payload,
        all_trials_rejected=all_trials_rejected,
        zero_force_commit_blocked=zero_force_commit_blocked,
        fallback_force_source=fallback_force_source,
        rejected_trial_count=rejected_trial_count,
        rejected_trial_backtrack_count=rejected_trial_backtrack_count,
        residual_growth_rejected_trial_count=residual_growth_rejected_trial_count,
        max_residual_rejected_trial_count=max_residual_rejected_trial_count,
        trust_region_limited_update_count=trust_region_limited_update_count,
        trust_region_shrink_count=trust_region_shrink_count,
        trust_region_growth_count=trust_region_growth_count,
        trust_region_rebound_backtrack_count=trust_region_rebound_backtrack_count,
        trust_region_rebound_stop_count=trust_region_rebound_stop_count,
        trust_region_rebound_stop_suppressed_count=(
            trust_region_rebound_stop_suppressed_count
        ),
        residual_continuation_iteration_count=residual_continuation_iteration_count,
        residual_continuation_rebound_secant_count=(
            residual_continuation_rebound_secant_count
        ),
        residual_continuation_rebound_secant_evaluation_extension_count=(
            residual_continuation_rebound_secant_evaluation_extension_count
        ),
        trust_region_effective_force_increment_n=trust_region_effective_force_increment_n,
        iqn_ils_least_squares_update_count=iqn_ils_least_squares_update_count,
        interface_map_amplification_max=interface_map_amplification[0],
        residual_jacobian_amplification_max=residual_jacobian_amplification[0],
        physical_interface_map_amplification_max=physical_interface_map_amplification[0],
        physical_residual_jacobian_amplification_max=(
            physical_residual_jacobian_amplification[0]
        ),
        diagnostic_interface_map_amplification_max=(
            diagnostic_interface_map_amplification[0]
        ),
        diagnostic_residual_jacobian_amplification_max=(
            diagnostic_residual_jacobian_amplification[0]
        ),
        interface_map_amplification_sample_count=interface_map_amplification[1],
        residual_jacobian_amplification_sample_count=(
            residual_jacobian_amplification[1]
        ),
        physical_interface_map_amplification_sample_count=(
            physical_interface_map_amplification[1]
        ),
        physical_residual_jacobian_amplification_sample_count=(
            physical_residual_jacobian_amplification[1]
        ),
        diagnostic_interface_map_amplification_sample_count=(
            diagnostic_interface_map_amplification[1]
        ),
        diagnostic_residual_jacobian_amplification_sample_count=(
            diagnostic_residual_jacobian_amplification[1]
        ),
        target_map_relaxation=solver_target_map_relaxation,
    )


def solve_and_apply_interface_reaction_step(
    *,
    initial_force_n: Sequence[float],
    save_state: Callable[[], None],
    restore_state: Callable[[], None],
    evaluate_target: Callable[[ForceVector], InterfaceReactionTargetEvaluation],
    apply_force: Callable[[ForceVector], None],
    commit_accepted_state: Callable[[object | None], None] | None = None,
    max_iterations: int,
    tolerance_n: float,
    initial_relaxation: float,
    use_aitken: bool,
    passivity_limit: bool,
    solver: str = "aitken",
    target_map_relaxation: float = 1.0,
    accept_evaluation: Callable[[InterfaceReactionTargetEvaluation], bool] | None = None,
    aitken_lower_bound: float = 0.01,
    aitken_upper_bound: float = 1.5,
    rejected_trial_backtrack: float = 1.0,
    residual_growth_rejection_factor: float = math.inf,
    max_accepted_residual_n: float = math.inf,
    trust_region_force_increment_n: float = math.inf,
    trust_region_adaptive: bool = False,
    trust_region_shrink_factor: float = 0.5,
    trust_region_growth_factor: float = 1.25,
    trust_region_rebound_factor: float = math.inf,
    trust_region_rebound_backtrack: float = 0.5,
    trust_region_rebound_stop_factor: float = math.inf,
    trust_region_rebound_stop_max_residual_n: float = math.inf,
    residual_continuation_iterations_max: int = 0,
    residual_continuation_threshold_n: float = math.inf,
    residual_continuation_rebound_secant_from_best: bool = False,
    residual_continuation_rebound_secant_factor: float = math.inf,
    residual_continuation_rebound_secant_evaluation_extensions_max: int = 0,
    all_rejected_trial_policy: str = "raise",
) -> InterfaceReactionFixedPointResult:
    """Solve one partitioned FSI step and commit the accepted interface force.

    The expensive solid/fluid work remains inside `evaluate_target`, so Taichi
    fields can stay device-resident. The solver saves the accepted state,
    restores before each fixed-point trial, restores to the accepted base state
    before returning unless the current trial state is provably reusable, and
    then commits the accepted force through `apply_force`.
    """
    save_state()
    result: InterfaceReactionFixedPointResult | None = None
    try:
        result = solve_interface_reaction_fixed_point(
            initial_force_n=initial_force_n,
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=max_iterations,
            tolerance_n=tolerance_n,
            initial_relaxation=initial_relaxation,
            use_aitken=use_aitken,
            passivity_limit=passivity_limit,
            solver=solver,
            target_map_relaxation=target_map_relaxation,
            accept_evaluation=accept_evaluation,
            aitken_lower_bound=aitken_lower_bound,
            aitken_upper_bound=aitken_upper_bound,
            rejected_trial_backtrack=rejected_trial_backtrack,
            residual_growth_rejection_factor=residual_growth_rejection_factor,
            max_accepted_residual_n=max_accepted_residual_n,
            trust_region_force_increment_n=trust_region_force_increment_n,
            trust_region_adaptive=trust_region_adaptive,
            trust_region_shrink_factor=trust_region_shrink_factor,
            trust_region_growth_factor=trust_region_growth_factor,
            trust_region_rebound_factor=trust_region_rebound_factor,
            trust_region_rebound_backtrack=trust_region_rebound_backtrack,
            trust_region_rebound_stop_factor=trust_region_rebound_stop_factor,
            trust_region_rebound_stop_max_residual_n=(
                trust_region_rebound_stop_max_residual_n
            ),
            residual_continuation_iterations_max=(
                residual_continuation_iterations_max
            ),
            residual_continuation_threshold_n=residual_continuation_threshold_n,
            residual_continuation_rebound_secant_from_best=(
                residual_continuation_rebound_secant_from_best
            ),
            residual_continuation_rebound_secant_factor=(
                residual_continuation_rebound_secant_factor
            ),
            residual_continuation_rebound_secant_evaluation_extensions_max=(
                residual_continuation_rebound_secant_evaluation_extensions_max
            ),
            all_rejected_trial_policy=all_rejected_trial_policy,
        )
    finally:
        if (
            result is not None
            and commit_accepted_state is not None
            and result.accepted_state_reusable
        ):
            commit_accepted_state(result.accepted_payload)
        else:
            restore_state()
    apply_force(result.force_n)
    return result
