"""Fixed Cartesian material-reference stencils for surface markers.

The map is deliberately a small host-side reference object.  It has no Taichi
or runtime side effects: callers may materialize its fixed eight-entry rows on
their preferred device, but the same stored rows must be used for both W and
W.T.
"""

from dataclasses import dataclass
import hashlib
import itertools

import numpy as np


_ALGORITHM_VERSION = b"material-surface-binding-v1"
_F64_OPERATION_ROUNDOFF_MULTIPLIER = 32.0


@dataclass(frozen=True, slots=True)
class MaterialSurfaceBinding:
    """Immutable fixed-stencil reference map from material particles to markers."""

    particle_indices: np.ndarray
    weights: np.ndarray
    stencil_sizes: np.ndarray
    particle_count: int
    marker_count: int
    reference_particle_positions_m: np.ndarray
    reference_marker_positions_m: np.ndarray
    particle_mass_kg: np.ndarray
    active_axes: tuple[int, ...]
    identity_sha256: str
    maximum_row_l1: float
    maximum_row_inverse_mass_gain: float


def _as_finite_array(name: str, value: object, shape_tail: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float64-compatible array") from exc
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _as_finite_vector(name: str, value: object, count: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float64-compatible vector") from exc
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    """Return a non-owning view over immutable bytes so writeability cannot return."""

    contiguous = np.array(array, copy=True, order="C")
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(contiguous.shape)


def _f32_coordinate_rounding_error(value: float) -> float:
    """Maximum round-to-nearest f32 error around one stored coordinate.

    The larger adjacent f32 gap covers asymmetric bins at powers of two and
    remains valid in the subnormal range.  Coordinates outside finite f32
    range cannot have originated from the f32 material storage contract.
    """

    with np.errstate(over="ignore"):
        quantized = np.float32(value)
    if not np.isfinite(quantized):
        raise ValueError("particle Cartesian coordinates must be finite f32-representable values")
    lower = np.nextafter(quantized, np.float32(-np.inf), dtype=np.float32)
    upper = np.nextafter(quantized, np.float32(np.inf), dtype=np.float32)
    gaps = [abs(float(quantized - neighbor)) for neighbor in (lower, upper) if np.isfinite(neighbor)]
    return 0.5 * max(gaps)


def _f64_operation_roundoff_bound(*values: np.ndarray | float) -> float:
    """Bound a short f64 expression from the magnitudes actually supplied."""

    magnitude = 0.0
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.size:
            magnitude = max(magnitude, float(np.max(np.abs(array))))
    if magnitude == 0.0:
        return float(np.nextafter(0.0, 1.0))
    return _F64_OPERATION_ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * magnitude


def _f32_quantized_difference_roundoff_bound(lower: float, upper: float) -> float:
    """Bound ``(upper - lower)`` versus the corresponding physical f32 inputs."""

    return (
        _f32_coordinate_rounding_error(lower)
        + _f32_coordinate_rounding_error(upper)
        + _f64_operation_roundoff_bound(lower, upper, upper - lower)
    )


def _half_cell_endpoint_roundoff_bound(values: np.ndarray, at_lower_end: bool) -> float:
    """Bound endpoint reconstruction from the stored f32 particle coordinates."""

    endpoint_index = 0 if at_lower_end else -1
    neighbor_index = 1 if at_lower_end else -2
    endpoint = float(values[endpoint_index])
    neighbor = float(values[neighbor_index])
    stored_endpoint = 1.5 * endpoint - 0.5 * neighbor
    return (
        1.5 * _f32_coordinate_rounding_error(endpoint)
        + 0.5 * _f32_coordinate_rounding_error(neighbor)
        + _f64_operation_roundoff_bound(endpoint, neighbor, stored_endpoint)
    )


def _coordinate_axis_values(particles: np.ndarray, axis: int) -> np.ndarray:
    values = np.unique(particles[:, axis])
    if values.size > 1:
        spacings = np.diff(values)
        nominal = float(spacings[0])
        for spacing_index, spacing in enumerate(spacings):
            tolerance = (
                _f32_quantized_difference_roundoff_bound(
                    float(values[spacing_index]), float(values[spacing_index + 1])
                )
                + _f32_quantized_difference_roundoff_bound(float(values[0]), float(values[1]))
                + _f64_operation_roundoff_bound(float(spacing), nominal, float(spacing - nominal))
            )
            if abs(float(spacing) - nominal) > tolerance:
                raise ValueError(
                    f"particle Cartesian axis {axis} must have uniform spacing within f32 input roundoff"
                )
    return values


def _axis_pair_and_weights(
    values: np.ndarray, marker_value: float, axis: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if values.size == 1:
        tolerance = _f32_coordinate_rounding_error(float(values[0])) + _f64_operation_roundoff_bound(
            float(values[0]), marker_value
        )
        if abs(marker_value - float(values[0])) > tolerance:
            raise ValueError(f"marker coordinate on inactive axis {axis} is not in the particle plane")
        return (0,), (1.0,)

    lower_spacing = float(values[1] - values[0])
    upper_spacing = float(values[-1] - values[-2])
    lower = float(values[0] - 0.5 * lower_spacing)
    upper = float(values[-1] + 0.5 * upper_spacing)
    # Physical wall markers are independently rounded into f32 storage. Both
    # operands of the endpoint comparison need their input quantization bound.
    marker_roundoff = _f32_coordinate_rounding_error(marker_value)
    lower_tolerance = _half_cell_endpoint_roundoff_bound(values, at_lower_end=True) + marker_roundoff
    upper_tolerance = _half_cell_endpoint_roundoff_bound(values, at_lower_end=False) + marker_roundoff
    if marker_value < lower - lower_tolerance or marker_value > upper + upper_tolerance:
        raise ValueError(f"marker coordinate on axis {axis} exceeds the allowed half-cell extrapolation")
    if marker_value <= values[0]:
        lower_index = 0
    elif marker_value >= values[-1]:
        lower_index = values.size - 2
    else:
        lower_index = int(np.searchsorted(values, marker_value, side="right") - 1)
    upper_index = lower_index + 1
    fraction = (marker_value - float(values[lower_index])) / float(
        values[upper_index] - values[lower_index]
    )
    return (lower_index, upper_index), (1.0 - fraction, fraction)


def _identity_sha256(
    particle_positions: np.ndarray,
    marker_positions: np.ndarray,
    masses: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    stencil_sizes: np.ndarray,
    active_axes: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(_ALGORITHM_VERSION)
    for name, array in (
        (b"particle_reference_positions", particle_positions),
        (b"marker_reference_positions", marker_positions),
        (b"particle_mass_kg", masses),
        (b"particle_indices", indices),
        (b"weights", weights),
        (b"stencil_sizes", stencil_sizes),
    ):
        digest.update(name)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(np.asarray(active_axes, dtype=np.int8).tobytes())
    return digest.hexdigest()


def build_cartesian_material_surface_binding(
    particle_reference_positions_m: object,
    marker_reference_positions_m: object,
    particle_mass_kg: object,
) -> MaterialSurfaceBinding:
    """Build W for a complete uniformly spaced Cartesian particle reference grid.

    An active axis is linearly interpolated from an enclosing pair, or the
    nearest boundary pair for no more than a half-cell extrapolation.  A
    singleton axis is a material plane and therefore admits no normal offset.
    """

    particles = _as_finite_array(
        "particle_reference_positions_m", particle_reference_positions_m, (3,)
    )
    markers = _as_finite_array(
        "marker_reference_positions_m", marker_reference_positions_m, (3,)
    )
    if particles.shape[0] == 0:
        raise ValueError("particle_reference_positions_m must not be empty")
    masses = _as_finite_vector("particle_mass_kg", particle_mass_kg, particles.shape[0])
    if np.any(masses <= 0.0):
        raise ValueError("particle_mass_kg must be strictly positive")

    axis_values = tuple(_coordinate_axis_values(particles, axis) for axis in range(3))
    lookup: dict[tuple[float, float, float], int] = {}
    for index, position in enumerate(particles):
        key = tuple(float(value) for value in position)
        if key in lookup:
            raise ValueError("particle Cartesian layout contains a duplicate particle")
        lookup[key] = index
    expected_count = int(np.prod([values.size for values in axis_values], dtype=np.int64))
    if expected_count != particles.shape[0]:
        raise ValueError("particle layout is not a complete Cartesian product (missing particle)")

    for coordinate_tuple in itertools.product(*axis_values):
        key = tuple(float(value) for value in coordinate_tuple)
        if key not in lookup:
            raise ValueError("particle layout is not a complete Cartesian product (missing particle)")
    active_axes = tuple(axis for axis, values in enumerate(axis_values) if values.size > 1)

    marker_count = markers.shape[0]
    indices = np.zeros((marker_count, 8), dtype=np.int32)
    weights = np.zeros((marker_count, 8), dtype=np.float64)
    stencil_sizes = np.zeros(marker_count, dtype=np.int32)
    marker_scale = float(np.max(np.abs(markers))) if marker_count else 0.0
    coordinate_reproduction_tolerance_m = _f64_operation_roundoff_bound(
        particles, marker_scale
    )

    for marker_index, marker in enumerate(markers):
        per_axis = tuple(
            _axis_pair_and_weights(axis_values[axis], float(marker[axis]), axis)
            for axis in range(3)
        )
        entries: list[tuple[int, float]] = []
        for combination in itertools.product(*(range(len(pair[0])) for pair in per_axis)):
            coordinate = tuple(float(axis_values[axis][per_axis[axis][0][combination[axis]]]) for axis in range(3))
            particle_index = lookup[coordinate]
            weight = float(np.prod([per_axis[axis][1][combination[axis]] for axis in range(3)]))
            entries.append((particle_index, weight))
        entries.sort(key=lambda entry: entry[0])
        count = len(entries)
        stencil_sizes[marker_index] = count
        indices[marker_index, :count] = [entry[0] for entry in entries]
        weights[marker_index, :count] = [entry[1] for entry in entries]
        row_indices = indices[marker_index, :count]
        row_weights = weights[marker_index, :count]
        # W*1 is dimensionless. Coordinate units must not tighten its f64
        # arithmetic budget; signed extrapolation uses the actual row L1 norm.
        unity_roundoff_bound = _f64_operation_roundoff_bound(float(np.sum(np.abs(row_weights))))
        if abs(float(row_weights.sum()) - 1.0) > unity_roundoff_bound:
            raise ValueError("material reference stencil failed unity reproduction")
        reconstructed = row_weights @ particles[row_indices]
        if np.max(np.abs(reconstructed - marker)) > coordinate_reproduction_tolerance_m:
            raise ValueError("material reference stencil failed affine coordinate reproduction")

    row_l1 = np.sum(np.abs(weights), axis=1) if marker_count else np.zeros(0, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_mass_gain = (
            np.sum((weights * weights) / masses[indices], axis=1)
            if marker_count
            else np.zeros(0, dtype=np.float64)
        )
    if not np.isfinite(inverse_mass_gain).all():
        raise ValueError("material reference stencil has nonfinite inverse-mass gain")
    identity = _identity_sha256(particles, markers, masses, indices, weights, stencil_sizes, active_axes)
    return MaterialSurfaceBinding(
        particle_indices=_readonly_copy(indices),
        weights=_readonly_copy(weights),
        stencil_sizes=_readonly_copy(stencil_sizes),
        particle_count=int(particles.shape[0]),
        marker_count=int(marker_count),
        reference_particle_positions_m=_readonly_copy(particles),
        reference_marker_positions_m=_readonly_copy(markers),
        particle_mass_kg=_readonly_copy(masses),
        active_axes=active_axes,
        identity_sha256=identity,
        maximum_row_l1=float(row_l1.max()) if marker_count else 0.0,
        maximum_row_inverse_mass_gain=float(inverse_mass_gain.max()) if marker_count else 0.0,
    )


def _validate_vector_field(name: str, values: object, count: int) -> np.ndarray:
    array = _as_finite_array(name, values, (3,))
    if array.shape[0] != count:
        raise ValueError(f"{name} must have shape ({count}, 3)")
    return array


def interpolate_material_surface(binding: MaterialSurfaceBinding, particle_values: object) -> np.ndarray:
    """Apply the stored reference map W to an ``(particle_count, 3)`` field."""

    values = _validate_vector_field("particle_values", particle_values, binding.particle_count)
    return np.sum(values[binding.particle_indices] * binding.weights[:, :, None], axis=1, dtype=np.float64)


def transpose_material_surface_loads(binding: MaterialSurfaceBinding, marker_forces_n: object) -> np.ndarray:
    """Apply the exact stored transpose W.T to marker loads in Newtons."""

    forces = _validate_vector_field("marker_forces_n", marker_forces_n, binding.marker_count)
    particle_forces = np.zeros((binding.particle_count, 3), dtype=np.float64)
    np.add.at(
        particle_forces,
        binding.particle_indices.ravel(),
        (binding.weights[:, :, None] * forces[:, None, :]).reshape(-1, 3),
    )
    return particle_forces
