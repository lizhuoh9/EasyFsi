from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class PairMapEntry:
    target_index: int
    source_index: int
    weight: float

    def __post_init__(self) -> None:
        if self.target_index < 0:
            raise ValueError("target_index must be non-negative")
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if not math.isfinite(float(self.weight)):
            raise ValueError("weight must be finite")


@dataclass(frozen=True)
class InterfacePairMap:
    target_count: int
    entries: tuple[PairMapEntry, ...]

    def __post_init__(self) -> None:
        if self.target_count <= 0:
            raise ValueError("target_count must be positive")
        entries = tuple(self.entries)
        if len(entries) == 0:
            raise ValueError("entries must be non-empty")
        for entry in entries:
            if entry.target_index >= self.target_count:
                raise ValueError("entry target_index exceeds target_count")
        object.__setattr__(self, "entries", entries)

    def map_vectors(self, source_vectors: Sequence[Sequence[float]]) -> tuple[Vector3, ...]:
        source = tuple(_vector3(vector, name="source_vectors") for vector in source_vectors)
        mapped = [[0.0, 0.0, 0.0] for _ in range(self.target_count)]
        for entry in self.entries:
            if entry.source_index >= len(source):
                raise ValueError("entry source_index exceeds source vector count")
            for axis in range(3):
                mapped[entry.target_index][axis] += (
                    float(entry.weight) * source[entry.source_index][axis]
                )
        return tuple(_list_to_vector3(vector) for vector in mapped)

    def map_normal_vectors(
        self,
        *,
        source_vectors: Sequence[Sequence[float]],
        target_normals: Sequence[Sequence[float]],
    ) -> tuple[Vector3, ...]:
        mapped = self.map_vectors(source_vectors)
        normals = tuple(_unit_vector3(vector, name="target_normals") for vector in target_normals)
        if len(normals) != self.target_count:
            raise ValueError("target_normals length must match target_count")
        normal_only = []
        for vector, normal in zip(mapped, normals):
            component = sum(value * axis for value, axis in zip(vector, normal))
            normal_only.append(tuple(component * axis for axis in normal))
        return tuple(normal_only)

    def transpose_forces(
        self,
        *,
        target_forces: Sequence[Sequence[float]],
        source_count: int,
        action_reaction_sign: float = -1.0,
    ) -> tuple[Vector3, ...]:
        if source_count <= 0:
            raise ValueError("source_count must be positive")
        sign = float(action_reaction_sign)
        if not math.isfinite(sign):
            raise ValueError("action_reaction_sign must be finite")
        target = tuple(_vector3(vector, name="target_forces") for vector in target_forces)
        if len(target) != self.target_count:
            raise ValueError("target_forces length must match target_count")
        source_forces = [[0.0, 0.0, 0.0] for _ in range(source_count)]
        for entry in self.entries:
            if entry.source_index >= source_count:
                raise ValueError("entry source_index exceeds source_count")
            force = target[entry.target_index]
            for axis in range(3):
                source_forces[entry.source_index][axis] += (
                    sign * float(entry.weight) * force[axis]
                )
        return tuple(_list_to_vector3(vector) for vector in source_forces)


def _vector3(values: Sequence[float], *, name: str) -> Vector3:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} components must be finite")
    return (vector[0], vector[1], vector[2])


def _unit_vector3(values: Sequence[float], *, name: str) -> Vector3:
    vector = _vector3(values, name=name)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1.0e-30:
        raise ValueError(f"{name} must contain non-zero vectors")
    return tuple(value / norm for value in vector)


def _list_to_vector3(values: Sequence[float]) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))
