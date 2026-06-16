from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .interface_pair import InterfacePairMap, Vector3


@dataclass(frozen=True)
class MovingBoundaryCondition:
    name: str
    pair_map: InterfacePairMap
    transfer_mode: str = "full"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.transfer_mode not in {"full", "normal"}:
            raise ValueError("transfer_mode must be 'full' or 'normal'")

    def mesh_displacements(
        self,
        source_displacements_m: Sequence[Sequence[float]],
        *,
        target_normals: Sequence[Sequence[float]] | None = None,
    ) -> tuple[Vector3, ...]:
        if self.transfer_mode == "full":
            return self.pair_map.map_vectors(source_displacements_m)
        if target_normals is None:
            raise ValueError("target_normals are required for normal transfer mode")
        return self.pair_map.map_normal_vectors(
            source_vectors=source_displacements_m,
            target_normals=target_normals,
        )
