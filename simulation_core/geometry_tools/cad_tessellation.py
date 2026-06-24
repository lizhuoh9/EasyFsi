from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


STEP_EXTENSIONS = {".step", ".stp"}
DEFAULT_STEP_RELATIVE_EDGE_LENGTH = 0.008
DEFAULT_STEP_CURVATURE_ELEMENTS_PER_2PI = 48.0
DEFAULT_STEP_MIN_CIRCLE_NODES = 24
DEFAULT_STEP_MIN_CURVE_NODES = 12
MIN_STEP_RELATIVE_EDGE_LENGTH = 1.0e-4
MIN_STEP_TARGET_EDGE_LENGTH = 1.0e-6


@dataclass(frozen=True)
class StepSurfaceEntity:
    entity_tag: int
    face_ids: np.ndarray
    centroid_m: np.ndarray
    name: str = ""


@dataclass(frozen=True)
class StepPartEntity:
    entity_dim: int
    entity_tag: int
    name: str
    surface_tags: tuple[int, ...]
    face_ids: np.ndarray
    centroid_m: np.ndarray


@dataclass(frozen=True)
class StepCurveEntity:
    entity_tag: int
    edge_vertex_pairs: np.ndarray
    face_ids: np.ndarray
    centroid_m: np.ndarray
    name: str = ""


@dataclass
class StepTessellationResult:
    mesh: trimesh.Trimesh
    surface_entities: list[StepSurfaceEntity] = field(default_factory=list)
    part_entities: list[StepPartEntity] = field(default_factory=list)
    curve_entities: list[StepCurveEntity] = field(default_factory=list)


@dataclass(frozen=True)
class StepTessellationSettings:
    relative_edge_length: float = DEFAULT_STEP_RELATIVE_EDGE_LENGTH
    curvature_elements_per_2pi: float = DEFAULT_STEP_CURVATURE_ELEMENTS_PER_2PI
    min_circle_nodes: int = DEFAULT_STEP_MIN_CIRCLE_NODES
    min_curve_nodes: int = DEFAULT_STEP_MIN_CURVE_NODES

    def __post_init__(self) -> None:
        if self.relative_edge_length <= 0.0:
            raise ValueError("relative_edge_length must be positive.")
        if self.curvature_elements_per_2pi < 0.0:
            raise ValueError("curvature_elements_per_2pi must be non-negative.")
        if self.min_circle_nodes < 3:
            raise ValueError("min_circle_nodes must be at least 3.")
        if self.min_curve_nodes < 2:
            raise ValueError("min_curve_nodes must be at least 2.")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_edge_length": float(self.relative_edge_length),
            "curvature_elements_per_2pi": float(self.curvature_elements_per_2pi),
            "min_circle_nodes": int(self.min_circle_nodes),
            "min_curve_nodes": int(self.min_curve_nodes),
        }

    @classmethod
    def from_mapping(
        cls,
        raw_data: Mapping[str, Any] | None,
    ) -> "StepTessellationSettings":
        if raw_data is None:
            return cls()
        return cls(
            relative_edge_length=float(
                raw_data.get("relative_edge_length", DEFAULT_STEP_RELATIVE_EDGE_LENGTH)
            ),
            curvature_elements_per_2pi=float(
                raw_data.get(
                    "curvature_elements_per_2pi",
                    DEFAULT_STEP_CURVATURE_ELEMENTS_PER_2PI,
                )
            ),
            min_circle_nodes=int(
                raw_data.get("min_circle_nodes", DEFAULT_STEP_MIN_CIRCLE_NODES)
            ),
            min_curve_nodes=int(
                raw_data.get("min_curve_nodes", DEFAULT_STEP_MIN_CURVE_NODES)
            ),
        )


def is_step_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in STEP_EXTENSIONS


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _map_gmsh_node_tags_to_vertex_indices(
    node_tags: np.ndarray,
    node_block: np.ndarray,
    element_label: str,
) -> np.ndarray:
    tags = np.asarray(node_tags, dtype=np.int64).reshape(-1)
    block = np.asarray(node_block, dtype=np.int64)
    tag_to_vertex_index = {int(tag): int(index) for index, tag in enumerate(tags)}
    mapped_flat: list[int] = []
    for tag in block.reshape(-1):
        try:
            mapped_flat.append(tag_to_vertex_index[int(tag)])
        except KeyError as exc:
            raise ValueError(
                f"{element_label} references unknown gmsh node tag {int(tag)}."
            ) from exc
    return np.asarray(mapped_flat, dtype=np.int64).reshape(block.shape)


def normalize_step_curve_edge_pairs(
    edge_vertex_pairs: np.ndarray | Sequence[Sequence[int]] | None,
) -> np.ndarray:
    if edge_vertex_pairs is None:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.asarray(edge_vertex_pairs, dtype=np.int64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = pairs.reshape(-1, 2)
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(pairs, axis=0)


def build_triangle_edge_face_map(
    face_array: np.ndarray | Sequence[Sequence[int]],
) -> dict[tuple[int, int], tuple[int, ...]]:
    faces = np.asarray(face_array, dtype=np.int64)
    if faces.size == 0:
        return {}
    faces = faces.reshape(-1, 3)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, tri in enumerate(faces):
        v0, v1, v2 = (int(tri[0]), int(tri[1]), int(tri[2]))
        for edge in ((v0, v1), (v1, v2), (v2, v0)):
            key = tuple(sorted(edge))
            edge_to_faces.setdefault(key, []).append(int(face_id))
    return {key: tuple(face_ids) for key, face_ids in edge_to_faces.items()}


def face_ids_for_step_curve_edges(
    edge_to_faces: Mapping[tuple[int, int], Sequence[int]],
    edge_vertex_pairs: np.ndarray | Sequence[Sequence[int]] | None,
) -> np.ndarray:
    normalized_pairs = normalize_step_curve_edge_pairs(edge_vertex_pairs)
    if normalized_pairs.size == 0:
        return np.zeros(0, dtype=np.intp)
    curve_face_ids: list[int] = []
    for edge_v0, edge_v1 in normalized_pairs:
        curve_face_ids.extend(
            int(face_id)
            for face_id in edge_to_faces.get((int(edge_v0), int(edge_v1)), ())
        )
    if not curve_face_ids:
        return np.zeros(0, dtype=np.intp)
    return np.unique(np.asarray(curve_face_ids, dtype=np.intp))


def resolve_step_part_face_claims(
    part_entities: Sequence[StepPartEntity],
) -> list[StepPartEntity]:
    resolved: list[StepPartEntity] = []
    claimed_faces: dict[int, tuple[int, int]] = {}

    def priority(entity: StepPartEntity) -> int:
        return 0 if int(entity.entity_dim) == 3 else 1

    for entity in sorted(
        list(part_entities),
        key=lambda item: (priority(item), int(item.entity_dim), int(item.entity_tag)),
    ):
        raw_faces = np.asarray(entity.face_ids, dtype=np.intp).reshape(-1)
        if raw_faces.size == 0:
            continue
        unique_faces = np.unique(raw_faces)
        kept_faces: list[int] = []
        conflicts: list[tuple[int, tuple[int, int]]] = []
        for face_id_raw in unique_faces:
            face_id = int(face_id_raw)
            previous = claimed_faces.get(face_id)
            if previous is None:
                kept_faces.append(face_id)
            else:
                conflicts.append((face_id, previous))

        if conflicts:
            previous_priorities = {
                0 if int(previous[0]) == 3 else 1 for _face, previous in conflicts
            }
            if int(entity.entity_dim) == 2 and previous_priorities == {0}:
                if len(kept_faces) < 3:
                    continue
            else:
                first_face, previous = conflicts[0]
                raise ValueError(
                    "STEP part face overlap detected: entity "
                    f"({int(entity.entity_dim)}, {int(entity.entity_tag)}) overlaps "
                    f"entity ({int(previous[0])}, {int(previous[1])}) on "
                    f"{len(conflicts)} face(s); sample face id {int(first_face)}."
                )

        if len(kept_faces) < 3:
            continue
        kept_array = np.asarray(kept_faces, dtype=np.intp)
        for face_id in kept_array:
            claimed_faces[int(face_id)] = (int(entity.entity_dim), int(entity.entity_tag))
        resolved.append(
            StepPartEntity(
                entity_dim=int(entity.entity_dim),
                entity_tag=int(entity.entity_tag),
                name=str(entity.name),
                surface_tags=tuple(int(tag) for tag in entity.surface_tags),
                face_ids=kept_array,
                centroid_m=np.asarray(entity.centroid_m, dtype=float),
            )
        )
    return resolved


def tessellate_step_cad(
    step_path: str | Path,
    settings: StepTessellationSettings | None = None,
) -> StepTessellationResult:
    import gmsh

    effective_settings = settings if settings is not None else StepTessellationSettings()
    cad_path = Path(step_path).resolve()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(cad_path.stem)
        imported_entities = gmsh.model.occ.importShapes(str(cad_path))
        if not imported_entities:
            raise ValueError(f"Failed to import STEP shape from {cad_path}.")
        gmsh.model.occ.synchronize()

        bounds = gmsh.model.getBoundingBox(-1, -1)
        bounds_min = np.asarray(bounds[:3], dtype=float)
        bounds_max = np.asarray(bounds[3:], dtype=float)
        bbox_diagonal = float(np.linalg.norm(bounds_max - bounds_min))
        target_edge_length = max(
            bbox_diagonal
            * max(
                effective_settings.relative_edge_length,
                MIN_STEP_RELATIVE_EDGE_LENGTH,
            ),
            MIN_STEP_TARGET_EDGE_LENGTH,
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature",
            float(effective_settings.curvature_elements_per_2pi),
        )
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_edge_length)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_edge_length)
        gmsh.option.setNumber("Mesh.MinCircleNodes", int(effective_settings.min_circle_nodes))
        gmsh.option.setNumber("Mesh.MinCurveNodes", int(effective_settings.min_curve_nodes))
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise ValueError(f"STEP tessellation for {cad_path} produced zero nodes.")
        vertices = np.asarray(coords, dtype=float).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=np.int64)

        all_faces: list[np.ndarray] = []
        entity_face_lists: list[tuple[int, list[int]]] = []
        global_face_offset = 0
        for _dim, entity_tag in gmsh.model.getEntities(dim=2):
            element_types, _, element_node_tags = gmsh.model.mesh.getElements(
                2,
                entity_tag,
            )
            entity_local_faces: list[int] = []
            for element_type, node_block in zip(element_types, element_node_tags):
                if int(element_type) != 2:
                    continue
                node_block = np.asarray(node_block, dtype=np.int64).reshape(-1, 3)
                mapped_faces = _map_gmsh_node_tags_to_vertex_indices(
                    node_tags,
                    node_block,
                    f"surface entity {int(entity_tag)} triangles",
                )
                local_ids = list(
                    range(global_face_offset, global_face_offset + len(mapped_faces))
                )
                entity_local_faces.extend(local_ids)
                global_face_offset += len(mapped_faces)
                all_faces.append(mapped_faces.astype(np.int64))
            if entity_local_faces:
                entity_face_lists.append((int(entity_tag), entity_local_faces))

        if not all_faces:
            raise ValueError(f"STEP tessellation for {cad_path} produced zero triangles.")
        face_array = np.vstack(all_faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=face_array, process=False)
        tri_centers = np.asarray(mesh.triangles_center, dtype=float)
        edge_to_faces = build_triangle_edge_face_map(face_array)

        surface_entities: list[StepSurfaceEntity] = []
        surface_face_map: dict[int, np.ndarray] = {}
        for entity_tag, face_ids_list in entity_face_lists:
            face_ids = np.asarray(face_ids_list, dtype=np.intp)
            valid_face_ids = face_ids[face_ids < len(tri_centers)]
            if valid_face_ids.size == 0:
                continue
            try:
                raw_name = gmsh.model.getEntityName(2, int(entity_tag))
            except Exception:
                raw_name = ""
            surface_face_map[int(entity_tag)] = valid_face_ids
            surface_entities.append(
                StepSurfaceEntity(
                    entity_tag=int(entity_tag),
                    face_ids=valid_face_ids,
                    centroid_m=tri_centers[valid_face_ids].mean(axis=0),
                    name=(raw_name or "").strip(),
                )
            )

        curve_entities: list[StepCurveEntity] = []
        for _dim, entity_tag in gmsh.model.getEntities(dim=1):
            element_types, _, element_node_tags = gmsh.model.mesh.getElements(
                1,
                entity_tag,
            )
            edge_blocks: list[np.ndarray] = []
            for element_type, node_block in zip(element_types, element_node_tags):
                if int(element_type) != 1:
                    continue
                node_block = np.asarray(node_block, dtype=np.int64).reshape(-1, 2)
                mapped_edges = _map_gmsh_node_tags_to_vertex_indices(
                    node_tags,
                    node_block,
                    f"curve entity {int(entity_tag)} edges",
                )
                edge_blocks.append(mapped_edges.astype(np.int64))
            if not edge_blocks:
                continue
            edge_vertex_pairs = normalize_step_curve_edge_pairs(np.vstack(edge_blocks))
            if edge_vertex_pairs.size == 0:
                continue
            face_ids = face_ids_for_step_curve_edges(edge_to_faces, edge_vertex_pairs)
            unique_vertex_ids = np.unique(edge_vertex_pairs.reshape(-1))
            if unique_vertex_ids.size == 0:
                continue
            try:
                raw_name = gmsh.model.getEntityName(1, int(entity_tag))
            except Exception:
                raw_name = ""
            curve_entities.append(
                StepCurveEntity(
                    entity_tag=int(entity_tag),
                    edge_vertex_pairs=edge_vertex_pairs,
                    face_ids=face_ids,
                    centroid_m=vertices[unique_vertex_ids].mean(axis=0),
                    name=(raw_name or "").strip(),
                )
            )

        part_entities: list[StepPartEntity] = []
        seen_entities: set[tuple[int, int]] = set()
        part_counter = 1
        for entity_dim, entity_tag in imported_entities:
            entity_key = (int(entity_dim), int(entity_tag))
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)
            if int(entity_dim) == 3:
                boundary_entities = gmsh.model.getBoundary(
                    [entity_key],
                    oriented=False,
                    recursive=False,
                )
                surface_tags = [
                    int(tag) for dim, tag in boundary_entities if int(dim) == 2
                ]
            elif int(entity_dim) == 2:
                surface_tags = [int(entity_tag)]
            else:
                continue
            face_blocks = [
                surface_face_map[tag] for tag in surface_tags if tag in surface_face_map
            ]
            if not face_blocks:
                continue
            part_face_ids = np.unique(np.concatenate(face_blocks).astype(np.intp))
            if part_face_ids.size == 0:
                continue
            try:
                raw_name = gmsh.model.getEntityName(int(entity_dim), int(entity_tag))
            except Exception:
                raw_name = ""
            part_name = (raw_name or "").strip() or f"STEP Part {part_counter}"
            part_counter += 1
            part_entities.append(
                StepPartEntity(
                    entity_dim=int(entity_dim),
                    entity_tag=int(entity_tag),
                    name=part_name,
                    surface_tags=tuple(surface_tags),
                    face_ids=part_face_ids,
                    centroid_m=tri_centers[part_face_ids].mean(axis=0),
                )
            )

        return StepTessellationResult(
            mesh=mesh,
            surface_entities=surface_entities,
            part_entities=resolve_step_part_face_claims(part_entities),
            curve_entities=curve_entities,
        )
    finally:
        gmsh.finalize()


def _face_ids_by_entity_tag(
    entities: Sequence[StepSurfaceEntity | StepCurveEntity | StepPartEntity],
) -> dict[int, np.ndarray]:
    return {
        int(entity.entity_tag): np.asarray(entity.face_ids, dtype=np.intp)
        for entity in entities
    }


def _unique_face_ids(face_blocks: Sequence[np.ndarray]) -> list[int]:
    valid_blocks = [
        np.asarray(block, dtype=np.intp).reshape(-1)
        for block in face_blocks
        if np.asarray(block, dtype=np.intp).size > 0
    ]
    if not valid_blocks:
        return []
    return [int(value) for value in np.unique(np.concatenate(valid_blocks))]


def face_ids_for_step_surface_tags(
    result: StepTessellationResult,
    tags: Sequence[int],
) -> list[int]:
    by_tag = _face_ids_by_entity_tag(result.surface_entities)
    return _unique_face_ids([by_tag[int(tag)] for tag in tags if int(tag) in by_tag])


def face_ids_for_step_curve_tags(
    result: StepTessellationResult,
    tags: Sequence[int],
) -> list[int]:
    by_tag = _face_ids_by_entity_tag(result.curve_entities)
    return _unique_face_ids([by_tag[int(tag)] for tag in tags if int(tag) in by_tag])


def face_ids_for_step_part_or_surface_tags(
    result: StepTessellationResult,
    tags: Sequence[int],
) -> list[int]:
    tag_set = {int(tag) for tag in tags}
    surface_by_tag = _face_ids_by_entity_tag(result.surface_entities)
    part_blocks = [
        np.asarray(part.face_ids, dtype=np.intp)
        for part in result.part_entities
        if set(int(tag) for tag in part.surface_tags).issubset(tag_set)
        and len(part.surface_tags) > 0
    ]
    if part_blocks:
        return _unique_face_ids(part_blocks)
    return _unique_face_ids(
        [surface_by_tag[tag] for tag in tag_set if tag in surface_by_tag]
    )


def _selection_source_tags(selection: Mapping[str, object]) -> tuple[str, tuple[int, ...]]:
    source = selection.get("selection_source", {})
    if not isinstance(source, Mapping):
        return "", ()
    kind = str(source.get("kind", "")).strip().lower()
    raw_tags = source.get("cad_tags", ())
    if not isinstance(raw_tags, Sequence) or isinstance(raw_tags, (str, bytes)):
        return kind, ()
    return kind, tuple(int(tag) for tag in raw_tags)


def remap_step_named_selection_face_ids(
    config: Mapping[str, object],
    result: StepTessellationResult,
) -> dict[str, object]:
    remapped = copy.deepcopy(dict(config))
    selections = remapped.get("named_selections", [])
    if not isinstance(selections, list):
        return remapped

    for selection in selections:
        if not isinstance(selection, dict):
            continue
        kind, tags = _selection_source_tags(selection)
        if not tags:
            continue
        if kind == "step_surface":
            face_ids = face_ids_for_step_surface_tags(result, tags)
        elif kind == "step_curve_loop":
            face_ids = face_ids_for_step_curve_tags(result, tags)
        elif kind == "step_part":
            face_ids = face_ids_for_step_part_or_surface_tags(result, tags)
        else:
            continue
        if not face_ids:
            raise ValueError(
                "STEP named selection remap produced zero faces for "
                f"selection {selection.get('id')} with kind {kind!r} and tags {tags}."
            )
        selection["face_ids"] = face_ids
        selection["source_mesh_face_count"] = int(len(face_ids))
    return remapped


def step_tessellation_report(result: StepTessellationResult) -> dict[str, object]:
    return {
        "mesh_vertex_count": int(result.mesh.vertices.shape[0]),
        "mesh_face_count": int(result.mesh.faces.shape[0]),
        "surface_count": int(len(result.surface_entities)),
        "curve_count": int(len(result.curve_entities)),
        "part_count": int(len(result.part_entities)),
        "surface_face_counts": {
            str(entity.entity_tag): int(len(entity.face_ids))
            for entity in result.surface_entities
        },
        "curve_face_counts": {
            str(entity.entity_tag): int(len(entity.face_ids))
            for entity in result.curve_entities
        },
        "part_face_counts": {
            str(entity.entity_tag): int(len(entity.face_ids))
            for entity in result.part_entities
        },
    }


def write_step_surface_mesh_cache(
    result: StepTessellationResult,
    cache_path: str | Path,
) -> str:
    output_path = Path(cache_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(output_path)
    return file_sha256(output_path)


def build_step_derived_source_config(
    source_config: Mapping[str, object],
    *,
    step_path: str | Path,
    step_sha256: str,
    surface_mesh_cache_path: str | Path,
    surface_mesh_cache_sha256: str,
    tessellation_settings: Mapping[str, object],
    mesh_scale_to_m: float,
    tessellation_report: Mapping[str, object],
) -> dict[str, object]:
    config = copy.deepcopy(dict(source_config))
    config["mesh_format"] = "step"
    config["mesh_path"] = str(step_path)
    config["surface_mesh_cache_path"] = str(surface_mesh_cache_path)
    config["mesh_scale_to_m"] = float(mesh_scale_to_m)

    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata["source_step"] = str(step_path)
    metadata["source_step_sha256"] = str(step_sha256)
    config["metadata"] = metadata

    config["mesh_import"] = {
        "kind": "step_surface_tessellation",
        "source_step_path": str(step_path),
        "source_step_sha256": str(step_sha256),
        "surface_mesh_cache_path": str(surface_mesh_cache_path),
        "surface_mesh_cache_sha256": str(surface_mesh_cache_sha256),
        "tessellation_settings": dict(tessellation_settings),
        "tessellation_report": dict(tessellation_report),
    }
    return config


__all__ = [
    "StepCurveEntity",
    "StepPartEntity",
    "StepSurfaceEntity",
    "StepTessellationResult",
    "StepTessellationSettings",
    "build_step_derived_source_config",
    "build_triangle_edge_face_map",
    "face_ids_for_step_curve_edges",
    "face_ids_for_step_curve_tags",
    "face_ids_for_step_part_or_surface_tags",
    "face_ids_for_step_surface_tags",
    "file_sha256",
    "is_step_path",
    "normalize_step_curve_edge_pairs",
    "remap_step_named_selection_face_ids",
    "resolve_step_part_face_claims",
    "step_tessellation_report",
    "tessellate_step_cad",
    "write_step_surface_mesh_cache",
]
