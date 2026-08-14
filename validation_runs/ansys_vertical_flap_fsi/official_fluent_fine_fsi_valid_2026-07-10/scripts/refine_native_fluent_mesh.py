"""Build a conforming fine mesh in Fluent's native ASCII ``.msh`` format.

This is an offline fallback for the Ansys vertical-flap FSI tutorial.  It does
not launch Fluent and it does not interpolate a flow solution.  The generated
mesh uses the official half-domain geometry and copies the official source
mesh's zone declarations:

* 0.100 m x 0.020 m fluid/solid half-domain;
* flap x = 0.050..0.053 m, y = 0..0.010 m;
* type-3 quadrilateral cells in ``solid.5``;
* type-1 triangular cells in ``fluid.4``;
* a conforming interface kept in ``default-interior``.  After ``solid.5`` is
  changed from fluid to solid, Fluent can split that cross-cell-zone interior
  into the coupled wall/shadow pair used by the official setup.

The default 0.25 mm spacing gives 480 solid quads and 63,040 fluid triangles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DOMAIN_X_M = 0.100
DOMAIN_Y_M = 0.020
FLAP_X0_M = 0.050
FLAP_X1_M = 0.053
FLAP_Y1_M = 0.010

SOLID_ZONE_ID = 1
FLUID_ZONE_ID = 2
FLAP_ATTACH_ZONE_ID = 3
WALL_ZONE_ID = 4
OUTLET_ZONE_ID = 5
SYMMETRY_ZONE_ID = 6
INLET_ZONE_ID = 7
INTERIOR_ZONE_ID = 9

EXPECTED_ZONE_NAMES = {
    SOLID_ZONE_ID: "solid.5",
    FLUID_ZONE_ID: "fluid.4",
    FLAP_ATTACH_ZONE_ID: "wall:008",
    WALL_ZONE_ID: "wall",
    OUTLET_ZONE_ID: "po.3",
    SYMMETRY_ZONE_ID: "symmetry.2",
    INLET_ZONE_ID: "velocity_inlet.1",
    INTERIOR_ZONE_ID: "default-interior",
}


@dataclass(frozen=True)
class Cell:
    zone_id: int
    element_type: int
    nodes: tuple[int, ...]


@dataclass(frozen=True)
class FluentAsciiMesh:
    nodes: dict[int, tuple[float, float]]
    cells: tuple[Cell, ...]
    edge_zones: dict[tuple[int, int], int]
    zone_declarations: dict[int, str]
    zone_names: dict[int, str]
    face_zone_boundary_codes: dict[int, int]


_FACE_HEADER = re.compile(
    r"^\(13 \(([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) "
    r"([0-9a-fA-F]+) ([0-9a-fA-F]+)\) \($"
)
_NODE_HEADER = re.compile(
    r"^\(10 \(([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) "
    r"([0-9a-fA-F]+)\) \($"
)
_CELL_HEADER = re.compile(
    r"^\(12 \(([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) "
    r"([0-9a-fA-F]+) ([0-9a-fA-F]+)\)\)$"
)
_ZONE_DECLARATION = re.compile(
    r"^\(45 \(([0-9a-fA-F]+) ([^ ]+) ([^ )]+)(?: [^)]*)?\)\(\)\)$"
)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _signed_area(nodes: dict[int, tuple[float, float]], ids: Iterable[int]) -> float:
    polygon = list(ids)
    return 0.5 * sum(
        nodes[polygon[index]][0] * nodes[polygon[(index + 1) % len(polygon)]][1]
        - nodes[polygon[(index + 1) % len(polygon)]][0]
        * nodes[polygon[index]][1]
        for index in range(len(polygon))
    )


def _copy_zone_template(
    source: Path,
) -> tuple[dict[int, str], dict[int, str], dict[int, int]]:
    declarations: dict[int, str] = {}
    names: dict[int, str] = {}
    boundary_codes: dict[int, int] = {}
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        zone_match = _ZONE_DECLARATION.match(line)
        if zone_match:
            zone_id = int(zone_match.group(1), 16)
            declarations[zone_id] = line
            names[zone_id] = zone_match.group(3)
            continue
        face_match = _FACE_HEADER.match(line)
        if face_match:
            zone_id = int(face_match.group(1), 16)
            if zone_id:
                boundary_codes[zone_id] = int(face_match.group(4), 16)

    if names != EXPECTED_ZONE_NAMES:
        raise ValueError(
            "The source mesh does not match the official vertical-flap zone "
            f"contract: expected {EXPECTED_ZONE_NAMES}, found {names}"
        )
    expected_face_codes = {
        FLAP_ATTACH_ZONE_ID: 3,
        WALL_ZONE_ID: 3,
        OUTLET_ZONE_ID: 5,
        SYMMETRY_ZONE_ID: 7,
        INLET_ZONE_ID: 10,
        INTERIOR_ZONE_ID: 2,
    }
    if boundary_codes != expected_face_codes:
        raise ValueError(
            "The source mesh face-zone codes drifted from the official "
            f"contract: expected {expected_face_codes}, found {boundary_codes}"
        )
    return declarations, names, boundary_codes


def _exact_divisions(length_m: float, spacing_m: float, label: str) -> int:
    divisions = round(length_m / spacing_m)
    if divisions <= 0 or not math.isclose(
        divisions * spacing_m, length_m, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"{spacing_m} m does not divide {label}={length_m} m")
    return divisions


def generate_structured_vertical_flap_mesh(
    *, spacing_m: float, zone_template_path: Path
) -> FluentAsciiMesh:
    """Generate the conforming structured mesh without invoking Fluent/Gmsh."""

    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    nx = _exact_divisions(DOMAIN_X_M, spacing_m, "domain width")
    ny = _exact_divisions(DOMAIN_Y_M, spacing_m, "domain height")
    flap_i0 = _exact_divisions(FLAP_X0_M, spacing_m, "flap x0")
    flap_i1 = _exact_divisions(FLAP_X1_M, spacing_m, "flap x1")
    flap_j1 = _exact_divisions(FLAP_Y1_M, spacing_m, "flap height")
    if nx > 5000 or ny > 5000:
        raise ValueError("requested mesh is too large for this offline generator")

    declarations, names, boundary_codes = _copy_zone_template(zone_template_path)

    def node_id(i: int, j: int) -> int:
        return j * (nx + 1) + i + 1

    nodes = {
        node_id(i, j): (i * spacing_m, j * spacing_m)
        for j in range(ny + 1)
        for i in range(nx + 1)
    }

    solid_cells: list[Cell] = []
    fluid_cells: list[Cell] = []
    for j in range(ny):
        for i in range(nx):
            bottom_left = node_id(i, j)
            bottom_right = node_id(i + 1, j)
            top_right = node_id(i + 1, j + 1)
            top_left = node_id(i, j + 1)
            if flap_i0 <= i < flap_i1 and 0 <= j < flap_j1:
                solid_cells.append(
                    Cell(
                        zone_id=SOLID_ZONE_ID,
                        element_type=3,
                        nodes=(bottom_left, bottom_right, top_right, top_left),
                    )
                )
            elif (i + j) % 2 == 0:
                fluid_cells.extend(
                    (
                        Cell(FLUID_ZONE_ID, 1, (bottom_left, bottom_right, top_right)),
                        Cell(FLUID_ZONE_ID, 1, (bottom_left, top_right, top_left)),
                    )
                )
            else:
                fluid_cells.extend(
                    (
                        Cell(FLUID_ZONE_ID, 1, (bottom_left, bottom_right, top_left)),
                        Cell(FLUID_ZONE_ID, 1, (bottom_right, top_right, top_left)),
                    )
                )
    cells = tuple([*solid_cells, *fluid_cells])

    adjacent: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cell_index, cell in enumerate(cells):
        for index, first in enumerate(cell.nodes):
            second = cell.nodes[(index + 1) % len(cell.nodes)]
            adjacent[_edge_key(first, second)].append(cell_index)

    tolerance = spacing_m * 1.0e-7

    def close(value: float, target: float) -> bool:
        return math.isclose(value, target, rel_tol=0.0, abs_tol=tolerance)

    edge_zones: dict[tuple[int, int], int] = {}
    for edge, cell_indices in adjacent.items():
        if len(cell_indices) == 2:
            edge_zones[edge] = INTERIOR_ZONE_ID
            continue
        if len(cell_indices) != 1:
            raise ValueError(f"non-manifold edge {edge}: {cell_indices}")
        first, second = (nodes[node] for node in edge)
        x0, y0 = first
        x1, y1 = second
        if close(y0, 0.0) and close(y1, 0.0):
            midpoint_x = 0.5 * (x0 + x1)
            edge_zones[edge] = (
                FLAP_ATTACH_ZONE_ID
                if FLAP_X0_M < midpoint_x < FLAP_X1_M
                else WALL_ZONE_ID
            )
        elif close(x0, 0.0) and close(x1, 0.0):
            edge_zones[edge] = INLET_ZONE_ID
        elif close(x0, DOMAIN_X_M) and close(x1, DOMAIN_X_M):
            edge_zones[edge] = OUTLET_ZONE_ID
        elif close(y0, DOMAIN_Y_M) and close(y1, DOMAIN_Y_M):
            edge_zones[edge] = SYMMETRY_ZONE_ID
        else:
            raise ValueError(f"unclassified boundary edge {edge}: {first}, {second}")

    mesh = FluentAsciiMesh(
        nodes=nodes,
        cells=cells,
        edge_zones=edge_zones,
        zone_declarations=declarations,
        zone_names=names,
        face_zone_boundary_codes=boundary_codes,
    )
    validate_mesh(mesh)
    return mesh


def _cell_ids(mesh: FluentAsciiMesh) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    ids: dict[int, int] = {}
    ranges: dict[int, tuple[int, int]] = {}
    next_id = 1
    for zone_id in (SOLID_ZONE_ID, FLUID_ZONE_ID):
        indices = [index for index, cell in enumerate(mesh.cells) if cell.zone_id == zone_id]
        if not indices:
            raise ValueError(f"cell zone {zone_id} is empty")
        first = next_id
        for index in indices:
            ids[index] = next_id
            next_id += 1
        ranges[zone_id] = (first, next_id - 1)
    if len(ids) != len(mesh.cells):
        raise ValueError("unexpected cell zone in mesh")
    return ids, ranges


def _face_records(
    mesh: FluentAsciiMesh, cell_ids: dict[int, int]
) -> dict[int, list[tuple[int, int, int, int]]]:
    occurrences: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for cell_index, cell in enumerate(mesh.cells):
        for index, first in enumerate(cell.nodes):
            second = cell.nodes[(index + 1) % len(cell.nodes)]
            occurrences[_edge_key(first, second)].append(
                (first, second, cell_ids[cell_index])
            )

    by_zone: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for edge in sorted(occurrences):
        sides = occurrences[edge]
        zone_id = mesh.edge_zones[edge]
        if len(sides) == 1:
            first, second, c0 = sides[0]
            by_zone[zone_id].append((first, second, c0, 0))
            continue
        if len(sides) != 2:
            raise ValueError(f"non-manifold edge {edge}: {sides}")
        first, second, c0 = sides[0]
        other_first, other_second, c1 = sides[1]
        if (first, second) != (other_second, other_first):
            raise ValueError(f"cell winding is inconsistent at edge {edge}")
        by_zone[zone_id].append((first, second, c0, c1))
    return {zone_id: records for zone_id, records in sorted(by_zone.items())}


def write_fluent_ascii_mesh(mesh: FluentAsciiMesh, output: Path) -> None:
    """Write a deterministic Fluent legacy-ASCII mesh."""

    validate_mesh(mesh)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing mesh: {output}")
    cell_ids, cell_ranges = _cell_ids(mesh)
    faces_by_zone = _face_records(mesh, cell_ids)
    face_count = sum(len(records) for records in faces_by_zone.values())
    lines = [
        '(0 "Offline-generated fine mesh for Ansys vertical-flap FSI")',
        '(0 "Native Fluent ASCII; no Fluent solve has been run")',
        "(2 2)",
        "",
        '(0 "Nodes:")',
        f"(10 (0 1 {len(mesh.nodes):x} 1))",
        f"(10 (c 1 {len(mesh.nodes):x} 1) (",
    ]
    for node_id in range(1, len(mesh.nodes) + 1):
        x, y = mesh.nodes[node_id]
        lines.append(f" {x:.12e}  {y:.12e}")
    lines.extend(("))", "", '(0 "Faces:")', f"(13 (0 1 {face_count:x} 0))"))

    next_face_id = 1
    for zone_id, records in faces_by_zone.items():
        first_face_id = next_face_id
        last_face_id = next_face_id + len(records) - 1
        boundary_code = mesh.face_zone_boundary_codes[zone_id]
        lines.append(
            f"(13 ({zone_id:x} {first_face_id:x} {last_face_id:x} "
            f"{boundary_code:x} 2) ("
        )
        lines.extend(
            f"{first:x} {second:x} {c0:x} {c1:x}"
            for first, second, c0, c1 in records
        )
        lines.append("))")
        next_face_id = last_face_id + 1

    lines.extend(("", '(0 "Cells:")', f"(12 (0 1 {len(mesh.cells):x} 0))"))
    for zone_id in (SOLID_ZONE_ID, FLUID_ZONE_ID):
        first, last = cell_ranges[zone_id]
        element_types = {
            cell.element_type for cell in mesh.cells if cell.zone_id == zone_id
        }
        if len(element_types) != 1:
            raise ValueError(f"mixed element types in cell zone {zone_id}")
        element_type = next(iter(element_types))
        lines.append(
            f"(12 ({zone_id:x} {first:x} {last:x} 1 {element_type:x}))"
        )

    lines.extend(("", '(0 "Zones:")'))
    lines.extend(mesh.zone_declarations[zone_id] for zone_id in sorted(mesh.zone_declarations))
    output.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def load_fluent_ascii_mesh(path: Path) -> FluentAsciiMesh:
    """Read the 2-D linear-face subset used by the official/generated meshes."""

    lines = path.read_text(encoding="ascii").splitlines()
    nodes: dict[int, tuple[float, float]] = {}
    face_rows: list[tuple[int, int, int, int, int]] = []
    cell_specs: dict[int, tuple[int, int, int]] = {}
    declarations: dict[int, str] = {}
    names: dict[int, str] = {}
    boundary_codes: dict[int, int] = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        node_match = _NODE_HEADER.match(line)
        if node_match and int(node_match.group(1), 16) != 0:
            first = int(node_match.group(2), 16)
            last = int(node_match.group(3), 16)
            index += 1
            for node_id in range(first, last + 1):
                x_text, y_text = lines[index].split()[:2]
                nodes[node_id] = (float(x_text), float(y_text))
                index += 1
            if lines[index].strip() != "))":
                raise ValueError("malformed node section")
        else:
            face_match = _FACE_HEADER.match(line)
            if face_match and int(face_match.group(1), 16) != 0:
                zone_id = int(face_match.group(1), 16)
                first = int(face_match.group(2), 16)
                last = int(face_match.group(3), 16)
                boundary_codes[zone_id] = int(face_match.group(4), 16)
                if int(face_match.group(5), 16) != 2:
                    raise ValueError("only linear 2-node faces are supported")
                index += 1
                for _ in range(first, last + 1):
                    n0, n1, c0, c1 = (
                        int(value, 16) for value in lines[index].split()[:4]
                    )
                    face_rows.append((zone_id, n0, n1, c0, c1))
                    index += 1
                if lines[index].strip() != "))":
                    raise ValueError("malformed face section")
            else:
                cell_match = _CELL_HEADER.match(line)
                if cell_match and int(cell_match.group(1), 16) != 0:
                    zone_id = int(cell_match.group(1), 16)
                    cell_specs[zone_id] = (
                        int(cell_match.group(2), 16),
                        int(cell_match.group(3), 16),
                        int(cell_match.group(5), 16),
                    )
                zone_match = _ZONE_DECLARATION.match(line)
                if zone_match:
                    zone_id = int(zone_match.group(1), 16)
                    declarations[zone_id] = line
                    names[zone_id] = zone_match.group(3)
        index += 1

    cell_zone_by_id: dict[int, tuple[int, int]] = {}
    for zone_id, (first, last, element_type) in cell_specs.items():
        for cell_id in range(first, last + 1):
            cell_zone_by_id[cell_id] = (zone_id, element_type)
    node_sets: dict[int, set[int]] = defaultdict(set)
    edge_zones: dict[tuple[int, int], int] = {}
    for zone_id, n0, n1, c0, c1 in face_rows:
        edge_zones[_edge_key(n0, n1)] = zone_id
        for cell_id in (c0, c1):
            if cell_id:
                node_sets[cell_id].update((n0, n1))

    cells: list[Cell] = []
    for cell_id in sorted(cell_zone_by_id):
        zone_id, element_type = cell_zone_by_id[cell_id]
        cell_nodes = node_sets[cell_id]
        expected_count = 3 if element_type == 1 else 4 if element_type == 3 else -1
        if len(cell_nodes) != expected_count:
            raise ValueError(
                f"cell {cell_id} type {element_type} has {len(cell_nodes)} nodes"
            )
        center_x = sum(nodes[node][0] for node in cell_nodes) / len(cell_nodes)
        center_y = sum(nodes[node][1] for node in cell_nodes) / len(cell_nodes)
        ordered = tuple(
            sorted(
                cell_nodes,
                key=lambda node: math.atan2(
                    nodes[node][1] - center_y, nodes[node][0] - center_x
                ),
            )
        )
        cells.append(Cell(zone_id, element_type, ordered))
    return FluentAsciiMesh(
        nodes=nodes,
        cells=tuple(cells),
        edge_zones=edge_zones,
        zone_declarations=declarations,
        zone_names=names,
        face_zone_boundary_codes=boundary_codes,
    )


def validate_mesh(mesh: FluentAsciiMesh) -> dict[str, object]:
    if sorted(mesh.nodes) != list(range(1, len(mesh.nodes) + 1)):
        raise ValueError("node ids must be contiguous and one-based")
    if mesh.zone_names != EXPECTED_ZONE_NAMES:
        raise ValueError(f"zone-name contract mismatch: {mesh.zone_names}")

    adjacent: dict[tuple[int, int], list[int]] = defaultdict(list)
    areas: list[float] = []
    cell_counts: Counter[int] = Counter()
    cell_types: dict[int, Counter[int]] = defaultdict(Counter)
    for index, cell in enumerate(mesh.cells):
        expected_nodes = 3 if cell.element_type == 1 else 4 if cell.element_type == 3 else -1
        if len(cell.nodes) != expected_nodes:
            raise ValueError(f"unsupported/malformed cell: {cell}")
        area = _signed_area(mesh.nodes, cell.nodes)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError(f"non-positive cell area {area}: {cell}")
        areas.append(area)
        cell_counts[cell.zone_id] += 1
        cell_types[cell.zone_id][cell.element_type] += 1
        for node_index, first in enumerate(cell.nodes):
            second = cell.nodes[(node_index + 1) % len(cell.nodes)]
            adjacent[_edge_key(first, second)].append(index)

    if set(adjacent) != set(mesh.edge_zones):
        raise ValueError("edge-zone map does not cover exactly the cell edges")
    cross_zone_faces = 0
    face_counts: Counter[int] = Counter()
    for edge, cell_indices in adjacent.items():
        zone_id = mesh.edge_zones[edge]
        face_counts[zone_id] += 1
        if len(cell_indices) not in (1, 2):
            raise ValueError(f"non-manifold edge {edge}: {cell_indices}")
        if len(cell_indices) == 1 and zone_id == INTERIOR_ZONE_ID:
            raise ValueError(f"boundary edge {edge} is marked interior")
        if len(cell_indices) == 2:
            if zone_id != INTERIOR_ZONE_ID:
                raise ValueError(f"internal edge {edge} is in boundary zone {zone_id}")
            zones = {mesh.cells[index].zone_id for index in cell_indices}
            cross_zone_faces += len(zones) == 2

    return {
        "node_count": len(mesh.nodes),
        "face_count": len(adjacent),
        "cell_count": len(mesh.cells),
        "cell_counts_by_zone_name": {
            mesh.zone_names[zone_id]: count for zone_id, count in sorted(cell_counts.items())
        },
        "cell_types_by_zone_name": {
            mesh.zone_names[zone_id]: dict(sorted(type_counts.items()))
            for zone_id, type_counts in sorted(cell_types.items())
        },
        "face_counts_by_zone_name": {
            mesh.zone_names[zone_id]: count for zone_id, count in sorted(face_counts.items())
        },
        "cross_cell_zone_face_count": cross_zone_faces,
        "minimum_cell_area_m2": min(areas),
        "maximum_cell_area_m2": max(areas),
        "bounds_m": {
            "x_min": min(x for x, _ in mesh.nodes.values()),
            "x_max": max(x for x, _ in mesh.nodes.values()),
            "y_min": min(y for _, y in mesh.nodes.values()),
            "y_max": max(y for _, y in mesh.nodes.values()),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone-template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--spacing-m", type=float, default=0.00025)
    args = parser.parse_args()

    mesh = generate_structured_vertical_flap_mesh(
        spacing_m=args.spacing_m,
        zone_template_path=args.zone_template,
    )
    report = validate_mesh(mesh)
    write_fluent_ascii_mesh(mesh, args.output)
    roundtrip = validate_mesh(load_fluent_ascii_mesh(args.output))
    if roundtrip != report:
        raise RuntimeError("native Fluent ASCII round-trip validation changed the mesh")

    manifest = {
        "status": "offline_mesh_generated_not_fluent_import_validated",
        "generator": str(Path(__file__).resolve()),
        "source_zone_template": str(args.zone_template.resolve()),
        "source_zone_template_sha256": _sha256(args.zone_template),
        "output_mesh": str(args.output.resolve()),
        "output_mesh_sha256": _sha256(args.output),
        "spacing_m": args.spacing_m,
        "geometry_m": {
            "domain": [DOMAIN_X_M, DOMAIN_Y_M],
            "flap_x": [FLAP_X0_M, FLAP_X1_M],
            "flap_y": [0.0, FLAP_Y1_M],
        },
        "validation": report,
        "next_gate": (
            "Read mesh in Fluent, change solid.5 to solid, rename the generated "
            "cross-zone interior to flap_wall, then require a 1-step nonzero "
            "intrinsic-structure displacement before any 50-step production run."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.manifest}")
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
