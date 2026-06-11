from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("faces contain vertex indices outside the mesh")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True)
class UvSphereResolution:
    latitude_bands: int
    longitude_segments: int

    def __post_init__(self) -> None:
        if self.latitude_bands < 2:
            raise ValueError("latitude_bands must be at least 2")
        if self.longitude_segments < 3:
            raise ValueError("longitude_segments must be at least 3")

    @property
    def face_count(self) -> int:
        return 2 * self.longitude_segments * (self.latitude_bands - 1)

    @property
    def vertex_count(self) -> int:
        return 2 + self.longitude_segments * (self.latitude_bands - 1)


PAPER_SPHERE_MESHES: tuple[UvSphereResolution, ...] = (
    UvSphereResolution(latitude_bands=8, longitude_segments=25),
    UvSphereResolution(latitude_bands=20, longitude_segments=27),
    UvSphereResolution(latitude_bands=32, longitude_segments=61),
)

REPRODUCTION_UV_SPHERE_MESHES: tuple[UvSphereResolution, ...] = (
    *PAPER_SPHERE_MESHES,
    UvSphereResolution(latitude_bands=38, longitude_segments=105),
)


def infer_uv_sphere_resolution(mesh: SurfaceMesh) -> UvSphereResolution:
    for resolution in REPRODUCTION_UV_SPHERE_MESHES:
        if mesh.vertex_count == resolution.vertex_count and mesh.face_count == resolution.face_count:
            return resolution
    raise ValueError(
        "mesh does not match a known HIBM-MPM reproduction UV sphere resolution; "
        "device-side initialization avoids CPU vertex uploads and requires a known analytic sphere"
    )


def make_uv_sphere(resolution: UvSphereResolution, radius_m: float) -> SurfaceMesh:
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")

    lat = resolution.latitude_bands
    lon = resolution.longitude_segments
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, radius_m)]

    for i in range(1, lat):
        theta = np.pi * i / lat
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for j in range(lon):
            phi = 2.0 * np.pi * j / lon
            vertices.append(
                (
                    radius_m * sin_theta * np.cos(phi),
                    radius_m * sin_theta * np.sin(phi),
                    radius_m * cos_theta,
                )
            )

    bottom_index = len(vertices)
    vertices.append((0.0, 0.0, -radius_m))

    def ring_index(ring: int, segment: int) -> int:
        return 1 + (ring - 1) * lon + (segment % lon)

    faces: list[tuple[int, int, int]] = []
    for j in range(lon):
        faces.append((0, ring_index(1, j), ring_index(1, j + 1)))

    for ring in range(1, lat - 1):
        next_ring = ring + 1
        for j in range(lon):
            a = ring_index(ring, j)
            b = ring_index(ring, j + 1)
            c = ring_index(next_ring, j + 1)
            d = ring_index(next_ring, j)
            faces.append((a, d, c))
            faces.append((a, c, b))

    last_ring = lat - 1
    for j in range(lon):
        faces.append((bottom_index, ring_index(last_ring, j + 1), ring_index(last_ring, j)))

    return orient_faces_outward(SurfaceMesh(np.asarray(vertices), np.asarray(faces)))


def orient_faces_outward(mesh: SurfaceMesh) -> SurfaceMesh:
    vertices = mesh.vertices
    faces = mesh.faces.copy()
    if faces.size == 0:
        return SurfaceMesh(vertices, faces)
    mesh_center = vertices.mean(axis=0)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    area_normals = np.cross(b - a, c - a)
    centroids = (a + b + c) / 3.0
    inward = np.einsum("ij,ij->i", area_normals, centroids - mesh_center) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return SurfaceMesh(vertices, faces)
