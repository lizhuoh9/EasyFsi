from .hibm_mpm import (
    HibmMpmSurfaceMarkerForceReport,
    HibmMpmSurfaceMarkers,
)


def _taichi_required(name: str) -> None:
    raise NotImplementedError(
        f"{name} must be implemented as a Taichi-resident HIBM-MPM solver path; "
        "the CPU/NumPy prototype is intentionally disabled."
    )


def classify_hibm_near_boundary_nodes(*args, **kwargs):
    _taichi_required("classify_hibm_near_boundary_nodes")


def build_hibm_ib_node_boundary_conditions(*args, **kwargs):
    _taichi_required("build_hibm_ib_node_boundary_conditions")


def compute_hibm_surface_tractions(*args, **kwargs):
    _taichi_required("compute_hibm_surface_tractions")


__all__ = [
    "HibmMpmSurfaceMarkerForceReport",
    "HibmMpmSurfaceMarkers",
    "build_hibm_ib_node_boundary_conditions",
    "classify_hibm_near_boundary_nodes",
    "compute_hibm_surface_tractions",
]
