from __future__ import annotations

from simulation_core.cad_import import (
    StepBrep,
    StepCadSummary,
    cad_provenance_report,
    parse_step_cad_summary,
)
from simulation_core.cad_tessellation import (
    StepCurveEntity,
    StepPartEntity,
    StepSurfaceEntity,
    StepTessellationResult,
    StepTessellationSettings,
    build_step_derived_source_config,
    remap_step_named_selection_face_ids,
    step_tessellation_report,
    tessellate_step_cad,
    write_step_surface_mesh_cache,
)
from simulation_core.coordinate_models import (
    Axisymmetric2DCoordinateModel,
    Cartesian2DCoordinateModel,
    Cartesian3DCoordinateModel,
    CoordinateModel,
)
from simulation_core.fluid_domain import (
    AxisAlignedBoundary,
    BoundaryRegion,
    FluidDomain,
)
from simulation_core.geometry import (
    PAPER_SPHERE_MESHES,
    REPRODUCTION_UV_SPHERE_MESHES,
    SurfaceMesh,
    UvSphereResolution,
    infer_uv_sphere_resolution,
    make_uv_sphere,
    orient_faces_outward,
)

__all__ = [
    "AxisAlignedBoundary",
    "Axisymmetric2DCoordinateModel",
    "BoundaryRegion",
    "Cartesian2DCoordinateModel",
    "Cartesian3DCoordinateModel",
    "CoordinateModel",
    "FluidDomain",
    "PAPER_SPHERE_MESHES",
    "REPRODUCTION_UV_SPHERE_MESHES",
    "StepBrep",
    "StepCadSummary",
    "StepCurveEntity",
    "StepPartEntity",
    "StepSurfaceEntity",
    "StepTessellationResult",
    "StepTessellationSettings",
    "SurfaceMesh",
    "UvSphereResolution",
    "build_step_derived_source_config",
    "cad_provenance_report",
    "infer_uv_sphere_resolution",
    "make_uv_sphere",
    "orient_faces_outward",
    "parse_step_cad_summary",
    "remap_step_named_selection_face_ids",
    "step_tessellation_report",
    "tessellate_step_cad",
    "write_step_surface_mesh_cache",
]
