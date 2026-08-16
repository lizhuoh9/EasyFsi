import unittest

import numpy as np
import taichi as ti

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)
from tests.solvers._hibm_component_face_ledger_contracts import (
    CanonicalComponentFaceLedgerContractMixin,
    _ComponentFaceClaim,
)


@ti.kernel
def _relocation_source_linear_key_probe(
    boundary: ti.template(),
    result: ti.template(),
):
    result[None] = boundary._velocity_dirichlet_relocation_source_linear_key(
        ti.Vector([1, 2, 3]),
        4,
        5,
    )


@ti.kernel
def _r22_component_face_storage_probe(
    boundary: ti.template(),
    velocity_field: ti.template(),
    obstacle_field: ti.template(),
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    cell_center_x_m: ti.template(),
    cell_center_y_m: ti.template(),
    cell_center_z_m: ti.template(),
    target_i: ti.i32,
    target_j: ti.i32,
    target_k: ti.i32,
    component_axis: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    result_i32: ti.template(),
    result_f64: ti.template(),
):
    target = ti.Vector([target_i, target_j, target_k])
    geometry_base = target
    geometry_base[component_axis] -= 1
    boundary_point = (
        boundary.velocity_dirichlet_component_face_segment_pair_boundary_point_m[
            target_i, target_j, target_k, component_axis
        ]
    )
    nominal_probe = (
        boundary.velocity_dirichlet_component_face_segment_pair_nominal_probe_m[
            target_i, target_j, target_k, component_axis
        ]
    )
    normal = boundary.velocity_dirichlet_component_face_segment_pair_normal[
        target_i, target_j, target_k, component_axis
    ]
    sample_found, _sample_velocity, accepted_point = (
        boundary._canonical_component_face_actual_interior_sample(
            velocity_field,
            obstacle_field,
            geometry_base,
            boundary_point,
            nominal_probe,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            nx,
            ny,
            nz,
        )
    )
    allow_obstacle_fluid_interface = ti.cast(
        ti.abs(normal[component_axis]) > 1.0e-6,
        ti.i32,
    )
    storage_valid, storage, alpha, error_code = (
        boundary._select_canonical_component_face_storage_device(
            geometry_base,
            component_axis,
            boundary_point,
            accepted_point,
            obstacle_field,
            allow_obstacle_fluid_interface,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            nx,
            ny,
            nz,
        )
    )
    result_i32[0] = sample_found
    result_i32[1] = storage.x
    result_i32[2] = storage.y
    result_i32[3] = storage.z
    result_i32[4] = storage_valid
    result_i32[5] = error_code
    result_f64[0] = alpha
    for axis in ti.static(range(3)):
        result_f64[axis + 1] = accepted_point[axis]


class HibmComponentFaceGeometryTests(
    CanonicalComponentFaceLedgerContractMixin,
    unittest.TestCase,
):
    """Run the canonical component-face geometry and transaction contracts."""

    _GRID_NODES = (4, 4, 4)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=cls._GRID_NODES, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda", default_fp="f32"),
        )
        cls.fluid.set_velocity_dirichlet_boundary_authority("canonical")
        cls.component_face_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=cls._GRID_NODES,
            marker_capacity=2,
        )
        cls.component_face_search = HibmMpmIbNodeSearch(
            grid_nodes=cls._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=2,
        )
        cls.component_face_markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        cls.segment_component_face_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=cls._GRID_NODES,
            marker_capacity=4,
        )
        cls.segment_component_face_search = HibmMpmIbNodeSearch(
            grid_nodes=cls._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=4,
        )
        cls.segment_component_face_markers = HibmMpmSurfaceMarkers(
            marker_capacity=4
        )
        cls._r22_storage_probe_i32 = ti.field(dtype=ti.i32, shape=6)
        cls._r22_storage_probe_f64 = ti.field(dtype=ti.f64, shape=4)
        cls.marker_mac_constraint_operator = None

    @classmethod
    def _reset_shared_fixture(cls) -> None:
        cls.fluid.velocity.fill((0.0, 0.0, 0.0))
        cls.fluid.obstacle.fill(0)
        cls.fluid.clear_velocity_dirichlet_boundary_rows()

    @classmethod
    def _run_r22_component_face_storage_probe(
        cls,
        target: tuple[int, int, int],
        component_axis: int,
    ) -> tuple[float | int, ...]:
        result_i32 = cls._r22_storage_probe_i32
        result_f64 = cls._r22_storage_probe_f64
        result_i32.fill(-1)
        result_f64.fill(np.nan)
        _r22_component_face_storage_probe(
            cls.segment_component_face_boundary,
            cls.fluid.velocity,
            cls.fluid.obstacle,
            cls.fluid.cell_face_x_m,
            cls.fluid.cell_face_y_m,
            cls.fluid.cell_face_z_m,
            cls.fluid.cell_center_x_m,
            cls.fluid.cell_center_y_m,
            cls.fluid.cell_center_z_m,
            int(target[0]),
            int(target[1]),
            int(target[2]),
            int(component_axis),
            int(cls._GRID_NODES[0]),
            int(cls._GRID_NODES[1]),
            int(cls._GRID_NODES[2]),
            result_i32,
            result_f64,
        )
        return (
            int(result_i32[0]),
            int(result_i32[1]),
            int(result_i32[2]),
            int(result_i32[3]),
            int(result_i32[4]),
            float(result_f64[0]),
            int(result_i32[5]),
            float(result_f64[1]),
            float(result_f64[2]),
            float(result_f64[3]),
        )

    def test_relocation_source_linear_key_compiles_as_taichi_function(
        self,
    ) -> None:
        result = ti.field(dtype=ti.i64, shape=())
        _relocation_source_linear_key_probe(self.component_face_boundary, result)
        self.assertEqual(int(result[None]), 33)

    def test_near_tie_storage_selector_keeps_cached_and_live_route_consistent(
        self,
    ) -> None:
        """A one-ULP face-distance tie must not change route across kernels."""

        class StopAfterPrepare(RuntimeError):
            pass

        fluid = self.fluid
        boundary = self.component_face_boundary
        source = (0, 1, 1)
        component_axis = 1
        lower_face = source
        upper_face = (0, 2, 1)
        boundary_point = np.asarray(
            (
                0.000375000003259629,
                0.0015275443438440561,
                0.04698626324534416,
            ),
            dtype=np.float32,
        )
        sample_point = np.asarray(
            (
                0.000375000003259629,
                0.0015061686281114817,
                0.04559388384222984,
            ),
            dtype=np.float32,
        )
        normal = (
            0.0,
            -0.015350125095497177,
            -0.9998821798889871,
        )
        axis_faces = {
            "x": np.asarray(
                (0.0, 0.00075, 0.00150, 0.00225, 0.00300),
                dtype=np.float32,
            ),
            "y": np.asarray(
                (
                    0.00140625,
                    0.0014843749813735485,
                    0.0015625000232830644,
                    0.001640625,
                    0.00171875,
                ),
                dtype=np.float32,
            ),
            "z": np.asarray(
                (0.045, 0.0465625, 0.046875, 0.0471875, 0.0475),
                dtype=np.float32,
            ),
        }
        original_axis_fields = {
            field_name: getattr(fluid, field_name).to_numpy().copy()
            for field_name in (
                "cell_face_x_m",
                "cell_center_x_m",
                "cell_face_y_m",
                "cell_center_y_m",
                "cell_face_z_m",
                "cell_center_z_m",
            )
        }

        def selector_metrics(
            staged_boundary: np.ndarray,
            staged_sample: np.ndarray,
        ) -> tuple[tuple[float, float], tuple[float, float]]:
            source_center = np.asarray(
                (
                    fluid.cell_center_x_m.to_numpy()[source[0]],
                    fluid.cell_center_y_m.to_numpy()[source[1]],
                    fluid.cell_center_z_m.to_numpy()[source[2]],
                ),
                dtype=np.float32,
            )
            segment = np.asarray(
                staged_sample - staged_boundary,
                dtype=np.float32,
            )
            distance_squared = np.float32(np.dot(segment, segment))
            progress = []
            tangential_squared = []
            for face_y in axis_faces["y"][1:3]:
                face_center = source_center.copy()
                face_center[component_axis] = face_y
                boundary_offset = np.asarray(
                    face_center - staged_boundary,
                    dtype=np.float32,
                )
                face_progress = np.float32(
                    np.dot(boundary_offset, segment) / distance_squared
                )
                tangential_offset = np.asarray(
                    boundary_offset - face_progress * segment,
                    dtype=np.float32,
                )
                progress.append(float(face_progress))
                tangential_squared.append(
                    float(
                        np.float32(
                            np.dot(tangential_offset, tangential_offset)
                        )
                    )
                )
            return tuple(progress), tuple(tangential_squared)

        def run_route_case(
            perturbed_boundary_z_m: float | None,
        ) -> tuple[int, int, int]:
            staged_boundary = boundary_point.copy()
            staged_sample = sample_point.copy()
            if perturbed_boundary_z_m is not None:
                staged_boundary[2] = np.float32(perturbed_boundary_z_m)
            self._load_component_face_claims(
                (
                    _ComponentFaceClaim(
                        source_row=source,
                        boundary_point_m=tuple(
                            float(value) for value in staged_boundary
                        ),
                        interior_point_m=tuple(
                            float(value) for value in staged_sample
                        ),
                        normal=normal,
                        target_velocity_mps=(0.0, 0.25, 0.0),
                        region_id=202,
                    ),
                )
            )
            fluid.velocity.fill((0.0, 0.0, 0.0))
            observed: dict[str, int] = {}

            def capture_stages(stage: str) -> None:
                if stage == "hibm_velocity_row_segment_pair_precompute_after":
                    observed["cached_offset"] = int(
                        boundary.velocity_dirichlet_component_face_direct_selected_storage_offset[
                            source
                        ][component_axis]
                    )
                elif stage == "hibm_velocity_row_claim_prepare_after":
                    observed["lower_claim_count"] = int(
                        boundary.velocity_dirichlet_component_face_claim_count[
                            lower_face
                        ][component_axis]
                    )
                    observed["upper_claim_count"] = int(
                        boundary.velocity_dirichlet_component_face_claim_count[
                            upper_face
                        ][component_axis]
                    )
                    raise StopAfterPrepare

            with self.assertRaises(StopAfterPrepare):
                self._assemble_component_face_ledger(
                    interpolate_interior_velocity=True,
                    surface_projection_inactive_axis=0,
                    stage_observer=capture_stages,
                )
            return (
                observed["cached_offset"],
                observed["lower_claim_count"],
                observed["upper_claim_count"],
            )

        try:
            for axis, faces in axis_faces.items():
                getattr(fluid, f"cell_face_{axis}_m").from_numpy(faces)
                getattr(fluid, f"cell_center_{axis}_m").from_numpy(
                    (0.5 * (faces[:-1] + faces[1:])).astype(np.float32)
                )

            tie_progress, tie_distance_squared = selector_metrics(
                boundary_point,
                sample_point,
            )
            tie_band = (
                4.0
                * float(np.finfo(np.float32).eps)
                * max(max(tie_distance_squared), 1.0e-24)
            )
            self.assertLessEqual(
                abs(tie_distance_squared[0] - tie_distance_squared[1]),
                tie_band,
            )
            self.assertLess(tie_progress[1], tie_progress[0])

            perturbed_boundary_z_m = 0.04698626697063446
            perturbed_boundary = boundary_point.copy()
            perturbed_boundary[2] = np.float32(perturbed_boundary_z_m)
            _, perturbed_distance_squared = selector_metrics(
                perturbed_boundary,
                sample_point,
            )
            perturbed_band = (
                4.0
                * float(np.finfo(np.float32).eps)
                * max(max(perturbed_distance_squared), 1.0e-24)
            )
            self.assertLess(
                perturbed_distance_squared[0] + perturbed_band,
                perturbed_distance_squared[1],
            )

            tie_forward = run_route_case(None)
            non_tie = run_route_case(perturbed_boundary_z_m)
            tie_repeat = run_route_case(None)
            self.assertEqual(tie_forward, (1, 0, 1))
            self.assertEqual(tie_repeat, tie_forward)
            self.assertEqual(non_tie, (0, 1, 0))
        finally:
            for field_name, values in original_axis_fields.items():
                getattr(fluid, field_name).from_numpy(values)

    def test_marker_target_closure_candidate_uses_packed_device_scatter(
        self,
    ) -> None:
        boundary = self.component_face_boundary
        boundary.velocity_dirichlet_marker_target_closure_value_mps.fill(
            (0.0, 0.0, 0.0)
        )
        boundary.velocity_dirichlet_component_face_claim_target_mps.fill(
            (0.0, 0.0, 0.0)
        )

        materialized_residual = boundary._commit_marker_target_closure_candidate(
            keys=((1, 1, 1, 1),),
            correction_mps=np.asarray((0.25,), dtype=np.float64),
            matrix=np.asarray(((1.0,),), dtype=np.float64),
            residual_mps=np.asarray((0.25,), dtype=np.float64),
            closure_tolerance_mps=1.0e-6,
        )

        self.assertEqual(materialized_residual, 0.0)
        self.assertAlmostEqual(
            float(
                boundary.velocity_dirichlet_marker_target_closure_value_mps[
                    1, 1, 1
                ][1]
            ),
            0.25,
        )
        self.assertAlmostEqual(
            float(
                boundary.velocity_dirichlet_component_face_claim_target_mps[
                    1, 1, 1
                ][1]
            ),
            0.25,
        )

if __name__ == "__main__":
    unittest.main()
