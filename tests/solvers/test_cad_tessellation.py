from __future__ import annotations

import unittest

import numpy as np
import trimesh

from simulation_core.cad_tessellation import (
    StepCurveEntity,
    StepPartEntity,
    StepSurfaceEntity,
    StepTessellationResult,
    build_step_derived_source_config,
    remap_step_named_selection_face_ids,
)


class CadTessellationTests(unittest.TestCase):
    def test_remap_step_named_selection_face_ids_uses_surface_part_and_curve_tags(
        self,
    ) -> None:
        result = StepTessellationResult(
            mesh=trimesh.Trimesh(
                vertices=np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                ),
                faces=np.asarray(
                    [
                        [0, 1, 2],
                        [0, 1, 3],
                        [0, 2, 3],
                        [1, 2, 3],
                    ],
                    dtype=np.int64,
                ),
                process=False,
            ),
            surface_entities=[
                StepSurfaceEntity(entity_tag=1, face_ids=np.asarray([0, 1]), centroid_m=np.zeros(3)),
                StepSurfaceEntity(entity_tag=2, face_ids=np.asarray([2]), centroid_m=np.zeros(3)),
            ],
            part_entities=[
                StepPartEntity(
                    entity_dim=3,
                    entity_tag=1,
                    name="part",
                    surface_tags=(1, 2),
                    face_ids=np.asarray([0, 1, 2]),
                    centroid_m=np.zeros(3),
                )
            ],
            curve_entities=[
                StepCurveEntity(
                    entity_tag=14,
                    edge_vertex_pairs=np.asarray([[0, 1]]),
                    face_ids=np.asarray([0, 3]),
                    centroid_m=np.zeros(3),
                )
            ],
        )
        config = {
            "named_selections": [
                {
                    "id": 8,
                    "face_ids": [999],
                    "selection_source": {"kind": "step_surface", "cad_tags": [1, 2]},
                },
                {
                    "id": 1,
                    "face_ids": [999],
                    "selection_source": {"kind": "step_part", "cad_tags": [1, 2]},
                },
                {
                    "id": 14,
                    "face_ids": [999],
                    "selection_source": {"kind": "step_curve_loop", "cad_tags": [14]},
                },
            ]
        }

        remapped = remap_step_named_selection_face_ids(config, result)

        by_id = {selection["id"]: selection for selection in remapped["named_selections"]}
        self.assertEqual(by_id[8]["face_ids"], [0, 1, 2])
        self.assertEqual(by_id[1]["face_ids"], [0, 1, 2])
        self.assertEqual(by_id[14]["face_ids"], [0, 3])
        self.assertEqual(by_id[8]["source_mesh_face_count"], 3)

    def test_build_step_derived_source_config_records_hash_provenance(self) -> None:
        base_config = {
            "mesh_path": "old.stl",
            "surface_mesh_cache_path": "old.stl",
            "metadata": {"case": "kept"},
        }

        config = build_step_derived_source_config(
            base_config,
            step_path="sim.STEP",
            step_sha256="abc123",
            surface_mesh_cache_path="sim.surface_mesh.stl",
            surface_mesh_cache_sha256="def456",
            tessellation_settings={"relative_edge_length": 0.008},
            mesh_scale_to_m=0.001,
            tessellation_report={
                "surface_count": 2,
                "curve_count": 1,
                "part_count": 1,
                "mesh_face_count": 4,
            },
        )

        self.assertEqual(config["mesh_path"], "sim.STEP")
        self.assertEqual(config["surface_mesh_cache_path"], "sim.surface_mesh.stl")
        self.assertEqual(config["mesh_scale_to_m"], 0.001)
        self.assertEqual(config["metadata"]["case"], "kept")
        self.assertEqual(config["mesh_import"]["source_step_sha256"], "abc123")
        self.assertEqual(config["mesh_import"]["surface_mesh_cache_sha256"], "def456")


if __name__ == "__main__":
    unittest.main()
