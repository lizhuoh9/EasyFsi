from __future__ import annotations

import hashlib
import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from simulation_core.fluids.preflow_snapshot import (
    PREFLOW_SNAPSHOT_FIELD_NAMES,
    PREFLOW_SNAPSHOT_SCHEMA_VERSION,
    PreflowSnapshot,
    PreflowSnapshotIdentity,
    PreflowSnapshotIntegrityError,
    PreflowSnapshotMismatchError,
    PreflowSnapshotValidationError,
    canonical_config_sha256,
    canonical_geometry_sha256,
    canonical_source_sha256,
    inspect_preflow_snapshot,
    load_preflow_snapshot,
    save_preflow_snapshot,
)


_DIRECT_HARD_FIXED_COMPONENT_MASK = (
    "velocity_dirichlet_boundary_hard_fixed_component_mask"
)
_EXTERNAL_EXACT_COMPONENT_MASK = (
    "velocity_dirichlet_boundary_external_exact_component_mask"
)
_DYNAMIC_OWNED_ROW = "velocity_dirichlet_boundary_owned_row"
_ENFORCEMENT_WEIGHT = "velocity_dirichlet_boundary_enforcement_weight"
_TRANSIENT_PROJECTION_MASK_NAMES = (
    "velocity_dirichlet_projection_hard_fixed_component_mask",
    "velocity_dirichlet_projection_fixed_component_mask",
)
_DIRECTED_EXTERNAL_BOUNDARY_MASK_NAMES = (
    "external_velocity_boundary_x_face_active_component_mask",
    "external_velocity_boundary_y_face_active_component_mask",
    "external_velocity_boundary_z_face_active_component_mask",
)
_DIRECTED_EXTERNAL_BOUNDARY_VALUE_NAMES = (
    "external_velocity_boundary_x_face_value_mps",
    "external_velocity_boundary_y_face_value_mps",
    "external_velocity_boundary_z_face_value_mps",
)
_DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES = (
    *_DIRECTED_EXTERNAL_BOUNDARY_MASK_NAMES,
    *_DIRECTED_EXTERNAL_BOUNDARY_VALUE_NAMES,
)
_CANONICAL_LEDGER_FIELD_NAMES = (
    "velocity_dirichlet_boundary_active_component_mask",
    "velocity_dirichlet_boundary_pressure_mobility",
    "velocity_dirichlet_boundary_component_enforcement_weight",
    "velocity_dirichlet_boundary_component_region_id",
    "velocity_dirichlet_boundary_owned_component_mask",
)
_SST_FIELD_NAMES = (
    "sst_turbulent_kinetic_energy",
    "sst_specific_dissipation_rate",
    "sst_eddy_viscosity_pa_s",
    "sst_wall_distance_m",
)


@dataclass(frozen=True)
class _CanonicalConfig:
    grid: tuple[int, int, int]
    solver: dict[str, object]


def _valid_fields(grid_shape: tuple[int, int, int] = (2, 3, 4)) -> dict[str, np.ndarray]:
    nx, ny, nz = grid_shape
    vector_shape = grid_shape + (3,)
    scalar_count = int(np.prod(grid_shape))
    vector_count = int(np.prod(vector_shape))
    direct_hard_mask = (
        np.arange(scalar_count, dtype=np.int32).reshape(grid_shape) % 8
    )
    fields = {
        "velocity": np.arange(vector_count, dtype=np.float32).reshape(vector_shape),
        "velocity_prev": np.full(vector_shape, 0.25, dtype=np.float32),
        "pressure": np.arange(scalar_count, dtype=np.float64).reshape(grid_shape),
        "fsi_pressure": np.full(grid_shape, 2.5, dtype=np.float64),
        "obstacle": np.zeros(grid_shape, dtype=np.int32),
        "hibm_base_obstacle": np.zeros(grid_shape, dtype=np.int32),
        "hibm_dynamic_solid_volume_obstacle": np.zeros(grid_shape, dtype=np.int32),
        "hibm_dynamic_solid_volume_external_carve": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "sst_turbulent_kinetic_energy": np.full(
            grid_shape, 0.375, dtype=np.float32
        ),
        "sst_specific_dissipation_rate": np.full(
            grid_shape, 125.0, dtype=np.float32
        ),
        "sst_eddy_viscosity_pa_s": np.full(
            grid_shape, 1.8e-4, dtype=np.float32
        ),
        "sst_wall_distance_m": np.full(
            grid_shape, 2.5e-3, dtype=np.float32
        ),
        # Keep the baseline fixture cross-field coherent: rows carrying hard
        # masks or projection weights are active and remain outside obstacles.
        "velocity_dirichlet_boundary_active": np.ones(grid_shape, dtype=np.int32),
        "velocity_dirichlet_boundary_value_mps": np.full(
            vector_shape, 1.25, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_projection_weight": np.ones(
            grid_shape, dtype=np.float32
        ),
        # These are ordinary direct rows, so the legacy enforcement and
        # pressure-projection weights intentionally agree in this fixture.
        _ENFORCEMENT_WEIGHT: np.ones(grid_shape, dtype=np.float32),
        "velocity_dirichlet_boundary_marker_region_id": np.full(
            grid_shape, -1, dtype=np.int32
        ),
        _DIRECT_HARD_FIXED_COMPONENT_MASK: direct_hard_mask,
        _EXTERNAL_EXACT_COMPONENT_MASK: direct_hard_mask & np.int32(0b100),
        _DYNAMIC_OWNED_ROW: np.zeros(grid_shape, dtype=np.int32),
    }
    plane_shapes = {
        "x": (2, ny, nz),
        "y": (2, nx, nz),
        "z": (2, nx, ny),
    }
    for axis_index, axis_name in enumerate(("x", "y", "z"), start=1):
        plane_shape = plane_shapes[axis_name]
        plane_count = int(np.prod(plane_shape))
        fields[
            f"external_velocity_boundary_{axis_name}_face_active_component_mask"
        ] = (
            np.arange(plane_count, dtype=np.int32).reshape(plane_shape)
            + np.int32(axis_index)
        ) % np.int32(8)
        value_shape = plane_shape + (3,)
        fields[f"external_velocity_boundary_{axis_name}_face_value_mps"] = (
            np.arange(int(np.prod(value_shape)), dtype=np.float32).reshape(
                value_shape
            )
            + np.float32(100.0 * axis_index)
        )
    return fields


def _identity(*, tag: str = "current") -> PreflowSnapshotIdentity:
    return PreflowSnapshotIdentity.from_inputs(
        config={"case": "vertical_flap", "grid": [2, 3, 4], "tag": tag},
        sources={
            "simulation_core/fluids/solver.py": "class CartesianFluidSolver:\n    pass\n",
            "cases/example.py": b"CONFIG = {}\n",
        },
        geometry={
            "cell_center_x_m": np.array([0.0, 0.5], dtype=np.float64),
            "surface_vertices_m": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
            ),
        },
    )


def _valid_canonical_fields(
    grid_shape: tuple[int, int, int] = (2, 3, 4),
) -> dict[str, np.ndarray]:
    fields = _valid_fields(grid_shape)
    vector_shape = grid_shape + (3,)
    active_component_mask = np.array(
        fields[_DIRECT_HARD_FIXED_COMPONENT_MASK],
        copy=True,
    )
    component_value = np.array(
        fields["velocity_dirichlet_boundary_value_mps"],
        copy=True,
    )
    component_pressure_mobility = np.ones(vector_shape, dtype=np.float32)
    component_enforcement_weight = np.zeros(vector_shape, dtype=np.float32)
    for axis in range(3):
        axis_active = (active_component_mask & (1 << axis)) != 0
        component_value[..., axis] = np.where(
            axis_active,
            component_value[..., axis],
            0.0,
        )
        component_pressure_mobility[..., axis] = np.where(
            axis_active,
            0.0,
            1.0,
        )
        component_enforcement_weight[..., axis] = np.where(
            axis_active,
            1.0,
            0.0,
        )
    fields.update(
        {
            "velocity_dirichlet_boundary_value_mps": component_value,
            "velocity_dirichlet_boundary_active_component_mask": (
                active_component_mask
            ),
            "velocity_dirichlet_boundary_pressure_mobility": (
                component_pressure_mobility
            ),
            "velocity_dirichlet_boundary_component_enforcement_weight": (
                component_enforcement_weight
            ),
            "velocity_dirichlet_boundary_component_region_id": np.full(
                vector_shape,
                -1,
                dtype=np.int32,
            ),
            "velocity_dirichlet_boundary_owned_component_mask": np.zeros(
                grid_shape,
                dtype=np.int32,
            ),
        }
    )
    return fields


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_manifest(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["manifest_sha256"] = canonical_config_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _rewrite_npz(path: Path, mutate) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    return arrays


def _downgrade_saved_snapshot(files, *, schema_version: int) -> None:
    if schema_version not in {5, 6, 7}:
        raise ValueError(f"unsupported test downgrade schema {schema_version}")
    dropped_fields = set(_SST_FIELD_NAMES)
    if schema_version <= 6:
        dropped_fields.update(_DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES)
    if schema_version == 5:
        dropped_fields.update(_CANONICAL_LEDGER_FIELD_NAMES)

    def drop_fields(arrays: dict[str, np.ndarray]) -> None:
        for field_name in dropped_fields:
            arrays.pop(field_name)

    _rewrite_npz(files.npz_path, drop_fields)

    def downgrade_manifest(manifest: dict[str, object]) -> None:
        manifest["schema_version"] = schema_version
        for field_name in dropped_fields:
            manifest["fields"].pop(field_name)
        if schema_version == 5:
            manifest.pop("velocity_dirichlet_boundary_authority")
            manifest.pop("velocity_dirichlet_component_ledger_generation")
        manifest["npz_sha256"] = _sha256_file(files.npz_path)

    _rewrite_manifest(files.metadata_path, downgrade_manifest)


class PreflowSnapshotTests(unittest.TestCase):
    def test_schema_v8_persists_sst_and_boundary_provenance(
        self,
    ) -> None:
        with self.subTest(contract="schema version"):
            self.assertEqual(PREFLOW_SNAPSHOT_SCHEMA_VERSION, 8)
        for field_name in _SST_FIELD_NAMES:
            with self.subTest(contract="SST state persisted", name=field_name):
                self.assertIn(field_name, PREFLOW_SNAPSHOT_FIELD_NAMES)
        with self.subTest(contract="enforcement weight persisted"):
            self.assertIn(_ENFORCEMENT_WEIGHT, PREFLOW_SNAPSHOT_FIELD_NAMES)
        with self.subTest(contract="direct mask persisted"):
            self.assertIn(
                _DIRECT_HARD_FIXED_COMPONENT_MASK,
                PREFLOW_SNAPSHOT_FIELD_NAMES,
            )
        with self.subTest(contract="external exact provenance persisted"):
            self.assertIn(_EXTERNAL_EXACT_COMPONENT_MASK, PREFLOW_SNAPSHOT_FIELD_NAMES)
        with self.subTest(contract="dynamic row provenance persisted"):
            self.assertIn(_DYNAMIC_OWNED_ROW, PREFLOW_SNAPSHOT_FIELD_NAMES)
        for field_name in _DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES:
            with self.subTest(
                contract="directed external boundary plane persisted",
                name=field_name,
            ):
                self.assertIn(field_name, PREFLOW_SNAPSHOT_FIELD_NAMES)
        for transient_name in _TRANSIENT_PROJECTION_MASK_NAMES:
            with self.subTest(contract="transient mask excluded", name=transient_name):
                self.assertNotIn(transient_name, PREFLOW_SNAPSHOT_FIELD_NAMES)

    def test_canonical_hashes_ignore_mapping_order_and_array_layout(self) -> None:
        config_a = {"grid": (2, 3, 4), "solver": {"tol": 1.0e-8, "kind": "cg"}}
        config_b = {"solver": {"kind": "cg", "tol": 1.0e-8}, "grid": [2, 3, 4]}
        self.assertEqual(
            canonical_config_sha256(config_a),
            canonical_config_sha256(config_b),
        )
        self.assertEqual(
            canonical_config_sha256(config_a),
            canonical_config_sha256(
                _CanonicalConfig(
                    grid=(2, 3, 4),
                    solver={"tol": 1.0e-8, "kind": "cg"},
                )
            ),
        )
        self.assertNotEqual(
            canonical_config_sha256(config_a),
            canonical_config_sha256({**config_b, "grid": [2, 3, 5]}),
        )

        sources_a = {"b.py": b"b = 2\n", "a.py": "a = 1\n"}
        sources_b = {"a.py": "a = 1\n", "b.py": b"b = 2\n"}
        self.assertEqual(
            canonical_source_sha256(sources_a),
            canonical_source_sha256(sources_b),
        )
        self.assertNotEqual(
            canonical_source_sha256(sources_a),
            canonical_source_sha256({**sources_b, "a.py": "a = 3\n"}),
        )

        geometry_c = np.arange(12, dtype=np.float64).reshape(3, 4)
        geometry_f = np.asfortranarray(geometry_c)
        geometry_big_endian = geometry_c.astype(">f8")
        self.assertEqual(
            canonical_geometry_sha256({"vertices": geometry_c}),
            canonical_geometry_sha256({"vertices": geometry_f}),
        )
        self.assertEqual(
            canonical_geometry_sha256({"vertices": geometry_c}),
            canonical_geometry_sha256({"vertices": geometry_big_endian}),
        )
        changed = geometry_c.copy()
        changed[0, 0] = 99.0
        self.assertNotEqual(
            canonical_geometry_sha256({"vertices": geometry_c}),
            canonical_geometry_sha256({"vertices": changed}),
        )

    def test_canonical_geometry_hash_accepts_empty_numeric_arrays(self) -> None:
        empty = np.empty((0, 3), dtype=np.int32)
        same_shape_and_dtype = np.empty((0, 3), dtype=np.int32, order="F")
        big_endian = np.empty((0, 3), dtype=">i4")

        digest = canonical_geometry_sha256(
            {"marker_projection_triangle_indices": empty}
        )
        self.assertEqual(
            digest,
            canonical_geometry_sha256(
                {"marker_projection_triangle_indices": same_shape_and_dtype}
            ),
        )
        self.assertEqual(
            digest,
            canonical_geometry_sha256(
                {"marker_projection_triangle_indices": big_endian}
            ),
        )
        self.assertNotEqual(
            digest,
            canonical_geometry_sha256(
                {
                    "marker_projection_triangle_indices": np.empty(
                        (3, 0), dtype=np.int32
                    )
                }
            ),
        )
        self.assertNotEqual(
            digest,
            canonical_geometry_sha256(
                {
                    "marker_projection_triangle_indices": np.empty(
                        (0, 4), dtype=np.int32
                    )
                }
            ),
        )
        self.assertNotEqual(
            digest,
            canonical_geometry_sha256(
                {
                    "marker_projection_triangle_indices": np.empty(
                        (0, 3), dtype=np.float32
                    )
                }
            ),
        )

    def test_round_trip_preserves_all_fields_identity_and_optional_history(self) -> None:
        fields = _valid_fields()
        snapshot = PreflowSnapshot(
            fields=fields,
            identity=_identity(),
            history={"iterations": 120, "residual_l2": [1.0, 0.1, 0.01]},
        )
        fields["velocity"][0, 0, 0, 0] = -999.0
        self.assertNotEqual(float(snapshot.fields["velocity"][0, 0, 0, 0]), -999.0)
        self.assertFalse(snapshot.fields["velocity"].flags.writeable)

        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "steady_preflow"
            with mock.patch(
                "simulation_core.fluids.preflow_snapshot.os.replace",
                wraps=os.replace,
            ) as replace:
                files = save_preflow_snapshot(prefix, snapshot)

            self.assertEqual(files.metadata_path, prefix.with_suffix(".json"))
            self.assertEqual(files.snapshot_path, prefix)
            self.assertEqual(files.npz_path.parent, prefix.parent)
            self.assertTrue(files.npz_path.name.startswith(prefix.name + "."))
            self.assertEqual(files.npz_path.suffix, ".npz")
            self.assertTrue(files.npz_path.is_file())
            self.assertTrue(files.metadata_path.is_file())
            self.assertEqual(replace.call_count, 2)
            self.assertEqual(Path(replace.call_args_list[-1].args[1]), files.metadata_path)
            self.assertEqual(list(prefix.parent.glob("*.tmp-*")), [])

            manifest = json.loads(files.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 8)
            for field_name in _SST_FIELD_NAMES:
                self.assertIn(field_name, manifest["fields"])
            self.assertIn(_ENFORCEMENT_WEIGHT, manifest["fields"])
            self.assertIn(_DIRECT_HARD_FIXED_COMPONENT_MASK, manifest["fields"])
            self.assertIn(_EXTERNAL_EXACT_COMPONENT_MASK, manifest["fields"])
            for field_name in _DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES:
                self.assertIn(field_name, manifest["fields"])
            for transient_name in _TRANSIENT_PROJECTION_MASK_NAMES:
                self.assertNotIn(transient_name, manifest["fields"])
            with np.load(files.npz_path, allow_pickle=False) as archive:
                for field_name in _SST_FIELD_NAMES:
                    self.assertIn(field_name, archive.files)
                self.assertIn(_ENFORCEMENT_WEIGHT, archive.files)
                self.assertIn(_DIRECT_HARD_FIXED_COMPONENT_MASK, archive.files)
                self.assertIn(_EXTERNAL_EXACT_COMPONENT_MASK, archive.files)
                for field_name in _DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES:
                    self.assertIn(field_name, archive.files)
                for transient_name in _TRANSIENT_PROJECTION_MASK_NAMES:
                    self.assertNotIn(transient_name, archive.files)

            loaded = load_preflow_snapshot(files, expected_identity=snapshot.identity)
            expected_artifact_identity = {
                "metadata_file_sha256": hashlib.sha256(
                    files.metadata_path.read_bytes()
                ).hexdigest(),
                "manifest_sha256": manifest["manifest_sha256"],
                "npz_file": files.npz_path.name,
                "npz_sha256": manifest["npz_sha256"],
            }
            self.assertEqual(
                dict(loaded.artifact_identity),
                expected_artifact_identity,
            )
            inspected = inspect_preflow_snapshot(prefix)
            self.assertEqual(
                inspected["artifact_identity"],
                expected_artifact_identity,
            )
            self.assertEqual(
                inspected["identity"],
                {
                    "config_sha256": snapshot.identity.config_sha256,
                    "source_sha256": snapshot.identity.source_sha256,
                    "geometry_sha256": snapshot.identity.geometry_sha256,
                },
            )
            with self.assertRaisesRegex(ValueError, "generation NPZ"):
                load_preflow_snapshot(
                    files.npz_path,
                    expected_identity=snapshot.identity,
                )

        self.assertEqual(loaded.identity, snapshot.identity)
        self.assertEqual(loaded.history, snapshot.history)
        self.assertEqual(tuple(loaded.fields), PREFLOW_SNAPSHOT_FIELD_NAMES)
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
            np.testing.assert_array_equal(loaded.fields[name], snapshot.fields[name])
            self.assertEqual(loaded.fields[name].dtype, snapshot.fields[name].dtype)
            self.assertFalse(loaded.fields[name].flags.writeable)

    def test_schema_v8_directed_external_boundary_plane_shapes_are_axis_exact(
        self,
    ) -> None:
        snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity())
        expected_shapes = {
            "external_velocity_boundary_x_face_active_component_mask": (2, 3, 4),
            "external_velocity_boundary_x_face_value_mps": (2, 3, 4, 3),
            "external_velocity_boundary_y_face_active_component_mask": (2, 2, 4),
            "external_velocity_boundary_y_face_value_mps": (2, 2, 4, 3),
            "external_velocity_boundary_z_face_active_component_mask": (2, 2, 3),
            "external_velocity_boundary_z_face_value_mps": (2, 2, 3, 3),
        }
        for field_name, expected_shape in expected_shapes.items():
            with self.subTest(field_name=field_name):
                self.assertEqual(snapshot.fields[field_name].shape, expected_shape)

    def test_schema_v6_canonical_snapshot_without_directed_planes_fails_closed(
        self,
    ) -> None:
        snapshot = PreflowSnapshot(
            fields=_valid_canonical_fields(),
            identity=_identity(),
            velocity_dirichlet_boundary_authority="canonical",
            velocity_dirichlet_component_ledger_generation=11,
        )
        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "schema-v6-canonical"
            files = save_preflow_snapshot(prefix, snapshot)

            def drop_directed_planes(arrays: dict[str, np.ndarray]) -> None:
                for field_name in _DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES:
                    arrays.pop(field_name)

            _rewrite_npz(files.npz_path, drop_directed_planes)

            def downgrade_manifest(manifest: dict[str, object]) -> None:
                manifest["schema_version"] = 6
                for field_name in _DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES:
                    manifest["fields"].pop(field_name)
                manifest["npz_sha256"] = _sha256_file(files.npz_path)

            _rewrite_manifest(files.metadata_path, downgrade_manifest)

            with self.assertRaisesRegex(
                PreflowSnapshotMismatchError,
                r"schema-6.*(?:canonical|directed external boundary)",
            ):
                load_preflow_snapshot(
                    prefix,
                    expected_identity=snapshot.identity,
                    expected_velocity_dirichlet_boundary_authority="canonical",
                )

    def test_schema_v8_directed_planes_remain_identity_guarded(self) -> None:
        snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity(tag="a"))
        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "identity"
            save_preflow_snapshot(prefix, snapshot)

            with self.assertRaisesRegex(
                PreflowSnapshotMismatchError,
                "config_sha256",
            ):
                load_preflow_snapshot(
                    prefix,
                    expected_identity=_identity(tag="b"),
                )

    def test_schema_v5_v6_v7_loads_upgrade_to_neutral_laminar_sst_state(
        self,
    ) -> None:
        neutral_values = {
            "sst_turbulent_kinetic_energy": np.float32(0.0),
            "sst_specific_dissipation_rate": np.float32(1.0),
            "sst_eddy_viscosity_pa_s": np.float32(0.0),
            "sst_wall_distance_m": np.float32(1.0e20),
        }
        for schema_version in (5, 6, 7):
            with self.subTest(schema_version=schema_version), TemporaryDirectory() as root:
                snapshot = PreflowSnapshot(
                    fields=_valid_fields(),
                    identity=_identity(tag=f"schema-{schema_version}"),
                )
                files = save_preflow_snapshot(
                    Path(root) / f"schema-v{schema_version}",
                    snapshot,
                )
                _downgrade_saved_snapshot(files, schema_version=schema_version)

                loaded = load_preflow_snapshot(
                    files,
                    expected_identity=snapshot.identity,
                )

                np.testing.assert_array_equal(
                    loaded.fields["velocity"], snapshot.fields["velocity"]
                )
                self.assertEqual(tuple(loaded.fields), PREFLOW_SNAPSHOT_FIELD_NAMES)
                for field_name, neutral_value in neutral_values.items():
                    self.assertEqual(loaded.fields[field_name].dtype, np.float32)
                    np.testing.assert_array_equal(
                        loaded.fields[field_name],
                        np.full((2, 3, 4), neutral_value, dtype=np.float32),
                    )

    def test_schema_v7_canonical_load_upgrades_only_missing_sst_state(self) -> None:
        with TemporaryDirectory() as root:
            snapshot = PreflowSnapshot(
                fields=_valid_canonical_fields(),
                identity=_identity(tag="schema-7-canonical"),
                velocity_dirichlet_boundary_authority="canonical",
                velocity_dirichlet_component_ledger_generation=11,
            )
            files = save_preflow_snapshot(Path(root) / "schema-v7", snapshot)
            _downgrade_saved_snapshot(files, schema_version=7)

            loaded = load_preflow_snapshot(
                files,
                expected_identity=snapshot.identity,
                expected_velocity_dirichlet_boundary_authority="canonical",
            )

            self.assertEqual(
                loaded.velocity_dirichlet_component_ledger_generation,
                11,
            )
            self.assertEqual(float(loaded.fields["sst_turbulent_kinetic_energy"].max()), 0.0)
            self.assertEqual(float(loaded.fields["sst_eddy_viscosity_pa_s"].max()), 0.0)
            self.assertEqual(float(loaded.fields["sst_specific_dissipation_rate"].min()), 1.0)
            self.assertEqual(
                loaded.fields["sst_wall_distance_m"].min(),
                np.float32(1.0e20),
            )

    def test_load_rejects_directed_plane_shape_and_nonfinite_tampering(
        self,
    ) -> None:
        snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity())
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            shape_files = save_preflow_snapshot(root / "plane-shape", snapshot)
            _rewrite_manifest(
                shape_files.metadata_path,
                lambda manifest: manifest["fields"][
                    "external_velocity_boundary_x_face_active_component_mask"
                ].update({"shape": [2, 3, 5]}),
            )
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "shape"):
                load_preflow_snapshot(
                    root / "plane-shape",
                    expected_identity=snapshot.identity,
                )

            finite_files = save_preflow_snapshot(root / "plane-finite", snapshot)
            _rewrite_npz(
                finite_files.npz_path,
                lambda arrays: arrays[
                    "external_velocity_boundary_z_face_value_mps"
                ].__setitem__((1, 0, 0, 2), np.nan),
            )
            _rewrite_manifest(
                finite_files.metadata_path,
                lambda manifest: manifest.update(
                    {"npz_sha256": _sha256_file(finite_files.npz_path)}
                ),
            )
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "finite"):
                load_preflow_snapshot(
                    root / "plane-finite",
                    expected_identity=snapshot.identity,
                )

    def test_load_rejects_previous_schema_after_provenance_contract_bump(self) -> None:
        snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity())
        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "schema-v4"
            files = save_preflow_snapshot(prefix, snapshot)
            _rewrite_manifest(
                files.metadata_path,
                lambda manifest: manifest.update({"schema_version": 4}),
            )

            with self.assertRaisesRegex(
                PreflowSnapshotIntegrityError,
                "unsupported snapshot schema version",
            ):
                load_preflow_snapshot(prefix, expected_identity=snapshot.identity)

    def test_schema_rejects_missing_extra_wrong_shape_dtype_and_nonfinite_fields(self) -> None:
        invalid_cases: list[tuple[str, dict[str, np.ndarray], str]] = []

        missing = _valid_fields()
        missing.pop("fsi_pressure")
        invalid_cases.append(("missing", missing, "missing"))

        extra = _valid_fields()
        extra["not_a_solver_field"] = np.zeros((2, 3, 4), dtype=np.float32)
        invalid_cases.append(("extra", extra, "unexpected"))

        wrong_shape = _valid_fields()
        wrong_shape["pressure"] = np.zeros((2, 3, 5), dtype=np.float64)
        invalid_cases.append(("shape", wrong_shape, "shape"))

        wrong_dtype = _valid_fields()
        wrong_dtype["velocity"] = wrong_dtype["velocity"].astype(np.float64)
        invalid_cases.append(("dtype", wrong_dtype, "dtype"))

        hard_fixed_wrong_shape = _valid_fields()
        hard_fixed_wrong_shape[_DIRECT_HARD_FIXED_COMPONENT_MASK] = np.zeros(
            (2, 3, 5), dtype=np.int32
        )
        invalid_cases.append(
            ("hard fixed mask shape", hard_fixed_wrong_shape, "shape")
        )

        hard_fixed_wrong_dtype = _valid_fields()
        hard_fixed_wrong_dtype[_DIRECT_HARD_FIXED_COMPONENT_MASK] = (
            hard_fixed_wrong_dtype[_DIRECT_HARD_FIXED_COMPONENT_MASK].astype(
                np.float32
            )
        )
        invalid_cases.append(
            ("hard fixed mask dtype", hard_fixed_wrong_dtype, "dtype")
        )

        external_exact_wrong_dtype = _valid_fields()
        external_exact_wrong_dtype[_EXTERNAL_EXACT_COMPONENT_MASK] = (
            external_exact_wrong_dtype[_EXTERNAL_EXACT_COMPONENT_MASK].astype(
                np.float32
            )
        )
        invalid_cases.append(
            ("external exact mask dtype", external_exact_wrong_dtype, "dtype")
        )

        directed_x_mask_wrong_shape = _valid_fields()
        directed_x_mask_wrong_shape[
            "external_velocity_boundary_x_face_active_component_mask"
        ] = np.zeros((2, 3, 5), dtype=np.int32)
        invalid_cases.append(
            (
                "directed x face mask shape",
                directed_x_mask_wrong_shape,
                "shape",
            )
        )

        directed_y_value_wrong_shape = _valid_fields()
        directed_y_value_wrong_shape[
            "external_velocity_boundary_y_face_value_mps"
        ] = np.zeros((2, 2, 5, 3), dtype=np.float32)
        invalid_cases.append(
            (
                "directed y face value shape",
                directed_y_value_wrong_shape,
                "shape",
            )
        )

        directed_z_mask_wrong_dtype = _valid_fields()
        directed_z_mask_wrong_dtype[
            "external_velocity_boundary_z_face_active_component_mask"
        ] = directed_z_mask_wrong_dtype[
            "external_velocity_boundary_z_face_active_component_mask"
        ].astype(np.float32)
        invalid_cases.append(
            (
                "directed z face mask dtype",
                directed_z_mask_wrong_dtype,
                "dtype",
            )
        )

        directed_mask_out_of_range = _valid_fields()
        directed_mask_out_of_range[
            "external_velocity_boundary_x_face_active_component_mask"
        ][0, 0, 0] = 8
        invalid_cases.append(
            (
                "directed face mask range",
                directed_mask_out_of_range,
                "three-bit",
            )
        )

        directed_value_nonfinite = _valid_fields()
        directed_value_nonfinite[
            "external_velocity_boundary_z_face_value_mps"
        ][1, 0, 0, 2] = np.nan
        invalid_cases.append(
            (
                "directed face value finite",
                directed_value_nonfinite,
                "finite",
            )
        )

        enforcement_wrong_dtype = _valid_fields()
        enforcement_wrong_dtype[_ENFORCEMENT_WEIGHT] = enforcement_wrong_dtype[
            _ENFORCEMENT_WEIGHT
        ].astype(np.float64)
        invalid_cases.append(
            ("enforcement weight dtype", enforcement_wrong_dtype, "dtype")
        )

        enforcement_out_of_range = _valid_fields()
        enforcement_out_of_range[_ENFORCEMENT_WEIGHT][0, 0, 0] = 1.01
        invalid_cases.append(
            (
                "enforcement weight range",
                enforcement_out_of_range,
                "enforcement_weight",
            )
        )

        nonfinite = _valid_fields()
        nonfinite["fsi_pressure"][0, 0, 0] = np.nan
        invalid_cases.append(("finite", nonfinite, "finite"))

        negative_k = _valid_fields()
        negative_k["sst_turbulent_kinetic_energy"][0, 0, 0] = -1.0
        invalid_cases.append(("negative SST k", negative_k, "non-negative"))

        zero_omega = _valid_fields()
        zero_omega["sst_specific_dissipation_rate"][0, 0, 0] = 0.0
        invalid_cases.append(("zero SST omega", zero_omega, "positive"))

        negative_eddy_viscosity = _valid_fields()
        negative_eddy_viscosity["sst_eddy_viscosity_pa_s"][0, 0, 0] = -1.0e-6
        invalid_cases.append(
            (
                "negative SST eddy viscosity",
                negative_eddy_viscosity,
                "non-negative",
            )
        )

        zero_wall_distance = _valid_fields()
        zero_wall_distance["sst_wall_distance_m"][0, 0, 0] = 0.0
        invalid_cases.append(
            ("zero SST wall distance", zero_wall_distance, "positive")
        )

        for label, fields, message in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(PreflowSnapshotValidationError, message):
                    PreflowSnapshot(fields=fields, identity=_identity())

    def test_external_exact_mask_requires_three_bits_direct_subset_and_active_row(
        self,
    ) -> None:
        invalid_cases: tuple[tuple[str, dict[str, np.ndarray], str], ...] = ()

        outside_three_bits = _valid_fields()
        outside_three_bits[_EXTERNAL_EXACT_COMPONENT_MASK][0, 0, 0] = 8
        invalid_cases += (("outside three bits", outside_three_bits, "external_exact"),)

        not_direct_subset = _valid_fields()
        not_direct_subset[_DIRECT_HARD_FIXED_COMPONENT_MASK][0, 0, 0] = 0b001
        not_direct_subset[_EXTERNAL_EXACT_COMPONENT_MASK][0, 0, 0] = 0b100
        invalid_cases += (("not direct subset", not_direct_subset, "subset"),)

        inactive = _valid_fields()
        inactive["velocity_dirichlet_boundary_active"][0, 0, 0] = 0
        inactive[_DIRECT_HARD_FIXED_COMPONENT_MASK][0, 0, 0] = 0b100
        inactive[_EXTERNAL_EXACT_COMPONENT_MASK][0, 0, 0] = 0b100
        inactive["velocity_dirichlet_boundary_projection_weight"][0, 0, 0] = 0.0
        inactive[_ENFORCEMENT_WEIGHT][0, 0, 0] = 0.0
        invalid_cases += (("inactive provenance", inactive, "inactive"),)

        dynamic_owned = _valid_fields()
        dynamic_owned[_DIRECT_HARD_FIXED_COMPONENT_MASK][0, 0, 0] = 0b100
        dynamic_owned[_EXTERNAL_EXACT_COMPONENT_MASK][0, 0, 0] = 0b100
        dynamic_owned[_DYNAMIC_OWNED_ROW][0, 0, 0] = 1
        invalid_cases += (("dynamic ownership overlap", dynamic_owned, "overlap"),)

        inactive_enforcement = _valid_fields()
        inactive_enforcement["velocity_dirichlet_boundary_active"][0, 0, 0] = 0
        inactive_enforcement[_DIRECT_HARD_FIXED_COMPONENT_MASK][0, 0, 0] = 0
        inactive_enforcement[_EXTERNAL_EXACT_COMPONENT_MASK][0, 0, 0] = 0
        inactive_enforcement[
            "velocity_dirichlet_boundary_projection_weight"
        ][0, 0, 0] = 0.0
        invalid_cases += (
            ("inactive enforcement", inactive_enforcement, "enforcement"),
        )

        inactive_hard = _valid_fields()
        inactive_hard["velocity_dirichlet_boundary_active"][0, 0, 0] = 0
        inactive_hard[_DIRECT_HARD_FIXED_COMPONENT_MASK][0, 0, 0] = 0b100
        inactive_hard[_EXTERNAL_EXACT_COMPONENT_MASK][0, 0, 0] = 0
        inactive_hard[
            "velocity_dirichlet_boundary_projection_weight"
        ][0, 0, 0] = 0.0
        inactive_hard[_ENFORCEMENT_WEIGHT][0, 0, 0] = 0.0
        invalid_cases += (("inactive legacy hard mask", inactive_hard, "active"),)

        for label, fields, message in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(PreflowSnapshotValidationError, message):
                    PreflowSnapshot(fields=fields, identity=_identity())

    def test_canonical_component_provenance_does_not_require_legacy_active_row(
        self,
    ) -> None:
        for label, external_exact_mask in (
            ("immersed hard component", 0),
            ("external exact component", 0b100),
        ):
            with self.subTest(label=label):
                fields = _valid_canonical_fields()
                row = (0, 0, 0)
                fields["velocity_dirichlet_boundary_active"][row] = 0
                fields["velocity_dirichlet_boundary_projection_weight"][row] = 0.0
                fields[_ENFORCEMENT_WEIGHT][row] = 0.0
                fields[_DIRECT_HARD_FIXED_COMPONENT_MASK][row] = 0b100
                fields[_EXTERNAL_EXACT_COMPONENT_MASK][row] = external_exact_mask
                fields[
                    "velocity_dirichlet_boundary_active_component_mask"
                ][row] = 0b100
                fields[
                    "velocity_dirichlet_boundary_pressure_mobility"
                ][row + (2,)] = 0.0
                fields[
                    "velocity_dirichlet_boundary_component_enforcement_weight"
                ][row + (2,)] = 1.0

                snapshot = PreflowSnapshot(
                    fields=fields,
                    identity=_identity(),
                    velocity_dirichlet_boundary_authority="canonical",
                    velocity_dirichlet_component_ledger_generation=11,
                )

                self.assertEqual(
                    snapshot.fields[
                        "velocity_dirichlet_boundary_active_component_mask"
                    ][row],
                    0b100,
                )

    def test_canonical_hard_mask_still_requires_matching_component_activity(
        self,
    ) -> None:
        fields = _valid_canonical_fields()
        fields[_DIRECT_HARD_FIXED_COMPONENT_MASK][0, 0, 0] = 0b100
        fields[_EXTERNAL_EXACT_COMPONENT_MASK][0, 0, 0] = 0
        fields["velocity_dirichlet_boundary_active_component_mask"][0, 0, 0] = 0

        with self.assertRaisesRegex(PreflowSnapshotValidationError, "subset"):
            PreflowSnapshot(
                fields=fields,
                identity=_identity(),
                velocity_dirichlet_boundary_authority="canonical",
                velocity_dirichlet_component_ledger_generation=11,
            )

    def test_canonical_component_ledger_accepts_legal_owned_obstacle_storage(
        self,
    ) -> None:
        fields = _valid_canonical_fields()
        row = (1, 0, 0)
        fields["velocity_dirichlet_boundary_active"][row] = 0
        fields["velocity_dirichlet_boundary_projection_weight"][row] = 0.0
        fields[_ENFORCEMENT_WEIGHT][row] = 0.0
        fields["obstacle"][row] = 1
        fields[_DIRECT_HARD_FIXED_COMPONENT_MASK][row] = 0b001
        fields[_EXTERNAL_EXACT_COMPONENT_MASK][row] = 0
        fields["velocity_dirichlet_boundary_active_component_mask"][row] = 0b001
        fields["velocity_dirichlet_boundary_owned_component_mask"][row] = 0b001
        fields["velocity_dirichlet_boundary_value_mps"][row] = (0.25, 0.0, 0.0)
        fields["velocity_dirichlet_boundary_pressure_mobility"][row] = (
            0.0,
            1.0,
            1.0,
        )
        fields[
            "velocity_dirichlet_boundary_component_enforcement_weight"
        ][row] = (1.0, 0.0, 0.0)

        snapshot = PreflowSnapshot(
            fields=fields,
            identity=_identity(),
            velocity_dirichlet_boundary_authority="canonical",
            velocity_dirichlet_component_ledger_generation=11,
        )

        self.assertEqual(snapshot.fields["obstacle"][row], 1)
        self.assertEqual(
            snapshot.fields["velocity_dirichlet_boundary_owned_component_mask"][
                row
            ],
            0b001,
        )

    def test_canonical_external_exact_mask_rejects_owned_component_overlap(
        self,
    ) -> None:
        fields = _valid_canonical_fields()
        row = (0, 0, 0)
        fields[_DIRECT_HARD_FIXED_COMPONENT_MASK][row] = 0b100
        fields[_EXTERNAL_EXACT_COMPONENT_MASK][row] = 0b100
        fields["velocity_dirichlet_boundary_active_component_mask"][row] = 0b100
        fields["velocity_dirichlet_boundary_owned_component_mask"][row] = 0b100

        with self.assertRaisesRegex(PreflowSnapshotValidationError, "overlap"):
            PreflowSnapshot(
                fields=fields,
                identity=_identity(),
                velocity_dirichlet_boundary_authority="canonical",
                velocity_dirichlet_component_ledger_generation=11,
            )

    def test_canonical_component_ledger_rejects_precommit_invariant_violations(
        self,
    ) -> None:
        invalid_cases: list[tuple[str, dict[str, np.ndarray], str]] = []

        hard_mobility = _valid_canonical_fields()
        hard_mobility[
            "velocity_dirichlet_boundary_pressure_mobility"
        ][0, 0, 1, 0] = 1.5e-6
        invalid_cases.append(("hard mobility", hard_mobility, "mobility"))

        hard_enforcement = _valid_canonical_fields()
        hard_enforcement[
            "velocity_dirichlet_boundary_component_enforcement_weight"
        ][0, 0, 1, 0] = 1.0 - 1.5e-6
        invalid_cases.append(
            ("hard enforcement", hard_enforcement, "enforcement")
        )

        soft_without_owner = _valid_canonical_fields()
        soft_without_owner["velocity_dirichlet_boundary_active_component_mask"][
            0, 0, 0
        ] = 0b001
        soft_without_owner[
            "velocity_dirichlet_boundary_pressure_mobility"
        ][0, 0, 0, 0] = 0.5
        soft_without_owner[
            "velocity_dirichlet_boundary_component_enforcement_weight"
        ][0, 0, 0, 0] = 0.5
        invalid_cases.append(
            ("soft component without owner", soft_without_owner, "ownership")
        )

        inactive_value = _valid_canonical_fields()
        inactive_value["velocity_dirichlet_boundary_value_mps"][0, 0, 0, 0] = 0.1
        invalid_cases.append(("inactive value", inactive_value, "inactive"))

        illegal_obstacle_storage = _valid_canonical_fields()
        row = (0, 0, 0)
        illegal_obstacle_storage["velocity_dirichlet_boundary_active"][row] = 0
        illegal_obstacle_storage[
            "velocity_dirichlet_boundary_projection_weight"
        ][row] = 0.0
        illegal_obstacle_storage[_ENFORCEMENT_WEIGHT][row] = 0.0
        illegal_obstacle_storage["velocity_dirichlet_boundary_active_component_mask"][
            row
        ] = 0b001
        illegal_obstacle_storage[_DIRECT_HARD_FIXED_COMPONENT_MASK][row] = 0b001
        illegal_obstacle_storage["velocity_dirichlet_boundary_owned_component_mask"][
            row
        ] = 0b001
        illegal_obstacle_storage[
            "velocity_dirichlet_boundary_pressure_mobility"
        ][row + (0,)] = 0.0
        illegal_obstacle_storage[
            "velocity_dirichlet_boundary_component_enforcement_weight"
        ][row + (0,)] = 1.0
        illegal_obstacle_storage["obstacle"][row] = 1
        invalid_cases.append(
            (
                "illegal obstacle storage",
                illegal_obstacle_storage,
                "obstacle storage",
            )
        )

        for label, fields, message in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(PreflowSnapshotValidationError, message):
                    PreflowSnapshot(
                        fields=fields,
                        identity=_identity(),
                        velocity_dirichlet_boundary_authority="canonical",
                        velocity_dirichlet_component_ledger_generation=11,
                    )

    def test_canonical_snapshot_still_rejects_incoherent_scalar_compatibility_rows(
        self,
    ) -> None:
        inactive_owned = _valid_canonical_fields()
        row = (0, 0, 0)
        inactive_owned["velocity_dirichlet_boundary_active"][row] = 0
        inactive_owned[_DYNAMIC_OWNED_ROW][row] = 1
        inactive_owned["velocity_dirichlet_boundary_projection_weight"][row] = 0.0
        inactive_owned[_ENFORCEMENT_WEIGHT][row] = 0.0

        active_obstacle = _valid_canonical_fields()
        active_obstacle["obstacle"][row] = 1

        inactive_marker_region = _valid_canonical_fields()
        inactive_marker_region["velocity_dirichlet_boundary_active"][row] = 0
        inactive_marker_region[
            "velocity_dirichlet_boundary_projection_weight"
        ][row] = 0.0
        inactive_marker_region[_ENFORCEMENT_WEIGHT][row] = 0.0
        inactive_marker_region[
            "velocity_dirichlet_boundary_marker_region_id"
        ][row] = 17

        for label, fields, message in (
            ("inactive owned row", inactive_owned, "owned rows"),
            ("active obstacle row", active_obstacle, "obstacle"),
            (
                "inactive marker region",
                inactive_marker_region,
                "marker region",
            ),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(PreflowSnapshotValidationError, message):
                    PreflowSnapshot(
                        fields=fields,
                        identity=_identity(),
                        velocity_dirichlet_boundary_authority="canonical",
                        velocity_dirichlet_component_ledger_generation=11,
                    )

    def test_load_rejects_each_expected_identity_mismatch(self) -> None:
        snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity())
        mismatches = {
            "config_sha256": canonical_config_sha256({"different": "config"}),
            "source_sha256": canonical_source_sha256({"different.py": "pass\n"}),
            "geometry_sha256": canonical_geometry_sha256(
                {"different": np.array([1.0], dtype=np.float64)}
            ),
        }

        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "preflow"
            save_preflow_snapshot(prefix, snapshot)
            for field_name, mismatched_hash in mismatches.items():
                values = {
                    "config_sha256": snapshot.identity.config_sha256,
                    "source_sha256": snapshot.identity.source_sha256,
                    "geometry_sha256": snapshot.identity.geometry_sha256,
                }
                values[field_name] = mismatched_hash
                expected = PreflowSnapshotIdentity(**values)
                with self.subTest(field_name=field_name):
                    with self.assertRaisesRegex(
                        PreflowSnapshotMismatchError,
                        field_name,
                    ):
                        load_preflow_snapshot(prefix, expected_identity=expected)

    def test_load_rejects_shape_dtype_content_hash_and_finite_tampering(self) -> None:
        snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity())
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            shape_files = save_preflow_snapshot(root / "shape", snapshot)
            _rewrite_manifest(
                shape_files.metadata_path,
                lambda manifest: manifest["fields"]["pressure"].update(
                    {"shape": [2, 3, 5]}
                ),
            )
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "shape"):
                load_preflow_snapshot(root / "shape", expected_identity=snapshot.identity)

            dtype_files = save_preflow_snapshot(root / "dtype", snapshot)
            _rewrite_manifest(
                dtype_files.metadata_path,
                lambda manifest: manifest["fields"]["velocity"].update(
                    {"dtype": "<f8"}
                ),
            )
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "dtype"):
                load_preflow_snapshot(root / "dtype", expected_identity=snapshot.identity)

            hard_fixed_shape_files = save_preflow_snapshot(
                root / "hard_fixed_shape", snapshot
            )
            _rewrite_manifest(
                hard_fixed_shape_files.metadata_path,
                lambda manifest: manifest["fields"][
                    _DIRECT_HARD_FIXED_COMPONENT_MASK
                ].update({"shape": [2, 3, 5]}),
            )
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "shape"):
                load_preflow_snapshot(
                    root / "hard_fixed_shape",
                    expected_identity=snapshot.identity,
                )

            hard_fixed_dtype_files = save_preflow_snapshot(
                root / "hard_fixed_dtype", snapshot
            )
            _rewrite_manifest(
                hard_fixed_dtype_files.metadata_path,
                lambda manifest: manifest["fields"][
                    _DIRECT_HARD_FIXED_COMPONENT_MASK
                ].update({"dtype": "<f4"}),
            )
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "dtype"):
                load_preflow_snapshot(
                    root / "hard_fixed_dtype",
                    expected_identity=snapshot.identity,
                )

            content_files = save_preflow_snapshot(root / "content", snapshot)
            _rewrite_npz(
                content_files.npz_path,
                lambda arrays: arrays["velocity"].__setitem__((0, 0, 0, 0), 77.0),
            )
            _rewrite_manifest(
                content_files.metadata_path,
                lambda manifest: manifest.update(
                    {"npz_sha256": _sha256_file(content_files.npz_path)}
                ),
            )
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "content hash"):
                load_preflow_snapshot(root / "content", expected_identity=snapshot.identity)

            hard_fixed_content_files = save_preflow_snapshot(
                root / "hard_fixed_content", snapshot
            )
            _rewrite_npz(
                hard_fixed_content_files.npz_path,
                lambda arrays: arrays[_DIRECT_HARD_FIXED_COMPONENT_MASK].__setitem__(
                    (0, 0, 0), 7
                ),
            )
            _rewrite_manifest(
                hard_fixed_content_files.metadata_path,
                lambda manifest: manifest.update(
                    {"npz_sha256": _sha256_file(hard_fixed_content_files.npz_path)}
                ),
            )
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "content hash"):
                load_preflow_snapshot(
                    root / "hard_fixed_content",
                    expected_identity=snapshot.identity,
                )

            semantic_files = save_preflow_snapshot(root / "semantic", snapshot)
            semantic_arrays = _rewrite_npz(
                semantic_files.npz_path,
                lambda arrays: arrays[_EXTERNAL_EXACT_COMPONENT_MASK].__setitem__(
                    (0, 0, 0), 0b100
                ),
            )

            def refresh_semantic_manifest(manifest: dict[str, object]) -> None:
                manifest["npz_sha256"] = _sha256_file(semantic_files.npz_path)
                manifest["fields"][_EXTERNAL_EXACT_COMPONENT_MASK]["sha256"] = (
                    canonical_geometry_sha256(
                        {
                            _EXTERNAL_EXACT_COMPONENT_MASK: semantic_arrays[
                                _EXTERNAL_EXACT_COMPONENT_MASK
                            ]
                        }
                    )
                )

            _rewrite_manifest(
                semantic_files.metadata_path,
                refresh_semantic_manifest,
            )
            with self.assertRaisesRegex(
                PreflowSnapshotIntegrityError,
                "payload failed schema validation.*subset",
            ):
                load_preflow_snapshot(
                    root / "semantic",
                    expected_identity=snapshot.identity,
                )

            finite_files = save_preflow_snapshot(root / "finite", snapshot)
            _rewrite_npz(
                finite_files.npz_path,
                lambda values: values["fsi_pressure"].__setitem__((0, 0, 0), np.inf),
            )

            def refresh_finite_manifest(manifest: dict[str, object]) -> None:
                manifest["npz_sha256"] = _sha256_file(finite_files.npz_path)

            _rewrite_manifest(finite_files.metadata_path, refresh_finite_manifest)
            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "finite"):
                load_preflow_snapshot(root / "finite", expected_identity=snapshot.identity)

    def test_interrupted_metadata_commit_cannot_load_a_mixed_snapshot(self) -> None:
        old_snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity(tag="old"))
        new_fields = _valid_fields()
        new_fields["velocity"] = new_fields["velocity"] + np.float32(1.0)
        new_snapshot = PreflowSnapshot(fields=new_fields, identity=_identity(tag="new"))
        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "preflow"
            save_preflow_snapshot(prefix, old_snapshot)
            real_replace = os.replace

            def fail_metadata_replace(source, destination) -> None:
                if Path(destination).suffix == ".json":
                    raise OSError("simulated metadata commit failure")
                real_replace(source, destination)

            with mock.patch(
                "simulation_core.fluids.preflow_snapshot.os.replace",
                side_effect=fail_metadata_replace,
            ):
                with self.assertRaisesRegex(OSError, "metadata commit failure"):
                    save_preflow_snapshot(prefix, new_snapshot)

            loaded_old = load_preflow_snapshot(
                prefix,
                expected_identity=old_snapshot.identity,
            )
            np.testing.assert_array_equal(
                loaded_old.fields["velocity"],
                old_snapshot.fields["velocity"],
            )
            with self.assertRaises(PreflowSnapshotMismatchError):
                load_preflow_snapshot(prefix, expected_identity=new_snapshot.identity)
            self.assertEqual(list(prefix.parent.glob("*.tmp-*")), [])
            self.assertEqual(len(list(prefix.parent.glob("preflow.*.npz"))), 2)

    def test_concurrent_writers_commit_one_complete_generation(self) -> None:
        fields_a = _valid_fields()
        fields_b = _valid_fields()
        fields_b["velocity"] = fields_b["velocity"] + np.float32(10.0)
        snapshots = {
            "writer-a": PreflowSnapshot(fields=fields_a, identity=_identity(tag="a")),
            "writer-b": PreflowSnapshot(fields=fields_b, identity=_identity(tag="b")),
        }
        a_npz_done = threading.Event()
        b_npz_done = threading.Event()
        b_manifest_done = threading.Event()
        real_replace = os.replace

        def ordered_replace(source, destination) -> None:
            writer = threading.current_thread().name
            destination_path = Path(destination)
            if destination_path.suffix == ".npz":
                if writer == "writer-b":
                    self.assertTrue(a_npz_done.wait(timeout=10.0))
                real_replace(source, destination)
                (a_npz_done if writer == "writer-a" else b_npz_done).set()
                return
            if writer == "writer-a":
                self.assertTrue(b_manifest_done.wait(timeout=10.0))
            else:
                self.assertTrue(b_npz_done.wait(timeout=10.0))
            real_replace(source, destination)
            if writer == "writer-b":
                b_manifest_done.set()

        def save_as(writer: str, prefix: Path) -> None:
            threading.current_thread().name = writer
            save_preflow_snapshot(prefix, snapshots[writer])

        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "concurrent"
            with mock.patch(
                "simulation_core.fluids.preflow_snapshot.os.replace",
                side_effect=ordered_replace,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(save_as, writer, prefix)
                        for writer in ("writer-a", "writer-b")
                    ]
                    for future in futures:
                        future.result(timeout=20.0)

            loaded = load_preflow_snapshot(
                prefix,
                expected_identity=snapshots["writer-a"].identity,
            )
            np.testing.assert_array_equal(
                loaded.fields["velocity"],
                snapshots["writer-a"].fields["velocity"],
            )

    def test_load_rejects_duplicate_json_keys_instead_of_using_last_value(self) -> None:
        snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity())
        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "duplicate_key"
            files = save_preflow_snapshot(prefix, snapshot)
            original = files.metadata_path.read_text(encoding="utf-8")
            files.metadata_path.write_text(
                '{"format":"untrusted-duplicate",' + original[1:],
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PreflowSnapshotIntegrityError, "duplicate"):
                load_preflow_snapshot(prefix, expected_identity=snapshot.identity)

    def test_history_and_hash_inputs_reject_noncanonical_values(self) -> None:
        with self.assertRaisesRegex(PreflowSnapshotValidationError, "history"):
            PreflowSnapshot(
                fields=_valid_fields(),
                identity=_identity(),
                history={"residual": np.nan},
            )
        with self.assertRaisesRegex(ValueError, "config"):
            canonical_config_sha256({"bad": object()})
        with self.assertRaisesRegex(ValueError, "source"):
            canonical_source_sha256({"bad.py": object()})
        with self.assertRaisesRegex(ValueError, "geometry"):
            canonical_geometry_sha256(
                {"vertices": np.array([object()], dtype=object)}
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            canonical_geometry_sha256(
                {"vertices": np.array([0.0, np.nan], dtype=np.float64)}
            )

    def test_snapshot_arrays_and_history_cannot_be_made_mutable(self) -> None:
        snapshot = PreflowSnapshot(
            fields=_valid_fields(),
            identity=_identity(),
            history={"residual": [1.0, 0.1]},
        )
        with self.assertRaises(ValueError):
            snapshot.fields["velocity"].setflags(write=True)
        with self.assertRaises(TypeError):
            snapshot.history["new"] = 1
        with self.assertRaises(AttributeError):
            snapshot.history["residual"].append(0.01)

    def test_self_hashed_malformed_npz_is_reported_as_integrity_error(self) -> None:
        snapshot = PreflowSnapshot(fields=_valid_fields(), identity=_identity())
        with TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "malformed"
            files = save_preflow_snapshot(prefix, snapshot)
            files.npz_path.write_bytes(b"PK\x03\x04not-a-valid-zip")
            _rewrite_manifest(
                files.metadata_path,
                lambda manifest: manifest.update(
                    {"npz_sha256": _sha256_file(files.npz_path)}
                ),
            )

            with self.assertRaises(PreflowSnapshotIntegrityError):
                load_preflow_snapshot(prefix, expected_identity=snapshot.identity)


if __name__ == "__main__":
    unittest.main()
