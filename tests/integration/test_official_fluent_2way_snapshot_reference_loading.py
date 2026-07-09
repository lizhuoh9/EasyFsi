from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SNAPSHOT_SCRIPT_PATH = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_official_fluent_2way_fsi50_snapshot.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SNAPSHOT = _load_module(
    "run_official_fluent_2way_fsi50_snapshot_for_tests",
    SNAPSHOT_SCRIPT_PATH,
)


class OfficialFluent2WaySnapshotReferenceLoadingTests(unittest.TestCase):
    """_reference_mesh_summary must never unpickle CLI-controlled NPZ files.

    The producer (_write_field_npz in src/refactored/validation/
    ansys_vertical_flap_fsi/official_fluent_reference.py) stores
    mesh_summary_json as a plain json.dumps() string, so allow_pickle=False
    is sufficient for all legitimate references and mandatory for untrusted
    ones.
    """

    def test_plain_json_string_mesh_summary_loads_without_pickle(self) -> None:
        mesh_summary = {
            "cell_zone_center_bounds": {"fluid.4": {"count": 12}},
            "face_zone_node_bounds": {
                "symmetry.2": {"count": 8},
                "velocity_inlet.1": {"count": 4},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            reference_root = Path(tmp)
            np.savez_compressed(
                reference_root / "steady_fluent_fields.npz",
                x=np.linspace(0.0, 1.0, 4),
                y=np.linspace(0.0, 1.0, 4),
                mesh_summary_json=json.dumps(mesh_summary, sort_keys=True),
            )

            loaded = SNAPSHOT._reference_mesh_summary(reference_root)

        self.assertEqual(loaded, mesh_summary)

    def test_coordinate_fallback_still_works_without_mesh_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_root = Path(tmp)
            np.savez_compressed(
                reference_root / "fsi50_final_fluent_fields.npz",
                x=np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]),
                y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            )

            summary = SNAPSHOT._reference_mesh_summary(reference_root)

        self.assertEqual(
            summary["derived_from_field_coordinates"],
            {"x_unique_count": 3, "y_unique_count": 2},
        )

    def test_object_array_npz_is_rejected_not_unpickled(self) -> None:
        # A malicious/corrupted NPZ that stores an object payload must fail
        # closed: numpy refuses object arrays when allow_pickle=False.
        with tempfile.TemporaryDirectory() as tmp:
            reference_root = Path(tmp)
            np.savez(
                reference_root / "steady_fluent_fields.npz",
                mesh_summary_json=np.array(
                    {"__reduce__": "arbitrary object"}, dtype=object
                ),
            )

            with self.assertRaises(ValueError):
                SNAPSHOT._reference_mesh_summary(reference_root)

    def test_snapshot_script_never_enables_allow_pickle(self) -> None:
        source = SNAPSHOT_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("allow_pickle=True", source)
        self.assertIn("allow_pickle=False", source)


if __name__ == "__main__":
    unittest.main()
