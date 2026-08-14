from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_native_fine_fsi_gate.py")
SPEC = importlib.util.spec_from_file_location("run_native_fine_fsi_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def copy_native_mesh_fixture(root: Path) -> tuple[Path, Path, dict]:
    mesh = root / "fine.msh"
    shutil.copy2(MODULE.DEFAULT_MESH, mesh)
    payload = json.loads(MODULE.DEFAULT_MESH_MANIFEST.read_text(encoding="utf-8"))
    payload["output_mesh"] = str(mesh)
    manifest = root / "fine_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return mesh, manifest, payload


class NativeFineFsiGateTests(unittest.TestCase):
    def test_mesh_manifest_gate_requires_native_quad_solid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh, manifest, _ = copy_native_mesh_fixture(root)

            report = MODULE.validate_offline_mesh(mesh, manifest)

        self.assertEqual(report["solid_cell_count"], 480)
        self.assertEqual(report["solid_cell_type_counts"], {3: 480})
        self.assertEqual(report["cross_cell_zone_face_count"], 92)
        self.assertEqual(
            report["actual_mesh_sha256"],
            "c80450a0806cac084fa847b91cb35bff38dad7b86e2f97e6a29f04d7409506ca",
        )
        self.assertEqual(
            report["parsed_bounds_m"],
            {"x_min": 0.0, "x_max": 0.1, "y_min": 0.0, "y_max": 0.02},
        )

    def test_mesh_manifest_gate_rejects_polygonal_solid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh, manifest, payload = copy_native_mesh_fixture(root)
            payload["validation"]["cell_types_by_zone_name"]["solid.5"] = {
                "3": 479,
                "7": 1,
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "type 7|unsupported"):
                MODULE.validate_offline_mesh(mesh, manifest)

    def test_mesh_gate_rejects_manifest_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh, manifest, payload = copy_native_mesh_fixture(root)
            payload["output_mesh_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                MODULE.validate_offline_mesh(mesh, manifest)

    def test_mesh_gate_parses_mesh_instead_of_trusting_manifest_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = root / "forged.msh"
            mesh.write_text("(2 2)\n", encoding="ascii")
            payload = json.loads(
                MODULE.DEFAULT_MESH_MANIFEST.read_text(encoding="utf-8")
            )
            payload["output_mesh"] = str(mesh)
            payload["output_mesh_sha256"] = hashlib.sha256(
                mesh.read_bytes()
            ).hexdigest()
            manifest = root / "forged_manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "zone-name|node ids|no cells"):
                MODULE.validate_offline_mesh(mesh, manifest)

    def test_steady_setup_is_official_and_contains_no_adaptation(self) -> None:
        commands = MODULE.official_steady_commands()
        joined = "\n".join(commands).lower()

        self.assertIn("kw-sst", joined)
        self.assertIn("vmag no 10", joined)
        self.assertIn("operating-pressure 1013250", joined)
        self.assertIn("constant 1.2", joined)
        self.assertIn("constant 1.8e-05", joined)
        self.assertNotIn("adapt", joined)
        self.assertNotIn("puma", joined)

    def test_serial_gate_configuration_rejects_parallel_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MODULE.NativeGateConfig(
                run_dir=Path(tmp) / "run",
                mesh_path=Path(tmp) / "mesh.msh",
                mesh_manifest_path=Path(tmp) / "mesh_manifest.json",
                processor_count=2,
            )
            with self.assertRaisesRegex(ValueError, "serial"):
                MODULE.validate_config(config)


if __name__ == "__main__":
    unittest.main()
