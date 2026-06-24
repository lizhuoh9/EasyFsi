from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from simulation_core.cad_import import (
    cad_provenance_report,
    parse_step_cad_summary,
)


class CadImportTests(unittest.TestCase):
    def test_step_summary_extracts_schema_units_hash_and_brep_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            step_path = Path(temp_dir) / "sample.STEP"
            step_text = """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=SI_UNIT(.MILLI.,.METRE.);
#11=MANIFOLD_SOLID_BREP('membrane',#20);
#12=MANIFOLD_SOLID_BREP('noozle',#30);
ENDSEC;
END-ISO-10303-21;
"""
            step_path.write_text(step_text, encoding="utf-8")

            summary = parse_step_cad_summary(step_path)

            self.assertEqual(summary.path, str(step_path.resolve()))
            self.assertEqual(summary.sha256, hashlib.sha256(step_path.read_bytes()).hexdigest())
            self.assertEqual(summary.length_unit, "millimetre")
            self.assertEqual(summary.file_schema, ("AUTOMOTIVE_DESIGN_CC2",))
            self.assertEqual([brep.name for brep in summary.breps], ["membrane", "noozle"])
            self.assertEqual([brep.step_id for brep in summary.breps], [11, 12])

    def test_cad_provenance_distinguishes_direct_step_from_cached_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cad_path = temp_path / "sim.STEP"
            cad_path.write_text(
                """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=SI_UNIT(.MILLI.,.METRE.);
#10=MANIFOLD_SOLID_BREP('chamber',#20);
ENDSEC;
END-ISO-10303-21;
""",
                encoding="utf-8",
            )
            cached_stl = temp_path / "current_sim_step_4body_surface_mesh.stl"
            cached_stl.write_text("solid cached\nendsolid cached\n", encoding="utf-8")
            source_config = {
                "mesh_format": "step",
                "mesh_path": str(cached_stl),
                "surface_mesh_cache_path": str(cached_stl),
            }

            report = cad_provenance_report(cad_path, source_config=source_config)

            self.assertTrue(report["cad_exists"])
            self.assertEqual(report["cad_step_brep_names"], ["chamber"])
            self.assertEqual(report["source_config_mesh_suffix"], ".stl")
            self.assertEqual(report["source_config_surface_mesh_cache_suffix"], ".stl")
            self.assertFalse(report["source_config_mesh_path_matches_cad_step"])
            self.assertFalse(report["source_config_surface_mesh_cache_path_matches_cad_step"])
            self.assertFalse(report["direct_cad_step_binding"])

            source_config["mesh_path"] = str(cad_path)
            report = cad_provenance_report(cad_path, source_config=source_config)
            self.assertTrue(report["source_config_mesh_path_matches_cad_step"])
            self.assertFalse(report["source_config_surface_mesh_cache_path_matches_cad_step"])
            self.assertTrue(report["surface_mesh_cache_requires_provenance"])
            self.assertFalse(report["direct_cad_step_binding"])

            source_config["surface_mesh_cache_path"] = str(cad_path)
            report = cad_provenance_report(cad_path, source_config=source_config)
            self.assertTrue(report["source_config_mesh_path_matches_cad_step"])
            self.assertTrue(report["direct_cad_step_binding"])

    def test_cad_provenance_reports_declared_step_source_without_calling_it_direct(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cad_path = temp_path / "sim.STEP"
            cad_path.write_text(
                """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=SI_UNIT(.MILLI.,.METRE.);
#10=MANIFOLD_SOLID_BREP('chamber',#20);
ENDSEC;
END-ISO-10303-21;
""",
                encoding="utf-8",
            )
            cached_stl = temp_path / "cached.stl"
            cached_stl.write_text("solid cached\nendsolid cached\n", encoding="utf-8")
            source_config = {
                "mesh_path": str(cached_stl),
                "surface_mesh_cache_path": str(cached_stl),
                "metadata": {"source_step": str(cad_path)},
            }

            report = cad_provenance_report(cad_path, source_config=source_config)

            self.assertTrue(report["real_cad_step_source_declared"])
            self.assertTrue(
                report["source_config_declared_source_step_path_matches_cad_step"]
            )
            self.assertFalse(report["direct_cad_step_binding"])

    def test_cad_provenance_accepts_verified_step_derived_surface_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cad_path = temp_path / "sim.STEP"
            cad_path.write_text(
                """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=SI_UNIT(.MILLI.,.METRE.);
#10=MANIFOLD_SOLID_BREP('chamber',#20);
ENDSEC;
END-ISO-10303-21;
""",
                encoding="utf-8",
            )
            cache_path = temp_path / "sim.surface_mesh.stl"
            cache_path.write_text("solid derived\nendsolid derived\n", encoding="utf-8")
            source_config = {
                "mesh_format": "step",
                "mesh_path": str(cad_path),
                "surface_mesh_cache_path": str(cache_path),
                "mesh_import": {
                    "source_step_path": str(cad_path),
                    "source_step_sha256": hashlib.sha256(cad_path.read_bytes()).hexdigest(),
                    "surface_mesh_cache_path": str(cache_path),
                    "surface_mesh_cache_sha256": hashlib.sha256(
                        cache_path.read_bytes()
                    ).hexdigest(),
                },
            }

            report = cad_provenance_report(cad_path, source_config=source_config)

            self.assertFalse(report["direct_cad_step_binding"])
            self.assertTrue(report["step_derived_surface_mesh_binding"])
            self.assertTrue(report["real_cad_step_binding"])


if __name__ == "__main__":
    unittest.main()
