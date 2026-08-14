from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

from simulation_core.coupling.hibm_mpm.core import (
    _assemble_and_seal_hibm_velocity_component_face_ledger,
    advance_hibm_mpm_sharp_mpm_step,
    assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class HibmCanonicalCoreMigrationTests(unittest.TestCase):
    def test_old_reconstructed_boundary_writer_is_deleted(self) -> None:
        production_roots = (
            REPOSITORY_ROOT / "simulation_core",
            REPOSITORY_ROOT / "cases",
            REPOSITORY_ROOT / "benchmarks" / "official",
        )
        old_symbol = "assemble_velocity_dirichlet_reconstructed_boundary_rows"
        offenders = []
        for root in production_roots:
            for path in root.rglob("*.py"):
                if old_symbol in path.read_text(encoding="utf-8"):
                    offenders.append(path.relative_to(REPOSITORY_ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_generic_hibm_step_uses_only_component_face_assembly(self) -> None:
        for function in (
            assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
            advance_hibm_mpm_sharp_mpm_step,
        ):
            source = inspect.getsource(function)
            self.assertIn(
                "_assemble_and_seal_hibm_velocity_component_face_ledger",
                source,
            )
            self.assertNotIn(
                "assemble_velocity_dirichlet_reconstructed_boundary_rows",
                source,
            )

    def test_relocation_source_key_remains_a_taichi_function(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "simulation_core"
            / "coupling"
            / "hibm_mpm"
            / "core.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "@ti.func\n    def _velocity_dirichlet_relocation_source_linear_key",
            source,
        )

    def test_shared_hibm_assembler_rejects_noncanonical_authority(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "HIBM-MPM requires canonical component-face",
        ):
            _assemble_and_seal_hibm_velocity_component_face_ledger(
                fluid=SimpleNamespace(
                    velocity_dirichlet_boundary_authority="legacy"
                ),
                markers=None,
                ib_search=None,
                ib_boundary=None,
                surface_projection_inactive_axis=None,
                primary_region_id=7,
                secondary_region_id=8,
                interpolate_interior_velocity=False,
            )

    def test_official_hibm_runner_selects_canonical_authority(self) -> None:
        runner_source = (
            REPOSITORY_ROOT
            / "benchmarks"
            / "official"
            / "solid_mpm_fsi_runner.py"
        ).read_text(encoding="utf-8")
        build_fluid_source = runner_source.split("def _build_fluid(", 1)[1].split(
            "def ",
            1,
        )[0]
        self.assertIn("_use_hibm_sharp_marker_boundary(config)", build_fluid_source)
        self.assertIn(
            'set_velocity_dirichlet_boundary_authority("canonical")',
            build_fluid_source,
        )

    def test_turek_hron_uses_canonical_external_face_boundaries(self) -> None:
        case_source = (
            REPOSITORY_ROOT / "cases" / "turek_hron_fsi.py"
        ).read_text(encoding="utf-8")
        kernel_source = (
            REPOSITORY_ROOT / "cases" / "turek_hron_kernels.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'set_velocity_dirichlet_boundary_authority("canonical")',
            case_source,
        )
        self.assertNotIn("velocity_dirichlet_face_symmetric", case_source)
        self.assertNotIn("th_channel_boundary_rows_kernel", case_source)
        self.assertIn("th_channel_external_velocity_faces_kernel", case_source)
        self.assertIn("th_channel_external_velocity_faces_kernel", kernel_source)

    def test_squid_sharp_builder_selects_canonical_authority(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "cases"
            / "squid_soft_robot"
            / "coupling_sharp.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'set_velocity_dirichlet_boundary_authority("canonical")',
            source,
        )


if __name__ == "__main__":
    unittest.main()
