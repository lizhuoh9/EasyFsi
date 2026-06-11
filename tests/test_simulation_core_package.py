from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from simulation_core import (
    CartesianGrid,
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    FluidDomainSpec,
    GradedGridSpec,
    InterfaceReactionFixedPointResult,
    InterfaceReactionTargetEvaluation,
    InterfaceReactionUpdate,
    NeoHookeanMaterial,
    RefinementRegion,
    RegionPairInterfaceReactionTarget,
    SurfaceMesh,
    TaichiRuntimeConfig,
    UvSphereResolution,
    ecoflex_0010_material,
    HibmMpmNoSlipResidualReport,
    HibmMpmSharpCouplingState,
    HibmMpmPressureNeumannGradientReport,
    advance_hibm_mpm_sharp_mpm_step,
    advance_hibm_mpm_sharp_neo_hookean_step,
    assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
    build_graded_grid,
    fsi_coupling_mode_report,
    hibm_mpm_sharp_step_summary,
    hibm_mpm_paper_requirements,
    region_pair_interface_reaction_forces,
    relax_interface_reaction_forces,
    require_implemented_fsi_coupling_mode,
    solve_and_apply_interface_reaction_step,
    solve_interface_reaction_fixed_point,
)
from simulation_core.geometry import make_uv_sphere, orient_faces_outward


class SimulationCorePackageTests(unittest.TestCase):
    def test_core_package_exports_basic_primitives(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        self.assertEqual(runtime.arch, "cuda")

        resolution = UvSphereResolution(latitude_bands=4, longitude_segments=8)
        mesh = make_uv_sphere(resolution, radius_m=1.0)
        self.assertIsInstance(mesh, SurfaceMesh)
        self.assertEqual(mesh.vertex_count, resolution.vertex_count)
        self.assertEqual(mesh.face_count, resolution.face_count)

        fluid = FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8))
        self.assertEqual(fluid.grid_nodes, (8, 8, 8))
        grid = build_graded_grid(
            GradedGridSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                farfield_spacing_m=0.25,
                max_growth_ratio=1.2,
                refinement_regions=(
                    RefinementRegion(
                        bounds_min_m=(0.25, 0.25, 0.25),
                        bounds_max_m=(0.75, 0.75, 0.75),
                        target_spacing_m=0.125,
                    ),
                ),
            )
        )
        self.assertIsInstance(grid, CartesianGrid)
        self.assertEqual(grid.bounds_min_m, (0.0, 0.0, 0.0))

        ecoflex = ecoflex_0010_material()
        self.assertIsInstance(ecoflex, NeoHookeanMaterial)
        self.assertGreater(ecoflex.shear_modulus_pa, 0.0)

    def test_hibm_mpm_phase0_paper_requirements_are_solver_level(self) -> None:
        requirements = hibm_mpm_paper_requirements()
        names = {item["requirement"] for item in requirements}

        expected = {
            "Taichi-resident solver path",
            "IB node search",
            "inside/outside classification",
            "normal reconstruction",
            "velocity Dirichlet no-slip",
            "pressure Neumann matrix rows",
            "full-stress traction",
            "per-marker MPM external force",
            "surface feedback",
        }
        self.assertTrue(expected <= names)
        self.assertTrue(
            all(item["current_status"] in {"missing", "partial"} for item in requirements)
        )
        status_by_name = {
            item["requirement"]: item["current_status"] for item in requirements
        }
        self.assertEqual(status_by_name["Taichi-resident solver path"], "partial")
        self.assertEqual(status_by_name["surface markers"], "partial")
        self.assertEqual(status_by_name["IB node search"], "partial")
        self.assertEqual(status_by_name["inside/outside classification"], "partial")
        self.assertEqual(status_by_name["velocity Dirichlet no-slip"], "partial")
        self.assertEqual(status_by_name["pressure Neumann matrix rows"], "partial")
        self.assertEqual(status_by_name["full-stress traction"], "partial")
        self.assertEqual(status_by_name["per-marker MPM external force"], "partial")
        self.assertEqual(status_by_name["surface feedback"], "partial")
        self.assertTrue(
            all("squid" not in item["requirement"].lower() for item in requirements)
        )

    def test_hibm_mpm_contract_does_not_use_numpy_cpu_solver_path(self) -> None:
        source = Path("simulation_core/hibm_mpm.py").read_text(encoding="utf-8")

        self.assertIn("Taichi-resident solver path", source)
        self.assertIn("NumPy loops", source)
        self.assertIn("host round-trips", source)
        self.assertNotIn("import numpy", source)
        self.assertNotIn(".to_numpy(", source)
        self.assertNotIn(".from_numpy(", source)

    def test_hibm_no_slip_dirichlet_uses_fluid_boundary_rows_not_legacy_constraints(
        self,
    ) -> None:
        hibm_source = Path("simulation_core/hibm_mpm.py").read_text(encoding="utf-8")
        fluid_source = Path("simulation_core/fluid.py").read_text(encoding="utf-8")

        self.assertIn("assemble_velocity_dirichlet_boundary_rows", hibm_source)
        self.assertIn("velocity_dirichlet_boundary_active", fluid_source)
        self.assertIn("_apply_velocity_dirichlet_boundary_rows_kernel(0)", fluid_source)
        self.assertNotIn("velocity_constraint_blend", hibm_source)
        self.assertNotIn("solid_mobility_ratio", hibm_source)

    def test_hibm_surface_feedback_is_marker_to_mpm_taichi_field_path(self) -> None:
        source = Path("simulation_core/hibm_mpm.py").read_text(encoding="utf-8")

        self.assertIn("load_markers_from_surface_fields", source)
        self.assertIn("_load_markers_from_surface_fields_kernel", source)
        self.assertIn("surface_position_m", source)
        self.assertIn("surface_normal", source)
        self.assertIn("surface_area_m2", source)
        self.assertIn("surface_region_id", source)
        self.assertIn("surface_velocity_mps", source)
        self.assertIn("_load_markers_from_surface_velocity_fields_kernel", source)
        self.assertIn("update_surface_feedback_from_mpm_particles", source)
        self.assertIn("update_surface_feedback_from_mpm_surface_particles", source)
        self.assertIn("particle_position_m", source)
        self.assertIn("particle_velocity_mps", source)
        self.assertIn("particle_normal", source)
        self.assertIn("particle_area_m2", source)
        self.assertIn("report_surface_feedback_invalid_marker_count", source)
        self.assertIn("geometry_updated_marker_count", source)
        self.assertNotIn(".to_numpy(", source)
        self.assertNotIn(".from_numpy(", source)

    def test_hibm_sharp_load_assembly_is_core_marker_to_mpm_path(self) -> None:
        source = Path("simulation_core/hibm_mpm.py").read_text(encoding="utf-8")
        mooney_source = Path("simulation_core/mooney_shell_mpm.py").read_text(
            encoding="utf-8"
        )
        hibm_tests = Path("tests/test_hibm.py").read_text(encoding="utf-8")

        self.assertTrue(callable(assemble_hibm_mpm_sharp_fluid_to_mpm_loads))
        self.assertTrue(callable(HibmMpmSharpCouplingState))
        self.assertTrue(callable(HibmMpmNoSlipResidualReport))
        self.assertTrue(callable(HibmMpmPressureNeumannGradientReport))
        self.assertTrue(callable(advance_hibm_mpm_sharp_mpm_step))
        self.assertTrue(callable(advance_hibm_mpm_sharp_neo_hookean_step))
        self.assertTrue(callable(hibm_mpm_sharp_step_summary))
        self.assertIn("assemble_hibm_mpm_sharp_fluid_to_mpm_loads", source)
        self.assertIn("HibmMpmSharpCouplingState", source)
        self.assertIn("advance_hibm_mpm_sharp_mpm_step", source)
        self.assertIn("advance_hibm_mpm_sharp_neo_hookean_step", source)
        self.assertIn("hibm_mpm_sharp_step_summary", source)
        self.assertIn('"hibm_no_slip_residual_max_mps"', source)
        self.assertIn('"hibm_ib_invalid_projection_count"', source)
        self.assertIn("search_and_classify_grid_fields", source)
        self.assertIn("cell_center_x_m=fluid.cell_center_x_m", source)
        self.assertIn("cell_center_y_m=fluid.cell_center_y_m", source)
        self.assertIn("cell_center_z_m=fluid.cell_center_z_m", source)
        self.assertIn("solid_step", source)
        self.assertIn("mpm_particle_velocity_mps", source)
        self.assertIn("mpm_particle_normal", source)
        self.assertIn("mpm_particle_area_m2", source)
        self.assertIn("surface_normal", mooney_source)
        self.assertIn("_update_particle_surface_normals", mooney_source)
        self.assertIn("TriMooneyShellMpmState", hibm_tests)
        self.assertIn("clear_mpm_external_forces", source)
        self.assertIn("scatter_marker_forces_to_mpm_particles", source)
        self.assertIn("sample_no_slip_residual", source)
        self.assertIn("no_slip_residual", source)
        self.assertIn("sample_fluid_stress_to_marker_tractions", source)
        self.assertIn("fluid_substep_dt", source)
        self.assertIn("fluid.predict(", source)
        self.assertIn("dt_s=fluid_substep_dt", source)
        self.assertIn("advection_scheme=advection_scheme", source)
        self.assertIn("fluid_predictor_applied", source)
        self.assertIn(
            "update_pressure_neumann_gradient_from_fluid_predictor",
            source,
        )
        self.assertIn("(predictor_velocity - self.v_gamma_mps[marker]).dot", source)
        self.assertIn("pressure_neumann_density_kgm3", source)
        self.assertIn("update_surface_feedback_from_mpm_particles", source)
        self.assertNotIn("interface_reaction_force", source)
        self.assertNotIn("target damping", source.lower())

    def test_hibm_compat_module_does_not_export_cpu_numpy_solver_path(self) -> None:
        hibm_source = Path("simulation_core/hibm.py").read_text(encoding="utf-8")
        init_source = Path("simulation_core/__init__.py").read_text(encoding="utf-8")

        self.assertNotIn("import numpy", hibm_source)
        self.assertNotIn("np.", hibm_source)
        self.assertNotIn("from .hibm import", init_source)
        self.assertNotIn('"classify_hibm_near_boundary_nodes"', init_source)
        self.assertNotIn('"compute_hibm_surface_tractions"', init_source)

    def test_hibm_mpm_mode_report_does_not_relabel_legacy_coupling(self) -> None:
        legacy_report = fsi_coupling_mode_report(FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED)
        sharp_report = fsi_coupling_mode_report(FSI_COUPLING_MODE_HIBM_MPM_SHARP)

        self.assertTrue(legacy_report["legacy"])
        self.assertTrue(legacy_report["implemented"])
        self.assertFalse(legacy_report["paper_hibm_mpm"])
        self.assertTrue(legacy_report["main_tail_region_reaction_diagnostic_only"])
        self.assertEqual(legacy_report["solver_layer"], "simulation_core")

        self.assertFalse(sharp_report["legacy"])
        self.assertTrue(sharp_report["implemented"])
        self.assertTrue(sharp_report["paper_hibm_mpm"])
        self.assertTrue(sharp_report["core_runner_available"])
        self.assertTrue(sharp_report["case_runner_available"])
        self.assertFalse(sharp_report["phase5_validation_complete"])
        self.assertEqual(sharp_report["missing"], ["Phase 5 fine-nozzle validation"])
        self.assertNotIn("surface markers", sharp_report["missing"])
        self.assertNotIn("pressure Neumann matrix rows", sharp_report["missing"])

    def test_hibm_mpm_sharp_mode_is_runnable_but_not_phase5_validated(self) -> None:
        report = require_implemented_fsi_coupling_mode(FSI_COUPLING_MODE_HIBM_MPM_SHARP)

        self.assertEqual(report["mode"], FSI_COUPLING_MODE_HIBM_MPM_SHARP)
        self.assertTrue(report["implemented"])
        self.assertTrue(report["core_runner_available"])
        self.assertTrue(report["case_runner_available"])
        self.assertTrue(report["paper_hibm_mpm"])
        self.assertFalse(report["phase5_validation_complete"])
        self.assertEqual(report["missing"], ["Phase 5 fine-nozzle validation"])

        legacy_report = require_implemented_fsi_coupling_mode(
            FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED
        )
        self.assertEqual(
            legacy_report["mode"],
            FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
        )

    def test_phase0_paper_vs_code_table_lists_missing_hibm_mpm_requirements(self) -> None:
        table = Path("HIBM_MPM_PAPER_VS_CODE.md").read_text(encoding="utf-8")

        required_terms = (
            "IB node search",
            "inside/outside classification",
            "normal reconstruction",
            "velocity Dirichlet no-slip",
            "pressure Neumann",
            "full-stress traction",
            "per-marker MPM external force",
            "surface feedback",
            "legacy projected/reduced",
            "not paper HIBM-MPM",
        )
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, table)

    def test_core_package_contains_no_case_modules(self) -> None:
        paths = sorted(Path("simulation_core").glob("*.py"))
        names = {path.name for path in paths}
        forbidden_names = {
            "run_case.py",
            "reference.py",
            "ball_check_valve.py",
            "mems_fsi.py",
            "fluent_fsi_flap.py",
            "tri_surface_mpm.py",
            "soft_sphere.py",
            "capsule.py",
            "rigid_sphere.py",
            "rigid_sphere_ib.py",
            "taichi_inflation.py",
            "device_immersed_boundary.py",
            "membrane.py",
            "mpm_shell.py",
            "surface_coupling.py",
        }
        self.assertFalse(names & forbidden_names)
        self.assertFalse(Path("simulation_core/cases").exists())

    def test_core_package_exports_no_obsolete_uv_shell_helpers(self) -> None:
        import simulation_core

        obsolete_public_names = (
            "DeviceSphereSharpInterfaceGrid",
            "DeviceSurfaceStressReport",
            "DeviceUvSphereBoundary",
            "MembraneElasticityReport",
            "SharpInterfaceEnforcementReport",
            "SharpInterfaceReconstructionReport",
            "ShellMpmState",
            "ShellMpmTransferReport",
            "SurfaceStressCouplingReport",
            "UvMembraneElasticity",
            "UvSurfaceStressCoupler",
        )

        for name in obsolete_public_names:
            self.assertFalse(hasattr(simulation_core, name), name)

    def test_core_package_exports_no_old_feedback_api(self) -> None:
        exported_primitives = (
            InterfaceReactionFixedPointResult,
            InterfaceReactionTargetEvaluation,
            InterfaceReactionUpdate,
            RegionPairInterfaceReactionTarget,
            region_pair_interface_reaction_forces,
            relax_interface_reaction_forces,
            solve_and_apply_interface_reaction_step,
            solve_interface_reaction_fixed_point,
        )
        for primitive in exported_primitives:
            self.assertTrue(callable(primitive), primitive)

        import simulation_core

        self.assertFalse(hasattr(simulation_core, "FeedbackUpdate"))
        self.assertFalse(hasattr(simulation_core, "FeedbackTargetEvaluation"))
        self.assertFalse(hasattr(simulation_core, "FeedbackFixedPointResult"))
        self.assertFalse(hasattr(simulation_core, "relax_feedback_forces"))
        self.assertFalse(hasattr(simulation_core, "solve_feedback_fixed_point"))
        self.assertFalse(hasattr(simulation_core, "solve_interface_reaction_step"))

    def test_orient_faces_outward_uses_mesh_center_for_translated_meshes(self) -> None:
        vertices = np.array(
            [
                [99.0, -1.0, -1.0],
                [99.0, 1.0, -1.0],
                [99.0, 0.0, 1.0],
                [101.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        mesh = SurfaceMesh(vertices=vertices, faces=np.array([[0, 1, 2]], dtype=np.int32))

        oriented = orient_faces_outward(mesh)
        face = oriented.faces[0]
        a, b, c = oriented.vertices[face]
        normal = np.cross(b - a, c - a)
        centroid = (a + b + c) / 3.0
        mesh_center = oriented.vertices.mean(axis=0)

        self.assertGreater(float(np.dot(normal, centroid - mesh_center)), 0.0)

    def test_core_source_has_no_case_runner_imports(self) -> None:
        forbidden_tokens = (
            "simulation_code.cases",
            "run_case",
            "ball_check_valve",
            "mems_fsi",
            "fluent_fsi_flap",
            "soft_sphere",
            "capsule_shear",
            "falling_sphere",
            "inflating_sphere",
        )
        for path in Path("simulation_core").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, source, msg=f"{path}: {token}")

    def test_core_source_has_no_z_only_or_old_compatibility_interfaces(self) -> None:
        forbidden_tokens = (
            "target_velocity_z",
            "primary_velocity_z",
            "secondary_velocity_z",
            "main_velocity_z",
            "tail_velocity_z",
            "nozzle_velocity_z",
            "pressure_force_scale",
            "component_pair(",
        )
        for path in Path("simulation_core").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, source, msg=f"{path}: {token}")

    def test_core_source_has_no_case_region_defaults(self) -> None:
        forbidden_tokens = (
            "primary_region_id: int = 7",
            "secondary_region_id: int = 8",
            "primary_region_id=7",
            "secondary_region_id=8",
        )
        for path in Path("simulation_core").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, source, msg=f"{path}: {token}")

    def test_core_runtime_paths_do_not_download_taichi_fields_to_numpy(self) -> None:
        allowed_from_numpy_calls = {
            "self.x.from_numpy(vertices)",
            "self.rest_x.from_numpy(vertices)",
            "self.saved_x.from_numpy(vertices)",
            "self.face_indices.from_numpy(faces)",
            "self.face_region_id.from_numpy(regions)",
            "self.edge_indices.from_numpy(edges)",
        }
        allowed_to_numpy_calls = {
            "snapshot = self.report_host_snapshot.to_numpy()",
            "for region in tri_surface.region_id.to_numpy()[: int(tri_surface.face_count)]",
        }
        for path in Path("simulation_core").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                if ".to_numpy(" in line:
                    self.assertIn(stripped, allowed_to_numpy_calls, msg=f"{path}: {stripped}")
                if ".from_numpy(" not in line:
                    continue
                self.assertEqual(path.name, "mooney_shell_mpm.py", msg=f"{path}: {stripped}")
                self.assertIn(stripped, allowed_from_numpy_calls, msg=f"{path}: {stripped}")

    def test_no_legacy_simulation_code_package_or_imports(self) -> None:
        self.assertFalse(Path("simulation_code").exists())

        for path in Path("tests").glob("test_*.py"):
            if path.name == Path(__file__).name:
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("simulation_code", source, msg=str(path))

    def test_no_stale_legacy_runner_bytecode(self) -> None:
        forbidden_stems = (
            "device_immersed_boundary",
            "membrane",
            "mpm_shell",
            "prepare_config",
            "render_current_squid_visit_gif",
            "render_squid_jet_style_gif",
            "run_fullmodel_real_preflight",
            "run_squid_latest_core",
            "surface_coupling",
            "test_device_immersed_boundary",
            "test_membrane",
            "test_mpm_shell",
            "test_surface_coupling",
        )
        search_roots = (
            Path("__pycache__"),
            Path("simulation_core/__pycache__"),
            Path("tests/__pycache__"),
            Path("squid_soft_robot_latest_core_20260603/__pycache__"),
            Path("squid_soft_robot_ecoflex0010_run_20260603/__pycache__"),
        )

        stale = [
            path
            for root in search_roots
            if root.exists()
            for path in root.glob("*.pyc")
            if any(path.name.startswith(stem + ".") for stem in forbidden_stems)
        ]

        self.assertEqual(stale, [])

    def test_no_legacy_preflight_or_runner_entrypoints(self) -> None:
        forbidden_entrypoints = (
            Path("run_fullmodel_real_preflight.py"),
            Path("squid_soft_robot_latest_core_20260603/run_squid_latest_core.py"),
            Path("squid_soft_robot_ecoflex0010_run_20260603/prepare_config.py"),
        )

        existing = [path for path in forbidden_entrypoints if path.exists()]

        self.assertEqual(existing, [])

    def test_case_specific_squid_scripts_do_not_live_at_repository_root(self) -> None:
        root_case_scripts = [
            path
            for path in Path(".").glob("*.py")
            if "squid" in path.stem.lower() and path.name != "run_simulation.py"
        ]

        self.assertEqual(root_case_scripts, [])

    def test_legacy_squid_run_directories_do_not_live_at_repository_root(self) -> None:
        forbidden_directories = (
            Path("squid_soft_robot_latest_core_20260603"),
            Path("squid_soft_robot_ecoflex0010_run_20260603"),
        )

        existing = [path for path in forbidden_directories if path.exists()]

        self.assertEqual(existing, [])

    def test_case_specific_squid_artifact_directories_do_not_live_at_repository_root(self) -> None:
        root_squid_directories = [
            path
            for path in Path(".").iterdir()
            if path.is_dir() and "squid" in path.name.lower()
        ]

        self.assertEqual(root_squid_directories, [])

    def test_core_usage_doc_has_no_deleted_module_or_case_rendering_paths(self) -> None:
        source = Path("SIMULATION_CORE_USAGE.md").read_text(encoding="utf-8")
        forbidden_tokens = (
            "device_immersed_boundary.py",
            "surface_coupling.py",
            "membrane.py",
            "mpm_shell.py",
            "simulation_core.surface_coupling",
            "simulation_core.device_immersed_boundary",
            "surface_pressure_pa",
            "pressure_force_scale",
            "Fixed Squid Jet Rendering",
        )

        for token in forbidden_tokens:
            self.assertNotIn(token, source)

    def test_core_usage_doc_describes_velocity_damping_as_nonconservative(self) -> None:
        source = Path("SIMULATION_CORE_USAGE.md").read_text(encoding="utf-8")

        self.assertIn("velocity_damping` defaults to `1.0`", source)
        self.assertIn("velocity_damping < 1", source)
        self.assertIn("non-conservative numerical damper", source)
        self.assertIn("transfer_relative_error", source)


if __name__ == "__main__":
    unittest.main()
