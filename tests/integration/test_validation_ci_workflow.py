from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ansys-vertical-flap-validation.yml"
REQUIREMENTS = ROOT / "requirements.txt"


class ValidationCiWorkflowTests(unittest.TestCase):
    def test_ci_has_fast_linux_full_windows_and_scheduled_cuda_layers(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("schedule:", workflow)
        self.assertIn("quality-and-fast-contracts:", workflow)
        self.assertIn("runs-on: ubuntu-latest", workflow)
        self.assertIn("python -m pip install -r requirements.txt", workflow)
        self.assertIn("python -m ruff check", workflow)
        self.assertIn("contracts:", workflow)
        self.assertIn("runs-on: windows-latest", workflow)
        self.assertIn("scheduled-cuda-step50:", workflow)
        self.assertIn("[self-hosted, windows, x64, cuda]", workflow)
        self.assertIn("github.event_name == 'schedule'", workflow)
        self.assertIn("run_traction_selected_formulation_coupled_step50.py", workflow)

    def test_fast_gate_runs_new_cpu_only_main_audit_regressions(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        fast_gate = workflow.split(
            "- name: Run fast platform-independent contracts",
            1,
        )[1].split("\n  contracts:", 1)[0]

        for module in (
            "tests.cases.test_squid_import_contracts",
            "tests.cases.test_squid_checkpoint_atomicity",
            "tests.cases.test_squid_explicit_context_contract",
            "tests.cases.test_squid_sharp_contract_fixes",
            "tests.benchmarks.test_main_audit_particle_bin_generation_wiring",
            "tests.solvers.test_main_audit_graded_grid_performance",
            "tests.solvers.test_main_audit_numeric_contracts",
            "tests.integration.test_ansys_vertical_flap_step50_cli",
        ):
                with self.subTest(module=module):
                    self.assertIn(module, fast_gate)

        for test_name in (
            "test_step_part_entity_tag_expands_all_part_surfaces_on_tag_collision",
            "test_sharp_mpm_step_requires_fresh_external_force_before_solid_advance",
            "test_f32_kernel_parameters_reject_f64_overflow_before_launch",
            "test_run_checkpoint_version_is_4",
            "test_full_domain_runner_passes_full_step_neumann_dt_once",
            "test_full_domain_runner_uses_full_span_flaps",
            "test_canonical_runner_rejects_underresolved_solid_particles_for_fine_grid",
            "test_full_domain_runner_uses_local_surface_force_support_radius",
            "test_full_domain_runner_persists_solid_substeps_in_process_updates",
            "test_full_domain_runner_has_official_style_stationary_preflow_option",
            "test_two_synthetic_snapshots_render_with_one_fixed_scale_and_make_gif",
        ):
            with self.subTest(test_name=test_name):
                self.assertIn(test_name, fast_gate)

    def test_ci_tools_and_direct_render_dependency_are_locked(self) -> None:
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower()

        self.assertRegex(requirements, r"(?m)^pillow==\d")
        self.assertRegex(requirements, r"(?m)^ruff==\d")
        self.assertRegex(requirements, r"(?m)^scipy==1\.15\.3$")
        self.assertNotRegex(requirements, r"(?m)^scipy==1\.16\.")

    def test_scheduled_cuda_gate_runs_main_audit_regressions(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        cuda_gate = workflow.split("  scheduled-cuda-step50:", 1)[1].split(
            "- name: Run staged CUDA step-50 validation",
            1,
        )[0]

        for test_name in (
            "tests.solvers.test_main_audit_coupling_contracts",
            "tests.solvers.test_main_audit_fluid_state_contracts",
            "tests.solvers.test_main_audit_solid_input_contracts",
            "tests.solvers.test_mooney_shell_post_step_oob_cuda",
            "test_report_only_active_force_cells_match_final_cancelled_force_field",
        ):
            with self.subTest(test_name=test_name):
                self.assertIn(test_name, cuda_gate)


if __name__ == "__main__":
    unittest.main()
