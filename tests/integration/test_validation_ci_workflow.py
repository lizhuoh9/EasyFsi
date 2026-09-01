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
            "test_run_checkpoint_version_is_7",
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

    def test_fast_gate_has_cpu_only_r24_r24b_r24c_evidence_contracts(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("fetch-depth: 0", workflow)
        gate = workflow.split(
            "- name: Run CPU-only R24 through R24C evidence gates",
            1,
        )[1].split("- name: Run fast platform-independent contracts", 1)[0]

        for path in (
            "kalman_oracle_headroom_analysis.py",
            "kalman_oracle_headroom_artifacts.py",
            "kalman_oracle_headroom_verification.py",
            "oracle_threshold_common.py",
            "oracle_threshold_probe_contracts.py",
            "oracle_threshold_iqn_first_update.py",
            "oracle_threshold_displacement_evidence.py",
            "oracle_threshold_lineage.py",
            "oracle_threshold_prefix_decisions.py",
            "oracle_threshold_publication.py",
            "oracle_threshold_reuse_evidence.py",
            "oracle_threshold_evidence.py",
            "audit_ansys_vertical_flap_oracle_threshold.py",
            "r24c_post_publication.py",
            "r24c_post_publication_contracts.py",
            "seal_ansys_vertical_flap_r24c.py",
            "test_oracle_threshold_iqn_first_update.py",
            "test_oracle_threshold_lineage.py",
            "test_oracle_threshold_publication.py",
            "test_oracle_threshold_reuse_evidence.py",
            "test_r24c_post_publication.py",
            "test_our_solver_vertical_flap_runner.py",
            "test_ansys_vertical_flap_runner_loop_contract.py",
        ):
            with self.subTest(path=path):
                self.assertIn(path, gate)
        for test_path in (
            "tests/validation/test_kalman_oracle_headroom.py",
            "tests/validation/test_kalman_oracle_headroom_fail_closed.py",
            "tests/validation/test_kalman_statistical_calibration.py",
            "tests/validation/test_oracle_threshold_iqn_first_update.py",
            "tests/validation/test_oracle_threshold_lineage.py",
            "tests/validation/test_oracle_threshold_publication.py",
            "tests/validation/test_oracle_threshold_reuse_evidence.py",
            "tests/validation/test_our_solver_vertical_flap_runner.py",
        ):
            with self.subTest(test_path=test_path):
                self.assertIn(test_path, gate)
        for node in (
            "test_iqn_runner_maps_generic_threshold_audit_histories",
            "test_research_probe_recaptures_and_compares_after_each_rollback",
            "test_research_probe_rejects_iqn_history_reuse",
            "test_research_probe_terminal_satisfies_official_report_contract",
        ):
            with self.subTest(node=node):
                self.assertIn(node, gate)
        for cli in (
            "tools/audit_ansys_vertical_flap_kalman.py --help",
            "tools/audit_ansys_vertical_flap_oracle_headroom.py --help",
            "tools/audit_ansys_vertical_flap_oracle_threshold.py --help",
            "tools/seal_ansys_vertical_flap_r24c.py --help",
        ):
            with self.subTest(cli=cli):
                self.assertIn(cli, gate)
        pytest_gate = gate.split("python -m pytest -q", 1)[1]
        self.assertIn(
            "tests/validation/test_r24c_post_publication.py",
            pytest_gate,
        )
        self.assertIn('git diff --check "$base...HEAD"', gate)

    def test_windows_gate_compiles_and_runs_r24c_cpu_contracts(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        windows_gate = workflow.split("  contracts:", 1)[1].split(
            "  scheduled-cuda-step50:",
            1,
        )[0]
        for path in (
            "tools\\validation\\r24c_post_publication.py",
            "tools\\validation\\r24c_post_publication_contracts.py",
            "tools\\seal_ansys_vertical_flap_r24c.py",
            "tests\\validation\\test_r24c_post_publication.py",
        ):
            with self.subTest(path=path):
                self.assertIn(path, windows_gate)
        for node in (
            "tests.benchmarks.test_main_audit_particle_bin_generation_wiring.ParticleBinGenerationWiringTests.test_main_runner_owns_initial_generation_and_preflow_propagates_it",
            "tests.benchmarks.test_main_audit_particle_bin_generation_wiring.ParticleBinGenerationWiringTests.test_every_runner_particle_bin_consumer_receives_the_owned_generation",
            "tests.benchmarks.test_main_audit_particle_bin_generation_wiring.ParticleBinGenerationWiringTests.test_preflow_scatter_receives_current_generation_and_support_radius",
            "tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_full_domain_runner_persists_solid_substeps_in_process_updates",
            "tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_public_ansys_entrypoints_reject_removed_cell_obstacle_backend",
        ):
            with self.subTest(node=node):
                self.assertIn(node, windows_gate)
        self.assertIn(
            "python -m pytest -q tests\\validation\\test_r24c_post_publication.py",
            windows_gate,
        )

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
