from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
GEOMETRY_PATH = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fixed_flow"
    / "preprocess"
    / "geometry_mask.npz"
)
BC_PATH = (
    ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fixed_flow"
    / "preprocess"
    / "bc_map.npz"
)
BASELINE_ROOT = ROOT / "validation_runs" / "ansys_vertical_flap_fixed_flow"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

REQUIRED_SOLVER_OUTPUTS = {
    "fields/final_fields_stabilized.npz",
    "logs/solver_history_stabilized.csv",
    "logs/mass_balance_stabilized.csv",
    "logs/poisson_history_stabilized.csv",
    "diagnostics/quality_comparison_step2_vs_stabilized.json",
    "diagnostics/initialization_sensitivity.csv",
    "case_manifest_step4_solver_stabilization.json",
}

REQUIRED_POSTPROCESS_OUTPUTS = {
    "speed_full_fluent_scale_0_28p1.png",
    "speed_full_autoscale.png",
    "streamwise_minus_Uz_fluent_scale_0_28p1.png",
    "streamwise_minus_Uz_autoscale.png",
    "Uy_full.png",
    "pressure_full.png",
    "geometry_overlay.png",
    "solver_history_plot.png",
    "mass_balance_plot.png",
    "centerline_streamwise_minus_Uz.csv",
    "throat_profile_streamwise_minus_Uz.csv",
    "downstream_profiles_streamwise_minus_Uz.csv",
    "validation_report.md",
    "case_manifest_step3.json",
}

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AnsysVerticalFlapFixedFlowStep4SolverStabilizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from refactored.validation.ansys_vertical_flap_fixed.projection_solver import (
            run_stabilized_projection_solver,
        )

        for path in (GEOMETRY_PATH, BC_PATH):
            if not path.exists():
                raise AssertionError(str(path))

        tmp_parent = ROOT / "tmp"
        tmp_parent.mkdir(exist_ok=True)
        cls.tmpdir = tempfile.TemporaryDirectory(
            prefix="ansys_fixed_flow_step4_", dir=tmp_parent
        )
        cls.solver_root = Path(cls.tmpdir.name) / "stabilized_solver"
        cls.postprocess_root = (
            Path(cls.tmpdir.name) / "rendered_results" / "step4_stabilized_fluent_style"
        )
        cls.result = run_stabilized_projection_solver(
            GEOMETRY_PATH,
            BC_PATH,
            cls.solver_root,
            baseline_root=BASELINE_ROOT,
            postprocess_root=cls.postprocess_root,
            config={
                "solver": {
                    "max_steps": 48,
                    "cfl": 0.20,
                    "steady_tolerance": 0.0,
                    "poisson_method": "sor",
                    "poisson_max_iters": 220,
                    "poisson_tolerance_abs": 1.0e-4,
                    "poisson_tolerance_rel": 1.0e-3,
                    "poisson_omega": 1.65,
                    "poisson_check_interval": 20,
                    "poisson_compatibility_correction": True,
                    "initialization_mode": "uniform",
                    "outlet_flux_correction": True,
                    "history_interval": 12,
                },
                "sensitivity": {
                    "max_steps": 18,
                    "poisson_max_iters": 120,
                },
            },
        )
        cls.final_fields_path = (
            cls.solver_root / "fields" / "final_fields_stabilized.npz"
        )
        cls.history_path = (
            cls.solver_root / "logs" / "solver_history_stabilized.csv"
        )
        cls.mass_path = (
            cls.solver_root / "logs" / "mass_balance_stabilized.csv"
        )
        cls.poisson_path = (
            cls.solver_root / "logs" / "poisson_history_stabilized.csv"
        )
        cls.comparison_path = (
            cls.solver_root
            / "diagnostics"
            / "quality_comparison_step2_vs_stabilized.json"
        )
        cls.sensitivity_path = (
            cls.solver_root / "diagnostics" / "initialization_sensitivity.csv"
        )
        cls.manifest_path = (
            cls.solver_root / "case_manifest_step4_solver_stabilization.json"
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_poisson_sor_reduces_relative_residual_on_masked_domain(self):
        from refactored.validation.ansys_vertical_flap_fixed.poisson import (
            solve_pressure_poisson_sor,
        )

        rows, cols = 20, 28
        fluid = np.ones((rows, cols), dtype=bool)
        solid = np.zeros_like(fluid)
        fluid[7:13, 11:14] = False
        solid[7:13, 11:14] = True
        outlet = np.zeros_like(fluid)
        outlet[:, -1] = fluid[:, -1]
        active = fluid & ~outlet

        rhs = np.zeros((rows, cols), dtype=np.float64)
        rhs[5, 6] = 8.0
        rhs[15, 6] = -8.0
        rhs[10, 20] = 4.0
        rhs[3, 20] = -4.0
        rhs[~active] = 0.0

        pressure, info = solve_pressure_poisson_sor(
            rhs,
            np.zeros_like(rhs),
            fluid,
            solid,
            outlet,
            0.0,
            1.0,
            1.0,
            max_iters=900,
            tolerance_abs=1.0e-9,
            tolerance_rel=1.0e-3,
            omega=1.65,
            compatibility_correction=True,
            check_interval=25,
        )

        self.assertEqual(pressure.shape, rhs.shape)
        self.assertTrue(info["converged"], msg=info)
        self.assertLessEqual(info["poisson_iters"], 900)
        self.assertLess(info["poisson_residual_linf_relative"], 1.0e-3)
        self.assertGreater(info["rhs_linf"], 0.0)
        self.assertEqual(info["method"], "masked_sor")

    def test_projection_reduces_divergence_and_preserves_boundaries(self):
        with np.load(self.final_fields_path) as fields:
            fluid = fields["fluid_mask"].astype(bool)
            solid = fields["solid_mask"].astype(bool)
            inlet = fields["inlet_mask"].astype(bool)
            u = fields["u"]
            v = fields["v"]

            self.assertFalse(np.isnan(u[fluid]).any())
            self.assertFalse(np.isinf(u[fluid]).any())
            self.assertFalse(np.isnan(v[fluid]).any())
            self.assertFalse(np.isinf(v[fluid]).any())
            np.testing.assert_allclose(u[solid], 0.0)
            np.testing.assert_allclose(v[solid], 0.0)
            np.testing.assert_allclose(u[inlet], 7.0)
            np.testing.assert_allclose(v[inlet], 0.0)

        comparison = json.loads(self.comparison_path.read_text(encoding="utf-8"))
        self.assertGreater(
            comparison["stabilized"]["initial_divergence_l2_excluding_near_solid"],
            comparison["stabilized"]["divergence_l2_excluding_near_solid"],
        )
        self.assertIn("divergence_linf", comparison["stabilized"])
        self.assertIn("divergence_l2_excluding_near_solid", comparison["stabilized"])

    def test_uniform_initialization_sensitivity_is_recorded(self):
        rows = _read_csv(self.sensitivity_path)
        modes = {row["initialization_mode"]: row for row in rows}
        self.assertIn("uniform", modes)
        self.assertIn("structured_jet", modes)

        uniform = modes["uniform"]
        structured = modes["structured_jet"]
        self.assertLess(float(uniform["initial_centerline_max_u"]), 8.0)
        self.assertGreater(float(uniform["final_centerline_max_u"]), 7.5)
        self.assertGreater(float(structured["final_centerline_max_u"]), 7.5)
        self.assertEqual(uniform["fluent_parity"], "not_claimed")
        self.assertEqual(structured["fluent_parity"], "not_claimed")

    def test_required_stabilized_solver_and_postprocess_artifacts_exist(self):
        for relative in REQUIRED_SOLVER_OUTPUTS:
            path = self.solver_root / relative
            self.assertTrue(path.exists(), msg=str(path))

        for relative in REQUIRED_POSTPROCESS_OUTPUTS:
            path = self.postprocess_root / relative
            self.assertTrue(path.exists(), msg=str(path))
            if relative.endswith(".png"):
                self.assertEqual(path.read_bytes()[:8], PNG_MAGIC, msg=relative)
                self.assertGreater(path.stat().st_size, 1024, msg=relative)

        for path in (self.history_path, self.mass_path, self.poisson_path):
            rows = _read_csv(path)
            self.assertGreaterEqual(len(rows), 2, msg=str(path))

    def test_stabilized_quality_improves_without_parity_claims(self):
        comparison = json.loads(self.comparison_path.read_text(encoding="utf-8"))
        stabilized = comparison["stabilized"]
        self.assertLess(abs(stabilized["mass_imbalance_rel_corrected"]), 0.02)
        self.assertLess(stabilized["poisson_residual_linf_relative"], 1.0e-3)
        self.assertLessEqual(
            stabilized["divergence_l2_excluding_near_solid"],
            comparison["baseline_step2"]["divergence_l2"],
        )

        report = (self.postprocess_root / "validation_report.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("No Fluent parity claim", report)
        self.assertIn("No FSI claim", report)
        self.assertIn("traction_shared_snapshot_diagnostics not used", report)
        self.assertNotIn("parity achieved", report.lower())

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["claims"]["fluent_parity"], "not_claimed")
        self.assertEqual(manifest["claims"]["fsi"], "not_claimed")
        self.assertEqual(
            manifest["forbidden_sources"]["traction_shared_snapshot_diagnostics"],
            "not_used",
        )
        self.assertEqual(manifest["solver_config"]["initialization_mode"], "uniform")
        self.assertIn(
            manifest["quality"]["overall_status"],
            {"diagnostic_only_not_parity", "candidate_not_parity"},
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
