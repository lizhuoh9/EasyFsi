from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
STEP2_ROOT = ROOT / "validation_runs" / "ansys_vertical_flap_fixed_flow"
FINAL_FIELDS = STEP2_ROOT / "fields" / "final_fields.npz"
SOLVER_HISTORY = STEP2_ROOT / "logs" / "solver_history.csv"
MASS_BALANCE = STEP2_ROOT / "logs" / "mass_balance.csv"
STEP2_MANIFEST = STEP2_ROOT / "case_manifest_step2.json"
DEFAULT_STEP3_ROOT = STEP2_ROOT / "rendered_results" / "step3_fluent_style"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

REQUIRED_OUTPUTS = {
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

REQUIRED_PROFILE_COLUMNS = {
    "s",
    "y",
    "u",
    "Uz",
    "Uy",
    "speed",
    "fluid_mask",
    "near_solid_mask",
}

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AnsysVerticalFlapFixedFlowStep3PostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from refactored.validation.ansys_vertical_flap_fixed.postprocess_fluent_style import (
            run_fluent_style_postprocess,
        )

        for path in (FINAL_FIELDS, SOLVER_HISTORY, MASS_BALANCE, STEP2_MANIFEST):
            if not path.exists():
                raise AssertionError(str(path))

        tmp_parent = ROOT / "tmp"
        tmp_parent.mkdir(exist_ok=True)
        cls.tmpdir = tempfile.TemporaryDirectory(
            prefix="ansys_fixed_flow_step3_", dir=tmp_parent
        )
        cls.output_root = Path(cls.tmpdir.name) / "step3_fluent_style"
        cls.result = run_fluent_style_postprocess(
            FINAL_FIELDS,
            SOLVER_HISTORY,
            MASS_BALANCE,
            STEP2_MANIFEST,
            cls.output_root,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_postprocessor_consumes_step2_artifacts(self):
        self.assertTrue(FINAL_FIELDS.exists(), msg=str(FINAL_FIELDS))
        self.assertTrue(SOLVER_HISTORY.exists(), msg=str(SOLVER_HISTORY))
        self.assertTrue(MASS_BALANCE.exists(), msg=str(MASS_BALANCE))
        self.assertTrue(STEP2_MANIFEST.exists(), msg=str(STEP2_MANIFEST))
        self.assertEqual(self.result["claims"]["fluent_parity"], "not_claimed")
        self.assertEqual(self.result["claims"]["fsi"], "not_claimed")
        self.assertIn("quality", self.result)

    def test_required_plots_profiles_report_and_manifest_are_generated(self):
        for filename in REQUIRED_OUTPUTS:
            path = self.output_root / filename
            self.assertTrue(path.exists(), msg=str(path))

        for filename in REQUIRED_OUTPUTS:
            if filename.endswith(".png"):
                path = self.output_root / filename
                self.assertEqual(path.read_bytes()[:8], PNG_MAGIC, msg=filename)
                self.assertGreater(path.stat().st_size, 1024, msg=filename)

        for filename in (
            "centerline_streamwise_minus_Uz.csv",
            "throat_profile_streamwise_minus_Uz.csv",
            "downstream_profiles_streamwise_minus_Uz.csv",
        ):
            rows = _read_csv(self.output_root / filename)
            self.assertGreaterEqual(len(rows), 10, msg=filename)
            self.assertTrue(REQUIRED_PROFILE_COLUMNS.issubset(rows[0]), msg=filename)

    def test_report_contains_honest_quality_gates_and_non_parity_claims(self):
        report = (self.output_root / "validation_report.md").read_text(
            encoding="utf-8"
        )
        for text in (
            "No Fluent parity claim",
            "No FSI claim",
            "traction_shared_snapshot_diagnostics not used",
            "Step 2 solver output",
            "poisson_residual_linf",
            "divergence_linf",
            "mass_imbalance_rel",
            "diagnostic_only_not_parity",
        ):
            self.assertIn(text, report)
        self.assertNotIn("parity achieved", report.lower())
        self.assertNotIn("validated against fluent", report.lower())

    def test_step3_manifest_is_strict_and_does_not_hide_solver_residuals(self):
        manifest = json.loads(
            (self.output_root / "case_manifest_step3.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["claims"]["fluent_parity"], "not_claimed")
        self.assertEqual(manifest["claims"]["fsi"], "not_claimed")
        self.assertEqual(
            manifest["sources"]["final_fields"],
            "validation_runs/ansys_vertical_flap_fixed_flow/fields/final_fields.npz",
        )
        self.assertEqual(
            manifest["sources"]["solver_history"],
            "validation_runs/ansys_vertical_flap_fixed_flow/logs/solver_history.csv",
        )
        self.assertEqual(
            manifest["sources"]["mass_balance"],
            "validation_runs/ansys_vertical_flap_fixed_flow/logs/mass_balance.csv",
        )
        self.assertEqual(
            manifest["forbidden_sources"]["traction_shared_snapshot_diagnostics"],
            "not_used",
        )
        self.assertIn(
            manifest["quality"]["overall_status"],
            {"diagnostic_only_not_parity", "candidate_not_parity"},
        )
        if (
            manifest["quality"]["metrics"]["poisson_residual_linf"]
            > manifest["quality"]["thresholds"]["max_poisson_residual_linf_warn"]
        ):
            self.assertNotEqual(
                manifest["quality"]["incompressibility_quality"]["status"],
                "pass",
            )

    def test_default_runner_artifact_bundle_exists(self):
        for filename in REQUIRED_OUTPUTS:
            path = DEFAULT_STEP3_ROOT / filename
            self.assertTrue(path.exists(), msg=str(path))

        manifest = json.loads(
            (DEFAULT_STEP3_ROOT / "case_manifest_step3.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["generated_files"]["validation_report"],
            "validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/validation_report.md",
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
