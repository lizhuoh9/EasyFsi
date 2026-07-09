from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refactored.validation.ansys_vertical_flap_fixed.quality_gates import (
    evaluate_quality_gates,
)


class EvaluateQualityGatesMissingEvidenceTests(unittest.TestCase):
    def test_completely_empty_evidence_fails_mass_and_incompressibility_gates(
        self,
    ) -> None:
        # Audit probe: with history_rows=[] and mass_rows=[], every
        # `.get(key, 0.0)` default in the old code reads as "0 divergence, 0
        # residual, 0 mass imbalance" -- i.e. perfect -- so mass_quality and
        # incompressibility_quality used to report "pass" despite there
        # being zero solver evidence behind them.
        quality = evaluate_quality_gates([], [], {})

        self.assertEqual(quality["mass_quality"]["status"], "fail")
        self.assertIn("missing metric: mass_imbalance_rel", quality["mass_quality"]["reason"])
        self.assertEqual(quality["incompressibility_quality"]["status"], "fail")
        self.assertIn("missing metric:", quality["incompressibility_quality"]["reason"])
        self.assertIn("mass_imbalance_rel", quality["missing_metrics"])
        self.assertIn("divergence_l2", quality["missing_metrics"])
        self.assertIn("divergence_linf", quality["missing_metrics"])
        self.assertIn("poisson_residual_linf", quality["missing_metrics"])
        self.assertIn("poisson_residual_linf_relative", quality["missing_metrics"])
        self.assertEqual(quality["overall_status"], "diagnostic_only_not_parity")

    def test_visual_evidence_without_mass_or_divergence_history_is_not_a_candidate(
        self,
    ) -> None:
        # Sharper reproduction: final_summary carries plausible-looking
        # velocity numbers (so visual_candidate legitimately passes) but
        # history_rows/mass_rows -- the only source of mass/divergence/
        # residual evidence -- are empty. The old code still let this reach
        # "candidate_not_parity" purely because the missing metrics defaulted
        # to values that satisfied their gates.
        final_summary = {
            "max_u": 20.0,
            "centerline_max_u": 20.0,
            "p99_speed": 20.0,
        }

        quality = evaluate_quality_gates([], [], final_summary)

        self.assertEqual(quality["visual_candidate"]["status"], "pass")
        self.assertEqual(quality["mass_quality"]["status"], "fail")
        self.assertEqual(quality["incompressibility_quality"]["status"], "fail")
        self.assertNotEqual(quality["overall_status"], "candidate_not_parity")
        self.assertEqual(quality["overall_status"], "diagnostic_only_not_parity")

    def test_legitimate_full_evidence_still_passes_as_before(self) -> None:
        history_rows = [
            {
                "max_u": 20.0,
                "max_speed": 20.0,
                "p99_speed": 18.0,
                "divergence_l2": 1.0,
                "divergence_linf": 5.0,
                "divergence_l2_excluding_near_solid": 0.5,
                "divergence_linf_excluding_near_solid": 2.0,
                "poisson_residual_linf": 1.0e-6,
                "poisson_residual_linf_relative": 1.0e-6,
            }
        ]
        mass_rows = [{"mass_imbalance_rel": 0.001}]
        final_summary = {"centerline_max_u": 15.0}

        quality = evaluate_quality_gates(history_rows, mass_rows, final_summary)

        self.assertEqual(quality["missing_metrics"], [])
        self.assertEqual(quality["visual_candidate"]["status"], "pass")
        self.assertEqual(quality["mass_quality"]["status"], "pass")
        self.assertEqual(quality["incompressibility_quality"]["status"], "pass")
        self.assertEqual(quality["overall_status"], "candidate_not_parity")

    def test_legitimate_poor_evidence_still_fails_as_before(self) -> None:
        # Genuinely bad (not missing) divergence/mass data must keep the old
        # "warn"/"fail" tiers driven by the actual numbers, not the new
        # missing-metric escalation path.
        history_rows = [
            {
                "max_u": 20.0,
                "max_speed": 20.0,
                "p99_speed": 18.0,
                "divergence_l2": 5000.0,
                "divergence_linf": 50000.0,
                "poisson_residual_linf": 1.0e9,
                "poisson_residual_linf_relative": 0.5,
            }
        ]
        mass_rows = [{"mass_imbalance_rel": 0.5}]
        final_summary = {"centerline_max_u": 15.0}

        quality = evaluate_quality_gates(history_rows, mass_rows, final_summary)

        self.assertEqual(quality["missing_metrics"], [])
        self.assertEqual(quality["mass_quality"]["status"], "fail")
        self.assertNotIn("missing metric", quality["mass_quality"]["reason"])
        self.assertEqual(quality["incompressibility_quality"]["status"], "warn")
        self.assertNotIn("missing metric", quality["incompressibility_quality"]["reason"])

    def test_missing_absolute_poisson_residual_does_not_force_fail_when_relative_passes(
        self,
    ) -> None:
        # Partial evidence: poisson_residual_linf is absent but the relative
        # residual is present and legitimately converged, and divergence is
        # fine too. The OR semantics of poisson_pass must still allow a pass
        # -- missing one of two alternative metrics should not punish a
        # gate that has sufficient alternate evidence to pass on its own.
        history_rows = [
            {
                "max_u": 20.0,
                "max_speed": 20.0,
                "p99_speed": 18.0,
                "divergence_l2": 1.0,
                "divergence_linf": 5.0,
                "poisson_residual_linf_relative": 1.0e-6,
            }
        ]
        mass_rows = [{"mass_imbalance_rel": 0.001}]
        final_summary = {"centerline_max_u": 15.0}

        quality = evaluate_quality_gates(history_rows, mass_rows, final_summary)

        self.assertIn("poisson_residual_linf", quality["missing_metrics"])
        self.assertEqual(quality["incompressibility_quality"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
