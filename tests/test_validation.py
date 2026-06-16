from __future__ import annotations

import math
import unittest

from simulation_core.validation import (
    ReferenceCurve,
    boundary_drive_compliance_report,
    checks_passed,
    finite_field_diagnostics,
    force_nonzero_when_loaded,
    vector_norm,
)


class ValidationHelperTests(unittest.TestCase):
    def test_reference_curve_interpolates_and_reports_relative_error(self) -> None:
        curve = ReferenceCurve(
            name="speed",
            units="m/s",
            source="unit test",
            points=((0.0, 0.0), (1.0, 2.0), (2.0, 4.0)),
        )

        self.assertEqual(curve.value_at(0.0), 0.0)
        self.assertEqual(curve.value_at(1.5), 3.0)
        self.assertAlmostEqual(
            curve.relative_error_at(time_s=2.0, computed_value=4.2),
            0.05,
        )

    def test_reference_curve_rejects_non_monotone_times(self) -> None:
        with self.assertRaises(ValueError):
            ReferenceCurve(
                name="bad",
                units="m",
                source="unit test",
                points=((0.0, 0.0), (0.0, 1.0)),
            )

    def test_vector_norm_uses_all_components(self) -> None:
        self.assertAlmostEqual(vector_norm((2.0, -3.0, 6.0)), 7.0)

    def test_finite_field_diagnostics_accepts_finite_numeric_values(self) -> None:
        rows = [
            {"step": 0, "velocity_mps": 0.0, "force_n": "1.25"},
            {"step": 1, "velocity_mps": -2.0, "force_n": 0.5},
        ]

        diagnostics = finite_field_diagnostics(rows, ("velocity_mps", "force_n"))

        self.assertEqual(diagnostics, [])

    def test_finite_field_diagnostics_reports_missing_non_numeric_and_nonfinite(self) -> None:
        rows = [
            {"step": 0, "velocity_mps": math.nan},
            {"step": 1, "velocity_mps": "bad", "force_n": math.inf},
            {"step": 2, "force_n": 1.0},
        ]

        diagnostics = finite_field_diagnostics(rows, ("velocity_mps", "force_n"))

        self.assertEqual(len(diagnostics), 5)
        self.assertEqual(
            {key: diagnostics[0][key] for key in ("step", "field", "reason")},
            {"step": 0, "field": "velocity_mps", "reason": "nonfinite"},
        )
        self.assertTrue(math.isnan(diagnostics[0]["value"]))
        self.assertEqual(
            diagnostics[1:],
            [
                {
                    "step": 0,
                    "field": "force_n",
                    "value": None,
                    "reason": "missing",
                },
                {
                    "step": 1,
                    "field": "velocity_mps",
                    "value": "bad",
                    "reason": "not_numeric",
                },
                {
                    "step": 1,
                    "field": "force_n",
                    "value": math.inf,
                    "reason": "nonfinite",
                },
                {
                    "step": 2,
                    "field": "velocity_mps",
                    "value": None,
                    "reason": "missing",
                },
            ],
        )

    def test_force_nonzero_when_loaded_requires_force_only_for_loaded_models(self) -> None:
        self.assertFalse(
            force_nonzero_when_loaded(
                force_components_n=(0.0, 0.0, 0.0),
                load_value=8_000.0,
                force_required=True,
            )
        )
        self.assertTrue(
            force_nonzero_when_loaded(
                force_components_n=(0.0, 0.0, -0.25),
                load_value=8_000.0,
                force_required=True,
            )
        )
        self.assertTrue(
            force_nonzero_when_loaded(
                force_components_n=(0.0, 0.0, 0.0),
                load_value=0.0,
                force_required=True,
            )
        )
        self.assertTrue(
            force_nonzero_when_loaded(
                force_components_n=(0.0, 0.0, 0.0),
                load_value=8_000.0,
                force_required=False,
            )
        )

    def test_force_nonzero_when_loaded_honors_tolerance(self) -> None:
        self.assertFalse(
            force_nonzero_when_loaded(
                force_components_n=(0.0, 0.0, 1.0e-7),
                load_value=1.0,
                force_required=True,
                tolerance_n=1.0e-6,
            )
        )

    def test_boundary_drive_compliance_rejects_prescribed_drives(self) -> None:
        self.assertEqual(
            boundary_drive_compliance_report(
                prescribed_velocity_boundary=False,
                prescribed_pressure_or_flow_boundary=False,
                nonzero_fluid_traction_scale=0.0,
            ),
            {
                "prescribed_velocity_boundary": False,
                "prescribed_pressure_or_flow_boundary": False,
                "nonzero_fluid_traction_scale": 0.0,
                "compliant": True,
            },
        )
        self.assertFalse(
            boundary_drive_compliance_report(
                prescribed_velocity_boundary=True,
                prescribed_pressure_or_flow_boundary=False,
                nonzero_fluid_traction_scale=0.0,
            )["compliant"]
        )
        self.assertFalse(
            boundary_drive_compliance_report(
                prescribed_velocity_boundary=False,
                prescribed_pressure_or_flow_boundary=True,
                nonzero_fluid_traction_scale=0.0,
            )["compliant"]
        )
        self.assertFalse(
            boundary_drive_compliance_report(
                prescribed_velocity_boundary=False,
                prescribed_pressure_or_flow_boundary=False,
                nonzero_fluid_traction_scale=1.0e-6,
            )["compliant"]
        )

    def test_checks_passed_requires_all_checks(self) -> None:
        self.assertTrue(checks_passed({"a": True, "b": 1}))
        self.assertFalse(checks_passed({"a": True, "b": 0}))
        self.assertTrue(
            force_nonzero_when_loaded(
                force_components_n=(0.0, 0.0, 2.0e-6),
                load_value=1.0,
                force_required=True,
                tolerance_n=1.0e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
