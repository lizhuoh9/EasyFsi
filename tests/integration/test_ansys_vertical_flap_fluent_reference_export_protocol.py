from __future__ import annotations

import unittest
from pathlib import Path


PROTOCOL = (
    Path("docs")
    / "validation"
    / "ANSYS_VERTICAL_FLAP_FLUENT_REFERENCE_EXPORT_PROTOCOL_2026-06-29.md"
)


class AnsysVerticalFlapFluentReferenceExportProtocolTests(unittest.TestCase):
    def test_protocol_locks_required_scope_and_exports(self):
        text = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn("does not run EasyFsi", text)
        self.assertIn("does not run HIBM-MPM", text)
        self.assertIn("does not claim Fluent parity", text)
        self.assertIn("metadata evidence", text)
        self.assertIn("must not be used as numeric parity truth", text)
        self.assertIn("Step count: `50`", text)
        self.assertIn("Time step: `0.0005`", text)
        self.assertIn("Final time: `0.025`", text)
        self.assertIn("ansys_vertical_flap_fluent_reference_contract_v1", text)
        self.assertIn("active_fluent_reference_contract_manifest_v1", text)

    def test_protocol_lists_all_source_export_headers(self):
        text = PROTOCOL.read_text(encoding="utf-8")

        expected = {
            "fluent_tip_displacement_history.csv": (
                "step,time_s,tip_displacement_x_m,tip_displacement_y_m,"
                "tip_displacement_z_m,tip_displacement_norm_m,"
                "max_displacement_m,source"
            ),
            "fluent_force_history.csv": (
                "step,time_s,force_x_N,force_y_N,force_z_N,"
                "primary_force_z_N,secondary_force_z_N,source"
            ),
            "fluent_flow_balance_history.csv": (
                "step,time_s,inlet_flow_rate_m3s,outlet_flow_rate_m3s,"
                "pressure_outlet_flux_m3s,velocity_outlet_flux_m3s,source"
            ),
            "fluent_pressure_summary_history.csv": (
                "step,time_s,pressure_min_pa,pressure_max_pa,"
                "pressure_range_pa,source"
            ),
            "fluent_metadata_2026-06-28.md": "displacement definition",
        }
        for filename, required_text in expected.items():
            self.assertIn(filename, text)
            self.assertIn(required_text, text)


if __name__ == "__main__":
    unittest.main()
