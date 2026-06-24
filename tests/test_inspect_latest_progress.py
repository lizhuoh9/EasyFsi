from __future__ import annotations

import unittest

from tools.diagnostics.inspect_latest_progress import _compact_payload


class InspectLatestProgressTests(unittest.TestCase):
    def test_compact_payload_supports_flat_latest_runner_fields(self) -> None:
        payload = _compact_payload(
            {
                "step": 2,
                "total_steps": 4,
                "percent": 50.0,
                "simulation": {
                    "time_s": 0.001,
                    "projected_ibm_residual_mps": 0.0,
                    "max_fluid_speed_mps": 0.03,
                    "divergence_l2": 1.0e-4,
                    "main_displacement_z_m": -1.0e-5,
                    "tail_displacement_z_m": 2.0e-6,
                    "lip_flow_negative_z_m3s": 1.0e-8,
                    "outlet_flow_negative_z_m3s": 2.0e-8,
                    "downstream_flow_negative_z_m3s": 3.0e-8,
                    "lip_sample_count": 8,
                    "outlet_sample_count": 9,
                    "downstream_sample_count": 10,
                },
            }
        )

        self.assertIn("physical_monitor_summary not present", payload["_warnings"][0])
        self.assertEqual(payload["membrane"]["main"], -1.0e-5)
        self.assertEqual(payload["membrane"]["tail"], 2.0e-6)
        self.assertEqual(payload["jet_sections"]["lip"]["flow_rate_m3ps"], 1.0e-8)
        self.assertEqual(payload["jet_sections"]["outlet"]["sample_count"], 9)
        self.assertEqual(payload["jet_sections"]["downstream"]["flow_rate_m3ps"], 3.0e-8)

    def test_compact_payload_ignores_old_fsi_velocity_residual_key(self) -> None:
        payload = _compact_payload(
            {
                "simulation": {
                    "time_s": 0.001,
                    "fsi_velocity_residual_mps": 0.123,
                    "max_fluid_speed_mps": 0.03,
                    "divergence_l2": 1.0e-4,
                    "lip_flow_negative_z_m3s": 1.0e-8,
                    "outlet_flow_negative_z_m3s": 2.0e-8,
                    "downstream_flow_negative_z_m3s": 3.0e-8,
                },
            }
        )

        warnings = "\n".join(payload["_warnings"])
        self.assertIn("missing core progress keys", warnings)
        self.assertIsNone(payload["residual_mps"])

    def test_compact_payload_supports_summary_final_schema(self) -> None:
        payload = _compact_payload(
            {
                "validation_passed": False,
                "final": {
                    "step": 3,
                    "time_s": 0.0015,
                    "max_speed_mps": 0.04,
                    "divergence_l2": 2.0e-5,
                    "main_displacement_z_m": -2.0e-5,
                    "tail_displacement_z_m": 4.0e-6,
                    "projected_ibm_residual_mps": 1.0e-4,
                    "projected_ibm_residual_l2_mps": 5.0e-5,
                    "projected_ibm_sample_count": 12,
                    "fsi_probe_valid_fraction": 0.75,
                    "lip_flow_negative_z_m3s": 1.0e-8,
                    "outlet_flow_negative_z_m3s": 2.0e-8,
                    "downstream_flow_negative_z_m3s": 3.0e-8,
                    "lip_sample_count": 8,
                    "outlet_sample_count": 9,
                    "downstream_sample_count": 10,
                },
            }
        )

        self.assertIn("summary final record detected", payload["_warnings"][1])
        self.assertEqual(payload["step"], 3)
        self.assertEqual(payload["residual_mps"], 1.0e-4)
        self.assertEqual(payload["max_fluid_speed_mps"], 0.04)
        self.assertEqual(payload["projected_ibm"]["sample_count"], 12)
        self.assertEqual(payload["projected_ibm"]["probe_valid_fraction"], 0.75)
        self.assertEqual(payload["jet_sections"]["outlet"]["flow_rate_m3ps"], 2.0e-8)

    def test_compact_payload_warns_when_current_flat_schema_keys_are_missing(self) -> None:
        payload = _compact_payload({"simulation": {"time_s": 0.002}})

        warnings = "\n".join(payload["_warnings"])
        self.assertIn("physical_monitor_summary not present", warnings)
        self.assertIn("missing flat section keys", warnings)
        self.assertIn("missing core progress keys", warnings)
        self.assertIsNone(payload["residual_mps"])
        self.assertIsNone(payload["max_fluid_speed_mps"])
        self.assertIsNone(payload["jet_sections"]["outlet"]["flow_rate_m3ps"])

    def test_compact_payload_names_nonfinite_progress_fields_and_suppresses_values(self) -> None:
        payload = _compact_payload(
            {
                "simulation": {
                    "time_s": 0.001,
                    "projected_ibm_residual_mps": float("nan"),
                    "max_fluid_speed_mps": float("inf"),
                    "divergence_l2": 1.0e-4,
                    "main_displacement_z_m": float("-inf"),
                    "tail_displacement_z_m": 0.0,
                    "lip_flow_negative_z_m3s": 1.0e-8,
                    "outlet_flow_negative_z_m3s": float("nan"),
                    "downstream_flow_negative_z_m3s": 3.0e-8,
                    "lip_sample_count": 8,
                    "outlet_sample_count": 9,
                    "downstream_sample_count": 10,
                },
            }
        )

        warnings = "\n".join(payload["_warnings"])
        self.assertIn("non-finite progress fields", warnings)
        self.assertIn("residual_mps", warnings)
        self.assertIn("max_fluid_speed_mps", warnings)
        self.assertIn("membrane.main", warnings)
        self.assertIn("jet_sections.outlet.flow_rate_m3ps", warnings)
        self.assertIsNone(payload["residual_mps"])
        self.assertIsNone(payload["max_fluid_speed_mps"])
        self.assertIsNone(payload["membrane"]["main"])
        self.assertIsNone(payload["jet_sections"]["outlet"]["flow_rate_m3ps"])


if __name__ == "__main__":
    unittest.main()
