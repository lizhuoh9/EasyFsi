from __future__ import annotations

from unittest.mock import patch

from cases import ansys_vertical_flap_fsi as vertical_flap


def _completed_report() -> dict[str, object]:
    return {
        "max_displacement_m": 0.0,
        "max_displacement_relative_error": 0.0,
    }


def test_no_arg_cli_uses_the_declared_smoke_preset() -> None:
    captured = []
    with patch.object(
        vertical_flap,
        "run_vertical_flap_fsi_smoke",
        side_effect=lambda config: captured.append(config) or _completed_report(),
    ):
        vertical_flap.main([])

    assert len(captured) == 1
    config = captured[0]
    assert config.flow_driver_mode == "projection_only"
    assert config.flow_advection_scheme == "euler"
    assert config.flow_turbulence_model == "laminar"
    assert config.flow_sst_near_wall_treatment == "resolved"


def test_case_metadata_locks_the_no_arg_cli_preset_identity() -> None:
    preset = vertical_flap.ANSYS_VERTICAL_FLAP_CASE_METADATA[
        "default_cli_preset"
    ]

    assert preset == {
        "name": "smoke",
        "flow_driver_mode": "projection_only",
        "flow_advection_scheme": "euler",
        "flow_turbulence_model": "laminar",
        "flow_sst_near_wall_treatment": "resolved",
    }
