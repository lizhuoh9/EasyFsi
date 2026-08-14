from __future__ import annotations

from cases.ansys_vertical_flap_fsi import (
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    VerticalFlapFsiConfig,
    selected_formulation_solver_config,
)
from src.refactored.validation.ansys_vertical_flap_fsi import (
    native_fine_final_contracts as final_contracts,
)


EXPECTED_DAMPING_IDENTITY = {
    "native_fluent_structure_damping_enabled": False,
    "solver_net_velocity_damping_per_physical_step": 1.0,
}


def test_canonical_case_matches_the_undamped_fluent_structure() -> None:
    assert ANSYS_VERTICAL_FLAP_CASE_METADATA["structure_damping"] == (
        EXPECTED_DAMPING_IDENTITY
    )
    assert VerticalFlapFsiConfig().velocity_damping == 1.0
    assert selected_formulation_solver_config(step_count=50).velocity_damping == 1.0


def test_final_identity_records_native_and_solver_damping() -> None:
    assert final_contracts.FINAL_FINE_DAMPING_IDENTITY == EXPECTED_DAMPING_IDENTITY
    assert final_contracts.FINAL_FINE_CONFIG_IDENTITY["velocity_damping"] == 1.0

    identity = final_contracts.validate_final_run_identity(
        {"config": dict(final_contracts.FINAL_FINE_CONFIG_IDENTITY)},
        {"solver_npz_summary": dict(final_contracts.FINAL_FINE_EXPORT_IDENTITY)},
    )

    assert identity["schema"] == "our_solver_final_native_fine_identity_v3"
    assert identity["structure_damping"] == EXPECTED_DAMPING_IDENTITY
