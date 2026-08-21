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
    "solver_net_velocity_damping_per_physical_step": 0.995,
}


def test_canonical_case_records_validated_direct_solver_damping() -> None:
    assert ANSYS_VERTICAL_FLAP_CASE_METADATA["structure_damping"] == (
        EXPECTED_DAMPING_IDENTITY
    )
    assert VerticalFlapFsiConfig().velocity_damping == 0.995
    assert selected_formulation_solver_config(step_count=50).velocity_damping == 0.995


def test_final_identity_records_native_and_solver_damping() -> None:
    assert final_contracts.FINAL_FINE_DAMPING_IDENTITY == EXPECTED_DAMPING_IDENTITY
    assert final_contracts.FINAL_FINE_CONFIG_IDENTITY["velocity_damping"] == 0.995

    identity = final_contracts.validate_final_run_identity(
        {"config": dict(final_contracts.FINAL_FINE_CONFIG_IDENTITY)},
        {"solver_npz_summary": dict(final_contracts.FINAL_FINE_EXPORT_IDENTITY)},
    )

    assert identity["schema"] == "our_solver_final_native_fine_identity_v3"
    assert identity["structure_damping"] == EXPECTED_DAMPING_IDENTITY


def test_final_identity_preserves_validated_tiny_unreached_cleanup() -> None:
    assert (
        final_contracts.FINAL_FINE_CONFIG_IDENTITY[
            "flow_hibm_tiny_unreached_cleanup_component_cells"
        ]
        == 128
    )
