from __future__ import annotations

from cases.ansys_vertical_flap_fsi import (
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    VerticalFlapFsiConfig,
)
from src.refactored.validation.ansys_vertical_flap_fsi import (
    native_fine_final_contracts,
)


def test_ansys_identity_locks_direct_partitioned_time_layers() -> None:
    assert (
        "flow_post_solid_kinematic_projection_enabled"
        not in VerticalFlapFsiConfig.__dataclass_fields__
    )
    assert ANSYS_VERTICAL_FLAP_CASE_METADATA["coupling_time_layer"] == {
        "scheme": "direct_explicit_partitioned",
        "physical_step_owner": (
            "benchmarks.official.solid_mpm_fsi_runner.run_hibm_mpm_fsi"
        ),
        "step_end_flow_stage": "pre_solid_projection",
        "step_end_structure_geometry_stage": "post_solid_observer",
        "transport_advanced_by_step_end_projection": False,
        "fail_closed_on_solver_health": True,
    }
    assert (
        "flow_post_solid_kinematic_projection_enabled"
        not in native_fine_final_contracts.FINAL_FINE_CONFIG_IDENTITY
    )
    assert native_fine_final_contracts.FINAL_FINE_TIME_LAYER_IDENTITY == {
        "scheme": "explicit_loose",
        "step_end_flow_stage": "pre_solid_projection",
        "step_end_structure_geometry_stage": "post_solid_observer",
        "transport_advanced_by_step_end_projection": False,
        "fluent_strong_coupling_equivalent": False,
    }
