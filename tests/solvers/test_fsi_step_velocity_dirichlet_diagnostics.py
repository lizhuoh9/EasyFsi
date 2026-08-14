from __future__ import annotations

import ast
import inspect
import textwrap

from benchmarks.official import solid_mpm_fsi_runner


def _velocity_dirichlet_report(seed: int) -> dict[str, object]:
    return {
        "hibm_velocity_dirichlet_authority": "canonical",
        "hibm_velocity_dirichlet_ledger_generation": seed,
        "hibm_velocity_dirichlet_authority_registered": True,
        "hibm_velocity_dirichlet_authority_sealed": True,
        "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count": seed,
        "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count": seed,
        "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio": 0.25,
        "canonical_velocity_dirichlet_report": {"schema_version": 5},
    }


def test_velocity_dirichlet_mapping_can_qualify_post_solid_observer_stage() -> None:
    observer_report = _velocity_dirichlet_report(23)

    fields = solid_mpm_fsi_runner._hibm_velocity_dirichlet_mapping_fields(
        observer_report,
        stage="observer",
    )

    expected = {
        (
            "hibm_observer_canonical_velocity_dirichlet_report"
            if key == "canonical_velocity_dirichlet_report"
            else key.replace(
                "hibm_velocity_dirichlet_",
                "hibm_observer_velocity_dirichlet_",
                1,
            )
        ): value
        for key, value in observer_report.items()
    }
    assert fields == expected
    assert not (set(fields) & set(observer_report))


def test_fsi_step_history_keeps_flow_and_observer_stage_diagnostics() -> None:
    """The saved history must describe the same post-solid state as its snapshot."""

    function_source = textwrap.dedent(
            inspect.getsource(
                solid_mpm_fsi_runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
            )
    )
    tree = ast.parse(function_source)
    mapping_calls: set[tuple[str, str | None]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "_hibm_velocity_dirichlet_mapping_fields"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            continue
        stage = None
        for keyword in node.keywords:
            if (
                keyword.arg == "stage"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                stage = keyword.value.value
        mapping_calls.add((node.args[0].id, stage))

    assert ("latest_flow_report", None) in mapping_calls
    assert ("latest_observer_topology_report", "observer") in mapping_calls
