from types import SimpleNamespace

import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner


class _DefaultingReport(dict[str, object]):
    def __missing__(self, key: str) -> object:
        return 0


class _FakeSolid:
    particle_count = 1
    external_force_n = object()
    x = object()


class _FakeMarkers:
    def aggregate_region_forces(self, **_kwargs):
        return SimpleNamespace(total_marker_force_n=(0.0, 0.0, 0.0))

    def clear_mpm_external_forces(self, *_args, **_kwargs):
        return None

    def scatter_marker_forces_to_mpm_particles(self, *_args, **_kwargs):
        return SimpleNamespace(
            total_mpm_external_force_n=(0.0, 0.0, 0.0),
            invalid_marker_count=0,
            active_marker_count=0,
            active_pair_count=0,
        )

    def stress_marker_diagnostics(self):
        return []

    def stress_face_diagnostics(self, **_kwargs):
        return {}


_REPORT_FIELD_HELPERS = (
    "_hibm_velocity_dirichlet_mapping_fields",
    "_flow_projection_report_fields",
    "_flow_source_report_fields",
    "_flow_transport_report_fields",
    "_marker_projection_boundary_report_fields",
    "_marker_force_report_fields",
    "_stress_sampling_report_fields",
    "_marker_traction_report_fields",
    "_scatter_report_fields",
)


@pytest.mark.parametrize(
    ("detailed", "expected_phases"),
    (
        (False, ["preflow_step", "preflow_step"]),
        (True, ["preflow_step", "preflow_stage", "preflow_step"]),
    ),
)
def test_fixed_preflow_stage_progress_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    detailed: bool,
    expected_phases: list[str],
) -> None:
    events: list[dict[str, object]] = []
    forwarded_stage_observers = []

    def flow_advance(*_args, preflow_stage_observer=None, **_kwargs):
        forwarded_stage_observers.append(preflow_stage_observer)
        if preflow_stage_observer is not None:
            preflow_stage_observer("behavior_probe")
        return _DefaultingReport()

    monkeypatch.setattr(runner, "_flow_advance_current_step", flow_advance)
    monkeypatch.setattr(
        runner,
        "_apply_marker_feedback_to_fluid",
        lambda *_args, **_kwargs: _DefaultingReport(),
    )
    monkeypatch.setattr(
        runner,
        "_use_hibm_sharp_marker_boundary",
        lambda _config: True,
    )
    monkeypatch.setattr(
        runner,
        "_sample_stress_to_marker_forces",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid_marker_count=0,
            invalid_marker_count=0,
            two_sided_pressure_marker_count=0,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_preflow_traction_readiness",
        lambda *_args, **_kwargs: runner.PREFLOW_TRACTION_EVALUATED,
    )
    monkeypatch.setattr(runner, "_marker_total_area_m2", lambda _markers: 0.0)
    for helper_name in _REPORT_FIELD_HELPERS:
        monkeypatch.setattr(
            runner,
            helper_name,
            lambda *_args, **_kwargs: {},
        )

    report = runner._run_fixed_solid_preflow(
        markers=_FakeMarkers(),
        fluid=object(),
        solid=_FakeSolid(),
        config=SimpleNamespace(
            preflow_steps=1,
            preflow_convergence_tolerance=0.0,
            preflow_convergence_mode="single_step_legacy",
            preflow_traction_readiness_mode="flow_only",
            preflow_flow_driver_mode="sustained_boundary_predictor",
            detailed_preflow_stage_progress=detailed,
            apply_marker_feedback_to_fluid=True,
            flow_reset_pressure_each_step=False,
            mpm_support_radius_m=0.001,
            dt_s=5.0e-4,
            export_final_flow_snapshot=False,
        ),
        progress_observer=events.append,
    )

    assert report["preflow_steps_completed"] == 1
    assert [event["phase"] for event in events] == expected_phases

    step_events = [event for event in events if event["phase"] == "preflow_step"]
    assert [event["preflow_steps_completed"] for event in step_events] == [0, 1]
    assert all(event["preflow_step"] == 1 for event in step_events)

    if detailed:
        assert callable(forwarded_stage_observers[0])
        assert events[1]["preflow_stage"] == "behavior_probe"
        assert events[1]["preflow_steps_completed"] == 0
    else:
        assert forwarded_stage_observers == [None]
        assert not any(event["phase"] == "preflow_stage" for event in events)
