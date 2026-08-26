from __future__ import annotations

import inspect
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig


class _FakeSolid:
    def __init__(self) -> None:
        self.failures_remaining = 1
        self.events: list[str] = []
        self.step_dt_s: list[float] = []
        self.final_report = object()

    def save_state(self) -> None:
        self.events.append("save")

    def restore_state(self) -> None:
        self.events.append("restore")

    def begin_out_of_bounds_guard_batch(self) -> None:
        self.events.append("begin")

    def step(self, **kwargs: object) -> None:
        self.events.append("step")
        self.step_dt_s.append(float(kwargs["dt_s"]))
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise runner.SolidTrialRejectedError("forced solid rejection")

    def end_out_of_bounds_guard_batch(self) -> object:
        self.events.append("end")
        return self.final_report

    def abort_out_of_bounds_guard_batch(self) -> None:
        self.events.append("abort")


class _AdaptiveFakeSolid(_FakeSolid):
    """Host-only accepted-state fake for the proposed macro-step controller."""

    def __init__(
        self,
        *,
        accepted_speeds: tuple[float, ...] = (0.0,),
        failures_remaining: int = 0,
        reports: tuple[object, ...] | None = None,
    ) -> None:
        super().__init__()
        self.accepted_speeds = accepted_speeds
        self.accepted_speed_index = 0
        self.failures_remaining = failures_remaining
        self.events = []
        self.external_force_n = 11.0
        self.retry_force_n: float | None = None
        self.successful_step_forces: list[float] = []
        self.reports = reports or (_finite_solid_report(),)
        self.report_index = 0

    def accepted_particle_max_speed(self) -> float:
        speed = self.accepted_speeds[self.accepted_speed_index]
        self.events.append(f"accepted_speed:{speed:g}")
        return speed

    def restore_state(self) -> None:
        super().restore_state()
        self.external_force_n = 0.0
        self.accepted_speed_index = min(
            self.accepted_speed_index + 1,
            len(self.accepted_speeds) - 1,
        )

    def step(self, **kwargs: object) -> None:
        super().step(**kwargs)
        if self.failures_remaining == 0:
            self.successful_step_forces.append(self.external_force_n)
            if self.retry_force_n is not None:
                assert self.external_force_n == self.retry_force_n

    def end_out_of_bounds_guard_batch(self) -> object:
        report = self.reports[min(self.report_index, len(self.reports) - 1)]
        self.report_index += 1
        self.events.append("end")
        return report

class _RequiredShellRegionFailureFakeSolid(_AdaptiveFakeSolid):
    """Expose a retryable final guard report fault after a completed trial."""

    def __init__(self, *, accepted_speeds: tuple[float, ...]) -> None:
        super().__init__(accepted_speeds=accepted_speeds)
        self.empty_region_failures_remaining = 1

    def end_out_of_bounds_guard_batch(self) -> object:
        self.events.append("end")
        if self.empty_region_failures_remaining:
            self.empty_region_failures_remaining -= 1
            raise runner.MpmRequiredRegionEmptyError(
                "primary shell region has no in-grid MPM particles"
            )
        return self.final_report


class _UnexpectedFailureFakeSolid(_AdaptiveFakeSolid):
    """Model an error which must roll back but must not be retried."""

    def step(self, **kwargs: object) -> None:
        self.events.append("step")
        self.step_dt_s.append(float(kwargs["dt_s"]))
        raise RuntimeError("unexpected solid implementation failure")


class _NonFinalClampFakeSolid(_AdaptiveFakeSolid):
    """A clamp is observed in the first substep but not the final report."""

    def __init__(self) -> None:
        super().__init__(
            reports=(
                _finite_solid_report(deformation_clamp_count=1),
            )
        )
        self.substep_deformation_clamp_counts = (1, 0)
        self.substep_index = 0

    def step(self, **kwargs: object) -> None:
        super().step(**kwargs)
        self.deformation_clamp_count = self.substep_deformation_clamp_counts[
            min(self.substep_index, len(self.substep_deformation_clamp_counts) - 1)
        ]
        self.substep_index += 1


def _finite_solid_report(
    *, max_speed_mps: float = 1.0,
    deformation_clamp_count: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        particle_count=1,
        active_grid_nodes=1,
        grid_out_of_bounds_particle_count=0,
        particle_spacing_m=1.0,
        grid_spacing_m=(1.0, 1.0, 1.0),
        total_mass_kg=1.0,
        total_volume_m3=1.0,
        primary_mean_displacement_m=(0.0, 0.0, 0.0),
        primary_mean_velocity_mps=(0.0, 0.0, 0.0),
        secondary_mean_displacement_m=(0.0, 0.0, 0.0),
        secondary_mean_velocity_mps=(0.0, 0.0, 0.0),
        particle_momentum_kg_mps=(0.0, 0.0, 0.0),
        grid_momentum_kg_mps=(0.0, 0.0, 0.0),
        external_force_n=(0.0, 0.0, 0.0),
        transfer_relative_error=0.0,
        max_speed_mps=max_speed_mps,
        max_abs_j=1.0,
        deformation_clamp_count=deformation_clamp_count,
        mean_radial_stretch=1.0,
        max_radial_stretch_error=0.0,
        primary_particle_count=1,
        secondary_particle_count=1,
    )


def _config(**changes: object) -> VerticalFlapFsiConfig:
    return replace(
        VerticalFlapFsiConfig(
            grid_nodes=(4, 256, 320),
            solid_substeps=None,
            step_count=1,
        ),
        **changes,
    )


def test_default_is_explicit_adaptive_mode_not_fixed_1600() -> None:
    config = _config()

    report = runner.solid_substep_cfl_report(
        config, max_particle_speed_mps=0.0
    )

    assert config.solid_substeps is None
    assert report["solid_substeps_mode"] == "adaptive"
    assert report["solid_substeps_requested"] is None
    assert report["solid_substeps_selected"] == 1280



@pytest.mark.parametrize(
    "solid_substeps",
    (True, False, 1.5, "2", 0, -1),
)
def test_fixed_substep_override_validator_rejects_non_positive_or_non_integer_values(
    solid_substeps: object,
) -> None:
    with pytest.raises(ValueError, match="solid_substeps"):
        runner._validate_rectangular_solid_config(
            _config(solid_substeps=solid_substeps)
        )


def test_fixed_substep_override_validator_accepts_none_auto_mode() -> None:
    runner._validate_rectangular_solid_config(_config(solid_substeps=None))


def test_selector_includes_accepted_speed_and_fixed_override() -> None:
    slow = runner.solid_substep_cfl_report(
        _config(), max_particle_speed_mps=0.0
    )
    fast = runner.solid_substep_cfl_report(
        _config(), max_particle_speed_mps=100.0
    )
    fixed = runner.solid_substep_cfl_report(
        _config(solid_substeps=1600), max_particle_speed_mps=0.0
    )

    assert fast["solid_substeps_selected"] > slow["solid_substeps_selected"]
    assert fixed["solid_substeps_mode"] == "fixed_override"
    assert fixed["solid_substeps_selected"] == 1600
    finer = runner.solid_substep_cfl_report(
        _config(grid_nodes=(4, 640, 1280)), max_particle_speed_mps=0.0
    )
    stiffer = runner.solid_substep_cfl_report(
        _config(young_modulus_pa=4.0e6), max_particle_speed_mps=0.0
    )
    longer = runner.solid_substep_cfl_report(
        _config(dt_s=1.0e-3), max_particle_speed_mps=0.0
    )
    stricter = runner.solid_substep_cfl_report(
        _config(solid_cfl_target=0.07), max_particle_speed_mps=0.0
    )
    for report in (slow, fast, fixed, finer, stiffer, longer, stricter):
        assert isinstance(report["solid_substeps_selected"], int)
        assert report["solid_substeps_selected"] > 0
    assert finer["solid_substeps_selected"] > slow["solid_substeps_selected"]
    assert stiffer["solid_substeps_selected"] > slow["solid_substeps_selected"]
    assert longer["solid_substeps_selected"] > slow["solid_substeps_selected"]
    assert stricter["solid_substeps_selected"] > slow["solid_substeps_selected"]


@pytest.mark.parametrize(
    "changes",
    (
        {"dt_s": 0.0},
        {"solid_cfl_target": 0.0},
        {"solid_density_kgm3": float("nan")},
        {"young_modulus_pa": float("inf")},
    ),
)
def test_selector_fails_closed_for_invalid_physical_inputs(
    changes: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError):
        runner.solid_substep_cfl_report(_config(**changes))
    monkeypatch.setattr(
        runner, "_solid_mpm_grid_spacing_m", lambda _: (0.0, 1.0, 1.0)
    )
    with pytest.raises(ValueError):
        runner.solid_substep_cfl_report(_config())

@pytest.mark.parametrize("speed", [float("nan"), float("inf"), -1.0])
def test_selector_fails_closed_for_invalid_accepted_speed(speed: float) -> None:
    with pytest.raises(ValueError, match="max_particle_speed_mps"):
        runner.solid_substep_cfl_report(
            _config(), max_particle_speed_mps=speed
        )


def test_selector_preserves_full_time_and_macro_damping() -> None:
    config = _config(velocity_damping=0.995)
    report = runner.solid_substep_cfl_report(
        config, max_particle_speed_mps=5.0
    )
    selected = int(report["solid_substeps_selected"])
    substep_dt_s = float(report["solid_substep_dt_s"])

    assert math.isclose(
        selected * substep_dt_s, config.dt_s, rel_tol=0.0, abs_tol=1.0e-18
    )
    assert math.isclose(
        runner._solid_substep_velocity_damping(
            config, solid_substeps=selected
        )
        ** selected,
        config.velocity_damping,
        rel_tol=0.0,
        abs_tol=3.0e-14,
    )


def test_rejected_solid_trial_restores_before_full_dt_retry() -> None:
    solid = _FakeSolid()
    prepared = 0

    def reprepare() -> None:
        nonlocal prepared
        prepared += 1

    result = runner._advance_solid_macro_step_with_retries(
        solid,
        SimpleNamespace(
            enforce_plane_strain_x=False,
            fixed_node_lock_policy="any_fixed_particle",
            solid_constitutive_model="plane_stress_linear_elastic",
            solid_velocity_transfer_flip_blend=0.0,
            velocity_damping=0.995,
            dt_s=1.0e-3,
            solid_max_substep_retries=2,
        ),
        selected_substeps=2,
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=reprepare,
    )

    assert result["solid_rejected_trial_count"] == 1
    assert result["solid_substeps_selected"] == 4
    assert result["solid_accepted_time_s"] == pytest.approx(1.0e-3)
    assert result["solid_remaining_unadvanced_time_s"] == 0.0
    assert prepared == 1
    assert solid.events[:4] == ["save", "begin", "step", "abort"]
    assert "restore" in solid.events
    assert result["solid_accepted_substep_count"] == 4
    assert len(solid.step_dt_s) == 5
    assert result["solid_substeps_executed_total"] == 5
    assert result["solid_step_kernel_launch_count"] == 5
    assert result["solid_guard_batch_count"] == 2
    assert result["solid_packed_report_device_to_host_transfer_count"] == 1


def test_solid_wall_time_synchronization_is_profile_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_events: list[str] = []
    monkeypatch.setattr(
        runner,
        "_synchronize_hibm_sharp_boundary_stage_timing",
        lambda: sync_events.append("sync"),
    )
    profiled_solid = _FakeSolid()
    profiled_solid.failures_remaining = 0
    profiled_solid.final_report = _finite_solid_report()

    profiled = runner._advance_solid_macro_step_with_retries(
        profiled_solid,
        _solid_macro_config(),
        selected_substeps=2,
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=lambda: None,
        profile_wall_time=True,
    )

    assert sync_events == ["sync", "sync"]
    assert profiled["solid_wall_time_synchronized"] is True

    sync_events.clear()
    unprofiled_solid = _FakeSolid()
    unprofiled_solid.failures_remaining = 0
    unprofiled_solid.final_report = _finite_solid_report()
    unprofiled = runner._advance_solid_macro_step_with_retries(
        unprofiled_solid,
        _solid_macro_config(),
        selected_substeps=2,
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=lambda: None,
        profile_wall_time=False,
    )

    assert sync_events == []
    assert unprofiled["solid_wall_time_synchronized"] is False


def test_selected_solid_macro_wall_time_covers_selector_and_save_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _TimedSolid(_AdaptiveFakeSolid):
        def accepted_particle_max_speed(self) -> float:
            events.append("selector")
            return super().accepted_particle_max_speed()

        def save_state(self) -> None:
            events.append("save")
            super().save_state()

    monkeypatch.setattr(
        runner,
        "_synchronize_hibm_sharp_boundary_stage_timing",
        lambda: events.append("sync"),
    )
    monkeypatch.setattr(
        runner,
        "solid_substep_cfl_report",
        lambda _config, *, max_particle_speed_mps: {
            "solid_substeps_selected": 1,
            "solid_elastic_wave_speed_mps": 2.0,
            "solid_max_particle_speed_mps": max_particle_speed_mps,
            "solid_min_grid_spacing_m": 1.0,
        },
    )
    profiled = runner._select_and_advance_solid_macro_step(
        _TimedSolid(),
        _config(),
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=lambda: None,
        profile_wall_time=True,
    )

    assert events[:3] == ["sync", "selector", "save"]
    assert events[-1] == "sync"
    assert events.count("sync") == 2
    assert profiled["solid_wall_time_synchronized"] is True

    events.clear()
    unprofiled = runner._select_and_advance_solid_macro_step(
        _TimedSolid(),
        _config(),
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=lambda: None,
        profile_wall_time=False,
    )

    assert "sync" not in events
    assert unprofiled["solid_wall_time_synchronized"] is False


def test_solid_position_snapshot_capture_includes_the_device_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Field:
        def to_numpy(self) -> np.ndarray:
            events.append("read")
            return np.arange(6, dtype=float).reshape(2, 3)

    solid = SimpleNamespace(x=_Field(), particle_count=1)
    monkeypatch.setattr(
        runner,
        "_synchronize_hibm_sharp_boundary_stage_timing",
        lambda: events.append("sync"),
    )

    positions, wall_time_s = runner._capture_solid_positions_for_step(
        solid,
        profile_wall_time=True,
    )

    assert events == ["sync", "read", "sync"]
    assert positions.tolist() == [[0.0, 1.0, 2.0]]
    assert wall_time_s >= 0.0

    events.clear()
    positions, wall_time_s = runner._capture_solid_positions_for_step(
        solid,
        profile_wall_time=False,
    )

    assert events == ["read"]
    assert positions.tolist() == [[0.0, 1.0, 2.0]]
    assert wall_time_s == 0.0


def test_nonfinite_completed_solid_trial_is_rejected_with_typed_cause() -> None:
    solid = _FakeSolid()
    solid.failures_remaining = 0
    solid.final_report = _finite_solid_report(max_speed_mps=float("nan"))

    with pytest.raises(RuntimeError) as error_info:
        runner._advance_solid_macro_step_with_retries(
            solid,
            SimpleNamespace(
                enforce_plane_strain_x=False,
                fixed_node_lock_policy="any_fixed_particle",
                solid_constitutive_model="plane_stress_linear_elastic",
                solid_velocity_transfer_flip_blend=0.0,
                velocity_damping=0.995,
                dt_s=1.0e-3,
                solid_max_substep_retries=0,
                solid_max_automatic_substeps=16,
            ),
            selected_substeps=2,
            mu_pa=2.0,
            lambda_pa=3.0,
            retry_prepare=lambda: None,
        )

    assert isinstance(error_info.value.__cause__, runner.SolidTrialRejectedError)


def test_macro_step_controller_selects_from_each_accepted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_counts: list[int] = []
    selected_speeds: list[float] = []

    def select_from_speed(
        config: object,
        *,
        max_particle_speed_mps: float,
    ) -> dict[str, object]:
        selected_speeds.append(max_particle_speed_mps)
        selected = 3 if max_particle_speed_mps == 0.0 else 7
        return {
            "solid_substeps_selected": selected,
            "solid_substep_dt_s": float(getattr(config, "dt_s")) / selected,
            "solid_estimated_cfl": 0.25,
            "solid_elastic_wave_speed_mps": 10.0,
        }

    def advance(
        solid: object,
        config: object,
        *,
        selected_substeps: int,
        **_: object,
    ) -> dict[str, object]:
        selected_counts.append(selected_substeps)
        return {
            "solid_substeps_selected": selected_substeps,
            "solid_accepted_substep_count": selected_substeps,
            "solid_accepted_time_s": float(getattr(config, "dt_s")),
            "solid_rejected_trial_count": 0,
            "solid_remaining_unadvanced_time_s": 0.0,
            "solid_wall_time_s": 0.0,
        }

    monkeypatch.setattr(runner, "solid_substep_cfl_report", select_from_speed)
    monkeypatch.setattr(runner, "_advance_solid_macro_step_with_retries", advance)
    config = _config()

    slow = runner._select_and_advance_solid_macro_step(
        _AdaptiveFakeSolid(accepted_speeds=(0.0,)),
        config,
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=lambda: None,
    )
    fast = runner._select_and_advance_solid_macro_step(
        _AdaptiveFakeSolid(accepted_speeds=(20.0,)),
        config,
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=lambda: None,
    )

    assert selected_speeds == [0.0, 20.0]
    assert selected_counts == [3, 7]
    assert slow["solid_substeps_selected"] == 3
    assert fast["solid_substeps_selected"] == 7
    assert fast["solid_max_particle_speed_mps"] == 20.0
    assert slow["solid_selector_device_to_host_scalar_read_count"] == 1
    assert fast["solid_selector_device_to_host_scalar_read_count"] == 1


def test_macro_step_controller_reselects_after_restore_and_rebuilds_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_speeds: list[float] = []

    def select_from_speed(
        config: object,
        *,
        max_particle_speed_mps: float,
    ) -> dict[str, object]:
        selected_speeds.append(max_particle_speed_mps)
        selected = 2 if max_particle_speed_mps == 1.0 else 7
        return {
            "solid_substeps_selected": selected,
            "solid_substep_dt_s": float(getattr(config, "dt_s")) / selected,
            "solid_estimated_cfl": 0.25,
            "solid_elastic_wave_speed_mps": 10.0,
        }

    monkeypatch.setattr(runner, "solid_substep_cfl_report", select_from_speed)
    solid = _AdaptiveFakeSolid(
        accepted_speeds=(1.0, 9.0), failures_remaining=1
    )

    def rebuild_external_force() -> None:
        solid.events.append("retry_prepare")
        solid.retry_force_n = 17.0
        solid.external_force_n = 17.0

    config = _config(dt_s=1.0e-3, solid_max_substep_retries=2)
    result = runner._select_and_advance_solid_macro_step(
        solid,
        config,
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=rebuild_external_force,
    )

    assert selected_speeds == [1.0, 9.0]
    assert solid.events.index("restore") < solid.events.index("retry_prepare")
    assert solid.events.index("retry_prepare") < solid.events.index("accepted_speed:9")
    assert result["solid_substeps_selected"] == 7
    assert result["solid_accepted_substep_count"] == 7
    assert result["solid_accepted_time_s"] == pytest.approx(1.0e-3)
    assert result["solid_rejected_trial_count"] == 1
    assert solid.successful_step_forces == [17.0] * 7
    assert result["solid_selector_device_to_host_scalar_read_count"] == 2
    assert result["solid_step_kernel_launch_count"] == 8
    assert result["solid_step_kernel_launch_count"] == solid.events.count("step")
    assert result["solid_guard_batch_count"] == 2
    assert result["solid_packed_report_device_to_host_transfer_count"] == 1


def test_macro_step_controller_rejects_initial_selector_above_limit_before_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "solid_substep_cfl_report",
        lambda *_args, **_kwargs: {
            "solid_substeps_selected": 9,
            "solid_substep_dt_s": 1.0e-3 / 9.0,
            "solid_estimated_cfl": 0.25,
            "solid_elastic_wave_speed_mps": 10.0,
        },
    )
    solid = _AdaptiveFakeSolid()

    with pytest.raises(RuntimeError, match="solid_max_automatic_substeps"):
        runner._select_and_advance_solid_macro_step(
            solid,
            _config(solid_max_automatic_substeps=8),
            mu_pa=2.0,
            lambda_pa=3.0,
            retry_prepare=lambda: None,
        )

    assert solid.events == ["accepted_speed:0"]


def test_runner_contract_uses_per_macro_adaptive_controller_and_exports_counts() -> None:
    runner_source = inspect.getsource(runner.run_hibm_mpm_fsi)
    loop_source = runner_source[runner_source.index("for step_index in range"):]
    history_start = loop_source.index("history.append(")
    history_row_source = loop_source[
        history_start:
        loop_source.index("if step_observer is not None:", history_start)
    ]
    helper_source = inspect.getsource(runner._select_and_advance_solid_macro_step)
    summary_source = inspect.getsource(runner._solid_substep_run_summary)

    assert "_select_and_advance_solid_macro_step(" in loop_source
    assert "_advance_solid_substeps_batched(" not in loop_source
    assert "solid_substep_cfl = solid_substep_cfl_report(config)" not in (
        runner_source[: runner_source.index("for step_index in range")]
    )
    assert "accepted_particle_max_speed" in helper_source
    assert "retry_selected_substeps" in helper_source
    assert '"solid_selector_evaluation_count"' in history_row_source
    for field_name in (
        "solid_substeps_selected",
        "solid_substep_dt_s",
        "solid_estimated_cfl",
        "solid_elastic_wave_speed_mps",
        "solid_max_particle_speed_mps",
        "solid_accepted_time_s",
        "solid_rejected_trial_count",
        "solid_remaining_unadvanced_time_s",
        "solid_substeps_total",
        "solid_substeps_min",
        "solid_substeps_max",
        "solid_substeps_mean",
        "solid_retry_count_total",
        "solid_wall_time_s",
        "solid_step_kernel_launch_count",
        "solid_selector_device_to_host_scalar_read_count",
        "solid_packed_report_device_to_host_transfer_count",
        "solid_guard_batch_count",
    ):
        assert field_name in runner_source or field_name in summary_source


def test_runner_persists_initialized_runtime_identity_in_both_top_level_reports() -> None:
    runner_source = inspect.getsource(runner.run_hibm_mpm_fsi)
    preflow_source = inspect.getsource(runner._preflow_only_report)

    assert "taichi_runtime_identity" in runner_source
    assert "runtime_identity = taichi_runtime_identity()" in runner_source
    assert '"taichi_runtime_identity": dict(runtime_identity)' in runner_source
    assert '"profile_wall_time_enabled": bool(profile_wall_time)' in runner_source
    assert '"taichi_runtime_identity": dict(runtime_identity)' in preflow_source
    assert '"profile_wall_time_enabled": bool(profile_wall_time)' in preflow_source


def test_solid_run_summary_counts_rejected_trial_work_separately() -> None:
    summary = runner._solid_substep_run_summary(
        [
            {
                "solid_substeps_executed_total": 5,
                "solid_accepted_substep_count": 3,
                "solid_step_kernel_launch_count": 5,
                "solid_selector_device_to_host_scalar_read_count": 2,
                "solid_packed_report_device_to_host_transfer_count": 2,
                "solid_guard_batch_count": 2,
                "solid_retry_count": 1,
                "solid_rejected_trial_count": 1,
                "solid_wall_time_s": 2.0,
            },
            {
                "solid_substeps_executed_total": 4,
                "solid_accepted_substep_count": 4,
                "solid_step_kernel_launch_count": 4,
                "solid_selector_device_to_host_scalar_read_count": 1,
                "solid_packed_report_device_to_host_transfer_count": 1,
                "solid_guard_batch_count": 1,
                "solid_retry_count": 0,
                "solid_rejected_trial_count": 0,
                "solid_wall_time_s": 3.0,
            },
        ]
    )

    assert summary == {
        "solid_substeps_total": 9,
        "solid_substeps_min": 4,
        "solid_substeps_max": 5,
        "solid_substeps_mean": 4.5,
        "solid_step_kernel_launch_count_total": 9,
        "solid_selector_device_to_host_scalar_read_count_total": 3,
        "solid_packed_report_device_to_host_transfer_count_total": 3,
        "solid_guard_batch_count_total": 3,
        "solid_accepted_substeps_total": 7,
        "solid_substeps_selected_min": 3,
        "solid_substeps_selected_max": 4,
        "solid_substeps_selected_mean": 3.5,
        "solid_retry_count_total": 1,
        "solid_rejected_trial_count_total": 1,
        "solid_wall_time_s": 5.0,
    }


def test_solid_run_summary_does_not_invent_missing_packed_report_reads() -> None:
    with pytest.raises(
        KeyError,
        match="solid_packed_report_device_to_host_transfer_count",
    ):
        runner._solid_substep_run_summary(
            [
                {
                    "solid_substeps_executed_total": 1,
                    "solid_accepted_substep_count": 1,
                    "solid_step_kernel_launch_count": 1,
                    "solid_selector_device_to_host_scalar_read_count": 1,
                    "solid_guard_batch_count": 1,
                    "solid_retry_count": 0,
                    "solid_rejected_trial_count": 0,
                    "solid_wall_time_s": 1.0,
                }
            ]
        )


def _solid_macro_config(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "enforce_plane_strain_x": False,
        "fixed_node_lock_policy": "any_fixed_particle",
        "solid_constitutive_model": "plane_stress_linear_elastic",
        "solid_velocity_transfer_flip_blend": 0.0,
        "velocity_damping": 0.995,
        "dt_s": 1.0e-3,
        "solid_max_substep_retries": 1,
        "solid_max_automatic_substeps": 16,
        "solid_max_deformation_clamp_count_per_macro_step": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_nonfinal_deformation_clamp_is_accumulated_then_rejected_and_rolled_back() -> None:
    step_source = inspect.getsource(runner.NeoHookeanMpmState._step_kernel)
    reset_source = inspect.getsource(
        runner.NeoHookeanMpmState._reset_out_of_bounds_guard_batch_kernel
    )
    sticky_name = "deformation_clamp_guard_batch_total_count"
    assert sticky_name in reset_source
    assert f"self.{sticky_name}[None] += (" in step_source
    assert (
        f"self.{sticky_name}[None]," in step_source
    )

    solid = _NonFinalClampFakeSolid()

    with pytest.raises(RuntimeError) as error_info:
        runner._advance_solid_macro_step_with_retries(
            solid,
            _solid_macro_config(solid_max_substep_retries=0),
            selected_substeps=2,
            mu_pa=2.0,
            lambda_pa=3.0,
            retry_prepare=lambda: None,
        )

    assert isinstance(error_info.value.__cause__, runner.SolidTrialRejectedError)
    assert solid.events == [
        "save",
        "begin",
        "step",
        "step",
        "end",
        "restore",
    ]


def test_required_shell_region_failure_restores_reselects_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_speeds: list[float] = []

    def select_from_speed(
        config: object,
        *,
        max_particle_speed_mps: float,
    ) -> dict[str, object]:
        selected_speeds.append(max_particle_speed_mps)
        selected = 2 if len(selected_speeds) == 1 else 3
        return {
            "solid_substeps_selected": selected,
            "solid_substep_dt_s": float(getattr(config, "dt_s")) / selected,
            "solid_estimated_cfl": 0.25,
            "solid_elastic_wave_speed_mps": 10.0,
        }

    monkeypatch.setattr(runner, "solid_substep_cfl_report", select_from_speed)
    solid = _RequiredShellRegionFailureFakeSolid(
        accepted_speeds=(1.0, 9.0)
    )
    retry_prepare_calls = 0

    def retry_prepare() -> None:
        nonlocal retry_prepare_calls
        retry_prepare_calls += 1
        solid.events.append("retry_prepare")

    result = runner._select_and_advance_solid_macro_step(
        solid,
        _config(solid_max_substep_retries=1),
        mu_pa=2.0,
        lambda_pa=3.0,
        retry_prepare=retry_prepare,
    )

    assert selected_speeds == [1.0, 9.0]
    assert retry_prepare_calls == 1
    assert solid.events.index("restore") < solid.events.index("retry_prepare")
    assert solid.events.index("retry_prepare") < solid.events.index(
        "accepted_speed:9"
    )
    assert result["solid_rejected_trial_count"] == 1
    assert result["solid_substeps_selected"] == 4
    assert result["solid_accepted_time_s"] == pytest.approx(5.0e-4)
    assert result["solid_guard_batch_count"] == 2
    assert result["solid_packed_report_device_to_host_transfer_count"] == 2


def test_unexpected_solid_failure_restores_then_rethrows_without_retry_prepare() -> None:
    solid = _UnexpectedFailureFakeSolid()
    retry_prepare_calls = 0

    def retry_prepare() -> None:
        nonlocal retry_prepare_calls
        retry_prepare_calls += 1

    with pytest.raises(RuntimeError, match="unexpected solid implementation failure"):
        runner._advance_solid_macro_step_with_retries(
            solid,
            _solid_macro_config(),
            selected_substeps=2,
            mu_pa=2.0,
            lambda_pa=3.0,
            retry_prepare=retry_prepare,
        )

    assert solid.events == ["save", "begin", "step", "abort", "restore"]
    assert retry_prepare_calls == 0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("solid_max_substep_retries", True),
        ("solid_max_substep_retries", 1.5),
        ("solid_max_substep_retries", "2"),
        ("solid_max_substep_retries", -1),
        ("solid_max_automatic_substeps", True),
        ("solid_max_automatic_substeps", 1.5),
        ("solid_max_automatic_substeps", "2"),
        ("solid_max_automatic_substeps", 0),
        ("solid_max_automatic_substeps", -1),
        ("solid_max_deformation_clamp_count_per_macro_step", True),
        ("solid_max_deformation_clamp_count_per_macro_step", 1.5),
        ("solid_max_deformation_clamp_count_per_macro_step", "2"),
        ("solid_max_deformation_clamp_count_per_macro_step", -1),
    ),
)
def test_solid_retry_cap_and_clamp_limits_require_non_boolean_integers(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError):
        runner._validate_rectangular_solid_config(
            _config(**{field_name: invalid_value})
        )


def test_substep_dt_underflow_fails_closed_before_solid_trial() -> None:
    solid = _FakeSolid()
    solid.failures_remaining = 0

    with pytest.raises(ValueError, match="solid substep dt"):
        runner._advance_solid_macro_step_with_retries(
            solid,
            _solid_macro_config(
                dt_s=math.ulp(0.0),
                solid_max_substep_retries=0,
            ),
            selected_substeps=2,
            mu_pa=2.0,
            lambda_pa=3.0,
            retry_prepare=lambda: None,
        )

    assert solid.events == []
