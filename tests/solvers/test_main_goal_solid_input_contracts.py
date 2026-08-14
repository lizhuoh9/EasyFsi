from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from simulation_core.solids.mooney_shell import (
    TriMooneyShellMpmState,
    UvMooneyShellMpmState,
)
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmState


def _rejects_before_kernel(state: object, kernel_name: str, call) -> None:
    kernel_calls: list[tuple[object, ...]] = []
    setattr(state, kernel_name, lambda *args: kernel_calls.append(args))
    before = vars(state).copy()

    with pytest.raises(ValueError):
        call()

    assert kernel_calls == []
    assert vars(state) == before


@pytest.mark.parametrize(
    ("method_name", "kernel_name", "kwargs"),
    [
        (
            "set_region_normal_pressure",
            "_set_region_normal_pressure_kernel",
            {"region_id": 1, "pressure_pa": math.nan},
        ),
        (
            "add_region_normal_pressure",
            "_add_region_normal_pressure_kernel",
            {"region_id": 1, "pressure_pa": 1.0e100},
        ),
        (
            "set_uniform_velocity",
            "_set_uniform_velocity_kernel",
            {"velocity_mps": (0.0, math.inf, 0.0)},
        ),
        (
            "set_uniform_external_force",
            "_set_uniform_external_force_kernel",
            {"force_n": (0.0, 1.0e-50, 0.0)},
        ),
    ],
)
def test_neo_public_mutators_reject_invalid_f32_before_kernel(
    method_name: str,
    kernel_name: str,
    kwargs: dict[str, object],
) -> None:
    state = object.__new__(NeoHookeanMpmState)
    state.particle_count = 1

    _rejects_before_kernel(
        state,
        kernel_name,
        lambda: getattr(state, method_name)(**kwargs),
    )


def test_neo_public_mutators_pass_quantized_f32_values() -> None:
    state = object.__new__(NeoHookeanMpmState)
    state.particle_count = 1
    calls: list[tuple[object, ...]] = []
    state._set_uniform_velocity_kernel = lambda *args: calls.append(args)

    state.set_uniform_velocity((0.1, -0.2, 0.3))

    assert calls == [
        (
            float(np.float32(0.1)),
            float(np.float32(-0.2)),
            float(np.float32(0.3)),
            1,
        )
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"thickness_m": 1.0e-50},
        {"density_kgm3": 1.0e100},
        {"c1_pa": math.inf},
        {"primary_thickness_m": 1.0e-50},
    ],
)
def test_mooney_constructor_rejects_nonpositive_f32_materials(
    overrides: dict[str, float],
) -> None:
    arguments = {
        "mesh": SimpleNamespace(vertex_count=3, face_count=1),
        "thickness_m": 0.1,
        "density_kgm3": 1.0,
        "c1_pa": 1.0,
        "c2_pa": 0.0,
        "primary_region_id": 1,
        "secondary_region_id": 2,
        **overrides,
    }

    with pytest.raises(ValueError):
        TriMooneyShellMpmState(**arguments)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dt_s": -1.0e-3},
        {"dt_s": 1.0e-50},
        {"pressure_pa": math.nan},
        {"pressure_pa": 1.0e100},
        {"pressure_pa": 1.0e-50},
        {"velocity_damping": math.inf},
        {"velocity_damping": -0.01},
        {"velocity_damping": 1.0 + 1.0e-8},
    ],
)
def test_mooney_step_rejects_invalid_dynamics_before_kernel(
    overrides: dict[str, float],
) -> None:
    state = object.__new__(TriMooneyShellMpmState)
    state.particle_count = 1
    arguments = {
        "dt_s": 1.0e-3,
        "pressure_pa": 0.0,
        "velocity_damping": 1.0,
        "read_report": False,
        **overrides,
    }

    _rejects_before_kernel(
        state,
        "_step_kernel",
        lambda: state.step(**arguments),
    )


@pytest.mark.parametrize(
    ("state_type", "method_name", "kernel_name", "kwargs"),
    [
        (
            TriMooneyShellMpmState,
            "advance_region_loads",
            "_step_region_kernel",
            {
                "dt_s": -1.0e-3,
                "primary_region_id": 1,
                "secondary_region_id": 2,
                "primary_area_load_npm2": (0.0, 0.0, 0.0),
                "primary_interface_reaction_n": (0.0, 0.0, 0.0),
                "secondary_interface_reaction_n": (0.0, 0.0, 0.0),
            },
        ),
        (
            TriMooneyShellMpmState,
            "advance_with_external_forces",
            "_step_region_kernel",
            {
                "dt_s": 1.0e-3,
                "primary_region_id": 1,
                "secondary_region_id": 2,
                "velocity_damping": 1.01,
            },
        ),
        (
            UvMooneyShellMpmState,
            "step",
            "_step_kernel",
            {"dt_s": 1.0e-3, "pressure_pa": math.inf},
        ),
    ],
)
def test_all_mooney_step_boundaries_validate_before_kernel(
    state_type: type,
    method_name: str,
    kernel_name: str,
    kwargs: dict[str, object],
) -> None:
    state = object.__new__(state_type)
    state.particle_count = 1

    _rejects_before_kernel(
        state,
        kernel_name,
        lambda: getattr(state, method_name)(**kwargs),
    )


def test_mooney_zero_dt_remains_a_valid_static_probe() -> None:
    state = object.__new__(UvMooneyShellMpmState)
    kernel_calls: list[tuple[object, ...]] = []
    expected_report = object()
    state._step_kernel = lambda *args: kernel_calls.append(args)
    state.report = lambda **_kwargs: expected_report

    actual_report = state.step(dt_s=0.0, pressure_pa=0.0)

    assert actual_report is expected_report
    assert kernel_calls[0][0] == 0.0
