"""Exact macro-work and discrete-decision identity for R24C prefix replay."""

from __future__ import annotations

from typing import Any, Mapping

from .oracle_threshold_common import require


_PREFIX_WORK_KEYS = (
    "trial_count",
    "fluid_solve_count",
    "solid_macro_solve_count",
    "feedback_consumed_trial_count",
    "cg_iterations_total",
    "flow_momentum_advection_substeps_total",
    "flow_sst_transport_substeps_total",
    "solid_substeps_executed_total",
)
# Exact work deliberately means the eight macro counters above.  Inner-solver
# telemetry such as the final SST Helmholtz iteration count may vary by one
# across otherwise equivalent CUDA replays and is not represented as exact.


def _required(mapping: Mapping[str, Any], key: str, *, label: str) -> Any:
    require(key in mapping, f"{label} {key} missing")
    return mapping[key]


def _typed(value: Any, kind: str, *, label: str) -> Any:
    if kind == "int":
        require(
            type(value) is int and value >= 0,
            f"{label} must be a nonnegative JSON integer",
        )
    elif kind == "bool":
        require(type(value) is bool, f"{label} must be a JSON boolean")
    elif kind == "text":
        require(isinstance(value, str) and bool(value), f"{label} must be text")
    elif kind == "sha256":
        require(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"{label} must be a lowercase SHA-256",
        )
    elif kind == "optional_text":
        require(
            value is None or isinstance(value, str) and bool(value),
            f"{label} must be null or text",
        )
    else:  # pragma: no cover - specifications are module constants
        raise AssertionError(f"unknown strict JSON kind: {kind}")
    return value


def _identity(
    mapping: Mapping[str, Any],
    fields: Mapping[str, str],
    *,
    label: str,
) -> dict[str, Any]:
    return {
        key: _typed(
            _required(mapping, key, label=label),
            kind,
            label=f"{label} {key}",
        )
        for key, kind in fields.items()
    }


def _history(
    value: Any,
    kind: str,
    *,
    label: str,
) -> tuple[Any, ...]:
    require(isinstance(value, list), f"{label} must be a list")
    return tuple(
        _typed(item, kind, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def prefix_work_identity(step: Any) -> dict[str, int]:
    """Return the complete, strictly typed macro-work counter identity."""

    label = f"step {step.step} trial work"
    raw = step.history.get("hibm_fsi_trial_work_report")
    require(isinstance(raw, Mapping), f"{label} missing")
    return {
        key: _typed(
            _required(raw, key, label=label),
            "int",
            label=f"{label} {key}",
        )
        for key in _PREFIX_WORK_KEYS
    }


def _initial_guess_identity(step: Any) -> dict[str, Any]:
    label = f"step {step.step} decision initial guess"
    history = step.history
    report = _required(history, "initial_guess_report", label=label)
    require(isinstance(report, Mapping), f"{label} report must be an object")
    return {
        "top": _identity(
            history,
            {
                "initial_guess_mode_requested": "text",
                "initial_guess_mode_used": "text",
                "initial_guess_fallback_reason": "optional_text",
            },
            label=label,
        ),
        "report": _identity(
            report,
            {
                "accepted_step_count": "int",
                "begin_count": "int",
                "deployable": "bool",
                "discard_count": "int",
                "fallback_reason": "optional_text",
                "has_active_step": "bool",
                "kalman_accepted_state_count": "int",
                "kalman_prediction_used": "bool",
                "kalman_ready": "bool",
                "mode": "text",
                "mode_used": "text",
                "offline_oracle": "bool",
                "oracle_replay_cursor": "int",
            },
            label=f"{label} report",
        ),
    }


def _coupling_identity(step: Any) -> dict[str, Any]:
    label = f"step {step.step} decision coupling"
    history = step.history
    scalars = _identity(
        history,
        {
            "hibm_fsi_coupling_converged": "bool",
            "hibm_fsi_coupling_iterations_used": "int",
            "hibm_fsi_coupling_rejected_trial_count": "int",
            "hibm_fsi_coupling_explicit_single_pass": "bool",
            "hibm_fsi_coupling_base_assembly_count": "int",
            "hibm_fsi_coupling_residual_source": "text",
            "hibm_fsi_coupling_iqn_fallback_count": "int",
        },
        label=label,
    )
    iterations = scalars["hibm_fsi_coupling_iterations_used"]
    require(iterations > 0, f"{label} iterations must be positive")
    update_count = iterations - 1
    update_modes = _history(
        _required(history, "hibm_fsi_coupling_update_mode_history", label=label),
        "text",
        label=f"{label} update modes",
    )
    ranks = _history(
        _required(history, "hibm_fsi_coupling_iqn_rank_history", label=label),
        "int",
        label=f"{label} IQN ranks",
    )
    fallback_reasons = _history(
        _required(history, "hibm_fsi_coupling_iqn_fallback_reasons", label=label),
        "optional_text",
        label=f"{label} IQN fallback reasons",
    )
    limited = _history(
        _required(
            history,
            "hibm_fsi_coupling_iqn_update_limited_history",
            label=label,
        ),
        "bool",
        label=f"{label} IQN update-limited history",
    )
    for name, values in (
        ("update mode", update_modes),
        ("rank", ranks),
        ("fallback", fallback_reasons),
        ("update-limited", limited),
    ):
        require(
            len(values) == update_count,
            f"{label} {name} history disagrees with iterations",
        )
    fallback_count = scalars["hibm_fsi_coupling_iqn_fallback_count"]
    require(
        fallback_count == sum(item is not None for item in fallback_reasons),
        f"{label} IQN fallback count disagrees",
    )
    return {
        "scalars": scalars,
        "update_modes": update_modes,
        "iqn_ranks": ranks,
        "iqn_fallback_reasons": fallback_reasons,
        "iqn_update_limited": limited,
    }


def _reuse_identity(step: Any) -> dict[str, Any]:
    label = f"step {step.step} decision IQN reuse"
    reuse = _required(step.history, "hibm_iqn_reuse", label=label)
    require(isinstance(reuse, Mapping), f"{label} must be an object")
    identity = _identity(
        reuse,
        {
            "enabled": "bool",
            "first_update_mode": "optional_text",
            "imported_pair_count": "int",
            "local_pair_count": "int",
            "reset_reason": "optional_text",
            "retained_pair_count": "int",
            "used": "bool",
        },
        label=label,
    )
    source_step = _required(reuse, "source_step", label=label)
    identity["source_step"] = (
        None
        if source_step is None
        else _typed(source_step, "int", label=f"{label} source_step")
    )
    return identity


def _topology_and_health_identity(step: Any) -> dict[str, Any]:
    label = f"step {step.step} decision topology/health"
    history = step.history
    identity = _identity(
        history,
        {
            "feedback_available_before_projection": "bool",
            "fluid_projection_consumed_feedback": "bool",
            "fluid_recomputed": "bool",
            "fluid_recomputed_after_feedback": "bool",
            "material_binding_identity": "sha256",
            "pressure_pair_anchor_current_marker_geometry_revision": "int",
            "pressure_pair_anchor_current_marker_geometry_sha256": "sha256",
            "pressure_pair_anchor_map_sha256": "sha256",
            "pressure_pair_anchor_runtime_refresh_count": "int",
            "pressure_pair_anchor_source_marker_geometry_revision": "int",
            "pressure_pair_anchor_source_marker_geometry_sha256": "sha256",
            "hibm_velocity_dirichlet_authority": "text",
            "hibm_velocity_dirichlet_authority_registered": "bool",
            "hibm_velocity_dirichlet_authority_sealed": "bool",
            "hibm_velocity_dirichlet_ledger_generation": "int",
        },
        label=label,
    )
    current_geometry_sha256 = identity.pop(
        "pressure_pair_anchor_current_marker_geometry_sha256"
    )
    source_geometry_sha256 = identity.pop(
        "pressure_pair_anchor_source_marker_geometry_sha256"
    )
    require(
        source_geometry_sha256 == current_geometry_sha256,
        f"{label} source/current marker geometry SHA-256 disagrees",
    )
    require(
        identity["pressure_pair_anchor_source_marker_geometry_revision"]
        == identity["pressure_pair_anchor_current_marker_geometry_revision"],
        f"{label} source/current marker geometry revision disagrees",
    )
    healthy = {
        "flow_projection_cg_converged_all": ("bool", True),
        "flow_projection_cg_breakdown_count": ("int", 0),
        "flow_projection_pressure_solve_failed": ("bool", False),
        "mpm_grid_out_of_bounds_particle_count": ("int", 0),
        "mpm_deformation_clamp_count": ("int", 0),
        "solid_retry_count": ("int", 0),
        "hibm_no_slip_invalid_marker_count": ("int", 0),
    }
    for key, (kind, expected) in healthy.items():
        value = _typed(
            _required(history, key, label=label),
            kind,
            label=f"{label} {key}",
        )
        require(value == expected, f"{label} {key} is unhealthy")
        identity[key] = value
    canonical = _required(
        history,
        "canonical_velocity_dirichlet_report",
        label=label,
    )
    require(isinstance(canonical, Mapping), f"{label} canonical report missing")
    closure = _required(canonical, "marker_target_closure", label=label)
    require(isinstance(closure, Mapping), f"{label} closure report missing")
    invalid_axes = _typed(
        _required(closure, "projection_only_invalid_axis_count", label=label),
        "int",
        label=f"{label} projection-only invalid axes",
    )
    require(invalid_axes == 0, f"{label} projection-only closure is unhealthy")
    identity["projection_only_invalid_axis_count"] = invalid_axes
    return identity


def prefix_decision_identity(step: Any) -> dict[str, Any]:
    """Return strictly typed discrete decisions that define one replay path."""

    return {
        "initial_guess": _initial_guess_identity(step),
        "coupling": _coupling_identity(step),
        "reuse": _reuse_identity(step),
        "topology_and_health": _topology_and_health_identity(step),
    }


__all__ = ("prefix_decision_identity", "prefix_work_identity")
