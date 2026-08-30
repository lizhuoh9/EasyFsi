"""Material-reference evidence gates for the current IQN fine50 profile.

This profile composes the current IQN contract unchanged, then requires the
immutable Cartesian reference-to-surface binding and its per-step adjoint
audit.  It is an input-identity contract only; it never claims legacy
acceptance or Fluent parity.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .current_iqn_adaptive_fine_contracts import (
    CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY,
    EXPECTED_STEPS,
    validate_current_iqn_adaptive_fine50,
)

PROFILE_ID = "current_iqn_adaptive_material_reference_fine50_v1"
MATERIAL_REFERENCE_FINE_CONFIG_IDENTITY = {
    **CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY,
    "surface_transfer_method": "cartesian_reference_adjoint_v1",
    "preserve_marker_area_during_surface_feedback": True,
}
MATERIAL_REACTION_FIELDS = (
    "mpm_direct_fixed_external_force_n",
    "mpm_support_reaction_impulse_n_s",
    "mpm_support_reaction_angular_impulse_n_m_s",
    "mpm_damping_impulse_n_s",
    "mpm_damping_angular_impulse_n_m_s",
)
MATERIAL_AUDIT_FIELDS = (
    "material_transfer_verified",
    "material_binding_identity",
    "scatter_action_reaction_residual_N",
    "force_roundoff_bound_n",
    "torque_residual_n_m",
    "torque_roundoff_bound_n_m",
    "material_power_residual_w",
    "material_power_roundoff_bound_w",
    *MATERIAL_REACTION_FIELDS,
)
PROFILE_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    {"profile": PROFILE_ID, "config": MATERIAL_REFERENCE_FINE_CONFIG_IDENTITY,
     "material_audit_fields": MATERIAL_AUDIT_FIELDS},
    sort_keys=True, separators=(",", ":"),
).encode("utf-8")).hexdigest()


class MaterialReferenceFineContractError(RuntimeError):
    """Artifacts do not establish the material-reference fine50 contract."""


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise MaterialReferenceFineContractError(label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _positive_count_product(value: Any) -> int:
    _require(isinstance(value, (list, tuple)) and value, "solid_particle_counts must be nonempty")
    counts = []
    for count in value:
        _require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
                 "solid_particle_counts must contain positive integers")
        counts.append(count)
    return math.prod(counts)


def _nonnegative_finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"{label} must be finite and nonnegative")
    return result


def _identity(value: Any, label: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{label} must be a lowercase SHA-256")
    return value


def _finite_vector(value: Any, label: str) -> list[float]:
    _require(isinstance(value, (list, tuple)) and len(value) == 3,
             f"{label} must contain three measured components")
    _require(all(isinstance(component, (int, float)) and not isinstance(component, bool)
                 for component in value), f"{label} components must be numeric")
    result = [float(component) for component in value]
    _require(all(math.isfinite(component) for component in result),
             f"{label} components must be finite")
    return result


def _material_configuration(summary: Mapping[str, Any], physical_marker_count: int,
                            particle_count: int) -> tuple[str, dict[str, float | int | str]]:
    configuration = _mapping(summary.get("material_transfer_configuration"),
                             "summary material transfer configuration")
    _require(configuration.get("method") == "cartesian_reference_adjoint_v1",
             "summary material transfer method mismatch")
    identity = _identity(configuration.get("identity_sha256"), "material binding identity")
    for key, expected in (("particle_count", particle_count), ("marker_count", physical_marker_count)):
        count = configuration.get(key)
        _require(isinstance(count, int) and not isinstance(count, bool) and count == expected,
                 f"summary material {key} must match the positive integer configuration")
    report: dict[str, float | int | str] = {
        "method": "cartesian_reference_adjoint_v1",
        "identity_sha256": identity,
        "particle_count": particle_count,
        "marker_count": physical_marker_count,
    }
    for key in ("maximum_row_l1", "maximum_row_inverse_mass_gain"):
        report[key] = _nonnegative_finite(configuration.get(key), f"summary {key}")
    return identity, report


def _material_audit(audit: Mapping[str, Any], *, identity: str, label: str) -> dict[str, Any]:
    _require(audit.get("material_transfer_verified") is True,
             f"{label} material transfer is not verified")
    _require(audit.get("material_binding_identity") == identity,
             f"{label} material binding identity changed")
    report: dict[str, Any] = {
        "material_transfer_verified": True,
        "material_binding_identity": identity,
    }
    for residual, bound in (
        ("scatter_action_reaction_residual_N", "force_roundoff_bound_n"),
        ("torque_residual_n_m", "torque_roundoff_bound_n_m"),
        ("material_power_residual_w", "material_power_roundoff_bound_w"),
    ):
        measured = _nonnegative_finite(audit.get(residual), f"{label} {residual}")
        limit = _nonnegative_finite(audit.get(bound), f"{label} {bound}")
        _require(measured <= limit, f"{label} {residual} exceeds {bound}")
        report[residual] = measured
        report[bound] = limit
    for field in MATERIAL_REACTION_FIELDS:
        report[field] = _finite_vector(audit.get(field), f"{label} {field}")
    return report


def _validate_material_reference_fine50(
    manifest: Mapping[str, Any], summary: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    trial_frames: Sequence[Mapping[str, Any]] | Callable[[int], Mapping[str, Any]],
    *, pressure_semantics_mode: str,
    base_validator: Callable[..., dict[str, Any]],
    profile_id: str,
    profile_contract_sha256: str,
    schema: str,
) -> dict[str, Any]:
    """Compose one IQN identity contract with immutable material evidence."""
    base = base_validator(
        manifest, summary, histories, trial_frames,
        pressure_semantics_mode=pressure_semantics_mode,
    )
    config = _mapping(manifest.get("config"), "manifest config")
    _require(config.get("surface_transfer_method") == "cartesian_reference_adjoint_v1",
             "manifest surface transfer method mismatch")
    _require(config.get("preserve_marker_area_during_surface_feedback") is True,
             "manifest must preserve marker area during surface feedback")
    particle_count = _positive_count_product(config.get("solid_particle_counts"))
    physical_marker_count = base["physical_marker_count"]
    identity, configuration = _material_configuration(
        summary, physical_marker_count, particle_count,
    )
    _require(len(histories) == EXPECTED_STEPS, "material profile requires 50 histories")
    history_reports = []
    for step, history in enumerate(histories, 1):
        history_reports.append(_material_audit(
            _mapping(history, f"history {step}"), identity=identity,
            label=f"history {step}",
        ))
    _require(all(key in summary for key in MATERIAL_AUDIT_FIELDS),
             "summary material audit is incomplete")
    summary_audit = _material_audit(summary, identity=identity, label="summary")
    _require(summary_audit == history_reports[-1],
             "summary material audit differs from the final accepted history")
    return {
        **base,
        "schema": schema,
        "status": "passed",
        "comparison_profile": profile_id,
        "profile_contract_sha256": profile_contract_sha256,
        "legacy_final_identity_satisfied": False,
        "legacy_final_acceptance_claimed": False,
        "requires_iqn_trial_vectors": True,
        "requires_material_reference_audit": True,
        "physical_marker_count": physical_marker_count,
        "physical_marker_count_cross_check": base["physical_marker_count_cross_check"],
        "marker_layout_sha256": base["marker_layout_sha256"],
        "trial_trace_reports": base["trial_trace_reports"],
        "material_transfer_configuration": configuration,
        "material_history_audit_reports": history_reports,
        "summary_material_audit": summary_audit,
    }


def validate_material_reference_fine50(
    manifest: Mapping[str, Any], summary: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    trial_frames: Sequence[Mapping[str, Any]] | Callable[[int], Mapping[str, Any]],
    *, pressure_semantics_mode: str,
) -> dict[str, Any]:
    """Validate the current IQN contract plus material-reference evidence."""
    return _validate_material_reference_fine50(
        manifest,
        summary,
        histories,
        trial_frames,
        pressure_semantics_mode=pressure_semantics_mode,
        base_validator=validate_current_iqn_adaptive_fine50,
        profile_id=PROFILE_ID,
        profile_contract_sha256=PROFILE_CONTRACT_SHA256,
        schema="current_iqn_adaptive_material_reference_fine50_identity_v1",
    )
