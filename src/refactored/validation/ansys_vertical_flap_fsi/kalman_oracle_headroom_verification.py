"""Independent bottom-up verification for R24B decision artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .kalman_oracle_headroom_analysis import analyze_oracle_headroom
from .kalman_oracle_headroom_artifacts import (
    _completed_blend_response,
    _initial_blend_response,
    _source_manifest_payload,
    _step_csv_bytes,
    _summary_payload,
    _verify_self_sha256,
)
from .kalman_oracle_headroom_contracts import (
    EXPECTED_STEPS,
    OracleHeadroomContractError,
    _REQUIRED_ARTIFACTS,
    _as_finite_float,
    _load_run,
    _read_json,
    _require,
    _sha256_file,
    _validate_pair,
)

_REQUIRED_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
_INTERMEDIATE_ALPHAS = (0.25, 0.5, 0.75)


def _absolute_run_root(value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} root missing")
    root = Path(value).expanduser()
    _require(root.is_absolute(), f"{label} root must be absolute")
    resolved = root.resolve()
    _require(resolved.is_dir(), f"{label} root does not exist: {resolved}")
    return resolved


def _identity_root(value: Any, *, label: str) -> Path:
    _require(isinstance(value, dict), f"{label} identity missing")
    return _absolute_run_root(value.get("root"), label=label)


def _completed_blend_inputs(
    blend: Mapping[str, Any],
    *,
    q0_root: Path,
    q3_root: Path,
) -> tuple[dict[float, Path], dict[float, Path]]:
    rows = blend.get("results")
    _require(
        isinstance(rows, list)
        and len(rows) == len(_REQUIRED_ALPHAS)
        and all(isinstance(row, dict) for row in rows),
        "completed blend results must contain the five required rows",
    )
    actual_alphas = tuple(
        _as_finite_float(row.get("alpha"), label="blend result alpha")
        for row in rows
    )
    _require(
        actual_alphas == _REQUIRED_ALPHAS,
        "completed blend results alpha order mismatch",
    )
    _require(
        _absolute_run_root(rows[0].get("run_root"), label="alpha 0 consumer")
        == q0_root
        and _absolute_run_root(rows[-1].get("run_root"), label="alpha 1 consumer")
        == q3_root,
        "completed blend endpoint roots mismatch",
    )

    producer_identity = blend.get("producer_identity")
    consumer_identity = blend.get("consumer_identity")
    expected_keys = {f"{alpha:.2f}" for alpha in _INTERMEDIATE_ALPHAS}
    _require(
        isinstance(producer_identity, dict)
        and set(producer_identity) == expected_keys,
        "completed blend producer identity keys mismatch",
    )
    _require(
        isinstance(consumer_identity, dict)
        and set(consumer_identity) == expected_keys,
        "completed blend consumer identity keys mismatch",
    )

    producer_roots: dict[float, Path] = {}
    consumer_roots: dict[float, Path] = {}
    for index, alpha in enumerate(_INTERMEDIATE_ALPHAS, start=1):
        key = f"{alpha:.2f}"
        producer_roots[alpha] = _identity_root(
            producer_identity[key],
            label=f"alpha {alpha:.2f} producer",
        )
        consumer = _identity_root(
            consumer_identity[key],
            label=f"alpha {alpha:.2f} consumer",
        )
        row_root = _absolute_run_root(
            rows[index].get("run_root"),
            label=f"alpha {alpha:.2f} result",
        )
        _require(
            consumer == row_root,
            f"alpha {alpha:.2f} consumer identity/result root mismatch",
        )
        consumer_roots[alpha] = consumer
    return producer_roots, consumer_roots


def verify_oracle_artifacts(output_dir: Path | str) -> dict[str, Any]:
    """Recompute the bundle from its live Q0/Q3 and alpha run identities."""

    output = Path(output_dir).expanduser().resolve()
    _require(output.is_dir(), f"artifact directory missing: {output}")
    for name in _REQUIRED_ARTIFACTS:
        _require((output / name).is_file(), f"required artifact missing: {name}")

    source_path = output / "oracle_source_manifest.json"
    csv_path = output / "oracle_step_metrics.csv"
    summary_path = output / "oracle_headroom_summary.json"
    blend_path = output / "oracle_blend_response.json"
    source = _read_json(source_path)
    summary = _read_json(summary_path)
    blend = _read_json(blend_path)
    _verify_self_sha256(source, label="oracle source manifest")
    _verify_self_sha256(summary, label="oracle headroom summary")
    _verify_self_sha256(blend, label="oracle blend response")
    _require(
        summary.get("oracle_source_manifest_sha256") == _sha256_file(source_path),
        "summary/source-manifest SHA mismatch",
    )
    _require(
        summary.get("oracle_step_metrics_sha256") == _sha256_file(csv_path),
        "summary/step-metrics SHA mismatch",
    )
    _require(
        blend.get("headroom_summary_self_sha256") == summary.get("self_sha256"),
        "blend/headroom summary identity mismatch",
    )

    q0_root = _identity_root(source.get("q0"), label="Q0")
    q3_root = _identity_root(source.get("q3"), label="Q3")
    q0 = _load_run(q0_root, expected_mode="carry_forward")
    q3 = _load_run(q3_root, expected_mode="oracle_replay")
    _validate_pair(q0, q3)
    analysis = analyze_oracle_headroom(q0_root, q3_root)

    expected_source = _source_manifest_payload(analysis, q0, q3)
    _require(
        source == expected_source,
        "recomputed source manifest does not match the stored bundle",
    )
    expected_csv = _step_csv_bytes(analysis["steps"])
    try:
        actual_csv = csv_path.read_bytes()
    except OSError as exc:
        raise OracleHeadroomContractError(
            f"cannot read step metrics: {csv_path}"
        ) from exc
    _require(
        actual_csv == expected_csv,
        "recomputed step metrics do not match the stored CSV",
    )
    expected_summary = _summary_payload(
        analysis,
        source_manifest_sha256=_sha256_file(source_path),
        step_metrics_sha256=_sha256_file(csv_path),
    )
    _require(
        summary == expected_summary,
        "recomputed headroom summary does not match the stored bundle",
    )

    status = blend.get("status")
    if status in {"REQUIRED_PENDING", "NOT_RUN_ORACLE_GATE_FAILED"}:
        expected_blend = _initial_blend_response(analysis, expected_summary)
    else:
        _require(
            status in {"COMPLETED", "FAILED_HEALTH"},
            "oracle blend response status is invalid",
        )
        producers, consumers = _completed_blend_inputs(
            blend,
            q0_root=q0_root,
            q3_root=q3_root,
        )
        expected_blend = _completed_blend_response(
            q0_root=q0_root,
            q3_root=q3_root,
            blend_producers=producers,
            blend_runs=consumers,
            headroom_summary=expected_summary,
        )
    _require(
        blend == expected_blend,
        "consumer identity or recomputed blend response mismatch",
    )

    return {
        "classification": expected_summary.get("classification"),
        "blend_status": expected_blend.get("status"),
        "row_count": EXPECTED_STEPS,
        "artifact_sha256": {
            name: _sha256_file(output / name) for name in _REQUIRED_ARTIFACTS
        },
    }
