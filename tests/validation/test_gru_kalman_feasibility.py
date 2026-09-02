"""Focused synthetic contract tests for the frozen R25A harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.validation.gru_kalman.artifacts import (
    artifact_sha256,
    ensure_empty_output,
    freeze_selection,
    load_model_state_bundle,
    manifest_for_trace,
    save_model_state_bundle,
    verify_artifact_sha256,
    verify_selection_seal,
)
from tools.validation.gru_kalman.baselines import (
    K1_FINGERPRINT,
    K0_FINGERPRINT,
    exact_k1_candidate,
    exact_k0_candidate,
    evaluate_baseline,
    make_baseline_adapter,
)
from tools.validation.gru_kalman import campaign as campaign_module
from tools.validation.gru_kalman.dataset import (
    ACTIVE_AXES,
    D0_FIT_STEPS,
    D0_SELECTION_STEPS,
    D1_SCORE_STEPS,
    EXPECTED_DT_S,
    EXPECTED_LAYOUT_ID,
    AcceptedTrace,
    build_temporal_samples,
    split_d0_trace,
    validate_trace,
)
from tools.validation.gru_kalman.evaluation import (
    SeedMetrics,
    compute_metrics,
    evaluate_gate_boundaries,
    hybrid_gate,
    oracle_predictions,
)
from tools.validation.gru_kalman.models import (
    GRUArchitecture,
    build_gru,
    configure_deterministic_cpu,
)
from tools.validation.gru_kalman.pod import fit_pod, fit_normalization, fit_pod_ar
from tools.validation.gru_kalman.artifacts import SelectionSeal
from tools.validation.gru_kalman.training import (
    PreparedGRUData,
    TrainingConfig,
    fit_gru,
)
from tools.validation.gru_kalman.campaign import CampaignConfig


def _trace(count: int = 200, markers: int = 5) -> AcceptedTrace:
    steps = np.arange(count, dtype=np.float64)[:, None, None]
    marker = np.arange(markers, dtype=np.float64)[None, :, None]
    values = np.zeros((count, markers, 3), dtype=np.float64)
    values[..., 1] = 0.01 * steps[..., 0] + marker[..., 0]
    values[..., 2] = np.sin(0.1 * steps[..., 0]) + 0.1 * marker[..., 0]
    return AcceptedTrace.synthetic(
        values,
        name="synthetic",
        dt_s=EXPECTED_DT_S,
        layout_id=EXPECTED_LAYOUT_ID,
        source_fingerprint=hashlib.sha256(b"source").hexdigest(),
    )


def test_frozen_matrix_and_split_constants() -> None:
    assert D0_FIT_STEPS == tuple(range(1, 101))
    assert D0_SELECTION_STEPS == tuple(range(101, 201))
    assert D1_SCORE_STEPS == tuple(range(9, 51))
    assert ACTIVE_AXES == (False, True, True)
    assert EXPECTED_DT_S == 0.0005
    assert EXPECTED_LAYOUT_ID == "373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164"
    assert K0_FINGERPRINT == "383f9fc10475449cd88ce4fbc9b0d3b7595b47e62e7ef4aa53a516dd0058e03e"
    assert K1_FINGERPRINT == "603ec011922df847f61a0d8a91216ba2a2e3b2c60eb757092f910df37678d91e"


def test_k0_and_k1_candidate_payloads_are_frozen() -> None:
    k0 = exact_k0_candidate()
    k1 = exact_k1_candidate()
    assert k0.fingerprint == K0_FINGERPRINT
    assert k1.fingerprint == K1_FINGERPRINT
    assert k0.to_payload()["warmup_accepted_states"] == 6
    assert k0.to_payload()["model"] == "production"


def test_trace_validation_and_temporal_provenance() -> None:
    trace = _trace()
    validate_trace(trace, expected_steps=200)
    samples = build_temporal_samples(trace, window=4, start_step=5, end_step=8)
    assert samples[0].target_step == 5
    assert samples[0].source_steps == (1, 2, 3, 4)
    assert max(samples[0].source_steps) < samples[0].target_step
    with pytest.raises(ValueError, match="x axis"):
        bad = np.array(trace.values, copy=True)
        bad[0, 0, 0] = 1.0
        validate_trace(
            AcceptedTrace.synthetic(
                bad,
                name="bad",
                dt_s=EXPECTED_DT_S,
                layout_id=EXPECTED_LAYOUT_ID,
                source_fingerprint=hashlib.sha256(b"source").hexdigest(),
            ),
            expected_steps=200,
        )


def test_train_only_pod_and_normalization_mask_x() -> None:
    trace = _trace()
    fit, selection = split_d0_trace(trace)
    pod = fit_pod(fit.values, rank=4, fit_steps=fit.source_steps)
    normalization = fit_normalization(pod.encode(fit.values), fit_steps=fit.source_steps)
    changed = np.array(trace.values, copy=True)
    changed[100:] *= 1000.0
    pod_again = fit_pod(changed[:100], rank=4, fit_steps=range(1, 101))
    assert np.allclose(pod.mean, pod_again.mean)
    assert np.allclose(pod.basis, pod_again.basis)
    decoded = pod.decode(pod.encode(selection.values))
    assert np.all(decoded[..., 0] == 0.0)
    assert normalization.fit_max_step == 100


def test_baseline_warmup_and_transaction_rollback() -> None:
    trace = _trace(12, 2)
    for model in ("kalman0", "kalman1"):
        result = evaluate_baseline(trace, model=model)
        assert len(result.predictions) == 12
        assert all(np.all(row.effective_prediction[..., 0] == 0.0) for row in result.rows[:5])
        assert result.rows[5].ready
        assert np.all(result.rows[0].raw_prediction[..., 0] == 0.0)


def test_baseline_discard_replays_identical_raw_prediction() -> None:
    trace = _trace(8, 2)
    adapter = make_baseline_adapter("kalman1", (2, 3))
    first = adapter.begin_step(
        target_step=1,
        accepted_state_source_step=0,
        dt_s=EXPECTED_DT_S,
        layout_id=EXPECTED_LAYOUT_ID,
    )
    adapter.discard_step()
    second = adapter.begin_step(
        target_step=1,
        accepted_state_source_step=0,
        dt_s=EXPECTED_DT_S,
        layout_id=EXPECTED_LAYOUT_ID,
    )
    assert np.array_equal(first, second)
    adapter.accept_step(trace.values[0])


def test_current_and_future_values_do_not_change_causal_features() -> None:
    trace = _trace()
    reference = build_temporal_samples(trace, window=4, start_step=10, end_step=10)[0]
    changed = np.array(trace.values, copy=True)
    changed[9] += np.array([0.0, 100.0, -100.0])
    changed[10:] += np.array([0.0, 200.0, 300.0])
    altered = AcceptedTrace.synthetic(
        changed,
        name="changed",
        dt_s=EXPECTED_DT_S,
        layout_id=EXPECTED_LAYOUT_ID,
        source_fingerprint=hashlib.sha256(b"changed").hexdigest(),
    )
    candidate = build_temporal_samples(altered, window=4, start_step=10, end_step=10)[0]
    assert np.array_equal(reference.history, candidate.history)
    assert not np.array_equal(reference.target, candidate.target)


def test_d0_selection_can_use_fit_context_but_holdout_cannot_use_d0() -> None:
    trace = _trace()
    selection = build_temporal_samples(
        trace, window=4, start_step=101, end_step=101, allowed_history_max_step=100
    )[0]
    assert selection.source_steps == (97, 98, 99, 100)
    with pytest.raises(ValueError):
        build_temporal_samples(
            _trace(50), window=4, start_step=9, end_step=9,
            allowed_history_max_step=0,
        )


def test_pod_ar_fit_is_train_only_and_has_fixed_ridge() -> None:
    trace = _trace()
    pod = fit_pod(trace.values[:100], rank=4, fit_steps=range(1, 101))
    norm = fit_normalization(pod.encode(trace.values[:100]), fit_steps=range(1, 101))
    model = fit_pod_ar(norm.normalize(pod.encode(trace.values[:100])), rank=4, window=4, ridge=1e-6, fit_steps=range(1, 101))
    altered = np.array(trace.values[:100], copy=True)
    altered[90:] *= 99.0
    other = fit_pod_ar(norm.normalize(pod.encode(altered)), rank=4, window=4, ridge=1e-6, fit_steps=range(1, 101))
    assert np.array_equal(model.weights, other.weights) is False
    assert model.ridge == pytest.approx(1e-6)
    with pytest.raises(ValueError):
        fit_pod_ar(norm.normalize(pod.encode(trace.values)), rank=4, window=4, ridge=1e-6, fit_steps=range(1, 201))


def test_gru_is_one_layer_float64_deterministic_and_zero_residual() -> None:
    configure_deterministic_cpu()
    architecture = GRUArchitecture(rank=4, window=4, hidden=8)
    model = build_gru("gru", architecture, seed=0)
    assert model.gru.num_layers == 1
    assert model.gru.dropout == 0.0
    assert model.gru.bidirectional is False
    assert next(model.parameters()).dtype == __import__("torch").float64
    features = __import__("torch").zeros((2, 4, 4), dtype=__import__("torch").float64)
    carry = __import__("torch").ones((2, 4), dtype=__import__("torch").float64)
    assert __import__("torch").allclose(model(features, carry), carry)


def test_metrics_and_gate_boundaries_are_explicit() -> None:
    truth = np.zeros((3, 2, 3), dtype=np.float64)
    carry = np.zeros_like(truth)
    carry[..., 1:] = 1.0
    prediction = np.zeros_like(truth)
    prediction[..., 1:] = 0.95
    metrics = compute_metrics(
        prediction,
        truth,
        carry_prediction=carry,
        d0_train_axis_rms=np.ones(3),
        seed_metrics=(),
        score_start_step=1,
    )
    assert metrics.global_active_yz_nrmse == pytest.approx(0.95)
    assert evaluate_gate_boundaries(0.95, 0.98, 0.60, 1.0) is True
    assert np.array_equal(oracle_predictions(type("T", (), {"values": truth})()), truth)


def test_state_bundle_is_weights_only_and_hash_verified(tmp_path) -> None:
    torch = __import__("torch")
    model = build_gru("gru", GRUArchitecture(4, 4, 8), seed=0)
    path = tmp_path / "model_state.pt"
    save_model_state_bundle(
        path,
        {
            family: {seed: model.state_dict() for seed in (0, 1, 2)}
            for family in ("gru", "kalman0_gru", "kalman1_gru")
        },
    )
    payload = torch.load(path, weights_only=True)
    assert payload["schema_version"] == 1
    assert artifact_sha256(path) == verify_artifact_sha256(path, artifact_sha256(path))
    assert load_model_state_bundle(path)["families"] == ["gru", "kalman0_gru", "kalman1_gru"]
    ensure_empty_output(tmp_path / "new-output")
    with pytest.raises(ValueError):
        ensure_empty_output(tmp_path)


def test_d1_open_requires_selection_seal_and_occurs_at_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[str] = []
    trace = _trace(50, 2)

    def fake_loader(*args, **kwargs):
        events.append("d1_loader")
        return trace

    monkeypatch.setattr(campaign_module, "load_accepted_trace", fake_loader)
    with pytest.raises(ValueError):
        campaign_module.open_d1_holdout(None, "d0", "d1", artifact_paths={})
    with pytest.raises(ValueError, match="exact"):
        SelectionSeal("a" * 64, {})
    names = (
        "pod_basis.npz",
        "normalization.json",
        "model_config.json",
        "pod_ar_state.json",
        "training_history.csv",
        "selection_metrics.csv",
        "model_state.pt",
    )
    paths = {
        name: tmp_path / name
        for name in names
    }
    for path in paths.values():
        path.write_bytes(path.name.encode("utf-8"))
    fabricated = SelectionSeal("a" * 64, {name: "a" * 64 for name in names})
    with pytest.raises(ValueError):
        campaign_module.open_d1_holdout(
            fabricated, "d0", "d1", artifact_paths=paths
        )
    assert events == []
    seal = freeze_selection(paths, constants={"frozen": True})
    opened = campaign_module.open_d1_holdout(
        seal, "d0", "d1", artifact_paths=paths
    )
    assert opened is trace
    assert events == ["d1_loader"]


def test_open_d1_rehashes_sealed_artifacts_before_loader(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    trace = _trace(50, 2)

    def fake_loader(*args, **kwargs):
        events.append("d1_loader")
        return trace

    monkeypatch.setattr(campaign_module, "load_accepted_trace", fake_loader)
    names = (
        "pod_basis.npz",
        "normalization.json",
        "model_config.json",
        "pod_ar_state.json",
        "training_history.csv",
        "selection_metrics.csv",
        "model_state.pt",
    )
    paths = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path
    seal = freeze_selection(paths, constants={"frozen": True})
    paths["model_config.json"].write_bytes(b"mutated")
    with pytest.raises(ValueError):
        campaign_module.open_d1_holdout(
            seal, "d0", "d1", artifact_paths=paths
        )
    assert events == []


def test_campaign_rejects_report_equal_to_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="report"):
        CampaignConfig(output_root=output, report_path=output)


def test_campaign_rejects_report_inside_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="report"):
        CampaignConfig(output_root=output, report_path=output / "report.md")


def test_campaign_rejects_report_symlink_alias_to_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="report"):
        CampaignConfig(output_root=output, report_path=alias / "report.md")


def _gate_stub(
    nrmse: float,
    carry_p95: float,
    paired_p95: float,
    fraction: float = 0.60,
    seed: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        nrmse=nrmse,
        fraction_beating_paired=fraction,
        metrics=SimpleNamespace(
            rho_p95=carry_p95,
            paired_rho_p95=paired_p95,
        ),
    )


def test_hybrid_gate_compares_median_carry_relative_p95_to_paired_g0() -> None:
    g0 = tuple(_gate_stub(1.0, 1.0, 0.5, seed=seed) for seed in (0, 1, 2))
    gk = tuple(_gate_stub(0.80, 1.20, 0.50, seed=seed) for seed in (0, 1, 2))
    assert hybrid_gate(
        gk,
        g0,
        matching_kalman_nrmse=1.0,
    ) is False


def test_hybrid_gate_requires_paired_step_evidence() -> None:
    g0 = tuple(_gate_stub(1.0, 1.0, 0.5, seed=seed) for seed in (0, 1, 2))
    gk = [_gate_stub(0.80, 0.90, 0.50, seed=seed) for seed in (0, 1, 2)]
    gk[1].fraction_beating_paired = None
    assert hybrid_gate(gk, g0, matching_kalman_nrmse=1.0) is False


def test_hybrid_gate_requires_ordered_unique_r25a_seeds() -> None:
    g0 = tuple(_gate_stub(1.0, 1.0, 0.5, seed=seed) for seed in (0, 1, 2))
    duplicate = tuple(_gate_stub(0.80, 0.90, 0.50, seed=seed) for seed in (0, 0, 2))
    reversed_rows = tuple(_gate_stub(0.80, 0.90, 0.50, seed=seed) for seed in (2, 1, 0))
    assert hybrid_gate(duplicate, g0, matching_kalman_nrmse=1.0) is False
    assert hybrid_gate(reversed_rows, g0, matching_kalman_nrmse=1.0) is False


def test_fit_gru_requires_d0_selection_data() -> None:
    gru = build_gru("gru", GRUArchitecture(4, 4, 8), seed=0)
    data = PreparedGRUData(
        features=np.zeros((1, 4, 4)),
        carry=np.zeros((1, 4)),
        target=np.zeros((1, 4)),
        target_steps=(5,),
        source_steps=((1, 2, 3, 4),),
    )
    with pytest.raises(ValueError, match="selection"):
        fit_gru(
            "gru",
            GRUArchitecture(4, 4, 8),
            seed=0,
            train=data,
            selection=None,
            config=TrainingConfig(max_epochs=1, patience=1),
        )


def test_campaign_rejects_nonfrozen_training_constants() -> None:
    with pytest.raises(ValueError, match="training"):
        CampaignConfig(training=TrainingConfig(max_epochs=1))


@pytest.mark.parametrize(
    "drift",
    ("k0", "k1", "predictor", "d0_identity", "d1_identity", "step_evidence"),
)
def test_sealed_manifest_rejects_locked_identity_drift(
    tmp_path: Path,
    monkeypatch,
    drift: str,
) -> None:
    source = Path(campaign_module.SEALED_MANIFEST_PATH)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if drift == "k0":
        next(
            item for item in payload["candidate_matrix"] if item["candidate_id"] == "K0"
        )["locked_config"]["warmup_accepted_states"] = 7
    elif drift == "k1":
        next(
            item for item in payload["candidate_matrix"] if item["candidate_id"] == "K1"
        )["q_xyz"][1] += 1.0
    elif drift == "predictor":
        payload["source_compatibility"]["executed_predictor_source_sha256"] = "0" * 64
    elif drift == "d0_identity":
        payload["D0"]["source_fingerprint"] = "0" * 64
    elif drift == "d1_identity":
        payload["D1"]["source_fingerprint"] = "0" * 64
    else:
        payload["D0"]["step_evidence"][0]["step_fields_sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        campaign_module,
        "SEALED_MANIFEST_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError):
        campaign_module.verify_sealed_manifest(path)


def test_loaded_trace_metadata_is_bound_to_sealed_manifest() -> None:
    trace = _trace(4, 2)
    sealed = manifest_for_trace(trace, role="D0")
    assert campaign_module.validate_trace_against_manifest(trace, sealed, role="D0") is trace
    r24_schema = dict(sealed)
    r24_schema.pop("role")
    r24_schema["completed_attempt_root"] = r24_schema.pop("attempt_root")
    assert (
        campaign_module.validate_trace_against_manifest(
            trace, r24_schema, role="D0"
        )
        is trace
    )
    changed = dict(sealed)
    changed["step_evidence"] = [dict(row) for row in sealed["step_evidence"]]
    changed["step_evidence"][0]["step_fields_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        campaign_module.validate_trace_against_manifest(trace, changed, role="D0")


def test_metric_rows_include_global_and_per_step_evidence() -> None:
    truth = np.zeros((3, 2, 3), dtype=np.float64)
    carry = np.zeros_like(truth)
    carry[..., 1:] = 1.0
    prediction = np.zeros_like(truth)
    prediction[..., 1:] = 0.5
    metric = compute_metrics(
        prediction,
        truth,
        carry_prediction=carry,
        d0_train_axis_rms=np.ones(3),
        score_start_step=1,
    )
    summary = campaign_module._metric_rows({"Q": metric})[0]
    per_step = campaign_module._proxy_rows({"Q": metric})
    assert {"axis_rmse_x", "axis_rmse_y", "axis_rmse_z"} <= set(summary)
    assert {"axis_bias_x", "axis_bias_y", "axis_bias_z"} <= set(summary)
    assert {"global_marker_p95", "global_marker_max"} <= set(summary)
    assert {
        "per_step_rms",
        "marker_p95",
        "marker_max",
        "rho",
        "alpha_parallel",
        "r_perp",
    } <= set(per_step[0])
    assert summary["model"] == "Q"


def test_runtime_identity_binds_current_harness_sources() -> None:
    identity = campaign_module.runtime_identity(Path(".").resolve())
    assert identity["python_version"]
    assert identity["numpy_version"]
    assert identity["pytorch_version"]
    assert identity["cuda_available"] is False
    assert identity["cuda_version"] is None
    assert identity["cpu_only"] is True
    assert len(identity["base_commit"]) == 40
    assert all(character in "0123456789abcdef" for character in identity["base_commit"])
    assert isinstance(identity["working_tree_dirty"], bool)
    expected_dirty_state = (
        "implementation_files_uncommitted"
        if identity["working_tree_dirty"]
        else "clean"
    )
    assert identity["dirty_state"] == expected_dirty_state
    assert (
        identity["base_commit_is_not_implementation_identity"]
        is identity["working_tree_dirty"]
    )
    assert set(identity["harness_source_sha256"]) == set(
        campaign_module.R25A_HARNESS_SOURCE_FILES
    )
    assert all(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for digest in identity["harness_source_sha256"].values()
    )


def test_report_binds_historical_runtime_identity_to_r25a_commit() -> None:
    report = Path(
        "docs/validation/"
        "ANSYS_VERTICAL_FLAP_GRU_KALMAN_FEASIBILITY_REPORT_2026-09-02.md"
    ).read_text(encoding="utf-8")
    encoded = report.split("RUNTIME_IDENTITY_JSON:\n", 1)[1].split(
        "\nEND_RUNTIME_IDENTITY_JSON", 1
    )[0]
    historical = json.loads(encoded)
    assert historical["base_commit"] == "fbf4b729a68fab4c69316568cadcf46f234202d9"
    assert historical["working_tree_dirty"] is True
    assert historical["dirty_state"] == "implementation_files_uncommitted"
    r25a_commit = "adb2a0470085ecca1f772bae14d292df76c963d9"
    committed_hashes = {}
    for relative in campaign_module.R25A_HARNESS_SOURCE_FILES:
        blob = subprocess.run(
            ["git", "show", f"{r25a_commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        committed_hashes[relative] = hashlib.sha256(blob).hexdigest()
    assert len(committed_hashes) == 11
    assert historical["harness_source_sha256"] == committed_hashes


def test_nonempty_report_is_refused_before_training(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "report.md"
    report.write_text("existing", encoding="utf-8")
    config = CampaignConfig(output_root=tmp_path / "out", report_path=report)
    monkeypatch.setattr(
        campaign_module,
        "verify_sealed_manifest",
        lambda *_: pytest.fail("manifest checked before report refusal"),
    )
    with pytest.raises(ValueError, match="report"):
        campaign_module.run_campaign(config)


def test_write_report_wrapper_succeeds_with_synthetic_metrics(tmp_path: Path) -> None:
    truth = np.zeros((3, 2, 3), dtype=np.float64)
    carry = np.zeros_like(truth)
    carry[..., 1:] = 1.0
    metric = compute_metrics(
        carry,
        truth,
        carry_prediction=carry,
        d0_train_axis_rms=np.ones(3),
        score_start_step=1,
    )
    names = (
        "pod_basis.npz",
        "normalization.json",
        "model_config.json",
        "pod_ar_state.json",
        "training_history.csv",
        "selection_metrics.csv",
        "model_state.pt",
    )
    hashes = {name: "a" * 64 for name in names}
    seal = SelectionSeal("a" * 64, hashes)
    result = SimpleNamespace(
        selection_seal=seal,
        d1_metrics={"C0": metric},
        classifications={"G0": "FAIL_OFFLINE_GRU_VALUE", "GK0": "FAIL_OFFLINE_KALMAN_GRU_VALUE", "GK1": "FAIL_OFFLINE_KALMAN_GRU_VALUE", "overall": "FAIL_OFFLINE_GRU_AND_KALMAN_GRU_VALUE"},
    )
    trace = SimpleNamespace(values=truth)
    report = tmp_path / "report.md"
    campaign_module._write_report(
        report,
        result=result,
        d0=trace,
        d1=trace,
        selected_architectures={"gru": "S1"},
        runtime_identity={"base_commit": "synthetic"},
        selection_artifact_hashes=hashes,
        artifact_hashes={"d1_holdout_metrics.csv": "b" * 64},
    )
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# ANSYS Vertical Flap R25A")
    assert "| C0 |" in text
    assert "rho median" in text
    assert "synthetic" in text


def test_selection_seal_rehash_rejects_post_seal_mutation(tmp_path: Path) -> None:
    names = (
        "pod_basis.npz",
        "normalization.json",
        "model_config.json",
        "pod_ar_state.json",
        "training_history.csv",
        "selection_metrics.csv",
        "model_state.pt",
    )
    paths = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path
    seal = freeze_selection(paths, constants={"frozen": True})
    paths["model_config.json"].write_bytes(b"mutated")
    with pytest.raises(ValueError):
        verify_selection_seal(seal, paths)


def test_direct_cli_help_bootstraps_repository_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(repo_root / "tools" / "run_ansys_vertical_flap_gru_study.py"),
            "--help",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "R25A" in completed.stdout
