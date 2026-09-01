from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import seal_ansys_vertical_flap_r24c as seal_cli
from tools.validation import r24c_post_publication as subject


HEAD = "a" * 40


def test_production_source_map_contract_constants_are_exact_literals() -> None:
    assert subject.EXPECTED_SOURCE_COUNT == 139
    assert subject.EXPECTED_SOURCE_MAP_SHA256 == (
        "a14a313568d86f6773c8fcbb2d5b1611e833389eb7455272554ae2e78d566b00"
    )
    assert subject.EXPECTED_Q0_COMPACT_REPORT_SHA256 == {
        "omega_0_50": "a1e8cc0dcd2dee73b33ded7d9e808ce09f0eb4b8ee51d769166e9da65b93c69e",
        "omega_0_75": "feee24643817a0c0d3ee5e6fc9283534a5b31f404f565adb7c4c5693a952fd81",
        "omega_1_00": "7b3db40d75d4f8e077e96e5570194ea5a10a07dd85d1e830c61ed016c1d77270",
    }


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _run(conclusion: str = "success", head: str = HEAD) -> dict[str, object]:
    return {
        "databaseId": 7,
        "workflowName": subject.WORKFLOW_NAME,
        "headSha": head,
        "conclusion": conclusion,
        "jobs": [
            {"name": "quality-and-fast-contracts", "conclusion": conclusion},
            {"name": "contracts", "conclusion": conclusion},
        ],
    }


def _contract(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    source_map = {"src/oracle_threshold_prefix_decisions.py": "f" * 64}
    source_sha = subject.source_map_sha256(source_map)
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_COUNT", 1)
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_MAP_SHA256", source_sha)
    artifacts = {
        "displacement": "1" * 64,
        "threshold_response": "2" * 64,
        "threshold_source_manifest": "3" * 64,
        "threshold_summary": "4" * 64,
        "iqn_reuse": "5" * 64,
        "legacy_projection": "6" * 64,
    }
    legacy_raw = {
        "displacement_evidence": artifacts["displacement"],
        "threshold_response": artifacts["threshold_response"],
        "threshold_source_manifest": artifacts["threshold_source_manifest"],
        "threshold_summary": artifacts["threshold_summary"],
    }
    legacy = {
        "schema_version": 2,
        "deployable": False,
        "bottom_up_reverification": False,
        "release": False,
        "release_recommendation": False,
        "raw_artifact_sha256": legacy_raw,
    }
    preflow = {name: (str(index) * 64) for index, name in enumerate(
        ("config_sha256", "source_sha256", "geometry_sha256", "manifest_sha256", "npz_sha256"), 1
    )}
    producer = {"execution_source": {
        "git_head_commit": "b" * 40,
        "mode": "source_map_bound_working_tree",
        "source_count": 1,
        "source_map_sha256": source_sha,
    }}
    runtime = {
        "requested_arch": "cuda",
        "actual_arch": "cuda",
        "strict_arch_verified": True,
        "default_fp": "f32",
        "random_seed": 0,
        "taichi_version": "1.7.4",
        "producer_python": {"recorded": False},
        "cuda_driver": {"recorded": False},
        "gpu": {"recorded": False},
        "q0_compact_report_sha256": copy.deepcopy(
            subject.EXPECTED_Q0_COMPACT_REPORT_SHA256
        ),
    }
    return {
        "legacy_projection": legacy,
        "bindings": {
            "artifact_sha256": artifacts,
            "source_map": source_map,
            "producer_identity": producer,
            "preflow_hashes": preflow,
        },
        "github": _run(),
        "head_commit": HEAD,
        "numerical_runtime": runtime,
        "attestation_host": {
            "python": {"recorded": True, "version": "3.10"},
            "taichi": {"recorded": True, "version": "1.7.4"},
            "cuda": {
                "recorded": True,
                "version": "12.8",
                "driver": "550.1",
            },
            "gpu": {
                "recorded": True,
                "name": "Synthetic GPU",
                "device": "GPU-uuid",
                "devices": ["Synthetic GPU [GPU-uuid]"],
            },
        },
    }


def _pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict, dict]:
    projection, attestation = subject.build_pair(**_contract(monkeypatch))
    projection_path = tmp_path / "projection.json"
    attestation_path = tmp_path / "attestation.json"
    source_map = attestation["attestation_core"]["source_map"]["source_sha256"]
    subject.write_pair(projection_path, attestation_path, projection, attestation, validated_source_map=source_map)
    return projection_path, attestation_path, projection, attestation


def test_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(subject.R24CPostPublicationError, match="duplicate"):
        subject.load_json_object(duplicate)
    nan_value = tmp_path / "nan.json"
    nan_value.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(subject.R24CPostPublicationError, match="non-finite"):
        subject.load_json_object(nan_value)
    with pytest.raises(subject.R24CPostPublicationError, match="non-finite"):
        subject.assert_portable({"x": float("nan")})
    with pytest.raises(subject.R24CPostPublicationError, match="credential"):
        subject.assert_portable({"oracle_threshold_prefix_decisions.py": "f" * 64})


@pytest.mark.parametrize("value", ["/tmp/x", r"\server\share", "file:///tmp/x", "../x", "a/../../x"])
def test_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(subject.R24CPostPublicationError):
        subject._safe_relative_path(value)
    with pytest.raises(subject.R24CPostPublicationError):
        subject.assert_portable({"value": value})


def test_github_requires_green_matching_head_and_both_jobs() -> None:
    with pytest.raises(subject.R24CPostPublicationError, match="matching success"):
        subject.verify_github_run(_run("failure"), HEAD)
    with pytest.raises(subject.R24CPostPublicationError, match="HEAD"):
        subject.verify_github_run(_run(head="b" * 40), HEAD)
    missing = _run()
    missing["jobs"] = [{"name": "contracts", "conclusion": "success"}]
    with pytest.raises(subject.R24CPostPublicationError, match="job"):
        subject.verify_github_run(missing, HEAD)
    identity = subject.verify_github_run(_run(), HEAD)
    assert identity["conclusion"] == "success"
    assert identity["jobs"] == {"quality-and-fast-contracts": "success", "contracts": "success"}


def test_full_source_map_is_byte_checked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_path = tmp_path / "src.py"
    source_path.write_bytes(b"source")
    source_map = {"src.py": subject.sha256_file(source_path)}
    source_sha = subject.source_map_sha256(source_map)
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_COUNT", 1)
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_MAP_SHA256", source_sha)
    monkeypatch.setattr(subject, "validate_complete_source_map",
                        lambda run: {"source_count": 1, "source_map_sha256": source_sha})
    manifest = {
        "execution_source": {
            "mode": "source_map_bound_working_tree",
            "git_head_commit": "b" * 40,
            "source_count": 1,
            "source_map_sha256": source_sha,
        },
        "source_sha256": source_map,
    }
    identity = subject.verify_source_map(tmp_path, manifest)
    assert identity == {"source_count": 1, "source_map_sha256": source_sha, "source_sha256": source_map}
    inconsistent = copy.deepcopy(manifest)
    inconsistent["source_count"] = 2
    with pytest.raises(subject.R24CPostPublicationError, match="disagrees"):
        subject.verify_source_map(tmp_path, inconsistent)
    source_path.write_bytes(b"drift")
    with pytest.raises(subject.R24CPostPublicationError, match="byte drift"):
        subject.verify_source_map(tmp_path, manifest)


def test_preflow_publishes_only_verified_hashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prefix = tmp_path / "preflow"
    state = prefix.with_name("preflow.json")
    npz = tmp_path / "step_fields.npz"
    state.write_bytes(b"state")
    npz.write_bytes(b"npz")
    identity = {"config_sha256": "1" * 64, "source_sha256": "2" * 64, "geometry_sha256": "3" * 64}
    artifact = {"metadata_file_sha256": subject.sha256_file(state),
                "manifest_sha256": "4" * 64, "npz_file": npz.name,
                "npz_sha256": subject.sha256_file(npz)}
    manifest = {"preflow_snapshot": {"prefix": str(prefix), "identity": identity,
                                     "artifact_identity": artifact}}
    from src.refactored.validation.ansys_vertical_flap_fsi import kalman_oracle_headroom_contracts
    monkeypatch.setattr(kalman_oracle_headroom_contracts, "_preflow_snapshot_identity",
                        lambda _: {"identity": identity, "artifact_identity": artifact})
    assert subject.verify_preflow_snapshot(manifest) == {
        "config_sha256": identity["config_sha256"], "source_sha256": identity["source_sha256"],
        "geometry_sha256": identity["geometry_sha256"], "manifest_sha256": artifact["metadata_file_sha256"],
        "npz_sha256": artifact["npz_sha256"],
    }
    npz.write_bytes(b"changed")
    with pytest.raises(subject.R24CPostPublicationError, match="NPZ hash"):
        subject.verify_preflow_snapshot(manifest)


def test_bottom_up_verifiers_and_artifact_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    threshold = tmp_path / "threshold"
    threshold.mkdir()
    displacement = tmp_path / "displacement.json"
    reuse = tmp_path / "reuse.json"
    source = threshold / "oracle_threshold_source_manifest.json"
    response = threshold / "oracle_threshold_response.json"
    summary = threshold / "oracle_threshold_summary.json"
    source_file = tmp_path / "src.py"
    source_file.write_bytes(b"source")
    source_map = {"src.py": subject.sha256_file(source_file)}
    source_map_sha = subject.source_map_sha256(source_map)
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_COUNT", 1)
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_MAP_SHA256", source_map_sha)
    manifest = {
        "execution_source": {
            "mode": "source_map_bound_working_tree",
            "git_head_commit": "b" * 40,
            "source_count": 1,
            "source_map_sha256": source_map_sha,
        },
        "source_sha256": source_map,
    }
    source.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(subject, "verify_source_map", lambda *_: None)
    for path, value in ((displacement, b"d"), (reuse, b"r"),
                        (response, b"q"), (summary, b"s")):
        path.write_bytes(value)
    raw = {
        "displacement_evidence": subject.sha256_file(displacement),
        "threshold_response": subject.sha256_file(response),
        "threshold_source_manifest": subject.sha256_file(source),
        "threshold_summary": subject.sha256_file(summary),
    }
    legacy = {"schema_version": 2, "deployable": False, "bottom_up_reverification": False,
              "raw_artifact_sha256": raw}
    legacy_path = tmp_path / "legacy.json"
    _json(legacy_path, legacy)
    ok = {"deployable": False, "bottom_up_reverification": True}
    monkeypatch.setattr(subject, "verify_displacement_evidence",
                        lambda _: {**ok, "classification": "PASS_ACCEPTED_DISPLACEMENT_AUDIT",
                                   "artifact_sha256": subject.sha256_file(displacement)})
    monkeypatch.setattr(subject, "verify_threshold_evidence",
                        lambda _: {**ok, "classification": "PASS_ORACLE_THRESHOLD_MATRIX",
                                   "artifact_sha256": {
                                       "oracle_threshold_response.json": subject.sha256_file(response),
                                       "oracle_threshold_source_manifest.json": subject.sha256_file(source),
                                       "oracle_threshold_summary.json": subject.sha256_file(summary),
                                   }})
    monkeypatch.setattr(subject, "verify_reuse_evidence",
                        lambda _: {**ok, "status": "reuse_matrix_authorized",
                                   "classification": "PASS_IQN_REUSE_FACTOR_MATRIX",
                                   "artifact_sha256": subject.sha256_file(reuse)})
    source_identity = manifest["execution_source"]
    artifacts = subject.verify_existing_evidence(
        displacement,
        threshold,
        reuse,
        legacy_path,
        source,
        tmp_path,
        source_identity,
    )
    assert set(artifacts) == {"displacement", "threshold_response", "threshold_source_manifest",
                              "threshold_summary", "iqn_reuse", "legacy_projection"}
    def drift(_: Path) -> dict[str, object]:
        response.write_bytes(b"drift")
        return {**ok, "classification": "PASS_ORACLE_THRESHOLD_MATRIX",
                "artifact_sha256": {}}
    monkeypatch.setattr(subject, "verify_threshold_evidence", drift)
    with pytest.raises(subject.R24CPostPublicationError, match="mutated"):
        subject.verify_existing_evidence(
            displacement,
            threshold,
            reuse,
            legacy_path,
            source,
            tmp_path,
            source_identity,
        )


def test_commit_bridge_covers_both_verifier_modules_and_restores_globals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_COUNT", 1)
    source_file = tmp_path / "src.py"
    source_file.write_bytes(b"source")
    source_map = {"src.py": subject.sha256_file(source_file)}
    source_map_sha = subject.source_map_sha256(source_map)
    threshold = tmp_path / "threshold"
    threshold.mkdir()
    displacement = tmp_path / "displacement.json"
    reuse = tmp_path / "reuse.json"
    source = threshold / "oracle_threshold_source_manifest.json"
    response = threshold / "oracle_threshold_response.json"
    summary = threshold / "oracle_threshold_summary.json"
    for path, value in ((displacement, b"d"), (reuse, b"r"),
                        (response, b"q"), (summary, b"s")):
        path.write_bytes(value)
    execution = {
        "mode": "source_map_bound_working_tree",
        "git_head_commit": "b" * 40,
        "source_count": 1,
        "source_map_sha256": source_map_sha,
    }
    source.write_text(json.dumps({
        "execution_source": execution,
        "source_sha256": source_map,
    }), encoding="utf-8")
    raw = {
        "displacement_evidence": subject.sha256_file(displacement),
        "threshold_response": subject.sha256_file(response),
        "threshold_source_manifest": subject.sha256_file(source),
        "threshold_summary": subject.sha256_file(summary),
    }
    legacy_path = tmp_path / "legacy.json"
    _json(legacy_path, {
        "schema_version": 2,
        "deployable": False,
        "bottom_up_reverification": False,
        "raw_artifact_sha256": raw,
    })
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_MAP_SHA256", source_map_sha)
    monkeypatch.setattr(subject, "verify_source_map", lambda *_: None)
    original_threshold = object()
    original_reuse = object()
    monkeypatch.setattr(subject._threshold_module,
                        "threshold_execution_source_identity", original_threshold)
    monkeypatch.setattr(subject._reuse_module,
                        "threshold_execution_source_identity", original_reuse)
    validate_calls: list[object] = []

    def validate(_: object) -> dict[str, object]:
        validate_calls.append(object())
        return {"source_count": 1, "source_map_sha256": source_map_sha}

    monkeypatch.setattr(subject._threshold_module,
                        "validate_complete_source_map", validate)
    monkeypatch.setattr(subject._reuse_module,
                        "validate_complete_source_map", validate)
    ok = {"bottom_up_reverification": True}
    bridge_results: list[tuple[str, str]] = []
    monkeypatch.setattr(subject, "verify_displacement_evidence", lambda _: {
        **ok,
        "classification": "PASS_ACCEPTED_DISPLACEMENT_AUDIT",
        "artifact_sha256": subject.sha256_file(displacement),
    })

    def threshold_fake(_: Path) -> dict[str, object]:
        identity = subject._threshold_module.threshold_execution_source_identity(
            SimpleNamespace()
        )
        bridge_results.append(("threshold", identity["git_head_commit"]))
        return {
            **ok,
            "classification": "PASS_ORACLE_THRESHOLD_MATRIX",
            "artifact_sha256": {
                "oracle_threshold_response.json": subject.sha256_file(response),
                "oracle_threshold_source_manifest.json": subject.sha256_file(source),
                "oracle_threshold_summary.json": subject.sha256_file(summary),
            },
        }

    def reuse_fake(_: Path) -> dict[str, object]:
        identity = subject._reuse_module.threshold_execution_source_identity(
            SimpleNamespace()
        )
        bridge_results.append(("reuse", identity["git_head_commit"]))
        return {
            **ok,
            "classification": "PASS_IQN_REUSE_FACTOR_MATRIX",
            "status": "reuse_matrix_authorized",
            "artifact_sha256": subject.sha256_file(reuse),
        }

    monkeypatch.setattr(subject, "verify_threshold_evidence", threshold_fake)
    monkeypatch.setattr(subject, "verify_reuse_evidence", reuse_fake)
    subject.verify_existing_evidence(
        displacement,
        threshold,
        reuse,
        legacy_path,
        source,
        tmp_path,
        execution,
    )
    assert bridge_results == [("threshold", "b" * 40), ("reuse", "b" * 40)]
    assert len(validate_calls) == 2
    assert subject._threshold_module.threshold_execution_source_identity is original_threshold
    assert subject._reuse_module.threshold_execution_source_identity is original_reuse


def test_numerical_runtime_consensus_records_absent_producers(tmp_path: Path) -> None:
    runtime = {"requested_arch": "cuda", "actual_arch": "cuda", "strict_arch_verified": True,
               "default_fp": "f32", "random_seed": 0,
               "compiler_configuration": {"taichi_version": "1.7.4"}}
    roots = []
    for index in range(3):
        path = tmp_path / f"q0-{index}.json"
        _json(
            path,
            {"taichi_runtime_identity": runtime, "run_index": index},
        )
        roots.append(path)
    consensus = subject.numerical_runtime_consensus(roots)
    assert consensus["producer_python"] == {"recorded": False}
    assert consensus["cuda_driver"] == {"recorded": False}
    assert consensus["gpu"] == {"recorded": False}


def test_pair_binds_full_core_and_separates_producer_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    projection_path, attestation_path, projection, attestation = _pair(tmp_path, monkeypatch)
    result = subject.verify_pair(projection_path, attestation_path)
    core = attestation["attestation_core"]
    assert result["projection_sha256"] == subject.sha256_file(projection_path)
    assert core["source_map"]["source_count"] == 1
    assert core["producer_identity"]["execution_source"]["git_head_commit"] == "b" * 40
    assert core["attestation_host"]["python"]["version"] == "3.10"
    assert core["head_commit"] == HEAD
    assert core["bottom_up_reverification"] is True
    assert projection["bottom_up_reverification"] is False
    assert "publication_projection_sha256" not in attestation
    assert core["github"]["conclusion"] == "success"
    assert subject.attestation_core_sha256(core) == attestation["attestation_core_sha256"]


def test_pair_rejects_legacy_mismatch_and_tampering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _contract(monkeypatch)
    args["legacy_projection"] = copy.deepcopy(args["legacy_projection"])
    args["legacy_projection"]["raw_artifact_sha256"]["threshold_summary"] = "0" * 64
    with pytest.raises(subject.R24CPostPublicationError, match="legacy"):
        subject.build_pair(**args)
    projection_path, attestation_path, _, _ = _pair(tmp_path, monkeypatch)
    original_projection = projection_path.read_bytes()
    original_attestation = attestation_path.read_bytes()
    projection = json.loads(original_projection)
    projection["post_publication"]["iqn_reuse_artifact_sha256"] = "0" * 64
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    with pytest.raises(subject.R24CPostPublicationError, match="projection"):
        subject.verify_pair(projection_path, attestation_path)
    projection_path.write_bytes(original_projection)
    attestation = json.loads(original_attestation)
    attestation["attestation_core"]["release"] = True
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(subject.R24CPostPublicationError, match="core SHA"):
        subject.verify_pair(projection_path, attestation_path)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (("schema", "core schema"), ("flags", "release flags"), ("host", "host")),
)
def test_pair_rejects_recomputed_core_and_projection_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    projection_path, attestation_path, _, _ = _pair(tmp_path, monkeypatch)
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    core = attestation["attestation_core"]
    if tamper == "schema":
        core["schema_version"] = 2
    elif tamper == "flags":
        core["release"] = True
    else:
        core["attestation_host"]["python"] = {"recorded": True}
    core_sha = subject.attestation_core_sha256(core)
    attestation["attestation_core_sha256"] = core_sha
    projection["post_publication"]["attestation_core_sha256"] = core_sha
    _json(projection_path, projection)
    attestation["publication_projection_sha256"] = subject.sha256_file(projection_path)
    _json(attestation_path, attestation)
    with pytest.raises(subject.R24CPostPublicationError, match=message):
        subject.verify_pair(projection_path, attestation_path)


def test_build_pair_rejects_empty_attestation_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _contract(monkeypatch)
    args["attestation_host"] = {}
    with pytest.raises(subject.R24CPostPublicationError, match="host"):
        subject.build_pair(**args)


def test_write_pair_preflights_distinct_and_existing_destinations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    projection, attestation = subject.build_pair(**_contract(monkeypatch))
    source_map = attestation["attestation_core"]["source_map"]["source_sha256"]
    with pytest.raises(subject.R24CPostPublicationError, match="identity"):
        subject.write_pair(tmp_path / "wrong-p.json", tmp_path / "wrong-a.json", projection, attestation, validated_source_map=dict(source_map))
    alias_paths = (tmp_path / "alias-p.json", tmp_path / "alias-a.json")
    aliased = {**attestation, "unexpected": source_map}
    with pytest.raises(subject.R24CPostPublicationError, match="credential"):
        subject.write_pair(alias_paths[0], alias_paths[1], projection, aliased, validated_source_map=source_map)
    assert not any(path.exists() for path in alias_paths)
    same = tmp_path / "same.json"
    with pytest.raises(subject.R24CPostPublicationError, match="differ"):
        subject.write_pair(same, same, projection, attestation, validated_source_map=source_map)
    subject.write_pair(same, tmp_path / "attestation.json", projection, attestation, validated_source_map=source_map)
    with pytest.raises(subject.R24CPostPublicationError, match="already exists"):
        subject.write_pair(same, tmp_path / "second.json", projection, attestation, validated_source_map=source_map)


def test_seal_host_identity_uses_exact_gpu_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def check_output(command: list[str], **_kwargs: object) -> str:
        calls.append(command)
        if command == ["nvidia-smi"]:
            return "NVIDIA-SMI 550.1\n| CUDA Version: 12.8 |\n"
        return "NVIDIA A100, GPU-uuid, 550.1\n"

    monkeypatch.setattr(
        seal_cli.importlib_metadata,
        "version",
        lambda name: "1.7.4" if name == "taichi" else "",
    )
    monkeypatch.setattr(seal_cli.platform, "python_version", lambda: "3.10.12")
    monkeypatch.setattr(seal_cli.subprocess, "check_output", check_output)
    identity = seal_cli._host_identity(subject)

    assert calls == [
        ["nvidia-smi"],
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
    ]
    assert all(identity[name]["recorded"] is True
               for name in ("python", "taichi", "cuda", "gpu"))
    assert identity["cuda"]["version"] == "12.8"
    assert identity["gpu"]["device"] == "GPU-uuid"


def test_cli_verify_does_not_probe_gpu_or_github(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    projection = tmp_path / "projection.json"
    attestation = tmp_path / "attestation.json"
    projection.write_text("{}", encoding="utf-8")
    attestation.write_text("{}", encoding="utf-8")
    calls: list[str] = []

    class FakeSeal:
        @staticmethod
        def verify_pair(projection_path: Path, attestation_path: Path) -> dict[str, str]:
            calls.extend([str(projection_path), str(attestation_path)])
            return {"projection_sha256": "1" * 64, "attestation_core_sha256": "2" * 64}

    def unexpected_probe(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("verify mode must not probe external services")

    monkeypatch.setattr(seal_cli, "_seal_module", lambda: FakeSeal())
    monkeypatch.setattr(seal_cli.subprocess, "check_output", unexpected_probe)
    assert seal_cli.main([
        "--verify",
        "--projection",
        str(projection),
        "--attestation",
        str(attestation),
    ]) == 0
    assert calls == [str(projection.resolve()), str(attestation.resolve())]
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


def _q0_runtime() -> dict[str, object]:
    return {
        "requested_arch": "cuda",
        "actual_arch": "cuda",
        "strict_arch_verified": True,
        "default_fp": "f32",
        "random_seed": 0,
        "compiler_configuration": {"taichi_version": "1.7.4"},
    }


def test_numerical_runtime_consensus_binds_q0_raw_sha_and_ignores_diagnostic_nan(
    tmp_path: Path,
) -> None:
    roots: dict[str, Path] = {}
    expected: dict[str, str] = {}
    labels = {
        "0.5": "omega_0_50",
        "0.75": "omega_0_75",
        "1.0": "omega_1_00",
    }
    for index, (root_label, binding_label) in enumerate(labels.items()):
        path = tmp_path / f"q0-{index}.json"
        _json(
            path,
            {
                "taichi_runtime_identity": _q0_runtime(),
                "diagnostic": float("nan"),
                "run_index": index,
            },
        )
        roots[root_label] = path
        expected[binding_label] = subject.sha256_file(path)
    consensus = subject.numerical_runtime_consensus(roots)
    assert consensus["q0_compact_report_sha256"] == expected


@pytest.mark.parametrize("nonfinite", [float("inf"), float("-inf")])
def test_numerical_runtime_consensus_rejects_infinity_outside_runtime(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    roots: dict[str, Path] = {}
    for index, root_label in enumerate(("0.5", "0.75", "1.0")):
        path = tmp_path / f"q0-{index}.json"
        _json(
            path,
            {
                "taichi_runtime_identity": _q0_runtime(),
                "diagnostic": nonfinite if root_label == "0.75" else index,
                "run_index": index,
            },
        )
        roots[root_label] = path
    with pytest.raises(subject.R24CPostPublicationError, match="non-finite"):
        subject.numerical_runtime_consensus(roots)


@pytest.mark.parametrize(
    "nonfinite",
    [float("nan"), float("inf"), float("-inf")],
)
def test_numerical_runtime_consensus_rejects_nonfinite_runtime_subtree(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    roots: dict[str, Path] = {}
    for index, root_label in enumerate(("0.5", "0.75", "1.0")):
        runtime = _q0_runtime()
        if root_label == "0.75":
            runtime["compiler_configuration"] = {
                "taichi_version": "1.7.4",
                "nested_diagnostic": nonfinite,
            }
        path = tmp_path / f"q0-{index}.json"
        _json(path, {"taichi_runtime_identity": runtime})
        roots[root_label] = path
    with pytest.raises(subject.R24CPostPublicationError, match="non-finite"):
        subject.numerical_runtime_consensus(roots)


def test_numerical_runtime_consensus_rejects_duplicate_runtime_key(
    tmp_path: Path,
) -> None:
    roots: dict[str, Path] = {}
    encoded_runtime = json.dumps(_q0_runtime(), sort_keys=True)
    for index, root_label in enumerate(("0.5", "0.75", "1.0")):
        path = tmp_path / f"q0-{index}.json"
        if root_label == "0.75":
            path.write_text(
                (
                    '{"taichi_runtime_identity": '
                    + encoded_runtime
                    + ', "taichi_runtime_identity": '
                    + encoded_runtime
                    + "}"
                ),
                encoding="utf-8",
            )
        else:
            _json(path, {"taichi_runtime_identity": _q0_runtime()})
        roots[root_label] = path
    with pytest.raises(subject.R24CPostPublicationError, match="duplicate"):
        subject.numerical_runtime_consensus(roots)


@pytest.mark.parametrize("invalid_runtime", [None, []])
def test_numerical_runtime_consensus_rejects_missing_or_nonobject_runtime(
    tmp_path: Path,
    invalid_runtime: object,
) -> None:
    roots: dict[str, Path] = {}
    for index, root_label in enumerate(("0.5", "0.75", "1.0")):
        path = tmp_path / f"q0-{index}.json"
        runtime = invalid_runtime if root_label == "0.75" else _q0_runtime()
        _json(
            path,
            {"taichi_runtime_identity": runtime, "run_index": index},
        )
        roots[root_label] = path
    with pytest.raises(
        subject.R24CPostPublicationError,
        match="runtime identity missing",
    ):
        subject.numerical_runtime_consensus(roots)


def test_numerical_runtime_consensus_rejects_duplicate_path_or_bytes(
    tmp_path: Path,
) -> None:
    roots: dict[str, Path] = {}
    for index, root_label in enumerate(("0.5", "0.75", "1.0")):
        path = tmp_path / f"q0-{index}.json"
        _json(path, {"taichi_runtime_identity": _q0_runtime()})
        roots[root_label] = path
    with pytest.raises(subject.R24CPostPublicationError, match="bytes"):
        subject.numerical_runtime_consensus(roots)

    roots["1.0"] = roots["0.75"]
    with pytest.raises(subject.R24CPostPublicationError, match="paths"):
        subject.numerical_runtime_consensus(roots)


def test_pair_requires_exact_q0_compact_report_hash_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_bindings = [
        {
            "omega_0_50": "a" * 64,
            "omega_0_75": "b" * 64,
        },
        {
            "omega_0_50": "a" * 64,
            "omega_0_75": "a" * 64,
            "omega_1_00": "c" * 64,
        },
        {
            **subject.EXPECTED_Q0_COMPACT_REPORT_SHA256,
            "omega_0_50": "0" * 64,
        },
    ]
    for binding in invalid_bindings:
        args = _contract(monkeypatch)
        args["numerical_runtime"]["q0_compact_report_sha256"] = binding
        with pytest.raises(
            subject.R24CPostPublicationError,
            match="Q0 compact report SHA",
        ):
            subject.build_pair(**args)

    args = _contract(monkeypatch)
    args["numerical_runtime"]["unexpected"] = True
    with pytest.raises(
        subject.R24CPostPublicationError,
        match="runtime identity",
    ):
        subject.build_pair(**args)


def test_pair_rejects_rehashed_q0_report_binding_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    projection_path, attestation_path, _, _ = _pair(tmp_path, monkeypatch)
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    core = attestation["attestation_core"]
    core["numerical_runtime"]["q0_compact_report_sha256"]["omega_0_50"] = (
        "0" * 64
    )
    core_sha = subject.attestation_core_sha256(core)
    attestation["attestation_core_sha256"] = core_sha
    projection["post_publication"]["attestation_core_sha256"] = core_sha
    _json(projection_path, projection)
    attestation["publication_projection_sha256"] = subject.sha256_file(
        projection_path
    )
    _json(attestation_path, attestation)
    with pytest.raises(
        subject.R24CPostPublicationError,
        match="Q0 compact report SHA",
    ):
        subject.verify_pair(projection_path, attestation_path)
