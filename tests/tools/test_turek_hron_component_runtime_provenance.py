from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.validation import run_turek_hron_component_gates as runner
from tools.validation import turek_hron_component_gate_contracts as contracts
from tools.validation import turek_hron_component_gate_runtimes as runtimes


RUNTIME_IDENTITY = {
    "requested_arch": "cuda",
    "actual_arch": "cuda",
    "default_fp": "f32",
    "random_seed": 0,
    "compiler_configuration": {
        "taichi_version": "1.7.4",
        "default_ip": "i32",
        "cfg_optimization": True,
        "opt_level": 1,
        "advanced_optimization": True,
        "fast_math": False,
        "debug": False,
    },
    "offline_cache_identity": {"enabled": True, "file_path": "/cache/a"},
    "strict_arch_verified": True,
}


def test_runtime_identity_validation_and_numerical_comparison_contract():
    accepted = contracts.validate_taichi_runtime_identity(RUNTIME_IDENTITY)
    cache_variant = deepcopy(RUNTIME_IDENTITY)
    cache_variant["offline_cache_identity"] = {
        "enabled": False,
        "file_path": "/cache/b",
    }
    assert contracts.numerical_taichi_runtime_identity(accepted) == (
        contracts.numerical_taichi_runtime_identity(cache_variant)
    )
    compiler_variant = deepcopy(RUNTIME_IDENTITY)
    compiler_variant["compiler_configuration"]["fast_math"] = True
    assert contracts.numerical_taichi_runtime_identity(accepted) != (
        contracts.numerical_taichi_runtime_identity(compiler_variant)
    )
    invalid = deepcopy(RUNTIME_IDENTITY)
    invalid["actual_arch"] = "cpu"
    with pytest.raises(ValueError, match="FAIL_TAICHI_RUNTIME_IDENTITY"):
        contracts.validate_taichi_runtime_identity(invalid)


def test_runtime_constructor_rejects_callable_returning_prebuilt_instance():
    effective = {"mode": "solid-only"}

    class Runtime:
        effective_arch = "cuda"

        def __init__(self, config):
            self.effective_config = config

    assert isinstance(runtimes.construct_component_runtime(Runtime, effective), Runtime)
    class WrongArch(Runtime):
        effective_arch = "cpu"
    with pytest.raises(ValueError, match="FAIL_EFFECTIVE_ARCH"):
        runtimes.construct_component_runtime(WrongArch, effective)
    prebuilt = Runtime(effective)
    with pytest.raises(ValueError, match="FAIL_RUNTIME_TYPE"):
        runtimes.construct_component_runtime(lambda _: prebuilt, effective)


def test_runtime_identity_is_hashed_in_summary_manifest_and_read_back(tmp_path: Path):
    run_dir = runner._claim_output_dir(tmp_path, "identity")
    result = runner._write_artifacts(
        run_dir,
        "identity",
        {"mode": "synthetic"},
        [{"value": 1}],
        {"value": np.asarray([1])},
        {"status": "PASS_COMPONENT_ONLY", "completed": True, "metrics": {}},
        taichi_runtime_identity=RUNTIME_IDENTITY,
    )
    manifest = result["manifest"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["taichi_runtime_identity"] == RUNTIME_IDENTITY
    assert manifest["taichi_runtime_identity"] == RUNTIME_IDENTITY
    assert manifest["taichi_runtime_identity_sha256"] == runner.hashlib.sha256(
        runner._canonical(RUNTIME_IDENTITY)
    ).hexdigest()
    runner._read_completed(run_dir)

    manifest["taichi_runtime_identity"]["actual_arch"] = "cpu"
    manifest["taichi_runtime_identity_sha256"] = runner.hashlib.sha256(
        runner._canonical(manifest["taichi_runtime_identity"])
    ).hexdigest()
    (run_dir / "run_manifest.json").write_bytes(runner._canonical(manifest) + b"\n")
    with pytest.raises(ValueError, match="FAIL_TAICHI_RUNTIME_IDENTITY"):
        runner._read_completed(run_dir)


@pytest.mark.parametrize(
    ("tamper", "failure"),
    (
        ("manifest", "FAIL_TAICHI_RUNTIME_IDENTITY_MODE"),
        ("summary", "FAIL_TAICHI_RUNTIME_IDENTITY_MISMATCH"),
        ("hash", "FAIL_TAICHI_RUNTIME_IDENTITY_HASH"),
    ),
)
def test_comparison_runtime_identity_rejects_tampering(
    tmp_path: Path,
    tamper: str,
    failure: str,
):
    run_dir = runner._claim_output_dir(tmp_path, tamper)
    result = runner._write_artifacts(
        run_dir,
        tamper,
        {"mode": "compare"},
        [{"relative_vector_delta": 0.0}],
        {"comparison_vector": np.asarray([0.0])},
        {"status": "PASS_COMPONENT_ONLY", "completed": True, "metrics": {}},
    )
    manifest = result["manifest"]
    if tamper == "manifest":
        manifest["taichi_runtime_identity"] = "tampered"
        manifest["taichi_runtime_identity_sha256"] = runner.hashlib.sha256(
            runner._canonical("tampered")
        ).hexdigest()
    elif tamper == "summary":
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["taichi_runtime_identity"] = "tampered"
        summary_path.write_bytes(runner._canonical(summary) + b"\n")
        manifest["artifacts"]["summary.json"] = runner._sha_file(summary_path)
    else:
        manifest["taichi_runtime_identity_sha256"] = "tampered"
    (run_dir / "run_manifest.json").write_bytes(
        runner._canonical(manifest) + b"\n"
    )

    with pytest.raises(ValueError, match=failure):
        runner._read_completed(run_dir)
