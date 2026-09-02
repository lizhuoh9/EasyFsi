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


def test_solid_runtime_uses_production_particle_mass_field():
    class Field:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def to_numpy(self):
            return self.value.copy()

        def from_numpy(self, value):
            self.value = np.asarray(value, dtype=np.float32).copy()

    class Solid:
        particle_count = 2
        mass_kg = Field([2.0, 3.0, 99.0])
        external_force_n = Field(np.zeros((2, 3), dtype=np.float32))

    runtime = object.__new__(runtimes.SolidOnlyRuntime)
    runtime.solid = Solid()
    runtime._acceleration = np.asarray([0.0, 0.01, 0.0], dtype=np.float64)
    runtime._apply_mass_proportional_force()

    np.testing.assert_allclose(
        runtime.solid.external_force_n.to_numpy(),
        [[0.0, 0.02, 0.0], [0.0, 0.03, 0.0]],
        rtol=1.0e-6,
        atol=0.0,
    )


def test_external_y_face_residual_reads_production_ledger():
    mask = np.full((2, 3, 4), 7, dtype=np.int32)
    ledger = np.zeros((2, 3, 4, 3), dtype=np.float32)
    expected = np.zeros((2, 3, 4, 3), dtype=np.float64)
    residual = runtimes.FixedFluidRuntime._external_y_face_ledger_residual

    assert residual(mask, ledger, expected) == 0.0
    ledger[1, 2, 2, 0] = 1.0e-3
    assert residual(mask, ledger, expected) == pytest.approx(1.0e-3)
    ledger[1, 2, 2, 0] = np.nan
    assert np.isinf(residual(mask, ledger, expected))
    ledger[1, 2, 2, 0] = 0.0
    assert np.isinf(residual(mask, ledger, expected[..., :2]))
    mask[0, 0, 0] = 3
    assert np.isinf(residual(mask, ledger, expected))


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
