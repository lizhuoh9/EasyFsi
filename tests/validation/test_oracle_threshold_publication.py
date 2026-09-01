"""Portable publication projection contracts for R24C."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_common import (
    OracleThresholdContractError,
)


def test_publication_projection_selects_metrics_without_local_paths() -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_publication as subject,
    )

    displacement = {
        "classification": "PASS_ACCEPTED_DISPLACEMENT_AUDIT",
        "deployable": False,
        "q0_root": "/home/private/q0",
        "q3_root": "C:\\Users\\private\\q3",
        "source_validation": {
            "mode": "immutable_git_commit",
            "commit": "b" * 40,
            "repo_root": "/home/private/repo",
            "source_count": 129,
            "source_map_sha256": "1" * 64,
        },
        "identity": {
            "physical_marker_count_per_face": 64,
            "physical_marker_count_total": 128,
            "interface_state_row_count": 128,
        },
        "thresholds": {"displacement_nrmse_max": 5.0e-3},
        "gates": {"accepted_displacement_contract": True},
        "aggregate": {
            "marker_displacement_nrmse_max": 6.8e-6,
            "solid_displacement_nrmse_max": 5.3e-6,
        },
    }
    threshold = {
        "campaign": "ansys_vertical_flap_oracle_threshold_iqn_first_update_r24c",
        "classification": "PASS_ORACLE_THRESHOLD_MATRIX",
        "deployable": False,
        "identity": {
            "target_steps": [2, 5, 8],
            "omegas": [0.5, 0.75, 1.0],
            "alphas": [0.9, 0.99, 1.0],
        },
        "gates": {"complete_fail_closed_matrix": True},
        "arms": [
            {
                "omega": 0.5,
                "target_step": 2,
                "carry_iterations": 3,
                "alpha_3_to_2": 0.99,
                "alpha_2_to_1": 1.0,
                "rows": [{"local_path": "/home/private/raw"}],
            }
        ],
        "omega_summary": [{"omega": 0.5, "safe": True}],
        "best_safe_omega": 0.5,
        "reuse_branch": {
            "authorized": True,
            "status": "reuse_matrix_authorized",
            "reason": "best_safe_threshold_at_or_below_0.9900",
        },
        "predictor_decision": "threshold_supports_first_update_mechanism_follow_up",
    }
    raw_sha = {
        "displacement_evidence": "2" * 64,
        "threshold_source_manifest": "3" * 64,
        "threshold_summary": "4" * 64,
        "threshold_response": "5" * 64,
    }
    threshold_source_identity = {
        "mode": "source_map_bound_working_tree",
        "git_head_commit": "c" * 40,
        "source_count": 141,
        "source_map_sha256": "6" * 64,
    }

    projection = subject.build_publication_projection(
        displacement,
        threshold,
        threshold_source_identity=threshold_source_identity,
        raw_artifact_sha256=raw_sha,
    )

    encoded = json.dumps(projection, sort_keys=True)
    assert projection["deployable"] is False
    assert projection["bottom_up_reverification"] is False
    assert "source_commit" not in projection
    assert projection["source_identities"] == {
        "displacement_producer": {
            "mode": "immutable_git_commit",
            "commit": "b" * 40,
            "source_count": 129,
            "source_map_sha256": "1" * 64,
        },
        "threshold_producer": threshold_source_identity,
    }
    assert projection["raw_artifact_sha256"] == raw_sha
    assert projection["logical_arms"] == [
        {
            "arm_id": "omega050_step02",
            "omega": 0.5,
            "target_step": 2,
            "carry_iterations": 3,
            "alpha_3_to_2": 0.99,
            "alpha_2_to_1": 1.0,
        }
    ]
    assert "/home/private" not in encoded
    assert "C:\\\\Users" not in encoded
    subject.assert_portable_projection(projection)


def test_publication_raw_artifact_sha_requires_exact_logical_keys() -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_publication as subject,
    )

    expected = {
        "displacement_evidence": "1" * 64,
        "threshold_response": "2" * 64,
        "threshold_source_manifest": "3" * 64,
        "threshold_summary": "4" * 64,
    }
    assert subject._sha_map(expected) == expected

    with pytest.raises(OracleThresholdContractError, match="exactly four logical"):
        subject._sha_map({**expected, "oracle_threshold_response.json": "5" * 64})
    with pytest.raises(OracleThresholdContractError, match="exactly four logical"):
        subject._sha_map(
            {
                key: value
                for key, value in expected.items()
                if key != "threshold_summary"
            }
        )


def test_write_publication_projection_uses_logical_keys_when_basenames_collide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_publication as subject,
    )

    displacement = tmp_path / "oracle_threshold_response.json"
    threshold = tmp_path / "threshold"
    threshold.mkdir()
    response = threshold / "oracle_threshold_response.json"
    manifest = threshold / "oracle_threshold_source_manifest.json"
    summary = threshold / "oracle_threshold_summary.json"
    displacement.write_text('{"result": {}}\n', encoding="utf-8")
    response.write_text("{}\n", encoding="utf-8")
    manifest.write_text('{"execution_source": {}}\n', encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(subject, "verify_displacement_evidence", lambda _path: {})
    monkeypatch.setattr(subject, "verify_threshold_evidence", lambda _root: {})
    captured: dict[str, str] = {}

    def fake_build(
        _displacement: object,
        _threshold: object,
        *,
        threshold_source_identity: object,
        raw_artifact_sha256: dict[str, str],
    ) -> dict[str, object]:
        del threshold_source_identity
        captured.update(raw_artifact_sha256)
        return {}

    monkeypatch.setattr(subject, "build_publication_projection", fake_build)
    output = tmp_path / "publication.json"

    subject.write_publication_projection(displacement, threshold, output)

    assert set(captured) == {
        "displacement_evidence",
        "threshold_response",
        "threshold_source_manifest",
        "threshold_summary",
    }
    assert len(captured) == 4
    assert output.is_file()


@pytest.mark.parametrize(
    "value",
    (
        "/home/private/evidence",
        "C:\\Users\\private\\evidence",
        "ghp_123456789012345678901234567890123456",
        "https://user:password@example.invalid/private",
        "prefix /home/private/evidence",
        "line one\nC:\\Users\\private\\evidence",
        "https://tokenonly@example.invalid/private",
        "sk-proj-123456789012345678901234567890",
        "gho_123456789012345678901234567890123456",
        "Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123",
        "https://example.invalid/?access_token=secret-value",
        "-----BEGIN PRIVATE KEY-----",
    ),
)
def test_publication_projection_rejects_paths_and_credentials(value: str) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_publication as subject,
    )

    with pytest.raises(OracleThresholdContractError, match="portable"):
        subject.assert_portable_projection({"unsafe": value})


def test_publication_snapshot_rejects_input_change_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_publication as subject,
    )

    displacement = tmp_path / "accepted_displacement_metrics.json"
    threshold = tmp_path / "threshold"
    threshold.mkdir()
    response = threshold / "oracle_threshold_response.json"
    manifest = threshold / "oracle_threshold_source_manifest.json"
    summary = threshold / "oracle_threshold_summary.json"
    for path in (displacement, response, manifest, summary):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        subject,
        "verify_displacement_evidence",
        lambda _path: {},
    )

    def mutate(_root: Path) -> dict[str, object]:
        response.write_text('{"tampered": true}\n', encoding="utf-8")
        return {}

    monkeypatch.setattr(subject, "verify_threshold_evidence", mutate)

    with pytest.raises(
        OracleThresholdContractError,
        match="changed during verification",
    ):
        subject._verified_input_snapshot(displacement, threshold)
