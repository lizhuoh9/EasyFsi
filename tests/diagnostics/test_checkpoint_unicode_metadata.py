"""Diagnostic text stays tagged JSON; physical NPZ arrays stay numeric."""

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from simulation_core.diagnostics.checkpoint_codec import CheckpointStateCodec
from simulation_core.diagnostics.checkpoint_store import (
    append_history, load_checkpoint, save_checkpoint,
)


@dataclass(frozen=True)
class DiagnosticState:
    payload: object
    calls: ClassVar[int] = 0

    def __post_init__(self):
        type(self).calls += 1


def _codec():
    return CheckpointStateCodec(
        {"diagnostic": DiagnosticState}, allow_unicode_scalars=True,
    )


@pytest.mark.parametrize("source", [
    np.asarray("pre_solid_projection"),
    np.asarray("post_solid_observer"),
    np.asarray("a" * 64),
    np.asarray("阶段😀"),
    np.asarray(""),
    np.asarray("a\x00b", dtype="<U9"),
    np.asarray("stage", dtype=">U12"),
])
def test_unicode_scalar_preserves_exact_bytes_through_numeric_only_store(
    tmp_path: Path, source: np.ndarray,
):
    codec = _codec()
    state = DiagnosticState({
        "stage": source, "pressure": np.asarray([1.25], dtype=np.float64),
    })
    encoded = codec.encode(state)
    assert all(array.dtype.kind in "biuf" for array in encoded.arrays.values())
    prefix = tmp_path / "accepted"
    tail = append_history(prefix, step=1, record={"step": 1})
    save_checkpoint(
        prefix, accepted_step=1, metadata=encoded.metadata,
        arrays=encoded.arrays, history_tail=tail,
    )
    loaded = load_checkpoint(prefix)
    restored = codec.decode(loaded.metadata, loaded.arrays).payload["stage"]
    assert restored.shape == ()
    assert restored.dtype == source.dtype
    assert restored.tobytes() == source.tobytes()
    assert restored.item() == source.item()
    assert not restored.flags.writeable
    with pytest.raises(ValueError):
        restored.setflags(write=True)
    source[...] = "changed"
    assert restored.tobytes() != source.tobytes()


def test_unicode_metadata_is_explicitly_opt_in_on_encode_and_decode():
    strict = CheckpointStateCodec({"diagnostic": DiagnosticState})
    source = np.asarray("stage")

    with pytest.raises(TypeError, match="numeric"):
        strict.encode(source)
    encoded = _codec().encode(source)
    with pytest.raises(ValueError, match="Unicode|unicode"):
        strict.decode(encoded.metadata, encoded.arrays)


@pytest.mark.parametrize("source", [
    np.asarray(["stage"]),
    np.asarray([["stage"]]),
    np.asarray("stage", dtype=object),
    np.asarray(b"stage"),
    np.asarray(("stage",), dtype=[("stage", "U5")]),
    np.asarray([float("nan")]),
    np.ndarray((), dtype="<U0"),
    np.asarray("\ud800"),
])
def test_text_opt_in_does_not_admit_other_array_kinds(source):
    codec = _codec()
    with pytest.raises((TypeError, ValueError)):
        codec.encode(source)


@pytest.mark.parametrize("damage", [
    {"dtype": "<U999999999999999999999"},
    {"dtype": "<U4"},
    {"dtype": "<U05"},
    {"dtype": "<U0"},
    {"dtype": "O"},
    {"dtype": "<f8"},
    {"dtype": ["<U5"]},
    {"value": 5},
    {"value": "\ud800abcd"},
    {"shape": []},
])
def test_malformed_text_validates_before_any_dataclass_constructor(damage):
    codec = _codec()
    encoded = codec.encode((DiagnosticState(1), np.asarray("stage")))
    metadata = copy.deepcopy(encoded.metadata)
    metadata["state"]["items"][1].update(damage)
    DiagnosticState.calls = 0
    with pytest.raises((TypeError, ValueError)):
        codec.decode(metadata, encoded.arrays)
    assert DiagnosticState.calls == 0


def test_unicode_cannot_be_smuggled_into_numeric_npz_payload():
    codec = _codec()
    encoded = codec.encode(np.asarray([1.0]))
    arrays = {"a000001": np.asarray("stage")}
    with pytest.raises(TypeError, match="numeric"):
        codec.decode(encoded.metadata, arrays)
