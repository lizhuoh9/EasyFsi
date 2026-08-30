"""Host-only tests for the explicit typed checkpoint state codec."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from simulation_core.diagnostics.checkpoint_codec import (
    CheckpointStateCodec,
    EncodedCheckpointState,
)
from simulation_core.diagnostics.checkpoint_store import (
    append_history,
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class ChildState:
    label: str
    samples: tuple[int, ...]


@dataclass(frozen=True)
class RootState:
    child: ChildState
    matrix: np.ndarray
    settings: tuple[object, ...]
    records: list[object]


@dataclass(frozen=True)
class UnregisteredState:
    value: int


@dataclass(frozen=True)
class ProbeState:
    value: int

    calls: ClassVar[int] = 0

    def __post_init__(self) -> None:
        type(self).calls += 1


def _codec(*, allow_nonfinite_scalars: bool = False) -> CheckpointStateCodec:
    return CheckpointStateCodec(
        {"child": ChildState, "root": RootState, "probe": ProbeState},
        allow_nonfinite_scalars=allow_nonfinite_scalars,
    )


def _state() -> RootState:
    return RootState(
        child=ChildState("inner", (1, 2, 3)),
        matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        settings=("stable", {"enabled": True, "gain": np.float64(0.5)}),
        records=[None, np.int64(7), np.asarray([True, False], dtype=np.bool_)],
    )


def test_round_trip_nested_frozen_dataclasses_and_arrays_are_isolated() -> None:
    source = _state()
    codec = _codec()

    encoded = codec.encode(source)
    decoded = codec.decode(encoded.metadata, encoded.arrays)

    assert isinstance(encoded, EncodedCheckpointState)
    assert encoded.metadata["schema_version"] == 1
    assert tuple(encoded.arrays) == ("a000001", "a000002")
    assert isinstance(decoded, RootState)
    assert decoded.child == source.child
    assert decoded.settings == ("stable", {"enabled": True, "gain": 0.5})
    assert decoded.records[:2] == [None, 7]
    assert np.array_equal(decoded.matrix, source.matrix)
    assert decoded.matrix.flags.writeable is False
    assert decoded.records[2].flags.writeable is False

    source.matrix[0, 0] = 99.0
    assert decoded.matrix[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        decoded.matrix[0, 0] = 8.0


def test_encode_rejects_unknown_classes_and_invalid_whitelists() -> None:
    codec = _codec()

    with pytest.raises(TypeError):
        codec.encode(UnregisteredState(3))
    with pytest.raises((TypeError, ValueError)):
        CheckpointStateCodec({"not-a-class": int})
    with pytest.raises((TypeError, ValueError)):
        CheckpointStateCodec({"one": ChildState, "two": ChildState})


def test_decode_rejects_schema_unknown_tags_exact_keys_and_fields() -> None:
    codec = _codec()
    encoded = codec.encode(_state())

    unknown_schema = copy.deepcopy(encoded.metadata)
    unknown_schema["schema_version"] = 2
    with pytest.raises((TypeError, ValueError)):
        codec.decode(unknown_schema, encoded.arrays)

    unknown_tag = copy.deepcopy(encoded.metadata)
    unknown_tag["state"] = {"tag": "surprise"}
    with pytest.raises((TypeError, ValueError)):
        codec.decode(unknown_tag, encoded.arrays)

    bad_keys = copy.deepcopy(encoded.metadata)
    bad_keys["state"]["extra"] = True
    with pytest.raises((TypeError, ValueError)):
        codec.decode(bad_keys, encoded.arrays)

    duplicate_field = {
        "schema_version": 1,
        "state": {
            "tag": "dataclass",
            "type": "child",
            "fields": [
                {"name": "label", "value": {"tag": "string", "value": "x"}},
                {"name": "label", "value": {"tag": "string", "value": "y"}},
                {"name": "samples", "value": {"tag": "tuple", "items": []}},
            ],
        },
    }
    with pytest.raises((TypeError, ValueError)):
        codec.decode(duplicate_field, {})

    missing_array = copy.deepcopy(encoded.metadata)
    missing_array["state"]["fields"][1]["value"] = {
        "tag": "ndarray",
        "array": "a999999",
    }
    with pytest.raises((TypeError, ValueError)):
        codec.decode(missing_array, encoded.arrays)


def test_decode_validates_whole_tree_before_any_dataclass_constructor() -> None:
    ProbeState.calls = 0
    codec = _codec()
    malformed = {
        "schema_version": 1,
        "state": {
            "tag": "dataclass",
            "type": "probe",
            "fields": [
                {"name": "value", "value": {"tag": "int", "value": 1}},
                {"name": "unexpected", "value": {"tag": "int", "value": 2}},
            ],
        },
    }

    with pytest.raises((TypeError, ValueError)):
        codec.decode(malformed, {})

    assert ProbeState.calls == 0


def test_nonfinite_scalars_require_opt_in_but_arrays_never_allow_them() -> None:
    strict = _codec()
    with pytest.raises(ValueError):
        strict.encode(float("nan"))

    permissive = _codec(allow_nonfinite_scalars=True)
    encoded = permissive.encode((float("nan"), float("inf"), float("-inf")))
    assert "NaN" not in repr(encoded.metadata)
    decoded = permissive.decode(encoded.metadata, encoded.arrays)
    assert math.isnan(decoded[0])
    assert decoded[1] == float("inf")
    assert decoded[2] == float("-inf")

    with pytest.raises(ValueError):
        permissive.encode(np.asarray([np.nan], dtype=np.float32))


@pytest.mark.parametrize(
    "array",
    (
        np.asarray([{"bad": 1}], dtype=object),
        np.asarray(["bad"], dtype=str),
        np.asarray([(1, 2)], dtype=[("x", "i4"), ("y", "i4")]),
        np.asarray([np.inf], dtype=np.float64),
    ),
)
def test_arrays_always_require_finite_plain_numeric_or_bool(array: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        _codec(allow_nonfinite_scalars=True).encode(array)


def test_decode_rejects_unknown_array_extras_and_repeated_refs_are_copies() -> None:
    codec = _codec()
    encoded = codec.encode(np.asarray([1.0, 2.0], dtype=np.float32))
    bad_arrays = dict(encoded.arrays)
    bad_arrays["a000002"] = np.asarray([3.0], dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        codec.decode(encoded.metadata, bad_arrays)

    repeated = {
        "schema_version": 1,
        "state": {
            "tag": "tuple",
            "items": [
                {"tag": "ndarray", "array": "a000001"},
                {"tag": "ndarray", "array": "a000001"},
            ],
        },
    }
    decoded = codec.decode(repeated, encoded.arrays)
    assert np.array_equal(decoded[0], decoded[1])
    assert decoded[0] is not decoded[1]
    assert decoded[0].flags.writeable is False


def test_encode_rejects_cycles_and_decode_rejects_nonfinite_array_payload() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError):
        _codec().encode(cyclic)

    encoded = _codec().encode(np.asarray([1.0], dtype=np.float32))
    bad_arrays = {"a000001": np.asarray([np.nan], dtype=np.float32)}
    with pytest.raises((TypeError, ValueError)):
        _codec().decode(encoded.metadata, bad_arrays)


@pytest.mark.parametrize(
    "source",
    (
        np.asarray(1.25, dtype=np.float32),
        np.asarray(7, dtype=np.int64),
        np.asarray(True, dtype=np.bool_),
    ),
)
def test_zero_dimensional_arrays_survive_codec_store_round_trip(
    tmp_path: Path,
    source: np.ndarray,
) -> None:
    codec = _codec()
    encoded = codec.encode(source)
    prefix = tmp_path / "state"
    tail = append_history(
        prefix,
        step=1,
        record={"accepted_step": 1},
    )
    save_checkpoint(
        prefix,
        accepted_step=1,
        metadata=encoded.metadata,
        arrays=encoded.arrays,
        history_tail=tail,
    )

    restored = load_checkpoint(prefix)
    decoded = codec.decode(restored.metadata, restored.arrays)

    assert decoded.shape == ()
    assert decoded.dtype == source.dtype
    assert decoded.item() == source.item()
    assert decoded.flags.writeable is False
