"""Explicit host-only codec for typed accepted-step checkpoint state."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
import re
from typing import Any, Mapping

import numpy as np


_SCHEMA_VERSION = 1
_ARRAY_NAME_PATTERN = re.compile(r"a[0-9]{6}")
_UNICODE_DTYPE_PATTERN = re.compile(r"[<>]U[1-9][0-9]*")


@dataclass(frozen=True)
class EncodedCheckpointState:
    """JSON-safe metadata plus finite, readonly NPZ-ready arrays."""

    metadata: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]


class CheckpointStateCodec:
    """Encode only explicitly whitelisted dataclass state without pickle."""

    def __init__(
        self,
        types: Mapping[str, type],
        *,
        allow_nonfinite_scalars: bool = False,
        allow_unicode_scalars: bool = False,
    ) -> None:
        if not isinstance(types, Mapping) or not types:
            raise ValueError("types must be a non-empty mapping")
        by_id: dict[str, type] = {}
        by_type: dict[type, str] = {}
        for class_id, state_type in types.items():
            if not isinstance(class_id, str) or not class_id:
                raise ValueError("checkpoint dataclass ids must be non-empty strings")
            if (
                not isinstance(state_type, type)
                or not is_dataclass(state_type)
                or state_type in by_type
            ):
                raise TypeError(
                    "checkpoint types must be unique dataclass classes"
                )
            by_id[class_id] = state_type
            by_type[state_type] = class_id
        self._types_by_id = by_id
        self._ids_by_type = by_type
        self._allow_nonfinite_scalars = bool(allow_nonfinite_scalars)
        self._allow_unicode_scalars = bool(allow_unicode_scalars)

    def encode(self, value: object) -> EncodedCheckpointState:
        """Encode a whitelisted state tree and copy every array immutably."""

        arrays: dict[str, np.ndarray] = {}
        active: set[int] = set()
        state = self._encode_node(value, arrays, active)
        return EncodedCheckpointState(
            metadata={"schema_version": _SCHEMA_VERSION, "state": state},
            arrays=arrays,
        )

    def decode(
        self,
        metadata: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
    ) -> object:
        """Validate the complete tree before calling any dataclass constructor."""

        if not isinstance(metadata, Mapping) or set(metadata) != {
            "schema_version",
            "state",
        }:
            raise ValueError("checkpoint metadata has an unknown or missing field")
        if (
            isinstance(metadata["schema_version"], bool)
            or metadata["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported checkpoint codec schema")
        checked_arrays = self._checked_arrays(arrays)
        referenced_arrays: list[str] = []
        validated = self._validate_node(metadata["state"], referenced_arrays)
        if set(referenced_arrays) != set(checked_arrays):
            raise ValueError("checkpoint array references do not exactly match arrays")
        return self._construct_node(validated, checked_arrays)

    def _encode_node(
        self,
        value: object,
        arrays: dict[str, np.ndarray],
        active: set[int],
    ) -> dict[str, object]:
        if value is None:
            return {"tag": "none"}
        if isinstance(value, (bool, np.bool_)):
            return {"tag": "bool", "value": bool(value)}
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return {"tag": "int", "value": int(value)}
        if isinstance(value, (float, np.floating)):
            return self._encode_float(float(value))
        if isinstance(value, str):
            return {"tag": "string", "value": value}
        if isinstance(value, np.ndarray):
            if self._allow_unicode_scalars and value.ndim == 0 and value.dtype.kind == "U":
                dtype = value.dtype.str
                encoding = "utf-32-le" if dtype[0] == "<" else "utf-32-be"
                # Preserve fixed-width NUL padding and byte order as JSON text;
                # the physical NPZ payload remains finite numeric arrays only.
                node = {
                    "tag": "unicode_scalar", "dtype": dtype,
                    "value": value.tobytes().decode(encoding),
                }
                self._validate_unicode_scalar(node)
                return node
            array_name = f"a{len(arrays) + 1:06d}"
            arrays[array_name] = self._readonly_array_copy(value)
            return {"tag": "ndarray", "array": array_name}

        tracked = isinstance(value, (tuple, list, Mapping)) or type(value) in self._ids_by_type
        if tracked:
            object_id = id(value)
            if object_id in active:
                raise ValueError("checkpoint state contains a cycle")
            active.add(object_id)
        try:
            if isinstance(value, tuple):
                return {
                    "tag": "tuple",
                    "items": [
                        self._encode_node(item, arrays, active) for item in value
                    ],
                }
            if isinstance(value, list):
                return {
                    "tag": "list",
                    "items": [
                        self._encode_node(item, arrays, active) for item in value
                    ],
                }
            if isinstance(value, Mapping):
                items: list[dict[str, object]] = []
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("checkpoint mappings require string keys")
                    items.append(
                        {
                            "key": key,
                            "value": self._encode_node(item, arrays, active),
                        }
                    )
                return {"tag": "mapping", "items": items}
            class_id = self._ids_by_type.get(type(value))
            if class_id is not None:
                encoded_fields: list[dict[str, object]] = []
                for field in fields(value):
                    if field.init:
                        encoded_fields.append(
                            {
                                "name": field.name,
                                "value": self._encode_node(
                                    getattr(value, field.name), arrays, active
                                ),
                            }
                        )
                return {
                    "tag": "dataclass",
                    "type": class_id,
                    "fields": encoded_fields,
                }
        finally:
            if tracked:
                active.remove(id(value))
        raise TypeError(f"unsupported checkpoint state value {type(value)!r}")

    def _encode_float(self, value: float) -> dict[str, object]:
        if math.isfinite(value):
            return {"tag": "float", "value": value}
        if not self._allow_nonfinite_scalars:
            raise ValueError("checkpoint scalar floats must be finite")
        if math.isnan(value):
            encoded = "nan"
        elif value > 0.0:
            encoded = "+inf"
        else:
            encoded = "-inf"
        return {"tag": "nonfinite", "value": encoded}

    def _validate_unicode_scalar(self, node: Mapping[str, object]) -> tuple[str, bytes]:
        if not self._allow_unicode_scalars:
            raise ValueError("Unicode scalar metadata is not permitted")
        dtype, value = node["dtype"], node["value"]
        if not isinstance(dtype, str) or _UNICODE_DTYPE_PATTERN.fullmatch(dtype) is None:
            raise ValueError("Unicode scalar dtype must have positive width and explicit byte order")
        if not isinstance(value, str):
            raise TypeError("Unicode scalar metadata must contain text")
        if dtype[2:] != str(len(value)):
            raise ValueError("Unicode scalar dtype width differs from its complete text payload")
        encoding = "utf-32-le" if dtype[0] == "<" else "utf-32-be"
        # Validate before any constructor. The dtype cannot advertise a larger
        # allocation than the supplied text; padding is already in that text.
        return dtype, value.encode(encoding)

    @staticmethod
    def _readonly_array_copy(value: object) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError("checkpoint arrays must have boolean or real numeric dtype")
        if not bool(np.isfinite(array).all()):
            raise ValueError("checkpoint arrays must be finite")
        original_shape = array.shape
        contiguous = np.ascontiguousarray(array)
        copied = np.frombuffer(
            contiguous.tobytes(order="C"),
            dtype=contiguous.dtype,
        ).reshape(original_shape)
        return copied

    def _checked_arrays(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not isinstance(arrays, Mapping):
            raise TypeError("checkpoint arrays must be a mapping")
        checked: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            if not isinstance(name, str) or _ARRAY_NAME_PATTERN.fullmatch(name) is None:
                raise ValueError("checkpoint array name is invalid")
            if name in checked:
                raise ValueError(f"duplicate checkpoint array {name!r}")
            checked[name] = self._readonly_array_copy(value)
        return checked

    def _validate_node(
        self,
        node: object,
        referenced_arrays: list[str],
    ) -> tuple[str, object]:
        if not isinstance(node, Mapping):
            raise TypeError("checkpoint state node must be a tagged mapping")
        tag = node.get("tag")
        if not isinstance(tag, str):
            raise ValueError("checkpoint state node tag is invalid")
        scalar_tags = {
            "none": set(),
            "bool": {"value"},
            "int": {"value"},
            "float": {"value"},
            "string": {"value"},
            "nonfinite": {"value"},
            "ndarray": {"array"},
            "unicode_scalar": {"dtype", "value"},
            "tuple": {"items"},
            "list": {"items"},
            "mapping": {"items"},
            "dataclass": {"type", "fields"},
        }
        expected = scalar_tags.get(tag)
        if expected is None or set(node) != {"tag", *expected}:
            raise ValueError("checkpoint state node has an unknown or missing field")
        if tag == "none":
            return (tag, None)
        if tag == "bool":
            if not isinstance(node["value"], bool):
                raise TypeError("bool node must contain bool")
            return (tag, node["value"])
        if tag == "int":
            if isinstance(node["value"], bool) or not isinstance(node["value"], int):
                raise TypeError("int node must contain int")
            return (tag, int(node["value"]))
        if tag == "float":
            value = node["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("float node must contain a number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("float node must be finite")
            return (tag, value)
        if tag == "string":
            if not isinstance(node["value"], str):
                raise TypeError("string node must contain string")
            return (tag, node["value"])
        if tag == "nonfinite":
            if not self._allow_nonfinite_scalars or node["value"] not in {
                "nan",
                "+inf",
                "-inf",
            }:
                raise ValueError("nonfinite scalar node is not permitted")
            return (tag, node["value"])
        if tag == "ndarray":
            name = node["array"]
            if not isinstance(name, str) or _ARRAY_NAME_PATTERN.fullmatch(name) is None:
                raise ValueError("ndarray node references an invalid array")
            referenced_arrays.append(name)
            return (tag, name)
        if tag == "unicode_scalar":
            return (tag, self._validate_unicode_scalar(node))
        if tag in {"tuple", "list"}:
            items = node["items"]
            if not isinstance(items, (list, tuple)):
                raise TypeError("sequence node items must be a JSON array")
            return (tag, [self._validate_node(item, referenced_arrays) for item in items])
        if tag == "mapping":
            items = node["items"]
            if not isinstance(items, (list, tuple)):
                raise TypeError("mapping node items must be a JSON array")
            pairs: list[tuple[str, tuple[str, object]]] = []
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, Mapping) or set(item) != {"key", "value"}:
                    raise ValueError("mapping entry has an unknown or missing field")
                key = item["key"]
                if not isinstance(key, str) or key in seen:
                    raise ValueError("mapping keys must be unique strings")
                seen.add(key)
                pairs.append((key, self._validate_node(item["value"], referenced_arrays)))
            return (tag, pairs)

        class_id = node["type"]
        if not isinstance(class_id, str) or class_id not in self._types_by_id:
            raise ValueError("dataclass node references an unknown type")
        field_nodes = node["fields"]
        if not isinstance(field_nodes, (list, tuple)):
            raise TypeError("dataclass fields must be a JSON array")
        state_type = self._types_by_id[class_id]
        expected_names = {field.name for field in fields(state_type) if field.init}
        values: dict[str, tuple[str, object]] = {}
        for field_node in field_nodes:
            if not isinstance(field_node, Mapping) or set(field_node) != {
                "name",
                "value",
            }:
                raise ValueError("dataclass field has an unknown or missing field")
            name = field_node["name"]
            if not isinstance(name, str) or name in values:
                raise ValueError("dataclass field names must be unique strings")
            values[name] = self._validate_node(field_node["value"], referenced_arrays)
        if set(values) != expected_names:
            raise ValueError("dataclass fields do not match its init fields")
        return (tag, (state_type, values))

    def _construct_node(
        self,
        validated: tuple[str, object],
        arrays: Mapping[str, np.ndarray],
    ) -> object:
        tag, payload = validated
        if tag in {"none", "bool", "int", "float", "string"}:
            return payload
        if tag == "nonfinite":
            return {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}[
                payload
            ]
        if tag == "ndarray":
            return self._readonly_array_copy(arrays[payload])
        if tag == "unicode_scalar":
            dtype, buffer = payload
            return np.frombuffer(buffer, dtype=dtype).reshape(())
        if tag == "tuple":
            return tuple(self._construct_node(item, arrays) for item in payload)
        if tag == "list":
            return [self._construct_node(item, arrays) for item in payload]
        if tag == "mapping":
            return {
                key: self._construct_node(value, arrays) for key, value in payload
            }
        if tag == "dataclass":
            state_type, values = payload
            return state_type(
                **{
                    name: self._construct_node(value, arrays)
                    for name, value in values.items()
                }
            )
        raise AssertionError(f"unhandled checkpoint node tag {tag!r}")
