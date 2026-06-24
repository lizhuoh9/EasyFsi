from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StepBrep:
    index: int
    step_id: int
    name: str

    def as_dict(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "step_id": int(self.step_id),
            "name": str(self.name),
        }


@dataclass(frozen=True)
class StepCadSummary:
    path: str
    sha256: str
    file_schema: tuple[str, ...]
    length_unit: str
    breps: tuple[StepBrep, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "file_schema": list(self.file_schema),
            "length_unit": self.length_unit,
            "brep_count": len(self.breps),
            "brep_names": [brep.name for brep in self.breps],
            "breps": [brep.as_dict() for brep in self.breps],
        }


_BREP_RE = re.compile(
    r"#(?P<step_id>\d+)\s*=\s*MANIFOLD_SOLID_BREP\s*\(\s*'(?P<name>(?:''|[^'])*)'",
    re.IGNORECASE,
)
_FILE_SCHEMA_RE = re.compile(r"FILE_SCHEMA\s*\(\s*\((?P<body>.*?)\)\s*\)", re.IGNORECASE | re.DOTALL)
_QUOTED_RE = re.compile(r"'((?:''|[^'])*)'")
_SI_UNIT_RE = re.compile(
    r"SI_UNIT\s*\(\s*(?P<prefix>\.[A-Z]+\.|\$)\s*,\s*(?P<unit>\.[A-Z]+\.|\$)\s*\)",
    re.IGNORECASE,
)


def _step_string(value: str) -> str:
    return value.replace("''", "'")


def _file_schema(text: str) -> tuple[str, ...]:
    match = _FILE_SCHEMA_RE.search(text)
    if match is None:
        return ()
    return tuple(_step_string(item) for item in _QUOTED_RE.findall(match.group("body")))


def _length_unit(text: str) -> str:
    for match in _SI_UNIT_RE.finditer(text):
        prefix = match.group("prefix").strip(".").lower()
        unit = match.group("unit").strip(".").lower()
        if unit == "metre":
            if prefix == "milli":
                return "millimetre"
            if prefix in {"$", ""}:
                return "metre"
            return f"{prefix}metre"
    return "unknown"


def parse_step_cad_summary(path: str | Path) -> StepCadSummary:
    step_path = Path(path).resolve()
    payload = step_path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    breps = tuple(
        StepBrep(
            index=index,
            step_id=int(match.group("step_id")),
            name=_step_string(match.group("name")),
        )
        for index, match in enumerate(_BREP_RE.finditer(text), start=1)
    )
    return StepCadSummary(
        path=str(step_path),
        sha256=hashlib.sha256(payload).hexdigest(),
        file_schema=_file_schema(text),
        length_unit=_length_unit(text),
        breps=breps,
    )


def _source_config_path_value(source_config: Mapping[str, object], key: str) -> object:
    value = source_config.get(key)
    if value is not None:
        return value
    analysis = source_config.get("analysis_settings", {})
    if isinstance(analysis, Mapping):
        value = analysis.get(key)
        if value is not None:
            return value
    domains = source_config.get("domains", {})
    if isinstance(domains, Mapping):
        for domain in domains.values():
            if isinstance(domain, Mapping):
                value = domain.get(key)
                if value is not None:
                    return value
    return None


def _source_config_nested_path_values(
    source_config: Mapping[str, object],
) -> tuple[object, ...]:
    values: list[object] = []
    for container_key in ("metadata", "mesh_import"):
        container = source_config.get(container_key, {})
        if not isinstance(container, Mapping):
            continue
        for key in ("source_step", "source_step_path", "step_path"):
            value = container.get(key)
            if value is not None:
                values.append(value)
    return tuple(values)


def _source_config_mesh_import_value(
    source_config: Mapping[str, object],
    key: str,
) -> object:
    container = source_config.get("mesh_import", {})
    if isinstance(container, Mapping):
        value = container.get(key)
        if value is not None:
            return value
    return None


def _normalize_optional_path(
    value: object,
    *,
    base_dir: Path | None,
) -> Path | None:
    if value is None:
        return None
    candidate_text = str(value).strip()
    if not candidate_text:
        return None
    candidate = Path(candidate_text)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def _path_text(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _suffix(path: Path | None) -> str | None:
    return None if path is None else path.suffix.lower()


def _same_path(left: Path | None, right: Path | None) -> bool:
    return left is not None and right is not None and left == right


def _safe_file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cad_provenance_report(
    cad_step_path: str | Path | None,
    *,
    source_config: Mapping[str, object] | None = None,
    source_config_path: str | Path | None = None,
) -> dict[str, object]:
    if cad_step_path is None or str(cad_step_path).strip() == "":
        return {
            "enabled": False,
            "cad_step_path": None,
            "cad_exists": False,
            "cad_step_brep_names": [],
            "direct_cad_step_binding": False,
            "step_derived_surface_mesh_binding": False,
            "real_cad_step_binding": False,
        }

    cad_path = Path(cad_step_path).resolve()
    base_dir = (
        Path(source_config_path).resolve().parent
        if source_config_path is not None
        else None
    )
    source = source_config if source_config is not None else {}
    mesh_path = _normalize_optional_path(
        _source_config_path_value(source, "mesh_path"),
        base_dir=base_dir,
    )
    surface_cache_path = _normalize_optional_path(
        _source_config_path_value(source, "surface_mesh_cache_path"),
        base_dir=base_dir,
    )
    mesh_path_matches = _same_path(mesh_path, cad_path)
    surface_cache_matches = _same_path(surface_cache_path, cad_path)
    declared_source_paths = tuple(
        path
        for path in (
            _normalize_optional_path(value, base_dir=base_dir)
            for value in _source_config_nested_path_values(source)
        )
        if path is not None
    )
    declared_source_matches = any(_same_path(path, cad_path) for path in declared_source_paths)
    direct_surface_path_ok = surface_cache_path is None or surface_cache_matches
    direct_cad_step_binding = (
        mesh_path_matches
        and _suffix(mesh_path) in {".step", ".stp"}
        and direct_surface_path_ok
    )
    declared_mesh_import_step_sha256 = _source_config_mesh_import_value(
        source,
        "source_step_sha256",
    )
    declared_mesh_import_surface_cache_sha256 = _source_config_mesh_import_value(
        source,
        "surface_mesh_cache_sha256",
    )
    surface_cache_actual_sha256 = _safe_file_sha256(surface_cache_path)
    report: dict[str, object] = {
        "enabled": True,
        "cad_step_path": str(cad_path),
        "cad_exists": cad_path.exists(),
        "cad_suffix": cad_path.suffix.lower(),
        "source_config_path": None if source_config_path is None else str(Path(source_config_path).resolve()),
        "source_config_declared_mesh_format": (
            None if not isinstance(source, Mapping) else source.get("mesh_format")
        ),
        "source_config_mesh_path": _path_text(mesh_path),
        "source_config_mesh_suffix": _suffix(mesh_path),
        "source_config_surface_mesh_cache_path": _path_text(surface_cache_path),
        "source_config_surface_mesh_cache_suffix": _suffix(surface_cache_path),
        "source_config_mesh_path_matches_cad_step": mesh_path_matches,
        "source_config_surface_mesh_cache_path_matches_cad_step": surface_cache_matches,
        "source_config_declared_source_step_paths": [
            str(path) for path in declared_source_paths
        ],
        "source_config_declared_source_step_path_matches_cad_step": declared_source_matches,
        "surface_mesh_cache_requires_provenance": (
            surface_cache_path is not None and not surface_cache_matches
        ),
        "direct_cad_step_binding": direct_cad_step_binding,
        "step_derived_surface_mesh_binding": False,
        "real_cad_step_binding": bool(direct_cad_step_binding),
        "real_cad_step_source_declared": bool(
            direct_cad_step_binding or declared_source_matches
        ),
        "source_config_mesh_import_source_step_sha256": (
            None
            if declared_mesh_import_step_sha256 is None
            else str(declared_mesh_import_step_sha256)
        ),
        "source_config_mesh_import_surface_cache_sha256": (
            None
            if declared_mesh_import_surface_cache_sha256 is None
            else str(declared_mesh_import_surface_cache_sha256)
        ),
        "source_config_surface_mesh_cache_actual_sha256": surface_cache_actual_sha256,
    }
    if not cad_path.exists():
        report.update(
            {
                "cad_step_sha256": None,
                "cad_step_file_schema": [],
                "cad_step_length_unit": None,
                "cad_step_brep_count": 0,
                "cad_step_brep_names": [],
                "cad_step_breps": [],
            }
        )
        return report
    summary = parse_step_cad_summary(cad_path)
    step_sha_matches = (
        declared_mesh_import_step_sha256 is not None
        and str(declared_mesh_import_step_sha256).lower() == summary.sha256.lower()
    )
    surface_cache_sha_matches = (
        declared_mesh_import_surface_cache_sha256 is not None
        and surface_cache_actual_sha256 is not None
        and str(declared_mesh_import_surface_cache_sha256).lower()
        == surface_cache_actual_sha256.lower()
    )
    step_derived_surface_mesh_binding = bool(
        mesh_path_matches
        and _suffix(mesh_path) in {".step", ".stp"}
        and surface_cache_path is not None
        and not surface_cache_matches
        and declared_source_matches
        and step_sha_matches
        and surface_cache_sha_matches
    )
    real_cad_step_binding = bool(
        direct_cad_step_binding or step_derived_surface_mesh_binding
    )
    report.update(
        {
            "cad_step_sha256": summary.sha256,
            "cad_step_file_schema": list(summary.file_schema),
            "cad_step_length_unit": summary.length_unit,
            "cad_step_brep_count": len(summary.breps),
            "cad_step_brep_names": [brep.name for brep in summary.breps],
            "cad_step_breps": [brep.as_dict() for brep in summary.breps],
            "source_config_mesh_import_source_step_sha256_matches_cad": (
                step_sha_matches
            ),
            "source_config_mesh_import_surface_cache_sha256_matches_file": (
                surface_cache_sha_matches
            ),
            "step_derived_surface_mesh_binding": (
                step_derived_surface_mesh_binding
            ),
            "real_cad_step_binding": real_cad_step_binding,
            "real_cad_step_source_declared": bool(
                real_cad_step_binding or declared_source_matches
            ),
        }
    )
    return report


__all__ = [
    "StepBrep",
    "StepCadSummary",
    "cad_provenance_report",
    "parse_step_cad_summary",
]
