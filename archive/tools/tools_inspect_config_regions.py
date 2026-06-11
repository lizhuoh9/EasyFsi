"""Inspect source-config regions/FSI flags and sharp marker region wiring."""
from __future__ import annotations

import json

CONFIG = r"_codex_validation/squid_case_run_render_20260610_005/simulation_config.json"
d = json.load(open(CONFIG, encoding="utf-8"))
for ns in d.get("named_selections", []):
    print(
        ns.get("region_id"),
        "|",
        ns.get("name"),
        "| bc=",
        ns.get("boundary_type"),
        "| fsi=",
        ns.get("fsi_contact"),
        "| faces=",
        len(ns.get("face_ids", [])),
    )
print("---loads/actuators/constraints---")
for key in ("loads", "actuators", "constraints"):
    for item in d.get(key) or []:
        print(
            key,
            "|",
            item.get("name"),
            "| region=",
            item.get("region_id"),
            "| type=",
            item.get("type") or item.get("boundary_type"),
        )
print("---analysis pressure schedule keys---")
analysis = d.get("analysis_settings", {})
for key in sorted(analysis):
    if "pressure" in key or "region" in key or "tail" in key or "main" in key:
        print(key, "=", analysis[key])
