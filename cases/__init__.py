from __future__ import annotations

CASE_MODULES = {
    "ansys-vertical-flap-fsi": "cases.ansys_vertical_flap_fsi",
    "comsol-multibody-mechanism-fsi": "cases.comsol_multibody_mechanism_fsi",
    "comsol-water-balloon-fsi": "cases.comsol_water_balloon_fsi",
    "squid-soft-robot": "cases.squid_soft_robot",
}
AVAILABLE_CASES = tuple(CASE_MODULES)

__all__ = ["AVAILABLE_CASES", "CASE_MODULES"]
