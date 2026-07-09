from __future__ import annotations

# Module registry: every registered benchmark/case module, including
# spec-only benchmark cases that expose programmatic smoke entrypoints
# (run_*_fsi_smoke) but no CLI main().
CASE_MODULES = {
    "ansys-vertical-flap-fsi": "cases.ansys_vertical_flap_fsi",
    "comsol-multibody-mechanism-fsi": "cases.comsol_multibody_mechanism_fsi",
    "comsol-water-balloon-fsi": "cases.comsol_water_balloon_fsi",
    "squid-soft-robot": "cases.squid_soft_robot",
    "turek-hron-fsi": "cases.turek_hron_fsi",
}
# Public runnable registry advertised by run_simulation.py: only cases whose
# module exposes a CLI-compatible main(argv). The two COMSOL benchmark cases
# are intentionally excluded -- they only provide run_*_fsi_smoke(config)
# programmatic entrypoints (see tests/contracts, which pin them in
# CASE_MODULES as registered spec modules).
CASES_WITHOUT_CLI_MAIN = (
    "comsol-multibody-mechanism-fsi",
    "comsol-water-balloon-fsi",
)
AVAILABLE_CASES = tuple(
    case for case in CASE_MODULES if case not in CASES_WITHOUT_CLI_MAIN
)

__all__ = ["AVAILABLE_CASES", "CASE_MODULES"]
