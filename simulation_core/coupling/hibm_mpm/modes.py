from .paper_requirements import _SHARP_VALIDATION_MISSING

FSI_COUPLING_MODE_HIBM_MPM_SHARP = "hibm_mpm_sharp"


def hibm_mpm_sharp_coupling_report() -> dict[str, object]:
    return {
        "mode": FSI_COUPLING_MODE_HIBM_MPM_SHARP,
        "solver_layer": "simulation_core",
        "implemented": True,
        "core_runner_available": True,
        "case_runner_available": True,
        "phase5_validation_complete": False,
        "paper_hibm_mpm": True,
        "sharp_interface": True,
        "primary_coupling_variable": "per-marker HIBM-MPM surface traction",
        "region_pair_reaction_diagnostic_only": False,
        "missing": list(_SHARP_VALIDATION_MISSING),
    }
