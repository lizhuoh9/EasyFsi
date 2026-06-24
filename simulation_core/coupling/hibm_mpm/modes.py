from typing import Any

from .paper_requirements import _SHARP_MISSING, _SHARP_VALIDATION_MISSING

FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED = "legacy_projected_reduced"
FSI_COUPLING_MODE_HIBM_MPM_SHARP = "hibm_mpm_sharp"
FSI_COUPLING_MODE_CHOICES = (
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
)

def fsi_coupling_mode_report(mode: str) -> dict[str, Any]:
    mode_name = str(mode)
    if mode_name == FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED:
        return {
            "mode": mode_name,
            "solver_layer": "simulation_core",
            "implemented": True,
            "core_runner_available": False,
            "case_runner_available": True,
            "phase5_validation_complete": False,
            "legacy": True,
            "paper_hibm_mpm": False,
            "sharp_interface": False,
            "primary_coupling_variable": (
                "projected-IBM velocity residual plus reduced primary/secondary "
                "region-pair interface reaction"
            ),
            "region_pair_reaction_diagnostic_only": True,
            "legacy_projected_reduced": True,
            "not_paper_hibm_mpm": True,
            "missing": list(_SHARP_MISSING),
        }
    if mode_name == FSI_COUPLING_MODE_HIBM_MPM_SHARP:
        return {
            "mode": mode_name,
            "solver_layer": "simulation_core",
            "implemented": True,
            "core_runner_available": True,
            "case_runner_available": True,
            "phase5_validation_complete": False,
            "legacy": False,
            "paper_hibm_mpm": True,
            "sharp_interface": True,
            "primary_coupling_variable": "per-marker HIBM-MPM surface traction",
            "region_pair_reaction_diagnostic_only": False,
            "legacy_projected_reduced": False,
            "not_paper_hibm_mpm": False,
            "missing": list(_SHARP_VALIDATION_MISSING),
        }
    choices = ", ".join(FSI_COUPLING_MODE_CHOICES)
    raise ValueError(f"fsi_coupling_mode must be one of: {choices}")


def require_implemented_fsi_coupling_mode(mode: str) -> dict[str, Any]:
    report = fsi_coupling_mode_report(mode)
    if not bool(report["implemented"]):
        missing = ", ".join(str(item) for item in report["missing"])
        raise NotImplementedError(
            f"{mode} is declared but not implemented in simulation_core yet; "
            f"missing solver requirements: {missing}"
        )
    return report