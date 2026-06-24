from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from simulation_core import (
    CG_PRECONDITIONER_CHOICES,
    FSI_COUPLING_MODE_CHOICES,
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    INTERFACE_REACTION_SOLVER_CHOICES,
)

from .source_config import DEFAULT_SOURCE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]

FSI_STABILIZATION_PRESET_CHOICES = ("off", "conservative", "aggressive")

FSI_STABILIZATION_PRESET_CONFLICT_POLICY = (
    "reject_explicit_managed_options"
)

FSI_STABILIZATION_PRESET_MANAGED_FIELDS = (
    "fsi_coupling_target_map_relaxation",
    "fsi_coupling_rejected_trial_backtrack",
    "fsi_coupling_residual_growth_rejection_factor",
    "fsi_coupling_max_accepted_residual_n",
    "fsi_coupling_trust_region_force_increment_n",
    "fsi_coupling_trust_region_adaptive",
    "fsi_coupling_trust_region_shrink_factor",
    "fsi_coupling_trust_region_growth_factor",
    "fsi_coupling_trust_region_rebound_factor",
    "fsi_coupling_trust_region_rebound_backtrack",
    "fsi_coupling_trust_region_rebound_stop_factor",
    "fsi_coupling_trust_region_rebound_stop_max_residual_n",
)

_FSI_STABILIZATION_PRESET_PARAMETERS = {
    "off": {
        "fsi_coupling_target_map_relaxation": 1.0,
        "fsi_coupling_rejected_trial_backtrack": 1.0,
        "fsi_coupling_residual_growth_rejection_factor": math.inf,
        "fsi_coupling_max_accepted_residual_n": math.inf,
        "fsi_coupling_trust_region_force_increment_n": math.inf,
        "fsi_coupling_trust_region_adaptive": False,
        "fsi_coupling_trust_region_shrink_factor": 0.5,
        "fsi_coupling_trust_region_growth_factor": 1.25,
        "fsi_coupling_trust_region_rebound_factor": math.inf,
        "fsi_coupling_trust_region_rebound_backtrack": 0.5,
        "fsi_coupling_trust_region_rebound_stop_factor": math.inf,
        "fsi_coupling_trust_region_rebound_stop_max_residual_n": math.inf,
    },
    "conservative": {
        "fsi_coupling_target_map_relaxation": 0.35,
        "fsi_coupling_rejected_trial_backtrack": 0.5,
        "fsi_coupling_residual_growth_rejection_factor": 1.25,
        "fsi_coupling_max_accepted_residual_n": math.inf,
        "fsi_coupling_trust_region_force_increment_n": 0.25,
        "fsi_coupling_trust_region_adaptive": True,
        "fsi_coupling_trust_region_shrink_factor": 0.5,
        "fsi_coupling_trust_region_growth_factor": 1.1,
        "fsi_coupling_trust_region_rebound_factor": 1.5,
        "fsi_coupling_trust_region_rebound_backtrack": 0.5,
        "fsi_coupling_trust_region_rebound_stop_factor": 2.0,
        "fsi_coupling_trust_region_rebound_stop_max_residual_n": math.inf,
    },
    "aggressive": {
        "fsi_coupling_target_map_relaxation": 0.65,
        "fsi_coupling_rejected_trial_backtrack": 0.75,
        "fsi_coupling_residual_growth_rejection_factor": 2.0,
        "fsi_coupling_max_accepted_residual_n": math.inf,
        "fsi_coupling_trust_region_force_increment_n": 1.0,
        "fsi_coupling_trust_region_adaptive": True,
        "fsi_coupling_trust_region_shrink_factor": 0.7,
        "fsi_coupling_trust_region_growth_factor": 1.25,
        "fsi_coupling_trust_region_rebound_factor": 3.0,
        "fsi_coupling_trust_region_rebound_backtrack": 0.65,
        "fsi_coupling_trust_region_rebound_stop_factor": 4.0,
        "fsi_coupling_trust_region_rebound_stop_max_residual_n": math.inf,
    },
}

PRESSURE_SOLVER_CHOICES = ("auto", "jacobi", "compact_jacobi", "fv_jacobi", "fv_multigrid", "fv_cg")

PRESSURE_SOLVE_FAILURE_POLICY_CHOICES = ("raise", "report")

FLUID_ADVECTION_SCHEME_CHOICES = ("euler", "rk2")

INTERFACE_REACTION_ROBIN_TARGET_CHOICES = ("stabilized", "physical")

def resolve_fsi_stabilization_preset_parameters(preset: str) -> dict[str, object]:
    preset_name = str(preset)
    if preset_name not in _FSI_STABILIZATION_PRESET_PARAMETERS:
        choices = ", ".join(FSI_STABILIZATION_PRESET_CHOICES)
        raise ValueError(f"--fsi-stabilization-preset must be one of: {choices}")
    return dict(_FSI_STABILIZATION_PRESET_PARAMETERS[preset_name])

def _raw_cli_option_present(raw_args: Sequence[str], option: str) -> bool:
    prefix = f"{option}="
    return any(token == option or token.startswith(prefix) for token in raw_args)

def _fsi_stabilization_preset_conflicts(raw_args: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"--{field.replace('_', '-')}"
        for field in FSI_STABILIZATION_PRESET_MANAGED_FIELDS
        if _raw_cli_option_present(raw_args, f"--{field.replace('_', '-')}")
    )

def _apply_fsi_stabilization_preset(
    args: argparse.Namespace,
    *,
    raw_args: Sequence[str],
    parser: argparse.ArgumentParser,
) -> None:
    preset_name = str(args.fsi_stabilization_preset)
    if preset_name == "off":
        return
    conflicts = _fsi_stabilization_preset_conflicts(raw_args)
    if conflicts:
        parser.error(
            "--fsi-stabilization-preset cannot be combined with explicit "
            f"managed options under {FSI_STABILIZATION_PRESET_CONFLICT_POLICY}: "
            + ", ".join(conflicts)
        )
    for field, value in resolve_fsi_stabilization_preset_parameters(
        preset_name
    ).items():
        setattr(args, field, value)

def fsi_stabilization_effective_parameters_from_args(
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        field: getattr(args, field)
        for field in FSI_STABILIZATION_PRESET_MANAGED_FIELDS
    }

def raise_for_unsupported_hibm_mpm_sharp_iteration_options(
    *,
    fsi_coupling_mode: str,
    fsi_coupling_iterations: int,
) -> None:
    if str(fsi_coupling_mode) != FSI_COUPLING_MODE_HIBM_MPM_SHARP:
        return
    if int(fsi_coupling_iterations) < 1:
        raise ValueError("--fsi-coupling-iterations must be at least 1")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument(
        "--cad-step-path",
        default=None,
        help=(
            "Optional real STEP CAD path used to audit source-config geometry provenance. "
            "This is an input contract only; it does not prescribe forces, velocity, or flow."
        ),
    )
    parser.add_argument(
        "--require-real-cad-step",
        action="store_true",
        help=(
            "Fail before initialization unless --source-config either directly "
            "references --cad-step-path as a .step/.stp file or its generated "
            "surface mesh cache records matching STEP and cache SHA256 hashes."
        ),
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "cases" / "output_008step"))
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of physical time steps. Default runs through the full configured "
            "pressure waveform; pass an explicit small value for smoke tests."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Build the reduced case spec and grid diagnostics, write preflight_summary.json, "
            "and exit before Taichi/MPM/FSI initialization."
        ),
    )
    parser.add_argument(
        "--pressure-t0-s",
        type=float,
        default=None,
        help="Optional case pressure schedule t0 override in seconds.",
    )
    parser.add_argument(
        "--pressure-t1-s",
        type=float,
        default=None,
        help="Optional case pressure schedule t1 override in seconds.",
    )
    parser.add_argument(
        "--pressure-t2-s",
        type=float,
        default=None,
        help="Optional case pressure schedule t2 override in seconds.",
    )
    parser.add_argument(
        "--pressure-p0-pa",
        type=float,
        default=None,
        help="Optional case pressure schedule p0 override in Pa.",
    )
    parser.add_argument(
        "--pressure-p1-pa",
        type=float,
        default=None,
        help="Optional case pressure schedule p1 override in Pa.",
    )
    parser.add_argument(
        "--pressure-p2-pa",
        type=float,
        default=None,
        help="Optional case pressure schedule p2 override in Pa.",
    )
    parser.add_argument("--projection-iterations", type=int, default=3000)
    parser.add_argument(
        "--hibm-post-dirichlet-consistency-projections",
        type=int,
        default=3,
        help=(
            "Number of post-substep HIBM velocity-Dirichlet reconstruction/"
            "pressure-projection consistency passes on the sharp path."
        ),
    )
    parser.add_argument(
        "--pressure-solver",
        choices=PRESSURE_SOLVER_CHOICES,
        default="auto",
        help=(
            "Pressure projection solver. auto uses fv_multigrid on uniform FV grids "
            "and fv_cg on graded FV grids."
        ),
    )
    parser.add_argument(
        "--pressure-solve-failure-policy",
        choices=PRESSURE_SOLVE_FAILURE_POLICY_CHOICES,
        default="raise",
        help=(
            "Policy when the pressure solve reports nonconvergence: raise aborts "
            "the step; report returns the failure state in diagnostics."
        ),
    )
    parser.add_argument(
        "--fluid-advection-scheme",
        choices=FLUID_ADVECTION_SCHEME_CHOICES,
        default="euler",
        help=(
            "Fluid predictor semi-Lagrangian backtrace scheme. euler preserves the "
            "legacy single-backtrace predictor; rk2 uses a midpoint backtrace."
        ),
    )
    parser.add_argument(
        "--cg-tolerance",
        type=float,
        default=1.0e-6,
        help="Relative residual tolerance for --pressure-solver fv_cg.",
    )
    parser.add_argument(
        "--cg-preconditioner",
        choices=CG_PRECONDITIONER_CHOICES,
        default="auto",
        help=(
            "Preconditioner for --pressure-solver fv_cg. auto uses multigrid on "
            "graded FV grids only when no active pressure-interface matrix is present; "
            "otherwise it uses Jacobi."
        ),
    )
    parser.add_argument(
        "--multigrid-cycles",
        type=int,
        default=None,
        help="Optional V-cycle count when --pressure-solver resolves to fv_multigrid.",
    )
    parser.add_argument(
        "--divergence-cleanup-iterations",
        type=int,
        default=8,
        help=(
            "Optional local post-projection divergence cleanup iterations. This enforces "
            "the fluid incompressibility constraint and does not prescribe nozzle velocity, "
            "pressure, or flow."
        ),
    )
    parser.add_argument(
        "--divergence-cleanup-relaxation",
        type=float,
        default=0.7,
        help="Relaxation for local post-projection divergence cleanup; must be in [0, 1].",
    )
    parser.add_argument(
        "--diagnostic-disable-pressure-neumann-matrix-rows",
        action="store_true",
        help=(
            "Diagnostic-only HIBM-MPM sharp switch: keep no-slip velocity "
            "Dirichlet rows and wall BCs but suppress pressure-Neumann "
            "interface matrix/RHS rows."
        ),
    )
    parser.add_argument(
        "--diagnostic-dump-zero-correctable-cells",
        action="store_true",
        help=(
            "Diagnostic-only HIBM-MPM sharp switch: dump interior fluid cells whose "
            "divergence stencil has no pressure-correctable faces."
        ),
    )
    parser.add_argument(
        "--diagnostic-dump-high-residual-cells",
        action="store_true",
        help=(
            "Diagnostic-only HIBM-MPM sharp switch: dump the highest post-projection "
            "divergence residual cells with nearby marker and pressure-row context."
        ),
    )
    parser.add_argument(
        "--diagnostic-dump-pressure-neumann-invalid-rows",
        action="store_true",
        help=(
            "Diagnostic-only HIBM-MPM sharp switch: dump pressure-Neumann "
            "interface rows rejected during reconstruction."
        ),
    )
    parser.add_argument(
        "--projection-divergence-tolerance",
        type=float,
        default=1.0e-2,
        help="Validation gate for post-projection divergence L2.",
    )
    parser.add_argument(
        "--closure-coverage-floor",
        type=int,
        default=0,
        help=(
            "Fail fast when hibm_full_stress_far_pressure_closed_marker_count "
            "stays below this floor for --closure-coverage-floor-patience "
            "consecutive steps. 0 disables the guard."
        ),
    )
    parser.add_argument(
        "--closure-coverage-floor-patience",
        type=int,
        default=10,
        help=(
            "Consecutive steps below --closure-coverage-floor before the "
            "closure coverage floor guard raises."
        ),
    )
    parser.add_argument("--grid-scale", type=float, default=1.0)
    parser.add_argument(
        "--use-graded-grid",
        action="store_true",
        help=(
            "Use a tensor-product graded Cartesian fluid grid with a nozzle refinement "
            "column. This changes only mesh resolution, not nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--graded-grid-target-spacing-m",
        type=float,
        default=None,
        help="Target cell spacing inside the nozzle refinement column. Default is nozzle_radius/5.",
    )
    parser.add_argument(
        "--graded-grid-farfield-spacing-m",
        type=float,
        default=3.0e-3,
        help="Far-field fluid cell spacing for --use-graded-grid.",
    )
    parser.add_argument(
        "--graded-grid-growth-ratio",
        type=float,
        default=1.2,
        help="Maximum adjacent-cell spacing ratio for --use-graded-grid; must be greater than 1.",
    )
    parser.add_argument(
        "--graded-grid-max-cells",
        type=int,
        default=0,
        help="Maximum generated fluid cells for --use-graded-grid. Use 0 to disable this guard.",
    )
    parser.add_argument(
        "--use-tail-refinement",
        action="store_true",
        help=(
            "Add an optional region 8 tail FSI bounding-box refinement region to the "
            "graded Cartesian fluid grid. This changes only mesh resolution, not "
            "velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--tail-refinement-target-spacing-m",
        type=float,
        default=None,
        help=(
            "Target cell spacing inside the optional region 8 tail refinement box. "
            "Default is min(tail membrane thickness, graded-grid far-field spacing)."
        ),
    )
    parser.add_argument(
        "--tail-refinement-padding-m",
        type=float,
        default=None,
        help=(
            "Padding around source-config region 8 vertex bounds for optional tail "
            "mesh refinement. Default is two tail target cells."
        ),
    )
    parser.add_argument(
        "--time-step-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the source configuration time step for time-refinement studies. "
            "Use more steps to keep the same physical duration when this is below 1."
        ),
    )
    parser.add_argument(
        "--solid-model",
        choices=("tri_mooney_shell_mpm", "neo_hookean_mpm"),
        default="tri_mooney_shell_mpm",
        help=(
            "Solid model. tri_mooney_shell_mpm is the paper-calibrated arbitrary-triangle "
            "shell MPM; neo_hookean_mpm is the volumetric layered branch."
        ),
    )
    parser.add_argument("--solid-mpm-layers", type=int, default=2)
    parser.add_argument(
        "--solid-mpm-substeps",
        type=int,
        default=0,
        help="Neo-Hookean MPM substeps per fluid step. Use 0 for Ecoflex CFL-based auto substepping.",
    )
    parser.add_argument(
        "--membrane-thickness-scale",
        type=float,
        default=1.0,
        help=(
            "Positive multiplier for main/tail shell thickness. This changes the "
            "physical shell surface mass and membrane thickness; default 1 preserves "
            "the baseline Ecoflex geometry."
        ),
    )
    parser.add_argument(
        "--solid-density-scale",
        type=float,
        default=1.0,
        help=(
            "Positive multiplier for the Ecoflex solid density. This isolates "
            "rho_s*h_s surface-mass scaling without changing the membrane modulus; "
            "default 1 preserves the baseline material card."
        ),
    )
    parser.add_argument("--solid-mpm-cfl", type=float, default=0.35)
    parser.add_argument("--solid-mpm-velocity-damping", type=float, default=1.0)
    parser.add_argument(
        "--solid-mpm-flip-blend",
        type=float,
        default=0.95,
        help="Tri-Mooney shell MPM G2P blend: 0 is PIC, 1 is FLIP.",
    )
    parser.add_argument("--mooney-membrane-force-scale", type=float, default=1.0)
    parser.add_argument("--poissons-ratio", type=float, default=0.49)
    parser.add_argument("--arch", default="cuda")
    parser.add_argument("--constraint-force-scale", type=float, default=1.0)
    parser.add_argument(
        "--fsi-constraint-force-solid-mobility-ratio",
        type=float,
        default=0.0,
        help=(
            "Dimensionless solid/fluid mobility ratio for the projected-IBM "
            "constraint force. Zero preserves the explicit fluid-mass force; "
            "positive values scale the constraint force by 1/(1+ratio)."
        ),
    )
    parser.add_argument(
        "--fsi-solid-response-mobility-coupling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the measured trial solid velocity response to compute per-region "
            "projected-IBM constraint-force mobility ratios. Disabled by default; "
            "when enabled, this changes the raw physical interface operator rather "
            "than only relaxing the fixed-point solver map."
        ),
    )
    parser.add_argument(
        "--fsi-velocity-target-solid-mobility-ratio",
        type=float,
        default=0.0,
        help=(
            "Dimensionless solid/fluid mobility ratio for the projected-IBM "
            "target boundary velocity. Zero preserves the explicit target "
            "velocity. Positive values use sampled fluid velocity plus "
            "(solid-target minus sampled-fluid)/(1+ratio), changing both the "
            "constraint residual and FSI volume source."
        ),
    )
    parser.add_argument(
        "--fsi-solid-response-velocity-mobility-coupling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the measured trial solid velocity response to add per-region "
            "projected-IBM target-velocity mobility ratios. Disabled by default; "
            "when enabled, this changes the local interface boundary-velocity "
            "operator, not just the fixed-point solver map."
        ),
    )
    parser.add_argument(
        "--fsi-velocity-constraint-blend",
        type=float,
        default=0.0,
        help=(
            "Blend factor for enforcing no-slip velocity on the FSI water-side "
            "probe cells before projection. Nonzero values are reported as a "
            "prescribed interface velocity constraint in validation."
        ),
    )
    parser.add_argument(
        "--fsi-velocity-constraint-solid-mobility-ratio",
        type=float,
        default=0.0,
        help=(
            "Dimensionless solid/fluid mobility ratio for the FSI velocity "
            "constraint. Zero preserves the hard overwrite operator; positive "
            "values apply blend/(1+ratio) to emulate a coupled mobility denominator."
        ),
    )
    parser.add_argument(
        "--interface-reaction-relaxation",
        type=float,
        default=0.5,
        help=(
            "Under-relaxation for the interface reaction fed back to the solid. "
            "This is a partitioned FSI coupling relaxation, not a nozzle boundary condition."
        ),
    )
    parser.add_argument(
        "--interface-reaction-aitken",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Aitken Delta^2 adaptation for both step-internal interface-reaction "
            "fixed-point updates and accepted-step interface-reaction residual updates. "
            "Enabled by default for added-mass stability; use --no-interface-reaction-aitken "
            "only for diagnostics."
        ),
    )
    parser.add_argument(
        "--interface-reaction-aitken-lower-bound",
        type=float,
        default=0.01,
        help=(
            "Lower clipping bound for Aitken Delta^2 relaxation used in "
            "interface-reaction fixed-point and accepted-step updates. The "
            "default 0.01 preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--interface-reaction-aitken-upper-bound",
        type=float,
        default=1.5,
        help=(
            "Upper clipping bound for Aitken Delta^2 relaxation used in "
            "interface-reaction fixed-point and accepted-step updates. The "
            "default 1.5 preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--interface-reaction-passivity-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional diagnostic limiter for committed interface reactions. Disabled by "
            "default because it projects positive-power committed reactions to the "
            "zero-power boundary and can otherwise change the projected-IBM reaction "
            "used by the solid. It never prescribes nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--interface-reaction-robin-impedance-ns-m",
        type=float,
        default=0.0,
        help=(
            "Explicit Phase-C Robin-Neumann interface impedance in N*s/m. The "
            "default 0 preserves the existing partitioned Aitken/IQN path. "
            "Positive values add -Z*(v_n-v_{n-1}) to the accepted step-to-step "
            "interface reaction target; this changes only the interface coupling "
            "law, not nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--interface-reaction-robin-matrix-impedance-ns-m",
        type=float,
        default=0.0,
        help=(
            "Opt-in Phase-1B interface impedance in N*s/m that is scattered as "
            "per-marker terms into the FV-CG pressure matrix. The default 0 "
            "preserves the existing explicit partitioned path; positive values "
            "require --pressure-solver fv_cg."
        ),
    )
    parser.add_argument(
        "--interface-reaction-robin-target-mode",
        choices=INTERFACE_REACTION_ROBIN_TARGET_CHOICES,
        default="stabilized",
        help=(
            "How the fluid-side Robin impedance contribution enters the committed "
            "solid interface reaction. stabilized preserves the current Phase-C "
            "path; physical subtracts the Robin term from the returned target so "
            "the impedance acts only as a fluid-side boundary stabilizer."
        ),
    )
    parser.add_argument(
        "--min-outlet-to-main-volume-flux-ratio",
        type=float,
        default=0.1,
        help=(
            "Validation gate for real sampled outlet flux relative to the kinematic "
            "main-membrane volume-flux estimate. Values far below this mean the "
            "reported jet is not present in the fluid field."
        ),
    )
    parser.add_argument(
        "--pressure-outlet-source-ratio-tolerance",
        type=float,
        default=0.1,
        help=(
            "Validation tolerance for the pressure-outlet boundary-face velocity "
            "flux ratio relative to the FSI volume source. The pressure-implied "
            "flux is reported as a finite diagnostic, not as this conservation gate."
        ),
    )
    parser.add_argument(
        "--fluid-substeps",
        type=int,
        default=1,
        help=(
            "Number of fluid predictor/IBM/projection substeps per physical solid step. "
            "This is a time-integration refinement for CFL stability, not a nozzle "
            "velocity, pressure, or flow boundary."
        ),
    )
    parser.add_argument(
        "--adaptive-fluid-substeps",
        action="store_true",
        help=(
            "Increase the next step's fluid substeps from previously computed CFL "
            "diagnostics. This is a generic CFL time-integration control and does "
            "not prescribe pressure, velocity, force, or flow results."
        ),
    )
    parser.add_argument(
        "--adaptive-fluid-substeps-target-cfl",
        type=float,
        default=0.25,
        help="Target CFL used when --adaptive-fluid-substeps is enabled.",
    )
    parser.add_argument(
        "--adaptive-fluid-substeps-max",
        type=int,
        default=16,
        help="Maximum fluid substeps allowed by --adaptive-fluid-substeps.",
    )
    parser.add_argument(
        "--adaptive-fluid-substeps-safety",
        type=float,
        default=1.25,
        help="Safety multiplier applied to previous CFL when choosing adaptive substeps.",
    )
    parser.add_argument(
        "--ibm-correction-iterations",
        type=int,
        default=2,
        help=(
            "Number of force-spread/body-force/projection correction passes per fluid step. "
            "This repeats the projected IBM no-slip correction; it does not prescribe nozzle "
            "velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-iterations",
        type=int,
        default=1,
        help=(
            "Solid-fluid fixed-point iterations per physical MPM time step. "
            "hibm_mpm_sharp uses a marker-level position/velocity fixed point "
            "when this is greater than 1; legacy_projected_reduced keeps its "
            "older region-reaction trial re-advance diagnostic path."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-adaptive-iterations-max",
        type=int,
        default=0,
        help=(
            "Optional residual-triggered maximum for legacy projected/reduced "
            "step-internal interface-reaction iterations. 0 disables the "
            "adaptive budget and uses --fsi-coupling-iterations every step."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-adaptive-iterations-residual-threshold-n",
        type=float,
        default=math.inf,
        help=(
            "Use --fsi-coupling-adaptive-iterations-max on the next step when "
            "the previous step's FSI coupling residual norm exceeds this "
            "Newton threshold. The default infinity disables the trigger."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-adaptive-iterations-cfl-threshold",
        type=float,
        default=math.inf,
        help=(
            "Use --fsi-coupling-adaptive-iterations-max on the next step when "
            "the previous step's sampled CFL exceeds this threshold. The "
            "default infinity disables the CFL trigger."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-same-step-rerun-iterations-max",
        type=int,
        default=0,
        help=(
            "Optional maximum for rerunning the current legacy "
            "projected/reduced FSI step when the first fixed-point attempt "
            "finishes above the same-step residual threshold. 0 disables the "
            "same-step rerun path."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-same-step-rerun-residual-threshold-n",
        type=float,
        default=math.inf,
        help=(
            "Physical force residual threshold in Newtons that triggers a "
            "same-step FSI rerun when the first attempt did not converge. The "
            "default infinity disables the trigger."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-same-step-rerun-fluid-substep-factor",
        type=float,
        default=1.0,
        help=(
            "Optional same-step fluid-substep multiplier used when every "
            "projected/reduced FSI trial is rejected by safety gates. Values "
            "above 1 rerun the current step with more fluid substeps, capped by "
            "--adaptive-fluid-substeps-max; 1 disables this path."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-residual-continuation-iterations-max",
        type=int,
        default=0,
        help=(
            "Optional extra fixed-point iterations appended inside the current "
            "legacy projected/reduced FSI solve when the base iteration budget "
            "ends above --fsi-coupling-residual-continuation-threshold-n. "
            "0 disables continuation."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-residual-continuation-threshold-n",
        type=float,
        default=math.inf,
        help=(
            "Accepted physical force residual threshold in Newtons for "
            "continuing the current fixed-point solve beyond the base "
            "--fsi-coupling-iterations budget. The default infinity disables "
            "continuation."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-residual-continuation-rebound-secant-from-best",
        action="store_true",
        help=(
            "When residual continuation is active and a continuation trial "
            "rebounds away from the best accepted trial, restart from the best "
            "trial with a diagonal secant force update instead of stopping from "
            "the best trial."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-residual-continuation-rebound-secant-factor",
        type=float,
        default=math.inf,
        help=(
            "Residual rebound factor for triggering the optional "
            "residual-continuation secant-from-best update. The default "
            "infinity makes the secant trigger inherit "
            "--fsi-coupling-trust-region-rebound-stop-factor."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-residual-continuation-rebound-secant-evaluation-extensions-max",
        type=int,
        default=0,
        help=(
            "Maximum number of extra same-step fixed-point evaluations reserved "
            "only for evaluating a rebound secant-from-best candidate produced "
            "at the end of the residual-continuation budget. 0 preserves the "
            "strict configured continuation budget."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trial-interior-divergence-tolerance",
        type=float,
        default=math.inf,
        help=(
            "Optional acceptance gate for legacy projected/reduced FSI trials. "
            "When finite, reject otherwise CFL-safe trial states whose sampled "
            "post-projection interior_divergence_l2 exceeds this tolerance. "
            "The default infinity preserves the previous CFL-only acceptance."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-mode",
        choices=FSI_COUPLING_MODE_CHOICES,
        default=FSI_COUPLING_MODE_HIBM_MPM_SHARP,
        help=(
            "Solver-level FSI coupling mode. hibm_mpm_sharp selects the generic "
            "sharp-interface HIBM-MPM solver path. legacy_projected_reduced is an "
            "explicit legacy diagnostic option that keeps the old projected-IBM "
            "plus reduced region-reaction path."
        ),
    )
    parser.add_argument(
        "--fsi-stabilization-preset",
        choices=FSI_STABILIZATION_PRESET_CHOICES,
        default="off",
        help=(
            "Auditable bundle of existing FSI fixed-point stabilization controls. "
            "off preserves the current explicit defaults; conservative and "
            "aggressive expand to target-map relaxation, residual growth gates, "
            "and force-increment trust-region settings without prescribing "
            "pressure, velocity, or flow."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-solver",
        choices=INTERFACE_REACTION_SOLVER_CHOICES,
        default="aitken",
        help=(
            "Step-internal interface-reaction fixed-point solver. aitken preserves the "
            "existing scalar-relaxed path; iqn_ils uses inverse least-squares secant "
            "updates of the same interface residual equation."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-tolerance-n",
        type=float,
        default=1.0e-3,
        help="Convergence tolerance for the two-component step-internal interface-reaction residual in Newtons.",
    )
    parser.add_argument(
        "--fsi-marker-coupling-tolerance-mps",
        type=float,
        default=1.0e-4,
        help=(
            "Convergence tolerance for sharp HIBM-MPM marker fixed-point "
            "position/velocity residual in m/s."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-target-map-relaxation",
        type=float,
        default=1.0,
        help=(
            "Semi-implicit fixed-point target-map relaxation beta in (0, 1]. "
            "beta=1 preserves the raw physical target map. beta<1 solves the "
            "same physical fixed point through F + beta*(T(F)-F) and reports "
            "both solver-map and raw physical-map amplification."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-rejected-trial-backtrack",
        type=float,
        default=1.0,
        help=(
            "Backtracking fraction in (0, 1] applied after an interface-reaction "
            "trial is rejected by the stability predicate. The default 1.0 "
            "preserves the previous no-backtracking behavior."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-residual-growth-rejection-factor",
        type=float,
        default=math.inf,
        help=(
            "Reject an otherwise stability-accepted interface-reaction trial "
            "when its physical residual norm exceeds the best accepted residual "
            "by this factor. Values must be >= 1; the default infinity disables "
            "this residual-aware trust gate."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-max-accepted-residual-n",
        type=float,
        default=math.inf,
        help=(
            "Reject an otherwise stability-accepted interface-reaction trial "
            "when its physical residual norm exceeds this Newton threshold. "
            "The default infinity disables this absolute residual trust gate."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trust-region-force-increment-n",
        type=float,
        default=math.inf,
        help=(
            "Limit the norm of each proposed interface-reaction force update "
            "between fixed-point trials. The default infinity disables this "
            "force-increment trust region."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trust-region-adaptive",
        action="store_true",
        help=(
            "Adapt the force-increment trust radius inside each fixed-point "
            "solve: shrink after physical residual growth and grow back after "
            "residual reduction. Requires a finite trust-region increment."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trust-region-shrink-factor",
        type=float,
        default=0.5,
        help=(
            "Adaptive trust-region shrink factor in (0, 1] applied after a "
            "trial's physical residual grows relative to the previous trial."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trust-region-growth-factor",
        type=float,
        default=1.25,
        help=(
            "Adaptive trust-region growth factor >= 1 applied after a trial's "
            "physical residual decreases relative to the previous trial."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trust-region-rebound-factor",
        type=float,
        default=math.inf,
        help=(
            "Backtrack the next interface-reaction trial toward the best "
            "accepted trial when an otherwise accepted trial's physical "
            "residual exceeds the best accepted residual by this factor. "
            "The default infinity disables this rebound trust backtrack."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trust-region-rebound-backtrack",
        type=float,
        default=0.5,
        help=(
            "Rebound trust backtrack factor in (0, 1) used to place the next "
            "trial between the best accepted force and the rebounded force."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trust-region-rebound-stop-factor",
        type=float,
        default=math.inf,
        help=(
            "Stop the current interface-reaction fixed-point solve and commit "
            "the best accepted trial when a later otherwise accepted trial's "
            "physical residual exceeds the best accepted residual by this "
            "factor. The default infinity disables this best-trial stop policy."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-trust-region-rebound-stop-max-residual-n",
        type=float,
        default=math.inf,
        help=(
            "Only allow the best-trial rebound-stop policy to stop early when "
            "the best accepted physical residual is at or below this Newton "
            "ceiling. The default infinity preserves the previous stop policy."
        ),
    )
    parser.add_argument(
        "--reuse-accepted-fsi-trial-state",
        action="store_true",
        help=(
            "Experimental performance path: when the fixed-point solver proves the "
            "last trial is the accepted FSI state, reuse that full-report trial state "
            "instead of re-advancing the accepted solid/fluid step. Disabled by default."
        ),
    )
    parser.add_argument("--disable-pressure-outlet-zmin", action="store_true")
    parser.add_argument("--disable-reduced-obstacles", action="store_true")
    parser.add_argument(
        "--source-config-intersect-reduced-water-domain",
        action="store_true",
        help=(
            "Legacy diagnostic topology path: when the source config provides a "
            "CAD-derived fluid active mask, intersect it with the reduced analytic "
            "squid water domain. Disabled by default so real CAD fluid topology is "
            "not narrowed by case-specific analytic geometry."
        ),
    )
    parser.add_argument(
        "--source-config-connect-surface-seeds-to-zmin",
        action="store_true",
        help=(
            "Diagnostic topology repair: minimally carve obstacle cells so "
            "surface-seeded active-water components connect to the z-min pressure "
            "outlet component. Disabled by default because it changes the CAD-derived "
            "initial obstacle mask."
        ),
    )
    parser.add_argument(
        "--source-config-surface-seed-zmin-connection-max-carve-cells",
        type=int,
        default=256,
        help=(
            "Maximum obstacle cells the surface-seed-to-zmin diagnostic topology "
            "repair may carve when --source-config-connect-surface-seeds-to-zmin is set."
        ),
    )
    parser.add_argument(
        "--use-region14-aperture-carve",
        action="store_true",
        help=(
            "Use source-config region 14 open-edge aperture geometry to set the reduced "
            "nozzle/outlet carve center and radius. This changes only the obstacle/opening "
            "geometry, not nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--disable-region14-aperture-carve",
        action="store_true",
        help=(
            "Disable source-config-driven region 14 aperture carve even when the "
            "source config declares selection 14 as the solid obstacle opening."
        ),
    )
    parser.add_argument(
        "--open-downstream-farfield",
        action="store_true",
        help=(
            "With region 14 aperture carve enabled, keep the external domain below "
            "the region 14 aperture plane as active water instead of a narrow outlet plume. "
            "This is an obstacle/topology correction, not a flow boundary condition."
        ),
    )
    parser.add_argument(
        "--use-nozzle-taper",
        action="store_true",
        help=(
            "Use an analytic converging inlet taper upstream of the reduced nozzle throat. "
            "This changes only obstacle geometry, not nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--nozzle-taper-length-m",
        type=float,
        default=None,
        help=(
            "Length of the analytic nozzle taper. Default with --use-nozzle-taper is "
            "min(nozzle_length, chamber_z_min - downstream_z)."
        ),
    )
    parser.add_argument(
        "--nozzle-taper-inlet-radius-m",
        type=float,
        default=None,
        help="Inlet radius of the analytic taper. Default is the reduced chamber radius.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument(
        "--max-wall-time-s",
        type=float,
        default=0.0,
        help=(
            "Stop gracefully after the current completed step once this wall-time "
            "budget is exceeded. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-step",
        action="store_true",
        help=(
            "Write a restart checkpoint after every completed physical step. This is "
            "intended for long validation trends that must be resumed across runs."
        ),
    )
    parser.add_argument(
        "--fluid-snapshot-interval",
        type=int,
        default=0,
        help=(
            "Write a compact visualization snapshot (fluid speed slices + marker "
            "positions, .npz) every N completed steps in the sharp runner. 0 disables."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help=(
            "Resume from the checkpoint path and append to the existing history.csv. "
            "The checkpoint and history must agree on the completed step count."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Path for --checkpoint-every-step/--resume-from-checkpoint. Defaults to "
            "run_checkpoint.npz inside --output-dir."
        ),
    )
    args = parser.parse_args(argv)
    raw_args = sys.argv[1:] if argv is None else list(argv)
    _apply_fsi_stabilization_preset(args, raw_args=raw_args, parser=parser)
    args.divergence_cleanup_iterations_explicit = any(
        token == "--divergence-cleanup-iterations"
        or token.startswith("--divergence-cleanup-iterations=")
        for token in raw_args
    )
    args.steps_explicit = any(token == "--steps" or token.startswith("--steps=") for token in raw_args)
    if args.graded_grid_max_cells is not None and args.graded_grid_max_cells < 0:
        parser.error("--graded-grid-max-cells must be non-negative")
    if args.graded_grid_max_cells == 0:
        args.graded_grid_max_cells = None
    if args.use_tail_refinement and not args.use_graded_grid:
        parser.error("--use-tail-refinement requires --use-graded-grid")
    if (
        args.tail_refinement_target_spacing_m is not None
        and args.tail_refinement_target_spacing_m <= 0.0
    ):
        parser.error("--tail-refinement-target-spacing-m must be positive")
    if (
        args.tail_refinement_padding_m is not None
        and args.tail_refinement_padding_m < 0.0
    ):
        parser.error("--tail-refinement-padding-m must be non-negative")
    return args
