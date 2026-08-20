from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simulation_core import CG_PRECONDITIONER_CHOICES

from .source_config import DEFAULT_SOURCE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]

PRESSURE_SOLVER_CHOICES = ("auto", "jacobi", "compact_jacobi", "fv_jacobi", "fv_multigrid", "fv_cg")

PRESSURE_SOLVE_FAILURE_POLICY_CHOICES = ("raise", "report")

FLUID_ADVECTION_SCHEME_CHOICES = ("euler", "rk2")

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
    parser.add_argument(
        "--fixed-rim-region-id",
        type=int,
        default=5,
        help=(
            "Named CAD region used for the fixed membrane rim and the sharp "
            "air-backing reachability barrier. It must differ from both moving "
            "FSI regions and must contain faces."
        ),
    )
    parser.add_argument(
        "--neo-fixed-node-lock-policy",
        choices=("any_fixed_particle", "pure_fixed_mass"),
        default=None,
        help=(
            "Neo-Hookean-only grid-node locking policy. The Neo default is "
            "pure_fixed_mass, which avoids locking mixed free/fixed nodes. "
            "Tri-Mooney clamps fixed rim vertices directly and rejects this option "
            "when it is explicitly supplied. The case remains zero-gravity."
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
    parser.add_argument(
        "--interface-reaction-relaxation",
        type=float,
        default=0.5,
        help=(
            "Under-relaxation for the marker-state fixed point. This is a "
            "partitioned FSI coupling control, not a nozzle boundary condition."
        ),
    )
    parser.add_argument(
        "--interface-reaction-aitken",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Aitken Delta^2 adaptation for step-internal marker-state "
            "fixed-point updates. Enabled by default for added-mass stability; "
            "use --no-interface-reaction-aitken only for diagnostics."
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
        "--fsi-coupling-iterations",
        type=int,
        default=1,
        help=(
            "Fixed marker-state iterations per physical MPM step. One preserves "
            "the validated explicit baseline; values above one use the same "
            "sharp trial body with marker position/velocity under-relaxation."
        ),
    )
    parser.add_argument(
        "--fsi-marker-coupling-tolerance-mps",
        type=float,
        default=1.0e-4,
        help=(
            "Convergence tolerance for the sharp marker fixed-point velocity "
            "residual L2 norm in m/s."
        ),
    )
    parser.add_argument("--disable-pressure-outlet-zmin", action="store_true")
    parser.add_argument(
        "--far-pressure-air-backed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Classify the prescribed far-pressure compartment as air-backed on "
            "the sharp path. Use --no-far-pressure-air-backed as an escape hatch. "
            "Air backing requires the z-min pressure outlet."
        ),
    )
    parser.add_argument(
        "--far-pressure-inside-probe-max-multiplier",
        type=float,
        default=12.0,
        help="Maximum far-pressure inside probe distance multiplier; must be >= 3.",
    )
    parser.add_argument(
        "--two-sided-probe-max-multiplier",
        type=float,
        default=12.0,
        help="Maximum two-sided pressure probe distance multiplier; must be >= 3.",
    )
    parser.add_argument(
        "--one-sided-probe-max-multiplier",
        type=float,
        default=12.0,
        help="Maximum one-sided pressure probe distance multiplier; must be >= 3.",
    )
    parser.add_argument(
        "--far-pressure-air-backed-probe-normal-sign",
        type=float,
        choices=(-1.0, 0.0, 1.0),
        default=0.0,
        help=(
            "Independent air-backed seed probe side: -1 or +1 restricts seeding "
            "to that marker-normal side; safe default 0 probes both sides. This "
            "does not change the separately derived far-pressure traction sign."
        ),
    )
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
