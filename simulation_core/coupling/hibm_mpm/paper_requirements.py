from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class HibmMpmPaperRequirement:
    requirement: str
    paper_section: str
    paper_mechanism: str
    current_status: str
    required_solver_work: str

_PAPER_REQUIREMENTS = (
    HibmMpmPaperRequirement(
        requirement="Taichi-resident solver path",
        paper_section="Implementation constraint",
        paper_mechanism=(
            "Solver state for HIBM-MPM search, reconstruction, matrix boundary "
            "terms, stress sampling, and MPM force scatter must remain on the "
            "Taichi side during runtime."
        ),
        current_status="partial",
        required_solver_work=(
            "Keep extending the Taichi field assembly into a complete sharp "
            "step; do not use NumPy loops or host round-trips as the solver "
            "path."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="surface markers",
        paper_section="Sections 3.2-4",
        paper_mechanism=(
            "The solid surface is represented by material points and an "
            "unstructured triangular interface carrying positions, velocities, "
            "normals, and surface areas."
        ),
        current_status="partial",
        required_solver_work=(
            "Build a solver-owned marker field x_gamma, v_gamma, n_gamma, "
            "A_gamma, region_id, traction, and force."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="IB node search",
        paper_section="Section 3.4",
        paper_mechanism=(
            "Fluid grid nodes within a radius comparable to local grid spacing "
            "from interface triangle centroids are marked near-boundary nodes."
        ),
        current_status="partial",
        required_solver_work=(
            "Search fluid grid nodes against the current surface each step and "
            "store IB node counts in solver diagnostics."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="inside/outside classification",
        paper_section="Section 3.4",
        paper_mechanism=(
            "The sign of dot(n_face, x_node - x_face_center) separates external "
            "fluid-side IB nodes from internal solid-side nodes."
        ),
        current_status="partial",
        required_solver_work=(
            "Classify near-boundary nodes with local surface normals and expose "
            "invalid or ambiguous classifications."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="normal reconstruction",
        paper_section="Section 3.4",
        paper_mechanism=(
            "Velocity and pressure values at IB nodes are reconstructed along "
            "the well-defined normal to the body; nearest-surface interpolation "
            "is used only when projection is not unique."
        ),
        current_status="partial",
        required_solver_work=(
            "Find boundary and interior fluid points along each local normal and "
            "record fallback counts."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="velocity Dirichlet no-slip",
        paper_section="Section 2, Equation 11; Section 3.4",
        paper_mechanism=(
            "Fluid velocity on the immersed surface equals the velocity of the "
            "moving or deforming body."
        ),
        current_status="partial",
        required_solver_work=(
            "Keep no-slip Dirichlet rows wired to the fluid projection boundary "
            "condition phase, then connect them to the complete sharp-interface "
            "solve loop."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="pressure Neumann matrix rows",
        paper_section="Section 3.4",
        paper_mechanism=(
            "Pressure boundary conditions at IB nodes are Neumann conditions "
            "derived from normal momentum balance."
        ),
        current_status="partial",
        required_solver_work=(
            "Keep HIBM pressure Neumann RHS row assembly wired to FV-CG fields, "
            "then connect it to the full sharp-interface solve loop."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="full-stress traction",
        paper_section="Section 2, Equation 12; Section 4",
        paper_mechanism=(
            "The full fluid stress, pressure plus viscous stress, is interpolated "
            "onto the body surface to form traction."
        ),
        current_status="partial",
        required_solver_work=(
            "Keep marker-level sigma_f = -pI + mu(grad v + grad v^T) traction "
            "sampling in the sharp-interface path and finish the complete "
            "HIBM-MPM solve loop."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="per-marker MPM external force",
        paper_section="Sections 3.2-4",
        paper_mechanism=(
            "Surface traction becomes the MPM external force through the "
            "background-grid shape functions."
        ),
        current_status="partial",
        required_solver_work=(
            "Use the sharp load assembly as the primary solid load path and "
            "finish the full HIBM-MPM step without reduced region reaction."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="surface feedback",
        paper_section="Section 4",
        paper_mechanism=(
            "After the solid solve updates material point positions and "
            "velocities, the next fluid step rebuilds IB boundary conditions "
            "from the new surface."
        ),
        current_status="partial",
        required_solver_work=(
            "Use Taichi surface-field feedback for all production MPM solid "
            "paths and validate the rebuilt next-step IB search and boundary "
            "rows in long runs."
        ),
    ),
)


_SHARP_MISSING = tuple(item.requirement for item in _PAPER_REQUIREMENTS)
_SHARP_VALIDATION_MISSING = (
    "long-run validation",
)


def hibm_mpm_paper_requirements() -> tuple[dict[str, str], ...]:
    return tuple(asdict(item) for item in _PAPER_REQUIREMENTS)