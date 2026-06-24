# Simulation Core Usage

`simulation_core/` is the case-free Taichi simulation package. It contains only reusable solver primitives:

- `runtime.py`: GPU-only Taichi initialization.
- `geometry.py`: surface mesh and analytic UV sphere helpers.
- `fluid.py`: 3D Cartesian fluid state, pressure projection, force spreading, and diagnostics.
- `tri_surface.py`: region-based surface diagnostics, IBM force spreading, and velocity constraints.
- `projected_ibm.py`: projected immersed-boundary fluid step for a pair of surface regions.
- `fsi_coupling.py`: action/reaction interface-force balance and fixed-point relaxation.
- `mooney_shell_mpm.py`: triangular and UV Mooney membrane MPM states.
- `neo_hookean_mpm.py`: volumetric Neo-Hookean MLS-MPM state.
- `hyperelastic.py`: Neo-Hookean material cards, including Ecoflex 00-10 defaults.
- `validation.py`: lightweight validation result helpers.

It intentionally excludes:

- case runners
- paper or vendor benchmark configs
- `run_case.py`
- `reference.py`
- output plotting scripts
- any module named for a specific benchmark or vendor example

Official/vendor benchmark runners live under `benchmarks/official/`.

## Generic Simulation Entrypoint

Use the generic entrypoint for end-to-end case runs:

```powershell
& 'D:/TOOL/Anaconda/python.exe' .\run_simulation.py squid-soft-robot --steps 1 --output-dir .\output\squid_smoke
```

`squid-soft-robot` is a case adapter. It owns the squid mesh regions, pressure
schedule, outlet/lip/downstream monitors, and rendering conventions. Reusable
fluid, MPM, IBM/HIBM coupling, fixed-point relaxation, material, and diagnostic
logic belongs in `simulation_core/`, not in a squid-named runner.

Do not put reusable numerical or physical logic in squid-named files. A squid
file may load squid geometry, regions, monitors, pressure schedules, and output
formatting only.

## Minimal Import Check

Run from this folder:

```powershell
& 'D:/TOOL/Anaconda/python.exe' -c "import simulation_core; print(simulation_core.__version__)"
```

## Basic Fluid State

```python
from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig

spec = FluidDomainSpec(
    bounds_min_m=(0.0, 0.0, 0.0),
    bounds_max_m=(0.02, 0.02, 0.04),
    grid_nodes=(32, 32, 48),
    density_kgm3=1000.0,
    viscosity_pa_s=1.0e-3,
    dt_s=1.0e-4,
)

fluid = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
fluid.set_uniform_velocity((0.0, 0.0, 0.0))
fluid.compute_divergence()
projection = fluid.project(iterations=120)
print(projection)
```

## Basic Surface And Coupling Types

```python
from simulation_core import (
    FluidDomainSpec,
    ProjectedIbmRegionPairStepConfig,
    RegionPairInterfaceReactionTarget,
    SurfaceMesh,
)

mesh = SurfaceMesh(
    vertices=[
        (0.0, 0.0, 0.0),
        (1.0e-3, 0.0, 0.0),
        (0.0, 1.0e-3, 0.0),
    ],
    faces=[(0, 1, 2)],
)

fluid_spec = FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16))
ibm_step = ProjectedIbmRegionPairStepConfig(
    primary_region_id=1,
    secondary_region_id=2,
    primary_velocity_mps=(0.01, 0.0, -0.02),
    secondary_velocity_mps=(-0.005, 0.0, 0.01),
    dt_s=1.0e-4,
    ibm_correction_iterations=1,
    projection_iterations=80,
    pressure_outlet_zmin=False,
    velocity_constraint_blend=1.0,
    constraint_force_scale=1.0,
    density_kgm3=1000.0,
    viscosity_pa_s=1.0e-3,
    bounds_min_m=fluid_spec.bounds_min_m,
    bounds_max_m=fluid_spec.bounds_max_m,
    grid_nodes=fluid_spec.grid_nodes,
)
target = RegionPairInterfaceReactionTarget(
    primary_force_n=(0.0, 0.0, 1.0),
    secondary_force_n=(0.0, 0.0, -1.0),
)
print(mesh.vertex_count, ibm_step.dt_s, target.primary_force_n, target.secondary_force_n)
```

Projected IBM force spreading samples pressure and viscous stress from the fluid
fields, and uses the direct-forcing constraint `rho * control_volume *
(u_solid - u_fluid) / dt`. The solid-side interface reaction is the
equal-and-opposite force actually spread to the fluid grid.

## Ecoflex 00-10 Material

Ecoflex 00-10 is available as a reusable material card instead of a case-specific
constant. The default card is derived from Smooth-On's Ecoflex technical bulletin:

- specific gravity `1.04`, mapped to `1040 kg/m^3`
- 100% modulus `8 psi`, used to estimate the Neo-Hookean shear modulus
- tensile strength `120 psi`
- break elongation `800%`
- hardness `Shore 00-10`

```python
from simulation_core import (
    NeoHookeanStressProbe,
    TaichiRuntimeConfig,
    ecoflex_0010_material,
    incompressible_uniaxial_nominal_stress_pa,
)

material = ecoflex_0010_material(poissons_ratio=0.49)
print(material.shear_modulus_pa, material.bulk_modulus_pa)

nominal_100_pa = incompressible_uniaxial_nominal_stress_pa(
    stretch=2.0,
    shear_modulus_pa=material.shear_modulus_pa,
)
print(nominal_100_pa)

probe = NeoHookeanStressProbe(material, runtime=TaichiRuntimeConfig(arch="cuda"))
report = probe.evaluate_diagonal_stretch((2.0, 1.0 / 2.0**0.5, 1.0 / 2.0**0.5))
print(report.jacobian, report.first_piola_x_pa, report.cauchy_x_pa)
```

This is a starter Neo-Hookean model for simulation setup and solver testing.
For final Ecoflex accuracy, fit Mooney-Rivlin, Ogden, or Yeoh coefficients to
measured uniaxial/biaxial/planar test data and plug the fitted law into the same
material-model layer.

## Damping And Momentum Diagnostics

MPM `velocity_damping` defaults to `1.0`. Keep that default when checking
momentum conservation or transfer consistency. Setting `velocity_damping < 1`
is an explicit non-conservative numerical damper: it removes grid momentum, and
`transfer_relative_error` is expected to report that loss instead of hiding it
inside the expected momentum.

## How To Build A New Simulation

1. Define your geometry and units in SI.
2. Create a `FluidDomainSpec` for the Eulerian grid.
3. Create surface or MPM state objects for the solid boundary.
4. Use `tri_surface`, `projected_ibm`, and `fsi_coupling` to exchange velocity, pressure, and force.
5. Keep case-specific parameters, loading schedules, plotting, and official comparisons outside `simulation_core/`.

The package is GPU-only by design. Use `TaichiRuntimeConfig(arch="cuda")` on this machine.
