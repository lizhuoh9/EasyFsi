# Turek–Hron FSI Validation Report

Status as of 2026-07-07. Solver: HIBM-MPM (sharp immersed boundary + Material
Point Method), Python + Taichi, CUDA. Interpreter for all commands:
`"D:/TOOL/Anaconda/python.exe"`. Case: `cases/turek_hron_fsi.py`.

This report records what has been **verified by runnable experiment**, what has
been **diagnosed but not fixed**, and what is a **method-limited frontier**. It
deliberately separates confirmed results from confounded comparisons. Every
physical number below EMERGES from the unchanged force/displacement integrals —
nothing was scaled or assigned to match a reference.

## 1. Fixes landed this campaign

| Commit | Change | Verification |
|---|---|---|
| `e5c80cb` | **Momentum-consistent step pressure** — `project()` increment mode (`accumulate_pressure_into_previous`) so the per-step re-projections stop overwriting the physical pressure | FSI1 total drag 0.32× → **0.92×** reference; form drag ~0 → 9.05 N/m |
| `e5c80cb` | **Obstacle-mask surface force integration** (pressure/form + viscous) | Cylinder-only reproduces Schäfer–Turek 2D-1: **Cd 5.79 vs 5.58 (within 4%)** |
| `e5c80cb` | **y-symmetry face-symmetric Dirichlet (mode 2)** — static wall rows only | Mirror unit test passes; structurally correct (dynamical impact small — see §4) |
| `e5c80cb` | **Re-projection budget** (1200 iters / 1e-4 tol) | Measured A/B: **1.72× speedup**, accuracy deltas < 0.2% |
| `7f7da08` | **Strong coupling (Picard + Aitken)** at the case level, gated `fsi_coupling_iterations` (default 1 = legacy) | Defeats FSI3 added-mass divergence: 30 ms blow-up → 1200 steps stable |

The pressure-overwrite root cause (each step ran up to 3 `fluid.project()` calls;
the re-projections solved a tiny residual that overwrote the ~200 Pa physical
field, collapsing cylinder stagnation) is documented in
`docs/` and the project memory.

## 2. Results vs reference (settled windows)

| Case | Quantity | Ours | Reference | Verdict |
|---|---|---|---|---|
| **Cylinder-only Re=20** | Cd | 5.79 | 5.58 (Schäfer–Turek 2D-1) | ✅ **4%** — solver + force integration validated |
| **FSI1** (steady) | total drag /span | 13.10 N/m | 14.295 | 0.92× (was 0.32×) |
| | tip uy | +0.25 mm | +0.82 | sign correct (was −0.58); magnitude ~30% |
| | form/friction drag | 9.05 / 4.53 | — | form recovered from ~0 |
| **FSI2** (ratio 10) | stability | 2000 steps, no blow-up | — | ✅ stable at real loads |
| | tip uy (transient) | rings ±6 mm @ ~1 Hz | +1.23 ± 80.6 @ 2.0 Hz | limit cycle not reached in 2 s (see §4) |
| **FSI3** (ratio 1) | stability | 1200 steps stable (strong coupling) | — | ✅ no more added-mass blow-up |
| | tip uy | +18 ± 6.5 mm @ 2.5–2.9 Hz | +1.5 ± 33.5 @ 5.3 Hz | ❌ self-excited flutter NOT reproduced (§4) |
| | total drag /span | 637 N/m | 452 | high; comparison partly confounded (§4) |

## 3. FSI3 diagnosis (complete, evidence-based)

The FSI3 self-excited flutter is **not reproduced**; the beam damps to a steady
+17 mm deflection instead of the ±33.5 mm limit cycle. The failure was localized
by three isolating experiments:

1. **Wake sheds correctly** — cylinder-only at Re=200 sheds von Kármán vortices
   (Cl ± 0.15, ~5 Hz, close to the reference 5.3 Hz flutter frequency).
2. **The shedding force reaches the beam** — the coupled lift oscillates at
   3.75 Hz.
3. **The beam does not lock in** — tip_uy responds at ~2.5 Hz, near its own
   natural frequency (measured 2.67 Hz in vacuum, blend-independent), far below
   the 5 Hz shedding, so it never synchronizes into the lock-in limit cycle.

Cheap levers **ruled out by direct measurement** (not argued away):

- **Coupling convergence** is not the limiter — the existing run already
  converges (post-ramp residual median 6e-4, 82% of steps below 1e-3 tolerance);
  more iterations cannot remove damping that is not there.
- **PIC dissipation** is not the limiter — a vacuum beam free-vibration test
  shows the numerical damping ratio ζ ≈ 0.008 and is essentially independent of
  the PIC/FLIP blend (0.0087 → 0.0081 across blend 0.0 → 0.9).
- **2× grid does not fix the drag** — cylinder-only Cd is 3.25 (base) vs 3.19
  (2×): not resolution-sensitive at this refinement.

## 4. Diagnosed limitations (honest caveats)

- **FSI3 flutter is method-limited.** Reproducing lock-in flutter at density
  ratio 1 would require, together: substantially finer grid (> 2×), a
  less-dissipative advection scheme (the current predictor is semi-Lagrangian,
  which smears the fine wake structures that pump energy into the beam), and a
  clean reference decomposition. Investment is large with uncertain payoff.
- **Drag "41% high" is partly a confounded comparison.** The cylinder-only Cd
  was compared against a value inferred from the cylinder+beam (oscillating)
  reference; a static-beam solver state and an oscillating-beam reference are not
  the same configuration. The clean, unconfounded anchor is the Re=20
  cylinder-only match (4%).
- **MPM beam is ~35% stiffer than Euler–Bernoulli** (vacuum f₁ 2.67 Hz vs 1.97
  Hz theory) — a separate solid-accuracy item, not chased here.
- **y-symmetry mode 2 is structurally correct but dynamically small** — it fixes
  the wall face-constraint asymmetry (unit-tested) but does not by itself move
  the FSI1 tip_uy magnitude; the residual is resolution.
- **FSI2 limit cycle needs longer physical time** — 2 s of simulation shows
  transient ringing but not the developed ±80 mm cycle (which establishes over
  many periods); this is a time-budget × step-cost limit, not a stability limit.
- **CUDA backend is not run-to-run bit-deterministic** (float-atomic scatter
  ordering); regression checks use a variance-envelope test, not bit-identity.

## 5. Reproduction

```bash
cd "D:/working/squid robot/simulation/src/reference/papers/HIBM-MPM/refactored"
PY="D:/TOOL/Anaconda/python.exe"

# FSI1 steady (700 steps, ~2.5 h): total drag → ~13.1 vs ref 14.295
$PY -c "from dataclasses import replace; import cases.turek_hron_fsi as t; \
  t.run_turek_hron_fsi(replace(t.fsi1_config(step_count=700), inlet_ramp_time_s=1.0), preset='fsi1', output_dir='out_fsi1')"

# FSI3 strong-coupled (density ratio 1; needs fsi_coupling_iterations>1 or it diverges)
$PY -c "from dataclasses import replace; import cases.turek_hron_fsi as t; \
  t.run_turek_hron_fsi(replace(t.fsi3_config(step_count=500), inlet_ramp_time_s=0.3, \
    flow_predictor_substeps=4, fsi_coupling_iterations=4, fsi_coupling_tolerance=1e-3), \
    preset='fsi3', output_dir='out_fsi3')"
```

Key config knobs (all default to legacy/off so other cases are unchanged):
`fsi_coupling_iterations` (1 = explicit single pass), `fsi_coupling_tolerance`,
`accumulate_reprojection_pressure`, `flow_reprojection_iterations`,
`velocity_dirichlet_face_symmetric` (case sets mode 2).

## 6. Open frontiers (if FSI3 flutter is pursued later)

1. **Grid convergence, full FSI** — 2× grid + dt/2 for the whole cylinder+beam
   (tens of GPU-hours) to see whether flutter onsets with sharper wake forcing.
2. **Conservative flux advection** — replace the semi-Lagrangian predictor with
   a `div(u⊗u)` finite-volume convection (core-solver change, stability-gated per
   case); reduces the numerical dissipation that damps the wake–beam energy
   transfer.
3. **IQN-ILS coupling** — quasi-Newton interface solver (parts already exist in
   `simulation_core/coupling/fsi_coupling.py`) for tighter convergence with less
   artificial relaxation damping than Aitken.
4. **MPM beam stiffness calibration** — reconcile the ~35% over-stiffness vs
   Euler–Bernoulli.
