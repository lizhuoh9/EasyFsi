# Squid Jet FSI — Coupling Architecture Review

**Status:** root cause proven; four fixes attempted and defeated; architecture re-evaluated.
**Tree state:** clean (`e99a851`) — every experiment below was a flag, an env-gated diagnostic, or a change that was fully reverted. No production code is modified.
**Date:** 2026-06-17

---

## 0. Executive summary

The squid jet case (`hibm_mpm_sharp`, real‑CAD STEP source config) does not eject a jet: `interior_divergence_l2` accumulates each step until the projection guard trips (step 7 of the 600‑step run, `1.075e-2 > 1.0e-2`). The mantle displaces volume (`volume_flux_m3s ≈ 1e-5`) but essentially nothing exits the nozzle (`outlet_flow_negative_z_m3s ≈ 1e-10`).

**Proven root cause:** the sharp HIBM‑MPM imposes the membrane velocity as a **hard velocity‑Dirichlet pin** on the immersed‑boundary band. Every pressure‑routing operator skips pinned cells, so the displaced volume is both *uncorrectable* and *unreachable from the z‑min outlet*. It cannot drain → divergence accumulates → no jet.

**Four fixes were attempted on the sharp path; all were defeated by the same wall** (§3). The existing **diffuse coupling mode** (`legacy_projected_reduced`) *eliminates* the walling (`interior_div_l2 = 0`) but is **numerically unstable** — a pressure runaway at the membrane interface blows the velocity up in 2–3 steps (§5).

**Conclusion:** neither existing coupling mode is a complete solution; they fail in opposite, complementary ways. A complete fix needs a coupling that is **both non‑pinning (avoid walling) and stably strong‑coupled (avoid blowup)**. Recommended path: **port the sharp path's convergent strong coupling (Aitken/IQN‑ILS) onto the diffuse/projected‑IBM path** (§6).

---

## 1. The failure (baseline, `hibm_mpm_sharp`)

Run: `_codex_validation/phaseE_sharp_fsi_iters6_tol25e-5_600step_20260617_002/`.

- Guard trips at **step 7**: `interior_divergence_l2 = 1.0755e-2 > 1.0e-2` ([cases/squid_soft_robot.py:6114-6121](cases/squid_soft_robot.py)).
- `interior_div_l2` grows ~linearly: `1.94e-4 → 1.21e-3 → 2.37e-3 → 3.92e-3 → 5.85e-3 → 8.15e-3 → 1.075e-2` (steps 1–7). Steady accumulation, **not** a blow‑up.
- **No jet:** `volume_flux_m3s` (mantle displacement) grows `3.1e-7 → 1.1e-5`; `outlet_flow_negative_z_m3s` stays `~1e-10`. Ratio `~2.6e-5`.
- Everything *else* is healthy: pressure CG converges (`converged_all=True`, `breakdown=0`); FSI marker fixed point converges (3 iters, residual `3.1e-5 m/s`); HIBM velocity/pressure reconstruction invalid counts = 0.

Exclusion experiments (from the original report): more post‑Dirichlet projections do not help; `time_step_scale=0.5` fails at the *same physical time* → a volume‑conservation/topology problem, **not** CFL.

---

## 2. Root cause (sharp velocity‑Dirichlet pin walls the cavity)

The mantle's displaced volume is imposed **only** as a hard velocity‑Dirichlet overwrite on the IB band cells: `_apply_velocity_dirichlet_boundary_rows_kernel` ([simulation_core/fluid.py:2830](simulation_core/fluid.py)), applied after the pressure solve + gradient subtract at [fluid.py:7607](simulation_core/fluid.py).

**The crux — every routing operator gates on `velocity_dirichlet_boundary_active == 0`:**
- pressure RHS / correctable‑face test `_divergence_stencil_has_pressure_correctable_face` ([fluid.py:3991-4027](simulation_core/fluid.py)),
- gradient subtraction ([fluid.py:5852-5863](simulation_core/fluid.py)),
- z‑min outlet reachability seed + flood ([fluid.py:1852](simulation_core/fluid.py) / [1789-1824](simulation_core/fluid.py)),
- local divergence cleanup ([fluid.py:6071-6096](simulation_core/fluid.py)).

So the pinned band's divergence is **uncorrectable** (the projection cannot touch it) **and unreachable** (the flood cannot pass through it). Combined with a **coarse grid** (~2.5 mm cells vs a ~1.6 mm nozzle aperture and a thin cavity), the 1‑cell‑thick pin band consumes the thin cavity, leaving no routable interior fluid.

**Staged‑divergence evidence** (history columns `*_divergence_l2`):
```
pre(pre-projection)=1.7e-2 → projection=5.8e-3 (only ~50% removed) → post=1.27e-2 (grows back)
pressure_outlet_projection_*_velocity_flux ~1e-9   (needed ~1e-5 to drain the displacement)
```
The projection removes only ~half because the pinned band over‑determines it; the outlet flux it can drive is ~4 orders of magnitude too small.

> Note: an earlier multi‑agent diagnosis attributed the seal to obstacle/air/barrier conversions sealing the nozzle. **That theory was refuted by experiment** (§3.1). The operative mechanism is the velocity‑Dirichlet pin, confirmed by the diffuse‑mode result (§4).

---

## 3. Fixes attempted on the sharp path — all defeated by the same wall

### 3.1 Keep‑open (exempt the nozzle/cavity from obstacle conversions) — INERT
Added a `hibm_keep_open_cell` mask exempting the drainage channel from the internal‑obstacle stamp, the solid‑band conversion, the air‑backed (S2‑A12) conversion, and the reachability barrier. **Result: `interior_div_l2` byte‑identical to baseline** across three runs (band/air/barrier exempt; + internal‑stamp exempt). `hibm_internal_obstacle_cell_count = 0` — the internal stamp never fires. A standalone unit test proved the exemptions *work*; they are simply **inert against this failure** (the cavity is not sealed by obstacle conversions). The base obstacle mask is wide open (2000+ open cells/z‑level). Reverted.

### 3.2 Option A (re‑project after the constraint) — REJECTED in analysis
A second pressure projection after the post‑projection velocity overwrite re‑hits the *same* `velocity_dirichlet_boundary_active==0` gate: the re‑injected divergence sits in pinned cells the projection, the gradient subtract, and the reachability flood all skip. The re‑projection only makes the diagnostic honestly report the still‑trapped divergence. Not implemented.

### 3.3 L2 (volume‑source replace) — IMPLEMENTED, TESTED, DEFEATED
Deliver the mantle marker normal flux `v·n·A` as an interior `volume_source_s` (which the projection *can* route — proven by [tests/test_squid_latest_core_config.py:2670](tests/test_squid_latest_core_config.py), an interior source routes to the z‑min outlet with ratio≈1), while suppressing the wall‑normal component of the pin to avoid double counting. All kernels compiled and were unit‑correct (flux conservation exact; pinned‑cell gating correct; normal suppression correct).

**Result: `interior_div_l2` byte‑identical to baseline.** Instrumented diagnostic explains why: of **35574 markers, ~100% unplaced at 1× probe and still ~90% unplaced at 4× probe** — every marker's probe lands in a pinned (`velocity_dirichlet_active`) or obstacle cell. **The cavity around the membrane is entirely pinned band + obstacle, so there is no routable interior fluid to source into.** The minority that placed *did* route (outlet flow rose 10–30×, confirming the routing mechanism), but biased‑negative‑sign and ~4 orders too small. Reverted.

### 3.4 Common thread
Keep‑open, re‑project, and L2 are all blocked by the **same wall**: the velocity‑Dirichlet membrane band walls the cavity from the outlet. The only sharp‑side lever that addresses it is **L1 — relax the pin** (partial `velocity_dirichlet_boundary_projection_weight` at [fluid.py:5852-5859](simulation_core/fluid.py) + reachability change). Not attempted (user opted to re‑evaluate architecture).

---

## 4. Architecture re‑evaluation — the diffuse mode confirms the diagnosis

The codebase already exposes a non‑pinning coupling: `fsi_coupling_mode = legacy_projected_reduced` ([constants hibm_mpm.py:18-19](simulation_core/hibm_mpm.py); CLI [squid_soft_robot.py:13988](cases/squid_soft_robot.py); sharp is the default at [squid_soft_robot.py:554](cases/squid_soft_robot.py)). It runs `advance_projected_ibm_region_pair_fluid_step` ([simulation_core/projected_ibm.py:456](simulation_core/projected_ibm.py)) which uses **spread‑force + `volume_source`** ([projected_ibm.py:588-645](simulation_core/projected_ibm.py)) and **no hard velocity pin**.

**Run** (`--fsi-coupling-mode legacy_projected_reduced --source-config <STEP>`, zero code):
- **Step 1: `interior_div_l2 = 0.000`.** The walling/divergence‑accumulation problem is **gone** — the non‑pinning coupling routes the volume and the projection is divergence‑free. This **confirms the sharp pin is the root cause** of the divergence accumulation.
- **Step 2: CFL blows up** (`cfl=7.2 ≥ 0.5` guard).

| | walling (divergence) | stability |
|---|---|---|
| **sharp** (`hibm_mpm_sharp`) | ❌ walls cavity → divergence accumulates, no jet | ✅ stable |
| **diffuse** (`legacy_projected_reduced`) | ✅ **zero divergence** (no walling) | ❌ blows up in 2–3 steps |

The two modes fail in **opposite** ways.

---

## 5. Diffuse explosion diagnosis

Env‑gated per‑field instrumentation at the end of the diffuse step (reverted). Max magnitudes:

| field | step 1 | step 2 | growth | location |
|---|---|---|---|---|
| `\|pressure\|` | 4.5e2 Pa | **1.47e6 Pa** | **3200×** | cell (37,29,43) |
| `\|velocity\|` | 1.1e-2 | **35 m/s** | 3000× | (36,30,44) (adjacent) |
| `\|force\|` | 2.4e3 | 4.4e4 | 18×/step | (35,30,45) |
| `\|volume_source\|` | 5.1e-2 | 2.4e-1 | 5× (stable) | (19,35,56) |

All spikes are at **z ≈ 1.006–1.011 m — the region‑7/8 membrane/cavity interface**. The **pressure** explodes first (3200×) and drags the velocity with it; the force grows 18×/step (a runaway feedback); `volume_source` is stable.

**Interpretation:** not a localized degenerate‑marker source spike — a **systemic FSI interface pressure‑coupling instability** (a runaway force↔pressure feedback at the membrane, amplified by interface ill‑conditioning). This is why moderate relaxation / smaller dt only delayed it by a step, and why *iterating* the correction made it **worse** (`--ibm-correction-iterations 8` → `cfl=1e6` at step 1): **the diffuse path has no convergent strong coupling**. Stabilization attempts (`--time-step-scale 0.2 --interface-reaction-relaxation 0.1 --adaptive-fluid-substeps`) reached step 3 then blew up (`cfl=169`).

---

## 6. Conclusion and recommended paths

Neither existing coupling mode solves the squid case. A complete fix needs a coupling that is **both non‑pinning** (to avoid the sharp wall) **and stably strong‑coupled** (to avoid the diffuse blowup).

| option | what | effort | risk | notes |
|---|---|---|---|---|
| **B (recommended)** | Port the sharp path's convergent strong coupling (Aitken / IQN‑ILS adaptive relaxation, already used by the sharp marker fixed point at [squid_soft_robot.py:9126](cases/squid_soft_robot.py)) onto the diffuse/`projected_ibm` path | medium‑high (new dev in `projected_ibm.py` + likely tuning) | medium | Keeps the diffuse `div=0` while damping the interface pressure runaway. Directly targets the diagnosed instability. |
| C | Cut‑cell / ghost‑fluid pressure coupling (boundary cuts cells; pressure couples across with a proper flux BC) | highest (pressure‑solve rewrite) | high | The rigorous long‑term sharp fix; removes both the walling and the over‑constraint. |
| L1 | Partial‑weight the sharp pin ([fluid.py:5852-5859](simulation_core/fluid.py)) + let reachability pass | medium | med‑high | Un‑walls sharp but changes no‑slip fidelity; the cavity is also thin (see D). |
| D | Refine the fluid grid near the cavity/nozzle | low code, high compute | low | The thin cavity / ~1.6 mm aperture is only ~1–2 cells; **likely needed regardless** so any coupling has routable interior + a ≥3‑cell throat. Combine with B/C/L1. |

**Recommendation:** pursue **B**, almost certainly combined with **D**. The diffuse mode already solves the hard half (no walling, `div=0`); the remaining work is to give it the strong coupling the sharp path has.

---

## 7. Reproduction

Baseline failure (sharp):
```
"D:/TOOL/Anaconda/python.exe" run_simulation.py squid-soft-robot \
  --source-config _codex_validation/codex_squid_realcad_step_source_config_20260615_002/simulation_config.step_derived.json \
  --steps 8 --fsi-coupling-iterations 6 --fsi-marker-coupling-tolerance-mps 2.5e-4 \
  --projection-divergence-tolerance 0.1 --pressure-solve-failure-policy report --progress --progress-interval 1
```
Diffuse mode (zero divergence, then CFL blowup):
```
"D:/TOOL/Anaconda/python.exe" run_simulation.py squid-soft-robot \
  --source-config <same> --fsi-coupling-mode legacy_projected_reduced \
  --steps 4 --projection-divergence-tolerance 0.1 --pressure-solve-failure-policy report --progress --progress-interval 1
```
Validation artifacts (history.csv, console logs, masks) live under `_codex_validation/` (`phaseE_*`, `L2_*`, `diffuse_*`).

---

## 8. Do NOT retry (proven inert / defeated)

- **Keep‑open / obstacle‑conversion exemptions** (band, air‑backed S2‑A12, reachability barrier, internal‑obstacle stamp): byte‑identical to baseline; the cavity is not sealed by obstacle conversions.
- **Re‑project after the velocity overwrite (Option A)**: re‑hits the same `velocity_dirichlet_boundary_active==0` gate.
- **Naive volume‑source injection (L2)** *without* un‑walling the band: the source cannot be placed (~90% of markers find only pinned/obstacle cells near the membrane).
- **Iterating the diffuse correction** (`--ibm-correction-iterations` up) to stabilize: it *diverges* (no convergent relaxation) and makes the blowup worse.

The single highest‑leverage finding: **the divergence/walling problem is caused by the sharp velocity‑Dirichlet pin and is eliminated by a non‑pinning coupling (`div=0`).** The remaining open problem is purely the diffuse path's **FSI interface stability** (no strong coupling).
