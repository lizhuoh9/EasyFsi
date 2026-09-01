# ANSYS Vertical Flap R25A POD-GRU / Kalman-Residual-GRU Feasibility Goal

This document freezes the CPU-only R25A research harness. It does not contain
holdout results; the dated report is generated only by the parent after an
independent D1 load.

## Scope and matrix

The harness reuses the immutable `AcceptedTrace` and existing evidence loader.
D0 physical steps 1--100 are the only fit data for POD, coefficient
normalization, POD-AR, and neural training. D0 steps 101--200 are used only for
architecture selection and per-seed early stopping. D1 is loaded once, cold
started, and scored on the predeclared physical steps 9--50. Q is exact target
and an error lower bound only; it is excluded from ranking and gates.

The fixed architecture matrix is `(rank, window, hidden) =
[(4,4,8), (8,4,16), (8,8,16), (16,8,16)]`, with seeds 0, 1, and 2 retained.
The families are C0 carry, C1 beta=1 linear extrapolation, exact production K0,
frozen R24 K1, G0 POD-GRU, GK0 K0-residual-GRU, GK1 K1-residual-GRU, and Q.
The GRU is one-layer, residual-output, linear-head, MSE, float64 CPU with
deterministic algorithms, one CPU thread, full-batch AdamW
(`lr=1e-3`, `weight_decay=1e-4`, `grad_clip=1`, `max_epochs=500`,
`patience=50`, `min_delta=1e-8`). Output heads start at zero, so the initial
neural prediction equals its causal baseline.

## Frozen identities and safety

The sealed R24 manifest SHA256 is
`81c7613c1e3a12e0d8f4294b39a22bff9afc6ea3bd49e4abcb16b6dfc830c35d`, its tuning
fingerprint is `3c329be89ee0da8e24840ece6dab8e9f8767dc2b37bd024a2051b7749d799f47`,
the layout is `373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164`,
and `dt_s=0.0005`. The active axes are exactly y/z; x is required to remain
bitwise zero. The production predictor source SHA256 is
`5bbb7735ba43493ce4b768a4c87008f86347192b2c71f12385dde97ee556856b`.
K0 and K1 identities are bound in `model_config.json` and rechecked before
holdout evaluation.

Selection artifacts and all selected states are hashed before the D1 loader is
called. D1 cannot inherit D0 context: its Kalman state, GRU history, and rolling
state are reset. Past innovations are measurement minus raw prediction; current
accepted values, current innovations, rejected trials, and future provenance
are forbidden from features. The output directory and dated report refuse
nonempty overwrite.

## Gate and reporting contract

`PASS_OFFLINE_GRU_VALUE` requires median D1 NRMSE improvements of at least 5%
over C0 and 2% over selected AR, median seed step-fraction beating carry at
least 60%, and no seed more than 10% worse than carry. `PASS_OFFLINE_KALMAN_GRU_VALUE`
is evaluated separately for GK0 and GK1: at least 5% over matching Kalman and
2% over paired G0, median paired seed fraction at least 55%, median p95 rho no
more than 10% worse than G0, and at least two of three seeds strictly improve
against both references. Percentage/fraction thresholds are inclusive; ties do
not count as per-seed favorable improvements.

The report must state that D1 is an independently generated frozen holdout but
the same operating condition already inspected by R24, not an unseen or
cross-condition generalization test. Lower offline error is not solver
acceleration; R24B oracle/IQN evidence is nondeployable; and K1 was not better
than carry. Stop classifications include G0 fail/GK fail, G0 pass/GK fail, G0
fail/any GK pass (review required), G0 pass/any GK pass, and AR approximately
matching G0. R25B/C, CUDA, IQN reuse interaction, online updates, solver
integration, exact20/exact50 probes, commit, and push remain out of scope.

## Initial TDD evidence

The first focused test run intentionally preceded implementation and failed at
collection with `ModuleNotFoundError: No module named
'tools.validation.gru_kalman'`. The parent should preserve that RED evidence in
the final report alongside the current focused GREEN command and result.
