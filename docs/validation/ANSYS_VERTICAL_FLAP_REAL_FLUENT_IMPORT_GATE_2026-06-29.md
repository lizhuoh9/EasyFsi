# ANSYS Vertical-Flap Real Fluent Import Gate 2026-06-29

## Gate

The ANSYS vertical-flap workflow may import real Fluent reference data only after four CSV source exports are present and provenance-complete:

- `fluent_tip_displacement_history.csv`
- `fluent_force_history.csv`
- `fluent_flow_balance_history.csv`
- `fluent_pressure_summary_history.csv`

Each export must include `step = 50` and the expected final physical time for the case. Schema-only files are allowed to keep the pipeline fail-closed, but they cannot unlock parity claims.

## Required Provenance

The metadata file must include complete source document, run id, author, date, Fluent version, mesh/domain source, material model, boundary conditions, export procedure, displacement definition, and force/flow/pressure sign conventions.

The CSV `source` columns must not contain `EasyFsi` placeholders and must not contain `HIBM-MPM` placeholders. EasyFsi/HIBM-MPM output can be compared against Fluent later, but it cannot be used as the Fluent reference source.

## Promotion Rule

The collection validator must report complete real reference coverage before the active manifest promotion is allowed. Complete coverage means all required reference metrics, source provenance, comparison metadata, and tolerances are available and schema-valid.

The active manifest promotion may point at a new versioned Fluent reference contract only after the collection validator passes this gate. Until then, `active_fluent_reference_contract.json` must keep the current incomplete contract active.

The parity runner may compare a candidate against whatever active contract exists, but it cannot claim Fluent parity unless the real Fluent import gate and the active contract gate both pass.
