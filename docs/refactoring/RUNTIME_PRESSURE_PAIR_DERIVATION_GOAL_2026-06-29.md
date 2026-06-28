# Runtime Pressure Pair Derivation Goal - 2026-06-29

## Source context

This goal follows the remote review of commit
`615d40bd8fe5799b9708c41063e7953d015f9e11` on branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

The previous checkpoint was the correct direction:

- `simulation_core/generic_fsi_solver.py` introduced a case-agnostic solver
  boundary and `solve_fsi(...)` entrypoint;
- `AnsysVerticalFlapProblem` adapts the ANSYS vertical-flap selected
  formulation into generic concepts;
- `run_ansys_vertical_flap_generic_solver.py` can run the official case through
  `solve_fsi(...)` for 50 steps;
- EasyFsi output is exported in Fluent-comparable CSV schemas;
- Fluent parity remains unclaimed and the Fluent reference line remains
  fail-closed.

The remaining blocker is not Fluent. The official generic runner still reports:

```text
candidate_status = generic_solver_selected_formulation_step50_transition_passed
pressure_pair_runtime_generation_complete = false
pair_source_status = transition_seeded_from_anchor_artifact
runtime_pressure_pair_generation_pending
```

That means the official generic path still depends on the selected anchor marker
diagnostics JSON. This goal retires that transition dependency from the official
generic path.

## Short active goal reference

The active goal should reference this file instead of inlining the full
contract:

```text
Implement docs/refactoring/RUNTIME_PRESSURE_PAIR_DERIVATION_GOAL_2026-06-29.md:
add a case-agnostic pressure sample pair contract, make the official ANSYS
generic selected-formulation runner use runtime-generated pressure pairs instead
of selected anchor JSON, keep replay/debug mode fail-closed, regenerate artifacts,
verify focused tests, commit, and push the branch.
```

## Objective

Remove the official generic ANSYS vertical-flap selected-formulation runner's
dependency on `fixed_solid_selected_per_face_one_sided_probe0p51_markers.json`
by introducing a runtime pressure sample pair contract and making the official
generic path report runtime-generated pressure pairs.

The implementation must keep the existing 50-step EasyFsi behavior passing while
making the transition state explicit only in debug/replay APIs, not in the
official generic artifact.

## Scope

### In scope

- Add a case-agnostic pressure sample pair contract in `simulation_core`.
- Provide deterministic pair-map diagnostics:
  - `marker_index`
  - `region_id`
  - `inside_cell`
  - `outside_cell`
  - `sample_status`
  - `fallback_status`
  - `diagnostic_reason`
  - `pair_map_sha256`
- Add or extend a provider abstraction for:
  - `runtime_anchored_cell_pair`
  - `replay_from_diagnostics`
- Make the ANSYS generic adapter default to `runtime_anchored_cell_pair` without
  requiring `selected_anchor_markers_json`.
- Keep `replay_from_diagnostics` fail-fast unless a replay JSON path is provided.
- Keep replay/debug support separate from the official generic artifact.
- Regenerate `generic_solver_selected_formulation_diagnostics/` so the official
  matrix reports runtime generation complete.
- Update generic artifact tests so transition-seeded official artifacts are no
  longer accepted.
- Add focused unit/source tests for the pressure pair contract and ANSYS adapter
  mode rules.
- Add explicit workflow `py_compile` entries for:
  - `simulation_core/generic_fsi_solver.py`
  - the new pressure-pair contract module;
  - `run_ansys_vertical_flap_generic_solver.py`;
  - generic architecture and artifact tests.

### Out of scope

- Do not fabricate Fluent source data.
- Do not mark the Fluent reference complete.
- Do not claim Fluent parity.
- Do not change Fluent reference/parity artifacts unless existing tests require
  checksum-only updates.
- Do not add heavy coupled runners to GitHub Actions.
- Do not implement the full generic coupled loop. `solve_fsi(...)` may remain an
  executor-backed boundary for this stage.
- Do not delete old selected anchor artifacts; they remain valid historical and
  replay/debug evidence.

## Runtime pressure pair contract

Add a module such as:

```text
simulation_core/pressure_sample_pairs.py
```

Required types:

```python
@dataclass(frozen=True)
class PressureSamplePair:
    marker_index: int
    region_id: str
    inside_cell: tuple[int, int, int]
    outside_cell: tuple[int, int, int]
    sample_status: str
    fallback_status: str
    diagnostic_reason: str

@dataclass(frozen=True)
class PressureSamplePairMap:
    pairs: tuple[PressureSamplePair, ...]
    pair_map_sha256: str
    provider_mode: str
    fallback_count: int
    selected_count: int
```

Also provide a protocol or callable contract:

```python
class PressureSamplePairProviderProtocol(Protocol):
    def compute_pairs(self, markers, fluid_state, interface_surface) -> PressureSamplePairMap:
        ...
```

The contract must be deterministic and case-agnostic. It must not include
`ANSYS`, `Fluent`, or vertical-flap names.

## Runtime provider requirements

Implement the first conservative runtime provider mode:

```text
runtime_anchored_cell_pair
```

The first implementation may be intentionally narrow, but it must live under the
generic provider contract and produce the same required diagnostics. It should
derive pairs from marker/region/grid information rather than requiring a
selected anchor JSON file in the official generic adapter.

Required behavior:

- runtime-generated pair map has `pair_source_status = runtime_generated`;
- `transition_backed = false`;
- `fallback_count = 0` for the official 50-step artifact;
- `selected_count >= 24` for the official 50-step artifact;
- pair-map SHA is deterministic for equivalent inputs;
- missing/invalid pair data fails closed in tests.

## ANSYS adapter mode rules

Change `AnsysVerticalFlapProblem` to the following shape or equivalent:

```python
@dataclass(frozen=True)
class AnsysVerticalFlapProblem:
    pressure_pair_provider_mode: str = "runtime_anchored_cell_pair"
    selected_anchor_markers_json: str | None = None
    step_count: int = 50
```

Rules:

```text
runtime_anchored_cell_pair -> must not require selected_anchor_markers_json
replay_from_diagnostics -> must require selected_anchor_markers_json
runtime_anchored_cell_pair with JSON -> must not use the JSON as the official source
```

The adapter may still pass enough configuration to the existing lower-level
runner to preserve behavior, but the official generic diagnostics and adapter
contract must no longer report a transition artifact dependency.

## Official generic artifact requirements

Update:

```text
validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/
```

The official matrix must report:

```text
candidate_status = generic_solver_selected_formulation_step50_passed
pressure_pair_policy.mode = runtime_anchored_cell_pair
pressure_pair_policy.pair_source_status = runtime_generated
pressure_pair_runtime_generation_complete = true
transition_artifact_dependency = false
```

It must not include:

```text
runtime_pressure_pair_generation_pending
transition_seeded_from_anchor_artifact
```

50-step gates must remain:

```text
completed_step_count = 50
invalid_marker_count_max = 0
sample_pair_fallback_count_max = 0
one_sided_marker_count_min >= 24
force_action_reaction_residual_max_n <= 1e-8
```

The Fluent-comparable EasyFsi CSV exports must keep the existing headers and 50
rows.

## Tests

Add or update:

```text
tests/solvers/test_pressure_sample_pair_provider_contract.py
tests/cases/test_ansys_vertical_flap_fsi.py
tests/integration/test_ansys_vertical_flap_generic_solver_artifacts.py
tests/contracts/test_generic_fsi_solver_architecture.py
```

Required coverage:

- pair map SHA is deterministic;
- missing pairs fail closed;
- fallback and selected counts are computed from pair rows;
- required pair schema fields are preserved;
- pressure pair provider contract has no ANSYS/Fluent/case-specific names;
- `build_ansys_vertical_flap_generic_problem(step_count=50)` does not require
  `selected_anchor_markers_json`;
- `replay_from_diagnostics` without JSON fails fast;
- runtime mode with a JSON path does not use the JSON as the official source;
- official generic artifact is runtime-generated, not transition-seeded;
- replay/debug mode remains represented by source/unit tests, not by the
  official generic artifact.

## Verification plan

Use the reliable local interpreter:

```text
D:\working\taichi\env\python.exe
```

Run:

```text
D:\working\taichi\env\python.exe -m py_compile simulation_core/generic_fsi_solver.py simulation_core/pressure_sample_pairs.py validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py tests/contracts/test_generic_fsi_solver_architecture.py tests/integration/test_ansys_vertical_flap_generic_solver_artifacts.py tests/solvers/test_pressure_sample_pair_provider_contract.py
D:\working\taichi\env\python.exe validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py
D:\working\taichi\env\python.exe -m unittest -v tests.solvers.test_pressure_sample_pair_provider_contract tests.contracts.test_generic_fsi_solver_architecture tests.integration.test_ansys_vertical_flap_generic_solver_artifacts
D:\working\taichi\env\python.exe -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic
git diff --check
```

## Push condition

Only push after:

- the goal file exists and the short active goal references it;
- official generic artifact reports runtime generation complete;
- no official generic artifact contains `transition_seeded_from_anchor_artifact`;
- focused tests and fail-closed Fluent tests pass;
- staged diff has been reviewed for accidental Fluent parity overclaiming.
