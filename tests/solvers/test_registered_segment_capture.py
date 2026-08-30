"""Full-source capture and sole-publication contracts for registered segments."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tests.solvers import test_hibm_component_face_geometry as geometry_tests


@pytest.fixture(scope="module")
def capture_fixture():
    fixture_type = geometry_tests.HibmComponentFaceGeometryTests
    fixture_type.setUpClass()
    return fixture_type(methodName="runTest")


def _prepare_sources(fixture, relocated_count, mirrored):
    """Create physical 3/7-author embeddings, not just parameterized labels."""
    fixture._load_same_segment_three_author_relocation_fixture()
    boundary = fixture.segment_component_face_boundary
    markers = fixture.segment_component_face_markers
    search = fixture.segment_component_face_search
    markers.set_projection_segments(((0, 1),))

    def transform(row):
        return (3 - row[0], row[1], row[2]) if mirrored else row

    obstacle_sources = [transform(row) for row in (
        (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 3, 1), (1, 3, 2),
    )[:relocated_count]]
    direct_sources = [transform(row) for row in ((1, 2, 2), (2, 2, 2))]
    sign = -1.0 if mirrored else 1.0
    boundary_x = 0.55 if mirrored else 0.45
    sample_x = 0.25 if mirrored else 0.75
    boundary.active_ib_node.fill(0)
    fixture.fluid.obstacle.fill(0)
    for marker, y in enumerate((0.5, 0.75)):
        markers.x_gamma_m[marker] = (boundary_x, y, 0.625)
        markers.v_gamma_mps[marker] = (sign * marker, 0.0, 0.0)
        markers.n_gamma[marker] = (sign, 0.0, 0.0)
    all_sources = obstacle_sources + direct_sources
    for index, source in enumerate(all_sources):
        weight = 0.2 + 0.6 * index / (len(all_sources) - 1)
        boundary.active_ib_node[source] = 1
        fixture.fluid.obstacle[source] = int(source in obstacle_sources)
        boundary.pressure_neumann_normal_field[source] = (sign, 0.0, 0.0)
        boundary.velocity_dirichlet_mps_field[source] = (sign * weight, 0.0, 0.0)
        search.node_boundary_point_m[source] = (boundary_x, 0.5 + 0.25 * weight, 0.625)
        search.node_interior_fluid_point_m[source] = (sample_x, 0.5 + 0.25 * weight, 0.625)
        search.node_projection_marker_indices[source] = (0, 1, -1)
        search.node_projection_marker_weights[source] = (1.0 - weight, weight, 0.0)
        search.nearest_marker[source] = int(weight > 0.5)
    search._last_search_support_radius_xyz_m = (0.5, 0.5, 0.5)
    search._last_search_support_anisotropic = False
    search._last_search_inactive_axis = 2
    return boundary, all_sources, sign


def _assemble(fixture, stage_observer=None):
    # The focused routing fixture has only two markers; the separate numerical
    # gates retain full marker closure.  No geometry audit is bypassed here.
    return fixture._assemble_component_face_ledger(
        interpolate_interior_velocity=False,
        use_marker_geometry=True,
        use_segment_fixture=True,
        provide_marker_topology=True,
        surface_projection_inactive_axis=2,
        close_marker_constraints=False,
        stage_observer=stage_observer,
    )


@pytest.mark.parametrize("relocated_count", (1, 5))
@pytest.mark.parametrize("mirrored", (False, True))
def test_registered_segment_capture_retains_direct_and_shadow_authors_precommit(
    capture_fixture, relocated_count, mirrored,
):
    boundary, all_sources, _ = _prepare_sources(capture_fixture, relocated_count, mirrored)
    ledger_before = capture_fixture._canonical_ledger_bytes()
    stages = []

    def stop_after_capture(stage):
        stages.append(stage)
        if stage == "hibm_velocity_row_full_source_capture_after":
            raise RuntimeError("capture-only test stop")

    with pytest.raises(RuntimeError, match="capture-only test stop"):
        _assemble(capture_fixture, stage_observer=stop_after_capture)

    assembler = boundary._registered_segment_assembler
    raw_valid = assembler.raw_route_valid.to_numpy()
    raw_kind = assembler.raw_route_kind.to_numpy()
    raw_target = assembler.raw_route_target.to_numpy()
    raw_generation = assembler.raw_route_generation.to_numpy()
    records = np.argwhere(raw_valid != 0)
    captured_sources = {tuple(int(value) for value in record[:3]) for record in records}
    assert "hibm_velocity_row_full_source_capture_before" in stages
    assert "hibm_velocity_row_full_source_capture_after" in stages
    assert set(all_sources) == captured_sources
    assert {int(raw_kind[tuple(record)]) for record in records} == {0, 1}
    assert all(int(raw_generation[tuple(record)]) > 0 for record in records)
    target_counts = Counter(
        tuple(int(value) for value in raw_target[tuple(record)]) + (int(record[3]),)
        for record in records
    )
    assert target_counts == {(2, 2, 2, 0): relocated_count + 2}
    assert int(assembler.face_raw_count.to_numpy().max()) == relocated_count + 2
    assert capture_fixture._canonical_ledger_bytes() == ledger_before


@pytest.mark.parametrize("relocated_count", (1, 5))
@pytest.mark.parametrize("mirrored", (False, True))
def test_registered_segment_capture_full_entry_commits_all_raw_authors(
    capture_fixture, relocated_count, mirrored,
):
    boundary, _, sign = _prepare_sources(capture_fixture, relocated_count, mirrored)
    ledger_before = capture_fixture._canonical_ledger_bytes()
    observed = []

    def verify_precommit(stage):
        observed.append(stage)
        assert capture_fixture._canonical_ledger_bytes() == ledger_before

    report = _assemble(capture_fixture, stage_observer=verify_precommit)
    assembler = boundary._registered_segment_assembler
    canonical = report["canonical_velocity_dirichlet_report"]
    assert "hibm_velocity_row_full_source_capture_after" in observed
    assert "hibm_velocity_row_report_after" in observed
    assert canonical["claim_conflict_count"] == 0
    assert canonical["final_owned_component_count"] == 1
    assert canonical["duplicate_claim_component_count"] == relocated_count + 1
    assert assembler.audit_raw_count[2, 2, 2, 0] == relocated_count + 2
    assert assembler.audit_valid[2, 2, 2, 0] == 1
    state = capture_fixture._canonical_component_state((2, 2, 2), 0)
    assert state == {
        "active": True, "value_mps": pytest.approx(sign * 0.5),
        "pressure_mobility": 0.0, "enforcement_weight": 1.0,
        "region_id": 71, "owned": True,
    }
    assert capture_fixture._canonical_ledger_bytes() != ledger_before


def test_registered_segment_capture_corrupt_raw_record_never_publishes(capture_fixture):
    boundary, sources, _ = _prepare_sources(capture_fixture, 5, False)
    ledger_before = capture_fixture._canonical_ledger_bytes()

    def corrupt_after_capture(stage):
        assert capture_fixture._canonical_ledger_bytes() == ledger_before
        if stage == "hibm_velocity_row_full_source_capture_after":
            boundary._registered_segment_assembler.raw_route_nominal_sample_m[sources[0] + (0,)] = (
                float("nan"), 0.55, 0.625,
            )

    with pytest.raises(RuntimeError, match="source audit rejected"):
        _assemble(capture_fixture, stage_observer=corrupt_after_capture)
    assert capture_fixture._canonical_ledger_bytes() == ledger_before


@pytest.mark.parametrize("invalid_normal", ((float("nan"),) * 3, (0.0, 0.0, 0.0)))
def test_registered_segment_capture_rejects_invalid_normal_before_axis_filter(
    capture_fixture, invalid_normal,
):
    boundary, sources, _ = _prepare_sources(capture_fixture, 1, False)
    boundary.pressure_neumann_normal_field[sources[0]] = invalid_normal
    ledger_before = capture_fixture._canonical_ledger_bytes()
    with pytest.raises(RuntimeError, match="full-source capture rejected"):
        _assemble(capture_fixture)
    assert capture_fixture._canonical_ledger_bytes() == ledger_before
