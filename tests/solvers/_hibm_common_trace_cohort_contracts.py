"""Source-matched compact Turek-Hron common-trace transaction contracts.

Original affine/tilted cases, R15 fallback variants, and rejection controls
share the existing assembly/reset entrypoints.
This mixin creates no runtime and needs the shared capacity-four test class.
"""

import copy
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent / 'fixtures' / 'turek_hron_common_trace_cohorts.json'
DATA_SHA = '529df0f530f5fb9fe40dca9bbae58ed32cb8360f5c32a3d4ad39f2094d252db4'
assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == DATA_SHA
DATA = json.loads(DATA_PATH.read_text())
R14_DATA_PATH = DATA_PATH.with_name('turek_hron_common_trace_r14_tilted.json')
R14_DATA_SHA = '4e0888122217433d64b571b1e3caab9c06ec99355cefbabb166e4a51a9737a37'
R15_DATA_PATH = DATA_PATH.with_name('turek_hron_common_trace_r15_fallback.json')
R15_DATA_SHA = 'c41415690040c7a10533d29aeacde9ce352403203345ad0c679b95d3cdecb727'
COORDINATE_FIELDS = tuple(DATA['coordinate_fields'])
SUPPORT_ATTRIBUTES = ('_last_search_support_radius_xyz_m', '_last_search_support_anisotropic', '_last_search_inactive_axis')
AFFINE_TOLERANCE_MPS = 2e-6  # F32 interpolation/serialization; a wrong MAC z center differs by ~3e-4.
EXPECTED_CACHE_IDENTITY = {305: (6, 2, 0, 1), 306: (6, 2, 0, 1), 347: (5, 6, 0, 0)}
EXPECTED_PREPARE_COUNT = {305: 3, 306: 3, 347: 2}


def affine_value(point, constant=None):
    if constant is not None:
        return float(constant)
    return 0.2 + 0.25 * float(point[1]) + 0.1 * float(point[2])


def affine_velocity(point, constant=None):
    return (0.0, 0.0, float(np.float32(affine_value(point, constant))))


def mac_velocity(coordinates, constant=None):
    velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
    if constant is not None:
        velocity[..., 2] = np.float32(constant)
        return velocity
    y = np.asarray(coordinates['cell_center_y_m'], dtype=np.float64)
    z = np.asarray(coordinates['cell_face_z_m'][:4], dtype=np.float64)
    velocity[..., 2] = (0.2 + 0.25 * y[:, None] + 0.1 * z[None, :]).astype(np.float32)[None, :, :]
    return velocity


def local_row(original, origin):
    if tuple(original) == (-1, -1, -1):
        return (-1, -1, -1)
    result = tuple(int(original[i]) - origin[i] for i in range(3))
    assert all(0 <= value < 4 for value in result), ('row outside compact fixture', original)
    return result


def local_winner(original, origin):
    if original == np.iinfo(np.int64).max:
        return int(original)
    row = (original // (96 * 400), (original // 400) % 96, original % 400)
    i, j, k = local_row(row, origin)
    return (i * 4 + j) * 4 + k


def accepted_values(payload, origin, constant=None):
    """Map recorded addresses; change sampled velocities only for the affine field."""
    values = dict(payload['fields'])
    for name in ('velocity_dirichlet_relocation_shadow_source_row',
                 'velocity_dirichlet_relocation_shadow_storage_base_row'):
        values[name] = local_row(values[name], origin)
    name = 'velocity_dirichlet_relocation_winner_source_linear_key'
    values[name] = local_winner(values[name], origin)
    for prefix, valid in (('velocity_dirichlet_component_face_actual_sample_', 'valid'),
                          ('velocity_dirichlet_relocation_shadow_sample_', None)):
        flag = values[prefix + valid] if valid else values['velocity_dirichlet_relocation_shadow_claim_valid']
        if flag:
            values[prefix + 'velocity_mps'] = affine_velocity(values[prefix + 'point_m'], constant)
    return values


def shadow_source_bindings(case):
    """Use the captured identity to address every valid S original source."""
    origin = case['origin']
    direct_rows = set()
    for source in case['sources']:
        if source['kind'] == 0 and source['active']:
            row = local_row(source['original_source_row'], origin)
            assert row == tuple(source['local_original_source_row']), ('direct source mapping', source['slot'])
            direct_rows.add(row)
    bindings = []
    for index, payload in enumerate(case['accepted']):
        fields = payload['fields']
        if not fields['velocity_dirichlet_relocation_shadow_claim_valid']:
            continue
        original_source = tuple(fields['velocity_dirichlet_relocation_shadow_source_row'])
        source_row = local_row(original_source, origin)
        storage_row = local_row(payload['original_storage_row'], origin)
        assert storage_row == tuple(payload['local_storage_row']), ('shadow storage mapping', index)
        matches = [source for source in case['sources'] if source['kind'] == 1
                   and tuple(source['original_source_row']) == original_source
                   and tuple(source['local_storage_row']) == storage_row]
        assert len(matches) == 1, ('captured shadow source identity', original_source, storage_row)
        source = matches[0]
        assert source_row == tuple(source['local_original_source_row']), ('shadow source mapping', source['slot'])
        assert all(0 <= value < 4 for value in source_row), ('invalid shadow source row', original_source)
        base = local_row(fields['velocity_dirichlet_relocation_shadow_storage_base_row'], origin)
        assert base == tuple(source['local_geometry_base_row']), ('shadow base mapping', source['slot'])
        key = local_winner(fields['velocity_dirichlet_relocation_winner_source_linear_key'], origin)
        assert key == (source_row[0] * 4 + source_row[1]) * 4 + source_row[2], ('shadow winner identity', source['slot'])
        assert source_row not in direct_rows, ('shadow original source collides with active D actual payload', source_row)
        bindings.append({'accepted_index': index, 'slot': source['slot'], 'original_source': original_source,
                         'source': source_row, 'storage': storage_row, 'active_direct_collision': False})
    assert len({binding['source'] for binding in bindings}) == len(bindings), 'duplicate original S source'
    return bindings, sorted(direct_rows)


def sample_payload_guards(actual, shadow):
    guards = {'valid_exact': actual['valid'] == shadow['valid'] == 1}
    for name in ('point_m', 'velocity_mps'):
        left, right = (np.asarray(values[name], dtype=np.float32) for values in (actual, shadow))
        guards[name + '_exact_f32'] = left.shape == right.shape == (3,) and left.tobytes() == right.tobytes()
    return guards


def precompute_guards(case, observation):
    cache = observation['cache']
    identity = tuple(cache[name] for name in ('first_author_linear_key', 'second_author_linear_key',
                                              'first_author_kind', 'second_author_kind'))
    expected_identity = (case['expected_cache_identity'] if 'expected_cache_identity' in case
                         else EXPECTED_CACHE_IDENTITY[case['original_face'][2]])
    guards = {'cache_identity': identity == tuple(expected_identity),
              'cache_admission_and_full': cache['admission_valid'] == cache['full_valid'] == 1,
              'route_candidates': observation['route_candidate_slots'] == case['expected_original_routed_slots']}
    for name in ('boundary_point_m', 'normal', 'nominal_probe_m'):
        expected_pair = case.get('expected_fallback_pair', case['original_cached_pair'])
        expected = np.asarray(expected_pair[name], dtype=np.float32)
        actual = np.asarray(cache[name], dtype=np.float32)
        if 'expected_fallback_pair' in case:
            tolerance = float(cache['geometry_tolerance'])
            guards[name + '_finite'] = actual.shape == (3,) and bool(np.all(np.isfinite(actual)))
            if name != 'normal':
                guards[name + '_same_physical_trace'] = (math.isfinite(tolerance) and tolerance > 0
                    and float(np.linalg.norm(actual.astype(np.float64) - expected)) <= tolerance)
        else:
            guards[name + '_exact_f32'] = actual.shape == (3,) and actual.tobytes() == expected.tobytes()
    # U_B changed with the manufactured velocity; the original physical U_B is not an oracle.
    expected_boundary_value = affine_value(cache['boundary_point_m'], case.get('constant_velocity_mps'))
    guards['boundary_target_matches_affine_field'] = (math.isfinite(cache['boundary_target_mps'])
        and abs(cache['boundary_target_mps'] - expected_boundary_value) <= AFFINE_TOLERANCE_MPS)
    guards['fallback_flag'] = observation['fallback_valid'] == case.get('expected_fallback_valid', 0)
    if 'expected_prior_adjacent_direct' in case:
        guards['prior_adjacent_direct'] = observation['prior_adjacent_direct'] == case['expected_prior_adjacent_direct']
    if case.get('capture_id') == 'r14_tilted_z305':
        shadow = next(route for route in observation['routes'] if route['slot'] == 3)
        storage = next(source['local_storage_row'] for source in case['sources'] if source['slot'] == 3)
        primary_face = [*storage[:2], storage[2] + shadow['selected']]
        faces = case['route_and_cache_faces']
        # The materialized Splus route and the captured cache address are
        # separate observations; this does not query the canonical B/Q selector.
        guards.update(
            r14_splus_primary_route_face_306=(primary_face == faces['local_splus_primary_route_face']
                and faces['original_splus_primary_route_face'] == [0, 50, 306]),
            r14_cached_certified_face_305=(case['local_face'] == faces['local_cached_certified_face']
                and faces['original_cached_certified_face'] == case['original_face'] == [0, 50, 305]),
            r14_splus_route_differs_from_cached_face=primary_face != case['local_face'],
            r14_Splus_candidate_excluded=(shadow['selected'] == 1 and shadow['expected_offset'] == 0
                and observation['route_candidate_slots'] == [0, 2]),
            r14_consumed_membership_separate=(case['expected_consumed_slots'] == [0, 2, 3]
                and case['expected_consumed_mask'] == 13),
            r14_recorded_tilt_nonzero=cache['normal'][2] != 0.0,
        )
        for name in ('geometry_tolerance', 'clamp_support_ratio'):
            guards['r14_' + name + '_exact_f32'] = (
                np.float32(cache[name]).tobytes() == np.float32(case['original_cached_pair'][name]).tobytes())
        for name in ('direct_face_owner_shadow', 'endpoint_clamped'):
            guards['r14_' + name + '_exact'] = cache[name] == case['original_cached_pair'][name]
    return guards, expected_boundary_value



COMMON_MODE = 256
EXPECTED_SEED_MASK = {305: 12, 306: 12, 347: 5}
EXPECTED_CONSUMED_MASK = {305: 13, 306: 13, 347: 9}
EXPECTED_ACTUAL_KEYS = {305: [5, 6], 306: [5, 6], 347: [5, 2]}
EXPECTED_OWNER_INDICES = {305: [0, 1, -1], 306: [1, 2, -1], 347: [1, 2, -1]}
COMMON_NEUTRAL_FIELDS = {
    'velocity_dirichlet_component_face_segment_pair_owner_indices': -1,
    'velocity_dirichlet_component_face_common_trace_seed_mask': 0,
    'velocity_dirichlet_component_face_common_trace_proved_mask': 0,
    'velocity_dirichlet_component_face_common_trace_consumed_mask': 0,
    'velocity_dirichlet_component_face_common_trace_fallback_valid': 0,
    'velocity_dirichlet_component_face_common_trace_fallback_prior_adjacent_direct': 0,
}
FAULT_CASES = {
    'bad_extra_source_anchor': {'z': 305, 'slot': 0},
    'bad_actual_sample': {'z': 347, 'slot': 3},
    'shadow_original_velocity_mismatch': {'z': 347, 'slot': 3},
    'wrong_shadow_base': {'z': 305, 'slot': 3},
    'unused_cached_direct_seed_anchor': {'z': 347, 'slot': 2},
    'reversed_registered_owner_edge': {'z': 305, 'slot': 2},
    'contaminated_common_mode': {'z': 305, 'slot': None},
    'r15_bad_actual_projection': {'z': 342, 'slot': 3},
    'r15_actual_shadows_only': {'z': 342, 'slot': None, 'phase': 'after_precompute'},
    'r15_conflicting_candidate_probes': {'z': 342, 'slot': 2},
}


def common_trace_guards(case, record):
    z = case['original_face'][2]
    consumed, proved = record['consumed_mask'], record['proved_mask']
    expected = case.get('expected_common') or {
        'seed_mask': EXPECTED_SEED_MASK[z], 'consumed_mask': EXPECTED_CONSUMED_MASK[z],
        'actual_keys': EXPECTED_ACTUAL_KEYS[z], 'owner_indices': EXPECTED_OWNER_INDICES[z]}
    return {'common_mode_exact': record['mode'] == COMMON_MODE,
            'seed_mask_exact': record['seed_mask'] == expected['seed_mask'],
            'consumed_mask_exact': consumed == expected['consumed_mask'],
            'consumed_subset_of_proved': 0 <= proved <= 15 and (consumed & proved) == consumed,
            'actual_first_two_keys_exact': record['actual_keys'] == expected['actual_keys'],
            'registered_owner_indices_exact': record['owner_indices'] == expected['owner_indices']}


def bad_extra_target_rejection_guards(case, observed):
    """Validate the local rejection without relying on the globally first failure."""
    prepared = observed['prepared']
    common = prepared['common_trace']
    precomputed, current = observed['precompute'], prepared['cache_and_routes']
    slots = case['expected_original_routed_slots']
    sources = {source['slot']: source for source in case['sources']}
    keys = [sum(int(value) * stride for value, stride in zip(
        sources[slot]['local_original_source_row'], (16, 4, 1))) for slot in slots]
    assert slots == [0, 2, 3] and keys == [5, 6, 2]
    node_count = 4 * 4 * 4
    prepare_witness = lambda key: -2 - (3 * node_count + node_count - 1 - key)
    cardinality_witness = lambda first, second: -2 - ((first + 1) * (node_count + 1) + second + 1)
    alternatives = {
        'prepare_pair_Dminus_Dplus': [prepare_witness(keys[0]), prepare_witness(keys[1])],
        'prepare_pair_Dminus_Splus': [prepare_witness(keys[0]), prepare_witness(keys[2])],
        'prepare_author_cardinality': [cardinality_witness(keys[0], keys[1]), cardinality_witness(keys[2], -1)],
    }
    witnesses = prepared['native_author_witnesses']
    matched_paths = [name for name, expected in alternatives.items() if witnesses == expected]
    expected_membership = sum(1 << slot for slot in slots)
    proved = common['proved_mask']
    guards = {
        'local_target_exact': tuple(case['local_face']) == (0, 1, 2) and case['original_face'][2] == 305,
        'precompute_cache_guards': all(precomputed['guards'].values()),
        'prepared_cache_guards': all(prepared['cache_guards'].values()),
        'cache_and_candidate_routes_unchanged': all(current[name] == precomputed[name]
            for name in ('cache', 'routes', 'route_candidate_slots')),
        'three_actual_claims': prepared['count'] == 3,
        'cached_Dplus_Splus_seed_exact': common['seed_mask'] == 12,
        'later_Dplus_Splus_still_proved': 0 <= proved <= 15 and (proved & 12) == 12,
        'bad_Dminus_not_proved': (proved & 1) == 0,
        'actual_membership_not_covered': expected_membership == 13 and (expected_membership & proved) != expected_membership,
        'common_mode_not_admitted': (common['mode'] & COMMON_MODE) == 0,
        'consumed_certificate_not_published': common['consumed_mask'] == 0,
        'failure_only_local_witnesses': len(witnesses) == 2 and all(value < -1 for value in witnesses),
        'registered_prepare_or_cardinality_provenance': bool(matched_paths),
    }
    return {'component_face': list(case['local_face']), 'component_axis': 2,
            'source_slot_order': slots, 'original_source_keys': keys, 'expected_membership_mask': expected_membership,
            'native_author_witnesses': witnesses, 'matched_native_encoding': matched_paths,
            'expected_failure_witnesses': alternatives, 'guards': guards,
            'all_guards_passed': all(guards.values()), 'global_first_face_required': False}


def native_health_error_record(error):
    frames = traceback.extract_tb(error.__traceback__)
    final = frames[-1] if frames else None
    origin = None if final is None else {'filename': final.filename, 'function': final.name, 'line': final.lineno}
    allowed = (type(error) is RuntimeError and final is not None
               and final.filename.replace('\\', '/').endswith('/simulation_core/coupling/hibm_mpm/core.py')
               and final.name in ('_validate_canonical_velocity_dirichlet_target_conflict_precommit',
                                  '_validate_canonical_velocity_dirichlet_relocation_precommit'))
    return {'type': type(error).__name__, 'message': str(error), 'origin': origin,
            'native_health_RuntimeError': allowed}


def ledger_hashes(values):
    return {name: {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            for name, payload in values}


class CommonTraceCohortContractMixin:
    def _systemic_load_fixture(self, case):
        self._reset_component_face_claim_fixture(use_segment_fixture=True)
        fluid, search, markers, boundary = (self.fluid, self.segment_component_face_search,
                                            self.segment_component_face_markers, self.segment_component_face_boundary)
        self.assertEqual(tuple(self._GRID_NODES), (4, 4, 4))
        for name, values in case['coordinates'].items():
            getattr(fluid, name).from_numpy(np.asarray(values, dtype=np.float32))
        fluid.obstacle.from_numpy(np.asarray(case['obstacle'], dtype=np.int32))
        marker = case['markers']
        velocities = tuple(affine_velocity(point, case.get('constant_velocity_mps')) for point in marker['positions'])
        markers.load_markers(positions_m=marker['positions'], velocities_mps=velocities,
                             normals=marker['normals'], areas_m2=marker['areas'], region_ids=marker['regions'])
        markers.set_projection_segments(marker['edges'])
        search._last_search_support_radius_xyz_m = tuple(DATA['support_radius_xyz_m'])
        search._last_search_support_anisotropic = True
        search._last_search_inactive_axis = 0
        search.node_kind_code.fill(0)
        for source in case['sources']:
            row = tuple(source['local_original_source_row'])
            boundary.active_ib_node[row] = source['active']
            search.node_kind_code[row] = source['node_kind']
            search.node_boundary_point_m[row] = source['B']
            search.node_interior_fluid_point_m[row] = source['nominal_Q']
            boundary.pressure_neumann_normal_field[row] = source['normal']
            search.node_projection_marker_indices[row] = source['local_projection_indices']
            search.node_projection_marker_weights[row] = source['weights']
            search.nearest_marker[row] = source['local_nearest_marker']
            a, b = source['local_projection_indices'][:2]
            target = np.float32(source['weights'][0] * velocities[a][2] + source['weights'][1] * velocities[b][2])
            boundary.velocity_dirichlet_mps_field[row] = (0.0, 0.0, float(target))
        fluid.velocity.from_numpy(mac_velocity(case['coordinates'], case.get('constant_velocity_mps')))

    def _systemic_sample_input(self, row, *, shadow=False):
        boundary = self.segment_component_face_boundary
        prefix = ('velocity_dirichlet_relocation_shadow_sample_' if shadow
                  else 'velocity_dirichlet_component_face_actual_sample_')
        flag = ('velocity_dirichlet_relocation_shadow_claim_valid' if shadow else prefix + 'valid')
        return {'valid': int(getattr(boundary, flag)[row]),
                **{name: [float(value) for value in getattr(boundary, prefix + name)[row]]
                   for name in ('point_m', 'velocity_mps')}}

    def _systemic_seed_accepted(self, case):
        boundary = self.segment_component_face_boundary
        # Validate the captured address mappings and D collisions before any write.
        bindings, direct_rows = shadow_source_bindings(case)
        seeded = []
        for payload in case['accepted']:
            row = tuple(payload['local_storage_row'])
            values = accepted_values(payload, case['origin'], case.get('constant_velocity_mps'))
            self.assertEqual(set(values), set(DATA['accepted_seed_fields']))
            for name, value in values.items():
                getattr(boundary, name)[row] = tuple(value) if isinstance(value, list) else value
            seeded.append({'storage': row, 'values': values})
        direct_before = {row: self._systemic_sample_input(row) for row in direct_rows}
        synchronized = []
        for binding in bindings:
            # The original S namespace must describe the same recorded accepted
            # sample as its storage shadow after the affine velocity substitution.
            shadow = self._systemic_sample_input(binding['storage'], shadow=True)
            source_row = binding['source']
            boundary.velocity_dirichlet_component_face_actual_sample_valid[source_row] = shadow['valid']
            boundary.velocity_dirichlet_component_face_actual_sample_point_m[source_row] = tuple(shadow['point_m'])
            boundary.velocity_dirichlet_component_face_actual_sample_velocity_mps[source_row] = tuple(shadow['velocity_mps'])
            actual = self._systemic_sample_input(source_row)
            guards = sample_payload_guards(actual, shadow)
            synchronized.append({**binding, 'actual_sample': actual, 'shadow_sample': shadow, 'guards': guards})
            self.assertTrue(all(guards.values()), ('original S actual sample differs from shadow input', binding, guards))
        direct_preserved = []
        for row, before in direct_before.items():
            after = self._systemic_sample_input(row)
            guards = sample_payload_guards(after, before)
            direct_preserved.append({'source': row, 'before': before, 'after': after, 'guards': guards})
            self.assertTrue(all(guards.values()), ('active D actual sample changed by S synchronization', row, guards))
        return seeded, {'expected_shadow_source_count': len(bindings), 'synchronized_shadow_sources': synchronized,
                        'active_direct_samples': direct_preserved, 'all_guards_passed': True}

    def _systemic_pair_observation(self, case):
        boundary = self.segment_component_face_boundary
        target = tuple(case['local_face'])
        pair = (*target, 2)
        record = {}
        for suffix in ('admission_valid', 'full_valid', 'first_author_linear_key', 'second_author_linear_key',
                       'first_author_kind', 'second_author_kind', 'direct_face_owner_shadow', 'endpoint_clamped'):
            record[suffix] = int(getattr(boundary, 'velocity_dirichlet_component_face_segment_pair_' + suffix)[pair])
        for suffix in ('boundary_point_m', 'normal', 'nominal_probe_m'):
            record[suffix] = [float(value) for value in getattr(boundary, 'velocity_dirichlet_component_face_segment_pair_' + suffix)[pair]]
        for suffix in ('boundary_target_mps', 'geometry_tolerance', 'clamp_support_ratio'):
            record[suffix] = float(getattr(boundary, 'velocity_dirichlet_component_face_segment_pair_' + suffix)[pair])
        routes, candidate_slots = [], []
        for source in case['sources']:
            storage = tuple(source['local_storage_row'])
            expected_offset = 1 - source['slot'] // 2
            if source['kind'] == 0:
                row = tuple(source['local_original_source_row'])
                selected = int(boundary.velocity_dirichlet_component_face_direct_selected_storage_offset[row][2])
                pair_offset = int(boundary.velocity_dirichlet_component_face_direct_relocation_pair_offset[row][2])
                routed = selected == expected_offset or pair_offset == expected_offset
            else:
                selected = int(boundary.velocity_dirichlet_relocation_shadow_selected_storage_offset[storage][2])
                pair_offset = None
                routed = selected == expected_offset
            routes.append({'slot': source['slot'], 'kind': source['kind'], 'selected': selected,
                           'direct_pair_offset': pair_offset, 'expected_offset': expected_offset})
            if routed:
                candidate_slots.append(source['slot'])
        return {'cache': record, 'routes': routes, 'route_candidate_slots': candidate_slots,
                'fallback_valid': int(boundary.velocity_dirichlet_component_face_common_trace_fallback_valid[pair]),
                'prior_adjacent_direct': int(boundary.velocity_dirichlet_component_face_common_trace_fallback_prior_adjacent_direct[pair]),
                'candidate_basis': 'Selected offset OR direct pair offset is route-candidate evidence, not final membership.'}

    def _systemic_common_observation(self, case):
        boundary = self.segment_component_face_boundary
        pair = (*case['local_face'], 2)
        record = {'mode': int(boundary.velocity_dirichlet_component_face_segment_projection_only_seam[pair]),
                  'owner_indices': [int(value) for value in boundary.velocity_dirichlet_component_face_segment_pair_owner_indices[pair]],
                  'actual_keys': [int(getattr(boundary, 'velocity_dirichlet_component_face_segment_' + name)[pair])
                                  for name in ('first_author_linear_key', 'second_author_linear_key')]}
        for suffix in ('seed_mask', 'proved_mask', 'consumed_mask'):
            record[suffix] = int(getattr(boundary, 'velocity_dirichlet_component_face_common_trace_' + suffix)[pair])
        return record

    def _systemic_common_neutral(self):
        records = {}
        for name, neutral in COMMON_NEUTRAL_FIELDS.items():
            values = getattr(self.segment_component_face_boundary, name).to_numpy()
            records[name] = {'shape': list(values.shape), 'neutral': neutral,
                             'all_neutral': bool(np.all(values == neutral)),
                             'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
        return {'fields': records, 'all_neutral': all(record['all_neutral'] for record in records.values())}

    def _systemic_apply_fault(self, case, fault):
        boundary, search, markers = (self.segment_component_face_boundary,
                                      self.segment_component_face_search, self.segment_component_face_markers)
        definition = FAULT_CASES[fault]
        self.assertEqual(case['original_face'][2], definition['z'])
        source = next((source for source in case['sources'] if source['slot'] == definition['slot']), None)
        row = tuple(source['local_original_source_row']) if source else tuple(case['local_face'])
        storage = tuple(source['local_storage_row']) if source else tuple(case['local_face'])
        record = {'name': fault, 'source_slot': definition['slot'], 'source': row, 'storage': storage,
                  'original_source': source['original_source_row'] if source else None,
                  'all_addresses_in_bounds': all(0 <= value < 4 for value in row + storage), 'writes': []}
        self.assertTrue(record['all_addresses_in_bounds'])

        def write(owner_name, name, address, value):
            owner = {'boundary': boundary, 'search': search, 'markers': markers}[owner_name]
            field = getattr(owner, name)
            before = np.asarray(field[address]).copy()
            field[address] = tuple(value) if isinstance(value, (list, tuple, np.ndarray)) else value
            after = np.asarray(field[address]).copy()
            record['writes'].append({'owner': owner_name, 'field': name, 'address': address,
                                     'before': before.tolist(), 'after': after.tolist(),
                                     'dtype': str(after.dtype), 'changed': before.tobytes() != after.tobytes()})
            return before, after

        if fault in ('bad_extra_source_anchor', 'unused_cached_direct_seed_anchor'):
            anchor = np.asarray(search.node_boundary_point_m[row], dtype=np.float32).copy()
            anchor[2] += np.float32(1e-3)
            before, after = write('search', 'node_boundary_point_m', row, anchor)
            record['physical_point_in_compact_grid'] = all(
                case['coordinates']['cell_face_' + axis + '_m'][0] <= after[index]
                <= case['coordinates']['cell_face_' + axis + '_m'][-1] for index, axis in enumerate('xyz'))
            record['weights_retained'] = np.asarray(search.node_projection_marker_weights[row], dtype=np.float32).tobytes() == np.asarray(source['weights'], dtype=np.float32).tobytes()
            record['intended_fault_verified'] = (bool(np.all(np.isfinite(after))) and before[2] != after[2]
                                                 and record['weights_retained'] and record['physical_point_in_compact_grid'])
        elif fault == 'bad_actual_sample':
            # Keep the two S namespaces equal but make the accepted Q lie on B:
            # zero wall distance is invalid for the recorded exterior sample.
            point = tuple(float(value) for value in search.node_boundary_point_m[row])
            write('boundary', 'velocity_dirichlet_component_face_actual_sample_point_m', row, point)
            write('boundary', 'velocity_dirichlet_relocation_shadow_sample_point_m', storage, point)
            actual, shadow = self._systemic_sample_input(row), self._systemic_sample_input(storage, shadow=True)
            record['sample_after'] = {'actual': actual, 'shadow': shadow, 'B': point}
            record['intended_fault_verified'] = (all(sample_payload_guards(actual, shadow).values())
                                                 and tuple(actual['point_m']) == point and all(math.isfinite(value) for value in point))
        elif fault == 'shadow_original_velocity_mismatch':
            value = np.asarray(boundary.velocity_dirichlet_component_face_actual_sample_velocity_mps[row], dtype=np.float32).copy()
            value[2] += np.float32(1e-3)
            write('boundary', 'velocity_dirichlet_component_face_actual_sample_velocity_mps', row, value)
            actual, shadow = self._systemic_sample_input(row), self._systemic_sample_input(storage, shadow=True)
            guards = sample_payload_guards(actual, shadow)
            record['sample_after'] = {'actual': actual, 'shadow': shadow, 'guards': guards}
            record['intended_fault_verified'] = guards['valid_exact'] and guards['point_m_exact_f32'] and not guards['velocity_mps_exact_f32']
        elif fault == 'wrong_shadow_base':
            before, after = write('boundary', 'velocity_dirichlet_relocation_shadow_storage_base_row', storage, (0, 1, 1))
            record['intended_fault_verified'] = (tuple(int(value) for value in before) == storage
                                                 and tuple(int(value) for value in after) != storage
                                                 and all(0 <= value < 4 for value in after))
        elif fault == 'reversed_registered_owner_edge':
            # set_projection_segments sorts endpoints. Mutate its real registered
            # input field, preserving its -1 sentinel and all endpoint bounds.
            edge_index = case['markers']['edges'].index([0, 1])
            before, after = write('markers', 'projection_triangle_indices', edge_index, (1, 0, -1))
            record['registered_edge_index'] = edge_index
            record['intended_fault_verified'] = (tuple(before) == (0, 1, -1) and tuple(after) == (1, 0, -1)
                                                 and 0 <= edge_index < markers.projection_segment_count
                                                 and all(0 <= value < len(case['markers']['positions']) for value in after[:2]))
        elif fault == 'r15_bad_actual_projection':
            weights = np.asarray(search.node_projection_marker_weights[row], dtype=np.float32)
            before, after = write('search', 'node_projection_marker_weights', row,
                                  (weights[1], weights[0], weights[2]))
            record['intended_fault_verified'] = (before[0] != after[0] and before[1] != after[1]
                and float(np.sum(before)) == float(np.sum(after)) == 1.0 and bool(np.all(after >= 0)))
        elif fault == 'r15_conflicting_candidate_probes':
            self.assertEqual(row, (0, 1, 2))
            protected = self._systemic_r15_geometry_input_identities()
            original = search.node_interior_fluid_point_m.to_numpy().copy()
            desired = original.copy()
            desired[row][1] = np.float32(desired[row][1] + np.float32(1e-5))
            before, after = write('search', 'node_interior_fluid_point_m', row, desired[row])
            self.assertEqual(search.node_interior_fluid_point_m.to_numpy().tobytes(), desired.tobytes())
            self.assertEqual(self._systemic_r15_geometry_input_identities(), protected)
            record['raw_y_delta_m'] = float(np.float32(after[1]) - np.float32(before[1]))
            record['one_raw_coordinate_changed'] = int(np.count_nonzero(original != desired)) == 1
            record['protected_field_count'] = len(protected)
            record['protected_identities_sha256'] = hashlib.sha256(
                json.dumps(protected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            record['candidate_geometry_probe'] = self._systemic_r15_candidate_geometry_probe(case, record['raw_y_delta_m'])
            self.assertEqual(self._systemic_r15_geometry_input_identities(), protected)
            self.assertEqual(search.node_interior_fluid_point_m.to_numpy().tobytes(), desired.tobytes())
            record['protected_fields_and_raw_Q_unchanged_by_probe'] = True
            record['intended_fault_verified'] = (record['one_raw_coordinate_changed']
                and record['raw_y_delta_m'] > 0 and record['candidate_geometry_probe']['all_guards_passed'])
        elif fault == 'r15_actual_shadows_only':
            # Prepare recomputes direct routes from geometry. Deactivate actual
            # D authority after proof; cached selector edits do not remove it.
            protected_names = set(DATA['accepted_seed_fields']) | set(COMMON_NEUTRAL_FIELDS)
            protected_names.update('velocity_dirichlet_component_face_segment_pair_' + suffix
                                   for suffix in self._systemic_pair_observation(case)['cache'])
            protected = {name: getattr(boundary, name).to_numpy().tobytes() for name in protected_names}
            original_active = boundary.active_ib_node.to_numpy().copy()
            expected_active = original_active.copy()
            direct_rows = []
            for direct in (item for item in case['sources'] if item['kind'] == 0):
                address = tuple(direct['local_original_source_row'])
                before, after = write('boundary', 'active_ib_node', address, 0)
                self.assertEqual(int(before), 1)
                self.assertEqual(int(after), 0)
                expected_active[address] = 0
                direct_rows.append(address)
            self.assertEqual(len(set(direct_rows)), 2)
            self.assertEqual(boundary.active_ib_node.to_numpy().tobytes(), expected_active.tobytes())
            self.assertEqual(int(np.count_nonzero(original_active != expected_active)), 2)
            shadows = []
            for shadow in (item for item in case['sources'] if item['kind'] == 1):
                source = tuple(shadow['local_original_source_row'])
                storage = tuple(shadow['local_storage_row'])
                self.assertNotIn(source, direct_rows)
                active = int(boundary.active_ib_node[source])
                valid = int(boundary.velocity_dirichlet_relocation_shadow_claim_valid[storage])
                self.assertEqual((active, valid), (1, 1))
                shadows.append({'source': source, 'storage': storage, 'active': active, 'claim_valid': valid})
            self.assertEqual(len(shadows), 2)
            self.assertEqual({name: getattr(boundary, name).to_numpy().tobytes() for name in protected}, protected)
            record['inactive_direct_rows'] = direct_rows
            record['active_shadow_authors'] = shadows
            record['protected_field_sha256'] = {name: hashlib.sha256(raw).hexdigest()
                                                for name, raw in sorted(protected.items())}
            record['cache_proof_and_shadow_inputs_byte_equal'] = True
            record['intended_fault_verified'] = True
            record['scope'] = 'Only two direct activity flags cleared after proof; actual S/S consumers must reject during prepare.'
        elif fault == 'contaminated_common_mode':
            before, after = write('boundary', 'velocity_dirichlet_component_face_segment_projection_only_seam',
                                  (*case['local_face'], 2), COMMON_MODE | 4)
            record['intended_fault_verified'] = int(before) == COMMON_MODE and int(after) == COMMON_MODE | 4
        else:
            raise AssertionError(('unregistered input fault', fault))
        record['intended_fault_verified'] = bool(record['intended_fault_verified'] and all(write['changed'] for write in record['writes']))
        self.assertTrue(record['intended_fault_verified'], record)
        return record

    def _systemic_run_fixture(self, original_z, *, fault=None, fixture_case=None):
        case = (next(case for case in DATA['cases'] if case['original_face'][2] == original_z)
                if fixture_case is None else fixture_case)
        self.assertEqual(case['original_face'][2], original_z)
        expected_count = (case['expected_prepare_count'] if 'expected_prepare_count' in case
                          else EXPECTED_PREPARE_COUNT[original_z])
        fluid, search, boundary = self.fluid, self.segment_component_face_search, self.segment_component_face_boundary
        original_coordinates = {name: getattr(fluid, name).to_numpy().copy() for name in COORDINATE_FIELDS}
        original_support = {name: getattr(search, name) for name in SUPPORT_ATTRIBUTES}
        original_kind = search.node_kind_code.to_numpy().copy()
        positive_guards = fault is None or fault == 'contaminated_common_mode'
        observed = {'original_face': case['original_face'], 'expected_result': 'SUCCESS' if fault is None else 'NATIVE_REJECTION',
                    'operator_focused_fixture': True, 'physical_steps_executed': 0,
                    'expected_route_candidate_slots': case['expected_original_routed_slots'],
                    'expected_prepare_count': expected_count, 'events': []}
        if fault is not None:
            observed['rejection_evidence_scope'] = ('target_common_trace_admission_rejected'
                if fault == 'bad_extra_source_anchor' else 'global_input_invalid_native_health_control')
        self._systemic_cohort_last_observed = observed
        try:
            self._systemic_load_fixture(case)
            target = tuple(case['local_face'])
            ledger_before = self._canonical_ledger_bytes()
            self.assertEqual(len(ledger_before), 8)
            observed['canonical_before'] = ledger_hashes(ledger_before)

            def observe(stage):
                if stage == 'hibm_velocity_row_segment_pair_precompute_before':
                    observed['events'].append(stage)
                    observed['seeded_accepted_inputs'], observed['shadow_source_inputs'] = self._systemic_seed_accepted(case)
                    if (fault is not None and fault != 'contaminated_common_mode'
                            and FAULT_CASES[fault].get('phase') != 'after_precompute'):
                        observed['fault'] = self._systemic_apply_fault(case, fault)
                        observed['fault']['stage'] = stage
                elif stage == 'hibm_velocity_row_segment_pair_precompute_after':
                    observed['events'].append(stage)
                    observed['precompute'] = self._systemic_pair_observation(case)
                    guards, reference = precompute_guards(case, observed['precompute'])
                    observed['precompute'].update(guards=guards, all_guards_passed=all(guards.values()),
                                                  affine_boundary_target_reference_mps=reference)
                    if positive_guards:
                        self.assertTrue(all(guards.values()), ('compact fixture precompute guards', guards))
                    if fault is not None and FAULT_CASES[fault].get('phase') == 'after_precompute':
                        observed['fault'] = self._systemic_apply_fault(case, fault)
                        observed['fault']['stage'] = stage
                elif stage == 'hibm_velocity_row_segment_reconstruct_before':
                    observed['events'].append(stage)
                    observed['reconstruction_samples'] = {
                        'scope': 'global counter delta for all component faces in the reconstruction kernel',
                        'target_specific_call_count': None, 'target_call_count_proven': False,
                        'before': int(boundary.report_velocity_dirichlet_component_face_actual_sample_evaluation_count[None])}
                elif stage in ('hibm_velocity_row_claim_prepare_after', 'hibm_velocity_row_segment_reconstruct_after'):
                    observed['events'].append(stage)
                    values = {name: float(getattr(boundary, 'velocity_dirichlet_component_face_claim_' + name)[target][2])
                              for name in ('target_mps', 'alpha')}
                    values['count'] = int(boundary.velocity_dirichlet_component_face_claim_count[target][2])
                    values['conflicts'] = int(boundary.report_velocity_dirichlet_component_face_target_conflict_count[None])
                    values['native_author_witnesses'] = [int(getattr(boundary, 'velocity_dirichlet_component_face_segment_' + name)[(*target, 2)])
                                                         for name in ('first_author_linear_key', 'second_author_linear_key')]
                    values['native_first_conflict'] = boundary._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
                    values['common_trace'] = self._systemic_common_observation(case)
                    observed['prepared' if stage.endswith('prepare_after') else 'reconstructed'] = values
                    if stage.endswith('prepare_after'):
                        values['exact_count_guard_passed'] = values['count'] == expected_count
                        values['common_guards'] = common_trace_guards(case, values['common_trace'])
                        values['cache_and_routes'] = self._systemic_pair_observation(case)
                        values['cache_guards'], _ = precompute_guards(case, values['cache_and_routes'])
                        if positive_guards:
                            self.assertEqual(values['count'], expected_count, 'exact prepared cohort size')
                            self.assertTrue(all(values['common_guards'].values()), values)
                            self.assertTrue(all(values['cache_guards'].values()), values)
                        if fault == 'contaminated_common_mode':
                            observed['fault'] = self._systemic_apply_fault(case, fault)
                            observed['fault']['stage'] = stage
                    else:
                        samples = observed['reconstruction_samples']
                        samples['after'] = int(boundary.report_velocity_dirichlet_component_face_actual_sample_evaluation_count[None])
                        samples['delta'] = samples['after'] - samples['before']

            try:
                report = self._assemble_component_face_ledger(
                    interpolate_interior_velocity=True, use_marker_geometry=True, use_segment_fixture=True,
                    provide_marker_topology=True, surface_projection_inactive_axis=0,
                    primary_region_id=101, secondary_region_id=202, stage_observer=observe,
                )['canonical_velocity_dirichlet_report']
            except BaseException as error:
                after = self._canonical_ledger_bytes()
                observed['canonical_after_error_before_reset'] = ledger_hashes(after)
                observed['all_eight_canonical_fields_unchanged_before_reset'] = after == ledger_before
                observed['native_error'] = native_health_error_record(error)
                if fault is None or not observed['native_error']['native_health_RuntimeError']:
                    raise
                self.assertTrue(observed.get('fault', {}).get('intended_fault_verified'), observed)
                self.assertTrue(observed['fault']['all_addresses_in_bounds'])
                self.assertIn('hibm_velocity_row_segment_pair_precompute_after', observed['events'])
                self.assertEqual(after, ledger_before, 'all eight canonical fields before fixture reset')
                if fault == 'bad_extra_source_anchor':
                    observed['target_rejection_evidence'] = bad_extra_target_rejection_guards(case, observed)
                    self.assertTrue(observed['target_rejection_evidence']['all_guards_passed'],
                                    observed['target_rejection_evidence'])
                if fault in ('r15_bad_actual_projection', 'r15_actual_shadows_only'):
                    prepared = observed['prepared']
                    common = prepared['common_trace']
                    self.assertEqual(observed['native_error']['origin']['function'],
                                     '_validate_canonical_velocity_dirichlet_target_conflict_precommit')
                    self.assertEqual(observed['precompute']['fallback_valid'], 1)
                    self.assertTrue(all(observed['precompute']['guards'].values()))
                    self.assertEqual(prepared['count'], 2)
                    self.assertEqual(common['mode'] & COMMON_MODE, 0)
                    self.assertEqual(common['consumed_mask'], 0)
                    if fault == 'r15_bad_actual_projection':
                        self.assertEqual(common['seed_mask'], 5)
                        self.assertNotEqual(common['proved_mask'] & 9, 9)
                        self.assertIn(prepared['native_author_witnesses'], ([-395, -2], [-252, -255]))
                    else:
                        self.assertGreater(prepared['conflicts'], 0)
                        self.assertEqual(common['seed_mask'], 5)
                        self.assertEqual(common['proved_mask'], 15)
                        self.assertTrue(observed['fault']['cache_proof_and_shadow_inputs_byte_equal'])
                        self.assertIn(prepared['native_author_witnesses'], ([-135, -2], [-256, -255]))
                    observed['target_fallback_rejection_verified'] = True
                if fault == 'r15_conflicting_candidate_probes':
                    prepared, precompute = observed['prepared'], observed['precompute']
                    common, cache = prepared['common_trace'], precompute['cache']
                    self.assertEqual(observed['native_error']['origin']['function'],
                                     '_validate_canonical_velocity_dirichlet_target_conflict_precommit')
                    self.assertTrue(observed['fault']['candidate_geometry_probe']['all_guards_passed'])
                    self.assertEqual(precompute['fallback_valid'], 0)
                    self.assertEqual([cache[name] for name in ('first_author_linear_key', 'second_author_linear_key',
                                                                'first_author_kind', 'second_author_kind')], [5, 1, 0, 1])
                    self.assertEqual((cache['admission_valid'], cache['full_valid'], cache['endpoint_clamped']), (0, 0, 1))
                    self.assertEqual(prepared['cache_and_routes']['cache'], cache)
                    self.assertEqual(prepared['cache_and_routes']['fallback_valid'], 0)
                    self.assertEqual(prepared['count'], 2)
                    self.assertEqual(common['mode'] & COMMON_MODE, 0)
                    self.assertEqual(common['consumed_mask'], 0)
                    self.assertIn(prepared['native_author_witnesses'], ([-395, -2], [-252, -255]))
                    observed['target_fallback_rejection_verified'] = True
                observed['common_fields_after_native_rejection'] = self._systemic_common_neutral()
                self.assertTrue(observed['common_fields_after_native_rejection']['all_neutral'])
                self._assert_component_face_relocation_transient_neutral(use_segment_fixture=True)
                if fault in ('r15_conflicting_candidate_probes', 'r15_actual_shadows_only'):
                    adjacent = boundary.velocity_dirichlet_component_face_adjacent_direct_pair_target_valid.to_numpy()
                    self.assertTrue(bool(np.all(adjacent == 0)))
                    observed['native_cleanup_field_count'] = 41
                    observed['adjacent_pair_field_after_native_rejection'] = {
                        'all_neutral': True, 'sha256': hashlib.sha256(adjacent.tobytes()).hexdigest()}
                observed['result'] = 'PASS_EXPECTED_NATIVE_REJECTION'
                return observed
            if fault is not None:
                self.fail(('native assembly accepted intentionally invalid input', fault))
            state = self._canonical_component_state(target, 2)
            physical_face = [case['coordinates']['cell_center_x_m'][target[0]],
                             case['coordinates']['cell_center_y_m'][target[1]],
                             case['coordinates']['cell_face_z_m'][target[2]]]
            expected = affine_value(physical_face, case.get('constant_velocity_mps'))
            observed.update(final_state=state, physical_face=physical_face, independent_affine_expected_mps=expected)
            self.assertTrue(state['active'] and state['owned'])
            self.assertEqual(int(report['target_conflict_count']), 0)
            self.assertEqual(observed['prepared']['count'], expected_count)
            self.assertTrue(math.isfinite(observed['reconstructed']['alpha']))
            self.assertGreater(observed['reconstructed']['alpha'], 1e-6)
            self.assertLessEqual(observed['reconstructed']['alpha'], 1.0)
            self.assertTrue(all(common_trace_guards(case, observed['reconstructed']['common_trace']).values()))
            self.assertAlmostEqual(float(state['value_mps']), expected, delta=AFFINE_TOLERANCE_MPS)
            self.assertEqual(float(state['value_mps']), observed['reconstructed']['target_mps'])
            self._assert_component_face_relocation_transient_neutral(use_segment_fixture=True)
            observed['common_fields_after_commit'] = self._systemic_common_neutral()
            self.assertTrue(observed['common_fields_after_commit']['all_neutral'])
            observed['result'] = 'PASS'
            return observed
        except BaseException as error:
            observed['result'] = 'FAIL_EXPECTED_SUCCESS' if fault is None else 'FAIL_EXPECTED_NATIVE_REJECTION'
            observed['error'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            try:
                # The reversed registry is raw input; restore it without touching
                # any native selector, certificate, health counter or reset path.
                for write in observed.get('fault', {}).get('writes', []):
                    if write['owner'] == 'markers':
                        getattr(self.segment_component_face_markers, write['field'])[write['address']] = tuple(write['before'])
                self._reset_component_face_claim_fixture(use_segment_fixture=True)
                observed['common_fields_after_reset'] = self._systemic_common_neutral()
                self.assertTrue(observed['common_fields_after_reset']['all_neutral'])
            finally:
                for name, values in original_coordinates.items():
                    getattr(fluid, name).from_numpy(values)
                for name, value in original_support.items():
                    setattr(search, name, value)
                search.node_kind_code.from_numpy(original_kind)
                observed['all_nine_coordinates_restored'] = all(
                    getattr(fluid, name).to_numpy().tobytes() == value.tobytes() for name, value in original_coordinates.items())
                print('SYSTEMIC_COHORT_OBSERVATION ' + json.dumps(observed), flush=True)
                self.assertTrue(observed['all_nine_coordinates_restored'])

    def test_systemic_r14_tilted_ds_common_cohort_reconstructs_affine_velocity(self):
        self.assertEqual(hashlib.sha256(R14_DATA_PATH.read_bytes()).hexdigest(), R14_DATA_SHA)
        data = json.loads(R14_DATA_PATH.read_text())
        for name in ('grid_nodes', 'original_grid_nodes', 'marker_capacity', 'coordinate_fields',
                     'accepted_seed_fields', 'support_radius_xyz_m'):
            self.assertEqual(data[name], DATA[name], name)
        self.assertEqual(len(data['cases']), 1)
        case = data['cases'][0]
        self.assertEqual(case['capture_id'], 'r14_tilted_z305')
        self._systemic_run_fixture(305, fixture_case=case)

    def test_systemic_z305_common_cohort_reconstructs_affine_velocity(self):
        self._systemic_run_fixture(305)

    def test_systemic_z306_common_cohort_reconstructs_affine_velocity(self):
        self._systemic_run_fixture(306)

    def test_systemic_z347_common_cohort_reconstructs_affine_velocity(self):
        self._systemic_run_fixture(347)

    def test_systemic_common_trace_bad_extra_source_anchor_fails_atomically(self):
        self._systemic_run_fixture(305, fault='bad_extra_source_anchor')

    def test_systemic_common_trace_bad_actual_sample_fails_atomically(self):
        self._systemic_run_fixture(347, fault='bad_actual_sample')

    def test_systemic_common_trace_shadow_original_velocity_mismatch_fails_atomically(self):
        self._systemic_run_fixture(347, fault='shadow_original_velocity_mismatch')

    def test_systemic_common_trace_wrong_shadow_base_fails_atomically(self):
        self._systemic_run_fixture(305, fault='wrong_shadow_base')

    def test_systemic_common_trace_unused_cached_direct_seed_anchor_fails_atomically(self):
        self._systemic_run_fixture(347, fault='unused_cached_direct_seed_anchor')

    def test_systemic_common_trace_reversed_registered_owner_edge_fails_atomically(self):
        self._systemic_run_fixture(305, fault='reversed_registered_owner_edge')

    def test_systemic_common_trace_contaminated_common_mode_fails_atomically(self):
        self._systemic_run_fixture(305, fault='contaminated_common_mode')

    def _systemic_r15_case(self, *, mirror=False, constant=False, omit_unused_direct=False):
        self.assertEqual(hashlib.sha256(R15_DATA_PATH.read_bytes()).hexdigest(), R15_DATA_SHA)
        data = json.loads(R15_DATA_PATH.read_text())
        for name in ('grid_nodes', 'original_grid_nodes', 'marker_capacity', 'coordinate_fields',
                     'accepted_seed_fields', 'support_radius_xyz_m'):
            self.assertEqual(data[name], DATA[name], name)
        case = copy.deepcopy(data['cases'][0])
        self.assertEqual(case['capture_id'], 'r15_invalid_first_pair_z342')
        if mirror:
            reflection = 2.0 * case['coordinates']['cell_face_z_m'][2]

            def reflect_point(point):
                return [point[0], point[1], float(np.float32(reflection - point[2]))]

            def reflect_row(row):
                if row == [-1, -1, -1]:
                    return row
                return [row[0], row[1], 2 * case['origin'][2] + 3 - row[2]]

            def reflect_local(row):
                return [row[0], row[1], 3 - row[2]]

            for kind in ('face', 'center'):
                name = 'cell_' + kind + '_z_m'
                case['coordinates'][name] = [float(np.float32(reflection - value))
                                              for value in reversed(case['coordinates'][name])]
            case['coordinates']['cell_width_z_m'].reverse()
            case['obstacle'] = np.asarray(case['obstacle'])[:, :, ::-1].tolist()
            case['markers']['positions'] = [reflect_point(point) for point in case['markers']['positions']]
            for normal in case['markers']['normals']:
                normal[2] = -normal[2]
            for source in case['sources']:
                source['slot'] ^= 2
                source['original_source_row'] = reflect_row(source['original_source_row'])
                for name in ('local_original_source_row', 'local_storage_row', 'local_geometry_base_row'):
                    source[name] = reflect_local(source[name])
                for name in ('B', 'nominal_Q'):
                    source[name] = reflect_point(source[name])
                source['normal'][2] = -source['normal'][2]
            case['sources'].sort(key=lambda source: source['slot'])
            for payload in case['accepted']:
                payload['original_storage_row'] = reflect_row(payload['original_storage_row'])
                payload['local_storage_row'] = reflect_local(payload['local_storage_row'])
                fields = payload['fields']
                for name in ('velocity_dirichlet_component_face_actual_sample_point_m',
                             'velocity_dirichlet_relocation_shadow_sample_point_m'):
                    fields[name] = reflect_point(fields[name])
                for name in ('velocity_dirichlet_relocation_shadow_source_row',
                             'velocity_dirichlet_relocation_shadow_storage_base_row'):
                    fields[name] = reflect_row(fields[name])
                key = fields['velocity_dirichlet_relocation_winner_source_linear_key']
                original = [key // (96 * 400), (key // 400) % 96, key % 400]
                i, j, k = reflect_row(original)
                fields['velocity_dirichlet_relocation_winner_source_linear_key'] = (i * 96 + j) * 400 + k
            for name in ('boundary_point_m', 'nominal_probe_m'):
                case['expected_fallback_pair'][name] = reflect_point(case['expected_fallback_pair'][name])
            # A reflected exactly tangential ray still chooses offset 0 at a tie.
            # One f32 ULP makes this manufactured S0 ray prefer offset 1.
            shadow = next(source for source in case['sources'] if source['slot'] == 1)
            payload = next(item for item in case['accepted']
                           if item['local_storage_row'] == shadow['local_storage_row'])
            point = payload['fields']['velocity_dirichlet_relocation_shadow_sample_point_m']
            point[2] = float(np.nextafter(np.float32(point[2]), np.float32(-np.inf)))
            case['expected_original_routed_slots'] = [1, 2, 3]
            case['expected_cache_identity'] = [5, 1, 0, 1]
            case['expected_common'] = {'seed_mask': 3, 'consumed_mask': 6,
                                       'actual_keys': [1, 6], 'owner_indices': [1, 2, -1]}
            case['expected_consumed_slots'], case['expected_consumed_mask'] = [1, 2], 6
            case['derived_mirror'] = True
        if omit_unused_direct:
            self.assertFalse(mirror)
            direct = next(source for source in case['sources'] if source['slot'] == 2)
            direct['active'] = direct['node_kind'] = 0
            case['expected_cache_identity'] = [5, 2, 0, 1]
            case['expected_common']['seed_mask'] = 9
            case['omitted_witness_slot'] = 2
        if constant:
            case['constant_velocity_mps'] = 0.375
            # Keep both accepted S namespaces consistent. Match the ordinary
            # face-progress ratios using physical B/Q; no certificate is edited.
            first = next(source for source in case['sources'] if source['slot'] == 0)
            direct_payload = next(payload for payload in case['accepted']
                                  if payload['local_storage_row'] == first['local_storage_row'])
            point = direct_payload['fields']['velocity_dirichlet_component_face_actual_sample_point_m']
            face_y = case['coordinates']['cell_center_y_m'][case['local_face'][1]]
            progress = (face_y - first['B'][1]) / (point[1] - first['B'][1])
            for payload in case['accepted']:
                shadow = next(source for source in case['sources'] if source['kind'] == 1
                              and source['local_storage_row'] == payload['local_storage_row'])
                q = payload['fields']['velocity_dirichlet_relocation_shadow_sample_point_m']
                q[1] = float(np.float32(shadow['B'][1] + (face_y - shadow['B'][1]) / progress))
            case['constant_sample_progress_reference'] = progress
        return case

    def test_systemic_r15_invalid_first_pair_reconstructs_affine_velocity(self):
        self._systemic_run_fixture(342, fixture_case=self._systemic_r15_case())

    def test_systemic_r15_mirrored_invalid_pair_keeps_shadow_first_membership(self):
        self._systemic_run_fixture(342, fixture_case=self._systemic_r15_case(mirror=True))

    def test_systemic_r15_constant_field_requires_common_fallback(self):
        self._systemic_run_fixture(342, fixture_case=self._systemic_r15_case(constant=True))

    def test_systemic_r15_two_consumers_equal_seed_reconstruct(self):
        self._systemic_run_fixture(342, fixture_case=self._systemic_r15_case(omit_unused_direct=True))

    def test_systemic_r15_constant_two_consumers_equal_seed_reconstruct(self):
        self._systemic_run_fixture(342, fixture_case=self._systemic_r15_case(constant=True, omit_unused_direct=True))

    def test_systemic_r15_equal_targets_bad_consumed_projection_fails_atomically(self):
        self._systemic_run_fixture(342, fault='r15_bad_actual_projection',
                                   fixture_case=self._systemic_r15_case(constant=True))

    def test_systemic_r15_proved_direct_witnesses_do_not_authorize_shadow_only_consumers(self):
        self._systemic_run_fixture(342, fault='r15_actual_shadows_only',
                                   fixture_case=self._systemic_r15_case(constant=True))

    def _systemic_r15_consumer_case(self, consumer_count):
        self.assertIn(consumer_count, (0, 1))
        case = self._systemic_r15_case()
        # Only velocities become constant; retain all captured sample points.
        case['constant_velocity_mps'] = 0.375
        inactive_slots = (0, 1) if consumer_count == 1 else (0, 1, 3)
        for source in case['sources']:
            if source['slot'] in inactive_slots:
                source['active'] = source['node_kind'] = 0
        for payload in case['accepted']:
            fields = payload['fields']
            if payload['local_storage_row'] == [0, 1, 1]:
                fields['velocity_dirichlet_component_face_actual_sample_valid'] = 0
                fields['velocity_dirichlet_relocation_shadow_claim_valid'] = 0
            elif consumer_count == 0:
                fields['velocity_dirichlet_relocation_shadow_claim_valid'] = 0
        case['consumer_count'] = consumer_count
        case['expected_prepare_count'] = consumer_count
        case['expected_materialized_routed_slots'] = [3] if consumer_count else []
        case['consumer_control_scope'] = (
            'D1/S1 remain materialized; D1 routes away and S1 alone reaches the target.'
            if consumer_count else
            'Only D1 remains materialized and routes away; no fallback candidate is required.')
        return case

    def _systemic_run_consumer_boundary(self, consumer_count):
        case = self._systemic_r15_consumer_case(consumer_count)
        fluid, search, boundary = self.fluid, self.segment_component_face_search, self.segment_component_face_boundary
        original_coordinates = {name: getattr(fluid, name).to_numpy().copy() for name in COORDINATE_FIELDS}
        original_support = {name: getattr(search, name) for name in SUPPORT_ATTRIBUTES}
        original_kind = search.node_kind_code.to_numpy().copy()
        observed = {
            'schema': 'R15_CONSUMER_BOUNDARY_V1', 'consumer_count': consumer_count,
            'physical_case_sha256': hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'original_face': case['original_face'], 'local_face': case['local_face'], 'component_axis': 2,
            'expected_result': 'SUCCESS', 'operator_focused_fixture': True, 'physical_steps_executed': 0,
            'constant_velocity_mps': 0.375, 'control_scope': case['consumer_control_scope'], 'events': [],
        }
        self._systemic_cohort_last_observed = observed
        try:
            self._systemic_load_fixture(case)
            target, pair = tuple(case['local_face']), (*case['local_face'], 2)
            initial_target = self._canonical_component_state(target, 2)
            before = self._canonical_ledger_bytes()
            self.assertEqual(len(before), 8)
            observed['canonical_before'] = ledger_hashes(before)

            def observe(stage):
                if stage == 'hibm_velocity_row_segment_pair_precompute_before':
                    observed['events'].append(stage)
                    observed['seeded_accepted_inputs'], observed['shadow_source_inputs'] = self._systemic_seed_accepted(case)
                elif stage == 'hibm_velocity_row_segment_pair_precompute_after':
                    observed['events'].append(stage)
                    observed['precompute'] = self._systemic_pair_observation(case)
                    routes = {item['slot']: item for item in observed['precompute']['routes']}
                    materialized, routed = [], []
                    for source in case['sources']:
                        row, storage = tuple(source['local_original_source_row']), tuple(source['local_storage_row'])
                        self.assertEqual(int(boundary.active_ib_node[row]), source['active'])
                        self.assertEqual(int(search.node_kind_code[row]), source['node_kind'])
                        if not source['active']:
                            continue
                        sample = self._systemic_sample_input(storage if source['kind'] else row,
                                                            shadow=bool(source['kind']))
                        self.assertEqual(sample['valid'], 1)
                        author = {'slot': source['slot'], 'kind': source['kind'], 'source': list(row),
                                  'storage': list(storage), 'source_linear_key': row[0] * 16 + row[1] * 4 + row[2],
                                  'sample': sample, 'route': routes[source['slot']]}
                        materialized.append(author)
                        if source['slot'] in observed['precompute']['route_candidate_slots']:
                            routed.append(author)
                    self.assertEqual([item['slot'] for item in materialized], [2, 3] if consumer_count else [2])
                    self.assertEqual([item['slot'] for item in routed], case['expected_materialized_routed_slots'])
                    self.assertEqual(routes[2]['selected'], 1)
                    self.assertEqual(routes[2]['direct_pair_offset'], -1)
                    observed['materialized_authors'], observed['materialized_routed_authors'] = materialized, routed
                elif stage in ('hibm_velocity_row_claim_prepare_after', 'hibm_velocity_row_segment_reconstruct_after'):
                    observed['events'].append(stage)
                    record = {
                        'count': int(boundary.velocity_dirichlet_component_face_claim_count[target][2]),
                        'target_mps': float(boundary.velocity_dirichlet_component_face_claim_target_mps[target][2]),
                        'alpha': float(boundary.velocity_dirichlet_component_face_claim_alpha[target][2]),
                        'region_id': int(boundary.velocity_dirichlet_component_face_claim_region_id[target][2]),
                        'native_author_witnesses': [int(getattr(boundary, 'velocity_dirichlet_component_face_segment_' + suffix)[pair])
                                                    for suffix in ('first_author_linear_key', 'second_author_linear_key')],
                        'segment_mode': int(boundary.velocity_dirichlet_component_face_segment_projection_only_seam[pair]),
                    }
                    self.assertEqual(record['count'], consumer_count)
                    self.assertEqual(record['native_author_witnesses'], [-1, -1])
                    self.assertEqual(record['segment_mode'], 0)
                    self.assertEqual(record['region_id'], 101 if consumer_count else -1)
                    self.assertTrue(math.isfinite(record['target_mps']) and math.isfinite(record['alpha']))
                    if consumer_count:
                        self.assertAlmostEqual(record['target_mps'], 0.375, delta=AFFINE_TOLERANCE_MPS)
                        self.assertGreater(record['alpha'], 1e-6)
                        self.assertLessEqual(record['alpha'], 1.0)
                    else:
                        self.assertEqual((record['target_mps'], record['alpha']), (0.0, 0.0))
                    observed['prepared' if stage.endswith('prepare_after') else 'reconstructed'] = record

            report = self._assemble_component_face_ledger(
                interpolate_interior_velocity=True, use_marker_geometry=True, use_segment_fixture=True,
                provide_marker_topology=True, surface_projection_inactive_axis=0,
                primary_region_id=101, secondary_region_id=202, stage_observer=observe,
            )['canonical_velocity_dirichlet_report']
            self.assertEqual(int(report['target_conflict_count']), 0)
            self.assertEqual(observed['prepared'], observed['reconstructed'])
            state = self._canonical_component_state(target, 2)
            if consumer_count:
                self.assertTrue(state['active'] and state['owned'])
                self.assertEqual(state['region_id'], 101)
                self.assertEqual(state['value_mps'], observed['reconstructed']['target_mps'])
            else:
                self.assertEqual(state, initial_target)
                self.assertFalse(state['active'] or state['owned'])
            after = self._canonical_ledger_bytes()
            self.assertEqual(len(after), 8)
            observed['final_state'] = state
            observed['canonical_after_commit'] = ledger_hashes(after)
            observed['canonical_after_commit_bytes_hex'] = {name: payload.hex() for name, payload in after}
            self._assert_component_face_relocation_transient_neutral(use_segment_fixture=True)
            observed['common_fields_after_commit'] = self._systemic_common_neutral()
            self.assertTrue(observed['common_fields_after_commit']['all_neutral'])
            observed['result'] = 'PASS'
            return observed
        except BaseException as error:
            observed['result'] = 'FAIL_EXPECTED_SUCCESS'
            observed['error'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            try:
                self._reset_component_face_claim_fixture(use_segment_fixture=True)
                observed['common_fields_after_reset'] = self._systemic_common_neutral()
                self.assertTrue(observed['common_fields_after_reset']['all_neutral'])
            finally:
                for name, values in original_coordinates.items():
                    getattr(fluid, name).from_numpy(values)
                for name, value in original_support.items():
                    setattr(search, name, value)
                search.node_kind_code.from_numpy(original_kind)
                observed['all_nine_coordinates_restored'] = all(
                    getattr(fluid, name).to_numpy().tobytes() == value.tobytes() for name, value in original_coordinates.items())
                self.assertTrue(observed['all_nine_coordinates_restored'])
                print('SYSTEMIC_COHORT_OBSERVATION ' + json.dumps(observed), flush=True)

    def test_systemic_r15_single_consumer_preserves_original_assembly(self):
        observed = self._systemic_run_consumer_boundary(1)
        self.assertEqual(observed['precompute']['fallback_valid'], 1)

    def test_systemic_r15_zero_consumers_preserve_original_assembly(self):
        observed = self._systemic_run_consumer_boundary(0)
        self.assertEqual(observed['precompute']['fallback_valid'], 0)

    def _systemic_r15_geometry_input_identities(self):
        boundary, search, markers = (self.segment_component_face_boundary,
                                      self.segment_component_face_search, self.segment_component_face_markers)
        fields = {
            'boundary.' + name: value for name, value in vars(boundary).items()
            if name.startswith(('velocity_dirichlet_component_face_', 'velocity_dirichlet_relocation_'))
            and hasattr(value, 'to_numpy')}
        for name in ('active_ib_node', 'pressure_neumann_normal_field', 'velocity_dirichlet_mps_field'):
            fields['boundary.' + name] = getattr(boundary, name)
        for name in ('node_kind_code', 'node_boundary_point_m', 'node_projection_marker_indices',
                     'node_projection_marker_weights', 'nearest_marker'):
            fields['search.' + name] = getattr(search, name)
        for name in ('x_gamma_m', 'v_gamma_mps', 'region_id', 'projection_triangle_indices'):
            fields['markers.' + name] = getattr(markers, name)
        for name in (*COORDINATE_FIELDS, 'velocity', 'obstacle', *self._CANONICAL_LEDGER_FIELDS):
            fields['fluid.' + name] = getattr(self.fluid, name)
        return {name: {'shape': list(values.shape), 'dtype': str(values.dtype),
                       'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
                for name, field in fields.items() for values in (field.to_numpy(),)}

    def _systemic_r15_candidate_geometry_probe(self, case, raw_y_delta_m):
        """Query only D0/S1 and D1/S1, writing dedicated test output fields."""
        import taichi as ti

        boundary, search, markers, fluid = (self.segment_component_face_boundary, self.segment_component_face_search,
                                             self.segment_component_face_markers, self.fluid)
        self.assertEqual(case['local_face'], [0, 1, 2])
        self.assertEqual(int(markers.projection_vertex_count), 4)
        self.assertEqual(int(markers.projection_segment_count), 3)
        self.assertEqual(search._last_search_inactive_axis, 0)
        self.assertTrue(search._last_search_support_anisotropic)
        support_values = tuple(search._last_search_support_radius_xyz_m)
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in support_values))
        flags = ti.Vector.field(14, dtype=ti.i64, shape=2)
        geometry = ti.Vector.field(12, dtype=ti.f64, shape=2)

        @ti.kernel
        def probe(boundary: ti.template(), search: ti.template(), markers: ti.template(), fluid: ti.template(),
                  flags: ti.template(), geometry: ti.template(), support_x: ti.f32, support_y: ti.f32, support_z: ti.f32):
            for query in range(2):
                target = ti.Vector([0, 1, 2])
                first_slot, second_slot = 2 * query, 3
                support = ti.Vector([support_x, support_y, support_z])
                first_valid, first, first_storage, _first_kind, first_point, _first_velocity = (
                    boundary._canonical_component_face_materialized_source_identity(
                        target, 2, first_slot, 0, fluid.obstacle, search.node_boundary_point_m, search.node_interior_fluid_point_m,
                        fluid.cell_face_x_m, fluid.cell_face_y_m, fluid.cell_face_z_m,
                        fluid.cell_center_x_m, fluid.cell_center_y_m, fluid.cell_center_z_m, 4, 4, 4))
                second_valid, second, second_storage, _second_kind, second_point, _second_velocity = (
                    boundary._canonical_component_face_materialized_source_identity(
                        target, 2, second_slot, 0, fluid.obstacle, search.node_boundary_point_m, search.node_interior_fluid_point_m,
                        fluid.cell_face_x_m, fluid.cell_face_y_m, fluid.cell_face_z_m,
                        fluid.cell_center_x_m, fluid.cell_center_y_m, fluid.cell_center_z_m, 4, 4, 4))
                first_complete, second_complete, region = 0, 0, -1
                admission, full, owner_exception, endpoint, unique = 0, 0, 0, 0, 0
                owner = ti.Vector([-1, -1, -1])
                B, Q, normal = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
                boundary_target, tolerance, clamp_ratio = 0.0, 0.0, 0.0
                first_key, second_key = -1, -1
                if first_valid and second_valid:
                    first_key = first.x * 16 + first.y * 4 + first.z
                    second_key = second.x * 16 + second.y * 4 + second.z
                    nearest = search.nearest_marker[first]
                    if nearest >= 0 and nearest < 4:
                        region = markers.region_id[nearest]
                    first_complete = boundary._canonical_component_face_complete_segment_author(
                        first, 2, region, search.node_boundary_point_m, search.node_projection_marker_indices,
                        search.node_projection_marker_weights, search.nearest_marker, markers.x_gamma_m,
                        markers.v_gamma_mps, markers.region_id, markers.projection_triangle_indices, 3, 1, 4, 0, 1, 1,
                        support, fluid.cell_center_x_m, fluid.cell_center_y_m, fluid.cell_center_z_m)
                    second_complete = boundary._canonical_component_face_complete_segment_author(
                        second, 2, region, search.node_boundary_point_m, search.node_projection_marker_indices,
                        search.node_projection_marker_weights, search.nearest_marker, markers.x_gamma_m,
                        markers.v_gamma_mps, markers.region_id, markers.projection_triangle_indices, 3, 1, 4, 0, 1, 1,
                        support, fluid.cell_center_x_m, fluid.cell_center_y_m, fluid.cell_center_z_m)
                    first_center = ti.Vector([fluid.cell_center_x_m[first.x], fluid.cell_center_y_m[first.y], fluid.cell_center_z_m[first.z]])
                    second_center = ti.Vector([fluid.cell_center_x_m[second.x], fluid.cell_center_y_m[second.y], fluid.cell_center_z_m[second.z]])
                    face_center = ti.Vector([fluid.cell_center_x_m[0], fluid.cell_center_y_m[1], fluid.cell_face_z_m[2]])
                    if query == 0:
                        first_support = ti.Vector([fluid.cell_center_x_m[first_storage.x], fluid.cell_center_y_m[first_storage.y],
                                                   fluid.cell_center_z_m[first_storage.z]])
                        second_support = ti.Vector([fluid.cell_center_x_m[second_storage.x], fluid.cell_center_y_m[second_storage.y],
                                                    fluid.cell_center_z_m[second_storage.z]])
                        admission, full, B, normal, Q, boundary_target, endpoint, clamp_ratio, tolerance, owner_exception, owner = (
                            boundary._canonical_component_face_finite_segment_union_owner_geometry(
                                target, 2, face_center, first_center, second_center, first_support, second_support,
                                search.node_boundary_point_m[first], search.node_boundary_point_m[second],
                                search.node_interior_fluid_point_m[first], search.node_interior_fluid_point_m[second],
                                first_point, second_point, boundary.pressure_neumann_normal_field[first], boundary.pressure_neumann_normal_field[second],
                                search.node_projection_marker_indices[first], search.node_projection_marker_indices[second],
                                search.node_projection_marker_weights[first], search.node_projection_marker_weights[second],
                                search.nearest_marker[first], search.nearest_marker[second], region, 1, 1, support,
                                markers.projection_triangle_indices, 3, 1, 0, 0, 0, -1, 0, 0,
                                markers.x_gamma_m, markers.v_gamma_mps, markers.region_id,
                                fluid.cell_face_x_m, fluid.cell_face_y_m, fluid.cell_face_z_m))
                    else:
                        admission, full, B, normal, Q, boundary_target, tolerance, endpoint, clamp_ratio = (
                            boundary._canonical_component_face_same_storage_direct_relocation_geometry(
                                target, 2, face_center, first_center, second_center,
                                search.node_boundary_point_m[first], search.node_interior_fluid_point_m[first], first_point,
                                search.node_boundary_point_m[second], second_point,
                                boundary.pressure_neumann_normal_field[first], boundary.pressure_neumann_normal_field[second],
                                search.node_projection_marker_indices[first], search.node_projection_marker_indices[second],
                                search.node_projection_marker_weights[first], search.node_projection_marker_weights[second],
                                search.nearest_marker[first], search.nearest_marker[second],
                                boundary.velocity_dirichlet_mps_field[first][2], boundary.velocity_dirichlet_mps_field[second][2],
                                region, 1, 1, support, markers.projection_triangle_indices, 3, 1,
                                markers.x_gamma_m, markers.v_gamma_mps, markers.region_id, 4, 0,
                                fluid.cell_face_x_m, fluid.cell_face_y_m, fluid.cell_face_z_m))
                        if full:
                            owner = search.node_projection_marker_indices[first]
                    if full and owner.x >= 0 and owner.y > owner.x and owner.y < 4 and owner.z == -1:
                        unique = boundary._canonical_component_face_registered_unique_segment(owner, markers.projection_triangle_indices, 3, 1)
                flags[query] = ti.Vector([first_valid, second_valid, first_complete, second_complete, admission, full,
                                          owner_exception, unique, owner.x, owner.y, owner.z, first_key, second_key, endpoint])
                geometry[query] = ti.Vector([B.x, B.y, B.z, Q.x, Q.y, Q.z, normal.x, normal.y, normal.z,
                                             boundary_target, tolerance, clamp_ratio])

        probe(boundary, search, markers, fluid, flags, geometry, *support_values)
        flag_values, geometry_values = flags.to_numpy(), geometry.to_numpy()
        records = []
        for index, slots in enumerate(([0, 3], [2, 3])):
            integers, values = flag_values[index].tolist(), geometry_values[index].tolist()
            self.assertEqual(integers[:6], [1, 1, 1, 1, 1, 1])
            self.assertEqual(integers[6:11], [0, 1, 1, 2, -1])
            self.assertEqual(integers[11:13], [5 if index == 0 else 6, 2])
            self.assertTrue(all(math.isfinite(value) for value in values))
            self.assertGreater(values[10], 0.0)
            records.append({'slots': slots, 'materialized_valid': integers[:2], 'complete_author_valid': integers[2:4],
                            'admission_valid': integers[4], 'full_valid': integers[5], 'owner_exception': integers[6],
                            'registered_unique_owner': integers[7], 'owner_indices': integers[8:11],
                            'original_source_keys': integers[11:13], 'endpoint_clamped': integers[13],
                            'boundary_point_m': values[:3], 'nominal_probe_m': values[3:6], 'normal': values[6:9],
                            'boundary_target_mps': values[9], 'geometry_tolerance': values[10], 'clamp_support_ratio': values[11]})
        tolerance = max(item['geometry_tolerance'] for item in records)
        boundary_gap = float(np.linalg.norm(geometry_values[0, :3] - geometry_values[1, :3]))
        probe_gap = float(np.linalg.norm(geometry_values[0, 3:6] - geometry_values[1, 3:6]))
        expected_Q = np.asarray(case['expected_fallback_pair']['nominal_probe_m'], dtype=np.float64)
        shifted_Q = expected_Q.copy()
        shifted_Q[1] += raw_y_delta_m
        guards = {'same_B_within_tolerance': boundary_gap <= tolerance, 'different_Q_beyond_tolerance': probe_gap > tolerance,
                  'untouched_union_Q_matches_original': float(np.linalg.norm(geometry_values[0, 3:6] - expected_Q)) <= tolerance,
                  'same_storage_Q_matches_raw_shift': float(np.linalg.norm(geometry_values[1, 3:6] - shifted_Q)) <= tolerance}
        self.assertTrue(all(guards.values()), guards)
        return {'queries': records, 'native_geometry_query_count': 2, 'boundary_gap_m': boundary_gap, 'probe_gap_m': probe_gap,
                'max_geometry_tolerance_m': tolerance, 'guards': guards, 'all_guards_passed': True,
                'writes_only_dedicated_test_fields': True, 'cached_trace_target_sample_evaluated': False}

    def test_systemic_r15_conflicting_valid_candidate_probes_fail_atomically(self):
        self._systemic_run_fixture(342, fault='r15_conflicting_candidate_probes', fixture_case=self._systemic_r15_case())
