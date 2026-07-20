from __future__ import annotations

import unittest
from types import SimpleNamespace

from simulation_core.coupling.hibm_mpm.core import (
    HibmMpmPreparedNoSlipSamplingIdentity,
    HibmMpmSurfaceMarkers,
)


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __getitem__(self, key):
        if key is not None:
            raise AssertionError(f"unexpected scalar key {key!r}")
        return self.value


class HibmSamplingIdentityPublishAtomicityTests(unittest.TestCase):
    @staticmethod
    def _make_owner(*, failing_stage: str):
        payload_fields = tuple(object() for _ in range(6))
        obstacle = SimpleNamespace(shape=(4, 4, 4))
        valid_mask = SimpleNamespace(shape=(4, 4, 4))
        faces = tuple(SimpleNamespace(shape=(5,)) for _ in range(3))
        centers = tuple(SimpleNamespace(shape=(4,)) for _ in range(3))
        generation = 41
        owner = SimpleNamespace(
            marker_count=2,
            _no_slip_sampling_identity_generation=generation,
            _current_no_slip_sampling_identity=None,
            _prepared_no_slip_unresolved_marker_count=_FakeScalar(
                1 if failing_stage in {"fallback", "snapshot"} else 0
            ),
            _prepared_no_slip_sample_valid=payload_fields[0],
            _prepared_no_slip_sample_source_code=payload_fields[1],
            _prepared_no_slip_sample_invalid_reason_code=payload_fields[2],
            _prepared_no_slip_sample_position_m=payload_fields[3],
            _prepared_no_slip_marker_position_snapshot_m=payload_fields[4],
            _prepared_no_slip_marker_normal_snapshot=payload_fields[5],
        )
        old_identity = HibmMpmPreparedNoSlipSamplingIdentity(
            generation=generation,
            topology_generation=17,
            component_face_valid_mask_generation=29,
            marker_count=2,
            sample_valid=payload_fields[0],
            sample_source_code=payload_fields[1],
            sample_invalid_reason_code=payload_fields[2],
            sample_position_m=payload_fields[3],
            marker_position_snapshot_m=payload_fields[4],
            marker_normal_snapshot=payload_fields[5],
            _owner=owner,
            _obstacle_field=obstacle,
            _component_face_valid_mask=valid_mask,
            _cell_face_x_m=faces[0],
            _cell_face_y_m=faces[1],
            _cell_face_z_m=faces[2],
            _cell_center_x_m=centers[0],
            _cell_center_y_m=centers[1],
            _cell_center_z_m=centers[2],
        )
        owner._current_no_slip_sampling_identity = old_identity

        def stage_callback(stage: str):
            def callback(*_args):
                if owner._current_no_slip_sampling_identity is not None:
                    raise AssertionError(
                        "old sampling identity remained current before "
                        f"the {stage} device write"
                    )
                if stage == failing_stage:
                    raise RuntimeError(f"injected {stage} failure")

            return callback

        owner._prepare_no_slip_sampling_direct_identity_kernel = stage_callback(
            "direct"
        )
        owner._prepare_no_slip_sampling_fallback_identity_kernel = stage_callback(
            "fallback"
        )
        owner._snapshot_no_slip_sampling_identity_payload_kernel = stage_callback(
            "snapshot"
        )
        arguments = dict(
            obstacle_field=obstacle,
            component_face_valid_mask=valid_mask,
            cell_face_x_m=faces[0],
            cell_face_y_m=faces[1],
            cell_face_z_m=faces[2],
            cell_center_x_m=centers[0],
            cell_center_y_m=centers[1],
            cell_center_z_m=centers[2],
            grid_nodes=(4, 4, 4),
            topology_generation=17,
            component_face_valid_mask_generation=29,
        )
        return owner, old_identity, generation, arguments

    def test_failed_prepare_invalidates_old_identity_without_publishing_generation(
        self,
    ) -> None:
        for failing_stage in ("direct", "fallback", "snapshot"):
            with self.subTest(failing_stage=failing_stage):
                owner, old_identity, generation, arguments = self._make_owner(
                    failing_stage=failing_stage
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"injected {failing_stage} failure",
                ):
                    HibmMpmSurfaceMarkers.prepare_no_slip_sampling_identity(
                        owner,
                        **arguments,
                    )

                self.assertIsNone(owner._current_no_slip_sampling_identity)
                self.assertEqual(
                    owner._no_slip_sampling_identity_generation,
                    generation,
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "stale sampling identity generation",
                ):
                    HibmMpmSurfaceMarkers._audit_prepared_no_slip_sampling_identity(
                        owner,
                        old_identity,
                        topology_generation=17,
                        component_face_valid_mask_generation=29,
                    )


if __name__ == "__main__":
    unittest.main()
