"""Focused contracts for classified invalid shared-marker samples.

An invalid prepared sample is still an immutable transaction payload.  Its
reason must distinguish a marker outside the half-open grid from an in-domain
marker for which no complete three-component MAC support can be found.  Both
the diagnostic sampler and marker-MAC projector must audit that reason field.
"""

from __future__ import annotations

import unittest

import numpy as np

from simulation_core.coupling.hibm_mpm.core import (
    HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NO_COMPLETE_MAC_SUPPORT,
    HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NONE,
    HIBM_NO_SLIP_SAMPLE_INVALID_REASON_OUTSIDE_HALF_OPEN_DOMAIN,
)
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HibmMpmMarkerMacConstraintOperator,
)
from tests.solvers.test_hibm_shared_marker_sampling_identity import (
    _SharedSamplingFixture,
)


class HibmSharedMarkerInvalidReasonContractTests(unittest.TestCase):
    _fixture: _SharedSamplingFixture | None = None

    @classmethod
    def fixture(cls) -> _SharedSamplingFixture:
        if cls._fixture is None:
            cls._fixture = _SharedSamplingFixture()
        return cls._fixture

    @staticmethod
    def prepare_operator(fixture, identity) -> HibmMpmMarkerMacConstraintOperator:
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=1,
        )
        operator.prepare(
            markers=fixture.markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )
        return operator

    def test_prepared_token_outside_domain_maps_to_outside_failure(self) -> None:
        fixture = self.fixture()
        fixture.reset(
            position=(1.0, 0.375, 0.375),
            normal=(-1.0, 0.0, 0.0),
        )

        identity = fixture.prepare_identity()

        self.assertEqual(int(identity.sample_valid[0]), 0)
        self.assertEqual(int(identity.sample_source_code[0]), 0)
        self.assertEqual(
            int(identity.sample_invalid_reason_code[0]),
            HIBM_NO_SLIP_SAMPLE_INVALID_REASON_OUTSIDE_HALF_OPEN_DOMAIN,
        )
        velocity_before = fixture.velocity.to_numpy().copy()
        with self.assertRaisesRegex(
            RuntimeError,
            "outside.*half-open|half-open.*domain|marker.*outside",
        ):
            self.prepare_operator(fixture, identity)
        np.testing.assert_array_equal(fixture.velocity.to_numpy(), velocity_before)

    def test_prepared_token_without_complete_support_maps_to_support_failure(
        self,
    ) -> None:
        fixture = self.fixture()
        fixture.reset(
            position=(0.5, 0.375, 0.375),
            normal=(0.0, 0.0, 0.0),
        )
        fixture.obstacle.fill(1)
        incomplete_candidate = (1, 1, 1)
        fixture.obstacle[incomplete_candidate] = 0
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[incomplete_candidate] = (1 << 0) | (1 << 1)
        fixture.component_face_valid_mask.from_numpy(valid_mask)

        identity = fixture.prepare_identity()

        self.assertEqual(int(identity.sample_valid[0]), 0)
        self.assertEqual(int(identity.sample_source_code[0]), 0)
        self.assertEqual(
            int(identity.sample_invalid_reason_code[0]),
            HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NO_COMPLETE_MAC_SUPPORT,
        )
        velocity_before = fixture.velocity.to_numpy().copy()
        with self.assertRaisesRegex(
            RuntimeError,
            "no valid MAC component support|complete MAC support",
        ):
            self.prepare_operator(fixture, identity)
        np.testing.assert_array_equal(fixture.velocity.to_numpy(), velocity_before)

    def test_invalid_reason_payload_tamper_is_rejected_atomically(self) -> None:
        fixture = self.fixture()
        fixture.reset()
        identity = fixture.prepare_identity()
        self.assertEqual(
            int(identity.sample_invalid_reason_code[0]),
            HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NONE,
        )
        velocity_before = fixture.velocity.to_numpy().copy()
        identity.sample_invalid_reason_code[0] = (
            HIBM_NO_SLIP_SAMPLE_INVALID_REASON_OUTSIDE_HALF_OPEN_DOMAIN
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "payload|sampling identity|changed|tamper",
        ):
            fixture.sample_residual(
                identity,
                topology_generation=fixture.TOPOLOGY_GENERATION,
                component_face_valid_mask_generation=(
                    fixture.VALID_MASK_GENERATION
                ),
            )
        np.testing.assert_array_equal(fixture.velocity.to_numpy(), velocity_before)

        # Reach the projector-owned snapshot audit independently: synchronize
        # the marker-owned baseline after preparing the operator, while leaving
        # the operator's private transaction snapshot unchanged.
        fixture.reset()
        identity = fixture.prepare_identity()
        operator = self.prepare_operator(fixture, identity)
        self.assertEqual(
            int(operator._sampling_invalid_reason_snapshot[0]),
            HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NONE,
        )
        state_before = tuple(
            field.to_numpy().tobytes(order="C")
            for field in (
                fixture.velocity,
                fixture.component_face_valid_mask,
                fixture.obstacle,
            )
        )
        identity.sample_invalid_reason_code[0] = (
            HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NO_COMPLETE_MAC_SUPPORT
        )
        fixture.markers._prepared_no_slip_sample_invalid_reason_code_snapshot[0] = (
            HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NO_COMPLETE_MAC_SUPPORT
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "stale|payload|sampling identity|changed|tamper",
        ):
            operator.solve_device(
                max_iterations=32,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=(
                    fixture.component_face_valid_mask
                ),
                topology_generation=fixture.TOPOLOGY_GENERATION,
                component_face_valid_mask_generation=(
                    fixture.VALID_MASK_GENERATION
                ),
                obstacle_field=fixture.obstacle,
            )
        state_after = tuple(
            field.to_numpy().tobytes(order="C")
            for field in (
                fixture.velocity,
                fixture.component_face_valid_mask,
                fixture.obstacle,
            )
        )
        self.assertEqual(state_after, state_before)
        self.assertFalse(operator.report().committed)


if __name__ == "__main__":
    unittest.main()
