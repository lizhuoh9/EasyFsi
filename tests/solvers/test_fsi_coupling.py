from __future__ import annotations

import math
import unittest

from simulation_core.fsi_coupling import (
    InterfaceReactionRelaxationState,
    InterfaceReactionStepUpdate,
    InterfaceReactionTargetEvaluation,
    RegionPairInterfaceReactionTarget,
    _iqn_ils_interface_reaction_guess,
    _least_squares_coefficients,
    action_reaction_balance,
    aitken_relaxation_factor,
    interface_reaction_force,
    region_pair_interface_reaction_forces,
    relax_interface_reaction_forces,
    robin_neumann_impedance_force,
    solve_and_apply_interface_reaction_step,
    solve_interface_reaction_fixed_point,
    update_interface_reaction_for_next_step,
)


class ForceBalanceTests(unittest.TestCase):
    def test_action_reaction_balance_reports_zero_for_equal_and_opposite_forces(self) -> None:
        report = action_reaction_balance((1.0, -2.0, 0.5), (-1.0, 2.0, -0.5))

        self.assertEqual(report.residual_components_n, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(report.residual_norm_n, 0.0)
        self.assertAlmostEqual(report.relative_error, 0.0)

    def test_action_reaction_balance_is_componentwise_not_total_sum_only(self) -> None:
        report = action_reaction_balance((1.0, -1.0), (-0.5, 0.5))

        self.assertEqual(report.residual_components_n, (0.5, -0.5))
        self.assertGreater(report.residual_norm_n, 0.0)
        self.assertGreater(report.relative_error, 0.0)

    def test_action_reaction_balance_rejects_mismatched_vectors(self) -> None:
        with self.assertRaises(ValueError):
            action_reaction_balance((1.0, 2.0), (-1.0,))


class InterfaceReactionFixedPointTests(unittest.TestCase):
    def test_interface_reaction_relaxation_state_updates_without_mutation(self) -> None:
        state = InterfaceReactionRelaxationState(relaxation=0.5)

        first = update_interface_reaction_for_next_step(
            previous_force_n=(0.0, 0.0),
            target_force_n=(10.0, -8.0),
            velocity_mps=(-1.0, 1.0),
            state=state,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
        )
        second = update_interface_reaction_for_next_step(
            previous_force_n=first.update.force_n,
            target_force_n=(7.5, -6.0),
            velocity_mps=(-1.0, 1.0),
            state=first.next_state,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
        )

        self.assertIsInstance(first, InterfaceReactionStepUpdate)
        self.assertEqual(state.previous_residual_n, None)
        self.assertAlmostEqual(first.relaxation, 0.5)
        self.assertEqual(first.update.residual_n, (5.0, -4.0))
        self.assertEqual(first.update.force_n, (5.0, -4.0))
        self.assertEqual(first.next_state.previous_residual_n, (5.0, -4.0))
        self.assertAlmostEqual(first.next_state.relaxation, 0.5)
        self.assertAlmostEqual(second.relaxation, 1.0)
        self.assertAlmostEqual(second.update.force_n[0], 7.5)
        self.assertAlmostEqual(second.update.force_n[1], -6.0)
        self.assertEqual(second.next_state.previous_residual_n, (0.0, 0.0))

    def test_interface_reaction_aitken_lower_bound_is_configurable(self) -> None:
        state = InterfaceReactionRelaxationState(
            previous_residual_n=(1.0,),
            previous_velocity_mps=(0.0,),
            relaxation=0.5,
        )

        result = update_interface_reaction_for_next_step(
            previous_force_n=(0.0,),
            target_force_n=(2.0,),
            velocity_mps=(0.0,),
            state=state,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
            aitken_lower_bound=0.005,
        )

        self.assertAlmostEqual(result.relaxation, 0.005)
        self.assertAlmostEqual(result.update.force_n[0], 0.01)
        self.assertAlmostEqual(result.next_state.relaxation, 0.005)

    def test_interface_reaction_aitken_upper_bound_is_configurable(self) -> None:
        state = InterfaceReactionRelaxationState(
            previous_residual_n=(1.0,),
            previous_velocity_mps=(0.0,),
            relaxation=0.5,
        )

        result = update_interface_reaction_for_next_step(
            previous_force_n=(0.0,),
            target_force_n=(0.5,),
            velocity_mps=(0.0,),
            state=state,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
            aitken_upper_bound=0.25,
        )

        self.assertAlmostEqual(result.relaxation, 0.25)
        self.assertAlmostEqual(result.update.force_n[0], 0.125)
        self.assertAlmostEqual(result.next_state.relaxation, 0.25)

    def test_interface_reaction_step_update_reuses_passivity_limit(self) -> None:
        result = update_interface_reaction_for_next_step(
            previous_force_n=(0.0, 0.0),
            target_force_n=(10.0, -8.0),
            velocity_mps=(0.1, 0.2),
            state=InterfaceReactionRelaxationState(relaxation=0.5),
            initial_relaxation=0.5,
            use_aitken=False,
            passivity_limit=True,
        )

        self.assertEqual(result.update.force_n, (5.0, -4.0))
        self.assertAlmostEqual(sum(result.update.power_w), -0.3)
        self.assertFalse(result.update.passivity_limited[0])
        self.assertAlmostEqual(result.update.force_n[1], -4.0)
        self.assertFalse(result.update.passivity_limited[1])
        self.assertEqual(result.next_state.previous_residual_n, (5.0, -4.0))

    def test_robin_impedance_opposes_interface_velocity_increment(self) -> None:
        state = InterfaceReactionRelaxationState(
            previous_residual_n=(1.0, -1.0),
            previous_velocity_mps=(0.0, 0.2),
            relaxation=0.5,
        )

        result = update_interface_reaction_for_next_step(
            previous_force_n=(0.0, 0.0),
            target_force_n=(10.0, -8.0),
            velocity_mps=(0.2, -0.1),
            state=state,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            robin_impedance_ns_per_m=30.0,
        )

        self.assertEqual(state.previous_velocity_mps, (0.0, 0.2))
        self.assertAlmostEqual(result.robin_impedance_force_n[0], -6.0)
        self.assertAlmostEqual(result.robin_impedance_force_n[1], 9.0)
        self.assertAlmostEqual(result.update.force_n[0], 4.0)
        self.assertAlmostEqual(result.update.force_n[1], 1.0)
        self.assertAlmostEqual(result.update.residual_n[0], 0.0)
        self.assertAlmostEqual(result.update.residual_n[1], 0.0)
        self.assertEqual(result.next_state.previous_residual_n, (0.0, 0.0))
        self.assertEqual(result.next_state.previous_velocity_mps, (0.2, -0.1))

    def test_robin_impedance_is_inactive_without_previous_velocity(self) -> None:
        result = update_interface_reaction_for_next_step(
            previous_force_n=(0.0,),
            target_force_n=(10.0,),
            velocity_mps=(0.2,),
            state=InterfaceReactionRelaxationState(relaxation=1.0),
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            robin_impedance_ns_per_m=30.0,
        )

        self.assertEqual(result.robin_impedance_force_n, (0.0,))
        self.assertEqual(result.update.force_n, (10.0,))
        self.assertEqual(result.next_state.previous_velocity_mps, (0.2,))

    def test_passivity_limit_projects_force_to_zero_power_boundary(self) -> None:
        update = relax_interface_reaction_forces(
            previous_force_n=(0.0, 0.0),
            target_force_n=(4.0, 2.0),
            velocity_mps=(1.0, 0.0),
            relaxation=1.0,
            passivity_limit=True,
        )

        self.assertEqual(update.force_n, (0.0, 2.0))
        self.assertEqual(update.residual_n, (4.0, 0.0))
        self.assertAlmostEqual(sum(update.power_w), 0.0)
        self.assertEqual(update.passivity_limited, (True, False))

    def test_passivity_limited_step_stores_residual_against_committed_force(self) -> None:
        result = update_interface_reaction_for_next_step(
            previous_force_n=(1.0,),
            target_force_n=(3.0,),
            velocity_mps=(1.0,),
            state=InterfaceReactionRelaxationState(relaxation=1.0),
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=True,
        )

        self.assertEqual(result.update.force_n, (0.0,))
        self.assertEqual(result.update.residual_n, (3.0,))
        self.assertAlmostEqual(result.update.residual_norm_n, 3.0)
        self.assertEqual(result.next_state.previous_residual_n, (3.0,))

    def test_interface_reaction_helpers_accept_generic_vector_lengths(self) -> None:
        target = interface_reaction_force((1.0, -2.0, -3.0))
        update = relax_interface_reaction_forces(
            previous_force_n=(0.0, 0.0, 0.0),
            target_force_n=target,
            velocity_mps=(-1.0, 2.0, -3.0),
            relaxation=0.5,
            passivity_limit=True,
        )

        self.assertEqual(target, (-1.0, 2.0, 3.0))
        self.assertAlmostEqual(update.force_n[0], -0.5)
        self.assertAlmostEqual(update.force_n[1], 1.0)
        self.assertAlmostEqual(update.force_n[2], 1.5)
        self.assertFalse(any(update.passivity_limited))

    def test_interface_reaction_uses_opposite_actual_fluid_force(self) -> None:
        target = interface_reaction_force((3.5, -4.25))

        self.assertEqual(target, (-3.5, 4.25))

    def test_interface_reaction_rejects_empty_vectors(self) -> None:
        with self.assertRaises(ValueError):
            interface_reaction_force(())

    def test_region_pair_reaction_uses_full_equal_and_opposite_fluid_forces(self) -> None:
        target = region_pair_interface_reaction_forces(
            primary_fluid_force_n=(1.0, -2.0, 3.0),
            secondary_fluid_force_n=(-4.0, 5.0, -6.0),
        )

        self.assertIsInstance(target, RegionPairInterfaceReactionTarget)
        self.assertEqual(target.primary_force_n, (-1.0, 2.0, -3.0))
        self.assertEqual(target.secondary_force_n, (4.0, -5.0, 6.0))
        self.assertFalse(hasattr(target, "component_pair"))
        self.assertEqual(
            action_reaction_balance(target.primary_force_n, (1.0, -2.0, 3.0)).relative_error,
            0.0,
        )
        self.assertEqual(
            action_reaction_balance(target.secondary_force_n, (-4.0, 5.0, -6.0)).relative_error,
            0.0,
        )

    def test_region_pair_reaction_rejects_mismatched_force_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            region_pair_interface_reaction_forces(
                primary_fluid_force_n=(1.0, 2.0, 3.0),
                secondary_fluid_force_n=(1.0, 2.0),
            )

    def test_region_pair_reaction_rejects_non_3d_force_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary_fluid_force_n"):
            region_pair_interface_reaction_forces(
                primary_fluid_force_n=(1.0, 2.0),
                secondary_fluid_force_n=(-1.0, -2.0),
            )

        with self.assertRaisesRegex(ValueError, "secondary_fluid_force_n"):
            region_pair_interface_reaction_forces(
                primary_fluid_force_n=(1.0, 2.0, 3.0),
                secondary_fluid_force_n=(-1.0,),
            )

    def test_interface_reaction_rejects_nonfinite_force_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "interface_fluid_force_n.*finite"):
            interface_reaction_force((float("nan"),))

        with self.assertRaisesRegex(ValueError, "target_force_n.*finite"):
            relax_interface_reaction_forces(
                previous_force_n=(0.0,),
                target_force_n=(float("inf"),),
                velocity_mps=(0.0,),
                relaxation=1.0,
                passivity_limit=False,
            )

        with self.assertRaisesRegex(ValueError, "velocity_mps.*finite"):
            robin_neumann_impedance_force(
                velocity_mps=(float("nan"),),
                previous_velocity_mps=(0.0,),
                impedance_ns_per_m=1.0,
            )

    def test_fixed_point_rejects_nonfinite_scalar_controls(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(force_n[0] + 1.0,),
                velocity_mps=(0.0,),
            )

        cases = (
            ("tolerance_n", {"tolerance_n": float("nan"), "initial_relaxation": 1.0}),
            ("tolerance_n", {"tolerance_n": float("inf"), "initial_relaxation": 1.0}),
            ("initial_relaxation", {"tolerance_n": 0.0, "initial_relaxation": float("nan")}),
            ("initial_relaxation", {"tolerance_n": 0.0, "initial_relaxation": float("inf")}),
            ("initial_relaxation", {"tolerance_n": 0.0, "initial_relaxation": None}),
            (
                "aitken_lower_bound",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "aitken_lower_bound": float("nan"),
                },
            ),
            (
                "aitken_lower_bound",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "aitken_lower_bound": -0.1,
                },
            ),
            (
                "aitken_lower_bound",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "aitken_lower_bound": 1.6,
                },
            ),
            (
                "aitken_upper_bound",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "aitken_upper_bound": float("nan"),
                },
            ),
            (
                "aitken_upper_bound",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "aitken_lower_bound": 0.5,
                    "aitken_upper_bound": 0.25,
                },
            ),
            (
                "aitken_upper_bound",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "aitken_upper_bound": 1.6,
                },
            ),
            (
                "rejected_trial_backtrack",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "rejected_trial_backtrack": float("nan"),
                },
            ),
            (
                "rejected_trial_backtrack",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "rejected_trial_backtrack": 0.0,
                },
            ),
            (
                "rejected_trial_backtrack",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "rejected_trial_backtrack": -0.1,
                },
            ),
            (
                "rejected_trial_backtrack",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "rejected_trial_backtrack": 1.1,
                },
            ),
            (
                "residual_growth_rejection_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_growth_rejection_factor": float("nan"),
                },
            ),
            (
                "residual_growth_rejection_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_growth_rejection_factor": 0.0,
                },
            ),
            (
                "residual_growth_rejection_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_growth_rejection_factor": 0.5,
                },
            ),
            (
                "max_accepted_residual_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "max_accepted_residual_n": float("nan"),
                },
            ),
            (
                "max_accepted_residual_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "max_accepted_residual_n": -0.1,
                },
            ),
            (
                "trust_region_force_increment_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_force_increment_n": float("nan"),
                },
            ),
            (
                "trust_region_force_increment_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_force_increment_n": 0.0,
                },
            ),
            (
                "trust_region_force_increment_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_force_increment_n": -0.1,
                },
            ),
            (
                "trust_region_shrink_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_force_increment_n": 1.0,
                    "trust_region_shrink_factor": 0.0,
                },
            ),
            (
                "trust_region_growth_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_force_increment_n": 1.0,
                    "trust_region_growth_factor": 0.5,
                },
            ),
            (
                "trust_region_rebound_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_rebound_factor": float("nan"),
                },
            ),
            (
                "trust_region_rebound_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_rebound_factor": 0.5,
                },
            ),
            (
                "trust_region_rebound_backtrack",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_rebound_backtrack": 0.0,
                },
            ),
            (
                "trust_region_rebound_backtrack",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_rebound_backtrack": 1.0,
                },
            ),
            (
                "trust_region_rebound_stop_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_rebound_stop_factor": float("nan"),
                },
            ),
            (
                "trust_region_rebound_stop_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_rebound_stop_factor": 0.5,
                },
            ),
            (
                "trust_region_rebound_stop_max_residual_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_rebound_stop_max_residual_n": float("nan"),
                },
            ),
            (
                "trust_region_rebound_stop_max_residual_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_rebound_stop_max_residual_n": -0.1,
                },
            ),
            (
                "residual_continuation_iterations_max",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_continuation_iterations_max": -1,
                },
            ),
            (
                "residual_continuation_threshold_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_continuation_threshold_n": float("nan"),
                },
            ),
            (
                "residual_continuation_threshold_n",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_continuation_threshold_n": -0.1,
                },
            ),
            (
                "residual_continuation_rebound_secant_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_continuation_rebound_secant_factor": float("nan"),
                },
            ),
            (
                "residual_continuation_rebound_secant_factor",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_continuation_rebound_secant_factor": 0.5,
                },
            ),
            (
                "residual_continuation_rebound_secant_evaluation_extensions_max",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "residual_continuation_rebound_secant_evaluation_extensions_max": -1,
                },
            ),
            (
                "trust_region_adaptive",
                {
                    "tolerance_n": 0.0,
                    "initial_relaxation": 1.0,
                    "trust_region_adaptive": True,
                },
            ),
        )
        for field_name, kwargs in cases:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    solve_interface_reaction_fixed_point(
                        initial_force_n=(0.0,),
                        evaluate_target=evaluate_target,
                        restore_state=restore_state,
                        max_iterations=1,
                        use_aitken=False,
                        passivity_limit=False,
                        **kwargs,
                    )

    def test_fixed_point_rejects_nonfinite_trial_and_target_values(self) -> None:
        def restore_state() -> None:
            return None

        with self.assertRaisesRegex(ValueError, "initial_force_n.*finite"):
            solve_interface_reaction_fixed_point(
                initial_force_n=(float("nan"),),
                evaluate_target=lambda force_n: InterfaceReactionTargetEvaluation(
                    target_force_n=(0.0,),
                    velocity_mps=(0.0,),
                ),
                restore_state=restore_state,
                max_iterations=1,
                tolerance_n=0.0,
                initial_relaxation=1.0,
                use_aitken=False,
                passivity_limit=False,
            )

        with self.assertRaisesRegex(ValueError, "target_force_n.*finite"):
            solve_interface_reaction_fixed_point(
                initial_force_n=(0.0,),
                evaluate_target=lambda force_n: InterfaceReactionTargetEvaluation(
                    target_force_n=(float("inf"),),
                    velocity_mps=(0.0,),
                ),
                restore_state=restore_state,
                max_iterations=1,
                tolerance_n=0.0,
                initial_relaxation=1.0,
                use_aitken=False,
                passivity_limit=False,
            )

    def test_aitken_rejects_nonfinite_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "previous_residual.*finite"):
            aitken_relaxation_factor(
                0.5,
                previous_residual=(float("nan"),),
                current_residual=(0.0,),
            )
        with self.assertRaisesRegex(ValueError, "previous_relaxation.*finite"):
            aitken_relaxation_factor(
                float("inf"),
                previous_residual=(1.0,),
                current_residual=(0.5,),
            )

    def test_solver_re_evaluates_with_updated_interface_reaction_guess_each_trial(self) -> None:
        events: list[tuple[str, tuple[float, float] | None]] = []

        def restore_state() -> None:
            events.append(("restore", None))

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            events.append(("evaluate", force_n))
            target = (force_n[0] + 4.0, force_n[1] - 2.0)
            return InterfaceReactionTargetEvaluation(target_force_n=target, velocity_mps=(-1.0, 1.0))

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0, 0.0),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=0.5,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 3)
        self.assertEqual(
            result.trial_force_history_n,
            ((0.0, 0.0), (2.0, -1.0), (4.0, -2.0)),
        )
        self.assertEqual(
            result.target_force_history_n,
            ((4.0, -2.0), (6.0, -3.0), (8.0, -4.0)),
        )
        self.assertEqual(
            result.residual_history_n,
            ((4.0, -2.0), (4.0, -2.0), (4.0, -2.0)),
        )
        self.assertEqual(
            events,
            [
                ("restore", None),
                ("evaluate", (0.0, 0.0)),
                ("restore", None),
                ("evaluate", (2.0, -1.0)),
                ("restore", None),
                ("evaluate", (4.0, -2.0)),
            ],
        )
        self.assertEqual(result.force_n, (4.0, -2.0))

    def test_fixed_point_uses_configured_aitken_lower_bound(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(2.0 * force_n[0] + 1.0,),
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
            aitken_lower_bound=0.005,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 2)
        self.assertAlmostEqual(result.relaxation, 0.005)

    def test_fixed_point_uses_configured_aitken_upper_bound(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(1.0,),
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
            aitken_upper_bound=0.25,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 2)
        self.assertAlmostEqual(result.relaxation, 0.25)

    def test_unconverged_fixed_point_commits_best_evaluated_trial_not_next_guess(self) -> None:
        guesses: list[tuple[float, ...]] = []

        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            guesses.append(force_n)
            if force_n[0] == 0.0:
                target = (10.0,)
            else:
                target = (force_n[0] + 100.0,)
            return InterfaceReactionTargetEvaluation(target_force_n=target, velocity_mps=(-1.0,))

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (10.0,)))
        self.assertEqual(result.residual_history_n, ((10.0,), (100.0,)))
        self.assertEqual(result.force_n, (0.0,))
        self.assertEqual(result.residual_norm_n, 10.0)

    def test_fixed_point_rejects_converged_trial_when_acceptance_predicate_fails(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                return InterfaceReactionTargetEvaluation(
                    target_force_n=(1.0,),
                    velocity_mps=(0.0,),
                    payload={"trial_cfl": 0.2},
                )
            return InterfaceReactionTargetEvaluation(
                target_force_n=force_n,
                velocity_mps=(0.0,),
                payload={"trial_cfl": 2.0},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["trial_cfl"] < 0.5),
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (1.0,)))
        self.assertEqual(result.accepted_trial_index, 0)
        self.assertEqual(result.rejected_trial_count, 1)
        self.assertEqual(result.rejected_trial_backtrack_count, 0)
        self.assertEqual(result.force_n, (0.0,))
        self.assertEqual(result.accepted_payload, {"trial_cfl": 0.2})

    def test_rejected_trial_backtrack_retries_between_accepted_and_rejected_force(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                return InterfaceReactionTargetEvaluation(
                    target_force_n=(10.0,),
                    velocity_mps=(0.0,),
                    payload={"trial_cfl": 0.1},
                )
            if force_n[0] == 10.0:
                return InterfaceReactionTargetEvaluation(
                    target_force_n=(10.0,),
                    velocity_mps=(0.0,),
                    payload={"trial_cfl": 2.0},
                )
            return InterfaceReactionTargetEvaluation(
                target_force_n=(10.0,),
                velocity_mps=(0.0,),
                payload={"trial_cfl": 0.1},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["trial_cfl"] < 0.5),
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            rejected_trial_backtrack=0.5,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (10.0,), (5.0,)))
        self.assertEqual(result.accepted_trial_index, 2)
        self.assertEqual(result.rejected_trial_count, 1)
        self.assertEqual(result.rejected_trial_backtrack_count, 1)
        self.assertEqual(result.force_n, (5.0,))
        self.assertEqual(result.accepted_payload, {"trial_cfl": 0.1})

    def test_all_rejected_trials_raise_instead_of_zero_force(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(force_n[0] + 1.0,),
                velocity_mps=(0.0,),
                payload={"trial_cfl": 2.0},
            )

        with self.assertRaisesRegex(RuntimeError, "all FSI trials rejected"):
            solve_interface_reaction_fixed_point(
                initial_force_n=(3.0,),
                evaluate_target=evaluate_target,
                accept_evaluation=lambda evaluation: bool(
                    evaluation.payload["trial_cfl"] < 0.5
                ),
                restore_state=restore_state,
                max_iterations=2,
                tolerance_n=0.0,
                initial_relaxation=1.0,
                use_aitken=False,
                passivity_limit=False,
                rejected_trial_backtrack=0.5,
            )

    def test_residual_growth_rejection_backtracks_otherwise_accepted_trial(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (1.0,)
            elif force_n[0] == 1.0:
                target = (100.0,)
            else:
                target = (0.75,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
                payload={"trial_cfl": 0.1},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["trial_cfl"] < 0.5),
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            rejected_trial_backtrack=0.5,
            residual_growth_rejection_factor=2.0,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (1.0,), (0.5,)))
        self.assertEqual(result.residual_history_n, ((1.0,), (99.0,), (0.25,)))
        self.assertEqual(result.accepted_trial_index, 2)
        self.assertEqual(result.rejected_trial_count, 1)
        self.assertEqual(result.rejected_trial_backtrack_count, 1)
        self.assertEqual(result.residual_growth_rejected_trial_count, 1)
        self.assertEqual(result.force_n, (0.5,))

    def test_residual_growth_rejection_is_disabled_by_default(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            target = (1.0,) if force_n[0] == 0.0 else (100.0,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            rejected_trial_backtrack=0.5,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (1.0,)))
        self.assertEqual(result.rejected_trial_count, 0)
        self.assertEqual(result.residual_growth_rejected_trial_count, 0)
        self.assertEqual(result.max_residual_rejected_trial_count, 0)
        self.assertEqual(result.trust_region_limited_update_count, 0)
        self.assertEqual(result.trust_region_rebound_backtrack_count, 0)
        self.assertEqual(result.force_n, (0.0,))

    def test_trust_region_limits_next_force_update_without_rejecting_trial(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(10.0,),
                velocity_mps=(0.0,),
                payload={"trial_cfl": 0.1},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["trial_cfl"] < 0.5),
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_force_increment_n=2.0,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (2.0,), (4.0,)))
        self.assertEqual(result.residual_history_n, ((10.0,), (8.0,), (6.0,)))
        self.assertEqual(result.rejected_trial_count, 0)
        self.assertEqual(result.trust_region_limited_update_count, 3)
        self.assertEqual(result.force_n, (4.0,))

    def test_adaptive_trust_region_shrinks_after_residual_growth(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (10.0,)
            elif force_n[0] == 2.0:
                target = (100.0,)
            else:
                target = (3.5,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=4,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_force_increment_n=2.0,
            trust_region_adaptive=True,
            trust_region_shrink_factor=0.5,
            trust_region_growth_factor=2.0,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (2.0,), (3.0,), (3.5,)))
        self.assertEqual(result.residual_history_n, ((10.0,), (98.0,), (0.5,), (0.0,)))
        self.assertEqual(result.trust_region_limited_update_count, 2)
        self.assertEqual(result.trust_region_shrink_count, 1)
        self.assertEqual(result.trust_region_growth_count, 1)
        self.assertEqual(result.trust_region_effective_force_increment_n, 2.0)
        self.assertEqual(result.force_n, (3.5,))

    def test_rejected_trial_does_not_grow_trust_region(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (1.0,)
                accept = True
            elif force_n[0] == 1.0:
                target = (101.0,)
                accept = True
            else:
                target = (2.5,)
                accept = False
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
                payload={"accept": accept},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["accept"]),
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_force_increment_n=2.0,
            trust_region_adaptive=True,
            trust_region_shrink_factor=0.5,
            trust_region_growth_factor=2.0,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (1.0,), (2.0,)))
        self.assertEqual(result.residual_history_n, ((1.0,), (100.0,), (0.5,)))
        self.assertEqual(result.rejected_trial_count, 1)
        self.assertEqual(result.trust_region_shrink_count, 1)
        self.assertEqual(result.trust_region_growth_count, 0)
        self.assertEqual(result.trust_region_effective_force_increment_n, 1.0)
        self.assertEqual(result.force_n, (0.0,))

    def test_trust_region_growth_uses_previous_accepted_residual(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (1.0,)
                accept = True
            elif force_n[0] == 1.0:
                target = (101.0,)
                accept = False
            else:
                target = (2.5,)
                accept = True
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
                payload={"accept": accept},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["accept"]),
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            rejected_trial_backtrack=0.5,
            trust_region_force_increment_n=2.0,
            trust_region_adaptive=True,
            trust_region_shrink_factor=0.5,
            trust_region_growth_factor=2.0,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (1.0,), (0.5,)))
        self.assertEqual(result.residual_history_n, ((1.0,), (100.0,), (2.0,)))
        self.assertEqual(result.rejected_trial_count, 1)
        self.assertEqual(result.rejected_trial_backtrack_count, 1)
        self.assertEqual(result.trust_region_shrink_count, 2)
        self.assertEqual(result.trust_region_growth_count, 0)
        self.assertEqual(result.trust_region_effective_force_increment_n, 0.5)
        self.assertEqual(result.force_n, (0.0,))

    def test_trust_region_rebound_backtracks_from_worse_trial_to_best_trial(
        self,
    ) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (2.0,)
            elif force_n[0] == 2.0:
                target = (3.0,)
            elif force_n[0] == 3.0:
                target = (10.0,)
            else:
                target = (2.6,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=4,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_rebound_factor=2.0,
            trust_region_rebound_backtrack=0.5,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (2.0,), (3.0,), (2.5,)))
        self.assertEqual(result.residual_history_n[:3], ((2.0,), (1.0,), (7.0,)))
        self.assertAlmostEqual(result.residual_history_n[3][0], 0.1)
        self.assertEqual(result.accepted_trial_index, 3)
        self.assertEqual(result.rejected_trial_count, 0)
        self.assertEqual(result.trust_region_rebound_backtrack_count, 1)
        self.assertEqual(result.force_n, (2.5,))

    def test_trust_region_rebound_stop_commits_best_trial_without_rejecting(
        self,
    ) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (2.0,)
            elif force_n[0] == 2.0:
                target = (3.0,)
            else:
                target = (10.0,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=4,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_rebound_stop_factor=2.0,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 3)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (2.0,), (3.0,)))
        self.assertEqual(result.residual_history_n, ((2.0,), (1.0,), (7.0,)))
        self.assertEqual(result.accepted_trial_index, 1)
        self.assertEqual(result.rejected_trial_count, 0)
        self.assertEqual(result.trust_region_rebound_stop_count, 1)
        self.assertEqual(result.trust_region_rebound_backtrack_count, 0)
        self.assertEqual(result.force_n, (2.0,))
        self.assertEqual(result.residual_norm_n, 1.0)

    def test_rebound_stop_residual_ceiling_continues_high_residual_best_trial(
        self,
    ) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (2.0,)
            elif force_n[0] == 2.0:
                target = (3.0,)
            elif force_n[0] == 3.0:
                target = (10.0,)
            else:
                target = (3.25,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=4,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_rebound_stop_factor=2.0,
            trust_region_rebound_stop_max_residual_n=0.5,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 4)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (2.0,), (3.0,), (10.0,)))
        self.assertEqual(result.residual_history_n, ((2.0,), (1.0,), (7.0,), (-6.75,)))
        self.assertEqual(result.accepted_trial_index, 1)
        self.assertEqual(result.trust_region_rebound_stop_count, 0)
        self.assertEqual(result.trust_region_rebound_stop_suppressed_count, 2)
        self.assertEqual(result.force_n, (2.0,))
        self.assertEqual(result.residual_norm_n, 1.0)

    def test_residual_continuation_extends_only_until_quality_threshold(
        self,
    ) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (1.0,)
            elif force_n[0] == 1.0:
                target = (1.75,)
            else:
                target = (1.9,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            residual_continuation_iterations_max=3,
            residual_continuation_threshold_n=0.5,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 3)
        self.assertEqual(result.residual_continuation_iteration_count, 1)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (1.0,), (1.75,)))
        self.assertEqual(result.residual_history_n[:2], ((1.0,), (0.75,)))
        self.assertAlmostEqual(result.residual_history_n[2][0], 0.15)
        self.assertEqual(result.force_n, (1.75,))
        self.assertAlmostEqual(result.residual_norm_n, 0.15)

    def test_residual_continuation_rebound_secant_restarts_from_best_trial(
        self,
    ) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (2.0,)
            elif force_n[0] == 2.0:
                target = (3.0,)
            elif force_n[0] == 3.0:
                target = (10.0,)
            elif force_n[0] < 2.0:
                target = (force_n[0] + 0.1,)
            else:
                target = (20.0,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_force_increment_n=2.0,
            trust_region_rebound_stop_factor=2.0,
            residual_continuation_iterations_max=2,
            residual_continuation_threshold_n=0.5,
            residual_continuation_rebound_secant_from_best=True,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 4)
        self.assertEqual(result.residual_continuation_iteration_count, 2)
        self.assertEqual(result.residual_continuation_rebound_secant_count, 1)
        self.assertEqual(
            result.residual_continuation_rebound_secant_evaluation_extension_count,
            0,
        )
        self.assertEqual(result.trust_region_rebound_stop_count, 0)
        self.assertEqual(result.trial_force_history_n[:3], ((0.0,), (2.0,), (3.0,)))
        self.assertAlmostEqual(result.trial_force_history_n[3][0], 2.0 - 1.0 / 6.0)
        self.assertEqual(result.accepted_trial_index, 3)
        self.assertAlmostEqual(result.force_n[0], 2.0 - 1.0 / 6.0)
        self.assertAlmostEqual(result.residual_norm_n, 0.1)

    def test_residual_continuation_rebound_secant_factor_can_trigger_before_stop2(
        self,
    ) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (2.0,)
            elif force_n[0] == 2.0:
                target = (3.0,)
            elif force_n[0] == 3.0:
                target = (4.9,)
            elif force_n[0] < 2.0:
                target = (force_n[0] + 0.1,)
            else:
                target = (20.0,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_force_increment_n=2.0,
            trust_region_rebound_stop_factor=2.0,
            residual_continuation_iterations_max=2,
            residual_continuation_threshold_n=0.5,
            residual_continuation_rebound_secant_from_best=True,
            residual_continuation_rebound_secant_factor=1.0,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 4)
        self.assertEqual(result.residual_continuation_rebound_secant_count, 1)
        self.assertEqual(result.trust_region_rebound_stop_count, 0)
        self.assertEqual(result.accepted_trial_index, 3)
        self.assertAlmostEqual(result.force_n[0], 2.0 - 1.0 / 0.9)
        self.assertAlmostEqual(result.residual_norm_n, 0.1)

    def test_residual_continuation_rebound_secant_extension_evaluates_final_candidate(
        self,
    ) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (2.0,)
            elif force_n[0] == 2.0:
                target = (3.0,)
            elif force_n[0] == 3.0:
                target = (10.0,)
            elif force_n[0] < 2.0:
                target = (force_n[0] + 0.1,)
            else:
                target = (20.0,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            trust_region_force_increment_n=2.0,
            trust_region_rebound_stop_factor=2.0,
            residual_continuation_iterations_max=1,
            residual_continuation_threshold_n=0.5,
            residual_continuation_rebound_secant_from_best=True,
            residual_continuation_rebound_secant_evaluation_extensions_max=1,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 4)
        self.assertEqual(result.residual_continuation_iteration_count, 1)
        self.assertEqual(result.residual_continuation_rebound_secant_count, 1)
        self.assertEqual(
            result.residual_continuation_rebound_secant_evaluation_extension_count,
            1,
        )
        self.assertEqual(result.trial_force_history_n[:3], ((0.0,), (2.0,), (3.0,)))
        self.assertAlmostEqual(result.trial_force_history_n[3][0], 2.0 - 1.0 / 6.0)
        self.assertEqual(result.accepted_trial_index, 3)
        self.assertAlmostEqual(result.residual_norm_n, 0.1)

    def test_max_accepted_residual_backtracks_otherwise_accepted_trial(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 0.0:
                target = (1.0,)
            elif force_n[0] == 1.0:
                target = (100.0,)
            else:
                target = (0.75,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
                payload={"trial_cfl": 0.1},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["trial_cfl"] < 0.5),
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            rejected_trial_backtrack=0.5,
            max_accepted_residual_n=2.0,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (1.0,), (0.5,)))
        self.assertEqual(result.residual_history_n, ((1.0,), (99.0,), (0.25,)))
        self.assertEqual(result.accepted_trial_index, 2)
        self.assertEqual(result.rejected_trial_count, 1)
        self.assertEqual(result.rejected_trial_backtrack_count, 1)
        self.assertEqual(result.max_residual_rejected_trial_count, 1)
        self.assertEqual(result.force_n, (0.5,))

    def test_rejected_first_trial_backtracks_toward_zero_force_without_accepted_anchor(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            if force_n[0] == 8.0:
                return InterfaceReactionTargetEvaluation(
                    target_force_n=force_n,
                    velocity_mps=(0.0,),
                    payload={"trial_cfl": 2.0},
                )
            return InterfaceReactionTargetEvaluation(
                target_force_n=(8.0,),
                velocity_mps=(0.0,),
                payload={"trial_cfl": 0.1},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(8.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["trial_cfl"] < 0.5),
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            rejected_trial_backtrack=0.25,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((8.0,), (2.0,)))
        self.assertEqual(result.accepted_trial_index, 1)
        self.assertEqual(result.rejected_trial_count, 1)
        self.assertEqual(result.rejected_trial_backtrack_count, 1)
        self.assertEqual(result.force_n, (2.0,))

    def test_all_rejected_trials_can_explicitly_fallback_to_initial_force(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=force_n,
                velocity_mps=(0.0,),
                payload={"trial_cfl": 2.0},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(8.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["trial_cfl"] < 0.5),
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            rejected_trial_backtrack=0.25,
            all_rejected_trial_policy="initial_force",
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((8.0,), (2.0,)))
        self.assertIsNone(result.accepted_trial_index)
        self.assertEqual(result.rejected_trial_count, 2)
        self.assertEqual(result.rejected_trial_backtrack_count, 2)
        self.assertEqual(result.force_n, (8.0,))
        self.assertTrue(result.all_trials_rejected)
        self.assertTrue(result.zero_force_commit_blocked)
        self.assertEqual(result.fallback_force_source, "initial_force")
        self.assertTrue(math.isinf(result.residual_norm_n))

    def test_all_rejected_trials_are_not_reported_as_converged(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=force_n,
                velocity_mps=(0.0,),
                payload={"trial_cfl": 2.0},
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(8.0,),
            evaluate_target=evaluate_target,
            accept_evaluation=lambda evaluation: bool(evaluation.payload["trial_cfl"] < 0.5),
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            rejected_trial_backtrack=0.25,
            all_rejected_trial_policy="initial_force",
        )

        self.assertFalse(result.converged)
        self.assertTrue(result.all_trials_rejected)
        self.assertTrue(result.zero_force_commit_blocked)
        self.assertEqual(result.fallback_force_source, "initial_force")
        self.assertIsNone(result.accepted_trial_index)
        self.assertEqual(result.force_n, (8.0,))

    def test_passivity_limit_is_committed_after_fixed_point_trials_not_used_as_guess(self) -> None:
        guesses: list[tuple[float, ...]] = []

        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            guesses.append(force_n)
            return InterfaceReactionTargetEvaluation(
                target_force_n=(13.333333333333334,),
                velocity_mps=(1.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=True,
        )

        self.assertEqual(guesses[0], (0.0,))
        self.assertAlmostEqual(guesses[1][0], 13.333333333333334)
        self.assertEqual(result.force_n, (0.0,))

    def test_passivity_limited_fixed_point_reports_committed_residual_norm(self) -> None:
        guesses: list[tuple[float, ...]] = []

        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            guesses.append(force_n)
            target = (10.0,) if force_n[0] == 0.0 else (10.5,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(1.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=True,
        )

        self.assertFalse(result.converged)
        self.assertEqual(guesses, [(0.0,), (10.0,)])
        self.assertEqual(result.force_n, (0.0,))
        self.assertAlmostEqual(result.residual_norm_n, 10.5)

    def test_passivity_limited_fixed_point_rechecks_convergence_after_projection(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=force_n,
                velocity_mps=(1.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(10.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=1,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=True,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.force_n, (0.0,))
        self.assertAlmostEqual(result.residual_norm_n, 10.0)

    def test_iqn_ils_solves_cross_coupled_linear_fixed_point(self) -> None:
        calls: list[tuple[float, ...]] = []
        coupling_matrix = ((0.25, 0.35), (-0.2, 0.1))
        load_n = (2.0, -1.0)
        expected_force_n = (1.946308724832215, -1.5436241610738255)

        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            calls.append(force_n)
            target = (
                coupling_matrix[0][0] * force_n[0]
                + coupling_matrix[0][1] * force_n[1]
                + load_n[0],
                coupling_matrix[1][0] * force_n[0]
                + coupling_matrix[1][1] * force_n[1]
                + load_n[1],
            )
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(-0.1, 0.2),
            )

        aitken_result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0, 0.0),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=4,
            tolerance_n=1.0e-10,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
        )
        calls.clear()

        iqn_result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0, 0.0),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=4,
            tolerance_n=1.0e-10,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
            solver="iqn_ils",
        )

        self.assertFalse(aitken_result.converged)
        self.assertGreater(aitken_result.residual_norm_n, 1.0e-2)
        self.assertTrue(iqn_result.converged)
        self.assertLess(iqn_result.residual_norm_n, 1.0e-10)
        self.assertEqual(len(calls), iqn_result.iterations_used)
        self.assertAlmostEqual(iqn_result.force_n[0], expected_force_n[0], places=9)
        self.assertAlmostEqual(iqn_result.force_n[1], expected_force_n[1], places=9)

    def test_iqn_ils_uses_least_squares_in_six_component_production_iteration_budget(self) -> None:
        coupling_matrix = ((0.25, 0.35), (-0.2, 0.1))
        load_pairs_n = ((2.0, -1.0), (0.5, 3.0), (-2.5, 1.25))
        determinant = 0.745
        expected_force_n = tuple(
            component
            for load_x, load_y in load_pairs_n
            for component in (
                (0.9 * load_x + 0.35 * load_y) / determinant,
                (-0.2 * load_x + 0.75 * load_y) / determinant,
            )
        )

        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            target: list[float] = []
            for pair_index, (load_x, load_y) in enumerate(load_pairs_n):
                force_x = force_n[2 * pair_index]
                force_y = force_n[2 * pair_index + 1]
                target.extend(
                    (
                        coupling_matrix[0][0] * force_x
                        + coupling_matrix[0][1] * force_y
                        + load_x,
                        coupling_matrix[1][0] * force_x
                        + coupling_matrix[1][1] * force_y
                        + load_y,
                    )
                )
            return InterfaceReactionTargetEvaluation(
                target_force_n=tuple(target),
                velocity_mps=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=6,
            tolerance_n=1.0e-10,
            initial_relaxation=0.5,
            use_aitken=True,
            passivity_limit=False,
            solver="iqn_ils",
        )

        self.assertTrue(result.converged)
        self.assertGreater(result.iqn_ils_least_squares_update_count, 0)
        self.assertLessEqual(result.iterations_used, 4)
        for actual, expected in zip(result.force_n, expected_force_n, strict=True):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_iqn_ils_uses_one_residual_difference_in_six_component_budget(self) -> None:
        load_n = (2.0, -1.0, 0.5, 3.0, -2.5, 1.25)

        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=tuple(0.2 * force + load for force, load in zip(force_n, load_n)),
                velocity_mps=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=1.0e-12,
            initial_relaxation=0.5,
            use_aitken=False,
            passivity_limit=False,
            solver="iqn_ils",
        )

        self.assertEqual(result.iterations_used, 2)
        self.assertGreater(result.iqn_ils_least_squares_update_count, 0)

    def test_iqn_ils_update_keeps_residual_component_outside_history_span(self) -> None:
        proposed, used_least_squares = _iqn_ils_interface_reaction_guess(
            trial_force_history=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
            ),
            residual_history=(
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            current_residual_n=(1.0, 1.0, 1.0),
            current_target_force_n=(2.0, 2.0, 1.0),
            current_velocity_mps=(0.0, 0.0, 0.0),
            relaxation=0.5,
        )

        self.assertTrue(used_least_squares)
        self.assertEqual(proposed, (0.0, 0.0, 1.0))

    def test_iqn_ils_least_squares_filters_dependent_history_columns(self) -> None:
        coefficients = _least_squares_coefficients(
            (
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            (1.0, 2.0, 0.0),
        )

        self.assertIsNotNone(coefficients)
        assert coefficients is not None
        reconstructed = tuple(
            sum(column[axis] * coefficient for column, coefficient in zip(
                (
                    (1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                coefficients,
            ))
            for axis in range(3)
        )
        self.assertEqual(reconstructed, (1.0, 2.0, 0.0))

    def test_iqn_ils_falls_back_to_relaxed_update_for_rank_deficient_history(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(force_n[0] + 4.0,),
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=1.0e-10,
            initial_relaxation=0.5,
            use_aitken=False,
            passivity_limit=False,
            solver="iqn_ils",
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.trial_force_history_n, ((0.0,), (2.0,), (4.0,)))
        self.assertEqual(result.residual_history_n, ((4.0,), (4.0,), (4.0,)))
        self.assertEqual(result.force_n, (4.0,))

    def test_fixed_point_reports_interface_map_secant_amplification(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(2.0 * force_n[0] + 3.0,),
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertAlmostEqual(result.interface_map_amplification_max, 2.0)
        self.assertAlmostEqual(result.residual_jacobian_amplification_max, 1.0)
        self.assertEqual(result.interface_map_amplification_sample_count, 1)
        self.assertEqual(result.residual_jacobian_amplification_sample_count, 1)

    def test_fixed_point_reports_unmeasured_secant_when_only_one_trial_exists(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=force_n,
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(2.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations_used, 1)
        self.assertEqual(result.interface_map_amplification_max, 0.0)
        self.assertEqual(result.interface_map_amplification_sample_count, 0)
        self.assertEqual(result.physical_interface_map_amplification_sample_count, 0)
        self.assertEqual(result.diagnostic_interface_map_amplification_sample_count, 0)

    def test_target_map_relaxation_stabilizes_solver_map_without_changing_physical_fixed_point(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(-3.0 * force_n[0] + 4.0,),
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=1.0e-12,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
            target_map_relaxation=0.25,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.force_n, (1.0,))
        self.assertEqual(result.physical_target_force_history_n, ((4.0,), (1.0,)))
        self.assertEqual(result.target_force_history_n, ((1.0,), (1.0,)))
        self.assertAlmostEqual(result.physical_interface_map_amplification_max, 3.0)
        self.assertAlmostEqual(result.interface_map_amplification_max, 0.0)
        self.assertEqual(result.physical_interface_map_amplification_sample_count, 1)
        self.assertEqual(result.interface_map_amplification_sample_count, 1)
        self.assertAlmostEqual(result.residual_norm_n, 0.0)
        self.assertAlmostEqual(result.target_map_relaxation, 0.25)

    def test_fixed_point_reports_diagnostic_target_map_separately_from_solver_target(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(2.0 * force_n[0],),
                diagnostic_target_force_n=(5.0 * force_n[0],),
                velocity_mps=(0.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(1.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertEqual(result.physical_target_force_history_n, ((2.0,), (4.0,)))
        self.assertEqual(result.diagnostic_target_force_history_n, ((5.0,), (10.0,)))
        self.assertAlmostEqual(result.physical_interface_map_amplification_max, 2.0)
        self.assertAlmostEqual(result.diagnostic_interface_map_amplification_max, 5.0)
        self.assertEqual(result.physical_interface_map_amplification_sample_count, 1)
        self.assertEqual(result.diagnostic_interface_map_amplification_sample_count, 1)

    def test_fixed_point_rejects_invalid_target_map_relaxation(self) -> None:
        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=force_n,
                velocity_mps=(0.0,),
            )

        for value in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "target_map_relaxation"):
                    solve_interface_reaction_fixed_point(
                        initial_force_n=(0.0,),
                        evaluate_target=evaluate_target,
                        restore_state=restore_state,
                        max_iterations=1,
                        tolerance_n=0.0,
                        initial_relaxation=1.0,
                        use_aitken=False,
                        passivity_limit=False,
                        target_map_relaxation=value,
                    )

    def test_single_iteration_fixed_point_reports_not_converged_when_residual_remains(self) -> None:
        def restore_state() -> None:
            pass

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            return InterfaceReactionTargetEvaluation(
                target_force_n=(force_n[0] + 10.0, force_n[1] - 5.0),
                velocity_mps=(-1.0, 1.0),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0, 0.0),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=1,
            tolerance_n=1.0e-6,
            initial_relaxation=0.5,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations_used, 1)
        self.assertAlmostEqual(result.residual_norm_n, (10.0 * 10.0 + 5.0 * 5.0) ** 0.5)
        self.assertEqual(result.residual_history_n, ((10.0, -5.0),))
        self.assertEqual(result.force_n, (0.0, 0.0))

    def test_solve_and_apply_step_restores_then_commits_accepted_force(self) -> None:
        events: list[tuple[str, tuple[float, float] | None]] = []

        def save_state() -> None:
            events.append(("save", None))

        def restore_state() -> None:
            events.append(("restore", None))

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            force = (force_n[0], force_n[1])
            events.append(("evaluate", force))
            target = (-2.0, 3.0) if force == (0.0, 0.0) else force
            return InterfaceReactionTargetEvaluation(target_force_n=target, velocity_mps=(-1.0, 1.0))

        def apply_force(force_n: tuple[float, ...]) -> None:
            events.append(("apply", (force_n[0], force_n[1])))

        result = solve_and_apply_interface_reaction_step(
            initial_force_n=(0.0, 0.0),
            save_state=save_state,
            restore_state=restore_state,
            evaluate_target=evaluate_target,
            apply_force=apply_force,
            max_iterations=3,
            tolerance_n=1.0e-8,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.force_n, (-2.0, 3.0))
        self.assertEqual(
            events,
            [
                ("save", None),
                ("restore", None),
                ("evaluate", (0.0, 0.0)),
                ("restore", None),
                ("evaluate", (-2.0, 3.0)),
                ("restore", None),
                ("apply", (-2.0, 3.0)),
            ],
        )

    def test_solve_and_apply_step_can_commit_reusable_current_trial_state(self) -> None:
        events: list[tuple[str, object]] = []

        def save_state() -> None:
            events.append(("save", None))

        def restore_state() -> None:
            events.append(("restore", None))

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            events.append(("evaluate", force_n))
            return InterfaceReactionTargetEvaluation(
                target_force_n=force_n,
                velocity_mps=(-1.0, 1.0),
                payload={"trial_force": force_n},
            )

        def commit_state(payload: object | None) -> None:
            events.append(("commit", payload))

        def apply_force(force_n: tuple[float, ...]) -> None:
            events.append(("apply", force_n))

        result = solve_and_apply_interface_reaction_step(
            initial_force_n=(2.0, -3.0),
            save_state=save_state,
            restore_state=restore_state,
            evaluate_target=evaluate_target,
            apply_force=apply_force,
            commit_accepted_state=commit_state,
            max_iterations=2,
            tolerance_n=1.0e-12,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertTrue(result.converged)
        self.assertTrue(result.accepted_state_reusable)
        self.assertEqual(result.accepted_trial_index, 0)
        self.assertEqual(result.accepted_payload, {"trial_force": (2.0, -3.0)})
        self.assertEqual(
            events,
            [
                ("save", None),
                ("restore", None),
                ("evaluate", (2.0, -3.0)),
                ("commit", {"trial_force": (2.0, -3.0)}),
                ("apply", (2.0, -3.0)),
            ],
        )

    def test_solve_and_apply_step_restores_when_passivity_changes_accepted_force(self) -> None:
        events: list[tuple[str, object]] = []

        def save_state() -> None:
            events.append(("save", None))

        def restore_state() -> None:
            events.append(("restore", None))

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            events.append(("evaluate", force_n))
            return InterfaceReactionTargetEvaluation(
                target_force_n=force_n,
                velocity_mps=(1.0,),
                payload="passivity-would-change-force",
            )

        def commit_state(payload: object | None) -> None:
            events.append(("commit", payload))

        def apply_force(force_n: tuple[float, ...]) -> None:
            events.append(("apply", force_n))

        result = solve_and_apply_interface_reaction_step(
            initial_force_n=(4.0,),
            save_state=save_state,
            restore_state=restore_state,
            evaluate_target=evaluate_target,
            apply_force=apply_force,
            commit_accepted_state=commit_state,
            max_iterations=1,
            tolerance_n=1.0e-12,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=True,
        )

        self.assertFalse(result.accepted_state_reusable)
        self.assertEqual(result.force_n, (0.0,))
        self.assertEqual(
            events,
            [
                ("save", None),
                ("restore", None),
                ("evaluate", (4.0,)),
                ("restore", None),
                ("apply", (0.0,)),
            ],
        )

    def test_solve_and_apply_step_restores_when_accepted_trial_is_not_current(self) -> None:
        events: list[tuple[str, object]] = []

        def save_state() -> None:
            events.append(("save", None))

        def restore_state() -> None:
            events.append(("restore", None))

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            events.append(("evaluate", force_n))
            if force_n == (0.0,):
                target = (1.0,)
            else:
                target = (101.0,)
            return InterfaceReactionTargetEvaluation(
                target_force_n=target,
                velocity_mps=(0.0,),
                payload={"force": force_n},
            )

        def commit_state(payload: object | None) -> None:
            events.append(("commit", payload))

        def apply_force(force_n: tuple[float, ...]) -> None:
            events.append(("apply", force_n))

        result = solve_and_apply_interface_reaction_step(
            initial_force_n=(0.0,),
            save_state=save_state,
            restore_state=restore_state,
            evaluate_target=evaluate_target,
            apply_force=apply_force,
            commit_accepted_state=commit_state,
            max_iterations=2,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=False,
        )

        self.assertFalse(result.converged)
        self.assertFalse(result.accepted_state_reusable)
        self.assertEqual(result.accepted_trial_index, 0)
        self.assertEqual(result.force_n, (0.0,))
        self.assertEqual(
            events,
            [
                ("save", None),
                ("restore", None),
                ("evaluate", (0.0,)),
                ("restore", None),
                ("evaluate", (1.0,)),
                ("restore", None),
                ("apply", (0.0,)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
