from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

from cases.turek_hron_fsi import (
    TurekHronFsiConfig,
    _validate_fsi_coupling_controls,
    run_turek_hron_fsi,
)


def _config(**overrides: object) -> TurekHronFsiConfig:
    return replace(TurekHronFsiConfig(), **overrides)


class TurekHronFsiCouplingControlGuardTests(unittest.TestCase):
    """Config-level guards for the strong-coupling / reseed controls.

    These reproduce failure modes at the guard, not via a solver run: e.g.
    marker_reseed_interval_steps=0 used to survive config validation and
    only crash at the first gated reseed check (``step_index % 0`` ->
    ZeroDivisionError at step 2).
    """

    def test_default_config_passes_the_guard(self) -> None:
        _validate_fsi_coupling_controls(TurekHronFsiConfig())

    def test_run_turek_hron_fsi_calls_the_guard_before_solver_setup(self) -> None:
        source = inspect.getsource(run_turek_hron_fsi)
        self.assertIn("_validate_fsi_coupling_controls(config)", source)
        self.assertLess(
            source.index("_validate_fsi_coupling_controls(config)"),
            source.index("TaichiRuntimeConfig"),
        )

    # -- fsi_coupling_iterations -------------------------------------------

    def test_coupling_iterations_accepts_one_and_many(self) -> None:
        _validate_fsi_coupling_controls(_config(fsi_coupling_iterations=1))
        _validate_fsi_coupling_controls(_config(fsi_coupling_iterations=12))

    def test_coupling_iterations_rejects_zero_instead_of_silent_clamp(self) -> None:
        with self.assertRaisesRegex(ValueError, "fsi_coupling_iterations"):
            _validate_fsi_coupling_controls(_config(fsi_coupling_iterations=0))

    def test_coupling_iterations_rejects_negative_float_and_bool(self) -> None:
        for bad in (-3, 1.5, True, "2"):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(
                    ValueError, "fsi_coupling_iterations"
                ):
                    _validate_fsi_coupling_controls(
                        _config(fsi_coupling_iterations=bad)
                    )

    # -- fsi_coupling_tolerance --------------------------------------------

    def test_coupling_tolerance_accepts_positive_finite(self) -> None:
        _validate_fsi_coupling_controls(_config(fsi_coupling_tolerance=1.0e-3))

    def test_coupling_tolerance_rejects_zero_negative_and_non_finite(self) -> None:
        for bad in (0.0, -1.0e-3, float("nan"), float("inf"), "loose"):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "fsi_coupling_tolerance"):
                    _validate_fsi_coupling_controls(
                        _config(fsi_coupling_tolerance=bad)
                    )

    # -- fsi_aitken_initial_relaxation -------------------------------------

    def test_aitken_relaxation_accepts_full_valid_range(self) -> None:
        for good in (0.0, 0.5, 1.5):
            with self.subTest(value=good):
                _validate_fsi_coupling_controls(
                    _config(fsi_aitken_initial_relaxation=good)
                )

    def test_aitken_relaxation_rejects_out_of_range_and_non_finite(self) -> None:
        for bad in (-0.1, 1.6, float("nan"), float("-inf"), "half"):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(
                    ValueError, "fsi_aitken_initial_relaxation"
                ):
                    _validate_fsi_coupling_controls(
                        _config(fsi_aitken_initial_relaxation=bad)
                    )

    # -- fsi_coupling_accelerator ------------------------------------------

    def test_accelerator_accepts_known_names_with_normalization(self) -> None:
        for good in ("aitken", "iqn_ils", " AITKEN ", "IQN_ILS"):
            with self.subTest(value=good):
                _validate_fsi_coupling_controls(
                    _config(fsi_coupling_accelerator=good)
                )

    def test_accelerator_rejects_unknown_names_instead_of_silent_fallback(
        self,
    ) -> None:
        for bad in ("iqn-ils", "newton", ""):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(
                    ValueError, "fsi_coupling_accelerator"
                ):
                    _validate_fsi_coupling_controls(
                        _config(fsi_coupling_accelerator=bad)
                    )

    # -- marker_reseed_interval_steps --------------------------------------

    def test_reseed_interval_accepts_none_and_positive_ints(self) -> None:
        _validate_fsi_coupling_controls(
            _config(marker_reseed_interval_steps=None)
        )
        _validate_fsi_coupling_controls(_config(marker_reseed_interval_steps=1))
        _validate_fsi_coupling_controls(
            _config(marker_reseed_interval_steps=50)
        )

    def test_reseed_interval_zero_is_rejected_at_config_time(self) -> None:
        # 0 previously survived until the step loop evaluated
        # ``step_index % 0`` (ZeroDivisionError at step 2).
        with self.assertRaisesRegex(
            ValueError, "marker_reseed_interval_steps"
        ) as raised:
            _validate_fsi_coupling_controls(
                _config(marker_reseed_interval_steps=0)
            )
        self.assertIn("ZeroDivisionError", str(raised.exception))

    def test_reseed_interval_rejects_negative_bool_and_non_int(self) -> None:
        for bad in (-5, True, 2.5, "10"):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(
                    ValueError, "marker_reseed_interval_steps"
                ):
                    _validate_fsi_coupling_controls(
                        _config(marker_reseed_interval_steps=bad)
                    )


if __name__ == "__main__":
    unittest.main()
