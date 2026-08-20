from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

from benchmarks.official import solid_mpm_fsi_runner


class _FakeSolid:
    def __init__(self, *, fail_step: int | None = None, fail_end: bool = False) -> None:
        self.fail_step = fail_step
        self.fail_end = fail_end
        self.events: list[object] = []
        self.step_kwargs: list[dict[str, object]] = []
        self.final_report = object()

    def begin_out_of_bounds_guard_batch(self) -> None:
        self.events.append("begin")

    def step(self, **kwargs):
        step_number = len(self.step_kwargs) + 1
        self.events.append(("step", step_number))
        self.step_kwargs.append(dict(kwargs))
        if self.fail_step == step_number:
            raise RuntimeError(f"step {step_number} failed")
        return None

    def enforce_rest_x_plane(self) -> None:
        self.events.append("plane")

    def end_out_of_bounds_guard_batch(self):
        self.events.append("end")
        if self.fail_end:
            raise RuntimeError("batch end failed")
        return self.final_report

    def abort_out_of_bounds_guard_batch(self) -> None:
        self.events.append("abort")


def _config(*, enforce_plane_strain_x: bool) -> SimpleNamespace:
    return SimpleNamespace(
        enforce_plane_strain_x=enforce_plane_strain_x,
        fixed_node_lock_policy="any_fixed_particle",
        solid_constitutive_model="plane_stress_linear_elastic",
        solid_velocity_transfer_flip_blend=0.25,
    )


class RectangularSolidOutOfBoundsBatchTests(unittest.TestCase):
    def test_rectangular_runner_delegates_the_solid_loop_to_batch_helper(self) -> None:
        source = inspect.getsource(
            solid_mpm_fsi_runner.run_hibm_mpm_fsi
        )

        self.assertIn(
            "latest_solid_report = _advance_solid_substeps_batched(", source
        )
        self.assertNotIn("for _solid_substep in range(solid_substeps)", source)

    def test_success_preserves_substep_and_plane_enforcement_order(self) -> None:
        solid = _FakeSolid()

        report = solid_mpm_fsi_runner._advance_solid_substeps_batched(
            solid,
            _config(enforce_plane_strain_x=True),
            solid_substeps=3,
            solid_substep_dt_s=1.25e-6,
            mu_pa=2.0,
            lambda_pa=3.0,
            solid_substep_velocity_damping=0.999,
        )

        self.assertIs(report, solid.final_report)
        self.assertEqual(
            solid.events,
            [
                "begin",
                ("step", 1),
                "plane",
                ("step", 2),
                "plane",
                ("step", 3),
                "plane",
                "end",
            ],
        )
        self.assertEqual(len(solid.step_kwargs), 3)
        for kwargs in solid.step_kwargs:
            self.assertFalse(kwargs["read_report"])
            self.assertEqual(kwargs["dt_s"], 1.25e-6)
            self.assertEqual(kwargs["mu_pa"], 2.0)
            self.assertEqual(kwargs["lambda_pa"], 3.0)
            self.assertEqual(kwargs["primary_region_id"], 101)
            self.assertEqual(kwargs["secondary_region_id"], 202)
            self.assertEqual(kwargs["velocity_damping"], 0.999)
            self.assertEqual(kwargs["fixed_node_lock_policy"], "any_fixed_particle")
            self.assertEqual(
                kwargs["constitutive_model"], "plane_stress_linear_elastic"
            )
            self.assertEqual(kwargs["velocity_transfer_flip_blend"], 0.25)

    def test_success_without_plane_strain_does_not_insert_enforcement(self) -> None:
        solid = _FakeSolid()

        solid_mpm_fsi_runner._advance_solid_substeps_batched(
            solid,
            _config(enforce_plane_strain_x=False),
            solid_substeps=2,
            solid_substep_dt_s=1.0e-6,
            mu_pa=2.0,
            lambda_pa=3.0,
            solid_substep_velocity_damping=1.0,
        )

        self.assertEqual(
            solid.events,
            ["begin", ("step", 1), ("step", 2), "end"],
        )

    def test_substep_failure_aborts_batch_and_propagates(self) -> None:
        solid = _FakeSolid(fail_step=2)

        with self.assertRaisesRegex(RuntimeError, "step 2 failed"):
            solid_mpm_fsi_runner._advance_solid_substeps_batched(
                solid,
                _config(enforce_plane_strain_x=True),
                solid_substeps=3,
                solid_substep_dt_s=1.0e-6,
                mu_pa=2.0,
                lambda_pa=3.0,
                solid_substep_velocity_damping=1.0,
            )

        self.assertEqual(
            solid.events,
            ["begin", ("step", 1), "plane", ("step", 2), "abort"],
        )

    def test_final_report_failure_aborts_closed_batch_and_propagates(self) -> None:
        solid = _FakeSolid(fail_end=True)

        with self.assertRaisesRegex(RuntimeError, "batch end failed"):
            solid_mpm_fsi_runner._advance_solid_substeps_batched(
                solid,
                _config(enforce_plane_strain_x=False),
                solid_substeps=1,
                solid_substep_dt_s=1.0e-6,
                mu_pa=2.0,
                lambda_pa=3.0,
                solid_substep_velocity_damping=1.0,
            )

        self.assertEqual(
            solid.events,
            ["begin", ("step", 1), "end", "abort"],
        )


if __name__ == "__main__":
    unittest.main()
