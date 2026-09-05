from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from cases.turek_hron_fsi import (
    TurekHronFsiConfig,
    _validate_turek_hron_physical_config,
    fluid_cell_spacing_m,
    run_turek_hron_fsi,
)
from simulation_core.materials import NeoHookeanMaterial


def _config(**overrides: object) -> TurekHronFsiConfig:
    return replace(TurekHronFsiConfig(), **overrides)


def _default_stable_solid_dt_s(config: TurekHronFsiConfig) -> float:
    """Independent public-material reference for the default 3D material."""

    material = NeoHookeanMaterial.from_youngs_modulus(
        name="turek-hron-test",
        density_kgm3=config.solid_density_kgm3,
        youngs_modulus_pa=config.young_modulus_pa,
        poissons_ratio=config.poisson_ratio,
    )
    dx, dy, dz = fluid_cell_spacing_m(config)
    active_spacings = (dy, dz) if config.enforce_plane_strain_x else (dx, dy, dz)
    return material.stable_explicit_dt_s(min(active_spacings))


class TurekHronPhysicalConfigGuardTests(unittest.TestCase):
    def test_default_config_passes(self) -> None:
        _validate_turek_hron_physical_config(TurekHronFsiConfig())

    def test_rejects_nonfinite_or_nonpositive_macro_dt(self) -> None:
        for bad in (0.0, -1.0e-3, float("nan"), float("inf")):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "dt_s"):
                    _validate_turek_hron_physical_config(_config(dt_s=bad))

    def test_rejects_non_integral_or_boolean_step_controls(self) -> None:
        for field_name in (
            "step_count",
            "solid_substeps",
            "flow_predictor_substeps",
        ):
            for bad in (True, 1.5, 0, -1):
                with self.subTest(field=field_name, value=bad):
                    with self.assertRaisesRegex(ValueError, field_name):
                        _validate_turek_hron_physical_config(
                            _config(**{field_name: bad})
                        )

    def test_rejects_malformed_grid_and_domain_lengths(self) -> None:
        for overrides, expected in (
            ({"grid_nodes": (4, 48)}, "grid_nodes"),
            ({"grid_nodes": (4, 48.0, 288)}, "grid_nodes"),
            ({"grid_nodes": (4, True, 288)}, "grid_nodes"),
            ({"grid_nodes": (4, 3, 288)}, "grid_nodes"),
            ({"span_m": 0.0}, "span_m"),
            ({"channel_height_m": float("nan")}, "channel_height_m"),
            ({"channel_length_m": -2.5}, "channel_length_m"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_turek_hron_physical_config(_config(**overrides))

    def test_rejects_invalid_solid_material(self) -> None:
        for field_name, bad in (
            ("solid_density_kgm3", 0.0),
            ("young_modulus_pa", float("inf")),
            ("poisson_ratio", -0.01),
            ("poisson_ratio", 0.5),
        ):
            with self.subTest(field=field_name, value=bad):
                with self.assertRaisesRegex(ValueError, field_name):
                    _validate_turek_hron_physical_config(
                        _config(**{field_name: bad})
                    )

    def test_substep_at_public_material_limit_is_accepted(self) -> None:
        config = TurekHronFsiConfig()
        limit_s = _default_stable_solid_dt_s(config)
        boundary = replace(
            config,
            dt_s=limit_s * config.solid_substeps,
        )
        self.assertEqual(
            boundary.dt_s / boundary.solid_substeps,
            limit_s,
        )
        _validate_turek_hron_physical_config(boundary)

    def test_substep_above_public_material_limit_is_rejected(self) -> None:
        config = TurekHronFsiConfig()
        limit_s = _default_stable_solid_dt_s(config)
        unstable = replace(
            config,
            dt_s=limit_s * (1.0 + 1.0e-12) * config.solid_substeps,
        )
        with self.assertRaisesRegex(
            ValueError,
            "dt_sub_s=.*stable_limit_s=.*required_min_substeps",
        ):
            _validate_turek_hron_physical_config(unstable)

    def test_unstable_l1_fails_before_fluid_or_solid_builder(self) -> None:
        config = _config(
            grid_nodes=(4, 96, 576),
            markers_per_side="auto",
            markers_per_tip="auto",
            ib_anisotropic_envelope=True,
        )
        fluid_builder = Mock(side_effect=AssertionError("fluid builder called"))
        solid_builder = Mock(side_effect=AssertionError("solid builder called"))
        with patch(
            "cases.turek_hron_fsi._build_fluid", fluid_builder
        ), patch("cases.turek_hron_fsi._build_solid", solid_builder):
            with self.assertRaisesRegex(
                ValueError,
                "dt_sub_s=.*stable_limit_s=.*required_min_substeps",
            ):
                run_turek_hron_fsi(config)
        fluid_builder.assert_not_called()
        solid_builder.assert_not_called()

    def test_s0_and_corrected_l1_reach_solid_builder_without_rewriting_config(self) -> None:
        l1 = _config(
            grid_nodes=(4, 96, 576),
            markers_per_side="auto",
            markers_per_tip="auto",
            ib_anisotropic_envelope=True,
        )
        configurations = (
            TurekHronFsiConfig(),
            l1,
            replace(l1, grid_nodes=(4, 144, 864)),
        )
        for candidate in configurations:
            config = replace(
                candidate,
                solid_substeps=math.ceil(
                    candidate.dt_s / _default_stable_solid_dt_s(candidate)
                ),
            )
            with self.subTest(grid_nodes=config.grid_nodes):
                fluid_builder = Mock(return_value=object())
                solid_builder = Mock(
                    side_effect=RuntimeError("solid builder sentinel")
                )
                with patch(
                    "cases.turek_hron_fsi._build_fluid", fluid_builder
                ), patch("cases.turek_hron_fsi._build_solid", solid_builder):
                    with self.assertRaisesRegex(
                        RuntimeError, "solid builder sentinel"
                    ):
                        run_turek_hron_fsi(config)
                received_config = fluid_builder.call_args.args[0]
                self.assertEqual(
                    received_config.solid_substeps, config.solid_substeps
                )
                self.assertEqual(received_config.dt_s, config.dt_s)
                solid_builder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
