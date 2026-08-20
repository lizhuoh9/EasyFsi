from __future__ import annotations

import csv
import contextlib
import inspect
import io
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from cases.squid_soft_robot import (
    checkpointing,
    diagnostics,
    history,
    runner,
    source_config,
)
from cases.squid_soft_robot.cli import parse_args
from cases.squid_soft_robot.step_loop import (
    run_squid_step_loop,
)


@dataclass(frozen=True)
class _MinimalCheckpointSpec:
    source_config_path: str


class SquidSharpCliContractTests(unittest.TestCase):
    def test_safe_defaults_preserve_existing_walk_lengths_and_enable_air_backing(
        self,
    ) -> None:
        args = parse_args([])

        self.assertEqual(args.fixed_rim_region_id, 5)
        self.assertTrue(args.far_pressure_air_backed)
        self.assertEqual(args.far_pressure_inside_probe_max_multiplier, 12.0)
        self.assertEqual(args.two_sided_probe_max_multiplier, 12.0)
        self.assertEqual(args.one_sided_probe_max_multiplier, 12.0)
        self.assertEqual(args.far_pressure_air_backed_probe_normal_sign, 0.0)
        self.assertIsNone(args.neo_fixed_node_lock_policy)

    def test_air_backing_and_probe_controls_have_cli_escape_hatches(self) -> None:
        args = parse_args(
            [
                "--no-far-pressure-air-backed",
                "--far-pressure-inside-probe-max-multiplier",
                "9",
                "--two-sided-probe-max-multiplier",
                "10",
                "--one-sided-probe-max-multiplier",
                "11",
                "--far-pressure-air-backed-probe-normal-sign",
                "-1",
                "--fixed-rim-region-id",
                "6",
                "--neo-fixed-node-lock-policy",
                "any_fixed_particle",
            ]
        )

        self.assertFalse(args.far_pressure_air_backed)
        self.assertEqual(args.far_pressure_inside_probe_max_multiplier, 9.0)
        self.assertEqual(args.two_sided_probe_max_multiplier, 10.0)
        self.assertEqual(args.one_sided_probe_max_multiplier, 11.0)
        self.assertEqual(args.far_pressure_air_backed_probe_normal_sign, -1.0)
        self.assertEqual(args.fixed_rim_region_id, 6)
        self.assertEqual(args.neo_fixed_node_lock_policy, "any_fixed_particle")

    def test_air_backing_requires_the_zmin_pressure_outlet(self) -> None:
        args = parse_args(["--disable-pressure-outlet-zmin"])

        with self.assertRaisesRegex(ValueError, "air-backed.*z-min pressure outlet"):
            runner.validate_sharp_case_cli_contract(args)

        compatible = parse_args(
            ["--disable-pressure-outlet-zmin", "--no-far-pressure-air-backed"]
        )
        runner.validate_sharp_case_cli_contract(compatible)

    def test_coupling_mode_selector_is_not_a_cli_option(self) -> None:
        for removed_value in ("hibm_mpm_sharp", "legacy_projected_reduced"):
            with self.subTest(removed_value=removed_value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(["--fsi-coupling-mode", removed_value])

    def test_probe_and_seed_sign_options_fail_closed_at_case_startup(self) -> None:
        args = parse_args([])
        args.two_sided_probe_max_multiplier = 2.99
        with self.assertRaisesRegex(ValueError, "two-sided.*>= 3"):
            runner.validate_sharp_case_cli_contract(args)

        args = parse_args([])
        args.far_pressure_air_backed_probe_normal_sign = 0.5
        with self.assertRaisesRegex(ValueError, "normal sign.*-1.0, 0.0, or 1.0"):
            runner.validate_sharp_case_cli_contract(args)

    def test_fixed_rim_must_be_distinct_and_present(self) -> None:
        validator = getattr(source_config, "validate_fixed_rim_region_contract")
        with self.assertRaisesRegex(ValueError, "distinct.*primary"):
            validator(
                fixed_rim_region_id=7,
                primary_region_id=7,
                secondary_region_id=8,
                available_region_ids=(5, 7, 8),
            )
        with self.assertRaisesRegex(ValueError, "matched no faces"):
            validator(
                fixed_rim_region_id=5,
                primary_region_id=7,
                secondary_region_id=8,
                available_region_ids=(7, 8),
            )

    def test_neo_lock_policy_is_explicit_and_mooney_does_not_silently_ignore_it(
        self,
    ) -> None:
        neo_default = parse_args(["--solid-model", "neo_hookean_mpm"])
        self.assertEqual(
            runner.resolve_neo_fixed_node_lock_policy(neo_default),
            "pure_fixed_mass",
        )

        mooney_explicit = parse_args(
            [
                "--solid-model",
                "tri_mooney_shell_mpm",
                "--neo-fixed-node-lock-policy",
                "any_fixed_particle",
            ]
        )
        with self.assertRaisesRegex(ValueError, "Neo-only"):
            runner.resolve_neo_fixed_node_lock_policy(mooney_explicit)

    def test_new_physics_options_participate_in_checkpoint_fingerprint(self) -> None:
        expected = {
            "fixed_rim_region_id",
            "far_pressure_air_backed",
            "far_pressure_inside_probe_max_multiplier",
            "two_sided_probe_max_multiplier",
            "one_sided_probe_max_multiplier",
            "far_pressure_air_backed_probe_normal_sign",
            "neo_fixed_node_lock_policy",
        }
        self.assertTrue(
            expected.issubset(checkpointing.CHECKPOINT_ARG_FINGERPRINT_FIELDS)
        )

    def test_checkpoint_fingerprint_hashes_geometry_cache_and_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            step_path = root / "squid.step"
            cache_path = root / "squid.surface.stl"
            config_path = root / "simulation_config.json"
            step_path.write_bytes(b"step-v1")
            cache_path.write_bytes(b"cache-v1")

            def write_config(face_ids: list[int]) -> None:
                config_path.write_text(
                    json.dumps(
                        {
                            "mesh_path": str(step_path),
                            "surface_mesh_cache_path": str(cache_path),
                            "named_selections": [{"id": 7, "face_ids": face_ids}],
                        }
                    ),
                    encoding="utf-8",
                )

            write_config([1, 2])
            args = parse_args(["--source-config", str(config_path)])
            spec = _MinimalCheckpointSpec(source_config_path=str(config_path))

            baseline = checkpointing.checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=1,
            )
            step_path.write_bytes(b"step-v2")
            changed_step = checkpointing.checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=1,
            )
            step_path.write_bytes(b"step-v1")
            cache_path.write_bytes(b"cache-v2")
            changed_cache = checkpointing.checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=1,
            )
            cache_path.write_bytes(b"cache-v1")
            write_config([1, 3])
            changed_topology = checkpointing.checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=1,
            )

        self.assertNotEqual(baseline, changed_step)
        self.assertNotEqual(baseline, changed_cache)
        self.assertNotEqual(baseline, changed_topology)

    def test_checkpoint_fingerprint_hashes_effective_volume_particle_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "simulation_config.json"
            particle_cache_path = root / "simulation_config.mesh.volume_particles.npz"
            config_path.write_text(
                json.dumps(
                    {
                        "analysis_settings": {
                            "fluid_active_mask_enabled": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            particle_cache_path.write_bytes(b"volume-cache-v1")
            args = parse_args(["--source-config", str(config_path)])
            spec = _MinimalCheckpointSpec(source_config_path=str(config_path))

            baseline = checkpointing.checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=1,
            )
            original_stat = particle_cache_path.stat()
            particle_cache_path.write_bytes(b"volume-cache-v2")
            os.utime(
                particle_cache_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            changed = checkpointing.checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=1,
            )

        self.assertNotEqual(baseline, changed)

    def test_checkpoint_fingerprint_ignores_volume_cache_for_surface_only_mask(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "simulation_config.json"
            particle_cache_path = root / "simulation_config.mesh.volume_particles.npz"
            config_path.write_text(
                json.dumps(
                    {
                        "analysis_settings": {
                            "fluid_active_mask_enabled": True,
                            "solid_obstacle_surface_only_region_ids": [5],
                        }
                    }
                ),
                encoding="utf-8",
            )
            particle_cache_path.write_bytes(b"unused-cache-v1")
            args = parse_args(["--source-config", str(config_path)])
            spec = _MinimalCheckpointSpec(source_config_path=str(config_path))

            baseline = checkpointing.checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=1,
            )
            particle_cache_path.write_bytes(b"unused-cache-v2-expanded")
            changed = checkpointing.checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=1,
            )

        self.assertEqual(baseline, changed)

    def test_checkpoint_fingerprint_includes_post_dirichlet_projection_count(
        self,
    ) -> None:
        self.assertIn(
            "hibm_post_dirichlet_consistency_projections",
            checkpointing.CHECKPOINT_ARG_FINGERPRINT_FIELDS,
        )
        spec = _MinimalCheckpointSpec(source_config_path="source.json")
        default = checkpointing.checkpoint_run_fingerprint(
            args=parse_args([]),
            spec=spec,
            step_count=1,
            full_pressure_waveform_steps=1,
        )
        changed = checkpointing.checkpoint_run_fingerprint(
            args=parse_args(["--hibm-post-dirichlet-consistency-projections", "2"]),
            spec=spec,
            step_count=1,
            full_pressure_waveform_steps=1,
        )

        self.assertNotEqual(default, changed)

    def test_checkpoint_fingerprint_uses_the_effective_neo_lock_policy(self) -> None:
        default_args = parse_args(["--solid-model", "neo_hookean_mpm"])
        explicit_args = parse_args(
            [
                "--solid-model",
                "neo_hookean_mpm",
                "--neo-fixed-node-lock-policy",
                "pure_fixed_mass",
            ]
        )
        spec = _MinimalCheckpointSpec(source_config_path="source.json")

        default_fingerprint = checkpointing.checkpoint_run_fingerprint(
            args=default_args,
            spec=spec,
            step_count=1,
            full_pressure_waveform_steps=1,
        )
        explicit_fingerprint = checkpointing.checkpoint_run_fingerprint(
            args=explicit_args,
            spec=spec,
            step_count=1,
            full_pressure_waveform_steps=1,
        )

        self.assertEqual(default_fingerprint, explicit_fingerprint)

    def test_sharp_step_uses_named_values_instead_of_case_magic_constants(self) -> None:
        source = inspect.getsource(run_squid_step_loop)
        call = source.split("sharp_coupling_state.advance_mpm_step(", 1)[1].split(
            "fluid_dt_s=", 1
        )[0]

        self.assertIn(
            "far_pressure_barrier_region_id=settings.fixed_rim_region_id",
            call,
        )
        self.assertIn("far_pressure_air_backed=settings.far_pressure_air_backed", call)
        for setting_name in (
            "far_pressure_air_backed_probe_normal_sign",
            "far_pressure_inside_probe_max_multiplier",
            "two_sided_probe_max_multiplier",
            "one_sided_probe_max_multiplier",
        ):
            self.assertIn(f"settings.{setting_name}", call)
        self.assertNotIn(
            "pressure_far_side_normal_sign",
            call.split("far_pressure_air_backed_probe", 1)[1],
        )

    def test_neo_step_receives_the_resolved_lock_policy(self) -> None:
        runner_source = inspect.getsource(runner.run)
        step_loop_source = inspect.getsource(run_squid_step_loop)
        self.assertIn(
            "fixed_node_lock_policy=neo_fixed_node_lock_policy",
            runner_source,
        )
        self.assertIn(
            "fixed_node_lock_policy=settings.neo_fixed_node_lock_policy",
            step_loop_source,
        )


class SquidClosureGuardContractTests(unittest.TestCase):
    def test_missing_coverage_field_fails_closed_once_patience_window_exists(
        self,
    ) -> None:
        field = "hibm_full_stress_far_pressure_closed_marker_count"
        rows = [
            {"step": 1, field: 5},
            {"step": 2},
            {"step": 3, field: 4},
        ]

        with self.assertRaisesRegex(KeyError, field):
            diagnostics._raise_for_closure_coverage_floor(rows, floor=7, patience=3)


class SquidCsvAtomicWriteContractTests(unittest.TestCase):
    def test_replace_failure_uses_unique_same_directory_temps_and_cleans_them(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            target = directory / "history.csv"
            attempted_sources: list[Path] = []

            def fail_replace(source, destination) -> None:
                attempted_sources.append(Path(source))
                self.assertEqual(Path(source).parent, directory)
                self.assertEqual(Path(destination), target)
                raise PermissionError("locked")

            with (
                mock.patch.object(history.os, "replace", side_effect=fail_replace),
                mock.patch.object(history.time, "sleep"),
            ):
                for _ in range(2):
                    with self.assertRaises(PermissionError):
                        history.write_csv(target, [{"step": 1}])

            unique_sources = {path for path in attempted_sources}
            self.assertEqual(len(unique_sources), 2)
            self.assertTrue(all(not path.exists() for path in unique_sources))

    def test_csv_serialization_failure_cleans_unique_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            target = directory / "history.csv"
            with mock.patch.object(
                csv.DictWriter,
                "writerows",
                side_effect=RuntimeError("serialize failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "serialize failed"):
                    history.write_csv(target, [{"step": 1}])

            self.assertEqual(list(directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
