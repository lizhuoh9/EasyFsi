import inspect
import math
import unittest
from dataclasses import replace

import numpy as np

from cases import CASE_MODULES
from cases.turek_hron_fsi import (
    TUREK_HRON_CASE_ID,
    TUREK_HRON_CASE_METADATA,
    TUREK_HRON_HISTORY_FLUSH_INTERVAL_STEPS,
    TUREK_HRON_PRESET_PARAMETERS,
    TUREK_HRON_REFERENCE_RESULTS,
    TUREK_HRON_WALL_BOUNDARY_MODEL,
    TurekHronFsiConfig,
    beam_box_solver_m,
    beam_fixed_particle_mask,
    beam_root_x_m,
    beam_surface_force_support_radius_m,
    build_cylinder_obstacle_mask,
    build_marker_layout,
    build_turek_hron_final_fields_snapshot,
    _flush_history_csv,
    _write_final_fields_contour_png,
    cylinder_center_solver_m,
    fluid_cell_spacing_m,
    fsi1_config,
    fsi2_config,
    fsi3_config,
    inlet_profile_mps,
    inlet_ramp_factor,
    resolved_marker_counts,
    run_turek_hron_fsi,
    solver_z_from_turek_hron_x_m,
    thin_beam_pressure_probe_max_multiplier,
    with_beam_surface_force_support,
    _reseed_turek_hron_markers,
    _resample_marker_group_arrays,
)
from benchmarks.official.solid_mpm_fsi_runner import SECONDARY_UNUSED_REGION_ID
from simulation_core.coupling.hibm_mpm import advance_hibm_mpm_sharp_mpm_step


class TurekHronConfigPresetTests(unittest.TestCase):
    def test_shared_canonical_geometry_defaults(self):
        for config in (fsi1_config(), fsi2_config(), fsi3_config()):
            self.assertAlmostEqual(config.channel_length_m, 2.5)
            self.assertAlmostEqual(config.channel_height_m, 0.41)
            self.assertAlmostEqual(config.cylinder_center_x_m, 0.2)
            self.assertAlmostEqual(config.cylinder_center_y_m, 0.2)
            self.assertAlmostEqual(config.cylinder_radius_m, 0.05)
            self.assertAlmostEqual(config.beam_length_m, 0.35)
            self.assertAlmostEqual(config.beam_thickness_m, 0.02)
            self.assertAlmostEqual(config.beam_tip_x_m, 0.6)
            self.assertAlmostEqual(config.inlet_ramp_time_s, 2.0)
            self.assertAlmostEqual(config.fluid_density_kgm3, 1000.0)
            self.assertAlmostEqual(config.fluid_viscosity_pa_s, 1.0)
            self.assertAlmostEqual(config.span_m, 0.05)
            self.assertEqual(config.grid_nodes, (4, 48, 288))

    def test_fsi1_preset_matches_spec_table(self):
        config = fsi1_config()
        self.assertAlmostEqual(config.mean_inlet_velocity_mps, 0.2)
        self.assertAlmostEqual(config.solid_density_kgm3, 1000.0)
        self.assertAlmostEqual(config.young_modulus_pa, 1.4e6)
        self.assertAlmostEqual(config.poisson_ratio, 0.4)
        self.assertAlmostEqual(config.dt_s, 5.0e-3)

    def test_fsi2_preset_matches_spec_table(self):
        config = fsi2_config()
        self.assertAlmostEqual(config.mean_inlet_velocity_mps, 1.0)
        self.assertAlmostEqual(config.solid_density_kgm3, 10000.0)
        self.assertAlmostEqual(config.young_modulus_pa, 1.4e6)
        self.assertAlmostEqual(config.poisson_ratio, 0.4)
        self.assertAlmostEqual(config.dt_s, 1.0e-3)

    def test_fsi3_preset_matches_spec_table(self):
        config = fsi3_config()
        self.assertAlmostEqual(config.mean_inlet_velocity_mps, 2.0)
        self.assertAlmostEqual(config.solid_density_kgm3, 1000.0)
        self.assertAlmostEqual(config.young_modulus_pa, 5.6e6)
        self.assertAlmostEqual(config.poisson_ratio, 0.4)
        self.assertAlmostEqual(config.dt_s, 1.0e-3)

    def test_preset_parameter_table_pins_spec_values(self):
        table = TUREK_HRON_PRESET_PARAMETERS
        self.assertEqual(set(table), {"fsi1", "fsi2", "fsi3"})
        self.assertAlmostEqual(table["fsi1"]["mean_inlet_velocity_mps"], 0.2)
        self.assertAlmostEqual(table["fsi2"]["mean_inlet_velocity_mps"], 1.0)
        self.assertAlmostEqual(table["fsi3"]["mean_inlet_velocity_mps"], 2.0)
        self.assertAlmostEqual(table["fsi2"]["solid_density_kgm3"], 10000.0)
        self.assertAlmostEqual(table["fsi3"]["young_modulus_pa"], 5.6e6)
        for preset in table.values():
            self.assertAlmostEqual(preset["poisson_ratio"], 0.4)

    def test_reference_results_pin_canonical_fsi1_values(self):
        reference = TUREK_HRON_REFERENCE_RESULTS["fsi1"]
        self.assertAlmostEqual(reference["ux_a_m"], 2.27e-5)
        self.assertAlmostEqual(reference["uy_a_m"], 8.209e-4)
        self.assertAlmostEqual(reference["drag_n_per_m"], 14.295)
        self.assertAlmostEqual(reference["lift_n_per_m"], 0.7638)
        self.assertEqual(reference["regime"], "steady")

    def test_reference_results_pin_ls_dyna_fsi3_ranges(self):
        reference = TUREK_HRON_REFERENCE_RESULTS["fsi3"]
        self.assertAlmostEqual(reference["ux_a_mean_m"], -2.35e-3)
        self.assertAlmostEqual(reference["ux_a_amplitude_m"], 2.45e-3)
        self.assertAlmostEqual(reference["uy_a_mean_m"], 1.5e-3)
        self.assertAlmostEqual(reference["uy_a_amplitude_m"], 33.5e-3)
        self.assertAlmostEqual(reference["drag_mean_n_per_m"], 452.0)
        self.assertAlmostEqual(reference["drag_amplitude_n_per_m"], 31.0)
        self.assertAlmostEqual(reference["lift_mean_n_per_m"], 3.3)
        self.assertAlmostEqual(reference["lift_amplitude_n_per_m"], 83.1)

    def test_case_metadata_pins_geometry_and_wall_model(self):
        geometry = TUREK_HRON_CASE_METADATA["geometry"]
        self.assertAlmostEqual(geometry["channel_length_m"], 2.5)
        self.assertAlmostEqual(geometry["channel_height_m"], 0.41)
        self.assertAlmostEqual(geometry["cylinder_radius_m"], 0.05)
        self.assertAlmostEqual(geometry["beam_tip_x_m"], 0.6)
        self.assertEqual(
            TUREK_HRON_CASE_METADATA["wall_boundary_model"],
            TUREK_HRON_WALL_BOUNDARY_MODEL,
        )
        self.assertEqual(
            TUREK_HRON_CASE_METADATA["reference_results"],
            TUREK_HRON_REFERENCE_RESULTS,
        )


class TurekHronCoordinateConversionTests(unittest.TestCase):
    def test_solver_z_from_turek_hron_x(self):
        config = TurekHronFsiConfig()
        self.assertAlmostEqual(solver_z_from_turek_hron_x_m(0.6, config), 1.9)
        self.assertAlmostEqual(solver_z_from_turek_hron_x_m(0.0, config), 2.5)
        self.assertAlmostEqual(solver_z_from_turek_hron_x_m(0.25, config), 2.25)

    def test_cylinder_center_maps_to_solver_coordinates(self):
        config = TurekHronFsiConfig()
        center_y, center_z = cylinder_center_solver_m(config)
        self.assertAlmostEqual(center_y, 0.2)
        self.assertAlmostEqual(center_z, 2.3)

    def test_beam_box_spans_root_to_tip_in_solver_z(self):
        config = TurekHronFsiConfig()
        self.assertAlmostEqual(beam_root_x_m(config), 0.25)
        box_min, box_max = beam_box_solver_m(config)
        self.assertAlmostEqual(box_min[1], 0.19)
        self.assertAlmostEqual(box_max[1], 0.21)
        self.assertAlmostEqual(box_min[2], 1.9)
        self.assertAlmostEqual(box_max[2], 2.25)
        self.assertAlmostEqual(box_min[0], 0.0)
        self.assertAlmostEqual(box_max[0], config.span_m)


class TurekHronInletProfileTests(unittest.TestCase):
    def test_parabolic_profile_peak_and_walls(self):
        config = fsi1_config()
        height_m = config.channel_height_m
        t_steady_s = config.inlet_ramp_time_s + 1.0
        self.assertAlmostEqual(
            inlet_profile_mps(0.5 * height_m, t_steady_s, config),
            1.5 * config.mean_inlet_velocity_mps,
        )
        self.assertAlmostEqual(inlet_profile_mps(0.0, t_steady_s, config), 0.0)
        self.assertAlmostEqual(inlet_profile_mps(height_m, t_steady_s, config), 0.0)

    def test_cosine_ramp_endpoints_and_midpoint(self):
        config = fsi1_config()
        self.assertAlmostEqual(inlet_ramp_factor(0.0, config), 0.0)
        self.assertAlmostEqual(inlet_ramp_factor(config.inlet_ramp_time_s, config), 1.0)
        self.assertAlmostEqual(inlet_ramp_factor(10.0, config), 1.0)
        self.assertAlmostEqual(
            inlet_ramp_factor(0.5 * config.inlet_ramp_time_s, config),
            0.5 * (1.0 - math.cos(0.5 * math.pi)),
        )

    def test_profile_mean_equals_mean_inlet_velocity(self):
        config = fsi1_config()
        height_m = config.channel_height_m
        y = np.linspace(0.0, height_m, 20001)
        profile = np.array(
            [inlet_profile_mps(value, config.inlet_ramp_time_s, config) for value in y]
        )
        mean_mps = float(np.trapezoid(profile, y) / height_m)
        self.assertAlmostEqual(mean_mps, config.mean_inlet_velocity_mps, places=6)


class TurekHronGeometryBuilderTests(unittest.TestCase):
    def test_cylinder_mask_cell_count_matches_disc_area(self):
        config = replace(TurekHronFsiConfig(), grid_nodes=(2, 82, 500))
        mask = build_cylinder_obstacle_mask(config)
        self.assertEqual(mask.shape, config.grid_nodes)
        self.assertEqual(mask.dtype, np.int32)
        np.testing.assert_array_equal(mask[0], mask[1])
        _, dy, dz = fluid_cell_spacing_m(config)
        expected_cells = math.pi * config.cylinder_radius_m**2 / (dy * dz)
        actual_cells = int(mask[0].sum())
        self.assertLess(
            abs(actual_cells - expected_cells), 0.15 * expected_cells
        )

    def test_cylinder_mask_cells_stay_inside_radius(self):
        config = replace(TurekHronFsiConfig(), grid_nodes=(2, 82, 500))
        mask = build_cylinder_obstacle_mask(config)
        _, dy, dz = fluid_cell_spacing_m(config)
        center_y, center_z = cylinder_center_solver_m(config)
        j_idx, k_idx = np.nonzero(mask[0])
        y = (j_idx + 0.5) * dy
        z = (k_idx + 0.5) * dz
        distance = np.sqrt((y - center_y) ** 2 + (z - center_z) ** 2)
        self.assertLessEqual(float(distance.max()), config.cylinder_radius_m)
        cell_diagonal = math.hypot(dy, dz)
        self.assertGreater(
            float(distance.max()), config.cylinder_radius_m - cell_diagonal
        )

    def test_beam_fixed_mask_fixes_root_band_and_cylinder_overlap(self):
        config = replace(
            TurekHronFsiConfig(), solid_particle_counts=(1, 4, 70)
        )
        box_min, box_max = beam_box_solver_m(config)
        particle_dz = config.beam_length_m / config.solid_particle_counts[2]
        rest = np.array(
            [
                [0.025, 0.2, box_min[2] + 0.5 * particle_dz],
                [0.025, 0.2, 0.5 * (box_min[2] + box_max[2])],
                [0.025, 0.2, box_max[2] - 0.5 * particle_dz],
                [0.025, 0.2, 2.3],
            ]
        )
        fixed = beam_fixed_particle_mask(rest, config)
        self.assertFalse(fixed[0])
        self.assertFalse(fixed[1])
        self.assertTrue(fixed[2])
        self.assertTrue(fixed[3])

    def test_marker_layout_covers_long_faces_and_tip(self):
        config = TurekHronFsiConfig()
        positions, normals, areas = build_marker_layout(config)
        expected_count = 2 * config.markers_per_side + config.markers_per_tip
        self.assertEqual(len(positions), expected_count)
        self.assertEqual(len(normals), expected_count)
        self.assertEqual(len(areas), expected_count)
        box_min, box_max = beam_box_solver_m(config)
        _, dy, dz = fluid_cell_spacing_m(config)
        per_side = config.markers_per_side
        lower_face = positions[:per_side]
        upper_face = positions[per_side : 2 * per_side]
        tip_face = positions[2 * per_side :]
        for position in lower_face:
            self.assertAlmostEqual(position[1], box_min[1] - 0.51 * dy)
        for position in upper_face:
            self.assertAlmostEqual(position[1], box_max[1] + 0.51 * dy)
        for position in tip_face:
            self.assertAlmostEqual(position[2], box_min[2] - 0.51 * dz)
        for normal in normals[:per_side]:
            self.assertEqual(normal, (0.0, -1.0, 0.0))
        for normal in normals[per_side : 2 * per_side]:
            self.assertEqual(normal, (0.0, 1.0, 0.0))
        for normal in normals[2 * per_side :]:
            self.assertEqual(normal, (0.0, 0.0, -1.0))
        long_area = sum(areas[: 2 * per_side])
        self.assertAlmostEqual(long_area, 2.0 * config.span_m * config.beam_length_m)
        tip_area = sum(areas[2 * per_side :])
        self.assertAlmostEqual(tip_area, config.span_m * config.beam_thickness_m)

    def test_support_radius_is_thickness_limited_and_positive(self):
        config = TurekHronFsiConfig()
        radius_m = beam_surface_force_support_radius_m(config)
        self.assertGreater(radius_m, 0.0)
        self.assertLessEqual(radius_m, 0.5 * config.beam_thickness_m + 1.0e-12)
        supported = with_beam_surface_force_support(config)
        self.assertAlmostEqual(supported.mpm_support_radius_m, radius_m)
        explicit = with_beam_surface_force_support(
            replace(config, mpm_support_radius_m=0.004)
        )
        self.assertAlmostEqual(explicit.mpm_support_radius_m, 0.004)


class TurekHronSolverContractTests(unittest.TestCase):
    def test_case_module_registered(self):
        self.assertEqual(CASE_MODULES["turek-hron-fsi"], "cases.turek_hron_fsi")
        self.assertEqual(TUREK_HRON_CASE_ID, "turek-hron-fsi")

    def test_advance_step_exposes_velocity_inlet_zmax_contract(self):
        parameters = inspect.signature(advance_hibm_mpm_sharp_mpm_step).parameters
        self.assertIn("velocity_inlet_zmax", parameters)
        self.assertIs(parameters["velocity_inlet_zmax"].default, False)

    def test_run_loop_uses_velocity_inlet_and_pressure_outlet(self):
        source = inspect.getsource(run_turek_hron_fsi)
        self.assertIn("velocity_inlet_zmax=True", source)
        self.assertIn("pressure_outlet_zmin=True", source)
        self.assertIn("_write_channel_boundary_rows(fluid, config, t_s)", source)
        self.assertIn(
            "fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)",
            source,
        )


class TurekHronPressureTopologyGuardTests(unittest.TestCase):
    """D1-D4: guards against the discrete pressure-topology flip at t~0.735.

    See run_turek_hron_fsi's search_radius_m / interior_probe_distance_m
    derivation and thin_beam_pressure_probe_max_multiplier for the fix.
    """

    def test_plane_spacing_envelope_stays_below_beam_thickness(self):
        # D1: search_radius/interior_probe must be derived from the
        # wall-normal/streamwise plane spacings (dy, dz) only - NOT
        # max(dx, dy, dz), which pulls in the physically irrelevant span
        # spacing dx and, at the default grid, exceeds the beam thickness.
        config = TurekHronFsiConfig()
        dx, dy, dz = fluid_cell_spacing_m(config)
        plane_spacing_m = max(dy, dz)
        search_radius_m = 1.5 * plane_spacing_m
        interior_probe_distance_m = 1.0 * plane_spacing_m

        # Exact values at the default grid (dy=0.41/48, dz=2.5/288).
        self.assertAlmostEqual(dx, 0.0125)
        self.assertAlmostEqual(plane_spacing_m, 2.5 / 288.0)
        self.assertAlmostEqual(search_radius_m, 0.013020833333333334)
        self.assertAlmostEqual(interior_probe_distance_m, 0.008680555555555556)

        # Invariants: radius must stay under the full beam thickness, probe
        # must stay under the half thickness, so the extended two-sided
        # probe never reaches the opposite wetted face by construction.
        self.assertLess(search_radius_m, config.beam_thickness_m)
        self.assertLess(interior_probe_distance_m, 0.5 * config.beam_thickness_m)

        # The old, defective derivation used the span spacing dx and
        # exceeded the beam thickness - pin that this is no longer the
        # active derivation by asserting the old values are in fact unsafe.
        old_max_spacing_m = max(dx, dy, dz)
        self.assertAlmostEqual(old_max_spacing_m, dx)
        self.assertGreaterEqual(3.0 * old_max_spacing_m, config.beam_thickness_m)
        self.assertGreaterEqual(2.0 * old_max_spacing_m, 0.5 * config.beam_thickness_m)

    def test_run_loop_derives_probe_envelope_from_plane_spacing_only(self):
        source = inspect.getsource(run_turek_hron_fsi)
        self.assertNotIn("max(fluid_cell_spacing_m(config))", source)
        self.assertIn("plane_spacing_m = max(plane_dy_m, plane_dz_m)", source)
        self.assertIn("search_radius_m = 1.5 * plane_spacing_m", source)
        self.assertIn("interior_probe_distance_m = 1.0 * plane_spacing_m", source)
        self.assertIn("search_radius_m=search_radius_m", source)
        self.assertIn(
            "interior_probe_distance_m=interior_probe_distance_m", source
        )

    def test_run_loop_wires_far_pressure_region_id_as_barrier_sentinel(self):
        # D2: the sampling_obstacle_field guard in core.py is armed by
        # far_pressure_region_id != -1, but the beam markers must NOT match
        # that id (else every beam marker flips into "closure" sampling and
        # silently substitutes far_pressure_pa on a missed side). The case
        # must use SECONDARY_UNUSED_REGION_ID (owned by no marker) as a
        # barrier-only sentinel, not PRIMARY_REGION_ID (owned by every beam
        # marker).
        source = inspect.getsource(run_turek_hron_fsi)
        self.assertIn(
            "far_pressure_region_id=SECONDARY_UNUSED_REGION_ID", source
        )
        self.assertNotIn("far_pressure_region_id=PRIMARY_REGION_ID", source)

    def test_two_sided_probe_max_multiplier_uses_code_default_floor(self):
        # D3: base_multiplier must fall back to the code default (3.0), not
        # the old hardcoded 12.0 compensation floor. At the default grid the
        # thickness-driven term still dominates, recomputed from the new,
        # smaller D1 envelope.
        config = TurekHronFsiConfig()
        multiplier = thin_beam_pressure_probe_max_multiplier(config)
        self.assertAlmostEqual(multiplier, 6.365853658536586)
        self.assertGreater(multiplier, 3.0)
        self.assertLess(multiplier, 12.0)

        source = inspect.getsource(thin_beam_pressure_probe_max_multiplier)
        self.assertIn("base_multiplier = 3.0", source)
        self.assertNotIn("base_multiplier = 12.0", source)
        self.assertIn("1.5 * plane_spacing_m", source)

    def test_tip_markers_are_documented_as_permanently_invalid(self):
        # D4: the markers_per_tip tip markers face -z into the beam body
        # and are structurally unsampleable; this was expected (not a bug)
        # under the old two-sided sampling mode, which needed both an
        # inside AND an outside probe hit.
        source = inspect.getsource(build_marker_layout)
        self.assertIn("structurally unsampleable", source)
        self.assertIn("stress_invalid_marker_count == 4", source)

        # D5 update: one-sided per-face sampling only needs the outside
        # (fluid-side) probe, so the tip markers are now documented as
        # expected to be VALID (invalid count 4 -> 0), not permanently
        # invalid.
        self.assertIn("now expected to be VALID", source)
        self.assertIn("drops from 4 to 0", source)


class TurekHronOneSidedPressureSamplingTests(unittest.TestCase):
    """D5: switch from two-sided pressure-jump to one-sided per-face
    sampling, mirroring the ANSYS vertical-flap precedent
    (cases/ansys_vertical_flap_fsi.py's selected_formulation_solver_config).

    Root cause: every beam marker shares PRIMARY_REGION_ID and was sampled
    two-sided (traction = (p_inside - p_outside) * normal), where the
    "inside" probe deliberately crosses the 0.02 m-thin beam. Summing both
    long faces' forces at full area double-counts the net beam force to
    ~2 * dp * A. One-sided per-face sampling uses only the fluid-side probe
    along each marker's own outward normal (traction =
    (p_ref - p_outside) * normal, gauge p_ref=0), giving the true net
    dp * A instead of 2x.
    """

    def test_run_loop_wires_one_sided_per_face_sampling_kwargs(self):
        source = inspect.getsource(run_turek_hron_fsi)
        # All beam markers already carry outward-into-fluid normals (lower
        # face (0,-1,0), upper face (0,1,0), tip face (0,0,-1)), and
        # fluid_side_normal_sign is relative to each marker's own stored
        # normal - not absolute space - so a single shared region with
        # sign=+1.0 is correct for every face simultaneously. No
        # marker-layout/region-id split is required (unlike the flap, whose
        # two regions exist for anchor-pair bookkeeping between two
        # physically distinct faces, not because the sign differs).
        self.assertIn(
            "one_sided_pressure_primary_region_id=PRIMARY_REGION_ID", source
        )
        self.assertIn("one_sided_primary_fluid_side_normal_sign=1.0", source)
        # The legacy single-slot one_sided_pressure_region_id and the
        # per-face primary/secondary slots are mutually exclusive at the
        # sample_fluid_stress_to_marker_tractions call (core.py raises if
        # both are set); this case must use only the per-face primary slot.
        self.assertNotIn("one_sided_pressure_secondary_region_id=PRIMARY_REGION_ID", source)

        # Per-face one-sided sampling only exists on the pressure-only fast
        # path (viscosity_pa_s == 0.0). stress_viscosity_pa_s_override
        # decouples the marker-traction sampling viscosity from the real
        # fluid viscosity (1.0 Pa*s for fsi1/fsi3, 1.0 Pa*s for fsi2 too)
        # so this call can reach that fast path without changing the
        # fluid's own physical viscosity used elsewhere (predictor,
        # projection).
        self.assertIn("stress_viscosity_pa_s_override=0.0", source)

        # D2's far_pressure_region_id barrier sentinel wiring must remain
        # untouched (core.py's assemble_hibm_mpm_sharp_fluid_to_mpm_loads
        # now auto-skips that machinery for per-face one-sided callers, so
        # this argument is harmless dead wiring for this call but the D2
        # guard test above still must see it).
        self.assertIn(
            "far_pressure_region_id=SECONDARY_UNUSED_REGION_ID", source
        )

    def test_advance_step_exposes_per_face_one_sided_contract(self):
        parameters = inspect.signature(advance_hibm_mpm_sharp_mpm_step).parameters
        for name in (
            "one_sided_pressure_primary_region_id",
            "one_sided_pressure_secondary_region_id",
            "one_sided_primary_reference_pressure_pa",
            "one_sided_secondary_reference_pressure_pa",
            "one_sided_primary_fluid_side_normal_sign",
            "one_sided_secondary_fluid_side_normal_sign",
            "stress_viscosity_pa_s_override",
        ):
            self.assertIn(name, parameters)
        self.assertEqual(
            parameters["one_sided_pressure_primary_region_id"].default, -1
        )
        self.assertEqual(
            parameters["one_sided_pressure_secondary_region_id"].default, -1
        )
        self.assertIsNone(parameters["stress_viscosity_pa_s_override"].default)

    def test_marker_layout_normals_all_point_outward_for_shared_sign(self):
        # Precondition for using a single shared per-face region: every
        # marker's stored normal must already point away from the beam body
        # (outward into the fluid), so fluid_side_normal_sign=+1.0 samples
        # the correct (outside) probe direction for all three faces.
        config = TurekHronFsiConfig()
        positions, normals, _ = build_marker_layout(config)
        box_min, box_max = beam_box_solver_m(config)
        center_y = 0.5 * (box_min[1] + box_max[1])
        center_z = 0.5 * (box_min[2] + box_max[2])
        for position, normal in zip(positions, normals):
            outward_point = (
                position[0],
                position[1] + normal[1] * 1.0e-3,
                position[2] + normal[2] * 1.0e-3,
            )
            dy_before = abs(position[1] - center_y)
            dz_before = abs(position[2] - center_z)
            dy_after = abs(outward_point[1] - center_y)
            dz_after = abs(outward_point[2] - center_z)
            # Moving along the marker's own +normal must move it further
            # from the beam's central axis in at least the dominant
            # component of that normal (i.e. away from the body).
            if normal[1] != 0.0:
                self.assertGreaterEqual(dy_after, dy_before)
            if normal[2] != 0.0:
                self.assertGreaterEqual(dz_after, dz_before)


class TurekHronDiscreteStateObservabilityTests(unittest.TestCase):
    """Discrete-state observability counters in the per-step history row.

    Diagnosis: the T-H thin-beam case suffered a discrete jump caused by
    EXTERNAL_IB row-set membership flips (velocity-Dirichlet rows
    added/removed as advected markers cross the search-radius distance
    gate). The fix landed separately; this pins that the history row now
    records the counters needed to SEE future flips, sourced from cheap
    scalars already materialized on the advance_hibm_mpm_sharp_mpm_step
    report (report.next_* reflects the row-set state governing the next
    step, i.e. where a flip becomes active).
    """

    def test_history_row_records_discrete_state_counters(self):
        source = inspect.getsource(run_turek_hron_fsi)
        expected_key_to_attribute_path = {
            "hibm_next_external_ib_node_count": (
                "latest_report.next_ib_node_search.external_ib_node_count"
            ),
            "hibm_next_internal_node_count": (
                "latest_report.next_ib_node_search.internal_node_count"
            ),
            "hibm_next_internal_obstacle_cell_count": (
                "latest_report.next_internal_obstacle_cell_count"
            ),
            "hibm_next_velocity_dirichlet_active_rows": (
                "latest_report.next_velocity_dirichlet."
                "active_velocity_dirichlet_rows"
            ),
            "hibm_next_pressure_neumann_active_rows": (
                "latest_report.next_pressure_neumann."
                "active_pressure_neumann_rows"
            ),
            "hibm_next_solid_band_nonprojectable_cell_count": (
                "latest_report.next_solid_band_nonprojectable_cell_count"
            ),
            "hibm_stress_two_sided_extended_marker_count": (
                "load.fluid_stress.two_sided_extended_marker_count"
            ),
        }
        for key, attribute_path in expected_key_to_attribute_path.items():
            self.assertIn(f'"{key}"', source)
            self.assertIn(attribute_path, source)

    def test_history_row_counters_are_cheap_report_scalars_only(self):
        # Guard against regressing to expensive per-step recomputation: the
        # new counters must only wrap existing report attributes in int(),
        # never call .to_numpy() or otherwise pull fresh field data.
        source = inspect.getsource(run_turek_hron_fsi)
        history_row_start = source.index('row: dict[str, Any] = {')
        history_row_end = source.index("history.append(row)")
        history_row_source = source[history_row_start:history_row_end]
        self.assertIn("hibm_next_external_ib_node_count", history_row_source)
        self.assertIn(
            "hibm_stress_two_sided_extended_marker_count", history_row_source
        )
        self.assertNotIn(".to_numpy()", history_row_source)


class TurekHronObservabilityExportTests(unittest.TestCase):
    """Field-snapshot export + incremental history flush (observability parity
    with the vertical-flap case). Source-pin tests only, no GPU."""

    def test_history_flush_interval_default_is_twenty_five(self):
        self.assertEqual(TUREK_HRON_HISTORY_FLUSH_INTERVAL_STEPS, 25)

    def test_run_signature_exposes_export_and_flush_controls(self):
        parameters = inspect.signature(run_turek_hron_fsi).parameters
        self.assertIn("export_final_flow_snapshot", parameters)
        self.assertIs(parameters["export_final_flow_snapshot"].default, True)
        self.assertIn("history_flush_interval_steps", parameters)
        self.assertEqual(
            parameters["history_flush_interval_steps"].default,
            TUREK_HRON_HISTORY_FLUSH_INTERVAL_STEPS,
        )

    def test_snapshot_builder_returns_midspan_slice_and_marker_cloud(self):
        source = inspect.getsource(build_turek_hron_final_fields_snapshot)
        # Reuses the fluid solver's field accessors (run-END to_numpy is fine).
        self.assertIn("fluid.velocity.to_numpy()", source)
        self.assertIn("fluid.pressure.to_numpy()", source)
        self.assertIn("fluid.obstacle.to_numpy()", source)
        # Mid-span y-z slice keys plus the deflected beam marker cloud.
        for key in (
            '"velocity_magnitude_yz_mps"',
            '"pressure_yz_pa"',
            '"obstacle_mask_yz"',
            '"y_centers_m"',
            '"z_centers_m"',
            '"beam_marker_current_xyz_m"',
            '"beam_marker_rest_xyz_m"',
            '"beam_marker_displacement_xyz_m"',
        ):
            self.assertIn(key, source)
        # Marker cloud is read from the solid (deflected + rest positions).
        self.assertIn("solid.x.to_numpy()", source)
        self.assertIn("solid.rest_x.to_numpy()", source)

    def test_run_loop_calls_snapshot_export_and_savez_when_enabled(self):
        source = inspect.getsource(run_turek_hron_fsi)
        self.assertIn("if export_final_flow_snapshot:", source)
        self.assertIn(
            "build_turek_hron_final_fields_snapshot(fluid, solid, config)", source
        )
        self.assertIn('"turek_hron_final_fields.npz"', source)
        self.assertIn("np.savez(npz_path, **snapshot)", source)
        self.assertIn('"turek_hron_final_fields.png"', source)
        self.assertIn("_write_final_fields_contour_png(snapshot, png_path)", source)

    def test_contour_png_overlays_obstacle_and_deflected_markers(self):
        source = inspect.getsource(_write_final_fields_contour_png)
        self.assertIn('matplotlib.use("Agg")', source)
        self.assertIn("contourf", source)
        self.assertIn("beam_marker_current_xyz_m", source)
        self.assertIn("ax.scatter", source)
        # Must degrade gracefully if matplotlib is unavailable (never blocks
        # the .npz export).
        self.assertIn("return False", source)

    def test_run_loop_flushes_history_incrementally(self):
        source = inspect.getsource(run_turek_hron_fsi)
        # Incremental flush wiring: path set up before the loop, appended every
        # flush_interval steps, with a trailing flush for the remainder.
        self.assertIn(
            'incremental_history_path = Path(output_dir) / "turek_hron_fsi_history.csv"',
            source,
        )
        self.assertIn("(step_index + 1) % flush_interval == 0", source)
        self.assertIn("_flush_history_csv(", source)
        self.assertIn("history[last_flushed_index:]", source)

    def test_flush_helper_appends_and_writes_header_once(self):
        source = inspect.getsource(_flush_history_csv)
        # Robustness: append mode after first write, header emitted exactly once.
        self.assertIn('mode = "a" if header_written else "w"', source)
        self.assertIn("if not header_written:", source)
        self.assertIn("writer.writeheader()", source)
        self.assertIn("return True", source)


class TurekHronMarkerReseedConfigTests(unittest.TestCase):
    """Tier-2 marker re-seeding config gate (2026-07-09).

    marker_reseed_interval_steps defaults to None so every existing run of
    this case (including the three canonical presets) is byte-for-byte
    unaffected until a caller opts in.
    """

    def test_marker_reseed_interval_steps_defaults_to_none(self):
        self.assertIsNone(TurekHronFsiConfig().marker_reseed_interval_steps)
        for config in (fsi1_config(), fsi2_config(), fsi3_config()):
            self.assertIsNone(config.marker_reseed_interval_steps)

    def test_field_is_settable_via_replace(self):
        config = replace(TurekHronFsiConfig(), marker_reseed_interval_steps=50)
        self.assertEqual(config.marker_reseed_interval_steps, 50)


class TurekHronMarkerReseedRunLoopWiringTests(unittest.TestCase):
    """Pins the run-loop gate and its position relative to the per-step
    boundary-row write and the strong-coupling base snapshot (source
    inspection only -- no GPU)."""

    def test_run_loop_gates_reseed_call_on_interval(self):
        source = inspect.getsource(run_turek_hron_fsi)
        self.assertIn(
            "config.marker_reseed_interval_steps is not None", source
        )
        self.assertIn("step_index > 0", source)
        self.assertIn(
            "step_index % int(config.marker_reseed_interval_steps) == 0",
            source,
        )
        self.assertIn("_reseed_turek_hron_markers(markers, config)", source)

    def test_reseed_call_precedes_boundary_rows_and_strong_coupling_base(self):
        source = inspect.getsource(run_turek_hron_fsi)
        reseed_index = source.index(
            "_reseed_turek_hron_markers(markers, config)"
        )
        # First occurrence of each: the top-of-step boundary-row write and
        # the strong-coupling per-step base snapshot (fluid.save_state()
        # only appears on the gated strong-coupling path).
        boundary_index = source.index(
            "_write_channel_boundary_rows(fluid, config, t_s)"
        )
        save_state_index = source.index("fluid.save_state()")
        self.assertLess(reseed_index, boundary_index)
        self.assertLess(reseed_index, save_state_index)

    def test_reseed_call_is_the_first_statement_in_the_step_loop(self):
        source = inspect.getsource(run_turek_hron_fsi)
        loop_index = source.index(
            "for step_index in range(int(config.step_count)):"
        )
        reseed_index = source.index(
            "_reseed_turek_hron_markers(markers, config)"
        )
        t_s_index = source.index("t_s = (step_index + 1) * float(config.dt_s)")
        self.assertLess(loop_index, reseed_index)
        self.assertLess(reseed_index, t_s_index)


class TurekHronMarkerGroupResampleTests(unittest.TestCase):
    """Direct numpy unit tests for _resample_marker_group_arrays."""

    def test_straight_line_preserves_count_area_sum_and_monotonic_velocity(self):
        count = 12
        z = np.linspace(1.9, 2.25, count)
        x = np.stack([np.full(count, 0.025), np.full(count, 0.19), z], axis=1)
        n = np.tile(np.array([0.0, -1.0, 0.0]), (count, 1))
        a = np.linspace(0.5e-4, 1.5e-4, count)
        v = np.stack(
            [np.zeros(count), np.linspace(0.0, 0.02, count), np.zeros(count)],
            axis=1,
        )

        x2, n2, a2, v2 = _resample_marker_group_arrays(x, n, a, v, count)

        self.assertEqual(x2.shape, (count, 3))
        self.assertEqual(n2.shape, (count, 3))
        self.assertEqual(a2.shape, (count,))
        self.assertEqual(v2.shape, (count, 3))
        self.assertAlmostEqual(float(np.sum(a2)), float(np.sum(a)), places=12)
        np.testing.assert_allclose(x2[0], x[0], atol=1e-9)
        np.testing.assert_allclose(x2[-1], x[-1], atol=1e-9)
        # v's wall-normal component increases monotonically with arc length
        # on the input curve, so the resampled stations must stay
        # monotonically non-decreasing too.
        self.assertTrue(bool(np.all(np.diff(v2[:, 1]) >= -1e-12)))

    def test_bent_curve_preserves_count_area_sum_and_monotonic_velocity(self):
        count = 20
        theta = np.linspace(0.0, 0.6, count)
        y = 0.19 + 0.01 * np.sin(theta)
        z = 1.9 + 0.35 * theta / theta[-1]
        x = np.stack([np.full(count, 0.025), y, z], axis=1)
        n = np.tile(np.array([0.0, -1.0, 0.0]), (count, 1))
        a = np.full(count, 2.0e-4)
        v = np.stack(
            [
                np.zeros(count),
                np.linspace(-0.01, 0.03, count),
                np.linspace(0.0, -0.05, count),
            ],
            axis=1,
        )

        x2, n2, a2, v2 = _resample_marker_group_arrays(x, n, a, v, count)

        self.assertEqual(x2.shape[0], count)
        self.assertEqual(n2.shape[0], count)
        self.assertEqual(a2.shape[0], count)
        self.assertEqual(v2.shape[0], count)
        self.assertAlmostEqual(float(np.sum(a2)), float(np.sum(a)), places=12)
        # v[:, 1] increases and v[:, 2] decreases monotonically with marker
        # index; since z (and hence arc length) is also strictly increasing
        # with index (theta is monotonic and the sin() wobble in y is small
        # relative to the z span), both remain monotonic in arc length too.
        self.assertTrue(bool(np.all(np.diff(v2[:, 1]) >= -1e-9)))
        self.assertTrue(bool(np.all(np.diff(v2[:, 2]) <= 1e-9)))

    def test_rejects_count_below_two(self):
        x = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            _resample_marker_group_arrays(x, x, np.zeros(1), x, 1)

    def test_rejects_shape_mismatch_between_x_and_count(self):
        x = np.zeros((5, 3))
        n = np.zeros((5, 3))
        a = np.zeros(5)
        v = np.zeros((5, 3))
        with self.assertRaises(ValueError):
            _resample_marker_group_arrays(x, n, a, v, 4)


class TurekHronMarkerGroupOrderTests(unittest.TestCase):
    """Pins the build_marker_layout ordering that _reseed_turek_hron_markers
    assumes: lower-face side-count markers, then upper-face side-count
    markers, then tip-count tip markers."""

    def test_build_marker_layout_group_order_matches_reseed_assumption(self):
        config = TurekHronFsiConfig()
        side_count, tip_count = resolved_marker_counts(config)
        positions, normals, areas = build_marker_layout(config)

        self.assertEqual(len(positions), 2 * side_count + tip_count)
        self.assertEqual(len(normals), 2 * side_count + tip_count)
        self.assertEqual(len(areas), 2 * side_count + tip_count)

        lower_normals = normals[:side_count]
        upper_normals = normals[side_count : 2 * side_count]
        tip_normals = normals[2 * side_count :]
        for normal in lower_normals:
            self.assertEqual(normal, (0.0, -1.0, 0.0))
        for normal in upper_normals:
            self.assertEqual(normal, (0.0, 1.0, 0.0))
        for normal in tip_normals:
            self.assertEqual(normal, (0.0, 0.0, -1.0))

        # Each group is ordered (monotonic along its own natural axis), the
        # precondition for treating it as an open polyline.
        lower_z = [position[2] for position in positions[:side_count]]
        upper_z = [
            position[2] for position in positions[side_count : 2 * side_count]
        ]
        tip_y = [position[1] for position in positions[2 * side_count :]]
        self.assertEqual(lower_z, sorted(lower_z))
        self.assertEqual(upper_z, sorted(upper_z))
        self.assertEqual(tip_y, sorted(tip_y))

    def test_reseed_helper_signature_matches_module_layout_functions(self):
        # Cheap sanity check that the helper is wired against the SAME
        # resolved_marker_counts/build_marker_layout this test verifies,
        # rather than a stale/private copy of the counts.
        source = inspect.getsource(_reseed_turek_hron_markers)
        self.assertIn("resolved_marker_counts(config)", source)
        self.assertIn("slice(0, side_count)", source)
        self.assertIn("slice(side_count, 2 * side_count)", source)
        self.assertIn(
            "slice(2 * side_count, 2 * side_count + tip_count)", source
        )


if __name__ == "__main__":
    unittest.main()
