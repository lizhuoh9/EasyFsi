from __future__ import annotations

import unittest

import numpy as np

from simulation_core.coupling.tri_surface import TriSurfaceRegionDiagnostics


_NODES = (4, 4, 4)
_CELL_SHAPE = _NODES
_BAD_CELL_SHAPE = (3, 4, 4)


class _Field:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _GridFields:
    def __init__(self) -> None:
        self.cell_face_x_m = _Field((_NODES[0] + 1,))
        self.cell_face_y_m = _Field((_NODES[1] + 1,))
        self.cell_face_z_m = _Field((_NODES[2] + 1,))
        self.cell_center_x_m = _Field((_NODES[0],))
        self.cell_center_y_m = _Field((_NODES[1],))
        self.cell_center_z_m = _Field((_NODES[2],))
        self.cell_width_x_m = _Field((_NODES[0],))
        self.cell_width_y_m = _Field((_NODES[1],))
        self.cell_width_z_m = _Field((_NODES[2],))
        self.obstacle = _Field(_CELL_SHAPE)
        self.velocity_constraint_primary_sum = _Field(_CELL_SHAPE)
        self.velocity_constraint_primary_weight = _Field(_CELL_SHAPE)
        self.velocity_constraint_secondary_sum = _Field(_CELL_SHAPE)
        self.velocity_constraint_secondary_weight = _Field(_CELL_SHAPE)


def _device_work_must_not_start(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("device work must not start")


def _bare_diagnostics() -> TriSurfaceRegionDiagnostics:
    diagnostics = object.__new__(TriSurfaceRegionDiagnostics)
    diagnostics.face_count = 0
    diagnostics.last_report_host_reads = 0
    diagnostics._spread_pressure_interface_matrix_terms_kernel = (
        _device_work_must_not_start
    )
    diagnostics._spread_fsi_force_kernel = _device_work_must_not_start
    diagnostics._spread_fsi_velocity_constraint_kernel = _device_work_must_not_start
    diagnostics._diagnose_from_fields_kernel = _device_work_must_not_start
    return diagnostics


def _grid_arguments(grid_fields: _GridFields, grid_nodes=_NODES) -> dict[str, object]:
    return {
        "grid_fields": grid_fields,
        "probe_distance_m": 0.01,
        "bounds_min_m": (0.0, 0.0, 0.0),
        "bounds_max_m": (1.0, 1.0, 1.0),
        "spacing_m": (0.25, 0.25, 0.25),
        "grid_nodes": grid_nodes,
    }


def _pressure_arguments(grid_fields: _GridFields, grid_nodes=_NODES) -> dict[str, object]:
    return {
        **_grid_arguments(grid_fields, grid_nodes),
        "primary_region_id": 7,
        "secondary_region_id": 8,
        "primary_pressure_robin_impedance_ns_m": 0.0,
        "secondary_pressure_robin_impedance_ns_m": 0.0,
        "primary_pressure_robin_reference_pa": 0.0,
        "secondary_pressure_robin_reference_pa": 0.0,
        "primary_interface_area_m2": 0.0,
        "secondary_interface_area_m2": 0.0,
        "density_kgm3": 1.0,
        "dt_s": 0.001,
    }


def _force_arguments(grid_fields: _GridFields) -> dict[str, object]:
    return {
        **_grid_arguments(grid_fields),
        "primary_region_id": 7,
        "secondary_region_id": 8,
        "primary_velocity_mps": (0.0, 0.0, 0.0),
        "secondary_velocity_mps": (0.0, 0.0, 0.0),
        "density_kgm3": 1.0,
        "viscosity_pa_s": 0.0,
        "dt_s": 0.001,
        "constraint_force_scale": 0.0,
    }


class TriSurfaceShapeContracts(unittest.TestCase):
    def test_grid_nodes_reject_bool_and_fractional_values_before_kernel(self) -> None:
        diagnostics = _bare_diagnostics()
        fields = (_Field(_CELL_SHAPE),) * 3
        for nodes in (
            (4.0, 4, 4),
            (4.9, 4, 4),
            (True, 4, 4),
            (4, np.bool_(True), 4),
        ):
            with self.subTest(nodes=nodes), self.assertRaisesRegex(
                ValueError,
                "grid_nodes",
            ):
                diagnostics.spread_pressure_interface_matrix_terms(
                    *fields,
                    **_pressure_arguments(_GridFields(), nodes),
                )

        contract = diagnostics._grid_contract(
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=(0.25, 0.25, 0.25),
            grid_nodes=(np.int64(4), np.int64(4), np.int64(4)),
        )
        self.assertEqual(contract[-1], _NODES)

    def test_every_cartesian_coordinate_field_must_match_grid_nodes(self) -> None:
        diagnostics = _bare_diagnostics()
        fields = _GridFields()
        expected_shapes = {
            "cell_face_x_m": (5,),
            "cell_face_y_m": (5,),
            "cell_face_z_m": (5,),
            "cell_center_x_m": (4,),
            "cell_center_y_m": (4,),
            "cell_center_z_m": (4,),
            "cell_width_x_m": (4,),
            "cell_width_y_m": (4,),
            "cell_width_z_m": (4,),
        }
        for name, expected in expected_shapes.items():
            with self.subTest(field=name):
                setattr(fields, name, _Field((expected[0] + 1,)))
                with self.assertRaisesRegex(ValueError, name):
                    diagnostics._grid_field_tuple(fields, _NODES)
                setattr(fields, name, _Field(expected))

    def test_each_public_grid_kernel_rejects_bad_cell_shape_before_launch(self) -> None:
        cell = _Field(_CELL_SHAPE)
        bad = _Field(_BAD_CELL_SHAPE)
        cases = (
            (
                "spread_pressure_interface_matrix_terms.diagonal_field",
                lambda diagnostics: diagnostics.spread_pressure_interface_matrix_terms(
                    bad,
                    cell,
                    cell,
                    **_pressure_arguments(_GridFields()),
                ),
            ),
            (
                "spread_fsi_forces.velocity_field",
                lambda diagnostics: diagnostics.spread_fsi_forces(
                    bad,
                    cell,
                    cell,
                    cell,
                    cell,
                    read_full_report=False,
                    read_force_pair_report=False,
                    **_force_arguments(_GridFields()),
                ),
            ),
            (
                "diagnose_fsi_forces_from_fields.force_field",
                lambda diagnostics: diagnostics.diagnose_fsi_forces_from_fields(
                    cell,
                    cell,
                    bad,
                    cell,
                    cell,
                    **_force_arguments(_GridFields()),
                ),
            ),
            (
                "spread_fsi_velocity_constraints.weight_field",
                lambda diagnostics: diagnostics.spread_fsi_velocity_constraints(
                    cell,
                    bad,
                    primary_region_id=7,
                    secondary_region_id=8,
                    primary_velocity_mps=(0.0, 0.0, 0.0),
                    secondary_velocity_mps=(0.0, 0.0, 0.0),
                    read_full_report=False,
                    **_grid_arguments(_GridFields()),
                ),
            ),
            (
                "diagnose_from_fields.pressure_field",
                lambda diagnostics: diagnostics.diagnose_from_fields(
                    cell,
                    bad,
                    primary_region_id=7,
                    secondary_region_id=8,
                    primary_velocity_mps=(0.0, 0.0, 0.0),
                    secondary_velocity_mps=(0.0, 0.0, 0.0),
                    viscosity_pa_s=0.0,
                    **_grid_arguments(_GridFields()),
                ),
            ),
        )
        for name, invoke in cases:
            with self.subTest(entry=name), self.assertRaisesRegex(ValueError, name):
                invoke(_bare_diagnostics())

    def test_optional_region_constraint_fields_are_shape_checked(self) -> None:
        diagnostics = _bare_diagnostics()
        grid_fields = _GridFields()
        grid_fields.velocity_constraint_secondary_weight = _Field(_BAD_CELL_SHAPE)
        with self.assertRaisesRegex(
            ValueError,
            "velocity_constraint_secondary_weight",
        ):
            diagnostics.spread_fsi_velocity_constraints(
                _Field(_CELL_SHAPE),
                _Field(_CELL_SHAPE),
                primary_region_id=7,
                secondary_region_id=8,
                primary_velocity_mps=(0.0, 0.0, 0.0),
                secondary_velocity_mps=(0.0, 0.0, 0.0),
                read_full_report=False,
                **_grid_arguments(grid_fields),
            )


if __name__ == "__main__":
    unittest.main()
