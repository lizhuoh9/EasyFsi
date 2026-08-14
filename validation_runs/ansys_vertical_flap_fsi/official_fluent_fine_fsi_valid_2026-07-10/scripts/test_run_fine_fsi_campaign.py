from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


MODULE_PATH = Path(__file__).with_name("run_fine_fsi_campaign.py")
SPEC = importlib.util.spec_from_file_location("run_fine_fsi_campaign", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_case(path: Path, solid_types: list[int]) -> None:
    fluid_count = 10
    solid_count = len(solid_types)
    with h5py.File(path, "w") as case:
        topology = case.create_group("meshes/1/cells/zoneTopology")
        topology.create_dataset("name", data=np.asarray([b"solid.5;fluid.4"]))
        topology.create_dataset("id", data=np.asarray([1, 2], dtype=np.int32))
        topology.create_dataset("minId", data=np.asarray([1, solid_count + 1], dtype=np.uint64))
        topology.create_dataset(
            "maxId",
            data=np.asarray([solid_count, solid_count + fluid_count], dtype=np.uint64),
        )
        topology.create_dataset("zoneType", data=np.asarray([1, 1], dtype=np.int32))

        ctype = case.create_group("meshes/1/cells/ctype")
        solid = ctype.create_group("1")
        solid.attrs["minId"] = np.asarray([1], dtype=np.uint64)
        solid.attrs["maxId"] = np.asarray([solid_count], dtype=np.uint64)
        if len(set(solid_types)) == 1:
            solid.attrs["elementType"] = np.asarray([solid_types[0]], dtype=np.int16)
        else:
            solid.attrs["elementType"] = np.asarray([0], dtype=np.int16)
            solid.create_dataset("cell-types", data=np.asarray(solid_types, dtype=np.int16))

        fluid = ctype.create_group("2")
        fluid.attrs["minId"] = np.asarray([solid_count + 1], dtype=np.uint64)
        fluid.attrs["maxId"] = np.asarray(
            [solid_count + fluid_count], dtype=np.uint64
        )
        fluid.attrs["elementType"] = np.asarray([1], dtype=np.int16)


class ValueSetting:
    def __init__(self) -> None:
        self.value = None

    def set_state(self, value):
        self.value = value

    def get_state(self):
        return self.value


class WriteOnlyValueSetting(ValueSetting):
    def get_state(self):
        return None


class FakeIndicator:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


class FakeCellZoneCommand:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args) -> str:
        self.calls.append(args)
        return ""


class FakeSession:
    def __init__(self) -> None:
        settings = type("Settings", (), {})()
        settings.adaption_method = ValueSetting()
        settings.cell_zones = ValueSetting()
        settings.maximum_cell_count = ValueSetting()
        settings.maximum_refinement_level = ValueSetting()
        settings.minimum_edge_length = ValueSetting()
        adapt = type("Adapt", (), {})()
        adapt.set = settings
        mesh = type("Mesh", (), {})()
        mesh.adapt = adapt
        self.mesh = mesh

        self.indicator = FakeIndicator()
        error_based = type("ErrorBased", (), {})()
        error_based.pressure_hessian_indicator = self.indicator
        aerodynamics = type("Aerodynamics", (), {})()
        aerodynamics.error_based = error_based
        predefined = type("Predefined", (), {})()
        predefined.aerodynamics = aerodynamics
        tui_adapt = type("TuiAdapt", (), {})()
        tui_adapt.predefined_criteria = predefined
        tui_adapt.set = type("TuiAdaptSet", (), {})()
        tui_adapt.set.cell_zones = FakeCellZoneCommand()
        tui_mesh = type("TuiMesh", (), {})()
        tui_mesh.adapt = tui_adapt
        tui = type("Tui", (), {})()
        tui.mesh = tui_mesh
        self.tui = tui


class FakeDynamicZones:
    def __init__(self, *, fail_zone: str | None = None) -> None:
        self.fail_zone = fail_zone
        self.create_calls: list[tuple[object, ...]] = []
        self.list_calls = 0

    def create(self, *args) -> str:
        self.create_calls.append(args)
        if args and args[0] == self.fail_zone:
            raise RuntimeError(f"cannot create {args[0]}")
        return "created"

    def list(self) -> str:
        self.list_calls += 1
        return (
            "zone flap_wall-shadow type intrinsic-fsi\n"
            "zone wall type deforming"
        )


def fake_dynamic_session(zones: FakeDynamicZones):
    dynamic_mesh = type("DynamicMesh", (), {})()
    dynamic_mesh.zones = zones
    define = type("Define", (), {})()
    define.dynamic_mesh = dynamic_mesh
    tui = type("Tui", (), {})()
    tui.define = define
    session = type("DynamicSession", (), {})()
    session.tui = tui
    return session


class FakeVector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class FakeReduction:
    def __init__(self, *, flap, inlet, outlet) -> None:
        self.flap = flap
        self.inlet = inlet
        self.outlet = outlet
        self.calls: list[tuple[str, list[object]]] = []

    def force(self, *, locations):
        self.calls.append(("force", locations))
        if locations != [self.flap]:
            raise AssertionError("force must use the fluid-side flap wall")
        return FakeVector(12.5, -3.25, 0.0)

    def mass_flow(self, *, locations):
        self.calls.append(("mass_flow", locations))
        if locations == [self.inlet]:
            return -1.2
        if locations == [self.outlet]:
            return 1.19
        raise AssertionError(f"unexpected mass-flow locations: {locations!r}")


def fake_integral_session():
    flap = object()
    inlet = object()
    outlet = object()
    boundary_conditions = type("BoundaryConditions", (), {})()
    boundary_conditions.wall = {"flap_wall-shadow": flap}
    boundary_conditions.velocity_inlet = {"velocity_inlet.1": inlet}
    boundary_conditions.pressure_outlet = {"po.3": outlet}
    setup = type("Setup", (), {})()
    setup.boundary_conditions = boundary_conditions
    session = type("IntegralSession", (), {})()
    session.setup = setup
    return session, FakeReduction(flap=flap, inlet=inlet, outlet=outlet)


class CampaignTests(unittest.TestCase):
    def test_solid_gate_accepts_original_quadrilateral_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "valid.cas.h5"
            write_case(case_path, [3] * 30)
            report = MODULE.require_supported_solid_topology(case_path)

        self.assertEqual(report["cell_count"], 30)
        self.assertEqual(report["cell_type_counts"], {3: 30})

    def test_solid_gate_rejects_polyhedral_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "invalid.cas.h5"
            write_case(case_path, [3] * 29 + [7])
            with self.assertRaisesRegex(RuntimeError, "type 7"):
                MODULE.require_supported_solid_topology(case_path)

    def test_solid_gate_accepts_explicit_prebuilt_fine_cell_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "fine.cas.h5"
            write_case(case_path, [3] * 48)
            report = MODULE.require_supported_solid_topology(
                case_path, expected_cell_count=48
            )

        self.assertEqual(report["cell_count"], 48)
        self.assertEqual(report["cell_type_counts"], {3: 48})

    def test_fatal_transcript_detection_is_fail_loud(self) -> None:
        errors = MODULE.find_fatal_transcript_errors(
            "Compute processes interrupted.\n"
            "Error at Node 0: Error: element type not implemented\n"
        )
        self.assertIn("compute processes interrupted", errors)
        self.assertIn("element type not implemented", errors)

    def test_required_dynamic_zone_creation_is_recorded_and_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zones = FakeDynamicZones()
            report = MODULE.create_required_dynamic_mesh_zones(
                fake_dynamic_session(zones), Path(tmp)
            )

            events = (Path(tmp) / "events.jsonl").read_text(encoding="utf-8")

        self.assertEqual(
            zones.create_calls,
            [
                ("po.3",),
                ("symmetry.2",),
                ("velocity_inlet.1",),
                ("wall",),
                ("flap_wall-shadow", "intrinsic-fsi"),
            ],
        )
        self.assertEqual(zones.list_calls, 1)
        self.assertIn("flap_wall-shadow", report["list_output"])
        self.assertIn("intrinsic-fsi", report["list_output"])
        for zone_name in ("po.3", "symmetry.2", "velocity_inlet.1", "wall"):
            self.assertIn(zone_name, events)

    def test_intrinsic_fsi_dynamic_zone_creation_exception_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zones = FakeDynamicZones(fail_zone="flap_wall-shadow")
            with self.assertRaisesRegex(RuntimeError, "cannot create flap_wall-shadow"):
                MODULE.create_required_dynamic_mesh_zones(
                    fake_dynamic_session(zones), Path(tmp)
                )

        self.assertEqual(zones.list_calls, 0)

    def test_dynamic_zone_evidence_requires_intrinsic_flap_zone(self) -> None:
        report = MODULE.require_dynamic_zone_setup_evidence(
            list_output="flap_wall-shadow intrinsic-fsi",
            transcript_text="",
        )
        self.assertEqual(report["flap_wall_shadow_evidence"], "flap_wall-shadow")
        self.assertEqual(report["motion_evidence"], "intrinsic-fsi")

        with self.assertRaisesRegex(RuntimeError, "flap_wall-shadow"):
            MODULE.require_dynamic_zone_setup_evidence(
                list_output="wall deforming",
                transcript_text="",
            )

    def test_fsi_setup_transcript_cursor_precedes_setup_and_is_scanned_before_write(self) -> None:
        events: list[str] = []

        class FakeFile:
            def read_case_data(self, *, file_name: str) -> None:
                events.append("read")

            def write_case_data(self, *, file_name: str) -> None:
                events.append("write")

        class FakeFsiSession:
            file = FakeFile()

            def exit(self) -> None:
                events.append("exit")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MODULE.CampaignConfig(
                run_dir=root / "run",
                source_case=root / "steady.cas.h5",
                source_data=root / "steady.dat.h5",
            )
            with (
                mock.patch.object(
                    MODULE,
                    "require_supported_solid_topology",
                    return_value={"cell_count": 30, "cell_type_counts": {3: 30}},
                ),
                mock.patch.object(MODULE, "launch_fluent", return_value=FakeFsiSession()),
                mock.patch.object(
                    MODULE,
                    "transcript_cursor",
                    side_effect=lambda _run_dir: events.append("cursor") or (None, 0),
                ),
                mock.patch.object(
                    MODULE,
                    "apply_official_fsi_setup",
                    side_effect=lambda *_args: events.append("setup")
                    or {"dynamic_zones": {"list_output": "flap_wall-shadow intrinsic-fsi"}},
                ),
                mock.patch.object(
                    MODULE,
                    "transcript_delta",
                    side_effect=lambda *_args: events.append("delta")
                    or ("Error: element type not implemented", (Path("x.trn"), 1)),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "element type not implemented"):
                    MODULE.run_fsi_phase(
                        config,
                        phase_name="gate",
                        steady_case=config.source_case,
                        steady_data=config.source_data,
                        step_count=1,
                        require_first_step_displacement=True,
                    )

        self.assertLess(events.index("cursor"), events.index("setup"))
        self.assertLess(events.index("setup"), events.index("delta"))
        self.assertNotIn("write", events[: events.index("delta")])

    def test_pressure_hessian_is_restricted_to_fluid_zone(self) -> None:
        session = FakeSession()
        report = MODULE.configure_fluid_only_pressure_hessian(
            session,
            maximum_cell_count=100_000,
            maximum_refinement_level=3,
            minimum_edge_length_m=1.0e-5,
        )

        self.assertEqual(session.mesh.adapt.set.adaption_method.value, "puma")
        self.assertEqual(session.mesh.adapt.set.cell_zones.value, ["fluid.4"])
        self.assertEqual(session.indicator.calls, 1)
        self.assertEqual(report["cell_zones"], ["fluid.4"])
        self.assertEqual(
            session.tui.mesh.adapt.set.cell_zones.calls,
            [(2, "()"), (2, "()")],
        )

    def test_pressure_hessian_allows_write_only_cell_zone_setting(self) -> None:
        session = FakeSession()
        session.mesh.adapt.set.cell_zones = WriteOnlyValueSetting()

        report = MODULE.configure_fluid_only_pressure_hessian(
            session,
            maximum_cell_count=100_000,
            maximum_refinement_level=3,
            minimum_edge_length_m=1.0e-5,
        )

        self.assertEqual(session.mesh.adapt.set.cell_zones.value, ["fluid.4"])
        self.assertIsNone(report["cell_zones_get_state"])
        self.assertFalse(report["cell_zones_get_state_available"])
        self.assertEqual(
            report["cell_zone_scope_verification"],
            "post_adapt_hdf_solid_topology",
        )

    def test_structure_gate_requires_nonzero_finite_displacement(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-zero"):
            MODULE.require_nonzero_structure_displacement(
                {"target_total_displacement_m": 0.0}, tolerance_m=1.0e-12
            )
        MODULE.require_nonzero_structure_displacement(
            {"target_total_displacement_m": 5.0e-6}, tolerance_m=1.0e-12
        )

    def test_surface_integrals_use_fluid_side_flap_and_signed_mass_flow(self) -> None:
        session, reduction = fake_integral_session()

        report = MODULE.read_surface_integrals(session, reduction_api=reduction)

        self.assertEqual(report["flap_fluid_force_zone"], "flap_wall-shadow")
        self.assertEqual(report["flap_fluid_force_x_n"], 12.5)
        self.assertEqual(report["flap_fluid_force_y_n"], -3.25)
        self.assertEqual(report["inlet_mass_flow_kg_s"], -1.2)
        self.assertEqual(report["outlet_mass_flow_kg_s"], 1.19)
        self.assertAlmostEqual(report["net_mass_flow_kg_s"], -0.01)
        self.assertAlmostEqual(report["relative_mass_imbalance"], 1.0 / 120.0)
        self.assertEqual(
            [name for name, _ in reduction.calls],
            ["force", "mass_flow", "mass_flow"],
        )

    def test_every_step_measurement_rejects_nonfinite_structure_or_flow(self) -> None:
        structure = {
            "target_x_displacement_m": 1.0e-6,
            "target_y_displacement_m": 2.0e-6,
            "target_total_displacement_m": 3.0e-6,
            "solid_max_total_displacement_m": 4.0e-6,
        }
        flow = {
            "velocity_min_mps": 0.0,
            "velocity_mean_mps": 1.0,
            "velocity_max_mps": 2.0,
            "pressure_min_pa": -1.0,
            "pressure_mean_pa": 0.0,
            "pressure_max_pa": 1.0,
        }

        MODULE.require_valid_step_measurements(structure, flow)
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            MODULE.require_valid_step_measurements(
                {**structure, "solid_max_total_displacement_m": np.nan},
                flow,
            )
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            MODULE.require_valid_step_measurements(
                structure,
                {**flow, "velocity_mean_mps": np.inf},
            )

    def test_every_step_measurement_rejects_zero_solid_displacement(self) -> None:
        structure = {
            "target_x_displacement_m": 0.0,
            "target_y_displacement_m": 0.0,
            "target_total_displacement_m": 0.0,
            "solid_max_total_displacement_m": 0.0,
        }
        flow = {
            "velocity_min_mps": 0.0,
            "velocity_mean_mps": 1.0,
            "velocity_max_mps": 2.0,
            "pressure_min_pa": -1.0,
            "pressure_mean_pa": 0.0,
            "pressure_max_pa": 1.0,
        }

        with self.assertRaisesRegex(RuntimeError, "zero structural displacement"):
            MODULE.require_valid_step_measurements(structure, flow)

    def test_gate_only_cli_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MODULE.config_from_cli(
                ["--gate-only", "--run-dir", str(Path(tmp) / "new-run")]
            )
        self.assertTrue(config.gate_only)
        self.assertEqual(config.processor_count, 1)

    def test_prebuilt_fine_cli_skips_solver_adaption_explicitly(self) -> None:
        config = MODULE.config_from_cli(
            ["--skip-adaptation", "--expected-solid-cell-count", "480"]
        )

        self.assertTrue(config.skip_adaptation)
        self.assertEqual(config.expected_solid_cell_count, 480)

    def test_campaign_rejects_nonserial_processor_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "source.cas.h5"
            data_path = root / "source.dat.h5"
            write_case(case_path, [3] * 30)
            data_path.touch()
            config = MODULE.CampaignConfig(
                run_dir=root / "new-run",
                source_case=case_path,
                source_data=data_path,
                processor_count=2,
            )

            with self.assertRaisesRegex(ValueError, "serial"):
                MODULE.validate_config(config)


if __name__ == "__main__":
    unittest.main()
