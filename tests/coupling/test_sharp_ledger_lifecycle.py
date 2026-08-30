"""Host control-flow checks; the real canonical/CUDA guard is tested separately.

Compile unchanged production statement blocks and nested helpers with a small
generation-aware protocol double.  This exercises early exits and closure
report writeback without importing or allocating a Taichi solver.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


CORE_PATH = Path(__file__).resolve().parents[2] / "simulation_core/coupling/hibm_mpm/core.py"
_BAND = "mark_hibm_solid_band_nonprojectable_cells"
_AIR = "convert_hibm_air_backed_cells"


def _calls(node: ast.AST, method: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == method
        for child in ast.walk(node)
    )


def _statement_lists(tree: ast.AST):
    for node in ast.walk(tree):
        for _, children in ast.iter_fields(node):
            if isinstance(children, list) and children and all(
                isinstance(child, ast.stmt) for child in children
            ):
                yield children


class _LedgerProtocol:
    def __init__(self, increments, *, conversion="band"):
        self.generation = 0
        self.sealed_generation = 0
        self.increments = iter(increments)
        self.conversion = conversion
        self.read_report = lambda: {"generation": self.generation}
        self.events = []
        self.last_hibm_row_cloud_orphan_component_count = 1
        self.last_hibm_air_backed_component_count = 0
        self.last_hibm_air_backed_cell_volume_m3 = 0.0

    def seal(self):
        self.sealed_generation = self.generation
        self.events.append(("seal", self.generation))
        return {"generation": self.generation}

    def _mutate(self, count):
        self.generation += 1
        self.sealed_generation = -1
        self.events.append(("mutate", self.generation))
        return count

    def mark_hibm_solid_band_nonprojectable_cells(self, **kwargs):
        # The actual band sweep invalidates even when it adds no cells.
        return self._mutate(next(self.increments))

    def convert_hibm_air_backed_cells(self):
        return self._mutate(next(self.increments))

    def convert_hibm_row_cloud_orphan_components(self, **kwargs):
        kind = "orphan"
        if kwargs.get("overflow_singletons_only"):
            kind = "overflow"
        elif kwargs.get("convert_unstamped_small_components"):
            kind = "tiny"
        count = next(self.increments, 0) if kind == self.conversion else 0
        return self._mutate(count) if count > 0 else 0

    def mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(self, **kwargs):
        if self.sealed_generation != self.generation:
            raise AssertionError("reader reached an unsealed current generation")
        if self.read_report()["generation"] != self.generation:
            raise AssertionError("reader reached a stale outer velocity report")
        self.events.append(("read", self.generation))
        return 0


def _namespace(ledger):
    return {
        "fluid": ledger,
        "pressure_outlet_zmin": False,
        "ib_search": SimpleNamespace(node_kind_code=None),
        "HibmMpmIbNodeSearch": SimpleNamespace(_NODE_NONE=0),
        "_debug_stage_progress": lambda stage: None,
        "assemble_velocity_component_face_ledger": ledger.seal,
        "assemble_next_velocity_component_face_ledger": ledger.seal,
        "HIBM_OVERFLOW_SINGLETON_NO_SLIP_PROTECTION_RADIUS_CELLS": 1,
        "HIBM_TINY_UNREACHED_COMPONENT_CLEANUP_THRESHOLD_CELLS": 2,
        "HIBM_PRESSURE_DISCONNECTED_SMALL_COMPONENT_THRESHOLD_CELLS": 2,
        "solid_band_nonprojectable_cell_count": 0,
        "next_solid_band_nonprojectable_cell_count": 0,
        "velocity_report": ledger.seal(),
        "next_velocity_report": ledger.seal(),
    }


def _execute_block(statements, ledger, report_name):
    namespace = _namespace(ledger)
    ledger.read_report = lambda: namespace[report_name]
    module = ast.Module(body=copy.deepcopy(statements), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(CORE_PATH), "exec"), namespace)
    ledger.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells()
    return namespace


class SharpLedgerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"))

    def test_all_band_loops_seal_before_zero_or_budget_exit(self):
        loops = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.For)
            and any(isinstance(stmt, ast.Assign) and _calls(stmt, _BAND) for stmt in node.body)
        ]
        self.assertEqual(len(loops), 4)
        for loop in loops:
            report = "next_velocity_report" if loop.target.id.startswith("_next") else "velocity_report"
            for increments in ([0], [2, 0], [1] * 8):
                with self.subTest(loop=loop.target.id, increments=increments):
                    ledger = _LedgerProtocol(increments)
                    _execute_block([loop], ledger, report)
                    self.assertEqual(ledger.generation, len(increments))
                    self.assertEqual(ledger.sealed_generation, ledger.generation)

    def test_post_solid_first_band_seals_before_first_reader(self):
        blocks = []
        for statements in _statement_lists(self.tree):
            for index, stmt in enumerate(statements):
                if not isinstance(stmt, ast.Assign) or not _calls(stmt, _BAND):
                    continue
                if any(isinstance(target, ast.Name) and target.id == "next_solid_band_nonprojectable_cell_count" for target in stmt.targets):
                    end = next(
                        offset for offset in range(index + 1, len(statements))
                        if isinstance(statements[offset], ast.If)
                    )
                    blocks.append(statements[index:end + 1])
        self.assertEqual(len(blocks), 1)
        for count in (0, 2):
            with self.subTest(count=count):
                _execute_block(blocks[0], _LedgerProtocol([count]), "next_velocity_report")

    def test_both_air_conversions_seal_even_when_zero(self):
        blocks = []
        for statements in _statement_lists(self.tree):
            for index, stmt in enumerate(statements):
                if isinstance(stmt, ast.Assign) and _calls(stmt, _AIR):
                    end = next(
                        offset for offset in range(index + 1, len(statements))
                        if isinstance(statements[offset], ast.If)
                    )
                    name = stmt.targets[0].id
                    report = "next_velocity_report" if name.startswith("next_") else "velocity_report"
                    blocks.append((statements[index:end + 1], report))
        self.assertEqual(len(blocks), 2)
        for statements, report in blocks:
            for count in (0, 2):
                with self.subTest(report=report, count=count):
                    _execute_block(statements, _LedgerProtocol([count, 0]), report)

    def test_cleanup_helpers_publish_current_report_at_first_reader(self):
        functions = {node.name: node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef)}
        for post_solid in (False, True):
            prefix = "next_" if post_solid else ""
            report = f"{prefix}velocity_report"
            names = [
                f"convert_{prefix}row_cloud_orphans_until_saturated",
                f"convert_{prefix}overflow_singletons_without_row_reload",
                f"convert_{prefix}projection_topology_cleanup_until_saturated",
            ]
            if post_solid:
                names.append("rebuild_next_velocity_rows")
            for kind in ("overflow", "tiny"):
                with self.subTest(post_solid=post_solid, kind=kind):
                    ledger = _LedgerProtocol([2, 0], conversion=kind)
                    namespace = _namespace(ledger)
                    declarations = [f"{report} = fluid.seal()"]
                    for counter in (
                        "pressure_disconnected_nonprojectable_cell_count",
                        "row_cloud_orphan_cell_count", "row_cloud_orphan_component_count",
                        "overflow_singleton_cleanup_cell_count", "overflow_singleton_cleanup_component_count",
                        "projection_tiny_unreached_cleanup_cell_count", "projection_tiny_unreached_cleanup_component_count",
                    ):
                        declarations.append(f"{prefix}{counter} = 0")
                    declarations.append(f"fluid.read_report = lambda: {report}")
                    wrapper = ast.parse("def run():\n" + "\n".join(f"    {line}" for line in declarations)).body[0]
                    wrapper.body.extend(copy.deepcopy(functions[name]) for name in names)
                    entry = names[1] if kind == "overflow" else names[2]
                    wrapper.body.extend(ast.parse(f"{entry}()\nreturn {report}").body)
                    module = ast.Module(body=[wrapper], type_ignores=[])
                    exec(compile(ast.fix_missing_locations(module), str(CORE_PATH), "exec"), namespace)
                    final_report = namespace["run"]()
                    self.assertEqual(ledger.generation, 1)
                    self.assertEqual(final_report["generation"], ledger.generation)
                    self.assertIn(("read", 1), ledger.events)


if __name__ == "__main__":
    unittest.main()
