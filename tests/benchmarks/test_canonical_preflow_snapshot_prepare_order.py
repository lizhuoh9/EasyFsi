from __future__ import annotations

import ast
from pathlib import Path
import unittest


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "official"
    / "solid_mpm_fsi_runner.py"
)
PREPARE_TUPLE_NAME = "_CANONICAL_SNAPSHOT_RESTORE_PREPARE_METHODS"


def _canonical_prepare_consumers() -> tuple[str, ...]:
    module = ast.parse(
        RUNNER_PATH.read_text(encoding="utf-8"),
        filename=str(RUNNER_PATH),
    )
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == PREPARE_TUPLE_NAME
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        return tuple(str(consumer) for consumer, _method_name in value)
    raise AssertionError(f"{PREPARE_TUPLE_NAME} is missing from {RUNNER_PATH}")


class CanonicalPreflowSnapshotPrepareOrderContracts(unittest.TestCase):
    def test_projection_is_prepared_after_its_physical_dependencies(self) -> None:
        consumers = _canonical_prepare_consumers()
        positions = {consumer: index for index, consumer in enumerate(consumers)}
        dependencies = (
            "apply",
            "divergence",
            "reachability",
            "fv_operator",
            "gradient",
            "multigrid",
        )

        self.assertEqual(
            len(positions),
            len(consumers),
            "prepare consumers must be unique",
        )
        for consumer in (*dependencies, "projection"):
            with self.subTest(required_consumer=consumer):
                self.assertIn(consumer, positions)
        for dependency in dependencies:
            with self.subTest(dependency=dependency):
                self.assertLess(positions[dependency], positions["projection"])

    def test_projection_is_prepared_before_sealed_only_consumers(self) -> None:
        consumers = _canonical_prepare_consumers()
        positions = {consumer: index for index, consumer in enumerate(consumers)}
        sealed_only_consumers = ("no_slip", "reference", "snapshot")

        self.assertIn("projection", positions)
        for consumer in sealed_only_consumers:
            with self.subTest(sealed_only_consumer=consumer):
                self.assertIn(consumer, positions)
                self.assertLess(positions["projection"], positions[consumer])


if __name__ == "__main__":
    unittest.main()
