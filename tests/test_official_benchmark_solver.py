from __future__ import annotations

import unittest

from simulation_core.fsi_driver import FsiCaseSpec
from simulation_core.benchmarking.official_benchmark_solver import (
    OfficialBenchmarkRunSpec,
    run_official_fsi_benchmark,
)


class OfficialBenchmarkSolverTests(unittest.TestCase):
    def test_shared_runner_adds_standard_case_fields(self) -> None:
        case_spec = FsiCaseSpec(
            case_id="toy-official-fsi",
            source_url="https://example.invalid/toy",
            coordinate_model="cartesian-2d",
            geometry={"kind": "toy"},
            fluid={"kind": "toy"},
            solid={"kind": "toy"},
            boundary_conditions={"interface": {"type": "two-way-fsi"}},
            reference_results={"value": 1.0},
            acceptance_tolerance=0.05,
        )

        report = run_official_fsi_benchmark(
            OfficialBenchmarkRunSpec(
                case_spec=case_spec,
                solver_family="toy-family",
                case_metadata={"source": "unit-test"},
                boundary_conditions=case_spec.boundary_conditions,
                config={"steps": 1},
                runner=lambda _config: {
                    "computed_result_sources": {"value": "computed"},
                    "value": 1.0,
                },
            )
        )

        self.assertEqual(report["case"], "toy-official-fsi")
        self.assertEqual(report["solver_family"], "toy-family")
        self.assertEqual(report["acceptance_tolerance"], 0.05)
        self.assertEqual(report["reference_results"], {"value": 1.0})

    def test_shared_runner_rejects_wrong_case_id(self) -> None:
        case_spec = FsiCaseSpec(
            case_id="expected",
            source_url="https://example.invalid/toy",
            coordinate_model="cartesian-2d",
            geometry={},
            fluid={},
            solid={},
            boundary_conditions={},
            reference_results={},
        )

        with self.assertRaisesRegex(ValueError, "expected"):
            run_official_fsi_benchmark(
                OfficialBenchmarkRunSpec(
                    case_spec=case_spec,
                    solver_family="toy-family",
                    case_metadata={},
                    boundary_conditions={},
                    config=None,
                    runner=lambda _config: {
                        "case": "wrong",
                        "computed_result_sources": {},
                    },
                )
            )


if __name__ == "__main__":
    unittest.main()
