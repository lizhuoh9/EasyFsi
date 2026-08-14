from __future__ import annotations

import unittest

from simulation_core.drivers.case_spec import FsiCaseSpec
from benchmarks.official.official_benchmark_solver import (
    OfficialBenchmarkRunSpec,
    run_official_fsi_benchmark,
)


class OfficialBenchmarkSolverTests(unittest.TestCase):
    @staticmethod
    def _case_spec() -> FsiCaseSpec:
        return FsiCaseSpec(
            case_id="expected",
            source_url="https://example.invalid/toy",
            coordinate_model="cartesian-2d",
            geometry={},
            fluid={},
            solid={},
            boundary_conditions={"wall": "no-slip"},
            reference_results={"value": 1.0},
            acceptance_tolerance=0.05,
        )

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
        case_spec = self._case_spec()

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

    def test_runner_cannot_override_authoritative_acceptance_contract(self) -> None:
        case_spec = self._case_spec()
        for field, forged_value in (
            ("acceptance_tolerance", 999.0),
            ("reference_results", {"value": 999.0}),
            ("boundary_conditions", {"wall": "slip"}),
            ("case_metadata", {"source": "forged"}),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    run_official_fsi_benchmark(
                        OfficialBenchmarkRunSpec(
                            case_spec=case_spec,
                            solver_family="toy-family",
                            case_metadata={"source": "trusted"},
                            boundary_conditions=case_spec.boundary_conditions,
                            config=None,
                            runner=lambda _config, field=field, value=forged_value: {
                                "computed_result_sources": {"value": "computed"},
                                field: value,
                            },
                        )
                    )

    def test_computed_result_sources_must_be_a_nonempty_mapping(self) -> None:
        case_spec = self._case_spec()
        for invalid_sources in ({}, [], None):
            with self.subTest(computed_result_sources=invalid_sources):
                with self.assertRaisesRegex(ValueError, "computed_result_sources"):
                    run_official_fsi_benchmark(
                        OfficialBenchmarkRunSpec(
                            case_spec=case_spec,
                            solver_family="toy-family",
                            case_metadata={"source": "trusted"},
                            boundary_conditions=case_spec.boundary_conditions,
                            config=None,
                            runner=lambda _config, sources=invalid_sources: {
                                "computed_result_sources": sources,
                            },
                        )
                    )


if __name__ == "__main__":
    unittest.main()
