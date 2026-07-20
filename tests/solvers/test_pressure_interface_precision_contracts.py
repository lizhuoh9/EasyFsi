from __future__ import annotations

import inspect
import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class PressureInterfacePrecisionContracts(unittest.TestCase):
    """Keep every projection-divergence pass on the f64 interface ledger."""

    def test_cleanup_projection_divergence_keeps_dt_over_rho_f64(self) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._accumulate_pressure_interface_projection_divergence_kernel
        )
        self.assertIn(
            "dt_over_rho: ti.f64",
            source,
            "cleanup re-projection may not truncate dt/rho after the primary "
            "f64 interface operator pass",
        )


if __name__ == "__main__":
    unittest.main()
