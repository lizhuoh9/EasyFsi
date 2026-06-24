from __future__ import annotations

import importlib
import sys
from pathlib import Path


IMPORTS = (
    "simulation_core",
    "simulation_core.fluids",
    "simulation_core.coupling",
    "simulation_core.coupling.hibm_mpm",
    "simulation_core.solids",
    "simulation_core.geometry_tools",
    "simulation_core.materials",
    "simulation_core.diagnostics",
    "cases.squid_soft_robot",
    "benchmarks.official",
    "tools.diagnostics",
    "tools.rendering",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    for name in IMPORTS:
        importlib.import_module(name)
    print("structure imports OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
