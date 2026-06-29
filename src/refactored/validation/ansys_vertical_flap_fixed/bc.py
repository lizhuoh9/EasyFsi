from __future__ import annotations

from pathlib import Path

import numpy as np


def build_bc_map(config: dict, geometry: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fluid = geometry["fluid_mask"].astype(bool)
    wall = geometry["wall_mask"].astype(bool)
    flap = geometry["flap_mask"].astype(bool)

    inlet = np.zeros_like(fluid, dtype=bool)
    outlet = np.zeros_like(fluid, dtype=bool)
    inlet[:, 0] = fluid[:, 0]
    outlet[:, -1] = fluid[:, -1]

    bc_type = np.full(fluid.shape, "fluid", dtype="<U16")
    bc_type[wall] = "wall_noslip"
    bc_type[flap] = "flap_noslip"
    bc_type[inlet] = "inlet"
    bc_type[outlet] = "outlet"

    boundary_config = config["boundary_conditions"]
    inlet_Uz = np.zeros(fluid.shape, dtype=np.float64)
    inlet_Uy = np.zeros(fluid.shape, dtype=np.float64)
    outlet_pressure = np.zeros(fluid.shape, dtype=np.float64)
    inlet_Uz[inlet] = float(boundary_config["inlet_Uz"])
    inlet_Uy[inlet] = float(boundary_config["inlet_Uy"])
    outlet_pressure[outlet] = float(boundary_config["outlet_pressure"])

    return {
        "bc_type": bc_type,
        "inlet_mask": inlet,
        "outlet_mask": outlet,
        "wall_noslip_mask": wall,
        "flap_noslip_mask": flap,
        "inlet_Uz": inlet_Uz,
        "inlet_Uy": inlet_Uy,
        "outlet_pressure": outlet_pressure,
    }


def save_bc_map(path: Path, bc_map: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bc_map)
