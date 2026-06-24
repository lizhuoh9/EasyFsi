from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from simulation_core.fsi_coupling import action_reaction_balance
from simulation_core.interface_pair import InterfacePairMap, PairMapEntry
from simulation_core.moving_boundary import MovingBoundaryCondition
from benchmarks.official.rigid_multibody import EqualOppositeHingePair


@dataclass(frozen=True)
class PairHingeMultibodyFsiConfig:
    step_count: int
    dt_s: float
    active_until_s: float
    max_fin_rotation_deg: float
    rotation_profile: str = "linear"
    fin_length_m: float = 0.04
    body_mass_kg: float = 0.05
    pair_force_damping_ns_per_m: float = 0.2
    body_drag_ns_per_m: float = 0.0
    lateral_velocity_thrust_ns_per_m: float = 0.0
    lateral_added_mass_kg: float = 0.0

    def __post_init__(self) -> None:
        if self.step_count <= 0:
            raise ValueError("step_count must be positive")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.active_until_s <= 0.0:
            raise ValueError("active_until_s must be positive")
        if self.rotation_profile not in {"linear", "quarter-sine"}:
            raise ValueError("unsupported rotation_profile")
        if self.fin_length_m <= 0.0:
            raise ValueError("fin_length_m must be positive")
        if self.body_mass_kg <= 0.0:
            raise ValueError("body_mass_kg must be positive")
        if self.pair_force_damping_ns_per_m < 0.0:
            raise ValueError("pair_force_damping_ns_per_m must be non-negative")
        if self.body_drag_ns_per_m < 0.0:
            raise ValueError("body_drag_ns_per_m must be non-negative")
        if self.lateral_velocity_thrust_ns_per_m < 0.0:
            raise ValueError("lateral_velocity_thrust_ns_per_m must be non-negative")
        if self.lateral_added_mass_kg < 0.0:
            raise ValueError("lateral_added_mass_kg must be non-negative")


def run_pair_hinge_multibody_fsi_smoke(
    config: PairHingeMultibodyFsiConfig,
) -> dict[str, object]:
    pair_map = InterfacePairMap(
        target_count=2,
        entries=(
            PairMapEntry(target_index=0, source_index=0, weight=1.0),
            PairMapEntry(target_index=1, source_index=1, weight=1.0),
        ),
    )
    full_boundary = MovingBoundaryCondition(
        name="full-solid-displacement",
        pair_map=pair_map,
        transfer_mode="full",
    )
    normal_boundary = MovingBoundaryCondition(
        name="normal-sliding-displacement",
        pair_map=pair_map,
        transfer_mode="normal",
    )
    hinge_pair = EqualOppositeHingePair(
        positive_name="positive-fin",
        negative_name="negative-fin",
        schedule_end_angle_rad=math.radians(config.max_fin_rotation_deg),
        active_until_s=config.active_until_s,
        profile=config.rotation_profile,
    )

    previous_mesh_displacements = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    previous_lateral_speed_mps = 0.0
    forward_velocity_mps = 0.0
    max_abs_fin_rotation_rad = 0.0
    max_mesh_displacement_m = 0.0
    max_fluid_force_norm_n = 0.0
    max_action_reaction_relative_error = 0.0
    history: list[dict[str, object]] = []

    for step_index in range(config.step_count):
        time_s = float(step_index + 1) * config.dt_s
        angles = hinge_pair.angles_at(time_s)
        source_displacements = (
            _fin_tip_displacement(config.fin_length_m, angles["positive-fin"]),
            _fin_tip_displacement(config.fin_length_m, angles["negative-fin"]),
        )
        full_displacements = full_boundary.mesh_displacements(source_displacements)
        normal_displacements = normal_boundary.mesh_displacements(
            source_displacements,
            target_normals=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        )
        mesh_displacements = tuple(
            (
                full_vector[0],
                normal_vector[1] + full_vector[1],
                full_vector[2],
            )
            for full_vector, normal_vector in zip(full_displacements, normal_displacements)
        )
        mesh_velocities = tuple(
            tuple(
                (current[axis] - previous[axis]) / config.dt_s
                for axis in range(3)
            )
            for current, previous in zip(mesh_displacements, previous_mesh_displacements)
        )
        fluid_forces = tuple(
            tuple(
                -config.pair_force_damping_ns_per_m * velocity[axis]
                for axis in range(3)
            )
            for velocity in mesh_velocities
        )
        solid_reactions = pair_map.transpose_forces(
            target_forces=fluid_forces,
            source_count=2,
            action_reaction_sign=-1.0,
        )
        total_fluid_force = _sum_vectors(fluid_forces)
        total_solid_force = _sum_vectors(solid_reactions)
        lateral_speed_mps = sum(abs(velocity[1]) for velocity in mesh_velocities)
        lateral_acceleration_mps2 = (
            lateral_speed_mps - previous_lateral_speed_mps
        ) / config.dt_s
        if time_s <= config.active_until_s:
            unsteady_thrust_force = (
                config.lateral_velocity_thrust_ns_per_m * lateral_speed_mps
                + config.lateral_added_mass_kg * lateral_acceleration_mps2,
                0.0,
                0.0,
            )
        else:
            unsteady_thrust_force = (0.0, 0.0, 0.0)
        body_drag_force = (
            -config.body_drag_ns_per_m * forward_velocity_mps,
            0.0,
            0.0,
        )
        total_fluid_force = (
            total_fluid_force[0] - body_drag_force[0] - unsteady_thrust_force[0],
            total_fluid_force[1] - body_drag_force[1] - unsteady_thrust_force[1],
            total_fluid_force[2] - body_drag_force[2] - unsteady_thrust_force[2],
        )
        total_solid_force = (
            total_solid_force[0] + body_drag_force[0] + unsteady_thrust_force[0],
            total_solid_force[1] + body_drag_force[1] + unsteady_thrust_force[1],
            total_solid_force[2] + body_drag_force[2] + unsteady_thrust_force[2],
        )
        balance = action_reaction_balance(total_fluid_force, total_solid_force)
        forward_velocity_mps += total_solid_force[0] / config.body_mass_kg * config.dt_s
        max_abs_fin_rotation_rad = max(
            max_abs_fin_rotation_rad,
            max(abs(value) for value in angles.values()),
        )
        max_mesh_displacement_m = max(
            max_mesh_displacement_m,
            max(_norm(vector) for vector in mesh_displacements),
        )
        max_fluid_force_norm_n = max(
            max_fluid_force_norm_n,
            _norm(total_fluid_force),
        )
        max_action_reaction_relative_error = max(
            max_action_reaction_relative_error,
            balance.relative_error,
        )
        history.append(
            {
                "step": step_index + 1,
                "time_s": time_s,
                "fin_angles_rad": angles,
                "mesh_displacements_m": mesh_displacements,
                "fluid_forces_n": fluid_forces,
                "solid_reactions_n": solid_reactions,
                "body_drag_force_n": body_drag_force,
                "unsteady_thrust_force_n": unsteady_thrust_force,
                "lateral_speed_mps": lateral_speed_mps,
                "lateral_acceleration_mps2": lateral_acceleration_mps2,
                "total_fluid_force_n": total_fluid_force,
                "total_solid_force_n": total_solid_force,
                "action_reaction_relative_error": balance.relative_error,
                "forward_velocity_mps": forward_velocity_mps,
            }
        )
        previous_mesh_displacements = mesh_displacements
        previous_lateral_speed_mps = lateral_speed_mps

    return {
        "case": "generic-pair-hinge-multibody-fsi",
        "config": asdict(config),
        "computed_result_sources": {
            "fin_rotation_rad": "EqualOppositeHingePair.angles_at(time)",
            "mesh_displacement_m": "MovingBoundaryCondition.mesh_displacements",
            "fluid_force_n": "damping * mesh_velocity",
            "solid_reaction_n": "InterfacePairMap.transpose_forces",
            "body_drag_force_n": "-body_drag_ns_per_m * forward_velocity_mps",
            "unsteady_thrust_force_n": (
                "lateral_velocity_thrust_ns_per_m*lateral_speed + "
                "lateral_added_mass_kg*lateral_acceleration"
            ),
            "forward_velocity_mps": "integral(total_solid_force/body_mass, dt)",
        },
        "max_abs_fin_rotation_rad": max_abs_fin_rotation_rad,
        "max_mesh_displacement_m": max_mesh_displacement_m,
        "max_fluid_force_norm_n": max_fluid_force_norm_n,
        "max_action_reaction_relative_error": max_action_reaction_relative_error,
        "final_forward_velocity_mps": forward_velocity_mps,
        "history": history,
    }


def _fin_tip_displacement(length_m: float, angle_rad: float) -> tuple[float, float, float]:
    return (
        length_m * (1.0 - math.cos(angle_rad)),
        length_m * math.sin(angle_rad),
        0.0,
    )


def _sum_vectors(vectors: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    return tuple(sum(vector[axis] for vector in vectors) for axis in range(3))


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))
