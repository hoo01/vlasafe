"""Privileged LIBERO/MuJoCo diagnostics used only for labels and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EVENT_SCHEMA_VERSION = "0.1.0"


def _model_contact_geoms(model: Any) -> set[str]:
    return set(getattr(model, "contact_geoms", []) or [])


def _gripper_contact_geoms(gripper: Any) -> set[str]:
    if isinstance(gripper, dict):
        result: set[str] = set()
        for value in gripper.values():
            result.update(_model_contact_geoms(value))
        return result
    return _model_contact_geoms(gripper)


@dataclass
class SimDiagnostics:
    """Read privileged simulator state; never expose this object to predictors."""

    sim: Any
    robot_geoms: set[str]
    joint_limits: np.ndarray
    joint_tolerance: float = 1e-6

    @classmethod
    def from_vector_env(cls, env: Any) -> "SimDiagnostics":
        wrapper = env.envs[0]
        libero_env = wrapper._env
        robot = libero_env.robots[0]
        sim = libero_env.sim
        robot_geoms = _model_contact_geoms(robot.robot_model)
        robot_geoms.update(_gripper_contact_geoms(getattr(robot, "gripper", None)))
        joint_names = list(robot.robot_model.joints)
        joint_ids = [sim.model.joint_name2id(name) for name in joint_names]
        joint_limits = np.asarray(sim.model.jnt_range[joint_ids], dtype=np.float64)
        if joint_limits.shape != (7, 2):
            raise RuntimeError(f"expected 7 robot joint limits, got {joint_limits.shape}")
        if not robot_geoms:
            raise RuntimeError("no robot contact geoms found")
        return cls(sim=sim, robot_geoms=robot_geoms, joint_limits=joint_limits)

    def sample(self, observation: dict[str, Any]) -> dict[str, Any]:
        import mujoco

        joint_pos = np.asarray(observation["robot_state"]["joints"]["pos"])[0]
        eef_pos = np.asarray(observation["robot_state"]["eef"]["pos"])[0]
        lower_margin = joint_pos - self.joint_limits[:, 0]
        upper_margin = self.joint_limits[:, 1] - joint_pos
        min_margin = np.minimum(lower_margin, upper_margin)
        joint_violation = bool(np.any(min_margin < -self.joint_tolerance))

        self_collision = False
        self_collision_pairs: set[tuple[str, str]] = set()
        robot_contact_count = 0
        max_robot_contact_force = 0.0
        max_robot_contact_pair: list[str] | None = None
        force = np.zeros(6, dtype=np.float64)
        model = getattr(self.sim.model, "_model", self.sim.model)
        data = getattr(self.sim.data, "_data", self.sim.data)
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            name1 = self.sim.model.geom_id2name(int(contact.geom1))
            name2 = self.sim.model.geom_id2name(int(contact.geom2))
            involved1 = name1 in self.robot_geoms
            involved2 = name2 in self.robot_geoms
            if not (involved1 or involved2):
                continue
            robot_contact_count += 1
            if involved1 and involved2:
                self_collision = True
                self_collision_pairs.add(tuple(sorted((name1, name2))))
            mujoco.mj_contactForce(model, data, index, force)
            magnitude = float(np.linalg.norm(force[:3]))
            if magnitude > max_robot_contact_force:
                max_robot_contact_force = magnitude
                max_robot_contact_pair = [name1, name2]

        return {
            "events_instrumented": True,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "event_availability": {
                "self_collision": True,
                "joint_violation": True,
                "workspace_violation": False,
                "impact": False,
            },
            "self_collision": self_collision,
            "joint_violation": joint_violation,
            "joint_violation_indices": np.flatnonzero(
                min_margin < -self.joint_tolerance
            ).astype(int).tolist(),
            # Bounds must be frozen from a documented LIBERO workspace protocol first.
            "workspace_violation": None,
            # Record force now; freeze an impact threshold after the pilot distribution.
            "impact": None,
            "eef_position": eef_pos.astype(float).tolist(),
            "joint_position": joint_pos.astype(float).tolist(),
            "joint_lower_limits": self.joint_limits[:, 0].astype(float).tolist(),
            "joint_upper_limits": self.joint_limits[:, 1].astype(float).tolist(),
            "min_joint_limit_margin": float(np.min(min_margin)),
            "joint_limit_margins": min_margin.astype(float).tolist(),
            "self_collision_pairs": [list(pair) for pair in sorted(self_collision_pairs)],
            "robot_contact_count": robot_contact_count,
            "max_robot_contact_force": max_robot_contact_force,
            "max_robot_contact_pair": max_robot_contact_pair,
            "impact_threshold": None,
        }
