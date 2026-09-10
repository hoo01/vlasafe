"""Inspect the installed LIBERO/robosuite diagnostic API without collecting data."""

from __future__ import annotations

import os

import mujoco
import numpy as np

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig


def describe(name: str, value: object) -> None:
    if value is None:
        print(f"{name}: <missing>")
        return
    try:
        array = np.asarray(value)
        print(f"{name}: shape={array.shape} dtype={array.dtype} value={array}")
    except Exception:
        print(f"{name}: type={type(value).__module__}.{type(value).__name__} value={value!r}")


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cfg = LiberoEnvConfig(
        task="libero_spatial",
        task_ids=[0],
        observation_height=64,
        observation_width=64,
        episode_length=2,
    )
    env = cfg.create_envs(n_envs=1, use_async_envs=False)["libero_spatial"][0]
    try:
        obs, _ = env.reset(seed=123)
        wrapper = env.envs[0]
        libero_env = wrapper._env
        robosuite_env = getattr(libero_env, "env", libero_env)
        sim = libero_env.sim
        robot = libero_env.robots[0]
        controller = robot.controller

        print("wrapper_type:", type(wrapper).__module__, type(wrapper).__name__)
        print("libero_env_type:", type(libero_env).__module__, type(libero_env).__name__)
        print("robosuite_env_type:", type(robosuite_env).__module__, type(robosuite_env).__name__)
        print("sim_type:", type(sim).__module__, type(sim).__name__)
        print("robot_type:", type(robot).__module__, type(robot).__name__)
        print("controller_type:", type(controller).__module__, type(controller).__name__)
        describe("robot0_joint_pos", obs["robot_state"]["joints"]["pos"][0])
        describe("robot0_eef_pos", obs["robot_state"]["eef"]["pos"][0])
        for attr in (
            "_ref_joint_pos_indexes",
            "_ref_joint_vel_indexes",
            "joint_indexes",
            "joint_limits",
        ):
            describe(f"robot.{attr}", getattr(robot, attr, None))
        for attr in ("position_limits", "orientation_limits", "input_min", "input_max"):
            describe(f"controller.{attr}", getattr(controller, attr, None))

        contact_geoms = list(getattr(robot.robot_model, "contact_geoms", []))
        print("robot_contact_geoms_count:", len(contact_geoms))
        print("robot_contact_geoms_sample:", contact_geoms[:20])
        print("sim_ncon:", int(sim.data.ncon))
        print("mujoco_mj_contactForce:", hasattr(mujoco, "mj_contactForce"))
        print("model_jnt_range_shape:", np.asarray(sim.model.jnt_range).shape)
        print("model_geom_names_count:", len(sim.model.geom_names))

        if sim.data.ncon:
            print("active_contacts:")
            force = np.zeros(6, dtype=np.float64)
            for index in range(min(int(sim.data.ncon), 20)):
                contact = sim.data.contact[index]
                name1 = sim.model.geom_id2name(int(contact.geom1))
                name2 = sim.model.geom_id2name(int(contact.geom2))
                mujoco.mj_contactForce(sim.model._model, sim.data._data, index, force)
                print(index, name1, name2, "force6=", force.copy())
        else:
            print("active_contacts: none")
    finally:
        env.close()


if __name__ == "__main__":
    main()
