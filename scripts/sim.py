"""Simulation"""

import time
import logging
from typing import List

import fire
import numpy as np
import pybullet as pb
import gymnasium as gym

from gym_pybullet_adrp.utils import load_config, load_controller, sync
from gym_pybullet_adrp.utils.enums import RaceMode
from gym_pybullet_adrp.utils.wrapper import DroneObservationWrapper
from user_controller import BaseController


def simulate(
    config: str="config/getting_started.yaml",
    controller: str | List[str]=[
        "user_controller/HardCodedController.py",
        "user_controller/RLController.py",
    ],
    n_runs: int=10,
    n_drones: int=2,
    gui: bool=True,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file.
        controller: The path(s) to the controller module(s).
        n_runs: The number of episodes.
        n_drones: The number of drones participating.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    config = load_config(config)

    # create race environment
    env = gym.make(
        "multi-race-aviary-v0",
        race_config=config,
        num_drones=n_drones,
        gui=gui,
        racemode=RaceMode.COMPARE
    )
    # NOTE: for evaluating "twogates.yaml"
    # env = DroneObservationWrapper(env)

    gui_timer = pb.addUserDebugText("", np.ones(3), physicsClientId=env.CLIENT)

    # initialize drone agents
    agents: List[BaseController] = []
    if isinstance(controller, str):
        controller = [controller]
    if isinstance(controller, list) and len(controller) != n_drones:
        controller = controller * n_drones

    # track episode statistics
    stats = {
        "episode_times": [0] * n_runs,
        "episode_rewards": [0] * n_runs
    }

    for run in range(n_runs):
        episode_start = time.time()
        sim_time, episode_step = 0, 0
        terminated, truncated = False, False
        obs, info = env.reset()
        agents = []
        for drone_id, c in enumerate(controller):
            info["delay"] = drone_id
            agents.append(load_controller(c)(drone_id, obs[drone_id], info))

        while not (terminated or truncated):
            sim_time = episode_step / config.ctrl_freq

            # update timer
            gui_timer = pb.addUserDebugText(
                f"Ep. time: {sim_time:.2f}s",
                textPosition=[0, 0, 1.5],
                textColorRGB=[1, 0, 0],
                textSize=1.5,
                lifeTime=3 / config.ctrl_freq,
                replaceItemUniqueId=gui_timer,
                physicsClientId=env.CLIENT,
            )

            # select an action for each agent
            actions = [a.predict(obs[i], ep_time=sim_time) for i, a in enumerate(agents)]

            if all(isinstance(a, np.ndarray) for a in actions):
                actions = np.array(actions)

            # perform one step in the environment
            obs, reward, terminated, truncated, _ = env.step(actions)

            # log statistics
            stats["episode_rewards"][run] += reward

            if gui:
                sync(sim_time, episode_start, 1 / config.ctrl_freq)
            episode_step += 1

        stats["episode_times"][run] = sim_time

    env.close()

    return stats["episode_times"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(simulate)
