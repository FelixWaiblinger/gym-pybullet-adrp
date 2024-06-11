"""Simulation"""

import time
import logging
from typing import List

import fire
import numpy as np
import pybullet as pb
import gymnasium as gym

from gym_pybullet_adrp.utils import load_config, load_controller, sync
from scripts.controller import Controller


def simulate(
    config: str="config/getting_started.yaml",
    controller: str | List[str]="scripts/controller.py",
    n_runs: int=1,
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
    env = gym.make("multi-race-aviary-v0", race_config=config, gui=gui)
    gui_timer = pb.addUserDebugText("", np.ones(3), physicsClientId=env.CLIENT)

    # initialize drone agents
    if isinstance(controller, str):
        controller = [controller] * n_drones
    agents: List[Controller] = [load_controller(c)() for c in controller]

    # track episode statistics
    stats = {
        "episode_times": [0] * n_runs,
        "episode_rewards": [0] * n_runs
    }

    for run in range(n_runs):
        episode_start = time.time()
        sim_time, episode_step = 0, 0
        terminated, truncated = False, False
        obs, _ = env.reset()

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
            actions = np.vstack([a.predict(obs) for a in agents])

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
