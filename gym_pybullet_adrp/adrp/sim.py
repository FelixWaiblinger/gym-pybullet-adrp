"""Test"""

import time
import logging

import fire
import numpy as np
import pybullet as pb
import gymnasium as gym

from gym_pybullet_adrp.utils import load_config, load_controller, sync


def simulate(
    config: str = "config/getting_started.yaml",
    controller: str | list[str] = "examples/controller.py",
    n_runs: int = 1,
    gui: bool = True,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file.
        controller: The path to the controller module.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    config = load_config(config)

    # create race environment
    env = gym.make("MultiRaceAviary", race_config=config, gui=gui)

    # initialize drone agents
    if isinstance(controller, str):
        controller = [controller]
    agents = [load_controller(c)(env) for c in controller]

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
                lifeTime=3 / config.ctrl_freq,
                textSize=1.5,
                parentObjectUniqueId=0,
                parentLinkIndex=-1,
                replaceItemUniqueId=gui_timer,
                physicsClientId=env.pyb_client_id,
            )

            # select an action for each agent
            actions = np.array([a.predict(obs) for a in agents])

            # perform one step in the environment
            obs, reward, terminated, truncated, _ = env.step(actions)

            # log statistics
            stats["episode_rewards"][run] += reward

            if gui:
                sync(sim_time, episode_start, 1 / config.ctrl_freq)
            sim_time += 1

    env.close()

    return stats["episode_time"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(simulate)
