"""Reinforcement Learning Controller"""

import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_adrp.utils.enums import Command
from gym_pybullet_adrp.utils.constants import ZERO3
from gym_pybullet_adrp.utils.utils import map2pi
from user_controller import BaseController


AGENT_PATH = "baseline_level3"


class RLController(BaseController):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.
    """

    def __init__(
        self,
        drone_id: int,
        initial_obs: np.ndarray=None,
        initial_info: dict=None,
        buffer_size: int=100,
        verbose: bool=False
    ):
        super().__init__(drone_id, initial_obs, initial_info, buffer_size, verbose)

        self.agent = PPO.load(AGENT_PATH)
        self.action_scale = np.array([1, 1, 1, np.pi])
        self.drone_pose = initial_obs[[0, 1, 2, 5]]
        self.time = 0

###############################################################################

    def predict(self,
        obs: np.ndarray,
        reward: float=None,
        done: bool=None,
        info: dict=None,
        ep_time: float=None
    ) -> np.ndarray:
        """Predict the next action."""
        obs = self._observation_transform(obs)

        # store additional infos
        self.drone_pose = obs[[0, 1, 2, 5]]
        self.time = ep_time

        action, _ = self.agent.predict(obs, deterministic=True)
        action = self._action_transform(action)

        return action

###############################################################################

    def _action_transform(self, action):
        """TODO"""
        action[3] = 0
        action = self.drone_pose + (action * self.action_scale)
        action[3] = map2pi(action[3])  # Ensure yaw is in [-pi, pi]

        cmd = Command.FULLSTATE
        args = [action[:3], ZERO3, ZERO3, action[3], ZERO3, self.time]

        action = (cmd, args)

        return action

###############################################################################

    def _observation_transform(self, observation):
        """TODO"""
        return observation
