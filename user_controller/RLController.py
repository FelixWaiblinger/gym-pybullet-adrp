"""Reinforcement Learning Controller"""

import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_adrp.utils.enums import Command
from gym_pybullet_adrp.utils.constants import ZERO3
from gym_pybullet_adrp.utils.utils import map2pi
from user_controller import BaseController


AGENT_PATH = "user_controller/twogates"


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
        #reshape obs to fit the model
        # store additional infos
        #reshape obs to (n_env, 1, 49)
        obs = np.expand_dims(obs, axis=0)
        self.time = ep_time

        action, _ = self.agent.predict(obs, deterministic=True)
        action = self._action_transform(action)

        return action

###############################################################################

    def _action_transform(self, action):
        """Transform the action predicted by the agent before propagating to
        the environment.
        """
        action[0,3] = 0
        action = (action * self.action_scale) 
        action[0,3] = map2pi(action[0,3])  # Ensure yaw is in [-pi, pi]

        cmd = Command.FULLSTATE
        args = [action[0,:3], ZERO3, ZERO3, action[0,3], ZERO3, self.time]

        action = (cmd, args)

        return action

###############################################################################

    def _observation_transform(self, observation):
        """Transform the observations returned by the environment before
        propagating them to the agent.
        """
        return observation
