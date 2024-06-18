"""SACController"""

import numpy as np
from stable_baselines3 import SAC

from user_controller import BaseController


AGENT_PATH = "./sac_agent"


class SACController(BaseController):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.
    """

###############################################################################

    def __init__(self,
        drone_id: int,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int=100,
        verbose: bool=False
    ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.
        """
        super().__init__(drone_id, initial_obs, initial_info, buffer_size, verbose)
        self.agent = SAC.load(AGENT_PATH)
        # NOTE: anything else to do here?

###############################################################################

    def reset(self):
        """Initialize/reset data buffers and counters."""
        super().reset()
        # NOTE: anything else to do here?

###############################################################################

    def episode_reset(self):
        """Reset the controller's internal state and models if necessary."""
        # NOTE: anything else to do here?

###############################################################################

    def step_learn(self,
        action: np.ndarray,
        obs: np.ndarray,
        reward: float,
        done: bool,
        info: dict
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.

        """
        super().step_learn(action, obs, reward, done, info)
        # NOTE: anything else to do here?

###############################################################################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.s
        """
        # NOTE: anything else to do here?

###############################################################################

    def predict(self,
        obs: np.ndarray,
        reward: float=None,
        done: bool=None,
        info: dict=None,
        ep_time: float=None
    ) -> np.ndarray:
        """Predict the next action."""
        return self.agent.predict(obs)[0].astype(np.float32)
