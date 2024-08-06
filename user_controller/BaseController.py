"""Controller"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class BaseController(ABC):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.
    """

###############################################################################

    def __init__(self,
        drone_id: int,
        initial_obs: np.ndarray=None,
        initial_info: dict=None,
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

        self.drone_id = drone_id
        self.initial_obs = initial_obs
        self.initial_info = initial_info
        self.buffer_size = buffer_size
        self.verbose = verbose

        self.action_buffer = deque([], maxlen=buffer_size)
        self.obs_buffer = deque([], maxlen=buffer_size)
        self.reward_buffer = deque([], maxlen=buffer_size)
        self.done_buffer = deque([], maxlen=buffer_size)
        self.info_buffer = deque([], maxlen=buffer_size)

        self.reset()
        self.episode_reset()

###############################################################################

    def reset(self):
        """Initialize/reset data buffers and counters."""
        self.action_buffer = deque([], maxlen=self.buffer_size)
        self.obs_buffer = deque([], maxlen=self.buffer_size)
        self.reward_buffer = deque([], maxlen=self.buffer_size)
        self.done_buffer = deque([], maxlen=self.buffer_size)
        self.info_buffer = deque([], maxlen=self.buffer_size)

        # NOTE: can be implemented by its subclasses

###############################################################################

    def episode_reset(self):
        """Reset the controller's internal state and models if necessary."""
        # NOTE: can be implemented by its subclasses

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
        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        # NOTE: can be implemented by its subclasses

###############################################################################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.s
        """
        # NOTE: can be implemented by its subclasses

###############################################################################

    @abstractmethod
    def predict(self,
        obs: np.ndarray,
        reward: float=None,
        done: bool=None,
        info: dict=None,
        ep_time: float=None
    ) -> np.ndarray:
        """Predict the next action."""
        # NOTE: must be implemented by its subclasses
