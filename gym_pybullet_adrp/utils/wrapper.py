"""Gymnasium wrapper classes"""

from __future__ import annotations
from typing import Any

import numpy as np
from gymnasium import Env, Wrapper


class DroneObservationWrapper(Wrapper):
    """Wrapper to alter the default observation space from the environment for
    RL training.
    """

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """
        obs, info = self.env.reset(*args, **kwargs)
        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for
            details.

        Returns:
            The next observation, the reward, the terminated and truncated
            flags, and the info dict.
        """
        # guarantuee yaw actions to be zero
        action[0,3] = 0

        obs, reward, terminated, truncated, info = self.env.step(action)

        # end the simulation early after passing the first two gates
        if self.env.current_gate[0] >= 2:
            terminated = True

        return obs, reward, terminated, truncated, info


class RewardWrapper(Wrapper):
    """Wrapper to alter the default reward function from the environment for RL
    training.
    """

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)
        self.current_gate_id = None
        self.current_target = None
        self.previous_pos = None

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """
        obs, info = self.env.reset(*args, **kwargs)

        # internal state of the reward wrapper
        self.current_gate_id = int(obs[0, -1])
        self.current_target = obs[0, 12:15]
        self.previous_pos = obs[0, :3]

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for
            details.

        Returns:
            The next observation, the reward, the terminated and truncated
            flags, and the info dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict
    ) -> float:
        """Compute the reward for the current step.

        Args:
            obs: The current observation.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment.

        Returns:
            The computed reward.
        """
        # sparse reward for collisions, gate passage and lap completion
        r_passed = 0
        r_collision = 0
        r_lab = 0
        gate_id = int(obs[0, -1])
        # Assuming gate poses start at index 12 and each gate's pose is
        # represented by 4 consecutive values
        # For example, gate 0 is at obs[0, 12:16], gate 1 at obs[0, 16:20]
        gate_positions = {
            0: obs[0, 12:16],
            1: obs[0, 16:20],
            2: obs[0, 20:24],
            3: obs[0, 24:28],
        }

        if gate_id > (self.current_gate_id) % 4:
            self.current_gate_id = gate_id
            self.current_target = gate_positions[gate_id]
            r_passed = 5

        r_collision = -1 if terminated and not info["task_completed"] else 0
        r_lab = 10 if terminated and info["task_completed"] else 0

        # compute gate progress for movement in x and y direction using l2 norm
        distance_previous_xy = np.linalg.norm(
            self.current_target[0:2] - self.previous_pos[0:2], ord=2
        )
        distance_current_xy = np.linalg.norm(
            self.current_target[0:2] - obs[0][:2], ord=2
        )
        gate_progress_xy = distance_previous_xy - distance_current_xy

        # compute gate progress for movement in z direction using l1 norm
        # (penalizes stronger)
        distance_previous_z = np.abs(
            self.current_target[2] - self.previous_pos[2]
        )
        distance_current_z = np.abs(self.current_target[2] - obs[0][2])
        gate_progress_z = distance_previous_z - distance_current_z

        reward = gate_progress_xy + gate_progress_z + r_passed + r_collision + r_lab

        # Update the previous position
        self.previous_pos = obs[0, :3]

        return reward
