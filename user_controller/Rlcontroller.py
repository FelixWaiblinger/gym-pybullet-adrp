"""HardCodedController"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from user_controller import BaseController
from gym_pybullet_adrp.utils import draw_trajectory
from gym_pybullet_adrp.utils.enums import Command
from gym_pybullet_adrp.utils.constants import Z_LOW, Z_HIGH, CTRL_FREQ, CTRL_DT
from gym_pybullet_adrp.utils.utils import map2pi
from gym_pybullet_adrp.utils.enums import *
import os
from stable_baselines3 import PPO


class Rlcontroller(BaseController):
    """Template controller class."""

###############################################################################

    def __init__(
        self,
        drone_id: int,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. Consists of
                [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range,
                gate_id]
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
        """
        super().__init__(drone_id, initial_obs, initial_info, buffer_size, verbose)
        # Save environment and control parameters.
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self._drone_pose = None
        self.action_scale = np.array([1, 1, 1, np.pi])
        self.state = 0

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #NOTE: no need to pass the enviroment to PPO.load
        # get the the relative path of the model
        MODEL = "baseline_level2"
        # global PATH directory
        PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", MODEL))
        self.model = PPO.load(PATH)

    def reset(self):
        self._drone_pose = self.initial_obs[[0, 1, 2, 5]]



###############################################################################


    def predict(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        zero = np.zeros(3)
        action, _ = self.model.predict(obs, deterministic=True)
        action[3] = 0
        action = self._action_transform(action).astype(float)
        command_type = Command.FULLSTATE
        action_trans = np.array([1,1,1]).astype(float)
        args = [action_trans, zero, zero, action[3], zero, ep_time]
        #print("RL controller")
        #print(command_type, args)
        return command_type, args

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
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
        self._drone_pose = obs[[0, 1, 2, 5]]
        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer


    def _action_transform(self, action: np.ndarray) -> np.ndarray:
        """Transform the action to the format expected by the firmware env.

        Args:
            action: The action to transform.

        Returns:
            The transformed action.
        """
        action = self._drone_pose + (action * self.action_scale)
        action[3] = map2pi(action[3])  # Ensure yaw is in [-pi, pi]
        return action
    
    def _check_state(self, time, info):
        if self.state == 0: # initialization state
            return 1
        elif self.state == 1 and time < 1: # take off state
            return 2
        elif self.state == 2 and time < 5:#info["task_completed"]: # flying state
            return 3
        elif self.state == 3: # notify state
            return 4
        elif self.state == 4: # landing state
            return 5
        else: # finished state
            return self.state
        