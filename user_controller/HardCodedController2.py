"""HardCodedController"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from user_controller import BaseController
from gym_pybullet_adrp.utils import draw_trajectory
from gym_pybullet_adrp.utils.enums import Command
from gym_pybullet_adrp.utils.constants import Z_LOW, Z_HIGH, CTRL_FREQ, CTRL_DT


class HardCodedController2(BaseController):
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
        self.CTRL_TIMESTEP = CTRL_DT # NOTE: changed
        self.CTRL_FREQ = CTRL_FREQ # NOTE: changed
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_obs[12:28].reshape((4, 4)) # NOTE: changed
        self.NOMINAL_OBSTACLES = initial_obs[32:44].reshape((4, 3)) # NOTE: changed

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled
        waypoints = []
        waypoints.append([self.initial_obs[0], self.initial_obs[1], 0.3])
        gates = self.NOMINAL_GATES
        z_low = Z_LOW # NOTE: changed
        z_high = Z_HIGH # NOTE: changed
        waypoints.append([1, 0, z_low])
        waypoints.append([gates[0][0] + 0.2, gates[0][1] + 0.1, z_low])
        waypoints.append([gates[0][0] + 0.1, gates[0][1], z_low])
        waypoints.append([gates[0][0] - 0.1, gates[0][1], z_low])
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.7,
                (gates[0][1] + gates[1][1]) / 2 - 0.3,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.5,
                (gates[0][1] + gates[1][1]) / 2 - 0.6,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append([gates[1][0] - 0.3, gates[1][1] - 0.2, z_high])
        waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])
        waypoints.append([gates[2][0], gates[2][1] - 0.4, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.2, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.2, z_high + 0.2])
        waypoints.append([gates[3][0], gates[3][1] + 0.1, z_high])
        waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.1])
        waypoints.append(
            [
                -0.5,
                -1.2,
                z_high
                # initial_info["x_reference"][0],
                # initial_info["x_reference"][2],
                # initial_info["x_reference"][4],
            ]
        )
        waypoints.append(
            [
                -0.5,
                -1.4,
                z_high
                # initial_info["x_reference"][0],
                # initial_info["x_reference"][2] - 0.2,
                # initial_info["x_reference"][4],
            ]
        )
        waypoints = np.array(waypoints)

        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        self.waypoints = waypoints
        duration = 12
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))
        self.ref_x, self.ref_y, self.ref_z = interpolate.splev(t, tck)
        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        self._take_off = False
        self._setpoint_land = False
        self._land = False

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
            obs: The environment's observation [drone_xyz_yaw, gates_xyz_yaw, gates_in_range,
                obstacles_xyz, obstacles_in_range, gate_id].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        iteration = int(ep_time * self.CTRL_FREQ)

        # Handcrafted solution for getting_stated scenario.

        # if not self._take_off:
        #     command_type = Command.TAKEOFF
        #     args = [0.3, 2]  # Height, duration
        #     self._take_off = True  # Only send takeoff command once
        # else:
        step = iteration - 1 * self.CTRL_FREQ  # Account for 2s delay due to takeoff
        #     if ep_time - 2 > 0 and step < len(self.ref_x):
        target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
        #print(f"Step: {step}, Target: {target_pos}")
        print(f"Current position: {obs[0], obs[2], obs[4]}")
        target_vel = np.zeros(3)
        target_acc = np.ones(3) * 0.5
        target_yaw = 0.0
        target_rpy_rates = np.zeros(3)
        command_type = Command.FULLSTATE
        args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            # elif step >= len(self.ref_x) and not self._setpoint_land:
            #     command_type = Command.NOTIFY
            #     args = []
            #     self._setpoint_land = True
            # elif step >= len(self.ref_x) and not self._land:
            #     command_type = Command.LAND
            #     args = [0.0, 2.0]  # Height, duration
            #     self._land = True  # Send landing command only once
            # else:
            #     command_type = Command.NONE
            #     args = []

        return command_type, args
