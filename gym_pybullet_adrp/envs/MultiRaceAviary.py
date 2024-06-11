"""Drone Racing for multiple drones on one race track"""

import numpy as np
import pybullet as pb
from munch import Munch

from gym_pybullet_adrp.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_adrp.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_adrp.utils.constants import CTRL_FREQ, FIRMWARE_FREQ, DEG_TO_RAD


DIR = "gym_pybullet_adrp/assets/"


class MultiRaceAviary(BaseRLAviary):
    """Multi-agent RL problem: head-to-head race."""

    ################################################################################

    def __init__(self,
        race_config: Munch,
        drone_model: DroneModel=DroneModel.CF2X,
        num_drones: int=2,
        neighbourhood_radius: float=np.inf,
        physics: Physics=Physics.PYB,
        pyb_freq: int=FIRMWARE_FREQ,
        ctrl_freq: int=CTRL_FREQ,
        gui: bool=False,
        record: bool=False,
        obs: ObservationType=ObservationType.KIN,
        act: ActionType=ActionType.VEL
    ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        race_config : Munch
            The initial configuration for the chosen race environment.
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thrust and torques, waypoint with PID control)
        """
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=np.array(race_config.init_states.pos),
            initial_rpys=np.array(race_config.init_states.rpy) * DEG_TO_RAD,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )

        self.config = race_config
        self.env_bounds = np.array([3, 3, 2]) # as stated in drone racing paper
        self.drones_eliminated = np.array([False] * num_drones)

        self._addGatesAndObstacles()

###############################################################################

    def _addGatesAndObstacles(self):
        """Add gates and obstacles to the environment.

        Overrides BaseAviary's method.
        """
        gates = np.array(self.config.gates)
        for g in gates:
            pb.loadURDF(
                DIR + ("low_portal.urdf" if g[-1] > 0 else "portal.urdf"),
                g[:3],
                pb.getQuaternionFromEuler(g[3:6]),
                physicsClientId=self.CLIENT
            )

        obstacles = np.array(self.config.obstacles)
        for o in obstacles:
            pb.loadURDF(
                DIR + "obstacle.urdf",
                o[:3],
                pb.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
            )

###############################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.
        """
        reward = 0

        # TODO: design this according to your needs
        # states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for _ in range(self.NUM_DRONES):
            pass

        return reward

###############################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.
        """
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)

            out_of_bounds = np.any(np.abs(state[:3]) > self.env_bounds)
            unstable = False # np.any(np.abs(state[13:16]) > 0.5) # TODO arbitrary theshold
            crashed = False # pb.getContactPoints() # TODO check collision with ground and obstacles

            self.drones_eliminated[i] = out_of_bounds or unstable or crashed

        return np.all(self.drones_eliminated)

###############################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.
        """
        return self.step_counter / self.PYB_FREQ > self.config.episode_len_sec

###############################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
