"""Drone Racing for multiple drones on one race track"""

import multiprocessing as mp

import numpy as np
import pybullet as pb
import xml.etree.ElementTree as etxml
from munch import Munch
from gymnasium import spaces

from gym_pybullet_adrp.envs.BaseAviary import BaseAviary
from gym_pybullet_adrp.control import low_level_control
from gym_pybullet_adrp.utils.constants import *
from gym_pybullet_adrp.utils.enums import *


KIN_PHYSICS = [
    Physics.DYN,
    Physics.PYB_GND,
    Physics.PYB_DRAG,
    Physics.PYB_DW,
    Physics.PYB_GND_DRAG_DW
]


class MultiRaceAviary(BaseAviary):
    """Multi-agent RL problem: head-to-head race."""

    ################################################################################

    def __init__(
        self,
        race_config: Munch,
        drone_model: DroneModel=DroneModel.CF2X,
        num_drones: int=2,
        physics: Physics=Physics.PYB,
        pyb_freq: int=FIRMWARE_FREQ,
        ctrl_freq: int=CTRL_FREQ,
        gui: bool=False,
        record: bool=False,
        racemode: RaceMode=RaceMode.COMPARE,
        obs: ObservationType=ObservationType.KIN,
        act: ActionType=ActionType.PID
    ):
        """Initialization of a multi-agent RL environment.

        Using the generic base aviary superclass.

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
        self.config = race_config
        self.observation_type = obs
        self.action_type = act
        self.racemode = racemode
        self.gates_urdf, self.obstacles_urdf = [], []
        self.gates_nominal, self.obstacles_nominal = [], []
        self.gates_actual, self.obstacles_actual = [], []
        self.collision_objects = []
        self.num_gates = len(self.config.gates)
        self.current_gate = np.zeros(num_drones)
        self.env_bounds = np.array(self.config.bounds) # NOTE: see config
        self.done_on_completion = self.config.done_on_completion

        drones = [d for d in self.config.init_states]
        xyzs = [getattr(self.config.init_states, d).pos for d in drones]
        rpys = [getattr(self.config.init_states, d).rpy for d in drones]

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=np.array(xyzs[:num_drones]),
            initial_rpys=np.array(rpys[:num_drones]) * DEG_TO_RAD,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            obstacles=True,
            record=record
        )

        assert drone_model in [DroneModel.CF2X, DroneModel.CF2P], \
            f"DroneModel {drone_model} not supported in MultiRaceAviary!"

        self.ctrl = []
        for drone_id in range(num_drones):
            parent_conn, child_conn = mp.Pipe()
            process = mp.Process(
                target=low_level_control,
                args=(drone_id, child_conn)
            )
            process.start()
            self.ctrl.append((process, parent_conn))

        self.action_scale = np.array([1, 1, 1, np.pi])

        self.step_counter = 0
        self.rpms = np.zeros((self.NUM_DRONES, 4))
        self.prev_rpms = np.zeros((self.NUM_DRONES, 4))
        self.drones_eliminated = np.zeros(num_drones, dtype=bool)
        self.drones_finished = np.zeros(num_drones, dtype=bool)

###############################################################################

    def reset(self, seed: int=None, options: dict=None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray: Initial observations according to `_computeObs()`
        dict: Initial additional information according to `_computeInfo()`
        """
        self.current_gate = np.zeros(self.NUM_DRONES)
        initial_obs, initial_info = super().reset(seed, options)
        self._drone_init()

        # reset mellinger controllers
        for _, connection in self.ctrl:
            command = ("reset", (initial_obs, initial_info))
            connection.send(command)

        self.collision_objects = self.gates_urdf + self.obstacles_urdf + [self.PLANE_ID]
        # add drones to collision objects to detect elimination from race
        if self.racemode == RaceMode.COMPETE:
            self.collision_objects += self.DRONE_IDS.tolist()
        # disable drone collisions
        else:
            for i in self.DRONE_IDS:
                for j in self.DRONE_IDS:
                    pb.setCollisionFilterPair(i, j, -1, -1, 0)

        self.drones_eliminated = np.zeros(self.NUM_DRONES, dtype=bool)
        self.drones_finished = np.zeros(self.NUM_DRONES, dtype=bool)
        self.step_counter = 0
        self.rpms = np.zeros((self.NUM_DRONES, 4))
        self.prev_rpms = np.zeros((self.NUM_DRONES, 4))

        return initial_obs, initial_info

###############################################################################

    def step(self, action):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray: Observation according to `_computeObs()`
        float: Reward value(s) according to `_computeReward()`
        bool: Whether the current episode is over according to `_computeTerminated()`
        bool: Whether the current episode is truncated according to `_computeTruncated()`
        dict: Additional information as a dictionary according to `_computeInfo()`
        """

        # convert numpy actions to fullstate commands
        if isinstance(action, np.ndarray):
            action = [(
                Command.FULLSTATE,
                (act[:3], ZERO3, ZERO3, act[3], ZERO3, self.step_counter)
            ) for act in action]

        # send high-level commands to each drone (e.g. FULLSTATE + args)
        for (_, connection), (cmd, args) in zip(self.ctrl, action):
            command = ("command", (cmd, args))
            connection.send(command)
            # do not wait for ok if "no" command has been sent
            if cmd != Command.NONE and connection.recv() != "ok":
                raise RuntimeError("SYNCHRONIZATION ERROR")

        # Repeat for as many as the aggregate physics steps
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # Update and store the drones kinematic info for certain
            if (self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in KIN_PHYSICS):
                self._updateAndStoreKinematicInformation()

            # Step the simulation using the desired physics update
            # TODO: add disturbances
            self._apply_physics(self.rpms, self.prev_rpms)

            # Update and store the drones kinematic information
            self._updateAndStoreKinematicInformation()

            obs = self._computeObs()

            # update the state of mellinger controller
            for i, (_, connection) in enumerate(self.ctrl):
                command = (
                    "step",
                    (
                        self.step_counter,
                        obs[i, :3], # pos
                        obs[i, 3:6], # rpy
                        obs[i, 6:9], # vel
                        ZERO3,
                        ZERO3
                    )
                )
                connection.send(command)

                # Save the last applied action (e.g. to compute drag)
                self.prev_rpms[i] = self.rpms[i]

                # get motor rpms from mellinger controllers
                self.rpms[i] = connection.recv()


        # Track gate progress
        for i in range(self.NUM_DRONES):
            self._gate_progress(i)

        # Prepare the return values
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        # Advance the step counter
        self.step_counter += self.PYB_STEPS_PER_CTRL

        return obs, reward, terminated, truncated, info

###############################################################################

    def close(self):
        """Terminates the environment."""
        super().close()
        for p, c in self.ctrl:
            c.send(("close", None)) # close controller processes
            c.close() # close connection from this side
            p.join() # wait for processes to be stopped

###############################################################################

    def _actionSpace(self):
        """Returns the action space of the environment."""
        act_limit = np.ones((self.NUM_DRONES, 4))
        return spaces.Box(low=-1 * act_limit, high=act_limit, dtype=float)

###############################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray: A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12+37)
            depending on the observation type.

        """
        if self.observation_type == ObservationType.RGB:
            return spaces.Box(
                low=0, high=255, dtype=np.uint8,
                shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4),
            )

        if self.observation_type == ObservationType.KIN:
            # lower observation bounds
            obs_lower = np.concatenate([
                [-5] * 3,                   # agent_pos_low
                [-np.pi] * 3,               # agent_rot_low
                [-10] * 3,                  # agent_vel_low
                [-10] * 3,                  # agent_ang_low
                [-5, -5, -5, -np.pi] * 4,   # gate_pose_low
                [-1] * 4,                   # gate_range_low
                [-5] * 3 * 4,               # obst_pose_low
                [-1] * 4,                   # obst_range_low
                [-1],                       # gate_id_low
            ])

            # upper observation bounds
            obs_upper = np.concatenate([
                [5] * 3,                    # agent_pos_high
                [np.pi] * 3,                # agent_rot_high
                [10] * 3,                   # agent_vel_high
                [10] * 3,                   # agent_ang_high
                [5, 5, 5, np.pi] * 4,       # gate_pose_high
                [1] * 4,                    # gate_range_high
                [5] * 3 * 4,                # obst_pose_high
                [1] * 4,                    # obst_range_high
                [4],                        # gate_id_high
            ])

            if self.racemode == RaceMode.COMPETE:
                # position and rotation of other drones
                others_low = ([-5] * 3 + [-np.pi] * 3) * (self.NUM_DRONES - 1)
                others_high = ([5] * 3 + [np.pi] * 3) * (self.NUM_DRONES - 1)
                obs_lower = np.concatenate([obs_lower, others_low])
                obs_upper = np.concatenate([obs_upper, others_high])

            obs_lower = np.vstack([obs_lower for _ in range(self.NUM_DRONES)])
            obs_upper = np.vstack([obs_upper for _ in range(self.NUM_DRONES)])

            return spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float64)

###############################################################################

    def _addObstacles(self):
        """Add gates and obstacles to the environment.

        Overrides BaseAviary's method.
        """
        self.gates_urdf, self.gates_actual = [], []
        self.obstacles_urdf, self.obstacles_actual = [], []
        self.gates_nominal = [g[:6] for g in self.config.gates]
        self.obstacles_nominal = self.config.obstacles
        num_gates = len(self.gates_nominal)
        num_obstacles = len(self.obstacles_nominal)

        if self.config.random_gates_obstacles:
            # randomly offset position and yaw of gates
            g_info = self.config.random_gates_obstacles_info.gates
            g_distrib = getattr(np.random, g_info.distrib)
            g_low, g_high = g_info.range
            g_offsets = g_distrib(g_low, g_high, size=(num_gates, 3))

            for n, o in zip(self.gates_nominal, g_offsets):
                temp = np.array(n)
                temp[[0, 1, 5]] += o
                self.gates_actual.append(temp.tolist())

            # randomly offset position of obstacles
            o_info = self.config.random_gates_obstacles_info.obstacles
            o_distrib = getattr(np.random, o_info.distrib)
            o_low, o_high = o_info.range
            o_offsets = o_distrib(o_low, o_high, size=(num_obstacles, 2))

            for n, o in zip(self.obstacles_nominal, o_offsets):
                temp = np.array(n)
                temp[[0, 1]] += o
                self.obstacles_actual.append(temp.tolist())

        # no randomization of gates and obstacles
        else:
            self.gates_actual = self.gates_nominal
            self.obstacles_actual = self.obstacles_nominal

        # spawn gates
        for pose, g in zip(self.gates_actual, self.config.gates):
            self.gates_urdf.append(pb.loadURDF(
                URDF_DIR + ("low_portal.urdf" if g[-1] > 0 else "portal.urdf"),
                pose[:3],
                pb.getQuaternionFromEuler(pose[3:6]),
                physicsClientId=self.CLIENT
            ))

        # spawn obstacles
        for o in self.obstacles_actual:
            self.obstacles_urdf.append(pb.loadURDF(
                URDF_DIR + "obstacle.urdf",
                o[:3],
                pb.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
            ))

###############################################################################

    def _drone_init(self):
        """TODO"""
        URDF_TREE = etxml.parse(URDF_DIR + "cf2x.urdf").getroot()
        drones = [d for d in self.config.init_states]
        properties = {
            "M": float(URDF_TREE[1][0][1].attrib['value']),
            "Ixx": float(URDF_TREE[1][0][2].attrib['ixx']),
            "Iyy": float(URDF_TREE[1][0][2].attrib['iyy']),
            "Izz": float(URDF_TREE[1][0][2].attrib['izz']),
        }

        for i, drone in enumerate(self.DRONE_IDS):
            if self.config.random_drone_inertia:
                for key in properties:
                    bla = getattr(self.config.random_drone_inertia_info, key)
                    distrib = getattr(self.np_random, bla.distrib)
                    offset = distrib(*bla.range)
                    properties[key] = np.clip(properties[key] + offset, 0, 100)

            pb.changeDynamics(
                drone,
                linkIndex=-1,  # Base link.
                mass=properties["M"],
                localInertiaDiagonal=[properties["Ixx"], properties["Iyy"], properties["Izz"]],
                physicsClientId=self.CLIENT,
            )

            if self.config.random_drone_state:
                position = self.config.random_drone_state_info.pos
                rotation = self.config.random_drone_state_info.rot

                pos_distrib = getattr(self.np_random, position.distrib)
                rot_distrib = getattr(self.np_random, rotation.distrib)

                pos_offset = np.array([
                    pos_distrib(*position.x),
                    pos_distrib(*position.y),
                    pos_distrib(*position.z),
                ])
                rot_offset = np.array([
                    rot_distrib(*rotation.r),
                    rot_distrib(*rotation.p),
                    rot_distrib(*rotation.y),
                ])
            else:
                pos_offset, rot_offset = ZERO3, ZERO3

            state = getattr(self.config.init_states, drones[i])

            pb.resetBasePositionAndOrientation(
                drone,
                state.pos + pos_offset,
                pb.getQuaternionFromEuler(state.rpy + rot_offset),
                physicsClientId=self.CLIENT,
            )
            pb.resetBaseVelocity(
                drone,
                state.vel,
                state.pqr,
                physicsClientId=self.CLIENT
            )

###############################################################################

    def _gate_progress(self, drone_id: int):
        gate = int(self.current_gate[drone_id])

        if (self.num_gates > 0 and gate < self.num_gates):
            x, y, _, _, _, rot = self.gates_actual[gate]

            if self.config.gates[gate][6] == 0:
                height = 1.0 #Z_HIGH  # URDF dependent.
            elif self.config.gates[gate][6] == 1:
                height = 0.525 #Z_LOW  # URDF dependent.
            else:
                raise ValueError("Unknown gate type.")

            half_length = 0.1875  # Obstacle URDF dependent.
            delta_x = 0.05 * np.cos(rot)
            delta_y = 0.05 * np.sin(rot)
            fr = [[x, y, height - half_length]]
            to = [[x, y, height + half_length]]

            for i in [1, 2, 3]:
                fr.append([x + i * delta_x, y + i * delta_y, height - half_length])
                fr.append([x - i * delta_x, y - i * delta_y, height - half_length])
                to.append([x + i * delta_x, y + i * delta_y, height + half_length])
                to.append([x - i * delta_x, y - i * delta_y, height + half_length])

            rays = pb.rayTestBatch(
                rayFromPositions=fr,
                rayToPositions=to,
                physicsClientId=self.CLIENT
            )

            if any(r[2] < 0.9999 and r[0] == self.DRONE_IDS[drone_id] for r in rays):
                self.current_gate[drone_id] += 1

        if gate >= self.num_gates:
            self.drones_finished[drone_id] = True

###############################################################################

    def _apply_physics(self, action, prev_action, disturbance=None):
        """Apply pybullet physics to the drones for one step."""
        for i in range(self.NUM_DRONES):
            if self.PHYSICS == Physics.PYB:
                self._physics(action[i, :], i)
            elif self.PHYSICS == Physics.DYN:
                self._dynamics(action[i, :], i)
            elif self.PHYSICS == Physics.PYB_GND:
                self._physics(action[i, :], i)
                self._groundEffect(action[i, :], i)
            elif self.PHYSICS == Physics.PYB_DRAG:
                self._physics(action[i, :], i)
                self._drag(prev_action[i, :], i)
            elif self.PHYSICS == Physics.PYB_DW:
                self._physics(action[i, :], i)
                self._downwash(i)
            elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                self._physics(action[i, :], i)
                self._groundEffect(action[i, :], i)
                self._drag(prev_action[i, :], i)
                self._downwash(i)

            if disturbance is not None:
                pos = self._getDroneStateVector(i)[:3]
                pb.applyExternalForce(
                    self.DRONE_IDS[i],
                    linkIndex=4,  # Link attached to the quadrotor's center of mass.
                    forceObj=disturbance,
                    posObj=pos,
                    flags=pb.WORLD_FRAME,
                    physicsClientId=self.CLIENT)

        #### PyBullet computes the new state, unless Physics.DYN ###
        if self.PHYSICS != Physics.DYN:
            pb.stepSimulation(physicsClientId=self.CLIENT)

###############################################################################

    def _collision(self, drone_id: int):
        for obj_id in self.collision_objects:
            # NOTE: only returning the first collision per step
            if pb.getContactPoints(
                bodyA=obj_id,
                bodyB=self.DRONE_IDS[drone_id],
                physicsClientId=self.CLIENT,
            ):
                return obj_id

        return None

###############################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray: Observations of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12+37)
            depending on the observation type.
        """
        if self.observation_type == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = \
                        self._getDroneImages(i, segmentation=False)

                    # Printing observation to PNG frames example
                    if self.RECORD:
                        self._exportImage(
                            img_type=ImageType.RGB,
                            img_input=self.rgb[i],
                            path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                            frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                        )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')

        if self.observation_type == ObservationType.KIN:
            obs_size = 49
            # position and rotation of all other drones
            if self.racemode == RaceMode.COMPETE:
                obs_size += 6 * (self.NUM_DRONES - 1)

            # observation space of size (NUM_DRONES, 49(+6*(NUM_DRONES-1)))
            all_obs = np.zeros((self.NUM_DRONES, obs_size))

            # NOTE: for compatibility with BaseAviary
            if not self.gates_urdf:
                return all_obs

            num_gates = len(self.gates_urdf)
            num_obstacles = len(self.obstacles_urdf)

            for i in range(self.NUM_DRONES):
                # drone observations
                obs = self._getDroneStateVector(i)
                drone = np.hstack([obs[:3], obs[7:10], obs[10:13], obs[13:16]])

                # gate observations
                gate_poses = np.zeros((num_gates, 4))
                gate_range = np.zeros(num_gates)
                for j, gate in enumerate(self.gates_urdf):
                    closest_points = pb.getClosestPoints(
                        bodyA=gate,
                        bodyB=self.DRONE_IDS[i],
                        distance=VISIBILITY_RANGE,
                        physicsClientId=self.CLIENT,
                    )
                    if len(closest_points) > 0:
                        gate_poses[j] = np.array(self.gates_actual)[j, [0,1,2,5]]
                        gate_range[j] = True
                    else:
                        gate_poses[j] = np.array(self.gates_nominal)[j, [0,1,2,5]]
                        gate_range[j] = False

                # obstacle observations
                obstacles_poses = np.zeros((num_obstacles, 3))
                obstacles_range = np.zeros(num_obstacles)
                for j, obstacle in enumerate(self.obstacles_urdf):
                    closest_points = pb.getClosestPoints(
                        bodyA=obstacle,
                        bodyB=self.DRONE_IDS[i],
                        distance=VISIBILITY_RANGE,
                        physicsClientId=self.CLIENT,
                    )
                    if len(closest_points) > 0:
                        obstacles_poses[j] = np.array(self.obstacles_actual[j])[:3]
                        obstacles_range[j] = True
                    else:
                        obstacles_poses[j] = np.array(self.obstacles_nominal[j])[:3]
                        obstacles_range[j] = False

                # combine drone, gate, obstacle and progress observation
                all_obs[i, :12] = drone.reshape((12,))
                all_obs[i, 12:28] = gate_poses.flatten()
                all_obs[i, 28:32] = gate_range
                all_obs[i, 32:44] = obstacles_poses.flatten()
                all_obs[i, 44:48] = obstacles_range
                all_obs[i, 48] = self.current_gate[i]

                # add other drone poses to observation
                if self.racemode == RaceMode.COMPETE:
                    others = [k for k in range(self.NUM_DRONES) if k != i]
                    for idx, j in enumerate(others):
                        obs = self._getDroneStateVector(j)
                        idx = 49 + 6 * idx
                        all_obs[i, idx:idx+6] = np.hstack([obs[:3], obs[7:10]])

            return all_obs.astype(np.float64)

###############################################################################

    def _computeReward(self):
        """Computes the current reward value.

        NOTE: Unused (should be overwritten by wrapper anyway)
        """
        return 0

###############################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool: Whether the current episode is done.
        """
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)

            out_of_bounds = np.any(np.abs(state[:3]) > self.env_bounds[1])
            # unstable = np.any(np.abs(state[13:16]) > 20) # TODO replace arbitrary theshold
            unstable = False
            crashed = self._collision(i) is not None

            # TODO debugging
            # print(f"{out_of_bounds = }, {unstable = }, {crashed = }")

            self.drones_eliminated[i] = out_of_bounds or unstable or crashed

        all_crashed = np.all(self.drones_eliminated)
        all_finished = np.all(self.drones_finished)
        crsh_or_fin = np.logical_or(self.drones_eliminated, self.drones_finished)

        # print(f"{all_crashed = }, {all_finished = }, {crsh_or_fin = }")

        return all_crashed or all_finished or np.all(crsh_or_fin)

###############################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool: Whether the current episode timed out.
        """
        return self.step_counter / self.PYB_FREQ > self.config.episode_len_sec

###############################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).
        
        NOTE: Unused
        """
        #### Calculated by the Deep Thought supercomputer in 7.5M years
        return {"answer": 42}

###############################################################################

    def _preprocessAction(self, action):
        """Not implemented in this subclass"""
        raise NotImplementedError()
