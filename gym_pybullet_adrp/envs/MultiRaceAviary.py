"""Drone Racing for multiple drones on one race track"""

import os

import numpy as np
import pybullet as pb
from munch import Munch
from gymnasium import spaces
from PIL import Image
from scipy.spatial.transform import Rotation as R

from gym_pybullet_adrp.envs.BaseAviary import BaseAviary
from gym_pybullet_adrp.control import MellingerControl, DSLPIDControl
from gym_pybullet_adrp.utils.enums import \
    DroneModel, Physics, ActionType, ObservationType, ImageType, Command
from gym_pybullet_adrp.utils.constants import *


PHYSICS_WITH_KIN_INFO = [
    Physics.DYN,
    Physics.PYB_GND,
    Physics.PYB_DRAG,
    Physics.PYB_DW,
    Physics.PYB_GND_DRAG_DW
]


class MultiRaceAviary(BaseAviary):
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
        drone_collisions: bool=False,
        obs: ObservationType=ObservationType.KIN,
        act: ActionType=ActionType.PID
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
        self.config = race_config
        self.observation_type = obs
        self.action_type = act
        self.drone_collisions = drone_collisions
        self.gates_urdf, self.obstacles_urdf = [], []
        self.gates_nominal, self.obstacles_nominal = [], []
        self.gates_actual, self.obstacles_actual = [], []
        self.num_gates = len(self.config.gates)
        self.current_gate = np.zeros(num_drones)

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
        )

        assert drone_model in [DroneModel.CF2X, DroneModel.CF2P], \
            f"DroneModel {drone_model} not supported in MultiRaceAviary!"

        self.ctrl = [MellingerControl(i, DroneModel.CF2X) for i in range(num_drones)]
        # self.ctrl = [DSLPIDControl(DroneModel.CF2X) for i in range(num_drones)]

        # assert self.ctrl[0].firm is not self.ctrl[1].firm, \
        #     "Controllers are the same!"

        self.env_bounds = np.array([3, 3, 2]) # as stated in drone racing paper
        self.drones_eliminated = np.zeros(num_drones, dtype=bool)
        self.action_scale = np.array([1, 1, 1, np.pi])
        self.last_clipped_action = np.zeros((num_drones, 4))
        self.step_counter = 0
        self.previous_pos = np.zeros((num_drones, 3))
        self.previous_rpy = np.zeros((num_drones, 3))
        self.previous_vel = np.zeros((num_drones, 3))

###############################################################################

    def reset(self, seed : int = None, options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of
            `_computeObs()` in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific
            implementation of `_computeInfo()` in each subclass for its format.
        """
        initial_obs, initial_info = super().reset(seed, options)
        self._build_racetrack()
        self.current_gate = np.zeros(self.NUM_DRONES)
        initial_obs = self._computeObs()

        # reset mellinger controllers
        for mellinger in self.ctrl:
            mellinger.reset(initial_obs, initial_info)

        self.drones_eliminated = np.array([False] * self.NUM_DRONES)
        self.step_counter = 0
        self.previous_pos = initial_obs[:, :3]
        self.previous_rpy = initial_obs[:, 3:6]
        self.previous_vel = initial_obs[:, 6:9]

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
        ndarray | dict[..]
            The step's observation, check the specific implementation of
            `_computeObs()` in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of
            `_computeReward()` in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific
            implementation of `_computeTerminated()` in each subclass for its
            format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific
            implementation of `_computeTruncated()` in each subclass for its
            format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific
            implementation of `_computeInfo()` in each subclass for its format.

        """
        # # Save PNG video frames if RECORD=True and GUI=False
        # if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
        #     [w, h, rgb, dep, seg] = pb.getCameraImage(
        #         width=self.VID_WIDTH,
        #         height=self.VID_HEIGHT,
        #         shadow=1,
        #         viewMatrix=self.CAM_VIEW,
        #         projectionMatrix=self.CAM_PRO,
        #         renderer=pb.ER_TINY_RENDERER,
        #         flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        #         physicsClientId=self.CLIENT
        #     )
        #     (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
        #     #### Save the depth or segmentation view instead #######
        #     # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
        #     # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
        #     # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
        #     # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
        #     self.FRAME_NUM += 1
        #     if self.VISION_ATTR:
        #         for i in range(self.NUM_DRONES):
        #             self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
        #             #### Printing observation to PNG frames example ############
        #             self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
        #                             img_input=self.rgb[i],
        #                             path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
        #                             frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
        #                             )

        # # Read the GUI's input parameters
        # if self.GUI and self.USER_DEBUG:
        #     current_input_switch = pb.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
        #     if current_input_switch > self.last_input_switch:
        #         self.last_input_switch = current_input_switch
        #         self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False

        # if self.USE_GUI_RPM:
        #     for i in range(4):
        #         self.gui_input[i] = pb.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
        #     clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
        #     if self.step_counter%(self.PYB_FREQ/2) == 0:
        #         self.GUI_INPUT_TEXT = [pb.addUserDebugText("Using GUI RPM",
        #                                                   textPosition=[0, 0, 0],
        #                                                   textColorRGB=[1, 0, 0],
        #                                                   lifeTime=1,
        #                                                   textSize=2,
        #                                                   parentObjectUniqueId=self.DRONE_IDS[i],
        #                                                   parentLinkIndex=-1,
        #                                                   replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
        #                                                   physicsClientId=self.CLIENT
        #                                                   ) for i in range(self.NUM_DRONES)]

        # Save, preprocess, and clip the action to the max. RPM
        if isinstance(action, np.ndarray):
            action = [(
                Command.FULLSTATE,
                (act[:3], VEC3_ZERO, VEC3_ZERO, act[3], VEC3_ZERO, self.step_counter)
            ) for act in action]

        print(action)
        self._send_commands(action)

        # Repeat for as many as the aggregate physics steps
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # Update and store the drones kinematic info for certain
            # Between aggregate steps for certain types of update
            if (
                self.PYB_STEPS_PER_CTRL > 1
                and
                self.PHYSICS in PHYSICS_WITH_KIN_INFO
            ):
                self._updateAndStoreKinematicInformation()

            clipped_action = np.zeros((self.NUM_DRONES, 4))
            # update the state of mellinger controller
            for i, drone in enumerate(self.ctrl):
                clipped_action[i] = drone.computeControl(
                    self.step_counter,
                    self.previous_pos[i],
                    self.previous_rpy[i],
                    self.previous_vel[i],
                    VEC3_ZERO,
                    VEC3_ZERO
                )[0]

            # TODO debugging
            print(clipped_action)

            # Step the simulation using the desired physics update
            self._apply_physics(clipped_action)

            # Save the last applied action (e.g. to compute drag)
            self.last_clipped_action = clipped_action

        # Update and store the drones kinematic information
        self._updateAndStoreKinematicInformation()

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
        self.previous_pos = obs[:, :3]
        self.previous_rpy = obs[:, 3:6]
        self.previous_vel = obs[:, 6:9]

        # TODO debugging
        # for c in self.ctrl:
        #     print(c.control)
        #     print(c.command_queue)
        #     print(id(c.firm))
        return obs, reward, terminated, truncated, info

###############################################################################

    def _actionSpace(self):
        # positional movement: x, y, z, yaw (absolute / relative)
        act_lower = np.array([-1*np.ones(4) for _ in range(self.NUM_DRONES)])
        act_upper = np.array([+1*np.ones(4) for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower, high=act_upper, dtype=np.float32)

###############################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12+37)
            depending on the observation type.

        """
        if self.observation_type == ObservationType.RGB:
            return spaces.Box(
                low=0, high=255, dtype=np.uint8,
                shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4),
            )

        if self.observation_type == ObservationType.KIN:
            # Drone observations: X, Y, Z, R, P, Y, VX, VY, VZ, WX, WY, WZ
            dro_lower = -10*np.ones((self.NUM_DRONES, 12))
            dro_upper = 10*np.ones((self.NUM_DRONES, 12))

            # Add obstacles to observation space
            obs_lower = np.concatenate([
                -10 * np.ones((4, 4)).flatten(), # gate poses
                0   * np.ones(4), # gates in range
                -10 * np.ones((4, 3)).flatten(), # obstacle poses
                0   * np.ones(4), # obstacles in range
                0   * np.ones(1) # current gate id
            ])
            obs_upper = np.concatenate([
                10 * np.ones((4, 4)).flatten(), # gate poses
                1  * np.ones(4), # gates in range
                10 * np.ones((4, 3)).flatten(), # obstacle poses
                1  * np.ones(4), # obstacles in range
                4  * np.ones(1) # current gate id
            ])
            obs_lower = np.vstack([obs_lower for _ in range(self.NUM_DRONES)])
            obs_upper = np.vstack([obs_upper for _ in range(self.NUM_DRONES)])

            return spaces.Box(
                low=np.hstack([dro_lower, obs_lower]),
                high=np.hstack([dro_upper, obs_upper]),
                dtype=np.float32
            )

###############################################################################

    def _build_racetrack(self):
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
            g_info = self.config.random_gates_obstacles.gates
            g_distrib = getattr(np.random, g_info.distrib)
            g_low, g_high = g_info.range
            g_offsets = g_distrib(g_low, g_high, size=(num_gates, 3))
            for n, o in zip(self.gates_nominal, g_offsets):
                temp = np.array(n)
                temp[[0, 1, 5]] += o
                self.gates_actual.append(temp.tolist())

            # randomly offset position of obstacles
            o_info = self.config.random_gates_obstacles.obstacles
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
        for g in self.config.gates:
            self.gates_urdf.append(pb.loadURDF(
                URDF_DIR + ("low_portal.urdf" if g[-1] > 0 else "portal.urdf"),
                g[:3],
                pb.getQuaternionFromEuler(g[3:6]),
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

    def _gate_progress(self, drone_id: int):
        current_gate = int(self.current_gate[drone_id])

        if (
            # self.pyb_step_counter > 0.5 * self.PYB_FREQ and
            self.num_gates > 0 and
            current_gate < self.num_gates
        ):
            x, y, _, _, _, rot = self.gates_actual[current_gate]
            if self.config.gates[current_gate][6] == 0: # TODO
                height = Z_HIGH  # URDF dependent.
            elif self.config.gates[current_gate][6] == 1:
                height = Z_LOW  # URDF dependent.
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
                rayFromPositions=fr, rayToPositions=to, physicsClientId=self.CLIENT
            )
            if any(r[2] < 0.9999 for r in rays):
                self.current_gate[drone_id] += 1

###############################################################################

    def _apply_physics(self, action):
        """Apply pybullet physics to the drones for one step."""
        for i in range (self.NUM_DRONES):
            if self.PHYSICS == Physics.PYB:
                self._physics(action[i, :], i)
            elif self.PHYSICS == Physics.DYN:
                self._dynamics(action[i, :], i)
            elif self.PHYSICS == Physics.PYB_GND:
                self._physics(action[i, :], i)
                self._groundEffect(action[i, :], i)
            elif self.PHYSICS == Physics.PYB_DRAG:
                self._physics(action[i, :], i)
                self._drag(self.last_clipped_action[i, :], i)
            elif self.PHYSICS == Physics.PYB_DW:
                self._physics(action[i, :], i)
                self._downwash(i)
            elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                self._physics(action[i, :], i)
                self._groundEffect(action[i, :], i)
                self._drag(self.last_clipped_action[i, :], i)
                self._downwash(i)
        #### PyBullet computes the new state, unless Physics.DYN ###
        if self.PHYSICS != Physics.DYN:
            pb.stepSimulation(physicsClientId=self.CLIENT)

###############################################################################

    def _send_commands(self, action):
        for mellinger, (command, args) in zip(self.ctrl, action):
            if command == Command.FULLSTATE:
                mellinger.sendFullStateCmd(*args)
            elif command == Command.TAKEOFF:
                mellinger.sendTakeoffCmd(*args)
            elif command == Command.TAKEOFFYAW:
                mellinger.sendTakeoffYawCmd(*args)
            elif command == Command.TAKEOFFVEL:
                mellinger.sendTakeoffVelCmd(*args)
            elif command == Command.LAND:
                mellinger.sendLandCmd(*args)
            elif command == Command.LANDYAW:
                mellinger.sendLandYawCmd(*args)
            elif command == Command.LANDVEL:
                mellinger.sendLandVelCmd(*args)
            elif command == Command.GOTO:
                mellinger.sendGotoCmd(*args)
            elif command == Command.STOP:
                mellinger.sendStopCmd(*args)
            elif command == Command.NOTIFY:
                mellinger.notifySetpointStop(*args)

            mellinger._process_command_queue(args[-1])

###############################################################################
#TODO
    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(self.NUM_DRONES):
            target = action[k, :]

            if self.action_type == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))

            elif self.action_type == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                )
                rpm_k, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_rpy=state[7:10],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=next_pos
                )
                rpm[k,:] = rpm_k

            elif self.action_type == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_rpy=state[3:6],
                    cur_vel=state[6:9],
                    cur_ang_vel=state[9:12],
                    target_pos=state[0:3], # same as the current position
                    target_rpy=np.array([0,0,state[6]]), # keep current yaw
                    target_vel=SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                )
                rpm[k,:] = temp

        return rpm

###############################################################################

    def _collision(self, drone_id: int):
        objects = self.gates_urdf + self.obstacles_urdf + [self.PLANE_ID]
        if self.drone_collisions:
            objects += [d for i, d in enumerate(self.DRONE_IDS) if i != drone_id]

        print(objects)
        for obj_id in objects:
            # NOTE: only returning the first collision per step
            if pb.getContactPoints(
                bodyA=obj_id,
                bodyB=self.DRONE_IDS[drone_id],
                physicsClientId=self.CLIENT,
            ):
                print(f"collided with {obj_id}")
                return obj_id

        return None

###############################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12+37) depending
            on the observation type.
        """
        if self.observation_type == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = \
                        self._getDroneImages(i, segmentation=False)

                    # Printing observation to PNG frames example
                    if self.RECORD:
                        self._exportImage(
                            img_type=ImageType.RGB,
                            img_input=self.rgb[i],
                            path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                            frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                        )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')

        if self.observation_type == ObservationType.KIN:
            # observation space of size (NUM_DRONES, 49)
            obs_49 = np.zeros((self.NUM_DRONES, 49))

            # NOTE: for compatibility with BaseAviary
            if not self.gates_urdf:
                return obs_49

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
                obs_49[i, :12] = drone.reshape(12,)
                obs_49[i, 12:28] = gate_poses.flatten()
                obs_49[i, 28:32] = gate_range
                obs_49[i, 32:44] = obstacles_poses.flatten()
                obs_49[i, 44:48] = obstacles_range
                obs_49[i, -1] = self.current_gate[i]

            return obs_49.astype(np.float32)

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
            print("Unstable: ", np.abs(state[13:16]))
            # unstable = np.any(np.abs(state[13:16]) > 20) # TODO replace arbitrary theshold
            unstable = False
            crashed = self._collision(i) is not None

            # TODO debugging
            print(f"{out_of_bounds = }, {unstable = }, {crashed = }")

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
