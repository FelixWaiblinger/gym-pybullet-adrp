"""Drone Racing for multiple drones on one race track"""

import os

import numpy as np
import pybullet as pb
from munch import Munch
from gymnasium import spaces
from PIL import Image

from gym_pybullet_adrp.envs.BaseAviary import BaseAviary
from gym_pybullet_adrp.control import MellingerControl
from gym_pybullet_adrp.utils.enums import \
    DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_adrp.utils.constants import \
    FIRMWARE_FREQ, CTRL_FREQ, DEG_TO_RAD, URDF_DIR, Z_LOW, Z_HIGH

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
            record=record
        )

        assert drone_model in [DroneModel.CF2X, DroneModel.CF2P], \
            f"DroneModel {drone_model} not supported in MultiRaceAviary!"

        self.ctrl = [MellingerControl(i, DroneModel.CF2X) for i in range(num_drones)]

        assert self.ctrl[0].firm is not self.ctrl[1].firm, \
            "Controllers are the same!"

        self.config = race_config
        self.observation_type = obs
        self.action_type = act
        self.env_bounds = np.array([3, 3, 2]) # as stated in drone racing paper
        self.drones_eliminated = None
        self.current_gate = None

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
        self.drones_eliminated = np.array([False] * self.NUM_DRONES)

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
        # Save PNG video frames if RECORD=True and GUI=False
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = pb.getCameraImage(
                width=self.VID_WIDTH,
                height=self.VID_HEIGHT,
                shadow=1,
                viewMatrix=self.CAM_VIEW,
                projectionMatrix=self.CAM_PRO,
                renderer=pb.ER_TINY_RENDERER,
                flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.CLIENT
            )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )
        
        # Read the GUI's input parameters
        if self.GUI and self.USER_DEBUG:
            current_input_switch = pb.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = pb.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [pb.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]

        # Save, preprocess, and clip the action to the max. RPM
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
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

            # update the state of mellinger controller
            for drone in self.ctrl:
                drone.computeControl() # TODO

            # Step the simulation using the desired physics update
            self._apply_physics(clipped_action)

            # Save the last applied action (e.g. to compute drag)
            self.last_clipped_action = clipped_action

        # Update and store the drones kinematic information
        self._updateAndStoreKinematicInformation()

        # Prepare the return values
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        # Advance the step counter
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info

###############################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4 representing target position and yaw.

        """
        size = 4 # x, y, z, yaw
        act_lower_bound = np.array([-1*np.ones(size) for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for _ in range(self.NUM_DRONES)])
        #
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

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
                -10*np.ones((4, 4)).flatten(), # gate poses
                0*np.ones(4), # gates in range
                -10*np.ones((4, 3)).flatten(), # obstacle poses
                0*np.ones(4), # obstacles in range
                np.ones(1) # current gate id
            ])
            obs_upper = np.concatenate([
                10*np.ones((4, 4)).flatten(), # gate poses
                1*np.ones(4), # gates in range
                10*np.ones((4, 3)).flatten(), # obstacle poses
                1*np.ones(4), # obstacles in range
                4*np.ones(1) # current gate id
            ])
            obs_lower = np.vstack([obs_lower for _ in range(self.NUM_DRONES)])
            obs_upper = np.vstack([obs_upper for _ in range(self.NUM_DRONES)])

            return spaces.Box(
                low=np.hstack([dro_lower, obs_lower]),
                high=np.hstack([dro_upper, obs_lower]),
                dtype=np.float32
            )

        # unrecognized observation type
        print("[ERROR] in BaseRLAviary._observationSpace()")

###############################################################################

    def _build_racetrack(self):
        """Add gates and obstacles to the environment.

        Overrides BaseAviary's method.
        """
        self.gates, self.obstacles = [], []
        self.NUM_GATES = len(self.config.gates)

        gate_init = np.array(self.config.gates)
        for g in gate_init:
            self.gates.append(pb.loadURDF(
                URDF_DIR + ("low_portal.urdf" if g[-1] > 0 else "portal.urdf"),
                g[:3],
                pb.getQuaternionFromEuler(g[3:6]),
                physicsClientId=self.CLIENT
            ))

        obstacle_init = np.array(self.config.obstacles)
        for o in obstacle_init:
            self.obstacles.append(pb.loadURDF(
                URDF_DIR + "obstacle.urdf",
                o[:3],
                pb.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
            ))

###############################################################################

    def _gate_progress(self, drone_id: int):
        current_gate = self.current_gate[drone_id]

        if (
            self.pyb_step_counter > 0.5 * self.PYB_FREQ
            and self.NUM_GATES > 0
            and self.current_gate < self.NUM_GATES
        ):
            x, y, _, _, _, rot = self.config.gates[current_gate]
            if self.gates[current_gate][6] == 0:
                height = Z_HIGH  # URDF dependent.
            elif self.gates[self.current_gate][6] == 1:
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
            self.stepped_through_gate = False
            for r in rays:
                if r[2] < 0.9999:
                    self.current_gate += 1
                    self.stepped_through_gate = True
                    break

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
#TODO
    def _preprocessAction(self,
                          action
                          ):
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
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

###############################################################################

    def _collision(self, drone_id: int):
        self.currently_collided = False

        for obj_id in self.gates + self.obstacles + [self.PLANE_ID]:
            ret = pb.getContactPoints(
                bodyA=obj_id,
                bodyB=drone_id,
                physicsClientId=self.CLIENT,
            )
            if ret:
                self.currently_collided = True
                # NOTE: only returning the first collision per step
                return obj_id

        return None

###############################################################################
#TODO
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.
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

        elif self.observation_type == ObservationType.KIN:
            # OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

            return ret

        else:
            print("[ERROR] in BaseRLAviary._computeObs()")

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
            crashed = self._collision(i) is not None

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
