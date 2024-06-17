"""Mellinger Controller"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_pybullet_adrp.control import BaseControl
from gym_pybullet_adrp.utils import get_quaternion_from_euler, load_firmware
from gym_pybullet_adrp.utils.enums import DroneModel
from gym_pybullet_adrp.utils.constants import *

if TYPE_CHECKING:
    import pycffirmware as firmware


class MellingerControl(BaseControl):
    """Mellinger control class for Crazyflie drones."""

    ################################################################################

    def __init__(self,
        drone_id: int,
        drone_model: DroneModel,
        g: float=9.8
    ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        # load firmware bindings for each mellinger controller seperately
        self.firm: firmware = load_firmware("../")
        self.drone_id = drone_id

        assert drone_model in [DroneModel.CF2X, DroneModel.CF2P], \
            "[ERROR] in MellingerControl.__init__(), requires \
            DroneModel.CF2X or DroneModel.CF2P"

        self.DRONE_MODEL = drone_model
        self.GRAVITY = g*self._getURDFParameter('m')
        self._init_variables()

###############################################################################

    def reset(self, init_obs, init_info):
        """Resets the control classes.

        Copied and slightly modified from safe-control-gym.firmware_wrapper
        """
        super().reset()

        # NOTE: whole section taken from safe-control-gym.firmware_wrapper
        self.states = []
        self.takeoff_sent = False

        # Initialize history
        self.action_history = [[0, 0, 0, 0] for _ in range(ACTION_DELAY)]
        self.sensor_history = [[[0, 0, 0], [0, 0, 0]] for _ in range(SENSOR_DELAY)]
        self.state_history = []

        # Initialize gyro lpf
        self.acclpf = [self.firm.lpf2pData() for _ in range(3)]
        self.gyrolpf = [self.firm.lpf2pData() for _ in range(3)]
        for i in range(3):
            self.firm.lpf2pInit(self.acclpf[i], FIRMWARE_FREQ, GYRO_LPF_CUTOFF_FREQ)
            self.firm.lpf2pInit(self.gyrolpf[i], FIRMWARE_FREQ, ACCEL_LPF_CUTOFF_FREQ)

        # Initialize state objects
        self.control = self.firm.control_t()
        self.setpoint = self.firm.setpoint_t()
        self.sensorData = self.firm.sensorData_t()
        self.state = self.firm.state_t()
        self.tick = 0
        self.pwms = [0, 0, 0, 0]
        self.action = [0, 0, 0, 0]
        self.command_queue = []

        self.tumble_counter = 0
        self.prev_vel = np.array([0, 0, 0])
        self.prev_rpy = np.array([0, 0, 0])
        self.prev_time_s = None
        self.last_pos_pid_call = 0
        self.last_att_pid_call = 0

        # Initialize state flags
        self._error = False
        self.sensorData_set = False
        self.state_set = False
        self.full_state_cmd_override = True  # When true, high level commander is not called

        self.firm.controllerMellingerInit()
        # logger.debug("Mellinger controller init test:", firm.controllerMellingerTest())

        # observations about the drone owned by this controller
        drone = init_obs[self.drone_id, :12]

        self.firm.crtpCommanderHighLevelInit()
        self._update_state(
            0, drone[:3], drone[6:9], VEC3_UP, drone[3:6] * RAD_TO_DEG
        )

        self.prev_rpy = drone[3:6]
        self.prev_vel = drone[6:9]
        self.firm.crtpCommanderHighLevelTellState(self.state)

        # Initialize visualization tools
        self.first_motor_killed_print = True
        self.last_visualized_setpoint = None

        self.results_dict = {
            "obs": [],
            "reward": [],
            "done": [],
            "info": [],
            "action": [],
        }

###############################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_rpy,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the Mellinger control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        # Get state values from pybullet
        body_rot = R.from_euler("XYZ", cur_rpy).inv()

        if self.takeoff_sent:
            self.states += [
                [self.tick / FIRMWARE_FREQ, cur_pos[0], cur_pos[1], cur_pos[2]]
            ]

        # body coord, rad/s
        cur_rotation_rates = (cur_rpy - self.prev_rpy) / FIRMWARE_DT
        self.prev_rpy = cur_rpy

        # global coord
        cur_acc = (cur_vel - self.prev_vel) / FIRMWARE_DT / 9.8 + VEC3_UP
        self.prev_vel = cur_vel

        # Update state
        state_timestamp = int(self.tick / FIRMWARE_FREQ * 1e3)
        self._update_state(
            state_timestamp,
            cur_pos,
            cur_vel,
            cur_acc,
            cur_rpy * RAD_TO_DEG,
        )

        # Update sensor data
        sensor_timestamp = int(self.tick / FIRMWARE_FREQ * 1e6)
        if SENSOR_DELAY:
            self._update_sensorData(sensor_timestamp, *self.sensor_history[0])
            self.sensor_history = self.sensor_history[1:] + [
                [body_rot.apply(cur_acc), cur_rotation_rates * RAD_TO_DEG]
            ]
        else:
            self._update_sensorData(
                sensor_timestamp,
                body_rot.apply(cur_acc),
                cur_rotation_rates * RAD_TO_DEG,
            )

        # Update setpoint
        self._updateSetpoint(self.tick / FIRMWARE_FREQ)  # setpoint looks right

        # Step controller
        self._step_controller()
        self.control_counter += 1

        clipped_pwms = np.clip(np.array(self.pwms), MIN_PWM, MAX_PWM)
        rpms = self.KF * (PWM2RPM_SCALE * clipped_pwms + PWM2RPM_CONST) ** 2
        rpms = rpms[[3, 2, 1, 0]]

        return rpms, None, None

###############################################################################

# region CONTROLLER

    def _init_variables(self):
        # self.KF = self._getURDFParameter('kf')
        self.KF = 3.16e-10 # NOTE: taken from safe-control-gym.firmware_wrapper
        self.KM = self._getURDFParameter('km')
        self.control_counter = 0
        self.takeoff_sent = False
        self.prev_rpy = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.tick = 0
        self.control = self.firm.control_t()
        self.setpoint = self.firm.setpoint_t()
        self.sensorData = self.firm.sensorData_t()
        self.state = self.firm.state_t()
        self.tick = 0
        self.pwms = [0, 0, 0, 0]
        self.action = [0, 0, 0, 0]
        self.command_queue = []
        self.tumble_counter = 0
        self.prev_vel = np.array([0, 0, 0])
        self.prev_rpy = np.array([0, 0, 0])
        self.prev_time_s = None
        self.last_pos_pid_call = 0
        self.last_att_pid_call = 0
        self._error = False
        self.sensorData_set = False
        self.state_set = False
        self.full_state_cmd_override = True
        self.acclpf = []
        self.gyrolpf = []

###############################################################################

    def _process_command_queue(self, sim_time):
        if len(self.command_queue) > 0:
            # Reset planner object
            self.firm.crtpCommanderHighLevelStop()

            # Sets commander time variable --- this is time in s from start of flight
            self.firm.crtpCommanderHighLevelUpdateTime(sim_time)
            command, args = self.command_queue.pop(0)
            getattr(self, command)(*args)

###############################################################################

    def _update_sensorData(self, timestamp, acc_vals, gyro_vals, baro_vals=[1013.25, 25]):
        """
        Axis3f acc;               // Gs
        Axis3f gyro;              // deg/s
        Axis3f mag;               // gauss
        baro_t baro;              // C, Pa
        #ifdef LOG_SEC_IMU
            Axis3f accSec;            // Gs
            Axis3f gyroSec;           // deg/s
        #endif
        uint64_t interruptTimestamp;   // microseconds
        """
        self._update_acc(*acc_vals)
        self._update_gyro(*gyro_vals)

        self.sensorData.interruptTimestamp = timestamp
        self.sensorData_set = True

###############################################################################

    def _update_gyro(self, x, y, z):
        self.sensorData.gyro.x = self.firm.lpf2pApply(self.gyrolpf[0], float(x))
        self.sensorData.gyro.y = self.firm.lpf2pApply(self.gyrolpf[1], float(y))
        self.sensorData.gyro.z = self.firm.lpf2pApply(self.gyrolpf[2], float(z))

###############################################################################

    def _update_acc(self, x, y, z):
        self.sensorData.acc.x = self.firm.lpf2pApply(self.acclpf[0], x)
        self.sensorData.acc.y = self.firm.lpf2pApply(self.acclpf[1], y)
        self.sensorData.acc.z = self.firm.lpf2pApply(self.acclpf[2], z)

###############################################################################

    def _updateSetpoint(self, timestep):
        if not self.full_state_cmd_override:
            self.firm.crtpCommanderHighLevelTellState(self.state)
            # Sets commander time variable --- this is time in s from start of flight
            self.firm.crtpCommanderHighLevelUpdateTime(timestep)
            self.firm.crtpCommanderHighLevelGetSetpoint(self.setpoint, self.state)

###############################################################################

    def _step_controller(self):
        # if not (self.sensorData_set):
        #     logger.warning("sensorData has not been updated since last controller call.")
        # if not (self.state_set):
        #     logger.warning("state has not been updated since last controller call.")
        self.sensorData_set = False
        self.state_set = False

        # Check for tumbling crazyflie
        if self.state.acc.z < -0.5:
            self.tumble_counter += 1
        else:
            self.tumble_counter = 0
        if self.tumble_counter >= 30:
            # logger.warning("CrazyFlie is Tumbling. Killing motors to save propellers.")
            self.pwms = [0, 0, 0, 0]
            self.tick += 1
            self._error = True
            return

        # Determine tick based on time passed, allowing us to run pid slower than the 1000Hz it was
        # designed for
        cur_time = self.tick / FIRMWARE_FREQ
        if (cur_time - self.last_att_pid_call > 0.002) and (
            cur_time - self.last_pos_pid_call > 0.01
        ):
            _tick = 0  # Runs position and attitude controller
            self.last_pos_pid_call = cur_time
            self.last_att_pid_call = cur_time
        elif cur_time - self.last_att_pid_call > 0.002:
            self.last_att_pid_call = cur_time
            _tick = 2  # Runs attitude controller
        else:
            _tick = 1  # Runs neither controller

        self.firm.controllerMellinger(
            self.control, self.setpoint, self.sensorData, self.state, _tick
        )

        # Get pwm values from control object
        self._powerDistribution(self.control)
        self.tick += 1

###############################################################################

    def _powerDistribution(self, control_t):
        motor_pwms = []
        if QUAD_FORMATION_X:
            r = control_t.roll / 2
            p = control_t.pitch / 2

            motor_pwms += [
                self._motorsGetPWM(self._limitThrust(control_t.thrust - r + p + control_t.yaw))
            ]
            motor_pwms += [
                self._motorsGetPWM(self._limitThrust(control_t.thrust - r - p - control_t.yaw))
            ]
            motor_pwms += [
                self._motorsGetPWM(self._limitThrust(control_t.thrust + r - p + control_t.yaw))
            ]
            motor_pwms += [
                self._motorsGetPWM(self._limitThrust(control_t.thrust + r + p - control_t.yaw))
            ]
        else:
            motor_pwms += [
                self._motorsGetPWM(self._limitThrust(control_t.thrust + control_t.pitch + control_t.yaw))
            ]
            motor_pwms += [
                self._motorsGetPWM(self._limitThrust(control_t.thrust - control_t.roll - control_t.yaw))
            ]
            motor_pwms += [
                self._motorsGetPWM(self._limitThrust(control_t.thrust - control_t.pitch + control_t.yaw))
            ]
            motor_pwms += [
                self._motorsGetPWM(self._limitThrust(control_t.thrust + control_t.roll - control_t.yaw))
            ]

        if MOTOR_SET_ENABLE:
            self.pwms = motor_pwms
        else:
            self.pwms = np.clip(motor_pwms, MIN_PWM, MAX_PWM).tolist()

###############################################################################

    def _motorsGetPWM(self, thrust):
        thrust = thrust / 65536 * 60
        volts = -0.0006239 * thrust**2 + 0.088 * thrust
        percentage = min(1, volts / SUPPLY_VOLTAGE)
        ratio = percentage * MAX_PWM
        return ratio

###############################################################################

    def _limitThrust(self, val):
        if val > MAX_PWM:
            return MAX_PWM
        elif val < 0:
            return 0
        return val

###############################################################################

    def _update_state(self, timestamp, pos, vel, acc, rpy, quat=None):
        """
        attitude_t attitude;      // deg (legacy CF2 body coordinate system, where pitch is inverted)
        quaternion_t attitudeQuaternion;
        point_t position;         // m
        velocity_t velocity;      // m/s
        acc_t acc;                // Gs (but acc.z without considering gravity)
        """
        # RPY required for PID and high level commander
        self._update_attitude_t(self.state.attitude, timestamp, *rpy)
        # Quat required for Mellinger
        self._update_attitudeQuaternion(
            self.state.attitudeQuaternion, timestamp, *rpy
        )

        self._update_3D_vec(self.state.position, timestamp, *pos)
        self._update_3D_vec(self.state.velocity, timestamp, *vel)
        self._update_3D_vec(self.state.acc, timestamp, *acc)
        self.state_set = True

###############################################################################

    def _update_3D_vec(self, point, timestamp, x, y, z):
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        point.timestamp = timestamp

###############################################################################

    def _update_attitudeQuaternion(self, quaternion_t, timestamp, qx, qy, qz, qw=None):
        """Updates attitude quaternion.

        Note:
            if qw is present, input is taken as a quat. Else, as roll, pitch, and yaw in deg
        """
        quaternion_t.timestamp = timestamp

        if qw is None:  # passed roll, pitch, yaw
            qx, qy, qz, qw = get_quaternion_from_euler(
                qx / RAD_TO_DEG,
                qy / RAD_TO_DEG,
                qz / RAD_TO_DEG
            )

        quaternion_t.x = qx
        quaternion_t.y = qy
        quaternion_t.z = qz
        quaternion_t.w = qw

###############################################################################

    def _update_attitude_t(self, attitude_t, timestamp, roll, pitch, yaw):
        attitude_t.timestamp = timestamp
        attitude_t.roll = float(roll)
        attitude_t.pitch = float(-pitch)  # Legacy representation in CF firmware
        attitude_t.yaw = float(yaw)

# endregion

###############################################################################

# region COMMAND

    def sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        """Adds a sendfullstate command to command processing queue.

        Notes:
            Overrides any high level commands being processed.

        Args:
            pos (list): [x, y, z] position of the CF (m)
            vel (list): [x, y, z] velocity of the CF (m/s)
            acc (list): [x, y, z] acceleration of the CF (m/s^2)
            yaw (float): yaw of the CF (rad)
            rpy_rate (list): roll, pitch, yaw rates (rad/s)
            timestep (float): simulation time when command is sent (s)
        """
        self.command_queue += [["_sendFullStateCmd", [pos, vel, acc, yaw, rpy_rate, timestep]]]

###############################################################################

    def _sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        self.setpoint.position.x = float(pos[0])
        self.setpoint.position.y = float(pos[1])
        self.setpoint.position.z = float(pos[2])
        self.setpoint.velocity.x = float(vel[0])
        self.setpoint.velocity.y = float(vel[1])
        self.setpoint.velocity.z = float(vel[2])
        self.setpoint.acceleration.x = float(acc[0])
        self.setpoint.acceleration.y = float(acc[1])
        self.setpoint.acceleration.z = float(acc[2])

        self.setpoint.attitudeRate.roll = float(rpy_rate[0]) * RAD_TO_DEG
        self.setpoint.attitudeRate.pitch = float(rpy_rate[1]) * RAD_TO_DEG
        self.setpoint.attitudeRate.yaw = float(rpy_rate[2]) * RAD_TO_DEG

        quat = get_quaternion_from_euler(0, 0, yaw)
        self.setpoint.attitudeQuaternion.x = float(quat[0])
        self.setpoint.attitudeQuaternion.y = float(quat[1])
        self.setpoint.attitudeQuaternion.z = float(quat[2])
        self.setpoint.attitudeQuaternion.w = float(quat[3])

        # initilize setpoint modes to match cmdFullState
        self.setpoint.mode.x = self.firm.modeAbs
        self.setpoint.mode.y = self.firm.modeAbs
        self.setpoint.mode.z = self.firm.modeAbs

        self.setpoint.mode.quat = self.firm.modeAbs
        self.setpoint.mode.roll = self.firm.modeDisable
        self.setpoint.mode.pitch = self.firm.modeDisable
        self.setpoint.mode.yaw = self.firm.modeDisable

        # TODO: This may end up skipping control loops
        self.setpoint.timestamp = int(timestep * 1000)
        self.full_state_cmd_override = True

###############################################################################

    def sendTakeoffCmd(self, height, duration):
        """Adds a takeoff command to command processing queue.

        Args:
            height (float): target takeoff height (m)
            duration: (float): length of manuever
        """
        self.command_queue += [["_sendTakeoffCmd", [height, duration]]]

###############################################################################

    def _sendTakeoffCmd(self, height, duration):
        # logger.info(f"{self.tick}: Takeoff command sent.")
        self.takeoff_sent = True
        self.firm.crtpCommanderHighLevelTakeoff(height, duration)
        self.full_state_cmd_override = False

###############################################################################

    def sendTakeoffYawCmd(self, height, duration, yaw):
        """Adds a takeoffyaw command to command processing queue.

        Args:
            height (float): target takeoff height (m)
            duration: (float): length of manuever
            yaw (float): target yaw (rad)
        """
        self.command_queue += [["_sendTakeoffYawCmd", [height, duration, yaw]]]

###############################################################################

    def _sendTakeoffYawCmd(self, height, duration, yaw):
        # logger.info(f"{self.tick}: Takeoff command sent.")
        self.firm.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)
        self.full_state_cmd_override = False

###############################################################################

    def sendTakeoffVelCmd(self, height, vel, relative):
        """Adds a takeoffvel command to command processing queue.

        Args:
            height (float): target takeoff height (m)
            vel (float): target takeoff velocity (m/s)
            relative: (bool): whether takeoff height is relative to CF's current position
        """
        self.command_queue += [["_sendTakeoffVelCmd", [height, vel, relative]]]

###############################################################################

    def _sendTakeoffVelCmd(self, height, vel, relative):
        # logger.info(f"{self.tick}: Takeoff command sent.")
        self.firm.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

###############################################################################

    def sendLandCmd(self, height, duration):
        """Adds a land command to command processing queue.

        Args:
            height (float): target landing height (m)
            duration: (float): length of manuever
        """
        self.command_queue += [["_sendLandCmd", [height, duration]]]

###############################################################################

    def _sendLandCmd(self, height, duration):
        # logger.info(f"{self.tick}: Land command sent.")
        self.firm.crtpCommanderHighLevelLand(height, duration)
        self.full_state_cmd_override = False

###############################################################################

    def sendLandYawCmd(self, height, duration, yaw):
        """Adds a landyaw command to command processing queue.

        Args:
            height (float): target landing height (m)
            duration: (float): length of manuever
            yaw (float): target yaw (rad)
        """
        self.command_queue += [["_sendLandYawCmd", [height, duration, yaw]]]

###############################################################################

    def _sendLandYawCmd(self, height, duration, yaw):
        # logger.info(f"{self.tick}: Land command sent.")
        self.firm.crtpCommanderHighLevelLandYaw(height, duration, yaw)
        self.full_state_cmd_override = False

###############################################################################

    def sendLandVelCmd(self, height, vel, relative):
        """Adds a landvel command to command processing queue.

        Args:
            height (float): target landing height (m)
            vel (float): target landing velocity (m/s)
            relative: (bool): whether landing height is relative to CF's current position
        """
        self.command_queue += [["_sendLandVelCmd", [height, vel, relative]]]

###############################################################################

    def _sendLandVelCmd(self, height, vel, relative):
        # logger.info(f"{self.tick}: Land command sent.")
        self.firm.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

###############################################################################

    def sendStopCmd(self):
        """Adds a stop command to command processing queue."""
        self.command_queue += [["_sendStopCmd", []]]

###############################################################################

    def _sendStopCmd(self):
        # logger.info(f"{self.tick}: Stop command sent.")
        self.firm.crtpCommanderHighLevelStop()
        self.full_state_cmd_override = False

###############################################################################

    def sendGotoCmd(self, pos, yaw, duration_s, relative):
        """Adds a goto command to command processing queue.

        Args:
            pos (list): [x, y, z] target position (m)
            yaw (float): target yaw (rad)
            duration_s (float): length of manuever
            relative (bool): whether setpoint is relative to CF's current position
        """
        self.command_queue += [["_sendGotoCmd", [pos, yaw, duration_s, relative]]]

###############################################################################

    def _sendGotoCmd(self, pos, yaw, duration_s, relative):
        # logger.info(f"{self.tick}: Go to command sent.")
        self.firm.crtpCommanderHighLevelGoTo(*pos, yaw, duration_s, relative)
        self.full_state_cmd_override = False

    def notifySetpointStop(self):
        """Adds a notifySetpointStop command to command processing queue."""
        self.command_queue += [["_notifySetpointStop", []]]

    def _notifySetpointStop(self):
        """Adds a notifySetpointStop command to command processing queue."""
        # logger.info(f"{self.tick}: Notify setpoint stop command sent.")
        self.firm.crtpCommanderHighLevelTellState(self.state)
        self.full_state_cmd_override = False

# endregion
