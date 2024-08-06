"""Mellinger Controller"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_pybullet_adrp.control import BaseControl
from gym_pybullet_adrp.utils import get_quaternion_from_euler, load_firmware
from gym_pybullet_adrp.utils.enums import DroneModel, Command
from gym_pybullet_adrp.utils.constants import *

if TYPE_CHECKING:
    import pycffirmware as firmware


def low_level_control(drone: int, conn):
    """Low-level control function to be run by an external process."""
    controller = MellingerControl(drone, DroneModel.CF2X)

    # controller waits for commands until being shutdown
    while True:
        command, args = conn.recv()

        if command == "reset":
            obs, _ = args
            controller.reset(obs)
        elif command == "step":
            t, pos, rpy, vel, acc, ang = args
            rpm = controller.computeControl(t, pos, rpy, vel, acc, ang)[0]
            conn.send(rpm)
        elif command == "command":
            cmd, args = args
            if cmd == Command.FULLSTATE:
                controller.sendFullStateCmd(*args)
            elif cmd == Command.TAKEOFF:
                controller.sendTakeoffCmd(*args)
            elif cmd == Command.TAKEOFFYAW:
                controller.sendTakeoffYawCmd(*args)
            elif cmd == Command.TAKEOFFVEL:
                controller.sendTakeoffVelCmd(*args)
            elif cmd == Command.LAND:
                controller.sendLandCmd(*args)
            elif cmd == Command.LANDYAW:
                controller.sendLandYawCmd(*args)
            elif cmd == Command.LANDVEL:
                controller.sendLandVelCmd(*args)
            elif cmd == Command.GOTO:
                controller.sendGotoCmd(*args)
            elif cmd == Command.STOP:
                controller.sendStopCmd(*args)
            elif cmd == Command.NOTIFY:
                controller.notifySetpointStop(*args)
            else:
                continue
            controller.process_command_queue(args[-1])
            conn.send("ok") # NOTE: for synchronization
        else:
            conn.close()
            return


class MellingerControl(BaseControl):
    """Mellinger control class for Crazyflie drones."""

    ################################################################################

    def __init__(
        self,
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

    def reset(self, init_obs):
        """Resets the control classes.

        Copied and slightly modified from safe-control-gym.firmware_wrapper
        """
        super().reset()

        # NOTE: whole section taken from safe-control-gym.firmware_wrapper
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
        self.sensor_data = self.firm.sensorData_t()
        self.state = self.firm.state_t()
        self.tick = 0
        self.command_queue = []

        self.tumble_counter = 0
        self.last_pos_pid_call = 0
        self.last_att_pid_call = 0

        # Initialize state flags
        self._error = False
        self.sensor_data_set = False
        self.state_set = False

        # When true, high level commander is not called
        self.full_state_cmd_override = True

        self.firm.controllerMellingerInit()
        # print("Mellinger init test:", self.firm.controllerMellingerTest())

        # observations about the drone owned by this controller
        drone = init_obs[self.drone_id, :12]

        self.firm.crtpCommanderHighLevelInit()
        self._update_state(0, drone[:3], drone[6:9], VEC3_UP, drone[3:6] * RAD_TO_DEG)

        self.prev_rpy = drone[3:6]
        self.prev_vel = drone[6:9]
        self.firm.crtpCommanderHighLevelTellState(self.state)

###############################################################################

    def computeControl(
        self,
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
        self._update_setpoint(self.tick / FIRMWARE_FREQ)

        # Step controller
        pwms = self._step_controller()

        # =====================================
        # TODO: check whether we need all this?
        clipped_pwms = np.clip(np.array(pwms), MIN_PWM, MAX_PWM)
        thrust = self.KF * (PWM2RPM_SCALE * clipped_pwms + PWM2RPM_CONST) ** 2
        # thrust = thrust[[3, 2, 1, 0]] # assign values to correct motor

        # convert to quad motor rpm commands
        pwms = self._thr2pwm(
            thrust, PWM2RPM_SCALE, PWM2RPM_CONST, self.KF, MIN_PWM, MAX_PWM
        )
        # =====================================

        rpms = PWM2RPM_SCALE * pwms + PWM2RPM_CONST

        return rpms, None, None

###############################################################################

# region CONTROLLER

    def _init_variables(self):
        # NOTE: taken from safe-control-gym.firmware_wrapper
        self.KF = 3.16e-10
        self.KM = self._getURDFParameter('km')
        self.prev_rpy = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.tick = 0
        self.control = self.firm.control_t()
        self.setpoint = self.firm.setpoint_t()
        self.sensor_data = self.firm.sensorData_t()
        self.state = self.firm.state_t()
        self.command_queue = []
        self.tumble_counter = 0
        self.last_pos_pid_call = 0
        self.last_att_pid_call = 0
        self._error = False
        self.sensor_data_set = False
        self.state_set = False
        self.full_state_cmd_override = True
        self.acclpf = []
        self.gyrolpf = []

###############################################################################

    def process_command_queue(self, sim_time):
        """Pop and execute next command in the queue using the high-level
        controller.
        """
        if len(self.command_queue) > 0:
            # Reset planner object
            self.firm.crtpCommanderHighLevelStop()

            # Sets commander time variable --- this is time in s from start of flight
            self.firm.crtpCommanderHighLevelUpdateTime(sim_time)
            command, args = self.command_queue.pop(0)
            getattr(self, command)(*args)

###############################################################################

    def _thr2pwm(self, thrust, pwm2rpm_scale, pwm2rpm_const, ct, pwm_min, pwm_max):
        """Generic cmd to pwm function.

        For 1D, thrust is the total of all 4 motors; for 2D, 1st thrust is total of motor
        1 & 4, 2nd thrust is total of motor 2 & 3; for 4D, thrust is thrust of each motor.

        Args:
            thrust (ndarray): array of length 1, 2 containing target thrusts.
            pwm2rpm_scale (float): scaling factor between PWM and RPMs.
            pwm2rpm_const (float): constant factor between PWM and RPMs.
            ct (float): torque coefficient.
            pwm_min (float): pwm lower bound.
            pwm_max (float): pwm upper bound.

        Returns:
            ndarray: array of length 4 containing PWM.

        """
        n_motor = 4 // int(thrust.size)

        # Make sure thrust is not negative.
        thrust = np.clip(thrust, np.zeros_like(thrust), None)

        motor_pwm = (np.sqrt(thrust / n_motor / ct) - pwm2rpm_const) / pwm2rpm_scale

        if thrust.size == 1:  # 1D case
            motor_pwm = np.repeat(motor_pwm, 4)
        elif thrust.size == 2:  # 2D case
            motor_pwm = np.concatenate([motor_pwm, motor_pwm[::-1]], 0)
        elif thrust.size == 4:  # 3D case
            motor_pwm = np.array(motor_pwm)
        else:
            raise ValueError("Input action shape not supported.")

        motor_pwm = np.clip(motor_pwm, pwm_min, pwm_max)

        return motor_pwm

###############################################################################

    def _update_sensorData(self, timestamp, accs, gyros):
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
        acc = [self.firm.lpf2pApply(self.acclpf[i], v) for i, v in enumerate(accs)]
        gyro = [self.firm.lpf2pApply(self.gyrolpf[i], v) for i, v in enumerate(gyros)]
        self._update_vector(self.sensor_data.acc, ("x", "y", "z"), acc)
        self._update_vector(self.sensor_data.gyro, ("x", "y", "z"), gyro)

        self.sensor_data.interruptTimestamp = timestamp
        self.sensor_data_set = True

###############################################################################

    def _update_setpoint(self, timestep):
        if not self.full_state_cmd_override:
            self.firm.crtpCommanderHighLevelTellState(self.state)
            # Sets commander time variable (time in s from start of flight)
            self.firm.crtpCommanderHighLevelUpdateTime(timestep)
            self.firm.crtpCommanderHighLevelGetSetpoint(self.setpoint, self.state)

###############################################################################

    def _step_controller(self):
        self.sensor_data_set = False
        self.state_set = False

        # Check for tumbling crazyflie
        if self.state.acc.z < -0.5:
            self.tumble_counter += 1
        else:
            self.tumble_counter = 0
        if self.tumble_counter >= 30:
            # logger.warning("CrazyFlie is Tumbling. Killing motors to save propellers.")
            self.tick += 1
            self._error = True
            return np.zeros(4)

        # Determine tick based on time passed, allowing us to run pid slower than the 1000Hz it was
        # designed for
        cur_time = self.tick / FIRMWARE_FREQ

        # Runs position and attitude controller
        if (cur_time - self.last_att_pid_call > 0.002) and \
           (cur_time - self.last_pos_pid_call > 0.01):
            _tick = 0
            self.last_pos_pid_call = cur_time
            self.last_att_pid_call = cur_time

        # Runs attitude controller
        elif cur_time - self.last_att_pid_call > 0.002:
            self.last_att_pid_call = cur_time
            _tick = 2

        # Runs neither controller
        else:
            _tick = 1

        self.firm.controllerMellinger(
            self.control, self.setpoint, self.sensor_data, self.state, _tick
        )

        # Get pwm values from control object
        self.tick += 1
        return self._compute_pwms(self.control)

###############################################################################

    def _compute_pwms(self, control_t):
        """Return clipped PWM values for each rotor"""

        assert QUAD_FORMATION_X, \
            "MultiRaceAviary currently only supports drones in X formation!"

        r = control_t.roll / 2
        p = control_t.pitch / 2
        y = control_t.yaw
        t = control_t.thrust

        thrust = np.array([t-r+p+y, t-r-p-y, t+r-p+y, t+r+p-y])
        thrust = np.clip(thrust, 0, MAX_PWM)
        thrust = thrust / MAX_PWM * 60

        volts = -0.0006239 * thrust**2 + 0.088 * thrust
        percentage = np.minimum(1, volts / SUPPLY_VOLTAGE)
        motor_pwms = percentage * MAX_PWM

        return motor_pwms # NOTE: return unclipped because of MOTOR_SET_ENABLE

###############################################################################

    def _update_state(self, timestamp, pos, vel, acc, rpy):
        """
        attitude_t attitude;      // deg (legacy CF2 body coordinate system, where pitch is inverted)
        quaternion_t attitudeQuaternion;
        point_t position;         // m
        velocity_t velocity;      // m/s
        acc_t acc;                // Gs (but acc.z without considering gravity)
        """
        # RPY required for PID and high level commander
        self._update_vector(
            self.state.attitude,
            ("roll", "pitch", "yaw", "timestamp"),
            [*(rpy * np.array([1, -1, 1])), timestamp] # legacy inverted pitch
        )

        # Quat required for Mellinger
        self._update_vector(
            self.state.attitudeQuaternion,
            ("x", "y", "z", "w", "timestamp"),
            [*get_quaternion_from_euler(*(rpy * DEG_TO_RAD)), timestamp]
        )

        self._update_vector(self.state.position, values=[*pos, timestamp])
        self._update_vector(self.state.velocity, values=[*vel, timestamp])
        self._update_vector(self.state.acc, values=[*acc, timestamp])
        self.state_set = True

###############################################################################

    def _update_vector(
        self,
        vector,
        attrs=("x", "y", "z", "timestamp"),
        values=[0, 0, 0, 0]
    ):
        """Set attributes of a vector individually"""
        for attr, val in zip(attrs, values):
            setattr(vector, attr, val)

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
        self.command_queue += \
            [["_sendFullStateCmd", [pos, vel, acc, yaw, rpy_rate, timestep]]]

###############################################################################

    def _sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2]
        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]
        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.setpoint.attitudeRate.roll = rpy_rate[0] * RAD_TO_DEG
        self.setpoint.attitudeRate.pitch = rpy_rate[1] * RAD_TO_DEG
        self.setpoint.attitudeRate.yaw = rpy_rate[2] * RAD_TO_DEG

        quat = get_quaternion_from_euler(0, 0, yaw)
        self.setpoint.attitudeQuaternion.x = quat[0]
        self.setpoint.attitudeQuaternion.y = quat[1]
        self.setpoint.attitudeQuaternion.z = quat[2]
        self.setpoint.attitudeQuaternion.w = quat[3]

        # initilize setpoint modes to match cmdFullState
        self.setpoint.mode.x = self.firm.modeAbs
        self.setpoint.mode.y = self.firm.modeAbs
        self.setpoint.mode.z = self.firm.modeAbs

        self.setpoint.mode.quat = self.firm.modeAbs
        self.setpoint.mode.roll = self.firm.modeDisable
        self.setpoint.mode.pitch = self.firm.modeDisable
        self.setpoint.mode.yaw = self.firm.modeDisable

        # NOTE: This may end up skipping control loops
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
        self.command_queue += \
            [["_sendGotoCmd", [pos, yaw, duration_s, relative]]]

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
