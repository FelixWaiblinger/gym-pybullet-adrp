"""Constants"""

import math

import numpy as np


###############################################################################
# math
RAD_TO_DEG = 180 / math.pi
DEG_TO_RAD = math.pi / 180
VEC3_UP = np.array([0, 0, 1])
VEC3_ZERO = np.zeros(3)

###############################################################################
# PATH
URDF_DIR = "gym_pybullet_adrp/assets/"
ROOT_DIR_FELIX = "~/Desktop/TUM/ADRP/"
FIRMWARE_PATH = "pycffirmware/wrapper/pycffirmware.py"

###############################################################################
# lsy-drone-racing
Z_LOW = 0.3
Z_HIGH = 0.775
VISIBILITY_RANGE = 0.45

###############################################################################
# crazyflie firmware
FIRMWARE_FREQ = 500
FIRMWARE_DT = 1.0 / 500
CTRL_FREQ = 25
CTRL_DT = 1.0 / 25
MIN_PWM = 20000
MAX_PWM = 65535
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
ACTION_DELAY = 0
SENSOR_DELAY = 0
GYRO_LPF_CUTOFF_FREQ = 80
ACCEL_LPF_CUTOFF_FREQ = 30
SUPPLY_VOLTAGE = 3
MOTOR_SET_ENABLE = True
QUAD_FORMATION_X = True
SPEED_LIMIT = 10 # TODO: replace arbitrary speed limit
