"""Enums"""

from enum import Enum


###############################################################################

class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"   # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0 in the + configuration
    RACE = "racer"  # Racer drone in the X configuration

###############################################################################

class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"                         # Base PyBullet physics update
    DYN = "dyn"                         # Explicit dynamics model
    PYB_GND = "pyb_gnd"                 # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"               # PyBullet physics update with drag
    PYB_DW = "pyb_dw"                   # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # PyBullet physics update with ground effect, drag, and downwash

###############################################################################

class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0     # Red, green, blue (and alpha)
    DEP = 1     # Depth
    SEG = 2     # Segmentation by object id
    BW = 3      # Black and white

###############################################################################

class ActionType(Enum):
    """Action type enumeration class."""
    MEL = "mel"                 # Mellinger position control
    RPM = "rpm"                 # RPMS
    PID = "pid"                 # PID control
    VEL = "vel"                 # Velocity input (using PID control)
    ONE_D_RPM = "one_d_rpm"     # 1D (identical input to all motors) with RPMs
    ONE_D_PID = "one_d_pid"     # 1D (identical input to all motors) with PID control

###############################################################################

class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"     # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"     # RGB camera capture in each drone's POV

###############################################################################

class Command(Enum):
    """Mellinger controller high-level command class."""
    FULLSTATE = "fst"
    TAKEOFF = "tko"
    TAKEOFFYAW = "toy"
    TAKEOFFVEL = "tov"
    LAND = "lnd"
    LANDYAW = "ldy"
    LANDVEL = "ldv"
    STOP = "stp"
    GOTO = "gto"
    NOTIFY = "ntf"
    NONE = "non"

###############################################################################

class State(Enum):
    """State machine controlled behaviour states"""
    INIT = 0
    TAKEOFF = 1
    CONTROL = 2
    LAND = 3
    END = 4

###############################################################################

class RaceMode(Enum):
    """Race mode defines collision behaviour and observation space structure"""
    COMPARE = 0
    COMPETE = 1
