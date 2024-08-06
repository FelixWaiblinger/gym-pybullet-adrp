"""General use functions.
"""

import sys
import time
import importlib
import argparse
from pathlib import Path

import yaml
import numpy as np
import pybullet as pb
from munch import Munch, munchify

from gym_pybullet_adrp.utils.constants import FIRMWARE_PATH, URDF_DIR


################################################################################

def get_quaternion_from_euler(roll, pitch, yaw):
    """Convert an Euler angle to a quaternion.

    Args:
        roll (float): The roll (rotation around x-axis) angle in radians.
        pitch (float): The pitch (rotation around y-axis) angle in radians.
        yaw (float): The yaw (rotation around z-axis) angle in radians.

    Returns:
        list: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) \
        - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) \
        + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)

    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) \
        - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)

    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) \
        + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i % (int(1 / (24 * timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep * i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")

###############################################################################

def load_config(path: str | Path) -> Munch:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The munchified config dict.
    """
    if isinstance(path, str):
        path = Path(path)
    assert path.exists(), f"Configuration file not found: {path}"
    with open(path, "r", encoding="utf-8") as file:
        return munchify(yaml.safe_load(file))

###############################################################################

def load_controller(path: str | Path, class_name: str=None):
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    if isinstance(path, str):
        path = Path(path)
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    if not class_name:
        class_name = path.name.split(".")[0]
        print(class_name)
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)
    assert hasattr(controller_module, class_name)
    ctrl_class = controller_module.__dict__[class_name]

    try:
        return ctrl_class
    except ImportError as e:
        raise e

###############################################################################

def load_firmware(path_to_repos: str | Path):
    """Load an individual version of the crazyflie firmware c bindings."""
    if isinstance(path_to_repos, str):
        path_to_repos = Path(path_to_repos)
    path = path_to_repos / FIRMWARE_PATH
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("firmware", path)
    firmware = importlib.util.module_from_spec(spec)
    sys.modules["firmware"] = firmware
    spec.loader.exec_module(firmware)

    return firmware

###############################################################################

def draw_trajectory(
    initial_info: dict,
    waypoints: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_z: np.ndarray,
):
    """Draw a trajectory in PyBullet's GUI."""
    for point in waypoints:
        urdf_path = Path(URDF_DIR) / "sphere.urdf"
        pb.loadURDF(
            str(urdf_path),
            [point[0], point[1], point[2]],
            pb.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=initial_info["pyb_client"],
        )
    step = int(ref_x.shape[0] / 50)
    for i in range(step, ref_x.shape[0], step):
        pb.addUserDebugLine(
            lineFromXYZ=[ref_x[i - step], ref_y[i - step], ref_z[i - step]],
            lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
            lineColorRGB=[1, 0, 0],
            physicsClientId=initial_info["pyb_client"],
        )
    pb.addUserDebugLine(
        lineFromXYZ=[ref_x[i], ref_y[i], ref_z[i]],
        lineToXYZ=[ref_x[-1], ref_y[-1], ref_z[-1]],
        lineColorRGB=[1, 0, 0],
        physicsClientId=initial_info["pyb_client"],
    )
