"""General use functions.
"""

import sys
import time
import importlib
import argparse
from pathlib import Path

import yaml
from munch import Munch, munchify

from gym_pybullet_adrp.control import BaseControl

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
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

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

################################################################################

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

################################################################################

def load_controller(path: str | Path) -> BaseControl:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    if isinstance(path, str):
        path = Path(path)
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)
    assert hasattr(controller_module, "Controller")
    assert issubclass(controller_module.Controller, BaseControl)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e
