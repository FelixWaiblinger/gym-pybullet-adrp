"""HoverController"""

import numpy as np

from user_controller import BaseController
from gym_pybullet_adrp.utils.enums import Command

class HoverController(BaseController):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.
    """

###############################################################################

    def predict(self,
        obs: np.ndarray,
        reward: float=None,
        done: bool=None,
        info: dict=None,
        ep_time: float=None
    ) -> np.ndarray:
        """Predict the next action."""
        return np.array([1.5, 0.5, 1 + self.drone_id, 0], dtype=np.float64)
