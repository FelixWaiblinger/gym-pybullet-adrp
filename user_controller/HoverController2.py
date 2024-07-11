"""HoverController"""

import numpy as np

from user_controller import BaseController


class HoverController2(BaseController):
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
        return np.array([obs[0]+0.1, obs[1]+0.1, 1 + self.drone_id, 0], dtype=np.float32)
