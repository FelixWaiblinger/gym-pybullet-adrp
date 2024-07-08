"""StateMachine"""


from gym_pybullet_adrp.utils.enums import State, Command


class StateMachine:
    """State machine for administering ADRP drone behaviour"""

    def __init__(self, takeoff: bool=True, land: bool=True) -> None:
        self.takeoff = takeoff
        self.land = land
        self.state = State.INIT
        self.timer = 0

    def reset(self):
        """Restart the state machine"""
        self.state = State.INIT

    def step(self, action):
        """Advance to the next or a given state"""
        # a finished drone must be reset
        if self.state == State.END:
            return

        # transition to (given) next state
        self._transition()


    def _transition(self):
        """Move to the next state"""
        self.state.value += 1

        # skip unwanted states
        if (
            (not self.takeoff and self.state == State.TAKEOFF)
            or
            (not self.land and self.state == State.LAND)
        ):
            self.state.value += 1
