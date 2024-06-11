"""Environment registry"""

from gymnasium.envs.registration import register

register(
    id='ctrl-aviary-v0',
    entry_point='gym_pybullet_adrp.envs:CtrlAviary',
)

register(
    id='velocity-aviary-v0',
    entry_point='gym_pybullet_adrp.envs:VelocityAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_pybullet_adrp.envs:HoverAviary',
)

register(
    id='multihover-aviary-v0',
    entry_point='gym_pybullet_adrp.envs:MultiHoverAviary',
)

register(
     id="multi-race-aviary-v0",
     entry_point="gym_pybullet_adrp.envs:MultiRaceAviary",
)
