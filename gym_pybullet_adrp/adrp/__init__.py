"""Environment registration"""

from gymnasium.envs.registration import register


register(
     id="MultiRaceAviary-v0",
     entry_point="gym_pybullet_drones.envs:MultiRaceAviary"
)
