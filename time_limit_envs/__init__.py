from gym.envs.registration import register
import roboschool


# register(
#     id="FrozenLakeLargeShiftedIce-v0",
#     entry_point="frozen_lakes.lake_envs:FrozenLakeLargeShiftedIceEnv",
# )



# This one has T = 300 and has time and flag in observation
register(
    id='RoboschoolHopperLimited-v1',
    #entry_point='roboschool:RoboschoolHopper',
    entry_point='time_limit_envs.limited_envs:RoboschoolHopperLimited',
    #max_episode_steps=1000,
    max_episode_steps=300,
    reward_threshold=2500.0,
    tags={ "pg_complexity": 8*1000000 },
)
