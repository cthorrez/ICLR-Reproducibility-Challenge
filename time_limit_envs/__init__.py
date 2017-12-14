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


# This one has T = 300 and has time and flag in observation
register(
    id='RoboschoolReacherLimited-v1',
    #entry_point='roboschool:RoboschoolHopper',
    entry_point='time_limit_envs.limited_envs:RoboschoolReacherLimited',
    #max_episode_steps=1000,
    max_episode_steps=50,
    reward_threshold=18.0,
    tags={ "pg_complexity": 1*1000000 },
)


register(
    id='RoboschoolInvertedPendulumLimited-v1',
    entry_point='time_limit_envs.limited_envs:RoboschoolInvertedPendulumLimited',
    max_episode_steps=1000,
    reward_threshold=950.0,
    tags={ "pg_complexity": 1*1000000 },
)




# envs without time as feature, but with different than normal time limits
register(
    id='RoboschoolReacherCT-v1',
    entry_point='roboschool:RoboschoolReacher',
    max_episode_steps=50,
    reward_threshold=18.0,
    tags={ "pg_complexity": 1*1000000 },
    )

register(
    id='RoboschoolHopperCT-v1',
    entry_point='roboschool:RoboschoolHopper',
    max_episode_steps=300,
    reward_threshold=2500.0,
    tags={ "pg_complexity": 8*1000000 },
    )

register(
    id='RoboschoolInvertedPendulumCT-v1',
    entry_point='roboschool:RoboschoolInvertedPendulum',
    max_episode_steps=1000,
    reward_threshold=950.0,
    tags={ "pg_complexity": 1*1000000 },
    )







