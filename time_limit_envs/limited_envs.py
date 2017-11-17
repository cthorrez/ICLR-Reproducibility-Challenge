from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from roboschool.gym_mujoco_walkers import RoboschoolHopper



# class FrozenLakeLargeShiftedIceEnv(FrozenLakeEnv):

#     def __init__(self):
#         desc = ["SFFFFF",
#                 "FFHHFF",
#                 "FFFFFF",
#                 "FFFFFG"]
#         super(FrozenLakeLargeShiftedIceEnv, self).__init__(desc=desc)


#     def _step(self, action):
#         observation, reward, done, info = \
#             super(FrozenLakeLargeShiftedIceEnv, self)._step(action)
#         if reward == 0 and done:
#             reward = -1
#         return observation, reward, done, info


class RoboschoolHopperLimited(RoboschoolHopper):


    def _step(self, action):
        observation, reward, done, info = super(RoboschoolHopperLimited, self)._step(action)
        return observation, reward, done, info

