from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from roboschool.gym_mujoco_walkers import RoboschoolHopper
from roboschool.gym_reacher import RoboschoolReacher
from roboschool.gym_pendulums import RoboschoolInvertedPendulum
from gym import Wrapper
import time
import numpy as np
from gym import spaces


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
    def __init__(self):
        super(RoboschoolHopperLimited, self).__init__()
        self._max_episode_steps = 300
        self._elapsed_steps = 0

        low = self.observation_space.low
        high = self.observation_space.high
        low = np.concatenate((low, np.array([300])))
        high = np.concatenate((high, np.array([300])))
        self.observation_space = spaces.Box(low, high)



    def _step(self, action):
        observation, reward, done, info = super(RoboschoolHopperLimited, self)._step(action)
        self._elapsed_steps += 1
        time_remaining = self._max_episode_steps - self._elapsed_steps
        time_remaining = np.array([time_remaining])
        #print(time_remaining.shape)
        observation = np.concatenate((observation, time_remaining))
        return observation, reward, done, info

    def _reset(self):
        self._elapsed_steps = 0
        tmp = super(RoboschoolHopperLimited, self)._reset()
        out = np.concatenate((tmp, np.array([0])))
        return out


class RoboschoolReacherLimited(RoboschoolReacher):
    def __init__(self):
        super(RoboschoolReacherLimited, self).__init__()
        self._max_episode_steps = 50
        self._elapsed_steps = 0

        low = self.observation_space.low
        high = self.observation_space.high
        low = np.concatenate((low, np.array([50])))
        high = np.concatenate((high, np.array([50])))
        self.observation_space = spaces.Box(low, high)


    def _step(self, action):
        observation, reward, done, info = super(RoboschoolReacherLimited, self)._step(action)
        self._elapsed_steps += 1
        time_remaining = self._max_episode_steps - self._elapsed_steps
        time_remaining = np.array([time_remaining])
        #print(time_remaining.shape)
        observation = np.concatenate((observation, time_remaining))
        return observation, reward, done, info

    def _reset(self):
        self._elapsed_steps = 0
        tmp = super(RoboschoolReacherLimited, self)._reset()
        out = np.concatenate((tmp, np.array([0])))
        return out


class RoboschoolInvertedPendulumLimited(RoboschoolInvertedPendulum):
    def __init__(self):
        super(RoboschoolInvertedPendulumLimited, self).__init__()
        self._max_episode_steps = 1000
        self._elapsed_steps = 0

        low = self.observation_space.low
        high = self.observation_space.high
        low = np.concatenate((low, np.array([1000])))
        high = np.concatenate((high, np.array([1000])))
        self.observation_space = spaces.Box(low, high)


    def _step(self, action):
        observation, reward, done, info = super(RoboschoolInvertedPendulumLimited, self)._step(action)
        self._elapsed_steps += 1
        time_remaining = self._max_episode_steps - self._elapsed_steps
        time_remaining = np.array([time_remaining])
        #print(time_remaining.shape)
        observation = np.concatenate((observation, time_remaining))
        return observation, reward, done, info

    def _reset(self):
        self._elapsed_steps = 0
        tmp = super(RoboschoolInvertedPendulumLimited, self)._reset()
        out = np.concatenate((tmp, np.array([0])))
        return out










class TimeLimit(Wrapper):
    def __init__(self, env, max_episode_seconds=None, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_seconds = max_episode_seconds
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = 0
        self._episode_started_at = None

    @property
    def _elapsed_seconds(self):
        return time.time() - self._episode_started_at

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            logger.debug("Env has passed the step limit defined by TimeLimit.")
            return True

        if self._max_episode_seconds is not None and self._max_episode_seconds <= self._elapsed_seconds:
            logger.debug("Env has passed the seconds limit defined by TimeLimit.")
            return True

        return False

    def _step(self, action):
        assert self._episode_started_at is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._past_limit():
            if self.metadata.get('semantics.autoreset'):
                _ = self.reset() # automatically reset the env
            done = True 

        return observation, reward, done, info

    def _reset(self):
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        return self.env.reset()

