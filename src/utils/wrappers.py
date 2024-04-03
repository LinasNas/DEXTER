# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
import gym
from gym import ObservationWrapper
from gym.wrappers import TimeLimit
from gym.spaces import Box
from collections import deque
import imageio

from typing import Optional

# from stable_baselines3.common.monitor import Monitor


class FrameStack(ObservationWrapper):
    """
    Observation wrapper that stacks the observations in a rolling manner.
    """

    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high)

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.concatenate(self.frames, -1)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.observation()


# taken from https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/wrappers/time_feature.py
class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.0]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


# taken from https://github.com/openai/gym/blob/8a96440084a6b9be66b2216b984a1c170e4a061c/gym/wrappers/normalize.py#L43
class NormalizeObservation(gym.core.Wrapper):
    def __init__(
        self,
        env,
        epsilon=1e-8,
    ):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, dones, infos

    def reset(self, seed: Optional[int] = None):
        obs = self.env.reset()
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs

    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class RewardObsWrapper(gym.Wrapper):
    """concatenates the reward to the observation"""

    def __init__(self, env, concat_pos=0, default_rew=0):
        self.concat_pos = concat_pos
        self.default_rew = default_rew
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.0]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward_obs = self._get_reward_obs(obs, reward)
        return reward_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        reward_obs = self._get_reward_obs(obs, self.default_rew)
        return reward_obs

    def _get_reward_obs(self, obs, reward):
        if self.concat_pos == 0:
            return np.append(reward, obs)
        elif self.concat_pos == -1:
            return np.append(obs, reward)
        else:
            raise ValueError("unknwon concat position for RewardObsWrapper")


class VideoWrapper(gym.Wrapper):
    def __init__(self, env, save_path=None, file_format=".mp4"):
        super().__init__(env)
        self.file_format = file_format
        self.save_path = save_path or f"tmp/{self.env.spec.id}_{np.random.randint(1e5, 1e6)}{self.file_format}"

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.img_lst.append(self.render(mode="rgb_array"))
        if done:
            self.save(self.save_path)
        return obs, reward, done, info

    def reset(self):
        self.img_lst = []
        obs = self.env.reset()
        self.img_lst.append(self.env.render(mode="rgb_array"))
        return obs

    def save(self, save_path, fps=29):
        imageio.mimsave(save_path, np.array(self.img_lst), fps=fps)

    def show_video(self):
        if self.save_path.endswith(".mp4"):
            from IPython.display import Video

            return Video(self.save_path, html_attributes="loop autoplay")
        elif self.save_path.endswith(".gif"):
            from IPython.display import HTML

            return HTML(f'<img src="{self.save_path}">')
