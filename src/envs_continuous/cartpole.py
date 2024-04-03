import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from .anom_mj_env import AnomMJEnv

PENDULUM_LENGTH = 0.6


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/cartpole.xml" % dir_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = PENDULUM_LENGTH
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, PENDULUM_LENGTH]))) / (cost_lscale**2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + self.np_random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array([x0 - PENDULUM_LENGTH * np.sin(theta), -PENDULUM_LENGTH * np.cos(theta)])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

class AnomCartpoleEnv(AnomMJEnv, CartpoleEnv):
    pass