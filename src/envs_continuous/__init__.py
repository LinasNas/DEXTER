from .anom_mj_env import AnomMJEnv
from .cartpole import CartpoleEnv
from .half_cheetah import HalfCheetahEnv
from .pusher import PusherEnv
from .reacher import Reacher3DEnv
from gym.envs.registration import register

class AnomCartpoleEnv(AnomMJEnv, CartpoleEnv):
    pass

class AnomPusherEnv(AnomMJEnv, PusherEnv):
    pass

class AnomReacherEnv(AnomMJEnv, Reacher3DEnv):
    pass

class AnomHalfCheetahEnv(AnomMJEnv, HalfCheetahEnv):
    pass

register(id="MJCartpole-v0", entry_point="envs_continuous.cartpole:AnomCartpoleEnv", max_episode_steps=200)
register(id="MJPusher-v0", entry_point="envs_continuous.pusher:AnomPusherEnv", max_episode_steps=150)
register(id="MJReacher-v0", entry_point="envs_continuous.reacher:AnomReacher3DEnv", max_episode_steps=150)
register(id="MJHalfCheetah-v0", entry_point="envs_continuous.half_cheetah:AnomHalfCheetahEnv", max_episode_steps=1000)
