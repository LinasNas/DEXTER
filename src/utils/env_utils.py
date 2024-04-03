# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------



import gym
from gym.wrappers import TimeLimit
# import gymnasium as gym
# from gymnasium.wrappers import TimeLimit

from utils.wrappers import TimeFeatureWrapper

powers_dict = {
    "AntBulletEnv-v0": 2.5,
    "HalfCheetahBulletEnv-v0": 0.9,
    "HopperBulletEnv-v0": 0.75,
    "Walker2DBulletEnv-v0": 0.4,
}


anomalous_powers_dict = {
    "AntBulletEnv-v0": 1.5,
    "HalfCheetahBulletEnv-v0": 0.6,
    "HopperBulletEnv-v0": 0.65,
    "Walker2DBulletEnv-v0": 0.35,
}


def get_task_horizon(env):
    e = env
    try:
        while not type(e) == TimeLimit:
            e = e.env
            assert not e == e.env
        task_horizon = e._max_episode_steps
    except Exception as e:
        raise ValueError(e, "env needs to be wrapped in a TimeLimit wrapper")
    return task_horizon


def make_env(env_id, anomaly_delay=None, mod=None, seed=None):
    gym.logger.set_level(40)
    if env_id in {"Acrobot-v1", "CartPole-v0", "CartPole-v1", "LunarLander-v2"}:
        if mod is None:
            env = gym.make(env_id)
        else:
            env_base_name = env_id.split("-v")[0]
            mod_env_name = f"{env_base_name}Mod-v{mod}"
            env = gym.make(mod_env_name)
            env.when_anomaly_starts = anomaly_delay

    elif env_id in {"AntBulletEnv-v0", "HalfCheetahBulletEnv-v0", "HopperBulletEnv-v0", "Walker2DBulletEnv-v0"}:
        import pybullet_envs  # noqa

        if mod is None:
            env = gym.make(env_id, power=powers_dict[env_id])
        else:
            env = gym.make(env_id, power=anomalous_powers_dict[env_id], case=mod, anomaly_injection=anomaly_delay)
        env = TimeFeatureWrapper(env)

    elif env_id in {"MJCartpole-v0", "MJHalfCheetah-v0", "MJPusher-v0", "MJReacher-v0"}:
        import mujoco_envs  # noqa: F401
        #import envs
        
        if mod is None:
            env = gym.make(env_id)
        else:
            env = gym.make("Mod" + env_id, mod=mod, anomaly_delay=anomaly_delay)

    else:
        raise ValueError(f"{env_id} not configured")
    if seed:
        env.seed(seed)
    return env
