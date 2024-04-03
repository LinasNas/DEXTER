# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import os
import pickle
from collections import namedtuple

import numpy as np
from tqdm import tqdm

rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])


def episodes_mu_sigma(episodes):
    flat_obs = np.concatenate([ep.states for ep in episodes])
    flat_acts = np.concatenate([ep.actions for ep in episodes])
    obs_mu = flat_obs.mean(0)
    obs_sigma = flat_obs.std(0)
    acts_mu = flat_acts.mean(0)
    acts_sigma = flat_acts.std(0)
    return obs_mu, obs_sigma, acts_mu, acts_sigma


def normalize_episodes(episodes, obs_mu, obs_sigma, acts_mu, acts_sigma):
    normalized_episodes = []
    for ep in episodes:
        norm_states = (ep.states - obs_mu) / obs_sigma
        norm_actions = (ep.actions - acts_mu) / acts_sigma
        normalized_episodes.append(rollout(norm_states, norm_actions, ep.rewards, ep.dones))
    return normalized_episodes


def split_train_test_episodes(episodes, test_split=0.1, shuffle=True):
    if shuffle:
        np.random.shuffle(episodes)
    i_split = int(len(episodes) * (1 - test_split))
    train_episodes = episodes[:i_split]
    test_episodes = episodes[i_split:]
    return train_episodes, test_episodes


def policy_rollout(env, policy, max_steps=2e3):
    states, actions, rewards, dones = (
        [],
        [],
        [],
        [],
    )
    state = env.reset()
    states.append(state)
    if hasattr(policy, "reset"):
        policy.reset()
        
    done = False
    while not done:
        action, _ = policy.predict(state, deterministic=True)
        n_state, reward, done, _info = env.step(action)
        states.append(n_state)
        actions.append(action)
        rewards.append(reward)
        state = n_state.copy()
        if len(states) > max_steps:
            print("aborting long episode")
            done = True
        dones.append(done)

    return np.array(states), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)


def n_policy_rollouts(env, policy, n_episodes, verbose=False):
    episodes = []
    trange = tqdm(range(n_episodes), position=0, desc="episode", ncols=80) if verbose else range(n_episodes)
    for n in trange:
        episodes.append(rollout(*policy_rollout(env, policy)))
    return episodes


def random_rollout(env):
    states, actions, rewards = [], [], []
    state = env.reset()
    states.append(state)
    done = False
    while not done:
        action = env.action_space.sample()
        n_state, reward, done, _info = env.step(action)
        states.append(n_state)
        actions.append(action)
        rewards.append(reward)
    return np.array(states), np.array(actions), np.array(rewards)


def save_object(object, savepath):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "wb") as f:
        pickle.dump(object, f)
    print("saved at:", savepath)


def load_object(savepath):
    with open(savepath, "rb") as f:
        data = pickle.load(f)
    print("loaded from:", savepath)
    return data
