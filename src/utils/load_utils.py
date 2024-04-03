# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import os
import sys

from detectors.haider.classical import *
from detectors.haider.pedm.pedm_detector import PEDM_Detector

RIQN_REPO_PATH = "../baselines/riqn/"
sys.path.append(RIQN_REPO_PATH)
# gym.logger.set_level(40)


def load_policy(policy_name, env, device, path=None):
    if policy_name == "RIQN":
        from riqn_policy import RIQN_Policy

        path = path or os.path.join(RIQN_REPO_PATH, "models", env.spec.id, "policy.pt")
        policy = RIQN_Policy.load(path, env=env, device=device)

    elif policy_name == "TD3":
        from stable_baselines3 import TD3

        if env.spec.id in {"AntBulletEnv-v0", "HalfCheetahBulletEnv-v0", "HopperBulletEnv-v0", "Walker2DBulletEnv-v0"}:
            path = path or os.path.join(RIQN_REPO_PATH, "models", env.spec.id, "best_model.zip")
        elif env.spec.id in {"MJCartpole-v0", "MJHalfCheetah-v0", "MJReacher-v0", "MJPusher-v0"}:
            path = path or os.path.join("data", "checkpoints", env.spec.id, "TD3", "best_model.zip")
        policy = TD3.load(path, env=env, device=device)

    elif policy_name == "PETS":
        from pets.pets import PETS

        path = path or os.path.join("data", "checkpoints", env.spec.id, "PETS", "pets.pt")
        policy = PETS.load(path, device=device)

    return policy


def load_detector(detector_name, detector_path):

    try:
        detector_cls = globals()[detector_name]
    except Exception as e:
        raise ValueError(f"class of detector < {detector_name} not found;", e)

    detector = detector_cls.load(detector_path)

    return detector
