# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import gym


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def do_rollout(self, env):
        raise NotImplementedError

    def predict(self, state, *args, **kwargs):
        return self.env.action_space.sample(), None

    def learn():
        pass
