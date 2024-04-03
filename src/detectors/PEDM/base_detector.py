# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import os
from abc import ABC, abstractmethod
from typing import List

import cloudpickle
import numpy as np
import torch
from utils.data import episodes_mu_sigma, normalize_episodes


class Base_Detector(ABC):
    """
    Base class that all detectors must follow. Most importantly, every detector has to implement <_predict_scores> and <_fit>
    """

    def __init__(self, normalize_data: bool = True) -> None:
        """
        Args:
            normalize_data: pass True to normalize all data ((X-mean)/std)
        """
        self.normalize_data = normalize_data

    @abstractmethod
    def _predict_scores(self, obs, acts) -> np.ndarray:
        pass

    @abstractmethod
    def _fit(self, *args, **kwargs):
        pass

    def predict(self, obs: np.ndarray, acts: np.ndarray, *args, **kwargs) -> List[bool]:
        """given a list of transitions (sequential observations and actions) predict the label of each transition

        Args:
            obs: [seq_len, state_dims] list of sequential observations
            acts: [seq_len, action_dims] list of sequential actions

        Returns:
            label: boolean label of each state/transition
        """
        return self.predict_scores > self.th

    def predict_scores(self, obs: np.ndarray, acts: np.ndarray, *args, **kwargs) -> np.ndarray:
        """given a list of transitions (sequential observations and actions) predict the anomaly score of each transition

        Args:
            obs: [seq_len, state_dims] list of sequential observations
            acts: [seq_len, action_dims] list of sequential actions

        Returns:
            scroes: [seq_len, 1] anomaly score each state/transition
        """
        if self.normalize_data:
            obs = self._normalize_obs(obs)
            acts = self._normalize_acts(acts)
        return self._predict_scores(obs, acts)

    def fit(self, train_ep_data, val_ep_data, *args, **kwargs) -> None:
        """
        fit the detector to the given data
        Args:
            data_buffer: a buffer that stores training and validation data
        """

        if self.normalize_data:
            self.obs_mu, self.obs_sigma, self.acts_mu, self.acts_sigma = episodes_mu_sigma(train_ep_data)

            train_ep_data = normalize_episodes(
                train_ep_data, self.obs_mu, self.obs_sigma, self.acts_mu, self.acts_sigma
            )

            val_ep_data = normalize_episodes(val_ep_data, self.obs_mu, self.obs_sigma, self.acts_mu, self.acts_sigma)

        self._fit(train_ep_data, val_ep_data, *args, **kwargs)

    def set_threshold(self, th: float) -> None:
        self.th = th

    def calc_threshold(self, obs: np.ndarray, acts: np.ndarray, percentile) -> float:
        """
        calculate the threshold of the detecotr s.t. the percentile of the data are below this threshold

        Args:
            obs: [seq_len, state_dims] list of sequential observations
            acts: [seq_len, action_dims] list of sequential actions
        """
        self.percentile = percentile
        scores = self.predict_socres(obs, acts)
        self.th = np.percentile(scores, percentile)
        return self.th

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        normalize given obs w.r.t. to the previously set normalization parameters

        Args:
            obs: [seq_len, state_dims] list of sequential observations

        Returns:
            obs: [seq_len, state_dims] list of sequential observations
        """
        return (obs - self.obs_mu) / self.obs_sigma

    def _normalize_acts(self, acts: np.ndarray) -> np.ndarray:
        """
        normalize given acts w.r.t. to the previously set normalization parameters

        Args:
            acts: [seq_len, state_dims] list of sequential actions

        Returns:
            acts: [seq_len, state_dims] list of sequential actions
        """
        return (acts - self.acts_mu) / self.acts_sigma

    def save(self, file_path: str) -> None:
        """save the model to the given path"""

        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self._save(file_path)

    def _save(self, file_path):
        print("saving detector at: ", file_path)
        torch.save({"detector_obj": self}, f=file_path, pickle_module=cloudpickle)

    @classmethod
    def load(cls, file_path):
        print("loading detector from: ", file_path)
        saved_variables = torch.load(file_path)
        detector_obj = saved_variables["detector_obj"]
        return detector_obj