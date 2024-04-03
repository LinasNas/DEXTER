# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import os
import pickle
import sys
from typing import Optional

import numpy as np
import torch
from scipy.special import ndtr
from torch.distributions.multivariate_normal import MultivariateNormal

sys.path.append("../nn_models")

from detectors.PEDM.base_detector import Base_Detector
from utils.callbacks import TrainCallback
from utils.stats import one_step_batch_stats  # noqa

from detectors.PEDM.nn_models.default_cfg import model_cfg_dict
from detectors.PEDM.nn_models.prob_dyn_model import PEDM  # noqa


class PEDM_Detector(Base_Detector):
    """Probabilistic Ensemble Dynamics Model ODD Detector for RL agents"""

    def __init__(
        self,
        env,
        num_train_episodes,
        dyn_model_kwargs={},
        n_part: Optional[int] = 1_000,
        horizon: Optional[int] = 1,
        criterion: Optional[str] = "pred_error_samples",
        aggregation_function: Optional[str] = "min_mean",
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            dyn_model: (probabilistic) dynamics model for the detector
            n_part: number of particles to sample from the detector
            criterion: specifies method to calculate anomaly score from model predictions
            aggregation_function: specifies method to aggregate anomaly scores for multiple particles
            normalize_data: flag to normalize all data ((X-mean)/std)
        """
        if not dyn_model_kwargs:
            dyn_model_kwargs = model_cfg_dict[env.spec.id]["dyn_model_kwargs"]

        self.dyn_model = PEDM(**dyn_model_kwargs)
        self.n_part = n_part
        self.criterion = criterion
        self.aggregation_function = aggregation_function
        
        self.num_train_episodes = num_train_episodes

        if horizon != 1:
            raise NotImplementedError
        super().__init__(*args, **kwargs)

    def _predict_scores(self, obs, acts) -> np.ndarray:
        """given a list of transitions (sequential observations and actions) predict the anomaly score of each transition based on the criterion and aggregation function

        Args:
            obs: [seq_len, state_dims] list of sequential observations
            acts: [seq_len, action_dims] list of sequential actions

        Returns:
            scroes: [seq_len, 1] anomaly score each state/transition
        """

        if not isinstance(self.dyn_model, PEDM):
            raise ValueError("dyn model not initialized")

        if self.criterion == "pred_error_samples":
            preds = self.dyn_model.one_step_batch_preds(states=obs[:-1], actions=acts, n_part=self.n_part)
            pred_err, pred_std = one_step_batch_stats(preds, obs[1:])
            return self._aggregate(pred_err)

        elif self.criterion == "pred_std_samples":
            preds = self.dyn_model.one_step_batch_preds(states=obs[:-1], actions=acts, n_part=self.n_part)
            pred_err, pred_std = one_step_batch_stats(preds, obs[1:])
            return self._aggregate(pred_std)

        elif self.criterion == "pred_error_pdf":
            mean, var = self.dyn_model.predict_mean_var(states=obs[:-1], actions=acts, n_part=self.dyn_model.ens_size)
            n_obs = torch.from_numpy(obs[1:]).to(mean.device)
            log_probs = []
            for i in range(self.dyn_model.ens_size):
                m, v = mean[:, i, :], var[:, i, :]
                lp = MultivariateNormal(m, torch.diag_embed(v)).log_prob(n_obs)
                log_probs.append(lp.cpu().numpy())
            log_probs = np.vstack(log_probs)
            return -self._aggregate(log_probs)

        elif self.criterion == "p_value":
            mean, var = self.dyn_model.predict_mean_var(states=obs[:-1], actions=acts, n_part=self.dyn_model.ens_size)
            mean = mean.cpu().numpy()
            std = var.sqrt().cpu().numpy() * 10
            n_obs = obs[1:]
            p_values = []

            for i in range(self.dyn_model.ens_size):
                m, s = mean[:, i, :], std[:, i, :]
                z = (n_obs - m) / s
                p = np.prod(ndtr(-np.abs(z)), axis=1)
                p_values.append(p)

            p_values = np.array(p_values)
            return -self._aggregate(p_values)

        else:
            raise ValueError

    def _aggregate(self, X) -> np.ndarray:
        """
        aggregation function for individial particle's anomaly score

        Args:
            X: [seq_len, n_part, part_dim]

        Returns:
            aggr_score: [seq_len]
        """
        if self.criterion in {"pred_error_samples", "pred_std_samples"}:
            aggr_axis = -1
        else:
            aggr_axis = 0

        if self.aggregation_function == "min_mean":
            return np.min(np.mean(X, axis=aggr_axis), axis=aggr_axis)
        elif self.aggregation_function == "max_mean":
            return np.max(np.mean(X, axis=aggr_axis), axis=aggr_axis)
        elif self.aggregation_function == "mean_mean":
            return np.mean(np.mean(X, axis=aggr_axis), axis=aggr_axis)
        elif self.aggregation_function == "median_mean":
            return np.median(np.mean(X, axis=aggr_axis), axis=aggr_axis)
        elif self.aggregation_function == "min":
            return np.min(X, axis=aggr_axis)
        elif self.aggregation_function == "max":
            return np.max(X, axis=aggr_axis)
        elif self.aggregation_function == "mean":
            return np.mean(X, axis=aggr_axis)
        elif self.aggregation_function == "median":
            return np.median(X, axis=aggr_axis)
        else:
            raise NotImplementedError

    # def _fit(self, train_ep_data, val_ep_data, n_train_epochs=2_000, *args, **kwargs) -> None:
    def _fit(self, train_ep_data, val_ep_data, *args, **kwargs) -> None:

        """
        fit the detector to the given data or load if it's already available
        Args:
            data_buffer: a buffer that stores training and validation data
            env_id: name of the environment (used for loading/saving the model)
            n_train_epochs: number of epochs to train for
            dyn_model_kwargs: optional kwargs to pass to the dyn_model constructor
            batch_size: for training the dyn_model
        """
        
        train_callback = TrainCallback(scheduler=None, patience=1e6, stop_loss=0.0)
        self.dyn_model.fit(
            train_ep_data, val_ep_data, n_train_epochs=self.num_train_episodes, callback=train_callback, *args, **kwargs
        )
