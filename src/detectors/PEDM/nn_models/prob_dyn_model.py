# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


from typing import Callable, Tuple

import numpy as np
import torch
from detectors.PEDM.nn_models.prob_ensemble import ProbEnsemble
from collections import namedtuple

class PEDM(ProbEnsemble):
    """Probabilistic Ensemble Dynamics Model with state_action inputs"""

    def __init__(self, obs_preproc=None, obs_postproc=None, targ_proc=None, *args, **kwargs):
        """
        Args:
            obs_preproc: function to preprocess observations (not required)
            obs_postproc: function to postprocess observations (e.g. add model output to current state)
            targ_proc: function to process targets (e.g. substract model output from next state)
        """

        super().__init__(*args, **kwargs)
        self.obs_preproc = obs_preproc or (lambda obs: obs)
        self.obs_postproc = obs_postproc or (lambda obs, pred: obs + pred)
        self.targ_proc = targ_proc or (lambda obs, n_obs: n_obs - obs)
        self._constructor_kwargs.update(
            {"obs_preproc": obs_preproc, "obs_postproc": obs_postproc, "targ_proc": targ_proc}
        )

    def fit(self, train_ep_data, val_ep_data, *args, **kwargs) -> Tuple:
        """
        fit the model to the given states, actions and next_states and use eval_* for validation epochs

        Args:
            train_states: [N, state_dims]
            train_n_states: [N, state_dims]
            train_actions: [N, action_dims]
            eval_states: [N, state_dims]
            eval_n_states: [N, state_dims]
            eval_actions: [N, action_dims]

        Returns:
            (ep_train_loss, ep_eval_loss)
        """

        X_train, y_train = self.preproc_ep_data(train_ep_data)
        X_val, y_val = self.preproc_ep_data(val_ep_data)

        return super().fit(X_train, y_train, X_val, y_val, *args, **kwargs)

    @torch.no_grad()
    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor, n_part: int) -> torch.Tensor:
        """
        predict next state given current state and action

        Args:
            state: [n_part, state_dim]
            action: [n_part, action_dim]

        Returns:
            next_state: [n_part, state_dim]
        """

        # state:  (nopt * n_part, state_dim)
        # action: (n_candidates x action dim)
        # distribute the particles over the nets
        _state = self._expand(
            self.obs_preproc(state), n_part
        )  # (nopt * n_part,state_dim) -> (n_nets, n_candidates*n_part//n_nets, state_dim)
        _acs = self._expand(
            action, n_part
        )  # (n_candidates x action dim) -> (n_nets, n_candidates*n_part//n_nets, action_dim)
        inputs = torch.cat((_state, _acs), dim=-1)  # (n_nets, n_candidates*n_part//n_nets, state_dim+action_dim)

        mean, var = self.forward(inputs)
        predictions = mean + torch.randn_like(mean, device=self.device) * var.sqrt()

        # reshape the distributed particles to state-shape
        predictions = self._flatten(
            predictions, n_part
        )  # (n_nets, n_candidates*n_part//n_nets, state_dim) -> (nopt * n_part, state_dim)

        # self._unflatten(predictions, n_part)

        return self.obs_postproc(state, predictions)

    @torch.no_grad()
    def one_step_batch_preds(self, states: np.ndarray, actions: np.ndarray, n_part: int) -> np.ndarray:
        """
        efficient batch version of to predict multiple next states given multiple states and actions>

        Args:
            states: [n, state_dim]
            actions: [n, action_dim]

        Returns:
            next_states: [n, n_part, state_dim]
        """

        st = torch.from_numpy(states).float().to(self.device).detach()
        st = st.repeat_interleave(repeats=n_part, dim=0)  # (ep_length,state_dim) -> (n_part*ep_length, state_dim)
        act = torch.from_numpy(actions).float().to(self.device).detach()
        act = act.repeat_interleave(repeats=n_part, dim=0)
        preds = self.predict_next_state(state=st, action=act, n_part=n_part)
        preds = self._unflatten(preds, n_part=n_part)
        preds = preds.reshape(states.shape[0], n_part, -1)
        return preds.cpu().numpy()

    @torch.no_grad()
    def predict_n_states(self, state: np.ndarray, actions: np.ndarray, n_part: int) -> torch.Tensor:
        """
        takes a start state and a sequence of actions of length n_ and predicts the n_ next states

        Args:
            state: [state_dim]
            actions: [seq_len, action_dim]

        Returns:
            next_state: [seq_len, n_part, state_dim]
        """

        n_states = []
        s_t = torch.from_numpy(state).float().to(self.device)
        s_t = s_t.unsqueeze(0)  # (state_dim,) -> (state_dim)
        s_t = s_t.expand(n_part, -1)
        for a in actions:
            a_t = torch.from_numpy(a).float().to(self.device)
            a_t = a_t.unsqueeze(0)  # (action_dim,) -> (action_dim)
            a_t = a_t.expand(n_part, -1)
            s_t = self.predict_next_state(s_t, a_t, n_part=n_part)
            n_states.append(s_t.detach().clone())
        return n_states

    @torch.no_grad()
    def predict_mean_var(self, states: np.ndarray, actions: np.ndarray, n_part=5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        batch predict mean and variance of the next state given current states and actions

        Args:
            state: [n, n_part, state_dim]
            action: [n, n_part, action_dim]

        Returns:
            mean, var: ([n, n_part, state_dim], [n, n_part, state_dim])
        """

        st = torch.from_numpy(states).half().to(self.device).detach()
        st = st.repeat_interleave(repeats=n_part, dim=0)  # (ep_length,state_dim) -> (n_part*ep_length, state_dim)
        act = torch.from_numpy(actions).float().to(self.device).detach()
        act = act.repeat_interleave(repeats=n_part, dim=0)

        # distribute the particles over the nets
        _state = self._expand(self.obs_preproc(st), n_part)
        _acs = self._expand(act, n_part)
        inputs = torch.cat((_state, _acs), dim=-1)

        mean, var = self.forward(inputs)

        mean = self._flatten(mean, n_part)
        mean = self.obs_postproc(st, mean)
        mean = self._unflatten(mean, n_part=n_part)
        mean = mean.reshape(states.shape[0], n_part, -1)
        var = self._unflatten(var, n_part=n_part)
        var = var.reshape(states.shape[0], n_part, -1)
        return mean, var

    def preproc_ep_data(self, ep_data):
        # stkd_obs = np.array([ep.states for ep in ep_data])[:, :-1, :]
        # stkd_n_obs = np.array([ep.states for ep in ep_data])[:, 1:, :]
        # stkd_acts = np.array([ep.actions for ep in ep_data])
        # stkd_obs = np.concatenate(stkd_obs, axis=0)
        # stkd_n_obs = np.concatenate(stkd_n_obs, axis=0)
        # stkd_acts = np.concatenate(stkd_acts, axis=0)

        #Allow for variable length episodes
        stkd_obs = [ep.states[:-1] for ep in ep_data]
        stkd_n_obs = [ep.states[1:] for ep in ep_data]
        stkd_acts = [ep.actions for ep in ep_data]

        stkd_obs = np.concatenate(stkd_obs, axis=0)
        stkd_n_obs = np.concatenate(stkd_n_obs, axis=0)
        stkd_acts = np.concatenate(stkd_acts, axis=0)

        X = np.concatenate([self.obs_preproc(stkd_obs), stkd_acts], axis=-1)
        y = self.targ_proc(stkd_obs, stkd_n_obs)

        return X, y

    def save(self, path) -> None:
        """save the model to the given path"""

        print("saving PEDM model at: ", path)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "constructor_kwargs": self._constructor_kwargs,
                "save_attrs": self._save_attrs,
            },
            f=path,
        )

    @classmethod
    def load(cls, path, device="auto") -> None:
        """load the model from the given path"""

        print("loading PEDM model from: ", path)
        saved_variables = torch.load(path)
        model = cls(**saved_variables["constructor_kwargs"], device=device)
        model.load_state_dict(saved_variables["state_dict"])
        model._save_attrs = saved_variables["save_attrs"]
        for k, v in model._save_attrs.items():
            setattr(model, k, v)
        return model
