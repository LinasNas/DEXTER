# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
import torch
from scipy.stats import truncnorm
from torch import nn as nn
from torch.nn import functional as F
from utils.callbacks import BaseCallback
from utils.gpu_utils import get_device


def truncated_normal(size, std, mean=0.0):
    "values more than two standard deviations from the mean are discarded and re-draw"
    return truncnorm.rvs(-2, 2, loc=mean, scale=std, size=size)


def get_affine_params(ens_size, in_features, out_features):
    w = truncated_normal(size=(ens_size, in_features, out_features), std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
    b = nn.Parameter(torch.zeros(ens_size, 1, out_features))
    return w, b


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class LinLayer(nn.Module):
    def __init__(self, ens_size, in_f, out_f, activation="swish"):
        super().__init__()
        self.lin_w, self.lin_b = get_affine_params(ens_size, in_f, out_f)
        if activation:
            self.activation = nn.ModuleDict(
                [
                    ["lrelu", nn.LeakyReLU()],
                    ["relu", nn.ReLU()],
                    ["swish", nn.SiLU()],
                ]
            )[activation]
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = x.matmul(self.lin_w) + self.lin_b
        x = self.activation(x)
        return x


class ProbEnsemble(nn.Module):
    """Probabilistic Ensemble"""

    def __init__(
        self, ens_size, layer_sizes, decays=None, normalize_data=True, activation_fn="swish", lr=0.001, device="auto"
    ):
        super().__init__()

        self.ens_size = ens_size
        self.in_features = layer_sizes[0]
        self.out_features = layer_sizes[-1] * 2  # out_features * 2 because we output both the mean and the variance
        self.decays = decays
        self.activation_fn = activation_fn

        self.fc_layers = nn.Sequential(
            *[
                LinLayer(ens_size, in_f, out_f, activation=self.activation_fn)
                for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:-1])
            ]
        )
        self.fc_layers.add_module("out_layer", LinLayer(ens_size, layer_sizes[-2], self.out_features, activation=None))

        self.inputs_mu = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, self.out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, self.out_features // 2, dtype=torch.float32) * 10.0)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = get_device(device)
        self.to(self.device)
        self.normalize_data = normalize_data

        self._constructor_kwargs = {"ens_size": ens_size, "layer_sizes": layer_sizes, "decays": decays, "lr": lr}
        self._save_attrs = {}

    def compute_decays(self):
        dec = []
        for lin_dec, layer in zip(self.decays, self.fc_layers):
            dec.append(lin_dec * (layer.lin_w**2).sum() / 2.0)
        return sum(dec)

    def norm_data(self, X_train):
        mu = torch.mean(X_train, dim=0, keepdims=True)
        sigma = torch.std(X_train, dim=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        self.inputs_mu.data = mu.to(self.device).float()
        self.inputs_sigma.data = sigma.to(self.device).float()

    def forward(self, inputs, ret_logvar=False):
        if self.normalize_data:
            inputs = (inputs - self.inputs_mu) / self.inputs_sigma  # normalize inputs first
        inputs = self.fc_layers(inputs)  # forward pass through layers
        mean = inputs[:, :, : (self.out_features // 2)]  # 1st half of out-features is mean
        logvar = inputs[:, :, (self.out_features // 2) :]  # 2nd half of out-features is var
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if ret_logvar:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    @torch.no_grad()
    def predict(self, inputs, n_part=20):
        # distribute the particles over the nets
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(self.device).float()
        inputs = inputs.repeat_interleave(repeats=n_part, dim=0)
        inputs = self._expand(inputs, n_part)
        mean, var = self.forward(inputs)
        predictions = mean + torch.randn_like(mean, device=self.device) * var.sqrt()
        predictions = self._flatten(predictions, n_part)
        return predictions

    def fit(self, X_train, y_train, X_val, y_val, n_train_epochs, batch_size=512, callback=BaseCallback()):
        callback.init_callback(_locals=locals())
        callback.on_train_begin(_locals=locals())

        X_train = torch.from_numpy(X_train).to(self.device).float()
        y_train = torch.from_numpy(y_train).to(self.device).float()
        X_val = torch.from_numpy(X_val).to(self.device).float()
        y_val = torch.from_numpy(y_val).to(self.device).float()

        if self.normalize_data:
            self.norm_data(X_train)

        idxs = np.random.randint(len(X_train), size=[self.ens_size, len(X_train)])
        for ep in range(n_train_epochs):
            callback.on_ep_begin(_locals=locals())
            ep_train_loss = self.train_epoch(X_train, y_train, idxs=idxs, batch_size=batch_size)
            ep_val_loss = self.val_epoch(X_val, y_val).mean() if len(X_val) > 0 else "no val data"
            idxs = shuffle_rows(idxs)
            if callback.on_ep_end(_locals=locals()):
                break
        result = callback.on_train_end(_locals=locals())
        self._val_threshold(X_val=X_val, y_val=y_val)
        return result if result else ep_train_loss, ep_val_loss

    def train_epoch(self, X_train, y_train, idxs, batch_size):
        self.train()
        acc_loss = []
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        for batch_num in range(num_batch):
            batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

            loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())

            if self.decays is not None:
                loss += self.compute_decays()

            input_mb = X_train[batch_idxs, :]
            target_mb = y_train[batch_idxs, :]

            mean, logvar = self.forward(input_mb, ret_logvar=True)
            inv_var = torch.exp(-logvar)

            train_losses = ((mean - target_mb) ** 2) * inv_var + logvar
            train_losses = (
                train_losses.mean(-1).mean(-1).sum()
            )  # reduce mean over last and second to last dim -> first dimension (ens_size) stays

            loss += train_losses

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            acc_loss.append(loss.item())
        return np.mean(acc_loss)

    def val_epoch(self, X_val, y_val, return_se=False):
        self.eval()
        with torch.no_grad():
            idxs = np.random.randint(len(X_val), size=[self.ens_size, len(X_val)])
            val_in = X_val[idxs, :]
            val_targ = y_val[idxs, :]
            mean, _ = self.forward(val_in)
            se_loss = (mean - val_targ) ** 2
            mse_loss = se_loss.mean(-1).mean(-1)
        if not return_se:
            return mse_loss.cpu().numpy()
        else:
            return mse_loss.cpu().numpy(), se_loss.cpu().numpy()

    def _val_threshold(self, X_val, y_val):
        mse_loss, se_loss = self.val_epoch(X_val=X_val, y_val=y_val, return_se=True)
        self.val_se_loss = se_loss.mean(-1).min(0)
        self._save_attrs["val_se_loss"] = self.val_se_loss

    def _expand(self, mat, n_part):
        # (*dims) -> (n_nets, n_candidates*n_part//n_nets, state_dim)
        dim = mat.shape[-1]
        reshaped = mat.reshape(-1, self.ens_size, n_part // self.ens_size, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.reshape(self.ens_size, -1, dim)
        return reshaped

    def _flatten(self, arr, n_part):
        # (n_nets, n_candidates*n_part//n_nets, state_dim) -> (nopt * n_part, state_dim)
        dim = arr.shape[-1]
        reshaped = arr.reshape(self.ens_size, -1, n_part // self.ens_size, dim)
        transposed = reshaped.transpose(0, 1)  #
        reshaped = transposed.reshape(-1, dim)  #
        return reshaped

    def _unflatten(self, arr, n_part):
        dim = arr.shape[-1]
        n_opt = arr.shape[0] // n_part
        reshaped = arr.reshape(n_opt, self.ens_size, -1, dim)
        # transposed = reshaped.transpose(1,0)
        # reshaped = arr.reshape(self.ens_size, -1, dim)
        # reshaped = arr.reshape(self.ens_size, -1, n_part // self.ens_size, dim)
        return reshaped

    def load(self, path):
        print("loading PE model from: ", path)
        self.load_state_dict(torch.load(path))

    def save(self, path):
        print("saving PE model at: ", path)
        torch.save(self.state_dict(), path)
