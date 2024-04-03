# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


def default_obs_preproc(full_obs):
    return full_obs


def default_obs_postproc(obs, pred):
    return obs + pred


def default_targ_proc(obs, next_obs):
    return next_obs - obs


model_cfg_dict = {
    "Acrobot-v1": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [7, 500, 500, 500, 6],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },
    "CartPole-v1": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [5, 500, 500, 500, 4],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },
    "LunarLander-v2": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [9, 500, 500, 500, 8],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },
    #LINAS: add cartpole-v0, with the same params as Cartpole-v0 (since MJCartpole-v0 also has the same params as Cartpole-v1)
    "CartPole-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [5, 500, 500, 500, 4],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },

    "IMANOCartpoleEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [5, 500, 500, 500, 4],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },

    "IMANSCartpoleEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [5, 500, 500, 500, 4],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },

    "TimeSeriesEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [2, 500, 500, 500, 1],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },

    "TestModCartpoleEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [2, 500, 500, 500, 1],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },

    "IMANOAcrobotEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [7, 500, 500, 500, 6],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },

    "AntBulletEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "decays": [0.00025, 0.0005, 0.0005, 0.0005, 0.00075],
            "layer_sizes": [36, 200, 200, 200, 200, 28],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },
    "HalfCheetahBulletEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "decays": [0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
            "layer_sizes": [32, 200, 200, 200, 200, 26],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },
    "HopperBulletEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "decays": [0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
            "layer_sizes": [18, 200, 200, 200, 200, 15],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },
    "Walker2DBulletEnv-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "decays": [0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
            "layer_sizes": [28, 200, 200, 200, 200, 22],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },
    "MJCartpole-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [5, 500, 500, 500, 4],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },

    "CartPole-v1": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [5, 500, 500, 500, 4],
            "decays": [0.0001, 0.00025, 0.00025, 0.0005],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
            "lr": 0.001,
        }
    },

    "MJHalfCheetah-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [24, 200, 200, 200, 200, 18],
            "decays": [0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
        }
    },
    "MJReacher-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [24, 200, 200, 200, 17],
            "decays": [0.00025, 0.0005, 0.0005, 0.00075],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
        }
    },
    "MJPusher-v0": {
        "dyn_model_kwargs": {
            "ens_size": 5,
            "layer_sizes": [27, 200, 200, 200, 20],
            "decays": [0.00025, 0.0005, 0.0005, 0.00075],
            "obs_preproc": default_obs_preproc,
            "obs_postproc": default_obs_postproc,
            "targ_proc": default_targ_proc,
        }
    },
}
