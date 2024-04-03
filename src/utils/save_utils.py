# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import os

import pandas as pd
import yaml


def save_results(cfg, results_dict, conf_d=None):

    os.makedirs(cfg.results_save_dir, exist_ok=True)
    print("saving results at: ", cfg.results_save_dir)

    df = pd.DataFrame(results_dict.values())
    df.to_hdf(os.path.join(cfg.results_save_dir, "results.h5"), key="df")
    yaml.dump(dict(vars(cfg)), open(os.path.join(cfg.results_save_dir, "cfg.yaml"), "w"))
    # conf_df.to_csv(os.path.join(save_dir, "conf_matrix.csv"))
