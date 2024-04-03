# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import os

import ray
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch


def tune_model(
    fn,
    fn_kwargs,
    search_space,
    exp_name,
    save_dir,
    num_samples=50,
    metric="val_loss",
    mode="min",
    grace_period=50,
    max_epochs=500,
    total_cpus=16,
    total_gpus=4,
    cpus_per_trail=1,
    gpus_per_trial=0.25,
):

    resources = {"num_cpus": total_cpus, "num_gpus": total_gpus}
    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=True, **resources)
    print(ray.available_resources())

    train_fn = tune.with_parameters(fn, **fn_kwargs)
    train_fn = tune.with_resources(train_fn, {"cpu": cpus_per_trail, "gpu": gpus_per_trial})
    asha_scheduler = ASHAScheduler(max_t=max_epochs, grace_period=grace_period)
    hyperopt_search = HyperOptSearch()
    reporter = CLIReporter(
        print_intermediate_tables=True,
    )

    tuner = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            scheduler=asha_scheduler,
            search_alg=hyperopt_search,
        ),
        run_config=air.RunConfig(name=exp_name, local_dir=save_dir, progress_reporter=reporter, verbose=1),
        param_space=search_space,
    )

    result_grid = tuner.fit()
    print(result_grid.get_best_result(metric=metric, mode=mode))
