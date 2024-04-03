# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import multiprocessing as mp
import time
from multiprocessing import Pool

#import pynvml
import torch

#pynvml.nvmlInit()


def convert_size(b, unit="MB"):
    size_name = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}
    i = size_name[unit]
    p = 1024**i
    return f"{b/p:.0f}{unit}"


# def get_next_avail_gpu(t_inter=10, verbose=True, device_ids=None):

#     if not device_ids:
#         device_ids = range(pynvml.nvmlDeviceGetCount())

#     while True:
#         for i in device_ids:
#             h = pynvml.nvmlDeviceGetHandleByIndex(i)
#             info = pynvml.nvmlDeviceGetMemoryInfo(h)
#             util_t1 = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
#             time.sleep(2)
#             util_t2 = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
#             if util_t1 < 1 and util_t2 < 1 and info.free > info.total * 0.5:
#                 if verbose:
#                     print(
#                         f"next free gpu: {i} (mem: {convert_size(info.free)}/{convert_size(info.total)} util: {util_t2}%)"
#                     )
#                 return f"cuda:{i}"
#         if verbose:
#             print(f"no free gpu, waiting {t_inter} secs")
#         time.sleep(t_inter)


def run_multi_gpu(f, configs, attr="device", device_ids=None):
    for c in configs:
        device = get_next_avail_gpu(device_ids=device_ids)
        c.__setattr__(attr, device)
        proc = mp.Process(target=f, args=(c,))
        proc.start()
        time.sleep(1)


def run_single_gpu(f, configs, n_procs=4, attr="device", device="cuda"):
    for c in configs:
        c.__setattr__(attr, device)
    with Pool(processes=n_procs) as p:
        p.map(f, configs, chunksize=1)


def get_device(device_str="auto"):
    if device_str == "auto":
        return torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif device_str == "multi":
        return torch.device(get_next_avail_gpu(t_inter=1, verbose=False))
    else:
        try:
            return torch.device(device_str)
        except Exception as e:
            print("handling device error:")
            print(e)
