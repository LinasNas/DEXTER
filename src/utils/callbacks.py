# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


from copy import deepcopy
from datetime import datetime as dt
from time import time
# from ray import tune # requires ray installation


class BaseCallback:
    def __init__(self):
        super().__init__()

    def init_callback(self, _locals):
        pass

    def on_train_begin(self, _locals):
        pass

    def on_train_end(self, _locals):
        pass

    def on_ep_begin(self, _locals):
        pass

    def on_ep_end(self, _locals):
        pass


class TrainCallback(BaseCallback):
    def __init__(self, patience=10, stop_loss=0.001, scheduler=None, min_lr=1e-5):
        super().__init__()
        self.patience = patience
        self.stop_loss = stop_loss
        self._scheduler = scheduler
        self.min_lr = min_lr
        #LINAS
        #self.p_ctr = 0  # Add this attribute and initialize it to 0

    def init_callback(self, _locals):
        if self._scheduler == "ReduceLROnPlateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            self.scheduler = ReduceLROnPlateau(_locals.get("self").optim, "min", patience=10, min_lr=self.min_lr)

    def on_train_begin(self, _locals):
        self.best_loss = float("inf")
        now = dt.now().strftime("%Y-%m-%d_%H-%M")
        print(f"starting training at: {now}")

    def on_ep_begin(self, _locals):
        self.t0 = time()

    def on_ep_end(self, _locals):
        ep_train_loss = _locals.get("ep_train_loss")
        ep_val_loss = _locals.get("ep_val_loss")
        ep = _locals.get("ep")
        n_train_epochs = _locals.get("n_train_epochs")
        model = _locals.get("self")
        if self.best_loss > ep_val_loss:
            self.p_ctr = 0
            self.best_state_dict = deepcopy(model.state_dict())
            self.best_loss = ep_val_loss
        else:
            self.p_ctr += 1

        print(
            f'epoch:{ep+1}/{n_train_epochs}  train_loss:{ep_train_loss:.4f} val_loss:{ep_val_loss:.4f} best_val_loss:{self.best_loss:.4f} current lr: {model.optim.param_groups[0]["lr"]} time p.e.: {time()-self.t0:.3f}     ',
            end="\n",  # Use end="\n" to add a newline after each line
            flush=True,
        )

        if ep_val_loss <= self.stop_loss:
            print(f"\n aborting (stop loss reached)")
            return True

        elif self.p_ctr >= self.patience:
            print(f"\n aborting (patience reached)")
            return True

        if self._scheduler:
            prev_lr = model.optim.param_groups[0]["lr"]
            self.scheduler.step(ep_val_loss)
            next_lr = model.optim.param_groups[0]["lr"]
            if prev_lr != next_lr:
                self.p_ctr = 0

        def on_train_end(self, _locals):
            model = _locals.get("self")
            model.load_state_dict(self.best_state_dict)
            print("best_loss:", self.best_loss)


class TuneReportCallBack(BaseCallback):
    def on_ep_end(self, _locals):
        val_loss = _locals.get("ep_val_loss")
        tune.report(val_loss=val_loss)
