# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


from collections import defaultdict

import numpy as np
import sklearn
import sklearn.metrics

from utils.data import policy_rollout
from utils.wrappers import TimeFeatureWrapper


def n_step_loss(states, actions, model, hor=1):
    """
    given a sequences of states and corresponding actions, calculate the n_step prediction
    loss for each state that is, in each state, roll out the sequence of next hor actions
    on the model and compare the resulting state to the actual state from the state-sequence
    """
    all_errors = []
    for t in range(len(states) - hor):
        st, acs = states[t], actions[t : t + hor]
        target = states[t + hor].copy()
        pred = model.predict_n_states(state=st, actions=acs)[-1].cpu().numpy()
        sq_error = (pred - target) ** 2
        all_errors.append(sq_error.sum(-1))
    return np.array(all_errors)


def n_step_std(states, actions, model, hor=1):
    """
    given a sequences of states and corresponding actions, calculate the std of the prediction for the nth state
    that is, in each state, roll out the sequence of next hor actions
    on the model and calculate the std of the individual predictions from the p_ensemble
    """
    total_std = []
    for t in range(len(states) - hor):
        st, acs = states[t], actions[t : t + hor]
        target = states[t + hor].copy()
        pred = model.predict_n_states(state=st, actions=acs)[-1].cpu().numpy()
        mean_std = pred.std(axis=0).mean()
        total_std.append(mean_std)
    return np.array(total_std)


def n_step_stats(states, next_states, actions, model, n_part, hor=1):
    """
    given a sequences of states and corresponding actions, calculate statistics consisting of
    (predictions_errors, predictions_stds, predictions) for prediction horizon hor
    """
    err_lst, std_lst, pred_lst = [], [], []
    for t in range(len(states) - hor):
        st, acs = states[t], actions[t : t + hor]
        target = next_states[t + hor - 1].copy()
        pred = model.predict_n_states(state=st, actions=acs, n_part=n_part)[-1].cpu().numpy()
        sq_error = (pred - target) ** 2
        pred_err = sq_error
        pred_std = pred.std(axis=0)
        std_lst.append(pred_std)
        err_lst.append(pred_err)
        pred_lst.append(pred)
    return np.array(err_lst), np.array(std_lst), np.array(pred_lst)


def one_step_batch_stats(preds, targets):
    """
    given a batch predictions and targets calculate the squared error and the std over the predictions

    Args:
        preds: [seq_len, n_part, target_dim]
        targets: [seq_len, target_dim]

    Returns:
        sq_error: [seq_len, n_part, target_dim]
        std: [seq_len, target_dim]
    """
    sq_error = (preds - targets[:, None, :]) ** 2
    std = preds.std(axis=1)
    return sq_error, std


#ORIGINAL
# def eval_metrics(scores, anom_occurrence, pos_label=None):
#     fpr, tpr, _thresholds = sklearn.metrics.roc_curve(anom_occurrence, scores)
#     auroc = sklearn.metrics.auc(fpr, tpr)
#     ap = sklearn.metrics.average_precision_score(anom_occurrence, scores)
    
#     fpr95 = fpr[np.where(tpr > 0.95)[0][0]]
#     #fpr95 = fpr[np.where(tpr > 0.8)[0][0]]
#     if fpr95 == []:
#         fpr95 = 0
        
#     return auroc, ap, fpr95

#MODIFIED
def eval_metrics(scores, anom_occurrence):
    fpr, tpr, _thresholds = sklearn.metrics.roc_curve(anom_occurrence, scores)
    auroc = sklearn.metrics.auc(fpr, tpr)
    ap = sklearn.metrics.average_precision_score(anom_occurrence, scores)

    #GET FPR where TPR is >95%
    fpr95 = fpr[np.where(tpr > 0.95)[0][0]]
    if fpr95.size == 0:
        fpr95 = 0

    #ADDITION:
    # Get TPR where FPR < 5%
    idx_tpr95 = np.where(fpr < 0.05)[0]
    if idx_tpr95.size:
        tpr95 = tpr[idx_tpr95[-1]]  # Get the last TPR value before FPR increases.
    else:
        tpr95 = 1

    # return auroc, ap, fpr95, tpr95
    return auroc

def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def acc_scores(y_true, y_scores, tau):
    y_pred = y_scores > tau
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    return acc


def conf_scores(y_true, y_scores, tau):
    y_pred = y_scores > tau
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return {"tnr": tn / (tn + fp), "fpr": fp / (fp + tn), "fnr": fn / (fn + tp), "tpr": tp / (tp + fn)}


def f1_score(y_true, y_scores, tau):
    y_pred = y_scores > tau
    f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred)
    return f1


def pred_metrics(y_true, y_scores, tau):
    acc = acc_scores(y_true=y_true, y_scores=y_scores, tau=tau)
    conf_mat = conf_scores(y_true=y_true, y_scores=y_scores, tau=tau)
    f1 = f1_score(y_true=y_true, y_scores=y_scores, tau=tau)
    return acc, conf_mat, f1
