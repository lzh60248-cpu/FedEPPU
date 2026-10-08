"""Method-neutral checkpoint selection and threshold calibration."""
from copy import deepcopy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score


def split_proxy(x, y):
    y_np = y.detach().cpu().numpy().reshape(-1)
    pos = np.flatnonzero(y_np > 0); neg = np.flatnonzero(y_np < 0)
    if min(len(pos), len(neg)) < 2:
        raise ValueError("proxy selection/calibration requires at least two samples per class")
    sel = np.concatenate((pos[::2], neg[::2])); cal = np.concatenate((pos[1::2], neg[1::2]))
    return (x[sel], y[sel]), (x[cal], y[cal])


def model_scores(model, x, two_logit=False):
    model.eval(); parts=[]
    with torch.no_grad():
        for start in range(0, len(x), 512):
            parts.append(model(x[start:start+512]).detach().cpu())
    out=torch.cat(parts).numpy()
    return out[:, 1]-out[:, 0] if two_logit else out.reshape(-1)


def ensemble_scores(models, weights, x):
    weights=np.asarray(weights, dtype=np.float64); weights/=weights.sum()
    return sum(float(w)*model_scores(m, x) for w, m in zip(weights, models))


def metrics(labels, scores, threshold):
    labels=np.asarray(labels).reshape(-1); scores=np.asarray(scores).reshape(-1)
    pred=np.where(scores >= threshold, 1, -1)
    try: auc=float(roc_auc_score(labels, scores))
    except ValueError: auc=.5
    return {"acc":float(accuracy_score(labels,pred)), "bacc":float(balanced_accuracy_score(labels,pred)),
            "f1":float(f1_score(labels,pred,pos_label=1,zero_division=0)),
            "recall":float(recall_score(labels,pred,pos_label=1,zero_division=0)),
            "auc":auc, "threshold":float(threshold)}


def calibrated_threshold(labels, scores):
    labels=np.asarray(labels).reshape(-1); scores=np.asarray(scores).reshape(-1)
    values=np.unique(scores); candidates=np.concatenate(([np.nextafter(values[0], -np.inf)],
        (values[:-1]+values[1:])/2, [np.nextafter(values[-1], np.inf)]))
    best=None
    for threshold in candidates:
        result=metrics(labels, scores, threshold)
        rank=(result["acc"], result["recall"], -abs(float(threshold)), -float(threshold))
        if best is None or rank > best[0]: best=(rank,float(threshold))
    return best[1]


class ProxySelector:
    def __init__(self, interval=5, tolerance=.001):
        self.interval=interval; self.tolerance=tolerance; self.best_auc=-np.inf
        self.selected_auc=-np.inf; self.selected_round=None; self.state=None

    def consider(self, round_index, rounds, scores, labels, state):
        if (round_index+1) % self.interval and round_index != rounds-1: return float("nan")
        auc=metrics(labels, scores, 0.0)["auc"]
        self.best_auc=max(self.best_auc, auc)
        if auc >= self.best_auc-self.tolerance:
            self.selected_auc=auc; self.selected_round=round_index+1; self.state=deepcopy(state)
        return auc
