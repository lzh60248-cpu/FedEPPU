"""Paired data protocol for fully supervised and PU comparison methods."""
from dataclasses import dataclass
from typing import List

import numpy as np

from datasets.dataset import (
    binarize_labels,
    get_chest_validation,
    get_chest_xray,
    get_cifar10,
    get_fashion_mnist,
    get_kolektor_sdd2,
)


@dataclass
class ClientData:
    x: np.ndarray
    observed_y: np.ndarray
    true_y: np.ndarray
    prior: float


@dataclass
class ProtocolData:
    clients: List[ClientData]
    test_x: np.ndarray
    test_y: np.ndarray
    input_channels: int
    proxy_x: np.ndarray = None
    proxy_y: np.ndarray = None


CONFIGS = {
    "fashion_mnist": (1000, 57976, 1024, 1),
    "cifar10": (1000, 47976, 1024, 3),
    "chest": (1000, 3960, 256, 1),
    "kolektor_sdd2": (120, 1900, 256, 3),
}


def _raw_binary(dataset, chest_root=None, chest_val_root=None, kolektor_root=None):
    if dataset == "fashion_mnist":
        train, test = get_fashion_mnist()
    elif dataset == "cifar10":
        train, test = get_cifar10()
    elif dataset == "chest":
        train, test = get_chest_xray(root=chest_root)
        # Validate the same required external path as the formal experiment.
        get_chest_validation(chest_val_root)
    elif dataset == "kolektor_sdd2":
        train, test = get_kolektor_sdd2(root=kolektor_root)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    x_tr, y_tr = train
    x_te, y_te = test
    if dataset not in ("chest", "kolektor_sdd2"):
        y_tr, y_te = binarize_labels(y_tr, y_te, dataset)
    return x_tr, y_tr.astype(np.int32), x_te, y_te.astype(np.int32)


def _split_indices(y, n_clients, alpha, min_pos_samples):
    client_idx = [[] for _ in range(n_clients)]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        floor = min_pos_samples if label == 1 else 10
        if len(idx) < floor * n_clients:
            floor = len(idx) // n_clients
        if floor > 0:
            for client in range(n_clients):
                client_idx[client].extend(idx[:floor])
                idx = idx[floor:]
        if len(idx) > 0:
            probabilities = np.random.dirichlet(np.repeat(alpha, n_clients))
            counts = np.random.multinomial(len(idx), probabilities)
            splits = np.split(idx, np.cumsum(counts)[:-1])
            for client in range(n_clients):
                client_idx[client].extend(splits[client])
    return [np.asarray(idx, dtype=np.int64) for idx in client_idx]


def _make_client(x, true_y, n_labeled):
    pos_idx = np.where(true_y == 1)[0]
    neg_idx = np.where(true_y == -1)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    n_labeled = min(int(n_labeled), len(pos_idx))
    labelled = pos_idx[:n_labeled]
    unlabeled = np.concatenate((pos_idx[n_labeled:], neg_idx))
    np.random.shuffle(unlabeled)
    order = np.concatenate((labelled, unlabeled))
    observed = np.concatenate((
        np.ones(len(labelled), dtype=np.int32),
        -np.ones(len(unlabeled), dtype=np.int32),
    ))
    permutation = np.random.permutation(len(order))
    prior = float(np.mean(true_y[unlabeled] == 1)) if len(unlabeled) else 0.0
    return ClientData(
        x=x[order][permutation].astype(np.float32, copy=False),
        observed_y=observed[permutation],
        true_y=true_y[order][permutation].astype(np.int32, copy=False),
        prior=prior,
    )


def load_protocol(dataset, n_clients=3, alpha=0.1, chest_root=None,
                  chest_val_root=None, kolektor_root=None):
    """Reproduce the formal split while retaining inaccessible PN truth."""
    if dataset not in CONFIGS:
        raise ValueError(f"unsupported dataset: {dataset}")
    n_labeled, n_unlabeled, n_proxy, channels = CONFIGS[dataset]
    x_tr, y_tr, x_te, y_te = _raw_binary(
        dataset, chest_root, chest_val_root, kolektor_root
    )
    required = n_proxy + n_labeled + n_unlabeled
    if required > len(x_tr):
        raise ValueError(f"{dataset} has {len(x_tr)} samples, need {required}")

    if dataset == "chest":
        n_pos = n_proxy // 2
        n_neg = n_proxy - n_pos
        pos_idx = np.random.permutation(np.flatnonzero(y_tr == 1))
        neg_idx = np.random.permutation(np.flatnonzero(y_tr == -1))
        holdout = np.concatenate((pos_idx[:n_pos], neg_idx[:n_neg]))
        np.random.shuffle(holdout)
        keep = np.ones(len(x_tr), dtype=bool)
        keep[holdout] = False
        remaining = np.flatnonzero(keep)
        np.random.shuffle(remaining)
        remaining = remaining[:n_labeled + n_unlabeled]
        proxy_idx = holdout
    else:
        permutation = np.random.permutation(len(x_tr))[:required]
        proxy_idx = permutation[:n_proxy]
        remaining = permutation[n_proxy:]

    x_rem, y_rem = x_tr[remaining], y_tr[remaining]
    base, extra = divmod(n_labeled, n_clients)
    quotas = [base + (client < extra) for client in range(n_clients)]
    raw_indices = _split_indices(
        y_rem, n_clients, alpha, max(quotas) + 20
    )
    clients = [
        _make_client(x_rem[idx], y_rem[idx], quotas[client])
        for client, idx in enumerate(raw_indices)
    ]
    return ProtocolData(
        clients, x_te.astype(np.float32, copy=False), y_te, channels,
        x_tr[proxy_idx].astype(np.float32, copy=False),
        y_tr[proxy_idx].astype(np.int32, copy=False),
    )


def pooled_pu_prior(clients):
    """Compute the pooled prior from disclosed client priors, without truth."""
    u_counts = np.asarray([(c.observed_y == -1).sum() for c in clients], float)
    priors = np.asarray([c.prior for c in clients], float)
    if u_counts.sum() == 0:
        raise ValueError("pooled PU data has no unlabeled samples")
    return float(np.dot(u_counts, priors) / u_counts.sum())
