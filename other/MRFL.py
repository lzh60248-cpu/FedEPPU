"""MRFL-nnPU adaptation for the binary federated PU protocol.

The original MRFL method assumes fully labelled multi-class data with label
noise. This adaptation retains its two-stage structure, GMM client division,
sample reliability screening, and reliability-mass aggregation, while every
local optimiser step uses the standard binary nnPU risk.

This file is independent of FedEPPU and must be reported as ``MRFL-nnPU
adaptation``, rather than as an exact implementation of multi-class MRFL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from .fedavg_pu import local_train_fedavg


def _minmax_columns(values: np.ndarray) -> np.ndarray:
    """Apply MRFL's per-class normalization to PU loss components."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("client loss vectors must form a two-dimensional matrix")
    low = values.min(axis=0)
    span = values.max(axis=0) - low
    return np.divide(
        values - low,
        span,
        out=np.zeros_like(values),
        where=span > np.finfo(np.float64).eps,
    )


@dataclass
class MRFLState:
    """Server state for warm-up and two-component client separation."""

    num_clients: int
    warmup_rounds: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_clients < 1:
            raise ValueError("num_clients must be positive")
        if self.warmup_rounds < 1:
            raise ValueError("warmup_rounds must be positive")
        self.loss_vectors = np.zeros((self.num_clients, 2), dtype=np.float64)
        self.noisy = np.zeros(self.num_clients, dtype=bool)

    def update_client_division(
        self, loss_vectors: np.ndarray, round_idx: int
    ) -> np.ndarray:
        """Classify high-loss clients after the warm-up stage."""
        values = np.asarray(loss_vectors, dtype=np.float64)
        if values.shape != (self.num_clients, 2):
            raise ValueError(
                f"expected loss vectors with shape {(self.num_clients, 2)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("client loss vectors must be finite")
        self.loss_vectors = values.copy()
        if round_idx + 1 < self.warmup_rounds or self.num_clients < 2:
            self.noisy[:] = False
            return self.noisy.copy()

        normalized = _minmax_columns(values)
        if np.allclose(normalized, normalized[0]):
            self.noisy[:] = False
            return self.noisy.copy()
        gmm = GaussianMixture(
            n_components=2, random_state=self.seed, n_init=5, reg_covar=1e-6
        ).fit(normalized)
        high_loss_component = int(np.argmax(gmm.means_.mean(axis=1)))
        self.noisy = gmm.predict(normalized) == high_loss_component
        return self.noisy.copy()


def pu_detection_vector(model, x: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    """Return P and U loss components used only for client detection."""
    labels = t.reshape(-1)
    positive = labels == 1
    unlabeled = labels == -1
    if not torch.any(positive) or not torch.any(unlabeled):
        raise ValueError("MRFL-nnPU requires both labelled P and U samples")
    model.eval()
    p_total = 0.0
    u_total = 0.0
    p_count = 0
    u_count = 0
    with torch.no_grad():
        for start in range(0, len(x), 512):
            stop = min(start + 512, len(x))
            logits = model(x[start:stop]).reshape(-1)
            local_labels = labels[start:stop]
            p_mask = local_labels == 1
            u_mask = local_labels == -1
            if torch.any(p_mask):
                p_total += float(F.softplus(-logits[p_mask]).sum().cpu())
                p_count += int(p_mask.sum().item())
            if torch.any(u_mask):
                u_total += float(F.softplus(logits[u_mask]).sum().cpu())
                u_count += int(u_mask.sum().item())
    return np.asarray([p_total / p_count, u_total / u_count], dtype=np.float64)


def _select_reliable_pseudo_positives(
    model, x: torch.Tensor, t: torch.Tensor, prior: float, seed: int
) -> Tuple[torch.Tensor, int, int]:
    """Recover confident latent positives in U with a two-component GMM."""
    labels = t.reshape(-1)
    unlabeled_idx = torch.nonzero(labels == -1, as_tuple=True)[0]
    positive_count = int((labels == 1).sum().item())
    if unlabeled_idx.numel() < 2:
        return t.clone(), positive_count, 0

    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, unlabeled_idx.numel(), 512):
            idx = unlabeled_idx[start:start + 512]
            logits.append(model(x[idx]).reshape(-1).detach().cpu())
    scores = torch.cat(logits).numpy().reshape(-1, 1)
    if not np.isfinite(scores).all() or np.allclose(scores, scores[0]):
        return t.clone(), positive_count, 0

    gmm = GaussianMixture(
        n_components=2, random_state=int(seed), n_init=5, reg_covar=1e-6
    ).fit(scores)
    positive_component = int(np.argmax(gmm.means_.reshape(-1)))
    posterior = gmm.predict_proba(scores)[:, positive_component]
    candidates = np.flatnonzero(posterior >= 0.5)
    maximum = min(
        int(round(float(prior) * unlabeled_idx.numel())),
        int(unlabeled_idx.numel()),
    )
    if maximum <= 0 or candidates.size == 0:
        return t.clone(), positive_count, 0
    if candidates.size > maximum:
        order = np.argsort(posterior[candidates], kind="stable")[-maximum:]
        candidates = candidates[order]

    promoted = unlabeled_idx[
        torch.as_tensor(candidates, device=unlabeled_idx.device, dtype=torch.long)
    ]
    adapted = t.clone()
    adapted.reshape(-1)[promoted] = 1
    selected = int(promoted.numel())
    return adapted, positive_count + selected, selected


def local_train_mrfl_nnpu(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    prior: float,
    epochs: int,
    lr: float,
    batch_size: int,
    is_noisy: bool = False,
    seed: int = 42,
    max_steps_per_epoch=None,
):
    """Train one client with standard nnPU in both MRFL stages."""
    if is_noisy:
        adapted, reliable_mass, selected = _select_reliable_pseudo_positives(
            model, x, t, prior, seed
        )
    else:
        adapted = t
        reliable_mass = len(t)
        selected = 0
    model, diagnostics = local_train_fedavg(
        model,
        x,
        adapted,
        prior,
        epochs,
        lr,
        batch_size,
        max_steps_per_epoch=max_steps_per_epoch,
    )
    diagnostics.update(
        {
            "mrfl_stage": "robust" if is_noisy else "warmup_or_clean",
            "noisy_client": bool(is_noisy),
            "pseudo_positive_count": selected,
            "aggregation_mass": int(reliable_mass),
        }
    )
    return model, diagnostics


local_train_mrfl = local_train_mrfl_nnpu
