#!/usr/bin/env python3
"""Standalone FedAvg + nnPU baseline for binary image classification.

This file is intentionally independent of the rest of the repository.  It
contains data preparation, PU observation construction, client partitioning,
the model, nnPU optimization, FedAvg aggregation, evaluation, checkpointing,
and a small self-test.

FedAvg follows McMahan et al. (AISTATS 2017): selected clients start every
round from the same global model, run local SGD, and the server averages the
returned models in proportion to each selected client's number of unique PU
training observations.

The training label is an observation label: +1 means labeled positive and -1
means unlabeled.  Hidden binary truth is retained only to construct controlled
partitions and to report protocol diagnostics; it is never passed to nnPU.
The test set uses binary truth for evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# CUDA >= 10.2 requires this setting before the first cuBLAS operation when
# torch deterministic algorithms are enabled.  setdefault preserves an
# explicit cluster/user choice if one is already present.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class ClientPUData:
    client_id: int
    positive_indices: np.ndarray
    unlabeled_indices: np.ndarray
    sample_count: int
    unlabeled_prior: float
    labeled_positive_count: int
    hidden_unlabeled_positive_count: int


CONTROLLED_PRIORS: Dict[str, Tuple[float, float, float]] = {
    "mild": (0.30, 0.50, 0.70),
    "moderate": (0.10, 0.50, 0.90),
    "extreme": (0.01, 0.50, 0.99),
}

CONTROLLED_QUANTITIES: Dict[str, Tuple[int, int, int]] = {
    "balanced": (1, 1, 1),
    "moderate": (6, 3, 1),
    "extreme": (8, 1, 1),
}


@dataclass
class Metrics:
    accuracy: float
    balanced_accuracy: float
    auc: float
    f1: float
    recall: float
    specificity: float
    loss: float


def parse_int_list(value: str) -> Tuple[int, ...]:
    values = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif torch.cuda.is_available():
        # Match the formal FedEPPU throughput-oriented CUDA protocol.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def dataset_targets(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("dataset must expose a targets attribute")
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return np.asarray(targets, dtype=np.int64)


def load_dataset(name: str, root: str) -> Tuple[Dataset, Dataset, int]:
    """Load a public benchmark without using any repository-specific code."""
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise RuntimeError("torchvision is required for benchmark datasets") from exc

    name = name.lower()
    if name == "cifar10":
        train = datasets.CIFAR10(root=root, train=True, download=True,
                                 transform=None)
        test = datasets.CIFAR10(root=root, train=False, download=True,
                                transform=None)
        return train, test, 3
    if name == "fashion_mnist":
        train = datasets.FashionMNIST(root=root, train=True, download=True,
                                      transform=None)
        test = datasets.FashionMNIST(root=root, train=False, download=True,
                                     transform=None)
        return train, test, 1
    raise ValueError("dataset must be one of: cifar10, fashion_mnist")


def materialize_image_tensor(dataset: Dataset, dataset_name: str,
                             device: torch.device) -> torch.Tensor:
    """Convert the complete benchmark split once instead of per sampled image."""
    data = getattr(dataset, "data", None)
    if data is None:
        raise ValueError("benchmark dataset must expose a data array")
    x = torch.as_tensor(np.asarray(data))
    if dataset_name == "cifar10":
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"unexpected CIFAR-10 shape: {tuple(x.shape)}")
        x = x.permute(0, 3, 1, 2)
    elif dataset_name == "fashion_mnist":
        if x.ndim != 3:
            raise ValueError(f"unexpected Fashion-MNIST shape: {tuple(x.shape)}")
        x = x.unsqueeze(1)
    else:
        raise ValueError(f"unsupported tensor materialization dataset: {dataset_name}")
    return x.to(device=device, dtype=torch.float32).div_(255.0).contiguous()


def normalize_images(x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    if dataset_name == "cifar10":
        mean = x.new_tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        std = x.new_tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
    elif dataset_name == "fashion_mnist":
        mean = x.new_tensor((0.2860,)).view(1, 1, 1, 1)
        std = x.new_tensor((0.3530,)).view(1, 1, 1, 1)
    else:
        raise ValueError(f"unsupported normalization dataset: {dataset_name}")
    return (x - mean) / std


def augment_and_normalize_batch(
    x: torch.Tensor,
    dataset_name: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Vectorized per-image crop/flip on the resident GPU tensor."""
    n, channels, height, width = x.shape
    if dataset_name == "cifar10":
        flip = torch.rand(n, device=x.device, generator=generator) < 0.5
        out = x.clone()
        if flip.any():
            out[flip] = torch.flip(out[flip], dims=(3,))
        pad = 4
        padded = F.pad(out, (pad, pad, pad, pad), mode="reflect")
    elif dataset_name == "fashion_mnist":
        pad = 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
    else:
        return normalize_images(x, dataset_name)
    padded_width = padded.shape[3]
    offset_y = torch.randint(
        0, 2 * pad + 1, (n, 1, 1, 1), device=x.device, generator=generator
    )
    offset_x = torch.randint(
        0, 2 * pad + 1, (n, 1, 1, 1), device=x.device, generator=generator
    )
    row_idx = offset_y + torch.arange(height, device=x.device)[None, None, :, None]
    row_idx = row_idx.expand(n, channels, height, padded_width)
    cropped = padded.gather(2, row_idx).contiguous()
    col_idx = offset_x + torch.arange(width, device=x.device)[None, None, None, :]
    col_idx = col_idx.expand(n, channels, height, width)
    cropped = cropped.gather(3, col_idx).contiguous()
    return normalize_images(cropped, dataset_name)


def to_binary_labels(multiclass: np.ndarray,
                     positive_classes: Sequence[int]) -> np.ndarray:
    positive = np.asarray(tuple(positive_classes), dtype=np.int64)
    return np.isin(multiclass, positive).astype(np.int64)


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_client_size: int,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> List[np.ndarray]:
    """Partition every observation exactly once using class-wise Dirichlet draws."""
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    if len(labels) < num_clients * min_client_size:
        raise ValueError("dataset is too small for requested min_client_size")

    classes = np.unique(labels)
    for _ in range(max_attempts):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.flatnonzero(labels == cls).copy()
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            for client_id, part in enumerate(np.split(cls_idx, cuts)):
                buckets[client_id].extend(part.tolist())
        arrays = [np.asarray(sorted(x), dtype=np.int64) for x in buckets]
        if min(map(len, arrays)) >= min_client_size:
            joined = np.concatenate(arrays)
            if len(joined) != len(labels) or len(np.unique(joined)) != len(labels):
                raise AssertionError("partition is not a disjoint cover")
            return arrays
    raise RuntimeError(
        "could not obtain a Dirichlet partition satisfying min_client_size; "
        "increase alpha or lower min_client_size"
    )


def construct_client_pu_data(
    partitions: Sequence[np.ndarray],
    binary_truth: np.ndarray,
    labeled_positive_fraction: float,
    min_labeled_positives: int,
    rng: np.random.Generator,
) -> List[ClientPUData]:
    """Create disjoint labeled-P and U pools within each client.

    The supplied nnPU prior is the exact positive fraction of that client's U
    pool.  It is computed at construction time for a controlled experiment and
    is not estimated from test labels during training.
    """
    if not (0.0 < labeled_positive_fraction < 1.0):
        raise ValueError("labeled_positive_fraction must lie strictly in (0, 1)")
    clients: List[ClientPUData] = []
    for client_id, client_idx in enumerate(partitions):
        client_idx = np.asarray(client_idx, dtype=np.int64)
        positives = client_idx[binary_truth[client_idx] == 1].copy()
        rng.shuffle(positives)
        requested = max(min_labeled_positives,
                        int(round(len(positives) * labeled_positive_fraction)))
        # At least one hidden positive is retained in U when possible.
        upper = max(1, len(positives) - 1)
        n_labeled = min(requested, upper)
        if len(positives) < 2 or n_labeled < 1:
            raise RuntimeError(
                f"client {client_id} has {len(positives)} positives; cannot build "
                "disjoint P and U pools. Increase alpha or min_client_size."
            )
        labeled_p = np.sort(positives[:n_labeled])
        is_labeled = np.zeros(len(binary_truth), dtype=bool)
        is_labeled[labeled_p] = True
        unlabeled = np.sort(client_idx[~is_labeled[client_idx]])
        hidden_u_pos = int(binary_truth[unlabeled].sum())
        prior = hidden_u_pos / len(unlabeled)
        if not (0.0 < prior < 1.0):
            raise RuntimeError(
                f"client {client_id} U prior is {prior:.6f}; nnPU needs both "
                "latent classes in U for this benchmark protocol"
            )
        clients.append(ClientPUData(
            client_id=client_id,
            positive_indices=labeled_p,
            unlabeled_indices=unlabeled,
            sample_count=int(len(labeled_p) + len(unlabeled)),
            unlabeled_prior=float(prior),
            labeled_positive_count=int(len(labeled_p)),
            hidden_unlabeled_positive_count=hidden_u_pos,
        ))
    return clients


def _take_disjoint(
    pool: np.ndarray,
    cursor: int,
    count: int,
    description: str,
) -> Tuple[np.ndarray, int]:
    end = cursor + int(count)
    if count < 0 or end > len(pool):
        raise RuntimeError(
            f"insufficient {description}: requested through {end}, available {len(pool)}"
        )
    return np.sort(pool[cursor:end].copy()), end


def construct_controlled_prior_data(
    binary_truth: np.ndarray,
    level: str,
    rng: np.random.Generator,
    labeled_positive_per_client: int = 333,
    unlabeled_per_client: int = 15640,
) -> List[ClientPUData]:
    """Equal client sizes with only the U-pool latent-positive prior changed."""
    if level not in CONTROLLED_PRIORS:
        raise ValueError(f"unknown controlled prior level: {level}")
    positive_pool = np.flatnonzero(binary_truth == 1).copy()
    negative_pool = np.flatnonzero(binary_truth == 0).copy()
    rng.shuffle(positive_pool)
    rng.shuffle(negative_pool)
    p_cursor = n_cursor = 0
    clients: List[ClientPUData] = []
    for client_id, target_prior in enumerate(CONTROLLED_PRIORS[level]):
        labeled_p, p_cursor = _take_disjoint(
            positive_pool, p_cursor, labeled_positive_per_client,
            "positive observations",
        )
        hidden_p_count = int(round(unlabeled_per_client * target_prior))
        hidden_n_count = unlabeled_per_client - hidden_p_count
        hidden_p, p_cursor = _take_disjoint(
            positive_pool, p_cursor, hidden_p_count,
            "hidden positive observations",
        )
        hidden_n, n_cursor = _take_disjoint(
            negative_pool, n_cursor, hidden_n_count,
            "negative observations",
        )
        unlabeled = np.sort(np.concatenate([hidden_p, hidden_n]))
        realized_prior = hidden_p_count / unlabeled_per_client
        clients.append(ClientPUData(
            client_id=client_id,
            positive_indices=labeled_p,
            unlabeled_indices=unlabeled,
            sample_count=labeled_positive_per_client + unlabeled_per_client,
            unlabeled_prior=float(realized_prior),
            labeled_positive_count=labeled_positive_per_client,
            hidden_unlabeled_positive_count=hidden_p_count,
        ))
    return clients


def construct_controlled_quantity_data(
    binary_truth: np.ndarray,
    level: str,
    rng: np.random.Generator,
    total_observations: int = 48000,
    labeled_positive_fraction: float = 0.02,
    target_unlabeled_prior: float = 0.488491,
) -> List[ClientPUData]:
    """Fixed global budget and U prior with only client quantity ratios changed.

    The labeled-positive allocation is proportional to client size, so the PU
    observation composition is held fixed as closely as integer counts permit.
    """
    if level not in CONTROLLED_QUANTITIES:
        raise ValueError(f"unknown controlled quantity level: {level}")
    ratio = np.asarray(CONTROLLED_QUANTITIES[level], dtype=np.int64)
    if total_observations % int(ratio.sum()) != 0:
        raise ValueError("controlled_quantity_total must be divisible by ratio sum")
    client_totals = ratio * (total_observations // int(ratio.sum()))
    positive_pool = np.flatnonzero(binary_truth == 1).copy()
    negative_pool = np.flatnonzero(binary_truth == 0).copy()
    rng.shuffle(positive_pool)
    rng.shuffle(negative_pool)
    p_cursor = n_cursor = 0
    clients: List[ClientPUData] = []
    for client_id, sample_count_value in enumerate(client_totals.tolist()):
        labeled_count = max(1, int(round(sample_count_value * labeled_positive_fraction)))
        unlabeled_count = sample_count_value - labeled_count
        hidden_p_count = int(round(unlabeled_count * target_unlabeled_prior))
        hidden_n_count = unlabeled_count - hidden_p_count
        labeled_p, p_cursor = _take_disjoint(
            positive_pool, p_cursor, labeled_count, "positive observations"
        )
        hidden_p, p_cursor = _take_disjoint(
            positive_pool, p_cursor, hidden_p_count,
            "hidden positive observations",
        )
        hidden_n, n_cursor = _take_disjoint(
            negative_pool, n_cursor, hidden_n_count, "negative observations"
        )
        unlabeled = np.sort(np.concatenate([hidden_p, hidden_n]))
        clients.append(ClientPUData(
            client_id=client_id,
            positive_indices=labeled_p,
            unlabeled_indices=unlabeled,
            sample_count=sample_count_value,
            unlabeled_prior=float(hidden_p_count / unlabeled_count),
            labeled_positive_count=labeled_count,
            hidden_unlabeled_positive_count=hidden_p_count,
        ))
    return clients


def assert_disjoint_client_protocol(clients: Sequence[ClientPUData]) -> None:
    pools: List[np.ndarray] = []
    for client in clients:
        if np.intersect1d(client.positive_indices, client.unlabeled_indices).size:
            raise AssertionError(f"client {client.client_id} has overlapping P and U")
        pools.extend([client.positive_indices, client.unlabeled_indices])
    joined = np.concatenate(pools)
    if len(joined) != len(np.unique(joined)):
        raise AssertionError("controlled protocol reuses a training observation")


class BetterCifarCNN(nn.Module):
    """FedEPPU-matched classification backbone without a ranking head."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.10),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 1),
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def nnpu_loss(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    prior: float,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Non-negative PU risk using the standard branchwise update.

    Logistic loss is softplus(-y*f(x)).  For y=+1 this is softplus(-f),
    and for y=-1 it is softplus(f).
    """
    if positive_logits.numel() == 0 or unlabeled_logits.numel() == 0:
        raise ValueError("nnPU requires nonempty positive and unlabeled batches")
    pi = positive_logits.new_tensor(float(prior))
    positive_risk = pi * F.softplus(-positive_logits).mean()
    negative_risk = (
        F.softplus(unlabeled_logits).mean()
        - pi * F.softplus(positive_logits).mean()
    )
    corrected = bool((negative_risk.detach() < -float(beta)).item())
    if corrected:
        objective = -float(gamma) * negative_risk
    else:
        objective = positive_risk + negative_risk
    diagnostics = {
        "positive_risk": float(positive_risk.detach().cpu()),
        "negative_risk": float(negative_risk.detach().cpu()),
        "nnpu_corrected": float(corrected),
    }
    return objective, diagnostics


def draw_pu_batch(
    train_images: torch.Tensor,
    positive_indices: torch.Tensor,
    unlabeled_indices: torch.Tensor,
    batch_size: int,
    positive_batch_fraction: float,
    dataset_name: str,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    n_p = min(batch_size - 1, max(1, int(round(batch_size * positive_batch_fraction))))
    n_u = batch_size - n_p
    p_idx = positive_indices[torch.randint(
        positive_indices.numel(), (n_p,), device=train_images.device,
        generator=generator,
    )]
    u_idx = unlabeled_indices[torch.randint(
        unlabeled_indices.numel(), (n_u,), device=train_images.device,
        generator=generator,
    )]
    combined = torch.cat((train_images[p_idx], train_images[u_idx]), dim=0)
    combined = augment_and_normalize_batch(combined, dataset_name, generator)
    p_images = combined[:n_p]
    u_images = combined[n_p:]
    return p_images, u_images


def local_client_update(
    global_model: nn.Module,
    train_images: torch.Tensor,
    client: ClientPUData,
    args: argparse.Namespace,
    device: torch.device,
    round_index: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    model = copy.deepcopy(global_model).to(device)
    model.train()
    progress = (round_index - 1) / max(args.rounds, 1)
    round_lr = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=round_lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(client.sample_count / args.batch_size))
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    generator = torch.Generator(device=device)
    generator.manual_seed(
        args.seed * 1_000_003 + round_index * 10_007 + client.client_id * 101
    )
    positive_indices = torch.as_tensor(
        client.positive_indices, dtype=torch.long, device=device
    )
    unlabeled_indices = torch.as_tensor(
        client.unlabeled_indices, dtype=torch.long, device=device
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_sum = pos_sum = neg_sum = correction_sum = grad_sum = 0.0
    steps = 0
    for _ in range(args.local_epochs):
        for _ in range(steps_per_epoch):
            p_images, u_images = draw_pu_batch(
                train_images, positive_indices, unlabeled_indices,
                args.batch_size, args.positive_batch_fraction,
                args.dataset, generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp_enabled,
            ):
                p_logits = model(p_images)
                u_logits = model(u_images)
            loss, diag = nnpu_loss(
                p_logits.float(), u_logits.float(), client.unlabeled_prior,
                beta=args.nnpu_beta, gamma=args.nnpu_gamma,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite client loss at round={round_index}, "
                    f"client={client.client_id}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.sqrt(sum(
                (p.grad.detach().float() ** 2).sum() for p in model.parameters()
                if p.grad is not None
            ))
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu())
            pos_sum += diag["positive_risk"]
            neg_sum += diag["negative_risk"]
            correction_sum += diag["nnpu_corrected"]
            grad_sum += float(torch.as_tensor(grad_norm).detach().cpu())
            steps += 1
    device_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return device_state, {
        "client_id": client.client_id,
        "sample_count": client.sample_count,
        "unlabeled_prior": client.unlabeled_prior,
        "steps": steps,
        "mean_loss": loss_sum / steps,
        "mean_positive_risk": pos_sum / steps,
        "mean_negative_risk": neg_sum / steps,
        "correction_fraction": correction_sum / steps,
        "mean_preclip_gradient_norm": grad_sum / steps,
        "learning_rate": round_lr,
    }


def fedavg_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
    sample_counts: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """Sample-count weighted FedAvg over selected clients."""
    if not states or len(states) != len(sample_counts):
        raise ValueError("states and sample_counts must have equal nonzero length")
    counts = np.asarray(sample_counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("all selected client sample counts must be positive")
    weights = counts / counts.sum()
    keys = tuple(states[0].keys())
    if any(tuple(s.keys()) != keys for s in states[1:]):
        raise ValueError("client state dictionaries do not share identical keys")
    result: Dict[str, torch.Tensor] = {}
    for key in keys:
        first = states[0][key]
        if first.is_floating_point():
            acc = torch.zeros_like(first)
            for weight, state in zip(weights, states):
                acc.add_(state[key], alpha=float(weight))
            result[key] = acc
        else:
            # Integer buffers (for example BatchNorm counters) are copied from
            # the largest selected client; averaging integer counters is invalid.
            result[key] = states[int(np.argmax(counts))][key].clone()
    return result


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC via average ranks, including tied scores, without sklearn."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    normalized_images: torch.Tensor,
    binary_truth: np.ndarray,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.0,
) -> Metrics:
    model.eval()
    scores: List[np.ndarray] = []
    losses: List[float] = []
    truth_tensor = torch.as_tensor(
        binary_truth, device=device, dtype=torch.float32
    )
    for offset in range(0, len(normalized_images), batch_size):
        images = normalized_images[offset:offset + batch_size]
        logits = model(images).float()
        n = len(images)
        truth = truth_tensor[offset:offset+n]
        losses.append(float(F.binary_cross_entropy_with_logits(
            logits, truth, reduction="sum").cpu()))
        scores.append(logits.cpu().numpy())
    score = np.concatenate(scores)
    pred = (score >= threshold).astype(np.int64)
    y = np.asarray(binary_truth, dtype=np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    return Metrics(
        accuracy=(tp + tn) / len(y),
        balanced_accuracy=0.5 * (recall + specificity),
        auc=binary_auc(y, score),
        f1=f1,
        recall=recall,
        specificity=specificity,
        loss=sum(losses) / len(y),
    )


def select_clients(num_clients: int, fraction: float,
                   rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < fraction <= 1.0):
        raise ValueError("client_fraction must lie in (0, 1]")
    count = max(1, int(math.ceil(num_clients * fraction)))
    return np.sort(rng.choice(num_clients, size=count, replace=False))


def atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_protocol_manifest(
    clients: Sequence[ClientPUData],
    path: Path,
) -> Tuple[str, List[str]]:
    """Persist exact indices so other methods can reuse the matched partition."""
    arrays: Dict[str, np.ndarray] = {}
    client_hashes: List[str] = []
    global_hasher = hashlib.sha256()
    for client in clients:
        p = np.asarray(client.positive_indices, dtype=np.int64)
        u = np.asarray(client.unlabeled_indices, dtype=np.int64)
        arrays[f"client{client.client_id}_positive"] = p
        arrays[f"client{client.client_id}_unlabeled"] = u
        hasher = hashlib.sha256()
        hasher.update(p.tobytes())
        hasher.update(u.tobytes())
        digest = hasher.hexdigest()
        client_hashes.append(digest)
        global_hasher.update(bytes.fromhex(digest))
    np.savez_compressed(path, **arrays)
    return global_hasher.hexdigest(), client_hashes


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed, args.deterministic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, test_dataset, in_channels = load_dataset(args.dataset, args.data_root)
    train_truth = to_binary_labels(dataset_targets(train_dataset), args.positive_classes)
    test_truth = to_binary_labels(dataset_targets(test_dataset), args.positive_classes)
    rng = np.random.default_rng(args.partition_seed)
    if args.protocol == "dirichlet":
        partitions = dirichlet_partition(
            train_truth, args.num_clients, args.dirichlet_alpha,
            args.min_client_size, rng,
        )
        clients = construct_client_pu_data(
            partitions, train_truth, args.labeled_positive_fraction,
            args.min_labeled_positives, rng,
        )
    elif args.protocol == "controlled_prior":
        if args.dataset != "cifar10" or args.num_clients != 3:
            raise ValueError("controlled_prior requires CIFAR-10 and exactly 3 clients")
        clients = construct_controlled_prior_data(
            train_truth, args.prior_level, rng,
            labeled_positive_per_client=args.controlled_labeled_positive_per_client,
            unlabeled_per_client=args.controlled_unlabeled_per_client,
        )
    elif args.protocol == "controlled_quantity":
        if args.dataset != "cifar10" or args.num_clients != 3:
            raise ValueError("controlled_quantity requires CIFAR-10 and exactly 3 clients")
        clients = construct_controlled_quantity_data(
            train_truth, args.quantity_level, rng,
            total_observations=args.controlled_quantity_total,
            labeled_positive_fraction=args.controlled_quantity_labeled_fraction,
            target_unlabeled_prior=args.controlled_quantity_prior,
        )
    else:
        raise ValueError(f"unsupported protocol: {args.protocol}")
    assert_disjoint_client_protocol(clients)

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    train_images = materialize_image_tensor(train_dataset, args.dataset, device)
    test_images = normalize_images(
        materialize_image_tensor(test_dataset, args.dataset, device), args.dataset
    )
    model = BetterCifarCNN(in_channels).to(device)
    history: List[Dict[str, object]] = []
    run_name = args.run_name or (
        f"fedavg_nnpu_{args.dataset}_{args.protocol}_seed{args.seed}_partition{args.partition_seed}"
    )
    history_path = output_dir / f"{run_name}.json"
    model_path = output_dir / f"{run_name}.pt"
    manifest_path = output_dir / f"{run_name}_protocol_indices.npz"
    data_fingerprint, client_fingerprints = save_protocol_manifest(
        clients, manifest_path
    )
    client_protocol = [{
        "client_id": c.client_id,
        "sample_count": c.sample_count,
        "labeled_positive_count": c.labeled_positive_count,
        "unlabeled_count": int(len(c.unlabeled_indices)),
        "hidden_unlabeled_positive_count": c.hidden_unlabeled_positive_count,
        "unlabeled_prior": c.unlabeled_prior,
        "data_fingerprint": client_fingerprints[c.client_id],
    } for c in clients]
    print("algorithm=FedAvg+nnPU standalone; backbone=FedEPPU-matched classification-only")
    print("aggregation=selected-client sample-count weighted FedAvg")
    print("inference=final-round raw model; threshold=0")
    print(f"data_pipeline=resident_{device.type}_tensor+vectorized_batch_augmentation; "
          f"amp={bool(args.amp and device.type == 'cuda')}")
    print(f"device={device}; dataset={args.dataset}; protocol={args.protocol}; "
          f"clients={args.num_clients}; "
          f"rounds={args.rounds}; seed={args.seed}; partition_seed={args.partition_seed}")
    print("client_protocol=" + json.dumps(client_protocol, separators=(",", ":")))
    print(f"data_fingerprint={data_fingerprint}")
    print(f"protocol_manifest={manifest_path}")

    selection_rng = np.random.default_rng(args.seed + 7919)
    started = time.perf_counter()
    for round_index in range(1, args.rounds + 1):
        selected = select_clients(args.num_clients, args.client_fraction, selection_rng)
        states: List[Dict[str, torch.Tensor]] = []
        counts: List[int] = []
        local_diags: List[Dict[str, float]] = []
        round_started = time.perf_counter()
        for client_id in selected.tolist():
            state, diag = local_client_update(
                model, train_images, clients[client_id], args, device, round_index
            )
            states.append(state)
            counts.append(clients[client_id].sample_count)
            local_diags.append(diag)
        new_state = fedavg_state_dicts(states, counts)
        model.load_state_dict(new_state, strict=True)
        model.to(device)
        metrics = evaluate(
            model, test_images, test_truth, args.eval_batch_size,
            device, threshold=0.0,
        )
        weights = (np.asarray(counts, dtype=np.float64) / sum(counts)).tolist()
        record: Dict[str, object] = {
            "round": round_index,
            "selected_clients": selected.tolist(),
            "aggregation_weights": weights,
            "local": local_diags,
            "metrics": asdict(metrics),
            "round_seconds": time.perf_counter() - round_started,
        }
        history.append(record)
        print(
            f"round={round_index:03d}/{args.rounds} "
            f"selected={selected.tolist()} "
            f"acc={metrics.accuracy*100:.2f} "
            f"bacc={metrics.balanced_accuracy*100:.2f} "
            f"auc={metrics.auc*100:.2f} f1={metrics.f1*100:.2f} "
            f"loss={metrics.loss:.5f} seconds={record['round_seconds']:.2f}",
            flush=True,
        )
        if round_index == 1 or round_index % args.checkpoint_interval == 0:
            atomic_json_dump({
                "status": "running",
                "arguments": vars(args),
                "client_protocol": client_protocol,
                "data_fingerprint": data_fingerprint,
                "protocol_manifest": str(manifest_path),
                "history": history,
            }, history_path)

    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arguments": vars(args),
        "client_protocol": client_protocol,
        "data_fingerprint": data_fingerprint,
        "protocol_manifest": str(manifest_path),
        "final_metrics": history[-1]["metrics"],
    }, model_path)
    atomic_json_dump({
        "status": "complete",
        "arguments": vars(args),
        "client_protocol": client_protocol,
        "data_fingerprint": data_fingerprint,
        "protocol_manifest": str(manifest_path),
        "history": history,
        "final_metrics": history[-1]["metrics"],
        "elapsed_seconds": time.perf_counter() - started,
        "model_path": str(model_path),
    }, history_path)
    print("training_complete=true")
    print("final_metrics=" + json.dumps(history[-1]["metrics"], separators=(",", ":")))
    print(f"history_path={history_path}")
    print(f"model_path={model_path}")


def run_self_test() -> None:
    # nnPU finite scalar and backward check.
    p = torch.tensor([1.0, 0.5, -0.2], requires_grad=True)
    u = torch.tensor([-1.0, 0.2, 0.7, -0.4], requires_grad=True)
    loss, diag = nnpu_loss(p, u, prior=0.4)
    assert torch.isfinite(loss)
    loss.backward()
    assert p.grad is not None and u.grad is not None
    assert set(diag) == {"positive_risk", "negative_risk", "nnpu_corrected"}

    # Exact sample-count weighting and preservation of integer buffers.
    states = [
        {"weight": torch.tensor([1.0, 3.0]), "counter": torch.tensor(2)},
        {"weight": torch.tensor([5.0, 7.0]), "counter": torch.tensor(9)},
    ]
    result = fedavg_state_dicts(states, [1, 3])
    assert torch.allclose(result["weight"], torch.tensor([4.0, 6.0]))
    assert result["counter"].item() == 9

    # Partition is a disjoint cover and respects minimum size.
    labels = np.array([0, 1] * 100, dtype=np.int64)
    parts = dirichlet_partition(labels, 4, 1.0, 10,
                                np.random.default_rng(123))
    joined = np.concatenate(parts)
    assert len(joined) == len(labels)
    assert len(np.unique(joined)) == len(labels)
    assert min(map(len, parts)) >= 10

    # AUC sanity checks.
    assert abs(binary_auc(np.array([0, 1]), np.array([0.0, 1.0])) - 1.0) < 1e-12
    assert abs(binary_auc(np.array([0, 1]), np.array([1.0, 0.0])) - 0.0) < 1e-12
    assert abs(binary_auc(np.array([0, 1]), np.array([0.5, 0.5])) - 0.5) < 1e-12

    # Controlled protocols use disjoint observations and realize their design.
    synthetic_truth = np.array([1] * 25000 + [0] * 25000, dtype=np.int64)
    prior_clients = construct_controlled_prior_data(
        synthetic_truth, "extreme", np.random.default_rng(7)
    )
    assert_disjoint_client_protocol(prior_clients)
    assert [c.sample_count for c in prior_clients] == [15973] * 3
    assert np.allclose([c.unlabeled_prior for c in prior_clients],
                       CONTROLLED_PRIORS["extreme"], atol=1 / 15640)
    quantity_clients = construct_controlled_quantity_data(
        synthetic_truth, "extreme", np.random.default_rng(8)
    )
    assert_disjoint_client_protocol(quantity_clients)
    assert [c.sample_count for c in quantity_clients] == [38400, 4800, 4800]
    assert max(abs(c.unlabeled_prior - 0.488491) for c in quantity_clients) < 2e-4
    print("self_test=ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone sample-count FedAvg with standard nnPU local training"
    )
    parser.add_argument("--dataset", choices=("cifar10", "fashion_mnist"),
                        default="cifar10")
    parser.add_argument(
        "--protocol",
        choices=("dirichlet", "controlled_prior", "controlled_quantity"),
        default="dirichlet",
    )
    parser.add_argument("--prior_level", choices=tuple(CONTROLLED_PRIORS),
                        default="mild")
    parser.add_argument("--quantity_level", choices=tuple(CONTROLLED_QUANTITIES),
                        default="balanced")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", default="./fedavg_nnpu_outputs")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--positive_classes", type=parse_int_list,
                        default=(0, 1, 2, 3, 4),
                        help="comma-separated original classes mapped to positive")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--client_fraction", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--min_client_size", type=int, default=128)
    parser.add_argument("--labeled_positive_fraction", type=float, default=0.1)
    parser.add_argument("--min_labeled_positives", type=int, default=16)
    parser.add_argument("--controlled_labeled_positive_per_client", type=int,
                        default=333)
    parser.add_argument("--controlled_unlabeled_per_client", type=int,
                        default=15640)
    parser.add_argument("--controlled_quantity_total", type=int, default=48000)
    parser.add_argument("--controlled_quantity_labeled_fraction", type=float,
                        default=0.02)
    parser.add_argument("--controlled_quantity_prior", type=float,
                        default=0.488491)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=6)
    parser.add_argument("--max_steps_per_epoch", type=int, default=15,
                        help="0 disables the cap")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--positive_batch_fraction", type=float, default=0.25)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=0.0015)
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="deprecated compatibility option; AdamW does not use momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="FedEPPU-matched pre-update gradient-norm cap")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="write the growing JSON history every N rounds")
    parser.add_argument("--nnpu_beta", type=float, default=0.0)
    parser.add_argument("--nnpu_gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition_seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true",
                        help="use CUDA autocast for network forward/backward; nnPU risk stays FP32")
    parser.add_argument("--device", default="",
                        help="empty selects CUDA when available, otherwise CPU")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_clients < 1 or args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("num_clients, rounds, and local_epochs must be positive")
    if not (0.0 < args.positive_batch_fraction < 1.0):
        raise ValueError("positive_batch_fraction must lie strictly in (0, 1)")
    if args.batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("invalid batch size")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.momentum < 0:
        raise ValueError("invalid optimizer setting")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if args.controlled_labeled_positive_per_client < 1:
        raise ValueError("controlled labeled-positive count must be positive")
    if args.controlled_unlabeled_per_client < 2:
        raise ValueError("controlled unlabeled count must be at least 2")
    if args.controlled_quantity_total < 3:
        raise ValueError("controlled quantity total is too small")
    if not (0.0 < args.controlled_quantity_labeled_fraction < 1.0):
        raise ValueError("controlled quantity labeled fraction must lie in (0, 1)")
    if not (0.0 < args.controlled_quantity_prior < 1.0):
        raise ValueError("controlled quantity prior must lie in (0, 1)")


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    validate_args(args)
    run_training(args)


if __name__ == "__main__":
    main()
