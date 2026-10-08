"""Main7 engine with cached ranking-balance updates every five batches.

This file preserves the v2 calculations and execution order. It only groups
the implementation into the five modules used by the paper: unified prior
correction, dual-head PU objective, deviation-aware local optimization,
bounded reliability aggregation, and EMA/selection/quantile inference.
"""

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from pu_loss7 import (
    prior_consistency_loss,
    pu_loss,
    pu_pairwise_ranking_loss,
)
_PU_LOSS_MODULE = "pu_loss7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# 训练超参数（论文/实验会调的量）
# ===========================================================================

@dataclass(frozen=True)
class Config:
    # optimizer
    lr_base: float = 0.0015
    weight_decay: float = 5e-4
    grad_clip: float = 1.0

    # local PU training
    base_epochs: int = 6
    batch_size: int = 256
    max_steps_per_epoch: int = 128
    # Empirical-Bayes prior correction
    n_shrink: float = 200.0

    # model shape; these change the network capacity, not ordinary tuning knobs
    cnn_channels: Tuple[int, int, int] = (64, 128, 256)
    cnn_hidden: int = 128


CFG = Config()


# ===========================================================================
# 内部调度常量（不作为对外超参数，但集中在一处，避免散落各函数体内）
# ===========================================================================

@dataclass(frozen=True)
class InternalSchedule:
    # One fixed evaluation-model EMA; no round-dependent momentum formula.
    model_ema_momentum: float = 0.95

SCHEDULE = InternalSchedule()


@dataclass(frozen=True)
class PriorState:
    """Outputs of the unified prior-correction module."""

    raw: np.ndarray
    counts: np.ndarray
    center: float
    effective: np.ndarray
    extremes: np.ndarray
    target: float


def _annealing_scale(round_idx: int, total_rounds: int) -> float:
    """Standard cosine scale determined only by round progress."""
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive.")
    progress = round_idx / total_rounds
    return float(0.5 * (1.0 + np.cos(np.pi * progress)))


def _prior_annealing_scale(round_idx: int, total_rounds: int) -> float:
    """Full cosine used by prior consistency in the original implementation."""
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive.")
    progress = round_idx / max(total_rounds - 1, 1)
    return float(0.5 * (1.0 + np.cos(np.pi * progress)))


def _round_lr(base_lr: float, round_idx: int, total_rounds: int) -> float:
    """Parameter-free cosine-annealed learning rate."""
    return float(base_lr * _annealing_scale(round_idx, total_rounds))


# ===========================================================================
# 数值工具
# ===========================================================================

def _flat(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _sigmoid_np(logits) -> np.ndarray:
    """数值安全 sigmoid（分段实现，无需 clip）。"""
    z = _flat(logits).astype(np.float64)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _to_float(v) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(np.asarray(v).reshape(-1)[0]) if isinstance(v, np.ndarray) else float(v)


def _update_model_ema(ema_model: nn.Module, current_model: nn.Module, mom: float) -> None:
    ema_sd = ema_model.state_dict()
    cur_sd = current_model.state_dict()
    for name, cur_v in cur_sd.items():
        cur_v = cur_v.to(device)
        if torch.is_floating_point(ema_sd[name]):
            ema_sd[name].mul_(mom).add_(cur_v, alpha=1.0 - mom)
        else:
            ema_sd[name].copy_(cur_v)
    ema_model.load_state_dict(ema_sd)


# ===========================================================================
# 模型（dropout 硬编码，aug 参数硬编码）
# ===========================================================================

class BetterCifarCNN(nn.Module):
    """VGG-style CNN for 32×32 RGB inputs."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        c1, c2, c3 = CFG.cnn_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, c1, 3, padding=1, bias=False),
            nn.GroupNorm(8, c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.GroupNorm(8, c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.10),

            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.GroupNorm(8, c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.GroupNorm(8, c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),

            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.GroupNorm(8, c3), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1, bias=False),
            nn.GroupNorm(8, c3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, CFG.cnn_hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(CFG.cnn_hidden, 1),
        )
        self.ranking_head = nn.Linear(CFG.cnn_hidden, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.classifier[0](z)
        z = self.classifier[1](z)
        return self.classifier[2](z)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier[4](self.classifier[3](features))

    def forward_dual(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        return (
            self.classify_features(features),
            self.ranking_head(features),
            features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


def _shared_gradient_norm(
    loss: torch.Tensor, shared_features: torch.Tensor
) -> torch.Tensor:
    """Gradient norm used to balance classification and ranking automatically."""
    gradient = torch.autograd.grad(
        loss,
        shared_features,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    return gradient.detach().float().norm(p=2)


# ===========================================================================
# 数据工具
# ===========================================================================

def _sanitize(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tuple(x.shape)}")
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def batch_aug_cifar(x: torch.Tensor) -> torch.Tensor:
    """Spatial augmentation for CIFAR-10 and single-channel Chest X-rays.

    Vectorized random crop: a two-step gather (first H then W) replaces the
    former per-sample Python loop, which consumed ~70% of per-step wall time.
    Output is element-for-element identical to the loop version.
    """
    x = _sanitize(x)
    _, channels, h, w = x.shape
    is_cifar = channels == 3 and h == 32 and w == 32
    is_chest = channels == 1 and h >= 64 and w >= 64
    if not (is_cifar or is_chest):
        return x
    n = x.shape[0]
    out = x.clone()

    flip = torch.rand(n, device=x.device) < 0.5
    if flip.any():
        out[flip] = torch.flip(out[flip], dims=[3])

    pad = 4 if is_cifar else max(4, int(round(min(h, w) * 0.08)))
    padded = F.pad(out, (pad,) * 4, mode="reflect")
    ph, pw = padded.shape[2], padded.shape[3]
    oh = torch.randint(0, ph - h + 1, (n, 1, 1, 1), device=x.device)
    ow = torch.randint(0, pw - w + 1, (n, 1, 1, 1), device=x.device)

    # Row offsets for every (sample, channel, output row) position.
    row_idx = oh + torch.arange(h, device=x.device)[None, None, :, None]
    row_idx = row_idx.expand(n, channels, h, pw)
    cropped = padded.gather(2, row_idx).contiguous()  # [n, C, h, pw]

    # Column offsets for every (sample, channel, output row, output col).
    col_idx = ow + torch.arange(w, device=x.device)[None, None, None, :]
    col_idx = col_idx.expand(n, channels, h, w)
    return cropped.gather(3, col_idx).contiguous()


def parse_train_data(train: Sequence[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ts = zip(*train)
    x = _sanitize(torch.from_numpy(np.stack(xs).astype(np.float32)).to(device))
    t = torch.from_numpy(np.array(ts, dtype=np.int32)).to(device)
    return x, t


# ===========================================================================
# Prior 校正 — Empirical Bayes 收缩（唯一 prior 裁剪入口）
# ===========================================================================

def compute_global_prior(
    raw_priors: Union[np.ndarray, Sequence],
    prior_counts: Optional[Union[np.ndarray, Sequence]] = None,
) -> float:
    """任务全局先验中心；有样本数时按各客户端未标记样本数加权。"""
    pis = np.asarray([_to_float(p) for p in raw_priors], dtype=np.float64).reshape(-1)
    if pis.size == 0:
        return 0.5
    if prior_counts is None:
        center = float(np.mean(pis))
    else:
        counts = np.asarray(prior_counts, dtype=np.float64).reshape(-1)
        if counts.size != pis.size:
            raise ValueError("prior_counts and raw_priors must have the same length.")
        valid = np.isfinite(pis) & np.isfinite(counts) & (counts > 0)
        center = (float(np.average(pis[valid], weights=counts[valid]))
                  if valid.any() else 0.5)
    if not np.isfinite(center):
        return 0.5
    eps = 1e-12
    if center <= eps:
        return eps
    if center >= 1.0 - eps:
        return 1.0 - eps
    return center


def compute_prior_extreme(prior: Union[float, int, np.floating, torch.Tensor],
                          global_prior: Union[float, int, np.floating, torch.Tensor]) -> float:
    """
    以任务全局先验 global_prior 为中心度量客户端先验极端程度：
        |prior - global_prior| / max(global_prior, 1-global_prior)
    当 global_prior=0.5 时退化为 |prior-0.5|/0.5。
    """
    p = _to_float(prior)
    c = _to_float(global_prior)
    eps = 1e-12
    if not np.isfinite(c):
        c = 0.5
    if c <= eps:
        c = eps
    elif c >= 1.0 - eps:
        c = 1.0 - eps
    denom = max(c, 1.0 - c, eps)
    return float(abs(p - c) / denom)


def build_effective_priors(
    raw_priors: np.ndarray,
    sample_counts: np.ndarray,
    center_prior: float,
    n_shrink: float = CFG.n_shrink,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将每个客户端 prior 向任务全局先验中心收缩：
        λ_i  = n_shrink / (n_i + n_shrink)
        eff_i = (1−λ_i)*pi_i + λ_i*center_prior
    EB 后验均值本身随样本量自适应；这里只做机器精度范围保护，不再用
    固定 0.05/0.95 边界系统性改写大样本客户端的可靠先验。
    extreme 由 compute_prior_extreme(eff_i, center_prior) 给出。

    center_prior 由调用方（compute_global_prior(raw_priors)）算好后传入，
    避免同一个量在两处重复计算导致悄悄不一致。
    """
    pis = raw_priors.astype(np.float64)
    ns  = np.maximum(sample_counts.astype(np.float64), 1.0)
    lam = n_shrink / (ns + n_shrink)
    eps = np.finfo(np.float64).eps
    eff = np.clip(
        (1.0 - lam) * pis + lam * center_prior,
        eps,
        1.0 - eps,
    )
    extremes = np.array(
        [compute_prior_extreme(p, center_prior) for p in eff],
        dtype=np.float64,
    )
    return eff, extremes


# ===========================================================================
# 推理
# ===========================================================================

def _infer_logits(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    parts: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            parts.append(
                model(x[i: i + 512].to(device)).detach().cpu().numpy().reshape(-1)
            )
    return np.concatenate(parts)


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    target_prior: Optional[float] = None,
    fixed_threshold: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    """Return (acc, recall, auc, f1, threshold)."""
    return evaluate_detailed(
        model, x, t, target_prior=target_prior, fixed_threshold=fixed_threshold
    )[:5]


def evaluate_detailed(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    target_prior: Optional[float] = None,
    fixed_threshold: Optional[float] = None,
) -> Tuple[float, float, float, float, float, float, float, int, int]:
    """Return core metrics plus specificity, BAcc, false positives and negatives."""
    logits = _infer_logits(model, x)
    probs  = _sigmoid_np(logits)
    t_cpu  = _flat(t)

    if fixed_threshold is not None:
        thr = float(fixed_threshold)
    else:
        if target_prior is None:
            raise ValueError("target_prior must be provided when fixed_threshold is None.")
        thr = _quantile_threshold(logits, float(target_prior))

    preds  = np.where(logits >= thr, 1, -1)
    acc = accuracy_score(t_cpu, preds)
    rec = recall_score(t_cpu, preds, pos_label=1, zero_division=0)
    f1  = f1_score(t_cpu, preds, pos_label=1, zero_division=0)
    negative = t_cpu == -1
    positive = t_cpu == 1
    fp = int(np.sum(negative & (preds == 1)))
    tn = int(np.sum(negative & (preds == -1)))
    fn = int(np.sum(positive & (preds == -1)))
    specificity = float(tn / max(tn + fp, 1))
    bacc = float(balanced_accuracy_score(t_cpu, preds))
    try:
        auc = roc_auc_score(t_cpu, probs)
    except Exception:
        auc = 0.5
    return acc, rec, auc, f1, thr, specificity, bacc, fp, fn


# ===========================================================================
# EB-prior quantile decision rule
# ===========================================================================

def _quantile_threshold(logits: np.ndarray, target_prior: float) -> float:
    logits = _flat(logits)
    k = int(round(len(logits) * target_prior))
    k = 1 if k < 1 else (len(logits) - 1 if k > len(logits) - 1 else k)
    return float(np.sort(logits)[::-1][k - 1])


# ===========================================================================
# 客户端权重（方案二：接入先验可靠性信号）
# ===========================================================================

def client_weights(
    sample_counts: np.ndarray,
    prior_extremes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample-aware + reliability-aware aggregation without a hard client cap.

    - 基础权重使用标准 FedAvg 样本量 n_k；
    - 若提供先验偏离 d_k，可靠性固定定义为 exp(-d_k)；
    - 归一化后直接用于聚合，不再强制单客户端权重不超过 1/2。
    """
    sw = np.maximum(np.asarray(sample_counts, dtype=np.float64), 1.0)

    if prior_extremes is not None:
        d = np.maximum(np.asarray(prior_extremes, dtype=np.float64), 0.0)
        reliability = np.exp(-d)
        sw = sw * reliability

    sw /= sw.sum() + 1e-12

    return sw


# ===========================================================================
# Five paper-facing modules (numerically equivalent to v2)
# ===========================================================================

class UnifiedPriorCorrection:
    """Module 1: construct one consistent EB-prior state for all consumers."""

    @staticmethod
    def build(
        raw_priors: np.ndarray,
        prior_counts: np.ndarray,
        n_shrink: float,
    ) -> PriorState:
        center = compute_global_prior(raw_priors, prior_counts)
        effective, extremes = build_effective_priors(
            raw_priors,
            prior_counts,
            center_prior=center,
            n_shrink=n_shrink,
        )
        target = float(np.clip(
            center,
            np.finfo(np.float64).eps,
            1.0 - np.finfo(np.float64).eps,
        ))
        return PriorState(
            raw=raw_priors,
            counts=prior_counts,
            center=center,
            effective=effective,
            extremes=extremes,
            target=target,
        )


class ClassificationRankingPUObjective:
    """Module 2: the unchanged classification-ranking dual-head objective."""

    @staticmethod
    def compute(
        logits: torch.Tensor,
        ranking_logits: Optional[torch.Tensor],
        shared_features: Optional[torch.Tensor],
        targets: torch.Tensor,
        prior: float,
        lambda_prior_k: float,
        cached_ranking_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_fp32 = logits.float()
        classification_loss, _ = pu_loss(logits_fp32, targets, prior)
        classification_loss = classification_loss + lambda_prior_k * prior_consistency_loss(
            logits_fp32, targets, prior
        )

        if ranking_logits is not None and shared_features is not None:
            ranking_loss = pu_pairwise_ranking_loss(
                ranking_logits.float(), targets, prior
            )
            if cached_ranking_scale is None:
                classification_norm = _shared_gradient_norm(
                    classification_loss, shared_features
                )
                ranking_norm = _shared_gradient_norm(ranking_loss, shared_features)
                machine_eps = torch.finfo(ranking_norm.dtype).eps
                ranking_scale = torch.where(
                    ranking_norm > machine_eps,
                    classification_norm / ranking_norm.clamp_min(machine_eps),
                    torch.zeros_like(ranking_norm),
                ).detach()
            else:
                ranking_scale = cached_ranking_scale.to(
                    device=ranking_loss.device, dtype=ranking_loss.dtype
                ).detach()
            loss = classification_loss + ranking_scale * ranking_loss
        else:
            ranking_scale = logits_fp32.new_zeros(())
            loss = classification_loss
        return loss, ranking_scale


class DeviationAwareLocalOptimization:
    """Module 3: derive the unchanged client-specific local controls."""

    @staticmethod
    def controls(
        extreme: float,
        lr: float,
        lambda_prior: float,
        prior_round_scale: float,
    ) -> Tuple[float, float, float]:
        prior_scale = float(np.exp(-(extreme ** 2)))
        actual_lr = lr * prior_scale
        lambda_prior_k = float(
            lambda_prior * prior_round_scale * np.exp(-(extreme ** 2))
        )
        return prior_scale, actual_lr, lambda_prior_k


class ReliabilityAggregation:
    """Module 4: compute uncapped reliability weights and aggregate models."""

    @staticmethod
    def weights(sample_counts: np.ndarray, extremes: np.ndarray) -> np.ndarray:
        return client_weights(sample_counts, prior_extremes=extremes)

    @staticmethod
    def aggregate(
        global_model: nn.Module,
        client_models: Sequence[nn.Module],
        weights: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        global_state = global_model.state_dict()
        new_state = {
            key: (torch.zeros_like(value) if torch.is_floating_point(value)
                  else torch.zeros_like(value, dtype=torch.float64))
            for key, value in global_state.items()
        }
        for i, weight in enumerate(weights):
            weight_tensor = torch.tensor(float(weight), device=device)
            for key, value in client_models[i].state_dict().items():
                if torch.is_floating_point(value):
                    new_state[key] += weight_tensor * value.to(device)
                else:
                    new_state[key] += weight_tensor.double() * value.to(
                        device=device, dtype=torch.float64
                    )
        for key, reference in global_state.items():
            if not torch.is_floating_point(reference):
                new_state[key] = new_state[key].round().to(dtype=reference.dtype)
        return new_state


class EMASelectionQuantileInference:
    """Module 5: fixed EMA, proxy-AUC selection, and prior quantile inference."""

    @staticmethod
    def update_ema(
        eval_model: nn.Module,
        global_model: nn.Module,
        round_idx: int,
        momentum: float,
    ) -> None:
        if round_idx == 0:
            eval_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        else:
            _update_model_ema(eval_model, global_model, mom=momentum)

    @staticmethod
    def proxy_auc(logits: np.ndarray, labels: torch.Tensor) -> float:
        proxy_labels = _flat(labels)
        if np.unique(proxy_labels).size < 2:
            return -np.inf
        return float(roc_auc_score(
            (proxy_labels > 0).astype(np.int64), logits
        ))

    @staticmethod
    def threshold(logits: np.ndarray, target_prior: float) -> float:
        return _quantile_threshold(logits, target_prior)

    @staticmethod
    def calibrated_threshold(logits: np.ndarray, labels: torch.Tensor) -> float:
        """Select the final threshold by proxy balanced accuracy.

        This is called exactly once after restoring the proxy-AUC-selected
        checkpoint.  Proxy labels never enter local training or aggregation.
        Ties are resolved deterministically by choosing the threshold closest
        to zero, followed by the smaller threshold.
        """
        threshold, _, _ = EMASelectionQuantileInference.calibrated_threshold_with_score(
            logits, labels, objective="bacc"
        )
        return threshold

    @staticmethod
    def calibrated_threshold_with_score(
        logits: np.ndarray,
        labels: torch.Tensor,
        min_specificity: float = 0.95,
        objective: str = "bacc",
    ) -> Tuple[float, float, float]:
        """Choose the best threshold subject to proxy specificity.

        objective="bacc" maximizes balanced accuracy; objective="acc"
        maximizes raw (unbalanced) accuracy on the calibration proxy set.
        """
        if not 0.0 <= min_specificity <= 1.0:
            raise ValueError("min_specificity must be in [0, 1].")
        if objective not in ("bacc", "acc"):
            raise ValueError("objective must be 'bacc' or 'acc'.")
        scores = np.asarray(logits, dtype=np.float64).reshape(-1)
        targets = (_flat(labels) > 0).astype(np.int64)
        if scores.size != targets.size:
            raise ValueError("Proxy logits and labels must have the same length.")
        if scores.size == 0 or np.unique(targets).size < 2:
            raise ValueError(
                "Final proxy Balanced-Accuracy calibration requires both classes."
            )

        unique_scores = np.unique(scores)
        if unique_scores.size == 1:
            # A threshold just above the constant score predicts every proxy
            # sample as negative and therefore satisfies any specificity
            # constraint. Recall is necessarily zero in this degenerate case.
            threshold = float(np.nextafter(unique_scores[0], np.inf))
            predictions = (scores >= threshold).astype(np.int64)
            specificity = float(np.mean(predictions[targets == 0] == 0))
            if objective == "acc":
                score = float(np.mean(predictions == targets))
            else:
                score = float(balanced_accuracy_score(targets, predictions))
            return threshold, score, specificity
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates = np.concatenate((
            [np.nextafter(unique_scores[0], -np.inf)],
            midpoints,
            [np.nextafter(unique_scores[-1], np.inf)],
        ))

        best_score = -np.inf
        best_recall = -np.inf
        best_specificity = -np.inf
        best_threshold = float(candidates[0])
        for threshold in candidates:
            predictions = (scores >= threshold).astype(np.int64)
            specificity = float(np.mean(predictions[targets == 0] == 0))
            if specificity + 1e-12 < min_specificity:
                continue
            if objective == "acc":
                score = float(np.mean(predictions == targets))
            else:
                score = float(balanced_accuracy_score(targets, predictions))
            recall = float(np.mean(predictions[targets == 1] == 1))
            rank = (score, recall, -abs(float(threshold)), -float(threshold))
            best_rank = (
                best_score, best_recall, -abs(best_threshold), -best_threshold
            )
            if rank > best_rank:
                best_score = score
                best_recall = recall
                best_specificity = specificity
                best_threshold = float(threshold)
        return best_threshold, float(best_score), float(best_specificity)


# ===========================================================================
# 本地训练
# ===========================================================================

def _ensure_mixed_batch(
    batch_idx: torch.Tensor,
    t: torch.Tensor,
    pos_idx: torch.Tensor,
    unl_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Ensure a PU batch contains at least one positive and one unlabeled sample.

    When the final mini-batch has size 1, replacing batch_idx[0] and
    batch_idx[-1] would target the same slot; in that case we explicitly
    build a two-sample batch [P, U].
    """
    if batch_idx.numel() == 0:
        return batch_idx

    has_pos = torch.any(t[batch_idx] == 1)
    has_unl = torch.any(t[batch_idx] == -1)
    if has_pos and has_unl:
        return batch_idx

    p_sel = pos_idx[torch.randint(0, pos_idx.numel(), (1,), device=device)]
    u_sel = unl_idx[torch.randint(0, unl_idx.numel(), (1,), device=device)]

    if batch_idx.numel() == 1:
        return torch.stack([p_sel[0], u_sel[0]])

    batch_idx = batch_idx.clone()
    if not has_pos:
        batch_idx[0] = p_sel[0]
    if not has_unl:
        batch_idx[-1] = u_sel[0]
    return batch_idx


def local_train(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    prior: float,
    extreme: float,
    lr: float,
    lambda_prior: float,
    prior_round_scale: float,
    batch_size: int,
    max_steps_per_epoch: int,
    local_epochs: int,
    use_amp: bool,
    ranking_balance_update_interval: int = 5,
    schedule: InternalSchedule = SCHEDULE,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    一轮本地训练。

    prior_scale = exp(-extreme²) 只控制客户端学习率。所有客户端使用
    相同的本地 epochs，删除可靠性信号对训练轮数的二次控制。

    返回 (model, diag)：diag 记录本轮实际使用的 epochs/actual_lr/
    prior_scale，供主循环写入 history 以便事后复现分析，
    而不是像旧版一样算完就扔。
    """
    n = len(x)
    # Reuse the same parameter-free reliability mapping used by the prior
    # consistency term.  This removes a second coefficient for the same d_k.
    prior_scale, actual_lr, lambda_prior_k = (
        DeviationAwareLocalOptimization.controls(
            extreme, lr, lambda_prior, prior_round_scale
        )
    )
    if ranking_balance_update_interval < 1:
        raise ValueError("ranking_balance_update_interval must be positive.")
    epochs     = local_epochs

    opt = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=CFG.weight_decay)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model.train()
    pos_idx = torch.where(t == 1)[0]
    unl_idx = torch.where(t == -1)[0]
    mixed   = pos_idx.numel() > 0 and unl_idx.numel() > 0
    # Reset for every client update so no value is shared across clients.
    cached_ranking_scale: Optional[torch.Tensor] = None
    local_batch_index = 0
    ranking_balance_updates = 0

    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        sample_budget = min(n, batch_size * max_steps_per_epoch)
        idx = idx[:sample_budget]

        for i in range(0, sample_budget, batch_size):
            batch_idx = idx[i: i + batch_size]

            if mixed:
                batch_idx = _ensure_mixed_batch(batch_idx, t, pos_idx, unl_idx)

            bx = batch_aug_cifar(x[batch_idx])
            bt = t[batch_idx]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                if hasattr(model, "forward_dual"):
                    logits, ranking_logits, shared_features = model.forward_dual(bx)
                else:
                    logits = model(bx)
                    ranking_logits = None
                    shared_features = None

            # Evaluate PU risks in FP32 even under AMP.  This retains AMP's
            # convolution speedup without exposing the subtraction in nnPU to
            # unnecessary half-precision cancellation.
            refresh_ranking_scale = (
                cached_ranking_scale is None
                or local_batch_index % ranking_balance_update_interval == 0
            )
            loss, ranking_scale = ClassificationRankingPUObjective.compute(
                logits,
                ranking_logits,
                shared_features,
                bt,
                prior,
                lambda_prior_k,
                cached_ranking_scale=(
                    None if refresh_ranking_scale else cached_ranking_scale
                ),
            )
            if ranking_logits is not None and shared_features is not None:
                if refresh_ranking_scale:
                    ranking_balance_updates += 1
                cached_ranking_scale = ranking_scale.detach()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)  # CLIP #3
            scaler.step(opt)
            scaler.update()
            local_batch_index += 1

    diag = {
        "epochs": float(epochs),
        "actual_lr": float(actual_lr),
        "prior_scale": prior_scale,
        "lambda_prior_k": lambda_prior_k,
        "ranking_enabled": float(hasattr(model, "forward_dual")),
        "ranking_balance_update_interval": float(ranking_balance_update_interval),
        "ranking_balance_updates": float(ranking_balance_updates),
        "local_batches": float(local_batch_index),
        "batch_size": float(batch_size),
        "max_steps_per_epoch": float(max_steps_per_epoch),
        "samples_per_epoch": float(min(n, batch_size * max_steps_per_epoch)),
        "amp_enabled": float(amp_enabled),
    }
    return model, diag


# ===========================================================================
# 主联邦训练循环
# ===========================================================================

def train_fedavg_proposed(
    client_data: List[Tuple[torch.Tensor, torch.Tensor]],
    x_test: torch.Tensor,
    t_test: torch.Tensor,
    proxy_x: torch.Tensor,
    priors: Sequence,
    anchor_model: nn.Module,
    proxy_t: Optional[torch.Tensor] = None,
    rounds: int = 100,
    n_clients: int = 3,
    lambda_prior: float = 0.10,
    proxy_min_specificity: float = 0.95,
    threshold_objective: str = "bacc",
    validation_source: str = "training_proxy",
    batch_size: int = CFG.batch_size,
    max_steps_per_epoch: int = CFG.max_steps_per_epoch,
    local_epochs: int = CFG.base_epochs,
    use_amp: bool = True,
    ranking_balance_update_interval: int = 5,
    n_shrink: float = CFG.n_shrink,
    round_callback: Optional[Callable] = None,
    schedule: InternalSchedule = SCHEDULE,
    annealing_fn: Optional[Callable[[int, int], float]] = None,
    prior_annealing_fn: Optional[Callable[[int, int], float]] = None,
    annealing_name: str = "standard_cosine",
) -> Tuple[nn.Module, Dict, float]:
    """Simplified federated PU training loop.

    Client priors are EB-corrected, local models optimize prior-consistent nnPU
    and the server performs reliability-aware weighted FedAvg.
    Proxy labels select the EMA checkpoint by AUC and calibrate one final
    threshold by the configured proxy objective. Per-round test metrics are reporting-only
    and never affect optimization, selection, or calibration.
    """
    raw_priors    = np.array([_to_float(p) for p in priors], np.float64)
    opt_annealing = annealing_fn or _annealing_scale
    pc_annealing = prior_annealing_fn or _prior_annealing_scale

    def checked_scale(
        fn: Callable[[int, int], float], round_idx: int, *, prior: bool = False
    ) -> float:
        # Follow the requested run length. A 300-round experiment trains for
        # all 300 rounds instead of freezing at the old fixed 200-round point.
        value = float(fn(round_idx, rounds))
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(
                "Annealing functions must return a finite value in [0, 1], "
                f"got {value} at round {round_idx}."
            )
        return value

    if lambda_prior < 0:
        raise ValueError("lambda_prior must be non-negative.")
    if not 0.0 <= proxy_min_specificity <= 1.0:
        raise ValueError("proxy_min_specificity must be in [0, 1].")
    if n_shrink < 0:
        raise ValueError("n_shrink must be non-negative.")
    if validation_source not in (
        "training_proxy", "training_proxy_fallback", "independent_deployment"
    ):
        raise ValueError(
            "Unsupported validation_source."
        )
    if (batch_size < 1 or max_steps_per_epoch < 1 or local_epochs < 1 or
            ranking_balance_update_interval < 1):
        raise ValueError(
            "batch_size, max_steps_per_epoch, local_epochs and "
            "ranking_balance_update_interval must be positive."
        )
    if len(client_data) != n_clients or len(raw_priors) != n_clients:
        raise ValueError(
            "n_clients must equal both len(client_data) and len(priors): "
            f"got {n_clients}, {len(client_data)}, {len(raw_priors)}."
        )
    if proxy_t is not None and len(proxy_t) != len(proxy_x):
        raise ValueError("proxy_t and proxy_x must have the same length.")

    # Split each validation class independently so checkpoint selection and
    # threshold calibration never reuse the same image. The original proxy is
    # already shuffled by dataset; alternating indices gives deterministic,
    # balanced and disjoint halves without consuming another RNG stream.
    selection_proxy_x = proxy_x
    selection_proxy_t = proxy_t
    calibration_proxy_x = proxy_x
    calibration_proxy_t = proxy_t
    if proxy_t is not None:
        positive_idx = torch.nonzero(proxy_t.reshape(-1) > 0, as_tuple=False).reshape(-1)
        negative_idx = torch.nonzero(proxy_t.reshape(-1) < 0, as_tuple=False).reshape(-1)
        if positive_idx.numel() < 2 or negative_idx.numel() < 2:
            raise ValueError(
                "Disjoint proxy selection/calibration requires at least two "
                "samples from each class."
            )
        selection_idx = torch.cat((positive_idx[::2], negative_idx[::2]))
        calibration_idx = torch.cat((positive_idx[1::2], negative_idx[1::2]))
        selection_proxy_x = proxy_x[selection_idx]
        selection_proxy_t = proxy_t[selection_idx]
        calibration_proxy_x = proxy_x[calibration_idx]
        calibration_proxy_t = proxy_t[calibration_idx]
        selection_pos = int((selection_proxy_t > 0).sum().item())
        selection_neg = int((selection_proxy_t < 0).sum().item())
        calibration_pos = int((calibration_proxy_t > 0).sum().item())
        calibration_neg = int((calibration_proxy_t < 0).sum().item())
        print(
            f"validation_source={validation_source}; "
            "validation_protocol=disjoint_stratified_halves; "
            f"selection_n={len(selection_idx)}; calibration_n={len(calibration_idx)}; "
            f"selection_pos={selection_pos}; selection_neg={selection_neg}; "
            f"calibration_pos={calibration_pos}; calibration_neg={calibration_neg}; "
            "selection_calibration_overlap=0"
        )
    sample_counts = np.array([len(d[0]) for d in client_data], np.float64)
    median_client_steps = max(
        1, int(np.ceil(float(np.median(sample_counts)) / batch_size))
    )
    # raw prior is measured only on U examples.  Weighting all clients equally
    # makes a tiny extreme client move the global threshold as much as a large
    # client; using U counts estimates the pooled unlabeled prior instead.
    prior_counts = np.array(
        [int((labels < 0).sum().item()) for _, labels in client_data],
        dtype=np.float64,
    )
    prior_state = UnifiedPriorCorrection.build(
        raw_priors, prior_counts, n_shrink
    )
    center_prior = prior_state.center
    effective_priors = prior_state.effective
    extremes = prior_state.extremes
    target_prior = prior_state.target
    global_model = copy.deepcopy(anchor_model).to(device)
    eval_model   = copy.deepcopy(global_model).to(device)

    history: Dict[str, List] = {
        key: [] for key in (
            "acc", "auc", "f1", "recall", "specificity", "bacc",
            "fp", "fn", "threshold", "proxy_auc",
        )
    }
    history["client_diag"] = []

    print("Prior correction (Empirical Bayes):")
    for i in range(n_clients):
        print(f"  [Client {i+1}] n={int(sample_counts[i])}, n_u={int(prior_counts[i])}, "
              f"raw_pi={raw_priors[i]:.4f}, eff_pi={effective_priors[i]:.4f}, "
              f"extreme={extremes[i]:.4f}")
    print(f"center_prior={center_prior:.4f} | target_prior={target_prior:.4f} "
          f"(pooled center prior, no test labels) | n_shrink={n_shrink:.0f}")
    strategy = (
        "disjointProxyAUCSelection+proxy"
        f"{threshold_objective.upper()}Calibration+fixedModelEMA+reliabilityFedAvg"
    )
    print(f"threshold_strategy={strategy}")
    print(f"proxy_min_specificity={proxy_min_specificity:.4f}")
    print(
        f"annealing_horizon=total_rounds({rounds}); "
        f"training_rounds={rounds}; frozen_reporting_rounds=0"
    )
    print(
        "prior_consistency="
        f"lambda_base:{lambda_prior:.4f}, "
        "lambda_k(r)=lambda_base*exp(-extreme^2)*cosine_scale(r)"
    )
    print(f"prior_cosine_decay={annealing_name}")
    print(
        "model_selection=proxyAUC_best_EMA; "
        + (
            "selection-half proxy AUC selects checkpoint; disjoint calibration-half labels set threshold"
            if proxy_t is not None
            else "proxy labels unavailable"
        )
    )
    print(
        f"validation_protocol={validation_source}_disjoint_stratified_halves; "
        f"threshold=per_round_and_final_proxy_{threshold_objective.upper()}; gradient_training=none; "
        "test_labels=per_round_reporting_only"
    )
    print("prior_source=dataset_provided_client_PU_prior; EB_correction=enabled")
    print(
        "dual_head=classification:nnPU, ranking:PU-corrected-pairwise; "
        "task_balance:cached_shared_feature_gradient_norm, manual_weight:none"
    )
    print(
        f"ranking_balance_update_interval={ranking_balance_update_interval}; "
        "first_local_batch=refresh; intervening_batches=reuse_detached; "
        "cache_scope=per_client"
    )
    print(
        "compute_budget="
        f"batch_size:{batch_size}, max_steps_per_epoch:{max_steps_per_epoch}, "
        f"base_epochs:{local_epochs}, AMP:{bool(use_amp and device.type == 'cuda')}, "
        f"annealing:{annealing_name}, "
        "step_schedule:fixed_min(user_cap,median_client_batches)"
    )
    cap_rounds = (0, rounds // 2, max(rounds - 1, 0))
    fixed_step_cap = min(max_steps_per_epoch, median_client_steps)
    cap_values = [fixed_step_cap for _ in cap_rounds]
    print(
        "local_step_cap_schedule="
        f"start:{cap_values[0]}, middle:{cap_values[1]}, end:{cap_values[2]}"
    )
    estimated_steps = []
    for count in sample_counts:
        steps_per_epoch = min(
            int(np.ceil(count / batch_size)),
            max_steps_per_epoch,
            median_client_steps,
        )
        estimated_steps.append(local_epochs * steps_per_epoch)
    print("initial_local_steps_per_round=" + ", ".join(
        f"client{i+1}:{steps}" for i, steps in enumerate(estimated_steps)
    ))
    print("local_epoch_strategy=uniform; reliability controls learning_rate only")
    print(
        "chest_compute_balance=shared_median_client_budget:"
        f"{median_client_steps}_batches_per_epoch_before_annealing"
    )
    prior_lambdas = lambda_prior * np.exp(-(extremes ** 2))
    print("prior_consistency_weights=" + ", ".join(
        f"client{i+1}:{value:.4f}"
        for i, value in enumerate(prior_lambdas)
    ))
    prior_scale_points = [
        checked_scale(pc_annealing, idx, prior=True) for idx in cap_rounds
    ]
    print(
        "prior_round_scale="
        f"start:{prior_scale_points[0]:.4f}, "
        f"middle:{prior_scale_points[1]:.4f}, "
        f"end:{prior_scale_points[2]:.4f}"
    )
    print(f"[Clip] prior EB + grad_norm≤{CFG.grad_clip}"
          " | output_norm=removed (0 triggers in baseline)"
          " | KD=off | proxy-logit EMA=removed")
    print("[Method] stage1=EBPriorCorrection; "
          "stage2=ClassificationRankingDualHeadPU; "
          "stage3=DeviationAwareUncappedAggregation")
    print("[Training] prior-consistent nnPU + nonnegative-corrected PU ranking + "
          "stop-gradient shared-feature gradient-norm balancing")
    print("[Inference] fixed model EMA + proxyAUC checkpoint selection + "
          f"proxy-{threshold_objective.upper()} threshold calibration; per-round test "
          "metrics are reporting-only")
    print(
        "ablation_parameters="
        f"n_shrink:{n_shrink:g}, lambda_prior:{lambda_prior:g}"
    )
    print("aggregation=reliabilityAwareFedAvg(sample_count,prior_deviation); "
          "cap=off; exponent_parameter=none")
    print(
        "fixed_implementation_settings="
        f"prior_bounds:machine_epsilon, model_ema:{schedule.model_ema_momentum:g}; "
        "not ablated"
    )

    aggregation_weights = ReliabilityAggregation.weights(
        sample_counts, extremes
    )
    print("aggregation_weights=" + ", ".join(
        f"client{i+1}:{w:.4f}" for i, w in enumerate(aggregation_weights)
    ))

    best_proxy_auc = -np.inf
    best_round = -1
    best_eval_state = None

    for r in range(rounds):
        annealing_scale = checked_scale(opt_annealing, r)
        round_lr = CFG.lr_base * annealing_scale
        # Anneal only the update magnitude. Keeping a fixed per-round compute
        # budget avoids prematurely stopping representation learning.
        current_max_steps = min(max_steps_per_epoch, median_client_steps)
        prior_round_scale = checked_scale(pc_annealing, r, prior=True)

        client_diags: List[Dict[str, float]] = []
        client_models = []
        for i in range(n_clients):
            m_i, diag_i = local_train(
                copy.deepcopy(global_model),
                client_data[i][0], client_data[i][1],
                prior=float(effective_priors[i]),
                extreme=float(extremes[i]),
                lr=round_lr,
                lambda_prior=lambda_prior,
                prior_round_scale=prior_round_scale,
                batch_size=batch_size,
                max_steps_per_epoch=current_max_steps,
                local_epochs=local_epochs,
                use_amp=use_amp,
                ranking_balance_update_interval=ranking_balance_update_interval,
                schedule=schedule,
            )
            client_models.append(m_i)
            client_diags.append(diag_i)

        # Reliability-aware weighted FedAvg without the former 0.5 cap.
        new_state = ReliabilityAggregation.aggregate(
            global_model, client_models, aggregation_weights
        )
        global_model.load_state_dict(new_state)

        EMASelectionQuantileInference.update_ema(
            eval_model,
            global_model,
            round_idx=r,
            momentum=schedule.model_ema_momentum,
        )

        # Proxy labels are used only for threshold-free checkpoint selection.
        selection_proxy_logits = _infer_logits(eval_model, selection_proxy_x)
        proxy_auc = -np.inf
        if selection_proxy_t is not None:
            proxy_auc = EMASelectionQuantileInference.proxy_auc(
                selection_proxy_logits, selection_proxy_t
            )
            if proxy_auc > best_proxy_auc:
                best_proxy_auc = proxy_auc
                best_round = r
                best_eval_state = copy.deepcopy(eval_model.state_dict())
        history["proxy_auc"].append(float(proxy_auc))
        # Use the same configured proxy-label calibration during reporting as in the
        # final evaluation.  The previous center-prior quantile could drift
        # across domains and make Accuracy collapse while AUC remained high.
        # This affects evaluation only; no test label or proxy gradient enters
        # local training or aggregation.
        if calibration_proxy_t is not None:
            calibration_proxy_logits = _infer_logits(
                eval_model, calibration_proxy_x
            )
            threshold, proxy_bacc, proxy_specificity = (
                EMASelectionQuantileInference.calibrated_threshold_with_score(
                    calibration_proxy_logits,
                    calibration_proxy_t,
                    proxy_min_specificity,
                    objective=threshold_objective,
                )
            )
        else:
            threshold = EMASelectionQuantileInference.threshold(
                selection_proxy_logits, target_prior
            )
            proxy_bacc = float("nan")
            proxy_specificity = float("nan")
        acc, rec, auc, f1, thr, spec, bacc, fp, fn = evaluate_detailed(
            eval_model, x_test, t_test,
            target_prior=target_prior, fixed_threshold=threshold,
        )
        for key, value in zip(
            ("acc", "auc", "f1", "recall", "specificity", "bacc", "fp", "fn", "threshold"),
            (acc, auc, f1, rec, spec, bacc, fp, fn, thr),
        ):
            history[key].append(float(value))
        history["client_diag"].append(client_diags)
        history.setdefault("proxy_bacc", []).append(float(proxy_bacc))
        history.setdefault("proxy_specificity", []).append(float(proxy_specificity))

        if round_callback is not None:
            round_callback(r, acc, rec, auc, f1, thr, spec, bacc, fp, fn)

    # Restore only after all communication rounds have completed.  Proxy
    # labels calibrate the decision threshold exactly once on the selected
    # checkpoint; they never affect gradients, aggregation, or later rounds.
    if best_eval_state is not None:
        eval_model.load_state_dict(best_eval_state)
    else:
        best_round = rounds - 1
        best_proxy_auc = history["proxy_auc"][-1] if history["proxy_auc"] else -np.inf

    final_proxy_logits = _infer_logits(eval_model, calibration_proxy_x)
    final_proxy_bacc = float("nan")
    if calibration_proxy_t is not None:
        final_threshold, final_proxy_bacc, final_proxy_specificity = (
            EMASelectionQuantileInference.calibrated_threshold_with_score(
                final_proxy_logits,
                calibration_proxy_t,
                proxy_min_specificity,
                objective=threshold_objective,
            )
        )
    else:
        final_threshold = EMASelectionQuantileInference.threshold(
            final_proxy_logits, target_prior
        )
        final_proxy_specificity = float("nan")
    # Final evaluation after checkpoint selection and threshold determination.
    final_acc, final_rec, final_auc, final_f1, final_thr, final_spec, final_bacc, final_fp, final_fn = evaluate_detailed(
        eval_model, x_test, t_test,
        target_prior=target_prior, fixed_threshold=final_threshold,
    )
    history["selected"] = {
        "round": int(best_round + 1),
        "proxy_auc": float(best_proxy_auc),
        "proxy_bacc": float(final_proxy_bacc),
        "proxy_specificity": float(final_proxy_specificity),
        "threshold_objective": threshold_objective,
        "acc": float(final_acc),
        "recall": float(final_rec),
        "auc": float(final_auc),
        "f1": float(final_f1),
        "specificity": float(final_spec),
        "bacc": float(final_bacc),
        "fp": int(final_fp),
        "fn": int(final_fn),
        "threshold": float(final_thr),
    }
    print(
        f"[Final threshold calibration] metric=proxy_{threshold_objective.upper()}_subject_to_specificity, "
        f"min_specificity={proxy_min_specificity:.4f}, "
        f"proxy_specificity={final_proxy_specificity:.4f}, "
        f"score={final_proxy_bacc:.4f}, threshold={final_thr:.4f}"
    )
    print(
        f"[Final selection] round={best_round + 1}, "
        f"proxy_AUC={best_proxy_auc:.4f}, threshold={final_thr:.4f}"
    )
    return eval_model, history, float(final_acc)
