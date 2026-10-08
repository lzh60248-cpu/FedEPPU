"""Main1: original dual-head method with stratified P/U minibatches.

Known client priors are used by PU risks, aggregation uses capped sample
counts, and raw periodic proxy AUC selects a tolerance-qualified checkpoint.
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

from pu_loss import pu_loss, pu_pairwise_ranking_loss
_PU_LOSS_MODULE = "pu_loss"

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
    max_steps_per_epoch: int = 15
    # Known client priors are used directly by default.
    n_shrink: float = 0.0

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

def client_weights(sample_counts: np.ndarray) -> np.ndarray:
    """
    Sample-aware aggregation with capped redistribution.

    - 基础权重使用标准 FedAvg 样本量 n_k；
    - 单客户端权重不超过 1/2，使其不超过其余客户端的权重总和；
    - 超出上限的质量按其余客户端相对权重重新分配。
    """
    sw = np.maximum(np.asarray(sample_counts, dtype=np.float64), 1.0)

    sw /= sw.sum() + 1e-12

    n_clients = sw.size
    if n_clients <= 1:
        return np.ones_like(sw, dtype=np.float64)

    # No single client may outweigh all remaining clients combined.
    cap = 0.5

    out = np.zeros_like(sw, dtype=np.float64)
    free = np.ones(n_clients, dtype=bool)
    remaining_mass = 1.0

    while free.any():
        free_sum = sw[free].sum()
        if free_sum <= 0:
            out[free] = remaining_mass / free.sum()
            break

        proposal = remaining_mass * sw[free] / free_sum
        over = proposal > cap
        free_idx = np.where(free)[0]
        if not over.any():
            out[free_idx] = proposal
            break

        capped_idx = free_idx[over]
        out[capped_idx] = cap
        remaining_mass -= cap * len(capped_idx)
        if remaining_mass < 0:
            remaining_mass = 0.0
        free[capped_idx] = False

    out /= out.sum() + 1e-12
    return out


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_fp32 = logits.float()
        classification_loss, _ = pu_loss(logits_fp32, targets, prior)

        if ranking_logits is not None and shared_features is not None:
            ranking_loss = pu_pairwise_ranking_loss(
                ranking_logits.float(), targets, prior
            )
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
            loss = classification_loss + ranking_scale * ranking_loss
        else:
            ranking_scale = logits_fp32.new_zeros(())
            loss = classification_loss
        return loss, ranking_scale


class CappedSampleAggregation:
    """Compute capped sample-count weights and aggregate client models."""

    @staticmethod
    def weights(sample_counts: np.ndarray) -> np.ndarray:
        return client_weights(sample_counts)

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
    """Fixed EMA, proxy-AUC selection, and proxy threshold calibration."""

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


def sample_pu_indices(pos_idx, unl_idx, batch_size):
    """Sample each pool uniformly with replacement; do not change the U prior."""
    if batch_size < 4 or pos_idx.numel() == 0 or unl_idx.numel() == 0:
        raise ValueError("Stratified PU requires batch_size >= 4 and nonempty P/U pools.")
    n_p = batch_size // 4
    p = pos_idx[torch.randint(pos_idx.numel(), (n_p,), device=pos_idx.device)]
    u = unl_idx[torch.randint(unl_idx.numel(), (batch_size - n_p,), device=unl_idx.device)]
    return torch.cat((p, u))


def local_train(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    prior: float,
    lr: float,
    batch_size: int,
    max_steps_per_epoch: int,
    local_epochs: int,
    use_amp: bool,
    schedule: InternalSchedule = SCHEDULE,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    一轮本地训练。

    所有客户端使用相同的轮次学习率和本地 epochs。真实先验只进入
    nnPU 与 PU 排序风险，不再额外约束平均预测概率。

    返回 (model, diag)，供主循环写入 history 以便事后复现分析。
    """
    n = len(x)
    actual_lr = float(lr)
    epochs     = local_epochs

    opt = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=CFG.weight_decay)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model.train()
    pos_idx = torch.where(t == 1)[0]
    unl_idx = torch.where(t == -1)[0]
    if batch_size < 4 or not pos_idx.numel() or not unl_idx.numel():
        raise ValueError("Stratified PU requires batch_size >= 4 and nonempty P/U pools.")
    steps = corrections = ranking_active = 0
    scale_sum = scale_max = loss_sum = 0.0

    for _ in range(epochs):
        sample_budget = min(n, batch_size * max_steps_per_epoch)

        for i in range(0, sample_budget, batch_size):
            batch_idx = sample_pu_indices(pos_idx, unl_idx, batch_size)

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
            loss, ranking_scale = ClassificationRankingPUObjective.compute(
                logits,
                ranking_logits,
                shared_features,
                bt,
                prior,
            )

            # Read-only diagnostics: no extra forward/backward or RNG consumption.
            with torch.no_grad():
                z = logits.float().reshape(-1)
                neg_risk = F.softplus(z[bt == -1]).mean() - prior * F.softplus(z[bt == 1]).mean()
                scale = float(ranking_scale.item())
                corrections += int(neg_risk.item() < 0)
                ranking_active += int(scale > 0)
                scale_sum += scale
                scale_max = max(scale_max, scale)
                loss_sum += float(loss.item())
                steps += 1
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)  # CLIP #3
            scaler.step(opt)
            scaler.update()

    diag = {
        "epochs": float(epochs),
        "actual_lr": float(actual_lr),
        "ranking_head_present": float(hasattr(model, "forward_dual")),
        "ranking_active_fraction": ranking_active / max(steps, 1),
        "ranking_scale_mean": scale_sum / max(steps, 1),
        "ranking_scale_max": scale_max,
        "nnpu_correction_fraction": corrections / max(steps, 1),
        "mean_loss": loss_sum / max(steps, 1),
        "steps": steps,
        "positive_per_batch": batch_size // 4,
        "unlabeled_per_batch": batch_size - batch_size // 4,
        "batch_size": float(batch_size),
        "max_steps_per_epoch": float(max_steps_per_epoch),
        "samples_per_epoch": float(int(np.ceil(min(n, batch_size * max_steps_per_epoch) / batch_size)) * batch_size),
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
    proxy_min_specificity: float = 0.0,
    threshold_objective: str = "acc",
    annealing_horizon: int = 200,
    selection_interval: int = 5,
    selection_auc_tolerance: float = 0.001,
    batch_size: int = CFG.batch_size,
    max_steps_per_epoch: int = CFG.max_steps_per_epoch,
    local_epochs: int = CFG.base_epochs,
    use_amp: bool = True,
    n_shrink: float = CFG.n_shrink,
    round_callback: Optional[Callable] = None,
    schedule: InternalSchedule = SCHEDULE,
    annealing_fn: Optional[Callable[[int, int], float]] = None,
    annealing_name: str = "standard_cosine",
) -> Tuple[nn.Module, Dict, float]:
    """Simplified federated PU training loop.

    Known client priors are used by nnPU/ranking risks. The server performs
    capped sample-count FedAvg. Periodic raw selection-proxy AUC chooses the
    latest EMA checkpoint within a fixed tolerance of the best score; the
    disjoint calibration proxy sets the threshold.
    """
    raw_priors    = np.array([_to_float(p) for p in priors], np.float64)
    opt_annealing = annealing_fn or _annealing_scale

    def checked_scale(fn: Callable[[int, int], float], round_idx: int) -> float:
        effective_round = min(round_idx, annealing_horizon)
        value = float(fn(effective_round, annealing_horizon))
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(
                "Annealing functions must return a finite value in [0, 1], "
                f"got {value} at round {round_idx}."
            )
        return value

    if not 0.0 <= proxy_min_specificity <= 1.0:
        raise ValueError("proxy_min_specificity must be in [0, 1].")
    if n_shrink < 0:
        raise ValueError("n_shrink must be non-negative.")
    if annealing_horizon < 1:
        raise ValueError("annealing_horizon must be positive.")
    if selection_interval < 1:
        raise ValueError("Selection interval must be positive.")
    if selection_auc_tolerance < 0:
        raise ValueError("Selection AUC tolerance must be non-negative.")
    if batch_size < 4 or max_steps_per_epoch < 1 or local_epochs < 1:
        raise ValueError("batch_size must be >= 4; step and epoch budgets must be positive.")
    if len(client_data) != n_clients or len(raw_priors) != n_clients:
        raise ValueError(
            "n_clients must equal both len(client_data) and len(priors): "
            f"got {n_clients}, {len(client_data)}, {len(raw_priors)}."
        )
    if proxy_t is not None and len(proxy_t) != len(proxy_x):
        raise ValueError("proxy_t and proxy_x must have the same length.")

    # Split each proxy class independently so checkpoint selection and
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
        print(
            "proxy_protocol=disjoint_stratified_halves; "
            f"selection_n={len(selection_idx)}; calibration_n={len(calibration_idx)}; "
            f"overlap=0"
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
            "proxy_score", "proxy_specificity",
        )
    }
    history["client_diag"] = []

    prior_mode = "known" if n_shrink == 0 else "eb_shrink"
    print(
        "Prior mode: known client priors (no shrinkage)"
        if prior_mode == "known"
        else "Prior mode: empirical-Bayes shrinkage"
    )
    for i in range(n_clients):
        print(f"  [Client {i+1}] n={int(sample_counts[i])}, n_u={int(prior_counts[i])}, "
              f"raw_pi={raw_priors[i]:.4f}, eff_pi={effective_priors[i]:.4f}, "
              f"extreme={extremes[i]:.4f}")
    print(f"center_prior={center_prior:.4f} | target_prior={target_prior:.4f} "
          f"(pooled center prior, no test labels) | n_shrink={n_shrink:.0f}")
    constraint_name = (
        "unconstrained" if proxy_min_specificity <= 0.0
        else f"minSpec{proxy_min_specificity:g}"
    )
    strategy = (
        "disjointPeriodicProxyAUCSelection+latestWithinTolerance+proxy"
        f"{threshold_objective.upper()}Calibration({constraint_name})+"
        "fixedModelEMA+cappedSampleFedAvg"
    )
    print(f"threshold_strategy={strategy}")
    print(f"proxy_min_specificity={proxy_min_specificity:.4f}")
    print(
        f"annealing_horizon={annealing_horizon}; "
        f"training_rounds={min(rounds, annealing_horizon)}; "
        f"frozen_reporting_rounds={max(rounds - annealing_horizon, 0)}"
    )
    print("prior_usage=nnPU+ranking_only; prior_consistency=off")
    print(f"sampling=stratified_with_replacement; P={batch_size // 4}; U={batch_size - batch_size // 4}; prior=original_U_prior")
    print(
        "model_selection=periodicRawProxyAUC_latestWithinTolerance_EMA; "
        + (
            "selection-half proxy AUC selects checkpoint; disjoint calibration-half labels set threshold"
            if proxy_t is not None
            else "proxy labels unavailable"
        )
    )
    print(
        "validation_protocol=disjoint_proxy_selection_and_calibration; "
        f"selection_interval={selection_interval}; auc_tolerance={selection_auc_tolerance:g}; "
        "threshold=per_round_reporting_and_final_calibration; "
        "test_labels=per_round_reporting_only"
    )
    print(
        "prior_source=dataset_ground_truth_client_PU_prior; "
        f"prior_mode={prior_mode}; EB_correction={'disabled' if prior_mode == 'known' else 'enabled'}"
    )
    print(
        "dual_head=classification:nnPU, ranking:PU-corrected-pairwise; prior_moment_loss=off; "
        "task_balance:shared_feature_gradient_norm, manual_weight:none"
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
    print("local_optimization=uniform_lr+uniform_epochs; prior_consistency=off")
    print(
        "compute_balance=shared_median_client_budget:"
        f"{median_client_steps}_batches_per_epoch_before_annealing"
    )
    print(f"[Clip] prior bounds + grad_norm≤{CFG.grad_clip}"
          " | output_norm=removed (0 triggers in baseline)"
          " | KD=off | proxy-logit EMA=removed")
    prior_stage = (
        "KnownClientPriorForPURisk"
        if prior_mode == "known"
        else "EBPriorForPURisk"
    )
    print(f"[Method] stage1={prior_stage}; "
          "stage2=ClassificationRankingDualHeadPU; "
          "stage3=CappedSampleAggregation")
    print("deviation_usage=none; aggregation_signal=sample_count_only")
    print("[Training] nnPU + nonnegative-corrected PU ranking + "
          "stop-gradient shared-feature gradient-norm balancing")
    print("[Inference] fixed model EMA + periodic raw proxyAUC tolerance selection + "
          f"proxy-{threshold_objective.upper()} threshold calibration; per-round test "
          "metrics are reporting-only")
    print(
        "method_parameters="
        f"n_shrink:{n_shrink:g}, selection_interval:{selection_interval}, "
        f"selection_auc_tolerance:{selection_auc_tolerance:g}, "
        f"proxy_min_specificity:{proxy_min_specificity:g}"
    )
    print("aggregation=cappedSampleFedAvg(sample_count); "
          "cap=0.5(single_client_not_greater_than_all_others)")
    print(
        "fixed_implementation_settings="
        f"prior_bounds:machine_epsilon, model_ema:{schedule.model_ema_momentum:g}; "
        "not ablated"
    )

    aggregation_weights = CappedSampleAggregation.weights(sample_counts)
    print("aggregation_weights=" + ", ".join(
        f"client{i+1}:{w:.4f}" for i, w in enumerate(aggregation_weights)
    ))

    best_proxy_auc = -np.inf
    selected_proxy_auc = -np.inf
    best_round = -1
    best_eval_state = None

    for r in range(rounds):
        annealing_scale = checked_scale(opt_annealing, r)
        round_lr = CFG.lr_base * annealing_scale
        # Anneal only the update magnitude. Keeping a fixed per-round compute
        # budget avoids prematurely stopping representation learning.
        current_max_steps = min(max_steps_per_epoch, median_client_steps)
        training_active = r < annealing_horizon
        client_diags: List[Dict[str, float]] = []
        if training_active:
            client_models = []
            for i in range(n_clients):
                m_i, diag_i = local_train(
                    copy.deepcopy(global_model),
                    client_data[i][0], client_data[i][1],
                    prior=float(effective_priors[i]),
                    lr=round_lr,
                    batch_size=batch_size,
                    max_steps_per_epoch=current_max_steps,
                    local_epochs=local_epochs,
                    use_amp=use_amp,
                    schedule=schedule,
                )
                client_models.append(m_i)
                client_diags.append(diag_i)

            # Capped sample-count FedAvg. Client prior deviation is not an
            # aggregation reliability signal because it can represent valid
            # Non-IID heterogeneity.
            new_state = CappedSampleAggregation.aggregate(
                global_model, client_models, aggregation_weights
            )
            global_model.load_state_dict(new_state)

            # Keep the first 200 rounds numerically aligned with Main9's
            # 200-round trajectory. Once frozen, neither model is updated.
            EMASelectionQuantileInference.update_ema(
                eval_model,
                global_model,
                round_idx=r,
                momentum=schedule.model_ema_momentum,
            )
        else:
            client_diags = [
                {"training_frozen": 1.0, "lr": 0.0}
                for _ in range(n_clients)
            ]

        # Compare EMA checkpoints periodically. The selected checkpoint is the
        # latest one whose own raw proxy AUC is within a fixed tolerance of the
        # best raw proxy AUC observed so far; every score therefore corresponds
        # directly to the stored model state.
        last_training_round = min(rounds, annealing_horizon) - 1
        evaluated = training_active and (
            ((r + 1) % selection_interval == 0) or (r == last_training_round)
        )
        proxy_auc = float("nan")
        if evaluated and selection_proxy_t is not None:
            selection_proxy_logits = _infer_logits(eval_model, selection_proxy_x)
            proxy_auc = EMASelectionQuantileInference.proxy_auc(
                selection_proxy_logits, selection_proxy_t
            )
            if proxy_auc > best_proxy_auc:
                best_proxy_auc = float(proxy_auc)
            if proxy_auc >= best_proxy_auc - selection_auc_tolerance:
                selected_proxy_auc = float(proxy_auc)
                best_round = r
                best_eval_state = copy.deepcopy(eval_model.state_dict())
        history["proxy_auc"].append(float(proxy_auc))

        # Preserve the original per-round metric format for monitoring and
        # plotting. These test metrics never update the model, checkpoint
        # choice, or final threshold. The disjoint calibration proxy supplies
        # the reporting threshold without using test labels.
        if calibration_proxy_t is not None:
            calibration_proxy_logits = _infer_logits(
                eval_model, calibration_proxy_x
            )
            reporting_threshold, proxy_score, proxy_specificity = (
                EMASelectionQuantileInference.calibrated_threshold_with_score(
                    calibration_proxy_logits,
                    calibration_proxy_t,
                    proxy_min_specificity,
                    objective=threshold_objective,
                )
            )
        else:
            reporting_logits = _infer_logits(eval_model, selection_proxy_x)
            reporting_threshold = EMASelectionQuantileInference.threshold(
                reporting_logits, target_prior
            )
            proxy_score = float("nan")
            proxy_specificity = float("nan")

        acc, rec, auc, f1, thr, spec, bacc, fp, fn = evaluate_detailed(
            eval_model,
            x_test,
            t_test,
            target_prior=target_prior,
            fixed_threshold=reporting_threshold,
        )
        for key, value in zip(
            ("acc", "auc", "f1", "recall", "specificity", "bacc", "fp", "fn", "threshold"),
            (acc, auc, f1, rec, spec, bacc, fp, fn, thr),
        ):
            history[key].append(float(value))
        history["proxy_score"].append(float(proxy_score))
        history["proxy_specificity"].append(float(proxy_specificity))
        history["client_diag"].append(client_diags)
        if training_active:
            print("[PU diag] " + "; ".join(
                f"client{i+1}: steps={d['steps']}, correction={d['nnpu_correction_fraction']:.3f}, "
                f"ranking_active={d['ranking_active_fraction']:.3f}, "
                f"scale_mean={d['ranking_scale_mean']:.5g}, scale_max={d['ranking_scale_max']:.5g}"
                for i, d in enumerate(client_diags)
            ))
        if evaluated:
            print(f"[Proxy] round={r+1}; selection_auc={proxy_auc:.6f}; calibration_score={proxy_score:.6f}")

        if round_callback is not None:
            round_callback(r, acc, rec, auc, f1, thr, spec, bacc, fp, fn)

    # Restore only after all communication rounds have completed. Per-round
    # calibration above is reporting-only; the selected checkpoint receives
    # a fresh final calibration here. Proxy and test labels never affect
    # gradients, aggregation, or checkpoint selection.
    if best_eval_state is not None:
        eval_model.load_state_dict(best_eval_state)
    else:
        # Fallback for runs with no scheduled proxy evaluation.
        best_round = rounds - 1
        selection_proxy_logits = _infer_logits(eval_model, selection_proxy_x)
        best_proxy_auc = EMASelectionQuantileInference.proxy_auc(
            selection_proxy_logits, selection_proxy_t
        )
        selected_proxy_auc = float(best_proxy_auc)

    final_proxy_logits = _infer_logits(eval_model, calibration_proxy_x)
    final_proxy_score = float("nan")
    if calibration_proxy_t is not None:
        final_threshold, final_proxy_score, final_proxy_specificity = (
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
        "proxy_auc": float(selected_proxy_auc),
        "proxy_auc_max": float(best_proxy_auc),
        "proxy_score": float(final_proxy_score),
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
    threshold_rule = (
        f"proxy_{threshold_objective.upper()}_unconstrained"
        if proxy_min_specificity <= 0.0
        else f"proxy_{threshold_objective.upper()}_subject_to_specificity"
    )
    print(
        f"[Final threshold calibration] metric={threshold_rule}, "
        f"min_specificity={proxy_min_specificity:.4f}, "
        f"proxy_specificity={final_proxy_specificity:.4f}, "
        f"score={final_proxy_score:.4f}, threshold={final_thr:.4f}"
    )
    print(
        f"[Final selection] round={best_round + 1}, "
        f"selected_proxy_AUC={selected_proxy_auc:.4f}, max_proxy_AUC={best_proxy_auc:.4f}, "
        f"tolerance={selection_auc_tolerance:g}, "
        f"threshold={final_thr:.4f}"
    )
    return eval_model, history, float(final_acc)
