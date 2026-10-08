"""Main2: Main1 without model EMA.

Known client priors are used by PU risks. Server aggregation uses only the
parameter-free complement-mass weights; negative support remains diagnostic.
Raw global models are evaluated for proxy-AUC checkpoint selection.
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
    # model shape; these change the network capacity, not ordinary tuning knobs
    cnn_channels: Tuple[int, int, int] = (64, 128, 256)
    cnn_hidden: int = 128


CFG = Config()

SELECTION_INTERVAL = 5
SELECTION_AUC_TOLERANCE = 0.001


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


# ===========================================================================
# 模型（dropout 硬编码，aug 参数硬编码）
# ===========================================================================

class BetterCifarCNN(nn.Module):
    """Shared dual-head CNN for grayscale and RGB image datasets."""

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
    Parameter-free complement-mass constrained FedAvg base weights.

    Each client's effective count is min(n_k, sum_{j != k} n_j).  Standard
    sample-count FedAvg is unchanged unless one client holds an absolute
    majority.  The resulting one-half bound is derived, not configured.
    """
    counts = np.maximum(np.asarray(sample_counts, dtype=np.float64), 1.0)
    if counts.size <= 1:
        return np.ones_like(counts, dtype=np.float64)
    total = float(counts.sum())
    effective_counts = np.minimum(counts, total - counts)
    return effective_counts / effective_counts.sum()


# ===========================================================================
# Core paper-facing components
# ===========================================================================

class KnownPriorState:
    """Construct one state from the known client priors without shrinkage."""

    @staticmethod
    def build(
        raw_priors: np.ndarray,
        prior_counts: np.ndarray,
    ) -> PriorState:
        center = compute_global_prior(raw_priors, prior_counts)
        eps = np.finfo(np.float64).eps
        effective = np.clip(
            raw_priors.astype(np.float64), eps, 1.0 - eps
        )
        extremes = np.array(
            [compute_prior_extreme(p, center) for p in effective],
            dtype=np.float64,
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


def linear_ranking_support(prior: float, unlabeled_count: int) -> float:
    """Expected-negative support q=min(1, (1-prior)*n_U)."""
    return float(np.clip((1.0 - float(prior)) * int(unlabeled_count), 0.0, 1.0))


class ClassificationRankingPUObjective:
    """Unweighted nnPU plus PU-ranking objective; support is server-only."""

    @staticmethod
    def compute(
        logits: torch.Tensor,
        ranking_logits: Optional[torch.Tensor],
        shared_features: Optional[torch.Tensor],
        targets: torch.Tensor,
        prior: float,
        diagnostics: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_fp32 = logits.float()
        classification_loss, _ = pu_loss(logits_fp32, targets, prior)

        if ranking_logits is not None and shared_features is not None:
            ranking_diagnostics: Dict[str, float] = {}
            ranking_loss = pu_pairwise_ranking_loss(
                ranking_logits.float(), targets, prior,
                diagnostics=ranking_diagnostics,
            )
            support_value = linear_ranking_support(
                prior, int((targets.reshape(-1) == -1).sum().item())
            )
            ranking_scale = logits_fp32.new_ones(()).detach()
            loss = classification_loss + ranking_scale * ranking_loss
            if diagnostics is not None:
                diagnostics.update(ranking_diagnostics)
                diagnostics["ranking_support"] = support_value
                diagnostics["effective_ranking_scale"] = float(
                    ranking_scale.item()
                )
                diagnostics["local_support_factor"] = 1.0
        else:
            ranking_scale = logits_fp32.new_zeros(())
            loss = classification_loss
            if diagnostics is not None:
                diagnostics.update({
                    "ranking_raw_risk": 0.0,
                    "ranking_abs_correction_active": 0.0,
                    "ranking_support": 0.0,
                    "effective_ranking_scale": 0.0,
                    "local_support_factor": 0.0,
                })
        return loss, ranking_scale


class ComplementMassAggregation:
    """Compute support-aware complement-mass weights and aggregate models."""

    @staticmethod
    def weights(sample_counts: np.ndarray) -> np.ndarray:
        return client_weights(sample_counts)

    @staticmethod
    def support_aware_weights(
        base_weights: np.ndarray,
        support_means: np.ndarray,
    ) -> np.ndarray:
        """Renormalize complement-mass base weights by negative support.

        The support values are round means of the same parameter-free
        expected-negative support used by the local objective.  If no client
        has usable support, retain the base weights so aggregation is defined.
        """
        base = np.asarray(base_weights, dtype=np.float64).reshape(-1)
        support = np.asarray(support_means, dtype=np.float64).reshape(-1)
        if base.size == 0 or base.shape != support.shape:
            raise ValueError("base_weights and support_means must have the same nonzero shape")
        if not np.all(np.isfinite(base)) or np.any(base < 0.0):
            raise ValueError("base_weights must be finite and nonnegative")
        support = np.clip(np.nan_to_num(support, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        adjusted = base * support
        adjusted_sum = float(adjusted.sum())
        if adjusted_sum <= np.finfo(np.float64).eps:
            base_sum = float(base.sum())
            if base_sum <= np.finfo(np.float64).eps:
                raise ValueError("base_weights must contain positive mass")
            return base / base_sum
        return adjusted / adjusted_sum

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


class ProxyCheckpointInference:
    """Raw-model proxy-AUC selection and proxy threshold calibration."""

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
    def calibrated_threshold_with_score(
        logits: np.ndarray,
        labels: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Choose the threshold with maximum unconstrained proxy accuracy."""
        scores = np.asarray(logits, dtype=np.float64).reshape(-1)
        targets = (_flat(labels) > 0).astype(np.int64)
        if scores.size != targets.size:
            raise ValueError("Proxy logits and labels must have the same length.")
        if scores.size == 0 or np.unique(targets).size < 2:
            raise ValueError(
                "Final proxy accuracy calibration requires both classes."
            )

        unique_scores = np.unique(scores)
        if unique_scores.size == 1:
            # A threshold just above the constant score predicts every proxy
            # sample as negative. Recall is necessarily zero in this
            # degenerate case.
            threshold = float(np.nextafter(unique_scores[0], np.inf))
            predictions = (scores >= threshold).astype(np.int64)
            specificity = float(np.mean(predictions[targets == 0] == 0))
            score = float(np.mean(predictions == targets))
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
            score = float(np.mean(predictions == targets))
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
    raw_rank_risk_sum = 0.0
    rank_abs_corrections = 0
    support_sum = 0.0
    support_min, support_max = float("inf"), 0.0

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
            batch_diagnostics: Dict[str, float] = {}
            loss, ranking_scale = ClassificationRankingPUObjective.compute(
                logits,
                ranking_logits,
                shared_features,
                bt,
                prior,
                diagnostics=batch_diagnostics,
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
                raw_rank_risk_sum += batch_diagnostics.get(
                    "ranking_raw_risk", 0.0
                )
                rank_abs_corrections += int(batch_diagnostics.get(
                    "ranking_abs_correction_active", 0.0
                ))
                support_value = batch_diagnostics.get("ranking_support", 0.0)
                support_sum += support_value
                support_min = min(support_min, support_value)
                support_max = max(support_max, support_value)
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
        "ranking_support_mean": support_sum / max(steps, 1),
        "ranking_support_min": 0.0 if steps == 0 else support_min,
        "ranking_support_max": support_max,
        "local_support_factor_mean": 1.0 if steps else 0.0,
        "ranking_raw_risk_mean": raw_rank_risk_sum / max(steps, 1),
        "ranking_abs_correction_fraction": rank_abs_corrections / max(steps, 1),
        "nnpu_correction_fraction": corrections / max(steps, 1),
        "mean_loss": loss_sum / max(steps, 1),
        "steps": steps,
        "positive_per_batch": batch_size // 4,
        "unlabeled_per_batch": batch_size - batch_size // 4,
        "batch_size": float(batch_size),
        "max_steps_per_epoch": float(max_steps_per_epoch),
        "actual_steps_per_epoch": steps / max(epochs, 1),
        "actual_steps_per_round": float(steps),
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
    batch_size: int = CFG.batch_size,
    max_steps_per_epoch: int = CFG.max_steps_per_epoch,
    local_epochs: int = CFG.base_epochs,
    use_amp: bool = True,
    round_callback: Optional[Callable] = None,
    annealing_fn: Optional[Callable[[int, int], float]] = None,
    annealing_name: str = "standard_cosine",
) -> Tuple[nn.Module, Dict, float]:
    """Simplified federated PU training loop.

    Known client priors are used by nnPU/ranking risks. The server performs
    complement-mass constrained FedAvg. Periodic raw proxy AUC chooses the
    latest raw-model checkpoint within a fixed tolerance of the best score; the
    disjoint calibration proxy sets the threshold.
    """
    raw_priors    = np.array([_to_float(p) for p in priors], np.float64)
    opt_annealing = annealing_fn or _annealing_scale

    def checked_scale(fn: Callable[[int, int], float], round_idx: int) -> float:
        value = float(fn(round_idx, rounds))
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(
                "Annealing functions must return a finite value in [0, 1], "
                f"got {value} at round {round_idx}."
            )
        return value

    if rounds < 1 or batch_size < 4 or max_steps_per_epoch < 1 or local_epochs < 1:
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
    prior_state = KnownPriorState.build(raw_priors, prior_counts)
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

    print("Prior mode: known client priors (no shrinkage)")
    for i in range(n_clients):
        print(f"  [Client {i+1}] n={int(sample_counts[i])}, n_u={int(prior_counts[i])}, "
              f"raw_pi={raw_priors[i]:.4f}, eff_pi={effective_priors[i]:.4f}, "
              f"extreme={extremes[i]:.4f}")
    print(f"center_prior={center_prior:.4f} | target_prior={target_prior:.4f} "
          "(pooled center prior, no test labels)")
    strategy = (
        "disjointPeriodicProxyAUCSelection+latestWithinTolerance+proxy"
        "ACCCalibration(unconstrained)+"
        "rawGlobalModel+complementMassAggregation"
    )
    print(f"threshold_strategy={strategy}")
    print(f"annealing_horizon=rounds={rounds}; frozen_reporting_rounds=0")
    print("prior_usage=nnPU+ranking_only; prior_consistency=off")
    print(f"sampling=stratified_with_replacement; P={batch_size // 4}; U={batch_size - batch_size // 4}; prior=original_U_prior")
    print(
        "model_selection=periodicRawProxyAUC_latestWithinTolerance_rawGlobalModel; "
        + (
            "selection-half proxy AUC selects checkpoint; disjoint calibration-half labels set threshold"
            if proxy_t is not None
            else "proxy labels unavailable"
        )
    )
    print(
        "validation_protocol=disjoint_proxy_selection_and_calibration; "
        f"selection_interval={SELECTION_INTERVAL}; "
        f"auc_tolerance={SELECTION_AUC_TOLERANCE:g}; "
        "threshold=per_round_reporting_and_final_calibration; "
        "test_labels=per_round_reporting_only"
    )
    print(
        "prior_source=dataset_ground_truth_client_PU_prior; "
        "prior_mode=known; EB_correction=absent"
    )
    print(
        "dual_head=classification:nnPU, ranking:PU-corrected-pairwise; "
        "local_objective=nnPU+ranking(equal_unit_weights); "
        "ranking_support=diagnostic_only"
    )
    print(
        "compute_budget="
        f"batch_size:{batch_size}, step_cap_per_local_epoch:{max_steps_per_epoch}, "
        f"local_epochs:{local_epochs}, maximum_optimizer_steps_per_round:"
        f"{local_epochs * max_steps_per_epoch}, AMP:{bool(use_amp and device.type == 'cuda')}, "
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
    print("[Method] stage1=KnownClientPriorForPURisk; "
          "stage2=ClassificationRankingDualHeadPU; "
          "stage3=ComplementMassAggregation")
    print("deviation_usage=none; aggregation_signal=complement_mass_only")
    print("[Training] nnPU + PU ranking with equal unit weights; "
          "gradient_balance=off; local_support_scaling=off")
    print("[Inference] raw global model + periodic raw proxyAUC tolerance selection + "
          "proxy-ACC threshold calibration; per-round test "
          "metrics are reporting-only")
    print(
        "fixed_selection_settings="
        f"selection_interval:{SELECTION_INTERVAL}, "
        f"selection_auc_tolerance:{SELECTION_AUC_TOLERANCE:g}"
    )
    print("aggregation=complementMassFedAvg; negative_support=diagnostic_only; "
          "effective_count=min(client_count,sum_other_counts); tunable_cap=none")
    print("fixed_implementation_settings=prior_bounds:machine_epsilon; model_ema=off")

    raw_aggregation_weights = sample_counts / sample_counts.sum()
    aggregation_weights = ComplementMassAggregation.weights(sample_counts)
    print("raw_sample_weights=" + ", ".join(
        f"client{i+1}:{w:.4f}" for i, w in enumerate(raw_aggregation_weights)
    ))
    print("complement_mass_base_weights=" + ", ".join(
        f"client{i+1}:{w:.4f}" for i, w in enumerate(aggregation_weights)
    ))
    print("complement_mass_constraint_active=" + str(bool(
        np.any(np.abs(raw_aggregation_weights - aggregation_weights) > 1e-12)
    )))

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
        client_diags: List[Dict[str, float]] = []
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
            )
            client_models.append(m_i)
            client_diags.append(diag_i)

        # Preserve negative-support diagnostics, but Main1 deliberately does
        # not use them in server aggregation.
        round_support = np.asarray([
            diag["ranking_support_mean"] for diag in client_diags
        ], dtype=np.float64)
        round_aggregation_weights = aggregation_weights.copy()
        for i, diag in enumerate(client_diags):
            diag["base_aggregation_weight"] = float(aggregation_weights[i])
            diag["support_aggregation_factor"] = float(round_support[i])
            diag["effective_aggregation_weight"] = float(round_aggregation_weights[i])
        new_state = ComplementMassAggregation.aggregate(
            global_model, client_models, round_aggregation_weights
        )
        global_model.load_state_dict(new_state)
        # Main2 evaluates the current raw global model directly. No EMA state
        # or EMA momentum remains in the training or inference pipeline.
        eval_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

        # Compare raw global-model checkpoints periodically. The selected one is the
        # latest one whose own raw proxy AUC is within a fixed tolerance of the
        # best raw proxy AUC observed so far; every score therefore corresponds
        # directly to the stored model state.
        evaluated = (
            ((r + 1) % SELECTION_INTERVAL == 0) or (r == rounds - 1)
        )
        proxy_auc = float("nan")
        if evaluated and selection_proxy_t is not None:
            selection_proxy_logits = _infer_logits(eval_model, selection_proxy_x)
            proxy_auc = ProxyCheckpointInference.proxy_auc(
                selection_proxy_logits, selection_proxy_t
            )
            if proxy_auc > best_proxy_auc:
                best_proxy_auc = float(proxy_auc)
            if proxy_auc >= best_proxy_auc - SELECTION_AUC_TOLERANCE:
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
                ProxyCheckpointInference.calibrated_threshold_with_score(
                    calibration_proxy_logits, calibration_proxy_t
                )
            )
        else:
            reporting_logits = _infer_logits(eval_model, selection_proxy_x)
            reporting_threshold = ProxyCheckpointInference.threshold(
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
        print("[PU diag] " + "; ".join(
                f"client{i+1}: steps={d['steps']}, correction={d['nnpu_correction_fraction']:.3f}, "
                f"ranking_active={d['ranking_active_fraction']:.3f}, "
                f"effective_scale_mean={d['ranking_scale_mean']:.5g}, "
                f"effective_scale_max={d['ranking_scale_max']:.5g}, "
                f"support_mean={d['ranking_support_mean']:.5g}, "
                f"support_range=[{d['ranking_support_min']:.5g},{d['ranking_support_max']:.5g}], "
                f"local_support_factor={d['local_support_factor_mean']:.1f}, "
                f"rank_raw={d['ranking_raw_risk_mean']:.5g}, "
                f"rank_abs_correction={d['ranking_abs_correction_fraction']:.3f}, "
                f"steps_per_epoch={d['actual_steps_per_epoch']:.1f}, "
                f"steps_per_round={d['actual_steps_per_round']:.0f}, "
                f"base_weight={d['base_aggregation_weight']:.4f}, "
                f"support_weight={d['support_aggregation_factor']:.5g}, "
                f"effective_aggregation_weight={d['effective_aggregation_weight']:.4f}"
            for i, d in enumerate(client_diags)
        ))
        if evaluated:
            print(f"[Proxy] round={r+1}; selection_auc={proxy_auc:.6f}; calibration_score={proxy_score:.6f}")

        if round_callback is not None:
            round_callback(r, acc, rec, auc, f1, thr, spec, bacc, fp, fn)

    # Restore only after all communication rounds have completed. Per-round
    # calibration above is reporting-only; the selected checkpoint receives
    # a fresh final calibration here. Proxy labels select/calibrate the final
    # model, while test labels never affect gradients, aggregation, or selection.
    if best_eval_state is not None:
        eval_model.load_state_dict(best_eval_state)
    else:
        # Fallback for runs with no scheduled proxy evaluation.
        best_round = rounds - 1
        selection_proxy_logits = _infer_logits(eval_model, selection_proxy_x)
        best_proxy_auc = ProxyCheckpointInference.proxy_auc(
            selection_proxy_logits, selection_proxy_t
        )
        selected_proxy_auc = float(best_proxy_auc)

    final_proxy_logits = _infer_logits(eval_model, calibration_proxy_x)
    final_proxy_score = float("nan")
    if calibration_proxy_t is not None:
        final_threshold, final_proxy_score, final_proxy_specificity = (
            ProxyCheckpointInference.calibrated_threshold_with_score(
                final_proxy_logits, calibration_proxy_t
            )
        )
    else:
        final_threshold = ProxyCheckpointInference.threshold(
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
        "threshold_objective": "acc",
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
        "[Final threshold calibration] metric=proxy_ACC_unconstrained, "
        f"proxy_specificity={final_proxy_specificity:.4f}, "
        f"score={final_proxy_score:.4f}, threshold={final_thr:.4f}"
    )
    print(
        f"[Final selection] round={best_round + 1}, "
        f"selected_proxy_AUC={selected_proxy_auc:.4f}, max_proxy_AUC={best_proxy_auc:.4f}, "
        f"tolerance={SELECTION_AUC_TOLERANCE:g}, "
        f"threshold={final_thr:.4f}"
    )
    return eval_model, history, float(final_acc)
