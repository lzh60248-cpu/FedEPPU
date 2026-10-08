"""Multi-seed robust D2 variant with capped FedAvg and proxy-AUC selection.

The paper-facing method contains only three extensions: empirical-Bayes prior
correction, prior-consistent nnPU local learning, and reliability-aware FedAvg.
A fixed model EMA and an EB-prior quantile provide evaluation and prediction.
Version 6 keeps version 5 as its default numerical path and additionally lets
the Chest entry point provide a faster parameter-free annealing function.
AMP, clipping, augmentation, and bounded local steps are implementation details.
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
    roc_curve,
)

from pu_loss6 import pu_loss

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


def _annealing_scale(round_idx: int, total_rounds: int) -> float:
    """Parameter-free standard cosine learning-rate scale."""
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive.")
    progress = float(round_idx) / float(total_rounds)
    return float(0.5 * (1.0 + np.cos(np.pi * progress)))


def _prior_annealing_scale(round_idx: int, total_rounds: int) -> float:
    """B4: fixed prior-consistency strength without cosine annealing."""
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive.")
    return 1.0


def _round_lr(base_lr: float, round_idx: int, total_rounds: int) -> float:
    """Cosine-annealed base learning rate."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


def _forward_with_features(
    model: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return logits and penultimate features when the backbone exposes them."""
    if hasattr(model, "forward_features") and hasattr(model, "classify_features"):
        features = model.forward_features(x)
        return model.classify_features(features), features
    return model(x), None


def _prototype_alignment_loss(
    features: Optional[torch.Tensor],
    logits: torch.Tensor,
    labels: torch.Tensor,
    prototypes: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Margin-free positive/negative prototype alignment.

    Labeled positives approach the global positive prototype. U examples are
    attracted to the negative prototype in proportion to their detached
    negative probability. No confidence threshold, temperature, or margin is
    introduced.
    """
    if features is None or prototypes is None:
        return logits.sum() * 0.0
    positive_bank, negative_bank = prototypes
    z = F.normalize(features.float(), dim=1)
    positive_bank = F.normalize(positive_bank.float(), dim=1)
    negative_bank = F.normalize(negative_bank.float(), dim=1)
    # The best matching client prototype preserves multimodal class structure;
    # the bank size is determined by the participating clients.
    sim_pos = torch.max(z @ positive_bank.T, dim=1).values
    sim_neg = torch.max(z @ negative_bank.T, dim=1).values
    labels = labels.reshape(-1)
    terms: List[torch.Tensor] = []

    positive = labels == 1
    if torch.any(positive):
        terms.append(F.softplus(sim_neg[positive] - sim_pos[positive]).mean())

    unlabeled = labels == -1
    if torch.any(unlabeled):
        negative_weight = (1.0 - torch.sigmoid(
            logits.reshape(-1)[unlabeled].detach().float()
        ))
        per_sample = F.softplus(sim_pos[unlabeled] - sim_neg[unlabeled])
        denominator = negative_weight.sum().clamp_min(
            torch.finfo(negative_weight.dtype).eps
        )
        terms.append(torch.sum(per_sample * negative_weight) / denominator)

    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _global_prototypes(
    model: nn.Module,
    client_data: List[Tuple[torch.Tensor, torch.Tensor]],
    effective_priors: np.ndarray,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Build one positive and one negative prototype per eligible client."""
    if not hasattr(model, "forward_features"):
        return None
    model.eval()
    positive_bank: List[torch.Tensor] = []
    negative_bank: List[torch.Tensor] = []

    with torch.no_grad():
        for (x, labels), prior in zip(client_data, effective_priors):
            positive_parts: List[torch.Tensor] = []
            unlabeled_features: List[torch.Tensor] = []
            unlabeled_logits: List[torch.Tensor] = []
            for start in range(0, len(x), 512):
                bx = x[start:start + 512]
                bt = labels[start:start + 512]
                logits, features = _forward_with_features(model, bx)
                if features is None:
                    return None
                if torch.any(bt == 1):
                    positive_parts.append(features[bt == 1].float())
                if torch.any(bt == -1):
                    unlabeled_features.append(features[bt == -1].float())
                    unlabeled_logits.append(logits.reshape(-1)[bt == -1].float())

            if positive_parts:
                positive_features = torch.cat(positive_parts, dim=0)
                positive_bank.append(positive_features.mean(dim=0))

            if unlabeled_features:
                u_features = torch.cat(unlabeled_features, dim=0)
                u_logits = torch.cat(unlabeled_logits, dim=0)
                negative_count = int(round((1.0 - float(prior)) * len(u_logits)))
                negative_count = min(max(negative_count, 1), len(u_logits))
                selected = torch.topk(
                    u_logits, k=negative_count, largest=False, sorted=False
                ).indices
                negative_bank.append(u_features[selected].mean(dim=0))

    if not positive_bank or not negative_bank:
        return None
    return (
        torch.stack(positive_bank, dim=0).detach(),
        torch.stack(negative_bank, dim=0).detach(),
    )


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
    """Spatial augmentation for CIFAR-10 and single-channel Chest X-rays."""
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
    mh, mw = padded.shape[2] - h, padded.shape[3] - w
    oh = torch.randint(0, mh + 1, (n,), device=x.device)
    ow = torch.randint(0, mw + 1, (n,), device=x.device)

    # Vectorized per-sample crop. The previous Python loop launched one slice
    # assignment per image (256 launches for a normal batch), which dominated
    # runtime when clients performed hundreds of local steps. These two GPU
    # gathers preserve the same independently sampled offsets.
    channels = padded.shape[1]
    padded_w = padded.shape[3]
    row_index = (
        oh[:, None] + torch.arange(h, device=x.device)[None, :]
    )[:, None, :, None].expand(n, channels, h, padded_w)
    height_cropped = torch.gather(padded, dim=2, index=row_index)
    col_index = (
        ow[:, None] + torch.arange(w, device=x.device)[None, :]
    )[:, None, None, :].expand(n, channels, h, w)
    return torch.gather(height_cropped, dim=3, index=col_index).contiguous()


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
    with torch.inference_mode():
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
    try:
        auc = roc_auc_score(t_cpu, probs)
    except Exception:
        auc = 0.5
    return acc, rec, auc, f1, thr


# ===========================================================================
# EB-prior quantile decision rule
# ===========================================================================

def _quantile_threshold(logits: np.ndarray, target_prior: float) -> float:
    logits = _flat(logits)
    k = int(round(len(logits) * target_prior))
    k = 1 if k < 1 else (len(logits) - 1 if k > len(logits) - 1 else k)
    return float(np.sort(logits)[::-1][k - 1])


def _proxy_balanced_threshold(
    logits: np.ndarray,
    labels: torch.Tensor,
    fallback_prior: float,
) -> float:
    """Select the labeled-proxy threshold with maximum balanced accuracy."""
    z = _flat(logits)
    y = (_flat(labels) > 0).astype(np.int64)
    if z.size != y.size:
        raise ValueError("Proxy logits and labels must have the same length.")
    if np.unique(y).size < 2:
        return _quantile_threshold(z, fallback_prior)
    fpr, tpr, thresholds = roc_curve(y, z)
    balanced = 0.5 * (tpr + 1.0 - fpr)
    best = int(np.flatnonzero(balanced == np.max(balanced))[0])
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):
        return _quantile_threshold(z, fallback_prior)
    return threshold


# ===========================================================================
# 客户端权重（方案二：接入先验可靠性信号）
# ===========================================================================

def client_weights(
    sample_counts: np.ndarray,
    prior_extremes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample-aware + reliability-aware aggregation with capped redistribution.

    - 基础权重使用标准 FedAvg 样本量 n_k；
    - 若提供先验偏离 d_k，可靠性固定定义为 exp(-d_k)；
    - 单客户端权重不超过 1-1/K，确保其余客户端合计至少占 1/K；
    - 超出上限的质量按其余客户端相对权重重新分配。
    """
    sw = np.maximum(np.asarray(sample_counts, dtype=np.float64), 1.0)

    if prior_extremes is not None:
        d = np.maximum(np.asarray(prior_extremes, dtype=np.float64), 0.0)
        reliability = np.exp(-d)
        sw = sw * reliability

    sw /= sw.sum() + 1e-12

    n_clients = sw.size
    if n_clients <= 1:
        return np.ones_like(sw, dtype=np.float64)

    cap = 1.0 - 1.0 / n_clients

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
    global_prototypes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    schedule: InternalSchedule = SCHEDULE,
) -> nn.Module:
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
    prior_scale = float(np.exp(-(extreme ** 2)))

    actual_lr  = lr * prior_scale
    epochs     = local_epochs
    lambda_prototype_k = float(
        lambda_prior * prior_round_scale * np.exp(-(extreme ** 2))
    )

    opt = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=CFG.weight_decay)
    # A14: force full FP32 training.
    amp_enabled = False
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model.train()
    pos_idx = torch.where(t == 1)[0]
    unl_idx = torch.where(t == -1)[0]
    mixed   = pos_idx.numel() > 0 and unl_idx.numel() > 0

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
                logits, features = _forward_with_features(model, bx)

            # Evaluate PU risks in FP32 even under AMP.  This retains AMP's
            # convolution speedup without exposing the subtraction in nnPU to
            # unnecessary half-precision cancellation.
            logits_fp32 = logits.float()
            loss, _ = pu_loss(logits_fp32, bt, prior)
            loss = loss + lambda_prototype_k * _prototype_alignment_loss(
                features, logits_fp32, bt, global_prototypes
            )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)  # CLIP #3
            scaler.step(opt)
            scaler.update()

    return model


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
    rounds: int = 200,
    n_clients: int = 3,
    lambda_prior: float = 0.10,
    batch_size: int = CFG.batch_size,
    max_steps_per_epoch: int = CFG.max_steps_per_epoch,
    local_epochs: int = CFG.base_epochs,
    use_amp: bool = True,
    n_shrink: float = CFG.n_shrink,
    schedule: InternalSchedule = SCHEDULE,
    annealing_fn: Optional[Callable[[int, int], float]] = None,
    prior_annealing_fn: Optional[Callable[[int, int], float]] = None,
    annealing_name: str = "standard_cosine_lr",
) -> Tuple[nn.Module, Dict, float]:
    """Multi-seed robust D2 federated PU training loop.

    Raw client priors enter standard nnPU, while prior consistency is disabled.
    The method combines deviation-scaled local learning, a client prototype
    bank, dominance-capped sample-count FedAvg, cosine learning-rate annealing,
    fixed EMA, proxy-AUC checkpoint selection and proxy-quantile inference.
    """
    raw_priors    = np.array([_to_float(p) for p in priors], np.float64)
    opt_annealing = annealing_fn or _annealing_scale
    pc_annealing = prior_annealing_fn or _prior_annealing_scale

    def checked_scale(fn: Callable[[int, int], float], round_idx: int) -> float:
        value = float(fn(round_idx, rounds))
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(
                "Annealing functions must return a finite value in [0, 1], "
                f"got {value} at round {round_idx}."
            )
        return value

    if lambda_prior < 0:
        raise ValueError("lambda_prior must be non-negative.")
    if n_shrink < 0:
        raise ValueError("n_shrink must be non-negative.")
    if batch_size < 1 or max_steps_per_epoch < 1 or local_epochs < 1:
        raise ValueError("batch_size, max_steps_per_epoch and local_epochs must be positive.")
    if len(client_data) != n_clients or len(raw_priors) != n_clients:
        raise ValueError(
            "n_clients must equal both len(client_data) and len(priors): "
            f"got {n_clients}, {len(client_data)}, {len(raw_priors)}."
        )
    if proxy_t is None:
        raise ValueError("Version 6 requires proxy labels for proxy-AUC model selection.")
    if len(proxy_t) != len(proxy_x):
        raise ValueError("proxy_t and proxy_x must have the same length.")
    proxy_binary_labels = (_flat(proxy_t) > 0).astype(np.int64)
    if np.unique(proxy_binary_labels).size < 2:
        raise ValueError("Proxy-AUC selection requires both proxy classes.")
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
    center_prior = compute_global_prior(raw_priors, prior_counts)

    # A1: raw client priors are used directly without EB shrinkage.
    eps = np.finfo(np.float64).eps
    effective_priors = np.clip(raw_priors, eps, 1.0 - eps)
    extremes = np.asarray(
        [compute_prior_extreme(p, center_prior) for p in effective_priors],
        dtype=np.float64,
    )

    target_prior = float(np.clip(
        center_prior,
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    ))
    global_model = copy.deepcopy(anchor_model).to(device)
    eval_model   = copy.deepcopy(global_model).to(device)

    history: Dict[str, object] = {}

    print("Prior correction: disabled (raw client priors are used)")
    for i in range(n_clients):
        print(f"  [Client {i+1}] n={int(sample_counts[i])}, n_u={int(prior_counts[i])}, "
              f"raw_pi={raw_priors[i]:.4f}, eff_pi={effective_priors[i]:.4f}, "
              f"extreme={extremes[i]:.4f}")
    print(f"center_prior={center_prior:.4f} | target_prior={target_prior:.4f} "
          "(pooled raw prior, no test labels) | n_shrink=disabled")
    print(
        "training=nnPU+ClientPrototypeBank | D2=PriorConsistencyOff | EB=off | "
        "aggregation=boundedFedAvg | precision=FP32(AMP off) | "
        "annealing=standardCosineLR | selection=bestProxyAUC(noEarlyStop) | "
        "proxyBAccCalibration=off | final_threshold=centerPriorQuantile"
    )

    # Sample-count FedAvg followed by the parameter-free 1-1/K dominance cap.
    aggregation_weights = client_weights(sample_counts, prior_extremes=None)
    aggregation_cap = 1.0 if n_clients <= 1 else 1.0 - 1.0 / n_clients
    print(
        "aggregation_strategy=boundedFedAvg(sample_count_only); "
        f"reliability=disabled; cap=1-1/K={aggregation_cap:.4f}"
    )
    print("aggregation_weights=" + ", ".join(
        f"client{i+1}:{w:.4f}" for i, w in enumerate(aggregation_weights)
    ))

    global_prototypes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    best_proxy_auc = -np.inf
    best_round = 0
    best_model_state: Optional[Dict[str, torch.Tensor]] = None

    for r in range(rounds):
        annealing_scale = checked_scale(opt_annealing, r)
        round_lr = CFG.lr_base * annealing_scale
        # Anneal only the update magnitude. Keeping a fixed per-round compute
        # budget avoids prematurely stopping representation learning.
        current_max_steps = min(max_steps_per_epoch, median_client_steps)
        prior_round_scale = checked_scale(pc_annealing, r)

        client_models = []
        for i in range(n_clients):
            m_i = local_train(
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
                global_prototypes=global_prototypes,
                schedule=schedule,
            )
            client_models.append(m_i)

        # 方案二：聚合权重同时考虑样本量与先验可靠性 (extremes = d_k)
        weights = aggregation_weights

        # 加权 FedAvg
        global_state = global_model.state_dict()
        new_state = {
            k: (torch.zeros_like(v) if torch.is_floating_point(v)
                else torch.zeros_like(v, dtype=torch.float64))
            for k, v in global_state.items()
        }
        for i, w in enumerate(weights):
            wt = torch.tensor(float(w), device=device)
            for k, v in client_models[i].state_dict().items():
                if torch.is_floating_point(v):
                    new_state[k] += wt * v.to(device)
                else:
                    new_state[k] += wt.double() * v.to(device=device, dtype=torch.float64)

        # Integer buffers (if a custom model has any) are combined without
        # privileging client 0, then converted back to their declared dtype.
        for k, ref in global_state.items():
            if not torch.is_floating_point(ref):
                new_state[k] = new_state[k].round().to(dtype=ref.dtype)

        # Dominance-capped sample-count FedAvg; no reliability score or momentum.
        global_model.load_state_dict(new_state)

        # Build client-specific prototype banks with the newly aggregated model,
        # so all entries share one feature coordinate system. Round 1 establishes
        # the banks; nearest-prototype alignment starts naturally from round 2.
        if r + 1 < rounds:
            global_prototypes = _global_prototypes(
                global_model, client_data, effective_priors
            )
            if r == 0 and global_prototypes is not None:
                print(
                    "prototype_bank="
                    f"positive:{global_prototypes[0].shape[0]}, "
                    f"negative:{global_prototypes[1].shape[0]}, "
                    "matching:maxCosine"
                )

        # Fixed-momentum model EMA; no round-dependent EMA schedule.
        if r == 0:
            eval_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        else:
            _update_model_ema(
                eval_model, global_model, mom=schedule.model_ema_momentum
            )

        # The proxy quantile supplies the same label-free decision threshold
        # used at final deployment. Proxy labels affect checkpoint selection
        # through AUC only; they never enter local training or threshold search.
        round_proxy_logits = _infer_logits(eval_model, proxy_x)
        round_threshold = _quantile_threshold(round_proxy_logits, target_prior)
        round_proxy_auc = float(roc_auc_score(
            proxy_binary_labels, round_proxy_logits
        ))
        if round_proxy_auc > best_proxy_auc:
            best_proxy_auc = round_proxy_auc
            best_round = r + 1
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in eval_model.state_dict().items()
            }

        round_acc, round_rec, round_auc, round_f1, round_thr = evaluate(
            eval_model,
            x_test,
            t_test,
            fixed_threshold=round_threshold,
        )
        print(
            f"Round {r + 1:03d} | AUC:{round_auc:.4f} F1:{round_f1:.4f} "
            f"Acc:{round_acc:.4f} Recall:{round_rec:.4f} Thr:{round_thr:.4f}"
        )

    if best_model_state is None:
        raise RuntimeError("No proxy-AUC checkpoint was recorded.")
    eval_model.load_state_dict(best_model_state)

    # Recompute the label-free prior-quantile threshold once for the selected
    # EMA checkpoint, then apply that unchanged threshold to the test set.
    final_proxy_logits = _infer_logits(eval_model, proxy_x)
    final_threshold = _quantile_threshold(final_proxy_logits, target_prior)
    final_proxy_auc = float(roc_auc_score(
        proxy_binary_labels, final_proxy_logits
    ))
    final_acc, final_rec, final_auc, final_f1, final_thr = evaluate(
        eval_model,
        x_test,
        t_test,
        target_prior=target_prior,
        fixed_threshold=final_threshold,
    )
    history["selected"] = {
        "round": int(best_round),
        "proxy_auc": final_proxy_auc,
        "acc": float(final_acc),
        "recall": float(final_rec),
        "auc": float(final_auc),
        "f1": float(final_f1),
        "threshold": float(final_thr),
    }
    print(
        f"[Selected checkpoint] round={best_round}, "
        f"proxy_AUC={final_proxy_auc:.4f}, threshold={final_thr:.4f} "
        "(proxy-AUC selection; center-prior proxy quantile)"
    )
    return eval_model, history, float(final_acc)
