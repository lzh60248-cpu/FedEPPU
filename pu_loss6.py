"""Standard nnPU risk used by the multi-seed robust version 6."""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

_NNPU_BETA: float = 0.0
_NNPU_GAMMA: float = 1.0


def _resolve_prior(pi) -> float:
    if isinstance(pi, torch.Tensor):
        pi = float(pi.detach().cpu().item())
    pi = float(pi)
    return 1e-3 if pi < 1e-3 else (1.0 - 1e-3 if pi > 1.0 - 1e-3 else pi)


def _as_binary_logits(t: torch.Tensor) -> torch.Tensor:
    if t.dim() > 1 and t.shape[1] != 1:
        raise ValueError(f"Binary PU loss expects 1 logit per sample; got {tuple(t.shape)}.")
    return t.reshape(-1)


def pu_loss(
    outputs: torch.Tensor,
    labels:  torch.Tensor,
    pi:      Union[float, int, np.floating, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard nnPU risk with beta=0 and gamma=1.

    outputs : raw binary logits [N] or [N,1]
    labels  : 1 = positive, -1 = unlabeled
    pi      : positive-class prior
    """
    pi     = _resolve_prior(pi)
    logits = _as_binary_logits(outputs)
    labels = labels.reshape(-1)
    pos_m, unl_m = labels == 1, labels == -1

    bce = F.binary_cross_entropy_with_logits
    n_pos = int(pos_m.sum().detach().cpu().item())
    n_unl = int(unl_m.sum().detach().cpu().item())

    # 极端小 batch 的安全退化：不返回新建叶子常量 0。
    # 这样即便 local_train 的混合保护在某些边界 batch 上无法同时放入 P/U，
    # 当前 batch 仍会给模型一个与 logits 相连的梯度信号。
    if n_pos == 0 and n_unl == 0:
        z = logits.sum() * 0.0
        return z, z.detach()
    if n_pos == 0:
        unl_l = logits[unl_m]
        total = bce(unl_l, torch.zeros_like(unl_l))
        return total, total.detach()
    if n_unl == 0:
        pos_l = logits[pos_m]
        total = pi * bce(pos_l, torch.ones_like(pos_l))
        return total, total.detach()

    pos_l, unl_l = logits[pos_m], logits[unl_m]

    risk_p_pos = pi * bce(pos_l, torch.ones_like(pos_l))
    risk_p_neg = pi * bce(pos_l, torch.zeros_like(pos_l))
    risk_u_neg = bce(unl_l, torch.zeros_like(unl_l))
    est_neg    = risk_u_neg - risk_p_neg

    # Kiryo et al. nnPU correction: when the estimated negative risk becomes
    # negative, reverse its gradient instead of imposing an unrelated positive
    # floor.  Do not detach est_neg: the corrective branch needs its gradient.
    if est_neg.detach().item() < -_NNPU_BETA:
        total = -_NNPU_GAMMA * est_neg
    else:
        total = risk_p_pos + est_neg

    return total, total.detach()
