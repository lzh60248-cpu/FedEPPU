"""Binary specialization of Lin et al.'s ICML 2022 FedPU risk.

For one labeled positive class (class 1) and one negative class (class 0),
the official multi-class source risk reduces to

    CE(P, 1) + pu_weight * (CE(U, 0) - pi * CE(P, 0)).

The multi-negative cross-class term is empty in a binary problem. Labels use
the project PU protocol: 1 is labeled positive and -1 is unlabeled.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FedPUBinaryLoss(nn.Module):
    """Paper/source-faithful binary reduction of ``MPULoss_V2``."""

    def __init__(self, pu_weight=1.0):
        super().__init__()
        self.pu_weight = float(pu_weight)

    def forward(self, outputs, observed_labels, prior):
        if outputs.ndim != 2 or outputs.size(1) != 2:
            raise ValueError("FedPU binary loss requires two output logits")
        labels = observed_labels.reshape(-1)
        p_mask = labels == 1
        u_mask = labels != 1
        if not bool(p_mask.any()) or not bool(u_mask.any()):
            raise ValueError("each FedPU update batch must contain both P and U")

        logits_p = outputs[p_mask].float()
        logits_u = outputs[u_mask].float()
        positive_ce = F.cross_entropy(
            logits_p, torch.ones(len(logits_p), device=outputs.device,
                                 dtype=torch.long)
        )
        unlabeled_negative = F.cross_entropy(
            logits_u, torch.zeros(len(logits_u), device=outputs.device,
                                  dtype=torch.long)
        )
        positive_as_negative = F.cross_entropy(
            logits_p, torch.zeros(len(logits_p), device=outputs.device,
                                  dtype=torch.long)
        )
        correction = unlabeled_negative - float(prior) * positive_as_negative
        total = positive_ce + self.pu_weight * correction
        return total, correction, positive_ce, unlabeled_negative, positive_as_negative


FedPUMPULoss = FedPUBinaryLoss


def _take_cyclic(pool, count, generator):
    """Draw count indices, reshuffling and wrapping when necessary."""
    if pool.numel() == 0:
        raise ValueError("FedPU requires non-empty P and U pools")
    parts = []
    remaining = int(count)
    while remaining:
        order = pool[torch.randperm(pool.numel(), generator=generator,
                                    device=pool.device)]
        take = min(remaining, order.numel())
        parts.append(order[:take])
        remaining -= take
    return torch.cat(parts)


def local_train_fedpu(model, x, t, prior, epochs, lr, batch_size,
                      pu_weight=1.0, momentum=0.9,
                      max_steps_per_epoch=None, seed=None):
    """Train one FedPU client with batches containing both P and U."""
    if not 0.0 <= float(prior) <= 1.0:
        raise ValueError("prior must lie in [0, 1]")
    if batch_size < 2:
        raise ValueError("FedPU batch_size must be at least two")
    p_pool = torch.nonzero(t.reshape(-1) == 1, as_tuple=True)[0]
    u_pool = torch.nonzero(t.reshape(-1) != 1, as_tuple=True)[0]
    if p_pool.numel() == 0 or u_pool.numel() == 0:
        raise ValueError("FedPU client requires both labeled P and U samples")

    observed_fraction = p_pool.numel() / float(p_pool.numel() + u_pool.numel())
    n_p = min(batch_size - 1, max(1, int(round(batch_size * observed_fraction))))
    n_u = batch_size - n_p
    natural_steps = int(math.ceil(len(x) / float(batch_size)))
    steps = natural_steps if max_steps_per_epoch is None else min(
        natural_steps, int(max_steps_per_epoch)
    )
    if steps < 1:
        raise ValueError("max_steps_per_epoch must be positive")

    criterion = FedPUBinaryLoss(pu_weight).to(x.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(torch.initial_seed() if seed is None else seed))
    totals = {"loss": 0.0, "correction": 0.0, "positive_ce": 0.0,
              "unlabeled_negative": 0.0, "positive_as_negative": 0.0}
    updates = 0
    model.train()
    for _ in range(int(epochs)):
        p_order = _take_cyclic(p_pool, steps * n_p, generator)
        u_order = _take_cyclic(u_pool, steps * n_u, generator)
        for step in range(steps):
            idx = torch.cat((p_order[step*n_p:(step+1)*n_p],
                             u_order[step*n_u:(step+1)*n_u]))
            idx = idx[torch.randperm(idx.numel(), generator=generator,
                                     device=idx.device)]
            optimizer.zero_grad(set_to_none=True)
            values = criterion(model(x[idx]), t[idx], prior)
            values[0].backward()
            optimizer.step()
            for key, value in zip(totals, values):
                totals[key] += float(value.detach().cpu())
            updates += 1
    diagnostics = {key: value / updates for key, value in totals.items()}
    diagnostics.update({
        "updates": updates, "pu_weight": float(pu_weight),
        "p_per_batch": n_p, "u_per_batch": n_u, "prior": float(prior),
    })
    return model, diagnostics
