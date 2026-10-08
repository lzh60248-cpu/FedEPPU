"""Shared utilities for fair binary nnPU federated baselines."""
from __future__ import annotations
import copy
from typing import Callable, Dict, Iterator, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .standard_nnpu import nnpu_loss

def mixed_batches(x, t, batch_size, max_steps_per_epoch=None):
    if len(x) != len(t) or len(x) == 0:
        raise ValueError("x and t must be non-empty and have equal length")
    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    order = torch.randperm(len(x), device=x.device)
    steps = int(np.ceil(len(x) / batch_size))
    if max_steps_per_epoch is not None:
        if max_steps_per_epoch < 1:
            raise ValueError("max_steps_per_epoch must be positive")
        steps = min(steps, int(max_steps_per_epoch))
    for step in range(steps):
        idx = order[step * batch_size:(step + 1) * batch_size]
        if idx.numel() == 0:
            break
        yield x[idx], t[idx]

def train_epoch(model, x, t, prior, optimizer, batch_size, proximal_reference=None,
                mu=0.0, gradient_correction=None, max_steps_per_epoch=None):
    model.train(); total = 0.0; updates = 0
    reference = None if proximal_reference is None or mu <= 0 else {n: p.detach() for n, p in proximal_reference.named_parameters()}
    for bx, bt in mixed_batches(x, t, batch_size, max_steps_per_epoch):
        optimizer.zero_grad(set_to_none=True)
        loss = nnpu_loss(model(bx), bt, prior)
        if reference is not None:
            prox = loss.new_zeros(())
            for name, p in model.named_parameters(): prox = prox + (p - reference[name]).pow(2).sum()
            loss = loss + 0.5 * float(mu) * prox
        loss.backward()
        if gradient_correction is not None:
            ci, c = gradient_correction
            for name, p in model.named_parameters():
                if p.grad is not None: p.grad.add_(c[name] - ci[name])
        optimizer.step(); total += float(loss.detach().cpu()); updates += 1
    return total / max(updates, 1), updates

def clone_model(model): return copy.deepcopy(model)

def model_delta_norm(local, global_model):
    with torch.no_grad(): return float(np.sqrt(sum(float((lp-gp).pow(2).sum().cpu()) for lp,gp in zip(local.parameters(), global_model.parameters()))))

def average_models(target, models, weights):
    if not models or len(models) != len(weights): raise ValueError("models/weights mismatch")
    weights = np.asarray(weights, np.float64); weights /= weights.sum(); refs = target.state_dict(); out = {}
    for key, ref in refs.items():
        value = torch.zeros_like(ref) if torch.is_floating_point(ref) else torch.zeros_like(ref, dtype=torch.float64)
        for weight, model in zip(weights, models): value.add_(model.state_dict()[key].to(ref.device, value.dtype), alpha=float(weight))
        out[key] = value if torch.is_floating_point(ref) else value.round().to(ref.dtype)
    target.load_state_dict(out)
