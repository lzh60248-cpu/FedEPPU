"""SCAFFOLD Option-II with a standard binary nnPU local objective."""
from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
from .nnpu_common import train_epoch

def zero_control(model): return {name: torch.zeros_like(p) for name, p in model.named_parameters()}

def local_train_scaffold(model, global_model, x, t, prior, client_control,
                         server_control, epochs, lr, batch_size,
                         max_steps_per_epoch=None):
    if lr <= 0: raise ValueError("lr must be positive")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses=[]; updates=0
    for _ in range(epochs):
        loss, count = train_epoch(
            model, x, t, prior, optimizer, batch_size,
            gradient_correction=(client_control, server_control),
            max_steps_per_epoch=max_steps_per_epoch,
        )
        losses.append(loss); updates += count
    if updates == 0: raise RuntimeError("SCAFFOLD client performed no updates")
    lp, gp = dict(model.named_parameters()), dict(global_model.named_parameters()); new_control={}
    with torch.no_grad():
        for name in client_control:
            new_control[name]=(client_control[name]-server_control[name]+(gp[name]-lp[name])/(updates*lr)).detach().clone()
    return model,new_control,{"loss":sum(losses)/max(len(losses),1),"updates":updates}

def update_server_control(server_control, old_controls, new_controls, total_clients):
    with torch.no_grad():
        for name in server_control:
            delta=torch.zeros_like(server_control[name])
            for old,new in zip(old_controls,new_controls): delta.add_(new[name]-old[name])
            server_control[name].add_(delta,alpha=1.0/total_clients)
