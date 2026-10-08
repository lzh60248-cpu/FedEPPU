"""Centralized PN upper bound and centralized standard-nnPU reference."""
import time
import torch
import torch.nn.functional as F

from .nnpu_common import mixed_batches, train_epoch


def train_centralized_pn(model, x, true_y, rounds, epochs, lr, batch_size,
                         max_steps_per_epoch, round_callback=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print(f"Centralized PN training started | device={x.device}", flush=True)
    for round_index in range(rounds):
        for epoch_index in range(epochs):
            started = time.monotonic()
            total_loss, updates = 0.0, 0
            model.train()
            for bx, by in mixed_batches(x, true_y, batch_size,
                                        max_steps_per_epoch):
                target = (by.reshape(-1) == 1).float()
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    model(bx).reshape(-1), target
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                updates += 1
            print(
                f"Centralized PN | Round {round_index + 1:03d}/{rounds} | "
                f"Epoch {epoch_index + 1}/{epochs} | steps={updates} | "
                f"loss={total_loss / max(updates, 1):.6f} | "
                f"seconds={time.monotonic() - started:.1f}", flush=True,
            )
        if round_callback is not None:
            round_callback(round_index, model)
    return model


def train_centralized_nnpu(model, x, observed_y, prior, rounds, epochs, lr,
                           batch_size, max_steps_per_epoch,
                           round_callback=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print(f"Centralized nnPU training started | device={x.device}", flush=True)
    for round_index in range(rounds):
        for epoch_index in range(epochs):
            started = time.monotonic()
            loss, updates = train_epoch(model, x, observed_y, prior, optimizer, batch_size,
                        max_steps_per_epoch=max_steps_per_epoch)
            print(
                f"Centralized nnPU | Round {round_index + 1:03d}/{rounds} | "
                f"Epoch {epoch_index + 1}/{epochs} | steps={updates} | "
                f"loss={loss:.6f} | seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
        if round_callback is not None:
            round_callback(round_index, model)
    return model
