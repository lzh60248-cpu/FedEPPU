"""Independent client training with no communication or aggregation."""
import time
from .fedavg_pu import local_train_fedavg


def train_local_models(models, clients, priors, rounds, epochs, lr, batch_size,
                       max_steps_per_epoch, round_callback=None):
    print("Local nnPU training started", flush=True)
    for round_index in range(rounds):
        for index, model in enumerate(models):
            x, labels = clients[index]
            for epoch_index in range(epochs):
                started = time.monotonic()
                models[index], diag = local_train_fedavg(
                    model, x, labels, priors[index], 1, lr, batch_size,
                    max_steps_per_epoch=max_steps_per_epoch,
                )
                print(
                    f"Local nnPU | Round {round_index + 1:03d}/{rounds} | "
                    f"Client {index + 1}/{len(models)} | "
                    f"Epoch {epoch_index + 1}/{epochs} | "
                    f"steps={diag['updates']} | loss={diag['loss']:.6f} | "
                    f"seconds={time.monotonic() - started:.1f}", flush=True,
                )
        if round_callback is not None:
            round_callback(round_index, models)
    return models
