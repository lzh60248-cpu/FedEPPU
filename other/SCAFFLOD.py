import copy
import math
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


PriorLike = Union[float, np.float32, np.float64, torch.Tensor, Dict[object, float]]


def prior_to_float(pi_dict: PriorLike) -> float:
    if isinstance(pi_dict, (float, int, np.float32, np.float64)):
        return float(pi_dict)

    if torch.is_tensor(pi_dict):
        return float(pi_dict.detach().cpu().item())

    if isinstance(pi_dict, dict):
        return float(sum(float(v) for v in pi_dict.values()))

    raise TypeError(f"Unsupported pi_dict type: {type(pi_dict)}")


def update_global_avg_risk(
    prev_avg: Optional[torch.Tensor],
    current_risk: Optional[torch.Tensor],
    momentum: float = 0.9
) -> Optional[torch.Tensor]:
    if current_risk is None:
        return prev_avg

    if torch.is_tensor(current_risk):
        current = current_risk.detach().float().cpu()
    else:
        current = torch.tensor(float(current_risk), dtype=torch.float32)

    if prev_avg is None:
        return current

    if torch.is_tensor(prev_avg):
        prev = prev_avg.detach().float().cpu()
    else:
        prev = torch.tensor(float(prev_avg), dtype=torch.float32)

    momentum = float(np.clip(momentum, 0.0, 0.999))
    return momentum * prev + (1.0 - momentum) * current


def extract_binary_logits(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 2 and outputs.shape[1] == 2:
        return outputs[:, 1] - outputs[:, 0]

    if outputs.ndim == 2 and outputs.shape[1] == 1:
        return outputs.view(-1)

    if outputs.ndim == 1:
        return outputs

    raise ValueError(f"Unexpected outputs shape: {outputs.shape}")


def dynamic_beta_boundary(
    prior: float,
    n_samples: int,
    beta_decay: float,
    round_idx: int
) -> float:
    prior = float(np.clip(prior, 1e-4, 1.0 - 1e-4))
    n_samples = max(int(n_samples), 1)
    beta_decay = max(float(beta_decay), 1e-4)

    sample_term = 1.0 / math.sqrt(n_samples)
    rarity_term = math.sqrt((1.0 - prior) / prior)
    round_term = 1.0 / math.sqrt(float(round_idx) + 1.0)

    beta = beta_decay * (
        0.04
        + 0.08 * sample_term * rarity_term
        + 0.02 * round_term
    )

    return float(np.clip(beta, 0.003, 0.20))


def smooth_non_negative_risk(
    est_neg_risk: torch.Tensor,
    beta: float,
    softness: float = 0.1
) -> torch.Tensor:
    softness = max(float(softness), 1e-6)

    beta_tensor = torch.tensor(
        float(beta),
        dtype=est_neg_risk.dtype,
        device=est_neg_risk.device
    )

    return beta_tensor + softness * F.softplus(
        (est_neg_risk - beta_tensor) / softness
    )


def nnpu_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    pi_dict: PriorLike,
    n_samples: int,
    global_avg_risk: Optional[torch.Tensor] = None,
    lam: float = 0.01,
    beta_decay: float = 0.1,
    round_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = extract_binary_logits(outputs)
    labels = labels.view(-1).to(outputs.device)

    prior = float(np.clip(prior_to_float(pi_dict), 0.01, 0.99))

    pos_mask = labels == 1
    unlabeled_mask = labels == -1

    if pos_mask.sum().item() == 0 or unlabeled_mask.sum().item() == 0:
        zero = outputs.sum() * 0.0
        return zero, zero.detach()

    pos_logits = outputs[pos_mask]
    u_logits = outputs[unlabeled_mask]

    risk_p_pos = prior * F.binary_cross_entropy_with_logits(
        pos_logits,
        torch.ones_like(pos_logits),
        reduction="mean"
    )

    risk_u_neg = F.binary_cross_entropy_with_logits(
        u_logits,
        torch.zeros_like(u_logits),
        reduction="mean"
    )

    risk_p_neg = prior * F.binary_cross_entropy_with_logits(
        pos_logits,
        torch.zeros_like(pos_logits),
        reduction="mean"
    )

    est_neg_risk = risk_u_neg - risk_p_neg

    beta = dynamic_beta_boundary(
        prior=prior,
        n_samples=n_samples,
        beta_decay=beta_decay,
        round_idx=round_idx
    )

    est_neg_risk = torch.clamp(est_neg_risk, min=beta * 0.5)

    neg_risk = smooth_non_negative_risk(
        est_neg_risk=est_neg_risk,
        beta=beta,
        softness=0.1
    )

    pure_risk = risk_p_pos + neg_risk
    total_risk = pure_risk

    if global_avg_risk is not None and lam > 0:
        if torch.is_tensor(global_avg_risk):
            anchor = global_avg_risk.detach().to(
                device=outputs.device,
                dtype=outputs.dtype
            )
        else:
            anchor = torch.tensor(
                float(global_avg_risk),
                dtype=outputs.dtype,
                device=outputs.device
            )

        total_risk = total_risk + float(lam) * torch.pow(
            pure_risk - anchor,
            2
        )

    return total_risk, pure_risk.detach()


def clone_model_params(model: torch.nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone().cpu() for p in model.parameters()]


def init_control_variates(model: torch.nn.Module) -> List[torch.Tensor]:
    return [
        torch.zeros_like(p.detach(), device="cpu")
        for p in model.parameters()
    ]


def get_model_delta(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module
) -> List[torch.Tensor]:
    return [
        lp.detach().cpu() - gp.detach().cpu()
        for lp, gp in zip(local_model.parameters(), global_model.parameters())
    ]


def add_model_delta_(
    model: torch.nn.Module,
    delta: List[torch.Tensor],
    scale: float = 1.0
) -> None:
    with torch.no_grad():
        for p, d in zip(model.parameters(), delta):
            p.add_(
                d.to(device=p.device, dtype=p.dtype),
                alpha=float(scale)
            )


def average_deltas(
    deltas: List[List[torch.Tensor]]
) -> List[torch.Tensor]:
    if len(deltas) == 0:
        raise ValueError("deltas is empty.")

    avg_delta = []
    num_clients = len(deltas)

    for params in zip(*deltas):
        avg = torch.zeros_like(params[0].detach().cpu())

        for p in params:
            avg += p.detach().cpu()

        avg /= num_clients
        avg_delta.append(avg)

    return avg_delta


def average_risks(
    risks: List[Optional[torch.Tensor]]
) -> Optional[torch.Tensor]:
    valid = [
        r.detach().float().cpu()
        for r in risks
        if r is not None
    ]

    if len(valid) == 0:
        return None

    return torch.stack(valid).mean()


def local_train_scaffold_nnpu(
    global_model: torch.nn.Module,
    c_local: List[torch.Tensor],
    c_global: List[torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    prior: PriorLike,
    base_epochs: int,
    lr: float,
    batchsize: int,
    device: Union[str, torch.device],
    global_avg_risk: Optional[torch.Tensor] = None,
    lam: float = 0.01,
    beta_decay: float = 0.1,
    round_idx: int = 0,
) -> Tuple[
    torch.nn.Module,
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    Optional[torch.Tensor]
]:
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    optimizer = optim.SGD(local_model.parameters(), lr=lr)

    init_params = clone_model_params(local_model)

    num_samples = len(x)
    steps = max(1, base_epochs * math.ceil(num_samples / batchsize))

    x = x.to(device)
    t = t.to(device)

    risk_records: List[torch.Tensor] = []

    for _ in range(base_epochs):
        indices = torch.randperm(num_samples, device=device)

        for start in range(0, num_samples, batchsize):
            batch_indices = indices[start:start + batchsize]

            bx = x[batch_indices]
            bt = t[batch_indices]

            optimizer.zero_grad()

            logits = local_model(bx)

            loss, pure_risk = nnpu_loss(
                outputs=logits,
                labels=bt,
                pi_dict=prior,
                n_samples=num_samples,
                global_avg_risk=global_avg_risk,
                lam=lam,
                beta_decay=beta_decay,
                round_idx=round_idx
            )

            loss.backward()

            with torch.no_grad():
                for p, cl, cg in zip(
                    local_model.parameters(),
                    c_local,
                    c_global
                ):
                    if p.grad is not None:
                        correction = cg.to(
                            device=p.device,
                            dtype=p.dtype
                        ) - cl.to(
                            device=p.device,
                            dtype=p.dtype
                        )

                        p.grad.add_(correction)

            optimizer.step()

            risk_records.append(pure_risk.detach().cpu())

    with torch.no_grad():
        new_c_local = []

        for p_init, p_new, cl, cg in zip(
            init_params,
            local_model.parameters(),
            c_local,
            c_global
        ):
            delta_param = p_init.detach().cpu() - p_new.detach().cpu()

            new_cl = (
                cl.detach().cpu()
                - cg.detach().cpu()
                + delta_param / (steps * lr + 1e-12)
            )

            new_c_local.append(new_cl)

    delta_y = get_model_delta(
        local_model.cpu(),
        global_model.cpu()
    )

    delta_c = [
        new_cl.detach().cpu() - old_cl.detach().cpu()
        for new_cl, old_cl in zip(new_c_local, c_local)
    ]

    local_avg_risk = average_risks(risk_records)

    return (
        local_model.cpu(),
        new_c_local,
        delta_y,
        delta_c,
        local_avg_risk
    )


def scaffold_nnpu_train_round(
    global_model: torch.nn.Module,
    client_controls: Dict[int, List[torch.Tensor]],
    c_global: List[torch.Tensor],
    client_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    selected_clients: List[int],
    prior: PriorLike,
    base_epochs: int,
    lr: float,
    batchsize: int,
    device: Union[str, torch.device],
    eta_g: float = 1.0,
    global_avg_risk: Optional[torch.Tensor] = None,
    risk_momentum: float = 0.9,
    lam: float = 0.01,
    beta_decay: float = 0.1,
    round_idx: int = 0,
) -> Tuple[
    torch.nn.Module,
    Dict[int, List[torch.Tensor]],
    List[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor]
]:
    global_model.cpu()

    delta_y_list: List[List[torch.Tensor]] = []
    delta_c_list: List[List[torch.Tensor]] = []
    local_risk_list: List[torch.Tensor] = []

    for client_id in selected_clients:
        x_client, t_client = client_data[client_id]

        _, new_c_local, delta_y, delta_c, local_avg_risk = local_train_scaffold_nnpu(
            global_model=global_model,
            c_local=client_controls[client_id],
            c_global=c_global,
            x=x_client,
            t=t_client,
            prior=prior,
            base_epochs=base_epochs,
            lr=lr,
            batchsize=batchsize,
            device=device,
            global_avg_risk=global_avg_risk,
            lam=lam,
            beta_decay=beta_decay,
            round_idx=round_idx
        )

        client_controls[client_id] = new_c_local

        delta_y_list.append(delta_y)
        delta_c_list.append(delta_c)

        if local_avg_risk is not None:
            local_risk_list.append(local_avg_risk)

    avg_delta_y = average_deltas(delta_y_list)
    avg_delta_c = average_deltas(delta_c_list)

    add_model_delta_(
        model=global_model,
        delta=avg_delta_y,
        scale=eta_g
    )

    num_total_clients = len(client_controls)
    num_selected_clients = len(selected_clients)

    c_global = [
        cg.detach().cpu()
        + (num_selected_clients / num_total_clients) * dc.detach().cpu()
        for cg, dc in zip(c_global, avg_delta_c)
    ]

    current_round_risk = average_risks(local_risk_list)

    global_avg_risk = update_global_avg_risk(
        prev_avg=global_avg_risk,
        current_risk=current_round_risk,
        momentum=risk_momentum
    )

    return (
        global_model.cpu(),
        client_controls,
        c_global,
        global_avg_risk,
        current_round_risk
    )


def train_scaffold_nnpu(
    model: torch.nn.Module,
    client_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    prior: PriorLike,
    rounds: int,
    base_epochs: int,
    lr: float,
    batchsize: int,
    device: Union[str, torch.device],
    sample_ratio: float = 1.0,
    eta_g: float = 1.0,
    lam: float = 0.01,
    beta_decay: float = 0.1,
    risk_momentum: float = 0.9,
    seed: Optional[int] = None,
) -> Tuple[
    torch.nn.Module,
    Dict[int, List[torch.Tensor]],
    List[torch.Tensor],
    Optional[torch.Tensor],
    List[Optional[float]]
]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    num_clients = len(client_data)

    if num_clients <= 0:
        raise ValueError("client_data is empty.")

    global_model = copy.deepcopy(model).cpu()

    c_global = init_control_variates(global_model)

    client_controls = {
        client_id: init_control_variates(global_model)
        for client_id in client_data.keys()
    }

    sample_ratio = float(np.clip(sample_ratio, 1e-6, 1.0))
    clients_per_round = max(1, int(num_clients * sample_ratio))

    global_avg_risk: Optional[torch.Tensor] = None
    risk_history: List[Optional[float]] = []

    client_ids = list(client_data.keys())

    for r in range(rounds):
        perm = torch.randperm(num_clients)[:clients_per_round].tolist()
        selected_clients = [client_ids[i] for i in perm]

        (
            global_model,
            client_controls,
            c_global,
            global_avg_risk,
            current_round_risk
        ) = scaffold_nnpu_train_round(
            global_model=global_model,
            client_controls=client_controls,
            c_global=c_global,
            client_data=client_data,
            selected_clients=selected_clients,
            prior=prior,
            base_epochs=base_epochs,
            lr=lr,
            batchsize=batchsize,
            device=device,
            eta_g=eta_g,
            global_avg_risk=global_avg_risk,
            risk_momentum=risk_momentum,
            lam=lam,
            beta_decay=beta_decay,
            round_idx=r
        )

        risk_value = None

        if current_round_risk is not None:
            risk_value = float(current_round_risk.detach().cpu().item())

        risk_history.append(risk_value)

    return (
        global_model.cpu(),
        client_controls,
        c_global,
        global_avg_risk,
        risk_history
    )