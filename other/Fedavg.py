import copy
import torch
import torch.optim as optim
import torch.nn.functional as F


def nnpu_loss(logits, targets, pi, beta=0.0, gamma=1.0):
    """
    nnPU loss

    targets:
        正样本: 1
        未标记样本: -1

    pi:
        正类先验概率
    """

    if isinstance(pi, dict):
        pi = list(pi.values())[0]

    pi = float(pi)

    targets = targets.view(-1)

    positive_mask = targets == 1
    unlabeled_mask = targets == -1

    if logits.dim() == 2 and logits.size(1) == 2:
        scores = logits[:, 1]
    else:
        scores = logits.view(-1)

    loss_pos = F.softplus(-scores)
    loss_neg = F.softplus(scores)

    if positive_mask.sum() > 0:
        positive_risk_positive = loss_pos[positive_mask].mean()
        positive_risk_negative = loss_neg[positive_mask].mean()
    else:
        positive_risk_positive = torch.tensor(0.0, device=logits.device)
        positive_risk_negative = torch.tensor(0.0, device=logits.device)

    if unlabeled_mask.sum() > 0:
        unlabeled_risk_negative = loss_neg[unlabeled_mask].mean()
    else:
        unlabeled_risk_negative = torch.tensor(0.0, device=logits.device)

    positive_risk = pi * positive_risk_positive
    negative_risk = unlabeled_risk_negative - pi * positive_risk_negative

    pure_risk = positive_risk + negative_risk

    if negative_risk < -beta:
        loss = -gamma * negative_risk
    else:
        loss = pure_risk

    return loss, pure_risk


def local_train_nnpu(model, x, t, pi, local_epochs, lr, batchsize, device):
    """
    FedAvg 客户端本地训练：使用 nnPU loss
    """

    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    if isinstance(pi, dict):
        pi = list(pi.values())[0]

    pi = float(pi)

    t = torch.where(t > 0, 1, -1)

    n_samples = len(x)

    for epoch in range(local_epochs):
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batchsize):
            idx = indices[start:start + batchsize]

            bx = x[idx].to(device)
            bt = t[idx].to(device)

            if len(bx.shape) == 2:
                bx = bx.view(-1, 1, 28, 28)

            optimizer.zero_grad()

            logits = model(bx)

            loss, pure_risk = nnpu_loss(
                logits=logits,
                targets=bt,
                pi=pi,
                beta=0.0,
                gamma=1.0
            )

            loss.backward()
            optimizer.step()

    return model.cpu()


def fedavg_aggregate(local_models, local_sample_nums):
    """
    FedAvg 服务端加权平均聚合

    local_models:
        每个客户端本地训练后的模型列表

    local_sample_nums:
        每个客户端样本数量列表
    """

    global_model = copy.deepcopy(local_models[0])

    total_samples = sum(local_sample_nums)

    global_state_dict = global_model.state_dict()

    for key in global_state_dict.keys():
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

        for local_model, n_k in zip(local_models, local_sample_nums):
            local_state_dict = local_model.state_dict()
            weight = n_k / total_samples
            global_state_dict[key] += weight * local_state_dict[key]

    global_model.load_state_dict(global_state_dict)

    return global_model


def fedavg_nnpu_train(
    global_model,
    clients_data,
    pi_dict,
    communication_rounds,
    client_fraction,
    local_epochs,
    lr,
    batchsize,
    device
):
    """
    完整 FedAvg + nnPU 训练流程

    参数说明：

    global_model:
        全局模型

    clients_data:
        客户端数据列表，格式为：
        [
            (x_client_1, t_client_1),
            (x_client_2, t_client_2),
            ...
        ]

    pi_dict:
        正类先验概率，可以是 float，也可以是 dict

    communication_rounds:
        通信轮数

    client_fraction:
        每轮参与训练的客户端比例 C

    local_epochs:
        每个客户端本地训练轮数 E

    lr:
        本地学习率

    batchsize:
        本地 batch size

    device:
        "cuda" 或 "cpu"
    """

    num_clients = len(clients_data)

    for round_idx in range(communication_rounds):
        m = max(int(client_fraction * num_clients), 1)

        selected_clients = torch.randperm(num_clients)[:m].tolist()

        local_models = []
        local_sample_nums = []

        for client_id in selected_clients:
            x_client, t_client = clients_data[client_id]

            local_model = local_train_nnpu(
                model=global_model,
                x=x_client,
                t=t_client,
                pi=pi_dict,
                local_epochs=local_epochs,
                lr=lr,
                batchsize=batchsize,
                device=device
            )

            local_models.append(local_model)
            local_sample_nums.append(len(x_client))

        global_model = fedavg_aggregate(
            local_models=local_models,
            local_sample_nums=local_sample_nums
        )

        print(
            f"Round [{round_idx + 1}/{communication_rounds}] "
            f"finished, selected clients: {selected_clients}"
        )

    return global_model