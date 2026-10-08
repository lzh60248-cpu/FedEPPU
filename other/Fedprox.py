import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def nnpu_loss(logits, targets, pi, beta=0.0, gamma=1.0):
    """
    nnPU loss

    targets:
        正样本: 1
        未标记样本: -1
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

    if negative_risk < -beta:
        loss = -gamma * negative_risk
    else:
        loss = positive_risk + negative_risk

    pure_risk = positive_risk + negative_risk

    return loss, pure_risk


def local_train_fedprox_nnpu(
    model,
    global_model,
    x,
    t,
    pi_dict,
    epochs,
    lr,
    batchsize,
    device,
    mu=0.01,
    is_straggler=False
):
    """
    FedProx + nnPU 本地训练函数

    不包含主函数，供其他文件直接调用。

    参数:
        model: 当前客户端本地模型
        global_model: 当前轮服务器下发的全局模型
        x: 本地输入数据
        t: 本地标签，正样本 > 0，未标记样本 <= 0
        pi_dict: 类先验 pi，可以是 float 或 dict
        epochs: 本地训练轮数
        lr: 学习率
        batchsize: batch size
        device: cuda 或 cpu
        mu: FedProx proximal 系数
        is_straggler: 是否模拟 straggler

    返回:
        训练后的本地模型
    """

    model.to(device)
    global_model.to(device)

    model.train()
    global_model.eval()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    n_samples = len(x)

    if isinstance(pi_dict, dict):
        pi = list(pi_dict.values())[0]
    else:
        pi = pi_dict

    pi = float(pi)

    t = torch.where(t > 0, 1, -1)

    actual_epochs = epochs
    if is_straggler:
        actual_epochs = np.random.randint(1, epochs + 1) if epochs > 1 else 1

    for _ in range(actual_epochs):
        indices = torch.randperm(n_samples)

        for i in range(0, n_samples, batchsize):
            idx = indices[i:i + batchsize]

            bx = x[idx].to(device)
            bt = t[idx].to(device)

            if len(bx.shape) == 2:
                bx = bx.view(-1, 1, 28, 28)

            optimizer.zero_grad()

            logits = model(bx)

            loss_pu, _ = nnpu_loss(
                logits=logits,
                targets=bt,
                pi=pi,
                beta=0.0,
                gamma=1.0
            )

            proximal_term = torch.tensor(0.0, device=device)

            for w, w_global in zip(model.parameters(), global_model.parameters()):
                proximal_term += torch.sum((w - w_global.detach()) ** 2)

            loss = loss_pu + (mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()

    return model