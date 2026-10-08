"""FedProx with a standard binary nnPU local objective."""
import torch
from .nnpu_common import train_epoch

def local_train_fedprox(model,global_model,x,t,prior,epochs,lr,batch_size,mu=0.01,
                        max_steps_per_epoch=None):
    if mu<0: raise ValueError("mu must be non-negative")
    global_model.eval()
    for parameter in global_model.parameters(): parameter.requires_grad_(False)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr)
    losses=[]; updates=0
    for _ in range(epochs):
        loss,count=train_epoch(model,x,t,prior,optimizer,batch_size,global_model,mu,
                               max_steps_per_epoch=max_steps_per_epoch)
        losses.append(loss); updates+=count
    return model,{"loss":sum(losses)/max(len(losses),1),"updates":updates}
