"""FedAvg with a standard binary nnPU local objective."""
import torch
from .nnpu_common import train_epoch

def local_train_fedavg(model,x,t,prior,epochs,lr,batch_size,max_steps_per_epoch=None):
    optimizer=torch.optim.SGD(model.parameters(),lr=lr)
    losses=[]; updates=0
    for _ in range(epochs):
        loss,count=train_epoch(model,x,t,prior,optimizer,batch_size,
                               max_steps_per_epoch=max_steps_per_epoch)
        losses.append(loss); updates+=count
    return model,{"loss":sum(losses)/max(len(losses),1),"updates":updates}
