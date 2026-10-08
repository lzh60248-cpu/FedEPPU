"""Standalone standard non-negative PU risk for binary comparison methods."""
import torch
import torch.nn.functional as F

def nnpu_loss(logits,labels,prior):
    logits=logits.reshape(-1); labels=labels.reshape(-1); prior=float(prior)
    if not 0.0<prior<1.0: raise ValueError("nnPU prior must be in (0,1)")
    pos=labels==1; unl=labels==-1
    if not torch.any(pos) and not torch.any(unl): return logits.sum()*0.0
    if not torch.any(pos): return F.binary_cross_entropy_with_logits(logits[unl],torch.zeros_like(logits[unl]))
    if not torch.any(unl): return prior*F.binary_cross_entropy_with_logits(logits[pos],torch.ones_like(logits[pos]))
    positive=prior*F.binary_cross_entropy_with_logits(logits[pos],torch.ones_like(logits[pos]))
    positive_as_negative=prior*F.binary_cross_entropy_with_logits(logits[pos],torch.zeros_like(logits[pos]))
    unlabeled_as_negative=F.binary_cross_entropy_with_logits(logits[unl],torch.zeros_like(logits[unl]))
    negative=unlabeled_as_negative-positive_as_negative
    # Kiryo et al. nnPU correction with beta=0 and gamma=1.
    return -negative if negative.detach().item()<0.0 else positive+negative
