"""Method-neutral model, preprocessing and evaluation for nnPU baselines.

This module intentionally contains no EB correction, reliability weighting,
prior consistency, EMA, proxy calibration, rollback or adaptive compute rule.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, recall_score,
    roc_auc_score,
)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineCNN(nn.Module):
    """Common-capacity backbone used by every method for a fair comparison."""
    def __init__(self,input_channels=3,num_outputs=1):
        super().__init__(); c1,c2,c3=64,128,256
        self.features=nn.Sequential(
            nn.Conv2d(input_channels,c1,3,padding=1,bias=False),nn.GroupNorm(8,c1),nn.ReLU(True),
            nn.Conv2d(c1,c1,3,padding=1,bias=False),nn.GroupNorm(8,c1),nn.ReLU(True),nn.MaxPool2d(2),nn.Dropout2d(.10),
            nn.Conv2d(c1,c2,3,padding=1,bias=False),nn.GroupNorm(8,c2),nn.ReLU(True),
            nn.Conv2d(c2,c2,3,padding=1,bias=False),nn.GroupNorm(8,c2),nn.ReLU(True),nn.MaxPool2d(2),nn.Dropout2d(.15),
            nn.Conv2d(c2,c3,3,padding=1,bias=False),nn.GroupNorm(8,c3),nn.ReLU(True),
            nn.Conv2d(c3,c3,3,padding=1,bias=False),nn.GroupNorm(8,c3),nn.ReLU(True),nn.AdaptiveAvgPool2d(1))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(c3,128),nn.ReLU(True),nn.Dropout(.30),nn.Linear(128,num_outputs))
        self.apply(self._init)
    @staticmethod
    def _init(module):
        if isinstance(module,nn.Conv2d): nn.init.kaiming_normal_(module.weight,nonlinearity="relu")
        elif isinstance(module,nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
    def forward(self,x): return self.classifier(self.features(x))

def parse_train_data(records):
    xs,labels=zip(*records)
    x=torch.from_numpy(np.stack(xs).astype(np.float32)).to(device)
    t=torch.from_numpy(np.asarray(labels,dtype=np.int32)).to(device)
    return x,t

def infer_logits(model,x,batch_size=512):
    model.eval(); parts=[]
    with torch.no_grad():
        for start in range(0,len(x),batch_size): parts.append(model(x[start:start+batch_size]).reshape(-1).cpu())
    return torch.cat(parts).numpy()

def evaluate_fixed(model,x,t,threshold=0.0):
    logits=infer_logits(model,x); labels=t.reshape(-1).detach().cpu().numpy(); pred=np.where(logits>=threshold,1,-1)
    acc=accuracy_score(labels,pred); rec=recall_score(labels,pred,pos_label=1,zero_division=0); f1=f1_score(labels,pred,pos_label=1,zero_division=0)
    try: auc=roc_auc_score(labels,logits)
    except ValueError: auc=.5
    return float(acc),float(rec),float(auc),float(f1),float(threshold)

def evaluate_two_logit(model,x,t):
    model.eval(); parts=[]
    with torch.no_grad():
        for start in range(0,len(x),512): parts.append(model(x[start:start+512]).cpu())
    logits=torch.cat(parts).numpy(); labels=t.reshape(-1).detach().cpu().numpy(); pred=np.where(np.argmax(logits,axis=1)==1,1,-1)
    shifted=logits[:,1]-logits[:,0]
    acc=accuracy_score(labels,pred); rec=recall_score(labels,pred,pos_label=1,zero_division=0); f1=f1_score(labels,pred,pos_label=1,zero_division=0)
    try: auc=roc_auc_score(labels,shifted)
    except ValueError: auc=.5
    return float(acc),float(rec),float(auc),float(f1),0.0


def evaluate_metrics(model, x, t, two_logit=False, threshold=0.0):
    """Return all table metrics in one inference pass."""
    model.eval(); parts=[]
    with torch.no_grad():
        for start in range(0,len(x),512):
            parts.append(model(x[start:start+512]).cpu())
    output=torch.cat(parts).numpy()
    labels=t.reshape(-1).detach().cpu().numpy()
    if two_logit:
        pred=np.where(np.argmax(output,axis=1)==1,1,-1)
        score=output[:,1]-output[:,0]
        used_threshold=0.0
    else:
        score=output.reshape(-1)
        pred=np.where(score>=threshold,1,-1)
        used_threshold=float(threshold)
    try: auc=roc_auc_score(labels,score)
    except ValueError: auc=.5
    return {
        "acc":float(accuracy_score(labels,pred)),
        "bacc":float(balanced_accuracy_score(labels,pred)),
        "recall":float(recall_score(labels,pred,pos_label=1,zero_division=0)),
        "f1":float(f1_score(labels,pred,pos_label=1,zero_division=0)),
        "auc":float(auc),
        "threshold":used_threshold,
    }
