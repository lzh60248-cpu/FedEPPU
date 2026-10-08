"""Run one baseline with the same disjoint proxy protocol as FedEPPU."""
import argparse, hashlib, json, random
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .baseline_runtime import BaselineCNN, device
from .centralized import train_centralized_nnpu, train_centralized_pn
from .data_protocol import load_protocol, pooled_pu_prior
from .fedavg_nnpu import local_train_fedavg
from .fedprox_nnpu import local_train_fedprox
from .fedpu import local_train_fedpu
from .local_nnpu import train_local_models
from .mrfl_nnpu import MRFLState, local_train_mrfl_nnpu, pu_detection_vector
from .nnpu_common import average_models, clone_model
from .proxy_protocol import ProxySelector, calibrated_threshold, ensemble_scores, metrics, model_scores, split_proxy
from .scaffold_nnpu import local_train_scaffold, update_server_control, zero_control

METHODS=("centralized_pn","centralized_nnpu","local_nnpu","fedavg_nnpu","fedprox_nnpu","scaffold_nnpu","fedpu","mrfl_nnpu")
def args():
    p=argparse.ArgumentParser(); p.add_argument("--method",required=True,choices=METHODS);p.add_argument("--dataset",default="cifar10")
    p.add_argument("--rounds",type=int,default=200);p.add_argument("--n_clients",type=int,default=3);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--alpha",type=float,default=.1);p.add_argument("--batch_size",type=int,default=256);p.add_argument("--local_epochs",type=int,default=6)
    p.add_argument("--max_steps_per_epoch",type=int,default=15);p.add_argument("--lr",type=float,default=.0015);p.add_argument("--mu",type=float,default=.01)
    p.add_argument("--fedpu_lr",type=float,default=.01);p.add_argument("--fedpu_weight",type=float,default=1.);p.add_argument("--mrfl_warmup_rounds",type=int,default=20)
    p.add_argument("--output",required=True); return p.parse_args()
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=True
def tensor(x,dtype):return torch.from_numpy(np.asarray(x,dtype=dtype)).to(device)
def main():
    a=args();seed_all(a.seed);p=load_protocol(a.dataset,a.n_clients,a.alpha)
    clients=[(tensor(c.x,np.float32),tensor(c.observed_y,np.int32)) for c in p.clients];truths=[tensor(c.true_y,np.int32) for c in p.clients]
    priors=np.clip(np.asarray([c.prior for c in p.clients]),np.finfo(float).eps,1-np.finfo(float).eps);counts=np.asarray([len(c.x) for c in p.clients],float)
    xt,yt=tensor(p.test_x,np.float32),tensor(p.test_y,np.int32);xp,yp=tensor(p.proxy_x,np.float32),tensor(p.proxy_y,np.int32)
    (xs,ys),(xc,yc)=split_proxy(xp,yp); ys_np=ys.cpu().numpy();yc_np=yc.cpu().numpy();yt_np=yt.cpu().numpy()
    two=a.method=="fedpu";model=BaselineCNN(p.input_channels,2 if two else 1).to(device);selector=ProxySelector();history=[]
    def state_of(obj):return [deepcopy(m.state_dict()) for m in obj] if isinstance(obj,list) else deepcopy(obj.state_dict())
    def scores_of(obj,x):return ensemble_scores(obj,counts,x) if isinstance(obj,list) else model_scores(obj,x,two)
    def observe(r,obj):
        test_scores=scores_of(obj,xt);raw=metrics(yt_np,test_scores,0.0);proxy_auc=selector.consider(r,a.rounds,scores_of(obj,xs),ys_np,state_of(obj));history.append({"round":r+1,**raw,"proxy_auc":proxy_auc});print(f"Round {r+1:03d} | AUC:{raw['auc']:.4f} BAcc:{raw['bacc']:.4f} proxyAUC:{proxy_auc:.4f}",flush=True)
    common=dict(epochs=a.local_epochs,lr=a.lr,batch_size=a.batch_size,max_steps_per_epoch=a.max_steps_per_epoch)
    if a.method in ("centralized_pn","centralized_nnpu"):
        px=torch.cat([x for x,_ in clients]); steps=a.n_clients*a.max_steps_per_epoch
        if a.method=="centralized_pn":train_centralized_pn(model,px,torch.cat(truths),a.rounds,a.local_epochs,a.lr,a.batch_size,steps,observe)
        else:train_centralized_nnpu(model,px,torch.cat([y for _,y in clients]),pooled_pu_prior(p.clients),a.rounds,a.local_epochs,a.lr,a.batch_size,steps,observe)
        current=model
    elif a.method=="local_nnpu":
        current=[clone_model(model) for _ in clients];train_local_models(current,clients,priors,a.rounds,a.local_epochs,a.lr,a.batch_size,a.max_steps_per_epoch,observe)
    else:
        sc=zero_control(model) if a.method=="scaffold_nnpu" else None;cc=[zero_control(model) for _ in clients] if sc is not None else None
        mrfl=MRFLState(a.n_clients,a.mrfl_warmup_rounds,a.seed) if a.method=="mrfl_nnpu" else None
        for r in range(a.rounds):
            locals=[];diags=[];olds=[];news=[];ref=clone_model(model)
            for i,(x,y) in enumerate(clients):
                local=clone_model(model)
                if a.method=="fedavg_nnpu":local,d=local_train_fedavg(local,x,y,priors[i],**common)
                elif a.method=="fedprox_nnpu":local,d=local_train_fedprox(local,ref,x,y,priors[i],mu=a.mu,**common)
                elif a.method=="fedpu":
                    kw=dict(common);kw["lr"]=a.fedpu_lr*(.995**r);local,d=local_train_fedpu(local,x,y,priors[i],pu_weight=a.fedpu_weight,seed=a.seed*100000+r*a.n_clients+i,**kw)
                elif a.method=="scaffold_nnpu":
                    old={k:v.clone() for k,v in cc[i].items()};local,new,d=local_train_scaffold(local,ref,x,y,priors[i],cc[i],sc,**common);olds.append(old);news.append(new);cc[i]=new
                else:local,d=local_train_mrfl_nnpu(local,x,y,priors[i],is_noisy=bool(mrfl.noisy[i]),seed=a.seed+r*a.n_clients+i,**common)
                locals.append(local);diags.append(d)
            weights=np.asarray([d["aggregation_mass"] for d in diags]) if mrfl else counts;average_models(model,locals,weights)
            if sc is not None:update_server_control(sc,olds,news,a.n_clients)
            if mrfl is not None:mrfl.update_client_division(np.stack([pu_detection_vector(model,x,y) for x,y in clients]),r)
            observe(r,model)
        current=model
    final_zero=metrics(yt_np,scores_of(current,xt),0.0)
    if isinstance(current,list):
        for m,s in zip(current,selector.state):m.load_state_dict(s)
    else:current.load_state_dict(selector.state)
    threshold=calibrated_threshold(yc_np,scores_of(current,xc));selected=metrics(yt_np,scores_of(current,xt),threshold)
    fingerprint=hashlib.sha256(np.ascontiguousarray(p.proxy_x).tobytes()+np.ascontiguousarray(p.proxy_y).tobytes()).hexdigest()
    result={"method":a.method,"dataset":a.dataset,"seed":a.seed,"protocol":"matched_disjoint_proxy","selection_interval":5,"auc_tolerance":.001,
            "proxy_fingerprint":fingerprint,"client_priors":priors.tolist(),"client_sample_counts":counts.astype(int).tolist(),
            "proxy_selection_n":len(ys),"proxy_calibration_n":len(yc),"round200_zero":final_zero,
            "selected_round":selector.selected_round,"selected_proxy_auc":selector.selected_auc,"best_proxy_auc":selector.best_auc,"selected_calibrated":selected,"history":history}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({"result":result,"model":state_of(current)},out);out.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"Selected round: {selector.selected_round}\nRound-200 AUC: {final_zero['auc']:.6f}\nSelected AUC: {selected['auc']:.6f}\nCalibrated BAcc: {selected['bacc']:.6f}\nCalibrated F1: {selected['f1']:.6f}\nOutput -> {out}")
if __name__=="__main__":main()
