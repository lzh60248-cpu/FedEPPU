"""HeteRo-Select scoring adapted to binary PU client priors.

The six paper score components and dynamic-temperature sampling are retained.
Because true private class histograms are unavailable in PU learning, binary
prior distributions [pi_k, 1-pi_k] replace oracle label histograms.
"""
import math
import numpy as np

class HeteRoSelect:
    def __init__(self,num_clients,client_priors,fairness_eta=0.3,staleness_gamma=0.7,norm_alpha=0.3,tau0=2.0,tmax=20,eps=1e-8):
        if len(client_priors)!=num_clients: raise ValueError("client_priors length mismatch")
        self.num_clients=num_clients; self.fairness_eta=fairness_eta; self.staleness_gamma=staleness_gamma
        self.norm_alpha=norm_alpha; self.tau0=tau0; self.tmax=tmax; self.eps=eps
        self.participation=np.zeros(num_clients); self.last_round=np.full(num_clients,-1,dtype=np.int64)
        self.curr_loss=np.zeros(num_clients); self.prev_loss=np.zeros(num_clients); self.prev_prev_loss=np.zeros(num_clients)
        self.last_update_norm=np.zeros(num_clients)
        priors=np.clip(np.asarray(client_priors,np.float64),eps,1-eps)
        distributions=np.stack((priors,1-priors),axis=1); global_dist=distributions.mean(axis=0)
        self.base_js=np.array([self.js_divergence(p,global_dist) for p in distributions])

    def js_divergence(self,p,q):
        m=.5*(p+q)
        kl=lambda a,b: np.sum(a*np.log((a+self.eps)/(b+self.eps)))
        return .5*kl(p,m)+.5*kl(q,m)

    def update_losses(self,new_losses):
        self.prev_prev_loss=self.prev_loss.copy(); self.prev_loss=self.curr_loss.copy(); self.curr_loss=np.asarray(new_losses,np.float64)

    def update_after_training(self,selected_clients,update_norms,round_idx):
        for client,norm in zip(selected_clients,update_norms):
            self.participation[client]+=1; self.last_round[client]=round_idx; self.last_update_norm[client]=norm

    def select_clients(self,round_idx,available,m):
        if not 1<=m<=len(available): raise ValueError("invalid selection size")
        losses=np.array([self.curr_loss[c] for c in available]); lo,hi=losses.min(),losses.max()
        values={c:(self.curr_loss[c]-lo)/(hi-lo+self.eps) for c in available}; scores=[]
        for c in available:
            diversity=self.base_js[c]*(2.0*(1-.5*min(round_idx/100.,1.)))
            raw=0. if self.prev_prev_loss[c]<=0 else (self.prev_prev_loss[c]-self.prev_loss[c])/(self.prev_prev_loss[c]+self.eps)
            momentum=2./(1.+np.exp(np.clip(-5.*raw,-500,500)))-.5
            max_h=max(float(self.participation.max()),1.); fairness=(1.+self.fairness_eta*self.participation[c]/max_h)**-2-1.
            stale=self.tmax if self.last_round[c]<0 else min(round_idx-self.last_round[c],self.tmax)
            staleness=self.staleness_gamma*math.log1p(stale)
            positive=self.last_update_norm[self.last_update_norm>0]
            norm_penalty=0.
            if len(positive) and self.last_update_norm[c]>0:
                ratio=self.last_update_norm[c]/(positive.mean()+self.eps)
                norm_penalty=-self.norm_alpha*(2./(1.+np.exp(np.clip(-3.*ratio,-500,500)))-1.)
            scores.append(values[c]+diversity+momentum+fairness+staleness+norm_penalty)
        scores=np.asarray(scores); tau=max(self.tau0*(1-.5*min(round_idx/100.,1.)),1e-6)
        logits=scores/tau; probs=np.exp(logits-logits.max()); probs/=probs.sum()
        chosen=np.random.choice(available,size=m,replace=False,p=probs)
        return chosen.tolist(),scores,probs
