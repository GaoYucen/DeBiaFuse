"""BRIDGE-V3-01 repaired protocol: exact frozen-control replay + gate-only novelty test.

The verified R3B global_only checkpoint is the control. New scale-selection gates are
trained on top of the frozen backbone only. This removes stochastic backbone retraining
as a confound and isolates the proposed mechanism.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

R2ROOT=Path('/workspace/code-bridge/repro/20260914-r2')
R3BROOT=Path('/workspace/code-bridge/repro/20260914-r3b')
R3B_SRC=R3BROOT/'source'/'bridge_r3b.py'
PROTOCOL='bridge-v3-r1-frozen-backbone-horizon-scale-selection-v1'
VARIANTS=('V0_frozen_global_only','V1_static_spectral','V2_horizon_only','V3_spectral_horizon')
SENSORS=('DD-EN-09#','DD-ES-09#','DD-WN-09#','DD-WS-09#')
LOOK_BACK=60; HORIZON=30

spec=importlib.util.spec_from_file_location('bridge_r3b_frozen_base',R3B_SRC)
r3b=importlib.util.module_from_spec(spec); spec.loader.exec_module(r3b)
r2=r3b.r2; DeBiaFuseV2=r3b.DeBiaFuseV2

def seed_all(s):
    r2.seed_all(s)

def save_json(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)); t.replace(p)

def descriptors(low,high,component_mask):
    hist=torch.cat([low.unsqueeze(-1),high],dim=-1); b,l,k=hist.shape
    mask5=torch.cat([torch.ones((b,1),device=hist.device,dtype=hist.dtype),component_mask],dim=1)
    x=hist-hist.mean(dim=1,keepdim=True); eps=1e-6
    energy=x.square().mean(dim=1)+eps; er=energy/energy.sum(dim=1,keepdim=True).clamp_min(eps)
    fx=torch.fft.rfft(x,dim=1); power=fx.abs().square()
    if power.shape[1]>1:
        power=power.clone(); power[:,0,:]=0
    f=torch.linspace(0.,1.,power.shape[1],device=x.device,dtype=x.dtype).view(1,-1,1)
    den=power.sum(dim=1).clamp_min(eps); centroid=(power*f).sum(dim=1)/den
    dom=power.argmax(dim=1).to(x.dtype)/max(power.shape[1]-1,1)
    x0,x1=x[:,:-1,:],x[:,1:,:]; num=(x0*x1).mean(dim=1)
    ac=num/torch.sqrt(x0.square().mean(dim=1).clamp_min(eps)*x1.square().mean(dim=1).clamp_min(eps))
    feat=torch.stack([dom,centroid,er,ac.clamp(-1,1)],dim=-1)
    return feat*mask5.unsqueeze(-1)

class ScaleGate(nn.Module):
    def __init__(self,mode,n_components=5,emb_dim=4,hidden=16,horizon=HORIZON):
        super().__init__(); self.mode=mode; self.horizon=horizon
        self.comp_emb=nn.Embedding(n_components,emb_dim)
        din=emb_dim+(4 if mode in ('static_spectral','spectral_horizon') else 0)+(1 if mode in ('horizon_only','spectral_horizon') else 0)
        self.net=nn.Sequential(nn.Linear(din,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,d):
        b,k,_=d.shape
        ce=self.comp_emb(torch.arange(k,device=d.device)).view(1,1,k,-1).expand(b,self.horizon,k,-1)
        z=[ce]
        if self.mode in ('static_spectral','spectral_horizon'): z.append(d[:,None].expand(b,self.horizon,k,4))
        if self.mode in ('horizon_only','spectral_horizon'):
            h=torch.arange(1,self.horizon+1,device=d.device,dtype=d.dtype)/float(self.horizon)
            z.append(h.view(1,self.horizon,1,1).expand(b,self.horizon,k,1))
        return 2*torch.sigmoid(self.net(torch.cat(z,dim=-1)).squeeze(-1))

class FrozenGateModel(nn.Module):
    def __init__(self,sensor,variant):
        super().__init__(); self.variant=variant
        self.backbone=DeBiaFuseV2(4,LOOK_BACK,HORIZON)
        ck=R3BROOT/'cases'/f'{LOOK_BACK}to{HORIZON}'/sensor/'global_only-seed42'/'best.pt'
        state=torch.load(ck,map_location='cpu',weights_only=False)['state_dict']; self.backbone.load_state_dict(state)
        for p in self.backbone.parameters(): p.requires_grad_(False)
        if variant=='V0_frozen_global_only': self.gate=None
        elif variant=='V1_static_spectral': self.gate=ScaleGate('static_spectral')
        elif variant=='V2_horizon_only': self.gate=ScaleGate('horizon_only')
        elif variant=='V3_spectral_horizon': self.gate=ScaleGate('spectral_horizon')
        else: raise ValueError(variant)
    def components(self,low,high,cm):
        self.backbone.eval()
        with torch.no_grad():
            lp,hp,delta=self.backbone(low,high,cm)
            comps=torch.cat([lp.unsqueeze(-1),hp],dim=-1)
        return comps
    def forward(self,low,high,cm,base):
        comps=self.components(low,high,cm)
        if self.gate is None: g=torch.ones_like(comps)
        else: g=self.gate(descriptors(low,high,cm))
        return base[:,None]+(comps*g).sum(-1),g

def evaluate(model,part,device):
    model.eval(); outs=[]; gs=[]
    with torch.no_grad():
        for batch in DataLoader(r2.tensor_data(part),batch_size=64,shuffle=False):
            x,y,low,high,ly,hy,cm,obs,base=[v.to(device) for v in batch]
            p,g=model(low,high,cm,base); outs.append(p.cpu().numpy()); gs.append(g.cpu().numpy())
    return np.concatenate(outs),np.concatenate(gs)

def run_case(args):
    seed_all(args.seed); device=torch.device(args.device); root=Path(args.output)
    cdir=root/'cases'/f'{LOOK_BACK}to{HORIZON}'/args.sensor/f'{args.variant}-seed{args.seed}'
    if (cdir/'result.json').exists(): print('SKIP',cdir); return
    cdir.mkdir(parents=True,exist_ok=True)
    parts,data=r2.load_windows(Path(args.cache)/f'{args.sensor}.npz',LOOK_BACK,HORIZON)
    tr,va,te=(parts[k] for k in ('train','val','test'))
    model=FrozenGateModel(args.sensor,args.variant).to(device)
    best_epoch=0; best=None; hist=[]; train_seconds=0.
    if model.gate is not None:
        opt=torch.optim.Adam(model.gate.parameters(),lr=1e-3)
        loader=DataLoader(r2.tensor_data(tr),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        val=[t.to(device) for t in r2.tensor_data(va).tensors]
        gs=max(float(np.percentile(np.abs(tr['y']-tr['base'][:,None]),75)),1e-3)
        wait=0; score=float('inf'); t0=time.perf_counter()
        for ep in range(1,args.epochs+1):
            model.gate.train(); total=0.; nb=0
            for bb in loader:
                x,y,low,high,ly,hy,cm,obs,base=[v.to(device) for v in bb]
                opt.zero_grad(set_to_none=True); pred,_=model(low,high,cm,base)
                loss=r2.masked_huber((pred-base[:,None])/gs,(y-base[:,None])/gs,obs)
                loss.backward(); opt.step(); total+=float(loss.detach()); nb+=1
            model.eval()
            with torch.no_grad():
                x,y,low,high,ly,hy,cm,obs,base=val; vp,_=model(low,high,cm,base); v=float(r2.masked_mean(torch.abs(vp-y),obs))
            hist.append({'epoch':ep,'train_loss':total/max(nb,1),'val_MAE_normalized':v})
            if v<score:
                score=v; wait=0; best_epoch=ep; best={k:v.detach().cpu().clone() for k,v in model.gate.state_dict().items()}
            else: wait+=1
            if wait>=args.patience: break
        train_seconds=time.perf_counter()-t0; model.gate.load_state_dict(best)
    pred,gates=evaluate(model,te,device)
    # Exact V0 checkpoint replay guard against original stored predictions.
    replay_max=None
    if args.variant=='V0_frozen_global_only':
        old=R3BROOT/'cases'/f'{LOOK_BACK}to{HORIZON}'/args.sensor/'global_only-seed42'/'predictions.npz'
        oldp=np.load(old,allow_pickle=False)['y_pred']; width=float(data['scale_hi']-data['scale_lo']); lo=float(data['scale_lo'])
        replay_max=float(np.max(np.abs((pred*width+lo)-oldp)))
        if replay_max>1e-3: raise AssertionError(f'checkpoint replay mismatch {args.sensor} {replay_max}')
    info={'parameter_count_total':sum(p.numel() for p in model.parameters()),'parameter_count_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'best_epoch':best_epoch,'epochs_run':len(hist),'train_seconds':train_seconds,'protocol':PROTOCOL,'checkpoint_replay_max_abs_raw':replay_max}
    torch.save({'backbone_frozen':True,'gate_state':None if model.gate is None else model.gate.state_dict(),'variant':args.variant,'protocol':PROTOCOL,'seed':args.seed},cdir/'best.pt')
    r2.save_json(cdir/'training_history.json',hist); np.save(cdir/'mean_gate_by_horizon_component.npy',gates.mean(0))
    args.model=args.variant; args.look_back=LOOK_BACK; args.horizon=HORIZON
    oldp=r2.PROTOCOL; r2.PROTOCOL=PROTOCOL
    try: r2.save_case(args.variant,parts,data,pred,info,args,cdir)
    finally: r2.PROTOCOL=oldp

def summarize(root,seed):
    rows=[]
    for p in sorted((Path(root)/'cases').rglob('result.json')):
        d=json.loads(p.read_text());
        if int(d['seed'])==seed: rows.append({'dataset':d['dataset'],'variant':d['model'],**d['observed']})
    df=pd.DataFrame(rows); df.to_csv(Path(root)/'summary_seed42.csv',index=False)
    m=df.groupby('variant',as_index=False).agg(mean_MAE=('MAE','mean'),mean_RMSE=('RMSE','mean'),sensors=('dataset','nunique')); m.to_csv(Path(root)/'summary_macro_seed42.csv',index=False)
    v0=float(m.loc[m.variant=='V0_frozen_global_only','mean_MAE'].iloc[0]); v3=float(m.loc[m.variant=='V3_spectral_horizon','mean_MAE'].iloc[0])
    a=df[df.variant=='V0_frozen_global_only'].set_index('dataset').MAE; b=df[df.variant=='V3_spectral_horizon'].set_index('dataset').MAE
    per=[{'dataset':s,'v0_mae':float(a[s]),'v3_mae':float(b[s]),'v3_rel_improvement':float(1-b[s]/a[s])} for s in sorted(a.index)]
    wins=sum(x['v3_mae']<x['v0_mae'] for x in per); imp=1-v3/v0
    if imp>=.01 and wins>=3: dec='GREEN'; claim='strengthened'; closed=True
    elif imp>0 and wins>=2: dec='YELLOW'; claim='inconclusive'; closed=False
    else: dec='RED'; claim='weakened'; closed=True
    out={'protocol':PROTOCOL,'v0_mean_MAE':v0,'v3_mean_MAE':v3,'v3_rel_improvement':imp,'v3_sensor_wins':wins,'sensor_comparison':per,'decision_state':dec,'claim_delta':claim,'evidence_slot_closed':closed,'variants':{r.variant:{'mean_MAE':float(r.mean_MAE),'mean_RMSE':float(r.mean_RMSE)} for _,r in m.iterrows()}}
    save_json(Path(root)/'comparison_seed42.json',out); print(json.dumps(out,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['run','summarize']); ap.add_argument('--output',required=True); ap.add_argument('--cache',default=str(R2ROOT/'cache')); ap.add_argument('--sensor',choices=SENSORS); ap.add_argument('--variant',choices=VARIANTS); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='cuda'); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--patience',type=int,default=15); a=ap.parse_args()
    if a.action=='run': run_case(a)
    else: summarize(a.output,a.seed)
if __name__=='__main__': main()
