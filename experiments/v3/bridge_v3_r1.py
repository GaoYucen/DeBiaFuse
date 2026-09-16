"""BRIDGE-V3-01: bounded innovation-discrimination experiment for Hongfu 60->30.

Scientific contract is frozen in GaoYucen/server-control:
  research-plans/code-bridge/R4_REVIEWER_DEFENSE_METHOD_BASELINE_CLOSURE.md

This file implements exactly four variants:
  V0_global_only
  V1_static_spectral
  V2_horizon_only
  V3_spectral_horizon

No test-driven architecture search is allowed.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

R2ROOT = Path('/workspace/code-bridge/repro/20260914-r2')
R3B_SRC = Path('/workspace/code-bridge/repro/20260914-r3b/source/bridge_r3b.py')
PROTOCOL = 'bridge-v3-r1-horizon-conditioned-scale-selection-v1'
VARIANTS = ('V0_global_only','V1_static_spectral','V2_horizon_only','V3_spectral_horizon')
SENSORS = ('DD-EN-09#','DD-ES-09#','DD-WN-09#','DD-WS-09#')
LOOK_BACK = 60
HORIZON = 30

spec = importlib.util.spec_from_file_location('bridge_r3b_v3_base', R3B_SRC)
r3b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3b)
r2 = r3b.r2
DeBiaFuseV2 = r3b.DeBiaFuseV2


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def historical_descriptors(low, high, component_mask):
    """Return [B,5,4] descriptors from history only.

    Features (frozen): normalized dominant frequency, spectral centroid,
    energy ratio across the 5 components, lag-1 autocorrelation.
    """
    hist = torch.cat([low.unsqueeze(-1), high], dim=-1)  # B,L,5
    b,l,k = hist.shape
    mask5 = torch.cat([torch.ones((b,1), device=hist.device, dtype=hist.dtype), component_mask], dim=1)
    x = hist - hist.mean(dim=1, keepdim=True)
    eps = 1e-6
    energy = x.square().mean(dim=1) + eps
    energy_ratio = energy / energy.sum(dim=1, keepdim=True).clamp_min(eps)

    fx = torch.fft.rfft(x, dim=1)
    power = fx.abs().square()
    if power.shape[1] > 1:
        power = power.clone(); power[:,0,:] = 0.0
    freq = torch.linspace(0.0, 1.0, power.shape[1], device=hist.device, dtype=hist.dtype).view(1,-1,1)
    denom = power.sum(dim=1).clamp_min(eps)
    centroid = (power * freq).sum(dim=1) / denom
    dom_idx = power.argmax(dim=1).to(hist.dtype)
    dom = dom_idx / max(power.shape[1]-1, 1)

    x0, x1 = x[:,:-1,:], x[:,1:,:]
    num = (x0*x1).mean(dim=1)
    den = torch.sqrt(x0.square().mean(dim=1).clamp_min(eps) * x1.square().mean(dim=1).clamp_min(eps))
    lag1 = (num/den).clamp(-1.0, 1.0)

    feat = torch.stack([dom, centroid, energy_ratio, lag1], dim=-1)
    return feat * mask5.unsqueeze(-1)


class ScaleGate(nn.Module):
    def __init__(self, mode: str, n_components: int = 5, emb_dim: int = 4, hidden: int = 16, horizon: int = HORIZON):
        super().__init__()
        assert mode in ('static_spectral','horizon_only','spectral_horizon')
        self.mode = mode; self.n_components=n_components; self.horizon=horizon
        self.comp_emb = nn.Embedding(n_components, emb_dim)
        in_dim = emb_dim + (4 if mode in ('static_spectral','spectral_horizon') else 0) + (1 if mode in ('horizon_only','spectral_horizon') else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, descriptors):
        b,k,_ = descriptors.shape
        comp = self.comp_emb(torch.arange(k, device=descriptors.device)).view(1,1,k,-1).expand(b,self.horizon,k,-1)
        pieces=[comp]
        if self.mode in ('static_spectral','spectral_horizon'):
            pieces.append(descriptors[:,None,:,:].expand(b,self.horizon,k,4))
        if self.mode in ('horizon_only','spectral_horizon'):
            h = torch.arange(1,self.horizon+1,device=descriptors.device,dtype=descriptors.dtype)/float(self.horizon)
            pieces.append(h.view(1,self.horizon,1,1).expand(b,self.horizon,k,1))
        z = torch.cat(pieces, dim=-1)
        logits = self.net(z).squeeze(-1)
        return 2.0 * torch.sigmoid(logits)  # initialized exactly at gate=1


class V3Model(nn.Module):
    def __init__(self, variant: str, look_back=LOOK_BACK, horizon=HORIZON):
        super().__init__(); self.variant=variant; self.horizon=horizon
        self.backbone = DeBiaFuseV2(4, look_back, horizon)
        if variant == 'V0_global_only': self.gate = None
        elif variant == 'V1_static_spectral': self.gate = ScaleGate('static_spectral', horizon=horizon)
        elif variant == 'V2_horizon_only': self.gate = ScaleGate('horizon_only', horizon=horizon)
        elif variant == 'V3_spectral_horizon': self.gate = ScaleGate('spectral_horizon', horizon=horizon)
        else: raise ValueError(variant)

    def forward(self, low, high, component_mask, base):
        lp,hp,_ = self.backbone(low,high,component_mask)
        comps = torch.cat([lp.unsqueeze(-1), hp], dim=-1)  # B,H,5
        if self.gate is None:
            gates = torch.ones_like(comps)
        else:
            desc = historical_descriptors(low,high,component_mask)
            gates = self.gate(desc)
        delta = (comps*gates).sum(dim=-1)
        return delta + base[:,None], gates


def train_case(args):
    seed_all(args.seed)
    device=torch.device(args.device)
    root=Path(args.output)
    cdir=root/'cases'/f'{LOOK_BACK}to{HORIZON}'/args.sensor/f'{args.variant}-seed{args.seed}'
    if (cdir/'result.json').exists():
        print('SKIP_EXISTING',cdir,flush=True); return
    cdir.mkdir(parents=True,exist_ok=True)
    parts,data=r2.load_windows(Path(args.cache)/f'{args.sensor}.npz',LOOK_BACK,HORIZON)
    tr,va,te=(parts[k] for k in ('train','val','test'))
    model=V3Model(args.variant).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loader=DataLoader(r2.tensor_data(tr),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    validation=[t.to(device) for t in r2.tensor_data(va).tensors]
    gs=max(float(np.percentile(np.abs(tr['y']-tr['base'][:,None]),75)),1e-3)
    best=float('inf'); state=None; wait=0; best_epoch=-1; hist=[]; started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; nb=0
        for data_batch in loader:
            batch=[v.to(device) for v in data_batch]
            x,y,low,high,low_y,high_y,cm,obs,base=batch
            if not bool(obs.sum()>0): continue
            opt.zero_grad(set_to_none=True)
            pred,gates=model(low,high,cm,base)
            loss=r2.masked_huber((pred-base[:,None])/gs,(y-base[:,None])/gs,obs)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            x,y,low,high,low_y,high_y,cm,obs,base=validation
            vp,_=model(low,high,cm,base)
            val=float(r2.masked_mean(torch.abs(vp-y),obs))
        hist.append({'epoch':epoch,'train_global_huber':total/max(nb,1),'val_MAE_normalized':val})
        print('EPOCH',args.sensor,args.variant,args.seed,epoch,'val',val,flush=True)
        if val < best-1e-7:
            best=val; wait=0; best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
        if wait>=args.patience: break
    if state is None: raise RuntimeError('no valid checkpoint')
    train_seconds=time.perf_counter()-started
    model.load_state_dict(state); model.eval()
    outs=[]; gate_means=[]
    test_loader=DataLoader(r2.tensor_data(te),batch_size=64,shuffle=False)
    if device.type=='cuda': torch.cuda.synchronize()
    started=time.perf_counter()
    with torch.no_grad():
        for data_batch in test_loader:
            batch=[v.to(device) for v in data_batch]
            x,y,low,high,low_y,high_y,cm,obs,base=batch
            p,g=model(low,high,cm,base); outs.append(p.cpu().numpy()); gate_means.append(g.mean(dim=0).cpu().numpy())
    if device.type=='cuda': torch.cuda.synchronize()
    pred=np.concatenate(outs); infer_seconds=time.perf_counter()-started
    gate_mean=np.mean(np.stack(gate_means),axis=0)
    torch.save({'state_dict':state,'variant':args.variant,'protocol':PROTOCOL,'seed':args.seed,
                'look_back':LOOK_BACK,'horizon':HORIZON,'best_epoch':best_epoch},cdir/'best.pt')
    np.save(cdir/'mean_gate_by_horizon_component.npy',gate_mean)
    r2.save_json(cdir/'training_history.json',hist)
    info={'parameter_count':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,
          'epochs_run':len(hist),'validation_MAE_normalized':best,'train_seconds':train_seconds,
          'inference_seconds':infer_seconds,'loss_variant':'global_only','protocol':PROTOCOL}
    args.model=args.variant
    args.look_back=LOOK_BACK
    args.horizon=HORIZON
    old_protocol=r2.PROTOCOL; r2.PROTOCOL=PROTOCOL
    try: r2.save_case(args.variant,parts,data,pred,info,args,cdir)
    finally: r2.PROTOCOL=old_protocol
    print('RESULT',args.sensor,args.variant,args.seed,json.dumps(json.loads((cdir/'result.json').read_text())['observed']),flush=True)


def summarize(root: Path, seed: int):
    rows=[]
    for p in sorted((root/'cases').rglob('result.json')):
        d=json.loads(p.read_text())
        if int(d['seed'])!=seed: continue
        row={'dataset':d['dataset'],'variant':d['model'],'seed':d['seed'],**d['observed']}; rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(root/f'summary_seed{seed}.csv',index=False)
    expected=len(SENSORS)*len(VARIANTS)
    if len(df)!=expected: raise AssertionError(f'expected {expected} cases, got {len(df)}')
    macro=df.groupby('variant',as_index=False).agg(mean_MAE=('MAE','mean'),mean_RMSE=('RMSE','mean'),sensors=('dataset','nunique'))
    macro.to_csv(root/f'summary_macro_seed{seed}.csv',index=False)
    v0=float(macro.loc[macro.variant=='V0_global_only','mean_MAE'].iloc[0]); v3=float(macro.loc[macro.variant=='V3_spectral_horizon','mean_MAE'].iloc[0])
    a=df[df.variant=='V0_global_only'].set_index('dataset')['MAE']; b=df[df.variant=='V3_spectral_horizon'].set_index('dataset')['MAE']
    common=sorted(set(a.index)&set(b.index))
    per=[{'dataset':s,'v0_mae':float(a[s]),'v3_mae':float(b[s]),'v3_rel_improvement':float(1-b[s]/a[s])} for s in common]
    wins=sum(x['v3_mae']<x['v0_mae'] for x in per); improvement=1-v3/v0
    if improvement>=0.01 and wins>=3:
        decision='GREEN'; claim='strengthened'; closed=True; nxt='Proceed to E2 modern-baseline closure with V3 frozen; do not modify architecture.'
    elif improvement>0 and wins>=2:
        decision='YELLOW'; claim='inconclusive'; closed=False; nxt='Run exactly one predefined five-seed V0-vs-V3 diagnostic; no architecture changes.'
    else:
        decision='RED'; claim='weakened'; closed=True; nxt='Stop new gate research. Do not search a second gate family; return to paper-level judgment/current global-only baseline audit.'
    partial={r['variant']:{'mean_MAE':float(r['mean_MAE']),'mean_RMSE':float(r['mean_RMSE'])} for _,r in macro.iterrows()}
    comp={'protocol':PROTOCOL,'seed':seed,'expected_cases':expected,'actual_cases':len(df),'complete':True,
          'variants':partial,'v0_mean_MAE':v0,'v3_mean_MAE':v3,'v3_rel_improvement':float(improvement),
          'v3_sensor_wins':int(wins),'sensor_comparison':per,'decision_state':decision,'claim_delta':claim,
          'evidence_slot_closed':closed,'next_action':nxt}
    save_json(root/f'comparison_seed{seed}.json',comp)
    print('=== MACRO ==='); print(macro.to_string(index=False)); print('=== COMPARISON ==='); print(json.dumps(comp,indent=2))
    return comp


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['run','summarize']); ap.add_argument('--output',required=True)
    ap.add_argument('--cache',default=str(R2ROOT/'cache')); ap.add_argument('--sensor',choices=SENSORS); ap.add_argument('--variant',choices=VARIANTS)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='cuda'); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--patience',type=int,default=15)
    args=ap.parse_args(); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    if args.action=='run':
        if not args.sensor or not args.variant: ap.error('run requires sensor and variant')
        train_case(args)
    else: summarize(Path(args.output),args.seed)

if __name__=='__main__': main()
