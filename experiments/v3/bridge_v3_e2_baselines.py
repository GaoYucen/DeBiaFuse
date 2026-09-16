"""BRIDGE-V3-02 reviewer-relevant modern baselines on frozen Hongfu protocol.

Generic forecasting models receive the matched 149-day raw history budget and predict 30 days.
EMD-LSTM uses the same leakage-safe causal decomposition as R3B, but a conventional LSTM head.
No test-based hyperparameter search is performed.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RAW_HISTORY=149; NOMINAL_LOOK_BACK=60; HORIZON=30
SENSORS=('DD-EN-09#','DD-ES-09#','DD-WN-09#','DD-WS-09#')
MODELS=('autoformer','fedformer','itransformer','emd_lstm')
TSLIB_SHA='4e938a1767106324dd753b2a44832bf870a0252e'
PROTOCOL='bridge-v3-e2-modern-reviewer-baselines-v1'
R2CACHE=Path('/workspace/code-bridge/repro/20260914-r2/cache')
R3B_SRC=Path('/workspace/code-bridge/repro/20260914-r3b/source/bridge_r3b.py')

def save_json(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)); t.replace(p)
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def starts_for_split(n):
    a,b=int(n*.7),int(n*.7)+int(n*.1)
    return {k:np.arange(max(RAW_HISTORY,l),r-HORIZON+1,dtype=np.int64) for k,l,r in [('train',0,a),('val',a,b),('test',b,n)]}
def raw_windows(cache):
    with np.load(cache,allow_pickle=False) as zf: d={k:zf[k].copy() for k in zf.files}
    z,obs=d['z'],d['observed']; parts={}
    for k,st in starts_for_split(len(z)).items():
        xi=st[:,None]-np.arange(RAW_HISTORY,0,-1)[None,:]; yi=st[:,None]+np.arange(HORIZON)[None,:]
        parts[k]={'x':z[xi].astype('float32'),'y':z[yi].astype('float32'),'observed':obs[yi].astype('float32'),'starts':st,'target_dates':d['dates'][yi],'base':z[st-1].astype('float32')}
    return parts,d

def masked_mean(x,m):
    m=torch.broadcast_to(m,x.shape); return (x*m).sum()/m.sum().clamp_min(1)
def metrics(y,p,obs,ref):
    m=obs.astype(bool); truth=y[m].astype(float); pred=p[m].astype(float); base=ref[m].astype(float)
    mae=float(np.abs(truth-pred).mean()); rmse=float(np.sqrt(np.square(truth-pred).mean())); pmae=float(np.abs(truth-base).mean())
    return {'n':int(m.sum()),'MAE':mae,'RMSE':rmse,'Skill_MAE':float(1-mae/pmae) if pmae>0 else None,'Persistence_MAE':pmae}

def common_cfg(kind):
    # One frozen moderate-capacity config for small-data Hongfu; no model-specific test tuning.
    return dict(task_name='long_term_forecast',seq_len=RAW_HISTORY,label_len=RAW_HISTORY//2,pred_len=HORIZON,
                enc_in=1,dec_in=1,c_out=1,d_model=64,n_heads=4,e_layers=2,d_layers=1,d_ff=128,
                factor=3,moving_avg=25,dropout=.1,embed='fixed',freq='d',activation='gelu',
                learning_rate=1e-3,batch_size=32,source='TSLib pinned reviewer-baseline preset; matched 149-day raw-history budget')

def clean_modules():
    for k in list(sys.modules):
        if k=='models' or k.startswith('models.') or k=='layers' or k.startswith('layers.'): del sys.modules[k]
def build_tslib(kind,vendor):
    clean_modules(); root=Path(vendor)/'tslib'; sys.path.insert(0,str(root)); cfg=common_cfg(kind)
    try:
        if kind=='autoformer': from models.Autoformer import Model
        elif kind=='fedformer': from models.FEDformer import Model
        elif kind=='itransformer': from models.iTransformer import Model
        else: raise ValueError(kind)
        c=SimpleNamespace(**{k:v for k,v in cfg.items() if k not in ('learning_rate','batch_size','source')})
        m=Model(c)
    finally:
        try: sys.path.remove(str(root))
        except ValueError: pass
    return m,cfg

def forward_tslib(model,kind,x):
    xe=x.unsqueeze(-1); label=RAW_HISTORY//2
    xd=torch.cat([xe[:,-label:,:],torch.zeros((xe.shape[0],HORIZON,1),device=xe.device,dtype=xe.dtype)],dim=1)
    return model(xe,None,xd,None).squeeze(-1)

def tensor_ds(part): return TensorDataset(torch.from_numpy(part['x']),torch.from_numpy(part['y']),torch.from_numpy(part['observed']))

def train_tslib(args):
    parts,d=raw_windows(R2CACHE/f'{args.sensor}.npz'); device=torch.device(args.device); model,cfg=build_tslib(args.model,args.vendor); model=model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']); loader=DataLoader(tensor_ds(parts['train']),batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    vx,vy,vm=[z.to(device) for z in tensor_ds(parts['val']).tensors]; best=float('inf'); state=None; wait=0; hist=[]; be=-1; t0=time.perf_counter()
    for ep in range(1,args.epochs+1):
        model.train(); tot=0.; nb=0
        for x,y,m in loader:
            x,y,m=x.to(device),y.to(device),m.to(device); opt.zero_grad(set_to_none=True); p=forward_tslib(model,args.model,x)
            loss=masked_mean(nn.functional.huber_loss(p,y,reduction='none'),m); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=float(loss.detach()); nb+=1
        model.eval();
        with torch.no_grad(): v=float(masked_mean(torch.abs(forward_tslib(model,args.model,vx)-vy),vm))
        hist.append({'epoch':ep,'train_huber':tot/max(nb,1),'val_MAE_normalized':v})
        if v<best-1e-7: best=v; wait=0; be=ep; state={k:z.detach().cpu().clone() for k,z in model.state_dict().items()}
        else: wait+=1
        if wait>=args.patience: break
    trsec=time.perf_counter()-t0; model.load_state_dict(state); model.eval(); outs=[]
    t0=time.perf_counter()
    with torch.no_grad():
        for x,_,_ in DataLoader(tensor_ds(parts['test']),batch_size=64,shuffle=False): outs.append(forward_tslib(model,args.model,x.to(device)).cpu().numpy())
    pred=np.concatenate(outs); infsec=time.perf_counter()-t0; return parts,d,pred,cfg,model,state,hist,be,best,trsec,infsec

class EMDLSTM(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn=nn.LSTM(5,32,2,batch_first=True,dropout=.1); self.head=nn.Linear(32,HORIZON)
    def forward(self,x): return self.head(self.rnn(x)[0][:,-1])
def load_r3b():
    sp=importlib.util.spec_from_file_location('bridge_r3b_e2',R3B_SRC); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.r2

def train_emd_lstm(args):
    r2=load_r3b(); parts,d=r2.load_windows(R2CACHE/f'{args.sensor}.npz',NOMINAL_LOOK_BACK,HORIZON); device=torch.device(args.device)
    def ds(p):
        xx=np.concatenate([p['low'][:,:,None],p['high']],axis=-1).astype('float32'); yy=(p['y']-p['base'][:,None]).astype('float32')
        return TensorDataset(torch.from_numpy(xx),torch.from_numpy(yy),torch.from_numpy(p['observed'].astype('float32')))
    model=EMDLSTM().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loader=DataLoader(ds(parts['train']),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    vx,vy,vm=[z.to(device) for z in ds(parts['val']).tensors]; best=float('inf'); state=None; wait=0; hist=[]; be=-1; t0=time.perf_counter()
    for ep in range(1,args.epochs+1):
        model.train(); tot=0.; nb=0
        for x,y,m in loader:
            x,y,m=x.to(device),y.to(device),m.to(device); opt.zero_grad(set_to_none=True); p=model(x); loss=masked_mean(nn.functional.huber_loss(p,y,reduction='none'),m); loss.backward(); opt.step(); tot+=float(loss.detach()); nb+=1
        model.eval();
        with torch.no_grad(): v=float(masked_mean(torch.abs(model(vx)-vy),vm))
        hist.append({'epoch':ep,'train_huber':tot/max(nb,1),'val_residual_MAE_normalized':v})
        if v<best-1e-7: best=v; wait=0; be=ep; state={k:z.detach().cpu().clone() for k,z in model.state_dict().items()}
        else: wait+=1
        if wait>=args.patience: break
    trsec=time.perf_counter()-t0; model.load_state_dict(state); model.eval(); outs=[]; t0=time.perf_counter()
    with torch.no_grad():
        for x,_,_ in DataLoader(ds(parts['test']),batch_size=64): outs.append(model(x.to(device)).cpu().numpy())
    pred=np.concatenate(outs)+parts['test']['base'][:,None]; infsec=time.perf_counter()-t0
    cfg={'input':'same leakage-safe causal MA+EMD components as R3B','lstm_hidden':32,'lstm_layers':2,'dropout':.1,'learning_rate':1e-3,'batch_size':32,'source':'matched EMD-LSTM baseline'}
    return parts,d,pred,cfg,model,state,hist,be,best,trsec,infsec

def run(args):
    seed_all(args.seed); root=Path(args.output); case=root/'cases'/args.sensor/f'{args.model}-seed{args.seed}'
    if (case/'result.json').exists(): print('SKIP',case); return
    case.mkdir(parents=True,exist_ok=True)
    if args.model=='emd_lstm': parts,d,pred,cfg,model,state,hist,be,best,trsec,infsec=train_emd_lstm(args); raw_budget=149
    else: parts,d,pred,cfg,model,state,hist,be,best,trsec,infsec=train_tslib(args); raw_budget=RAW_HISTORY
    width=float(d['scale_hi']-d['scale_lo']); lo=float(d['scale_lo']); true=parts['test']['y']*width+lo; pp=pred*width+lo; ref=np.repeat(parts['test']['base'][:,None],HORIZON,axis=1)*width+lo
    sc=metrics(true,pp,parts['test']['observed'],ref)
    np.savez_compressed(case/'predictions.npz',y_true=true,y_pred=pp,persistence=ref,observed=parts['test']['observed'])
    torch.save({'state_dict':state,'model':args.model,'config':cfg,'protocol':PROTOCOL,'seed':args.seed},case/'best.pt'); save_json(case/'training_history.json',hist)
    res={'status':'success','protocol':PROTOCOL,'dataset':args.sensor,'model':args.model,'seed':args.seed,'raw_history_budget':raw_budget,'horizon':HORIZON,'cache_sha256':digest(R2CACHE/f'{args.sensor}.npz'),'observed':sc,'model_info':{'parameter_count':sum(p.numel() for p in model.parameters()),'best_epoch':be,'epochs_run':len(hist),'validation_metric':best,'train_seconds':trsec,'inference_seconds':infsec,'config':cfg,'vendor_commit':TSLIB_SHA if args.model!='emd_lstm' else None}}
    save_json(case/'result.json',res); print('RESULT',json.dumps(res,ensure_ascii=False))
def summarize(root):
    rows=[]
    for p in sorted((Path(root)/'cases').rglob('result.json')):
        d=json.loads(p.read_text()); rows.append({'dataset':d['dataset'],'model':d['model'],'MAE':d['observed']['MAE'],'RMSE':d['observed']['RMSE'],'params':d['model_info']['parameter_count']})
    df=pd.DataFrame(rows); df.to_csv(Path(root)/'summary.csv',index=False); m=df.groupby('model',as_index=False).agg(mean_MAE=('MAE','mean'),mean_RMSE=('RMSE','mean'),sensors=('dataset','nunique'),params=('params','max')); m.to_csv(Path(root)/'summary_macro.csv',index=False); print(m.to_string(index=False))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['run','summarize']); ap.add_argument('--output',required=True); ap.add_argument('--vendor',default='/workspace/code-bridge/repro/20260916-v3-e2/vendor'); ap.add_argument('--sensor',choices=SENSORS); ap.add_argument('--model',choices=MODELS); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='cuda'); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=15); a=ap.parse_args(); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    if a.action=='run': run(a)
    else: summarize(a.output)
if __name__=='__main__': main()
