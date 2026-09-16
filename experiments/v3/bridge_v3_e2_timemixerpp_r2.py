"""TimeMixer++ baseline for BRIDGE-V3-02 with deterministic alignment padding.

Scientific protocol remains 149 observed history days -> 30 forecast days.  The
PyPOTS TimeMixer++ backbone with two x2 downsampling stages requires aligned
cross-scale spatial sizes for odd input lengths.  We therefore prepend three
copies of the earliest value in each 149-day window, yielding an internal length
of 152 (= 38 * 4).  This introduces no additional observed information and does
not change origins, targets, masks, split, horizon, or validation semantics.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
RAW_HISTORY=149; MODEL_HISTORY=152; LEFT_PAD=MODEL_HISTORY-RAW_HISTORY; HORIZON=30
SENSORS=('DD-EN-09#','DD-ES-09#','DD-WN-09#','DD-WS-09#')
PYPOTS_SHA='53b3eac34be9491ac3f28e65ee1993436e9318af'
PROTOCOL='bridge-v3-e2-timemixerpp-pypots-backbone-v2-pad149to152'
CACHE=Path('/workspace/code-bridge/repro/20260914-r2/cache')
def save_json(p,o): Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
def starts(n):
 a,b=int(n*.7),int(n*.7)+int(n*.1); return {k:np.arange(max(RAW_HISTORY,l),r-HORIZON+1,dtype=np.int64) for k,l,r in [('train',0,a),('val',a,b),('test',b,n)]}
def windows(cache):
 with np.load(cache,allow_pickle=False) as zf: d={k:zf[k].copy() for k in zf.files}
 z,obs=d['z'],d['observed']; out={}
 for k,st in starts(len(z)).items():
  xi=st[:,None]-np.arange(RAW_HISTORY,0,-1)[None,:]; yi=st[:,None]+np.arange(HORIZON)[None,:]
  out[k]={'x':z[xi].astype('float32'),'y':z[yi].astype('float32'),'obs':obs[yi].astype('float32'),'base':z[st-1].astype('float32')}
 return out,d
def ds(p): return TensorDataset(torch.from_numpy(p['x']),torch.from_numpy(p['y']),torch.from_numpy(p['obs']))
def masked_mean(x,m): return (x*m).sum()/m.sum().clamp_min(1)
def metrics(y,p,obs,ref):
 m=obs.astype(bool); yy=y[m].astype(float); pp=p[m].astype(float); rr=ref[m].astype(float); mae=float(np.abs(yy-pp).mean()); pmae=float(np.abs(yy-rr).mean()); return {'n':int(m.sum()),'MAE':mae,'RMSE':float(np.sqrt(np.square(yy-pp).mean())),'Skill_MAE':float(1-mae/pmae) if pmae>0 else None,'Persistence_MAE':pmae}
def model_input(x):
 # x: [B,149]; prepend copies of the earliest observed history value only.
 if x.shape[1] != RAW_HISTORY: raise ValueError(x.shape)
 pad=x[:,0:1].expand(-1,LEFT_PAD)
 return torch.cat([pad,x],dim=1).unsqueeze(-1)
def run(a):
 seed_all(a.seed); root=Path(a.output); case=root/'cases'/a.sensor/f'timemixerpp-seed{a.seed}'
 if (case/'result.json').exists(): print('SKIP',case); return
 case.mkdir(parents=True,exist_ok=True); p,d=windows(CACHE/f'{a.sensor}.npz'); device=torch.device(a.device)
 sys.path.insert(0,str(Path(a.vendor)/'pypots'))
 from pypots.nn.modules.timemixerpp import BackboneTimeMixerPP
 model=BackboneTimeMixerPP(task_name='long_term_forecast',n_steps=MODEL_HISTORY,n_features=1,n_pred_steps=HORIZON,n_pred_features=1,n_layers=2,d_model=32,d_ffn=32,n_heads=1,dropout=.1,top_k=5,n_kernels=3,channel_mixing=True,channel_independence=True,downsampling_layers=2,downsampling_window=2,downsampling_method='avg',use_future_temporal_feature=False,use_norm=True).to(device)
 opt=torch.optim.Adam(model.parameters(),lr=1e-3); loader=DataLoader(ds(p['train']),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(a.seed)); vx,vy,vm=[z.to(device) for z in ds(p['val']).tensors]
 best=float('inf'); state=None; wait=0; be=-1; hist=[]; t=time.perf_counter()
 for ep in range(1,a.epochs+1):
  model.train(); tot=0.; nb=0
  for x,y,m in loader:
   x,y,m=x.to(device),y.to(device),m.to(device); opt.zero_grad(set_to_none=True); pr=model.forecast(model_input(x),None).squeeze(-1); loss=masked_mean(nn.functional.huber_loss(pr,y,reduction='none'),m); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=float(loss.detach()); nb+=1
  model.eval()
  with torch.no_grad(): val=float(masked_mean(torch.abs(model.forecast(model_input(vx),None).squeeze(-1)-vy),vm))
  hist.append({'epoch':ep,'train_huber':tot/max(nb,1),'val_MAE_normalized':val})
  if val<best-1e-7: best=val; wait=0; be=ep; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else: wait+=1
  if wait>=a.patience: break
 if state is None: raise RuntimeError('no valid checkpoint')
 train_seconds=time.perf_counter()-t; model.load_state_dict(state); model.eval(); outs=[]; t=time.perf_counter()
 with torch.no_grad():
  for x,_,_ in DataLoader(ds(p['test']),batch_size=64): outs.append(model.forecast(model_input(x.to(device)),None).squeeze(-1).cpu().numpy())
 pred=np.concatenate(outs); infer_seconds=time.perf_counter()-t; width=float(d['scale_hi']-d['scale_lo']); lo=float(d['scale_lo']); y=p['test']['y']*width+lo; pp=pred*width+lo; ref=np.repeat(p['test']['base'][:,None],HORIZON,axis=1)*width+lo; sc=metrics(y,pp,p['test']['obs'],ref)
 np.savez_compressed(case/'predictions.npz',y_true=y,y_pred=pp,persistence=ref,observed=p['test']['obs']); torch.save({'state_dict':state,'protocol':PROTOCOL,'seed':a.seed},case/'best.pt'); save_json(case/'training_history.json',hist)
 cfg={'real_history_days':RAW_HISTORY,'internal_model_steps':MODEL_HISTORY,'left_replication_padding':LEFT_PAD,'padding_information_gain':0,'n_pred_steps':30,'task':'long_term_forecast','n_layers':2,'d_model':32,'d_ffn':32,'top_k':5,'n_heads':1,'n_kernels':3,'dropout':.1,'channel_mixing':True,'channel_independence':True,'downsampling_layers':2,'downsampling_window':2,'use_norm':True,'batch_size':32,'lr':1e-3,'epochs':a.epochs,'patience':a.patience}
 res={'status':'success','protocol':PROTOCOL,'dataset':a.sensor,'model':'timemixerpp','seed':a.seed,'raw_history_budget':RAW_HISTORY,'internal_model_steps':MODEL_HISTORY,'horizon':30,'cache_sha256':digest(CACHE/f'{a.sensor}.npz'),'observed':sc,'model_info':{'vendor':'WenjieDu/PyPOTS','vendor_commit':PYPOTS_SHA,'parameter_count':sum(z.numel() for z in model.parameters()),'best_epoch':be,'validation_metric':best,'config':cfg,'train_seconds':train_seconds,'inference_seconds':infer_seconds}}
 save_json(case/'result.json',res); print('RESULT',json.dumps(res,ensure_ascii=False))
def summarize(root):
 rows=[]
 for p in sorted((Path(root)/'cases').rglob('result.json')):
  d=json.loads(p.read_text()); rows.append({'dataset':d['dataset'],'model':d['model'],'MAE':d['observed']['MAE'],'RMSE':d['observed']['RMSE']})
 df=pd.DataFrame(rows); df.to_csv(Path(root)/'timemixerpp_summary.csv',index=False); print(df.to_string(index=False)); print('MEAN',float(df.MAE.mean()),float(df.RMSE.mean()))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['run','summarize']); ap.add_argument('--output',required=True); ap.add_argument('--vendor',default='/workspace/code-bridge/repro/20260916-v3-e2/vendor'); ap.add_argument('--sensor',choices=SENSORS); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='cuda'); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=15); a=ap.parse_args(); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4'))); run(a) if a.action=='run' else summarize(a.output)
if __name__=='__main__': main()
