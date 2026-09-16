"""TimeMixer++ baseline for BRIDGE-V3-02 using PyPOTS forecasting implementation."""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
RAW_HISTORY=149; HORIZON=30
SENSORS=('DD-EN-09#','DD-ES-09#','DD-WN-09#','DD-WS-09#')
PYPOTS_SHA='53b3eac34be9491ac3f28e65ee1993436e9318af'
PROTOCOL='bridge-v3-e2-timemixerpp-pypots-v1-149to30'
CACHE=Path('/workspace/code-bridge/repro/20260914-r2/cache')
def save_json(p,o): Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def starts(n):
 a,b=int(n*.7),int(n*.7)+int(n*.1); return {k:np.arange(max(RAW_HISTORY,l),r-HORIZON+1,dtype=np.int64) for k,l,r in [('train',0,a),('val',a,b),('test',b,n)]}
def windows(cache):
 with np.load(cache,allow_pickle=False) as zf: d={k:zf[k].copy() for k in zf.files}
 z,obs=d['z'],d['observed']; out={}
 for k,st in starts(len(z)).items():
  xi=st[:,None]-np.arange(RAW_HISTORY,0,-1)[None,:]; yi=st[:,None]+np.arange(HORIZON)[None,:]
  out[k]={'x':z[xi].astype('float32'),'y':z[yi].astype('float32'),'obs':obs[yi].astype(bool),'base':z[st-1].astype('float32')}
 return out,d
def metrics(y,p,obs,ref):
 m=obs.astype(bool); yy=y[m].astype(float); pp=p[m].astype(float); rr=ref[m].astype(float); mae=float(np.abs(yy-pp).mean()); pmae=float(np.abs(yy-rr).mean()); return {'n':int(m.sum()),'MAE':mae,'RMSE':float(np.sqrt(np.square(yy-pp).mean())),'Skill_MAE':float(1-mae/pmae) if pmae>0 else None,'Persistence_MAE':pmae}
def run(a):
 random.seed(a.seed); np.random.seed(a.seed)
 root=Path(a.output); case=root/'cases'/a.sensor/f'timemixerpp-seed{a.seed}'; case.mkdir(parents=True,exist_ok=True)
 if (case/'result.json').exists(): print('SKIP',case); return
 p,d=windows(CACHE/f'{a.sensor}.npz')
 # Make unobserved targets explicit NaN; PyPOTS forecasting supports partially observed targets.
 sets={}
 for k in ('train','val','test'):
  yp=p[k]['y'].copy(); yp[~p[k]['obs']]=np.nan
  sets[k]={'X':p[k]['x'][...,None],'X_pred':yp[...,None]}
 sys.path.insert(0,str(Path(a.vendor)/'pypots'))
 from pypots.forecasting import TimeMixerPP
 model=TimeMixerPP(n_steps=RAW_HISTORY,n_features=1,n_pred_steps=HORIZON,n_pred_features=1,term='long',n_layers=2,d_model=32,d_ffn=32,top_k=5,n_heads=1,n_kernels=3,dropout=.1,channel_mixing=True,channel_independence=True,downsampling_layers=2,downsampling_window=2,use_norm=True,batch_size=32,epochs=a.epochs,patience=a.patience,device='cuda',saving_path=None,model_saving_strategy=None,verbose=False)
 t=time.perf_counter(); model.fit(sets['train'],sets['val']); train_seconds=time.perf_counter()-t
 t=time.perf_counter(); pred=model.predict(sets['test'])['forecasting'].squeeze(-1); infer_seconds=time.perf_counter()-t
 width=float(d['scale_hi']-d['scale_lo']); lo=float(d['scale_lo']); y=p['test']['y']*width+lo; pp=pred*width+lo; ref=np.repeat(p['test']['base'][:,None],HORIZON,axis=1)*width+lo; sc=metrics(y,pp,p['test']['obs'],ref)
 np.savez_compressed(case/'predictions.npz',y_true=y,y_pred=pp,persistence=ref,observed=p['test']['obs'])
 cfg={'n_steps':149,'n_pred_steps':30,'term':'long','n_layers':2,'d_model':32,'d_ffn':32,'top_k':5,'n_heads':1,'n_kernels':3,'dropout':.1,'channel_mixing':True,'channel_independence':True,'downsampling_layers':2,'downsampling_window':2,'use_norm':True,'batch_size':32,'epochs':a.epochs,'patience':a.patience}
 res={'status':'success','protocol':PROTOCOL,'dataset':a.sensor,'model':'timemixerpp','seed':a.seed,'raw_history_budget':149,'horizon':30,'cache_sha256':digest(CACHE/f'{a.sensor}.npz'),'observed':sc,'model_info':{'vendor':'WenjieDu/PyPOTS','vendor_commit':PYPOTS_SHA,'config':cfg,'train_seconds':train_seconds,'inference_seconds':infer_seconds}}
 save_json(case/'result.json',res); print('RESULT',json.dumps(res,ensure_ascii=False))
def summarize(root):
 rows=[]
 for p in sorted((Path(root)/'cases').rglob('result.json')):
  d=json.loads(p.read_text()); rows.append({'dataset':d['dataset'],'model':d['model'],'MAE':d['observed']['MAE'],'RMSE':d['observed']['RMSE']})
 df=pd.DataFrame(rows); df.to_csv(Path(root)/'timemixerpp_summary.csv',index=False); print(df.to_string(index=False)); print('MEAN',df.MAE.mean(),df.RMSE.mean())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['run','summarize']); ap.add_argument('--output',required=True); ap.add_argument('--vendor',default='/workspace/code-bridge/repro/20260916-v3-e2/vendor'); ap.add_argument('--sensor',choices=SENSORS); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=15); a=ap.parse_args(); run(a) if a.action=='run' else summarize(a.output)
if __name__=='__main__': main()
