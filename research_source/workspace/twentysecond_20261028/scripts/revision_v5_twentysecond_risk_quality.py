"""Risk quality on fixed selected populations, never own-selection-only scoring."""
from revision_v5_twentysecond_common import *
from revision_v5_sixteenth_moments import fit_row_weights
import numpy as np

def quality(p,y,w):
 p=np.asarray(p,float);y=np.asarray(y,float);w=np.asarray(w,float);w=w/w.sum();loss=(p-y)**2;nll=-y*np.log(np.clip(p,1e-7,1-1e-7))-(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7));out=dict(n=len(y),positives=int(y.sum()),weighted_base=float(w@y),Brier=float(w@loss),NLL=float(w@nll),predicted_event=int((p>.5).sum()),FN=int(((p<=.5)&(y==1)).sum()),FP=int(((p>.5)&(y==0)).sum()))
 for name,mask in [('positive',y==1),('negative',y==0)]:out[name]=dict(n=int(mask.sum()),Brier=float(w[mask]@loss[mask]/w[mask].sum())if mask.any()else None,NLL=float(w[mask]@nll[mask]/w[mask].sum())if mask.any()else None,mean_prediction=float(w[mask]@p[mask]/w[mask].sum())if mask.any()else None)
 return out

def main():
 z=np.load(BASE/'supply55/dataset.npz');n=len(z['scene_ids']);ii=np.arange(n);allout={};d=z['drives'];sw=np.array([1/(d==v).sum()for v in d]);roww=fit_row_weights(d,z['valid']);valid=z['valid']
 for phase in ['oof','robust']:
  out=BASE/'runs/physical_NC'/phase;p=np.load(out/'head_predictions.npz');y=(p['physical_NC']<1).astype(float);f=p['fold'];const=np.zeros((n,12))
  for k in range(3):
   tr=f!=k;w=fit_row_weights(d[tr],valid[tr]);rate=float(w@y[tr][valid[tr]]/w.sum());const[f==k]=rate
  selectors={'all_valid':None,'fixed_SK0':np.load(BASE/'runs/SK0'/phase/'oof.npz')['choice'],'fixed_Geometry':z['choice_Geometry_HC']};rr={}
  for pop,choice in selectors.items():
   rr[pop]={}
   for name,pr in [('filtered_BCE',p['filtered']),('physical_BCE',p['physical']),('TRAIN_base_constant',const)]:
    if choice is None:pp=pr[valid];yy=y[valid];ww=roww
    else:pp=pr[ii,choice];yy=y[ii,choice];ww=sw
    rr[pop][name]=quality(pp,yy,ww)
  allout[phase]=rr
 write(BASE/'runs/physical_NC/quality.json',dict(created_utc=now(),scope='TRAIN rawNC, drive-equal then scene/candidate; fixed selection populations; fold-fit constant; exploratory no new calibration',results=allout));print('B22_RISK_QUALITY',{p:{k:{m:round(r['Brier'],6)for m,r in v.items()}for k,v in rr.items()}for p,rr in allout.items()},flush=True)
if __name__=='__main__':main()
