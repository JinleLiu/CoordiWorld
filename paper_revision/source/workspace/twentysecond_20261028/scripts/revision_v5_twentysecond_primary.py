"""R12 frozen PNC_EC inference on prepared current-only candidate features.
Six consumed heads: Q/QP/QTTC/QLK, DAC and raw-NC BCE. Original NC rank/QR are
not used. Native simulated states remain body-slot arrays for official EC.
Pool preparation and world inference costs are outside this readout interface.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import xgboost as xgb
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort

def project(raw):
 y=np.stack([raw[k].astype(float)for k in ['q','qp','qttc','qlk']],-1);q=y[...,0].copy();ms=np.sort(np.maximum(y[...,1:],0),axis=-1)[...,::-1];total=q.copy();count=np.ones_like(q)
 for k in range(3):
  active=ms[...,k]>q;total+=np.where(active,ms[...,k],0);count+=active;q=total/count
 q=np.clip(q,0,1);return np.concatenate([q[...,None],np.clip(y[...,1:],0,q[...,None])],-1)

class PNC_EC:
 def __init__(self,base):
  base=Path(base);self.models={};self.bindings={}
  for h in ['q','qp','qttc','qlk','dac_event','physical_nc']:
   path=base/('runs/physical_NC/final/fit/physical/model.ubj'if h=='physical_nc'else'runs/K0/final/fit/'+h+'/model.ubj');m=xgb.Booster();m.load_model(path);m.set_param({'nthread':1});self.models[h]=m;self.bindings[h]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path),features=m.num_features())
 def predict(self,f198,x674,v24,r8,c32,valid,geometry_index,native_execution=None,timestamp_us=None,previous_native=None,previous_timestamp_us=None):
  valid=np.asarray(valid,bool);assert valid.shape==(12,)and valid.any();assert valid[geometry_index];ref=int(valid.argmax());expected=np.concatenate([f198,np.broadcast_to(f198[ref],f198.shape),f198-f198[ref]],-1);np.testing.assert_array_equal(x674[:,:594],expected);raw={}
  for h,m in self.models.items():
   xx=np.concatenate([x674,v24],-1)if h=='q'else(np.concatenate([x674,r8],-1)if h=='dac_event'else(np.concatenate([x674,c32],-1)if h=='physical_nc'else x674));raw[h]=m.inplace_predict(xx.astype(np.float32))
  joint=project(raw);score=(5*joint[:,1]+5*joint[:,2]+2*joint[:,3]+2*f198[:,197]*joint[:,0])/14;dac=valid&(raw['dac_event']<=raw['dac_event'][geometry_index]);safe=dac&(raw['physical_nc']<=.5);eligible=safe if safe.any()else dac&(raw['physical_nc']<=raw['physical_nc'][geometry_index]);assert eligible.any();local_choice=int(np.where(eligible,score,-np.inf).argmax());ec_known=False;ec=np.full(12,np.nan);offset=None;final_score=score.copy()
  if previous_native is not None:
   assert native_execution is not None and timestamp_us is not None and previous_timestamp_us is not None;elapsed=(int(timestamp_us)-int(previous_timestamp_us))/1e5;offset=round(elapsed)
   if 0<offset<min(len(previous_native),native_execution.shape[1]):
    length=min(native_execution.shape[1],len(previous_native)-offset);current=native_execution[:,:length];prior=np.broadcast_to(previous_native[offset:offset+length],current.shape);ec=ego_is_two_frame_extended_comfort(current,prior,np.arange(length)*.1).astype(float);final_score=(14*score+2*joint[:,0]*ec)/16;ec_known=True
  choice=int(np.where(eligible,final_score,-np.inf).argmax());return dict(choice=choice,local_choice=local_choice,eligible=eligible,absolute_safe_available=bool(safe.any()),raw=raw,joint=joint,local_scores=score,final_scores=final_score,EC=ec,EC_known=ec_known,previous_offset_steps=offset,rawNC_semantics='unfiltered simulator NC<1, not calibrated realroad probability')
