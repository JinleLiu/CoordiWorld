"""Frozen R12 heads; arbitrary candidate count, unchanged head/qualification."""
from revision_v5_twentythird_common import *
from revision_v5_twentysecond_primary import project
import numpy as np,xgboost as xgb
class PNC:
 def __init__(self,phase='final',fold=None):
  sub='fit'if phase=='final'else'fold'+str(fold);self.models={};self.bindings={}
  for h in ['q','qp','qttc','qlk','dac_event','physical_nc']:
   path=B22/('runs/physical_NC/'+phase+'/'+sub+'/physical/model.ubj'if h=='physical_nc'else'runs/K0/'+phase+'/'+sub+'/'+h+'/model.ubj');m=xgb.Booster();m.load_model(path);m.set_param({'nthread':1});self.models[h]=m;self.bindings[h]=dict(path=str(path),sha256=sha(path))
 def predict(self,a,ec=None):
  x=a['x674'];f=a['features198'];v=a['valid'];g=int(a['geo']);raw={}
  for h,m in self.models.items():
   xx=np.concatenate([x,a['V24']],-1)if h=='q'else(np.concatenate([x,a['R8']],-1)if h=='dac_event'else(np.concatenate([x,a['C32']],-1)if h=='physical_nc'else x));raw[h]=m.inplace_predict(xx.astype(np.float32))
  pr=project(raw);ss=(5*pr[:,1]+5*pr[:,2]+2*pr[:,3]+2*f[:,197]*pr[:,0])/14;da=v&(raw['dac_event']<=raw['dac_event'][g]);el=da&(raw['physical_nc']<=.5)
  if not el.any():el=da&(raw['physical_nc']<=raw['physical_nc'][g])
  score=ss if ec is None else(14*ss+2*pr[:,0]*ec)/16
  return dict(choice=int(np.where(el,score,-np.inf).argmax()),scores=ss,final_scores=score,eligible=el,projected=pr,**{'raw_'+k:z for k,z in raw.items()})
