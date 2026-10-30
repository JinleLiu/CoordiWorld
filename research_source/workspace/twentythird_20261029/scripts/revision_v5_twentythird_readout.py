"""Fair same-history selection with frozenPNC/IC0/Geometry readouts."""
from revision_v5_twentythird_common import *
from revision_v5_twentythird_policy import PNC
from revision_v5_eighteenth_policy import Policy
from coordiworld.revision_v5.component_known_b13 import geometry_known_score
import numpy as np
class Readouts:
 def __init__(self,phase='final',fold=None):
  self.pnc=PNC(phase,fold);self.ic=Policy(dict(value='QV',nc='NR',dac='DR',projection='IP',isotonic=False),phase,fold)
 def predict(self,a,ec=None):
  p=self.pnc.predict(a,ec);b=self.ic.predict(a['old_x674'][None],a['features198'][None],a['valid'][None],np.array([a['geo']]),dict(V24=a['V24'][None],R8=a['R8'][None]));ic={k:v[0]for k,v in b.items()};ic['projected']=np.stack([ic['projected_'+h]for h in ['q','qp','qttc','qlk']],-1);ic['final_scores']=ic['scores']if ec is None else(14*ic['scores']+2*ic['projected'][:,0]*ec)/16;ic['choice']=int(np.where(ic['eligible'],ic['final_scores'],-np.inf).argmax())
  score=geometry_known_score(a['features198'][None],a['features198'][None,:,188:],a['valid'][None])[0];ss=score if ec is None else score+2/14*ec;g=dict(scores=score,final_scores=ss,eligible=a['valid'],choice=int(np.where(a['valid'],ss,-np.inf).argmax()),projected=np.ones((len(score),4)))
  return dict(PNC=p,IC0=ic,Geometry=g)
