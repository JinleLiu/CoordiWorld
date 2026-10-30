"""Safe bounded current-context cache: keep the exact5s PDM reference.
Unlike rejected8s-prefix adapter, this only reuses the SAME computation for
identical four-frame history; candidate projections are always recomputed.
"""
from revision_v5_twentythird_context_cache import CapturedProposer,key
from revision_v5_twentythird_common import *
import numpy as np,copy
class MemoProgress:
 def __init__(self,original):self.original=original;self.current_key=None;self.reference=None;self.info=None;self.center=None
 def extract(self,history,assets):
  kk=key(history)
  if kk!=self.current_key:
   p,info=self.original.extract(history,assets);self.center=self.original.processor._pdm_closed._centerline;self.reference={k:copy.deepcopy(p[k])for k in ['pdm_distance','pdm_planned','pdm_executed','centerline','pdm_arc']};self.info=copy.deepcopy(info);self.current_key=kk;return p,info
  from shapely import points
  from coordiworld.data.openscene import ego_pose
  origin=np.asarray(ego_pose(history[-1]));c,s=np.cos(origin[2]),np.sin(origin[2]);ex=assets['execution_full_local'].copy();ex[:,:,:2]=ex[:,:,:2]@np.array([[c,s],[-s,c]])+origin[:2];ex[:,:,2]+=origin[2];offset=self.info['rear_axle_to_center'];xy=ex[...,:2]+offset*np.stack([np.cos(ex[...,2]),np.sin(ex[...,2])],-1);arc=np.asarray(self.center.project(points(xy[...,0],xy[...,1])));d=np.maximum(arc[...,-1]-arc[...,0],0);rem=self.center.length-arc[...,-1];return dict(**copy.deepcopy(self.reference),distance=d,remaining=rem,candidate_executed=ex,candidate_arc=arc),copy.deepcopy(self.info)
def install(pipe):
 pipe.proposer=CapturedProposer(pipe.proposer);pipe.progress=MemoProgress(pipe.progress);return pipe
