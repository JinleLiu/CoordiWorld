"""Bridge-only repair for a valid two-point rail rejected by upstream endpoint truncation.

Normal upstream paths are returned unchanged. No extrapolation, actor removal,
or official evaluation change. Invalid zero-length geometry still raises.
"""
import numpy as np
STATS={'calls':0,'repaired':0,'cases':[]}
def install():
 import nuplan.planning.simulation.observation.idm.idm_agent as mod
 from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
 from nuplan.common.actor_state.state_representation import ProgressStateSE2
 old=mod.create_path_from_se2
 if getattr(old,'_cw_short_path',False):return
 def fixed(states):
  STATS['calls']+=1
  try:return old(states)
  except AssertionError as exc:
   if str(exc)!='Path has to has more than 1 element!':raise
   a=np.array([[s.x,s.y,s.heading]for s in states],float)
   if len(a)<2 or not np.isfinite(a).all():raise
   progress=np.r_[0.,np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1).cumsum()]
   if np.isclose(progress[-1],0.):raise
   # Preserve known geometry, retaining the endpoint omitted by upstream
   # zip(states, progress, np.diff(progress)). Only its assertion path changes.
   keep=np.r_[True,~np.isclose(np.diff(progress),0.)]
   points=[ProgressStateSE2(progress=float(p),x=float(r[0]),y=float(r[1]),heading=float(r[2]))for p,r in zip(progress[keep],a[keep])]
   if len(points)<2:raise
   STATS['repaired']+=1;STATS['cases'].append(dict(points=a.tolist(),length=float(progress[-1]),upstream_error=str(exc)))
   return InterpolatedPath(points)
 fixed._cw_short_path=True;mod.create_path_from_se2=fixed
