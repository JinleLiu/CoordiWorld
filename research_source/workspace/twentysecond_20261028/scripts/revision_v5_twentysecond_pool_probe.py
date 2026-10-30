"""Current-state continuous braking mechanism probe on already exposed failures.
Candidate curves are generated without outcome labels. Scores are diagnostics.
"""
from revision_v5_twentysecond_common import *
import copy,time,numpy as np,pandas as pd
from revision_v5_twentysecond_pool_labels import *
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import get_collision_type
class CaptureMechanism(Capture):
 def __init__(self,scorer):self.scorer=scorer;super().__init__(scorer)
 def score(self,*args,**kwargs):
  result=super().score(*args,**kwargs);sc=self.scorer;i=1 if len(self.calls)==1 else 0;t=sc._collision_time_idcs[i];rec=self.calls[-1];rec['first_collision_seconds']=float(t*.1)if np.isfinite(t)else None;rec['first_ttc_seconds']=float(sc._ttc_time_idcs[i]*.1)if np.isfinite(sc._ttc_time_idcs[i])else None;rec['initial_state']=sc._states[i,0].tolist();rec['terminal_state']=sc._states[i,-1].tolist();rec['initial_collided_tokens']=list(sc._observation.collided_track_ids);rec['contacts']=[]
  if np.isfinite(t):
   t=int(t);contacts=sc._observation[t].query(sc._ego_polygons[:,t],predicate='intersects')
   for pi,gi in zip(*contacts):
    if pi!=i:continue
    token=sc._observation[t].tokens[gi]
    if token not in sc._observation.unique_objects:continue
    actor=sc._observation.unique_objects[token];typ=get_collision_type(sc._states[i,t],sc._ego_polygons[i,t],actor,sc._observation[t][token]);rec['contacts'].append(dict(token=token,kind=str(typ),type=str(actor.tracked_object_type),current_center=[actor.center.x,actor.center.y,actor.center.heading],ego_state=sc._states[i,t].tolist(),actor_polygon_centroid=[sc._observation[t][token].centroid.x,sc._observation[t][token].centroid.y]))
  return result

def brake(v0,a0,reference,decel,jerk):
 # Follow the existing reference spatial curve; replace only time parametrization.
 p=np.vstack([np.zeros(3),reference.astype(float)]);ds=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1);arc=np.r_[0,np.cumsum(ds)];keep=np.r_[True,ds>1e-6];p=p[keep];arc=arc[keep];v=max(0.,v0);a=a0;distance=0.;rows=[];hist=[]
 for it in range(1,401):
  dt=.01;an=max(-decel,a-jerk*dt);vn=max(0.,v+an*dt);distance+=(v+vn)*dt/2;a=(vn-v)/dt;v=vn
  if it%50==0:
   if len(arc)>1 and distance<=arc[-1]:xy=np.array([np.interp(distance,arc,p[:,j])for j in [0,1]]);yaw=np.interp(distance,arc,np.unwrap(p[:,2]))
   else:yaw=p[-1,2]if len(arc)>1 else 0.;xy=p[-1,:2]+(distance-arc[-1])*np.array([np.cos(yaw),np.sin(yaw)])
   rows.append([*xy,yaw]);hist.append([it*.01,distance,v,a])
 return np.array(rows),np.array(hist)

def main():
 out=BASE/'pool_probe';out.mkdir(exist_ok=True);rows={r['scene_id']:r for r in json.load(open(B21/'exposed408/choices.json'))};cases=json.load(open(BASE/'critical_cases.json'))['cases'];record('continuous_brake_probe','All valid candidates in five failed pools have NC0; test speed/acceleration continuity and stopping feasibility','Reference-path jerk-limited3mps2/6mps2 braking from observed native v/a','Five exposed DEVELOPMENT failure pools; candidate feasibility only, no adoption or TRAIN fitting')
 from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
 install_corrected_builder()
 import nuplan.planning.simulation.observation.idm.idm_agent as idm
 from diagnose_idm_path import retry_dropped_endpoint
 from coordiworld.navsim_adapter import OfficialComponentScorer
 import revision_v5_collect_native_pilot as pilot
 orig=idm.create_path_from_se2;retries=[];idm.create_path_from_se2=lambda poses:retry_dropped_endpoint(orig,poses,retries,True);owner=OfficialComponentScorer(ROOT/'data/maps',out/'metric_cache','reactive');capture=CaptureMechanism(owner.scorer);allrows=[]
 for case in cases:
  sid=case['scene_id'];r=rows[sid];dest=out/sid;dest.mkdir(exist_ok=True);src=Path(r['source_folder']);inp=np.load(src/'inputs.npz');current=json.load(open(src/'current.json'));scene=Scene.load_from_disk(Path(r['source']['server_path']),None,SensorConfig.build_no_sensors());cache=owner.processor.compute_metric_cache(NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0'));v0=cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.x;a0=cache.ego_state.dynamic_car_state.rear_axle_acceleration_2d.x;reference=inp['candidates'][r['reference']];candidates={'old_IC0':inp['candidates'][r['choices']['IC0']],'brake3':brake(v0,a0,reference,3,3)[0],'brake6':brake(v0,a0,reference,6,6)[0]};res=[]
  for name,poses in candidates.items():
   pilot.reset_rng(17);capture.calls=[];tab,ex=pdm_score(copy.deepcopy(cache),Trajectory(poses,owner.prediction_sampling),owner.sampling,owner.simulator,owner.scorer,owner.traffic);rec=dict(scene_id=sid,method=name,components={k:float(tab.iloc[0][k])for k in METRICS[:8]},U=float(tab.iloc[0].pdm_score),calls=copy.deepcopy(capture.calls),initial_native_v=v0,initial_native_a=a0,ex_end_v=float(ex[-1,3]));write(dest/(name+'.json'),rec);np.savez_compressed(dest/(name+'.npz'),candidate=poses,execution_native=ex);res.append(rec);print('B22_POOL_PROBE',sid,name,'NC',rec['components'][METRICS[0]],'collision',rec['calls'][0]['first_collision_seconds'],'v0',v0,'endv',rec['ex_end_v'],flush=True)
  allrows.extend(res)
 write(out/'result.json',dict(complete=True,created_utc=now(),cases=allrows,source_sha=sha(__file__),scope='exposed failure diagnostic; not TRAIN/generalization evidence'))
if __name__=='__main__':main()
