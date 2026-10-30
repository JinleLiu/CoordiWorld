"""B17 P16 from current-only official PDM reconstruction; no outcome features."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('NUPLAN_MAPS_ROOT',__import__('coordiworld_paths').workspace_path('data/maps'))
import argparse,datetime as dt,json,time,gc,traceback,signal,hashlib
from pathlib import Path
import numpy as np
from revision_v5_sixteenth_common import write,sha
from revision_v5_seventeenth_math import physical_progress_proxy
ROOT=Path(__import__('coordiworld_paths').workspace_path(''));BASE=ROOT/'seventeenth_20261027';OLD=ROOT/'sixteenth_20261027'
NAMES=['candidate_d_over100','official_PDM_d_over100','pair_max_d_over100','candidate_minus_PDM_d_over100','P0','pair_short_le5m','candidate_minus_reference_d_over100','P0_minus_reference_P0','candidate_minus_Geometry_d_over100','P0_minus_Geometry_P0','official_current_PDM_known','legacy_progress_known','legacy_progress_proxy','fallback_legacy','unknown_P0','centerline_remaining_at_end_over100']
def p_context(distance,pdm_distance,remaining,known,legacy_q,legacy_known,valid,ref,geo):
    M=len(valid);p0,k=physical_progress_proxy(distance,pdm_distance,np.full(M,known));p0=np.where(k,p0,np.where(legacy_known,legacy_q,0));d=np.where(k,distance,0);b=np.full(M,pdm_distance if known else 0.);den=np.maximum(d,b)
    vals=[d/100,b/100,den/100,(d-b)/100,p0,np.where(k,den<=5,0),(d-d[ref])/100,p0-p0[ref],(d-d[geo])/100,p0-p0[geo],k,legacy_known,legacy_q,~k&legacy_known,~k&~legacy_known,np.where(k,remaining,0)/100]
    f=np.stack(vals,-1).astype(np.float32);f[~valid]=0;p0=np.where(valid,p0,0);assert f.shape==(12,16)and np.isfinite(f).all();return f,p0
class Extractor:
    def __init__(self):
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        self.sampling=TrajectorySampling(num_poses=40,interval_length=.1);self.processor=MetricCacheProcessor(None,False,self.sampling);self.sim=PDMSimulator(self.sampling)
    def extract(self,history,assets):
        from navsim.common.dataclasses import Scene,SensorConfig
        from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
        from navsim.evaluate.pdm_score import get_trajectory_as_array
        from shapely import points
        assert len(history)==4
        scene=Scene.from_scene_dict_list(history,Path('.'),4,0,SensorConfig.build_no_sensors());sc=NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0')
        inp,init=self.processor._get_planner_inputs(sc);self.processor._pdm_closed.initialize(init)
        try:trajectory=self.processor._pdm_closed.compute_planner_trajectory(inp)
        finally:gc.enable()
        planned=get_trajectory_as_array(trajectory,self.sampling,sc.initial_ego_state.time_point);executed=self.sim.simulate_proposals(planned[None],sc.initial_ego_state)[0];center=self.processor._pdm_closed._centerline
        from coordiworld.data.openscene import ego_pose
        origin=np.asarray(ego_pose(history[-1]));c,s=np.cos(origin[2]),np.sin(origin[2]);ex=assets['execution_full_local'].copy();ex[:,:,:2]=ex[:,:,:2]@np.array([[c,s],[-s,c]])+origin[:2];ex[:,:,2]+=origin[2]
        offset=sc.initial_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center
        def progress(states):
            xy=states[...,:2]+offset*np.stack([np.cos(states[...,2]),np.sin(states[...,2])],-1);arc=np.asarray(center.project(points(xy[...,0],xy[...,1])));return np.maximum(arc[...,-1]-arc[...,0],0),center.length-arc[...,-1],arc
        d,rem,arc=progress(ex);db,rb,barc=progress(executed)
        return dict(distance=d,pdm_distance=float(db),remaining=rem,pdm_planned=planned,pdm_executed=executed,candidate_executed=ex,centerline=np.array([[x.x,x.y,x.heading]for x in center.discrete_path]),candidate_arc=arc,pdm_arc=barc),dict(current_only=True,current_timestamp_us=int(history[-1]['timestamp']),history_frames=4,future_frames=0,mission_goal=str(sc.get_mission_goal()),PDM_initialization='exact MetricCacheProcessor._get_planner_inputs + fixed _pdm_closed only; no compute_metric_cache',rear_axle_to_center=offset,centerline_length_m=float(center.length))
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['probe','train','dev'],required=True);args=p.parse_args();start=time.monotonic();reg=json.load(open(BASE/'START.json'));end=dt.datetime.fromisoformat(reg['deadline_utc']).timestamp()
    from coordiworld.data.openscene import load_log
    z=np.load(OLD/'train_inputs.npz');ids=z['scene_ids'];features=z['features'];geo=z['choice_Geometry_HC'];valid=z['valid']
    old=np.load(ROOT/'eleventh_20261026/runs/physical_features_v1/dataset.npz');inv={r['scene_id']:r for r in json.load(open(ROOT/'runs/paired_geometry_v2_train128_dev64/input_screening.json'))['rows']};rows=[inv[str(s)]for s in ids]
    if args.phase=='dev':
        assert (BASE/'PRIMARY_FREEZE.json').exists();dv=np.load(OLD/'dev_primary/predictions.npz');man=json.load(open(OLD/'dev_primary/frozen_manifest.json'))['records'];ids=dv['scene_ids'];features=dv['features198'];valid=dv['valid'];geo=np.array([r['selections']['Geometry_HC']for r in man]);rows=[r['input']for r in man]
    indices=list(range(len(ids)))
    if args.phase=='probe':indices=sorted(indices,key=lambda i:hashlib.sha256(('B17_PROGRESS_V1|'+str(ids[i])).encode()).hexdigest())[:6]
    indices.sort(key=lambda i:(rows[i]['log_name'],rows[i]['index']));out=BASE/'progress'/args.phase;out.mkdir(parents=True,exist_ok=False);ext=Extractor();last=None;records=[];arr={};p0s={};failures=[]
    write(out/'registration.json',dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),phase=args.phase,scene_ids=[str(ids[i])for i in indices],names=NAMES,source_sha256=sha(__file__),input_scope='Only4history,current map and existing frozen current-only execution; no future input or outcome-derived selection',fallback='legacy known q else0 with distinct masks',deadline=reg['deadline_utc']))
    def alarm(*_):raise TimeoutError('current PDM extraction scene limit120s')
    signal.signal(signal.SIGALRM,alarm)
    for i in indices:
        assert time.time()<end-5400,'reserve90min';sid=str(ids[i]);r=rows[i];folder=out/sid;folder.mkdir();t=time.monotonic();a=dict(np.load(OLD/'features'/('assets_dev'if args.phase=='dev'else'assets_train')/sid/'assets.npz'));error=None;details={};passets={};ref=int(valid[i].argmax());g=int(geo[i])
        try:
            signal.alarm(120)
            if last!=r['log_name']:
                raw=ROOT/'data/openscene/trainval'/(r['log_name']+'.pkl');assert sha(raw)==r['raw_log_sha256'];frames=load_log(raw);last=r['log_name']
            history=frames[r['index']-3:r['index']+1];assert history[-1]['token']==sid
            passets,details=ext.extract(history,a);d=passets['distance'];db=passets['pdm_distance'];remaining=passets['remaining'];known=True
            details.update(reference=ref,Geometry=g,ref_is_Geometry=ref==g,PDM_ref_executed_XY_max_m=float(abs(passets['pdm_executed'][:,:2]-passets['candidate_executed'][ref,:,:2]).max()),PDM_Geometry_executed_XY_max_m=float(abs(passets['pdm_executed'][:,:2]-passets['candidate_executed'][g,:,:2]).max()))
            if args.phase=='probe':
                # Audit-only official future-bearing cache; never passed into extract or p_context.
                from coordiworld.navsim_adapter import OfficialComponentScorer
                from navsim.evaluate.pdm_score import get_trajectory_as_array
                scorer=OfficialComponentScorer(ROOT/'data/maps',folder/'audit_cache');cache=scorer.build_metric_cache(frames,r['index']);official=get_trajectory_as_array(cache.trajectory,ext.sampling,cache.ego_state.time_point);cline=np.array([[x.x,x.y,x.heading]for x in cache.centerline.discrete_path]);np.testing.assert_array_equal(official,passets['pdm_planned']);np.testing.assert_array_equal(cline,passets['centerline']);details['official_PDM_planned_and_centerline_exact']=True
            np.savez_compressed(folder/'pdm_assets.npz',**passets)
        except Exception as e:
            error=dict(scene_id=sid,error=repr(e),traceback=traceback.format_exc());failures.append(error);write(folder/'failure.json',error);known=False;d=np.zeros(12);db=0;remaining=np.zeros(12)
        finally:signal.alarm(0)
        legacy_known=features[i,:,195].astype(bool)&valid[i];legacy=np.where(legacy_known,features[i,:,196],0);f,p0=p_context(d,db,remaining,known,legacy,legacy_known,valid[i],ref,g);arr[sid]=f;p0s[sid]=p0
        rec=dict(scene_id=sid,index=int(i),current_PDM_known=known,reference=ref,Geometry=g,legacy_fallback_count=int((~np.full(12,known)&legacy_known).sum()),unknown_count=int((valid[i]&~np.full(12,known)&~legacy_known).sum()),seconds=time.monotonic()-t,identity=details,error=error['error']if error else None);write(folder/'record.json',rec);records.append(rec)
        if args.phase=='probe'or len(records)%100==0:print('PROGRESS_INPUT',len(records),len(indices),len(failures),time.monotonic()-start,flush=True)
    selected=np.array([str(s)for s in ids if str(s)in arr]);matrix=np.stack([arr[s]for s in selected]);np.savez_compressed(BASE/'progress'/('P16_'+args.phase+'.npz'),scene_ids=selected,features=matrix,p0=np.stack([p0s[s]for s in selected]),names=np.array(NAMES))
    result=dict(complete=True,scenes=len(records),current_PDM_known=sum(r['current_PDM_known']for r in records),records=records,failures=failures,seconds=time.monotonic()-start,dimensions=16,old658_unchanged=True,outcome_features=False);write(out/'result.json',result);print('PROGRESS_DONE',args.phase,len(records),len(failures),result['seconds'],flush=True)
if __name__=='__main__':main()
