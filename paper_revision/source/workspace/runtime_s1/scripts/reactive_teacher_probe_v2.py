"""Train-only extraction feasibility; privileged IDM simulator labels,not real counterfactual truth."""
import argparse
import copy
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
import torch
from coordiworld.data.openscene import load_log,ego_pose,rotation,transform_pose
from coordiworld.navsim_adapter import OfficialComponentScorer,NAVSIM_COMMIT,NUPLAN_COMMIT
from diagnose_idm_path import retry_dropped_endpoint


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_vehicle_tracks(detections,tracks,eligible,origin):
    n=len(eligible);states=np.zeros((len(detections),n,7),np.float32);valid=np.zeros((len(detections),n),bool)
    lookup={str(token):i for i,token in enumerate(tracks) if eligible[i]};assert len(lookup)==sum(bool(eligible[i]) for i in range(len(tracks)))
    for t,frame in enumerate(detections):
        for obj in frame.tracked_objects:
            token=str(obj.track_token)
            if token not in lookup:continue
            j=lookup[token];assert not valid[t,j],'Duplicate track token';assert obj.tracked_object_type.name=='VEHICLE','Object class changed'
            pose=transform_pose(np.array([obj.center.x,obj.center.y,obj.center.heading]),origin);velocity=rotation(-origin[2])@np.array([obj.velocity.x,obj.velocity.y])
            states[t,j]=np.r_[pose,velocity,obj.box.length,obj.box.width];valid[t,j]=True
    assert np.isfinite(states).all() and (states[...,5:7][valid]>0).all()
    return states,valid


def teacher_origin(frame,initial):
    from pyquaternion import Quaternion
    initial=np.asarray(initial,dtype=np.float64)
    expected=np.r_[np.asarray(frame['ego2global_translation'],dtype=np.float64)[:2],Quaternion(*frame['ego2global_rotation']).yaw_pitch_roll[0]]
    assert np.max(np.abs(transform_pose(initial,expected)))<1e-5,'Pinned simulator origin mismatch'
    legacy=ego_pose(frame)
    return initial.copy(),{'frame':'pinned NAVSIM current rear-axle local coordinates','quaternion_convention':'normalized Quaternion.yaw_pitch_roll[0]','legacy_adapter_origin':legacy.tolist(),'simulator_origin':initial.tolist(),'simulator_in_legacy_frame':transform_pose(initial,legacy).tolist(),'old_inputs_or_scores_modified':False}


class CapturingTraffic:
    def __init__(self,inner):self.inner=inner;self.calls=[]
    def simulate_environment(self,ego,cache):
        result=self.inner.simulate_environment(ego,cache)
        self.calls.append((np.asarray(ego).copy(),copy.deepcopy(result)))
        return result
    def __getattr__(self,name):return getattr(self.inner,name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    spec=json.loads(Path(args.protocol).read_text());root=Path(args.output);root.mkdir(exist_ok=False)
    def save(name,value):(root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    results=[];current=None;start=time.monotonic();original=None
    try:
        assert spec['allowed_splits']==['train'] and spec['scene_count']==2 and spec['optimizer_updates']==0 and not spec['test_arrays_allowed']
        torch.set_num_threads(1)
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
        for repo,expected in [('navsim',NAVSIM_COMMIT),('nuplan-devkit',NUPLAN_COMMIT)]:
            assert subprocess.check_output(['git','-C','third_party/'+repo,'rev-parse','HEAD'],text=True).strip()==expected
            assert not subprocess.check_output(['git','-C','third_party/'+repo,'status','--porcelain'],text=True).strip(),'Third-party source changed'
        manifest=Path(spec['manifest']);data=json.loads(manifest.read_text());assert sha(manifest)==spec['manifest_sha256']
        records=sorted((r for r in data['scenes'] if r['split']=='train'),key=lambda r:hashlib.sha256(('reactive-teacher-probe-v1:'+r['scene_id']).encode()).hexdigest());selected=[];logs=set()
        for r in records:
            if r['log_name'] in logs:continue
            selected.append(r);logs.add(r['log_name'])
            if len(selected)==2:break
        assert len(selected)==2
        ids={r['scene_id'] for r in selected};labels_path=Path(spec['official_labels']);labels={}
        with labels_path.open() as f:
            for line in f:
                if any('"scene_id": "'+token+'"' in line for token in ids):
                    row=json.loads(line);assert row['scene_id'] in ids;labels[row['scene_id']]=row
        assert set(labels)==ids
        inputs={str(manifest):sha(manifest),str(labels_path):sha(labels_path)}
        for f in ['FAILURE.json','coordinate_diagnostic.json','coordinate_frame_confirmation.json']:
            path=Path('runs/reactive_teacher_probe_v1')/f;inputs[str(path)]=sha(path)
        source_logs={r['name']:r['sha256'] for r in data['source_files']}
        for r in selected:
            log=Path(spec['log_root'])/(r['log_name']+'.pkl');assert sha(log)==source_logs[log.name];inputs[str(log)]=sha(log);inputs[str(manifest.parent/r['path'])]=sha(manifest.parent/r['path'])
        for name in ['src','scripts','configs','tests','docs']:shutil.copytree(name,root/'source_snapshot'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','._*'))
        for rel in ['navsim/evaluate/pdm_score.py','navsim/traffic_agents_policies/abstract_traffic_agents_policy.py','navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py','navsim/common/dataclasses.py']:
            source=Path('third_party/navsim')/rel;dest=root/'source_snapshot/third_party/navsim'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);inputs[str(source)]=sha(source)
        sources={str(p.relative_to(root/'source_snapshot')):sha(p) for p in (root/'source_snapshot').rglob('*') if p.is_file()}
        save('plan.json',{'protocol':spec,'selected_scenes':selected,'input_sha256':inputs,'source_sha256':sources,'created_utc':datetime.now(timezone.utc).isoformat(),'scope':spec['scope']})
        import nuplan.planning.simulation.observation.idm.idm_agent as idm
        original=idm.create_path_from_se2;events=[];idm.create_path_from_se2=lambda states:retry_dropped_endpoint(original,states,events,True)
        for r in selected:
            current=r['scene_id'];save('status.json',{'state':'running','completed':len(results),'total':2,'current':current,'pid':__import__('os').getpid()})
            frames=load_log(Path(spec['log_root'])/(r['log_name']+'.pkl'));indices={str(f['token']):i for i,f in enumerate(frames)};idx=indices[current];origin=ego_pose(frames[idx])
            with np.load(manifest.parent/r['path'],allow_pickle=False) as z:
                candidates=z['input_candidates'][0].copy();candidate_valid=z['input_candidate_valid'][0].copy();observed=z['input_agents'][0,-1].copy();eligible=(z['input_agent_class'][0]==1)&z['input_agent_valid'][0,-1];meta=json.loads(str(z['metadata_json']));legacy_shapes={k:list(z[k].shape) for k in z.files if k.startswith('target_')}
            assert meta['scene_id']==current and meta['candidate_manifest']['sha256']==r['candidate_sha256']==labels[current]['candidate_sha256']
            scorer=OfficialComponentScorer(spec['map_root'],root/'metric_cache','reactive');capture=CapturingTraffic(scorer.traffic);scorer.traffic=capture;cache=scorer.build_metric_cache(frames,idx)
            initial=np.array([cache.ego_state.rear_axle.x,cache.ego_state.rear_axle.y,cache.ego_state.rear_axle.heading]);origin,frame_metadata=teacher_origin(frames[idx],initial)
            m=len(candidates);n=len(eligible);states=np.zeros((m,41,n,7),np.float32);valid=np.zeros((m,41,n),bool);ego_states={};scores={};cases=[]
            for j in np.flatnonzero(candidate_valid):
                capture.calls.clear();events.clear();t=time.monotonic();score=scorer.score(cache,candidates[j]);assert 1<=len(capture.calls)<=2
                simulated,detections=capture.calls[0];assert len(detections)==len(simulated)==41
                teacher,mask=extract_vehicle_tracks(detections,meta['tracks'],eligible,origin)
                assert np.array_equal(mask[0],eligible),'Current retained vehicle coverage mismatch'
                current_error=float(np.max(np.abs(teacher[0,eligible]-observed[eligible]))) if eligible.any() else 0.;assert current_error<=1e-3,'Current object alignment mismatch'
                expected=labels[current]['scores'][str(j)];assert set(score)==set(expected);score_error=max(abs(score[k]-expected[k]) for k in score);assert score_error<=1e-8,'Existing component scores changed'
                states[j]=teacher;valid[j]=mask;ego_states[int(j)]=simulated;scores[str(j)]=score
                cases.append({'candidate':int(j),'current_state_max_error':current_error,'score_max_error':score_error,'captured_environment_calls':len(capture.calls),'current_eligible_vehicles':int(eligible.sum()),'matched_future_vehicle_frames':int(mask[1:].sum()),'endpoint_compat_events':copy.deepcopy(events),'seconds':time.monotonic()-t})
                gc.enable();gc.collect()
            first=int(np.flatnonzero(candidate_valid)[0]);capture.calls.clear();events.clear();repeat_score=scorer.score(cache,candidates[first]);repeat,repeat_valid=extract_vehicle_tracks(capture.calls[0][1],meta['tracks'],eligible,origin)
            assert np.array_equal(repeat,states[first]) and np.array_equal(repeat_valid,valid[first]) and repeat_score==scores[str(first)],'Repeated simulation changed'
            deltas=[]
            for j in np.flatnonzero(candidate_valid):
                if j==first:continue
                common=valid[j,1:]&valid[first,1:];distance=np.linalg.norm(states[j,1:,:,:2]-states[first,1:,:,:2],axis=-1)
                deltas.append({'candidate':int(j),'common_vehicle_frames':int(common.sum()),'mean_position_delta_m':float(distance[common].mean()) if common.any() else None,'max_position_delta_m':float(distance[common].max()) if common.any() else None,'vehicle_frames_delta_over_1mm':int((common&(distance>1e-3)).sum())})
            sample=np.arange(5,41,5);np.savez_compressed(root/(current+'.npz'),states_01s=states,valid_01s=valid,states_05s=states[:,sample],valid_05s=valid[:,sample],candidate_valid=candidate_valid,eligible_current_vehicle=eligible,sample_indices=sample,**{'simulated_ego_'+str(j):v for j,v in ego_states.items()})
            row={'scene_id':current,'log_name':r['log_name'],'split':'train','candidate_sha256':r['candidate_sha256'],'first_valid':first,'tracks':meta['tracks'],'coordinate_frame':frame_metadata,'legacy_target_shapes':legacy_shapes,'candidate_cases':cases,'reaction_vs_first':deltas,'repeated_first_exact':True,'sample_times_s':(sample*.1).tolist(),'scope':spec['scope']};save(current+'.json',row);results.append(row)
        assert all(sha(p)==h for p,h in inputs.items());assert all(sha(root/'source_snapshot'/p)==h for p,h in sources.items())
        save('report.json',{'scenes':results,'completed':2,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-start,'scope':spec['scope']});save('integrity.json',{'passed':True,'scenes':2,'candidate_simulations':sum(len(r['candidate_cases']) for r in results),'repeat_simulations':2,'scores_unchanged':True,'initial_objects_aligned':True,'inputs_sources_unchanged':True,'test_arrays_loaded':False,'optimizer_updates':0,'checked_utc':datetime.now(timezone.utc).isoformat()});save('status.json',{'state':'complete','completed':2,'total':2});print('COMPLETE:2 train scenes;reactive trajectory extraction aligned;prior scores andrepeat simulation matched')
    except Exception:
        save('FAILURE.json',{'current':current,'traceback':traceback.format_exc()});save('status.json',{'state':'failed','completed':len(results),'total':2});raise
    finally:
        if original is not None:idm.create_path_from_se2=original
        gc.enable()


if __name__=='__main__':main()
