"""Frozen current-only identifiability and rail-response experiment.

Same-family simulator reconstruction is a mechanism/control, not independent
prediction success. Prediction never receives future frames or teacher arrays.
"""
import argparse, copy, hashlib, json, os, signal, time, traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing
import numpy as np
import torch
import revision_v5_train_s1_exploratory as train
import revision_v5_collect_native_pilot as pilot
from coordiworld.revision_v5.current_rail_response import current_context, restrict_context, make_policy, rollout_current
from coordiworld.revision_v5.native_adapter import predict_current_controller_execution


def save(path,value):
    path=Path(path); temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def initialize(root,output,variants):
    global ROOT,OUT,VARIANTS,POLICY,EVENTS
    ROOT=Path(root); OUT=Path(output); VARIANTS=variants; EVENTS=[]
    os.environ['NUPLAN_MAPS_ROOT']=str(ROOT/'data/maps');torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
    install_corrected_builder()
    import nuplan.planning.simulation.observation.idm.idm_agent as idm
    from diagnose_idm_path import retry_dropped_endpoint
    original=idm.create_path_from_se2
    idm.create_path_from_se2=lambda poses:retry_dropped_endpoint(original,poses,EVENTS,True)
    POLICY=make_policy(ROOT/'data/maps')


def row_metrics(pred,valid,truth,tvalid,cv,ref,current,role,cohort,sid,drive):
    mask=tvalid&tvalid[ref][None]&cv[:,None,None];mask[ref]=False
    supported=valid&valid[ref][None]&cv[:,None,None]
    delta=pred[...,:2]-pred[ref,...,:2][None]
    delta=np.where(supported[...,None],delta,0.)
    target=truth[...,:2]-truth[ref,...,:2][None]
    error=np.linalg.norm(delta-target,axis=-1);zero=np.linalg.norm(target,axis=-1)
    groups={'all':np.ones_like(mask),'slow_current':np.broadcast_to(np.linalg.norm(current[:,3:5],axis=-1)<=.5,mask.shape),
        'moving_current':np.broadcast_to(np.linalg.norm(current[:,3:5],axis=-1)>.5,mask.shape),
        'positive_target_over_1cm_DIAGNOSTIC':zero>.01}
    result={}
    for name,sub in groups.items():
        use=mask&sub;n=int(use.sum())
        result[name]={'scene_id':sid,'drive':drive,'role':role,'cohort':cohort,'cells':n,
            'error_m':float(error[use].mean()) if n else None,'zero_m':float(zero[use].mean()) if n else None,
            'missing_prediction_cells':int((use&~supported).sum()),'max_error_m':float(error[use].max()) if n else None}
    both=tvalid&valid&cv[:,None,None]
    result['absolute_teacher']={'cells':int(both.sum()),'missing':int((tvalid&cv[:,None,None]&~valid).sum()),
        'max_state_error':float(np.max(np.abs(pred[both]-truth[both]))) if both.any() else None,
        'validity_identical':bool(np.array_equal(valid[cv],tvalid[cv]))}
    return result


def work(row):
    start=time.monotonic();sid=row['scene_id'];folder=OUT/sid;folder.mkdir()
    def timeout(*_):raise TimeoutError('fixed 240s scene budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(240)
    try:
        from coordiworld.data.openscene import load_log
        from reactive_teacher_probe_v2 import extract_vehicle_tracks, teacher_origin
        from coordiworld.revision_v5.native_adapter import read_legacy_inputs
        requested=json.loads((Path(row['root'])/'requested_scene.json').read_text())
        raw=ROOT/'data/openscene/trainval'/ (row['log_name']+'.pkl')
        if train.sha(raw)!=requested['raw_log_sha256']:raise ValueError('Raw source changed')
        frames=load_log(raw);i=requested['index'];history=frames[i-3:i+1];del frames
        if str(history[-1]['token'])!=sid:raise ValueError('Raw scene identity mismatch')
        scene,metadata=read_legacy_inputs(requested['cache_path'],requested)
        execution,full,receipt=predict_current_controller_execution(history,scene.candidates[0].numpy(),scene.candidate_valid[0].numpy())
        context=current_context(history,ROOT/'data/maps')
        origin,_=teacher_origin(history[-1],np.array([context.ego_state.rear_axle.x,context.ego_state.rear_axle.y,context.ego_state.rear_axle.heading]))
        eligible=((scene.agent_class[0]==1)&scene.agent_valid[0,-1]).numpy();cv=scene.candidate_valid[0].numpy();ref=int(np.flatnonzero(cv)[0])
        predictions={};timings={};counts={};events={}
        # Crucially, no teacher array has been opened before ALL predictions.
        for variant in VARIANTS:
            ctx=context if variant=='full_current' else restrict_context(context,metadata['tracks'],scene if variant=='tensor_roi' else None)
            states=np.zeros((len(cv),41,len(eligible),7),np.float32);valid=np.zeros(states.shape[:-1],bool)
            counts[variant]=len(ctx.current_tracked_objects[0].tracked_objects);tic=time.monotonic();EVENTS.clear()
            for j in np.flatnonzero(cv):
                pilot.reset_rng(17)
                detections=rollout_current(POLICY,copy.deepcopy(ctx),full[int(j)])
                states[j],valid[j]=extract_vehicle_tracks(detections,metadata['tracks'],eligible,origin)
            predictions[variant]=(states,valid);timings[variant]=time.monotonic()-tic;events[variant]=len(EVENTS)
        arrays={}
        for variant,(states,valid) in predictions.items():arrays[variant+'_states']=states[:,5::5];arrays[variant+'_valid']=valid[:,5::5]
        np.savez_compressed(folder/'predictions.npz',**arrays)
        # Evaluation boundary, hashes verified against frozen collection receipts.
        for name,digest in row['files_sha256'].items():
            if train.sha(Path(row['root'])/name)!=digest:raise ValueError('Teacher collection changed')
        with np.load(Path(row['root'])/'teacher.npz',allow_pickle=False) as a:
            truth=a['states_01s'];tvalid=a['valid_01s'];controller_error=max(float(np.max(np.abs(full[int(j)]-a['simulated_ego_'+str(j)]))) for j in np.flatnonzero(cv))
        if controller_error>1e-7:raise ValueError('Controller mismatch')
        metrics={k:row_metrics(p[:,5::5],v[:,5::5],truth[:,5::5],tvalid[:,5::5],cv,ref,scene.agents[0,-1].numpy(),row['role'],row['sampling_cohort'],sid,row['drive']) for k,(p,v) in predictions.items()}
        result={'status':'complete','scene_id':sid,'drive':row['drive'],'role':row['role'],'cohort':row['sampling_cohort'],
            'valid_candidates':int(cv.sum()),'current_object_counts':counts,'seconds_by_variant':timings,'seconds':time.monotonic()-start,
            'controller_max_error':controller_error,'endpoint_events':events,'metrics':metrics,'prediction_sha256':train.sha(folder/'predictions.npz'),
            'future_frames_passed_to_predictor':0,'teacher_arrays_opened_after_predictions':True,'same_family_teacher_baseline':True}
        save(folder/'result.json',result);return result
    except Exception as e:
        result={'status':'failed','scene_id':sid,'error':repr(e),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start}
        save(folder/'FAILURE.json',result);return result
    finally:signal.alarm(0)


def aggregate(results,variant,group,cohort=None):
    rows=[r['metrics'][variant][group] for r in results if r['status']=='complete' and (cohort is None or r['cohort']==cohort)]
    supported=[r for r in rows if r['cells']];bydrive=defaultdict(list)
    for r in supported:bydrive[r['drive']].append(r)
    drives=[{'drive':d,'error_m':float(np.mean([r['error_m'] for r in rs])),'zero_m':float(np.mean([r['zero_m'] for r in rs]))} for d,rs in sorted(bydrive.items())]
    return {'requested_completed_scenes':len(rows),'supported_scenes':len(supported),'supported_drives':len(drives),'cells':sum(r['cells'] for r in rows),
        'missing_prediction_cells':sum(r['missing_prediction_cells'] for r in rows),
        'error_m':float(np.mean([r['error_m'] for r in drives])) if drives else None,'zero_m':float(np.mean([r['zero_m'] for r in drives])) if drives else None,
        'max_error_m':max((r['max_error_m'] for r in supported),default=None),'drive_rows':drives}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['probe','dev'],required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False)
    plan=train.make_plan([root/'runs/paired_geometry_v2_train128_dev64',root/'runs/follower_cohort_v1'],'results/2026-10-18-data-protocol-v2/frozen-roster.json')
    records=[r for r in plan['records'] if r['role']==('probe' if a.phase=='probe' else 'development_validation')]
    records.sort(key=lambda r:hashlib.sha256(('current-rail-v1:'+r['scene_id']).encode()).hexdigest())
    if a.limit:records=records[:a.limit]
    variants=['full_current','roi_current','tensor_roi'] if a.phase=='probe' else ['full_current','tensor_roi']
    sources={f:train.sha(f) for f in [__file__,'src/coordiworld/revision_v5/current_rail_response.py','src/coordiworld/revision_v5/idm_teacher_geometry.py','scripts/diagnose_idm_path.py']}
    save(out/'pre_registration.json',{'phase':a.phase,'records':records,'variants':variants,'source_sha256':sources,'workers':2,'seconds_per_scene':240,
        'selection':'all pre-existing fixed TRAIN probe or B DEV identities; no response-conditioned selection','future_predictor_input':False,
        'input_extension':'complete static road graph; full_current also all original current annotations/order; tensor_roi retains tensor actors/states/order but full static graph',
        'fixed_rule':'No parameter fitting; native IDM current settings; reference-preserving response delta. All variants report all failures and missing coverage.',
        'same_family_control_not_independent_model_success':True,'factual_weights_unchanged':True,'original_paper_goals_unchanged':True,
        'pre_experiment_expected_full_current_error_m':0.0,'numerical_parity_tolerance_m':1e-5,'expected_roi_error_m':None})
    results=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(root),str(out),variants)) as pool:
        futures=[pool.submit(work,row) for row in records]
        for f in as_completed(futures):
            try:r=f.result()
            except Exception as e:r={'status':'failed','error':repr(e)}
            results.append(r);save(out/'status.json',{'requested':len(records),'completed':sum(r['status']=='complete' for r in results),'failed':sum(r['status']!='complete' for r in results),'seconds':time.monotonic()-start})
            print(json.dumps({k:r.get(k) for k in ['status','scene_id','seconds','error']}),flush=True)
    groups=['all','slow_current','moving_current','positive_target_over_1cm_DIAGNOSTIC']
    summary={v:{g:aggregate(results,v,g) for g in groups} for v in variants}
    cohorts=sorted({r['sampling_cohort'] for r in records})
    summary_cohorts={v:{c:aggregate(results,v,'all',c) for c in cohorts} for v in variants}
    save(out/'result.json',{'phase':a.phase,'requested':len(records),'completed':sum(r['status']=='complete' for r in results),'failed':[r for r in results if r['status']!='complete'],
        'summary':summary,'cohort_summary':summary_cohorts,'scene_results':results,'seconds':time.monotonic()-start,
        'sources_unchanged':all(train.sha(f)==h for f,h in sources.items()),'same_family_control':True,'qualified':False,'full_EPDMS':None})

if __name__=='__main__':main()
