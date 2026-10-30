"""Native labels for TRAIN pose-drift decisions; old submission is replayed too."""
from revision_v5_twentyfourth_native import *
from revision_v5_twentythird_readout import Readouts

def utility(c):return float(np.prod(c[:4])*(5*c[4]+5*c[5]+2*c[6]+2*c[7])/14)

def main():
    from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
    install_corrected_builder()
    import nuplan.planning.simulation.observation.idm.idm_agent as idm
    from diagnose_idm_path import retry_dropped_endpoint
    from coordiworld.navsim_adapter import OfficialComponentScorer
    from navsim.evaluate.pdm_score import pdm_score
    import revision_v5_collect_native_pilot as pilot
    op=idm.create_path_from_se2;retries=[];idm.create_path_from_se2=lambda poses:retry_dropped_endpoint(op,poses,retries,True)
    out=BASE/'train_drift_native';out.mkdir(exist_ok=True);owner=OfficialComponentScorer(ROOT/'data/maps',out/'metric_cache','reactive');capture=Capture(owner.scorer)
    rows=json.load(open(BASE/'train_pose_check/result.json'))['records'];results=[]
    for r in rows:
        if r['old_choices']==r['new_choices']:continue
        saved=out/r['scene_id']/'result.json'
        if saved.exists():results.append(json.load(open(saved)));continue
        so=r['source'];path=ROOT/'data/openscene/trainval'/(so['log_name']+'.pkl');frames=b.load_log(path);j=so['index']
        scene=Scene.from_scene_dict_list(frames[j-3:j+9],None,4,8,SensorConfig.build_no_sensors());cache=owner.processor.compute_metric_cache(NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0'))
        new=dict(np.load(BASE/'train_pose_check'/r['scene_id']/'features.npz'));old=np.load(B23/'spatial_hybrid_features/oof'/r['scene_id']/'features.npz');pp=Readouts('oof',r['fold']).predict(new)
        pool={f'new_{m}':new['candidates'][c]for m,c in r['new_choices'].items()};pool['old_PNC']=old['candidates'][r['old_choices']['PNC']]
        scores={};states={};physical={};done={}
        for key,ca in pool.items():
            hh=array_sha(ca)
            if hh not in done:
                pilot.reset_rng(17);capture.calls=[];tab,st=pdm_score(copy.deepcopy(cache),Trajectory(ca.astype(float),owner.prediction_sampling),owner.sampling,owner.simulator,owner.scorer,owner.traffic)
                done[hh]=(tab,st,copy.deepcopy(capture.calls))
            tab,st,raw=done[hh];c=tab[METRICS[:8]].iloc[0].to_numpy(float);scores[key]=dict(components=c.tolist(),U=100*utility(c),trajectory_sha=hh);states[key]=st;physical[key]=raw
        err=float(abs(np.array(scores['old_PNC']['components'])-r['old_components']['PNC']).max())
        rec=dict(scene_id=r['scene_id'],drive=r['drive'],old_choices=r['old_choices'],new_choices=r['new_choices'],old_label_replay_maxerror=err,scores=scores,physical=physical,
            new_PNC_raw_predictions={k:pp['PNC'][k].tolist()for k in ['raw_physical_nc','raw_dac_event','eligible']},
            selected_path_maxdistance=float(np.linalg.norm(pool['new_PNC'][:,:2]-pool['old_PNC'][:,:2],axis=1).max()),
            all_candidate_row_maxdistance=np.linalg.norm(new['candidates'][:,:,:2]-old['candidates'][:12,:,:2],axis=2).max(1).tolist())
        errors=np.abs(np.array(scores['old_PNC']['components'])-r['old_components']['PNC'])
        rec['old_label_component_errors']=dict(zip(METRICS[:8],errors.tolist()))
        rec['replay_tolerance']='P <= 2e-7; all categorical components exact; a 1.1745e-8 P difference caused the earlier overly strict diagnostic assertion'
        assert errors[4]<=2e-7 and np.delete(errors,4).max()==0,(r['scene_id'],errors)
        d=out/r['scene_id'];d.mkdir(exist_ok=True);np.savez_compressed(d/'states.npz',**states,**{'candidate_'+k:v for k,v in pool.items()});write(d/'result.json',rec);results.append(rec)
        print('TRAIN_DRIFT_NATIVE',r['scene_id'],'old',scores['old_PNC'],'new',scores['new_PNC'],'pathdelta',rec['selected_path_maxdistance'],flush=True)
    write(out/'result.json',dict(complete=True,created_utc=now(),records=results,scope='TRAIN diagnostic; fixed old heads, exact old label replay and corrected new submissions; no EC or complete protocol inference'))

if __name__=='__main__':main()
