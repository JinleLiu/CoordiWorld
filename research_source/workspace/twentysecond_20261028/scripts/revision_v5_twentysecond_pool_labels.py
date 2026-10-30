"""Existing-candidate development diagnostics; never added to TRAIN labels."""
from revision_v5_twentysecond_common import *
import argparse,copy,time,shutil,signal
import numpy as np,pandas as pd
from revision_v5_twentyfirst_full import Capture
from revision_v5_seventeenth_strict_aggregation import METRICS
from revision_v5_seventeenth_live import b
from navsim.common.dataclasses import Scene,SensorConfig,Trajectory
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.evaluate.pdm_score import pdm_score
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--complete',action='store_true');a=ap.parse_args();out=BASE/'critical_pools';out.mkdir(exist_ok=True);cases=json.load(open(BASE/'critical_cases.json'))['cases'];rows={r['scene_id']:r for r in json.load(open(B21/'exposed408/choices.json'))};start=time.monotonic()
    from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
    install_corrected_builder()
    import nuplan.planning.simulation.observation.idm.idm_agent as idm
    from diagnose_idm_path import retry_dropped_endpoint
    from coordiworld.navsim_adapter import OfficialComponentScorer
    import revision_v5_collect_native_pilot as pilot
    orig=idm.create_path_from_se2;retries=[];idm.create_path_from_se2=lambda poses:retry_dropped_endpoint(orig,poses,retries,True);owner=OfficialComponentScorer(ROOT/'data/maps',out/'metric_cache','reactive');capture=Capture(owner.scorer)
    record('critical_pool_labels_v1','Distinguish wrong ranking, rejection and absent safe candidates','Supplement original12 then remaining11 only as DEVELOPMENT diagnostics','five fixed old pools; no input/policy/model/TRA IN change')
    allresult=[]
    for case in cases:
        check_time();sid=case['scene_id'];r=rows[sid];src=Path(r['source_folder']);dest=out/sid;dest.mkdir(exist_ok=True);inp=np.load(src/'inputs.npz');assert sha(src/'inputs.npz')==r['input_sha'];np.testing.assert_array_equal(inp['valid'],np.isin(np.arange(12),case['valid']));ids=case['first_diagnostic_candidates']+(case['then_remaining_for_complete_pool']if a.complete else[]);cache=None
        for cid in case['valid']:
            old=B21/'exposed408/scenes'/sid/f'candidate_{cid}_filtered.pkl';phys=B21/'exposed408/scenes'/sid/f'candidate_{cid}_physical.json'
            if old.exists():
                if not(dest/old.name).exists():shutil.copy2(old,dest/old.name);shutil.copy2(phys,dest/phys.name)
            elif cid in ids and not(dest/f'candidate_{cid}_filtered.pkl').exists():
                if cache is None:
                    item=r['source'];assert sha(item['server_path'])==item['sha256']
                    if r['origin']=='original':frames=b.load_log(item['server_path']);j=item['current_index'];scene=Scene.from_scene_dict_list(frames[j-3:j+9],None,4,8,SensorConfig.build_no_sensors())
                    else:scene=Scene.load_from_disk(Path(item['server_path']),None,SensorConfig.build_no_sensors())
                    cache=owner.processor.compute_metric_cache(NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0'))
                tt=time.monotonic();pilot.reset_rng(17);capture.calls=[];poses=inp['candidates'][cid].astype(float);tab,ex=pdm_score(copy.deepcopy(cache),Trajectory(poses,owner.prediction_sampling),owner.sampling,owner.simulator,owner.scorer,owner.traffic);ep=relative_to_absolute_poses(cache.ego_state.rear_axle,[StateSE2(*poses[-1])])[0];tab['valid']=True;tab['token']=sid;tab['log_name']=cache.log_name;tab['frame_type']=cache.scene_type;tab['start_time']=cache.timepoint.time_s;tab['endpoint_x']=ep.x;tab['endpoint_y']=ep.y;tab['executed_endpoint_x']=ex[-1,0];tab['executed_endpoint_y']=ex[-1,1];tab['start_point_x']=cache.ego_state.rear_axle.x;tab['start_point_y']=cache.ego_state.rear_axle.y;tab['ego_simulated_states']=[ex];tab.to_pickle(dest/f'candidate_{cid}_filtered.pkl');write(dest/f'candidate_{cid}_physical.json',dict(candidate=cid,calls=copy.deepcopy(capture.calls),filtered={k:float(tab.iloc[0][k])for k in METRICS[:8]},trajectory_sha=hashlib.sha256(poses.astype(inp['candidates'].dtype).tobytes()).hexdigest(),input_sha=r['input_sha'],protocol_pin=PIN,seconds=time.monotonic()-tt,scope='offline exposed DEVELOPMENT only'));print('POOL_LABEL',sid,cid,tab[METRICS[:8]].to_dict('records'),flush=True)
        labels={}
        for cid in case['valid']:
            p=dest/f'candidate_{cid}_filtered.pkl'
            if p.exists():
                t=pd.read_pickle(p).iloc[0];labels[str(cid)]={k:float(t[k])for k in METRICS[:8]};labels[str(cid)]['U']=float(t['pdm_score']);labels[str(cid)]['old_eligible']=cid in case['eligible']
        safe=[int(k)for k,v in labels.items()if all(v[c]==1 for c in METRICS[:4])];row=dict(scene_id=sid,valid=case['valid'],labels=labels,complete=len(labels)==len(case['valid']),all_hard_safe=safe,old_eligible_safe=[i for i in safe if i in case['eligible']],source=r['source'],input_sha=r['input_sha'],diagnostic_not_training=True);write(dest/'result.json',row);allresult.append(row);print('POOL_CASE',sid,'complete',row['complete'],'safe',safe,flush=True)
    write(out/('complete_result.json'if a.complete else'first12_result.json'),dict(created_utc=now(),cases=allresult,seconds=time.monotonic()-start,new_requested=sum(len(c['first_diagnostic_candidates'])+(len(c['then_remaining_for_complete_pool'])if a.complete else 0)for c in cases),path_end_retries=retries))
if __name__=='__main__':main()
