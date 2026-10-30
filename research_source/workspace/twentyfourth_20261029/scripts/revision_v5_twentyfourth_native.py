"""Pinned scorer with declared compatibility changes and exact trajectory reuse.
Per-mapping checkpoints allow runtime repair without a policy or cohort change.
"""
from revision_v5_twentyfourth_common import *
import argparse,copy,time,gc,signal,shutil
import numpy as np,pandas as pd
from revision_v5_seventeenth_strict_aggregation import aggregate_complete,METRICS
from revision_v5_seventeenth_live import b
from navsim.common.dataclasses import Scene,SensorConfig,Trajectory
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from revision_v5_twentythird_native import Capture

def label_index(scope):
    root=B23/'delivery/verified_files';index=json.load(open(root/'LABEL_INDEX.json')).get(scope,{})
    old_rows_path=B23/'validation/final_comparison'/scope/'choices.json'
    sources={r['scene_id']:r['source']['sha256']for r in json.load(open(old_rows_path))}if old_rows_path.exists()else{}
    return {sid:{r['trajectory_sha']:dict(filtered=root/r['filtered'],physical=root/r['physical'],source_raw_sha=sources[sid])for r in values.values()}for sid,values in index.items()}

def mapping(g):return {(g['initial_token'],g['followup_token']):[tuple(p)for p in g['synthetic_initial_followup_pairs']]}

def run(scope,shard=0,shards=1,aggregate_only=False):
    out=BASE/'validation'/scope;manifest=json.load(open(BASE/'cohorts'/f'{scope}.json'));groups=manifest['plan']['groups']
    if manifest['role']=='CONFIRMATION':assert json.load(open(BASE/'DEVELOPMENT_FREEZE.json'))['complete']
    from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
    install_corrected_builder()
    import nuplan.planning.simulation.observation.idm.idm_agent as idm
    from diagnose_idm_path import retry_dropped_endpoint
    from coordiworld.navsim_adapter import OfficialComponentScorer
    from navsim.evaluate.pdm_score import pdm_score
    import revision_v5_collect_native_pilot as pilot
    oldpath=idm.create_path_from_se2;retries=[];idm.create_path_from_se2=lambda poses:retry_dropped_endpoint(oldpath,poses,retries,True)
    owner=OfficialComponentScorer(ROOT/'data/maps',out/f'metric_cache_shard{shard}','reactive');capture=Capture(owner.scorer)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('native scene300s')))
    aliases=label_index(scope);calls=0;reused=0;start=time.monotonic()
    for g in ([]if aggregate_only else groups[shard::shards]):
        gd=out/'groups'/str(g['registered_group_index'])
        freeze=json.load(open(gd/'selection_freeze.json'));assert freeze['complete']and sha(gd/'choices.json')==freeze['choices_sha']
        if (gd/'native_complete.json').exists():continue
        rows=json.load(open(gd/'choices.json'))
        for r in rows:
            check_time();d=out/'scenes'/r['scene_id'];inp=np.load(d/'inputs.npz');assert sha(d/'inputs.npz')==r['input_sha']
            assert array_sha(inp['candidates'])==r['candidate_sha']and array_sha(inp['valid'])==r['valid_sha']
            if (d/'native_receipt.json').exists():continue
            signal.alarm(300);cache=None;human=None;checks=[];t=time.monotonic()
            for cid in r['candidates_to_evaluate']:
                trajectory=inp['candidates'][cid];th=array_sha(trajectory);dest=d/f'candidate_{cid}_filtered.pkl';phys=d/f'candidate_{cid}_physical.json'
                if dest.exists()and phys.exists():
                    assert json.load(open(phys)).get('trajectory_sha',th)==th
                    checks.append(dict(candidate=cid,resumed=True,trajectory_sha=th));continue
                alias=aliases.get(r['scene_id'],{}).get(th)
                if alias is not None:
                    assert alias['source_raw_sha']==r['source']['sha256']
                    shutil.copy2(alias['filtered'],dest);shutil.copy2(alias['physical'],phys)
                    p=json.load(open(phys));assert p.get('trajectory_sha',th)==th
                    checks.append(dict(candidate=cid,reused=True,trajectory_sha=th,source=str(alias['filtered']),source_sha=sha(alias['filtered']),identity='same scene/submitted float32 trajectory, fixed human, native domain/pin; learned input is not part of native label identity'))
                    reused+=1;continue
                if cache is None:
                    item=r['source'];assert sha(item['server_path'])==item['sha256']
                    if r['origin']=='original':
                        frames=b.load_log(item['server_path']);j=item['current_index'];scene=Scene.from_scene_dict_list(frames[j-3:j+9],None,4,8,SensorConfig.build_no_sensors())
                    else:scene=Scene.load_from_disk(Path(item['server_path']),None,SensorConfig.build_no_sensors())
                    cache=owner.processor.compute_metric_cache(NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0'))
                pilot.reset_rng(17);capture.calls=[]
                tab,states=pdm_score(copy.deepcopy(cache),Trajectory(trajectory.astype(float),owner.prediction_sampling),owner.sampling,owner.simulator,owner.scorer,owner.traffic)
                captured=copy.deepcopy(capture.calls);assert len(captured)==(2 if r['origin']=='original'else 1)
                if len(captured)==2:
                    if human is not None:assert human==captured[1]['raw']
                    human=captured[1]['raw']
                ep=relative_to_absolute_poses(cache.ego_state.rear_axle,[StateSE2(*trajectory[-1].astype(float))])[0]
                tab['valid']=True;tab['token']=r['scene_id'];tab['log_name']=cache.log_name;tab['frame_type']=cache.scene_type;tab['start_time']=cache.timepoint.time_s
                tab['endpoint_x']=ep.x;tab['endpoint_y']=ep.y;tab['executed_endpoint_x']=states[-1,0];tab['executed_endpoint_y']=states[-1,1]
                tab['start_point_x']=cache.ego_state.rear_axle.x;tab['start_point_y']=cache.ego_state.rear_axle.y;tab['ego_simulated_states']=[states]
                tab.to_pickle(dest);write(phys,dict(candidate=cid,calls=captured,filtered={k:float(tab.iloc[0][k])for k in METRICS[:8]},trajectory_sha=th,endpoint_source='submitted trajectory last pose transformed with official initial rear axle',endpoint=[ep.x,ep.y],executed_endpoint=states[-1,:2].tolist(),protocol=PIN,domain=DOMAIN))
                checks.append(dict(candidate=cid,reused=False,trajectory_sha=th,source_raw_sha=r['source']['sha256']));calls+=1
            signal.alarm(0);write(d/'native_receipt.json',dict(scene_id=r['scene_id'],seconds=time.monotonic()-t,checks=checks,human_fixed=True))
            if cache is not None:del cache,scene;gc.collect()
        for method in METHODS:
            raw=pd.concat([pd.read_pickle(out/'scenes'/r['scene_id']/f'candidate_{r["choices"][method]}_filtered.pkl')for r in rows],ignore_index=True)
            scored,summary=aggregate_complete(raw,mapping(g),owner.sampling);write(out/f'group{g["registered_group_index"]}-{method}.json',summary)
        write(gd/'native_complete.json',dict(complete=True,created_utc=now(),selection_freeze_sha=sha(gd/'selection_freeze.json'),domain=DOMAIN,scenes=len(rows)))
        print('R14_NATIVE_GROUP',scope,g['registered_group_index'],len(rows),'new',calls,'reuse',reused,'sec',round(time.monotonic()-start,1),flush=True)
    write(out/f'path_end_retries_shard{shard}.json',retries)
    if all((out/'groups'/str(g['registered_group_index'])/'native_complete.json').exists()for g in groups):
        assert (out/'selection_freeze.json').exists();freeze=json.load(open(out/'selection_freeze.json'));assert sha(out/'choices.json')==freeze['choices_sha']
        rows=json.load(open(out/'choices.json'));mm={k:v for g in groups for k,v in mapping(g).items()};summaries={}
        for method in METHODS:
            raw=pd.concat([pd.read_pickle(out/'scenes'/r['scene_id']/f'candidate_{r["choices"][method]}_filtered.pkl')for r in rows],ignore_index=True);raw.to_pickle(out/(method+'-native.pkl'))
            scored,summary=aggregate_complete(raw,mm,owner.sampling);scored.to_pickle(out/(method+'-scored.pkl'));summaries[method]=summary
        write(out/'result.json',dict(complete=True,created_utc=now(),scope=manifest['role'],cohort=scope,methods=summaries,scenes=len(rows),mappings=len(groups),drives=len({r['drive']for r in rows}),domain=DOMAIN,protocol=PIN,pose_version=POSE_VERSION,selection_freeze_sha=sha(out/'selection_freeze.json'),cohort_sha=sha(BASE/'cohorts'/f'{scope}.json')))
        print('R14_FULL_RESULT',scope,{m:100*v['combined']['score']for m,v in summaries.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--aggregate-only',action='store_true');a=p.parse_args();run(a.scope,a.shard,a.shards,a.aggregate_only)
