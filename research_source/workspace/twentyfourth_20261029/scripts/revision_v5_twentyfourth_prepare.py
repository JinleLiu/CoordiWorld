"""One corrected input version; five frozen policies with their own legal history.

No official future labels are read by preparation. Original native body-state
arrays are preserved; the R12 corrected feature view remains separate.
"""
import copy,time,argparse
import numpy as np,yaml
from revision_v5_twentythird_features import (Pipeline,fix_s64,recover_legacy,
    contact_features,vf,rf,add_candidate,replace_from_appended)
from revision_v5_twentythird_readout import Readouts
from revision_v5_twentythird_memo import install
from revision_v5_twentythird_factory import continuation,ec_matrix
from revision_v5_twentythird_spatial_probe import spatial_dynamic
from revision_v5_twentythird_null_probe import current_profile_candidate
from revision_v5_twentythird_progress_readout import select as pnc_select
from revision_v5_twentythird_control_progress import select as ic_select
from revision_v5_twentysecond_pool_probe import brake
from revision_v5_twentieth_pool_probe import slot
from revision_v5_twentyfirst_full import history
from coordiworld.revision_v5.compiled_solve_controller import predict_vector_controller_execution
from coordiworld.data.openscene import ego_pose
from pyquaternion import Quaternion
from revision_v5_twentyfourth_common import *

def base_inputs(pipe,h,source_sha):
    t=time.perf_counter();b=pipe.inputs(h,source_sha)
    ss,view=fix_s64(b['assets'],b['current'],b['S64'],b['S64_trace'])
    x=b['x'][0].copy();x[:,594:658]=ss
    cf,_=contact_features(b['assets'],b['current'],b['areas'])
    a=dict(candidates=b['candidates'],valid=b['valid'],geo=np.array(b['Geometry_HC']),
        features198=b['features198'],x674=x,old_x674=b['x'][0],
        V24=vf(b['assets'])[0],R8=rf(b['areas'],b['valid']),C32=cf)
    asset=b['assets'];native=view['execution_native'];receipt=dict(brake_supply=False)
    if (((cf[:,0]>0)|(cf[:,1]>0))|~a['valid']).all():
        ref=int(a['valid'].argmax());j,_=slot(a['candidates'],a['valid'],ref)
        dy=h[-1]['ego_dynamic_state'];ca,_=brake(float(dy[0]),float(dy[2]),a['candidates'][ref],6,6)
        a,asset,native,rr=add_candidate(pipe,h,source_sha,a,asset,b['current'],ca.astype(np.float32))
        a,asset=replace_from_appended(a,asset,j)
        native=recover_legacy(asset['execution_full_local'],b['current']['rear_axle_pose'])
        receipt.update(brake_supply=True,brake_slot=int(j),brake_receipt=rr)
    origin=np.asarray(ego_pose(h[-1]));expected=np.r_[h[-1]['ego2global_translation'][:2],Quaternion(*h[-1]['ego2global_rotation']).yaw_pitch_roll[0]]
    np.testing.assert_array_equal(origin,expected)
    # Native execution retains official global rear axle and body kinematics.
    np.testing.assert_allclose(native[a['valid'],0,:3],np.broadcast_to(origin,(a['valid'].sum(),3)),atol=1e-7,rtol=0)
    assert np.max(abs(a['x674'][int(a['valid'].argmax()),396:594]))==0
    old_yaws=[]
    for frame in h:
        w,xq,y,z=np.asarray(frame['ego2global_rotation'],float)
        old_yaws.append(float(np.arctan2(2*(w*z+xq*y),1-2*(y*y+z*z))))
    new_yaws=[float(ego_pose(f)[2])for f in h]
    receipt.update(pose_version=POSE_VERSION,seconds=time.perf_counter()-t,old_yaws=old_yaws,new_yaws=new_yaws,
        max_yaw_difference=float(abs(np.arctan2(np.sin(np.array(old_yaws)-new_yaws),np.cos(np.array(old_yaws)-new_yaws))).max()),
        timestamps=[int(f['timestamp'])for f in h],native_body_state_unmodified=True,
        world_comparison='same R14 pose version only',runtime_pose_source=sha(ROOT/'twentyfourth_20261029/runtime_src/coordiworld/data/openscene.py'))
    return a,asset,native,b['current'],receipt

def load_base(pipe,r,logs,out):
    d=out/'base'/r['scene_id'];d.mkdir(parents=True,exist_ok=True)
    if (d/'receipt.json').exists():
        receipt=json.load(open(d/'receipt.json'));assert receipt['pose_version']==POSE_VERSION
        assert receipt['source_sha']==r['source']['sha256']
        return dict(np.load(d/'features.npz')),dict(np.load(d/'world_assets.npz')),np.load(d/'native.npz')['states'],json.load(open(d/'current.json')),receipt
    h=history(r,logs);a,asset,native,current,receipt=base_inputs(pipe,h,r['source']['sha256'])
    receipt['source_sha']=r['source']['sha256']
    np.savez_compressed(d/'features.npz',**a);np.savez_compressed(d/'world_assets.npz',**asset)
    np.savez_compressed(d/'native.npz',states=native);write(d/'current.json',current);write(d/'receipt.json',receipt)
    return a,asset,native,current,receipt

def prepare(scope,group_index=None,shard=0,shards=1):
    out=BASE/'validation'/scope;out.mkdir(parents=True,exist_ok=True)
    manifest=json.load(open(BASE/'cohorts'/f'{scope}.json'));rows=copy.deepcopy(manifest['rows']);plan=manifest['plan']
    if manifest['role']=='CONFIRMATION':
        freeze=json.load(open(BASE/'DEVELOPMENT_FREEZE.json'));assert freeze['complete']
        assert manifest['authoritative_index_verified'] and manifest['downstream_unexposed']
    model=Readouts();pipe=install(Pipeline(yaml.safe_load(open(ROOT/'runtime_s1/configs/group_cache_v2.yaml'))))
    by={r['scene_id']:r for r in rows};logs={'hashes':set()};start=time.monotonic()
    groups=plan['groups'][shard::shards] if group_index is None else [plan['groups'][group_index]]
    for g in groups:
        gd=out/'groups'/str(g['registered_group_index']);gd.mkdir(parents=True,exist_ok=True)
        if (gd/'selection_freeze.json').exists():continue
        group_rows=[];contracts=[]
        for pair in [(g['initial_token'],g['followup_token']),*g['synthetic_initial_followup_pairs']]:
            check_time();a={};asset={};native={};current={};pred={};bank={};valid={}
            for sid in pair:
                a[sid],asset[sid],native[sid],current[sid],_=load_base(pipe,by[sid],logs,out)
                pred[sid]=model.predict(a[sid]);bank[sid]=list(a[sid]['candidates']);valid[sid]=list(a[sid]['valid'])
            prev,cur=sorted(pair,key=lambda s:current[s]['timestamp_us']);off=round((current[cur]['timestamp_us']-current[prev]['timestamp_us'])/1e5)
            assert 0<off<6;choices={s:{}for s in pair};h=None;feature_cache={}
            for method in METHODS:
                family='Geometry'if method.startswith('Geometry')else ('IC0'if method.startswith('IC0')else'PNC')
                pc=int(pred[prev][family]['choice']);choices[prev][method]=pc;prior=native[prev][pc]
                ec=ec_matrix(native[cur],prior,off);parent=model.predict(a[cur],ec)[family]
                trigger=method!='PNC_EC'and not bool((ec[a[cur]['valid']]>0).any())
                pool=a[cur];world=asset[cur];nn=native[cur];ee=ec;fm={k:k for k in range(12)};variants=[]
                if trigger:
                    if h is None:h=history(by[cur],logs)
                    for variant in (['current']if method=='PNC_NULL'else ['history','current','spatial']):
                        if variant=='history':ca,detail=continuation(native[cur][int(a[cur]['valid'].argmax()),0],prior,current[cur]['timestamp_us'],current[prev]['timestamp_us'],2.,'dynamic')
                        elif variant=='current':ca,detail=current_profile_candidate(native[cur],parent['choice'],current[cur]['timestamp_us'],current[prev]['timestamp_us'])
                        else:ca,detail=spatial_dynamic(native[cur][int(a[cur]['valid'].argmax()),0],prior,off)
                        duplicate=any(np.linalg.norm(ca[:,:2]-old[:,:2],axis=1).max()<1e-3 and abs(np.arctan2(np.sin(ca[:,2]-old[:,2]),np.cos(ca[:,2]-old[:,2]))).max()<1e-4 for old in pool['candidates'][pool['valid']])
                        vr=dict(variant=variant,duplicate=duplicate,admitted=False,detail=detail)
                        if not duplicate or method=='PNC_NULL':
                            _,fn,_=predict_vector_controller_execution(h,ca[None],np.ones(1,bool));ce=float(ec_matrix(fn[0][None],prior,off)[0]);vr['EC']=ce
                            if ce>0:
                                key=hashlib.sha256(pool['candidates'].tobytes()+pool['valid'].tobytes()+ca.tobytes()).hexdigest()
                                if key not in feature_cache:feature_cache[key]=add_candidate(pipe,h,by[cur]['source']['sha256'],pool,world,current[cur],ca)
                                pool,world,nn,rr=feature_cache[key];np.testing.assert_allclose(nn[-1],fn[0],atol=1e-7,rtol=0)
                                ee=ec_matrix(nn,prior,off);bi=next((k for k,c in enumerate(bank[cur])if np.array_equal(c,ca)),None)
                                if bi is None:bi=len(bank[cur]);bank[cur].append(ca);valid[cur].append(True)
                                fm[len(pool['valid'])-1]=bi;vr.update(admitted=True,bank_index=bi,world_maxerr=rr['world_maxerr'],world_comparison='same R14 version')
                        variants.append(vr)
                pp=model.predict(pool,ee);ch=pp[family]['choice'];merged=pool|{'EC':ee}|{family+'_'+k:v for k,v in pp[family].items()}
                if method=='PNC_H3P':ch=pnc_select(merged)
                elif method=='IC0_H3P':ch=ic_select(merged)
                choices[cur][method]=int(fm[ch])
                d=out/'policy_inputs'/method/cur;d.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(d/'features.npz',**pool,native=nn,EC=ee,previous_native=prior,**{family+'_'+k:v for k,v in pp[family].items()})
                rec=dict(method=method,current=cur,previous=prev,previous_choice=pc,previous_timestamp_us=current[prev]['timestamp_us'],timestamp_us=current[cur]['timestamp_us'],offset=off,trigger=trigger,variants=variants,feature_to_bank=fm,parent_choice=int(parent['choice']),choice=choices[cur][method],previous_trajectory_sha=array_sha(a[prev]['candidates'][pc]),first_frame='own frozen policy, no legal earlier issued plan')
                write(d/'receipt.json',rec);contracts.append(rec)
            for sid in pair:
                d=out/'scenes'/sid;d.mkdir(parents=True,exist_ok=True)
                ca=np.stack(bank[sid]);vv=np.array(valid[sid]);np.savez_compressed(d/'inputs.npz',candidates=ca,valid=vv)
                r={k:copy.deepcopy(by[sid][k])for k in ['scene_id','drive','group','origin','source']}
                r.update(reference=int(vv.argmax()),choices=choices[sid],source_folder=str(d),input_sha=sha(d/'inputs.npz'),world_path=str(out/'base'/sid/'world_assets.npz'),world_sha=sha(out/'base'/sid/'world_assets.npz'),candidate_sha=array_sha(ca),valid_sha=array_sha(vv),candidates_to_evaluate=sorted(set(choices[sid].values())),pose_version=POSE_VERSION)
                write(d/'selection.json',r);group_rows.append(r)
        write(gd/'choices.json',group_rows);write(gd/'history_contracts.json',contracts)
        write(gd/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(gd/'choices.json'),models_PNC=model.pnc.bindings,models_IC0=model.ic.bindings,methods=METHODS,pose_version=POSE_VERSION,protocol=PIN,domain=DOMAIN,cohort_sha=sha(BASE/'cohorts'/f'{scope}.json'),new_native_labels_read=False))
        print('R14_GROUP_FROZEN',scope,g['registered_group_index'],len(group_rows),round(time.monotonic()-start,1),flush=True)
    if all((out/'groups'/str(g['registered_group_index'])/'selection_freeze.json').exists()for g in plan['groups']):
        allrows=[];allcontracts=[]
        for g in plan['groups']:
            gd=out/'groups'/str(g['registered_group_index']);allrows+=json.load(open(gd/'choices.json'));allcontracts+=json.load(open(gd/'history_contracts.json'))
        assert len(allrows)==len(rows)and len({r['scene_id']for r in allrows})==len(rows)
        write(out/'choices.json',allrows);write(out/'history_contracts.json',allcontracts)
        write(out/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(out/'choices.json'),methods=METHODS,pose_version=POSE_VERSION,domain=DOMAIN,new_native_labels_read=False,cohort_sha=sha(BASE/'cohorts'/f'{scope}.json')))
        print('R14_ALL_FROZEN',scope,len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope');p.add_argument('--group',type=int);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args();prepare(args.scope,args.group,args.shard,args.shards)
