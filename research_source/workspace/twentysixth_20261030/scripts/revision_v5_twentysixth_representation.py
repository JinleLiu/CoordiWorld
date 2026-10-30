"""Replace all forecast-dependent R15 v02 relations; never change the C prefix.

Both arms use the same current support, type, dimensions and producer labels.
Producer labels describe the common upstream support, not CV accuracy. Missing
current actors must already be inactive: we do not infer a current state from
Hybrid forecasts. This contract is checked on every exported scene.
"""
from revision_v5_twentysixth_common import *
import argparse,time,copy
from types import SimpleNamespace
import numpy as np,torch
from revision_v5_twentyfifth_features import (WORLD_S64,WORLD_C32,WORLD_V24,
    STATIC_S64,STATIC_C32,STATIC_V24)
from revision_v5_sixteenth_semantics import semantic_features,current_areas
from revision_v5_twentysecond_kinematics import fix_s64
from revision_v5_twentysecond_contact_features import contact_features
from revision_v5_eighteenth_source_features import features as v24
from coordiworld.revision_v5.world_score_features import candidate_features

def dynamic56(a):
    device='cuda';dtype=torch.float32
    def t(x):return torch.as_tensor(x,device=device)
    ego=torch.zeros((1,4,7),dtype=dtype,device=device)
    ego[0,-1,5:7]=t(a['ego_box'])
    scene=SimpleNamespace(ego=ego,ego_rear_axle_to_center=t(a['rear_axle_offset']).reshape(-1),
        agent_valid=t(a['actor_valid'])[None,None].expand(1,4,-1),map_valid=torch.ones((1,1),dtype=torch.bool,device=device))
    ex=t(a['execution']);world=t(a['world'])[None].expand(len(ex),-1,-1,-1)
    rules=torch.zeros((len(ex),8,1,4),dtype=ex.dtype,device=device)
    return candidate_features(scene,ex,rules,torch.zeros_like(rules,dtype=torch.bool),world)[:,64:120].cpu().numpy()

def relations(a,cur,area):
    dynamic=dynamic56(a)
    s,_,tr=semantic_features(a,area);s,_=fix_s64(a,cur,s,tr)
    c,ct=contact_features(a,cur,area);v,vt=v24(a)
    f=np.concatenate([dynamic,s[:,WORLD_S64],c[:,WORLD_C32],v[:,WORLD_V24]],-1).astype(np.float32)
    assert f.shape==(len(a['candidate_valid']),143) and np.isfinite(f).all()
    return f,dict(dynamic56=dynamic,S64=s,C32=c,V24=v),dict(risk_actor=tr['risk_actor_index'],first_contact_actor=ct['first_actor'],first_contact_step=ct['first_step'])

def representations(a,cur,current_prefix,old_world_features=None):
    assert not np.any(a['actor_valid']&~a['current_known']), 'Active actor lacks legal current state; revise both-arm missing contract before continuing'
    assert np.isfinite(a['current'][a['actor_valid']]).all()
    area,ainfo=current_areas(cur,a['execution_full_local'],ROOT/'data/maps')
    cv={k:v.copy()for k,v in a.items()}
    cv['world']=np.broadcast_to(a['current'],a['world'].shape).copy()
    times=np.arange(1,9,dtype=cv['world'].dtype)*.5
    cv['world'][...,:2]+=times[:,None,None]*a['current'][None,:,3:5]
    # Constant observed velocity, yaw and dimensions; no future existence head.
    ff={};blocks={};trace={};seconds={}
    for name,asset in [('hybrid',a),('cv',cv)]:
        tt=time.perf_counter();ff[name],blocks[name],trace[name]=relations(asset,cur,area)
        torch.cuda.synchronize();seconds[name]=time.perf_counter()-tt
    for field,idx in [('S64',STATIC_S64),('C32',STATIC_C32),('V24',STATIC_V24)]:
        np.testing.assert_array_equal(blocks['hybrid'][field][:,idx],blocks['cv'][field][:,idx])
    valid=a['candidate_valid'];ref=int(valid.argmax());out={'current':current_prefix.copy()}
    for name,f in ff.items():
        wx=np.concatenate([f,np.broadcast_to(f[ref],f.shape),f-f[ref]],-1)
        out[name]=np.concatenate([current_prefix,wx],-1).astype(np.float32);out[name][~valid]=0
        np.testing.assert_array_equal(out[name][:,:1260],current_prefix)
        assert not wx[ref,286:].any()
    olderr=None
    if old_world_features is not None:
        olderr=float(abs(out['hybrid']-old_world_features).max(initial=0))
    receipt=dict(reference=ref,objects=len(a['actor_valid']),active_objects=int(a['actor_valid'].sum()),
        missing_active=0,source_counts={s:int(((a['sources']==s)&a['actor_valid']).sum())for s in sorted(set(a['sources']))},
        old_Hybrid_feature_maxerror=olderr,cv_hybrid_maxdiff=float(abs(out['cv']-out['hybrid']).max(initial=0)),
        changed_future_columns=int(np.any(out['cv'][:,1260:]!=out['hybrid'][:,1260:],axis=0).sum()),
        current_prefix_exact=True,shared_static_exact=True,seconds=seconds,map=ainfo,
        world_sha=array_sha(a['world']),cv_world_sha=array_sha(cv['world']),current_sha=array_sha(a['current']),
        support_sha=array_sha(a['actor_valid']),times_s=times.tolist())
    return out,receipt,dict(**{'hybrid_'+k:v for k,v in trace['hybrid'].items()},**{'cv_'+k:v for k,v in trace['cv'].items()}),cv

def export_train(limit=None):
    reg=json.load(open(B25/'train/manifest.json'));rows=reg['rows'][:limit]if limit else reg['rows'];t0=time.monotonic()
    for i,r in enumerate(rows):
        check_time();sid=r['scene_id'];src=B25/'train/scenes'/sid;out=BASE/'features/train'/sid
        if(out/'receipt.json').exists():continue
        a=dict(np.load(src/'world_assets.npz'));cur=json.load(open(src/'current.json'));old=np.load(src/'matched_features_v02.npz')
        x,rec,trace,cv=representations(a,cur,old['current'],old['world']);out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'features.npz',**x,**trace)
        if i<4:
            np.savez_compressed(out/'assets.npz',**a);np.savez_compressed(out/'cv_assets.npz',**cv);write(out/'current.json',cur)
        write(out/'receipt.json',rec|dict(complete=True,scene_id=sid,drive=r['drive'],fold=r['fold'],created_utc=now(),source_assets_sha=sha(src/'world_assets.npz'),features_sha=sha(out/'features.npz'),source_sha=sha(__file__)))
        if i<4 or (i+1)%20==0:print('R16_FEATURE',i+1,sid,rec['old_Hybrid_feature_maxerror'],rec['cv_hybrid_maxdiff'],round(time.monotonic()-t0,1),flush=True)
    if not limit:
        z=dict(np.load(B25/'train/matched_dataset_v02.npz'));names=['current','cv','hybrid'];new={k:[]for k in names};rr=[]
        for r in reg['rows']:
            d=BASE/'features/train'/r['scene_id'];f=np.load(d/'features.npz');valid=np.load(B25/'train/scenes'/r['scene_id']/'features.npz')['valid']
            for name in names:new[name].append(f[name][valid])
            rr.append(json.load(open(d/'receipt.json')))
        for name in names:new[name]=np.concatenate(new[name])
        np.testing.assert_array_equal(new['current'],z['current'])
        z.pop('world');z.update(new);out=BASE/'train';out.mkdir(exist_ok=True);np.savez_compressed(out/'dataset.npz',**z)
        write(out/'records.json',json.load(open(B25/'train/matched_scene_records_v02.json')))
        write(out/'feature_receipts.json',rr);write(out/'freeze.json',dict(complete=True,created_utc=now(),dataset_sha=sha(out/'dataset.npz'),source_dataset_sha=sha(B25/'train/matched_dataset_v02.npz'),scenes=len(rows),drives=len(set(z['drive'])),rows=len(z['U']),dimensions={k:z[k].shape[1]for k in names},seconds=time.monotonic()-t0,current_prefix_bitexact=True,labels_changed=False,world_policy='common original support/source/current; CV holds yaw,dimensions,velocity and extrapolates xy at .5s; Hybrid frozen reference world; all143 relations recomputed',source_sha=sha(__file__)))
        print('R16_FEATURE_COMPLETE',len(rows),round(time.monotonic()-t0,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;export_train(a.limit)
