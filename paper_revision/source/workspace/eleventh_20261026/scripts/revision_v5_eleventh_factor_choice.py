"""Frozen-factor ablation: training-only transfer and actor-relation features.
Same 150-round HGB capacity, three drive-held-out TRAIN folds and selector gate.
"""
import argparse
import datetime as dt
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from revision_v5_eleventh_choice_common import *


def pair_features(x, valid):
    reference=x[np.arange(len(x)),valid.argmax(1),None]
    anchor=np.broadcast_to(reference,x.shape)
    return np.concatenate((x,anchor,x-anchor),-1)


def train(x, valid, truth, drives, ix):
    mask=valid[ix];xx=x[ix][mask];pp=pair_features(x[ix],mask)[mask]
    refs=mask.argmax(1);ii=np.arange(len(ix));yy=truth['utility'][ix]
    delta=yy-yy[ii,refs,None]
    w=np.broadcast_to((scene_weights(drives[ix])/mask.sum(1))[:,None],mask.shape)[mask]*mask.sum()
    kw=dict(max_iter=150,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10.,early_stopping=False,random_state=17)
    models={}
    targets=dict(relative=delta[mask],absolute=yy[mask],progress=truth['components'][ix,:,4][mask])
    for name,y in targets.items():
        model=HistGradientBoostingRegressor(**kw)
        model.fit(pp if name=='relative' else xx,y,sample_weight=w);models[name]=model
    for j in range(2):
        y=truth['events'][ix,:,j][mask]
        if len(set(y))<2:models['event'+str(j)]=float(y[0])
        else:
            model=HistGradientBoostingClassifier(**kw);model.fit(xx,y,sample_weight=w);models['event'+str(j)]=model
    return models


def pred(models,x,valid):
    shape=valid.shape;dim=x.shape[-1];xx=x.reshape(-1,dim);pp=pair_features(x,valid).reshape(-1,3*dim)
    ref=valid.argmax(1);ii=np.arange(len(x))
    direct=models['absolute'].predict(xx).reshape(shape);relative=models['relative'].predict(pp).reshape(shape)
    direct-=direct[ii,ref,None];relative-=relative[ii,ref,None]
    probability=np.full((*shape,10),.5)
    probability[...,4]=models['progress'].predict(xx).reshape(shape).clip(1e-6,1-1e-6)
    for j in range(2):
        model=models['event'+str(j)]
        probability[...,8+j]=model if isinstance(model,float) else model.predict_proba(xx)[:,1].reshape(shape)
    probability=probability.clip(1e-6,1-1e-6);logits=np.log(probability/(1-probability))
    return dict(direct_tree=direct,relative_tree=relative),logits


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--folds',required=True);p.add_argument('--variant',choices=['expanded_plain','expanded_relation','original_relation','expanded_map_relation'],required=True);p.add_argument('--base-count',type=int,default=502);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    with np.load(a.dataset) as z:
        f=z['features'];relation=z['relation'];relation_available=z['relation_available'];geometry=z['shared_geometry'];v=z['valid'];d=z['drives'];ids=z['scene_ids'];truth={k:z[k] for k in ['utility','components','events']};roles={k[5:]:z[k] for k in z.files if k.startswith('role_')}
    if a.variant=='original_relation':roles['train']=roles['train'][roles['train']<a.base_count]
    if 'relation' in a.variant:
        assert relation_available.all(),'Incomplete relation cache: do not silently change evaluation denominator'
        f=np.concatenate((f,relation),-1)
    if 'map' in a.variant:f=np.concatenate((f,np.broadcast_to(geometry[:,None],(*f.shape[:3],geometry.shape[-1]))),-1)
    assignment=json.loads(Path(a.folds).read_text())['folds'];folds=np.array([assignment.get(x,-1) for x in d]);ti=roles['train']
    write(out/'pre_registration.json',dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),dataset_sha256=sha(a.dataset),source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('revision_v5_eleventh_choice_common.py')]},
        variant=a.variant,train_scenes=len(ti),features_dimension=f.shape[-1],hypothesis='Hold candidate pool and B evaluation labels fixed; test more TRAIN data, then actor relation semantics, with matched no-world/CTRA/factual/full worlds',
        triggered_after='Original304-scene neural and HGB OOF failed; training label transfer candidate identities verified; known cross-source teacher differences disclosed',
        parameters=dict(max_iter=150,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=17),
        models='one relative-gain regressor, one joint-utility regressor, one progress regressor, two event classifiers',
        normalization='no fitted scaling needed; all training groups isolated from OOF, CAL and probe/DEV',
        weighting='drive/scene/valid candidate equal; normalized to mean sample weight1',
        folds=assignment,modes=MODES,policy_grid=[.005,.02,.05],selector='same TRAIN OOF-only rule as neural pilot',
        deterministic_fit=True,number_of_independent_training_seeds_claimed=1,full_EPDMS=False,old_development=True,old_B_evaluation_targets_unchanged=True))
    results=[]
    for mode in MODES:
        if dt.datetime.now(dt.timezone.utc).isoformat()>'2026-10-26T08:35:00+00:00':
            raise RuntimeError('Budget reserve reached; partial results retained')
        x=world_features(f,mode)
        if mode=='no_world' and 'relation' in a.variant:x[...,188:284]=0
        oof={k:np.zeros(v.shape) for k in ['direct_tree','relative_tree']};ol=np.zeros((*v.shape,10))
        for fold in range(3):
            fit=ti[folds[ti]!=fold];hold=ti[folds[ti]==fold]
            models=train(x,v,truth,d,fit);gg,ll=pred(models,x[hold],v[hold])
            for k in gg:oof[k][hold]=gg[k]
            ol[hold]=ll
            print(json.dumps(dict(mode=mode,fold=fold,seconds=time.monotonic()-start)),flush=True)
        models=train(x,v,truth,d,ti);gg,ll=pred(models,x,v)
        joblib.dump(models,out/(mode+'_models.joblib'))
        np.savez_compressed(out/(mode+'_predictions.npz'),**gg,logits=ll)
        np.savez_compressed(out/(mode+'_oof.npz'),**{k:val[ti] for k,val in oof.items()},logits=ol[ti],indices=ti)
        for family in gg:
            grid=[]
            for margin in [.005,.02,.05]:
                sel=choose(oof[family][ti],ol[ti],v[ti],margin)
                ev=evaluate(sel,oof[family][ti],ol[ti],{k:z[ti] for k,z in truth.items()},v[ti],d[ti])
                okay=ev['gain_points']>=.5 and ev['NC']<=ev['reference_NC'] and ev['DAC']<=ev['reference_DAC'] and ev['progress_delta']>=-.005
                grid.append(dict(margin=margin,eligible=okay,evaluation=ev))
            eligible=[g for g in grid if g['eligible']]
            margin=max(eligible,key=lambda g:(g['evaluation']['gain_points'],-g['evaluation']['switches'],g['margin']))['margin'] if eligible else None
            write(out/(mode+'_'+family+'_selector.json'),dict(mode=mode,family=family,margin=margin,grid=grid))
            gain=gg[family];selected=choose(gain,ll,v,margin) if margin is not None else v.argmax(1)
            ci=roles['head_calibration'];cal=reporting_calibration(ll[ci],selected[ci],truth['events'][ci],d[ci]);evs={};raw={}
            for role,ix in roles.items():
                yy={k:z[ix] for k,z in truth.items()}
                evs[role]=evaluate(selected[ix],gain[ix],ll[ix],yy,v[ix],d[ix],cal)
                rawsel=np.where(v[ix],gain[ix],-np.inf).argmax(1)
                raw[role]=evaluate(rawsel,gain[ix],ll[ix],yy,v[ix],d[ix])
            results.append(dict(mode=mode,family=family,seed=17,margin=margin,evaluations=evs,raw_argmax_evaluations=raw,calibration=cal))
            print(json.dumps(dict(mode=mode,family=family,margin=margin,probe_gain=evs['probe']['gain_points'],probe_NC=evs['probe']['NC'],probe_DAC=evs['probe']['DAC'],probe_switches=evs['probe']['switches'],DEV_gain=evs['development_validation']['gain_points'],raw_probe_gain=raw['probe']['gain_points'],seconds=time.monotonic()-start)),flush=True)
        write(out/'partial_result.json',dict(complete=False,rows=results,seconds=time.monotonic()-start))
    write(out/'result.json',dict(complete=True,rows=results,seconds=time.monotonic()-start,qualified=False,full_EPDMS=False))


if __name__=='__main__':main()
