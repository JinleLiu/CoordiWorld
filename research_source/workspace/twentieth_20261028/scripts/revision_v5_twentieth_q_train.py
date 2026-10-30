"""Matched Q-head representation comparison; frozen remaining IC0 heads.

The continuous label is NC*DAC*DDC*TLC, not a binary safety probability.
All recipe decisions are TRAIN-only; development is evaluated after freeze.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,time,json,datetime as dt,hashlib
from pathlib import Path
import numpy as np,xgboost as xgb
from revision_v5_eighteenth_common import load,metrics,comparison,macro
from revision_v5_eighteenth_policy import Policy
from revision_v5_eighteenth_projection import project
from revision_v5_sixteenth_moments import fit_row_weights
ROOT=Path(__import__('coordiworld_paths').workspace_path(''));BASE=ROOT/'twentieth_20261028';OLD=ROOT/'eighteenth_20261027'
POLICY={'value':'QV','nc':'NR','dac':'DR','projection':'IP','isotonic':False}
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recipe',required=True);ap.add_argument('--phase',choices=['oof','robust','final'],default='oof');ap.add_argument('--fold',type=int);args=ap.parse_args()
    reg=json.load(open(BASE/'MAIN_REGISTRATION.json'));r=next(r for r in reg['recipes']if r['id']==args.recipe);assert xgb.__version__=='2.1.4';assert time.time()<dt.datetime.fromisoformat(json.load(open(BASE/'START.json'))['deadline_utc']).timestamp()-3600
    z,x=load();extra={k:np.load(OLD/'features'/f'{k}_train.npz')['features']for k in ['V24','R8']};a=np.load(BASE/'signal_train/features.npz');np.testing.assert_array_equal(a['scene_ids'],z['scene_ids']);f=a['features'];xx=np.concatenate([x,extra['V24'],f[:,:,:r['added_features']]],-1).astype(np.float32)
    par=json.load(open(OLD/'runs/QV/oof/registration.json'))['parameters'];assignment=json.load(open(ROOT/'sixteenth_20261027/drive_folds.json'))
    if args.phase=='robust':assignment={d:i%3 for i,d in enumerate(sorted(set(z['drives']),key=lambda d:hashlib.sha256(('B16_ROBUST_V1|'+d).encode()).hexdigest()))}
    folds=np.array([assignment[str(d)]for d in z['drives']]);out=BASE/'runs'/args.recipe/args.phase;out.mkdir(parents=True,exist_ok=True)
    assert not(out/'result.json').exists(),'do not overwrite completed run'
    write(out/'registration.json',dict(recipe=r,phase=args.phase,created_utc=now(),folds=assignment,parameters=par,dimensions=xx.shape[-1],rounds=150,source_sha=sha(__file__),features_sha=sha(BASE/'signal_train/features.npz'),pool_sha=sha(ROOT/'sixteenth_20261027/train_inputs.npz'),unchanged='candidate/valid/reference/world/labels/remaining5heads; argmax with original tie order',row_weight='1/(fit drive scenes * scene valid candidates), normalize fit row mean1; applied once'))
    oof={};receipts=[]
    for fold in ([None]if args.phase=='final'else[args.fold]if args.fold is not None else range(3)):
        hi=np.array([],int)if fold is None else np.flatnonzero(folds==fold);fi=np.arange(len(xx))if fold is None else np.flatnonzero(folds!=fold);fi=np.array(sorted(fi,key=lambda i:str(z['scene_ids'][i])));sub='fit'if fold is None else'fold'+str(fold);folder=out/sub;folder.mkdir(exist_ok=True);tt=time.monotonic()
        assert not(folder/'receipt.json').exists(),'no duplicate fit'
        valid=z['valid'][fi];row=fit_row_weights(z['drives'][fi],valid);label=z['components'][fi].astype(float)[...,:4].prod(-1)[valid].astype(np.float32);ref=xgb.QuantileDMatrix(xx[fi][valid],weight=row,max_bin=256,nthread=1);dm=xgb.QuantileDMatrix(xx[fi][valid],label=label,weight=row,ref=ref,max_bin=256,nthread=1)
        np.testing.assert_array_equal(dm.get_weight(),row.astype(np.float32));model=xgb.train(par,dm,num_boost_round=150,verbose_eval=False);model.save_model(folder/'model.ubj');write(folder/'booster_config.json',json.loads(model.save_config()));dump=[json.loads(t)for t in model.get_dump(dump_format='json')];write(folder/'tree_dump.json',dump)
        np.savez_compressed(folder/'fit_layout.npz',indices=fi,scene_ids=z['scene_ids'][fi],drives=z['drives'][fi],candidate_indices=np.broadcast_to(np.arange(12),valid.shape)[valid],row_weight=row,label=label)
        policy=Policy(POLICY,args.phase,fold)
        def predict(ix,part):
            old=policy.predict(x[ix],z['features'][ix],z['valid'][ix],z['choice_Geometry_HC'][ix],{k:v[ix]for k,v in extra.items()});q=model.inplace_predict(xx[ix].reshape(-1,xx.shape[-1])).reshape(len(ix),12)
            if r['added_features']==0:np.testing.assert_array_equal(q,old['raw_q'])
            raw={h:old['raw_'+h]for h in ['q','qp','qttc','qlk']};raw['q']=q;proj=project(raw);score=(5*proj[...,1]+5*proj[...,2]+2*proj[...,3]+2*z['features'][ix,:,197]*proj[...,0])/14;ch=np.where(old['eligible'],score,-np.inf).argmax(1)
            result=dict(indices=ix,choice=ch,scores=score,eligible=old['eligible'],risk=old['risk'],dac=old['dac'],old_choice=old['choice'],projected=proj,**{'raw_'+h:raw[h]for h in raw});np.savez_compressed(folder/f'{part}_predictions.npz',**result);return result
        fit=predict(fi,'fit');hold=predict(hi,'hold')if len(hi)else None
        def leaves(n):return 1 if'leaf'in n else sum(leaves(c)for c in n['children'])
        nl=[leaves(t)for t in dump];assert len(nl)==150 and max(nl)<=15
        rec=dict(recipe=r,phase=args.phase,fold=fold,fit_metrics=metrics(fit['choice'],z,fi),hold_metrics=metrics(hold['choice'],z,hi)if hold else None,trees=len(nl),leaves=sum(nl),splitting_trees=sum(n>1 for n in nl),seconds=time.monotonic()-tt,model_sha=sha(folder/'model.ubj'),old_model_bindings=policy.bindings,zero_extra_exact_Q_reproduction=r['added_features']==0)
        write(folder/'receipt.json',rec);receipts.append(rec);print('B20_FIT',args.recipe,args.phase,fold,rec['hold_metrics'],round(rec['seconds'],3),flush=True)
        if hold:
            for k,a in hold.items():
                if k=='indices':continue
                if k not in oof:oof[k]=np.zeros((len(x),*a.shape[1:]),a.dtype)
                oof[k][hi]=a
    if args.fold is not None:write(out/'health.json',dict(complete=True,receipt=receipts[0]));return
    if args.phase=='final':write(out/'result.json',dict(complete=True,recipe=r,phase='final',receipt=receipts[0]));return
    np.savez_compressed(out/'oof.npz',scene_ids=z['scene_ids'],drives=z['drives'],folds=folds,**oof);c=oof['choice'];old=oof['old_choice'];result=dict(complete=True,recipe=r,phase=args.phase,metrics=metrics(c,z),vsIC0=comparison(c,old,z),vsGeometry=comparison(c,z['choice_Geometry_HC'],z),fit_metrics=[r['fit_metrics']for r in receipts],seconds=sum(r['seconds']for r in receipts),scope='head drive OOF on developed TRAIN, same world developed; no new confirmation',receipts=receipts);write(out/'result.json',result);print('B20_OOF',args.recipe,args.phase,result['metrics'],result['vsIC0']['delta_U'],flush=True)
if __name__=='__main__':main()
