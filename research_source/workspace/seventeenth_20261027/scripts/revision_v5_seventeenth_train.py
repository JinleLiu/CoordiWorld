"""B17 matched progress-head experiments, fixed TRAIN folds and budget."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,datetime as dt
from pathlib import Path
import numpy as np,xgboost as xgb
from revision_v5_sixteenth_common import *
from revision_v5_sixteenth_moments import fit_row_weights,targets
from revision_v5_seventeenth_math import centered_objective_scale
ROOT=Path(__import__('coordiworld_paths').workspace_path(''));BASE=ROOT/'seventeenth_20261027';OLD=ROOT/'sixteenth_20261027';HEADS=['q','qp','qttc','qlk']
def objective_factory(target,groups,weight,lam):
    _,inv,cnt=np.unique(groups,return_inverse=True,return_counts=True);scale=centered_objective_scale(groups,weight,lam);h=scale*weight*(1+lam*(cnt[inv]>1));history=[]
    def compute(p):
        e=p.astype(float)-target;mean=np.bincount(inv,weights=e,minlength=len(cnt))/cnt;ce=e-mean[inv];loss=.5*scale*np.sum(weight*(e*e+lam*ce*ce));grad=scale*weight*(e+lam*ce);return float(loss),grad
    def obj(p,d):
        assert len(d.get_weight())==0,'Custom weights only once';loss,g=compute(p);assert np.isfinite(g).all()and np.isfinite(h).all();history.append(dict(iteration=len(history),loss=loss,gradient_max=float(abs(g).max()),hessian_mean=float(h.mean())));return g.astype(np.float32),h.astype(np.float32)
    return obj,compute,scale,history
def compose(raw,p0,h,mode):
    q=np.clip(raw['q'].astype(float),0,1);u=raw['qp'].astype(float)
    if mode in ['PR','PRC']:u=p0*q+u
    p=np.clip(u,0,q);t=np.clip(raw['qttc'].astype(float),0,q);l=np.clip(raw['qlk'].astype(float),0,q);return (5*p+5*t+2*l+2*h*q)/14,dict(q=q,qp=p,qttc=t,qlk=l),u
def predictions(models,x,xold,mode,p0,h,valid,geo):
    raw={};shas={}
    for name,(path,old_input)in models.items():
        mm=xgb.Booster();mm.load_model(path);xx=xold if old_input else x;raw[name]=mm.inplace_predict(xx.reshape(-1,xx.shape[-1])).reshape(valid.shape);shas[name]=sha(path)
    score,proj,before=compose(raw,p0,h,mode);eligible=valid.copy()
    if 'nc_event'in raw:
        ix=np.arange(len(valid));eligible&=(raw['nc_event']<=raw['nc_event'][ix,geo,None])&(raw['dac_event']<=raw['dac_event'][ix,geo,None]);assert eligible[ix,geo].all()
    choice=choose(score,eligible)
    return dict(scores=score,choice=choice,eligible=eligible,qp_before_clip=before,**{'raw_'+k:v for k,v in raw.items()},**{'projected_'+k:v for k,v in proj.items()}),shas
def diagnostics(pred,z,ix,p0):
    c=z['components'][ix].astype(float);v=z['valid'][ix];d=z['drives'][ix];q=c[...,:4].prod(-1);y=q*c[...,4];p=pred['projected_qp'];e=p-y;means=(e*v).sum(1)/v.sum(1);relative=((e-means[:,None])**2*v).sum(1)/v.sum(1)
    selected=pred['choice'];geo=z['choice_Geometry_HC'][ix];ii=np.arange(len(ix));a=c[ii,selected];b=c[ii,geo];pure=(a[:,[0,1,2,3,5,6,7]]==b[:,[0,1,2,3,5,6,7]]).all(1)&(a[:,4]<b[:,4]);du=z['utility'][ix][ii,selected].astype(float)-z['utility'][ix][ii,geo].astype(float)
    useful=(a[:,5]>b[:,5])&(a[:,4]<b[:,4])&(du>0)
    return dict(QP_MSE=macro((e*e*v).sum(1)/v.sum(1),d),QP_centered_MSE=macro(relative,d),pure_P_loss_scenes=int(pure.sum()),pure_P_loss_drives=len(set(d[pure])),pure_P_loss_points=100*macro(np.where(pure,du,0),d),pure_P_ids=z['scene_ids'][ix][pure].tolist(),beneficial_TTC_P_scenes=int(useful.sum()),beneficial_TTC_P_ids=z['scene_ids'][ix][useful].tolist(),QP_clip_count=int((v&(abs(pred['qp_before_clip']-p)>0)).sum()),QP_clip_magnitude_mean=float(abs(pred['qp_before_clip']-p)[v].mean()),p0_known_source='progress/P16_*; prior is causal proxy, not trueP')
def resolve_base(recipe,phase,fold):
    sub='fit'if phase=='final'else'fold'+str(fold)
    if recipe.get('event_base'):
        p=BASE/'runs'/recipe['event_base']/phase/sub;binding=json.load(open(p/'bindings.json'));return {k:(Path(v['path']),v['old_input'])for k,v in binding.items()if k in HEADS}
    p=OLD/'runs/J4SR'/phase/sub;return {h:(p/h/'model.ubj',True)for h in ['q','qttc','qlk']}
def fit(recipe,fitix,holdix,x,xold,p0,z,folder,params,phase,fold,deadline):
    assert time.time()<deadline-3600,'reserve one hour';folder.mkdir(parents=True,exist_ok=False);start=time.monotonic();order=np.array(sorted(fitix,key=lambda i:str(z['scene_ids'][i])));v=z['valid'][order];xx=x[order][v];rw=fit_row_weights(z['drives'][order],v);tt=targets(z['components'][order]);targetqp=tt['qp']if recipe['qp_mode']not in ['PR','PRC']else tt['qp']-tt['q']*p0[order]
    labels={'qp':targetqp,'q':tt['q'],'qttc':tt['qttc'],'qlk':tt['qlk'],'nc_event':(z['components'][order,:,0]<1).astype(float),'dac_event':(z['components'][order,:,1]<1).astype(float)}
    heads=['nc_event','dac_event']if recipe.get('event_base')else(HEADS if recipe.get('update_joint')else['qp']);models=resolve_base(recipe,phase,fold)
    for path,old_input in models.values():
        layout=np.load(path.parent.parent/'fit_layout.npz');np.testing.assert_array_equal(layout['indices'],order)
    groups=np.repeat(np.arange(len(order)),v.sum(1));ref=xgb.QuantileDMatrix(xx,weight=rw,max_bin=256,nthread=1)
    np.savez_compressed(folder/'fit_layout.npz',indices=order,scene_ids=z['scene_ids'][order],drives=z['drives'][order],candidate_ids=np.broadcast_to(np.arange(12),v.shape)[v],row_weight=rw,groups=groups,p0=p0[order][v],**{k:labels[k][v].astype(np.float32)for k in heads})
    receipts=[]
    for h in heads:
        hd=folder/h;hd.mkdir();event=h.endswith('_event');custom=h=='qp'and recipe['qp_mode']in ['PC','PRC'];y=labels[h][v].astype(np.float32);par=params|dict(objective='binary:logistic'if event else'reg:squarederror',max_leaves=7 if event else 15,base_score=.5 if event else 0)
        dm=xgb.QuantileDMatrix(xx,label=y,weight=None if custom else rw,ref=ref,max_bin=256,nthread=1);ind,cut=dm.get_quantile_cut();refind,refcut=ref.get_quantile_cut();np.testing.assert_array_equal(ind,refind);np.testing.assert_array_equal(cut,refcut)
        np.savez_compressed(hd/'quantile_cuts.npz',indptr=ind,cut=cut);obj=None;scale=1.;history=[];compute=None
        if custom:obj,compute,scale,history=objective_factory(y,groups,rw,1.)
        else:np.testing.assert_array_equal(dm.get_weight(),rw.astype(np.float32))
        model=xgb.train(par,dm,num_boost_round=150,obj=obj,verbose_eval=False);model.save_model(hd/'model.ubj');dump=[json.loads(t)for t in model.get_dump(dump_format='json')]
        def leaves(n):return 1 if'leaf'in n else sum(leaves(j)for j in n['children'])
        nl=[leaves(t)for t in dump];assert len(nl)==150 and max(nl)<=par['max_leaves'];write(hd/'tree_dump.json',dump);write(hd/'booster_config.json',json.loads(model.save_config()));models[h]=(hd/'model.ubj',False)
        tail=model.inplace_predict(xx);write(hd/'objective_history.json',dict(custom=custom,pair_lambda=1 if custom else 0,scale=scale,iterations=history,final_loss=compute(tail)[0]if compute else None,shared_weighted_quantile_cuts=True,weights_in_objective_only=custom,mean_upper_hessian=float(rw.mean()),coupled_objective_diagonal_upper_not_official_convergence_guarantee=custom))
        receipts.append(dict(head=h,parameters=par,model_sha256=sha(hd/'model.ubj'),trees=150,leaves=sum(nl),splitting_trees=sum(t>1 for t in nl),target_min=float(y.min()),target_max=float(y.max()),fit_scale=scale))
    write(folder/'bindings.json',{k:dict(path=str(p),old_input=o,sha256=sha(p))for k,(p,o)in models.items()})
    fp,_=predictions(models,x[order],xold[order],recipe['qp_mode'],p0[order],z['features'][order,:,197],v,z['choice_Geometry_HC'][order]);np.savez_compressed(folder/'fit_predictions.npz',indices=order,**fp)
    receipt=dict(recipe=recipe,fit_scenes=len(order),fit_drives=len(set(z['drives'][order])),rows=int(v.sum()),row_weight_mean=float(rw.mean()),heads=receipts,fit_metrics=metrics(fp['choice'],z,order),fit_diagnostics=diagnostics(fp,z,order,p0),seconds=time.monotonic()-start,dimensions=x.shape[-1],source={p.name:sha(p)for p in Path(__file__).parent.glob('revision_v5_seventeenth*.py')},cut_grid='one weighted fit-only QuantileDMatrix reference shared across all losses; custom training DMatrix has no weights')
    hp=None
    if len(holdix):
        hp,_=predictions(models,x[holdix],xold[holdix],recipe['qp_mode'],p0[holdix],z['features'][holdix,:,197],z['valid'][holdix],z['choice_Geometry_HC'][holdix]);np.savez_compressed(folder/'hold_predictions.npz',indices=holdix,**hp);receipt.update(hold_metrics=metrics(hp['choice'],z,holdix),hold_diagnostics=diagnostics(hp,z,holdix,p0))
    write(folder/'receipt.json',receipt);print('FIT',recipe['id'],phase,fold,receipt['fit_metrics'],receipt.get('hold_metrics'),flush=True);return hp,receipt
def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--phase',choices=['oof','robust','final'],default='oof');p.add_argument('--health-only',action='store_true');a=p.parse_args();assert xgb.__version__=='2.1.4';reg=json.load(open(BASE/'REGISTRATION.json'));recipe=next(r for r in reg['recipes']if r['id']==a.recipe);deadline=dt.datetime.fromisoformat(json.load(open(BASE/'START.json'))['deadline_utc']).timestamp()
    if a.phase!='oof':assert json.load(open(BASE/'PRIMARY_FREEZE.json'))['recipe_id']==a.recipe or recipe['id']in json.load(open(BASE/'PRIMARY_FREEZE.json')).get('dependencies',[])
    z=dict(np.load(OLD/'train_inputs.npz'));s=np.load(OLD/'features/S64_train.npz');p16=np.load(BASE/'progress/P16_train.npz');np.testing.assert_array_equal(z['scene_ids'],p16['scene_ids']);np.testing.assert_array_equal(z['scene_ids'],s['scene_ids']);xold=np.concatenate([pair_features(z['features'],z['valid']),s['features']],-1);x=np.concatenate([xold,p16['features']],-1).astype(np.float32);p0=p16['p0'];out=BASE/'runs'/a.recipe/a.phase;out.mkdir(parents=True,exist_ok=True)
    assignment=json.load(open(OLD/'drive_folds.json'))
    if a.phase=='robust':assignment={d:i%3 for i,d in enumerate(sorted(set(z['drives']),key=lambda d:hashlib.sha256(('B16_ROBUST_V1|'+d).encode()).hexdigest()))}
    registration=dict(recipe=recipe,phase=a.phase,assignment=assignment,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),P16_sha256=sha(BASE/'progress/P16_train.npz'),S64_sha256=sha(OLD/'features/S64_train.npz'),train_input_sha256=sha(OLD/'train_inputs.npz'),deadline=deadline,health_then_resume_same_folds=True)
    if (out/'registration.json').exists():
        previous=json.load(open(out/'registration.json'));assert {k:v for k,v in previous.items()if k!='created_utc'}=={k:v for k,v in registration.items()if k!='created_utc'}
        write(out/('resume_'+dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')+'.json'),dict(created_utc=registration['created_utc'],reason='resume validated health fold; reuse existing fold without fitting again',source_sha256=sha(__file__)))
    else:write(out/'registration.json',registration)
    if a.phase=='final':
        _,rr=fit(recipe,np.arange(len(x)),np.array([],int),x,xold,p0,z,out/'fit',reg['parameters'],a.phase,None,deadline);write(out/'result.json',dict(complete=True,recipe=recipe,phase='final',fit=rr));return
    folds=np.array([assignment[str(d)]for d in z['drives']]);pred=None;receipts=[]
    for k in range(3):
        hi=np.flatnonzero(folds==k);fi=np.flatnonzero(folds!=k);assert not set(z['drives'][hi])&set(z['drives'][fi]);folder=out/('fold'+str(k))
        if (folder/'receipt.json').exists():rr=json.load(open(folder/'receipt.json'));np.testing.assert_array_equal(np.load(folder/'fit_layout.npz')['indices'],sorted(fi,key=lambda i:str(z['scene_ids'][i])));hp=dict(np.load(folder/'hold_predictions.npz'));np.testing.assert_array_equal(hp.pop('indices'),hi)
        else:hp,rr=fit(recipe,fi,hi,x,xold,p0,z,folder,reg['parameters'],a.phase,k,deadline)
        receipts.append(rr)
        if pred is None:pred={n:np.zeros((len(x),*v.shape[1:]),v.dtype)for n,v in hp.items()}
        for n,vv in hp.items():pred[n][hi]=vv
        if a.health_only:print('HEALTH_DONE',a.recipe,flush=True);return
    np.savez_compressed(out/'oof.npz',scene_ids=z['scene_ids'],drives=z['drives'],fold=folds,**pred);m=metrics(pred['choice'],z);oldchoice=np.load(OLD/'runs/J4SR'/('robust'if a.phase=='robust'else'oof')/'oof.npz')['choice'];comparisons={n:comparison(pred['choice'],z['choice_'+n],z)for n in ['Geometry_HC','B14_reg','B14_rank','B15_bestset']};comparisons['B16_J4SR']=comparison(pred['choice'],oldchoice,z)
    result=dict(complete=True,recipe=recipe,phase=a.phase,metrics=m,gate=gate(m),comparisons=comparisons,diagnostics=diagnostics(pred,z,np.arange(len(x)),p0),fit_metrics=[r['fit_metrics']for r in receipts],fold_metrics=[r['hold_metrics']for r in receipts],seconds=sum(r['seconds']for r in receipts),full_EPDMS=False,independent_confirmation=False);write(out/'result.json',result);print('OOF_RESULT',a.recipe,a.phase,m,'GATE',gate(m),flush=True)
if __name__=='__main__':main()
