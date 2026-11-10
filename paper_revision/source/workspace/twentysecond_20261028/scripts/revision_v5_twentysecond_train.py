"""Matched coordinate repair of frozen IC0/QR: no protocol or label changes."""
from revision_v5_twentysecond_common import *
import argparse,time,numpy as np,xgboost as xgb
from revision_v5_eighteenth_common import load
from revision_v5_eighteenth_projection import project
from revision_v5_eighteenth_policy import Policy
from revision_v5_twentieth_route_signal_policy import apply_local_q
from revision_v5_sixteenth_common import metrics,comparison
from revision_v5_sixteenth_moments import fit_row_weights
from revision_v5_seventeenth_train import objective_factory
HEADS=['q','qp','qttc','qlk','nc_rank','dac_event','qr']
def inputs(a,legacy=False):
 x=a['old_x674']if legacy else a['x674'];sf=a['old_route46']if legacy else a['route46'];q=np.concatenate([x,a['V24']],-1).astype(np.float32)
 return dict(q=q,qp=x,qttc=x,qlk=x,nc_rank=x,dac_event=np.concatenate([x,a['R8']],-1).astype(np.float32),qr=np.concatenate([q,sf],-1).astype(np.float32))
def decide(raw,a,ix):
 pr=project(raw);ss=(5*pr[...,1]+5*pr[...,2]+2*pr[...,3]+2*a['features198'][ix,:,197]*pr[...,0])/14;ii=np.arange(len(ix));geo=a['geo'][ix];el=a['valid'][ix]&(raw['nc_rank']>=raw['nc_rank'][ii,geo,None])&(raw['dac_event']<=raw['dac_event'][ii,geo,None]);assert el[ii,geo].all();ch=np.where(el,ss,-np.inf).argmax(1);base=dict(eligible=el,**{'raw_'+h:raw[h]for h in HEADS[:4]});local=apply_local_q(base,a['features198'][ix],a['valid'][ix],a['route46'][ix],raw['qr']);return dict(choice=ch,QR_local_choice=local['choice'],scores=ss,QR_scores=local['scores'],eligible=el,signal_mask=local['signal_application_mask'],projected=pr,**{'raw_'+h:r for h,r in raw.items()})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['oof','robust','final'],default='oof');ap.add_argument('--legacy-health',action='store_true');ap.add_argument('--fold',type=int);args=ap.parse_args();assert xgb.__version__=='2.1.4';z,_=load();a=dict(np.load(BASE/'features/train/dataset.npz'));np.testing.assert_array_equal(a['scene_ids'],z['scene_ids']);xx=inputs(a,args.legacy_health)
 if args.legacy_health:a['route46']=a['old_route46']
 recipe='legacy_health'if args.legacy_health else'K0';out=BASE/'runs'/recipe/args.phase;out.mkdir(parents=True,exist_ok=True);assignment=json.load(open(B16/'drive_folds.json'))
 if args.phase=='robust':assignment={d:i%3 for i,d in enumerate(sorted(set(z['drives']),key=lambda d:hashlib.sha256(('B16_ROBUST_V1|'+d).encode()).hexdigest()))}
 folds=np.array([assignment[str(d)]for d in z['drives']]);parq=json.load(open(B18/'runs/QV/oof/registration.json'))['parameters'];parj=json.load(open(B17/'REGISTRATION.json'))['parameters'];parnc=json.load(open(B18/'runs/NR/oof/fold0/receipt.json'))['parameters'];pardac=json.load(open(B18/'runs/DR/oof/registration.json'))['params'];write(out/'registration.json',dict(created_utc=now(),recipe=recipe,phase=args.phase,assignment=assignment,source_sha=sha(__file__),input_sha=sha(BASE/'features/train/dataset.npz'),parameters=dict(q=parq,joint=parj,nc=parnc,dac=pardac),rounds=150,scope='TRAIN only; drive folds unchanged',weights_once=True))
 pred=None;receipts=[];started=time.monotonic()
 for k in ([None]if args.phase=='final'else([args.fold]if args.fold is not None else range(3))):
  check_time(3600);tr=np.array(sorted(np.arange(len(folds))if k is None else np.flatnonzero(folds!=k),key=lambda i:str(z['scene_ids'][i])));va=np.arange(len(folds))if k is None else np.flatnonzero(folds==k);dest=out/('fit'if k is None else'fold'+str(k));dest.mkdir(exist_ok=True)
  if(dest/'receipt.json').exists():rr=json.load(open(dest/'receipt.json'));pp=dict(np.load(dest/'hold_predictions.npz'));assert np.array_equal(pp.pop('indices'),va)
  else:
   t=time.monotonic();valid=z['valid'][tr];rw=fit_row_weights(z['drives'][tr],valid);gw=np.array([1/(z['drives'][tr]==d).sum()for d in z['drives'][tr]]);gw/=gw.mean();groups=np.repeat(np.arange(len(tr)),valid.sum(1));cc=z['components'][tr].astype(float);q=cc[...,:4].prod(-1);targets=dict(q=q,qr=q,qp=q*cc[...,4],qttc=q*cc[...,5],qlk=q*cc[...,6],nc_rank=(cc[...,0]==1).astype(float),dac_event=(cc[...,1]<1).astype(float));models={};bindings={};quantiles={h:xgb.QuantileDMatrix(xx[h][tr][valid],weight=rw,max_bin=256,nthread=1)for h in ['q','qr','qp']}
   np.savez_compressed(dest/'fit_layout.npz',indices=tr,hold_indices=va,scene_ids=z['scene_ids'][tr],drives=z['drives'][tr],row_weight=rw,query_weight=gw,groups=groups,valid=valid)
   for h in HEADS:
    hd=dest/h;hd.mkdir(exist_ok=True);obj=None;hist=[];scale=None;y=targets[h][valid].astype(np.float32)
    if h=='nc_rank':par=parnc.copy();dm=xgb.DMatrix(xx[h][tr][valid],label=y,weight=gw,group=valid.sum(1),nthread=1)
    elif h=='dac_event':par=pardac.copy();dm=xgb.DMatrix(xx[h][tr][valid],label=y,weight=rw,nthread=1)
    else:
     par=parq.copy()if h in ['q','qr']else parj|dict(objective='reg:squarederror',max_leaves=15,base_score=0);dm=xgb.QuantileDMatrix(xx[h][tr][valid],label=y,weight=None if h=='qp'else rw,ref=quantiles[h if h in ['q','qr']else'qp'],max_bin=256,nthread=1)
     if h=='qp':obj,compute,scale,hist=objective_factory(y,groups,rw,1.)
    model=xgb.train(par,dm,num_boost_round=150,obj=obj,verbose_eval=False);model.save_model(hd/'model.ubj');write(hd/'booster_config.json',json.loads(model.save_config()));write(hd/'tree_dump.json',[json.loads(t)for t in model.get_dump(dump_format='json')]);write(hd/'objective.json',dict(custom=h=='qp',scale=scale,history=hist));models[h]=model;bindings[h]=dict(sha256=sha(hd/'model.ubj'),dimensions=xx[h].shape[-1],parameters=par,rounds=150);print('B22_HEAD',recipe,args.phase,k,h,round(time.monotonic()-t,1),flush=True)
   def predict(ix):return decide({h:m.inplace_predict(xx[h][ix].reshape(-1,xx[h].shape[-1])).reshape(len(ix),12)for h,m in models.items()},a,ix)
   pp=predict(va);fp=predict(tr);np.savez_compressed(dest/'hold_predictions.npz',indices=va,**pp);np.savez_compressed(dest/'fit_predictions.npz',indices=tr,**fp);health={}
   if args.legacy_health:
    old=Policy(dict(value='QV',nc='NR',dac='DR',projection='IP',isotonic=False),args.phase,k);op=old.predict(a['old_x674'][va],a['features198'][va],a['valid'][va],a['geo'][va],dict(V24=a['V24'][va],R8=a['R8'][va]));health=dict(old_choices_equal=bool(np.array_equal(pp['choice'],op['choice'])),old_score_max_error=float(abs(pp['scores']-op['scores']).max()),head_model_sha_equal={h:bindings[h]['sha256']==b['sha256']for h,b in old.bindings.items()});assert health['old_choices_equal']and health['old_score_max_error']==0,health
   rr=dict(complete=True,models=bindings,fit_scenes=len(tr),hold_scenes=len(va),fit_metrics=metrics(fp['choice'],z,tr),hold_metrics=metrics(pp['choice'],z,va),fit_QR=metrics(fp['QR_local_choice'],z,tr),hold_QR=metrics(pp['QR_local_choice'],z,va),health=health,seconds=time.monotonic()-t);write(dest/'receipt.json',rr)
  receipts.append(rr)
  if pred is None:pred={n:np.zeros((len(folds),*v.shape[1:]),v.dtype)for n,v in pp.items()}
  for n,v in pp.items():pred[n][va]=v
  print('B22_FOLD',recipe,args.phase,k,rr['hold_metrics'],flush=True)
 if args.fold is not None:return
 np.savez_compressed(out/('oof.npz'if args.phase!='final'else'train_predictions.npz'),scene_ids=z['scene_ids'],drives=z['drives'],fold=folds,**pred)
 result=dict(complete=True,recipe=recipe,phase=args.phase,metrics=metrics(pred['choice'],z),QR_metrics=metrics(pred['QR_local_choice'],z),vsGeometry=comparison(pred['choice'],z['choice_Geometry_HC'],z),receipts=receipts,seconds=time.monotonic()-started)
 if args.phase!='final':old=np.load(B21/'train_H95'/args.phase/'oof.npz');result.update(vsIC0=comparison(pred['choice'],old['IC0_choice'],z),QRvsOldQR=comparison(pred['QR_local_choice'],old['QR_local_choice'],z),changes=int((pred['choice']!=old['IC0_choice']).sum()))
 write(out/'result.json',result);print('B22_RESULT',recipe,args.phase,result['metrics'],result.get('changes'),flush=True)
if __name__=='__main__':main()
