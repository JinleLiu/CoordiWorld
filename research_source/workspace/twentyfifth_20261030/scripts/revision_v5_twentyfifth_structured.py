"""Second registered matched C/W pair: joint moments and explicit risk heads.
The common Geometry reference and upstream PNC bank/history are held fixed.
"""
from revision_v5_twentyfifth_common import *
import numpy as np,xgboost as xgb,argparse,time,pandas as pd
from revision_v5_twentyfifth_matched import dataset,weights,PARAMS,macro,interval,SHORT
from revision_v5_twentysecond_primary import project
HEADS=['q','qp','qttc','qlk','physical_nc','dac_event']

def context(z):
 path=BASE/'structured/context.npz';path.parent.mkdir(exist_ok=True)
 if path.exists():return dict(np.load(path))
 hc=[];p0=[];known=[];geos=[];records=json.load(open(BASE/'train/matched_scene_records_v02.json'))
 for r in records:
  a=np.load(BASE/'train/scenes'/r['scene_id']/'features.npz');bb=np.load(BASE/'train/scenes'/r['scene_id']/'base_features.npz');ids=np.array(r['candidate_ids']);hc.extend(a['features198'][ids,197]);p0.extend(a['x674'][ids,662]);known.extend(a['x674'][ids,668]);geos.append([int(bb['geo']),int(a['geo'])])
 out=dict(HC=np.array(hc),P0=np.array(p0),P0known=np.array(known),geos=np.array(geos));np.savez_compressed(path,**out);return out

def select(raw,HC,EC,P0,known,valid,geo,candidate):
 pr=project(raw);score=(5*pr[:,1]+5*pr[:,2]+2*pr[:,3]+2*HC*pr[:,0]+2*EC*pr[:,0])/16;extra=candidate>=12;mask=extra&(known>0);score[mask]+=5/16*(pr[mask,0]*P0[mask]-pr[mask,1]);da=valid&(raw['dac_event']<=raw['dac_event'][geo]);el=da&(raw['physical_nc']<=.5)
 if not el.any():el=da&(raw['physical_nc']<=raw['physical_nc'][geo])
 el|=extra&valid&(EC>0)&(raw['physical_nc']<=.5)&(raw['dac_event']<=.5)
 assert el.any();return int(np.where(el,score,-np.inf).argmax()),score,el,pr

def fit(final=False):
 z=dataset();ct=context(z);out=BASE/'structured'/('final'if final else'oof');out.mkdir(exist_ok=True);c=z['components'];q=np.prod(c[:,:4],-1);ys=dict(q=q,qp=q*c[:,4],qttc=q*c[:,5],qlk=q*c[:,6],physical_nc=(z['raw'][:,0]<1).astype(float),dac_event=(c[:,1]<1).astype(float));pred={s:{h:np.full(len(q),np.nan)for h in HEADS}for s in ['current','world']};t=time.monotonic();assert xgb.__version__=='2.1.4'
 for fold in ([None]if final else range(3)):
  check_time();tr=np.arange(len(q))if fold is None else np.flatnonzero(z['fold']!=fold);ho=np.arange(len(q))if fold is None else np.flatnonzero(z['fold']==fold);tr=tr[np.lexsort((z['candidate'][tr],z['scene_id'][tr]))];w=weights(z['scene_id'][tr],z['drive'][tr]);d=out/('fit'if fold is None else'fold'+str(fold));d.mkdir(exist_ok=True);np.savez_compressed(d/'layout.npz',train=tr,hold=ho,weights=w)
  for side in pred:
   for head in HEADS:
    params=PARAMS.copy()
    if head in ['physical_nc','dac_event']:params.update(objective='binary:logistic',max_leaves=7,base_score=.5)
    m=xgb.train(params,xgb.DMatrix(z[side][tr],label=ys[head][tr],weight=w,nthread=1),num_boost_round=150);name=side+'_'+head;m.save_model(d/(name+'.ubj'));pred[side][head][ho]=m.inplace_predict(z[side][ho]);fp=m.inplace_predict(z[side][tr]);np.savez_compressed(d/(name+'_fit.npz'),indices=tr,predictions=fp);tab=m.trees_to_dataframe();write(d/(name+'_receipt.json'),dict(sha256=sha(d/(name+'.ubj')),config=json.loads(m.save_config()),dimensions=m.num_features(),trees=m.num_boosted_rounds(),leaves=int((tab.Feature=='Leaf').sum()),fit_weighted_MSE=float(np.average((fp-ys[head][tr])**2,weights=w))))
   print('R15_STRUCT_FIT',fold,side,round(time.monotonic()-t,1),flush=True)
 np.savez_compressed(out/'predictions.npz',**{s+'_'+h:v for s,p in pred.items()for h,v in p.items()});write(out/'registration.json',json.load(open(BASE/'STRUCTURED_PAIR_REGISTRATION.json'))|dict(source_sha=sha(__file__),dataset_sha=sha(BASE/'train/matched_dataset_v02.npz')));analyze(z,ct,pred,out);write(out/'complete.json',dict(complete=True,created_utc=now(),seconds=time.monotonic()-t))

def analyze(z,ct,pred,out):
 records=json.load(open(BASE/'train/matched_scene_records_v02.json'));rows=[]
 for ri,r in enumerate(records):
  ii=np.arange(r['row_start'],r['row_start']+r['row_count']);ids=z['candidate'][ii]
  for pi,pool in enumerate(['M12','H3']):
   valid=(ids<12)if pool=='M12'else np.ones(len(ii),bool);g=int(np.flatnonzero(ids==ct['geos'][ri,pi])[0]);oracle=float(z['U'][ii[valid]].max())
   for side,pr in pred.items():
    raw={h:p[ii]for h,p in pr.items()};ch,score,el,joint=select(raw,ct['HC'][ii],z['EC'][ii],ct['P0'][ii],ct['P0known'][ii],valid,g,ids);k=ii[ch];c=z['components'][k];rows.append(dict(method=side,pool=pool,scene_id=r['scene_id'],drive=r['drive'],fold=r['fold'],candidate=int(z['candidate'][k]),family=z['family'][k],U=z['U'][k],EC=z['EC'][k],regret=oracle-z['U'][k],rawNC=z['raw'][k,0],**dict(zip(SHORT,c))))
 df=pd.DataFrame(rows);df.to_csv(out/'per_scene.csv',index=False);df.groupby(['pool','method','drive'])[['U','P','EC','regret']].mean().reset_index().to_csv(out/'per_drive.csv',index=False);summary={};comparisons={}
 for pool,pg in df.groupby('pool'):
  summary[pool]={}
  for side,g in pg.groupby('method'):summary[pool][side]=dict(U=100*macro(g.U.to_numpy(),g.drive.to_numpy()),P=macro(g.P.to_numpy(),g.drive.to_numpy()),oracle_regret=100*macro(g.regret.to_numpy(),g.drive.to_numpy()),events={m:int((g[m]<1).sum())for m in ['NC','DAC','DDC','TLC','TTC','HC','EC','rawNC']})
  n=pg[pg.method=='world'].set_index('scene_id');o=pg[pg.method=='current'].set_index('scene_id').loc[n.index];dv=100*(n.U-o.U).groupby(n.drive).mean();ev={}
  for m in ['NC','DAC','DDC','TLC','TTC','HC','EC','rawNC']:
   ne=n[m]<1;oe=o[m]<1;ev[m]=dict(old=int(oe.sum()),new=int(ne.sum()),fixed=list(n.index[oe&~ne]),added=list(n.index[~oe&ne]))
  comparisons[pool]=dict(U=interval(dv.values),deltaP=macro((n.P-o.P).to_numpy(),n.drive.to_numpy()),events=ev,changed=int((n.candidate!=o.candidate).sum()),per_drive=dv.to_dict())
 write(out/'result.json',dict(complete=True,created_utc=now(),summary=summary,comparisons=comparisons,scope='conditional240TRAIN driveOOF, completepool',ref='Geometry identity common and frozen; not a world-free whole policy'));print('R15_STRUCT_RESULT',summary,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();fit(a.final)
