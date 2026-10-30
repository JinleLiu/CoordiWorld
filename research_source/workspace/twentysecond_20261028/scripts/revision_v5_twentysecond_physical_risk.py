"""Matched filtered vs physical-NC BCE, shared continuous-supply pool."""
from revision_v5_twentysecond_common import *
from revision_v5_twentysecond_train import inputs
from revision_v5_sixteenth_moments import fit_row_weights
from revision_v5_sixteenth_common import metrics,comparison
import numpy as np,xgboost as xgb,argparse,time

def select(base,a,risk,absolute=False):
 ii=np.arange(len(risk));geo=a['geo'];dac=a['valid']&(base['raw_dac_event']<=base['raw_dac_event'][ii,geo,None]);el=dac&(risk<=risk[ii,geo,None]);has=np.zeros(len(risk),bool)
 if absolute:
  safe=dac&(risk<=.5);has=safe.any(1);el=np.where(has[:,None],safe,el)
 assert el.any(1).all();choice=np.where(el,base['scores'],-np.inf).argmax(1);return dict(choice=choice,eligible=el,risk=risk,absolute_available=has)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['oof','robust','final'],default='oof');args=ap.parse_args();phase=args.phase;out=BASE/'runs/physical_NC'/phase;out.mkdir(parents=True,exist_ok=True);z=dict(np.load(BASE/'supply55/dataset.npz'));a=dict(np.load(BASE/'supply55/features_identity_fixed.npz'));c=np.load(BASE/'supply55/C32.npz')['features'];xx=np.concatenate([a['x674'],c],-1).astype(np.float32);lab=np.load(BASE/'raw_nc/labels.npz');np.testing.assert_array_equal(z['scene_ids'],lab['scene_ids']);physical=lab['physical_NC'].copy()
 for r in json.load(open(BASE/'supply55/input_freeze.json'))['records']:
  nr=json.load(open(BASE/'supply55/scenes'/r['scene_id']/'native.json'));physical[r['index'],r['new_slot']]=nr['calls'][str(r['new_slot'])][0]['raw']['no_at_fault_collisions']
 base=dict(np.load(BASE/'runs/SK0'/phase/('train_predictions.npz'if phase=='final'else'oof.npz')));folds=base['fold'];par=json.load(open(B18/'runs/DR/oof/registration.json'))['params'];target={'filtered':(z['components'][...,0]<1).astype(float),'physical':(physical<1).astype(float)};n=len(folds);pred={h:np.zeros((n,12),np.float32)for h in target};receipts=[];t=time.monotonic();record('physical_NC_'+phase,'Original human-filtered NC supervision disagrees with raw hazard and synthetic downstream NC','Two matched BCE706 heads filtered vs raw; shared SK0 value/DAC; relativeGeometry and fixed .5-class qualification, no grid','TRAIN1755/40 shared supply pool '+phase);write(out/'registration.json',dict(created_utc=now(),source_sha=sha(__file__),parameters=par,rounds=150,label_sha=sha(BASE/'raw_nc/labels.npz'),supply_labels_sha=sha(BASE/'supply55/dataset.npz'),input_sha=sha(BASE/'supply55/features_identity_fixed.npz'),weights='once drive-scene-candidate',selection_variants=['relativeGeometry','absolute_.5_ifavailable_else_relative'],main_metric='unchanged official humanfiltered',physical_target='rawNC<1 includes .5; no future feature; same oldmodel input+C32'))
 for k in ([None]if phase=='final'else range(3)):
  check_time();tr=np.array(sorted(np.arange(n)if k is None else np.flatnonzero(folds!=k),key=lambda i:str(z['scene_ids'][i])));hi=np.arange(n)if k is None else np.flatnonzero(folds==k);dest=out/('fit'if k is None else'fold'+str(k));dest.mkdir(exist_ok=True);v=z['valid'][tr];rw=fit_row_weights(z['drives'][tr],v);rr={};fp={}
  for h,y in target.items():
   hd=dest/h;hd.mkdir(exist_ok=True);dm=xgb.DMatrix(xx[tr][v],label=y[tr][v],weight=rw,nthread=1);m=xgb.train(par,dm,num_boost_round=150,verbose_eval=False);m.save_model(hd/'model.ubj');write(hd/'booster_config.json',json.loads(m.save_config()));write(hd/'tree_dump.json',[json.loads(t)for t in m.get_dump(dump_format='json')]);pred[h][hi]=m.inplace_predict(xx[hi].reshape(-1,706)).reshape(len(hi),12);fp[h]=m.inplace_predict(xx[tr].reshape(-1,706)).reshape(len(tr),12);rr[h]=dict(sha256=sha(hd/'model.ubj'),fit_Brier=float(((fp[h][v]-y[tr][v])**2).mean()),dimensions=706);print('B22_RAW_HEAD',phase,k,h,round(time.monotonic()-t,1),flush=True)
  np.savez_compressed(dest/'fit_predictions.npz',indices=tr,**fp);np.savez_compressed(dest/'layout.npz',indices=tr,hold=hi,weights=rw,valid=v);write(dest/'receipt.json',rr);receipts.append(rr)
 results={};ii=np.arange(n);v=z['valid'];np.savez_compressed(out/'head_predictions.npz',**pred,physical_NC=physical,scene_ids=z['scene_ids'],drives=z['drives'],fold=folds)
 for h,risk in pred.items():
  for absolute in [False,True]:
   name=('PNC'if h=='physical'else'FNC')+('_abs'if absolute else'_rel');pp=select(base,a,risk,absolute);ch=pp['choice'];np.savez_compressed(out/(name+'.npz'),scene_ids=z['scene_ids'],drives=z['drives'],fold=folds,**pp);results[name]=dict(metrics=metrics(ch,z),vsSK0=comparison(ch,base['choice'],z),vsGeometry=comparison(ch,z['choice_Geometry_HC'],z),physical_selected_events=int((physical[ii,ch]<1).sum()),raw_Brier=float(((risk[v]-target['physical'][v])**2).mean()),filtered_Brier=float(((risk[v]-target['filtered'][v])**2).mean()),changed=int((ch!=base['choice']).sum()));print('B22_RAW_RESULT',phase,name,results[name]['metrics'],'physicalNC',results[name]['physical_selected_events'],flush=True)
 write(out/'result.json',dict(complete=True,created_utc=now(),recipes=results,receipts=receipts,SK0_physical_events=int((physical[ii,base['choice']]<1).sum()),seconds=time.monotonic()-t))
if __name__=='__main__':main()
