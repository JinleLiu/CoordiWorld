"""Same-candidate current map transfer, controller parity, and missing-map rebuild.

No target is used to decide transfer eligibility. Initial 502 B labels remain.
New A training rows with execution mismatch are excluded from TRAIN roles but
retained in the saved dataset and audit. No evaluation row is removed.
"""
import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
import argparse,datetime as dt,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from coordiworld.data.openscene import load_log,ego_pose
from coordiworld.revision_v5.native_adapter import read_legacy_inputs
from coordiworld.revision_v5.compiled_controller import predict_vector_controller_execution
from current_map_export import export_current_map
from deployment_features import build_trace

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def static_features(e,t):
    known=t[...,11].all(1);center=e[...,:2]*100+1.461*e[...,2:4]
    distance=np.zeros(e.shape[:2]);distance[:,1:]=np.linalg.norm(np.diff(center,axis=1),axis=-1);distance[(t[...,4]>0)|(t[...,5]>0)]=0
    cum=np.concatenate([np.zeros((len(e),1)),np.cumsum(distance,axis=1)],1);right=np.arange(41)+1;left=np.maximum(0,np.arange(41)-10);wrong=(cum[:,right]-cum[:,left]).max(1)
    direction=np.where(wrong<2,1,np.where(wrong<6,.5,0));drivable=t[...,:4].all((1,2)).astype(float);raw=np.maximum(0,t[:,-1,9]);maximum=max(float(raw.max()),.05)
    out=np.c_[known,drivable,direction,raw,raw/maximum,wrong/100,t[...,:9].mean(1),t[...,10].mean(1),t[...,10].max(1),t[...,9].min(1)]
    out[~known]=0
    assert out.shape==(len(e),18) and np.isfinite(out).all()
    return out.astype(np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--feature-dir',required=True);p.add_argument('--transfer',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root);fd=Path(a.feature_dir);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=dt.datetime.fromisoformat('2026-10-26T08:15:00+00:00').timestamp()
 source=json.loads((Path(a.transfer).parent/'metadata.json').read_text());audit=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),feature_dir=str(fd),transfer_sha256=sha(a.transfer),code_sha256=sha(__file__),allowed_inputs='current complete100m map/current route/current lights and causal controller; shared across all world modes',parity='8 samples: max XY <=1e-4m, sin/cos yaw <=2e-5; exact valid IDs',label_filter='Only imported TRAIN cases with action-execution mismatch excluded from training role, without examining outcomes; all original B rows and all evaluation masks retained',geometry_unknown='18 zeros including known=0, no automatic risk and no candidate removal',source=source['source_sha256'])
 (out/'pre_registration.json').write_text(json.dumps(audit,indent=2)+'\n')
 while not (fd/'result.json').exists():
  if time.time()>deadline:raise RuntimeError('Feature wait deadline reached')
  time.sleep(15)
 base=dict(np.load(fd/'dataset.npz'));z=np.load(a.transfer);ex=z['execution'];tr=z['trace'];va=z['valid'];by={s:i for i,s in enumerate(z['scene_ids'])}
 inv=json.loads((root/'runs/paired_geometry_v2_train128_dev64/input_screening.json').read_text())['rows'];lookup={r['scene_id']:r for r in inv};N=len(base['scene_ids']);features=np.zeros((N,12,18),np.float32);records=[];rebuild=[];exclude=[]
 for i,sid in enumerate(base['scene_ids']):
  j=by.get(sid);err=angle=None;okay=False
  if j is not None:
   ids=np.flatnonzero(base['valid'][i]);m=base['features'][i,0,:,:64].reshape(12,8,8)
   err=float(abs(m[ids,:,:2]*50-ex[j,ids,5::5,:2]*100).max());angle=float(max(abs(m[ids,:,2]-ex[j,ids,5::5,3]).max(),abs(m[ids,:,3]-ex[j,ids,5::5,2]).max()))
   okay=np.array_equal(base['valid'][i],va[j]) and err<=1e-4 and angle<=2e-5
  if okay:features[i]=static_features(ex[j],tr[j])
  else:
   rebuild.append(i)
   if i>=502:exclude.append(i)
  records.append(dict(scene_id=str(sid),transfer_available=j is not None,transferred=bool(okay),execution_XY_error_m=err,execution_angle_sincos_error=angle,train_import_excluded=i in exclude))
 torch.set_num_threads(1);torch._dynamo.config.recompile_limit=128;last=None;frames=None
 for i in sorted(rebuild,key=lambda i:(lookup[base['scene_ids'][i]]['log_name'],lookup[base['scene_ids'][i]]['index'])):
  if time.time()>deadline:records[i]['rebuild_error']='fixed deadline';continue
  sid=base['scene_ids'][i];r=lookup[sid]
  try:
   if last!=r['log_name']:
    raw=root/'data/openscene/trainval'/(r['log_name']+'.pkl');assert sha(raw)==r['raw_log_sha256'];frames=load_log(raw);last=r['log_name']
   history=frames[r['index']-3:r['index']+1];f=history[-1];scene,meta=read_legacy_inputs(r['cache_path'],r);local,full,receipt=predict_vector_controller_execution(history,scene.candidates[0].numpy(),scene.candidate_valid[0].numpy());ids=np.flatnonzero(scene.candidate_valid[0]);origin=np.array(receipt['coordinate_origin']);co,si=np.cos(origin[2]),np.sin(origin[2]);poses=np.stack([full[int(j)][:,:3] for j in ids]);poses[...,:2]=(poses[...,:2]-origin[:2])@np.array([[co,-si],[si,co]]);poses[...,2]-=origin[2]
   assert np.max(abs(local[ids,:,:2]-poses[:,5::5,:2]))<1e-4
   dynamic=np.asarray(f['ego_dynamic_state'],float);current=dict(scene_id=str(sid),timestamp_us=int(f['timestamp']),latest_source_timestamp_us=int(f['timestamp']),map_name=f['map_location'],rear_axle_pose=ego_pose(f).tolist(),velocity=dynamic[:2].tolist(),acceleration=dynamic[2:].tolist(),route_roadblock_ids=list(f['roadblock_ids']),traffic_lights=[list(x) for x in f['traffic_lights']],source_container_sha256=r['raw_log_sha256'])
   bundle,coords,native=export_current_map(current,root/'data/maps',poses);e,t=build_trace(bundle,poses);features[i,ids]=static_features(e,t);records[i]['rebuilt']=True;records[i]['geometry_sha256']=bundle['provenance']['geometry_sha256'];records[i]['known']=bool(t[...,11].all())
   np.savez_compressed(out/(str(sid)+'_current_map.npz'),execution=e,trace=t,valid_ids=ids)
   del bundle,coords,native
  except Exception as error:records[i]['rebuild_error']=repr(error);records[i]['traceback']=traceback.format_exc()
  print(json.dumps(dict(rebuilt_scene=str(sid),known=int(features[i,:,0].sum()),error=records[i].get('rebuild_error'),seconds=time.monotonic()-start)),flush=True)
 base['shared_geometry']=features;base['role_train_before_execution_parity']=base['role_train'].copy();base['role_train']=base['role_train'][~np.isin(base['role_train'],exclude)]
 # Avoid treating the audit-only previous role as an evaluation population.
 before=base.pop('role_train_before_execution_parity');np.savez_compressed(out/'dataset.npz',**base)
 result=dict(complete=True,records=records,transferred=sum(r['transferred'] for r in records),rebuild_requested=len(rebuild),rebuild_errors=[r for r in records if 'rebuild_error' in r],train_before=len(before),train_after=len(base['role_train']),excluded_imported_train_scene_ids=base['scene_ids'][exclude].tolist(),all_evaluation_rows_retained=True,source_dataset_sha256=sha(fd/'dataset.npz'),dataset_sha256=sha(out/'dataset.npz'),seconds=time.monotonic()-start,feature_dimension=18,full_EPDMS=False)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','rebuild_errors']}),flush=True)
if __name__=='__main__':main()
