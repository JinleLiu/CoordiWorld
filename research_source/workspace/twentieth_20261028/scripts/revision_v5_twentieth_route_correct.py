"""Current-only PDM corrected route support, separate TRAIN exploration."""
import os
os.environ.setdefault("NUPLAN_MAPS_ROOT",__import__('coordiworld_paths').workspace_path('data/maps'))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,gc,hashlib
import numpy as np,shapely
from pathlib import Path
from coordiworld.data.openscene import load_log
from navsim.common.dataclasses import Scene,SensorConfig
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from revision_v5_seventeenth_progress import Extractor
from revision_v5_twentieth_q_train import ROOT,BASE,write,sha,now
from revision_v5_twentieth_signal_features import features,NAMES
class CachedRoute:
 def __init__(self,receipt,tokens):self.receipt=receipt;self.tokens=set(tokens)
 def get(self,current,execution):
  r=self.receipt.copy();r['records']=[v for v in r['records']if v['token']in self.tokens];r['known_signal_count']=sum(v['state']!='unknown'for v in r['records']);r['matched_signal_count']=r['known_signal_count'];r['route_basis']='current-only initialized PDM route_roadblock_correction';groups={k:shapely.union_all([shapely.from_wkb(bytes.fromhex(v['wkb_hex']))for v in r['records']if v['state']==k])for k in ['red','green','unknown']};return groups,r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args();out=BASE/'exploration'/('corrected_route_probe'if args.limit else'corrected_route');out.mkdir(parents=True,exist_ok=False);write(out/'registration.json',dict(created_utc=now(),observation='Uncorrected current roadblocks drop30/31 current-red event rows. Need distinguish route repair from unsupported off-route red proxy.',mechanism='Initialize same current-only PDM and apply its deterministic route correction using current ego/map; no trajectory/world/label changes. Cache corrected route lane IDs. Match six full current-only planner routes.',minimal_control='Reuse QS744 and append route-scoped signal relationship summary, or reject if no novel supported information; no DEVELOPMENT selection',continue_if='legal route export works, yields meaningful distinction, and future TRAIN comparison passes same safety sanity checks',source_sha=sha(__file__)))
 z=np.load(ROOT/'sixteenth_20261027/train_inputs.npz');inv={r['scene_id']:r for r in json.load(open(ROOT/'runs/paired_geometry_v2_train128_dev64/input_screening.json'))['rows']};indices=list(range(len(z['scene_ids'])))
 if args.limit:indices=sorted(indices,key=lambda i:hashlib.sha256(('B20_ROUTE_CHECK|'+str(z['scene_ids'][i])).encode()).hexdigest())[:args.limit]
 indices.sort(key=lambda i:(inv[str(z['scene_ids'][i])]['log_name'],inv[str(z['scene_ids'][i])]['index']));ext=Extractor();last=None;values={};records=[];start=time.monotonic()
 for n,i in enumerate(indices):
  sid=str(z['scene_ids'][i]);r=inv[sid];path=ROOT/'data/openscene/trainval'/(r['log_name']+'.pkl')
  if str(path)!=last:assert sha(path)==r['raw_log_sha256'];frames=load_log(path);last=str(path)
  history=frames[r['index']-3:r['index']+1];assert len(history)==4 and history[-1]['token']==sid
  scene=Scene.from_scene_dict_list(history,Path('.'),4,0,SensorConfig.build_no_sensors());scenario=NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0');inp,init=ext.processor._get_planner_inputs(scenario);planner=ext.processor._pdm_closed;planner.initialize(init);planner._route_roadblock_correction(inp.history.current_state[0]);gc.enable();tokens=sorted(planner._route_lane_dict);blocks=list(planner._route_roadblock_dict);assets=dict(np.load(ROOT/'sixteenth_20261027/features/assets_train'/sid/'assets.npz'));current=json.load(open(ROOT/'sixteenth_20261027/features/assets_train'/sid/'current.json'));g=json.load(open(BASE/'signal_train/scenes'/sid/'geometry.json'));a=np.load(BASE/'signal_train/scenes'/sid/'features.npz')
  if args.limit:
   ext.extract(history,assets);assert tokens==sorted(planner._route_lane_dict),'shortcut differs from full current-only planner'
  f,detail,rec=features(current,assets,a['candidates'],CachedRoute(g,tokens));np.testing.assert_array_equal(f[:,:20],a['features'][:,:20]);values[sid]=f;dest=out/'scenes'/sid;dest.mkdir(parents=True);np.savez_compressed(dest/'features.npz',features=f,**detail);record=dict(scene_id=sid,index=int(i),source_sha=r['raw_log_sha256'],history_timestamps=[int(h['timestamp'])for h in history],lane_ids=tokens,roadblock_ids=blocks,current_uncorrected_roadblocks=current['route_roadblock_ids'],shortcut_verified_full_planner=bool(args.limit));write(dest/'route.json',record);write(dest/'geometry.json',rec);records.append(record)
  if(n+1)%100==0 or args.limit:print('B20_CORRECT_ROUTE',n+1,len(indices),round(time.monotonic()-start,2),flush=True)
 ids=[str(s)for s in z['scene_ids']if str(s)in values];f=np.stack([values[s]for s in ids]);np.savez_compressed(out/'features.npz',scene_ids=np.array(ids),features=f,names=np.array(NAMES));write(out/'result.json',dict(complete=True,scenes=len(ids),seconds=time.monotonic()-start,source_sha=sha(__file__),feature_sha=sha(out/'features.npz')))
if __name__=='__main__':main()
