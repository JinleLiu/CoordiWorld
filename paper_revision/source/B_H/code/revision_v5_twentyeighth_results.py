"""R18 actual-denominator summaries; no inherited closed-loop numbers."""
import argparse,csv,datetime,hashlib,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
CORE=['BasicAgent25','World4Scorer_route_tp','TFpp_single','CoordiWorld_BaseEC','CoordiWorld_Full']
TRANSFER=['World4Scorer_route_tp','World4Scorer_EC','World4Scorer_EC_H3','World4Scorer_EC_Gspeed']
NATIVE_EXEC=['World4Scorer_NativeEC','World4Scorer_NativeEC_H3','World4Scorer_NativeEC_Gspeed']
SPATIAL=['CoordiWorld_BaseEC_v05','CoordiWorld_BaseEC_MapSupply_v04']
VIABILITY=['CoordiWorld_BaseEC_Guard_v01','CoordiWorld_MapSupply_Guard_v01']
SUPPLY=['CoordiWorld_BaseSupportWitness_v01','CoordiWorld_MapTightWitness_v02']
CONFIRM=SUPPLY+['World4Scorer_route_tp','TFpp_single']
EVENTS=['collisions_layout','collisions_pedestrian','collisions_vehicle','red_light','stop_infraction','outside_route_lanes','min_speed_infractions','yield_emergency_vehicle_infractions','scenario_timeouts','route_dev','vehicle_blocked','route_timeout']
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvout(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w',newline='')as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def efficiency(record):
 vals=[];excluded=0
 for text in (record or{}).get('infractions',{}).get('min_speed_infractions',[]):
  m=re.search(r'\b\d+\.?\d*%',text)
  if m is None:excluded+=1;continue
  v=float(m.group().rstrip('%'))
  if v>1000:excluded+=1;continue
  vals.append(v)
 return dict(driving_efficiency_percent=float(np.mean(vals))if vals else None,efficiency_segments=len(vals),efficiency_excluded_segments=excluded)
def summary(rows):
 by=defaultdict(list)
 for r in rows:by[r['route']].append(r)
 n=len(rows)
 return dict(status_counts={k:sum(r['status']==k for r in rows)for k in sorted({r['status']for r in rows})},wall_cap_timeouts=sum(bool(r['wall_timeout'])for r in rows),route_seed_count=n,route_count=len(by),town_count=len({r['town']for r in rows}),town_values=sorted({r['town']for r in rows}),seed_values=sorted({r['seed']for r in rows}),DS_route_equal=np.mean([np.mean([r['DS']for r in rr])for rr in by.values()]).item()if n else None,DS_episode_mean=float(np.mean([r['DS']for r in rows]))if n else None,RC_episode_mean=float(np.mean([r['RC']for r in rows]))if n else None,success=sum(r['success']for r in rows),failure=sum(not r['success']for r in rows),route_completion100=sum(r['completed100']for r in rows),official_records=sum(r['has_official_record']for r in rows),policy_missing_record=sum(not r['has_official_record']for r in rows),wall_seconds=sum(r['wall_seconds']for r in rows),driving_efficiency_episode_mean=float(np.mean([r['driving_efficiency_percent']for r in rows if r['driving_efficiency_percent']is not None]))if any(r['driving_efficiency_percent']is not None for r in rows)else None,efficiency_record_count=sum(r['driving_efficiency_percent']is not None for r in rows),missing_event_records=sum(not r['has_official_record']for r in rows),**{e:sum(r[e]for r in rows if r[e]is not None)for e in EVENTS})
def paired(a,b):
 aa={(r['route'],r['seed']):r for r in a};bb={(r['route'],r['seed']):r for r in b};keys=sorted(aa.keys()&bb.keys());rows=[];by=defaultdict(list)
 for k in keys:
  x,y=aa[k],bb[k];r=dict(route=k[0],seed=k[1],DS_candidate=x['DS'],DS_reference=y['DS'],delta_DS=x['DS']-y['DS'],delta_RC=x['RC']-y['RC'],success_candidate=x['success'],success_reference=y['success'])
  for e in EVENTS:r['new_'+e]=max(0,x[e]-y[e])if x[e]is not None and y[e]is not None else None;r['reduced_'+e]=max(0,y[e]-x[e])if x[e]is not None and y[e]is not None else None
  rows.append(r);by[k[0]].append(r['delta_DS'])
 d=np.array([np.mean(v)for v in by.values()]);ci=None
 if len(d)>1:
  rng=np.random.default_rng(17);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
 return dict(n_route_seed=len(keys),n_routes=len(by),delta_DS_route_equal=float(d.mean())if len(d)else None,route_cluster_bootstrap95=ci,bootstrap='10000 draws of route clusters; each route first averages its paired seeds; seed17 RNG; CI omitted if<2routes',per_route_delta={k:float(np.mean(v))for k,v in by.items()},positive_routes=int((d>1e-9).sum()),negative_routes=int((d< -1e-9).sum()),zero_routes=int((abs(d)<=1e-9).sum()),leave_one_route_out={k:float(np.mean([np.mean(v)for kk,v in by.items()if kk!=k]))for k in by}if len(by)>1 else{},event_count_differences={e:dict(increased=sum(r['new_'+e]for r in rows if r['new_'+e]is not None),reduced=sum(r['reduced_'+e]for r in rows if r['reduced_'+e]is not None),unknown_pairs=sum(r['new_'+e]is None for r in rows),increased_route_seeds=[f"{r['route']}:{r['seed']}"for r in rows if r['new_'+e]],reduced_route_seeds=[f"{r['route']}:{r['seed']}"for r in rows if r['reduced_'+e]])for e in EVENTS}),rows

def main(base,out):
 out.mkdir(parents=True,exist_ok=True);rows=[];startup=[];route_meta=json.load(open(base/'bench2drive/route_metadata.json')) if (base/'bench2drive/route_metadata.json').exists() else {}
 for p in sorted((base/'bench2drive/episodes').glob('seed*/*/*/FINAL.json')):
  x=json.load(open(p));r=x.get('official_record');decision=x.get('decision_started',x.get('model_control_steps',x.get('control_steps_logged',0))>0)
  if not decision:startup.append(dict(source=str(p),method=x['method'],route=x['route']['id'],seed=x['seed'],returncode=x.get('returncode'),outcome=x.get('outcome')));continue
  sc=r.get('scores',{})if r else{};ev=r.get('infractions',{})if r else{};status=r.get('status','MISSING_RECORD')if r else'MISSING_RECORD';ds=float(sc.get('score_composed',x.get('score_if_missing',0)));rc=float(sc.get('score_route',0));assert np.isfinite([ds,rc]).all()
  row=dict(method=x['method'],route=x['route']['id'],seed=x['seed'],town=x['route'].get('town'),scenario=r.get('scenario_name')if r else None,weather=r.get('weather_id')if r else None,DS=ds,RC=rc,penalty=float(sc.get('score_penalty',0)),success=int(status in ['Completed','Perfect']and not any(v for k,v in ev.items()if k!='min_speed_infractions')),completed100=int(rc>=99.9999),status=status,has_official_record=bool(r),wall_seconds=x['wall_seconds'],duration_game=r.get('meta',{}).get('duration_game')if r else None,duration_system=r.get('meta',{}).get('duration_system')if r else None,control_steps=x.get('control_steps_logged'),model_control_steps=x.get('model_control_steps'),wall_timeout=x.get('wall_timeout'),config_sha=x['config_sha'],agent_sha=x.get('agent_source_sha'),source=str(p),source_sha256=sha(p),**efficiency(r),**{e:len(ev.get(e,[]))if r else None for e in EVENTS});rows.append(row)
 
 for row in rows:
  rm=route_meta.get(row['route'],{});row['scenario']=row['scenario'] or rm.get('scenario_types');row['weather_sha256']=rm.get('weather_sha256');row['route_metadata_source']=rm.get('source')
 csvout(out/'per_route_seed_method.csv',rows);write(out/'precontrol_failures.json',startup)
 by={m:[r for r in rows if r['method']==m]for m in sorted({r['method']for r in rows})};s=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),round='R18',data=json.load(open(base/'bench2drive/PROTOCOL.json')).get('cohort','Dev10_exposed')+'; real CARLA0.9.15; actual route-seed denominators',official_success_definition='Completed/Perfect and zero infractions excluding min_speed_infractions, matching pinned merge_route_json.py',missing_record='post-decision failure gets preregistered0; precontrol failures separate',available_unmatched={m:summary(v)for m,v in by.items()},common={})
 for name,methods in [('native5',CORE),('public_native3',CORE[:3]),('CW2',CORE[-2:]),('W4_transfer4',TRANSFER),('all8',list(dict.fromkeys(CORE+TRANSFER))),('native_execution3',NATIVE_EXEC),('spatial_supply2',SPATIAL),('shared_viability2',VIABILITY),('matched_tight_supply2',SUPPLY),('confirmation4',CONFIRM)]:
  sets=[{(r['route'],r['seed'])for r in by.get(m,[])}for m in methods];keys=set.intersection(*sets);s['common'][name]=dict(methods=methods,route_seeds=[list(k)for k in sorted(keys)],results={m:summary([r for r in by.get(m,[])if(r['route'],r['seed'])in keys])for m in methods})
 contrasts=[('CoordiWorld_Full','CoordiWorld_BaseEC'),('CoordiWorld_Full','BasicAgent25'),('CoordiWorld_Full','World4Scorer_route_tp'),('CoordiWorld_Full','TFpp_single'),('World4Scorer_route_tp','BasicAgent25'),('TFpp_single','BasicAgent25'),('TFpp_single','World4Scorer_route_tp'),('World4Scorer_EC','World4Scorer_route_tp'),('World4Scorer_EC_H3','World4Scorer_EC'),('World4Scorer_EC_H3','World4Scorer_EC_Gspeed')];s['paired']={}
 contrasts+= [('World4Scorer_NativeEC_H3','World4Scorer_NativeEC'),('World4Scorer_NativeEC_H3','World4Scorer_NativeEC_Gspeed')]
 contrasts+= [(SPATIAL[1],SPATIAL[0]),(VIABILITY[1],VIABILITY[0])]
 contrasts+= [(SUPPLY[1],SUPPLY[0]),(SUPPLY[1],'World4Scorer_route_tp'),(SUPPLY[1],'TFpp_single'),('CoordiWorld_Full_Runway_v05','CoordiWorld_BaseEC_Runway_v05')]
 for a,b in contrasts:
  ss,rr=paired(by.get(a,[]),by.get(b,[]));s['paired'][a+'--'+b]=ss;csvout(out/(a+'--'+b+'.csv'),rr)
 keys={tuple(k)for k in s['common']['W4_transfer4']['route_seeds']};s['paired_common_W4_transfer4']={}
 for a,b in contrasts:
  if a not in TRANSFER or b not in TRANSFER:continue
  aa=[r for r in by.get(a,[])if(r['route'],r['seed'])in keys];bb=[r for r in by.get(b,[])if(r['route'],r['seed'])in keys];ss,rr=paired(aa,bb);s['paired_common_W4_transfer4'][a+'--'+b]=ss;csvout(out/('common_W4_transfer4__'+a+'--'+b+'.csv'),rr)
 s['common_by_seed']={}
 for name,group in s['common'].items():
  s['common_by_seed'][name]={}
  for seed in sorted({r['seed']for r in rows}):
   keys={tuple(k)for k in group['route_seeds']if k[1]==seed};s['common_by_seed'][name][str(seed)]={m:summary([r for r in by.get(m,[])if(r['route'],r['seed'])in keys])for m in group['methods']}
 write(out/'summary.json',s);print(json.dumps(dict(common={k:len(v['route_seeds'])for k,v in s['common'].items()},available={m:dict(n=v['route_seed_count'],DS=v['DS_route_equal'],success=v['success'])for m,v in s['available_unmatched'].items()}),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.base,a.out)
