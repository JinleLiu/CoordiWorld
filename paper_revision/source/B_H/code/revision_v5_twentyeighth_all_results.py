"""Actual common cohorts and fixed confirmation prefixes; all partial/failed runs retained."""
import argparse,csv,datetime,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from functools import lru_cache
from revision_v5_twentyeighth_results import main,summary,paired,csvout,CORE,TRANSFER,SUPPLY,CONFIRM,EVENTS
@lru_cache(None)
def control_trace(ad):
 traces=list(Path(ad).rglob('trace.jsonl'));times=[];replans=[];xy=[];ts=[];count=0;previous_replans=0
 if len(traces)!=1:return times,replans,None,0,False
 for line in open(traces[0]):
  x=json.loads(line);count+=1
  if x.get('run_step_seconds')is not None:times.append(x['run_step_seconds'])
  elif x.get('basic_agent_step_seconds')is not None:times.append(x['basic_agent_step_seconds'])
  if(x.get('replan_seconds')or 0)>0:replans.append(x['replan_seconds'])
  elif x.get('replans',0)!=previous_replans and(x.get('plan_seconds')or 0)>0:replans.append(x['plan_seconds'])
  previous_replans=x.get('replans',0)
  pos=x.get('body_pose',x.get('pose'))
  if pos is None and isinstance(x.get('ego'),dict):pos=x['ego'].get('location')
  if pos is None and isinstance(x.get('ego_world'),dict):pos=[x['ego_world'].get('x'),x['ego_world'].get('y')]
  if pos is not None and x.get('sim_seconds')is not None:xy.append(pos[:2]);ts.append(x['sim_seconds'])
 good=len(ts)==count and count>1 and np.isfinite(xy).all()and np.allclose(np.diff(ts),.05,atol=1e-6,rtol=0)
 km=float(np.linalg.norm(np.diff(np.asarray(xy,float),axis=0),axis=1).sum()/1000)if good else None
 return times,replans,km,count,bool(good)
def run(B,O):
 O.mkdir(parents=True,exist_ok=True);roots={'main':B};roots.update({p.parent.name:p.parent for p in sorted(B.glob('*_run/bench2drive'))});records={};allrows=[];infra=[];attempt_rows=[]
 for name,root in roots.items():
  if not list((root/'bench2drive/episodes').glob('seed*/*/*/FINAL.json')):continue
  out=O/'by_run'/name;main(root,out);p=out/'per_route_seed_method.csv';rows=list(csv.DictReader(open(p)))if p.exists()else[]
  for ap in (root/'bench2drive').rglob('ATTEMPT.json'):
   ar=json.load(open(ap));decision=ar.get('decision_started',ar.get('model_control_steps',ar.get('control_steps_logged',0))>0);attempt_rows.append(dict(run=name,method=ar.get('method'),route=ar.get('route',{}).get('id'),seed=ar.get('seed'),attempt=ar.get('attempt'),started_utc=ar.get('started_utc'),ended_utc=ar.get('ended_utc'),decision_started=decision,returncode=ar.get('returncode'),terminal=ar.get('terminal'),official_record=bool(ar.get('official_record')),wall_timeout=ar.get('wall_timeout'),config_sha=ar.get('config_sha'),wall_seconds=ar.get('wall_seconds'),agent_source_sha=ar.get('agent_source_sha'),queue_source_sha=ar.get('queue_source_sha'),source_zip='raw/'+str(ap.relative_to(B))))
  for r in rows:
   for k in ['seed','success','completed100','control_steps','model_control_steps']:
    if r.get(k)not in [None,'']:r[k]=int(r[k])
   for k in ['DS','RC','wall_seconds','duration_game','duration_system','driving_efficiency_percent']+EVENTS:r[k]=float(r[k])if r.get(k)not in ['',None]else None
   r['has_official_record']=r['has_official_record']=='True';r['wall_timeout']=r['wall_timeout']=='True';r['run']=name;r['source_zip']='raw/'+str(Path(r['source']).relative_to(B));allrows.append(r)
  records[name]=rows;infra.extend(dict(run=name,**r)for r in json.load(open(out/'precontrol_failures.json')))
 studies={}
 def add(name,rows,methods,scope,ordered=None):
  by={m:[r for r in rows if r['method']==m]for m in methods};sets=[{(r['route'],r['seed'])for r in by[m]}for m in methods];common=set.intersection(*sets)if sets else set();prefix=None
  if ordered is not None:
   prefix=[]
   for route in ordered:
    if(route,17)not in common:break
    prefix.append(route)
   keys={(r,17)for r in prefix}
  else:keys=common
  selected={m:[r for r in by[m]if(r['route'],r['seed'])in keys]for m in methods};s=dict(scope=scope,methods=methods,common_keys=sorted(keys),complete_prefix=prefix,all_complete_common_keys=sorted(common),available={m:summary(by[m])for m in methods},primary={m:summary(selected[m])for m in methods},paired={})
  for m in methods[1:]:
   ss,rr=paired(selected[m],selected[methods[0]]);s['paired'][m+'--'+methods[0]]=ss;csvout(O/'studies'/f'{name}__{m}--{methods[0]}.csv',rr)
  extras=[]
  if name=='native5':extras=[(CORE[-1],CORE[-2]),(CORE[-1],'TFpp_single'),('TFpp_single','World4Scorer_route_tp')]
  if name=='W4_transfer4':extras=[(TRANSFER[2],TRANSFER[1]),(TRANSFER[2],TRANSFER[3])]
  if name in ['new_type_confirmation','fresh_parking_confirmation']:extras=[(SUPPLY[1],'World4Scorer_route_tp'),(SUPPLY[1],'TFpp_single')]
  for a,b in extras:
   ss,rr=paired(selected[a],selected[b]);s['paired'][a+'--'+b]=ss;csvout(O/'studies'/f'{name}__{a}--{b}.csv',rr)
  studies[name]=s;return selected
 (O/'studies').mkdir(exist_ok=True)
 chosen={};chosen['native5']=add('native5',records.get('main',[]),CORE,'Dev10 exposed; three traffic seeds; native-system comparison with different input permissions')
 chosen['W4_transfer4']=add('W4_transfer4',records.get('main',[]),TRANSFER,'Same10 developed routes×3seeds; frozen public model and native controller; own histories, whole-pool rescoring')
 chosen['Map04_DEV3']=add('Map04_DEV3',records.get('mapsupply_pilot_run',[])+records.get('mapsupply_blocks_run',[]),['CoordiWorld_BaseEC_v05','CoordiWorld_BaseEC_MapSupply_v04'],'Three developed routes,seed17; common v05 runway and low-speed adapter')
 rr=sum([records.get(k,[])for k in ['continuation_pilot_run','tight_pilot_run','replication23_run','replication42_run','support_blocked_run']],[])
 rr=[r for r in rr if r['method']in SUPPLY];assert len({(r['method'],r['route'],r['seed'])for r in rr})==len(rr)
 chosen['Tight_DEV']=add('Tight_DEV',rr,SUPPLY,'Developed3514×3seeds and27494×seed17; unequal seeds averaged within route; common v06 support and witness')
 rr=records.get('tight_pilot_run',[])+records.get('tight_ablation_run',[])
 chosen['Tight_witness_ablation']=add('Tight_witness_ablation',rr,['CoordiWorld_MapTight_NoWitness_v01',SUPPLY[1]],'One developed route/seed; identical tight supply, witness off versus on; no route-cluster interval')
 for name,rootname in [('new_type_confirmation','confirmation_run'),('fresh_parking_confirmation','parking_confirmation_run')]:
  q=B/rootname/'bench2drive';order=[r['id']for r in json.load(open(q/'PROTOCOL.json'))['routes']]if(q/'PROTOCOL.json').exists()else[]
  scope='Frozen original ordered prefix; B-development-unexposed routes, public pretraining exposure unknown; new scenario types'if name=='new_type_confirmation'else'Frozen original ordered prefix; fresh routes of already-developed ParkingExit family, not new-type generalization'
  chosen[name]=add(name,records.get(rootname,[]),CONFIRM,scope,order)
 if 'runtime_replay_run'in records:
  chosen['runtime_replay_23695']=add('runtime_replay_23695',records['runtime_replay_run'],CONFIRM,'One already-exposed route23695/seed17; four frozen methods replayed serially after the main queue to diagnose a prior CARLA RPC hang. Not a replacement, new confirmation, or best-of result.')
 effect=studies['new_type_confirmation']['paired'][SUPPLY[1]+'--'+SUPPLY[0]];dd=effect['per_route_delta'];nn=len(dd)
 attribution=dict(scope='Posthoc contribution bookkeeping only. Primary population, scores and events remain unchanged. In14909 control first differs at7.55s and original candidate/features already differ, before the only new-candidate selection at16.05s. Its40point paired difference cannot be isolated as a direct supply effect from these episodes. Withholding its contribution is not a counterfactual outcome or an adjusted causal estimate.',route_count=nn,primary_delta_DS=effect['delta_DS_route_equal'],route14909_delta_DS=dd.get('14909'),route14909_contribution_fixed_original_denominator=dd.get('14909',0)/nn if nn else None,other_routes_contribution_fixed_original_denominator=sum(v for k,v in dd.items()if k!='14909')/nn if nn else None)
 (O/'ATTRIBUTION_SENSITIVITY.json').write_text(json.dumps(attribution,indent=2)+'\n')
 wide=[]
 def wide_rows(cohort,by,scope,primary):
  for m,rr in by.items():
   if not rr:continue
   ss=summary(rr);km=0.;nmissing=0;attempts=0;all_attempt_wall=0.;attempt_wall_missing=0;versions=set();sources=[];times=[];replans=[];observed_metric=0;actual_km=0.;actual_n=0
   for r in rr:
    p=Path(r['source']);f=json.load(open(p));record=f.get('official_record');ad=p.parent/f"attempt{f['attempt']}"/'trace';cfg=f.get('method_registration',{}).get('config')
    if cfg is None:cfg=json.load(open(ad.parent/'agent.json'))
    versions.add(cfg.get('bridge_version','official_native_or_registered_transfer'));sources.append(r['source_zip']);attempts+=len(list(p.parent.glob('attempt*/ATTEMPT.json')));observed_metric+=int((ad/'actual_metric_10hz.jsonl').exists())
    if record and record.get('meta',{}).get('route_length')is not None:km+=record['meta']['route_length']*record['scores']['score_route']/100000.
    else:nmissing+=1
    for ap in p.parent.glob('attempt*/ATTEMPT.json'):
     ar=json.load(open(ap));av=ar.get('wall_seconds');all_attempt_wall+=float(av)if av is not None else 0.;attempt_wall_missing+=int(av is None)
    tt,pp,ak,ac,ag=control_trace(str(ad));times.extend(tt);replans.extend(pp)
    if ag and(r.get('control_steps')in [None,'']or ac==r['control_steps']):actual_km+=ak;actual_n+=1
   reference=next(iter(by));pa=paired(rr,by[reference])[0]if primary and m!=reference else{};ci=pa.get('route_cluster_bootstrap95')or[None,None];coll=sum(ss[e]for e in ['collisions_vehicle','collisions_layout','collisions_pedestrian']);n=ss['route_seed_count']
   wide.append(dict(method=m,cohort=cohort,version='|'.join(sorted(versions)),primary_common_comparison=primary,exposure=scope,attempts=attempts,algorithm_episodes=n,official_records=ss['official_records'],policy_missing_records=ss['policy_missing_record'],success=ss['success'],failure=ss['failure'],routes=ss['route_count'],towns=ss['town_count'],seeds='|'.join(map(str,ss['seed_values'])),DS_0_100_route_equal=ss['DS_route_equal'],RC_percent_episode_mean=ss['RC_episode_mean'],reference=reference if primary else None,delta_DS_route_equal=pa.get('delta_DS_route_equal'),route_bootstrap95_low=ci[0],route_bootstrap95_high=ci[1],official_completed_route_km=km,distance_unknown_records=nmissing,actual_20Hz_trace_episodes=actual_n,actual_20Hz_body_path_km=actual_km if actual_n else None,collision_penalties_per_actual_20Hz_path_km=coll/actual_km if actual_km>0 and actual_n==n and not ss['missing_event_records']else None,collision_penalties=coll if not ss['missing_event_records']else None,known_collision_penalties=coll,collision_penalties_per_completed_route_km=coll/km if km>0 and not nmissing else None,**{e:ss[e]if not ss['missing_event_records']else None for e in EVENTS},**{'known_'+e:ss[e]for e in EVENTS},event_unknown_records=ss['missing_event_records'],event_count_scope='Totals are NA if any official event record is missing; known_ columns are observed subtotals only',scored_attempt_wall_seconds_total=ss['wall_seconds'],all_saved_attempt_wall_seconds_total=all_attempt_wall,attempt_wall_missing=attempt_wall_missing,scored_attempt_wall_seconds_episode_mean=ss['wall_seconds']/n,actual_10Hz_metric_episodes=observed_metric,policy_callback_p95_seconds=float(np.quantile(times,.95))if times else None,replanning_p50_seconds=float(np.quantile(replans,.5))if replans else None,replanning_p95_seconds=float(np.quantile(replans,.95))if replans else None,replan_samples=len(replans),config_shas='|'.join(sorted({r['config_sha']for r in rr})),source_index='all_episodes.csv',source_paths='|'.join(sources)))
 for name,by in chosen.items():wide_rows(name,by,studies[name]['scope'],True)
 # The raw per-run view preserves incomplete populations, all failed branches,
 # and every source/configuration; it is explicitly not a common-cohort table.
 for name,rr in records.items():
  if name=='main':continue
  by={m:[r for r in rr if r['method']==m]for m in sorted({r['method']for r in rr})};wide_rows(name+'_available',by,'Per-run available episodes; see protocol and registration. May be incomplete; not used for the primary common comparison.',False)
 csvout(O/'all_episodes.csv',allrows);csvout(O/'all_attempts.csv',attempt_rows);csvout(O/'all_precontrol_failures.csv',infra);csvout(O/'method_cohort_summary.csv',[{k:('NA'if v is None else v)for k,v in r.items()}for r in wide]);(O/'studies.json').write_text(json.dumps(studies,indent=2)+'\n');(O/'ANALYSIS_SCOPE.txt').write_text('Primary comparisons use actual common populations. Confirmation uses the largest fully evaluated original ordered prefix per cohort. Later/partial blocks and every completed failed branch remain in available rows and all_episodes. First average seeds within route, then routes; paired intervals resample route clusters10000 times, omitted for one route. Actual20Hz distance sensitivity uses measured body-center horizontal paths with continuous.05s timestamps and matching logged-control count; missing coverage is not imputed. Collision/km is official counted penalties divided by completed route length (length×RC), not total physical contacts or a universal safety rate. Wide-table event totals are NA when any record is missing; known_ columns preserve observed subtotals, matching the explicitly known-count report table. Event records missing after control are unknown, not zero; policy score remains preregistered0, including post-control simulator hangs. Infrastructure failures before control are separate. all_attempts also retains archived prelaunch attempts. Scored-attempt wall totals exclude prior failed startup attempts; all_saved_attempt_wall includes ATTEMPT records within corresponding episode directories. Separate archived prelaunch attempts remain in all_attempts. Timing is from a shared host with up to3 simultaneous simulator slots, not isolated real-time deployment; the separately registered runtime_replay_23695 is serial and must not be pooled into that timing comparison.\n');print(json.dumps({k:dict(primary_pairs=len(v['common_keys']),prefix=v['complete_prefix'])for k,v in studies.items()},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw,a.out)
