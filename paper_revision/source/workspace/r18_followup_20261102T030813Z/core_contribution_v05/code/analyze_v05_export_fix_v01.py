"""One complete v05 matrix analysis. Partial matrices reveal counts only.
Bridge official results are exported separately and never pooled in confirmation.
"""
from pathlib import Path
from collections import Counter
import json,csv,sys,hashlib,datetime,math
from v05_statistics import contrast
from v05_motion import motion,summary
EVENTS=['collisions_layout','collisions_pedestrian','collisions_vehicle','red_light','stop_infraction','outside_route_lanes','min_speed_infractions','yield_emergency_vehicle_infractions','scenario_timeouts','route_dev','vehicle_blocked','route_timeout']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jl(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]if Path(p).exists()else[]
def official(record):
 scores=(record or{}).get('scores',{});events=(record or{}).get('infractions',{});ds=float(scores.get('score_composed',0));rc=float(scores.get('score_route',0));assert math.isfinite(ds)and math.isfinite(rc)
 if record:assert abs(ds-max(0.,rc*float(scores['score_penalty'])))<1e-3
 status=(record or{}).get('status','MISSING_OFFICIAL_RECORD')
 return dict(DS=ds,RC=rc,status=status,official_record_present=record is not None,success=int(bool(record)and status in ['Completed','Perfect']and not any(v for k,v in events.items()if k!='min_speed_infractions')),events={e:len(events[e])if record and e in events else None for e in EVENTS},other_official_events={e:len(v)for e,v in events.items()if e not in EVENTS},event_details=events,official_route_completed_km=float(record['meta']['route_length']*rc/100000)if record else None)
def flatten(x,prefix=''):
 out={}
 for k,v in x.items():
  n=prefix+k
  if isinstance(v,dict):out.update(flatten(v,n+'__'))
  elif isinstance(v,list):out[n]=json.dumps(v,ensure_ascii=False,separators=(',',':'))
  elif v is None:out[n]='UNKNOWN'if 'event'in n or n in ['a_count','b_count','difference']else'MISSING'
  else:out[n]=v
 return out
def write_csv(p,rows):
 flat=[flatten(r)for r in rows];lead=['record_type','phase','method','route','seed'];cols=lead+sorted(set().union(*(r.keys()for r in flat))-set(lead))
 with Path(p).open('w',newline='',encoding='utf-8-sig')as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(flat)
 with Path(p).open(encoding='utf-8-sig',newline='')as f:assert len(list(csv.DictReader(f)))==len(rows)
def mechanics(plans,job,phase):
 attempts=[];c=Counter();prefix=[]
 for p in plans:
  c['replans']+=1;c['selected_appended_candidate']+=int(p['choice']>=12);receipt=p['base_receipt'];total=0
  for stage,maximum in [('map_supply',3),('map_timing',2)]:
   r=receipt.get(stage,{})
   c[stage+'_triggered']+=int(bool(r.get('trigger')));aa=r.get('attempts',[]);assert len(aa)<=maximum;total+=len(aa)
   for slot,x in enumerate(aa):
    c['attempts']+=1;c['admitted']+=int(bool(x.get('admitted')));c['eligible_after_stage']+=int(bool(x.get('eligible')))
    selected=bool(x.get('admitted')and x.get('index')==p['choice']);c['selected_supply']+=int(selected)
    reason=x.get('reject')or('duplicate'if x.get('duplicate')else'admitted'if x.get('admitted')else'UNSPECIFIED_NOT_ADMITTED');c['reason:'+reason]+=1
    attempts.append(dict(record_type='SUPPLY_ATTEMPT',phase=phase,**job,replan=p['replan'],timestamp_us=p['timestamp_us'],source_sha=p['source_sha'],stage=stage,supply_slot=slot,trigger=r.get('trigger'),source_type=r.get('topology',{}).get('source_type',receipt.get('map_supply',{}).get('topology',{}).get('source_type')),candidate_index=x.get('index'),selected_final=selected,rejection_or_admission=reason,detail={k:v for k,v in x.items()if k not in ['candidate','feature_receipt']},candidate_sha256=hashlib.sha256(json.dumps(x['candidate'],separators=(',',':')).encode()).hexdigest()if 'candidate'in x else None,feature_receipt=x.get('feature_receipt')))
  assert total<=5
 return dict(c,plan_seconds=summary([float(p['seconds'])for p in plans]),base_seconds=summary([float(p['base_seconds'])for p in plans])),attempts
def collect(v,phase):
 q=v/(phase+'_v01');f=read(v/'protocol'/f'{phase.upper()}_FREEZE_v01.json');fh=sha(v/'protocol'/f'{phase.upper()}_FREEZE_v01.json');out=[];attempts=[];supplies=[]
 for j in f['schedule']:
  job=read(q/'jobs'/f"{j['index']:03d}.json");fp=Path(job['final']);a=read(fp);assert job['job']==j==a['job']and fh==job['freeze_sha256']==a['freeze_sha256'];assert a['attempt']==1
  ad=fp.parent/'attempt1';ap=ad/'ATTEMPT.json';assert read(ap)==a;tr=ad/'trace';plans=jl(tr/'policy/plans.jsonl');trace=jl(tr/'trace.jsonl');mv=motion(trace,jl(tr/'actual_metric_10hz.jsonl'),'historical_internal');mech,ss=mechanics(plans,j,phase);supplies+=ss
  row=dict(record_type='EPISODE',phase=phase,**j,cohort=a['cohort'],historical_id=a['historical_id'],**official(a['official_record']),motion=mv,mechanics=mech,freeze_sha256=fh,wall_seconds=a['wall_seconds'],returncode=a['returncode'],wall_timeout=a['wall_timeout'],decision_started=a['decision_started'],interface_validation=a['interface_validation'],interface_errors=a['parse_or_interface_errors'],gate_passed=job['gate_passed'],source=str(fp),source_sha256=sha(fp),config_sha256=a['config_sha256'],run_step_seconds=summary([float(r['run_step_seconds'])for r in trace if r.get('run_step_seconds')is not None]))
  row['timeout_or_stall']=bool(a['wall_timeout']or any((row['events'].get(e)or 0)>0 for e in ['vehicle_blocked','route_timeout','scenario_timeouts'])or 'timeout'in row['status'].lower()or 'blocked'in row['status'].lower());out.append(row)
  attempts.append(dict(record_type='ATTEMPT',phase=phase,**j,attempt=1,decision_started=a['decision_started'],returncode=a['returncode'],wall_timeout=a['wall_timeout'],wall_seconds=a['wall_seconds'],official_record_present=a['official_record']is not None,interface_errors=a['parse_or_interface_errors'],source=str(ap),source_sha256=sha(ap),freeze_sha256=fh))
 return out,attempts,supplies

def arm_summary(rows,seeds):
 def average_route(name):
  vals=[];missing=[]
  for rr in sorted({r['route']for r in rows}):
   vs=[r[name]for r in rows if r['route']==rr]
   if len(vs)==len(seeds)and all(v is not None for v in vs):vals.append(sum(vs)/len(seeds))
   else:missing.append(rr)
  return dict(route_equal_mean=sum(vals)/len(vals)if vals else None,complete_routes=len(vals),missing_routes=missing)
 ev={}
 for e in EVENTS:
  known=[r for r in rows if r['events'][e]is not None];withkm=[r for r in known if r['motion']['actual_logged_ego_path_km']is not None];km=sum(r['motion']['actual_logged_ego_path_km']for r in withkm)
  ev[e]=dict(known_episodes=len(known),unknown_episodes=len(rows)-len(known),known_event_total=sum(r['events'][e]for r in known),all_scheduled_total=sum(r['events'][e]for r in known)if len(known)==len(rows)else None,distance_matched_episodes=len(withkm),actual_km=km,events_per_actual_km=sum(r['events'][e]for r in withkm)/km if km else None)
 motion_fields=['stopped_time_fraction_speed_lt_0_2','actual_logged_ego_path_km','observed_time_seconds']
 for key in ['moving_speed_mps','moving_lon_accel_abs','moving_lat_accel_abs','moving_lon_jerk_abs','moving_lat_jerk_abs']:
  motion_fields.append(key+'_p95')
 diag={}
 for key in motion_fields:
  temp=[dict(r,value=r['motion'].get(key)if not key.endswith('_p95')else r['motion'][key[:-4]]['p95'])for r in rows];old=rows;rows=temp;diag[key]=average_route('value');rows=old
 return dict(episodes=len(rows),DS=average_route('DS'),RC=average_route('RC'),success_integer=sum(r['success']for r in rows),success_denominator=len(rows),official_records=sum(r['official_record_present']for r in rows),status_counts=dict(Counter(r['status']for r in rows)),timeout_or_stall_episodes=sum(r['timeout_or_stall']for r in rows),all_attempt_wall_seconds=sum(r['wall_seconds']for r in rows),events=ev,motion=diag,mechanics=dict(sum((Counter({k:x for k,x in r['mechanics'].items()if isinstance(x,int)})for r in rows),Counter())))

def event_pairs(rows,routes,seeds,a,b):
 idx={(r['route'],r['seed'],r['method']):r for r in rows};out=[]
 for route in routes:
  for seed in seeds:
   for e in EVENTS:
    ra=idx[route['id'],seed,a];rb=idx[route['id'],seed,b];av=ra['events'][e];bv=rb['events'][e];delta=None if av is None or bv is None else av-bv
    out.append(dict(record_type='PAIRED_EVENT',phase='confirmation',route=route['id'],seed=seed,arm_a=a,arm_b=b,event=e,a_count=av,b_count=bv,difference=delta,newly_penalized=None if delta is None else av>0 and bv==0,no_longer_penalized=None if delta is None else av==0 and bv>0,a_details=ra['event_details'].get(e),b_details=rb['event_details'].get(e),interpretation='observed matched route-seed counts, not same-history counterfactual event identities'))
 return out

def main(v):
 q=v/'confirmation_v01';jobs=list((q/'jobs').glob('*.json'));fpath=v/'protocol/CONFIRMATION_FREEZE_v01.json';f=read(fpath)if fpath.exists()else None;final=read(q/'COORDINATOR_FINAL.json')if(q/'COORDINATOR_FINAL.json').exists()else None
 expected=f['episode_budget']if f else 72
 if not f or len(jobs)!=expected or not final or final['status']!='MATRIX_COMPLETE':return dict(status='WAIT_COMPLETE_MATRIX',expected=expected,completed=len(jobs),final=final,no_scores_read=True)
 out=v/'analysis';out.mkdir(exist_ok=True)
 if(out/'FINAL_v01.json').exists():return dict(status='ALREADY_ANALYZED',path=str(out/'FINAL_v01.json'),sha256=sha(out/'FINAL_v01.json'))
 for rel in ['code/analyze_v05.py','code/v05_motion.py','code/v05_statistics.py','code/audit_v05_inputs.py']:assert sha(v/rel)==f['file_sha256'][rel]
 audit=read(v/'runtime/CONFIRMATION_INPUT_AUDIT.json');assert audit['passed'];rows,attempts,supplies=collect(v,'confirmation');bridge,bridge_attempts,bridge_supplies=collect(v,'bridge');assert all(r['gate_passed']for r in rows)
 arms={m:arm_summary([r for r in rows if r['method']==m],f['seeds'])for m in f['methods']};comparisons={};events=[];export=[*rows,*attempts,*bridge,*bridge_attempts]
 for a,b,label in [('H-Map','H-Generic','PRIMARY'),('H-Map','H-Base','SECONDARY')]:
  c=contrast(rows,f['routes'],f['seeds'],a,b);comparisons[label]=c;export.append(dict(record_type=label+'_COMPARISON',phase='confirmation',**c));export.extend(dict(record_type=label+'_ROUTE_EFFECT',phase='confirmation',direction=c['direction'],**r)for r in c['route_effects']);events+=event_pairs(rows,f['routes'],f['seeds'],a,b)
 export.extend(dict(record_type='ARM_SUMMARY',phase='confirmation',method=m,**x)for m,x in arms.items());export+=events
 export.extend(dict(record_type='INPUT_PAIR_AUDIT',phase='confirmation',**x)for x in audit['pairs']);write_csv(out/'RESULTS_v01.csv',export);write_csv(out/'SUPPLY_ATTEMPTS_v01.csv',supplies+bridge_supplies)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='COMPLETE_FROZEN_MATRIX_ANALYZED',scheduled=expected,completed=len(rows),bridge_episodes=len(bridge),freeze_sha256=sha(fpath),arms=arms,comparisons=comparisons,rows=rows,bridge_rows=bridge,attempt_rows=attempts+bridge_attempts,input_audit=audit,analysis_source_sha256=sha(__file__),result_csv_sha256=sha(out/'RESULTS_v01.csv'),supply_table_sha256=sha(out/'SUPPLY_ATTEMPTS_v01.csv'),supply_attempt_rows=len(supplies),bridge_supply_attempt_rows=len(bridge_supplies),no_old_cohort_pooling=True,old_Supply_v2_negative_results_preserved=True,no_CW_vs_Simple_inference=True,no_witness_ablation_claim=True)
 (out/'FINAL_v01.json').write_text(json.dumps(result,indent=2)+'\n');return dict(status=result['status'],completed=len(rows),comparisons=comparisons,result_csv=str(out/'RESULTS_v01.csv'))

def self_test():
 assert official(None)['DS']==0 and all(x is None for x in official(None)['events'].values())
 record=dict(scores=dict(score_composed=100,score_route=100,score_penalty=1),status='Completed',infractions={e:[]for e in EVENTS},meta=dict(route_length=100));assert official(record)['success']==1
 record['infractions']['collisions_vehicle']=['test'];assert official(record)['success']==0
 import tempfile
 with tempfile.TemporaryDirectory()as d:assert main(Path(d))['no_scores_read']
 return dict(passed=True,checks=['unknown missing events','official DS arithmetic','success excludes collision','incomplete matrix does not read outcomes'])
if __name__=='__main__':print(json.dumps(self_test()if '--self-test'in sys.argv else main(Path(sys.argv[1]))))
